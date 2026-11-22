"""Рендер полигона в Blender (Cycles, Metal). Запуск:
  Blender -b --factory-startup --python sim/blender_render.py -- spec.json
spec.json: камера (интринсики + позы), поверхности (прямоугольники пола), объекты, свет, выход.
На каждый кадр пишет color.png, depth.exr (Z, м), index.exr (номер объекта)."""
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

spec = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text(encoding="utf-8"))
ROOT = Path(spec["root"])
ASSETS = ROOT / "sim" / "assets"
OUT = Path(spec["out"])
OUT.mkdir(parents=True, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
prefs = bpy.context.preferences.addons["cycles"].preferences
import os
for dev_type in (os.environ.get("CYCLES_DEVICE"), "OPTIX", "CUDA", "METAL"):
    if not dev_type:
        continue
    try:
        prefs.compute_device_type = dev_type
        prefs.get_devices()
        if any(d.type == dev_type for d in prefs.devices):
            break
    except TypeError:
        continue
for d in prefs.devices:
    d.use = d.type == prefs.compute_device_type
scene.cycles.device = "GPU"
scene.cycles.samples = spec.get("samples", 48)
scene.cycles.use_denoising = True
scene.cycles.max_bounces = 6
cam_spec = spec["camera"]
scene.render.resolution_x = cam_spec["width"]
scene.render.resolution_y = cam_spec["height"]
scene.render.resolution_percentage = 100
scene.view_settings.view_transform = "Standard"
scene.view_settings.exposure = spec.get("exposure", 0.0)
scene.view_layers[0].use_pass_z = True
scene.view_layers[0].use_pass_object_index = True

world = bpy.data.worlds.new("w")
scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.35, 0.35, 0.36, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = spec.get("ambient", 0.3)


def image_material(name, tex_path, size_m, roughness, bump=0.0, rot=0.0, offset=(0.0, 0.0)):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Specular IOR Level"].default_value = 0.5
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1 / size_m, 1 / size_m, 1)
    mp.inputs["Rotation"].default_value = (0, 0, -rot)
    mp.inputs["Location"].default_value = (-offset[0] / size_m, -offset[1] / size_m, 0)
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(tex_path))
    tex.extension = "REPEAT"
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    if bump > 0:
        b = nt.nodes.new("ShaderNodeBump")
        b.inputs["Strength"].default_value = bump
        nt.links.new(tex.outputs["Color"], b.inputs["Height"])
        nt.links.new(b.outputs["Normal"], bsdf.inputs["Normal"])
    return m


SURF_IDS = {"tile": 0, "laminate": 1, "carpet": 2, "shag": 3}
surf_cfg = json.loads((ASSETS / "surfaces.json").read_text())["surfaces"]
ROUGH = {"tile": 0.38, "laminate": 0.35, "carpet": 0.95, "shag": 1.0}
BUMP = {"tile": 0.0, "laminate": 0.1, "carpet": 0.6, "shag": 0.9}
for i, s in enumerate(spec["surfaces"]):
    cfg = surf_cfg[s["type"]]
    x0, y0, x1, y1 = s["rect"]
    z = s.get("z", 0.0) + i * 1e-4
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, z))
    ob = bpy.context.active_object
    ob.name = f"surface_{i}_{s['type']}"
    me = ob.data
    for v in me.vertices:
        v.co.x = x0 if v.co.x < 0 else x1
        v.co.y = y0 if v.co.y < 0 else y1
    ob.data.materials.append(image_material(ob.name, ASSETS / cfg["texture"], cfg["size_m"], ROUGH[s["type"]],
                                            BUMP[s["type"]], s.get("rot", 0.0), tuple(s.get("offset", (0, 0)))))
    ob.pass_index = 100 + SURF_IDS[s["type"]]
    if s["type"] in ("carpet", "shag"):
        ob.scale = (1, 1, 1)
        mod = ob.modifiers.new("solid", "SOLIDIFY")
        mod.thickness = s.get("thickness", 0.012)
        mod.offset = 1

MAT_PROPS = {"metal": (0.85, 0.3, 0.0), "glass": (0.0, 0.05, 0.7), "plastic": (0.0, 0.35, 0.0),
             "paper": (0.0, 0.8, 0.0), "wood": (0.0, 0.6, 0.0), "mineral": (0.0, 0.9, 0.0)}


def object_material(name, material):
    metallic, rough, trans = MAT_PROPS.get(material, (0, 0.5, 0))
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    attr = nt.nodes.new("ShaderNodeVertexColor")
    attr.layer_name = "col"
    nt.links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Transmission Weight"].default_value = trans
    return m


def load_mesh(oid, path, material):
    d = np.load(path)
    v, f, c = d["verts"], d["faces"], d["colors"]
    me = bpy.data.meshes.new(oid)
    me.vertices.add(len(v))
    me.vertices.foreach_set("co", v.reshape(-1))
    me.loops.add(f.size)
    me.loops.foreach_set("vertex_index", f.reshape(-1))
    me.polygons.add(len(f))
    me.polygons.foreach_set("loop_start", np.arange(0, f.size, 3))
    me.polygons.foreach_set("loop_total", np.full(len(f), 3))
    me.update()
    me.validate()
    ca = me.color_attributes.new("col", "FLOAT_COLOR", "POINT")
    rgba = np.concatenate([c, np.ones((len(c), 1), np.float32)], 1)
    ca.data.foreach_set("color", rgba.reshape(-1))
    me.materials.append(object_material(oid, material))
    for p in me.polygons:
        p.use_smooth = True
    return me


def primitive_mesh(name, o):
    kind = o["primitive"]
    if kind == "box":
        bpy.ops.mesh.primitive_cube_add(size=1)
    elif kind == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.5, depth=1)
    else:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=0.5)
    tmp = bpy.context.active_object
    dx, dy, dz = o["dims"]
    for v in tmp.data.vertices:
        v.co.x *= dx
        v.co.y *= dy
        v.co.z = v.co.z * dz + dz / 2
    me = tmp.data
    me.name = name
    bpy.data.objects.remove(tmp)
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*o["color"], 1)
    metallic, rough, trans = MAT_PROPS.get(o.get("material", "plastic"), (0, 0.5, 0))
    b.inputs["Metallic"].default_value = metallic
    b.inputs["Roughness"].default_value = rough
    b.inputs["Transmission Weight"].default_value = trans
    me.materials.append(m)
    return me


for k, o in enumerate(spec.get("objects", []), start=1):
    if "primitive" in o:
        me = primitive_mesh(f"obj{k}_{o['id']}", o)
    else:
        path = Path(o["mesh"]) if "mesh" in o else ASSETS / "objects" / f"{o['id']}.npz"
        me = load_mesh(f"obj{k}_{o['id']}", path, o.get("material", "plastic"))
    ob = bpy.data.objects.new(me.name, me)
    scene.collection.objects.link(ob)
    ob.location = (o["x"], o["y"], o.get("z", 0.0))
    ob.rotation_euler = (0, 0, o.get("yaw", 0.0))
    s = o.get("scale", 1.0)
    ob.scale = (s, s, s)
    ob.pass_index = k

for b in spec.get("props", []):
    bpy.ops.mesh.primitive_cube_add(size=1, location=(b["x"], b["y"], b["h"] / 2))
    ob = bpy.context.active_object
    ob.scale = (b["w"], b["d"], b["h"])
    ob.rotation_euler = (0, 0, b.get("yaw", 0))
    m = bpy.data.materials.new("prop")
    m.use_nodes = True
    m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (*b.get("color", (0.05, 0.05, 0.05)), 1)
    m.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.5
    ob.data.materials.append(m)
    ob.pass_index = 99

for L in spec["lights"]:
    ld = bpy.data.lights.new("L", L.get("type", "AREA"))
    ld.energy = L["energy"]
    if ld.type == "AREA":
        ld.size = L.get("size", 1.0)
    ld.color = L.get("color", (1, 1, 1))
    lo = bpy.data.objects.new("L", ld)
    scene.collection.objects.link(lo)
    lo.location = L["pos"]
    tgt = L.get("target", [L["pos"][0], L["pos"][1], 0])
    d = np.array(tgt, float) - np.array(L["pos"], float)
    lo.rotation_euler = Matrix.Identity(3).to_euler()
    from mathutils import Vector
    lo.rotation_euler = Vector(d.tolist()).to_track_quat("-Z", "Y").to_euler()

cd = bpy.data.cameras.new("cam")
co = bpy.data.objects.new("cam", cd)
scene.collection.objects.link(co)
scene.camera = co
W, H = cam_spec["width"], cam_spec["height"]
cd.sensor_fit = "HORIZONTAL"
cd.sensor_width = 36.0
cd.lens = cam_spec["fx"] * 36.0 / W
scene.render.pixel_aspect_x = 1.0
scene.render.pixel_aspect_y = cam_spec["fx"] / cam_spec["fy"]
cd.shift_x = (W / 2 - cam_spec["cx"]) / W
cd.shift_y = (cam_spec["cy"] - H / 2) / W
cd.clip_start = 0.05
cd.clip_end = 50

scene.use_nodes = True
nt = scene.node_tree
for n in list(nt.nodes):
    nt.nodes.remove(n)
rl = nt.nodes.new("CompositorNodeRLayers")
fo = nt.nodes.new("CompositorNodeOutputFile")
fo.format.file_format = "OPEN_EXR"
fo.format.color_depth = "32"
fo.format.exr_codec = "ZIP"
fo.file_slots.clear()
fo.file_slots.new("depth")
fo.file_slots.new("index")
nt.links.new(rl.outputs["Depth"], fo.inputs["depth"])
nt.links.new(rl.outputs["IndexOB"], fo.inputs["index"])
comp = nt.nodes.new("CompositorNodeComposite")
nt.links.new(rl.outputs["Image"], comp.inputs["Image"])

CV2BL = Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
for i, pose in enumerate(cam_spec["poses"]):
    R = Matrix([list(r) for r in pose["R"]]) @ CV2BL
    M = R.to_4x4()
    M.translation = pose["t"]
    co.matrix_world = M
    fdir = OUT / f"f_{i:04d}"
    fdir.mkdir(exist_ok=True)
    fo.base_path = str(fdir) + "/"
    scene.render.filepath = str(fdir / "color.png")
    scene.frame_set(1)
    bpy.ops.render.render(write_still=True)
    print(f"RENDERED {i + 1}/{len(cam_spec['poses'])}", flush=True)
