"""Build README media: GIFs from recorded frames and stills from the renderer and the generator.

  python -m polygon.make_media --out docs/media
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

from .fast_scene import add_robot, random_trash, trash_material
from .fastrt import FastCam, FloorTex, Mat, Prim, Scene, render, to_u8

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def gif(frames: list[str], out: Path, fps: int, width: int):
    tmp = out.with_suffix(".list")
    tmp.write_text("".join(f"file '{Path(f).resolve()}'\nduration {1 / fps:.4f}\n" for f in frames))
    vf = f"scale={width}:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
    subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(tmp), "-vf", vf,
                    "-loop", "0", str(out)], check=True)
    tmp.unlink()
    print(out, f"{out.stat().st_size / 1e6:.1f} MB")


def video_gif(src: Path, out: Path, start: float, dur: float, fps: int, width: int):
    vf = f"fps={fps},scale={width}:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
    subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-ss", str(start), "-t", str(dur), "-i", str(src), "-vf", vf,
                    "-loop", "0", str(out)], check=True)
    print(out, f"{out.stat().st_size / 1e6:.1f} MB")


def showcase(out: Path):
    """One curated CG frame with every material: glass, metal, plastic, paper, wood, stone."""
    sc = Scene()
    rng = np.random.default_rng(11)
    floor = FloorTex(HERE / "assets/floor_tile.png")
    add_robot(sc, -0.55, -0.15, 0.25, brush_angle=0.4, wheel_angle=0.6)
    picks = [("glass", (0.32, 0.18)), ("metal", (0.45, -0.12)), ("plastic", (0.62, 0.22)), ("paper", (0.18, -0.32)),
             ("wood", (0.72, -0.05)), ("mineral", (0.5, 0.42)), ("plastic", (0.85, 0.35)), ("glass", (0.95, -0.3))]
    for k, (mat, (x, y)) in enumerate(picks):
        t = random_trash(rng, mat)
        t.place(sc, x, y, rng.uniform(0, 6.28), k)
    cam = FastCam.look_at((-0.25, -0.95, 0.62), (0.45, 0.05, 0.05), W=1600, H=900, fov_deg=48)
    img = to_u8(render(sc, cam, floor=floor, spp=8)["rgb"])
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(out)


def generator_grid(src: Path, out: Path, n=4):
    fs = sorted(glob.glob(str(src / "train/*")))[:n]
    rows = []
    for f in fs:
        c = cv2.imread(f + "/color.png")
        d = cv2.imread(f + "/depth.png", -1).astype(np.float32)
        dv = cv2.applyColorMap(np.clip((d - 200) / 1200 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        dv[d == 0] = 0
        rows.append(np.hstack([cv2.resize(c, (640, 360)), cv2.resize(dv, (640, 360))]))
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(out)


def real_vs_synth(data: Path, synth: Path, out: Path):
    lab = json.loads((data / "labels/real_train.json").read_text())
    names = {i["id"]: Path(i["file_name"]).parent.name for i in lab["images"]}
    used = sorted({names[a["image_id"]] for a in lab["annotations"]})
    real = [cv2.imread(str(data / "real/train" / used[k] / "color.png")) for k in (5, 60)]
    syn = [cv2.imread(f + "/color.png") for f in sorted(glob.glob(str(synth / "train/*")))[4:6]]
    tile = lambda im, txt: cv2.putText(cv2.resize(im, (640, 360)), txt, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                        (255, 255, 255), 2, cv2.LINE_AA)
    g = np.vstack([np.hstack([tile(real[0], "real D435"), tile(syn[0], "synthetic")]),
                   np.hstack([tile(real[1], "real D435"), tile(syn[1], "synthetic")])])
    cv2.imwrite(str(out), g, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(out)


def volume_chart(out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folds = ["fold 1", "fold 2", "fold 3"]
    old, pred, oracle = [0.595, 0.524, 0.603], [0.691, 0.629, 0.649], [0.718, 0.719, 0.787]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=150)
    for i, (vals, lab, col) in enumerate([(old, "старая геометрия", "#9aa1a9"), (pred, "модуль, материал модели", "#4a90d9"),
                                          (oracle, "модуль, верный материал", "#2e7d4f")]):
        b = ax.bar(x + (i - 1) * 0.26, vals, 0.24, label=lab, color=col)
        ax.bar_label(b, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x, folds)
    ax.set_ylim(0.4, 0.85)
    ax.set_ylabel("V (официальный scorer)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=3)
    fig.tight_layout()
    fig.savefig(out)
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "docs/media")
    ap.add_argument("--raw", type=Path, default=REPO / "docs/media/raw")
    ap.add_argument("--gen", type=Path, default=REPO / "polygon_out/gen_fast")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    chase = sorted(glob.glob(str(args.raw / "chase/*.jpg")))
    gif(chase[::12], args.out / "polygon_autopilot.gif", fps=8, width=480)
    model = sorted(glob.glob(str(args.raw / "model/*.jpg")))
    gif(model, args.out / "model_view.gif", fps=3, width=560)
    video_gif(REPO / "polygon_out/robot.mp4", args.out / "robot_animation.gif", 1, 7, 8, 480)
    showcase(args.out / "renderer_materials.jpg")
    generator_grid(args.gen, args.out / "generator_rgbd.jpg")
    real_vs_synth(REPO / "datasets/raw", args.gen, args.out / "real_vs_synthetic.jpg")
    volume_chart(args.out / "volume_by_fold.png")


if __name__ == "__main__":
    main()
