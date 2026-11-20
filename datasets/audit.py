from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .common import PROJECT_ROOT, frame_id, image_path, load_coco, load_json, resolve_data_root


def _distribution(values: list[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None, "q1": None, "q3": None, "iqr_outliers": 0}
    q1, q3 = np.quantile(values, [0.25, 0.75])
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    return {"count": len(values), "min": float(min(values)), "median": float(statistics.median(values)),
            "max": float(max(values)), "q1": float(q1), "q3": float(q3),
            "iqr_outliers": int(sum(value < lo or value > hi for value in values))}


def _audit_annotations(data_root: Path, name: str) -> dict[str, Any]:
    coco = load_coco(data_root, name)
    id_by_image = coco["_image_by_id"]
    ann_by_image = coco["_annotations_by_image"]
    attrs = [ann.get("attributes", {}) for ann in coco["annotations"]]
    forms = [a.get("form") for a in attrs]
    materials = [a.get("material") for a in attrs]
    volumes = [float(a["volume_ml"]) for a in attrs if a.get("volume_ml") is not None]
    masses = [float(a["mass_g"]) for a in attrs if a.get("mass_g") is not None]
    object_ids = [str(a["object_id"]) for a in attrs if a.get("object_id") is not None]
    form_materials = Counter((a.get("form"), a.get("material")) for a in attrs)
    per_image = Counter(int(ann["image_id"]) for ann in coco["annotations"])
    missing_files: list[str] = []
    bad_dimensions: list[str] = []
    invalid_boxes: list[int] = []
    invalid_polygons: list[int] = []
    bbox_polygon_mismatch: list[int] = []
    actual_sizes = Counter()
    depth_zero_share: list[float] = []
    depth_sizes = Counter()
    meta_holes: list[float] = []
    missing_depths: list[str] = []
    missing_floor_planes: list[str] = []
    meta_surfaces: list[str] = []
    meta_lighting: list[str] = []
    meta_time_count = 0
    synthetic_flag_count = 0
    for image in coco["images"]:
        path = image_path(data_root, image)
        if not path.is_file():
            missing_files.append(image["file_name"])
        elif path.is_file():
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                missing_files.append(image["file_name"])
            else:
                actual_sizes[f"{frame.shape[1]}x{frame.shape[0]}"] += 1
                if frame.shape[1] != int(image["width"]) or frame.shape[0] != int(image["height"]):
                    bad_dimensions.append(image["file_name"])
        depth_path = path.parent / "depth.png"
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED) if depth_path.is_file() else None
        if depth is None:
            missing_depths.append(str(depth_path))
        else:
            depth_zero_share.append(float(np.mean(depth == 0)))
            depth_sizes[f"{depth.shape[1]}x{depth.shape[0]}:{depth.dtype}"] += 1
        meta_path = path.parent / "meta.json"
        if not meta_path.is_file():
            missing_floor_planes.append(str(meta_path))
        else:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            plane = meta.get("floor_plane")
            if not isinstance(plane, list) or len(plane) != 4:
                missing_floor_planes.append(str(meta_path))
            if meta.get("holes_share") is not None:
                meta_holes.append(float(meta["holes_share"]))
            if meta.get("surface") is not None:
                meta_surfaces.append(str(meta["surface"]))
            if meta.get("lighting") is not None:
                meta_lighting.append(str(meta["lighting"]))
            meta_time_count += int(meta.get("time") is not None)
            synthetic_flag_count += int(bool(meta.get("synthetic", False)))
        for ann in ann_by_image.get(int(image["id"]), []):
            x, y, w, h = [float(v) for v in ann.get("bbox", [0, 0, 0, 0])]
            if not np.isfinite([x, y, w, h]).all() or w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > image["width"] + 1 or y + h > image["height"] + 1:
                invalid_boxes.append(int(ann["id"]))
            seg = ann.get("segmentation", [])
            points = []
            if isinstance(seg, list):
                for poly in seg:
                    if len(poly) < 6 or len(poly) % 2:
                        continue
                    xy = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
                    if not np.isfinite(xy).all():
                        continue
                    points.append(xy)
            if not points:
                invalid_polygons.append(int(ann["id"]))
            else:
                all_points = np.concatenate(points)
                if (all_points[:, 0].min() < -1 or all_points[:, 1].min() < -1 or
                    all_points[:, 0].max() > image["width"] + 1 or all_points[:, 1].max() > image["height"] + 1):
                    invalid_polygons.append(int(ann["id"]))
                else:
                    bx1, by1 = all_points.min(axis=0)
                    bx2, by2 = all_points.max(axis=0)
                    if max(abs(x - bx1), abs(y - by1), abs((x + w) - bx2), abs((y + h) - by2)) > 2.0:
                        bbox_polygon_mismatch.append(int(ann["id"]))
    attrs_missing = sum(any(key not in a or a[key] is None for key in
                            ("form", "material", "volume_ml", "mass_g", "take", "object_id")) for a in attrs)
    object_label_conflicts: dict[str, int] = {}
    by_object: dict[str, set[tuple]] = defaultdict(set)
    for a in attrs:
        oid = a.get("object_id")
        if oid is not None:
            by_object[str(oid)].add((a.get("form"), a.get("material"), a.get("volume_ml"), a.get("mass_g")))
    object_label_conflicts = {key: len(values) for key, values in by_object.items() if len(values) > 1}
    return {
        "frames": len(coco["images"]), "objects": len(coco["annotations"]),
        "object_ids_unique": len(set(object_ids)),
        "forms": _distribution(forms), "materials": _distribution(materials),
        "form_material_counts": {f"{key[0]}|{key[1]}": value for key, value in sorted(form_materials.items())},
        "volume_ml": _numeric_summary(volumes), "mass_g": _numeric_summary(masses),
        "take": _distribution([a.get("take") for a in attrs]),
        "instances_per_frame": _distribution(list(per_image.values())),
        "empty_frames": len(coco["images"]) - len(per_image),
        "image_sizes": dict(actual_sizes), "missing_or_unreadable_images": missing_files,
        "metadata_dimension_mismatches": bad_dimensions,
        "missing_required_attributes": attrs_missing,
        "depth_maps_missing": missing_depths, "depth_sizes_and_types": dict(depth_sizes),
        "depth_zero_share": _numeric_summary(depth_zero_share),
        "metadata_missing_floor_plane": missing_floor_planes,
        "meta_holes_share": _numeric_summary(meta_holes),
        "metadata_surface": _distribution(meta_surfaces),
        "metadata_lighting": _distribution(meta_lighting),
        "metadata_time_count": meta_time_count,
        "metadata_synthetic_true_count": synthetic_flag_count,
        "invalid_boxes": invalid_boxes, "invalid_or_out_of_frame_polygons": invalid_polygons,
        "bbox_polygon_extrema_diff_gt_2px": bbox_polygon_mismatch,
        "object_label_conflicts": object_label_conflicts,
        "ids_by_object": {key: sorted(values)[0] for key, values in by_object.items() if values},
    }


def run_audit(data_root: Path, sample_submission: Path) -> dict[str, Any]:
    real = _audit_annotations(data_root, "real_train")
    synth = _audit_annotations(data_root, "synth_train")
    camera = load_json(data_root / "camera.json")
    density = pd.read_csv(data_root / "densities.csv").to_dict(orient="records")
    sample_ids = pd.read_csv(sample_submission)["image_id"].astype(str).tolist()
    test_frames = sorted(path for path in (data_root / "real" / "test").iterdir() if path.is_dir())
    test_ids = [path.name for path in test_frames]
    test_missing = []
    test_meta_missing_floor = []
    test_depth_zero: list[float] = []
    test_depth_sizes = Counter()
    for folder in test_frames:
        color = cv2.imread(str(folder / "color.png"), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(folder / "depth.png"), cv2.IMREAD_UNCHANGED)
        meta_path = folder / "meta.json"
        if color is None or depth is None or not meta_path.is_file():
            test_missing.append(folder.name)
            continue
        meta = load_json(meta_path)  # meta contains sensor/context only; test labels are never loaded.
        if not isinstance(meta.get("floor_plane"), list) or len(meta["floor_plane"]) != 4:
            test_meta_missing_floor.append(folder.name)
        test_depth_zero.append(float(np.mean(depth == 0)))
        test_depth_sizes[f"{depth.shape[1]}x{depth.shape[0]}:{depth.dtype}"] += 1
    real_ids = set(real.pop("ids_by_object"))
    synth_ids = set(synth.pop("ids_by_object"))
    # Attribute agreement between copies is verified against both train-label files only.
    real_coco = load_coco(data_root, "real_train")
    synth_coco = load_coco(data_root, "synth_train")
    key_for = lambda ann: (ann["attributes"]["form"], ann["attributes"]["material"],
                           float(ann["attributes"]["volume_ml"]), float(ann["attributes"]["mass_g"]))
    real_obj, synth_obj = {}, {}
    for ann in real_coco["annotations"]:
        real_obj.setdefault(str(ann["attributes"]["object_id"]), set()).add(key_for(ann))
    for ann in synth_coco["annotations"]:
        synth_obj.setdefault(str(ann["attributes"]["object_id"]), set()).add(key_for(ann))
    overlap = real_ids & synth_ids
    mismatches = sorted(oid for oid in overlap if real_obj.get(oid) != synth_obj.get(oid))
    report = {
        "real_train": real, "synth_train": synth,
        "real_test": {
            "frames": len(test_ids), "sample_submission_rows": len(sample_ids),
            "sample_ids_unique": len(sample_ids) == len(set(sample_ids)),
            "test_ids_match_sample_submission": sorted(test_ids) == sorted(sample_ids),
            "missing_rgb_depth_or_meta": test_missing,
            "meta_missing_floor_plane": test_meta_missing_floor,
            "depth_map_shapes_and_types": dict(test_depth_sizes),
            "depth_zero_share": _numeric_summary(test_depth_zero),
            "label_source": "No real_test labels file found; test meta is sensor/context only.",
        },
        "camera": camera,
        "densities_rows": density,
        "real_synth_object_id_overlap": len(overlap),
        "real_synth_object_ids_union": len(real_ids | synth_ids),
        "real_synth_attribute_mismatches": mismatches,
        "leakage_note": "Synthetic and real training records share every physical object_id; filter synthetic frames by grouped split before all training.",
    }
    return report


def write_markdown(report: dict[str, Any], path: Path) -> None:
    real, synth, test = report["real_train"], report["synth_train"], report["real_test"]
    lines = [
        "# Astramis data audit", "",
        "The report reads only `real_train.json` and `synth_train.json` for labels. Test data contributes RGB-D/meta availability and submission-ID checks; no test labels are read.", "",
        "## Dataset inventory", "",
        f"- Real train: {real['frames']} frames, {real['objects']} objects, {real['object_ids_unique']} physical object IDs, {real['empty_frames']} empty frames.",
        f"- Synthetic train: {synth['frames']} frames, {synth['objects']} objects, {synth['object_ids_unique']} physical object IDs.",
        f"- Real test: {test['frames']} frames, {test['sample_submission_rows']} submission rows; IDs match: {test['test_ids_match_sample_submission']}.",
        f"- Actual RGB dimensions: real {real['image_sizes']}, synthetic {synth['image_sizes']}; depth types/sizes: real {real['depth_sizes_and_types']}, synthetic {synth['depth_sizes_and_types']}, test {test['depth_map_shapes_and_types']}.",
        "",
        "## Labels", "",
        f"- Real forms: `{real['forms']}`; materials: `{real['materials']}`.",
        f"- Synthetic forms: `{synth['forms']}`; materials: `{synth['materials']}`.",
        f"- Real volume (ml): `{real['volume_ml']}`; mass (g): `{real['mass_g']}`.",
        f"- Missing required real attributes: {real['missing_required_attributes']}; invalid boxes: {len(real['invalid_boxes'])}; invalid polygons: {len(real['invalid_or_out_of_frame_polygons'])}.",
        f"- Synthetic bbox/polygon extrema differ by >2 px in {len(synth['bbox_polygon_extrema_diff_gt_2px'])} annotations.",
        "",
        "## RGB-D and leakage", "",
        f"- Camera: {report['camera']['model']}, {report['camera']['width']}x{report['camera']['height']}, fx={report['camera']['fx']:.3f}, fy={report['camera']['fy']:.3f}; depth units `{report['camera']['depth_unit']}` and aligned to color.",
        f"- Floor plane missing: real {len(real['metadata_missing_floor_plane'])}, synthetic {len(synth['metadata_missing_floor_plane'])}, test {len(test['meta_missing_floor_plane'])}.",
        f"- Depth zero-share: real `{real['depth_zero_share']}`, synthetic `{synth['depth_zero_share']}`, test `{test['depth_zero_share']}`.",
        f"- Real metadata holes_share: `{real['meta_holes_share']}`; synthetic `holes_share` is absent. Real lighting/surface: `{real['metadata_lighting']}` / `{real['metadata_surface']}`; synthetic: `{synth['metadata_lighting']}` / `{synth['metadata_surface']}`.",
        f"- Real/synthetic object ID overlap: {report['real_synth_object_id_overlap']} of {report['real_synth_object_ids_union']}; attribute mismatches: {len(report['real_synth_attribute_mismatches'])}.",
        "- Therefore synthetic views must be filtered by the same held-out `object_id` split. The pipeline enforces this in `datasets/splits.py`.",
        "",
        "## Quality flags", "",
        f"- Real missing/unreadable images: {len(real['missing_or_unreadable_images'])}; metadata size mismatches: {len(real['metadata_dimension_mismatches'])}; object-ID label conflicts: {len(real['object_label_conflicts'])}.",
        f"- Synthetic missing/unreadable images: {len(synth['missing_or_unreadable_images'])}; metadata size mismatches: {len(synth['metadata_dimension_mismatches'])}; object-ID label conflicts: {len(synth['object_label_conflicts'])}.",
        f"- Test missing RGB/depth/meta: {len(test['missing_rgb_depth_or_meta'])}; test metadata missing floor planes: {len(test['meta_missing_floor_plane'])}.",
        "",
        "## Notes", "",
        "Volume/mass IQR outlier counts are recorded in `data_audit.json`; they flag review candidates and do not cause samples to be deleted. Geometric volume is an observed-surface estimate, so hidden surfaces and depth holes remain a known source of bias.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--sample-submission", default=None)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "references"))
    args = parser.parse_args()
    data_root = resolve_data_root(args.data_root)
    if args.sample_submission:
        sample = Path(args.sample_submission)
    else:
        sample = PROJECT_ROOT / "references" / "sample_submission.csv"
        if not sample.is_file():
            sample = PROJECT_ROOT.parent / "sample_submission.csv"
    report = run_audit(data_root, sample)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "data_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, out / "data_audit.md")
    print(f"Wrote {out / 'data_audit.md'} and {out / 'data_audit.json'}")


if __name__ == "__main__":
    main()

