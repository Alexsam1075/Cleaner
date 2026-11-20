from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    project_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_dir.parent))
    __package__ = f"{project_dir.name}.evaluation"

from ..datasets.common import PROJECT_ROOT

EXPERIMENTS = ["A_real", "B_synth_then_real", "C_joint", "D_domain_randomized"]
FUSIONS = ["rgb_only", "late", "intermediate", "intermediate_geom"]
PRIMARY_FUSION = "intermediate_geom"


def _report(root: Path, experiment: str, fusion: str) -> dict:
    path = root / "logs" / "validation" / experiment / fusion / "report.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing validation report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_cell(metric: dict) -> str:
    return (f"{metric['D']:.3f} / {metric['M']:.3f} / {metric['V']:.3f} / "
            f"{metric['S']:.3f} / {metric['Score']:.2f}")


def create_report(root: Path, output_path: Path) -> Path:
    primary = {name: _report(root, name, PRIMARY_FUSION) for name in EXPERIMENTS}
    fusion_reports = {fusion: _report(root, "C_joint", fusion) for fusion in FUSIONS}
    split = primary["A_real"]["split"]
    lines = [
        "# Astramis experiment report", "",
        "Metrics are computed by the supplied, unchanged `score.py` on the fixed validation frames grouped by physical `object_id`. No test annotations are used.", "",
        "## Validation split", "",
        f"- Seed {split['seed']}, fold {split['fold']}: {len(split['validation_object_ids'])} held-out physical IDs and {split['validation_frames']} real frames ({split['validation_objects']} annotated objects).",
        f"- Held-out IDs by material: `{json.dumps(split['validation_object_ids_by_material'], ensure_ascii=False)}`.",
        "- The same fold was used for every experiment. Synthetic frames containing any held-out ID were excluded from pretraining and joint training.",
        "- M is macro accuracy by represented material; S averages take accuracy for GT mass groups above and at/below 500 g.",
        "",
        "## Data strategy results", "",
        "D / M / V / S / Score are shown as fractions / points, with Score on the 0–100 scale.", "",
        "| Condition | Baseline logic on same detector | Train-fit geometry calibration | RGB-D multimodal | Mass MAE (g) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in EXPERIMENTS:
        report = primary[name]
        metrics = report["metrics"]
        mass_mae = report["error_analysis"]["multimodal"].get("mass_mae_g_on_matched_objects")
        mass_text = "n/a" if mass_mae is None else f"{mass_mae:.1f}"
        lines.append(f"| `{name}` | {_metric_cell(metrics['baseline_logic_same_detector'])} | "
                     f"{_metric_cell(metrics['geometry_calibrated_train_fit'])} | "
                     f"{_metric_cell(metrics['multimodal'])} | {mass_text} |")
    lines += ["", "## Fusion ablation on joint training data", "",
              "The RGB-only, late-fusion, intermediate RGB-D, and intermediate RGB-D plus geometry heads share the C_joint detector and the same object-ID split.", "",
              "| Attribute fusion | D / M / V / S / Score | Matched-object mass MAE (g) |",
              "| --- | ---: | ---: |"]
    for fusion in FUSIONS:
        report = fusion_reports[fusion]
        metrics = report["metrics"]["multimodal"]
        mass_mae = report["error_analysis"]["multimodal"].get("mass_mae_g_on_matched_objects")
        mass_text = "n/a" if mass_mae is None else f"{mass_mae:.1f}"
        lines.append(f"| `{fusion}` | {_metric_cell(metrics)} | {mass_text} |")
    best_strategy = max(EXPERIMENTS, key=lambda name: primary[name]["metrics"]["multimodal"]["Score"])
    best_fusion = max(FUSIONS, key=lambda name: fusion_reports[name]["metrics"]["multimodal"]["Score"])
    lines += ["", "## Error details", "",
              "Per-material class accuracy and per-form matched recall are recorded in each validation `report.json`. Volume accuracy by GT range and mass MAE by material are also included there.", "",
              f"Highest observed data-strategy Score on this one grouped fold: `{best_strategy}`.",
              f"Highest observed fusion Score on this one grouped fold: `{best_fusion}`.",
              "These are selection results on a small validation population, not a final generalization estimate. Some represented rare materials are supported by one held-out physical object; use grouped cross-validation or more real objects before making a deployment claim.", "",
              "## Baseline interpretation", "",
              "The baseline comparator restores majority-material prediction, depth/floor-plane cell summation, median-volume fallback, and density-derived mass/take while using the same RGB detector as the associated model. The calibrated comparator fits a linear correction in log-volume space using training object IDs only. This controlled comparison is not an exact reproduction of the notebook's original split or epoch settings.", "",
              "## Known limitations and next work", "",
              "- Visible depth cannot recover hidden surfaces; report geometric and learned volume errors separately and collect more real object IDs.",
              "- Mass MAE is supplementary because the provided scorer does not score mass directly; take remains scored.",
              "- Masks are used by the geometry and visualization pipeline, while the supplied score measures bounding boxes rather than mask AP.",
              "- A follow-up should run grouped cross-validation, inspect per-object repeated-frame errors, and only keep synthetic/domain-randomization paths if held-out Score and generation outputs both improve.", ""]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=str(PROJECT_ROOT))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    root = Path(args.project).resolve()
    output = Path(args.output).resolve() if args.output else root / "reports" / "experiment_report.md"
    print(create_report(root, output))


if __name__ == "__main__":
    main()

