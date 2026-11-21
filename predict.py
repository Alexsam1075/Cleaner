from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    project_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(project_dir.parent))
    __package__ = project_dir.name

from .datasets.common import PROJECT_ROOT, resolve_data_root
from .inference.predictor import predict_ids, sample_submission_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create an Astramis test submission.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output", default=str(PROJECT_ROOT))
    parser.add_argument("--experiment", default="C_joint")
    parser.add_argument("--fusion", default="intermediate_geom")
    parser.add_argument("--detector", default=None)
    parser.add_argument("--attributes", default=None)
    parser.add_argument("--sample-submission", default=None)
    parser.add_argument("--submission", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--conf", type=float, default=0.08)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--volume-backend", choices=["net", "parts"], default="net")
    parser.add_argument("--volume-calibrator", default=None,
                        help="calibrator.json from evaluation.volume_parts_cv (needed for parts)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.volume_backend == "parts" and not args.volume_calibrator:
        raise SystemExit("--volume-backend parts needs --volume-calibrator")
    output = Path(args.output).resolve()
    data_root = resolve_data_root(args.data_root)
    detector = Path(args.detector).resolve() if args.detector else \
        output / "checkpoints" / "experiments" / args.experiment / "detector.pt"
    attributes = Path(args.attributes).resolve() if args.attributes else \
        output / "checkpoints" / "experiments" / args.experiment / f"attributes_{args.fusion}.pt"
    sample = Path(args.sample_submission).resolve() if args.sample_submission else \
        (output / "references" / "sample_submission.csv" if (output / "references" / "sample_submission.csv").is_file()
         else output.parent / "sample_submission.csv")
    submission = Path(args.submission).resolve() if args.submission else output / "submissions" / "submission.csv"
    visual_dir = output / "visualizations" / "test" if args.visualize else None
    predict_ids(data_root, sample_submission_ids(sample), detector, attributes, submission,
                args.conf, args.imgsz, args.device, visualize_dir=visual_dir,
                max_visualizations=24, volume_backend=args.volume_backend,
                volume_calibrator_path=args.volume_calibrator)
    print(f"Wrote {submission}")
    if visual_dir is not None:
        print(f"Saved overlays under {visual_dir}")


if __name__ == "__main__":
    main()

