from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import robustness_report, save_gradcam
from .config import load_config
from .data import create_split_manifest
from .inference import predict_ensemble_video, predict_image, predict_video
from .preprocessing import preprocess_dataset
from .training import evaluate_checkpoint, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deepfake detection training pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("preprocess")
    split = commands.add_parser("split")
    split.add_argument("--force", action="store_true")

    for name in ("train-frame", "train-temporal"):
        command = commands.add_parser(name)
        command.add_argument("--device")

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("checkpoint")
    evaluate.add_argument("--device")

    predict = commands.add_parser("predict-image")
    predict.add_argument("checkpoint")
    predict.add_argument("image")
    predict.add_argument("--output", default="outputs/prediction.jpg")
    predict.add_argument("--device")

    predict_video_command = commands.add_parser("predict-video")
    predict_video_command.add_argument("checkpoint")
    predict_video_command.add_argument("video")
    predict_video_command.add_argument("--output", default="outputs/prediction.mp4")
    predict_video_command.add_argument("--device")
    predict_video_command.add_argument("--sequence-length", type=int, default=16)

    ensemble = commands.add_parser("predict-ensemble-video")
    ensemble.add_argument("frame_checkpoint")
    ensemble.add_argument("temporal_checkpoint")
    ensemble.add_argument("video")
    ensemble.add_argument("--output", default="outputs/ensemble.mp4")
    ensemble.add_argument("--device")
    ensemble.add_argument("--sequence-length", type=int, default=16)
    ensemble.add_argument("--frame-weight", type=float, default=0.4)

    robustness = commands.add_parser("robustness")
    robustness.add_argument("checkpoint")
    robustness.add_argument("--device")

    gradcam = commands.add_parser("gradcam")
    gradcam.add_argument("checkpoint")
    gradcam.add_argument("image")
    gradcam.add_argument("--output", default="outputs/gradcam.jpg")
    gradcam.add_argument("--device")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "preprocess":
        print(preprocess_dataset(config.data))
    elif args.command == "split":
        print(create_split_manifest(config, force=args.force))
    elif args.command == "train-frame":
        print(train(config, "frame", args.device))
    elif args.command == "train-temporal":
        print(train(config, "temporal", args.device))
    elif args.command == "evaluate":
        print(json.dumps(evaluate_checkpoint(config, args.checkpoint, args.device), indent=2))
    elif args.command == "predict-image":
        print(
            json.dumps(
                predict_image(args.checkpoint, args.image, args.output, args.device), indent=2
            )
        )
    elif args.command == "predict-video":
        print(
            json.dumps(
                predict_video(
                    args.checkpoint,
                    args.video,
                    args.output,
                    args.device,
                    args.sequence_length,
                ),
                indent=2,
            )
        )
    elif args.command == "predict-ensemble-video":
        print(
            json.dumps(
                predict_ensemble_video(
                    args.frame_checkpoint,
                    args.temporal_checkpoint,
                    args.video,
                    args.output,
                    args.device,
                    args.sequence_length,
                    args.frame_weight,
                ),
                indent=2,
            )
        )
    elif args.command == "robustness":
        print(json.dumps(robustness_report(config, args.checkpoint, args.device), indent=2))
    elif args.command == "gradcam":
        save_gradcam(args.checkpoint, args.image, args.output, args.device)
        print(Path(args.output))


if __name__ == "__main__":
    main()
