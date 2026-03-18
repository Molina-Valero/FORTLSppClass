"""
YOLO Image Classification Trainer
Usage:
  python train_classifier.py --data "C:/path/to/dataset"
  python train_classifier.py --data "C:/path/to/dataset" --model yolov8s-cls.pt --epochs 100 --imgsz 640 --batch 16 --name my_run
"""

import sys
import os
import argparse
from multiprocessing import freeze_support


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Image Classification Trainer")
    parser.add_argument("--data",   required=True,             help="Path to dataset folder or .yaml file")
    parser.add_argument("--model",  default="yolov8n-cls.pt",  help="Model to use (default: yolov8n-cls.pt)")
    parser.add_argument("--epochs", default=50,   type=int,    help="Number of epochs (default: 50)")
    parser.add_argument("--imgsz",  default=640,  type=int,    help="Image size (default: 640)")
    parser.add_argument("--batch",  default=16,   type=int,    help="Batch size (default: 16)")
    parser.add_argument("--name",   default="my_classifier",   help="Experiment name (default: my_classifier)")
    parser.add_argument("--device", default=None,              help="Device: 0 (GPU), cpu, mps. Auto-detected if not set.")
    parser.add_argument("--test",   default=None,              help="Optional: path to a test image for prediction")
    return parser.parse_args()


def main():
    args = parse_args()

    # ─────────────────────────────────────────
    #  CHECK DEPENDENCIES
    # ─────────────────────────────────────────

    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n[ERROR] ultralytics is not installed.")
        print("Run: python -m pip install ultralytics\n")
        sys.exit(1)

    # ─────────────────────────────────────────
    #  AUTO-DETECT DEVICE
    # ─────────────────────────────────────────

    try:
        import torch

        if args.device is not None:
            device = args.device
            print(f"[INFO] Using device: {device}")
        elif torch.cuda.is_available():
            device = 0
            print(f"[INFO] GPU detected: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = "mps"
            print("[INFO] Apple Silicon (MPS) detected")
        else:
            device = "cpu"
            print("[INFO] No GPU detected — training on CPU (will be slower)")

    except ImportError:
        device = "cpu"
        print("[WARNING] torch not found, defaulting to CPU")

    # ─────────────────────────────────────────
    #  VALIDATE DATASET PATH
    # ─────────────────────────────────────────

    if not os.path.exists(args.data):
        print(f"\n[ERROR] Dataset path not found: '{args.data}'")
        print("Make sure the folder exists and the path is correct.\n")
        sys.exit(1)

    # ─────────────────────────────────────────
    #  TRAIN
    # ─────────────────────────────────────────

    print("\n" + "="*50)
    print("  YOLO Classification Training")
    print("="*50)
    print(f"  Model      : {args.model}")
    print(f"  Dataset    : {args.data}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Image size : {args.imgsz}")
    print(f"  Batch size : {args.batch}")
    print(f"  Device     : {device}")
    print(f"  Output     : runs/classify/{args.name}/")
    print("="*50 + "\n")

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        name=args.name,
    )

    # ─────────────────────────────────────────
    #  EVALUATE
    # ─────────────────────────────────────────

    print("\n[INFO] Evaluating on validation set...")
    metrics = model.val()
    print(f"\n  Top-1 Accuracy : {metrics.top1:.4f}")
    print(f"  Top-5 Accuracy : {metrics.top5:.4f}")

    # ─────────────────────────────────────────
    #  PREDICT (optional)
    # ─────────────────────────────────────────

    if args.test and os.path.exists(args.test):
        print(f"\n[INFO] Running prediction on: {args.test}")
        results = model.predict(args.test)
        probs = results[0].probs
        print(f"  Predicted class : {results[0].names[probs.top1]}")
        print(f"  Confidence      : {probs.top1conf:.4f}")

    print("\n[DONE] Training complete!")
    print(f"       Best weights saved to: runs/classify/{args.name}/weights/best.pt\n")


if __name__ == '__main__':
    freeze_support()
    main()
