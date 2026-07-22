"""
YOLO Image Classification Trainer
Usage:
  python train_classifier.py --data "C:/path/to/dataset"
  python train_classifier.py --data "C:/path/to/dataset" --model yolov8s-cls.pt --epochs 100 --imgsz 640 --batch 16 --name my_run

  # Or point directly at separate train/val folders (no need to physically
  # merge or copy them — a small linked folder is created automatically):
  python train_classifier.py --train "C:/path/to/train" --val "C:/path/to/val" --name my_run
"""

import sys
import os
import argparse
import subprocess
import shutil
from multiprocessing import freeze_support


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Image Classification Trainer")
    parser.add_argument("--data",   default=None,              help="Path to dataset folder (with train/ and val/ subfolders) or .yaml file")
    parser.add_argument("--train",  default=None,              help="Path to the training images folder (per-class subfolders). Use together with --val instead of --data.")
    parser.add_argument("--val",    default=None,              help="Path to the validation images folder (per-class subfolders). Use together with --train instead of --data.")
    parser.add_argument("--model",  default="yolov8n-cls.pt",  help="Model to use (default: yolov8n-cls.pt)")
    parser.add_argument("--epochs", default=50,   type=int,    help="Number of epochs (default: 50)")
    parser.add_argument("--imgsz",  default=640,  type=int,    help="Image size (default: 640)")
    parser.add_argument("--batch",  default=16,   type=int,    help="Batch size (default: 16)")
    parser.add_argument("--name",   default="my_classifier",   help="Experiment name (default: my_classifier)")
    parser.add_argument("--device", default=None,              help="Device: 0 (GPU), cpu, mps. Auto-detected if not set.")
    parser.add_argument("--test",   default=None,              help="Optional: path to a test image for prediction")
    args = parser.parse_args()

    if args.data and (args.train or args.val):
        parser.error("Use either --data OR --train/--val, not both.")
    if bool(args.train) != bool(args.val):
        parser.error("--train and --val must be provided together.")
    if not args.data and not args.train:
        parser.error("You must provide either --data, or both --train and --val.")

    return args


def make_junction(link_path, target_path):
    """Create (or replace) a Windows directory junction at link_path pointing to target_path.
    Junctions don't require admin rights and don't copy any files."""
    target_path = os.path.abspath(target_path)

    if os.path.islink(link_path) or os.path.isdir(link_path):
        # Remove the existing junction/link itself (NOT the real target folder).
        # rmdir on a junction only removes the link, never the linked contents.
        try:
            os.rmdir(link_path)
        except OSError:
            shutil.rmtree(link_path)

    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", link_path, target_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"\n[ERROR] Failed to create junction:\n  {link_path} -> {target_path}")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)


def build_dataset_dir(train_path, val_path, name):
    """Build a dataset_root/train + dataset_root/val layout using junctions,
    without copying any images. Returns the dataset_root path."""
    for label, p in (("--train", train_path), ("--val", val_path)):
        if not os.path.exists(p):
            print(f"\n[ERROR] {label} path not found: '{p}'\n")
            sys.exit(1)

    dataset_root = os.path.abspath(os.path.join("runs", "dataset_links", name))
    os.makedirs(dataset_root, exist_ok=True)

    train_link = os.path.join(dataset_root, "train")
    val_link = os.path.join(dataset_root, "val")

    print(f"[INFO] Linking train folder: {train_link} -> {train_path}")
    make_junction(train_link, train_path)
    print(f"[INFO] Linking val folder  : {val_link} -> {val_path}")
    make_junction(val_link, val_path)

    return dataset_root


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
    #  RESOLVE DATASET PATH
    # ─────────────────────────────────────────

    if args.train and args.val:
        data_path = build_dataset_dir(args.train, args.val, args.name)
    else:
        data_path = args.data
        if not os.path.exists(data_path):
            print(f"\n[ERROR] Dataset path not found: '{data_path}'")
            print("Make sure the folder exists and the path is correct.\n")
            sys.exit(1)

    # ─────────────────────────────────────────
    #  TRAIN
    # ─────────────────────────────────────────

    print("\n" + "="*50)
    print("  YOLO Classification Training")
    print("="*50)
    print(f"  Model      : {args.model}")
    print(f"  Dataset    : {data_path}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Image size : {args.imgsz}")
    print(f"  Batch size : {args.batch}")
    print(f"  Device     : {device}")
    print(f"  Output     : runs/classify/{args.name}/")
    print("="*50 + "\n")

    model = YOLO(args.model)

    model.train(
        data=data_path,
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
