"""
YOLO Image Classification Predictor
Usage:
  python predict_classifier.py --source "C:/path/to/image.jpg"
  python predict_classifier.py --source "C:/path/to/folder" --model best.pt --imgsz 640 --conf 0.25 --save
  python predict_classifier.py --source "C:/path/to/folder" --output results.csv
"""

import sys
import os
import csv
import argparse
from collections import defaultdict
from multiprocessing import freeze_support


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Image Classification Predictor")
    parser.add_argument("--source",  required=True,           help="Path to image, folder, or video file")
    parser.add_argument("--model",   default="best.pt",       help="Path to model weights (default: best.pt)")
    parser.add_argument("--imgsz",   default=640,  type=int,  help="Image size (default: 640)")
    parser.add_argument("--conf",    default=0.25, type=float,help="Confidence threshold (default: 0.25)")
    parser.add_argument("--top_k",   default=5,    type=int,  help="Show top-K predictions (default: 5)")
    parser.add_argument("--save",    action="store_true",     help="Save annotated results to disk")
    parser.add_argument("--save_txt",action="store_true",     help="Save predictions as .txt files")
    parser.add_argument("--device",  default=None,            help="Device: 0 (GPU), cpu, mps. Auto-detected if not set.")
    parser.add_argument("--output",  default="predictions.csv",help="Output CSV file path (default: predictions.csv)")
    parser.add_argument("--tree_output", default="predictions_tree.csv",
                         help="Output CSV path for per-tree (voted) predictions (default: predictions_tree.csv)")
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
            print("[INFO] No GPU detected — running on CPU")

    except ImportError:
        device = "cpu"
        print("[WARNING] torch not found, defaulting to CPU")

    # ─────────────────────────────────────────
    #  VALIDATE PATHS
    # ─────────────────────────────────────────

    if not os.path.exists(args.model):
        print(f"\n[ERROR] Model weights not found: '{args.model}'")
        print("Make sure the path to best.pt is correct.\n")
        sys.exit(1)

    if not os.path.exists(args.source):
        print(f"\n[ERROR] Source path not found: '{args.source}'")
        print("Make sure the image/folder path is correct.\n")
        sys.exit(1)

    # ─────────────────────────────────────────
    #  PRINT CONFIGURATION
    # ─────────────────────────────────────────

    print("\n" + "="*50)
    print("  YOLO Classification Prediction")
    print("="*50)
    print(f"  Model      : {args.model}")
    print(f"  Source     : {args.source}")
    print(f"  Image size : {args.imgsz}")
    print(f"  Confidence : {args.conf}")
    print(f"  Top-K      : {args.top_k}")
    print(f"  Device     : {device}")
    print(f"  Save       : {args.save}")
    print(f"  Save txt   : {args.save_txt}")
    print(f"  Output CSV : {args.output}")
    print(f"  Tree CSV   : {args.tree_output}")
    print("="*50 + "\n")

    # ─────────────────────────────────────────
    #  LOAD MODEL
    # ─────────────────────────────────────────

    print(f"[INFO] Loading model: {args.model}")
    model = YOLO(args.model)

    # ─────────────────────────────────────────
    #  PREDICT
    # ─────────────────────────────────────────

    print(f"[INFO] Running prediction on: {args.source}\n")

    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=device,
        save=args.save,
        save_txt=args.save_txt,
    )

    # ─────────────────────────────────────────
    #  DISPLAY RESULTS & WRITE CSV
    # ─────────────────────────────────────────

    csv_rows = []

    # tree_predictions[treeID] = list of (species, confidence) — one entry per image
    tree_predictions = defaultdict(list)

    print("-"*50)
    for i, result in enumerate(results):
        source_path = result.path if hasattr(result, "path") else f"sample_{i}"
        picture_id = os.path.splitext(os.path.basename(source_path))[0]
        # treeID is the prefix of the image filename, e.g. "00069_1" -> "00069"
        tree_id = picture_id.split("_")[0]
        print(f"\n  File : {os.path.basename(source_path)}")

        probs = result.probs
        if probs is not None:
            # Top-1 result
            predicted_species = result.names[probs.top1]
            confidence = probs.top1conf.item()
            print(f"  Predicted class : {predicted_species}")
            print(f"  Confidence      : {confidence:.4f}")

            # Top-K results
            top_k = min(args.top_k, len(result.names))
            top_k_indices = probs.top5[:top_k]
            print(f"\n  Top-{top_k} predictions:")
            for rank, idx in enumerate(top_k_indices, start=1):
                class_name = result.names[idx]
                conf_k = probs.data[idx].item()
                print(f"    {rank}. {class_name:<20} {conf_k:.4f}")

            csv_rows.append({
                "picture_id": picture_id,
                "species":    predicted_species,
                "confidence": f"{confidence:.4f}",
            })
            tree_predictions[tree_id].append((predicted_species, confidence))
        else:
            print("  [WARNING] No classification probabilities found in result.")
            csv_rows.append({
                "picture_id": picture_id,
                "species":    "N/A",
                "confidence": "N/A",
            })

    print("-"*50)

    # ─────────────────────────────────────────
    #  AGGREGATE PER-TREE PREDICTIONS (MAJORITY VOTE)
    # ─────────────────────────────────────────
    #  For each treeID (built from the 4 images belonging to that tree),
    #  the winning species is the one predicted by the most images.
    #  Ties are broken by the highest summed confidence among the tied species.

    tree_rows = []
    for tree_id in sorted(tree_predictions.keys()):
        preds = tree_predictions[tree_id]

        votes = defaultdict(int)
        conf_sum = defaultdict(float)
        for species, confidence in preds:
            votes[species] += 1
            conf_sum[species] += confidence

        # Pick species with most votes; tie-break by highest total confidence
        winning_species = max(votes.keys(), key=lambda s: (votes[s], conf_sum[s]))

        tree_rows.append({
            "treeID": tree_id,
            "predicted_species": winning_species,
        })

    # ─────────────────────────────────────────
    #  SAVE CSV
    # ─────────────────────────────────────────

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["picture_id", "species", "confidence"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n[INFO] CSV saved to: {args.output}  ({len(csv_rows)} rows)")

    # ─────────────────────────────────────────
    #  SAVE PER-TREE CSV
    # ─────────────────────────────────────────

    with open(args.tree_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["treeID", "predicted_species"],
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(tree_rows)

    print(f"[INFO] Tree-level CSV saved to: {args.tree_output}  ({len(tree_rows)} rows)")

    if args.save:
        print(f"[INFO] Annotated results saved to: runs/classify/predict/")

    print("\n[DONE] Prediction complete!\n")


if __name__ == '__main__':
    freeze_support()
    main()
