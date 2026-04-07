"""
train_all_peak.py — Retrain All 6 Models at Their Peak Configuration
======================================================================
Runs train.py for each model sequentially with per-model-optimised
hyperparameters (see PER_MODEL_CONFIGS in train.py).  All checkpoints
are overwritten in data/processed/checkpoints/.

After this script finishes, run comprehensive_benchmark.py to compare
all models at their peak performance.

Usage (from repo root):
    python src/group11/training/train_all_peak.py
    python src/group11/training/train_all_peak.py --models gcn gat mpnn
"""

import argparse
import subprocess
import sys
import time

MODELS = ["gcn", "gat", "mpnn", "custom", "routenet", "tgnn"]


def run_training(model: str) -> bool:
    """Train a single model. Returns True on success."""
    print("\n" + "=" * 70)
    print(f"  TRAINING: {model.upper()}")
    print("=" * 70 + "\n")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "src/group11/training/train.py", "--model", model],
        check=False,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[ERROR] {model} training failed (exit {result.returncode}).")
        return False
    print(f"\n[OK] {model} finished in {elapsed/60:.1f} min.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Retrain all models at peak config.")
    parser.add_argument(
        "--models", nargs="+", choices=MODELS, default=MODELS,
        help="Subset of models to train (default: all 6).",
    )
    args = parser.parse_args()

    total_start = time.time()
    results = {}

    for model in args.models:
        results[model] = run_training(model)

    # Summary
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print("  TRAINING SUMMARY")
    print("=" * 70)
    for model in args.models:
        status = "OK" if results[model] else "FAILED"
        print(f"  {model:<12} {status}")
    print(f"\n  Total time: {total_elapsed/60:.1f} min")
    print("=" * 70)

    failed = [m for m, ok in results.items() if not ok]
    if failed:
        print(f"\nFailed models: {failed}")
        print("Fix errors above and re-run with: --models " + " ".join(failed))
        sys.exit(1)
    else:
        print("\nAll models trained successfully.")
        print("\nNext step — run the benchmark:")
        print("  python src/group11/training/comprehensive_benchmark.py")


if __name__ == "__main__":
    main()
