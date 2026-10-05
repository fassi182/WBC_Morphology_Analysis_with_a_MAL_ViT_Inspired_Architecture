"""Evaluate the full image -> attributes -> WBC pipeline on held-out images."""

import argparse
from pathlib import Path

from config import DEVICE
from data.dataloader import create_dataloaders
from training.losses import compute_total_loss
from training.test import evaluate_test_set, print_test_results, save_results_json
from utils.model_loading import load_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, help="Optional metrics JSON path")
    args = parser.parse_args(argv)
    _, _, test_loader = create_dataloaders()
    model = load_model(args.checkpoint, device=DEVICE)
    results = evaluate_test_set(model, test_loader, compute_total_loss, DEVICE)
    results["checkpoint_source"] = model.checkpoint_source
    results["concept_representation"] = model.checkpoint_metadata()["concept_representation"]
    print_test_results(results)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_results_json(results, args.output)
    return results


if __name__ == "__main__":
    main()
