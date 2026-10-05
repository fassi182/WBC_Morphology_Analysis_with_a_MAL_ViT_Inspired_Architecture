"""Runnable example of loading the trained pipeline once in another application."""

import argparse
import json
from pathlib import Path
import sys

# This example works when launched from another current working directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from wbc_predictor import WBCPredictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--attribute-checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    predictor = WBCPredictor(args.checkpoint, device=args.device,
                             attribute_checkpoint_path=args.attribute_checkpoint)
    result = predictor.predict(args.image)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
