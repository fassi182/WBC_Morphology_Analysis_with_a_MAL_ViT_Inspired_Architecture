"""Combine trained stages into one inference checkpoint without retraining."""

import argparse
from pathlib import Path

import torch

from utils.model_loading import load_model


def export_checkpoint(output_path, checkpoint_path=None, attribute_checkpoint_path=None):
    """Save all weights and label metadata; never overwrite an existing file."""
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}. Choose a new filename.")
    model = load_model(checkpoint_path, device="cpu",
                       attribute_checkpoint_path=attribute_checkpoint_path)
    checkpoint = {
        **model.checkpoint_metadata(),
        "checkpoint_kind": "inference_export",
        "source_checkpoints": model.checkpoint_source,
        "model_state_dict": model.state_dict(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as stream:
        torch.save(checkpoint, stream)
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--attribute-checkpoint", type=Path)
    args = parser.parse_args(argv)
    path = export_checkpoint(args.output, args.checkpoint, args.attribute_checkpoint)
    print(f"Exported trained pipeline to {path}. No training was performed.")


if __name__ == "__main__":
    main()
