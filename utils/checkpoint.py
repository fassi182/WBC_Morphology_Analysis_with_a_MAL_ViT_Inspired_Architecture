"""
checkpoint.py

Save and load MAL-ViT checkpoints.
"""

import torch


def save_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    epoch=0,
    metric=None,
):

    checkpoint = {

        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "metric": metric,
    }

    if optimizer is not None:

        checkpoint[
            "optimizer_state_dict"
        ] = optimizer.state_dict()

    if scheduler is not None:

        checkpoint[
            "scheduler_state_dict"
        ] = scheduler.state_dict()

    torch.save(
        checkpoint,
        path,
    )


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    map_location="cpu",
):

    checkpoint = torch.load(
        path,
        map_location=map_location,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    if (
        optimizer is not None
        and "optimizer_state_dict"
        in checkpoint
    ):

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

    if (
        scheduler is not None
        and "scheduler_state_dict"
        in checkpoint
    ):

        scheduler.load_state_dict(
            checkpoint[
                "scheduler_state_dict"
            ]
        )

    return checkpoint


if __name__ == "__main__":

    print("=" * 60)
    print("Checkpoint Utility Test")
    print("=" * 60)

    model = torch.nn.Linear(
        10,
        2,
    )

    path = "test_checkpoint.pth"

    save_checkpoint(
        path=path,
        model=model,
        epoch=1,
        metric=0.5,
    )

    new_model = torch.nn.Linear(
        10,
        2,
    )

    checkpoint = load_checkpoint(
        path=path,
        model=new_model,
    )

    print(
        "Loaded epoch:",
        checkpoint["epoch"],
    )

    print(
        "Loaded metric:",
        checkpoint["metric"],
    )

    print(
        "\nCheckpoint validation: PASSED"
    )