"""
Average Meter
-------------

Tracks running averages of training metrics.
"""


class AverageMeter:

    def __init__(self):

        self.reset()

    def reset(self):

        self.value = 0.0
        self.total = 0.0
        self.count = 0
        self.average = 0.0

    def update(
        self,
        value,
        n=1,
    ):

        self.value = float(value)

        self.total += (
            float(value) * n
        )

        self.count += n

        self.average = (
            self.total / self.count
        )

    def __str__(self):

        return (
            f"{self.average:.4f}"
        )


if __name__ == "__main__":

    print("=" * 60)
    print("Average Meter Test")
    print("=" * 60)

    meter = AverageMeter()

    meter.update(2.0)
    meter.update(4.0)
    meter.update(6.0)

    print(
        "\nCurrent value:",
        meter.value,
    )

    print(
        "Count:",
        meter.count,
    )

    print(
        "Average:",
        meter.average,
    )

    assert meter.average == 4.0

    print(
        "\nAverage meter validation: PASSED"
    )