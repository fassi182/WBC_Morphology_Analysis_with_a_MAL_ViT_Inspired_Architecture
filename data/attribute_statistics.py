# data/attribute_statistics.py

from collections import Counter

from data.encoders import ATTRIBUTE_NAMES


def get_attribute_class_counts(dataset):

    counters = {
        name: Counter()
        for name in ATTRIBUTE_NAMES
    }

    for sample in dataset:

        attributes = sample[
            "attributes"
        ]

        for idx, name in enumerate(
            ATTRIBUTE_NAMES
        ):

            label = int(
                attributes[idx]
            )

            counters[name][label] += 1

    class_counts = {}

    for name in ATTRIBUTE_NAMES:

        max_index = max(
            counters[name].keys()
        )

        class_counts[name] = [
            counters[name].get(
                index,
                0
            )
            for index in range(
                max_index + 1
            )
        ]

    return class_counts