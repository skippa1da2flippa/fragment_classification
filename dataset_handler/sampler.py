from collections import defaultdict
import numpy as np
import random
from torch.utils.data import Sampler

def group_indices_by_label(labels):
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)
    return label_to_indices



def create_balanced_batches(labels, batch_size=64, min_per_class=2):
    label_to_indices = group_indices_by_label(labels)
    all_classes = list(label_to_indices.keys())
    batches = []

    while True:
        if any(len(idxs) < min_per_class for idxs in label_to_indices.values()):
            break  # Not enough data to continue

        batch = []
        # Sample k=2 per class
        for cls in all_classes:
            selected = random.sample(label_to_indices[cls], min_per_class)
            batch.extend(selected)
            # Remove them from pool
            for idx in selected:
                label_to_indices[cls].remove(idx)

        # Fill remaining with random indices (any class)
        remaining = batch_size - len(batch)
        available = [idx for idxs in label_to_indices.values() for idx in idxs]
        if len(available) < remaining:
            break  # not enough data to fill the batch

        batch.extend(random.sample(available, remaining))
        batches.append(batch)

    return batches


class FixedBalancedBatchSampler(Sampler):
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


# train_batches = create_balanced_batches(train_dataset.targets, batch_size=64, min_per_class=5)
# train_sampler = FixedBalancedBatchSampler(train_batches)
# train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)