import math
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, TypeVar

import torch
from torch.utils.data import Dataset, DistributedSampler, Sampler

T_co = TypeVar("T_co", covariant=True)


class BalancedSampler(Sampler[T_co]):
    def __init__(
        self,
        dataset: Dataset,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        class_column: str = "target",
    ) -> None:
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.class_column = class_column
        self.balanced_indices = self._get_class_indices()
        self.total_size = len(self.balanced_indices)

    def _get_class_indices(self) -> List[int]:
        class_indices = defaultdict(list)
        for idx, class_label in zip(self.dataset.indices, self.dataset.targets):
            class_indices[class_label].append(idx)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed)
            for class_label in class_indices:
                indices = class_indices[class_label]
                class_indices[class_label] = torch.randperm(
                    len(indices), generator=g
                ).tolist()

        balanced_indices = []
        min_class_size = min(len(indices) for indices in class_indices.values())

        for class_label in class_indices:
            indices = class_indices[class_label][:min_class_size]
            balanced_indices.extend(indices)
        return balanced_indices

    def __iter__(self) -> Iterator[T_co]:
        balanced_indices = self.balanced_indices
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed)
            balanced_indices = torch.randperm(
                len(balanced_indices), generator=g
            ).tolist()

        if not self.drop_last:
            padding_size = self.total_size - len(balanced_indices)
            if padding_size <= len(balanced_indices):
                balanced_indices += balanced_indices[:padding_size]
            else:
                balanced_indices += (
                    balanced_indices * math.ceil(padding_size / len(balanced_indices))
                )[:padding_size]
        else:
            balanced_indices = balanced_indices[: self.total_size]

        return iter(balanced_indices)

    def __len__(self) -> int:
        return self.total_size


class BalancedDistributedSampler(DistributedSampler[T_co]):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        class_column: str = "target",
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.dataset = dataset
        self.balanced_indices = self._get_class_indices()
        self.total_size = len(self.balanced_indices)
        self.num_samples = self.total_size // self.num_replicas

    def _get_class_indices(self) -> Dict[int, List[int]]:
        class_indices = defaultdict(list)
        for idx, class_label in zip(self.dataset.indices, self.dataset.targets):
            class_indices[class_label].append(idx)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed)
            for class_label in class_indices:
                indices = class_indices[class_label]
                class_indices[class_label] = torch.randperm(
                    len(indices), generator=g
                ).tolist()

        balanced_indices = []
        min_class_size = min(len(indices) for indices in class_indices.values())

        for class_label in class_indices:
            indices = class_indices[class_label][:min_class_size]
            balanced_indices.extend(indices)
        return balanced_indices

    def __iter__(self) -> Iterator[T_co]:
        balanced_indices = self.balanced_indices
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            balanced_indices = torch.randperm(
                len(balanced_indices), generator=g
            ).tolist()

        if not self.drop_last:
            padding_size = self.total_size - len(balanced_indices)
            if padding_size <= len(balanced_indices):
                balanced_indices += balanced_indices[:padding_size]
            else:
                balanced_indices += (
                    balanced_indices * math.ceil(padding_size / len(balanced_indices))
                )[:padding_size]
        else:
            balanced_indices = balanced_indices[: self.total_size]

        indices = balanced_indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples
