# type: ignore
import os
import sys
from dataclasses import asdict, dataclass
from io import BytesIO
from typing import Optional, Union

import h5py
import numpy as np
import pandas as pd
import pandas.api.types
import torch
from mamba_clip.sampler import DistributedWeightedRandomSampler
from mamba_clip.utils import logging
from mamba_clip.utils.data_utils import generate_report_v2
from mamba_clip.utils.dist_utils import is_master
from open_clip import (
    AugmentationCfg,
    SimpleTokenizer,
)
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from open_clip.tokenizer import HFTokenizer
from open_clip.transform import PreprocessCfg
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from timm.data import create_transform
from timm.data.transforms import CenterCropOrPad, ResizeKeepRatio
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms.transforms import InterpolationMode
from transformers.convert_slow_tokenizer import Tokenizer

logger = logging.get_logger(__name__)


def get_transform(
    aug_cfg: Optional[Union[AugmentationCfg, dict]] = None,
    pp_cfg: Optional[PreprocessCfg] = None,
    is_train: bool = False,
):
    if pp_cfg is None:
        pp_cfg = PreprocessCfg()
    interpolation = pp_cfg.interpolation
    image_size = pp_cfg.size
    fill_color = pp_cfg.fill_color

    mean = pp_cfg.mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = pp_cfg.std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    interpolation = interpolation or "bicubic"
    assert interpolation in ["bicubic", "bilinear", "random"]

    interpolation = interpolation or "bicubic"
    assert interpolation in ["bicubic", "bilinear", "random"]
    # NOTE random is ignored for interpolation_mode, so defaults to BICUBIC for inference if set
    interpolation_mode = (
        InterpolationMode.BILINEAR
        if interpolation == "bilinear"
        else InterpolationMode.BICUBIC
    )
    normalize = transforms.Normalize(mean=mean, std=std)
    if is_train:
        if isinstance(aug_cfg, dict):
            aug_cfg = AugmentationCfg(**aug_cfg)
        else:
            aug_cfg = aug_cfg or AugmentationCfg()

        if isinstance(image_size, (tuple, list)):
            assert len(image_size) >= 2
            input_size = (3,) + image_size[-2:]
        else:
            input_size = (3, image_size, image_size)

        aug_cfg_dict = {k: v for k, v in asdict(aug_cfg).items() if v is not None}

        # not appropriate with medical images
        aug_cfg_dict.setdefault("color_jitter", None)
        aug_cfg_dict.pop("color_jitter_prob", None)
        aug_cfg_dict.pop("gray_scale_prob", None)
        aug_cfg_dict.pop("use_timm", None)
        hflip = aug_cfg_dict.get("horizontal_flip", 0.5)

        return create_transform(
            input_size=input_size,
            is_training=True,
            hflip=hflip,
            mean=mean,
            std=std,
            re_mode="pixel",
            interpolation=interpolation,
            **aug_cfg_dict,
        )

    interpolation_mode = "bilinear" if interpolation == "random" else interpolation
    pipe = [
        ResizeKeepRatio(image_size, interpolation=interpolation_mode, longest=1),
        CenterCropOrPad(image_size, fill=fill_color),
        lambda x: x.convert("RGB"),
        transforms.ToTensor(),
        normalize,
    ]
    return transforms.Compose(pipe)


def get_sampling_probabilities(class_count, mode="instance", ep=None, n_eps=None):
    """
    Note that for progressive sampling I use n_eps-1, which I find more intuitive.
    If you are training for 10 epochs, you pass n_eps=10 to this function. Then, inside
    the training loop you would have sth like 'for ep in range(n_eps)', so ep=0,...,9,
    and all fits together.
    """
    if mode == "instance":
        q = 0
    elif mode == "class":
        q = 1
    elif mode == "sqrt":
        q = 0.5  # 1/2
    elif mode == "cbrt":
        q = 0.125  # 1/8
    elif mode == "prog":
        assert (
            ep is not None and n_eps is not None
        ), "progressive sampling requires to pass values for ep and n_eps"
        relative_freq_imbal = class_count**0 / (class_count**0).sum()
        relative_freq_bal = class_count**1 / (class_count**1).sum()
        sampling_probabilities_imbal = relative_freq_imbal ** (-1)
        sampling_probabilities_bal = relative_freq_bal ** (-1)
        return (1 - ep / (n_eps - 1)) * sampling_probabilities_imbal + (
            ep / (n_eps - 1)
        ) * sampling_probabilities_bal
    else:
        sys.exit("not a valid mode")

    relative_freq = class_count**q / (class_count**q).sum()
    sampling_probabilities = relative_freq ** (-1)

    return sampling_probabilities


def modify_loader(loader, mode, ep=None, n_eps=None, distributed=False):
    class_count = np.unique(loader.dataset.targets, return_counts=True)[1]
    sampling_probs = get_sampling_probabilities(
        class_count, mode=mode, ep=ep, n_eps=n_eps
    )
    sample_weights = sampling_probs[loader.dataset.targets]

    if distributed:
        mod_sampler = DistributedWeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights)
        )
    else:
        mod_sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights)
        )
    mod_loader = DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        sampler=mod_sampler,
        num_workers=loader.num_workers,
    )
    return mod_loader, mod_sampler


def get_combo_loader(loader, base_sampling="instance", distributed=False):
    if base_sampling == "instance":
        imbalanced_loader = loader
    else:
        imbalanced_loader, _ = modify_loader(
            loader, mode=base_sampling, distributed=distributed
        )

    balanced_loader, _ = modify_loader(loader, mode="class", distributed=distributed)
    combo_loader = ComboLoader([imbalanced_loader, balanced_loader])
    return combo_loader


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: Optional[Sampler] = None
    shared_epoch: Optional[torch.Tensor] = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)


class ComboIter(object):
    """An iterator."""

    def __init__(self, my_loader):
        self.my_loader = my_loader
        self.loader_iters = [iter(loader) for loader in self.my_loader.loaders]

    def __iter__(self):
        return self

    def __next__(self):
        # When the shortest loader (the one with minimum number of batches)
        # terminates, this iterator will terminates.
        # The `StopIteration` raised inside that shortest loader's `__next__`
        # method will in turn gets out of this `__next__` method.
        batches = [next(loader_iter) for loader_iter in self.loader_iters]
        return self.my_loader.combine_batch(batches)

    def __len__(self):
        return len(self.my_loader)


class ComboLoader(object):
    """This class wraps several pytorch DataLoader objects, allowing each time
    taking a batch from each of them and then combining these several batches
    into one. This class mimics the `for batch in loader:` interface of
    pytorch `DataLoader`.
    Args:
    loaders: a list or tuple of pytorch DataLoader objects
    """

    def __init__(self, loaders):
        self.loaders = loaders
        self.dataset = loaders[0].dataset

    def __iter__(self):
        return ComboIter(self)

    def __len__(self):
        return min([len(loader) for loader in self.loaders])

    # Customize the behavior of combining batches here.
    def combine_batch(self, batches):
        return batches


class IsicChallengeDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        metadata_or_path: str,
        tokenizer: Union[SimpleTokenizer, HFTokenizer, Tokenizer] = None,
        transform: Optional[transforms.Compose] = None,
        is_train: bool = False,
        include_target: bool = False,
        small_test: bool = False,
    ):
        """
        Args:
            data_path (string): Path to the hdf5 file with image data or to the directory
            with images.
            metadata_or_path (string): Path to the csv file with text data or the pandas
            dataframe with text data.
            tokenizer (callable, optional): Optional tokenizer to be applied on the
            text.
            transform (callable, optional): Optional transform to be applied on an image.
            is_train (bool): Whether the dataset is for training or evaluation.
        """
        self.data_path = data_path
        if data_path.endswith(".h5") or data_path.endswith(".hdf5"):
            self.data_path = data_path
        if isinstance(metadata_or_path, str):
            self.metadata_path = metadata_or_path
            self.text_data = pd.read_csv(metadata_or_path).set_index("isic_id")
        else:
            self.text_data = metadata_or_path
            if "isic_id" not in self.text_data.columns:
                self.text_data["isic_id"] = self.text_data.index
            self.text_data = self.text_data.set_index("isic_id")

        self.transform = transform

        self.indices = self.text_data.index
        if "target" in self.text_data.columns:
            self.targets = self.text_data["target"].values.tolist()
        else:
            self.targets = None

        # Open the HDF5 file
        self.hdf5_file = None
        if data_path.endswith(".h5") or data_path.endswith(".hdf5"):
            self.hdf5_file = h5py.File(data_path, "r", libver="latest", swmr=True)
        self.tokenizer = tokenizer

        self.is_train = is_train
        self.small_test = small_test
        self.include_target = include_target

    def __len__(self):
        return len(self.indices)

    def _load_hdf5(self, idx):
        image: bytes = self.hdf5_file[idx][()]

        image_bytes = BytesIO(image)
        image = Image.open(image_bytes)
        if self.transform:
            image = self.transform(image)
        return image

    def _load_image(self, idx):
        if self.data_path.endswith(".h5") or self.data_path.endswith(".hdf5"):
            return self._load_hdf5(idx)
        if isinstance(idx, str):
            image_path = os.path.join(self.data_path, f"{idx}.jpg")
            image = Image.open(image_path)
            if self.transform:
                image = self.transform(image)
            return image

    def _get_text(self, text: str):
        tokens = None
        if isinstance(self.tokenizer, HFTokenizer):
            tokens = self.tokenizer(text)
        elif isinstance(self.tokenizer, SimpleTokenizer):
            tokens = self.tokenizer.encode(text)
        elif isinstance(self.tokenizer, Tokenizer):
            tokens = self.tokenizer.encode(text)
        else:
            return None
        if tokens.shape[0] == 1 and len(tokens.shape) == 2:
            tokens = tokens[0]
        return tokens

    def __getitem__(self, idx):
        ids = self.indices[idx]

        # Get image
        image = self._load_image(ids)

        # Get text
        batch = self.text_data.loc[ids]

        if self.tokenizer is None:
            return image, torch.tensor(batch["target"])
        pos_texts = []
        neg_texts = []

        target_text = []
        if self.is_train:
            target_text.append(
                generate_report_v2(
                    batch,
                    is_eval=False,
                    include_target=self.include_target,
                    shuffle=True,
                    dropout=0.1,
                )
            )

        else:
            texts = generate_report_v2(
                batch, is_eval=True, include_target=self.include_target
            )
            if isinstance(texts, tuple):
                neg_txt, pos_txt = texts
                pos_texts.append(pos_txt)
                neg_texts.append(neg_txt)
                target_text.append(batch["target"])
            else:
                target_text.append(texts)

        targets = None
        if self.targets is not None:
            targets = torch.tensor(self.targets[idx])
        if len(neg_texts) and len(pos_texts):
            return (
                image,
                self._get_text(neg_texts),
                self._get_text(pos_texts),
                torch.tensor(target_text),
            )
        else:
            return (
                image,
                self._get_text(target_text),
                targets,
            )

    def close(self):
        self.hdf5_file.close()


def train_test_split(data, test_size=0.2, random_state=None, stratify=None):
    """
    Splits the data into training and testing sets.

    Parameters:
    data (array-like): The data to split.
    test_size (float or int): If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples.
    random_state (int): Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls.
    stratify (array-like): If not None, data is split in a stratified fashion, using this as the class labels.

    Returns:
    train (array-like): Training set.
    test (array-like): Test set.
    """
    if stratify is not None:
        unique_classes, y_indices = np.unique(stratify, return_inverse=True)
        train_indices = []
        test_indices = []

        for class_index in unique_classes:
            class_mask = y_indices == class_index
            class_data_indices = np.where(class_mask)[0]

            if random_state is not None:
                np.random.seed(random_state)

            np.random.shuffle(class_data_indices)

            if isinstance(test_size, float):
                n_test = int(len(class_data_indices) * test_size)
            else:
                n_test = test_size

            test_indices.extend(class_data_indices[:n_test])
            train_indices.extend(class_data_indices[n_test:])
    else:
        indices = np.arange(len(data))
        if random_state is not None:
            np.random.seed(random_state)

        np.random.shuffle(indices)

        if isinstance(test_size, float):
            n_test = int(len(data) * test_size)
        else:
            n_test = test_size

        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

    train_data = data.loc[train_indices, :]
    test_data = data.loc[test_indices, :]

    return train_data, test_data


def get_data(args, preprocess_train, preprocess_val, tokenizer):
    if is_master(args):
        logger.info("Loading data...")
    # Create dataset
    test_dataset_path = args.data_path + "test-image.hdf5"
    test_metadata_path = args.data_path + "test-metadata.csv"
    if is_master(args):
        logger.info(f"Loading test dataset from {test_dataset_path}")
    test_dataset = IsicChallengeDataset(
        data_path=test_dataset_path,
        metadata_or_path=test_metadata_path,
        tokenizer=tokenizer,
        transform=preprocess_val,
        is_train=False,
    )
    num_test_samples = len(test_dataset)

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.workers,
        drop_last=False,
    )
    test_data_loader.num_samples = num_test_samples
    test_data_loader.num_batches = len(test_data_loader)

    test_data = DataInfo(test_data_loader)

    train_dataset_path = args.data_path + "train-image/image"
    train_metadata_path = args.data_path + "train-metadata.csv"
    train_metadata = pd.read_csv(train_metadata_path)
    # stratify by target
    targets = train_metadata["target"]
    if is_master(args):
        logger.info(f"Loding train dataset from {train_dataset_path}")
        logger.info(f"Stratifying by target: {targets.value_counts()}")

    train_metadata, val_metadata = train_test_split(
        train_metadata, test_size=0.2, stratify=targets, random_state=args.seed
    )
    if args.undersample:

        def select_interesting_samples(tbl, n=None, col=None, sort_by=None):
            if n is None:
                return tbl
            if sort_by is not None and col is not None:
                if sort_by == "asc":
                    tbl = tbl.sort_values(col)
                elif sort_by == "desc":
                    tbl = tbl.sort_values(col, ascending=False)
                elif "/" in sort_by:
                    n_0_p, n_1_p = map(int, sort_by.split("/"))
                    n_0_p = n_0_p / (n_0_p + n_1_p)
                    n_0 = int(n * n_0_p)
                    n_1 = n - n_0
                    tbl = tbl.sort_values(col)
                    tbl_0 = tbl.head(n_0)
                    tbl_1 = tbl.tail(n_1)
                    return pd.concat([tbl_0, tbl_1])
                elif sort_by == "uniform":
                    tbl = tbl.sort_values(col)
                    steps = len(tbl) // args.undersample
                    return tbl.iloc[::steps]
                else:
                    raise ValueError(f"Unknown sort_by value: {sort_by}")
                return tbl.head(n)
            return tbl.sample(n=n, replace=False)

        new_train_metadata = []
        for c in train_metadata["target"].unique():
            tbl = train_metadata[train_metadata["target"] == c]
            n_samples = args.undersample if args.undersample < len(tbl) else None
            new_train_metadata.append(
                select_interesting_samples(
                    tbl,
                    n_samples,
                    col=args.undersample_by,
                    sort_by=args.undersample_sort_by,
                )
            )
        new_train_metadata = pd.concat(new_train_metadata)
        if args.add_remaining_samples:
            index_to_add = list(
                set(list(train_metadata.index)) - set(list(new_train_metadata.index))
            )
            val_metadata = pd.concat([val_metadata, train_metadata.loc[index_to_add]])
        train_metadata = new_train_metadata
    if isinstance(args.class_weighted_loss, bool) and args.class_weighted_loss:
        args.class_weighted_loss = compute_class_weight(
            "balanced", classes=np.unique(targets), y=targets
        )

    train_dataset = IsicChallengeDataset(
        data_path=train_dataset_path,
        metadata_or_path=train_metadata,
        tokenizer=tokenizer,
        transform=preprocess_train,
        is_train=True,
        include_target=args.stage == 1,
        small_test=args.small_test,
    )

    val_dataset = IsicChallengeDataset(
        data_path=train_dataset_path,
        metadata_or_path=val_metadata,
        tokenizer=tokenizer,
        transform=preprocess_val,
        is_train=False,
        include_target=args.stage == 1,
        small_test=args.small_test,
    )

    num_train_samples = len(train_dataset)
    num_val_samples = len(val_dataset)

    train_sampler = None
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
    shuffle = not args.distributed
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=args.workers,
        drop_last=True,
    )
    train_data_loader.num_samples = num_train_samples
    train_data_loader.num_batches = len(train_data_loader)

    train_data = DataInfo(train_data_loader)

    num_val_samples = len(val_dataset)

    val_data_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.workers,
        drop_last=False,
    )
    val_data_loader.num_samples = num_val_samples
    val_data_loader.num_batches = len(val_data_loader)

    val_data = DataInfo(val_data_loader)

    data = {"train": train_data, "val": val_data}
    return data
