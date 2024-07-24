# type: ignore
import glob
import json
import logging
import math
import multiprocessing
import os
import random
import re
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO, TextIOWrapper
from typing import Any, Dict, Iterator, List, Optional, TypeVar

import fsspec
import h5py
import numpy as np
import pandas as pd
import pandas.api.types
import torch
import torch.distributed as dist
import torch.nn.functional as F
from open_clip import (
    AugmentationCfg,
    CustomTextCLIP,
    SimpleTokenizer,
    create_model_and_transforms,
    create_model_from_pretrained,
    get_tokenizer,
    trace_model,
)
from open_clip.tokenizer import HFTokenizer
from open_clip.transform import PreprocessCfg, image_transform_v2
from open_clip_train.scheduler import const_lr, const_lr_cooldown, cosine_lr
from open_clip_train.train import backward, unwrap_model
from PIL import Image
from sklearn.metrics import auc, roc_curve
from torch import optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from torchvision import transforms
from tqdm.auto import tqdm
from transformers.convert_slow_tokenizer import Tokenizer

try:
    import wandb
except ImportError:
    wandb = None
try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None


LATEST_CHECKPOINT_NAME = "latest.pt"


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


@dataclass
class Args:
    data_path: str
    val_data_path: Optional[str] = None
    train_num_samples: Optional[int] = None
    val_num_samples: Optional[int] = None
    num_classes: int = None
    dataset_type: str = "auto"
    device: str = "auto"
    csv_separator: str = "\t"
    csv_img_key: str = "filepath"
    csv_caption_key: str = "title"
    imagenet_val: Optional[str] = None
    imagenet_v2: Optional[str] = None
    logs: str = "./logs/"
    log_local: bool = False
    name: Optional[str] = None
    workers: int = 4
    batch_size: int = 64
    epochs: int = 3
    epochs_cooldown: Optional[int] = None
    lr: Optional[float] = 1e-4
    beta1: Optional[float] = None
    beta2: Optional[float] = None
    eps: Optional[float] = None
    wd: float = 0.2
    warmup: int = 10000
    use_bn_sync: bool = False
    skip_scheduler: bool = False
    lr_scheduler: str = "cosine"
    lr_cooldown_end: float = 0.0
    lr_cooldown_power: float = 1.0
    save_frequency: int = 1
    save_most_recent: bool = False
    zeroshot_frequency: int = 2
    val_frequency: int = 1
    resume: Optional[str] = None
    precision: str = "amp"
    model: str = "BioMedCLIP-PubMedBERT_256-vit_base_patch16_224"
    pretrained: str = ""
    pretrained_image: bool = False
    lock_image: bool = False
    lock_image_unlocked_groups: int = 0
    lock_image_freeze_bn_stats: bool = False
    image_mean: Optional[List[float]] = None
    image_std: Optional[List[float]] = None
    image_interpolation: Optional[str] = None
    image_resize_mode: Optional[str] = None
    aug_cfg: Optional[List[str]] = field(default_factory=list)
    grad_checkpointing: bool = False
    local_loss: bool = False
    gather_with_grad: bool = False
    force_image_size: Optional[List[int]] = None
    force_quick_gelu: bool = False
    force_patch_dropout: Optional[float] = None
    force_custom_text: bool = False
    torchscript: bool = False
    torchcompile: bool = False
    trace: bool = False
    accum_freq: int = 1
    dist_url: str = "env://"
    dist_backend: str = "nccl"
    report_to: str = ""
    wandb_notes: str = ""
    wandb_project_name: str = "open-clip"
    debug: bool = False
    copy_codebase: bool = False
    ddp_static_graph: bool = False
    no_set_device_rank: bool = False
    seed: int = 0
    grad_clip_norm: Optional[float] = None
    lock_text: bool = False
    lock_text_unlocked_layers: int = 0
    lock_text_freeze_layer_norm: bool = False
    log_every_n_steps: int = 100
    coca_caption_loss_weight: float = 2.0
    coca_contrastive_loss_weight: float = 1.0
    distributed: bool = False
    remote_sync: Optional[str] = None
    remote_sync_frequency: int = 300
    remote_sync_protocol: str = "fsspec"
    delete_previous_checkpoint: bool = False
    distill_model: Optional[str] = None
    distill_pretrained: Optional[str] = None
    use_bnb_linear: Optional[str] = None
    siglip: bool = False


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


class ClipLoss(torch.nn.Module):
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            # use nccl to gather all features
            all_image_features, all_text_features = all_gather(
                image_features,
                text_features,
                self.local_loss,
                self.gather_with_grad,
                self.rank,
                self.world_size,
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = (
                    logit_scale * all_image_features @ all_text_features.T
                )
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        return logits_per_image, logits_per_text

    # def _gather_labels(self, labels):
    #     if self.world_size > 1:
    #         gathered_labels = [torch.zeros_like(labels) for _ in range(self.world_size)]
    #         dist.all_gather(gathered_labels, labels)
    #         if not self.local_loss:
    #             gathered_labels[self.rank] = labels
    #         return torch.cat(gathered_labels, dim=0)
    #     return labels

    def forward(
        self,
        image_features,
        text_features,
        logit_scale,
        output_dict=True,
        target=None,
    ):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(
            image_features, text_features, logit_scale
        )

        if target is not None:
            labels = self._gather_labels(target)
        else:
            labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels)
            + F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class IsicChallengeDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        metadata_or_path: str,
        tokenizer: SimpleTokenizer | HFTokenizer | Tokenizer = None,
        transform: Optional[transforms.Compose] = None,
        is_train: bool = False,
        upsample: bool = None,
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

        self.upsample = upsample if upsample is not None else (is_train or small_test)
        if self.upsample and self.targets is not None:
            self.indices, self.targets = self._get_class_indices(small_test)

        # Open the HDF5 file
        self.hdf5_file = None
        if data_path.endswith(".h5") or data_path.endswith(".hdf5"):
            self.hdf5_file = h5py.File(data_path, "r", libver="latest", swmr=True)
        self.tokenizer = tokenizer

        self.is_train = is_train
        self.small_test = small_test

    def _get_class_indices(self, small_test: bool = False) -> List[int]:
        class_ids = defaultdict(list)
        class_indices = defaultdict(list)
        for idx, (ids, class_label) in enumerate(zip(self.indices, self.targets)):
            class_ids[class_label].append(ids)
            class_indices[class_label].append(idx)

        balanced_ids, balanced_targets = [], []
        max_class_size = max([len(indices) for indices in class_ids.values()])

        for class_label in class_ids:
            indices = class_indices[class_label]
            ids = class_ids[class_label]
            if len(ids) < max_class_size:
                oversampled_indices = np.random.choice(
                    indices, size=max_class_size - len(indices), replace=True
                ).tolist()
                if small_test:
                    oversampled_indices = oversampled_indices[:10]
                else:
                    oversampled_indices = indices + oversampled_indices
                new_ids = [self.indices[idx] for i in oversampled_indices]
                new_targets = [self.targets[idx] for idx in oversampled_indices]
            elif small_test:
                new_ids = ids[:10]
                new_targets = [self.targets[idx] for idx in indices[:10]]
            else:
                new_ids = ids
                new_targets = [self.targets[idx] for idx in indices]
            balanced_ids.extend(new_ids)
            balanced_targets.extend(new_targets)

        return balanced_ids, balanced_targets

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
            raise ValueError("Tokenizer not recognized")
        if tokens.shape[0] == 1 and len(tokens.shape) == 2:
            tokens = tokens[0]
        return tokens

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            ids = [self.indices[i] for i in idx.tolist()]
        else:
            ids = self.indices[idx]

        # Get image
        image = self._load_image(ids)

        # Get text
        batch = self.text_data.loc[ids]

        pos_texts = []
        neg_texts = []

        target_text = []
        if isinstance(ids, list) or torch.is_tensor(ids):
            for _, row in batch.iterrows():
                if self.is_train:
                    target_text.append(
                        generate_report(row, is_eval=False, include_target=False)
                    )
                else:
                    texts = generate_report(row, is_eval=True, include_target=False)
                    if isinstance(texts, tuple):
                        neg_txt, pos_txt = texts
                        pos_texts.append(pos_txt)
                        neg_texts.append(neg_txt)
                        target_text.append(row["target"])
                    else:
                        target_text.append(texts)
        else:
            if self.is_train:
                target_text.append(
                    generate_report(batch, is_eval=False, include_target=False)
                )
            else:
                texts = generate_report(batch, is_eval=True, include_target=False)
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


class ClipModel(torch.nn.Module):
    output_dict = torch.jit.Final[bool]

    def __init__(self, model: CustomTextCLIP):
        super().__init__()
        self.output_dict = True
        self.visual = model.visual
        self.text = model.text
        self.context_length = self.text.context_length
        self.vocab_size = self.text.vocab_size
        self.logit_scale = model.logit_scale
        self.logit_bias = model.logit_bias

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def forward(self, image, text, secondary_text=None) -> dict:
        image_features = (
            self.encode_image(image, normalize=True) if image is not None else None
        )
        text_features = (
            self.encode_text(text, normalize=True) if text is not None else None
        )
        secondary_text_features = None
        if secondary_text is not None:
            secondary_text_features = self.encode_text(secondary_text, normalize=True)

        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp(),
            }
            if secondary_text is not None:
                out_dict["secondary_text_features"] = secondary_text_features
            if self.logit_bias is not None:
                out_dict["logit_bias"] = self.logit_bias
            return out_dict

        out = (image_features, text_features, self.logit_scale.exp())
        if secondary_text is not None:
            out += (secondary_text_features,)
        if self.logit_bias is not None:
            out += (self.logit_bias,)
        return out

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(
            unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats
        )

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        if not unlocked_layers:  # full freezing
            for n, p in self.text.transformer.named_parameters():
                p.requires_grad = (
                    (not freeze_layer_norm) if "LayerNorm" in n.split(".") else False
                )
            return

        encoder = (
            self.text.transformer.encoder
            if hasattr(self.text.transformer, "encoder")
            else self.text.transformer
        )
        layer_list = getattr(encoder, "layer", getattr(encoder, "block", None))
        embeddings = getattr(
            self.text.transformer,
            "embeddings",
            getattr(self.text.transformer, "embed_tokens", None),
        )
        if layer_list is None:
            logging.warning(
                f"Could not find layer list in model of type {self.text.config.model_type}"
            )
            return
        if embeddings is None:
            logging.warning(
                f"Could not find embeddings in model of type {self.text.config.model_type}"
            )
            return
        logging.info(
            f"Unlocking {unlocked_layers}/{len(layer_list) + 1} layers of hf model"
        )
        modules = [embeddings, *layer_list][:-unlocked_layers]
        # freeze layers
        for module in modules:
            for n, p in module.named_parameters():
                p.requires_grad = (
                    (not freeze_layer_norm) if "LayerNorm" in n.split(".") else False
                )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def get_logits(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        image_logits = self.logit_scale.exp() * image_features @ text_features.T
        if self.logit_bias is not None:
            image_logits += self.logit_bias
        text_logits = image_logits.T
        return image_logits, text_logits


class ClipClassifier(torch.nn.Module):
    def __init__(self, clip_model: ClipModel, feature_dim=None, num_classes: int = 2):
        super().__init__()
        self.clip_model = unwrap_model(self.clip_model)
        self.num_classes = num_classes

        # Freeze the CLIP model parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
        # Classification layer
        if feature_dim is None:
            image_feature_dim = getattr(
                self.clip_model.visual,
                "embed_dim",
                getattr(
                    self.clip_model.visual,
                    "output_dim",
                    getattr(self.clip_model.visual, "d_model", None),
                ),
            )
            text_feature_dim = getattr(
                self.clip_model.text,
                "embed_dim",
                getattr(
                    self.clip_model.text,
                    "output_dim",
                    getattr(self.clip_model.text, "d_model", None),
                ),
            )
            if text_feature_dim is None or image_feature_dim is None:
                with torch.no_grad():
                    dummy_image = torch.randn(1, 3, 224, 224)
                    dummy_text = torch.randint(0, 1000, (1, 77))
                    features = self.clip_model(dummy_image, dummy_text)
                    image_feature_dim = features["image_features"].shape[-1]
                    text_feature_dim = features["text_features"].shape[-1]
            feature_dim = image_feature_dim + text_feature_dim

        self.classifier = torch.nn.Linear(feature_dim, num_classes)

    def forward(self, image, text):
        # Get CLIP features
        clip_output = self.clip_model(image, text)
        image_features = clip_output["image_features"]
        text_features = clip_output["text_features"]

        # Concatenate image and text features
        combined_features = torch.cat((image_features, text_features), dim=1)

        # Classification
        logits = self.classifier(combined_features)

        return logits

    def classify(self, image, text):
        logits = self.forward(image, text)
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
        return predicted_class, probabilities


# NCCL UTILS #


def all_gather(
    image_features,
    text_features,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
):
    # We gather tensors from all gpus
    if gather_with_grad:
        all_image_features = torch.cat(dist.nn.all_gather(image_features), dim=0)
        all_text_features = torch.cat(dist.nn.all_gather(text_features), dim=0)
    else:
        gathered_image_features = [
            torch.zeros_like(image_features) for _ in range(world_size)
        ]
        gathered_text_features = [
            torch.zeros_like(text_features) for _ in range(world_size)
        ]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        if not local_loss:
            # ensure grads for local rank when all_* features don't have a gradient
            gathered_image_features[rank] = image_features
            gathered_text_features[rank] = text_features
        all_image_features = torch.cat(gathered_image_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


# DIST UTILS #


def world_info_from_env():
    local_rank = 0
    for v in (
        "LOCAL_RANK",
        "MPI_LOCALRANKID",
        "SLURM_LOCALID",
        "OMPI_COMM_WORLD_LOCAL_RANK",
    ):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ("RANK", "PMI_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ("WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


def init_device(args):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    if torch.cuda.is_available() and (args.device == "auto" or "cuda" in args.device):
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    args.distributed = False
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0
    device = args.device
    if args.device == "auto" or "cuda" in args.device:
        if is_using_distributed():
            if "SLURM_PROCID" in os.environ:
                # DDP via SLURM
                args.local_rank, args.rank, args.world_size = world_info_from_env()
                # SLURM var -> torch.distributed vars in case needed
                os.environ["LOCAL_RANK"] = str(args.local_rank)
                os.environ["RANK"] = str(args.rank)
                os.environ["WORLD_SIZE"] = str(args.world_size)
                torch.distributed.init_process_group(
                    backend=args.dist_backend,
                    init_method=args.dist_url,
                    world_size=args.world_size,
                    rank=args.rank,
                )
            else:
                # DDP via torchrun, torch.distributed.launch
                args.local_rank, _, _ = world_info_from_env()
                torch.distributed.init_process_group(
                    backend=args.dist_backend, init_method=args.dist_url
                )
                args.world_size = torch.distributed.get_world_size()
                args.rank = torch.distributed.get_rank()
            args.distributed = True

        if torch.cuda.is_available():
            if args.distributed and not args.no_set_device_rank:
                device = "cuda:%d" % args.local_rank
            else:
                device = "cuda:0"
            torch.cuda.set_device(device)
        else:
            device = "cpu"
    args.device = device
    device = torch.device(device)
    return device


def broadcast_object(args, obj, src=0):
    # broadcast a pickle-able python object from rank-0 to all ranks
    if args.rank == src:
        objects = [obj]
    else:
        objects = [None]
    dist.broadcast_object_list(objects, src=src)
    return objects[0]


def is_global_master(args):
    return args.rank == 0


def is_local_master(args):
    return args.local_rank == 0


def is_master(args, local=False):
    return is_local_master(args) if local else is_global_master(args)


def is_using_distributed():
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"]) > 1
    if "SLURM_NTASKS" in os.environ:
        return int(os.environ["SLURM_NTASKS"]) > 1
    return False


# LOGGING UTILS #


def setup_logging(log_file, level, include_host=False):
    if include_host:
        import socket

        hostname = socket.gethostname()
        formatter = logging.Formatter(
            f"%(asctime)s |  {hostname} | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d,%H:%M:%S",
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d,%H:%M:%S"
        )

    logging.root.setLevel(level)
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)


# AMP UTILS #


def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        from contextlib import suppress

        return suppress


def get_input_dtype(precision: str):
    input_dtype = None
    if precision in ("bf16", "pure_bf16"):
        input_dtype = torch.bfloat16
    elif precision in ("fp16", "pure_fp16"):
        input_dtype = torch.float16
    return input_dtype


# METRICS UTILS #


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


# ISIC UTILS #


def generate_report(
    row,
    file: Optional[TextIOWrapper] = None,
    is_eval: bool = False,
    include_target=False,
):
    age = row["age_approx"]
    if np.isnan(age):
        age = "unknown"
    sex = row["sex"]
    if sex is None:
        sex = "unknown"
    age_sex_info = ""
    if age != "unknown" and sex != "unknown":
        age_sex_info = f" of {age:.0f} years old {sex}"
    elif age != "unknown":
        age_sex_info = f" of {age:.0f} years old patient"
    elif sex != "unknown":
        age_sex_info = f" of a {sex} patient"

    lesion_size = row["clin_size_long_diam_mm"]
    lesion_size_info = ""
    if not np.isnan(lesion_size):
        lesion_size_info = (
            f"millimeters and a diameter of {lesion_size:.3f} millimeters. "
        )

    area = row["tbp_lv_areaMM2"]
    area_info = ""
    if not np.isnan(area):
        area_info = f"The lesion has an area of {area:.3f} square "

    perimeter = row["tbp_lv_perimeterMM"]
    perimeter_info = ""
    if not np.isnan(perimeter):
        perimeter_info = f"The perimeter of the lesion is {perimeter:.3f} millimeters. "

    eccentricity = row["tbp_lv_eccentricity"]
    eccentricity_info = ""
    if not np.isnan(eccentricity):
        eccentricity_info = (
            f"eccentricity of the lesion, indicating shape irregularity, is "
            f"{eccentricity:.3f}. "
        )

    border_irregularity = row["tbp_lv_norm_border"]
    border_irregularity_info = ""
    if not np.isnan(border_irregularity):
        border_irregularity_info = (
            f"Border irregularity is rated at {border_irregularity:.3f} on a scale of 0 to "
            "10. "
        )

    color_variation = row["tbp_lv_norm_color"]
    color_variation_info = ""
    if not np.isnan(color_variation):
        color_variation_info = (
            f"Color variation within the lesion is rated at {color_variation:.3f} on a "
            "scale of 0 to 10. "
        )

    anatom_site = row["anatom_site_general"]

    report = (
        f"A photo was taken of a skin lesion located on the {anatom_site}{age_sex_info}"
        ". The patient suspects skin cancer. "
        f"{area_info}"
        f"{lesion_size_info}"
        f"{perimeter_info}"
        f"{eccentricity_info}"
        f"{border_irregularity_info}"
        f"{color_variation_info}"
    ).strip()
    if "target" in row and include_target:
        target = "malignant" if row["target"] == 1 else "benign"
        report += (
            f" Based on the image and the description, the lesion is likely {target}."
        )
    if file is not None:
        isic_id = row["isic_id"]
        file.write(f"{isic_id}\t{report}\n")
    if include_target and is_eval:
        report1 = (
            report
            + " Based on the image and the description, the lesion is likely benign."
        )
        report2 = (
            report
            + " Based on the image and the description, the lesion is likely malignant."
        )
        return report1, report2
    return report


def generate_reports(metadata, path):
    with open(path, "w") as file:
        file.write("isic_id\treport\n")
        for _, row in tqdm(metadata.iterrows(), total=metadata.shape[0]):
            generate_report(row, file)


def init_model(aug_kwargs: Optional[Dict[str, Any]] = None):
    if aug_kwargs is None:
        aug_kwargs = {"use_timm": True}
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _ = create_model_from_pretrained(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )  # type: ignore
    tokenizer = get_tokenizer(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    pp_cfg = PreprocessCfg(**model.visual.preprocess_cfg)

    aug_cfg = AugmentationCfg(**aug_kwargs)
    preprocess_train = image_transform_v2(
        pp_cfg,
        is_train=True,
        aug_cfg=aug_cfg,
    )
    preprocess_val = image_transform_v2(
        pp_cfg,
        is_train=False,
    )
    return model, preprocess_train, preprocess_val, tokenizer


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def get_latest_checkpoint(path: str, remote: bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(
            ["aws", "s3", "ls", path + "/"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [
            os.path.join(path, x.split(" ")[-1])
            for x in result.stdout.decode().split("\n")[:-1]
        ]
    else:
        checkpoints = glob.glob(path + "**/*.pt", recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


# FILE UTILS #
def keep_running_remote_sync(sync_every, local_dir, remote_dir, protocol):
    while True:
        time.sleep(sync_every)
        remote_sync(local_dir, remote_dir, protocol)


def pt_save(pt_obj, file_path):
    of = fsspec.open(file_path, "wb")
    with of as f:
        torch.save(pt_obj, file_path)


def pt_load(file_path, map_location=None):
    if file_path.startswith("s3"):
        logging.info("Loading remote checkpoint, which may take a bit.")
    of = fsspec.open(file_path, "rb")
    with of as f:
        out = torch.load(f, map_location=map_location)
    return out


def start_sync_process(sync_every, local_dir, remote_dir, protocol):
    p = multiprocessing.Process(
        target=keep_running_remote_sync,
        args=(sync_every, local_dir, remote_dir, protocol),
    )
    return p


def remote_sync_s3(local_dir, remote_dir):
    # skip epoch_latest which can change during sync.
    result = subprocess.run(
        ["aws", "s3", "sync", local_dir, remote_dir, "--exclude", "*epoch_latest.pt"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        logging.error(
            f"Error: Failed to sync with S3 bucket {result.stderr.decode('utf-8')}"
        )
        return False

    logging.info("Successfully synced with S3 bucket")
    return True


def remote_sync_fsspec(local_dir, remote_dir):
    # FIXME currently this is slow and not recommended. Look into speeding up.
    a = fsspec.get_mapper(local_dir)
    b = fsspec.get_mapper(remote_dir)

    for k in a:
        # skip epoch_latest which can change during sync.
        if "epoch_latest.pt" in k:
            continue

        logging.info(f"Attempting to sync {k}")
        if k in b and len(a[k]) == len(b[k]):
            logging.debug(f"Skipping remote sync for {k}.")
            continue

        try:
            logging.info(f"Successful sync for {k}.")
            b[k] = a[k]
        except Exception as e:
            logging.info(f"Error during remote sync for {k}: {e}")
            return False

    return True


def remote_sync(local_dir, remote_dir, protocol):
    logging.info("Starting remote sync.")
    if protocol == "s3":
        return remote_sync_s3(local_dir, remote_dir)
    elif protocol == "fsspec":
        return remote_sync_fsspec(local_dir, remote_dir)
    else:
        logging.error("Remote protocol not known")
        return False


# END REMOVE SYNC PROCESS #


def partial_auc(y_true, y_pred, min_tpr=0.8):
    # rescale the target. set 0s to 1s and 1s to 0s (since sklearn only has max_fpr)
    v_gt = abs(np.asarray(y_true) - 1)

    # flip the submissions to their compliments
    v_pred = -1.0 * np.asarray(y_pred)

    max_fpr = abs(1 - min_tpr)

    # using sklearn.metric functions: (1) roc_curve and (2) auc
    fpr, tpr, _ = roc_curve(v_gt, v_pred, sample_weight=None)
    if max_fpr is None or max_fpr == 1:
        return auc(fpr, tpr)
    if max_fpr <= 0 or max_fpr > 1:
        raise ValueError("Expected min_tpr in range [0, 1), got: %r" % min_tpr)

    # Add a single point at max_fpr by linear interpolation
    stop = np.searchsorted(fpr, max_fpr, "right")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    partial_auc = auc(fpr, tpr)
    return partial_auc


def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if "val" in data and (
        args.val_frequency
        and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)
    ):
        dataloader = data["val"].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        all_image_features, all_text_features = [], []
        all_probs = []
        all_targets = []
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                images, texts, targets = batch
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                targets = targets.to(device=device, non_blocking=True)
                all_targets.append(targets.cpu())

                with autocast():
                    model_out = model(images, texts)
                    batch_size = images.shape[0]
                    if (
                        isinstance(model_out, dict)
                        and "image_features" in model_out
                        and "text_features" in model_out
                        and "logit_scale" in model_out
                    ):
                        image_features = model_out["image_features"]
                        text_features = model_out["text_features"]
                        logit_scale = model_out["logit_scale"]
                        # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                        # however, system RAM is easily exceeded and compute time becomes problematic
                        all_image_features.append(image_features.cpu())
                        all_text_features.append(text_features.cpu())
                        logit_scale = logit_scale.mean()
                        logits_per_image = (
                            logit_scale * image_features @ text_features.t()
                        )
                        logits_per_text = logits_per_image.t()

                        labels = torch.arange(batch_size, device=device).long()
                        total_loss = (
                            F.cross_entropy(logits_per_image, labels)
                            + F.cross_entropy(logits_per_text, labels)
                        ) / 2
                    else:
                        logits = model_out
                        total_loss = F.cross_entropy(logits, targets)
                        probs = F.softmax(logits, dim=1)
                        all_probs.append(probs.cpu())

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Loss: {cumulative_loss / num_samples:.6f}\t"
                    )

            metrics["val_loss"] = (cumulative_loss / num_samples).item()
            if len(all_probs):
                all_probs = torch.cat(all_probs, dim=0)
                all_targets = torch.cat(all_targets, dim=0)
                p_auc = partial_auc(all_targets.cpu().numpy(), all_probs[:, 1].numpy())
                metrics["partial_auc"] = p_auc
            metrics.update(
                {
                    "epoch": epoch,
                    "num_samples": num_samples,
                }
            )

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, "Please install wandb."
        if "train" in data:
            dataloader = data["train"].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data["epoch"] = epoch
        wandb.log(log_data, step=step)

    return metrics


def train_one_epoch(
    model,
    data,
    loss,
    epoch,
    optimizer,
    scaler,
    scheduler,
    dist_model,
    args,
    tb_writer=None,
):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data["train"].set_epoch(
        epoch
    )  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data["train"].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images, texts, targets = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        targets = targets.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                model_out = model(images, texts)
                logit_scale = None
                if isinstance(model_out, dict):
                    if "logit_scale" in model_out:
                        logit_scale = model_out["logit_scale"]
                else:
                    model_out = {"input": model_out}
                if args.distill:
                    with torch.no_grad():
                        dist_model_out = dist_model(images, texts)
                    model_out.update(
                        {f"dist_{k}": v for k, v in dist_model_out.items()}
                    )
                losses = loss(**model_out, target=targets)

                if isinstance(losses, dict):
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss
                else:
                    total_loss = losses
                    losses = {"loss": losses}
            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop(
                        "logit_scale"
                    )
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(
                            accumulated[:j] + [model_out[key]] + accumulated[j + 1 :]
                        )

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip_norm, norm_type=2.0
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip_norm, norm_type=2.0
                )
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        if hasattr(unwrap_model(model), "logit_scale"):
            with torch.no_grad():
                unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (
            i_accum % args.log_every_n_steps == 0
            or batch_count == num_batches_per_epoch
        ):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item() if logit_scale is not None else None
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = (
                args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            )
            samples_per_second_per_gpu = (
                args.accum_freq * args.batch_size / batch_time_m.val
            )
            log_info = (
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
            )
            if logit_scale_scalar is not None:
                log_info += f"Scale: {logit_scale_scalar:.3f} "
            log_info += loss_log

            logging.info(log_info)

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"],
            }
            log_data.update({name: val.val for name, val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, "Please install wandb."
                log_data["step"] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def train(args):
    device = init_device(args)
    args.lr *= args.world_size
    data, model_stage_1, tokenizer, targets = get_data_model(args)
    model_stage_1.to(device)

    def setup_train(args):
        # get the name of the experiments
        if args.name is None:
            # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
            model_name_safe = args.model.replace("/", "-")
            date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
            if args.distributed:
                # sync date_str from master to all ranks
                date_str = broadcast_object(args, date_str)
            args.name = "-".join(
                [
                    date_str,
                    f"model_{model_name_safe}",
                    f"lr_{args.lr}",
                    f"b_{args.batch_size}",
                    f"j_{args.workers}",
                    f"p_{args.precision}",
                ]
            )

        args._resume_latest = args.resume == "latest"
        log_base_path = os.path.join(args.logs, args.name)
        args.log_path = None
        if is_master(args, local=args.log_local):
            os.makedirs(log_base_path, exist_ok=True)
            log_filename = f"out-{args.rank}" if args.log_local else "out.log"
            args.log_path = os.path.join(log_base_path, log_filename)
            if os.path.exists(args.log_path) and not args._resume_latest:
                print(
                    "Error. Experiment already exists. Use --name {} to specify a new experiment."
                )
                return -1

        # Setup text logger
        args.log_level = logging.DEBUG if args.debug else logging.INFO
        setup_logging(args.log_path, args.log_level)

        # Setup wandb, tensorboard, checkpoint logging
        args.wandb = "wandb" in args.report_to or "all" in args.report_to
        args.tensorboard = "tensorboard" in args.report_to or "all" in args.report_to
        args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
        if is_master(args):
            args.tensorboard_path = (
                os.path.join(log_base_path, "tensorboard") if args.tensorboard else ""
            )
            for dirname in [args.tensorboard_path, args.checkpoint_path]:
                if dirname:
                    os.makedirs(dirname, exist_ok=True)
        else:
            args.tensorboard_path = ""
        if args._resume_latest:
            resume_from = None
            checkpoint_path = args.checkpoint_path
            # If using remote_sync, need to check the remote instead of the local checkpoints folder.
            if args.remote_sync is not None:
                checkpoint_path = os.path.join(
                    args.remote_sync, args.name, "checkpoints"
                )
                if args.save_most_recent:
                    print(
                        "Error. Cannot use save-most-recent with remote_sync and resume latest."
                    )
                    return -1
                if args.remote_sync_protocol != "s3":
                    print(
                        "Error. Sync protocol not supported when using resume latest."
                    )
                    return -1
            if is_master(args):
                # Checking for existing checkpoint via master rank only. It is possible for
                # different rank processes to see different files if a shared file-system is under
                # stress, however it's very difficult to fully work around such situations.
                if args.save_most_recent:
                    # if --save-most-recent flag is set, look for latest at a fixed filename
                    resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                    if not os.path.exists(resume_from):
                        # If no latest checkpoint has been saved yet, don't try to resume
                        resume_from = None
                else:
                    # otherwise, list checkpoint dir contents and pick the newest checkpoint
                    resume_from = get_latest_checkpoint(
                        checkpoint_path, remote=args.remote_sync is not None
                    )
                if resume_from:
                    logging.info(f"Found latest resume checkpoint at {resume_from}.")
                else:
                    logging.info(
                        f"No latest resume checkpoint found in {checkpoint_path}."
                    )
            if args.distributed:
                # sync found checkpoint path to all ranks
                resume_from = broadcast_object(args, resume_from)
            args.resume = resume_from

        if args.copy_codebase:
            copy_codebase(args)

        # start the sync proces if remote-sync is not None
        remote_sync_process = None
        if is_master(args) and args.remote_sync is not None:
            # first make sure it works
            result = remote_sync(
                os.path.join(args.logs, args.name),
                os.path.join(args.remote_sync, args.name),
                args.remote_sync_protocol,
            )
            if result:
                logging.info("remote sync successful.")
            else:
                logging.info("Error: remote sync failed. Exiting.")
                return -1
            # if all looks good, start a process to do this every args.remote_sync_frequency seconds
            remote_sync_process = start_sync_process(
                args.remote_sync_frequency,
                os.path.join(args.logs, args.name),
                os.path.join(args.remote_sync, args.name),
                args.remote_sync_protocol,
            )
            remote_sync_process.start()

        if args.device == "auto":
            args.device = "cuda" if torch.cuda.is_available() else "cpu"

        if args.precision == "fp16":
            logging.warning(
                "It is recommended to use AMP mixed-precision instead of FP16. "
                "FP16 support needs further verification and tuning, especially for train."
            )

        if args.distributed:
            logging.info(
                f"Running in distributed mode with multiple processes. Device: {args.device}."
                f"Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}."
            )
        else:
            logging.info(f"Running with a single process. Device {args.device}.")
        return args

    def prepare_params(model, data, args):
        dist_model = None
        args.distill = (
            args.distill_model is not None and args.distill_pretrained is not None
        )
        if args.distill:
            # FIXME: support distillation with grad accum.
            assert args.accum_freq == 1
            # FIXME: support distillation with coca.
            assert "coca" not in args.model.lower()

        if (
            isinstance(args.force_image_size, (tuple, list))
            and len(args.force_image_size) == 1
        ):
            # arg is nargs, single (square) image size list -> int
            args.force_image_size = args.force_image_size[0]
        random_seed(args.seed, 0)
        model_kwargs = {}
        if args.siglip:
            model_kwargs["init_logit_scale"] = np.log(10)  # different from CLIP
            model_kwargs["init_logit_bias"] = -10

        if args.distill:
            # FIXME: currently assumes the model you're distilling from has the same tokenizer & transforms.
            dist_model, _, _ = create_model_and_transforms(
                args.distill_model,
                args.distill_pretrained,
                device=device,
                precision=args.precision,
                output_dict=True,
            )
        if args.use_bnb_linear is not None:
            print(
                "=> using a layer from bitsandbytes.\n"
                "   this is an experimental feature which requires two extra pip installs\n"
                "   pip install bitsandbytes triton"
                "   please make sure to use triton 2.0.0"
            )
            import bitsandbytes as bnb
            from open_clip.utils import replace_linear

            print(f"=> replacing linear layers with {args.use_bnb_linear}")
            linear_replacement_cls = getattr(
                bnb.nn.triton_based_modules, args.use_bnb_linear
            )
            replace_linear(model, linear_replacement_cls)
            model = model.to(device)

        random_seed(args.seed, args.rank)

        if args.trace:
            model = trace_model(model, batch_size=args.batch_size, device=device)

        if args.lock_image and hasattr(model, "lock_image_tower"):
            # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
            model.lock_image_tower(
                unlocked_groups=args.lock_image_unlocked_groups,
                freeze_bn_stats=args.lock_image_freeze_bn_stats,
            )
        if args.lock_text and hasattr(model, "lock_text_tower"):
            model.lock_text_tower(
                unlocked_layers=args.lock_text_unlocked_layers,
                freeze_layer_norm=args.lock_text_freeze_layer_norm,
            )

        if args.grad_checkpointing:
            model.set_grad_checkpointing()

        if is_master(args):
            logging.info("Model:")
            logging.info(f"{str(model)}")
            logging.info("Params:")
            params_file = os.path.join(args.logs, args.name, "params.txt")
            with open(params_file, "w") as f:
                for name in sorted(vars(args)):
                    val = getattr(args, name)
                    logging.info(f"  {name}: {val}")
                    f.write(f"{name}: {val}\n")

        if args.distributed:
            if args.use_bn_sync:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[device], static_graph=args.ddp_static_graph
            )

            if args.distill:
                dist_model = torch.nn.parallel.DistributedDataParallel(
                    dist_model, device_ids=[device], static_graph=args.ddp_static_graph
                )

        # create optimizer and scaler
        optimizer_stage_1 = None
        scaler_stage_1 = None

        if "train" in data or args.dataset_type == "synthetic":
            assert not args.trace, "Cannot train with traced model"

            def exclude(n, p):
                return (
                    p.ndim < 2
                    or "bn" in n
                    or "ln" in n
                    or "bias" in n
                    or "logit_scale" in n
                )

            def include(n, p):
                return not exclude(n, p)

            named_parameters = list(model.named_parameters())
            gain_or_bias_params = [
                p for n, p in named_parameters if exclude(n, p) and p.requires_grad
            ]
            rest_params = [
                p for n, p in named_parameters if include(n, p) and p.requires_grad
            ]

            optimizer_stage_1 = optim.AdamW(
                [
                    {"params": gain_or_bias_params, "weight_decay": 0.0},
                    {"params": rest_params, "weight_decay": args.wd},
                ],
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                eps=args.eps,
            )
            scaler_stage_1 = GradScaler() if args.precision == "amp" else None

        # optionally resume from a checkpoint
        start_epoch = 0
        if args.resume is not None:
            checkpoint = pt_load(args.resume, map_location="cpu")
            if "epoch" in checkpoint:
                # resuming a train checkpoint w/ epoch and optimizer state
                start_epoch = checkpoint["epoch"]
                sd = checkpoint["state_dict"]
                if not args.distributed and next(iter(sd.items()))[0].startswith(
                    "module"
                ):
                    sd = {k[len("module.") :]: v for k, v in sd.items()}
                model.load_state_dict(sd)
                if optimizer_stage_1 is not None:
                    optimizer_stage_1.load_state_dict(checkpoint["optimizer"])
                if scaler_stage_1 is not None and "scaler" in checkpoint:
                    scaler_stage_1.load_state_dict(checkpoint["scaler"])
                logging.info(
                    f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})"
                )
            else:
                # loading a bare (model only) checkpoint for fine-tune or evaluation
                model.load_state_dict(checkpoint)
                logging.info(
                    f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})"
                )

        # initialize datasets
        assert len(data), "At least one train or eval dataset must be specified."

        # create scheduler if train
        scheduler_stage_1 = None
        if "train" in data and optimizer_stage_1 is not None:
            total_steps = (
                data["train"].dataloader.num_batches // args.accum_freq
            ) * args.epochs
            if args.lr_scheduler == "cosine":
                scheduler_stage_1 = cosine_lr(
                    optimizer_stage_1, args.lr, args.warmup, total_steps
                )
            elif args.lr_scheduler == "const":
                scheduler_stage_1 = const_lr(
                    optimizer_stage_1, args.lr, args.warmup, total_steps
                )
            elif args.lr_scheduler == "const-cooldown":
                assert (
                    args.epochs_cooldown is not None
                ), "Please specify the number of cooldown epochs for this lr schedule."
                cooldown_steps = (
                    data["train"].dataloader.num_batches // args.accum_freq
                ) * args.epochs_cooldown
                scheduler_stage_1 = const_lr_cooldown(
                    optimizer_stage_1,
                    args.lr,
                    args.warmup,
                    total_steps,
                    cooldown_steps,
                    args.lr_cooldown_power,
                    args.lr_cooldown_end,
                )
            else:
                logging.error(
                    f"Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown."
                )
                exit(1)

        # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
        args.save_logs = args.logs and args.logs.lower() != "none" and is_master(args)
        writer = None
        if args.save_logs and args.tensorboard:
            assert tensorboard is not None, "Please install tensorboard."
            writer = tensorboard.SummaryWriter(args.tensorboard_path)

        if args.wandb and is_master(args):
            assert wandb is not None, "Please install wandb."
            logging.debug("Starting wandb.")
            args.train_sz = data["train"].dataloader.num_samples
            if args.val_data_path is not None or "val" in data:
                args.val_sz = data["val"].dataloader.num_samples
            # you will have to configure this for your project!
            wandb.init(
                project=args.wandb_project_name,
                name=args.name,
                id=args.name,
                notes=args.wandb_notes,
                tags=[],
                resume="auto" if args.resume == "latest" else None,
                config=vars(args),
            )
            if args.debug:
                wandb.watch(model, log="all")
            wandb.save(params_file)
            logging.debug("Finished loading wandb.")

        # Pytorch 2.0 adds '_orig_mod.' prefix to keys of state_dict() of compiled models.
        # For compatibility, we save state_dict() of the original model, which shares the
        # weights without the prefix.
        original_model = model
        if args.torchcompile:
            logging.info("Compiling model...")
            model = torch.compile(original_model)

        if "train" not in data:
            # If using int8, convert to inference mode.
            if args.use_bnb_linear is not None:
                from open_clip.utils import convert_int8_model_to_inference_mode

                convert_int8_model_to_inference_mode(model)
            # Evaluate.
            evaluate(
                model,
                data,
                start_epoch,
                args,
                tb_writer=writer,
                tokenizer=tokenizer,
            )
            return None

        return (
            original_model,
            model,
            optimizer_stage_1,
            scaler_stage_1,
            scheduler_stage_1,
            dist_model,
            writer,
            start_epoch,
            args,
        )

    def train_loop(
        loss,
        original_model,
        model,
        optimizer,
        scaler,
        scheduler,
        dist_model,
        writer,
        start_epoch,
        args,
        save_prefix,
    ):
        for epoch in range(start_epoch, args.epochs):
            if is_master(args):
                logging.info(f"Start epoch {epoch}")

            train_one_epoch(
                model,
                data,
                loss,
                epoch,
                optimizer,
                scaler,
                scheduler,
                dist_model,
                args,
                tb_writer=writer,
            )
            completed_epoch = epoch + 1

            if any(v in data for v in ("val", "imagenet-val", "imagenet-v2")):
                evaluate(
                    model,
                    data,
                    completed_epoch,
                    args,
                    tb_writer=writer,
                    tokenizer=tokenizer,
                )

            # Saving checkpoints.
            if args.save_logs:
                checkpoint_dict = {
                    "epoch": completed_epoch,
                    "name": args.name,
                    "state_dict": original_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                if scaler is not None:
                    checkpoint_dict["scaler"] = scaler.state_dict()

                if completed_epoch == args.epochs or (
                    args.save_frequency > 0
                    and (completed_epoch % args.save_frequency) == 0
                ):
                    torch.save(
                        checkpoint_dict,
                        os.path.join(
                            args.checkpoint_path,
                            f"{save_prefix}epoch_{completed_epoch}.pt",
                        ),
                    )
                if args.delete_previous_checkpoint:
                    previous_checkpoint = os.path.join(
                        args.checkpoint_path,
                        f"{save_prefix}epoch_{completed_epoch - 1}.pt",
                    )
                    if os.path.exists(previous_checkpoint):
                        os.remove(previous_checkpoint)

                if args.save_most_recent:
                    # try not to corrupt the latest checkpoint if save fails
                    tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
                    latest_save_path = os.path.join(
                        args.checkpoint_path, f"{save_prefix}{LATEST_CHECKPOINT_NAME}"
                    )
                    torch.save(checkpoint_dict, tmp_save_path)
                    os.replace(tmp_save_path, latest_save_path)
        if args.wandb and is_master(args):
            wandb.finish()
        return model

    args = setup_train(args)
    params = prepare_params(model_stage_1, data, args)
    if params is not None:
        loss_stage_1 = ClipLoss(
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
        )

        model_stage_1 = train_loop(
            loss_stage_1,
            *params,
            "stage_1_",
        )
        if args.num_classes:
            model_stage_2 = ClipClassifier(
                model_stage_1,
                num_classes=args.num_classes,
            )
            model_stage_2.to(device)
            params = prepare_params(model_stage_2, data, args)
            loss_stage_2 = F.cross_entropy
            args.epochs = 50
            model_stage_2 = train_loop(
                loss_stage_2,
                *params,
                "stage_2_",
            )


def copy_codebase(args):
    from shutil import copytree, ignore_patterns

    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(
        current_code_path, new_code_path, ignore=ignore_patterns("log", "logs", "wandb")
    )
    print("Done copying code.")
    return 1


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


def get_data_model(args):
    clip_model, preprocess_train, preprocess_val, tokenizer = init_model()
    model = ClipModel(clip_model)

    if is_master(args):
        logging.info("Loading data...")
    # Create dataset
    test_dataset_path = args.data_path + "test-image.hdf5"
    test_metadata_path = args.data_path + "test-metadata.csv"
    if is_master(args):
        logging.info(f"Loading test dataset from {test_dataset_path}")
    test_dataset = IsicChallengeDataset(
        data_path=test_dataset_path,
        metadata_or_path=test_metadata_path,
        tokenizer=tokenizer,
        transform=preprocess_val,
        # is_train=False,
        is_train=True,
    )
    num_test_samples = len(test_dataset)

    # test_sampler = BalancedSampler(test_dataset)
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        # sampler=test_sampler,
        pin_memory=True,
        num_workers=args.workers,
        drop_last=False,
        # drop_last=True,
    )
    # test_data_loader.num_samples = len(test_sampler)
    test_data_loader.num_samples = num_test_samples
    test_data_loader.num_batches = len(test_data_loader)

    test_data = DataInfo(test_data_loader)
    # test_data.sampler = test_sampler

    # fake_test_dataset = IsicChallengeDataset(
    #     data_path=test_dataset_path,
    #     metadata_or_path=test_metadata_path,
    #     tokenizer=tokenizer,
    #     transform=preprocess_val,
    #     # is_train=False,
    #     is_train=False,
    # )
    # fake_val_data_loader = DataLoader(
    #     fake_test_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     pin_memory=True,
    #     num_workers=args.workers,
    #     drop_last=False,
    # )
    # fake_val_data_loader.num_samples = num_test_samples
    # fake_val_data_loader.num_batches = len(fake_val_data_loader)
    #
    # fake_val_data = DataInfo(fake_val_data_loader)

    train_dataset_path = args.data_path + "train-image/image"
    train_metadata_path = args.data_path + "train-metadata.csv"
    train_metadata = pd.read_csv(train_metadata_path)
    # stratify by target
    targets = train_metadata["target"]
    if is_master(args):
        logging.info(f"Loding train dataset from {train_dataset_path}")
        logging.info(f"Stratifying by target: {targets.value_counts()}")
    train_metadata, val_metadata = train_test_split(
        train_metadata, test_size=0.2, stratify=targets
    )

    train_dataset = IsicChallengeDataset(
        data_path=train_dataset_path,
        metadata_or_path=train_metadata,
        tokenizer=tokenizer,
        transform=preprocess_train,
        is_train=True,
        # small_test=True,
    )

    val_dataset = IsicChallengeDataset(
        data_path=train_dataset_path,
        metadata_or_path=val_metadata,
        tokenizer=tokenizer,
        transform=preprocess_val,
        is_train=False,
        # small_test=True,
    )

    num_train_samples = len(train_dataset)
    num_val_samples = len(val_dataset)

    train_sampler = DistributedSampler(train_dataset) if args.distributed else None
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

    train_data = DataInfo(train_data_loader, sampler=train_sampler)

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
    return data, model, tokenizer, train_dataset.targets


# %%
if __name__ == "__main__":
    args = Args(
        num_classes=2,
        lr=1e-6,
        eps=1e-8,
        beta1=0.9,
        beta2=0.999,
        wd=0.05,
        epochs=3,
        report_to="wandb",
        batch_size=128,
        # device="cpu",
        save_most_recent=True,
        workers=4,
        data_path="data/isic-2024-challenge/",
        lock_image=True,
        lock_image_unlocked_groups=1,
        lock_text=True,
        lock_text_unlocked_layers=1,
    )

    # pre-training
    train(args)
    # %%
