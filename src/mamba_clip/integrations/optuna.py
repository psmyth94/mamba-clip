# type: ignore
import copy
import os
from functools import partial
from typing import Any

import joblib
import numpy as np
import optuna
import torch
from optuna.samplers import TPESampler
from optuna.storages import JournalRedisStorage, RDBStorage
from optuna.study.study import create_study
from timm.data import create_transform
from transformers import AutoModel

from mamba_clip.data import get_data, get_metadata, get_transform, undersample_data
from mamba_clip.loss import cross_entropy_loss
from mamba_clip.model import VSSM, MambaVisionClassifier
from mamba_clip.pipeline import (
    init_wandb,
    prepare_params,
    setup_paths,
    setup_train,
    step,
)
from mamba_clip.utils.dist_utils import (
    world_info_from_env,
)
from mamba_clip.utils.generic_utils import random_seed
from mamba_clip.utils.logging import get_logger, logger_setup

try:
    from optuna.storages import RedisStorage  # optuna<=3.0.0

    JournalRedisStorage = None
except ImportError:
    try:
        from optuna.storages import (
            JournalRedisStorage,  # optuna>=3.1.0,<4.0.0
        )
        from optuna.storages import (
            JournalStorage as RedisStorage,
        )

    except ImportError:
        try:
            from optuna.storages import JournalStorage as RedisStorage
            from optuna.storages.journal import (
                JournalRedisBackend as JournalRedisStorage,
            )

        except ImportError:
            RedisStorage = None
            JournalRedisStorage = None

try:
    import wandb
except ImportError:
    wandb = None

try:
    import tensorboard
except ImportError:
    tensorboard = None

logger = get_logger(__name__)


def load_data(args):
    preprocess_train = get_transform(is_train=True)
    preprocess_val = get_transform(is_train=False)
    train_metadata, val_metadata, _ = get_metadata(args)
    return train_metadata, val_metadata, preprocess_train, preprocess_val


def setup(args, data, device):
    if args.model is None or args.model == "VSSM":
        model = VSSM(depths=[2, 2, 8, 2], dims=[64, 128, 256, 512], num_classes=2)
    elif isinstance(args.model, str):
        model = AutoModel.from_pretrained(args.model)
        if "mamba" in args.model.lower():
            model = MambaVisionClassifier(model, num_classes=2)
    else:
        raise ValueError("Model not recognized")

    model = model.to(device)
    params, args = prepare_params(model, data, device, args)
    if isinstance(args.class_weighted_loss, (np.ndarray, list, tuple, torch.Tensor)):
        class_weighted_loss = args.class_weighted_loss
        if not torch.is_tensor(class_weighted_loss):
            class_weighted_loss = torch.tensor(
                class_weighted_loss, dtype=torch.float32
            ).to(device)
        params["loss"] = partial(cross_entropy_loss, weight=class_weighted_loss)
    else:
        params["loss"] = cross_entropy_loss

    return args, params


def optimize(trial: optuna.Trial, data, args) -> dict[str, Any]:
    new_args = copy.deepcopy(args)
    train_metadata, val_metadata, preprocess_train, preprocess_val = data
    new_args.device = (
        "cuda:%d" % args.local_rank if torch.cuda.is_available() else "cpu"
    )
    torch.cuda.set_device(new_args.device)
    device = torch.device(new_args.device)

    new_args.undersample = trial.suggest_int("undersample", 10000, 100000, step=10000)
    train_metadata_trial, val_metadata_trial = undersample_data(
        new_args, train_metadata, val_metadata
    )
    data = get_data(
        new_args,
        train_metadata=train_metadata_trial,
        val_metadata=val_metadata_trial,
        preprocess_train=preprocess_train,
        preprocess_val=preprocess_val,
    )
    new_args.epochs = trial.suggest_int("epochs", 1, 5)
    new_args.lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    new_args.beta1 = trial.suggest_float("beta1", 0.9, 0.999)
    new_args.beta2 = trial.suggest_float("beta2", 0.9, 0.999)
    new_args.eps = trial.suggest_float("eps", 1e-9, 1e-7, log=True)
    new_args.wd = trial.suggest_float("wd", 1e-4, 1e-1, log=True)
    new_args.warmup = trial.suggest_float("warmup", 0, 1)
    new_args.lr_scheduler = "cosine"
    new_args.lr_restart_interval = trial.suggest_categorical(
        "lr_restart_interval", [1, None]
    )
    new_args.batch_size = trial.suggest_categorical(
        "batch_size", [8, 16, 32, 64, 128, 256]
    )
    new_args.accum_freq = 1
    new_args.grad_clip_norm = trial.suggest_float("grad_clip_norm", 1e-2, 1e2, log=True)
    new_args.balanced_mixup = trial.suggest_float("balanced_mixup", 0.0, 1.0)

    new_args = setup_paths(new_args, trial_id=trial.number)
    new_args = setup_train(new_args, checkpoint_prefix=f"stage_{new_args.stage}_")

    params_file = os.path.join(new_args.logs, new_args.name, "params.txt")
    new_args, params = setup(new_args, data, device)
    if (
        hasattr(params["model"], "config")
        and hasattr(params["model"].config, "mean")
        and hasattr(params["model"].config, "std")
        and hasattr(params["model"].config, "crop")
        and hasattr(params["model"].config, "crop_pct")
    ):
        input_resolution = (3, 224, 224)
        preprocess_train = create_transform(
            input_size=input_resolution,
            is_training=True,
            mean=params["model"].config.mean,
            std=params["model"].config.std,
            crop_mode=params["model"].config.crop,
            crop_pct=params["model"].config.crop_pct,
        )
        preprocess_val = create_transform(
            input_size=input_resolution,
            is_training=False,
            mean=params["model"].config.mean,
            std=params["model"].config.std,
            crop_mode=params["model"].config.crop,
            crop_pct=params["model"].config.crop_pct,
        )

    if new_args.wandb:
        init_wandb(new_args, data, params["model"], params_file)

    try:
        metrics = step(
            data=data,
            loss=params["loss"],
            model=params["model"],
            original_model=params["original_model"],
            tokenizer=params.get("tokenizer", None),
            optimizer=params["optimizer"],
            scaler=params["scaler"],
            scheduler=params["scheduler"],
            start_epoch=params["start_epoch"],
            writer=params.get("writer", None),
            args=new_args,
            save_prefix=f"stage_{new_args.stage}_",
        )
    except ValueError as e:
        # check if it was because input contains NaN
        if "input contains nan" in str(e).lower():
            metrics = {
                "train_loss": float("inf"),
                "val_loss": float("inf"),
                "auc": 0,
                "partial_auc": 0,
                "acc": 0,
            }
        else:
            raise e

    # if args.distributed:
    #     dist.barrier()
    #     dist.destroy_process_group()
    del params
    return metrics[new_args.eval_loss]


def optuna_pipeline(args):
    if args.eval_loss is None:
        args.eval_loss = "val_loss"
        mode = "mininimize"
    elif args.eval_loss in ["partial_auc", "auc", "acc"]:
        mode = "maximize"

    args.log_local = True

    args.local_rank, args.rank, args.world_size = world_info_from_env()
    args.world_size = 1
    logger_setup(rank=args.rank, local_rank=args.local_rank)
    metadata = load_data(args)
    # so that the GPUs don't sample the same seed

    args.seed = args.seed + args.rank
    sampler = TPESampler(seed=args.seed, multivariate=True)
    random_seed(args.seed)
    if args.optuna_study_name is not None:
        storage = None
        if args.optuna_storage is not None:
            if args.optuna_storage.startswith("redis"):
                if JournalRedisStorage is not None:
                    storage = RedisStorage(JournalRedisStorage(url=args.optuna_storage))
                else:
                    storage = RedisStorage(url=args.optuna_storage)
            else:
                storage = RDBStorage(url=args.optuna_storage)
        study = create_study(
            direction=mode,
            study_name=args.optuna_study_name,
            sampler=sampler,
            storage=storage,
            load_if_exists=True,
        )
    else:
        study = create_study(
            direction=mode, study_name="AutoTrain", sampler=sampler, load_if_exists=True
        )

    study.optimize(
        lambda trial: optimize(
            trial,
            data=metadata,
            args=args,
        ),
        n_trials=args.training_iterations,
    )

    args = setup_paths(args)
    # save study
    with open(os.path.join(args.log_base_path, "study.joblib"), "wb") as f:
        joblib.dump(study, f)
