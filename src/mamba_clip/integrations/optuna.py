# type: ignore
import copy
import os
from functools import partial
from typing import Any

import joblib
import numpy as np
import torch
import torch.distributed as dist

import optuna
from mamba_clip.data import get_data, get_transform
from mamba_clip.loss import cross_entropy_loss
from mamba_clip.model import VSSM
from mamba_clip.pipeline import prepare_params, setup_paths, setup_train, step
from mamba_clip.utils.dist_utils import (
    broadcast_object,
    init_device,
    is_master,
    world_info_from_env,
)
from mamba_clip.utils.logging import get_logger
from optuna.samplers import TPESampler
from optuna.storages import JournalRedisStorage, RDBStorage
from optuna.study.study import create_study

try:
    from optuna.storages import RedisStorage  # optuna<=3.0.0

    JournalRedisStorage = None
except ImportError:
    try:
        from optuna.storages import (
            JournalRedisStorage,  # optuna>=3.1.0,<4.0.0
            JournalStorage as RedisStorage,
        )

    except ImportError:
        try:
            from optuna.storages.journal import (
                JournalRedisBackend as JournalRedisStorage,
            )
            from optuna.storages import JournalStorage as RedisStorage

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
    return get_data(args, preprocess_train, preprocess_val, tokenizer=None)


def setup(args, data, device):
    model = VSSM(depths=[2, 2, 8, 2], dims=[64, 128, 256, 512], num_classes=2)
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
    if os.environ.get("LOCAL_RANK") is not None:
        device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK')}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    new_args, params = setup(new_args, data, device)

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
    sampler = TPESampler(seed=42, multivariate=True)
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
            data=load_data(args),
            args=args,
        ),
        n_trials=args.training_iterations,
    )

    args = setup_paths(args)
    # save study
    with open(os.path.join(args.log_base_path, "study.joblib"), "wb") as f:
        joblib.dump(study, f)
