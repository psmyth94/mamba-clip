# type: ignore
import os
import joblib
import copy
from functools import partial
from typing import Any

import numpy as np
import optuna
import torch

from mamba_clip.data import get_data, get_transform
from mamba_clip.loss import cross_entropy_loss
from mamba_clip.model import VSSM
from mamba_clip.pipeline import prepare_params, setup_paths, setup_train, step
from mamba_clip.utils.dist_utils import init_device
from mamba_clip.utils.logging import get_logger

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
    preprocess_train = get_transform({}, {}, is_train=True)
    preprocess_val = get_transform({}, {}, is_train=False)
    return get_data(args, preprocess_train, preprocess_val)


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
    device = init_device(new_args)
    new_args = setup_paths(new_args)
    new_args = setup_train(new_args, checkpoint_prefix=f"stage_{new_args.stage}_")
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
    new_args, params = setup(new_args, data, device)

    metrics = step(
        data=data,
        loss=params["loss"],
        model=params["model"],
        original_model=params["original_model"],
        tokenizer=params["tokenizer"],
        optimizer=params["optimizer"],
        scaler=params["scaler"],
        scheduler=params["scheduler"],
        start_epoch=params["start_epoch"],
        args=new_args,
        save_prefix=f"stage_{new_args.stage}_",
    )
    if args.distributed:
        # Cleanup dist
        torch.distributed.destroy_process_group()
    del params
    return metrics[new_args.eval_loss]


def optuna_pipeline(args):
    from optuna.samplers import TPESampler
    from optuna.study.study import create_study

    if args.eval_loss is None:
        args.eval_loss = "val_loss"
        mode = "min"
    elif args.eval_loss in ["partial_auc", "auc", "acc"]:
        mode = "max"

    sampler = TPESampler(seed=42, multivariate=True)
    study = create_study(direction=mode, study_name="AutoTrain", sampler=sampler)

    study.optimize(
        lambda trial: optimize(
            trial,
            data=load_data(args),
        ),
        n_trials=args.training_iterations,
    )

    args = setup_paths(args)
    # save study
    with open(os.path.join(args.log_base_path, "study.joblib"), "wb") as f:
        joblib.dump(study, f)
