# type: ignore
import inspect
import os
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import optuna
import ray
import torch
from ray import tune
from ray.air import CheckpointConfig, RunConfig
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.util.joblib import register_ray

from mamba_clip.data import get_data
from mamba_clip.loss import ClipLoss, cross_entropy_loss
from mamba_clip.model import ClipClassifier, init_model
from mamba_clip.pipeline import prepare_params, setup_paths, setup_train, step
from mamba_clip.train import LATEST_CHECKPOINT_NAME
from mamba_clip.utils.dist_utils import (
    is_master,
    world_info_from_env,
)
from mamba_clip.utils.file_utils import (
    load_checkpoint,
    pt_load,
)
from mamba_clip.utils.logging import create_log_path, get_logger

try:
    import wandb
except ImportError:
    wandb = None

try:
    import tensorboard
except ImportError:
    tensorboard = None

logger = get_logger(__name__)


def suggest_config(trial: optuna.Trial, args) -> dict[str, Any]:
    args.lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    args.beta1 = trial.suggest_float("beta1", 0.9, 0.999)
    args.beta2 = trial.suggest_float("beta2", 0.9, 0.999)
    args.eps = trial.suggest_float("eps", 1e-9, 1e-7, log=True)
    args.wd = trial.suggest_float("wd", 1e-4, 1e-1, log=True)
    args.warmup = trial.suggest_float("warmup", 0, 1)
    args.lr_scheduler = "cosine"
    args.lr_restart_interval = trial.suggest_categorical(
        "lr_restart_interval", [1, None]
    )
    args.lr_cooldown_end = trial.suggest_float("lr_cooldown_end", 1e-6, 1e-3, log=True)
    args.lr_cooldown_power = trial.suggest_float("lr_cooldown_power", 0.5, 1.5)
    args.batch_size = trial.suggest_int("batch_size", 8, 128)
    args.accum_freq = 1
    args.grad_clip_norm = trial.suggest_float("grad_clip_norm", 1e-2, 1e2, log=True)
    args.balanced_mixup = trial.suggest_float("balanced_mixup", 0.0, 1.0)
    return asdict(args)


class Trainable(tune.Trainable):
    def setup(self, config):
        from mamba_clip.cli.main import Args

        def get_kwargs(kwargs, method):
            method_args = [
                p.name
                for p in inspect.signature(method).parameters.values()
                if p != p.VAR_KEYWORD
            ]
            return {k: kwargs[k] for k in method_args if k in kwargs}

        config_kwargs = get_kwargs(config, Args.__init__)
        args = Args(**config_kwargs)
        for k, v in config.items():
            setattr(args, k, v)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        tokenizer = None
        if args.stage == 1:
            model_stage_1, preprocess_train, preprocess_val, tokenizer = init_model(
                args.model_stage_1, use_tokenizer=not args.use_visual_only
            )
            data = get_data(args, preprocess_train, preprocess_val, tokenizer)
            model_stage_1.to(device)
            params, args = prepare_params(model_stage_1, data, device, args)
            if tokenizer is not None:
                loss = ClipLoss(
                    local_loss=args.local_loss,
                    gather_with_grad=args.gather_with_grad,
                    cache_labels=True,
                    rank=args.rank,
                    world_size=args.world_size,
                )
            else:
                if args.class_weighted_loss is not None:
                    loss = partial(cross_entropy_loss, weight=args.class_weighted_loss)
                else:
                    loss = cross_entropy_loss
        elif args.stage == 2:
            model_stage_1, preprocess_train, preprocess_val, tokenizer = init_model(
                args.model_stage_1
            )
            data = get_data(args, preprocess_train, preprocess_val, tokenizer)
            if not args.use_original_model:
                # get the latest checkpoint from stage 1
                args.model_stage_1_name = create_log_path(
                    args, args.model_stage_1, latest=True
                )
                checkpoint_path = os.path.join(
                    args.logs,
                    args.model_stage_1_name,
                    "checkpoints",
                    f"stage_1_{LATEST_CHECKPOINT_NAME}",
                )
                out = load_checkpoint(
                    args,
                    pt_load(checkpoint_path, map_location="cpu"),
                    model_stage_1,
                )
                model_stage_1 = out[0]
            model_stage_2 = ClipClassifier(
                model_stage_1,
                feature_dim=1024
                if not (
                    args.use_visual_only or args.use_text_only or args.use_inner_prod
                )
                else 512,
                num_classes=args.num_classes,
                use_visual_only=args.use_visual_only,
                use_text_only=args.use_text_only,
                use_inner_prod=args.use_inner_prod,
            )
            model_stage_2.to(device)
            setup_paths(args)
            setup_train(args, checkpoint_prefix=f"stage_{args.stage}_")
            params, args = prepare_params(model_stage_2, data, device, args)
            if not isinstance(args.class_weighted_loss, bool):
                if not torch.is_tensor(args.class_weighted_loss):
                    args.class_weighted_loss = torch.tensor(
                        args.class_weighted_loss, dtype=torch.float32
                    ).to(device)
                loss = partial(cross_entropy_loss, weight=args.class_weighted_loss)
            else:
                loss = cross_entropy_loss
        self.optimizer = params["optimizer"]
        self.scheduler = params["scheduler"]
        self.scaler = params["scaler"]
        self.scaler = params["scaler"]
        self.original_model = params["original_model"]
        self.model = params["model"]
        self.tokenizer = tokenizer
        self.data = data
        self.loss = loss
        self.start_epoch = params["start_epoch"]
        self.args = args

    def step(self):
        return step(
            data=self.data,
            loss=self.loss,
            model=self.model,
            original_model=self.original_model,
            tokenizer=self.tokenizer,
            optimizer=self.optimizer,
            scaler=self.scaler,
            scheduler=self.scheduler,
            writer=None,
            start_epoch=self.start_epoch,
            args=self.args,
            save_prefix=f"stage_{self.args.stage}_",
        )

    def save_checkpoint(self, checkpoint_dir):
        path = Path(checkpoint_dir) / "checkpoint.pth"
        torch.save(self.model.state_dict(), path)
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))


def ray_tune_pipeline(args):
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    # 1 gpu per trial
    args.distributed = False
    args = setup_paths(args)
    args = setup_train(args, checkpoint_prefix=f"stage_{args.stage}_")
    if args.eval_loss is None:
        args.eval_loss = "val_loss"
        mode = "min"
    elif args.eval_loss in ["partial_auc", "auc", "acc"]:
        mode = "max"
    # if wandb is not None:
    #     wandb.init(
    #         project=args.wandb_project_name,
    #         name=args.name,
    #         id=args.name,
    #         notes=args.wandb_notes,
    #         tags=[],
    #         resume="auto" if args.resume == "latest" else None,
    #         config=vars(args),
    #     )
    if "ip_head" in os.environ and "redis_password" in os.environ:
        ip_head = os.environ["ip_head"]
        redis_password = os.environ["redis_password"]
        if is_master(args):
            logger.info(f"ip head: {ip_head}")
            logger.info(f"redis pwd: {redis_password}")
        _node_ip_addr = ip_head.split(":")[0]
        if is_master(args):
            logger.info(f"node ip addr: {_node_ip_addr}")
        ray.init(
            address=ip_head,
            _redis_password=redis_password,
            _node_ip_address=_node_ip_addr,
        )

        register_ray()

    run_config = RunConfig(
        callbacks=[
            WandbLoggerCallback(project=args.wandb_project_name),
        ],
        sync_config=ray.train.SyncConfig(),
        stop={"training_iteration": args.training_iterations},
        checkpoint_config=CheckpointConfig(checkpoint_at_end=True),
        name=args.name,
    )
    optuna_search = OptunaSearch(
        partial(suggest_config, args=args),
        metric=args.eval_loss,
        mode=mode,
    )
    # if args.resume_from:
    #     logger.info(f"Restoring previous state from {args.resume_from}")
    #     optuna_search.restore_from_dir(args.resume_from)

    workers = 1
    if "SLURM_CPUS_PER_TASK" in os.environ:
        workers = int(os.environ["SLURM_CPUS_PER_TASK"])
    tuner = tune.Tuner(
        tune.with_resources(
            Trainable, {"gpu": 1 if torch.cuda.is_available() else 0, "cpu": workers}
        ),
        tune_config=tune.TuneConfig(
            scheduler=ASHAScheduler(metric=args.eval_loss, mode=mode),
            num_samples=30,
            search_alg=optuna_search,
        ),
        run_config=run_config,
    )
    tuner.fit()
