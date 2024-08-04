# type: ignore
import logging
import os
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
from open_clip import trace_model
from open_clip_train.scheduler import const_lr, const_lr_cooldown, cosine_lr
from torch import optim
from torch.cuda.amp import GradScaler

from mamba_clip.data import ComboLoader, get_combo_loader, get_data, modify_loader
from mamba_clip.eval import evaluate
from mamba_clip.loss import ClipLoss, cross_entropy_loss
from mamba_clip.model import ClipClassifier, init_model
from mamba_clip.train import LATEST_CHECKPOINT_NAME, train_one_epoch
from mamba_clip.utils.dist_utils import broadcast_object, init_device, is_master
from mamba_clip.utils.file_utils import (
    load_checkpoint,
    pt_load,
    remote_sync,
    start_sync_process,
)
from mamba_clip.utils.generic_utils import get_latest_checkpoint, random_seed
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


def pipeline(args):
    device = init_device(args)
    args.lr *= args.world_size

    def setup_paths(args):
        # get the name of the experiments
        if args.stage == 1:
            args.model = (
                args.model_stage_1
                if isinstance(args.model_stage_1, str)
                else args.model_stage_1.__name__
            )
        elif args.stage == 2:
            args.model = (
                args.model_stage_2
                if isinstance(args.model_stage_2, str)
                else args.model_stage_2.__name__
            )
        if args.name is None:
            args.name = create_log_path(args, args.model)
        resume_latest = args.resume == "latest"
        args.log_base_path = os.path.join(args.logs, args.name)
        args.log_path = None
        if is_master(args, local=args.log_local):
            os.makedirs(args.log_base_path, exist_ok=True)
            log_filename = f"out-{args.rank}" if args.log_local else "out.log"
            args.log_path = os.path.join(args.log_base_path, log_filename)
            if os.path.exists(args.log_path) and not resume_latest:
                print(
                    "Error. Experiment already exists. Use --name {} to specify a new experiment."
                )
                return -1

    def setup_train(args, checkpoint_prefix=""):
        # Setup text logger
        args.log_level = logging.DEBUG if args.debug else logging.INFO
        # Setup wandb, tensorboard, checkpoint logging
        args.wandb = "wandb" in args.report_to or "all" in args.report_to
        args.tensorboard = "tensorboard" in args.report_to or "all" in args.report_to
        args.checkpoint_path = os.path.join(args.log_base_path, "checkpoints")
        if is_master(args):
            args.tensorboard_path = (
                os.path.join(args.log_base_path, "tensorboard")
                if args.tensorboard
                else ""
            )
            for dirname in [args.tensorboard_path, args.checkpoint_path]:
                if dirname:
                    os.makedirs(dirname, exist_ok=True)
        else:
            args.tensorboard_path = ""
        resume_latest = args.resume == "latest"
        if resume_latest:
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
                    resume_from = os.path.join(
                        checkpoint_path, f"{checkpoint_prefix}{LATEST_CHECKPOINT_NAME}"
                    )
                    if not os.path.exists(resume_from):
                        # If no latest checkpoint has been saved yet, don't try to resume
                        resume_from = None
                else:
                    # otherwise, list checkpoint dir contents and pick the newest checkpoint
                    resume_from = get_latest_checkpoint(
                        checkpoint_path, remote=args.remote_sync is not None
                    )
                if resume_from:
                    logger.info(f"Found latest resume checkpoint at {resume_from}.")
                else:
                    logger.info(
                        f"No latest resume checkpoint found in {checkpoint_path}."
                    )
            if args.distributed:
                # sync found checkpoint path to all ranks
                resume_from = broadcast_object(args, resume_from)
            args.resume = resume_from

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
                logger.info("remote sync successful.")
            else:
                logger.info("Error: remote sync failed. Exiting.")
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
            logger.warning(
                "It is recommended to use AMP mixed-precision instead of FP16. "
                "FP16 support needs further verification and tuning, especially for train."
            )

        if args.distributed:
            logger.info(
                f"Running in distributed mode with multiple processes. Device: {args.device}."
                f"Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}."
            )
        else:
            logger.info(f"Running with a single process. Device {args.device}.")

    def prepare_params(model, data, args):
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
            logger.info("Model:")
            logger.info(f"{str(model)}")
            logger.info("Params:")
            params_file = os.path.join(args.logs, args.name, "params.txt")
            with open(params_file, "w") as f:
                for name in sorted(vars(args)):
                    val = getattr(args, name)
                    logger.info(f"  {name}: {val}")
                    f.write(f"{name}: {val}\n")

        if args.distributed:
            if args.use_bn_sync:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[device], static_graph=args.ddp_static_graph
            )

        # create optimizer and scaler
        optimizer = None
        scaler = None

        if "train" in data:
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

            optimizer = optim.AdamW(
                [
                    {"params": gain_or_bias_params, "weight_decay": 0.0},
                    {"params": rest_params, "weight_decay": args.wd},
                ],
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                eps=args.eps,
            )
            scaler = GradScaler() if args.precision == "amp" else None

        # optionally resume from a checkpoint
        start_epoch = 0
        if args.resume is not None:
            checkpoint = pt_load(args.resume, map_location="cpu")
            model, optimizer, scaler, start_epoch = load_checkpoint(
                args, checkpoint, model, optimizer, scaler
            )

        # initialize datasets
        assert len(data), "At least one train or eval dataset must be specified."

        # create scheduler if train
        scheduler = None
        if "train" in data and optimizer is not None:
            total_steps = (
                data["train"].dataloader.num_batches // args.accum_freq
            ) * args.epochs
            if args.lr_scheduler == "cosine":
                scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
            elif args.lr_scheduler == "const":
                scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
            elif args.lr_scheduler == "const-cooldown":
                assert (
                    args.epochs_cooldown is not None
                ), "Please specify the number of cooldown epochs for this lr schedule."
                cooldown_steps = (
                    data["train"].dataloader.num_batches // args.accum_freq
                ) * args.epochs_cooldown
                scheduler = const_lr_cooldown(
                    optimizer,
                    args.lr,
                    args.warmup,
                    total_steps,
                    cooldown_steps,
                    args.lr_cooldown_power,
                    args.lr_cooldown_end,
                )
            else:
                logger.error(
                    f"Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown."
                )
                exit(1)

        # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
        args.save_logs = args.logs and args.logs.lower() != "none" and is_master(args)
        writer = None
        if args.save_logs and args.tensorboard:
            assert tensorboard is not None, "Please install tensorboard."
            writer = tensorboard.SummaryWriter(args.tensorboard_path)

        if args.wandb and is_master(args) and args.epochs > 0:
            assert wandb is not None, "Please install wandb."
            logger.debug("Starting wandb.")
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
            logger.debug("Finished loading wandb.")

        # Pytorch 2.0 adds '_orig_mod.' prefix to keys of state_dict() of compiled models.
        # For compatibility, we save state_dict() of the original model, which shares the
        # weights without the prefix.
        original_model = model
        if args.torchcompile:
            logger.info("Compiling model...")
            model = torch.compile(original_model)

            return None

        return {
            "original_model": original_model,
            "model": model,
            "optimizer": optimizer,
            "scaler": scaler,
            "scheduler": scheduler,
            "writer": writer,
            "start_epoch": start_epoch,
        }, args

    def train_loop(
        data,
        loss,
        original_model,
        model,
        optimizer,
        scaler,
        scheduler,
        writer,
        start_epoch,
        args,
        save_prefix,
    ):
        for epoch in range(start_epoch, args.epochs):
            if is_master(args):
                logger.info(f"Start epoch {epoch}")

            if args.sampling is not None:
                num_samples = data["train"].dataloader.num_samples
                data["train"].dataloader, data["train"].sampler = modify_loader(
                    loader=data["train"].dataloader,
                    mode=args.sampling,
                    ep=epoch,
                    n_eps=args.epochs,
                    distributed=args.distributed,
                )
                data["train"].dataloader.num_samples = num_samples
                data["train"].dataloader.num_batches = len(data["train"].dataloader)
            elif args.balanced_mixup and not isinstance(
                data["train"].dataloader, ComboLoader
            ):
                num_samples = data["train"].dataloader.num_samples
                data["train"].dataloader = get_combo_loader(
                    data["train"].dataloader, distributed=args.distributed
                )
                data["train"].dataloader.num_samples = num_samples
                data["train"].dataloader.num_batches = len(data["train"].dataloader)

            train_one_epoch(
                model,
                data,
                loss,
                epoch,
                optimizer,
                scaler,
                scheduler,
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

    if args.stage == 1:
        model_stage_1, preprocess_train, preprocess_val, tokenizer = init_model(
            args.model_stage_1, use_tokenizer=not args.use_visual_only
        )
        data = get_data(args, preprocess_train, preprocess_val, tokenizer)
        model_stage_1.to(device)
        setup_paths(args)
        setup_train(args, checkpoint_prefix=f"stage_{args.stage}_")
        params, args = prepare_params(model_stage_1, data, args)
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

        resume_latest = args.resume == "latest"
        if "train" not in data:
            # If using int8, convert to inference mode.
            if args.use_bnb_linear is not None:
                from open_clip.utils import convert_int8_model_to_inference_mode

                convert_int8_model_to_inference_mode(model_stage_1)
            # Evaluate.
            evaluate(
                model_stage_1,
                data,
                params["start_epoch"],
                args,
                tb_writer=params["writer"],
                tokenizer=tokenizer,
            )
        elif resume_latest and args.epochs == 0:
            logger.info("Resuming latest checkpoint, skipping training.")
            model_stage_1 = params[1]
        else:
            train_loop(
                data,
                loss,
                **params,
                args=args,
                save_prefix=f"stage_{args.stage}_",
            )
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
                args, pt_load(checkpoint_path, map_location="cpu"), model_stage_1
            )
            model_stage_1 = out[0]
        model_stage_2 = ClipClassifier(
            model_stage_1,
            feature_dim=1024
            if not (args.use_visual_only or args.use_text_only or args.use_inner_prod)
            else 512,
            num_classes=args.num_classes,
            use_visual_only=args.use_visual_only,
            use_text_only=args.use_text_only,
            use_inner_prod=args.use_inner_prod,
        )
        model_stage_2.to(device)
        setup_paths(args)
        setup_train(args, checkpoint_prefix=f"stage_{args.stage}_")
        params, args = prepare_params(model_stage_2, data, args)
        if not isinstance(args.class_weighted_loss, bool):
            if not torch.is_tensor(args.class_weighted_loss):
                args.class_weighted_loss = torch.tensor(
                    args.class_weighted_loss, dtype=torch.float32
                ).to(device)
            loss_stage_2 = partial(cross_entropy_loss, weight=args.class_weighted_loss)
        else:
            loss_stage_2 = cross_entropy_loss
        model_stage_2 = train_loop(
            data,
            loss_stage_2,
            **params,
            args=args,
            save_prefix="stage_2_",
        )
    if args.distributed:
        dist.destroy_process_group()
