import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .utils import (
    get_autocast,
    get_input_dtype,
    is_master,
    logging,
)

try:
    import wandb
except ImportError:
    wandb = None

logger = logging.get_logger(__name__)

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


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2],
    }


def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def get_model_inputs(
    args,
    images,
    texts,
    targets=None,
    balanced_images=None,
    balanced_texts=None,
    balanced_targets=None,
):
    if args.balanced_mixup:
        lam = np.random.beta(a=args.balanced_mixup, b=1)

        images = (1 - lam) * images + lam * balanced_images
        if lam > 0.5 and texts is not None and balanced_texts is not None:
            texts = balanced_texts

        n_classes = args.num_classes
        targets = F.one_hot(targets, n_classes)
        targets = (1 - lam) * targets + lam * F.one_hot(balanced_targets, n_classes)
    if texts is None:
        input = (images,)
    else:
        input = (images, texts)
    return input


def train_one_epoch(
    model,
    data,
    loss,
    epoch,
    optimizer,
    scaler,
    scheduler,
    args,
    tb_writer=None,
):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)
    model = model.to(device=device)

    model.train()

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

        if args.balanced_mixup:
            if len(batch[0]) == 3:
                images, texts, targets = batch[0][0], batch[0][1], batch[0][2]
            elif len(batch[0]) == 2:
                images, targets = batch[0][0], batch[0][1]
                texts = None

            if len(batch[1]) == 3:
                balanced_images, balanced_texts, balanced_targets = (
                    batch[1][0],
                    batch[1][1],
                    batch[1][2],
                )
            elif len(batch[1]) == 2:
                balanced_images, balanced_targets = batch[1][0], batch[1][1]
                balanced_texts = None
            balanced_images = balanced_images.to(device=device, dtype=input_dtype)
            balanced_targets = balanced_targets.to(device=device)
            balanced_texts = (
                balanced_texts.to(device=device) if balanced_texts is not None else None
            )
        else:
            if len(batch) == 3:
                images, texts, targets = batch
            elif len(batch) == 2:
                images, targets = batch
                texts = None
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = (
            texts.to(device=device, non_blocking=True) if texts is not None else None
        )
        targets = targets.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                input = get_model_inputs(
                    args,
                    images,
                    texts,
                    targets,
                    balanced_images,
                    balanced_texts,
                    balanced_targets,
                )
                if balanced_images is not None:
                    del balanced_images
                if balanced_targets is not None:
                    del balanced_targets
                if balanced_texts is not None:
                    del balanced_texts
                model_out = model(*input)
                if isinstance(model_out, dict) and "logits" in model_out:
                    model_out = {"input": model_out["logits"]}
                elif not isinstance(model_out, dict):
                    model_out = {"input": model_out}
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
                    input = get_model_inputs(
                        args,
                        images,
                        texts,
                        targets,
                        balanced_images,
                        balanced_texts,
                        balanced_targets,
                    )
                    if balanced_images is not None:
                        del balanced_images
                    if balanced_targets is not None:
                        del balanced_targets
                    if balanced_texts is not None:
                        del balanced_texts
                    model_out = model(*input)

                    if isinstance(model_out, dict) and "logits" in model_out:
                        model_out = {"input": model_out["logits"]}
                    elif not isinstance(model_out, dict):
                        model_out = {"input": model_out}

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
                    input = get_model_inputs(
                        args,
                        images,
                        texts,
                        targets,
                        balanced_images,
                        balanced_texts,
                        balanced_targets,
                    )
                    model_out = model(*input)
                    if not isinstance(model_out, dict):
                        model_out = {"input": model_out}

                    inputs_no_accum = {}
                    if "logit_scale" in model_out:
                        inputs_no_accum["logit_scale"] = logit_scale = model_out.pop(
                            "logit_scale", None
                        )
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop(
                            "logit_bias", None
                        )

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(
                            accumulated[:j] + [model_out[key]] + accumulated[j + 1 :]
                        )

                    losses = loss(**model_out, target=targets)

                    del inputs
                    del inputs_no_accum
                    if isinstance(losses, dict):
                        total_loss = sum(losses.values())
                        losses["loss"] = total_loss
                    else:
                        total_loss = losses
                        losses = {"loss": losses}

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

            logit_scale = model_out.get("logit_scale", None)
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

            logger.info(log_info)

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
