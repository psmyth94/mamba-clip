# type: ignore
import json
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, roc_curve

from .utils import is_master, get_autocast, get_input_dtype


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
        cumulative_gen_loss = 0.0
        all_image_features, all_neg_text_features, all_pos_text_features = [], [], []
        all_logits = []
        all_targets = []
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                images, neg_texts, pos_texts, targets = batch
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                neg_texts = neg_texts.to(device=device, non_blocking=True)
                pos_texts = pos_texts.to(device=device, non_blocking=True)
                targets = targets.to(device=device, non_blocking=True)
                all_targets.append(targets.cpu())

                with autocast():
                    model_out = model(images, neg_texts, pos_texts)
                    image_features = model_out["image_features"]
                    neg_text_features = model_out["text_features"]
                    pos_text_features = model_out["secondary_text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_neg_text_features.append(neg_text_features.cpu())
                    all_pos_text_features.append(pos_text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_neg_text = (
                        logit_scale * image_features @ neg_text_features.t()
                    ).t()
                    logits_per_pos_text = (
                        logit_scale * image_features @ pos_text_features.t()
                    ).t()
                    # concatenate logits for negative and positive texts
                    logits = torch.cat(
                        [logits_per_neg_text, logits_per_pos_text], dim=1
                    )
                    all_logits.append(logits.cpu())
                    # convert to probabilities
                    probs = F.softmax(logits, dim=1)

                    auc_score = partial_auc(
                        targets.cpu().numpy(), probs[:, 1].cpu().numpy()
                    )

                cumulative_loss += auc_score * args.batch_size
                num_samples += args.batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t"
                    )

            loss = cumulative_loss / num_samples
            all_logits = torch.cat(all_logits, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            overall_auc = partial_auc(
                all_targets.cpu().numpy(), all_logits[:, 1].numpy()
            )
            metrics.update(
                {
                    "overall_auc": overall_auc,
                    "clip_val_loss": loss.item(),
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
