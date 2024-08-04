# This module is an adaptation of learning rate schedulers from Open-CLIP
# Original source: https://github.com/mlfoundations/open-clip
# Adapted by: Patrick Smyth
# Date: 2024-08-04
# Modifications include the introduction of warmup restarts for each type of scheduler
import numpy as np


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def const_lr(optimizer, base_lr, warmup_length, total_steps, restart_interval=None):
    def _lr_adjuster(step):
        if restart_interval:
            step_in_cycle = step % restart_interval
        else:
            step_in_cycle = step

        if step_in_cycle < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step_in_cycle)
        else:
            lr = base_lr
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def const_lr_cooldown(
    optimizer,
    base_lr,
    warmup_length,
    total_steps,
    cooldown_steps,
    restart_interval=None,
    cooldown_power=1.0,
    cooldown_end_lr=0.0,
):
    def _lr_adjuster(step):
        if restart_interval:
            step_in_cycle = step % restart_interval
            start_cooldown_step = restart_interval - cooldown_steps
        else:
            step_in_cycle = step
            start_cooldown_step = total_steps - cooldown_steps

        if step_in_cycle < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step_in_cycle)
        else:
            if step_in_cycle < start_cooldown_step:
                lr = base_lr
            else:
                e = step_in_cycle - start_cooldown_step
                es = (
                    restart_interval - start_cooldown_step
                    if restart_interval
                    else total_steps - start_cooldown_step
                )
                decay = (1 - (e / es)) ** cooldown_power
                lr = decay * (base_lr - cooldown_end_lr) + cooldown_end_lr
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def cosine_lr(optimizer, base_lr, warmup_length, total_steps, restart_interval=None):
    def _lr_adjuster(step):
        if restart_interval:
            step_in_cycle = step % restart_interval
        else:
            step_in_cycle = step

        if step_in_cycle < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step_in_cycle)
        else:
            e = step_in_cycle - warmup_length
            es = (
                restart_interval - warmup_length
                if restart_interval
                else total_steps - warmup_length
            )
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster
