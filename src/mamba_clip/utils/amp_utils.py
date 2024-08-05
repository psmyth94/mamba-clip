# This module is an adaptation of amp utils from Open-CLIP
# Source: https://github.com/mlfoundations/open-clip
# Adapted by: Patrick Smyth
# Date: 2024-08-04
# Modifications include removal deprecated functions and renaming of functions
from contextlib import suppress
from typing import Callable, Optional, Union

import torch

# Constants for precision types
PRECISION_AMP = "amp"
PRECISION_AMP_BFLOAT16 = "amp_bfloat16"
PRECISION_AMP_BF16 = "amp_bf16"
PRECISION_BFLOAT16_OPTIONS = {"bf16", "pure_bf16"}
PRECISION_FLOAT16_OPTIONS = {"fp16", "pure_fp16"}


def get_autocast(precision: str) -> Union[Callable, suppress]:
    """
    Returns the appropriate autocast context manager based on the given precision.

    Parameters:
    precision (str): Precision type. Options are "amp", "amp_bfloat16", "amp_bf16".

    Returns:
    Callable or contextlib.suppress: Autocast context manager or suppress if precision is not matched.
    """
    if precision == PRECISION_AMP:
        return torch.cuda.amp.autocast
    elif precision in {PRECISION_AMP_BFLOAT16, PRECISION_AMP_BF16}:
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress


def get_input_dtype(precision: str) -> Optional[torch.dtype]:
    """
    Returns the appropriate input data type based on the given precision.

    Parameters:
    precision (str): Precision type. Options are "bf16", "pure_bf16", "fp16", "pure_fp16".

    Returns:
    Optional[torch.dtype]: Corresponding torch data type or None if precision is not matched.
    """
    if precision in PRECISION_BFLOAT16_OPTIONS:
        return torch.bfloat16
    elif precision in PRECISION_FLOAT16_OPTIONS:
        return torch.float16
    return None
