import multiprocessing
import time
from contextlib import closing
from typing import Optional, Union

import fsspec
import torch

from .logging import get_logger

logger = get_logger(__name__)


def keep_running_remote_sync(
    sync_every: int, local_dir: str, remote_dir: str, protocol: str
) -> None:
    """
    Periodically syncs the local directory with the remote directory.

    Parameters:
    sync_every (int): Time interval (in seconds) between syncs.
    local_dir (str): Local directory path.
    remote_dir (str): Remote directory path.
    protocol (str): Protocol for syncing (e.g., 's3', 'fsspec').
    """
    while True:
        time.sleep(sync_every)
        remote_sync(local_dir, remote_dir, protocol)


def pt_save(pt_obj: torch.nn.Module, file_path: str) -> None:
    """
    Save a PyTorch object to a specified file path.

    Parameters:
    pt_obj (torch.nn.Module): The PyTorch object to save.
    file_path (str): The file path to save the object to.
    """
    with closing(fsspec.open(file_path, "wb")) as of:
        torch.save(pt_obj, file_path)


def pt_load(
    file_path: str, map_location: Optional[Union[str, torch.device]] = None
) -> torch.nn.Module:
    """
    Load a PyTorch object from a specified file path.

    Parameters:
    file_path (str): The file path to load the object from.
    map_location (Optional[Union[str, torch.device]]): The location to map the loaded object to.

    Returns:
    torch.nn.Module: The loaded PyTorch object.
    """
    if file_path.startswith("s3"):
        logger.info("Loading remote checkpoint, which may take a bit.")
    with fsspec.open(file_path, "rb") as of:
        return torch.load(of, map_location=map_location)


def start_sync_process(
    sync_every: int, local_dir: str, remote_dir: str, protocol: str
) -> multiprocessing.Process:
    """
    Start a new process to periodically sync the local directory with the remote directory.

    Parameters:
    sync_every (int): Time interval (in seconds) between syncs.
    local_dir (str): Local directory path.
    remote_dir (str): Remote directory path.
    protocol (str): Protocol for syncing (e.g., 's3', 'fsspec').

    Returns:
    multiprocessing.Process: The started process.
    """
    return multiprocessing.Process(
        target=keep_running_remote_sync,
        args=(sync_every, local_dir, remote_dir, protocol),
    )


def load_checkpoint(args, checkpoint, model, optimizer=None, scaler=None):
    if "epoch" in checkpoint:
        # resuming a train checkpoint w/ epoch and optimizer state
        start_epoch = checkpoint["epoch"]
        sd = checkpoint["state_dict"]
        if not args.distributed and next(iter(sd.items()))[0].startswith("module"):
            sd = {k[len("module.") :]: v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scaler is not None and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        logger.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
    else:
        # loading a bare (model only) checkpoint for fine-tune or evaluation
        model.load_state_dict(checkpoint)
        logger.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")
    return model, optimizer, scaler, start_epoch


def remote_sync(local_dir: str, remote_dir: str, protocol: str) -> bool:
    """
    Sync the local directory with the remote directory using fsspec.

    Parameters:
    local_dir (str): Local directory path.
    remote_dir (str): Remote directory path.

    Returns:
    bool: True if sync was successful, False otherwise.
    """
    logger.info("Starting remote sync.")
    a, b = fsspec.get_mapper(local_dir), fsspec.get_mapper(remote_dir)
    for k in a:
        if "epoch_latest.pt" in k:
            continue
        if k in b and len(a[k]) == len(b[k]):
            logger.debug(f"Skipping remote sync for {k}.")
            continue
        try:
            b[k] = a[k]
            logger.info(f"Successfully synced {k}.")
        except Exception as e:
            logger.error(f"Error during remote sync for {k}: {e}")
            return False
    return True
