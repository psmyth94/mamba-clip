# type: ignore
import os
from datetime import datetime, timedelta

import torch
import torch.distributed as dist


def world_info_from_env():
    local_rank = 0
    for v in (
        "LOCAL_RANK",
        "MPI_LOCALRANKID",
        "SLURM_LOCALID",
        "OMPI_COMM_WORLD_LOCAL_RANK",
    ):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ("RANK", "PMI_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ("WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


def init_device(args):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    if torch.cuda.is_available() and (args.device == "auto" or "cuda" in args.device):
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    args.distributed = False
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0
    device = args.device
    if args.device == "auto" or "cuda" in args.device:
        if is_using_distributed():
            if "SLURM_PROCID" in os.environ:
                # DDP via SLURM
                args.local_rank, args.rank, args.world_size = world_info_from_env()
                # SLURM var -> torch.distributed vars in case needed
                os.environ["LOCAL_RANK"] = str(args.local_rank)
                os.environ["RANK"] = str(args.rank)
                os.environ["WORLD_SIZE"] = str(args.world_size)
                torch.distributed.init_process_group(
                    backend=args.dist_backend,
                    init_method=args.dist_url,
                    world_size=args.world_size,
                    timeout=timedelta(0, 3600),
                    rank=args.rank,
                )
            else:
                # DDP via torchrun, torch.distributed.launch
                args.local_rank, _, _ = world_info_from_env()
                torch.distributed.init_process_group(
                    backend=args.dist_backend,
                    init_method=args.dist_url,
                    timeout=datetime.timedelta(0, 3600),
                    rank=args.rank,
                )
                args.world_size = torch.distributed.get_world_size()
                args.rank = torch.distributed.get_rank()
            args.distributed = True

        if torch.cuda.is_available():
            if args.distributed and not args.no_set_device_rank:
                device = "cuda:%d" % args.local_rank
            else:
                device = "cuda:0"
            torch.cuda.set_device(device)
        else:
            device = "cpu"
    args.device = device
    device = torch.device(device)
    return device


def broadcast_object(args, obj, src=0):
    # broadcast a pickle-able python object from rank-0 to all ranks
    if args.rank == src:
        objects = [obj]
    else:
        objects = [None]
    dist.broadcast_object_list(objects, src=src)
    return objects[0]


def is_global_master(args):
    if "MASTER_NODE" in os.environ and "SLURM_NODELIST" in os.environ:
        return os.environ["MASTER_NODE"] in os.environ["SLURM_NODELIST"]
    return args.rank == 0


def is_local_master(args):
    return args.local_rank == 0


def is_master(args, local=False):
    return is_local_master(args) if local else is_global_master(args)


def is_using_distributed():
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"]) > 1
    if "SLURM_NTASKS" in os.environ:
        return int(os.environ["SLURM_NTASKS"]) > 1
    return False
