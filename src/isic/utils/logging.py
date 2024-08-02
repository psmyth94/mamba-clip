from datetime import datetime
import logging
import os
import sys
from typing import Optional

from isic.utils.dist_utils import broadcast_object


def _get_library_name() -> str:
    return __name__.split(".")[0]


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a logger with the specified name.

    This function is not supposed to be directly accessed unless you are writing a custom goml module.
    """

    if name is None:
        name = _get_library_name()

    setup_logging()
    return logging.getLogger(name)


# output the log to a text file
def setup_logging(log_file=None, level=None):
    if level is None:
        level = logging.INFO
    _default_handler = logging.StreamHandler()  # Set sys.stderr as stream.
    # set defaults based on https://github.com/pyinstaller/pyinstaller/issues/7334#issuecomment-1357447176
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")

    _default_handler.flush = sys.stderr.flush
    library_root_logger = logging.getLogger(__name__.split(".")[0])
    library_root_logger.addHandler(_default_handler)
    library_root_logger.setLevel(level)

    header = "[%(levelname)1.1s %(asctime)s]"
    message = "%(message)s"
    if _color_supported():
        import colorlog

        fh = colorlog.ColoredFormatter(
            f"%(log_color)s{header}%(reset)s {message}",
        )
    else:
        fh = logging.Formatter(f"{header} {message}")
    _default_handler.setFormatter(fh)
    library_root_logger.propagate = False
    if log_file:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(fh)
        library_root_logger.addHandler(file_handler)
    return library_root_logger


def _color_supported() -> bool:
    """Detection of color support."""
    # NO_COLOR environment variable:
    if os.environ.get("NO_COLOR", None):
        return False
    if not hasattr(sys.stderr, "isatty") or not sys.stderr.isatty():
        return False
    else:
        return True


def create_log_path(args, model, latest=False):
    # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
    model_name_safe = model.replace("/", "-")
    date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    if not latest:
        if args.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args, date_str)
        return "-".join([
            date_str,
            f"model_{model_name_safe}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])
    else:
        logs = []
        for f in os.listdir(args.logs):
            if f"model_{model_name_safe}" in f and os.path.exists(
                os.path.join(args.logs, f, "checkpoints", "stage_1_latest.pt")
            ):
                logs.append(f)

        # sort by date, then take the latest
        logs = sorted(logs, key=lambda x: x.split("-")[0])
        if logs:
            return logs[-1]
        else:
            raise ValueError(f"No logs found for model: {model_name_safe}")
