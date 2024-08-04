# coding=utf-8
# Copyright 2020 Optuna, Hugging Face
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Logging utilities."""

import functools
import logging
import os
import sys
import threading
import warnings
from datetime import datetime
from logging import (
    CRITICAL,  # NOQA
    DEBUG,  # NOQA
    ERROR,  # NOQA
    FATAL,  # NOQA
    INFO,  # NOQA
    NOTSET,  # NOQA
    WARN,  # NOQA
    WARNING,  # NOQA
)
from logging import captureWarnings as _captureWarnings
from pathlib import Path
from typing import Optional

from tqdm import auto as tqdm_lib

from mamba_clip.utils.dist_utils import broadcast_object

_lock = threading.Lock()
_default_handler: Optional[logging.Handler] = None

log_levels = {
    "detail": logging.DEBUG,  # will also print filename and line number
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

_default_log_level = logging.WARNING

_tqdm_active = True


def _get_default_logging_level():
    """
    If GOML_VERBOSITY env var is set to one of the valid choices return that as the new default level. If it is
    not - fall back to `_default_log_level`
    """
    env_level_str = os.getenv("GOML_VERBOSITY", None)
    if env_level_str:
        if env_level_str in log_levels:
            return log_levels[env_level_str]
        else:
            logging.getLogger().warning(
                f"Unknown option GOML_VERBOSITY={env_level_str}, "
                f"has to be one of: {', '.join(log_levels.keys())}"
            )
    return _default_log_level


def _get_library_name() -> str:
    return __name__.split(".")[0]


def _get_library_root_logger() -> logging.Logger:
    return logging.getLogger(_get_library_name())


def _configure_library_root_logger() -> None:
    global _default_handler

    with _lock:
        if _default_handler:
            # This library has already configured the library root logger.
            return
        _default_handler = logging.StreamHandler()  # Set sys.stderr as stream.
        # set defaults based on https://github.com/pyinstaller/pyinstaller/issues/7334#issuecomment-1357447176
        if sys.stderr is None:
            sys.stderr = open(os.devnull, "w")

        _default_handler.flush = sys.stderr.flush

        # Apply our default configuration to the library root logger.
        library_root_logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(_get_default_logging_level())
        # if logging level is debug, we add pathname and lineno to formatter for easy debugging
        if os.getenv("GOML_VERBOSITY", None) == "detail":
            formatter = logging.Formatter(
                "[%(levelname)s|%(pathname)s:%(lineno)s] %(asctime)s >> %(message)s"
            )
            _default_handler.setFormatter(formatter)

        library_root_logger.propagate = False


def _reset_library_root_logger() -> None:
    global _default_handler

    with _lock:
        if not _default_handler:
            return

        library_root_logger = _get_library_root_logger()
        library_root_logger.removeHandler(_default_handler)
        library_root_logger.setLevel(logging.NOTSET)
        _default_handler = None


def get_log_levels_dict():
    return log_levels


def captureWarnings(capture):
    """
    Calls the `captureWarnings` method from the logging library to enable management of the warnings emitted by the
    `warnings` library.

    Read more about this method here:
    https://docs.python.org/3/library/logging.html#integration-with-the-warnings-module

    All warnings will be logged through the `py.warnings` logger.

    Careful: this method also adds a handler to this logger if it does not already have one, and updates the logging
    level of that logger to the library's root logger.
    """
    logger = get_logger("py.warnings")

    if not logger.handlers:
        logger.addHandler(_default_handler)

    logger.setLevel(_get_library_root_logger().level)

    _captureWarnings(capture)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a logger with the specified name.

    This function is not supposed to be directly accessed unless you are writing a custom goml module.
    """

    if name is None:
        name = _get_library_name()

    _configure_library_root_logger()
    return logging.getLogger(name)


def get_verbosity() -> int:
    """
    Return the current level for the goml's root logger as an int.

    Returns:
        `int`: The logging level.

    <Tip>

    goml has following logging levels:

    - 50: `goml.logging.CRITICAL` or `goml.logging.FATAL`
    - 40: `goml.logging.ERROR`
    - 30: `goml.logging.WARNING` or `goml.logging.WARN`
    - 20: `goml.logging.INFO`
    - 10: `goml.logging.DEBUG`

    </Tip>"""

    _configure_library_root_logger()
    return _get_library_root_logger().getEffectiveLevel()


def set_verbosity(verbosity: int) -> None:
    """
    Set the verbosity level for the 🤗 goml's root logger.

    Args:
        verbosity (`int`):
            Logging level, e.g., one of:

            - `goml.logging.CRITICAL` or `goml.logging.FATAL`
            - `goml.logging.ERROR`
            - `goml.logging.WARNING` or `goml.logging.WARN`
            - `goml.logging.INFO`
            - `goml.logging.DEBUG`
    """

    _configure_library_root_logger()
    _get_library_root_logger().setLevel(verbosity)


def set_verbosity_info():
    """Set the verbosity to the `INFO` level."""
    return set_verbosity(INFO)


def set_verbosity_warning():
    """Set the verbosity to the `WARNING` level."""
    return set_verbosity(WARNING)


def set_verbosity_debug():
    """Set the verbosity to the `DEBUG` level."""
    return set_verbosity(DEBUG)


def set_verbosity_error():
    """Set the verbosity to the `ERROR` level."""
    return set_verbosity(ERROR)


def disable_default_handler() -> None:
    """Disable the default handler of the goml's root logger."""

    _configure_library_root_logger()

    assert _default_handler is not None
    _get_library_root_logger().removeHandler(_default_handler)


def enable_default_handler() -> None:
    """Enable the default handler of the goml's root logger."""

    _configure_library_root_logger()

    assert _default_handler is not None
    _get_library_root_logger().addHandler(_default_handler)


def set_default_handler(handler: logging.Handler) -> None:
    """Set the default handler of the goml's root logger."""

    _reset_library_root_logger()

    assert handler is not None
    _default_handler = handler


def add_handler(handler: logging.Handler) -> None:
    """adds a handler to the goml's root logger."""

    _configure_library_root_logger()

    assert handler is not None
    _get_library_root_logger().addHandler(handler)


def set_formatter(formatter: logging.Formatter) -> None:
    """adds a formatter to the goml's default handler"""
    global _default_handler

    _configure_library_root_logger()

    assert formatter is not None
    _default_handler.setFormatter(formatter)


def remove_handler(handler: logging.Handler) -> None:
    """removes given handler from the goml's root logger."""

    _configure_library_root_logger()

    assert handler is not None and handler not in _get_library_root_logger().handlers
    # log optuna study to
    _get_library_root_logger().removeHandler(handler)


def disable_propagation() -> None:
    """
    Disable propagation of the library log outputs. Note that log propagation is disabled by default.
    """

    _configure_library_root_logger()
    _get_library_root_logger().propagate = False


def enable_propagation() -> None:
    """
    Enable propagation of the library log outputs. Please disable the goml's default handler to
    prevent double logging if the root logger has been configured.
    """

    _configure_library_root_logger()
    _get_library_root_logger().propagate = True


def enable_explicit_format() -> None:
    """
    Enable explicit formatting for every goml's logger. The explicit formatter is as follows:
    ```
        [LEVELNAME|FILENAME|LINE NUMBER] TIME >> MESSAGE
    ```
    All handlers currently bound to the root logger are affected by this method.
    """
    handlers = _get_library_root_logger().handlers

    for handler in handlers:
        formatter = logging.Formatter(
            "[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s"
        )
        handler.setFormatter(formatter)


def reset_format() -> None:
    """
    Resets the formatting for goml's loggers.

    All handlers currently bound to the root logger are affected by this method.
    """
    handlers = _get_library_root_logger().handlers

    for handler in handlers:
        handler.setFormatter(None)


def warning_advice(self, *args, **kwargs):
    """
    This method is identical to `logger.warning()`, but if env var GOML_NO_ADVISORY_WARNINGS=1 is set, this
    warning will not be printed
    """
    no_advisory_warnings = os.getenv("GOML_NO_ADVISORY_WARNINGS", False)
    if no_advisory_warnings:
        return
    self.warning(*args, **kwargs)


logging.Logger.warning_advice = warning_advice


@functools.lru_cache(None)
def warning_once(self, *args, **kwargs):
    """
    This method is identical to `logger.warning()`, but will emit the warning with the same message only once

    Note: The cache is for the function arguments, so 2 different callers using the same arguments will hit the cache.
    The assumption here is that all warning messages are unique across the code. If they aren't then need to switch to
    another type of cache that includes the caller frame information in the hashing function.
    """
    self.warning(*args, **kwargs)


logging.Logger.warning_once = warning_once


class EmptyTqdm:
    """Dummy tqdm which doesn't do anything."""

    def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
        self._iterator = args[0] if args else None

    def __iter__(self):
        return iter(self._iterator)

    def __getattr__(self, _):
        """Return empty function."""

        def empty_fn(*args, **kwargs):  # pylint: disable=unused-argument
            return

        return empty_fn

    def __enter__(self):
        return self

    def __exit__(self, type_, value, traceback):
        return


class _tqdm_cls:
    def __call__(self, *args, **kwargs):
        if _tqdm_active:
            return tqdm_lib.tqdm(*args, **kwargs)
        else:
            return EmptyTqdm(*args, **kwargs)

    def set_lock(self, *args, **kwargs):
        self._lock = None
        if _tqdm_active:
            return tqdm_lib.tqdm.set_lock(*args, **kwargs)

    def get_lock(self):
        if _tqdm_active:
            return tqdm_lib.tqdm.get_lock()


tqdm = _tqdm_cls()


def is_progress_bar_enabled() -> bool:
    """Return a boolean indicating whether tqdm progress bars are enabled."""
    global _tqdm_active
    return bool(_tqdm_active)


def enable_progress_bar():
    """Enable tqdm progress bar."""
    global _tqdm_active
    _tqdm_active = True


def disable_progress_bar():
    """Disable tqdm progress bar."""
    global _tqdm_active
    _tqdm_active = False


def silence():
    warnings.filterwarnings("ignore")
    verb = get_verbosity()
    is_pb_enabled = is_progress_bar_enabled()
    set_verbosity(0)
    disable_progress_bar()
    return verb, is_pb_enabled


def unsilence(verbosity, is_pb_enabled):
    warnings.filterwarnings("default")
    set_verbosity(verbosity)
    if is_pb_enabled:
        enable_progress_bar()


# output the log to a text file
def logger_setup(output_dir=None, log_file=None):
    if log_file is not None:
        if isinstance(log_file, bool) and log_file:
            log_file = f"{datetime.datetime.now().strftime('%y-%m-%d_%h-%m-%s')}.err"
        if not log_file.endswith(".err"):
            log_file = Path(log_file).with_suffix(".err").name
        log_file = os.path.join(output_dir, log_file)
        handler = logging.StreamHandler()
        sys.stderr = open(log_file, "w")
        handler.flush = sys.stderr.flush
        handler.propagate = False
        set_default_handler(handler)

    set_verbosity_info()
    header = "[%(levelname)1.1s %(asctime)s]"
    message = "%(message)s"
    if _color_supported():
        import colorlog

        fh = colorlog.ColoredFormatter(
            f"%(log_color)s{header}%(reset)s {message}",
        )
    else:
        fh = logging.Formatter(f"{header} {message}")
    set_formatter(fh)
    enable_propagation()


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
