# MIT License
#
# Copyright (c) 2025 Laurence Davies, Dan Mackinlay, Rafael Oliveira, Scott A. Sisson
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""
A custom logging class that contains

1. a helper to log messages to a particular dir and force DEBUG level logging
2. some helper functions to support old-style template strings and the classic string `format` method
"""

import logging
import sys
from string import Template


class BraceMessage:
    def __init__(self, fmt, *args, **kwargs):
        self.fmt = fmt
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return self.fmt.format(*self.args, **self.kwargs)


class DollarMessage:
    def __init__(self, fmt, /, **kwargs):
        self.fmt = fmt
        self.kwargs = kwargs

    def __str__(self):
        return Template(self.fmt).substitute(**self.kwargs)


_bm = BraceMessage  # Alias for convenience
_dm = DollarMessage


# Create and configure a global logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)
logger.addHandler(handler)


def _log_message(level_func, *args, **kwargs):
    if len(args) == 1:
        # Single argument could be a pre-formatted string, a BraceMessage instance, or a DollarMessage instance.
        message = str(args[0])
    else:
        # Multiple arguments: join them by space
        message = " ".join(str(arg) for arg in args)

    level_func(message, **kwargs)


def debug(*args, **kwargs):
    _log_message(logger.debug, *args, **kwargs)


def info(*args, **kwargs):
    _log_message(logger.info, *args, **kwargs)


def warning(*args, **kwargs):
    _log_message(logger.warning, *args, **kwargs)


def error(*args, **kwargs):
    _log_message(logger.error, *args, **kwargs)


def critical(*args, **kwargs):
    _log_message(logger.critical, *args, **kwargs)


def set_log_directory(output_dir):
    # Remove any existing handlers
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    # Set up a file handler that writes to the specified directory
    file_handler = logging.FileHandler(f"{output_dir}/debug.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
