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

import numpy as np
import torch
import logging
import sys


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def is_pos_def_torch(x):
    test = torch.real(torch.linalg.eigvals(torch.real(x)))
    return torch.all(test > 0)


def make_pos_def_torch(x):
    """
    Force a square symmetric matrix to be positive semi definite
    """
    w, v = torch.linalg.eigh(torch.real(x))
    w_pos = torch.clip(w, 0, None)
    nonzero_w = w_pos[w_pos > 0]
    w_new = w_pos
    if nonzero_w.shape[0] > 0:
        if nonzero_w.shape[0] < w.shape[0]:
            # min_w = max(torch.max(nonzero_w)*1e-5,torch.min(nonzero_w))
            min_w = max(torch.max(nonzero_w) * 0.1, torch.min(nonzero_w))
            w_new = w_pos + min_w
    elif nonzero_w.shape[0] == 0:
        logging.warning(
            f"No positive eigenvalues for A {x}. w={w} {w.shape}, w_pos={w_pos} {w_pos.shape}, nonzero_w={nonzero_w}"
        )
        w_new = torch.ones_like(w)
    x_star = v @ np.diag(w_new) @ v.T
    p = torch.sqrt(torch.sum(torch.abs(w)) / torch.sum(torch.abs(w_new)))
    return p * x_star


def make_pos_def(x):
    """
    Force a square symmetric matrix to be positive semi definite
    """
    w, v = np.linalg.eigh(x)
    w_pos = np.clip(w, 0, None)
    nonzero_w = w_pos[w_pos > 0]
    w_new = w_pos
    if nonzero_w.shape[0] > 0:
        if nonzero_w.shape[0] < w.shape[0]:
            min_w = max(np.max(nonzero_w) * 1e-5, np.min(nonzero_w))
            w_new = w_pos + min_w
    else:
        logging.warning(
            f"No positive eigenvalues for A {x}. w={w} {w.shape}, w_pos={w_pos} {w_pos.shape}, nonzero_w={nonzero_w}"
        )
        w_new = np.ones_like(w)
    # w_neg = np.abs(np.clip(w,None,0))
    x_star = v @ np.diag(w_new) @ v.T
    p = np.sqrt(np.sum(np.abs(w)) / np.sum(w_new))
    return p * x_star


def safe_logdet(x):
    sign, ld = np.linalg.slogdet(x)
    while not np.isfinite(ld):
        w, v = np.linalg.eigh(x)
        maxw = np.max(np.abs(w))
        w += 0.1 * maxw
        logging.info(
            f"log det of {x} is not finite. Adding {maxw} to eigvalues {w} {v}"
        )
        x = v @ np.diag(w) @ v.T
        logging.info(f"New x is {x}")
        sign, ld = np.linalg.slogdet(x)
        # sys.exit(0)
    return sign, ld


def safe_inv(x):
    try:
        return np.linalg.pinv(x)
    except Exception as ex:
        logging.info(ex)
        # logging.info(x)
        sys.exit(0)
        x = make_pos_def(x)
        logging.info(x)
        sys.exit(0)

    w, v = np.linalg.eigh(x)
    if np.any(w < 0):
        logging.info(f"WARNING: negative eigenvalue in inverse of {x}, {w}, {v}")
        # w[w<0] = 0
        w = np.abs(w)
    if np.any(w == 0):
        logging.info(f"WARNING: zero eigenvalue in inverse of {x}, {w}, {v}")
        maxw = np.max(np.abs(w))
        minw = np.min(np.abs(w[w != 0]))
        # w[w==0] = minw
        # w += minw
        w += 0.001 * minw
        logging.info(f"new eigenvalues {w}")
    return v @ np.diag(w ** (-1)) @ v.T


def sum_along_axis(a, axis=1):
    # assumes it is a numpy array
    if a.ndim <= axis:
        return a
    else:
        return a.sum(axis=axis)
