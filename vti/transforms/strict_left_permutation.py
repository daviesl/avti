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

import torch
import torch.nn as nn
from torch.nn import functional as F
import logging

from nflows.utils import torchutils

from nflows.transforms.base import Transform
from nflows.transforms.permutations import Permutation

class StrictLeftPermutation(Permutation):
    """
    Permute to strict left active order.
    Requires context to be not None.
    Assumes context is a mask of same dimension as inputs.
    Forward will permute to strict left active order.
    Inverse will assume inputs are in strict left order
            and will permute back to order described by context.
    """
    def __init__(self, features, context_to_mask, dim=1):
        """
        initialise with dummy permutation.
        """
        self._dim = dim
        self.ctm = context_to_mask # just map to number of inputs.
        super().__init__(torch.arange(features), dim=self._dim)

    @staticmethod
    def _get_permutation_from_context(context):
        """
        Args:
        Row-wise permutation that brings the indices of non-zero entries
        to the front (ascending), followed by the indices of the zeros
        (ascending).  Works on GPU/CPU and is fully vectorised.
    
        Parameters
        ----------
        x : torch.Tensor  # shape (B, M), any dtype
    
        Returns
        -------
        perm : torch.LongTensor  # shape (B, M)
            Row k holds a permutation of 0…M-1 for row k of `x`.
    
        Example
        -------
        >>> a = torch.tensor([[1., 1., 0., 1., 0., 0., 1.],
        ...                  [0., 1., 1., 1., 1., 0., 0.]])
        >>> nz_first_permutation(a)
        tensor([[0, 1, 3, 6, 2, 4, 5],
                [1, 2, 3, 4, 0, 5, 6]])
        """
        if context.dim() != 2:
            raise ValueError("context must be 2-D")
    
        B, M = context.shape
        idx = torch.arange(M, device=context.device).expand(B, M)   # [[0,1,…,M-1], …]
        zero = (context == 0)
    
        #  Non-zeros get keys 0…M-1, zeros get keys M…2M-1
        sort_key = zero.to(idx.dtype) * M + idx
        return torch.argsort(sort_key, dim=1)

    @staticmethod
    def _permute(inputs, permutation, dim):
        if dim >= inputs.ndim:
            raise ValueError(f"No dimension {dim} in inputs.")
        if inputs.shape[dim] != permutation.shape[1]:
            raise ValueError(f"Permutation length must match dimension size. inputs dim={inputs.shape[dim]}, permutation dim={permutation.shape[1]}")
    
        # bring `dim` to position 1 so we can gather easily
        x = inputs.transpose(0, dim)          # shape (M, …, B)
        out = torch.gather(x, 0, permutation.T)  # (M, …, B)
        out = out.transpose(0, dim)           # back to original order
    
        return out
            

    def _inverse_permutation(self, permutation):
        return torch.argsort(permutation, dim=self._dim)

    def _forward_no_logabsdet(self, inputs, context):
        return self._permute(inputs, self._get_permutation_from_context(self.ctm(context)), self._dim)


    def forward(self, inputs, context=None):
        logabsdet = inputs.new_zeros(inputs.shape[0])
        return self._permute(inputs, self._get_permutation_from_context(self.ctm(context)), self._dim), logabsdet

    def inverse(self, inputs, context=None):
        logabsdet = inputs.new_zeros(inputs.shape[0])
        return self._permute(inputs, self._inverse_permutation(self._get_permutation_from_context(self.ctm(context))), self._dim), logabsdet

