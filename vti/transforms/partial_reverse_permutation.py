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
from nflows.transforms.permutations import Permutation
from typing import Callable


class PartialReversePermutation(Permutation):
    """
    Reverse only the *left-aligned* non-zero block of each row.

    Let Nᵢ = ∑ⱼ 𝟙{context[i,j] ≠ 0}.  
    For every row i we reverse the first Nᵢ elements and leave the rest
    unchanged:

        inputs[i, :Nᵢ]  ->  inputs[i, Nᵢ-1 : -1 : 0]
        inputs[i, Nᵢ:]  ->  unchanged

    The operation is its own inverse, so `forward` = `inverse`.
    No Python loops – everything is batch-vectorised.
    """

    def __init__(self, features: int, context_to_mask: Callable[[torch.Tensor], torch.Tensor], dim: int = 1):
        """
        Parameters
        ----------
        features : int
            Width of the input (size in `dim`).
        dim : int
            Dimension along which to permute (default 1, i.e. channel/features
            axis when inputs are (B, F)).
        """
        # dummy identity permutation needed by the parent ctor
        super().__init__(torch.arange(features), dim)
        self.ctm = context_to_mask # just map to number of inputs.
        self._features = features          # keep for sanity checks

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _build_permutation(context: torch.Tensor) -> torch.LongTensor:
        """
        Construct a (B, M) index tensor suitable for `torch.gather`
        implementing the partial reversal.
        """
        if context.ndim != 2:
            raise ValueError("context must be 2-D (batch × features)")

        B, M = context.shape
        idx = torch.arange(M, device=context.device).expand(B, M)        # 0…M-1
        n_nonzero = (context != 0).sum(dim=1, keepdim=True)              # (B, 1)

        # For positions j < Nᵢ take Nᵢ-1-j, else j
        perm = torch.where(idx < n_nonzero, n_nonzero - 1 - idx, idx)
        return perm.long()

    @staticmethod
    def _apply_permutation(inputs, permutation, dim):
        out = torch.gather(inputs, dim, permutation)
        logabsdet = inputs.new_zeros(inputs.shape[0])
        return out, logabsdet

    # ---------------------------------------------------------------- forward / inverse
    def forward(self, inputs, context=None):
        if context is None:
            raise ValueError("context must be provided")
        if inputs.shape[self._dim] != self._features:
            raise ValueError("inputs have wrong feature dimension size")
        perm = self._build_permutation(self.ctm(context))
        return self._apply_permutation(inputs, perm, self._dim)

    # the operation is involutory ⇒ same implementation
    inverse = forward

