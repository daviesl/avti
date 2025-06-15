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
from torch import nn
from torch.distributions import Categorical

from vti.model_samplers.abstract import AbstractModelSampler
from vti.utils.debug import check_for_nans
from vti.utils.math_helpers import ensure_2d
import logging


class SoftmaxSurrogateSampler(AbstractModelSampler):
    """
    Surrogate sampler class, which models the logits of a categorical distribution.
    """

    def __init__(
        self, surrogate, check_nans=True, squish_utility=True, device=None, dtype=None
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.surrogate = surrogate
        self.squish_utility = squish_utility  # apply nonlinear transform
        if check_nans:
            self.check_for_nans = check_for_nans
        else:
            self.check_for_nans = lambda x: x

    def p_squish(self, l):
        """
        squish onto the simplex
        """
        return torch.softmax(l, dim=0)

    def p_unsquish(self, p):
        """
        map from simplex to unconstrained space
        """
        return p.log()

    def logits(self):
        return self.surrogate.mean()

    def probs(self):
        return self.p_squish(self.logits())

    def action_logits(self):
        return self.surrogate.utility_UCB()

    def action_probs(self):
        return self.p_squish(self.action_logits())

    def sample(self, batch_size=1):
        """
        Falls back to action sample per default
        """
        return self.action_sample(batch_size)

    def action_sample_and_log_prob(self, batch_size):
        action_dist = self.action_dist()
        mk_catsamples = action_dist.sample((batch_size,))
        mk_log_probs = action_dist.log_prob(mk_catsamples)
        return mk_catsamples, mk_log_probs

    def _log_prob(self, mk_catsamples):
        action_dist = self.action_dist()
        return action_dist.log_prob(mk_catsamples)

    def log_prob(self, mk_catsamples):
        return self._log_prob(mk_catsamples)

    def action_sample(self, batch_size):
        action_dist = self.action_dist()
        mk_catsamples = action_dist.sample((batch_size,))

        return mk_catsamples

    def action_logits(self):
        return self.surrogate.utility_UCB()

    def action_dist(self):
        if self.squish_utility:
            # we get superior performance if we apply a transformation to the logits
            return Categorical(logits=self.p_squish(self.action_logits()))
        return Categorical(logits=self.action_logits())

    def debug_log(self):
        self.surrogate.debug_log()

    def observe(self, mk_catsamples, loss_hat, iteration):
        return self.surrogate.observe(mk_catsamples, loss_hat)

    def evolve(self, mk_cat_samples, ell, optimizer, loss, iteration):
        """
        update the prior based on the loss of the flow model.
        A no-op by default, presuming stationarity.
        """
        return self.surrogate.evolve(mk_cat_samples, ell, optimizer, loss)

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        state["surrogate_state_dict"] = self.surrogate.state_dict()
        return state

    def load_state_dict(self, state_dict, strict=True):
        if "surrogate_state_dict" in state_dict:
            self.surrogate.load_state_dict(state_dict.pop("surrogate_state_dict"))
        # Load the rest of the state_dict
        super().load_state_dict(state_dict, strict=strict)


class BinaryStringSSSampler(SoftmaxSurrogateSampler):
    """
    This class acts as the transform that respects the
    isomorphism between a binary string random variable
    and a categorical random variable over a support space
    of the same cardinality.
    """

    def __init__(
        self, surrogate, check_nans=True, squish_utility=True, device=None, dtype=None
    ):
        super().__init__(surrogate, check_nans, squish_utility, device, dtype)
        # self.binary_string_length = self._binary_string_length(...)

    def action_sample(self, batch_size):
        samples = super().action_sample(batch_size)
        return self._integer_to_right_aligned_binary_tensor(
            inputs=samples,
            num_digits=self._binary_string_length(
                self.surrogate._num_categories() - 1  # count from zero
            ),
            dtype=self.dtype,
        )

    def action_sample_and_log_prob(self, batch_size):
        samples, log_probs = super().action_sample_and_log_prob(batch_size)
        # transform the samples to a binary string
        return self._integer_to_right_aligned_binary_tensor(
            inputs=samples,
            num_digits=self._binary_string_length(
                self.surrogate._num_categories() - 1  # count from zero
            ),
            dtype=self.dtype,
        ), log_probs

    def log_prob(self, mk_samples):
        # convert binary strings to integer categories
        # logging.info(f"Converting to categories {mk_samples}")
        return super()._log_prob(self._mk_identifier_to_cat(mk_samples).to(torch.int64))

    @staticmethod
    def _binary_string_length(integer: int) -> int:
        """
        Return the length of the binary string (excluding '0b') required to represent `integer`.
        """
        # Special case: if integer == 0, the binary string is "0" of length 1
        if integer == 0:
            return 1
        return len(bin(integer)[2:])

    @staticmethod
    def _integer_to_right_aligned_binary_tensor(
        inputs: torch.Tensor, num_digits: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Convert each element in a 1D tensor `inputs` to a right-aligned binary representation
        (zero-padded on the left) of length `num_digits`, in a fully vectorized manner.

        Args:
            inputs (torch.Tensor): 1D tensor of integers (or floats, which will be truncated to int).
            num_digits (int): Number of binary digits in the output representation.
            dtype (torch.dtype): Desired dtype of the output tensor (e.g., torch.int64, torch.float32, etc.).

        Returns:
            torch.Tensor: A 2D tensor of shape (len(inputs), num_digits) on the same device as `inputs`,
                          containing the right-aligned binary representation of each integer in `inputs`.
        """
        # Ensure `inputs` is 1D
        if inputs.dim() != 1:
            raise ValueError("`inputs` must be a 1D tensor.")

        # Convert inputs to integer type (truncates floats)
        inputs_int = inputs.to(torch.int)

        # Check for negative values
        if torch.any(inputs_int < 0):
            raise ValueError(
                "Negative values are not supported for binary representation. "
                f"Found: {inputs_int[inputs_int < 0]}"
            )

        # Check that values fit within `num_digits` bits
        #    i.e., max representable value is (2^num_digits - 1).
        max_val = (1 << num_digits) - 1  # 2**num_digits - 1
        if torch.any(inputs_int > max_val):
            raise ValueError(
                f"Some integer(s) in `inputs` exceed what can be represented with {num_digits} bits. "
                f"Max representable is {max_val}."
            )

        # Vectorized bit extraction
        #    Create a range of bit positions from the most significant (num_digits-1) down to 0
        #    so the leftmost column in the output corresponds to the highest bit.
        device = inputs_int.device
        bit_positions = torch.arange(
            num_digits - 1, -1, -1, device=device, dtype=inputs_int.dtype
        )
        # Shape of bit_positions: [num_digits]

        # For each input integer, we shift right by bit_positions and take & 1
        # This produces shape (len(inputs), num_digits).
        bits = (inputs_int.unsqueeze(1) >> bit_positions) & 1

        # Cast the bits to the desired dtype
        bits = bits.to(dtype)

        # logging.info(f"BS SSS sampled {bits}")

        return bits

    def _mk_identifier_to_cat(self, mk_samples):
        """
        Converts binary string mk identifier to
        categorical random variable, i.e. integer,
        which is passed to the surrogate
        """
        binary_string_length = self._binary_string_length(
            self.surrogate._num_categories() - 1  # count from zero
        )  # could also get this from mk_samples dim
        assert (
            mk_samples.shape[1] == binary_string_length
        ), f"Dimension mismatch between input mk samples and expected string length, {mk_samples.shape[1]} <> {binary_string_length}"
        pow2 = torch.pow(2, torch.arange(binary_string_length - 1, -1, -1)).view(1, -1)
        return (ensure_2d(mk_samples) * pow2).sum(dim=1)

    def observe(self, mk_samples, loss_hat, iteration):
        # convert samples to categories
        mk_catsamples = self._mk_identifier_to_cat(mk_samples).to(torch.int64)
        return super().observe(mk_catsamples, loss_hat, iteration)

    def evolve(self, mk_samples, ell, optimizer, loss, iteration):
        # convert samples to categories
        mk_catsamples = self._mk_identifier_to_cat(mk_samples).to(torch.int64)
        return super().evolve(mk_catsamples, ell, optimizer, loss, iteration)
