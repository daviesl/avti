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
from torch.nn import functional as F
import copy
from torch.distributions import Categorical

from vti.model_samplers import AbstractSFEModelSampler
from vti.utils.math_helpers import ensure_2d
from vti.utils.debug import check_for_nans
from torch.distributions import Distribution

Distribution.set_default_validate_args(False)

import logging
import math


class SFECategorical(AbstractSFEModelSampler):
    def __init__(self, num_categories, ig_threshold, lr, device=None, dtype=None):
        # scale the learning rate to the number of categories
        lr = lr * math.log2(num_categories)

        super().__init__(ig_threshold, lr, device=device, dtype=dtype)
        self._logits = nn.Parameter(
            torch.zeros((num_categories,), device=device, dtype=dtype)
        )
        self.register_buffer(
            "num_categories",
            torch.tensor(num_categories, device=self.device, dtype=torch.int32),
        )

        self._setup_optimizer()

    def logits(self):
        # return self._logits
        raise NotImplementedError("logits() not implemented in SFECategorical")

    def _probabilities(self, logits=None):
        if logits is None:
            logits = self._logits
        return F.softmax(logits, dim=0)

    def _estimate_entropy_update(self, inputs, lr, grad):
        # updated_logits = copy.deepcopy(self.logits())
        with torch.no_grad():
            updated_logits = self._logits.detach().clone()
            # params = [updated_logits]
            # for p, g in zip(params, grad):
            #    p -= lr * g
            # probs = self._probabilities(params[0])
            # print("sfe categorical probs", probs, params[0], self._logits, "grad",grad)
            probs = self._probabilities(
                updated_logits - lr * grad[0]
            )  # we know there is only one element in grad, the logits
        return self._entropy_given_probs(probs)

    def _estimate_entropy(self, inputs):
        return self._entropy_given_probs(self._probabilities())

    #def _entropy_given_probs(self, probs):
    #    with torch.no_grad():
    #        probs = torch.clamp(probs.detach(), 1e-20, 1 - 1e-20)
    #        probs = probs / probs.sum()
    #        entropy = -(probs * probs.log()).sum()
    #    return entropy

    def _entropy_given_probs(self, probs):
        with torch.no_grad():
            # Detach probabilities from the graph.
            probs = probs.detach()
            total = probs.sum()
            if total == 0:
                raise ValueError("The sum of probabilities is zero; cannot normalize.")
            tolerance = 1e-8  # Pre-defined tolerance for floating point error.
            if abs(total.item() - 1.0) > tolerance:
                #logging.info(f"The sum of probabilities is {total.item()} != 1 within tolerance {tolerance}. Normalizing.")
                probs = probs / total
            # Create a mask for nonzero probabilities.
            nonzero_mask = probs > 0
            # Compute entropy only for nonzero probabilities:
            # For p > 0, p * log(p) is computed; for p = 0, the contribution is 0.
            entropy = -(probs[nonzero_mask] * torch.log(probs[nonzero_mask])).sum()
        return entropy

    def action_sample(self, batch_size):
        return self._sample(batch_size)

    def _sample(self, batch_size):
        with torch.no_grad():
            dist = Categorical(logits=self._logits)
            samples = dist.sample((batch_size,))
        return samples

    def _log_prob(self, inputs):
        # dist = Categorical(logits=self._logits)
        # return dist.log_prob(inputs)
        # manual version
        log_probs = torch.log_softmax(self._logits, dim=-1)
        categories = inputs
        selected_log_probs = torch.gather(log_probs, dim=-1, index=categories)
        # logging.info(f"cats {categories}\nlog_probs {selected_log_probs}")
        # return selected_log_probs.squeeze(-1)
        return selected_log_probs

    def log_prob(self, mk_catsamples):
        """
        public facing method
        """
        return self._log_prob(mk_catsamples)

    def action_sample_and_log_prob(self, num_samples):
        samples = self._sample(num_samples)
        return samples, self._log_prob(samples)


class SFECategoricalBinaryString(SFECategorical):
    """
    This class acts as the transform that respects the
    isomorphism between a binary string random variable
    and a categorical random variable over a support space
    of the same cardinality.
    """

    def _num_categories(self):
        return self.num_categories

    def action_sample_and_log_prob(self, batch_size):
        catsamples, log_probs = super().action_sample_and_log_prob(batch_size)
        # transform the samples to a binary string
        with torch.no_grad():
            samples = self._integer_to_right_aligned_binary_tensor(
                inputs=catsamples.detach(),
                num_digits=self._binary_string_length(
                    self._num_categories() - 1  # count from zero
                ),
                dtype=self.dtype,
            )
            samples.grad = catsamples.grad
        return samples, log_probs

    def action_sample(self, batch_size):
        catsamples = super().action_sample(batch_size)
        with torch.no_grad():
            samples = self._integer_to_right_aligned_binary_tensor(
                inputs=catsamples.detach(),
                num_digits=self._binary_string_length(
                    self._num_categories() - 1  # count from zero
                ),
                dtype=self.dtype,
            )
            samples.grad = catsamples.grad
        return samples

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
            self._num_categories() - 1  # count from zero
        )  # could also get this from mk_samples dim
        assert (
            mk_samples.shape[1] == binary_string_length
        ), f"Dimension mismatch between input mk samples and expected string length, {mk_samples.shape[1]} <> {binary_string_length}"
        pow2 = torch.pow(2, torch.arange(binary_string_length - 1, -1, -1)).view(1, -1)
        return (ensure_2d(mk_samples) * pow2).sum(dim=1)

    def observe(self, mk_samples, loss_hat, mk_log_prob, iteration):
        # convert samples to categories
        mk_catsamples = self._mk_identifier_to_cat(mk_samples).to(torch.int64)
        return super().observe(mk_catsamples, loss_hat, mk_log_prob, iteration)

    def evolve(self, mk_samples, ell, mk_log_prob, optimizer, loss, iteration):
        # convert samples to categories
        mk_catsamples = self._mk_identifier_to_cat(mk_samples).to(torch.int64)
        return super().evolve(
            mk_catsamples, ell, mk_log_prob, optimizer, loss, iteration
        )
