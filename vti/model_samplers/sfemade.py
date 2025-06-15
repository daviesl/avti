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
from nflows.transforms import made as made_module
from torch.distributions import Bernoulli
from vti.model_samplers import AbstractSFEModelSampler

# import vti.utils.logging as logging
import logging

PRINTSCALE = True


def log_sigmoid(x):
    return -torch.log(1 + torch.exp(-x))


def stable_binary_cross_entropy_with_logits(logits, targets):
    """
    Compute binary cross-entropy loss in a numerically stable way.

    Parameters:
    - logits (torch.Tensor): The logits values.
    - targets (torch.Tensor): The target values, must be the same shape as logits.

    Returns:
    - torch.Tensor: The computed binary cross-entropy loss for each element.
    """
    # Clipping logits to avoid extreme sigmoid outputs
    logits_clipped = torch.clamp(logits, min=-500, max=500)

    # Computing binary cross-entropy loss safely
    # Note: 'reduction' is set to 'none' to compute loss for each element individually
    loss = F.binary_cross_entropy_with_logits(logits_clipped, targets, reduction="none")

    return loss


def sigmoid_ratio(z1, z2):
    """
    z1 and z2 are tensors of the same shape.
    returns sigmoid(z1)/sigmoid(z2)
    """

    # log(1 + e^-z1) = logsumexp over {0, -z1}
    l1 = torch.logsumexp(torch.stack([torch.zeros_like(z1), -z1], dim=-1), dim=-1)
    # log(1 + e^-z2) = logsumexp over {0, -z2}
    l2 = torch.logsumexp(torch.stack([torch.zeros_like(z2), -z2], dim=-1), dim=-1)

    # log(σ(z1)/σ(z2)) = [l2 - l1]
    log_ratio = l2 - l1

    # σ(z1)/σ(z2) = exp(log_ratio)
    return torch.exp(log_ratio)


class UniformMADE(made_module.MADE):
    """
    Only zeros the last layer
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initialize_uniform()

    def initialize_uniform(self):
        # Only zero out the final layer weights and biases.
        # Leave other layers as-is for better optimization dynamics.
        if hasattr(self, "final_layer"):
            if isinstance(self.final_layer, nn.Linear):
                nn.init.constant_(self.final_layer.weight, 0.0)
                if self.final_layer.bias is not None:
                    nn.init.constant_(self.final_layer.bias, 0.0)


class SFEMADEBinaryString(AbstractSFEModelSampler):
    def __init__(
        self, num_bits, ig_threshold, lr, init_uniform=True, device=None, dtype=None
    ):
        """
        args:
            num_bits: length of binary string (int)
        """
        super().__init__(ig_threshold, lr, device=device, dtype=dtype)
        self.register_buffer(
            "num_bits", torch.tensor(num_bits, device=self.device, dtype=torch.int32)
        )

        def bitstoblocks(nb):
            if nb <= 8:
                return 2
            elif nb <= 16:
                return 3
            elif nb <= 32:
                return 4

        hidden_features = int(2 * bitstoblocks(num_bits) * num_bits)  # Tune this, or make init param.

        logging.info(f"num_bits={self.num_bits}, hidden_features={hidden_features}")

        mmclass = UniformMADE if init_uniform else made_module.MADE


        self.made = mmclass(
            features=num_bits,
            hidden_features=hidden_features,
            context_features=None,
            num_blocks=bitstoblocks(num_bits),
            output_multiplier=1,
            activation=torch.nn.functional.leaky_relu,
            # use_batch_norm=True,
        ).to(self.device)

        self._setup_optimizer()

    def _estimate_entropy(self, inputs):
        return self._estimate_entropy_from_update(inputs, self.made)

    def _estimate_entropy_update(self, inputs, lr, grads):
        # Clone the MADE module
        updated_made = copy.deepcopy(self.made)

        # logging.info(f"_eeu:lr={lr}")

        # Apply the gradient update to the cloned module
        params = [p for p in updated_made.parameters()]
        with torch.no_grad():
            for p, g in zip(params, grads):
                # if lr>0:
                #    logging.info(f"_eeu:lr={lr}\n_eeu:params={p}\n_eeu:grads={g}")
                p -= lr * g
                # if lr>0:
                #    logging.info(f"_eeu:updated_params={p}")

        return self._estimate_entropy_from_update(inputs, updated_made)

    def _estimate_entropy_from_update(self, inputs, updated_made):
        """
        Monte Carlo estimate of the entropy of the MADE distribution.
        If gradients are provided, estimates the entropy after applying the gradient update.
        """

        # Compute log probabilities under the updated MADE
        # biased, reuses inputs (requires some thought)
        # rewritten using importance weighting for correct entropy
        prev_logits = self.made(inputs)
        logits = updated_made(inputs)
        # Safely compute importance weights
        importance_weights = sigmoid_ratio(logits, prev_logits)
        log_probs_tensor = -stable_binary_cross_entropy_with_logits(logits, inputs)
        log_probs = (importance_weights * log_probs_tensor).sum(dim=1)

        # Estimate entropy
        entropy_estimate = -log_probs.mean()

        return entropy_estimate

    def _log_prob(self, string):
        # Compute logits for the sampled string (gradient tracking enabled)
        logits = self.made(string)
        # Compute log probabilities
        return -stable_binary_cross_entropy_with_logits(logits, string).sum(dim=1)

    def log_prob(self, string):
        """
        publif facing method
        """
        return self._log_prob(string)

    def action_sample(self, batch_size):
        return self._sample(batch_size)

    def _sample(self, batch_size):
        with torch.no_grad():
            string = torch.zeros((batch_size, self.num_bits), device=self.device)
            # Sequentially sample from MADE
            for i in range(self.num_bits):
                # Compute logits based on current string
                logits = self.made(string)
                # use Bernoulli dist with logits to avoid numerical errors
                dist = Bernoulli(logits=logits[:, i])
                # Sample bit from Bernoulli distribution
                # Update the string with the sampled bit
                string[:, i] = dist.sample()
        return string

    def action_sample_and_log_prob(self, batch_size):
        string = self._sample(batch_size)
        return string, self._log_prob(string)
