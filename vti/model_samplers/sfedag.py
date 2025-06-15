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
from vti.utils.jit import *
from torch import nn
from torch.nn import functional as F
#import copy
from vti.transforms import madeplus as made_module
from torch.distributions import Bernoulli, Categorical
from vti.model_samplers import AbstractSFEModelSampler
from vti.utils.math_helpers import upper_bound_power_of_2, log_sigmoid

# import vti.utils.logging as logging
import logging

PRINTSCALE = True


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


# class UniformMADEPlus(made_module.MADEPlus):
#    def __init__(self, *args, **kwargs):
#        super().__init__(*args, **kwargs)
#
#        # Initialize weights and biases to output zero logits
#        self.initialize_uniform()
#
#    def initialize_uniform(self):
#        for module in self.modules():
#            if isinstance(module, nn.Linear):
#                # Set weights to zero
#                nn.init.constant_(module.weight, 0.0)
#                if module.bias is not None:
#                    # Set biases to zero
#                    nn.init.constant_(module.bias, 0.0)
#


class UniformMADEPlus(made_module.MADEPlus):
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


class SFEMADEDAG(AbstractSFEModelSampler):
    def __init__(
        self, num_nodes, ig_threshold, lr, init_uniform=True, device=None, dtype=None
    ):
        """
        args:
            num_bits: length of binary string (int)
        """
        super().__init__(ig_threshold, lr, device=device, dtype=dtype)
        # min number of nodes is 2
        assert isinstance(num_nodes, int), "ERROR: integer number of nodes required"
        assert num_nodes >= 2, "ERROR: minimum allowed num nodes is 2"
        # sum of arithmetic sequence up to n, where we start at 2.
        # because last entry has one category, we don't need a random var for that.
        # number of binary features is n(n-1)/2
        # number of categorical features is n-1
        # number of binary logits is same as binary features
        # number of categorical logits is n(1+n)/2 - 1
        # total number of logits output is  = (2n^2 + 2n)/2 = n^2 + n.
        # permutation matrix features (categorical vars of decreasing d)
        P_log_prob_outputs = int(num_nodes * (num_nodes + 1) // 2 - 1)
        P_features = int(num_nodes - 1)
        # upper triangular matrix features (binary vars)
        U_features = int(num_nodes * (num_nodes - 1) // 2)
        U_log_prob_outputs = U_features
        features = P_features + U_features
        #hidden_features = int(upper_bound_power_of_2(2 * features))  # Tune this, or make init param.
        #hidden_features = int(upper_bound_power_of_2(2*(P_log_prob_outputs+U_log_prob_outputs)))  # Tune this, or make init param.
        hidden_features = int(upper_bound_power_of_2((P_log_prob_outputs+U_log_prob_outputs)))  # Tune this, or make init param.

        # Register them as buffers, setting device and dtype
        self.register_buffer(
            "num_nodes", torch.tensor(num_nodes, device=self.device, dtype=torch.int32)
        )
        self.register_buffer(
            "P_log_prob_outputs",
            torch.tensor(P_log_prob_outputs, device=self.device, dtype=torch.int32),
        )
        self.register_buffer(
            "P_features",
            torch.tensor(P_features, device=self.device, dtype=torch.int32),
        )
        self.register_buffer(
            "U_features",
            torch.tensor(U_features, device=self.device, dtype=torch.int32),
        )
        self.register_buffer(
            "U_log_prob_outputs",
            torch.tensor(U_log_prob_outputs, device=self.device, dtype=torch.int32),
        )
        self.register_buffer(
            "features", torch.tensor(features, device=self.device, dtype=torch.int32)
        )
        self.register_buffer(
            "hidden_features", torch.tensor(hidden_features, device=self.device, dtype=torch.int32)
        )

        self.dag_made_output_multiplier = (
            lambda i: num_nodes - i if i < num_nodes - 1 else 1
        )

        mmclass = UniformMADEPlus if init_uniform else made_module.MADEPlus

        # number of blocks should be determined by an empirical result
        # At this stage we lack sufficient experimental evidence to guide us
        # so we will trial setting the number of blocks to reflect a function
        # of the number of topological orderings.
        # As we have d-1 categorical variables representing the topological orderings

        if False:
            import math
            nb_U = max(2,int(math.ceil(math.log2(num_nodes*(num_nodes-1)//2))))
            num_blocks = nb_U + (num_nodes-1)//2 # tune. Maybe more, maybe 1 less.
        elif True:
            num_blocks = 2 + (num_nodes-1)//2 # tune. Maybe more, maybe 1 less.
        else:
            import math
            num_blocks = max(2,int(math.ceil(math.log2(num_nodes))))

        logging.info(f"Constructing SFE-DAG MADE neural distribution with {features} inputs, {hidden_features} hidden features, and {P_log_prob_outputs+U_log_prob_outputs} outputs")
        self.made = mmclass(
            features=features,
            hidden_features=hidden_features,
            context_features=None,
            num_blocks=num_blocks,
            output_multiplier_fn=self.dag_made_output_multiplier,
            #activation=torch.nn.functional.leaky_relu,
            activation=torch.nn.functional.relu,
            #activation=torch.nn.functional.hardtanh,
            #use_batch_norm=True,
        ).to(self.device)

        self._setup_optimizer()

    def _estimate_entropy(self, inputs):
        with torch.no_grad():
            logits = self.made(inputs)
            log_p = self._log_prob_from_logits(inputs, logits)
            entropy_estimate = torch.mean(-log_p)
        return entropy_estimate
        #return self._estimate_entropy_from_update(inputs, self.made)


    def _estimate_entropy_from_logits(self, inputs, prev_logits, new_logits):
        """
        Core importance-sampling formula, given old logits (prev_logits)
        and new logits (new_logits). Called by _estimate_entropy_update().
        """
        with torch.no_grad():
            log_p_old = self._log_prob_from_logits(inputs, prev_logits)
            log_p_new = self._log_prob_from_logits(inputs, new_logits)

            if self.dtype == torch.float32:
                # FP32: Use log-space computations for improved stability.
                a = log_p_new - log_p_old  # Difference in log probabilities.
                # Compute -log_p_new; ensure we don't take log(0) by clamping to a small positive value.
                b = torch.clamp(-log_p_new, min=1e-12)
                # Log of the product: log(x_i) = a + log(b)
                log_product = a + torch.log(b)
                # Flatten if needed
                log_product_flat = log_product.view(-1)
                n = log_product_flat.numel()
                # Compute log of the mean using the log-sum-exp trick:
                # log(mean) = logsumexp(log_product) - log(n)
                log_mean = torch.logsumexp(log_product_flat, dim=0) - torch.log(
                    torch.tensor(n, dtype=self.dtype, device=log_product.device)
                )
                entropy_estimate = torch.exp(log_mean)
            else:
                # FP64: Use the original direct computation.
                w = torch.exp(log_p_new - log_p_old)
                neg_log_p_new = -log_p_new
                entropy_estimate = torch.mean(w * neg_log_p_new)
        return entropy_estimate

    def _estimate_entropy_update(self, inputs, lr, grads):
        """
        Instead of copying the entire model, we:
          1) Do a forward pass to get old (prev) logits from self.made
          2) Backup current parameters
          3) In-place update: p -= lr*g
          4) Forward pass for new logits
          5) Restore old parameters
          6) Compute the importance-sampling estimate of entropy
             by calling _estimate_entropy_from_logits()
        """
        with torch.no_grad():
            # 1) old logits
            prev_logits = self.made(inputs)

            # 2) backup
            old_params = [p.detach().clone() for p in self.made.parameters()]

            # 3) in-place update
            for p, g in zip(self.made.parameters(), grads):
                p -= lr * g

            # 4) new logits
            new_logits = self.made(inputs)

            # 5) revert old params
            for p, oldp in zip(self.made.parameters(), old_params):
                p.copy_(oldp)

        # 6) compute entropy
        return self._estimate_entropy_from_logits(inputs, prev_logits, new_logits)

    def action_sample(self, batch_size):
        return self._sample(batch_size)

    def _sample(self, batch_size):
        with torch.no_grad():
            string = torch.zeros((batch_size, self.features), device=self.device, dtype=self.dtype)

            offset = 0  # offset for output logits

            # Sequentially sample categoricals from MADE
            num_cats = self.num_nodes - 1
            for i in range(num_cats):
                logits = self.made(string)
                length = self.num_nodes - i
                dist = Categorical(logits=logits[:, offset : offset + length])
                string[:, i] = dist.sample()
                offset += length

            # at this point, offset should be self.P_log_prob_outputs
            #assert (
            #    offset == self.P_log_prob_outputs
            #), "Wrong offset, expected {}, got {}".format(
            #    self.P_log_prob_outputs, offset
            #)

            # Sequentially sample Bernoullis from MADE
            num_bernoullis = self.num_nodes * (self.num_nodes - 1) // 2
            for i in range(num_cats, num_cats + num_bernoullis):
                # Compute logits based on current string
                logits = self.made(string)
                # use Bernoulli dist with logits to avoid numerical errors
                dist = Bernoulli(logits=logits[:, offset])
                # Sample bit from Bernoulli distribution
                # Update the string with the sampled bit
                string[:, i] = dist.sample()
                offset += 1

        return string

    def print_probabilities(self):
        """
        Sample this dist and estimate probs
        """
        assert self.num_nodes <= 5, "ERROR: Don't be silly. Too many models."

        # TODO move below to LinearDAG class
        def mk2A(mk):
            from vti.dgp.lineardag import LinearDAG

            P = LinearDAG._cat_to_perm_matrix(mk[:, : self.P_features])
            Pt = P.transpose(1, 2)
            d = self.P_features + 1
            U = LinearDAG.build_full_matrix_triu_indices(
                mk.shape[0], d, mk[:, self.P_features :]
            )
            A = Pt.bmm(U).bmm(P)
            return A

        n = 10000
        samples, lp = self.sample_and_log_prob(n)
        unique_samples, counts_samples = samples.unique(return_counts=True, dim=0)

        A = mk2A(samples)
        A = A.reshape(A.shape[0], -1)
        unique_A, counts_A = A.unique(return_counts=True, dim=0)
        logging.info("Mk probs = ", torch.column_stack([unique_A, counts_A / float(n)]))
        return unique_samples, counts_samples / float(n)

    def _log_prob_from_logits(self, string, logits):
        """
        Computes log p(z) for the given samples 'string' and corresponding 'logits' in a vectorized manner.
        """

        batch_size = string.size(0)
        device = string.device

        # Categorical log probs
        cat_log_probs = []
        offset = 0
        # Categorical variables: P_features = num_nodes - 1
        for i in range(self.P_features):
            length = self.num_nodes - i
            cat_logits = logits[:, offset : offset + length]  # [B, length]
            # Index of chosen category
            chosen = string[:, i].long().unsqueeze(-1)  # [B,1]
            chosen_logits = torch.gather(cat_logits, 1, chosen).squeeze(-1)  # [B]

            # stable log_softmax
            max_logits, _ = torch.max(cat_logits, dim=1, keepdim=True)
            logsumexp = max_logits.squeeze(-1) + torch.log(
                torch.sum(torch.exp(cat_logits - max_logits), dim=1)
            )
            cat_log_probs.append(chosen_logits - logsumexp)

            offset += length

        cat_log_probs = (
            torch.stack(cat_log_probs, dim=1).sum(dim=1)
            if cat_log_probs
            else torch.zeros(batch_size, device=device)
        )

        # Bernoulli log probs (remaining are U_features)
        bern_logits = logits[:, offset:]  # [B, U_features]
        bern_targets = string[:, self.P_features :]  # [B, U_features]
        # For Bernoulli:
        # log p(z_i) = z_i * log_sigmoid(logit_i) + (1-z_i)*log_sigmoid(-logit_i)
        # we can use stable binary cross entropy:
        # BCE = - [z_i*log_sigmoid(logit_i) + (1-z_i)*log_sigmoid(-logit_i)]
        # => log p(z_i) = -BCE
        bern_bce = F.binary_cross_entropy_with_logits(
            bern_logits, bern_targets, reduction="none"
        )
        bern_log_probs = -bern_bce.sum(dim=1)

        # logging.info(f"clp {cat_log_probs.shape} blp {bern_log_probs.shape}")

        return cat_log_probs + bern_log_probs

    def _log_prob(self, string):
        logits = self.made(string)
        return self._log_prob_from_logits(string, logits)

    def log_prob(self, string):
        """
        public facing method
        """
        return self._log_prob(string)

    def action_sample_and_log_prob(self, batch_size):
        string = self._sample(batch_size)
        log_probs = self._log_prob(string)
        return string, log_probs
