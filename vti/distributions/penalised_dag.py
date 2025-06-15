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
from torch.nn import functional as F
from vti.distributions import PermutationDAGUniformDistribution

DEBUGMODE=False

class PermutationDAGPenalizedDistribution(PermutationDAGUniformDistribution):
    """
    A prior over (P, U) that is uniform over permutations and uniform over U
    except for an additional exponential penalty on the number of active edges.
    p(U) ~ (1/2^#U_features) * exp(-gamma * sum(U_bits))
    p(P) ~ 1/d!  (only if you keep P_log_prob from the parent)
    """

    def __init__(self, num_nodes, gamma=0.0, device=None, dtype=None):
        super().__init__(num_nodes, device, dtype)
        self.gamma = torch.tensor(gamma, device=device, dtype=dtype)
        # If gamma=0 => uniform. If gamma>0 => penalize edges.

    def log_prob(self, inputs):
        # same shape check
        if DEBUGMODE:
            assert (
                inputs.shape[1] == self.P_features + self.U_features
            ), "Feature mismatch, expected {}, got {}".format(
                self.P_features + self.U_features, inputs.shape[1]
            )
        # Partition into P-cats and U-bins
        # parent uses self.flat_U_dist for the uniform bits
        U_log_prob = self.flat_U_dist.log_prob(inputs[:, self.P_features :])

        # parent also has self.P_log_prob as the sum of log(1/(num_nodes-i)) => log(1/d!)
        log_p_perm = self.P_log_prob

        # count edges
        U_bin = inputs[:, self.P_features :]  # shape (batch_size, U_features)
        # sum along dimension 1 => number of active edges in each sample
        edges_count = U_bin.sum(dim=1)

        # penalty = - gamma * edges_count
        penalty_term = -self.gamma * edges_count

        # total log prob
        return U_log_prob + log_p_perm + penalty_term
