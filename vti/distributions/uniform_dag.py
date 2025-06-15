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
from vti.distributions.uniform_bs import UniformBinaryString

DEBUGMODE=False

class PermutationDAGUniformDistribution(torch.nn.Module):
    def __init__(self, num_nodes, device, dtype):
        """
        Constructs prior over DAGs using
        permutation P and upper triangular binary U matrices.
        """
        super().__init__()
        assert isinstance(num_nodes, int), "ERROR: num_nodes must be integer"
        self.device = device
        self.dtype = dtype
        self.num_nodes = num_nodes
        self.U_features = int(num_nodes * (num_nodes - 1) // 2)
        self.P_features = int(num_nodes - 1)
        self.flat_U_dist = UniformBinaryString(self.U_features, device, dtype)
        # precompute the uniform categorical probs
        cat_log_probs = torch.zeros(
            (self.P_features,), device=self.device, dtype=self.dtype
        )
        offset = 0
        for i in range(self.P_features):
            length = self.num_nodes - i
            # clp = -torch.log(torch.tensor(length, dtype=self.dtype, device=self.device))
            cat_log_probs[i] = -torch.log(
                torch.tensor(length, dtype=self.dtype, device=self.device)
            )
        self.P_log_prob = cat_log_probs.sum()

    def log_prob(self, inputs):
        if DEBUGMODE:
            assert (
                inputs.shape[1] == self.P_features + self.U_features
            ), "Feature mismatch, expected {}, got {}".format(
                self.P_features + self.U_features, inputs.shape[1]
            )
        # the below would change for a different prior on the number of edges
        U_log_prob = self.flat_U_dist.log_prob(inputs[:, self.P_features :])
        # sum of log probs for U and P
        return U_log_prob + self.P_log_prob
