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

from vti.utils.debug import check_for_nans

"""
Samplers are a bit like random distributions, but they use an adaptive surrogate model to draw their samples.
"""


class AbstractModelSampler(nn.Module):
    """
    Abstract Sampler class, which models the logits of a categorical distribution.
    """

    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.dtype = dtype or torch.float64

    def logits(self):
        """
        Falls back to action logits per default
        """
        return self.action_logits()

    def probs(self):
        raise NotImplementedError("probs not implemented")

    def action_probs(self):
        raise NotImplementedError("action_probs not implemented")

    def sample(self, batch_size=1):
        """
        Falls back to action sample per default
        """
        return self.action_sample(batch_size)

    def sample_and_log_prob(self, batch_size=1):
        return self.action_sample_and_log_prob(batch_size)

    def log_prob(self, samples):
        raise NotImplementedError("{__class__.__name__}.log_prob() not implemented")

    def action_dist(self):
        """
        We can realise these logits as a Categorical distribution.
        (although perhaps not all samplers will admit that?)
        """
        # return Categorical(logits=self.action_logits())
        raise NotImplementedError("No action distribution is defined")

    def dist(self):
        return self.action_dist()

    def entropy(self):
        return self.dist().entropy()

    def action_entropy(self):
        return self.action_dist().entropy()

    def action_logits(self):
        raise NotImplementedError("action_logits not implemented")

    def action_sample_and_log_prob(self, batch_size):
        raise NotImplementedError("sample and log prob not implemented")

    def action_sample(self, batch_size):
        raise NotImplementedError("action sample not implemented")

    def debug_log(self):
        raise NotImplementedError("debug_log not implemented")

    def observe(self, mk_catsamples, loss_hat, iteration):
        raise NotImplementedError("observe not implemented")

    def evolve(self, mk_cat_samples, ell, optimizer, loss, iteration):
        """
        update the prior based on the loss of the flow model.
        A no-op by default, presuming stationarity.
        """
        pass
