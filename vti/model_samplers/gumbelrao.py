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

"""
Gumbel-Rao distribution, based on the paper by Paulus et al 2021 ICLR:
  [1] https://openreview.net/pdf?id=Mk6PZtgAgfq
  [1a] https://arxiv.org/abs/2010.04838
Adapted from https://github.com/nshepperd/gumbel-rao-pytorch

"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import OneHotCategorical, Exponential, RelaxedOneHotCategorical


class GumbelRao(nn.Module):
    def __init__(self, num_categories, prior_logits, k=1, device=None, dtype=None):
        super().__init__()
        self.num_categories = num_categories
        self.k = k
        self.logits = nn.Parameter(
            torch.empty((num_categories,), device=device, dtype=dtype)
        )
        self.prior_logits = prior_logits

    def _replace_gradient(self, value, surrogate):
        """Returns `value` but backpropagates gradients through `surrogate`."""
        return surrogate + (value - surrogate).detach()

    def _conditional_gumbel(self, D, k=1):
        """Outputs k samples of Q = StandardGumbel(), such that argmax(logits
        + Q) is given by D (one hot vector)."""

        with torch.no_grad():
            # iid. exponential
            batchsize = D.shape[0]
            E = Exponential(rate=torch.ones_like(self.logits)).sample([batchsize, k])
            # E of the chosen class
            D = D.unsqueeze(1)
            Ei = (D * E).sum(dim=-1, keepdim=True)
            # partition function (normalization constant)
            Z = self.logits.exp().sum(dim=-1, keepdim=True)
            # Sampled gumbel-adjusted logits
            adjusted = D * (-torch.log(Ei) + torch.log(Z)) + (1 - D) * -torch.log(
                E / torch.exp(self.logits) + Ei / Z
            )
            res = adjusted - self.logits
        return res

    def _rsample(self, num_samples, k=1, temp=1.0, dist=None):
        # I = torch.distributions.categorical.Categorical(logits=self.logits).sample(num_samples)
        # D = torch.nn.functional.one_hot(I, self.num_categories).to(dtype=self.logits.dtype)
        if dist is None:
            dist = OneHotCategorical(logits=self.logits)
        D = dist.sample(num_samples)
        adjusted = self.logits + self._conditional_gumbel(D, k=k)
        surrogate = F.softmax(adjusted / temp, dim=-1).mean(dim=1)
        return self._replace_gradient(D, surrogate)

    def sample_and_log_prob(self, num_samples, temperature):
        logits = self.logits - self.logits.logsumexp(dim=0, keepdim=True)
        # dist = OneHotCategorical(logits=self.logits)
        temperature = torch.as_tensor(
            temperature, dtype=logits.dtype, device=logits.device
        )
        dist = RelaxedOneHotCategorical(logits=logits, temperature=temperature)
        relaxed_samples = dist.rsample((num_samples,))
        # samples = self._rsample((num_samples,), k=self.k, temp=temperature, dist=dist)
        # log_prob = dist.log_prob(samples)
        # HACK to make log_prob work
        hacked_soft_sample = relaxed_samples / relaxed_samples.sum(dim=-1, keepdim=True)
        log_prob = dist.log_prob(hacked_soft_sample)

        priordist = RelaxedOneHotCategorical(
            logits=self.prior_logits, temperature=temperature
        )
        prior_log_prob = priordist.log_prob(hacked_soft_sample)

        # D = QuantizeCategorical.apply(relaxed_samples)
        D = QuantizeCategorical.apply(hacked_soft_sample)
        if False:
            # Gumbel Rao
            adjusted = logits + self._conditional_gumbel(D, k=self.k)
            surrogate = F.softmax(adjusted / temperature, dim=-1).mean(dim=1)
            samples = self._replace_gradient(D, surrogate)

            return samples, log_prob - prior_log_prob
        else:
            # straight through gumbel softmax
            return D, log_prob - prior_log_prob


#
# class RelaxedOneHotCategoricalStraightThrough(RelaxedOneHotCategorical):
#    """
#    An implementation of
#    :class:`~torch.distributions.relaxed_categorical.RelaxedOneHotCategorical`
#    with a straight-through gradient estimator.
#
#    This distribution has the following properties:
#
#    - The samples returned by the :meth:`rsample` method are discrete/quantized.
#    - The :meth:`log_prob` method returns the log probability of the
#      relaxed/unquantized sample using the GumbelSoftmax distribution.
#    - In the backward pass the gradient of the sample with respect to the
#      parameters of the distribution uses the relaxed/unquantized sample.
#
#    References:
#
#    [1] The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables,
#        Chris J. Maddison, Andriy Mnih, Yee Whye Teh
#    [2] Categorical Reparameterization with Gumbel-Softmax,
#        Eric Jang, Shixiang Gu, Ben Poole
#    """
# [docs]    def rsample(self, sample_shape=torch.Size()):
#        soft_sample = super().rsample(sample_shape)
#        soft_sample = clamp_probs(soft_sample)
#        hard_sample = QuantizeCategorical.apply(soft_sample)
#        return hard_sample
#
#
# [docs]    def log_prob(self, value):
#        value = getattr(value, '_unquantize', value)
#        return super().log_prob(value)
#
#
#
class QuantizeCategorical(torch.autograd.Function):
    @staticmethod
    def forward(ctx, soft_value):
        argmax = soft_value.max(-1)[1]
        hard_value = torch.zeros_like(soft_value)
        hard_value._unquantize = soft_value
        if argmax.dim() < hard_value.dim():
            argmax = argmax.unsqueeze(-1)
        return hard_value.scatter_(-1, argmax, 1)

    @staticmethod
    def backward(ctx, grad):
        return grad


#
