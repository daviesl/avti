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
from nflows.transforms.base import Transform
from nflows.utils import torchutils
import logging


class TransdimensionalFlow(nn.Module):
    """
    Purpose of this class is just for NLL evaluation
    """

    def __init__(self, q_mk, theta_transform, base_dist, device=None, dtype=None):
        super().__init__()
        # TODO check types and assert device and dtype match all inputs
        self.base_dist = base_dist
        self.q_mk = q_mk
        self.theta_transform = theta_transform
        self.device = device
        self.dtype = dtype

    #def _sample(self, batch_size, dim, mk_to_context=lambda x: x):
    #    mk_samples = self.q_mk.sample(batch_size).to(dtype=self.dtype)
    #    #logging.info(f'mk samples {mk_samples}')
    #    base_samples = self.base_dist.sample((batch_size, dim))
    #    #logging.info(
    #    #    f"base_samples {base_samples.shape} mk {mk_samples.shape} {mk_to_context(mk_samples).shape}"
    #    #)
    #    theta_samples, __ = self.theta_transform.inverse(
    #        base_samples, context=mk_to_context(mk_samples)
    #    )
    #    return mk_samples, theta_samples

    def _sample(self, batch_size, dim, mk_to_context=lambda x: x, CHUNK_SIZE=1024):
        total_mk_samples = []
        total_theta_samples = []
    
        for start in range(0, batch_size, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, batch_size)
            current_batch_size = end - start
    
            mk_samples = self.q_mk.sample(current_batch_size).to(dtype=self.dtype)
            base_samples = self.base_dist.sample((current_batch_size, dim))
    
            theta_samples, __ = self.theta_transform.inverse(
                base_samples, context=mk_to_context(mk_samples)
            )
    
            total_mk_samples.append(mk_samples)
            total_theta_samples.append(theta_samples)
    
        # Concatenate all collected samples along the first dimension
        final_mk_samples = torch.cat(total_mk_samples, dim=0)
        final_theta_samples = torch.cat(total_theta_samples, dim=0)
    
        return final_mk_samples, final_theta_samples
    

    def _model_log_prob(self, mk):
        return self.q_mk.log_prob(mk)

    def _cond_param_log_prob(self, mk, theta, mk_to_context=lambda x: x):
        base_samples, log_prob_theta_tf = self.theta_transform.forward(
            theta, context=mk_to_context(mk)
        )
        log_prob_base = torchutils.sum_except_batch(self.base_dist.log_prob(base_samples), num_batch_dims=1)
        return log_prob_base + log_prob_theta_tf

    def log_prob(self, mk, theta, mk_to_context=lambda x: x):
        log_prob_mk = self._model_log_prob(mk)
        log_prob_theta = self._cond_param_log_prob(mk, theta, mk_to_context)
        return log_prob_theta + log_prob_mk
        # Note: log prob is on saturated space,
        # need to substract reference log prob downstream
