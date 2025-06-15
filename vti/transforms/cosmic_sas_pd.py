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

import math
import torch
from torch import nn
from torch.nn import functional as F

from nflows.transforms.base import Transform
from nflows.utils import torchutils
import nflows.utils.typechecks as check
from vti.transforms.sas import *


def inverse_softplus(x):
    return x + math.log(-math.expm1(-x))


class CoSMICSinhArcSinhTransform(Transform):
    def __init__(
        self,
        features,
        context_to_mask,
        context_encoder_depth,
        context_encoder_features,
        context_encoder_hidden_features,
        context_activation=nn.ReLU(),
        context_transform=lambda x:x,
        myname="generic",
        device=None,
        dtype=None,
    ):
        if not check.is_positive_int(features):
            raise TypeError("Number of features must be a positive integer.")
        if not check.is_positive_int(context_encoder_depth):
            raise TypeError("Depth of context encoder must be a positive integer.")
        if not check.is_positive_int(context_encoder_hidden_features):
            raise TypeError(
                "Number of context hidden features must be a positive integer."
            )
        super().__init__()

        self.myname = myname
        self.register_buffer('_eps', torch.tensor(1e-6))

        self.features = features
        self._shape = torch.Size([features])

        bargs = [
            nn.Linear(
                context_encoder_features,
                context_encoder_hidden_features,
                #  device=device,
                dtype=dtype,
            ),
            context_activation,
        ]
        for i in range(2):
            bargs += [
                nn.Linear(
                    context_encoder_hidden_features,
                    context_encoder_hidden_features,
                    # device=device,
                    dtype=dtype,
                ),
                context_activation,
            ]
        bargs += [
            nn.Linear(
                context_encoder_hidden_features,
                self._output_dim_multiplier() * features,
                # device=device,
                dtype=dtype,
            )
        ]
        self.context_encoder = nn.Sequential(*bargs)
        self.ctm = self._inputs_to_context_mask(
            context_to_mask
        )  # lambda function that converts context to a mask
        self.ctf = context_transform


    def forward(self, inputs, context=None):
        context_params = self.context_encoder(self.ctf(context))
        if context is not None:
            mask = self.ctm(context)
            context_params = (1 - mask) * self._static_point(
                context_params
            ) + mask * context_params
        outputs, logabsdet = self._elementwise_forward(inputs, context_params)
        # print("sas forward context",context_params)
        # print("sas forward outputs,logabsdet",outputs,outputs.shape,logabsdet.shape)
        return outputs, torchutils.sum_except_batch(logabsdet, num_batch_dims=1)

    def inverse(self, inputs, context=None):
        num_inputs = int(torch.tensor(inputs.shape[1:]).prod())
        logabsdet = None
        device = inputs.device
        dtype = inputs.dtype
        if context is not None:
            mask = self.ctm(context)
        else:
            mask = self._get_open_mask(num_inputs, device=device, dtype=dtype)
        context_params = self.context_encoder(self.ctf(context))
        context_params = (1 - mask) * self._static_point(
            context_params
        ) + mask * context_params
        outputs, logabsdet = self._elementwise_inverse(inputs, context_params)
        # print("sas inverse outputs,logabsdet",outputs.shape,logabsdet.shape)
        return outputs, torchutils.sum_except_batch(logabsdet, num_batch_dims=1)

    def _get_open_mask(self, num_inputs, device=None, dtype=None):
        return torch.ones(
            num_inputs * self._output_dim_multiplier(), device=device, dtype=dtype
        )

    def _output_dim_multiplier(self):
        return 2

    def _elementwise_forward(self, inputs, context_params):
        epsilon, unconstrained_delta = self._contex_params_as_sas_params(context_params)
        delta = F.softplus(unconstrained_delta) + self._eps
        return sas_forward(inputs, epsilon, delta)

    def _elementwise_inverse(self, inputs, context_params):
        epsilon, unconstrained_delta = self._contex_params_as_sas_params(context_params)
        delta = F.softplus(unconstrained_delta) + self._eps
        return sas_inverse(inputs, epsilon, delta)

    def _static_point(self, context_params):
        epsilon = 0
        unconstrained_delta = inverse_softplus(1 - self._eps)
        a = torch.zeros_like(context_params)
        aview = a.view(-1, self.features, self._output_dim_multiplier())
        aview[..., 0] = epsilon
        aview[..., 1] = unconstrained_delta
        return a

    def _inputs_to_context_mask(self, ctm):
        return lambda context: ctm(context).repeat_interleave(
            self._output_dim_multiplier(), dim=-1
        )

    def _contex_params_as_sas_params(self, context_params):
        context_params = context_params.view(
            -1, self.features, self._output_dim_multiplier()
        )
        epsilon = context_params[..., 0]
        unconstrained_delta = context_params[..., 1]
        return epsilon, unconstrained_delta
