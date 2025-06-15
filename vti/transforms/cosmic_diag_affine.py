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
import numpy as np

from nflows.transforms.base import Transform
from nflows.utils import torchutils
import nflows.utils.typechecks as check
from vti.utils.math_helpers import inverse_softplus



class CoSMICDiagonalAffineTransform(Transform):
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
        if not check.is_nonnegative_int(context_encoder_depth):
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

        if context_encoder_depth == 0:
            bargs = [
                nn.Linear(
                    context_encoder_features,
                    self._output_dim_multiplier() * features,
                    #  device=device,
                    dtype=dtype,
                )
            ]
        elif True:
            bargs = [
                nn.Linear(
                    context_encoder_features,
                    context_encoder_hidden_features,
                    #  device=device,
                    dtype=dtype,
                ),
                context_activation,
            ]
            for i in range(context_encoder_depth - 1):
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
        else:
            # trapezoid
            cehf = [
                int(l)
                for l in np.geomspace(
                    context_encoder_features,
                    self._output_dim_multiplier() * features,
                    num=context_encoder_depth + 2,
                )
            ]
            print("cehf", cehf)
            bargs = []
            for i in range(0, context_encoder_depth + 1):
                bargs += [nn.Linear(cehf[i], cehf[i + 1])]

        self.context_encoder = nn.Sequential(*bargs)
        self.ctm = self._inputs_to_context_mask(
            context_to_mask
        )  # lambda function that converts context to a mask
        self.ctf = context_transform

        # print("model params",list(self.parameters()))
        # sys.exit(0)

        self.initialize_uniform()

    def initialize_uniform(self):
        # Only zero out the final layer weights and biases.
        # Leave other layers as-is for better optimization dynamics.
        if isinstance(self.context_encoder[-1], nn.Linear):
                nn.init.constant_(self.context_encoder[-1].weight, 0.0)
                if self.context_encoder[-1].bias is not None:
                    nn.init.constant_(self.context_encoder[-1].bias, 0.0)

    def forward(self, inputs, context=None):
        if context is not None:
            if False:
                test_context = torch.tensor(
                    [[1, 0], [0, 1]], dtype=input.dtype, device=input.device
                )
                test_cp = self.context_encoder(test_context)
                print("context params test", test_context, test_cp)
            context_params = self.context_encoder(self.ctf(context))
            mask = self.ctm(context)
            context_params = (1 - mask) * self._static_point(
                context_params
            ) + mask * context_params
            outputs, logabsdet = self._elementwise_forward(inputs, context_params)
            # print("outputs",outputs[:5],logabsdet[:5])
            return outputs, logabsdet
        else:
            raise Exception("Context cannot be None")

    def inverse(self, inputs, context=None):
        if context is not None:
            # num_inputs = int(torch.tensor(inputs.shape[1:]).prod())
            mask = self.ctm(context)
            context_params = self.context_encoder(self.ctf(context))
            context_params = (1 - mask) * self._static_point(
                context_params
            ) + mask * context_params
            outputs, logabsdet = self._elementwise_inverse(inputs, context_params)
            # return outputs, torchutils.sum_except_batch(logabsdet, num_batch_dims=1)
            return outputs, logabsdet
        else:
            raise Exception("Context cannot be None")
            # mask = self._get_open_mask(num_inputs)
            # context_params = self._static_point(torch.zeros((inputs.shape[0],num_inputs*self._output_dim_multiplier())))

    def _get_open_mask(self, num_inputs, dtype=None, device=None):
        return torch.ones(
            num_inputs * self._output_dim_multiplier(), device=device, dtype=dtype
        )

    def _output_dim_multiplier(self):
        return 2

    def _elementwise_forward(self, inputs, context_params):
        # print("ewfwf da",self.myname)
        shift, unconstrained_scale = self._contex_params_as_sas_params(context_params)
        #scale = F.softplus(unconstrained_scale) + self._eps
        scale = F.softplus(unconstrained_scale,beta=torch.tensor(2.).log()) + self._eps
        outputs = scale * inputs + shift
        # print("ewfwd scale",scale)
        # print("ewfwd shift",shift)
        logabsdet = torchutils.sum_except_batch(scale.log(), num_batch_dims=1)
        return outputs, logabsdet

    def _elementwise_inverse(self, inputs, context_params):
        # print("ewinv da",self.myname)
        shift, unconstrained_scale = self._contex_params_as_sas_params(context_params)
        #scale = F.softplus(unconstrained_scale) + self._eps
        scale = F.softplus(unconstrained_scale,beta=torch.tensor(2.).log()) + self._eps
        outputs = (inputs - shift) / scale
        logabsdet = torchutils.sum_except_batch(-scale.log(), num_batch_dims=1)
        return outputs, logabsdet

    def _static_point(self, context_params):
        shift = 0
        #unconstrained_scale = inverse_softplus(1 - self._eps)
        unconstrained_scale = inverse_softplus(1 - self._eps,beta=torch.tensor(2.).log())
        a = torch.zeros_like(context_params)
        aview = a.view(-1, self.features, self._output_dim_multiplier())
        aview[..., 0] = shift
        aview[..., 1] = unconstrained_scale
        return a

    def _inputs_to_context_mask(self, ctm):
        return lambda context: ctm(context).repeat_interleave(
            self._output_dim_multiplier(), dim=-1
        )

    def _contex_params_as_sas_params(self, context_params):
        context_params = context_params.view(
            -1, self.features, self._output_dim_multiplier()
        )
        shift = context_params[..., 0]
        unconstrained_scale = context_params[..., 1]
        return shift, unconstrained_scale
