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

from nflows.transforms.base import Transform
import torch


def _sas(x, epsilon, delta):
    return torch.sinh((torch.arcsinh(x) + epsilon) / delta)


def _isas(x, epsilon, delta):
    return torch.sinh(delta * torch.arcsinh(x) - epsilon)


def _ldisas(x, epsilon, delta):
    return torch.log(
        torch.abs(
            delta
            * torch.cosh(epsilon - delta * torch.arcsinh(x))
            / torch.sqrt(1 + x**2)
        )
    )


def sas_forward(X, epsilon, delta):
    XX = _sas(X, epsilon, delta)
    ld = -_ldisas(XX, epsilon, delta)
    # ld = ld.flatten()
    return XX, ld


def sas_inverse(X, epsilon, delta):
    ld = _ldisas(X, epsilon, delta)
    # ld = ld.flatten()
    return _isas(X, epsilon, delta), ld


class SinhArcSinhTransform(Transform):
    def __init__(self, e, d):
        super().__init__()
        self.epsilon = e
        self.delta = d

    def forward(self, X, context=None):
        outputs, ld = sas_forward(X, self.epsilon, self.delta)
        return outputs, ld.flatten()  # ld.sum(dim=-1)

    def inverse(self, X, context=None):
        outputs, ld = sas_inverse(X, self.epsilon, self.delta)
        return outputs, ld.flatten()  # ld.sum(dim=-1) # TODO sumexceptbatch?
