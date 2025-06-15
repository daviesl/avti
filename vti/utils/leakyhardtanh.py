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
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


# definition of the leaky hardtanh function
def leaky_hardtanh(inputs, min_val=- 1.0, max_val=1.0, slope=0.01):
    '''
        Defines the Leaky Hardtanh function
    '''
    # Below 7x slower than leaky relu due to where function
    #return torch.where(inputs < max_val, F.leaky_relu(inputs-min_val, slope)+min_val, (inputs-max_val)*slope+max_val)
    y = torch.clamp(inputs, min=min_val, max=max_val)
    return y + slope * (inputs - y)

  
# create a class wrapper from PyTorch nn.Module
class LeakyHardtanh(nn.Module):
    r"""Applies the Leaky HardTanh function element-wise.

    Leaky HardTanh is defined as:

    .. math::
        \text{LeakyHardTanh}(x) = \begin{cases}
            (x - \text{max\_val}) \times \text{slope} +  \text{max\_val} & \text{ if } x > \text{ max\_val } \\
            (x - \text{min\_val}) \times \text{slope} +  \text{min\_val} & \text{ if } x < \text{ min\_val } \\
            x & \text{ otherwise } \\
        \end{cases}

    Args:
        min_val: minimum value of the linear region range. Default: -1
        max_val: maximum value of the linear region range. Default: 1
        slope: Controls the angle of the region outside [min_val,max_val]. Default: 0.01

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.LeakyHardtanh(-2, 2, 0.001)
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    def __init__(self, min_val: float =- 1.0, max_val: float=1.0, slope: float=0.01):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.slope = slope
        assert self.max_val > self.min_val, "max_val must be larger than min_val"

    def forward(self, inputs: Tensor) -> Tensor:
        return leaky_hardtanh(inputs, self.min_val, self.max_val, self.slope)
