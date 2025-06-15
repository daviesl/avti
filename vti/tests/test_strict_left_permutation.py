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
import pytest

# import the class under test
from vti.transforms.strict_left_permutation import StrictLeftPermutation   # adjust path if needed


@pytest.fixture(scope="module")
def example_tensors():
    """
    Returns
    -------
    context : (2, 7) float tensor
    inputs  : (2, 7) float tensor, random but reproducible
    """
    torch.manual_seed(0)
    context = torch.tensor([[1., 1., 0., 1., 0., 0., 1.],
                            [0., 1., 1., 1., 1., 0., 0.]])
    inputs = torch.randn((2, 7))
    return context, inputs


def expected_permutation(context):
    """
    Compute the reference permutation shown in the problem statement.
    """
    return torch.tensor([[0, 1, 3, 6, 2, 4, 5],
                         [1, 2, 3, 4, 0, 5, 6]], dtype=torch.long)


def test_forward_inverse_roundtrip(example_tensors):
    context, inputs = example_tensors
    perm_transform = StrictLeftPermutation(inputs.shape[1], dim=1)

    # --- forward -----------------------------------------------------------
    outputs, logdet_fwd = perm_transform.forward(inputs, context)

    # forward must equal inputs gathered with the reference permutation
    ref_perm = expected_permutation(context)
    expected_outputs = inputs.gather(1, ref_perm)
    assert torch.allclose(outputs, expected_outputs), "Forward permutation incorrect"

    # log-|det| must be zero-vector
    assert torch.allclose(logdet_fwd, torch.zeros_like(logdet_fwd)), "log|det J| not zero"

    # --- inverse -----------------------------------------------------------
    recovered, logdet_inv = perm_transform.inverse(outputs, context)

    # recovered tensor must match the original inputs
    assert torch.allclose(recovered, inputs), "Inverse did not restore original inputs"

    # inverse log-|det| also zero
    assert torch.allclose(logdet_inv, torch.zeros_like(logdet_inv)), "inverse log|det J| not zero"

