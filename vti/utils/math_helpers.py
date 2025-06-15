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
MIT license

"""

import torch
from torch.autograd import grad

import types


def is_function(obj):
    return isinstance(
        obj, (types.FunctionType, types.BuiltinFunctionType, types.LambdaType)
    )

def log_sigmoid(x):
    return -torch.log(1 + torch.exp(-x))


def inverse_softplus(x, beta=1):
    """
    Computes the inverse of the softplus function using PyTorch operations.
    For a softplus defined as:
        softplus(x) = (1/beta) * log(1 + exp(beta*x)),
    its inverse is given by:
        inverse_softplus(x) = x + (1/beta) * log(1 - exp(-beta*x))
    which can be written using torch.expm1 for numerical stability.
    """
    return x + (1.0 / beta) * torch.log(-torch.expm1(-beta * x))

def inverse_softmax(probs):
    # Ensure the probabilities are valid
    probs = torch.clamp(probs, min=1e-10, max=1 - 1e-10)

    # Calculate the logits (log-odds)
    logits = torch.log(probs)
    return logits


def upper_bound_power_of_2(n):
    """
    WARNING only use for initialisation of nn modules, not during inference.
    """
    import math
    return 2 ** math.ceil(math.log2(max(n, 1)))

def shift_nonzeros_left(tensor):
    # Get a mask of non-zero entries
    non_zero_mask = tensor != 0

    # Flatten non-zero entries and preserve gradients
    flattened_non_zeros = tensor[non_zero_mask]

    # Count non-zero entries in each row
    non_zero_counts = non_zero_mask.sum(dim=1)

    # Prepare a new tensor filled with zeros of the same shape and type as the original
    result = torch.zeros_like(tensor)

    # Get the maximum number of non-zero entries in any row for consistent indexing
    max_non_zeros = non_zero_counts.max()

    # Iterate over each row and set the non-zero entries
    for i in range(tensor.size(0)):
        num_non_zeros = non_zero_counts[i]
        if num_non_zeros > 0:
            result[i, :num_non_zeros] = tensor[i, non_zero_mask[i]]

    return result


def is_scalar(v):
    """
    how is this not a builtin?
    """
    v = torch.as_tensor(v)
    return v.shape == torch.Size([]) or v.shape == torch.Size([1])


def jacobian_factory(func):
    """
    vector jacobians of a function
    This is pretty retro now; modern functorch has better ways.
    """

    def jacobian_func(input_tensor, *func_args, **func_kwargs):
        input_tensor.requires_grad_(True)

        # Call the original function with any additional arguments
        output_tensor = func(input_tensor, *func_args, **func_kwargs)

        # Initialize the Jacobian matrix
        jacobian = torch.zeros(output_tensor.numel(), input_tensor.numel())

        # Compute the Jacobian entries
        for i in range(output_tensor.numel()):
            grad_output = torch.zeros_like(output_tensor)
            grad_output.view(-1)[i] = 1.0
            grad_input = grad(
                output_tensor, input_tensor, grad_outputs=grad_output, retain_graph=True
            )[0]
            jacobian[i, :] = grad_input.view(-1)

        return jacobian

    return jacobian_func


def atol_rtol(dtype, m, n=None, atol=0.0, rtol=None):
    if rtol is not None:
        return atol, rtol
    elif rtol is None and atol > 0.0:
        return atol, 0.0
    else:
        if n is None:
            n = m
        # choose bigger of m, n
        mn = max(m, n)
        # choose based on eps for float type
        eps = torch.finfo(dtype).eps
        return 0.0, eps * mn


def convert_2d_to_1d(array_2d):
    """
    Converts a 2D array or a batch of 2D arrays to a 1D array or a batch of 1D arrays.
    Preserves the batch dimension if present.
    """
    return array_2d.view(*array_2d.shape[:-2], -1)


def convert_1d_to_2d(array_1d):
    """
    Converts a 1D array or a batch of 1D arrays to a 2D square array or a batch of 2D square arrays.
    Preserves the batch dimension if present.
    Automatically deduces the grid size from the size of the 1D array.
    Assumes the length of the 1D array is a perfect square.
    """
    grid_size = int(array_1d.shape[-1] ** 0.5)
    return array_1d.view(*array_1d.shape[:-1], grid_size, grid_size)


def integers_to_binary(int_tensor, dim, dtype):
    # Create a mask for each bit position from 0 to context_dim-1
    masks = 1 << torch.arange(dim - 1, -1, -1, dtype=torch.int64)

    # Apply the mask to the tensor and right shift to bring the masked bit to the least significant position
    binary_matrix = ((int_tensor.unsqueeze(-1) & masks) > 0).to(
        dtype
    )  # Convert boolean to integer

    return binary_matrix


def ensure_2d(tensor):
    tensor = torch.as_tensor(tensor)
    return tensor.view(1, -1) if tensor.dim() == 1 else tensor


def log_normal_pdf(x, mu, sigma):
    """
    Compute the log probability of x under the normal distribution N(mu, sigma^2).
    This is one of many implementations of the log normal pdf used in this codebase.
    Not quite sure why?

    Args:
    x (torch.Tensor): Tensor of values.
    mu (torch.Tensor): Tensor of means.
    sigma (torch.Tensor): Tensor of standard deviations.

    Returns:
    torch.Tensor: Log probability of x under the normal distributions.
    """
    # Ensure sigma is positive
    sigma = torch.abs(sigma)

    # Compute the log probability using the formula for the log of the normal PDF
    log_prob = -0.5 * torch.log(2 * torch.pi * sigma**2) - 0.5 * ((x - mu) / sigma) ** 2
    # print("log_prob",log_prob.shape)

    return log_prob


class SafeLog(torch.autograd.Function):
    """
    SafeLog is a custom PyTorch autograd Function that applies a logarithm to inputs
    in a numerically stable manner. It clamps very small values to a minimum threshold
    (1e-30) before taking the log, preventing potential underflow issues. During the
    backward pass, gradients for inputs that are below the threshold are set to zero,
    further avoiding numerical instability.

    Attributes:
        forward: Clamps input values to the minimum threshold and computes their natural log.
        backward: Computes the gradient by dividing the upstream gradient by the input,
            clamped to the same threshold. Gradients are set to zero for inputs below the threshold.
    """

    @staticmethod
    def forward(ctx, inputs):
        ctx.save_for_backward(inputs)
        return torch.log(inputs.clamp(min=1e-30))

    @staticmethod
    def backward(ctx, grad_output):
        (inputs,) = ctx.saved_tensors
        safe_grad = grad_output / inputs.clamp(min=1e-30)
        safe_grad[inputs < 1e-30] = 0  # Setting gradients exactly at zero input to zero
        return safe_grad


safe_log = SafeLog.apply

