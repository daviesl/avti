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
import logging


def move_optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        if isinstance(state, dict):
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)


def print_gradient_l2_norm(model):
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    logging.info(f"L2 norm of the gradients: {total_norm}")


# Example usage:
# Assume `model` is your neural network model and you have done a backward pass
# print_gradient_l2_norm(model)


def ensure_dtype(dtype):
    # If it's a string, convert it to the corresponding torch.dtype
    if isinstance(dtype, str):
        return getattr(torch, dtype)
    # If it's already a torch.dtype, return it as-is
    elif isinstance(dtype, torch.dtype):
        return dtype
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def ensure_device(device=None):
    if device is None:
        # Return the default device (could be extended to return 'cuda' if available)
        return torch.device("cpu")  # or torch.device('cuda') if preferred by default
    return torch.device(device)  # Cast string or pass-through torch.device


def extract_adam_step_moments(optimizer):
    """
    Extracts the  step and gradient moments from the Adam optimizer.
    """
    # Initialize lists to store the first moment, second moment, and effective step for all parameters
    first_moments = []
    second_moments = []
    effective_steps = []

    # Access and compute the effective step for each parameter after scheduling
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is None:
                continue

            # Access the optimizer state for the current parameter
            state = optimizer.state[param]

            # Get the first and second moment estimates (exp_avg, exp_avg_sq)
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            step = state["step"]

            # Extract the updated learning rate from the optimizer (after scheduler adjustment)
            lr = group["lr"]  # This reflects the scheduled learning rate now
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            # Bias correction terms
            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step

            # Compute the effective step size
            denom = (exp_avg_sq.sqrt() / bias_correction2.sqrt()) + eps
            step_size = lr / bias_correction1

            # Effective step applied to the parameter
            effective_step = step_size * exp_avg / denom

            # Collect the moments and effective step for this parameter
            first_moments.append(exp_avg.view(-1))  # Flatten and store
            second_moments.append(exp_avg_sq.view(-1))  # Flatten and store
            effective_steps.append(effective_step.view(-1))  # Flatten and store

    # Concatenate all parameters into single vectors
    first_moment_vector = torch.cat(first_moments)
    second_moment_vector = torch.cat(second_moments)
    step_vector = torch.cat(effective_steps)
    return step_vector, first_moment_vector, second_moment_vector
