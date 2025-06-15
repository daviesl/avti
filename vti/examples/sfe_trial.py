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

from pathlib import Path
from time import time

import torch

from vti.utils.seed import set_seed
from vti.infer import VTIMCGEstimator
from vti.utils.torch_nn_helpers import ensure_dtype, ensure_device
from vti.model_samplers import (
    SFECategorical,
    SFECategoricalBinaryString,
    SFEMADEBinaryString,
    SFEMADEDAG,
)
from vti.dgp import create_dgp_from_key

# from vti.utils.math_helpers import is_function
from vti.dgp import dgp_seed_fns
import logging


def sfe_trial(
    ## inference parameters
    dgp_key,  # preset configuration for data generation and problem type
    dgp_seed,  # seed for data generation
    model_sampler_key,  # either SFECategorical, SFEMADEBinaryString, or SFEMADEDAG
    seed,  # optimizer seed
    num_iterations=1000,
    batch_size=32,
    # resume=None,
    flow_type="affine5",
    ig_threshold=1e-3,  # we suspect this is related to max entropy gain
    squish_utility=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    dtype="float64",
    ## Utility parameters
    output_dir="___output",
    grad_norm_clip=20.0,
    plot=False,
    job_id=None,  # unused
    smoke_test=False,
    ## extra parameters
    **kwargs,
):
    """
    Same thing as surrogate trial but for SFE.
    """
    if smoke_test:
        num_iterations = min(num_iterations, 1000)

    device = ensure_device(device)
    dtype = ensure_dtype(dtype)

    torch.set_default_dtype(dtype)
    torch.set_default_device(device)

    # if is_function(dgp_seed):
    #    dgpseedval = dgp_seed(seed) # function of optimizer seed
    # else:
    #    dgpseedval = dgp_seed
    if isinstance(dgp_seed, str):
        if dgp_seed not in dgp_seed_fns.keys():
            raise NotImplementedError(
                f"DGP seed function {dgp_seed} not found. Available are {list(dgp_seed_fns.keys())}"
            )
        dgpseedval = dgp_seed_fns[dgp_seed](seed)
        logging.info(f"DGP seed is set to {dgpseedval} via the function {dgp_seed}")
    elif isinstance(dgp_seed, (int, float)):
        dgpseedval = dgp_seed
        logging.info(
            f"DGP seed is set to {dgpseedval} as dgp_seed={dgp_seed} is a valid seed."
        )
    else:
        raise NotImplementedError(f"Unsupported dgp_seed value {dgp_seed}")

    dgp = create_dgp_from_key(dgp_key, dgpseedval, device=device, dtype=dtype)

    logging.info(f"setting seed to {seed} in sfe_trial()")

    output_dir = Path(output_dir)
    set_seed(seed)
    start_time = time()

    # choose sampler based on the model sampler key
    if model_sampler_key == "SFEMADEBinaryString":
        sampler = SFEMADEBinaryString(
            num_bits=int(dgp.dimension - 1),  # this will break!
            ig_threshold=ig_threshold,
            lr=dgp.get_sfe_lr(),
            device=device,
            dtype=dtype,
        )
    elif model_sampler_key == "SFEMADEDAG":
        sampler = SFEMADEDAG(
            num_nodes=int(dgp.num_nodes),
            ig_threshold=ig_threshold,
            lr=dgp.get_sfe_lr(),
            device=device,
            dtype=dtype,
        )
    elif model_sampler_key == "SFECategorical":
        sampler = SFECategorical(
            num_categories=dgp.num_categories(),
            ig_threshold=ig_threshold,
            lr=dgp.get_sfe_lr(),
            device=device,
            dtype=dtype,
        )
    elif model_sampler_key == "SFECategoricalBinaryString":
        sampler = SFECategoricalBinaryString(
            num_categories=dgp.num_categories(),
            ig_threshold=ig_threshold,
            lr=dgp.get_sfe_lr(),
            device=device,
            dtype=dtype,
        )
    else:
        raise NotImplementedError(f"Unsupported model sampler {model_sampler_key}")

    problem = VTIMCGEstimator(
        dgp,
        sampler,
        flow_type=flow_type,
        output_dir=output_dir,
        grad_norm_clip=grad_norm_clip,
        device=device,
        dtype=dtype,
        **kwargs,
    ).to(device=device)

    problem.setup_optimizer()
    if plot:
        problem.dgp.plot_joints(problem.base_dist, problem.param_transform)

    loss = problem.optimize(
        batch_size=batch_size,
        num_iterations=num_iterations,
        callbacks=[
            # periodic_save_callback,
            # latest_save_callback,
            # logging_callback,
        ],
    )
    total_time = time() - start_time
    return dict(problem=problem, loss=loss.item(), time=total_time)
