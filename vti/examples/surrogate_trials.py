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
a GP Surrogate run encapsulates the definition of a likelihood function from our library, the inference using our algorithm to fit it.
This is a handy utility function that can be used to run a single experiment wrapped into sweeps and experiments and optimisations and so on.

DANGER: per default gp_surrogate_trial will return its output, which can be enormous.
You should wrap it, and return only the statistics you need if you are invoking this on slurm, or face all kinds of disk-access and memory-overflowing hells.
"""

from pathlib import Path
from time import time

import torch

from vti.utils.seed import set_seed
from vti.infer.surrogate import VTISurrogateEstimator
from vti.utils.torch_nn_helpers import ensure_dtype, ensure_device
from vti.surrogates import DiagonalGaussianSurrogate, EnsembleGaussianSurrogate
from vti.model_samplers import SoftmaxSurrogateSampler, BinaryStringSSSampler
from vti.dgp import create_dgp_from_key
from vti.utils.linalg_lowrank import reduced_mean_dev
from vti.utils.callbacks import SurrogateLoggingCallback
import vti.utils.logging as logging
from vti.utils.debug import dump_node_info

# from vti.utils.math_helpers import is_function
from vti.dgp import dgp_seed_fns


def gp_surrogate_trial(
    ## inference parameters
    num_iterations=1000,
    batch_size=32,
    seed=4,
    ## likelihood parameters
    dgp_key="diagnorm",
    dgp_seed=5,
    model_sampler_key="SSS",  # either SSS or BSSSS
    ## flow parameters
    flow_type="diagnorm",
    ## surrogate parameters
    surrogate_type="diagnorm",
    basis_rank=50,
    basis_reduction="random",
    basis_normalize="False",
    f_coupling=1e2,
    prior_diag_variance=1e2,
    lr_variance_scale=1.0,
    obs_variance=1.0,
    obs_beta=0.99,
    max_entropy_gain=0.1,  # per obs
    squish_utility=True,
    diffuse_prior=1.0,
    ## Utility parameters
    output_dir="___output",
    device="cuda",
    dtype="float64",
    plot=False,
    smoke_test=False,
    ## extra parameters
    **kwargs,
):
    """
    I construct a problem and solve it, adding some extra instrumentation (timing, logging).
    I accept the mandatory arguments of the experiment executors
    (`job_id`, `smoke_test`, `output_dir`) which I might use to doctor the problem or change the instrumentation.
    """
    assert dgp_key is not None, "Argument dgp_key needs to be set"
    assert dgp_seed is not None, "Argument dgp_seed needs to be set"

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
    elif isinstance(dgp_seed, (int, float)):
        dgpseedval = dgp_seed
    else:
        raise NotImplementedError(f"Unsupported dgp_seed value {dgp_seed}")

    if kwargs:
        logging.warning(f"Unused arguments: {kwargs}")
    if smoke_test:
        num_iterations = min(num_iterations, 1000)
        # FIXME smoke test args might also  need to be set in the dgp constructor in addition to here,
        # so that we can NOT ONLY make the number of problems we solve a small number,
        # BUT ALSO make the problems themselves small.
        # This will of course depend upon our purpose; we might want to keep the problems themselves large
        # in order that we can e.g. optimize the batch size.

    # logging.info(dump_node_info())
    device = ensure_device(device)
    dtype = ensure_dtype(dtype)

    torch.set_default_dtype(dtype)
    torch.set_default_device(device)

    # create the dgp
    dgp = create_dgp_from_key(dgp_key, dgpseedval, device, dtype)

    # if output_dir=="___output":
    #     raise ValueError("failed to override path ")
    output_dir = Path(output_dir)
    set_seed(seed)
    start_time = time()
    if surrogate_type == "diagnorm":
        surrogate = DiagonalGaussianSurrogate(
            num_categories=dgp.num_categories(),
            prior_mean=0.0,
            prior_diag_variance=prior_diag_variance,
            f_coupling=f_coupling,
            obs_beta=obs_beta,
            diffuse_prior=diffuse_prior,
            obs_variance=obs_variance,
            max_entropy_gain=max_entropy_gain,
            device=device,
            dtype=dtype,
        )
    elif surrogate_type == "ensemble":
        # Low rank surrogate
        # For now we hard code the basis to be a random projection of the masks
        # the mean is not informative but the covariance might be.
        mean_, dev = reduced_mean_dev(
            features=dgp.all_masks(),  # TODO DISCUSS what does this do exactly? The features that are included in the flow?
            rank=basis_rank,
            reduction=basis_reduction,
            normalize=basis_normalize,
        )
        surrogate = EnsembleGaussianSurrogate(
            num_categories=dgp.num_categories(),
            prior_mean=0.0,
            prior_diag_variance=prior_diag_variance,
            prior_dev=dev,
            f_coupling=f_coupling,
            lr_variance_scale=lr_variance_scale,
            obs_beta=obs_beta,
            diffuse_prior=diffuse_prior,
            obs_variance=obs_variance,
            max_entropy_gain=max_entropy_gain,
            device=device,
            dtype=dtype,
        )
    else:
        raise ValueError(f"Unknown surrogate type: {surrogate_type}")
    if model_sampler_key == "SSS":
        sampler = SoftmaxSurrogateSampler(
            # sampler = TanhSurrogateSampler(
            surrogate,
            squish_utility=squish_utility,
            device=device,
            dtype=dtype,
        )
    elif model_sampler_key == "BSSSS":
        sampler = BinaryStringSSSampler(
            surrogate,
            squish_utility=squish_utility,
            device=device,
            dtype=dtype,
        )
    else:
        raise Exception(f"Model sampler {model_sampler_key} not supported")

    problem = VTISurrogateEstimator(
        dgp,
        sampler,
        flow_type=flow_type,
        output_dir=output_dir,
        device=device,
        dtype=dtype,
        **kwargs,
    )
    # periodic_save_callback = CheckpointCallback(
    #     model=problem,
    #     interval=100,
    #     filename_template="checkpoint_{step:05d}.pt"
    # )

    # # Initialize latest checkpoint callback
    # latest_save_callback = CheckpointCallback(
    #     model=problem,
    #     filename_template="latest_checkpoint.pt",
    #     save_last=True
    # )

    # the below only works for diagnorm generator
    # logging_callback = SurrogateLoggingCallback(
    #    problem, model_weight_target=dgp.modelprobs
    # )

    problem.setup_optimizer()

    # plot test
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
