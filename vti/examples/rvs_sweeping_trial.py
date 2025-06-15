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

# vti/examples/rvs_sweeping_trial.py

def rvs_sweeping_trial(replicate=None, job_id=None, **kwargs):
    """
    Run a single sweeping trial.
    """
    import os
    # Disable Torch compilation to avoid BackendCompilerFailed errors.
    os.environ["ENABLE_TORCH_COMPILE"] = "False"
    try:
        import torch._dynamo
        torch._dynamo.disable()
    except Exception:
        pass

    import sysconfig
    # Ensure the CPATH is set in the worker.
    os.environ.setdefault('CPATH', sysconfig.get_paths()['include'])

    import logging
    import pickle
    import torch
    import numpy as np
    from vti.flows.transdimensional import TransdimensionalFlow
    from vti.examples.sfe_trial import sfe_trial

    # Set up output directory and logging.
    output_dir = kwargs.get('output_dir', 'default_directory')
    seed = kwargs.get('seed', 'no_seed')
    os.makedirs(output_dir, exist_ok=True)

    log_file = f"{output_dir}/job_{replicate}_{job_id}_{seed}.log"

    logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            force=True,
        )

    sweep_value = kwargs.pop('sweep_value')

    logging.info(f'seed is set to {seed}')

    # Run the trial.
    res = sfe_trial(**kwargs)
    problem = res["problem"]
    with torch.no_grad():
        tdf = TransdimensionalFlow(
            problem.model_sampler,
            problem.param_transform,
            problem.dgp.reference_dist(),
        )
        # run rjmcmc for true model probs
        problem.dgp.sample_true_mk_probs(plot_bivariates=False, store_samples=True)  
        mk_identifiers = problem.dgp.mk_identifiers()
        true_mk_ids, true_mk_probs = problem.dgp.true_mk_identifiers, problem.dgp.true_mk_probs
        N = mk_identifiers.size(0)

        # Initialize the result tensor with zeros
        mk_probs_true = torch.zeros(N, device=mk_identifiers.device, dtype=true_mk_probs.dtype)

        # Compute a boolean mask where each element indicates whether the row in true_mk_ids matches a row in mk_identifiers.
        # The mask shape will be [M, N]
        mask = (true_mk_ids.unsqueeze(1) == mk_identifiers.unsqueeze(0)).all(dim=-1)

        # Get indices of matches. Each row in 'mask.nonzero()' returns [i, j]
        # where i is the row index in true_mk_ids and j is the corresponding row index in mk_identifiers.
        indices = mask.nonzero(as_tuple=False)

        # For each match, assign the probability from true_mk_probs[i] to mk_probs_true[j]
        mk_probs_true[indices[:, 1]] = true_mk_probs[indices[:, 0]]

        # estimated model probs
        mk_probs_hat = problem.model_sampler.log_prob(mk_identifiers).exp()

        mk_cond_nll = torch.zeros_like(mk_probs_hat)
        for i,mk in enumerate(mk_identifiers):
            mk_cond_nll[i] = problem.dgp.compute_conditional_average_nll(tdf,mk)

        logging.info(f"Attempting to access dgp mk identifier")
        # get the index of the dgp mk identifier
        dgp_mk_identifier = problem.dgp.dgp_mk_identifier
        dgp_index = torch.nonzero((mk_identifiers == dgp_mk_identifier).all(dim=1), as_tuple=False)[0].item()
        logging.info(f"dgp mk identifier {dgp_mk_identifier} at index {dgp_index}")

        mk_probs_true = mk_probs_true.detach().cpu()
        mk_probs_hat = mk_probs_hat.detach().cpu()
        mk_cond_nll = mk_cond_nll.detach().cpu()

    try:
        with open(f"{output_dir}/res.txt", 'w') as f:
            f.write(f"dgp_seed\t{kwargs['dgp_seed']}\n")
            f.write(f"seed\t{kwargs['seed']}\n")
            f.write(f"dgp_key\t{kwargs['dgp_key']}\n")
            f.write(f"loss\t{res['loss']:.5f}\n")
            f.write(f"time\t{res['time']:.5f}\n")
            f.write(f"dgp_idx\t{dgp_index}\n")
        torch.save(mk_probs_true,f"{output_dir}/mk_probs_true.pt")
        torch.save(mk_probs_hat,f"{output_dir}/mk_probs_hat.pt")
        torch.save(mk_cond_nll,f"{output_dir}/mk_cond_nll.pt")
    except Exception as e:
        logging.info(f"Could not write result to {output_dir}/res.txt : {str(e)}")

    try:
        x_data, y_data = problem.dgp.get_data()
        torch.save(x_data.detach().cpu(), f"{output_dir}/x_data_{replicate}_{job_id}.pt")
        torch.save(y_data.detach().cpu(), f"{output_dir}/y_data_{replicate}_{job_id}.pt")
    except Exception as e:
        logging.info(f"Could not save data to {output_dir}: {str(e)}")

    float_loss = res["loss"]
    float_time = res["time"]

    for handler in logging.getLogger().handlers:
        handler.flush()
    #logging.handlers.clear()

    # Free any GPU memory that was allocated during the trial.
    del tdf, problem, res
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return dict(
            loss=float_loss,
            mk_cond_nll=mk_cond_nll.detach(), 
            mk_probs_true=mk_probs_true.detach(), 
            mk_probs_hat=mk_probs_hat.detach(),
            dgp_idx=dgp_index,
            time=float_time,
            replicate=replicate,
            job_id=job_id,
            sweep_value=sweep_value,
        )


