#!/usr/bin/env python3

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

import sysconfig
import os
# Set CPATH in the main process (may help; we also set it in the worker)
os.environ['CPATH'] = sysconfig.get_paths()['include']
# Also disable torch compile in the main process (just in case)
os.environ["ENABLE_TORCH_COMPILE"] = "False"

import asyncio
import concurrent.futures
import multiprocessing as mp
import functools
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import your worker function
from vti.examples.sweeping_trial import sweeping_trial

# Define your base parameters (adjust as needed)
base_kwargs = dict(
    num_iterations=50000,
    batch_size=1024,
    seed=3,
    ig_threshold=5e-3,
    dgp_key="nonlineardagmlpnobias_10_ndata_1024",
    dgp_seed=1004,
    flow_type="affine510f128",
    model_sampler_key="SFEMADEDAG",
    grad_norm_clip=50.0,
    device="cuda",
    dtype="float64",
    plot=False,
)

# Worker initializer: ensure CPATH and disable torch compile in each worker.
def init_worker():
    import sysconfig, os
    os.environ['CPATH'] = sysconfig.get_paths()['include']
    os.environ["ENABLE_TORCH_COMPILE"] = "False"
    try:
        import torch._dynamo
        torch._dynamo.disable()
    except Exception:
        pass

async def run_local_param_sweep(sweeping_trial_fn, base_kwargs, sweep_param, sweep_vals, num_replicates, base_output_dir, ctx):
    tasks = []
    loop = asyncio.get_running_loop()
    # Create a ProcessPoolExecutor with spawn context and initializer
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=1, mp_context=ctx, initializer=init_worker
    ) as executor:
        for sweep_val in sweep_vals:
            for rep in range(num_replicates):
                # Copy base parameters and update for this job.
                kwargs = base_kwargs.copy()
                kwargs[sweep_param] = sweep_val
                kwargs['sweep_value'] = sweep_val
                # Create a unique output directory.
                job_output_dir = os.path.join(base_output_dir, f"{sweep_param}_{sweep_val}_rep{rep}")
                os.makedirs(job_output_dir, exist_ok=True)
                kwargs['output_dir'] = job_output_dir
                # Bind positional arguments using functools.partial.
                partial_fn = functools.partial(
                    sweeping_trial_fn,
                    rep,
                    f"{sweep_param}_{sweep_val}_rep{rep}",
                    **kwargs
                )
                task = loop.run_in_executor(executor, partial_fn)
                tasks.append(task)
        results = await asyncio.gather(*tasks)
    results_df = pd.DataFrame(results)
    results_df.to_pickle(os.path.join(base_output_dir, "aggregated_results.df"))
    return results_df

async def main():
    # Define sweep parameters.
    sweep_param = "dgp_key"
    sweep_vals = [f'nonlineardagmlpnobias_10_ndata_{int(2**i)}' for i in [4, 5, 6, 7, 8, 9, 10]]
    num_replicates = 10

    # Build the output directory: _experiments/{experiment_name}_{datetimestring}/outputs/
    experiment_name = "vti10dagmlpnobias_sweep"
    datetimestring = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join(os.getcwd(), "_experiments", f"{experiment_name}_{datetimestring}", "outputs")
    os.makedirs(base_output_dir, exist_ok=True)
    print("Base output directory:", base_output_dir)

    # Create a spawn context.
    ctx = mp.get_context("spawn")
    results_df = await run_local_param_sweep(sweeping_trial, base_kwargs, sweep_param, sweep_vals, num_replicates, base_output_dir, ctx)
    print("Parameter sweep completed. Results saved in:", base_output_dir)

    # Plot results.
    metrics = ['F1', 'SHD', 'Brier', 'AUROC', 'loss']
    for metric in metrics:
        plt.figure(figsize=(8, 5))
        ax = sns.boxplot(data=results_df, x='sweep_value', y=metric)
        ax.set_xlabel(sweep_param)
        ax.set_ylabel(metric)
        ax.set_title(f"Parameter sweep: {metric} vs {sweep_param}")
        png_path = os.path.join(base_output_dir, f"{sweep_param}_{metric}.png")
        pdf_path = os.path.join(base_output_dir, f"{sweep_param}_{metric}.pdf")
        plt.savefig(png_path, format="png")
        plt.savefig(pdf_path, format="pdf")
        plt.show()

if __name__ == '__main__':
    asyncio.run(main())

