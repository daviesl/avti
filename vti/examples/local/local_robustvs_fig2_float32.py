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
from tueplots import bundles
import seaborn as sns
import sys
import numpy as np
import torch
import math

# Import your worker function
#from vti.examples.dag_sweeping_trial import dag_sweeping_trial as the_sweeping_trial
from vti.examples.rvs_sweeping_trial import rvs_sweeping_trial as the_sweeping_trial

# Define your base parameters (adjust as needed)
base_kwargs = dict(
    num_iterations=10000,
    #num_iterations=100,
    batch_size=1024,
    seed=0,
    ig_threshold=5e-3,
    dgp_key="robust_vs_8",
    flow_type="shareddiagspline46",
    device="cuda" if torch.cuda.is_available() else "cpu",
    dtype="float32",
    dgp_seed="dgpseedfn1000",
    model_sampler_key="SFECategoricalBinaryString",
    grad_norm_clip=50.0,
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
        max_workers=3, mp_context=ctx, initializer=init_worker
    ) as executor:
        for sweep_val in sweep_vals:
            for rep in range(num_replicates):
                # Copy base parameters and update for this job.
                kwargs = base_kwargs.copy()
                kwargs[sweep_param] = sweep_val
                kwargs['sweep_value'] = sweep_val
                kwargs['seed'] = rep
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
    sweep_param = "flow_type"
    sweep_vals=["diagnorm","affine55","shareddiagspline46"]
    num_replicates = 10

    labeldictall = {"diagnorm":"Diagonal Gaussian MLP", 
                     "affine55": "Affine MAF (5,5)",
                     "shareddiagspline46": "Spline MAF (4,6)",
                     }

    # Build the output directory: _experiments/{experiment_name}_{datetimestring}/outputs/
    experiment_name = f"vti_rvs_fig1_float32"
    datetimestring = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join(os.getcwd(), "_experiments", f"{experiment_name}_{datetimestring}", "outputs")
    os.makedirs(base_output_dir, exist_ok=True)
    print("Base output directory:", base_output_dir)

    # run two sweeps, one for each misspec level
    mslevels = ['mid','high']
    dgp_key_dict = {'mid':'robust_vs_midms_8',
                    'high':'robust_vs_8'}
    results_dict = {}
    for misspeclevel in mslevels[::-1]:
        # Create a spawn context.
        ctx = mp.get_context("spawn")
        base_kwargs['dgp_key'] = dgp_key_dict[misspeclevel]
        ms_output_dir = os.path.join(base_output_dir, misspeclevel)
        os.makedirs(ms_output_dir, exist_ok=True)
        #results_df = await run_local_param_sweep(the_sweeping_trial, base_kwargs, sweep_param, sweep_vals, num_replicates, base_output_dir, ctx)
        results_df = await run_local_param_sweep(the_sweeping_trial, base_kwargs, sweep_param, sweep_vals[::-1], num_replicates, ms_output_dir, ctx)
        print("Parameter sweep completed. Results saved in:", ms_output_dir)
        results_dict[misspeclevel] = results_df

    if True:
        # Number of columns is equal to the length of sweep_vals
        ncols = len(sweep_vals)
        nrows = 2  # Two rows as specified

        # Create a figure with subplots
        plotscale=0.9
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols*2, sharex=True, figsize=(plotscale*2.5 * ncols*2, plotscale*2.3 * nrows))

        from matplotlib.cm import get_cmap
        # Apply ICML 2024 style
        plt.rcParams.update(bundles.icml2024())    
        #plt.rcParams.update({'font.size': 14})
        print("plotting")

        #for isright,dfpath in enumerate([dfpath_l, dfpath_r]):
        for isright,misspeclevel in enumerate(mslevels):
            #df_saved = pd.read_pickle(dfpath)
            #results_df = df_saved
            results_df = results_dict[misspeclevel]

            # Define a color map
            #color_map = get_cmap('Dark2')
            color_map = get_cmap('tab10')
            num_colors = 10  # Set this based on the maximum expected rows per filtered_df


            symlog_linthresh=6e-4
            symlog_ticks=[0, 1e-3, 1e-2, 1e-1, 1]
            symlog_ticklabels=[r"$0$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^0$"]

            nll_lb = 1e-2 # 1e-4
            nll_ub = 1000

            nll_yticks = [10**i for i in range(int(math.log10(nll_lb)), int(math.log10(nll_ub)) + 1)]
            if True:
                nll_yticklabels = [
                    r"$\leq 10^{" + f"{int(math.log10(nll_lb))}" + r"}$" if tick == nll_yticks[0]
                    else " " if tick == nll_yticks[-1]
                    else rf"$10^{{{int(math.log10(tick))}}}$"
                    for tick in nll_yticks
                ]

            USE_SYMLOG=False

            prob_lb=1e-4
            prob_ub=1.

            #exclude=[0,1,4]
            exclude=[]

            prob_yticks = [10**i for i in range(int(math.log10(prob_lb)), int(math.log10(prob_ub)) + 1)]
            prob_yticklabels = [
                r"$\leq 10^{" + f"{int(math.log10(prob_lb))}" + r"}$" if tick == prob_yticks[0]
                else rf"$10^{{{int(math.log10(tick))}}}$"
                for tick in prob_yticks
            ]

            # Loop over each sweep value
            for col_idx, sv in enumerate(sweep_vals):
                filtered_df = results_df[results_df[sweep_param] == sv]
                num_rows = len(filtered_df)

                # Top row for predicted vs true probs
                ax_true_hat = axes[0, col_idx+ncols*isright]
                #for idx, row_df in filtered_df.iterrows():   
                for idx, (run_id, row_df) in enumerate(filtered_df.iterrows()):
                    if idx in exclude:
                        continue
                    color = color_map(idx / float(num_rows-len(exclude)))  # Get a color from the color map
                    mk_probs_true = row_df['mk_probs_true']
                    mk_probs_hat = row_df['mk_probs_hat']
                    mk_probs_true_np = mk_probs_true.cpu().numpy() if torch.is_tensor(mk_probs_true) else mk_probs_true
                    mk_probs_hat_np = mk_probs_hat.cpu().numpy() if torch.is_tensor(mk_probs_hat) else mk_probs_hat
                    if True:
                        spl = ax_true_hat.scatter(np.clip(mk_probs_true_np, prob_lb, prob_ub), np.clip(mk_probs_hat_np, prob_lb, prob_ub), alpha=.9,  marker='x',linewidths=0.25, s=10,zorder=idx) 

                #ax_true_hat.plot([0, 1], [0, 1], 'k--', linewidth=0.5)
                ax_true_hat.plot([0, 1], [0, 1], linestyle='dashed',color='black', linewidth=0.5)
                #for ofst in [1e-3,1e-2,1e-1]:
                for ofst in prob_yticks[1:-1]:
                    ax_true_hat.plot(np.linspace(ofst, 1,10000), np.linspace(0, 1-ofst,10000), linestyle='dotted',color='grey', linewidth=0.5)
                    ax_true_hat.plot(np.linspace(0, 1-ofst,10000), np.linspace(ofst, 1,10000), linestyle='dotted',color='grey', linewidth=0.5)

                print(f'labeldict {labeldictall} {sv}')
                if isright==1:
                    dgptitle='Misspecification level: High'
                else:
                    dgptitle='Misspecification level: Medium'
                ax_true_hat.set_title(f'{dgptitle}\n{labeldictall[sv]}')
                if True:
                    ax_true_hat.set_xlim((prob_lb, prob_ub))
                    ax_true_hat.set_ylim((prob_lb, prob_ub))
                    ax_true_hat.set_xscale('log')
                    ax_true_hat.set_yscale('log')
                    # ticks for probs in log scale
                    ax_true_hat.set_xticks(prob_yticks)
                    ax_true_hat.set_xticklabels(prob_yticklabels)
                    ax_true_hat.set_yticks(prob_yticks)
                    ax_true_hat.set_yticklabels(prob_yticklabels)

                if col_idx==0 and isright==0:
                    ax_true_hat.set_ylabel(r"$q_{\psi}(m)$")
                #ax_true_hat.grid(True,color='grey',linewidth=0.5,linestyle='dotted')

                # Bottom row for cond nll per model
                ax_nll = axes[1, col_idx+ncols*isright]
                for idx, (run_id, row_df) in enumerate(filtered_df.iterrows()):
                    if idx in exclude:
                        continue
                    #print(f'idx {idx} {col_idx}')
                    color = color_map(idx / float(num_rows-len(exclude)))  # Get a color from the color map
                    mk_cond_nll = row_df['mk_cond_nll']
                    mk_probs_hat = row_df['mk_probs_hat']
                    mk_probs_true = row_df['mk_probs_true']
                    mk_cond_nll_np = mk_cond_nll.cpu().numpy() if torch.is_tensor(mk_cond_nll) else mk_cond_nll
                    mk_cond_nll_np = np.clip(mk_cond_nll_np, a_min=0.0, a_max=None)
                    mk_probs_hat_np = mk_probs_hat.cpu().numpy() if torch.is_tensor(mk_probs_hat) else mk_probs_hat
                    mk_probs_true_np = mk_probs_true.cpu().numpy() if torch.is_tensor(mk_probs_true) else mk_probs_true
                    #ax_nll.scatter(mk_probs_hat_np, mk_cond_nll_np, alpha=0.6,  marker='x',linewidths=1.5) #color=color,
                    if True:
                        spl = ax_nll.scatter( np.clip(mk_probs_true_np, prob_lb, prob_ub), np.clip(mk_cond_nll_np, nll_lb, nll_ub), alpha=.9,  marker='x',linewidths=0.25, s=10, zorder=idx)
                        splcolor = spl.get_facecolor()[0]
                        print(f'color for {idx} is {splcolor}')

                #ax_nll.set_xlabel(r"$q_{\psi}(m)$")
                ax_nll.set_xlabel(r"$\pi(m)$")
                if True:
                    ax_nll.set_xscale('log')

                    # explicit tick labels for log
                    ax_nll.set_xticks(prob_yticks)
                    ax_nll.set_xticklabels(prob_yticklabels)

                    ax_nll.set_xlim((prob_lb, prob_ub))

                if col_idx==0 and isright==0:
                    ax_nll.set_ylabel(r"$H(\pi(\theta_m|m),q_{\psi,\phi}(\theta_m|m))$")

                ax_nll.set_ylim((nll_lb, 100))
                ax_nll.set_yscale('log')

                ax_nll.set_yticks(nll_yticks)
                ax_nll.set_yticklabels(nll_yticklabels)

                ax_nll.grid(True,color='grey',linewidth=0.5,linestyle='dotted')

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.35, hspace=0) # colapse vertical axes

        png_path = os.path.join(base_output_dir, f"fig1_8may.png")
        pdf_path = os.path.join(base_output_dir, f"fig1_8may.pdf")
        plt.savefig(png_path, format="png")
        plt.savefig(pdf_path, format="pdf")


if __name__ == '__main__':
    asyncio.run(main())

