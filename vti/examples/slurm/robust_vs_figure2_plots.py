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

import marimo

__generated_with = "0.10.9"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""## Basic imports""")
    return


@app.cell(hide_code=True)
def _():
    from pathlib import Path
    from time import time
    import os
    os.environ["ENABLE_TORCH_COMPILE"]="False"
    import logging

    import numpy as np
    import torch
    import dotenv
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from tueplots import bundles
    import numpy as np
    import asyncio

    from vti.utils.kld import kld_logits
    from vti.utils.seed import set_seed
    from vti.infer import VTIMCGEstimator, VTISurrogateEstimator
    from vti.utils.torch_nn_helpers import ensure_dtype, ensure_device
    from vti.model_samplers import (
        SFECategorical,
        SFEMADEBinaryString,
        SFEMADEDAG,
        BinaryStringSSSampler,
    )
    from vti.utils.linalg_lowrank import reduced_mean_dev
    from vti.utils.callbacks import SurrogateLoggingCallback, CheckpointCallback
    from vti.utils.experiment import (
        list_experiments,
        get_latest_experiment,
        ParamSweepGenerator,
        AxParameterGenerator,
        ParamSweepExperiment,
        AxExperiment,
    )
    from vti.flows.transdimensional import TransdimensionalFlow

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    slurm_account = os.getenv("SLURM_ACCOUNT", False)
    return (
        AxExperiment,
        AxParameterGenerator,
        BinaryStringSSSampler,
        CheckpointCallback,
        ParamSweepExperiment,
        ParamSweepGenerator,
        Path,
        SFECategorical,
        SFEMADEBinaryString,
        SFEMADEDAG,
        SurrogateLoggingCallback,
        TransdimensionalFlow,
        VTIMCGEstimator,
        VTISurrogateEstimator,
        asyncio,
        bundles,
        dotenv,
        ensure_device,
        ensure_dtype,
        get_latest_experiment,
        kld_logits,
        list_experiments,
        logging,
        np,
        os,
        pd,
        plt,
        reduced_mean_dev,
        set_seed,
        slurm_account,
        sns,
        time,
        torch,
    )


@app.cell
def _(torch):
    USE_BO=False
    if USE_BO:
        base_kwargs = dict(
            ## inference parameters
            num_iterations=20000,
            #num_iterations=10, # smoketest
            batch_size=128,
            seed=4,
            # dgp parameters
            dgp_key="robust_vs_8",
            dgp_seed="dgpseedfn1000",
            model_sampler_key="BSSSS",
            ## flow parameters
            flow_type="spline46",
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
            max_entropy_gain=0.05,  # per obs, was 0.025
            squish_utility=True,
            diffuse_prior=1.0,
            ## Utility parameters
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype="float64",
            plot=False,
            # job_id=None,  # unused
        )
    else:
        base_kwargs = dict(
            num_iterations=20000, 
            batch_size=1024,
            seed=0,
            #dgp_key="robust_vs_8",
            #dgp_key="robust_vs_wide_prior_8",
            #dgp_key="robust_vs_midms_8",
            #dgp_key="robust_vs_midms_wide_prior_8",
            #dgp_key="robust_vs_noms_8",
            dgp_key="robust_vs_noms_wide_prior_8",
            dgp_seed="dgpseedfn1000",
            flow_type="spline46",
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype="float64",
            plot=False,
            #sfe_cat_kwargs
            ig_threshold=7e-4,
            # model sampler parameters
            # model_sampler_key="SFEMADEBinaryString",
            model_sampler_key="SFECategoricalBinaryString",
            # job_id=None
        )
    return USE_BO, base_kwargs


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## The main trial functions

        We set upon `gp_surrogate_trial`, which creates a probelm, fits a gp surrogate on it, and returns various performance metrics.
        .
        """
    )
    return


@app.cell
def _():
    from vti.examples.sfe_trial import sfe_trial
    from vti.examples.surrogate_trials import gp_surrogate_trial
    return gp_surrogate_trial, sfe_trial


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Run a single trial

        This approximates the CLI script that ran single trials before.
        """
    )
    return


@app.cell(disabled=True)
def _(
    TransdimensionalFlow,
    base_kwargs,
    gp_surrogate_trial,
    logging,
    plt,
    torch,
):
    def diagnostic_trial(**kwargs):
        """
        Verbose, messy  run, for ease of diagnosis.
        """
        res = gp_surrogate_trial(**kwargs, **base_kwargs)
        #res = sfe_trial(**kwargs, **base_kwargs)
        problem = res["problem"]

        # set up tdf
        tdf = TransdimensionalFlow(                                                                                                
                    problem.model_sampler,
                    problem.param_transform,
                    problem.dgp.reference_dist(),  
                    device=problem.dgp.device,
                    dtype=problem.dgp.dtype,
                )


        problem.dgp.plot_q_tdf(
                tdf, title="Variational posterior misspecified robust variable selection", font_size="6"
            )

        if False:
            # run rjmcmc for true model probs
            problem.dgp.sample_true_mk_probs(plot_bivariates=True, store_samples=True)  
            mk_probs_true = (
                problem.dgp.true_mk_probs_all
            )  # should be returned from previous statement

            # log avg NLL
            nll = problem.dgp.compute_average_nll(tdf)                                                                                 
            logging.info(f"Average NLL = {nll}")  

            mk_probs_hat = problem.model_sampler.probs()
            mk_identifiers = problem.dgp.mk_identifiers()

            mk_cond_nll = torch.zeros_like(mk_probs_hat)
            for i,mk in enumerate(mk_identifiers):
                mk_cond_nll[i] = problem.dgp.compute_conditional_average_nll(tdf,mk)
            mk_cond_nll = torch.clip(mk_cond_nll,min=None,max=100) # tune to taste

            # print model probs
            logging.info("sorted model probabilities by target")
            sortidx = torch.argsort(mk_probs_true, dim=0)

            logging.info("id\ttrue weight\tpredicted weight\tCond NLL")
            for i in sortidx:
                logging.info(f"{mk_identifiers[i]}\t{mk_probs_true[i]}\t{mk_probs_hat[i]}\t{mk_cond_nll[i]}")

            logging.info(f"terminal loss = {res['loss']}")

            # problem.dgp.plot_q(problem.param_transform, mk_probs_hat, 8192)
            #problem.dgp.plot_q_mk_selected(
            #    problem.param_transform, mk_identifiers, mk_probs_hat, 8192
            #)

            # plot rjmcmc marginal
            #problem.dgp.plot_q_mk_selected(
            #    problem.param_transform,
            #    problem.dgp.true_mk_identifiers,
            #    problem.dgp.true_mk_probs,
            #    8192,
            #)

            # Assume mk_probs_true and mk_probs_hat are existing Torch tensors
            # Convert tensors to numpy and plot them
            mk_probs_true_np = mk_probs_true.cpu().numpy()
            mk_probs_hat_np = mk_probs_hat.cpu().numpy()
            mk_cond_nll_np = mk_cond_nll.cpu().numpy()

            plt.figure(figsize=(3, 3))  # Set the plot size to 3x3 inches
            plt.scatter(mk_probs_true_np, mk_probs_hat_np, alpha=0.6)  # Scatter plot with partial transparency
            plt.plot([0, 1], [0, 1], 'k--', linewidth=0.5)  # Plotting the line y=x as a dotted thin black line
            plt.xlabel(r"$\pi(m)$")  # Label for x-axis
            plt.ylabel(r"$q_{\psi}(m)$")   # Label for y-axis
            plt.xlim((0,1))
            plt.ylim((0,1))
            plt.title(r'Scatter Plot of $\pi(m)$ vs. $q_{\psi}(m)$')  # Title of the plot
            plt.grid(True)  # Enable grid
            plt.show()  # Display the plot



            plt.figure(figsize=(3, 3))  # Set the plot size to 3x3 inches
            plt.scatter(mk_probs_hat_np, mk_cond_nll_np, alpha=0.6)  # Scatter plot with partial transparency
            plt.ylabel(r"$H(\pi(\theta_m|m),q_{\psi,\phi}(\theta_m|m))$")  # Label for x-axis
            plt.xlabel(r"$q_{\psi}(m)$")   # Label for y-axis
            plt.xlim((0,1))
            #plt.ylim((0,1))
            plt.title(r'Scatter Plot of conditional cross entropy vs. $q_{\psi}(m)$')  # Title of the plot
            plt.grid(True)  # Enable grid
            plt.show()  # Display the plot

    diagnostic_trial()
    return (diagnostic_trial,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Parameter sweep""")
    return


@app.cell
def _(
    TransdimensionalFlow,
    USE_BO,
    gp_surrogate_trial,
    logging,
    sfe_trial,
    torch,
):
    def sweeping_trial(replicate=None, job_id=None, **kwargs):
        """
        Compact run which just returns the important stuff, for ease of analysis.
        Run this in a sweep.
        """
        output_dir = kwargs.get('output_dir', 'default_directory')  # Provides a default if 'output_dir' is not passed

        log_file = f"{output_dir}/job.log"

        # Configure basic logging to file
        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        if USE_BO:
            res = gp_surrogate_trial(**kwargs)
        else:
            res = sfe_trial(**kwargs)

        problem = res["problem"]

        with torch.no_grad():
            # set up tdf
            tdf = TransdimensionalFlow(                                                                                                
                        problem.model_sampler,
                        problem.param_transform,
                        problem.dgp.reference_dist(),  
                        device=problem.dgp.device,
                        dtype=problem.dgp.dtype,
                    )

            # run rjmcmc for true model probs
            problem.dgp.sample_true_mk_probs(plot_bivariates=False, store_samples=True)  
            #mk_probs_true = (
            #    problem.dgp.true_mk_probs_all
            #)  # should be returned from previous statement

            mk_identifiers = problem.dgp.mk_identifiers()

            if True:
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

            #if USE_BO:
            #    mk_probs_hat = problem.model_sampler.probs()
            #else:
            #    mk_probs_hat = problem.model_sampler.log_prob(mk_identifiers).exp()

            mk_probs_hat = problem.model_sampler.log_prob(mk_identifiers).exp()

            mk_cond_nll = torch.zeros_like(mk_probs_hat)
            for i,mk in enumerate(mk_identifiers):
                mk_cond_nll[i] = problem.dgp.compute_conditional_average_nll(tdf,mk)
            #mk_cond_nll = torch.clip(mk_cond_nll,min=None,max=100) # tune to taste

        return dict(mk_cond_nll=mk_cond_nll.detach(), mk_probs_true=mk_probs_true.detach(), mk_probs_hat=mk_probs_hat.detach())
    return (sweeping_trial,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Plot results

        We are going to be fancy and "resume" the experiment, trying to load it from disk so that we don't re-run the job every time we re-do the plots
        """
    )
    return


@app.cell
def _(ParamSweepExperiment, bundles, logging, np, plt, torch):
    async def param_sweep_plot(
        experiment_prefix,
        sweeping_trial_fn,
        sweep_param,
        sweep_vals,
        smoke_test=False,
        base_kwargs={},
        num_replicates=12,
        executor_params={},
        clobber=True,
    ):
        """
        Perform a parameter sweep, collect results, and generate a boxplot.

        Parameters:
            sweeping_trial (function): Function to execute for each parameter set.
            base_kwargs (dict): Base parameters for the sweeping trial.
            sweep_param (str): The parameter to sweep.
            sweep_vals (list): Values of the parameter to sweep over.
            num_replicates (int): Number of replicates for each parameter value.
            executor_params (dict): Parameters for the executor.
            experiment_name (str): Name of the experiment.
            output_dir (str): Directory to save the plots.
        """
        experiment_name = f"{experiment_prefix}_{sweep_param}"

        # Create a new parameter sweep experiment
        pexp = ParamSweepExperiment.get_or_create(
            function=sweeping_trial_fn,
            sweep_params={sweep_param: sweep_vals},
            num_replicates=num_replicates,
            experiment_name=experiment_name,
            executor_params=executor_params,
            smoke_test=smoke_test,
            base_params=base_kwargs,
            clobber=clobber,
        )

        # Generate and run the experiment
        gen = pexp.create_parameter_generator()
        logging.info("launching jobs")
        await pexp.run_async(gen)
        logging.info("waiting for results")

        # Collect results
        results_df = await pexp.collect_results_async()
        # set paths
        output_path_png = f"{pexp.output_dir}/{experiment_name}_{sweep_param}.png"
        output_path_pdf = f"{pexp.output_dir}/{experiment_name}_{sweep_param}.pdf"
        output_path_df = f"{pexp.output_dir}/{experiment_name}_{sweep_param}.df"
        # save immediately 
        results_df.to_pickle(output_path_df)
        # try plotting
        logging.info("plotting")

        # Apply ICML 2024 style
        plt.rcParams.update(bundles.icml2024())
        #plt.rcParams.update({'font.size': 14})

        from matplotlib.cm import get_cmap

        # Number of columns is equal to the length of sweep_vals
        ncols = len(sweep_vals)
        nrows = 2  # Two rows as specified

        # Define a color map
        color_map = get_cmap('Dark2')
        num_colors = 10  # Set this based on the maximum expected rows per filtered_df

        # Create a figure with subplots
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(2.5 * ncols, 2.5 * nrows))

        # dict for labels
        #labeldict = {"diagnorm":"Diagonal Gaussian MLP", "affine55f128": "Affine MAF (5,5,128)","spline23f128": "Spline MAF (2,3,128)","spline410f128" : "Spline MAF (4,10,128)"}
        #labeldict = {"diagnorm":"Diagonal Gaussian MLP", "affine52f128": "Affine MAF (5,2,128)", "affine55f128": "Affine MAF (5,5,128)","spline22f128": "Spline MAF (2,2,128)", "spline23f128": "Spline MAF (2,3,128)","spline46f128" : "Spline MAF (4,6,128)", "spline410f128" : "Spline MAF (4,10,128)"}
        labeldict = {"diagnorm":"Diagonal Gaussian MLP", 
                 "affine52f128": "Affine MAF (5,2,128)", 
                 "affine55f128": "Affine MAF (5,5,128)",
                 "spline22f128": "Spline MAF (2,2,128)", 
                 "spline23f128": "Spline MAF (2,3,128)",
                 "spline46f128" : "Spline MAF (4,6,128)", 
                 "spline410f128" : "Spline MAF (4,10,128)",
                 "affine52": "Affine MAF (5,2)", 
                 "affine55": "Affine MAF (5,5)",
                 "spline22": "Spline MAF (2,2)", 
                 "spline23": "Spline MAF (2,3)",
                 "spline46" : "Spline MAF (4,6)", 
                 "spline410" : "Spline MAF (4,10)",
                 "diagspline22": "Diag Gauss + Spline MAF (2,2)",
                 "diagspline46": "Diag Gauss + Spline MAF (4,6)",
                 "shareddiagspline22": "Shared loc scale + Spline MAF (2,2)",
                 "shareddiagspline46": "Shared loc scale + Spline MAF (4,6)",
                }

        symlog_linthresh=6e-4
        symlog_ticks=[0, 1e-3, 1e-2, 1e-1, 1]
        symlog_ticklabels=[r"$0$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^0$"]

        nll_lb = 1e-3
        nll_ub = 100

        # Loop over each sweep value
        for col_idx, sv in enumerate(sweep_vals):
            filtered_df = results_df[results_df[sweep_param] == sv]
            num_rows = len(filtered_df)

            # Top row for predicted vs true probs
            ax_true_hat = axes[0, col_idx]
            for idx, row_df in filtered_df.iterrows():
                color = color_map(idx / num_rows)  # Get a color from the color map
                mk_probs_true = row_df['mk_probs_true']
                mk_probs_hat = row_df['mk_probs_hat']
                mk_probs_true_np = mk_probs_true.cpu().numpy() if torch.is_tensor(mk_probs_true) else mk_probs_true
                mk_probs_hat_np = mk_probs_hat.cpu().numpy() if torch.is_tensor(mk_probs_hat) else mk_probs_hat
                ax_true_hat.scatter(mk_probs_true_np, mk_probs_hat_np, alpha=1,  marker='x',linewidths=0.5, s=5) #color=color,

            ax_true_hat.plot([0, 1], [0, 1], 'k--', linewidth=0.5)

            for ofst in [1e-3,1e-2,1e-1]:
                ax_true_hat.plot(np.linspace(ofst, 1,10000), np.linspace(0, 1-ofst,10000), linestyle='dotted',color='grey', linewidth=0.5)
                ax_true_hat.plot(np.linspace(0, 1-ofst,10000), np.linspace(ofst, 1,10000), linestyle='dotted',color='grey', linewidth=0.5)

            ax_true_hat.set_xlim((0, 1))
            ax_true_hat.set_ylim((0, 1))
            ax_true_hat.set_title(f'{labeldict[sv]}')
            ax_true_hat.set_xlabel(r"$\pi(m)$")
            ax_true_hat.set_xscale('symlog', linthresh=symlog_linthresh, linscale=0.1)
            ax_true_hat.set_yscale('symlog', linthresh=symlog_linthresh, linscale=0.1)

            # explicit tick labels for symlog
            ax_true_hat.set_xticks(symlog_ticks)
            ax_true_hat.set_xticklabels(symlog_ticklabels)
            ax_true_hat.set_yticks(symlog_ticks)
            ax_true_hat.set_yticklabels(symlog_ticklabels)

            ax_true_hat.set_ylabel(r"$q_{\psi}(m)$")
            #ax_true_hat.grid(True)

            # Bottom row for cond nll per model
            ax_nll = axes[1, col_idx]
            for idx, row_df in filtered_df.iterrows():
                color = color_map(idx / num_rows)  # Get a color from the color map
                mk_cond_nll = row_df['mk_cond_nll']
                mk_probs_hat = row_df['mk_probs_hat']
                mk_probs_true = row_df['mk_probs_true']
                mk_cond_nll_np = mk_cond_nll.cpu().numpy() if torch.is_tensor(mk_cond_nll) else mk_cond_nll
                mk_cond_nll_np = np.clip(mk_cond_nll_np, a_min=0.0, a_max=None)
                mk_probs_hat_np = mk_probs_hat.cpu().numpy() if torch.is_tensor(mk_probs_hat) else mk_probs_hat
                mk_probs_true_np = mk_probs_true.cpu().numpy() if torch.is_tensor(mk_probs_true) else mk_probs_true
                #ax_nll.scatter(mk_probs_hat_np, mk_cond_nll_np, alpha=0.6,  marker='x',linewidths=1.5) #color=color,
                ax_nll.scatter(mk_probs_true_np, np.clip(mk_cond_nll_np, nll_lb, nll_ub), alpha=1,  marker='x',linewidths=0.5, s=5) #color=color,

            #ax_nll.set_xlabel(r"$q_{\psi}(m)$")
            ax_nll.set_xlabel(r"$\pi(m)$")
            ax_nll.set_xscale('symlog', linthresh=symlog_linthresh, linscale=0.1)

            # explicit tick labels for symlog
            ax_nll.set_xticks(symlog_ticks)
            ax_nll.set_xticklabels(symlog_ticklabels)

            ax_nll.set_xlim((0, 1))
            ax_nll.set_ylabel(r"$H(\pi(\theta_m|m),q_{\psi,\phi}(\theta_m|m))$")
            #ax_nll.set_ylim((0, 100))


            ax_nll.set_ylim((nll_lb, nll_ub))
            ax_nll.set_yscale('log')

            ax_nll.set_yticks([1e-3, 1e-2, 1e-1, 1, 10, nll_ub])
            ax_nll.set_yticklabels([ r"$\leq 10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^0$",  r"$10^{1}$",  r"$\geq 10^{2}$"])

            #ax_nll.set_yscale('linear')
            ax_nll.grid(True)

        # Adjust layout to prevent overlap
        plt.tight_layout()
        #plt.show()

        # Save the plot as PNG and PDF
        output_path_png = f"{pexp.output_dir}/{experiment_name}_{sweep_param}.png"
        output_path_pdf = f"{pexp.output_dir}/{experiment_name}_{sweep_param}.pdf"
        plt.savefig(output_path_png, format="png")
        plt.savefig(output_path_pdf, format="pdf")

        # Show the plot
        return plt.gcf(), results_df
    return (param_sweep_plot,)


@app.cell
async def _(
    base_kwargs,
    logging,
    param_sweep_plot,
    slurm_account,
    sweeping_trial,
):
    if not slurm_account:
        raise Exception("No SLURM ACCOUNT SPECIFIED")
    else:
        logging.info(f"slurm account {slurm_account}")

    predvstrue_fig, predvstrue_df = await param_sweep_plot(
        experiment_prefix="vti_robust_vs_noms_wide_prior_we_did_it_sfe_15k_noCE_sat15mar_prior1p5_shareddiagspline",
        sweeping_trial_fn=sweeping_trial,
        sweep_param="flow_type",
        #sweep_vals=["diagnorm", "affine5","spline23","spline46"],
        #sweep_vals=["diagnorm", "affine52f128","affine55f128","spline22f128","spline46f128"],
        #sweep_vals=["diagnorm", "affine52","affine55","spline22","spline46","diagspline22","diagspline46"],
        sweep_vals=["diagnorm", "affine52","affine55","spline22","spline46","shareddiagspline22","shareddiagspline46"],
        base_kwargs={**base_kwargs},
        num_replicates=10,
        executor_params={
            "timeout_min": 30,
            "tasks_per_node": 1,
            "mem_gb": 4,
            "cpus_per_task": 1,
            "gpus_per_node": 1,
            "slurm_account": slurm_account,
            "slurm_array_parallelism": 70,  # this throttles our job allocation and is requested by admins for disk-heavy workloads
        },
    )
    predvstrue_fig
    return predvstrue_df, predvstrue_fig


@app.cell(disabled=True)
def _(predvstrue_df):
    i=predvstrue_df.idxmax('time')
    predvstrue_df.iloc[i]
    return (i,)


@app.cell(disabled=True)
def _(bundles, logging, np, pd, plt, torch):
    sweep_param="flow_type"
    basedir='/scratch3/dav718/projects/virga/vti/'
    #outdir='/scratch3/dav718/projects/virga/vti/_experiments/vti_robust_vs_we_did_it_sfe_20k_flow_type_20250307-171250/outputs/'
    #dfpath = f'{outdir}vti_robust_vs_we_did_it_sfe_20k_flow_type_flow_type.df'
    #sweep_vals=["diagnorm", "affine55f128","spline23f128","spline410f128"]
    #outdir='/scratch3/dav718/projects/virga/vti/_experiments/vti_robust_vs_we_did_it_sfe_20k_eve_flow_type_20250307-184959/outputs/'
    #dfpath=f'{outdir}vti_robust_vs_we_did_it_sfe_20k_eve_flow_type_flow_type.df'

    #outdir='/scratch3/dav718/projects/virga/vti/_experiments/vti_robust_vs_we_did_it_sfe_20k_newce_sat8mar_flow_type_20250308-153441/outputs/'
    #dfpath=f'{outdir}vti_robust_vs_we_did_it_sfe_20k_newce_sat8mar_flow_type_flow_type.df'
    #sweep_vals=["diagnorm", "affine52f128","affine55f128","spline22f128","spline46f128"]


    #outdir=f'{basedir}_experiments/vti_robust_vs_mildmisspec_we_did_it_sfe_15k_noCE_fri14mar_f_flow_type_20250314-091634/outputs/'
    #dfpath=f'{outdir}vti_robust_vs_mildmisspec_we_did_it_sfe_15k_noCE_fri14mar_f_flow_type_flow_type.df'

    outdir=f'{basedir}_experiments/vti_robust_vs_mildmisspec_we_did_it_sfe_15k_noCE_fri14mar_prior1p5_flow_type_20250314-095530/outputs/'
    dfpath=f'{outdir}vti_robust_vs_mildmisspec_we_did_it_sfe_15k_noCE_fri14mar_prior1p5_flow_type_flow_type.df'

    #sweep_vals=["diagnorm", "affine52","affine55","spline22","spline46","diagspline22","diagspline46"]
    sweep_vals=["diagnorm", "affine52","spline22"]
    # dict for labels
    #labeldict = {"diagnorm":"Diagonal Gaussian MLP", "affine55f128": "Affine MAF (5,5,128)","spline23f128": "Spline MAF (2,3,128)","spline410f128" : "Spline MAF (4,10,128)"}
    labeldict = {"diagnorm":"Diagonal Gaussian MLP", 
                 "affine52f128": "Affine MAF (5,2,128)", 
                 "affine55f128": "Affine MAF (5,5,128)",
                 "spline22f128": "Spline MAF (2,2,128)", 
                 "spline23f128": "Spline MAF (2,3,128)",
                 "spline46f128" : "Spline MAF (4,6,128)", 
                 "spline410f128" : "Spline MAF (4,10,128)",
                 "affine52": "Affine MAF (5,2)", 
                 "affine55": "Affine MAF (5,5)",
                 "spline22": "Spline MAF (2,2)", 
                 "spline23": "Spline MAF (2,3)",
                 "spline46" : "Spline MAF (4,6)", 
                 "spline410" : "Spline MAF (4,10)",
                 "diagspline22": "Diag Gauss + Spline MAF (2,2)",
                 "diagspline46": "Diag Gauss + Spline MAF (4,6)",
                }
    df_saved = pd.read_pickle(dfpath)
    if True:

        results_df = df_saved
        logging.info("plotting")

        # Apply ICML 2024 style
        plt.rcParams.update(bundles.icml2024())
        #plt.rcParams.update({'font.size': 14})

        from matplotlib.cm import get_cmap

        # Number of columns is equal to the length of sweep_vals
        ncols = len(sweep_vals)
        nrows = 2  # Two rows as specified

        # Define a color map
        color_map = get_cmap('Dark2')
        num_colors = 10  # Set this based on the maximum expected rows per filtered_df


        symlog_linthresh=6e-4
        symlog_ticks=[0, 1e-3, 1e-2, 1e-1, 1]
        symlog_ticklabels=[r"$0$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^0$"]

        ONERUN = True
        runlim = 0

        nll_lb = 1e-3 # 1e-4

        for runlim in range(10):
            # Create a figure with subplots
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(2.5 * ncols, 2.5 * nrows))

            # Loop over each sweep value
            for col_idx, sv in enumerate(sweep_vals):
                filtered_df = results_df[results_df[sweep_param] == sv]
                num_rows = len(filtered_df)

                # Top row for predicted vs true probs
                ax_true_hat = axes[0, col_idx]
                #for idx, row_df in filtered_df.iterrows():   
                for idx, (_, row_df) in enumerate(filtered_df.iterrows()):
                    if ONERUN and idx != runlim:
                        continue
                    color = color_map(idx / num_rows)  # Get a color from the color map
                    mk_probs_true = row_df['mk_probs_true']
                    mk_probs_hat = row_df['mk_probs_hat']
                    mk_probs_true_np = mk_probs_true.cpu().numpy() if torch.is_tensor(mk_probs_true) else mk_probs_true
                    mk_probs_hat_np = mk_probs_hat.cpu().numpy() if torch.is_tensor(mk_probs_hat) else mk_probs_hat
                    ax_true_hat.scatter(mk_probs_true_np, mk_probs_hat_np, alpha=1,  marker='x',linewidths=0.5, s=20) #color=color,

                #ax_true_hat.plot([0, 1], [0, 1], 'k--', linewidth=0.5)
                ax_true_hat.plot([0, 1], [0, 1], linestyle='dashed',color='black', linewidth=0.5)
                for ofst in [1e-3,1e-2,1e-1]:
                    #ax_true_hat.plot([ofst, 1], [0, 1-ofst], linestyle='dotted',color='grey', linewidth=0.5)
                    #ax_true_hat.plot([0, 1-ofst], [ofst, 1], linestyle='dotted',color='grey', linewidth=0.5)
                    ax_true_hat.plot(np.linspace(ofst, 1,10000), np.linspace(0, 1-ofst,10000), linestyle='dotted',color='grey', linewidth=0.5)
                    ax_true_hat.plot(np.linspace(0, 1-ofst,10000), np.linspace(ofst, 1,10000), linestyle='dotted',color='grey', linewidth=0.5)
                ax_true_hat.set_xlim((0, 1))
                ax_true_hat.set_ylim((0, 1))
                ax_true_hat.set_title(f'{labeldict[sv]}')
                ax_true_hat.set_xlabel(r"$\pi(m)$")
                ax_true_hat.set_xscale('symlog', linthresh=symlog_linthresh, linscale=0.1)
                ax_true_hat.set_yscale('symlog', linthresh=symlog_linthresh, linscale=0.1)

                # explicit tick labels for symlog
                ax_true_hat.set_xticks(symlog_ticks)
                ax_true_hat.set_xticklabels(symlog_ticklabels)
                ax_true_hat.set_yticks(symlog_ticks)
                ax_true_hat.set_yticklabels(symlog_ticklabels)

                ax_true_hat.set_ylabel(r"$q_{\psi}(m)$")
                #ax_true_hat.grid(True,color='grey',linewidth=0.5,linestyle='dotted')

                # Bottom row for cond nll per model
                ax_nll = axes[1, col_idx]
                for idx, (_, row_df) in enumerate(filtered_df.iterrows()):
                    #logging.info(f'idx {idx} {col_idx}')
                    if ONERUN and idx != runlim:
                        continue
                    color = color_map(idx / num_rows)  # Get a color from the color map
                    mk_cond_nll = row_df['mk_cond_nll']
                    mk_probs_hat = row_df['mk_probs_hat']
                    mk_probs_true = row_df['mk_probs_true']
                    mk_cond_nll_np = mk_cond_nll.cpu().numpy() if torch.is_tensor(mk_cond_nll) else mk_cond_nll
                    mk_cond_nll_np = np.clip(mk_cond_nll_np, a_min=0.0, a_max=None)
                    mk_probs_hat_np = mk_probs_hat.cpu().numpy() if torch.is_tensor(mk_probs_hat) else mk_probs_hat
                    mk_probs_true_np = mk_probs_true.cpu().numpy() if torch.is_tensor(mk_probs_true) else mk_probs_true
                    #ax_nll.scatter(mk_probs_hat_np, mk_cond_nll_np, alpha=0.6,  marker='x',linewidths=1.5) #color=color,
                    ax_nll.scatter(mk_probs_true_np, np.clip(mk_cond_nll_np, nll_lb, None), alpha=1,  marker='x',linewidths=0.5, s=20) #color=color,

                #ax_nll.set_xlabel(r"$q_{\psi}(m)$")
                ax_nll.set_xlabel(r"$\pi(m)$")
                ax_nll.set_xscale('symlog', linthresh=symlog_linthresh, linscale=0.1)

                # explicit tick labels for symlog
                ax_nll.set_xticks(symlog_ticks)
                ax_nll.set_xticklabels(symlog_ticklabels)

                ax_nll.set_xlim((0, 1))
                ax_nll.set_ylabel(r"$H(\pi(\theta_m|m),q_{\psi,\phi}(\theta_m|m))$")

                #ax_nll.set_ylim((0, 100))
                #ax_nll.set_yscale('linear')

                ax_nll.set_ylim((nll_lb, 100))
                ax_nll.set_yscale('log')

                #ax_nll.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2])
                #ax_nll.set_yticklabels([ r"$\leq 10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^0$",  r"$10^{1}$",  r"$\geq 10^{2}$"])

                ax_nll.set_yticks([1e-3, 1e-2, 1e-1, 1, 10, 1e2])
                ax_nll.set_yticklabels([ r"$\leq 10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^0$",  r"$10^{1}$",  r"$\geq 10^{2}$"])

                ax_nll.grid(True,color='grey',linewidth=0.5,linestyle='dotted')

            # Adjust layout to prevent overlap
            plt.tight_layout()

            output_path_png = f"{outdir}/plot_{runlim}.png"
            output_path_pdf = f"{outdir}/plot_{runlim}.pdf"
            plt.savefig(output_path_png, format="png")
            plt.savefig(output_path_pdf, format="pdf")

            plt.show()
    return (
        ONERUN,
        ax_nll,
        ax_true_hat,
        axes,
        basedir,
        col_idx,
        color,
        color_map,
        df_saved,
        dfpath,
        fig,
        filtered_df,
        get_cmap,
        idx,
        labeldict,
        mk_cond_nll,
        mk_cond_nll_np,
        mk_probs_hat,
        mk_probs_hat_np,
        mk_probs_true,
        mk_probs_true_np,
        ncols,
        nll_lb,
        nrows,
        num_colors,
        num_rows,
        ofst,
        outdir,
        output_path_pdf,
        output_path_png,
        results_df,
        row_df,
        runlim,
        sv,
        sweep_param,
        sweep_vals,
        symlog_linthresh,
        symlog_ticklabels,
        symlog_ticks,
    )


@app.cell(disabled=True)
def _():
    return


@app.cell
def _(bundles, logging, np, pd, plt, torch):
    def plotall2():
        import math
        sweep_param="flow_type"
        basedir='/scratch3/dav718/projects/virga/vti/'

        #outdir=f'{basedir}_experiments/vti_robust_vs_mildmisspec_we_did_it_sfe_15k_noCE_fri14mar_prior1p5_flow_type_20250314-104008/outputs/'
        #dfpath=f'{outdir}vti_robust_vs_mildmisspec_we_did_it_sfe_15k_noCE_fri14mar_prior1p5_flow_type_flow_type.df'

        #outdir = f'{basedir}_experiments/vti_robust_vs_mildmisspec_we_did_it_sfe_15k_noCE_fri14mar_prior1p5_lindiagspline_flow_type_20250314-181015/outputs/'
        #dfpath = f'{outdir}vti_robust_vs_mildmisspec_we_did_it_sfe_15k_noCE_fri14mar_prior1p5_lindiagspline_flow_type_flow_type.df'

        if False:
            # prior 1.5 scale
            # high misspec
            outdir = f'{basedir}_experiments/vti_robust_vs_mildmisspec_we_did_it_sfe_15k_noCE_fri14mar_prior1p5_shareddiagspline_flow_type_20250314-232259/outputs/'
            dfpath = f'{outdir}vti_robust_vs_mildmisspec_we_did_it_sfe_15k_noCE_fri14mar_prior1p5_shareddiagspline_flow_type_flow_type.df'
        elif False:
            # prior 10. scale
            # high misspec
            outdir = f'{basedir}_experiments/vti_robust_vs_wideprior_we_did_it_sfe_15k_noCE_sat15mar_prior1p5_shareddiagspline_flow_type_20250315-114508/outputs/'
            dfpath = f'{outdir}vti_robust_vs_wideprior_we_did_it_sfe_15k_noCE_sat15mar_prior1p5_shareddiagspline_flow_type_flow_type.df'
        elif False:
            # prior 10. scale
            # mid misspec (no correlation)
            outdir = f'{basedir}_experiments/vti_robust_vs_midms_wideprior_we_did_it_sfe_15k_noCE_sat15mar_prior1p5_shareddiagspline_flow_type_20250315-131153/outputs/'
            dfpath = f'{outdir}vti_robust_vs_midms_wideprior_we_did_it_sfe_15k_noCE_sat15mar_prior1p5_shareddiagspline_flow_type_flow_type.df'
        elif False:
            # prior 1.5 scale
            # mid misspec (no correlation)
            outdir = f'{basedir}_experiments/vti_robust_vs_midms_we_did_it_sfe_15k_noCE_sat15mar_prior1p5_shareddiagspline_flow_type_20250315-135745/outputs/'
            dfpath = f'{outdir}vti_robust_vs_midms_we_did_it_sfe_15k_noCE_sat15mar_prior1p5_shareddiagspline_flow_type_flow_type.df'
        elif False:
            # prior 1.5 scale
            # no misspec
            outdir = f'{basedir}_experiments/vti_robust_vs_noms_we_did_it_sfe_15k_noCE_sat15mar_prior1p5_shareddiagspline_flow_type_20250315-150930/outputs/'
            dfpath = f'{outdir}vti_robust_vs_noms_we_did_it_sfe_15k_noCE_sat15mar_prior1p5_shareddiagspline_flow_type_flow_type.df'
        elif True:
            # prior 10. scale
            # no misspec
            outdir = f'{basedir}_experiments/vti_robust_vs_noms_wide_prior_we_did_it_sfe_15k_noCE_sat15mar_prior1p5_shareddiagspline_flow_type_20250315-155255/outputs/'
            dfpath = f'{outdir}vti_robust_vs_noms_wide_prior_we_did_it_sfe_15k_noCE_sat15mar_prior1p5_shareddiagspline_flow_type_flow_type.df'
        
        sweep_vals=["diagnorm","affine55","shareddiagspline46"]
        #sweep_vals=["diagnorm", "affine52","affine55","spline22","spline46","shareddiagspline22","shareddiagspline46"]
        #sweep_vals=["diagnorm", "affine52","affine55","spline22","spline46","diagspline22","diagspline46"]
        #sweep_vals=["diagnorm", "affine52", "affine55","spline22", "spline46"]
        # dict for labels
        #labeldict = {"diagnorm":"Diagonal Gaussian MLP", "affine55f128": "Affine MAF (5,5,128)","spline23f128": "Spline MAF (2,3,128)","spline410f128" : "Spline MAF (4,10,128)"}
        labeldictall = {"diagnorm":"Diagonal Gaussian MLP", 
                     "affine52": "Affine MAF (5,2)", 
                     "affine55": "Affine MAF (5,5)",
                     "spline22": "Spline MAF (2,2)", 
                     "spline23": "Spline MAF (2,3)",
                     "spline46" : "Spline MAF (4,6)", 
                     "spline410" : "Spline MAF (4,10)",
                     "diagspline22": "Diag Gauss + Spline MAF (2,2)",
                     "diagspline46": "Diag Gauss + Spline MAF (4,6)",
                     "affine52f128": "Affine MAF (5,2,128)", 
                     "affine55f128": "Affine MAF (5,5,128)",
                     "spline22f128": "Spline MAF (2,2,128)", 
                     "spline23f128": "Spline MAF (2,3,128)",
                     "spline46f128" : "Spline MAF (4,6,128)", 
                     "spline410f128" : "Spline MAF (4,10,128)",
                     "shareddiagspline22": "Shared loc scale + Spline MAF (2,2)",
                     "shareddiagspline46": "Shared loc scale + Spline MAF (4,6)",
                    }
        logging.info(f"label dict {labeldictall}")
        df_saved = pd.read_pickle(dfpath)


        results_df = df_saved
        logging.info("plotting")

        # Apply ICML 2024 style
        plt.rcParams.update(bundles.icml2024())
        #plt.rcParams.update({'font.size': 14})

        from matplotlib.cm import get_cmap

        # Number of columns is equal to the length of sweep_vals
        ncols = len(sweep_vals)
        nrows = 2  # Two rows as specified

        # Define a color map
        color_map = get_cmap('Dark2')
        num_colors = 10  # Set this based on the maximum expected rows per filtered_df


        symlog_linthresh=6e-4
        symlog_ticks=[0, 1e-3, 1e-2, 1e-1, 1]
        symlog_ticklabels=[r"$0$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^0$"]

        nll_lb = 1e-2 # 1e-4
        nll_ub = 1000

        nll_yticks = [10**i for i in range(int(math.log10(nll_lb)), int(math.log10(nll_ub)) + 1)]
        nll_yticklabels = [
            r"$\leq 10^{" + f"{int(math.log10(nll_lb))}" + r"}$" if tick == nll_yticks[0]
            else r"$\geq 10^{" + f"{int(math.log10(nll_ub))}" + r"}$" if tick == nll_yticks[-1]
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

        #dgp_indices={0:68,1:66,2:96,3:96,4:24,5:36,6:24,7:5,8:12,9:96,10:68,11:66,12:96,13:96,14:24,15:36,16:24,17:5,18:12,19:96,20:68,21:66,22:96,23:96,24:24,25:36,26:24,27:5,28:12,29:96,30:68,31:66,32:96,33:96,34:24,35:36,36:24,37:5,38:12,39:96,40:68,41:66,42:96,43:96,44:24,45:36,46:24,47:5,48:12,49:96,50:68,51:66,52:96,53:96,54:24,55:36,56:24,57:5,58:12,59:96,60:68,61:66,62:96,63:96,64:24,65:36,66:24,67:5,68:12,69:96}
        dgp_indices={
            0:17,
            1:33,
            2:3,
            3:3,
            4:12,
            5:18,
            6:12,
            7:80,
            8:24,
            9:3,
        }

        # Create a figure with subplots
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(3 * ncols, 2.5 * nrows))

        # Loop over each sweep value
        for col_idx, sv in enumerate(sweep_vals):
            filtered_df = results_df[results_df[sweep_param] == sv]
            num_rows = len(filtered_df)

            # Top row for predicted vs true probs
            ax_true_hat = axes[0, col_idx]
            #for idx, row_df in filtered_df.iterrows():   
            for idx, (run_id, row_df) in enumerate(filtered_df.iterrows()):
                if idx in exclude:
                    continue
                color = color_map(idx / float(num_rows-len(exclude)))  # Get a color from the color map
                mk_probs_true = row_df['mk_probs_true']
                mk_probs_hat = row_df['mk_probs_hat']
                mk_probs_true_np = mk_probs_true.cpu().numpy() if torch.is_tensor(mk_probs_true) else mk_probs_true
                mk_probs_hat_np = mk_probs_hat.cpu().numpy() if torch.is_tensor(mk_probs_hat) else mk_probs_hat
                if USE_SYMLOG:
                    ax_true_hat.scatter(mk_probs_true_np, mk_probs_hat_np, alpha=1,  marker='x',linewidths=0.5, s=10) #color=color,
                else:
                    spl = ax_true_hat.scatter(np.clip(mk_probs_true_np, prob_lb, prob_ub), np.clip(mk_probs_hat_np, prob_lb, prob_ub), alpha=.7,  marker='x',linewidths=0.25, s=10,zorder=idx) 
                    splcolor = spl.get_facecolor()[0]
                    ax_true_hat.scatter(mk_probs_true_np[dgp_indices[idx]],mk_probs_hat_np[dgp_indices[idx]],alpha=1,  marker='^',linewidths=1, s=50, facecolors='none', edgecolors=splcolor, zorder=20*idx)
                    ax_true_hat.scatter(mk_probs_true_np[0],mk_probs_hat_np[0],alpha=1,  marker='o',linewidths=1, s=50, facecolors='none', edgecolors=splcolor, zorder=20*idx) # null model
                    logging.info(f"color for {idx} is {splcolor}")
                    logging.info(f'Model {idx} prob rjmcmc {mk_probs_true_np[dgp_indices[idx]]} vs VTI {mk_probs_hat_np[dgp_indices[idx]]}')

            #ax_true_hat.plot([0, 1], [0, 1], 'k--', linewidth=0.5)
            ax_true_hat.plot([0, 1], [0, 1], linestyle='dashed',color='black', linewidth=0.5)
            #for ofst in [1e-3,1e-2,1e-1]:
            for ofst in prob_yticks[1:-1]:
                ax_true_hat.plot(np.linspace(ofst, 1,10000), np.linspace(0, 1-ofst,10000), linestyle='dotted',color='grey', linewidth=0.5)
                ax_true_hat.plot(np.linspace(0, 1-ofst,10000), np.linspace(ofst, 1,10000), linestyle='dotted',color='grey', linewidth=0.5)

            logging.info(f'labeldict {labeldictall} {sv}')
            ax_true_hat.set_title(f'{labeldictall[sv]}')
            ax_true_hat.set_xlabel(r"$\pi(m)$")
            if USE_SYMLOG:
                ax_true_hat.set_xlim((0, 1))
                ax_true_hat.set_ylim((0, 1))
                ax_true_hat.set_xscale('symlog', linthresh=symlog_linthresh, linscale=0.1)
                ax_true_hat.set_yscale('symlog', linthresh=symlog_linthresh, linscale=0.1)
                # explicit tick labels for symlog
                ax_true_hat.set_xticks(symlog_ticks)
                ax_true_hat.set_xticklabels(symlog_ticklabels)
                ax_true_hat.set_yticks(symlog_ticks)
                ax_true_hat.set_yticklabels(symlog_ticklabels)
            else:
                ax_true_hat.set_xlim((prob_lb, prob_ub))
                ax_true_hat.set_ylim((prob_lb, prob_ub))
                ax_true_hat.set_xscale('log')
                ax_true_hat.set_yscale('log')
                # ticks for probs in log scale
                ax_true_hat.set_xticks(prob_yticks)
                ax_true_hat.set_xticklabels(prob_yticklabels)
                ax_true_hat.set_yticks(prob_yticks)
                ax_true_hat.set_yticklabels(prob_yticklabels)


            ax_true_hat.set_ylabel(r"$q_{\psi}(m)$")
            #ax_true_hat.grid(True,color='grey',linewidth=0.5,linestyle='dotted')

            # Bottom row for cond nll per model
            ax_nll = axes[1, col_idx]
            for idx, (run_id, row_df) in enumerate(filtered_df.iterrows()):
                if idx in exclude:
                    continue
                #logging.info(f'idx {idx} {col_idx}')
                color = color_map(idx / float(num_rows-len(exclude)))  # Get a color from the color map
                mk_cond_nll = row_df['mk_cond_nll']
                mk_probs_hat = row_df['mk_probs_hat']
                mk_probs_true = row_df['mk_probs_true']
                mk_cond_nll_np = mk_cond_nll.cpu().numpy() if torch.is_tensor(mk_cond_nll) else mk_cond_nll
                mk_cond_nll_np = np.clip(mk_cond_nll_np, a_min=0.0, a_max=None)
                mk_probs_hat_np = mk_probs_hat.cpu().numpy() if torch.is_tensor(mk_probs_hat) else mk_probs_hat
                mk_probs_true_np = mk_probs_true.cpu().numpy() if torch.is_tensor(mk_probs_true) else mk_probs_true
                #ax_nll.scatter(mk_probs_hat_np, mk_cond_nll_np, alpha=0.6,  marker='x',linewidths=1.5) #color=color,
                if USE_SYMLOG:
                    ax_nll.scatter(mk_probs_true_np, np.clip(mk_cond_nll_np, nll_lb, nll_ub), alpha=1,  marker='x',linewidths=0.5, s=10) #color=color,
                else:
                    spl = ax_nll.scatter( np.clip(mk_probs_true_np, prob_lb, prob_ub), np.clip(mk_cond_nll_np, nll_lb, nll_ub), alpha=.7,  marker='x',linewidths=0.25, s=10, zorder=idx)
                    splcolor = spl.get_facecolor()[0]
                    ax_nll.scatter(mk_probs_true_np[dgp_indices[idx]],mk_cond_nll_np[dgp_indices[idx]],alpha=1,  marker='^',linewidths=1, s=50, facecolors='none', edgecolors=splcolor, zorder=20*idx)
                    ax_nll.scatter(mk_probs_true_np[0],mk_cond_nll_np[0],alpha=1,  marker='o',linewidths=1, s=50, facecolors='none', edgecolors=splcolor, zorder=10*idx) # null model, no covariates

            #ax_nll.set_xlabel(r"$q_{\psi}(m)$")
            ax_nll.set_xlabel(r"$\pi(m)$")
            if USE_SYMLOG:
                ax_nll.set_xscale('symlog', linthresh=symlog_linthresh, linscale=0.1)

                # explicit tick labels for symlog
                ax_nll.set_xticks(symlog_ticks)
                ax_nll.set_xticklabels(symlog_ticklabels)

                ax_nll.set_xlim((0, 1))
            else:
                ax_nll.set_xscale('log')

                # explicit tick labels for log
                ax_nll.set_xticks(prob_yticks)
                ax_nll.set_xticklabels(prob_yticklabels)

                ax_nll.set_xlim((prob_lb, prob_ub))

            ax_nll.set_ylabel(r"$H(\pi(\theta_m|m),q_{\psi,\phi}(\theta_m|m))$")

            #ax_nll.set_ylim((0, 100))
            #ax_nll.set_yscale('linear')

            ax_nll.set_ylim((nll_lb, 100))
            ax_nll.set_yscale('log')

            #ax_nll.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2])
            #ax_nll.set_yticklabels([ r"$\leq 10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^0$",  r"$10^{1}$",  r"$\geq 10^{2}$"])

            #ax_nll.set_yticks([1e-3, 1e-2, 1e-1, 1, 10, 1e2])
            #ax_nll.set_yticklabels([ r"$\leq 10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^0$",  r"$10^{1}$",  r"$\geq 10^{2}$"])

            ax_nll.set_yticks(nll_yticks)
            ax_nll.set_yticklabels(nll_yticklabels)

            ax_nll.grid(True,color='grey',linewidth=0.5,linestyle='dotted')

        # Adjust layout to prevent overlap
        plt.tight_layout()

        output_path_png = f"{outdir}/plot_all.png"
        output_path_pdf = f"{outdir}/plot_all.pdf"
        plt.savefig(output_path_png, format="png")
        plt.savefig(output_path_pdf, format="pdf")

        plt.show()

    plotall2()
    return (plotall2,)


@app.cell
def _(bundles, logging, np, pd, plt, torch):
    def plotfig1():
        import math
        sweep_param="flow_type"
        basedir='/scratch3/dav718/projects/virga/vti/'

        if False:
            # prior 1.5 scale
            # high misspec
            outdir = f'{basedir}_experiments/vti_robust_vs_mildmisspec_we_did_it_sfe_15k_noCE_fri14mar_prior1p5_shareddiagspline_flow_type_20250314-232259/outputs/'
            dfpath = f'{outdir}vti_robust_vs_mildmisspec_we_did_it_sfe_15k_noCE_fri14mar_prior1p5_shareddiagspline_flow_type_flow_type.df'
        elif False:
            # prior 10. scale
            # high misspec
            outdir = f'{basedir}_experiments/vti_robust_vs_wideprior_we_did_it_sfe_15k_noCE_sat15mar_prior1p5_shareddiagspline_flow_type_20250315-114508/outputs/'
            dfpath = f'{outdir}vti_robust_vs_wideprior_we_did_it_sfe_15k_noCE_sat15mar_prior1p5_shareddiagspline_flow_type_flow_type.df'
        elif False:
            # prior 10. scale
            # mid misspec (no correlation)
            outdir = f'{basedir}_experiments/vti_robust_vs_midms_wideprior_we_did_it_sfe_15k_noCE_sat15mar_prior1p5_shareddiagspline_flow_type_20250315-131153/outputs/'
            dfpath = f'{outdir}vti_robust_vs_midms_wideprior_we_did_it_sfe_15k_noCE_sat15mar_prior1p5_shareddiagspline_flow_type_flow_type.df'
        elif False:
            # prior 1.5 scale
            # mid misspec (no correlation)
            outdir = f'{basedir}_experiments/vti_robust_vs_midms_we_did_it_sfe_15k_noCE_sat15mar_prior1p5_shareddiagspline_flow_type_20250315-135745/outputs/'
            dfpath = f'{outdir}vti_robust_vs_midms_we_did_it_sfe_15k_noCE_sat15mar_prior1p5_shareddiagspline_flow_type_flow_type.df'
        elif False:
            # prior 1.5 scale
            # no misspec
            outdir = f'{basedir}_experiments/vti_robust_vs_noms_we_did_it_sfe_15k_noCE_sat15mar_prior1p5_shareddiagspline_flow_type_20250315-150930/outputs/'
            dfpath = f'{outdir}vti_robust_vs_noms_we_did_it_sfe_15k_noCE_sat15mar_prior1p5_shareddiagspline_flow_type_flow_type.df'
        elif False:
            # prior 10. scale
            # no misspec
            outdir = f'{basedir}_experiments/vti_robust_vs_noms_wide_prior_we_did_it_sfe_15k_noCE_sat15mar_prior1p5_shareddiagspline_flow_type_20250315-155255/outputs/'
            dfpath = f'{outdir}vti_robust_vs_noms_wide_prior_we_did_it_sfe_15k_noCE_sat15mar_prior1p5_shareddiagspline_flow_type_flow_type.df'

        # prior 1.5 scale
        # mid misspec (no correlation)
        outdir_l = f'{basedir}_experiments/vti_robust_vs_midms_we_did_it_sfe_15k_noCE_sat15mar_prior1p5_shareddiagspline_flow_type_20250315-135745/outputs/'
        dfpath_l = f'{outdir_l}vti_robust_vs_midms_we_did_it_sfe_15k_noCE_sat15mar_prior1p5_shareddiagspline_flow_type_flow_type.df'
        # prior 1.5 scale
        # high misspec
        outdir_r = f'{basedir}_experiments/vti_robust_vs_mildmisspec_we_did_it_sfe_15k_noCE_fri14mar_prior1p5_shareddiagspline_flow_type_20250314-232259/outputs/'
        dfpath_r = f'{outdir_r}vti_robust_vs_mildmisspec_we_did_it_sfe_15k_noCE_fri14mar_prior1p5_shareddiagspline_flow_type_flow_type.df'
        
        sweep_vals=["diagnorm","affine55","shareddiagspline46"]
        #sweep_vals=["diagnorm", "affine52","affine55","spline22","spline46","shareddiagspline22","shareddiagspline46"]
        #sweep_vals=["diagnorm", "affine52","affine55","spline22","spline46","diagspline22","diagspline46"]
        #sweep_vals=["diagnorm", "affine52", "affine55","spline22", "spline46"]
        # dict for labels
        #labeldict = {"diagnorm":"Diagonal Gaussian MLP", "affine55f128": "Affine MAF (5,5,128)","spline23f128": "Spline MAF (2,3,128)","spline410f128" : "Spline MAF (4,10,128)"}
        labeldictall = {"diagnorm":"Diagonal Gaussian MLP", 
                     "affine52": "Affine MAF (5,2)", 
                     "affine55": "Affine MAF (5,5)",
                     "spline22": "Spline MAF (2,2)", 
                     "spline23": "Spline MAF (2,3)",
                     "spline46" : "Spline MAF (4,6)", 
                     "spline410" : "Spline MAF (4,10)",
                     "diagspline22": "Diag Gauss + Spline MAF (2,2)",
                     "diagspline46": "Diag Gauss + Spline MAF (4,6)",
                     "affine52f128": "Affine MAF (5,2,128)", 
                     "affine55f128": "Affine MAF (5,5,128)",
                     "spline22f128": "Spline MAF (2,2,128)", 
                     "spline23f128": "Spline MAF (2,3,128)",
                     "spline46f128" : "Spline MAF (4,6,128)", 
                     "spline410f128" : "Spline MAF (4,10,128)",
                     #"shareddiagspline22": "Shared loc scale + Spline MAF (2,2)",
                     #"shareddiagspline46": "Shared loc scale + Spline MAF (4,6)",
                     "shareddiagspline22": "Spline MAF (2,2)",
                     "shareddiagspline46": "Spline MAF (4,6)",
                    }
        logging.info(f"label dict {labeldictall}")
        #dgp_indices={0:68,1:66,2:96,3:96,4:24,5:36,6:24,7:5,8:12,9:96,10:68,11:66,12:96,13:96,14:24,15:36,16:24,17:5,18:12,19:96,20:68,21:66,22:96,23:96,24:24,25:36,26:24,27:5,28:12,29:96,30:68,31:66,32:96,33:96,34:24,35:36,36:24,37:5,38:12,39:96,40:68,41:66,42:96,43:96,44:24,45:36,46:24,47:5,48:12,49:96,50:68,51:66,52:96,53:96,54:24,55:36,56:24,57:5,58:12,59:96,60:68,61:66,62:96,63:96,64:24,65:36,66:24,67:5,68:12,69:96}
        dgp_indices={
            0:17,
            1:33,
            2:3,
            3:3,
            4:12,
            5:18,
            6:12,
            7:80,
            8:24,
            9:3,
        }

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
        logging.info("plotting")

        for isright,dfpath in enumerate([dfpath_l, dfpath_r]):
            df_saved = pd.read_pickle(dfpath)
            results_df = df_saved

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
            if False:
                nll_yticklabels = [
                    r"$\leq 10^{" + f"{int(math.log10(nll_lb))}" + r"}$" if tick == nll_yticks[0]
                    else r"$\geq 10^{" + f"{int(math.log10(nll_ub))}" + r"}$" if tick == nll_yticks[-1]
                    else rf"$10^{{{int(math.log10(tick))}}}$"
                    for tick in nll_yticks
                ]
            else:
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
                    if USE_SYMLOG:
                        ax_true_hat.scatter(mk_probs_true_np, mk_probs_hat_np, alpha=1,  marker='x',linewidths=0.5, s=10) #color=color,
                    else:
                        spl = ax_true_hat.scatter(np.clip(mk_probs_true_np, prob_lb, prob_ub), np.clip(mk_probs_hat_np, prob_lb, prob_ub), alpha=.9,  marker='x',linewidths=0.25, s=10,zorder=idx) 
                        if False:
                            # do we plot true DGP triangle and null model square
                            splcolor = spl.get_facecolor()[0]
                            ax_true_hat.scatter(mk_probs_true_np[dgp_indices[idx]],mk_probs_hat_np[dgp_indices[idx]],alpha=1,  marker='^',linewidths=1, s=50, facecolors='none', edgecolors=splcolor, zorder=20*idx)
                            ax_true_hat.scatter(mk_probs_true_np[0],mk_probs_hat_np[0],alpha=1,  marker='o',linewidths=1, s=50, facecolors='none', edgecolors=splcolor, zorder=20*idx) # null model
                            logging.info(f"color for {idx} is {splcolor}")
                            logging.info(f'Model {idx} prob rjmcmc {mk_probs_true_np[dgp_indices[idx]]} vs VTI {mk_probs_hat_np[dgp_indices[idx]]}')
        
                #ax_true_hat.plot([0, 1], [0, 1], 'k--', linewidth=0.5)
                ax_true_hat.plot([0, 1], [0, 1], linestyle='dashed',color='black', linewidth=0.5)
                #for ofst in [1e-3,1e-2,1e-1]:
                for ofst in prob_yticks[1:-1]:
                    ax_true_hat.plot(np.linspace(ofst, 1,10000), np.linspace(0, 1-ofst,10000), linestyle='dotted',color='grey', linewidth=0.5)
                    ax_true_hat.plot(np.linspace(0, 1-ofst,10000), np.linspace(ofst, 1,10000), linestyle='dotted',color='grey', linewidth=0.5)
        
                logging.info(f'labeldict {labeldictall} {sv}')
                if isright==1:
                    dgptitle='Misspecification level: High'
                else:
                    dgptitle='Misspecification level: Medium'
                ax_true_hat.set_title(f'{dgptitle}\n{labeldictall[sv]}')
                if False:
                    ax_true_hat.set_xlabel(r"$\pi(m)$")
                if USE_SYMLOG:
                    ax_true_hat.set_xlim((0, 1))
                    ax_true_hat.set_ylim((0, 1))
                    ax_true_hat.set_xscale('symlog', linthresh=symlog_linthresh, linscale=0.1)
                    ax_true_hat.set_yscale('symlog', linthresh=symlog_linthresh, linscale=0.1)
                    # explicit tick labels for symlog
                    ax_true_hat.set_xticks(symlog_ticks)
                    ax_true_hat.set_xticklabels(symlog_ticklabels)
                    ax_true_hat.set_yticks(symlog_ticks)
                    ax_true_hat.set_yticklabels(symlog_ticklabels)
                else:
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
                    #logging.info(f'idx {idx} {col_idx}')
                    color = color_map(idx / float(num_rows-len(exclude)))  # Get a color from the color map
                    mk_cond_nll = row_df['mk_cond_nll']
                    mk_probs_hat = row_df['mk_probs_hat']
                    mk_probs_true = row_df['mk_probs_true']
                    mk_cond_nll_np = mk_cond_nll.cpu().numpy() if torch.is_tensor(mk_cond_nll) else mk_cond_nll
                    mk_cond_nll_np = np.clip(mk_cond_nll_np, a_min=0.0, a_max=None)
                    mk_probs_hat_np = mk_probs_hat.cpu().numpy() if torch.is_tensor(mk_probs_hat) else mk_probs_hat
                    mk_probs_true_np = mk_probs_true.cpu().numpy() if torch.is_tensor(mk_probs_true) else mk_probs_true
                    #ax_nll.scatter(mk_probs_hat_np, mk_cond_nll_np, alpha=0.6,  marker='x',linewidths=1.5) #color=color,
                    if USE_SYMLOG:
                        ax_nll.scatter(mk_probs_true_np, np.clip(mk_cond_nll_np, nll_lb, nll_ub), alpha=1,  marker='x',linewidths=0.5, s=10) #color=color,
                    else:
                        spl = ax_nll.scatter( np.clip(mk_probs_true_np, prob_lb, prob_ub), np.clip(mk_cond_nll_np, nll_lb, nll_ub), alpha=.9,  marker='x',linewidths=0.25, s=10, zorder=idx)
                        splcolor = spl.get_facecolor()[0]
                        logging.info(f'color for {idx} is {splcolor}')
                        if False:
                            # do we plot true DGP triangle and null model square
                            splcolor = spl.get_facecolor()[0]
                            ax_nll.scatter(mk_probs_true_np[dgp_indices[idx]],mk_cond_nll_np[dgp_indices[idx]],alpha=1,  marker='^',linewidths=1, s=50, facecolors='none', edgecolors=splcolor, zorder=20*idx)
                            ax_nll.scatter(mk_probs_true_np[0],mk_cond_nll_np[0],alpha=1,  marker='o',linewidths=1, s=50, facecolors='none', edgecolors=splcolor, zorder=10*idx) # null model, no covariates
        
                #ax_nll.set_xlabel(r"$q_{\psi}(m)$")
                ax_nll.set_xlabel(r"$\pi(m)$")
                if USE_SYMLOG:
                    ax_nll.set_xscale('symlog', linthresh=symlog_linthresh, linscale=0.1)
        
                    # explicit tick labels for symlog
                    ax_nll.set_xticks(symlog_ticks)
                    ax_nll.set_xticklabels(symlog_ticklabels)
        
                    ax_nll.set_xlim((0, 1))
                else:
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

        output_path_png = f"{basedir}/fig1_sun16mar.png"
        output_path_pdf = f"{basedir}/fig1_sun16mar.pdf"
        plt.savefig(output_path_png, format="png")
        plt.savefig(output_path_pdf, format="pdf")

        plt.show()

    plotfig1()
    return (plotfig1,)


if __name__ == "__main__":
    app.run()
