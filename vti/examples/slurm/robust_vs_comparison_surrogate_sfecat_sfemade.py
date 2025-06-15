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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Basic imports""")
    return


@app.cell(hide_code=True)
def _():
    from pathlib import Path
    from time import time
    import os
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
    base_kwargs = dict(
        num_iterations=30000,
        batch_size=1024,
        seed=4,
        dgp_key="robust_vs_6",
        dgp_seed="dgpseedfn1000",
        #flow_type="affine510",
        flow_type="shareddiagspline46",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="float64",
        plot=False,
        # job_id=None
    )
    sfe_cat_kwargs = dict(
        ig_threshold=5e-4,
        #num_iterations=50000,
        # model sampler parameters
        # model_sampler_key="SFEMADEBinaryString",
        model_sampler_key="SFECategoricalBinaryString",
    )
    sfe_made_kwargs = dict(
        ig_threshold=5e-4,
        # model sampler parameters
        model_sampler_key="SFEMADEBinaryString",
        # model_sampler_key="SFECategoricalBinaryString",
    )
    diag_surrogate_kwargs = dict(
        batch_size=128,
        model_sampler_key="BSSSS",
        ## surrogate parameters
        surrogate_type="diagnorm",
        basis_rank=50,
        basis_reduction="random",
        basis_normalize="False",
        f_coupling=1e2,
        prior_diag_variance=1e2,
        lr_variance_scale=10.,
        obs_variance=1.0,
        obs_beta=0.2,
        max_entropy_gain=0.01,  # per obs
        squish_utility=False,
        diffuse_prior=1.0,
    )
    ens_surrogate_kwargs = dict(
        model_sampler_key="BSSSS",
        ## surrogate parameters
        surrogate_type="ensemble",
        basis_rank=50,
        basis_reduction="random",
        basis_normalize="False",
        f_coupling=1e2,
        prior_diag_variance=1e2,
        lr_variance_scale=10.,
        obs_variance=1.0,
        obs_beta=0.2,
        max_entropy_gain=0.01,  # per obs
        squish_utility=False,
        diffuse_prior=1.0,
    )
    base_kwargs["sfe_cat_kwargs"] = sfe_cat_kwargs
    base_kwargs["sfe_made_kwargs"] = sfe_made_kwargs
    base_kwargs["diag_surrogate_kwargs"] = diag_surrogate_kwargs
    base_kwargs["ens_surrogate_kwargs"] = ens_surrogate_kwargs
    return (
        base_kwargs,
        diag_surrogate_kwargs,
        ens_surrogate_kwargs,
        sfe_cat_kwargs,
        sfe_made_kwargs,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## The main trial functions

        We set upon `combined_trial`, which creates a problem, fits a neural distribution to it, and returns various performance metrics.
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
def _(TransdimensionalFlow, base_kwargs, logging, sfe_trial, torch):
    def diagnostic_trial(**kwargs):
        """
        Verbose, messy  run, for ease of diagnosis.
        """
        res = sfe_trial(**kwargs, **base_kwargs)
        problem = res["problem"]

        RUN_RJMCMC = True

        if RUN_RJMCMC:
            # set up transd flow and log average NLL
            tdf = TransdimensionalFlow(
                problem.model_sampler,
                problem.param_transform,
                problem.dgp.reference_dist(),
            )
            # run rjmcmc for true model probs
            problem.dgp.sample_true_mk_probs(plot_bivariates=True, store_samples=True)
            mk_probs_true = (
                problem.dgp.true_mk_probs_all
            )  # should be returned from previous statement
            # log avg NLL
            nll = problem.dgp.compute_average_nll(tdf)
            logging.info(f"Average NLL = {nll}")

        # mk_probs_hat = problem.model_sampler.probs() # not implemented for SFE
        mk_identifiers = problem.dgp.mk_identifiers()
        mk_log_probs_hat = problem.model_sampler.log_prob(mk_identifiers)
        if not RUN_RJMCMC:
            mk_probs_true = torch.zeros_like(mk_log_probs_hat)

        # logging.info("Comparison of log probs exp and softmax")
        # logging.info(f"{torch.column_stack([mk_log_probs_hat.exp(),torch.nn.functional.softmax(mk_log_probs_hat,dim=-1)])}")
        mk_probs_hat = mk_log_probs_hat.exp()

        # mk_probs_hat = problem.model_sampler.probs()
        # log_mk_probs_hat = problem.model_sampler.logits()
        # rkld = kld_logits(log_mk_probs_hat, problem.dgp.modelprobs.log()).item()
        # kld = kld_logits(problem.dgp.modelprobs.log(), log_mk_probs_hat).item()

        # logging.info(f"KLD predicted to target model probabilities = {kld}/{rkld}")

        # print model probs
        logging.info("sorted model probabilities by target")
        if RUN_RJMCMC:
            sortidx = torch.argsort(mk_probs_true, dim=0)
        else:
            sortidx = torch.argsort(mk_probs_hat, dim=0)

        logging.info("id\ttrue weight\tpredicted weight")
        for i in sortidx:
            logging.info(f"{mk_identifiers[i]}\t{mk_probs_true[i]}\t{mk_probs_hat[i]}")

        logging.info(f"terminal loss = {res['loss']}")

        problem.dgp.plot_q_mk_selected(
            problem.param_transform, mk_identifiers, mk_probs_hat, 8192
        )

    diagnostic_trial()
    return (diagnostic_trial,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Parameter sweep""")
    return


@app.cell
def _(TransdimensionalFlow, gp_surrogate_trial, logging, sfe_trial, torch):
    def sweeping_trial(replicate=None, job_id=None, **kwargs):
        """
        Compact run which just returns the important stuff, for ease of analysis.
        """
        output_dir = kwargs.get('output_dir', 'default_directory')  # Provides a default if 'output_dir' is not passed

        log_file = f"{output_dir}/job.log"

        # Configure basic logging to file
        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        # separate arguments for separate trials. TODO clean this up
        sfe_cat_child_kwargs = kwargs.pop("sfe_cat_kwargs", {})
        sfe_made_child_kwargs = kwargs.pop("sfe_made_kwargs", {})
        diag_surrogate_child_kwargs = kwargs.pop("diag_surrogate_kwargs", {})
        ens_surrogate_child_kwargs = kwargs.pop("ens_surrogate_kwargs", {})

        sfe_cat_kwargs = kwargs.copy()
        sfe_cat_kwargs.update(sfe_cat_child_kwargs)

        sfe_made_kwargs = kwargs.copy()
        sfe_made_kwargs.update(sfe_made_child_kwargs)

        diag_surrogate_kwargs = kwargs.copy()
        diag_surrogate_kwargs.update(diag_surrogate_child_kwargs)

        ens_surrogate_kwargs = kwargs.copy()
        ens_surrogate_kwargs.update(ens_surrogate_child_kwargs)

        # run trials
        sfe_cat_res = sfe_trial(**sfe_cat_kwargs)
        # ease of reference
        sfe_cat_problem = sfe_cat_res["problem"]

        # immediately run rjmcmc to test
        # now run rjmcmc
        sfe_cat_problem.dgp.sample_true_mk_probs(
            plot_bivariates=False, store_samples=True
        )

        sfe_made_res = sfe_trial(**sfe_made_kwargs)
        sfe_made_problem = sfe_made_res["problem"]

        # run trials
        diag_surrogate_res = gp_surrogate_trial(**diag_surrogate_kwargs)
        #ens_surrogate_res = gp_surrogate_trial(**ens_surrogate_kwargs)
        diag_surrogate_problem = diag_surrogate_res["problem"]
        #ens_surrogate_problem = ens_surrogate_res["problem"]

        # set up amortized transdimensional variational distributions
        sfe_cat_tdf = TransdimensionalFlow(
            sfe_cat_problem.model_sampler,
            sfe_cat_problem.param_transform,
            sfe_cat_problem.dgp.reference_dist(),
        )
        sfe_made_tdf = TransdimensionalFlow(
            sfe_made_problem.model_sampler,
            sfe_made_problem.param_transform,
            sfe_made_problem.dgp.reference_dist(),
        )
        diag_surrogate_tdf = TransdimensionalFlow(
            diag_surrogate_problem.model_sampler,
            diag_surrogate_problem.param_transform,
            diag_surrogate_problem.dgp.reference_dist(),
        )
        #ens_surrogate_tdf = TransdimensionalFlow(
        #    ens_surrogate_problem.model_sampler,
        #    ens_surrogate_problem.param_transform,
        #    ens_surrogate_problem.dgp.reference_dist(),
        #)
        if False:
            mk_probs_true = (
                sfe_cat_problem.dgp.true_mk_probs_all
            )  # should be returned from previous statement
        elif False:
            mk_identifiers = sfe_cat_problem.dgp.mk_identifiers()
            true_mk_ids, true_mk_probs = sfe_cat_problem.dgp.true_mk_identifiers, sfe_cat_problem.dgp.true_mk_probs
            N = mk_identifiers.size(0)
            # Initialize the result tensor with zeros
            mk_probs_true = torch.zeros(N, device=mk_identifiers.device, dtype=true_mk_probs.dtype)
            mask = (true_mk_ids.unsqueeze(1) == mk_identifiers.unsqueeze(0)).all(dim=-1)
            indices = mask.nonzero(as_tuple=False)
            mk_probs_true[indices[:, 1]] = true_mk_probs[indices[:, 0]]
        # log avg NLL for metrics. Note we ran RJMCMC in the sfe_cat dgp, so this will be used to eval NLL.
        sfe_cat_nll_val = sfe_cat_problem.dgp.compute_average_nll(sfe_cat_tdf)
        sfe_made_nll_val = sfe_cat_problem.dgp.compute_average_nll(sfe_made_tdf)
        diag_surrogate_nll_val = sfe_cat_problem.dgp.compute_average_nll(diag_surrogate_tdf)
        #ens_surrogate_nll_val = sfe_cat_problem.dgp.compute_average_nll(ens_surrogate_tdf)
        return dict(
            sfe_cat_loss=sfe_cat_res["loss"],
            sfe_cat_time=sfe_cat_res["time"],
            sfe_cat_nll=sfe_cat_nll_val,
            sfe_made_loss=sfe_made_res["loss"],
            sfe_made_time=sfe_made_res["time"],
            sfe_made_nll=sfe_made_nll_val,
            diag_surrogate_loss=diag_surrogate_res["loss"],
            diag_surrogate_time=diag_surrogate_res["time"],
            diag_surrogate_nll=diag_surrogate_nll_val,
            #ens_surrogate_loss=ens_surrogate_res["loss"],
            #ens_surrogate_time=ens_surrogate_res["time"],
            #ens_surrogate_nll=ens_surrogate_nll_val,
        )
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
def _(ParamSweepExperiment, bundles, logging, plt, sns):
    async def param_sweep_plot(
        experiment_prefix,
        sweeping_trial_fn,
        sweep_param,
        sweep_vals,
        smoke_test=False,  # Does nothing
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
        output_path_df = f"{pexp.output_dir}/{experiment_name}_{sweep_param}.df"
        results_df.to_pickle(output_path_df)

        logging.info(f"Results\n {results_df}")
        logging.info("plotting")

        # Apply ICML 2024 style
        plt.rcParams.update(bundles.icml2024())

        # melt the data frame on NLL
        df_melted = results_df.melt(
            id_vars=sweep_param,
            #value_vars=["sfe_cat_nll", "sfe_made_nll", "diag_surrogate_nll", "ens_surrogate_nll"],
            value_vars=["sfe_cat_nll", "sfe_made_nll", "diag_surrogate_nll"],
            var_name="ModelSampler",
            value_name="NLL",
        )

        # Create the boxplot using seaborn
        plt.figure(figsize=(4, 3))
        ax = sns.boxplot(
            data=df_melted, x=sweep_param, y="NLL", hue="ModelSampler", palette="Set2"
        )

        # Add labels and title
        ax.set_xlabel(sweep_param)
        ax.set_ylabel("NLL")
        #ax.set_title(f"Distribution of Average NLL vs {sweep_param}")
        ax.set_title(f"Distribution of average NLL vs cardinality")
        ax.set_xticks(
            ticks=plt.xticks()[0], labels=[f"{tick:.1f}" for tick in plt.xticks()[0]]
        )
        # labels = [x.get_text() for x in ax.get_ticklabels()]
        ax.set_xticklabels([str(x) for x in sweep_vals])
        ax.legend(title="ModelSampler")

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

    # TODO change to dgp key sweep
    dgpkey_fig, dgpkey_df = await param_sweep_plot(
        experiment_prefix="vti_robust_vs_highms_compare_modelsamplers_spline46shared_mid_cardinality_tue18mar_longer",
        sweeping_trial_fn=sweeping_trial,
        sweep_param="dgp_key",
        #sweep_vals=["robust_vs_6", "robust_vs_8", "robust_vs_10", "robust_vs_15", "robust_vs_20", "robust_vs_25"],
        #sweep_vals=["robust_vs_midms_10", "robust_vs_midms_15", "robust_vs_midms_20", "robust_vs_midms_25"],
        sweep_vals=["robust_vs_10", "robust_vs_15", "robust_vs_20", "robust_vs_25"],
        base_kwargs=base_kwargs,
        num_replicates=10,
        executor_params={
            "timeout_min": 400,
            "tasks_per_node": 1,
            "mem_gb": 32,
            "cpus_per_task": 1,
            "gpus_per_node": 1,
            "slurm_account": slurm_account,
            "slurm_array_parallelism": 40,  # this throttles our job allocation and is requested by admins for disk-heavy workloads
        },
    )
    dgpkey_fig
    return dgpkey_df, dgpkey_fig


@app.cell
def _(dgpkey_df):
    dgpkey_df
    return


@app.cell(disabled=True)
def _(bundles, dgpkey_df, logging, plt, sns):
    def plot_clipped_fig2(output_dir, results_df):
        logging.info("plotting")

        sweep_param="dgp_key"
        sweep_vals=["robust_vs_midms_10", "robust_vs_midms_15", "robust_vs_midms_20", "robust_vs_midms_25"]

        xlabeldict = {"robust_vs_midms_10":r"$2^9$", 
                      "robust_vs_midms_15":r"$2^{14}$",
                      "robust_vs_midms_20":r"$2^{19}$",
                      "robust_vs_midms_25":r"$2^{24}$"}

        # Apply ICML 2024 style
        plt.rcParams.update(bundles.icml2024())

        # melt the data frame on NLL
        df_melted = results_df.melt(
            id_vars=sweep_param,
            #value_vars=["sfe_cat_nll", "sfe_made_nll", "diag_surrogate_nll", "ens_surrogate_nll"],
            value_vars=["sfe_cat_nll", "sfe_made_nll", "diag_surrogate_nll"],
            var_name="ModelSampler",
            value_name="NLL",
        )

        label_dict = {
            "sfe_cat_nll": "Categorical MCG",
            "sfe_made_nll": "MADE MCG",
            "diag_surrogate_nll": "Diagonal Surrogate",
            }

        # Map the 'ModelSampler' column using the label dictionary
        df_melted['ModelSampler'] = df_melted['ModelSampler'].map(label_dict)


        # Create the boxplot using seaborn
        plt.figure(figsize=(4, 2))
        ax = sns.boxplot(
            data=df_melted, x=sweep_param, y="NLL", hue="ModelSampler", palette="Set2", fliersize=1,
        )

        # Add labels and title
        #ax.set_xlabel(sweep_param)
        ax.set_xlabel(r"$|\mathcal{M}|$")
        #ax.set_ylabel("NLL")
        ax.set_ylabel(r"$H(\pi,q_{\psi,\phi})$")
        #ax.set_ylim(1,100)
        #ax.set_yscale('log')
        ax.set_ylim(0,27)
        ax.set_yscale('linear')
        #ax.set_title(f"Distribution of Average NLL vs {sweep_param}")
        ax.set_title(f"Misspecification level: Medium\nFull cross-entropy vs cardinality")
        ax.set_xticks(
            ticks=plt.xticks()[0], labels=[f"{tick:.1f}" for tick in plt.xticks()[0]]
        )
        # labels = [x.get_text() for x in ax.get_ticklabels()]
        ax.set_xticklabels([str(xlabeldict[x]) for x in sweep_vals])
        ax.legend(title=r"$q_{\psi}(m)$ type", loc="lower right")

        # Save the plot as PNG and PDF
        output_path_png = f"{output_dir}/clipped_fig2_midms_ylin.png"
        output_path_pdf = f"{output_dir}/clipped_fig2_midms_ylin.pdf"
        plt.savefig(output_path_png, format="png")
        plt.savefig(output_path_pdf, format="pdf")

        # Show the plot
        return plt.gcf()

    #plot_clipped_fig2('_experiments/vti_robust_vs_midms_compare_modelsamplers_affine510_mid_cardinality_mon17mar_longer_dgp_key_20250317-095302/outputs/', dgpkey_df)
    plot_clipped_fig2('_experiments/vti_robust_vs_midms_compare_modelsamplers_spline46shared_mid_cardinality_mon17mareve_longer_dgp_key_20250317-183231/outputs/', dgpkey_df)
    return (plot_clipped_fig2,)


@app.cell
def _(bundles, dgpkey_df, logging, plt, sns):
    def plot_clipped_fig_highms(output_dir, results_df):
        logging.info("plotting")

        sweep_param="dgp_key"
        sweep_vals=["robust_vs_midms_10", "robust_vs_midms_15", "robust_vs_midms_20", "robust_vs_midms_25"]

        xlabeldict = {"robust_vs_midms_10":r"$2^9$", 
                      "robust_vs_midms_15":r"$2^{14}$",
                      "robust_vs_midms_20":r"$2^{19}$",
                      "robust_vs_midms_25":r"$2^{24}$"}

        # Apply ICML 2024 style
        plt.rcParams.update(bundles.icml2024())

        # melt the data frame on NLL
        df_melted = results_df.melt(
            id_vars=sweep_param,
            #value_vars=["sfe_cat_nll", "sfe_made_nll", "diag_surrogate_nll", "ens_surrogate_nll"],
            value_vars=["sfe_cat_nll", "sfe_made_nll", "diag_surrogate_nll"],
            var_name="ModelSampler",
            value_name="NLL",
        )

        label_dict = {
            "sfe_cat_nll": "Categorical MCG",
            "sfe_made_nll": "MADE MCG",
            "diag_surrogate_nll": "Diagonal Surrogate",
            }

        # Map the 'ModelSampler' column using the label dictionary
        df_melted['ModelSampler'] = df_melted['ModelSampler'].map(label_dict)


        # Create the boxplot using seaborn
        plt.figure(figsize=(4, 2))
        ax = sns.boxplot(
            data=df_melted, x=sweep_param, y="NLL", hue="ModelSampler", palette="Set2", fliersize=1,
        )

        # Add labels and title
        #ax.set_xlabel(sweep_param)
        ax.set_xlabel(r"$|\mathcal{M}|$")
        #ax.set_ylabel("NLL")
        ax.set_ylabel(r"$H(\pi,q_{\psi,\phi})$")
        #ax.set_ylim(1,100)
        #ax.set_yscale('log')
        ax.set_ylim(0,35)
        ax.set_yscale('linear')
        #ax.set_title(f"Distribution of Average NLL vs {sweep_param}")
        ax.set_title(f"Misspecification level: High\nFull cross-entropy vs cardinality")
        ax.set_xticks(
            ticks=plt.xticks()[0], labels=[f"{tick:.1f}" for tick in plt.xticks()[0]]
        )
        # labels = [x.get_text() for x in ax.get_ticklabels()]
        ax.set_xticklabels([str(xlabeldict[x]) for x in sweep_vals])
        ax.legend(title=r"$q_{\psi}(m)$ type", loc="lower right")

        # Save the plot as PNG and PDF
        output_path_png = f"{output_dir}/clipped_fig2_highms_ylin.png"
        output_path_pdf = f"{output_dir}/clipped_fig2_highms_ylin.pdf"
        plt.savefig(output_path_png, format="png")
        plt.savefig(output_path_pdf, format="pdf")

        # Show the plot
        return plt.gcf()

    #plot_clipped_fig2('_experiments/vti_robust_vs_midms_compare_modelsamplers_affine510_mid_cardinality_mon17mar_longer_dgp_key_20250317-095302/outputs/', dgpkey_df)
    plot_clipped_fig_highms('_experiments/vti_robust_vs_highms_compare_modelsamplers_spline46shared_mid_cardinality_tue18mar_longer_dgp_key_20250318-124935/outputs/', dgpkey_df)
    return (plot_clipped_fig_highms,)


if __name__ == "__main__":
    app.run()
