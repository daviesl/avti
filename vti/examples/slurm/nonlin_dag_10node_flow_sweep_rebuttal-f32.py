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

__generated_with = "0.11.26"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""## Basic imports""")
    return


@app.cell
def _():
    from pathlib import Path
    from time import time
    import os
    os.environ["TQDM_DISABLE"]="1"
    os.environ["ENABLE_TORCH_COMPILE"]="False"
    # Setting the environment variable for SBATCH_EXCLUDE
    os.environ['SBATCH_EXCLUDE'] = 'g088'
    import logging
    # import vti.utils.logging as logging

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
    from vti.infer import (
        VTIMCGEstimator,
    )
    from vti.utils.torch_nn_helpers import ensure_dtype, ensure_device
    from vti.model_samplers import SFECategorical, SFEMADEBinaryString, SFEMADEDAG
    from vti.utils.linalg_lowrank import reduced_mean_dev
    from vti.utils.callbacks import CheckpointCallback
    from vti.utils.experiment import (
        list_experiments,
        get_latest_experiment,
        ParamSweepGenerator,
        AxParameterGenerator,
        ParamSweepExperiment,
        AxExperiment,
    )
    from vti.flows.transdimensional import TransdimensionalFlow
    import cloudpickle as pickle

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    slurm_account = os.getenv("SLURM_ACCOUNT", False)
    return (
        AxExperiment,
        AxParameterGenerator,
        CheckpointCallback,
        ParamSweepExperiment,
        ParamSweepGenerator,
        Path,
        SFECategorical,
        SFEMADEBinaryString,
        SFEMADEDAG,
        TransdimensionalFlow,
        VTIMCGEstimator,
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
        pickle,
        plt,
        reduced_mean_dev,
        set_seed,
        slurm_account,
        sns,
        time,
        torch,
    )


@app.cell
def _():
    base_kwargs = dict(
        num_iterations=50000,
        # num_iterations=10, # for SMOKE test
        batch_size=1024,
        seed=3, #optimiser seed
        ig_threshold=5e-3,
        #dgp_key="nonlineardagmlp_10",
        dgp_key="nonlineardagmlpnobias_10_ndata_1024",
        #dgp_key="nonlineardagmlp_5",
        dgp_seed=1101, # data-generating process seed
        #dgp_seed="dgpseedfn1000",
        #flow_type="spline46",
        #flow_type="spline46",
        #flow_type="affine510f128",
        flow_type="affine510f128",
        # model sampler parameters
        model_sampler_key="SFEMADEDAG",
        #device="cuda" if torch.cuda.is_available() else "cpu",
        grad_norm_clip=50.0,
        device="cuda",
        dtype="float32",
        plot=False,
    )
    return (base_kwargs,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The main trial functions

        We set upon `sfe_trial`, which creates a problem, fits a neural distribution to it, and returns various performance metrics.
        .
        """
    )
    return


@app.cell
def _():
    from vti.examples.sfe_trial import sfe_trial
    return (sfe_trial,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Run a single trial

        This approximates the CLI script that ran single trials before.
        """
    )
    return


@app.cell(disabled=True)
def _(TransdimensionalFlow, base_kwargs, logging, pickle, sfe_trial):
    # def diagnostic_trial(**kwargs):
    #     """
    #     Verbose, messy  run, for ease of diagnosis.
    #     """
    #     res = sfe_trial(**kwargs, **base_kwargs)
    #     problem = res["problem"]

    #     # set up transd flow and log average NLL
    #     tdf = TransdimensionalFlow(
    #         problem.model_sampler,
    #         problem.param_transform,
    #         problem.dgp.reference_dist(),
    #     )

    #     # get F1 and SHD and Brier
    #     vti_F1, vti_SHD, vti_brier = problem.dgp.compute_metrics(tdf, num_samples=1024)
    #     logging.info(f"VTI average F1 score {vti_F1}")
    #     logging.info(f"VTI average SHD score {vti_SHD}")
    #     logging.info(f"Brier score {vti_brier}")

    #     if False:
    #         # plot
    #         problem.dgp.plot_q_tdf(
    #             tdf, num_samples=2048, title="Posterior Flow Cytometry DAGs", font_size="6"
    #         )

    #     dagma_summary = problem.dgp.cdt_dagma_metrics()
    #     logging.info(f"DAGMA summary {dagma_summary}")

    #     if False:
    #         dagma_F1, dagma_SHD, dagma_brier = problem.dgp.cdt_dagma_f1_shd()
    #         logging.info(f"DAGMA F1 score {dagma_F1}")
    #         logging.info(f"DAGMA SHD score {dagma_SHD}")
    #         logging.info(f"DAGMA Brier score {dagma_brier}")

    # diagnostic_trial()

    def diagnostic_trial(**kwargs):
        """
        Verbose, messy  run, for ease of diagnosis.
        """
        res = sfe_trial(**kwargs, **base_kwargs)
        problem = res["problem"]

        # set up transd flow and log average NLL
        tdf = TransdimensionalFlow(
            problem.model_sampler,
            problem.param_transform,
            problem.dgp.reference_dist(),
        )

        for i in range(1):
            vti_F1, vti_SHD, vti_brier, vti_auroc = problem.dgp.compute_metrics(tdf, num_samples=1024)
            logging.info(f"VTI average F1 score {vti_F1}")
            logging.info(f"VTI average SHD score {vti_SHD}")
            logging.info(f"Brier score {vti_brier}")
            logging.info(f"AUROC score {vti_auroc}")

        if False:
            problem.dgp.plot_q_tdf(
                tdf,
                title="Posterior Flow Cytometry DAGs",
                font_size="5",
                num_samples=4096,
                figsize=(20,20),
                #saveto="~/SachsPosterior.pdf"
            )
        output_dir = kwargs.get('output_dir', '/scratch3/dav718/projects/virga/vti/temp/')  # Provides a default if 'output_dir' is not passed
        try:
            with open(f"{output_dir}/mon17mar_mlp5_p0_ig5em3_tdf.pkl", 'wb') as f:
                pickle.dump(tdf, f)
        except Exception as e:
            error_message = str(e)
            logging.info(f"Could not pickle TDF to {output_dir}: {error_message}")

        if False: # run DAGMA
            if True:
                dagma_summary = problem.dgp.cdt_dagma_metrics(sweeplen=10,nonlinear=False)
                logging.info(f"DAGMA summary {dagma_summary}")
            if True:
                dagma_summary = problem.dgp.cdt_dagma_metrics(sweeplen=10,nonlinear=True)
                logging.info(f"DAGMA non-linear summary {dagma_summary}")

    # diagnostic_trial()
    return (diagnostic_trial,)


@app.cell
def _(mo):
    mo.md(r"""## Parameter sweep""")
    return


@app.cell
def _(TransdimensionalFlow, logging, sfe_trial, torch):
    def sweeping_trial(replicate=None, job_id=None, **kwargs):
        """
        Compact run which just returns the important stuff, for ease of analysis.
        """
        output_dir = kwargs.get('output_dir', 'default_directory')  # Provides a default if 'output_dir' is not passed

        log_file = f"{output_dir}/job.log"

        # Configure basic logging to file
        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        res = sfe_trial(**kwargs)
        problem = res["problem"]
        tdf = TransdimensionalFlow(
            problem.model_sampler,
            problem.param_transform,
            problem.dgp.reference_dist(),
        )
        # get F1 and SHD and Brier
        vti_F1, vti_SHD, vti_Brier, vti_auroc = problem.dgp.compute_metrics(tdf, num_samples=2048)
        logging.info(f"VTI average F1 score {vti_F1}")
        logging.info(f"VTI average SHD score {vti_SHD}")
        logging.info(f"Brier score {vti_Brier}")
        logging.info(f"AUROC score {vti_auroc}")

        try:
             with open(f"{output_dir}/scores.txt", 'w') as f:
                 f.write(f"dgp_seed\t{kwargs['dgp_seed']}\n")
                 f.write(f"seed\t{kwargs['seed']}\n")
                 f.write(f"dgp_key\t{kwargs['dgp_key']}\n")
                 f.write(f"loss\t{res['loss']:.5f}\n")
                 f.write(f"time\t{res['time']:.5f}\n")
                 f.write(f"F1\t{vti_F1.detach().cpu().item():.5f}\n")
                 f.write(f"SHD\t{vti_SHD.detach().cpu().item():.5f}\n")
                 f.write(f"Brier\t{vti_Brier.detach().cpu().item():.5f}\n")
                 f.write(f"AUROC\t{vti_auroc:.5f}\n")
        except Exception as e:
            logging.info(f"Could not write scores to {output_dir}/scores.txt : {str(e)}")


        # try:
        #     with open(f"{output_dir}/tdf.pkl", 'wb') as f:
        #         pickle.dump(tdf, f)
        # except Exception as e:
        #     error_message = str(e)
        #     logging.info(f"Could not pickle TDF to {output_dir}: {error_message}")

        # save problem to disk for other metrics
        try:
            true_A = problem.dgp.true_adjacency_matrix().detach().cpu()
            torch.save(true_A,f"{output_dir}/true_A.pt")
        except Exception as e:
            error_message = str(e)
            logging.info(f"Could not save true_A to {output_dir}: {error_message}")

        try:
            x_data = problem.dgp.get_x_data().detach().cpu()
            torch.save(x_data,f"{output_dir}/x_data.pt")
        except Exception as e:
            error_message = str(e)
            logging.info(f"Could not save x_data to {output_dir}: {error_message}")


        if False:
            # breaks on gpu for some reason
            dagma_summary = problem.dgp.cdt_dagma_metrics(sweeplen=3)
            #logging.info(f"DAGMA summary {dagma_summary}")
            dagma_F1 = dagma_summary["f1"]
            dagma_SHD = dagma_summary["shd"]
            dagma_brier = dagma_summary["brier"]
            dagma_lambda = dagma_summary["lambda"]
            logging.info(f"DAGMA F1 score {dagma_F1}")
            logging.info(f"DAGMA SHD score {dagma_SHD}")
            logging.info(f"DAGMA Brier score {dagma_brier}")

        else:
            dagma_F1 = 0.
            dagma_SHD = 0.
            dagma_brier = 0.
            dagma_lambda = 0.

        return dict(loss=res["loss"],
                    F1=vti_F1.detach().cpu().item(),
                    SHD=vti_SHD.detach().cpu().item(),
                    Brier=vti_Brier.detach().cpu().item(),
                    AUROC=vti_auroc,
                    #dagma_F1=dagma_F1,
                    #dagma_SHD=dagma_SHD,
                    #dagma_brier=dagma_brier,
                    #dagma_lambda=dagma_lambda,
                    time=res["time"])
    return (sweeping_trial,)


@app.cell
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
        # set paths
        output_path_df = f"{pexp.output_dir}/{experiment_name}_{sweep_param}.df"
        # save immediately
        results_df.to_pickle(output_path_df)
        # try plotting
        logging.info("plotting")

        # Apply ICML 2024 style
        plt.rcParams.update(bundles.icml2024())

        for metric in ['F1','SHD','Brier', 'AUROC', 'loss']:

            # Create the boxplot using seaborn
            plt.figure(figsize=(8, 5))
            ax = sns.boxplot(data=results_df, x=sweep_param, y=metric)
            # Add labels and title
            ax.set_xlabel(sweep_param)
            ax.set_ylabel(metric)
            ax.set_title(f"Synthetic nonlinear MLP DAG Distribution of {metric} vs {sweep_param}")
            ax.set_xticks(
                ticks=plt.xticks()[0], labels=[f"{tick:.1f}" for tick in plt.xticks()[0]]
            )
            # labels = [x.get_text() for x in ax.get_ticklabels()]
            ax.set_xticklabels([str(x) for x in sweep_vals],rotation=90)

            # Save the plot as PNG and PDF
            output_path_png = f"{pexp.output_dir}/{experiment_name}_{sweep_param}_{metric}.png"
            output_path_pdf = f"{pexp.output_dir}/{experiment_name}_{sweep_param}_{metric}.pdf"
            plt.savefig(output_path_png, format="png")
            plt.savefig(output_path_pdf, format="pdf")
            plt.show()

        # Show the plot
        #return plt.gcf(), results_df
        return results_df
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



    dgp_key_df = await param_sweep_plot(
        experiment_prefix="vti10dagmlpnobias_pedge05_sweep_ndata_Tue25March_zllMADE_fixprior_50kiter_ig5em3_affine510f128_cenc_dgpseed1101",
        sweeping_trial_fn=sweeping_trial,
        #sweep_param="flow_type",
        #sweep_vals=['affine55f512','affine510f512','affine515f512'],
        sweep_param="dgp_key",
        sweep_vals=[f'nonlineardagmlpnobias_10_ndata_{int(2**i)}' for i in [4,5,6,7,8,9,10]],
        #sweep_vals=['affine55f256','affine510f256','affine515f256'],
        #sweep_vals=['spline410','spline415','affine55','affine510','affine515'],
        #sweep_vals=['spline46','spline410','spline415','spline420','affine55','affine510','affine515','affine520'],
        #sweep_vals=['spline23','spline46','affine55'],
        #sweep_vals=['affine510','affine515','affine520'],
        base_kwargs=base_kwargs,
        num_replicates=10,
        # num_replicates=2,   # for SMOKE test
        executor_params={
            # "timeout_min": 5, # for SMOKE test
            "timeout_min": 119,
            "tasks_per_node": 1,
            "mem_gb": 6,
            "cpus_per_task": 1,
            "gpus_per_node": 1,
            "slurm_account": slurm_account,
            "slurm_array_parallelism": 70,  # this throttles our job allocation and is requested by admins for disk-heavy workloads
            #"additional_parameters":{"exclude": "g088"},
        },
    )
    return (dgp_key_df,)


@app.cell(disabled=True)
def _(dgp_key_df):
    dgp_key_df['F1'].idxmax()
    return


@app.cell(disabled=True)
def _(dgp_key_df):
    dgp_key_df.iloc[27]
    return


@app.cell(disabled=True)
def _(dgp_key_df):
    timemaxidx = dgp_key_df['time'].idxmax()
    dgp_key_df['time'].iloc[timemaxidx]
    return (timemaxidx,)


@app.cell(disabled=True)
def _(dgp_key_df, plt):
    hiddim=512
    a510_df=dgp_key_df[dgp_key_df['flow_type']==f'affine510f{hiddim}']
    a55_df=dgp_key_df[dgp_key_df['flow_type']==f'affine55f{hiddim}']
    a515_df=dgp_key_df[dgp_key_df['flow_type']==f'affine515f{hiddim}']
    #j=a55_df['SHD'].idxmax()
    #dgp_key_df.iloc[j]
    #a55_df.shape
    plt.figure(figsize=(8, 5))
    plt.scatter(a510_df['loss'],a510_df['F1'], label=f"Affine MAF T=5, H=10, width={hiddim}")
    plt.scatter(a515_df['loss'],a515_df['F1'], label=f"Affine MAF T=5, H=15, width={hiddim}")
    plt.scatter(a55_df['loss'],a55_df['F1'], label=f"Affine MAF T=5, H=5, width={hiddim}")
    plt.xlabel('loss')
    plt.ylabel('F1')
    plt.legend()
    plt.title('Comparison loss vs F1')
    plt.xlim([None,15200])
    plt.show()

    # SHD
    plt.figure(figsize=(8, 5))
    plt.scatter(a510_df['loss'],a510_df['SHD'], label=f"Affine MAF T=5, H=10, width={hiddim}")
    plt.scatter(a515_df['loss'],a515_df['SHD'], label=f"Affine MAF T=5, H=15, width={hiddim}")
    plt.scatter(a55_df['loss'],a55_df['SHD'], label=f"Affine MAF T=5, H=5, width={hiddim}")
    plt.xlabel('loss')
    plt.ylabel('SHD')
    plt.legend()
    plt.title('Comparison loss vs SHD')
    plt.xlim([None,15200])
    plt.show()

    # AUROC
    plt.figure(figsize=(8, 5))
    plt.scatter(a510_df['loss'],a510_df['AUROC'], label=f"Affine MAF T=5, H=10, width={hiddim}")
    plt.scatter(a515_df['loss'],a515_df['AUROC'], label=f"Affine MAF T=5, H=15, width={hiddim}")
    plt.scatter(a55_df['loss'],a55_df['AUROC'], label=f"Affine MAF T=5, H=5, width={hiddim}")
    plt.xlabel('loss')
    plt.ylabel('AUROC')
    plt.legend()
    plt.title('Comparison loss vs AUROC')
    plt.xlim([None,15200])
    plt.show()

    # Brier
    plt.figure(figsize=(8, 5))
    plt.scatter(a510_df['loss'],a510_df['Brier'], label=f"Affine MAF T=5, H=10, width={hiddim}")
    plt.scatter(a515_df['loss'],a515_df['Brier'], label=f"Affine MAF T=5, H=15, width={hiddim}")
    plt.scatter(a55_df['loss'],a55_df['Brier'], label=f"Affine MAF T=5, H=5, width={hiddim}")
    plt.xlabel('loss')
    plt.ylabel('Brier')
    plt.legend()
    plt.title('Comparison loss vs Brier')
    plt.xlim([None,15200])
    plt.show()
    return a510_df, a515_df, a55_df, hiddim


@app.cell(disabled=True)
def _(bundles, dgp_key_df, plt, sns):
    results_df = dgp_key_df
    if True:
        sweep_param="flow_type"
        sweep_vals=['spline410','spline415','affine55','affine510','affine515']
        # Apply ICML 2024 style
        plt.rcParams.update(bundles.icml2024())

        for metric in ['F1','SHD','Brier', 'AUROC', 'loss']:

            # Create the boxplot using seaborn
            plt.figure(figsize=(8, 5))
            ax = sns.boxplot(data=results_df, x=sweep_param, y=metric)
            # Add labels and title
            ax.set_xlabel(sweep_param)
            ax.set_ylabel(metric)
            ax.set_title(f"Synthetic MLP DAG 10 node Distribution of {metric} vs {sweep_param}")
            ax.set_xticks(
                ticks=plt.xticks()[0], labels=[f"{tick:.1f}" for tick in plt.xticks()[0]]
            )
            # labels = [x.get_text() for x in ax.get_ticklabels()]
            ax.set_xticklabels([str(x) for x in sweep_vals],rotation=90)

            # Save the plot as PNG and PDF
            #output_path_png = f"{pexp.output_dir}/{experiment_name}_{sweep_param}_{metric}.png"
            #output_path_pdf = f"{pexp.output_dir}/{experiment_name}_{sweep_param}_{metric}.pdf"
            #plt.savefig(output_path_png, format="png")
            #plt.savefig(output_path_pdf, format="pdf")
            plt.show()
    return ax, metric, results_df, sweep_param, sweep_vals


@app.cell(disabled=True)
def _(dgp_key_df, pd):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    dgp_key_df.to_string()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
