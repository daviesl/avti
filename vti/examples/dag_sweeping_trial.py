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

# vti/examples/dag_sweeping_trial.py

def dag_sweeping_trial(replicate=None, job_id=None, **kwargs):
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
    tdf = TransdimensionalFlow(
        problem.model_sampler,
        problem.param_transform,
        problem.dgp.reference_dist(),
    )
    # Compute and log metrics.
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

    if False:
        # Save objects.
        # Fails locally, can't save model sampler for some reason.
        try:
            with open(f"{output_dir}/tdf_{replicate}_{job_id}.pkl", 'wb') as f:
                pickle.dump(tdf, f)
        except Exception as e:
            logging.info(f"Could not pickle TDF to {output_dir}: {str(e)}")

    try:
        true_A = problem.dgp.true_adjacency_matrix().detach().cpu()
        torch.save(true_A, f"{output_dir}/true_A_{replicate}_{job_id}.pt")
    except Exception as e:
        logging.info(f"Could not save true_A to {output_dir}: {str(e)}")

    try:
        x_data = problem.dgp.get_x_data().detach().cpu()
        torch.save(x_data, f"{output_dir}/x_data_{replicate}_{job_id}.pt")
    except Exception as e:
        logging.info(f"Could not save x_data to {output_dir}: {str(e)}")

    float_vti_F1 = vti_F1.detach().cpu().item()
    float_vti_SHD = vti_SHD.detach().cpu().item()
    float_vti_Brier = vti_Brier.detach().cpu().item()
    foat_vti_auroc = vti_auroc
    float_loss = res["loss"]
    float_time = res["time"]

    for handler in logging.getLogger().handlers:
        handler.flush()
    logging.handlers.clear()

    # Free any GPU memory that was allocated during the trial.
    del tdf, problem, res, vti_F1, vti_SHD, vti_Brier, vti_auroc
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return dict(
        loss=float_loss,
        F1=float_vti_F1,
        SHD=float_vti_SHD,
        Brier=float_vti_Brier,
        AUROC=float_vti_auroc,
        time=float_time,
        replicate=replicate,
        job_id=job_id,
        sweep_value=sweep_value,
    )

