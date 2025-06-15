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

import bz2
import torch
import cloudpickle
import submitit
import asyncio
import time
import json
import pandas as pd
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar
import logging
from pprint import pformat
import numpy as np

from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.exceptions.generation_strategy import MaxParallelismReachedException
from ax.exceptions.core import DataRequiredError
from ax.utils.measurement.synthetic_functions import hartmann6

# ===================================
# Exceptions & Interfaces
# ===================================

T = TypeVar("T", bound="BaseExperiment")


def make_hashable(obj):
    if isinstance(obj, dict):
        # Convert dict to frozenset of key-value pairs
        return frozenset((k, make_hashable(v)) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set)):
        # Convert list, tuple, or set to a frozenset of their items
        # This ignores order and duplicate items
        return frozenset(make_hashable(i) for i in obj)
    else:
        return obj


class ParameterStalledException(Exception):
    """Raised when parameter generation is stalled due to parallelism or data requirements."""


class ParameterGenerator:
    def __iter__(self):
        raise NotImplementedError


class ParamSweepGenerator(ParameterGenerator):
    """
    Generator for parameter sweeps, yielding parameter combinations.
    """

    def __init__(
        self,
        base_params: Dict[str, Any],
        sweep_params: Dict[str, List[Any]],
        num_replicates: int,
        smoke_test: bool,
    ):
        self.base_params = base_params
        self.sweep_params = sweep_params
        self.num_replicates = num_replicates
        self.smoke_test = smoke_test
        self._generate_combinations()

    def _generate_combinations(self):
        from itertools import product

        keys = list(self.sweep_params.keys())
        values = list(self.sweep_params.values())
        self.combinations = list(product(*values))
        if self.smoke_test:
            self.combinations = self.combinations[:1]
        self.total = len(self.combinations) * self.num_replicates
        self.current = 0

    def __iter__(self):
        for combination in self.combinations:
            for replicate in range(1, self.num_replicates + 1):
                params = self.base_params.copy()
                params.update(
                    {k: v for k, v in zip(self.sweep_params.keys(), combination)}
                )
                params["replicate"] = replicate
                params["job_id"] = self.current
                params["seed"] = (
                    replicate  # self.current  # Optional: for reproducibility
                )
                self.current += 1
                yield params


class AxParameterGenerator(ParameterGenerator):
    """
    Parameter generator tailored for Ax experiments.
    """

    def __init__(self, ax_client: AxClient, trial_budget: int):
        self.ax_client = ax_client
        self.trial_budget = trial_budget
        self.count = 0

    def __iter__(self):
        while self.count < self.trial_budget:
            try:
                parameters, trial_index = self.ax_client.get_next_trial()
                parameters["trial_index"] = trial_index
                parameters["job_id"] = trial_index  # Use trial_index as job_id
                parameters["seed"] = trial_index  # Optional: for reproducibility
                yield parameters
                self.count += 1
            except (MaxParallelismReachedException, DataRequiredError):
                raise ParameterStalledException(
                    "No new trials can be generated at this moment."
                )


# ===================================
# Helpers
# ===================================
# def run_async_coroutine(coro):
#     try:
#         loop = asyncio.get_running_loop()
#     except RuntimeError:
#         loop = None
#     if loop and loop.is_running():
#         # Schedule the coroutine and return the task
#         return asyncio.create_task(coro)
#     else:
#         # Run the coroutine and block until complete
#         return asyncio.run(coro)


def clean_ax_results_objectives(results, objective_names):
    """
    Clean up the results object from Ax to ensure that all objectives are scalars.
    """
    for result in results:
        for objective_name in objective_names:
            objective = result.get(objective_name, None)
            if isinstance(objective, tuple):
                result[objective_name] = objective[0]
    return results


# ===================================
# BaseExperiment
# ===================================


class BaseExperiment:
    def __init__(
        self,
        function: Callable[..., Any],
        experiment_name: str,
        executor_params: Dict[str, Any],
        experiment_dir: Optional[str] = None,  # Added experiment_dir parameter
        subdir: Optional[str] = None,
        parent_dir: str = "_experiments",
        output_dir: str = "outputs",
        smoke_test: bool = False,
        debug: bool = False,  # use debug executor for better stack traces
        local: bool = False,  # force local executor for development on cluster
        base_params: Optional[Dict[str, Any]] = None,
    ):
        self.function = function
        self.smoke_test = smoke_test
        self.debug = debug
        self.local = local
        if self.smoke_test:
            experiment_name = f"__smoke__{experiment_name}"
        self.experiment_name = experiment_name
        # executor_params must not be empty
        assert executor_params, "executor_params must not be empty."
        self.executor_params = executor_params
        self.base_params = base_params or {}

        if experiment_dir:
            self.experiment_dir = Path(experiment_dir)
            if not self.experiment_dir.exists():
                raise FileNotFoundError(
                    f"Experiment directory {experiment_dir} does not exist."
                )
        elif subdir:
            self.experiment_dir = Path(parent_dir) / subdir
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.experiment_dir = self._create_unique_experiment_dir(parent_dir)

        if self.smoke_test:
            logging.warning("Running in smoke test mode.")

        self.output_dir = self.experiment_dir / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = self.experiment_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        executor_class = submitit.AutoExecutor
        if self.local:
            executor_class = submitit.LocalExecutor
        if self.debug:
            executor_class = submitit.DebugExecutor

        self.executor = executor_class(folder=str(self.log_dir))
        self.executor.update_parameters(
            name=self.experiment_dir.name, **self.executor_params
        )

        self.metadata_path = self.experiment_dir / "metadata.json"
        self.function_path = self.experiment_dir / "function.pkl.bz2"
        self.job_info: List[Dict[str, Any]] = []
        self._save_metadata()

        if function is not None and not self.function_path.exists():
            self._save_function()

    @classmethod
    def get_or_create(
        cls,
        experiment_name: str,
        parent_dir: str = "_experiments",
        # subdir: Optional[str] = None,  # Optional subdir parameter
        clobber: bool = False,
        **kwargs,
    ) -> "BaseExperiment":
        """
        Attempt to resume an existing experiment with the given name.
        If no existing experiment is found, create a new one.

        Parameters:
            experiment_name (str): The name of the experiment.
            parent_dir (str): The parent directory where experiments are stored.
            subdir (Optional[str]): Subdirectory for the experiment.
            **kwargs: Additional arguments required for creating a new experiment.

        Returns:
            BaseExperiment: An instance of the experiment (resumed or new).
        """
        if clobber:
            logging.info(f"Creating new experiment found for '{experiment_name}'.")

            return cls.create_new(
                experiment_name=experiment_name, parent_dir=parent_dir, **kwargs
            )
        else:
            exp_dir = get_latest_experiment(parent_dir, experiment_name)

            if exp_dir:
                try:
                    experiment = cls.resume(str(exp_dir))
                    logging.info(f"Resumed experiment from {exp_dir}")
                    return experiment
                except Exception as e:
                    logging.warning(f"Failed to resume experiment from {exp_dir}: {e}")
                    logging.info("Creating a new experiment instead.")

            logging.info(
                f"No existing experiment found for '{experiment_name}'. Creating a new one."
            )
            return cls.create_new(
                experiment_name=experiment_name, parent_dir=parent_dir, **kwargs
            )

    @classmethod
    def create_new(cls, **kwargs) -> "BaseExperiment":
        """
        Create a new instance of the experiment.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement create_new method.")

    @classmethod
    def resume(cls, experiment_dir: str) -> "BaseExperiment":
        """
        Resume an existing experiment from the given directory.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement resume method.")

    async def run_async(self, parameter_generator: ParameterGenerator):
        raise NotImplementedError("Subclasses should implement run_async method.")

    def _save_function(self):
        with bz2.open(self.function_path, "wb") as f:
            cloudpickle.dump(self.function, f)
        logging.info(f"Function saved to {self.function_path}")

    def _load_function(self):
        with bz2.open(self.function_path, "rb") as f:
            self.function = cloudpickle.load(f)
        logging.info(f"Function loaded from {self.function_path}")

    def _create_unique_experiment_dir(self, parent_dir: str) -> Path:
        parent = Path(parent_dir)
        parent.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        unique_dir = parent / f"{self.experiment_name}_{timestamp}"
        unique_dir.mkdir(parents=True, exist_ok=False)
        return unique_dir

    def _save_metadata(self):
        metadata = {
            "experiment_name": self.experiment_name,
            "smoke_test": self.smoke_test,
            "debug": self.debug,
            "local": self.local,
            "executor_params": self.executor_params,
        }
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    def load_metadata(self) -> Dict[str, Any]:
        with open(self.metadata_path, "r") as f:
            metadata = json.load(f)
        return metadata

    def save_job_info(self, filename: str = "jobinfo.pkl.bz2"):
        path = self.log_dir / filename
        with bz2.open(path, "wb") as f:
            cloudpickle.dump(self.job_info, f)
        logging.info(f"Job info saved to {path}")

    def load_job_info(self, filename: str = "jobinfo.pkl.bz2"):
        path = self.log_dir / filename
        if not path.exists():
            logging.warning("No job info found.")
            return
        with bz2.open(path, "rb") as f:
            self.job_info = cloudpickle.load(f)
        logging.info(f"Job info loaded from {path}")

    def run_sync(self, parameter_generator: ParameterGenerator):
        """
        Synchronously run the experiment by executing the asynchronous run_async method.
        This method blocks until all jobs are submitted and (optionally) completed.
        """
        asyncio.run(self.run_async(parameter_generator))

    async def collect_results_async(self) -> pd.DataFrame:
        raise NotImplementedError(
            "Subclasses should implement collect_results_async method."
        )

    def collect_results_sync(self, wait_all: bool = True) -> pd.DataFrame:
        """
        Synchronously collect results by executing the asynchronous collect_results_async method.

        Parameters:
            wait_all (bool):
                - If True, wait for all jobs to complete before collecting results.
                - If False, collect results from only the jobs that have completed so far.

        Returns:
            pd.DataFrame: Aggregated results from the jobs.
        """
        return asyncio.run(self.collect_results_async(wait_all=wait_all))

    def terminate(self, delete_files: bool = False):
        if not self.job_info:
            self.load_job_info()

        for info in self.job_info:
            job = info["job"]
            if job.state in ("RUNNING", "PENDING"):
                try:
                    job.cancel()
                    logging.info(f"Cancelled job {job}")
                except Exception as e:
                    logging.warning(f"Failed to cancel job {job}: {e}")

        if delete_files:
            if self.experiment_dir.exists():
                import shutil

                shutil.rmtree(self.experiment_dir)
                logging.info(f"Deleted experiment directory: {self.experiment_dir}")


# ===================================
# ParamSweepExperiment
# ===================================


class ParamSweepExperiment(BaseExperiment):
    """
    Sweep over pre-defined parameters on a pre-defined grid.
    """

    def __init__(
        self,
        function: Callable[..., Any],
        base_params: Dict[str, Any],
        sweep_params: Dict[str, List[Any]],
        num_replicates: int,
        experiment_name: str,
        executor_params: Dict[str, Any],
        experiment_dir: Optional[str] = None,  # Added experiment_dir parameter
        subdir: Optional[str] = None,
        parent_dir: str = "_experiments",
        output_dir: str = "outputs",
        smoke_test: bool = False,
        debug: bool = False,
        local: bool = False,
    ):
        super().__init__(
            function=function,
            experiment_name=experiment_name,
            executor_params=executor_params,
            experiment_dir=experiment_dir,  # Pass experiment_dir to BaseExperiment
            subdir=subdir,
            parent_dir=parent_dir,
            output_dir=output_dir,
            smoke_test=smoke_test,
            debug=debug,
            local=local,
            base_params=base_params,
        )
        # Ensure sweep_params are lists for serialization
        for key in sweep_params:
            sweep_params[key] = list(sweep_params[key])
        self.sweep_params = sweep_params
        self.num_replicates = num_replicates
        self._save_sweep_metadata()

    def _save_sweep_metadata(self):
        """
        Save sweep-specific metadata to a JSON file for persistence and resuming experiments.
        """
        metadata = {
            "base_params": self.base_params,
            "sweep_params": self.sweep_params,
            "num_replicates": self.num_replicates,
        }
        with open(self.experiment_dir / "sweep_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

    def create_parameter_generator(self) -> ParameterGenerator:
        """
        Create a parameter generator based on the sweep parameters and replicates.
        """
        return ParamSweepGenerator(
            self.base_params, self.sweep_params, self.num_replicates, self.smoke_test
        )

    @classmethod
    def create_new(
        cls: Type[T],
        function: Callable[..., Any],
        base_params: Dict[str, Any],
        sweep_params: Dict[str, List[Any]],
        num_replicates: int,
        experiment_name: str,
        executor_params: Dict[str, Any],
        output_dir: str = "outputs",
        smoke_test: bool = False,
        debug: bool = False,
        local: bool = False,
        parent_dir: str = "_experiments",
        subdir: Optional[str] = None,
    ) -> T:
        """
        Create a new ParamSweepExperiment instance.
        """
        return cls(
            function=function,
            base_params=base_params,
            sweep_params=sweep_params,
            num_replicates=num_replicates,
            experiment_name=experiment_name,
            executor_params=executor_params,
            subdir=subdir,
            parent_dir=parent_dir,
            output_dir=output_dir,
            smoke_test=smoke_test,
            debug=debug,
            local=local,
        )

    @classmethod
    def resume(cls: Type[T], experiment_dir: str) -> T:
        """
        Resume an existing ParamSweepExperiment from the given directory.
        """
        experiment_dir_path = Path(experiment_dir)
        if not experiment_dir_path.exists():
            raise FileNotFoundError(
                f"Experiment directory {experiment_dir} does not exist."
            )

        with open(experiment_dir_path / "metadata.json", "r") as f:
            metadata = json.load(f)

        with open(experiment_dir_path / "sweep_metadata.json", "r") as f:
            sweep_metadata = json.load(f)

        executor_params = metadata.get("executor_params", {})  # Load executor_params
        # executor_params must not be empty
        assert executor_params, f"executor_params must not be empty {metadata}"

        exp = cls(
            function=None,  # Will load from disk
            base_params=sweep_metadata["base_params"],
            sweep_params=sweep_metadata["sweep_params"],
            num_replicates=sweep_metadata["num_replicates"],
            experiment_name=metadata["experiment_name"],
            executor_params=executor_params,  # Pass the loaded executor_params
            experiment_dir=experiment_dir,  # Pass experiment_dir here
            output_dir="outputs",
            smoke_test=metadata.get("smoke_test", False),
            debug=metadata.get("debug", False),
            local=metadata.get("local", False),
        )

        exp._load_function()
        exp.load_job_info()
        return exp

    async def run_async(self, parameter_generator: ParameterGenerator):
        # Ensure function is loaded before submission
        if self.function is None:
            self._load_function()

        # Load previously submitted jobs if they exist
        if not self.job_info:
            self.load_job_info()

        # Create a set of previously submitted parameters for quick lookup
        previously_submitted = {make_hashable(info["params"]) for info in self.job_info}
        # logging.info(f"Trying to create a batch with config {self.job_info}")
        with self.executor.batch():
            for params in parameter_generator:
                param_signature = make_hashable(params)
                # Skip if we've already submitted this exact set of parameters
                if param_signature in previously_submitted:
                    # Optionally check if the results for these parameters are already on disk
                    # and if so, just skip them entirely.
                    continue

                job_id = params["job_id"]
                job_output_dir = self.output_dir / "_jobs" / str(job_id)
                job_output_dir.mkdir(parents=True, exist_ok=True)
                job_params = {
                    **self.base_params,
                    **params,
                    "output_dir": str(job_output_dir),
                }
                logging.info(
                    f"Submitting job {job_id} with params: {pformat(job_params)}"
                )
                # breakpoint()
                job = self.executor.submit(
                    self.function,
                    **job_params,
                )
                self.job_info.append({"job": job, "params": params})

        self.save_job_info()

    async def collect_results_async(self, wait_all: bool = True) -> pd.DataFrame:
        """
        Asynchronously collect results from submitted jobs.

        Parameters:
            wait_all (bool):
                - If True, wait for all jobs to complete before collecting results.
                - If False, collect results from only the jobs that have completed so far.

        Returns:
            pd.DataFrame: Aggregated results from the jobs.
        """
        if not self.job_info:
            self.load_job_info()

        coroutines = []
        for info in self.job_info:
            job = info["job"]
            params = info["params"]
            job_id = params.get("job_id")

            # Define an inner function to handle each job
            async def get_result(job=job, params=params, job_id=job_id):
                try:
                    if wait_all:
                        # Wait for the job to complete
                        res = await job.awaitable().result()
                    else:
                        if not job.done:
                            # Skip incomplete jobs
                            return None
                        res = await job.awaitable().result()

                    if not isinstance(res, dict):
                        raise ValueError("Job result must be a dictionary.")
                    record = {**params, **res, "success": True}
                    return record
                except Exception as e:
                    logging.error(f"Exception for job_id {job_id}: {e}", exc_info=True)
                    record = {**params, "success": False, "error": str(e)}
                    return record

            coroutines.append(get_result())

        # Await all coroutines
        results = await asyncio.gather(*coroutines)

        # Filter out None results (from incomplete jobs when wait_all=False)
        if not wait_all:
            results = [r for r in results if r is not None]

        # Create a Pandas DataFrame from the results
        df = pd.DataFrame(results)
        # Keep only columns with scalar data (numbers, strings, booleans)
        scalar_types = (int, float, str, bool, type(None), torch.Tensor)
        scalar_columns = [
            col
            for col in df.columns
            if df[col].apply(lambda x: isinstance(x, scalar_types)).all()
        ]
        non_scalar_columns = set(df.columns) - set(scalar_columns)
        if non_scalar_columns:
            logging.warning(
                f"Ignoring non-scalar columns: {', '.join(non_scalar_columns)}"
            )
        df = df[scalar_columns]
        # ensure proper string dtype instead of 'object'
        # df = df.astype(
        #    {col: "string" for col in df.columns if df[col].dtype == "object"},
        #    errors="ignore",
        # )
        results_path = self.output_dir / f"{self.experiment_name}_results.parquet"
        try:
            df.to_parquet(results_path)
            logging.info(f"Results saved to {results_path}")
        except Exception as e:
            logging.error(f"Failed to save results to {results_path}: {e}")
            logging.error(f"df: {df}")
        return df


# ===================================
# AxExperiment
# ===================================


class AxExperiment(BaseExperiment):
    def __init__(
        self,
        function: Callable[..., Any],
        ax_client: AxClient,
        trial_budget: int,  # Represents total_trial_budget
        experiment_name: str,
        executor_params: Dict[str, Any],
        experiment_dir: Optional[str] = None,  # Added experiment_dir parameter
        subdir: Optional[str] = None,  # Renamed from experiment_dir
        parent_dir: str = "_experiments",  # Added parent_dir
        output_dir: str = "outputs",
        smoke_test: bool = False,
        debug: bool = False,
        local: bool = False,
        base_params: Optional[Dict[str, Any]] = None,
    ):
        """
        AxExperiment manages a Bayesian optimization loop via AxClient, respecting a total trial budget.
        The 'trial_budget' here represents the total number of trials desired over the entire experiment's lifetime.
        """
        super().__init__(
            function=function,
            experiment_name=experiment_name,
            executor_params=executor_params,
            experiment_dir=experiment_dir,  # Pass experiment_dir to BaseExperiment
            subdir=subdir,
            parent_dir=parent_dir,  # Pass parent_dir to BaseExperiment
            output_dir=output_dir,
            smoke_test=smoke_test,
            debug=debug,
            local=local,
            base_params=base_params,
        )
        self.total_trial_budget = trial_budget
        self.ax_client = ax_client
        self.ax_state_path = self.experiment_dir / "ax_state.json"
        self._save_ax_metadata()
        self.save_ax_state()

    def _save_ax_metadata(self):
        metadata = self.load_metadata()
        metadata["total_trial_budget"] = self.total_trial_budget
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    def save_ax_state(self):
        self.ax_client.save_to_json_file(str(self.ax_state_path))
        logging.info(f"AxClient state saved to {self.ax_state_path}")

    def load_ax_state(self):
        if self.ax_state_path.exists():
            self.ax_client = AxClient.load_from_json_file(str(self.ax_state_path))
            logging.info(f"AxClient state loaded from {self.ax_state_path}")

    def _reattach_existing_trials(self):
        """
        Reattach existing trials to ensure that failed or missing trials are handled appropriately.
        """
        updated = False
        for info in self.job_info:
            params = info["params"]
            trial_index = params.get("trial_index")
            if trial_index is not None:
                trial = self.ax_client.experiment.trials.get(trial_index)
                if trial is None:
                    logging.warning(f"Trial {trial_index} not found. Skipping.")
                    continue
                if trial.status.is_completed:
                    logging.info(f"Trial {trial_index} is already completed.")
                    continue
                elif trial.status.is_failed:
                    logging.info(f"Trial {trial_index} has failed. Reattaching.")
                    # Reattach the trial for resubmission
                    clean_params = self._filter_ax_parameters(params)
                    try:
                        parameters, new_trial_index = self.ax_client.attach_trial(
                            parameters=clean_params
                        )
                        if new_trial_index != trial_index:
                            logging.info(
                                f"Reattached trial {trial_index} as {new_trial_index}."
                            )
                            params["trial_index"] = new_trial_index
                            params["job_id"] = new_trial_index
                            updated = True
                    except Exception as e:
                        logging.error(f"Failed to reattach trial {trial_index}: {e}")
                elif trial.status.is_running:
                    logging.info(f"Trial {trial_index} is still running.")
                    # No special action needed here, just acknowledge it.
                else:
                    logging.info(f"Trial {trial_index} has status {trial.status}.")
        if updated:
            self.save_job_info()

    @classmethod
    def create_new(
        cls: Type[T],
        function: Callable[..., Any],
        ax_client: AxClient,
        trial_budget: int,
        experiment_name: str,
        executor_params: Dict[str, Any],
        output_dir: str = "outputs",
        smoke_test: bool = False,
        debug: bool = False,
        local: bool = False,
        parent_dir: str = "_experiments",
        subdir: Optional[str] = None,  # Allow specifying subdir
    ) -> T:
        """
        Create a new AxExperiment instance.
        """
        return cls(
            function=function,
            ax_client=ax_client,
            trial_budget=trial_budget,
            experiment_name=experiment_name,
            executor_params=executor_params,
            subdir=subdir,
            parent_dir=parent_dir,
            output_dir=output_dir,
            smoke_test=smoke_test,
            debug=debug,
            local=local,
        )

    @classmethod
    def resume(
        cls: Type[T], experiment_dir: str, parent_dir: str = "_experiments"
    ) -> T:
        """
        Resume an existing AxExperiment from the given directory.
        """
        experiment_dir_path = Path(experiment_dir)
        if not experiment_dir_path.exists():
            raise FileNotFoundError(
                f"Experiment directory {experiment_dir} does not exist."
            )

        with open(experiment_dir_path / "metadata.json", "r") as f:
            metadata = json.load(f)

        ax_state_path = experiment_dir_path / "ax_state.json"
        if not ax_state_path.exists():
            raise FileNotFoundError("No ax_state.json found for resuming")

        ax_client = AxClient.load_from_json_file(str(ax_state_path))
        total_trial_budget = metadata.get("total_trial_budget", 50)

        executor_params = metadata.get("executor_params", {})  # Load executor_params

        exp = cls(
            function=None,  # Will load from disk
            ax_client=ax_client,
            trial_budget=total_trial_budget,
            experiment_name=metadata["experiment_name"],
            executor_params=executor_params,  # Pass the loaded executor_params
            experiment_dir=experiment_dir,  # Pass experiment_dir here
            output_dir="outputs",
            smoke_test=metadata.get("smoke_test", False),
            debug=metadata.get("debug", False),
            local=metadata.get("local", False),
        )

        exp._load_function()
        exp.load_job_info()
        exp._reattach_existing_trials()
        return exp

    def set_total_trial_budget(self, new_budget: int):
        """
        Increase the total_trial_budget.
        The new budget must be greater than or equal to the current one.
        """
        if new_budget < self.total_trial_budget:
            raise ValueError(
                "New trial budget must be >= the current total_trial_budget."
            )
        self.total_trial_budget = new_budget
        # Update metadata
        with open(self.metadata_path, "r+") as f:
            metadata = json.load(f)
            metadata["total_trial_budget"] = new_budget
            f.seek(0)
            json.dump(metadata, f, indent=4)
            f.truncate()
        logging.info(f"Updated total_trial_budget to {new_budget}.")
        # Update AxClient's trial budget
        self.ax_client.experiment.trial_generator.set_total_trials(new_budget)

    def create_parameter_generator(
        self, number_of_trials: Optional[int] = None
    ) -> ParameterGenerator:
        """
        Create a parameter generator for a given number_of_trials. If not specified,
        it will compute the remaining trials that fit into the total_trial_budget.
        """
        if number_of_trials is None:
            existing_trials = len(self.ax_client.experiment.trials)
            remaining_budget = self.total_trial_budget - existing_trials
            if remaining_budget <= 0:
                logging.info(
                    "No remaining trials to submit based on the current total_trial_budget."
                )
                return AxParameterGenerator(self.ax_client, 0)
            return AxParameterGenerator(self.ax_client, remaining_budget)
        else:
            return AxParameterGenerator(self.ax_client, number_of_trials)

    async def run_async(self, parameter_generator: Optional[ParameterGenerator] = None):
        """
        Asynchronously run the AxExperiment by submitting trials generated by the parameter_generator.
        """
        # Ensure function is loaded before submission
        if self.function is None:
            self._load_function()

        # If user didn't provide a parameter_generator, generate one from remaining_budget
        if parameter_generator is None:
            existing_trials = len(self.ax_client.experiment.trials)
            remaining_budget = self.total_trial_budget - existing_trials
            if remaining_budget <= 0:
                logging.info(
                    "No remaining trials to submit based on the current total_trial_budget."
                )
                return
            parameter_generator = AxParameterGenerator(self.ax_client, remaining_budget)

        # Obtain an iterator from the parameter_generator
        iterator = iter(parameter_generator)

        # Iterate over the generator and handle stalls by waiting and retrying
        while True:
            try:
                params = next(iterator)
                job_id = params["job_id"]
                trial_index = params["trial_index"]
                job_output_dir = self.output_dir / "_jobs" / str(job_id)
                job_output_dir.mkdir(parents=True, exist_ok=True)
                job_params = {
                    **self.base_params,
                    **params,
                    "output_dir": str(job_output_dir),
                }
                logging.info(
                    f"Submitting job {job_id} with params: {pformat(job_params)}"
                )
                job = self.executor.submit(
                    self.function,
                    **job_params,
                )
                self.job_info.append({"job": job, "params": params})
            except StopIteration:
                # No more trials to generate
                break
            except ParameterStalledException:
                # Stall detected: wait and then retry
                logging.info("Stalled; waiting before retrying...")
                await asyncio.sleep(5)
                continue
            except Exception as e:
                logging.error(
                    f"Unexpected exception during run_async: {e}", exc_info=True
                )
                break

        self.save_job_info()

    async def collect_results_async(self, wait_all: bool = True) -> pd.DataFrame:
        """
        Asynchronously collect results from submitted jobs.

        Parameters:
            wait_all (bool):
                - If True, wait for all jobs to complete before collecting results.
                - If False, collect results from only the jobs that have completed so far.

        Returns:
            pd.DataFrame: Aggregated results from the jobs.
        """
        if not self.job_info:
            self.load_job_info()

        coroutines = []
        for info in self.job_info:
            job = info["job"]
            params = info["params"]
            job_id = params.get("job_id")
            trial_index = params.get("trial_index", None)

            # Define an inner function to handle each job
            async def get_result(
                job=job, params=params, job_id=job_id, trial_index=trial_index
            ):
                try:
                    if wait_all:
                        # Wait for the job to complete
                        res = await job.awaitable().result()
                    else:
                        # Check if the job is done
                        if not job.done:
                            # Skip incomplete jobs
                            return None
                        res = await job.awaitable().result()

                    if not isinstance(res, dict):
                        raise ValueError("Job result must be a dictionary.")
                    record = {**params, **res, "success": True}
                    if trial_index is not None:
                        trial = self.ax_client.experiment.trials[trial_index]
                        if trial.status.is_running:
                            # Update AxClient with the trial's results
                            self.ax_client.complete_trial(
                                trial_index=trial_index, raw_data=res
                            )
                            logging.info(f"Logged trial {trial_index} with {res}")
                        else:
                            logging.warning(f"Trial {trial_index} is not running.")
                    return record
                except Exception as e:
                    logging.error(f"Exception for job_id {job_id}: {e}", exc_info=True)
                    record = {**params, "success": False, "error": str(e)}
                    if trial_index is not None:
                        trial = self.ax_client.experiment.trials.get(trial_index)
                        if trial and trial.status.is_running:
                            self.ax_client.log_trial_failure(
                                trial_index=trial_index, metadata={"error": str(e)}
                            )
                    return record

            coroutines.append(get_result())

        # Await all coroutines
        results = await asyncio.gather(*coroutines)

        # Filter out None results (from incomplete jobs when wait_all=False)
        if not wait_all:
            results = [r for r in results if r is not None]

        # Clean up the results (e.g., ensure all objectives are scalars)
        results = clean_ax_results_objectives(results, self.ax_client.objective_names)

        # Create a Pandas DataFrame from the results
        df = pd.DataFrame(results)
        results_path = self.output_dir / f"{self.experiment_name}_results.parquet"

        # Save AxClient state
        self.save_ax_state()

        try:
            df.to_parquet(results_path)
            logging.info(f"Results saved to {results_path}")
        except Exception as e:
            logging.error(f"Failed to save results to {results_path}: {e}")
        return df

    def _filter_ax_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter parameters to include only those defined in the Ax experiment's search space.
        """
        valid_params = {p.name for p in self.ax_client.experiment.parameters}
        return {k: v for k, v in params.items() if k in valid_params}


# ===================================
# Utility functions
# ===================================


def list_experiments(
    parent_dir: str = "_experiments", experiment_name: Optional[str] = None
) -> List[Path]:
    """
    Lists all experiment directories within the parent directory, optionally filtering by experiment name.

    Args:
        parent_dir (str, optional): Parent directory containing experiments. Defaults to "_experiments".
        experiment_name (str, optional): Specific experiment name to filter by. Defaults to None.

    Returns:
        List[Path]: List of experiment directory paths.
    """
    parent_path = Path(parent_dir)
    if experiment_name:
        pattern = f"{experiment_name}*"
    else:
        pattern = "*"
    experiment_dirs = sorted(parent_path.glob(pattern))
    return experiment_dirs


def get_latest_experiment(
    parent_dir: str = "_experiments", experiment_name: Optional[str] = None
) -> Optional[Path]:
    """
    Retrieves the latest experiment directory matching the specified name.

    Args:
        parent_dir (str, optional): Parent directory containing experiments. Defaults to "_experiments".
        experiment_name (str, optional): Specific experiment name to filter by. Defaults to None.

    Returns:
        Optional[Path]: Path to the latest experiment directory or None if none found.
    """
    experiments = list_experiments(parent_dir, experiment_name)
    if experiments:
        return experiments[-1]
    return None


def terminate_latest_experiment(
    experiment_name: str,
    parent_dir: str = "_experiments",
    delete_files: bool = False,
) -> None:
    """
    Terminates the latest experiment matching the specified name and optionally deletes associated files.

    Args:
        experiment_name (str): The base name of the experiment to terminate.
        parent_dir (str, optional): Parent directory containing experiments. Defaults to "_experiments".
        delete_files (bool, optional): If True, deletes the experiment directory after termination. Defaults to False.
    """
    latest_experiment_dir = get_latest_experiment(parent_dir, experiment_name)
    if not latest_experiment_dir:
        logging.info(f"No experiments found with name {experiment_name}")
        return

    logging.info(f"Terminating experiment in directory: {latest_experiment_dir}")

    # Determine experiment type
    if (latest_experiment_dir / "sweep_metadata.json").exists():
        # It's a ParamSweepExperiment
        try:
            experiment = ParamSweepExperiment.resume(str(latest_experiment_dir))
        except Exception as e:
            logging.error(f"Failed to resume ParamSweepExperiment: {e}")
            return
    elif (latest_experiment_dir / "ax_state.json").exists():
        # It's an AxExperiment
        try:
            experiment = AxExperiment.resume(str(latest_experiment_dir))
        except Exception as e:
            logging.error(f"Failed to resume AxExperiment: {e}")
            return
    else:
        logging.error(f"Unknown experiment type in {latest_experiment_dir}")
        return

    experiment.terminate(delete_files=delete_files)
    logging.info(f"Experiment {latest_experiment_dir} terminated successfully.")


# ===================================
# Main examples
# ===================================

"""
This is what trial fucntions need to look like:
```
def example_trail_func(
        *args,
        seed, job_id, replicate=0,
        smoke_test=False, output_dir="outputs"):
    return {"loss": 0.0, "accuracy": 1.0}
```

It is a good idea if they accept all the helpful diagnostic information that is passed to them.
They don't need to do anything with it, but maybe you want to e.g. set the task to be easier if we are doing a smoke test.

They had better return PLAIN PYTHON FLOATS, INTS, STRINGS.
Pandas will happily accept other things (e.g. `torch` tensors) but all kinds of stupid behavior will result from that because they are interpreted as opaque objects and they crash things.

**tldr** call `.item()` on torch tensors before returning them.
"""


def param_sweep_example_synchronous():
    def sweep_trial(a, b, seed, job_id, smoke_test=False, output_dir="outputs"):
        np.random.seed(seed)
        val = a + b + (0 if smoke_test else np.random.randn())
        # The function can store additional output files in output_dir if desired
        return {"value": val}

    import os
    import numpy as np
    import logging
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    experiment_name = "example_sweep"

    executor_params = {
        "timeout_min": 10,
        "tasks_per_node": 1,
        "mem_gb": 8,
        "cpus_per_task": 1,
        "gpus_per_node": 0,
        "slurm_account": os.getenv("SLURM_ACCOUNT"),
    }

    # Parameters for a new experiment
    new_experiment_kwargs = {
        "function": sweep_trial,
        "base_params": {},
        "sweep_params": {"a": 10 ** np.linspace(-3, 3, 13, endpoint=True)},
        "num_replicates": 12,
        "executor_params": executor_params,
        "experiment_name": experiment_name,
        "output_dir": "outputs",
        "smoke_test": False,
        "debug": False,
        "local": False,
        "parent_dir": "_experiments",
    }

    # Attempt to get or create the experiment
    pexp = ParamSweepExperiment.get_or_create(
        experiment_name=experiment_name,
        parent_dir="_experiments",
        **new_experiment_kwargs,
    )

    # Create a parameter generator
    gen = pexp.create_parameter_generator()

    # Run the experiment synchronously
    pexp.run_sync(gen)

    # Collect the results synchronously
    results_df = pexp.collect_results_sync()
    print("Param Sweep Results:\n", results_df)

    # Optionally terminate the experiment
    # pexp.terminate(delete_files=True)

    return pexp


"""
# How we would invoke this inside a notebook
def sweep_trial(a, b, seed, job_id, replicate=0, smoke_test=False, output_dir="outputs"):
    np.random.seed(seed)
    val = a + b + (0 if smoke_test else np.random.randn())
    # The function can store additional output files in output_dir if desired
    return {"value": val}

import os
import numpy as np
import logging

# Ensure logging is configured
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

executor_params = {
    "timeout_min": 10,
    "tasks_per_node": 1,
    "mem_gb": 8,
    "cpus_per_task": 1,
    "gpus_per_node": 0,
    'slurm_account': os.getenv('SLURM_ACCOUNT')
}

# Create a new param sweep experiment
pexp = ParamSweepExperiment.create_new(
    function=sweep_trial,
    base_params={},
    sweep_params={"f_coupling": 10**np.linspace(-3, 3, 13, endpoint=True)},
    num_replicates=12,
    experiment_name="surrogate_sweep",
    executor_params=executor_params,
    debug=False,
    # smoke_test=True,  # Uncomment for a smoke test
)

gen = pexp.create_parameter_generator()

# Run the experiment asynchronously
await pexp.run_async(gen)

# Collect the results asynchronously
results_df = await pexp.collect_results_async()
print("Param Sweep Results:\n", results_df)
"""


def ax_experiment_example_synchronous():
    from ax.utils.measurement.synthetic_functions import hartmann6

    executor_params = {
        "timeout_min": 10,
        "slurm_partition": "dev",
        "tasks_per_node": 1,
        "mem_gb": 1,
        "cpus_per_task": 1,
        "gpus_per_node": 0,
    }

    def hartmann_trial(job_id, seed, output_dir="outputs", **params):
        x = np.array([params.get(f"x{i+1}") for i in range(6)])
        val = hartmann6(x)
        return {"foo": val}

    ax_client = AxClient()
    ax_client.create_experiment(
        name="ax_hartmann_test",
        parameters=[
            {"name": f"x{i+1}", "type": "range", "bounds": [0.0, 1.0]} for i in range(6)
        ],
        objectives={"foo": ObjectiveProperties(minimize=True)},
    )

    # Create a new AxExperiment with a trial_budget of 5
    ax_exp = AxExperiment.create_new(
        function=hartmann_trial,
        ax_client=ax_client,
        trial_budget=5,  # Initial total_trial_budget
        experiment_name="ax_exp_test",
        executor_params=executor_params,
        output_dir="outputs",
        smoke_test=False,
        debug=True,
        parent_dir="_experiments",
        # subdir="custom_ax_subdir",  # Optionally specify a subdir
    )
    ax_gen = ax_exp.create_parameter_generator()
    ax_exp.run_sync(ax_gen)
    ax_results = ax_exp.collect_results_sync()
    logging.info(f"Ax Results:\n{ax_results}")

    best_parameters, values = ax_client.get_best_parameters()
    logging.info(f"Best parameters: {best_parameters}, values: {values}")

    # Example of updating the trial budget to 7 and running additional trials
    logging.info("Increasing trial budget to 7 and submitting additional trials.")
    ax_exp.set_total_trial_budget(7)
    ax_gen = (
        ax_exp.create_parameter_generator()
    )  # Create a new generator for the additional trials
    ax_exp.run_sync(ax_gen)  # Run the additional trials
    ax_results_updated = ax_exp.collect_results_sync()  # Collect the updated results
    logging.info(f"Updated Ax Results:\n{ax_results_updated}")

    best_parameters, values = ax_client.get_best_parameters()
    logging.info(f"Updated Best parameters: {best_parameters}, values: {values}")

    # Clean up experiments if you want
    # ax_exp.terminate(delete_files=True)
    return ax_exp


"""
## An asynchronous example to show how to run this from an interactive notebook.

import os
import numpy as np
import asyncio
import logging

from ax.service.ax_client import AxClient

# Ensure logging is configured
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def hartmann_trial(job_id, seed, output_dir="outputs", **params):
    x = np.array([params.get(f"x{i+1}") for i in range(6)])
    val = hartmann6(x)
    return {"foo": val}

def ax_experiment_async_example():
    # Initialize AxClient and create an experiment
    ax_client = AxClient()
    ax_client.create_experiment(
        name="ax_hartmann_test",
        parameters=[
            {"name": f"x{i+1}", "type": "range", "bounds": [0.0, 1.0]} for i in range(6)
        ],
        objectives={"foo": ObjectiveProperties(minimize=True)},
    )

    executor_params = {
        "timeout_min": 10,
        "slurm_partition": "dev",
        "tasks_per_node": 1,
        "mem_gb": 1,
        "cpus_per_task": 1,
        "gpus_per_node": 0,
    }

    # Create a new AxExperiment with a trial_budget of 5
    ax_exp = AxExperiment.create_new(
        function=hartmann_trial,
        ax_client=ax_client,
        trial_budget=5,  # Initial total_trial_budget
        experiment_name="ax_exp_test",
        executor_params=executor_params,
        smoke_test=False,
        debug=True,
    )

    # Create a parameter generator
    ax_gen = ax_exp.create_parameter_generator()

    # Run the experiment asynchronously
    await ax_exp.run_async(ax_gen)

    # Collect results asynchronously
    ax_results = await ax_exp.collect_results_async()
    logging.info(f"Ax Results:\n{ax_results}")

    # Retrieve the best parameters found so far
    best_parameters, values = ax_client.get_best_parameters()
    logging.info(f"Best parameters: {best_parameters}, values: {values}")

    # Example of updating the trial budget to 7 and running additional trials
    logging.info("Increasing trial budget to 7 and submitting additional trials.")
    ax_exp.set_total_trial_budget(7)
    await ax_exp.run_async(ax_exp.create_parameter_generator())
    ax_results_updated = await ax_exp.collect_results_async()
    logging.info(f"Updated Ax Results:\n{ax_results_updated}")

    best_parameters, values = ax_client.get_best_parameters()
    logging.info(f"Updated Best parameters: {best_parameters}, values: {values}")

    # Clean up experiments if you want
    # ax_exp.terminate(delete_files=True)

    return ax_exp
"""


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    param_sweep_example_synchronous()
    ax_experiment_example_synchronous()


if __name__ == "__main__":
    main()
