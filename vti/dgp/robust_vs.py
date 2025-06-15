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

import torch
import torch.nn.functional as F
import numpy as np
from vti.distributions.uniform_bs import UniformBinaryString
import torch.distributions as dist
from vti.dgp import AbstractDGP
from vti.utils.seed import set_seed, restore_seed
import logging
from typing import Optional, Tuple, Any, Callable

from vti.dgp.param_transform_factory import (
    construct_param_transform,
    construct_diagnorm_param_transform,
)
from vti.utils.math_helpers import ensure_2d, upper_bound_power_of_2


def generate_random_signs(size):
    # Generate a tensor of 0s and 1s
    random_bits = torch.randint(
        0, 2, size
    )  # This generates a tensor of the given size with 0 or 1

    # Map 0 to -1 and 1 to +1
    signs = 2 * random_bits - 1  # This maps 0 to -1 and 1 to +1

    return signs


def integers_to_binary(int_tensor, dim, dtype):
    # Create a mask for each bit position from 0 to context_dim-1
    masks = 1 << torch.arange(dim - 1, -1, -1, dtype=torch.int64)

    # Apply the mask to the tensor and right shift to bring the masked bit to the least significant position
    binary_matrix = ((int_tensor.unsqueeze(-1) & masks) > 0).to(
        dtype
    )  # Convert boolean to integer

    return binary_matrix


def generate_coefficients(total_sum, num_coefficients=6):
    # Generate random positive numbers
    coefficients = torch.rand(num_coefficients)

    # Normalize the coefficients so they sum to `total_sum`
    coefficients /= coefficients.sum()
    coefficients *= total_sum

    return coefficients


class RobustVS(AbstractDGP):
    """
    Robust variable selection
    """

    def __init__(
        self,
        seed=1,
        dimension=10,
        num_data=100,
        param_prior_scale = 1.5, # narrow
        misspec_level='high', # 'high','mid','none'
        run_rjmcmc=False,
        one_hot_encode_context=True,
        smoke_test=False, # TODO delete
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(device, dtype)

        assert dimension > 1, "Dimension must be > 1. Y-intercept is first dimension"
        self.smoke_test = smoke_test
        self.seed = seed
        self.sigma1 = torch.tensor(1,device=self.device,dtype=self.dtype)
        self.sigma2 = torch.tensor(10,device=self.device,dtype=self.dtype)
        self.outlier_fraction = torch.tensor(0.1,device=self.device,dtype=self.dtype)
        self.param_prior_scale = param_prior_scale
        self.one_hot_encode_context = one_hot_encode_context

        self.ROBUST_VS_THETA_SCALE = 1.

        if misspec_level=='high':
            # hard problem
            num_data=50
            x_data, y_data = self._generate(
                seed,
                num_data,
                dimension,
                # misspecified to likelihood
                #sigma1=self.sigma1 * 2,
                #sigma2=self.sigma2 // 2,
                #sigma1=self.sigma1,
                #sigma2=self.sigma2,
                sigma1=4.,
                sigma2=4.,
                sparsity=0.4,
                correlation_proportion=0.4,
                #correlation_proportion=0.2,
                correlation_factor=0.1,
                sigmax=1,
                sigmabeta1=0.0,
                sigmabeta2=0.0,
                sigmabeta_split=0.5, # proportion of data that uses sigmabeta2. A value of 0 means all data uses sigmabeta1.
                beta1 = 0.5,
                beta2 = 1.5,
                outlier_fraction=self.outlier_fraction,
            )
        elif misspec_level=='mid':
            # middle level problem
            num_data=50
            x_data, y_data = self._generate(
                seed,
                num_data,
                dimension,
                # misspecified to likelihood
                sigma1=self.sigma1 * 2,
                sigma2=self.sigma2 // 2,
                #sigma1=self.sigma1,
                #sigma2=self.sigma2,
                #sigma1=4.,
                #sigma2=4.,
                sparsity=0.4,
                #correlation_proportion=0.4,
                correlation_proportion=0.0, # no correlation
                correlation_factor=0.1,
                sigmax=1,
                sigmabeta1=0.0,
                sigmabeta2=0.0,
                sigmabeta_split=0.5, # proportion of data that uses sigmabeta2. A value of 0 means all data uses sigmabeta1.
                beta1 = 0.5,
                beta2 = 1.5,
                outlier_fraction=self.outlier_fraction,
            )
        elif misspec_level=='none':
            # easy problem
            num_data = 50
            #self.sigma1=torch.tensor(0.5,device=self.device,dtype=self.dtype)
            #self.sigma2=torch.tensor(0.5,device=self.device,dtype=self.dtype)
            x_data, y_data = self._generate(
                seed,
                num_data,
                dimension,
                # defaults for matching the likelihood (easy problem)
                sigma1=self.sigma1, # give it the dgp for now
                sigma2=self.sigma2,
                sparsity=0.4,
                correlation_proportion=0.0,
                correlation_factor=0.0,
                sigmax=1,
                sigmabeta1=0.0,
                sigmabeta2=0.0,
                sigmabeta_split=0.0,
                beta1 = 0.5,
                beta2 = 0.5,
                outlier_fraction=self.outlier_fraction,
            )
        else:
            raise NotImplementedError(f"Unsupported misspecification level {misspec_level}")

        self.register_buffer("x_data", x_data.to(device=self.device, dtype=self.dtype))
        self.register_buffer("y_data", y_data.to(device=self.device, dtype=self.dtype))

        self.dimension = dimension  # full dimension
        self.data_dimension = dimension - 1
        self.num_bits = (
            self.data_dimension
        )  # TODO FIXME consolidate with dimension. Write docstrings.

        if run_rjmcmc:
            self.sample_true_mk_probs()

        logging.info("Initialised DGP")


    def get_data(self):
        return self.x_data, self.y_data


    def sample_true_mk_probs(self, plot_bivariates=False, store_samples=False, save_bivariates_path=False):
        mk_samples, theta_samples = self._run_rjmcmc()
        #if self.num_bits <= 32:
        #    self.true_mk_probs_all = self._convert_RJMCMC_to_model_probs(
        #        mk_samples, theta_samples
        #    )
        logging.info("converting RJMCMC to model probs")
        self.true_mk_identifiers, self.true_mk_probs = (
            self._convert_RJMCMC_to_model_probs_approx(
                mk_samples, theta_samples, plot_bivariates=plot_bivariates, save_bivariates_path=save_bivariates_path,
            )
        )
        logging.info("done converting RJMCMC to model probs")
        if store_samples:
            self.stored_rjmcmc_samples = {
                "mk": mk_samples.contiguous(),
                "theta": theta_samples.contiguous(),
            }
            # self.stored_rjmcmc_samples = {
            #    "mk": mk_samples[::thinning].contiguous(),
            #    "theta": theta_samples[::thinning].contiguous(),
            # }

    # def compute_average_nll(self, q):
    #    """
    #    Computes the average Negative Log-Likelihood (NLL) of the approximation q
    #    over the stored transdimensional RJMCMC samples.

    #    Parameters:
    #        q: An approximation distribution object with a `log_prob` method.
    #           The `log_prob` method should accept two arguments:
    #           - mk: Model indices
    #           - theta: Continuous parameters corresponding to each model

    #    Returns:
    #        float: The average NLL computed over the samples.
    #    """
    #    # Assert that the attribute 'stored_rjmcmc_samples' exists
    #    if not hasattr(self, "stored_rjmcmc_samples"):
    #        raise AttributeError(
    #            "The object must have a 'stored_rjmcmc_samples' attribute."
    #        )

    #    # Extract samples from the stored RJMCMC samples
    #    samples = self.stored_rjmcmc_samples

    #    # Ensure that both 'mk' and 'theta' are present in the samples
    #    if "mk" not in samples or "theta" not in samples:
    #        raise KeyError(
    #            "The 'stored_rjmcmc_samples' must contain both 'mk' and 'theta' keys."
    #        )

    #    mk = samples["mk"]
    #    theta = samples["theta"]

    #    # Compute the log probabilities using the approximation q
    #    # TODO FIXME the below is saturated log prob. Do we subtract the reference log prob?
    #    log_probs = q.log_prob(mk, theta) - self.reference_log_prob(mk, theta)

    #    # Compute the average Negative Log-Likelihood
    #    average_nll = (-log_probs).mean()

    #    return average_nll.item()

    def compute_average_nll(self, q, chunk_size=1024):
        """
        Computes the average Negative Log-Likelihood (NLL) of the approximation q
        over the stored transdimensional RJMCMC samples by processing data in chunks.

        Parameters:
            q: An approximation distribution object with a `log_prob` method.
               The `log_prob` method should accept two arguments:
               - mk: Model indices (torch.LongTensor or similar)
               - theta: Continuous parameters corresponding to each model (torch.Tensor)
            chunk_size (int, optional): The number of samples to process in each chunk. Default is 1024.

        Returns:
            float: The average NLL computed over the samples.
        """
        # Extract samples from the stored RJMCMC samples
        samples = self.stored_rjmcmc_samples

        # Ensure that both 'mk' and 'theta' are present in the samples
        if "mk" not in samples or "theta" not in samples:
            raise KeyError(
                "The 'stored_rjmcmc_samples' must contain both 'mk' and 'theta' keys."
            )

        mk = samples["mk"]  # Expected shape: (num_samples, ...)
        theta = samples["theta"]  # Expected shape: (num_samples, ...)

        # Ensure that mk and theta have the same number of samples
        if mk.shape[0] != theta.shape[0]:
            raise ValueError("The number of 'mk' and 'theta' samples must be the same.")

        total_neg_log_prob = torch.tensor(
            0.0, device=mk.device
        )  # Accumulates the total negative log probabilities
        total_samples = 0  # Keeps track of the total number of samples processed

        num_samples = mk.shape[0]

        # Disable gradient computation for efficiency
        with torch.no_grad():
            for start_idx in range(0, num_samples, chunk_size):
                end_idx = start_idx + chunk_size
                mk_chunk = mk[start_idx:end_idx]
                theta_chunk = theta[start_idx:end_idx]

                # Compute the log probabilities for the current chunk
                log_probs = q.log_prob(mk_chunk, theta_chunk)
                reference_log_probs = self.reference_log_prob(mk_chunk, theta_chunk)

                # Compute the negative log likelihood for the current chunk
                neg_log_probs = -(log_probs - reference_log_probs)

                # Accumulate the total negative log probabilities and sample count
                total_neg_log_prob += neg_log_probs.sum()
                total_samples += neg_log_probs.numel()

        if total_samples == 0:
            raise ValueError(
                "No samples were processed. Check if 'mk' and 'theta' are non-empty."
            )

        # Compute the average Negative Log-Likelihood
        average_nll = total_neg_log_prob / total_samples

        return average_nll.item()


    # def compute_conditional_average_nll(self, q, mk):
    #    """
    #    Same as compute_average_nll() but conditional on the model identifier mk
    #    """
    #    theta = ensure_2d(self._get_true_theta_given_mk(mk))
    #    mk = ensure_2d(mk)

    #    if len(theta) == 0:
    #        # return float("inf")
    #        return 0.0  # undecided here

    #    # Compute the log probabilities using the approximation q
    #    # TODO FIXME the below is saturated log prob. Do we subtract the reference log prob?
    #    # logging.info(f"cond nll mk {mk} theta {theta}")
    #    log_probs = q._cond_param_log_prob(
    #        mk, theta, self.mk_to_context
    #    ) - self.reference_log_prob(mk, theta)

    #    # Compute the average Negative Log-Likelihood
    #    average_nll = (-log_probs).mean()

    #    return average_nll.item()

    def compute_conditional_average_nll(self, q, mk, chunk_size=1024):
        # [Method implementation as provided above]
        mk = ensure_2d(mk)
        theta = ensure_2d(self._get_true_theta_given_mk(mk))

        if len(theta) == 0:
            return 0.0

        total_neg_log_prob = torch.tensor(0.0, device=self.device)
        total_samples = 0

        num_samples = theta.shape[0]

        with torch.no_grad():
            for start_idx in range(0, num_samples, chunk_size):
                end_idx = start_idx + chunk_size
                theta_chunk = theta[start_idx:end_idx]

                log_probs = q._cond_param_log_prob(mk, theta_chunk, self.mk_to_context)
                reference_log_probs = self.reference_log_prob(mk, theta_chunk)

                neg_log_probs = -(log_probs - reference_log_probs)

                total_neg_log_prob += neg_log_probs.sum()
                total_samples += neg_log_probs.numel()

        if total_samples == 0:
            raise ValueError(
                "No samples were processed. Check if 'mk' and 'theta' are non-empty."
            )

        average_nll = total_neg_log_prob / total_samples

        return average_nll.item()

    def _get_true_theta_given_mk(self, mk):
        """
        From RJMCMC output, get the rows of theta that have mk samples matching input argument
        """
        # Assert that the attribute 'stored_rjmcmc_samples' exists
        if not hasattr(self, "stored_rjmcmc_samples"):
            raise AttributeError(
                "The object must have a 'stored_rjmcmc_samples' attribute."
            )

        # Extract samples from the stored RJMCMC samples
        samples = self.stored_rjmcmc_samples

        # Ensure that both 'mk' and 'theta' are present in the samples
        if "mk" not in samples or "theta" not in samples:
            raise KeyError(
                "The 'stored_rjmcmc_samples' must contain both 'mk' and 'theta' keys."
            )

        mk_samples = samples["mk"]
        theta_samples = samples["theta"]

        matches = torch.all(mk_samples == mk, dim=1)
        indices = torch.nonzero(matches).squeeze()

        return theta_samples[indices]

    def get_sfe_lr(self):
        # return 1e-3
        # TODO tune these
        if self.dimension < 10:
            return 1e-1
        elif self.dimension <= 50:
            return 1e-1 # 1e-2
        else:
            return 1e-3

    def get_rjmcmc_config(self):
        # TODO tune these
        # num_iterations, burn_in, num_runs (batch_size),
        M = 1
        config = {
            "num_iterations": 10000,
            "burn_in": 10000*M,
            #"num_runs": 1024,
            #"num_runs": 2048,
            "num_runs": 4096,
            "thinning": 10*M, # 50*M
        }
        if self.smoke_test == True:
            config["num_iterations"] = 20
            config["burn_in"] = 10
            config["thinning"] = 1
        if self.dimension <= 5:
            config["num_iterations"] = 10000 * M
            config["thinning"] = 50 * M
        elif self.dimension <= 10:
            config["num_iterations"] = 20000 * M
            config["thinning"] = 100 * M
        elif self.dimension <= 20:
            config["num_iterations"] = 40000 * M
            config["thinning"] = 200 * M
        elif self.dimension <= 50:
            config["num_iterations"] = 160000 * M
            config["thinning"] = 400 * M
        else:
            config["num_iterations"] = 240000 * M
            config["thinning"] = 800 * M

        #config["thinning"] *= 10
        return config

    # def _run_rjmcmc_unvectorised(self):
    #    if False:
    #        # store data if desired
    #        alldatanp = np.column_stack(
    #            [self.y_data.detach().cpu().numpy(), self.x_data.detach().cpu().numpy()]
    #        )
    #        np.save(
    #            "rvs_data_dim{}_seed{}.npy".format(self.data_dimension + 1, self.seed),
    #            alldatanp,
    #        )

    #    # TODO expose these parameters for external configuration
    #    #rjmcmc_steps, burnin, num_runs = self.get_rjmcmc_config()
    #    config_args = self.get_rjmcmc_config()

    #    mk_samples_all = []
    #    theta_samples_all = []
    #    logging.info("RJMCMC baseline...")

    #    for i in range(num_runs):
    #        mk_samples, theta_samples = self.sample_RJMCMC_naive(rjmcmc_steps)

    #        # Elementwise multiply by mask
    #        theta_samples = theta_samples * self.mk_to_mask(mk_samples)

    #        # Detach and slice out burn-in region
    #        mk_samples = mk_samples.detach().clone()[burnin:]
    #        theta_samples = theta_samples.detach().clone()[burnin:]

    #        mk_samples_all.append(mk_samples)
    #        theta_samples_all.append(theta_samples)
    #        logging.info(f"Run {i+1} out of {num_runs} done")

    #    # Concatenate all samples as torch tensors
    #    mk_samples = torch.cat(mk_samples_all, dim=0)
    #    theta_samples = torch.cat(theta_samples_all, dim=0)

    #    return mk_samples, theta_samples

    def _run_rjmcmc(self):
        if False:
            # store data if desired
            alldatanp = np.column_stack(
                [self.y_data.detach().cpu().numpy(), self.x_data.detach().cpu().numpy()]
            )
            np.save(
                "rvs_data_dim{}_seed{}.npy".format(self.data_dimension + 1, self.seed),
                alldatanp,
            )

        # TODO expose these parameters for external configuration
        # rjmcmc_steps, burnin, num_runs, thinning = self.get_rjmcmc_config()
        config_args = self.get_rjmcmc_config()

        logging.info(f"RJMCMC baseline (vectorised) with config {config_args} ...")

        oldrngstate = set_seed(self.seed)  # for plot consistency
        mk_samples, theta_samples = self.sample_RJMCMC_vectorised(
            # rjmcmc_steps, num_runs, thinning=thinning
            **config_args
        )
        restore_seed(oldrngstate)

        logging.info(f"RJMCMC baseline done.")

        # Detach and slice out burn-in region
        mk_samples = mk_samples.detach()
        theta_samples = theta_samples.detach()

        # Reshape to concatenate all runs
        # From [num_runs, num_iterations * 2 - burnin, ...] to [num_runs * (num_iterations * 2 - burnin), ...]
        logging.info(f"RJMCMC mk samples\n {mk_samples.shape} {mk_samples}\n {theta_samples.shape} {theta_samples}")
        mk_samples = mk_samples.reshape(-1, self.data_dimension)
        theta_samples = theta_samples.reshape(-1, self.num_inputs())
        logging.info(f"RJMCMC mk samples\n {mk_samples.shape} {mk_samples}\n {theta_samples.shape} {theta_samples}")

        # Elementwise multiply by mask
        theta_samples = theta_samples * self.mk_to_mask(mk_samples)

        return mk_samples, theta_samples

    def _convert_RJMCMC_to_model_probs(
        self, mk_samples, theta_samples, plot_bivariates=False
    ):
        """
        Only run for small model spaces.
        """
        raise NotImplementedError("Do not use this method")
        assert (
            self.num_bits < 32
        ), "Model space too large to convert to enumerated list of model probs"

        # old code that groups
        unique_rows, reverse_indices = torch.unique(
            mk_samples, dim=0, return_inverse=True
        )

        # Create a list to hold the arrays of theta_samples corresponding to each unique row in mk_samples
        grouped_theta_samples = []

        modelprobs = torch.zeros(
            self.num_categories(), dtype=self.dtype, device=self.device
        )

        # Loop over each unique row index to gather corresponding rows from theta_samples
        for idx in range(len(unique_rows)):
            mask = (
                reverse_indices == idx
            )  # Create a mask for the current unique row index
            grouped_theta_samples.append(theta_samples[mask].detach().cpu().numpy())
            # mkprob = float(mask.sum()) / (rjmcmc_steps - burnin)
            mkprob = float(mask.sum()) / (mk_samples.shape[0])
            # logging.info(f"{unique_rows[idx]}\t\t{mkprob}")
            # logging.info(
            #    f"id2cat {int(self.mk_identifier_to_cat(unique_rows[idx]).item())} maxcat {modelprobs.shape[0]}"
            # )
            modelprobs[int(self.mk_identifier_to_cat(unique_rows[idx]).item())] = mkprob

        if plot_bivariates:
            from vti.utils.plots import plot_fit_marginals

            plot_fit_marginals(grouped_theta_samples[0], grouped_theta_samples[1:])

        return modelprobs

    def _convert_RJMCMC_to_model_probs_approx(
        self, mk_samples, theta_samples, plot_bivariates=False, save_bivariates_path=False,
    ):
        """
        Return two tensors: unique mk identifiers, and associated model probs.
        Sorted descending by model probs.
        """
        # old code that groups
        unique_rows, reverse_indices = torch.unique(
            mk_samples, dim=0, return_inverse=True
        )

        # Create a list to hold the arrays of theta_samples corresponding to each unique row in mk_samples
        grouped_theta_samples = []

        # topmodelprobs = torch.zeros(
        #    self.num_categories(), dtype=self.dtype, device=self.device
        # )
        topmodelprobs = torch.zeros(
            len(unique_rows), dtype=self.dtype, device=self.device
        )

        N = mk_samples.shape[0]

        # Loop over each unique row index to gather corresponding rows from theta_samples
        for idx in range(len(unique_rows)):
            mask = (
                reverse_indices == idx
            )  # Create a mask for the current unique row index
            grouped_theta_samples.append(theta_samples[mask].detach().cpu().numpy())
            # mkprob = float(mask.sum()) / (rjmcmc_steps - burnin)
            mkprob = float(mask.sum()) / N
            # logging.info(f"{unique_rows[idx]}\t\t{mkprob}")
            topmodelprobs[idx] = mkprob

        sortidx = torch.argsort(topmodelprobs, descending=True)

        topmodelprobs = topmodelprobs[sortidx]
        mk_unique_identifiers = unique_rows[sortidx]

        if plot_bivariates:
            from vti.utils.plots import plot_fit_marginals

            plot_fit_marginals(grouped_theta_samples[0], grouped_theta_samples[1:], saveto=save_bivariates_path, title="RJMCMC posterior misspecified robust variable selection")

        return mk_unique_identifiers, topmodelprobs

    def get_true_log_probs_for_mk_identifiers(
        self, mk_samples: torch.Tensor, force_refresh=False
    ) -> torch.Tensor:
        assert hasattr(self, "true_mk_identifiers"), "RJMCMC not run"
        assert hasattr(self, "true_mk_probs"), "RJMCMC not run"

        # Initialize the probs tensor with zeros and the same length as mk_samples.shape[0]
        probs = torch.zeros(mk_samples.shape[0], device=self.device, dtype=self.dtype)

        # Iterate over each sample
        for i, sample in enumerate(mk_samples):
            # Check if the sample matches any row in true_mk_identifiers
            for j, true_id in enumerate(self.true_mk_identifiers):
                # If the current row matches, update the probability
                if torch.equal(sample, true_id):
                    probs[i] = self.true_mk_probs[j]
                    break  # Stop checking once a match is found to avoid unnecessary computation

        return probs

    def _generate(
        self,
        seed,
        num_samples,
        dimension,
        sparsity=0.4,
        correlation_proportion=0.4,
        correlation_factor=0.0,
        outlier_fraction=0.1,
        sigma1=1.0,
        sigma2=10.0,
        sigmax=1,
        sigmabeta1=0.1,
        sigmabeta2=0.5,
        sigmabeta_split=0.5,
        beta1 = 1.,
        beta2 = 2.5,
    ):
        oldrngstate = set_seed(seed)

        # Generate X
        X = torch.randn(num_samples, dimension) * sigmax

        X[:, 0] = 1  # y-intercept

        # Generate true beta coefficients
        num_relevant = int(dimension * sparsity)  # Number of relevant predictors
        logging.info(f"num relevant {num_relevant}")
        assert (
            num_relevant > 1
        ), "Require number of included variables to be >1, inclusive of y-intercept"
        selectable_indices = (
            torch.randperm(dimension - 1) + 1
        )  # Randomly pick indices for non-zero coefficients
        relevant_indices = [0] + selectable_indices[: num_relevant - 1].tolist()
        logging.info(f"included covariates {relevant_indices}")
        relevant_indices = torch.tensor(relevant_indices)

        # create the dgp mk identifier
        dgp_incl_cols = torch.zeros(dimension, dtype=self.dtype)
        dgp_incl_cols[relevant_indices] = 1
        self.dgp_mk_identifier = dgp_incl_cols[1:]

        all_indices = torch.arange(dimension)
        mask = torch.ones(dimension, dtype=torch.bool)
        mask[relevant_indices] = False
        remaining_indices = all_indices[
            mask
        ]  # This gives the indices not in relevant_indices

        # Introduce correlation among predictors
        num_correlated = int(
            dimension * correlation_proportion
        )  # Number of correlated predictors

        # Select correlated indices from remaining indices
        perm = torch.randperm(remaining_indices.size(0))
        correlated_indices = remaining_indices[perm[:num_correlated]]
        logging.info(f"correlated idx {correlated_indices}")

        for idx in correlated_indices:
            X[:, idx] = X[:, idx] * (1 - correlation_factor)
            perm = torch.randperm(len(relevant_indices))
            ref_idxs = relevant_indices[perm[:2]]

            coeffs = generate_coefficients(correlation_factor, 2)
            logging.info(f"Robust VS correlation coeffs for idx {idx} are {coeffs}")
            X[:, idx] += X[:, ref_idxs[0]] * coeffs[0] * generate_random_signs(
                (1,)
            ) + X[:, ref_idxs[1]] * coeffs[1] * generate_random_signs((1,))

        if False:
            beta_true = torch.zeros(dimension)
            beta_true[relevant_indices] = (
                2 + generate_random_signs((num_relevant,)) +  
                (sigmabeta * torch.randn(num_relevant))
            )
            # Generate Y without outliers
            Y = X @ beta_true + torch.randn(num_samples) * sigma1

        else:
            assert sigmabeta_split <= 1 and sigmabeta_split >= 0, "sigmabeta_split must be in [0,1]"
            # split response into two separate beta values
            splitpoint = int(num_samples * sigmabeta_split)
            Y = torch.zeros(num_samples, dtype=self.dtype)
            # Generate Y without outliers
            beta_true1 = torch.zeros(dimension)
            beta_true1[relevant_indices] = beta1 + (sigmabeta1 * torch.randn(num_relevant))
            Y[:splitpoint] = X[:splitpoint] @ beta_true1 + torch.randn(splitpoint) * sigma1

            beta_true2 = torch.zeros(dimension)
            beta_true2[relevant_indices] = beta2 + (sigmabeta2 * torch.randn(num_relevant))
            Y[splitpoint:] = X[splitpoint:] @ beta_true2 + torch.randn(num_samples-splitpoint) * sigma1


            

        # Introduce outliers
        num_outliers = int(num_samples * outlier_fraction)
        outlier_indices = torch.randperm(num_samples)[:num_outliers]
        Y[outlier_indices] += (
            torch.randn(num_outliers) * sigma2
        )  # Adding larger noise for outliers

        restore_seed(oldrngstate)

        return X, Y

    def num_categories(self):
        return 2**self.data_dimension

    def num_inputs(self):
        return self.data_dimension + 1  # + y-intercept

    def num_context_features(self):
        if self.one_hot_encode_context:
            return self.num_categories()
        else:
            return self.data_dimension

    def mk_context_transform(self, mk_samples: torch.Tensor) -> torch.Tensor:
        """
        This method converts draws from the model sampler (mk dist) to a format
        that the flow context wants to see. By default, it is passthrough.

        This method is passed to the constructor of the normalizing flow.
        It is used to project the mk_samples onto a different support
        """
        if self.one_hot_encode_context:
            return F.one_hot(self.mk_identifier_to_cat(mk_samples).long(), num_classes=self.num_categories()).to(dtype=self.dtype)
        else:
            return mk_samples

    # def printVTIResults(self,mk_probs):
    #    logging.info(f"MK probs:  {mk_probs}")

    def printVTIResults(self, mk_probs):
        logging.info("sorted model probabilities by target")
        sortidx = torch.argsort(self.modelprobs, dim=0)
        # sortidx = torch.arange(self.num_categories())
        mkid = self.mk_identifiers()
        mkcomp = torch.column_stack(
            [
                mkid.sum(dim=-1),
                self.modelprobs,
                mk_probs,
            ]
        )
        logging.info("id\tdim\tRJMCMC weight\tpredicted weight")
        # logging.info("id\tdim\tpredicted weight")
        for i in sortidx:
            # logging.info("{}\t{}\t{}\t{}".format(
            logging.info(
                f"{mkid[i]}\t{1 + int(mkcomp[i, 0])}\t{mkcomp[i, 1]}\t{mkcomp[i, 2]}",
            )
        # Below will probably NaN because we'll have zero-prob models
        # logging.info("KLD predicted to target model probabilities = ",
        #        kld(mk_probs.log(),self.modelprobs.log()))

    def mk_prior_dist(self):
        """
        For VTI, the implementation of the model prior requires only
        a log_prob() evaluation method.
        """
        return UniformBinaryString(
            num_bits=self.data_dimension,
            device=self.device,
            dtype=self.dtype,
        )

    def mk_identifiers(self):
        #context_dim = self.num_context_features()
        context_dim =  self.data_dimension
        return torch.tensor(
            [
                list(map(int, format(i, f"0{context_dim}b")))
                for i in range(self.num_categories())
            ],
            dtype=self.dtype,
            device=self.device,
        )

    def mk_cat_to_identifier(self, cat_samples):
        return integers_to_binary(cat_samples, self.data_dimension, self.dtype)

    def mk_identifier_to_cat(self, mk_samples):
        pow2 = torch.pow(2, torch.arange(self.data_dimension - 1, -1, -1)).view(1, -1)
        return (ensure_2d(mk_samples) * pow2).sum(dim=1)

    def mk_to_mask(self, mk):
        return torch.column_stack([torch.ones(mk.shape[:-1]), mk])

    def reference_log_prob(self, mk, theta):
        original_shape = theta.shape
        static_prob = (
            (
                (1 - self.mk_to_mask(mk))
                * self.referencedist.log_prob(theta.reshape(-1, 1)).view(original_shape)
                #(1 - self.mk_to_mask(mk))
                #* self.referencedist.log_prob(theta.view(-1, 1)).view(original_shape)
            )
            .sum(dim=-1)
            .exp()
        )
        return static_prob.log()

    # def log_prob(self, mk, theta):
    #    """
    #    target <- function(x){
    #      p <- length(x)
    #      a <- X%*%x
    #      mn <- exp(-(y - a)^2/2) + exp(-(y - a)^2/200)/10 # Normal mix part
    #      phi_0 <- log(mn)   ## Log likelihood
    #      log_q <- sum(phi_0)  + sum(x^2/200)  ## Add a N(0,10) prior
    #      return(list(log_q = log_q))
    #    }
    #    """
    #    betas = theta
    #    gammas = self.mk_to_mask(mk)
    #    prior_dist = dist.Normal(0, 10)
    #    betas_active = betas * gammas * 3
    #    a = betas_active @ self.x_data.T
    #    log_like = torch.log(
    #        (1 - self.outlier_fraction)
    #        * torch.exp(-((self.y_data - a) ** 2) / (2 * self.sigma1**2))
    #        + self.outlier_fraction
    #        * torch.exp(-((self.y_data - a) ** 2) / (2 * self.sigma2**2))
    #    ).sum(dim=-1)
    #    log_prior = (gammas * prior_dist.log_prob(betas)).sum(dim=-1)
    #    reference_log_prob = self.reference_log_prob(mk, theta)
    #    return log_like + log_prior + reference_log_prob

    def _param_prior_dist(self):
        return dist.Normal(0., self.param_prior_scale)

    def log_prob(self, mk, theta):
        betas = theta * self.ROBUST_VS_THETA_SCALE
        gammas = self.mk_to_mask(mk)
        prior_dist = self._param_prior_dist()
        betas_active = betas * gammas
        a = betas_active @ self.x_data.T

        # Clamp sigmas to ensure numerical stability
        # safe_sigma1 = torch.clamp(self.sigma1, min=1e-10)
        # safe_sigma2 = torch.clamp(self.sigma2, min=1e-10)
        safe_sigma1 = self.sigma1
        safe_sigma2 = self.sigma2

        # Compute the exponent terms for the mixture model
        # u1 and u2 represent the exponent arguments for each component in the mixture
        u1 = -((self.y_data - a) ** 2) / (2 * safe_sigma1**2)
        u2 = -((self.y_data - a) ** 2) / (2 * safe_sigma2**2)

        # Use a log-sum-exp trick to safely compute log of mixture:
        # log((1 - f)*exp(u1) + f*exp(u2)) = max(u1,u2) + log((1-f)*exp(u1 - max) + f*exp(u2 - max))
        m = torch.max(u1, u2)
        mix = (1 - self.outlier_fraction) * torch.exp(
            u1 - m
        ) + self.outlier_fraction * torch.exp(u2 - m)
        # Add a tiny constant to avoid log(0)
        log_like_per_obs = m + torch.log(mix + 1e-15)

        # Sum over the specified dimension
        log_like = log_like_per_obs.sum(dim=-1)

        # Compute the prior term
        log_prior = (gammas * prior_dist.log_prob(betas)).sum(dim=-1)

        # Compute the reference log probability
        reference_log_prob = self.reference_log_prob(mk, theta)

        # Total log probability
        return log_like + log_prior + reference_log_prob

    def sample_RJMCMC_naive(self, num_iterations):
        """
        Perform RJMCMC sampling over the model and parameter space.

        Args:
            num_iterations (int): Number of MCMC iterations.

        Returns:
            mk_samples (torch.Tensor): Tensor of model identifiers sampled.
            theta_samples (torch.Tensor): Tensor of parameter samples.
        """

        # Initialize mk (model indicator) and theta (parameters)
        mk = torch.randint(
            0, 2, (self.data_dimension,), dtype=self.dtype, device=self.device
        )
        mk_samples = []
        theta_samples = []

        #acceptance probs
        ap_mk = []
        ap_theta = []

        # Initialize betas (including intercept)
        dimension = self.num_inputs()
        # betas = torch.zeros(dimension, dtype=self.dtype, device=self.device)
        betas = torch.randn(dimension, dtype=self.dtype, device=self.device)
        # Initialize betas where gamma=1 from the prior
        gammas = self.mk_to_mask(mk.reshape(1, -1)).squeeze()
        prior_dist = self._param_prior_dist()
        betas_active = prior_dist.sample([dimension]).to(self.device) * gammas

        for iteration in range(num_iterations):
            # RJMCMC Step: Propose new mk by flipping one bit
            mk_new = mk.clone()
            flip_index = torch.randint(0, self.data_dimension, (1,))
            mk_new[flip_index] = 1 - mk_new[flip_index]  # Flip the bit

            # Compute acceptance probability
            # Since betas remain the same, we need to adjust betas for new mk
            # For simplicity, we can set betas corresponding to new variables to prior mean (zero)
            gammas_new = self.mk_to_mask(mk_new.reshape(1, -1)).squeeze()
            betas_new = betas.clone()
            if False:
                # DON'T DO THIS. We get better acceptance rates by just bit flipping.
                # For variables turned on, sample new beta from prior
                turned_on = ((gammas_new == 1) & (gammas == 0)).squeeze()
                betas_new[turned_on] = prior_dist.sample([turned_on.sum()]).to(
                    self.device
                )
                # For variables turned off, set beta to zero
                turned_off = ((gammas_new == 0) & (gammas == 1)).squeeze()
                betas_new[turned_off] = torch.tensor(
                    0.0, dtype=self.dtype, device=self.device
                )

            # Compute log posterior probabilities
            log_prob_current = self.log_prob(mk.reshape(1, -1), betas.reshape(1, -1))
            log_prob_new = self.log_prob(
                mk_new.reshape(1, -1), betas_new.reshape(1, -1)
            )

            # Compute acceptance probability
            acceptance_log_prob = log_prob_new - log_prob_current
            acceptance_prob_mk = torch.exp(acceptance_log_prob).clamp(max=1.0)

            # logging.info(f"RJMCMC acc prob {acceptance_prob.item(}"))
            # Decide whether to accept
            if torch.rand(1).item() < acceptance_prob_mk.item():
                mk = mk_new
                betas = betas_new
                gammas = gammas_new

            # Store the samples
            mk_samples.append(mk.clone())
            theta_samples.append(betas.clone())

            # Now do a naive gaussian mcmc proposal
            # betas_new = betas.clone() + torch.randn(betas.shape[0]) * self.mk_to_mask(mk.reshape(1,-1)).reshape(-1)
            #betas_new = betas.clone() + 0.2 * torch.randn(betas.shape[0]) * gammas
            #betas_new = betas.clone() + 0.5 * torch.randn(betas.shape[0]) * gammas
            betas_new = betas.clone() + 0.75 * torch.randn(betas.shape[0]) * gammas
            # logging.info(f"betas, betas new, mk {betas}, {betas_new}, {mk}")
            log_prob_current = self.log_prob(mk.reshape(1, -1), betas.reshape(1, -1))
            log_prob_new = self.log_prob(mk.reshape(1, -1), betas_new.reshape(1, -1))
            acceptance_log_prob = log_prob_new - log_prob_current
            acceptance_prob_theta = torch.exp(acceptance_log_prob).clamp(max=1.0)
            # logging.info(f"acceptance prob within model {acceptance_prob}")

            # Decide whether to accept
            if torch.rand(1).item() < acceptance_prob_theta.item():
                betas = betas_new

            # Store the samples
            mk_samples.append(mk.clone())
            theta_samples.append(betas.clone())

        # Convert samples to tensors
        mk_samples = torch.stack(mk_samples)
        theta_samples = torch.stack(theta_samples)

        return mk_samples, theta_samples

    @torch.no_grad()
    def sample_RJMCMC_vectorised(
        self, num_iterations, burn_in, num_runs, thinning=100
    ):
        """
        Perform batched RJMCMC sampling over multiple runs.

        Args:
            num_iterations (int): Number of MCMC iterations per run.
            num_runs (int): Number of parallel RJMCMC chains.

        Returns:
            mk_samples (torch.Tensor): Tensor of shape [num_runs, num_iterations * 2, data_dimension].
            theta_samples (torch.Tensor): Tensor of shape [num_runs, num_iterations * 2, data_dimension].
        """
        # Initialize mk for all runs: [num_runs, data_dimension]
        mk = torch.randint(
            #0, 2, (num_runs, self.data_dimension), dtype=self.dtype, device=self.device
            0, 2, (num_runs, self.data_dimension), dtype=torch.int32, device=self.device
        )

        # Initialize betas for all runs: [num_runs, num_inputs]
        dimension = self.num_inputs()
        #betas = torch.randn(num_runs, dimension, dtype=self.dtype, device=self.device)
        # Initialize betas for all runs: [num_runs, num_inputs]
        # We must draw active betas from the prior and inactive ones from the reference distribution.
        # Define prior distribution (e.g., N(0, 10))
        prior_dist = self._param_prior_dist()
        # Define reference distribution
        # HACK swap out reference dist for the prior, then copy it back at the end
        old_ref_dist = self.referencedist
        self.referencedist = prior_dist
        ref_dist = self.referencedist
        # HACK change the scale, then scale at end
        OLD_ROBUST_VS_THETA_SCALE = self.ROBUST_VS_THETA_SCALE
        self.ROBUST_VS_THETA_SCALE = 1.
        # Create the mask: first column always active, others according to mk.
        mask = self.mk_to_mask(mk.to(dtype=self.dtype))
        # Sample betas: active variables from prior, inactive from reference.
        betas_prior = prior_dist.sample((num_runs, dimension)) #/ ROBUST_VS_THETA_SCALE
        betas_ref = ref_dist.sample((num_runs, dimension))
        betas = mask * betas_prior + (1 - mask) * betas_ref

        NUM_WITHIN_MCMC=5

        WITHIN_MCMC_SCALE=0.3 #/ROBUST_VS_THETA_SCALE

        # compute the jacobian for transforming from reference dist to prior dist
        # we do a bit flip without any transformation, so this should be 1. But this may cause issues,
        # so we do a transformation from the auxiliary (reference) distribution to the current target.
        # we can get the scale of prior_dist and ref_dist via prior_dist.scale and ref_dist.scale respectively.
        #LOG_JACOBIAN_FLIP_ON = 0. # work this out
        #SCALE_REF_TO_TARGET = 1. # work this out.

        # Preallocate tensors to store samples
        mk_samples = torch.zeros(
            num_runs,
            # num_iterations * 2,
            (num_iterations - burn_in) // thinning,
            self.data_dimension,
            dtype=self.dtype,
            device=self.device,
        )
        theta_samples = torch.zeros(
            num_runs,
            # num_iterations * 2,
            (num_iterations - burn_in) // thinning,
            dimension,
            dtype=self.dtype,
            device=self.device,
        )
        ar_mk = torch.zeros(
            num_runs,
            (num_iterations - burn_in),
            dtype=self.dtype,
            device=self.device,
        )
        ar_theta = torch.zeros(
            num_runs,
            (num_iterations - burn_in)*NUM_WITHIN_MCMC,
            dtype=self.dtype,
            device=self.device,
        )



        for iteration in range(num_iterations):
            # === RJMCMC Step: Propose new mk by flipping one bit per run ===
            flip_indices = torch.randint(
                low=0, high=self.data_dimension, size=(num_runs,), device=self.device
            )

            # Create flip_mask using one-hot encoding
            flip_mask = torch.zeros_like(mk)
            flip_mask.scatter_(1, flip_indices.unsqueeze(1), 1)

            # Propose new mk by flipping the selected bit
            mk_new = (mk + flip_mask) % 2  # Equivalent to flipping

            # Compute log probabilities
            log_prob_current = self.log_prob(mk.to(dtype=self.dtype), betas)  # [num_runs]
            log_prob_new = self.log_prob(mk_new.to(dtype=self.dtype), betas)  # [num_runs]

            # Compute acceptance probabilities
            acceptance_log_prob = log_prob_new - log_prob_current
            acceptance_prob_mk = torch.exp(acceptance_log_prob).clamp(
                max=1.0
            )  # [num_runs]

            # Generate uniform random numbers for acceptance
            uniform_randoms = torch.rand(num_runs, device=self.device, dtype=self.dtype)

            # Determine which proposals to accept
            accept_mask = (uniform_randoms <= acceptance_prob_mk).to(
                dtype=torch.int32
                #dtype=self.dtype
            )  # [num_runs]
            accept_mask = accept_mask.unsqueeze(1)  # [num_runs, 1]

            # Update mk where proposals are accepted
            mk = accept_mask * mk_new + (1 - accept_mask) * mk

            if iteration > burn_in:
                ar_mk[:, (iteration - burn_in)] = acceptance_prob_mk

            # if False:
            #    # Store the samples
            #    mk_samples[:, iteration * 2] = mk
            #    theta_samples[:, iteration * 2] = betas

            # === Gaussian MCMC Proposal for Betas ===
            for iwm in range(NUM_WITHIN_MCMC):
                #betas_proposal = betas + 0.3 * torch.randn_like(betas) * self.mk_to_mask(mk)
                if False:
                    # full Gaussian proposal
                    betas_proposal = betas + 0.2 * torch.randn_like(betas) * self.mk_to_mask(mk.to(dtype=self.dtype))
                else:
                    # Compute the mask: shape [num_runs, dimension]
                    mask = self.mk_to_mask(mk.to(dtype=self.dtype))
                    # For each run, randomly select one active index using multinomial sampling
                    selected_idx = torch.multinomial(mask, num_samples=1)  # shape: [num_runs, 1]
                    # Generate a random update for each run (only one value per run)
                    update_values = WITHIN_MCMC_SCALE*torch.randn(num_runs, 1, dtype=self.dtype, device=self.device)
                    # Build an update tensor of zeros and scatter the update_values at the selected indices
                    update = torch.zeros_like(betas)
                    update.scatter_(1, selected_idx, update_values)
                    # Propose the new betas by adding the sparse update
                    betas_proposal = betas + update

                # Compute log probabilities
                log_prob_current = self.log_prob(mk.to(dtype=self.dtype), betas)
                log_prob_new = self.log_prob(mk.to(dtype=self.dtype), betas_proposal)

                # Compute acceptance probabilities
                acceptance_log_prob = log_prob_new - log_prob_current
                acceptance_prob_theta = torch.exp(acceptance_log_prob).clamp(max=1.0)

                # Generate uniform random numbers for acceptance
                uniform_randoms = torch.rand(num_runs, device=self.device, dtype=self.dtype)

                # Determine which proposals to accept
                accept_mask = (uniform_randoms <= acceptance_prob_theta).to(
                    dtype=self.dtype
                )  # [num_runs]
                accept_mask = accept_mask.unsqueeze(1)  # [num_runs, 1]

                # Update betas where proposals are accepted
                betas = accept_mask * betas_proposal + (1 - accept_mask) * betas

                if iteration > burn_in:
                    ar_theta[:, (iteration - burn_in)*NUM_WITHIN_MCMC+iwm] = acceptance_prob_theta

            if iteration > burn_in and iteration % thinning == 0:
                # Store the samples
                mk_samples[:, (iteration - burn_in) // thinning] = mk.to(dtype=self.dtype)
                theta_samples[:, (iteration - burn_in) // thinning] = betas / OLD_ROBUST_VS_THETA_SCALE
                #ar_mk[:, (iteration - burn_in) // thinning] = acceptance_prob_mk
                #ar_theta[:, (iteration - burn_in) // thinning] = acceptance_prob_theta

            if iteration % 100 == 0:
                logging.info(f"RJMCMC step {iteration}")


            #if iteration > burn_in:
            #    ar_mk[:, (iteration - burn_in)] = acceptance_prob_mk
            #    ar_theta[:, (iteration - burn_in)] = acceptance_prob_theta

        logging.info(f"Acceptance prob mk {ar_mk.mean()}")
        logging.info(f"Acceptance prob theta {ar_theta.mean()}")

        # END HACK copy back the reference dist
        self.referencedist = old_ref_dist
        self.ROBUST_VS_THETA_SCALE = OLD_ROBUST_VS_THETA_SCALE

        return mk_samples, theta_samples

    def construct_param_transform(
        self, flow_type: str = "diagnorm"
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Build a param transform function for flows. Typically used to constrain
        or reparameterize parameters. The default uses transforms from 'param_transform_factory'.

        DIFFERS FROM PARENT CLASS: if self.dimension > 15, we use a context encoder and don't use one hot encoding.

        Args:
            flow_type (str): The type of flow or parameter transform to create.

        Returns:
            A param_transform object that child classes can use for reparam.
        """
        logging.info("Constructing param transform...")

        # A closure for mapping context -> param mask and its reverse.
        context_to_mask = lambda context: self.mk_to_mask(context)
        #context_to_mask_reverse = lambda context: torch.fliplr(self.mk_to_mask(context))

        context_transform = lambda context : self.mk_context_transform(context)

        if self.one_hot_encode_context:
            lce = None
        else:
            lce = [upper_bound_power_of_2(i*self.num_context_features()) for i in [2,4,16,64]]

        # --------------------
        # 1. Default config for construct_param_transform
        # --------------------
        default_config = {
            "num_pre_pmaf_layers": 0,
            "num_prqs_layers": 0,
            "num_pmaf_layers": 0,
            "num_pmaf_hidden_features": upper_bound_power_of_2(2*self.num_inputs()), # offers slight improvement over 2*num_inputs. Not enough
            "num_inputs": self.num_inputs(),
            "num_context_inputs": self.num_context_features(),
            "context_to_mask": context_to_mask,
            #"context_to_mask_reverse": context_to_mask_reverse,
            "context_transform": context_transform,
            "num_pqrs_hidden_features": upper_bound_power_of_2(40*self.num_inputs()), # depends on num bins
            #"num_pqrs_hidden_features": upper_bound_power_of_2(20*self.num_inputs()), # depends on num bins
            "num_pqrs_bins": 10,
            "num_pqrs_blocks": 2,
            "use_diag_affine_start": False,
            "use_diag_affine_end": False,
            "diag_affine_context_encoder_depth": 5,
            "diag_affine_context_encoder_hidden_features":upper_bound_power_of_2(2*self.num_inputs()),
            "learnable_context_encoder_arch":lce,
        }

        # --------------------
        # 2. Config overrides for each flow type
        # --------------------
        flow_configs = {
            "diagnorm": None,  # special-case (uses construct_diagnorm_param_transform)
            "diagnorm_deep": None,  # special-case (uses construct_diagnorm_param_transform)
            "diagnorm10128":None,
            "diagnorm40": None,  # special-case (uses construct_diagnorm_param_transform)
            "affine2": {"num_pre_pmaf_layers": 2},
            "affine220": {"num_pre_pmaf_layers": 2,  "num_affine_blocks": 20},
            "affine5": {"num_pre_pmaf_layers": 5},
            "affine52": {"num_pre_pmaf_layers": 5, "num_affine_blocks": 2},
            "affine55": {"num_pre_pmaf_layers": 5, "num_affine_blocks": 5},
            "affine52f128": {"num_pre_pmaf_layers": 5,  "num_affine_blocks": 2, "num_pmaf_hidden_features": 128},
            "affine55f128": {"num_pre_pmaf_layers": 5,  "num_affine_blocks": 5, "num_pmaf_hidden_features": 128},
            "affine7": {"num_pre_pmaf_layers": 7},
            "affine510": {"num_pre_pmaf_layers": 5,  "num_affine_blocks": 10},
            "affine515": {"num_pre_pmaf_layers": 5,  "num_affine_blocks": 15},
            "affine520": {"num_pre_pmaf_layers": 5,  "num_affine_blocks": 20},
            "affine525": {"num_pre_pmaf_layers": 5,  "num_affine_blocks": 25},
            "affine720": {"num_pre_pmaf_layers": 7,  "num_affine_blocks": 20},
            "affine10": {"num_pre_pmaf_layers": 10},
            "affine104": {"num_pre_pmaf_layers": 10, "num_affine_blocks": 4},
            "spline23": {"num_prqs_layers": 2, "num_pqrs_blocks": 3},
            "spline22f128": {"num_prqs_layers": 2, "num_pqrs_blocks": 2, "num_pqrs_hidden_features": 128},
            "spline22": {"num_prqs_layers": 2, "num_pqrs_blocks": 2},
            "spline23f128": {"num_prqs_layers": 2, "num_pqrs_blocks": 3, "num_pqrs_hidden_features": 128},
            "spline46": {"num_prqs_layers": 4, "num_pqrs_blocks": 6},
            "spline46f128": {"num_prqs_layers": 4, "num_pqrs_blocks": 6, "num_pqrs_hidden_features": 128},
            "spline410f128": {"num_prqs_layers": 4, "num_pqrs_blocks": 10, "num_pqrs_hidden_features": 128},
            "spline410": {"num_prqs_layers": 4, "num_pqrs_blocks": 10},
            "spline415": {"num_prqs_layers": 4, "num_pqrs_blocks": 15},
            "spline220": {"num_prqs_layers": 2, "num_pqrs_blocks": 20, "num_pqrs_hidden_features": 128},
            "spline220w": {"num_prqs_layers": 2, "num_pqrs_blocks": 20, "num_pqrs_hidden_features": 256},
            "spline220ww": {"num_prqs_layers": 2, "num_pqrs_blocks": 20, "num_pqrs_hidden_features": 512},
            "spline420": {"num_prqs_layers": 4, "num_pqrs_blocks": 20, "num_pqrs_hidden_features": 128},
            "spline420w": {"num_prqs_layers": 4, "num_pqrs_blocks": 20, "num_pqrs_hidden_features": 256},
            "affine2spline23": {
                "num_pmaf_layers": 2,
                "num_prqs_layers": 2,
                "num_pqrs_blocks": 3,
            },
            "affine2spline43": {
                "num_pmaf_layers": 2,
                "num_prqs_layers": 4,
                "num_pqrs_blocks": 3,
            },
            "diagaffinespline43": {
                "use_diag_affine_start": True,
                "diag_affine_context_encoder_depth": 5,
                "num_prqs_layers": 4,
                "num_pqrs_blocks": 3,
            },
            "diagaffinespline63": {
                "use_diag_affine_start": True,
                "diag_affine_context_encoder_depth": 5,
                "num_prqs_layers": 6,
                "num_pqrs_blocks": 3,
            },
            "spline43diag": {
                "use_diag_affine_end": True,
                "diag_affine_context_encoder_depth": 10,
                "diag_affine_context_encoder_hidden_features": upper_bound_power_of_2(self.num_context_features()),
                "num_prqs_layers": 4,
                "num_pqrs_blocks": 3,
            },
            "spline46diag": {
                "use_diag_affine_end": True,
                "diag_affine_context_encoder_depth": 10,
                "diag_affine_context_encoder_hidden_features": upper_bound_power_of_2(self.num_context_features()),
                "num_prqs_layers": 4,
                "num_pqrs_blocks": 6,
            },
            "spline410diag": {
                "use_diag_affine_end": True,
                "diag_affine_context_encoder_depth": 10,
                "diag_affine_context_encoder_hidden_features": upper_bound_power_of_2(self.num_context_features()),
                "num_prqs_layers": 4,
                "num_pqrs_blocks": 10,
            },
            "diagspline22": {
                "use_diag_affine_start": True,
                #"diag_affine_context_encoder_depth": 10,
                "diag_affine_context_encoder_depth": 0,
                "diag_affine_context_encoder_hidden_features": upper_bound_power_of_2(self.num_context_features()),
                "num_prqs_layers": 2,
                "num_pqrs_blocks": 2,
            },
            "diagspline43": {
                "use_diag_affine_start": True,
                #"diag_affine_context_encoder_depth": 10,
                "diag_affine_context_encoder_depth": 0,
                "diag_affine_context_encoder_hidden_features": upper_bound_power_of_2(self.num_context_features()),
                "num_prqs_layers": 4,
                "num_pqrs_blocks": 3,
            },
            "diagspline46": {
                "use_diag_affine_start": True,
                #"diag_affine_context_encoder_depth": 10,
                "diag_affine_context_encoder_depth": 0,
                "diag_affine_context_encoder_hidden_features": upper_bound_power_of_2(self.num_context_features()),
                "num_prqs_layers": 4,
                "num_pqrs_blocks": 6,
            },
            "diagspline410": {
                "use_diag_affine_start": True,
                #"diag_affine_context_encoder_depth": 10,
                "diag_affine_context_encoder_depth": 0,
                "diag_affine_context_encoder_hidden_features": upper_bound_power_of_2(self.num_context_features()),
                "num_prqs_layers": 4,
                "num_pqrs_blocks": 10,
            },
            # shared diag affine start
            "shareddiagspline22": {
                "use_shared_diag_affine_start": True,
                "num_prqs_layers": 2,
                "num_pqrs_blocks": 2,
            },
            "shareddiagspline43": {
                "use_shared_diag_affine_start": True,
                "num_prqs_layers": 4,
                "num_pqrs_blocks": 3,
            },
            "shareddiagspline46": {
                "use_shared_diag_affine_start": True,
                "num_prqs_layers": 4,
                "num_pqrs_blocks": 6,
            },
            "shareddiagspline410": {
                "use_shared_diag_affine_start": True,
                "num_prqs_layers": 4,
                "num_pqrs_blocks": 10,
            },
        }

        # --------------------
        # 3. Learning rate defaults + overrides
        # --------------------
        # default fallback if not found in the dictionaries below
        default_flow_lr = 1e-3
        default_sfe_lr = 1e-1

        # flow LR overrides
        flow_lr_map = {
            "diagnorm": 1e-1,
            "affine2": 1e-3,
            "affine5": 1e-3,
            "affine7": 1e-2,
            "spline23": 1e-2,
            "affine2spline23": 1e-2,
        }

        # sfe LR overrides
        sfe_lr_map = {
            "diagnorm": 1e-1,
            "affine2": 1e-1,
            "affine5": 1e-4,
            "affine7": 1e-4,
            "spline23": 1e-1,
            "affine2spline23": 1e-1,
        }

        # --------------------
        # 4. Build the actual param transform
        # --------------------
        if flow_type not in flow_configs:
            raise ValueError(f"Unknown flow_type: {flow_type}")

        if flow_type == "diagnorm":
            # Special-case transform
            param_transform = construct_diagnorm_param_transform(
                self.num_inputs(),
                self.num_context_features(),
                context_to_mask,
                #context_to_mask_reverse,
                context_transform,
                context_encoder_depth=5,
                context_encoder_hidden_features=upper_bound_power_of_2(2*self.num_context_features()),
            ).to(self.device)
        elif flow_type == "diagnorm_deep":
            # Special-case transform
            param_transform = construct_diagnorm_param_transform(
                self.num_inputs(),
                self.num_context_features(),
                context_to_mask,
                #context_to_mask_reverse,
                context_transform,
                context_encoder_depth=10,
                context_encoder_hidden_features=upper_bound_power_of_2(2*self.num_context_features()),
            ).to(self.device)
        elif flow_type == "diagnorm10128":
            # Special-case transform
            param_transform = construct_diagnorm_param_transform(
                self.num_inputs(),
                self.num_context_features(),
                context_to_mask,
                #context_to_mask_reverse,
                context_transform,
                context_encoder_depth=10,
                context_encoder_hidden_features=128,
            ).to(self.device)
        elif flow_type == "diagnorm40":
            # Special-case transform
            param_transform = construct_diagnorm_param_transform(
                self.num_inputs(),
                self.num_context_features(),
                context_to_mask,
                #context_to_mask_reverse,
                context_transform,
                context_encoder_depth=40,
                context_encoder_hidden_features=128,
            ).to(self.device)
        elif flow_type in list(flow_configs.keys()):
            # Clone the default config and update with overrides
            config = default_config.copy()
            config.update(flow_configs[flow_type])
            param_transform = construct_param_transform(**config).to(self.device)
        else:
            raise NotImplementedError("Unknown flow type {flow_type}")

        # --------------------
        # 5. Set learning rates using our dictionaries (with defaults)
        # --------------------
        self.flow_lr = flow_lr_map.get(flow_type, default_flow_lr)
        self.sfe_lr = sfe_lr_map.get(flow_type, default_sfe_lr)

        logging.info("...done!")
        return param_transform
