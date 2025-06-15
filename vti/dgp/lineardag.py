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
from vti.utils.jit import *
import numpy as np
#import math
from torch.distributions import Normal
from vti.dgp import AbstractDGP
from vti.utils.seed import set_seed, restore_seed
import vti.utils.logging as logging
from vti.utils.logging import _bm, _dm
from vti.distributions import (
    PermutationDAGUniformDistribution,
    PermutationDAGPenalizedDistribution,
)

# rjmcmc
from vti.utils.dag_helpers import *
from vti.utils.cdt_helpers import *
from vti.utils.math_helpers import ensure_2d, integers_to_binary, upper_bound_power_of_2


from vti.dgp.param_transform_factory import (
    construct_param_transform,
    construct_diagnorm_param_transform,
)
from typing import Optional, Tuple, Any, Callable


def _log_normal_pdf(x, mean=0.0, std=1.0):
    """Compute log of N(x | mean, std^2) on GPU."""
    mean = torch.tensor(mean, device=x.device, dtype=x.dtype)
    std = torch.tensor(std, device=x.device, dtype=x.dtype)
    var = std ** 2
    return -0.5 * torch.log(2 * torch.pi * var) - (x - mean) ** 2 / (2 * var)

def _log_normal_pdf_c(x, mean=0.0, log_const=None, inv2var=None):
    # Here, log_const and inv2var are precomputed constants (for tau or sigma)
    return log_const - (x - mean)**2 * inv2var

class AbstractLinearDAG(AbstractDGP):
    """
    Abstract base class for linear DAG data generation or loading.
    Child classes must implement:
       1) the `_generate()` method (for synthetic data), or
       2) an alternate approach (e.g. loading real data) that sets
          self.x_data, self.true_P, self.true_U, self.true_W,
    and must also handle `test_data_split` logic to create `test_x_data`.
    """

    def __init__(self, test_data_split: float = 0.0, device=None, dtype=None):
        super().__init__(device=device, dtype=dtype)
        # The child class's constructor will call _generate() or set up real data,
        # then call something like `_split_off_test_data(test_data_split)` defined here.
        #
        # We'll store the test split ratio in an attribute, so child classes can use it consistently.
        assert 0.0 <= test_data_split <= 1.0, "test_data_split must be in [0,1]"
        self.test_data_split = test_data_split
        self.chunk_size = 512

        # We'll define placeholders for x_data, test_x_data, true_P, true_U, true_W.
        # The child class is responsible for filling these in.
        self.register_buffer("x_data", None)
        self.register_buffer("test_x_data", None)
        self.register_buffer("true_P", None)
        self.register_buffer("true_U", None)
        self.register_buffer("true_W", None)

    def _register_precompute_buffers(self):
        """
        Call at end of __init__() for each subclass
        """
        # In __init__ of AbstractLinearDAG:
        two_pi = 2 * self.pi
        self.register_buffer('two_pi', torch.tensor(two_pi, device=self.device, dtype=self.dtype))
        self.register_buffer('sigma2', self.sigma ** 2)
        self.register_buffer('inv2sigma2', 1.0 / (2 * self.sigma ** 2))
        self.register_buffer('log_const_sigma', -0.5 * torch.log(self.two_pi * self.sigma2))
        self.register_buffer('tau2', self.tau ** 2)
        self.register_buffer('inv2tau2', 1.0 / (2 * self.tau ** 2))
        self.register_buffer('log_const_tau', -0.5 * torch.log(self.two_pi * self.tau2))
        

    def _split_off_test_data(self):
        """
        After self.x_data has been set, partition it into x_data (train) and test_x_data
        using proportion `self.test_data_split`.
        This is common logic for all child classes.
        """
        if self.test_data_split <= 0 or self.test_data_split >= 1:
            # If it's 0 or 1, we skip
            return

        n = self.x_data.shape[0]
        d = self.x_data.shape[1]
        n_test = int(n * self.test_data_split)

        # We'll pick n_test indices at random for the test set
        perm = torch.randperm(n, device=self.device)
        test_indices = perm[:n_test]
        train_indices = perm[n_test:]

        test_data = self.x_data[test_indices]
        train_data = self.x_data[train_indices]

        self.register_buffer("x_data", train_data)
        self.register_buffer("test_x_data", test_data)

    def amortized_variational_posterior_predictive_check(self, q, K=None): #self.num2tensor(1000)):
        """
        Perform an amortized variational posterior predictive distribution check
        using the variational posterior q.
        We want to evaluate how well q explains the 'test_x_data'.

        For each data sample in the test set, we approximate p(x | posterior).
             Because we have a transdimensional flow q, we can sample multiple
             (mk, theta) from q, compute the likelihood of x, and average.

        Return Monte Carlo estimate of posterior predictive check on the test data set.
        """
        assert hasattr(self, "test_x_data"), "Missing test_x_data for AVPPC"
        K = K if K is not None else self.num2tensor(1000)
        #total_ll = 0.0
        mk_sample, theta_sample = q._sample(K)
        ll = self.log_prob(mk_sample, theta_sample, use_test_data=True)
        #avg_ll = ll.log_sum_exp(dim=0) - math.log(K)
        avg_ll = ll.log_sum_exp(dim=0) - K.log()
        # Return average log-likelihood or negative log-likelihood, as you prefer
        return -avg_ll  # negative log-likelihood as the "score"

    def get_sfe_lr(self):
        # return 1e-3
        # TODO tune these
        if self.num_nodes == 1:
            return 1e-1
        if self.num_nodes == 2:
            return 1e-2
        if self.num_nodes == 3:
            return 1e-1 #5e-3
        elif self.num_nodes == 4:
            return 5e-2 #5e-3
        elif self.num_nodes == 5:
            return 1e-2 #5e-3
        elif self.num_nodes == 6:
            return 5e-3 #5e-3
        elif self.num_nodes <= 8:
            return 1e-3 #5e-3
        elif self.num_nodes < 10:
            return 5e-4  # 5e-5
        elif self.num_nodes == 10:
            return 1e-4  # 5e-5
        else:
            return 1e-5  # 1e-5

    def num_dag_nodes(self):
        return self.num_nodes

    # custom flow constructor that uses number of nodes to configure flow expressiveness

    def construct_param_transform(
        self, flow_type: str = "diagnorm"
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Build a param transform function for flows. Typically used to constrain
        or reparameterize parameters. The default uses transforms from 'param_transform_factory'.

        Args:
            flow_type (str): The type of flow or parameter transform to create.

        Returns:
            A param_transform object that child classes can use for reparam.
        """
        logging.info("Constructing param transform for DAG...")

        # A closure for mapping context -> param mask and its reverse.
        context_to_mask = lambda context: self.mk_to_mask(context)
        #context_to_mask_reverse = lambda context: torch.fliplr(self.mk_to_mask(context))

        context_transform = lambda context : self.mk_context_transform(context)

        # we always use a learnable context encoder for DAGs
        # this should be tuned
        #lce = [upper_bound_power_of_2(i*self.num_context_features()) for i in [2,4,32]]
        lce = []
        lce_max_width = 4096
        lce_hidden_dims = upper_bound_power_of_2(2*self.num_context_features())
        while lce_hidden_dims < lce_max_width:
            lce.append(lce_hidden_dims)
            lce_hidden_dims = lce_hidden_dims * 4

        lce.append(lce_max_width)
        logging.info(f"Learnable context encoder for DAG inference has hidden dims {lce}")


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
            "num_pqrs_hidden_features": upper_bound_power_of_2(4*self.num_inputs()),
            "num_pqrs_bins": 10,
            "num_pqrs_blocks": 0,
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
            "diagnorm40": None,  # special-case (uses construct_diagnorm_param_transform)
            "affine2": {"num_pre_pmaf_layers": 2},
            "affine220": {"num_pre_pmaf_layers": 2,  "num_affine_blocks": 20},
            "affine5": {"num_pre_pmaf_layers": 5},
            "affine55": {"num_pre_pmaf_layers": 5, "num_affine_blocks": 5},
            "affine7": {"num_pre_pmaf_layers": 7},
            "affine510": {"num_pre_pmaf_layers": 5,  "num_affine_blocks": 10},
            "affine515": {"num_pre_pmaf_layers": 5,  "num_affine_blocks": 15},
            "affine520": {"num_pre_pmaf_layers": 5,  "num_affine_blocks": 20},
            "affine525": {"num_pre_pmaf_layers": 5,  "num_affine_blocks": 25},
            "affine720": {"num_pre_pmaf_layers": 7,  "num_affine_blocks": 20},
            "affine10": {"num_pre_pmaf_layers": 10},
            "affine104": {"num_pre_pmaf_layers": 10, "num_affine_blocks": 4},
            "spline23": {"num_prqs_layers": 2, "num_pqrs_blocks": 3},
            "spline46": {"num_prqs_layers": 4, "num_pqrs_blocks": 6},
            "spline410": {"num_prqs_layers": 4, "num_pqrs_blocks": 10},
            "spline415": {"num_prqs_layers": 4, "num_pqrs_blocks": 15},
            "spline220": {"num_prqs_layers": 2, "num_pqrs_blocks": 20},
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
                context_encoder_hidden_features=2*self.num_context_features(),
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
            logging.info(f"Constructed flow composition with config {config}")
        else:
            raise NotImplementedError("Unknown flow type {flow_type}")

        # --------------------
        # 5. Set learning rates using our dictionaries (with defaults)
        # --------------------
        self.flow_lr = flow_lr_map.get(flow_type, default_flow_lr)
        self.sfe_lr = sfe_lr_map.get(flow_type, default_sfe_lr)

        logging.info("...done!")
        return param_transform

    def mk_prior_dist(self):
        """
        A fast implementation for the prior for
        the permutation upper-triangular representation
        """
        return PermutationDAGUniformDistribution(
            num_nodes=self.num_nodes, device=self.device, dtype=self.dtype
        )

    def num_categories(self):
        raise Exception("Unable to calculate number of categories for DAG")
        # return 2**self.data_dimension

    def num_inputs(self):
        # return self.num_nodes
        return self.U_features

    def num_context_features(self):
        # print("num context features",self.P_features + self.U_features)
        #return self.P_features + self.U_features
        return self.num_nodes**2

    def printVTIResults(self, mk_probs):
        logging.info(f"MK probs: {mk_probs}")

    def mk_identifiers(self):
        raise Exception("Cannot enumerate mk identifiers for DAGs")

    def mk_cat_to_identifier(self, cat_samples):
        raise Exception("Cannot convert category to mk identifier for DAGs")

    def mk_identifier_to_cat(self, mk_samples):
        raise Exception("Cannot convert mk identifier to category for DAGs")

    def mk_to_mask(self, mk):
        # obtain U binary string
        return mk[:, self.P_features :].view(mk.shape[0], self.U_features)

    def mk_context_transform(self, mk_samples: torch.Tensor) -> torch.Tensor:
        """
        This method converts draws from the model sampler (mk dist) to a format
        that the flow context wants to see. By default, it is passthrough.

        This method is passed to the constructor of the normalizing flow.
        It is used to project the mk_samples onto hopefully a smaller support
        e.g. the P,U representation of a DAG breaks the bijective equivalence to 
        the categorical, but if we transform via A=P U P^T, we get bijective
        equivalence and the flow does not have to work as hard.
        """
        A3d = convert_mk_to_adjacency(mk_samples, self.num_nodes)
        return A3d.reshape(mk_samples.shape[0], -1)

    def reference_log_prob(self, mk, theta):
        original_shape = theta.shape
        static_log_prob = (
            (1 - self.mk_to_mask(mk))
            * self.referencedist.log_prob(theta.view(-1, 1)).view(original_shape)
        ).sum(dim=-1)
        return static_log_prob

    def load_data(self, seed, d, n, data_file="data.pt"):
        """
        Load the data from disk. This assumes the same seed, d, n were used.
        """
        data = torch.load(data_file)
        assert data["seed"] == seed and data["d"] == d and data["n"] == n
        return data["X"], data["P"], data["U"], data["W"]

    def _num_edges(self, d):
        """
        d = num nodes
        """
        return d * (d - 1) // 2

    def log_prob(self, mk, theta, use_test_data=False):
        if ADJFORMPUPT:
            return self.log_prob_PUPT(mk, theta, use_test_data)
        else:
            return self.log_prob_PTUP(mk, theta, use_test_data)
    

    def log_prob_PTUP(self, mk, theta, use_test_data=False):
        """
        A=P^T * U * P
        """
        PARAM_SCALE = 1.0
        chunk_size = self.chunk_size
        # 1. Select the dataset
        X_data = self.test_x_data if use_test_data else self.x_data
    
        # 2. Parse model representation (all remain on GPU)
        P_cat = mk[:, :self.P_features]
        U_bin = mk[:, self.P_features:]
        W_val = theta*PARAM_SCALE
    
        # 3. Use constant sigma and tau from registered buffers
        sigma = self.sigma  # used for clarity; precomputed sigma2 etc. are available
    
        batch_size = P_cat.shape[0]
        d = P_cat.shape[1] + 1
    
        # 4. Convert categorical representation to permutation matrix (fully vectorized)
        #    Here, P has shape: (batch_size, d, d)
        P = cat_representation_to_perm_matrix_onehot_mask(P_cat)
    
        # 5. Build full U and W matrices (upper triangular, strictly so)
        U_full = build_full_matrix_triu_indices(batch_size, d, U_bin)
        W_full = build_full_matrix_triu_indices(batch_size, d, W_val)
    
        # 6. Combine to form U in the sorted space (edge-weights multiplied elementwise)
        #    That is, M is our U.
        M = W_full * U_full
    
        # 7. Compute reference log probability (implementation is GPU only)
        reference_log_prob = self.reference_log_prob(mk, theta)
    
        # 8. Compute log prior for included edges using precomputed constants for tau
        log_p_W_included = (_log_normal_pdf_c(W_val, mean=0.0,
                                               log_const=self.log_const_tau,
                                               inv2var=self.inv2tau2) * U_bin).sum(dim=1)
    
        # 9. Data likelihood computation
        n = X_data.shape[0]
        # We now compute the predicted data using the A = P^T U P formulation.
        #
        # Given: A = P^T U P  and  X_pred = X * A = X * (P^T U P)
        #
        # We perform the following steps:
        #   (i)   Sort the data: X_sorted = X.matmul(P^T)
        #   (ii)  Apply the DAG (in sorted space): X_pred_sorted = X_sorted.bmm(M)
        #   (iii) Return to canonical order: X_pred = X_pred_sorted.matmul(P)
        #
        # We compute the squared error (SSE) in canonical order.
    
        sse_list = []
        for X_chunk in torch.split(X_data, chunk_size, dim=0):
            # X_chunk: (chunk_size, d)
            # Step (i): Sort the data: use P^T
            X_sorted = X_chunk.unsqueeze(0).matmul(P.transpose(-1, -2))  # shape: (batch_size, chunk_size, d)
            # Step (ii): Apply the DAG structure in sorted space:
            X_pred_sorted = X_sorted.bmm(M)  # shape: (batch_size, chunk_size, d)
            # Step (iii): Return to canonical order:
            X_pred = X_pred_sorted.matmul(P)  # shape: (batch_size, chunk_size, d)
            # Compute error in canonical space:
            diff = X_chunk.unsqueeze(0) - X_pred
            sse_list.append((diff ** 2).sum(dim=(1, 2)))
        sse_batch = torch.stack(sse_list, dim=0).sum(dim=0)
    
        # 10. Compute Gaussian likelihood for each model using precomputed sigma constants:
        ll_batch = (n * d) * self.log_const_sigma - sse_batch * self.inv2sigma2
    
        # 11. Return the combined log probability:
        return ll_batch + reference_log_prob + log_p_W_included
    

    def log_prob_PUPT(self, mk, theta, use_test_data=False):
        """
        Below is A = P U P^T formulation
        """
        PARAM_SCALE = 1.0
        chunk_size = self.chunk_size
        # 1. Select the dataset
        X_data = self.test_x_data if use_test_data else self.x_data
    
        # 2. Parse model representation (all remain on GPU)
        P_cat = mk[:, :self.P_features]
        U_bin = mk[:, self.P_features:]
        W_val = theta * PARAM_SCALE
    
        # 3. Use constant sigma and tau from registered buffers
        # (self.sigma, self.sigma2, self.inv2sigma2, self.log_const_sigma are available)
        sigma = self.sigma  # used for clarity, but use precomputed sigma2 etc.
    
        batch_size = P_cat.shape[0]
        d = P_cat.shape[1] + 1
    
        # 4. Convert categorical representation to permutation matrix (fully vectorized)
        P = cat_representation_to_perm_matrix_onehot_mask(P_cat)  # shape: (batch_size, d, d)
    
        # 5. Build full U and W matrices (upper triangular)
        U_full = build_full_matrix_triu_indices(batch_size, d, U_bin)
        W_full = build_full_matrix_triu_indices(batch_size, d, W_val)
    
        # 6. Compute reference log probability (ensure its implementation is GPU only)
        reference_log_prob = self.reference_log_prob(mk, theta)
    
        # 7. Compute log prior for included edges using precomputed constants for tau:
        log_p_W_included = (_log_normal_pdf_c(W_val, mean=0.0,
                                             log_const=self.log_const_tau,
                                             inv2var=self.inv2tau2) * U_bin).sum(dim=1)
    
        # 8. Data likelihood
        n = X_data.shape[0]
        M = W_full * U_full  # elementwise product
    
        # Option A: Fully vectorized version (if memory allows)
        # X_perm = X_data.unsqueeze(0).matmul(P)  # shape: (batch_size, n, d)
        # X_pred = X_perm.bmm(M)  # shape: (batch_size, n, d)
        # sse_batch = (X_perm - X_pred).pow(2).sum(dim=(1, 2))
    
        # Option B: Chunked version using torch.split
        sse_list = []
        for X_chunk in torch.split(X_data, chunk_size, dim=0):
            # X_chunk: (chunk_size, d)
            X_perm_chunk = X_chunk.unsqueeze(0).matmul(P)  # (batch_size, chunk_size, d)
            X_pred_chunk = X_perm_chunk.bmm(M)  # (batch_size, chunk_size, d)
            diff_chunk = X_perm_chunk - X_pred_chunk
            sse_list.append((diff_chunk ** 2).sum(dim=(1, 2)))
        sse_batch = torch.stack(sse_list, dim=0).sum(dim=0)
    
        # 9. Compute Gaussian likelihood for each model
        # Use precomputed constant self.log_const_sigma and self.inv2sigma2:
        ll_batch = (n * d) * self.log_const_sigma - sse_batch * self.inv2sigma2
    
        # 10. Return the combined log probability
        return ll_batch + reference_log_prob + log_p_W_included

    def true_adjacency_matrix(self):
        # return self.true_P.transpose(0, 1).mm(self.true_U).mm(self.true_P)  # shape (d,d)
        return P_U_to_adjacency(self.true_P, self.true_U, self.device, self.dtype)

    def compute_metrics(self, tdf, num_samples=5000):
        """
        Sample adjacency matrices TransdimensionalFlow and compare them to
        the 'true' adjacency from self.true_P, self.true_U.

        Args:
            tdf: a TransdimensionalFlow  object that can sample DAG structures, i.e. has sample() method
                             returning (mk, theta)
            num_samples (int): how many samples from q_model_sampler to evaluate.

        Returns:
            (avg_f1, avg_shd, Brier): a tuple of floats giving the average F1 and average SHD and Brier
        """
        from vti.flows.transdimensional import TransdimensionalFlow

        assert isinstance(
            tdf, TransdimensionalFlow
        ), "tdf must be subclass of TransdimensionalFlow."

        with torch.no_grad():
    
            # 1) Build the "true" adjacency from the stored P, U
            A_true = self.true_adjacency_matrix()
            A_true_bin = (A_true > 0.5).int()  # Convert to 0/1
    
            d = A_true_bin.shape[0]

            # 2) Sample from tdf
            q_mk_samples, q_theta_samples = tdf._sample(
                num_samples, self.num_inputs(), self.mk_to_context
            )  # the num_inputs is a hack, because we were so basic with base_dist being univariate
    
            if False:
                A_est_samples = self.samples_to_weighted_adj(q_mk_samples, q_theta_samples)
                # logging.info(f"A_est {A_est_samples[:10]}")
            else:
                A_est_samples = convert_mk_to_adjacency(q_mk_samples, self.num_nodes)
                # shape (num_samples, d, d)

            logging.info(f"Average VTI A = {A_est_samples.mean(dim=0)}")
    
            # 3) Compute F1, SHD on each sample
            f1 = compute_averaged_f1_score(A_est_samples, A_true_bin)
            shd = compute_averaged_structured_hamming_distance(A_est_samples, A_true_bin)
            #brier = compute_averaged_brier_score(A_est_samples, A_true_bin)
            brier = compute_averaged_brier_score_single(A_est_samples, A_true_bin)
            auroc = compute_averaged_auroc(A_est_samples, A_true_bin)
    
            return f1, shd, brier, auroc

    def cdt_dagma_metrics(self, sweeplen=10, nonlinear=False):
        # from vti.utils.cdt_helpers import run_dagma_nonlinear_and_evaluate_f1_shd
        from vti.utils.cdt_helpers import (
            run_dagma_nonlinear_and_evaluate_f1_shd_fullsummary,
            run_dagma_linear_and_evaluate_f1_shd_fullsummary,
        )

        if nonlinear:
            return run_dagma_nonlinear_and_evaluate_f1_shd_fullsummary(
                self.x_data,
                self.true_adjacency_matrix(),
                # device=self.device,
                dtype=self.dtype,
                sweeplen=sweeplen,
            )
        else:
            return run_dagma_linear_and_evaluate_f1_shd_fullsummary(
                self.x_data,
                self.true_adjacency_matrix(),
                # device=self.device,
                dtype=self.dtype,
                sweeplen=sweeplen,
            )

    def get_x_data(self):
        return self.x_data


class LinearDAG(AbstractLinearDAG):
    """
    Directed Acyclic Graph with linear likelihood
    """

    def __init__(
        self,
        seed: int = 1,
        num_nodes: int = 3,
        num_data: int = 1000,
        test_data_split: float = 0.0,
        device=None,
        dtype=None,
    ):
        super().__init__(test_data_split=test_data_split, device=device, dtype=dtype)
        assert isinstance(num_nodes, int), "Num nodes must be integer."
        assert num_nodes > 2, "Num nodes must be > 2."

        self.register_buffer("sigma",self.num2tensor(1.0))  # data noise
        self.register_buffer("tau", self.num2tensor(1.0)) # prior stdev for MLP params

        self.chunk_size = 512

        x_data, P, U, W = generate_linear_DAG_data(
            seed,
            num_nodes,
            num_data,
            sigma=self.sigma,
            tau_min=0.3,
            tau_max=0.7,
            data_file=None,
            #data_file="dag_data_{}_{}_{}.pt".format(
            #    seed,
            #    num_nodes,
            #    num_data,
            #),
        )
        self.register_buffer("x_data", x_data)
        self._split_off_test_data()
        self.register_buffer("true_P", P)
        self.register_buffer("true_U", U)
        self.register_buffer("true_W", W)
        self.P_features = int(num_nodes - 1)
        self.U_features = int(num_nodes * (num_nodes - 1) // 2)
        self.num_nodes = num_nodes  # full dimension

        self.sfe_lr = self.get_sfe_lr()

        self._register_precompute_buffers()

        if True:
            # test likelihood
            # convert P into string of num_nodes-1 categoricals
            #A = torch.t(P) @ U @ P
            A = P_U_to_adjacency(P, U, self.device, self.dtype)
            logging.info(f"seed={seed}")
            logging.info(f"P = {P}\nU = {U}\nW = {W}\nA = {A}")
            mk = torch.zeros(
                [
                    self.P_features + self.U_features,
                ]
            )
            mk[: self.P_features] = permutation_matrix_to_integer_categoricals(P)[:-1]
            triu_U = torch.triu_indices(num_nodes, num_nodes, offset=1)
            mk[self.P_features :] = U[triu_U[0], triu_U[1]]
            theta = W[triu_U[0], triu_U[1]]
            mk = mk.reshape(1, -1)
            theta = theta.reshape(1, -1)
            logging.info(f"mk={mk}, theta={theta}")
            loglike = self.log_prob(mk, theta)
            logging.info(f"Log prob of true model = {loglike}")

        #logging.info("Initialised DGP")



class MisspecifiedLinearDAG(AbstractLinearDAG):
    def __init__(
        self,
        seed: int = 1,
        num_nodes: int = 3,
        num_data: int = 1000,
        prior_penalty_gamma: float = 0.0,
        test_data_split: float = 0.0,
        use_mlp_nonlinear_generator = True,
        device=None,
        dtype=None,
    ):
        super().__init__(test_data_split=test_data_split, device=device, dtype=dtype)
        assert isinstance(num_nodes, int), "Num nodes must be integer."
        assert num_nodes > 2, "Num nodes must be > 2."

        self.register_buffer("sigma",self.num2tensor(1.0))  # data noise
        self.register_buffer("tau", self.num2tensor(1.0)) # prior stdev for MLP params

        self.chunk_size = 512


        if use_mlp_nonlinear_generator==False:
            x_data, P, U, W = generate_nonlinear_DAG_data(
                seed,
                num_nodes,
                num_data,
                sigma=self.sigma,
                tau_min=0.3,
                tau_max=0.7,
                data_file=None,
                #data_file="nonlinear_dag_data_{}_{}_{}.pt".format(
                #    seed,
                #    num_nodes,
                #    num_data,
                #),
            )
            self.register_buffer("true_W", W)
        else:
            x_data, P, U, param_store = generate_nonlinear_mlp_DAG_data(
                seed,
                self.sigma, 
                num_nodes, 
                num_data, 
                self.hidden_dim, 
                self.activation, 
                sparsity=0.7,
                device=self.device, 
                dtype=self.dtype,
            )
            self.register_buffer("true_param_store", param_store)

        self.register_buffer("x_data", x_data)
        self._split_off_test_data()
        self.register_buffer("true_P", P)
        self.register_buffer("true_U", U)
        self.P_features = int(num_nodes - 1)
        self.U_features = int(num_nodes * (num_nodes - 1) // 2)
        self.num_nodes = num_nodes  # full dimension

        # penalty on number of nodes in DAG
        #self.prior_penalty_gamma = prior_penalty_gamma
        self.prior_penalty_gamma = self.num2tensor(prior_penalty_gamma)

        self.sfe_lr = self.get_sfe_lr()

        self._register_precompute_buffers()

        if True:
            # test likelihood
            # convert P into string of num_nodes-1 categoricals
            #A = torch.t(P) @ U @ P
            A = P_U_to_adjacency(P, U, self.device, self.dtype)
            logging.info(f"seed={seed}")
            logging.info(f"P = {P}\nU = {U}\nA = {A}")
            mk = torch.zeros(
                [
                    self.P_features + self.U_features,
                ]
            )
            mk[: self.P_features] = permutation_matrix_to_integer_categoricals(P)[:-1]
            triu_U = torch.triu_indices(num_nodes, num_nodes, offset=1)
            mk[self.P_features :] = U[triu_U[0], triu_U[1]]
            #theta = W[triu_U[0], triu_U[1]]
            mk = mk.reshape(1, -1)
            #theta = theta.reshape(1, -1)
            logging.info(f"mk={mk}")
            #loglike = self.log_prob(mk, theta)
            #logging.info(f"Log prob of true model = {loglike}")

        logging.info("Initialised misspecified (non-linear) DGP")

    def mk_prior_dist(self):
        """
        A fast implementation for the penalised prior for
        the permutation upper-triangular representation
        """
        return PermutationDAGPenalizedDistribution(
            num_nodes=self.num_nodes,
            gamma=self.prior_penalty_gamma,
            device=self.device,
            dtype=self.dtype,
        )





class SachsDAG(AbstractLinearDAG):
    """
    Child class that loads the real Sachs dataset from cdt,
    and uses the known adjacency as the 'true' DAG.
    """

    def __init__(
        self,
        test_data_split: float = 0.0,
        prior_penalty_gamma: float = 0.0,
        decimate_data: bool = False,
        device=None,
        dtype=None,
    ):
        # We choose some default for seed, num_nodes, etc., but it won't matter.
        # The parent's __init__ demands them, so let's do placeholders.
        super().__init__(test_data_split=test_data_split, device=device, dtype=dtype)
        # the parent's constructor calls self._generate, but let's overwrite important fields now

        self.chunk_size = 1024 #8192

        # load data
        X_torch, A_torch = load_sachs_data(self.device,self.dtype)

        # store data
        if decimate_data:
            self.register_buffer(
                "x_data", X_torch[::10].contiguous()
            )  # trim the data for performance tst
        else:
            self.register_buffer("x_data", X_torch.contiguous())  # trim the data for performance tst

        self._split_off_test_data()

        # we store the adjacency in "true_A"
        # to remain consistent with parent's usage, we want (P, U, W).
        # We can do: set all edges to weight=1.0 for convenience, or random weights if you prefer.
        W_all_ones = torch.ones_like(A_torch)
        # Then adjacency_to_P_U_W
        # (We must ensure that A_torch is a DAG. The Sachs DAG is indeed acyclic.)
        #P_sachs, U_sachs, W_sachs = adjacency_to_P_U_W(
        #    A_torch, W_all_ones, device=self.device, dtype=self.dtype
        #)

        #self.register_buffer("true_P", P_sachs)
        #self.register_buffer("true_U", U_sachs)
        #self.register_buffer("true_W", W_sachs)
        self.register_buffer("true_A", A_torch)

        self.num_nodes = X_torch.shape[1]  # should be 11 for Sachs
        # If you'd like to keep P_features, U_features consistent:
        self.P_features = self.num_nodes - 1
        self.U_features = (self.num_nodes * (self.num_nodes - 1)) // 2

        # set up priors
        self.register_buffer("sigma",self.num2tensor(1.0))  # data noise
        self.register_buffer("tau", self.num2tensor(1.0)) # prior stdev for MLP params

        # penalty on number of nodes in DAG
        #self.prior_penalty_gamma = self.num2tensor(prior_penalty_gamma)
        self.prior_penalty_gamma = self.num2tensor(prior_penalty_gamma)

        self._register_precompute_buffers()

        self.sfe_lr = self.get_sfe_lr()

        # Done. The data is loaded, and the "true" DAG is set.
        logging.info(
            f"SachsDAG loaded. n={self.x_data.shape[0]} samples, d={self.num_nodes}."
        )
        # no random generation performed

    def true_adjacency_matrix(self):
        return self.true_A

    def mk_prior_dist(self):
        """
        A fast implementation for the penalised prior for
        the permutation upper-triangular representation
        """
        return PermutationDAGPenalizedDistribution(
            num_nodes=self.num_nodes,
            gamma=self.prior_penalty_gamma,
            device=self.device,
            dtype=self.dtype,
        )
