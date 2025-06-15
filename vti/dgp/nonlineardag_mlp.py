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
import random
from torch.distributions import Normal
from vti.dgp.lineardag import AbstractLinearDAG
from vti.utils.dag_helpers import *
from vti.utils.cdt_helpers import *
from vti.utils.math_helpers import ensure_2d, integers_to_binary
import logging

from vti.utils.math_helpers import ensure_2d, integers_to_binary, upper_bound_power_of_2
from vti.dgp.param_transform_factory import (
    construct_param_transform,
    construct_diagnorm_param_transform,
)
from typing import Optional, Tuple, Any, Callable



########################################
# NonLinearDAG class with batched MLP
########################################



class NonLinearDAG_BatchedMLP(AbstractLinearDAG):
    """
    A fast single-hidden-layer MLP DAG approach. Data is shape (n,p) externally,
    but we store it as (p,n). No residual skip, purely MLP.

    Node j has param_count_j = (j+2)*hidden_dim + 1 => input j, hidden dim=hidden_dim => total = jH + H + H +1 = (j+2)*H +1
    We'll do a coarse mk_to_mask: if node j has no active parents, we mask out that entire chunk.
    """

    def __init__(
        self,
        seed=1,
        num_nodes=3,
        num_data=1024,
        test_data_split=0.0,
        hidden_dim=10,
        activation="relu",
        sigma=1.0,
        tau=1.0,
        prior_penalty_gamma: float = 0.0,
        use_mlp_nonlinear_generator=True,
        generate_synthetic_data=True,
        device=None,
        dtype=None,
    ):
        super().__init__(test_data_split=test_data_split, device=device, dtype=dtype)
        assert num_nodes > 2, "Num nodes must be > 2."
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.activation = activation.lower()

        self.chunk_size = 512

        self.register_buffer("sigma", self.num2tensor(sigma))  # data noise
        self.register_buffer("tau", self.num2tensor(tau))  # prior stdev for MLP params

        # We'll define arrays for P_features, U_features, ...
        d = num_nodes
        self.P_features = d - 1
        self.U_features = d * (d - 1) // 2

        # Summation for total param dimension
        self.param_count_per_node = []
        self._theta_dim = 0
        for j in range(d):
            if j == 0:
                pcj = 0
            else:
                # For each node j, parameters for MLP: (j * H + H) for first layer + (H * 1 + 1) for second layer
                pcj = (j + 2) * self.hidden_dim + 1
            self.param_count_per_node.append(pcj)
            self._theta_dim += pcj

        # penalty on number of nodes in DAG
        self.prior_penalty_gamma = prior_penalty_gamma

        # precompute often used values
        self._register_precompute_buffers()

        # set lr for sfe optim
        self.sfe_lr = self.get_sfe_lr()

        if generate_synthetic_data:
            # Generate data
            if use_mlp_nonlinear_generator == False:
                x_data, P, U, W = generate_nonlinear_DAG_data(
                    seed, num_nodes, num_data, sigma=self.sigma,
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
                    sparsity=0.5,
                    device=self.device,
                    dtype=self.dtype,
                )
                self.register_buffer("true_param_store", param_store)
            self.register_buffer("x_data", x_data.t().contiguous())
            # self._split_off_test_data()
            self.register_buffer("true_P", P)
            self.register_buffer("true_U", U)

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
            mk = mk.reshape(1, -1)
            logging.info(f"mk={mk}")
            #logging.info(f"Log prob of true model = {loglike}")

        logging.info(f"Initialized NonLinearDAG with d={num_nodes}, hidden_dim={hidden_dim}.")

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

    def num_inputs(self):
        return self._theta_dim

    def get_x_data(self):
        return self.x_data.t()

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
        logging.info("Constructing param transform for MLP DAG...")

        # A closure for mapping context -> param mask and its reverse.
        context_to_mask = lambda context: self.mk_to_mask(context)
        #context_to_mask_reverse = lambda context: torch.fliplr(self.mk_to_mask(context))

        context_transform = lambda context: self.mk_context_transform(context)

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
            "num_pmaf_hidden_features": upper_bound_power_of_2(2 * self.num_inputs()),  # offers slight improvement over 2*num_inputs. Not enough
            "num_inputs": self.num_inputs(),
            "num_context_inputs": self.num_context_features(),
            "context_to_mask": context_to_mask,
            #"context_to_mask_reverse": context_to_mask_reverse,
            "context_transform": context_transform,
            "num_pqrs_hidden_features": upper_bound_power_of_2(2 * self.num_inputs()),
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
            "affine220": {"num_pre_pmaf_layers": 2, "num_affine_blocks": 20},
            "affine5": {"num_pre_pmaf_layers": 5},
            "affine52": {"num_pre_pmaf_layers": 5, "num_affine_blocks": 2},
            "affine55": {"num_pre_pmaf_layers": 5, "num_affine_blocks": 5},
            "affine55f128": {"num_pre_pmaf_layers": 5, "num_affine_blocks": 5, "num_pmaf_hidden_features": 128},
            "affine55f256": {"num_pre_pmaf_layers": 5, "num_affine_blocks": 5, "num_pmaf_hidden_features": 256},
            "affine55f512": {"num_pre_pmaf_layers": 5, "num_affine_blocks": 5, "num_pmaf_hidden_features": 512},
            "affine7": {"num_pre_pmaf_layers": 7},
            "affine510": {"num_pre_pmaf_layers": 5, "num_affine_blocks": 10},
            "affine510f128": {"num_pre_pmaf_layers": 5, "num_affine_blocks": 10, "num_pmaf_hidden_features": 128},
            "affine510f256": {"num_pre_pmaf_layers": 5, "num_affine_blocks": 10, "num_pmaf_hidden_features": 256},
            "affine510f512": {"num_pre_pmaf_layers": 5, "num_affine_blocks": 10, "num_pmaf_hidden_features": 512},
            "affine515": {"num_pre_pmaf_layers": 5, "num_affine_blocks": 15},
            "affine515f128": {"num_pre_pmaf_layers": 5, "num_affine_blocks": 15, "num_pmaf_hidden_features": 128},
            "affine515f256": {"num_pre_pmaf_layers": 5, "num_affine_blocks": 15, "num_pmaf_hidden_features": 256},
            "affine515f512": {"num_pre_pmaf_layers": 5, "num_affine_blocks": 15, "num_pmaf_hidden_features": 512},
            "affine520": {"num_pre_pmaf_layers": 5, "num_affine_blocks": 20},
            "affine525": {"num_pre_pmaf_layers": 5, "num_affine_blocks": 25},
            "affine720": {"num_pre_pmaf_layers": 7, "num_affine_blocks": 20},
            "affine10": {"num_pre_pmaf_layers": 10},
            "affine104": {"num_pre_pmaf_layers": 10, "num_affine_blocks": 4},
            "spline23": {"num_prqs_layers": 2, "num_pqrs_blocks": 3},
            "spline210": {"num_prqs_layers": 2, "num_pqrs_blocks": 10},
            "spline46": {"num_prqs_layers": 4, "num_pqrs_blocks": 6},
            "spline410": {"num_prqs_layers": 4, "num_pqrs_blocks": 10},
            "spline410f128": {"num_prqs_layers": 4, "num_pqrs_blocks": 10, "num_pqrs_hidden_features": 128},
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
                context_encoder_hidden_features=2 * self.num_context_features(),
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

    def mk_to_mask(self, mk):
        r"""
        Compute a fine-grained mask for the MLP parameters (theta) based on the DAG
        structure encoded in mk.

        For each node j (with j ≥ 1):
          - We have a first-layer weight matrix W1 (of shape (H, j)), plus b1 of shape (H),
            plus W2 of shape(H), plus b2(1). The total param count for node j is (j*H + 2H + 1).
          - If at least one parent is included (U[i,j] = 1 for some i<j), then:
              * W1 is masked column-by-column, skipping columns i for which U[i,j] = 0
                but including columns for U[i,j] = 1.
              * b1, W2, b2 are always active in that scenario.
          - If no parent is included (i.e. sum(U[:j,j])=0), we exclude
            the entire MLP for node j => W1 + b1 + W2 + b2 all set to False.
    
        Args:
          mk (torch.Tensor): shape (batch_size, P_features + U_features) encoding the DAG.
    
        Returns:
          torch.BoolTensor: A mask tensor of shape (batch_size, _theta_dim)
                            where each entry is True if the corresponding parameter
                            is active (i.e. included in the prior), and False otherwise.
        """
        batch_size = mk.shape[0]
        d = self.num_nodes
        H = self.hidden_dim
    
        # Extract the U binary vector and build the full U matrix.
        U_bin = mk[:, self.P_features:]  # shape: (batch_size, U_features)
        U_full = build_full_matrix_triu_indices(batch_size, d, U_bin)  # shape: (batch_size, d, d)
    
        mask_list = []
        # For node 0, there are no parameters in your MLP specification.
        # For each node j >= 1:
        for j in range(1, d):
            # Get the parent connectivity for node j:
            # For each sample, U_full[b, :j, j] is a vector (length j) indicating which of the
            # potential j parents are active.
            parent_mask = U_full[:, :j, j]  # shape: (batch_size, j)
            # Convert to boolean (True if active)
            parent_mask_bool = parent_mask > 0  # shape: (batch_size, j)
    
            # For the first-layer weight matrix W1 of node j (flattened as j*H),
            # we replicate each parent's flag by H times for columns that are included:
            mask_W1 = parent_mask_bool.unsqueeze(2).expand(-1, -1, H).reshape(batch_size, j * H)
    
            # For the remaining parameters of node j (the biases for layer 1 and entire layer 2),
            # we normally want them active. total length = (H + (H + 1)) = 2H + 1
            ones_remainder = torch.ones(
                batch_size, (2 * H + 1),
                device=mk.device, dtype=torch.bool
            )
    
            # Concatenate the two parts: W1 mask + all-ones remainder => shape (batch_size, j*H + 2H + 1)
            mask_j = torch.cat([mask_W1, ones_remainder], dim=1)
    
            # == NEW STEP: if sum of parents == 0, exclude the entire MLP for node j
            # i.e. set all mask_j to False for that batch element
            #     i.e. W1 + b1 + W2 + b2 => all False.
            # (This is the minimal change from the original code.)
            sum_parents = parent_mask_bool.sum(dim=1)  # shape: (batch_size,)
            no_parent = (sum_parents == 0)             # shape: (batch_size,)
            # For those batch rows b where no_parent[b] is True, we set mask_j[b,:] = False
            mask_j[no_parent] = False
            # == END NEW STEP
    
            mask_list.append(mask_j)
    
        # Concatenate for all nodes j=1..d-1
        if len(mask_list) > 0:
            final_mask = torch.cat(mask_list, dim=1)  # shape: (batch_size, _theta_dim)
        else:
            final_mask = torch.empty(batch_size, 0, device=mk.device, dtype=torch.bool)
    
        return final_mask.to(dtype=self.dtype)

    def _apply_mlp_single_hidden_batch_batched(self, x_in, param_j):
        """
        x_in: shape => (batch_size, chunk_size, j)
        param_j: shape => (batch_size, (j+2)*hidden_dim +1 )
        We'll parse:
          W1 => shape (batch_size, H, j)
          b1 => shape (batch_size, H)
          W2 => shape (batch_size, 1, H)
          b2 => shape (batch_size)
        Then do z= x_in bmm W1^T => shape(bsz, chunk_size, H). Then add b1 => broadcast =>(bsz,1,H).
        Then activation => relu. Then out => shape(bsz, chunk_size,1). Then we add b2 => shape(bsz,1,1). We squeeze =>(bsz,chunk_size).
        """
        bsz, csize, j_dim = x_in.shape
        H = self.hidden_dim

        # Parse MLP parameters
        len1 = j_dim * H + H  # Weights and biases for first layer
        chunk_l1 = param_j[:, :len1]  # Shape: (bsz, len1)
        W1 = chunk_l1[:, : j_dim * H].view(bsz, H, j_dim)  # Shape: (bsz, H, j_dim)
        b1 = chunk_l1[:, j_dim * H : j_dim * H + H]  # Shape: (bsz, H)

        # Second layer parameters
        chunk_l2 = param_j[:, len1 : len1 + (H + 1)]  # Shape: (bsz, H + 1)
        W2 = chunk_l2[:, :H].view(bsz, 1, H)  # Shape: (bsz, 1, H)
        b2 = chunk_l2[:, H]  # Shape: (bsz,)

        # Perform first layer: z = x_in @ W1^T + b1
        # x_in: (bsz, csize, j_dim)
        # W1: (bsz, H, j_dim) => transpose to (bsz, j_dim, H) for bmm
        W1_t = W1.transpose(1, 2)  # Shape: (bsz, j_dim, H)
        z = torch.bmm(x_in, W1_t)  # Shape: (bsz, csize, H)
        z = z + b1.unsqueeze(1)  # Broadcast b1: (bsz, 1, H)

        # Activation
        z = torch.relu(z)  # Shape: (bsz, csize, H)

        # Second layer: out = z @ W2^T + b2
        # W2: (bsz, 1, H) => transpose to (bsz, H, 1) for bmm
        W2_t = W2.transpose(1, 2)  # Shape: (bsz, H, 1)
        out = torch.bmm(z, W2_t)  # Shape: (bsz, csize, 1)
        out = out + b2.view(bsz, 1, 1)  # Broadcast b2: (bsz, 1, 1)

        # Squeeze to (bsz, csize)
        out = out.squeeze(2)
        return out

    def _apply_mlp_single_hidden(self, x_in, chunk, node_j):
        """
        For a single sample.

        Args:
            x_in (torch.Tensor): Input tensor of shape (j_dim,).
            chunk (torch.Tensor): Parameter chunk for the node.
            node_j (int): Node index.

        Returns:
            torch.Tensor: Output scalar.
        """
        j = node_j
        H = self.hidden_dim
        len_l1 = j * H + H
        l1 = chunk[:len_l1]
        W1 = l1[: j * H].view(H, j)
        b1 = l1[j * H : j * H + H]
        l2 = chunk[len_l1 : len_l1 + (H + 1)]
        W2 = l2[:H]
        b2 = l2[H]

        z = W1.matmul(x_in) + b1
        if self.activation == "relu":
            z = torch.relu(z)
        elif self.activation == "tanh":
            z = torch.tanh(z)
        out = W2.dot(z) + b2
        return out

    def _log_normal_pdf(self, x, mean=0.0, std=1.0):
        """
        Compute the log probability of a normal distribution.

        Args:
            x (torch.Tensor): Input tensor.
            mean (float or torch.Tensor): Mean of the normal distribution.
            std (float or torch.Tensor): Standard deviation.

        Returns:
            torch.Tensor: Log probability tensor.
        """
        var = std**2
        return -0.5 *(2 * self.pi * var).log() - (x - mean) ** 2 / (2 * var)

    def reference_log_prob(self, mk, theta):
        original_shape = theta.shape
        static_log_prob = (
            (1 - self.mk_to_mask(mk))
            * self.referencedist.log_prob(theta.reshape(-1, 1)).view(original_shape)
        ).sum(dim=-1)
        return static_log_prob

    def log_prob(self, mk, theta, use_test_data=False):
        if ADJFORMPUPT:
            return self.log_prob_PUPT(mk, theta, use_test_data)
        else:
            return self.log_prob_PTUP(mk, theta, use_test_data)
    
    def log_prob_PUPT(self, mk, theta, use_test_data=False):
        """
        Compute the log probability of the data given the DAG parameters.
        A=P*U*P^T format
        
        Inputs:
          - mk: tensor of shape (batch_size, P_features + U_features)
          - theta: tensor of shape (batch_size, _theta_dim)
          - use_test_data: if True, use test data (self.test_x_data) otherwise self.x_data.
          - chunk_size: number of columns (data points) to process at once.
          
        Returns:
          - log probability tensor of shape (batch_size,)
        """
        # Select the data (assumed shape: (p, n))
        X_data = self.test_x_data if use_test_data else self.x_data
        p, n = X_data.shape
        batch_size = mk.shape[0]
        chunk_size = self.chunk_size

        # set a scale factor for parameters
        PARAM_SCALE = 1.0
        
        # Parse the DAG structure
        U_bin = mk[:, self.P_features :]  # (batch_size, U_features)
        U_full = build_full_matrix_triu_indices(batch_size, p, U_bin)  # (batch_size, p, p)
        
        P_cat = mk[:, :self.P_features]  # (batch_size, P_features)
        P = cat_representation_to_perm_matrix_onehot_mask(P_cat.to(self.device))  # (batch_size, p, p)
        
        # Compute reference log probability and prior log probability
        reference_lp = self.reference_log_prob(mk, theta)  # (batch_size,)
        if False:
            # BAD prior evaluation, because it does not mask
            prior_lp = self._log_normal_pdf(theta, mean=0.0, std=self.tau).sum(dim=1)  # (batch_size,)
        else:
            prior_lp = (
                self.mk_to_mask(mk)
                * self._log_normal_pdf(theta*PARAM_SCALE, mean=0.0, std=self.tau)
            ).sum(dim=-1)
        
        # Precompute offsets (if not already stored)
        # (Assume self._theta_offsets is a list of (start, end) for each node.)
        # TODO use mlpdag_param_count_per_node() for the below offset table
        offsets = self._theta_offsets if hasattr(self, "_theta_offsets") else [
            (sum(self.param_count_per_node[:j]), sum(self.param_count_per_node[:j+1]))
            for j in range(p)
        ]
        
        # Compute sum-of-squared errors (SSE) in chunks
        #sse_batch = torch.zeros(batch_size, device=self.device, dtype=self.dtype)
        sse_batches = []
        for X_chunk in torch.split(X_data, chunk_size, dim=1):  # X_chunk: (p, c)
            # Expand to batch dimension and apply permutation
            X_chunk_3d = X_chunk.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, p, c)
            X_perm = torch.bmm(P, X_chunk_3d)  # (batch_size, p, c)
            X_pred = torch.zeros_like(X_perm)
            # Loop over nodes (for j2 = 0,..., p-1)
            for j2 in range(p):
                if j2 == 0:
                    X_pred[:, 0, :] = 0.0
                else:
                    parents_mask = U_full[:, :j2, j2].float()  # (batch_size, j2)
                    Xp_j2 = X_perm[:, :j2, :]  # (batch_size, j2, c)
                    # Transpose to (batch_size, c, j2) and apply the parent mask
                    mlp_input = Xp_j2.transpose(1, 2) * parents_mask.unsqueeze(1)  # (batch_size, c, j2)
                    # Retrieve the parameter chunk for node j2
                    ofs_s, ofs_e = offsets[j2]
                    param_j2 = theta[:, ofs_s:ofs_e]*PARAM_SCALE  # (batch_size, param_count_j2)
                    # Compute MLP output for node j2 over the chunk
                    mlp_out = self._apply_mlp_single_hidden_batch_batched(mlp_input, param_j2)
                    X_pred[:, j2, :] = mlp_out
            diff = X_perm - X_pred
            #sse_batch += diff.pow(2).sum(dim=(1, 2))
            sse_batches.append(diff.pow(2).sum(dim=(1, 2)))

        sse_batch = torch.stack(sse_batches).sum(dim=0)
        
        # 6. Compute Gaussian log-likelihood using precomputed constant buffers
        ll_batch = (n * p) * self.log_const_sigma - sse_batch * self.inv2sigma2
        
        # 7. Total log probability
        return ll_batch + reference_lp + prior_lp
    

    def log_prob_PTUP(self, mk, theta, use_test_data=False):
        """
        Compute the log probability of the data given the DAG parameters.
        
        This version implements the A = P^T U P formulation.
        
        Inputs:
          - mk: tensor of shape (batch_size, P_features + U_features)
          - theta: tensor of shape (batch_size, _theta_dim)
          - use_test_data: if True, use test data (self.test_x_data) otherwise self.x_data.
          - chunk_size: number of columns (data points) to process at once.
            
        Returns:
          - log probability tensor of shape (batch_size,)
        """
        # 1. Select the data (stored as (p, n), with p nodes and n data points)
        X_data = self.test_x_data if use_test_data else self.x_data  # shape: (p, n)
        p, n = X_data.shape
        batch_size = mk.shape[0]
        chunk_size = self.chunk_size

        # set a scale factor for parameters
        PARAM_SCALE = 1.0
        
        # 2. Parse the DAG structure
        U_bin = mk[:, self.P_features :]  # (batch_size, U_features)
        U_full = build_full_matrix_triu_indices(batch_size, p, U_bin)  # (batch_size, p, p)
    
        P_cat = mk[:, :self.P_features]  # (batch_size, P_features)
        P = cat_representation_to_perm_matrix_onehot_mask(P_cat.to(self.device))  # (batch_size, p, p)
    
        # 3. Compute reference log probability and prior log probability
        reference_lp = self.reference_log_prob(mk, theta)  # (batch_size,)

        if False:
            # BAD prior evaluation, because it does not mask
            prior_lp = self._log_normal_pdf(theta, mean=0.0, std=self.tau).sum(dim=1)  # (batch_size,)
        else:
            prior_lp = (
                self.mk_to_mask(mk)
                * self._log_normal_pdf(theta*PARAM_SCALE, mean=0.0, std=self.tau)
            ).sum(dim=-1)
    
        # 4. Precompute offsets (if not already stored)
        # (Assume self._theta_offsets is a list of (start, end) for each node.)
        offsets = self._theta_offsets if hasattr(self, "_theta_offsets") else [
            (sum(self.param_count_per_node[:j]), sum(self.param_count_per_node[:j+1]))
            for j in range(p)
        ]
        
        # 5. Compute sum-of-squared errors (SSE) in chunks
        sse_batches = []
        for X_chunk in torch.split(X_data, chunk_size, dim=1):  # X_chunk: (p, c) where c is the chunk size
            # Expand to batch dimension; X_chunk_3d is in canonical order
            X_chunk_3d = X_chunk.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, p, c)
            
            # Step 1: Permute the canonical data to sorted order
            # X_sorted = P * X  (sorted data)
            X_sorted = torch.bmm(P, X_chunk_3d)  # shape: (batch_size, p, c)
            
            # Step 2: Compute predictions in the sorted space (node-by-node)
            X_pred_sorted = torch.zeros_like(X_sorted)
            for j2 in range(p):
                if j2 == 0:
                    X_pred_sorted[:, 0, :] = 0.0
                else:
                    # For node j2, only nodes 0 ... j2-1 are its potential parents in sorted order.
                    parents_mask = U_full[:, :j2, j2].float()  # (batch_size, j2)
                    Xp_j2 = X_sorted[:, :j2, :]  # (batch_size, j2, c)
                    # Transpose to (batch_size, c, j2) and apply the parent mask elementwise
                    mlp_input = Xp_j2.transpose(1, 2) * parents_mask.unsqueeze(1)  # (batch_size, c, j2)
                    # Retrieve the parameter chunk for node j2
                    ofs_s, ofs_e = offsets[j2]
                    param_j2 = theta[:, ofs_s:ofs_e]*PARAM_SCALE  # (batch_size, param_count_j2)
                    # Compute MLP output for node j2 over the chunk
                    mlp_out = self._apply_mlp_single_hidden_batch_batched(mlp_input, param_j2)  # (batch_size, c)
                    X_pred_sorted[:, j2, :] = mlp_out
            
            # Step 3: Convert the sorted prediction back to canonical order
            X_pred = torch.bmm(P.transpose(1, 2), X_pred_sorted)  # (batch_size, p, c)
            
            # Step 4: Compute the difference between canonical data and predicted canonical output
            diff = X_chunk_3d - X_pred  # (batch_size, p, c)
            sse_batches.append(diff.pow(2).sum(dim=(1, 2)))
        
        sse_batch = torch.stack(sse_batches).sum(dim=0)
        
        # 6. Compute Gaussian log-likelihood using precomputed constant buffers
        ll_batch = (n * p) * self.log_const_sigma - sse_batch * self.inv2sigma2
        
        # 7. Total log probability (adding reference and prior log probabilities)
        return ll_batch + reference_lp + prior_lp

    def compute_metrics(self, tdf, num_samples=5000):
        """
        # TODO FIXME compare with same method in AbstractLinearDAG and remove if same
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
            #A_true = (
            #    self.true_P.transpose(0, 1).mm(self.true_U).mm(self.true_P)
            #)  # shape (d,d)
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
            f1 = compute_averaged_f1_score(A_est_samples, A_true_bin, threshold=1e-3)
            shd = compute_averaged_structured_hamming_distance(A_est_samples, A_true_bin, threshold=1e-3)
            #brier = compute_averaged_brier_score(A_est_samples, A_true_bin, threshold=1e-3)
            brier = compute_averaged_brier_score_single(A_est_samples, A_true_bin, threshold=1e-3)
            auroc = compute_averaged_auroc(A_est_samples, A_true_bin)

    
            return f1, shd, brier, auroc

    def cdt_dagma_metrics(self, sweeplen=10, nonlinear=True):
        # from vti.utils.cdt_helpers import run_dagma_nonlinear_and_evaluate_f1_shd
        from vti.utils.cdt_helpers import (
            run_dagma_nonlinear_and_evaluate_f1_shd_fullsummary,
            run_dagma_linear_and_evaluate_f1_shd_fullsummary,
        )

        if nonlinear:
            return run_dagma_nonlinear_and_evaluate_f1_shd_fullsummary(
                self.x_data.t(),
                self.true_adjacency_matrix(),
                # device=self.device,
                dtype=self.dtype,
                sweeplen=sweeplen,
            )
        else:
            return run_dagma_linear_and_evaluate_f1_shd_fullsummary(
                self.x_data.t(),
                self.true_adjacency_matrix(),
                # device=self.device,
                dtype=self.dtype,
                sweeplen=sweeplen,
            )



class SachsNonLinearMLPDAG(NonLinearDAG_BatchedMLP):
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
        super().__init__(
            seed=None,
            num_nodes=11,  # Known for this data, and need to provide to parent constructor
            num_data=7500,
            test_data_split=test_data_split,
            #hidden_dim=10,
            hidden_dim=5,
            activation="relu",
            generate_synthetic_data=False,
            device=device,
            dtype=dtype,
        )
        # the parent's constructor calls self._generate, but let's overwrite important fields now
        self.chunk_size = 1024 #8192

        # load data
        X_torch, A_torch = load_sachs_data(self.device,self.dtype)

        # store data
        if decimate_data:
            self.register_buffer(
                "x_data", X_torch[::10].t().contiguous()
            )  # trim the data for performance tst
        else:
            self.register_buffer(
                "x_data", X_torch.t().contiguous()
            )  # trim the data for performance tst

        # self._split_off_test_data() # we transpose, so override this in parent and fix later. TODO FIXME

        # we store the adjacency in "true_A"
        # to remain consistent with parent's usage, we want (P, U, W).
        # We can do: set all edges to weight=1.0 for convenience, or random weights if you prefer.
        #W_all_ones = torch.ones_like(A_torch)
        # Then adjacency_to_P_U_W
        # (We must ensure that A_torch is a DAG. The Sachs DAG is indeed acyclic.)
        #P_sachs, U_sachs, W_sachs = adjacency_to_P_U_W(
        #    A_torch, W_all_ones, device=self.device, dtype=self.dtype
        #)

        #self.register_buffer("true_P", P_sachs)
        #self.register_buffer("true_U", U_sachs)
        #self.register_buffer("true_W", W_sachs)

        self.register_buffer("true_A", A_torch)

        logging.info(f"Adjacency matrix:\n{A_torch}")


        self.num_nodes = X_torch.shape[1]  # should be 11 for Sachs
        # If you'd like to keep P_features, U_features consistent:
        self.P_features = self.num_nodes - 1
        self.U_features = (self.num_nodes * (self.num_nodes - 1)) // 2

        # set up priors
        self.register_buffer("sigma",self.num2tensor(1.0))  # data noise
        self.register_buffer("tau", self.num2tensor(1.0)) # prior stdev for MLP params

        # penalty on number of nodes in DAG
        self.prior_penalty_gamma = prior_penalty_gamma

        self._register_precompute_buffers()

        self.sfe_lr = self.get_sfe_lr()

        # Done. The data is loaded, and the "true" DAG is set.
        logging.info(
            f"SachsDAG loaded. n={self.x_data.shape[1]} samples, d={self.num_nodes}."
        )
        # no random generation performed

    def true_adjacency_matrix(self):
        return self.true_A

