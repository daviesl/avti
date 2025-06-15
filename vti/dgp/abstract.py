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
from torch.distributions import Normal, Categorical
from nflows.utils import torchutils
import torch.nn.functional as F
from vti.utils.debug import tonp
from vti.utils.plots import plot_fit_marginals
from vti.dgp.param_transform_factory import (
    construct_param_transform,
    construct_diagnorm_param_transform,
)
from vti.utils.math_helpers import upper_bound_power_of_2

# from vti.utils.logging import logging
import logging
from typing import Optional, Tuple, Any, Callable


class AbstractDGP(torch.nn.Module):
    """
    Base class for a Data-Generating Process (DGP) in the 'vti' framework.

    Responsibilities:
    - Holding references to device, dtype, and a baseline reference distribution (`referencedist`).
    - Providing default or abstract methods for constructing and transforming parameters,
      for enumerating or handling "model masks," etc.
    - Child classes should override or implement the abstract methods:
        - num_categories()
        - num_inputs()
        - num_context_features()
        - printVTIResults()
        - log_prob()
    """

    def __init__(
        self,
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
        # referencedist: torch.distributions.Distribution = Normal(0, 1),
        referencedist=None,
        **kwargs: Any,
    ):
        """
        Args:
            device (torch.device or None): The device to place tensors on.
            dtype (torch.dtype or None): The data type (e.g., torch.float32).
            referencedist (Distribution): Reference distribution used for proposals or weighting.
                                          Defaults to N(0,1).
            **kwargs: Additional keyword args (not used internally).
        """
        super().__init__()
        self.device = device
        self.dtype = dtype
        torch.set_default_dtype(dtype)
        if referencedist is None:
            loc = torch.tensor(0.0, device=device, dtype=dtype)
            scale = torch.tensor(1.0, device=device, dtype=dtype)
            referencedist = Normal(loc, scale)
        self.referencedist = referencedist

        # useful torch constants
        self.pi = torch.acos(self.num2tensor(0.)).item() * 2

        # Log a warning if extra kwargs are passed in,
        # to alert devs that those kwargs are unrecognized.
        logging.warning(f"{self.__class__.__name__} unused kwargs: {kwargs}")

    def get_flow_lr(self) -> float:
        """
        Return the learning rate for flow-based methods (e.g., normalizing flows).
        Child classes might define self.flow_lr somewhere else, or via 'construct_param_transform()'.
        """
        return self.flow_lr

    def get_sfe_lr(self) -> float:
        """
        Return the learning rate for score-function estimators (SFE).
        """
        return self.sfe_lr

    def num_categories(self) -> int:
        """
        Number of "categories" for the model or parameter dimension.
        Child classes must implement.
        """
        raise NotImplementedError()

    def num_inputs(self) -> int:
        """
        Return number of input dimensions (e.g., total params).
        Child classes must implement.
        """
        raise NotImplementedError()

    def num_context_features(self) -> int:
        """
        Return the dimensionality of any 'context' that might select or mask parameters.
        Child classes must implement.
        """
        raise NotImplementedError()

    def reference_dist(self) -> torch.distributions.Distribution:
        """
        Returns the reference distribution object.
        """
        return self.referencedist

    def reference_dist_sample_and_log_prob(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Draw samples from the reference distribution and get their log probability.

        Args:
            batch_size (int): Number of samples to draw.

        Returns:
            (torch.Tensor, torch.Tensor):
                - samples: shape [batch_size, num_inputs()]
                - log_prob: shape [batch_size], the log probability under 'referencedist'.
        """
        samples = self.referencedist.rsample(
            sample_shape=torch.Size([batch_size, self.num_inputs()])
        )
        # sum_except_batch sums across all dims except the first -> log prob per sample
        lp = torchutils.sum_except_batch(
            self.referencedist.log_prob(samples), num_batch_dims=1
        )
        return samples, lp

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
        logging.info("Constructing param transform...")

        # A closure for mapping context -> param mask and its reverse.
        context_to_mask = lambda context: self.mk_to_mask(context)
        #context_to_mask_reverse = lambda context: torch.fliplr(self.mk_to_mask(context))

        context_transform = lambda context : self.mk_context_transform(context)

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
            "num_pqrs_bins": 10,
            "num_pqrs_blocks": 2,
            "use_diag_affine_start": False,
            "use_diag_affine_end": False,
            "diag_affine_context_encoder_depth": 5,
            "diag_affine_context_encoder_hidden_features":upper_bound_power_of_2(2*self.num_inputs()),
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
                #ncontext_to_mask_reverse,
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

    def printVTIResults(self, mk_probs: torch.Tensor) -> None:
        """
        Placeholder method to display results.
        Child classes should override.
        """
        raise NotImplementedError()

    #### diagnostic plots

    def plot_q_mk_selected(
        self, param_transform, q_mk_identifiers, q_mk_probs, num_samples
    ):
        """ """
        q_theta = []
        for k, mk in enumerate(q_mk_identifiers):
            N = int(num_samples * q_mk_probs[k])
            if N > 0:
                # b_samples = base_dist._sample(N, context=None)
                b_samples, _blp = self.reference_dist_sample_and_log_prob(N)
                qt, _ = param_transform.inverse(b_samples, context=mk.view(1, -1))
                qt = qt * self.mk_to_mask(mk.view(1, -1))
                q_theta.append(tonp(qt))
        plot_fit_marginals(q_theta[0], q_theta[1:])

    def plot_q(self, param_transform, q_mk_probs, num_samples):
        """ """
        q_theta = []
        for k, mk in enumerate(self.mk_identifiers()):
            N = int(num_samples * q_mk_probs[k])
            if N > 0:
                # b_samples = base_dist._sample(N, context=None)
                b_samples, _blp = self.reference_dist_sample_and_log_prob(N)
                qt, _ = param_transform.inverse(b_samples, context=mk.view(1, -1))
                qt = qt * self.mk_to_mask(mk.view(1, -1))
                q_theta.append(tonp(qt))
        plot_fit_marginals(q_theta[0], q_theta[1:])

    def plot_q_tdf(
        self,
        tdf,
        num_samples=1024,
        title="Variational transdimensional density plot",
        font_size="6",
        figsize=(8, 8),
        saveto=False,
    ):
        with torch.no_grad():
            q_theta = []
            mk_samples, theta_samples = tdf._sample(
                num_samples, self.num_inputs(), self.mk_to_context
            )  # the num_inputs is a hack, because we were so basic with base_dist being univariate
            unique_mk, reverse_indices = torch.unique(
                mk_samples, return_inverse=True, dim=0
            )
            for i, mk in enumerate(unique_mk):
                ts = theta_samples[reverse_indices == i]
                mask = self.mk_to_mask(mk.view(1, -1))
                ts = (ts * mask).detach().cpu().numpy()
                q_theta.append(ts)
            plot_fit_marginals(
                q_theta[0],
                q_theta[1:],
                title=title,
                font_size=font_size,
                figsize=figsize,
                saveto=saveto,
            )

    def plot_joints(self, param_transform):
        # TODO move this to abstract dgp
        # tbsamples, _ = base_dist.sample_and_log_prob(10000)
        tbsamples, _ = self.reference_dist_sample_and_log_prob(10000)
        cttest = torch.zeros(
            self.num_categories(), dtype=self.dtype, device=self.device
        )
        cttest[1] = 1
        tpsamples, __ = param_transform.inverse(tbsamples, context=cttest.view(1, -1))
        return plot_fit_marginals(tonp(tpsamples), [], 1e-10)

    def mk_prior_dist(self):
        """
        For VTI, the implementation of the model prior requires only
        a log_prob() evaluation method.
        """
        return Categorical(
            logits=torch.zeros(
                (self.num_categories(),),
                dtype=self.dtype,
                device=self.device,
            )
        )

    def mk_identifiers(self) -> torch.Tensor:
        """
        Returns an enumerated tensor of every model identifier.
        """
        raise NotImplementedError(
            "{__class__.__name__}.mk_identifiers() not implemented."
        )

    def mk_to_context(self, mk_samples: torch.Tensor) -> torch.Tensor:
        """
        This method converts draws from the model sampler (mk dist) to a format
        that the flow context wants to see. By default, it is passthrough.
        """
        return mk_samples

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
        return mk_samples


    def mk_cat_to_identifier(self, cat_samples: torch.Tensor) -> torch.Tensor:
        """
        The mapping from a categorical random variable (integers) to the model identifier random variable.

        Args:
            cat_samples (torch.Tensor): 1D integer tensor with values in [0, num_categories).

        Returns:
            torch.Tensor: shape [batch_size, <mk identifier length>].
        """
        raise NotImplementedError(
            "{__class__.__name__}.mk_cat_to_identifier() not implmemented."
        )

    def mk_identifier_to_cat(self, mk_samples: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "{__class__.__name__}.mk_identifier_to_cat() not implemented"
        )

    def get_true_log_probs_for_mk_identifiers(
        self, mk_samples: torch.Tensor, force_refresh=False
    ) -> torch.Tensor:
        """
        For each row in mk_samples, return the "true" log_prob.
        Most DGPs will not have true log_probs available, but may offer approximation via RJMCMC sampling.
        """
        raise NotImplementedError(
            "{__class__.__name__}.get_true_log_probs_for_mk_identifiers() not implemented"
        )

    def mk_to_mask(self, mk: torch.Tensor) -> torch.Tensor:
        """
        Default approach: mapping from mk identifier to the mask used by
        reference_log_prob() and log_prob() methods for segregating elements in the
        tuple (theta_m, u_m) where theta_m are parameters used by log_prob()
        and u_m are the auxiliary variables to be evaluated by reference_log_prob().

        Args:
            mk (torch.Tensor): shape [N, <mk identifier length>] .

        Returns:
            torch.Tensor: shape [N, self.dim()],
        """
        # if mk.dim() != 2:
        #    raise ValueError("mk should be 2D")
        raise NotImplementedError("{__class__.__name__}.mk_to_mask() not implemented.")

    def reference_log_prob(self, mk: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Another method for computing a reference log probability.
        By default, it takes the 'off' entries of mk,
        multiplies them by log_prob under the reference distribution, sums, exponentiates, logs again.

        This is effectively:
            sum_dim( (1 - mk_to_mask(mk)) * log_prob(... ) ).exp().log()

        Which might be a fancy way of turning off certain dimensions of theta from the log-prob
        or adding them in. Implementation specifics can vary.

        Args:
            mk (torch.Tensor): shape [N, num_categories] (by default).
            theta (torch.Tensor): shape [N, #parameters], or sometimes [N,1] depending on usage.

        Returns:
            torch.Tensor: shape [N].
        """
        original_shape = theta.shape
        static_prob = (
            (
                (1 - self.mk_to_mask(mk))
                * self.referencedist.log_prob(theta.reshape(-1, 1)).view(original_shape)
            )
            .sum(dim=-1)
            .exp()
        )
        return static_prob.log()

    def log_prob(self, mk: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to compute the log probability (or log posterior).
        Must be overridden in child classes.
        """
        raise NotImplementedError()

    def num2tensor(self, number):
        # Determine the default data type from the number if dtype is not provided
        if isinstance(number, int):
            dtype = torch.int64  # Default torch integer type
        elif isinstance(number, float):
            dtype = self.dtype  # Default torch float type
        else:
            dtype = self.dtype

        # Create the tensor
        return torch.tensor(number, device=self.device, dtype=dtype)
