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

"""
factory function for log_probs for diagnorm Gaussians
"""

import torch
from torch.distributions import Categorical
from vti.dgp import AbstractDGP
import random
from vti.utils.seed import set_seed, restore_seed
from torch.nn import functional as F
from vti.utils.kld import kld_probs
from vti.utils.math_helpers import safe_log, log_normal_pdf


class DiagNormGenerator(AbstractDGP):
    """
    A pure log likelihood for a Gaussian mixture with no data term.
    """

    def __init__(
        self,
        num_categories=33,
        dim_min=None,
        dim_max=None,
        seed=0,
        num_inputs=None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(device, dtype)
        self._num_categories = num_categories
        if num_inputs is None and dim_max is None:
            raise ValueError("Either num_inputs or dim_max must be specified")
        if num_inputs is None:
            num_inputs = dim_max
        if dim_min is None:
            dim_min = num_inputs // 2
        if dim_max is None:
            dim_max = num_inputs

        self.seed = seed
        self.dim_min = dim_min
        self.dim_max = dim_max
        self._generate()

    def num_categories(self):
        return self._num_categories

    def num_context_features(self):
        return self.num_categories()

    def num_inputs(self):
        return self.dim_max

    def mk_identifiers(self) -> torch.Tensor:
        """
        Returns an enumerated tensor of every model identifier.
        For the DiagNorm DGP, this method returns
        an identity matrix (size [num_categories x num_categories]),
        so each 'model index' is effectively a one-hot vector.

        This works if you interpret "model index" as a single integer in [0, num_categories).
        For subset enumeration, you'd typically override this in child classes.

        Returns:
            torch.Tensor: shape [num_categories, num_categories].
        """
        return torch.eye(self.num_categories(), device=self.device, dtype=self.dtype)

    def mk_cat_to_identifier(self, cat_samples: torch.Tensor) -> torch.Tensor:
        """
        Converts draws from a categorical distribution (integers) to one-hot vectors.

        Args:
            cat_samples (torch.Tensor): 1D integer tensor with values in [0, num_categories).

        Returns:
            torch.Tensor: shape [batch_size, num_categories], a one-hot encoding.
        """
        ids = F.one_hot(cat_samples, num_classes=self.num_categories()).to(
            dtype=self.dtype, device=self.device
        )
        return ids

    def mk_to_context(self, samples):
        """
        This method converts draws from the model sampler (mk dist) to a format that the flow context wants to see. By default, it is passthrough,
        BUT in this DGP the model sampler returns categorical variables as integers, and we want to convert to one hot vectors.
        """
        return self.mk_cat_to_identifier(samples)

    def generate_exponential_tensor(self, num_categories):
        # Generate an exponentially decaying tensor
        exp_tensor = torch.exp(
            0.75
            * torch.linspace(
                0,
                -num_categories + 1,
                num_categories,
                dtype=self.dtype,
                device=self.device,
            )
        )

        # Normalize the tensor to sum to 1
        normalized_tensor = exp_tensor / exp_tensor.sum()

        # Take a random permutation of the tensor
        permuted_tensor = normalized_tensor[torch.randperm(num_categories)]

        return permuted_tensor

    def printVTIResults(self, mk_probs):
        print("sorted model probabilities by target")
        sortidx = torch.argsort(self.modelprobs, dim=0)
        mkcomp = torch.column_stack(
            [
                torch.tensor(self.dims, dtype=self.dtype, device=self.device),
                self.modelprobs,
                mk_probs,
            ]
        )
        print("id\tdim\ttrue weight\tpredicted weight")
        for i in sortidx:
            # print("{i}\t{mkcomp[i,0]}\t{mkcomp[i,1]}\t{mkcomp[i,2]}")
            print(f"{i}\t{mkcomp[i,0]}\t{mkcomp[i,1]}\t{mkcomp[i,2]}")
        print(
            "KLD predicted to target model probabilities = ",
            kld_probs(self.modelprobs),
            mk_probs,
        )

    def _generate(self):
        oldrndstate = set_seed(self.seed)
        self.dims = [
            random.randint(self.dim_min, self.dim_max)
            for _ in range((self.num_categories()))
        ]
        # diag norm is scale and shift for each dimension.
        # create scale and shift matrices at num_categories x dim_max dimension
        # for variables that are not in models, set to static points.
        self.scalemat = torch.ones(
            (self.num_categories(), self.dim_max), device=self.device, dtype=self.dtype
        )
        self.shiftmat = torch.zeros(
            (self.num_categories(), self.dim_max), device=self.device, dtype=self.dtype
        )
        for i, d in enumerate(self.dims):
            self.scalemat[i, :d] = (
                F.softplus(torch.randn((d,))) + 0.25
            )  # 1e-10 # Add a small constant for numerical stability
            self.shiftmat[i, :d] = torch.randn((d,))
            # self.scalemat[i,:d] = 0.1*(F.softplus(torch.randn((d,))) + 0.25) #1e-10 # Add a small constant for numerical stability
            # self.shiftmat[i,:d] = 0.1*torch.randn((d,))*2
        # print("scale mat",self.scalemat)
        # print("shift mat",self.shiftmat)
        self.masks = torch.stack(
            [
                torch.concatenate(
                    [
                        torch.ones(d, device=self.device, dtype=self.dtype),
                        torch.zeros(
                            self.dim_max - d, device=self.device, dtype=self.dtype
                        ),
                    ]
                )
                for d in self.dims
            ]
        )
        self.modelprobs = self.generate_exponential_tensor(self.num_categories())
        restore_seed(oldrndstate)

    def mk_to_mask(self, mk: torch.Tensor) -> torch.Tensor:
        """
        Interpret 'mk' as one-hot vectors in shape [N, num_categories],
        then do a matrix multiply with self.masks.
        This is different from subset-based approaches in some child classes.

        By default, we do:
            (mk) [N, num_categories] @ self.masks [num_categories, (some dimension)]
        so you must define 'self.masks' somewhere or override this method.

        Args:
            mk (torch.Tensor): shape [N, num_categories] one-hot indicators.

        Returns:
            torch.Tensor: shape [N, ???], a masked version of parameters.

        """
        if mk.dim() != 2:
            raise ValueError("mk should be 2D")
        return (
            torch.as_tensor(mk, dtype=self.dtype).view(-1, self.num_categories())
            @ self.masks
        )

    # TODO DISCUSS below mk_cat_to_mask and all_masks may not be necessary.
    def mk_cat_to_mask(self, mk_cat: torch.Tensor) -> torch.Tensor:
        """
        Another approach: directly index self.masks using mk_cat.

        Args:
            mk_cat (torch.Tensor): shape [N], integer indexes in [0, num_categories).

        Returns:
            torch.Tensor: shape [N, ???], a masked version from self.masks.
        """
        return self.masks[mk_cat]

    def all_masks(self) -> torch.Tensor:
        """
        Returns all possible masks by enumerating mk_cat in [0, num_categories).

        Returns:
            torch.Tensor: shape [num_categories, ???], each row is a mask.
        """
        return self.mk_cat_to_mask(torch.arange(self.num_categories()))

    def reference_log_prob(self, mk, theta):
        original_shape = theta.shape
        return (
            (1 - self.mk_to_mask(mk))
            * self.referencedist.log_prob(theta.view(-1, 1)).view(original_shape)
        ).sum(dim=-1)

    def log_prob(self, mk, theta):
        """
        return the log prob of the mixture
        """
        # mk_p = mk.view(-1,self.num_categories()) * self.modelprobs.view(1,self.num_categories) # batch_size x num_categories
        log_mk_p = safe_log(
            mk.view(-1, self.num_categories())
        ) + self.modelprobs.log().view(1, self.num_categories())
        # log_mk_p = torch.log(mk.view(-1,self.num_categories())) + self.modelprobs.log().view(1,self.num_categories())
        # log prob of each model is norm(theta;shift,scale^2)
        condtarget_lp_vars = log_normal_pdf(
            theta.unsqueeze(1), self.shiftmat, self.scalemat
        )  # batch_size x num_categories x num_params
        condtarget_lp = (
            self.mk_to_mask(mk).unsqueeze(1) * condtarget_lp_vars
        )  # batch_size x num_categories x num_params
        condtarget_lp = condtarget_lp.sum(dim=-1)  # batch_size x num_categories
        # target_log_prob = (mk_p.log() + condtarget_lp).logsumexp(dim=-1)
        target_log_prob = (log_mk_p + condtarget_lp).logsumexp(dim=-1)
        reference_log_prob = self.reference_log_prob(mk, theta)

        return target_log_prob + reference_log_prob

    def sample_and_log_prob_qmk(self, batch_size):
        catdist = Categorical(probs=self.modelprobs)
        samples = catdist.sample((batch_size,))
        log_prob = catdist.log_prob(samples)
        return samples, log_prob

    def sample_q(self, mk):
        # draw iid gaussian, then shift and scale according to mk.
        refvars = torch.randn((mk.shape[0], self.dim_max))
        scale = mk @ self.scalemat
        shift = mk @ self.shiftmat
        return scale * refvars + shift

    def log_prob_q(self, mk, theta, log_mk_p):
        """
        A test variational density where we fix it to the target parameters, but allow mk probs to be specified.
        """
        condtarget_lp_vars = log_normal_pdf(
            theta.unsqueeze(1), self.shiftmat, self.scalemat
        )  # batch_size x num_categories x num_params
        condtarget_lp = (
            self.mk_to_mask(mk).unsqueeze(1) * condtarget_lp_vars
        )  # batch_size x num_categories x num_params
        condtarget_lp = condtarget_lp.sum(dim=-1)  # batch_size x num_categories
        target_log_prob = (log_mk_p + condtarget_lp).logsumexp(dim=-1)
        reference_log_prob = self.reference_log_prob(mk, theta)

        return target_log_prob + reference_log_prob
