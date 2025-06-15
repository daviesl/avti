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
from torch import nn
from vti.utils.torch_nn_helpers import extract_adam_step_moments
from vti.utils.linalg_lowrank import (
    update_ensemble,
    ens_from_mean_dev,
    mean_dev_from_ens,
    inv_lr,
)
from warnings import warn
from torch.distributions import LowRankMultivariateNormal
import logging


class ModelSurrogate(nn.Module):
    """
    A surrogate over some scalar parameter of the class.
    This will be mapped into useful things like probabilities or logits by a sampler class
    """

    def __init__(self, num_categories=1, device=None, dtype=None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.num_categories = num_categories

    def _num_categories(self):
        return self.num_categories

    def utility_UCB(self):
        raise NotImplementedError("utility_UCB not implemented")

    def utility_Thompson(self):
        raise NotImplementedError("utility_UCB not implemented")

    def action_logits(self):
        return self.utility_UCB()

    def logits(self):
        """
        Falls back to action logits per default
        """
        return self.action_logits()


class GaussianSurrogate(ModelSurrogate):
    """
    An abstract class for a surrogate over class (pseudo-)logits, with a Gaussian prior and gaussian lieklihood.
    """

    def __init__(self, num_categories=1, device=None, dtype=None):
        super().__init__(num_categories=num_categories, device=device, dtype=dtype)

    def mean(self):
        raise NotImplementedError("GaussianSurrogate is abstract")

    def diag_variance(self):
        raise NotImplementedError("GaussianSurrogate is abstract")

    def sd(self):
        """
        sd is not defined for general multivariates,
        but for utility we might want sqrt of all marginal variances,
        which is the sqrt of the variance diagonal.
        """
        return self.diag_variance().sqrt()

    def debug_log(self):
        raise NotImplementedError("GaussianSurrogate is abstract")

    def utility_UCB(self):
        """
        Upper confidence bound utility.
        """
        return self.mean() + 2 * (self.sd())

    def utility_Thompson(self):
        """
        This looks more like a sample from the Thompson-weighted categorical than the actual utility?
        """
        return self.mean() + torch.randn(self.num_categories) * 2 * self.sd()

    def observe(self, idx_tensor, x):
        raise NotImplementedError("GaussianSurrogate is abstract")

    def evolve(self, cat_samples, ell, optimizer, loss):
        """
        update the prior based on the state changes of the optimisation.

        Not all the hacks here are equally principled.
        """
        raise NotImplementedError("GaussianSurrogate is abstract")


class DiagonalGaussianSurrogate(GaussianSurrogate):
    """
    A surrogate over class (pseudo-)logits, with a Gaussian prior and diagonal covariance,
    and a likelihood that is a Gaussian with diagonal covariance.
    """

    def __init__(
        self,
        num_categories=1,
        prior_mean=0.0,
        prior_diag_variance=1e2,  # actually just diagonal variance
        obs_variance=1.0,
        f_coupling=1e2,  # if > 0, apply optimizer coupling to prior
        obs_beta=0.99,  # update observation precision from residuals
        diffuse_prior=1.0,  # if <1.0, inflate variance without exploiting side information
        max_entropy_gain=0.0,
        device=None,
        dtype=None,
    ):
        super().__init__(num_categories=num_categories, device=device, dtype=dtype)

        _prior_mean = torch.as_tensor(prior_mean, device=device, dtype=dtype)
        if _prior_mean.dim() == 0:
            _prior_mean = torch.full(
                (num_categories,), _prior_mean.item(), device=device, dtype=dtype
            )
        self.register_buffer("_prior_mean", _prior_mean)

        # Diagonal of prior covariance
        _prior_diag_variance_diag = torch.as_tensor(
            prior_diag_variance, device=device, dtype=dtype
        )
        if _prior_diag_variance_diag.dim() == 0:
            _prior_diag_variance_diag = torch.full(
                (num_categories,),
                _prior_diag_variance_diag.item(),
                device=device,
                dtype=dtype,
            )
        self.register_buffer("_prior_diag_variance_diag", _prior_diag_variance_diag)

        _obs_variance = torch.as_tensor(obs_variance, device=device, dtype=dtype)
        self.register_buffer("_obs_variance", _obs_variance)

        # Evolution parameters
        # make all these rintosors as well for consistency
        f_coupling = torch.as_tensor(f_coupling, dtype=dtype, device=device)
        self.register_buffer("f_coupling", f_coupling)
        obs_beta = torch.as_tensor(obs_beta, dtype=dtype, device=device)
        self.register_buffer("_obs_beta", obs_beta)
        diffuse_prior = torch.as_tensor(diffuse_prior, dtype=dtype, device=device)
        self.register_buffer("diffuse_prior", diffuse_prior)

        # Entropy gain constraint
        _max_entropy_gain = torch.as_tensor(
            max_entropy_gain, dtype=dtype, device=device
        )
        self.register_buffer("_max_entropy_gain", _max_entropy_gain)

        # Evolution diagnostics
        self._qll = 0.0

    def mean(self):
        return self._prior_mean

    def diag_variance(self):
        return self._prior_diag_variance_diag

    def debug_log(self):
        logging.debug(f"prior_mean={self.mean()}")
        logging.debug(f"prior_diag_variance={self.diag_variance()}")
        logging.debug(f"obs_variance={self._obs_variance.item()}")
        logging.debug(f"qll={self._qll}")

    def utility_UCB(self):
        return self.mean() + 2 * self.sd()

    def utility_Thompson(self):
        """
        This looks more like a sample from the Thompson-weighted categorical than the actual utility?
        """
        return self.mean() + torch.randn(self.num_categories) * 2 * self.sd()

    def observe(self, idx_tensor, x):
        for i, idx in enumerate(idx_tensor):
            m = self._prior_mean[idx]
            v = self._prior_diag_variance_diag[idx]

            if self._max_entropy_gain > 0.0:
                # Compute minimal obs_variance to approximately limit entropy gain <= max_entropy_gain
                min_obs_variance = v / (torch.exp(2 * self._max_entropy_gain) - 1)
                # Ensure obs_variance is at least min_obs_variance
                obs_variance = torch.maximum(self._obs_variance, min_obs_variance)
            else:
                obs_variance = self._obs_variance

            posterior_variance = 1.0 / (1.0 / v + 1.0 / obs_variance)
            posterior_mean = posterior_variance * (m / v + x[i] / obs_variance)

            self._prior_mean[idx] = posterior_mean
            self._prior_diag_variance_diag[idx] = posterior_variance

    def evolve(self, cat_samples, ell, optimizer, loss):
        """
        update the prior based on the state changes of the optimisation.

        Not all the hacks here are equally principled.
        """
        # Naive adaptation of prior:
        # empirical update of observation precision from residual, exponentially-weighted average
        if self._obs_beta <= 1:
            residual_precision = 1.0 / (self.mean()[cat_samples] - ell).detach().var()
            # weight by the number of sites observed, since this is a shared parameter over all sites
            new_weight = torch.unique(cat_samples).numel() / self.num_categories
            self._obs_variance = (
                self._obs_beta * self._obs_variance
                + (1 - self._obs_beta) * residual_precision * new_weight
            )
        # Diffuse everything, at a rate determined by how often we visit everything;
        # This is too aggressive.
        if self.diffuse_prior < 1.0:
            self._prior_diag_variance_diag /= self.diffuse_prior ** (
                cat_samples.numel() / self.num_categories
            )
        # Linearized dynamics update
        if self.f_coupling > 0:
            step_vector, _, _ = extract_adam_step_moments(optimizer)
            # In Adam the step is scaled by the variance, so the variance is the mean  of the step itself squared.
            qll = self.f_coupling * (step_vector**2).mean().item()

            self._prior_diag_variance_diag += qll
            # For diagnostics
            self._qll = qll


class EnsembleGaussianSurrogate(GaussianSurrogate):
    """
    A surrogate sample with a Gaussian prior
    and a likelihood that is a Gaussian with diagonal covariance, but the kernel is induced by an ensemble of samples
    """

    def __init__(
        self,
        num_categories=1,
        prior_mean=0.0,  # imposed on the ensemble
        prior_diag_variance=1e10,
        prior_ensemble=None,  # should be a matrix
        prior_dev=None,  # alternate spec for prior_ensemble because I got confused
        lr_variance_scale=1e2,  # rescale deviance
        obs_variance=1e2,
        f_coupling=1.0,  # if > 0, apply optimizer coupling to prior
        obs_beta=0.99,  # update observation precision from residuals
        max_entropy_gain=0.0,  # per obs
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(num_categories=num_categories, device=device, dtype=dtype)
        _obs_variance = torch.as_tensor(obs_variance, device=device, dtype=dtype)
        self.register_buffer("_obs_variance", _obs_variance)

        # ok we have either a basis or an ensemble
        if prior_dev is None and prior_ensemble is None:
            raise ValueError("prior_dev or prior_ensemble must be provided")
        if prior_ensemble is not None:
            prior_mean, prior_dev = mean_dev_from_ens(prior_ensemble)

        if (
            prior_dev is None
            or prior_dev.size(0) != num_categories
            or prior_dev.size(1) < 2
        ):
            raise ValueError(
                f"prior_dev must be a matrix of shape (num_categories, num_samples) but got {prior_dev.shape}"
            )

        rank = prior_dev.size(1)

        _prior_mean = torch.as_tensor(prior_mean, device=device, dtype=dtype)
        _prior_mean = torch.zeros((num_categories,), device=device, dtype=dtype)
        self.register_buffer("_prior_mean", _prior_mean)
        _prior_mean[:] = prior_mean

        _prior_nugget = torch.as_tensor(prior_mean, device=device, dtype=dtype)
        _prior_nugget = torch.zeros((num_categories,), device=device, dtype=dtype)
        self.register_buffer("_prior_nugget", _prior_nugget)
        _prior_nugget[:] = prior_diag_variance

        _prior_dev = torch.zeros((num_categories, rank), device=device, dtype=dtype)
        self.register_buffer("_prior_dev", _prior_dev)
        _prior_dev[:] = prior_dev * lr_variance_scale

        # evolution parameters
        self.f_coupling = f_coupling
        self._obs_beta = obs_beta

        # evolution diagnostics
        self._qll = 0.0

        max_entropy_gain = torch.as_tensor(max_entropy_gain, dtype=dtype, device=device)
        self.register_buffer("max_entropy_gain", max_entropy_gain)

        # warn user about unused kwargs
        if kwargs:
            warn(f"Unused keyword arguments: {kwargs}")

    def get_ensemble_prior(self):
        return ens_from_mean_dev(self._prior_mean, self._prior_dev)

    def get_mean_dev_prior(self):
        return self._prior_mean, self._prior_dev

    def set_ensemble_prior(self, ens):
        """
        receive a mean, and a variance, diag(sig2) + L @ L.T
        Convert these to information vec and precision lam = diag(kappa) - R @ R.T to store.
        """
        mean, dev = mean_dev_from_ens(ens)
        self.set_mean_dev_prior(mean, dev)

    def set_mean_dev_prior(self, mean, dev):
        self._prior_mean[:] = mean
        self._prior_dev[:] = dev

    def mean(self):
        return self._prior_mean

    def diag_variance(self):
        """
        This is the diagonal of the covariance matrix, including the low rank bases and the nugget term
        """
        return self.get_mean_and_diag_variance()[1]

    def get_mean_and_diag_variance(self):
        mean, dev = self.get_mean_dev_prior()
        lr_diag = torch.sum(dev * dev, dim=-1)
        return mean, self._prior_nugget + lr_diag

    def utility_UCB(self):
        """
        Upper confidence bound utility.
        """
        mean, vdiag = self.get_mean_and_diag_variance()
        return mean + torch.sqrt(vdiag) * 2

    def debug_log(self):
        logging.debug(f"prior_mean={self._prior_mean}")
        logging.debug(f"prior_dev={self._prior_dev}")
        logging.debug(f"prior_nugget={self._prior_nugget}")
        logging.debug(f"obs_variance={self._obs_variance.item()}")
        logging.debug(f"qll={self._qll}")

    def observe(self, idx_tensor, x):
        """
        Gaussian posterior update for a vector of observations.

        Parameters:
        - idx_tensor: Tensor of shape (p,), indices of observed variables.
        - x: Tensor of shape (p,), observations corresponding to idx_tensor.
        """
        # Retrieve the ensemble prior
        ens = self.get_ensemble_prior()  # ens: Tensor of shape (n, M)
        assert ens.dim() == 2, f"ens must be a 2D tensor, got shape {ens.shape}"
        n, M = ens.shape  # n: number of variables, M: ensemble size

        # Get mean and deviation of the prior
        mean, dev = self.get_mean_dev_prior()  # mean: (n,), dev: (n, M)
        assert mean.shape == (n,), f"mean must be of shape ({n},), got {mean.shape}"
        assert dev.shape == (n, M), f"dev must be of shape ({n}, {M}), got {dev.shape}"

        # Ensure idx_tensor and x have matching shapes
        assert (
            idx_tensor.dim() == 1
        ), f"idx_tensor must be a 1D tensor, got shape {idx_tensor.shape}"
        p = idx_tensor.shape[0]  # p: number of observations
        assert x.shape == (p,), f"x must be of shape ({p},), got {x.shape}"

        # Get the max entropy gain parameter
        max_entropy_gain = self.max_entropy_gain  # Scalar tensor
        assert isinstance(
            max_entropy_gain, torch.Tensor
        ), f"max_entropy_gain must be a tensor, got {type(max_entropy_gain)}"

        # Compute the current entropy of the prior distribution
        curr_entropy = LowRankMultivariateNormal(
            mean, dev, self._prior_nugget
        ).entropy()
        # dev.T: (M, n), LowRankMultivariateNormal expects cov_factor of shape (n, rank)
        # Since M might be large, ensure that computations are feasible

        # Compute the observation precision
        obs_precision = 1.0 / self._obs_variance  # Scalar or tensor
        s_Y = self._obs_variance  # Nugget variance for Y
        assert torch.numel(s_Y) == 1 or s_Y.shape == (
            p,
        ), f"s_Y must be scalar or of shape ({p},), got {s_Y.shape}"

        # Approximate entropy capping
        # Update the ensemble with the observations
        new_ens, new_mean, new_dev = update_ensemble(ens, idx_tensor, s_Y, x)
        # new_ens: (n, M), new_mean: (n,), new_dev: (n, M)
        assert new_ens.shape == (
            n,
            M,
        ), f"new_ens must be of shape ({n}, {M}), got {new_ens.shape}"
        assert new_mean.shape == (
            n,
        ), f"new_mean must be of shape ({n},), got {new_mean.shape}"
        assert new_dev.shape == (
            n,
            M,
        ), f"new_dev must be of shape ({n}, {M}), got {new_dev.shape}"

        # Compute the entropy after the update
        new_entropy = LowRankMultivariateNormal(
            new_mean, new_dev, self._prior_nugget
        ).entropy()

        # Check if the entropy gain exceeds the maximum allowed
        entropy_gain = new_entropy - curr_entropy
        if max_entropy_gain > 0.0 and entropy_gain > max_entropy_gain:
            # Adjust the observation precision to limit entropy gain
            obs_precision_adjustment = entropy_gain / max_entropy_gain
            obs_precision /= obs_precision_adjustment

            # Recompute s_Y with adjusted obs_precision
            s_Y = 1.0 / obs_precision
            assert torch.numel(s_Y) == 1 or s_Y.shape == (
                p,
            ), f"s_Y must be scalar or of shape ({p},), got {s_Y.shape}"

            # Update the ensemble again with adjusted s_Y
            new_ens, new_mean, new_dev = update_ensemble(ens, idx_tensor, s_Y, x)
            # Verify shapes
            assert new_ens.shape == (
                n,
                M,
            ), f"new_ens must be of shape ({n}, {M}), got {new_ens.shape}"
            assert new_mean.shape == (
                n,
            ), f"new_mean must be of shape ({n},), got {new_mean.shape}"
            assert new_dev.shape == (
                n,
                M,
            ), f"new_dev must be of shape ({n}, {M}), got {new_dev.shape}"

            # Compute the entropy after the adjusted update
            newer_entropy = LowRankMultivariateNormal(
                new_mean, new_dev, self._prior_nugget
            ).entropy()
            new_entropy_gain = newer_entropy - curr_entropy

            if new_entropy_gain > max_entropy_gain:
                warn(
                    f"Entropy gain constraint not satisfied {new_entropy_gain} > {max_entropy_gain}"
                )

        # Update the prior with the new mean and deviation
        self.set_mean_dev_prior(new_mean, new_dev)

    def evolve(self, cat_samples, ell, optimizer, loss):
        """
        update the prior based on the state changes of the optimisation.
        """
        mean, dev = self.get_mean_dev_prior()
        # Naive adaptation of prior:
        # empirical update of observation precision from residual, exponentially-weighted average
        if self._obs_beta <= 1:
            residual_precision = 1.0 / (mean[cat_samples] - ell).detach().var()
            # weight by the number of sites observed, since this is a shared parameter over all sites
            new_weight = torch.unique(cat_samples).numel() / self.num_categories
            self._obs_variance = (
                self._obs_beta * self._obs_variance
                + (1 - self._obs_beta) * residual_precision * new_weight
            )

        # Linearized dynamics update
        if self.f_coupling > 0:
            step_vector, _, _ = extract_adam_step_moments(optimizer)
            # In Adam the step is scaled by the variance, so the variance is the mean  of the step itself squared.
            qll = self.f_coupling * (step_vector**2).mean().item()

            self._prior_nugget += qll
            # For diagnostics
            self._qll = qll


class LowRankGaussianSurrogate(GaussianSurrogate):
    """
    A surrogate sample with a Gaussian prior
    and a likelihood that is a Gaussian with diagonal covariance, but the kernel is low rank.

    Low rank factors give model covariance as L @ L.T + sig2 * I
    """

    def __init__(
        self,
        num_categories=1,
        prior_mean=0.0,
        prior_diag_variance=1e10,
        prior_dev=0.0,  # should be a matrix
        lr_variance_scale=1e2,  # off-diagonal variance scale
        obs_variance=1e2,
        f_coupling=1.0,  # if > 0, apply optimizer coupling to prior
        obs_beta=0.99,  # update observation precision from residuals
        diffuse_prior=1.0,  # if <1.0, inflate variance without exploiting side information
        device=None,
        dtype=None,
    ):
        super().__init__(num_categories=num_categories, device=device, dtype=dtype)
        # information vector = \Sigma^{-1}\mu
        _eta = torch.as_tensor(prior_mean, device=device, dtype=dtype)
        _eta = torch.zeros((num_categories,), device=device, dtype=dtype)
        self.register_buffer("_eta", _eta)
        # _lam = diag(kappa) - R @ R.T
        _kappa = torch.zeros((num_categories,), device=device, dtype=dtype)
        self.register_buffer("_kappa", _kappa)
        prior_dev = torch.atleast_2d(
            torch.as_tensor(prior_dev, device=device, dtype=dtype)
        )
        _R = torch.zeros(
            (num_categories, prior_dev.shape[1]), device=device, dtype=dtype
        )
        self.register_buffer("_R", _R)
        _obs_variance = torch.as_tensor(obs_variance, device=device, dtype=dtype)
        self.register_buffer("_obs_variance", _obs_variance)

        # Initialize the prior
        self.set_lr_moments_prior(
            prior_mean,
            prior_diag_variance,
            prior_dev
            * torch.sqrt(
                torch.as_tensor(lr_variance_scale, device=device, dtype=dtype)
            ),
        )

        # evolution parameters
        self.f_coupling = f_coupling
        self._obs_beta = obs_beta
        self.diffuse_prior = diffuse_prior

        # evolution diagnostics
        self._qll = 0.0

    def get_lr_canonical_prior(self):
        return self._eta, self._kappa, self._R

    def get_lr_moments_prior(self):
        eta, kappa, R = self.get_lr_canonical_prior()
        sig2, L = inv_lr(kappa, R, sgn=-1.0)
        mean = sig2 * eta
        mean += L @ (L.T @ eta)
        return mean, sig2, L

    def set_lr_canonical_prior(self, eta, kappa, R):
        """
        receive an information vector and a precision matrix lam = diag(kappa) - R @ R.T.
        """
        self._eta[:] = torch.as_tensor(eta, device=self.device, dtype=self.dtype)
        self._kappa[:] = torch.as_tensor(kappa, device=self.device, dtype=self.dtype)
        self._R[:] = torch.as_tensor(R, device=self.device, dtype=self.dtype)

    def set_lr_moments_prior(self, mean, sig2, L):
        """
        receive a mean, and a variance, diag(sig2) + L @ L.T
        Convert these to information vec and precision lam = diag(kappa) - R @ R.T to store.
        """
        # first a lot of casting because things might be floats or tensors
        mean = torch.as_tensor(mean, device=self.device, dtype=self.dtype)
        mean = torch.atleast_1d(mean)
        mean = mean.expand(self.num_categories) if mean.size(0) == 1 else mean
        sig2 = torch.as_tensor(sig2, device=self.device, dtype=self.dtype)
        sig2 = torch.atleast_1d(sig2)
        sig2 = sig2.expand(self.num_categories) if sig2.size(0) == 1 else sig2
        L = torch.as_tensor(L, device=self.device, dtype=self.dtype)
        L = torch.atleast_2d(L)
        L = L.expand(self.num_categories, -1) if L.size(0) == 1 else L
        # actual calcs. First Lam, the precision, lam = diag(kappa) - R @ R.T
        kappa, R = inv_lr(sig2, L, sgn=1.0)
        # now the information vector, eta = Variance ** (-1) mean
        # = sig2 * mean - R @ (R.T @ mean)
        eta = mean * kappa
        eta -= R @ (R.T @ mean)
        self.set_lr_canonical_prior(eta, kappa, R)

    def mean(self):
        return self.get_lr_moments_prior()[0]

    def diag_variance(self):
        """
        This is the diagonal of the covariance matrix, including the low rank bases.
        """
        return self.get_mean_and_diag_variance()[1]

    def get_mean_and_diag_variance(self):
        mean, sig2, L = self.get_lr_moments_prior()
        # This needs to be clipped to avoid negative variances
        lr_diag = torch.sum(L * L, dim=-1)
        return mean, sig2 + lr_diag

    def utility_UCB(self):
        """
        Upper confidence bound utility.
        """
        mean, vdiag = self.get_mean_and_diag_variance()
        return mean + torch.sqrt(vdiag) * 2

    def debug_log(self):
        logging.debug(f"prior_eta={self._eta}")
        logging.debug(f"prior_kappa={self._kappa}")
        logging.debug(f"prior_R={self._R}")
        logging.debug(f"obs_variance={self._obs_variance}".item())
        logging.debug(f"qll={self._qll}")

    def observe(self, idx_tensor, x):
        """
        Gaussian posterior update for a vector of observations.
        """
        eta, kappa, R = self.get_lr_canonical_prior()
        # Note that since the observations are themselves diagonal,
        # there is no direct update to the off-diagonal terms.
        eta.scatter_add_(0, idx_tensor, (x / self._obs_variance))
        kappa.scatter_add_(
            0, idx_tensor, (1.0 / self._obs_variance).expand(idx_tensor.size())
        )
        # probably redundant because we mutated these in place already
        self.set_lr_canonical_prior(eta, kappa, R)

    def evolve(self, cat_samples, ell, optimizer, loss):
        """
        update the prior based on the state changes of the optimisation.

        Not all the hacks here are equally principled.
        """
        # Naive adaptation of prior:
        # empirical update of observation precision from residual, exponentially-weighted average
        mean, sig2, L = self.get_lr_moments_prior()
        if self._obs_beta <= 1:
            residual_precision = 1.0 / (mean[cat_samples] - ell).detach().var()
            # weight by the number of sites observed, since this is a shared parameter over all sites
            new_weight = torch.unique(cat_samples).numel() / self.num_categories
            self._obs_variance = (
                self._obs_beta * self._obs_variance
                + (1 - self._obs_beta) * residual_precision * new_weight
            )
        # Diffuse everything, at a rate determined by how often we visit everything;
        # This is too aggressive.
        if self.diffuse_prior < 1.0:
            sig2 /= self.diffuse_prior ** (cat_samples.numel() / self.num_categories)
        # Linearized dynamics update
        if self.f_coupling > 0:
            step_vector, _, _ = extract_adam_step_moments(optimizer)
            # In Adam the step is scaled by the variance, so the variance is the mean  of the step itself squared.
            qll = self.f_coupling * (step_vector**2).mean().item()

            sig2 += qll
            # For diagnostics
            self._qll = qll
        self.set_lr_moments_prior(mean, sig2, L)
