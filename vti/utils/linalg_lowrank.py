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
import warnings
from math import ceil, sqrt
from .math_helpers import atol_rtol, is_scalar

"""
Operations on low-rank factors.
"""


def factor_reduction_svd(lr_factor, rank, atol=0.0, rtol=None):
    """
    SVD-based rank reduction of a low-rank factor.

    If atol or rtol > 0, we trim to the non-zero singular values.

    TODO: Switch to the faster torch.svd_lowrank when appropriate.
    """
    # lr_factor: Tensor of shape (m, n)
    m, n = lr_factor.shape
    assert rank <= min(m, n), f"rank must be <= min({m}, {n})"

    # Perform SVD
    U, S, _ = torch.linalg.svd(lr_factor, full_matrices=False)
    # U: (m, k), S: (k,), where k = min(m, n)

    # Reduce U and S to the desired rank
    U = U[:, :rank]  # U: (m, rank)
    S = S[:rank]  # S: (rank,)

    # Get absolute tolerances
    atol, rtol = atol_rtol(lr_factor.dtype, m, atol=atol, rtol=rtol)

    if atol > 0.0 or rtol > 0.0:
        s1 = S.abs().max()
        nonzeroish = S.abs() > s1 * rtol + atol
        U = U[:, nonzeroish]  # U: (m, r'), r' <= rank
        S = S[nonzeroish]  # S: (r',)

    # Return the reduced factor
    result = U * S  # Broadcasting S over columns of U
    assert result.shape == (
        m,
        U.shape[1],
    ), f"Result shape mismatch, expected ({m}, {U.shape[1]}), got {result.shape}"
    return result  # Shape: (m, r')


def normalize_rows(A):
    """
    Normalize the rows of matrix A.

    A: Tensor of shape (n, m)
    Returns: Tensor of shape (n, m), with each row normalized.
    """
    assert A.dim() == 2, f"A must be a 2D tensor, got shape {A.shape}"
    # Compute norms of each row and avoid division by zero
    norms = torch.linalg.norm(A, dim=1, keepdim=True).clamp(min=1e-8)  # Shape: (n, 1)
    result = A / norms  # Broadcasting division over rows
    assert (
        result.shape == A.shape
    ), f"Result shape mismatch, expected {A.shape}, got {result.shape}"
    return result  # Shape: (n, m)


def factor_reduction_random(L, rank):
    """
    Take a random projection to cheaply approximate L @ L.T in expectation.

    L: Tensor of shape (n, m)
    Returns: Tensor of shape (n, rank)
    """
    assert L.dim() == 2, f"L must be a 2D tensor, got shape {L.shape}"
    n, m = L.shape

    # Create a random projection matrix
    projection = torch.randn(
        m, rank, dtype=L.dtype, device=L.device
    )  # Shape: (m, rank)
    result = L @ projection  # Shape: (n, rank)
    assert result.shape == (
        n,
        rank,
    ), f"Result shape mismatch, expected ({n}, {rank}), got {result.shape}"
    return result


def reduced_mean_dev(features, rank=None, reduction="random", normalize=True):
    """
    Compute the reduced mean and deviation of features.

    features: Tensor of shape (n, m)
    Returns:
    - mean: Tensor of shape (n,)
    - dev_reduced: Tensor of shape (n, k), where k <= rank
    """
    assert (
        features.dim() == 2
    ), f"Features must be a 2D tensor, got shape {features.shape}"
    n, m = features.shape

    if normalize:
        features = normalize_rows(features.T).T  # Normalize over samples
        # Now features still has shape (n, m)

    if rank is None:
        rank = ceil(sqrt(m))  # Rank is approximately sqrt(number of samples)

    mean, dev = mean_dev_from_ens(features)  # mean: (n,), dev: (n, m)
    assert mean.shape == (n,), f"Mean shape mismatch, expected ({n},), got {mean.shape}"
    assert dev.shape == (
        n,
        m,
    ), f"Deviation shape mismatch, expected ({n}, {m}), got {dev.shape}"

    if reduction == "random":
        dev_reduced = factor_reduction_random(dev, rank=rank)  # Shape: (n, rank)
    elif reduction == "svd":
        dev_reduced = factor_reduction_svd(dev, rank=rank)  # Shape: (n, r'), r' <= rank
    else:
        raise ValueError(f"Unknown reduction method '{reduction}'")

    assert dev_reduced.shape == (
        n,
        rank,
    ), f"Reduced dev shape mismatch, expected ({n}, {rank}), got {dev_reduced.shape}"
    return mean, dev_reduced  # mean: (n,), dev_reduced: (n, k)


def reduced_ens(features, **kwargs):
    mean, dev_reduced = reduced_mean_dev(features, **kwargs)
    # Now dev_reduced has shape (n, k)
    return ens_from_mean_dev(mean, dev_reduced)


def inv_capacitance_sqrt(
    sig2, lr_factor, sgn=1.0, max_rank=99999, atol=0.0, rtol=None, retain_all=False
):
    """
    sqrt(sig2 + sgn * lr_factor.T @ lr_factor)**(-1)

    Increases the diagonal until the matrix is invertible and returns the
    nearly-low-rank inverse, with an optional maximum rank.

    While the diagonal inflation is not a weird move for a covariance matrix, I
    know of no meaningful interpretation of it for a precision matrix with
    sgn=-1, so we fret about that.

    TODO: logarithmic speedup if we switch to torch.svd_lowrank.
    """
    atol, rtol = atol_rtol(lr_factor.dtype, lr_factor.shape[0], atol=atol, rtol=rtol)
    sig2 = torch.as_tensor(sig2, dtype=lr_factor.dtype)
    kappa2 = torch.reciprocal(sig2)
    if is_scalar(kappa2):
        capacitance = kappa2 * lr_factor.adjoint() @ lr_factor
    else:
        capacitance = lr_factor.adjoint() @ (kappa2[:, None] * lr_factor)
    d = torch.diagonal(capacitance)
    d += sgn
    Lam, Q = torch.linalg.eigh(sgn * capacitance)

    if not torch.all(Lam > 0.0):
        warnings.warn(f"matrix has unexpected sign {Lam}")

    # eigs are in ascending order
    if sgn == 1:
        s1 = Lam[-1]
        thresh = s1 * rtol + atol
        if retain_all:
            # inflate diagonal
            smallest = Lam[0]
            if smallest < atol:
                deficit = thresh - smallest
                Lam += deficit
                kappa2 += deficit
                warnings.warn(f"capacitance_inv inflated kappa2={kappa2} by {deficit}")

        else:
            # delete all the small eigs
            keep = Lam > thresh
            if not torch.all(keep):
                warnings.warn(f"capacitance_inv nulled eig={Lam[~keep]}")
            Lam = Lam[keep]
            Q = Q[:, keep]

        if max_rank is not None:
            Lam = Lam[-max_rank:]
            Q = Q[:, -max_rank:]

        # sqrt of inverse
        L = Q * (Lam ** (-0.5))
        return kappa2, L

    elif sgn == -1:
        # trickier!
        # we have negated the matrix to make the capacitance eigs positive.
        # Small eigs are important now
        # All eigs should sandwiched between 0 and |kappa2|;
        # what do we do if they bleed out of *both* sides of that interval?
        # let us be conservative and try not to touch anything
        s1 = Lam[-1]
        thresh = s1 * rtol + atol
        if retain_all:
            # deflate diagonal
            # inflate diagonal
            smallest = Lam[0]
            if smallest < thresh:
                deficit = thresh - smallest
                Lam += deficit
                kappa2 += deficit
                warnings.warn(f"capacitance_inv inflated kappa2={kappa2} by {deficit}")
        else:
            # delete all the small eigs
            keep = Lam > atol
            if any(~keep):
                warnings.warn(f"capacitance_inv nulled eig={Lam[~keep]}")
            Lam = Lam[keep]
            Q = Q[:, keep]

        # actually take sqrt of inverse eigenvals
        invLam = Lam ** (-0.5)

        if max_rank is not None:
            invLam = invLam[-max_rank:]
            Q = Q[:, -max_rank:]

        L = Q * invLam
        return kappa2, L

    else:
        raise ValueError("sgn must be +/-1")


def inv_lr(sig2, lr_factor, sgn=1.0, atol=0.0, rtol=None, retain_all=False):
    """
    return a new DiagonalPlusLowRank whose value is
    self^{-1} by Woodbury identities of tall skinny factors.
    """
    kappa2 = 1.0 / sig2
    kappa2_, L = inv_capacitance_sqrt(
        sig2, lr_factor, sgn, retain_all=retain_all, atol=atol, rtol=rtol
    )
    if torch.any(kappa2 != kappa2_):
        warnings.warn(f"capacitance_inv returned kappa2={kappa2_}!={kappa2}")
        kappa2 = kappa2_

    R = kappa2[:, None] * lr_factor @ L
    return kappa2, R


def mean_dev_from_ens(ens):
    """
    Compute the mean and deviation matrix of an ensemble.

    ens: Tensor of shape (n, m), where n is the number of variables and m is the number of samples.
    Returns:
    - m: Tensor of shape (n,)
    - dev: Tensor of shape (n, m)
    """
    assert ens.dim() == 2, f"ens must be a 2D tensor, got shape {ens.shape}"
    n, k = ens.shape

    # Compute mean over samples (columns)
    mean = ens.mean(dim=1)  # Shape: (n,)
    # Compute deviations
    dev = (ens - mean.unsqueeze(1)) / sqrt(k - 1)  # Shape: (n, m)
    assert mean.shape == (n,), f"Mean shape mismatch, expected ({n},), got {mean.shape}"
    assert dev.shape == (
        n,
        k,
    ), f"Deviation shape mismatch, expected ({n}, {k}), got {dev.shape}"
    return mean, dev


def ens_from_mean_dev(mean, dev):
    """
    Recover the ensemble from mean and deviation.

    mean: Tensor of shape (n,)
    dev: Tensor of shape (n, k)
    Returns: Tensor of shape (n, k)
    """
    assert mean.dim() == 1, f"Mean m must be a 1D tensor, got shape {mean.shape}"
    assert dev.dim() == 2, f"Deviation dev must be a 2D tensor, got shape {dev.shape}"
    n, k = dev.shape
    assert (
        mean.shape[0] == n
    ), f"Mean and deviation dimensions must match, got {mean.shape[0]} and {n}"
    assert k > 2, f"Number of samples k must be positive, got {k}"

    # Recover the ensemble
    ensemble = mean.unsqueeze(1) + dev * sqrt(k - 1)
    assert ensemble.shape == (
        n,
        k,
    ), f"Ensemble shape mismatch, expected ({n}, {k}), got {ensemble.shape}"
    return ensemble


def update_ensemble(L, sites, s_Y, y_t, L_mean=None, L_dev=None):
    """
    Update the ensemble L using the Matheron update, i.e., an ensemble update under observation y = H L for a selection matrix H.
    For efficiency, we allow the user to pass in the mean and deviation representation of the ensemble and return the updated ensemble in the same format.

    Parameters:
    - L: Tensor of shape (n, k), the ensemble before update.
    - sites: Tensor of shape (p,), indices for the selection matrix H.
    - s_Y: Scalar or tensor of shape (p,), the nugget variance for Y.
    - y_t: Tensor of shape (p,), observation at iteration t.

    Returns:
    - L: Tensor of shape (n, k), the updated ensemble.
    - L_mean: Tensor of shape (n,), mean of the updated ensemble.
    - L_dev: Tensor of shape (n, k), deviation of the updated ensemble.
    """

    # L: (n, k), where n is the number of variables, k is the ensemble size.
    n, k = L.shape
    assert L.dim() == 2, f"L must be a 2D tensor, got shape {L.shape}"

    # If mean and deviation are not provided, compute them.
    if L_mean is None or L_dev is None:
        L_mean, L_dev = mean_dev_from_ens(L)  # L_mean: (n,), L_dev: (n, k)
    else:
        assert L_mean.shape == (
            n,
        ), f"L_mean must be of shape ({n},), got {L_mean.shape}"
        assert L_dev.shape == (
            n,
            k,
        ), f"L_dev must be of shape ({n}, {k}), got {L_dev.shape}"

    # sites: (p,), indices for observed variables.
    assert sites.dim() == 1, f"sites must be a 1D tensor, got shape {sites.shape}"
    p = sites.shape[0]

    # y_t: (p,), observations at iteration t.
    assert y_t.shape == (p,), f"y_t must be of shape ({p},), got {y_t.shape}"

    # s_Y: Scalar or tensor of shape (p,), nugget variance for Y.
    assert torch.numel(s_Y) == 1 or s_Y.shape == (
        p,
    ), f"s_Y must be a scalar or of shape ({p},), got {s_Y.shape}"

    # Extract the ensemble members corresponding to observed variables.
    Y = L[sites, :]  # Y: (p, k)
    assert Y.shape == (p, k), f"Y must be of shape ({p}, {k}), got {Y.shape}"

    # Compute mean and deviation of Y.
    Y_mean, Y_dev = mean_dev_from_ens(Y)  # Y_mean: (p,), Y_dev: (p, k)
    assert Y_mean.shape == (p,), f"Y_mean must be of shape ({p},), got {Y_mean.shape}"
    assert Y_dev.shape == (
        p,
        k,
    ), f"Y_dev must be of shape ({p}, {k}), got {Y_dev.shape}"

    # Compute the covariance matrix S.
    # Y_cov = Y_dev @ Y_dev.T, where Y_cov: (p, p)
    Y_cov = Y_dev @ Y_dev.T  # Shape: (p, p)
    assert Y_cov.shape == (
        p,
        p,
    ), f"Y_cov must be of shape ({p}, {p}), got {Y_cov.shape}"

    # Add nugget variance to the covariance matrix.
    if torch.numel(s_Y) == 1:
        S = Y_cov + s_Y * torch.eye(p, device=L.device)  # Shape: (p, p)
    else:
        S = Y_cov + torch.diag(s_Y)  # Shape: (p, p)
    assert S.shape == (p, p), f"S must be of shape ({p}, {p}), got {S.shape}"

    # Compute the cross-covariance matrix c between L and Y.
    # c = L_dev @ Y_dev.T, c: (n, p)
    c = L_dev @ Y_dev.T  # Shape: (n, p)
    assert c.shape == (n, p), f"c must be of shape ({n}, {p}), got {c.shape}"

    # Solve the linear system S x = c^T for x to find the gain matrix G.
    try:
        # Use Cholesky decomposition if S is positive-definite.
        S_chol = torch.linalg.cholesky(S)  # Shape: (p, p)
        x = torch.cholesky_solve(c.T, S_chol)  # x: (p, n)
    except RuntimeError:
        # Use pseudo-inverse if S is not positive-definite.
        S_inv = torch.linalg.pinv(S)  # Shape: (p, p)
        x = S_inv @ c.T  # x: (p, n)
    assert x.shape == (p, n), f"x must be of shape ({p}, {n}), got {x.shape}"

    # Transpose x to get the gain matrix G.
    G = x.T  # G: (n, p)
    assert G.shape == (n, p), f"G must be of shape ({n}, {p}), got {G.shape}"

    # Compute the innovation vector delta_Y.
    # delta_Y = y_t.unsqueeze(1) - Y, delta_Y: (p, k)
    delta_Y = y_t.unsqueeze(1) - Y
    assert delta_Y.shape == (
        p,
        k,
    ), f"delta_Y must be of shape ({p}, {k}), got {delta_Y.shape}"

    # Update the ensemble L.
    # L = L + G @ delta_Y, where G: (n, p), delta_Y: (p, k), result: (n, k)
    delta_L = G @ delta_Y  # Shape: (n, k)
    assert delta_L.shape == (
        n,
        k,
    ), f"delta_L must be of shape ({n}, {k}), got {delta_L.shape}"

    L_updated = L + delta_L  # Updated L: (n, k)
    assert L_updated.shape == (
        n,
        k,
    ), f"L_updated must be of shape ({n}, {k}), got {L_updated.shape}"

    # Recompute mean and deviation for the updated ensemble.
    L_mean_updated, L_dev_updated = mean_dev_from_ens(L_updated)  # Shapes: (n,), (n, k)
    assert L_mean_updated.shape == (
        n,
    ), f"L_mean_updated must be of shape ({n},), got {L_mean_updated.shape}"
    assert L_dev_updated.shape == (
        n,
        k,
    ), f"L_dev_updated must be of shape ({n}, {k}), got {L_dev_updated.shape}"

    return L_updated, L_mean_updated, L_dev_updated
