"""Enhanced Two-Step Floating Catchment Area (E2SFCA).

Pure-Python port of `Refugee_WASH_Accessibility-main/Accessibility/utils/ACC.R`.

Parameters (must match Ahn et al. exactly for validation):
- `d0 = 1609 m` catchment radius
- Gaussian decay `exp(-d^2 / sigma^2)` with `sigma = 396 m`
- 50 m grid (handled by the demand-cell coordinates)

For an MxK distance matrix `D[i,j]` (demand i, supply j), per-cell pop `p_i`,
per-facility capacity `n_j`:

    K[i,j] = exp(-D[i,j]^2 / sigma^2) * (D[i,j] <= d0)
    R_j    = n_j / sum_i K[i,j] * p_i
    A_i    = sum_j K[i,j] * R_j
"""

from __future__ import annotations

import numpy as np

D0 = 1609.0
SIGMA = 396.0


def kernel_matrix(D: np.ndarray, d0: float = D0, sigma: float = SIGMA) -> np.ndarray:
    K = np.exp(-(D * D) / (sigma * sigma))
    K[D > d0] = 0.0
    return K


def e2sfca(
    p: np.ndarray,
    n: np.ndarray,
    D: np.ndarray,
    d0: float = D0,
    sigma: float = SIGMA,
) -> np.ndarray:
    """Vectorized E2SFCA matching the R reference.

    Args:
        p: shape (M,) demand population per cell.
        n: shape (K,) supply capacity per facility.
        D: shape (M, K) distance matrix in meters.
        d0: catchment radius in meters.
        sigma: Gaussian-decay sigma in meters.

    Returns:
        Accessibility A of shape (M,).
    """
    p = np.asarray(p, dtype=float)
    n = np.asarray(n, dtype=float)
    K = kernel_matrix(D, d0=d0, sigma=sigma)
    # Step 1: facility-side competition denominator. denom[j] = sum_i K[i,j] * p_i
    denom = (K * p[:, None]).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        Rj = np.where(denom > 0, n / denom, 0.0)
    Rj = np.nan_to_num(Rj, nan=0.0, posinf=0.0, neginf=0.0)
    # Step 2: demand-side aggregation
    A = (K * Rj[None, :]).sum(axis=1)
    return A


def euclidean_distance_matrix(
    dem_xy: np.ndarray, sup_xy: np.ndarray
) -> np.ndarray:
    """Euclidean distance matrix in the coordinate system of the inputs."""
    dem_xy = np.asarray(dem_xy, dtype=float)
    sup_xy = np.asarray(sup_xy, dtype=float)
    # broadcasting (M,1,2) - (1,K,2) -> (M,K,2)
    diff = dem_xy[:, None, :] - sup_xy[None, :, :]
    return np.sqrt((diff * diff).sum(axis=-1))


def e2sfca_delta(
    p: np.ndarray,
    n_old: np.ndarray,
    D_old: np.ndarray,
    n_new: np.ndarray,
    D_new: np.ndarray,
    d0: float = D0,
    sigma: float = SIGMA,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute baseline accessibility and post-placement accessibility when
    `n_new`/`D_new` adds candidate facilities to the supply set.

    Returns (A_baseline, A_post).
    """
    A_baseline = e2sfca(p, n_old, D_old, d0=d0, sigma=sigma)
    n_all = np.concatenate([n_old, n_new])
    D_all = np.concatenate([D_old, D_new], axis=1)
    A_post = e2sfca(p, n_all, D_all, d0=d0, sigma=sigma)
    return A_baseline, A_post
