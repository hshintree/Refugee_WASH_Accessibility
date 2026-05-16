"""Linearized marginal accessibility gain for each (cell, candidate) pair.

The proposal models the post-intervention accessibility as

    A_i(x) ≈ A_i^(0) + Σ_j Δa_ij * x_j

where `Δa_ij` is the gain to demand cell `i` from adding one latrine of
unit capacity at candidate site `j`. This is the linearization that makes
the IP tractable; we recompute the full nonlinear E2SFCA after selection
to measure the gap.

Derivation. Adding one extra facility `j*` of capacity `n*` at distance
`d_{i,j*}` from cell `i` adds to step-2 of E2SFCA:

    Δ A_i  =  K(d_{i,j*}) * R_{j*}                     (1)
    R_{j*} =  n* / Σ_i K(d_{i,j*}) * p_i               (2)

The first-order linearization ignores the small change in `R_j` of the
*existing* facilities caused by extra demand sharing — which is zero here
because adding a *supply* facility does not change the existing supply
ratios. Step-2 is exactly linear in the new facility's K-column. So eq.
(1) IS exact for any single candidate's contribution. The approximation is
only that the union of K placed candidates is treated as Σ of singletons,
i.e. we ignore second-order competition between newly placed candidates
sharing the same demand cells. This is an over-estimate; the validator
quantifies the gap.

Capacity convention: a new latrine block has the same supply structure as
a typical 2022 latrine: we assume 1 all-gender stance per candidate by
default. The caller can scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from e2sfca import D0, SIGMA, kernel_matrix


@dataclass
class MarginalGains:
    K: np.ndarray  # (M_dem, N_cand) Gaussian kernel
    R: np.ndarray  # (N_cand,) facility supply ratio per candidate
    delta: np.ndarray  # (M_dem, N_cand) = K * R
    pop_used: np.ndarray  # (M_dem,) population used in denom

    def per_candidate_total_gain(self, weight: np.ndarray | None = None) -> np.ndarray:
        """Return shape (N_cand,) total population-weighted gain per
        candidate. Used by the greedy baseline and as the linear objective
        coefficient for the IP.
        """
        if weight is None:
            weight = np.ones(self.delta.shape[0])
        return (self.delta * weight[:, None]).sum(axis=0)


def euclid_distance(dem_xy: np.ndarray, cand_xy: np.ndarray) -> np.ndarray:
    dx = dem_xy[:, 0:1] - cand_xy[None, :, 0]
    dy = dem_xy[:, 1:2] - cand_xy[None, :, 1]
    return np.sqrt(dx * dx + dy * dy)


def compute(
    dem_xy: np.ndarray,
    pop: np.ndarray,
    cand_xy: np.ndarray,
    *,
    capacity: float = 1.0,
    d0: float = D0,
    sigma: float = SIGMA,
    distance_matrix: np.ndarray | None = None,
) -> MarginalGains:
    """Compute Δa_ij for every (i,j).

    Args:
        dem_xy: (M, 2) demand cell centroids.
        pop: (M,) demand population (used for the supply ratio denom).
        cand_xy: (N, 2) candidate facility positions.
        capacity: per-candidate facility capacity (default 1 stance).
        distance_matrix: optional pre-computed (M, N) distance matrix. When
            `None`, falls back to Euclidean in the coordinate system of
            `dem_xy`/`cand_xy`. Pass a network-distance matrix from
            `network_distance.network_distance_matrix` to match Ahn et al.
    """
    D = distance_matrix if distance_matrix is not None else euclid_distance(dem_xy, cand_xy)
    if D.shape != (len(dem_xy), len(cand_xy)):
        raise ValueError(
            f"distance_matrix shape {D.shape} ≠ ({len(dem_xy)}, {len(cand_xy)})"
        )
    K = kernel_matrix(D, d0=d0, sigma=sigma)
    denom = (K * pop[:, None]).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(denom > 0, capacity / denom, 0.0)
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    delta = K * R[None, :]
    return MarginalGains(K=K, R=R, delta=delta, pop_used=pop.copy())
