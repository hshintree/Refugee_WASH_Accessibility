"""Validate selected sites by recomputing the full E2SFCA.

After the linearized IP picks K candidate indices, we append those
positions (with capacity 1 stance each) to the existing latrine supply
and rerun the unmodified E2SFCA over the full demand grid. We then report:

- linearized objective vs full-recompute objective (the "linearization
  gap")
- per-cell accessibility before vs after
- the same metrics restricted to Camp 22 cells.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from e2sfca import D0, SIGMA, e2sfca, euclidean_distance_matrix


@dataclass
class RecomputeResult:
    A_before_t: np.ndarray  # global cells, total
    A_after_t: np.ndarray
    A_before_f: np.ndarray
    A_after_f: np.ndarray


def recompute_with_new_sites(
    dem_xy: np.ndarray,
    pop: np.ndarray,
    pop_f: np.ndarray,
    sup_xy: np.ndarray,
    sup_capacity_t: np.ndarray,
    sup_capacity_f: np.ndarray,
    new_xy: np.ndarray,
    *,
    new_capacity_t: float = 1.0,
    new_capacity_f: float = 0.5,
    d0: float = D0,
    sigma: float = SIGMA,
) -> RecomputeResult:
    """Run E2SFCA before and after appending `new_xy` to the supply set.

    `new_capacity_t` is the total stance count contributed by each new
    site; `new_capacity_f` is the female-effective count (e.g. 0.5 if a
    block of 2 stances is split 1 male / 1 female, or for the Scenario 2
    safety penalty applied to all-gender stances).
    """
    D_old = euclidean_distance_matrix(dem_xy, sup_xy)
    A_before_t = e2sfca(pop, sup_capacity_t, D_old, d0=d0, sigma=sigma)
    A_before_f = e2sfca(pop_f, sup_capacity_f, D_old, d0=d0, sigma=sigma)

    if len(new_xy):
        D_new = euclidean_distance_matrix(dem_xy, new_xy)
        D_all = np.concatenate([D_old, D_new], axis=1)
        n_t_new = np.full(len(new_xy), new_capacity_t)
        n_f_new = np.full(len(new_xy), new_capacity_f)
        n_t_all = np.concatenate([sup_capacity_t, n_t_new])
        n_f_all = np.concatenate([sup_capacity_f, n_f_new])
        A_after_t = e2sfca(pop, n_t_all, D_all, d0=d0, sigma=sigma)
        A_after_f = e2sfca(pop_f, n_f_all, D_all, d0=d0, sigma=sigma)
    else:
        A_after_t = A_before_t.copy()
        A_after_f = A_before_f.copy()

    return RecomputeResult(
        A_before_t=A_before_t,
        A_after_t=A_after_t,
        A_before_f=A_before_f,
        A_after_f=A_after_f,
    )


def metrics_table(
    A_before_t: np.ndarray,
    A_after_t: np.ndarray,
    A_before_f: np.ndarray,
    A_after_f: np.ndarray,
    pop: np.ndarray,
    pop_f: np.ndarray,
    *,
    sphere_threshold: float = 1.0 / 20.0,
    bottom_decile_mask_f: np.ndarray | None = None,
) -> dict:
    populated_t = pop > 0
    populated_f = pop_f > 0

    def wmean(a, w, m):
        m = m & populated_t  # default mask = populated
        if not m.any():
            return float("nan")
        return float(np.average(a[m], weights=w[m]))

    def share_below(a, m, thr):
        m = m & populated_f
        if not m.any():
            return float("nan")
        return float((a[m] < thr).mean())

    def p10(a, m):
        m = m & populated_f
        if not m.any():
            return float("nan")
        return float(np.quantile(a[m], 0.10))

    all_mask = np.ones_like(pop, dtype=bool)
    out = {
        "pop_w_mean_LT_t_before": wmean(A_before_t, pop, all_mask),
        "pop_w_mean_LT_t_after": wmean(A_after_t, pop, all_mask),
        "pop_w_mean_LT_f_before": wmean(A_before_f, pop_f, all_mask),
        "pop_w_mean_LT_f_after": wmean(A_after_f, pop_f, all_mask),
        "p10_LT_f_before": p10(A_before_f, all_mask),
        "p10_LT_f_after": p10(A_after_f, all_mask),
        "share_below_sphere_f_before": share_below(A_before_f, all_mask, sphere_threshold),
        "share_below_sphere_f_after": share_below(A_after_f, all_mask, sphere_threshold),
    }
    if bottom_decile_mask_f is not None:
        out["bottom_decile_mean_LT_f_before"] = wmean(A_before_f, pop_f, bottom_decile_mask_f)
        out["bottom_decile_mean_LT_f_after"] = wmean(A_after_f, pop_f, bottom_decile_mask_f)
    return out
