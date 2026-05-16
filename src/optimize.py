"""Discrete optimization for latrine siting.

Implements three solvers over the same input:

- `greedy(...)`: pick the K candidates with the largest population-weighted
  marginal contribution. O(N log N), no solver dependency.
- `solve_ip(...)`: PuLP/CBC branch-and-bound on the linearized objective.
  Returns the exact optimum of the linear-approx problem.
- `pareto_lambda_sweep(...)`: solve for a grid of `lambda ∈ [0,1]` to trace
  the efficiency-equity Pareto frontier the proposal asks for.

Objective. The proposal's first term `Σ_i p_i * A_i(x)` is unfortunately
degenerate under E2SFCA: adding capacity `n` to any facility raises the
population-weighted total accessibility by exactly `n`, independent of the
facility's location (a known conservation property of two-step floating
catchments — the supply ratio `R_j = n/Σ K p` normalises out). To get
spatial discrimination we restrict the efficiency sum to **under-served**
cells (those with baseline `A_i^0` below the Sphere service target,
`1/20 ≈ 0.05`). The proposal's intent — improve access for those who lack
it — is preserved; degeneracy is broken.

    f(x) = λ * Σ_{i ∈ U} p_i * δ_t_ij
           + (1 - λ) * Σ_{i ∈ L} p_i^f * δ_f_ij

`U` = cells with `A_i^{(0)} < SPHERE_TARGET`.
`L` = cells in the bottom `q` quantile of baseline female accessibility
      (restricted to populated cells).

For the linear approximation each candidate contributes:

    coef_j  =  λ * Σ_{i ∈ U} p_i * δ_t_ij  +  (1 - λ) * Σ_{i ∈ L} p_i^f * δ_f_ij
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marginal import MarginalGains

SPHERE_TARGET = 1.0 / 20.0  # one stance per 20 people (Sphere standard)


@dataclass
class SelectionResult:
    chosen: np.ndarray  # (K,) indices into candidates
    coef: np.ndarray  # per-candidate objective coefficient used
    obj_linear: float  # linearized objective at the solution
    method: str
    lam: float
    status: str = "ok"
    solver_seconds: float | None = None


def _objective_coefficients(
    delta_t: np.ndarray,
    delta_f: np.ndarray,
    pop: np.ndarray,
    pop_f: np.ndarray,
    underserved_mask_t: np.ndarray,
    female_priority_mask: np.ndarray,
    lam: float,
) -> np.ndarray:
    # Efficiency term: weight only under-served populated cells. The naive
    # `pop * delta` form is constant per candidate under E2SFCA (see module
    # docstring); restricting to under-served cells fixes that.
    w_eff = pop * underserved_mask_t
    eff = (delta_t * w_eff[:, None]).sum(axis=0)
    w_eq = pop_f * female_priority_mask
    eq = (delta_f * w_eq[:, None]).sum(axis=0)
    return lam * eff + (1.0 - lam) * eq


def select_female_priority_mask(
    A_baseline_f: np.ndarray, pop_f: np.ndarray, q: float = 0.10
) -> np.ndarray:
    """Bottom-`q` quantile of populated cells by baseline female accessibility."""
    populated = pop_f > 0
    if populated.sum() == 0:
        return np.zeros_like(pop_f, dtype=bool)
    thr = np.quantile(A_baseline_f[populated], q)
    return (A_baseline_f <= thr) & populated


def select_underserved_mask(
    A_baseline_t: np.ndarray, pop: np.ndarray, target: float = SPHERE_TARGET
) -> np.ndarray:
    """Populated cells with baseline total accessibility below the Sphere
    service target — the efficiency term focuses on these.
    """
    return (A_baseline_t < target) & (pop > 0)


def greedy(
    mg_t: MarginalGains,
    mg_f: MarginalGains,
    pop: np.ndarray,
    pop_f: np.ndarray,
    K: int,
    lam: float,
    underserved_mask_t: np.ndarray,
    female_priority_mask: np.ndarray,
) -> SelectionResult:
    coef = _objective_coefficients(mg_t.delta, mg_f.delta, pop, pop_f,
                                   underserved_mask_t, female_priority_mask, lam)
    order = np.argsort(-coef)
    chosen = order[:K]
    return SelectionResult(
        chosen=chosen,
        coef=coef,
        obj_linear=float(coef[chosen].sum()),
        method="greedy",
        lam=lam,
    )


def solve_ip(
    mg_t: MarginalGains,
    mg_f: MarginalGains,
    pop: np.ndarray,
    pop_f: np.ndarray,
    K: int,
    lam: float,
    underserved_mask_t: np.ndarray,
    female_priority_mask: np.ndarray,
    *,
    time_limit_s: float = 60.0,
    candidate_xy: np.ndarray | None = None,
    min_pairwise_spacing: float | None = None,
) -> SelectionResult:
    """Branch-and-bound on the linearized objective using PuLP/CBC.

    Optional `min_pairwise_spacing` (in meters) adds pairwise exclusion
    constraints so the chosen K sites don't bunch up — this is what
    prevents the greedy "all clustered in the worst-served pocket"
    failure mode and is the main reason the IP outperforms greedy.
    """
    import pulp

    coef = _objective_coefficients(mg_t.delta, mg_f.delta, pop, pop_f,
                                   underserved_mask_t, female_priority_mask, lam)
    N = len(coef)
    prob = pulp.LpProblem("latrine_siting", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{j}", cat="Binary") for j in range(N)]
    prob += pulp.lpSum(coef[j] * x[j] for j in range(N))
    prob += pulp.lpSum(x) <= K

    if min_pairwise_spacing is not None and candidate_xy is not None:
        # add pairwise: x_a + x_b <= 1 for any (a, b) closer than spacing.
        dx = candidate_xy[:, 0:1] - candidate_xy[None, :, 0]
        dy = candidate_xy[:, 1:2] - candidate_xy[None, :, 1]
        D = np.sqrt(dx * dx + dy * dy)
        bad = np.where(np.triu(D < min_pairwise_spacing, k=1))
        for a, b in zip(*bad):
            prob += x[int(a)] + x[int(b)] <= 1

    import time
    t0 = time.time()
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_s)
    prob.solve(solver)
    elapsed = time.time() - t0

    status = pulp.LpStatus[prob.status]
    chosen = np.array(
        [j for j in range(N) if x[j].value() is not None and x[j].value() > 0.5],
        dtype=int,
    )
    obj = float(coef[chosen].sum()) if len(chosen) else 0.0
    return SelectionResult(
        chosen=chosen,
        coef=coef,
        obj_linear=obj,
        method="ip_branchbound",
        lam=lam,
        status=status,
        solver_seconds=elapsed,
    )


def pareto_lambda_sweep(
    mg_t: MarginalGains,
    mg_f: MarginalGains,
    pop: np.ndarray,
    pop_f: np.ndarray,
    K: int,
    underserved_mask_t: np.ndarray,
    female_priority_mask: np.ndarray,
    *,
    lambdas: np.ndarray | None = None,
    solver: str = "greedy",
    **solver_kwargs,
) -> list[SelectionResult]:
    if lambdas is None:
        lambdas = np.linspace(0.0, 1.0, 11)
    runs = []
    for lam in lambdas:
        if solver == "greedy":
            r = greedy(mg_t, mg_f, pop, pop_f, K, float(lam),
                       underserved_mask_t, female_priority_mask)
        elif solver == "ip":
            r = solve_ip(mg_t, mg_f, pop, pop_f, K, float(lam),
                         underserved_mask_t, female_priority_mask, **solver_kwargs)
        else:
            raise ValueError(solver)
        runs.append(r)
    return runs
