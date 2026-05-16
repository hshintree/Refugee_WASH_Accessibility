"""Validate the Python E2SFCA against Ahn et al.'s published GPKG values.

Strategy: reproduce `ACC22_S2.gpkg` (Scenario 2, Euclidean OR network) for
Camp 22 cells using our pure-Python E2SFCA. The published Scenario 1 vs
Scenario 2 differ only in how female stances are aggregated; the LT_t
(total) values are identical to Scenario 1. We focus on the Euclidean
variant first (`ACC22_euclidean.gpkg`) since matching network distance
requires the full road graph; once Euclidean matches, network is just a
substitution of D.

Note: Ahn et al.'s E2SFCA is computed over ALL global demand cells with ALL
latrines inside the 100m camp buffer (`Camp_100m_buffer.shp`). The cell-
level supply ratio `R_j` therefore depends on cells outside Camp 22 that
also draw on the same latrines. For exact reproduction we run the full
~9959 x ~37k computation and then filter to Camp 22 cells for reporting.
With 50m grid and Gaussian decay (sigma=396), the matrix is large but
manageable; we mask distance > d0 first to keep memory bounded.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from e2sfca import D0, SIGMA, e2sfca
from load_latrines import filter_latrines_to_bbox, load_latrines_2022
from loaders import (
    ACC_OUT_DIR,
    filter_baseline_to_camp,
    load_accessibility,
    load_camp_polygon,
)


def chunked_distance_kernel(
    dem_xy: np.ndarray, sup_xy: np.ndarray, d0: float, sigma: float, chunk: int = 1024
) -> np.ndarray:
    """Compute the masked Gaussian kernel in chunks of demand rows to bound
    peak memory. Returns dense (M, K) kernel matrix.
    """
    M = dem_xy.shape[0]
    K = sup_xy.shape[0]
    out = np.zeros((M, K), dtype=np.float64)
    sx = sup_xy[:, 0]
    sy = sup_xy[:, 1]
    for start in range(0, M, chunk):
        end = min(M, start + chunk)
        dx = dem_xy[start:end, 0:1] - sx[None, :]
        dy = dem_xy[start:end, 1:2] - sy[None, :]
        D2 = dx * dx + dy * dy
        d2_max = d0 * d0
        within = D2 <= d2_max
        K_block = np.where(within, np.exp(-D2 / (sigma * sigma)), 0.0)
        out[start:end] = K_block
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camp", default="Camp 22")
    parser.add_argument("--gpkg", default="ACC22_S2.gpkg")
    parser.add_argument("--use-S2", action="store_true",
                        help="Use Scenario 2 female-safety capacity for LT_f")
    args = parser.parse_args()

    t0 = time.time()
    camp = load_camp_polygon(args.camp)
    base_full = load_accessibility(args.gpkg)
    print(f"Loaded {args.gpkg}: {len(base_full.cell_x)} cells in {time.time()-t0:.1f}s")

    latr = load_latrines_2022()
    print(f"Loaded {len(latr.xy_merc)} latrines (global)")

    # Use the global demand grid (all cells from the GPKG), supply set = all
    # latrines within the demand bbox + 100m buffer (a generous proxy for the
    # Camp_100m_buffer.shp filter used in the R code).
    dem_xy = np.column_stack([base_full.cell_x, base_full.cell_y])
    pop = base_full.pop_total
    pop_f = base_full.pop_female
    pop_m = base_full.pop_male

    bx0 = base_full.cell_x.min() - 100
    by0 = base_full.cell_y.min() - 100
    bx1 = base_full.cell_x.max() + 100
    by1 = base_full.cell_y.max() + 100
    latr_b, _ = filter_latrines_to_bbox(latr, bx0, by0, bx1, by1)
    print(f"Latrines within demand-bbox+100m: {len(latr_b.xy_merc)}")

    print("Computing distance kernel (M x K may be large)...")
    t1 = time.time()
    K = chunked_distance_kernel(dem_xy, latr_b.xy_merc, d0=D0, sigma=SIGMA)
    print(f"  kernel shape={K.shape}, nnz≈{(K > 0).sum():,}, in {time.time()-t1:.1f}s")

    # Total latrine accessibility (LT_t) uses `LT` as supply.
    t1 = time.time()
    denom = (K * pop[:, None]).sum(axis=0)
    Rj = np.where(denom > 0, latr_b.LT / denom, 0.0)
    A_t = (K * Rj[None, :]).sum(axis=1)
    print(f"  LT_t computed in {time.time()-t1:.1f}s")

    # Female accessibility
    n_f = latr_b.LT_female_S2 if args.use_S2 else latr_b.LT_female_sum
    denom_f = (K * pop_f[:, None]).sum(axis=0)
    Rj_f = np.where(denom_f > 0, n_f / denom_f, 0.0)
    A_f = (K * Rj_f[None, :]).sum(axis=1)

    # Male accessibility
    denom_m = (K * pop_m[:, None]).sum(axis=0)
    Rj_m = np.where(denom_m > 0, latr_b.LT_male_sum / denom_m, 0.0)
    A_m = (K * Rj_m[None, :]).sum(axis=1)

    pub_t = base_full.lt_total
    pub_f = base_full.lt_female
    pub_m = base_full.lt_male

    # Camp 22 subset for reporting
    base_c22 = filter_baseline_to_camp(base_full, camp)
    from geo import points_in_poly
    mask_c22 = points_in_poly(base_full.cell_x, base_full.cell_y, camp.merc_list())
    print(f"\nCamp 22 cells: {mask_c22.sum()}")

    def summary(name: str, pub: np.ndarray, mine: np.ndarray, mask: np.ndarray) -> None:
        p_sub = pub[mask]
        m_sub = mine[mask]
        denom = np.linalg.norm(p_sub) + 1e-12
        rel = np.linalg.norm(m_sub - p_sub) / denom
        bias = (m_sub - p_sub).mean()
        corr = np.corrcoef(p_sub, m_sub)[0, 1] if p_sub.std() > 0 else float("nan")
        print(
            f"  {name}: corr={corr:.4f} rel_err={rel:.4f} "
            f"bias={bias:+.4f} my_mean={m_sub.mean():.4f} pub_mean={p_sub.mean():.4f}"
        )

    print("\nGlobal:")
    summary("LT_t", pub_t, A_t, np.ones_like(pub_t, dtype=bool))
    summary("LT_f", pub_f, A_f, np.ones_like(pub_f, dtype=bool))
    summary("LT_m", pub_m, A_m, np.ones_like(pub_m, dtype=bool))
    print("\nCamp 22:")
    summary("LT_t", pub_t, A_t, mask_c22)
    summary("LT_f", pub_f, A_f, mask_c22)
    summary("LT_m", pub_m, A_m, mask_c22)

    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
