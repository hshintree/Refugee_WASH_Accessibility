"""End-to-end pipeline for the Camp 22 pilot.

Stages:
1. Load Camp 22 polygon, demand grid (from ACC22_S2.gpkg), existing
   latrines, common-facility constraint layer.
2. Generate feasible 50 m candidate sites.
3. Compute Δa_ij linearized marginal gains.
4. Solve greedy and IP for several K, λ ∈ {0, 0.25, 0.5, 0.75, 1}.
5. Recompute full E2SFCA on the IP picks and report linearization gap.
6. Persist results to `results/camp22/`.

Run:
    python3 src/run_camp22.py --K 20

Output: `results/camp22/summary.csv`, per-lambda selected-sites CSVs,
and a JSON manifest with timings + linearization gaps.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import marginal
import optimize
import candidates as cmod
from loaders import (
    PROJECT_ROOT,
    filter_baseline_to_camp,
    filter_points_to_camp,
    load_accessibility,
    load_camp_polygon,
    load_common_facilities,
    load_footpath_polylines,
)
from load_latrines import load_latrines_2022, filter_latrines_to_bbox
from network_distance import build_graph, network_distance_matrix
from recompute import metrics_table, recompute_with_new_sites


def _camp_slug(name: str) -> str:
    return name.lower().replace(" ", "").replace("/", "-")


def run(args) -> dict:
    t_total = time.time()
    results_dir = PROJECT_ROOT / "results" / _camp_slug(args.camp)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("== Loading inputs ==")
    camp = load_camp_polygon(args.camp)
    baseline_full = load_accessibility(args.gpkg)
    baseline = filter_baseline_to_camp(baseline_full, camp)
    print(f"  Camp {args.camp}: {len(baseline.cell_x)} demand cells, "
          f"pop_total={baseline.pop_total.sum():.0f}, "
          f"pop_female={baseline.pop_female.sum():.0f}")

    all_latr = load_latrines_2022()
    cf = load_common_facilities(args.camp)
    print(f"  existing latrines: {len(all_latr.xy_merc)} global, "
          f"{len(filter_points_to_camp(all_latr.xy_merc, camp))} inside {args.camp}")
    print(f"  common facilities: {len(cf.df)}, sensitive: {cf.df['is_sensitive'].sum()}")

    print("\n== Generating candidates ==")
    cands = cmod.generate(
        camp=camp,
        baseline=baseline_full,
        existing_latrines_xy=all_latr.xy_merc,
        common=cf,
        use_demand_grid_only=args.demand_grid_only,
        min_latrine_setback=args.min_latrine_setback,
        sensitive_buffer=args.sensitive_buffer,
    )
    print(f"  feasible candidates: {len(cands.xy)}")
    if len(cands.xy) == 0:
        raise SystemExit("No feasible candidates — relax exclusion buffers.")

    dem_xy = np.column_stack([baseline.cell_x, baseline.cell_y])

    D_dem_cand = None
    fg = None
    distance_fn = None
    if args.distance == "network":
        print("\n== Building footpath network graph ==")
        t = time.time()
        bx0 = camp.merc[:, 0].min() - 2000
        by0 = camp.merc[:, 1].min() - 2000
        bx1 = camp.merc[:, 0].max() + 2000
        by1 = camp.merc[:, 1].max() + 2000
        all_paths = load_footpath_polylines()
        paths_local = [
            p for p in all_paths
            if any(bx0 <= x <= bx1 and by0 <= y <= by1 for x, y in p)
        ]
        fg = build_graph(paths_local, snap_tol_m=0.5)
        print(f"  graph: {len(fg.nodes_xy)} nodes, {fg.graph.nnz//2} edges, "
              f"{fg.n_components} components ({time.time()-t:.2f}s)")
        D_dem_cand = network_distance_matrix(dem_xy, cands.xy, fg, d0=1609.0)

        def distance_fn(orig_xy: np.ndarray, dest_xy: np.ndarray) -> np.ndarray:
            return network_distance_matrix(orig_xy, dest_xy, fg, d0=1609.0)

    label = "network" if args.distance == "network" else "Euclidean"
    print(f"\n== Computing marginal gains ({label} E2SFCA) ==")
    t = time.time()
    mg_t = marginal.compute(
        dem_xy=dem_xy,
        pop=baseline.pop_total,
        cand_xy=cands.xy,
        capacity=1.0,
        distance_matrix=D_dem_cand,
    )
    mg_f = marginal.compute(
        dem_xy=dem_xy,
        pop=baseline.pop_female,
        cand_xy=cands.xy,
        capacity=0.5,
        distance_matrix=D_dem_cand,
    )
    print(f"  done in {time.time()-t:.2f}s; "
          f"delta_t shape={mg_t.delta.shape}, mean per-cand gain="
          f"{mg_t.per_candidate_total_gain(baseline.pop_total).mean():.4f}")

    female_priority = optimize.select_female_priority_mask(
        baseline.lt_female, baseline.pop_female, q=args.bottom_quantile
    )
    underserved = optimize.select_underserved_mask(
        baseline.lt_total, baseline.pop_total, target=optimize.SPHERE_TARGET
    )
    print(f"  female bottom-decile cells in Camp: {female_priority.sum()}")
    print(f"  under-served cells (LT_t<{optimize.SPHERE_TARGET:.3f}): {underserved.sum()}"
          f" / populated {(baseline.pop_total > 0).sum()}")

    lambdas = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    print(f"\n== Solving for K={args.K}, lambdas={lambdas.tolist()} ==")

    rows = []
    site_dumps = []

    # Prep recompute supply (Camp-22-local latrines so the metrics speak to
    # the camp scale)
    c22_lat_mask = np.isin(
        np.arange(len(all_latr.xy_merc)),
        np.where(
            filter_points_to_camp(all_latr.xy_merc, camp).shape[0] > 0
            and np.array([True] * len(all_latr.xy_merc))
        )[0] if False else np.array([], dtype=int),
    )
    # Simpler approach: filter by polygon directly
    from geo import points_in_poly
    c22_mask = points_in_poly(
        all_latr.xy_merc[:, 0], all_latr.xy_merc[:, 1], camp.merc_list()
    )
    sup_xy = all_latr.xy_merc[c22_mask]
    sup_t = all_latr.LT[c22_mask]
    sup_f = all_latr.LT_female_S2[c22_mask]  # use Scenario 2 capacities

    for lam in lambdas:
        print(f"\n  lambda={lam:.2f}")
        # Greedy
        gr = optimize.greedy(
            mg_t, mg_f, baseline.pop_total, baseline.pop_female,
            args.K, float(lam), underserved, female_priority,
        )
        # IP
        ip = optimize.solve_ip(
            mg_t, mg_f, baseline.pop_total, baseline.pop_female,
            args.K, float(lam), underserved, female_priority,
            time_limit_s=args.ip_time_limit,
            candidate_xy=cands.xy,
            min_pairwise_spacing=args.min_pairwise_spacing,
        )
        print(f"    greedy obj_linear={gr.obj_linear:.6f}; "
              f"IP obj_linear={ip.obj_linear:.6f} ({ip.status}, "
              f"{ip.solver_seconds:.2f}s)")

        for tag, res in (("greedy", gr), ("ip", ip)):
            new_xy = cands.xy[res.chosen]
            rc = recompute_with_new_sites(
                dem_xy=dem_xy,
                pop=baseline.pop_total,
                pop_f=baseline.pop_female,
                sup_xy=sup_xy,
                sup_capacity_t=sup_t,
                sup_capacity_f=sup_f,
                new_xy=new_xy,
                distance_fn=distance_fn,
            )
            mt = metrics_table(
                rc.A_before_t, rc.A_after_t, rc.A_before_f, rc.A_after_f,
                baseline.pop_total, baseline.pop_female,
                bottom_decile_mask_f=female_priority,
            )
            row = {
                "lambda": float(lam),
                "method": tag,
                "K": args.K,
                "n_candidates": len(cands.xy),
                "n_chosen": len(res.chosen),
                "obj_linear": res.obj_linear,
                "solver_status": res.status,
                "solver_seconds": res.solver_seconds,
            }
            row.update(mt)
            rows.append(row)
            for k, idx in enumerate(res.chosen):
                site_dumps.append({
                    "lambda": float(lam),
                    "method": tag,
                    "rank": k,
                    "candidate_idx": int(idx),
                    "x_merc": float(cands.xy[idx, 0]),
                    "y_merc": float(cands.xy[idx, 1]),
                    "coef": float(res.coef[idx]),
                })

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "summary.csv", index=False)
    pd.DataFrame(site_dumps).to_csv(results_dir / "selected_sites.csv", index=False)

    manifest = {
        "camp": args.camp,
        "K": args.K,
        "lambdas": lambdas.tolist(),
        "n_demand_cells": int(len(baseline.cell_x)),
        "n_existing_latrines_in_camp": int(c22_mask.sum()),
        "n_candidates": int(len(cands.xy)),
        "exclusion_settings": {
            "min_latrine_setback_m": args.min_latrine_setback,
            "sensitive_buffer_m": args.sensitive_buffer,
            "min_pairwise_spacing_m": args.min_pairwise_spacing,
        },
        "total_seconds": time.time() - t_total,
    }
    (results_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {results_dir}/summary.csv ({len(df)} rows), "
          f"selected_sites.csv ({len(site_dumps)} rows), manifest.json")
    print(f"Total time: {manifest['total_seconds']:.1f}s")
    return manifest


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--camp", default="Camp 22")
    p.add_argument("--gpkg", default="ACC22_S2.gpkg")
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--min-latrine-setback", type=float, default=25.0)
    p.add_argument("--sensitive-buffer", type=float, default=50.0)
    p.add_argument("--min-pairwise-spacing", type=float, default=50.0,
                   help="prevent the IP from picking two candidates closer than this")
    p.add_argument("--bottom-quantile", type=float, default=0.10)
    p.add_argument("--demand-grid-only", action="store_true")
    p.add_argument("--ip-time-limit", type=float, default=60.0)
    p.add_argument(
        "--distance",
        choices=("euclid", "network"),
        default="euclid",
        help="distance metric for E2SFCA: Euclidean (fast) or network (matches Ahn et al.)",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
