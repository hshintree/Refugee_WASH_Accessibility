"""Render the proposal figures from a finished run.

Reads `results/<camp>/summary.csv` and `selected_sites.csv` and produces
PNGs in `results/<camp>/figures/`.

Run AFTER `run_camp22.py` has populated results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import plots
from geo import points_in_poly
from load_latrines import load_latrines_2022
from loaders import (
    PROJECT_ROOT,
    filter_baseline_to_camp,
    filter_points_to_camp,
    load_accessibility,
    load_camp_polygon,
    load_common_facilities,
)
from recompute import recompute_with_new_sites


def render(camp_name: str, lam_pick: float = 0.5, results_dir: Path | None = None):
    if results_dir is None:
        results_dir = PROJECT_ROOT / "results" / camp_name.lower().replace(" ", "")
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    camp = load_camp_polygon(camp_name)
    base_full = load_accessibility("ACC22_S2.gpkg")
    base = filter_baseline_to_camp(base_full, camp)

    sel = pd.read_csv(results_dir / "selected_sites.csv")
    summary = pd.read_csv(results_dir / "summary.csv").to_dict("records")

    # 1. Baseline LT_t
    plots.heatmap(
        cell_rings=base.rings,
        values=base.lt_total,
        camp_polygon=camp.merc_list(),
        title=f"{camp_name}: 2022 baseline latrine accessibility (LT_t)",
    )[0].savefig(fig_dir / "baseline_LT_t.png", bbox_inches="tight",
                  facecolor=plots.PALETTE["bg"])
    import matplotlib.pyplot as plt
    plt.close("all")

    # 2. Baseline LT_f
    plots.heatmap(
        cell_rings=base.rings,
        values=base.lt_female,
        camp_polygon=camp.merc_list(),
        title=f"{camp_name}: 2022 baseline female latrine accessibility (LT_f, S2)",
    )[0].savefig(fig_dir / "baseline_LT_f.png", bbox_inches="tight",
                  facecolor=plots.PALETTE["bg"])
    plt.close("all")

    # 3. Chosen sites overlay (greedy + IP at lam_pick)
    cf = load_common_facilities(camp_name)
    latr = load_latrines_2022()
    latr_in = filter_points_to_camp(latr.xy_merc, camp)

    # `lambda` is a Python keyword so pandas.query rejects it — use the
    # bracket form instead.
    sub_greedy = sel[(sel["method"] == "greedy") & np.isclose(sel["lambda"], lam_pick)]
    sub_ip = sel[(sel["method"] == "ip") & np.isclose(sel["lambda"], lam_pick)]
    greedy_xy = sub_greedy[["x_merc", "y_merc"]].to_numpy() if len(sub_greedy) else np.zeros((0, 2))
    ip_xy = sub_ip[["x_merc", "y_merc"]].to_numpy() if len(sub_ip) else np.zeros((0, 2))

    fig, ax = plots.heatmap(
        cell_rings=base.rings,
        values=base.lt_female,
        camp_polygon=camp.merc_list(),
        title=f"{camp_name}: chosen sites overlay (λ={lam_pick})",
        overlays=[
            (latr_in, dict(s=4, c=plots.PALETTE["latrine"], alpha=0.55,
                            label="2022 latrines", zorder=2)),
            (cf.sensitive_xy(), dict(s=12, marker="s",
                                      c=plots.PALETTE["sensitive"],
                                      alpha=0.7,
                                      label="sensitive sites",
                                      zorder=3)),
            (greedy_xy, dict(s=80, marker="X", facecolors="none",
                              edgecolors=plots.PALETTE["chosen_greedy"],
                              linewidths=1.8, label="greedy", zorder=5)),
            (ip_xy, dict(s=110, marker="o", facecolors="none",
                          edgecolors=plots.PALETTE["chosen_ip"],
                          linewidths=1.8, label="IP", zorder=6)),
        ],
    )
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    fig.savefig(fig_dir / "chosen_sites.png", bbox_inches="tight",
                 facecolor=plots.PALETTE["bg"])
    plt.close(fig)

    # 4. Post-placement and delta heatmaps for IP at lam_pick
    dem_xy = np.column_stack([base.cell_x, base.cell_y])
    c22_mask = points_in_poly(latr.xy_merc[:, 0], latr.xy_merc[:, 1],
                               camp.merc_list())
    sup_xy = latr.xy_merc[c22_mask]
    sup_t = latr.LT[c22_mask]
    sup_f = latr.LT_female_S2[c22_mask]

    rc = recompute_with_new_sites(
        dem_xy=dem_xy,
        pop=base.pop_total,
        pop_f=base.pop_female,
        sup_xy=sup_xy,
        sup_capacity_t=sup_t,
        sup_capacity_f=sup_f,
        new_xy=ip_xy,
    )

    plots.heatmap(
        cell_rings=base.rings,
        values=rc.A_after_t,
        camp_polygon=camp.merc_list(),
        title=f"{camp_name}: post-placement total accessibility (IP λ={lam_pick})",
        overlays=[(ip_xy, dict(s=70, marker="o", facecolors="none",
                                edgecolors=plots.PALETTE["chosen_ip"],
                                linewidths=1.6))],
    )[0].savefig(fig_dir / "post_LT_t.png", bbox_inches="tight",
                  facecolor=plots.PALETTE["bg"])
    plt.close("all")

    delta = rc.A_after_t - rc.A_before_t
    # Diverging colormap centered at zero
    import matplotlib.colors as mc
    cmap = mc.LinearSegmentedColormap.from_list(
        "div", [plots.PALETTE["low"], plots.PALETTE["bg"], plots.PALETTE["high"]]
    )
    vlim = max(1e-6, float(np.abs(delta).max()))
    plots.heatmap(
        cell_rings=base.rings,
        values=delta,
        camp_polygon=camp.merc_list(),
        title=f"{camp_name}: Δ accessibility from {len(ip_xy)} new sites",
        cmap=cmap,
        vmin=-vlim,
        vmax=vlim,
        overlays=[(ip_xy, dict(s=60, marker="o", facecolors="none",
                                edgecolors=plots.PALETTE["ink"],
                                linewidths=1.4))],
    )[0].savefig(fig_dir / "delta_LT_t.png", bbox_inches="tight",
                  facecolor=plots.PALETTE["bg"])
    plt.close("all")

    # 5. Pareto
    plots.pareto_plot(summary, save_path=fig_dir / "pareto.png")

    print(f"Wrote figures to {fig_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--camp", default="Camp 22")
    p.add_argument("--lambda-pick", type=float, default=0.5,
                   help="which lambda's selection to visualize")
    p.add_argument("--results", default=None)
    args = p.parse_args()
    render(args.camp,
           lam_pick=args.lambda_pick,
           results_dir=Path(args.results) if args.results else None)


if __name__ == "__main__":
    main()
