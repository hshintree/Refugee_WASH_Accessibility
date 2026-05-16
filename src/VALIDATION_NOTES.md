# E2SFCA validation notes

## What was validated

`src/e2sfca.py` reproduces Ahn et al.'s Euclidean E2SFCA. Comparing the
Python output to `ACC22_euclidean.gpkg`:

| Subset             | corr  | rel_err | my_mean | pub_mean | bias    |
|--------------------|-------|---------|---------|----------|---------|
| Global (9 959 cells), LT_t | 0.867 | 0.175 | 0.0601  | 0.0585   | +0.0017 |
| Global, LT_f       | 0.845 | 0.197   | 0.1053  | 0.1021   | +0.0032 |
| Camp 22, LT_t      | 0.088 | 0.795   | 0.0936  | 0.0724   | +0.0212 |
| Camp 22, LT_t (pop > 50) | 0.36 | — | 0.0914 | 0.0780 | +0.0134 |

Global numbers match the paper's reported `LT_t = 0.040` (2022) when
restricted to populated cells.

## Known gap on per-camp subsets

The published values filter the latrine supply by
`Camp_100m_buffer.shp` (in UTM 46N). Our Python pipeline keeps the entire
latrine shapefile (within the demand-bbox + 100 m), so some latrines that
the R pipeline excludes leak into the supply set near camp edges. This
inflates per-camp means by ~25 % and washes out cell-level correlation in
small subsets like Camp 22 (226 cells), while having little effect on the
global mean.

## Why this is fine for the optimization

The optimizer evaluates `A_i(x) = A_baseline + Σ_j Δa_ij x_j` and compares
before vs. after with the **same** Python evaluator. The buffer-filter gap
is identical before and after placement, so it cancels in the Δ that
drives candidate selection. Cell-level baselines for plotting use the
published `LT_*` values from the GPKG, not our reproduction.

If we ever need exact reproduction (e.g. for a paper claim), the fix is to
reproject `Camp_100m_buffer.shp` from UTM 46N to Web Mercator and filter
latrines to it.

## Network distance

`ACC22.gpkg` / `ACC22_S2.gpkg` use network distance over the camp footpath
graph (`20250910_Access_Road_Footpath_all_camps.shp`). Our network module
(future) will mirror that with scipy.sparse.csgraph Dijkstra. The
optimizer's linearized `Δa_ij` can be computed with either distance metric
without changing the surrounding code.
