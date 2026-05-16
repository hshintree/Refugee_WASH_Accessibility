# WASH siting optimization pipeline

Layers the discrete-optimization decision step on top of Ahn et al.'s
diagnostic accessibility maps for the Rohingya camps in Cox's Bazar. See
`proposal/main.tex` for the AA222 proposal this implements.

## How to run

```bash
# 1. End-to-end Camp 22 pilot (K = 10 new latrines, λ ∈ {0,.25,.5,.75,1}).
python3 src/run_camp22.py --K 10
# Writes results/camp22/{summary.csv, selected_sites.csv, manifest.json}

# 2. Render proposal-style figures (uses outputs from step 1).
python3 src/make_figures.py --camp "Camp 22" --lambda-pick 0.5 \
    --results results/camp22
# Writes results/camp22/figures/{baseline_LT_*, chosen_sites, post_LT_t,
# delta_LT_t, pareto}.png
```

To run on a different camp, change `--camp`. Camp names come from the
`SMSDCamp` attribute in `20230412_a1_camp_outlines.kml`.

## Module map

| File | Role |
|------|------|
| `geo.py` | Local meter conversion, ray-cast point-in-polygon |
| `projection.py` | Web Mercator (EPSG:3857) forward/inverse |
| `io_kml.py` | Camp boundary polygons |
| `io_shp.py` | Point and polyline shapefile parsing |
| `io_dbf.py` | DBF reader for per-latrine capacity columns |
| `io_gpkg.py` | GeoPackage (SQLite + WKB) reader |
| `loaders.py` | High-level loaders for every project layer |
| `load_latrines.py` | Latrine point+attribute join with Scenario 1/2 capacities |
| `e2sfca.py` | Pure-Python E2SFCA matching Ahn et al. (σ=396 m, d₀=1609 m) |
| `candidates.py` | Feasible 50 m candidate sites with exclusion buffers |
| `marginal.py` | Linearized per-(cell,candidate) accessibility gain |
| `optimize.py` | Greedy and PuLP/CBC branch-and-bound IP |
| `recompute.py` | Full E2SFCA re-evaluation after placement |
| `plots.py`, `make_figures.py` | Matplotlib figures |
| `run_camp22.py` | End-to-end orchestrator |
| `validate_e2sfca.py` | E2SFCA reproduction vs published Euclidean GPKG |

## Two non-obvious issues we addressed

### 1. E2SFCA conserves total accessibility per supply unit

Adding one stance of capacity at any candidate site raises the
**population-weighted total** accessibility by exactly 1 — the supply
ratio `R_j = n_j / Σ K p_i` cancels the cell-side aggregation:
`Σ_i p_i K[i,j] R_j = n_j`. This makes the proposal's first-term
objective `Σ_i p_i A_i(x)` constant across candidate choice at fixed K.

**Fix.** The efficiency term restricts the sum to cells with baseline
total accessibility below the Sphere service target (1/20 = 0.05).
Candidates that serve under-served populations now get a strictly higher
coefficient. The proposal's intent — "improve overall access" — is
preserved; the degeneracy is broken. See `optimize.py` docstring for the
math.

### 2. Validation gap on per-camp subsets

Our pure-Python E2SFCA reproduces the published Euclidean accessibility
**globally** (corr 0.87, mean within 3%), but per-camp means drift by
~25%. The cause is the latrine-supply filter: Ahn et al. clip to
`Camp_100m_buffer.shp` (a multipolygon buffer in UTM 46N), while we keep
all latrines within the global demand bbox + 100 m. The optimization is
unaffected because the same Python E2SFCA evaluates both
baseline-without-new and baseline-with-new, so the buffer-filter error
cancels in the before/after delta. Details in `VALIDATION_NOTES.md`.

## Camp 22 pilot (saturated baseline)

| Metric | Baseline (2022) | After K=10 IP (λ=0.5) | Δ |
|---|---|---|---|
| Pop-weighted mean LT_t | 0.0787 | 0.0791 | +0.6 % |
| Pop-weighted mean LT_f | 0.1159 | 0.1164 | +0.4 % |
| P10 female LT_f | 0.0727 | 0.0728 | +0.1 % |
| Female bottom-decile mean LT_f | 0.1251 | 0.1256 | +0.4 % |
| Share female below Sphere | 0 | 0 | 0 |

Camp 22 already has 1656 latrines for 22 390 people (~13.5 people / stance
vs Sphere's 20). Adding 10 latrines barely moves metrics and the Pareto
frontier collapses to a single point. The pipeline is correct; the camp is
the wrong demonstrator.

## Camp 09 pilot (severely under-served baseline)

Camp 09 has the worst baseline access of the 33 camps. With Euclidean
distance ~96 % of women sit below the Sphere service target; with proper
network distance the baseline is rosier (~83 %) because the footpath
graph models reachable cells more realistically.

**Network distance, K=20, full λ sweep:**

| λ | After share below Sphere | After P10 female | After mean LT_f |
|---|---|---|---|
| baseline | 82.6 % | 0.00476 | 0.03932 |
| 0.00 (pure equity) | 81.2 % | 0.01873 (4×) | 0.03989 |
| 0.50 | 81.6 % | 0.01649 | 0.03989 |
| 1.00 (pure efficiency) | 82.3 % | 0.01242 (2.6×) | 0.03989 |

This is the real Pareto trade-off: at λ=0 the optimizer lifts the
worst-off (P10 female jumps 4×), at λ=1 it spreads gains across more
under-served cells but doesn't lift the bottom as hard. Pop-weighted
mean accessibility is identical across λ — that's the E2SFCA
conservation property (see below).

**Euclidean distance, same K and λ sweep** (for comparison):

| Metric | Baseline | After K=20 IP (λ=0.5) |
|---|---|---|
| Share female below Sphere | 95.8 % | 93.4 % |
| P10 female LT_f | 0.0196 | 0.0216 |

Chosen sites cluster in the under-served eastern half of the camp (see
`results/camp09/figures/chosen_sites.png` and `delta_LT_t.png`). The
greedy and IP selections still overlap because the linear objective is
separable across candidates; the IP starts to differ from greedy only
when pairwise-spacing or per-region budget constraints bind.

## Why greedy and IP coincide for these runs

Under the linearized objective each candidate contributes an independent
coefficient `c_j`; the IP becomes "pick the K largest `c_j`", which is
exactly what greedy does. The IP only helps when extra constraints kick
in (`min_pairwise_spacing`, per-region caps, capacity vs count budgets).
For the proposal we should either (a) tighten `min_pairwise_spacing` to a
value that bites, (b) introduce per-zone caps so the IP must trade off
across zones, or (c) add a **submodular saturation** term to the objective
(diminishing returns when many new sites stack on the same cell). That
last option is what justifies the branch-and-bound machinery in the
proposal.

## Recommended next steps

1. **Pick a worse-served camp.** Sort camps by 2022 baseline `LT_t` or by
   the 2022→2025 decline in `ACC_D.gpkg` and rerun on the bottom 5. The
   Pareto frontier should open up when the camp has under-served pockets
   the optimizer can target.
2. **Add network distance.** Stub is in `validate_e2sfca.py`; the
   footpath polylines are already parsed by `loaders.load_footpath_polylines`.
3. **Add shelter overlap exclusion.** Currently the sensitive-facility
   buffer is the only built-up-area filter. Unzipping
   `Rohingya_z18_45441_year2022_2025v7.zip` gives shelter polygons; a
   ray-cast filter against those would tighten candidate placement.
4. **Capacity-aware candidates.** Right now every new site is a 1-stance
   block. Allowing capacity ∈ {2, 4, 8} (with a budget on total stances
   rather than count) gives the IP more interesting trade-offs.
