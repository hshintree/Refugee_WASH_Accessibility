# Python accessibility pipeline

This is a port of `Accessibility/main.R` and `Accessibility/utils/*.R` using GeoPandas, NumPy, python-igraph, and SciPy.

## Setup

### Conda (recommended for geospatial stacks)

From the repository root:

```bash
conda env create -f Accessibility/python/environment.yml
conda activate wash-access
```

To refresh the environment after dependency changes:

```bash
conda env update -f Accessibility/python/environment.yml --prune
```

### venv + pip

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r Accessibility/python/requirements.txt
```

## Run

Default data root is the `Accessibility/` directory (the parent of this folder). Override with `WASH_ACC_DATA_ROOT` if needed.

```bash
# With conda activated (or substitute .venv/bin/python for venv)
PYTHONPATH=Accessibility/python python -m wash_access
```

Useful flags:

| Flag | Effect |
|------|--------|
| `--no-road-network` | Skip network-based E2SFCA (saves time and RAM); still writes `ACC*_euclidean.gpkg` and `ACC_D_euclidean.gpkg`. |
| `--no-validation` | Skip Figure S5 correlation outputs. |
| `--densify-m 10` | Spacing when discretizing roads for the graph (meters). |

Road distances are approximated by routing on a densified line graph with the same hybrid shortcut as the R script (Euclidean vs connector sum). Results will not match R or cppRouting exactly but follow the same E2SFCA algebra (`sigma = 396`, default `d0 = 1609` m).
