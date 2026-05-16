# Accessibility

![](./data/result/Output_road.png)


E2SFCA-based WASH accessibility for the Rohingya camps. R + `RcppParallel`.

## Structure

```         
Accessibility/
├─ main.R            # "for reproducing the results"
├─ utils/
│  ├─ ACC.R          # RcppParallel E2SFCA, distance matrix
│  ├─ spatial.R      # grid, projection, areal interpolation
│  └─ utils.R        # runtime helpers
├─ data/
│  ├─ camp_outline/
│  │  ├─ 20230412_a1_camp_outlines.kml
│  │  └─ Population.csv
│  ├─ facility/
│  │  └─ Rohingya_refugee_response.zip  # read via GDAL /vsizip/
│  ├─ road/
│  │  └─ 20250910_Access_Road_Footpath_all_camps.shp
│  ├─ img_sample/
│  │  └─ 115258_198184.png* # sample segmented image - for CRS information
│  └─ result/
│     ├─ Camp_100m_buffer.* # Boundary buffer layer (100 m) around each camp polygon
│     │ 
│     │  # Building footprint layer derived from our segmentation model.
│     ├─ Rohingya_z18_45441_year2022_v7.gpkg 
│     └─ Rohingya_z18_00000_year2025_v7.gpkg
└─ out/              # generated ACC outputs
```

## Dependencies

-   Tested with R 4.5

-   R packages used in code:

    -   The packages below are required to run main.R
        -   `tidyverse`, `data.table`, `tibble`, `magrittr`
        -   `sf`, `terra`, `tidyterra`,
        -   `sfnetworks`, `tidygraph`, `cppRouting`
        -   `Rcpp`, `RcppParallel`, `Rfast`
    -   optional:
        -   `SpatialAcc` - for the comparison with our 2SFCA implementation
        -   `tmap`, `classInt`, `viridis`, `ggplot2`, `ggspatial`, `ggnewscale`, `cowplot`, `basemaps`, `leafem`, `ggmap`
        -   `microbenchmark`, `tictoc`

-   System libs (Ubuntu/Debian) - need for geospatial libraries. (not required for Windows machines, but you'll need [Rtools](https://cran.r-project.org/bin/windows/Rtools/rtools45/rtools.html).)

    ``` bash         
    sudo apt-get update
    sudo apt-get install -y gdal-bin libgdal-dev libgeos-dev libproj-dev \
        libudunits2-dev libsqlite3-dev libcurl4-openssl-dev libxml2-dev \
        libglpk-dev make g++ cmake
    ```

## Reproducible environment with `renv`

### Restoring the environment on a new machine

``` bash         
R -q -e 'install.packages("renv"); renv::restore(lockfile="renv.lock", prompt=FALSE)'
```

## Data

-   Facility layers are read directly from `./data/facility/Rohingya_refugee_response.zip` via GDAL `/vsizip/`. No manual unzip needed.

-   Population columns used:

    -   2022: `Total22Feb`, `Male22Feb`, `Female22Feb`
    -   2025: `Total25Jan`, `Male25Jan`, `Female25Jan`

-   Road network: `data/road/20250910_Access_Road_Footpath_all_camps.*` ([download](https://rohingyaresponse.org/wp-content/uploads/2025/09/20250910_Access_Road_Footpath_all_camps.zip), [source](https://rohingyaresponse.org/resources-data/))
-   Building footprint layers (`./data/result/Rohingya_z18_*.gpkg`):

    -    These datasets contain **building footprints detected by our segmentation model** from high-resolution imagery.  
    -    Each polygon represents an individual building with its estimated **floor area**, used to assess built-up density and potential service demand within each camp. 
    
## Run

From the R project root:

``` bash
Rscript ./main.R
```

### Python port (igraph + NumPy)

**Conda** (from repo root):

``` bash
conda env create -f Accessibility/python/environment.yml
conda activate wash-access
PYTHONPATH=Accessibility/python python -m wash_access
```

**venv + pip**:

``` bash
python -m venv .venv && .venv/bin/pip install -r Accessibility/python/requirements.txt
PYTHONPATH=Accessibility/python .venv/bin/python -m wash_access
```

- Full run (road network OD matrices) is memory- and CPU-intensive; use `--no-road-network` for a faster pass that only writes the Euclidean GPKGs.
- Outputs go to `./out/` by default (same as the R workflow).

## Outputs

Written to `./out/`:

- Road distance
  -   `ACC22.gpkg`, `ACC25.gpkg`, `ACC_D.gpkg` (scenario 1)
  -   `ACC22_S2.gpkg` (scenario 2)
- Euclidean distance
  -   `ACC22_euclidean.gpkg`, `ACC25_euclidean.gpkg`, `ACC_D_euclidean.gpkg`


## Method summary

**Enhanced two-step floating catchment area (E2SFCA) method**

1. **Provider-to-population ratio**

   $$
   R_j =
   \frac{S_j}
   {\displaystyle\sum_{i\in\,d_{ij}\le d_0}
     P_i\,\exp\!\left(-\frac{d_{ij}^2}{\sigma^2}\right)}
   $$

   where  
   - $R_j$: Provider-to-population ratio at location $j$
   - $S_j$: service capacity of facility $j$ (e.g., number of hand pumps)  
   - $P_i$: population at location $i$  
   - $d_{ij}$: distance or travel time between $i$ and $j$  
   - $d_0$: catchment threshold  
   - $\sigma = 396$: Gaussian decay parameter
   
     $R_j$ represents the **service capacity available per person** within the catchment area of facility $j$.  
    It quantifies how much supply (e.g., number of functional WASH units, pumps, or latrines) is available relative to the total nearby population that can reach that facility, weighted by the distance decay function.  
    Higher $R_j$ indicates less competition for that facility’s services, while lower $R_j$ reflects higher demand pressure or limited service capacity.

2. **Accessibility score**

   $$
   A_i =
   \sum_{j\in\,d_{ij}\le d_0}
   R_j\,\exp\!\left(-\frac{d_{ij}^2}{\sigma^2}\right)
   $$
   
   where
   - $A_i$: Overall accessibility for population location $i$ within the distance threshold.
   - $R_j$: Provider-to-population ratio at location $j$
   - $d_{ij}$: distance or travel time between $i$ and $j$  

 $A_i$ represents the **overall accessibility experienced by the population at location $i$**.  
    It aggregates the contributions of all nearby facilities, each weighted by both the facility’s service-to-population ratio ($R_j$) and the distance decay between $i$ and $j$.  
    In other words, $A_i$ quantifies how much effective service capacity is reachable from a given population point.  
    Higher $A_i$ values indicate better service availability and proximity, while lower $A_i$ values reflect limited access or longer travel distances to essential facilities.

---

Distances: Euclidean via `Rfast` or network shortest path via `sfnetworks`+`cppRouting`.

## Performance

`utils/ACC.R` implements `RcppParallel`. Large ($i$ $\times$ $j$) matrices can be memory-intensive (32GB+ RAM recommended). Use tiling or sampling if needed.
 
 
### Test environment

* 12 CPU cores
* 64GB RAM
* Ubuntu 24.04.3 LTS


## Issues

Open issues at the repo root and tag “Accessibility”.
