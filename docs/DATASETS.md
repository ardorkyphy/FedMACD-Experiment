# FedMACD dataset guide

This guide lists the benchmark datasets, their sources and the steps to prepare them. Input files remain at their original sources.

## Availability

Input files are not redistributed here. Prepare them with the commands below and set FEDMACD_DATA_DIR to the output directory parent.

## Dataset table

| Dataset | Type | Shape used in paper | Variables / columns | Source and acquisition | Processing and graph convention |
|---|---|---:|---|---|---|
| Asia | Bayesian network | 10,000 x 8 | BIF-defined names | bnlearn Bayesian-network repository; expected file `asia.bif` (not bundled in this source release) | Forward sample the BIF; graph edges are read from the BIF in row-to-column direction. |
| Child | Bayesian network | 10,000 x 20 | BIF-defined names | bnlearn Bayesian-network repository; expected file `child.bif` (not bundled in this source release) | Forward sample the BIF; graph edges are read from the BIF in row-to-column direction. |
| Alarm | Bayesian network | 10,000 x 37 | BIF-defined names | bnlearn Bayesian-network repository; expected file `alarm.bif` (not bundled in this source release) | Forward sample the BIF; graph edges are read from the BIF in row-to-column direction. |
| Survey | Bayesian network | 10,000 x 6 | BIF-defined names | bnlearn Bayesian-network repository; expected file `survey.bif` (not bundled in this source release) | Forward sample the BIF; graph edges are read from the BIF in row-to-column direction. |
| Earthquake | Bayesian network | 10,000 x 5 | BIF-defined names | bnlearn Bayesian-network repository; expected file `earthquake.bif` (not bundled in this source release) | Forward sample the BIF; graph edges are read from the BIF in row-to-column direction. |
| Auto_MPG | Continuous table | 392 x 5 | `Displacement`, `Mpg`, `Horsepower`, `Weight`, `Acceleration` | [Auto MPG data CSV](https://github.com/mas-takayama/LLM-and-SCD/blob/5aaeaea89e71e2cff443289e473c06d2f79172c5/benchmark%20datasets/Auto%20MPG%20data.csv), linked by the [MATMCD project README](https://github.com/D2I-Group/matmcd/blob/1203da8579e30f598e4767d6e9a358ebdbedcfd0/README.md); original dataset: [UCI Auto MPG](https://archive.ics.uci.edu/dataset/9/auto+mpg) | The corresponding graph is transposed to the row-to-column convention used by this code. |
| DWD_climate | Continuous table | 349 x 6 | `Altitude`, `Temperature`, `Precipitation`, `Longitude`, `Sunshine`, `Latitude` | [DWD climate data CSV](https://github.com/mas-takayama/LLM-and-SCD/blob/5aaeaea89e71e2cff443289e473c06d2f79172c5/benchmark%20datasets/DWD%20climate%20data.csv), linked by the [MATMCD project README](https://github.com/D2I-Group/matmcd/blob/1203da8579e30f598e4767d6e9a358ebdbedcfd0/README.md); original collection: [Tuebingen Cause-Effect Pairs](https://webdav.tuebingen.mpg.de/cause-effect/) | The corresponding graph is transposed to the row-to-column convention used by this code. |
| Sachs | Continuous table | 7466 x 11 | `raf`, `mek`, `plc`, `pip2`, `pip3`, `erk`, `akt`, `pka`, `pkc`, `p38`, `jnk` | [Sachs data CSV](https://github.com/mas-takayama/LLM-and-SCD/blob/5aaeaea89e71e2cff443289e473c06d2f79172c5/benchmark%20datasets/Sachs%20data.csv), linked by the [MATMCD project README](https://github.com/D2I-Group/matmcd/blob/1203da8579e30f598e4767d6e9a358ebdbedcfd0/README.md); source study: [Sachs et al.](https://doi.org/10.1126/science.1105809) | The corresponding graph is transposed to the row-to-column convention used by this code. |
| 10nodes | Synthetic continuous table | 5000 x 10 | `X0` ... `X9` | FedPC benchmark files at pinned source: [10nodes_5000.txt](https://github.com/Xianjie-Guo/FedPC/blob/c6b0bf99c23b1d23b9bbd8ad043ec271d63d9345/FedPC-code%20%28matlab%29/dataset/10nodes_5000.txt) and [10_DAG.txt](https://github.com/Xianjie-Guo/FedPC/blob/c6b0bf99c23b1d23b9bbd8ad043ec271d63d9345/FedPC-code%20%28matlab%29/dataset/10_DAG.txt) | Convert the whitespace-delimited observations to a headerless CSV; use the supplied graph. |
| ER20 | Synthetic continuous table | 256 x 20 | Numeric columns | Generated with [NOTEARS-ADMM](https://github.com/ignavierng/notears-admm/tree/5c9a274d6745d384209997222f4817e43ea031f7). | Generated with seed 1 using `simulate_dag(20, 20, "ER")` and `simulate_linear_sem(graph, 256, "gauss")` from NOTEARS-ADMM. The graph is stored in row-to-column orientation. |

## Input versions and preparation

The five Bayesian networks are downloaded from bnlearn. The preparation script checks the file versions and sets the variable order used by the experiments for Asia.

Use these source versions: LLM-and-SCD `5aaeaea89e71e2cff443289e473c06d2f79172c5` (linked by MATMCD), FedPC `c6b0bf99c23b1d23b9bbd8ad043ec271d63d9345`, and NOTEARS-ADMM `5c9a274d6745d384209997222f4817e43ea031f7`. ER20 generation also requires `igraph==1.0.0`.

Run both commands from the repository root and use an empty output directory. The script checks source files and does not overwrite existing inputs. Detailed checksums and validation results are in [DATA_VALIDATION.md](DATA_VALIDATION.md).

```bash
python scripts/prepare_public_datasets.py --output-root /path/to/local-datasets --download-bnlearn
python scripts/prepare_public_datasets.py --llm-scd-root /path/to/LLM-and-SCD --fedpc-root /path/to/FedPC --notears-root /path/to/notears-admm --output-root /path/to/local-datasets
```

## Reconstructing a fold

The paper uses homogeneous and heterogeneous scenarios, seeds 0/1/2, and four clients. Bayesian-network folds are generated by client-specific soft interventions and independent sampling. Continuous folds use preference-region-weighted sampling without replacement within each client; source records may appear in more than one client. The default heterogeneity strength is 0.7. These settings are implemented in `src_v4/` and the `scripts/run_v436_*.py` drivers.

Main-experiment BIF folds use the fold seed. Homogeneous baseline inputs begin from a seed-0 sample before splitting. These paths generate observations differently, so regenerated stochastic samples need not be byte-identical to every historical fold.

## References

- Dimensions and column orders above follow the paper and v4.36 release configuration.
- Third-party source files are linked at their upstream repositories; check those repositories for their applicable terms.
