# FedMACD dataset guide

This file is the public data dictionary for the ten datasets used by the paper. It records the dataset role, dimensions, variable order, graph convention, and preprocessing. The repository does not silently substitute a different dataset with the same name.

## Availability

This guide describes benchmark inputs used by the experiment and how to obtain or regenerate them. Third-party inputs remain at their original public sources and are not redistributed by this repository. Set `FEDMACD_DATA_DIR` to a local directory with the layout below. Use the pinned source versions and the conversion script in `scripts/prepare_public_datasets.py` for the three MATMCD tables and the 10nodes benchmark. ER20 is regenerated study data; the BIF files are sampled by the experiment code.

## Dataset table

| Dataset | Type | Shape used in paper | Variables / columns | Source and acquisition | Processing and graph convention |
|---|---|---:|---|---|---|
| Asia | Bayesian network | 10,000 x 8 | BIF-defined names | bnlearn Bayesian-network repository; expected file `asia.bif` (not bundled in this source release) | Forward sample the BIF; graph edges are read from the BIF in row-to-column direction. |
| Child | Bayesian network | 10,000 x 20 | BIF-defined names | bnlearn Bayesian-network repository; expected file `child.bif` (not bundled in this source release) | Forward sample the BIF; graph edges are read from the BIF in row-to-column direction. |
| Alarm | Bayesian network | 10,000 x 37 | BIF-defined names | bnlearn Bayesian-network repository; expected file `alarm.bif` (not bundled in this source release) | Forward sample the BIF; graph edges are read from the BIF in row-to-column direction. |
| Survey | Bayesian network | 10,000 x 6 | BIF-defined names | bnlearn Bayesian-network repository; expected file `survey.bif` (not bundled in this source release) | Forward sample the BIF; graph edges are read from the BIF in row-to-column direction. |
| Earthquake | Bayesian network | 10,000 x 5 | BIF-defined names | bnlearn Bayesian-network repository; expected file `earthquake.bif` (not bundled in this source release) | Forward sample the BIF; graph edges are read from the BIF in row-to-column direction. |
| Auto_MPG | Continuous table | 392 x 5 | `Displacement`, `Mpg`, `Horsepower`, `Weight`, `Acceleration` | [Auto MPG data CSV](https://github.com/mas-takayama/LLM-and-SCD/blob/5aaeaea89e71e2cff443289e473c06d2f79172c5/benchmark%20datasets/Auto%20MPG%20data.csv), linked by the [MATMCD project README](https://github.com/D2I-Group/matmcd/blob/1203da8579e30f598e4767d6e9a358ebdbedcfd0/README.md); original dataset: [UCI Auto MPG](https://archive.ics.uci.edu/dataset/9/auto+mpg) | The pinned CSV matches the study table numerically. Its matching ground-truth matrix is transposed to produce the study `graph.txt`; see the preparation script. |
| DWD_climate | Continuous table | 349 x 6 | `Altitude`, `Temperature`, `Precipitation`, `Longitude`, `Sunshine`, `Latitude` | [DWD climate data CSV](https://github.com/mas-takayama/LLM-and-SCD/blob/5aaeaea89e71e2cff443289e473c06d2f79172c5/benchmark%20datasets/DWD%20climate%20data.csv), linked by the [MATMCD project README](https://github.com/D2I-Group/matmcd/blob/1203da8579e30f598e4767d6e9a358ebdbedcfd0/README.md); original collection: [Tuebingen Cause-Effect Pairs](https://webdav.tuebingen.mpg.de/cause-effect/) | The pinned CSV matches the study table numerically. Its matching ground-truth matrix is transposed to produce the study `graph.txt`; see the preparation script. |
| Sachs | Continuous table | 7466 x 11 | `raf`, `mek`, `plc`, `pip2`, `pip3`, `erk`, `akt`, `pka`, `pkc`, `p38`, `jnk` | [Sachs data CSV](https://github.com/mas-takayama/LLM-and-SCD/blob/5aaeaea89e71e2cff443289e473c06d2f79172c5/benchmark%20datasets/Sachs%20data.csv), linked by the [MATMCD project README](https://github.com/D2I-Group/matmcd/blob/1203da8579e30f598e4767d6e9a358ebdbedcfd0/README.md); source study: [Sachs et al.](https://doi.org/10.1126/science.1105809) | The pinned CSV matches the study table numerically. Its matching ground-truth matrix is transposed to produce the study `graph.txt`; see the preparation script. |
| 10nodes | Synthetic continuous table | 5000 x 10 | `X0` ... `X9` | FedPC benchmark files at pinned source: [10nodes_5000.txt](https://github.com/Xianjie-Guo/FedPC/blob/c6b0bf99c23b1d23b9bbd8ad043ec271d63d9345/FedPC-code%20%28matlab%29/dataset/10nodes_5000.txt) and [10_DAG.txt](https://github.com/Xianjie-Guo/FedPC/blob/c6b0bf99c23b1d23b9bbd8ad043ec271d63d9345/FedPC-code%20%28matlab%29/dataset/10_DAG.txt) | The public source files match the retained originals byte-for-byte. Convert whitespace-delimited rows to headerless CSV; retain the supplied graph. |
| ER20 | Synthetic continuous table | 256 x 20 | Numeric columns | Generated with [NOTEARS-ADMM](https://github.com/ignavierng/notears-admm/tree/5c9a274d6745d384209997222f4817e43ea031f7). | The preparation script uses `set_random_seed(1)`, `simulate_dag(d=20, s0=20, graph_type="ER")`, `simulate_parameter(B_true)`, then `simulate_linear_sem(B_true, n=256, sem_type="gauss")`; it saves data without a header and the graph in row-to-column form. The retained input matrix was numerically recovered with the pinned utility and the environment described below; fold splits are applied later by the experiment code. |

The five BIF network files are obtained from bnlearn: [Asia](https://www.bnlearn.com/bnrepository/asia/asia.bif.gz), [Child](https://www.bnlearn.com/bnrepository/child/child.bif.gz), [Alarm](https://www.bnlearn.com/bnrepository/alarm/alarm.bif.gz), [Survey](https://www.bnlearn.com/bnrepository/survey/survey.bif.gz), and [Earthquake](https://www.bnlearn.com/bnrepository/earthquake/earthquake.bif.gz). The preparation script checks the downloaded SHA-256 hashes and refuses changed upstream files. Child, Alarm, Survey and Earthquake match the retained study BIFs byte-for-byte. The current Asia download has identical edges, states and conditional probabilities but different block order. The script restores the historical alphabetical variable/probability block order. This matters because node order affects seeded sampling and column order. The restored Asia model matches the historical model's node, edge and CPD order; forward samples of 10,000 rows were checked for seeds 0, 1 and 2.

Expected local layout: `bayesian_network_structures/{asia,child,alarm,survey,earthquake}.bif`; `{Auto_MPG,DWD_climate,Sachs,10nodes,ER20}/data.csv` and `graph.txt`. `FEDMACD_DATA_DIR` points to their parent directory.

Obtain local checkouts of these exact upstream commits (the first checkout is **LLM-and-SCD**, linked by MATMCD, rather than the MATMCD code repository):

- `https://github.com/mas-takayama/LLM-and-SCD`, commit `5aaeaea89e71e2cff443289e473c06d2f79172c5`.
- `https://github.com/Xianjie-Guo/FedPC`, commit `c6b0bf99c23b1d23b9bbd8ad043ec271d63d9345`.
- `https://github.com/ignavierng/notears-admm`, commit `5c9a274d6745d384209997222f4817e43ea031f7`.

ER20 preparation additionally requires `igraph==1.0.0` (`python -m pip install igraph==1.0.0`), alongside the repository requirements. Run each command once, using an empty output directory initially:

```bash
python scripts/prepare_public_datasets.py --output-root /path/to/local-datasets --download-bnlearn
python scripts/prepare_public_datasets.py --llm-scd-root /path/to/LLM-and-SCD --fedpc-root /path/to/FedPC --notears-root /path/to/notears-admm --output-root /path/to/local-datasets
```

The script refuses to overwrite existing inputs. It checks the four externally sourced continuous inputs against the study checksums below. ER20 uses seed 1, `simulate_dag(20, 20, "ER")`, `simulate_parameter(graph)` (whose return value is not used), then `simulate_linear_sem(graph, 256, "gauss")`. Retaining the parameter call is necessary because it advances the random-number state. In the checked environment the regenerated graph matches the historical graph byte-for-byte and the 256 x 20 data matrix matches numerically (maximum absolute difference approximately 1.3e-15). CSV byte identity is not asserted across numerical/serialization environments.

## Historical local-file checksums

These are SHA-256 hashes of the study's local `data.csv` and `graph.txt` files. The upstream public files and conversion steps above identify how the corresponding input files were obtained. The checksums are for the converted study inputs, not a claim that the third-party files may be redistributed.

The pinned upstream repository object IDs for the third-party files are:

| Source file | Git blob ID |
|---|---|
| MATMCD Auto MPG data | `1114e0af8a71a839b26b55ebb2d77047c58b0cee` |
| MATMCD Auto MPG graph | `e07b2315089c7937a46089b00483ad5c382b4f37` |
| MATMCD DWD climate data | `6870e05a7a874374794fb39d64494eab81f322e0` |
| MATMCD DWD climate graph | `cad65ed03224912fc899f195ca9d81663b9f3123` |
| MATMCD Sachs data | `217e096444280fc24548fd37643129421acc6398` |
| MATMCD Sachs graph | `3fd032304bac2207de85756beafa4a686a20c54d` |
| FedPC 10nodes observations | `3fb2af4886a043ab178bafd754186946097a12db` |
| FedPC 10nodes graph | `e444e41f534958bc69b76441f44c1a5e08f82d5e` |

| Dataset | File | SHA-256 |
|---|---|---|
| Auto_MPG | `data.csv` | `32dddbccc91b57c4c03505b9bd7b4ab221709d0339886ce18f573b6aba2a7fad` |
| Auto_MPG | `graph.txt` | `1a3fed17979b125c4943525fb34f64c4d20498015dbda587209797ab5ec07051` |
| DWD_climate | `data.csv` | `c3426411aaa857b9c60e6a99b859336dc0edc2fdbafe29e8594914f3ab7e6041` |
| DWD_climate | `graph.txt` | `dba3a21b858140b09551477751d47c54099c310d3d5c1255f89b2e73d860540b` |
| Sachs | `data.csv` | `70cedff4412c53e17610d662ed1d7e7dd8bc8ec103ff0c741a70d339f8f34606` |
| Sachs | `graph.txt` | `7be7ddcf108b32ad612c7e48f42ad386565a94afedcdb3901ee052a03d5a0f39` |
| 10nodes | `data.csv` | `5913e900d532738704839e0371d45c18a43b65fff888dea2600fe1b938fb6b10` |
| 10nodes | `graph.txt` | `1d981c99be2cf83a64f3ada23c2032ab289d70f1c7bb429392d8ac970fd47a4a` |
| ER20 | `graph.txt` | `8e981bda526c3b498fcd2fd694aa41c476fd894b182b16ef44d11bca2cf2d8d6` |

## Reconstructing a fold

The paper uses homogeneous and heterogeneous scenarios, seeds 0/1/2, and four clients. Bayesian-network folds are generated by client-specific soft interventions and independent sampling. Continuous folds use preference-region-weighted sampling without replacement within each client; source records may appear in more than one client. The default heterogeneity strength is 0.7. These settings are implemented in `src_v4/` and the `scripts/run_v436_*.py` drivers.

The current continuous-data loader reads these fixed tables and then applies the fold-specific split. Main-experiment BIF folds are sampled from their BIF structures with the fold seed (`sampling_seed=-1`); homogeneous baseline preparation uses a seed-0 sample before creating the seeded splits. These paths are intentionally documented separately because they produce different generated observations. The checksums above identify the retained source tables and graph files in the project tree; regenerated stochastic samples should not be described as byte-identical historical fold inputs.

## References

- Dimensions and column orders above follow the paper and v4.36 release configuration.
- Third-party source files are linked at their upstream repositories; check those repositories for their applicable terms.
