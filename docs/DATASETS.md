# FedMACD dataset guide

This file is the public data dictionary for the ten datasets used by the paper. It records the dataset role, dimensions, variable order, graph convention, and preprocessing. The repository does not silently substitute a different dataset with the same name.

## Availability

The five Bayesian-network structures are third-party BIF files. ER20 is study-generated. The code and Zenodo results archive do not bundle any input dataset. Acquire the BIFs and external tables from their sources, or generate ER20 with the study utility, then place them under `FEDMACD_DATA_DIR` using the layout below. The exact historical files for four external tables cannot currently be recovered from a verified public locator.

The four external tables used in the historical study are identified by provenance and SHA-256 records in the private study materials. Those hashes are not hashes of a public download unless the source file and conversion chain are independently verified.

## Dataset table

| Dataset | Type | Shape used in paper | Variables / columns | Source and acquisition | Processing and graph convention |
|---|---|---:|---|---|---|
| Asia | Bayesian network | 10,000 x 8 | BIF-defined names | bnlearn Bayesian-network repository, `asia.bif` | Forward sample the BIF; graph edges are read from the BIF in row-to-column direction. |
| Child | Bayesian network | 10,000 x 20 | BIF-defined names | bnlearn Bayesian-network repository, `child.bif` | Forward sample the BIF; graph edges are read from the BIF in row-to-column direction. |
| Alarm | Bayesian network | 10,000 x 37 | BIF-defined names | bnlearn Bayesian-network repository, `alarm.bif` | Forward sample the BIF; graph edges are read from the BIF in row-to-column direction. |
| Survey | Bayesian network | 10,000 x 6 | BIF-defined names | bnlearn Bayesian-network repository, `survey.bif` | Forward sample the BIF; graph edges are read from the BIF in row-to-column direction. |
| Earthquake | Bayesian network | 10,000 x 5 | BIF-defined names | bnlearn Bayesian-network repository, `earthquake.bif` | Forward sample the BIF; graph edges are read from the BIF in row-to-column direction. |
| Auto_MPG | Continuous table | 392 x 5 | `Displacement`, `Mpg`, `Horsepower`, `Weight`, `Acceleration` | UCI Auto MPG, dataset 9: https://archive.ics.uci.edu/dataset/9/auto+mpg | Use the five-column study table; the study graph was converted from the source col-to-row adjacency convention to row-to-column. Exact study-file version is not verified. |
| DWD_climate | Continuous table | 349 x 6 | `Altitude`, `Temperature`, `Precipitation`, `Longitude`, `Sunshine`, `Latitude` | Tuebingen Cause-Effect Pairs / DWD climate: https://webdav.tuebingen.mpg.de/cause-effect/ | Use the six-column study table; convert the source col-to-row adjacency to row-to-column. Exact pair/file version is not verified. |
| Sachs | Continuous table | 7466 x 11 | `raf`, `mek`, `plc`, `pip2`, `pip3`, `erk`, `akt`, `pka`, `pkc`, `p38`, `jnk` | Sachs protein-signalling data; the bnlearn Sachs page is a network reference, not the continuous table: https://www.bnlearn.com/bnrepository/discrete-small.html#sachs | Use the 11-column study table; convert the source col-to-row adjacency to row-to-column. Exact continuous-table locator/version is not verified. |
| 10nodes | Synthetic continuous table | 5000 x 10 | `X0` ... `X9` | Ten-node benchmark distributed with the FedPC experiment code; no verified public URL was identified during release preparation. | Whitespace-delimited source data converted to headerless CSV; retain the supplied row-to-column graph. |
| ER20 | Synthetic continuous table | 256 x 20 | Numeric columns | Generated with the NOTEARS-ADMM simulation utility. | Simulation settings in the historical preparation script: `set_random_seed(1)`, `simulate_dag(d=20, s0=20, graph_type="ER")`, `simulate_linear_sem(B_true, n=256, sem_type="gauss")`; save data without a header and the graph in row-to-column form. |

The five BIF network files are listed at https://www.bnlearn.com/bnrepository/. Verify their structure and variable order before use; a public download is not asserted to be byte-identical to the local historical file.

Expected local layout: `bayesian_network_structures/{asia,child,alarm,survey,earthquake}.bif`; `{Auto_MPG,DWD_climate,Sachs,10nodes,ER20}/data.csv` and `graph.txt`. `FEDMACD_DATA_DIR` must point to the parent of those paths. A source page alone does not supply the study-specific graph conversion for the four external tables.

## Historical local-file checksums

These are SHA-256 hashes of the study's local `data.csv` and `graph.txt` files. They help identify the exact input if it is found; they do not establish a public source location or permission to redistribute it.

| Dataset | data.csv SHA-256 | graph.txt SHA-256 |
|---|---|---|
| Auto_MPG | `32dddbccc91b57c4c03505b9bd7b4ab221709d0339886ce18f573b6aba2a7fad` | `1a3fed17979b125c4943525fb34f64c4d20498015dbda587209797ab5ec07051` |
| DWD_climate | `c3426411aaa857b9c60e6a99b859336dc0edc2fdbafe29e8594914f3ab7e6041` | `dba3a21b858140b09551477751d47c54099c310d3d5c1255f89b2e73d860540b` |
| Sachs | `70cedff4412c53e17610d662ed1d7e7dd8bc8ec103ff0c741a70d339f8f34606` | `7be7ddcf108b32ad612c7e48f42ad386565a94afedcdb3901ee052a03d5a0f39` |
| 10nodes | `5913e900d532738704839e0371d45c18a43b65fff888dea2600fe1b938fb6b10` | `1d981c99be2cf83a64f3ada23c2032ab289d70f1c7bb429392d8ac970fd47a4a` |

## Reconstructing a fold

The paper uses homogeneous and heterogeneous scenarios, seeds 0/1/2, and four clients. Bayesian-network folds are generated by client-specific soft interventions and independent sampling. Continuous folds use preference-region-weighted sampling without replacement within each client; source records may appear in more than one client. The default heterogeneity strength is 0.7. These settings are implemented in `src_v4/` and the `scripts/run_v436_*.py` drivers.

For Auto_MPG, DWD_climate, Sachs, and 10nodes, a complete public byte-identical reconstruction is currently unavailable because the exact historical source versions and conversion chains were not retained. The Sachs page does not locate the continuous study table, and no verified 10nodes URL is known. Do not describe a newly downloaded table as the historical input without checking its checksum and conversion steps.

## References

- The exact dimensions and column orders above follow the latest KBS and DMKD manuscripts and the release configuration.
- See `SOURCE_DATA_LIMITATIONS.md` for unresolved provenance items. The Zenodo results archive documents the reported experiment outputs separately.
