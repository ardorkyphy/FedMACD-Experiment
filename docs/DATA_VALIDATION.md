# Input validation details

The following checks support the acquisition steps in [DATASETS.md](DATASETS.md).

## Upstream source files

The source files are pinned to commits listed in the dataset guide. Git blob IDs of individual files: Auto MPG data `1114e0af8a71a839b26b55ebb2d77047c58b0cee`, graph `e07b2315089c7937a46089b00483ad5c382b4f37`; DWD climate data `6870e05a7a874374794fb39d64494eab81f322e0`, graph `cad65ed03224912fc899f195ca9d81663b9f3123`; Sachs data `217e096444280fc24548fd37643129421acc6398`, graph `3fd032304bac2207de85756beafa4a686a20c54d`; FedPC 10nodes observations `3fb2af4886a043ab178bafd754186946097a12db`, graph `e444e41f534958bc69b76441f44c1a5e08f82d5e`.

## Prepared input checksums

SHA-256 hashes identify the converted study inputs, rather than the original third-party files.

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

## Bayesian-network files

SHA-256 of the decompressed bnlearn BIF files checked by the preparation script:

| Network | SHA-256 |
|---|---|
| Asia | `9f770c96940dc4d860b581602120790ac538983b30b28d38631b8184dc7b5f89` |
| Child | `432c22e661cd0e8235c8d95599ac22ccc261002459f5c4e6e7235b057908fe6e` |
| Alarm | `701e6c561f71b55669070c29614f0724b761289aa2c4a35bcc97b638ee881fa2` |
| Survey | `3698ccc0f1f74b9f935617d91989e4cdb39edb5360b558216af64047fc88b3b0` |
| Earthquake | `74dd131e2c6437a39de17f2963d2d16fadd91e70450d1fc20501df05e8dc4f2e` |

The bnlearn Asia file has the same nodes, edges, states and conditional probabilities as the experiment input. The preparation script restores the alphabetical variable/probability block order used by the experiment, so node, edge and CPD order also match. Forward samples of 10,000 rows were compared for seeds 0, 1 and 2. The other four BIF files match the experiment inputs byte-for-byte.

## ER20 regeneration

With the pinned NOTEARS-ADMM source and `igraph==1.0.0`, the generated graph matches the experiment graph byte-for-byte. The 256 x 20 data matrix agrees numerically; maximum absolute difference was approximately `1.3e-15` in the checked environment. CSV byte-for-byte identity can vary across numerical libraries and serialization environments. The `simulate_parameter(graph)` call is retained because it advances the random-number state used to generate the observations.
