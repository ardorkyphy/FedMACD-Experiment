# FedMACD

FedMACD is a federated multi-agent causal discovery framework that reconciles statistical evidence with variable semantics. Raw observations remain at the clients; the server receives evidence packages, debates candidate edges, judges their existence and orientation, and removes directed cycles deterministically. This is not a differential privacy guarantee: uploaded statistics and semantic interactions may still reveal information.

This repository provides the code and data-preparation instructions for the experiments reported in the paper. The experiment configuration is v4.36.

## Paper configuration

- 10 datasets: Asia, Child, Alarm, Survey, Earthquake, Auto MPG, DWD climate, Sachs, 10nodes, and ER20.
- 2 client scenarios: homogeneous and heterogeneous.
- 3 split seeds: 0, 1, and 2.
- 4 clients per fold.
- 3 reasoning backbones: qwen3.8-flash, deepseek-v4-flash, and glm-5.3-flash.
- A registered pool of local causal-discovery algorithms; the available pool depends on installed optional packages.
- Two protocol stages: edge existence followed by edge orientation.
- The five Bayesian-network experiments use 10,000 generated observations, including Alarm.

## Heterogeneous construction

The heterogeneous scenario preserves the causal graph while inducing client-specific distribution shift. Its implementation depends on the benchmark type:

- Discrete BIF datasets: each client independently draws observations from a client-specific softly intervened Bayesian network.
- Continuous CSV datasets: each client independently receives a preference-region-weighted draw without replacement within that client's draw. Source rows may occur at more than one client.

The default heterogeneity strength is 0.7.

For BIF benchmarks, clients sample from Bayesian networks with client-specific soft interventions. For continuous benchmarks, clients receive client-specific weighted samples from the observed records; the same source row may occur at more than one client. The strength parameter controls the shift within each method.

## Repository layout

| Path | Purpose |
|---|---|
| `src_v4/` | FedMACD implementation and the paper experiment drivers |
| `scripts/run_v436_main.py` | Reproduce one main-experiment fold |
| `scripts/run_v436_ablation.py` | Reproduce one component-ablation fold |
| `scripts/run_v436_bonus.py` | Run additional analyses with the DeepSeek provider |
| `scripts/run_v436_bonus_zhipu.py` | Run the paper-reported additional analyses with the GLM provider |
| `src_v4/util/results/` | Compute structural metrics and render per-run result summaries |
| `tests/` | Algorithm and data-layer regression tests |
| `scripts/paper_figures/` | Regenerate paper figures and tables from the extracted Zenodo results |

The repository includes the experiment code, data-preparation scripts, tests, and documentation.

## Environment

The source-release development environment used Python 3.12.7 and the package versions in `requirements.txt`. Install:

```bash
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add only the provider keys you intend to use. Never commit `.env`. Some registered algorithms require optional libraries; unavailable algorithms are filtered from the candidate pool at runtime.

## Data

The public release distinguishes regenerated study data from third-party source data. The data dictionary and acquisition instructions are in [`docs/DATASETS.md`](docs/DATASETS.md).

- Bayesian-network structures are used to generate 10,000-observation study instances by forward sampling.
- ER20 is generated with the NOTEARS-ADMM simulation utility.
- Auto MPG, DWD climate, Sachs, and 10nodes retain their cited provenance; redistribution is governed by their original sources.

Prepare the inputs and set `FEDMACD_DATA_DIR` to their parent directory as described in [`docs/DATASETS.md`](docs/DATASETS.md).

## Experiment results

The companion [Zenodo record](https://doi.org/10.5281/zenodo.23615384) contains the paper's experiment outputs: fold-level records, predicted graphs, metrics, and selection ledgers. It is independent of the third-party input sources documented in `docs/DATASETS.md`.

## Reproducing a main fold

After preparing the inputs, run from the repository root:

```bash
python scripts/run_v436_main.py asia homogeneous 0
```

This command runs one fold and may call a paid API. Hosted model outputs and optional algorithm availability can change, so a new run may not reproduce the exact archived metrics. To regenerate the reported figures and tables without API calls, use the archived fold records as described below. Select a model before launching the process:

```powershell
$env:FEDMACD_PROVIDER = "deepseek" # deepseek-v4-flash, default
$env:FEDMACD_PROVIDER = "qwen"     # qwen3.8-flash
$env:FEDMACD_PROVIDER = "zhipu"    # glm-5.3-flash
```

Set only one value at a time. The main and component ablation drivers separate results by provider and by model name if its configured name changes. The paper component ablations use `zhipu`:

```powershell
$env:FEDMACD_PROVIDER = "zhipu"
python scripts/run_v436_ablation.py no_debate child heterogeneous 0
```

The GLM additional analysis driver fixes its provider to `zhipu`. Valid examples are:

```powershell
python scripts/run_v436_bonus_zhipu.py strength survey heterogeneous 0 0.3
python scripts/run_v436_bonus_zhipu.py nc child homogeneous 0 7
python scripts/run_v436_bonus_zhipu.py rounds1 survey homogeneous 0
python scripts/run_v436_bonus_zhipu.py rounds3 DWD_climate heterogeneous 0
```

Strength 0.7, four clients, and two rounds come from main folds; zero rounds comes from the `no_debate` ablation. The complete main design is 10 datasets × 2 scenarios × 3 seeds = 60 folds per model. For an offline check that makes no API calls, use:

```bash
python -B -m unittest discover -s tests -p "test_*.py" -v
```

## Result metrics and visualization

Each experiment fold uses `src_v4/util/results/` to compute structural metrics
and produce its local `metrics_v4.png` summary. Set `FEDMACD_RESULTS_DIR` to
the extracted Zenodo `results/` directory and run scripts under
`scripts/paper_figures/` to regenerate paper figures and tables without API
calls.

To regenerate the ten-dataset component-ablation heatmap (Fig. 5) as PDF and EPS:

```bash
python scripts/paper_figures/fig5_ablation.py --results-dir /path/to/extracted-zenodo/results --output-dir /path/to/figure-output
```

The script reads the main glm-5.3-flash ledger and all 180 ablation records from the
Zenodo results archive. The archive contains the underlying results; the figure is
generated by this repository's script.

## Public-release boundaries

GitHub contains the core implementation, processing and experiment code,
metrics, general visualization, paper figure/table scripts, tests, and docs.
Zenodo contains curated experiment results, a result manifest, and checksums.

This source repository is hosted at [GitHub](https://github.com/ardorkyphy/FedMACD-Experiment); the experiment records are available from [Zenodo](https://doi.org/10.5281/zenodo.23615384).

## License

No open-source license has yet been selected. Until the authors add a license file, copyright is reserved and public availability does not grant reuse rights. Third-party data and software retain their original licenses.
