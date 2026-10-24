# FedMACD

FedMACD is a federated multi-agent causal discovery framework that reconciles statistical evidence with variable semantics. Raw observations remain at the clients; the server receives evidence packages, debates candidate edges, judges their existence and orientation, and removes directed cycles deterministically. This is not a differential privacy guarantee: uploaded statistics and semantic interactions may still reveal information.

This repository is the paper-aligned public source release. The authoritative experiment configuration is **v4.36**; older notes and backup scripts are outside this repository.

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

The heterogeneous scenario preserves the causal graph while inducing client-specific distribution shift. It is a shared experimental objective, not one identical sampling algorithm:

- Discrete BIF datasets: each client independently draws observations from a client-specific softly intervened Bayesian network.
- Continuous CSV datasets: each client independently receives a preference-region-weighted draw without replacement within that client's draw. Source rows may occur at more than one client.

The default heterogeneity strength is 0.7. The paper and code describe this as a **type-specific heterogeneous construction**.

The two implementations differ because the BIF benchmarks provide an explicit
Bayesian network and conditional probability tables: client-specific soft
interventions can therefore generate new observations while preserving the
DAG. The continuous benchmarks provide fixed observational tables, so using
the same mechanism would require fitting an additional generative model and
would introduce assumptions not present in the study. Their weighted
resampling instead creates client-specific preference-region shifts directly
from the observed records. Consequently, a strength value such as 0.7 controls
the shift within each construction; it is not a claim that discrete and
continuous shifts are numerically identical.

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

Private working files, old result trees, logs, prompt-response traces, local environments, third-party source snapshots, submission correspondence, and secrets are deliberately excluded from the public release.

## Environment

The source-release development environment used Python 3.12.7 and the package versions in `requirements.txt`. Install:

```bash
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add only the provider keys you intend to use. Never commit `.env`. Some registered algorithms require optional libraries; unavailable algorithms are filtered from the candidate pool at runtime.

## Data

The public release distinguishes regenerated study data from third-party source data. The data dictionary and acquisition instructions are in [`docs/DATASETS.md`](docs/DATASETS.md).

- Bayesian-network structures are used to generate 10,000-observation study instances by forward sampling.
- ER20 can be regenerated with the documented NOTEARS-ADMM simulation utility and settings; the regenerated input was checked against the historical matrix to floating-point precision.
- Auto MPG, DWD climate, Sachs, and 10nodes retain their cited provenance; redistribution is governed by their original sources.
- Only 10nodes and ER20 use anonymous numeric variable names and therefore have no usable semantic descriptions. Survey and Sachs have readable variable semantics, although their orientations remain challenging.

After obtaining study inputs, point the code to the parent directory of the dataset paths described in `docs/DATASETS.md`:

```bash
# PowerShell
$env:FEDMACD_DATA_DIR = "path\to\local-datasets"

# bash/zsh
export FEDMACD_DATA_DIR=/path/to/local-datasets
```

To prepare the three MATMCD tables, 10nodes and generated ER20, obtain the
pinned LLM-and-SCD (linked by MATMCD), FedPC and NOTEARS-ADMM source checkouts, then run
`scripts/prepare_public_datasets.py` as described in `docs/DATASETS.md`.
Download current bnlearn BIF files separately with
`python scripts/prepare_public_datasets.py --output-root /path/to/local-datasets --download-bnlearn`.
The script checks upstream BIF hashes and restores the historical Asia block order,
which affects seeded sampling. It also verifies checksums of the four external
continuous inputs. See the data guide for ER20 numerical verification and environment requirements.

## Experiment results

The companion [Zenodo record](https://doi.org/10.5281/zenodo.23611058) contains the paper's experiment outputs: fold-level records, predicted graphs, metrics, and selection ledgers. It is independent of the third-party input sources documented in `docs/DATASETS.md`.

## Reproducing a main fold

After preparing the inputs above, run from the repository root:

```bash
python scripts/run_v436_main.py asia homogeneous 0
```

This command runs one fold and may call a paid API. New runs can differ from the reported results because hosted LLM outputs and optional algorithm availability may vary. Use the archived fold records to regenerate the reported tables and figures; a new API run is a methodological replication, not a promise of identical metrics. Select a paper model before launching a new process:

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

## Public-release boundaries

GitHub contains the core implementation, processing and experiment code,
metrics, general visualization, paper figure/table scripts, tests, and docs.
Zenodo contains curated experiment results, a result manifest, and checksums.

This source repository is hosted at https://github.com/ardorkyphy/FedMACD-Experiment. Its Zenodo companion contains experiment results.

## License

No open-source license has yet been selected. Until the authors add a license file, copyright is reserved and public availability does not grant reuse rights. Third-party data and software retain their original licenses.
