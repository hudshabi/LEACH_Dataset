# Yield-driven cluster formation under finite retries

Research files for **Yield-Driven Cluster Formation under Finite Retries in
LEACH-Style Wireless Sensor Networks: A Matched-Energy Selection-and-Confirmation Study**.
Authors: Mohammed Al-Shabi and Anmar Abuhamdah.

This repository contains the files used for the main cluster-formation and timing
experiments, their scored outcomes, the figures, and the one-round forecast check.
It is a curated subset of the original `Enhanced_LEACH_Study.zip`, not a new experiment.
Simulation data are outputs of executed simulations, not field measurements.

## Included scope

| Stage | Scenarios | Scored simulations | Role |
| --- | --- | ---: | --- |
| Development | 7000–7005, nominal condition | 54 | Select horizon and periodic comparator |
| Exploratory extension | 8000–8019, four conditions | 2,880 | Evaluate gate and select repair-only formation policy |
| Fresh confirmation | 9000–9039, four conditions | 5,120 | Confirm selected policy, including the added uncapped comparator |
| Independent one-round moment check | 10,000 stationary-channel realizations | Not counted as full scored simulations | Check affordable retry and delivery moments |

The three scored stages total **8,054 simulations**. Conditions are nominal,
poor channel, far sink and heterogeneous energy. Sensor-energy caps are
10, 20, 30 and 40 J. Confirmation uses 40 paired scenario clusters, averaging
four conditions within each scenario; 5,120 runs are not 5,120 independent replicates.

All methods, conditions and outcomes in those cohorts are retained, including
negative and inconclusive results. `extension/confirmation_metrics.csv` contains
4,480 runs and `confirmation_extra_metrics.csv` the 640 added comparator runs.

## Main results and interpretation

At 40 J, the selected formation policy delivers 6,693.70 mean source samples
versus 6,230.65625 for the original formation heuristic with mandatory repair:
+7.43%, Holm p = 9.31496e-14. This is an executed heuristic comparator, not a
faithful reproduction of published LEACH-RLC MILP performance.

Against five-round periodic reclustering, the pooled mean change is +1.74%, but
is inconclusive after correction (Holm p = 0.0631346). Scheduled sensor control
bits are 56.82% lower than with that schedule, but 7.38% higher than with the
original repair-only comparator. Mean elapsed durations differ: 224.94 versus
198.07 attempted rounds against the five-round schedule. Equal energy does not
mean equal time or longer lifetime.

Against the five-round schedule, condition-specific corrected delivery gains
occur in poor channel (+3.73%, Holm p about .0494) and heterogeneous energy
(+2.87%, Holm p about .00285). The nominal comparison is inconclusive; far sink
has a -2.42% mean difference without corrected significance. The survival
adjustment has no demonstrated independent gain over the fairer uncapped
forecast (+0.71%, Holm p about .270). The extra voluntary gate is not a successful
component. Do not reinterpret non-significance as equivalence.

## Repository contents

- Root Python modules: scored simulator, formation optimizer, evaluation,
  analyses, validation and nine extension tests.
- `extension/*.json`: actual plans, selection, amendment, outcomes and source-version map.
- `extension/*.csv`: episode-level inputs and published numerical summaries,
  including raw and Holm-corrected p values.
- `extension/ablation_cost.png`, `confirmation_yield.png`,
  `confirmation_ablation.png`: the three plotted figures.
- `raw_round_logs/*.zip`: original per-run metrics, round traces and gate decisions,
  grouped by stage, condition and seed block; no data values changed.
- `raw_round_logs/index.json`: archive inventory.
- `SOURCE_MAP.json`: source paths and SHA256 hashes for byte-identical copied files.
- `SHA256SUMS.txt`: integrity hashes for this curated distribution.

The source imports `run_study.py` and `run_equal_budget.py`, which import
`design.json` and `final_validation_plan.json`. These are retained as required
compatibility dependencies. Their historical training/evaluation entry points
are not workflows supported by this reduced distribution. The main-study
protocols are the JSON files under `extension/`.

## Reproduce tables and figures from the supplied scored data

Use Python 3.12. The analysis-only requirements do not need trained models or Torch:

```bash
python -m pip install -r requirements-analysis.txt
python analyze_extension.py
python analyze_confirmation.py
```

Both commands read the supplied episode-level CSVs and rebuild statistical
summaries and figures. The first script also produces an additional diagnostic
plot; only the three figures listed above are included in this distribution.
Bootstrap intervals are descriptive, not simultaneous. Pooled Holm family size
is seven in confirmation and condition-specific secondary family size is 28.
See the original plans and amendment for selection chronology.

## Rerun the simulations and checks

For the original simulator imports, install the recorded full dependencies:

```bash
python -m pip install -r requirements.txt
python extension_protocol.py development
python extension_protocol.py test
python analyze_extension.py
python extension_protocol.py confirmation
python run_forecast_ablation.py
python analyze_confirmation.py
python validate_extension_forecast.py
python -m unittest test_extension -v
```

Torch, Gymnasium and Stable-Baselines3 are inherited module dependencies, even
though the current extension is deterministic and trains no new DQN. No trained
weights or external datasets are needed for these workflows. Run from the
repository root; do not execute the historical training entry points in
`base_simulator.py`, `run_study.py` or `run_equal_budget.py` separately.

The supplied selection freezes H=20 and period=20. Development reselects them
by its documented rule. Runtime varies by hardware and excludes initialization;
all scientific outcomes should reproduce under the recorded versions. Four
workers run the principal simulations, two run the extra comparator.

The nine extension tests are supplied; the earlier statement that 39 tests passed
refers to the full historical archive, not to 39 tests in this reduced package.
The 10,000-replicate one-round check does not validate multi-round battery-survival
forecasts, and same-project verification is not external-team replication.

## Access the complete original round traces

Analyses above do not require extraction of the raw-log archives. To inspect or
restore the original directory layout, run:

```bash
python unpack_round_logs.py
```

It restores paths under `extension/development`, `extension/test`,
`extension/confirmation` and `extension/confirmation_extra`. Every retained raw
file is an actual output of the scored main experiments. `python verify_files.py`
checks distribution integrity before extraction or reanalysis.

## What is intentionally excluded

Earlier RL training/checkpoints, reward-intervention and old validation datasets,
legacy supplied materials, superseded figures and manuscript drafts, Word/PDF
files, execution console logs, caches and dependencies themselves are excluded.
The earlier RL investigation is context, not an additional replication of this
main deterministic study. This repository does not reproduce every historical
claim from previous manuscript versions or their appendices. No unpublished
third-party reference article is redistributed here.

## Licenses and attribution

`LICENSE` applies the existing MIT license to the included newly authored Python
code. `LICENSE_DATA.md` preserves the existing CC BY 4.0 attribution for new
simulation data, configuration records and figures. Dependencies retain their
own licenses and are installed separately. No license to the manuscript or
third-party publications is granted by this code/data distribution.

`CITATION.cff` provides the article title and authors. No DOI, accepted-paper
status, public release URL or repository address has been invented. Add the
real repository URL and any later publication DOI when they exist.

## Publish on GitHub

1. Extract the outer distribution ZIP on your computer.
2. Create a repository and set its visibility to **Public** if public access is intended.
3. Upload the **contents of this directory**, keeping README.md at the repository root.
   Do not upload the large outer ZIP as the only repository file.
4. Keep the smaller `raw_round_logs/*.zip` files compressed. Each is below 25 MiB.
   For the full directory tree, GitHub Desktop or Git is convenient; the web uploader
   accepts up to 100 files at once and 25 MiB per file.
5. After the actual upload, cite the real repository link in the paper's Data and
   Code Availability section. Publication of this prepared archive has not yet occurred.

GitHub upload documentation:
https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository
