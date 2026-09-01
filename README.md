# Home Credit Default Risk

An end-to-end probability of default (PD) project built on the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset. The workflow covers multi-table feature engineering, WoE/IV selection, an interpretable logistic-regression scorecard, a LightGBM benchmark, model validation, and credit strategy simulation.

> The [Chinese README](readme中文版.md) is the primary and complete project document. It includes the author's full original experiments, cutoff analysis, differentiated-pricing simulation, and monitoring design.

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.1-F7931E?style=flat-square&logo=scikitlearn)](https://scikit-learn.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-blue?style=flat-square)](https://lightgbm.readthedocs.io)

## Results

| Model | Train AUC | Holdout AUC | Holdout KS | Holdout Gini | PSI |
|---|---:|---:|---:|---:|---:|
| DummyClassifier | 0.500 | 0.502 | 0.003 | 0.003 | - |
| **Logistic Regression (WoE scorecard)** | **0.763** | **0.762** | **0.3989** | **0.524** | **0.0002** |
| LightGBM | 0.828 | 0.772 | 0.4149 | 0.5446 | - |

Logistic regression is selected for the scorecard because it offers competitive discrimination, full interpretability, and supports model review and governance. LightGBM is retained as a performance benchmark.

These metrics come from `notebooks/08_validation.ipynb`, using 246,008 training rows and 61,503 stratified random holdout rows (80/20, `random_state=42`) with 56 final features. PSI = 0.0002 indicates little score-distribution difference within this random split; it is not evidence of temporal stability.

## Strategy Summary

The strategy results are business simulations from `notebooks/09_strategy_cutoff_analysis.ipynb`, not model-discrimination metrics.

| Scenario | Cutoff | Approval Rate | Bad Rate | Simulated ROI |
|---|---:|---:|---:|---:|
| Fixed-rate constrained optimum | 520 | 37.3% | 3.0% | 4.5% |
| Differentiated-pricing constrained optimum | 520 | 37.3% | 3.0% | 6.7% |

The recommended portfolio scenario is now selected algorithmically: among integer cutoffs with approved bad rate at or below 3% and overall approval rate at or above 35%, maximize expected net profit under the fixed-rate assumptions. After correcting the scorecard intercept sign, this selects `Cutoff=520`; the same approval population is then used for the differentiated-pricing simulation. All ROI figures depend on stated lending, margin, review-pass, and loss assumptions and should be interpreted as scenario analysis.

## Pipeline

```text
Raw Home Credit tables
        |
        v
EDA and multi-table aggregation
        |
        v
164 candidate features
        |
        v
WoE binning + IV filtering + correlation filtering
        |
        v
56 final features
        |
        +--------------------------+
        |                          |
        v                          v
Logistic regression          LightGBM + SHAP
        |                          |
        v                          |
PDO scorecard <-------------------+
        |
        v
Validation -> cutoff, pricing and ROI simulation
```

Important engineered features include external credit scores, income and credit ratios, employment stability, bureau delinquency history, previous-application outcomes, installment delays, credit-card utilization, and POS/CASH payment behavior.

## Validation Protocol

The project uses a stratified random holdout split (`test_size=0.2`, `random_state=42`). The holdout is excluded from IV filtering, WoE fitting, and model training.

Home Credit does not provide a reliable application timestamp for a genuine out-of-time split. The reported holdout results therefore measure random-split generalization only and must not be described as future-customer or temporal validation.

Historical features from bureau, previous applications, installments, credit cards, and POS/CASH records are intended to use information available no later than the application decision time. Production implementation would need to enforce this point-in-time requirement explicitly.

## Original Work

In addition to the core scorecard workflow, the author completed four project-specific analyses:

1. IV-threshold sensitivity analysis (`0.02` versus stricter thresholds).
2. PDO sensitivity analysis and its effect on score dispersion.
3. OptimalBinning versus equal-frequency binning comparison.
4. Cutoff strategy, three-stage approval, fixed-rate versus differentiated-pricing ROI, and post-deployment monitoring design.

The full tables, reasoning, and conclusions are intentionally retained in the [Chinese README - Personal Contributions](readme中文版.md#个人增量贡献), which is the authoritative detailed version.

## Repository Structure

```text
home-credit-default-risk/
|-- README.md
|-- readme中文版.md
|-- requirements.txt
|-- environment.yml
|-- data/
|   |-- raw/                         # Kaggle CSV files, not tracked
|   `-- processed/                   # Generated parquet files, not tracked
|-- notebooks/
|   |-- 01_data_overview.ipynb
|   |-- 02_eda_main.ipynb
|   |-- 03_feature_engineering.ipynb
|   |-- 04_woe_iv_selection.ipynb
|   |-- 04.1_experiment_binning_comparison.ipynb
|   |-- 05_baseline_models.ipynb
|   |-- 06_lgbm_model.ipynb
|   |-- 07_scorecard.ipynb
|   |-- 08_validation.ipynb
|   |-- 09_strategy_cutoff_analysis.ipynb
|   `-- 10_final_summary.ipynb
`-- models/
    |-- README.md
    |-- artifacts_manifest.json
    |-- scorecard_table.csv
    `-- generated model and analysis artifacts
```

## Notebooks

| Step | Notebook | Main output |
|---:|---|---|
| 01 | Data overview | Schemas, table sizes, and join keys |
| 02 | Exploratory data analysis | Missingness, distributions, and statistical tests |
| 03 | Feature engineering | Multi-table aggregated master table |
| 04 | WoE / IV selection | WoE data, binning rules, and 56-feature list |
| 04.1 | Optional binning experiment | Optimal versus equal-frequency comparison |
| 05 | Baseline models | Logistic-regression scorecard model |
| 06 | LightGBM + SHAP | Performance benchmark and feature explanations |
| 07 | Scorecard | PDO-scaled scorecard and applicant scores |
| 08 | Validation | AUC, KS, Gini, PSI, and calibration |
| 09 | Strategy analysis | Cutoff, approval, bad-rate, pricing, and ROI scenarios |
| 10 | Final summary | Model comparison dashboard and conclusions |

## Quickstart

### Prerequisites

- Python 3.11.x (tested with 3.11.4)
- A Kaggle account for downloading the competition data

### Install

```bash
git clone https://github.com/Mekhty111/Home-Credit-Default-Risk.git
cd Home-Credit-Default-Risk

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Alternatively, create the pinned Conda environment:

```bash
conda env create -f environment.yml
```

### Download Data

```bash
kaggle competitions download -c home-credit-default-risk -p data/raw/
```

Extract the competition files into `data/raw/`, then start Jupyter:

```bash
jupyter notebook
```

Run notebooks `01` through `10` in order. Notebook `04.1` is an optional comparison experiment. Each main notebook produces artifacts consumed by later steps.

## Reproducibility

Direct dependencies are pinned in `requirements.txt`; `environment.yml` records the tested Python environment. Model pickle files are version-sensitive, so regenerate them after changing scikit-learn, LightGBM, OptBinning, or the feature list.

Artifact provenance, feature count, split definition, and serialization versions are documented in `models/README.md` and `models/artifacts_manifest.json`.

## Dataset

The project uses seven Home Credit tables, including the main application table, bureau history, previous applications, installment payments, credit-card balances, and POS/CASH balances. The main training table contains 307,511 applicants; `TARGET = 1` indicates default, with an approximately 8% positive rate.

## License

See [LICENSE](LICENSE).
