# Misinformation Diffusion Research

## Research question

How do Fake–Real differences in observed social-media engagement and in text detectability vary across health misinformation datasets that define and source misinformation differently? The primary comparison is CoAID News against FakeHealth HealthStory. The COVID-19 Rumor Dataset and two FakeNewsNet collections (PolitiFact, GossipCop) serve as external checks. Because period, labels, sources and collection differ together between datasets, results are contrasts between dataset contexts and cannot isolate a period effect.

This file is the working project README. The deposit README (`README.md` in the released repository) describes what the release contains and how to use it.

## Dataset roles

- **CoAID News**: fact-checked COVID-19 claims (Fake) against material from health agencies and health-information publishers (Real).
- **FakeHealth HealthStory**: mainstream health news labeled by expert-rated reporting quality.
- **COVID-19 Rumor Dataset**: independent COVID-19 rumors used as an external text target.
- **FakeNewsNet**: PolitiFact and GossipCop, used for out-of-domain engagement and title-only text checks.

Dataset identity always remains an explicit variable. The original label values and meanings were checked against each dataset's documentation, and terms such as “fake,” “real,” and “misinformation” are not assumed to be interchangeable across datasets.

## Data safety and layout

```text
data/
  raw/
    coaid/          # Put CoAID CSV files here
    fakenewsnet/    # Put FakeNewsNet raw files here
  processed/        # Derived, reproducible data only
notebooks/          # Exploratory and reporting notebooks
src/                # Reusable scripts
outputs/
  figures/          # Generated plots
  tables/           # Generated summary tables
```

**Never modify files under `data/raw/` directly.** Treat them as immutable source material. Cleaning, joins, label mappings, and derived variables belong under `data/processed/`, with dataset source retained explicitly.

Raw datasets are excluded from Git because they may be large and may have redistribution restrictions. Empty directories are retained with `.gitkeep` files.

## Setup (macOS/Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

To make the environment available in Jupyter or VS Code:

```bash
python -m ipykernel install --user --name misinformation-diffusion --display-name "Python (misinformation-diffusion)"
```

The optional kernel-registration command changes the user's Jupyter configuration and is therefore not run automatically by project setup.

## Inspect CoAID

Place downloaded CoAID CSV files anywhere beneath `data/raw/coaid/`, preserving their original filenames and subdirectories. Then run:

```bash
.venv/bin/python src/inspect_coaid.py
```

The script reads raw files without modifying them and writes `outputs/tables/coaid_dataset_summary.csv`.

## Python versions and environments (release note, 2026-10-13)

- **Classical and engagement analyses:** Python **3.14.2** in `.venv`, with `requirements.txt` (NumPy 2.5.1, pandas 3.0.5, scikit-learn 1.9.0, SciPy 1.18.0, statsmodels 0.15.0, joblib 1.5.3, threadpoolctl 3.6.0, matplotlib 3.11.1).
- **Transformer analyses:** Python **3.12.7** in `.venv-transformer`, with `requirements-transformer.txt` (PyTorch 2.14.0, Transformers 5.17.0; checkpoint `distilbert/distilroberta-base` at revision `fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b`). Reported transformer results were produced on Apple Silicon (MPS). The scripts now select `mps`, then `cuda`, then `cpu`, and log a warning that bitwise reproducibility across devices is not claimed.
- **Review-package rendering:** any Python 3.12+ with `requirements-render.txt` plus a Chrome/Chromium executable (`REVIEW_CHROME`).
- The versions above are the ones recorded in `outputs/tables/engagement_model_run_metadata.json`, `outputs/tables/transformer_environment_audit.csv` and Supplementary Table S7a.
