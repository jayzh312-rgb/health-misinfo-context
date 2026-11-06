# Extension v1, round 2 — encoding-artifact sensitivity and FakeNewsNet engagement

Date: 2026-10-12. Frozen baseline untouched: nothing under `outputs/tables/`, `outputs/models/`, `outputs/manuscript/` or `data/processed/` was modified. Protocols were written before any label- or outcome-dependent computation (`protocols/coaid_encoding_sensitivity.md`, `protocols/fakenewsnet_engagement_protocol.md`). Scripts: `scripts/task_a_encoding_audit.py`, `scripts/task_b_fakenewsnet_engagement.py`. Log: `logs/extension_v1_fakenewsnet_engagement.log`.

## Task A — CoAID encoding-artifact audit and sensitivity

### A1. Prevalence (`tables/coaid_encoding_prevalence.csv`)

Population: the frozen combined-text cohorts (3,577 CoAID, 1,647 FakeHealth) loaded through the validated baseline loader. A label-blind character scan showed that CoAID's non-ASCII content is dominated by `Ò` (184 documents), `Ó` (182) and `Õ` (73), the Latin-1 readings of Mac-Roman curly quotes; FakeHealth's is dominated by proper curly quotes and dashes. No Windows-1252/UTF-8 sequences (`â€`, `Ã`, `Â`) and no U+FFFD occur in either cohort.

Share of documents containing at least one match:

| Dataset, class | N | Any mojibake | Mac-Roman quote substitute | C1 control | Year token | Month name | Month name excl. "may" | Credit fragment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CoAID Fake | 825 | **25.9%** | 25.9% | 0.0% | 8.4% | 8.7% | 5.7% | 0.1% |
| CoAID Real | 2,752 | **0.0%** | 0.0% | 0.0% | 49.0% | 57.3% | 30.0% | 24.8% |
| FakeHealth Fake | 459 | 0.4% | 0.0% | 0.4% | 58.8% | 73.0% | 32.9% | 23.1% |
| FakeHealth Real | 1,188 | 0.2% | 0.0% | 0.2% | 62.5% | 76.4% | 34.3% | 17.9% |

By CoAID publisher family (families with ≥20 eligible rows): the Mac-Roman substitutes are confined to Fake rows with unresolved or missing publisher identity (607 rows, 33.8%) and to instagram.com (21 rows, 33.3%); every identified Real family (cdc.gov, healthline.com, medicalnewstoday.com, nih.gov, webmd.com, who.int) has 0.0%. Year tokens concentrate in webmd.com (95.7%) and cdc.gov (73.3%); credit fragments in healthline.com (69.6%) and webmd.com (45.0%). The encoding artifact is therefore a perfect class marker in CoAID (present in a quarter of Fake documents, absent from all Real documents), and the date and credit tokens are near-perfect Real markers.

### A2. Sensitivity under the locked protocol (`tables/coaid_encoding_sensitivity.csv`)

ftfy is not installed in `.venv`; a hand-written mapping was used (recorded in the script and protocol). Locked configuration cfg_05 (unigram, min_df 2, C 10, balanced), saved 2,875/702 training-pool/holdout split, unchanged pipeline, no reselection. The Mac-Roman rules touched 182 (`Ò`), 63 (`Õ`) and 61 (`Ó`) CoAID documents; the cp1252 mapping touched none; NFKC changed 8.

| Variant | Macro F1 | ROC-AUC | AP | Balanced acc. | Fake F1 | Vocab | Train / holdout docs changed |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen saved | 0.9260 | 0.9827 | 0.9280 | 0.9313 | 0.8876 | 6,428 | — |
| Natural refit check | 0.9260 | 0.9827 | 0.9280 | 0.9313 | 0.8876 | 6,428 | 0 / 0 |
| Repaired (mojibake map + NFKC) | 0.9281 | 0.9824 | 0.9264 | 0.9343 | 0.8909 | 6,412 | 171 / 51 |
| Repaired + years, months, credits removed | 0.9188 | 0.9793 | 0.9168 | 0.9276 | 0.8772 | 6,379 | 2,651 / 658 |

Reading: the frozen result reproduces exactly. Repairing the encoding artifacts leaves holdout separability unchanged (macro F1 +0.002, ROC-AUC −0.0003): the mojibake tokens are redundant with other Fake cues, not the carrier of the signal. Removing dates, month names and credit fragments as well lowers macro F1 by 0.007 and ROC-AUC by 0.003; CoAID separability is not materially dependent on these formatting tokens either, although they are strongly class-associated. Residual: `Ô` (11 documents, probably a left-single-quote substitute) was not in the mapping; the "may" removal also removes the modal verb (month share excl. "may" is reported alongside).

## Task B — FakeNewsNet engagement under the frozen estimand

Protocol fixed in advance (see file), including the declaration that feasibility medians were already observed. Rows per subset after excluding the two label-conflicting PolitiFact ids: PolitiFact 1,052 (430 Fake, 622 Real), GossipCop 22,140 (5,323 / 16,817). Outcome: distinct tweet IDs per row, zeros included (255 and 1,246 zero rows). Design: intercept + Fake indicator (no release structure exists). Replies and retweets are unavailable.

### Dependence graph (`tables/fakenewsnet_overlap_graph_summary.csv`)

| Subset | Components | Multi-row components (rows) | Largest component rows (Fake/Real) | Largest component share of post memberships |
|---|---:|---:|---:|---:|
| PolitiFact | 948 | 18 (122) | 84 (5 / 79) | **66.7%** |
| GossipCop | 20,777 | 371 (1,734) | 692 (582 / 110) | 19.1% |

PolitiFact has one component of 84 mostly-Real rows that carries two thirds of all post memberships; its resampling dominates the interval. GossipCop's largest component is mostly Fake and carries a fifth of memberships.

### Family selection and estimates (`tables/fakenewsnet_engagement_model_comparison.csv`, `tables/fakenewsnet_engagement_results.csv`)

The frozen rule selected the **hurdle (logit + zero-truncated NB2)** in both subsets: AIC improvement over NB2 of 128.7 (PolitiFact) and 81.4 (GossipCop), both fits stable. 1,000 successful component-bootstrap refits were reached in 1,002 and 1,001 attempts (2 and 1 failures; 106 and 45 same-draw multistart retries). Because the design has only a Fake indicator, the standardized means equal the fitted class means and coincide with the raw class means.

| Subset | Standardized Fake mean | Standardized Real mean | Fake/Real ratio [95% percentile] | P(posts>0) Fake / Real | E(posts \| >0) Fake / Real | Conditional ratio [95%] |
|---|---:|---:|---:|---:|---:|---:|
| PolitiFact | 375.0 | 670.5 | **0.559 [0.439, 1.868]** | 0.907 / 0.654 | 413 / 1,025 | 0.404 [0.336, 1.255] |
| GossipCop | 112.4 | 52.4 | **2.144 [1.212, 3.290]** | 0.965 / 0.937 | 117 / 56 | 2.083 [1.185, 3.179] |

PolitiFact: Fake items are more likely to have any tweet (0.91 vs 0.65) but Real items that are tweeted have far more tweets, driven by the 84-row component; the ratio's interval is wide and right-skewed (bootstrap median 0.62, upper bound 1.87), so direction is not settled. GossipCop: Fake items have more expected tweets than Real items by a factor of about two, with the interval excluding 1; the medians run the other way (12 vs 45), so the mean ratio reflects a heavy Fake tail rather than typical items.

### Ratio-of-ratios against the saved references (`tables/fakenewsnet_engagement_contrast.csv`)

Full Cartesian product of 1,000 FakeNewsNet draws with the 1,000 saved reference draws, frozen `independent_product`, linear quantiles. C = R_subset / R_reference.

| Subset | Reference (saved ratio) | Ratio-of-ratios | 95% interval | Bootstrap median | Excludes 1? |
|---|---|---:|---:|---:|---|
| PolitiFact | CoAID A (0.2528) | 2.212 | [0.987, 9.894] | 2.597 | no (lower bound 0.99) |
| PolitiFact | FakeHealth (0.8403) | 0.666 | [0.488, 2.291] | 0.686 | **no** |
| GossipCop | CoAID A (0.2528) | 8.481 | [2.821, 18.225] | 7.196 | yes |
| GossipCop | FakeHealth (0.8403) | 2.552 | [1.357, 4.087] | 2.447 | **yes** |

### Duplicate-title sensitivity (`tables/fakenewsnet_duplicate_title_sensitivity.csv`)

Dropping all rows whose normalized title occurs more than once in the subset removed 112 PolitiFact rows (7 Fake, 105 Real) and 2,316 GossipCop rows (818 Fake, 1,498 Real). The fixed-family point ratio moved from 0.559 to 0.622 in PolitiFact (+0.106 log) and from 2.144 to 1.767 in GossipCop (−0.194 log). Directions are unchanged; the GossipCop magnitude is sensitive to repeated titles.

### Interpretation bounds

FakeNewsNet tweets were collected by Twitter title search over 2016–2018 collections, with no reply or retweet mapping and inherited labels of a different construct from both references. The PolitiFact interval is dominated by one shared-tweet component. These are observational dataset-context quantities under the frozen estimand; they do not identify a period or platform effect and alter no frozen estimate. They add two contexts in which the Fake/Real observed-post ratio is not below 1 with any confidence (PolitiFact) or is clearly above 1 (GossipCop), which widens the range of the ratio across datasets rather than supporting a single direction.

## Files written

`protocols/coaid_encoding_sensitivity.md`, `protocols/fakenewsnet_engagement_protocol.md`; `tables/coaid_encoding_prevalence.csv`, `tables/coaid_encoding_sensitivity.csv`, `tables/fakenewsnet_overlap_graph_summary.csv`, `tables/fakenewsnet_engagement_model_comparison.csv`, `tables/fakenewsnet_engagement_bootstrap_refits.csv`, `tables/fakenewsnet_engagement_results.csv`, `tables/fakenewsnet_engagement_contrast.csv`, `tables/fakenewsnet_duplicate_title_sensitivity.csv`, `tables/fakenewsnet_engagement_run_metadata.json`; `scripts/task_a_encoding_audit.py`, `scripts/task_b_fakenewsnet_engagement.py`.
