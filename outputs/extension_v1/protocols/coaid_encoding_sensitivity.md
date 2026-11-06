# Protocol: CoAID encoding-artifact audit and text-repair sensitivity (extension_v1, round 2)

Written 2026-10-12 before any label- or outcome-dependent computation for this task. Frozen baseline untouched; outputs only under `outputs/extension_v1/`.

## A1. Prevalence audit (descriptive, label-blind patterns applied to all rows, then tabulated by class)

Population: the frozen `primary_modeling_eligible` combined-text cohorts loaded through `run_tfidf_logistic_baseline.load_cohort` (3,577 CoAID, 1,647 FakeHealth; this loader re-validates the saved split files). Text field: `combined_text` only.

Pattern families, each a fixed regex evaluated on the raw stored string:
- `cp1252_utf8`: UTF-8 bytes decoded as Windows-1252/Latin-1, i.e. the sequences `â€` (curly quotes, dashes, ellipsis), `Ã` followed by any character (accented letters), and a stray `Â`.
- `macroman_quote`: `õ`/`Õ` between word characters (apostrophe substitute, e.g. donÕt), `ò`/`Ò` at the start of a word (opening quote substitute), `ó`/`Ó` at the end of a word (closing quote substitute). Upper- and lower-case forms are both matched because the stored text is case-preserved and the earlier model tokens were lowercased.
- `c1_control`: any character in U+0080–U+009F.
- `replacement_char`: U+FFFD.
- `mojibake_any`: any of the above.
Content-cue patterns: `year_token` = `\b(?:1[89]|20)\d{2}\b`; `month_name` = the twelve English month names, word-bounded, case-insensitive (reported with and without "may", which is also a modal verb); `credit_fragment` = `getty`, `images`, `reuters`, `ap photo`, `medscape`, word-bounded, case-insensitive.
Tabulation: share of documents containing at least one match, by dataset × class, and for CoAID by `publisher_family` from the frozen `outputs/tables/publisher_family_map.csv` (families with fewer than 20 eligible rows are pooled as "other"; rows without a resolved family are reported as "unresolved/missing"). Shares only; no example strings are written to outputs.

## A2. Sensitivity refit under the locked protocol

- Configuration: the locked CoAID natural-text LR configuration from `outputs/models/tfidf_logistic/selection_lock.json` (cfg_05: unigram, min_df 2, C 10, balanced) built with the unchanged `run_tfidf_logistic_baseline.make_pipeline` (lowercase, no accent stripping, sublinear TF, smoothed IDF, L2, float64, liblinear, tol 1e-4, max_iter 5,000, seed 42). No reselection, no grid, no tie rule is run.
- Split: the saved training pool (`holdout_assignment == 'train'`, 2,875 rows) and reserved holdout (`'test'`, 702 rows) exactly as validated by `load_cohort`. No row moves.
- The only change is a deterministic text transformation applied identically to training and holdout text before vectorization:
  1. `repaired`: fixed mojibake-repair mapping followed by Unicode NFKC. **ftfy is not installed in `.venv`** (checked 2026-10-12: `ModuleNotFoundError`), so the mapping is hand-written and recorded in `scripts/task_a_encoding_audit.py`: the Windows-1252 sequences `â€™ â€˜ â€œ â€ â€“ â€” â€¦ â€¢` and the `Ã`-pairs for é è á à ó í ú ñ ü ö ä ç É are mapped to their intended characters; stray `Â` is removed; the three Mac-Roman quote substitutes are mapped to `'` and `"`; C1 controls and U+FFFD are removed; then NFKC.
  2. `repaired_strict`: `repaired` plus replacement of year tokens, the twelve month names and the credit fragments (same regexes as A1) by a single space.
- Reference: the frozen holdout metrics macro F1 0.9260, ROC-AUC 0.9827, AP 0.9280 from `outputs/tables/tfidf_logistic_holdout_results.csv`. A `natural_refit_check` row refits the unchanged text with the same pipeline to confirm the frozen numbers reproduce before any repaired variant is read.
- Metrics: `run_tfidf_logistic_baseline.metrics` (threshold 0.5, Fake = 1). Descriptive; no significance test.
- Outputs: `tables/coaid_encoding_prevalence.csv`, `tables/coaid_encoding_sensitivity.csv`.

Limitations fixed in advance: the Mac-Roman rules can alter genuine Portuguese/Spanish words containing õ/ò/ó; the `Ã` rule can match genuine "Ã"; removing "may" removes a modal verb. Counts of documents changed by each rule are reported so these costs are visible. The sensitivity shows whether the within-CoAID holdout separability depends on encoding artifacts and date/credit tokens; it does not re-estimate transfer and is not a replacement primary result.
