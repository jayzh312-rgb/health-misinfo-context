# Supplement v3 build report — 2026-10-12

Built `outputs/manuscript/revision_v3/supplementary_material_v3.md` (2869 lines; SHA-256 `cb150c329e79f41b1c489f84937b5498c1d52a868edd251df9a736f623f60469`) from `review_package_v2/supplementary_material.md` with `revision_v3/build_supplement_v3.py`. Nothing under `review_package_v2/`, `outputs/tables/` or `outputs/extension_v1/` was modified; no analysis ran. Sections S1–S7 and Tables S3–S10 are the v2 text verbatim with migrated prose appended at the end of S1, S2, S3, S4, S6 and S7; S8 and Tables S11–S17 are new.

## Task 1 — Migration of the 33 missing protocol facts

All 33 facts from the "Facts appearing nowhere" table of `v3_verification.md`, plus the two engagement sample sizes (facts 2.2d, 2.2f) and the C = R_CoAID / R_FakeHealth definition (2.5k, part of the 33), were inserted as prose in the existing voice with the exact v2 values. Log: `supplement_migration_log.csv` (35 rows, all `migrated`, each with the v3 supplement line of its anchor phrase).

| Section | Facts migrated | New supplement lines |
|---|---|---|
| S1 | 2.3f | [17] |
| S2 | 2.2d, 2.2f, 2.4d, 2.4e, 2.4i, 2.5k | [27, 29] |
| S3 | 2.6a, 2.6c, 2.6e, 2.6m, 2.6n, 2.6p, 2.8h | [41] |
| S4 | 2.7d, 2.7p, 2.8d, 2.8f, 2.9e | [76] |
| S6 | transformer recipe as one paragraph (2.10a, 2.10b, 2.10e, 2.10f, 2.10i, 2.10l, 2.10o, 2.10p, 2.10t) and a COVID-Rumor paragraph (2.11a, 2.11c, 2.11e, 2.11f) | [96, 98] |
| S7 | Supplementary Table S7a: exact software versions, checkpoint revision, seed policy (2.12a, 2.12b, 2.12c) | [110, 116, 120] |

## Task 2 — Extension outputs registered as supplementary tables

Each new table has a caption, a per-file provenance line naming the source CSV path and its SHA-256, and the same panel layout and four-decimal display rule as Tables S3–S10.

| Table | Content | Source CSV | Supplement line |
|---|---|---|---:|
| S11 | CoAID A hurdle components (occurrence, conditional mean, product check) | `outputs/extension_v1/tables/coaid_hurdle_components.csv` | 2234 |
| S12 | Top 25 Fake- and Real-leaning unigrams per dataset, with shared-vocabulary, other-model-sign and mask columns (4 panels) | `outputs/extension_v1/tables/top_fake_real_terms.csv` | 2265 |
| S13 | FakeNewsNet title-only, LR-only external cohort (S13.1) and transfer results by scope (S13.2) | `fakenewsnet_external_cohort_summary.csv`, `fakenewsnet_external_transfer_results.csv` | 2417 |
| S14 | Source–class association, Cramér's V, both row scopes | `outputs/tables/text_source_label_association.csv` | 2489 |
| S15 | Engagement comparability support, aggregate rows only (8 panels; 654 native tag/category rows omitted) | `outputs/tables/engagement_comparability_support.csv` | 2516 |
| S16 | CoAID encoding-artifact prevalence and locked-protocol repair sensitivity (round 2) | `coaid_encoding_sensitivity.csv`, `coaid_encoding_prevalence.csv` | 2679 |
| S17 | FakeNewsNet engagement under the frozen estimand: graph, model comparison, results, contrast, duplicate-title sensitivity (round 2) | five `fakenewsnet_engagement_*.csv` / `fakenewsnet_overlap_graph_summary.csv` / `fakenewsnet_duplicate_title_sensitivity.csv` | 2745 |

Round-2 tables were included: `outputs/extension_v1/reports/extension_v1_round2.md` exists, so S16 and S17 are populated rather than placeholders. A new S8 section (line 125) describes the three extension protocols and round 2 and states that they do not alter frozen results.

## Task 3 — Consistency

### Placeholder mapping in `manuscript_v3_draft.md`

| v3 line | Placeholder | Concrete reference |
|---:|---|---|
| 153 (3.1 hurdle components, no bracket but numbers from extension) | occurrence ratio and conditional ratio | **Table S11** |
| 175 | "Supplementary Table [Sx]" for the most heavily weighted words | **Table S12** |
| 191 | "Supplementary [Sx]" for the FakeNewsNet title-only transfer | **Table S13** |
| 220 | Cramér's V ≈ 0.97 vs ≈ 0.35, "Supplementary Methods S3 — confirm" | **Table S14** (values 0.969/0.353 all rows; 0.971/0.352 primary-eligible); S3 prose unchanged |
| 256 | "Pending quantification" of encoding-error words | **Table S12** (the two Mac-Roman tokens rank 4 and 20 among CoAID Fake-leaning terms) and **Table S16** (25.9% of CoAID Fake documents vs 0.0% of Real; repaired-text holdout macro F1 0.9281) |
| 262 | Feasibility wording and support-table placement | **Table S15** |
| 155 (conditional FakeNewsNet engagement extension) | now exists | **Table S17** |

### Check 1 re-run against the new supplement

The 160-fact search from `v3_verification.md` was re-run with the v3 Methods side held at its adjudicated status and the supplement side evaluated on `supplementary_material_v3.md` (`check1_recheck_v3_supplement.csv`).

| Status | Count |
|---|---:|
| Both v3 Methods and supplement | 77 |
| Supplement only | 73 |
| v3 Methods only | 10 |
| **Nowhere** | **0** |

Target met: 0 facts appear nowhere. The 10 "v3 Methods only" facts are ones the v3 draft keeps in the main text (release directories, label thresholds, AIC rule, exposure-offset statement, Fake = 1 coding, calibration note, source-only vocabulary, group separation, 49-component count, transformer fold reuse); none needs to move.

## Follow-ups for the author

- Replace the bracketed `[Sx]` references at v3 lines 175 and 191 with S12 and S13, and point lines 153, 220, 256, 262 to S11, S14, S12/S16 and S15.
- The S7a table and S6 paragraph carry the exact versions and checkpoint hash; v3 Methods 2.11 can keep its generic statement.
- If Juul and Ugander (2021) or the FakeNewsNet citation are added, the supplement needs no change.
