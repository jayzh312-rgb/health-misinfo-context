# Text quality and source-cue audit

Audit only; no classifier/vectorizer fitted and no publisher-held-out split created. Original cohort fields and split files are unchanged (SHA-256 verified). All rows remain in the new CSVs; use primary_modeling_eligible to select the frozen cohort.

| dataset | previously_eligible | modeling_ready | newly_artifact_excluded | high_risk_title_excluded | preexisting_exclusions |
| --- | --- | --- | --- | --- | --- |
| FakeHealth | 1648 | 1647 | 1 | 0 | 42 |
| CoAID | 3680 | 3577 | 103 | 0 | 1657 |

## Frozen quality decisions

Reviewed all 13 FakeHealth and 105 CoAID eligible mixed-artifact rows, using prepared text only. Specific article/claim headlines remain substantive even when bodies contain only branding; those bodies are ineligible for body-only sensitivity. Generic CDC section headings, error pages and social-profile names with boilerplate do not supply an article claim and are artifact-dominated. For FakeHealth branding bodies, >=25% of combined characters is major; shorter branding is minor. Two Video News rows contain substantive Disney business-news descriptions: retained as substantive, with possible topic/capture mismatch documented. No raw content was revisited.

| dataset | artifact_audit_class | N |
| --- | --- | --- |
| CoAID | ARTIFACT_DOMINATED | 103 |
| CoAID | SUBSTANTIVE_WITH_MAJOR_ARTIFACT | 2 |
| FakeHealth | SUBSTANTIVE_WITH_MAJOR_ARTIFACT | 3 |
| FakeHealth | SUBSTANTIVE_WITH_MINOR_ARTIFACT | 10 |

Reviewed all 86 eligible FakeHealth titles not marked fetched_content. All are marked review_original_title, not review_editorial_title. Of these, 85 are ordinary article headlines without explicit ratings/verdicts (low leakage risk). One is a review-site funding announcement with an empty body: moderate provenance risk, excluded as non-article artifact, not counted as high-risk ground-truth disclosure. Skeptical reporting of scientific evidence does not itself disclose the dataset label. CoAID title provenance remains ambiguous between fetched and original source-title fallback; this is documented as uncertain, not asserted to be review leakage. The audit does not prove the absence of hidden artifacts in unflagged rows.

## Source cues and masking

Counts are conservative direct-string detections, not source-identity accuracy. A cue may name a cited outlet/institution rather than the publishing outlet. Categories overlap. Attribution templates require a matched source identifier plus by/copyright/rights/attribution syntax; generic personal bylines alone do not count. URL matching recognizes explicit http(s)/www hosts; bare known domains match the metadata dictionary. Unknown bare domains, indirect stylistic cues, unnamed boilerplate and unlisted spelling variants can remain.

| dataset | scope | field | category | denominator_N | cue_N | percent | fake_N | real_N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FakeHealth | cohort_eligible | title_text | any_explicit_source_cue | 1648 | 2 | 0.121 | 0 | 2 |
| FakeHealth | modeling_ready | title_text | any_explicit_source_cue | 1647 | 2 | 0.121 | 0 | 2 |
| FakeHealth | cohort_eligible | body_text | any_explicit_source_cue | 1648 | 1079 | 65.473 | 287 | 792 |
| FakeHealth | modeling_ready | body_text | any_explicit_source_cue | 1647 | 1079 | 65.513 | 287 | 792 |
| FakeHealth | cohort_eligible | combined_text | any_explicit_source_cue | 1648 | 1079 | 65.473 | 287 | 792 |
| FakeHealth | modeling_ready | combined_text | any_explicit_source_cue | 1647 | 1079 | 65.513 | 287 | 792 |
| CoAID | cohort_eligible | title_text | any_explicit_source_cue | 3680 | 244 | 6.63 | 52 | 192 |
| CoAID | modeling_ready | title_text | any_explicit_source_cue | 3577 | 240 | 6.71 | 52 | 188 |
| CoAID | cohort_eligible | body_text | any_explicit_source_cue | 3680 | 962 | 26.141 | 183 | 779 |
| CoAID | modeling_ready | body_text | any_explicit_source_cue | 3577 | 872 | 24.378 | 183 | 689 |
| CoAID | cohort_eligible | combined_text | any_explicit_source_cue | 3680 | 1087 | 29.538 | 223 | 864 |
| CoAID | modeling_ready | combined_text | any_explicit_source_cue | 3577 | 994 | 27.789 | 223 | 771 |

Cue categories in final combined text (nonexclusive counts):

| dataset | scope | field | category | denominator_N | cue_N | percent | fake_N | real_N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CoAID | modeling_ready | combined_text | publisher_name | 3577 | 982 | 27.453 | 216 | 766 |
| CoAID | modeling_ready | combined_text | domain | 3577 | 20 | 0.559 | 7 | 13 |
| CoAID | modeling_ready | combined_text | explicit_url_host | 3577 | 18 | 0.503 | 4 | 14 |
| CoAID | modeling_ready | combined_text | source_attribution_or_copyright_template | 3577 | 16 | 0.447 | 0 | 16 |
| CoAID | modeling_ready | combined_text | institutional_boilerplate | 3577 | 0 | 0.0 | 0 | 0 |
| FakeHealth | modeling_ready | combined_text | publisher_name | 1647 | 939 | 57.013 | 247 | 692 |
| FakeHealth | modeling_ready | combined_text | source_attribution_or_copyright_template | 1647 | 510 | 30.965 | 151 | 359 |
| FakeHealth | modeling_ready | combined_text | domain | 1647 | 208 | 12.629 | 47 | 161 |
| FakeHealth | modeling_ready | combined_text | explicit_url_host | 1647 | 32 | 1.943 | 8 | 24 |
| FakeHealth | modeling_ready | combined_text | institutional_boilerplate | 1647 | 11 | 0.668 | 3 | 8 |

The dictionary uses only publisher_source, article_host and article_url from prepared cohorts, plus fixed identity aliases activated by metadata hosts. It never uses labels, ratings, engagement, review URLs, learned features or statistical class associations. Exact identifier matches use word boundaries; ambiguous TIME/STAT/WHO and Slate/Vox use case-sensitive forms. Overlapping spans are merged, then replaced once with [SOURCE]. Explicit URL hosts are masked syntactically, retaining paths and surrounding content. Identical text receives identical masking across datasets. This dictionary uses evaluation-set metadata, openly, as a prespecified transductive sensitivity transform; it is not an unseen-publisher production experiment. It removes explicit mentions, including legitimate references to institutions, but cannot eliminate publisher style, URL-path cues or collection confounding.

## Source/class association and later publisher evaluation

| dataset | scope | source_measure | fake_only_sources | real_only_sources | mixed_sources | exclusive_rows | mixed_rows | mixed_fake_N | mixed_real_N | missing_source_N | fake_support_sources | real_support_sources |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FakeHealth | cohort_eligible | publisher_source | 8 | 5 | 35 | 19 | 1613 | 442 | 1171 | 16 | 43 | 40 |
| FakeHealth | cohort_eligible | article_host | 12 | 31 | 47 | 73 | 1574 | 443 | 1131 | 1 | 59 | 78 |
| FakeHealth | modeling_ready | publisher_source | 8 | 5 | 35 | 19 | 1613 | 442 | 1171 | 15 | 43 | 40 |
| FakeHealth | modeling_ready | article_host | 12 | 31 | 47 | 73 | 1574 | 443 | 1131 | 0 | 59 | 78 |
| CoAID | cohort_eligible | publisher_source | 142 | 29 | 10 | 3027 | 610 | 566 | 44 | 43 | 152 | 39 |
| CoAID | cohort_eligible | article_host | 142 | 29 | 10 | 3027 | 610 | 566 | 44 | 43 | 152 | 39 |
| CoAID | modeling_ready | publisher_source | 139 | 27 | 10 | 2930 | 604 | 561 | 43 | 43 | 149 | 37 |
| CoAID | modeling_ready | article_host | 139 | 27 | 10 | 2930 | 604 | 561 | 43 | 43 | 149 | 37 |

Missing sources are counted separately, never treated as one publisher. Source identities here are the exact existing metadata categories; hosts/subdomains and publisher aliases are not silently consolidated. This can overstate distinct publishers, so a later publisher-held-out design needs an audited alias/host-family definition and joint resource-group barriers. Cramér’s V is not predictive performance. Class-exclusive sources are retained.

A publisher-held-out experiment is potentially feasible because both datasets have class support across multiple source categories. CoAID has very limited mixed-class-source support, so pure publisher effects remain difficult to separate from class. Restricting to mixed-class sources would lose all exclusive-source rows plus missing-source rows (counts above); such restriction is neither required for holding out publishers nor recommended for this baseline. Actual class-balanced publisher/resource partition feasibility is not established here; crossing leakage groups and alias consolidation may reduce available partitions. No alternative split was generated.

## Prespecified model inputs and validation

1. PRIMARY / natural: combined_text on primary_modeling_eligible == True. Preserve title + body exactly as prepared. Exclude prior ineligible rows, newly artifact-dominated rows, and high-risk title leakage. Retain low/moderate/uncertain provenance rows unless independently artifact-dominated; do not filter by source purity.
2. TITLE / BODY sensitivity: title_text on title_only_modeling_eligible, or body_text on body_only_modeling_eligible. These flags intersect primary eligibility with the existing field-specific eligibility. Sample sizes may differ; assignments do not. All usable fields preserve original language.
3. SOURCE-MASKED sensitivity: combined_text_source_masked on the exact primary cohort. Optional corresponding title/body masked fields exist. Only these named text fields are predictors; labels, ratings, IDs, provenance, cue/quality flags, engagement and assignments must never be concatenated into model text.

Use cv_fold >= 0 and cv_fold != f for fold-f training; cv_fold == f for validation. cv_fold == -1 is reserved holdout, and -2 denotes an originally ineligible row with no split. No split regeneration. Filtering preserved row/group assignments, train/holdout separation, and both classes in every training/validation/holdout partition for all conditions. Source masking can produce new identical strings across original groups; these reflect deliberate information removal and are not new resource identities. Existing group barriers remain unchanged.

| dataset | condition | partition | N | fake_N | real_N | leakage_groups | both_classes | split_integrity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FakeHealth | primary_natural | all | 1647 | 459 | 1188 | 1634 | True | True |
| FakeHealth | primary_natural | train_pool | 1317 | 367 | 950 | 1305 | True | True |
| FakeHealth | primary_natural | holdout | 330 | 92 | 238 | 329 | True | True |
| FakeHealth | primary_natural | cv_0_validation | 264 | 74 | 190 | 263 | True | True |
| FakeHealth | primary_natural | cv_0_training | 1053 | 293 | 760 | 1042 | True | True |
| FakeHealth | primary_natural | cv_1_validation | 262 | 72 | 190 | 261 | True | True |
| FakeHealth | primary_natural | cv_1_training | 1055 | 295 | 760 | 1044 | True | True |
| FakeHealth | primary_natural | cv_2_validation | 264 | 74 | 190 | 254 | True | True |
| FakeHealth | primary_natural | cv_2_training | 1053 | 293 | 760 | 1051 | True | True |
| FakeHealth | primary_natural | cv_3_validation | 264 | 74 | 190 | 264 | True | True |
| FakeHealth | primary_natural | cv_3_training | 1053 | 293 | 760 | 1041 | True | True |
| FakeHealth | primary_natural | cv_4_validation | 263 | 73 | 190 | 263 | True | True |
| FakeHealth | primary_natural | cv_4_training | 1054 | 294 | 760 | 1042 | True | True |
| CoAID | primary_natural | all | 3577 | 825 | 2752 | 2828 | True | True |
| CoAID | primary_natural | train_pool | 2875 | 659 | 2216 | 2163 | True | True |
| CoAID | primary_natural | holdout | 702 | 166 | 536 | 665 | True | True |
| CoAID | primary_natural | cv_0_validation | 732 | 133 | 599 | 126 | True | True |
| CoAID | primary_natural | cv_0_training | 2143 | 526 | 1617 | 2037 | True | True |
| CoAID | primary_natural | cv_1_validation | 538 | 134 | 404 | 509 | True | True |
| CoAID | primary_natural | cv_1_training | 2337 | 525 | 1812 | 1654 | True | True |
| CoAID | primary_natural | cv_2_validation | 533 | 129 | 404 | 513 | True | True |
| CoAID | primary_natural | cv_2_training | 2342 | 530 | 1812 | 1650 | True | True |
| CoAID | primary_natural | cv_3_validation | 536 | 131 | 405 | 513 | True | True |
| CoAID | primary_natural | cv_3_training | 2339 | 528 | 1811 | 1650 | True | True |
| CoAID | primary_natural | cv_4_validation | 536 | 132 | 404 | 502 | True | True |
| CoAID | primary_natural | cv_4_training | 2339 | 527 | 1812 | 1661 | True | True |

All condition-specific partition counts are in text_modeling_ready_summary.csv. Reproduce with `.venv/bin/python src/audit_text_quality_and_source_cues.py`; validate with `.venv/bin/python -m unittest discover -s src -p "test_text_quality_and_source_cues.py"`. Input fingerprints and dictionary scope are recorded in the adjacent metadata JSON. The first TF-IDF + Logistic Regression baseline is ready under this frozen protocol, with training-fold-only fitting and resource-generalization interpretation. No training is authorized by this audit.

Outputs: data/processed/{fakehealth,coaid}_text_modeling_ready.csv; outputs/tables/text_quality_audit.csv, text_source_cue_audit.csv, text_modeling_ready_summary.csv, text_source_mask_dictionary.csv, text_source_confounding_audit.csv, text_modeling_exclusion_summary.csv.
