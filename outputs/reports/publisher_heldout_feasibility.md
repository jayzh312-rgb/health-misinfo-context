# Publisher/source-held-out feasibility audit

This audit uses only the primary modeling-ready cohort and prepared metadata. No classifier is fitted, and no final publisher assignments or modified cohorts/splits are written. Existing content/resource generalization and unseen-publisher generalization are different questions.

## Feasibility conclusions and recommended next design

CoAID: 216 identified source families (including 53 account proxies), 8 mixed families and 32 mixed-family rows. StratifiedGroupKFold supports 5, 4, 3 folds by the >=20-per-class evaluation diagnostic. The mixed-family restriction retains 32/3577 rows (19 Fake, 13 Real), discarding 3545 (99.11%).

CoAID fixed holdout is numerically feasible: 599 test rows (54 Fake, 545 Real), 20.01% of the identified constraint-safe cohort. Concentration/source-class risk flag: True.

FakeHealth: 49 identified source families (including 0 account proxies), 34 mixed families and 1624 mixed-family rows. StratifiedGroupKFold supports 5, 4, 3 folds by the >=20-per-class evaluation diagnostic. The mixed-family restriction retains 1624/1647 rows (449 Fake, 1175 Real), discarding 23 (1.40%).

FakeHealth fixed holdout is numerically feasible: 330 test rows (92 Fake, 238 Real), 20.04% of the identified constraint-safe cohort. Concentration/source-class risk flag: True.

FakeHealth is substantially more scientifically useful for this sensitivity: 98.60% of rows are in mixed families, and its full cohort is retained. Recommend a three-fold StratifiedGroupKFold over publisher-resource components, sorted content_uid input, shuffle=True, random_state=42, on all primary eligible FakeHealth rows. Five folds are class-supported, but three folds reduce single-publisher class concentration in these count diagnostics. This is a proposed external evaluation structure; any later tuning must use only its training partitions with the same publisher/resource barriers. Do not execute or save those splits without later authorization.

CoAID supports numerical stratification on only 2994/3577 identifiable rows, after excluding 583 unresolved rows (554 Fake and 29 Real). This removes 67.15% of original Fake support. 98.93% of identified rows remain in class-exclusive families; the three largest Real-support families supply 73.34% of identified Real rows. Ordinary GroupKFold produces class-absent validation partitions for every tested fold count. StratifiedGroupKFold retains both classes but every evaluation fold is flagged for source-class risk. Thus CoAID would be an explicitly restricted source-shift/dataset-construction stress test, not a clean test of general misinformation language, and is not recommended as the next publisher experiment.

CoAID mixed-only evaluation is too sparse: no family reaches even 5 Fake and 5 Real, so the 5/5, 10/10 and 20/20 restrictions are empty. It would not represent the original population. The apparent mixed-source support in the earlier raw-host audit partly pooled different platform users; resolving that distinction strengthens rather than removes the source/class confounding concern. FakeHealth remains broadly mixed-label after normalization.

Resource constraints are compatible through connected components: the three CoAID multi-family resource groups (six rows) are co-assigned; FakeHealth has none. These few links do not materially prevent candidate splitting. The overlap table identifies shared observed source families across datasets; overlap is relevant for later transfer interpretation but does not establish identical articles or eliminate domain shift.

## Conservative identity rules

Publisher/source name=`publisher_source`; host=`article_host`; fallback URL=`article_url`; resource barrier=`leakage_group`. Name normalization uses Unicode NFKC, casefold and whitespace only. Hosts use lowercase and removal of www; subdomains collapse only to an observed parent or a previously audited identity host. Audited identities take precedence over broader container hosts. Hosted tenants remain separate. Existing explicit aliases come from the unchanged source-audit ALIASES constant. Unique observed name-host links and exact names connect aliases. Where multiple attribution names appear on one host, the exact publication host defines the source family and those names do not bridge other hosts. Conflicting audited identity hosts are not merged. There is no corporate-ownership, political-orientation, article-text or label-derived merging. Website/account families are operational proxies, not a fully resolved corporate publisher ontology; ambiguous cross-host aliases may remain separate.

Platform domains are not publisher identities. Only explicit Twitter/X account-status URLs and YouTube account/channel URLs supply account proxies; opaque video/post links and other unresolved platform routes remain unidentified. Accounts are kept separate, with no inferred account-to-organization ownership. Missing metadata and unresolved platform authors are separate categories, not one synthetic publisher. No labels enter normalization. This is more conservative than treating every article host as a publisher; the row-level map preserves all evidence and unresolved cases.

## Family and class support

| dataset | cohort_N | publisher_families | fake_only_families | real_only_families | mixed_families | fake_only_rows | real_only_rows | mixed_rows | mixed_fake_N | mixed_real_N | missing_source_rows | unresolved_platform_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CoAID | 3577 | 216 | 181 | 27 | 8 | 252 | 2710 | 32 | 19 | 13 | 43 | 540 |
| FakeHealth | 1647 | 49 | 7 | 8 | 34 | 10 | 13 | 1624 | 449 | 1175 | 0 | 0 |

| dataset | median_rows_per_publisher | p90_rows_per_publisher | largest_publisher_N | largest_publisher_fake_N | largest_publisher_real_N | fake_support_families | real_support_families | fake_largest_family_share | real_largest_family_share | fake_top3_family_share | real_top3_family_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CoAID | 1.000 | 3.000 | 809 | 0 | 809 | 189 | 35 | 0.077 | 0.297 | 0.125 | 0.733 |
| FakeHealth | 12.000 | 98.800 | 257 | 71 | 186 | 41 | 42 | 0.155 | 0.157 | 0.412 | 0.363 |

## Earlier source-confounding comparison

The existing source-confounding audit counts raw names and hosts, which are not necessarily publisher identities. Family/account normalization and explicit platform uncertainty can change counts and the identifiable population. Compare class support without interpreting hosting-platform aggregates as organizations.

| dataset | scope | source_measure | fake_only_sources | real_only_sources | mixed_sources | exclusive_rows | mixed_rows | mixed_fake_N | mixed_real_N | missing_source_N | fake_support_sources | real_support_sources |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FakeHealth | modeling_ready | publisher_source | 8 | 5 | 35 | 19 | 1613 | 442 | 1171 | 15 | 43 | 40 |
| FakeHealth | modeling_ready | article_host | 12 | 31 | 47 | 73 | 1574 | 443 | 1131 | 0 | 59 | 78 |
| CoAID | modeling_ready | publisher_source | 139 | 27 | 10 | 2930 | 604 | 561 | 43 | 43 | 149 | 37 |
| CoAID | modeling_ready | article_host | 139 | 27 | 10 | 2930 | 604 | 561 | 43 | 43 | 149 | 37 |

## Resource constraints

Publisher-family and resource-group separation are enforced jointly using connected components of the publisher↔resource graph. Multiple publishers in one resource group must travel together; that is a constraint, not permission to split the resource. Any component connected to an unidentified source is excluded from the strict prospective design, so neither barrier is fragmented. Mixed-family restrictions are also reported both directly and after retaining only intact original components.

| dataset | identified_rows | strict_eligible_N | resource_connected_exclusion_N | ambiguous_host_name_rows |
| --- | --- | --- | --- | --- |
| CoAID | 2994 | 2994 | 0 | 11 |
| FakeHealth | 1647 | 1647 | 0 | 807 |

Conflict records: 3; affected rows: 6. The conflicts CSV separates multiple known families from known-plus-unresolved groups.

## Candidate algorithms and feasibility

A: GroupKFold without shuffling and StratifiedGroupKFold(shuffle=True, random_state=42), evaluated at 5, 4 and 3 folds. Input rows sort by content_uid; groups are publisher-resource connected components. With no resource conflicts, this is publisher-family grouping. All candidates use the full identified, constraint-safe cohort; these are prospective alternate splits, not modifications to existing resource splits.

B: deterministic approximately 20% holdout. Aggregate sorted connected components into total/Fake/Real counts; examine prefixes of the sorted order plus 2,048 permutations from NumPy default_rng(42). Reject candidates missing either class in train or test. Prefer candidates with >=20 of each class in both partitions; then minimize the mean squared relative departure of held-out total/Fake/Real fractions from 0.20. Break ties lexicographically by sorted component IDs. This is a finite count-only search, not proof of global optimality. A 15–25% held-out fraction is a descriptive approximation check.

C: mixed-publisher-only sensitivity; identical A algorithms/counts on intact components wholly contained within mixed families. This changes the target population. No threshold is chosen using future model performance.

The detailed feasibility CSV reports both train and evaluation counts, class proportions, family/component support and minimum evaluation counts. Here are evaluation partitions:

| dataset | design | algorithm | folds | fold | partition | N | fake_N | real_N | publisher_families | both_labels | low_class_support | source_class_proxy_risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CoAID | A | GroupKFold | 5 | 0 | validation | 809 | 0 | 809 | 1 | False | True | True |
| CoAID | A | GroupKFold | 5 | 1 | validation | 626 | 0 | 626 | 1 | False | True | True |
| CoAID | A | GroupKFold | 5 | 2 | validation | 562 | 0 | 562 | 1 | False | True | True |
| CoAID | A | GroupKFold | 5 | 3 | validation | 499 | 37 | 462 | 41 | True | False | True |
| CoAID | A | GroupKFold | 5 | 4 | validation | 498 | 234 | 264 | 172 | True | False | True |
| CoAID | A | GroupKFold | 4 | 0 | validation | 809 | 0 | 809 | 1 | False | True | True |
| CoAID | A | GroupKFold | 4 | 1 | validation | 729 | 92 | 637 | 72 | True | False | True |
| CoAID | A | GroupKFold | 4 | 2 | validation | 728 | 81 | 647 | 70 | True | False | True |
| CoAID | A | GroupKFold | 4 | 3 | validation | 728 | 98 | 630 | 73 | True | False | True |
| CoAID | A | GroupKFold | 3 | 0 | validation | 987 | 163 | 824 | 110 | True | False | True |
| CoAID | A | GroupKFold | 3 | 1 | validation | 986 | 108 | 878 | 104 | True | False | True |
| CoAID | A | GroupKFold | 3 | 2 | validation | 1021 | 0 | 1021 | 2 | False | True | True |
| CoAID | A | StratifiedGroupKFold | 5 | 0 | validation | 865 | 54 | 811 | 40 | True | False | True |
| CoAID | A | StratifiedGroupKFold | 5 | 1 | validation | 685 | 54 | 631 | 40 | True | False | True |
| CoAID | A | StratifiedGroupKFold | 5 | 2 | validation | 622 | 55 | 567 | 41 | True | False | True |
| CoAID | A | StratifiedGroupKFold | 5 | 3 | validation | 308 | 54 | 254 | 56 | True | False | True |
| CoAID | A | StratifiedGroupKFold | 5 | 4 | validation | 514 | 54 | 460 | 39 | True | False | True |
| CoAID | A | StratifiedGroupKFold | 4 | 0 | validation | 880 | 68 | 812 | 49 | True | False | True |
| CoAID | A | StratifiedGroupKFold | 4 | 1 | validation | 704 | 68 | 636 | 52 | True | False | True |
| CoAID | A | StratifiedGroupKFold | 4 | 2 | validation | 706 | 67 | 639 | 50 | True | False | True |
| CoAID | A | StratifiedGroupKFold | 4 | 3 | validation | 704 | 68 | 636 | 65 | True | False | True |
| CoAID | A | StratifiedGroupKFold | 3 | 0 | validation | 924 | 91 | 833 | 83 | True | False | True |
| CoAID | A | StratifiedGroupKFold | 3 | 1 | validation | 951 | 90 | 861 | 67 | True | False | True |
| CoAID | A | StratifiedGroupKFold | 3 | 2 | validation | 1119 | 90 | 1029 | 66 | True | False | True |
| CoAID | C | GroupKFold | 5 | 0 | validation | 8 | 6 | 2 | 1 | True | True | True |
| CoAID | C | GroupKFold | 5 | 1 | validation | 7 | 3 | 4 | 1 | True | True | True |
| CoAID | C | GroupKFold | 5 | 2 | validation | 6 | 3 | 3 | 2 | True | True | True |
| CoAID | C | GroupKFold | 5 | 3 | validation | 6 | 4 | 2 | 2 | True | True | True |
| CoAID | C | GroupKFold | 5 | 4 | validation | 5 | 3 | 2 | 2 | True | True | True |
| CoAID | C | GroupKFold | 4 | 0 | validation | 8 | 6 | 2 | 1 | True | True | True |
| CoAID | C | GroupKFold | 4 | 1 | validation | 9 | 4 | 5 | 2 | True | True | True |
| CoAID | C | GroupKFold | 4 | 2 | validation | 7 | 4 | 3 | 2 | True | True | True |
| CoAID | C | GroupKFold | 4 | 3 | validation | 8 | 5 | 3 | 3 | True | True | False |
| CoAID | C | GroupKFold | 3 | 0 | validation | 12 | 8 | 4 | 3 | True | True | True |
| CoAID | C | GroupKFold | 3 | 1 | validation | 10 | 5 | 5 | 2 | True | True | True |
| CoAID | C | GroupKFold | 3 | 2 | validation | 10 | 6 | 4 | 3 | True | True | True |
| CoAID | C | StratifiedGroupKFold | 5 | 0 | validation | 5 | 3 | 2 | 2 | True | True | True |
| CoAID | C | StratifiedGroupKFold | 5 | 1 | validation | 7 | 3 | 4 | 1 | True | True | True |
| CoAID | C | StratifiedGroupKFold | 5 | 2 | validation | 8 | 6 | 2 | 1 | True | True | True |
| CoAID | C | StratifiedGroupKFold | 5 | 3 | validation | 5 | 3 | 2 | 2 | True | True | True |
| CoAID | C | StratifiedGroupKFold | 5 | 4 | validation | 7 | 4 | 3 | 2 | True | True | True |
| CoAID | C | StratifiedGroupKFold | 4 | 0 | validation | 8 | 6 | 2 | 1 | True | True | True |
| CoAID | C | StratifiedGroupKFold | 4 | 1 | validation | 10 | 6 | 4 | 3 | True | True | True |
| CoAID | C | StratifiedGroupKFold | 4 | 2 | validation | 7 | 3 | 4 | 1 | True | True | True |
| CoAID | C | StratifiedGroupKFold | 4 | 3 | validation | 7 | 4 | 3 | 3 | True | True | True |
| CoAID | C | StratifiedGroupKFold | 3 | 0 | validation | 10 | 7 | 3 | 2 | True | True | True |
| CoAID | C | StratifiedGroupKFold | 3 | 1 | validation | 11 | 7 | 4 | 4 | True | True | False |
| CoAID | C | StratifiedGroupKFold | 3 | 2 | validation | 11 | 5 | 6 | 2 | True | True | True |
| CoAID | B | count_only_seeded_component_search | 1 | 0 | test | 599 | 54 | 545 | 47 | True | False | True |
| FakeHealth | A | GroupKFold | 5 | 0 | validation | 330 | 80 | 250 | 9 | True | False | True |
| FakeHealth | A | GroupKFold | 5 | 1 | validation | 330 | 116 | 214 | 10 | True | False | True |
| FakeHealth | A | GroupKFold | 5 | 2 | validation | 329 | 106 | 223 | 9 | True | False | False |
| FakeHealth | A | GroupKFold | 5 | 3 | validation | 329 | 76 | 253 | 10 | True | False | False |
| FakeHealth | A | GroupKFold | 5 | 4 | validation | 329 | 81 | 248 | 11 | True | False | False |
| FakeHealth | A | GroupKFold | 4 | 0 | validation | 412 | 119 | 293 | 12 | True | False | True |
| FakeHealth | A | GroupKFold | 4 | 1 | validation | 412 | 133 | 279 | 12 | True | False | False |
| FakeHealth | A | GroupKFold | 4 | 2 | validation | 412 | 124 | 288 | 13 | True | False | False |
| FakeHealth | A | GroupKFold | 4 | 3 | validation | 411 | 83 | 328 | 12 | True | False | False |
| FakeHealth | A | GroupKFold | 3 | 0 | validation | 549 | 132 | 417 | 15 | True | False | True |
| FakeHealth | A | GroupKFold | 3 | 1 | validation | 549 | 175 | 374 | 17 | True | False | False |
| FakeHealth | A | GroupKFold | 3 | 2 | validation | 549 | 152 | 397 | 17 | True | False | False |
| FakeHealth | A | StratifiedGroupKFold | 5 | 0 | validation | 332 | 101 | 231 | 7 | True | False | True |
| FakeHealth | A | StratifiedGroupKFold | 5 | 1 | validation | 324 | 92 | 232 | 14 | True | False | True |
| FakeHealth | A | StratifiedGroupKFold | 5 | 2 | validation | 336 | 85 | 251 | 12 | True | False | False |
| FakeHealth | A | StratifiedGroupKFold | 5 | 3 | validation | 327 | 93 | 234 | 6 | True | False | True |
| FakeHealth | A | StratifiedGroupKFold | 5 | 4 | validation | 328 | 88 | 240 | 10 | True | False | False |
| FakeHealth | A | StratifiedGroupKFold | 4 | 0 | validation | 420 | 120 | 300 | 8 | True | False | True |
| FakeHealth | A | StratifiedGroupKFold | 4 | 1 | validation | 409 | 113 | 296 | 15 | True | False | False |
| FakeHealth | A | StratifiedGroupKFold | 4 | 2 | validation | 410 | 113 | 297 | 15 | True | False | False |
| FakeHealth | A | StratifiedGroupKFold | 4 | 3 | validation | 408 | 113 | 295 | 11 | True | False | True |
| FakeHealth | A | StratifiedGroupKFold | 3 | 0 | validation | 549 | 153 | 396 | 15 | True | False | False |
| FakeHealth | A | StratifiedGroupKFold | 3 | 1 | validation | 549 | 152 | 397 | 18 | True | False | False |
| FakeHealth | A | StratifiedGroupKFold | 3 | 2 | validation | 549 | 154 | 395 | 16 | True | False | False |
| FakeHealth | C | GroupKFold | 5 | 0 | validation | 327 | 79 | 248 | 6 | True | False | True |
| FakeHealth | C | GroupKFold | 5 | 1 | validation | 326 | 117 | 209 | 7 | True | False | True |
| FakeHealth | C | GroupKFold | 5 | 2 | validation | 325 | 105 | 220 | 7 | True | False | False |
| FakeHealth | C | GroupKFold | 5 | 3 | validation | 323 | 70 | 253 | 7 | True | False | False |
| FakeHealth | C | GroupKFold | 5 | 4 | validation | 323 | 78 | 245 | 7 | True | False | False |
| FakeHealth | C | GroupKFold | 4 | 0 | validation | 406 | 114 | 292 | 8 | True | False | True |
| FakeHealth | C | GroupKFold | 4 | 1 | validation | 407 | 132 | 275 | 9 | True | False | True |
| FakeHealth | C | GroupKFold | 4 | 2 | validation | 405 | 123 | 282 | 9 | True | False | False |
| FakeHealth | C | GroupKFold | 4 | 3 | validation | 406 | 80 | 326 | 8 | True | False | False |
| FakeHealth | C | GroupKFold | 3 | 0 | validation | 542 | 130 | 412 | 11 | True | False | True |
| FakeHealth | C | GroupKFold | 3 | 1 | validation | 542 | 172 | 370 | 12 | True | False | False |
| FakeHealth | C | GroupKFold | 3 | 2 | validation | 540 | 147 | 393 | 11 | True | False | False |
| FakeHealth | C | StratifiedGroupKFold | 5 | 0 | validation | 328 | 99 | 229 | 4 | True | False | True |
| FakeHealth | C | StratifiedGroupKFold | 5 | 1 | validation | 321 | 94 | 227 | 4 | True | False | True |
| FakeHealth | C | StratifiedGroupKFold | 5 | 2 | validation | 340 | 85 | 255 | 8 | True | False | False |
| FakeHealth | C | StratifiedGroupKFold | 5 | 3 | validation | 313 | 87 | 226 | 10 | True | False | True |
| FakeHealth | C | StratifiedGroupKFold | 5 | 4 | validation | 322 | 84 | 238 | 8 | True | False | False |
| FakeHealth | C | StratifiedGroupKFold | 4 | 0 | validation | 402 | 110 | 292 | 10 | True | False | True |
| FakeHealth | C | StratifiedGroupKFold | 4 | 1 | validation | 403 | 108 | 295 | 9 | True | False | False |
| FakeHealth | C | StratifiedGroupKFold | 4 | 2 | validation | 416 | 119 | 297 | 8 | True | False | False |
| FakeHealth | C | StratifiedGroupKFold | 4 | 3 | validation | 403 | 112 | 291 | 7 | True | False | True |
| FakeHealth | C | StratifiedGroupKFold | 3 | 0 | validation | 535 | 148 | 387 | 11 | True | False | False |
| FakeHealth | C | StratifiedGroupKFold | 3 | 1 | validation | 544 | 153 | 391 | 9 | True | False | False |
| FakeHealth | C | StratifiedGroupKFold | 3 | 2 | validation | 545 | 148 | 397 | 14 | True | False | False |
| FakeHealth | B | count_only_seeded_component_search | 1 | 0 | test | 330 | 92 | 238 | 8 | True | False | True |

Both classes make ROC-AUC and two-class AP meaningful to compute. AP may numerically return a degenerate value with one class, but is flagged uninterpretable here. Macro F1 is computable with fixed class labels, but we require >=20 of each evaluation class for the descriptive support flag. These are feasibility diagnostics, not precision/power guarantees. Source-class proxy risk is flagged when >=80% of evaluation rows belong to class-exclusive families, or one publisher supplies >=50% of either evaluation class. Concentration and class-exclusive fractions must accompany any metric.

## Prespecified restrictions and population changes

| dataset | restriction | scope | N | fake_N | real_N | publisher_families | lost_N | retained_percent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CoAID | all_identified | direct_restriction | 2994 | 271 | 2723 | 216 | 583 | 83.701 |
| CoAID | all_identified | intact_publisher_resource_components | 2994 | 271 | 2723 | 216 | 583 | 83.701 |
| CoAID | mixed_publishers | direct_restriction | 32 | 19 | 13 | 8 | 3545 | 0.895 |
| CoAID | mixed_publishers | intact_publisher_resource_components | 32 | 19 | 13 | 8 | 3545 | 0.895 |
| CoAID | min5_each | direct_restriction | 0 | 0 | 0 | 0 | 3577 | 0.000 |
| CoAID | min5_each | intact_publisher_resource_components | 0 | 0 | 0 | 0 | 3577 | 0.000 |
| CoAID | min10_each | direct_restriction | 0 | 0 | 0 | 0 | 3577 | 0.000 |
| CoAID | min10_each | intact_publisher_resource_components | 0 | 0 | 0 | 0 | 3577 | 0.000 |
| CoAID | min20_each | direct_restriction | 0 | 0 | 0 | 0 | 3577 | 0.000 |
| CoAID | min20_each | intact_publisher_resource_components | 0 | 0 | 0 | 0 | 3577 | 0.000 |
| FakeHealth | all_identified | direct_restriction | 1647 | 459 | 1188 | 49 | 0 | 100.000 |
| FakeHealth | all_identified | intact_publisher_resource_components | 1647 | 459 | 1188 | 49 | 0 | 100.000 |
| FakeHealth | mixed_publishers | direct_restriction | 1624 | 449 | 1175 | 34 | 23 | 98.604 |
| FakeHealth | mixed_publishers | intact_publisher_resource_components | 1624 | 449 | 1175 | 34 | 23 | 98.604 |
| FakeHealth | min5_each | direct_restriction | 1465 | 420 | 1045 | 20 | 182 | 88.950 |
| FakeHealth | min5_each | intact_publisher_resource_components | 1465 | 420 | 1045 | 20 | 182 | 88.950 |
| FakeHealth | min10_each | direct_restriction | 1221 | 335 | 886 | 11 | 426 | 74.135 |
| FakeHealth | min10_each | intact_publisher_resource_components | 1221 | 335 | 886 | 11 | 426 | 74.135 |
| FakeHealth | min20_each | direct_restriction | 824 | 233 | 591 | 5 | 823 | 50.030 |
| FakeHealth | min20_each | intact_publisher_resource_components | 824 | 233 | 591 | 5 | 823 | 50.030 |

No class-exclusive publishers have been removed from existing cohorts. These are hypothetical restriction counts only. Pure all-publisher evaluation is not certified for rows with unknown actual publishers; the strict eligible cohort explicitly records that loss. Mixed-only and minimum-count restrictions can disproportionately remove Fake or Real support, so they cannot silently replace the original CoAID population.

## Cross-dataset overlap

9 normalized families/account proxies overlap; affected rows: FakeHealth 356, CoAID 1105. No overlapping rows are removed. This is a metadata identity overlap audit, not a cross-context transfer rerun.

## Validation and outputs

Validation checks metadata-only normalization, deterministic aliases/counts, row accounting including unresolved identities, intact resource and publisher constraints, and unchanged input hashes. Runtime fit guards prohibit classifier/vectorizer/pipeline fitting. Candidate row assignments exist only transiently in memory; only aggregate partition counts are saved. Original cohort and split files are never opened for writing.

Script: `src/audit_publisher_heldout_feasibility.py`. Tables: `outputs/tables/publisher_family_map.csv`, `publisher_label_support.csv`, `publisher_resource_group_conflicts.csv`, `publisher_heldout_feasibility.csv`, `publisher_mixed_source_cohort_sizes.csv`, `cross_dataset_publisher_overlap.csv`; supplemental `publisher_family_summary.csv`. Adjacent report JSON records inputs, protocol and output fingerprints.
