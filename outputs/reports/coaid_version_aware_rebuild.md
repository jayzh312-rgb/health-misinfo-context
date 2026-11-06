# CoAID News version-aware rebuild

## Decision and scope

Specification A is PRIMARY: it does not assume cross-release content equivalence when the audit found insufficient evidence. B assumes stability only within the same validated specific resource, while still splitting material versions. C excludes audited insufficient-version and unresolved cross-URL contexts. No raw data or previous primary sample was modified; no EDA, regressions, hurdle models, or cross-context comparisons were run.

## Rebuilt samples

| specification | final_entity_count | fake_count | real_count | included_raw_observations | collapsed_raw_observations | excluded_raw_observations | multi_release_entities |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A_primary_strict | 5337 | 922 | 4415 | 5457 | 120 | 0 | 101 |
| B_resource_sensitivity | 5205 | 921 | 4284 | 5457 | 252 | 0 | 210 |
| C_high_certainty | 5082 | 920 | 4162 | 5202 | 120 | 255 | 101 |

## Identity and version rules

The nine overmerged parents are replaced by exactly the existing corrected-resource proposals, then the content-version audit is applied within each eligible resource. Same/minor groups use the existing proposed version partitions; all five material candidates use their exact complete-link proposed partitions in every specification. Every within-version pair is checked against A/B pair evidence. Six unresolved cross-URL contexts have no validated aliases: A/B retain each source capture separately and C excludes them. No domain, generic title, boilerplate, branding, social overlap, or newly inferred/transitive alias creates a merge.

The audit covers repeated candidate groups, not every conceivable duplicate in the dataset. Original singletons are retained without asserting historical stability. Of the 127 corrected-resource groups, 98 are single observations and one is a two-observation, single-release resource: these remain in all specifications under the explicit corrected-resource rule, not as validated cross-release versions. C is high-certainty relative to the audited cross-release decisions, not a claim that every singleton has verified content or every possible duplicate has been adjudicated. Legacy parent-level possible-duplicate flags are preserved as provenance, not used to redefine C.

Resource keys reuse the prior offline, item-specific URL normalization; they are not newly verified redirects. Nonspecific keys remain blank and cannot merge records. Final UIDs hash the specific resource, aggregation mode, and sorted original source UIDs, so they are independent of dataframe order and consistent across specifications for identical units. Audited version UIDs are also preserved. All rows receive the new UID namespace; membership-change counts below do not count mere UID renaming as a substantive change.

## Versions and uncertainty

| specification | validated_content_version_entities | version_split_entities | uncertain_captures_retained_separately | unresolved_identity_captures | uncertain_capture_post_memberships | uncertain_capture_reply_memberships |
| --- | --- | --- | --- | --- | --- | --- |
| A_primary_strict | 110 | 11 | 241 | 14 | 26014 | 32408 |
| B_resource_sensitivity | 110 | 11 | 0 | 14 | 0 | 0 |
| C_high_certainty | 110 | 11 | 0 | 0 | 0 | 0 |

Validated content-version entities include same/minor collapses and the material candidate partitions; the material candidates produce 11 entities (six more than their five former parents). Insufficient-version capture counts exclude the 14 separate cross-URL identity-uncertain captures. Capture-attached engagement is not proof that each social event refers to the historical page text shown in that release; the stored bodies and linkage do not resolve that attribution.

## Membership changes and engagement accounting

| specification | previous_entities_membership_unchanged | previous_entities_changed_or_excluded | net_entity_count_change | total_unique_post_memberships | total_unique_reply_memberships | global_unique_posts | global_unique_replies | entities_with_zero_posts | entities_with_zero_replies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_primary_strict | 4970 | 103 | 264 | 152223 | 120604 | 148243 | 119617 | 2660 | 4144 |
| B_resource_sensitivity | 5053 | 20 | 132 | 152144 | 120604 | 148243 | 119617 | 2605 | 4036 |
| C_high_certainty | 4970 | 103 | 9 | 124743 | 86165 | 121561 | 85686 | 2600 | 4007 |

A changed previous record means its exact source-member set is no longer an output entity, including full/partial exclusions. Excluded counts refer to raw source observations, not final entities that never existed. Collapsed observations equal included raw observations minus final entity count. Counts are unique ID set unions within each legitimate entity, never sums of old aggregate counts. Summed entity-level memberships can increase after splitting because a social ID may legitimately remain attached to multiple analytical captures. Global distinct ID coverage is separately reconciled; A/B preserve every original News-linked social ID.

## Cross-entity overlap (retained, never silently deleted)

| specification | cross_entity_shared_post_ids | cross_entity_shared_reply_ids | cross_entity_excess_post_memberships | cross_entity_excess_reply_memberships | shared_post_ids_touching_uncertain_captures | shared_reply_ids_touching_uncertain_captures | excess_post_memberships_among_uncertain_captures | excess_reply_memberships_among_uncertain_captures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_primary_strict | 3108 | 906 | 3980 | 987 | 595 | 391 | 230 | 90 |
| B_resource_sensitivity | 3035 | 906 | 3901 | 987 | 0 | 0 | 0 | 0 |
| C_high_certainty | 2395 | 455 | 3182 | 479 | 0 | 0 | 0 | 0 |

One overlap-audit row represents one social ID/type/specification with multiple owners. Excess memberships equal owner count minus one. The 'among uncertain captures' columns count excess ownership within that subset only; 'touching' columns also include overlaps with other types of entity. These overlap definitions are not interchangeable. Boolean/count flags are present on every final entity. Shared IDs and repeated resources imply dependence concerns for later inference; structural validity does not establish independent observations.

## Class composition: descriptive correction only

| specification | veracity | entity_count | post_memberships | reply_memberships | entity_count_change | entity_count_pct_change | post_memberships_change | post_memberships_pct_change | reply_memberships_change | reply_memberships_pct_change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| previous_primary | fake | 903 | 10443 | 7417 | 0 | 0 | 0 | 0 | 0 | 0 |
| previous_primary | real | 4170 | 140205 | 113119 | 0 | 0 | 0 | 0 | 0 | 0 |
| A_primary_strict | fake | 922 | 10443 | 7417 | 19 | 2.104 | 0 | 0 | 0 | 0 |
| A_primary_strict | real | 4415 | 141780 | 113187 | 245 | 5.875 | 1575 | 1.123 | 68 | 0.060 |
| B_resource_sensitivity | fake | 921 | 10443 | 7417 | 18 | 1.993 | 0 | 0 | 0 | 0 |
| B_resource_sensitivity | real | 4284 | 141701 | 113187 | 114 | 2.734 | 1496 | 1.067 | 68 | 0.060 |
| C_high_certainty | fake | 920 | 10443 | 7417 | 17 | 1.883 | 0 | 0 | 0 | 0 |
| C_high_certainty | real | 4162 | 114300 | 78748 | -8 | -0.192 | -25905 | -18.477 | -34371 | -30.385 |

## Effect attributable to the nine corrected parents

| specification | veracity | entity_count | post_memberships | reply_memberships | entity_count_change | entity_count_pct_change | post_memberships_change | post_memberships_pct_change | reply_memberships_change | reply_memberships_pct_change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| previous_primary | fake | 2 | 18 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| previous_primary | real | 7 | 18967 | 21579 | 0 | 0 | 0 | 0 | 0 | 0 |
| A_primary_strict | fake | 20 | 18 | 3 | 18 | 900 | 0 | 0 | 0 | 0 |
| A_primary_strict | real | 137 | 20417 | 21625 | 130 | 1857.143 | 1450 | 7.645 | 46 | 0.213 |
| B_resource_sensitivity | fake | 20 | 18 | 3 | 18 | 900 | 0 | 0 | 0 | 0 |
| B_resource_sensitivity | real | 107 | 20417 | 21625 | 100 | 1428.571 | 1450 | 7.645 | 46 | 0.213 |
| C_high_certainty | fake | 20 | 18 | 3 | 18 | 900 | 0 | 0 | 0 | 0 |
| C_high_certainty | real | 81 | 9776 | 8774 | 74 | 1057.143 | -9191 | -48.458 | -12805 | -59.340 |

These are assignment and sample-composition corrections, not evidence of a substantive Fake/Real engagement effect. In A/B the same underlying source IDs and veracity labels are preserved; changed membership totals arise from overlaps exposed by splits. C additionally removes uncertain observations and their memberships. The table isolates corrected parents from other version splits/exclusions so these mechanisms are not conflated.

## Previous 110-member failure

| specification | final_entity_count | included_raw_observations | excluded_raw_observations | post_memberships | reply_memberships | global_unique_posts | global_unique_replies | invalid_domain_merge_remains | passed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_primary_strict | 108 | 110 | 0 | 19878 | 21309 | 18428 | 21263 | False | True |
| B_resource_sensitivity | 80 | 110 | 0 | 19878 | 21309 | 18428 | 21263 | False | True |
| C_high_certainty | 56 | 58 | 52 | 9447 | 8490 | 8010 | 8474 | False | True |

The invalid connected component is removed in every specification. Any multi-member descendant must have one specific resource key and remain inside a single audited corrected-resource proposal. C excludes uncertain descendants rather than forcing them into a resource-level entity. This fixes the domain-level overmerge, not the unresolved historical attribution of individual social events.

## Individual material-version validation

| specification | previous_entity_uid | final_entity_count | passed |
| --- | --- | --- | --- |
| A_primary_strict | coaid_news_entity_692d8ad3343f700b80dbb715 | 3 | True |
| A_primary_strict | coaid_news_entity_3f6e08179da5f196e3e94304 | 2 | True |
| A_primary_strict | coaid_news_entity_ae5516dc4e2775a12fbf2b25 | 2 | True |
| A_primary_strict | coaid_news_entity_24dfab096d0a1f7b5a3e930b | 2 | True |
| A_primary_strict | coaid_news_entity_b00d022248bda06e8c8a2eff | 2 | True |
| B_resource_sensitivity | coaid_news_entity_692d8ad3343f700b80dbb715 | 3 | True |
| B_resource_sensitivity | coaid_news_entity_3f6e08179da5f196e3e94304 | 2 | True |
| B_resource_sensitivity | coaid_news_entity_ae5516dc4e2775a12fbf2b25 | 2 | True |
| B_resource_sensitivity | coaid_news_entity_24dfab096d0a1f7b5a3e930b | 2 | True |
| B_resource_sensitivity | coaid_news_entity_b00d022248bda06e8c8a2eff | 2 | True |
| C_high_certainty | coaid_news_entity_692d8ad3343f700b80dbb715 | 3 | True |
| C_high_certainty | coaid_news_entity_3f6e08179da5f196e3e94304 | 2 | True |
| C_high_certainty | coaid_news_entity_ae5516dc4e2775a12fbf2b25 | 2 | True |
| C_high_certainty | coaid_news_entity_24dfab096d0a1f7b5a3e930b | 2 | True |
| C_high_certainty | coaid_news_entity_b00d022248bda06e8c8a2eff | 2 | True |

All nine overmerge checks per specification are recorded alongside the five material-version checks in `outputs/tables/coaid_version_aware_component_validation.csv`. All pass.

## Integrity and pipeline handoff

Passed: unique final UIDs; complete/disjoint source lineage and explicit exclusions; raw-to-processed-to-audit social-set agreement; final within-entity set uniqueness; no source assigned twice; exact approved version partitions; no surviving old overmerged component; no multi-resource/domain-only merge; global social coverage and overlap reconciliation; all legacy primary columns retained; written CSV count/flag/lineage round-trip validation; unchanged input SHA-256 hashes. The entity-lineage CSV and per-row capture metadata preserve raw file/row references, original IDs, releases, titles, text and representative selection. Timestamps are recomputed from final post unions and remain secondary sensitivity variables.

The three CSVs are internally valid and schema-ready. Existing EDA/model scripts still hard-code `coaid_news_primary.csv`, so they will NOT automatically select these samples. A later authorized pipeline run must explicitly select the new input path/specification and respect its overlap/uncertainty flags; legacy sensitivity branches do not define these new A/B/C specifications. No pipeline scripts were changed or executed here.

## Reproduction and artifacts

Run `.venv/bin/python src/rebuild_coaid_version_aware_samples.py`. Run the accompanying tests with `.venv/bin/python -m unittest discover -s src -p 'test_coaid_version_aware_rebuild.py'`. Input hashes and validation results are in `outputs/tables/coaid_version_aware_rebuild_metadata.json`. Existing source/audit inputs remain unchanged. The script writes only the three requested new samples and version-aware rebuild artifacts, and refuses to proceed when authoritative audit partitions no longer agree.

Fingerprint manifest covers 35 input files, including every raw News content/social mapping read.
