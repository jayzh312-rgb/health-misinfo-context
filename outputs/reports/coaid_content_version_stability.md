# CoAID content-version stability audit

Same item-specific URL is not sufficient to establish the same immutable content version. The stored content fields are excerpts (maximum 500 raw characters), and source-title inconsistencies are often metadata discrepancies rather than evidence of a changed article. The audit compares those fields separately and does not rebuild data or rerun EDA/models.

Scope: all 185 previous high-confidence same-resource components, including the 121 flagged source-title cases and 64 unflagged controls; six probable/ambiguous multi-URL identity contexts; and 28 cross-release resource groups proposed inside the nine over-merged parents. The remaining 99 proposed groups have only one observation or only within-release repeats and are inventoried without claiming cross-release stability. Do not pool these cohorts as a random sample of CoAID.

## Same-resource primary results

| classification | resources | source_rows | sum_resource_unique_posts | sum_resource_unique_replies |
| --- | --- | --- | --- | --- |
| A. SAME CONTENT | 6 | 13 | 162 | 21 |
| B. MINOR UPDATE | 91 | 198 | 12723 | 12937 |
| C. MATERIAL CONTENT UPDATE | 5 | 13 | 1795 | 1670 |
| D. REPURPOSED RESOURCE | 0 | 0 | 0 | 0 |
| E. INSUFFICIENT EVIDENCE | 83 | 185 | 15294 | 19557 |

Of 185 resource components, 5 have candidate material/repurposed versions, 91 appear to have only minor updates, and 83 have insufficient evidence. Among the 121 source-title-flagged components, the classification counts are {"B. MINOR UPDATE": 70, "C. MATERIAL CONTENT UPDATE": 4, "E. INSUFFICIENT EVIDENCE": 47}. These are evidence-based audit labels awaiting manual review, not validated whole-page equivalence judgments.

Candidate materially changing/repurposed primary resources contain 13 source rows, 1,795 unique-post memberships and 1,670 unique-reply memberships. These are associated with changing resources, not counts attributable to a specific version or known erroneous counts. Corrected controls and multi-URL identity contexts are quantified separately in the summary CSV.

## Similarity evidence and limits

All raw source title, scraped newstitle, body, abstract, URL, fact-check URL, publication date, archive URL, type, publisher host, and keyword fields are preserved in the new member-evidence CSV. Pair evidence contains exact normalized equality, token Jaccard, normalized edit similarity and body/abstract TF-IDF cosine (word unigrams/bigrams, English stop words, sublinear TF, L2 normalization). Engagement is used only to prioritize human review and quantify consequences, never to classify identity.

The rubric combines fields rather than applying one cosine cutoff. Identical informative body plus unchanged metadata supports SAME CONTENT in the observed capture. Body identity with metadata-only changes supports MINOR UPDATE. Small edits require agreement from multiple body measures plus compatible title/abstract. Material candidates require substantial body change with corroborating abstract/title/update-language/date evidence, or evolving surveillance numbers. Repurposing requires strong divergence in specific scraped title, body and informative abstract together. Missing/error/boilerplate captures, uncorroborated excerpt changes, and unresolved resource aliases are INSUFFICIENT EVIDENCE. Dates are metadata, not capture timestamps, and repeated dates such as 2/11/20 do not prove stability.

Low similarity can reflect different extracted portions of a page, not a different topic. Conversely, an identical 500-character introduction does not establish that later sections, recommendations, or data stayed unchanged. Source-title shifts alone never establish material change or repurposing. Generic CDC headers and error pages never establish sameness.

Qualitative inspection refined four resource decisions: the outbreak FAQ introductions reflect wording/date edits, while healthcare-index, park-guidance, and older-adult captures may select different page sections and remain insufficient. These decisions are bound to exact evidence fingerprints; initial rubric labels/reasons are preserved beside the final decisions. A repeated retired-data notice is treated as a minor notice edit rather than a new surveillance version. This review is of stored excerpts, not a completed human archive adjudication.

## Institutional resource behavior

| resource_behavior | classification | resources |
| --- | --- | --- |
| article/news page (candidate) | A. SAME CONTENT | 6 |
| article/news page (candidate) | B. MINOR UPDATE | 15 |
| article/news page (candidate) | E. INSUFFICIENT EVIDENCE | 21 |
| dashboard/surveillance (likely evolving) | B. MINOR UPDATE | 6 |
| dashboard/surveillance (likely evolving) | C. MATERIAL CONTENT UPDATE | 1 |
| index/resource page (likely maintained) | B. MINOR UPDATE | 7 |
| index/resource page (likely maintained) | E. INSUFFICIENT EVIDENCE | 11 |
| institutional guidance (likely maintained) | B. MINOR UPDATE | 63 |
| institutional guidance (likely maintained) | C. MATERIAL CONTENT UPDATE | 4 |
| institutional guidance (likely maintained) | E. INSUFFICIENT EVIDENCE | 50 |
| social post | E. INSUFFICIENT EVIDENCE | 1 |

Resource types are transparent URL/publisher heuristics, not verified historical diagnoses. CDC guidance is often maintained; dashboards/surveillance report changing data; index/resource pages can retain an identical introduction while their contents change. Dated press releases are more article-like but still require capture comparison. A continuously updated page must not be silently treated as an immutable article. Stored archive references are exported where present; no live archives were retrieved, so these findings concern available dataset captures only.

## Veracity proportions

| cohort | veracity | audited_resources | material_or_repurposed | within_group_percent | insufficient |
| --- | --- | --- | --- | --- | --- |
| same-resource primary | fake | 2 | 0 | 0 | 1 |
| same-resource primary | real | 183 | 5 | 2.732 | 82 |
| corrected resource controls | fake | 0 | 0 | nan | 0 |
| corrected resource controls | real | 28 | 0 | 0 | 26 |

Interpret within-group proportions and insufficient-evidence counts, not just the number of Real cases. The primary same-resource cohort has only two Fake resources, one of which has insufficient evidence. All 28 eligible cross-release corrected controls are Real; the repeated Fake control has only within-release observations and is inventoried outside the version-comparison denominator. Thus apparent concentration among Real resources is descriptive and confounded with resource type and data availability; this audit cannot establish a reliable population-level Fake/Real difference in version instability.

## Proposed version groups and review

| resource_uid | resource_behavior | member_count | classification | candidate_version_count | observed_unique_posts |
| --- | --- | --- | --- | --- | --- |
| coaid_news_entity_3f6e08179da5f196e3e94304 | institutional guidance (likely maintained) | 3 | C. MATERIAL CONTENT UPDATE | 2 | 992 |
| coaid_news_entity_692d8ad3343f700b80dbb715 | institutional guidance (likely maintained) | 3 | C. MATERIAL CONTENT UPDATE | 3 | 778 |
| coaid_news_entity_ae5516dc4e2775a12fbf2b25 | institutional guidance (likely maintained) | 3 | C. MATERIAL CONTENT UPDATE | 2 | 12 |
| coaid_news_entity_b00d022248bda06e8c8a2eff | dashboard/surveillance (likely evolving) | 2 | C. MATERIAL CONTENT UPDATE | 2 | 10 |
| coaid_news_entity_24dfab096d0a1f7b5a3e930b | institutional guidance (likely maintained) | 2 | C. MATERIAL CONTENT UPDATE | 2 | 3 |

SAME CONTENT/MINOR UPDATE resources receive provisional collapse recommendations with all capture provenance retained. MATERIAL CONTENT UPDATE receives canonical_resource_uid + version proposals. REPURPOSED RESOURCE, if supported, receives distinct content-version groups even at the same URL. Version clustering is complete-link: every member pair in a proposed version must support same/minor equivalence, so uncertain or materially different pairs cannot join through a transitive chain. INSUFFICIENT EVIDENCE rows have no assigned version UID and require manual review or conservative sensitivity exclusion. None of these proposals is applied.

The manual-review CSV contains 45 selected high-impact/difficult cases, including institutional pages, material candidates, unresolved aliases, controls from large over-merged parents, and both veracity groups. Complete side-by-side captured evidence is embedded and human adjudication fields remain blank. The six probable/ambiguous original identity contexts must resolve URL aliases/translations first; their conservative E labels are not a claim that those resources changed materially.

## Analytical unit and rebuild decision

Use **validated content resource + validated content version**, with a separate source-record/capture layer recording release/file/row IDs and observed engagement links. A URL is an identity candidate; a release is a capture label, not automatically a content version. Preserve version uncertainty rather than assigning every capture to an immutable resource entity.

Rebuild rule: (1) resolve the prior over-merges into item-specific resource groups and adjudicate cross-URL aliases; (2) validate raw source-title/URL/social-link alignment; (3) mask boilerplate/error text and assess informative body/abstract/title/version evidence; (4) collapse only same/minor version-equivalent captures with provenance; (5) separate materially updated/repurposed versions using complete-link compatibility; (6) keep insufficient cases unresolved, with a pre-specified conservative exclusion sensitivity. Union social IDs only after item/version linkage is justified.

We are not yet ready for an unconditional primary rebuild. The audit supplies executable proposals, but material/insufficient cases need adjudication or an explicit conservative inclusion policy. Release-level post/reply links may not reveal which historical version a post actually referenced; a capture's content version cannot automatically be assigned to every associated social ID. Keep that linkage uncertainty visible before aggregating version-specific engagement. Stable/minor proposals can form a conservative core once prior resource and metadata issues are resolved. No primary data or previous audit outputs were changed.
