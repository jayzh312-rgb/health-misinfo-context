# CoAID targeted entity-resolution validation

The current merging procedure is not sufficiently reliable for the primary analysis as a whole. Most audited components have a common specific URL, but a small number of false components contain substantial engagement. This is a validation/proposal audit only: no primary dataset, raw file, or entity assignment was changed and no engagement model was rerun.

All 200 multi-release components were reconstructed exactly from the original clear-duplicate edge graph. Every member was traced to its raw release CSV row, and post/reply ID sets were reread as strings from the original linkage files. All per-member and component union counts match the existing tables. 584 source rows and 6,512 internal pairs were examined, including every pair within the 110- and 19-row components, all top-20 post components, and all four Fake components.

## Validation categories and consequences

| validation_category | components | percent_of_200_components | source_rows | sum_entity_unique_posts | sum_entity_unique_replies |
| --- | --- | --- | --- | --- | --- |
| A. HIGH-CONFIDENCE SAME ENTITY | 185 | 92.5 | 409 | 29974 | 34185 |
| B. PROBABLE SAME ENTITY | 3 | 1.5 | 7 | 400 | 399 |
| C. AMBIGUOUS | 3 | 1.5 | 7 | 1020 | 1610 |
| D. LIKELY OVER-MERGED | 0 | 0 | 0 | 0 | 0 |
| E. DEFINITELY OVER-MERGED | 9 | 4.5 | 161 | 18985 | 21582 |

Ambiguous/over-merged components contain 20,005 within-entity unique post memberships (13.28% of total CoAID posts) and 23,192 reply memberships (19.24% of total replies). These are affected counts, not counts known to be invalid or recommended for deletion. The consequence CSV also gives unique ID unions across affected components.

Questionable components: 10 Real and 2 Fake. Within the audited groups this is 5.10% of Real and 50.00% of Fake components, with only four Fake components available. Most affected components/counts are Real in absolute terms, but the Real component error/ambiguity rate is not higher. This targeted audit does not estimate error rates among unaudited single-release entities.

Labels are rule-based audit judgments supported by local source evidence, not completed human adjudication. A = all members share one specific normalized resource key; B = corroborated informative identity/path-migration evidence without one common key; C = unresolved aliases, insufficient evidence, or weak identification; D = suspect original links plus inconsistent resource/content evidence; E = explicit different item/resource identifiers and incompatible evidence (e.g. different Twitter statuses, dated series installments, CDC press releases, or unrelated source descriptions sharing error/boilerplate text). Similarity thresholds are diagnostic, not semantic proof.

Resource keys are normalized offline: lowercase host, preserve path case, remove fragments and known tracking parameters, retain identity query parameters/social IDs. No live redirect or historical archive lookup was performed. A high-confidence URL match establishes repeated resource capture, not identical historical article text. Same-URL source-title inconsistencies are flagged separately; some July source descriptions appear inconsistent with their linked page metadata.

In particular, 121 of the 185 A-category resource matches have at least one same-URL source-title pair with token Jaccard below 0.30. This is a lexical metadata flag, not 121 proven false merges. For example, the travel-in-the-us URL is described as 'Coronavirus and Travel in the United States' in May and 'If You Have Pets' in July. The A label must therefore not be read as validation of the source-title alignment or an immutable story-level identity. This separate raw metadata issue needs investigation before a rebuild is trusted.

The ambiguous cases explicitly include candidate translated URL variants and a same-directory index/article pair. Different URLs or low lexical similarity alone are not sufficient to call those definitively over-merged; they need historical alias and unit-of-analysis adjudication.

## Original rule failures and transitive links

The previous code admitted an edge as clear if any normalized news URL matched, any normalized content of >=100 characters matched, or an exact title coincided with an exact fact_check_url. It then collapsed connected components. Content boilerplate was not excluded, domain-only values such as cdc.gov were treated as fact-check identifiers, and generic/repeated headings could pass the per-release title-frequency filter. Probable links were not the original collapsing edges; unsafe links had already been labeled clear.

Questionable unsupported original-edge rules (an edge/component may have multiple reasons):

| suspect_rule | questionable_components | unsupported_original_clear_edges |
| --- | --- | --- |
| boilerplate_or_error_content_text_exact | 4 | 1233 |
| generic_title_or_domain_only_factcheck_used_as_identity | 9 | 29 |

Generic-title/domain-only-factcheck matching affects more components; unfiltered exact boilerplate content dominates the number of bad edges because it creates the giant CDC component. Distinguishing component frequency from edge frequency avoids letting one dense graph hide the broader rule failure.

Every original clear edge, reconstruction spanning edge, and all other internal pairs are retained in the link CSV with link_reason, trigger rules, similarities, exact identifiers, and shared social-ID lists. A spanning edge records deterministic reconstruction order, not necessarily a unique historical cause. A weak single-edge bridge is a true graph cut edge; weak cross-group edges can also create a redundant cut set with no single cut edge. Both are reported to avoid missing dense transitive over-merges.

## The 110-member component

`coaid_news_entity_518045a5c665be041af4c5bb` is E. DEFINITELY OVER-MERGED. The 110 records are not all one underlying item. They contain 80 different normalized URL-resource keys, yielding 80 proposed resource groups. This is a conservative proposed resource count, not a proven number of immutable articles: aliases/redirects and changing pages still need adjudication.

It contributes 18,428 unique posts (12.23% of total CoAID posts) and 21,263 unique replies (17.64% of total replies). There is no shared item-specific identifier across all members; cdc.gov identifies the source organization, not a single story.

Original sufficient trigger counts: {"content_text_exact_ge100_chars_alone_sufficient": 1197, "news_url_exact_alone_sufficient": 35, "title_exact_plus_fact_check_url_exact": 9}.

There are 1193 original clear edges crossing validated evidence groups, 0 weak single-edge cut bridges, and 3058 conflicting member pairs without a direct original clear edge. Transitive contamination can therefore persist through redundant weak links, not only one fragile bridge.

Representative raw-source evidence:

| release/id | source title | news URL |
| --- | --- | --- |
| 05-01-2020/1217 | Laboratory Capacity | https://www.cdc.gov/coronavirus/2019-ncov/php/open-america/laboratory.html |
| 07-01-2020/2581 | Talking with children about Coronavirus Disease 2019 | https://www.cdc.gov/coronavirus/2019-ncov/lab/publications.html |
| 05-01-2020/1229 | NCHS Mortality Surveillance Data | https://www.cdc.gov/coronavirus/2019-ncov/covid-data/covidview/04242020/nchs-mortality-report.html |
| 11-01-2020/4340 | Information for Laboratories about Coronavirus (COVID-19) | https://www.cdc.gov/coronavirus/2019-nCoV/lab/index.html |
| 05-01-2020/1282 | Community Mitigation | https://www.cdc.gov/coronavirus/2019-ncov/php/open-america/community-mitigation.html |
| 11-01-2020/4444 | Worker Safety and Support | https://www.cdc.gov/coronavirus/2019-ncov/community/worker-safety-support/index.html |
| 09-01-2020/3315 | Public Service Announcements (PSAs) | https://www.cdc.gov/coronavirus/2019-ncov/communication/public-service-announcements.html |
| 07-01-2020/2403 | People Who Are at Increased Risk for Severe Illness | https://www.cdc.gov/coronavirus/2019-ncov/cases-updates/special-populations/index.html |

Repeated captured text includes the CDC site-branding line about 'saving lives, protecting people'; it is not article identity. Other components repeat a Twitter JavaScript-disabled message, a mod_security rejection, CDC missing-page text, or a recurring series heading. These are content-capture artifacts or shared templates.

The companion 110-member report lists every proposed group and all original member identifiers. The member-evidence CSV preserves complete titles, source titles, captured text and social ID lists for all 200 components.

## Exact proposed replacement rule

1. Keep original release/file/ID provenance immutable. Use item-level normalized news URL/social post ID as the primary resource identity, with conservative normalization and explicit missing/generic-value rejection.
2. Form groups by one common resource key; do not union conflicting keys through generic titles, common source domains, error pages, templated text, or shared social IDs. Do not treat fact_check_url equality alone as original-content identity: one fact-check can discuss multiple items.
3. Combine different resource keys only through an explicitly reviewed alias/redirect mapping, or an adjudicated content-equivalence decision with distinctive title AND informative body evidence. Mask capture errors and sitewide/recurring templates first. Flag translated pages, changing landing pages and differing dated editions for review.
4. For missing URLs, require corroborated distinctive content/title identity with exactly one compatible group; otherwise retain separate unresolved records. Require component-wide compatibility and veto incompatible item IDs instead of allowing transitive chains across different resource keys.
5. Audit source-title/URL/social-link alignment in the raw release records before trusting URL grouping as story identity. Only after adjudication and approval, union post/reply IDs within each corrected group and retain release-level membership/observation provenance. Rebuild affected primary outputs and rerun the later audits/models in a separate authorized step.

There are 9 over-merged components with explicit proposed partitions in coaid_entity_resolution_proposed_groups.csv; none was applied. The 45-component stratified review CSV includes all large (>=10), all Fake, all top-20 post components, all ambiguous/over-merged components, and additional medium/small/high-confidence examples. Human decision fields remain blank.

The sample should be rebuilt after adjudication of the proposals and replacement rules. Even corrected entity resolution will not solve unequal collection windows: valid repeated URLs still accumulate unioned engagement over different releases, some pages evolve, and release membership/capture intensity is highly imbalanced by veracity. Correct identity and comparable observation windows are separate requirements before cross-context analysis.
