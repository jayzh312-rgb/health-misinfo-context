# Engagement comparability feasibility audit

Audit date: 2026-10-09. Status: feasibility only; frozen analyses and manuscript preserved.

## Judgment and recommended next step

**C — Existing data do not support a defensible publisher-and-topic restricted comparison of meaningful breadth.** Ten identifiable publisher families occur in both engagement populations, but only ABC News and CNN contain both analytical classes in each dataset. Their common-support population contains just **11 CoAID entities (5 Fake, 6 Real)** and **97 FakeHealth items (28 Fake, 69 Real)**. Only 2 of those CoAID Fake entities and 1 Real entity have any observed posts. There is no validated common topic taxonomy, and topic restriction within those publishers could only reduce support.

A pooled contrast among all shared publishers is numerically definable, but it retains only 12 CoAID Fake entities and leaves source and class strongly separated. It would not establish that the primary contrast persists after a useful reduction in publisher and topic differences. This judgment concerns the requested restricted comparison; it does not invalidate the frozen, explicitly observational full-cohort result.

**One recommended next step:** prepare a collection protocol for a new dataset with the same publisher panel, common topic annotations, consistent label definitions, and identical post-observation rules across contexts. Do not start collection or modeling under this audit. Requirements appear below.

## 1. Populations and metadata-only procedure

| Population | CoAID News | FakeHealth HealthStory |
|---|---:|---:|
| Primary engagement population, N (Fake / Real) | 5,337 (922 / 4,415) | 1,690 (472 / 1,218) |
| Frozen eligible text-modeling population, N (Fake / Real) | 3,577 (825 / 2,752) | 1,647 (459 / 1,188) |

The audit targets CoAID `A_primary_strict` and the complete HealthStory engagement cohort, including entities with zero observed posts. It does not apply text-quality or text-modeling eligibility exclusions. HealthRelease, CoAID Claims, and other datasets are outside the comparison.

Prepared metadata in the two `*_text_modeling_ready.csv` files retain all 5,337 and 1,690 engagement UIDs, respectively. Only metadata columns were selected. One-to-one joins verified complete UID coverage, identical analytical labels, and agreement between prepared article URLs and the engagement cohorts’ `news_url` / `source_link` fields. Article text, model predictions, weights, notebooks, social-post IDs, and bootstrap draws were not inspected.

Publisher identities use the existing label-blind definitions in `src/audit_publisher_heldout_feasibility.py` and the fixed aliases in `src/audit_text_quality_and_source_cues.py`. The existing metadata identity helper was applied in memory to the full cohorts; no scientific analysis or model entrypoint was executed. Hosts, audited aliases, and unambiguous name–host associations define website/source families. Explicit account URLs may identify account proxies; unresolved platform hosts remain unresolved. These are publication/source proxies, not verified corporate ownership or endorsements of every attributed item.

Existing family IDs were anchored to `outputs/tables/publisher_family_map.csv`. Adding metadata changed the helper’s lexical representative for one identity group; its frozen name was preserved. No previously identified family was split or merged, and all 5,224 existing row-level family/status assignments were retained. The source map was not rewritten. Labels entered only subsequent support counts.

## 2. Publisher overlap and class support

Counts below are engagement entities/items. Parentheses contain **Fake / Real**. Families with unresolved identities are excluded from identified-family restrictions.

| Restriction | CoAID families | CoAID N (Fake / Real) | CoAID retained | FakeHealth families | FakeHealth N (Fake / Real) | FakeHealth retained |
|---|---:|---:|---:|---:|---:|---:|
| Identified families, including account proxies | 262 | 4,689 (306 / 4,383) | 87.86% | 51 | 1,689 (471 / 1,218) | 99.94% |
| Both classes within the respective dataset | 9 | 38 (24 / 14) | 0.71% | 36 | 1,666 (461 / 1,205) | 98.58% |
| Shared families, regardless of class | 10 | 2,190 (12 / 2,178) | 41.03% | 10 | 382 (151 / 231) | 22.60% |
| Shared families with both classes in each dataset | 2 | 11 (5 / 6) | 0.21% | 2 | 97 (28 / 69) | 5.74% |
| Shared families with ≥5 per class in each dataset | 0 | 0 (0 / 0) | 0.00% | 0 | 0 (0 / 0) | 0.00% |

The ≥5 threshold is a descriptive sparsity check, not a power calculation or a claim that five items would suffice. No CoAID family, even outside the shared set, has five entities in each class. Shared ≥10 and ≥20 thresholds also retain zero families. The CSV includes these checks and the corresponding within-dataset restrictions.

### Exact shared-family counts

| Prepared publisher family | CoAID Fake | CoAID Real | HealthStory Fake | HealthStory Real | Both classes in both datasets? |
|---|---:|---:|---:|---:|---|
| abcnews.go.com | 2 | 2 | 9 | 20 | Yes |
| azcentral.com | 0 | 1 | 1 | 0 | No |
| cbsnews.com | 2 | 0 | 19 | 16 | No |
| cnn.com | 3 | 4 | 19 | 49 | Yes |
| foxnews.com | 4 | 0 | 20 | 9 | No |
| medicalnewstoday.com | 0 | 459 | 0 | 2 | No |
| nbcnews.com | 0 | 1 | 9 | 26 | No |
| newsweek.com | 1 | 0 | 7 | 19 | No |
| theguardian.com | 0 | 1 | 15 | 6 | No |
| webmd.com | 0 | 1710 | 52 | 84 | No |

Shared publisher identity does not establish within-publisher class support. WebMD and Medical News Today contribute 2,169 of the 2,178 retained CoAID Real entities, with no CoAID Fake entities. Restricting to the ten shared families retains 41.03% of CoAID overall but only 12/922 Fake entities (1.30%). It retains 22.60% of HealthStory. Pooling this set still compares very different publisher compositions across classes.

Requiring both labels within each shared family retains only 0.21% of CoAID and 5.74% of HealthStory. ABC News supplies CoAID 2 Fake / 2 Real; CNN supplies 3 Fake / 4 Real. In these two families, ABC News has 2 Fake / 0 Real entities with positive observed-post counts, and CNN has 0 Fake / 1 Real. All zeros remain legitimate observed-count records; they are not exclusions. The resulting population would be a very small selected mainstream-news subset, excluding most social-platform misinformation sources, institutional guidance, and health-publisher content. It cannot represent the original dataset-context contrast broadly.

### Missing identities and the earlier sparse-support finding

| Engagement metadata status | CoAID N (Fake / Real) | HealthStory N (Fake / Real) |
|---|---:|---:|
| Identified website/source family | 4,600 (233 / 4,367) | 1,689 (471 / 1,218) |
| Identified account proxy | 89 (73 / 16) | 0 (0 / 0) |
| Missing source metadata | 43 (40 / 3) | 1 (1 / 0) |
| Unresolved platform publisher | 605 (576 / 29) | 0 (0 / 0) |

CoAID therefore has 648 unresolved/missing rows (616 Fake, 32 Real). Removing them loses 66.81% of engagement-cohort Fake support. Platform hosting is not an identifiable shared publisher, and unresolved rows cannot be matched by assigning a single “unknown” publisher. HealthStory has 19 rows without a prepared source name (8 Fake, 11 Real); host evidence resolves all but one. Missing source-name metadata and unresolved publisher identity are distinct.

The earlier **text-cohort** audit remains unchanged: CoAID had 8 mixed-label families containing 32 eligible rows (19 Fake, 13 Real), versus 34 mixed-label HealthStory families containing 1,624 eligible rows (449 Fake, 1,175 Real). Its cross-dataset table listed 9 shared families. The full engagement audit finds 9 CoAID mixed families with 38 rows and adds Newsweek to shared-family overlap; it does not overturn the earlier sparse-support conclusion. CoAID matching cannot manufacture the absent within-publisher class cells.

## 3. Existing topic metadata

| Existing field | Engagement coverage and class counts | Comparability assessment |
|---|---|---|
| CoAID prepared topic annotation | No dedicated topic field in the 5,337-row engagement cohort or prepared metadata | No validated common topic classification |
| CoAID original `meta_keywords` | Listed in the existing source-schema inventory for all eight News content tables; nonmissing coverage not established here | Unstandardized source keywords; not retained as a common annotation in prepared cohorts |
| HealthStory `review_tags_json` | Nonempty: 1,194 (296 Fake / 898 Real); empty: 496 (176 / 320); 589 distinct literal tags | Native, multiple tags per item; no validated CoAID crosswalk |
| HealthStory `review_category` | Nonempty: 1,326 (342 / 984); missing: 364 (130 / 234) | Mixes subject, publisher, and review themes; not a uniform topic taxonomy |

For example, native HealthStory categories include Cancer (163: 41 Fake / 122 Real), Alzheimer’s (39: 11 / 28), ABC News (15: 8 / 7), and Conflict of interest (2: 1 / 1). Native tags include breast cancer (65: 14 / 51), prostate cancer (45: 11 / 34), and weight loss (37: 6 / 31). These are existing literal annotations, not newly assigned topics. The CSV preserves native category/tag counts separately; tag counts overlap and must not be summed as disjoint populations.

There are **no established common topic categories or publisher-by-topic groups supported with both labels in both datasets**. Their retained class counts are not assessable, rather than zero. The CSV marks these counts blank with `status=not_established`. Broad “health” membership or CoAID’s COVID focus does not establish within-topic comparability. The missing prepared CoAID annotation is already sufficient to identify this gap; original records were not reopened to count keywords.

If a later, separately authorized task needs topic composition, use a **label-blind annotation protocol**: define the research-relevant subject taxonomy and granularity before examining labels or engagement; distinguish subject matter from genre, publisher, and review-quality criteria; annotate comparable article fields across datasets with publisher/review cues removed where possible; hide dataset labels, ratings, and outcomes; allow multiple topics plus uncertain/unclassifiable cases; use independent annotators, adjudication, and an agreement/coverage audit. Specify missing-text treatment and freeze the crosswalk before calculating class support. COVID content may reveal context despite masking, so complete context blinding cannot be promised. No such labels or topic model were produced here. Annotation could inform a separate topic-only question, but cannot increase the 11-entity CoAID publisher common-support population.

## 4. Measurement and observation-window differences

**Collection and outcome.** The common primary outcome is unique observed posts per entity, including zeros. CoAID uses release/file-local content-to-post linkage and unions IDs across validated member captures; HealthStory uses the item mapping’s tweet IDs. FakeHealth’s separate retweet mappings are excluded because CoAID has no equivalent mapping. Reply IDs help define dependence and secondary outcomes; they are not added to the primary post count. These are captured engagement counts, without a common exposure denominator, complete cascades, or verified recall of historical platform activity.

The collection documentation and existing alignment audit do not establish identical search queries, URL handling, API/search coverage, crawl timing, or per-item stopping rules. FakeHealth’s README describes separate content/review/engagement components and API retrieval of released social IDs; this does not prove the collection procedure that generated every original mapping. CoAID’s README documents successive collection additions. Publisher/topic matching cannot remove these measurement differences or recover private, deleted, and uncaptured posts.

**Time-field coverage in the primary engagement populations:**

| Timing metadata | CoAID N (Fake / Real) | HealthStory N (Fake / Real) |
|---|---:|---:|
| Valid publication-date quality flag | 1,080 (139 / 941) | 1,038 (292 / 746) |
| Missing publication date | 4,247 | 652 |
| Unparsable / implausible-range publication date | 9 / 1 | 0 / 0 |
| First and last observed-post times present | 2,677 (147 / 2,530) | 1,683 (471 / 1,212) |

CoAID has 1,081 nonempty parsed date values, but one carries an implausible-range flag; it is excluded from the 1,080 valid dates above. Valid stored publication dates range from 2016-01-15 to 2020-12-31 in CoAID and 2009-12-10 to 2018-12-17 in HealthStory. Valid parsing does not verify publication or exposure timing, and undated HealthStory records are not individually certified as pre-COVID.

Stored observed-post timestamp extrema span 2019-12-01 to 2020-10-31 in CoAID and 2010-11-04 to 2019-12-03 in HealthStory. These extrema come from prepared fields, without reopening social IDs. They describe recorded activity, not collection opening/closing dates or comparable article-level follow-up. The historical date audit also flagged publication dates later than the first observed post; date presence alone is insufficient for a reliable exposure offset.

CoAID release directories are `05-01-2020`, `07-01-2020`, `09-01-2020`, and `11-01-2020`; its README describes successive additions for May–July, July–September, and September–November 2020. These are collection/release provenance. **First-recorded release is not an article publication date, collection-window start, or observation duration.** No verified common per-entity start/end observation window is available in the audited prepared metadata. No common follow-up duration or time adjustment was inferred.

**Labels.** CoAID retains original News Fake/Real filename categories. HealthStory uses expert-review rating <3 versus ≥3, reflecting reporting quality under the review criteria. This is not equivalent to factual-falsity versus truth. Shared publishers, health subjects, and analytical label names do not harmonize these constructs.

**Dependence and unit.** CoAID contains version-aware entities and separately retained uncertain captures; HealthStory uses reviewed items. Their frozen overlap graphs group entities sharing observed post/reply IDs. Source restriction does not establish independence, remove repeated-resource dependence, or resolve CoAID’s dominant overlap component. A future authorized comparison would need preserved entity lineage, overlap dependence, and consideration of additional publisher/time dependence. Two shared mixed-label publishers do not provide a convincing basis for publisher-cluster inference. No graph, interval, outcome, or model was recomputed.

## 5. Requirements for a new consistently collected dataset

1. Sample the same verified publisher panel across crisis and comparison contexts, with multiple publishers and enough Fake/Real support within comparable subject/genre cells. Establish feasibility and precision requirements before collecting outcomes; avoid selecting groups for favorable contrasts.
2. Use the same subject taxonomy and annotation protocol across contexts. Record original publication, syndication/attribution, account identity, corporate relationships where verified, and missing/uncertain identities separately.
3. Apply a common item-level factual-veracity protocol across periods. Record reporting quality as a separate construct if needed, with comparable verification and sampling procedures.
4. Collect the same platform outcome using identical query and linkage rules, a prespecified article-relative follow-up horizon, and recorded start/end times, query variants, API limits, retrieval dates, failures, deletions, and censoring. Distinguish observed zero from failed or missing collection. Keep posts, replies, and retweets separate.
5. Retain publication/update timestamps, content-version lineage, and observed social overlaps so repeated captures, shared posts, publisher effects, and time dependence can be handled without treating every row as independent.

A consistently collected panel could support an observational within-publisher/topic contrast. Publisher/topic matching alone would still not identify a causal COVID-period effect; contextual variation and residual confounding would require a separate design argument.

## 6. Sources, aggregate-output interpretation, and validation

Authoritative local inputs:

- Instructions and frozen science: `AGENTS.md`; `outputs/manuscript/methods_final.md` §§2.1–2.5 and 2.9; `outputs/manuscript/review_package_v2/manuscript.md` engagement Results §3.1; `outputs/reports/cross_context_engagement_contrast.md`.
- Engagement cohorts: `data/processed/coaid_news_primary_version_aware.csv`; `data/processed/fakehealth_healthstory_primary.csv`.
- Prepared source metadata: `data/processed/coaid_text_modeling_ready.csv`; `data/processed/fakehealth_text_modeling_ready.csv` (metadata columns across all rows).
- Prior publisher evidence: `outputs/reports/publisher_heldout_feasibility.md`; `outputs/tables/publisher_family_map.csv`; `outputs/tables/publisher_family_summary.csv`; `outputs/tables/cross_dataset_publisher_overlap.csv`; existing identity helper/alias source files named above.
- Measurement/collection evidence: `outputs/tables/coaid_fakehealth_measurement_alignment.csv`; `outputs/tables/coaid_fakehealth_feature_comparison.csv`; `outputs/tables/coaid_fakehealth_publication_date_audit.csv`; `outputs/tables/fakehealth_inspection_report.txt`; the two repository `README.md` documentation files under `data/raw/coaid/CoAID-master/` and `data/raw/FakeHealth-master/`. Only documentation was opened under raw directories.
- Original CoAID keyword-field schema: `outputs/tables/coaid_dataset_summary.csv`; no original content records were reopened.

**Validation: PASS.** Full engagement UID/label/URL joins passed; aggregate class counts reconcile; all 2 cohort totals match frozen Methods; all 5,224 existing publisher assignments remain unchanged. Input SHA-256 checks passed for 18 inspected source/documentation artifacts. Size/modification-time inventories confirmed that all 1,886 pre-existing files under the monitored project paths remained unchanged. The latter check is a session integrity check, not a new scientific reproducibility validation or a checksum claim for unopened models/draws. No protected artifact was written.

The support CSV contains aggregate cohort, publisher restriction, shared-family, metadata-coverage, and native category/tag rows. `retained_percent` always uses the full engagement cohort denominator. Positive-post columns are counts of entities with nonzero already-stored outcomes, not new engagement estimates. Overlapping restrictions and multiple tags must not be summed. Blank common-topic counts mean support is not established. No article text, article-level identifiers, URLs to individual articles, or social-post IDs are included.

Outputs:

- `outputs/reports/engagement_comparability_feasibility.md`
- `outputs/tables/engagement_comparability_support.csv`

Audit stopped here. No model, matching analysis, topic annotation/model, new scientific interval, or manuscript revision was performed.

### Fingerprints of principal prepared inputs

| Input | SHA-256 |
|---|---|
| `data/processed/coaid_news_primary_version_aware.csv` | `123b0486b0ad4569d2ded7f3cc7c94077ee2d8953f432bd56a0a6c69d492c5e8` |
| `data/processed/fakehealth_healthstory_primary.csv` | `c975e55efaad9010814d696b6886d1c7de6315a63f05c9fedae443e1d7655e9e` |
| `data/processed/coaid_text_modeling_ready.csv` | `859b3db8888962de771f4ff48779f139027c9472bbe19367755f0657ad33812d` |
| `data/processed/fakehealth_text_modeling_ready.csv` | `a38730db184727e3a84d2f603789e817c667fe75c6606d834f09ba106ef5cb91` |
| `outputs/tables/publisher_family_map.csv` | `1501c74686a0b59b8de70d1478d001c53228693fe2be753d27085494bd0b57d4` |
| `src/audit_publisher_heldout_feasibility.py` | `4eef8e4e8d1bdec6a477a6ce3dd95cb5f6cd7cdb8bc8d1e3fc4f737c0c106d55` |
| `src/audit_text_quality_and_source_cues.py` | `042dfc75b7ae77abe291ba2e0721e8757a2aa9d96486ce9f28ae0aa84c1dd9a0` |
