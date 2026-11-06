# Protocol: FakeNewsNet dependence sensitivities (extension_v1, round 3)

Written 2026-10-12 before any round-3 computation. Mirrors the frozen CoAID dependence checks (dominant-component exclusion in `src/run_cross_context_engagement_contrast.py`, overlap diagnostics in `src/run_version_aware_engagement_analysis.py`). Frozen baseline untouched; outputs only under `outputs/extension_v1/`; progress log under `logs/`. Inputs: the round-2 cohorts, dependence graphs and selected families (`protocols/fakenewsnet_engagement_protocol.md`, `tables/fakenewsnet_engagement_*.csv`), reused unchanged.

## A1. Dominant-component exclusion

For each subset, the largest shared-tweet component by post memberships (PolitiFact: 84 rows, 578,316 × 0.667 memberships; GossipCop: 692 rows, 19.1%) is dropped. The remaining rows define the sensitivity population and its standardization distribution. The round-2 selected family (hurdle, logit + zero-truncated NB2, intercept + Fake indicator) is refitted on the remaining rows without reselection; the standardized Fake/Real ratio is computed by the row-level-product rule over the remaining rows; 1,000 successful component-bootstrap refits of the remaining components are drawn with `SeedSequence([42, ordinal, attempt])`, ordinal 2 = PolitiFact exclusion, 3 = GossipCop exclusion (distinct from the round-2 streams 0 and 1), attempt cap 1,250, warm start plus frozen same-draw multistart retry, linear 2.5/97.5 percentiles. Ratio-of-ratios against the saved CoAID A (0.2528) and FakeHealth (0.8403) draws use the frozen Cartesian-product rule, C = subset ratio / reference ratio. **As in the frozen CoAID sensitivity, exclusion changes the reference population: the sensitivity ratio describes a different population from the primary ratio and is not a replacement estimate.** Reported next to the round-2 primary ratios.

## A2. Multi-membership diagnostic (label-blind)

Per subset: number of distinct tweet IDs; share attached to more than one row; distribution of rows per tweet ID in bins 1, 2, 3–4, 5–9, 10+ (ID counts and shares); and the share of all post memberships contributed by IDs attached to more than one row. For the PolitiFact 84-row component: distinct tweet IDs it contains, and the share of those IDs attached to ten or more rows. Labels are not used.

## A3. News-URL domain composition (label-blind extraction, tabulated by class)

Domain = lower-cased host of `news_url` with a leading `www.` removed; empty or unparsable URLs are reported as `(missing)`. For each subset and class, the ten most frequent domains with row counts and row shares within the class, plus the share of the class covered by the top ten. Labels enter only the final tabulation.

## Limitations

Same as round 2: title-search tweet collection, 2016–2018 crawls, no replies or retweets, inherited labels. Exclusion results are population-changing sensitivities, not alternative primaries. No frozen estimate is altered.

Outputs: `tables/fakenewsnet_round3_exclusion_results.csv`, `tables/fakenewsnet_round3_exclusion_contrast.csv`, `tables/fakenewsnet_round3_exclusion_bootstrap_refits.csv`, `tables/fakenewsnet_round3_multimembership.csv`, `tables/fakenewsnet_round3_domain_composition.csv`, `tables/fakenewsnet_round3_run_metadata.json`; `reports/extension_v1_round3.md`; log `logs/extension_v1_fakenewsnet_round3.log`.
