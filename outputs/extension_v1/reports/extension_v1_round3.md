# Extension v1, round 3 — FakeNewsNet dependence sensitivities

Date: 2026-10-12. Protocol written first: `protocols/fakenewsnet_round3_protocol.md`. Script: `scripts/task_c_fakenewsnet_round3.py`, reusing the round-2 loaders, dependence graph and selected family and the frozen fitting, standardization and Cartesian-contrast helpers. Frozen baseline untouched. Progress lines were appended to `logs/extension_v1_fakenewsnet_engagement.log` (the imported round-2 module had already configured logging), not to a separate round-3 file.

## A1. Dominant-component exclusion (`tables/fakenewsnet_round3_exclusion_results.csv`, `_contrast.csv`, `_bootstrap_refits.csv`)

The largest shared-tweet component was dropped (PolitiFact: 84 rows carrying 66.7% of post memberships, 5 Fake / 79 Real; GossipCop: 692 rows carrying 19.1%, 582 Fake / 110 Real). The round-2 hurdle was refitted on the remaining rows without reselection and re-standardized over them; 1,000 successful component-bootstrap refits were reached in 1,000 attempts for both subsets (seed 42, ordinals 2 and 3). **As in the frozen CoAID sensitivity, exclusion changes the reference population; these are not replacement estimates.**

| Subset | Rows kept | Primary ratio [95%] (round 2) | Exclusion ratio [95%] | P(posts>0) Fake / Real after exclusion | Conditional ratio after exclusion [95%] |
|---|---:|---:|---:|---:|---:|
| PolitiFact | 968 | 0.559 [0.439, 1.868] | **1.446 [1.073, 2.046]** | 0.906 / 0.604 | 0.964 [0.726, 1.351] |
| GossipCop | 21,448 | 2.144 [1.212, 3.290] | **1.373 [1.146, 1.621]** | 0.960 / 0.937 | 1.340 [1.118, 1.577] |

PolitiFact's primary ratio below 1 was produced entirely by the 84-row component: without it, Real items that are tweeted no longer have more tweets than Fake items (conditional ratio 0.96), and the higher Fake occurrence probability drives the overall ratio above 1. GossipCop's ratio falls from 2.1 to 1.4 but stays above 1 with its interval excluding 1.

Ratio-of-ratios against the saved references (subset ratio / reference ratio; frozen Cartesian-product rule; reference populations unchanged):

| Subset (excluded) | Reference | Ratio-of-ratios | 95% interval | Excludes 1? | Primary (round 2) |
|---|---|---:|---:|---|---:|
| PolitiFact | CoAID A (0.2528) | 5.720 | [2.496, 11.645] | yes | 2.212 [0.987, 9.894] |
| PolitiFact | FakeHealth (0.8403) | 1.721 | [1.215, 2.485] | yes | 0.666 [0.488, 2.291] |
| GossipCop | CoAID A (0.2528) | 5.433 | [2.506, 10.506] | yes | 8.481 [2.821, 18.225] |
| GossipCop | FakeHealth (0.8403) | 1.635 | [1.277, 2.057] | yes | 2.552 [1.357, 4.087] |

## A2. Multi-membership diagnostic, label-blind (`tables/fakenewsnet_round3_multimembership.csv`)

| Scope | Distinct tweet IDs | Share attached to >1 row | Rows per ID: 1 / 2 / 3–4 / 5–9 / 10+ (share of IDs) | Share of memberships from multi-row IDs | Max rows per ID |
|---|---:|---:|---|---:|---:|
| PolitiFact, all rows | 554,820 | 3.73% | 96.27% / 3.24% / 0.50% / 0.0004% / 0% | 7.6% | 6 |
| GossipCop, all rows | 1,390,131 | 3.70% | 96.30% / 3.03% / 0.41% / 0.12% / 0.14% | 9.5% | 18 |
| PolitiFact 84-row component | 365,712 | 5.09% | 10+: 0 IDs (0.00%) | 10.0% | 6 |
| GossipCop 692-row component | 244,503 | 12.23% | 10+: 11 IDs (0.0045%) | 24.2% | 10 |

The PolitiFact component is not held together by a few tweets attached to many rows: no tweet ID in it is attached to ten or more rows, and 95% of its 365,712 IDs belong to a single row. It is a chain of pairwise sharing among 84 mostly-Real items that together hold two thirds of all PolitiFact memberships. Treating it as one resampling unit is therefore the conservative choice, and its influence on the ratio is a property of the data, not of a few duplicated tweets.

## A3. News-URL domain composition, label-blind extraction (`tables/fakenewsnet_round3_domain_composition.csv`)

GossipCop Real (16,817 rows; 1,762 distinct domains; 13 missing): people.com 9.3%, dailymail.co.uk 4.6%, en.wikipedia.org 3.7%, etonline.com 3.5%, usmagazine.com 3.3%, longroom.com 3.3%, usatoday.com 1.8%, hollywoodreporter.com 1.8%, variety.com 1.5%, ew.com 1.3%; top ten cover 34.1%. GossipCop Fake (5,323 rows; 733 domains; 256 missing): hollywoodlife.com 8.6%, (missing) 4.8%, people.com 4.1%, dailymail.co.uk 3.6%, radaronline.com 3.3%, eonline.com 2.9%, usmagazine.com 2.8%, imdb.com 2.7%, newidea.com.au 2.6%, en.wikipedia.org 2.3%; top ten 37.7%.

**eonline.com accounts for 0 of 16,817 GossipCop Real rows (0.0%) and 154 of 5,323 Fake rows (2.9%).** etonline.com (Entertainment Tonight, a different site) is 3.5% of Real. The FakeNewsNet paper states that all E! Online articles were taken as real news; the `news_url` field of the local GossipCop real file does not reflect that, either because the released real set differs from the paper's description or because `news_url` records the story's subject page rather than the E! Online article. The premise that GossipCop Real is concentrated on E! Online is therefore not supported by the local copy, and the S13/S17 captions should describe GossipCop Real as "items released as real by FakeNewsNet" rather than as E! Online stories.

PolitiFact: web.archive.org is the top domain in both classes (Fake 16.1%, Real 20.4%); Real also includes youtube.com 7.6% and 9.2% missing URLs.

## Interpretation bounds

Same as round 2 (title-search tweet collection, 2016–2018 crawls, no replies or retweets, inherited labels), plus: the PolitiFact direction is not stable under the dominant-component exclusion, so no direction should be claimed for PolitiFact; GossipCop stays above 1 under both populations. No frozen estimate changes.

---

*Correction, 2026-10-13 (round 4, Check 7): the GossipCop exclusion ratio was displayed as 1.374; the saved value is 1.3734525 (`tables/fakenewsnet_round3_exclusion_results.csv`), which rounds half-up to 1.373. The table above now shows 1.373. No other value changed.*
