# Reference checks for the v3 draft's bracketed citations — 2026-10-12

Each source was retrieved from its official record or a readable rendering of its preprint (ar5iv) on 2026-10-12. Quotations are short verbatim excerpts. Where the quoted text comes from the arXiv version rather than the published version, this is stated. Draft line numbers refer to `manuscript_v3_draft.md`.

## 1. Dai, Sun and Wang (2020) — FakeHealth

**Metadata (official record, ojs.aaai.org; Crossref DOI).** Enyan Dai, Yiwei Sun, Suhang Wang. "Ginger Cannot Cure Cancer: Battling Fake Health News with a Comprehensive Data Repository." *Proceedings of the International AAAI Conference on Web and Social Media*, 14(1), 853–862, 2020 (published 2020-05-26). DOI 10.1609/icwsm.v14i1.7350. Preprint arXiv:2002.00837 (v2, 30 Mar 2020). Quotations below are from the arXiv v2 text (ar5iv rendering).

**Draft claims and support.**
- L21: "a story is labeled Fake when expert reviewers rate its reporting as poor, for example because it overstates benefits, omits harms or costs, or relies on weak evidence." — **Supported.** "On the HealthNewsReview.org, both news stories and news releases are evaluated by experts on 10 criteria." (Dataset Description); "The rating score ranges from 0 to 5."; "we treat news pieces whose scores lower than 3 as fake news." The criteria include "Does it adequately quantify the benefits...?", "Does it adequately explain/quantify the harms...?", "Does it adequately discuss the costs...?", "Does it seem to grasp the quality of the evidence?" (Appendix A, Table 6).
- L21: "These stories come largely from mainstream outlets that also publish stories rated Real." — **Supported** for the first half: "HealthNewsReview.org reviews news stories from main US media and news releases from institutes." (Dataset Collection). The second half (same outlets publish both classes) is the project's own finding (publisher-family audit: 36 of 51 families carry both classes), not a statement in the paper.
- L222: "FakeHealth's model leaned on vocabulary that echoes its reviewers' criteria, with press-release language toward Fake and discussion of costs toward Real." — **Supported** as to the criteria: "Does the story appear to rely solely or largely on a news release?" (criterion S10) and "Does it adequately discuss the costs of the intervention?" (C8) are both review criteria.
- Additional fact relevant to Discussion 4.2 (L226): FakeHealth also collected engagements by search. "Firstly, we used the titles as search queries to get tweets."; "Then, we extracted the key words of the title..."; "Finally, we set the long URL of the news as query to search tweets." (Data Collection Pipeline, Social Engagements). The draft presents search-based collection as a CoAID-specific explanation for Fake zeros; both datasets used title search, so the sentence should say that both did and that the difference lies elsewhere (release unions, query keywords/URL, crawl dates).

## 2. Cui and Lee (2020) — CoAID

**Metadata (arXiv record).** Limeng Cui, Dongwon Lee. "CoAID: COVID-19 Healthcare Misinformation Dataset." arXiv:2006.00885 (v1 22 May 2020; v3 3 Nov 2020). DOI 10.48550/arXiv.2006.00885. No journal version listed. Quotations from the arXiv text (ar5iv rendering).

**Draft claims and support.**
- L21 and L69–71: "Fake items are claims judged false by fact-checkers, while Real items come from official health agencies and established news sources." — **Supported with one wording correction.** Fake: "To collect fake news, we retrieved the URLs from several fact-checking websites, including:" LeadStories, PolitiFact, FactCheck.org, CheckYourFact, AFP Fact Check, Health Feedback (Section 4.1). Real: "we have crawled from 9 reliable media outlets that have been cross-checked as reliable, including:" Healthline, ScienceDaily, NIH, Medical News Today, Mayo Clinic, Cleveland Clinic, WebMD, WHO, CDC. These are health agencies and health-information publishers, not general news organizations; "established news sources" should read "reliable health-information outlets".
- L226: "If CoAID gathered posts by searching for each item" — **Supported.** "We used Twitter API to fetch user engagement data of both facts and misinformation."; "We used the titles of news articles as the search queries, and specified start and end dates" (Section 4.2). Collection window: "The publishing dates of the collected information range from December 1, 2019 to September 1, 2020." (v3 text; the local release directories extend to 11-01-2020.) See the FakeHealth note above: the conditional should not imply that FakeHealth did otherwise.

## 3. Shu, Mahudeswaran, Wang, Lee and Liu (2020) — FakeNewsNet

**Metadata (OpenAlex record of the journal DOI).** Kai Shu, Deepak Mahudeswaran, Suhang Wang, Dongwon Lee, Huan Liu. "FakeNewsNet: A Data Repository with News Content, Social Context, and Spatiotemporal Information for Studying Fake News on Social Media." *Big Data*, 8(3), 171–188, 2020 (published 2020-06-01). DOI 10.1089/big.2020.0062. PMID 32491943. Preprint arXiv:1809.01286. The publisher page could not be fetched (redirect to SAGE, not retrieved); metadata are from OpenAlex and quotations from the arXiv text (ar5iv rendering), which may differ in wording from the journal version.

**Draft claims and support.**
- L137: "Labels were inherited: PolitiFact from fact-checker verdicts" — **Supported.** "journalists and domain experts review the political news and provide fact-checking evaluation results" (Section 3, News Content).
- L137: "GossipCop from entertainment-site scores" — **Supported for Fake, not for Real.** "GossipCop provides rating scores on the scale of 0 to 10 to classify a news story as the degree from fake to real."; "We collect all the news stories from GossipCop with rating scores less than 5 as the fake news stories." Real items are not score-based: "We consider all the articles from E! Online as real news sources." Suggested wording: "GossipCop items scored below 5 of 10 were labelled fake; real items were those FakeNewsNet took from E! Online."
- **Local-data caveat for L137 and the S13/S17 captions.** The round-3 domain audit finds 0 of 16,817 GossipCop Real rows with an eonline.com `news_url` (etonline.com, a different site, is 3.5%; people.com 9.3% is the largest). The paper's description of the Real source does not match the `news_url` field of the local copy. Captions should say "GossipCop Real items as released by FakeNewsNet (documented as E! Online articles; local URLs point to a broad set of entertainment and news domains)".
- Protocol and S17 caption, tweet collection by title search — **Supported.** "are collected using search API provided by social media platforms such as the Twitter's Advanced Search API"; "The search queries for collecting user engagements are formed from the headlines of news articles" (Section 3, Social Context). No fixed time window is stated; collection was ongoing ("we run the news content crawler and update Tweet collector per day").

## 4. Juul and Ugander (2021) — PNAS

**Metadata (Crossref and Europe PMC).** Jonas L. Juul, Johan Ugander. "Comparing information diffusion mechanisms by matching on cascade size." *Proceedings of the National Academy of Sciences*, 118(46), e2100786118, 2021 (online 2021-11-08; issue 2021-11-16). DOI 10.1073/pnas.2100786118. PMID 34750252; PMCID PMC8609637. The PNAS and PMC pages could not be fetched directly (403 / CAPTCHA); the abstract text was retrieved through the Europe PMC REST API.

**Draft claim and support.**
- L25 (optional bracket): "Juul and Ugander (2021, PNAS) found that many structural differences between false and true cascades shrink once cascade size is matched." — **Supported, and the source is stronger than the draft.** Abstract: "we demonstrate the essentiality of controlling for cascade sizes when studying structural differences"; "the reported structural differences can almost entirely be explained by false-news cascades being larger"; "the mechanisms underlying true- and false-news diffusion being quite similar"; and for contrast, "For videos, images, news, and petitions, structural differences persist when controlling for size." The claim can be kept; "shrink" may be replaced by "largely disappear". It supports the draft's point in 1.2 and 4.3 that the Vosoughi et al. comparison concerns cascade structure conditional on circulation, which is distinct from the zero-inclusive occurrence question here. Recommendation: keep, with the verified entry added.

## Claims not supported as written

1. L137 "GossipCop from entertainment-site scores": the source defines only the Fake class by score; Real is "all the articles from E! Online". Rewrite as above.
2. L21/L71 "established news sources" for CoAID Real: the source lists reliable health-information outlets and agencies (Healthline, ScienceDaily, NIH, Medical News Today, Mayo Clinic, Cleveland Clinic, WebMD, WHO, CDC), not general news organizations.
3. L226 "If CoAID gathered posts by searching for each item": true, but FakeHealth did the same (title, keyword and URL search), so the sentence should not present search-based collection as a difference between the datasets.
4. L21 second half (same outlets publish Real and Fake in FakeHealth): a project finding, not a statement in Dai et al.; cite the publisher-family audit (Table S15) rather than the paper.

## Existing v2 bibliography entries checked against the records

`fakehealth` (v2 references.bib):

```
@inproceedings{fakehealth,
  author = {Dai, Enyan and Sun, Yiwei and Wang, Suhang},
  title = {Ginger Cannot Cure Cancer: Battling Fake Health News with a Comprehensive Data Repository},
  booktitle = {Proceedings of the International {AAAI} Conference on Web and Social Media},
  year = {2020},
  volume = {14},
  number = {1},
  pages = {853--862},
  doi = {10.1609/icwsm.v14i1.7350},
  url = {https://ojs.aaai.org/index.php/ICWSM/article/view/7350}
}
```
Official record: ICWSM 14(1), 853–862, 2020, DOI 10.1609/icwsm.v14i1.7350. Confirm the v2 entry carries volume 14, number 1, pages 853–862 and this DOI.

`coaid` (v2 references.bib):

```
@misc{coaid,
  author = {Cui, Limeng and Lee, Dongwon},
  title = {{CoAID}: {COVID-19} Healthcare Misinformation Dataset},
  year = {2020},
  eprint = {2006.00885},
  archivePrefix = {arXiv},
  note = {arXiv preprint},
  url = {https://arxiv.org/abs/2006.00885}
}
```
Record: arXiv:2006.00885, v3 dated 3 Nov 2020. Consistent.

New entries (FakeNewsNet; Juul and Ugander) are in `references_v3_additions.bib`.
