# v3 draft verification — 2026-10-12

Read-only check of `outputs/manuscript/revision_v3/manuscript_v3_draft.md` (6,253 words, header dated 2026-10-12) against `review_package_v2/manuscript.md`, `review_package_v2/supplementary_material.md` and the saved tables under `outputs/tables/`. Nothing outside `revision_v3/` was changed; no analysis was run. Line numbers refer to the files as they stand today.

## Check 1 — Methods coverage

v2 Methods (manuscript lines 55–195) was split into 160 protocol facts, one per sentence or settings-table row. Each was searched in v3 Methods (lines 45–142), in Supplement S1–S7 prose, Tables S1–S2 and the Table S3–S10 displays, and in the rest of v3. Pattern matches were adjudicated by reading both texts; adjudication notes are in the Status column.

**Count: 160 facts. Both v3 Methods and supplement 73. Supplement only 34. v3 Methods only 14. Supplement plus v3 outside Methods 6. Nowhere 33.**

### Facts appearing nowhere (the finding)

These 33 facts are in v2 Methods but in neither v3 Methods nor the supplement. They are listed with the v2 manuscript line and the supplement section they should move to. Nothing has been moved.

| ID | v2 line | Fact appearing nowhere | Move to |
|---|---:|---|---|
| 2.3f | 88 | Source-title changes or dates alone do not establish material change | S1 |
| 2.4d | 96 | FakeHealth retweets excluded from primary outcome also excluded from the graph | S2 |
| 2.4e | 96 | A connected component is a dependence unit, not a claim of identical content | S2 |
| 2.4i | 98 | Posts-only occurrence (hurdle) distinguished from any-post-or-reply indicator | S2 |
| 2.5k | 118 | C = R_CoAID/R_FakeHealth, equivalently theta = log difference; C<1 means smaller CoAID ratio | S2 |
| 2.6a | 126 | Text cohorts derived from HealthStory items and CoAID Spec A entities | S3 |
| 2.6c | 126 | combined_text joins title and body with a blank line; title-only or body-only item allowed when the other is missing | S3 |
| 2.6e | 126 | Substantive headline with poor body eligible for combined text but excluded from body-only | S3 |
| 2.6m | 130 | Exact generic-template matches could join evaluation groups | S3 |
| 2.6n | 130 | Publisher identity or broad topic alone did not define primary groups | S3 |
| 2.6p | 132 | Rows sorted by stable UID; seed-42 StratifiedGroupKFold candidate partitions | S3 |
| 2.7d | 136 | LR fit seed 42 (random_state) | S4 |
| 2.7p | 142 | Training-only most-frequent and seed-42 stratified dummy baselines | S4 |
| 2.8d | 146 | Reverse-direction results did not modify the opposite source model | S4 |
| 2.8f | 148 | Within comparators remain training-pool-to-holdout models; only evaluation rows matched, not training populations/sizes | S4 |
| 2.8h | 148 | Existing overlap flags and group safeguards retained; target cohorts not redefined by performance | S3 |
| 2.9e | 156 | Title design does not isolate text length (eligibility and configurations differ) | S4 |
| 2.10a | 166 | distilbert/distilroberta-base at immutable revision fb53ab88...; AutoModelForSequenceClassification, 2 labels | S6/S7 |
| 2.10b | 166 | Model card: RoBERTa teacher, DistilBERT procedure; attributions Liu 2019 / Sanh 2019 | S6 / references |
| 2.10e | 168 | FP32 on Apple Silicon MPS in Python 3.12 environment | S6/S7 |
| 2.10f | 168 | Microbatch 2, gradient accumulation 8, effective batch 16 | S6 |
| 2.10i | 168 | AdamW weight decay 0.01 (not bias/LayerNorm), betas (0.9, 0.999), eps 1e-8, grad clip 1.0 | S6 |
| 2.10l | 170 | Candidate LRs 2e-5 and 5e-5; epochs 2 and 3; 3-epoch trajectory validated at epochs 2 and 3 | S6 |
| 2.10o | 170 | No early stopping, mixed precision, oversampling, or target threshold tuning | S6 |
| 2.10p | 172 | Within final models on source training pool; transfer models from pretrained checkpoint on all eligible source rows incl. former holdout | S6 |
| 2.10t | 174 | CoAID target token sequences identical at both caps; only FH source prefix changes | S6 |
| 2.11a | 178 | Headerless NEWS table read via documented rumor-ID, label, statement, sentiment columns | S6 |
| 2.11c | 178 | IDs, sentiment, stance, dates, engagement, source metadata excluded as predictors | S6 |
| 2.11e | 180 | Quality thresholds: two alphabetic analyzer tokens and 90% Latin letters | S6 |
| 2.11f | 180 | English designation and script check do not establish complete language identification | S6 |
| 2.12a | 188 | Exact versions: Python 3.14.2, NumPy 2.5.1, pandas 3.0.5, scikit-learn 1.9.0, SciPy 1.18.0, statsmodels 0.15.0 | S7 |
| 2.12b | 188 | Transformer env: Python 3.12.7, PyTorch 2.14.0, Transformers 5.17.0; pins and checkpoint revision retained | S7 |
| 2.12c | 188 | Seed 42 primary; seeds 123 and 2026 only for the stability check; MPS seeded without bitwise reproducibility claim | S7 |

Grouped: the transformer training recipe is the largest gap (10 facts: checkpoint revision hash and model class, optimizer settings, batch/accumulation, candidate grid, no-early-stopping rule, within-vs-transfer training populations, token-sequence invariance, FP32/MPS). Exact software versions and the seed policy (3 facts) left v3 2.11 and were never listed in S7, which only names the files holding them. The COVID-Rumor quality thresholds (two alphabetic tokens, 90% Latin letters) and the language caveat (3 facts) are absent from S6, which describes URL stripping without the thresholds. Text-cohort construction details (7 facts: blank-line join, title/body fallbacks, generic-template grouping, publisher/topic not grouping, UID sort and seed-42 splitter, Spec A origin, headline-with-poor-body rule) are absent from S3. Engagement dependence details (4 facts) are absent from S2.

### Facts now in the supplement only (acceptable moves, listed for completeness)

34 facts moved out of v3 Methods and are covered by S1–S7 or Tables S1–S10, including the LR/SVM solver settings and grids (S4, Table S1), tie rules (S4), May reference level (S2), linear quantiles (S2), Hessian/convergence checks (S2), engagement sample sizes 5,337 and 1,690 (Table S3; these appear nowhere in v3 prose), publisher normalization (S5), checksum limitation (S7). The engagement sample sizes are the only numbers in this group that a reader would expect in the main text.

### Full fact table

| ID | v2 line | Protocol fact | v3 Methods | Supplement | Status | Target if missing |
|---|---:|---|:-:|:-:|---|---|
| 2.0a | 60 | Engagement counts never enter text models; no language-to-sharing mediation | yes | no | v3 Methods only |  |
| 2.1a | 68 | Observational cross-dataset design; period/context not independently varied, no causal COVID estimate | no | yes | supplement + v3 elsewhere |  |
| 2.1b | 72 | Spec A and natural combined-text LR are primary; alternatives are robustness | yes | yes | both |  |
| 2.1c | 72 | Final validated cohorts and split assignments used, not mixed stages | no | yes | supplement only |  |
| 2.2a | 76 | Four official CoAID release directories 05/07/09/11-01-2020 | yes | no | v3 Methods only |  |
| 2.2b | 76 | Fake/Real labels from original News filenames | yes | yes | both |  |
| 2.2c | 76 | Content→post via content ID; →reply via news ID, within release and file category; release-local IDs not globally unique | yes | yes | both |  |
| 2.2d | 76 | CoAID A engagement sample 5,337 entities (922 Fake, 4,415 Real) | no | yes | supplement only |  |
| 2.2e | 78 | FakeHealth review score <3 Fake, >=3 Real; HealthRelease not used | yes | no | v3 Methods only |  |
| 2.2f | 78 | FakeHealth engagement sample 1,690 items (472 Fake, 1,218 Real) | no | yes | supplement only |  |
| 2.2g | 78 | FakeHealth outcome uses mapping tweet IDs; retweet mapping excluded (no CoAID equivalent) | yes | yes | both |  |
| 2.2h | 80 | COVID-Rumor NEWS only; T→Real, F→Fake, U excluded; Twitter subset excluded | yes | yes | both |  |
| 2.2i | 80 | Each item contributes one statement; website records used only for overlap screening | no | yes | supplement only |  |
| 2.3a | 84 | Repeated release membership alone insufficient for identity; identity via audited URL/resource mapping and lineage | yes | yes | both |  |
| 2.3b | 84 | Shared domains, generic titles, branding, boilerplate, social overlap never merge | yes | yes | both |  |
| 2.3c | 86 | Spec A: collapse high-confidence same/minor-update captures; split material updates; keep insufficient-evidence captures separate | yes | yes | both |  |
| 2.3d | 86 | Analytical unit may be a content version or separately retained capture with explicit uncertainty flags | no | yes | supplement only |  |
| 2.3e | 88 | Version decisions combine body, title, abstract and metadata evidence; material change needs corroboration | no | yes | supplement only |  |
| 2.3f | 88 | Source-title changes or dates alone do not establish material change | no | no | nowhere | S1 |
| 2.3g | 88 | Complete-link pair checks for approved material-version partitions | yes | yes | both |  |
| 2.3h | 90 | Spec B assumes stability for insufficient-evidence captures of same resource; Spec C excludes uncertain-version and unresolved-identity captures | yes | yes | both |  |
| 2.3i | 90 | Stable entity IDs, source UIDs, release/file/row refs, member captures, flags, exclusions preserved | yes | yes | both |  |
| 2.3j | 90 | Social counts reconstructed from unique ID unions within an entity | yes | yes | both |  |
| 2.3k | 90 | Shared social IDs exposed by splitting are retained for dependence modeling, not deleted | yes | no | v3 Methods only |  |
| 2.4a | 94 | Outcome: unique observed posts per entity incl. zeros; repeated IDs counted once | yes | yes | both |  |
| 2.4b | 94 | No exposure denominator, cascades, network topology; deletions/private/uncaptured outside measurement | yes | no | v3 Methods only |  |
| 2.4c | 96 | Overlap graph per dataset and spec; edge = shared counted post or reply ID; components incl. singletons are clusters | yes | yes | both |  |
| 2.4d | 96 | FakeHealth retweets excluded from primary outcome also excluded from the graph | no | no | nowhere | S2 |
| 2.4e | 96 | A connected component is a dependence unit, not a claim of identical content | no | no | nowhere | S2 |
| 2.4f | 98 | Dominant-component exclusion: remaining entities define sensitivity population and standardization distribution | yes | yes | both |  |
| 2.4g | 98 | Common publishers/resources without shared IDs could add dependence not captured by graph | no | yes | supplement + v3 elsewhere |  |
| 2.4h | 98 | Reply counts and occurrence summaries are secondary diagnostics | yes | yes | both |  |
| 2.4i | 98 | Posts-only occurrence (hurdle) distinguished from any-post-or-reply indicator | no | no | nowhere | S2 |
| 2.5a | 102 | CoAID hurdle: logistic for any post + zero-truncated NB2 for positive counts | yes | yes | both |  |
| 2.5b | 102 | Both components include Fake indicator and first-recorded-release fixed effects, May as reference | no | yes | supplement only |  |
| 2.5c | 102 | First-recorded release = capture provenance, not publication/exposure time | yes | no | v3 Methods only |  |
| 2.5d | 102 | FakeHealth NB2 with Fake indicator; no exposure offset or text covariate | yes | no | v3 Methods only — v3 L91 states NB2 with Fake indicator and no exposure offset; only the 'no text covariate' clause is absent |  |
| 2.5e | 104 | Hurdle retained if AIC improves over NB2 by >2 or NB2 unstable; comparison on same full outcome counting all hurdle parameters | yes | no | v3 Methods only |  |
| 2.5f | 104 | Fits checked for convergence, gradients, Hessian behavior, parameter bounds | no | yes | supplement only |  |
| 2.5g | 104 | Model-family selection fixed during resampling | yes | yes | both |  |
| 2.5h | 106 | Standardized expected counts compared, not coefficients; R_d formula | yes | yes | both |  |
| 2.5i | 118 | Row-level products averaged before ratio; not positive-count coefficient nor product of averaged components | yes | yes | both |  |
| 2.5j | 118 | Each dataset retains its own covariate distribution incl. zero-count entities | yes | yes | both |  |
| 2.5k | 118 | C = R_CoAID/R_FakeHealth, equivalently theta = log difference; C<1 means smaller CoAID ratio | no | no | nowhere | S2 |
| 2.5l | 118 | Contrast does not require equivalent coefficients or a shared reference population | yes | yes | both |  |
| 2.5m | 120 | Unstratified pairs bootstrap of overlap components; K components with replacement; all entities of a selected component | yes | yes | both |  |
| 2.5n | 120 | Repeated component selections repeat their rows | no | yes | supplement only |  |
| 2.5o | 120 | Each draw refits and re-standardizes over its own sampled covariate distribution | yes | yes | both |  |
| 2.5p | 120 | Target 1,000 successful finite post-ratio refits per dataset/spec; failures recorded and excluded; retry on same sampled data; seed 42 | yes | yes | both |  |
| 2.5q | 122 | Cross-context: all 1,000×1,000 ordered pairs, equally weighted; 1,000,000 values not independent | yes | yes | both |  |
| 2.5r | 122 | Linear empirical quantiles for 95% percentile intervals; point estimate is ratio of fitted standardized ratios | no | yes | supplement only |  |
| 2.5s | 122 | Assumes independence across datasets; excludes uncertainty from missing links, model-family selection, entity resolution | yes | yes | both |  |
| 2.6a | 126 | Text cohorts derived from HealthStory items and CoAID Spec A entities | no | no | nowhere | S3 |
| 2.6b | 126 | Normalization: whitespace/null placeholders; retain case, punctuation, URLs | yes | yes | both |  |
| 2.6c | 126 | combined_text joins title and body with a blank line; title-only or body-only item allowed when the other is missing | no | no | nowhere | S3 |
| 2.6d | 126 | Empty/artifact-only records ineligible; label-blind quality audit excludes artifact-dominated content and reviews title provenance for label disclosure | yes | yes | both |  |
| 2.6e | 126 | Substantive headline with poor body eligible for combined text but excluded from body-only | no | no | nowhere | S3 |
| 2.6f | 128 | Final cohorts 1,647 FH (459/1,188), 3,577 CoAID (825/2,752) | yes | yes | both |  |
| 2.6g | 128 | Fake=1, Real=0 | yes | no | v3 Methods only |  |
| 2.6h | 128 | Predictors exclude IDs, ratings, metadata, flags, engagement, lineage, splits | yes | yes | both |  |
| 2.6i | 128 | Quality audit retains original strings and exclusion reasons | no | yes | supplement only |  |
| 2.6j | 128 | Masked representation: label-blind dictionary from prepared metadata incl. evaluation metadata; transductive sensitivity | yes | yes | both |  |
| 2.6k | 130 | Leakage groups = connected components of resource identity, lineage, exact/very-high-similarity title/body | yes | yes | both |  |
| 2.6l | 130 | CoAID versions of a resource grouped together; unresolved identity contexts grouped conservatively | yes | yes | both |  |
| 2.6m | 130 | Exact generic-template matches could join evaluation groups | no | no | nowhere | S3 |
| 2.6n | 130 | Publisher identity or broad topic alone did not define primary groups | no | no | nowhere | S3 |
| 2.6o | 130 | Label-blind cross-dataset lexical overlap screening; no flagged overlap affected final cohorts | no | yes | supplement only |  |
| 2.6p | 132 | Rows sorted by stable UID; seed-42 StratifiedGroupKFold candidate partitions | no | no | nowhere — v3 says 'grouped stratified split' without StratifiedGroupKFold, seed or UID sort; supplement S3 describes fold choice but not the splitter/seed; S5's StratifiedGroupKFold refers to the publisher design | S3 |
| 2.6q | 132 | Holdout fold chosen by closeness to 20% of total and each class (counts, not performance) | yes | yes | both |  |
| 2.6r | 132 | Five grouped CV folds within training pool; quality filtering preserves assignments | yes | yes | both |  |
| 2.6s | 132 | Training/holdout sizes 1,317/330 and 2,875/702 | yes | yes | both |  |
| 2.6t | 132 | Groups never cross training/validation/test; holdouts never enter tuning | yes | no | v3 Methods only |  |
| 2.6u | 132 | CoAID short excerpts vs FakeHealth long bodies are distinct information budgets | no | yes | supplement + v3 elsewhere |  |
| 2.7a | 136 | Vectorizer: lowercase, Unicode word tokens >=2 chars, sublinear TF, smoothed IDF, L2 norm, float64, no accent stripping, max_df 1.0 | no | yes | supplement only |  |
| 2.7b | 136 | LR grid: 24 configs = n-gram (1,1)/(1,2) × min_df 2/5 × C 0.1/1/10 × None/balanced | yes | yes | both |  |
| 2.7c | 136 | LR: L2 penalty, liblinear, tol 1e-4, max_iter 5,000, seed 42 | no | yes | supplement only |  |
| 2.7d | 136 | LR fit seed 42 (random_state) | no | no | nowhere — 'seed 42' in v3 Methods refers to the bootstrap; LR random_state is not stated in v3 or S4 | S4 |
| 2.7e | 138 | Every CV fit learns vocabulary and IDF from its training fold only | yes | yes | both |  |
| 2.7f | 138 | Selection maximizes unweighted mean macro F1 across five saved folds | yes | yes | both |  |
| 2.7g | 138 | Tie rule: mean within 0.001 tied; SD within 0.001 tied; then unigram, smaller C, None weighting, larger min_df, config ID | no | yes | supplement only |  |
| 2.7h | 138 | Nonconverged configurations ineligible | no | yes | supplement only |  |
| 2.7i | 138 | Selected natural settings locked before holdout prediction; final within fits use only training pool | yes | yes | both |  |
| 2.7j | 138 | Masked-text fits use the natural configuration without retuning | no | yes | supplement only |  |
| 2.7k | 140 | SVM: same cohorts/splits/TF-IDF; 32 configs adding C=0.01; LinearSVC L2, squared hinge, dual=auto, tol 1e-4, max_iter 20,000, seed 42 | yes | yes | both |  |
| 2.7l | 140 | SVM tie rule matches LR through unweighted preference, then config ID, without min_df preference | yes | yes | both |  |
| 2.7m | 142 | LR Fake at p>=0.5; SVM Fake at decision >=0; ROC/AP from probabilities/decision scores | yes | yes | both |  |
| 2.7n | 142 | Metrics: macro F1, class-specific, balanced accuracy, confusion counts, ROC-AUC, non-interpolated AP | yes | yes | both |  |
| 2.7o | 142 | SVM scores not treated as calibrated probabilities | yes | no | v3 Methods only |  |
| 2.7p | 142 | Training-only most-frequent and seed-42 stratified dummy baselines | no | no | nowhere | S4 |
| 2.8a | 146 | Transfer both directions; source-selected hyperparameters; refit on all eligible source rows incl. former holdout | yes | yes | both |  |
| 2.8b | 146 | Vocabulary/IDF from source text only; target transformed without fitting, calibration or threshold tuning | yes | no | v3 Methods only |  |
| 2.8c | 146 | Configurations locked before evaluation; predictions frozen before target labels joined | yes | yes | both |  |
| 2.8d | 146 | Reverse-direction results did not modify the opposite source model | no | no | nowhere — 'reverse' hit was 'reversed direction' in Results 3.3 | S4 |
| 2.8e | 148 | All-eligible and saved-holdout target scopes reported; holdout primary because identical target UIDs | yes | yes | both |  |
| 2.8f | 148 | Within comparators remain training-pool-to-holdout models; only evaluation rows matched, not training populations/sizes | no | no | nowhere | S4 |
| 2.8g | 148 | Natural and masked LR transfer use corresponding within-target representations | no | yes | supplement only |  |
| 2.8h | 148 | Existing overlap flags and group safeguards retained; target cohorts not redefined by performance | no | no | nowhere | S3 |
| 2.8i | 150 | Lexical alignment uses frozen within-context natural LR models, not transfer fits; 4,174 shared unigrams, Fake-positive | yes | yes | both |  |
| 2.8j | 150 | FakeHealth bigrams excluded from intersection but remain in its fitted model | no | yes | supplement only |  |
| 2.8k | 150 | Sign reversal = negative coefficient product; exact zeros accounted separately | yes | yes | both |  |
| 2.8l | 150 | Prespecified subsets: all; above either model's median |coef|; top 1,000/500/100 by max |coef|, lexical ties | yes | yes | both |  |
| 2.8m | 152 | Coefficients are conditional, not rescaled, not causal; alignment descriptive; no significance test | yes | yes | both |  |
| 2.9a | 156 | Title-only LR uses existing title_only_modeling_eligible flags and unchanged resource groups | yes | yes | both |  |
| 2.9b | 156 | Title-only tuned independently per dataset with same LR grid and tie policy; within fits on training pool; transfer refits on all eligible titles | yes | yes | both |  |
| 2.9c | 156 | Combined-vs-title transfer compared on identical title-eligible target rows | yes | yes | both |  |
| 2.9d | 156 | Masked-title sensitivities use audited masked field with locked natural-title hyperparameters | no | yes | supplement only |  |
| 2.9e | 156 | Title design does not isolate text length (eligibility and configurations differ) | no | no | nowhere | S4 |
| 2.9f | 158 | Publisher families from prepared names, hosts, specific URLs, audited aliases; no labels or article text | yes | yes | both |  |
| 2.9g | 158 | Hosted tenants and platform accounts distinct; unresolved sources not grouped as one publisher | yes | yes | both |  |
| 2.9h | 158 | Components combine publisher-family and resource/leakage barriers; 1,647 items, 49 families, 49 components | yes | no | v3 Methods only |  |
| 2.9i | 158 | Publisher families are source proxies, not corporate-ownership classification | no | yes | supplement only |  |
| 2.9j | 160 | Outer: 3-fold StratifiedGroupKFold, shuffled, seed 42; nested 3-fold inner tuning (2 permitted if infeasible; all used 3) | yes | yes | both — v3 L130 + S5 L70 (shuffled StratifiedGroupKFold, seed 42, three inner folds) |  |
| 2.9k | 160 | LR grid and tie rule unchanged; configuration locked before outer prediction; refit only on outer-training data | no | yes | supplement only |  |
| 2.9l | 160 | Source-masked publisher fits reuse selected settings | no | yes | supplement only |  |
| 2.9m | 160 | Outer-fold mean/SD/range and pooled summaries distinguished; ROC-AUC summarized within folds | no | yes | supplement only |  |
| 2.9n | 160 | Comparison with resource-grouped CV baseline is descriptive and unpaired | yes | yes | both |  |
| 2.9o | 162 | No CoAID publisher-held-out claim: class-exclusive families, sparse mixed support, unresolved rows mostly Fake | yes | yes | both |  |
| 2.10a | 166 | distilbert/distilroberta-base at immutable revision fb53ab88...; AutoModelForSequenceClassification, 2 labels | no | no | nowhere | S6/S7 |
| 2.10b | 166 | Model card: RoBERTa teacher, DistilBERT procedure; attributions Liu 2019 / Sanh 2019 | no | no | nowhere — 'RoBERTa' hit was inside 'DistilRoBERTa'; teacher/procedure attributions appear only in the header comment as citations to carry | S6 / references |
| 2.10c | 166 | Pair packing: full title kept, four special tokens reserved, body prefix fills budget; dynamic padding per microbatch | yes | yes | both |  |
| 2.10d | 166 | Max length 512 primary, 256 sensitivity; FakeHealth substantially truncated, CoAID not | yes | yes | both |  |
| 2.10e | 168 | FP32 on Apple Silicon MPS in Python 3.12 environment | no | no | nowhere | S6/S7 |
| 2.10f | 168 | Microbatch 2, gradient accumulation 8, effective batch 16 | no | no | nowhere | S6 |
| 2.10g | 168 | Weighted cross-entropy with class weight N_train/(2 N_train,c) from source-training labels only | no | yes | supplement only — v3 'class weight' hit was 'class weighting' in the LR grid; S6 states source-only recomputation but not the N/(2N_c) formula |  |
| 2.10h | 168 | Gradients normalized by sum of training class weights per effective batch incl. partial final batch | no | yes | supplement + v3 elsewhere |  |
| 2.10i | 168 | AdamW weight decay 0.01 (not bias/LayerNorm), betas (0.9, 0.999), eps 1e-8, grad clip 1.0 | no | no | nowhere | S6 |
| 2.10j | 168 | Constant LR schedule, zero warmup | no | yes | supplement only |  |
| 2.10k | 170 | Transformer selection uses same five grouped folds and holdouts as classical | yes | no | v3 Methods only |  |
| 2.10l | 170 | Candidate LRs 2e-5 and 5e-5; epochs 2 and 3; 3-epoch trajectory validated at epochs 2 and 3 | no | no | nowhere | S6 |
| 2.10m | 170 | Transformer tie rule: within 0.001 of best mean macro F1 prefer lower SD, fewer epochs, lower LR | no | yes | supplement only — v3 'tie' hit was the classical tie rule in 2.7; transformer tie policy only in S4 L49 |  |
| 2.10n | 170 | Both datasets selected 2e-5 and 3 epochs; locked before holdout/transfer evaluation | no | yes | supplement only |  |
| 2.10o | 170 | No early stopping, mixed precision, oversampling, or target threshold tuning | no | no | nowhere | S6 |
| 2.10p | 172 | Within final models on source training pool; transfer models from pretrained checkpoint on all eligible source rows incl. former holdout | no | no | nowhere | S6 |
| 2.10q | 172 | Target transform-only; probability >=0.5 = Fake; all-eligible and matched holdout scopes reported | no | yes | supplement only — v3 'Fake at probability' hit was the LR rule in 2.7; transformer threshold not stated in v3; scopes in Table S8 |  |
| 2.10r | 172 | 256-token sensitivity reuses each source's selected LR and epochs without retuning | yes | yes | both |  |
| 2.10s | 174 | Follow-up: FH→CoAID, 256 vs 512, seeds 42/123/2026; seed 42 reuses saved evaluations; other seeds refit full FH source | yes | yes | both |  |
| 2.10t | 174 | CoAID target token sequences identical at both caps; only FH source prefix changes | no | no | nowhere | S6 |
| 2.10u | 174 | Paired differences, means, SDs, ranges, prediction agreement; stability check not new search | yes | yes | both |  |
| 2.11a | 178 | Headerless NEWS table read via documented rumor-ID, label, statement, sentiment columns | no | no | nowhere | S6 |
| 2.11b | 178 | Binary T/F only; U and U(Twitter) excluded; website/repost records for screening only; no Twitter files/repost text | yes | yes | both |  |
| 2.11c | 178 | IDs, sentiment, stance, dates, engagement, source metadata excluded as predictors | no | no | nowhere — 'stance' hit was inside 'distance' | S6 |
| 2.11d | 180 | Quality rules fixed label-blind: normalization; exclude missing/artifact-only, <2 alphabetic tokens after URL stripping, <90% Latin letters | no | yes | supplement only |  |
| 2.11e | 180 | Quality thresholds: two alphabetic analyzer tokens and 90% Latin letters | no | no | nowhere | S6 |
| 2.11f | 180 | English designation and script check do not establish complete language identification | no | no | nowhere | S6 |
| 2.11g | 180 | 3,700 eligible; exact overlap exclusions leave 3,667 (3,015 Fake, 652 Real) | yes | yes | both |  |
| 2.11h | 182 | Overlap screening against source title/body/combined incl. ineligible source records; exact text or exact specific-resource URL excluded | yes | yes | both |  |
| 2.11i | 182 | URL checks: offline normalization, incl. website and review URLs; homepage-only matches not counted | no | yes | supplement only |  |
| 2.11j | 182 | High-similarity matches flagged not excluded; all-eligible and clean-without-high-similarity are sensitivity scopes | no | yes | supplement + v3 elsewhere |  |
| 2.11k | 182 | Missing website files do not exclude; redirects unresolved offline | no | yes | supplement only |  |
| 2.11l | 184 | Four frozen full-source LR/SVM reused without refitting; target fingerprinted before scoring; predictions frozen before labels | yes | yes | both |  |
| 2.11m | 184 | Binary eligibility exposes membership only before prediction; thresholds 0.5 and 0; identical clean rows | yes | yes | both — v3 L135 'applied unchanged' covers fixed thresholds and identical rows |  |
| 2.11n | 184 | Representation diagnostics: target unigram occurrences covered by source vocabulary; zero-feature rows; no bigram double counting | no | yes | supplement + v3 elsewhere |  |
| 2.11o | 184 | External evaluations descriptive; no winner test or target adaptation | yes | yes | both |  |
| 2.12a | 188 | Exact versions: Python 3.14.2, NumPy 2.5.1, pandas 3.0.5, scikit-learn 1.9.0, SciPy 1.18.0, statsmodels 0.15.0 | no | no | nowhere | S7 |
| 2.12b | 188 | Transformer env: Python 3.12.7, PyTorch 2.14.0, Transformers 5.17.0; pins and checkpoint revision retained | no | no | nowhere | S7 |
| 2.12c | 188 | Seed 42 primary; seeds 123 and 2026 only for the stability check; MPS seeded without bitwise reproducibility claim | no | no | nowhere | S7 |
| 2.12d | 190 | Raw data unmodified; processed records, UIDs, lineage, manifests, configs, scripts, provenance retained | yes | yes | both |  |
| 2.12e | 190 | Checksum comparisons for reuse where fingerprints recorded; original within-engagement run lacks historical output hashes | no | yes | supplement only |  |
| 2.12f | 190 | Cross-context stage fingerprinted saved outputs and checked completion, convergence, identities, interval endpoints, draw counts | no | yes | supplement only |  |
| 2.12g | 192 | Engagement uncertainty = overlap-cluster percentile intervals; fold/seed SDs not CIs; text comparisons descriptive, no significance tests | yes | yes | both — v3 L141 + supplement 'Reading the aggregate tables' |  |

## Check 2 — Number fidelity

Every numeric value in v3 prose, captions and Table 1 was matched to a source. Sources: v2 = `review_package_v2/manuscript.md`; T = a saved table under `outputs/tables/`; F = `outputs/reports/engagement_comparability_feasibility.md` (2026-10-09); A = `outputs/reports/audit_2026-10-12.md`; X = `outputs/extension_v1/` (not in the permitted source list).

| v3 line(s) | Value(s) | Source | Saved value | Status |
|---|---|---|---|---|
| 13, 147, 157 | 0.253, 0.840, 0.301, 0.154–0.645 | v2; T cross_context_engagement_contrast.csv | 0.2528, 0.8403, 0.3009, 0.1543–0.6452 | pass |
| 147 | CoAID 0.132–0.522; FakeHealth 0.714–0.997 | v2; T version_aware_engagement_results.csv | 0.1322–0.5220; 0.7141–0.9967 | pass |
| 149, 211 | B 0.286, C 0.343 | T cross_context_engagement_contrast.csv | 0.2857, 0.3433 | pass |
| 149, 212, 234 | 0.489 (0.310–0.710) | v2 Table 1; T cross_context_sensitivity_results.csv | 0.4895, 0.3103, 0.7101 | pass |
| 151 | 199.9, 237.9, 8.8, 34.9 | T version_aware_engagement_results.csv; T version_aware_hurdle_components.csv | 199.877, 237.874, 8.828, 34.920 | pass (placeholder, optional) |
| 153 | 11.7%, 61.7%, 51.4, 39.5, 1.30 (0.73–2.55) | T version_aware_hurdle_components.csv | 0.11720, 0.61678, 51.400, 39.528, 1.3004 [0.7285, 2.5464] | pass |
| 153 | occurrence ratio 0.190 (0.164–0.218) | X coaid_hurdle_components.csv (derived from saved draws; not in outputs/tables) | 0.19001 [0.1639, 0.2182] | **unsourced within the listed set** |
| 153 | "Nearly all FakeHealth items had observed posts" | T version_aware_engagement_results.csv zero_proportion | 0.0041 | pass |
| 19 | 6.2 and 6.4 percentage points | v2 line 13 | same | pass |
| 25 | 2006 to 2017 | v2 line 17 | 2006–2017 | pass |
| 69 | releases 05/07/09/11-01-2020 | v2 line 76 | same | pass |
| 79 | 110 source records | outputs/reports (CoAID entity-resolution report, "110-member component") | 110 | pass (not in v2; report-sourced) |
| 87 | 5,056 clusters, largest 187; 1,662, largest 4 | T version_aware_social_overlap_graph_summary.csv | 5056 / 187; 1662 / 4 | pass (not in v2) |
| 111 | 1,647 (459/1,188); 3,577 (825/2,752) | v2 line 128 | same | pass |
| 113 | 1,317/330; 2,875/702; ~20% | v2 line 132 | same | pass |
| 117 | 24 LR, 32 SVM configurations | v2 lines 136, 140 | same | pass |
| 123, 173, 183 | 4,174 shared words | v2 line 150 | same | pass |
| 128 | 49 families | v2 line 158 | 49 | pass |
| 129 | 71.22% FakeHealth truncated at 512; seeds 42/123/2026 | T transformer_truncation_audit.csv; v2 | 71.2204 | pass (percentage not in v2 prose; table-sourced) |
| 135, 187, 195 | 3,700; 33; 3,667; 3,015; 652; 82.22% | v2 line 180; T covid_rumor_external_cohort_summary.csv | same | pass |
| 137 | PolitiFact 429/596; GossipCop 5,315/16,799 | X fakenewsnet_external_cohort_summary.csv | same | **unsourced within the listed set** |
| 141 | Python 3.14, 3.12 | v2 line 188 | 3.14.2, 3.12.7 | pass (less precise) |
| 163 | LR 0.926/0.983; FH 0.675/0.770; SVM 0.913/0.980, 0.670/0.770 | T tfidf_logistic_holdout_results.csv; tfidf_linear_svm_holdout_results.csv | 0.9260/0.9827; 0.6752/0.7701; 0.9128/0.9795; 0.6697/0.7695 | pass |
| 163, 208 | masking 0.926→0.928; 0.675→0.673 | T tfidf_logistic_holdout_results.csv source_masked | 0.9278; 0.6726 | pass |
| 165, 209 | 0.620 (SD 0.021; 0.603–0.643); mean ROC-AUC 0.700; 0.660 | T fakehealth_publisher_heldout_summary.csv; v2 line 285 | 0.6196, 0.0210, 0.6030–0.6432, 0.6995; 0.6598 | pass; "mean ROC-AUC" is the within-fold mean, correctly labelled |
| 169 | 0.429/0.571 (drop 0.497); 0.451/0.619 (drop 0.224); SVM 0.425/0.583; 0.449/0.631 | T manuscript_key_results.csv R009–R018 | 0.4290/0.5705; 0.4507/0.6191; 0.4245/0.5827; 0.4493/0.6313; drops 0.4970, 0.2244 | pass (same_target_holdout scope, matches v3 wording) |
| 173, 183 | r −0.015; ρ −0.003; 48.4%; top-100 48.0% | T text_coefficient_alignment_summary.csv | −0.01477; −0.00343; 48.39; 48.0 | pass |
| 175 | "Between 9 and 14 of each dataset's 25 ... leaned the other way" | X top_fake_real_terms.csv | 11, 10, 14, 9 | **unsourced within the listed set** |
| 179 | N = 702; N = 330 | v2 | same | pass |
| 187 | ROC-AUC 0.700 vs 0.522; 0.682 vs 0.525; macro F1 0.558 vs 0.497; 0.531 vs 0.497 | T covid_rumor_external_transfer_results.csv | 0.6995/0.5219; 0.6816/0.5248; 0.5580/0.4969; 0.5314/0.4973 | pass |
| 189 | "about 90–94% of items Fake"; balanced accuracy 0.50–0.55; coverage 89.5%, 81.9%; no item lacked features | T covid_rumor_external_transfer_results.csv; covid_rumor_external_vocabulary_diagnostics.csv | predicted prevalence 0.8961, 0.9193, 0.9275, 0.9438; bal. acc. 0.5021–0.5529; 89.47, 81.87; zero rows 0 | **rounding flag**: lower bound is 89.6%, displayed as 90 |
| 191 | 0.608 vs 0.503; 0.532 vs 0.477; FakeHealth title within 0.513 | X fakenewsnet_external_transfer_results.csv; T tfidf_logistic_title_holdout_results.csv | 0.6084/0.5034; 0.5323/0.4770; 0.5127 | **first two unsourced within the listed set**; 0.513 pass |
| 199, 213 | 0.939, 0.621; transfer 0.442, 0.437 | T manuscript_key_results.csv R030–R035 | 0.9387, 0.6212, 0.4415, 0.4371 | pass (same_target_holdout) |
| 199 | seed range 0.287 to 0.510 | T distilroberta_budget_seed_summary.csv | same_target_holdout 512 macro F1 min 0.2867, max 0.5101 (all_eligible would be 0.254–0.439) | **scope unstated**: value is the same_target_holdout scope; v3 does not say so |
| 199, 214 | +0.101 (SD 0.079); ranking better in 1 of 3 | T manuscript_key_results.csv R036–R037 | 0.1011, 0.0790, positive seeds 1 | pass |
| 220 | Cramér's V ≈ 0.97 vs ≈ 0.35 | T text_source_label_association.csv | all rows 0.969/0.353; primary-eligible 0.971/0.352 | pass at displayed precision; **scope unstated** (both scopes round the same); placeholder already asks |
| 232 | 84% | v2 (0.840) | 0.8403 | pass |
| 258 | 82% Fake prevalence | v2 | 82.22 | pass |
| 262 | 11 (5/6), 97 (28/69); 66.81%; two families | F | same | pass |

**Count: 112 numeric values checked. 8 values in 4 statements have no source in v2, outputs/tables, the feasibility report or the audit (they come from outputs/extension_v1, dated 2026-10-12): the occurrence-probability ratio 0.190 (0.164–0.218), the FakeNewsNet class counts, the FakeNewsNet ROC-AUCs, and the 9–14 reversal counts. 1 rounding flag (89.6% displayed as 90). 2 scope-unstated values (seed range; Cramér's V). 0 values with a digit changed at displayed precision. 0 scope substitutions (no holdout/all-eligible, mean/pooled, within-fold/pooled or title/combined swap found).** The header comment at line 4 ("No numerical value differs from the manuscript or the saved project record") is therefore true for outputs/tables but not for the four extension-derived statements, which rest on outputs/extension_v1 only.

## Check 3 — Claim boundaries

| Boundary | Stated in v3? | Quote (line) |
|---|---|---|
| Not a causal COVID effect | yes, 3 places | L43: "Because COVID timing, label construction, source populations and collection procedures all differ between the two datasets, this design cannot attribute differences to the pandemic itself." Also L13 ("differences between dataset contexts rather than effects of the pandemic") and L252. |
| Counts are not cascades, exposure or reach | yes, 3 places | L85: "These are counts of captured posts. They do not measure exposure, complete cascades or network structure, and deleted, private or uncaptured activity is outside the measurement." Also L43 and L230. |
| Text models do not explain sharing | yes, 2 places | L43: "the text models predict labels; they do not model why items were shared." Also L246: "the text models here say nothing about why items were shared." |
| Labels are different constructs | yes, 3 places | L75: "The analytical labels share an orientation (Fake = 1) but not a construct: FakeHealth labels reporting quality, while CoAID and COVID-Rumor labels derive from veracity judgments." Also L13 and L220. |
| Dominant-component exclusion is not a replacement primary | **weakened** | L149: "Its magnitude depended on one cluster: excluding CoAID's largest group of items linked by shared posts moved the ratio-of-ratios to 0.489 (0.310–0.710), weaker but in the same direction." Table 1 (L206) lists it under "Robustness and sensitivity checks"; L234: "The direction is robust; the size depends on how a small number of heavily shared items is treated." v3 never states that the exclusion changes the reference population or that it is not a replacement estimate. v2 Table 1 (L273) said "Exclusion changes the reference population; dependence remains incompletely captured", and Supplement S2 keeps the distinction. One clause should be restored. |

**Count: 5 boundaries checked; 4 fully stated; 1 (dominant-component exclusion) stated as a sensitivity but without the reference-population / not-a-replacement wording.**

### Square-bracketed placeholders in v3

| Line | Placeholder | What it needs |
|---:|---|---|
| 5 | header note | Delete the HTML comment before rendering; correct line 4 (see Check 2). |
| 21 | Verify criteria wording against Dai et al. (2020) | Check FakeHealth review-criteria description against the paper; not verifiable from project files. |
| 21 | Verify against Cui and Lee (2020) | Check CoAID label-construction description against the paper. |
| 25 | Optional Juul and Ugander (2021) | Verify and add a bibliography entry, or delete. |
| 79 | Over-merge example placement | Decide main text vs S1. The 110-record component is documented in the CoAID entity-resolution report. |
| 87 | Cluster counts placement | Verified against version_aware_social_overlap_graph_summary.csv (5,056/187; 1,662/4). Decide main text vs S2. |
| 137 | Cite FakeNewsNet | Add a verified reference entry (none in v2 references.bib). |
| 137 | Verify FakeNewsNet label definitions | Not documented in the local copy; needs the repository documentation. |
| 141 | Data and code availability statement | Venue-dependent; nothing exists yet (see journal_shortlist.md). |
| 151 | Optional absolute levels | Numbers verified; decide whether to keep. |
| 155 | Conditional FakeNewsNet engagement extension | No such analysis exists; only title-level text scoring was done. Delete or keep as conditional. |
| 175 | Supplementary Table [Sx] for top words | The table exists only as outputs/extension_v1/tables/top_fake_real_terms.csv; it is not in the supplement. Needs a new S-table or removal. |
| 191 | Supplementary [Sx] for FakeNewsNet results | Exists only in outputs/extension_v1/tables; needs a new S-table or removal. |
| 220 | Cramér's V confirm | Confirmed: 0.969 (CoAID) vs 0.353 (FakeHealth) on all rows; 0.971 vs 0.352 on primary-eligible rows (text_source_label_association.csv). Not currently in Supplement S3; decide scope and placement. |
| 222 | Verify review criteria in Dai et al. (2020) | Same as line 21. |
| 226 | Verify collection method in Cui and Lee (2020) | Needs the paper; the project record documents release directories, not collection queries. |
| 256 | Pending quantification of encoding-error words | Supported by top_fake_real_terms.csv: two mojibake tokens rank 4 and 20 among the 25 most Fake-leaning CoAID unigrams. Not in outputs/tables. |
| 262 | Confirm feasibility wording; support table placement | Wording and counts match the 2026-10-09 report. Decide whether engagement_comparability_support.csv becomes an S-table. |
| 272 | References carry-over | Carry v2 references.bib (22 entries) plus any additions above. |

**Count: 19 placeholders (18 excluding the header comment). 4 are verified by this check (lines 87, 151, 220, 262); 2 point to tables that exist only under outputs/extension_v1 (lines 175, 191); 1 refers to an analysis that does not exist (line 155); the rest need literature checks or author decisions.**

## Summary counts

- Check 1: 160 protocol facts; **33 appear nowhere** (v3 Methods or supplement); 34 moved to the supplement only; 14 in v3 Methods only.
- Check 2: 112 numeric values; **8 values in 4 statements sourced only to outputs/extension_v1**; 1 rounding flag; 2 scope-unstated values; 0 digit-changing roundings; 0 scope substitutions.
- Check 3: 5 claim boundaries; 4 stated; **1 weakened** (dominant-component exclusion lacks the reference-population / not-a-replacement wording). 19 placeholders listed.
