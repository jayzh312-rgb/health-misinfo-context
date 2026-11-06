"""Assemble release/staging_v3/: a public deposit tree with code, protocols, manifests, locks, classical bundles, draws, aggregate tables,
predictions, the v3 review package, and de-texted/de-identified processed tables. Never modifies sources. Run: .venv/bin/python scripts/build_release_staging.py"""
import re, json, hashlib, shutil, sys, os
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; FINAL = ROOT/'release/staging_v3'   # git working tree; its .git/ is preserved
STAGE = ROOT/'release/.staging_v3_build'                                        # temporary build directory, synced into FINAL at the end
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
HOME = str(ROOT.parents[1])           # the user's home directory -> redacted
PLACEHOLDER = '[REDACTED_HOME]'
_terms = ROOT/'release/redaction_terms.txt'   # tab-separated 'term<TAB>replacement' lines; kept outside the staging tree so the deposit carries no personal name
NAME_PATTERNS = [tuple(l.rstrip('\n').split('\t', 1)) for l in _terms.read_text().splitlines() if '\t' in l] if _terms.exists() else []
TEXT_COLS = re.compile(r'^(title|text|content|title_original|body_original|title_text|body_text|combined_text|title_text_source_masked|body_text_source_masked|combined_text_source_masked|review_title|review_original_title|review_description|review_summary_json|review_criteria_json|predictor_text|normalized_text|boilerplate_text|informative_text|incompatible_informative_text|statement|left_body|right_body|left_abstract|right_abstract|left_title|right_title|left_text_excerpt|right_text_excerpt|titles_json|title_excerpt|\w+_excerpt|source_title)$', re.I)
KEEP_TEXT = {'outputs/extension_v1/tables/round4_politifact_component_top_titles.csv'}   # a reported result (normalized PolitiFact titles); kept as is and declared in the README
ID_NAME = re.compile(r'(tweet|post|reply|retweet|social|snowflake).*ids?(_json)?$|_ids_json$|^social_id$|member_capture_metadata_json|suspicious_id_examples', re.I)
LONG_ID = re.compile(r'\d{15,}')
EXCLUDE_DOCS = {  # working and editorial documents: not part of the release (reason recorded)
 'outputs/manuscript/availability_readiness_2026-10-13.md': 'internal readiness audit',
 'outputs/manuscript/journal_shortlist.md': 'venue planning note',
 'outputs/manuscript/mentor_integration_report.md': 'editorial working record',
 'outputs/manuscript/revision_v3/mentor_round1_inserts.md': 'editorial working record',
 'outputs/manuscript/revision_v3/reviewer_round1_inserts.md': 'editorial working record',
 'outputs/reports/audit_2026-10-12.md': 'internal repository audit',
 'outputs/manuscript/review_package_v3/internal/page_inspection/': 'rendering QA screenshots',
 'outputs/manuscript/journal_requirements.csv': 'venue planning table',
 'outputs/manuscript/review_package_report.md': 'editorial working record (private v2 package report)',
 'outputs/manuscript/review_package_v3/review_package_v3_report.md': 'editorial working record (package report)',
 'outputs/manuscript/review_package_v3/internal/write_report_v3.py': 'editorial working record (package report generator)',
 'outputs/manuscript/review_package_v3/internal/apply_mentor_round1.py': 'editorial working record',
 'outputs/manuscript/review_package_v3/internal/apply_reviewer_round1.py': 'editorial working record',
 'outputs/manuscript/review_package_v3/manuscript.html': 'superseded rendering (posted PDFs are in socarxiv_v1/)',
 'outputs/manuscript/review_package_v3/supplementary_material.html': 'superseded rendering (posted PDFs are in socarxiv_v1/)',
 'outputs/manuscript/review_package_v3/manuscript_review.pdf': 'superseded rendering (posted PDFs are in socarxiv_v1/)',
 'outputs/manuscript/review_package_v3/supplement_review.pdf': 'superseded rendering (posted PDFs are in socarxiv_v1/)',
}
EXCLUDE_FILES = {  # per-identifier or per-text evidence tables: excluded entirely (reason recorded)
 'outputs/tables/coaid_fakehealth_snowflake_id_details.csv': 'per-post-ID rows (90 MB of social IDs)',
 'outputs/tables/coaid_cross_entity_social_overlap.csv': 'per-post-ID rows',
 'outputs/tables/fakehealth_version_aware_social_overlap.csv': 'per-post-ID rows',
 'outputs/tables/coaid_entity_resolution_member_evidence.csv': 'per-capture article text and post-ID lists',
 'outputs/tables/coaid_content_version_member_evidence.csv': 'per-capture article text',
 'outputs/tables/coaid_entity_resolution_link_evidence.csv': 'per-pair article text and post-ID lists',
}
manifest = dict(copied=[], column_drops=[], redactions=[], excluded=[], notes=[])
def copy_file(src, rel=None):
    rel = rel or str(Path(src).resolve().relative_to(ROOT)); dst = STAGE/rel; dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst); manifest['copied'].append(dict(path=rel, bytes=dst.stat().st_size, source_sha256=sha(src))); return dst
def redact_copy(src, rel=None):
    rel = rel or str(Path(src).resolve().relative_to(ROOT)); dst = STAGE/rel; dst.parent.mkdir(parents=True, exist_ok=True)
    t = Path(src).read_text(errors='replace'); n = t.count(HOME) + t.count(HOME.replace(' ', '%20')) + sum(t.count(a) for a, _ in NAME_PATTERNS)
    out = t.replace(HOME.replace(' ', '%20'), PLACEHOLDER).replace(HOME, PLACEHOLDER)
    for a, b in NAME_PATTERNS: out = out.replace(a, b)
    dst.write_text(out)
    manifest['redactions'].append(dict(path=rel, occurrences_replaced=n, placeholder=PLACEHOLDER, source_sha256=sha(src), staged_sha256=sha(dst)))
def csv_copy(src, rel=None):
    rel = rel or str(Path(src).resolve().relative_to(ROOT))
    if rel in EXCLUDE_FILES: manifest['excluded'].append(dict(path=rel, reason=EXCLUDE_FILES[rel])); return
    df = pd.read_csv(src, dtype=str, keep_default_na=False, low_memory=False)
    drop = []
    for c in df.columns:
        if rel in KEEP_TEXT: break
        if TEXT_COLS.match(c):
            sample = df[c].head(5000).astype(str)
            numeric = pd.to_numeric(sample.replace('', pd.NA), errors='coerce')
            if sample.ne('').any() and numeric.notna().sum() == sample.ne('').sum(): continue   # numeric score column that merely shares a name (e.g. 'title' condition metric); empty text columns are dropped
            drop.append((c, 'text column'))
        elif ID_NAME.search(c):
            sample = df[c].head(5000).astype(str)
            if sample.str.contains(LONG_ID).any(): drop.append((c, 'post/tweet ID list'))
    if drop:
        dst = STAGE/rel; dst.parent.mkdir(parents=True, exist_ok=True)
        df.drop(columns=[c for c, _ in drop]).to_csv(dst, index=False)
        manifest['column_drops'].append(dict(path=rel, columns_dropped=[c for c, _ in drop], reasons={c: r for c, r in drop}, rows=len(df), source_sha256=sha(src), staged_sha256=sha(dst)))
        manifest['copied'].append(dict(path=rel, bytes=dst.stat().st_size, source_sha256=sha(src), modified='columns dropped'))
    else:
        t = Path(src).read_text(errors='replace')
        if HOME in t or HOME.replace(' ', '%20') in t: redact_copy(src, rel)
        else: copy_file(src, rel)
if STAGE.exists(): shutil.rmtree(STAGE)
STAGE.mkdir(parents=True)
# 1. code, tests, protocols, docs
for p in sorted((ROOT/'src').glob('*.py')): copy_file(p)
for p in sorted((ROOT/'outputs/extension_v1/scripts').glob('*.py')): copy_file(p)
for p in sorted((ROOT/'outputs/extension_v1/protocols').glob('*.md')): copy_file(p)
for p in sorted((ROOT/'outputs/extension_v1/reports').glob('*.md')): copy_file(p)
for p in sorted((ROOT/'outputs/extension_v1/round5').iterdir()):   # round 5: within-length re-scoring diagnostic (script, table, hash, README)
    if p.is_file() and p.name != '.DS_Store': (csv_copy if p.suffix == '.csv' else copy_file)(p)
for p in sorted((ROOT/'outputs/extension_v1/figures').glob('*')):
    if p.is_file() and p.name != '.DS_Store': copy_file(p)
for f in ['AGENTS.md', 'REPRODUCING.md', 'LICENSE', 'CITATION.cff', 'requirements.txt', 'requirements-transformer.txt', 'requirements-render.txt', 'CHANGELOG_release.md', 'README.md']:
    tt = (ROOT/f).read_text(errors='replace'); (redact_copy if (HOME in tt or (f not in ('LICENSE', 'CITATION.cff') and any(a in tt for a, _ in NAME_PATTERNS))) else copy_file)(ROOT/f, 'README_project.md' if f == 'README.md' else None)   # LICENSE and CITATION.cff carry the authors' names by design
copy_file(ROOT/'scripts/build_release_staging.py')
# 2. splits, locks, classical bundles
for p in sorted((ROOT/'outputs/splits').glob('*.csv')): csv_copy(p)
for p in sorted(ROOT.glob('outputs/models/*/*.json')):
    if 'distilroberta' in str(p) and p.name != 'locked_configurations.json': continue
    if p.name in ('selection_lock.json', 'prediction_lock.json', 'target_lock.json', 'locked_configurations.json', 'run_signature.json') or p.parent.name == 'fakehealth_publisher_heldout' and p.name.startswith('inner_'):
        t = p.read_text()
        (redact_copy if (HOME in t or any(a in t for a, _ in NAME_PATTERNS)) else copy_file)(p)
for p in sorted(ROOT.glob('outputs/models/*/*.joblib')):
    if 'distilroberta' not in str(p): copy_file(p)
# 3. tables (aggregate + draws), predictions, extension tables
for p in sorted((ROOT/'outputs/tables').iterdir()):
    if p.suffix == '.csv': csv_copy(p)
    elif p.suffix in ('.json', '.txt', '.md'):
        t = p.read_text(errors='replace'); (redact_copy if (HOME in t or HOME.replace(' ', '%20') in t or any(a in t for a, _ in NAME_PATTERNS)) else copy_file)(p)
for p in sorted((ROOT/'outputs/extension_v1/tables').iterdir()):
    if p.suffix == '.csv': csv_copy(p)
    else: copy_file(p)
for p in sorted((ROOT/'outputs/predictions').glob('*.csv')): csv_copy(p)
t = (ROOT/'outputs/reports/transformer_feasibility.json'); redact_copy(t)
for p in sorted(list((ROOT/'outputs/reports').glob('*.md')) + [q for q in (ROOT/'outputs/reports').glob('*.json') if q.name != 'transformer_feasibility.json']): 
    if str(p.relative_to(ROOT)) in EXCLUDE_DOCS: continue
    tt = p.read_text(errors='replace'); (redact_copy if (HOME in tt or any(a in tt for a, _ in NAME_PATTERNS)) else copy_file)(p)
# 4. processed tables (de-texted, de-identified)
for p in sorted((ROOT/'data/processed').glob('*.csv')): csv_copy(p)
# 5. review package v3: manuscript, supplement, figures, validation, workflow, data copies (de-texted), internal records
V3 = ROOT/'outputs/manuscript/review_package_v3'
for p in sorted(V3.rglob('*')):
    if not p.is_file() or p.name == '.DS_Store' or '__pycache__' in p.parts or p.suffix == '.pyc': continue
    rel = str(p.relative_to(ROOT))
    if 'page_inspection' in p.parts or rel in EXCLUDE_DOCS: continue
    if p.suffix == '.csv': csv_copy(p, rel)
    elif p.suffix in ('.md', '.json', '.py', '.html', '.css', '.mmd', '.bib', '.txt'):
        tt = p.read_text(errors='replace'); (redact_copy if (HOME in tt or HOME.replace(' ', '%20') in tt or any(a in tt for a, _ in NAME_PATTERNS)) else copy_file)(p, rel)
    else: copy_file(p, rel)
for p in sorted((ROOT/'outputs/manuscript/revision_v3').iterdir()):
    if p.is_file() and p.suffix in ('.md', '.csv', '.bib', '.py', '.json') and str(p.relative_to(ROOT)) not in EXCLUDE_DOCS: 
        tt = p.read_text(errors='replace'); (redact_copy if (HOME in tt or any(a in tt for a, _ in NAME_PATTERNS)) else copy_file)(p)
for p in sorted((ROOT/'outputs/manuscript/revision_v3/check1_inputs').iterdir()):
    tt = p.read_text(errors='replace'); (redact_copy if (HOME in tt or any(a in tt for a, _ in NAME_PATTERNS)) else copy_file)(p)
for p in sorted((ROOT/'outputs/manuscript/socarxiv_v1').rglob('*')):   # the posted SocArXiv preprint: PDFs and build record; author names kept by design
    if not p.is_file() or p.name == '.DS_Store' or '__pycache__' in p.parts: continue
    if p.suffix in ('.md', '.json', '.txt'):
        tt = p.read_text(errors='replace'); (redact_copy if (HOME in tt or HOME.replace(' ', '%20') in tt) else copy_file)(p)
    else: copy_file(p)
for k, v in EXCLUDE_DOCS.items(): manifest['excluded'].append(dict(path=k, reason=v))
manifest['notes'] += ['outputs/models/distilroberta* (88 GB of checkpoints) excluded; locked_configurations.json retained.',
                      'Processed tables are copied with article-text and post/tweet-ID columns removed; text-dependent scripts need the providers\' raw data to rebuild them.',
                      'Home-directory paths are replaced by ' + PLACEHOLDER + ' and the author\'s name by [AUTHOR] in copied text/JSON/CSV files; the originals are unchanged.']
# 6. licensing notice for derived tables, then deposit README
(STAGE/'LICENSE-DERIVED-TABLES.md').write_text('''# License for derived tables and figures

The derived aggregate tables, bootstrap draws, split manifests, predictions, figure source tables and figures in this deposit
(`outputs/tables/`, `outputs/extension_v1/tables/`, `outputs/extension_v1/round5/`, `outputs/extension_v1/figures/`, `outputs/splits/`, `outputs/predictions/`,
`outputs/manuscript/review_package_v3/figures/`, `outputs/manuscript/review_package_v3/data/`) are released under the
Creative Commons Attribution 4.0 International license (CC BY 4.0): https://creativecommons.org/licenses/by/4.0/
Attribution: cite the paper and this deposit (see CITATION.cff).

The code (`src/`, `outputs/extension_v1/scripts/`, `scripts/`, package build scripts) is under the MIT license in `LICENSE`.
The source datasets (CoAID, FakeHealth, COVID-19 Rumor Dataset, FakeNewsNet) are not redistributed and remain under their providers' own terms;
the de-texted processed tables in `data/processed/` derive from them and are shared only as counts, labels, lineage and eligibility fields.
''')
manifest['copied'].append(dict(path='LICENSE-DERIVED-TABLES.md', bytes=(STAGE/'LICENSE-DERIVED-TABLES.md').stat().st_size, source_sha256='generated'))
# deposit README
rev = 'fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b'
(STAGE/'README.md').write_text(f"""# health-misinfo-context

Code and derived tables for the paper "Fake–Real gaps in observed spread and detectability differ between health misinformation datasets built in different ways: a comparison of CoAID and FakeHealth". The study compares Fake and Real items on observed social-media engagement and on how well text classifiers transfer between datasets, across CoAID, FakeHealth, the COVID-19 Rumor Dataset and two FakeNewsNet collections.

FakeHealth labels mainstream health news by expert-rated reporting quality, whereas CoAID contrasts fact-checked COVID-19 claims with material from health agencies and health-information publishers. Counting items never posted and standardizing on a common model scale, Fake items drew fewer observed posts than Real items in both datasets, but the Fake/Real ratio was 0.253 in CoAID and 0.840 in FakeHealth (ratio-of-ratios 0.301, 95% interval 0.154–0.645). In GossipCop, Fake items drew more observed posts, and PolitiFact's interval included parity. Fake and Real were highly separable within CoAID and modestly within FakeHealth, transfer was poor in both directions, and document length transferred better than words. Because period, labels, sources and collection differ together, these comparisons describe dataset contexts and cannot isolate a pandemic effect.

The repository holds the analysis, audit and figure scripts, the pre-specified extension protocols, the saved splits, locks and classical model bundles, the aggregate tables and bootstrap draws behind the numbers in the manuscript, item-level predictions, processed tables with article text and post identifiers removed, and the manuscript and supplement sources with their validation record. Built {pd.Timestamp.now().strftime('%Y-%m-%d')} by `scripts/build_release_staging.py`; integrity in `SHA256SUMS`, file-level provenance in `DEPOSIT_MANIFEST.json`.

## Repository layout
- `src/`: analysis, audit and figure scripts with their unit tests (`src/test_*.py`). `outputs/extension_v1/scripts/`, `protocols/` and `reports/`: the extension analyses with their pre-specified protocols.
- `README.md`, `REPRODUCING.md`, `LICENSE` (MIT), `LICENSE-DERIVED-TABLES.md` (CC BY 4.0), `CITATION.cff`, `CHANGELOG_release.md`, `requirements.txt` (Python 3.14.2), `requirements-transformer.txt` (Python 3.12.7), `requirements-render.txt`; `README_project.md` and `AGENTS.md` are the working project notes.
- `outputs/splits/`: grouped-CV, holdout, outer-partition and transfer assignment manifests.
- `outputs/models/*/`: every selection, prediction and target lock, the transformer `locked_configurations.json`, and all classical joblib model bundles (TF-IDF LR/SVM, title-only, transfer, publisher-held-out).
- `outputs/extension_v1/round5/`: the within-length-quintile ROC-AUC re-scoring diagnostic (Supplementary Table S19.4): script, table, SHA-256 and README.
- `outputs/extension_v1/figures/`: the round-4 bootstrap-shape figures (also used as Figure S1 panels).
- `outputs/extension_v1/tables/round4_politifact_component_top_titles.csv` is the one table that carries article titles (normalized PolitiFact titles of the dominant component, a reported result); every other title, body, abstract and excerpt column is dropped.
- `outputs/tables/` and `outputs/extension_v1/tables/`: every aggregate table, including the four bootstrap-draw files (`version_aware_bootstrap_refits.csv`, `cross_context_no_dominant_bootstrap_refits.csv`, `fakenewsnet_engagement_bootstrap_refits.csv`, `fakenewsnet_round3_exclusion_bootstrap_refits.csv`); `outputs/predictions/`: item-level predictions keyed by internal UIDs; `outputs/reports/`: the report for each analysis stage.
- `data/processed/`: the processed tables **with article-text and post/tweet-ID columns removed** (see `DEPOSIT_MANIFEST.json` → `column_drops`); counts, labels, lineage, eligibility flags and split fields are retained.
- `outputs/manuscript/socarxiv_v1/`: the preprint as posted (combined PDF, manuscript-only and supplement-only PDFs, and the build record with their SHA-256).
- `outputs/manuscript/review_package_v3/`: the manuscript and supplement sources, Figures 1–4 and M1 with their scripts, the data behind the tables, and the validation record (see its README); `outputs/manuscript/revision_v3/`: verification, reference checks, migration log and supplement build inputs.

## Excluded and why
- The raw datasets are not redistributed. Obtain them from the providers: CoAID (https://github.com/cuilimeng/CoAID), FakeHealth (https://github.com/EnyanDai/FakeHealth; engagement content must be rehydrated through the X API), COVID-19 Rumor Dataset (https://github.com/MickeysClubhouse/COVID-19-rumor-dataset, MIT), FakeNewsNet (https://github.com/KaiDMML/FakeNewsNet). CoAID, FakeHealth and FakeNewsNet state no license; FakeNewsNet states that its complete dataset cannot be distributed for Twitter-policy and publisher-copyright reasons.
- Article text, tweet/post/reply identifiers and per-identifier evidence tables ({len(EXCLUDE_FILES)} files listed under `excluded` in the manifest): provider and platform terms unknown or restrictive.
- Transformer checkpoints (`outputs/models/distilroberta*`, 88 GB): size. The pinned checkpoint is `distilbert/distilroberta-base` at revision `{rev}`; locked settings, seeds and all predictions are included, so the transformer rows can be re-run on an MPS, CUDA or CPU device (bitwise reproducibility across devices is not claimed).
- Home-directory paths in environment audit records are replaced by `{PLACEHOLDER}` (see `redactions`).
- Working and editorial documents (readiness audit, venue planning notes, mentor and reviewer round records, editorial package reports, repository audit note, rendering QA screenshots): internal records, listed under `excluded` in the manifest.

## Licensing
- **Code** (`src/`, `outputs/extension_v1/scripts/`, `scripts/`, package build scripts): MIT, see `LICENSE`.
- **Derived tables, draws, manifests, predictions and figures** produced by this project: Creative Commons Attribution 4.0 International (CC BY 4.0), see `LICENSE-DERIVED-TABLES.md`; cite the paper and this repository.
- **Source datasets**: not redistributed; CoAID, FakeHealth, the COVID-19 Rumor Dataset and FakeNewsNet remain under their providers' own terms (the COVID-19 Rumor repository is MIT; the other three state no license, and FakeNewsNet states that its complete dataset cannot be distributed). The processed tables here are derivatives shared only as counts, labels, lineage and eligibility fields.

## Checking the reported numbers
`outputs/manuscript/review_package_v3/review_package_v3_validation.json` maps each number displayed in the manuscript and supplement to the table and column it comes from, and every one of those tables is in this repository. The validation ran on the unredacted research tables; `REDACTION_MANIFEST.md` shows that redaction changed no numeric cell in any aggregate table.

## Reproducing
See `REPRODUCING.md`. Checking the reported numbers needs only this repository; re-running the pipeline needs the four source datasets, in the order given there.

## How to cite
Cite the paper and this repository. `CITATION.cff` has both, and GitHub shows it under "Cite this repository". The preprint DOI will be added after posting.

## Redaction
Some staged files differ from their source in the research folder: article-text, title, abstract, excerpt and tweet-ID columns are dropped, tweet/post identifiers in URL, key, evidence and file-name fields are masked as `[TWEET_ID_REDACTED]`, and home-directory paths, the author name and personal file names are replaced in a few records. `REDACTION_MANIFEST.md` lists every such file with its source and staged SHA-256, the columns dropped or masked and the number of masked cells, and confirms that no numeric cell in any aggregate table under `outputs/tables/` or `outputs/extension_v1/tables/` changed. The validation hashes and fingerprints inside `outputs/manuscript/review_package_v3/` (for example `review_package_v3_validation.json`) and in the unit tests refer to the unredacted source tables, so they will not match the staged files that the manifest lists; `SHA256SUMS` holds the staged hashes.
""")

(STAGE/'.gitignore').write_text('__pycache__/\n*.pyc\n.DS_Store\n.venv*/\nlogs/\ndata/raw/\n*.ipynb\n.ipynb_checkpoints/\n*.zip\n*.pkg\n.env\n# transformer checkpoints stay out; the locked configuration is tracked\noutputs/models/distilroberta*/*\n!outputs/models/distilroberta/locked_configurations.json\n')
# 6b. mask tweet/post status identifiers that survive inside URL, key and evidence fields (17-19 digit integers not embedded in hashes or decimals)
TWEET_ID = re.compile(r'(?:(?<=%[0-9A-Fa-f]{2})|(?<![0-9A-Za-z.]))[1-9][0-9]{16,18}(?![0-9A-Za-z])')   # 17-19 digit identifiers in URLs, keys and file names (never start with 0; not inside hashes or decimals; percent-encoded prefixes allowed)
URL_TOKEN = re.compile(r'(?:https?://|www\.)[^\s,"\'<>|]+')                      # inside a URL, any 17-19 digit run bounded by non-digits is an identifier (article, pin, embed or status IDs)
URL_DIGITS = re.compile(r'(?<![0-9])[0-9]{17,19}(?![0-9])')
STATUS_URL = re.compile(r'(/status(?:es)?/)[0-9]{6,}')                               # pre-snowflake tweet IDs in URLs
DOI_PREFIX = re.compile(r'10\.\d{4,9}/$')
def _mask(t):
    n = [0]
    def f(m):
        if DOI_PREFIX.search(t[max(0, m.start()-12):m.start()]): return m.group(0)   # a DOI suffix, not an identifier
        n[0] += 1; return '[TWEET_ID_REDACTED]'
    out = TWEET_ID.sub(f, t)
    out, k = STATUS_URL.subn(r'\1[TWEET_ID_REDACTED]', out); n[0] += k
    def u(m):
        url = m.group(0)
        def g(mm):
            if DOI_PREFIX.search(url[max(0, mm.start()-12):mm.start()]) or '/doi/' in url[:mm.start()] or 'doi.org' in url[:mm.start()]: return mm.group(0)
            n[0] += 1; return '[TWEET_ID_REDACTED]'
        return URL_DIGITS.sub(g, url)
    out = URL_TOKEN.sub(u, out)
    return out, n[0]
for p in sorted(STAGE.rglob('*')):
    if p.is_file() and p.suffix in ('.csv', '.json', '.md', '.txt', '.py', '.html', '.bib', '.mmd', '.css', '.cff'):
        t = p.read_text(errors='replace'); out, n = _mask(t)
        if n:
            p.write_text(out)
            manifest['redactions'].append(dict(path=str(p.relative_to(STAGE)), occurrences_replaced=n, placeholder='[TWEET_ID_REDACTED]', kind='tweet_or_post_status_id', staged_sha256=sha(p)))
manifest['notes'].append('17-19 digit integers in URL, key, evidence and file-name fields (tweet/post status identifiers and similar platform IDs) are replaced by [TWEET_ID_REDACTED]; hashes, decimals and DOIs are untouched.')
# 6c. redaction manifest: every staged file that differs from its source, with masked/dropped columns and masked-cell counts
GENERATED = {'README.md', 'LICENSE-DERIVED-TABLES.md', 'DEPOSIT_MANIFEST.json', 'SHA256SUMS', '.gitignore', 'REDACTION_MANIFEST.md'}
PLACEHOLDERS = ['[TWEET_ID_REDACTED]', PLACEHOLDER, '[personal file]'] + sorted({b for _, b in NAME_PATTERNS})
is_num = lambda s: s.ne('').any() and pd.to_numeric(s[s.ne('')], errors='coerce').notna().all()
rrows, nchecks, problems = [], [], []
for q in sorted(p for p in STAGE.rglob('*') if p.is_file()):
    rel = str(q.relative_to(STAGE))
    if rel in GENERATED: continue
    src = ROOT/('README.md' if rel == 'README_project.md' else rel)
    if not src.exists(): problems.append(f'no source for {rel}'); continue
    s_sha, q_sha = sha(src), sha(q)
    if s_sha == q_sha: continue
    e = dict(path=rel, source=str(src.relative_to(ROOT)), source_sha=s_sha, staged_sha=q_sha, dropped=[], masked={}, cells=0)
    if q.suffix == '.csv':
        a = pd.read_csv(src, dtype=str, keep_default_na=False, low_memory=False); b = pd.read_csv(q, dtype=str, keep_default_na=False, low_memory=False)
        e['dropped'] = [c for c in a.columns if c not in b.columns]; common = [c for c in a.columns if c in b.columns]
        if [c for c in b.columns if c not in a.columns] or len(a) != len(b) or list(b.columns) != common: problems.append(f'{rel}: shape or column order changed')
        for c in common:
            d = a[c].values != b[c].values; n = int(d.sum())
            if n:
                e['masked'][c] = n; e['cells'] += n
                if [v for v in b[c].values[d] if not any(ph in v for ph in PLACEHOLDERS)]: problems.append(f'{rel}: column {c} changed without a placeholder')
        if rel.startswith(('outputs/tables/', 'outputs/extension_v1/tables/')):
            num = [c for c in common if is_num(a[c])]; tn = [c for c in num if c in e['masked']]; dn = [c for c in e['dropped'] if is_num(a[c])]
            nchecks.append((rel, len(num), int(sum((a[c].values == b[c].values).sum() for c in num)), tn, dn, sorted(e['masked']), e['dropped']))
            if tn or dn: problems.append(f'{rel}: numeric columns affected {tn + dn}')
    else:
        ta, tb = src.read_text(errors='replace'), q.read_text(errors='replace')
        e['masked'] = {ph: tb.count(ph) - ta.count(ph) for ph in PLACEHOLDERS if tb.count(ph) > ta.count(ph)}; e['cells'] = sum(e['masked'].values())
        if ta.count('\n') != tb.count('\n'): problems.append(f'{rel}: line count changed')
    rrows.append(e)
declared = {d['path'] for d in manifest['column_drops']} | {r['path'] for r in manifest['redactions']}
undeclared = sorted({e['path'] for e in rrows} - declared); declared_identical = sorted(declared - {e['path'] for e in rrows})
if undeclared or declared_identical: problems.append(f'manifest mismatch: undeclared {undeclared}, declared-but-identical {declared_identical}')
rm = ['# Redaction manifest for `release/staging_v3/`', '', 'Every staged file whose bytes differ from its source file in the research folder, with the columns dropped or masked and the number of masked cells. Files not listed are byte-identical to their source. Generated by `scripts/build_release_staging.py` on ' + pd.Timestamp.now().strftime('%Y-%m-%d') + '.',
      'Placeholders: `[TWEET_ID_REDACTED]` (tweet/post and similar platform identifiers in URL, key, evidence and file-name fields), `' + PLACEHOLDER + '` (home-directory paths), `[AUTHOR]`/`[author]` (author name), `[personal file]` (personal root file names).',
      'Validation hashes and fingerprints recorded inside `outputs/manuscript/review_package_v3/` (for example `review_package_v3_validation.json`) and in the test suite refer to the **unredacted source tables**; the staged SHA-256 values below are the ones in `SHA256SUMS`.', '',
      f"Summary: {len(rrows)} staged files differ from their source ({sum(e['path'].endswith('.csv') for e in rrows)} CSV, {sum(not e['path'].endswith('.csv') for e in rrows)} text/JSON); {sum(e['cells'] for e in rrows):,} masked cells or occurrences; {sum(len(e['dropped']) for e in rrows)} columns dropped. Generated files without a source: " + ', '.join(f'`{g}`' for g in sorted(GENERATED)) + f" (hashes in `SHA256SUMS`). Consistency problems: {len(problems)}.", '',
      '## Files that differ from their source', '', '| Staged path | Source | Source SHA-256 | Staged SHA-256 | Columns dropped | Columns masked (cells) | Masked cells |', '|---|---|---|---|---|---|---|']
for e in rrows: rm.append(f"| `{e['path']}` | `{e['source']}` | `{e['source_sha']}` | `{e['staged_sha']}` | {', '.join(f'`{c}`' for c in e['dropped']) or '—'} | {', '.join(f'`{c}` ({n})' for c, n in e['masked'].items()) or '—'} | {e['cells']} |")
rm += ['', '## Numeric-cell check for aggregate tables', '', 'For each table under `outputs/tables/` and `outputs/extension_v1/tables/` that differs from its source: a column is numeric when every non-empty source cell parses as a number. PASS means no numeric column was masked or dropped and every numeric cell is identical.', '',
       '| Table | Numeric columns | Numeric cells identical | Numeric columns masked or dropped | Non-numeric columns masked | Columns dropped | Result |', '|---|---|---|---|---|---|---|']
for rel, nn, same, tn, dn, masked, dropped in nchecks: rm.append(f"| `{rel}` | {nn} | {same:,} | {', '.join(tn + dn) or 'none'} | {', '.join(f'`{c}`' for c in masked) or '—'} | {', '.join(f'`{c}`' for c in dropped) or '—'} | {'PASS' if not (tn or dn) else 'FAIL'} |")
rm += ['', '## Consistency', '', f"- Differing files not declared in `DEPOSIT_MANIFEST.json`: {undeclared or 'none'}", f"- Declared as modified but byte-identical: {declared_identical or 'none'}", f"- Problems: {problems or 'none'}"]
(STAGE/'REDACTION_MANIFEST.md').write_text('\n'.join(rm) + '\n')
manifest['redaction_manifest'] = dict(differing_files=len(rrows), masked_cells=sum(e['cells'] for e in rrows), columns_dropped=sum(len(e['dropped']) for e in rrows), aggregate_tables_checked=len(nchecks), numeric_failures=[r[0] for r in nchecks if r[3] or r[4]], problems=problems)
# 7. manifest + SHA256SUMS (written last so every entry matches)
(STAGE/'DEPOSIT_MANIFEST.json').write_text(json.dumps(manifest, indent=1))
lines = []
for p in sorted(STAGE.rglob('*')):
    if p.is_file() and p.name != 'SHA256SUMS': lines.append(f"{sha(p)}  {p.relative_to(STAGE)}")
(STAGE/'SHA256SUMS').write_text('\n'.join(lines) + '\n')
total = sum(p.stat().st_size for p in STAGE.rglob('*') if p.is_file())
# 8. sync the built tree into the git working tree, preserving .git/
FINAL.mkdir(parents=True, exist_ok=True)
for child in FINAL.iterdir():
    if child.name == '.git': continue
    shutil.rmtree(child) if child.is_dir() else child.unlink()
for child in STAGE.iterdir(): shutil.move(str(child), str(FINAL/child.name))
shutil.rmtree(STAGE)
print(json.dumps(dict(files=len(lines) + 1, total_bytes=total, column_drop_files=len(manifest['column_drops']), redacted_files=len(manifest['redactions']), excluded=len(manifest['excluded']))))
