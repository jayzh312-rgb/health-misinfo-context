"""Frozen, label-blind text-quality decisions and metadata-only source masking.

Run from any directory. Reads prepared text cohorts and existing split manifests only.
Never fits a model, accesses raw data, or rewrites an input.
"""
from pathlib import Path
import logging
import hashlib
import json
import re
from urllib.parse import urlsplit
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('title_text', 'body_text', 'combined_text')
PREFIXES = {'FakeHealth': 'fakehealth', 'CoAID': 'coaid'}
META = ('publisher_source', 'article_host', 'article_url')
# Fixed identity aliases, activated only by a corresponding metadata host.
# No alias is inferred from class frequencies or text/label association.
ALIASES = {
 'cdc.gov': ['CDC', 'Centers for Disease Control and Prevention', 'cdc twenty four seven', 'saving lives protecting people'],
 'who.int': ['WHO', 'World Health Organization'],
 'nih.gov': ['NIH', 'National Institutes of Health'],
 'cnn.com': ['CNN', 'CNN Health', 'CNN Business'],
 'nbcnews.com': ['NBC News', 'NBCNews', 'Know Your Value'],
 'cbsnews.com': ['CBS News', 'CBSNews'],
 'abcnews.go.com': ['ABC News'], 'foxnews.com': ['Fox News'],
 'nytimes.com': ['The New York Times', 'New York Times'],
 'washingtonpost.com': ['The Washington Post', 'Washington Post'],
 'theguardian.com': ['The Guardian'], 'reuters.com': ['Reuters', 'Reuters Health'],
 'healthday.com': ['HealthDay', 'HealthDay News'], 'webmd.com': ['WebMD'],
 'medicalnewstoday.com': ['Medical News Today'], 'medicaldaily.com': ['Medical Daily'],
 'youtube.com': ['YouTube'], 'youtu.be': ['YouTube'], 'facebook.com': ['Facebook'],
 'healthnewsreview.org': ['HealthNewsReview', 'Health News Review'],
}
# Ambiguous ordinary words/acronyms use exact capitalization to limit false matches.
CASE_SENSITIVE = {'time', 'stat', 'slate', 'vox', 'who'}
IGNORECASE_EQUIVALENTS = str.maketrans({'İ': 'i', 'ı': 'i', 'ſ': 's', 'K': 'k'})
URL_HOST = re.compile(r'(?i)(?:https?://|www\.)([a-z0-9][a-z0-9.-]*\.[a-z]{2,})(?::\d+)?')
REVIEW_WORDS = re.compile(r'(?i)\b(fact[- ]check|rated|rating|misleading|unsubstantiated|story reviews?|reviewers?|ground truth|not satisfactory)\b')


def read(path):
    return pd.read_csv(path, keep_default_na=False)


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_dictionary(metadata):
    """Strict allowlist: even callers supplying extra columns cannot affect output."""
    metadata = metadata.loc[:, list(META)]
    entries = set()
    for r in metadata.itertuples(index=False):
        host = r.article_host.lower().removeprefix('www.')
        if not host and r.article_url:
            host = urlsplit(r.article_url if '://' in r.article_url else 'https://' + r.article_url).netloc.lower().removeprefix('www.')
        if host:
            entries.add((host, 'domain', host, 'article_host/article_url', False))
        publisher = r.publisher_source.strip()
        if publisher:
            domain = bool(re.fullmatch(r'[\w.-]+\.[a-zA-Z]{2,}', publisher))
            sensitive = publisher.casefold() in CASE_SENSITIVE
            # TIME and STAT are identifiable branding, unlike ordinary lower-case words.
            term = publisher.upper() if publisher.casefold() in {'time', 'stat'} else publisher
            entries.add((term, 'domain' if domain else 'publisher_name', host or publisher, 'publisher_source', sensitive))
        for key, aliases in ALIASES.items():
            if host == key or host.endswith('.' + key):
                for alias in aliases:
                    category = 'institutional_boilerplate' if alias in {'cdc twenty four seven', 'saving lives protecting people', 'Know Your Value'} else 'publisher_name'
                    entries.add((alias, category, key, 'fixed_identity_alias_of_metadata_host', alias.casefold() in CASE_SENSITIVE))
    return pd.DataFrame(sorted(entries), columns=['identifier', 'category', 'source_identity', 'provenance', 'case_sensitive'])


def compile_dictionary(dictionary):
    return [(re.compile(r'(?<![\w])' + re.escape(r.identifier) + r'(?![\w])', 0 if r.case_sensitive else re.I), r)
            for r in dictionary.itertuples(index=False)]


def cues_and_mask(text, compiled):
    spans, categories, identifiers = [], set(), set()
    lower_text = text.translate(IGNORECASE_EQUIVALENTS).lower()
    for pattern, entry in compiled:
        # Cheap literal screening before boundary matching; no change to matches.
        if entry.identifier.translate(IGNORECASE_EQUIVALENTS).lower() not in lower_text:
            continue
        for match in pattern.finditer(text):
            spans.append(match.span()); categories.add(entry.category); identifiers.add(entry.identifier)
    for match in URL_HOST.finditer(text):
        # Mask only hostname; scheme, path, punctuation and all other words survive.
        spans.append(match.span(1)); categories.add('explicit_url_host'); identifiers.add(match.group(1))
    # Source-specific attribution template, not arbitrary personal bylines.
    for identifier in identifiers:
        name = re.escape(identifier)
        attribution = (r'(?:\(\s*' + name + r'\s*\)|\bby\s+' + name
                       + r'\b|(?:copyright|©)[^.!?\n]{0,80}' + name
                       + r'|' + name + r'[^.!?\n]{0,80}all rights reserved)')
        if re.search(attribution, text, re.I):
            categories.add('source_attribution_or_copyright_template')
            break
    merged = []
    for start, end in sorted(set(spans)):
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    masked = text
    for start, end in reversed(merged):
        masked = masked[:start] + '[SOURCE]' + masked[end:]
    return masked, sorted(categories), sorted(identifiers)


def quality_decision(row):
    """Rules frozen after reviewing the 118 flagged rows (33 unique text pairs).

    A specific substantive headline can be retained with a boilerplate-only body.
    Generic site section headings/person names are not substantive article content.
    """
    if row.dataset == 'FakeHealth' and row.title_text.startswith('SLOWDOWN IN NUMBER OF SYSTEMATIC STORY REVIEWS'):
        return 'ARTIFACT_DOMINATED', 'Review-site funding/navigation announcement; no article body or health-news claim.'
    if not row.mixed_artifact_component_flag:
        return 'NOT_FLAGGED', 'Outside mixed-artifact review scope.'
    if row.dataset == 'FakeHealth':
        if row.title_text == 'Video News' and 'Disney' in row.body_text:
            return 'SUBSTANTIVE_WITH_MINOR_ARTIFACT', 'Generic Video News heading, but substantive Disney business-news description; possible topic/capture mismatch retained and documented.'
        if row.body_text == 'Know Your Value':
            fraction = len(row.body_text) / max(1, len(row.combined_text))
            return ('SUBSTANTIVE_WITH_MAJOR_ARTIFACT' if fraction >= .25 else 'SUBSTANTIVE_WITH_MINOR_ARTIFACT',
                    'Specific health-news headline retained; body is only Know Your Value branding; unsuitable for body-only sensitivity.')
    if row.title_text.strip('"').startswith(('Opening up North Carolina', 'Dr Judy Mikovits explains')):
        return 'SUBSTANTIVE_WITH_MAJOR_ARTIFACT', 'Specific news/health claim in video headline; body is YouTube boilerplate; title remains usable.'
    title = row.title_text.strip('"')
    if title in {'Page not found', '403 Forbidden', 'Not Acceptable!', 'CDC - Page Not Found'}:
        return 'ARTIFACT_DOMINATED', 'Error/access heading and error-message body; no substantive article content.'
    if title in {'Ashley Bird', 'John C. Johnson', 'Ryan Tee', 'Kimo Sabi', 'Steve Meads'} and row.body_text.startswith('join this group'):
        return 'ARTIFACT_DOMINATED', 'Social-profile name plus join-group prompt; no substantive news claim.'
    if title == 'Testing' and row.body_text.startswith('enjoy the videos'):
        return 'ARTIFACT_DOMINATED', 'Generic Testing heading plus YouTube boilerplate; no substantive claim.'
    if row.body_text.startswith('centers for disease control and prevention.') and row.body_quality_flag == 'clear_crawl_artifact':
        return 'ARTIFACT_DOMINATED', 'CDC topic/section heading with only repeated institutional branding; body supplies no article information.'
    return 'UNCERTAIN', 'Unrecognized mixed-artifact evidence; requires review before release.'


def title_decision(row):
    if row.dataset != 'FakeHealth':
        return False, 'uncertain', False, 'CoAID prepared provenance does not distinguish fetched title from source-title fallback; no review-title substitution inferred.'
    if row.title_origin == 'fetched_content':
        return False, 'low', False, 'Fetched title per validated cohort provenance; no additional fallback-title review.'
    evaluative = bool(REVIEW_WORDS.search(row.title_text))
    if row.title_text.startswith('SLOWDOWN IN NUMBER OF SYSTEMATIC STORY REVIEWS'):
        return True, 'moderate', True, 'Review-site administrative announcement, not a rating/verdict; excluded separately as non-article artifact.'
    if row.title_origin == 'review_original_title':
        return True, 'low', evaluative, 'Reviewed fallback: metadata identifies original article headline; no explicit assessment of article quality, rating, or verdict. Scientific skepticism is not ground truth.'
    if row.title_origin == 'review_editorial_title' and evaluative:
        return True, 'high', True, 'Editorial review title explicitly exposes evaluation language.'
    return True, 'uncertain', evaluative, 'Unresolved title provenance; requires documented review.'


def source_counts(frame, dataset, scope):
    result = []
    for measure in ('publisher_source', 'article_host'):
        x = frame[frame[measure].ne('')]
        table = pd.crosstab(x[measure], x.binary_analysis_label).reindex(columns=['Fake', 'Real'], fill_value=0)
        mixed = table.gt(0).all(axis=1)
        result.append(dict(dataset=dataset, scope=scope, source_measure=measure,
            fake_only_sources=int(((table.Fake > 0) & (table.Real == 0)).sum()),
            real_only_sources=int(((table.Real > 0) & (table.Fake == 0)).sum()),
            mixed_sources=int(mixed.sum()), exclusive_rows=int(table[~mixed].to_numpy().sum()),
            mixed_rows=int(table[mixed].to_numpy().sum()), mixed_fake_N=int(table.loc[mixed, 'Fake'].sum()),
            mixed_real_N=int(table.loc[mixed, 'Real'].sum()), missing_source_N=int(frame[measure].eq('').sum()),
            fake_support_sources=int(table.Fake.gt(0).sum()), real_support_sources=int(table.Real.gt(0).sum())))
    return result


def validate_and_summarize(frame, holdout, cv, dataset, condition, eligible):
    assert frame.content_uid.is_unique and holdout.content_uid.is_unique and cv.content_uid.is_unique
    x = frame[eligible]
    h, c = holdout.set_index('content_uid'), cv.set_index('content_uid')
    assert set(x.content_uid) <= set(h.index) & set(c.index)
    for r in x.itertuples():
        assert r.leakage_group == h.loc[r.content_uid, 'leakage_group'] == c.loc[r.content_uid, 'leakage_group']
        assert r.holdout_assignment == h.loc[r.content_uid, 'assignment']
        assert r.cv_fold == c.loc[r.content_uid, 'fold']
    assert x.groupby('leakage_group').holdout_assignment.nunique().le(1).all()
    assert x.groupby('leakage_group').cv_fold.nunique().le(1).all()
    assert (x.holdout_assignment.eq('test') == x.cv_fold.eq(-1)).all()
    partitions = [('all', x), ('train_pool', x[x.cv_fold >= 0]), ('holdout', x[x.cv_fold == -1])]
    for fold in sorted(cv.loc[cv.fold.ge(0), 'fold'].unique()):
        partitions += [(f'cv_{fold}_validation', x[x.cv_fold == fold]),
                       (f'cv_{fold}_training', x[(x.cv_fold >= 0) & (x.cv_fold != fold)])]
    rows = []
    for name, part in partitions:
        both = set(part.binary_analysis_label) == {'Fake', 'Real'}
        assert both, (dataset, condition, name, 'lost class support')
        rows.append(dict(dataset=dataset, condition=condition, partition=name, N=len(part),
                         fake_N=int(part.binary_analysis_label.eq('Fake').sum()), real_N=int(part.binary_analysis_label.eq('Real').sum()),
                         leakage_groups=part.leakage_group.nunique(), both_classes=both, split_integrity=True))
    return rows


def markdown(frame):
    return '| ' + ' | '.join(frame.columns) + ' |\n| ' + ' | '.join(['---'] * len(frame.columns)) + ' |\n' + '\n'.join('| ' + ' | '.join(map(str, row)) + ' |' for row in frame.itertuples(index=False, name=None))


def main():
    (ROOT / 'logs').mkdir(exist_ok=True)
    logging.basicConfig(filename=ROOT / 'logs/text_quality_and_source_cue_audit.log',
                        level=logging.INFO, format='%(asctime)s %(message)s', force=True)
    logging.info('Starting prepared-cohort audit; no model training')
    paths = [ROOT / f'data/processed/{p}_text_primary.csv' for p in PREFIXES.values()]
    paths += [ROOT / f'outputs/splits/{p}_text_{s}.csv' for p in PREFIXES.values() for s in ('holdout', 'grouped_cv')]
    before = {str(p.relative_to(ROOT)): fingerprint(p) for p in paths}
    frames = {ds: read(ROOT / f'data/processed/{p}_text_primary.csv') for ds, p in PREFIXES.items()}
    # Label-free transductive metadata dictionary shared across datasets and folds.
    # This is declared sensitivity preprocessing, not fitted vocabulary or performance selection.
    dictionary = build_dictionary(pd.concat([d[list(META)] for d in frames.values()], ignore_index=True))
    compiled = compile_dictionary(dictionary)
    cached_text_results = {}
    quality_rows, cue_rows, summary_rows, confounding, totals = [], [], [], [], []
    for ds, prefix in PREFIXES.items():
        logging.info('Auditing %s', ds)
        original = frames[ds]
        d = original.copy()
        decisions = [quality_decision(r) for r in d.itertuples()]
        titles = [title_decision(r) for r in d.itertuples()]
        d['artifact_audit_class'] = [a for a, _ in decisions]
        d['artifact_audit_reason'] = [b for _, b in decisions]
        assert not d.loc[d.primary_text_eligible & d.mixed_artifact_component_flag, 'artifact_audit_class'].eq('UNCERTAIN').any(), 'New mixed-artifact evidence requires review'
        for i, name in enumerate(('title_fallback_reviewed', 'title_leakage_risk', 'title_evaluative_language_flag', 'title_leakage_reason')):
            d[name] = [t[i] for t in titles]
        d['primary_modeling_eligible'] = d.primary_text_eligible & d.artifact_audit_class.ne('ARTIFACT_DOMINATED') & d.title_leakage_risk.ne('high')
        d['modeling_exclusion_reason'] = d.exclusion_reason
        for idx in d.index[d.primary_text_eligible]:
            reasons = []
            if d.at[idx, 'artifact_audit_class'] == 'ARTIFACT_DOMINATED': reasons.append('artifact_dominated')
            if d.at[idx, 'title_leakage_risk'] == 'high': reasons.append('high_risk_title_ground_truth_leakage')
            d.at[idx, 'modeling_exclusion_reason'] = ';'.join(reasons)
        d['title_only_modeling_eligible'] = d.primary_modeling_eligible & d.title_eligible
        d['body_only_modeling_eligible'] = d.primary_modeling_eligible & d.body_eligible
        h = read(ROOT / f'outputs/splits/{prefix}_text_holdout.csv')
        c = read(ROOT / f'outputs/splits/{prefix}_text_grouped_cv.csv')
        d['holdout_assignment'] = d.content_uid.map(h.set_index('content_uid').assignment).fillna('not_previously_eligible')
        d['cv_fold'] = d.content_uid.map(c.set_index('content_uid').fold).fillna(-2).astype(int)
        for field in FIELDS:
            logging.info('%s: source-cue matching and masking %s', ds, field)
            for text in d[field].unique():
                if text not in cached_text_results:
                    cached_text_results[text] = cues_and_mask(text, compiled)
            results = [cached_text_results[t] for t in d[field]]
            d[field + '_source_masked'] = [r[0] for r in results]
            d[field + '_source_cue_flag'] = [bool(r[1]) for r in results]
            d[field + '_source_cue_categories'] = [json.dumps(r[1]) for r in results]
            d[field + '_source_cue_identifiers'] = [json.dumps(r[2]) for r in results]
            for scope, eligible in [('cohort_eligible', d.primary_text_eligible), ('modeling_ready', d.primary_modeling_eligible)]:
                sub = d[eligible]
                categories = ['any_explicit_source_cue', 'publisher_name', 'domain', 'explicit_url_host', 'institutional_boilerplate', 'source_attribution_or_copyright_template']
                for category in categories:
                    mask = sub[field + '_source_cue_flag'] if category == categories[0] else sub[field + '_source_cue_categories'].map(lambda v: category in json.loads(v))
                    hit = sub[mask]
                    cue_rows.append(dict(dataset=ds, scope=scope, field=field, category=category, denominator_N=len(sub), cue_N=len(hit), percent=round(100 * len(hit) / len(sub), 3), fake_N=int(hit.binary_analysis_label.eq('Fake').sum()), real_N=int(hit.binary_analysis_label.eq('Real').sum())))
        reviewed = d.primary_text_eligible & (d.mixed_artifact_component_flag | d.title_fallback_reviewed)
        audit_cols = ['dataset', 'content_uid', 'mixed_artifact_component_flag', 'artifact_audit_class', 'artifact_audit_reason', 'title_origin', 'title_fallback_reviewed', 'title_leakage_risk', 'title_evaluative_language_flag', 'title_leakage_reason', 'primary_modeling_eligible', 'modeling_exclusion_reason']
        q = d.loc[reviewed, audit_cols].copy()
        q['title_excerpt'] = d.loc[reviewed, 'title_text'].str.slice(0, 150)
        q['body_characters'] = d.loc[reviewed, 'body_text'].str.len()
        q['text_sha256'] = d.loc[reviewed, 'combined_text'].map(lambda s: hashlib.sha256(s.encode()).hexdigest())
        quality_rows.append(q)
        for condition, flag in [('primary_natural', 'primary_modeling_eligible'), ('source_masked', 'primary_modeling_eligible'), ('title_only', 'title_only_modeling_eligible'), ('body_only', 'body_only_modeling_eligible')]:
            summary_rows += validate_and_summarize(d, h, c, ds, condition, d[flag])
        for scope, flag in [('cohort_eligible', 'primary_text_eligible'), ('modeling_ready', 'primary_modeling_eligible')]:
            confounding += source_counts(d[d[flag]], ds, scope)
        assert d[list(original.columns)].equals(original)
        assert d.loc[~d.primary_modeling_eligible, 'modeling_exclusion_reason'].ne('').all()
        totals.append(dict(dataset=ds, previously_eligible=int(d.primary_text_eligible.sum()), modeling_ready=int(d.primary_modeling_eligible.sum()), newly_artifact_excluded=int((d.primary_text_eligible & d.artifact_audit_class.eq('ARTIFACT_DOMINATED')).sum()), high_risk_title_excluded=int((d.primary_text_eligible & d.title_leakage_risk.eq('high')).sum()), preexisting_exclusions=int((~d.primary_text_eligible).sum())))
        d.to_csv(ROOT / f'data/processed/{prefix}_text_modeling_ready.csv', index=False)
    tables = ROOT / 'outputs/tables'
    quality = pd.concat(quality_rows, ignore_index=True)
    cues, summary, sources, counts = map(pd.DataFrame, (cue_rows, summary_rows, confounding, totals))
    for name, table in [('text_quality_audit', quality), ('text_source_cue_audit', cues), ('text_modeling_ready_summary', summary), ('text_source_mask_dictionary', dictionary), ('text_source_confounding_audit', sources), ('text_modeling_exclusion_summary', counts)]:
        table.to_csv(tables / (name + '.csv'), index=False)
    after = {str(p.relative_to(ROOT)): fingerprint(p) for p in paths}
    assert before == after
    metadata = dict(input_sha256=before, script_sha256=fingerprint(Path(__file__)), dictionary_sha256=fingerprint(tables / 'text_source_mask_dictionary.csv'),
        validation='passed', original_inputs_unchanged=True, no_split_reassignment=True, no_models_trained=True,
        feature_allowlist=list(FIELDS) + [f + '_source_masked' for f in FIELDS],
        dictionary_input_columns=list(META), dictionary_scope='All prepared cohort metadata in both datasets; label-free transductive fixed sensitivity dictionary; no text-derived/learned aliases.')
    (ROOT / 'outputs/reports/text_quality_and_source_cue_audit_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    report = [
        '# Text quality and source-cue audit', '',
        'Audit only; no classifier/vectorizer fitted and no publisher-held-out split created. Original cohort fields and split files are unchanged (SHA-256 verified). All rows remain in the new CSVs; use primary_modeling_eligible to select the frozen cohort.', '',
        markdown(counts), '',
        '## Frozen quality decisions', '',
        'Reviewed all 13 FakeHealth and 105 CoAID eligible mixed-artifact rows, using prepared text only. Specific article/claim headlines remain substantive even when bodies contain only branding; those bodies are ineligible for body-only sensitivity. Generic CDC section headings, error pages and social-profile names with boilerplate do not supply an article claim and are artifact-dominated. For FakeHealth branding bodies, >=25% of combined characters is major; shorter branding is minor. Two Video News rows contain substantive Disney business-news descriptions: retained as substantive, with possible topic/capture mismatch documented. No raw content was revisited.', '',
        markdown(quality[quality.mixed_artifact_component_flag].groupby(['dataset', 'artifact_audit_class']).size().rename('N').reset_index()), '',
        'Reviewed all 86 eligible FakeHealth titles not marked fetched_content. All are marked review_original_title, not review_editorial_title. Of these, 85 are ordinary article headlines without explicit ratings/verdicts (low leakage risk). One is a review-site funding announcement with an empty body: moderate provenance risk, excluded as non-article artifact, not counted as high-risk ground-truth disclosure. Skeptical reporting of scientific evidence does not itself disclose the dataset label. CoAID title provenance remains ambiguous between fetched and original source-title fallback; this is documented as uncertain, not asserted to be review leakage. The audit does not prove the absence of hidden artifacts in unflagged rows.', '',
        '## Source cues and masking', '',
        'Counts are conservative direct-string detections, not source-identity accuracy. A cue may name a cited outlet/institution rather than the publishing outlet. Categories overlap. Attribution templates require a matched source identifier plus by/copyright/rights/attribution syntax; generic personal bylines alone do not count. URL matching recognizes explicit http(s)/www hosts; bare known domains match the metadata dictionary. Unknown bare domains, indirect stylistic cues, unnamed boilerplate and unlisted spelling variants can remain.', '',
        markdown(cues[cues.category.eq('any_explicit_source_cue')]), '',
        'Cue categories in final combined text (nonexclusive counts):', '',
        markdown(cues[cues.scope.eq('modeling_ready') & cues.field.eq('combined_text') & cues.category.ne('any_explicit_source_cue')].sort_values(['dataset', 'cue_N'], ascending=[True, False])), '',
        'The dictionary uses only publisher_source, article_host and article_url from prepared cohorts, plus fixed identity aliases activated by metadata hosts. It never uses labels, ratings, engagement, review URLs, learned features or statistical class associations. Exact identifier matches use word boundaries; ambiguous TIME/STAT/WHO and Slate/Vox use case-sensitive forms. Overlapping spans are merged, then replaced once with [SOURCE]. Explicit URL hosts are masked syntactically, retaining paths and surrounding content. Identical text receives identical masking across datasets. This dictionary uses evaluation-set metadata, openly, as a prespecified transductive sensitivity transform; it is not an unseen-publisher production experiment. It removes explicit mentions, including legitimate references to institutions, but cannot eliminate publisher style, URL-path cues or collection confounding.', '',
        '## Source/class association and later publisher evaluation', '',
        markdown(sources), '',
        'Missing sources are counted separately, never treated as one publisher. Source identities here are the exact existing metadata categories; hosts/subdomains and publisher aliases are not silently consolidated. This can overstate distinct publishers, so a later publisher-held-out design needs an audited alias/host-family definition and joint resource-group barriers. Cramér’s V is not predictive performance. Class-exclusive sources are retained.', '',
        'A publisher-held-out experiment is potentially feasible because both datasets have class support across multiple source categories. CoAID has very limited mixed-class-source support, so pure publisher effects remain difficult to separate from class. Restricting to mixed-class sources would lose all exclusive-source rows plus missing-source rows (counts above); such restriction is neither required for holding out publishers nor recommended for this baseline. Actual class-balanced publisher/resource partition feasibility is not established here; crossing leakage groups and alias consolidation may reduce available partitions. No alternative split was generated.', '',
        '## Prespecified model inputs and validation', '',
        '1. PRIMARY / natural: combined_text on primary_modeling_eligible == True. Preserve title + body exactly as prepared. Exclude prior ineligible rows, newly artifact-dominated rows, and high-risk title leakage. Retain low/moderate/uncertain provenance rows unless independently artifact-dominated; do not filter by source purity.',
        '2. TITLE / BODY sensitivity: title_text on title_only_modeling_eligible, or body_text on body_only_modeling_eligible. These flags intersect primary eligibility with the existing field-specific eligibility. Sample sizes may differ; assignments do not. All usable fields preserve original language.',
        '3. SOURCE-MASKED sensitivity: combined_text_source_masked on the exact primary cohort. Optional corresponding title/body masked fields exist. Only these named text fields are predictors; labels, ratings, IDs, provenance, cue/quality flags, engagement and assignments must never be concatenated into model text.', '',
        'Use cv_fold >= 0 and cv_fold != f for fold-f training; cv_fold == f for validation. cv_fold == -1 is reserved holdout, and -2 denotes an originally ineligible row with no split. No split regeneration. Filtering preserved row/group assignments, train/holdout separation, and both classes in every training/validation/holdout partition for all conditions. Source masking can produce new identical strings across original groups; these reflect deliberate information removal and are not new resource identities. Existing group barriers remain unchanged.', '',
        markdown(summary[summary.condition.eq('primary_natural')]), '',
        'All condition-specific partition counts are in text_modeling_ready_summary.csv. Reproduce with `.venv/bin/python src/audit_text_quality_and_source_cues.py`; validate with `.venv/bin/python -m unittest discover -s src -p "test_text_quality_and_source_cues.py"`. Input fingerprints and dictionary scope are recorded in the adjacent metadata JSON. The first TF-IDF + Logistic Regression baseline is ready under this frozen protocol, with training-fold-only fitting and resource-generalization interpretation. No training is authorized by this audit.', '',
        'Outputs: data/processed/{fakehealth,coaid}_text_modeling_ready.csv; outputs/tables/text_quality_audit.csv, text_source_cue_audit.csv, text_modeling_ready_summary.csv, text_source_mask_dictionary.csv, text_source_confounding_audit.csv, text_modeling_exclusion_summary.csv.',
    ]
    (ROOT / 'outputs/reports/text_quality_and_source_cue_audit.md').write_text('\n'.join(report) + '\n')
    logging.info('Validation passed; all requested outputs written; inputs and split assignments unchanged')
    print(counts.to_json(orient='records'))
    print('Validation passed: original inputs unchanged; split/group integrity and both-class support preserved for every condition.')


if __name__ == '__main__':
    main()
