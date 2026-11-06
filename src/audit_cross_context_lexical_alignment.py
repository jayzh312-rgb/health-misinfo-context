"""Descriptive alignment of frozen natural-text LR coefficients; never fits models.

Run: .venv/bin/python src/audit_cross_context_lexical_alignment.py
No article text or target outcomes enter feature selection or importance cutoffs.
"""
from pathlib import Path
from contextlib import ExitStack, contextmanager
from unittest.mock import patch
import json
import platform
import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import scipy
from scipy.stats import pearsonr, spearmanr
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'lexical_alignment_matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import run_tfidf_logistic_baseline as baseline

ROOT = Path(__file__).resolve().parents[1]
TOL = 0.05  # Descriptive, fixed before inspecting coefficients, not significance.
TOP_NS = (1000, 500, 100)
CONFIGS = {'FakeHealth': (2, 5, 1.0), 'CoAID': (1, 2, 10.0)}
# Small, manually specified exact-token dictionary; deliberately non-exhaustive.
CONTEXT = {
    'covid_specific': {'covid', 'coronavirus', 'pandemic', 'quarantine', 'lockdown', 'sars', 'cov'},
    'election_political': {'trump', 'president', 'stimulus', 'election', 'biden', 'democrat', 'republican'},
    'generic_health': {'vaccine', 'vaccines', 'drug', 'drugs', 'treatment', 'disease', 'health',
                       'medical', 'patients', 'cancer', 'virus', 'infection'},
}
require = baseline.require
sha = baseline.sha


@contextmanager
def prohibit_fitting():
    """Fail immediately if any audit path attempts estimator fitting."""
    with ExitStack() as stack:
        for cls in (Pipeline, TfidfVectorizer, TfidfTransformer, LogisticRegression):
            for name in ('fit', 'fit_transform'):
                if hasattr(cls, name):
                    stack.enter_context(patch.object(cls, name, side_effect=RuntimeError('Audit prohibits fitting')))
        yield


def verify_inputs():
    meta = json.loads(baseline.META_PATH.read_text())
    require(meta['validation'] == 'passed', 'Baseline not validated; stop without refitting')
    sig = meta['signature']
    versions = dict(python=platform.python_version(), sklearn=sklearn.__version__,
                    numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__)
    require(versions == sig['versions'], 'Baseline environment changed; stop without refitting')
    require(sha(baseline.__file__) == sig['script_sha256'], 'Baseline code fingerprint failed')
    require(sig['protocol'] == baseline.PROTOCOL, 'Baseline protocol changed')
    # Hash only necessary artifacts, before deserializing any fitted model.
    paths = [baseline.STATE / 'selection_lock.json', baseline.STATE / 'run_signature.json',
             baseline.TABLES / 'tfidf_logistic_model_selection.csv',
             baseline.TABLES / 'tfidf_logistic_holdout_results.csv',
             baseline.TABLES / 'tfidf_logistic_top_features.csv',
             baseline.TABLES / 'text_source_mask_dictionary.csv']
    for prefix in baseline.PREFIXES.values():
        paths += [baseline.STATE / f'final_{prefix}_natural.joblib',
                  ROOT / f'data/processed/{prefix}_text_modeling_ready.csv']
    fingerprints = {}
    for path in paths:
        rel = str(path.relative_to(ROOT))
        expected = meta['outputs'].get(rel, sig['inputs'].get(rel))
        require(expected is not None and path.exists() and sha(path) == expected,
                f'Fingerprint failed: {rel}; stop without refitting')
        fingerprints[rel] = expected
    require(json.loads((baseline.STATE / 'run_signature.json').read_text()) == sig,
            'Run signature differs from validated metadata')
    fingerprints[str(baseline.META_PATH.relative_to(ROOT))] = sha(baseline.META_PATH)
    fingerprints[str(Path(baseline.__file__).relative_to(ROOT))] = sha(baseline.__file__)
    return fingerprints


def load_frozen():
    fingerprints = verify_inputs()
    selection = pd.read_csv(baseline.TABLES / 'tfidf_logistic_model_selection.csv')
    holdout = pd.read_csv(baseline.TABLES / 'tfidf_logistic_holdout_results.csv')
    locked = json.loads((baseline.STATE / 'selection_lock.json').read_text())['selections']
    features, vocab_sizes, models = {}, {}, {}
    for dataset, prefix in baseline.PREFIXES.items():
        bundle = joblib.load(baseline.STATE / f'final_{prefix}_natural.joblib')
        model, result = bundle['model'], bundle['result']
        vec, lr = model['tfidf'], model['logistic']
        ngram, min_df, c = CONFIGS[dataset]
        require(vec.ngram_range == (1, ngram) and vec.min_df == min_df and vec.analyzer == 'word', 'Wrong vectorizer')
        require(lr.C == c and lr.class_weight == 'balanced' and lr.solver == 'liblinear'
                and lr.l1_ratio == 0, 'Wrong LR configuration')
        require(np.array_equal(lr.classes_, [0, 1]) and lr.coef_.shape[0] == 1, 'Fake-positive orientation failed')
        selected = selection[selection.dataset.eq(dataset) & selection.condition.eq('natural') & selection.selected]
        require(len(selected) == 1, 'Missing unique model selection')
        for key, value in locked[dataset].items():
            require(result[key] == selected.iloc[0][key] == value, 'Selection mismatch')
        saved = holdout[holdout.dataset.eq(dataset) & holdout.condition.eq('natural') & holdout.model.eq('tfidf_logistic')]
        require(len(saved) == 1, 'Missing unique baseline result')
        for key in ('train_N', 'train_uid_sha256', 'train_group_sha256', 'vocabulary_sha256', 'selection_lock_sha256'):
            require(saved.iloc[0][key] == result[key], f'Baseline result mismatch: {key}')
        require(result['selection_lock_sha256'] == sha(baseline.STATE / 'selection_lock.json'), 'Selection lock mismatch')
        cols = ['content_uid', 'leakage_group', 'primary_modeling_eligible', 'cv_fold',
                'binary_analysis_target', 'binary_analysis_label']
        cohort = pd.read_csv(ROOT / f'data/processed/{prefix}_text_modeling_ready.csv', usecols=cols, keep_default_na=False)
        train = cohort[cohort.primary_modeling_eligible & cohort.cv_fold.ge(0)]
        require(train.content_uid.is_unique and len(train) == result['train_N'], 'Training identity mismatch')
        require(train.binary_analysis_target.eq(train.binary_analysis_label.eq('Fake').astype(int)).all()
                and set(train.binary_analysis_label) == {'Fake', 'Real'}, 'Fake=1 coding failed')
        require(baseline.digest(train.content_uid) == result['train_uid_sha256'], 'Training UID fingerprint failed')
        require(baseline.digest(train.leakage_group.unique()) == result['train_group_sha256'], 'Training group fingerprint failed')
        require(baseline.digest(vec.vocabulary_) == result['vocabulary_sha256'], 'Vocabulary fingerprint failed')
        require(vec.smooth_idf and vec.use_idf, 'DF inversion requires smoothed IDF')
        # sklearn's smoothed IDF = log((1 + N)/(1 + df)) + 1.
        inferred = (1 + len(train)) / np.exp(vec.idf_ - 1) - 1
        df = np.rint(inferred).astype(int)
        require(np.allclose(inferred, df, atol=1e-7, rtol=0) and np.all((df >= min_df) & (df <= len(train))), 'DF inversion failed')
        all_features = pd.DataFrame({'feature': vec.get_feature_names_out(),
                                     f'{prefix}_coefficient': lr.coef_[0],
                                     f'{prefix}_training_document_frequency': df})
        require(np.isfinite(lr.coef_).all(), 'Nonfinite coefficients')
        uni = all_features[~all_features.feature.str.contains(' ')].copy()
        uni[f'{prefix}_abs_coefficient'] = uni[f'{prefix}_coefficient'].abs()
        uni[f'{prefix}_sign'] = np.sign(uni[f'{prefix}_coefficient']).astype(int)
        # Rank among ALL unigrams of that dataset, before taking the intersection.
        uni[f'{prefix}_abs_rank_unigram_vocabulary'] = uni[f'{prefix}_abs_coefficient'].rank(method='min', ascending=False).astype(int)
        features[prefix] = uni
        vocab_sizes[dataset] = len(uni)
        models[dataset] = model
    shared = features['fakehealth'].merge(features['coaid'], on='feature', validate='one_to_one').sort_values('feature').reset_index(drop=True)
    require(shared.feature.tolist() == sorted(set(features['fakehealth'].feature) & set(features['coaid'].feature)), 'Nondeterministic intersection')
    for prefix in baseline.PREFIXES.values():
        shared[f'{prefix}_abs_rank_shared'] = shared[f'{prefix}_abs_coefficient'].rank(method='min', ascending=False).astype(int)
    shared['min_abs_coefficient'] = shared[['fakehealth_abs_coefficient', 'coaid_abs_coefficient']].min(axis=1)
    shared['max_abs_coefficient'] = shared[['fakehealth_abs_coefficient', 'coaid_abs_coefficient']].max(axis=1)
    # Cross-check saved baseline feature rows without reprocessing article text.
    old = pd.read_csv(baseline.TABLES / 'tfidf_logistic_top_features.csv', keep_default_na=False)
    for dataset, prefix in baseline.PREFIXES.items():
        check = old[old.dataset.eq(dataset)].merge(shared, on='feature')
        require(len(check) > 0 and np.allclose(check.coefficient, check[f'{prefix}_coefficient'], rtol=1e-12, atol=1e-12), 'Saved coefficient mismatch')
        require(np.array_equal(check.training_document_frequency, check[f'{prefix}_training_document_frequency']), 'Saved DF mismatch')
    return shared, vocab_sizes, models, fingerprints


def flag_features(shared, model):
    shared = shared.copy()
    dictionary = pd.read_csv(baseline.TABLES / 'text_source_mask_dictionary.csv', keep_default_na=False)
    tokenizer = model['tfidf'].build_tokenizer()
    exact, fragments = {}, {}
    for identifier in sorted(set(dictionary.identifier)):
        tokens = tokenizer(identifier.lower())
        normalized = ' '.join(tokens)
        if len(normalized) >= 3:
            exact.setdefault(normalized, []).append(identifier)
        for token in set(tokens):
            if len(token) >= 3:
                fragments.setdefault(token, []).append(identifier)
    shared['explicit_source_identifier'] = shared.feature.isin(exact)
    shared['source_identifier_fragment'] = shared.feature.isin(fragments)
    shared['source_identifier_matches'] = shared.feature.map(lambda x: json.dumps(exact.get(x, [])))
    shared['source_fragment_matches'] = shared.feature.map(lambda x: json.dumps(fragments.get(x, [])))
    for category, words in CONTEXT.items():
        shared[category] = shared.feature.isin(words)
    shared['any_context_flag'] = shared[list(CONTEXT)].any(axis=1)
    return shared


def importance_subsets(shared):
    """Prespecified, coefficient-only cutoffs; no text, labels or outcomes accepted."""
    median_mask = ((shared.fakehealth_abs_coefficient > shared.fakehealth_abs_coefficient.median()) |
                   (shared.coaid_abs_coefficient > shared.coaid_abs_coefficient.median()))
    ranked = shared.sort_values(['max_abs_coefficient', 'feature'], ascending=[False, True])
    return [('all', shared), ('above_median_either', shared[median_mask])] + [(f'top{n}', ranked.head(n)) for n in TOP_NS]


def summarize(name, frame):
    a, b = frame.fakehealth_coefficient, frame.coaid_coefficient
    sa, sb = np.sign(a), np.sign(b)
    row = dict(cutoff=name, N=len(frame), pearson=float(pearsonr(a, b).statistic),
               spearman=float(spearmanr(a, b).statistic),
               absolute_magnitude_spearman=float(spearmanr(a.abs(), b.abs()).statistic))
    masks = dict(sign_agreement=sa.eq(sb), sign_reversal=(a*b < 0),
                 positive_both=(a > 0) & (b > 0), negative_both=(a < 0) & (b < 0),
                 positive_fh_negative_ca=(a > 0) & (b < 0), negative_fh_positive_ca=(a < 0) & (b > 0),
                 either_exact_zero=(a == 0) | (b == 0))
    for key, mask in masks.items():
        row[key + '_N'] = int(mask.sum())
        row[key + '_percent'] = float(100 * mask.mean())
    return row


def ranked_categories(shared):
    meaningful = shared.min_abs_coefficient.ge(TOL)
    product = shared.fakehealth_coefficient * shared.coaid_coefficient
    masks = {'strong_sign_reversals': meaningful & product.lt(0),
             'stable_fake_features': meaningful & shared.fakehealth_coefficient.gt(0) & shared.coaid_coefficient.gt(0),
             'stable_real_features': meaningful & shared.fakehealth_coefficient.lt(0) & shared.coaid_coefficient.lt(0)}
    return {name: shared[mask].sort_values(['min_abs_coefficient', 'feature'], ascending=[False, True]).reset_index(drop=True)
            for name, mask in masks.items()}


def transfer_stability(shared):
    summary, details = [], []
    for source, target in [('fakehealth', 'coaid'), ('coaid', 'fakehealth')]:
        for association, sign in [('Fake', 1), ('Real', -1)]:
            chosen = shared[shared[f'{source}_sign'].eq(sign)].sort_values(
                [f'{source}_abs_coefficient', 'feature'], ascending=[False, True]).head(25).copy()
            chosen['target_status'] = np.where(chosen[f'{target}_abs_coefficient'] < TOL, 'near_zero',
                np.where(chosen[f'{target}_sign'].eq(sign), 'same_direction', 'opposite_direction'))
            chosen['source_dataset'], chosen['source_association'] = source, association
            chosen['source_rank'] = np.arange(1, len(chosen)+1)
            counts = chosen.target_status.value_counts()
            summary.append(dict(source_dataset=source, source_association=association, selected_N=len(chosen),
                **{k+'_N': int(counts.get(k, 0)) for k in ('same_direction', 'opposite_direction', 'near_zero')}))
            details.append(chosen)
    return pd.DataFrame(summary), pd.concat(details, ignore_index=True)


def enrichment(shared, categories):
    groups = {'all_shared': shared, 'meaningful_both': shared[shared.min_abs_coefficient.ge(TOL)]}
    for name, frame in categories.items():
        groups[name + '_all'] = frame
        groups[name + '_top100'] = frame.head(100)
    rows = []
    for name, frame in groups.items():
        for flag in ['explicit_source_identifier', 'source_identifier_fragment', *CONTEXT, 'any_context_flag']:
            percent = 100 * frame[flag].mean() if len(frame) else 0.
            base = 100 * shared[flag].mean()
            rows.append(dict(group=name, flag=flag, N=len(frame), flagged_N=int(frame[flag].sum()),
                             percent=percent, all_shared_percent=base, difference_percentage_points=percent-base))
    return pd.DataFrame(rows)


def validate_tables(shared, summary, categories, stability, details):
    require(shared.feature.is_unique and shared.feature.str.split().str.len().eq(1).all(), 'Unigram/unique key violation')
    require(summary.cutoff.tolist() == ['all', 'above_median_either', 'top1000', 'top500', 'top100'], 'Cutoff protocol changed')
    require(summary.cutoff.is_unique, 'Duplicate cutoff')
    require(not stability.duplicated(['source_dataset', 'source_association']).any(), 'Duplicate stability key')
    require(not details.duplicated(['source_dataset', 'source_association', 'feature']).any(), 'Duplicate feature stability key')
    require(stability[['same_direction_N', 'opposite_direction_N', 'near_zero_N']].sum(axis=1).eq(stability.selected_N).all(), 'Stability counts incomplete')
    for name, frame in categories.items():
        require(frame.feature.is_unique and frame.min_abs_coefficient.ge(TOL).all(), f'Invalid category: {name}')
    quadrants = ['positive_both_N', 'negative_both_N', 'positive_fh_negative_ca_N', 'negative_fh_positive_ca_N', 'either_exact_zero_N']
    require(summary[quadrants].sum(axis=1).eq(summary.N).all(), 'Sign accounting failed')
    require(categories['strong_sign_reversals'].eval('fakehealth_coefficient * coaid_coefficient < 0').all(), 'Reversal orientation failed')


def make_figures(shared, summary, directory):
    directory.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(shared.fakehealth_coefficient, shared.coaid_coefficient, alpha=.22, s=9, rasterized=True)
    ax.axhline(0, color='black', lw=.7)
    ax.axvline(0, color='black', lw=.7)
    ax.set(xlabel='FakeHealth LR coefficient (positive = Fake)', ylabel='CoAID LR coefficient (positive = Fake)',
           title=f'Shared unigrams: {len(shared):,}')
    fig.tight_layout()
    fig.savefig(directory / 'shared_unigram_coefficient_scatter.png', dpi=180)
    plt.close(fig)
    shown = summary.set_index('cutoff').loc[['all', 'top1000', 'top500', 'top100']]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(shown.index, shown.sign_agreement_percent, color='#407b9d')
    ax.set(ylim=(0, 100), ylabel='Sign agreement (%)', xlabel='Importance by maximum absolute coefficient')
    for i, value in enumerate(shown.sign_agreement_percent):
        ax.text(i, value+1.5, f'{value:.1f}%', ha='center')
    fig.tight_layout()
    fig.savefig(directory / 'sign_agreement_by_importance.png', dpi=180)
    plt.close(fig)


def write_report(path, shared, sizes, summary, categories, stability, source_summary):
    md = baseline.md
    both_n = int(shared.min_abs_coefficient.ge(TOL).sum())
    rev = categories['strong_sign_reversals']
    global_row = summary.iloc[0]
    agreement = ', '.join(f'{r.cutoff}: {r.sign_agreement_percent:.2f}%' for r in summary.itertuples())
    exact_rev = int(rev.explicit_source_identifier.sum())
    context_rev = int(rev.any_context_flag.sum())
    lines = ['# Cross-context lexical feature-alignment audit', '',
        'This is a descriptive audit of the two validated natural-text within-dataset models. No model was fitted, tuned, or used to rerun transfer. Positive coefficients predict Fake=1; negative coefficients predict Real=0.', '',
        '## Vocabulary and methods', '',
        *[f'- {ds}: {n:,} unigrams; {len(shared)/n:.2%} represented in the shared vocabulary.' for ds, n in sizes.items()],
        f'- Shared: {len(shared):,} unigrams. FakeHealth bigrams are excluded from comparisons, but remain part of the fitted model conditioning its unigram coefficients.',
        '- FakeHealth: unigrams+bigrams, min_df=5, C=1. CoAID: unigrams, min_df=2, C=10. Both use balanced L2/liblinear.',
        '- Document frequencies are recovered exactly (checked against integer bounds and existing top-feature rows) from saved smooth IDF and verified training N; no article text is read.',
        '- Absolute ranks use minimum rank for ties; both full unigram-vocabulary ranks and shared-vocabulary ranks are saved. Sorted lexical order breaks top-N ties deterministically.',
        '- The median cutoff uses each coefficient absolute-value median within the shared vocabulary, strictly exceeded in either dataset. Top-N uses maximum absolute coefficient. All cutoffs are fixed independently of target outcomes.',
        f'- Near zero means |beta| < {TOL}. Meaningful in both means min(|beta_FH|, |beta_CA|) >= {TOL}; “strong” is descriptive, not a statistical significance claim. Category rankings use this minimum.', '',
        '## Alignment at every prespecified cutoff', '',
        md(summary[['cutoff', 'N', 'pearson', 'spearman', 'absolute_magnitude_spearman', 'sign_agreement_percent', 'sign_reversal_percent']]), '',
        'Pearson and Spearman above describe signed coefficients; absolute-magnitude Spearman separately describes importance-rank stability. Quadrant counts and percentages, including exact-zero accounting, are in the summary CSV.', '',
        f'Strong reversals: {len(rev):,}/{len(shared):,} shared features ({len(rev)/len(shared):.2%}), or {len(rev):,}/{both_n:,} meaningful-in-both features ({len(rev)/both_n:.2%}).', '',
        '## Top-feature transfer stability', '', md(stability), '',
        'The detail CSV reports every selected source feature and its target coefficient/status. Near-zero target coefficients take precedence over sign comparisons.', '']
    for name, frame in categories.items():
        lines += [f'## {name.replace("_", " ").capitalize()}', '',
                  f'{len(frame):,} eligible features; CSV saves up to 100 ranked features. First 10:', '',
                  md(frame.head(10)[['feature', 'fakehealth_coefficient', 'coaid_coefficient', 'explicit_source_identifier', 'any_context_flag']]), '']
    lines += ['## Source identifiers and transparent context flags', '',
        'The unchanged baseline mask dictionary is used label-blind. Full tokenized identifiers (at least three characters) are distinguished from component/fragment matches, following baseline rules. Components can be ordinary words; neither flag establishes actual publisher attribution. Dictionary coverage is incomplete and includes institutional mentions. No new source dictionary or masking is introduced.', '',
        md(source_summary[source_summary.flag.eq('explicit_source_identifier')][['group', 'N', 'flagged_N', 'percent', 'difference_percentage_points']]), '',
        'Differences are descriptive percentage-point differences from all shared unigrams, without enrichment significance tests. The supplemental CSV also provides the meaningful-in-both comparison pool, broad fragments, each context category, and top-100 versus full-category counts.', '',
        'Manual exact-token context dictionary (fixed before coefficient inspection):', '',
        *[f'- {key}: {", ".join(sorted(words))}.' for key, words in CONTEXT.items()], '',
        md(source_summary[source_summary.flag.eq('any_context_flag')][['group', 'N', 'flagged_N', 'percent']]), '',
        'This small dictionary identifies examples; its low coverage cannot rule out topic/context dependence, and generic health terms do not uniquely identify a context.', '',
        '## Interpretation', '',
        f'Coefficient directions are weakly aligned: overall agreement is {global_row.sign_agreement_percent:.2f}%, with essentially zero signed Pearson ({global_row.pearson:.4f}) and Spearman ({global_row.spearman:.4f}) correlations. Absolute-magnitude rank correlation is weak ({global_row.absolute_magnitude_spearman:.4f}). Agreement by cutoff is {agreement}; there is no consistent improvement among more important features. These percentages are descriptive, not a test against a 50% null, because the marginal sign distributions differ.', '',
        f'Reversals are common among features meaningful in both models ({len(rev)/both_n:.2%}). Only {exact_rev}/{len(rev)} ({100*exact_rev/len(rev):.2f}%) are full known source identifiers, versus {100*shared.explicit_source_identifier.mean():.2f}% of all shared features; none of the top 100 reversals are exact identifiers. Thus known explicit identifiers do not account for much of the reversal. The stable Fake set has {int(categories["stable_fake_features"].explicit_source_identifier.sum())} exact identifiers and the stable Real set has {int(categories["stable_real_features"].explicit_source_identifier.sum())}; the full stable Real set is descriptively enriched relative to the shared pool, but the counts are small and its top 100 contain none.', '',
        f'The manual context flags identify {context_rev}/{len(rev)} reversals and {int(rev.head(100).any_context_flag.sum())}/100 of the strongest reversals (vaccine). Health/topic vocabulary supplies a concrete instability example, but these narrow flags do not establish that reversals are strongly or predominantly topic-related. The strongest reversals also include function words (and, to, of) and reporting terms (release, says, news), consistent with multiple possible distributional or stylistic differences rather than a uniquely identified mechanism. COVID-specific tokens absent from the intersection are outside this audit.', '',
        'Signed correlations and agreement rates above quantify whether lexical associations align. Importance-cutoff results must be read together rather than selecting a favorable cutoff. Stable lists identify associations in these two datasets only, not universal misinformation markers.', '',
        'Opposing coefficients plausibly contribute to transfer degradation, but this comparison does not isolate their contribution or establish that they explain the collapse. Coefficients are conditional on different vocabularies, correlated predictors, IDF scales, document distributions, regularization strengths, and label constructions. Maximum-absolute-coefficient ranking can favor the less regularized CoAID model. Raw coefficient magnitudes are not standardized, directly comparable causal effects.', '',
        'The models were trained only on their respective baseline training partitions. Comparing their fitted associations is a retrospective audit, not a feature-selection procedure for deployment. Cross-context differences do not prove a COVID-period effect, a causal sensitivity effect, or a universal misinformation mechanism.', '',
        'Optional transfer-error linkage was skipped: it is not needed for the primary audit and would require additional prediction/cohort joins. Transfer fitting and evaluation were not rerun.', '',
        '## Validation and outputs', '',
        'Validation passed: baseline model/input/selection fingerprints, environment and protocol, training UID/group/vocabulary fingerprints, Fake=1 class orientation, unigram-only deterministic intersection, recovered document frequencies, prespecified coefficient-only cutoffs, unique table keys, and sign accounting. Runtime guards prohibit estimator fitting throughout the audit. Input fingerprints are rechecked after writing outputs.', '',
        'Script: `src/audit_cross_context_lexical_alignment.py`. Tables: `outputs/tables/text_shared_unigram_coefficients.csv`, `text_coefficient_alignment_summary.csv`, `text_strong_sign_reversals.csv`, `text_stable_fake_features.csv`, `text_stable_real_features.csv`, `text_top_feature_transfer_stability.csv`; supplemental `text_top_feature_transfer_details.csv` and `text_feature_alignment_flag_summary.csv`. Figures: `outputs/figures/text_feature_alignment/`. Adjacent JSON records thresholds, fingerprints and validation.', '']
    path.write_text('\n'.join(lines))


def main():
    with prohibit_fitting():
        shared, sizes, models, fingerprints = load_frozen()
        shared = flag_features(shared, models['CoAID'])
        summary = pd.DataFrame([summarize(name, frame) for name, frame in importance_subsets(shared)])
        categories = ranked_categories(shared)
        stability, details = transfer_stability(shared)
        flags = enrichment(shared, categories)
        validate_tables(shared, summary, categories, stability, details)
        tables = {'shared_unigram_coefficients': shared, 'coefficient_alignment_summary': summary,
                  **{name: frame.head(100) for name, frame in categories.items()},
                  'top_feature_transfer_stability': stability, 'top_feature_transfer_details': details,
                  'feature_alignment_flag_summary': flags}
        paths = []
        for name, frame in tables.items():
            path = ROOT / f'outputs/tables/text_{name}.csv'
            frame.to_csv(path, index=False)
            paths.append(path)
        figures = ROOT / 'outputs/figures/text_feature_alignment'
        make_figures(shared, summary, figures)
        report = ROOT / 'outputs/reports/cross_context_lexical_alignment.md'
        write_report(report, shared, sizes, summary, categories, stability, flags)
        require(verify_inputs() == fingerprints, 'Protected inputs changed during audit')
        paths += [report, *sorted(figures.glob('*.png'))]
        baseline.write_json(report.with_suffix('.json'), dict(validation='passed', refit_count=0,
            fit_guard_enabled=True, script_sha256=sha(__file__), inputs=fingerprints,
            outputs={str(p.relative_to(ROOT)): sha(p) for p in paths},
            vocabulary_sizes=sizes, shared_unigrams=len(shared), near_zero_tolerance=TOL,
            top_ns=list(TOP_NS), median_scope='shared unigrams; strict greater-than in either dataset',
            context_dictionary={key: sorted(value) for key, value in CONTEXT.items()},
            meaningful_both_N=int(shared.min_abs_coefficient.ge(TOL).sum()),
            category_counts={key: len(value) for key, value in categories.items()},
            optional_transfer_error_linkage='skipped; not required for primary audit'))
    print(summary[['cutoff', 'N', 'pearson', 'spearman', 'sign_agreement_percent']].to_string(index=False))
    print('Validation passed; refits=0. Report: outputs/reports/cross_context_lexical_alignment.md')


if __name__ == '__main__':
    main()
