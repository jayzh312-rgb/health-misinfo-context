"""Metadata-only unseen-publisher feasibility; no models or final split files.

Run: .venv/bin/python src/audit_publisher_heldout_feasibility.py
Publisher identities are built without labels. Labels enter support counts and
count-stratified candidate partitions only. Candidate assignments stay in memory.
"""
from pathlib import Path
from urllib.parse import urlsplit
from contextlib import ExitStack
from unittest.mock import patch
import hashlib
import json
import re
import unicodedata
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from audit_text_quality_and_source_cues import ALIASES

ROOT = Path(__file__).resolve().parents[1]
PREFIXES = {'FakeHealth': 'fakehealth', 'CoAID': 'coaid'}
META = ['dataset', 'content_uid', 'publisher_source', 'article_host', 'article_url']
COLUMNS = META + ['primary_modeling_eligible', 'binary_analysis_target', 'binary_analysis_label', 'leakage_group']
PLATFORMS = {'youtube.com', 'youtu.be', 'facebook.com', 'twitter.com', 'x.com', 'medium.com',
             'wordpress.com', 'blogspot.com', 'substack.com', 'tumblr.com', 't.co'}
SEED = 42
HOLDOUT_PERMUTATIONS = 2048
TABLES = ROOT / 'outputs/tables'
REPORT = ROOT / 'outputs/reports/publisher_heldout_feasibility.md'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(values):
    return hashlib.sha256(json.dumps(sorted(map(str, values))).encode()).hexdigest()


def normalize_name(value):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(value)).casefold()).strip()


def host(value):
    value = str(value).strip()
    if not value:
        return ''
    try:
        parsed = urlsplit(value if '://' in value else 'https://' + value)
        h = (parsed.hostname or '').casefold().rstrip('.')
        return h.removeprefix('www.') if '.' in h and not re.search(r'\s', h) else ''
    except ValueError:
        return ''


def domain_source(value):
    value = str(value).strip()
    return host(value) if re.fullmatch(r'(?:https?://)?[\w.-]+\.[a-zA-Z]{2,}/?', value) else ''


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parent[max(a, b)] = min(a, b)


def platform_account(h, url):
    """Only explicit account identifiers, never opaque video/post IDs."""
    path = urlsplit(url if '://' in str(url) else 'https://' + str(url)).path
    bits = [x for x in path.split('/') if x]
    if h in {'twitter.com', 'x.com'} and len(bits) >= 3 and bits[1] == 'status' and bits[0].casefold() not in {'i', 'intent', 'share'}:
        return h + '/account/' + bits[0].casefold()
    if h == 'youtube.com' and bits:
        if bits[0].startswith('@'):
            return h + '/account/' + bits[0].casefold()
        if len(bits) >= 2 and bits[0] in {'channel', 'user', 'c'}:
            return h + '/' + bits[0] + '/' + bits[1]  # Channel identifiers can be case sensitive.
    return ''


def build_family_map(metadata):
    """Strict metadata allowlist makes identity invariant to label changes."""
    metadata = metadata.loc[:, META].copy().sort_values(['dataset', 'content_uid'])
    observed = {h for value in pd.concat([metadata.article_host, metadata.article_url]) for h in [host(value)] if h}
    observed |= {h for h in metadata.publisher_source.map(domain_source) if h}
    roots = observed | {r for r in ALIASES if any(h == r or h.endswith('.'+r) for h in observed)}

    def host_family(h):
        if not h:
            return ''
        # Never merge hosted tenants or a newsroom into a platform/container.
        # Explicit audited publishing identities take precedence over a broader
        # container host (for example abcnews.go.com must not become go.com).
        audited_parents = [p for p in ALIASES if p not in PLATFORMS and (h == p or h.endswith('.'+p))]
        if audited_parents:
            return max(audited_parents, key=lambda x: (x.count('.'), len(x), x))
        parents = [p for p in roots if p not in PLATFORMS and (h == p or h.endswith('.'+p))]
        if any(h.endswith('.'+p) for p in PLATFORMS):
            return h
        return min(parents, key=lambda x: (x.count('.'), len(x), x)) if parents else h

    alias = {}
    for root, names in ALIASES.items():
        if root in roots and root not in PLATFORMS:
            for name in names:
                alias.setdefault(normalize_name(name), set()).add(host_family(root))
    prepared = []
    host_names = {}
    for r in metadata.itertuples(index=False):
        raw_host = host(r.article_host) or host(r.article_url) or domain_source(r.publisher_source)
        h = host_family(raw_host)
        name = normalize_name(r.publisher_source) if not domain_source(r.publisher_source) else ''
        audited = next(iter(alias[name])) if name in alias and len(alias[name]) == 1 else ''
        name_node = 'host:' + audited if audited else 'name:' + name if name else ''
        rules = ['Unicode NFKC, casefold and whitespace for names; lowercase host and www removal']
        if raw_host != h:
            rules.append('observed parent host or previously audited identity host')
        if audited:
            rules.append('exact previously audited metadata alias')
        status, node = 'identified', ''
        if h in PLATFORMS:
            account = platform_account(h, r.article_url)
            if account:
                node = 'account:' + account
                rules.append('explicit platform account in prepared URL; account proxy, not corporate ownership')
                status = 'identified_account_proxy'
            else:
                status = 'unresolved_platform_publisher'
                rules.append('hosting platform is not an identified publisher; no account inferred')
        elif h or name_node:
            # Website/source-family identity takes precedence over conflicting
            # attribution names on that same site. Do not split an exact host.
            node = 'host:' + h if h else name_node
            if h and name_node:
                host_names.setdefault(h, set()).add(name_node)
        else:
            status = 'missing_source_metadata'
            rules.append('no usable publisher name, host or URL host')
        prepared.append(dict(dataset=r.dataset, content_uid=r.content_uid, original_source=r.publisher_source,
            original_article_host=r.article_host, original_article_url=r.article_url,
            normalized_source_name=name, normalized_host=h, name_node=name_node, provisional_node=node,
            identity_status=status, rule_evidence='; '.join(rules)))
    union = UnionFind()
    for r in prepared:
        if r['provisional_node']:
            union.find(r['provisional_node'])
    linked_hosts = set()
    for h, names in sorted(host_names.items()):
        if len(names) == 1 and not (h in ALIASES and next(iter(names)).startswith('host:') and next(iter(names)) != 'host:' + h):
            union.union('host:' + h, next(iter(names)))
            linked_hosts.add(h)
    rows = []
    for r in prepared:
        h, node = r['normalized_host'], r.pop('provisional_node')
        r.pop('name_node')
        r['publisher_family'] = union.find(node) if node else ''
        r['ambiguous_host_name_relationship'] = len(host_names.get(h, set())) > 1
        if h in linked_hosts and node:
            r['rule_evidence'] += '; unique prepared name-host association; exact normalized names link observed host variants'
        elif r['ambiguous_host_name_relationship']:
            r['rule_evidence'] += '; exact publication host retained; conflicting attribution names do not link other hosts'
        elif len(host_names.get(h, set())) == 1 and h not in linked_hosts:
            r['rule_evidence'] += '; conflicting audited identity hosts not merged'
        r['identity_resolved'] = bool(node)
        rows.append(r)
    return pd.DataFrame(rows).sort_values(['dataset', 'content_uid']).reset_index(drop=True)


def input_paths():
    return ([ROOT / f'data/processed/{p}_text_modeling_ready.csv' for p in PREFIXES.values()] +
            [TABLES / 'text_modeling_ready_summary.csv', TABLES / 'text_source_confounding_audit.csv',
             ROOT / 'src/audit_text_quality_and_source_cues.py'])


def load_rows():
    parts = []
    summary = pd.read_csv(TABLES / 'text_modeling_ready_summary.csv')
    for dataset, prefix in PREFIXES.items():
        d = pd.read_csv(ROOT / f'data/processed/{prefix}_text_modeling_ready.csv', usecols=COLUMNS, keep_default_na=False)
        require(d.primary_modeling_eligible.dtype == bool, 'Eligibility not boolean')
        d = d[d.primary_modeling_eligible].copy()
        require(d.content_uid.is_unique and d.dataset.eq(dataset).all(), 'Cohort identity failed')
        require(d.binary_analysis_target.eq(d.binary_analysis_label.eq('Fake').astype(int)).all(), 'Label coding failed')
        require(d.leakage_group.ne('').all(), 'Missing leakage group')
        old = summary[summary.dataset.eq(dataset) & summary.condition.eq('primary_natural') & summary.partition.eq('all')]
        require(len(old) == 1 and len(d) == old.iloc[0].N and d.binary_analysis_target.sum() == old.iloc[0].fake_N, 'Validated cohort count mismatch')
        parts.append(d)
    return pd.concat(parts, ignore_index=True).sort_values(['dataset', 'content_uid']).reset_index(drop=True)


def support_table(rows):
    out = []
    for (ds, family, status), part in rows.assign(support_status=np.where(rows.identity_resolved, 'identified', rows.identity_status)).groupby(['dataset', 'publisher_family', 'support_status'], dropna=False):
        fake = int(part.binary_analysis_target.sum())
        real = len(part)-fake
        out.append(dict(dataset=ds, publisher_family=family, identity_status=status,
            row_type='publisher_family' if family else 'unresolved_source', N=len(part), fake_N=fake, real_N=real,
            label_support='mixed' if fake and real else 'fake_only' if fake else 'real_only',
            fake_prevalence=fake/len(part), leakage_groups=part.leakage_group.nunique()))
    return pd.DataFrame(out)


def resource_components(rows):
    """Connected components of publisher families and existing resource groups.

    Any component containing an unidentified source is ineligible for a strict
    unseen-publisher design. This conservatively avoids splitting either barrier.
    """
    union = UnionFind()
    for r in rows.itertuples():
        resource = 'resource:' + r.leakage_group
        union.find(resource)
        if r.publisher_family:
            union.union(resource, 'family:' + r.publisher_family)
    out = rows.copy()
    out['constraint_component'] = [union.find('resource:' + x) for x in out.leakage_group]
    unresolved = set(out.loc[~out.identity_resolved, 'constraint_component'])
    out['strict_publisher_eligible'] = ~out.constraint_component.isin(unresolved)
    conflicts = []
    for resource, part in out.groupby('leakage_group'):
        families = sorted(set(part.publisher_family)-{''})
        if len(families) > 1 or (families and not part.identity_resolved.all()):
            conflicts.append(dict(dataset=part.dataset.iloc[0], leakage_group=resource, N=len(part),
                publisher_family_N=len(families), publisher_families=json.dumps(families),
                unresolved_N=int((~part.identity_resolved).sum()),
                conflict_type='multiple_identified_families' if len(families) > 1 else 'identified_plus_unresolved',
                resolution='indivisible publisher-resource component; exclude whole component if any source unresolved'))
    return out, conflicts


def publisher_summary(rows, support):
    out = []
    for ds, d in rows.groupby('dataset'):
        s = support[support.dataset.eq(ds) & support.row_type.eq('publisher_family')]
        largest = s.sort_values(['N', 'publisher_family'], ascending=[False, True]).iloc[0]
        row = dict(dataset=ds, cohort_N=len(d), fake_N=int(d.binary_analysis_target.sum()), real_N=int(d.binary_analysis_target.eq(0).sum()),
            publisher_families=len(s), missing_source_rows=int(d.identity_status.eq('missing_source_metadata').sum()),
            account_proxy_families=int(s.publisher_family.str.startswith('account:').sum()),
            unresolved_platform_rows=int(d.identity_status.eq('unresolved_platform_publisher').sum()),
            identified_account_proxy_rows=int(d.identity_status.eq('identified_account_proxy').sum()),
            identified_rows=int(d.identity_resolved.sum()), strict_eligible_N=int(d.strict_publisher_eligible.sum()),
            resource_connected_exclusion_N=int((d.identity_resolved & ~d.strict_publisher_eligible).sum()),
            median_rows_per_publisher=float(s.N.median()), p90_rows_per_publisher=float(s.N.quantile(.9)),
            largest_publisher_N=int(largest.N), largest_publisher_fake_N=int(largest.fake_N), largest_publisher_real_N=int(largest.real_N),
            largest_publisher_family=largest.publisher_family,
            ambiguous_host_name_rows=int(d.ambiguous_host_name_relationship.sum()))
        for kind in ['fake_only', 'real_only', 'mixed']:
            part = s[s.label_support.eq(kind)]
            row[kind+'_families'] = len(part)
            row[kind+'_rows'] = int(part.N.sum())
            row[kind+'_fake_N'] = int(part.fake_N.sum())
            row[kind+'_real_N'] = int(part.real_N.sum())
        for label in ['fake', 'real']:
            counts = s[label+'_N'].sort_values(ascending=False)
            row[label+'_support_families'] = int(counts.gt(0).sum())
            row[label+'_largest_family_share'] = float(counts.iloc[0]/counts.sum()) if counts.sum() else 0.
            row[label+'_top3_family_share'] = float(counts.head(3).sum()/counts.sum()) if counts.sum() else 0.
        out.append(row)
    return pd.DataFrame(out)


def restriction_sizes(rows, support):
    out = []
    for ds, d in rows.groupby('dataset'):
        s = support[support.dataset.eq(ds) & support.row_type.eq('publisher_family')]
        for name, threshold in [('all_identified', 0), ('mixed_publishers', 1), ('min5_each', 5), ('min10_each', 10), ('min20_each', 20)]:
            allowed = set(s.loc[s.fake_N.ge(threshold) & s.real_N.ge(threshold), 'publisher_family'])
            direct = d[d.publisher_family.isin(allowed)]
            valid_components = [c for c, part in d.groupby('constraint_component')
                                if part.strict_publisher_eligible.all() and set(part.publisher_family) <= allowed]
            safe = d[d.constraint_component.isin(valid_components)]
            for scope, part in [('direct_restriction', direct), ('intact_publisher_resource_components', safe)]:
                out.append(dict(dataset=ds, restriction=name, min_fake_per_family=threshold, min_real_per_family=threshold,
                    scope=scope, N=len(part), fake_N=int(part.binary_analysis_target.sum()), real_N=int(part.binary_analysis_target.eq(0).sum()),
                    publisher_families=part.publisher_family.nunique(), original_N=len(d), lost_N=len(d)-len(part),
                    lost_fake_N=int(d.binary_analysis_target.sum()-part.binary_analysis_target.sum()),
                    lost_real_N=int(d.binary_analysis_target.eq(0).sum()-part.binary_analysis_target.eq(0).sum()),
                    retained_percent=100*len(part)/len(d), constraint_components=part.constraint_component.nunique()))
    return pd.DataFrame(out)


def partition_diagnostics(part, family_support):
    n, fake = len(part), int(part.binary_analysis_target.sum())
    counts = part.groupby('publisher_family').binary_analysis_target.agg(['size', 'sum'])
    real_counts = counts['size']-counts['sum']
    exclusive = set(family_support.loc[family_support.label_support.ne('mixed'), 'publisher_family'])
    fshare = float(counts['sum'].max()/fake) if fake else 1.
    rshare = float(real_counts.max()/(n-fake)) if n-fake else 1.
    exclusive_share = float(part.publisher_family.isin(exclusive).mean()) if n else 0.
    return dict(N=n, fake_N=fake, real_N=n-fake, publisher_families=part.publisher_family.nunique(),
        constraint_components=part.constraint_component.nunique(), fake_prevalence=fake/n if n else 0.,
        both_labels=bool(fake and n-fake), low_class_support=bool(min(fake, n-fake) < 20),
        fake_support_publishers=int(counts['sum'].gt(0).sum()), real_support_publishers=int(real_counts.gt(0).sum()),
        largest_fake_publisher_share=fshare, largest_real_publisher_share=rshare,
        class_exclusive_row_share=exclusive_share,
        source_class_proxy_risk=bool(exclusive_share >= .8 or fshare >= .5 or rshare >= .5),
        roc_auc_computable=bool(fake and n-fake), ap_two_class_interpretable=bool(fake and n-fake),
        average_precision_numerically_computable=bool(n),
        macro_f1_interpretable=bool(fake >= 20 and n-fake >= 20),
        macro_f1_caveat='diagnostic only: both classes >=20 does not resolve publisher-class confounding')


def validate_partition(full, train, test):
    require(set(train.content_uid).isdisjoint(test.content_uid), 'Row crosses candidate partitions')
    require(set(train.content_uid) | set(test.content_uid) == set(full.content_uid), 'Candidate lost rows')
    require(set(train.publisher_family).isdisjoint(test.publisher_family), 'Publisher crosses candidate partitions')
    require(set(train.leakage_group).isdisjoint(test.leakage_group), 'Resource crosses candidate partitions')


def holdout_candidate(frame):
    """Count-only deterministic finite search, with no claims of global optimality."""
    counts = frame.groupby('constraint_component', sort=True).binary_analysis_target.agg(['size', 'sum'])
    keys = counts.index.to_numpy()
    totals = np.array([len(frame), frame.binary_analysis_target.sum(), frame.binary_analysis_target.eq(0).sum()], dtype=float)
    if len(keys) < 2 or min(totals[1:]) < 2:
        return None
    amounts = np.column_stack([counts['size'], counts['sum'], counts['size']-counts['sum']])
    rng = np.random.default_rng(SEED)
    orders = [np.arange(len(keys))] + [rng.permutation(len(keys)) for _ in range(HOLDOUT_PERMUTATIONS)]
    best, best_keys = None, None
    for order in orders:
        cumulative = np.cumsum(amounts[order], axis=0)[:-1]
        for position in np.flatnonzero((cumulative[:, 1:] > 0).all(axis=1) & ((totals-cumulative)[:, 1:] > 0).all(axis=1)):
            values = cumulative[position]
            inadequate = bool(min(*values[1:], *(totals-values)[1:]) < 20)
            loss = float(np.square((values/totals-.2)/.2).mean())
            selected = tuple(sorted(keys[order[:position+1]]))
            rank = (inadequate, loss, selected)
            if best is None or rank < best:
                best, best_keys = rank, selected
    return set(best_keys) if best_keys is not None else None


def evaluate_designs(rows, support):
    records = []
    for ds, full in rows.groupby('dataset'):
        family_support = support[support.dataset.eq(ds) & support.row_type.eq('publisher_family')]
        safe = full[full.strict_publisher_eligible].sort_values('content_uid').reset_index(drop=True)
        mixed = set(family_support.loc[family_support.label_support.eq('mixed'), 'publisher_family'])
        mixed_components = [c for c, part in safe.groupby('constraint_component') if set(part.publisher_family) <= mixed]
        cohorts = [('A', safe), ('C', safe[safe.constraint_component.isin(mixed_components)].reset_index(drop=True))]
        for design, cohort in cohorts:
            for algorithm in ['GroupKFold', 'StratifiedGroupKFold']:
                # All prespecified counts are reported, not selected by model scores.
                for k in (5, 4, 3):
                    common = dict(dataset=ds, design=design, algorithm=algorithm, folds=k, cohort_N=len(cohort), excluded_N=len(full)-len(cohort))
                    if cohort.constraint_component.nunique() < k or cohort.binary_analysis_target.nunique() < 2:
                        records.append(dict(**common, fold=-1, partition='unavailable', status='insufficient groups/classes'))
                        continue
                    splitter = GroupKFold(n_splits=k) if algorithm == 'GroupKFold' else StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEED)
                    eval_counts = []
                    local = []
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', UserWarning)
                        partitions = list(splitter.split(np.zeros((len(cohort), 1)), cohort.binary_analysis_target, cohort.constraint_component))
                    for fold, (tr, te) in enumerate(partitions):
                        train, test = cohort.iloc[tr], cohort.iloc[te]
                        validate_partition(cohort, train, test)
                        evaluation = partition_diagnostics(test, family_support)
                        eval_counts.append((evaluation['fake_N'], evaluation['real_N']))
                        for name, part in [('train', train), ('validation', test)]:
                            local.append(dict(**common, fold=fold, partition=name, status='evaluated', **partition_diagnostics(part, family_support)))
                    for row in local:
                        row['minimum_evaluation_fake_N'] = min(x[0] for x in eval_counts)
                        row['minimum_evaluation_real_N'] = min(x[1] for x in eval_counts)
                        row['all_partitions_both_labels'] = all(x['both_labels'] for x in local)
                        row['all_evaluation_classes_ge20'] = min(min(x) for x in eval_counts) >= 20
                    records.extend(local)
        chosen = holdout_candidate(safe)
        common = dict(dataset=ds, design='B', algorithm='count_only_seeded_component_search', folds=1, fold=0,
                      cohort_N=len(safe), excluded_N=len(full)-len(safe))
        if chosen is None:
            records.append(dict(**common, partition='unavailable', status='no two-class candidate found'))
        else:
            train, test = safe[~safe.constraint_component.isin(chosen)], safe[safe.constraint_component.isin(chosen)]
            validate_partition(safe, train, test)
            for name, part in [('train', train), ('test', test)]:
                records.append(dict(**common, partition=name, status='evaluated', **partition_diagnostics(part, family_support),
                    minimum_evaluation_fake_N=int(test.binary_analysis_target.sum()),
                    minimum_evaluation_real_N=int(test.binary_analysis_target.eq(0).sum()),
                    all_partitions_both_labels=True,
                    all_evaluation_classes_ge20=bool(min(test.binary_analysis_target.sum(), test.binary_analysis_target.eq(0).sum()) >= 20),
                    holdout_fraction=len(test)/len(safe), approximately_20_percent=bool(.15 <= len(test)/len(safe) <= .25)))
    return pd.DataFrame(records)


def cross_overlap(rows):
    identified = rows[rows.identity_resolved]
    shared = identified.groupby('publisher_family').dataset.nunique()
    out = []
    for family in sorted(shared[shared.eq(2)].index):
        part = identified[identified.publisher_family.eq(family)]
        out.append(dict(publisher_family=family, fakehealth_N=int(part.dataset.eq('FakeHealth').sum()),
                        coaid_N=int(part.dataset.eq('CoAID').sum()), identity_note='metadata family/account overlap; not evidence of identical articles'))
    return pd.DataFrame(out, columns=['publisher_family', 'fakehealth_N', 'coaid_N', 'identity_note'])


def md(frame):
    return '| ' + ' | '.join(frame.columns) + ' |\n| ' + ' | '.join(['---']*len(frame.columns)) + ' |\n' + '\n'.join(
        '| ' + ' | '.join(f'{x:.3f}' if isinstance(x, float) else str(x).replace('|', '/') for x in row) + ' |'
        for row in frame.itertuples(index=False, name=None))


def report(summary, support, conflicts, feasibility, sizes, overlap, prior):
    conclusions = []
    for r in summary.itertuples():
        proposed = feasibility[feasibility.dataset.eq(r.dataset) & feasibility.design.eq('A') &
            feasibility.algorithm.eq('StratifiedGroupKFold') & feasibility.partition.eq('validation')]
        supported = [k for k in [5, 4, 3] if len(proposed[proposed.folds.eq(k)]) == k and
                     proposed[proposed.folds.eq(k)].all_partitions_both_labels.all() and
                     proposed[proposed.folds.eq(k)].all_evaluation_classes_ge20.all()]
        fixed = feasibility[feasibility.dataset.eq(r.dataset) & feasibility.design.eq('B') & feasibility.partition.eq('test')]
        mixed = sizes[(sizes.dataset.eq(r.dataset)) & sizes.restriction.eq('mixed_publishers') & sizes.scope.eq('direct_restriction')].iloc[0]
        conclusions += [f'{r.dataset}: {r.publisher_families} identified source families (including {r.account_proxy_families} '
            f'account proxies), {r.mixed_families} mixed families and {r.mixed_rows} mixed-family rows. '
            f'StratifiedGroupKFold supports {", ".join(map(str, supported)) or "no tested"} folds by the >=20-per-class evaluation diagnostic. '
            f'The mixed-family restriction retains {mixed.N}/{r.cohort_N} rows ({mixed.fake_N} Fake, {mixed.real_N} Real), '
            f'discarding {mixed.lost_N} ({100*mixed.lost_N/r.cohort_N:.2f}%).']
        if len(fixed):
            b = fixed.iloc[0]
            conclusions += [f'{r.dataset} fixed holdout is numerically feasible: {int(b.N)} test rows '
                f'({int(b.fake_N)} Fake, {int(b.real_N)} Real), {b.holdout_fraction:.2%} of the identified constraint-safe cohort. '
                f'Concentration/source-class risk flag: {bool(b.source_class_proxy_risk)}.']
    fh = summary.set_index('dataset').loc['FakeHealth']
    ca = summary.set_index('dataset').loc['CoAID']
    ca_keep = sizes[sizes.dataset.eq('CoAID') & sizes.restriction.eq('all_identified') & sizes.scope.eq('intact_publisher_resource_components')].iloc[0]
    conclusions += [f'FakeHealth is substantially more scientifically useful for this sensitivity: '
        f'{100*fh.mixed_rows/fh.cohort_N:.2f}% of rows are in mixed families, and its full cohort is retained. '
        'Recommend a three-fold StratifiedGroupKFold over publisher-resource components, sorted content_uid input, '
        'shuffle=True, random_state=42, on all primary eligible FakeHealth rows. Five folds are class-supported, '
        'but three folds reduce single-publisher class concentration in these count diagnostics. '
        'This is a proposed external evaluation structure; any later tuning must use only its training partitions '
        'with the same publisher/resource barriers. Do not execute or save those splits without later authorization.',
        f'CoAID supports numerical stratification on only {ca_keep.N}/{ca.cohort_N} identifiable rows, '
        f'after excluding {ca_keep.lost_N} unresolved rows ({ca_keep.lost_fake_N} Fake and {ca_keep.lost_real_N} Real). '
        f'This removes {100*ca_keep.lost_fake_N/ca.fake_N:.2f}% of original Fake support. '
        f'{100*(ca.fake_only_rows+ca.real_only_rows)/ca.identified_rows:.2f}% of identified rows remain in class-exclusive families; '
        f'the three largest Real-support families supply {100*ca.real_top3_family_share:.2f}% of identified Real rows. '
        'Ordinary GroupKFold produces class-absent validation partitions for every tested fold count. '
        'StratifiedGroupKFold retains both classes but every evaluation fold is flagged for source-class risk. '
        'Thus CoAID would be an explicitly restricted source-shift/dataset-construction stress test, not a clean test '
        'of general misinformation language, and is not recommended as the next publisher experiment.',
        'CoAID mixed-only evaluation is too sparse: no family reaches even 5 Fake and 5 Real, so the 5/5, 10/10 and 20/20 '
        'restrictions are empty. It would not represent the original population. The apparent mixed-source support '
        'in the earlier raw-host audit partly pooled different platform users; resolving that distinction strengthens '
        'rather than removes the source/class confounding concern. FakeHealth remains broadly mixed-label after normalization.',
        'Resource constraints are compatible through connected components: the three CoAID multi-family resource groups '
        '(six rows) are co-assigned; FakeHealth has none. These few links do not materially prevent candidate splitting. '
        'The overlap table identifies shared observed source families across datasets; overlap is relevant for later transfer '
        'interpretation but does not establish identical articles or eliminate domain shift.']
    lines = ['# Publisher/source-held-out feasibility audit', '',
        'This audit uses only the primary modeling-ready cohort and prepared metadata. No classifier is fitted, and no final publisher assignments or modified cohorts/splits are written. Existing content/resource generalization and unseen-publisher generalization are different questions.', '',
        '## Feasibility conclusions and recommended next design', '',
        *[text for conclusion in conclusions for text in (conclusion, '')],
        '## Conservative identity rules', '',
        'Publisher/source name=`publisher_source`; host=`article_host`; fallback URL=`article_url`; resource barrier=`leakage_group`. Name normalization uses Unicode NFKC, casefold and whitespace only. Hosts use lowercase and removal of www; subdomains collapse only to an observed parent or a previously audited identity host. Audited identities take precedence over broader container hosts. Hosted tenants remain separate. Existing explicit aliases come from the unchanged source-audit ALIASES constant. Unique observed name-host links and exact names connect aliases. Where multiple attribution names appear on one host, the exact publication host defines the source family and those names do not bridge other hosts. Conflicting audited identity hosts are not merged. There is no corporate-ownership, political-orientation, article-text or label-derived merging. Website/account families are operational proxies, not a fully resolved corporate publisher ontology; ambiguous cross-host aliases may remain separate.', '',
        'Platform domains are not publisher identities. Only explicit Twitter/X account-status URLs and YouTube account/channel URLs supply account proxies; opaque video/post links and other unresolved platform routes remain unidentified. Accounts are kept separate, with no inferred account-to-organization ownership. Missing metadata and unresolved platform authors are separate categories, not one synthetic publisher. No labels enter normalization. This is more conservative than treating every article host as a publisher; the row-level map preserves all evidence and unresolved cases.', '',
        '## Family and class support', '',
        md(summary[['dataset', 'cohort_N', 'publisher_families', 'fake_only_families', 'real_only_families', 'mixed_families',
                    'fake_only_rows', 'real_only_rows', 'mixed_rows', 'mixed_fake_N', 'mixed_real_N', 'missing_source_rows', 'unresolved_platform_rows']]), '',
        md(summary[['dataset', 'median_rows_per_publisher', 'p90_rows_per_publisher', 'largest_publisher_N', 'largest_publisher_fake_N', 'largest_publisher_real_N', 'fake_support_families', 'real_support_families', 'fake_largest_family_share', 'real_largest_family_share', 'fake_top3_family_share', 'real_top3_family_share']]), '',
        '## Earlier source-confounding comparison', '',
        'The existing source-confounding audit counts raw names and hosts, which are not necessarily publisher identities. Family/account normalization and explicit platform uncertainty can change counts and the identifiable population. Compare class support without interpreting hosting-platform aggregates as organizations.', '',
        md(prior), '',
        '## Resource constraints', '',
        'Publisher-family and resource-group separation are enforced jointly using connected components of the publisher↔resource graph. Multiple publishers in one resource group must travel together; that is a constraint, not permission to split the resource. Any component connected to an unidentified source is excluded from the strict prospective design, so neither barrier is fragmented. Mixed-family restrictions are also reported both directly and after retaining only intact original components.', '',
        md(summary[['dataset', 'identified_rows', 'strict_eligible_N', 'resource_connected_exclusion_N', 'ambiguous_host_name_rows']]), '',
        f'Conflict records: {len(conflicts)}; affected rows: {int(conflicts.N.sum()) if len(conflicts) else 0}. The conflicts CSV separates multiple known families from known-plus-unresolved groups.', '',
        '## Candidate algorithms and feasibility', '',
        'A: GroupKFold without shuffling and StratifiedGroupKFold(shuffle=True, random_state=42), evaluated at 5, 4 and 3 folds. Input rows sort by content_uid; groups are publisher-resource connected components. With no resource conflicts, this is publisher-family grouping. All candidates use the full identified, constraint-safe cohort; these are prospective alternate splits, not modifications to existing resource splits.', '',
        'B: deterministic approximately 20% holdout. Aggregate sorted connected components into total/Fake/Real counts; examine prefixes of the sorted order plus 2,048 permutations from NumPy default_rng(42). Reject candidates missing either class in train or test. Prefer candidates with >=20 of each class in both partitions; then minimize the mean squared relative departure of held-out total/Fake/Real fractions from 0.20. Break ties lexicographically by sorted component IDs. This is a finite count-only search, not proof of global optimality. A 15–25% held-out fraction is a descriptive approximation check.', '',
        'C: mixed-publisher-only sensitivity; identical A algorithms/counts on intact components wholly contained within mixed families. This changes the target population. No threshold is chosen using future model performance.', '',
        'The detailed feasibility CSV reports both train and evaluation counts, class proportions, family/component support and minimum evaluation counts. Here are evaluation partitions:', '',
        md(feasibility[feasibility.partition.isin(['validation', 'test', 'unavailable'])][['dataset', 'design', 'algorithm', 'folds', 'fold', 'partition', 'N', 'fake_N', 'real_N', 'publisher_families', 'both_labels', 'low_class_support', 'source_class_proxy_risk']]), '',
        'Both classes make ROC-AUC and two-class AP meaningful to compute. AP may numerically return a degenerate value with one class, but is flagged uninterpretable here. Macro F1 is computable with fixed class labels, but we require >=20 of each evaluation class for the descriptive support flag. These are feasibility diagnostics, not precision/power guarantees. Source-class proxy risk is flagged when >=80% of evaluation rows belong to class-exclusive families, or one publisher supplies >=50% of either evaluation class. Concentration and class-exclusive fractions must accompany any metric.', '',
        '## Prespecified restrictions and population changes', '',
        md(sizes[['dataset', 'restriction', 'scope', 'N', 'fake_N', 'real_N', 'publisher_families', 'lost_N', 'retained_percent']]), '',
        'No class-exclusive publishers have been removed from existing cohorts. These are hypothetical restriction counts only. Pure all-publisher evaluation is not certified for rows with unknown actual publishers; the strict eligible cohort explicitly records that loss. Mixed-only and minimum-count restrictions can disproportionately remove Fake or Real support, so they cannot silently replace the original CoAID population.', '',
        '## Cross-dataset overlap', '',
        f'{len(overlap)} normalized families/account proxies overlap; affected rows: FakeHealth {int(overlap.fakehealth_N.sum())}, CoAID {int(overlap.coaid_N.sum())}. No overlapping rows are removed. This is a metadata identity overlap audit, not a cross-context transfer rerun.', '',
        '## Validation and outputs', '',
        'Validation checks metadata-only normalization, deterministic aliases/counts, row accounting including unresolved identities, intact resource and publisher constraints, and unchanged input hashes. Runtime fit guards prohibit classifier/vectorizer/pipeline fitting. Candidate row assignments exist only transiently in memory; only aggregate partition counts are saved. Original cohort and split files are never opened for writing.', '',
        'Script: `src/audit_publisher_heldout_feasibility.py`. Tables: `outputs/tables/publisher_family_map.csv`, `publisher_label_support.csv`, `publisher_resource_group_conflicts.csv`, `publisher_heldout_feasibility.csv`, `publisher_mixed_source_cohort_sizes.csv`, `cross_dataset_publisher_overlap.csv`; supplemental `publisher_family_summary.csv`. Adjacent report JSON records inputs, protocol and output fingerprints.', '']
    REPORT.write_text('\n'.join(lines))


def run_audit():
    before = {str(p.relative_to(ROOT)): sha(p) for p in input_paths()}
    # Hash existing split files without reading their records: only unchangedness
    # is needed because every original leakage group is already in prepared data.
    split_paths = [ROOT / f'outputs/splits/{p}_text_{s}.csv' for p in PREFIXES.values() for s in ['holdout', 'grouped_cv']]
    split_hashes = {str(p.relative_to(ROOT)): sha(p) for p in split_paths}
    rows = load_rows()
    mapping = build_family_map(rows)
    require(not mapping.duplicated(['dataset', 'content_uid']).any() and len(mapping) == len(rows), 'Mapping lost rows')
    rows = rows.merge(mapping.drop(columns=['original_source', 'original_article_host', 'original_article_url']),
                      on=['dataset', 'content_uid'], validate='one_to_one')
    components, conflict_records = [], []
    for _, part in rows.groupby('dataset'):
        component, conflict = resource_components(part)
        components.append(component); conflict_records.extend(conflict)
    rows = pd.concat(components, ignore_index=True)
    support = support_table(rows)
    require(int(support.N.sum()) == len(rows), 'Source support accounting failed')
    summary = publisher_summary(rows, support)
    conflicts = pd.DataFrame(conflict_records, columns=['dataset', 'leakage_group', 'N', 'publisher_family_N', 'publisher_families', 'unresolved_N', 'conflict_type', 'resolution'])
    sizes = restriction_sizes(rows, support)
    feasibility = evaluate_designs(rows, support)
    overlap = cross_overlap(rows)
    prior = pd.read_csv(TABLES / 'text_source_confounding_audit.csv')
    # Keep only the existing modeling-ready rows, whose scope is recorded by the
    # previous audit; no article-level old cohorts are reopened.
    prior = prior[prior.scope.str.contains('modeling', case=False)]
    outputs = dict(publisher_family_map=mapping, publisher_label_support=support,
        publisher_resource_group_conflicts=conflicts, publisher_heldout_feasibility=feasibility,
        publisher_mixed_source_cohort_sizes=sizes, cross_dataset_publisher_overlap=overlap,
        publisher_family_summary=summary)
    for name, frame in outputs.items():
        frame.to_csv(TABLES / f'{name}.csv', index=False)
    report(summary, support, conflicts, feasibility, sizes, overlap, prior)
    require(before == {p: sha(ROOT / p) for p in before}, 'Original input changed')
    require(split_hashes == {p: sha(ROOT / p) for p in split_hashes}, 'Original splits changed')
    paths = [TABLES / f'{name}.csv' for name in outputs] + [REPORT]
    metadata = dict(validation='passed', model_fit_count=0, final_split_assignments_written=False,
        script_sha256=sha(__file__), inputs=before, unchanged_splits=split_hashes,
        outputs={str(p.relative_to(ROOT)): sha(p) for p in paths},
        protocol=dict(seed=SEED, candidate_folds=[5, 4, 3], holdout_permutations=HOLDOUT_PERMUTATIONS,
            minimum_class_diagnostic=20, source_proxy_exclusive_fraction=.8, source_proxy_largest_class_share=.5,
            source_identity_uses_labels=False, platform_domains_are_not_publishers=True))
    REPORT.with_suffix('.json').write_text(json.dumps(metadata, indent=2, sort_keys=True)+'\n')
    print('Publisher feasibility audit complete; no models fitted or final splits written. See outputs/reports/publisher_heldout_feasibility.md')


def main():
    with ExitStack() as stack:
        for cls in [Pipeline, LogisticRegression, TfidfVectorizer]:
            for method in ['fit', 'fit_transform']:
                if hasattr(cls, method):
                    stack.enter_context(patch.object(cls, method, side_effect=RuntimeError('Publisher audit forbids model fitting')))
        run_audit()


if __name__ == '__main__':
    main()
