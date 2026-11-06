"""Round 4 Check 7: rounding consistency of displayed numbers in review_package_v3 manuscript text, Tables 1-2 and figure labels against saved source values (half-up at displayed precision). Lists proposed edits; edits nothing."""
import re, json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]; V3 = ROOT/'outputs/manuscript/review_package_v3'; T = ROOT/'outputs/extension_v1/tables'
hu = lambda v, d: str(Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))
# source values: provenance (v2), key results, figure sources, extension/hurdle/round tables, FakeNewsNet, encoding, Cramér
vals = []
for _, r in pd.read_csv(ROOT/'outputs/tables/manuscript_numeric_provenance.csv').iterrows(): vals.append((float(r.value), f"{r.source_artifact}#{r.field}"))
for _, r in pd.read_csv(ROOT/'outputs/tables/manuscript_key_results.csv').iterrows():
    try: vals.append((float(r.estimate), f"manuscript_key_results.csv#{r.result_id}:{r.primary_metric}"))
    except Exception: pass
for f in ['coaid_hurdle_components.csv','fakenewsnet_engagement_results.csv','fakenewsnet_engagement_contrast.csv','fakenewsnet_round3_exclusion_results.csv','fakenewsnet_round3_exclusion_contrast.csv','fakenewsnet_external_transfer_results.csv','coaid_encoding_sensitivity.csv','coaid_encoding_prevalence.csv','fakenewsnet_duplicate_title_sensitivity.csv','fakenewsnet_overlap_graph_summary.csv']:
    df = pd.read_csv(T/f, low_memory=False)
    for c in df.columns:
        s = pd.to_numeric(df[c], errors='coerce')
        for i, v in s.dropna().items(): vals.append((float(v), f"extension_v1/{f}#{c}[row{i+1}]"))
for f in ['text_source_label_association.csv','cross_context_sensitivity_results.csv','version_aware_engagement_results.csv','fakehealth_publisher_heldout_summary.csv','tfidf_logistic_holdout_results.csv','tfidf_linear_svm_holdout_results.csv','distilroberta_budget_seed_summary.csv','covid_rumor_external_transfer_results.csv','covid_rumor_external_vocabulary_diagnostics.csv','tfidf_logistic_title_holdout_results.csv','engagement_comparability_support.csv']:
    df = pd.read_csv(ROOT/'outputs/tables'/f, low_memory=False)
    for c in df.columns:
        s = pd.to_numeric(df[c], errors='coerce')
        for i, v in s.dropna().items(): vals.append((float(v), f"outputs/tables/{f}#{c}[row{i+1}]"))
fs = pd.read_csv(V3/'figures/figure_1_sources.csv')
for _, r in fs.iterrows():
    for c in ['estimate','CI_lower','CI_upper']: vals.append((float(r[c]), f"figure_1_sources.csv#{r.context}:{r.kind}:{c}"))
# manuscript display tokens with decimals (text, tables, captions); references excluded
text = (V3/'manuscript.md').read_text().split('## References')[0]; text = re.sub(r'\]\([^)]*\)', ']', text)
toks = [(m.group(0), text.count('\n', 0, m.start()) + 1) for m in re.finditer(r'(?<![\d.])\d+\.\d+(?![\d.])', text)]
findings = []
lines = text.split('\n')
for tok, line in toks:
    d = len(tok.split('.')[1]); t = float(tok); tol = 0.5 * 10 ** -d + 1e-12
    def near(v): return min(abs(v - t), abs(v * 100 - t))
    cands = [(v, src) for v, src in vals if near(v) <= tol]
    wide = [(v, src) for v, src in vals if tol < near(v) <= 2 * 10 ** -d]
    ctx = lines[line - 1][:110]
    if not cands and not wide: continue
    def passes(v): return hu(v, d) == tok or hu(v * 100, d) == tok
    if cands:
        ok = [c for c in cands if passes(c[0])]; bad = [c for c in cands if not passes(c[0])]
        if ok and not bad: continue
        status = 'none_pass' if not ok else 'mixed'
        shown = bad if status == 'none_pass' else (ok + bad)
        findings.append(dict(status=status, location=f'manuscript.md line {line}', context=ctx, displayed=tok, candidates='; '.join(f'{src}={v:.6g}->{hu(v if abs(v-t)<=abs(v*100-t) else v*100, d)}' for v, src in shown[:6])))
    else:
        findings.append(dict(status='no_candidate_within_half_unit', location=f'manuscript.md line {line}', context=ctx, displayed=tok, candidates='; '.join(f'{src}={v:.6g}->{hu(v if abs(v-t)<=abs(v*100-t) else v*100, d)}' for v, src in wide[:4])))
# figure labels: Figure 1 (package script, f'{x:.3f}') and Figures 2-4 (frozen script, f'{x:.3f}') versus half-up
fig_rows = []
for _, r in fs.iterrows():
    for c in ['estimate','CI_lower','CI_upper']:
        v = float(r[c]); py = f'{v:.3f}'; h = hu(v, 3)
        if py != h: fig_rows.append(dict(location=f'Figure 1 label ({r.context}, {r.kind}, {c})', displayed=py, source='figure_1_sources.csv', source_value=v, half_up=h, note="python '%.3f' (binary round-half-even) differs from half-up"))
fv = pd.read_csv(ROOT/'outputs/manuscript/figure_values.csv')
for _, r in fv.iterrows():
    try: v = float(r.value)
    except Exception: continue
    for d in (3, 2, 4):
        py = f'{v:.{d}f}'; h = hu(v, d)
        if py != h and d == 3: fig_rows.append(dict(location=f'Figure {r.figure} label ({r.field})', displayed=py, source=f'{r.source_artifact}#{r.field}', source_value=v, half_up=h, note="frozen figure script uses '%.3f'"))
out = pd.DataFrame(findings + fig_rows); out.to_csv(T/'round4_rounding_findings.csv', index=False)
pd.set_option('display.width', 250); print('tokens checked', len(toks), '| findings', len(out)); print(out.to_string(index=False) if len(out) else '(none)')
