"""Round 5: within-length-quintile ROC-AUC of the frozen text models. RE-SCORING ONLY: no refit, no retraining, no tuning, no bootstrap.
Reuses round4_check2_length.py's code paths by executing its scoring section (lines 1-34: cohorts, targets, frozen pipelines applied
transform-only with fit() patched to raise) and its quintile expression (line 50). Stops unless the reconstructed bins match
round4_length_bins.csv exactly and the unstratified ROC-AUC equals Table S19.1 at 4 decimals."""
import sys, hashlib, re
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]; R4 = ROOT/'outputs/extension_v1/scripts/round4_check2_length.py'; T = ROOT/'outputs/extension_v1/tables'
src = R4.read_text().split('\n')
assert src[49].strip().startswith("tok = r['tokens']; q = pd.qcut(pd.Series(tok).rank(method='first'), 5, labels=[1,2,3,4,5]).to_numpy()"), 'round-4 quintile line moved'
ns = {'__file__': str(R4), '__name__': 'round4_scoring'}
exec(compile('\n'.join(src[:34]), str(R4), 'exec'), ns)            # the four re-scored targets, frozen pipelines, transform-only
scored = ns['scored']; assert len(scored) == 8 and all(r['target'] in ns['targets'] for r in scored)
quintiles = lambda tok: pd.qcut(pd.Series(tok).rank(method='first'), 5, labels=[1,2,3,4,5]).to_numpy()   # identical to round-4 line 50
bins4 = pd.read_csv(T/'round4_length_bins.csv'); spear4 = pd.read_csv(T/'round4_length_spearman.csv')
# gate 1: reconstructed bins must match round4_length_bins.csv exactly
for r in scored:
    q = quintiles(r['tokens']); ref = bins4[(bins4.target == r['target']) & (bins4.classifier == r['classifier'])].sort_values('quintile')
    assert len(ref) == 5, (r['target'], r['classifier'])
    for _, row in ref.iterrows():
        m = q == row.quintile; y, s = r['y'][m], r['score'][m]
        ok = (int(m.sum()) == row.N and int(r['tokens'][m].min()) == row.tokens_min and int(r['tokens'][m].max()) == row.tokens_max
              and abs(float(y.mean()) - row.actual_fake_rate) < 1e-12 and abs(float(roc_auc_score(y, s)) - row.roc_auc) < 1e-9)
        if not ok: sys.exit(f'STOP: reconstructed bin differs from round4_length_bins.csv: {r["target"]} {r["classifier"]} quintile {row.quintile}')
print('gate 1 passed: all 40 reconstructed bins match round4_length_bins.csv')
def concordance(y, s):
    """number of (Fake, Real) pairs ordered correctly, ties 0.5; and the number of pairs"""
    f, g = s[y == 1], s[y == 0]
    if len(f) == 0 or len(g) == 0: return 0.0, 0
    diff = f[:, None] - g[None, :]
    return float((diff > 0).sum() + 0.5 * (diff == 0).sum()), int(len(f) * len(g))
rows = []
for r in scored:
    q = quintiles(r['tokens']); tok = r['tokens']; y = r['y']; s = r['score']
    ref_auc = float(spear4[(spear4.target == r['target']) & (spear4.classifier == r['classifier'])].frozen_roc_auc.iloc[0])
    unstrat = float(roc_auc_score(y, s))
    if round(unstrat, 4) != round(ref_auc, 4): sys.exit(f'STOP: unstratified ROC-AUC {unstrat:.6f} differs from Table S19.1 {ref_auc:.6f} for {r["target"]} {r["classifier"]}')
    conc_sum, pairs_sum = 0.0, 0
    for b in range(1, 6):
        m = q == b; yb, sb = y[m], s[m]; nf, nr = int(yb.sum()), int((yb == 0).sum())
        conc, pairs = concordance(yb, sb); conc_sum += conc; pairs_sum += pairs
        est = nf >= 5 and nr >= 5
        auc_b = float(roc_auc_score(yb, sb)) if (nf and nr) else np.nan
        if nf and nr: assert abs(auc_b - conc / pairs) < 1e-9          # sklearn AUC == Mann–Whitney with ties at 0.5
        rows.append(dict(target=r['target'], classifier=r['classifier'], source=r['source'], bin=str(b), tokens_min=int(tok[m].min()), tokens_max=int(tok[m].max()), N=int(m.sum()), fake_N=nf, real_N=nr,
                         same_bin_pairs=pairs, roc_auc=(auc_b if est else np.nan), estimable=bool(est), unstratified_frozen_roc_auc=np.nan, s19_1_frozen_roc_auc=np.nan))
    rows.append(dict(target=r['target'], classifier=r['classifier'], source=r['source'], bin='pooled_length_stratified', tokens_min=int(tok.min()), tokens_max=int(tok.max()), N=int(len(y)), fake_N=int(y.sum()), real_N=int((y == 0).sum()),
                     same_bin_pairs=pairs_sum, roc_auc=conc_sum / pairs_sum, estimable=True, unstratified_frozen_roc_auc=unstrat, s19_1_frozen_roc_auc=ref_auc))
out = pd.DataFrame(rows); p = HERE/'round5_within_length_auc.csv'; out.to_csv(p, index=False)
digest = hashlib.sha256(p.read_bytes()).hexdigest(); (HERE/'round5_within_length_auc.sha256').write_text(f'{digest}  round5_within_length_auc.csv\n')
pd.set_option('display.width', 250); print(out[out.bin.eq('pooled_length_stratified')][['target','classifier','N','same_bin_pairs','roc_auc','unstratified_frozen_roc_auc']].round(4).to_string(index=False))
print(out[out.bin.ne('pooled_length_stratified')][['target','classifier','bin','tokens_min','tokens_max','N','fake_N','real_N','roc_auc','estimable']].round(4).to_string(index=False)); print('sha256', digest)
