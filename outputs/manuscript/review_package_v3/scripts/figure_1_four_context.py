"""Figure 1 (v3): standardized Fake/Real observed-post ratios with 95% percentile intervals for four contexts, plus dominant-component-exclusion values as secondary markers.
Values are read only from saved tables; every value and its source row is written to figures/figure_1_sources.csv."""
import json, hashlib, sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT = Path(__file__).resolve().parents[4]; OUT = Path(__file__).resolve().parents[1]/'figures'
TAB = ROOT/'outputs/tables'; EXT = ROOT/'outputs/extension_v1/tables'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
BLUE, ORANGE, INK, GRAY, PURPLE, GREEN = '#295C7A', '#A65D32', '#252B30', '#707980', '#6B5B95', '#4F7F5C'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,'axes.labelsize':9.5,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':8.5,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
 'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white','figure.facecolor':'white','axes.facecolor':'white','text.color':INK,'axes.labelcolor':INK,'axes.edgecolor':GRAY,'xtick.color':INK,'ytick.color':INK,'svg.hashsalt':'review-package-v3'})
rows = []
def rec(context, kind, est, lo, hi, source, selector, fields):
    rows.append(dict(context=context, kind=kind, estimate=est, CI_lower=lo, CI_upper=hi, source_file=str(source.relative_to(ROOT)), source_sha256=sha(source), source_row_selector=selector, source_fields=fields)); return est, lo, hi
vr = pd.read_csv(TAB/'version_aware_engagement_results.csv')
def vr_row(spec): r = vr[vr.specification.eq(spec) & vr.outcome.eq('unique_post_count')].iloc[0]; return float(r.standardized_ratio), float(r.primary_CI_lower), float(r.primary_CI_upper)
P = {}
P['CoAID A'] = rec('CoAID News (Specification A)', 'primary', *vr_row('A_primary_strict'), TAB/'version_aware_engagement_results.csv', 'specification=A_primary_strict; outcome=unique_post_count', 'standardized_ratio; primary_CI_lower; primary_CI_upper')
P['FakeHealth'] = rec('FakeHealth HealthStory', 'primary', *vr_row('FakeHealth_HealthStory'), TAB/'version_aware_engagement_results.csv', 'specification=FakeHealth_HealthStory; outcome=unique_post_count', 'standardized_ratio; primary_CI_lower; primary_CI_upper')
sens = pd.read_csv(TAB/'cross_context_sensitivity_results.csv'); sr = sens[sens.sensitivity.eq('exclude_dominant_component')].iloc[0]
nd = pd.read_csv(TAB/'cross_context_no_dominant_bootstrap_refits.csv', low_memory=False); acc = nd[nd.success.astype(str).eq('True')].mean_ratio.to_numpy(float); assert len(acc) == 1000
X = {}
X['CoAID A'] = rec('CoAID News (Specification A)', 'exclude_dominant_component', float(sr.coaid_ratio), float(np.quantile(acc, .025)), float(np.quantile(acc, .975)), TAB/'cross_context_sensitivity_results.csv', 'sensitivity=exclude_dominant_component (point); interval = 2.5/97.5 linear percentiles of 1000 successful draws in cross_context_no_dominant_bootstrap_refits.csv', 'coaid_ratio; mean_ratio draws')
fr = pd.read_csv(EXT/'fakenewsnet_engagement_results.csv'); ex = pd.read_csv(EXT/'fakenewsnet_round3_exclusion_results.csv')
for s in ['PolitiFact', 'GossipCop']:
    r = fr[fr.subset.eq(s) & fr.measure.eq('mean_ratio')].iloc[0]; P[s] = rec(f'FakeNewsNet {s}', 'primary', float(r.estimate), float(r.CI_lower), float(r.CI_upper), EXT/'fakenewsnet_engagement_results.csv', f'subset={s}; measure=mean_ratio', 'estimate; CI_lower; CI_upper')
    r = ex[ex.subset.eq(s) & ex.measure.eq('mean_ratio')].iloc[0]; X[s] = rec(f'FakeNewsNet {s}', 'exclude_dominant_component', float(r.estimate), float(r.CI_lower), float(r.CI_upper), EXT/'fakenewsnet_round3_exclusion_results.csv', f'subset={s}; measure=mean_ratio', 'estimate; CI_lower; CI_upper')
order = [('CoAID A', 'CoAID News (A)', BLUE, 'o'), ('FakeHealth', 'FakeHealth HealthStory', ORANGE, 's'), ('PolitiFact', 'FakeNewsNet PolitiFact', PURPLE, 'D'), ('GossipCop', 'FakeNewsNet GossipCop', GREEN, '^')]
fig, ax = plt.subplots(figsize=(7.2, 4.6)); fig.subplots_adjust(left=.24, right=.96, bottom=.22, top=.80)
fig.suptitle('Observed post engagement across four contexts', x=.24, y=.965, ha='left', fontsize=12, fontweight='bold')
fig.text(.24, .875, 'Filled markers: primary populations. Open markers below: largest shared-post component\nexcluded, which changes the reference population. FakeHealth has no dominant component.', fontsize=8.4, va='top', linespacing=1.4)
for i, (k, label, color, marker) in enumerate(order):
    y = 3 - i; e, lo, hi = P[k]
    ax.plot([lo, hi], [y, y], color=color, lw=1.6); ax.plot([lo, hi], [y, y], linestyle='none', marker='|', ms=9, color=color); ax.plot(e, y, marker=marker, color=color, ms=7, zorder=3)
    ax.text(e, y + .22, f'{e:.3f} [{lo:.3f}, {hi:.3f}]', ha='center', fontsize=8.3)
    if k in X:
        e2, lo2, hi2 = X[k]; y2 = y - .33
        ax.plot([lo2, hi2], [y2, y2], color=GRAY, lw=1.0, ls=(0, (3, 2))); ax.plot(e2, y2, marker=marker, mfc='white', mec=GRAY, color=GRAY, ms=6, zorder=3)
        ax.text(hi2 * 1.06, y2, f'{e2:.3f} [{lo2:.3f}, {hi2:.3f}]', va='center', ha='left', fontsize=7.6, color=GRAY)
ax.set_xscale('log'); ax.set_xlim(.10, 4.6); ax.set_ylim(-.75, 3.55)
ax.set_xticks([.1, .2, .5, 1., 2., 4.], ['0.1', '0.2', '0.5', '1.0', '2.0', '4.0']); ax.minorticks_off()
ax.set_yticks([3, 2, 1, 0], [o[1] for o in order]); ax.tick_params(axis='y', length=0, pad=10); ax.spines['left'].set_visible(False)
ax.axvline(1, color=GRAY, lw=.9, ls='--', zorder=0); ax.set_xlabel('Standardized Fake/Real expected observed-post ratio', labelpad=9)
fig.text(.24, .05, 'Log scale · Bars: 95% component-bootstrap percentile intervals (1,000 draws each)', fontsize=8.2, color=GRAY)
for ext in ['svg', 'pdf', 'png']: fig.savefig(OUT/f'figure_1_engagement_context_contrast.{ext}', dpi=300, metadata=({'Creator': 'review_package_v3 figure script'} if ext != 'png' else None))
plt.close(fig)
pd.DataFrame(rows).to_csv(OUT/'figure_1_sources.csv', index=False)
print(json.dumps({k: [round(v, 4) for v in P[k]] for k in P}), json.dumps({k: [round(v, 4) for v in X[k]] for k in X}))
