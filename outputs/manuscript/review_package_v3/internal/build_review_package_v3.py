"""Assemble review_package_v3 (step 1 of 2): copies, Figure 1 (four contexts), Figure M1 regeneration, Table 2, logged manuscript edits, merged references, citation check.
Run with the project .venv. Step 2 (render + validate) is render_review_package_v3.py in the temporary render environment."""
import re, json, shutil, hashlib, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[4]; M = ROOT/'outputs/manuscript'; V2 = M/'review_package_v2'; V3 = M/'review_package_v3'; R3 = M/'revision_v3'
EXT = ROOT/'outputs/extension_v1/tables'; TAB = ROOT/'outputs/tables'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for d in ['assets','data','data/extension_v1','figures','workflow','internal','scripts']: (V3/d).mkdir(parents=True, exist_ok=True)
copied = {}
def copy(src, dst):
    shutil.copyfile(src, dst); assert sha(src) == sha(dst); copied[str(dst.relative_to(V3))] = dict(source=str(Path(src).relative_to(ROOT)), sha256=sha(dst))
# ---- 1. byte-identical copies
for f in ['review.css', 'mentor_review.css']: copy(V2/'assets'/f, V3/'assets'/f)
for n in [2, 3, 4]:
    for p in (V2/'figures').glob(f'figure_{n}_*'): copy(p, V3/'figures'/p.name)
for p in (V2/'data').glob('*'): copy(p, V3/'data'/p.name)
for name in ['coaid_hurdle_components.csv','top_fake_real_terms.csv','fakenewsnet_external_cohort_summary.csv','fakenewsnet_external_transfer_results.csv','coaid_encoding_sensitivity.csv','coaid_encoding_prevalence.csv',
             'fakenewsnet_overlap_graph_summary.csv','fakenewsnet_engagement_model_comparison.csv','fakenewsnet_engagement_results.csv','fakenewsnet_engagement_contrast.csv','fakenewsnet_duplicate_title_sensitivity.csv',
             'fakenewsnet_round3_exclusion_results.csv','fakenewsnet_round3_exclusion_contrast.csv','fakenewsnet_round3_multimembership.csv','fakenewsnet_round3_domain_composition.csv']:
    copy(EXT/name, V3/'data/extension_v1'/name)
for name in ['text_source_label_association.csv','engagement_comparability_support.csv']: copy(TAB/name, V3/'data'/name)
for p in ['internal/pinned_model_card.md','internal/provenance_notes.md','internal/methods_protocol_checklist.csv','internal/citation_verification.json','internal/mentor_reference_verification.json']: copy(V2/p, V3/p)
# ---- 2. Figure 1: four contexts with dominant-component-exclusion secondary markers
fig_script = V3/'scripts/figure_1_four_context.py'
fig_script.write_text(r'''"""Figure 1 (v3): standardized Fake/Real observed-post ratios with 95% percentile intervals for four contexts, plus dominant-component-exclusion values as secondary markers.
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
''')
print(subprocess.run([sys.executable, str(fig_script)], capture_output=True, text=True, check=True).stdout.strip())
# ---- 3. Figure M1 regenerated with a FakeNewsNet extension node
gen = (V2/'workflow/generate_workflow.py').read_text()
gen = gen.replace("'R':{'box':[35,830,440,160]", "'N':{'box':[35,1030,440,110],'lines':['Extension: FakeNewsNet (PolitiFact, GossipCop)','Title-only LR external target (Table S13)','Engagement under frozen estimand (Table S17)','Supplementary; no frozen result changes'],'branch':'external'},\n'R':{'box':[35,830,440,160]")
gen = gen.replace("edges=[['A','B'],['B','C'],['B','D'],['C','E'],['E','F'],['F','G'],['D','H'],['H','I'],['H','J'],['I','K'],['R','K']]", "edges=[['A','B'],['B','C'],['B','D'],['C','E'],['E','F'],['F','G'],['D','H'],['H','I'],['H','J'],['I','K'],['R','K'],['N','K'],['N','E']]")
gen = gen.replace("'R-K':[[475,900],[520,900],[520,787],[570,787]],", "'R-K':[[475,900],[520,900],[520,787],[570,787]],\n'N-K':[[475,1085],[540,1085],[540,787],[570,787]],\n'N-E':[[22,1085],[22,440],[35,440]],")
gen = gen.replace("layout={'canvas':[1120,1040]", "layout={'canvas':[1120,1160]")
gen = gen.replace("'K':'Methods 2.11: frozen natural full-source LR/SVM only; no target adaptation'}}", "'K':'Methods 2.11: frozen natural full-source LR/SVM only; no target adaptation','N':'Supplementary S8/S13/S17: FakeNewsNet extension; title-only LR scoring and engagement under the frozen estimand; populations and labels inherited'}}")
gen = gen.replace("figsize=(11.2,10.4)", "figsize=(11.2,11.6)").replace("ax.set_ylim(1040,0)", "ax.set_ylim(1160,0)")
gen = gen.replace("fontsize=13.2 if len(text)>46 else 14.3", "fontsize=12.2 if len(text)>44 else 14.3")
assert "'N':{" in gen and "'N-K'" in gen and "1160" in gen
(V3/'workflow/generate_workflow.py').write_text(gen)
copy(V2/'workflow/workflow_mentor_original.mmd', V3/'workflow/workflow_mentor_original.mmd'); copy(V2/'workflow/workflow.mmd', V3/'workflow/workflow_v2.mmd')
print(subprocess.run([sys.executable, str(V3/'workflow/generate_workflow.py')], capture_output=True, text=True, check=True).stdout.strip())
# ---- 4. Table 2 from S11 / S17 values
hc = pd.read_csv(EXT/'coaid_hurdle_components.csv', keep_default_na=False); fr = pd.read_csv(EXT/'fakenewsnet_engagement_results.csv'); ex = pd.read_csv(EXT/'fakenewsnet_round3_exclusion_results.csv')
vr = pd.read_csv(TAB/'version_aware_engagement_results.csv'); sens = pd.read_csv(TAB/'cross_context_sensitivity_results.csv')
def h(m): r = hc[hc.measure.eq(m)].iloc[0]; return float(r.estimate), float(r.CI_lower), float(r.CI_upper)
def f3(e, lo, hi): return f'{e:.3f} ({lo:.3f}–{hi:.3f})'
def pct(x): return f'{100*x:.1f}%'
def frow(df, s, m): r = df[df.subset.eq(s) & df.measure.eq(m)].iloc[0]; return float(r.estimate), float(r.CI_lower), float(r.CI_upper)
fhr = vr[vr.specification.eq('FakeHealth_HealthStory') & vr.outcome.eq('unique_post_count')].iloc[0]; car = vr[vr.specification.eq('A_primary_strict') & vr.outcome.eq('unique_post_count')].iloc[0]
nd = pd.read_csv(TAB/'cross_context_no_dominant_bootstrap_refits.csv', low_memory=False); acc = nd[nd.success.astype(str).eq('True')].mean_ratio.to_numpy(float)
t2 = ['**Table 2. Overall and component Fake/Real ratios by context and population.**', '',
      '| Context | Population | Overall ratio (95%) | Share with any post, Fake / Real | Posts once posted, Fake / Real (ratio, 95%) |', '| --- | --- | --- | --- | --- |']
e, lo, hi = h('full_ratio_fake_over_real'); pf, _, _ = h('P_posts_gt_0_fake'); pr, _, _ = h('P_posts_gt_0_real'); cf, _, _ = h('E_posts_given_positive_fake'); cr, _, _ = h('E_posts_given_positive_real'); cq = h('conditional_count_ratio_fake_over_real')
t2.append(f'| CoAID News (A) | All 5,337 entities | {f3(e, lo, hi)} | {pct(pf)} / {pct(pr)} | {cf:.1f} / {cr:.1f} ({f3(*cq)}) |')
t2.append(f'| CoAID News (A) | Largest shared-post component excluded (5,150 entities) | {f3(float(sens[sens.sensitivity.eq("exclude_dominant_component")].coaid_ratio.iloc[0]), float(np.quantile(acc, .025)), float(np.quantile(acc, .975)))} | not saved | not saved |')
t2.append(f'| FakeHealth HealthStory | All 1,690 items | {f3(float(fhr.standardized_ratio), float(fhr.primary_CI_lower), float(fhr.primary_CI_upper))} | NB2 model, no occurrence component (99.6% of items have a post) | — |')
for s, n_all in [('PolitiFact', 1052), ('GossipCop', 22140)]:
    for df, pop in [(fr, f'All {n_all:,} items'), (ex, 'Largest shared-tweet component excluded')]:
        e, lo, hi = frow(df, s, 'mean_ratio'); pf = frow(df, s, 'post_probability_fake')[0]; pr = frow(df, s, 'post_probability_real')[0]; cf = frow(df, s, 'positive_fake_mean')[0]; cr = frow(df, s, 'positive_real_mean')[0]; cq = frow(df, s, 'positive_mean_ratio')
        if df is ex: pop += f' ({int(df[df.subset.eq(s)].N.iloc[0]):,} items)'
        t2.append(f'| FakeNewsNet {s} | {pop} | {f3(e, lo, hi)} | {pct(pf)} / {pct(pr)} | {cf:.1f} / {cr:.1f} ({f3(*cq)}) |')
t2 += ['', '*Hurdle components (CoAID, PolitiFact, GossipCop): standardized probability of at least one observed post and standardized expected posts given at least one; intervals are 95% component-bootstrap percentiles from 1,000 draws. Excluding a dominant component changes the reference population, so those rows are sensitivities, not replacement estimates. Sources: Supplementary Tables S11 and S17 and Table S4.*']
TABLE2 = '\n'.join(t2)
# ---- 5. Manuscript assembly with logged edits
src = (R3/'manuscript_v3_draft.md').read_text(); text = src; changes = []
def edit(label, old, new, count=1, regex=False, required=False):
    global text
    n = len(re.findall(old, text)) if regex else text.count(old)
    if n == 0:
        marker = new[:80] if not regex else ''
        status = 'already present in draft' if (marker and marker in text) else 'skipped: anchor absent (author revised this passage)'
        assert not required, (label, status)
        changes.append(dict(edit=label, before=old[:400], after=new[:400], occurrences=0, status=status)); return
    assert n == count, (label, n)
    text = re.sub(old, new, text) if regex else text.replace(old, new)
    changes.append(dict(edit=label, before=old[:400], after=new[:400], occurrences=count, status='applied'))
edit('E1 remove build header comment', re.escape(src.split('-->')[0] + '-->\n\n'), '', regex=True, required=True)
mer = re.search(r'```mermaid\n.*?```\n', text, re.S).group(0)
edit('E2 replace Mermaid block with rendered Figure M1 image', mer, '![Figure M1. Analytical workflow](workflow/figure_M1_workflow.png)\n', required=True)
edit('E2b caption: name the FakeNewsNet extension node', 'COVID-Rumor serves only as an external text target.*', 'COVID-Rumor serves only as an external text target. The FakeNewsNet extension (title-only text target and engagement under the frozen estimand) is a supplementary branch described in Supplementary S8.*')
edit('E3 Juul and Ugander: optional bracket to verified sentence', '[Optional, if added after verification: Juul and Ugander (2021, PNAS) found that many structural differences between false and true cascades shrink once cascade size is matched.]',
     'Juul and Ugander later showed that the reported structural differences between false- and true-news cascades can almost entirely be explained by false-news cascades being larger. [Juul and Ugander (2021)](#ref-juul2021)')
edit('E4a CoAID repository citation', '`11-01-2020`), with Fake/Real labels from the original News filenames. [Cui and Lee (2020)](#ref-coaid)', '`11-01-2020`), with Fake/Real labels from the original News filenames. [Cui and Lee (2020)](#ref-coaid), [cuilimeng (n.d.)](#ref-coaid_repo)')
edit('E4a2 CoAID repository citation (re-anchored to revised sentence)', 'nine health agencies and health-information publishers. [Cui and Lee (2020)](#ref-coaid) The primary engagement population', 'nine health agencies and health-information publishers. [Cui and Lee (2020)](#ref-coaid), [cuilimeng (n.d.)](#ref-coaid_repo) The primary engagement population')
edit('E3b map draft anchor ref-juul to bibliography key juul2021', '(#ref-juul)', '(#ref-juul2021)', count=text.count('(#ref-juul)'))
edit('E4b FakeHealth repository citation', 'Real. [Dai et al. (2020)](#ref-fakehealth) Reviews,', 'Real. [Dai et al. (2020)](#ref-fakehealth), [EnyanDai (n.d.)](#ref-fakehealth_repo) Reviews,')
edit('E4c COVID-Rumor repository citation', 'Twitter subset. [Cheng et al. (2021)](#ref-covidrumor) Annotations', 'Twitter subset. [Cheng et al. (2021)](#ref-covidrumor), [MickeysClubhouse (n.d.)](#ref-covidrumor_repo) Annotations')
edit('E5 transformer checkpoint and attribution citations', '- **Transformer.** DistilRoBERTa (pinned revision), title plus body prefix', '- **Transformer.** DistilRoBERTa at a pinned revision [Sanh et al. (n.d.)](#ref-distilroberta_card), with architecture and distillation attributions to [Liu et al. (2019)](#ref-liu2019) and [Sanh et al. (2019)](#ref-sanh2019); title plus body prefix')
edit('E6 FakeNewsNet citation and verified label wording (reference_checks.md)', 'with the same quality and overlap rules. [Cite FakeNewsNet; add to references.] Only titles were available locally, so the existing title-only LR models were used; no title-only SVM exists. Labels were inherited: PolitiFact from fact-checker verdicts, GossipCop from entertainment-site scores. [Verify label definitions against the FakeNewsNet documentation.]',
     'with the same quality and overlap rules. [Shu et al. (2020)](#ref-fakenewsnet) Only titles were available locally, so the existing title-only LR models were used; no title-only SVM exists. Labels were inherited: PolitiFact from fact-checker verdicts; GossipCop items scored below 5 of 10 by GossipCop as Fake and items FakeNewsNet took from E! Online as Real, although the local news URLs of the Real items do not show E! Online (Supplementary Table S17). In a supplementary extension the same two collections were also analysed under the frozen engagement estimand (Methods 2.4–2.5; Supplementary S8).')
edit('E7 software citations', 'Analyses used Python 3.14 with scikit-learn and statsmodels, and Python 3.12 with PyTorch and Transformers for the transformer;', 'Analyses used Python 3.14 with scikit-learn [Pedregosa et al. (2011)](#ref-pedregosa2011) and statsmodels [Seabold and Perktold (2010)](#ref-seabold2010), and Python 3.12 with PyTorch [Paszke et al. (2019)](#ref-paszke2019) and Transformers [Wolf et al. (2020)](#ref-wolf2020) for the transformer;')
edit('E8 FakeNewsNet engagement paragraph and Table 2 replace conditional placeholder', '[If the FakeNewsNet engagement extension is completed: add PolitiFact and GossipCop as further contexts here and in Figure 1, with the same estimand.]',
     'Two non-health FakeNewsNet collections, analysed under the same estimand as a supplementary extension, gave ratios of 0.559 (0.439–1.868) for PolitiFact and 2.144 (1.212–3.290) for GossipCop. Excluding each collection\'s largest shared-tweet component, which changes the reference population, moved them to 1.446 (1.073–2.046) and 1.374 (1.146–1.621) (Figure 1; Table 2). The PolitiFact direction therefore rests on one component of 84 mostly Real items that carries two thirds of its post memberships, while GossipCop\'s Fake items drew more observed posts than its Real items under both populations.\n\n' + TABLE2)
edit('E8b Figure 1 caption: four contexts', '*Figure 1. Fake/Real observed-post ratios in CoAID and FakeHealth. Points are model-standardized ratios of expected unique observed posts, including items with no observed posts; bars are 95% cluster-bootstrap percentile intervals. Log scale; the dashed line marks parity. Ratio-of-ratios 0.301 (0.154–0.645).*',
     '*Figure 1. Fake/Real observed-post ratios in four contexts. Filled markers are model-standardized ratios of expected unique observed posts, including items with no observed posts, for CoAID News (A), FakeHealth HealthStory and the supplementary FakeNewsNet PolitiFact and GossipCop collections; bars are 95% component-bootstrap percentile intervals. Open markers below each primary value show the ratio after excluding the largest shared-post component, which changes the reference population; FakeHealth has no dominant component. Log scale; the dashed line marks parity. CoAID-to-FakeHealth ratio-of-ratios 0.301 (0.154–0.645).*')
edit('E9a Supplementary Table S12', '(Supplementary Table [Sx]). CoAID', '(Supplementary Table S12). CoAID')
edit('E9b Supplementary Table S13', 'weak evidence on its own (Supplementary [Sx]).', 'weak evidence on its own (Supplementary Table S13).')
edit('E12a CoAID Real source wording (reference_checks.md)', 'while Real items come from official health agencies and established news sources. [Verify against Cui and Lee (2020).]', 'while Real items come from official health agencies and reliable health-information outlets.')
edit('E12b remove verified FakeHealth criteria bracket', ' [Verify criteria wording against Dai et al. (2020).]', '')
edit('E12c remove verified review-criteria bracket', ' [verify the review criteria in Dai et al. (2020)]', '')
edit('E12d collection-method sentence (both datasets used title search)', 'If CoAID gathered posts by searching for each item [verify collection method in Cui and Lee (2020)], low-profile items may have fewer captured posts, so some Fake zeros could reflect discoverability rather than absence of sharing.',
     'Both CoAID and FakeHealth gathered posts by searching for each item\'s title [Cui and Lee (2020)](#ref-coaid), [Dai et al. (2020)](#ref-fakehealth); if search-based capture missed low-profile items more often among CoAID\'s sources, some Fake zeros could reflect discoverability rather than absence of sharing.')
edit('E13 encoding-artifact quantification (Supplementary S12, S16)', '[Pending quantification: two of CoAID\'s most Fake-leaning words are character-encoding errors, so part of CoAID\'s within-dataset separability may reflect how its Fake text was scraped.]',
     'Two of CoAID\'s 25 most Fake-leaning words are character-encoding errors, present in 25.9% of CoAID Fake documents and in no Real document (Supplementary Tables S12 and S16); repairing them left holdout macro F1 essentially unchanged (0.926 to 0.928), so the artifact marks the Fake class without carrying its separability.')
edit('E14 feasibility support table reference', ' [Confirm wording against the 2026-10-09 feasibility report; decide whether the support table goes in the Supplement.]', ' (Supplementary Table S15).')
edit('E15 Cramér\'s V reference', '[Cramér\'s V ≈ 0.97 against ≈ 0.35 in FakeHealth; Supplementary Methods S3 — confirm and decide whether to report in Results]', '(Cramér\'s V 0.97 against 0.35 in FakeHealth; Supplementary Table S14)')
# references
v2m = (V2/'manuscript.md').read_text(); refblock = v2m.split('## References')[1]
entries = dict(re.findall(r'<a id="ref-([a-z0-9_]+)"></a>\n\n(.*?)\n', refblock))
entries['fakenewsnet'] = 'Shu, K., Mahudeswaran, D., Wang, S., Lee, D., & Liu, H. (2020). [FakeNewsNet: A Data Repository with News Content, Social Context, and Spatiotemporal Information for Studying Fake News on Social Media](https://doi.org/10.1089/big.2020.0062). *Big Data*, 8(3), 171–188.'
entries['juul2021'] = 'Juul, J. L., & Ugander, J. (2021). [Comparing information diffusion mechanisms by matching on cascade size](https://doi.org/10.1073/pnas.2100786118). *Proceedings of the National Academy of Sciences*, 118(46), e2100786118.'
cited = sorted(set(re.findall(r'\(#ref-([a-z0-9_]+)\)', text)))
ordered = sorted(entries, key=lambda k: re.sub(r'[^a-z]', '', entries[k].split('(')[0].lower()))
reflist = '## References\n\n' + '\n\n'.join(f'<a id="ref-{k}"></a>\n\n{entries[k]}' for k in ordered) + '\n'
_head, _tail = text.split('## References', 1)
text = _head + reflist.rstrip('\n') + '\n'
changes.append(dict(edit='E11 references generated from merged bibliography (replaces the draft reference section, including its two hand-formatted entries)', before=('## References' + _tail)[:400], after=reflist[:400], occurrences=1, status='applied'))
# supplementary cross-reference links (markdown anchors; render step converts to PDF page links)
supp = (R3/'supplementary_material_v3.md').read_text()
def slug(h): import unicodedata; s = unicodedata.normalize('NFKD', h).encode('ascii', 'ignore').decode().lower(); s = re.sub(r'[^\w\s-]', '', s); return re.sub(r'[\s]+', '-', s).strip('-')
heads = {}
for m in re.finditer(r'^## (S(\d)\..*|Table S(\d+)\..*)$', supp, re.M):
    title = m.group(1); key = ('S' + m.group(2)) if m.group(2) else ('Table S' + m.group(3)); heads[key] = slug(title)
def link_sec(m):
    key = 'S' + m.group(2); return f'[{m.group(1)}](supplementary_material.md#{heads[key]})' if key in heads else m.group(0)
n_before = text
text = re.sub(r'(?<!\[)(Supplementary Methods (S(\d)))\b(?!\]\()', lambda m: f'[{m.group(1)}](supplementary_material.md#{heads["S"+m.group(3)]})' if 'S'+m.group(3) in heads else m.group(0), text)
text = re.sub(r'(?<!\[)(Supplementary Methods S5 and )(S6)\b', lambda m: m.group(1) + f'[S6](supplementary_material.md#{heads["S6"]})', text)
text = re.sub(r'(?<![\[\w])((?:Supplementary )?Tables? S(1[1-7]))\b(?!\]\()', lambda m: f'[{m.group(1)}](supplementary_material.md#{heads["Table S"+m.group(2)]})', text)
text = re.sub(r'(?<![\[\w])(Supplementary S8)\b', lambda m: f'[{m.group(1)}](supplementary_material.md#{heads["S8"]})', text)
changes.append(dict(edit='E10 supplementary cross-reference links', before='plain mentions of Supplementary Methods S1–S7, S8 and Tables S11–S17', after='markdown links to supplementary_material.md anchors (render step rewrites to supplement_review.pdf#page=N)', occurrences=len(re.findall(r'\]\(supplementary_material\.md#', text))))
text = text.replace('](figures/figure_1_engagement_context_contrast.png)', '](figures/figure_1_engagement_context_contrast.png)')
(V3/'manuscript.md').write_text(text)
(V3/'supplementary_material.md').write_text(supp.replace('<!-- supplementary_material_v3.md: derivative', '<!-- supplementary_material.md (review_package_v3), derivative'))
# merged bibliography
bib = (V2/'references.bib').read_text().rstrip() + '\n\n' + (R3/'references_v3_additions.bib').read_text()
(V3/'references.bib').write_text(bib)
keys = re.findall(r'^@\w+\{([^,]+),', bib, re.M); dup = sorted({k for k in keys if keys.count(k) > 1})
cited = sorted(set(re.findall(r'\(#ref-([a-z0-9_]+)\)', text))); anchors = sorted(set(re.findall(r'<a id="ref-([a-z0-9_]+)">', text)))
cit = dict(bib_entries=len(set(keys)), bib_duplicate_keys=dup, cited_keys=len(cited), citations_without_bib_entry=sorted(set(cited) - set(keys)), bib_entries_not_cited=sorted(set(keys) - set(cited)),
           citations_without_reference_anchor=sorted(set(cited) - set(anchors)), anchors_not_cited=sorted(set(anchors) - set(cited)), supplement_citation_anchors=len(re.findall(r'#ref-', supp)))
placeholders = [m.group(0) for m in re.finditer(r'\[[^\]\n]*\]', text) if not re.match(r'\[[^\]]*\]\(', text[m.start():m.end()+1]) and not re.search(r'^\[(Figure|Table) ', m.group(0)) and not re.match(r'\[[^\]]*\]\(', text[m.start():m.end()+1])]
placeholders = [p for p in placeholders if not re.search(r'\]\(', text[text.find(p):text.find(p)+len(p)+1])]
json.dump(dict(edits=changes, placeholders_remaining=placeholders, citations=cit, copied=copied, table2=TABLE2), open(V3/'internal/manuscript_changes.json', 'w'), indent=1)
print(json.dumps(cit, indent=0)); print('placeholders:', json.dumps(placeholders, indent=0)); print('edits:', len(changes))
