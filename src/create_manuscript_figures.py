"""Render frozen aggregate results for the manuscript; no fitting or inference.

Run: .venv/bin/python src/create_manuscript_figures.py
SVG/PDF are authoritative; PNG exports are 300 dpi. Outputs stay under
outputs/manuscript. A completed run verifies hashes without rendering again.
"""
from pathlib import Path
import os
import tempfile
import hashlib
import json
import csv
import math
import numpy as np
import pandas as pd
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'manuscript_figure_matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/manuscript'
FIG=OUT/'figures'
TAB=OUT/'tables'
VALID=OUT/'figure_validation.json'
SOURCES={}
EXPECTED={}
CACHE={}
MANIFEST=[]
VALUES=[]
CHECKS={}
OUTPUTS=[]
BLUE='#295C7A'
ORANGE='#A65D32'
INK='#252B30'
GRAY='#707980'
STYLE={'font.family':'DejaVu Sans','font.size':9.5,'axes.labelsize':9.5,'axes.titlesize':11,
 'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'axes.spines.top':False,
 'axes.spines.right':False,'axes.linewidth':.7,'xtick.major.width':.7,'ytick.major.width':.7,
 'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white',
 'figure.facecolor':'white','axes.facecolor':'white','text.color':INK,'axes.labelcolor':INK,
 'axes.edgecolor':GRAY,'xtick.color':INK,'ytick.color':INK,'svg.hashsalt':'frozen-manuscript-results'}
plt.rcParams.update(STYLE)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return str(Path(p).relative_to(ROOT))
def track(p):
    SOURCES[rel(p)]=sha(p)
    return p

def init_sources():
    synthesis=json.loads(track(ROOT/'outputs/reports/manuscript_synthesis_validation.json').read_text())
    assert synthesis['validation']=='passed'
    EXPECTED.update(synthesis['sources']);EXPECTED.update(synthesis['outputs'])
    lexical=json.loads(track(ROOT/'outputs/reports/cross_context_lexical_alignment.json').read_text())
    assert lexical['validation']=='passed'
    EXPECTED.update(lexical['outputs'])
    engage=json.loads(track(ROOT/'outputs/tables/cross_context_engagement_metadata.json').read_text())
    assert engage['complete'] and engage['input_fingerprints_verified']
    EXPECTED.update(engage['input_sha256'])
    for name in ['AGENTS.md','outputs/reports/manuscript_results_synthesis.md',
      'outputs/reports/manuscript_figure_plan.md','outputs/tables/manuscript_claim_evidence_matrix.csv']:
        p=track(ROOT/name)
        if name in EXPECTED:assert sha(p)==EXPECTED[name],name
    # Preserve manuscript prose without reading its content.
    for name in ['results_draft.md','results_numeric_checklist.csv','results_draft_validation.json']:
        p=OUT/name
        if p.exists():track(p)


def data(name):
    if name not in CACHE:
        p=track(ROOT/f'outputs/tables/{name}.csv')
        assert rel(p) in EXPECTED,'No validated fingerprint for '+name
        assert sha(p)==EXPECTED[rel(p)],'Changed source: '+name
        CACHE[name]=pd.read_csv(p)
    return CACHE[name]


def row(name,**where):
    d=data(name)
    for k,v in where.items():d=d[d[k].eq(v)]
    assert len(d)==1,(name,where,len(d))
    return d.iloc[0]


def value(fig,name,where,field):
    v=row(name,**where)[field]
    VALUES.append(dict(figure=fig,source_artifact=f'outputs/tables/{name}.csv',source_row_or_key=json.dumps(where,sort_keys=True),
                       field=field,value=v.item() if hasattr(v,'item') else v))
    return v


def save(fig,number,stem,inputs,scope,uncertainty):
    fig.canvas.draw()
    canvas=fig.bbox
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text():continue
        b=t.get_window_extent(fig.canvas.get_renderer())
        assert b.x0>=-1 and b.y0>=-1 and b.x1<=canvas.x1+1 and b.y1<=canvas.y1+1,(stem,'clipped text',t.get_text())
    for i,a in enumerate(fig.texts):
        for b in fig.texts[i+1:]:
            assert not a.get_window_extent().overlaps(b.get_window_extent()),(stem,'overlapping figure header text')
    CHECKS[stem+'_text_within_canvas']=True
    for ext in ['svg','pdf','png']:
        p=FIG/(stem+'.'+ext)
        assert not p.exists(),'Refusing overwrite: '+str(p)
        metadata={'Creator':'Frozen manuscript results renderer'} if ext=='pdf' else {'Creator':'Frozen manuscript results renderer'} if ext=='svg' else None
        fig.savefig(p,dpi=300,metadata=metadata)
        if ext=='png':
            with Image.open(p) as im:
                assert all(abs(d-300)<.1 for d in im.info['dpi'])
                assert im.width==round(fig.get_figwidth()*300) and im.height==round(fig.get_figheight()*300)
        if ext=='svg':assert '<image' not in p.read_text(),'Unexpected raster image in SVG'
        OUTPUTS.append(p)
        MANIFEST.append(dict(artifact_id=f'Figure {number}',file_path=rel(p),format=ext,width_inches=fig.get_figwidth(),
          height_inches=fig.get_figheight(),dpi=300 if ext=='png' else '',authoritative=ext in ['svg','pdf'],
          evaluation_scope=scope,uncertainty=uncertainty,source_artifacts=';'.join('outputs/tables/'+n+'.csv' for n in inputs),sha256=sha(p)))
    plt.close(fig)


def figure1():
    n='version_aware_engagement_results';c='cross_context_engagement_contrast'
    a={'coaid_specification':'A_primary_strict'}
    fig,ax=plt.subplots(figsize=(7.2,3.3));fig.subplots_adjust(left=.18,right=.95,bottom=.27,top=.68)
    fig.suptitle('Observed post engagement across contexts',x=.18,y=.965,ha='left',fontsize=12,fontweight='bold')
    contrast=value(1,c,a,'ratio_of_ratios');lo=value(1,c,a,'CI_lower_ratio');hi=value(1,c,a,'CI_upper_ratio')
    fig.text(.18,.855,f'Ratio-of-ratios = {contrast:.3f}; 95% bootstrap interval {lo:.3f}–{hi:.3f}',fontsize=10)
    for y,ds,spec,color,marker,col in [(1,'CoAID','A_primary_strict',BLUE,'o','coaid_ratio'),(0,'FakeHealth','FakeHealth_HealthStory',ORANGE,'s','fakehealth_ratio')]:
        w={'specification':spec,'outcome':'unique_post_count'}
        estimate=value(1,n,w,'standardized_ratio');lower=value(1,n,w,'primary_CI_lower');upper=value(1,n,w,'primary_CI_upper')
        assert math.isclose(estimate,row(c,**a)[col],rel_tol=1e-12)
        assert 'bootstrap' in row(n,**w).uncertainty_method.lower() and row(n,**w).successful_refits>0
        ax.plot([lower,upper],[y,y],color=color,lw=1.6)
        ax.plot([lower,upper],[y,y],linestyle='none',marker='|',ms=9,color=color)
        ax.plot(estimate,y,marker=marker,color=color,ms=7)
        ax.text(estimate,y+.26,f'{estimate:.3f} [{lower:.3f}, {upper:.3f}]',ha='center',fontsize=9)
    ax.set_xscale('log');ax.set_xlim(.10,1.3);ax.set_ylim(-.55,1.55)
    ax.set_xticks([.1,.2,.5,1.0],['0.1','0.2','0.5','1.0']);ax.minorticks_off()
    ax.set_yticks([1,0],['CoAID','FakeHealth']);ax.tick_params(axis='y',length=0,pad=10)
    ax.spines['left'].set_visible(False);ax.axvline(1,color=GRAY,lw=.9,ls='--',zorder=0)
    ax.set_xlabel('Standardized Fake/Real expected observed-post ratio',labelpad=9)
    fig.text(.18,.065,'Log scale · Bars show saved 95% bootstrap percentile intervals',fontsize=9,color=GRAY)
    CHECKS['figure_1_log_axis_and_saved_percentile_intervals']=ax.get_xscale()=='log'
    save(fig,1,'figure_1_engagement_context_contrast',[n,c],'All content items including zero observed posts; primary CoAID A and FakeHealth',
         '95% overlap-cluster bootstrap percentile; cross-context interval uses independent dataset distributions')


def figure2():
    fig,axes=plt.subplots(1,2,figsize=(7.2,4.5),sharey=True)
    fig.subplots_adjust(left=.235,right=.975,bottom=.255,top=.76,wspace=.18)
    fig.suptitle('Within-context versus cross-context text performance',x=.025,y=.965,ha='left',fontsize=12,fontweight='bold')
    legend=[Line2D([],[],color=INK,marker='o',linestyle='none',label='Within context',ms=6),
            Line2D([],[],color=GRAY,marker='s',markerfacecolor='white',linestyle='none',label='Cross-context transfer',ms=6)]
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.57,.915),ncol=2,frameon=False)
    positions=[3.5,2.5,1,0];labels=[];names=[]
    groups=[('CoAID','FakeHealth','Logistic Regression','tfidf_logistic'),('CoAID','FakeHealth','Linear SVM','tfidf_linear_svm'),
            ('FakeHealth','CoAID','Logistic Regression','tfidf_logistic'),('FakeHealth','CoAID','Linear SVM','tfidf_linear_svm')]
    for panel,(ax,metric,title) in enumerate(zip(axes,['macro_f1','roc_auc'],['A  Macro F1','B  ROC-AUC'])):
        for y,(target,source,model,prefix) in zip(positions,groups):
            wn=prefix+'_holdout_results';tn=prefix+'_transfer_results';names.extend([wn,tn])
            w={'dataset':target,'condition':'natural'};t={'source_dataset':source,'target_dataset':target,'condition':'natural','scope':'same_target_holdout'}
            wr,tr=row(wn,**w),row(tn,**t)
            assert wr.evaluation_uid_sha256==tr.evaluation_uid_sha256 and wr.N==tr.N
            within=value(2,wn,w,metric);transfer=value(2,tn,t,metric)
            ax.plot([transfer,within],[y,y],color='#B2B8BC',lw=1.3,zorder=1)
            ax.plot(within,y,marker='o',color=INK,ms=6,zorder=3)
            ax.plot(transfer,y,marker='s',color=GRAY,mfc='white',ms=6,zorder=3)
            ax.text(within,y+.20,f'{within:.3f}',ha='center',fontsize=8.5)
            ax.text(transfer,y-.29,f'{transfer:.3f}',ha='center',fontsize=8.5)
            if panel==0:labels.append(target+' target\n'+model)
        ax.set_title(title,loc='left',fontweight='bold',pad=16)
        ax.set_xlim(0,1.03);ax.set_xticks([0,.25,.5,.75,1],['0','.25','.50','.75','1'])
        ax.set_ylim(-.5,4.05);ax.set_xlabel('Score');ax.grid(axis='x',color='#E5E7E9',lw=.55)
        ax.axhline(1.85,color='#D2D5D7',lw=.7);ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0,pad=10)
    axes[0].set_yticks(positions,labels)
    fig.text(.025,.035,'Identical target holdouts: CoAID, N = 702; FakeHealth, N = 330.\nConnecting lines show descriptive score differences; no uncertainty intervals.',fontsize=9,color=GRAY)
    # N labels are copied from and checked against the same rows.
    assert row('tfidf_logistic_holdout_results',dataset='CoAID',condition='natural').N==702
    assert row('tfidf_logistic_holdout_results',dataset='FakeHealth',condition='natural').N==330
    CHECKS['figure_2_identical_target_holdout_uids_and_N']=True
    save(fig,2,'figure_2_within_vs_cross_context_transfer',list(dict.fromkeys(names)),
         'Natural combined text; identical reserved target holdouts within each target comparison','None; descriptive point scores')


def figure3():
    n='text_shared_unigram_coefficients';a='text_coefficient_alignment_summary';d=data(n);summary=row(a,cutoff='all')
    assert len(d)==int(summary.N)==4174 and d.feature.is_unique
    assert not d.feature.str.contains(r'\s',regex=True).any()
    assert np.isfinite(d[['fakehealth_coefficient','coaid_coefficient']].to_numpy()).all()
    rev=d.fakehealth_sign.ne(d.coaid_sign)
    assert int(rev.sum())==int(summary.sign_reversal_N)
    assert np.array_equal(np.sign(d.fakehealth_coefficient),d.fakehealth_sign)
    assert np.array_equal(np.sign(d.coaid_coefficient),d.coaid_sign)
    fig,ax=plt.subplots(figsize=(5.1,4.65));fig.subplots_adjust(left=.18,right=.975,bottom=.155,top=.69)
    fig.suptitle('Lexical coefficient alignment',x=.055,y=.97,ha='left',fontsize=12,fontweight='bold')
    pearson=value(3,a,{'cutoff':'all'},'pearson');spearman=value(3,a,{'cutoff':'all'},'spearman');reversal=value(3,a,{'cutoff':'all'},'sign_reversal_percent')
    fig.text(.055,.90,f'4,174 shared unigrams · Sign reversal: {reversal:.2f}%\nPearson: {pearson:.4f} · Spearman: {spearman:.4f}',fontsize=9.5,linespacing=1.5,va='top')
    for mask,marker,color,label in [(~rev,'o',GRAY,'Same direction'),(rev,'x',ORANGE,'Reversed sign')]:
        xy=d.loc[mask,['fakehealth_coefficient','coaid_coefficient']].to_numpy()
        coll=ax.scatter(xy[:,0],xy[:,1],s=12,marker=marker,linewidths=.55,
            alpha=.52,**({'facecolors':'none','edgecolors':color} if marker=='o' else {'color':color}),label=label)
        assert np.array_equal(np.asarray(coll.get_offsets()),xy)
    ax.axvline(0,color=INK,lw=.7,zorder=0);ax.axhline(0,color=INK,lw=.7,zorder=0)
    ax.set_xlabel('FakeHealth Logistic Regression coefficient')
    ax.set_ylabel('CoAID Logistic Regression coefficient')
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.55,.805),ncol=2,frameon=False,markerscale=1.7,fontsize=9)
    ax.set_xlim(-1.1,1.2);ax.set_xticks([-1,-.5,0,.5,1])
    ax.set_ylim(-10,4.5);ax.set_yticks([-10,-8,-6,-4,-2,0,2,4])
    CHECKS['figure_3_exact_shared_unigram_rows']=len(d)
    CHECKS['figure_3_offsets_equal_saved_coefficients']=True
    CHECKS['figure_3_no_regression_line_or_selected_word_labels']=True
    save(fig,3,'figure_3_lexical_alignment',[n,a],'All validated shared unigrams from natural within-context LR models',
         'None; saved descriptive coefficient correlations/sign statistics')


def figure4():
    n='covid_rumor_external_transfer_results';c='covid_rumor_external_source_comparison'
    fig,axes=plt.subplots(1,2,figsize=(7.2,4.25),sharey=True);fig.subplots_adjust(left=.085,right=.975,bottom=.27,top=.72,wspace=.16)
    fig.suptitle('Independent COVID-context external transfer',x=.025,y=.97,ha='left',fontsize=12,fontweight='bold')
    fig.text(.025,.895,'COVID-Rumor NEWS · Clean target N = 3,667 · Actual Fake prevalence = 82.22%',fontsize=9.5)
    for ax,metric,title in zip(axes,['roc_auc','macro_f1'],['A  ROC-AUC','B  Macro F1']):
        for offset,source,color,hatch in [(-.17,'CoAID',BLUE,''),(.17,'FakeHealth',ORANGE,'///')]:
            vals=[]
            for classifier in ['LR','SVM']:
                w={'source_dataset':source,'classifier':classifier,'scope':'clean_primary'};r=row(n,**w)
                assert r.N==3667 and round(100*r.actual_fake_prevalence,2)==82.22
                vals.append(value(4,n,w,metric))
                compare=row(c,classifier=classifier,scope='clean_primary',metric=metric)
                assert math.isclose(vals[-1],compare['coaid_source' if source=='CoAID' else 'fakehealth_source'],abs_tol=1e-14)
            bars=ax.bar(np.array([0,1])+offset,vals,width=.30,color=color,edgecolor=INK,linewidth=.55,hatch=hatch,label=source)
            ax.bar_label(bars,labels=[f'{v:.3f}' for v in vals],padding=4,fontsize=9)
        ax.set_title(title,loc='left',fontweight='bold',pad=12)
        ax.set_xticks([0,1],['Logistic\nRegression','Linear SVM']);ax.set_xlim(-.6,1.6);ax.set_ylim(0,1)
        ax.set_yticks([0,.25,.5,.75,1],['0','.25','.50','.75','1']);ax.set_axisbelow(True);ax.grid(axis='y',color='#E5E7E9',lw=.55)
    axes[0].axhline(.5,color=GRAY,lw=.8,ls='--',alpha=.75,zorder=0.8)
    axes[0].set_ylabel('Score')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.54,.015),ncol=2,frameon=False,title='Source training dataset',title_fontsize=9)
    selected=data(n).query("scope == 'clean_primary'")
    assert len(selected)==4 and selected.evaluation_uid_sha256.nunique()==1
    CHECKS['figure_4_clean_primary_identical_uids_N_3667']=True
    CHECKS['figure_4_no_error_bars']=True
    save(fig,4,'figure_4_external_covid_transfer',[n,c],'COVID-Rumor NEWS clean primary; identical target rows across four frozen source models','None; descriptive external point scores')


def robustness_table():
    master=data('manuscript_key_results').set_index('result_id')
    def v(r,fmt='.3f'):return format(master.loc[r,'estimate'],fmt)
    rows=[
    ['Engagement B/C specifications','Did entity/version choices alter the contrast?',f'Ratio-of-ratios: B {v("R004")}; C {v("R005")}; saved intervals below 1.','The primary ordering persisted.','Dependent sensitivity specifications; dataset/context confounding remains.'],
    ['Dominant-component exclusion','Was engagement sensitive to the largest overlap component?',f'Ratio-of-ratios {v("R006")} versus primary {v("R003")}.','The contrast attenuated without reversing.','Exclusion changes the reference population; dependence remains incompletely captured.'],
    ['Title-only transfer','Did removing bodies restore transfer?','Small macro-F1 gains accompanied lower ROC-AUC in both directions.','Title-only text did not consistently rescue transfer.','Target rows matched; source cohorts and selected configurations differed.'],
    ['FakeHealth publisher-held-out','Did signal persist for unseen publisher families?',f'Outer mean macro F1 {v("R027", ".4f")}; sample SD 0.0210.','Modest signal remained; performance was lower than the resource-grouped reference.','Unpaired comparison of evaluation designs; publisher/style confounding remains.'],
    ['Linear SVM replication','Did classical conclusions depend on LR?',f'Within macro F1: CoAID {v("R013")}, FakeHealth {v("R014")}; strong transfer degradation.','The within-context gap and transfer failure replicated.','Replication used the same datasets and resource-group holdouts.'],
    ['DistilRoBERTa primary','Did the transformer consistently improve transfer?','A modest macro-F1 gain in one direction coexisted with lower ROC-AUC/AP in both.','No consistent transfer improvement was observed.','Single-seed primary; architecture and available text budget differ from TF-IDF.'],
    ['Transformer budget/seed sensitivity','Was the shorter-cap improvement stable?',f'Paired macro-F1 change {v("R036", "+.4f")}; SD 0.0790; positive in 3/3 seeds. Ranking gains: 1/3.','Threshold gains persisted; ranking gains were seed-sensitive.','One transfer direction; seed SD is variability, not a confidence interval.'],
    ['Independent COVID NEWS target','Was CoAID-source transfer better on independent COVID text?',f'ROC-AUC: LR {v("R040")} vs {master.loc["R040","comparator_estimate"]:.3f}; SVM {v("R043")} vs {master.loc["R043","comparator_estimate"]:.3f}.','Both classifiers showed greater CoAID-source portability.','Descriptive comparison; high Fake prevalence and source/label/genre differences.']]
    assert 'sample SD 0.0210' in master.loc['R027','uncertainty_or_variability']
    assert 'sample SD 0.0790' in master.loc['R036','uncertainty_or_variability']
    for rid,count in [('R036',3),('R037',1),('R038',1)]:assert f'positive seeds {count}' in master.loc[rid,'uncertainty_or_variability']
    columns=['Analysis','Primary question','Headline result','Interpretation','Key limitation']
    d=pd.DataFrame(rows,columns=columns)
    path=TAB/'table_1_robustness_summary.csv';d.to_csv(path,index=False);OUTPUTS.append(path)
    header='# Table 1. Robustness and sensitivity summary\n\n'
    md='| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'
    md+='\n'.join('| '+' | '.join(r)+' |' for r in rows)
    md+='\n\nAll entries summarize validated results. Fold/seed SDs describe variability, not confidence intervals. Title comparisons use matched target rows; publisher comparisons are unpaired. LR = Logistic Regression; AP = average precision. In the final row, CoAID source is listed before FakeHealth source.\n'
    path=TAB/'table_1_robustness_summary.md';path.write_text(header+md);OUTPUTS.append(path)
    for p in OUTPUTS[-2:]:MANIFEST.append(dict(artifact_id='Table 1',file_path=rel(p),format=p.suffix[1:],width_inches='',height_inches='',dpi='',authoritative=True,
       evaluation_scope='Compact summary of validated sensitivity scopes',uncertainty='Saved bootstrap intervals referenced; fold/seed SD explicitly variability',
       source_artifacts='outputs/tables/manuscript_key_results.csv;outputs/tables/manuscript_claim_evidence_matrix.csv',sha256=sha(p)))
    CHECKS['table_1_rows']=len(d)


def captions():
    e=row('cross_context_engagement_contrast',coaid_specification='A_primary_strict')
    lexical=row('text_coefficient_alignment_summary',cutoff='all')
    content=f'''# Main-manuscript figure captions

## Figure 1. Cross-context observed post engagement

Points show the standardized Fake/Real expected unique observed-post ratios for primary CoAID News Specification A and FakeHealth HealthStory, including content items with zero observed posts. The x-axis is logarithmic; the dashed reference marks a ratio of one. Intervals are 95% independent overlap-cluster bootstrap percentile intervals: resampling was conducted within each dataset's observed overlap components, and the cross-context contrast used independent dataset bootstrap distributions. Each context retained its own standardization population. The ratio-of-ratios was {e.ratio_of_ratios:.3f} (95% percentile interval {e.CI_lower_ratio:.3f}–{e.CI_upper_ratio:.3f}). A smaller ratio denotes a larger relative Fake disadvantage. These are observed engagement/sharing counts rather than complete diffusion cascades; the contrast is observational. Within-context endpoints were taken directly from the final version-aware engagement results, not inferred from the cross-context interval.

Sources: `outputs/tables/version_aware_engagement_results.csv` (unique_post_count, A_primary_strict and FakeHealth_HealthStory); `outputs/tables/cross_context_engagement_contrast.csv` (A_primary_strict).

## Figure 2. Within-context and cross-context text performance

Panel A shows macro F1 and Panel B ROC-AUC for natural combined-text TF-IDF Logistic Regression and Linear SVM. Filled circles denote within-context evaluation; open squares denote transfer from the other dataset. Within and transfer scores use identical target holdout rows within each target dataset: CoAID target N = 702 and FakeHealth target N = 330. Within-context models used their dataset's training partition; transferred models used the full eligible source cohort with the locked source configuration. Macro F1 averages class-specific F1 scores; ROC-AUC summarizes ranking with Fake as the positive class. Connecting segments display descriptive differences and are not confidence intervals or significance tests. Source labels, text distributions and collection procedures differ across datasets.

Sources: `outputs/tables/tfidf_logistic_holdout_results.csv`, `tfidf_logistic_transfer_results.csv`, `tfidf_linear_svm_holdout_results.csv`, and `tfidf_linear_svm_transfer_results.csv`; natural condition, matched target holdouts.

## Figure 3. Dataset-specific lexical coefficient alignment

Each point represents one of the {int(lexical.N):,} validated shared unigrams in the natural within-context Logistic Regression models. Axes show the saved FakeHealth and CoAID coefficients; horizontal and vertical reference lines mark zero. Open circles indicate equal coefficient signs and crosses indicate reversed signs. Positive coefficients indicate association with Fake and negative coefficients association with Real. The saved Pearson correlation was {lexical.pearson:.4f}, Spearman correlation {lexical.spearman:.4f}, and overall sign reversal {lexical.sign_reversal_percent:.2f}%. Coefficients are dataset-specific conditional associations, not causal effects; vocabularies, IDF scales, correlated predictors and regularization differed. Coefficient magnitudes are model-specific and are not directly comparable across axes because the fitted models differ in feature space and regularization. The figure is intended to visualize directional alignment and sign reversal rather than equality of coefficient effect size. No regression line was fitted, no words were selected for annotation, and no coefficient or correlation was re-estimated.

Sources: `outputs/tables/text_shared_unigram_coefficients.csv` (entire validated shared cohort) and `outputs/tables/text_coefficient_alignment_summary.csv` (cutoff = all).

## Figure 4. Independent COVID-context external transfer

Panel A shows ROC-AUC and Panel B macro F1 on the identical clean COVID-Rumor NEWS primary target (N = 3,667; actual Fake prevalence = 82.22%). Solid bars indicate CoAID-source models and hatched bars FakeHealth-source models, shown separately for Logistic Regression and Linear SVM. Only verified binary NEWS statements passing the frozen quality and exact-overlap rules were evaluated. Frozen source vectorizers, models and thresholds were reused without target adaptation. CoAID-source scores were higher for both classifiers, particularly in ROC-AUC; absolute threshold performance remained modest and all models overpredicted Fake. Bars are point estimates without error bars because uncertainty intervals were not estimated. The dashed horizontal reference in Panel A marks ROC-AUC = 0.5; no corresponding reference is shown for macro F1. This descriptive comparison is consistent with greater portability of learned Fake/Real linguistic associations across COVID-focused datasets but does not isolate a causal COVID-period effect. Label construction, short-statement genre and source composition remain alternative explanations.

Sources: `outputs/tables/covid_rumor_external_transfer_results.csv` and `outputs/tables/covid_rumor_external_source_comparison.csv`; scope = clean_primary only.

## Table 1. Robustness and sensitivity summary

The table summarizes the question, headline result, interpretation and residual limitation for the validated robustness analyses. It distinguishes engagement specification/influence checks, representation and publisher sensitivities, classifier replication, transformer budget/seed variability, and independent COVID external validation. Fold and seed SDs describe variability and are not confidence intervals. Different evaluation scopes are retained rather than pooled into a common performance estimate.
'''
    path=OUT/'figure_captions.md';path.write_text(content);OUTPUTS.append(path)


def supplement():
    text='''# Supplementary figure plan

These six displays are recommended only for robustness and reproducibility support. None was generated in this task. Use saved aggregate results; do not rerun analyses.

| Proposed display | Purpose and scope | Validated inputs | Reporting boundary |
| --- | --- | --- | --- |
| S1. Engagement specification and influence checks | Show primary A, B/C, and dominant-component exclusion together; retain tail checks as point-only where intervals are absent. | cross_context_engagement_contrast.csv; cross_context_sensitivity_results.csv | Exclusions change populations; no fabricated tail intervals or independent-replication claim. |
| S2. Title-only versus combined-text transfer | Contrast macro F1 and ROC-AUC on identical title-eligible target holdout rows. | tfidf_logistic_title_vs_combined.csv | Small F1 improvements coexist with poorer ranking; source training cohorts/configurations differ. |
| S3. Publisher-held-out fold performance | Display each FakeHealth outer fold plus mean and pooled summaries, with masking sensitivity. | fakehealth_publisher_heldout_results.csv; fakehealth_publisher_heldout_summary.csv | Fold range/SD is variability; pooled ranking combines models; no equivalent CoAID claim. |
| S4. Transformer budget/seed paired differences | Show paired 256-minus-512 differences by seed for macro F1, ROC-AUC and AP in FakeHealth-to-CoAID transfer. | distilroberta_budget_seed_paired_deltas.csv; distilroberta_budget_seed_summary.csv | Preserve metric-specific direction changes; no CI or population seed-generalization claim. |
| S5. Transfer confusion matrices | Expose asymmetric class-prediction errors for matched classical holdouts and the clean COVID target using saved TN/FP/FN/TP. | tfidf_logistic_transfer_results.csv; tfidf_linear_svm_transfer_results.csv; covid_rumor_external_transfer_results.csv | Keep populations separate; show saved counts without new prediction evaluation. |
| S6. Predicted versus actual Fake prevalence | Compare fixed-threshold class proportions for each transfer direction and external source/model combination; include vocabulary coverage as a supporting table if useful. | Same transfer results; covid_rumor_external_vocabulary_diagnostics.csv | Prevalence departures are descriptive; no calibration or threshold changes. |

All input filenames refer to outputs/tables/. Existing final displays can be assessed for reuse during supplement layout. Main-paper captions and figures remain the authoritative visual summary of the selected primary scopes.
'''
    p=OUT/'supplementary_figure_plan.md';p.write_text(text);OUTPUTS.append(p)


def main():
    if VALID.exists():
        d=json.loads(VALID.read_text());assert d['script_sha256']==sha(__file__)
        for p,h in {**d['source_sha256'],**d['output_sha256']}.items():assert sha(ROOT/p)==h,p
        print('Completed figure bundle verified; no files modified.');return
    names=['figure_captions.md','figure_manifest.csv','supplementary_figure_plan.md','figure_values.csv']
    assert not any((OUT/n).exists() for n in names),'Refusing overwrite of existing manuscript visual outputs'
    assert not FIG.exists() or not any(FIG.iterdir()),'Existing manuscript figure directory is nonempty'
    assert not TAB.exists() or not any(TAB.iterdir()),'Existing manuscript table directory is nonempty'
    init_sources();FIG.mkdir(parents=True,exist_ok=True);TAB.mkdir(parents=True,exist_ok=True)
    figure1();figure2();figure3();figure4();robustness_table();captions();supplement()
    path=OUT/'figure_values.csv';pd.DataFrame(VALUES).to_csv(path,index=False);OUTPUTS.append(path)
    path=OUT/'figure_manifest.csv';pd.DataFrame(MANIFEST).to_csv(path,index=False);OUTPUTS.append(path)
    assert all(sha(ROOT/p)==h for p,h in SOURCES.items()),'Source or Results prose changed'
    assert len([p for p in OUTPUTS if p.suffix in ['.svg','.pdf','.png']])==12
    CHECKS.update(source_fingerprints_verified=True,validated_tables_unchanged=True,results_prose_unchanged=True,
        vector_exports_contain_no_raster_images=True,png_exports_300_dpi=True,no_new_models=True,no_new_bootstrap=True,
        no_new_predictions=True,no_new_inferential_statistics=True,no_SD_as_CI=True,no_significance_markers=True,
        readable_shape_and_hatch_encodings=True,lexical_labels_omitted=True)
    payload=dict(validation='passed',script_sha256=sha(__file__),source_sha256=SOURCES,
        output_sha256={rel(p):sha(p) for p in OUTPUTS},checks=CHECKS,main_figures=4,table_rows=8,
        numerical_provenance='figure_values.csv records displayed aggregate values; Figure 3 offsets exactly equal all saved coefficient pairs, keyed by unique feature in the fingerprinted source table.',
        engagement_provenance='Within-context estimates/CIs match version_aware_engagement_results.csv, whose hash was recorded by the completed cross-context analysis. Final cross-context contrast table matches the frozen manuscript-synthesis fingerprint.',
        visual_review='Pending inspection of PNG exports; automated text-bounds checks passed.',
        scientific_boundaries='Descriptive observational comparisons; no causal effect, inferred uncertainty or statistical winner.')
    VALID.write_text(json.dumps(payload,indent=2)+'\n')
    print('Created 4 figures × SVG/PDF/300-dpi PNG, 8-row robustness table, captions, manifest and supplementary plan. Automated validation passed.')


if __name__=='__main__':main()
