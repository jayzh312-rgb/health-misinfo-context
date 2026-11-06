"""Assemble final aggregate evidence only. No model, raw data, predictions or inference.

Run with .venv/bin/python src/build_manuscript_results_synthesis.py.
The output bundle is immutable on rerun; source and output fingerprints are checked.
"""
from pathlib import Path
import hashlib
import json
import math
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
TABLES=ROOT/'outputs/tables'
REPORTS=ROOT/'outputs/reports'
VALIDATION=REPORTS/'manuscript_synthesis_validation.json'
KEY=TABLES/'manuscript_key_results.csv'
CLAIMS=TABLES/'manuscript_claim_evidence_matrix.csv'
LIMITATIONS=TABLES/'manuscript_limitations.csv'
REPORT=REPORTS/'manuscript_results_synthesis.md'
PLAN=REPORTS/'manuscript_figure_plan.md'
LEDGER=TABLES/'manuscript_numeric_provenance.csv'
AGENTS=ROOT/'AGENTS.md'
METANAMES=['tfidf_logistic_baseline','tfidf_logistic_cross_context_transfer',
 'cross_context_lexical_alignment','tfidf_logistic_title_sensitivity','fakehealth_publisher_heldout',
 'tfidf_linear_svm_replication','distilroberta_text_experiment','distilroberta_budget_seed_stability',
 'covid_rumor_external_transfer','transformer_feasibility']
SNAPSHOT={}
EXPECTED={}
CACHE={}
ACCESSES=[]
RESULTS=[]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rel(path): return str(Path(path).relative_to(ROOT))
def remember(path):
    SNAPSHOT[rel(path)]=sha(path)
    return path


def metadata():
    def collect(value):
        if isinstance(value,dict):
            for k,v in value.items():
                if k.startswith(('outputs/tables/','outputs/reports/','outputs/figures/')) and isinstance(v,str) and len(v)==64:
                    EXPECTED.setdefault(k,set()).add(v)
                else: collect(v)
        elif isinstance(value,list):
            for x in value: collect(x)
    for name in METANAMES:
        p=remember(REPORTS/(name+'.json'));d=json.loads(p.read_text())
        if name=='distilroberta_text_experiment':
            assert d['status']=='complete' and d['post_render_validation']['artifact_hashes_verified']
            assert d['tests']['failures']==d['tests']['errors']==0
        elif name=='distilroberta_budget_seed_stability': assert d['validation']['passed']
        elif name=='transformer_feasibility': assert d['validation'].startswith('passed')
        else: assert d['validation']=='passed',name
        collect(d)
    p=remember(TABLES/'cross_context_engagement_metadata.json');d=json.loads(p.read_text())
    assert d['complete'] and d['input_fingerprints_verified'] and d['no_input_changes']
    collect(d)
    p=remember(TABLES/'cross_context_input_validation.csv');d=pd.read_csv(p)
    assert d.saved_interval_reproduced.all() and d.finite_positive_ratios.all()
    assert not d.original_models_refitted.any()


def load(name):
    if name not in CACHE:
        path=remember(TABLES/(name+'.csv'))
        if not name.startswith('cross_context_'):
            assert rel(path) in EXPECTED, 'No validation fingerprint for '+name
            assert sha(path) in EXPECTED[rel(path)], 'Changed validated artifact: '+name
        CACHE[name]=pd.read_csv(path)
    return CACHE[name]


def select(name,**where):
    d=load(name)
    for key,value in where.items(): d=d[d[key].eq(value)]
    assert len(d)==1,(name,where,len(d))
    return d.iloc[0]


def native(v):
    return v.item() if hasattr(v,'item') else v


def cell(name,where,field):
    value=native(select(name,**where)[field])
    ACCESSES.append(dict(source_artifact='outputs/tables/'+name+'.csv',row_selector=json.dumps(where,sort_keys=True),
                         field=field,value=value))
    return value


def ref(name,where,field):
    return 'outputs/tables/'+name+'.csv'+'#'+json.dumps(dict(row=where,field=field),sort_keys=True,separators=(',',':'))


def add(family,source,target,model,scope,metric,name,where,field=None,uncertainty='Not estimated; descriptive point estimate',
        comp=None,interpretation='descriptive predictive association',role='primary',limits='L01;L02;L05',rid=None):
    field=field or metric
    row=dict(result_id=rid or f'R{len(RESULTS)+1:03d}',analysis_family=family,
      **{'dataset/source':source,'target/context':target,'model/estimand':model},evaluation_scope=scope,
      primary_metric=metric,estimate=cell(name,where,field),uncertainty_or_variability=uncertainty,
      comparator='',comparator_estimate='',interpretation_class=interpretation,primary_or_sensitivity=role,
      limitation_code=limits,source_artifact=ref(name,where,field))
    if comp:
        label,cn,cw,cf=comp
        row.update(comparator=label,comparator_estimate=cell(cn,cw,cf))
        row['source_artifact']+=' | comparator='+ref(cn,cw,cf)
    RESULTS.append(row)
    return row


def md(rows,columns):
    def clean(value): return str(value).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+
      ['| '+' | '.join(clean(r[c]) for c in columns)+' |' for r in rows])


def source_link(name): return f'[{name}.csv](../tables/{name}.csv)'
def s(name,where,field,fmt='.4f'): return format(cell(name,where,field),fmt)


def build_results():
    eng='cross_context_engagement_contrast';a={'coaid_specification':'A_primary_strict'}
    for ds,field in [('FakeHealth','fakehealth_ratio'),('CoAID','coaid_ratio')]:
        add('engagement',ds,ds,'standardized Fake/Real expected unique observed-post ratio','all content including zero posts',
            'observed_post_ratio',eng,a,field,uncertainty='See cross-context interval; separate within-context interval not imported',
            interpretation='observational standardized association',limits='L01;L02;L03;L04;L10;L11')
    for spec,role in [('A_primary_strict','primary'),('B_resource_sensitivity','sensitivity'),('C_high_certainty','sensitivity')]:
        w={'coaid_specification':spec}
        ci=f"95% independent overlap-cluster bootstrap percentile interval [{s(eng,w,'CI_lower_ratio')}, {s(eng,w,'CI_upper_ratio')}]"
        add('engagement','CoAID / FakeHealth','separate dataset reference populations','ratio of standardized Fake/Real observed-post ratios',spec,
            'ratio_of_ratios',eng,w,uncertainty=ci,interpretation='observational cross-dataset contrast',role=role,limits='L01;L02;L03;L04;L10;L11')
    n='cross_context_sensitivity_results';w={'sensitivity':'exclude_dominant_component'}
    add('engagement','CoAID / FakeHealth','CoAID dominant component excluded','ratio of standardized observed-post ratios','changed reference population',
        'ratio_of_ratios',n,w,uncertainty=f"95% saved cluster-percentile interval [{s(n,w,'CI_lower_ratio')}, {s(n,w,'CI_upper_ratio')}]",
        comp=('full retained primary',eng,a,'ratio_of_ratios'),interpretation='descriptive influence sensitivity',role='sensitivity',limits='L01;L03;L04;L11')
    for model,prefix in [('LR','tfidf_logistic'),('Linear SVM','tfidf_linear_svm')]:
        for ds in ['CoAID','FakeHealth']:
            w={'dataset':ds,'condition':'natural'}
            add('within_context_text',ds,ds,model,'reserved resource/group holdout','macro_f1',prefix+'_holdout_results',w)
        for ds,target,direction in [('CoAID','FakeHealth','CA_to_FH'),('FakeHealth','CoAID','FH_to_CA')]:
            w={'source_dataset':ds,'condition':'natural','scope':'same_target_holdout'}
            for metric in ['macro_f1','roc_auc']:
                add('cross_context_text',ds,target,model,'identical target holdout to within-context evaluation',metric,
                    prefix+'_transfer_results',w,comp=('target-trained within-context '+model,prefix+'_holdout_results',{'dataset':target,'condition':'natural'},metric))
    n='text_coefficient_alignment_summary'
    for cutoff,metric in [('all','pearson'),('all','spearman'),('all','sign_reversal_percent'),('top100','sign_reversal_percent')]:
        add('lexical_instability','FakeHealth and CoAID','shared unigram intersection','natural within-context LR coefficients',cutoff,
            metric,n,{'cutoff':cutoff},interpretation='descriptive coefficient alignment',role='supporting',limits='L01;L02;L05;L12')
    n='tfidf_logistic_title_vs_combined'
    for direction,target in [('CA_to_FH','FakeHealth'),('FH_to_CA','CoAID')]:
        for metric in ['macro_f1','roc_auc']:
            w=dict(evaluation='transfer',direction=direction,condition='natural',scope='same_target_holdout',metric=metric)
            add('title_sensitivity',direction.split('_to_')[0],target,'title-only LR','identical title-eligible target holdout',metric,n,w,'title',
                comp=('combined-text LR on identical target rows',n,w,'combined_same_rows'),role='sensitivity',limits='L01;L02;L06;L13')
    n='fakehealth_publisher_heldout_summary'
    for metric in ['macro_f1','roc_auc']:
        w={'condition':'natural','metric':metric}
        variability=f"Three outer folds; sample SD {s(n,w,'outer_sample_sd')}; range [{s(n,w,'outer_min')}, {s(n,w,'outer_max')}]; not a CI"
        add('publisher_sensitivity','FakeHealth','unseen publisher families','nested publisher-group LR','unweighted outer-fold mean',metric,n,w,'outer_mean',
            uncertainty=variability,role='sensitivity',limits='L02;L05;L13')
    n='fakehealth_publisher_heldout_results'
    add('publisher_sensitivity','FakeHealth','unseen publisher families','source-masked LR','pooled outer predictions','macro_f1',n,
        {'condition':'source_masked','scope':'pooled'},comp=('natural LR, identical publisher folds',n,{'condition':'natural','scope':'pooled'},'macro_f1'),role='sensitivity',limits='L02;L05;L13')
    for ds in ['CoAID','FakeHealth']:
        add('transformer',''+ds,ds,'DistilRoBERTa, 512 tokens, seed 42','reserved resource/group holdout','macro_f1','distilroberta_holdout_results',
            {'target_dataset':ds,'max_length':512},comp=('natural LR, same target holdout','tfidf_logistic_holdout_results',{'dataset':ds,'condition':'natural'},'macro_f1'),role='sensitivity',limits='L01;L02;L06;L07;L14')
    for ds,target in [('CoAID','FakeHealth'),('FakeHealth','CoAID')]:
        for metric in ['macro_f1','roc_auc']:
            add('transformer',ds,target,'DistilRoBERTa, 512 tokens, seed 42','same_target_holdout',metric,'distilroberta_transfer_results',
                {'source_dataset':ds,'max_length':512,'scope':'same_target_holdout'},
                comp=('natural LR, identical target holdout','tfidf_logistic_transfer_results',{'source_dataset':ds,'condition':'natural','scope':'same_target_holdout'},metric),
                role='sensitivity',limits='L01;L02;L06;L07;L14')
    n='distilroberta_budget_seed_summary'
    for metric in ['macro_f1','roc_auc','average_precision']:
        w={'kind':'paired_delta','scope':'same_target_holdout','metric':metric}
        uncertainty=f"{s(n,w,'N_seeds','.0f')} seeds; sample SD {s(n,w,'sample_sd')}; range [{s(n,w,'minimum')}, {s(n,w,'maximum')}]; positive seeds {s(n,w,'positive_seeds','.0f')}; not a CI"
        add('transformer_budget_seed','FakeHealth','CoAID','DistilRoBERTa; paired 256 minus 512 token cap','same_target_holdout',
            'mean_paired_'+metric+'_change',n,w,'mean',uncertainty=uncertainty,role='sensitivity',limits='L06;L07;L14')
    n='covid_rumor_external_transfer_results'
    for model in ['LR','SVM']:
        for metric in ['macro_f1','roc_auc','balanced_accuracy']:
            add('external_COVID','CoAID','COVID-Rumor NEWS','TF-IDF '+model,'identical clean external target',metric,n,
                {'classifier':model,'source_dataset':'CoAID','scope':'clean_primary'},
                comp=('FakeHealth-source '+model+', identical clean target',n,{'classifier':model,'source_dataset':'FakeHealth','scope':'clean_primary'},metric),
                interpretation='descriptive external validation',role='external robustness',limits='L01;L02;L05;L08;L09;L15')
    n='covid_rumor_external_vocabulary_diagnostics'
    for metric in ['target_token_occurrence_coverage_percent','zero_feature_rows']:
        add('external_COVID_representation','CoAID','COVID-Rumor NEWS','LR/SVM source TF-IDF; identical vocabulary per source','clean_primary',metric,n,
            {'classifier':'LR','source_dataset':'CoAID','scope':'clean_primary'},
            comp=('FakeHealth source',n,{'classifier':'LR','source_dataset':'FakeHealth','scope':'clean_primary'},metric),
            interpretation='descriptive representation diagnostic',role='external robustness',limits='L06;L08;L15')
    return pd.DataFrame(RESULTS)


def build_limitations():
    rows=[
    ('L01','Dataset/context confounding','All cross-context comparisons','Direction cannot be separated from dataset construction','Within/transfer contrasts; independent COVID external target','Period/context, dataset, collection procedure and source composition remain inseparable.'),
    ('L02','Label-definition mismatch','All engagement and text comparisons','Direction unknown; target constructs differ','Original definitions retained; consistent analytical Fake=1/Real=0 coding; unverified NEWS excluded','FakeHealth expert-review quality framework, CoAID construction/veracity labels and COVID-Rumor fact-check-associated veracity are not identical ground truth.'),
    ('L03','Incomplete diffusion/network coverage','Engagement','Observed counts omit activity; direction of relative bias unknown','Unique observed-post estimand; zeros retained; observed-overlap dependence modeled','Counts do not recover complete cascades, exposures, network topology or population reach.'),
    ('L04','CoAID dependence and dominant overlap component','Engagement contrast and uncertainty','Removing the dominant component attenuates the contrast; direction remains unchanged','Overlap-cluster bootstrap; A/B/C specifications; dominant-component exclusion sensitivity','Graph misses other dependence; large component challenges many-small-cluster approximations; exclusion changes population; cross-dataset independence is assumed.'),
    ('L05','Source/publisher confounding','Text and cross-context comparisons','May contribute to predictability; full contribution unknown','Explicit-source masking; FakeHealth nested publisher-held-out evaluation','Style/topic remain; publisher families are proxies; masking uses label-blind evaluation metadata; no comparable unseen-publisher validation of CoAID.'),
    ('L06','Different article/text lengths','Classical and transformer transfer; lexical audit','Direction unknown; information budgets and genre differ','Matched-target title-only sensitivity; vocabulary diagnostics; token-budget sensitivity','Title-only changes training cohorts and tuned configurations; full-text TF-IDF and truncated encoders see different representations.'),
    ('L07','Transformer truncation asymmetry','DistilRoBERTa','FakeHealth loses more body text; performance direction varies by seed and metric','Full-title packing; paired token caps; identical CoAID target tokens verified','Source-budget changes interact with optimization; shorter cap is not a matched-information correction.'),
    ('L08','COVID-Rumor short-statement genre','External COVID transfer','Direction unknown; short claims differ from source articles','One NEWS statement per item; content-quality audit; source-only transforms','Cross-dataset portability cannot isolate COVID vocabulary from genre, fact-check framing or collection.'),
    ('L09','Target class imbalance','All transfer; especially external NEWS','High Fake prevalence can support high Fake F1/AP despite modest minority-class or threshold performance','Macro F1, balanced accuracy, ROC-AUC, class metrics and prevalence reported; natural prevalence retained','Fixed thresholds are not calibrated to external target prevalence; no claim of deployment readiness.'),
    ('L10','Historical platform missingness/deletions','Engagement and dataset coverage','Missing activity can reduce observed counts; differential Fake/Real/context direction unknown','Explicit observed-activity estimand; collection/coverage caveats retained','Deleted, private, uncaptured and unavailable activity cannot be recovered from these artifacts.'),
    ('L11','Non-causal observational interpretation','Integrated engagement and text argument','Causal direction and mechanisms unidentified','Descriptive contrasts; no causal p-value or causal identification claim','Text findings do not establish mediation or explain the engagement contrast.'),
    ('L12','Conditional coefficient comparability','Lexical instability','Magnitude/sign depend on feature sets, IDF, regularization and correlated predictors','Shared-unigram restriction; prespecified importance cutoffs; sign/rank summaries','Opposite coefficients are associations; correlations do not identify causes of transfer failure or causal words.'),
    ('L13','Non-equivalent sensitivity evaluation designs','Publisher and title comparisons','Direction depends on aggregation, populations and training protocol','Title comparisons use identical target rows; publisher comparisons explicitly unpaired','Publisher outer CV differs from original holdout/CV; pooled scores combine models; fold SD is not a CI.'),
    ('L14','Limited transformer seed coverage','Transformer robustness','Ranking effects reverse across seeds while macro F1 improves at shorter cap','Paired seed summaries for FakeHealth-to-CoAID; original seed reused','Limited seeds and one direction; primary experiment otherwise single-seed; MPS not asserted bitwise reproducible; no population uncertainty claim.'),
    ('L15','Residual external-target overlap/language uncertainty','COVID-Rumor external transfer','Residual bias direction unknown','Exact text/resource exclusions; approximate flags and sensitivity; repository English designation plus script-quality screen','Offline URLs cannot resolve all redirects; conservative normalization misses some equivalent text; English script is not full language identification.')]
    return pd.DataFrame(rows,columns=['limitation_code','limitation','analyses_affected','likely_direction_or_impact','mitigation_already_performed','residual_concern'])


def build_claims():
    rows=[
    ('C01','The relative Fake–Real observed-post gap differs across main contexts.','cross_context_engagement_contrast.csv; cross_context_sensitivity_results.csv','Dataset/context and labels confounded; dependence/coverage limits','Observational contrast with saved cluster-percentile interval','The standardized Fake/Real observed-post ratio is lower in CoAID than FakeHealth, indicating a larger relative Fake disadvantage in the CoAID dataset context.','COVID caused lower misinformation diffusion.'),
    ('C02','CoAID is easier to classify within context than FakeHealth.','tfidf_logistic_holdout_results.csv; tfidf_linear_svm_holdout_results.csv','Selected cohorts, source composition and label schemes differ','Descriptive replication across classical models','Both LR and Linear SVM achieve substantially higher within-context macro F1 in CoAID than FakeHealth.','COVID misinformation is inherently easier to detect.'),
    ('C03','Cross-context text associations have limited portability.','tfidf_logistic_transfer_results.csv; tfidf_linear_svm_transfer_results.csv; corresponding holdout/degradation tables','Matched target rows improve comparability but source populations and labels differ','Descriptive within-versus-transfer degradation','Both classical classifiers show substantial macro-F1 degradation when trained in the other dataset context.','All misinformation detectors fail outside their training domain.'),
    ('C04','Signed lexical coefficients align weakly across datasets.','text_coefficient_alignment_summary.csv','Coefficients conditional on distinct vocabularies/IDF/regularization; no causal decomposition','Descriptive coefficient instability','Shared-unigram LR coefficients are approximately uncorrelated, and sign reversals remain common among highly ranked shared terms.','Individual words are universal misinformation cues or causal signaling words.'),
    ('C05','Title-only text does not consistently rescue transfer.','tfidf_logistic_title_vs_combined.csv','Small macro-F1 gains coexist with lower ranking scores; training cohorts/configurations differ','Matched-target descriptive sensitivity','Title-only transfer shows small macro-F1 gains but poorer ROC-AUC in both directions, providing no consistent rescue.','Removing article bodies never improves any metric, or isolates the effect of length.'),
    ('C06','FakeHealth retains modest association under unseen publishers.','fakehealth_publisher_heldout_summary.csv; fakehealth_publisher_heldout_results.csv; fakehealth_publisher_heldout_baseline_comparison.csv','Publisher proxies incomplete; outer CV not paired with original holdout; no CoAID publisher test','Nested publisher-held-out descriptive evidence','FakeHealth retains modest predictive signal across unseen publisher families, although performance is lower than the resource-grouped baseline.','Publisher confounding has been eliminated or cannot explain CoAID performance.'),
    ('C07','Explicit source masking leaves residual signal.','fakehealth_publisher_heldout_results.csv','Label-blind mask uses evaluation metadata; style/topic and other identifiers can remain','Narrow source-identifier sensitivity','Similar pooled natural and masked publisher-held-out macro F1 suggests known explicit identifiers alone do not account for the surviving FakeHealth signal.','Source-independent linguistic misinformation detection is established.'),
    ('C08','Primary DistilRoBERTa does not consistently improve cross-context transfer.','distilroberta_transfer_results.csv; tfidf_logistic_transfer_results.csv; tfidf_linear_svm_transfer_results.csv','Single-seed primary; unequal token information; macro F1 and ranking disagree','Descriptive model-family robustness','The primary transformer offers no consistent transfer benefit: a modest macro-F1 gain in one direction coexists with lower ROC-AUC and AP in both directions.','Transformers always underperform linear models on every metric.'),
    ('C09','Transformer budget effects are seed- and metric-sensitive.','distilroberta_budget_seed_summary.csv; distilroberta_budget_seed_paired_deltas.csv','Limited seeds; only FakeHealth-to-CoAID; threshold/ranking effects differ','Paired descriptive variability; no inferential claim','The shorter cap improves macro F1 across tested seeds, but ROC-AUC and AP gains occur only for the original seed; the joint improvement is not stable.','Shorter source text consistently improves cross-context representation or all budget effects reverse.'),
    ('C10','COVID-focused source associations transfer better to independent COVID NEWS.','covid_rumor_external_transfer_results.csv; covid_rumor_external_source_comparison.csv','High Fake prevalence; modest threshold performance; source/label/genre differences','Descriptive external validation; no statistical winner','Both LR and SVM show greater CoAID-source than FakeHealth-source portability to independent COVID-Rumor NEWS, especially in ROC-AUC.','A causal sensitive-period effect or universal COVID misinformation vocabulary has been identified.'),
    ('C11','External differences are not explained by completely empty representations.','covid_rumor_external_vocabulary_diagnostics.csv','Nonzero vectors and higher coverage do not guarantee semantic alignment','Representation diagnostic','No clean target row has zero TF-IDF features; CoAID has somewhat higher token coverage, which may contribute to its transfer advantage.','Vocabulary coverage fully explains or rules out representation-related transfer failure.'),
    ('C12','Engagement and linguistic findings support a context-dependent interpretation.','C01–C11; manuscript_key_results.csv','No causal identification, mediation or item-level text–engagement linkage tested','Integrated observational interpretation','Observed Fake/Real engagement contrasts and the portability of learned linguistic associations vary across dataset contexts; independent COVID validation is consistent with greater portability within COVID-focused environments.','Lexical changes caused the engagement contrast, or COVID caused either pattern.')]
    return pd.DataFrame(rows,columns=['claim_id','proposed_claim','evidence_supporting','evidence_limiting','allowed_strength','recommended_wording','prohibited_overstatement'])


def manuscript(limitations,claims):
    e='cross_context_engagement_contrast';a={'coaid_specification':'A_primary_strict'}
    lines=['# Manuscript results synthesis','',
    'Authoritative synthesis of the validated final aggregate artifacts. Natural combined-text LR is the primary text baseline; Linear SVM is replication. Title, publisher and transformer experiments are sensitivities. The independent COVID target supplies descriptive external validation. Earlier reports’ prospective next-stage instructions are historical and do not govern the current status.',
    'All numbers below are copied from final tables, with source-row/field provenance in [the numeric ledger](../tables/manuscript_numeric_provenance.csv). No models, bootstrap resampling or new statistical tests were run. Text point estimates, fold variability and seed variability are not confidence intervals.','',
    '## A. Research question','',
    'How does the difference between misinformation and real-news observed engagement/sharing differ between the COVID-focused CoAID context and the non-COVID FakeHealth HealthStory comparator? The text branch asks whether learned Fake/Real associations generalize across these contexts, with independent COVID-Rumor NEWS used to assess greater portability across COVID-focused datasets. Dataset identity is inseparable from period/context in the primary comparison.','',
    '## B. Engagement findings','',
    f"The standardized Fake/Real expected observed-post ratio is {s(e,a,'coaid_ratio','.3f')} in CoAID and {s(e,a,'fakehealth_ratio','.3f')} in FakeHealth. Their ratio-of-ratios is {s(e,a,'ratio_of_ratios','.3f')} (95% independent overlap-cluster bootstrap percentile interval {s(e,a,'CI_lower_ratio','.3f')}–{s(e,a,'CI_upper_ratio','.3f')}). Thus the relative Fake disadvantage is substantially larger in CoAID. This is an observational cross-dataset contrast. "+source_link(e),
    'The estimand is the ratio of model-standardized expected unique observed-post counts, including zero-count items. CoAID uses its validated first-release-adjusted hurdle mean and FakeHealth its negative-binomial mean. Each keeps its own reference population; the estimates do not compare a shared covariate population and are not complete diffusion cascades.',
    f"The B and C specification ratios-of-ratios are {s(e,{'coaid_specification':'B_resource_sensitivity'},'ratio_of_ratios','.3f')} and {s(e,{'coaid_specification':'C_high_certainty'},'ratio_of_ratios','.3f')}; both saved intervals remain below one. Excluding the dominant CoAID overlap component changes the ratio-of-ratios to {s('cross_context_sensitivity_results',{'sensitivity':'exclude_dominant_component'},'ratio_of_ratios','.3f')} [{s('cross_context_sensitivity_results',{'sensitivity':'exclude_dominant_component'},'CI_lower_ratio','.3f')}, {s('cross_context_sensitivity_results',{'sensitivity':'exclude_dominant_component'},'CI_upper_ratio','.3f')}]. Direction persists, but magnitude and uncertainty are sensitive to that component. This exclusion changes the target population and remains a sensitivity. "+source_link('cross_context_sensitivity_results'),
    'Saved uncertainty assumes independent samples across datasets and resamples each dataset’s observed overlap components. Incomplete dependence capture and a dominant component limit nominal coverage. No bootstrap was rerun for this synthesis.','',
    '## C. Text within-context findings','',
    'CoAID is much easier than FakeHealth under the validated within-context protocols. LR and Linear SVM reproduce this pattern. These are reserved resource/group holdouts; evaluation metadata, engagement, IDs and labels were not predictors.','']
    rows=[]
    for model,prefix in [('LR','tfidf_logistic'),('Linear SVM','tfidf_linear_svm')]:
        for ds in ['CoAID','FakeHealth']:
            w={'dataset':ds,'condition':'natural'};n=prefix+'_holdout_results'
            rows.append({'Model':model,'Context':ds,'N':s(n,w,'N','.0f'),'Macro F1':s(n,w,'macro_f1'),'ROC-AUC':s(n,w,'roc_auc')})
    lines += [md(rows,['Model','Context','N','Macro F1','ROC-AUC']),'',source_link('tfidf_logistic_holdout_results')+'; '+source_link('tfidf_linear_svm_holdout_results'),
      'The easier dataset is not evidence of intrinsically easier COVID veracity recognition: labels, sources, topic and collection differ.','',
      '## D. Cross-context transfer findings','',
      'Transfer degrades strongly when compared with target-trained models on the identical target holdout rows. Full-source transfer fits include the former source holdout; the target vocabulary and threshold remain fixed by the source protocol. All-eligible scores are supplementary and must not be substituted for these matched-holdout comparisons.','']
    rows=[]
    for model,prefix in [('LR','tfidf_logistic'),('Linear SVM','tfidf_linear_svm')]:
        for ds,target,direction in [('FakeHealth','CoAID','FH_to_CA'),('CoAID','FakeHealth','CA_to_FH')]:
            w={'source_dataset':ds,'condition':'natural','scope':'same_target_holdout'};n=prefix+'_transfer_results'
            dw={'direction':direction,'metric':'macro_f1'}
            if model=='LR':dw['condition']='natural'
            rows.append({'Model':model,'Direction':ds+' → '+target,'Within target F1':s(prefix+'_holdout_results',{'dataset':target,'condition':'natural'},'macro_f1'),
              'Transfer F1':s(n,w,'macro_f1'),'Saved F1 change':s(prefix+'_transfer_degradation',dw,'transfer_drop','+.4f'),'Transfer ROC-AUC':s(n,w,'roc_auc')})
    lines += [md(rows,list(rows[0])),'',source_link('tfidf_logistic_transfer_results')+'; '+source_link('tfidf_linear_svm_transfer_results')+'; '+source_link('tfidf_logistic_transfer_degradation')+'; '+source_link('tfidf_linear_svm_transfer_degradation'),
      'The absolute drop is asymmetric because the target-trained baselines differ greatly. Both classical models overpredict Fake in FakeHealth-to-CoAID transfer and underpredict it in CoAID-to-FakeHealth transfer. These descriptive patterns do not identify the mechanism of domain shift.','',
      '**Title-only sensitivity.** Small macro-F1 gains do not provide a consistent rescue: ROC-AUC worsens in both directions on identical title-eligible target rows.','']
    rows=[]
    n='tfidf_logistic_title_vs_combined'
    for direction in ['FH_to_CA','CA_to_FH']:
        w=dict(evaluation='transfer',direction=direction,condition='natural',scope='same_target_holdout')
        rows.append({'Direction':direction,'N':s(n,{**w,'metric':'macro_f1'},'matched_target_N','.0f'),
         'Title F1':s(n,{**w,'metric':'macro_f1'},'title'),'Combined F1, same rows':s(n,{**w,'metric':'macro_f1'},'combined_same_rows'),
         'Title ROC-AUC':s(n,{**w,'metric':'roc_auc'},'title'),'Combined ROC-AUC, same rows':s(n,{**w,'metric':'roc_auc'},'combined_same_rows')})
    lines += [md(rows,list(rows[0])),'',source_link(n),
      'Matched target rows avoid cohort mismatch in these scores, but source training cohorts and tuned configurations still differ. This is a representation-protocol sensitivity, not an isolated body-length effect.','',
      '## E. Lexical-instability evidence','']
    n='text_coefficient_alignment_summary';w={'cutoff':'all'}
    lines += [f"Across {s(n,w,'N',',.0f')} shared unigrams, signed LR coefficients have Pearson correlation {s(n,w,'pearson')} and Spearman correlation {s(n,w,'spearman')}. Signs reverse for {s(n,w,'sign_reversal_percent','.2f')}% of shared features. Among the top {s(n,{'cutoff':'top100'},'N','.0f')} by the prespecified maximum absolute coefficient, {s(n,{'cutoff':'top100'},'sign_reversal_percent','.1f')}% reverse direction. Important shared words therefore often change association direction. "+source_link(n),
      'This audit concerns the frozen within-context training-partition models, not the later full-source external-transfer fits. Positive coefficients indicate Fake association and negative coefficients Real association. Different vocabularies, IDF scales, correlated features and regularization condition these coefficients. Weak alignment is consistent with limited portability, but does not establish that coefficient reversals caused transfer degradation. No individual word is a general misinformation cue.','',
      '## F. Publisher/source sensitivity','']
    n='fakehealth_publisher_heldout_summary';w={'condition':'natural','metric':'macro_f1'}
    lines += [f"FakeHealth retains modest predictive association for unseen publisher families: outer-fold mean macro F1 {s(n,w,'outer_mean')} (sample SD {s(n,w,'outer_sample_sd')}; range {s(n,w,'outer_min')}–{s(n,w,'outer_max')}); pooled macro F1 {s(n,w,'pooled')}. Mean outer-fold ROC-AUC is {s(n,{'condition':'natural','metric':'roc_auc'},'outer_mean')}. These are descriptive fold summaries, not uncertainty intervals. "+source_link(n)]
    n='fakehealth_publisher_heldout_baseline_comparison';w={'metric':'macro_f1','baseline_scope':'resource_grouped_cv','publisher_aggregation':'outer_mean'}
    lines += [f"The saved comparison gives publisher outer-mean macro F1 {s(n,w,'publisher_heldout')} versus selected resource-grouped CV {s(n,w,'resource_grouped')}, a difference of {s(n,w,'publisher_minus_resource','+.4f')}. This is unpaired: grouping, training/tuning populations and evaluation designs differ. "+source_link(n)]
    n='fakehealth_publisher_heldout_results'
    lines += [f"Source masking leaves pooled macro F1 essentially unchanged ({s(n,{'condition':'source_masked','scope':'pooled'},'macro_f1')} masked versus {s(n,{'condition':'natural','scope':'pooled'},'macro_f1')} natural). "+source_link(n),
      'Known source strings alone do not account for all surviving FakeHealth signal. Masking is a limited, label-blind transductive sensitivity; it does not remove style/topic or all publisher information. Publisher families are observed proxies, and this experiment does not establish unseen-publisher generalization for CoAID.','',
      '## G. Transformer robustness','',
      'The primary DistilRoBERTa experiment uses the original 512-token, seed-42 protocol. It reproduces the easier within-context CoAID pattern but offers no consistent transfer improvement.','']
    rows=[]
    for ds,target in [('FakeHealth','CoAID'),('CoAID','FakeHealth')]:
        w={'source_dataset':ds,'max_length':512,'scope':'same_target_holdout'};n='distilroberta_transfer_results'
        rows.append({'Direction':ds+' → '+target,'Transformer F1':s(n,w,'macro_f1'),'Transformer ROC-AUC':s(n,w,'roc_auc'),
                     'Transformer AP':s(n,w,'average_precision'),'Within-target transformer F1':s('distilroberta_holdout_results',{'target_dataset':target,'max_length':512},'macro_f1')})
    lines += [md(rows,list(rows[0])),'',source_link('distilroberta_transfer_results')+'; '+source_link('distilroberta_holdout_results'),
      'The modest CoAID-to-FakeHealth macro-F1 gain over both classical comparators coexists with lower ROC-AUC and AP. FakeHealth-to-CoAID macro F1 is also lower than both classical comparators. A smaller transfer-minus-within drop can partly reflect a weaker within-target baseline and should not be read as stronger generalization by itself.',
      '**Token budget and seeds.** The paired FakeHealth-to-CoAID follow-up qualifies the original shorter-cap result.','']
    rows=[];n='distilroberta_budget_seed_summary'
    for metric in ['macro_f1','roc_auc','average_precision']:
        w={'kind':'paired_delta','scope':'same_target_holdout','metric':metric}
        rows.append({'Metric':metric,'Mean 256−512':s(n,w,'mean','+.4f'),'Sample SD':s(n,w,'sample_sd'),
          'Range':s(n,w,'minimum','+.4f')+' to '+s(n,w,'maximum','+.4f'),'Positive seeds / tested':s(n,w,'positive_seeds','.0f')+' / '+s(n,w,'N_seeds','.0f')})
    lines += [md(rows,list(rows[0])),'',source_link(n)+'; '+source_link('distilroberta_budget_seed_paired_deltas'),
      'Macro F1 improves at the shorter cap across the tested seeds, while ROC-AUC and AP improve only for the original seed and decline for the other seeds. Lower predicted Fake prevalence can improve threshold behavior without improving ranking. The evidence does not support a stable claim that reducing source representation improves transfer generally. The follow-up covers one transfer direction and is not population uncertainty estimation.',
      f"At the primary cap, {s('transformer_truncation_audit',{'dataset':'FakeHealth','max_length':512},'pair_percent_truncated','.2f')}% of FakeHealth texts truncate versus {s('transformer_truncation_audit',{'dataset':'CoAID','max_length':512},'pair_percent_truncated','.0f')}% of CoAID; at the shorter cap, FakeHealth truncation is {s('transformer_truncation_audit',{'dataset':'FakeHealth','max_length':256},'pair_percent_truncated','.2f')}%. CoAID target token content is unchanged across the paired caps, but source budget and optimization vary together. "+source_link('transformer_truncation_audit'),'',
      '## H. Independent COVID-context external validation','']
    n='covid_rumor_external_cohort_summary'
    def cohort(measure):return s(n,{'measure':measure},'value',',.0f')
    lines += [f"The clean COVID-Rumor NEWS target contains {cohort('final_clean_binary_N')} verified items: {cohort('clean_Fake_N')} Fake and {cohort('clean_Real_N')} Real. Exact source-overlap rules remove {cohort('exact_source_overlap_excluded_N')} otherwise eligible items. Only one statement per NEWS item is scored; Unverified entries and the Twitter subset are excluded. The models and source vectorizers are reused unchanged, with target/prediction freezes preceding outcome-label evaluation. "+source_link(n),'']
    rows=[];n='covid_rumor_external_transfer_results'
    for classifier in ['LR','SVM']:
        for ds in ['CoAID','FakeHealth']:
            w={'classifier':classifier,'source_dataset':ds,'scope':'clean_primary'}
            rows.append({'Source/model':ds+' '+classifier,'Macro F1':s(n,w,'macro_f1'),'Balanced accuracy':s(n,w,'balanced_accuracy'),
              'ROC-AUC':s(n,w,'roc_auc'),'Predicted Fake %':format(100*cell(n,w,'predicted_fake_prevalence'),'.2f')})
    lines += [md(rows,list(rows[0])),'',source_link(n)]
    w={'classifier':'LR','source_dataset':'CoAID','scope':'clean_primary'}
    lines += [f"Actual Fake prevalence is {100*cell(n,w,'actual_fake_prevalence'):.2f}%. Both classifiers favor CoAID-source transfer across the reported comparison metrics, with the largest separation evident in ranking. Threshold performance remains modest, and every model overpredicts Fake."]
    n='covid_rumor_external_source_comparison'
    lines += [f"CoAID-minus-FakeHealth macro-F1 differences are {s(n,{'classifier':'LR','scope':'clean_primary','metric':'macro_f1'},'coaid_minus_fakehealth','+.4f')} for LR and {s(n,{'classifier':'SVM','scope':'clean_primary','metric':'macro_f1'},'coaid_minus_fakehealth','+.4f')} for SVM; ROC-AUC differences are {s(n,{'classifier':'LR','scope':'clean_primary','metric':'roc_auc'},'coaid_minus_fakehealth','+.4f')} and {s(n,{'classifier':'SVM','scope':'clean_primary','metric':'roc_auc'},'coaid_minus_fakehealth','+.4f')}, respectively. These are descriptive differences, not statistical winner declarations. "+source_link(n)]
    n='covid_rumor_external_vocabulary_diagnostics'
    lines += [f"Token-occurrence coverage is {s(n,{'classifier':'LR','source_dataset':'CoAID','scope':'clean_primary'},'target_token_occurrence_coverage_percent','.2f')}% for CoAID and {s(n,{'classifier':'LR','source_dataset':'FakeHealth','scope':'clean_primary'},'target_token_occurrence_coverage_percent','.2f')}% for FakeHealth, identical across LR/SVM within each source. Every model has zero zero-feature rows. Higher coverage may contribute to the advantage; nonempty vectors alone do not establish semantic compatibility. "+source_link(n),
      'All-eligible and approximate-overlap-excluded sensitivities retain the qualitative source ordering. This result is consistent with greater portability of linguistic Fake/Real associations across independent COVID-focused information environments. It does not identify a causal sensitive-period effect. Short-statement genre, label construction, source composition, high class imbalance and incomplete overlap/language screening remain limitations.','',
      '## I. Integrated interpretation','',
      'Observed Fake/Real engagement contrasts differ substantially across the two main dataset contexts, while learned linguistic associations also show limited portability across those contexts. Independent COVID external validation provides evidence that some textual associations are more portable within COVID-focused datasets than from the non-COVID health comparator. Together, the findings support a context-dependent interpretation of observed misinformation engagement/sharing and linguistic signals.',
      'The engagement and text branches provide complementary evidence; no analysis links coefficient changes causally to post counts or tests mediation. Dataset identity, period/context, source composition, label construction, genre and collection procedure remain confounded.',
      '**Consistency assessment.** No unresolved contradiction appears among the validated headline findings. Strong within-context classification and weak cross-context transfer answer different questions. Better within-COVID transfer is compatible with weak CoAID–FakeHealth transfer. Apparent disagreements concern metric choice: title-only and transformer sensitivities can improve macro F1 while worsening ranking. Dominant-component exclusion changes engagement magnitude without reversing direction. The latest seed follow-up qualifies the original single-seed budget interpretation; it does not replace the original primary protocol.',
      '**Readiness to draft.** No additional analysis is required for the bounded observational argument. A causal period claim, a broad source-independent mechanism or a deployment claim would require a different design. Preserve those boundaries rather than adding models without a manuscript gap.','',
      '## J. Limitations','',
      '[Full limitations table](../tables/manuscript_limitations.csv). Codes link directly to the master results table.','']
    lines += [md([{'Code':r.limitation_code,'Limitation':r.limitation,'Residual concern':r.residual_concern} for r in limitations.itertuples()],['Code','Limitation','Residual concern']),'',
      '## K. Claims that ARE supported','']
    lines += [f'- **{r.claim_id}:** {r.recommended_wording}' for r in claims.itertuples()]
    lines += ['', '## L. Claims that are NOT supported','',
      '- COVID caused the observed engagement contrast or the lexical differences.',
      '- The design identifies a causal sensitive-period effect, a universal vocabulary or causal signaling words.',
      '- Observed post counts represent complete cascades or all exposure and diffusion.',
      '- Lexical coefficient reversals explain or mediate the engagement contrast.',
      '- Publisher confounding is eliminated, or FakeHealth publisher results validate CoAID on unseen publishers.',
      '- Title-only models never improve any score, or transformers consistently improve transfer.',
      '- Shorter transformer inputs improve ranking across seeds, or seed/fold SDs are confidence intervals.',
      '- The independent COVID result establishes a statistical winner or deployment-ready detection.','',
      '## Validation note','',
      'Final machine-readable tables take precedence over rounded prose. Text artifacts were checked against saved validation fingerprints, and source/output integrity was rechecked after synthesis. Engagement completion metadata and saved validation flags were checked, and the final contrast tables were reconciled with their report and algebraic identities. That engagement stage did not save final-output hashes; the new synthesis fingerprint records current integrity without claiming a historical checksum. No prohibited raw data, notebooks, model checkpoints, prediction files or training logs were opened.','']
    return '\n'.join(lines)


def figure_plan():
    return '''# Main-paper figure and table plan

Five main-paper items are proposed. No new figures were generated. Existing figures are candidates for reuse; confirm labels and accessibility during manuscript layout. Do not recompute estimates or add uncertainty bars absent from validated artifacts.

## 1. Figure: standardized observed-post ratios and cross-context contrast

Use `outputs/figures/cross_context_engagement/within_and_cross_context_forest.png` as the existing candidate. Emphasize the primary A comparison; label B/C and dominant-component exclusion as sensitivities if included. Source: `cross_context_engagement_contrast.csv`, `cross_context_sensitivity_results.csv`, and the existing figure's validated within-context endpoints.

Caption: model-standardized expected unique observed-post ratios, including zeros; separate dataset reference populations; saved independent overlap-cluster percentile intervals. Include the null ratio line and dominant-component caveat. Do not label the estimand complete diffusion or a causal COVID effect. Existing display selection, not a new analysis.

## 2. Figure: LR/SVM within-context versus cross-context transfer

Plan matched panels by target context, with model-specific within-target and transfer macro F1; add ROC-AUC as a second panel only if legible. Use natural combined text and identical reserved target holdouts. Sources: `tfidf_logistic_holdout_results.csv`, `tfidf_logistic_transfer_results.csv`, `tfidf_linear_svm_holdout_results.csv`, `tfidf_linear_svm_transfer_results.csv`.

Existing candidates: `outputs/figures/tfidf_linear_svm/within_macro_f1.png` and `transfer_macro_f1.png`. A combined publication layout is planned, not created. Do not compare all-eligible transfer scores to holdout baselines or add significance markers.

## 3. Figure: lexical coefficient alignment and sign reversal

Reuse candidates `outputs/figures/text_feature_alignment/shared_unigram_coefficient_scatter.png` and `sign_agreement_by_importance.png` as panels in one figure. Aggregate source: `text_coefficient_alignment_summary.csv`.

Caption: shared unigrams in natural within-context LR models; positive is Fake association; ranks/cutoffs prespecified; different conditional feature sets, IDF scales and regularization. Individual labels illustrate conditional associations and must not be presented as causal or general cues. No new coefficient inspection is required.

## 4. Figure: independent COVID-Rumor NEWS transfer

Plan LR/SVM source comparisons on the identical clean NEWS target, with macro F1 and ROC-AUC panels. Include target prevalence, predicted Fake prevalence and a compact coverage/zero-feature annotation. Sources: `covid_rumor_external_transfer_results.csv`, `covid_rumor_external_source_comparison.csv`, `covid_rumor_external_vocabulary_diagnostics.csv`, `covid_rumor_external_cohort_summary.csv`.

No existing final external-transfer figure was produced, so creation is deferred to manuscript layout. Caption: descriptive external validation, exact-overlap exclusions, natural target prevalence, fixed source thresholds, short-statement/label/genre caveats. No confidence bars or winner labels.

## 5. Table: compact robustness summary

Use selected rows of `manuscript_key_results.csv`: title-only matched-target changes; publisher-held-out outer mean and fold SD; primary transformer transfer; paired budget/seed variability; engagement specification/dominant-component sensitivity. Show purpose, evaluation scope, key result and residual limitation. Fold/seed SD and range are variability, not confidence intervals. Report threshold-versus-ranking disagreements explicitly; the seed follow-up governs the budget interpretation.

The full key-results, claim-evidence and limitations tables remain drafting/supplement resources, not additional main-paper displays. Put exhaustive class metrics and protocol detail in the supplement if needed.
'''


def validate_content():
    e=load('cross_context_engagement_contrast')
    assert ((e.coaid_ratio/e.fakehealth_ratio-e.ratio_of_ratios).abs()<1e-12).all()
    assert e.CI_upper_ratio.lt(1).all()
    text_path=remember(REPORTS/'cross_context_engagement_contrast.md')
    old=text_path.read_text()
    assert all(x in old for x in ['0.253','0.840','0.301','0.154','0.645'])
    for classifier,prefix in [('LR','tfidf_logistic'),('SVM','tfidf_linear_svm')]:
        for ds,target in [('CoAID','FakeHealth'),('FakeHealth','CoAID')]:
            a=select(prefix+'_transfer_results',source_dataset=ds,condition='natural',scope='same_target_holdout')
            b=select(prefix+'_holdout_results',dataset=target,condition='natural')
            assert a.evaluation_uid_sha256==b.evaluation_uid_sha256
            assert a.macro_f1<b.macro_f1
    for classifier in ['LR','SVM']:
        for scope in ['clean_primary','all_eligible','clean_without_high_similarity']:
            a=select('covid_rumor_external_transfer_results',classifier=classifier,source_dataset='CoAID',scope=scope)
            b=select('covid_rumor_external_transfer_results',classifier=classifier,source_dataset='FakeHealth',scope=scope)
            assert a.evaluation_uid_sha256==b.evaluation_uid_sha256
            for field in ['macro_f1','roc_auc','balanced_accuracy','fake_f1','average_precision']:assert a[field]>b[field]
        for ds in ['CoAID','FakeHealth']:
            d=select('covid_rumor_external_vocabulary_diagnostics',classifier=classifier,source_dataset=ds,scope='clean_primary')
            assert d.zero_feature_rows==0
    load('distilroberta_budget_seed_paired_deltas')
    n='distilroberta_budget_seed_summary'
    for metric,positive in [('macro_f1',3),('roc_auc',1),('average_precision',1)]:
        assert select(n,kind='paired_delta',scope='same_target_holdout',metric=metric).positive_seeds==positive
    for ds in ['CoAID','FakeHealth']:
        w=dict(evaluation='transfer',direction='CA_to_FH' if ds=='CoAID' else 'FH_to_CA',condition='natural',scope='same_target_holdout')
        assert select('tfidf_logistic_title_vs_combined',**w,metric='macro_f1').title_minus_combined>0
        assert select('tfidf_logistic_title_vs_combined',**w,metric='roc_auc').title_minus_combined<0
    for r in RESULTS:
        assert isinstance(r['estimate'],(int,float)) and math.isfinite(r['estimate'])
    assert len(set(r['result_id'] for r in RESULTS))==len(RESULTS)


def main():
    if VALIDATION.exists():
        meta=json.loads(VALIDATION.read_text())
        assert sha(__file__)==meta['script_sha256']
        for name,h in {**meta['sources'],**meta['outputs']}.items():assert sha(ROOT/name)==h,name
        assert sha(AGENTS)==meta['agents_after_sha256']
        print('Synthesis fingerprints verified; no artifacts modified.')
        return
    outputs=[KEY,CLAIMS,LIMITATIONS,REPORT,PLAN,LEDGER]
    assert not any(p.exists() for p in outputs),'Refusing existing synthesis outputs'
    metadata()
    results=build_results();limitations=build_limitations();claims=build_claims()
    report=manuscript(limitations,claims);plan=figure_plan()
    validate_content()
    assert all('## '+c+'. ' in report for c in 'ABCDEFGHIJKL')
    assert plan.count('\n## ')==5
    assert len(claims.claim_id.unique())==len(claims)
    assert all(set(r.split(';'))<=set(limitations.limitation_code) for r in results.limitation_code)
    ledger=pd.DataFrame(ACCESSES).drop_duplicates()
    # Reconcile every accessed value against its exact source row, without inference.
    for r in ledger.itertuples():
        actual=select(Path(r.source_artifact).stem,**json.loads(r.row_selector))[r.field]
        assert actual==r.value or math.isclose(float(actual),float(r.value),rel_tol=1e-14)
    before_agents=AGENTS.read_bytes()
    for path,frame in [(KEY,results),(CLAIMS,claims),(LIMITATIONS,limitations),(LEDGER,ledger)]:frame.to_csv(path,index=False)
    REPORT.write_text(report);PLAN.write_text(plan)
    # Append only; close the pre-existing unmatched Markdown fence if necessary.
    prior=before_agents.decode()
    closing='\n```\n' if sum(line.lstrip().startswith('```') for line in prior.splitlines())%2 else ''
    addition=closing+'''\n\n## CURRENT STATUS — 2026-09-28

- Engagement stage is frozen.
- Classical text modeling and robustness stages are frozen.
- Transformer primary and budget/seed sensitivity stages are frozen.
- Independent COVID-Rumor NEWS external validation is complete.
- Manuscript results synthesis is complete; next phase is manuscript writing.
- No new modeling unless motivated by a specific manuscript gap.
- Current synthesis: `outputs/reports/manuscript_results_synthesis.md`; key results, claims and limitations are in `outputs/tables/manuscript_*.csv`.
- This section supersedes earlier next-stage statements; validated findings and interpretation limits remain in force.
'''
    with AGENTS.open('ab') as f:f.write(addition.encode())
    assert AGENTS.read_bytes().startswith(before_agents)
    assert all(sha(ROOT/name)==value for name,value in SNAPSHOT.items()),'Validated source changed'
    # Re-read written tables to verify shape and lossless numeric round trips.
    saved=pd.read_csv(KEY)
    assert len(saved)==len(results) and saved.result_id.is_unique
    assert all(math.isclose(x,y,rel_tol=1e-14,abs_tol=1e-14) for x,y in zip(saved.estimate,results.estimate))
    validation=dict(validation='passed',key_result_rows=len(results),claim_evidence_rows=len(claims),limitation_rows=len(limitations),
      numeric_provenance_rows=len(ledger),sources=SNAPSHOT,outputs={rel(p):sha(p) for p in outputs},script_sha256=sha(__file__),
      agents_before_sha256=hashlib.sha256(before_agents).hexdigest(),agents_after_sha256=sha(AGENTS),agents_append_only=True,
      new_models=0,new_bootstraps=0,new_statistical_tests=0,
      checks=['validated text-table fingerprints','saved engagement completion/validation flags and report/table identities',
        'all numeric source accesses reconciled by row selector and field','matched evaluation UIDs retained',
        'natural combined-text and 512-token primary scopes explicit','seed/fold variability not labeled confidence intervals',
        'latest paired-seed interpretation used','external comparisons descriptive','source/label/genre limits retained',
        'causal claims explicitly excluded','no full-cascade interpretation','exactly five main-paper display proposals',
        'prior source artifacts unchanged','AGENTS original bytes retained','unique keys and numeric round trips'],
      historical_hash_limitation='Engagement final outputs have no historical output hashes; completion flags, saved input-validation table, report agreement and internal identities checked; current hashes recorded.',
      restricted_access='Aggregate final tables, final reports and validation metadata only; no raw data, notebooks, checkpoints, predictions or training logs.')
    VALIDATION.write_text(json.dumps(validation,indent=2)+'\n')
    print(f'Synthesis complete: {len(results)} key results; {len(claims)} claims; {len(limitations)} limitations. Validation passed.')


if __name__=='__main__':main()
