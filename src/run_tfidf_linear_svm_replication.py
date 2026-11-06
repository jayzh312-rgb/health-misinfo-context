"""Natural combined-text TF-IDF -> LinearSVC robustness replication.

Run: .venv/bin/python src/run_tfidf_linear_svm_replication.py
Only new SVM CV/final fits run. Frozen LR results are comparators, never fitted.
Existing cohorts and grouped splits remain immutable. Checkpoints support reuse.
"""
from pathlib import Path
import hashlib
import itertools
import json
import logging
import os
import platform
import tempfile
import time
import warnings
from unittest.mock import patch

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'linear_svm_replication_matplotlib'))
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import run_tfidf_logistic_baseline as base

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT/'outputs/models/tfidf_linear_svm'
TABLES = ROOT/'outputs/tables'
REPORT = ROOT/'outputs/reports/tfidf_linear_svm_replication.md'
PRED_PATH = ROOT/'outputs/predictions/tfidf_linear_svm_predictions.csv'
TRANSFER_META = ROOT/'outputs/reports/tfidf_logistic_cross_context_transfer.json'
PREFIXES = dict(base.PREFIXES)
DIRECTIONS = {'FH_to_CA': ('FakeHealth','CoAID'), 'CA_to_FH': ('CoAID','FakeHealth')}
FEATURES = ['dataset','content_uid','combined_text','primary_modeling_eligible','leakage_group',
            'cv_fold','holdout_assignment','cross_dataset_overlap_flag','cross_dataset_overlap_group_flag']
LABELS = ['binary_analysis_target','binary_analysis_label']
COMPARE = ['macro_f1','balanced_accuracy','fake_f1','roc_auc','average_precision']
GRID = [dict(config_id=f'cfg_{i:02d}',ngram_max=n,min_df=m,C=c,class_weight=w)
        for i,(n,m,c,w) in enumerate(itertools.product([1,2],[2,5],[.01,.1,1.,10.],['none','balanced']))]
SVC_OPTIONS = dict(penalty='l2',loss='squared_hinge',dual='auto',tol=1e-4,max_iter=20000,
                   random_state=42,fit_intercept=True,intercept_scaling=1)
PROTOCOL = dict(seed=42,grid=GRID,svm=SVC_OPTIONS,predictor='combined_text',eligibility='primary_modeling_eligible',
    threshold=0.,tie_tolerance=.001,tie_order=['macro F1 within .001 of best','SD within .001 of lowest',
        'unigram','smaller C','class_weight none','config_id'],
    tfidf=dict(sublinear_tf=True,max_df=1.,lowercase=True,norm='l2',smooth_idf=True),
    transfer_source='all eligible source rows including former source holdout',
    calibration=False,target_labels='joined for scoring after all four prediction bundles frozen')
require,sha,digest,write_json = base.require,base.sha,base.digest,base.write_json


def signature():
    bm=json.loads(base.META_PATH.read_text());tm=json.loads(TRANSFER_META.read_text())
    require(bm['validation']==tm['validation']=='passed','LR comparators not validated')
    paths=[ROOT/f'data/processed/{p}_text_modeling_ready.csv' for p in PREFIXES.values()]
    paths += [ROOT/f'outputs/splits/{p}_text_{s}.csv' for p in PREFIXES.values() for s in ['grouped_cv','holdout']]
    paths += [TABLES/f'tfidf_logistic_{n}.csv' for n in ['holdout_results','transfer_results','transfer_degradation']]
    expected={**bm['signature']['inputs'],**bm['outputs'],**tm['outputs']}
    for path in paths:
        require(expected.get(str(path.relative_to(ROOT)))==sha(path),'Validated input changed: '+path.name)
    require(sha(base.__file__)==bm['signature']['script_sha256'],'Baseline utility code changed')
    paths += [base.META_PATH,TRANSFER_META,Path(base.__file__)]
    versions=dict(python=platform.python_version(),sklearn=sklearn.__version__,numpy=np.__version__,
                  scipy=scipy.__version__,pandas=pd.__version__)
    require(versions==bm['signature']['versions'],'Environment differs from validated baseline')
    return dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in paths},versions=versions,
                script_sha256=sha(__file__),protocol=PROTOCOL)


def load_features(dataset):
    prefix=PREFIXES[dataset]
    d=pd.read_csv(ROOT/f'data/processed/{prefix}_text_modeling_ready.csv',usecols=FEATURES,keep_default_na=False)
    require(d.primary_modeling_eligible.dtype==bool,'Eligibility must be boolean')
    d=d[d.primary_modeling_eligible].sort_values('content_uid').reset_index(drop=True)
    require(d.dataset.eq(dataset).all() and d.content_uid.is_unique,'Cohort identity error')
    require(d.combined_text.str.strip().ne('').all(),'Empty natural predictor')
    require(not d.cross_dataset_overlap_flag.any() and not d.cross_dataset_overlap_group_flag.any(),'Cross-dataset overlap')
    require(d.cv_fold.isin([-1,0,1,2,3,4]).all(),'Unexpected CV fold')
    require((d.cv_fold.eq(-1)==d.holdout_assignment.eq('test')).all(),'Holdout mismatch')
    require(d.groupby('leakage_group').cv_fold.nunique().eq(1).all(),'Group crosses partitions')
    for suffix,oldcol,col in [('grouped_cv','fold','cv_fold'),('holdout','assignment','holdout_assignment')]:
        old=pd.read_csv(ROOT/f'outputs/splits/{prefix}_text_{suffix}.csv',
            usecols=['content_uid','leakage_group',oldcol]).set_index('content_uid')
        require(old.index.is_unique and set(d.content_uid)<=set(old.index),'Frozen split UID mismatch')
        aligned=old.loc[d.content_uid]
        require(aligned[oldcol].tolist()==d[col].tolist(),'Split reassignment')
        require(aligned.leakage_group.tolist()==d.leakage_group.tolist(),'Leakage-group reassignment')
    require(not set(LABELS)&set(d.columns),'Labels in feature view')
    return d


def read_labels(dataset,uids):
    d=pd.read_csv(ROOT/f'data/processed/{PREFIXES[dataset]}_text_modeling_ready.csv',
                  usecols=['content_uid',*LABELS],keep_default_na=False).set_index('content_uid')
    require(d.index.is_unique,'Duplicate label UID')
    d=d.loc[list(uids)]
    require(d.binary_analysis_target.eq(d.binary_analysis_label.eq('Fake').astype(int)).all() and
            set(d.binary_analysis_label)=={'Fake','Real'},'Fake=1 coding or class support failed')
    return d.binary_analysis_target.to_numpy(dtype=int)


def source_view(dataset,features):
    require(features.dataset.eq(dataset).all(),'Mixed source datasets')
    source=features.copy();source['binary_analysis_target']=read_labels(dataset,source.content_uid)
    return source


def texts(frame):
    """Only natural combined_text, never source/engagement/split metadata."""
    return frame['combined_text'].to_numpy(dtype=str,copy=True)


def make_pipeline(config):
    return Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,int(config['ngram_max'])),min_df=int(config['min_df']),
        sublinear_tf=True,max_df=1.,lowercase=True,dtype=np.float64)),
        ('svm',LinearSVC(C=float(config['C']),class_weight=None if config['class_weight']=='none' else 'balanced',**SVC_OPTIONS))])


def metrics(y,scores):
    # Pass explicit zero-threshold predictions; base.metrics otherwise defaults
    # to probability threshold .5. AUC/AP accept arbitrary real decision scores.
    return base.metrics(y,scores,prediction=(np.asarray(scores)>=0).astype(int))


def fit_source(source,config):
    require(source.dataset.nunique()==1 and source.primary_modeling_eligible.all(),'Invalid source fit population')
    require(set(source.binary_analysis_target)=={0,1},'Training lacks a class')
    model=make_pipeline(config);started=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=1):
        warnings.simplefilter('always')
        model.fit(texts(source),source.binary_analysis_target.to_numpy())
    iterations=int(np.max(model['svm'].n_iter_))
    converged=iterations<SVC_OPTIONS['max_iter'] and not any(issubclass(w.category,ConvergenceWarning) for w in caught)
    require(np.array_equal(model['svm'].classes_,[0,1]),'Decision-score class orientation failed')
    fit=dict(converged=converged,n_iter=iterations,fit_seconds=time.perf_counter()-started,
        warnings='; '.join(str(w.message) for w in caught),train_N=len(source),source_dataset=source.dataset.iloc[0],
        train_uid_sha256=digest(source.content_uid),train_group_sha256=digest(source.leakage_group.unique()),
        vocabulary_size=len(model['tfidf'].vocabulary_),vocabulary_sha256=digest(model['tfidf'].vocabulary_),
        predictor_field='combined_text')
    if not converged:logging.warning('SVM fit did not converge: %s',fit)
    return model,fit


def predict_unlabeled(model,target):
    require(not {'binary_analysis_target','binary_analysis_label','true_binary_label'}&set(target.columns),'Target labels in prediction interface')
    vectorizer=model['tfidf'];vocabulary=dict(vectorizer.vocabulary_);idf=vectorizer.idf_.copy()
    matrix=vectorizer.transform(texts(target))
    scores=model['svm'].decision_function(matrix)
    require(np.isfinite(scores).all(),'Nonfinite decision scores')
    require(vocabulary==vectorizer.vocabulary_ and np.array_equal(idf,vectorizer.idf_),'Target modified source TF-IDF')
    return scores


def check_partition(train,valid):
    require(set(train.content_uid).isdisjoint(valid.content_uid),'UID leakage')
    require(set(train.leakage_group).isdisjoint(valid.leakage_group),'Group leakage')
    require(set(train.binary_analysis_target)==set(valid.binary_analysis_target)=={0,1},'Partition lacks a class')


def preflight(views):
    rows=[]
    for ds,d in views.items():
        labeled=source_view(ds,d)
        train,test=labeled[labeled.cv_fold.ge(0)],labeled[labeled.cv_fold.eq(-1)]
        check_partition(train,test)
        parts=[('all',labeled),('training_pool',train),('holdout',test)]
        for f in range(5):
            tr,va=train[train.cv_fold.ne(f)],train[train.cv_fold.eq(f)]
            check_partition(tr,va);check_partition(tr,test)
            parts += [(f'cv{f}_train',tr),(f'cv{f}_validation',va)]
        for name,p in parts:
            rows.append(dict(dataset=ds,partition=name,N=len(p),fake_N=int(p.binary_analysis_target.sum()),
                real_N=int(p.binary_analysis_target.eq(0).sum()),leakage_groups=p.leakage_group.nunique(),uid_sha256=digest(p.content_uid)))
    a,b=views.values()
    require(set(a.content_uid).isdisjoint(b.content_uid) and set(a.leakage_group).isdisjoint(b.leakage_group),'Cross-dataset overlap')
    return pd.DataFrame(rows)


def run_cv(dataset,features):
    train=source_view(dataset,features[features.cv_fold.ge(0)])
    rows=[]
    for config in GRID:
        for fold in range(5):
            tr,va=train[train.cv_fold.ne(fold)],train[train.cv_fold.eq(fold)]
            check_partition(tr,va)
            path=STATE/f'cv_{PREFIXES[dataset]}_{config["config_id"]}_{fold}.json'
            if path.exists():
                row=json.loads(path.read_text())
                require(row['train_uid_sha256']==digest(tr.content_uid) and row['validation_uid_sha256']==digest(va.content_uid),'Stale CV cache')
            else:
                model,fit=fit_source(tr,config)
                scores=predict_unlabeled(model,va.drop(columns='binary_analysis_target'))
                row=dict(dataset=dataset,condition='natural',**config,fold=fold,**fit,**metrics(va.binary_analysis_target,scores),
                    validation_uid_sha256=digest(va.content_uid),holdout_used=False,threshold=0.)
                write_json(path,row)
            rows.append(row)
        logging.info('%s %s: five SVM CV fits complete',dataset,config['config_id'])
    return rows


def summarize_cv(cv):
    rows=[]
    for (dataset,config_id),part in cv.groupby(['dataset','config_id'],sort=True):
        require(len(part)==5 and set(part.fold)==set(range(5)),'Incomplete CV')
        config=next(c for c in GRID if c['config_id']==config_id)
        row=dict(dataset=dataset,condition='natural',**config,all_converged=bool(part.converged.all()))
        for metric in base.METRICS:
            row[metric+'_mean']=float(part[metric].mean());row[metric+'_sd']=float(part[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def select_config(summary):
    candidates=summary[summary.all_converged].copy()
    require(len(candidates)>0,'All SVM configurations failed convergence; stop and document rather than silently expand grid')
    candidates=candidates[candidates.macro_f1_mean>=candidates.macro_f1_mean.max()-.001]
    candidates=candidates[candidates.macro_f1_sd<=candidates.macro_f1_sd.min()+.001]
    candidates['weight_priority']=candidates.class_weight.ne('none').astype(int)
    chosen=candidates.sort_values(['ngram_max','C','weight_priority','config_id']).iloc[0]
    return next(dict(c) for c in GRID if c['config_id']==chosen.config_id)


def lock_configs(selection):
    configs={ds:select_config(selection[selection.dataset.eq(ds)]) for ds in PREFIXES}
    payload=dict(configs=configs,protocol=PROTOCOL,cv_summary_sha256=hashlib.sha256(selection.to_json(orient='records').encode()).hexdigest(),
        scope='source training-pool grouped CV only; both configurations frozen before all final predictions')
    path=STATE/'selection_lock.json'
    if path.exists():require(json.loads(path.read_text())==payload,'Configuration lock changed')
    else:write_json(path,payload)
    return configs


def freeze_predictions(views,configs):
    selection_path=STATE/'selection_lock.json'
    require(selection_path.exists(),'Configurations must freeze before final fitting')
    require(json.loads(selection_path.read_text())['configs']==configs and set(configs)==set(PREFIXES),'Incomplete configuration lock')
    bundles={}
    for evaluation in ['within','transfer']:
        for direction,(source_ds,target_ds) in DIRECTIONS.items():
            if evaluation=='within':target_ds=source_ds
            source,target=views[source_ds],views[target_ds]
            if evaluation=='within':source=source[source.cv_fold.ge(0)];target=target[target.cv_fold.eq(-1)]
            require(set(source.content_uid).isdisjoint(target.content_uid) and set(source.leakage_group).isdisjoint(target.leakage_group),'Final fit/evaluation leakage')
            key=f'{evaluation}_{PREFIXES[source_ds]}';path=STATE/f'{key}.joblib'
            if path.exists():
                bundle=joblib.load(path)
                require(bundle['selection_lock_sha256']==sha(selection_path),'Stale final checkpoint')
                require(bundle['fit']['train_uid_sha256']==digest(source.content_uid) and bundle['target_uid_sha256']==digest(target.content_uid),'Checkpoint cohort differs')
            else:
                fitted,fit=fit_source(source_view(source_ds,source),configs[source_ds])
                require(fit['converged'],'Final SVM did not converge; stopping without methodological changes')
                scores=predict_unlabeled(fitted,target)
                p=target[['content_uid','leakage_group','cv_fold','holdout_assignment']].copy()
                p['evaluation'],p['source_dataset'],p['target_dataset']=evaluation,source_ds,target_ds
                p['direction']=direction if evaluation=='transfer' else 'within_'+PREFIXES[source_ds]
                p['condition']='natural';p['decision_score']=scores;p['predicted_label']=(scores>=0).astype(int)
                bundle=dict(model=fitted,fit=fit,predictions=p,selected_config=configs[source_ds],threshold=0.,
                    target_uid_sha256=digest(target.content_uid),selection_lock_sha256=sha(selection_path),
                    target_text_fitted=False,target_labels_in_prediction_interface=False)
                base.dump(path,bundle)
                logging.info('%s final predictions frozen',key)
            bundles[key]=bundle
    freeze=dict(selection_lock_sha256=sha(selection_path),threshold=0.,evaluation_labels_joined=False,
                bundles={str((STATE/f'{k}.joblib').relative_to(ROOT)):sha(STATE/f'{k}.joblib') for k in bundles})
    path=STATE/'prediction_lock.json'
    if path.exists():require(json.loads(path.read_text())==freeze,'Prediction lock changed')
    else:write_json(path,freeze)
    return bundles


def evaluate(bundles):
    lock_path=STATE/'prediction_lock.json';require(lock_path.exists(),'Missing prediction freeze before evaluation')
    lock=json.loads(lock_path.read_text())
    require(len(lock['bundles'])==4 and all(sha(ROOT/p)==h for p,h in lock['bundles'].items()),'Prediction freeze incomplete/changed')
    predictions=[]
    for b in bundles.values():
        p=b['predictions'].copy();require('true_binary_label' not in p,'Premature label join')
        p['true_binary_label']=read_labels(p.target_dataset.iloc[0],p.content_uid)
        predictions.append(p)
    predictions=pd.concat(predictions,ignore_index=True)
    within,external=[],[]
    for (evaluation,direction),all_rows in predictions.groupby(['evaluation','direction']):
        scopes=[('same_target_holdout',all_rows)] if evaluation=='within' else [
            ('all_eligible',all_rows),('same_target_holdout',all_rows[all_rows.cv_fold.eq(-1)])]
        for scope,p in scopes:
            row=dict(condition='natural',source_dataset=p.source_dataset.iloc[0],target_dataset=p.target_dataset.iloc[0],
                direction=direction,scope=scope,threshold=0.,**metrics(p.true_binary_label,p.decision_score),
                actual_fake_prevalence=float(p.true_binary_label.mean()),predicted_fake_prevalence=float(p.predicted_label.mean()),
                near_single_class_prediction=bool(p.predicted_label.mean()<=.05 or p.predicted_label.mean()>=.95),
                evaluation_uid_sha256=digest(p.content_uid))
            if evaluation=='within':row['dataset']=row['target_dataset'];within.append(row)
            else:external.append(row)
    return predictions,pd.DataFrame(within),pd.DataFrame(external)


def comparisons(within,external):
    old_w=pd.read_csv(TABLES/'tfidf_logistic_holdout_results.csv')
    old_w=old_w[old_w.condition.eq('natural') & old_w.model.eq('tfidf_logistic')].set_index('dataset')
    old_t=pd.read_csv(TABLES/'tfidf_logistic_transfer_results.csv')
    old_t=old_t[old_t.condition.eq('natural')].set_index(['direction','scope'])
    old_d=pd.read_csv(TABLES/'tfidf_logistic_transfer_degradation.csv')
    require(np.allclose(old_d.cross_context-old_d.within_target,old_d.transfer_drop),'LR transfer-drop arithmetic changed')
    comparison,drop=[],[]
    for evaluation,frame in [('within',within),('transfer',external)]:
        for r in frame.itertuples():
            previous=old_w.loc[r.target_dataset] if evaluation=='within' else old_t.loc[(r.direction,r.scope)]
            require(previous.evaluation_uid_sha256==r.evaluation_uid_sha256 and previous.N==r.N,'LR/SVM evaluation rows differ')
            for metric in COMPARE:
                comparison.append(dict(evaluation=evaluation,direction=r.direction,target_dataset=r.target_dataset,scope=r.scope,
                    metric=metric,N=r.N,logistic=float(previous[metric]),linear_svm=float(getattr(r,metric)),
                    svm_minus_logistic=float(getattr(r,metric)-previous[metric]),same_evaluation_uids=True,
                    evaluation_uid_sha256=r.evaluation_uid_sha256))
    for r in external[external.scope.eq('same_target_holdout')].itertuples():
        w=within[within.dataset.eq(r.target_dataset)].iloc[0]
        require(w.evaluation_uid_sha256==r.evaluation_uid_sha256,'Transfer/within holdout UID mismatch')
        for metric in COMPARE:
            drop.append(dict(direction=r.direction,target_dataset=r.target_dataset,metric=metric,target_holdout_N=r.N,
                within_target=float(w[metric]),cross_context=float(getattr(r,metric)),transfer_drop=float(getattr(r,metric)-w[metric]),
                target_uid_sha256=r.evaluation_uid_sha256))
    return pd.DataFrame(comparison),pd.DataFrame(drop)


def validate_outputs(cv,selection,within,external,predictions,views,fits,comparison,drop):
    require(len(cv)==320 and not cv.duplicated(['dataset','config_id','fold']).any(),'CV output count/keys failed')
    require(not cv.holdout_used.any(),'Holdout influenced selection')
    require(selection.groupby('dataset').selected.sum().eq(1).all(),'Selection count differs')
    require(len(fits)==4 and fits.converged.all(),'Final fit failure')
    for frame,keys in [(within,['dataset']),(external,['direction','scope']),
        (predictions,['evaluation','direction','content_uid']),
        (comparison,['evaluation','direction','scope','metric']),(drop,['direction','metric'])]:
        require(not frame.duplicated(keys).any(),'Duplicate output keys')
    require(np.array_equal(predictions.predicted_label,(predictions.decision_score>=0).astype(int)),'Threshold is not zero')
    for (evaluation,direction),part in predictions.groupby(['evaluation','direction']):
        expected=views[part.target_dataset.iloc[0]]
        if evaluation=='within':expected=expected[expected.cv_fold.eq(-1)]
        require(set(part.content_uid)==set(expected.content_uid),'Eligible target rows missing')
    require(within.threshold.eq(0).all() and external.threshold.eq(0).all(),'Threshold changed')
    require(not any('probability' in c or 'calibration' in c for c in predictions.columns),'SVM probability output created')


def figures(within,external,comparison):
    directory=ROOT/'outputs/figures/tfidf_linear_svm';directory.mkdir(parents=True,exist_ok=True)
    for evaluation,title,filename in [('within','Within-domain holdout','within_macro_f1'),('transfer','Cross-context transfer','transfer_macro_f1')]:
        part=comparison[comparison.evaluation.eq(evaluation) & comparison.metric.eq('macro_f1')]
        labels=[r.target_dataset if evaluation=='within' else r.direction+'\n'+('all target' if r.scope=='all_eligible' else 'target holdout') for r in part.itertuples()]
        x=np.arange(len(part));fig,ax=plt.subplots(figsize=(7,4))
        ax.bar(x-.18,part.logistic,.36,label='Logistic Regression');ax.bar(x+.18,part.linear_svm,.36,label='Linear SVM')
        ax.set(xticks=x,xticklabels=labels,ylim=(0,1),ylabel='Macro F1',title=title+' — same evaluation rows')
        ax.legend();fig.tight_layout();fig.savefig(directory/(filename+'.png'),dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(8,7))
    for ax,r in zip(axes.ravel(),external.itertuples()):
        cm=np.array([[r.tn,r.fp],[r.fn,r.tp]]);ax.imshow(cm,cmap='Blues')
        for i in range(2):
            for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Real','Fake'],yticklabels=['Real','Fake'],
               xlabel='Predicted',ylabel='Actual',title=r.direction+' / '+('all target' if r.scope=='all_eligible' else 'target holdout'))
    fig.suptitle('Linear SVM transfer confusion matrices');fig.tight_layout();fig.savefig(directory/'transfer_confusion_matrices.png',dpi=180);plt.close(fig)


def write_report(selection,within,external,comparison,drop,cohorts,cv,fits):
    md=base.md;w=within.set_index('dataset');s=selection[selection.selected]
    findings=[f'Within-domain SVM macro F1 is {w.loc["FakeHealth","macro_f1"]:.4f} for FakeHealth and '
        f'{w.loc["CoAID","macro_f1"]:.4f} for CoAID (difference {w.loc["CoAID","macro_f1"]-w.loc["FakeHealth","macro_f1"]:+.4f}).']
    for direction in DIRECTIONS:
        part=external[external.direction.eq(direction)].set_index('scope');h=part.loc['same_target_holdout'];a=part.loc['all_eligible']
        d=drop[drop.direction.eq(direction)].set_index('metric')
        c=comparison[comparison.evaluation.eq('transfer') & comparison.direction.eq(direction) & comparison.scope.eq('same_target_holdout')].set_index('metric')
        findings += [f'{direction}: macro F1 {h.macro_f1:.4f} on target holdout and {a.macro_f1:.4f} on all eligible target rows; '
            f'holdout transfer-minus-within {d.loc["macro_f1","transfer_drop"]:+.4f}. Holdout ROC-AUC/AP are {h.roc_auc:.4f}/{h.average_precision:.4f}. '
            f'SVM-minus-LR holdout macro F1 is {c.loc["macro_f1","svm_minus_logistic"]:+.4f}. '
            f'Predicted Fake prevalence is {h.predicted_fake_prevalence:.2%} versus actual {h.actual_fake_prevalence:.2%}; '
            f'near-single-class flag={h.near_single_class_prediction}.']
    hold=external[external.scope.eq('same_target_holdout')].set_index('direction')
    delta=drop[drop.metric.eq('macro_f1')].set_index('direction')
    findings += [f'Directional differences are metric-specific: FH→CA minus CA→FH holdout macro F1 is '
        f'{hold.loc["FH_to_CA","macro_f1"]-hold.loc["CA_to_FH","macro_f1"]:+.4f}, while their within-to-transfer macro-F1 '
        f'drops are {delta.loc["FH_to_CA","transfer_drop"]:+.4f} and {delta.loc["CA_to_FH","transfer_drop"]:+.4f}. '
        'Read predicted-class bias, Fake F1 and ranking metrics together rather than treating a single macro-F1 difference as the entire asymmetry.']
    lr_within=comparison[comparison.evaluation.eq('within') & comparison.metric.eq('macro_f1')].set_index('target_dataset')
    lr_transfer=comparison[comparison.evaluation.eq('transfer') & comparison.scope.eq('same_target_holdout') & comparison.metric.eq('macro_f1')].set_index('direction')
    svm_gap=float(w.loc['CoAID','macro_f1']-w.loc['FakeHealth','macro_f1'])
    lr_gap=float(lr_within.loc['CoAID','logistic']-lr_within.loc['FakeHealth','logistic'])
    lr_drops={direction:float(lr_transfer.loc[direction,'logistic']-lr_within.loc[target,'logistic'])
              for direction,(_,target) in DIRECTIONS.items()}
    shared_pattern=bool(svm_gap>0 and lr_gap>0 and delta.transfer_drop.lt(0).all() and all(x<0 for x in lr_drops.values()))
    findings += [f'The prespecified qualitative pattern is reproduced by both classifiers: {shared_pattern}. '
        f'The CoAID-minus-FakeHealth within-domain macro-F1 gap is {svm_gap:+.4f} for SVM and {lr_gap:+.4f} for LR. '
        f'LR target-holdout macro-F1 drops are FH→CA {lr_drops["FH_to_CA"]:+.4f} and CA→FH {lr_drops["CA_to_FH"]:+.4f}, '
        'to compare with the SVM drops above. Reproduction here means a positive within-domain gap and negative transfer drops '
        'in both directions, not a post-hoc statistical test of effect size or superiority. The reported metric values and '
        'class-prediction biases determine how severe and asymmetric the transfer failure remains.']
    lines=['# TF-IDF Linear SVM model-robustness replication','',
        *[text for finding in findings for text in (finding,'')],
        '## Scope and unchanged cohorts','',
        'This replication changes the classical linear classifier and its prespecified tuning grid. Only natural combined_text is a predictor, '
        'on primary_modeling_eligible rows in FakeHealth HealthStory and CoAID News Specification A. Fake=1, Real=0. '
        'No source masking, title/body-only sensitivity, publisher-held-out SVM, transformers, or engagement analyses run. LR results are frozen comparators.','',
        md(cohorts[cohorts.partition.isin(['all','training_pool','holdout'])]),'',
        'The existing grouped CV and holdout assignments are verified against original manifests. cv_fold=-1 is reserved holdout; '
        'the five training-pool folds are unchanged. Every training/validation/test partition retains both labels with zero resource-group overlap. '
        'Cohort, split and comparator fingerprints are checked before and after execution.','',
        '## Pipeline and selection','',
        'Pipeline: TfidfVectorizer → LinearSVC. TF-IDF uses sublinear_tf=True, max_df=1.0, lowercase=True and default L2 normalization/smoothed IDF. '
        'The 32 configurations cross unigram/unigram+bigram, min_df=2/5, C=0.01/0.1/1/10, and class_weight=None/balanced. '
        'LinearSVC uses L2 penalty, squared-hinge loss, dual="auto", tol=1e-4, max_iter=20000, fit_intercept=True, intercept_scaling=1 and random_state=42. '
        'No grid expansion occurs.','',
        'Selection uses source training-pool grouped CV only: mean macro F1 within 0.001 of best, then fold sample SD within 0.001 of lowest, '
        'then unigram, smaller C, class_weight=None and configuration ID. Unlike the LR baseline’s extra min_df tie priority, this follows '
        'the explicitly requested SVM configuration-ID final tie break. Holdout outcomes never enter selection.','',
        md(s[['dataset','config_id','ngram_max','min_df','C','class_weight','macro_f1_mean','macro_f1_sd']]),'',
        'Both configurations are locked before final fitting/predictions. Within-domain models refit only the dataset-specific training pool. '
        'Transfer models refit all eligible source rows, including the former source holdout; target data is transform-only. '
        'All four unlabeled prediction bundles freeze before labels are joined for evaluation. Labels are inspected separately for preflight class support '
        'and used as source labels in the reverse direction; they never enter the opposite source fitting/selection interface.','',
        'Positive decision scores favor Fake, with prediction Fake iff score >=0. ROC-AUC and average precision use raw decision_function scores. '
        'Scores are not probabilities. No probability calibration, Brier score, probability averages, target threshold tuning or recalibration is performed.','',
        '## Within-domain holdout','',
        md(within[['dataset','N','accuracy','balanced_accuracy','macro_f1','fake_precision','fake_recall','fake_f1','real_precision','real_recall','real_f1','roc_auc','average_precision']]),'',
        '## Cross-context transfer','',
        md(external[['direction','scope','N','accuracy','balanced_accuracy','macro_f1','fake_precision','fake_recall','fake_f1','real_f1','roc_auc','average_precision']]),'',
        '## Confusion counts and class bias','',
        md(within[['dataset','tn','fp','fn','tp','actual_fake_prevalence','predicted_fake_prevalence']]),'',
        md(external[['direction','scope','tn','fp','fn','tp','actual_fake_prevalence','predicted_fake_prevalence','near_single_class_prediction']]),'',
        'TN/FP/FN/TP use Fake as the positive class. The descriptive near-single-class flag means predicted Fake prevalence <=5% or >=95%; '
        'it does not modify fitting, classification thresholds or outputs.','',
        '## Transfer degradation on identical target holdout UIDs','',md(drop),'',
        '## SVM versus frozen Logistic Regression','',md(comparison[['evaluation','direction','scope','metric','N','logistic','linear_svm','svm_minus_logistic']]),'',
        'Every comparison uses identical evaluation UIDs, verified against the saved LR result fingerprints; no LR prediction files or fitted models are reopened. '
        'Same rows enable direct descriptive comparisons but no paired significance test is performed. Both classifiers have independently selected source-only '
        'hyperparameters, and their C grids differ as prespecified. Small differences should not be presented as model superiority.','',
        '## Interpretation and limits','',
        'The tables above assess three distinct questions: the CoAID–FakeHealth within-domain gap, loss of performance when crossing contexts, '
        'and direction-specific error/ranking patterns. Transfer asymmetry includes different class-prediction biases and different drops from the relevant '
        'within-target baseline; its exact numerical ordering need not be identical across metrics or classifiers. '
        'The experiment tests robustness to classical linear classifier choice, not an optimized leaderboard objective.','',
        'Similar LR/SVM behavior does NOT prove that the pattern is model-independent in general. It only shows robustness across two classical linear text '
        'classifiers using TF-IDF. Dataset collection, topic, publisher/style and label-definition differences remain entangled; '
        'the replication does not identify a causal COVID-period effect, universal misinformation mechanism or probability calibration.','',
        '## Validation and reproducibility','',
        f'{len(cv)} CV fits and {len(fits)} final fits completed. All CV fits converged: {bool(cv.converged.all())}; '
        f'all final fits converged: {bool(fits.converged.all())}. Inner nonconvergence is logged and disqualifies the configuration; '
        'final nonconvergence stops execution. No methodological changes are made in response to transfer results.','',
        'Checks cover frozen splits/groups, training-only TF-IDF, predictor allowlisting, label-free target transforms, source-only selection, '
        'both configurations locked before transfer, full target coverage, decision threshold zero, unique output keys and unchanged LR comparator inputs. '
        'A runtime guard prevents LogisticRegression.fit throughout the experiment. Fitted final models, prediction locks and CV checkpoints support '
        'completed-run reuse without new fitting or predictions.','',
        'Tables: `outputs/tables/tfidf_linear_svm_{cv_results,model_selection,holdout_results,transfer_results,transfer_degradation,vs_logistic,cohort_summary,fit_manifest}.csv`. '
        'Predictions: `outputs/predictions/tfidf_linear_svm_predictions.csv` (no text). Figures: `outputs/figures/tfidf_linear_svm/`. '
        'Models/locks: `outputs/models/tfidf_linear_svm/`. Progress: `logs/tfidf_linear_svm_replication.log`. '
        'Script: `src/run_tfidf_linear_svm_replication.py`. Adjacent report JSON records protocol, checksums and validation.','']
    REPORT.write_text('\n'.join(lines))


def run():
    STATE.mkdir(parents=True,exist_ok=True);(ROOT/'logs').mkdir(exist_ok=True)
    logging.basicConfig(filename=ROOT/'logs/tfidf_linear_svm_replication.log',level=logging.INFO,format='%(asctime)s %(message)s',force=True)
    sig=signature();meta_path=REPORT.with_suffix('.json')
    if meta_path.exists():
        meta=json.loads(meta_path.read_text())
        require(meta['signature']==sig,'Completed SVM signature changed; do not silently rerun')
        require(meta['validation']=='passed' and all(sha(ROOT/p)==h for p,h in meta['outputs'].items()),'Saved SVM outputs changed')
        print('Validated Linear SVM replication reused; no fits or predictions repeated.');return
    sig_path=STATE/'run_signature.json'
    if sig_path.exists():require(json.loads(sig_path.read_text())==sig,'Stale SVM checkpoints')
    else:write_json(sig_path,sig)
    views={ds:load_features(ds) for ds in PREFIXES};cohorts=preflight(views)
    cohorts.to_csv(TABLES/'tfidf_linear_svm_cohort_summary.csv',index=False)
    logging.info('Frozen cohort/split preflight passed for both datasets')
    cv=pd.DataFrame([row for ds,d in views.items() for row in run_cv(ds,d)])
    selection=summarize_cv(cv);configs=lock_configs(selection)
    selection['selected']=[r.config_id==configs[r.dataset]['config_id'] for r in selection.itertuples()]
    bundles=freeze_predictions(views,configs)
    predictions,within,external=evaluate(bundles)
    comparison,drop=comparisons(within,external)
    fits=pd.DataFrame([dict(bundle=key,**b['fit']) for key,b in bundles.items()])
    validate_outputs(cv,selection,within,external,predictions,views,fits,comparison,drop)
    tables=dict(cv_results=cv,model_selection=selection,holdout_results=within,transfer_results=external,
                transfer_degradation=drop,vs_logistic=comparison,cohort_summary=cohorts,fit_manifest=fits)
    paths=[]
    for name,frame in tables.items():
        path=TABLES/f'tfidf_linear_svm_{name}.csv';frame.to_csv(path,index=False);paths.append(path)
    predictions.to_csv(PRED_PATH,index=False);paths.append(PRED_PATH)
    figures(within,external,comparison)
    write_report(selection,within,external,comparison,drop,cohorts,cv,fits)
    require(signature()==sig,'Original cohorts/splits/LR outputs changed')
    paths += [REPORT,*STATE.glob('*.json'),*STATE.glob('*.joblib'),*(ROOT/'outputs/figures/tfidf_linear_svm').glob('*.png')]
    write_json(meta_path,dict(signature=sig,validation='passed',cv_fit_count=len(cv),final_fit_count=len(fits),
        all_fits_converged=bool(cv.converged.all() and fits.converged.all()),threshold=0.,probability_calibration=False,
        source_only_fitting=True,target_labels_joined_after_prediction_freeze=True,logistic_fit_count=0,
        original_inputs_unchanged=True,outputs={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}))
    logging.info('Complete: %d CV and %d final SVM fits; validation passed',len(cv),len(fits))
    print('Linear SVM replication complete; validation passed. Report: outputs/reports/tfidf_linear_svm_replication.md')


def main():
    with patch.object(LogisticRegression,'fit',side_effect=RuntimeError('LR refitting prohibited in SVM replication')):
        run()


if __name__=='__main__':
    main()
