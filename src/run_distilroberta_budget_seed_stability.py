"""Four fixed FH-source refits; reuse seed 42 and compare paired token budgets."""
from pathlib import Path
import argparse
import fcntl
import gc
import inspect
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import run_distilroberta_text_experiment as base

ROOT=base.ROOT
MODELS=ROOT/'outputs/models/distilroberta_budget_seed_stability'
TABLES=ROOT/'outputs/tables'
LOG=ROOT/'logs/distilroberta_budget_seed_stability.log'
REPORT=ROOT/'outputs/reports/distilroberta_budget_seed_stability.md'
PREDICTIONS=ROOT/'outputs/predictions/distilroberta_budget_seed_predictions.csv'
SEEDS=(42,123,2026)
BUDGETS=(512,256)
PAIRED_METRICS=base.METRICS+['predicted_fake_prevalence']
READ_LABELS=base.read_labels
ORIGINAL_SETTINGS=base.SETTINGS.copy()
ORIGINAL_MODELS=base.MODELS
sha=base.audit.sha
SEED42=[]


def table(name,rows):
    path=TABLES/f'distilroberta_budget_seed_{name}.csv'
    tmp=path.with_suffix('.tmp');pd.DataFrame(rows).to_csv(tmp,index=False);tmp.replace(path)


def prior_snapshot():
    # Metadata only: no unrelated analytical file contents are opened.
    return {str(p.relative_to(ROOT)):[p.stat().st_size,p.stat().st_mtime_ns]
        for p in (ROOT/'outputs').rglob('*') if p.is_file() and
        'distilroberta_budget_seed' not in str(p) and p.name!='fh_to_coaid_budget_seed_paired_deltas.svg'}


def source_labels(dataset,uids):
    if dataset!='FakeHealth':raise ValueError('Only FakeHealth labels may enter fitting')
    return READ_LABELS(dataset,uids)


def clear_device():
    gc.collect();(base.torch.mps.empty_cache() if base.DEVICE=='mps' else None)


def refresh_manifest():
    rows=list(SEED42)
    for spec_path in sorted(MODELS.glob('fits/*/fit_spec.json')):
        spec=json.loads(spec_path.read_text());folder=spec_path.parent
        epochs=[json.loads(p.read_text()) for p in sorted(folder.glob('epoch-*.json'))]
        prediction=folder/'predictions-FH_to_CA-unlabeled.csv'
        rows.append(dict(seed=spec['settings']['seed'],max_length=spec['max_length'],reused=False,
            fit_fingerprint=folder.name,source_dataset=spec['dataset'],source_N=1647,
            source_uid_sha256=spec['train_uid_sha256'],learning_rate=spec['learning_rate'],epochs=spec['epochs'],
            completed_epochs=len(epochs),completion_state='completed' if len(epochs)==spec['epochs'] else 'partial',
            checkpoint_revision=spec['settings']['revision'],microbatch=spec['microbatch'],
            gradient_accumulation=spec['accumulation'],protocol_fingerprint=base.digest(spec['settings']),
            input_fingerprint=base.digest(spec['signature']['inputs']),fit_spec=str(spec_path.relative_to(ROOT)),
            prediction_path=str(prediction.relative_to(ROOT)),
            prediction_sha256=sha(prediction) if prediction.exists() else '',
            epoch_seconds=sum(e['epoch_seconds'] for e in epochs)))
    table('fit_manifest',rows)
    return rows


def prepare():
    if sys.version_info[:2]!=(3,12) or Path(sys.prefix).resolve()!=ROOT/'.venv-transformer':
        raise RuntimeError('Use .venv-transformer/bin/python (3.12)')
    previous=json.loads((ROOT/'outputs/reports/distilroberta_text_experiment.json').read_text())
    if previous['status']!='complete':raise ValueError('Prior experiment not validated')
    if sha(base.__file__)!=previous['final_script_sha256']:raise ValueError('Prior helper code changed')
    old=previous['signature']
    if ORIGINAL_SETTINGS!=old['settings'] or base.torch.__version__!=old['torch_version']:
        raise ValueError('Prior protocol/runtime changed')
    if sha(base.audit.__file__)!=old['packer_sha256'] or sha(ROOT/'requirements-transformer.txt')!=old['requirements_sha256']:
        raise ValueError('Packer/environment fingerprint mismatch')
    for path,value in previous['outputs'].items():
        if sha(ROOT/path)!=value:raise ValueError('Prior artifact changed: '+path)
    for path,value in old['inputs'].items():
        if sha(ROOT/path)!=value:raise ValueError('Cohort/split fingerprint changed')
    selection=pd.read_csv(TABLES/'distilroberta_model_selection.csv')
    selected=selection[selection.dataset.eq('FakeHealth')]
    if len(selected)!=1:raise ValueError('Ambiguous source configuration')
    config=dict(learning_rate=float(selected.iloc[0].learning_rate),epochs=int(selected.iloc[0].epochs))
    cohorts={}
    for name,prefix in base.PREFIXES.items():
        d=pd.read_csv(ROOT/f'data/processed/{prefix}_text_modeling_ready.csv',
            usecols=['content_uid','primary_modeling_eligible','cv_fold','holdout_assignment','leakage_group'])
        if d.primary_modeling_eligible.dtype!=bool:raise ValueError('Nonboolean eligibility')
        d=d[d.primary_modeling_eligible].sort_values('content_uid').reset_index(drop=True)
        if not d.content_uid.is_unique or not d.groupby('leakage_group').cv_fold.nunique().eq(1).all():
            raise ValueError('Cohort/group separation changed')
        cohorts[name]=d
    source=cohorts['FakeHealth'];target=cohorts['CoAID']
    if (len(source),len(target))!=(1647,3577):raise ValueError('Wrong cohort sizes')
    holdout=pd.read_csv(ROOT/'outputs/splits/coaid_text_holdout.csv',usecols=['content_uid','assignment','leakage_group']).set_index('content_uid').loc[target.content_uid]
    if holdout.assignment.tolist()!=target.holdout_assignment.tolist() or holdout.leakage_group.tolist()!=target.leakage_group.tolist():
        raise ValueError('Target holdout/group assignment changed')
    if not target.cv_fold.eq(-1).eq(target.holdout_assignment.eq('test')).all() or target.cv_fold.eq(-1).sum()!=702:
        raise ValueError('Target holdout coverage changed')
    packed={cap:base.pack_features('CoAID',target,cap) for cap in BUDGETS}
    if packed[256]!=packed[512]:raise ValueError('STOP: unpadded CoAID content differs across budgets')
    content_audit=dict(target_N=len(target),identical_unpadded_content=True,
        target_uid_sha256=base.uid_hash(target.content_uid),holdout_uid_sha256=base.uid_hash(target[target.cv_fold.eq(-1)].content_uid),
        packed_256_sha256=base.digest(packed[256]),packed_512_sha256=base.digest(packed[512]),
        maximum_unpadded_length=max(map(len,packed[256].values())))
    weights=base.weights_from_training(source_labels('FakeHealth',source.content_uid))
    manifest=pd.read_csv(TABLES/'distilroberta_fit_manifest.csv')
    subset=manifest[manifest.dataset.eq('FakeHealth')&manifest.role.eq('transfer_refit')&manifest.epoch.eq(config['epochs'])]
    if len(subset)!=2 or set(subset.max_length)!=set(BUDGETS):raise ValueError('Missing seed-42 fits')
    lock=ORIGINAL_MODELS/'locked_configurations.json'
    if sha(lock)!=previous['config_lock_sha256']:raise ValueError('Original configuration lock changed')
    global SEED42
    SEED42=[];paths={}
    for row in subset.itertuples():
        folder=(ROOT/row.checkpoint).parent;spec_path=folder/'fit_spec.json';spec=json.loads(spec_path.read_text())
        expected=dict(dataset='FakeHealth',role='transfer_refit',fold=-1,learning_rate=config['learning_rate'],
            epochs=config['epochs'],max_length=row.max_length,train_uid_sha256=base.uid_hash(source.content_uid),
            validation_uid_sha256=base.uid_hash([]),training_label_sha256=base.digest(source_labels('FakeHealth',source.content_uid).tolist()),
            settings=ORIGINAL_SETTINGS,signature=old,microbatch=2,accumulation=8)
        if spec!=expected or base.digest(spec)!=folder.name:raise ValueError('Seed-42 fit/protocol mismatch')
        if row.fingerprint!=base.digest(dict(fit=folder.name,epoch=config['epochs'])) or not (ROOT/row.checkpoint).exists():
            raise ValueError('Seed-42 checkpoint provenance mismatch')
        np.testing.assert_allclose([row.class_weight_real,row.class_weight_fake],weights,rtol=0,atol=1e-15)
        path=folder/'predictions-FH_to_CA-unlabeled.csv';meta=json.loads(path.with_suffix('.json').read_text())
        if sha(path)!=meta['prediction_sha256'] or meta['observed_labels_joined']:
            raise ValueError('Seed-42 prediction freeze mismatch')
        e=meta['expected']
        if e['config_lock_sha256']!=sha(lock) or e['target_uid_sha256']!=content_audit['target_uid_sha256'] or e['model_checkpoint']!=row.checkpoint:
            raise ValueError('Seed-42 prediction/model identity mismatch')
        if e['identity']!=dict(evaluation='transfer',direction='FH_to_CA',source_dataset='FakeHealth',target_dataset='CoAID',max_length=row.max_length):
            raise ValueError('Seed-42 direction/budget mismatch')
        paths[(42,row.max_length)]=path
        SEED42.append(dict(seed=42,max_length=row.max_length,reused=True,fit_fingerprint=folder.name,
            source_dataset='FakeHealth',source_N=len(source),source_uid_sha256=base.uid_hash(source.content_uid),
            **config,completed_epochs=config['epochs'],completion_state='completed',checkpoint_revision=base.REVISION,
            microbatch=2,gradient_accumulation=8,protocol_fingerprint=base.digest(ORIGINAL_SETTINGS),
            input_fingerprint=base.digest(old['inputs']),fit_spec=str(spec_path.relative_to(ROOT)),
            prediction_path=str(path.relative_to(ROOT)),prediction_sha256=sha(path),epoch_seconds=0.))
    signature=dict(inputs=old['inputs'],prior_signature=old,original_script_sha256=sha(base.__file__),
        script_sha256=sha(__file__),config=config,settings=ORIGINAL_SETTINGS,seeds=list(SEEDS),budgets=list(BUDGETS),
        source_uid_sha256=base.uid_hash(source.content_uid),content_audit=content_audit,
        seed42_predictions={str(cap):sha(paths[(42,cap)]) for cap in BUDGETS})
    state_path=MODELS/'protocol_lock.json'
    if state_path.exists():
        state=json.loads(state_path.read_text())
        if state['signature']!=signature:raise ValueError('Stability protocol fingerprint changed')
        if state['prior_outputs']!=prior_snapshot():raise ValueError('Prior outputs changed')
        if state['classical_environment']!=base.old_environment_fingerprint():raise ValueError('Classical environment changed')
    else:
        state=dict(signature=signature,prior_outputs=prior_snapshot(),classical_environment=base.old_environment_fingerprint(),started_at=time.time())
        base.write_json(state_path,state)
    base.write_json(MODELS/'target_content_audit.json',content_audit)
    refresh_manifest()
    logging.info('Preflight passed: seed 42 reused; CoAID unpadded content identical at 256/512; no target labels read')
    return source,target,packed[256],config,signature,state,paths


def evaluate(paths,target):
    # Validate all six freezes before the first target-label read in this script.
    frames=[]
    for (seed,cap),path in sorted(paths.items()):
        meta=json.loads(path.with_suffix('.json').read_text())
        if sha(path)!=meta['prediction_sha256'] or meta['observed_labels_joined']:raise ValueError('Broken prediction freeze')
        p=pd.read_csv(path,float_precision='round_trip')
        if 'true_label' in p or not p.content_uid.is_unique or set(p.content_uid)!=set(target.content_uid):
            raise ValueError('Premature labels or incorrect target coverage')
        if not p.predicted_label.eq(p.fake_probability.ge(.5).astype(int)).all():raise ValueError('Threshold changed')
        p['seed']=seed;p['max_length']=cap;p['prediction_sha256']=meta['prediction_sha256'];frames.append(p)
    predictions=pd.concat(frames,ignore_index=True)
    labels=dict(zip(target.content_uid,READ_LABELS('CoAID',target.content_uid)))
    predictions['true_label']=predictions.content_uid.map(labels)
    old=pd.read_csv(TABLES/'distilroberta_transfer_results.csv',float_precision='round_trip')
    results=[]
    for (seed,cap),p in predictions.groupby(['seed','max_length']):
        for scope,part in [('all_eligible',p),('same_target_holdout',p[p.cv_fold.eq(-1)])]:
            metrics=base.metrics(part.true_label,part.fake_probability)
            if seed==42:
                original=old[old.direction.eq('FH_to_CA')&old.max_length.eq(cap)&old.scope.eq(scope)].iloc[0]
                if original.evaluation_uid_sha256!=base.uid_hash(part.content_uid):raise ValueError('Seed-42 evaluation rows changed')
                for key,value in metrics.items():
                    if not np.isclose(value,original[key],rtol=0,atol=2e-15):raise ValueError('Seed-42 metric mismatch: '+key)
                # Carry forward the exact stored floats, without rounding or refitting.
                metrics={key:original[key].item() if hasattr(original[key],'item') else original[key] for key in metrics}
            results.append(dict(seed=int(seed),max_length=int(cap),scope=scope,**metrics,
                evaluation_uid_sha256=base.uid_hash(part.content_uid),reused_seed42=seed==42))
    predictions.to_csv(PREDICTIONS,index=False);table('results',results)
    return pd.DataFrame(results),predictions


def summarize(results,predictions):
    deltas=[];summary=[];agreement=[]
    for (seed,scope),p in results.groupby(['seed','scope']):
        indexed=p.set_index('max_length')
        for metric in PAIRED_METRICS:
            deltas.append(dict(seed=int(seed),scope=scope,metric=metric,value_512=float(indexed.loc[512,metric]),
                value_256=float(indexed.loc[256,metric]),delta_256_minus_512=float(indexed.loc[256,metric]-indexed.loc[512,metric])))
    deltas=pd.DataFrame(deltas);table('paired_deltas',deltas)
    for (scope,cap),p in results.groupby(['scope','max_length']):
        for metric in ['macro_f1','roc_auc','average_precision','fake_f1']:
            a=p[metric]
            summary.append(dict(kind='budget',scope=scope,max_length=int(cap),metric=metric,N_seeds=len(a),
                mean=a.mean(),sample_sd=a.std(ddof=1),minimum=a.min(),maximum=a.max(),positive_seeds=None))
    for (scope,metric),p in deltas.groupby(['scope','metric']):
        a=p.delta_256_minus_512
        summary.append(dict(kind='paired_delta',scope=scope,max_length=None,metric=metric,N_seeds=len(a),
            mean=a.mean(),sample_sd=a.std(ddof=1),minimum=a.min(),maximum=a.max(),positive_seeds=int(a.gt(0).sum())))
    for seed,p in predictions.groupby('seed'):
        for scope,part in [('all_eligible',p),('same_target_holdout',p[p.cv_fold.eq(-1)])]:
            x=part[part.max_length.eq(512)].set_index('content_uid').sort_index()
            y=part[part.max_length.eq(256)].set_index('content_uid').sort_index()
            agreement.append(dict(seed=int(seed),scope=scope,identical_class_pct=100*x.predicted_label.eq(y.predicted_label).mean(),
                probability_correlation=x.fake_probability.corr(y.fake_probability),
                mean_absolute_probability_difference=(x.fake_probability-y.fake_probability).abs().mean()))
    table('summary',summary)
    base.write_json(MODELS/'prediction_agreement.json',agreement)
    return deltas,pd.DataFrame(summary),agreement


def validate(source,target,paths,signature,state,results,predictions):
    specs=list(MODELS.glob('fits/*/fit_spec.json'))
    if len(specs)!=4:raise ValueError('Expected exactly four new source fits')
    combinations=set();epoch_fps=[]
    for path in specs:
        s=json.loads(path.read_text());seed=s['settings']['seed'];cap=s['max_length'];combinations.add((seed,cap))
        expected_settings={**ORIGINAL_SETTINGS,'seed':seed}
        if s['settings']!=expected_settings or s['learning_rate']!=signature['config']['learning_rate'] or s['epochs']!=signature['config']['epochs']:
            raise ValueError('Hyperparameter drift')
        if s['dataset']!='FakeHealth' or s['train_uid_sha256']!=base.uid_hash(source.content_uid) or s['validation_uid_sha256']!=base.uid_hash([]):
            raise ValueError('Non-source rows entered fit')
        if (s['microbatch'],s['accumulation'])!=(2,8) or base.digest(s)!=path.parent.name:raise ValueError('Fit fingerprint drift')
        expected_weights=base.weights_from_training(source_labels('FakeHealth',source.content_uid))
        for epoch in range(1,s['epochs']+1):
            row=json.loads((path.parent/f'epoch-{epoch}.json').read_text());epoch_fps.append(row['fingerprint'])
            if row['fingerprint']!=base.digest(dict(fit=path.parent.name,epoch=epoch)) or not (ROOT/row['checkpoint']).exists():raise ValueError('Incomplete checkpoint')
            np.testing.assert_allclose([row['class_weight_real'],row['class_weight_fake']],expected_weights,rtol=0,atol=1e-15)
            if row['validation_metrics']:raise ValueError('Unexpected validation/tuning')
    if combinations!={(seed,cap) for seed in (123,2026) for cap in BUDGETS} or len(set(epoch_fps))!=len(epoch_fps):
        raise ValueError('Wrong/duplicated new fits')
    if len(paths)!=6 or len(results)!=12 or len(predictions)!=6*len(target) or predictions.duplicated(['seed','max_length','content_uid']).any():
        raise ValueError('Incomplete evaluation coverage')
    if len({str(p) for p in paths.values()})!=6 or len({sha(p) for p in paths.values()})!=6:raise ValueError('Nonunique prediction fingerprints')
    if prior_snapshot()!=state['prior_outputs'] or base.old_environment_fingerprint()!=state['classical_environment']:
        raise ValueError('Prior outputs/environment changed')
    if any(sha(ROOT/p)!=h for p,h in signature['inputs'].items()):raise ValueError('Inputs changed')
    return dict(passed=True,new_fits=4,reused_seed42_fits=2,new_epoch_checkpoints=len(epoch_fps),
        unique_fit_and_prediction_fingerprints=True,locked_protocol_unchanged=True,
        seed42_metrics_reused_and_verified=True,target_content_identical=True,source_only_fitting=True,
        freezes_verified_before_target_labels=True,threshold=.5,prior_outputs_unchanged=True,
        classical_environment_unchanged=True,prediction_rows=len(predictions),cohorts_and_splits_unchanged=True)


def finish(source,target,paths,signature,state):
    results,predictions=evaluate(paths,target);deltas,summary,agreement=summarize(results,predictions)
    manifest=refresh_manifest();validation=validate(source,target,paths,signature,state,results,predictions)
    primary=deltas[deltas.scope.eq('same_target_holdout')&deltas.metric.isin(['macro_f1','roc_auc','average_precision'])]
    all_positive=bool(primary.delta_256_minus_512.gt(0).all())
    interpretation=('All tested seeds agree on a positive direction for holdout macro F1, ROC-AUC and AP. '
        'Inspect all effect sizes below before calling the changes sizable; no post-hoc numerical cutoff is used.' if all_positive else
        'The original 256-token improvement is training-seed sensitive: paired effects reverse or disagree across seeds/metrics. '
        'It should not support a substantive claim that reducing the source representation consistently improves transfer.')
    report=['# FH-to-CoAID token-budget stability across training seeds','',interpretation,'',
        'This targeted check does not retune or replace the primary transformer experiment. Its main interpretation—'
        'no consistent cross-context benefit of the primary 512-token transformer—remains unchanged. '
        'Any token-budget claim must account for these paired-seed results. No other direction or model was fitted.','',
        '## Fixed protocol','',
        f'Checkpoint `{base.audit.CHECKPOINT}` revision `{base.REVISION}`. Learning rate '
        f'`{signature["config"]["learning_rate"]!r}`, epochs `{signature["config"]["epochs"]}` read directly from the selection CSV. '
        'Seeds 42, 123 and 2026; caps 512 and 256. Seed 42 reuses both original frozen predictions and exact stored metrics. '
        'Only four new all-FakeHealth-source fits were run. No CV, within-context refit, target fitting or calibration.','',
        json.dumps({**ORIGINAL_SETTINGS,'seed':list(SEEDS),'microbatch':2,'gradient_accumulation':8},sort_keys=True),'',
        'The unchanged training helper supplies AdamW, constant LR, zero warmup, weighted CE normalized within each effective batch, '
        'gradient clipping and dynamic padding. Class weights use only each fit’s FakeHealth labels. '
        'All 1,647 source rows and 3,577 target rows are identical across budgets/seeds; the existing target holdout contains 702 rows. '
        'All six prediction files were verified frozen before this script read any CoAID labels.','',
        '## Target-content identity','',
        json.dumps(signature['content_audit'],sort_keys=True),'',
        'All unpadded CoAID token sequences are empirically identical at both caps. Only title and body enter packing. '
        'The source FakeHealth body prefix changes; full titles are preserved. No article-level text inspection was performed.','',
        '## Seed-level results','',
        base.audit.markdown_table(results.to_dict('records'),['seed','max_length','scope','macro_f1','balanced_accuracy','fake_f1',
            'roc_auc','average_precision','actual_fake_prevalence','predicted_fake_prevalence','tn','fp','fn','tp']),'',
        '## Primary paired differences: 256 minus 512','',
        base.audit.markdown_table(deltas.to_dict('records'),['seed','scope','metric','value_512','value_256','delta_256_minus_512']),'',
        '## Descriptive seed summaries','',
        base.audit.markdown_table(summary.to_dict('records'),['kind','scope','max_length','metric','N_seeds','mean','sample_sd','minimum','maximum','positive_seeds']),'',
        'Sample SD uses ddof=1. Three seeds do not support reliable population uncertainty: no significance test or confidence interval was constructed.','',
        '## Prediction agreement','',base.audit.markdown_table(agreement,['seed','scope','identical_class_pct','probability_correlation','mean_absolute_probability_difference']),'',
        '## Validation and limitations','',json.dumps(validation,sort_keys=True),'',
        'Seeds control initialization, dropout and epoch shuffling using the same helper as the original experiment. '
        'MPS is not asserted bitwise reproducible. Source-budget and stochastic optimization interact; this is not a causal '
        'COVID effect, a universal cue claim or a matched-information correction. The original 71.22%/87.86% FakeHealth '
        'truncation at 512/256 versus 0% for CoAID remains relevant.','',
        f'Completed epoch timer sum: {sum(r["epoch_seconds"] for r in manifest if not r["reused"]):.1f} seconds. '
        f'Elapsed wall time: {time.time()-state["started_at"]:.1f} seconds. FP32 and numerical hyperparameters remain unchanged. '
        'The existing MPS cache-release resource policy is reused, with a fresh process after each completed new fit/prediction. '
        'Completed epoch checkpoints resume without retraining completed fits.','',
        'Outputs: `outputs/tables/distilroberta_budget_seed_*.csv`, `outputs/predictions/distilroberta_budget_seed_predictions.csv`, '
        '`outputs/models/distilroberta_budget_seed_stability/`, and `logs/distilroberta_budget_seed_stability.log`.','']
    REPORT.write_text('\n'.join(report))
    meta=dict(status='complete',signature=signature,validation=validation,all_holdout_directions_positive=all_positive,
        elapsed_seconds=time.time()-state['started_at'],outputs={str(p.relative_to(ROOT)):sha(p) for p in
        [*TABLES.glob('distilroberta_budget_seed_*.csv'),PREDICTIONS,REPORT]})
    base.write_json(REPORT.with_suffix('.json'),meta)
    logging.info('STABILITY CHECK COMPLETE: four new fits, seed 42 reused, all validations passed')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    MODELS.mkdir(parents=True,exist_ok=True)
    lock=(MODELS/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    logging.basicConfig(filename=LOG,level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    base.hf_logging.disable_progress_bar();base.hf_logging.set_verbosity_error();base.torch.set_num_threads(4)
    source,target,target_features,config,signature,state,paths=prepare()
    if args.preflight_only:return
    if base.DEVICE!='mps':base.logging.warning('Running on %s; reported results were produced on MPS and bitwise reproducibility across devices is not claimed',base.DEVICE)
    meta_path=REPORT.with_suffix('.json')
    if meta_path.exists():
        done=json.loads(meta_path.read_text())
        if done['status']=='complete' and done['signature']==signature:
            if not all(sha(ROOT/p)==h for p,h in done['outputs'].items()):raise ValueError('Completed output fingerprint mismatch')
            logging.info('Matching completed check reused; no fits run');return
    base.MODELS=MODELS;base.read_labels=source_labels;base.refresh_manifest=refresh_manifest;base.clear_device=clear_device
    from torch.optim.optimizer import register_optimizer_step_post_hook
    hook=register_optimizer_step_post_hook(base.release_step_cache)
    for seed in (123,2026):
        base.SEED=seed;base.SETTINGS={**ORIGINAL_SETTINGS,'seed':seed}
        for cap in BUDGETS:
            # No target feature/label object is passed to the fit helper.
            features=base.pack_features('FakeHealth',source,cap)
            logging.info('Fixed source fit seed=%s budget=%s',seed,cap)
            folder,_=base.fit('FakeHealth',source.content_uid.tolist(),features,config['learning_rate'],config['epochs'],
                cap,'budget_seed_transfer_refit',-1,signature,dict(microbatch=2,accumulation=8))
            del features
            prediction=folder/'predictions-FH_to_CA-unlabeled.csv';already_frozen=prediction.with_suffix('.json').exists()
            identity=dict(evaluation='transfer',direction='FH_to_CA',source_dataset='FakeHealth',target_dataset='CoAID',seed=seed,max_length=cap)
            paths[(seed,cap)]=base.frozen_prediction(folder,config['epochs'],target_features,target,2,identity,base.digest(signature))
            refresh_manifest()
            if not already_frozen:
                logging.info('Saved fit and frozen predictions seed=%s budget=%s; recycling MPS process',seed,cap)
                for handler in logging.getLogger().handlers:handler.flush()
                os.execv(sys.executable,[sys.executable,str(Path(__file__).resolve())])
    hook.remove();finish(source,target,paths,signature,state)


if __name__=='__main__':main()
