"""Pinned DistilRoBERTa experiment; resumable epochs, source-only fitting."""
from pathlib import Path
import argparse
import gc
import hashlib
import html
import inspect
import json
import logging
import math
import os
import random
import sys
import time

os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification
from transformers.utils import logging as hf_logging
import audit_transformer_feasibility as audit


def select_device():
    """Device selection for release portability: mps, then cuda, then cpu."""
    device='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    logging.warning('Selected torch device %s. Reported results were produced on Apple MPS; bitwise reproducibility across devices is not claimed.',device)
    return device
DEVICE=select_device()

ROOT=audit.ROOT
MODELS=ROOT/'outputs/models/distilroberta'
TABLES=ROOT/'outputs/tables'
FIGURES=ROOT/'outputs/figures/distilroberta'
LOG=ROOT/'logs/distilroberta_text_experiment.log'
REVISION='fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b'
SEED=42
METRICS=['macro_f1','balanced_accuracy','fake_f1','roc_auc','average_precision']
PREFIXES=audit.PREFIXES
SETTINGS=dict(checkpoint=audit.CHECKPOINT,revision=REVISION,seed=SEED,precision='float32',
    optimizer='torch.optim.AdamW',betas=[.9,.999],epsilon=1e-8,weight_decay=.01,
    decay_policy='all parameters except bias and LayerNorm weights',scheduler='constant',warmup_steps=0,
    gradient_clip_norm=1.,effective_batch_size=16,threshold=.5,
    packing='validated full-title pair; body prefix; dynamic padding',class_weight='N_train/(2*N_train,c)')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');temp.replace(path)


def write_table(name,rows):
    path=TABLES/f'distilroberta_{name}.csv';temp=path.with_suffix('.tmp')
    pd.DataFrame(rows).to_csv(temp,index=False);temp.replace(path)


def seed_all():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);(torch.mps.manual_seed(SEED) if DEVICE=='mps' else None)


def clear_device():
    gc.collect();(torch.mps.empty_cache() if DEVICE=='mps' else None)
    # Resource-only restart after a fully saved fit: release driver/graph caches
    # that empty_cache cannot reclaim. No training settings or weights change.
    if (os.environ.get('DISTILROBERTA_RECYCLE_AFTER_FIT')=='1' and
            inspect.currentframe().f_back.f_code.co_name=='fit'):
        logging.info('Recycle MPS worker after completed fit; reuse all completed checkpoints')
        for handler in logging.getLogger().handlers:handler.flush()
        os.execv(sys.executable,[sys.executable,str(Path(__file__).resolve()),'--phase','all'])


def new_model():
    seed_all()
    model=AutoModelForSequenceClassification.from_pretrained(audit.CHECKPOINT,revision=REVISION,
        cache_dir=audit.CACHE,local_files_only=True,use_safetensors=True,num_labels=2,
        id2label={0:'Real',1:'Fake'},label2id={'Real':0,'Fake':1})
    return model.to(device=DEVICE,dtype=torch.float32)


def parameter_digest(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def pack_features(dataset,assignment,max_length):
    # This is the only representation constructor; labels/metadata never enter it.
    d=pd.read_csv(ROOT/f'data/processed/{PREFIXES[dataset]}_text_modeling_ready.csv',
        usecols=['content_uid','title_text','body_text'],keep_default_na=False).set_index('content_uid').loc[assignment.content_uid]
    tokenizer=audit.load_tokenizer();result={}
    for row in d.itertuples():
        title=tokenizer(row.title_text,add_special_tokens=False,truncation=False)['input_ids']
        body=tokenizer(row.body_text,add_special_tokens=False,truncation=False)['input_ids']
        ids=audit.pack_title_body(tokenizer,title,body,max_length)
        assert ids==audit.pack_title_body(tokenizer,title,body,max_length)
        result[row.Index]=ids
    return result


def batch_tensor(features,uids,fixed_length=None):
    length=fixed_length or max(len(features[u]) for u in uids)
    ids=torch.tensor([features[u]+[1]*(length-len(features[u])) for u in uids],device=DEVICE)
    return dict(input_ids=ids,attention_mask=ids.ne(1))


def weights_from_training(y):
    counts=np.bincount(np.asarray(y,dtype=int),minlength=2)
    if (counts==0).any():raise ValueError('Training fit lacks a class')
    return len(y)/(2.*counts)


def memory_pilot(cohorts,inputs):
    fp=digest(dict(inputs=inputs,settings=SETTINGS,max_length=512,pilot_version=1))
    saved=MODELS/'memory_pilot.json'
    if saved.exists():
        previous=json.loads(saved.read_text())
        if previous['fingerprint']!=fp:raise ValueError('Pilot fingerprint changed')
        return previous
    features=pack_features('FakeHealth',cohorts['FakeHealth'].iloc[:2],512)
    uids=list(features);rows=[]
    for microbatch in (1,2):
        row=dict(microbatch=microbatch,max_length=512,device=DEVICE,forward_succeeded=False,
            backward_succeeded=False,parameters_unchanged=False,optimizer_steps=0,
            labels='synthetic alternating labels; no observed labels read',fingerprint=fp)
        model=None;loss=None;output=None;batch=None;start=time.perf_counter()
        try:
            model=new_model();model.train();before=parameter_digest(model)
            batch=batch_tensor(features,uids[:microbatch],fixed_length=512)
            y=torch.tensor([i%2 for i in range(microbatch)],device=DEVICE)
            # Unequal synthetic training counts exercise the weighted CE code path.
            weights=torch.tensor(weights_from_training([0,0,0,1]),device=DEVICE,dtype=torch.float32)
            output=model(**batch).logits
            loss=torch.nn.functional.cross_entropy(output,y,weight=weights)
            (torch.mps.synchronize() if DEVICE=='mps' else None);row['forward_succeeded']=bool(torch.isfinite(loss).item())
            if not row['forward_succeeded']:raise RuntimeError('Nonfinite pilot loss')
            loss.backward();(torch.mps.synchronize() if DEVICE=='mps' else None)
            if not all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None):
                raise RuntimeError('Nonfinite pilot gradient')
            row['backward_succeeded']=True
            row['sampled_mps_allocated_bytes']=(torch.mps.current_allocated_memory() if DEVICE=='mps' else 0)
            row['sampled_mps_driver_bytes']=(torch.mps.driver_allocated_memory() if DEVICE=='mps' else 0)
            row['parameters_unchanged']=parameter_digest(model)==before
            if not row['parameters_unchanged']:raise RuntimeError('Pilot changed parameters')
        except Exception as exc:
            row['error']=type(exc).__name__+': '+str(exc)
        finally:
            row['elapsed_seconds']=time.perf_counter()-start
            row['memory_note']='post-backward sample; not transient peak; optimizer state absent'
            del output,loss,batch,model;clear_device()
        rows.append(row);write_table('memory_pilot',rows)
        logging.info('Phase A microbatch=%s forward=%s backward=%s unchanged=%s elapsed=%.1fs',
            microbatch,row['forward_succeeded'],row['backward_succeeded'],row['parameters_unchanged'],row['elapsed_seconds'])
        if not (row['backward_succeeded'] and row['parameters_unchanged']):break
    passed=[r['microbatch'] for r in rows if r['backward_succeeded'] and r['parameters_unchanged']]
    chosen=max(passed) if passed else None
    result=dict(fingerprint=fp,passed=bool(passed),microbatch=chosen,accumulation=16//chosen if chosen else None,rows=rows)
    write_json(saved,result)
    return result


def configure():
    if sys.version_info[:2]!=(3,12) or Path(sys.prefix).resolve()!=ROOT/'.venv-transformer':
        raise RuntimeError('Use .venv-transformer Python 3.12')
    if DEVICE!='mps':logging.warning('Running on %s; reported results were produced on MPS and bitwise reproducibility across devices is not claimed',DEVICE)
    if audit.checkpoint_metadata()['revision']!=REVISION:raise RuntimeError('Wrong pinned revision')
    MODELS.mkdir(parents=True,exist_ok=True);FIGURES.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(filename=LOG,level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    for handler in logging.getLogger().handlers:handler.addFilter(MemoryLogFilter())
    if os.environ.get('DISTILROBERTA_RELEASE_CACHE_EACH_STEP')=='1':
        from torch.optim.optimizer import register_optimizer_step_post_hook
        global _CACHE_HOOK
        _CACHE_HOOK=register_optimizer_step_post_hook(release_step_cache)
    hf_logging.disable_progress_bar();hf_logging.set_verbosity_error()
    torch.set_num_threads(4)
    cohorts,inputs=audit.load_assignments()
    prior=json.loads(audit.REPORT.with_suffix('.json').read_text())
    if inputs!=prior['inputs']:raise ValueError('Cohort/split fingerprints changed')
    return cohorts,inputs


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['pilot','all'],default='pilot')
    args=parser.parse_args();cohorts,inputs=configure()
    pilot=memory_pilot(cohorts,inputs)
    if not pilot['passed']:raise RuntimeError('Phase A microbatch 1 failed; Phase B stopped')
    if args.phase=='all':run_experiment(cohorts,inputs,pilot)




def uid_hash(uids):
    return hashlib.sha256(json.dumps(sorted(map(str,uids)),ensure_ascii=False).encode()).hexdigest()


def read_labels(dataset,uids):
    d=pd.read_csv(ROOT/f'data/processed/{PREFIXES[dataset]}_text_modeling_ready.csv',
        usecols=['content_uid','binary_analysis_target','binary_analysis_label']).set_index('content_uid').loc[list(uids)]
    if not d.binary_analysis_target.eq(d.binary_analysis_label.eq('Fake').astype(int)).all():
        raise ValueError('Incorrect label coding')
    return d.binary_analysis_target.to_numpy(dtype=int)


def metrics(y,p):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float);pred=p>=.5
    if not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():raise ValueError('Invalid probability')
    tn=int(((y==0)&~pred).sum());fp=int(((y==0)&pred).sum())
    fn=int(((y==1)&~pred).sum());tp=int(((y==1)&pred).sum())
    def div(a,b):return a/b if b else 0.
    fake_f1=div(2*tp,2*tp+fp+fn);real_f1=div(2*tn,2*tn+fp+fn)
    # Group tied scores, matching empirical ROC and non-interpolated AP definitions.
    order=np.argsort(-p,kind='stable');ys=y[order];ps=p[order]
    ends=np.r_[np.flatnonzero(np.diff(ps)!=0),len(ps)-1]
    tps=np.cumsum(ys)[ends];fps=1+ends-tps
    positives=int(y.sum());negatives=len(y)-positives
    auc=float(np.trapezoid(np.r_[0,tps]/positives,np.r_[0,fps]/negatives)) if positives and negatives else float('nan')
    ap=float(np.sum(np.diff(np.r_[0,tps])/positives*(tps/(ends+1)))) if positives else float('nan')
    return dict(N=len(y),fake_N=positives,real_N=negatives,accuracy=(tn+tp)/len(y),
        balanced_accuracy=(div(tp,tp+fn)+div(tn,tn+fp))/2,macro_f1=(fake_f1+real_f1)/2,
        fake_precision=div(tp,tp+fp),fake_recall=div(tp,tp+fn),fake_f1=fake_f1,real_f1=real_f1,
        roc_auc=auc,average_precision=ap,tn=tn,fp=fp,fn=fn,tp=tp,threshold=.5,
        actual_fake_prevalence=float(y.mean()),predicted_fake_prevalence=float(pred.mean()))


def predict(model,features,uids,microbatch):
    model.eval();probs=[]
    with torch.inference_mode():
        for start in range(0,len(uids),microbatch):
            batch=batch_tensor(features,uids[start:start+microbatch])
            logits=model(**batch).logits
            probs.extend(torch.softmax(logits,dim=-1)[:,1].cpu().tolist())
    if len(probs)!=len(uids) or not np.isfinite(probs).all():raise ValueError('Prediction coverage/nonfinite failure')
    return probs


def to_cpu(value):
    if torch.is_tensor(value):return value.detach().cpu()
    if isinstance(value,dict):return {k:to_cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [to_cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(to_cpu(v) for v in value)
    return value


def optimizer_for(model,lr):
    decay=[];no_decay=[]
    for name,p in model.named_parameters():
        (no_decay if name.endswith('bias') or 'LayerNorm.weight' in name else decay).append(p)
    return torch.optim.AdamW([{'params':decay,'weight_decay':.01},{'params':no_decay,'weight_decay':0.}],
        lr=lr,betas=(.9,.999),eps=1e-8,foreach=False,fused=False)


def save_checkpoint(path,model,optimizer,epoch,fp):
    data=dict(model=to_cpu(model.state_dict()),optimizer=to_cpu(optimizer.state_dict()),epoch=epoch,fingerprint=fp,
        torch_rng=torch.get_rng_state(),mps_rng=(torch.mps.get_rng_state() if DEVICE=='mps' else None))
    temp=path.with_suffix('.tmp');torch.save(data,temp);temp.replace(path)
    del data


def refresh_manifest():
    rows=[]
    for path in sorted(MODELS.glob('fits/*/epoch-*.json')):
        row=json.loads(path.read_text())
        spec=json.loads((path.parent/'fit_spec.json').read_text())
        row.update(microbatch=spec['microbatch'],gradient_accumulation=spec['accumulation'],
            fit_total_epochs=spec['epochs'],seed=spec['settings']['seed'],
            optimizer=spec['settings']['optimizer'],weight_decay=spec['settings']['weight_decay'],
            scheduler=spec['settings']['scheduler'],warmup_steps=spec['settings']['warmup_steps'],
            gradient_clip_norm=spec['settings']['gradient_clip_norm'],precision=spec['settings']['precision'])
        rows.append(row)
    if rows:
        if len({r['fingerprint'] for r in rows})!=len(rows):raise ValueError('Nonunique epoch fit fingerprint')
        write_table('fit_manifest',rows)
    return rows


def fit(dataset,train_uids,features,lr,epochs,max_length,role,fold,signature,pilot,val_uids=None):
    # The function accepts exactly one source's features and source-training UIDs.
    if not set(train_uids)<=set(features):raise ValueError('Source UID coverage failed')
    val_uids=val_uids or []
    if set(train_uids)&set(val_uids):raise ValueError('Training/validation overlap')
    y=read_labels(dataset,train_uids);weights=weights_from_training(y)
    spec=dict(dataset=dataset,role=role,fold=fold,learning_rate=lr,epochs=epochs,max_length=max_length,
        train_uid_sha256=uid_hash(train_uids),validation_uid_sha256=uid_hash(val_uids),
        training_label_sha256=digest(y.tolist()),settings=SETTINGS,signature=signature,
        microbatch=pilot['microbatch'],accumulation=pilot['accumulation'])
    fp=digest(spec);folder=MODELS/'fits'/fp;folder.mkdir(parents=True,exist_ok=True)
    spec_path=folder/'fit_spec.json'
    if spec_path.exists() and json.loads(spec_path.read_text())!=spec:raise ValueError('Fit specification mismatch')
    write_json(spec_path,spec)
    completed=[]
    for epoch in range(1,epochs+1):
        path=folder/f'epoch-{epoch}.json'
        if path.exists():
            row=json.loads(path.read_text());ckpt=folder/f'epoch-{epoch}.pt'
            if row['fingerprint']!=digest(dict(fit=fp,epoch=epoch)) or not ckpt.exists():raise ValueError('Broken completed checkpoint')
            completed.append(row)
        else:break
    if len(completed)==epochs:
        logging.info('Reuse completed fit %s %s fold=%s lr=%g cap=%s',dataset,role,fold,lr,max_length)
        return folder,completed
    model=new_model();optimizer=optimizer_for(model,lr)
    if completed:
        state=torch.load(folder/f'epoch-{len(completed)}.pt',map_location='cpu',weights_only=True)
        if state['fingerprint']!=fp:raise ValueError('Resume fingerprint mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng']);(torch.mps.set_rng_state(state['mps_rng']) if DEVICE=='mps' and state.get('mps_rng') is not None else None);del state
    weight_tensor=torch.tensor(weights,dtype=torch.float32,device=DEVICE)
    started=time.perf_counter()
    for epoch in range(len(completed)+1,epochs+1):
        model.train();order=np.random.default_rng(SEED+epoch).permutation(len(train_uids))
        weighted_numerator=0.;denominator=0.;updates=0;grad_max=0.;epoch_start=time.perf_counter()
        for window_start in range(0,len(order),16):
            window=order[window_start:window_start+16];normalizer=float(weights[y[window]].sum())
            optimizer.zero_grad(set_to_none=True)
            for start in range(0,len(window),pilot['microbatch']):
                indices=window[start:start+pilot['microbatch']]
                batch=batch_tensor(features,[train_uids[i] for i in indices])
                target=torch.tensor(y[indices],device=DEVICE)
                logits=model(**batch).logits
                loss_sum=torch.nn.functional.cross_entropy(logits,target,weight=weight_tensor,reduction='sum')
                if not torch.isfinite(loss_sum).item():raise RuntimeError('Nonfinite training loss')
                (loss_sum/normalizer).backward()
                weighted_numerator+=float(loss_sum.detach().item())
                del logits,loss_sum,batch,target
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=False,error_if_nonfinite=True)
            grad_max=max(grad_max,float(norm.item()));optimizer.step();updates+=1;denominator+=normalizer
            if updates%50==0:logging.info('%s %s fold=%s lr=%g cap=%s epoch=%s updates=%s/%s',
                dataset,role,fold,lr,max_length,epoch,updates,math.ceil(len(train_uids)/16))
        optimizer.zero_grad(set_to_none=True);(torch.mps.synchronize() if DEVICE=='mps' else None)
        val_metrics={}
        if val_uids and epoch in (2,3):
            # Validation labels are accessed for metrics after model predictions exist.
            probabilities=predict(model,features,val_uids,pilot['microbatch'])
            val_metrics=metrics(read_labels(dataset,val_uids),probabilities)
        row=dict(fingerprint=digest(dict(fit=fp,epoch=epoch)),fit_fingerprint=fp,dataset=dataset,role=role,fold=fold,
            learning_rate=lr,max_length=max_length,epoch=epoch,completion_state='completed',
            train_N=len(train_uids),train_uid_sha256=uid_hash(train_uids),validation_uid_sha256=uid_hash(val_uids),
            class_weight_real=float(weights[0]),class_weight_fake=float(weights[1]),
            weighted_training_loss=weighted_numerator/denominator,optimizer_updates=updates,gradient_norm_max=grad_max,
            epoch_seconds=time.perf_counter()-epoch_start,holdout_used_for_selection=False,
            checkpoint_revision=REVISION,input_fingerprint=digest(signature['inputs']),
            split_fingerprint=digest({k:v for k,v in signature['inputs'].items() if '/splits/' in k}),
            checkpoint=str((folder/f'epoch-{epoch}.pt').relative_to(ROOT)),
            validation_macro_f1=val_metrics.get('macro_f1'),validation_metrics=val_metrics)
        save_checkpoint(folder/f'epoch-{epoch}.pt',model,optimizer,epoch,fp)
        write_json(folder/f'epoch-{epoch}.json',row);completed.append(row);refresh_manifest()
        logging.info('Completed %s %s fold=%s lr=%g cap=%s epoch=%s loss=%.4f val_F1=%s elapsed=%.1fs',
            dataset,role,fold,lr,max_length,epoch,row['weighted_training_loss'],row['validation_macro_f1'],row['epoch_seconds'])
    del optimizer,model,weight_tensor;clear_device()
    logging.info('Fit complete %s elapsed %.1fs',fp[:12],time.perf_counter()-started)
    return folder,completed


def select_configs(cv):
    selected=[]
    for dataset in PREFIXES:
        rows=[]
        for (lr,epoch),part in cv[cv.dataset.eq(dataset)].groupby(['learning_rate','epoch']):
            if set(part.fold)!={0,1,2,3,4}:raise ValueError('Incomplete grouped CV')
            rows.append(dict(dataset=dataset,learning_rate=float(lr),epochs=int(epoch),
                macro_f1_mean=float(part.macro_f1.mean()),macro_f1_sd=float(part.macro_f1.std(ddof=1))))
        best=max(r['macro_f1_mean'] for r in rows)
        tied=[r for r in rows if r['macro_f1_mean']>=best-.001]
        winner=min(tied,key=lambda r:(r['macro_f1_sd'],r['epochs'],r['learning_rate']))
        winner.update(selection='training-pool grouped CV only',max_length=512,locked=True)
        selected.append(winner)
    return selected


def frozen_prediction(folder,epoch,features,assignment,microbatch,identity,lock_hash):
    path=folder/f'predictions-{identity["direction"]}-unlabeled.csv'
    meta_path=path.with_suffix('.json')
    expected=dict(identity=identity,config_lock_sha256=lock_hash,target_uid_sha256=uid_hash(assignment.content_uid),
        model_checkpoint=str((folder/f'epoch-{epoch}.pt').relative_to(ROOT)))
    if meta_path.exists():
        meta=json.loads(meta_path.read_text())
        if meta['expected']!=expected or audit.sha(path)!=meta['prediction_sha256']:raise ValueError('Frozen predictions mismatch')
        return path
    state=torch.load(folder/f'epoch-{epoch}.pt',map_location='cpu',weights_only=True)
    model=new_model();model.load_state_dict(state['model']);del state
    probabilities=predict(model,features,assignment.content_uid.tolist(),microbatch)
    del model;clear_device()
    p=assignment[['content_uid','cv_fold','holdout_assignment']].copy()
    for key,value in identity.items():p[key]=value
    p['fake_probability']=probabilities;p['predicted_label']=(p.fake_probability>=.5).astype(int)
    if any('label' in c and c!='predicted_label' for c in p):raise ValueError('Observed labels in prediction freeze')
    p.to_csv(path,index=False)
    write_json(meta_path,dict(expected=expected,prediction_sha256=audit.sha(path),frozen_at=time.time(),observed_labels_joined=False))
    logging.info('Frozen unlabeled predictions %s cap=%s N=%s',identity['direction'],identity['max_length'],len(p))
    return path


def score_frozen(paths):
    results=[];predictions=[]
    for path in paths:
        meta=json.loads(path.with_suffix('.json').read_text())
        if audit.sha(path)!=meta['prediction_sha256']:raise ValueError('Prediction freeze broken')
        p=pd.read_csv(path)
        if 'true_label' in p:raise ValueError('Premature observed labels')
        p['true_label']=read_labels(p.target_dataset.iloc[0],p.content_uid)
        identity={k:p[k].iloc[0] for k in ['evaluation','direction','source_dataset','target_dataset','max_length']}
        scopes=[('same_target_holdout',p)] if identity['evaluation']=='within' else [
            ('all_eligible',p),('same_target_holdout',p[p.cv_fold.eq(-1)])]
        for scope,part in scopes:
            results.append(dict(**identity,scope=scope,**metrics(part.true_label,part.fake_probability),
                evaluation_uid_sha256=uid_hash(part.content_uid)))
        predictions.append(p)
    return results,pd.concat(predictions,ignore_index=True)


def prior_output_snapshot():
    return {str(p.relative_to(ROOT)):[p.stat().st_size,p.stat().st_mtime_ns]
        for p in (ROOT/'outputs').rglob('*') if p.is_file() and 'distilroberta' not in str(p.relative_to(ROOT))}


def old_environment_fingerprint():
    # Read-only metadata fingerprint; never execute/install into the classical env.
    paths=[ROOT/'.venv/pyvenv.cfg',*sorted((ROOT/'.venv/lib').glob('python*/site-packages/*.dist-info/METADATA'))]
    return digest({str(p.relative_to(ROOT)):audit.sha(p) for p in paths})


def start_signature(inputs):
    comparator=TABLES/'tfidf_linear_svm_vs_logistic.csv'
    core=[seed_all,new_model,pack_features,batch_tensor,weights_from_training,read_labels,metrics,predict,
        optimizer_for,save_checkpoint,fit,select_configs,frozen_prediction,score_frozen,run_experiment]
    signature=dict(inputs=inputs,training_code_sha256=digest({f.__name__:inspect.getsource(f) for f in core}),packer_sha256=audit.sha(audit.__file__),
        requirements_sha256=audit.sha(ROOT/'requirements-transformer.txt'),classical_metrics_sha256=audit.sha(comparator),
        settings=SETTINGS,torch_version=torch.__version__)
    path=MODELS/'experiment_state.json'
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['signature']!=signature:raise ValueError('Experiment fingerprint changed; refuse silent reuse')
    else:
        saved=dict(signature=signature,started_at=time.time(),prior_outputs=prior_output_snapshot(),
            original_environment_fingerprint=old_environment_fingerprint())
        write_json(path,saved)
    if prior_output_snapshot()!=saved['prior_outputs']:raise ValueError('Prior outputs changed')
    return signature,saved


def run_experiment(cohorts,inputs,pilot):
    signature,state=start_signature(inputs)
    final_meta=ROOT/'outputs/reports/distilroberta_text_experiment.json'
    if final_meta.exists():
        done=json.loads(final_meta.read_text())
        if done.get('status')=='complete' and done['signature']==signature:
            logging.info('Completed experiment fingerprint matches; nothing rerun');return
    features={d:pack_features(d,c,512) for d,c in cohorts.items()}
    cv=[]
    for dataset,cohort in cohorts.items():
        pool=cohort[cohort.cv_fold.ge(0)]
        for fold in range(5):
            train=pool[pool.cv_fold.ne(fold)];validation=pool[pool.cv_fold.eq(fold)]
            if set(train.leakage_group)&set(validation.leakage_group):raise ValueError('Group leakage')
            for lr in (2e-5,5e-5):
                _,rows=fit(dataset,train.content_uid.tolist(),features[dataset],lr,3,512,'cv',fold,signature,pilot,
                    validation.content_uid.tolist())
                for row in rows:
                    if row['epoch'] in (2,3):cv.append(dict(dataset=dataset,fold=fold,learning_rate=lr,epoch=row['epoch'],
                        macro_f1=row['validation_macro_f1'],holdout_used=False,fit_fingerprint=row['fit_fingerprint']))
                write_table('cv_results',cv)
    selection=select_configs(pd.DataFrame(cv));lock_path=MODELS/'locked_configurations.json'
    lock=dict(signature=signature,selected=selection,locked_before_holdout_and_transfer=True)
    if lock_path.exists() and json.loads(lock_path.read_text())!=lock:raise ValueError('Locked configurations changed')
    write_json(lock_path,lock);write_table('model_selection',selection);lock_hash=audit.sha(lock_path)
    logging.info('Both source configurations LOCKED: %s',json.dumps(selection))
    all_results=[];all_predictions=[]
    for cap in (512,256):
        if cap==256:
            # This loop is reached only after every 512 fit and evaluation is complete.
            write_json(MODELS/'primary_complete.json',dict(config_lock_sha256=lock_hash,completed_at=time.time()))
            features={d:pack_features(d,c,256) for d,c in cohorts.items()}
        paths=[]
        for config in selection:
            dataset=config['dataset'];cohort=cohorts[dataset]
            train=cohort[cohort.cv_fold.ge(0)];target=cohort[cohort.cv_fold.eq(-1)]
            folder,_=fit(dataset,train.content_uid.tolist(),features[dataset],config['learning_rate'],config['epochs'],
                cap,'within_refit',-1,signature,pilot)
            identity=dict(evaluation='within',direction='within_'+PREFIXES[dataset],source_dataset=dataset,
                target_dataset=dataset,max_length=cap)
            paths.append(frozen_prediction(folder,config['epochs'],features[dataset],target,pilot['microbatch'],identity,lock_hash))
        # Both within predictions freeze before scoring the held-out labels.
        within,wp=score_frozen(paths);all_results.extend(within);all_predictions.append(wp)
        write_table('holdout_results',[r for r in all_results if r['evaluation']=='within'])
        paths=[]
        for direction,source,target in [('FH_to_CA','FakeHealth','CoAID'),('CA_to_FH','CoAID','FakeHealth')]:
            config=next(c for c in selection if c['dataset']==source)
            if audit.sha(lock_path)!=lock_hash:raise ValueError('Source config changed before transfer')
            folder,_=fit(source,cohorts[source].content_uid.tolist(),features[source],config['learning_rate'],
                config['epochs'],cap,'transfer_refit',-1,signature,pilot)
            identity=dict(evaluation='transfer',direction=direction,source_dataset=source,target_dataset=target,max_length=cap)
            paths.append(frozen_prediction(folder,config['epochs'],features[target],cohorts[target],pilot['microbatch'],identity,lock_hash))
        # Freeze both transfer directions before any transfer evaluation label join.
        transfer,tp=score_frozen(paths);all_results.extend(transfer);all_predictions.append(tp)
        write_table('transfer_results',[r for r in all_results if r['evaluation']=='transfer'])
        prediction_path=ROOT/'outputs/predictions/distilroberta_predictions.csv'
        prediction_path.parent.mkdir(parents=True,exist_ok=True)
        pd.concat(all_predictions,ignore_index=True).to_csv(prediction_path,index=False)
        logging.info('Cap %s fully evaluated',cap)
    finish_experiment(all_results,pd.concat(all_predictions,ignore_index=True),selection,signature,state,pilot)

def bar_chart(path,title,groups,series,values):
    colors=['#607d8b','#d28b35','#326ca8'];width=1000;height=430;left=75;bottom=320;plot_width=880
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',f'<text x="30" y="30" font-family="sans-serif" font-size="20">{html.escape(title)}</text>']
    for tick in np.linspace(0,1,6):
        y=bottom-240*tick
        parts.append(f'<line x1="{left}" y1="{y}" x2="955" y2="{y}" stroke="#ddd"/><text x="35" y="{y+5}" font-family="sans-serif" font-size="13">{tick:.1f}</text>')
    spacing=plot_width/len(groups);bar_width=min(45,spacing/(len(series)+1))
    for i,group in enumerate(groups):
        for j,series_name in enumerate(series):
            value=values[i][j];x=left+i*spacing+15+j*bar_width;y=bottom-240*value
            parts.append(f'<rect x="{x}" y="{y}" width="{bar_width-3}" height="{240*value}" fill="{colors[j]}"/><text x="{x}" y="{y-5}" font-family="sans-serif" font-size="11">{value:.3f}</text>')
        parts.append(f'<text x="{left+i*spacing+15}" y="345" font-family="sans-serif" font-size="12">{html.escape(group)}</text>')
    for j,name in enumerate(series):
        parts.append(f'<rect x="{75+j*240}" y="380" width="15" height="15" fill="{colors[j]}"/><text x="{96+j*240}" y="394" font-family="sans-serif" font-size="14">{html.escape(name)}</text>')
    parts.append('</svg>');path.write_text('\n'.join(parts))


def figures(comparison,results):
    for evaluation in ['within','transfer']:
        d=comparison[comparison.evaluation.eq(evaluation)&comparison.metric.eq('macro_f1')]
        groups=[r.target_dataset if evaluation=='within' else r.direction+' / '+r.scope.replace('same_target_holdout','holdout').replace('all_eligible','all') for r in d.itertuples()]
        bar_chart(FIGURES/f'{evaluation}_model_comparison.svg',f'{evaluation.capitalize()}: macro F1 (512-token transformer)',
            groups,['Logistic regression','Linear SVM','DistilRoBERTa'],d[['logistic','linear_svm','distilroberta']].values.tolist())
    primary=results[results.max_length.eq(512)&results.evaluation.eq('transfer')]
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="660"><rect width="100%" height="100%" fill="white"/>',
        '<text x="25" y="30" font-family="sans-serif" font-size="20">DistilRoBERTa transfer: confusion matrices, threshold 0.5</text>']
    for i,r in enumerate(primary.itertuples()):
        x=40+(i%2)*500;y=75+(i//2)*295
        parts.append(f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="16">{r.direction}: {r.scope}</text>')
        for a,label in enumerate(['Real','Fake']):
            parts.append(f'<text x="{x+125+a*130}" y="{y+30}" font-family="sans-serif" font-size="13">Pred. {label}</text>')
            parts.append(f'<text x="{x}" y="{y+90+a*90}" font-family="sans-serif" font-size="13">Actual {label}</text>')
        for j,value in enumerate([r.tn,r.fp,r.fn,r.tp]):
            xx=x+105+(j%2)*130;yy=y+45+(j//2)*90
            parts.append(f'<rect x="{xx}" y="{yy}" width="125" height="85" fill="#d9e7f2"/><text x="{xx+45}" y="{yy+49}" font-family="sans-serif" font-size="22">{value}</text>')
    parts.append('</svg>');(FIGURES/'transfer_confusion_matrices.svg').write_text('\n'.join(parts))
    primary=results[results.max_length.eq(512)&results.scope.eq('same_target_holdout')]
    shorter=results[results.max_length.eq(256)&results.scope.eq('same_target_holdout')].set_index('direction')
    bar_chart(FIGURES/'512_vs_256_sensitivity.svg','Token-budget sensitivity: macro F1 on identical holdout rows',
        primary.direction.tolist(),['512 tokens','256 tokens'],[[r.macro_f1,shorter.loc[r.direction,'macro_f1']] for r in primary.itertuples()])


def finish_experiment(rows,predictions,selection,signature,state,pilot):
    results=pd.DataFrame(rows);primary=results[results.max_length.eq(512)]
    classical=pd.read_csv(TABLES/'tfidf_linear_svm_vs_logistic.csv')
    comparison=[];degradation=[];sensitivity=[]
    for r in primary.itertuples():
        for metric in METRICS:
            match=classical[classical.evaluation.eq(r.evaluation)&classical.direction.eq(r.direction)&
                classical.scope.eq(r.scope)&classical.metric.eq(metric)]
            if len(match)!=1:raise ValueError('Missing/ambiguous classical comparator')
            old=match.iloc[0]
            if old.N!=r.N or old.evaluation_uid_sha256!=r.evaluation_uid_sha256:
                raise ValueError('Comparator target rows differ')
            comparison.append(dict(evaluation=r.evaluation,direction=r.direction,target_dataset=r.target_dataset,
                scope=r.scope,metric=metric,N=r.N,logistic=float(old.logistic),linear_svm=float(old.linear_svm),
                distilroberta=getattr(r,metric),transformer_minus_lr=getattr(r,metric)-old.logistic,
                transformer_minus_svm=getattr(r,metric)-old.linear_svm,evaluation_uid_sha256=r.evaluation_uid_sha256))
    for r in results[results.evaluation.eq('transfer')&results.scope.eq('same_target_holdout')].itertuples():
        w=results[results.evaluation.eq('within')&results.target_dataset.eq(r.target_dataset)&results.max_length.eq(r.max_length)].iloc[0]
        if w.evaluation_uid_sha256!=r.evaluation_uid_sha256:raise ValueError('Within/transfer target rows differ')
        for metric in METRICS:
            degradation.append(dict(direction=r.direction,target_dataset=r.target_dataset,max_length=r.max_length,
                metric=metric,within_target=float(w[metric]),cross_context=getattr(r,metric),
                transfer_minus_within=getattr(r,metric)-w[metric],N=r.N,evaluation_uid_sha256=r.evaluation_uid_sha256))
    for r in primary.itertuples():
        other=results[results.max_length.eq(256)&results.direction.eq(r.direction)&results.scope.eq(r.scope)].iloc[0]
        if other.evaluation_uid_sha256!=r.evaluation_uid_sha256:raise ValueError('Sensitivity cohorts differ')
        for metric in METRICS:
            sensitivity.append(dict(evaluation=r.evaluation,direction=r.direction,scope=r.scope,metric=metric,
                primary_512=getattr(r,metric),sensitivity_256=float(other[metric]),
                delta_256_minus_512=other[metric]-getattr(r,metric),hyperparameters_reused=True))
    write_table('vs_classical',comparison);write_table('transfer_degradation',degradation);write_table('256_sensitivity',sensitivity)
    comparison=pd.DataFrame(comparison);figures(comparison,results)
    cv_summary=(pd.read_csv(TABLES/'distilroberta_cv_results.csv')
        .groupby(['dataset','learning_rate','epoch']).macro_f1.agg(['mean','std']).reset_index()
        .rename(columns={'mean':'macro_f1_mean','std':'macro_f1_sample_sd'}))
    manifest=refresh_manifest()
    if any(audit.sha(ROOT/p)!=h for p,h in signature['inputs'].items()):raise ValueError('Inputs changed')
    if old_environment_fingerprint()!=state['original_environment_fingerprint']:raise ValueError('Classical environment changed')
    if prior_output_snapshot()!=state['prior_outputs']:raise ValueError('Prior project outputs changed')
    elapsed=time.time()-state['started_at'];fit_seconds=sum(r['epoch_seconds'] for r in manifest)
    within=primary[primary.evaluation.eq('within')]
    transfer=primary[primary.evaluation.eq('transfer')]
    drops=pd.DataFrame(degradation);sens=pd.DataFrame(sensitivity)
    gap_comparison=[]
    for r in comparison[comparison.evaluation.eq('transfer')&comparison.scope.eq('same_target_holdout')].itertuples():
        baseline=comparison[comparison.evaluation.eq('within')&comparison.target_dataset.eq(r.target_dataset)&comparison.metric.eq(r.metric)].iloc[0]
        gap_comparison.append(dict(direction=r.direction,metric=r.metric,
            logistic=r.logistic-baseline.logistic,linear_svm=r.linear_svm-baseline.linear_svm,
            distilroberta=r.distilroberta-baseline.distilroberta))
    findings=[]
    for r in transfer[transfer.scope.eq('same_target_holdout')].itertuples():
        c=comparison[comparison.evaluation.eq('transfer')&comparison.direction.eq(r.direction)&comparison.scope.eq('same_target_holdout')&comparison.metric.eq('macro_f1')].iloc[0]
        w=classical[classical.evaluation.eq('within')&classical.target_dataset.eq(r.target_dataset)&classical.metric.eq('macro_f1')].iloc[0]
        drop=drops[drops.max_length.eq(512)&drops.direction.eq(r.direction)&drops.metric.eq('macro_f1')].iloc[0]
        findings.append(f'{r.direction}, identical target holdout: transformer macro F1 {r.macro_f1:.4f}, '
            f'LR {c.logistic:.4f}, SVM {c.linear_svm:.4f}. Transformer transfer-minus-within {drop.transfer_minus_within:+.4f}; '
            f'LR {c.logistic-w.logistic:+.4f}; SVM {c.linear_svm-w.linear_svm:+.4f}. '
            f'Fake predicted prevalence {r.predicted_fake_prevalence:.2%} versus actual {r.actual_fake_prevalence:.2%}.')
    improved=[r for r in comparison.itertuples() if r.evaluation=='transfer' and r.scope=='same_target_holdout' and r.metric=='macro_f1' and r.transformer_minus_lr>0 and r.transformer_minus_svm>0]
    summary=f'Primary 512-token transformer transfer macro F1 exceeds both frozen linear comparators in {len(improved)} of 2 directions. '
    summary+='This is descriptive, single-seed evidence; it does not establish statistical superiority or isolate contextual pretraining from token-budget effects. '
    summary+='Read ROC-AUC, AP, Fake F1 and prevalence alongside macro F1: changes in threshold behavior can differ from ranking quality.'
    ranking=comparison[comparison.evaluation.eq('transfer')&comparison.scope.eq('same_target_holdout')&
        comparison.metric.isin(['roc_auc','average_precision'])]
    ranking_interpretation=('ROC-AUC and AP are lower than both linear comparators in both transfer directions. '
        'Together with the absolute transfer scores, this provides no consistent evidence that contextual pretraining '
        'materially reduced cross-context failure. The modest CA-to-FH macro-F1 gain coexists with worse ranking quality; '
        'its smaller macro-F1 degradation is also partly a consequence of lower FakeHealth within-context macro F1. '
        'The original directional prevalence pattern persists: FH-to-CA overpredicts Fake and CA-to-FH underpredicts Fake.'
        if len(ranking)==4 and ranking.transformer_minus_lr.lt(0).all() and ranking.transformer_minus_svm.lt(0).all()
        else 'The direction-specific metric comparisons below determine whether any transfer gains are consistent across threshold and ranking metrics.')
    sensitivity_findings=[]
    for r in results[results.max_length.eq(256)&results.evaluation.eq('transfer')&results.scope.eq('same_target_holdout')].itertuples():
        change=sens[sens.direction.eq(r.direction)&sens.scope.eq(r.scope)&sens.metric.eq('macro_f1')].iloc[0]
        sensitivity_findings.append(f'{r.direction}, identical target holdout at 256: macro F1 {r.macro_f1:.4f} '
            f'({change.delta_256_minus_512:+.4f} versus 512), ROC-AUC {r.roc_auc:.4f}, AP {r.average_precision:.4f}.')
    report=['# DistilRoBERTa within-context and cross-context experiment','',summary,'',ranking_interpretation,'',*findings,'',
        '## Protocol and memory gate','',
        f'Pinned checkpoint `{audit.CHECKPOINT}` at `{REVISION}`. Phase A passed; microbatch {pilot["microbatch"]}, '
        f'accumulation {pilot["accumulation"]}, effective batch 16. Both pilot tests used synthetic labels and '
        'weighted cross-entropy; parameter hashes were unchanged and no optimizer step occurred. Pilot models were released. '
        'Reported MPS memory is sampled after backward, not a transient peak and excludes optimizer state.','',
        audit.markdown_table(pilot['rows'],['microbatch','forward_succeeded','backward_succeeded','parameters_unchanged',
            'sampled_mps_allocated_bytes','sampled_mps_driver_bytes','elapsed_seconds']),'',
        'FP32 MPS throughout; no mixed precision, oversampling, early stopping or threshold tuning. '+json.dumps(SETTINGS,sort_keys=True),
        'A constant learning-rate scheduler with zero warmup makes the epoch-2 checkpoint schedule identical to a two-epoch refit. '
        'AdamW uses decay 0.01 except biases/LayerNorm weights. Gradient norm is clipped to 1.0. '
        'Weighted CE gradients are normalized by the sum of source-training class weights within each effective batch, '
        'including a final partial batch. Class weights are recalculated from that fit’s training labels only.','',
        '## Cohorts, predictors and selection','',
        'FakeHealth HealthStory N=1,647; CoAID Specification A N=3,577. Fake=1, Real=0. Existing five grouped folds '
        'and reserved holdouts are unchanged. Only title_text and body_text enter the encoder. Full title is preserved, '
        'four official special tokens are reserved, and a body prefix fills the remaining 512-token budget. '
        'Dynamic padding is confined to each microbatch. No source, ID, engagement, rating or split metadata is a predictor.','',
        'Each learning rate was fitted once for three epochs in each fold, with validation at epochs 2 and 3. '
        'The 0.001 mean-macro-F1 tie set is ordered by lower sample SD, fewer epochs, then lower learning rate. '
        'Both configurations were locked before holdout evaluation and transfer.','',
        audit.markdown_table(selection,['dataset','learning_rate','epochs','macro_f1_mean','macro_f1_sd']),'',
        'All logical CV candidates (five existing grouped folds each):','',
        audit.markdown_table(cv_summary.to_dict('records'),
            ['dataset','learning_rate','epoch','macro_f1_mean','macro_f1_sample_sd']),'',
        '## Primary within-context holdout','',
        audit.markdown_table(within.to_dict('records'),['target_dataset','N','accuracy','balanced_accuracy','macro_f1',
            'fake_precision','fake_recall','fake_f1','real_f1','roc_auc','average_precision',
            'actual_fake_prevalence','predicted_fake_prevalence','tn','fp','fn','tp']),'',
        '## Primary transfer','',
        'Each direction starts from the same pretrained checkpoint and refits the locked source configuration on all '
        'eligible source rows, including its former holdout. Both unlabeled transfer prediction files are frozen before '
        'either transfer scoring join. A dataset’s labels serve training only in its source role; reverse-direction '
        'results cannot alter the locked model or its predictions. All eligible targets and the exact classical-model '
        'target holdouts are scored. Fake probability >=0.5 is fixed. No target tuning or calibration occurs.','',
        audit.markdown_table(transfer.to_dict('records'),['direction','scope','N','macro_f1','balanced_accuracy','fake_f1',
            'roc_auc','average_precision','actual_fake_prevalence','predicted_fake_prevalence','tn','fp','fn','tp']),'',
        '## Transfer degradation and frozen linear comparators','',
        audit.markdown_table(drops[drops.max_length.eq(512)].to_dict('records'),
            ['direction','metric','within_target','cross_context','transfer_minus_within']),'',
        audit.markdown_table(comparison.to_dict('records'),['evaluation','direction','scope','metric','logistic','linear_svm','distilroberta']),'',
        'Transfer minus within-target performance, for each model on identical target holdouts (negative means degradation):','',
        audit.markdown_table(gap_comparison,['direction','metric','logistic','linear_svm','distilroberta']),'',
        '## 256-token sensitivity','',
        *sensitivity_findings,'',
        'Started after primary 512 fits and evaluation completed. Each source reuses its 512-selected learning rate '
        'and epoch count; identical rows/splits, weighted loss and packing. No retuning. This tests available body-prefix '
        'dependence; it does not equalize retained information.','',
        'CoAID inputs fit below both caps, so its within-context refits retain identical text. '
        'Any small CoAID within-context difference is therefore refit/numerical variability, not a truncation effect. '
        'Seeding MPS does not guarantee bitwise-identical fits. With one refit per cap, sensitivity differences '
        'cannot fully separate text-budget dependence from refit variability. In CA-to-FH transfer the target '
        'FakeHealth prefix changes; in FH-to-CA transfer the source FakeHealth prefix changes.','',
        audit.markdown_table(sens.to_dict('records'),['evaluation','direction','scope','metric','primary_512','sensitivity_256','delta_256_minus_512']),'',
        '## Interpretation and limitations','',
        'At 512, 71.22% of FakeHealth documents truncate, versus 0% of CoAID. At 256 the corresponding proportions '
        'are 87.86% and 0%. Comparison with full-text TF-IDF combines a model-family change with a representation-budget '
        'change. Poor transfer cannot be attributed to truncation alone: source, topic, collection and label-definition '
        'differences remain plausible contributors. Within-dataset performance and cross-context generalization answer '
        'different questions. These findings imply no causal COVID effect, universal misinformation cues or independence '
        'of the transformer from the data-generating context. Minor within-domain ranks are not the scientific focus.','',
        'A smaller transfer-minus-within deficit can also arise from weaker within-target performance. '
        'It should therefore be interpreted together with absolute transfer macro F1, Fake F1, ROC-AUC and AP, '
        'rather than treated alone as evidence that contextual pretraining solved transfer failure.','',
        '## Runtime, checkpoints and validation','',
        f'Elapsed execution {elapsed/60:.1f} minutes; sum of epoch training/validation times {fit_seconds/60:.1f} minutes. '
        f'{len(manifest)} completed epoch checkpoints. Each includes model, optimizer and CPU/MPS RNG state. '
        'The manifest records unique fingerprints, input/split hashes, hyperparameters, weighted losses, clipping diagnostics '
        'and metrics. Completed fits are reused; interrupted fits resume from the last completed epoch. '
        'Finite losses/gradients were enforced. Three epochs is a fixed budget, not a claim of optimizer convergence. '
        'After the first 15 CV epochs, host memory pressure (about 4.8 GiB swap) slowed epochs from roughly 75 seconds '
        f'to as much as {max(r["epoch_seconds"] for r in manifest):.1f} seconds. Checkpoint resumes and worker recycling after completed fits were introduced. '
        'MPS allocator low/high watermarks were then set to 0.3/0.8; these are resource controls, with FP32, '
        'microbatch, optimizer and learning settings unchanged. An unfinished epoch may replay after interruption; '
        'After 19 saved CV epochs, unused MPS allocator cache was also released after each optimizer update. '
        'The first cleanup in each fit verifies that parameter version counters and MPS RNG state are unchanged. '
        'completed checkpoints are reused. See the [PyTorch allocator documentation]'
        '(https://docs.pytorch.org/docs/2.14/mps_environment_variables.html) and the progress log. '
        'Wall-clock runtime includes long overnight gaps between checkpoints despite much shorter epoch timer values. '
        'This suggests host sleep/suspension; it was not directly measured as a sleep event. '
        'A temporary caffeinate idle-sleep assertion was tied to the worker PID on September 22. '
        'Normal scheduling priority was restored without changing training calculations. '
        'Original `.venv` metadata, frozen cohorts/splits and all prior output file sizes/mtimes are unchanged. '
        'Reproducibility on MPS is seeded but bitwise reproducibility across devices/software is not asserted.','',
        'Automated validation results are recorded in the companion JSON and appended below.','',
        'Resume with the same fingerprints (completed fits are reused):','',
        '```bash',
        'PYTORCH_MPS_LOW_WATERMARK_RATIO=0.3 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8 '
        'DISTILROBERTA_RECYCLE_AFTER_FIT=1 DISTILROBERTA_RELEASE_CACHE_EACH_STEP=1 '
        '.venv-transformer/bin/python src/run_distilroberta_text_experiment.py --phase all',
        '```','',
        'Outputs: `outputs/tables/distilroberta_*.csv`, `outputs/predictions/distilroberta_predictions.csv`, '
        '`outputs/models/distilroberta/`, `outputs/figures/distilroberta/`, '
        '`logs/distilroberta_text_experiment.log`.','']
    report_path=ROOT/'outputs/reports/distilroberta_text_experiment.md';report_path.write_text('\n'.join(report))
    metadata=dict(status='pending_final_tests',signature=signature,elapsed_seconds=elapsed,epoch_seconds=fit_seconds,
        final_script_sha256=audit.sha(__file__),resource_policy=dict(recycle_after_fit=os.environ.get('DISTILROBERTA_RECYCLE_AFTER_FIT'),
            release_cache_each_step=os.environ.get('DISTILROBERTA_RELEASE_CACHE_EACH_STEP'),
            mps_low_watermark=os.environ.get('PYTORCH_MPS_LOW_WATERMARK_RATIO'),
            mps_high_watermark=os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO')),
        memory_pilot=pilot,selected=selection,fit_epoch_count=len(manifest),original_environment_unchanged=True,
        prior_outputs_unchanged=True,inputs_unchanged=True,prediction_rows=len(predictions),
        config_lock_sha256=audit.sha(MODELS/'locked_configurations.json'))
    meta_path=report_path.with_suffix('.json');write_json(meta_path,metadata)
    import unittest
    suite=unittest.defaultTestLoader.discover(str(ROOT/'src'),pattern='test_distilroberta_text_experiment.py')
    with LOG.open('a') as stream:result=unittest.TextTestRunner(stream=stream,verbosity=1).run(suite)
    metadata['tests']=dict(run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped))
    metadata['status']='complete' if result.wasSuccessful() and not result.skipped else 'validation_failed'
    with report_path.open('a') as f:f.write(f'\nFinal tests: {result.testsRun} run, {len(result.failures)} failures, {len(result.errors)} errors, {len(result.skipped)} skips.\n')
    metadata['outputs']={str(p.relative_to(ROOT)):audit.sha(p) for p in [*TABLES.glob('distilroberta_*.csv'),
        ROOT/'outputs/predictions/distilroberta_predictions.csv',report_path,*FIGURES.glob('*.svg')]}
    write_json(meta_path,metadata)
    if metadata['status']!='complete':raise RuntimeError('Final validation failed; inspect log')
    logging.info('EXPERIMENT COMPLETE; tests passed; no further model runs')


class MemoryLogFilter(logging.Filter):
    def filter(self,record):
        if str(record.msg).startswith(('Completed %s','%s %s fold=')):
            record.msg=record.getMessage()+(' MPS_alloc=%.2fGiB MPS_driver=%.2fGiB recommended=%.2fGiB' %
                ((torch.mps.current_allocated_memory() if DEVICE=='mps' else 0)/2**30,(torch.mps.driver_allocated_memory() if DEVICE=='mps' else 0)/2**30,
                 (torch.mps.recommended_max_memory() if DEVICE=='mps' else 0)/2**30))
            record.args=()
        return True


def release_step_cache(optimizer,args,kwargs):
    """Resource housekeeping only, after AdamW's completed parameter update."""
    (torch.mps.synchronize() if DEVICE=='mps' else None)
    first=not getattr(optimizer,'_distil_cache_verified',False)
    if first:
        versions=[p._version for group in optimizer.param_groups for p in group['params']]
        rng=(torch.mps.get_rng_state() if DEVICE=='mps' else None);before=(torch.mps.driver_allocated_memory() if DEVICE=='mps' else 0)
    (torch.mps.empty_cache() if DEVICE=='mps' else None)
    if first:
        if versions!=[p._version for group in optimizer.param_groups for p in group['params']]:
            raise RuntimeError('Cache cleanup modified parameters')
        if DEVICE=='mps' and not torch.equal(rng,torch.mps.get_rng_state()):raise RuntimeError('Cache cleanup modified RNG')
        optimizer._distil_cache_verified=True
        logging.info('Cache cleanup verified: unchanged parameter versions and MPS RNG; driver %.2f -> %.2f GiB',
            before/2**30,(torch.mps.driver_allocated_memory() if DEVICE=='mps' else 0)/2**30)


if __name__=='__main__':main()
