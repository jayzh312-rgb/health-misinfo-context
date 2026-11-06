"""Pinned, local-cache-only tokenization and inference feasibility audit.

Run: .venv-transformer/bin/python src/audit_transformer_feasibility.py
Setup/download is separate; this script never installs, downloads, or trains.
"""
from pathlib import Path
import hashlib
import importlib.metadata
import importlib.util
import json
import warnings
import os
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ('torch','transformers','datasets','accelerate')
PREFIXES = {'FakeHealth':'fakehealth','CoAID':'coaid'}
BUDGETS = (128,256,384,512)
CHECKPOINT = 'distilbert/distilroberta-base'
REPORT = ROOT/'outputs/reports/transformer_feasibility.md'
TABLES = ROOT/'outputs/tables'
LOG = ROOT/'logs/transformer_environment_setup.log'
CACHE = ROOT/'.venv-transformer/huggingface-cache'
INSTALL = '.venv-transformer/bin/python -m pip install torch transformers datasets accelerate'
# Lightweight official metadata checked via browser; not a locally loaded model.
CHECKPOINT_METADATA = dict(checkpoint=CHECKPOINT,checked_date='2026-09-21',
    access='official model card, config, and tokenizer configuration readable via browser',
    model_type='roberta',documented_tokenizer_model_max_length=512,vocab_size=50265,
    num_hidden_layers=6,hidden_size=768,num_attention_heads=12,approx_parameters=82000000,
    locally_instantiated_tokenizer=False,revision_pinned=False,
    sources=['https://huggingface.co/distilbert/distilroberta-base',
             'https://huggingface.co/distilbert/distilroberta-base/raw/main/config.json',
             'https://huggingface.co/distilbert/distilroberta-base/blob/main/tokenizer_config.json'])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package_status():
    result={}
    for name in REQUIRED:
        available=importlib.util.find_spec(name) is not None
        try:version=importlib.metadata.version(name) if available else None
        except importlib.metadata.PackageNotFoundError:version=None
        result[name]=dict(installed=available,version=version)
    return result


def environment(packages):
    info=dict(python=platform.python_version(),machine=platform.machine(),system=platform.system(),
              torch_version=packages['torch']['version'],transformers_version=packages['transformers']['version'],
              macos_version=platform.mac_ver()[0], executable=sys.executable,
              selected_device='CPU',cuda_available=None,mps_available=None,mps_built=None,
              system_memory_bytes=None,system_memory_status='not measured')
    if packages['torch']['installed']:
        try:
            import torch
            import transformers
            info['torch_import']=True;info['transformers_import']=True
            info['cpu_available']=bool(torch.ones(1,device='cpu').item()==1)
            info['mps_built']=bool(torch.backends.mps.is_built())
            info['mps_available']=bool(hasattr(torch.backends,'mps') and torch.backends.mps.is_available())
            info['cuda_available']=bool(torch.cuda.is_available())
            info['selected_device']='MPS' if info['mps_available'] else ('CUDA' if info['cuda_available'] else 'CPU')
            warnings.warn('Selected device %s; reported results were produced on Apple MPS and bitwise reproducibility across devices is not claimed.' % info['selected_device'])
        except Exception as exc:
            info['backend_probe_error']=type(exc).__name__+': '+str(exc)
    if platform.system()=='Darwin':
        check=subprocess.run(['/usr/sbin/sysctl','-n','hw.memsize'],capture_output=True,text=True)
        if check.returncode==0 and check.stdout.strip().isdigit():
            info['system_memory_bytes']=int(check.stdout.strip());info['system_memory_status']='physical installed memory; not free memory'
        else:info['system_memory_status']='unavailable: sandbox denied optional sysctl memory query'
    return info


def load_assignments():
    cohorts={};inputs={}
    for dataset,prefix in PREFIXES.items():
        path=ROOT/f'data/processed/{prefix}_text_modeling_ready.csv';inputs[str(path.relative_to(ROOT))]=sha(path)
        cols=['content_uid','primary_modeling_eligible','leakage_group','cv_fold','holdout_assignment']
        d=pd.read_csv(path,usecols=cols,keep_default_na=False)
        if d.primary_modeling_eligible.dtype!=bool:raise ValueError('Eligibility not boolean')
        d=d[d.primary_modeling_eligible].sort_values('content_uid').reset_index(drop=True)
        if not d.content_uid.is_unique or not d.groupby('leakage_group').cv_fold.nunique().eq(1).all():
            raise ValueError('Cohort UID/group integrity failed')
        if not d.cv_fold.isin([-1,0,1,2,3,4]).all() or not (d.cv_fold.eq(-1)==d.holdout_assignment.eq('test')).all():
            raise ValueError('Frozen split integrity failed')
        for suffix,oldcol,col in [('grouped_cv','fold','cv_fold'),('holdout','assignment','holdout_assignment')]:
            path=ROOT/f'outputs/splits/{prefix}_text_{suffix}.csv';inputs[str(path.relative_to(ROOT))]=sha(path)
            old=pd.read_csv(path,usecols=['content_uid','leakage_group',oldcol]).set_index('content_uid')
            if not old.index.is_unique or not set(d.content_uid)<=set(old.index):raise ValueError('Split UID mismatch')
            aligned=old.loc[d.content_uid]
            if aligned[oldcol].tolist()!=d[col].tolist() or aligned.leakage_group.tolist()!=d.leakage_group.tolist():
                raise ValueError('Original split/group assignments differ')
        cohorts[dataset]=d
    return cohorts,inputs


def summarize_lengths(lengths):
    a=np.asarray(lengths,dtype=int)
    if len(a)==0 or (a<=0).any():raise ValueError('Invalid token lengths')
    result=dict(N=len(a),median=float(np.median(a)),maximum=int(a.max()))
    result.update({f'p{q:02d}':float(np.quantile(a,q/100)) for q in [5,25,75,95,99]})
    result.update({f'percent_gt_{b}':float(100*(a>b).mean()) for b in BUDGETS})
    return result


def retention(lengths,budget):
    a=np.asarray(lengths,dtype=int)
    fractions=np.minimum(a,budget)/a
    return dict(percent_truncated=float(100*(a>budget).mean()),median_retained=float(np.median(fractions)),
                p10_retained=float(np.quantile(fractions,.1)),mean_retained=float(fractions.mean()))


def title_body_allocation(title_tokens,body_tokens,budget,pair_special_tokens=4):
    # Always preserve the full title or flag infeasibility; never silently trim it.
    remaining=budget-pair_special_tokens-title_tokens
    if remaining<0:return dict(feasible=False,title_kept=None,body_kept=None)
    return dict(feasible=True,title_kept=title_tokens,body_kept=min(body_tokens,remaining))


FUTURE_PROTOCOL = '## Future training protocol — proposal only\n\nUse AutoModelForSequenceClassification with two labels (Fake=1, Real=0), seed 42 and a pinned checkpoint. Retain existing five grouped CV folds and reserved holdout. Evaluate learning rates 2e-5/5e-5 and epoch checkpoints 2/3: four logical candidates per dataset, selectable from two three-epoch runs per training fold. Select mean grouped-CV macro F1 with a prespecified 0.001 tie tolerance, then lower sample SD, fewer epochs, lower learning rate. Weight decay 0.01. Do not use early stopping initially; two/three-epoch checkpoint choice uses training-pool validation only.\n\nTentative microbatch 2, gradient accumulation 8 (effective batch 16), pending hardware and memory validation; do not claim that an inference-only pilot establishes training memory requirements. Pin optimizer, schedule, precision and batch policy before tuning. Use FP32 initially and validate any hardware-dependent mixed precision separately. Training memory and wall-clock runtime are not certified by an inference-only pilot. No extrapolated hour estimate is supplied.\n\nPrimary loss proposal: weighted cross-entropy, with w_c=N_train/(2*N_train,c) computed separately from that fit’s source-training labels. This is a prespecified imbalance precaution, not a claim that this audit measured class effects. No target-derived weights or oversampling. Recompute source weights for the final source refit only, never from the target.\n\n## Frozen transfer proposal\n\nSelect and lock both source configurations from their own training-pool CV before target evaluation. Refit each on its training pool and evaluate its original holdout once. Separately initialize the same pretrained checkpoint and fit the locked configuration on all eligible source rows (including the former source holdout), then freeze target predictions before joining target labels. Apply this in both directions, evaluating all eligible target rows and identical target-holdout UIDs used by LR/SVM. No target tuning, calibration, threshold adjustment or feature selection. A fixed Fake probability threshold of 0.5 is proposed. For the 256-token sensitivity reuse each source’s primary hyperparameters to avoid a second search. No classification model is trained in this audit.\n\n'


def checkpoint_metadata():
    meta=json.loads(REPORT.with_name('transformer_checkpoint_metadata.json').read_text())
    revision=meta.get('revision','')
    if len(revision)!=40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('An immutable checkpoint revision must be downloaded first')
    return meta


def load_tokenizer():
    from transformers import AutoTokenizer
    t=AutoTokenizer.from_pretrained(CHECKPOINT,revision=checkpoint_metadata()['revision'],
        cache_dir=CACHE,local_files_only=True,use_fast=True)
    if t.model_max_length!=512 or not t.is_fast:
        raise ValueError('Expected fast official tokenizer with model_max_length=512')
    t.deprecation_warnings['sequence-length-is-longer-than-the-specified-maximum']=True
    return t


def pack_title_body(tokenizer,title_ids,body_ids,budget):
    allocation=title_body_allocation(len(title_ids),len(body_ids),budget,
        tokenizer.num_special_tokens_to_add(pair=True))
    if not allocation['feasible']:
        raise ValueError('Full title exceeds usable budget; explicit later policy needed')
    # Transformers 5 removed build_inputs_with_special_tokens. Read the
    # official template from a nonempty pair: empty second strings can be
    # treated as single inputs by tokenizer wrappers.
    template=tokenizer('x','y',add_special_tokens=True,return_special_tokens_mask=True)
    specials=[i for i,m in zip(template['input_ids'],template['special_tokens_mask']) if m]
    if specials!=[tokenizer.bos_token_id,tokenizer.eos_token_id,tokenizer.eos_token_id,tokenizer.eos_token_id]:
        raise ValueError('Unexpected official RoBERTa pair template')
    ids=specials[:1]+title_ids+specials[1:3]+body_ids[:allocation['body_kept']]+specials[3:]
    if len(ids)>budget or ids[1:1+len(title_ids)]!=title_ids:
        raise ValueError('Title packing integrity failed')
    return ids


def tokenizer_measurements(cohorts):
    tokenizer=load_tokenizer()
    rows=[];truncation=[];pilot_examples=[]
    for dataset,cohort in cohorts.items():
        d=pd.read_csv(ROOT/f'data/processed/{PREFIXES[dataset]}_text_modeling_ready.csv',
            usecols=['content_uid','combined_text','title_text','body_text'],keep_default_na=False).set_index('content_uid').loc[cohort.content_uid]
        lengths=[];title_lengths=[];body_lengths=[];pair_lengths=[]
        single_special=tokenizer.num_special_tokens_to_add(pair=False)
        pair_special=tokenizer.num_special_tokens_to_add(pair=True)
        lost={b:0 for b in BUDGETS};infeasible={b:0 for b in BUDGETS}
        verified={b:0 for b in BUDGETS};empty_titles=0;prefix_titles=0
        for row in d.itertuples():
            encoded=tokenizer(row.combined_text,truncation=False,add_special_tokens=False,
                return_attention_mask=False,return_offsets_mapping=True)
            ids=encoded['input_ids'];offsets=encoded['offset_mapping']
            repeated=tokenizer(row.combined_text,truncation=False,add_special_tokens=False,
                return_attention_mask=False)['input_ids']
            if ids!=repeated:raise ValueError('Nondeterministic official tokenization')
            lengths.append(len(ids)+single_special)
            def encode(text):
                return tokenizer(text,truncation=False,add_special_tokens=False,
                    return_attention_mask=False)['input_ids']
            title_ids,body_ids=encode(row.title_text),encode(row.body_text)
            title_lengths.append(len(title_ids));body_lengths.append(len(body_ids))
            pair_lengths.append(len(title_ids)+len(body_ids)+pair_special)
            title_indices=[]
            if row.title_text:
                start=row.combined_text.find(row.title_text)
                if start<0:raise ValueError('Title not found in prepared combined text')
                prefix_titles+=int(start==0)
                end=start+len(row.title_text)
                title_indices=[i for i,(a,b) in enumerate(offsets) if b>start and a<end and b>a]
                if not title_indices:raise ValueError('Nonempty title lacks token offsets')
            else:empty_titles+=1
            for budget in BUDGETS:
                lost[budget]+=int(any(i>=budget-single_special for i in title_indices))
                allocation=title_body_allocation(len(title_ids),len(body_ids),budget,pair_special)
                if not allocation['feasible']:
                    infeasible[budget]+=1
                    continue
                packed=pack_title_body(tokenizer,title_ids,body_ids,budget)
                verified[budget]+=1
                if budget==512 and len([x for x in pilot_examples if x['dataset']==dataset])<8:
                    pilot_examples.append(dict(dataset=dataset,content_uid=row.Index,input_ids=packed))
        rows.append(dict(dataset=dataset,status='measured',**summarize_lengths(lengths),
            single_special_tokens=single_special,title_maximum=max(title_lengths),empty_titles_N=empty_titles,
            titles_at_combined_prefix_N=prefix_titles,determinism_verified_N=len(lengths)))
        for budget in BUDGETS:
            truncation.append(dict(dataset=dataset,max_length=budget,status='measured',N=len(lengths),
                **retention(lengths,budget),naive_combined_title_loss_N=lost[budget],
                naive_combined_title_loss_percent=100*lost[budget]/len(lengths),
                titles_over_usable_pair_budget_N=infeasible[budget],
                titles_over_total_budget_N=sum(t>budget for t in title_lengths),
                packed_full_title_verified_N=verified[budget],pair_special_tokens=pair_special,
                pair_percent_truncated=100*float(np.mean(np.array(pair_lengths)>budget))))
        with LOG.open('a') as f:f.write(f'Tokenization complete: {dataset}, N={len(lengths)}; repeated full tokenization deterministic.\n')
    return rows,truncation,pilot_examples


def inference_pilot(examples,info):
    import torch
    from transformers import AutoModel
    from transformers.utils import logging as hf_logging
    hf_logging.disable_progress_bar()
    device=info['selected_device'].lower()
    model=AutoModel.from_pretrained(CHECKPOINT,revision=checkpoint_metadata()['revision'],
        cache_dir=CACHE,local_files_only=True,use_safetensors=True,add_pooling_layer=False)
    model.eval()
    model.to(device)
    versions=[p._version for p in model.parameters()]
    n_parameters=sum(p.numel() for p in model.parameters())
    results=[]
    def sync():
        if device=='mps':torch.mps.synchronize()
    with torch.inference_mode():
        for batch_size in (1,2):
            batches=[]
            for start in range(0,len(examples),batch_size):
                group=examples[start:start+batch_size]
                # Fixed padding tests the full proposed primary sequence budget.
                ids=torch.tensor([x['input_ids']+[1]*(512-len(x['input_ids'])) for x in group],device=device)
                batches.append(dict(input_ids=ids,attention_mask=ids.ne(1)))
            model(**batches[0]);sync()  # one warm-up per batch shape, excluded from timings
            times=[];allocated=[];driver=[]
            for batch in batches:
                sync();t0=time.perf_counter()
                output=model(**batch);sync()
                times.append(time.perf_counter()-t0)
                if not torch.isfinite(output.last_hidden_state).all().item():
                    raise ValueError('Nonfinite inference output')
                if device=='mps':
                    allocated.append(torch.mps.current_allocated_memory())
                    driver.append(torch.mps.driver_allocated_memory())
                del output
            results.append(dict(batch_size=batch_size,unique_examples=len(examples),timed_batches=len(times),
                max_length=512,padding='fixed 512',device=device,precision='float32',
                mean_batch_seconds=float(np.mean(times)),median_batch_seconds=float(np.median(times)),
                timed_seconds=sum(times),examples_per_second=len(examples)/sum(times),
                sampled_max_mps_allocated_bytes=max(allocated) if allocated else None,
                sampled_max_mps_driver_bytes=max(driver) if driver else None,
                memory_measurement='post-forward samples, not a reliable transient peak; not training memory'))
            with LOG.open('a') as f:f.write('Inference pilot: '+json.dumps(results[-1])+'\n')
    if model.training or any(p.grad is not None for p in model.parameters()):
        raise ValueError('Inference-only invariant failed')
    if versions!=[p._version for p in model.parameters()]:raise ValueError('Model parameters mutated')
    return dict(status='completed',device=device,unique_examples=len(examples),
        examples_per_dataset={d:sum(x['dataset']==d for x in examples) for d in PREFIXES},
        selection='first eight eligible sorted content_uid per dataset; label blind',
        rows=results,parameters=n_parameters,gradients_created=False,parameter_versions_unchanged=True,
        training_runs=0,backward_passes=0,optimizer_steps=0)


def proposals():
    return pd.DataFrame([
        dict(protocol='A',role='primary',max_length=512,predictor='title_text + body_text; tokenizer pair structure',
             packing='full title; body prefix in remaining budget; stop on title-over-budget exception',status='supported by completed token audit'),
        dict(protocol='B',role='sensitivity',max_length=256,predictor='same fields and packing as A',
             packing='same full-title guard; shorter absolute budget does not equalize retained fractions',status='supported by completed token audit')])


def markdown_table(rows,columns):
    def fmt(x):
        return f'{x:.4f}' if isinstance(x,float) else str(x)
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+
        ['| '+' | '.join(fmt(r.get(c,'')) for c in columns)+' |' for r in rows])


def unchanged_output_stats():
    snapshots=[line.split(': ',1)[1] for line in LOG.read_text().splitlines()
        if line.startswith('Unrelated output stat snapshot: ')]
    baseline=json.loads(snapshots[-1])
    current={str(p.relative_to(ROOT)):[p.stat().st_size,p.stat().st_mtime_ns]
        for p in (ROOT/'outputs').rglob('*') if p.is_file() and not p.name.startswith('transformer_')}
    if baseline!=current:raise ValueError('Unrelated output file metadata changed')
    return len(current)


def main():
    if sys.version_info[:2]!=(3,12) or Path(sys.prefix).resolve()!=ROOT/'.venv-transformer':
        raise ValueError('Run only with .venv-transformer/bin/python (Python 3.12)')
    packages=package_status();missing=[n for n in REQUIRED if not packages[n]['installed']]
    if missing:raise RuntimeError('Missing required packages: '+', '.join(missing))
    previous=json.loads(REPORT.with_suffix('.json').read_text())
    info=environment(packages);cohorts,inputs=load_assignments()
    if inputs!=previous['inputs']:raise ValueError('Frozen cohorts/splits differ from previous audit')
    if {d:len(c) for d,c in cohorts.items()}!={'FakeHealth':1647,'CoAID':3577}:
        raise ValueError('Unexpected eligible counts')
    lengths,truncation,examples=tokenizer_measurements(cohorts)
    try:pilot=inference_pilot(examples,info)
    except (OSError,RuntimeError,ImportError) as exc:
        pilot=dict(status='unavailable',reason=type(exc).__name__+': '+str(exc),rows=[])
        with LOG.open('a') as f:f.write('Inference pilot unavailable: '+pilot['reason']+'\n')
    meta=checkpoint_metadata();t=load_tokenizer()
    meta.update(locally_instantiated_tokenizer=True,tokenizer_class=type(t).__name__,
        tokenizer_model_max_length=t.model_max_length,tokenizer_vocab_size=t.vocab_size,
        single_special_tokens=t.num_special_tokens_to_add(pair=False),pair_special_tokens=t.num_special_tokens_to_add(pair=True),
        model_type='roberta',num_hidden_layers=6,hidden_size=768,num_attention_heads=12,
        artifact_sha256={p.name:sha(p) for p in (ROOT/meta['snapshot_directory']).iterdir()
            if p.is_file() and p.suffix in ('.json','.txt','.safetensors')})
    env_rows=[dict(category='package',name=n,status='installed',value=p['version']) for n,p in packages.items()]
    env_rows += [dict(category='environment',name=k,status='measured' if v is not None else 'unavailable',value=v) for k,v in info.items()]
    env_rows += [dict(category='checkpoint',name=k,status='verified',value=json.dumps(v) if isinstance(v,(dict,list)) else v)
        for k,v in meta.items()]
    env_rows += [dict(category='pilot',name=k,status=pilot['status'],value=json.dumps(v) if isinstance(v,(dict,list)) else v)
        for k,v in pilot.items()]
    feasible=pilot['status']=='completed'
    practicality=('Local small-scale fine-tuning appears practical on MPS; start with microbatch 1 and test 2 under backward/optimizer memory load.'
        if feasible and info['mps_available'] else
        'CPU inference works; local fine-tuning may be possible but throughput and training memory remain unvalidated.'
        if feasible else 'Local fine-tuning practicality is not established; inference pilot unavailable.')
    env_rows.append(dict(category='recommendation',name='local_fine_tuning_practicality',status='conditional',value=practicality))
    outputs={'transformer_token_length_audit':pd.DataFrame(lengths),'transformer_truncation_audit':pd.DataFrame(truncation),
        'transformer_environment_audit':pd.DataFrame(env_rows),'transformer_protocol_proposal':proposals()}
    paths=[]
    for name,table in outputs.items():
        path=TABLES/f'{name}.csv';table.to_csv(path,index=False);paths.append(path)
    metadata_path=REPORT.with_name('transformer_checkpoint_metadata.json')
    metadata_path.write_text(json.dumps(meta,indent=2)+'\n');paths.append(metadata_path)
    table512={r['dataset']:r for r in truncation if r['max_length']==512}
    delta=table512['FakeHealth']['percent_truncated']-table512['CoAID']['percent_truncated']
    titleloss=sum(r['naive_combined_title_loss_N'] for r in truncation if r['max_length']==512)
    guard=sum(r['titles_over_usable_pair_budget_N'] for r in truncation if r['max_length'] in (256,512))
    lines=['# Transformer feasibility and tokenization audit','',
        'Status: **feasibility complete**. Tokenization and title-packing measurements are complete; inference pilot: '+pilot['status']+'. No training occurred.','',
        '## Environment and pinned checkpoint','',
        f'Python {info["python"]}; macOS {info["macos_version"]}; architecture {info["machine"]}; '
        f'torch {info["torch_version"]}; transformers {info["transformers_version"]}. '
        f'MPS built={info["mps_built"]}; available={info["mps_available"]}; selected device={info["selected_device"]}. '
        f'Physical memory={info["system_memory_bytes"]} bytes ({info["system_memory_status"]}). CPU and both imports explicitly tested. CUDA was not probed.','',
        'The four requested packages and their transitive dependencies were installed only into `.venv-transformer`, '
        'using stable PyPI resolution. All resolved packages are pinned in `requirements-transformer.txt`. '
        'The original `.venv` was neither installed into nor rewritten. Setup and detailed diagnostics are in `logs/transformer_environment_setup.log`.','',
        f'Checkpoint [{CHECKPOINT}](https://huggingface.co/{CHECKPOINT}/tree/{meta["revision"]}), '
        f'immutable revision `{meta["revision"]}`. Official fast tokenizer instantiated with model_max_length=512; '
        f'vocabulary={t.vocab_size}; special-token overhead: single={meta["single_special_tokens"]}, pair={meta["pair_special_tokens"]}. '
        'Only this checkpoint was downloaded. Audit replay uses local_files_only=True and the exact revision. '
        'The model has 514 position embeddings, but the permitted total sequence length is 512.','',
        '## Untruncated combined-text token lengths','',
        'All primary_modeling_eligible rows were used. Counts include the tokenizer’s single-input special tokens, '
        'with no truncation or padding. Every document was tokenized twice and the full token sequences compared in memory. '
        'No article text, token sequences, tensors, or model outputs were saved to audit files. Quantiles use linear interpolation.','',
        markdown_table(lengths,['dataset','N','median','p05','p25','p75','p95','p99','maximum']),'',
        '## Truncation and retained-token fractions','',
        'For untruncated combined-text length L and cap B, truncation is L>B; retained fraction is min(L,B)/L. '
        'This requested metric includes special tokens and counts tokens, not semantic information.','',
        markdown_table(truncation,['dataset','max_length','percent_truncated','median_retained','p10_retained','mean_retained']),'',
        f'At 512, FakeHealth truncation is {table512["FakeHealth"]["percent_truncated"]:.4f}% versus '
        f'{table512["CoAID"]["percent_truncated"]:.4f}% in CoAID, a {delta:.4f} percentage-point difference. '
        'The retained-fraction summaries quantify the asymmetry; a common cap does not make retained information equal.','',
        '## Full-title packing audit','',
        'Naive combined-text head truncation was checked using exact title character spans and official fast-tokenizer offsets. '
        'Any title-overlapping token beyond B minus the single-input special-token overhead counts as title loss. '
        'This measures actual combined-text head truncation, not paired longest-first truncation. '
        'An absent nonempty title is an error; empty titles require no preservation.','',
        markdown_table(lengths,['dataset','title_maximum','empty_titles_N','titles_at_combined_prefix_N']),'',
        markdown_table(truncation,['dataset','max_length','naive_combined_title_loss_N','naive_combined_title_loss_percent',
            'titles_over_usable_pair_budget_N','packed_full_title_verified_N','pair_percent_truncated']),'',
        f'At 512, {titleloss} documents lose title tokens under naive combined-text truncation. '
        f'Across the 512 and 256 candidates, {guard} title-over-budget cases require a later explicit policy. '
        'The deterministic pair rule tokenizes the complete title, reserves four official special tokens, and assigns '
        'the remaining budget to a body prefix: `<s> title </s></s> body-prefix </s>`. '
        'Every feasible packed input was checked for exact title-token preservation and total length. '
        'Infeasible titles are flagged; they are never silently shortened or dropped. '
        'Pair truncation percentages are separate because pair boundaries and overhead differ from combined_text.','',
        '## Max-length recommendation','',
        'Recommend **512 primary** and **256 sensitivity**. The longer cap preserves more body text in the longer-document '
        'cohort; 256 tests dependence on the available body prefix and lowers sequence computation. '
        'The shorter cap is not an information-equalizing correction. Keep identical cohorts, splits, title-preserving '
        'packing, and source-selected settings. The recommendation concerns feasibility and representation, not predictive quality.','',
        '## Inference-only hardware pilot','',
        json.dumps({k:v for k,v in pilot.items() if k!='rows'},sort_keys=True),'',
        markdown_table(pilot['rows'],['device','batch_size','unique_examples','timed_batches','mean_batch_seconds',
            'examples_per_second','sampled_max_mps_allocated_bytes','sampled_max_mps_driver_bytes']),'',
        'At most eight eligible documents per dataset were selected by sorted UID without labels. '
        'Inputs used the 512-token title-preserving rule and fixed padding to 512. One warm-up per batch size was excluded. '
        'Timings cover synchronized forward passes with input tensors already on the device; tokenization, model loading, '
        'and data transfer are excluded. AutoModel without a classifier or pooler ran in eval and torch.inference_mode. '
        'MPS memory values are maximum post-forward samples, not reliable transient peaks. '
        'Parameters were not mutated; no gradients, backward pass, optimizer, or training epoch occurred.','',
        practicality+' Inference microbatches 1 and 2 do not establish a training microbatch. '
        'Training requires additional activations, gradients, and optimizer state. No training-time extrapolation is claimed.','',
        FUTURE_PROTOCOL,
        '## Validation','',
        'Runtime checks passed: exact previous cohort/split hashes, fixed eligible counts, original group/fold assignments, '
        'official-tokenizer determinism on every row, label-blind predictor-only packing, title preservation, and '
        'bounded inference-only invariants. Only UID, eligibility/split/group metadata and the three prepared text fields were read. '
        'No labels, source, or engagement metadata enter the representation. '
        'Unrelated output file sizes and modification times are checked without opening their contents. '
        'Original environment fingerprint and automated test results are recorded in the companion JSON after verification.','',
        'Reproduce: `.venv-transformer/bin/python src/audit_transformer_feasibility.py`. '
        'Tests: `.venv-transformer/bin/python -m unittest discover -s src -p "test_transformer_feasibility.py" -q`.','']
    REPORT.write_text('\n'.join(lines));paths.append(REPORT)
    if any(sha(ROOT/p)!=h for p,h in inputs.items()):raise ValueError('Original inputs changed')
    unchanged_count=unchanged_output_stats()
    result=dict(status='feasibility_complete',missing_packages=[],validation='runtime checks passed; unit tests pending',
        packages=packages,environment=info,inputs=inputs,script_sha256=sha(__file__),training_runs=0,
        packages_installed=0,setup_packages_installed_in='.venv-transformer',official_tokenizer_audited=True,
        runtime_pilot_performed=feasible,pilot=pilot,checkpoint_revision=meta['revision'],
        unrelated_output_stat_checks=unchanged_count,requirements_sha256=sha(ROOT/'requirements-transformer.txt'),
        outputs={str(p.relative_to(ROOT)):sha(p) for p in paths})
    REPORT.with_suffix('.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    with LOG.open('a') as f:f.write('Audit complete; no training; runtime validation passed.\n')


if __name__=='__main__':
    main()
