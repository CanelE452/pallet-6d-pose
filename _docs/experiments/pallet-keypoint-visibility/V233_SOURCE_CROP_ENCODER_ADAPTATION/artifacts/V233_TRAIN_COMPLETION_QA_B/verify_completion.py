"""V233 final-only saved-state QA; never instantiate a model or replay forward.

Future --entry requires approved_actual_completion_QA=True, both_final1000=True,
verifier_code, complete, root_fit_entry, failure_preservation, root_protocol,
common_draw, whitelist, output_directory. All are exact {path,sha256,size_bytes?}
bindings except flags/whitelist/output_directory. COMPLETE supplies checkpoints,
logs, samples_npz, all2_freeze, outputs, summary and embedded control_reuse.
No payload is read on import or before explicit approval + completed final seal.
"""
import argparse
import collections
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import traceback

R=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
H=R/'V233_TRAIN_COMPLETION_QA_B'
SOURCE_ROOT=R/'V233_ROOT_CROP_ADAPTATION'
ARMS=('CONTROL_LOCAL_CROP','TREAT_LAST_QKV_LORA4')
PROTOCOL_SHA='6755fecc3fd7927163357eb0caa677f4635e135f7819c954ae52316f3f29b28b'
REPAIR_PROTOCOL_SHA='1c55fd04849eec0634c181f04bbef0e9e48e069d118d90f46162e3fbe8763fba'
SAMPLE_SHA='837bbe7ff406e27c6ed2981493a83c2adab830cd5b2de498e4a48de456456e99'
OFFICIAL_SHA='f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb'
VENDOR_REVISION='7764ea0f912e53c92e82eb78a2a1631e92725fc8'
ADAPTER_KEYS={'blocks.11.attn.qkv.lora_A':(4,384),'blocks.11.attn.qkv.lora_B':(1152,4)}
HEAD_KEYS={'head.0.weight':(64,1186),'head.0.bias':(64,),'head.2.weight':(2,64),'head.2.bias':(2,)}

def safe(path):
    text=os.fspath(path)
    for token in ('v112','current223','publicheld','oldval','v177','v178','source_holdout','v182','private_evaluation'):
        if token in text.casefold():raise ValueError('Forbidden before FS: '+token)
    p=Path(os.path.abspath(text))
    if not p.is_relative_to(R):raise ValueError('Only closed local preparation/output paths')
    cursor=Path('/')
    for part in p.parts[1:]:
        cursor/=part
        try:s=os.lstat(cursor)
        except FileNotFoundError:break
        if stat.S_ISLNK(s.st_mode):raise ValueError('Symlink')
    return p

def identity(binding):
    if set(binding) not in ({'path','sha256'},{'path','sha256','size_bytes'}):raise ValueError('Closed binding keys')
    sha=binding['sha256']
    if not isinstance(sha,str)or len(sha)!=64 or any(c not in '0123456789abcdef'for c in sha):raise ValueError('SHA256 format')
    if 'size_bytes'in binding and(type(binding['size_bytes'])is not int or binding['size_bytes']<0):raise ValueError('Size metadata')
    return str(Path(os.path.abspath(binding['path']))),sha

def read_unbound_admin(path):
    p=safe(path)
    if p.suffix!='.json':raise ValueError('Entry JSON only')
    with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb')as f:return f.read()

class ClosedReader:
    def __init__(self,whitelist):
        self.allowed={identity(b):b for b in whitelist}
        if len(self.allowed)!=len(whitelist):raise ValueError('Duplicate binding')
        self.access=[]
    def read(self,binding,kind='metadata'):
        key=identity(binding)
        if key not in self.allowed:raise ValueError('Exact whitelist before FS')
        p=safe(key[0])
        if kind=='checkpoint':
            if not p.is_relative_to(SOURCE_ROOT)or p.suffix!='.pt':raise ValueError('Own final checkpoint only')
        elif kind=='samples':
            if str(p)!=str(SOURCE_ROOT/'crop_samples_v1/samples.npz')or binding['sha256']!=SAMPLE_SHA:raise ValueError('Only already-QA samples metadata NPZ')
        elif p.suffix not in {'.json','.jsonl','.csv','.md','.py','.pt'}:raise ValueError('No prefix/raw RGB/NPY/input arrays')
        # .pt is inert SHA-only unless caller explicitly chooses checkpoint kind.
        with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb')as f:raw=f.read()
        if hashlib.sha256(raw).hexdigest()!=binding['sha256']:raise ValueError('SHA mismatch')
        if 'size_bytes'in binding and len(raw)!=binding['size_bytes']:raise ValueError('Size mismatch')
        self.access.append({'path':str(p),'sha256':binding['sha256'],'bytes':len(raw),'kind':kind})
        return raw
    def json(self,binding):return json.loads(self.read(binding))

def tensor_state_sha(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        a=value.detach().cpu().contiguous().numpy()
        digest.update(name.encode());digest.update(str(a.dtype).encode())
        digest.update(json.dumps(list(a.shape)).encode());digest.update(a.tobytes())
    return digest.hexdigest()

def verify(entry_path):
    entryraw=read_unbound_admin(entry_path);e=json.loads(entryraw)
    if e.get('approved_actual_completion_QA')is not True or e.get('both_final1000')is not True:
        raise ValueError('Future root completion entry required; no pre-barrier payload read')
    with os.fdopen(os.open(safe(__file__),os.O_RDONLY|os.O_NOFOLLOW),'rb')as f:code=f.read()
    if identity(e['verifier_code'])!=(str(safe(__file__)),hashlib.sha256(code).hexdigest()):raise ValueError('Verifier code pin')
    reader=ClosedReader(e['whitelist'])
    # Final completion and both checkpoint bindings/barrier must close before PT/log/NPZ decode.
    complete=reader.json(e['complete'])
    if not(complete.get('completed')is True and complete.get('PASS')is True and complete['steps_each']==1000):raise ValueError('Both completed1000 first')
    if set(complete['checkpoints'])!=set(ARMS)or set(complete['logs'])!=set(ARMS):raise ValueError('Exact two final arms')
    freeze=reader.json(complete['all2_freeze'])
    if not(freeze.get('completed')is True and freeze.get('PASS')is True and freeze['steps_each']==1000 and freeze['DEV_forward_calls_before_freeze']==0):raise ValueError('Both-final-before-DEV barrier')
    if any(identity(freeze['checkpoints'][a])!=identity(complete['checkpoints'][a])for a in ARMS):raise ValueError('Barrier final states')
    if complete['DEV_forward_before_freeze']!=0 or not complete['numeric_prefix_rows_before_freeze_only_FIT']:raise ValueError('No pre-barrier trained DEV')
    if complete['new_RAW_RGB_GT_or_old_SOURCE_NPZ_reads']!=0 or complete['new_real_inference_or_model_threshold_selection']!=0:raise ValueError('SOURCE-only scope')
    if identity(complete['entry'])!=identity(e['root_fit_entry']):raise ValueError('Root actual fit entry')
    fit_entry=reader.json(e['root_fit_entry'])
    if fit_entry.get('approved_control_reuse_and_treatment_restart')is not True:raise ValueError('Explicit mechanical repair/reuse entry')
    if fit_entry['steps_each']!=1000 or fit_entry['threshold_logit']!=0:raise ValueError('Fixed budget/threshold')
    if fit_entry['reuse_control']['failed_treatment_optimizer_updates']!=0:raise ValueError('Treatment reuse not allowed after updates')
    protocol=reader.json(e['root_protocol'])
    if e['root_protocol']['sha256']!=PROTOCOL_SHA:raise ValueError('Original prospective root protocol')
    if identity(complete['protocol'])!=identity(fit_entry['protocol'])or complete['protocol']['sha256']not in {PROTOCOL_SHA,REPAIR_PROTOCOL_SHA}:raise ValueError('Pinned original/repair protocol')
    reader.json(complete['protocol'])
    failure=reader.json(e['failure_preservation'])
    if failure['actual_CONTROL_updates']!=1000 or failure['actual_TREAT_optimizer_updates']!=0 or failure['DEV_forward']!=0:raise ValueError('Failure/CONTROL preservation authority')
    reuse=complete['control_reuse']
    if reuse.get('no_CONTROL_retraining')is not True:raise ValueError('Control must be reused without update')
    if complete['new_CONTROL_optimizer_updates']!=0 or complete['new_TREAT_optimizer_updates']!=1000:raise ValueError('No repeated CONTROL fitting')
    if freeze['control_reuse']!=reuse:raise ValueError('Reuse receipt shared by barrier/complete')
    if identity(freeze['control_reuse_QA'])!=identity(complete['control_reuse_QA']):raise ValueError('Reuse QA binding')
    reuse_QA=reader.json(complete['control_reuse_QA'])
    if reuse_QA.get('completed')is not True or reuse_QA.get('PASS')is not True:raise ValueError('Completed reuse QA')
    if any(reuse_QA[k]!=v for k,v in reuse.items()):raise ValueError('Copied receipt body')
    if reuse_QA['original_1000_log_draws_exact']is not True or reuse_QA['initial_head_draws_base_official_asset_entry_exact']is not True:raise ValueError('Recorded reuse identity guards')
    if identity(reuse_QA['root_repair_scope'])!=identity(fit_entry['root_repair_scope']):raise ValueError('Root mechanical repair scope')
    reader.json(fit_entry['root_repair_scope'])
    if identity(reuse_QA['fit_feature_forward_equivalence_QA'])!=identity(fit_entry['fit_feature_forward_equivalence_QA']):raise ValueError('SOURCEFIT forward-equivalence receipt pin')
    equivalence=reader.json(reuse_QA['fit_feature_forward_equivalence_QA'])
    if equivalence.get('completed')is not True or equivalence.get('PASS')is not True:raise ValueError('Root recorded fixed-SOURCEFIT equivalence pass')
    for source_key,fit_key in [('original_checkpoint','checkpoint'),('original_log','log'),('original_entry','original_entry')]:
        if identity(reuse[source_key])!=identity(fit_entry['reuse_control'][fit_key]):raise ValueError('Reuse original binding')
    for arm_key,copied_key in [('checkpoints','copied_checkpoint'),('logs','copied_log')]:
        if identity(reuse[copied_key])!=identity(complete[arm_key][ARMS[0]]):raise ValueError('Reuse copied binding')
    if reuse['original_checkpoint']['sha256']!=failure['completed_control']['sha256']or reuse['original_log']['sha256']!=failure['logs']['CONTROL_LOCAL_CROP_TRAIN1000.jsonl']['sha256']:raise ValueError('Original CONTROL exact preserved bytes')
    reader.read(reuse['original_checkpoint']);reader.read(reuse['original_log'])
    if reuse['original_checkpoint']['sha256']!=reuse['copied_checkpoint']['sha256']or reuse['original_log']['sha256']!=reuse['copied_log']['sha256']:raise ValueError('CONTROL copy byte identity')
    # Only own metadata NPZ now decoded; raw RGB/prefix/source arrays stay closed.
    import numpy as np
    samplesraw=reader.read(complete['samples_npz'],'samples')
    with np.load(io.BytesIO(samplesraw),allow_pickle=False)as f:data={k:f[k]for k in f.files}
    if set(data)!={'full_view','central_view','fractions','central4','condition','target','family','fit','available','row_id'}:raise ValueError('Samples schema')
    if len(data['target'])!=7617:raise ValueError('Fixed sample count')
    eligible=data['fit']&data['available']&(data['target']>=0)
    pools=[np.flatnonzero(eligible&(data['family']==family)&(data['target']==target))for family in(0,1)for target in(0,1)]
    if any(len(p)==0 for p in pools):raise ValueError('Fixed class pools')
    rng=np.random.Generator(np.random.PCG64(17))
    draws=np.stack([np.concatenate([rng.choice(pool,8,replace=True)for pool in pools])for _ in range(1000)]).astype(np.int64)
    drawsha=hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest()
    common=reader.json(e['common_draw'])
    if common['seed']!=17 or not np.array_equal(np.asarray(common['indices'],np.int64),draws)or common['sha256']!=drawsha:raise ValueError('Full common ledger replay')
    if complete['draws_sha256']!=drawsha or freeze['same_draws_sha256']!=drawsha:raise ValueError('Shared draw SHA')
    pre=np.asarray(freeze['prebarrier_only_known_available_FIT_samples'],np.int64)
    if len(pre)==0 or len(set(pre.tolist()))!=len(pre)or not eligible[pre].all():raise ValueError('Pre-barrier FIT only')
    expected_views=np.unique(np.r_[data['full_view'][pre],data['central_view'][pre]])
    if not np.array_equal(np.asarray(freeze['prebarrier_view_indices']),expected_views):raise ValueError('Pre-barrier view/sample mapping')
    if not set(draws.reshape(-1).tolist()).issubset(set(pre.tolist())):raise ValueError('All loss draws guarded')
    log_counts={}
    for arm in ARMS:
        logs=[json.loads(line)for line in reader.read(complete['logs'][arm]).splitlines()if line]
        if len(logs)!=1000:raise ValueError('Exactly1000 log updates')
        for j,row in enumerate(logs):
            if row['update']!=j+1 or row['draws_sha256']!=drawsha or row['sample_indices']!=draws[j].tolist():raise ValueError('Actual paired draw/update replay')
            if row['family_counts']!=[16,16]or row['all_fit_available_known']is not True:raise ValueError('Loss scope')
            for family in(0,1):
                for target in(0,1):
                    if int(np.sum((data['family'][draws[j]]==family)&(data['target'][draws[j]]==target)))!=8:raise ValueError('Each family8D8E')
            a,b,total=(row[k]for k in('family0_BCE','family1_BCE','total_loss'))
            if not all(math.isfinite(x)and x>=0 for x in(a,b,total))or not math.isclose(total,.5*(a+b),rel_tol=2e-6,abs_tol=2e-7):raise ValueError('Logged finite loss arithmetic')
        log_counts[arm]=len(logs)
    # CPU safe-load own two final checkpoints only; never construct a backbone/head.
    import torch
    checkpoints={arm:torch.load(io.BytesIO(reader.read(complete['checkpoints'][arm],'checkpoint')),map_location='cpu',weights_only=True)for arm in ARMS}
    initial=None;base_sha=None;states={};adapter_changes={}
    for arm in ARMS:
        c=checkpoints[arm];treat=arm==ARMS[1]
        if c['arm']!=arm or c['updates']!=1000 or c['use_lora']!=treat:raise ValueError('Final arm/update')
        if c['head_parameters']!=76098 or c['adapter_parameters']!=(6144 if treat else 0):raise ValueError('Exact declared trainable capacity')
        if c['adapter_seed']!=17 or c['rank']!=4 or c['scale']!=1 or c['draws_sha256']!=drawsha:raise ValueError('Same mechanical init/HPS/draws')
        if c['vendor_revision']!=VENDOR_REVISION or c['official_base']['sha256']!=OFFICIAL_SHA:raise ValueError('Pinned official base metadata only')
        expected_entry=e['root_fit_entry']if treat else reuse['original_entry']
        if identity(c['entry'])!=identity(expected_entry):raise ValueError('Copied CONTROL versus resumed TREAT provenance')
        for statekey in('head_state_dict','initial_head_state_dict'):
            state=c[statekey]
            if set(state)!=set(HEAD_KEYS):raise ValueError('Exact head tensors')
            for name,shape in HEAD_KEYS.items():
                tensor=state[name]
                if tuple(tensor.shape)!=shape or tensor.dtype!=torch.float32 or not torch.isfinite(tensor).all():raise ValueError('Head finite FP32 scope')
        if any(c['initial_head_state_dict'][k].count_nonzero()for k in('head.2.weight','head.2.bias')):raise ValueError('Zero final initial classifier')
        init_sha=tensor_state_sha(c['initial_head_state_dict'])
        if init_sha!=c['initial_head_sha256']or init_sha!=complete['head_initial_sha256']or init_sha!=freeze['same_head_initial_sha256']:raise ValueError('Same exact initial head')
        if initial is None:initial=c['initial_head_state_dict']
        elif any(not torch.equal(initial[k],c['initial_head_state_dict'][k])for k in HEAD_KEYS):raise ValueError('Paired initial state exact')
        if base_sha is None:base_sha=c['base_state_sha256_before_after']
        if c['base_state_sha256_before_after']!=base_sha or complete['base_state_sha256_before_after']!=base_sha or freeze['same_base_sha256']!=base_sha:raise ValueError('Same unchanged base recorded SHA')
        for statekey in('adapter_state_dict','initial_adapter_state_dict'):
            state=c[statekey]
            if set(state)!=(set(ADAPTER_KEYS)if treat else set()):raise ValueError('Only final-qkv A/B tensors')
            for name,tensor in state.items():
                if tuple(tensor.shape)!=ADAPTER_KEYS[name]or tensor.dtype!=torch.float32 or not torch.isfinite(tensor).all():raise ValueError('Adapter finite FP32 scope')
        if treat:
            initial_adapter=c['initial_adapter_state_dict'];final_adapter=c['adapter_state_dict']
            if initial_adapter['blocks.11.attn.qkv.lora_B'].count_nonzero():raise ValueError('Zero B init')
            changed={k:not torch.equal(initial_adapter[k],final_adapter[k])for k in ADAPTER_KEYS}
            if not all(changed.values()):raise ValueError('Both adapter matrices must have learned changes')
            adapter_changes[arm]=changed
        opt=c['optimizer_state_dict'];groups=opt['param_groups']
        if len(groups)!=(2 if treat else 1):raise ValueError('Head plus only treatment adapter optimizer groups')
        expected_shapes=list(HEAD_KEYS.values())+(list(ADAPTER_KEYS.values())if treat else [])
        ids=[]
        for j,group in enumerate(groups):
            if group['lr']!=(1e-3 if j==0 else 1e-4)or group['weight_decay']!=1e-4 or tuple(group['betas'])!=(.9,.999)or group['eps']!=1e-8:raise ValueError('AdamW HPS fixed')
            if len(group['params'])!=(4 if j==0 else 2):raise ValueError('No base optimizer parameters')
            ids.extend(group['params'])
        if len(set(ids))!=len(ids)or set(opt['state'])!=set(ids):raise ValueError('Exact used optimizer state')
        for pid,shape in zip(ids,expected_shapes):
            state=opt['state'][pid]
            if float(state['step'])!=1000:raise ValueError('Every used tensor optimizer step1000')
            for k in('exp_avg','exp_avg_sq'):
                if tuple(state[k].shape)!=shape or not torch.isfinite(state[k]).all():raise ValueError('Finite optimizer state scope')
        states[arm]={'head_SHA256':tensor_state_sha(c['head_state_dict']),'adapter_SHA256':tensor_state_sha(c['adapter_state_dict']),'optimizer_tensors':len(ids),'updates':1000}
    # Scores and summary are opaque byte bindings here, not another numeric/model replay.
    for binding in complete['outputs'].values():reader.read(binding)
    reader.read(complete['summary'])
    for binding in complete['checkpoints'].values():reader.read(binding)  # still unchanged after output binding reads
    return {'completed':True,'PASS':True,'scope':'Saved completion/log/ledger/own final tensor QA only',
            'complete':e['complete'],'root_entry':{'path':str(safe(entry_path)),'sha256':hashlib.sha256(entryraw).hexdigest()},
            'states':states,'adapter_changes':adapter_changes,'log_rows':log_counts,'same_initial_head_SHA256':complete['head_initial_sha256'],'same_draw_SHA256':drawsha,
            'class_pool_sizes_family0D_E_family1D_E':[len(p)for p in pools],'loss_DEV_rows':0,'control_original_bytes_preserved':True,'threshold_logit':0,
            'checks':['1000 fixed updates each','full shared PCG64 ledger and per-family8D8E','FIT/available/known-only loss','control checkpoint/log exact copy','same headinitial/base recorded SHA','only6144 treatment adapter tensors change and finite','optimizer4/6 tensors at1000 fixed HPS','bothfinal barrier before trainedDEV output by sealed record+reviewed CODE','frozen checkpoint/output bindings exact'],
            'access':reader.access,'limits':['No model instantiated or neural forward/gradient replay.','Official base SHA invariance and scoring chronology rely on reviewed trainer guards+sealed metadata; original pretrained bytes not opened.','SOURCE score files are SHA-bound only; no additional performance selection/threshold/epoch/seed or input-collision analysis.','No raw source RGB/GT/NPY/prefix/original SOURCE NPZ/protected access.']}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--entry',required=True);args=parser.parse_args()
    # False/unapproved entry stops before any target checkpoint/log/packet reads.
    admin=json.loads(read_unbound_admin(args.entry))
    if admin.get('approved_actual_completion_QA')is not True or admin.get('both_final1000')is not True:
        raise ValueError('Actual final-only QA entry not authorized')
    out=safe(admin['output_directory'])
    if out.parent!=H or out.exists():raise ValueError('Fresh own actual review folder only')
    out.mkdir()
    try:
        result=verify(args.entry)
        with (out/'REVIEW.json').open('x')as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
        print(json.dumps({'completed':True,'PASS':True,'review':str(out/'REVIEW.json')}))
    except BaseException:
        with (out/'FAILED_REVIEW.json').open('x')as f:json.dump({'completed':False,'PASS':False,'automatic_retry':False,'traceback':traceback.format_exc()},f,indent=2)
        raise

if __name__=='__main__':main()
