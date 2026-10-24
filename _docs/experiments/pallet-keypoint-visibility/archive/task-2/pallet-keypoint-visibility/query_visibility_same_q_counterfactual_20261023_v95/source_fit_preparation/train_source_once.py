"""V95 C2: one source-only fit. Import reads no data or checkpoint tensors."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import importlib.util
import json
import random
import time
import traceback
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL = ROOT.parent/'query_visibility_matched_query_scalar_20261023_v94/operator_preparation/query_center_scalar.py'
MODEL_SHA = 'd62e1610ca45f24e3cbf90e66dff3a7d954f87713afea4f5ae7bdc404411b0cb'
PROTOCOL = HERE/'FIT_PROTOCOL_BEFORE_SOURCE_ARRAYS.json'
RUN = ROOT/'runs/C2_SAME_Q_COUNTERFACTUAL_SOURCE_V1'
VARIANTS = ('BEFORE', 'OPAQUE_COVER', 'CUE_PRESERVING_SHAM')
SETTINGS = dict(seed=194, updates=2000, batch=64, D_per_batch=32, E_per_batch=32,
                AdamW_lr=.001, AdamW_weight_decay=.0001, clip_norm=5.,
                validation_every=100, validation_update0_eligible=True,
                selector='earliest strict minimum mean BCE of four VAL strata',
                threshold_raw_logit=0., online_augmentation=False)
BINDINGS = ('trainer','protocol','model_code','candidate_definition','input_pack_manifest',
            'input_pack_seal','base_input_lineage','GT_ledger','generated_tests',
            'independent_preparation_review','independent_input_audit')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):
    p=Path(p); return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def read(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def write(p,x):
    with Path(p).open('x') as f: json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')
def verify(b): assert sha(b['path'])==b['sha256'], b['path']
def same_binding(a,b):
    return Path(a['path']).resolve()==Path(b['path']).resolve() and a['sha256']==b['sha256']
def operator():
    assert sha(MODEL)==MODEL_SHA
    spec=importlib.util.spec_from_file_location('v95_unchanged_scalar',MODEL)
    op=importlib.util.module_from_spec(spec);spec.loader.exec_module(op);return op

def canonical_split(source,raw_split):
    mapping={('LOCO','TRAIN'):'TRAIN',('LOCO','VALIDATION'):'VALIDATION',
             ('PALLOX','TRAIN_LAB'):'TRAIN',('PALLOX','VAL_HALL'):'VALIDATION'}
    assert (source,raw_split) in mapping,'Unfrozen source/split pair'
    return mapping[source,raw_split]

def require_entry(e):
    assert e.get('approved_actual_source_fit') is True, 'Root execution entry required before source reads'
    assert e.get('source_only') is True and e.get('current_allowed') is False and e.get('PnP_allowed') is False
    assert e.get('allow_restart_after_optimizer_update') is False
    for k in BINDINGS: verify(e[k])
    assert same_binding(e['trainer'],bind(__file__)) and same_binding(e['protocol'],bind(PROTOCOL))
    assert same_binding(e['model_code'],bind(MODEL)) and e['model_code']['sha256']==MODEL_SHA
    p=read(PROTOCOL);assert p['settings']==SETTINGS and p['source_only'] is True and p['current_allowed'] is False
    for k,b in p['bindings'].items(): assert k in e and same_binding(e[k],b),k
    for k in ['generated_tests','independent_preparation_review','independent_input_audit']:
        assert read(e[k]['path'])['PASS'] is True,k
    assert same_binding(read(e['generated_tests']['path'])['trainer'],e['trainer'])
    assert same_binding(read(e['independent_preparation_review']['path'])['trainer'],e['trainer'])
    assert Path(e['run_directory']).resolve()==RUN.resolve() and not RUN.exists()
    return p

def join_metadata(lineage,ledger):
    assert len(lineage)==len(ledger)==2880
    labels=[];ready=[]
    for i,(r,g) in enumerate(zip(lineage,ledger)):
        assert r['occurrence_index']==g['occurrence_index']==i
        assert r['slot_index']==g['base_row_index']==g['slot_index']==i//3
        assert r['variant']==g['variant']==VARIANTS[i%3] and r['occurrence_id']==g['occurrence_id']
        canonical_split(r['source'],r['split'])
        assert g['native_expected_part_state']=='UNKNOWN' and g['human_gold'] is False
        y=g['candidate_aux_hidden_label_GT_only'];assert y in [-1,0,1]
        assert g['known_candidate_GT_only']==(y>=0)
        labels.append(y);ready.append(bool(g['pair_ready']))
    y=np.array(labels,np.int8);accepted=np.array(ready,bool).reshape(960,3)
    assert (accepted==accepted[:,:1]).all()
    for i in range(960):
        rr=lineage[3*i:3*i+3];gg=ledger[3*i:3*i+3]
        assert len({r['source'] for r in rr})==len({r['split'] for r in rr})==1
        assert all(r['query_xy']==rr[0]['query_xy'] and r['L']==rr[0]['L'] for r in rr)
        original=gg[0]['original_natural_label_GT_only']
        assert original in [-1,0,1] and all(g['original_natural_label_GT_only']==original for g in gg)
        assert y[3*i]==original
        assert y[3*i:3*i+3].tolist()==([0,1,0] if accepted[i,0] else [original,-1,-1])
    source=np.array([r['source'] for r in lineage]);raw_split=np.array([r['split'] for r in lineage])
    split=np.array([canonical_split(r['source'],r['split']) for r in lineage])
    variant=np.array([r['variant'] for r in lineage]);before=variant=='BEFORE'
    assert [int(np.sum(before&(y==k))) for k in [0,1,-1]]==[143,20,797]
    counts={}
    for s in ['TRAIN','VALIDATION']:
        m=split==s
        counts[s]=dict(before_D=int(np.sum(m&before&(y==0))),before_E=int(np.sum(m&before&(y==1))),
            before_U=int(np.sum(m&before&(y<0))),accepted_pairs=int(np.sum(m[::3]&accepted[:,0])),
            known_D=int(np.sum(m&(y==0))),known_E=int(np.sum(m&(y==1))),unknown=int(np.sum(m&(y<0))))
    return dict(labels=y,known=y>=0,source=source,split=split,original_split=raw_split,variant=variant,
                pair_ready=accepted[:,0],counts=counts)

def load_metadata(e,p):
    seal=read(e['input_pack_seal']['path']);assert seal['completed'] and seal['PASS']
    assert same_binding(seal['manifest'],e['input_pack_manifest'])
    verify(seal['QA']);assert read(seal['QA']['path'])['PASS'] is True
    manifest=read(e['input_pack_manifest']['path'])
    assert manifest['completed'] and manifest['base_rows']==960 and manifest['occurrences']==2880
    verify(manifest['entry'])
    assert same_binding(read(manifest['entry']['path'])['occurrences'],e['GT_ledger'])
    assert same_binding(manifest['base_input_lineage'],e['base_input_lineage'])
    verify(manifest['input_lineage']);lineage=rows(manifest['input_lineage']['path'])
    ledger=rows(e['GT_ledger']['path']);meta=join_metadata(lineage,ledger)
    assert meta['counts']==p['expected_counts'],'Root frozen counts must match before any optimizer step'
    return manifest,lineage,meta

def materialize_tensor_rows(base,lineage,base_binding):
    arrays={k:[] for k in ['p3_roi','rgb96','crop_valid96']};cache={}
    def get(ref,key,index):
        path=ref['path']
        if path not in cache:
            verify(ref);cache[path]=dict(sha256=ref['sha256'],arrays={})
        assert ref['sha256']==cache[path]['sha256']
        if key not in cache[path]['arrays']:
            with np.load(path,allow_pickle=False) as f:cache[path]['arrays'][key]=np.array(f[key],copy=True)
        return np.array(cache[path]['arrays'][key][index],copy=True)
    for r in lineage:
        ref=r['input_ref'];i=r['slot_index']
        if ref['kind']=='BASE_ROW':
            assert ref['slot_index']==i and same_binding(ref['base_lineage'],base_binding)
            refs=base[i]['before_input_refs']
            x={k:get(refs[k]['file'],refs[k]['key'],refs[k]['index']) for k in arrays}
        else:
            assert ref['kind']=='CHANGED_CHUNK' and r['full_rgb_changed'] is True
            x={k:get(ref['file'],k,ref['index']) for k in arrays}
        if r['full_rgb_changed']: assert ref['kind']=='CHANGED_CHUNK'
        if 'tensor_value_sha256' in r:
            for k in arrays:assert hashlib.sha256(np.ascontiguousarray(x[k]).tobytes()).hexdigest()==r['tensor_value_sha256'][k]
        for k in arrays: arrays[k].append(x[k])
    return {k:np.ascontiguousarray(np.stack(v)) for k,v in arrays.items()}

def load_inputs(e,manifest,lineage):
    base=rows(e['base_input_lineage']['path']);assert len(base)==960
    assert all(set(r['tensor_value_sha256'])=={'p3_roi','rgb96','crop_valid96'} for r in lineage)
    data=materialize_tensor_rows(base,lineage,e['base_input_lineage'])
    assert data['p3_roi'].shape==(2880,65,9,9) and data['p3_roi'].dtype==np.float16 and np.isfinite(data['p3_roi']).all()
    assert data['rgb96'].shape==(2880,3,96,96) and data['rgb96'].dtype==np.uint8
    assert data['crop_valid96'].shape==(2880,1,96,96) and np.isin(data['crop_valid96'],[0,1]).all()
    assert np.isin(data['p3_roi'][:,64],[0,1]).all()
    available=(data['p3_roi'][:,64,4,4]>0)&data['crop_valid96'][:,0,47:49,47:49].all((1,2))
    assert np.array_equal(available,[r['input_facts']['operator_query_available'] for r in lineage])
    return data,available

def make_pools(meta):
    pools=[np.flatnonzero((meta['split']=='TRAIN')&meta['known']&(meta['labels']==k)) for k in [0,1]]
    assert all(len(p)>0 for p in pools);return pools
def draw(pools,g):
    a=torch.cat([torch.as_tensor(p)[torch.randint(len(p),(32,),generator=g)] for p in pools])
    return a[torch.randperm(64,generator=g)].numpy()
def selector(logits,meta):
    loss=np.logaddexp(0.,logits.astype(np.float64))-meta['labels']*logits.astype(np.float64)
    strata={}
    for name,v,k in [('BEFORE_D','BEFORE',0),('BEFORE_E','BEFORE',1),('SHAM_D',VARIANTS[2],0),('COVER_E',VARIANTS[1],1)]:
        m=(meta['split']=='VALIDATION')&(meta['variant']==v)&(meta['labels']==k)
        assert m.any(),f'Empty prospective selector stratum {name}'
        strata[name]=dict(count=int(m.sum()),mean_BCE=float(loss[m].mean()))
    return float(np.mean([x['mean_BCE'] for x in strata.values()])),strata
def decode(z,available):return np.where(available,(z>=0).astype(np.int8),-1)
def cm_metrics(z,available,meta,mask):
    pred=decode(z,available);y=meta['labels'];known=meta['known']
    cm=np.array([[np.sum(mask&known&(y==k)&(pred==p)) for p in [0,1,-1]] for k in [0,1]],np.int64)
    def ratio(a,b):return float(a/b) if b else None
    return dict(matrix=cm.tolist(),rows=['D','E'],columns=['D','E','U'],D_recall=ratio(cm[0,0],cm[0].sum()),E_recall=ratio(cm[1,1],cm[1].sum()),
        D_precision=ratio(cm[0,0],cm[:,0].sum()),E_precision=ratio(cm[1,1],cm[:,1].sum()),known_U_rate=ratio(cm[:,2].sum(),cm.sum()),
        all_N=int(mask.sum()),known_N=int(cm.sum()),unknown_GT_N=int(np.sum(mask&~known)),unknown_GT_predictions={str(p):int(np.sum(mask&~known&(pred==p))) for p in [0,1,-1]})
def report(z,available,meta):
    results={};variant=meta['variant'];split=meta['split'];pred=decode(z,available).reshape(960,3)
    for s in ['TRAIN','VALIDATION']:
        m=split==s;pairmask=meta['pair_ready']&m[::3]
        triple=(pred==np.array([0,1,0])).all(1)
        n=int(pairmask.sum())
        results[s]=dict(all_known=cm_metrics(z,available,meta,m),by_variant={v:cm_metrics(z,available,meta,m&(variant==v)) for v in VARIANTS},
            accepted_pairs=n,all3_correct=int(np.sum(triple&pairmask)),all3_correct_rate=float(np.sum(triple&pairmask)/n) if n else None)
    val=results['VALIDATION'];before=val['by_variant']['BEFORE']
    checks={
        'natural_before_D_recall':dict(value=before['D_recall'],minimum=.85),
        'natural_before_E_recall':dict(value=before['E_recall'],minimum=.75),
        'sham_D_recall':dict(value=val['by_variant'][VARIANTS[2]]['D_recall'],minimum=.85),
        'cover_E_recall':dict(value=val['by_variant'][VARIANTS[1]]['E_recall'],minimum=.75),
        'accepted_VAL_pair_all3_correct_rate':dict(value=val['all3_correct_rate'],minimum=.75),
        'VAL_all_known_U_rate':dict(value=val['all_known']['known_U_rate'],maximum=.05)}
    for c in checks.values(): c['PASS']=c['value'] is not None and (c['value']>=c['minimum'] if 'minimum' in c else c['value']<=c['maximum'])
    return results,checks,all(c['PASS'] for c in checks.values())

def forward(model,data,ix,device):
    return model(*(torch.as_tensor(np.ascontiguousarray(data[k][ix]),device=device) for k in ['p3_roi','rgb96','crop_valid96']))
@torch.no_grad()
def predict(model,data,device):
    model.eval();zs=[];avs=[]
    for start in range(0,2880,128):
        out=forward(model,data,np.arange(start,min(start+128,2880)),device)
        zs.append(out['query_logit'].float().cpu().numpy());avs.append(out['query_available'].cpu().numpy())
    return np.concatenate(zs),np.concatenate(avs)
def checkpoint(model,opt,update,e):
    return dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},optimizer_state=opt.state_dict(),update=update,seed=194,
                parameter_count=26573,model_code=e['model_code'],protocol=e['protocol'],source_input_bindings=e)

def run(entry):
    e=read(entry);p=require_entry(e)
    assert torch.__version__=='2.1.1+cu118' and torch.cuda.is_available()
    RUN.mkdir(parents=True);write(RUN/'RUN_STARTED.json',dict(entry=bind(entry),trainer=bind(__file__),optimizer_updates=0,restart_forbidden_after_any_update=True))
    updates=0
    try:
        manifest,lineage,meta=load_metadata(e,p)
        if any(meta['counts'][s]['accepted_pairs']==0 for s in ['TRAIN','VALIDATION']):
            write(RUN/'NO_FIT_FEASIBILITY_ONLY.json',dict(completed=True,fit_executed=False,reason='No accepted controlled pairs in TRAIN or VAL',counts=meta['counts'],current_allowed=False));return
        # Assert nonempty prospective selector cohorts before tensor/model access.
        selector(np.zeros(2880,np.float32),meta)
        data,factual=load_inputs(e,manifest,lineage)
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        random.seed(194);np.random.seed(194);torch.manual_seed(194);torch.cuda.manual_seed_all(194)
        op=operator();device='cuda:0';model=op.QueryCenterScalar().to(device)
        assert sum(p.numel() for p in model.parameters())==26573
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        pools=make_pools(meta);generator=torch.Generator(device='cpu').manual_seed(194)
        torch.save(checkpoint(model,opt,0,e),RUN/'INITIAL_CHECKPOINT.pt')
        best=float('inf');best_update=None;best_state=None;started=time.monotonic()
        def evaluate(update):
            nonlocal best,best_update,best_state
            z,a=predict(model,data,device);assert np.array_equal(a,factual)
            value,strata=selector(z,meta)
            rec=dict(update=update,criterion_four_VAL_strata_BCE=value,VAL_strata=strata,known_unavailable=int(np.sum(meta['known']&(meta['split']=='VALIDATION')&~a)))
            with (RUN/'HISTORY.jsonl').open('a') as f:f.write(json.dumps(rec,allow_nan=False)+'\n')
            if value<best:
                best=value;best_update=update;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                torch.save(checkpoint(model,opt,update,e),RUN/'BEST_QUERY_CENTER_SCALAR.pt')
        evaluate(0)
        for update in range(1,2001):
            ix=draw(pools,generator);model.train();opt.zero_grad(set_to_none=True)
            out=forward(model,data,ix,device);target=torch.as_tensor(meta['labels'][ix],device=device);known=torch.as_tensor(meta['known'][ix],device=device)
            loss=op.known_binary_loss(out,target,known);assert torch.isfinite(loss['loss'])
            loss['loss'].backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(grad)
            opt.step();updates=update
            rec=dict(update=update,occurrence_indices=ix.tolist(),batch_D=int(np.sum(meta['labels'][ix]==0)),batch_E=int(np.sum(meta['labels'][ix]==1)),
                     loss=float(loss['loss'].detach()),available_known_count=int(loss['available_known_count']),known_unavailable_count=int(loss['known_unavailable_count']),gradient_norm_before_clip=float(grad))
            assert rec['batch_D']==rec['batch_E']==32 and np.all(meta['split'][ix]=='TRAIN') and meta['known'][ix].all()
            with (RUN/'TRAIN_UPDATES.jsonl').open('a') as f:f.write(json.dumps(rec,allow_nan=False)+'\n')
            if update%100==0:evaluate(update)
        torch.save(checkpoint(model,opt,2000,e),RUN/'FINAL_CHECKPOINT.pt')
        model.load_state_dict(best_state);z,a=predict(model,data,device);assert np.array_equal(a,factual)
        value,strata=selector(z,meta);assert value==best
        results,checks,passed=report(z,a,meta)
        label=meta['labels'];nll=(np.logaddexp(np.float32(0),z)-np.where(label>=0,label,0)*z).astype(np.float32)
        np.savez_compressed(RUN/'SELECTED_ALL2880_SOURCE_SCORES.npz',query_logit=z.astype(np.float32),query_available=a,prediction=decode(z,a),NLL_FP32=nll,
            candidate_known=meta['known'],scalar_target_D0_E1_Uminus1=label,source=meta['source'],split=meta['split'],original_split=meta['original_split'],variant=meta['variant'],
            occurrence_index=np.arange(2880),occurrence_id=np.array([r['occurrence_id'] for r in lineage]),base_row_index=np.arange(2880)//3,
            input_id=np.array([r['input_id'] for r in lineage]),native_slot_logical_only=np.array([r['native_slot_logical_only'] for r in lineage]),
            image_sha256=np.array([r['image_sha256'] for r in lineage]),original_image_sha256=np.array([r['original_image_sha256'] for r in lineage]),
            query_xy_raw=np.array([r['query_xy'] if r['query_xy'] is not None else [np.nan,np.nan] for r in lineage],np.float64),
            L_raw=np.array([r['L'] if r['L'] is not None else np.nan for r in lineage],np.float64),accepted_pair=meta['pair_ready'])
        write(RUN/'SOURCE_GATE_RESULTS.json',dict(completed=True,PASS=passed,all_gates_pass=passed,gates=checks,current_inference_authorized=False))
        write(RUN/'SOURCE_RESULTS.json',dict(completed=True,source_only=True,selected_update=best_update,selector_four_VAL_strata_BCE=value,VAL_strata=strata,
            counts=meta['counts'],metrics=results,all2880_preserved=True,NLL_unknown_diagnostic_only=True,native_expected_part_state='UNKNOWN',human_gold=False))
        write(RUN/'MODEL_FREEZE.json',dict(completed=True,source_only=True,model_code=e['model_code'],checkpoint=bind(RUN/'BEST_QUERY_CENTER_SCALAR.pt'),
            initial_checkpoint=bind(RUN/'INITIAL_CHECKPOINT.pt'),final_checkpoint=bind(RUN/'FINAL_CHECKPOINT.pt'),parameter_count=26573,selected_update=best_update,updates=2000,
            selector=SETTINGS['selector'],threshold_rawlogit=0,protocol=e['protocol'],entry=bind(entry),source_input_bindings=e,current_inference_authorized=False))
        write(RUN/'FIT_COMPLETE.json',dict(completed=True,PASS_execution=True,optimizer_updates=2000,selected_update=best_update,seconds=time.monotonic()-started,
            MODEL_FREEZE=bind(RUN/'MODEL_FREEZE.json'),source_results=bind(RUN/'SOURCE_RESULTS.json'),source_gates=bind(RUN/'SOURCE_GATE_RESULTS.json'),
            scores=bind(RUN/'SELECTED_ALL2880_SOURCE_SCORES.npz'),history=bind(RUN/'HISTORY.jsonl'),train_updates=bind(RUN/'TRAIN_UPDATES.jsonl'),new_current_PnP=0))
        print(json.dumps(read(RUN/'FIT_COMPLETE.json')))
    except BaseException:
        write(RUN/'FAILED_EXECUTION.json',dict(completed=False,optimizer_updates_completed=updates,exception=traceback.format_exc(),restart_automatically=False))
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--entry',type=Path,required=True)
    run(parser.parse_args().entry)
