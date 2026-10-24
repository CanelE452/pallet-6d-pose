"""Frozen source-head observability only; --prepare never opens tensor bodies.

Root --entry authorizes existing source features/checkpoints and fixed512 head
replay. No encoder, optimizer, training, query correction or private inputs.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse, hashlib, importlib.util, json, traceback
import numpy as np
import torch

H=Path(__file__).resolve().parent; V=H.parent; P=V.parent
PREV=P/'query_visibility_mr6d_source_supervision_bridge_20261024_v154'
POLICY=V/'ROOT_POLICY_BEFORE_FROZEN_PROBE_AND_INDEPENDENT_AUXILIARY_CONTROL.json'
POLICY_SHA='6de9504b7a875659e4d8702ea707ae14fbaff6ea73e788a33316a1b14426b63b'
CONTROL=PREV/'runs/C35_C36_SEPARATE_QUERY_FIXED2000/CONTROL_COMPLETE.json'
CONTROL_SHA='45279eeb0bdb6d21c73d796062a279516f187970134e8c9ffa983e3ce17d476c'
AUDIT=PREV/'independent_saved_fit_review/REVIEW.json'
AUDIT_SHA='48d5e6950325c3adefba3d2cdb08302c4169a8ec7a04c09453c062ac4f1e0cd3'
PROTOCOL=H/'PROTOCOL_BEFORE_FROZEN_SOURCE_HEAD_REPLAY.json'
OUT=V/'frozen_hidden_probe_v1'
ARMS=('C35_MATCHED_VISIBILITY_ONLY_CONTROL','C36_MATCHED_VISIBILITY_PLUS_ENTITY_AND_POINT_QUALITY')
NAMES=('OLD6144','NEW20480','PALLOX512','MR1840'); SIZES=(6144,20480,512,1840)
STAGES=('C32_INITIAL','C35_FINAL','C36_FINAL')
OUTPUTS=('external_occlusion','target_entity','coarse_point_quality')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def check(b):
    path=Path(b['path']).resolve()
    assert path.is_relative_to(P) and '_v112' not in str(path).lower()
    assert bind(path)==b,b['path'];return path
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def packet(b):
    with np.load(check(b),allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
def state_sha(model):
    d=hashlib.sha256()
    for k,t in sorted(model.state_dict().items()):
        a=t.detach().cpu().contiguous().numpy();d.update(k.encode()+str(a.dtype).encode()+str(a.shape).encode()+a.tobytes())
    return d.hexdigest()

def hidden_and_outputs(model,x,g):
    # Exact expression from the pinned V153 SeparatedQueryHead.forward.
    T,Q=x[:,:384],x[:,384:768];C,Pv=x[:,1152:1536],x[:,1536:1920]
    visual=torch.cat((x,torch.abs(T-C),T*C,torch.abs(Q-Pv),Q*Pv),1)
    hidden=model.head[1](model.head[0](visual)+model.geometry(g))
    logits=torch.stack((model.head[2](hidden).squeeze(-1),model.entity(hidden).squeeze(-1),model.point_quality(hidden).squeeze(-1)),1)
    return hidden,logits

def covariance_summary(values):
    """Unbiased centered F64 covariance; entropy rank is a descriptor, not capacity."""
    x=np.asarray(values,np.float64);n,d=x.shape
    if not n:return dict(N=0,defined=False),np.full((d,d),np.nan),np.full(d,np.nan)
    mean=x.mean(0);centered=x-mean
    base=dict(N=n,defined=n>=2,mean=mean.tolist(),positive_fraction=(x>0).mean(0).tolist(),
              exact_nonzero_fraction=(x!=0).mean(0).tolist(),units_ever_nonzero=int(np.any(x!=0,axis=0).sum()),
              units_nonconstant_exact=int(np.any(x!=x[:1],axis=0).sum()))
    if n<2:return base,np.full((d,d),np.nan),np.full(d,np.nan)
    cov=centered.T@centered/(n-1);cov=(cov+cov.T)*.5
    ev=np.linalg.eigvalsh(cov)[::-1];tol=max(float(ev[0]),0.)*d*np.finfo(np.float64).eps
    assert ev.min()>=-max(tol,1e-12),ev.min()
    ev=np.maximum(ev,0.);total=float(ev.sum());p=ev/total if total else np.zeros(d)
    positive=p[p>0]
    base.update(total_variance=total,std=np.sqrt(np.maximum(np.diag(cov),0.)).tolist(),
        spectrum=ev.tolist(),numerical_rank_tolerance=tol,numerical_rank=int((ev>tol).sum()),
        entropy_effective_rank=float(np.exp(-(positive*np.log(positive)).sum())) if total else 0.,
        participation_effective_rank=float(1/(p@p)) if total else 0.,
        top_eigenvalue_fractions={str(k):float(p[:k].sum()) if total else None for k in (1,5,10,20)},
        zero_total_variance=not bool(total))
    return base,cov,ev

def head_cosine(model):
    w=np.stack([model.head[2].weight.detach().cpu().numpy()[0],model.entity.weight.detach().cpu().numpy()[0],model.point_quality.weight.detach().cpu().numpy()[0]]).astype(np.float64)
    b=np.array([model.head[2].bias.item(),model.entity.bias.item(),model.point_quality.bias.item()],np.float64)
    norms=np.linalg.norm(w,axis=1);valid=(norms[:,None]>0)&(norms[None,:]>0)
    cosine=np.full((3,3),np.nan);np.divide(w@w.T,norms[:,None]*norms[None,:],out=cosine,where=valid)
    return w,b,cosine,valid

def generated():
    torch.set_num_threads(1);torch.manual_seed(1550)
    class Invented(torch.nn.Module):
        def __init__(self):
            super().__init__();self.head=torch.nn.Sequential(torch.nn.Linear(3464,64),torch.nn.GELU(),torch.nn.Linear(64,1));self.geometry=torch.nn.Linear(42,64,bias=False);self.entity=torch.nn.Linear(64,1);self.point_quality=torch.nn.Linear(64,1)
    m=Invented().float();x=torch.randn(7,1928);g=torch.randn(7,42)
    h,z=hidden_and_outputs(m,x,g)
    t,q,c,pv=x[:,:384],x[:,384:768],x[:,1152:1536],x[:,1536:1920]
    visual=torch.cat([x,abs(t-c),t*c,abs(q-pv),q*pv],1)
    hh=torch.nn.functional.gelu(torch.nn.functional.linear(visual,m.head[0].weight,m.head[0].bias)+torch.nn.functional.linear(g,m.geometry.weight))
    assert torch.equal(h,hh)
    assert torch.equal(z[:,0],torch.nn.functional.linear(hh,m.head[2].weight,m.head[2].bias).flatten())
    fixture=np.array([[0.,1.],[2.,3.],[4.,5.]])
    s,cov,ev=covariance_summary(fixture);assert np.allclose(cov,np.cov(fixture,rowvar=False,ddof=1))
    assert np.allclose(ev,[8.,0.]) and np.isclose(s['entropy_effective_rank'],1.) and s['numerical_rank']==1
    s,_,_=covariance_summary(np.ones((4,3)));assert s['zero_total_variance'] and s['entropy_effective_rank']==0 and s['units_nonconstant_exact']==0
    assert not covariance_summary(np.ones((1,3)))[0]['defined'] and not covariance_summary(np.empty((0,3)))[0]['defined']
    torch.nn.init.zeros_(m.entity.weight);_,_,cs,valid=head_cosine(m);assert not valid[1].any() and np.isnan(cs[1]).all()
    assert not exact(np.array([0.],np.float32),np.array([-0.],np.float32))
    return dict(completed=True,PASS=True,invented_CPU_only=True,hidden_output_algebra=True,centered_covariance_ddof1_rank1_constant_empty_singleton=True,zero_head_cosine_undefined=True,signed_zero_byte_guard=True,actual_source_tensor_or_model_body_reads=0)

def prepare():
    assert sha(POLICY)==POLICY_SHA and sha(CONTROL)==CONTROL_SHA and sha(AUDIT)==AUDIT_SHA
    c=read(CONTROL);a=read(AUDIT);p=read(check(c['protocol']));old=read(check(p['parent_protocol']))
    assert c['completed'] and c['PASS'] and c['selected_model'] is None and a['completed'] and a['PASS']
    assert a['control_complete']==bind(CONTROL)
    features=[b['features'] for b in old['native']]+[p['Pallox_features'],p['MR_features']]
    labels=[b['GT'] for b in old['native']]+[old['public_labels'],p['source_labels']]
    output_lookup={str(Path(b['path']).resolve()):b for b in a['outputs']}
    stages=[]
    for stage,arm,which in ((STAGES[0],ARMS[0],'INITIAL'),(STAGES[1],ARMS[0],'FINAL'),(STAGES[2],ARMS[1],'FINAL')):
        fit=read(check(c['arms'][arm]['fit']));result=read(check(fit['source_results']));directory=Path(c['arms'][arm]['fit']['path']).parent
        initial=read(directory/'INITIAL_PARITY.json')
        stages.append(dict(name=stage,arm=arm,checkpoint=p['anchor_checkpoint'] if which=='INITIAL' else result['checkpoint'],
            expected_state_sha=initial['initial_state'] if which=='INITIAL' else result['final_state'],
            saved_outputs=[output_lookup[str((directory/f'{which}_{name}_OUTPUTS.npz').resolve())] for name in NAMES],
            source_fit=c['arms'][arm]['fit'],source_result=fit['source_results'],freeze=fit['model_freeze'],initial_parity=bind(directory/'INITIAL_PARITY.json')))
    checks=generated();write(H/'GENERATED_CPU_CHECKS.json',checks)
    protocol=dict(completed=True,prepared_only=True,code=bind(__file__),policy=bind(POLICY),control_complete=bind(CONTROL),independent_saved_audit=bind(AUDIT),
        parent_protocol=c['protocol'],parent_code=p['code'],head=p['head'],features=features,feature_seals=[b['feature_seal'] for b in old['native']]+[p['Pallox_feature_seal'],p['MR_feature_seal']],
        labels=labels,stages=stages,runtime=p['runtime'],generated=bind(H/'GENERATED_CPU_CHECKS.json'),datasets=list(NAMES),sizes=list(SIZES),rows=28976,
        forward_contract='Original full-pack constructed flatnonzero order, fixed512 chunks, FP32 CUDA TF32false, G42 allzero. Hidden expression copied literally from pinned forward; all3 logits byteexact to original forward and saved outputs before any summary.',
        group_contract='ALL28976, each dataset, and MR each original source split; for each full-constructed/inputvalid and all/knownD/knownE, retaining zero-count groups.',
        covariance_contract='F64 CPU centered sample covariance ddof1; N<2 undefined, constant variance rank0; eigvalsh clipped only numerical negative roundoff; entropy and participation ranks, top1/5/10/20 fractions.',
        unit_contract='GELU positive fraction and exact nonzero/nonconstant counts. These are descriptive, not ReLU dead-unit tests.',
        interpretation='Source empirical descriptors only. No intrinsic model capacity, causal auxiliary-interference, generalization, human-gold, threshold, or model-selection proof.',
        prepared_actual_features_checkpoints_scores_GT_or_models_read=False,selected_model=None,source_only=True)
    write(PROTOCOL,protocol)
    template=dict(approved_frozen_source_head_probe=False,code=protocol['code'],protocol=bind(PROTOCOL),policy=protocol['policy'],source_only=True,
        private_current223=False,V112=False,encoder_R0=False,training_optimizer=False,PnP=False,q_L_change=False,model_selection=False,automatic_retry=False)
    write(H/'ROOT_FROZEN_PROBE_ENTRY_TEMPLATE.json',template)
    ready=dict(completed=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),generated=protocol['generated'],template=bind(H/'ROOT_FROZEN_PROBE_ENTRY_TEMPLATE.json'),actual_source_tensor_or_model_body_reads=0)
    write(H/'PREPARATION_READY.json',ready);print(json.dumps(ready),flush=True)

def group_masks(dataset,split,known):
    bases=[('ALL28976',np.ones(len(dataset),bool))]+[(name,dataset==name) for name in NAMES]
    bases += [('MR1840::'+fold,(dataset=='MR1840')&(split==fold)) for fold in ('SOURCE_DEVELOPMENT','SOURCE_VALIDATION')]
    return [(name+'::'+truth,mask & (known==y) if y is not None else mask) for name,mask in bases for truth,y in (('ALL_STATES',None),('KNOWN_D',0),('KNOWN_E',1))]

def run(entry):
    e=read(entry);assert e['approved_frozen_source_head_probe'] and e['source_only']
    assert all(e[k] is False for k in ('private_current223','V112','encoder_R0','training_optimizer','PnP','q_L_change','model_selection','automatic_retry'))
    assert e['code']==bind(__file__) and e['protocol']==bind(PROTOCOL) and e['policy']==bind(POLICY) and not OUT.exists()
    p=read(PROTOCOL);assert p['code']==e['code'] and p['policy']==e['policy'] and sha(POLICY)==POLICY_SHA
    assert Path(sys.executable).resolve()==Path(p['runtime']['executable']).resolve()
    assert torch.__version__==p['runtime']['torch'] and np.__version__==p['runtime']['numpy'] and torch.cuda.is_available()
    for k in ('control_complete','independent_saved_audit','parent_protocol','parent_code','head','generated'):check(p[k])
    assert read(check(p['control_complete']))['selected_model'] is None
    for b in p['feature_seals']+p['features']+p['labels']:check(b)
    packs=[packet(b) for b in p['features']];labels=[];splits=[]
    for i,(b,n) in enumerate(zip(p['labels'],SIZES)):
        rr=lines(check(b))
        if i<2:rr=[r for r in rr if r['convention']=='CF_PRIMARY_DECLARED'];state=[r['teacher_state'] for r in rr]
        elif i==2:state=[r['semantic_state'] for r in rr]
        else:state=[r['role_state'] for r in rr]
        assert len(rr)==n
        for j,r in enumerate(rr):
            if i<2:
                assert r['scene_index']==j//8 and r['native_slot']==j%8
                assert r['query_id']==str(packs[i]['occurrence_id'][j])
            else:
                assert r['global_index']==j and r['native_slot']==j%8
                assert r['id']+':native'+str(j%8)==str(packs[i]['query_id'][j])
        fold=np.array([r['split'] for r in rr]);splits.append(fold)
        if 'split' in packs[i]:assert np.array_equal(fold,packs[i]['split'])
        labels.append(np.where(np.array(state)=='D',0,np.where(np.array(state)=='E',1,-1)).astype(np.int8))
    for a,n in zip(packs,SIZES):
        assert a['features'].shape==(n,1928) and a['features'].dtype==np.float32 and np.isfinite(a['features']).all()
        assert np.array_equal(a['global_index'],np.arange(n))
        av=a['query_available'];co=a.get('feature_constructed',np.ones(n,bool));assert av.dtype==co.dtype==bool and not (av&~co).any()
    dataset=np.concatenate([np.full(n,name) for name,n in zip(NAMES,SIZES)])
    split=np.concatenate(splits);known=np.concatenate(labels)
    av=np.concatenate([a['query_available'] for a in packs]);co=np.concatenate([a.get('feature_constructed',np.ones(n,bool)) for a,n in zip(packs,SIZES)])
    groups=group_masks(dataset,split,known)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    spec=importlib.util.spec_from_file_location('v155_fixed_separated_head',check(p['head']));head=importlib.util.module_from_spec(spec);spec.loader.exec_module(head)
    OUT.mkdir();write(OUT/'STARTED.json',dict(completed=False,entry=bind(entry),protocol=bind(PROTOCOL),automatic_retry=False))
    try:
        summaries={};artifacts=[];parities={}
        for st in p['stages']:
            for key in ('source_fit','source_result','freeze','initial_parity'):check(st[key])
            saved=[packet(b) for b in st['saved_outputs']]
            for a,s in zip(packs,saved):
                keys=set(a)-{'features'};assert keys==set(s)-{'raw_three_logits','primary_visibility_prediction','secondary_entity_gated_visibility'}
                for key in keys:assert exact(a[key],s[key]),(st['name'],key)
            torch.manual_seed(1540);torch.cuda.manual_seed_all(1540);model=head.SeparatedQueryHead().cuda().float()
            payload=torch.load(check(st['checkpoint']),map_location='cpu',weights_only=True)
            if st['name']=='C32_INITIAL':assert payload['update']==8000;model.load_visibility_state(payload['state_dict'])
            else:
                assert payload['update']==2000 and payload['parent_C32_update']==8000 and payload['arm']==st['arm']
                model.load_state_dict(payload['state_dict'],strict=True)
            del payload
            model.eval();model.requires_grad_(False)
            assert not model.geometry.weight.any() and state_sha(model)==st['expected_state_sha']
            hidden_all=[];raw_all=[]
            with torch.inference_mode():
                for a,s in zip(packs,saved):
                    n=len(a['features']);hidden=np.full((n,64),np.nan,np.float32);raw=np.full((n,3),np.nan,np.float32)
                    ix=np.flatnonzero(a.get('feature_constructed',np.ones(n,bool)))
                    x=torch.as_tensor(a['features'],device='cuda');z42=torch.zeros((512,42),device='cuda')
                    for lo in range(0,len(ix),512):
                        ii=ix[lo:lo+512];h,z=hidden_and_outputs(model,x[ii],z42[:len(ii)])
                        original=model(x[ii],z42[:len(ii)]);original=torch.stack([original[k] for k in OUTPUTS],1)
                        assert exact(z.cpu().numpy(),original.cpu().numpy())
                        hidden[ii]=h.cpu().numpy();raw[ii]=z.cpu().numpy()
                    assert exact(raw,s['raw_three_logits']),(st['name'],n,'saved raw parity')
                    assert np.array_equal(s['primary_visibility_prediction'],np.where(a['query_available'],(raw[:,0]>=0).astype(np.int8),-1))
                    hidden_all.append(hidden);raw_all.append(raw)
            assert state_sha(model)==st['expected_state_sha']
            # No summary or tensor output is emitted until all four dataset parity checks pass.
            hidden=np.concatenate(hidden_all);raw=np.concatenate(raw_all);w,b,cosine,cosvalid=head_cosine(model)
            result=dict(state_sha=st['expected_state_sha'],saved_all3_logits_byteexact=True,original_forward_byteexact=True,
                output_head_cosine=[[float(x) if np.isfinite(x) else None for x in row] for row in cosine],
                output_head_cosine_defined=cosvalid.tolist(),output_head_norms=np.linalg.norm(w,axis=1).tolist(),output_head_biases=b.tolist(),groups={})
            arrays=dict(hidden64=hidden,raw_three_logits=raw,dataset=dataset,split=split,visibility_truth=known,query_available=av,feature_constructed=co,
                dataset_local_global_index=np.concatenate([a['global_index'] for a in packs]),
                query_id=np.concatenate([a['query_id'] if 'query_id' in a else a['occurrence_id'] for a in packs]),
                query_xy=np.concatenate([a['query_xy'] for a in packs]),L=np.concatenate([a['L'] for a in packs]),
                output_weights=w,output_bias=b,output_weight_cosine=cosine,output_weight_cosine_defined=cosvalid)
            gi=0
            for base,mask in groups:
                for scope,eligible in (('CONSTRUCTED',co),('INPUTVALID',av)):
                    name=base+'::'+scope;m=mask&eligible
                    hs,hc,he=covariance_summary(hidden[m]);ls,lc,le=covariance_summary(raw[m])
                    result['groups'][name]=dict(hidden64=hs,logits3=ls,all_original_group_rows=int(mask.sum()),excluded_by_scope=int((mask&~eligible).sum()))
                    arrays[f'g{gi:03d}_hidden_covariance']=hc;arrays[f'g{gi:03d}_hidden_eigenvalues']=he
                    arrays[f'g{gi:03d}_logit_covariance']=lc;arrays[f'g{gi:03d}_logit_eigenvalues']=le;gi+=1
            arrays['group_names']=np.array(list(result['groups']))
            path=OUT/(st['name']+'_HIDDEN64_SOURCE_DIAGNOSTICS.npz');np.savez_compressed(path,**arrays);artifacts.append(bind(path))
            summaries[st['name']]=result;parities[st['name']]=dict(PASS=True,rows=28976,constructed=int(co.sum()),inputvalid=int(av.sum()),state_unchanged=True)
            del model;torch.cuda.empty_cache()
        summary=OUT/'SCALAR_SUMMARIES.json';write(summary,dict(completed=True,PASS=True,stages=summaries,interpretation=p['interpretation'],selected_model=None,fully_solved=False))
        receipt=dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),entry=bind(entry),control_complete=p['control_complete'],summary=bind(summary),outputs=artifacts,
            parities=parities,all28976_original_rows_retained=True,source_only=True,selected_model=None,fully_solved=False,
            encoder_R0_optimizer_training_PnP_query_correction_private_calls=0,labels_unchanged=True,model_states_unchanged=True,
            numerical_scope='Forward frozen source heads only. Empirical covariance/effective-rank and weight-angle descriptors do not identify cause or intrinsic capacity.')
        write(OUT/'PROBE_COMPLETE.json',receipt);print(json.dumps(receipt),flush=True)
    except BaseException:
        write(OUT/'FAILED_EXECUTION.json',dict(completed=False,automatic_retry=False,traceback=traceback.format_exc()));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--prepare',action='store_true');group.add_argument('--entry',type=Path)
    args=parser.parse_args();prepare() if args.prepare else run(args.entry)
