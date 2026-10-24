"""One preregistered conditional semantic-cue pilot. Never reads current data."""
import hashlib,json,os,random,sys,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from torch.nn import functional as F
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'model'))
from cue_visibility_head import CueVisibilityHead,uncertainty_heatmap_target,semantic_cue_loss
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))

def main():
    prot=json.loads((R/'FIT_PROTOCOL.json').read_text())
    assert prot['updates']==1000 and prot['validation_every']==50 and prot['seed']==20261023
    for b in prot['bindings']:assert sha(b['path'])==b['sha256']
    out=R/'runs/CUE_ATTENTION_V3';out.mkdir(parents=True,exist_ok=False)
    write(out/'FIT_STARTED.json',dict(protocol=bind(R/'FIT_PROTOCOL.json'),code=bind(__file__),current_read=False,PnP=False,updates=0))
    torch.set_num_threads(1);torch.manual_seed(20261023);torch.cuda.manual_seed_all(20261023)
    np.random.seed(20261023);random.seed(20261023)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    rows=[json.loads(l) for l in (R/'source_cache/GT_ONLY.jsonl').read_text().splitlines()]
    d=np.load(R/'source_cache/MODEL_INPUT_ONLY.npz',allow_pickle=False);x=np.load(R/'source_cache/P3_ROIS.npz',allow_pickle=False)['x']
    arrays=dict(p3_roi=x,rgb96=d['rgb96'],crop_valid96=d['crop_valid96'],query_xy=d['query_xy'].astype('float32'),box_xyxy=d['box_xyxy'].astype('float32'),side_L=d['L'].astype('float32'))
    inputs={k:torch.from_numpy(np.ascontiguousarray(v)).to('cuda') for k,v in arrays.items()}
    inputs['p3_roi']=inputs['p3_roi'].float()
    y=np.array([{'D':1,'E':0,'U':-1}[r['training_state']] for r in rows],np.int64)
    src=np.array([0 if r['source']=='SYNTH' else 1 for r in rows],np.int64)
    full=np.array([r['crop_availability']=='FULL' for r in rows])
    teachers={k:torch.as_tensor(v,device='cuda') for k,v in dict(class_target=y,source_index=src,full_available=full,input_valid=d['query_valid']).items()}
    heat=np.zeros((len(rows),9,9),np.float32)
    pos=np.flatnonzero(y==1)
    heat[pos]=uncertainty_heatmap_target(torch.tensor([rows[i]['uncertainty_bbox_xyxy'] for i in pos]),torch.tensor(d['query_xy'][pos]),torch.tensor(d['L'][pos])).numpy()
    teachers['heatmap_target']=torch.tensor(heat,device='cuda')
    pools={(s,c):np.array([i for i,r in enumerate(rows) if r['split']=='TRAIN' and src[i]==s and y[i]==c]) for s in [0,1] for c in [0,1]}
    assert all(len(v)>0 for v in pools.values())
    val=np.array([i for i,r in enumerate(rows) if r['split']=='VALIDATION' and y[i]>=0])
    assert len(val)==40 and {rows[i]['source'] for i in val}=={'YCB'}
    valgroups={}
    for i in val:valgroups.setdefault((rows[i]['source'],rows[i]['cue_refID']),[]).append(i)
    assert len(valgroups)==3 and all({y[i] for i in ids}=={0,1} for ids in valgroups.values())
    model=CueVisibilityHead().cuda();assert sum(p.numel() for p in model.parameters())==40682
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    rng=np.random.default_rng(20261023);best=float('inf');beststep=None;history=[];start=time.monotonic()
    def forward(ids):return model(**{k:v[ids] for k,v in inputs.items()})
    def score_val():
        model.eval()
        with torch.no_grad():z=forward(val)['logit_D_given_DE'].double().cpu().numpy()
        zd=dict(zip(val,z));bysource={}
        for (source,cue),ids in valgroups.items():
            posloss=[float(np.logaddexp(0,-zd[i])) for i in ids if y[i]==1]
            negloss=[float(np.logaddexp(0,zd[i])) for i in ids if y[i]==0]
            bysource.setdefault(source,[]).append(.5*np.mean(posloss)+.5*np.mean(negloss))
        return float(np.mean([np.mean(v) for v in bysource.values()]))
    for update in range(1,1001):
        ids=np.concatenate([rng.choice(pools[(s,c)],32,replace=True) for s in [0,1] for c in [0,1]])
        model.train();opt.zero_grad(set_to_none=True)
        result=semantic_cue_loss(forward(ids),**{k:v[ids] for k,v in teachers.items()},heatmap_weight=.25)
        assert torch.isfinite(result['loss']);result['loss'].backward();opt.step()
        if update==1:write(out/'FIRST_UPDATE.json',dict(update=1,loss=float(result['loss'].detach()),current_read=False))
        if update%50==0:
            value=score_val();saved=value<best
            if saved:
                best=value;beststep=update
                torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},out/'BEST_CUE_HEAD.pt')
            rec=dict(update=update,train_loss=float(result['loss'].detach()),binary_BCE=float(result['binary_BCE'].detach()),D_heatmap_CE=float(result['D_heatmap_CE'].detach()),VAL_balanced_source_cue_NLL=value,best_step=beststep,saved=saved,seconds=time.monotonic()-start)
            history.append(rec);print(json.dumps(rec),flush=True)
            write(out/'HISTORY.json',history)
    model.load_state_dict(torch.load(out/'BEST_CUE_HEAD.pt',map_location='cuda',weights_only=True));model.eval().requires_grad_(False)
    freeze=dict(completed=True,optimizer_updates=1000,best_update=beststep,best_VAL_balanced_source_cue_NLL=best,
      threshold_probability_D=.5,threshold_logit=0.,tie_is_D=True,checkpoint=bind(out/'BEST_CUE_HEAD.pt'),model_code=bind(R/'model/cue_visibility_head.py'),trainer_code=bind(__file__),protocol=bind(R/'FIT_PROTOCOL.json'),
      parameter_count=40682,current_used_for_fit_selection=False,current_used_for_threshold=False,PnP_calls=0,
      native8_gold=False,source_teacher='conditional_AI_visible_reference_plus_known_alpha_intervention',
      semantic_U_detector=False,self_occlusion_supervision=False,seconds=time.monotonic()-start)
    write(out/'MODEL_FREEZE.json',freeze)
    # Predictions are sealed before any source table/GT interpretation.
    zs=[];hs=[];ats=[]
    with torch.no_grad():
      for i in range(0,len(rows),128):
        pred=forward(np.arange(i,min(i+128,len(rows))))
        zs.append(pred['logit_D_given_DE'].cpu().numpy());hs.append(pred['cue_heatmap_logits'].cpu().numpy());ats.append(pred['cue_attention_weights'].cpu().numpy())
    np.savez_compressed(out/'SOURCE_SCORES.npz',logit=np.concatenate(zs),heatmap=np.concatenate(hs),attention=np.concatenate(ats))
    # Frozen v61 uses the exact same newly sampled full-image P3/query/L.
    sys.path.insert(0,str(R.parents[1]/'external_ray_proxy_head_20261022_v60'))
    from head_and_roi import ProxyHead
    v61=R.parents[1]/'target_presence_classifier_20261022_v61/BEST_PRESENCE_HEAD.pt'
    assert sha(v61)=='c75e04c41b8097ff63c2f134f76e12ec560603e7bcfe585446fe6042e31e4387'
    old=ProxyHead().cuda();old.load_state_dict(torch.load(v61,map_location='cuda',weights_only=True));old.eval().requires_grad_(False)
    before={k:v.detach().clone() for k,v in old.state_dict().items()};zz=[]
    with torch.no_grad():
      for i in range(0,len(rows),128):zz.append(old(inputs['p3_roi'][i:i+128]).cpu().numpy())
    assert all(torch.equal(v,old.state_dict()[k]) for k,v in before.items())
    np.savez_compressed(out/'V61_SOURCE_SCORES.npz',logit=np.concatenate(zz))
    write(out/'FIT_COMPLETE.json',dict(completed=True,optimizer_updates=1000,best_update=beststep,freeze=bind(out/'MODEL_FREEZE.json'),source_scores=bind(out/'SOURCE_SCORES.npz'),v61_scores=bind(out/'V61_SOURCE_SCORES.npz'),v61_checkpoint=bind(v61),v61_state_unchanged=True,current_inputs=0,PnP_calls=0))
    print(json.dumps(freeze),flush=True)

if __name__=='__main__':main()
