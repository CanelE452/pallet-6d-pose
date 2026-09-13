"""Retrospective session-separated AL; reveal selected GT only, train fixed budget."""
import argparse
import copy
import csv
import hashlib
import importlib
import importlib.util
import json
import math
import random
import subprocess
import sys
import time
from collections import Counter,defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader,Dataset
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset

spec=importlib.util.spec_from_file_location('acquisition_pilot',Path(__file__).with_name('run.py'))
AL=importlib.util.module_from_spec(spec);spec.loader.exec_module(AL)
from common.contracts import ROOT,R0,R0_SHA,sha,write,tensor_sha
from track_c.wiring import load_model
from track_c.train import HYP as OLD_HYP,DATA,optimizer,device_batch,batch_digest
from track_c.selection_diagnostic import P
from common.gpu_snapshot import snapshot

RAW=AL.RAW/'retrospective_v1';DOC=AL.DOC/'retrospective_v1'
METHODS=AL.METHODS
HYP=dict(OLD_HYP,epochs=10)
MANIFEST=ROOT/'challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json'
SYNTH=ROOT/'data/pallet/results/pallet_paper_contribution_screen_v1/C_geometry_preserving_da/dataset/BINDINGS.json'

def read(p):return json.loads(p.read_text())
def key(s):return hashlib.sha256(('AL_SIM_20260913:'+s).encode()).hexdigest()

def split_records(rows):
    groups=defaultdict(list)
    for r in rows:groups[r['capture_session']].append(r)
    strata=defaultdict(list)
    for g,rr in groups.items():
        assert len({r['object_type'] for r in rr})==1
        strata[rr[0]['object_type']].append(g)
    test=set()
    for object_type,gg in strata.items():
        assert len(gg)>=2
        test.update(sorted(gg,key=key)[:max(1,len(gg)//3)])
    return [r for r in rows if r['capture_session'] not in test],[r for r in rows if r['capture_session'] in test]

def setup():
    assert sha(R0)==R0_SHA
    assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()=='main'
    items=read(MANIFEST)['items'];assert len(items)==319
    # Resolve known aliases using exact original-image hashes, without annotation access.
    names={Path(r['image_path']).name for r in items};index=defaultdict(list)
    for line in subprocess.check_output(['rg','--files','--no-ignore','data/pallet/raw_data'],text=True).splitlines():
        p=ROOT/line
        if p.name in names and ('/rgb/' in line or '/wood/selected/' in line):index[p.name].append(p)
    rows=[];alias=defaultdict(set)
    for r in items:
        image=ROOT/r['image_path'];h=sha(image)
        matches=[p for p in index[image.name] if sha(p)==h]
        source={p.parent.parent.name if p.parent.name=='rgb' else p.parent.name for p in matches}
        assert len(source)<=1,(image,source)
        alias[r['session_id']].update(source)
        rows.append(dict(frame_id=r['frame_id'],image_path=r['image_path'],image_sha256=h,
            label_path=r['gt_v2_path'],manifest_session=r['session_id'],object_type=r['object_type'],
            paper_condition=r.get('domain','UNKNOWN'),source_matches=[str(p.relative_to(ROOT)) for p in matches]))
    parent={}
    def find(v):
        parent.setdefault(v,v)
        if parent[v]!=v:parent[v]=find(parent[v])
        return parent[v]
    for session,sources in alias.items():
        for source in sources:
            a,b=find('session:'+session),find('raw:'+source)
            parent[max(a,b)]=min(a,b)
    for r in rows:
        r['capture_session']=find('session:'+r['manifest_session'])
        stem=Path(r['image_path']).stem
        assert stem.isdigit()
        value=int(stem)
        r['selection_clock_kind']='timestamp_ns' if value>10**15 else 'frame_number_60frame_gap'
        # AL's common2e9 spacing corresponds to60frame indices; not an assumed FPS.
        r['timestamp_ns']=value if value>10**15 else value*(2_000_000_000//60+1)
    pool,test=split_records(rows)
    assert not {r['image_sha256'] for r in pool}&{r['image_sha256'] for r in test}
    assert not {r['capture_session'] for r in pool}&{r['capture_session'] for r in test}
    assert len({r['image_sha256'] for r in rows})==319
    negatives=P.population().negative.items
    assert not {r['image_sha256'] for r in rows}&{sha(ROOT/r.image) for r in negatives}
    bind={str(p.relative_to(ROOT)):sha(p) for p in (MANIFEST,R0,SYNTH,Path(AL.__file__),Path(P.E.__file__),
        ROOT/'scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py')}
    write(DOC/'SPLIT.json',dict(pool=pool,evaluation=test,raw_source_aliases={k:sorted(v) for k,v in alias.items()},
        image_hash_overlap=0,physical_or_manifest_session_overlap=0,
        session_provenance='Exact raw hashes where available; otherwise declared acquisition-session IDs. Not newly independent sessions.'))
    write(RAW/'pool/POOL.json',dict(eligible=pool,excluded=[]))
    write(DOC/'PROTOCOL_LOCK.json',dict(status='LOCKED_BEFORE_SELECTION_AND_TRAINING',
        start_main=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_bindings=bind,pool_frames=len(pool),evaluation_positive_frames=len(test),negative_frames=len(negatives),
        pool_sessions=dict(Counter(r['capture_session'] for r in pool)),evaluation_sessions=dict(Counter(r['capture_session'] for r in test)),
        methods=list(METHODS),selected_labels_per_method=30,selection_seed=20260913,student_seeds=[1,2,3],
        fits=12,updates_per_fit=300,total_optimizer_update_cap=3600,last_step_only=True,
        real_batch=8,synthetic_batch=24,synthetic_exposures_per_fit=7200,real_exposures_per_fit=2400,
        training_hyp=HYP,warmup_steps=30,epoch_update_count=30,epochs=10,
        model='All stock R0 parameters trainable except fixed DFL; no adapter',
        loss='Stock PoseModel.init_criterion: E2ELoss(PoseLoss26); GTv2 visibility1 remains supervised occluded, not old pseudo TRUE_IGNORE',
        target='Existing GT-v2 annotations and canonical box derived from eight corners; mixed manual/reconstructed provenance, not new independent sensor GT',
        annotation_access='Acquisition reads images/metadata only; after selection reveal union of selected pool labels; evaluation GT loaded only after all fits',
        schema_inspection_note='One historical eval_cad label schema inspected before split; no acquisition score or split uses its coordinates',
        label_export='Stock YOLO normalized9x3; finite in-frame visibility preserved; off-frame points visibility0 and coordinates clipped; bbox clipped to image',
        selection='Reuse locked fixed0.8 photometric instability and k-center controls; no performance-based changes',
        time_spacing='2seconds for timestamp names;60frame indices for video-frame names without claiming FPS',
        primary='Seed-mean pooled supervised kp median on four-arm common matched evaluation frames',
        signal_gate='Geometry-weighted improves primary vs BOTH random and diversity, >=2/3 seeds each; paired P90/gross20/Det nonworse; translation/yaw <=1.1x and pose coverage nonworse',
        evidence='RETROSPECTIVE_DEVELOPMENT; all319 historically consulted, not untouched confirmation',
        previous_97_frame_preview='Preserved, not used for training',negative_population_note='Fixed existing DEV_NEG2689; no independent-session claim for negatives'))
    print('Locked split:',len(pool),'pool,',len(test),'evaluation;12x300updates',flush=True)

def acquire():
    # Reuse the tested acquisition implementation in an isolated destination.
    split=read(DOC/'SPLIT.json');pooldir=RAW/'pool'
    AL.RAW=pooldir;AL.DOC=DOC/'acquisition'
    write(AL.DOC/'POOL_AUDIT.json',dict(source_bindings=read(DOC/'PROTOCOL_LOCK.json')['source_bindings']))
    write(AL.DOC/'PROTOCOL_LOCK.json',dict(preview_budget_per_method=30))
    if not (AL.DOC/'EXTRACTION_AUDIT.json').exists():AL.extract()
    AL.acquire()
    selected=read(AL.DOC/'PREVIEW_SELECTIONS.json')['selections']
    assert all(len(v)==30 for v in selected.values())
    write(DOC/'SELECTION_LOCK.json',dict(selections={m:[r['frame_id'] for r in rr] for m,rr in selected.items()},
        label_coordinates_read_by_acquisition=0,pool_image_sha256s=[r['image_sha256'] for r in split['pool']],
        acquisition_is_fixed_not_refit_per_student_seed=True))

def export_target(target,width,height):
    box=target.box_xyxy.copy();box[[0,2]]=np.clip(box[[0,2]],0,width);box[[1,3]]=np.clip(box[[1,3]],0,height)
    assert box[2]>box[0] and box[3]>box[1]
    xy=np.nan_to_num(target.keypoints_xy.copy());vis=target.visibility.copy()
    inside=np.isfinite(target.keypoints_xy).all(1)&(xy[:,0]>=0)&(xy[:,0]<width)&(xy[:,1]>=0)&(xy[:,1]<height)
    vis[~inside]=0;xy[:,0]=np.clip(xy[:,0]/width,0,1);xy[:,1]=np.clip(xy[:,1]/height,0,1)
    bbox=[(box[0]+box[2])/(2*width),(box[1]+box[3])/(2*height),(box[2]-box[0])/width,(box[3]-box[1])/height]
    points=np.c_[xy,vis];values=[0,*bbox,*points.flatten()]
    return ' '.join(format(float(v),'.12g') for v in values)+'\n',int(np.sum(target.keypoint_supervision_mask & ~inside))

def prepare():
    chosen=read(DOC/'SELECTION_LOCK.json')['selections'];split=read(DOC/'SPLIT.json')
    pool={r['frame_id']:r for r in split['pool']};evaluation={r['frame_id'] for r in split['evaluation']}
    union=set().union(*map(set,chosen.values()));assert union<=set(pool) and not union&evaluation
    members={i.frame_id:i for i in P.population().positive.items};targets={}
    for fid in sorted(union):targets[fid]=P.E._legacy_forbidden_target(members[fid])
    receipt={};syn=read(SYNTH)['synthetic']
    for name,records in [('synthetic',syn),*[(m,[pool[fid] for fid in sorted(ids)]) for m,ids in chosen.items()]]:
        folder=RAW/'dataset'/name
        (folder/'images').mkdir(parents=True,exist_ok=True);(folder/'labels').mkdir(exist_ok=True)
        rr=[]
        for j,r in enumerate(records):
            image=Path(r['image']) if name=='synthetic' else ROOT/r['image_path']
            basename=f'{j:04}_{image.name}';dest=folder/'images'/basename
            if not dest.exists():dest.symlink_to(image.resolve())
            label=folder/'labels'/Path(basename).with_suffix('.txt');masked=0
            if name=='synthetic':value=Path(r['label']).read_text();source=Path(r['label'])
            else:
                width,height=Image.open(image).size;value,masked=export_target(targets[r['frame_id']],width,height);source=ROOT/r['label_path']
            if label.exists():assert label.read_text()==value
            else:label.write_text(value)
            rr.append(dict(image=str(image.relative_to(ROOT)),image_sha256=sha(image),
                source_label=str(source.relative_to(ROOT)),source_label_sha256=sha(source),
                exported_label_sha256=sha(label),newly_offframe_masked=masked))
        receipt[name]=rr
    write(DOC/'LABEL_REVEAL_AUDIT.json',dict(status='PASS',selected_union=len(union),
        revealed_frame_ids=sorted(union),evaluation_labels_revealed=0,bindings=receipt,
        stock_visibility_semantics=True,real_labels_per_arm=30))
    print('Revealed selected existing GT only:',len(union),'frames',flush=True)

class Batches(Dataset):
    def __init__(self,name,seed,epoch):
        self.name,self.seed,self.epoch=name,seed,epoch
        hyp=get_cfg(overrides=HYP)
        if epoch>=7:hyp.mosaic=0.
        self.data=YOLODataset(img_path=str(RAW/'dataset'/name/'images'),imgsz=640,
            batch_size=24 if name=='synthetic' else 8,augment=True,hyp=hyp,rect=False,cache=False,
            stride=32,pad=0.,data=DATA,task='pose',prefix=name+': ')
    def __len__(self):return len(self.data)
    def __getitem__(self,key):
        idx,pos=key;s=self.seed*10000000+self.epoch*100000+pos+(0 if self.name=='synthetic' else 50000)
        random.seed(s);np.random.seed(s);torch.manual_seed(s)
        return self.data[idx]

def loader(name,seed,epoch):
    data=Batches(name,seed,epoch);batch=24 if name=='synthetic' else 8
    order=[];rng=np.random.default_rng(seed*1000+epoch)
    while len(order)<batch*30:order.extend(rng.permutation(len(data)).tolist())
    schedule=[[(order[j],j) for j in range(i,i+batch)] for i in range(0,batch*30,batch)]
    return DataLoader(data,batch_sampler=schedule,num_workers=2,collate_fn=YOLODataset.collate_fn,pin_memory=True)

def train(method,seed):
    torch.set_num_threads(4);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True,warn_only=True)
    out=RAW/f'runs/{method}_seed{seed}';out.mkdir(parents=True,exist_ok=True)
    assert not (out/'last.pt').exists() and not (out/'last_unverified.pt').exists(),'Never repeat or overwrite a fit'
    m=load_model().cuda();m.args=get_cfg(overrides=HYP);m.train();m.model[-1].dfl.requires_grad_(False)
    initial=tensor_sha(m.state_dict());criterion=m.init_criterion();opt=optimizer(m)
    assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in m.parameters() if p.requires_grad}
    trace=[];started=time.time()
    for epoch in range(10):
        for batch,real in zip(loader('synthetic',seed,epoch),loader(method,seed,epoch)):
            step=len(trace);ratio=(1+math.cos(math.pi*epoch/10))/2*.99+.01
            for g in opt.param_groups:
                g['lr']=float(np.interp(step,[0,30],[.1 if g['is_bias'] else 0.,.002*ratio])) if step<=30 else .002*ratio
                g['momentum']=float(np.interp(step,[0,30],[.8,.937])) if step<=30 else .937
            record=dict(step=step,synthetic=batch_digest(batch),real=batch_digest(real))
            opt.zero_grad(set_to_none=True);loss_value=0.
            for b in (batch,real):
                b=device_batch(b);loss=criterion(m(b['img']),b)[0].sum();assert torch.isfinite(loss)
                loss.backward();loss_value+=float(loss.detach())
            torch.nn.utils.clip_grad_norm_(m.parameters(),10.,error_if_nonfinite=True);opt.step();trace.append(record)
            if len(trace)%30==0:print(json.dumps(dict(method=method,seed=seed,updates=len(trace),loss=loss_value,elapsed_s=round(time.time()-started))),flush=True)
        criterion.update()
    saved=copy.deepcopy(m).cpu().eval();saved.args=vars(saved.args)
    if hasattr(saved,'criterion'):delattr(saved,'criterion')
    pending=out/'last_unverified.pt';torch.save(dict(model=saved,ema=None,train_args=HYP,epoch=9,optimizer=None),pending)
    write(out/'EXPOSURE.json',trace);assert len(trace)==300
    pending.rename(out/'last.pt')
    write(out/'TRAINING_AUDIT.json',dict(status='PASS',method=method,seed=seed,optimizer_updates=len(trace),
        init_state_sha256=initial,checkpoint_sha256=sha(out/'last.pt'),last_only=True,
        loss_class=type(criterion).__name__,visibility1_ignored=False,synthetic_exposure=7200,real_exposure=2400,
        trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad)))

def check_gpu(next_task):
    snapshot();gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
    if gpu:
        write(DOC/'RESOURCE_UNAVAILABLE.json',dict(status='NOT_RUN',next_task=next_task,processes=gpu));raise SystemExit('No waiting or foreign process mutation')

def driver():
    for seed in (1,2,3):
        for method in METHODS:
            name=f'{method}_seed{seed}';out=RAW/'runs'/name;out.mkdir(parents=True,exist_ok=True)
            if (out/'TRAINING_AUDIT.json').exists():continue
            check_gpu(name)
            with (out/'TRAIN.log').open('x') as log:
                p=subprocess.run([sys.executable,__file__,'train','--method',method,'--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT)
            print(name,'training exit',p.returncode,flush=True)
            if p.returncode:raise SystemExit(p.returncode)
    print('ALL12_FITS_COMPLETE; evaluation may begin',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['setup','acquire','prepare','train','driver'])
    ap.add_argument('--method',choices=METHODS);ap.add_argument('--seed',type=int,choices=[1,2,3]);a=ap.parse_args()
    if a.phase=='train':train(a.method,a.seed)
    else:globals()[a.phase]()
