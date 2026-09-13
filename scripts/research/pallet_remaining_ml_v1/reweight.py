"""Online example weights learned from a disjoint meta loss, bounded nine fits."""
import argparse
import copy
import math
import random
import subprocess
import sys
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from experiment import S,ROOT,RAW,DOC,P,read,write,sha,METHODS,SEEDS,check_gpu

BASE=RAW/'reweight'
HYP=dict(S.HYP)

def prepare():
    split=read(DOC/'SPLIT.json');members={i.frame_id:i for i in P.population().positive.items}
    bindings={}
    for name,records in [('synthetic',read(S.SYNTH)['synthetic']),('real',split['train']),('meta',split['meta_calibration'])]:
        folder=BASE/'dataset'/name
        (folder/'images').mkdir(parents=True,exist_ok=True);(folder/'labels').mkdir(exist_ok=True)
        rows=[]
        for j,r in enumerate(records):
            image=S.Path(r['image']) if name=='synthetic' else ROOT/r['image_path']
            source=S.Path(r['label']) if name=='synthetic' else ROOT/r['label_path']
            if name=='synthetic':value=source.read_text()
            else:
                with Image.open(image) as im:w,h=im.size
                value,_=S.export_target(P.E._legacy_forbidden_target(members[r['frame_id']]),w,h)
            stem=f'{j:04}_{image.name}';dest=folder/'images'/stem
            if not dest.exists():dest.symlink_to(image.resolve())
            label=folder/'labels'/S.Path(stem).with_suffix('.txt')
            if label.exists():assert label.read_text()==value
            else:label.write_text(value)
            rows.append(dict(image=str(image.relative_to(ROOT)),image_sha256=sha(image),source_label=str(source.relative_to(ROOT)),
                source_label_sha256=sha(source),exported_label=str(label.relative_to(ROOT)),exported_label_sha256=sha(label)))
        bindings[name]=rows
    write(DOC/'reweight/LABEL_BINDINGS.json',bindings)
    write(DOC/'LABEL_REVEAL_AUDIT.json',dict(revealed_frame_ids=[r['frame_id'] for k in ('train','meta_calibration') for r in split[k]],
        purpose='112 direct train plus62 meta supervision; evaluation145 excluded'))

class Samples(Dataset):
    def __init__(self,name,seed,epoch):
        self.name,self.seed,self.epoch=name,seed,epoch
        hyp=get_cfg(overrides=HYP)
        if name!='synthetic' or epoch>=7:hyp.mosaic=0.
        self.data=YOLODataset(img_path=str(BASE/'dataset'/name/'images'),imgsz=640,
            batch_size={'synthetic':24,'real':8,'meta':4}[name],augment=name!='meta',hyp=hyp,
            rect=False,cache=False,stride=32,pad=0.,data=S.DATA,task='pose',prefix=name+': ')
    def __len__(self):return len(self.data)
    def __getitem__(self,key):
        i,pos=key;seed=self.seed*10000000+self.epoch*100000+pos+{'synthetic':0,'real':50000,'meta':75000}[self.name]
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        return self.data[i]

def loader(name,seed,epoch):
    data=Samples(name,seed,epoch);size={'synthetic':24,'real':8,'meta':4}[name]
    rng=np.random.default_rng(seed*1000+epoch);order=[]
    while len(order)<size*30:order.extend(rng.permutation(len(data)).tolist())
    schedule=[[(order[j],j) for j in range(i,i+size)] for i in range(0,30*size,size)]
    return DataLoader(data,batch_sampler=schedule,num_workers=2,collate_fn=YOLODataset.collate_fn,pin_memory=True)

def slice_predictions(value,index):
    if torch.is_tensor(value):return value[index:index+1]
    if isinstance(value,dict):return {k:slice_predictions(v,index) for k,v in value.items()}
    if isinstance(value,list):return [slice_predictions(v,index) for v in value]
    if isinstance(value,tuple):return tuple(slice_predictions(v,index) for v in value)
    raise TypeError(type(value))

def single_target(batch,index):
    mask=batch['batch_idx']==index
    out={k:batch[k][mask] for k in ('cls','bboxes','keypoints')}
    out['batch_idx']=torch.zeros_like(batch['batch_idx'][mask])
    out['img']=batch['img'][index:index+1]
    return out

def proxy_parameters(model):
    return [(n,p) for n,p in model.named_parameters() if '.cv4_kpts.' in n or '.one2one_cv4_kpts.' in n]

def flat_grad(loss,params,retain_graph=False):
    grad=torch.autograd.grad(loss,params,retain_graph=retain_graph,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1) for p,g in zip(params,grad)])

def weights(method,losses,alignment):
    if method=='uniform':return torch.ones_like(losses)
    positive=(losses if method=='hard_loss' else alignment).detach().clamp_min(0)
    if float(positive.sum())<=1e-12:
        return torch.ones_like(positive) if method=='hard_loss' else torch.zeros_like(positive)
    return positive*(len(positive)/positive.sum())

def train(method,seed):
    out=BASE/'runs'/f'{method}_seed{seed}';out.mkdir(parents=True,exist_ok=True)
    assert not (out/'START.json').exists(),'Never silently repeat a started fit'
    check_gpu();torch.set_num_threads(4);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True,warn_only=True)
    m=S.load_model().cuda();m.args=get_cfg(overrides=HYP);m.train();m.model[-1].dfl.requires_grad_(False)
    for mod in m.modules():
        if isinstance(mod,torch.nn.modules.batchnorm._BatchNorm):mod.eval()
    buffers=S.tensor_sha(dict(m.named_buffers()));initial=S.tensor_sha(m.state_dict())
    criterion=m.init_criterion();opt=S.optimizer(m);named=proxy_parameters(m);params=[p for _,p in named]
    assert len(params)==12 and sum(p.numel() for p in params)==7452
    write(out/'START.json',dict(method=method,seed=seed,initial_state_sha256=initial,proxy_names=[n for n,_ in named],
        source_code_sha256=sha(S.Path(__file__)),split_sha256=sha(DOC/'SPLIT.json')))
    trace=[];started=time.time()
    for epoch in range(10):
        for syn,real,meta in zip(loader('synthetic',seed,epoch),loader('real',seed,epoch),loader('meta',seed,epoch)):
            step=len(trace);ratio=(1+math.cos(math.pi*epoch/10))/2*.99+.01
            for g in opt.param_groups:
                g['lr']=float(np.interp(step,[0,30],[.1 if g['is_bias'] else 0.,.002*ratio])) if step<=30 else .002*ratio
                g['momentum']=float(np.interp(step,[0,30],[.8,.937])) if step<=30 else .937
            record=dict(step=step,synthetic=S.batch_digest(syn),real=S.batch_digest(real),meta=S.batch_digest(meta))
            opt.zero_grad(set_to_none=True)
            syn=S.device_batch(syn);syn_loss=criterion(m(syn['img']),syn)[0].sum();syn_loss.backward()
            meta=S.device_batch(meta);meta_loss=criterion(m(meta['img']),meta)[0].sum()/4
            meta_grad=flat_grad(meta_loss,params)
            real=S.device_batch(real);pred=m(real['img'])
            losses=torch.stack([criterion(slice_predictions(pred,i),single_target(real,i))[0].sum() for i in range(8)])
            align=torch.stack([torch.dot(flat_grad(loss,params,True),meta_grad) for loss in losses])
            w=weights(method,losses,align)
            total=(w*losses).sum();assert torch.isfinite(total) and torch.isfinite(w).all() and torch.isfinite(align).all()
            total.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),10.,error_if_nonfinite=True);opt.step()
            record.update(losses=losses.detach().cpu().tolist(),alignment=align.cpu().tolist(),weights=w.cpu().tolist(),
                meta_loss=float(meta_loss.detach()),synthetic_loss=float(syn_loss.detach()))
            trace.append(record)
            if len(trace)%30==0:print(method,seed,'updates',len(trace),'seconds',round(time.time()-started),flush=True)
        criterion.update()
    assert len(trace)==300 and S.tensor_sha(dict(m.named_buffers()))==buffers
    saved=copy.deepcopy(m).cpu().eval();saved.args=vars(saved.args)
    if hasattr(saved,'criterion'):delattr(saved,'criterion')
    torch.save(dict(model=saved,ema=None,train_args=HYP,epoch=9,optimizer=None),out/'last.pt')
    write(out/'EXPOSURE.json',trace)
    write(out/'TRAINING_AUDIT.json',dict(status='PASS',optimizer_updates=300,method=method,seed=seed,
        init_state_sha256=initial,checkpoint_sha256=sha(out/'last.pt'),BN_buffers_frozen=True,
        proxy_parameters=7452,meta_direct_optimizer_updates=0,synthetic_slots=7200,real_slots=2400,meta_slots=1200,
        zero_real_weight_batches=sum(sum(r['weights'])==0 for r in trace),elapsed_seconds=time.time()-started))

def driver():
    for seed in SEEDS:
        for method in METHODS:
            out=BASE/'runs'/f'{method}_seed{seed}';out.mkdir(parents=True,exist_ok=True)
            if (out/'TRAINING_AUDIT.json').exists():continue
            assert not (out/'TRAIN.log').exists(),'Existing attempt needs explicit audit, no retry'
            with (out/'TRAIN.log').open('x') as log:
                code=subprocess.call([sys.executable,__file__,'train','--method',method,'--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT)
            print(method,seed,'exit',code,flush=True)
            if code:raise SystemExit(code)
    print('ALL9_REWEIGHT_FITS_COMPLETE',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','train','driver']);ap.add_argument('--method',choices=METHODS);ap.add_argument('--seed',type=int,choices=SEEDS);a=ap.parse_args()
    if a.phase=='train':train(a.method,a.seed)
    else:globals()[a.phase]()
