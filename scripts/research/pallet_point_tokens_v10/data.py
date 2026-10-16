"""Frozen observations and separately stored, camera-order point supervision."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment

KIT=Path(__file__).resolve().parent
ROOT=KIT.parents[2]
V9=ROOT/'scripts/research/pallet_point_attention_v9'
sys.path.insert(0,str(V9))
from model import GuidedDHT,point_maps
from prepare import Split,sha,EXPORT

RUN=ROOT/'data/pallet/results/pallet_point_tokens_v10'
PARENT=ROOT/'data/pallet/results/pallet_point_attention_v9'
V6=ROOT/'data/pallet/results/pallet_point_line_decisive_v6'


def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def observations(split,device='cpu'):
    data=Split(EXPORT/f'{split}.json');pr=read(PARENT/f'points_{split}.json')['rows']
    assert [r['id'] for r in pr]==[r['id'] for r in data.records]
    batches=[]
    for first in range(0,len(pr),32):
        O,_,_=data.batch(list(range(first,min(first+32,len(pr)))),'cpu',targets=False);batches.append(O)
    O={k:torch.cat([b[k] for b in batches]).to(device) for k in batches[0]}
    O['points']=torch.tensor([r['points'] for r in pr],dtype=torch.float32,device=device)
    return data,O


def donors(data,split):
    ids=sorted(range(len(data.records)),key=lambda i:hashlib.sha256(('v10:'+split+':'+data.records[i]['id']).encode()).digest())
    out=[0]*len(ids)
    for j,i in enumerate(ids):out[i]=ids[(j+1)%len(ids)]
    return out


def line_inputs(split,seed,O,donor=False):
    cache=np.load(RUN/f'lines_{split}_seed{seed}.npz');device=O['box'].device
    lines=torch.tensor(cache['lines'],device=device);mass=torch.tensor(cache['mass'],device=device)
    if donor:
        data=Split(EXPORT/f'{split}.json');di=donors(data,split);bb=O['box'];wh=bb[:,2:]-bb[:,:2]
        # Map donor infinite lines by the same affine that maps donor ROI to target ROI.
        l=lines[di];src=bb[di];srcwh=wh[di];scale=srcwh/wh
        n=l[...,:2]*scale[:,None]
        c=l[...,2]+(l[...,:2]*(src[:,None,:2]-scale[:,None]*bb[:,None,:2])).sum(-1)
        lines=torch.cat((n,c[...,None]),-1);mass=mass[di]
    return lines,mass


def prepare():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    RUN.mkdir(parents=True,exist_ok=True);audit={}
    for split in ('train','calibration','synth_val'):
        data,O=observations(split,'cuda:0');masks=point_maps(O['points'],O['box'])
        for seed in (1,2,3):
            out=RUN/f'lines_{split}_seed{seed}.npz'
            if out.exists():continue
            ck=PARENT/'models'/f'points_seed{seed}'/'final.pt'
            model=GuidedDHT(V6/f'cache_heads/hough_seed{seed}/checkpoint_final.pt',seed).to('cuda:0')
            model.load_state_dict(torch.load(ck,map_location='cuda:0',weights_only=False)['state_dict']);model.eval()
            ll=[];mm=[]
            with torch.inference_mode():
                for first in range(0,len(data.records),32):
                    ii=list(range(first,min(first+32,len(data.records))));obs={k:v[ii] for k,v in O.items()}
                    pred=model(obs,masks[ii],decode=True)
                    ll.append(pred['lines'].flatten(1,2).cpu());mm.append(pred['mass'].flatten(1,2).cpu())
            np.savez(out,lines=torch.cat(ll).numpy(),mass=torch.cat(mm).numpy(),ids=np.array([r['id'] for r in data.records]))
            write(RUN/f'lines_{split}_seed{seed}.json',dict(checkpoint_sha256=sha(ck),points_sha256=sha(PARENT/f'points_{split}.json'),
                cache_sha256=sha(out),frames=len(data.records),scope='Frozen DHT inference, no GT points or targets opened.'))
            print('DHT cache',split,seed,len(data.records),flush=True);del model
        del O,masks
    # Targets are prepared only after every inference cache is complete.
    for split in ('train','calibration','synth_val'):
        data=Split(EXPORT/f'{split}.json');rows=[];cross=[]
        for i,rec in enumerate(data.records):
            image=ROOT/rec['original_image'];label=Path(str(image).replace('/images/','/labels/')).with_suffix('.txt')
            txt=[np.fromstring(x,sep=' ') for x in label.read_text().splitlines() if x.strip()]
            assert len(txt)==1 and txt[0].shape==(32,) and np.isfinite(txt[0]).all()
            kp=txt[0][5:].reshape(9,3)[:8];w,h=Image.open(image).size
            gt=kp[:,:2]*[w,h]-100.;valid=kp[:,2]>0
            assert valid.any() and np.isin(kp[:,2],[0,2]).all()
            O,S,_=data.batch([i],'cpu',targets=True)
            assert [h,w]==[int(O['image_hw'][0,0])+200,int(O['image_hw'][0,1])+200]
            cost=np.linalg.norm(gt[valid,None]-S['points'][0,None,:8].numpy(),axis=-1);u,v=linear_sum_assignment(cost)
            err=float(cost[u,v].max());cross.append(err)
            rows.append(dict(id=rec['id'],points=gt.tolist(),valid=valid.tolist(),label=str(label.relative_to(ROOT)),
                             label_sha256=sha(label),renderer_set_max_error_px=err))
        write(RUN/f'targets_{split}.json',dict(rows=rows,scope='Original camera-facing YOLO IDs, raw pixels; v=0 excluded. '
            'No GT assignment is used at inference or primary evaluation. Renderer matching is label-audit only.'))
        audit[split]=dict(frames=len(rows),gt_valid_points=sum(sum(r['valid']) for r in rows),
                         renderer_cross_max_px=max(cross),renderer_cross_p90_px=float(np.quantile(cross,.9)))
        assert max(cross)<.01,(split,max(cross))
    write(RUN/'DATA_AUDIT.json',dict(status='PASS',splits=audit,
        scope='Checks coordinate/order conventions and internal renderer agreement, not real annotation accuracy.'))
    print('DATA PREPARATION COMPLETE',audit,flush=True)


if __name__=='__main__':prepare()
