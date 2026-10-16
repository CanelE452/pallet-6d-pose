"""Matched 2x2 screen on pretrained YOLO26 one-to-one output layers.

Images, backbone, detector, spatial pose features and flow density stay frozen.
This is a last-layer adaptation screen, not full-network YOLO training.
"""
from pathlib import Path
import copy, hashlib, json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
KIT = Path(__file__).resolve().parent
OUT = ROOT/'data/pallet/results/pallet_dimension_symmetry_v1'
CKPT = ROOT/'challenge/yolo_pose_one_model/spatial_concat_scratch/runs/YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt'
CKPT_SHA = '970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7'
META = ROOT/'challenge/yolo_pose_one_model/spatial_concat_scratch/PROBE_METADATA_60K.jsonl'
GEOM = ROOT/'challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz'
CONTRACT = ROOT/'_docs/experiments/pallet_translation_loss_v1/LOSS_SYMMETRY_CONTRACT.json'
ARMS = {'A_indexed': (False, False), 'B_symmetry': (False, True),
        'C_dimensions': (True, False), 'D_both': (True, True)}
LABELS = {'stock':'기존 YOLO', 'A_indexed':'A 기존 손실', 'B_symmetry':'B 대칭만',
          'C_dimensions':'C 치수만', 'D_both':'D 치수+대칭'}
EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def state_sha(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def read(path):return json.loads(path.read_text())

def base_model():
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    assert sha(CKPT)==CKPT_SHA
    m=YOLO(str(CKPT)).model.float().eval()
    m.args=get_cfg(overrides=dict(pose=12.,kobj=1.,rle=1.,box=7.5,cls=.5,dfl=1.5))
    for p in m.parameters():p.requires_grad_(False)
    return m

def c2_permutation(X):
    """Conjugate physical object-Y yaw 180 into the recorded dynamic corner IDs."""
    rot=np.asarray(X)*np.array([-1,1,-1])
    dist=np.linalg.norm(rot[:,None]-X[None],axis=-1)
    perm=dist.argmin(-1)
    assert len(set(perm.tolist()))==8 and dist[np.arange(8),perm].max()<1e-6
    p=np.r_[perm,8]
    assert np.array_equal(p[p],np.arange(9))
    assert {frozenset((p[a],p[b])) for a,b in EDGES}=={frozenset(e) for e in EDGES}
    return p

class OutputAdapter(nn.Module):
    def __init__(self,head,seed):
        super().__init__()
        torch.manual_seed(seed)
        self.weight=nn.Parameter(torch.stack([x.weight[:,:,0,0].detach().clone() for x in head.one2one_cv4_kpts]))
        self.bias=nn.Parameter(torch.stack([x.bias.detach().clone() for x in head.one2one_cv4_kpts]))
        self.sigma_weight=nn.Parameter(torch.stack([x.weight[:,:,0,0].detach().clone() for x in head.one2one_cv4_sigma]))
        self.sigma_bias=nn.Parameter(torch.stack([x.bias.detach().clone() for x in head.one2one_cv4_sigma]))
        self.condition=nn.Sequential(nn.Linear(3,32),nn.SiLU(),nn.Linear(32,90))
        nn.init.zeros_(self.condition[-1].weight);nn.init.zeros_(self.condition[-1].bias)

    def forward(self,z,use_dims):
        # Public model input is W,D,H in metres; normalization is fixed, not fitted on eval.
        d=torch.log(z['dims_whd']/z['dims_whd'].new_tensor([1.,1.,.15]))
        if not use_dims:d=torch.zeros_like(d)
        scale,bias=self.condition(d).chunk(2,-1)
        f=z['features']*(1+scale)+bias
        level=z['level'].long()
        pred=torch.bmm(self.weight[level],f[:,:,None]).squeeze(-1)+self.bias[level]
        sigma=torch.bmm(self.sigma_weight[level],f[:,:,None]).squeeze(-1)+self.sigma_bias[level]
        p=pred.reshape(-1,9,3)
        xy=p[:,:,:2]+z['anchor'][:,None,:]
        return torch.cat([xy,p[:,:,2:3],sigma.reshape(-1,9,2)],-1)

def loss_terms(pred,gt,area,flow):
    """Stock PoseLoss26 terms, reduced per object (one assigned anchor/object).

    The per-object RLE clamp makes the objective separable; it equals the stock
    criterion applied to that individual object. Batch-global RLE reduction is
    intentionally not claimed. Tests compare values and gradients to stock.
    """
    mask=gt[:,:,2]!=0
    n=mask.sum(-1).clamp_min(1)
    d=(pred[:,:,:2]-gt[:,:,:2]).square().sum(-1)
    e=d/((2/9)**2*(area[:,None]+1e-9)*2)
    point=((1-torch.exp(-e))*mask).sum(-1)/n
    vis=F.binary_cross_entropy_with_logits(pred[:,:,2],mask.float(),reduction='none').mean(-1)
    sigma=pred[:,:,-2:].sigmoid()
    err=((pred[:,:,:2]-gt[:,:,:2])/(sigma+1e-9)).clamp(-100,100)
    logphi=flow.log_prob(err.reshape(-1,2)).reshape(-1,9)
    rle=((torch.log(sigma)-logphi[:,:,None]+torch.log(2*sigma)+err.abs()).sum(-1)*mask).sum(-1)/n
    rle=rle.clamp_min(0)
    total=12*point+vis+rle
    return total,torch.stack([point,vis,rle],-1)

def objective(pred,z,flow,symmetry):
    gt=z['target']
    v0,t0=loss_terms(pred,gt,z['area'],flow)
    if not symmetry:return v0,t0,torch.zeros_like(v0,dtype=torch.long)
    g=gt.gather(1,z['perm'][:,:,None].expand(-1,-1,3))
    v1,t1=loss_terms(pred,g,z['area'],flow)
    # A single branch controls localization, visibility AND RLE for the object.
    choose=(z['order']==2)&(v1<v0)
    return torch.where(choose,v1,v0),torch.where(choose[:,None],t1,t0),choose.long()

def subset(z,idx):return {k:v[idx] for k,v in z.items()}

def load_cache(split):return torch.load(OUT/f'{split}.pt',map_location='cpu',weights_only=True)

def prepared_points(pred,z):
    xy=pred[:,:,:2]*z['stride'][:,None,None]
    return (xy-z['letterbox_offset'][:,None,:])/z['letterbox_gain'][:,None,None]-z['pad'][:,None,None]
