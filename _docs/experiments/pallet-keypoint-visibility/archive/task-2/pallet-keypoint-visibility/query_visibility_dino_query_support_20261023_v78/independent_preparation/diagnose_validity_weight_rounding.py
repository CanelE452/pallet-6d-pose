"""Bounded input-only FP64/CPU32/GPU32 validity-weight diagnostic, no model."""
from pathlib import Path
from functools import lru_cache
import json,hashlib
import numpy as np
import torch
from torch.nn import functional as F
from dino_pool_audit_math import patch_weights
R=Path(__file__).resolve().parent;V=R.parent

def load(p):return json.loads(Path(p).read_text())
def bind(p):return dict(path=str(Path(p).resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
prot=load(V/'feature_preparation/FEATURE_PROTOCOL.json');receipt=load(V/'feature_pack/FEATURE_INFERENCE_RECEIPT.json');torch.set_num_threads(1)
allchunks=[]
for bank in prot['banks']:
 for ch in load(bank['manifest']['path'])['chunks']:allchunks.append((bank['start']+ch['start'],bank['start']+ch['end'],ch['file']['path']))
@lru_cache(maxsize=6)
def valid_chunk(path):
 with np.load(path,allow_pickle=False)as z:return z['crop_valid96'],z['target_valid128']
def mask(gi,ki,kind):
 a,b,path=next(c for c in allchunks if c[0]<=gi<c[1]);lv,tv=valid_chunk(path)
 return lv[gi-a,ki,0]if kind=='local'else tv[gi-a,0]
rows=[]
for p in receipt['fixed_batch_repeat_and_pool_proofs']:
 with np.load(p['file']['path'],allow_pickle=False)as z:d={k:z[k]for k in z.files}
 masks=np.stack([mask(int(i),int(k),p['kind'])for i,k in zip(d['frame_index'],d['query_index'])]);ref64=patch_weights(masks)
 cpu32=F.avg_pool2d(F.interpolate(torch.from_numpy(masks[:,None]).float(),size=(224,224),mode='bilinear',align_corners=False),14,14)[:,0].numpy();gpu=d['token_validity_weights']
 delta=abs(gpu.astype(float)-ref64);where=np.unravel_index(delta.argmax(),delta.shape)
 sourcepoint=dict(proof_batch_index=int(where[0]),frame_index=int(d['frame_index'][where[0]]),query_index=int(d['query_index'][where[0]]),token_y=int(where[1]),token_x=int(where[2]),GPU32=float(gpu[where]),CPU32=float(cpu32[where]),independent_FP64=float(ref64[where]))
 if p['kind']=='local':
  fraction64=np.stack([ref64[:,7:9,7:9].mean((1,2)),ref64.mean((1,2))],1);fraction_cpu32=np.stack([cpu32[:,7:9,7:9].mean((1,2)),cpu32.mean((1,2))],1)
 else:fraction64=ref64.mean((1,2));fraction_cpu32=cpu32.mean((1,2))
 rows.append(dict(kind=p['kind'],regimes=p['regimes'],proof=p['file'],views=len(masks),GPU32_vs_FP64_maxabs=float(delta.max()),GPU32_vs_CPU32_maxabs=float(abs(gpu-cpu32).max()),GPU32_vs_CPU32_bit_mismatch_elements=int((gpu!=cpu32).sum()),CPU32_vs_FP64_maxabs=float(abs(cpu32.astype(float)-ref64).max()),GPUfractions_vs_FP64_maxabs=float(abs(d['fractions_fp32'].astype(float)-fraction64).max()),FP64fraction_to_FP16_vs_GPUfraction_to_FP16_mismatch=int((fraction64.astype(np.float16)!=d['fractions_fp32'].astype(np.float16)).sum()),CPU32fraction_to_FP16_vs_GPUfraction_to_FP16_mismatch=int((fraction_cpu32.astype(np.float16)!=d['fractions_fp32'].astype(np.float16)).sum()),largest_example=sourcepoint))
p=R/'VALIDITY_WEIGHT_ROUNDING_DIAGNOSTIC.json';assert not p.exists();p.write_text(json.dumps(dict(status='DIAGNOSTIC_NO_INPUT_OR_MODEL_CHANGES',code=bind(Path(__file__)),engineering_weight_assertion_origin=1e-6,rows=rows,source_RGB_P3_GT_model_forward_fit_current_PnP=0,runtime=dict(torch=str(torch.__version__),numpy=np.__version__,device='CPU')),indent=2)+'\n');print(json.dumps(dict(report=bind(p),rows=rows)))
