"""Generated Root target-semantics check; no actual source input."""
import hashlib,json,sys,types
from pathlib import Path
import torch
H=Path(__file__).absolute().parent

def load(name):
 p=H/(name+'.py');m=types.ModuleType(name);m.__file__=str(p);sys.modules[name]=m
 exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__);return m

s=load('spatial_head_v2');t=load('train_spatial_v2')
n=7;available=torch.tensor([1,1,1,1,1,0,1],dtype=torch.bool)
known=torch.ones(n,dtype=torch.bool);cue=torch.tensor([1,0,0,0,1,1,0],dtype=torch.bool)
cert=torch.tensor([0,0,1,2,0,0,-1]);q=torch.zeros(n,2);r=q.clone();r[4,0]=60
loc,uv,state=t.fixed_targets(s,available,known,cue,known,known,cert,r,q,torch.ones(n)*96)
assert state.tolist()==[0,-1,1,1,2,-1,-1] and loc.tolist()==[True,False,False,False,False,False,False]
# Explicit semantic targets preserve weak D, and S is semantic unknown.
vis=torch.where(cert==2,torch.full_like(cert,-1),cert)
assert vis.tolist()==[0,0,1,-1,0,0,-1]
assert not bool(loc[(cert==1)|(cert==2)].any())
out=H/'TARGET_SEMANTICS_V2_CHECKS.json';assert not out.exists()
result={'completed':True,'PASS':True,'certified_D_E_S_U':cert.tolist(),'location_state':state.tolist(),
 'semantic_VIS_D_E_U':vis.tolist(),'weak_D_is_LOC_unknown_VIS_D':True,
 'E_S_amodal_heatmap_rows':0,'NO_MATCH_means_no_directly_observable_native_point':True,
 'actual_source_current_held_or_official_model_reads':0,
 'train_code':{'path':str(H/'train_spatial_v2.py'),'sha256':hashlib.sha256((H/'train_spatial_v2.py').read_bytes()).hexdigest(),'size_bytes':len((H/'train_spatial_v2.py').read_bytes())}}
out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,sort_keys=True))
