"""After both LOC finals are frozen, fixed SYN DEV crop/rule comparison."""
import hashlib,io,json,sys,types
from pathlib import Path
import numpy as np
import torch
H=Path(__file__).absolute().parent

def load(path,name):
 m=types.ModuleType(name);m.__file__=str(path);sys.modules[name]=m
 exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m


def evaluate_arm(p, arm_binding, spatial, train, selector, rules, output_path):
 arm=json.loads(p.raw(arm_binding));assert arm['completed'] and arm['PASS'] and arm['result']['optimizer_updates']==1000
 cp=arm['outputs']['FINAL1000.pt'];checkpoint=torch.load(io.BytesIO(p.raw(cp)),map_location='cpu',weights_only=True)
 model=spatial.SpatialPointHead().cuda().eval();model.load_state_dict(checkpoint['state_dict'],strict=True)
 features=json.loads(p.raw(arm['feature_complete']))
 tokens=np.load(p.safe(features['outputs']['TOKENS_FP16.npy']['path']),mmap_mode='r',allow_pickle=False)
 support=np.load(p.safe(features['outputs']['TOKEN_SUPPORT.npy']['path']),allow_pickle=False)
 with np.load(io.BytesIO(p.raw(arm['outputs']['TARGETS_NONPIXEL.npz'])),allow_pickle=False) as z:
  target={k:np.array(z[k],copy=True) for k in z.files}
 dev=np.flatnonzero(~target['fit'])
 out=train.infer_tensors(spatial,model,tokens,support,target['native_slot'],dev)
 q=torch.from_numpy(target['q0'][dev]);side=torch.from_numpy(target['crop_side'][dev]);r=torch.from_numpy(target['reference_xy'][dev]);eligible=torch.from_numpy(target['LOC_valid'][dev])
 selected=selector.select_on_syn_dev(spatial,out,q,side,r,eligible,rules)
 raw=q+(out['uv_subpixel']-.5)*side[:,None]
 error=torch.linalg.vector_norm(raw-r,dim=1);good=eligible&torch.isfinite(error)
 values=error[good].numpy()
 selected['raw_eligible_error']={'count':len(values),'median_px':float(np.median(values)) if len(values) else None,
  'mean_px':float(np.mean(values)) if len(values) else None,'within3_count':int((values<=3).sum()),
  'original_q0_median_px':float(np.median(np.linalg.norm(target['q0'][dev][good]-target['reference_xy'][dev][good],axis=1))) if len(values) else None}
 selected['arm_complete']=arm_binding;selected['DEV_rows']=len(dev)
 np.savez(output_path,source_index=dev,**{k:v.numpy() for k,v in out.items()})
 selected['saved_DEV_outputs']=p.binding(output_path)
 return selected


def run():
 p=load(H/'probe_runtime.py','v263_selection_io')
 arms={'simple2D':p.binding(H/'LOC_simple2D_v1/COMPLETE.json'),
       'initial_geometry':p.binding(H/'LOC_initial_geometry_v1/COMPLETE.json')}
 parsed={name:json.loads(p.raw(b)) for name,b in arms.items()}
 assert all(c['result']['optimizer_updates']==1000 and c['new_DEV_head_forwards']==0 for c in parsed.values())
 assert len({c['result']['initial_state_sha256'] for c in parsed.values()})==1
 assert len({c['result']['draws_sha256'] for c in parsed.values()})==1
 freeze=H/'LOC_BOTH_FINAL_FROZEN_BEFORE_SYN_DEV.json';assert not freeze.exists()
 freeze.write_text(json.dumps({'completed':True,'PASS':True,'arms':arms,
  'checkpoints':{name:c['outputs']['FINAL1000.pt'] for name,c in parsed.items()},
  'DEV_head_forwards_before_barrier':0,'shared_initial_and_draws':True,'current_or_held_reads':0},indent=2,sort_keys=True)+'\n')
 # This is the first new learned head numeric DEV forward for either crop.
 s=p.module(parsed['simple2D']['head_code'],'v263_selector_head')
 t=p.module(parsed['simple2D']['training_code'],'v263_selector_train')
 selector=load(H/'select_acceptance.py','v263_rule_math')
 rules=json.loads((H/'ACCEPTANCE_RULES_BEFORE_FIT.json').read_text())['rules']
 evaluations={name:evaluate_arm(p,b,s,t,selector,rules,H/('SYN_DEV_'+name+'_OUTPUTS.npz')) for name,b in arms.items()}
 def key(name):
  d=evaluations[name];rows={x['rule']:x for x in d['table']};chosen=d['selected_rule']
  coverage=rows[chosen]['all_query_coverage'] if chosen!='ALL_ABSTAIN' else 0.
  median=d['raw_eligible_error']['median_px']
  return (-coverage,median if median is not None else float('inf'),0 if name=='simple2D' else 1)
 chosen=min(evaluations,key=key)
 result={'completed':True,'PASS':True,'freeze':p.binding(freeze),'evaluations':evaluations,'selected_crop':chosen,
  'selected_LOC_rule':evaluations[chosen]['selected_rule'],
  'crop_selection':'max qualifying ALL-query coverage; tie raw eligible median error; tie simple2D',
  'no_qualifying_rule':'all-abstain preserved; VIS/JOINT still train selected crop',
  'selection_data':'SYN DEV only, repeated source partition','current_or_held_reads':0}
 out=H/'SYN_DEV_CROP_AND_RULE_SELECTION.json';assert not out.exists();out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'completed':True,'PASS':True,'selection':p.binding(out),'selected_crop':chosen,'selected_LOC_rule':result['selected_LOC_rule'],'evaluations':evaluations}),flush=True)

if __name__=='__main__':run()
