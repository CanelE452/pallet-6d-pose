"""One Root-fixed SYN-only arm. Saves final before any DEV head forward."""
import hashlib,json,os,sys,time,types
from pathlib import Path
import numpy as np
import torch
H=Path(__file__).absolute().parent

def load(p,name):
 m=types.ModuleType(name);m.__file__=str(p);sys.modules[name]=m
 exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__);return m


def run(path):
 p=load(H/'probe_runtime.py','v263_arm_bound_io')
 e=json.loads(p.safe(path).read_bytes());assert e['approved_actual_syn_arm'] is True
 assert p.binding(Path(__file__).absolute())==e['code']
 assert e['mode'] in ['LOC','VIS','JOINT'] and e['crop'] in ['simple2D','initial_geometry']
 out=Path(e['output_directory']);p.safe(out.parent);assert not out.exists();out.mkdir()
 started=time.time()
 try:
  packet=json.loads(p.raw(e['label_packet_complete']));assert packet['completed'] and packet['PASS']
  rows=[json.loads(x) for x in p.raw(packet['outputs']['rows']).splitlines()]
  features=json.loads(p.raw(e['feature_complete']));assert features['completed'] and features['PASS']
  for key in ['TOKENS_FP16.npy','TOKEN_SUPPORT.npy']:
   b=features['outputs'][key];assert p.binding(Path(b['path']))==b
  tokens=np.load(p.safe(features['outputs']['TOKENS_FP16.npy']['path']),mmap_mode='r',allow_pickle=False)
  support=np.load(p.safe(features['outputs']['TOKEN_SUPPORT.npy']['path']),allow_pickle=False)
  assert tokens.shape==(7040,256,384) and tokens.dtype==np.float16 and support.dtype==bool
  n=len(rows);q=np.array([r['q0'] for r in rows],np.float64);rxy=np.array([r['reference_xy'] for r in rows],np.float64)
  native=np.array([r['native_slot'] for r in rows],np.int64);fit=np.array([r['fit'] for r in rows],bool)
  original=np.array([r['original_available'] for r in rows],bool)
  if e['crop']=='simple2D':
   side=np.array([r['L'] for r in rows],np.float64)
   central=np.array([r['input_supported'] for r in rows],bool)
  else:
   side=np.load(p.safe(features['outputs']['CROP_SIDE.npy']['path']),allow_pickle=False)
   central=np.load(p.safe(features['outputs']['CENTRAL4.npy']['path']),allow_pickle=False)
   for name in ['CROP_SIDE.npy','CENTRAL4.npy']:
    assert p.binding(Path(features['outputs'][name]['path']))==features['outputs'][name]
  s=p.module(e['head_code'],'v263_fixed_spatial_head')
  t=p.module(e['training_code'],'v263_fixed_fit_api')
  tensor=lambda a:torch.as_tensor(a)
  state=np.array([{'D':0,'E':1,'S':2,'U':-1}.get(r['point_combined_state'],-1) for r in rows],np.int64)
  loc,uv,location=t.fixed_targets(s,tensor(original&central),tensor(np.array([r['direct_cue_known'] for r in rows],bool)),
      tensor(np.array([r['direct_geometric_2edge'] for r in rows],bool)),tensor(np.array([r['identity_known'] for r in rows],bool)),
      tensor(np.array([r['identity_correct'] for r in rows],bool)),tensor(state),tensor(rxy),tensor(q),tensor(side))
  visibility=np.array([r['semantic_visibility'] for r in rows],np.int64)
  # All metadata is retained; only FIT token batches enter optimization.
  model,result,logs,draws=t.fit_tensors(s,tokens,support,native,fit,uv.numpy(),loc.numpy(),visibility,location.numpy(),e['mode'])
  trained={'VIS_trained':e['mode'] in ['VIS','JOINT'],'LOC_trained':e['mode'] in ['LOC','JOINT']}
  checkpoint={'state_dict':model.cpu().state_dict(),'result':result,'trained':trained,'crop':e['crop'],
              'native_namespace':'original immutable0..7','head_code':e['head_code'],'training_code':e['training_code']}
  torch.save(checkpoint,out/'FINAL1000.pt')
  (out/'TRAIN1000.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in logs))
  np.save(out/'DRAWS1000x32.npy',draws)
  census={}
  for role,mask in [('FIT',fit),('DEV',~fit)]:
   census[role]={'rows':int(mask.sum()),'LOC_valid':int((loc.numpy()&mask).sum()),
      'location_states':{str(v):int(((location.numpy()==v)&mask).sum()) for v in [-1,0,1,2]},
      'VIS_states':{str(v):int(((visibility==v)&mask).sum()) for v in [-1,0,1]}}
  np.savez(out/'TARGETS_NONPIXEL.npz',q0=q,crop_side=side,reference_xy=rxy,reference_uv=uv.numpy(),LOC_valid=loc.numpy(),
           location_target=location.numpy(),visibility_target=visibility,native_slot=native,fit=fit,original_available=original)
  complete={'completed':True,'PASS':True,'entry':p.binding(Path(path)),'code':e['code'],'head_code':e['head_code'],
      'training_code':e['training_code'],'label_packet_complete':e['label_packet_complete'],'feature_complete':e['feature_complete'],
      'mode':e['mode'],'crop':e['crop'],'trained':trained,'result':result,'target_census':census,
      'outputs':{name:p.binding(out/name) for name in ['FINAL1000.pt','TRAIN1000.jsonl','DRAWS1000x32.npy','TARGETS_NONPIXEL.npz']},
      'new_DEV_head_forwards':0,'current_or_held_reads':0,'GT_teacher_fields_in_head_inputs':False,
      'final_frozen_before_DEV_head_forward':True,'elapsed_seconds':time.time()-started}
  (out/'COMPLETE.json').write_text(json.dumps(complete,indent=2,sort_keys=True)+'\n')
  print(json.dumps({'completed':True,'PASS':True,'COMPLETE':p.binding(out/'COMPLETE.json'),'mode':e['mode'],'census':census}),flush=True)
 except BaseException as ex:
  (out/'FAILURE.json').write_text(json.dumps({'completed':False,'PASS':False,'error_type':type(ex).__name__,'error':str(ex),'automatic_retry':False},indent=2)+'\n');raise

if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--entry',required=True);run(Path(a.parse_args().entry))
