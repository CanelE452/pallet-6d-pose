"""New geometry crops centered at original q, INITIAL fixed solver pose only."""
import ast,collections,hashlib,io,json,sys,time,types
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
H=Path(__file__).absolute().parent

def load(path,name):
 m=types.ModuleType(name);m.__file__=str(path);sys.modules[name]=m
 exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m


def run(entry_path):
 p=load(H/'probe_runtime.py','v263_geometry_bound_io')
 e=json.loads(p.safe(entry_path).read_bytes());assert e['approved_actual_original7040_geometry_features'] is True
 assert e['simple_factor']==1. and e['object_fraction']==.25 and p.binding(Path(__file__).absolute())==e['code']
 out=Path(e['output_directory']);p.safe(out.parent);assert not out.exists();out.mkdir()
 started=time.time()
 try:
  packet=json.loads(p.raw(e['label_packet_complete']));assert packet['completed'] and packet['PASS'] and not packet['include_V255']
  groups=[json.loads(x) for x in p.raw(packet['outputs']['groups']).splitlines()]
  rows=[json.loads(x) for x in p.raw(packet['outputs']['rows']).splitlines()]
  initial=json.loads(p.raw(e['initial_complete']));assert initial['completed'] and initial['PASS'] and initial['packet_groups']==packet['outputs']['groups']
  poses=[json.loads(x) for x in p.raw(initial['output']).splitlines()]
  assert len(groups)==len(poses)==880 and len(rows)==7040
  spatial=p.module(e['head_code'],'v263_geometry_math');api=p.module(e['feature_api'],'v263_input_only_crops')
  closure=json.loads(p.raw(e['official_dependency_closure']))
  crop_module=p.module(closure['code']['spatial_crop'],'v263_pinned_V90_crop')
  crops=np.zeros((7040,3,96,96),np.uint8);valid=np.zeros((7040,1,96,96),bool)
  sides=np.zeros(7040,np.float64);central=np.zeros(7040,bool);fallback=np.zeros(7040,bool)
  support=np.zeros((7040,16,16),bool);accessed={};lru=collections.OrderedDict();join=[]
  for gi,(g,init) in enumerate(zip(groups,poses)):
   assert g['group_index']==init['group_index']==gi and g['group_id']==init['group_id']
   assert g['row_ids8']==init['row_ids8'] and g['source_indices8']==init['source_indices8'] and g['fit']==init['fit']
   assert np.array_equal(np.asarray(g['q8']),np.asarray(init['q8']))
   image=g['image'];path=image['path']
   if path not in lru:
    b=p.safe(path).read_bytes();assert hashlib.sha256(b).hexdigest()==image['sha256']
    with np.load(io.BytesIO(b),allow_pickle=False) as z:rgb=np.array(z['RGB_uint8'],copy=True)
    assert rgb.dtype==np.uint8 and tuple(rgb.shape)==tuple(g['raw_hw'])+(3,)
    lru[path]=rgb;accessed[path]={'path':path,'sha256':image['sha256'],'size_bytes':len(b),'member_decoded':'RGB_uint8','other_members_decoded':0}
    if len(lru)>16:lru.popitem(last=False)
   rgb=lru[path];lru.move_to_end(path)
   pose=init['initial_pose'];ok=bool(pose['available'])
   T=np.eye(4,dtype=np.float64)
   if ok:T[:3,:3]=np.asarray(pose['R_cf']);T[:3,3]=np.asarray(pose['centroid'])
   dims=pose.get('cf_extents',g['CF_dimensions_xyz'])
   # Registered input dimensions may be read; never GT_R_CF/GT_t or reference r.
   result=api.prepare_frame(spatial,crop_module.crop,rgb,g['q8'],g['L8'],g['original_available8'],T,g['K'],dims,ok)
   ix=np.asarray(g['source_indices8'],np.int64)
   assert all(rows[int(j)]['row_id']==name for j,name in zip(ix,g['row_ids8']))
   crops[ix]=result['RGB96'][1];valid[ix]=result['valid96'][1];sides[ix]=result['crop_side'][1]
   support[ix]=result['token_support'][1];central[ix]=result['central4_valid'][1];fallback[ix]=result['geometry_fallback_to_simple']
   join.append({'group_index':gi,'group_id':g['group_id'],'initial_solver_available':ok,'input_image':image,
      'source_indices8':g['source_indices8'],'crop_side8':result['crop_side'][1].tolist(),'fallback8':result['geometry_fallback_to_simple'].tolist()})
   if gi%80==0:print(json.dumps({'groups_prepared':gi+1,'total':880}),flush=True)
  for name,array in [('TOKEN_SUPPORT.npy',support),('CROP_SIDE.npy',sides),('CENTRAL4.npy',central),('GEOMETRY_FALLBACK.npy',fallback)]:np.save(out/name,array)
  (out/'CROP_INPUT_JOIN.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in join))
  (out/'RGB_ACCESS_WHITELIST.json').write_text(json.dumps(list(accessed.values()),indent=2,sort_keys=True)+'\n')
  train=p.module(e['training_code'],'v263_geometry_extract_api')
  p.module(closure['code']['crop_module'],'crop_modules');loader=p.module(closure['code']['model_loader'],'v263_geometry_official_loader')
  model=loader.load_backbone(closure['model_loader_config'],'cuda:0',treatment=False).eval()
  before=loader.tensor_state_sha(model.state_dict(),base=True);assert before==closure['expected_encoder_state_sha256']
  src=closure['code']['preprocess'];funcs=[n for n in ast.parse(p.raw(src)).body if isinstance(n,ast.FunctionDef) and n.name=='preprocess_views']
  ns={'torch':torch,'F':F,'MEAN':(.485,.456,.406),'STD':(.229,.224,.225)}
  exec(compile(ast.Module(body=funcs,type_ignores=[]),src['path'],'exec'),ns)
  features=np.lib.format.open_memmap(out/'TOKENS_FP16.npy',mode='w+',dtype=np.float16,shape=(7040,256,384));features[:]=0
  active=np.flatnonzero(support.reshape(7040,-1).any(1));repeat=True
  for start in range(0,len(active),32):
   ix=active[start:start+32];views,_=ns['preprocess_views'](torch.from_numpy(crops[ix]),torch.from_numpy(valid[ix]))
   tokens=train.extract_spatial(model,views[:len(ix)].cuda())
   if start==0:repeat=bool(torch.equal(tokens,train.extract_spatial(model,views[:len(ix)].cuda())));assert repeat
   features[ix]=tokens.cpu().numpy()
  features.flush();del features
  after=loader.tensor_state_sha(model.state_dict(),base=True);assert before==after
  names=['TOKENS_FP16.npy','TOKEN_SUPPORT.npy','CROP_SIDE.npy','CENTRAL4.npy','GEOMETRY_FALLBACK.npy','CROP_INPUT_JOIN.jsonl','RGB_ACCESS_WHITELIST.json']
  result={'completed':True,'PASS':True,'entry':p.binding(Path(entry_path)),'code':e['code'],'initial_complete':e['initial_complete'],
   'label_packet_complete':e['label_packet_complete'],'outputs':{n:p.binding(out/n) for n in names},
   'census':{'all_rows':7040,'groups':880,'initial_solver_available':sum(x['initial_solver_available'] for x in join),
     'geometry_fallback_rows':int(fallback.sum()),'supported_rows':len(active),'RGB_assets':len(accessed)},
   'encoder_state_before':before,'encoder_state_after':after,'encoder_frozen':True,'first_batch_repeat_FP16_byte_exact':repeat,
   'crop':'original q centre; fixedsolver INITIAL pose registered projection maxspan*.25, simpleL fallback',
   'GT_pose_reference_or_labels_in_encoder_inputs':False,'other_NPZ_members_decoded':0,'current_or_held_reads':0,
   'head_optimizer_updates':0,'elapsed_seconds':time.time()-started}
  (out/'COMPLETE.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'completed':True,'PASS':True,'COMPLETE':p.binding(out/'COMPLETE.json'),'census':result['census']}),flush=True)
 except BaseException as ex:
  (out/'FAILURE.json').write_text(json.dumps({'completed':False,'PASS':False,'error_type':type(ex).__name__,'error':str(ex),'automatic_retry':False},indent=2)+'\n');raise

if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--entry',required=True);run(Path(a.parse_args().entry))
