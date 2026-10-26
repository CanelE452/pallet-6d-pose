"""Frozen SYN-selected three-head current development inference, no GT/eval reads."""
import ast,hashlib,io,json,sys,time,types
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
H=Path(__file__).absolute().parent

def load(path,name):
 m=types.ModuleType(name);m.__file__=str(path);sys.modules[name]=m
 exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m


def run(entry_path):
 p=load(H/'probe_runtime.py','v263_current_guarded_io')
 e=json.loads(p.safe(entry_path).read_bytes());assert e['approved_frozen_current_development_inference'] is True
 assert p.binding(Path(__file__).absolute())==e['code']
 freeze=json.loads(p.raw(e['training_freeze']));assert freeze['completed'] and freeze['PASS'] and freeze['all_learned_settings']==4
 assert freeze['selected_crop']=='initial_geometry'
 out=Path(e['output_directory']);p.safe(out.parent);assert not out.exists();out.mkdir()
 started=time.time()
 try:
  # Only declared input-only fields from this historical entry are consumed.
  transport=json.loads(p.raw(e['canonical_transport_entry']))['encoded_alias_transport']['canonical_mappings']
  metadata=json.loads(p.raw(e['input_manifest']));frames=metadata['frames223'];source_rows=metadata['rows1784']
  geometry=json.loads(p.raw(e['geometry_input']))['rows']
  assert len(transport)==len(frames)==len(geometry)==223 and len(source_rows)==1784
  assert metadata['completed'] and metadata['PASS'] and not metadata['GT_or_score_features_added']
  selected=[x for x in freeze['current_primary_configurations'] if x!='ORIGINAL']
  s=p.module(freeze['head_code'],'v263_current_spatial_head');t=p.module(freeze['training_code'],'v263_current_train_api')
  api=p.module(e['feature_api'],'v263_current_input_crop_api')
  closure=json.loads(p.raw(e['official_dependency_closure']));crop=p.module(closure['code']['spatial_crop'],'v263_current_V90_crop').crop
  rgb96=np.zeros((2,1784,3,96,96),np.uint8);valid96=np.zeros((2,1784,1,96,96),bool)
  support=np.zeros((1784,16,16),bool);sides=np.zeros((2,1784),np.float64);fallback=np.zeros(1784,bool)
  original_available=np.zeros(1784,bool);q0=np.zeros((1784,2),np.float64);native=np.tile(np.arange(8,dtype=np.int64),223)
  base_rows=[];access=[]
  for fi,(frame,g,canonical) in enumerate(zip(frames,geometry,transport)):
   assert frame['frame_index']==canonical['frame_index']==fi and frame['id']==g['id']
   q=np.asarray(frame['original_query_xy8'],np.float64);L=np.asarray(frame['original_L8'],np.float64);avail=np.asarray(frame['original_query_available8'],bool)
   assert np.array_equal(q,np.asarray(g['original_initial_points'],np.float64)[:8]) and frame['native_slot']==list(range(8))
   binding=canonical['canonical_binding'];assert binding['sha256']==frame['source_fullRGB']['binding']['sha256']
   encoded=p.raw(binding)
   with Image.open(io.BytesIO(encoded)) as im:rgb=np.asarray(im.convert('RGB'),np.uint8).copy()
   assert tuple(rgb.shape[:2])==tuple(frame['raw_hw']) and hashlib.sha256(rgb.tobytes()).hexdigest()==frame['decoded_RGB_sha256']
   assert g['upstream_provenance']['GT_inputs'] is False
   # physical T and physical XYZ dimensions are paired; CF native_xyz is NOT transformed by physical T.
   packet=api.prepare_frame(s,crop,rgb,q,L,avail,g['initial_T_physical'],g['K'],g['registered_dimensions_xyz_m'],g['initial_pose_available'])
   ix=slice(fi*8,fi*8+8);rgb96[:,ix]=packet['RGB96'];valid96[:,ix]=packet['valid96'];support[ix]=packet['token_support'][1]
   sides[:,ix]=packet['crop_side'];fallback[ix]=packet['geometry_fallback_to_simple'];original_available[ix]=avail;q0[ix]=q
   for slot in range(8):
    index=fi*8+slot;source=source_rows[index]
    assert source['original_global_index']==index and source['native_slot']==slot and np.array_equal(source['query_xy'],q[slot])
    base_rows.append({'query_index':index,'global_index':index,'frame_index':fi,'frame_id':frame['id'],'native_slot':slot,
      'original_q0':q[slot].tolist(),'L':float(L[slot]),'available':bool(avail[slot]),'crop_side':float(packet['crop_side'][1,slot]),
      'crop_kind':'initial_geometry','geometry_fallback':bool(packet['geometry_fallback_to_simple'][slot]),
      'input_supported':bool(support[index].any()),'teacher_or_GT_in_inputs':False})
   access.append({'frame_index':fi,'id':frame['id'],'canonical_binding':binding,'decoded_RGB_sha256':hashlib.sha256(rgb.tobytes()).hexdigest()})
   if fi%40==0:print(json.dumps({'current_frames_cropped':fi+1,'total':223}),flush=True)
  np.savez_compressed(out/'CURRENT_RAW96_CROPS.npz',RGB96_simple=rgb96[0],valid96_simple=valid96[0],RGB96_initial_geometry=rgb96[1],
      valid96_initial_geometry=valid96[1],crop_side_simple=sides[0],crop_side_initial_geometry=sides[1],original_available=original_available)
  np.save(out/'TOKEN_SUPPORT.npy',support)
  (out/'INPUT_ROWS1784.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in base_rows))
  (out/'RGB223_ACCESS.json').write_text(json.dumps(access,indent=2,sort_keys=True)+'\n')
  p.module(closure['code']['crop_module'],'crop_modules');loader=p.module(closure['code']['model_loader'],'v263_current_official_loader')
  model=loader.load_backbone(closure['model_loader_config'],'cuda:0',treatment=False).eval()
  before=loader.tensor_state_sha(model.state_dict(),base=True);assert before==closure['expected_encoder_state_sha256']
  src=closure['code']['preprocess'];funcs=[n for n in ast.parse(p.raw(src)).body if isinstance(n,ast.FunctionDef) and n.name=='preprocess_views']
  ns={'torch':torch,'F':F,'MEAN':(.485,.456,.406),'STD':(.229,.224,.225)};exec(compile(ast.Module(body=funcs,type_ignores=[]),src['path'],'exec'),ns)
  features=np.lib.format.open_memmap(out/'SPATIAL_TOKENS_FP16.npy',mode='w+',dtype=np.float16,shape=(1784,256,384));features[:]=0
  active=np.flatnonzero(support.reshape(1784,-1).any(1));repeat=True
  for start in range(0,len(active),32):
   ix=active[start:start+32];views,_=ns['preprocess_views'](torch.from_numpy(rgb96[1,ix]),torch.from_numpy(valid96[1,ix]))
   token=t.extract_spatial(model,views[:len(ix)].cuda())
   if start==0:repeat=bool(torch.equal(token,t.extract_spatial(model,views[:len(ix)].cuda())));assert repeat
   features[ix]=token.cpu().numpy()
  features.flush();after=loader.tensor_state_sha(model.state_dict(),base=True);assert before==after
  rules=json.loads(p.raw(freeze['rule_definition']))['rules'];predictions={};heatmaps={}
  names=['ORIGINAL',*selected]
  for name in names:
   if name=='ORIGINAL':headout=None;trained_vis=False;trained_loc=False;accepted=torch.zeros(1784,dtype=torch.bool)
   else:
    cp=torch.load(io.BytesIO(p.raw(freeze['checkpoints'][name])),map_location='cpu',weights_only=True)
    head=s.SpatialPointHead().cuda().eval();head.load_state_dict(cp['state_dict'],strict=True)
    headout=t.infer_tensors(s,head,features,support,native,np.arange(1784))
    trained_vis=cp['trained']['VIS_trained'];trained_loc=cp['trained']['LOC_trained'];rule_name=freeze['selected_rules'][name]
    accepted=s.acceptance(headout,rules[rule_name])[0] if trained_loc and rule_name!='ALL_ABSTAIN' else torch.zeros(1784,dtype=torch.bool)
    if trained_loc:
     path=out/(name+'_HEATMAPS.npz');np.savez_compressed(path,probability=headout['probability'].numpy(),heatmap_logits=headout['heatmap_logits'].numpy())
     heatmaps[name]={'binding':p.binding(path),'probability_member':'probability','logit_member':'heatmap_logits','shape':[1784,16,16],'order':'query_index'}
   if trained_loc:
    proposed=q0+(headout['uv_subpixel'].numpy()-.5)*sides[1,:,None];final=np.where(accepted.numpy()[:,None],proposed,q0)
   else:proposed=None;final=q0.copy()
   if trained_vis:semantic=s.semantic_visibility(headout).numpy();logits=headout['visibility_E_logit'].numpy()
   rows=[]
   for i,base in enumerate(base_rows):
    row=dict(base);prediction={-1:'U',0:'D',1:'E'}[int(semantic[i])] if trained_vis else 'N_A'
    row.update({'arm':name,'VIS_trained':trained_vis,'LOC_trained':trained_loc,
     'visibility_E_logit':float(logits[i]) if trained_vis else None,'semantic_VIS_prediction':prediction,
     'LOC_proposed_xy':proposed[i].tolist() if trained_loc and base['input_supported'] else None,
     'accepted':bool(accepted[i]),'final_xy':final[i].tolist(),
     'location_state':int(headout['location_logits'][i].argmax()) if trained_loc and base['input_supported'] else -1,
     'solver_E_exclusion':bool(base['available'] and trained_vis and prediction=='E'),
     'original_solver_support_changed_by_U':False})
    rows.append(row)
   path=out/(name+'_ALL1784_PREDICTIONS.jsonl');path.write_text(''.join(json.dumps(x,sort_keys=True,allow_nan=False)+'\n' for x in rows));predictions[name]=p.binding(path)
  del features
  barrier={'completed':True,'PASS':True,'training_freeze':e['training_freeze'],'predictions':predictions,'heatmaps':heatmaps,
   'all_predictions_rows_each':1784,'current_annotations_or_GT_or_v61_baseline_reads_before_prediction_freeze':0,
   'fixed_E_threshold':'strict logit>0, only available predictedE excluded; legacy solver support for U unchanged',
   'no_current_based_configuration_change':True}
  frozen=out/'ALL_CURRENT_PREDICTIONS_FROZEN_BEFORE_EVALUATION.json';frozen.write_text(json.dumps(barrier,indent=2,sort_keys=True)+'\n')
  result={'completed':True,'PASS':True,'entry':p.binding(Path(entry_path)),'code':e['code'],'training_freeze':e['training_freeze'],
   'prediction_freeze':p.binding(frozen),'predictions':predictions,'heatmaps':heatmaps,'selected_crop':'initial_geometry',
   'crop_cache':{'binding':p.binding(out/'CURRENT_RAW96_CROPS.npz'),'layout':'NCHW','members':{'simple2D':['RGB96_simple','valid96_simple'],'initial_geometry':['RGB96_initial_geometry','valid96_initial_geometry']},'rows':1784,'order':'query_index'},
   'outputs':{n:p.binding(out/n) for n in ['TOKEN_SUPPORT.npy','SPATIAL_TOKENS_FP16.npy','INPUT_ROWS1784.jsonl','RGB223_ACCESS.json']},
   'frames':223,'rows':1784,'model_supported_rows':len(active),'geometry_fallback_rows':int(fallback.sum()),
   'encoder_state_before':before,'encoder_state_after':after,'first_batch_repeat_FP16_byte_exact':repeat,
   'new_optimizer_updates':0,'annotations_GT_baselines_read':0,'existing223_is_development_evaluation_not_blind':True,'elapsed_seconds':time.time()-started}
  (out/'COMPLETE.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'completed':True,'PASS':True,'COMPLETE':p.binding(out/'COMPLETE.json'),'prediction_freeze':p.binding(frozen)}),flush=True)
 except BaseException as ex:
  (out/'FAILURE.json').write_text(json.dumps({'completed':False,'PASS':False,'error_type':type(ex).__name__,'error':str(ex),'automatic_retry':False},indent=2)+'\n');raise

if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--entry',required=True);run(Path(a.parse_args().entry))
