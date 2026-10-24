"""V76 post-freeze current inference only. GT/annotations never read here."""
from pathlib import Path
import ast,csv,gzip,hashlib,importlib.util,json,sys,time
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];P=ROOT.parent
OLD=P/'query_visibility_cue_occlusion_20261023_v72'
F=ROOT/'mixed_fit/runs/MIXED_SUPPORT_V1/MODEL_FREEZE.json'
OUT=ROOT/'private_evaluation/MIXED_SUPPORT_V1'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for a in iter(lambda:f.read(1<<20),b''):h.update(a)
 return h.hexdigest()
def b(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(v):assert sha(v['path'])==v['sha256'],v['path']
def load(p):return json.loads(Path(p).read_text())
def save(p,v):
 assert not p.exists();p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def extract(path,names):
 t=ast.parse(path.read_text());body=[n for n in t.body if isinstance(n,(ast.FunctionDef,ast.Import,ast.ImportFrom)) and (not isinstance(n,ast.FunctionDef)or n.name in names)]
 body=[n for n in body if not isinstance(n,ast.ImportFrom) and (not isinstance(n,ast.Import)or all(a.name=='numpy'for a in n.names))]
 d={};exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),d);return d

def main():
 gate=load(ROOT/'private_evaluation/POST_SOURCE_FREEZE_GATE.json');check(gate['adapter_code']);assert gate['adapter_code']==b(Path(__file__)) and gate['approved'] and gate['source_model_freeze']==b(F)
 f=load(F);assert f['completed'] and f['optimizer_updates']==2000 and f['parameter_count']==41237 and f['visible_probability_threshold']==.5 and not f['current_used_for_fit_selection'] and f['PnP']==0
 for k in ('checkpoint','model_code','protocol','trainer','entry','source_inputs'):check(f[k])
 assert not OUT.exists();OUT.mkdir(parents=True)
 oldgate=load(OLD/'private_evaluation/CURRENT_INPUT_ADAPTER_GATE.json');oldprep=load(OLD/'private_evaluation/CUE_ATTENTION_V3/PREPARATION_SEAL.json')
 assert oldprep['completed'] and oldprep['frames']==223 and oldprep['slots']==1784
 for z in oldprep['outputs']:check(z)
 for z in oldgate['inputs'].values():check(z)
 pp=Path(oldgate['inputs']['prepared_inputs']['path']);frames=json.loads(gzip.open(pp,'rt').read()if pp.suffix=='.gz'else pp.read_text());metadata=load(oldgate['inputs']['feature_metadata']['path']);assert len(frames)==len(metadata)==223
 priorrows=[json.loads(x)for x in (OLD/'private_evaluation/CUE_ATTENTION_V3/ROWS_INPUT_ONLY.jsonl').read_text().splitlines()]
 with np.load(OLD/'private_evaluation/CUE_ATTENTION_V3/CURRENT_MODEL_INPUT.npz',allow_pickle=False)as z:data={k:z[k]for k in z.files}
 assert data['p3_roi'].shape==(1784,65,9,9) and len(priorrows)==1784
 C=extract(P/'query_visibility_loco_native_context_20261023_v74/native_context_cache/compile_input_pack.py',{'crop'})['crop']
 A=extract(OLD/'private_evaluation/current_input_adapter.py',{'initial_query_box','query_validity'})
 H=extract(P/'query_visibility_loco_native_context_20261023_v74/source_contract/native_context_contract_v2.py',{'input_context_features'})['input_context_features']
 targets=[];tv=[];contexts=[];boxes=[];originalq=[];originalL=[];images=[]
 for i,(frame,meta)in enumerate(zip(frames,metadata)):
  assert frame['id']==meta['id'] and all(priorrows[i*8+k]['id']==frame['id']for k in range(8))
  q=np.asarray(frame['initial_points'],np.float64)[:8];L=float(meta['side']);hw=frame['raw_hw'];box=A['initial_query_box'](q,hw)
  assert np.array_equal(q.astype(np.float32),data['query_xy'][i*8:(i+1)*8],equal_nan=True) and (np.float32(L)==data['side_L'][i*8:(i+1)*8]).all()
  ib=dict(path=frame['image'],sha256=frame['image_sha256']);check(ib);rgb=np.array(Image.open(frame['image']).convert('RGB'));assert list(rgb.shape[:2])==hw
  if box is None:target=np.zeros((3,128,128),np.uint8);valid=np.zeros((1,128,128),bool);safe_box=np.zeros(4)
  else:
   target,valid=C(rgb,(box[:2]+box[2:])/2,1.25*float(np.max(box[2:]-box[:2])),128);safe_box=box
  targets.append(target);tv.append(valid);contexts.extend([H(q,k,safe_box,hw)for k in range(8)]);boxes.append(safe_box);originalq.append(q);originalL.append(L);images.append(ib)
 inputs=dict(p3_roi=np.ascontiguousarray(data['p3_roi']),local_rgb96=np.ascontiguousarray(data['rgb96']),local_valid96=np.ascontiguousarray(data['crop_valid96']),target_rgb128=np.stack(targets),target_valid128=np.stack(tv),context42=np.stack(contexts))
 assert all(np.isfinite(x).all()for x in inputs.values())
 np.savez_compressed(OUT/'CURRENT_SIX_INPUTS.npz',**inputs)
 save(OUT/'INPUT_PREPARATION_SEAL.json',dict(completed=True,source_freeze=b(F),gate=b(ROOT/'private_evaluation/POST_SOURCE_FREEZE_GATE.json'),code=b(Path(__file__)),input=b(OUT/'CURRENT_SIX_INPUTS.npz'),prior_preparation=b(OLD/'private_evaluation/CUE_ATTENTION_V3/PREPARATION_SEAL.json'),images=images,all223_same_q_L_P3_RGB96=True,current_GT_or_annotations_or_scores_read=False,new_backbone=0,fit=0,PnP=0))
 torch.set_num_threads(1);torch.manual_seed(20261023);sp=importlib.util.spec_from_file_location('frozen_support',f['model_code']['path']);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);model=m.NativeContextSupportHead();model.load_state_dict(torch.load(f['checkpoint']['path'],map_location='cpu',weights_only=True));model.eval().requires_grad_(False);before={k:v.clone()for k,v in model.state_dict().items()}
 logits=[];avs=[];repeat_max=0.;start=time.monotonic()
 with torch.inference_mode():
  for a in range(0,1784,64):
   ids=np.arange(a,min(a+64,1784));v={k:torch.from_numpy(np.ascontiguousarray(x[ids//8 if k in ('target_rgb128','target_valid128')else ids]))for k,x in inputs.items()};out=model(**v);again=model(**v);repeat_max=max(repeat_max,float((out['support_logit']-again['support_logit']).abs().max()));logits.extend(out['support_logit'].numpy().tolist());avs.extend(out['input_available'].numpy().tolist())
 assert repeat_max==0. and all(torch.equal(before[k],v)for k,v in model.state_dict().items());check(f['checkpoint']);z=np.asarray(logits,np.float32);assert np.isfinite(z).all();av=np.asarray(avs,bool)
 rows=[]
 for i,r0 in enumerate(priorrows):
  r=dict(r0);valid=bool(r['valid'])and bool(av[i]);prob=float(np.exp(-np.logaddexp(0.,-float(z[i]))))
  r.update(valid=valid,status='OK'if valid else'UNKNOWN',logit=float(z[i])if valid else None,p_visible=prob if valid else None,present_at_frozen_threshold=bool(z[i]>=0)if valid else None,predicted_D_or_E=('D'if z[i]>=0 else'E')if valid else'UNKNOWN',raw_support_logit=float(z[i]),raw_p_local_support_visible=prob,support_target='REFERENCE_LOCAL_EXPECTED_TARGET_SUPPORT_WEAK_TEACHER',semantic_U_detector=False,self_occlusion_learned=False)
  rows.append(r)
 with (OUT/'REAL_QUERY_SCORES.csv').open('w',newline='')as fh:w=csv.DictWriter(fh,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 np.savez_compressed(OUT/'RAW_NETWORK_OUTPUTS.npz',support_logit=z.reshape(223,8),input_available=av.reshape(223,8),valid=np.asarray([r['valid']for r in rows]).reshape(223,8),original_query8=np.stack(originalq),original_L=np.asarray(originalL))
 save(OUT/'INFERENCE_SEAL.json',dict(completed=True,source_model_freeze=b(F),checkpoint=f['checkpoint'],code=b(Path(__file__)),preparation=b(OUT/'INPUT_PREPARATION_SEAL.json'),outputs=[b(OUT/'REAL_QUERY_SCORES.csv'),b(OUT/'RAW_NETWORK_OUTPUTS.npz')],frames=223,slots=1784,valid=sum(r['valid']for r in rows),threshold=.5,threshold_comparison='>=',repeat_logit_max_difference=repeat_max,model_unchanged=True,current_used_for_selection=False,current_GT_read=False,PnP=0,new_backbone=0,optimizer_updates=0,seconds=time.monotonic()-start))
 print(json.dumps(dict(completed=True,valid=sum(r['valid']for r in rows),seal=b(OUT/'INFERENCE_SEAL.json'))),flush=True)
if __name__=='__main__':main()
