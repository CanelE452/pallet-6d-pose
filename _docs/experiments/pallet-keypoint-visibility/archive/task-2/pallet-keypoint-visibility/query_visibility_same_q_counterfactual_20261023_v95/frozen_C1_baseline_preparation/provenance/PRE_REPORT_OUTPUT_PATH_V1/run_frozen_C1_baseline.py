"""Immutable C1 checkpoint on the chosen V95 sparse source pack, diagnostic only.
No fitting, checkpoint/threshold selection, initializer/backbone, or private IO.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
T=V/'source_fit_preparation/train_source_once.py'
T_SHA='92b50f1c6d65aedd88ce5bf2064f6cfdff7136620beaabc1b6297d63d6c6157b'
FREEZE=P/'query_visibility_matched_query_scalar_20261023_v94/runs/C1_NATURAL_KNOWNSET_SOURCE_V1/MODEL_FREEZE.json'
FREEZE_SHA='e276f2230902ef29120c6a0c88d5c106fc089666eebcfe636cd05634f5cd7064'
PROTOCOL=H/'BASELINE_PROTOCOL_BEFORE_SOURCE_READS.json'
OUT=V/'frozen_C1_source_baseline/BASELINE_RUN'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(d):assert sha(d['path'])==d['sha256'];return Path(d['path'])
def read(p):return json.loads(Path(p).read_text())
def write(p,o):
 with Path(p).open('x')as f:json.dump(o,f,indent=2,allow_nan=False);f.write('\n')
def same(a,b):return Path(a['path']).resolve()==Path(b['path']).resolve()and a['sha256']==b['sha256']
def trainer():
 assert sha(T)==T_SHA;s=importlib.util.spec_from_file_location('v95_frozen_shared_loader_metrics',T);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def guard(e):
 assert e.get('root_approved')is True and e.get('source_only')is True
 assert e.get('current_allowed')is False and e.get('training_allowed')is False and e.get('diagnostic_not_selector')is True
 assert e.get('PnP_allowed')is False
 for k in('code','protocol','generated','independent_review','trainer_code','model_freeze','checkpoint','input_pack_manifest','input_pack_seal','base_input_lineage','GT_ledger','independent_input_audit'):check(e[k])
 assert same(e['code'],bind(__file__))and check(e['protocol'])==PROTOCOL
 assert same(e['trainer_code'],bind(T))and e['trainer_code']['sha256']==T_SHA
 assert same(e['model_freeze'],dict(path=str(FREEZE),sha256=FREEZE_SHA))
 assert read(e['generated']['path'])['PASS']and same(read(e['generated']['path'])['code'],e['code'])
 assert read(e['independent_review']['path'])['PASS']and read(e['independent_input_audit']['path'])['PASS']
 f=read(e['model_freeze']['path']);assert f['completed']and f['source_only']and f['selected_update']==100 and same(f['checkpoint'],e['checkpoint'])
 assert f['source_input_bindings']['current_allowed']is False
 assert Path(e['output_directory']).resolve()==OUT.resolve()and not OUT.exists()
 return read(PROTOCOL),f

def state_sha(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):
  a=v.detach().cpu().contiguous().numpy();h.update(k.encode()+str(a.dtype).encode()+str(a.shape).encode()+a.tobytes())
 return h.hexdigest()

def diagnose(z,available,meta,t):
 results,_,_=t.report(z,available,meta)
 return dict(completed=True,diagnostic_only=True,model_selection_used=False,metrics=results,counts=meta['counts'],threshold_rawlogit=0,
  scalar='D0 iff z<0, auxiliary hiddenE1 iff z>=0; unavailable U-1',all2880_preserved=True,
  raw_split_preserved=True,canonical_map='LOCO TRAIN/VALIDATION; PALLOX TRAIN_LAB/VAL_HALL only',
  source_semantics='Conditional known-cue-set and controlled cover; native mapping UNKNOWN, human_gold false.',
  current_authorized=False,new_fit_checkpoint_selection_or_PnP=0)

def run(entry_path):
 e=read(entry_path);p,freeze=guard(e) # all source/pixels/parameter reads below root authorization
 import torch
 assert torch.__version__=='2.1.1+cu118'and torch.cuda.is_available()
 t=trainer();seal=read(e['input_pack_seal']['path']);assert seal['completed']and seal['PASS']and same(seal['manifest'],e['input_pack_manifest'])
 check(seal['QA']);assert read(seal['QA']['path'])['PASS']
 manifest=read(e['input_pack_manifest']['path']);assert manifest['completed']and manifest['base_rows']==960 and manifest['occurrences']==2880
 assert same(manifest['base_input_lineage'],e['base_input_lineage']);check(manifest['entry']);assert same(read(manifest['entry']['path'])['occurrences'],e['GT_ledger'])
 check(manifest['input_lineage']);lineage=t.rows(manifest['input_lineage']['path']);meta=t.join_metadata(lineage,t.rows(e['GT_ledger']['path']))
 data,factual=t.load_inputs(e,manifest,lineage)
 OUT.mkdir(parents=True,exist_ok=False);write(OUT/'STARTED.json',dict(entry=bind(entry_path),frozen_C1=e['model_freeze'],new_fit_current_PnP=0))
 torch.set_num_threads(1);torch.manual_seed(194);torch.cuda.manual_seed_all(194);torch.use_deterministic_algorithms(True)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 op=t.operator();model=op.QueryCenterScalar().to('cuda:0');check(freeze['model_code']);assert same(freeze['model_code'],bind(t.MODEL))
 payload=torch.load(e['checkpoint']['path'],map_location='cuda:0',weights_only=True)
 assert payload['update']==100 and payload['parameter_count']==26573 and same(payload['model_code'],freeze['model_code'])and same(payload['protocol'],freeze['protocol'])
 model.load_state_dict(payload['state_dict'],strict=True);model.eval().requires_grad_(False);before=state_sha(model)
 with torch.inference_mode():
  ix=np.arange(128);a=t.forward(model,data,ix,'cuda:0');b=t.forward(model,data,ix,'cuda:0')
  assert torch.equal(a['query_logit'],b['query_logit'])and torch.equal(a['query_available'],b['query_available'])
  z,available=t.predict(model,data,'cuda:0')
 assert np.array_equal(available,factual)and np.isfinite(z).all()and state_sha(model)==before
 assert np.all(z[~available]==0)
 np.savez_compressed(OUT/'C1_ALL2880_SOURCE_SCORES.npz',query_logit=z.astype(np.float32),query_available=available,prediction=t.decode(z,available),
  scalar_target_D0_E1_Uminus1=meta['labels'],candidate_known=meta['known'],source=meta['source'],split=meta['split'],original_split=meta['original_split'],variant=meta['variant'],
  occurrence_index=np.arange(2880),base_row_index=np.arange(2880)//3,occurrence_id=np.array([r['occurrence_id']for r in lineage]),
  input_id=np.array([r['input_id']for r in lineage]),native_slot_logical_only=np.array([r['native_slot_logical_only']for r in lineage]),
  image_sha256=np.array([r['image_sha256']for r in lineage]),original_image_sha256=np.array([r['original_image_sha256']for r in lineage]),accepted_pair=meta['pair_ready'])
 report=diagnose(z,available,meta,t);report.update(model_freeze=e['model_freeze'],checkpoint=e['checkpoint'],entry=bind(entry_path),input_pack_manifest=e['input_pack_manifest'],model_state_sha256=before,first128_repeat_bitexact=True)
 write(OUT/'C1_ON_V95_SOURCE_RESULTS.json',report)
 for k in('code','trainer_code','checkpoint','model_freeze'):check(e[k])
 write(OUT/'BASELINE_COMPLETE.json',dict(completed=True,PASS_execution=True,entry=bind(entry_path),code=bind(__file__),model_freeze=e['model_freeze'],checkpoint=e['checkpoint'],
  input_pack=e['input_pack_manifest'],results=bind(OUT/'C1_ON_V95_SOURCE_RESULTS.json'),scores=bind(OUT/'C1_ALL2880_SOURCE_SCORES.npz'),model_unchanged=True,new_fitting_backbone_initializer_private_current_PnP=0,diagnostic_not_selector=True))
 print(json.dumps(bind(OUT/'BASELINE_COMPLETE.json')))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--entry',required=True);run(ap.parse_args().entry)
