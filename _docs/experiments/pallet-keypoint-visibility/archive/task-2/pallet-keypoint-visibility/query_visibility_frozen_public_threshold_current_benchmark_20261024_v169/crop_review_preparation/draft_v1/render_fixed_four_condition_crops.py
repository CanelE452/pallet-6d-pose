"""Prospective V169 fixed crop readout. No current bodies during preparation.
Evaluation must freeze first; separate root visual entry then seals IDs before pixels.
"""
from pathlib import Path
import argparse,csv,hashlib,html,importlib.util,itertools,json,sys
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
PARENT=P/'query_visibility_separated_frozen_current_benchmark_20261024_v156/crop_review_preparation/render_fixed_saved_crops_alias_repair.py'
PARENT_PROTOCOL=PARENT.parent/'PROTOCOL_BEFORE_ALIAS_REPAIR_RENDER.json'
PARENT_SHA='1489d086d1c72d9ff92c6b5677fbd6eeb8d56ff663f831134f8e58adfb0ff80c'
PARENT_PROTOCOL_SHA='f0fe65cf9e677ec68598c5064bb94e780542d9cf60d848beaff8b7cd45c0f8c5'
POLICY=H/'SELECTION_RULE_BEFORE_CURRENT_BODIES.json'
PROTOCOL=H/'PROTOCOL_BEFORE_CURRENT_CROP_BODIES.json'
EVAL=V/'saved_seven_condition_evaluation_v1/EVALUATION_SEAL.json'
LEDGER=EVAL.parent/'ALL1784_NATIVE_LEDGER.csv'
OUT=V/'fixed_current_crop_review_v1'
CONDITIONS=('C45_ORIGINAL0','C45_PUBLICCAL_FULL','C46_ORIGINAL0','C46_PUBLICCAL_FULL')
BASELINES=('v61','legacy','C16');ALL_CONDITIONS=BASELINES+CONDITIONS
STRATA=('TP','FN','FP','TN','U')
RGB_KEYS=('rgb96','crop_valid96','query_xy','side_L')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def write(p,r):
 with Path(p).open('x') as f:json.dump(r,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def number(x):
 try:v=float(x);return v if np.isfinite(v) else None
 except (TypeError,ValueError):return None
def boolean(x):return x is True or x=='True'
def material(r):
 if r['group'].startswith('WOOD'):return 'WOOD'
 assert r['group'].startswith('SQUARE');return 'SQUARE'
def stratum(r,condition):
 pred=r[condition+'_D_E_U'];assert pred in ('D','E','U')
 if pred=='U':return 'U' # Any original row with factual model U; unknown truth is not recoded.
 truth={'DIRECT_VISIBLE':'D','EXTERNAL_OCCLUDED':'E'}.get(r['human_state'])
 return {('E','E'):'TP',('E','D'):'FN',('D','E'):'FP',('D','D'):'TN'}.get((truth,pred))
def select_fixed(rows):
 rows=sorted(rows,key=lambda r:int(r['global_index']));buckets=[];selected={}
 for condition in CONDITIONS:
  for mat in ('WOOD','SQUARE'):
   for category in STRATA:
    candidates=[r for r in rows if material(r)==mat and stratum(r,condition)==category]
    key=f'{condition}::{mat}::{category}';choice=candidates[0] if candidates else None
    buckets.append(dict(key=key,eligible_rows=len(candidates),selected_global_index=None if choice is None else int(choice['global_index']),empty=choice is None))
    if choice is not None:selected.setdefault(int(choice['global_index']),[]).append(key)
   candidates=[r for r in rows if material(r)==mat and r['human_state']=='DIRECT_VISIBLE' and r['annotation_coordinate_source']=='manual_click' and boolean(r['original_query_valid']) and number(r['initial_reference_error_px']) is not None and number(r['initial_reference_error_px'])<=3 and r[condition+'_D_E_U']=='E']
   key=f'{condition}::{mat}::MANUAL_REFERENCE_LE3_DIRECT_PRED_E';choice=candidates[0] if candidates else None
   buckets.append(dict(key=key,eligible_rows=len(candidates),selected_global_index=None if choice is None else int(choice['global_index']),empty=choice is None,physical_3px_gold=False))
   if choice is not None:selected.setdefault(int(choice['global_index']),[]).append(key)
 assert len(buckets)==48 and len(selected)<=48
 return buckets,selected

def prepare():
 assert sha(PARENT)==PARENT_SHA and sha(PARENT_PROTOCOL)==PARENT_PROTOCOL_SHA
 old=read(PARENT_PROTOCOL) # Descriptor/code only; never check/hash/read its current inputs here.
 policy=dict(completed=True,prepared_before_current_bodies=True,conditions=list(CONDITIONS),materials=['WOOD','SQUARE'],strata=list(STRATA),
  selection='For each condition×material×TP/FN/FP/TN/U choose lowest original global_index; empty remains empty, no replacement or frame preference. Dedup exact global_index across48 buckets.',
  U='U stratum includes any original row with decodedU. TP/FN/FP/TN require existing DIRECT_VISIBLE or EXTERNAL_OCCLUDED; self/uncertain are not negatives.',
  extra8='Per condition/material lowest global_index with existing DIRECT_VISIBLE, manual_click reference, finite existing initial_reference_error_px<=3, original_query_valid and predictionE. Fixed reference diagnostic, not physical3px gold.',
  maximum_unique_cases=48,selection_before_RGB=True,changed_ledger='Keep full1784 ledger plus exhaustive differences for every unordered pair of all7 baseline/new conditions; no omitted changed row. Changed ledger is not extra RGB selection.',
  no_new_labels_or_q_L_correction=True,no_actual_current_bodies_read=True)
 if not POLICY.exists():write(POLICY,policy)
 else:assert read(POLICY)==policy
 keys=('original_cache','prepared_frames','feature_metadata','C11_frame_facts','fullframe_decoder')
 protocol=dict(completed=True,prepared_only=True,code=bind(__file__),policy=bind(POLICY),parent_alias_renderer=bind(PARENT),parent_alias_protocol=bind(PARENT_PROTOCOL),
  **{k:old[k] for k in keys},conditions=list(CONDITIONS),baselines=list(BASELINES),evaluation_seal_path=str(EVAL),ledger_path=str(LEDGER),output=str(OUT),
  pipeline='Separate approved visual entry + frozen completed evaluation seal -> unchanged1784 ledger -> selection manifest durable BEFORE any RGB/cache hash/read -> exact original q/L metadata -> original pixel-cache whitelist only -> safe original alias SHA decode -> clean PNGs/card.',
  fullRGB_alias='Reuse V156 safe_original_image_path exactly; original descriptor SHA, symlink expansion guard, original alias descriptor equality, decoded RGB SHA. Do not require alias string to equal canonical spelling.',
  clean_crop='Original uint8CHW96 and boolvalid; save clean HWC raw96/central32[32:64]/valid PNG before overlays; verify decoded bytes. Whole-frame context with all originalnativeq is display only, not a new DINOtarget224 claim.',
  evaluator_schema='C45_ORIGINAL0/C45_PUBLICCAL_FULL/C46_ORIGINAL0/C46_PUBLICCAL_FULL *_D_E_U,*_original_response,*_raw_visibility_logit,*_threshold_float64,*_decision_score64 plus unchanged baseline responses and original identity/truth/qL/reference fields; final evaluator binding supplied only after freeze.',
  current_feature_label_score_RGB_body_or_cache_hash_reads=0,new_model_encoder_fit_PnP_qL_calls=0,actual_rendering=False)
 write(PROTOCOL,protocol);print(json.dumps(dict(prepared_only=True,code=bind(__file__),policy=bind(POLICY),protocol=bind(PROTOCOL))))

def run(entry):
 raise RuntimeError('Draft: complete evaluator-seal adapter and root final review required; no current body or pixel entry is available in this draft')

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
 if a.prepare and not a.entry:prepare()
 elif a.entry and not a.prepare:run(a.entry)
 else:raise SystemExit('Metadata-only --prepare; actual visualization requires separately approved frozen-evaluation entry')
