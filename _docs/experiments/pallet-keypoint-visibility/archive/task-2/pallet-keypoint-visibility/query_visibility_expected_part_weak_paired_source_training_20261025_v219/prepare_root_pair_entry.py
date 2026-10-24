"""Root exact SOURCE pair entry preparation; no arrays/images/models opened here."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path

V=Path(__file__).parent
P=V.parent
H=P/'publication_root_preparation/V219_TRAINING_PREPARATION_C'

def body(p):
 p=Path(p);assert p.is_absolute() and '..' not in p.parts and 'v112' not in str(p).lower()
 assert p.is_relative_to(P)
 for q in reversed([p,*p.parents]):assert not q.is_symlink()
 with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:return f.read()
def bind(p):
 b=body(p);return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),size_bytes=len(b))
def load(p):return json.loads(body(p))
def new(p,x):
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')

def main(mode,steps):
 assert mode in ['THROUGHPUT_PROBE','FIXED_PAIRED_FIT'] and type(steps) is int and steps>0
 a=load(V/'source_metadata_assembly_a/FINAL_READY.json')
 data=load(V/'source_fit_rows_v1/COMPLETE.json');assert data['completed'] and data['PASS'] and data['all_selected_masks_true'] and data['unselected_numeric_rows_decoded']==0
 q=load(V/'independent_training_code_review_b/REVIEW.json');assert q['completed'] and q['PASS']
 assert bind(H/'FINAL_READY.json')['sha256']=='832e04ebc781ada27dc0d01cb9a4e60cec05bd3e9f3cdff1945fd199555c707a'
 e=load(H/'ROOT_ACTUAL_V219_SOURCE_PAIR_ENTRY_FALSE_TEMPLATE.json')
 if mode=='THROUGHPUT_PROBE':
  assert steps<=32
  scope=V/'ROOT_ACTUAL_PAIRED_SOURCE_PROBE_SCOPE.json'
  new(scope,dict(completed=True,approved_source_pair_fit_or_probe=True,source_only=True,actual_mode=mode,source_inputs=bind(V/'source_fit_rows_v1/COMPLETE.json'),source_code_preflight=bind(V/'independent_training_code_review_b/REVIEW.json'),source_cache_policy=bind(V/'ROOT_PROSPECTIVE_FIT_ONLY_SOURCE_CACHE_SELECTION.json'),no_old_checkpoint=True,no_protected_evaluation_or_PnP=True,source_role='Only fit-only extracted source caches and weak45 FIT PNGs are used in disposable probe. No sourceDEV forwards or weakDEV PNGs.',goal_status='active',fully_solved=False))
  for filename,flag,authority,target,census,description in [
   ('ROOT_ADMITTED_SYNTHETIC_POINT_TARGETS.json','approved_strong_synthetic_point_targets','SYNTH_RENDERED_OWN_IMAGE_POINT0_TARGET_ALONE_D_COMBINED_DE',a['point_metadata7040'],dict(original_rows=7040,D=5853,E=243,excluded_UNKNOWN_or_other=944), 'Stored synthetic own-image point0 target. Positive target-alone D and combined D/E with exact own-image mapping, unchanged original q/L/U and actual crop/frame locality. S/U/outside/unknown remain -1. GT/reference is target-side only.'),
   ('ROOT_ADMITTED_PUBLIC_EXPECTED_PART_WEAK_TARGETS.json','approved_public_expected_part_weak_targets','ASSISTANT_CONSENSUS_EXPECTED_FINITE_PART_SOURCE_GT_QUERY',a['weak_rows360'],dict(all_rows=360,known=57,FIT_D=39,FIT_E=6,DEV_D=9,DEV_E=3), 'Independent assistant expected-finite-part weak consensus, not exact native point/human gold. q8 itself is GT derived in separate public training family. No additional GT pose/reference/ROI/mask feature. original query96/U11 unchanged. SOURCEDEV12 never loss/norm, only after both final states freeze.')]:
   new(V/filename,dict(completed=True,**{flag:True},authority=authority,target_metadata=target,census=census,interpretation=description,no_perfect_physical_point_gold_blanket_gate=True,private_evaluation_fit_or_selection=False))
 else:
  probe=load(V/'paired_source_probe_v1/PROBE_COMPLETE.json');assert probe['completed'] and probe['PASS']
  scope=V/'ROOT_ACTUAL_FIXED_PAIRED_SOURCE_FIT_SCOPE.json'
  new(scope,dict(completed=True,approved_source_pair_fit_or_probe=True,source_only=True,actual_mode=mode,source_inputs=bind(V/'source_fit_rows_v1/COMPLETE.json'),source_code_preflight=bind(V/'independent_training_code_review_b/REVIEW.json'),probe=bind(V/'paired_source_probe_v1/PROBE_COMPLETE.json'),both_final_freeze_before_DEV=True,models=['C64_strong_source_baseline','C65_strong_source_plus_expected_part_weak'],both_arms_retained_no_selection=True,no_old_checkpoint=True,no_protected_evaluation_or_PnP=True,goal_status='active',fully_solved=False))
 seconds_per_update=None
 if mode=='FIXED_PAIRED_FIT':
  measured=load(V/'paired_source_probe_v1/PROBE_COMPLETE.json');seconds_per_update=measured['seconds']/measured['updates']
 recipe=V/('ROOT_THROUGHPUT_PROBE_RECIPE.json' if mode=='THROUGHPUT_PROBE' else 'ROOT_FIXED_PAIRED_SOURCE_RECIPE.json')
 new(recipe,dict(completed=True,approved_frozen_recipe=True,budget_selection_basis='THROUGHPUT_ONLY_NO_SCORE_OR_ARM_SELECTION',steps=steps,learning_rate=.001,weight_decay=.0001,weak_weight=.25,initial_seed=1810,old_draw_seed=1810,strong_draw_seed=1811,weak_draw_seed=1812,threshold_logit=0.,force_binary_for_finite_known_inputs=True,seconds_per_update_probe=seconds_per_update,mode=mode,budget_policy='Probe <=32 updates/arm; full budget fixed from runtime only before any sourceDEV score. Both final models retained. No early stop/arm/checkpoint/threshold selection.'))
 e.update(approved_source_pair_fit_or_probe=True,mode=mode,root_scope=bind(scope),root_recipe=bind(recipe),strong_target_admission=bind(V/'ROOT_ADMITTED_SYNTHETIC_POINT_TARGETS.json'),weak_target_admission=bind(V/'ROOT_ADMITTED_PUBLIC_EXPECTED_PART_WEAK_TARGETS.json'),lineage_assembly_QA=a['QA'],old_caches=data['old_caches'],point_metadata7040=a['point_metadata7040'],expected_strong_D_E_counts=a['expected_strong_D_E_counts'],weak_rows360=a['weak_rows360'],output_directory=str(V/('paired_source_probe_v1' if mode=='THROUGHPUT_PROBE' else 'paired_source_fit_v1')))
 e['metadata_whitelist']=[e[k] for k in ['root_scope','root_recipe','strong_target_admission','weak_target_admission','lineage_assembly_QA','point_metadata7040','weak_rows360']]+[r['metadata'] for r in e['old_caches'].values()]
 rows=[json.loads(s) for s in body(e['weak_rows360']['path']).decode().splitlines()]
 known=[r['generated_input']['RGB96'] for r in rows if r['weak_loss_candidate']]
 assert len(known)==57
 e['binary_whitelist']=[r[k] for r in e['old_caches'].values() for k in ['inputs','targets']]+known
 target=V/('ROOT_ACTUAL_SOURCE_PAIR_PROBE_ENTRY.json' if mode=='THROUGHPUT_PROBE' else 'ROOT_ACTUAL_FIXED_SOURCE_PAIR_FIT_ENTRY.json')
 new(target,e);print(json.dumps(dict(completed=True,entry=bind(target),mode=mode,steps_each=steps,source_only=True)))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['THROUGHPUT_PROBE','FIXED_PAIRED_FIT']);ap.add_argument('--steps',required=True,type=int);a=ap.parse_args();main(a.mode,a.steps)
