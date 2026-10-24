import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime, timezone

P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
R=P/'query_visibility_matched_query_scalar_20261023_v94'
O=R/'pallox_actual_query_metadata_audit'
D=R/'pallox_actual_initializer/FIXED100'
V89=P/'query_visibility_public_cue_grid_20261023_v89'
V90=P/'query_visibility_public_cue_inputs_20261023_v90'
V91=P/'query_visibility_physical_cue_aux_20261023_v91'
F={
 'root_scope':R/'SCOPE_BEFORE_MATCHED_QUERY_CONTRACT.json',
 'actual_entry':R/'ROOT_ACTUAL_PALLOX100_INITIALIZER_ENTRY.json',
 'protocol':R/'pallox_actual_initializer_preparation/PROTOCOL_BEFORE_SOURCE_FORWARD.json',
 'producer':R/'pallox_actual_initializer_preparation/infer_fixed_pallox_queries.py',
 'intake_seal':D/'INTAKE_SEAL.json','result':D/'RESULT.json',
 'proposals100':D/'FROZEN_PROPOSALS_ALL100.jsonl',
 'queries800':D/'ORIGINAL_NATIVE_QUERY800_INPUT_ONLY.jsonl',
 'frames100':V89/'inputs_metadata/FRAMES_INPUTS_ONLY.jsonl',
 'v90_frames':V90/'input_pack/FRAME_FULL_P3_RECORDS.jsonl',
 'v90_seal':V90/'input_pack/INPUT_PACK_SEAL.json',
 'teacher_views604':V91/'teacher_preparation/corpus_build_v1/ALL604_CUE_VIEW_TEACHERS_GT_ONLY.jsonl',
 'teacher_cues302':V91/'teacher_preparation/corpus_build_v1/ALL302_CUES_SOURCE_IDS_GT_ONLY.jsonl',
 'teacher_contract':V91/'teacher_preparation/TEACHER_CONTRACT_BEFORE_BUILD.json',
 'teacher_seal':V91/'teacher_preparation/corpus_build_v1/GT_CORPUS_SEAL.json',
}
def bind(p):
 b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def load(k):return json.loads(F[k].read_text())
def rows(k):return [json.loads(x) for x in F[k].read_text().splitlines() if x.strip()]
def write(p,x):
 assert not p.exists(),str(p);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def dump_rows(p,rr):
 assert not p.exists();p.write_text(''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in rr))
def coverage(box,q,L):
 if q is None:return 'NO_QUERY'
 if box is None:return 'NULL_REFERENCE_U'
 x0,y0,x1,y1=box;assert x0<x1 and y0<y1
 a,b=q[0]-L/2,q[1]-L/2;c,d=q[0]+L/2,q[1]+L/2
 if a<=x0 and b<=y0 and x1<=c and y1<=d:return 'FULL'
 if min(x1,c)>max(x0,a) and min(y1,d)>max(y0,b):return 'PARTIAL'
 return 'MISS'
def validity(q,hw):
 if q is None:return False,False,False
 x,y=q;h,w=hw
 finite=math.isfinite(x) and math.isfinite(y)
 if not finite:return False,False,False
 return (0<=x<w and 0<=y<h,
         0<=math.floor(x+.5)<w and 0<=math.floor(y+.5)<h,
         -.5<=x<w-.5 and -.5<=y<h-.5)
def L_from_raw9(raw,hw):
 good=[p for p in raw[:8] if all(math.isfinite(z) for z in p) and p!=[-1,-1]]
 if len(good)<2:return 16.
 span=max(max(p[0] for p in good)-min(p[0] for p in good),max(p[1] for p in good)-min(p[1] for p in good))
 return float(min(max(.25*span,16),max(hw)))
def generated():
 assert coverage([0,0,2,2],[1,1],2)=='FULL'
 assert coverage([0,0,2,2],[2,1],2)=='PARTIAL'
 assert coverage([0,0,2,2],[3,1],2)=='MISS'
 assert coverage(None,[1,1],2)=='NULL_REFERENCE_U'
 assert coverage([0,0,2,2],None,None)=='NO_QUERY'
 assert validity([-.5,0],[480,640])==(False,True,True)
 assert validity([639.5,0],[480,640])==(True,False,False)
 assert validity([640,0],[480,640])==(False,False,False)
 assert L_from_raw9([[0,0],[100,200]]+[[0,0]]*6+[[100000,100000]],[480,640])==50.
 write(O/'GENERATED_METADATA_BOUNDARY_FIXTURES.json',dict(PASS=True,fixtures=9,producer_imports=0,pixels_arrays_models=0))
def hist_cat(n):return '0' if n==0 else '1' if n==1 else '>1'

def aggregate(rr):
 out=dict(N=len(rr),query_status=dict(Counter(x['query_status'] for x in rr)),nativecontinuous=sum(x['nativecontinuous_valid'] for x in rr),nearest=sum(x['nearest_valid'] for x in rr),RGBcentre=sum(x['RGBcentre_valid'] for x in rr),continuous_nearest_disagreement=sum(x['nativecontinuous_valid']!=x['nearest_valid'] for x in rr),nearest_RGBcentre_disagreement=sum(x['nearest_valid']!=x['RGBcentre_valid'] for x in rr),
  canonical_known_FULL_counts_all=dict(Counter(hist_cat(x['canonical_known_FULL_count']) for x in rr)),canonical_applicable_queries=sum(x['canonical_namespace_applicable'] for x in rr),
  canonical_known_FULL_counts_applicable=dict(Counter(hist_cat(x['canonical_known_FULL_count']) for x in rr if x['canonical_namespace_applicable'])),
  canonical_known_FULL_counts_nativevalid=dict(Counter(hist_cat(x['canonical_known_FULL_count']) for x in rr if x['nativecontinuous_valid'] and x['canonical_namespace_applicable'])),
  any_namespace_known_FULL_counts_all=dict(Counter(hist_cat(x['any_known_FULL_raw_count']) for x in rr)),
  distinct_certified_groups_FULL_counts_all=dict(Counter(hist_cat(x['known_FULL_certified_group_count']) for x in rr)),
  mixed_D_E_FULL_raw=sum(x['known_FULL_raw_mixed_D_E'] for x in rr),canonical_mixed_D_E_FULL=sum(x['canonical_FULL_mixed_D_E'] for x in rr),
  nativevalid_mixed_D_E_FULL_raw=sum(x['nativecontinuous_valid'] and x['known_FULL_raw_mixed_D_E'] for x in rr),
  seed_queries=sum(x['seed_namespace_applicable'] for x in rr),
  seed_known_FULL_raw_count_hist=dict(Counter(hist_cat(x['any_known_FULL_raw_count']) for x in rr if x['seed_namespace_applicable'])),
  seed_certified_groups_FULL_count_hist=dict(Counter(hist_cat(x['known_FULL_certified_group_count']) for x in rr if x['seed_namespace_applicable'])))
 return out

def main():
 write(O/'SCOPE_AND_ARITHMETIC_BEFORE_ALLROW_REPLAY.json',dict(schema='V94_ACTUAL_INITIALIZER_METADATA_AUDIT_POLICY',created_UTC=datetime.now(timezone.utc).isoformat(),bindings={k:bind(p) for k,p in F.items()},
  prior_exposure='Root summary and first serialized row/schema were visible before all-row arithmetic; no data-dependent selection/radius/threshold. Full fixed cohort is mandatory.',
  replay='Independent Python arithmetic; no import of producer selected_queries or V60 native_grid. First maximum score in saved candidate order; raw9 centroid excluded; first8 finite nonmissing span, quarter clipped16..maxHW.',
  validity='Nativecontinuous [0,W)x[0,H), nearest floor(q+.5), RGB-centre [-.5,W-.5) retained separately. No field silently replaces another.',
  coverage='All604cueviews x original8queries. Nullreference and noquery separately flagged even when their combined geometry code is NO_QUERY. Defined box positive-area overlap/containment uses unchanged float64 half-open q±L/2. OOB is a separate query flag; its underlying geometry is retained.',
  groups='V85 canonical slots never newly merged. V84 original8 retained; existing duplicate_group_link_only/size/weight records alone identify certified6 groups. Count >=1 originalFULL per certified group descriptively, no coordinate union or label repair.',
  per_query='0/1/>1 FULL canonical knowncue counts and separate seed/group counts. Mixed D/E is an annotation/candidate incompatibility diagnostic, not a query label. Never nearest/bestcue selection.',
  producer_certificate='P3/state/repeat/RGB equality independently bound to saved producer/entry/result assertions and V90 metadata; no arrays or weights read and no independent forward or pixel replay claim.',
  pixels_arrays_models_current_forward_fit_PnP=0,training_admission=False))
 generated()
 se=load('intake_seal');res=load('result');pr=load('protocol')
 assert se['completed'] and se['PASS'] and res['completed'] and res['PASS']
 assert bind(F['actual_entry'])['sha256']=='6c9b8f56d27698e8bc85ed551f35492306ee3c046bfdaa61081e1586fa374a1d'
 for a in se['files']:
  if a['path'].endswith('.json') or a['path'].endswith('.jsonl'):assert bind(Path(a['path']))['sha256']==a['sha256']
 assert bind(F['producer'])['sha256']==pr['code']['sha256']
 assert bind(F['frames100'])['sha256']==pr['frames']['sha256'] and bind(F['v90_frames'])['sha256']==pr['records']['sha256']
 props,qs,frames,v90=rows('proposals100'),rows('queries800'),rows('frames100'),rows('v90_frames')
 views,cues=rows('teacher_views604'),rows('teacher_cues302')
 assert len(props)==len(frames)==len(v90)==100 and len(qs)==800 and len(views)==604 and len(cues)==302
 assert len({q['query_id'] for q in qs})==800
 byframe=defaultdict(list)
 for v in views:byframe[v['frame_input_id']].append(v)
 assert set(byframe)=={f['frame_input_id'] for f in frames}
 qresults=[];combos=[];candidate_count=0;argmax_tie_frames=0
 for i,(p,f,z) in enumerate(zip(props,frames,v90)):
  for k,v in f.items():assert p[k]==v
  assert p['frame_index']==i and z['frame_index']==i and p['image_sha256']==z['image_sha256']
  assert p['reusable_full_P3']==z['full_P3'] and p['V90_P3_exact'] is True
  assert p['added_border']==z['added_border']==100 and p['canvas_shape']==z['canvas_shape'] and p['input_shape']==z['actual_input_shape']
  assert p['affine_gain']==z['gain'] and p['affine_offset']==z['offset']
  cand=p['candidates'];candidate_count+=len(cand)
  for j,c in enumerate(cand):
   assert c['candidate_index']==j and math.isfinite(c['score']) and len(c['box_xyxy'])==4 and len(c['keypoints_xy'])==len(c['keypoints_conf'])==9
   assert all(math.isfinite(a) for a in c['box_xyxy']+c['keypoints_conf']) and all(len(a)==2 and all(math.isfinite(t) for t in a) for a in c['keypoints_xy'])
  winner=None if not cand else max(range(len(cand)),key=lambda j:cand[j]['score'])
  assert winner==p['selected_index']
  if winner is not None:
   argmax_tie_frames+=sum(c['score']==cand[winner]['score'] for c in cand)>1
   raw=cand[winner]['keypoints_xy'];L=L_from_raw9(raw,p['raw_hw'])
  else:raw=None;L=None
  ownviews=byframe[f['frame_input_id']]
  assert len(ownviews) in [6,8]
  seed=bool(f['seed_V84_reuse']);assert seed==(len(ownviews)==8)
  for k in range(8):
   q=qs[i*8+k];xy=None if raw is None else raw[k]
   assert q['frame_index']==i and q['frame_input_id']==f['frame_input_id'] and q['pair_id']==f['pair_id'] and q['view']==f['view'] and q['split']==f['split'] and q['native_slot']==k
   assert q['query_id']==f['frame_input_id']+':native'+str(k) and q['query_xy']==xy and q['roi_side']==L
   cv,nv,rv=validity(xy,p['raw_hw'])
   assert q['original_query_continuous_valid']==cv and q['query_nearest_pixel_valid']==nv and q['query_rgb96_centre_domain_valid']==rv
   reason='NO_DETECTION_U' if xy is None else 'VALID_QUERY_NOT_A_VISIBILITY_LABEL' if cv else 'OOB_OR_INVALID_U'
   assert q['reason']==reason and q['native_semantic_state']=='UNKNOWN' and q['human_gold'] is False
   full=[];can=[];defined=0;counts=Counter()
   for v in ownviews:
    assert v['image_sha256']==f['image_sha256'] and v['view']==f['view'] and v['split']==f['split']
    r=v['reference_xy_unchanged'];bb=v['uncertainty_bbox_xyxy_half_open_unchanged'];lab=int(v['teacher_label'])
    assert (r is None)==(bb is None)
    cov=coverage(bb,xy,L);defined+=bb is not None;counts[cov]+=1
    known=lab in [0,1]
    record=dict(query_id=q['query_id'],frame_index=i,frame_input_id=f['frame_input_id'],pair_id=f['pair_id'],view=f['view'],split=f['split'],native_slot=k,original_q=xy,original_L=L,query_status=reason,nativecontinuous_valid=cv,nearest_valid=nv,RGBcentre_valid=rv,
      cue_view_index=v['cue_view_index'],cue_id=v['cue_id'],corpus=v['corpus'],cue_reference_xy_unchanged=r,cue_uncertainty_bbox_unchanged=bb,reference_null=r is None,uncertainty_inside_original_image=v['uncertainty_inside_original_image'],teacher_label=lab,teacher_state={0:'D_CONDITIONAL_AI',1:'E_CONDITIONAL_AI',-1:'U'}[lab],coverage=cov,certified_duplicate_group=v['duplicate_group_link_only'],duplicate_group_size=v['duplicate_group_size'],duplicate_weight=v['duplicate_weight'],native_expected_cue_binding='UNKNOWN',training_admission=False)
    combos.append(record)
    if known and cov=='FULL':
     full.append(record)
     if v['corpus']=='V85_CANONICAL294':can.append(record)
   groups={x['certified_duplicate_group'] for x in full}
   state_set={x['teacher_label'] for x in full};canstates={x['teacher_label'] for x in can}
   rr=dict(query_id=q['query_id'],frame_index=i,frame_input_id=f['frame_input_id'],pair_id=f['pair_id'],view=f['view'],split=f['split'],native_slot=k,original_q=xy,original_L=L,query_status=reason,nativecontinuous_valid=cv,nearest_valid=nv,RGBcentre_valid=rv,missing_sentinel_native_grid=(xy==[-1,-1]),
    canonical_namespace_applicable=not seed,seed_namespace_applicable=seed,source_cue_slots=len(ownviews),defined_reference_slots=defined,coverage_counts=dict(counts),
    canonical_known_FULL_count=len(can),canonical_known_FULL_D_count=sum(x['teacher_label']==0 for x in can),canonical_known_FULL_E_count=sum(x['teacher_label']==1 for x in can),canonical_FULL_mixed_D_E=len(canstates)>1,
    any_known_FULL_raw_count=len(full),any_known_FULL_D_count=sum(x['teacher_label']==0 for x in full),any_known_FULL_E_count=sum(x['teacher_label']==1 for x in full),known_FULL_raw_mixed_D_E=len(state_set)>1,
    known_FULL_certified_group_count=len(groups),known_FULL_original_cue_ids=[x['cue_id'] for x in full],known_FULL_certified_group_ids=sorted(groups),
    expected_native_cue_binding='UNKNOWN',scalar_visibility_label=None,training_admission=False,source_query_row_unchanged=q)
   qresults.append(rr)
 assert len(qresults)==800 and len(combos)==4832
 assert sum(x['nativecontinuous_valid'] for x in qresults)==res['valid_original_slots']==697
 assert sum(x['query_status']=='NO_DETECTION_U' for x in qresults)==40
 assert sum(x['query_status']=='OOB_OR_INVALID_U' for x in qresults)==63
 assert sum(p['selected_index'] is None for p in props)==res['no_detection_frames']==5
 assert res['all100_P3_exact_V90'] and res['repeat_first_prediction_andP3_exact'] and res['after_state_unchanged'] and res['checkpoint_unchanged']
 # Existing certified seed linkage only; no new canonical duplicate merge.
 seedviews=[v for v in views if v['corpus']!='V85_CANONICAL294']
 assert len(seedviews)==16 and len({v['cue_id'] for v in seedviews})==8
 seedgroups=defaultdict(list)
 for v in seedviews:
  if v['view']=='before':seedgroups[v['duplicate_group_link_only']].append(v['cue_id'])
 assert len(seedgroups)==6 and sorted(map(len,seedgroups.values()))==[1,1,1,1,2,2]
 dump_rows(O/'ALL800_QUERY_METADATA_AND_CUE_AVAILABILITY_GT_ONLY.jsonl',qresults)
 dump_rows(O/'ALL4832_QUERY_X_CUE_VIEW_AVAILABILITY_GT_ONLY.jsonl',combos)
 flat=[{k:r[k] for k in ['query_id','frame_index','pair_id','view','split','native_slot','query_status','nativecontinuous_valid','nearest_valid','RGBcentre_valid','canonical_namespace_applicable','canonical_known_FULL_count','canonical_known_FULL_D_count','canonical_known_FULL_E_count','canonical_FULL_mixed_D_E','any_known_FULL_raw_count','any_known_FULL_D_count','any_known_FULL_E_count','known_FULL_raw_mixed_D_E','known_FULL_certified_group_count']} for r in qresults]
 with (O/'ALL800_SUMMARY.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
 covcounts=Counter((x['query_status'],x['teacher_state'],x['coverage'],x['reference_null']) for x in combos)
 summary=dict(schema='V94_ACTUAL_PALLOX_QUERY800_METADATA_AND_CUE_AVAILABILITY_AUDIT_KO',completed=True,PASS_metadata_replay=True,
  counts=dict(frames=100,queries=800,candidate_count=candidate_count,argmax_tie_frames=argmax_tie_frames,cue_views=604,all_query_cue_combinations=4832,**aggregate(qresults)),
  by_split={s:aggregate([r for r in qresults if r['split']==s]) for s in sorted({r['split'] for r in qresults})},
  by_view={s:aggregate([r for r in qresults if r['view']==s]) for s in ['before','after']},
  combo_counts=[dict(query_status=k[0],teacher_state=k[1],coverage=k[2],reference_null=k[3],N=n) for k,n in sorted(covcounts.items())],
  per_split_combo_counts={s:[dict(query_status=k[0],teacher_state=k[1],coverage=k[2],reference_null=k[3],N=n) for k,n in sorted(Counter((x['query_status'],x['teacher_state'],x['coverage'],x['reference_null']) for x in combos if x['split']==s).items())] for s in sorted({x['split'] for x in combos})},
  producer_certificate_only=dict(all100_V90_fullP3_bindings_and_affine_metadata_exact=True,pixels_or_P3_array_values_independently_replayed=False,model_state_or_checkpoint_parameters_read=False,actual_forward_by_auditor=0,source_RGB_and_repeat_state_exactness='Producer-bound RESULT/entry/code certificates; not new independent numeric replay.'),
  assertions=dict(all100_frames_and800slots_original_order=True,first_argmax_and_all_candidate_metadata_replayed=True,centroid8_excluded_from_L_and_query=True,raw_q_L_all800_exact=True,all_nativecontinuous_nearest_RGBcentre_flags_recomputed=True,no_nearest_or_GTbestcue_selection=True,seed8_original_boxes_not_merged_or_averaged=True,native_mapping_unknown=True,old604_labels_unchanged=True),
  seed_certified_link_groups=dict(seedgroups),
  interpretation_KO='이 결과는 SAME actual initializer q/L의 cue 가용성 진단이다. FULL known cue가0이면 단서 미포함/불확실성과 native absence를 혼동하지 않는다. 1개여도 expected native slot binding이나 pixel/material label이 자동 생성되지 않으며, >1 또는 mixedD/E는 scalar task identity의 명시적 모호성이다. 별도 AUX namespace에서 확인된 physical cue 지원을 정의할 수 있지만 native/CAD 매핑을 모든 auxiliary fit의 필수조건으로 추가하지 않았다. 현 natural source는 one tall wooden bin이며 flat pallet의 모든 native corner 의미를 대체하지 않는다.',
  scope_limit='No source pixels/masks/depth/NPZ tensor bodies, learned models/weights, current223 artifacts, newforward/fit/PnP. All original source and producer files immutable. All scalar_visibility_label values remain null and training_admission false.',
  bindings={k:bind(p) for k,p in F.items()},new_pixels_arrays_model_current_forward_fit_PnP=0,training_admission=False,human_native_gold_added=0,goal_status='active',fully_solved=False)
 write(O/'REVIEW_KO.json',summary)
 write(O/'SEAL.json',dict(completed=True,PASS=True,scope=bind(O/'SCOPE_AND_ARITHMETIC_BEFORE_ALLROW_REPLAY.json'),code=bind(Path(__file__)),outputs=[bind(O/x) for x in ['GENERATED_METADATA_BOUNDARY_FIXTURES.json','ALL800_QUERY_METADATA_AND_CUE_AVAILABILITY_GT_ONLY.jsonl','ALL4832_QUERY_X_CUE_VIEW_AVAILABILITY_GT_ONLY.jsonl','ALL800_SUMMARY.csv','REVIEW_KO.json']],actual_producer_seal=bind(F['intake_seal']),new_pixels_arrays_models_current_forward_fit_PnP=0,original_modified=0,training_admission=False))
 print(json.dumps(dict(report=bind(O/'REVIEW_KO.json'),seal=bind(O/'SEAL.json'),counts=summary['counts'],by_split=summary['by_split']),ensure_ascii=False))

if __name__=='__main__':main()
