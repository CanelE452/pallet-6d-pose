"""Read-only V193 static review plus independent invented-only pure fixtures."""
from pathlib import Path
import ast,copy,hashlib,json,math,os,stat
import numpy as np
H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
C=V/'real_end_opposite_sham_preparation_c'
CODE=C/'build_real_end_opposite_sham_pairs.py'
OLD=P/'query_visibility_sampled_domain_direct_end_20261024_v189/real_end_pair_preparation_b/build_real_end_local_pairs.py'
G=P/'query_visibility_sampled_domain_direct_end_20261024_v189/end_locality_contract_b/finite_end_locality_contract.py'

def need(x,m):
 if not bool(x):raise AssertionError(m)
def opened(path):
 p=Path(path);need(p.is_absolute() and '..' not in p.parts and P in p.parents and not any('v112' in z.casefold() for z in p.parts),'Exact code/meta path before access')
 fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for part in p.parts[1:-1]:
   nx=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nx
  end=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 finally:os.close(fd)
 need(stat.S_ISREG(os.fstat(end).st_mode),'Ordinary code/text file');return os.fdopen(end,'rb')
def data(p):
 with opened(p)as f:return f.read()
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(data(p)).hexdigest())
def read(p):return json.loads(data(p))
def fns(p):return {n.name:n for n in ast.parse(data(p).decode()).body if isinstance(n,ast.FunctionDef)}

ready=read(C/'FINAL_READY.json');proto=read(C/'PROTOCOL_BEFORE_REAL_END_PAIR_SOURCE_PIXELS.json');checks=read(C/'GENERATED_INVENTED_REAL_END_PAIR_CHECKS.json')
need(ready['code']==bind(CODE) and ready['protocol']==bind(C/'PROTOCOL_BEFORE_REAL_END_PAIR_SOURCE_PIXELS.json') and ready['checks']==bind(C/'GENERATED_INVENTED_REAL_END_PAIR_CHECKS.json'),'Exact final closure')
need(checks['PASS'] and checks['count']==29 and all(c['PASS'] for c in checks['checks']),'Recorded invented29 complete')
old,new=fns(OLD),fns(CODE);same=['safe_path','opened','sha','ash','exact','checked_bytes','json_file','checked_json','write_new','load_functions','materialize','original_record','endpoint_preflight','fixed_transform_plan','fixed_placements','selected_sample_mask','prove','save_clean_png','contact']
for name in same:need(ast.dump(old[name],include_attributes=False)==ast.dump(new[name],include_attributes=False),'Frozen inherited source kernel '+name)
ns={'np':np,'copy':copy,'math':math};gtree=ast.parse(data(G).decode());pure=[n for n in gtree.body if isinstance(n,(ast.Assign,ast.FunctionDef))]
exec(compile(ast.Module(body=pure,type_ignores=[]),'frozen_end_contract::pure','exec'),ns)
geo=type('PureGeometry',(),{k:staticmethod(v) if callable(v) else v for k,v in ns.items() if k not in ['np','copy','math','__builtins__']})()
local={'np':np,'copy':copy};pick=['require','select_original_or_one_opposite_sham','contract14_proof','verify_old_supported_decoded_packets','exact']
exec(compile(ast.Module(body=[new[n] for n in pick],type_ignores=[]),'reviewed_V193::pure','exec'),local)
tests={};o=dict(development_role='SOURCE_DEVELOPMENT',native_slot=0,query_xy=[48.,48.],L=48.,raw_hw=[96,96],original_query_available=True)
endpoint=dict(authority=geo.REAL_AUTHORITY,native_slot=0,end_locality_id='INVENTED',evidence_binding={'invented':True},candidate_enclosures_xyxy=[[45.,45.,48.,48.]],surrounding_material_witnesses=[dict(raw_pixel_xy=[42,46],material_observed=True,same_selected_part_supported=True,evidence_binding={'invented':True})],**{k:True for k in ('selected_instance_supported','intended_part_supported','queried_end_localization_supported','all_plausible_candidates_enclosed','before_end_visible_supported')})
placement=dict(cover_bbox_xyxy=[44,44,50,50],cover_origin_xy=[44,44],sham_origin_xy=[54,44],sham_translation_xy=[10,0],horizontal_direction=1,transformed_hw=[6,6],opaque_anchor_xy=[1,1])
class CountGeo:
 def __init__(self):self.calls=0
 def evaluate_plan(self,*args):self.calls+=1;return geo.evaluate_plan(*args)
def call(e=endpoint,orig=o):
 cg=CountGeo();before=copy.deepcopy((orig,e,placement));a=local['select_original_or_one_opposite_sham'](orig,e,placement,cg);need((orig,e,placement)==before,'No original q/L/enclosure/witness/placement mutation');return a,cg.calls
(pp,plan,policy),n=call();tests['original_geometry_PASS_keeps_old_SHAM_one_plan']=n==1 and pp==placement and policy['opposite_attempted'] is False and plan['status']=='GEOMETRY_READY_PIXEL_PROOFS_PENDING'
e=copy.deepcopy(endpoint);e['surrounding_material_witnesses'][0]['raw_pixel_xy']=[56,46]
(pp,plan,policy),n=call(e);tests['old_failure_exactly_one_dx_reflection_same_COVER_shape']=n==2 and pp['sham_translation_xy']==[-10,0] and pp['cover_bbox_xyxy']==placement['cover_bbox_xyxy'] and pp['transformed_hw']==placement['transformed_hw'] and plan['status']=='GEOMETRY_READY_PIXEL_PROOFS_PENDING'
e=copy.deepcopy(endpoint);e['surrounding_material_witnesses'][0]['raw_pixel_xy']=[46,46]
(pp,plan,policy),n=call(e);tests['both_sides_failure_no_third_search_or_witness_replacement']=n==2 and plan['status']=='GEOMETRY_FAILURE_RETAINED' and policy['selection']=='OPPOSITE_ONE_ATTEMPT_FAILED_RETAINED'
e=copy.deepcopy(endpoint);e['queried_end_localization_supported']=False
(pp,plan,policy),n=call(e);tests['semantic_UNKNOWN_no_second_proposal_no_promotion']=n==1 and plan['status']=='UNKNOWN' and policy['opposite_attempted'] is False
(pp,plan,policy),n=call(orig=dict(o,original_query_available=False));tests['factual_input_U_no_second_proposal']=n==1 and plan['status']=='UNKNOWN' and plan['new_native_visibility_target']==-1
# Independent mapping test: actual evidence booleans, never source asset arrays.
keys=['all8_input_q_L_flags_validity_and_ID_unchanged','all8_original_initializer_flags_unchanged','all8_new_native_quality_owner_unknown','all8_central32_exact_RGB96_slice','end_every_candidate_raw_tap_COVER_alpha1','end_every_candidate_raw_tap_SHAM_alpha0','equal_positive_pasted_alpha_mass','outside_paste_fullRGB_exact','same_transformed_RGB_alpha_pasted_at_both_locations','end_support_actual_original96_bytes_changed_COVER','end_support_actual_original96_bytes_unchanged_SHAM','SHAM_donor_positive_original96_RGB_support','chosen_surrounding_samepart_RGB_unchanged_in_both']
q={k:True for k in keys};before={'rgb96':np.zeros((8,2,2,3),np.uint8)};cover={'rgb96':before['rgb96'].copy()};cover['rgb96'][0,0,0,0]=1
proof=local['contract14_proof'](q,{'all8_before':True},before,cover,{'PASS':True,'split':'TRAIN','donor_id':'YCB:INVENTED'},{'invented':True})
plan=geo.evaluate_plan(o,endpoint,placement);closed=geo.close_actual_proof(plan,proof)
names=[k for k in proof if k!='evidence_bindings'];tests['all14_mapped_proofs_required_and_separate_weak_authority']=len(names)==14 and closed['status']=='CONTROLLED_DIRECT_END_WEAK_PAIR_SUPPORTED' and closed['new_native_visibility_target']==closed['new_quality_target']==-1 and closed['training_admission'] is False
for key in names:
 bad=copy.deepcopy(proof);bad[key]=False;need(geo.close_actual_proof(plan,bad)['status']!='CONTROLLED_DIRECT_END_WEAK_PAIR_SUPPORTED','Each mapped proof necessary '+key)
tests['each14_proof_false_retains_UNKNOWN']=True
bad=local['contract14_proof'](q,{'all8_before':False},before,cover,{'PASS':True,'split':'TRAIN','donor_id':'YCB:INVENTED'},{'invented':True});tests['before_original_packet_parity_is_required']=bad['before_original_RGB96_valid96_target_context_exact'] is False
# Old5 comparison path is exercised with in-memory invented packet loader only.
arr=[dict(full_RGB=np.full((4,4,3),i,np.uint8),rgb96=np.full((8,2,2,3),i,np.uint8),native_slot=np.arange(8,dtype=np.int8),query_available=np.ones(8,bool)) for i in range(3)]
ca=np.zeros((4,4),bool);ca[0,0]=True;sa=np.zeros((4,4),bool);sa[0,1]=True
names3=('BEFORE','COVER','ADJACENT_SHAM');masks=(np.zeros((4,4),bool),ca,sa);saved={name:{**{k:v.copy() for k,v in a.items()},'pasted_alpha':mask.copy()} for name,a,mask in zip(names3,arr,masks)}
local['VARIANTS']=names3;local['load_bound']=lambda binding,keys:{k:saved[binding['invented']][k].copy() for k in keys}
oldrec=dict(PASS=True,variants={name:{'invented':name} for name in names3});par=local['verify_old_supported_decoded_packets'](oldrec,arr,ca,sa)
tests['old_supported_all3_every_emitted_array_dtype_shape_bytes_exact']=all(all(v.values()) for v in par.values())
saved['ADJACENT_SHAM']['rgb96'][0,0,0,0]^=1
try:local['verify_old_supported_decoded_packets'](oldrec,arr,ca,sa)
except AssertionError:tests['old_supported_single_byte_drift_fails_closed']=True
need(all(tests.values()),'Independent invented static preflight')
result=dict(completed=True,PASS=True,review_type='READ_ONLY_STATIC_CODE_AND_INVENTED_PURE_FIXTURES_NOT_ACTUAL_SOURCE_AUDIT',code=bind(CODE),protocol=bind(C/'PROTOCOL_BEFORE_REAL_END_PAIR_SOURCE_PIXELS.json'),producer_final_ready=bind(C/'FINAL_READY.json'),root_scope=bind(V/'ROOT_PROSPECTIVE_SOURCE_OPPOSITE_SHAM_POLICY_BEFORE_ASSET_READS.json'),producer_recorded_29_fixture_receipt=bind(C/'GENERATED_INVENTED_REAL_END_PAIR_CHECKS.json'),inherited_kernel_AST_identity=same,independent_invented_tests=tests,independent_invented_test_count=len(tests),review_script=bind(Path(__file__)),blocking_issues=[],static_findings=['Original geometry READY retains original SHAM; only geometry failure tries one sign-flipped dx, unchanged dy0/COVER/EDT scale/enclosures/witness/qL/native/donor.','Semantic/input/local-opportunity UNKNOWN is returned before source asset reads; all66/all8 and originalU retained.','Frozen positive F64 four-tap candidate/witness/local96 support and original materialization/paste/prove kernels unchanged by AST.','All14 closed weak proofs and selected-end actual96 byte change/SHAM equality are required; no bbox-only labels.','Old5 original placements and all3 freshly emitted decoded packet arrays require dtype/shape/byte parity; mismatch fails entire run.','Root explicit binary whitelist is checked before same-descriptor nofollow hash/load. No new native/quality/owner targets or training admission.'],limits=['No source JSONL population hash/read, source asset hash/header/body, RGB/alpha or actual generation was performed by this review.','The policy and invented tests do not predict how many new pairs will pass actual alpha/RGB proofs.','Weak source 2D composite authority does not certify natural occlusion, physical native gold, independent holdout or classifier improvement.'],source_asset_JSONL_header_pixel_model_feature_weight_eval_or_fit_reads=0,existing_frozen_files_modified=False)
with (H/'REVIEW.json').open('x')as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'completed':True,'PASS':True,'review':bind(H/'REVIEW.json'),'independent_invented_tests':len(tests),'inherited_AST_kernels':len(same)}))
