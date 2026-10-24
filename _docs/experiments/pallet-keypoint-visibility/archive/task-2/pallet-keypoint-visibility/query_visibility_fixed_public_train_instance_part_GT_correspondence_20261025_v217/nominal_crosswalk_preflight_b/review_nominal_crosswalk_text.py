"""V217 shared-three prior-text audit only; never dereference any GT/data binding."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import stat

H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
SCOPE=V/'ROOT_PROSPECTIVE_FIXED12_PUBLIC_GT_CORRESPONDENCE_AUDIT_SCOPE.json'
READINESS=P/'query_visibility_public_train_part_query_support_contract_20261025_v216/public_GT_text_readiness_a/REVIEW.json'
DEFINITION=P/'query_visibility_native_part_support_feasibility_20261024_v159/native_definition_audit/DEFINITION_AND_FEASIBILITY.json'
SEAL=P/'query_visibility_mr6d_source_supervision_bridge_20261024_v154/ROOT_SUPERVISION_SEAL.json'
INTAKE=P/'query_visibility_mr6d_full_public_source_20261024_v153/full_intake_preparation/intake_full_source.py'
INPUTS={SCOPE:'b396012313065b36781a927dc51222c8d6acf64d0fd19186f8a80d3e6ea1a4c4',READINESS:'66e94d2679e3323e3acfab84f6a8668a764d46506801ae6e43794e2bf953ff97',DEFINITION:'76c0a83888f3c93445a8c216ceb4c6af069a939af97451fefab34bdae38d4ddc',SEAL:'d6915b0abcfb8c7058bf4cc2d83b73492f57edd88e4862e55684712a310c4b34',INTAKE:'f2f564e30ef60428a13952d5d5bb8a907681d8b179546dd4dc7f73e24aa87920'}

def opened(path,write=False):
 p=Path(path);assert p.is_absolute()and'..'not in p.parts and'v112'not in str(p).casefold()
 assert p.parent==H if write else p in INPUTS or p==Path(__file__)
 fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for part in p.parts[1:-1]:
   n=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
  n=os.open(p.name,os.O_NOFOLLOW|(os.O_WRONLY|os.O_CREAT|os.O_EXCL if write else os.O_RDONLY),0o644,dir_fd=fd)
 finally:os.close(fd)
 assert stat.S_ISREG(os.fstat(n).st_mode)
 return os.fdopen(n,'wb'if write else'rb')

def read(path):
 with opened(path)as f:data=f.read()
 if path in INPUTS:assert hashlib.sha256(data).hexdigest()==INPUTS[path]
 return data

def bind(path):return dict(path=str(path),sha256=hashlib.sha256(read(path)).hexdigest())
def new(name,value):
 data=value.encode()if isinstance(value,str)else(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
 with opened(H/name,True)as f:f.write(data)

def top_selected(data,allowed):
 """Read member names and skip nonselected values lexically, not deserialize held/count sections."""
 text=data.decode();decoder=json.JSONDecoder();i=1;result={}
 while True:
  while text[i].isspace()or text[i]==',':i+=1
  if text[i]=='}':return result
  key,end=decoder.raw_decode(text,i);i=end
  while text[i].isspace():i+=1
  assert text[i]==':';i+=1
  while text[i].isspace():i+=1
  start=i;depth=0;quoted=False;escaped=False
  while i<len(text):
   c=text[i]
   if quoted:
    if escaped:escaped=False
    elif c=='\\':escaped=True
    elif c=='"':quoted=False
   elif c=='"':quoted=True
   elif c in '[{':depth+=1
   elif c in ']}':
    if depth==0:break
    depth-=1
   elif c==','and depth==0:break
   i+=1
  if key in allowed:result[key]=json.loads(text[start:i])

def main():
 scope=json.loads(read(SCOPE));assert scope['actual_GT_or_camera_or_mask_or_depth_or_CAD_or_NPZ_or_RGB_stat_hash_read_decode']is False
 definition=top_selected(read(DEFINITION),{'exact_coordinate_and_order_contract','authority_separation','native_expected_part_table','CAD_geometry_recalculation','required_native_part_proof','pose_and_dimension_uncertainty','masks_and_depth','concrete_preparation_completed'})
 seal=top_selected(read(SEAL),{'code','root_policy','source_only','all_original_queries_retained','original_q_L_flags_source_ids_and_splits_unchanged','human_gold','exact_real_point_gold','wrong_entity_never_E','real_point_quality_labels_created','no_GT_nearest_permutation_or_coordinate_correction'})
 readiness=top_selected(read(READINESS),{'native_crosswalk_and_observability_limits','prior_GT_use_not_first_use','datasets','evidence'})
 source=read(INTAKE).decode();contract=definition['exact_coordinate_and_order_contract']
 assert contract['permutation_direction']=='dynamic[k] = fixed[perm[k]]; never reapply to already dynamic stored arrays.'
 assert contract['projection'].endswith('points @ R.T + t.')and contract['native_names']==['front_top_left','front_top_right','front_bottom_right','front_bottom_left','rear_top_left','rear_top_right','rear_bottom_right','rear_bottom_left']
 assert "gt_instance_index=ix"in source and "gi[r['frame_key']][ix]"in source and "cam[r['frame_key']]['depth_scale']"in source and "semantic_native_labels='UNASSIGNED'"in source
 assert seal['wrong_entity_never_E']is True and seal['human_gold']is False and seal['exact_real_point_gold']is False and seal['no_GT_nearest_permutation_or_coordinate_correction']is True
 # Invented categorical fixtures illustrate direction/identity, not SOURCE geometry or new labels.
 fixed=list(range(8));perm=[3,0,7,4,1,6,2,5];dynamic=[fixed[p]for p in perm];inverse=[perm.index(k)for k in range(8)]
 assert dynamic==perm and dynamic!=inverse and[dynamic[p]for p in perm]!=dynamic
 mask_for_index={2:'INVENTED_FRAME_000002.png',7:'INVENTED_FRAME_000007.png'}
 choose=lambda selected:None if selected is None else mask_for_index[selected]
 assert choose(None)is None and choose(7)=='INVENTED_FRAME_000007.png'
 q=(100.,200.);reference=(120.,180.);L=192.;crop=[96*(r-x)/L+47.5 for r,x in zip(reference,q)]
 assert crop==[57.5,37.5]and q==(100.,200.)
 checks=['Saved_forward_permutation_direction_not_inverse_or_second_application','Intake_GTlist_index_mask_info_camera_identity_and_UNASSIGNED_semantics','Wrong_entity_never_E_and_no_nearest_correction_in_prior_seal','Invented_same_obj_class_different_GTlist_masks_no_first_fallback','Invented_original_q_unchanged_reference_crop_diagnostic_only']
 new('PURE_AND_TEXT_CHECKS.json',dict(completed=True,PASS=True,count=len(checks),checks=checks,invented_geometry_only=True,GT_data_or_projection_execution=0))
 table=[]
 for k,name in enumerate(contract['native_names']):
  table.append(dict(native_slot=k,native_name=name,canonical_index='perm['+str(k)+']',permutation_direction='dynamic[k]=fixed[perm[k]] once',nominal_anatomy='+Z deck neighborhood'if k in(0,1,4,5)else'-Z bottom-runner neighborhood',selected_instance='UNKNOWN unless explicitly associated; never obj_id or first-GT fallback',part_proof='Expected part dictionary is a prior nominal proposal; full/visible mask alone is not part identity',visibility='rear/front name and q mask/owner do not determine D/E/S',original_query='Unchanged q/L/U; crop reference offset is diagnostic, not coordinate error gold'))
 out=io.StringIO(newline='');writer=csv.DictWriter(out,fieldnames=list(table[0]),lineterminator='\n');writer.writeheader();writer.writerows(table);new('NATIVE8_NOMINAL_CROSSWALK_CONTRACT.csv',out.getvalue())
 future=[dict(id='immutable_row_and_instance_key',check='Join original source_id/frame/native/role to (official split,scene,frame_key,GT-list index,obj_id), retaining all same-class candidate instances. Selected index may remain null; no GT-nearest/first/silent fallback.'),dict(id='one_nominal_role_mapping',check='When separately authorized, derive/consume the declared nominal perm once and record dynamic native k -> canonical perm[k], +Z deck, front face/LR margin and ambiguity. Do not permute original deployed q8 or search a permutation.'),dict(id='part_vs_point',check='Keep nominal projected AABB corner, expected anatomical part neighborhood, actual visible local witness and original q as separate columns. The saved eight-role dictionary proposes a part; it does not certify each selected query.'),dict(id='mask_depth_support_axes',check='Match full/visible masks by GT-list index and frame key; retain nonexclusive overlaps, missing support, measured-depth validity, camera/depth scale and registration flags. Mask absence or a foreign object at q alone is not native E.'),dict(id='same_part_positive_foreground',check='Only positive same-intended-instance/part and foreground evidence can support a bounded future occlusion candidate. Wrong object/query OFF is correspondence/location evidence; retain UNKNOWN for native state, not E.'),dict(id='bounded_use',check='Produce all96 fixed rows/U11 with original SOURCE_FIT/DEV roles; future DEV descendants remain excluded from all fit branches. SOURCE nominal evidence can be useful without a perfect-mesh/corner certificate; ambiguous correspondence remains explicit UNKNOWN. No labels/admission or training here.')]
 prior_paths=[definition['native_expected_part_table'],definition['CAD_geometry_recalculation']]
 for rec in readiness['evidence']:
  if rec['path'].endswith('/native_mask_support_audit/COMPLETE.json')or rec['path'].endswith('/owner_supervision_bridge/COMPLETE.json'):prior_paths.append(dict(path=rec['path'],sha256=rec['sha256_observed']))
 review=dict(completed=True,PASS=True,status='PASS_CODE_TEXT_CONTRACT_WITH_UNRESOLVED_INSTANCE_PART_MAPPING',code=bind(Path(__file__)),scope=bind(SCOPE),prior_texts_exactly_three=[bind(x)for x in(DEFINITION,SEAL,INTAKE)],shared_subset_of_A16=True,declared_V216_readiness=bind(READINESS),coordinate_and_order_contract=contract,prior_bridge_invariants=seal,concrete_columns_and_functions=[dict(file=str(INTAKE),function='prepare()',lines='35-54',columns=['source_id','official_split','scene','frame_key','obj1_gt_instance_indices','source_files.full_masks[].gt_instance_index','source_files.visible_masks[].gt_instance_index']),dict(file=str(INTAKE),function='factual()',lines='168-182',columns=['gt[frame_key][ix].obj_id','gi[frame_key][ix]','instances[].gt_instance_index','instances[].semantic_native_labels=UNASSIGNED','depth_scale_to_mm','instance_pair_overlaps']),dict(file=str(DEFINITION),key='exact_coordinate_and_order_contract',authority='task nominal eight camera-dynamic AABB roles, not official MR6D native8 semantic labels')],prior_part_proposal_and_mask_diagnostic_pointers_not_opened=prior_paths,actionable_next_checks=future,unresolved=['Original deployed detection→GT-list-index association is not created by V153 intake; same obj_id is not same instance.','Shared three texts do not establish per-query intended-part correspondence for the fixed12/96; prior nominal dictionary and mask diagnostics are useful starting points, not current labels.','Source pose/dimension uncertainty and mask nonexclusivity prevent exact physical corner/visibility certification but do not prohibit bounded weak same-part review.'],A_final_schema_reported_not_independently_reopened=dict(ready={'path':str(V/'fixed12_GT_payload_preparation_a/FINAL_READY.json'),'sha256':'b97dc7de1d15e8e251bbb5dbd262611d69b7ab711a24c4a2b0f26f559af243e4'},selected_gt_instance_index=None,selected_instance_correspondence='UNKNOWN_NO_GT_BEST_OR_FIRST_INSTANCE_FALLBACK',source='A sealed schema-only message'),SOURCE_projection_policy='Only future separately approved SOURCE pose/camera nominal target generation; no projection executed here, no GT recenter/PnP/permutation search or deployment GT feature input.',raw_candidates_are_training_labels=False,training_admission=False,actual_GT_camera_mask_depth_CAD_NPZ_RGB_filesystem_stat_hash_read_decode=0,protected_model_scores_or_HOLD_payload_reads=0,new_observations_or_labels=0,fit_or_forward=0,goal_solved=False,frozen_no_further_edits=True)
 new('REVIEW.json',review)
 ko='''# V217 native·instance·부위 조인 코드/텍스트 검토

선언된 순서는 `dynamic[k] = fixed[perm[k]]`이며 한 번만 적용한다. 이미 dynamic인 q8/저장 좌표를 다시 재배열하거나 GT와 가까워지도록 permutation을 고르지 않는다. native0/1/4/5는 nominal +Z deck 쪽, 2/3/6/7은 −Z bottom-runner 쪽 역할이지만, 이는 실제 crop에서 그 부위가 보인다는 라벨이 아니다. rear라는 이름도 SELF를 뜻하지 않는다.

V153 `prepare()` 35–54행은 scene/frame의 모든 obj1 GT-list index와 각 mask 경로를 보존한다. `factual()` 168–182행은 `gt[frame_key][ix]`, info, camera/depth scale 및 같은 `ix`의 full/visible mask를 조인하고 native 의미를 `UNASSIGNED`로 남긴다. 검출된 대상이 어느 GT instance인지 선택하는 코드는 아니다. 같은 obj1의 다른 팔레트도 다른 instance이며, mask 파일의 GTID는 object class ID가 아니다.

A의 봉인 schema는 selected index null과 `UNKNOWN_NO_GT_BEST_OR_FIRST_INSTANCE_FALLBACK`을 유지한다고 전달됐다. 본 검토는 그 새 산출물을 재개방하지 않았다. 기존 3개 공유 텍스트만으로 고정12/96의 per-query intended-part 대응이 확정되지는 않는다.

다음 실제 점검에서는 원 row/source/native/q/L/U·FIT/DEV를 그대로 두고, (1) 명시적 instance 대응, (2) 한 번 적용한 nominal native→canonical과 expected part, (3) 원 q와 nominal reference의 crop 내 관계, (4) full/visible mask·depth support·겹침을 별도 열로 기록하면 된다. 전역 silhouette의 양성은 어느 target 표면일 뿐 intended part 증명이 아니며, 음성·depth hole·query OFF·다른 물체는 자동 E가 아니다. 같은 intended part의 양성 foreground 근거가 없는 경우 native 상태를 UNKNOWN으로 남긴다. 그렇다고 완벽한 mesh/코너 인증을 모든 약감독의 blanket gate로 요구하지 않는다.

보존된 nominal part 제안과 기존 mask 진단은 재사용할 수 있다. 아직 열지 않은 정확 포인터를 JSON에 기록했다. 이번 단계에서는 GT/카메라/마스크/depth/CAD/이미지/배열 파일의 stat·hash·read·decode, 실제 투영, 모델/점수·보호셋 접근, 새 라벨·admission·학습이 모두 0이다. 원 pose projection은 별도 actual entry 이후 불확실한 SOURCE nominal target 근거로 사용할 수 있으며, 배포 입력이나 q 교정으로 쓰지 않는다.

- [검토 JSON](REVIEW.json) · [8개 native 조인 표](NATIVE8_NOMINAL_CROSSWALK_CONTRACT.csv) · [가상/텍스트 검사](PURE_AND_TEXT_CHECKS.json)
- [현재 scope](../ROOT_PROSPECTIVE_FIXED12_PUBLIC_GT_CORRESPONDENCE_AUDIT_SCOPE.json)
- [V159 순서·부위 정의](../../query_visibility_native_part_support_feasibility_20261024_v159/native_definition_audit/DEFINITION_AND_FEASIBILITY.json)
- [V154 원 query/instance 약감독 경계](../../query_visibility_mr6d_source_supervision_bridge_20261024_v154/ROOT_SUPERVISION_SEAL.json)
- [V153 intake 코드](../../query_visibility_mr6d_full_public_source_20261024_v153/full_intake_preparation/intake_full_source.py)
'''
 new('KO.md',ko)
 print(json.dumps(dict(completed=True,PASS=True,prior_texts=3,checks=len(checks),review_sha256=hashlib.sha256((json.dumps(review,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()).hexdigest(),actual_payload_access=0),ensure_ascii=False))

if __name__=='__main__':main()
