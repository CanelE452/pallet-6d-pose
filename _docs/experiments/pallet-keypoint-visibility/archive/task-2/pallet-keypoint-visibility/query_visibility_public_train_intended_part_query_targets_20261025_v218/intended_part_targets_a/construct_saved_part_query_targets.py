#!/usr/bin/env python3
"""Saved-text-only actual target geometry construction, without image/model access."""
import collections,copy,csv,hashlib,io,json,math,os,stat,struct
P='/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility'
D=P+'/query_visibility_public_train_intended_part_query_targets_20261025_v218'
H=D+'/intended_part_targets_a'
SCOPE={'path':D+'/ROOT_PROSPECTIVE_INTENDED_PART_QUERY_TARGET_CONSTRUCTION_SCOPE.json','sha256':'f686c279a54fe75dcac1c54b5b437c9a6a86345e5e5e823b15909802bc0b0e68'}
def need(v,msg):
 if not v:raise ValueError(msg)
def opened(p,flags=os.O_RDONLY):
 need(isinstance(p,str) and 'v112' not in p.lower(),'Lexical forbidden before FS')
 need(os.path.isabs(p) and os.path.normpath(p)==p,'Normalized absolute path')
 fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
 try:
  for c in p.strip('/').split('/')[:-1]:
   n=os.open(c,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
  z=os.open(p.rsplit('/',1)[1],flags|os.O_NOFOLLOW,0o644,dir_fd=fd)
  if not stat.S_ISREG(os.fstat(z).st_mode):os.close(z);raise ValueError('Regular only')
  return z
 finally:os.close(fd)
def rb(p):
 with os.fdopen(opened(p),'rb') as f:return f.read()
def sha(z):return hashlib.sha256(z).hexdigest()
def bind(p,z):return {'path':p,'sha256':sha(z),'size_bytes':len(z)}
def checked(b):
 need(isinstance(b['sha256'],str) and len(b['sha256'])==64,'Exact metadata SHA before FS')
 z=rb(b['path']);need(sha(z)==b['sha256'],'Exact metadata binding');return z
def enc(x):return (json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def new(name,z):
 p=H+'/'+name
 with os.fdopen(opened(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL),'wb') as f:f.write(z)
 return bind(p,z)
def newj(name,x):return new(name,enc(x))
def newjl(name,rows):return new(name,b''.join((json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n').encode() for r in rows))
def csvbytes(rows):
 f=io.StringIO(newline='');w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows);return f.getvalue().encode()
def xy(p):return isinstance(p,list) and len(p)==2 and all(type(x) in (int,float) and math.isfinite(x) for x in p)
def fp32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def query_condition(group8,k):
 need(len(group8)==8 and [r['native_slot'] for r in group8]==list(range(8)),'Coherent original native8 order')
 need(len({(r['position'],r['frame_key'],r['component_id'],r['internal_source_role']) for r in group8})==1,'Same original frame/component/role')
 r=group8[k];mask=[False]*8;delta=[0.]*16
 if r['original_query_available']:
  need(xy(r['query_xy']) and type(r['L']) in (int,float) and math.isfinite(r['L']) and r['L']>0,'Available active original qL finite')
  for j,o in enumerate(group8):
   need(type(o['original_query_available']) is bool,'Availability mandatory')
   if j==k or not o['original_query_available']:continue
   need(xy(o['query_xy']),'Available other original q finite')
   mask[j]=True
   for a in range(2):delta[j*2+a]=fp32(max(-8.,min(8.,(float(o['query_xy'][a])-float(r['query_xy'][a]))/float(r['L']))))
 return {'native_onehot8_FP32':[float(j==k) for j in range(8)],'other_mask8':mask,'other_dxdy16_FP32':delta,'self_mask_and_xy_zero':True,'active_U_all_other_context_zero':not r['original_query_available'],'feature_sources':'ONLY_STORED_ORIGINAL_q8_native_L_AVAILABILITY','GT_ROI_part_visibility_role_features':False}
def roi_valid(r):return isinstance(r,list) and len(r)==4 and all(type(v) is int for v in r) and 0<=r[0]<=r[2]<=95 and 0<=r[1]<=r[3]<=95
def finite_support(a,c):
 need(roi_valid(a) and roi_valid(c),'Both unchanged integer inclusive original ROIs')
 runs=[]
 for y in range(96):
  intervals=sorted((r[0],r[2]) for r in (a,c) if r[1]<=y<=r[3]);merged=[]
  for lo,hi in intervals:
   if merged and lo<=merged[-1][1]+1:merged[-1][1]=max(merged[-1][1],hi)
   else:merged.append([lo,hi])
  runs.extend([[y,lo,hi] for lo,hi in merged])
 bbox=[min(a[0],c[0]),min(a[1],c[1]),max(a[2],c[2]),max(a[3],c[3])]
 return {'shape':[96,96],'coordinate_convention':'integer inclusive original RAW96 indices','positive_runs_y_x0_x1':runs,'positive_pixels':sum(hi-lo+1 for y,lo,hi in runs),'union_bbox':bbox,'target_values':{'inside_union':1,'outside_union':'IGNORE_NOT_BACKGROUND_NEGATIVE'},'meaning':'Positive weak finite witness support, not exact material-pixel segmentation or physical point','original_ROIs_preserved':[a,c]}
def intersects(a,b):return max(a[0],b[0])<=min(a[2],b[2]) and max(a[1],b[1])<=min(a[3],b[3])
def contains_qcenter(r):return r[0]-.5<=47.5<r[2]+.5 and r[1]-.5<=47.5<r[3]+.5
def pair_geometry(a,c):
 s=finite_support(a,c);cover=s['union_bbox'];w=cover[2]-cover[0]+1;h=cover[3]-cover[1]+1
 positions=[]
 for y in range(97-h):
  for x in range(97-w):
   b=[x,y,x+w-1,y+h-1]
   if not intersects(b,a) and not intersects(b,c):positions.append((abs(x-cover[0])+abs(y-cover[1]),y,x,b))
 sham=min(positions)[3] if positions else None
 return {'status':'BBOX_GEOMETRY_ONLY_AWAIT_VALIDITY_RGB_PROOFS' if sham else 'NO_GEOMETRIC_SHAM_KEEP_FACTUAL_NO_FORCING','COVER_bbox_inclusive':cover,'SHAM_bbox_inclusive':sham,'same_rectangle_pattern_and_alpha_mass':w*h,'SHAM_selection':'all in-canvas integer translations disjoint from both original ROIs; minimum L1 translation, then y,x; no q/native/score optimization','COVER_contains_display_qcenter':contains_qcenter(cover),'SHAM_contains_display_qcenter':None if sham is None else contains_qcenter(sham),'actual_controls_constructed':False,'actual_pair_labels_created':False,'required_actual_proofs':['original clean96+valid96 exact identity','alpha all-valid in both locations','full original ROI union opaque COVER','same donor RGB+alpha pattern and alpha mass in SHAM','COVER actual finite-support RGB changed','SHAM entire finite-support RGB unchanged','same-part visible context preserved without q/ROI shrink','original qL/native/U/role unchanged'],'planned_feature_visible_relation':{'BEFORE':1,'COVER':0,'SHAM':1},'planned_relation_authority':'FINITE_LOCAL_WITNESS_ONLY_NOT_NATIVE_E_OR_QUERY_SUPPORT','actual_feature_visible_targets':None,'training_admission':False}
def anatomy_status(frame,native):
 if frame==6 and native in (4,5,6):return 'F6_LINKED_BOARD_FEATURE_NOT_NATIVE_ADDRESSABLE'
 if (frame,native) in ((0,6),(5,7)):return 'DECK_VS_BOTTOM_RUNNER_UNRESOLVED'
 return 'WEAK_TOP_ROLE_PART_ALIGNMENT_REVIEW_REQUIRED'
def pure_checks():
 checks=[]
 def t(name,v):need(v,name);checks.append(name)
 a=[1,2,2,3];c=[4,2,4,2];s=finite_support(a,c)
 t('Exact_OR_not_envelope_no_shrink',s['positive_pixels']==5 and s['positive_runs_y_x0_x1']==[[2,1,2],[2,4,4],[3,1,2]] and s['union_bbox']==[1,2,4,3])
 t('Single_pixel_inclusive_ROI_allowed',finite_support([0,0,0,0],[0,0,0,0])['positive_pixels']==1)
 t('Boundary_negative_float_ROI_rejected',not roi_valid([-1,0,1,1]) and not roi_valid([0.,0,1,1]) and not roi_valid([0,0,96,1]))
 p=pair_geometry([40,40,50,50],[42,42,52,52]);sh=p['SHAM_bbox_inclusive']
 t('COVER_full_union_SHAM_disjoint_same_mass',p['same_rectangle_pattern_and_alpha_mass']==169 and not intersects(sh,[40,40,50,50]) and not intersects(sh,[42,42,52,52]))
 t('No_room_no_SHAM_not_shrunk',pair_geometry([0,0,95,95],[0,0,95,95])['SHAM_bbox_inclusive'] is None)
 t('Remote_COVER_and_center_SHAM_not_query_labels',not contains_qcenter([0,0,2,2]) and contains_qcenter([47,47,49,49]))
 g=[dict(native_slot=k,position=0,frame_key='invented',component_id='g',internal_source_role='SOURCE_FIT',original_query_available=k!=7,query_xy=[float(k),float(-k)],L=2.) for k in range(8)];b=copy.deepcopy(g)
 q=query_condition(g,0)
 t('Original_availability_self_mask_FP32',q['other_mask8']==[False,True,True,True,True,True,True,False] and q['other_dxdy16_FP32'][2:4]==[.5,-.5])
 t('Active_U_context_zero_no_finite_imputation',not any(query_condition(g,7)['other_mask8']) and query_condition(g,7)['other_dxdy16_FP32']==[0.]*16)
 g[6]['query_xy']=[999.,-999.];t('F64_normalized_clipping_after_original_q',query_condition(g,0)['other_dxdy16_FP32'][12:14]==[8.,-8.])
 t('Caller_original_metadata_immutable',b[0]==g[0])
 t('F6_and_runner_mismatch_not_nativeD',anatomy_status(6,6).startswith('F6_') and anatomy_status(0,6)=='DECK_VS_BOTTOM_RUNNER_UNRESOLVED')
 try:opened('/V112/not-opened');raise AssertionError('Forbidden path accepted')
 except ValueError:checks.append('Lexical_V112_rejected_before_FS')
 return {'completed':True,'PASS':True,'count':len(checks),'checks':checks,'actual_image_GT_model_reads':0}
def run():
 scope=json.loads(checked(SCOPE));allowed={b['path']:b for b in scope['allowed_saved_metadata']};inputs={'scope':SCOPE}
 def load(suffix,lines=False):
  matches=[b for p,b in allowed.items() if p.endswith(suffix)];need(len(matches)==1,'Exact allowed metadata suffix once');b=matches[0];inputs[suffix]=b;z=checked(b);return [json.loads(r) for r in z.splitlines()] if lines else json.loads(z)
 dictionary=load('/NATIVE8_EXPECTED_PART_AND_PROOF_REQUIREMENTS.jsonl',True)
 definition=load('/DEFINITION_AND_FEASIBILITY.json');semantic=load('/ROOT_ACTUAL_ALL12_VISUAL_AND_SEMANTIC_QA.json')
 joined=load('/ALL96_JOINED_CANDIDATES.json');synthesis=load('/ROOT_QUERY_INSTANCE_PART_SUPPORT_SYNTHESIS.json');nominal=load('/ALL120_CANDIDATE_NATIVE_QUERY_DIAGNOSTICS.json')
 need([r['native_slot'] for r in dictionary]==list(range(8)) and len(joined['rows96'])==len(semantic['rows96'])==96 and len(synthesis['rows120'])==len(nominal['rows120'])==120,'Original populations')
 js={r['input']['row_id']:r for r in joined['rows96']};ss={r['row_id']:r for r in semantic['rows96']};need(set(js)==set(ss),'Exact original row join')
 ns=collections.defaultdict(list)
 for r in nominal['rows120']:ns[r['original_input']['row_id']].append(r)
 ledger=[];conditions=[];targets=[];pairs=[];correspondence=[];pngs={}
 for row_id,j in js.items():
  o=j['input'];s=ss[row_id];n=o['native_slot'];f=o['position'];role=o['internal_source_role']
  need((f,n,role,not o['original_query_available'])==(s['frame_position'],s['native_slot'],s['internal_source_role'],s['original_inputU']),'Original role/native/U exact')
  group=sorted([r['input'] for r in js.values() if r['input']['position']==f],key=lambda x:x['native_slot']);cond=query_condition(group,n)
  context={'row_id':row_id,'frame_position':f,'original_global_index':o['original_global_index'],'original_query_available':o['original_query_available'],'original_q':o['query_xy'],'original_L':o['L'],'native_slot':n,'condition':cond}
  conditions.append(context);gt_candidates=[]
  for r in ns[row_id]:
   a=r['original_input'];need(a['query_xy']==o['query_xy'] and a['L']==o['L'] and a['original_query_available']==o['original_query_available'] and a['native_slot']==n,'Saved nominal original identity exact')
   gt_candidates.append({'GT_candidate_index':r['diagnostic_candidate_gt_list_index'],'selected_instance':False,'canonical_index':r['canonical_index'],'nominal_reference_xy':r['nominal_reference_xy'],'RAW96_reference_address':r['query_diagnostic']['RAW96_reference_address'],'nominal_mapping_status':r['nominal_mapping_status'],'physical_native_gold':False})
  syn=[r for r in synthesis['rows120'] if r['row_id']==row_id];need(len(syn)==len(gt_candidates),'120 candidate expansion retained')
  weak=s['root_weak_part_state']=='D';auxid='FINITE_LOCAL_PART__'+row_id if weak else None
  row={'input':copy.deepcopy(o),'native_role_dictionary':dictionary[n],'dictionary_is_8_roles_not_per_query':True,'historical_root_weak_part_state':s['root_weak_part_state'],'historical_query_feature_support_agreement':s['raw_query_support_agreement'],'historical_root_review_status':s['root_semantic_review_status'],'finite_local_part_target_id':auxid,'expected_native_part_correspondence':anatomy_status(f,n) if weak else 'UNRESOLVED_NO_NEW_PART_TARGET','native_visibility_D_E_S':'UNKNOWN','native_DE_target':-1,'GT_candidate_diagnostics_target_only':gt_candidates,'selected_GT_instance_index':None,'original_inputU_preserved':not o['original_query_available'],'deployment_condition_contains_GT_or_ROI':False,'primary_training_admission':False,'auxiliary_training_admission':False}
  ledger.append(row)
  if not weak:continue
  need(o['original_query_available'] and j['reader_A_axes']['witness']['status']==j['reader_C_axes']['witness']['status']=='VISIBLE','Positive two-reader visible evidence')
  a=j['reader_A_axes']['witness']['roi'];c=j['reader_C_axes']['witness']['roi'];support=finite_support(a,c);pair=pair_geometry(a,c)
  target={'target_id':auxid,'input':copy.deepcopy(o),'local_part_visible_weak_target':1,'authority':'SEALED_TWO_READER_PLUS_ROOT_SAME_PALLET_LOCAL_PART_WEAK_NOT_NATIVE_GOLD','observed_part_A':j['reader_A_axes']['part'],'observed_part_C':j['reader_C_axes']['part'],'observed_witness_A':j['reader_A_axes']['witness'],'observed_witness_C':j['reader_C_axes']['witness'],'historical_addressability_A':j['reader_A_axes']['addressability'],'historical_addressability_C':j['reader_C_axes']['addressability'],'query_support_A':j['reader_A_axes']['query_support'],'query_support_C':j['reader_C_axes']['query_support'],'query_feature_relation_agreement':s['raw_query_support_agreement'],'expected_native_part_correspondence':anatomy_status(f,n),'native_DE_target':-1,'finite_target_support':support,'support_offsets_from_original_qcenter':{'center':[47.5,47.5],'ROI_A_xyxy_minus47_5':[v-47.5 for v in a],'ROI_C_xyxy_minus47_5':[v-47.5 for v in c]},'ROI_use':'TRAINING_TARGET_ONLY_NOT_DEPLOYMENT_INPUT_NOT_corrected_q','positive_mask_outside_IGNORE_not_material_negative':True,'clean96_binding':j['display_bindings']['clean96'],'original_context_binding':j['display_bindings']['full_context'][0],'exact_valid96_binding':None,'valid96_binding_status':'MISSING_IN_ALLOWED_JOIN_NEEDS_ROOT_EXACT_BOUND_METADATA_BEFORE_PAIR_PIXEL_PROOF','potential_fit_role_only':role=='SOURCE_FIT','SOURCE_DEV_excluded_all_future_fit_branches':role=='SOURCE_DEV','training_admission':False,'new_images_viewed':0}
  targets.append(target);pairs.append({'target_id':auxid,'row_id':row_id,'component_id':o['component_id'],'source_group_key':o['source_group_key'],'original_role':'TRAIN','internal_source_role':role,'fit_role_allowed':role=='SOURCE_FIT','finite_support':support,'geometry_plan':pair,'original_qLnative_preserved':[o['query_xy'],o['L'],n],'RGB_and_validity_proofs':'PENDING_ROOT_EXACT_IMAGE_ENTRY','no_native_or_query_support_labels_from_cover_sham_names':True})
  correspondence.append({'row_id':row_id,'frame_position':f,'native_slot':n,'internal_source_role':role,'expected_anatomy':dictionary[n]['nominal_expected_part'],'observed_A':j['reader_A_axes']['witness']['description'],'observed_C':j['reader_C_axes']['witness']['description'],'correspondence_status':anatomy_status(f,n),'nominal_candidates':gt_candidates,'native_DE_target':-1,'finite_local_part_visible_target':1})
  for b in (target['clean96_binding'],target['original_context_binding']):pngs[b['path']]=b
 need(len(ledger)==96 and sum(r['original_inputU_preserved'] for r in ledger)==11 and len(targets)==9,'96/U11/weak9 preserve')
 need(collections.Counter(t['input']['internal_source_role'] for t in targets)=={'SOURCE_FIT':7,'SOURCE_DEV':2},'9 targets role7/2 unchanged')
 need(len({t['input']['position'] for t in targets})==5 and len(pngs)==14,'9clean+5context exact metadata only')
 outputs={}
 outputs['all96']=newjl('ALL96_INTENDED_PART_QUERY_CANDIDATES.jsonl',ledger)
 outputs['conditions96']=newjl('ALL96_DEPLOYABLE_QUERY_CONDITIONS.jsonl',conditions)
 outputs['finite_targets9']=newj('ALL9_EXPLICIT_FINITE_LOCAL_PART_TARGETS.json',{'completed':True,'targets9':targets,'training_admission':False})
 outputs['pair_plans9']=newj('ALL9_FINITE_WITNESS_COVER_SHAM_EXAMPLE_PLANS.json',{'completed':True,'plans9':pairs,'actual_controls_constructed':False,'training_admission':False})
 outputs['correspondence9']=newj('WEAK9_NATIVE_ANATOMY_CORRESPONDENCE_SEPARATED.json',{'completed':True,'rows9':correspondence,'native_D_E_created':0,'training_admission':False})
 outputs['PNG_whitelist']=newj('EXACT9_CLEAN96_AND5_CONTEXT_BINDINGS_NO_IMAGE_ACCESS.json',{'completed':True,'PNG_bindings14':list(pngs.values()),'PNG_stat_hash_read_decode':0,'future_approved_image_entry_required':True,'training_admission':False})
 flat=[{'row_id':r['input']['row_id'],'frame_position':r['input']['position'],'native_slot':r['input']['native_slot'],'source_role':r['input']['internal_source_role'],'input_available':r['input']['original_query_available'],'native_name':r['native_role_dictionary']['native_name'],'expected_anatomy':r['native_role_dictionary']['nominal_expected_part'],'local_part_visible_weak_target':1 if r['finite_local_part_target_id'] else -1,'correspondence_status':r['expected_native_part_correspondence'],'native_DE_target':-1,'training_admission':False} for r in ledger]
 outputs['CSV96']=new('ALL96_TARGET_CANDIDATE_READOUT.csv',csvbytes(flat))
 summary={'completed':True,'original_frames':12,'original_rows':96,'original_U':11,'original_roles':dict(collections.Counter(r['input']['internal_source_role'] for r in ledger)),'finite_local_part_visible_targets':9,'finite_target_roles':dict(collections.Counter(t['input']['internal_source_role'] for t in targets)),'correspondence_counts':dict(collections.Counter(r['correspondence_status'] for r in correspondence)),'positive_support_pixels9':[t['finite_target_support']['positive_pixels'] for t in targets],'bbox_only_SHAM_possible':sum(p['geometry_plan']['SHAM_bbox_inclusive'] is not None for p in pairs),'no_geometric_SHAM_preserved':sum(p['geometry_plan']['SHAM_bbox_inclusive'] is None for p in pairs),'native_labels_known':0,'native_unknown':96,'GT_candidates_unselected':15,'nominal_expanded_rows':120,'deployable_context_rows':96,'exact_future_PNG_bindings':14,'all_source_image_GT_payload_model_reads':0,'controls_constructed':0,'training_admission':False}
 outputs['summary']=newj('SUMMARY.json',summary)
 outputs['checks']=newj('INVENTED_FUNCTIONAL_CHECKS.json',pure_checks())
 contract={'completed':True,'schema':'V218_INTENDED_PART_QUERY_CANDIDATES_V1','scope':SCOPE,'dictionary':'8 camera-dynamic role rows, not per-query native labels','actual_constructed':'96-row exact metadata join;96 target-free FP32 conditions;9 positive weak finite ROI-support targets from exact OR;9 geometry-only pair plans','supervision_separation':{'native_D_E_S':'UNKNOWN all96; no old weak9 or mask/reference copied','finite_part':'Positive weak local pallet feature support. ROI only annotation target, outside IGNORE, not exact pixel segmentation. Explicit part targeting needed for future loss; not unconditioned native D/E','query_support':'Separate historical feature ON/NEAR/OFF/UNKNOWN. No COVER/SHAM relation copied','point_localization':'Original q/L and reference offsets unchanged; no corrected query/GT position output'},'finite_pair_contract':'Same alpha/RGB pattern+mass; COVER all original ROI union; SHAM disjoint/union-RGB unchanged; actual valid and RGB proof mandatory; no ROI shrinking/forceSHAM. Names never native E','fit_dev':'Outer TRAIN unchanged. Only7 SOURCE_FIT candidate positives can enter a future admitted fit;2 SOURCE_DEV and all descendants excluded all training branches. Prior exposure not erased','deployment_features':'Stored q8/native/L/availability only; self0; original U/other unavailable0; F64 normalization then clip[-8,8], FP32. No ROI/GT/part label/role','future_missing':'Exact valid96 PNG metadata binding required before actual pair pixel proofs. No automatic path construction, no fallback.14 known clean/context PNGs not opened now','training_admission':False,'new_fit_projection_or_payload_read':0}
 outputs['contract']=newj('TARGET_AND_CONDITION_CONTRACT.json',contract)
 outputs['future_image_entry_false']=newj('FUTURE_EXACT_IMAGE_ENTRY_TEMPLATE_APPROVED_FALSE.json',{'approved_actual_images_or_control_construction':False,'scope':SCOPE,'PNG_binding_plan':outputs['PNG_whitelist'],'exact_valid96_bindings':None,'target_candidates':outputs['finite_targets9'],'pair_plans':outputs['pair_plans9'],'native_DE_admission':False,'training_admission':False})
 lines=['# V218: 실제 part/query 타깃 후보 구성','', 'V159 사전은 query별 라벨이 아니라 camera-dynamic 8역할 정의다. top0/1/4/5는 상판 끝, bottom2/3/6/7은 하단 runner 끝이다. front/rear 이름만으로 보임/자기가림을 정하지 않는다.','', '원96행·U11·FIT64/DEV32를 보존했다. native D/E/S는 전96행 UNKNOWN이다. 기존 weak9는 nativeD9가 아니라 같은 팔레트의 local part가 보인다는 약한 관찰이다.','', '이번에 실제로 만든 데이터는 96행 대응표, 원 q8/native/L/availability만으로 만든 96행 FP32 조건, 두 원 ROI의 정확 OR로 만든 9개 finite support 타깃(7FIT/2DEV), 같은 support를 위한 9개 COVER/SHAM 기하 배치안이다. ROI는 학습 타깃에만 사용하며 배포 입력에 넣지 않는다. ROI 바깥은 음성 물질/배경 정답이 아니라 IGNORE다.','', '|frame/native|원역할|예상 native 부위|현재 대응 상태|','|---|---|---|---|']
 for r in correspondence:lines.append('|%d / %d|%s|%s|%s|'%(r['frame_position'],r['native_slot'],r['internal_source_role'],r['expected_anatomy'],r['correspondence_status']))
 lines+=['', 'f6의 3개 board/edge witness는 native 주소화와 분리했다. f0n6·f5n7은 상판 하단 윤곽 관찰과 사전의 bottom-runner가 다르므로 미해결이다. 나머지 top-role4행도 약한 부위 대응 검토 후보이며 native gold나 자동 native D가 아니다. 완벽한 3D mesh 인증이나 q와 참조점의 정확 일치를 일괄 선결조건으로 세운 것은 아니다.','', 'COVER는 원 두 ROI 합집합 전체를 가리고 SHAM은 같은 면적·패턴을 다른 위치에 두는 계획이다. 현재 수치는 bbox 기하 가능성뿐이다. actual alpha all-valid, COVER 실제 변경, SHAM finite support 불변, 같은 part 맥락 보존을 root 승인 이미지 단계에서 검증해야 한다. q가 바뀌었는지와 feature를 가렸는지는 다른 축이다. 배치가 없으면 factual 관찰만 남기며 ROI를 줄이지 않는다.','', '두 SOURCE_DEV 후보와 모든 파생형은 모든 학습 loss에서 제외한다. 이전 appearance 노출과 옛 SOURCE 학습 이력은 사라지지 않으며 독립 신규 검증이라고 부르지 않는다. 이번에는 새 이미지·GT payload·모델·fit·paired image를 만들거나 읽지 않았고 training admission은 모두 false다.','', '다음 한 단계는 [고정14 PNG 바인딩](EXACT9_CLEAN96_AND5_CONTEXT_BINDINGS_NO_IMAGE_ACCESS.json)과 별도로 필요한 valid96 정확 바인딩을 root가 고정한 뒤, 이 [9개 계획](ALL9_FINITE_WITNESS_COVER_SHAM_EXAMPLE_PLANS.json)의 실제 crop/valid/alpha/RGB proof를 수행하는 것이다. native part 대응이 미확정이어도 finite local-part auxiliary 타깃 자체는 보존한다.','', '데이터: [원96 CSV](ALL96_TARGET_CANDIDATE_READOUT.csv), [finite9](ALL9_EXPLICIT_FINITE_LOCAL_PART_TARGETS.json), [native 부위 분리표](WEAK9_NATIVE_ANATOMY_CORRESPONDENCE_SEPARATED.json), [계약](TARGET_AND_CONDITION_CONTRACT.json), [요약](SUMMARY.json).','']
 outputs['KO']=new('KO.md','\n'.join(lines).encode())
 code=bind(__file__,rb(__file__))
 done=newj('COMPLETE.json',{'completed':True,'PASS':True,'code':code,'protocol':bind(H+'/PROTOCOL_BEFORE_METADATA_JOIN_AND_TARGET_BUILD.json',rb(H+'/PROTOCOL_BEFORE_METADATA_JOIN_AND_TARGET_BUILD.json')),'input_metadata_bindings':inputs,'outputs':outputs,'summary':summary,'image_payload_stat_hash_read_decode':0,'new_native_labels_fit_models_or_GT_payload_reads':0,'training_admission':False,'frozen_no_further_edits':True})
 print(json.dumps({'completed':True,'PASS':True,'complete':done,'code':code,'summary':summary},ensure_ascii=False))
if __name__=='__main__':run()
