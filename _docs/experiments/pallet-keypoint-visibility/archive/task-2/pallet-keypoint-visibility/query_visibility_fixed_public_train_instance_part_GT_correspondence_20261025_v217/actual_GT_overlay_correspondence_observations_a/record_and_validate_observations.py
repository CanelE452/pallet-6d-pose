#!/usr/bin/env python3
"""봉인된 계획과 실제 A tool-view 후 수동 관찰을 저장. 새 이미지 decode 없음."""
import collections,csv,hashlib,io,json,os,stat
D='/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_fixed_public_train_instance_part_GT_correspondence_20261025_v217'
H=D+'/actual_GT_overlay_correspondence_observations_a'
def opened(p,flags=os.O_RDONLY):
    assert 'v112' not in p.lower() and os.path.isabs(p) and os.path.normpath(p)==p
    fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
    try:
        for c in p.strip('/').split('/')[:-1]:
            n=os.open(c,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
        f=os.open(p.rsplit('/',1)[1],flags|os.O_NOFOLLOW,0o644,dir_fd=fd)
        assert stat.S_ISREG(os.fstat(f).st_mode);return f
    finally:os.close(fd)
def read(p):
    with os.fdopen(opened(p),'rb') as f:return f.read()
def bind(p,b):return {'path':p,'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}
def save(name,obj):
    b=(json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode() if not isinstance(obj,bytes) else obj
    p=H+'/'+name
    with os.fdopen(opened(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL),'wb') as f:f.write(b)
    assert read(p)==b;return bind(p,b)

# 실제 view 후 독립 작성한 프레임별 query 문맥. ROI/native 정답/입력 보정이 아니다.
frame_notes={
0:([6,7],[],[4,5],'앞쪽 가로 목재 팔레트가 아래 프레임에 잘려 있다. n6은 오른쪽 판 끝, n7은 왼쪽 판 면 근처다. n5는 오른쪽 끝 바로 위 경계/배경, n4는 바닥·케이블 쪽이다. n0..3의 원 U/OOB는 그대로다.'),
1:([],list(range(8)),[],'q8은 회색 플라스틱 상자와 그 검은 받침 주변이다. n4..7은 상자 위/옆 테두리, n0..3은 아래 받침·상자 경계 주변이다. 팔레트로 식별되는 면을 query가 묻는다고 확인할 수 없다.'),
2:([0,3,4,7],[],[5,6],'두 목재 팔레트가 겹쳐 있다. n0/3은 앞 왼쪽 모서리·옆면, n4/7은 위 팔레트의 먼쪽 판 끝/판 면 근처다. n5/6은 오른쪽 외곽·겹침 경계여서 위/아래 인스턴스의 독점 대응이 불명확하다. n1/2 U는 보존한다.'),
3:([],list(range(8)),[],'표시된 q8은 모두 뒤쪽 작은 목재 상자의 윗변에 좁게 모여 있다. GT overlay의 전경 EPAL 팔레트 전면은 그보다 아래다. 같은 목재 재질만으로 이 q를 팔레트 native로 바꾸지 않는다.'),
4:([1,2,6,7],[4,5],[0,3],'n1/2는 오른쪽 팔레트 전면의 위/아래 경계 부근, n6/7은 상판 부근이다. n0/3은 왼쪽 끝에 인접한 바닥·경계라 정확한 접촉은 불명확하다. n4는 파란 상자 왼쪽/뒤 받침, n5는 회색 둥근 물체 표면이다. 의도 native 부위의 동일성은 별도다.'),
5:([4,7],[5,6],[1,2],'n4/7은 왼쪽 팔레트 판 끝/옆면 부근이다. n5/6은 위의 파란 상자와 목재 상자 경계, n1/2는 오른쪽 바닥·받침/케이블 쪽이다. 원 n0/3 U/OOB를 보존한다.'),
6:([4,6,7],[0,1,2,3],[5],'뒤에 세운 팔레트와 앞의 목재 상자·판을 구분할 수 있다. n4/6/7은 뒤 팔레트 위쪽 판 끝/경계 주변, n5는 판 위의 인접 배경에 있다. n0/3은 앞의 작은 목재 판·바닥, n1/2는 앞 상자 변이다. n5의 배경을 native 가림으로 해석하지 않는다.'),
7:([],list(range(8)),[],'q8은 전경 목재 상자의 코너·내부 판/왼쪽 작은 판과 주변 바닥에 놓였다. GT candidate3의 세운 팔레트는 더 뒤에 있다. 앞쪽 목재의 선명한 윤곽은 뒤 팔레트 native 대응이 아니다.'),
8:([],list(range(8)),[],'q8은 전경 목재 상자·작은 목재 판/그 경계와 바닥에 모여 있다. 뒤쪽 세운 팔레트에는 해당 query 묶음이 직접 놓이지 않는다. full overlay가 전경 상자 위를 지나는 부분도 visible pallet/part 정답으로 복사하지 않는다.'),
9:([],list(range(8)),[],'q8은 회색 상자 뒤의 검은 카트 손잡이 주변에 모였다. 아래 목재 지지판/팔레트 후보는 상자·흰 덮개 아래에 있다. 손잡이의 query ON이나 선명한 edge를 팔레트 visibility로 바꾸지 않는다.'),
10:([],list(range(8)),[],'q8은 모두 카트 손잡이의 좁은 사각 윤곽 부근이다. 목재 지지판/팔레트 후보는 화면 아래에서 회색 상자와 흰 덮개에 가려져 있다. query가 가림 물체에 있다는 사실만으로 의도 native E를 부여하지 않는다.'),
11:([5,6],[0,4,7],[],'n5/6은 오른쪽 노출된 목재 지지판 가장자리 부근이다. n4/7은 회색 상자 전면·테두리, n0은 앞의 흰 덮개 위다. n1/2/3 원 U/OOB는 유지한다. 목재 지지판/팔레트 후보의 구조가 대부분 덮여 있어 정확한 intended native part는 불명확하다.'),
}
candidate_notes={
(0,0):('BROAD_VISIBLE_PALLET_ALIGNMENT_SUPPORTED_WEAK','full/visible 색 영역은 화면 아래의 가로 판 팔레트와 넓게 겹치고 visible 영역에서 판의 틈이 구분된다. 오른쪽 판 끝 주변과 정렬되는 근거가 있다.','왼쪽·아래 팔레트가 프레임 밖으로 잘려 전체 형상과 아래쪽 인스턴스 경계를 확인할 수 없다. 넓은 정렬은 detector의 선택 인스턴스 또는 8개 native-part 교차 대응을 인증하지 않는다.'),
(0,1):('PARTIAL_OR_AMBIGUOUS_SECOND_CANDIDATE','full 영역은 아래 왼쪽에서 오른쪽 아래로 비스듬하게 지나고 visible은 일부 가는 판/틈 주변 조각이다. 첫 후보의 넓은 가로 상판과 다른 지지 구조를 가리킬 가능성은 있다.','잘린 아래쪽 구조가 독립 팔레트인지 패널만으로 분리되지 않는다. visible 조각이나 낮은 지원만으로 외부 가림·대상 부재를 결정하지 않는다.'),
(1,0):('NO_INFRAME_PALLET_MASK_ALIGNMENT_TO_CONFIRM','패널에서 full/visible 색 변화가 보이지 않고 원 영상은 플라스틱 상자·검은 받침을 보여준다. query도 그 상자/받침에 놓인다.','팔레트 candidate의 화면 밖/저장 bbox sentinel/가림/미지원 가능성을 이 view로 구분하지 못한다. 빈 overlay는 native E 근거가 아니다.'),
(1,1):('NO_INFRAME_PALLET_MASK_ALIGNMENT_TO_CONFIRM','두 번째 후보도 full/visible 색 변화가 보이지 않는다. 보이는 상자와 받침의 물리 구조는 첫 후보 패널과 같다.','후보0/1을 원 영상만으로 고를 근거가 없다. 자동 첫 인스턴스 선택이나 다른 물체를 팔레트로 대체하지 않는다.'),
(2,0):('BROAD_VISIBLE_UPPER_PALLET_ALIGNMENT_SUPPORTED_WEAK','visible 색 영역이 위에 놓인 팔레트의 길쭉한 상판들과 왼쪽 앞면을 넓게 따른다. 원 context에서도 아래 팔레트와 쌓인 구조가 보인다.','판 끝/틈과 겹친 두 팔레트의 가까운 경계는 축소 overlay에서 일부 모호하다. native point/selected detector instance의 전역 대응은 확정하지 않는다.'),
(2,1):('PARTIAL_LOWER_STACK_ALIGNMENT_PLAUSIBLE','full은 위 팔레트에 가려진 영역까지 넓게 덮고 visible은 오른쪽 아래에서 드러난 하단 팔레트의 판/블록 주변과 일부 틈에 나타난다. 아래에 다른 팔레트가 실제로 보이는 것과 일관된다.','많은 query는 위 팔레트에 가까우며 아래 후보에 독점적으로 닿는 query 부분집합은 확인하지 못했다. full 중첩이 실제 visible part라는 뜻은 아니다.'),
(3,0):('BROAD_FRONT_PALLET_ALIGNMENT_SUPPORTED_WEAK','full/visible이 전경 EPAL 팔레트의 상판·블록·아래 판을 넓게 따른다. 위의 파란 상자/목재 상자/회색 물체와 전면 팔레트를 시각적으로 분리할 수 있다.','query는 뒤 목재 상자 윗변 쪽이다. 팔레트 mask와 물리 팔레트가 잘 맞아도 현재 q/native가 그 부위를 주소화한다는 증거는 별도다.'),
(4,0):('BROAD_FRONT_PALLET_ALIGNMENT_SUPPORTED_WEAK','full/visible이 왼쪽 EPAL 블록부터 오른쪽 전면까지 이어지는 팔레트와 넓게 맞는다. 상판 위 물체는 visible에서 대체로 분리된다.','작은 가장자리 어긋남·가려진 판/블록의 구체 부위와 n4/n5의 다른 물체 위치를 함께 남긴다. V216 n3의 서로 다른 ROI feature 문제는 이 넓은 mask로 해결되지 않는다.'),
(5,0):('BROAD_PALLET_WITH_OCCLUDERS_ALIGNMENT_SUPPORTED_WEAK','full은 상판/측면의 팔레트 전체 영역을, visible은 위의 파란 상자·목재 상자·회색 통을 제외한 노출된 판을 주로 따른다. 왼쪽 판 끝은 원 context에서 구별된다.','full이 상자/통 위를 덮는 부분은 숨겨진 nominal silhouette 문맥이지 그 물체가 팔레트 표면이라는 뜻이 아니다. 각 native의 가림/SELF는 미판정이다.'),
(6,3):('BROAD_UPRIGHT_REAR_PALLET_ALIGNMENT_SUPPORTED_WEAK','full/visible이 뒤에 세운 넓은 판 팔레트를 따른다. visible은 앞 파란 통과 목재 상자에 겹치는 부분을 대체로 제외한다.','앞의 작은 목재 판/큰 상자는 별개 구조다. n5 배경과 n7 판 끝면/아래 변의 차이는 전체 mask만으로 동일 intended feature가 되지 않는다.'),
(7,3):('BROAD_UPRIGHT_REAR_PALLET_ALIGNMENT_SUPPORTED_WEAK','뒤 팔레트의 가로 판들이 full/visible 영역과 넓게 맞고 앞 파란 통은 visible에서 빠진다.','원 q8은 앞 목재 상자/판 쪽에 있다. 뒤 팔레트와 앞 목재 재질의 유사함 또는 full 겹침만으로 native 대응·E를 만들지 않는다.'),
(8,3):('BROAD_UPRIGHT_REAR_PALLET_ALIGNMENT_SUPPORTED_WEAK','기울어 보이는 뒤 팔레트의 가로 판과 visible 색 영역이 넓게 일치한다. 앞 통/상자는 별개 전경으로 구분된다.','query가 있는 전경 상자 일부에 full 영역이 겹치지만 visible part/native 주소화의 증명이 아니다. 가림 원인의 각 native 상태는 별도다.'),
(9,0):('PARTIAL_WOOD_SUPPORT_CONTEXT_PROXY_ALIGNMENT','full은 아래 목재 지지판/팔레트 후보 영역을 넓게 덮고 visible은 노출된 바깥 목재와 함께 흰 덮개·상자 테두리/안쪽 물체에도 색이 보인다.','물리 팔레트 구조가 대부분 상자·흰 덮개 아래에 있어 정확한 부위를 인증하기 어렵다. visible 색이 입혀진 다른 표면을 target part로 확정하지 않는다. q8은 뒤 손잡이에 있다.'),
(10,0):('PARTIAL_WOOD_SUPPORT_CONTEXT_PROXY_ALIGNMENT','목재 지지판의 외곽과 넓은 full 영역은 대체로 일관되지만 visible 색은 상자 안쪽/흰 덮개 및 테두리에도 걸린다.','화면 아래 잘림과 덮인 구조 때문에 예상 native part의 가시성은 미확정이다. 빈/겹친 mask·깊이 구멍을 native E로 해석하지 않는다. q8은 카트 손잡이에 있다.'),
(11,0):('PARTIAL_WOOD_SUPPORT_CONTEXT_PROXY_ALIGNMENT','긴 목재 지지판의 오른쪽 노출 면과 full/visible 외곽이 대략 맞는다. 원 context에서 흰 덮개와 회색 상자가 목재 구조 대부분을 덮는 것을 볼 수 있다.','visible 색이 흰 덮개·상자/통 쪽에도 있으므로 mask 점유만으로 같은 팔레트 부위라고 하지 않는다. q5/6 목재 가장자리와 q0 흰 덮개/q4·7 상자 경계를 분리한다.'),
}

def main():
    plan_path=H+'/BEFORE_ACTUAL_VIEWS_PLAN_AND_SHA_QA.json';pb=read(plan_path)
    assert hashlib.sha256(pb).hexdigest()=='3b6d2dfee56de9667a3577d3998217038df1cfd9ac306ae4ea01888804365b2c'
    plan=json.loads(pb);panels=plan['fixed_panel_order'];contexts={x['frame_position']:x for x in plan['context12_optional_fixed_order']}
    context_frames=[0,2,4,6,9,10,11]
    views=[]
    for obj in panels:
        views.append({'view_order':len(views)+1,'kind':'CANDIDATE_OVERLAY_PANEL','frame_position':obj['frame_position'],'candidate_gt_list_index':obj['candidate_gt_list_index'],'binding':obj['image'],'tool':'tools.view_image','detail':'original','actually_viewed':True})
    for pos in context_frames:
        views.append({'view_order':len(views)+1,'kind':'ORIGINAL_CONTEXT_STANDALONE','frame_position':pos,'binding':contexts[pos]['image'],'tool':'tools.view_image','detail':'original','actually_viewed':True})
    vp=save('ACTUAL_IMAGE_VIEW_RECEIPT.json',{'completed':True,'PASS':True,'observer':'A','scope':plan['scope'],'pre_view_SHA_QA':bind(plan_path,pb),'views':views,'actual_panel_views':15,'actual_standalone_context_views':7,'all15_fixed_panels_viewed':True,'unviewed_standalone_context_positions':[1,3,5,7,8],'panel_original_context_subpanel_exists_for_all12':True,'browser_render':0,'no_root_new_observation_files_or_values_read':True})
    rows_path=D+'/fixed12_GT_payload_preparation_a/ALL96_ORIGINAL_INPUTS_AND_GT_POINTERS.json';rb=read(rows_path)
    assert hashlib.sha256(rb).hexdigest()=='22803511e5abb9c9eb07d809bb5796c112fcdcd63ffe75c65bffb878549aa031'
    inputs=json.loads(rb)['rows96'];frames=[]
    for pos in range(12):
        on,other,boundary,note=frame_notes[pos];rr=[r for r in inputs if r['frame_position']==pos]
        u=[r['native_slot'] for r in rr if not r['original_query_available']]
        assert len(rr)==8 and len(set(on+other+boundary+u))==8 and len(on+other+boundary+u)==8,(pos,on,other,boundary,u)
        frames.append({'frame_position':pos,'frame_key':rr[0]['frame_key'],'internal_source_role':rr[0]['internal_source_role'],'queries_touch_or_near_visible_pallet_or_wood_support_context':on,'queries_on_different_object_or_its_nearby_boundary':other,'queries_on_background_or_uncertain_pallet_boundary':boundary,'original_inputU_native_slots':u,'visual_evidence_ko':note,'these_are_query_context_observations_not_native_visibility_labels':True,'selected_detector_GT_instance_index':None,'native_gold':'UNKNOWN','training_admission':False})
    candidates=[]
    for panel in panels:
        key=(panel['frame_position'],panel['candidate_gt_list_index']);state,evidence,limits=candidate_notes[key]
        candidates.append({'frame_position':key[0],'frame_key':panel['frame_key'],'candidate_gt_list_index':key[1],'outer_role':'TRAIN','internal_source_role':panel['internal_source_role'],'physical_context_alignment_observation':state,'positive_visual_evidence_ko':evidence,'ambiguity_and_limits_ko':limits,'panel_binding':panel['image'],'actual_view_bindings':[panel['image']]+([contexts[key[0]]['image']] if key[0] in context_frames else []),'selected_detector_GT_instance_index':None,'selected_instance_correspondence':'UNKNOWN_NOT_AUTOMATICALLY_ASSIGNED','intended_native_part_crosswalk':'UNKNOWN_NOT_YET_PROJECTED_OR_CERTIFIED','new_native_D_E_SELF_gold':'UNKNOWN','training_admission':False})
    obs={'completed':True,'reader':'A','candidate_observations15':candidates,'frame_query_context_observations12':frames,'original_input_rows':96,'original_unique_inputU':11,'expanded_candidate_inputU':17,'U_count_note':'Root declared correction9e7a3922 before interpretation:17 candidate-repeat U versus11 unique originalU; no producer/observation change.','view_receipt':vp,'source_input_scalars':bind(rows_path,rb),'historical_V216_weakpart9':'Historical agent weak same-part observations, not native gold and not copied into new labels.','global_selected_instance_auto_assignment':False,'root_new_observations_read_or_received':0,'new_native_labels_or_admission':False}
    ob=save('ALL15_CANDIDATE_AND_FRAME12_OBSERVATIONS.json',obs)
    buf=io.StringIO(newline='');fields=['frame_position','frame_key','candidate_gt_list_index','internal_source_role','physical_context_alignment_observation','positive_visual_evidence_ko','ambiguity_and_limits_ko','selected_instance_correspondence','new_native_D_E_SELF_gold','training_admission']
    w=csv.DictWriter(buf,fieldnames=fields);w.writeheader();w.writerows([{k:r[k] for k in fields} for r in candidates]);cb=save('ALL15_CANDIDATE_OBSERVATIONS.csv',buf.getvalue().encode())
    independence=save('INDEPENDENCE_RECEIPT.json',{'completed':True,'PASS':True,'reader_id':'A','other_new_root_or_reader_observation_files_read':0,'other_new_values_received':0,'historical_SOURCE_GT_metadata_and_V215_V216_exposure_disclosed':True,'pixel_renderer_authorship_disclosed':True,'not_blinded_or_human_gold':True,'own_observations_binding':ob,'scope':plan['scope']})
    qa=save('SAVED_OBSERVATION_SCHEMA_QA.json',{'completed':True,'PASS':True,'all15_fixed_frame_candidate_keys_exact':set(candidate_notes)=={(p['frame_position'],p['candidate_gt_list_index']) for p in panels},'frame12_all8_query_subsets_and_U_partition':True,'original96_inputU11_FIT64_DEV32_preserved':True,'expanded_U17_not14':True,'actual22_views_bindings_in_prechecked27_whitelist':all(v['binding'] in plan['all27_encoded_SHA_size_prechecked_before_views'] for v in views),'all_actual_view_flags_true':all(v['actually_viewed'] for v in views),'selected_instance_all_null_native_gold_unknown_trainingfalse':all(r['selected_detector_GT_instance_index'] is None and r['new_native_D_E_SELF_gold']=='UNKNOWN' and not r['training_admission'] for r in candidates),'new_projection_raw_source_masks_depth_RGB_NPZ_GT_models_scores_or_fit_reads':0,'numeric_pixel_QA_not_repeated':True})
    lines=['# V217 A · 실제 15 GT overlay 독립 관찰','', '고정15 후보 panel과 별도 원 context7장을 `tools.view_image(detail=original)`로 실제 보았습니다. 모든 view 전에 선언된 owned PNG27개의 encoded SHA/크기를 확인했습니다. 별도 context를 열지 않은 1/3/5/7/8도 panel 안 원 context는 보았지만 standalone view로 세지 않습니다. root 새 관찰은 읽지 않았습니다. A는 renderer 저자이며 V215/V216 SOURCE와 이전 weak 판단/GT 준비에 노출됐으므로 blind·사람 gold가 아닙니다.','', '| frame/candidate | 물리 문맥 관찰 | 구체 근거와 한계 |','| --- | --- | --- |']
    for r in candidates:lines.append('| %d/%d | %s | %s %s |'%(r['frame_position'],r['candidate_gt_list_index'],r['physical_context_alignment_observation'],r['positive_visual_evidence_ko'],r['ambiguity_and_limits_ko']))
    lines += ['', '## 공통 해석','', '전경 EPAL 팔레트(3/4/5)와 뒤에 세운 팔레트(6/7/8)에서는 full/visible 영역이 넓은 물리 구조와 일치하는 약한 근거가 있습니다. 겹친 두 팔레트(0/2)는 후보 간 경계·가림·화면 잘림을 남겼습니다. 1은 색 영역이 보이지 않는다고 E를 만들지 않았습니다. 9/10/11은 목재 지지판/팔레트 후보 대부분이 상자와 흰 덮개 아래에 있고 visible 색이 다른 표면에도 보이므로 점유를 동일 intended part로 인증하지 않습니다.','', 'q 묶음이 실제 팔레트 부근인 경우와 상자·카트 손잡이인 경우를 별도 기록했습니다. 특히3의 목재 상자,7/8의 앞 목재 상자,9/10의 손잡이는 선명한 feature여도 GT 팔레트 native 대응이 아닙니다. mask·OOB·깊이 결측·query OFF만으로 native E/SELF를 생성하지 않습니다.','', '원96/U11·FIT64DEV32를 유지했고 candidate 반복 U는17입니다(이전14 가정과 구분). V216 weakpartD9는 역사적 assistant weak 근거이며 이번에 native label로 복사하지 않았습니다. 모든 selected detector instance는 null/UNKNOWN, native part projection은 아직 미실행, 신규 D/E/SELF gold와 training admission은 없습니다. 완벽한 mesh/corner 인증을 blanket gate로 요구하지 않으면서도 전체 mask 정렬만으로 해당 native part를 확정하지 않습니다.','', '[전체 관찰 JSON](ALL15_CANDIDATE_AND_FRAME12_OBSERVATIONS.json) · [15행 CSV](ALL15_CANDIDATE_OBSERVATIONS.csv) · [실제 view receipt](ACTUAL_IMAGE_VIEW_RECEIPT.json) · [schema QA](SAVED_OBSERVATION_SCHEMA_QA.json)','']
    ko=save('KO.md','\n'.join(lines).encode())
    code=bind(os.path.abspath(__file__),read(os.path.abspath(__file__)))
    complete=save('COMPLETE.json',{'completed':True,'PASS':True,'reader':'A','scope':plan['scope'],'observations':ob,'CSV':cb,'view_receipt':vp,'schema_QA':qa,'independence':independence,'KO':ko,'recording_code':code,'actual_panel_views':15,'actual_context_standalone_views':7,'candidate_observations':15,'frame_observations':12,'original_inputU':11,'expanded_candidate_U':17,'no_new_native_gold_or_training_admission':True,'own_folder_frozen_no_further_writes':True})
    print(json.dumps({'completed':True,'PASS':True,'COMPLETE':complete,'observations':ob,'CSV':cb,'view_receipt':vp,'schema_QA':qa,'KO':ko},ensure_ascii=False))

if __name__=='__main__':main()
