#!/usr/bin/env python3
"""V216 sealed metadata -> four exclusive Korean reports. No image/GT/model read."""
import collections
import csv
import hashlib
import html
import io
import json
import os
import re
import stat
from html.parser import HTMLParser

D = '/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_public_train_part_query_support_contract_20261025_v216'
OUT = D + '/report_preparation_a'
INPUTS = {
 'scope': ('ROOT_PROSPECTIVE_SOURCE_PART_QUERY_SUPPORT_SCOPE.json', 'a714154e8c08020423c2dd2033dc3676708fe202ba149567cadbf62b9fe3804b'),
 'join_complete': ('joined_part_query_support_v2/COMPLETE.json', '5389f830f72c92f8269285bdf7d8076bd09d9bcb450bd475c27e04430eac5bb4'),
 'joined_rows': ('joined_part_query_support_v2/ALL96_JOINED_CANDIDATES.json', 'ffebab4da32966ec1ae5299e6ecdd3a8ba66aaad7b03388c3ef2c52020f60675'),
 'raw_summary': ('joined_part_query_support_v2/SUMMARY.json', '0d7ba94d61b1e9557205f631869aeeccb53c2232637bd7beb5b0bc0c34a5dc80'),
 'exact_OR': ('joined_part_query_support_v2/ALL96_EXACT_WITNESS_MASK_SIDECAR.json', 'cfe38fbefb330364c9b44c730ba7951f7b66fa5d654c46f9863a0d3c0e9d1799'),
 'root_semantic': ('ROOT_ACTUAL_ALL12_VISUAL_AND_SEMANTIC_QA.json', 'f91d9a1659df0f94664dae1da1934362dafbcb952591023ab4c212b82f1d04f7'),
 'root_semantic_csv': ('ROOT_ALL96_SEMANTIC_REVIEW.csv', '902d5aa09904569d25c27870bc835725daa859924aad445b1cb8e92eb10d4c9a'),
 'independent_join_QA': ('independent_saved_join_review_c/REVIEW.json', '3383ccb340095a4afac83c6eebbdec86b5b3087533edd153afaf43c11b8a4bf3'),
 'contract_ready': ('contract_preparation_a/FINAL_READY.json', 'b2130d4a3dc018d304556af28fa8b08556e813cdb1738c0004ea0b5582fccb99'),
 'split_ready': ('component_split_preparation_b/FINAL_READY.json', '2dcb8ce89caa7396dfb11ddca1de7fa9199fec81b7e2dacc1a90e657c72d36d4'),
 'join_ready': ('join_preparation_b_v2/FINAL_READY.json', 'f3ff8cef84352efe95b811823194018717214a10423b863ed450a193ad273957'),
 'repair_history': ('join_preparation_b_v2/V1_PRESERVATION_AND_MINIMAL_BINDING_REPAIR.json', 'a48604767aa6674949bda5a089436f20ac7ca5f90b30bab9dac7d62b7002c286'),
 'GT_text_readiness': ('public_GT_text_readiness_a/REVIEW.json', '66e94d2679e3323e3acfab84f6a8668a764d46506801ae6e43794e2bf953ff97'),
 'GT_text_KO': ('public_GT_text_readiness_a/KO.md', '4e0b1457c3be88d023d696103c934d5c31581492ff6383edfb07d4b661814b70'),
}

def opened(path, flags=os.O_RDONLY):
    if 'v112' in path.lower():
        raise ValueError('Forbidden lexical path before filesystem access')
    if not os.path.isabs(path) or os.path.normpath(path) != path:
        raise ValueError('Require normalized absolute path')
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for c in path.strip('/').split('/')[:-1]:
            nxt = os.open(c, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nxt
        result = os.open(path.rsplit('/', 1)[1], flags | os.O_NOFOLLOW, 0o644, dir_fd=fd)
        if not stat.S_ISREG(os.fstat(result).st_mode):
            os.close(result)
            raise ValueError('Require regular file')
        return result
    finally:
        os.close(fd)

def read(path):
    with os.fdopen(opened(path), 'rb') as f:
        return f.read()

def sha(b):
    return hashlib.sha256(b).hexdigest()

def bind(path, b):
    return {'path': path, 'sha256': sha(b), 'size_bytes': len(b)}

def write(path, b):
    with os.fdopen(opened(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL), 'wb') as f:
        f.write(b)
    return bind(path, b)

def rel(path):
    if 'v112' in path.lower():
        raise ValueError('Forbidden link')
    return os.path.relpath(path, D)

def mdlink(label, path):
    return '[' + label + '](' + rel(path) + ')'

def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |'] +
                     ['| ' + ' | '.join(str(x).replace('|', '\\|') for x in row) + ' |' for row in rows])

def htable(headers, rows):
    return '<div class="tablewrap"><table><thead><tr>' + ''.join('<th>'+html.escape(x)+'</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>'+html.escape(str(v))+'</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'

def main():
    B, J = {}, {}
    for key, (p, expected) in INPUTS.items():
        raw = read(D+'/'+p)
        assert sha(raw) == expected, ('Input binding changed', key)
        B[key] = bind(D+'/'+p, raw)
        J[key] = json.loads(raw) if p.endswith('.json') else raw.decode('utf-8')
    joint = J['joined_rows']['rows96']
    semantic = J['root_semantic']['rows96']
    masks = J['exact_OR']['rows96']
    S = {r['row_id']:r for r in semantic}
    M = {r['row_id']:r for r in masks}
    assert len(joint) == len(S) == len(M) == 96
    assert J['join_complete']['completed'] and J['join_complete']['PASS']
    assert J['independent_join_QA']['completed'] and J['independent_join_QA']['PASS']
    assert J['independent_join_QA']['join_complete']['sha256'] == B['join_complete']['sha256']
    assert J['root_semantic']['sealed_raw96']['sha256'] == B['joined_rows']['sha256']
    assert J['repair_history']['v1_runtime_execution_count'] == 0
    assert J['repair_history']['v1_preserved_in_place']
    csv_rows = list(csv.DictReader(io.StringIO(J['root_semantic_csv'])))
    assert len(csv_rows) == 96
    for a,b in zip(csv_rows, semantic):
        assert all(a[k] == str(v) for k,v in b.items()), ('CSV differs', b['row_id'])
    for r in joint:
        i=r['input']; s=S[i['row_id']]
        assert i['native_slot']==s['native_slot'] and i['internal_source_role']==s['internal_source_role']
        assert (not i['original_query_available'])==s['original_inputU']
        assert i['role']==s['original_role']=='TRAIN'
        assert not s['primary_DE_training_admission'] and not s['auxiliary_training_admission']
        assert s['physical_native_point_visibility_gold']=='UNKNOWN'
        assert s['root_weak_part_state'] in ('D','UNKNOWN')
        assert s['raw_native_weak_candidate'] in ('D','UNKNOWN')
        assert s['raw_native_weak_candidate']==r['raw_helper_candidate']['native_weak_visibility_agreement_candidate']
        assert s['raw_spatial_witness_candidate']==r['raw_helper_candidate']['spatial_witness_agreement_candidate']
        assert s['raw_query_support_agreement']==r['raw_helper_candidate']['query_support_agreement']
        assert M[i['row_id']]['original_A_C_ROIs_individually_preserved']

    def aggregate(rows):
        ss=[S[r['input']['row_id']] for r in rows]
        mm=[M[r['input']['row_id']] for r in rows]
        return {
         'N':len(rows),'frames':len({r['input']['position'] for r in rows}),
         'metadata_components':len({r['input']['component_id'] for r in rows}),
         'original_inputU':sum(s['original_inputU'] for s in ss),
         'reader_A_D':sum(r['reader_A_axes']['visibility']['state']=='D' for r in rows),
         'reader_C_D':sum(r['reader_C_axes']['visibility']['state']=='D' for r in rows),
         'raw_native_weak_D':sum(s['raw_native_weak_candidate']=='D' for s in ss),
         'raw_spatial_witness_candidates':sum(s['raw_spatial_witness_candidate'] for s in ss),
         'root_weak_same_pallet_part_D':sum(s['root_weak_part_state']=='D' for s in ss),
         'root_native_UNRESOLVED':sum(s['root_weak_part_state']=='UNKNOWN' for s in ss),
         'native_external_E_confirmed':0,'native_SELF_confirmed':0,
         'query_support_agreement':dict(collections.Counter(s['raw_query_support_agreement'] for s in ss)),
         'exact_OR_validity':{
          'valid':sum(m['candidate_union_all_valid96'] is True for m in mm),
          'invalid_preserved':sum(m['candidate_union_all_valid96'] is False for m in mm),
          'no_ROI':sum(m['candidate_union_all_valid96'] is None for m in mm)},
         'root_review_status':dict(collections.Counter(s['root_semantic_review_status'] for s in ss)),
        }
    role_sets={'ALL':joint}
    for role in ['SOURCE_FIT','SOURCE_DEV']:
        role_sets[role]=[r for r in joint if r['input']['internal_source_role']==role]
    counts={role:aggregate(rows) for role,rows in role_sets.items()}
    native_counts={role:{str(n):aggregate([r for r in rows if r['input']['native_slot']==n]) for n in range(8)} for role,rows in role_sets.items()}
    a=counts['ALL']
    assert [a[k] for k in ['N','original_inputU','raw_native_weak_D','raw_spatial_witness_candidates','root_weak_same_pallet_part_D','root_native_UNRESOLVED']]==[96,11,11,62,9,87]
    assert a['query_support_agreement']=={'UNKNOWN':44,'OFF':11,'ON':24,'NEAR':17}
    assert a['exact_OR_validity']=={'valid':79,'invalid_preserved':4,'no_ROI':13}
    assert [counts[x]['root_weak_same_pallet_part_D'] for x in ['SOURCE_FIT','SOURCE_DEV']]==[7,2]
    for role,c in counts.items():
        saved=J['raw_summary']['counts'][role]
        assert [c['N'],c['frames'],c['original_inputU'],c['raw_native_weak_D'],c['raw_spatial_witness_candidates']]==[saved['N'],saved['frames'],saved['original_inputU'],saved['join']['visibility_candidate'].get('D',0),saved['local_witness_candidates']]
        assert c['query_support_agreement']==saved['join']['query_support_agreement']
    assert counts['SOURCE_FIT']['N']==64 and counts['SOURCE_DEV']['N']==32
    assert counts['SOURCE_FIT']['original_inputU']==9 and counts['SOURCE_DEV']['original_inputU']==2
    withheld=[s for s in semantic if s['root_semantic_review_status']=='RAW_D_WITHHELD_SAME_FEATURE_UNRESOLVED']
    assert {(s['frame_position'],s['native_slot']) for s in withheld}=={(4,3),(6,7)}
    frames=[]
    for view in J['root_semantic']['all12_actual_viewed']:
        assert view['actually_viewed']
        p=view['frame_position']
        rr=[r for r in joint if r['input']['position']==p]
        assert len(rr)==8 and sorted(r['input']['native_slot'] for r in rr)==list(range(8))
        frames.append({'position':p,'frame_key':view['frame_key'],'internal_source_role':view['internal_source_role'],
                       'overlay':view['actual_overlay_view'],'root_observation_ko':view['manual_root_observation_ko'],
                       'counts':aggregate(rr)})
    assert len(frames)==12

    role_headers=['범위','프레임/행','원 U','원시 D 후보','의미 검토 후 weak 부위 D','UNKNOWN','공간 witness 후보']
    role_rows=[[k,str(c['frames'])+'/'+str(c['N']),c['original_inputU'],c['raw_native_weak_D'],c['root_weak_same_pallet_part_D'],c['root_native_UNRESOLVED'],c['raw_spatial_witness_candidates']] for k,c in counts.items()]
    native_headers=['native','N','원 U','공간 witness','원시 D 후보','검토 후 weak 부위 D','UNKNOWN','ON/NEAR/OFF/UNKNOWN']
    def native_row(n,c):
        q=c['query_support_agreement']
        return [n,c['N'],c['original_inputU'],c['raw_spatial_witness_candidates'],c['raw_native_weak_D'],c['root_weak_same_pallet_part_D'],c['root_native_UNRESOLVED'],'/'.join(str(q.get(x,0)) for x in ['ON','NEAR','OFF','UNKNOWN'])]
    native_rows=[native_row(n,native_counts['ALL'][str(n)]) for n in range(8)]
    detail_headers=['내부 역할','native','N','원 U','원시 D 후보','검토 후 weak 부위 D','UNKNOWN','공간 witness']
    detail_rows=[]
    for role in ['SOURCE_FIT','SOURCE_DEV']:
        for n in range(8):
            c=native_counts[role][str(n)]
            detail_rows.append([role,n,c['N'],c['original_inputU'],c['raw_native_weak_D'],c['root_weak_same_pallet_part_D'],c['root_native_UNRESOLVED'],c['raw_spatial_witness_candidates']])
    q_headers=['범위','ON','NEAR','OFF','UNKNOWN','ROI 유효/실패/없음']
    q_rows=[]
    for role,c in counts.items():
        q=c['query_support_agreement'];v=c['exact_OR_validity']
        q_rows.append([role]+[q.get(x,0) for x in ['ON','NEAR','OFF','UNKNOWN']]+[str(v['valid'])+'/'+str(v['invalid_preserved'])+'/'+str(v['no_ROI'])])
    raw_seals=J['join_complete']['sealed_observations']
    links=[
     ('전체 96 원시 join','joined_part_query_support_v2/ALL96_JOINED_CANDIDATES.json'),
     ('전체 96 root 의미 검토 CSV','ROOT_ALL96_SEMANTIC_REVIEW.csv'),
     ('전체 12 overlay 갤러리','joined_part_query_support_v2/PORTABLE_GALLERY_KO.md'),
     ('원 역할/native/관찰자 수표','joined_part_query_support_v2/SOURCE_COUNTS_BY_ROLE_NATIVE_READER.csv'),
     ('모든 축 차이 CSV','joined_part_query_support_v2/ALL96_AXIS_DIFFERENCES.csv'),
     ('두 원 ROI와 exact OR sidecar','joined_part_query_support_v2/ALL96_EXACT_WITNESS_MASK_SIDECAR.json'),
     ('독립 저장 join QA','independent_saved_join_review_c/REVIEW.json'),
     ('root 실제 12+2 시각·의미 검토','ROOT_ACTUAL_ALL12_VISUAL_AND_SEMANTIC_QA.json'),
     ('annotation 계약','contract_preparation_a/ANNOTATION_AND_SUPERVISION_CANDIDATE_CONTRACT.json'),
     ('순수 helper','contract_preparation_a/part_query_support_contract.py'),
     ('내부 FIT/DEV 고정 역할','component_split_preparation_b/INTERNAL_SOURCE_FIT_DEV_COMPONENT_SPLIT.json'),
     ('실제 두 관찰자 진입 기록','ROOT_ACTUAL_SAMEFRAME_TWO_READER_ANNOTATION_ENTRY.json'),
     ('실제 V2 join 진입 기록','ROOT_ACTUAL_TWO_READER_JOIN_AND_OVERLAY_V2_ENTRY.json'),
     ('V1 보존·V2 최소 메타데이터 수리','join_preparation_b_v2/V1_PRESERVATION_AND_MINIMAL_BINDING_REPAIR.json'),
     ('V2 코드 준비·검사 바인딩','join_preparation_b_v2/FINAL_READY.json'),
     ('기존 공개 GT text 준비도','public_GT_text_readiness_a/KO.md'),
     ('보고서 재현 코드','report_preparation_a/build_saved_v216_reports.py'),
    ]
    for who,b in raw_seals.items():
        links.append((who+' 봉인 관찰 원문',rel(b['path'])))
    code_path=os.path.abspath(__file__)
    code_binding=bind(code_path,read(code_path))
    results={
     'completed':True,'PASS':True,'phase':'V216','report_scope':'SEALED_SOURCE_TRAIN_METADATA_ONLY_NO_NEW_IMAGE_OR_GT_READ',
     'headline_ko':'같은 12 SOURCE 프레임의 독립 관찰에서 원시 D 후보 11개를 보존하고, root 의미 검토 후 9개를 약한 동일 팔레트 부위 visible 근거로 남겼다. 원시 native-point gold·외부 가림 개선·학습 허용은 아니다.',
     'counts_by_role':counts,'counts_by_role_and_native':native_counts,
     'raw_reader_visibility_counts':{role:J['raw_summary']['counts'][role]['readers'] for role in role_sets},
     'all_axis_disagreements':{role:J['raw_summary']['counts'][role]['join']['categorical_axis_disagreements'] for role in role_sets},
     'frame_readout':frames,'withheld_raw_D_cases':withheld,
     'all96_root_semantic_rows':semantic,
     'authority':{'root_weak_same_pallet_part':'ASSISTANT_TWO_READER_AND_ROOT_SOURCE_SAME_PART_WEAK_NOT_NATIVE_POINT_GOLD','local_spatial_witness':'ROI_FEATURE_ONLY_NOT_AUTOMATIC_PALLET_OR_NATIVE_LABEL','physical_native_point_visibility_gold':'UNKNOWN','primary_native_E_SELF_confirmed':0,'primary_DE_training_admission':False,'auxiliary_training_admission':False},
     'invariants':{'original_outer_role_TRAIN':True,'original_native0_to7_qL_RAW96_U_preserved':True,'two_original_ROIs_and_exact_pixel_OR_preserved':True,'ROI_shrink_or_repair':False,'future_DEV_all_loss_branch_sampler_and_normalization_exclusion_required':True,'physical_independence_established':False,'appearance_previously_exposed_not_blind':True},
     'query_axis_note':'ON/NEAR/OFF/UNKNOWN describe the readers’ local ROI feature relation to original q, not native coordinate correctness. OFF/foreign surface/absence do not imply E.',
     'validity_note':'79 valid is in-valid96-region support for the exact ROI OR, not semantic same-feature agreement or COVER/SHAM alpha proof. Four failures and thirteen no-ROI rows remain unchanged.',
     'execution_history':{'v1_runtime_execution_count':0,'v1_runtime_failure':False,'v1_reason':'Pre-execution path+SHA binding shape adapter; optional size annotation differed. Original preparation retained unchanged.','v2_actual_execution_count':1,'v2_parent_reported_terminal':{'session':1832,'initial_chunk':'ff6de7','terminal_chunk':'2f4e5e','exit_code':0},'v2_complete':B['join_complete'],'root_semantic_parent_reported_terminal':{'chunk':'96bcee','exit_code':0},'scientific_annotation_changes_by_binding_repair':False},
     'GT_readiness':{'historical_retained_receipts_exist':True,'not_first_GT_use':'V154 instance/role-query weak bridge and V159 nominal-part/mask/owner diagnostics already exist.','current_GT_payload_access':0,'payload_current_reverification':False,'next_one_bounded_check':'Existing selected SOURCE TRAIN instance/native-part/query/crop correspondence versus retained full/visible masks and depth; keep q/L/U and no GT-best permutation/PnP/recentering.','current_teacher_or_training_admission':False,'readiness':B['GT_text_readiness']},
     'latest_real_benchmark':'V213 unchanged, historical reference only; no protected result body opened for this report.',
     'new_fit_forward_evaluation_model_or_threshold_change':False,
     'report_builder_access':{'source_GT_arrays_models_scores':0,'PNG_bytes_or_decode_or_toolview':0,'browser_render':0,'sealed_metadata_text_only':True},
     'evidence_bindings':B,'sealed_reader_observations':raw_seals,'builder':code_binding,
     'tables':{'roles':{'headers':role_headers,'rows':role_rows},'native_all':{'headers':native_headers,'rows':native_rows},'role_native':{'headers':detail_headers,'rows':detail_rows},'query_and_exact_OR':{'headers':q_headers,'rows':q_rows}},
     'phase_freeze_performed_by_this_builder':False,
    }
    md=['# V216 · 공개 SOURCE 부위와 query support를 분리한 관찰',
      '같은 12 SOURCE TRAIN 프레임의 원 96개 native 입력을 유지해 두 관찰자가 독립 판단하고, root가 같은 부위 근거를 다시 확인했습니다. **원시 D 후보 11개 중 9개를 약한 동일 팔레트 부위 visible 근거로 남겼습니다(7 SOURCE_FIT / 2 SOURCE_DEV). 나머지 87개는 UNKNOWN입니다.** 이는 물리 native-point gold나 새로운 학습 허용이 아닙니다. 외부 E와 SELF의 양성 확정은 각각 0개입니다.',
      '진전은 데이터·질문·감독 권위의 구분입니다. 새 모델·학습·추론·평가·문턱 변경은 없고 안정적인 외부 가림 판별 개선을 입증하지 않았습니다. 최신 실제 실사 비교 V213은 그대로입니다. 이 보고서 작성에서는 그 보호된 결과 본문을 열지 않았습니다.',
      '## 범위와 원시 합의 이후의 의미 검토',table(role_headers,role_rows),
      '원래 외부 역할은 전부 TRAIN입니다. 사전에 고정한 내부 SOURCE_FIT은 8프레임 64행(U9), SOURCE_DEV는 4프레임 32행(U2)입니다. 각 prefix에 2 FIT/1 DEV이며 4개 prefix를 유지했습니다. 메타데이터 component 12개가 물리적으로 독립된 12개 대상이라는 뜻은 아닙니다. 이미 appearance가 노출된 내부 개발 분할이며 새 blind 검증이나 기존 HOLD가 아닙니다. 미래에는 DEV 및 모든 파생 pair/variant를 모든 loss·sampler·정규화 분모에서 제외해야 합니다. **현재 primary·auxiliary training admission은 모두 false입니다.**',
      'A는 D14, C는 D17을 각각 독립 기록했고 raw helper의 D 합의는 11개였습니다. 원시 관찰값과 helper 결과를 수정하지 않았습니다. root는 아래 두 행에서 ROI가 같은 팔레트의 서로 다른 feature를 포함한다고 보아 weak D 후보만 보류했습니다. 완벽한 mesh/정확한 3D corner를 모든 weak 부위 관찰의 선결조건으로 요구한 것은 아닙니다.',
      '| 보류 행 | 서로 다른 feature 근거 | 기존 panel |\n| --- | --- | --- |\n| frame 4 / native 3 | EPAL 블록 면의 위쪽과 아래 코너·바닥 경계 | '+mdlink('원 CLEAN96',J['root_semantic']['additional_two_actual_clean96_views'][0]['binding']['path'])+' |\n| frame 6 / native 7 | 위 판 끝면과 아래 변·틈/다음 판 시작 | '+mdlink('원 CLEAN96',J['root_semantic']['additional_two_actual_clean96_views'][1]['binding']['path'])+' |',
      '## native별 완전 분모',table(native_headers,native_rows),
      '모든 native0..7의 12행을 보존합니다. 표의 D는 원 q/native·프레임 맥락에서 설명 가능한 **같은 팔레트 부위의 assistant weak visible 관찰**이며 정확한 native 3D점의 gold가 아닙니다. native1/2/3의 D0도 해당 물리 부위가 없거나 가려졌다는 음성 정답은 아닙니다.',
      '## 공간 witness와 query support는 다른 축',table(q_headers,q_rows),
      '**공간 witness 후보 62개는 팔레트 라벨 62개가 아닙니다.** 목재·플라스틱 상자의 모서리, 카트 손잡이도 포함합니다. 원 q/native만으로 의도 팔레트 부위를 주소화하지 못하고 explicit ROI가 추가로 필요한 관찰은 LOCAL_FEATURE_ONLY로 보존하며 unconditioned native head에 복사하지 않습니다. ROI 종류·겹침이나 두 reader의 동의만으로 같은 intended part/instance가 증명되지는 않습니다.',
      'ON24/NEAR17/OFF11/UNKNOWN44는 **관찰 ROI feature에 대한 원 query의 관계**입니다. native 좌표 정확도·가시성 정답이 아닙니다. OFF, 바닥/배경, 다른 물체가 q에 있다는 이유로 E를 만들지 않았습니다. E에는 의도 부위를 실제로 가리는 외부 전경의 양성 근거가 필요하고 SELF와도 구분합니다.',
      'ROI는 원96 좌표의 inclusive 정수 영역입니다. A와 C의 원 ROI를 모두 보존하고 **정확한 픽셀 OR**만 기록했습니다. display용 enclosing bbox는 alpha/proof mask가 아닙니다. 79개는 원 valid96 내부, 4개는 실패, 13개는 ROI 없음으로 남았습니다. 실패 ROI를 줄이거나 이동시키지 않았고, valid는 동일 semantic feature나 실제 COVER/SHAM proof를 인증하지 않습니다.',
      '## 실제 시각 근거를 읽는 방법',
      '두 관찰자는 봉인 전 상대방 새 값을 보지 않고 같은 12프레임의 full context/all8 contact를 실제 확인했습니다. root는 저장된 12 overlay와 위의 추가 clean96 두 장을 실제 보았습니다. 아래 설명은 그 봉인된 시각 receipt에 근거합니다. 이번 집계 builder의 새 PNG 열람/렌더는 0입니다.',
    ]
    case_text={0:'n6은 같은 팔레트 상판 끝 윤곽의 weak 근거가 남습니다. n5의 판 끝/판 면 선택 차이와 원 U0..3은 그대로 보존했습니다.',6:'뒤의 세운 팔레트와 앞 목재 상자가 함께 있습니다. n4/5/6의 팔레트 판 끝 근거와 n0..3의 다른 상자 feature를 분리했고 n7은 서로 다른 feature 범위로 보류했습니다.',10:'q8이 카트 손잡이에 놓였습니다. 선명한 edge와 query ON 합의가 있어도 pallet native D/E로 바꿀 수 없음을 보여줍니다.'}
    for pos in [0,6,10]:
        f=frames[pos]
        md += ['### 고정 frame '+str(pos)+' · '+f['frame_key'],case_text[pos],mdlink('두 원 ROI overlay 보기',f['overlay']['path'])]
    md += [mdlink('전체 고정 12프레임 갤러리',D+'/joined_part_query_support_v2/PORTABLE_GALLERY_KO.md'),
      '## 내부 역할 × native',table(detail_headers,detail_rows),
      '## 기존 공개 GT 준비도와 다음 한 가지 점검',
      '공개 SOURCE GT가 없다는 전제는 철회합니다. V153의 과거 보존 영수증은 RGB/depth와 전체·가시 마스크, scene GT/camera 관련 자료를 기록합니다. 입력 전용 manifest에서 GT를 뺀 것은 feature 계약이며 미보유를 뜻하지 않습니다. V154의 instance/role-query weak bridge, V159의 nominal-part·mask 진단 및 owner bridge도 이미 있으므로 첫 GT 활용으로 재포장하지 않습니다. **보존은 역사적 영수증 근거이며 이번 phase에서 GT 본체·현재 payload 존재를 재확인한 것은 아닙니다. 실제 GT 값 접근은 0입니다.**',
      '다음 한 가지는 기존 고정 SOURCE TRAIN의 선택 인스턴스/native 부위/query/crop 대응을 retained 전체·가시 마스크와 depth 계약에 대조하는 bounded audit입니다. 먼저 정확한 TRAIN whitelist를 고정하고 별도 실제 접근 범위를 둡니다. 원 q/L/native/U, 순열 방향을 유지하고 GT-best 대응·PnP·q 재중심화는 하지 않습니다. 마스크 음성/타 물체/부위 불명확을 자동 E로 바꾸지 않으며 완벽한 surface gold를 blanket gate로 삼지도 않습니다. '+mdlink('공개 GT text 준비도와 증거',D+'/public_GT_text_readiness_a/KO.md'),
      '## 보존 이력·재현·한계',
      'join V1은 실행되지 않았습니다. 실행 전 artifact binding의 optional size 유무 때문에 whole-dict 비교가 부적합함을 발견하여 V1 원문을 보존하고 V2의 path+SHA 정규화 어댑터만 수리했습니다. 기존 수치·ROI·관찰 판단은 바꾸지 않았습니다. V2 join은 한 번 실제 실행되어 exit0으로 완료됐고 독립 저장 검산을 통과했습니다. 보존된 4 invalid ROI는 이 메타데이터 수리와 별개인 실제 실패 상태입니다.',
      '재현은 아래 봉인 코드와 exact entry를 사용합니다. 출력 경로는 exclusive이므로 기존 산출물 위로 재실행하지 않습니다. 이미 완료된 annotation/join을 다시 실행할 필요는 없습니다.',
      '```bash\npython -B join_preparation_b_v2/join_and_render_saved_two_reader_observations_v2.py --entry ROOT_ACTUAL_TWO_READER_JOIN_AND_OVERLAY_V2_ENTRY.json\npython -B report_preparation_a/build_saved_v216_reports.py\n```',
      '이 phase는 assistant의 약한 SOURCE 부위/feature 관찰입니다. 사람 gold, 물리 native-point gold, 자연 외부 가림의 완전 정답 집합, 무작위 대표 표본, 인과적 모델 개선 증거가 아닙니다. 모든 U11과 UNKNOWN을 남겼고 새 fit/admission/paired controls/model/threshold/PnP 작업을 하지 않았습니다. 미래 loss/입력 계약은 실제로 주소화한 weak 질문과 맞아야 합니다.',
      '## 원문과 실행 근거',
    ]
    md += ['- '+mdlink(label,D+'/'+p) for label,p in links]
    md += ['- [기계가 읽는 결과와 전체 수표](ROOT_RESULTS_KO.json)','- [HTML 읽기 화면](ROOT_READOUT_KO.html)','']
    markdown='\n\n'.join(md).encode('utf-8')

    cards=''
    for pos in [0,6,10]:
        f=frames[pos]
        cards+='<section class="card"><h3>고정 frame '+str(pos)+' · '+html.escape(f['frame_key'])+'</h3><p>'+html.escape(case_text[pos])+'</p><a href="'+rel(f['overlay']['path'])+'"><img loading="lazy" src="'+rel(f['overlay']['path'])+'" alt="고정 frame '+str(pos)+' 원 두-reader ROI overlay"></a></section>'
    wh=''.join('<a class="card small" href="'+rel(v['binding']['path'])+'"><img src="'+rel(v['binding']['path'])+'" alt="보류 원 CLEAN96"><p>frame '+str(v['frame_position'])+' / native '+str(v['native_slot'])+' · 서로 다른 feature 범위, 원시 D 보류</p></a>' for v in J['root_semantic']['additional_two_actual_clean96_views'])
    linkhtml='<ul>'+''.join('<li><a href="'+p+'">'+html.escape(label)+'</a></li>' for label,p in links)+'</ul>'
    htmltext='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>V216 SOURCE 부위·query support</title><style>body{font:16px/1.7 system-ui,sans-serif;background:#f6f7f9;color:#17232b;margin:0}main{max-width:1150px;margin:auto;padding:32px 22px}h1{line-height:1.25}h2{margin-top:38px}a{color:#155c91}table{border-collapse:collapse;width:100%;background:white;font-size:14px}th,td{border:1px solid #d8dfe4;padding:8px;text-align:left}th{background:#eaf0f4}.tablewrap{overflow-x:auto}.lead,.note,.card{background:white;border:1px solid #d8dfe4;border-radius:10px;padding:18px;margin:18px 0}.lead{border-left:6px solid #267c71}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:18px}.card img{width:100%;height:auto;display:block}.small img{width:192px;image-rendering:pixelated;max-width:100%;margin:auto}code,pre{background:#eaf0f4;padding:3px 5px}li{margin:5px 0}</style><main><h1>V216 · 같은 부위인가,<br>query는 무엇을 묻는가</h1><div class="lead"><strong>원시 D 후보 11 → 약한 동일 팔레트 부위 D 9</strong><p>7 SOURCE_FIT · 2 SOURCE_DEV / 나머지 87 UNKNOWN. 원 native 96행/U11을 모두 보존했습니다. 외부 E·SELF 양성 확정은 각각 0이며 physical native-point gold는 UNKNOWN, 모든 training admission은 false입니다.</p></div><p>진전은 데이터·타깃 정의의 구분입니다. 새 모델·fit·forward·평가·문턱 변경이 없고, 안정적인 외부 가림 성능 개선을 입증하지 않았습니다. 최신 실제 실사 V213은 그대로입니다.</p>'''
    htmltext+='<h2>완전 분모와 의미 검토</h2>'+htable(role_headers,role_rows)+'<p>A D14 / C D17에서 원시 D 합의는 11개였습니다. 원시 값은 수정하지 않고 root가 서로 다른 feature를 고른 2개만 보류했습니다. 같은 프레임의 두 reader 동의는 gold가 아닙니다. 완벽한 mesh/정확한 3D corner가 모든 weak 관찰의 선결조건도 아닙니다.</p><div class="grid">'+wh+'</div>'
    htmltext+='<h2>공간 feature와 native를 구분</h2><p><strong>공간 witness 62개는 pallet 라벨 62개가 아닙니다.</strong> 목재·플라스틱 상자와 카트 손잡이도 들어 있습니다. 원 q/native로 주소화하지 못하고 explicit ROI가 추가로 필요한 feature는 LOCAL_FEATURE_ONLY이며 unconditioned native head에 복사할 수 없습니다.</p>'+htable(q_headers,q_rows)+'<p>ON24/NEAR17/OFF11/UNKNOWN44는 ROI feature와 원 q의 관계입니다. native 좌표 정확도·가시성 정답이 아닙니다. query OFF, 다른 물체, 배경·빈 crop을 E로 만들지 않았습니다. 정확한 두 ROI 픽셀 OR를 보존했고 4 invalid를 줄이거나 이동하지 않았습니다. 79 valid도 semantic 동일 부위나 실제 alpha proof는 아닙니다.</p>'
    htmltext+='<h2>native0..7</h2>'+htable(native_headers,native_rows)+'<h2>고정 시각 근거</h2><p>root가 실제 확인한 12 overlay+추가 clean96 두 장의 receipt를 인용합니다. 이번 집계 코드의 새 이미지 decode/toolview/browser render는 0입니다. 원 q/L/U/RAW96는 그대로입니다.</p>'+cards
    htmltext+='<h2>내부 SOURCE_FIT / SOURCE_DEV</h2><p>원래 outer TRAIN은 바꾸지 않았습니다. 8 FIT 프레임/64행/U9와 4 DEV 프레임/32행/U2이며 4 prefix마다 2 FIT/1 DEV입니다. 메타데이터 component 12개가 물리적 독립성을 인증하지 않습니다. appearance는 이미 노출됐고 fresh blind/HOLD가 아닙니다. 미래 DEV와 파생 variant는 모든 loss·sampler·정규화에서 제외해야 합니다. 현재 모든 학습 admission은 false입니다.</p>'+htable(detail_headers,detail_rows)
    htmltext+='<h2>GT가 없었던 것이 아닙니다</h2><p>V153의 과거 영수증에는 공개 MR6D GT/depth/전체·가시 마스크 보존이 기록돼 있습니다. V154 instance/role-query bridge와 V159 nominal-part/mask/owner 진단도 이미 있으므로 첫 GT 활용으로 부르지 않습니다. 이번 phase 실제 GT 값 접근은 0이고 현재 payload 존재를 재확인하지 않았습니다.</p><p>다음 한 가지 점검은 고정 SOURCE TRAIN의 instance/native-part/query/crop 대응과 retained full/visible masks·depth 계약을 대조하는 bounded audit입니다. 정확 TRAIN whitelist·별도 실제 접근 범위가 먼저 필요합니다. q/L/U·순열을 유지하고 GT-best 대응/PnP/재중심화·mask 음성→E 자동변환을 하지 않습니다. <a href="public_GT_text_readiness_a/KO.md">GT text 준비도</a></p>'
    htmltext+='<h2>보존 이력과 재현</h2><p>V1 join은 미실행이며 그대로 보존했습니다. optional size가 다른 artifact binding을 whole-dict 비교하던 준비 어댑터만 V2 path+SHA identity로 수리했습니다. 원 판단·ROI·수치 로직은 그대로입니다. V2는 한 번 실행되어 exit0/독립 저장 QA PASS입니다.</p><pre>python -B join_preparation_b_v2/join_and_render_saved_two_reader_observations_v2.py --entry ROOT_ACTUAL_TWO_READER_JOIN_AND_OVERLAY_V2_ENTRY.json\npython -B report_preparation_a/build_saved_v216_reports.py</pre><p>exclusive 출력이므로 기존 경로에 재실행하지 않습니다. 새 SOURCE label/admission/paired controls/fit은 하지 않았습니다. 사람 gold·물리 native-point gold·자연 가림 완전 정답·인과적 모델 효과·전체 공개 데이터의 대표성을 주장하지 않습니다.</p><h2>원문·QA·코드</h2>'+linkhtml+'<p><a href="ROOT_RESULTS_KO.json">기계가 읽는 결과</a> · <a href="PORTABLE_REPORT_KO.md">Markdown 보고서</a></p></main></html>'
    outputs={
      'ROOT_RESULTS_KO.json':(json.dumps(results,ensure_ascii=False,indent=2)+'\n').encode(),
      'ROOT_REPORT_KO.md':markdown,'PORTABLE_REPORT_KO.md':markdown,
      'ROOT_READOUT_KO.html':htmltext.encode(),
    }
    # Preflight every intended link without reading image bytes or following symlinks.
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):
            self.links.extend(v for k,v in attrs if k in ('href','src') and v)
    parser=Links();parser.feed(htmltext)
    allrefs=re.findall(r'\]\(([^)]+)\)',markdown.decode())+parser.links
    targets=set()
    for link in allrefs:
        assert not link.startswith(('file:','/')) and 'v112' not in link.lower()
        if link.startswith(('https://','http://','#')):continue
        target=os.path.normpath(D+'/'+link.split('#')[0])
        assert 'v112' not in target.lower()
        if target not in {D+'/'+k for k in outputs}:
            f=opened(target);os.close(f)
        targets.add(target)
    generated={k:write(D+'/'+k,b) for k,b in outputs.items()}
    assert read(D+'/ROOT_REPORT_KO.md')==read(D+'/PORTABLE_REPORT_KO.md')
    for k,b in outputs.items():assert read(D+'/'+k)==b
    finish={
      'completed':True,'PASS':True,'FINAL_READY_FROZEN':True,'actual_build_completed':True,
      'builder':code_binding,'input_bindings':B,'outputs':generated,
      'checks':{'all96_join_root_CSV_exact':True,'role_native_tables_recomputed_and_raw_summary_exact':True,'raw11_vs_root9_separate':True,'U11_FIT64_DEV32_preserved':True,'exactOR79_4_13_preserved':True,'all_training_false_and_physical_point_gold_UNKNOWN':True,'MD_PORTABLE_byte_equal':True,'all_relative_links_regular_nofollow':True,'report_outputs_readback_exact':True},
      'local_link_references':len(allrefs),'local_unique_targets':len(targets),
      'report_only_new_image_GT_array_weight_model_score_reads':0,
      'browser_rendered':False,'whole_phase_frozen_by_this_builder':False,
      'own_report_files_frozen_no_further_writes':True,
    }
    final=write(OUT+'/FINAL_READY.json',(json.dumps(finish,ensure_ascii=False,indent=2)+'\n').encode())
    print(json.dumps({'completed':True,'PASS':True,'FINAL_READY':final,'outputs':generated,'role_counts':counts,'local_link_references':len(allrefs),'local_unique_targets':len(targets)},ensure_ascii=False))

if __name__=='__main__':
    main()
