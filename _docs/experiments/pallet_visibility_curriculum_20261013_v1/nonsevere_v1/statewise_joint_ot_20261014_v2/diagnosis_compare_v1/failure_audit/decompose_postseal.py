"""Supplement the sealed read-only audit with constraint and target checks."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[7]
sys.path.insert(0,str(ROOT))
import json
import re
import xml.etree.ElementTree as ET
import numpy as np
from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C

P=Path(__file__).resolve().parent
audit=C.read(P/'FAILURE_AUDIT.json')
score=C.read(P.parent.parent/'evaluation_v3/POSTSEAL_SCORING.json')
sc={r['id']:r for r in score['rows']}
parts={}
for part in ('CAL','TEST'):
    rows=[r for r in audit['rows'] if r['partition']==part and r['cohort']=='PRIMARY_NONSEVERE']
    compatible=[r for r in rows if r['alignment']['compatible']]
    pixels=[m for r in compatible for m in r['BI_measurements'] if m['human_source_state']=='DIRECT_VISIBLE']
    errors=[]
    integer_count=0
    gt_count=0
    for r in rows:
        projection=np.asarray(r['alignment']['reference_native_projection'])
        s=sc[r['id']]
        for k,(human,valid,xy) in enumerate(zip(s['human_states_native'],s['native_gt_valid'],s['native_gt'])):
            if human=='DIRECT_VISIBLE' and valid:
                xy=np.asarray(xy)
                errors.append(float(np.linalg.norm(projection[k]-xy)))
                integer_count+=int(np.max(abs(xy-np.round(xy)))<1e-8)
                gt_count+=1
    subsets={}
    for is_compatible in (True,False):
        group=[r for r in rows if r['alignment']['compatible']==is_compatible]
        arms=('ORIGINAL_PNP','FROZEN_BI_ONLY_PNP','ALL_OT_JOINT_SE3')
        subsets[str(is_compatible)]=dict(frames=len(group),translation_mean_cm={arm:float(np.mean([r['metrics'][arm]['translation_cm'] for r in group])) for arm in arms},success_count={arm:sum(r['metrics'][arm]['success_5cm_5deg'] for r in group) for arm in arms})
    parts[part]=dict(compatible_frames=len(compatible),
        reference_BI_budget_all_pass=sum(all(m['registered_reference_pose_support']['within_3sigma'] for m in r['BI_measurements']) for r in compatible),
        reference_BI_geometry_all_pass=sum(all(m['registered_reference_pose_support']['geometric_visibility_pass'] for m in r['BI_measurements']) for r in compatible),
        reference_BI_canvas_all_pass=sum(all(m['registered_reference_pose_support']['inside_closed_canvas'] for m in r['BI_measurements']) for r in compatible),
        eligible_DIRECT_human_coordinate_inside_BI_budget=sum(m['annotation_measurement_budget']['within_3sigma'] for m in pixels),eligible_DIRECT_BI_count=len(pixels),
        all_source_DIRECT_annotations=gt_count,integer_coordinate_annotation_count=integer_count,
        reference_projection_vs_human_DIRECT_px=dict(n=len(errors),mean=float(np.mean(errors)),median=float(np.median(errors)),P90=float(np.quantile(errors,.9)),max=max(errors),above_round_to_integer_only_bound_sqrt_half=sum(v>np.sqrt(.5) for v in errors)),
        fallback_frames=sum(r['final_fallback'] for r in rows),final_equals_initial_pose=sum(r['final_equals_initial_pose'] for r in rows),final_equals_BI_only_pose=sum(r['final_equals_BI_only_pose'] for r in rows),
        false_S_source_nonself_frames=[dict(id=r['id'],native_ids=r['semantic_mismatches']['false_S_source_nonself']) for r in rows if r['semantic_mismatches']['false_S_source_nonself']],
        source_model_incompatible_ids=[r['id'] for r in rows if not r['alignment']['compatible']],compatible_vs_incompatible_pose_summary=subsets)
    for r in compatible:
        assert r['alignment']['mapping_projection_max_gap_px']<1e-9
        assert min(r['alignment']['reference_depth'])>1e-5
for r in audit['rows']:
    assert r['alignment']['compatible']==sc[r['id']]['compatible_pixel_model']
seal=C.read(P/'DIAGNOSIS_SEAL.json')
assert C.sha(P/'FAILURE_AUDIT.json')==seal['audit_sha256']
assert C.sha(P/'PROTOCOL.json')==seal['protocol_sha256']
assert all(C.sha(ROOT/path)==digest for path,digest in C.read(P/'PROTOCOL.json')['source_sha256'].items())
for file in P.glob('*.svg'):
    ET.parse(file)
links=re.findall(r'(?:href|src)="([^"]+)"',(P/'index.html').read_text())
assert all((P/link).exists() for link in links)
C.write(P/'CONSTRAINT_DECOMPOSITION.json',dict(schema='POSTSEAL_CONSTRAINT_DECOMPOSITION_V1',parts=parts,
    note='B/I budget associations do not prove causality. All compatible reference depths exceed frozen positive_depth=1e-5. Integer rounding bound applies only if annotation rounds the same ideal projected corner; manual annotation/model errors remain possible. Main numeric audit tolerance=1e-6, frozen solver feasibility tolerance=1e-7; exclusion margins in the cited failures are far larger.'))
checks=dict(schema='POSTSEAL_DIAGNOSIS_VALIDATION_V1',all98_model_compatibility_flags_match_sealed_score=True,compatible_reference_projection_mapping_gap_below_1e_minus9_px=True,all_compatible_reference_depths_above_frozen_1e_minus5_m=True,all_source_hashes_unchanged=seal['frozen_inference_sources_unchanged'],all_figure_links_exist=True,SVG_XML_valid=True,audit_seal_hash_unchanged=True,no_training_no_tuning_no_new_PnP=True)
C.write(P/'VALIDATION.json',checks)
lines=['# 제약·reference target 교차 확인','',
    '봉인된 FAILURE_AUDIT.json을 유지하고 세부 집계를 추가했습니다. 98개 프레임의 모델 호환 판정은 기존 봉인 score와 모두 같으며, 호환 reference의 native projection은 고정 body reference/mapping 투영과 1e-9 px 이내로 일치합니다. 현재 source 19개 hash가 그대로입니다.', '',
    '| 분할 | 모델 호환 | reference BI budget 통과 | reference BI geometry 통과 | reference BI canvas 통과 | 실제 DIRECT 좌표가 BI budget 안 |',
    '|---|---:|---:|---:|---:|---:|']
for part,v in parts.items():
    lines.append(f"| {part} | {v['compatible_frames']} | {v['reference_BI_budget_all_pass']} | {v['reference_BI_geometry_all_pass']} | {v['reference_BI_canvas_all_pass']} | {v['eligible_DIRECT_human_coordinate_inside_BI_budget']}/{v['eligible_DIRECT_BI_count']} |")
lines+=['','Reference pose의 배제는 주로 B/I의 작은 covariance budget에서 발생합니다. Source human DIRECT 주석도 B/I budget 밖인 경우가 많습니다. 다만 reference pose는 독립 물리 ground truth가 아니므로 이 표만으로 실제 물리 정답을 배제했다고 단정할 수 없습니다.', '',
    '| 분할 | DIRECT 주석 / 정수 좌표 | 주석과 derived reference projection 평균 / 최대 px |', '|---|---:|---:|']
for part,v in parts.items():
    z=v['reference_projection_vs_human_DIRECT_px']
    lines.append(f"| {part} | {v['all_source_DIRECT_annotations']} / {v['integer_coordinate_annotation_count']} | {z['mean']:.3f} / {z['max']:.3f} |")
lines+=['','같은 이상적 투영을 가장 가까운 정수 픽셀로만 반올림한다면 이동 상한은 약 0.707 px입니다. 관측 불일치가 이보다 크다고 해도 annotation 실수, rigid cuboid의 실제 모서리 차이, pose fit 잔차, 카메라/등록/phase 문제 중 어느 원인인지는 이번 진단으로 확정하지 않았습니다. 하나의 rigid pose가 모든 주석에 정확히 맞지 않을 가능성을 구분해야 하며, 새 GT pose fit이나 재인덱싱을 하지 않았습니다.', '',
    '| TEST subset | N | 원본 / current 평균 cm | 원본 / current 성공 |', '|---|---:|---:|---:|']
for name,group in parts['TEST']['compatible_vs_incompatible_pose_summary'].items():
    t=group['translation_mean_cm'];s=group['success_count']
    lines.append(f"| 모델 대응 호환={name} | {group['frames']} | {t['ORIGINAL_PNP']:.3f} / {t['ALL_OT_JOINT_SE3']:.3f} | {s['ORIGINAL_PNP']} / {s['ALL_OT_JOINT_SE3']} |")
lines+=['','호환 subset에서는 평균 translation이 조금 줄었지만 success는 감소했습니다. 이 subset 분석은 사후 선택된 진단이며 full TEST의 열등 결과를 대체하는 superiority 근거가 아닙니다.', '',
    '[세부 JSON](CONSTRAINT_DECOMPOSITION.json) · [검증](VALIDATION.json) · [원래 봉인 진단](README_KO.md)']
(P/'CONSTRAINT_CHECKS_KO.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(C.finite(dict(parts=parts,validation=checks)),ensure_ascii=False))
