"""Posthoc saved-metric pairing only; no images, reference data or pose calls."""
from pathlib import Path
import datetime
import gzip
import hashlib
import json
import math
import re

ROOT=Path('/home/minjae/Documents/github/pallet-pose')
C=ROOT/'_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1'
OLD=C/'posefix_sub_comparison_v1'
B=C/'statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1'
CURRENT=B/'clean_fresh_BI_pose_eval_v4_32_r2'
OUTPUT=Path(__file__).resolve().parent
METHODS=('NATIVE_BASE','INITIAL_N3_SUBPIX')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def relative(path):return str(path.relative_to(ROOT))


def run():
    if (OUTPUT/'PAIR_METRICS.json').exists():raise FileExistsError('Preserve original extraction')
    scored=OLD/'SCORED_ROWS.jsonl.gz'
    # Existing independent baseline recount bound these exact stored bytes.
    if sha(scored)!='73402dc1219890fdd49fbcc5c6b308cc8a0ea4b2a56ba1639c351b86b042197f':
        raise ValueError('Historical scored metric archive changed')
    comparison=read(OLD/'COMPARISON.json');prediction_seal=read(OLD/'PREDICTION_SEAL.json')
    protocol=read(OLD/'PROTOCOL.json');current=read(CURRENT/'TEST_SCORE.json');current_seal=read(CURRENT/'SCORE_SEAL.json')
    assert comparison['prediction_seal_sha256']==sha(OLD/'PREDICTION_SEAL.json')
    assert prediction_seal['before_scoring']is True and prediction_seal['complete']is True
    assert prediction_seal['protocol_sha256']==sha(OLD/'PROTOCOL.json')
    assert current_seal['score_sha256']['TEST']==sha(CURRENT/'TEST_SCORE.json')
    ref=current['reference_binding'];refpath=(ROOT/ref['path']).resolve()
    assert comparison['isolated_scoring_sha256'][str(refpath)]==ref['sha256']
    ids=[row['id']for row in current['rows']]
    assert len(ids)==len(set(ids))==49 and all(row['partition']=='TEST'for row in current['rows'])
    assert set(ids)<=set(protocol['frame_ids'])
    # Metadata text filter precedes decoding; retain only requested metric fields.
    records={m:{}for m in METHODS};source_order={m:[]for m in METHODS}
    for line in gzip.open(scored,'rt'):
        if not re.search(r'"partition"\s*:\s*"TEST"',line)or not re.search(r'"method"\s*:\s*"(?:NATIVE_BASE|INITIAL_N3_SUBPIX)"',line):continue
        row=json.loads(line);method=row['method'];fid=row['id'];metric=row['metric']
        assert row['partition']=='TEST'and fid not in records[method]
        t=metric['translation_cm'];r=metric['rotation_deg']
        assert math.isfinite(t)and math.isfinite(r)
        assert metric['success_5cm_5deg']==(t<5 and r<5)
        assert t!=5 and r!=5
        records[method][fid]=dict(translation_cm=t,rotation_deg=r,success_5cm_5deg=metric['success_5cm_5deg'])
        source_order[method].append(fid)
    for method in METHODS:
        assert source_order[method]==ids and set(records[method])==set(ids)
        assert sum(x['success_5cm_5deg']for x in records[method].values())==comparison['summary']['TEST/'+method]['success_count']
    pairs=[]
    groups={key:[]for key in ('gain','loss','both_success','both_failure')}
    for fid in ids:
        a=records[METHODS[0]][fid];b=records[METHODS[1]][fid]
        kind='gain'if b['success_5cm_5deg']and not a['success_5cm_5deg']else 'loss'if a['success_5cm_5deg']and not b['success_5cm_5deg']else 'both_success'if a['success_5cm_5deg']else 'both_failure'
        groups[kind].append(fid)
        pairs.append(dict(id=fid,raw_yolo_native_base=a,N3_SUBPIX=b,classification=kind,
            delta_translation_cm=b['translation_cm']-a['translation_cm'],delta_rotation_deg=b['rotation_deg']-a['rotation_deg']))
    summaries={}
    for method in METHODS:
        values=list(records[method].values())
        summaries[method]=dict(count=49,successes=sum(x['success_5cm_5deg']for x in values),
            translation_lt5=sum(x['translation_cm']<5 for x in values),rotation_lt5=sum(x['rotation_deg']<5 for x in values),
            mean_translation_cm=sum(x['translation_cm']for x in values)/49,max_translation_cm=max(x['translation_cm']for x in values),
            mean_rotation_deg=sum(x['rotation_deg']for x in values)/49,max_rotation_deg=max(x['rotation_deg']for x in values))
    assert summaries[METHODS[0]]['successes']==30 and summaries[METHODS[1]]['successes']==40
    assert len(groups['gain'])-len(groups['loss'])==10
    inputs=[OLD/'COMPARISON.json',OLD/'PROTOCOL.json',OLD/'PREDICTION_SEAL.json',scored,CURRENT/'TEST_SCORE.json',CURRENT/'SCORE_SEAL.json']
    result=dict(schema='RAW_YOLO_NATIVE30_VS_N3_SUBPIX40_EXACT_TEST49_PAIRED_METRICS_V1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        posthoc_user_requested=True,source_sha256={relative(Path(__file__).resolve()):sha(Path(__file__).resolve())},input_sha256={relative(p):sha(p)for p in inputs},
        rows=49,ordered_ids=ids,same_ordered_TEST49_and_current_reference_identity=True,reference_binding=ref,reference_file_opened=False,
        thresholds=dict(translation_cm='strict <5',rotation_deg='strict <5',joint='both strict conditions',exact_boundary_rows=0,inclusive_counts_identical=True),
        methods=dict(NATIVE_BASE='original cached YOLO native q0 + unchanged W/D selection and F PnP; not raw network coordinates scored as 3D',
            INITIAL_N3_SUBPIX=protocol['N3_initial']),summaries=summaries,groups=groups,group_counts={k:len(v)for k,v in groups.items()},pairs=pairs,
        net_joint_gain=10,no_inference_PnP_optimizer_or_new_reference_reads=True,raw_pose_or_native_points_not_used=True,
        caveat='Existing TEST repeatedly used; saved annotation/PnP-derived reference is not independent metrology. Paired change is the whole N3+SubPix pipeline, not evidence for SubPix alone.')
    save(OUTPUT/'PAIR_METRICS.json',result)
    lines=['# 같은 TEST49: YOLO native 30 → N3+SUBPIX 40','',
        '**실제 개선 10건, 퇴행 0건, 순증 10건**이다. 두 방법 모두 성공 30건, 둘 다 실패 9건이다. 전체49를 빠짐없이 비교했다.','',
        '기준은 과거 평가식의 **위치<5cm 및 회전<5°**다. 정확히 5인 행은 없어 ≤ 기준도 같은 수다. 현재 TEST49의 ID·순서와 고정 참조 path/SHA가 일치한다. 참조 파일은 열지 않았다.','',
        'YOLO native는 원 q0 뒤 기존 F(W/D 선택+PnP)를 적용한 저장 초기 자세다. N3+SUBPIX는 YOLO→N3_DIM_SYM→SubPix→YOLO-anchor cap→같은 F의 전체 경로다. SubPix 단독 효과로 부르지 않는다.','',
        '| ID | YOLO T cm | N3+SUBPIX T cm | YOLO R ° | N3+SUBPIX R ° |','|---|---:|---:|---:|---:|']
    for row in pairs:
        if row['classification']=='gain':
            a=row['raw_yolo_native_base'];b=row['N3_SUBPIX'];lines.append(f"| {row['id']} | {a['translation_cm']:.4f} | {b['translation_cm']:.4f} | {a['rotation_deg']:.4f} | {b['rotation_deg']:.4f} |")
    lines+=['','10건 모두 YOLO 위치>5cm가 N3+SUBPIX 위치<5cm로 바뀌었고, 양쪽 회전은 모두5° 미만이다. 퇴행 목록은 빈 목록이며, 성공 수 순증을 맞추기 위해 사례를 선별하지 않았다.','',
        '49개 전체 T/R·분류·변화량, 모든 gain/loss ID와 입력 SHA는 [PAIR_METRICS.json](PAIR_METRICS.json)에 보존했다. 저장 metric만 읽었고 이미지·추론·새 PnP·optimizer·정답 재계산은 없다. 원 TEST의 반복 탐색·참조 유도 평가 한계는 남는다.']
    (OUTPUT/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    assets={name:sha(OUTPUT/name)for name in ('PAIR_METRICS.json','REPORT_KO.md','extract_metrics_v1.py')}
    save(OUTPUT/'SEAL.json',dict(schema='SEALED_POSTHOC_TEST49_PAIRED_METRIC_EXTRACTION_V1',rows=49,group_counts=result['group_counts'],assets=assets,input_sha256=result['input_sha256'],new_reference_RGB_or_pose_calls=0))
    return dict(seal_sha256=sha(OUTPUT/'SEAL.json'),metrics_sha256=assets['PAIR_METRICS.json'],group_counts=result['group_counts'])


if __name__=='__main__':print(run())
