"""Build an additive closeout from sealed saved evidence, without inference."""
from pathlib import Path
from copy import deepcopy
from collections import Counter
import argparse
import hashlib
import gzip
import json
import shutil

NAMES={'R0':'RAW + F','R1':'RAW + B/I','N3_F':'N3만 + F','N3_BI':'N3만 + B/I',
    'SUBPIX_F':'SubPix만 + F','SUBPIX_BI':'SubPix만 + B/I','N0':'N3→SubPix + F',
    'N1':'N3→SubPix + B/I','E0':'기존 e · 문맥 비교'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError('Keep previous closeout: '+str(p))
    p.write_text(x if isinstance(x,str) else json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def table(pop):
    rows=['| 경로 | 공동 성공/N | 위치 평균±SD cm | 위치 분산 cm² | 위치 최대 cm | 회전 평균±SD ° | 회전 분산 °² | 회전 최대 ° |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm,name in NAMES.items():
        a=pop['arms'][arm];c=a['counts'];t=a['errors']['translation_cm'];r=a['errors']['rotation_deg']
        rows.append(f'| {name} | {c["successes_joint_le5_n"]}/{c["full_n"]} | '
            f'{t["mean"]:.3f}±{t["sample_SD_ddof1"]:.3f} | {t["sample_variance_ddof1"]:.3f} | {t["max"]:.3f} | '
            f'{r["mean"]:.3f}±{r["sample_SD_ddof1"]:.3f} | {r["sample_variance_ddof1"]:.3f} | {r["max"]:.3f} |')
    return '\n'.join(rows)

def generate(folder,output,source_root):
    folder,output,source_root=map(Path,(folder,output,source_root));output.mkdir(parents=True,exist_ok=True)
    receipt=read(folder/'variance_v1/RECEIPT.json');data=read(folder/'variance_v1/STATISTICS.json')
    seal=read(folder/'variance_v1/SEAL.json')
    if receipt['failure_count'] or receipt['metric_summary_rows']!=108:raise ValueError('Full variance verification required')
    for name,h in seal['artifacts'].items():
        if sha(folder/'variance_v1'/name)!=h:raise ValueError('Variance evidence changed')
    for name in ('STATISTICS.json','STATISTICS.csv','RECEIPT.json','SEAL.json'):
        dst=output/'variance'/name;dst.parent.mkdir(exist_ok=True)
        if dst.exists():raise FileExistsError(dst)
        shutil.copyfile(folder/'variance_v1'/name,dst)
    for name in ('RESOURCE_REVIEW_V1.json','RESOURCE_REVIEW_KO_V1.md'):
        dst=output/'evidence'/name;dst.parent.mkdir(exist_ok=True)
        if dst.exists():raise FileExistsError(dst)
        shutil.copyfile(folder/'resource_review_v1'/name,dst)
    tail_source=source_root/'_docs/experiments/pallet_single_component_bi_v1/metrics/SCORED_ROWS.json.gz'
    with gzip.open(tail_source,'rt') as stream:
        saved=json.load(stream)['rows']
    tail=[r['metrics']['N0']['translation_cm'] for r in saved if r['partition']!='CAL']
    if len(tail)!=192 or abs(sum(tail)/len(tail)-data['populations']['PRIMARY']['arms']['N0']['errors']['translation_cm']['mean'])>1e-9:
        raise ValueError('The same full192 rows are required for tail interpretation')
    tail_sentence=(f'N3→SubPix + F의 위치 오차는5cm 이하가{sum(x<=5 for x in tail)}장인데, 위치·회전 동시 성공은108장이다. '
        f'위치50cm 초과{sum(x>50 for x in tail)}장,100cm 초과{sum(x>100 for x in tail)}장이 남고 '
        f'큰10장이 위치 오차 총합의{100*sum(sorted(tail)[-10:])/sum(tail):.2f}%를 차지한다. '
        '이런 꼬리가 평균과 분산을 키우므로 성공 프레임만 평균 내면 안 된다.')
    paragraphs=['# 평가 데이터 평균·분산과 지시문 마감',
        '**정량 실험·재집계·산출물 게시를 마감한다. 판정은 UPSTREAM_STILL_USEFUL이다.** '
        '과거의 복원 불가능한 정밀 시간 공백은 기록 제한으로 보존한다. 이를 완전 계측으로 바꾸거나 성능 실패를 지우지 않는다.',
        '주평가는 고정 TEST49+추가143=192장이다. CAL46을 주평가에 합치지 않았다. '
        '성공은 같은 물체 중심 위치≤5cm AND 같은 physical/C2 회전≤5°다. 모든 경로의 연속 오차는192/192개가 가용하며 큰 실패를 포함한다.',
        '**분산은 표본분산(ddof=1, n−1 분모)** 이고, SD는 그 제곱근인 표본표준편차다. '
        '위치 평균·SD는cm, 위치 분산은cm²이며 회전은각각°·°²다. '
        '이 산포는 영상 간 오차 분포이며 평균의 CI·표준오차·seed 간 변동이 아니다. '
        '모집단분산(ddof=0, n 분모)도 [CSV](variance/STATISTICS.csv)에 함께 보존했다.',
        '## 전체 평가192장',table(data['populations']['PRIMARY']),
        '8개 성분 비교에서 N3→SubPix + F가 위치 평균11.306cm, 회전 평균10.631°로 가장 낮다. '
        '여기에 B/I를 추가하면 성공 수는108로 같고 위치 평균13.384cm·분산915.320cm²로 커진다. '
        '따라서 N3와 SubPix를 유지하고 새 B/I 후단을 무조건 적용하는 운영 변경은 권하지 않는다.',
        '기존 e는 같은192장에서114장 성공하지만 위치 평균14.112cm·최대260.412cm다. '
        '성공 수와 평균·분산의 우열이 달라, e가 모든 기준에서 우월하거나 전체 문제가 해결됐다고 설명하지 않는다. '
        'E0는 같은 ID·참조·지표가 확인된 저장 결과의 문맥 비교이며 새 선택 후보가 아니다.',
        tail_sentence,
        '## 원49장',table(data['populations']['TEST']),
        '## 추가143장',table(data['populations']['REMAINDER']),
        '## CAL46 · 별도',table(data['populations']['CAL']),
        '## 원지시문의 여섯 판단',
        '1. RAW에 B/I를 붙이면 공동 성공66→82/192이며 회복21·손상5다. RAW 연구 후보의 추가 가치는 관찰됐다.\n'
        '2. 같은 B/I 뒤에도 N3+SubPix는108장, N3만90장, SubPix만100장, RAW82장이다. 둘을 제거하는 권고를 하지 않는다.\n'
        '3. 실행에 쓰지 않는 O/T domain 및 U 표시용 모델·builder를 분리했다. 실제 상태 ResNet 특징·Tiny·MobileSAM·B/I 측정·공분산·기존 solver는 유지했다.\n'
        '4. 일부 손상은 작은 픽셀 이동 뒤 적은 점의 다른 PnP 해를 고르는 단계에서 발생했다. e 후속 사례는 원 N3의 잘못된 코너 대응을 다시 주입한 해석과 일치한다. 모든 실패를 이 한 원인으로 확정하지 않는다.\n'
        '5. 관측 고정 좌표 교체 진단은 RAW82→68, 결합108→105였다. 좌표 수정의 조건부 기여이며 전체 이득의 인과 분해가 아니다.\n'
        '6. 운영 교체·논문 수정 전에 물리 코너 대응의 후단 전달과 독립 자료의 손상 억제를 확인해야 한다. 이번에 새 학습·촬영·정답 수정·추가 방법 탐색을 시작하지 않는다.',
        '## 종결 기록과 검산',
        f'원8경로1904행 독립검산117,037개, 이번 평균·분산 검산{receipt["checks"]:,}개에서 불일치0이다. '
        '새 예측·모델·정답·성공 기준은 바꾸지 않았다. CSV 포장 오류1건은 수리하고 원 시도 기록을 보존했다.',
        '[원8경로 전체 보고서](../pallet_single_component_bi_v1/RESULTS_KO.md), '
        '[모든 실패·전이](../pallet_single_component_bi_v1/metrics/CHANGED_CASES.csv), '
        '[실제32단계 그림](../pallet_single_component_bi_v1/gallery/GALLERY_KO.md), '
        '[최종 지시문 마감](CLOSEOUT_KO.md), [기존 자료 집합 대조](LEGACY_POPULATIONS_KO.md), '
        '[과거 시간 기록의 독립 검토](evidence/RESOURCE_REVIEW_KO_V1.md)를 함께 보존했다.']
    write(output/'RESULTS_KO.md','\n\n'.join(paragraphs)+'\n')
    original=read(source_root/'_docs/experiments/pallet_single_component_bi_v1/COMPLIANCE_AUDIT_FINAL.json')
    close=deepcopy(original)
    close['schema']='ADDITIVE_INSTRUCTION_CLOSEOUT_V1';close['supersedes_no_old_seal']=True
    close['execution_status']='COMPLETE';close['method_verdict']='UPSTREAM_STILL_USEFUL'
    close['audit_closeout']='CLOSED_WITH_DOCUMENTED_HISTORICAL_RECORDING_LIMITS'
    close['not_a_waiver_or_full_historical_timing_certification']=True
    close['current_publication_base']=receipt['published_source']['source_commit']
    close['new_statistics_recount']=dict(checks=receipt['checks'],failures=0,rows=108,new_inference=0)
    close['resolutions']=dict(
        legacy245_110='Actual distinct245 archive source, three49x5 prediction journals and same192 success110 reconciled separately; no hypothetical cohort substituted.',
        historical_timing='Exact unrecorded startup/read/probe times cannot be reconstructed; original recording defect remains disclosed, not marked as fully measured.',
        publication='All original and single-component results verified on main; additive closeout goes to main without creating a branch.')
    close['previous_overall']=close.get('overall')
    close['overall']='EXPERIMENT_AND_REPORTING_CLOSED_WITH_DOCUMENTED_HISTORICAL_LIMITATIONS'
    for requirement in close.get('requirements',[]):
        key=requirement['id']
        if key in ('legacy245_110','resources','new_main_push'):
            requirement['previous_status']=requirement['status']
            if key=='legacy245_110':
                requirement['status']='CLOSED_ACTUAL_SOURCES_RECONCILED'
                requirement['closeout_evidence']='LEGACY_POPULATIONS_KO.md; evidence/LEGACY_POPULATIONS_CLOSEOUT.json'
            elif key=='resources':
                requirement['status']='CLOSED_WITH_HISTORICAL_RECORDING_DEFECT_DISCLOSED'
                requirement['closeout_evidence']='evidence/RESOURCE_REVIEW_V1.json; exact missing historical times remain unavailable'
            else:
                requirement['status']='COMPLETE_PRIOR_WORK_VERIFIED_ON_MAIN'
                requirement['closeout_evidence']=receipt['published_source']['source_commit']
    close['status_counts']=dict(Counter(r['status'] for r in close.get('requirements',[])))
    write(output/'CLOSEOUT.json',close)
    write(output/'CLOSEOUT_KO.md','''# 지시문 마감 기록

**실험·평가·재현 검사·보고·main 게시를 마감한다. 과학적 판정은 UPSTREAM_STILL_USEFUL이다.** 처음 지시문이 요구한4경로952행과384개 좌표 진단을 완료했고, 후속 요청의 N3/SubPix 단독까지8경로1904행을 완료했다. 기존 실패와 모든 분모를 보존했다.

이 문서는 이전 미완료 감사나 봉인 파일을 덮어쓰지 않는다. [기계 판독 종결](CLOSEOUT.json)과 [평균·분산 결과](RESULTS_KO.md)를 추가한다.

- **자료 집합 대조 종결:** 실재하는245개 고유영상의312행 원장과 두 archive를 찾았다. 49×5=245 예측 행, 실제 다른245개 자료, 동일192장 중110개 성공을 각각 구분해 출처·ID·해시 차이를 기록했다. 확인되지 않은 새245/110평가셋을 가정하거나 severe·정사각형 때문에 빠졌다고 추측하지 않았다. [자료 근거](LEGACY_POPULATIONS_KO.md).
- **과거 시간 기록 제한 명시:** 일부 초기 읽기·probe·시작/종료 시간은 기록이 없어 복원할 수 없다. 이를0초나 완전 실측으로 바꾸지 않는다. 기록된 stage와 포함 구간의 검산은 보존하며, 역사 전체 자원 정산의 완전 인증은NA다. 새로 다시 실행한 시간을 과거 시간처럼 대체하지 않았다. [독립 검토](evidence/RESOURCE_REVIEW_KO_V1.md).
- **허용된 미계산:** 전체 RGB→최종 자세의 실제 호출이 연결되지 않은 전체시간/FPS는NA, 검정 정의가 검증되지 않은p/Holm p도NA다. 원문이 허용한 처리이며 동등성·실시간성의 증거가 아니다.
- **게시 지시 반영:** 사용자의 main 게시 요청이 원문의 연구 브랜치 규칙을 대체한다. 기존 minimal·guarded·단독 비교 결과는main a118730에 게시된 상태를 재확인했다. 이번 추가 종결 문서와 분산 재집계도main에 게시한다. 새 브랜치·강제 push·운영 기본값 교체·논문 변경은 없다.

따라서 실행 가능한 보충과 산출물 정리는 끝냈다. 복원 불가능한 역사 기록 결함까지100% 수행으로 소급 표시한 것은 아니다. 이미 끝난 실험을 이 제한 때문에 계속 재실행하거나 새 방법·데이터를 추가하지 않는다. 새 학습·seed·정답 수정은0이다.
''')
    write(output/'README.md','# 지시문 마감과 평가 평균·분산\n\n'
        '[평가 결과](RESULTS_KO.md) · [종결 기록](CLOSEOUT_KO.md) · [원집합 대조](LEGACY_POPULATIONS_KO.md) · '
        '[전체 통계 CSV](variance/STATISTICS.csv) · [이전 전체 실험](../pallet_single_component_bi_v1/RESULTS_KO.md)\n')
    write(output/'REPORT_INPUTS.json',dict(variance_seal_sha256=sha(folder/'variance_v1/SEAL.json'),
        variance_receipt_sha256=sha(folder/'variance_v1/RECEIPT.json'),
        original_audit_sha256=sha(source_root/'_docs/experiments/pallet_single_component_bi_v1/COMPLIANCE_AUDIT_FINAL.json'),
        resource_review_sha256=sha(folder/'resource_review_v1/RESOURCE_REVIEW_V1.json'),
        tail_scored_rows_sha256=sha(tail_source),
        report_source_sha256=sha(__file__),new_model_or_reference_or_image_opens=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-root',type=Path,required=True);a=p.parse_args();generate(a.folder,a.output,a.source_root)
