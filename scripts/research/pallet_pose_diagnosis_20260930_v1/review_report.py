"""Build the GitHub-readable report and explicit correction/reproduction notes."""
import csv
import json
from collections import Counter
from . import run as C
from .review_analysis import ARCHIVE,write

def num(value):return 'NA' if value is None else f'{value:.3f}'
def interval(pair,key):
    value=pair['recording_bootstrap']['intervals'][key]['percentile95']
    return 'NA' if value is None else '['+', '.join(num(v) for v in value)+']'

def main():
    allstats=C.read(C.DOC/'E7_ALL_COMPARISONS.json');comparisons=allstats['comparisons'];assert len(comparisons)==69
    for filename,kind in [('E2_ORACLE_SUMMARY.json','oracle'),('E2_DETECTION_SUMMARY.json','boxes')]:
        data=C.read(C.DOC/filename)
        if kind=='oracle':
            for arm,groups in data['models'].items():
                for pop in ('NATURAL99','CLEAN29'):
                    for choice in ('T_best','R_best'):groups[pop][choice]['paired']=comparisons[f'E2/{arm}/{choice}'][pop]['paired']
        else:
            for pop in ('NATURAL99','CLEAN29'):
                for arm,values in data['groups'][pop].items():values['best_box_minus_native']=comparisons[f'E2/{arm}/best_box'][pop]['paired']
        data['review_supplement']='E7_ALL_COMPARISONS.json'
        write(C.DOC/filename,data)
    # The old E4 recording groups were mixed populations. Replace them explicitly.
    e4=C.read(C.DOC/'E4_SUMMARY.json');metrics=C.read(C.RAW/'E4_METRICS.json');rows=C.metadata()
    for key in list(e4['groups']):
        if key.startswith('recording:'):del e4['groups'][key]
    for pop in ('NATURAL99','CLEAN29'):
        rr=[r for r in rows if (r['severity']=='CLEAN')==(pop=='CLEAN29')]
        for rec in sorted({r['recording'] for r in rr}):
            sub=[r for r in rr if r['recording']==rec]
            e4['groups'][f'{pop}/recording:{rec}']=dict(frames=len(sub),population=pop,recording=rec,
                models={a:C.summarize(v[r['id']] for r in sub) for a,v in metrics.items()},
                paired={a:C.paired(metrics['baseline_'+a.rsplit('_',1)[1]],v,sub,False) for a,v in metrics.items() if not a.startswith('baseline')})
    e4['review_supplement']='E7_ALL_COMPARISONS.json';write(C.DOC/'E4_SUMMARY.json',e4)
    status=C.read(C.DOC/'QUESTION_STATUS.json')
    status['E1_source_contract']=dict(status='ANSWERED',reason='Separate sidetable hash, K/padding, renderer pose, 8-index bijection/projection and current domain-specific symmetry checked for256. Source C1=62/C2=194; natural C2. Operational source refiner outputs remain unavailable.',evidence=['SOURCE_CONTRACT_REVIEW.json'])
    status['E1_source']['reason']='Source file and geometry contracts verified separately; FULL125 operational xy unavailable. Additional source model inference remains conditionally skipped; no old source R transferred.'
    status['E1_source']['evidence']=['SOURCE_CONTRACT_REVIEW.json','E7_REFERENCE_LIMITS.json']
    status['E7_statistics']['reason']='69 named pose comparisons, separate natural/clean recording bootstrap/LORO and descriptive metadata strata. TRAIN one-recording2D lacks identifiable cross-recording uncertainty; this is explicit.'
    status['E7_statistics']['evidence']=['E7_ALL_COMPARISONS.json','E7_BY_RECORDING_ALL.csv','E7_METADATA_STRATA_ALL.csv','STATISTICS_KO.md']
    status['review_output_fields']=dict(status='ANSWERED',reason='33 held-branch labels corrected; E4 intervention2D recomputed for768 rows. Original2964 T/R unchanged;1024 E4 pose rows independently rechecked.',evidence=['CSV_FIELD_REPAIR.json'])
    write(C.DOC/'QUESTION_STATUS.json',status)
    limits=C.read(C.DOC/'E7_REFERENCE_LIMITS.json');limits['source256']['geometry_contract_review']='SOURCE_CONTRACT_REVIEW.json';limits['source256']['current_symmetry_counts']={'1':62,'2':194};write(C.DOC/'E7_REFERENCE_LIMITS.json',limits)
    summary=C.read(C.DOC/'SUMMARY.json');summary['questions']=status;summary['github_review']=dict(comparisons=69,statistics='E7_ALL_COMPARISONS.json',CSV_field_repair='CSV_FIELD_REPAIR.json',source_contract='SOURCE_CONTRACT_REVIEW.json',gallery='GALLERY_NATURAL99.md');write(C.DOC/'SUMMARY.json',summary)
    stat=['# 전체 비교의 recording 불확실성','',
        '69개 명명된 pose 비교를 모집단별로 계산했다. 하나의 비교가 natural99와 clean29에 모두 있으면 두 행으로 표시한다. Δ는 after−before이며 음수가 개선이다. “중앙값 차이”와 “프레임별 차이의 중앙값”은 다르다. T 단위 cm, R 단위 °. paired recording bootstrap 2,000회·seed20260929이며 모든 비교에 같은 모집단/recording 표본추출 규칙을 적용했다.','',
        'natural99는 6개, clean29는 3개 recording뿐이고 반복 사용 DEV다. 구간은 탐색적이며 다중 비교 보정이나 일반화 보장이 아니다. CI에 0이 있으면 불확실성이 남는다는 뜻이다. Oracle은 평가 참조를 이용하므로 배포 성능이 아니다.','',
        '| 비교 | 모집단 / N / recording | ΔT 중앙값 | ΔR 중앙값 | 프레임 ΔT 중앙값 | 프레임 ΔR 중앙값 | T 95% 구간 | R 95% 구간 | 정의 불가 draw T/R |',
        '|---|---|---:|---:|---:|---:|---|---|---|']
    for name,groups in comparisons.items():
        for pop,v in groups.items():
            pp=v['paired'];d=pp['difference_of_conditional_medians'];m=pp['median_of_common_frame_differences'];ci=pp['recording_bootstrap']['intervals']
            stat.append(f'| `{name}` | {pop} / {v["N"]} / {len(v["recordings"])} | {num(d["translation_cm"])} | {num(d["rotation_deg"])} | {num(m["translation_cm"])} | {num(m["rotation_deg"])} | {interval(pp,"translation_cm")} | {interval(pp,"rotation_deg")} | {ci["translation_cm"]["undefined_resamples"]}/{ci["rotation_deg"]["undefined_resamples"]} |')
    stat+=['','recording별 N·유효/실패·절대값·차이: [전체 recording 표](E7_BY_RECORDING_ALL.csv). 가림·앙각·거리·bbox 크기·시점·기준 W/D·기준 매칭 여부의 단일 조건별 결과: [metadata 표](E7_METADATA_STRATA_ALL.csv). W/D·매칭 층은 개입 전 identity 기준으로 고정했다. 작은 교차 집단이나 회귀를 추가하지 않았다. LORO와 전체/공통 유효 분모는 [전체 JSON](E7_ALL_COMPARISONS.json)에 있다.','',
        'E5-A/B는 학습 REC_001 하나의 타깃 추종/복원 진단이다. 독립 평가 T/R 참조가 없고 recording이 하나이므로 cross-recording 신뢰구간을 계산할 수 없다. 동일 recording bootstrap을 반복해 폭 0인 구간을 일반화 근거로 만들지 않았다. E5-B의 29장×2마스크×3모델 입력·출력 2D 오차는 [프레임 표](E5B_FRAME_REVIEW.csv)에 있다.']
    write(C.DOC/'STATISTICS_KO.md','\n'.join(stat)+'\n')
    corrections='''# GitHub 게시 전 보완 및 검증

최초 완료 보고 이후 사용자 질문에 대한 재감사에서 실제 누락을 확인했다. [당시 감사 기록](INSTRUCTION_COMPLIANCE_AUDIT_KO.md)은 역사적 기록으로 남기고, 아래를 이번 GitHub 검토판에 반영했다. 기존 자연99/clean29의 주 T/R 결과는 바꾸지 않았다.

| 지적 | 이번 조치 | 검증 근거 |
|---|---|---|
| Source256은 파일 hash만 재확인 | side table hash, K와 padding, renderer R/t, 치수·중심, 8코너 전단사/재투영, 현재 평가 코드의 대칭 계약을 256장 검사 | [Source 계약](SOURCE_CONTRACT_REVIEW.json) |
| E2 CI/LORO 누락 및 E7 일부만 적용 | 69개 pose 비교 전체에 모집단별 paired recording 분석. 모든 비교의 recording·단일 metadata 층별 표 작성 | [통계 설명](STATISTICS_KO.md) |
| E4 recording 표에 natural/clean 혼합 | 두 모집단×recording으로 분리 | [E4 요약](E4_SUMMARY.json), [전체 recording CSV](E7_BY_RECORDING_ALL.csv) |
| held_identity 33행의 후보명 오표기 | 실제 사용 후보와 자유 GEO 후보를 분리 | [필드 수정 검증](CSV_FIELD_REPAIR.json) |
| E4 개입 768행에 baseline 2D 표시 | 실제 교체 좌표로 2D 오차 재계산. 입력과 출력 의미 분리 | [수정 CSV](FRAME_RESULTS.csv) |

Source의 최대 renderer 재투영 잔차는 약 0.000451 px, 준비된 label과는 약 0.000825 px다. 현재 Source256의 허용 대칭은 C1 62장/C2 194장이며 자연 평가의 C2와 동일하지 않다. 기존 Source의 rotation 집계값을 자연 평가 값으로 복사하지 않는다. renderer의 정확한 3D 대응점으로 한 PnP 검사는 기하 연결 점검이며 모델 정확도 결과가 아니다. FULL125의 해당 operational xy 캐시가 없어 Source 모델 T/R 신규 비교는 조건부 생략 상태로 유지한다.

검증은 실제 T/R 2,964행의 최초 독립 재계산에 더해, 이번 E4 1,024행의 pose 재검산·T/R 변경 없음·33행 후보명·768행 2D 대상 수정까지 포함한다. 새 모델 추론·학습·촬영·GT 변경은 0회다. 보완 분석의 CPU 비용은 REVIEW_COST 파일에 별도 기록한다.

자료 부족으로 남은 자연 가림의 대응 clean 타깃 정확도, RGB 효과의 일반화, 박스 불일치의 물체 정체, E5-B의 어려운 오류 전이, REALFT_A 선택 이력, 실측 재클릭 noise floor를 완료로 바꾸지 않았다. [현재 질문 상태](QUESTION_STATUS.json)를 확인한다.

최초 완료 시점의 manifest는 [history](history/RUN_MANIFEST_INITIAL_COMPLETION.json)에 보존하고, 변경된 산출물·코드의 이전 바이트는 로컬 `data/pallet/results/pallet_pose_diagnosis_20260930_v1/before_github_review/`에 보존했다. 최신 파일 hash는 PUBLICATION_MANIFEST.json의 경로에 적용한다. 과거 manifest의 hash를 최신 수정 파일에 적용하면 일치하지 않는 것이 정상이며, 변경 연결은 PUBLICATION_VALIDATION.json에 기록한다.
'''
    write(C.DOC/'REVIEW_CORRECTIONS_KO.md',corrections)
    reportpath=C.DOC/'REPORT_KO.md';old=ARCHIVE/reportpath.relative_to(C.ROOT);report=old.read_text() if old.exists() else reportpath.read_text()
    report=report.replace('**[확인] 지정 범위의 진단을 완료했다.', '**[확인] 핵심 실험을 수행했고, GitHub 검토판에서 재감사로 발견한 분석 누락과 표기 오류를 보완했다.')
    report=report.replace('## 무엇이 문제였고 어떻게 구별했나', '''## 이 보고서를 확인하는 순서

[입력·모델·평가 계약](INPUTS_AND_METHOD_KO.md) → 아래 결과와 그림 → [자연 가림 99장 전체 갤러리](GALLERY_NATURAL99.md) → [모든 비교의 통계](STATISTICS_KO.md) → [지시문 이행·수정 내역](REVIEW_CORRECTIONS_KO.md) 순서로 확인할 수 있다. [재현·파일 안내](REPRODUCE_KO.md), [프레임 CSV](FRAME_RESULTS.csv), [현재 질문 상태](QUESTION_STATUS.json)도 함께 제공한다.

![모델별 자연99 및 clean29 T/R](figures/01_pose_overview.png)

막대는 집계 중앙값이며 신뢰구간이 아니다. REALFT_A는 초기화·검출기·실사 감독이 다른 참고 모델이다. 순수 보정기 효과와 구별한다.

## 무엇이 문제였고 어떻게 구별했나''')
    report=report.replace('## E2: 후보 선택 여지와 검출 꼬리','''![자연99 프레임별 T/R 변화](figures/02_paired_changes.png)

왼쪽 아래는 같은 프레임의 T/R 동시 개선, 오른쪽 위는 동시 악화다. T축은 큰 꼬리와 0 근처를 함께 보이기 위한 symlog 축이다. 점 하나가 프레임 하나이며 recording을 색으로 구분했다.

![자연 가림 개선·악화 사례](figures/06_natural_examples.jpg)

각 방향 집단에서 ID 사전순 첫 프레임을 표시했다. 최상의 개선 사례를 골랐다는 의미가 아니며, 네 사례의 빈도는 위 99장 집계로 판단한다. 주황=모델 코너, 청록=저장 참조, 흰 점선=선택 검출 박스. 원본 이미지와 ID는 [사례 선택 기록](FIGURE_CASE_SELECTION.json), 모든 프레임은 [전체 갤러리](GALLERY_NATURAL99.md)에 있다.

## E2: 후보 선택 여지와 검출 꼬리''')
    pp=comparisons['E2/FULL125/T_best']['NATURAL99']['paired'];boxpp=comparisons['E2/FULL125/best_box']['NATURAL99']['paired']
    e2extra=f'''보완한 recording 분석에서 FULL125 T-best의 현재 GEO 대비 95% 구간은 ΔT {interval(pp,'translation_cm')} cm, ΔR {interval(pp,'rotation_deg')}°다. 이는 참조를 아는 후보 선택의 진단적 여지를 지지하지만 배포 가능한 선택기를 얻었다는 뜻은 아니다. best-box의 ΔT 구간 {interval(boxpp,'translation_cm')} cm는 0을 포함한다. LORO와 모든 후보 비교는 [통계 설명](STATISTICS_KO.md)을 참조한다.

![T 오류 상위10 프레임 전부](figures/07_detection_tail.jpg)

상위 10장 모두를 표시했다. 흰 점선과 청록 참조 박스가 어긋나는 경우를 직접 확인할 수 있다. 이 그림만으로 다른 실제 팔레트를 검출했는지 확정하지 않는다.

'''
    report=report.replace('## E3: clean29 동일 이미지 RGB×좌표',e2extra+'## E3: clean29 동일 이미지 RGB×좌표')
    e3figs='![RGB와 좌표 조건의 T/R](figures/04_rgb_coordinate_controls.png)\n\n'
    for rec in ('REC_021','REC_041','REC_044'):
        e3figs+=f'![{rec} clean 및 가림 실제 입력](figures/05_e3_inputs_{rec}.jpg)\n\n'
    e3figs+='각 clean recording의 ID 사전순 첫 프레임이다. 가운데는 실제 가림 RGB와 native qO, 오른쪽은 같은 qO를 clean RGB·clean 박스에 넣는 CO 입력이다. 그림의 코너는 입력 예측이고 보정기 출력으로 혼동하지 않는다. 원래 mask seed·사각형·실제 참조 겹침은 바꾸지 않았다.\n\n'
    report=report.replace('## E4: 사람 가시성의 부분 참조 교체',e3figs+'## E4: 사람 가시성의 부분 참조 교체')
    report=report.replace('## E6: 기존 실사 지도 모델 기준','![독립 및 동반 좌표 오류 복구](figures/08_correlated_stress.png)\n\n복구 분모 116은 29장×4코너이며 독립 이미지 116장이 아니다. clean R0 복원을 측정하며 자연 가림의 물리 타깃 정확도를 뜻하지 않는다.\n\n## E6: 기존 실사 지도 모델 기준')
    report=report.replace('## E7: recording·참조 신뢰성과 반론','## E7: recording·참조 신뢰성과 반론\n\n![recording별 변화](figures/03_recording_changes.png)\n\n')
    report=report.replace('LORO는 각 결과 JSON에 있다.', '전체 69개 비교의 모집단별 LORO·CI는 [E7 전체 결과](E7_ALL_COMPARISONS.json), 읽기용 표는 [STATISTICS_KO.md](STATISTICS_KO.md)에 있다.')
    report=report.replace('작은 집단을 교차한 회귀는 추가하지 않았다.', '전체 비교로 확대한 [recording 표](E7_BY_RECORDING_ALL.csv)와 [metadata 층별 표](E7_METADATA_STRATA_ALL.csv)를 추가했다. 작은 집단을 교차한 회귀는 추가하지 않았다.')
    report=report.replace('Source256의 image/label/renderer 768개 hash 연결은 확인했다.', 'Source256의 image/label/renderer 768개 hash 연결과 side table·K/padding·pose·인덱스·대칭 계약을 별도로 확인했다. Source는 C1 62장/C2 194장이고 자연 참조는 C2다. [기하 계약 검증](SOURCE_CONTRACT_REVIEW.json).')
    report=report.replace('모든 산출물은 로컬이며 commit/push/외부 공유하지 않았다. 표에 필요한 판단 근거를 담았으므로 장식용 그림은 추가하지 않았다.', '이 검토판은 Markdown·표·실제 이미지 비교·코드를 GitHub에서 함께 검토할 수 있도록 구성했다. 원본 데이터와 가중치는 로컬 입력으로 유지하며 파일 hash와 필요 경로를 명시했다. CSV 수정 내역과 E7 보완은 [검증 내역](REVIEW_CORRECTIONS_KO.md), 공개 파일 전체 hash는 PUBLICATION_MANIFEST.json에 있다.')
    report=report.replace('## 질문별 완료 상태','## 질문별 완료 상태\n\n아래 최초 질문 판정은 유지하되, Source 계약 확인과 결과 필드 수정은 추가로 완료했다. 최신 기계 판독 상태는 [QUESTION_STATUS.json](QUESTION_STATUS.json)에 있다. 이 표의 ANSWERED는 해당 진단 질문에 답했다는 뜻이며 자연 가림 일반화가 입증되었다는 뜻이 아니다.\n')
    write(reportpath,report)
    method='''# 입력·모델·평가 계약

이번 실행의 목적은 clean에서 배운 보정이 자연 가림의 6D pose 개선으로 이어지지 않는 이유를 분리하고 다음 개입 하나를 고르는 것이다. 새 학습은 0회다. 실제 실행 명령, 패키지 환경, 입력 및 체크포인트 SHA256은 [manifest](RUN_MANIFEST.json)와 [모델 계보](MODEL_LINEAGE.json)에 있다.

| 집합 | 이미지 수 | recording 수 | 역할 |
|---|---:|---:|---|
| full128 | 128 | 모집단별 아래 참조 | 동일 평가 계약의 전체 파이프라인 |
| natural99 | 99 = Moderate21 + Severe78 | 6 | 자연 가림 주 결과 |
| clean29 | 29 | 3 | clean 손상, 인공 가림, RGB×좌표 실험 |
| 기존 natural93 | 93 | 과거 집합 | current99와 공통92, old-only1/current-only7 |
| FULL125 학습 | 253 | 1, REC_001 | 실제 학습 의사 타깃 추종 |
| E5-B TRAIN 부분집합 | 29 | 1, REC_001 | ID 정렬 등간격 선택, 새 고정 마스크 |
| Source256 | 256 | 합성 | renderer 기하 계약 확인; 자연 정확도 증명 아님 |

clean29와 ST 학습 clean78은 다르다. natural99마다 대응하는 clean 사진이나 teacher 타깃이 있는 것은 아니다. REC_021/REC_041에는 clean과 natural 이미지가 함께 있어 recording 표에서도 모집단을 먼저 나눈다.

```mermaid
flowchart LR
    A[원본 RGB] --> B[고정 R0 검출·9점]
    B --> C[identity 또는 PRIOR1/FULL125]
    A --> C
    C --> D[동일 W/D 후보의 corner8 PnP]
    D --> E[기존 고정 GEO 선택]
    E --> F[T cm와 C2 회전 R]
    G[저장 2D·K·치수 참조] --> F
    B --> H[진단: 보정 전 W/D 고정]
    C --> H
    H --> F
```

| 모델 | 입력·학습 계보 | 이번 실행에서의 역할 |
|---|---|---|
| identity / R0 | 기존 합성 팔레트 기준 pose 모델; upstream COCO 사전학습 존재 | 보정 없이 통과 |
| PRIOR1 | 고정 synthetic PoseFix prior seed1 last6000 | 사전학습 보정 기준 |
| FULL125 | PRIOR1에서 실제253 짝 입력·의사 타깃으로 기존 적응; BN 고정 | 비교 대상 보정기; 이름125가 학습 이미지 수를 뜻하지 않음 |
| OLD_REF217 | 별도 ST 학생, 기존 REF_LR5 체크포인트 | 별도 현재 성능 기준 |
| REALFT_A | 실사157×20, negative259×6, 합성12000; R0와 다른 초기화 | 실사 지도 참고 모델; 순수 보정 효과 아님 |

이미지 SHA 중복과 recording 중복은 별도로 검사했다. REALFT_A는 학습 recording과 평가10/128장이 겹치며 natural9/clean1이다. 해당 natural9를 제외한90장에서도 개선은 남지만 이 역시 반복 사용 DEV다. 정확한 checkpoint hash는 확인했지만 과거 epoch60 last 설명과 파일 내부 선택 이력은 불일치하여 미해결로 남긴다.

좌표는 원본 영상 pixel xy다. R0는 기존 reflect padding100 경로를 사용하고, 보정기는 bbox 배율1.25·RGB288×384·inverse affine을 따른다. confidence·bbox·invalid point·중심 index8 처리 규칙을 유지한다. index0–7은 camera-facing cuboid 코너, index8은 중심이다. 유효 좌표를 가시점이라고 간주하지 않는다.

T는 팔레트 중심의 카메라 좌표 오차이며 `100 * ||t_pred - t_ref||` cm다. 자연 R은 physical registry 기준으로 I와 Ry180 중 작은 전체 회전각이다. W/D90 교환은 허용 대칭이 아니다. 같은 W/D를 고정해도 연속 R/t와 PnP 내부 해는 달라질 수 있다. 기존 solvePnP/RefineLM의 distortion=None을 그대로 따른다.

평가 참조는 저장된 2D 주석·K·알려진 치수로 만든 geometry-resolved pose다. 독립 장비로 측정한 물리 6D 정답이 아니다. 원클릭·투영점 출처 및 재클릭 공분산이 없어서 경험적 noise floor는 BLOCKED다. 2px 가정이나 CI 폭을 noise floor/MDE라고 부르지 않는다.

Pose 평가는 IoU gate를 새로 적용하지 않는다. 전체 분모, 실패/미검출, 매칭 성공 부분집합, 공통 유효 pose를 구별한다. 표의 실패는 무한대 정책과 조건부 통계를 함께 보존한다. 2D의 기존 IoU≥0.5 매칭은 pose 실패 제외 규칙이 아니다.

E3에서 CC=(clean RGB,qC), CO=(clean RGB,qO), OC=(가림 RGB,qC), OO=(가림 RGB,qO)다. controlled 네 조건은 clean bbox·affine·score·confidence를 공유한다. nativeOO는 실제 가림 검출 경로다. 마스크 seed42와 크기/색/형태는 출력 확인 전에 고정했고 위치만 cover/avoid로 바꾸었다. qC 기반 배치라 참조 기준 cover1장은0코너, avoid3장은1코너와 겹친다. 58개 조건은29개의 원촬영 반복이며 독립58장으로 세지 않는다.

E1 캐시는 기존 GPU 결과를 재사용했다. E3 최초 CPU R0와 기존 GPU cache의 최대 좌표차0.025px가 사전 tolerance0.01px를 넘어서, threshold를 완화하지 않고 E3 clean/가림을 모두 CPU로 맞췄다. 따라서 E1과 E3 clean의 작은 차이를 모델 효과로 해석하지 않는다. [CPU 정합 기록](E3_CPU_CONSISTENCY_ADDENDUM.json).
'''
    write(C.DOC/'INPUTS_AND_METHOD_KO.md',method)
    reproduce='''# 재현 및 파일 안내

보고서와 비교 이미지는 GitHub만으로 확인할 수 있다. 원본 전체 RGB, GT/카메라 배열, 모델 가중치와 대용량 raw cache는 이 게시물에 포함하지 않는다. 새 clone만으로 1,027회의 모델 추론을 즉시 재현할 수 있다고 주장하지 않는다. 로컬 데이터 소유자는 RUN_MANIFEST.json의 입력 경로·hash와 MODEL_LINEAGE.json의 checkpoint를 먼저 맞춰야 한다.

## 검토 파일

| 목적 | 파일 |
|---|---|
| 주요 관찰·반론·다음 비교 | [REPORT_KO.md](REPORT_KO.md) |
| 입력·집합·모델·평가 규칙 | [INPUTS_AND_METHOD_KO.md](INPUTS_AND_METHOD_KO.md) |
| 자연99 전체 사진과 전후 코너 | [GALLERY_NATURAL99.md](GALLERY_NATURAL99.md) |
| 모든 비교의 CI·분모·LORO | [STATISTICS_KO.md](STATISTICS_KO.md), [JSON](E7_ALL_COMPARISONS.json) |
| 프레임별 결과 | [FRAME_RESULTS.csv](FRAME_RESULTS.csv), [E2](E2_FRAME_REVIEW.csv), [E5-B](E5B_FRAME_REVIEW.csv) |
| recording·metadata 결과 | [recording](E7_BY_RECORDING_ALL.csv), [metadata](E7_METADATA_STRATA_ALL.csv) |
| 실제 미완료·생략·차단 질문 | [QUESTION_STATUS.json](QUESTION_STATUS.json) |
| 발견한 누락과 보완 | [REVIEW_CORRECTIONS_KO.md](REVIEW_CORRECTIONS_KO.md) |
| Source 별도 기하 확인 | [SOURCE_CONTRACT_REVIEW.json](SOURCE_CONTRACT_REVIEW.json) |
| 원래 지시문 | [REQUEST_PLAN.txt](REQUEST_PLAN.txt) |

CSV의 `selected_hypothesis_for_pose`와 `GEO_name`은 그 행의 T/R에 실제 사용한 후보다. `GEO_free_name`은 자유 GEO 재선택 후보다. `held_identity` 및 `_held`에서는 서로 다를 수 있다. `twoD_value_scope=condition_output`은 개입 후 좌표의 2D 오차다. 매칭 실패/유효 감독점 부재는 `twoD_NA_reason`으로 표시하며 pose의 행을 제거하지 않는다. ΔT/ΔR은 조건의 baseline 대비 차이이고 E3/E4/E5의 baseline은 해당 단계 조건에 맞춘 identity다.

## 실행 환경과 명령

이번 환경은 Python(pallet-yolo26), torch2.1.1+cu118, ultralytics8.4.60, OpenCV4.9.0, numpy1.26.4이며 실제 CUDA 사용은 불가능하여 CPU에서 추론했다. 모델 추론량1,027회와 새 fit0회를 혼동하지 않는다. 최초 실험 명령은 [EXECUTION_COMMANDS.md](EXECUTION_COMMANDS.md)에 있다. 명령들은 repository root에서 실행한다.

```bash
export MPLCONFIGDIR=/tmp/pallet-mpl
PY=/home/minjae/anaconda3/envs/pallet-yolo26/bin/python
$PY -m scripts.research.pallet_pose_diagnosis_20260930_v1.review_analysis source
$PY -m scripts.research.pallet_pose_diagnosis_20260930_v1.review_analysis repair
$PY -m scripts.research.pallet_pose_diagnosis_20260930_v1.review_analysis statistics
$PY -m scripts.research.pallet_pose_diagnosis_20260930_v1.review_figures
$PY -m scripts.research.pallet_pose_diagnosis_20260930_v1.review_report
```

보완 명령은 기존 raw 결과가 있어야 하며 새 추론을 하지 않는다. 수정 전 산출물은 before_github_review에 보관한다. 최초 run/close/report 명령은 exclusive-create라 기존 결과를 덮어쓰지 않으며, 새 추론 재실행은 별도 namespace와 데이터 계약 확인이 필요하다. 역사적 report.py는 최초 초안을 만들기 위한 코드이고 최신 GitHub 검토판은 review_report.py가 생성한다.

시각화는 저장된 실제 사진과 예측/참조 좌표로 만들었다. AI 생성 이미지나 성능을 보이게 조작한 사진이 아니다. gallery는 natural99 전체이며 recording·ID 순이다. 본문의 4개 방향 사례는 각 집단의 ID 첫 프레임, E3는 recording별 ID 첫 프레임, T꼬리는 상위10장 전부다. [선택 기록](FIGURE_CASE_SELECTION.json).

## 검증 범위

원래 2,964행 독립 T/R 검산, 원본 입력 hash 보존, 이번 CSV 필드 수정과 E4 pose 재검산, Source256 기하 대조, E7 모집단/recording 수, 이미지 파일·Markdown 상대 링크 및 Git 게시 목록을 점검한다. PUBLICATION_VALIDATION.json은 게시 전 검증이고 GitHub 원격 반영 여부는 실제 push 뒤 확인한다. 최초 완료 manifest는 history에 보존하며 최신 공개 파일 SHA256은 PUBLICATION_MANIFEST.json에 있다.
'''
    write(C.DOC/'REPRODUCE_KO.md',reproduce)
    write(C.DOC/'README.md','''# 팔레트 clean→가림 6D pose 진단 — 2026-09-30

현재 자연 가림99장의 동일 평가 계약에서 R0·PRIOR1·FULL125를 비교하고, 후보 선택·검출·가림 RGB·좌표 오류·학습 타깃·참조 불확실성을 진단했다. 새 학습0회, 고정 모델 CPU 추론1,027회. GitHub 게시 전 누락된 분석과 CSV 표시를 보완했다.

- **[상세 결과 보고서](REPORT_KO.md)** — 관찰, 표, 비교 이미지, 경쟁 설명, 다음 비교
- **[자연 가림99장 전체 이미지 비교](GALLERY_NATURAL99.md)** — 개선·실패를 모두 확인
- **[입력과 실험 방법](INPUTS_AND_METHOD_KO.md)** — 집합, 모델 계보, 좌표, 참조, 평가 계약
- **[69개 비교의 recording 통계](STATISTICS_KO.md)** — CI, 분모, paired 변화
- **[지시문 누락의 보완 내역](REVIEW_CORRECTIONS_KO.md)** — 수정한 부분과 남은 한계
- **[재현 및 파일 안내](REPRODUCE_KO.md)** — 실제 명령, 파일 의미, 로컬 입력 요구

![동일 GEO 기준 모델별 T/R](figures/01_pose_overview.png)

FULL125의 natural99 중앙값은 T12.403→11.986cm, R5.218→4.411°이나 recording 신뢰구간은0을 포함한다. 일반화 성공을 선언하지 않는다. 가장 강한 남은 가설은 가림 RGB에서 보정 이득이 약해지고 W/D 선택과 결합한다는 것이다. 큰 T 꼬리에는 검출/매칭 실패가 집중한다. 다음 하나는 새 학습이 아니라 고정12pose의 clean→가림→clean36장 짝 비교이며 이번에는 계획만 작성했다.
''')
    history=C.DOC/'history';history.mkdir(exist_ok=True)
    if not (history/'RUN_MANIFEST_INITIAL_COMPLETION.json').exists():write(history/'RUN_MANIFEST_INITIAL_COMPLETION.json',C.read(C.DOC/'RUN_MANIFEST_FINAL.json'))
    audit=C.DOC/'INSTRUCTION_COMPLIANCE_AUDIT_KO.md';text=audit.read_text()
    if not text.startswith('> 역사적 기록'):
        write(audit,'> 역사적 기록: 아래는 보완 전 재감사 결과다. 이후 조치와 현재 상태는 [GitHub 게시 전 보완 기록](REVIEW_CORRECTIONS_KO.md)을 확인한다.\n\n'+text)
    print('REVIEW_REPORT_DONE',len(comparisons),flush=True)

if __name__=='__main__':main()
