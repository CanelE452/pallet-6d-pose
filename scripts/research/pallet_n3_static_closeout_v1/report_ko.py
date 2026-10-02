"""Readable Korean entry point for the detailed local closeout artifacts."""
from .compute import C,DOC
from .paper import md

def write_report():
    core=C.read(DOC/'CORE_RESULTS.json')['methods'];paired=C.read(DOC/'PAIRED_POSE_ANALYSIS.json');cost=C.read(DOC/'EXECUTION_COST.json');verify=C.read(DOC/'VERIFY_STATIC_RESULTS.json')
    nrows=[]
    for name,arm in [('N0','N0_BASE_REPLAY'),('N1','N1_SYM_ONLY')]:
        h=core[arm]['mean']['headline'];nrows.append([name,f"{h['pose_translation_cm_median']:.3f} / {h['pose_translation_cm_P90']:.3f}",f"{h['pose_rotation_deg_median']:.3f} / {h['pose_rotation_deg_P90']:.3f}",f"{h['pose_yaw_deg_median']:.3f} / {h['pose_yaw_deg_P90']:.3f}",h['pose_IoU3D_median'],h['pose_ADDsym_AUC_full'],'319/319'])
    prows=[]
    for b,d in paired['backbones'].items():
        for metric in ['translation_cm_median','translation_cm_P90','rotation_deg_median','rotation_deg_P90']:
            h=d['mean_of_seed_statistics'][metric];prows.append([b,metric,h['mean_seed_delta'],f"[{h['CI95'][0]:.3f}, {h['CI95'][1]:.3f}]"])
    text='''# N3 비리프터 마감 결과

기존 입력으로 계산 가능한 비리프터 항목을 계산·검산하고, 국문 v3의 별도 Markdown 복사본에 표와 관련 본문을 반영했다. 사람 검수·독립 참조·원고 소스가 필요한 항목은 남아 있다. 따라서 **계산 실행 완료와 논문 전체 완성은 구분**한다.

이번 신규 학습은 **0회**, optimizer update도 **0회**다. 리프터 작업, PDF/PPT 생성, 외부 업로드와 push는 하지 않았다.

## 바로 확인할 파일

- [결과 반영 원고](manuscript_ko_v3_static_closeout.md)
- [표·본문 변경 목록](PAPER_RESULTS_PATCH_KO.md) · [원고 칸별 상태](PAPER_GAP_MATRIX.md) · [원래 x의 처리](ORIGINAL_X_DISPOSITION.json)
- [검증 결과](VERIFY_STATIC_RESULTS.json) · [출처와 해시](SOURCE_BINDINGS.json) · [남은 x와 이유](REMAINING_X.md)
- [전체 표의 LaTeX·CSV·Markdown](generated_tables/) · [실행 명령](README_RUN.md)

## 계산·검산·원고 반영까지 완료

| 항목 | 이번 처리 | 반영 상태 |
| --- | --- | --- |
| N0/N1 6D | 2개 구성 × 3 seed × 319장 = 1,914행의 누락 PnP 계산 | 원고의 자세·절제 표에 삽입 |
| R0/P/N2/N3 | 기존 원시 좌표를 다시 평가, 26개 프레임 점수 배열 일치 | 기존 수치 유지 |
| ResNet 명세 | 실제 10-epoch CONSTANT-fold RGB, DSNT 디코더, checkpoint·receipt·folding 검증 | 모델 설명 정정 |
| 짝지은 불확실성 | 세 기반과 N0~N3 절제의 T/R/yaw 및 2D, 13세션·10,000회 bootstrap | 주 결과와 보조표 삽입 |
| 가림·재질 | 같은 확정 주석을 세 기반에 연결, 미분류191장 유지 | 층별 표 삽입 |
| 가시성 | 기존 수동 상태 71점 재사용, 기존 DEV 좌표 유지 | YOLO 가시성 표 삽입 |
| 정사각형 | 602점/600점 두 모드, 모든 방법의 분모 및 원시 값 검산 | 두 패널과 N3−R0/N3−N2 차이 삽입 |
| D/L/PoseFix | 같은319장·8코너 원시 좌표부터 회귀검산 | 비교표 삽입 |
| 자기학습 대안 | 공통128장으로 R0·학생·N3 모두 재검산 | 별도128장 표로 삽입 |
| 시간 측정 | DOPE/ResNet의 동일 계약 결과 재사용, YOLO만 추가 측정 | 환경별 비용 표 삽입 |

N0/N1은 `reuse.py`의 `CORE_ARMS` 조건 때문에 원시 코너를 읽고도 pose 계산만 건너뛰고 있었다. 기존 전역 상수나 원본 결과를 바꾸지 않고 별도 평가 경로로 보완했다. 다음은 seed별 통계의 평균이며, 각 seed의 자세 산출률은 모두319/319다.

'''
    text+=md(['방법','T 중앙값/P90 cm','R 중앙값/P90 도','Yaw 중앙값/P90 도','IoU3D 중앙값','ADDsym 전체 AUC','자세 산출'],nrows)
    text+='''
세 seed 원시 값과 N2−N0, N1−N0, N3−N2, N3−N1의 차이는 [자세·절제 결과](N0_N1_POSE_RESULTS.json)와 [절제 구간](ABLATION_POSE_UNCERTAINTY.json)에 있다. 과거 P를 통제 재현 N0로 바꾸어 부르지 않았다.

## 개선된 부분과 남은 한계

세 기반 모두 T·R 중앙값의 seed 평균은 낮아졌다. 아래 변화량은 **N3−Base**다. 중앙값의 차이와 프레임별 차이의 중앙값을 구분했으며, 구간은 같은 세션 재표집 안에서 세 seed 통계 차이를 평균해 계산했다.

'''+md(['기반','지표','변화량','95% 구간'],prows)+'''
![자세 변화와 구간](figures/pose_uncertainty.png)

ResNet의 T 중앙값 구간은0을 포함한다. DOPE의 회전 P90과 ResNet의 이동 P90은 악화했다. 구간이 유리하지 않은 결과도 그대로 보존했다. 이 분석은 반복 사용한 DEV의 사후 불확실성 분석이며, 독립 시험이나 물리적인6D 정확도 검증이 아니다.

![기반별 보정 전후](figures/backbone_results.png)

N3가 N2나 PoseFix보다 항상 우수하지도 않다.

- 정사각형 YOLO의 선언602점 모드에서 N3−R0는 중앙값−0.522px, PCK10+4.042pp다. 반면 N3−N2는 중앙값+0.055px, PCK10−0.332pp로 악화했다. 전체 보정 패키지의 이득과 대칭 감독의 추가 이득을 구분해야 한다.
- PoseFix의 조건부 코너 중앙값5.561px는 N3의5.778px보다 낮다. P90은 N3가 더 낮다. 서로 다른 입력·구조·손실을 가진 전체 방법 비교다.
- ResNet의 비항등 대칭 목표 선택0회와 C4 학습 행0은 실제 관측값이다. 치수 logits 민감도는 입력 경로가 활성화됐다는 증거이며, 치수의 독립적인 정확도 이득을 입증하지 않는다.
- 결과는 세 기반에 각각 학습한 N3의 적용 근거다. 같은 가중치를 임의 백본에 전이하거나 모든 가림 조건에서 강건하다는 결론으로 확대하지 않는다.

## 실제 모델과 주석의 정리

ResNet은 **10-epoch CONSTANT-fold RGB/DSNT 모델**이다. CONSTANT는 치수 조건을0으로 고정했다는 뜻이며 이미지 출력이 상수라는 뜻이 아니다. 고정 FiLM을 마지막 convolution에 접어 기본 모델은 RGB만 받는다. N3가 이미지 특징과 W,D,H를 함께 입력받는다. 원본 protocol의 옛 epoch60·argmax·loss 설명은 수정하지 않고 [별도 정정 기록](PROTOCOL_CORRECTIONS.md)에 원본 해시와 실제 값을 연결했다. 기본 모델 선택 이력의 확인 범위도 명시했다.

가림 분류 차이는 `eval_pallet09:1778653661653195264` 한 프레임에서 확인됐다. 과거 SPLIT_LOCK은 중간, 현재 직접 검수 manifest는 심함이다. 변경 시각과 사유는 미확인이고, 개수를 맞추려고 주석을 수정하지 않았다.

기존 FINAL_V2의 수동 상태 검수에서 직접 가시66점과 외부 가림5점을 찾았다. 프레임 이미지 해시와 코너 ID를 대조해 **상태만** 재사용했다. 재클릭 좌표는 기존 DEV 참조와 달라 교체하지 않았다. 따라서 가시성 표는 기존 참조 오차의 층별 분석이며, 새 수동 클릭에 대한 정확도 표가 아니다. PnP 보조 검수 이력과 사전 예측 노출 미확인을 기록했다.

[주석 감사](STATIC_LABEL_AUDIT.md) · [모델·오차를 가린 검수 자료](review/index.html). 새 사람 검수를 완료했다고 기록하지 않았다.

![가림별 전체 분모 곡선](figures/occlusion_pck.png)

정사각형의 화면 밖 두 점은029710/코너0의(-42,289), 029844/코너4의(-7,310)이다. 이를 포함하는602점과 제외하는600점을 별도로 보고했다.119장은 한 세션·한 치수다. 독립적인 표준6D 참조가 없어 T/R은 x, 세션 간 일반화 CI는 NA다. 같은 코너로 PnP 정답을 새로 만들지 않았다.

모든 기반·seed에서 보정 전후 **성공 프레임 ID 집합이 같았다**. 신규 자세 실패와 복구는0이다. DOPE의109개 실패는 전체319장 AUC·산출률 분모에 남았다. [프레임별 가설 대조](POSE_HYPOTHESIS_ANALYSIS.md)에는 R/T/yaw, 가로·깊이 가설 변화, 평가 대칭 변화와 코너 이동을 연결했다. 가설 변화와 오차 감소의 동반 관측을 인과 원인으로 단정하지 않는다.

![정지 이미지 사례](figures/static_examples.png)

설명용 예시는 seed1과 기존 확정 등급 ID만 사용했다. 개선·무차이·악화 집단마다 ID 사전순 첫 영상을 고른 사후 예시이며, 정량 평가 대상을 바꾸지 않았다. 실제 예측 코너를 그렸고 PnP 재투영점은 사용하지 않았다.

## 결과가 있지만 아직 삽입하지 않은 항목

- 같은 국문 v3의 LaTeX 원본을 찾지 못했다. 정확한 table label별 `.tex` 조각은 준비했지만, 그 원고 소스에 삽입했다고 표시하지 않았다. 국문 Markdown 복사본에는 실제로 반영했다.
- DOPE/ResNet의 가시성 세부 결과는 `CROSS_BACKBONE_SUBGROUPS.json`에 있다. 본문 가시성 표에는 YOLO를 넣었고, 세 기반의 가림 결과는 보조표로 넣었다.

## 남은 입력과 제외 범위

| 항목 | 상태 | 필요한 근거 |
| --- | --- | --- |
| 미분류191장 | x / 사람 검수 | 직접 검수 레이블. 복사된66개 split 등급은 직접 검수 출처가 없어 승격하지 않음 |
| 가시성 미확인2428점 | x / 사람 검수 | 점별 상태. 유효 좌표를 가시성으로 대체하지 않음 |
| 정사각형 가림·가시성 | x / 사람 검수 | 승인된 상태 레이블 |
| 정사각형6D | x / 별도 참조 | 독립적인 표준 자세 참조와 좌표계 근거 |
| 원래319장 학생 비교 | BLOCKED_CONTRACT | 공통 비노출 계약 불성립.128장 별도 표로 대체 |
| 정사각형 일반화 CI | NA | 단일 세션에서 정의할 수 없음 |
| 과거 S0/S1와 새 정사각형의 이력 대응 | x / 출처 확인 | 개체·촬영·학습 노출의 완전한 대응 |
| D/L/PoseFix 동일 환경 속도 우월성 | 주장하지 않음 | 이번 표는 정확도 비교. 같은 경계의 비용 근거 없음 |
| 국문 v3 LaTeX와 기존 정적 도식3개 | 소스 대기 | 다른 원고로 대신하지 않음 |
| 리프터 | OUT_OF_SCOPE_USER | 이번 작업에서 전부 제외. 해당 장·표 보존 |

이미 있는 입력으로 가능한 필수 계산은 마쳤다. 남은 위 항목 때문에 새 학습이나 참조 생성, 추가 모델 선택을 시작하지 않았다. 리프터 제외는 실험 실패로 세지 않는다.

## 실행·검증과 실제 비용

'''
    text+=md(['작업','벽시계 초'],[[k,cost[k]] for k in ['core','paired_pose','pose_traces','extra_uncertainty','comparator_student_raw_regression','YOLO_runtime_wall_seconds']])
    text+=f'''
새 GPU 본측정은 YOLO의3경로×130=390회, 준비60회다. 보정 단독 계측을 위한 untimed 기본 모델 호출150회가 별도로 있다. DOPE/ResNet의 기존780개 본측정은 재사용했다. 같은 RTX3080이지만 YOLO의 라이브러리 환경은 달라 별도 패널로 공개했다. 새 YOLO 중앙시간은 Base11.195ms, N3 전체15.138ms, 보정만3.432ms다.

초기 상태 기록부터 최종 검증까지 작업 벽시계 시간은 약 **{cost['overall_wall_minutes']:.1f}분**이다. 위 계산 시간 합계와 전체 작업 시간은 다르다. CPU processor 시간은 별도 계측하지 않아 NA이며, 벽시계 시간을 CPU 사용량으로 바꾸어 표시하지 않았다.

기존 검증 테스트는 최초40개 통과·3개 실패였다. 실패 원인은 격리 worktree의 정사각형 주석 경로 부재였다. 원시 소유 checkout에서 해시가 같은119개 주석을 읽기용 symlink로 연결한 뒤, 실패3개 모두 통과했다. 숫자·허용오차·평가 대상을 바꾸지 않았다. 합계 **43개 테스트 통과**, 표 숫자 **{verify['table_numeric_checks']}개 대조 통과**, 기존 보호 파일 **{verify['protected_original_files']}개 해시 불변**을 확인했다.

구현 중 기존 JSON의 `per_seed`와 정규화 payload 구조를 맞추는 오류를 수정했다. 이미 저장된 완료 계산은 재사용했으며 성능이 불리하다는 이유로 학습·측정을 반복하지 않았다. GPU 경고에 대응해 라이브러리를 업그레이드하지도 않았다.

[검증 로그](TESTS_INITIAL.txt) · [경로 복원 후 재검증](TESTS_AFTER_INPUT_PATH_REPAIR.txt) · [실행 비용 JSON](EXECUTION_COST.json) · [실행 명령](README_RUN.md)
'''
    (DOC/'FINAL_REPORT_KO.md').write_text(text)

if __name__=='__main__':write_report()
