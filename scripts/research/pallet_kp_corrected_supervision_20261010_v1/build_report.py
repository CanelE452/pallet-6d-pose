"""Build the Korean report from completed public artifacts only; no new fits."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DOC = REPO / '_docs/experiments/pallet_kp_corrected_supervision_20261010_v1'
QUESTION = '가림 판단이 일부 틀려도 다른 유효 대응점으로 자세를 구하고 자기 가림 코너를 재투영했을 때, 기존 단순 대조보다 실제 위치·회전이 좋아졌는가?'


def load(name):
    return json.loads((DOC/name).read_text())


def number(value):
    return 'NA' if value is None else f'{value:.5f}'


def table(header, rows):
    return ['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join('---' for _ in header) + ' |'] + [
        '| ' + ' | '.join(map(str, row)) + ' |' for row in rows]


def main():
    metrics, training, runtime = load('METRICS.json'), load('TRAINING_COMPLETION.json'), load('RUNTIME.json')
    stress = load('REAL_STRESS_EXECUTION.json')
    stress_statistics = load('REAL_STRESS_STATISTICS.json')
    mask_n3 = load('MASK_OUTCOME_N3.json')['strata']['combined']['CORRECTED_IMAGE_ROLE']
    corner_n3 = load('CORNER_N3_AND_REPROJECTED_SELF.json')
    assert metrics['complete'] and training['complete'] and runtime['complete'] and stress['complete']
    assert runtime['execution']['pipeline_calls_complete'] == 600
    assert metrics['population']['frames'] == 245 and training['formal_updates'] == 9000
    methods = metrics['methods']
    chosen = ['BASE','N3_SUBPIX','CORRECTED_GEOMETRY_ONLY','CORRECTED_IMAGE_NO_ROLE',
              'CORRECTED_IMAGE_ROLE','CORRECTED_IMAGE_ROLE_NO_MASK_ROBUST',
              'CORRECTED_IMAGE_ROLE_STANDARD','CORRECTED_IMAGE_ROLE_POINT_LINE']
    primary = metrics['contrasts']['CORRECTED_IMAGE_ROLE_minus_N3_SUBPIX']['common_operational']
    lines = [QUESTION, '',
        '**이번 고정 실험에서는 아니요.** 쉬움153장·중간92장, 총245장에서 수정 감독으로 학습한 IMAGE_ROLE의 평균 위치·회전·ADDsym 오차는 **15.20239cm·14.56122°·23.77090cm**였다. 기존 N3→cornerSubPix는 **9.75455cm·10.91484°·17.02865cm**였다. 새 관측 공급과 자세 산출률은 회복됐지만, 위치와 회전은 악화됐다. 이 수치는 기존 **GEOMETRIC_PROXY** 참조로 계산한 실제 실행 결과이며 독립 실측 물리 자세가 아니다.', '',
        '기존 N3 좌표의 예측 자기 가림 제외+일반 PnP는 평균8.95157cm·10.61764°로 작은 개선을 보였다. 같은 좌표의 강건 PnP는 이 대조보다 나빴고, 세 가지 새 경계 모델도 기존 N3→cornerSubPix를 이기지 못했다. 아래 표는 유리한 한 경로만 고르지 않고 지정한 모든 비교를 보존한다.', '',
        '## 범위와 비교 기준', '',
        '사용자의 최신 요청 “319로하지말 고 쉬움 중간만 해 가림 어려움하지말고”에 따라, 새 실사 추론·자세 평가·오판 실험·시간 측정·이미지는 쉬움/중간245장으로 제한했다. 어려움/심한 가림74장은 새 경로에 실행하지 않았다. [COHORT.json](COHORT.json)에 포함245·제외74의 ID와 원본 영상 SHA가 있다. 최신 기존 분류는 clean153/moderate92/severe74이며 과거151/87/81과42개가 다르다. [COHORT_SOURCE_LABELS.json](COHORT_SOURCE_LABELS.json)은 기존 분류의 바이트 그대로를 보존한다.', '',
        'moderate는 기존 UI의 중간 분류다. 일부 부분 가림이 포함될 수 있고, 별도 가림 면적 마스크나 분류 기준의 독립 인증은 없다. “가림이 전혀 없는245장”이라고 부르지 않는다. 새 수동 어노테이션을 만들거나 성능을 보고 등급을 다시 정하지 않았다. 과거319장 실험은 기존 문서에 그대로 남고 이번245장 결과와 혼합하지 않는다.', '',
        '초기 RGB 추정기와 N3 가중치·설정은 고정했다. 수정 감독은 기존 P0 RGB/실제 USD 형상/마스크/깊이에서 만들었다. 박스 hull 마스크나 새 RGB는 사용하지 않았다. source split은 기존768/128/128이며 같은 장면의 분할을 바꾸지 않았다. 이번 실제 학습은 추가 승인된3×3000회, batch16, seed1, 같은 초기 텐서와 같은3000×16 배치 순서, 마지막 체크포인트만 사용했다. 원래 실행된9000회와 이번9000회는 별도이며 총 formal18000회다. 이번 throwaway update는0회다.', '',
        '## 같은 대상의 전체 운용 결과', '',
        '각 행은245장 전체를 유지한다. 기본 반환은 Base 전체 좌표와 초기 자세를 반환한 것이며 새 자세 산출로 세지 않는다. 완전 실패는 별도로 남긴다. 이번 새6경로는 모두245장의 최종 출력이 존재해, 공통 산출집합도245장이다.', '']
    rows = []
    for name in chosen:
        item = methods[name]; value=item['metrics']['operational']
        rows.append([name,item['total_frames'],item['new_pose_estimated'],item['fallback_used'],item['no_pose'],
                     number(value['translation_cm']['mean']),number(value['rotation_deg']['mean']),number(value['ADDsym_cm']['mean'])])
    lines += table(['방법','전체','새 자세','Base 반환','완전 실패','위치 평균(cm)','회전 평균(°)','ADDsym 평균(cm)'],rows)
    lines += ['', 'IMAGE_ROLE의 기본 반환 25장은 관측 수 부족 12장과 합의 부족 13장이다. 수치 실패·기하 배치 탈락·다중 가설은 원행의 reason, operation_counts, returned_solutions, invalid_solutions에 구분해 기록했다. 새 자세를 계산했다는 사실과 정확한 자세를 얻었다는 판단도 구분한다.', '',
        '쉬움과 중간을 따로 비교해도 새 모델의 평균은 개선되지 않았다. 각 범위는 지정한 모든 프레임과 기본 반환을 포함한다.', '']
    rows=[]
    for stratum in ('easy','medium'):
        for name in ('N3_SUBPIX','CORRECTED_IMAGE_ROLE'):
            item=metrics['strata'][stratum]['methods'][name]; value=item['metrics']['operational']
            rows.append([stratum,name,item['total_frames'],item['new_pose_estimated'],item['fallback_used']]+[
                number(value[k]['mean']) for k in ('translation_cm','rotation_deg','ADDsym_cm')])
    lines += table(['범위','방법','전체','새 자세','기본 반환','위치(cm)','회전(°)','ADDsym(cm)'],rows)
    lines += ['', 'BASE/N3_SUBPIX의 새 자세0은 고정 과거 대조의 상태 표시다. 관측 실패0이라는 뜻이 아니다. IMAGE_ROLE_POINT_LINE은 같은 IMAGE_ROLE 관측의 별도 절제이며, 점 생성에 사용한 선을 다시 중복 관측으로 넣지 않는다.', '',
        '## 평균·분산·표준편차·중앙값·P90·최대값', '',
        '표본분산/표본표준편차는 ddof=1이고 분위수는 NumPy linear다. cm²/degree²는 분산 단위다. 아래 값은 원행에서 계산했고 [METRICS.csv](METRICS.csv)에24방법×전체/쉬움/중간×전체운용/새자세의 모든 값이 있다.', '']
    rows=[]
    for name in chosen:
        for metric in ('translation_cm','rotation_deg','ADDsym_cm'):
            item=methods[name]['metrics']['operational'][metric]
            rows.append([name,metric,item['n']]+[number(item[k]) for k in ('mean','sample_variance','sample_std','median','P90','max')])
    lines += table(['방법','지표','n','평균','표본분산','표본SD','중앙값','P90','최대'],rows)
    lines += ['', '## 같은 프레임의 차이와 불확실성', '',
        '차이는 후보−대조이므로 양수가 악화다. 기존에 저장한10000×13 세션 multiplicity 행렬을 그대로 재사용했다. 새 bootstrap draw/seed를 만들지 않았다. SD는 오차의 표준편차이고 CI는 대응된 평균 차이의 세션 bootstrap 구간이다. 여러 절제의 CI에 다중 비교 보정은 없으며, 반복해서 본 DEV/diagnostic source-test를 새 독립 holdout이라고 부르지 않는다.', '']
    rows=[]
    for scope in ('common_operational','candidate_new_pose'):
        comparison=metrics['contrasts']['CORRECTED_IMAGE_ROLE_minus_N3_SUBPIX'][scope]
        for metric,value in comparison['metrics'].items():
            rows.append([scope,comparison['common_frames'],metric,number(value['mean_delta']),
                         '['+', '.join(number(x) for x in value['CI95'])+']',value['improved_frames'],value['worsened_frames']])
    lines += table(['대응 집합','n','지표','평균 차이','95% CI','개선 프레임','악화 프레임'],rows)
    lines += ['', '새 자세220장만 비교해도 위치 평균 차이+6.03958cm, 회전+4.01794°다. 기본 반환25장을 제외해 결론을 좋게 만든 것이 아니다. 큰 오차와 회전 약90° 사례도 원행·P90·최대값에 남겨 뒀다.', '',
        '## 필수 무학습 여섯 비교', '',
        '아래12행은 같은245장의 기존 실제 원행에서 다시 집계한 결과다. 이 표를 위해 추론/PnP를 반복 실행하지 않았으며 각 원행 출처와 별칭은 [HISTORICAL_FILTERED_ROWS.jsonl.gz](HISTORICAL_FILTERED_ROWS.jsonl.gz)와 [METRICS.json](METRICS.json)에 연결되어 있다. ORACLE_NOSELF와 ORACLE_VISIBLE은 사람 정보가 필요한 진단으로 배포 방법이 아니다. NO_MASK_STANDARD는 새 표준 솔버이고 고정 N3_SUBPIX는 기존 초기 자세 경로이므로 둘은 동일 출력이라고 가정하지 않는다.', '']
    rows=[]
    for arm in ('BASE','N3_SUBPIX'):
        for suffix in ('NO_MASK_STANDARD','NO_MASK_ROBUST','GEOM_NOSELF_STANDARD','GEOM_NOSELF_ROBUST','ORACLE_NOSELF_ROBUST','ORACLE_VISIBLE_ROBUST'):
            item=methods[arm+'_'+suffix];value=item['metrics']['operational']
            rows.append([arm,suffix,item['new_pose_estimated'],item['fallback_used'],item['no_pose'],
                         number(value['translation_cm']['mean']),number(value['rotation_deg']['mean']),number(value['ADDsym_cm']['mean'])])
    lines += table(['좌표','마스크/솔버','새 자세','기본 반환','실패','위치(cm)','회전(°)','ADDsym(cm)'],rows)
    lines += ['', '가림을 사용하지 않는 강건 PnP는 빠지지 않았다. N3_NO_MASK_ROBUST의11.59437cm·14.17031°는 단순 N3→cornerSubPix의9.75455cm·10.91484°보다 나쁘다. 예측 가림을 제외하면 같은 강건 솔버에서10.61539cm·14.00206°로 조금 낮아지지만 여전히 단순 대조를 이기지 못한다. 사람 마스크도 강건 PnP의 우수성을 보장하지 않았다.', '',
        '## 가림 판단·관측 선택·강건 솔버·재투영의 역할', '',
        '가림 판단: 일부 잘못된 마스크에서도 새 자세를 구했다. 마스크가 한 점 다르거나 재추정 뒤 바뀌었다는 이유로 프레임을 삭제하지 않는다. IMAGE_ROLE의 초기/최종 자기 가림 집합이 바뀐14장도 성능 집계에 그대로 남았다. [MASK_OUTCOME_N3.json](MASK_OUTCOME_N3.json)은 고정 N3 대조에 대한 성능과 마스크의 맞고 틀림을 교차 집계한다. [METRICS.json](METRICS.json)의 기존 mask_known_relation은 초기 Base 대조이며 두 기준을 혼동하지 않는다.', '',
        '관측 선택: 감독을 고치기 전 모델의 no-match 편향을 줄여 관측 공급과 산출률을 회복했다. 그러나 IMAGE_ROLE에서 참조상8px 이내 대응이4개 미만인100장 중75장에서도 솔버가 수치적으로 새 자세를 반환했다. 관측 합의가 참조 정확도와 같지 않음을 보여 준다. [POSTHOC_CORRESPONDENCE_ROWS.jsonl.gz](POSTHOC_CORRESPONDENCE_ROWS.jsonl.gz)에 남은 올바른 점의 ID·개수·배치·최종 inlier·잘못된 inlier를 공개했다. 정답이 부족하거나 없는 경우는 unknown으로 표시하며 0 오차로 채우지 않는다.', '',
        '강건 PnP: 각 좌표/K/치수/4-ID 가설은 마스크 사이에서 재사용했다. 최대8개 코너에서 최대70개의4점 부분집합, 등록된 두 치수 가설이 있으면 최대140개 생성 호출이다. 각 해법의 다중해를 보존하고 수치/배치 검사를 수행한다. 평면 IPPE와 비평면 SQPnPGeneric, residual8px, top3 refit은 그대로다. 단순히 기존6점 조건을4로 바꾸지 않았다. 기존 [SOLVER_CHECKS.json](../pallet_observation_refiner_20261009_v1/SOLVER_CHECKS.json)의4/5점·다중해 검사를 보존했고 이번 실사 평가에서도 같은 솔버를 사용했다.', '',
        '재투영: 새 R,t가 나오면 자기 가림으로 제외한 초기 좌표를 최종 재투영으로 교체했다. 그 점을 독립 관측으로 다시 fit하지 않았다. IMAGE_ROLE에서216프레임에267개 H 좌표를 실제 교체했다. 원행의 solver fit_input_ids와 H는 서로 겹치지 않으며 public 검산이 최종 R,t의 투영 좌표와 표시 좌표를 확인한다. 재투영 자체가 관측을 더 만드는 것은 아니며 새 자세가 나쁘면 숨은 점도 나빠질 수 있다.', '',
        '영상/역할: 기하만의15.77039cm·20.66581°보다 영상 모델이 회전을 줄였으나 N3보다 좋지는 않았다. 역할 포함은 역할 없음보다 산출률220/245 대193/245를 높였지만 위치 평균은15.20239 대14.20976cm로 더 나쁘다. 역할이 이번 목표에 필수라고 주장할 수 없다. 점+선은 같은 관측에서 산출률245/245와11.66459°를 얻었으나13.83273cm로 N3보다 위치가 나빴다. 이 결과로 새 가림 분류기, 대형 분할망, seed 반복 학습이 필요하다고 결론 내리지 않는다.', '',
        '## 직접 가시점 손상과 숨은 점 오차', '',
        '다음 before는 각 방법의 초기 Base 좌표에 대한 동일 참조상 오차이고 after는 최종 표시 좌표다. N3와 비교하는 자세 표와 기준이 다르다. 직접 가시점과 SELF, 알고리즘 H 재투영 집합도 서로 구분한다.', '']
    damage=metrics['visibility_damage']['CORRECTED_IMAGE_ROLE']['operational']
    rows=[]
    for label in ('DIRECT_VISIBLE','SELF_OCCLUDED','ALGORITHM_REPROJECTED_IDS'):
        item=damage[label]
        rows.append([label,item['corners'],item['frames'],number(item['before']['mean']),number(item['after']['mean']),
                     number(item['before']['median']),number(item['after']['median']),item['good5_to_bad10']])
    lines += table(['집합','대응 코너','프레임','전 평균(px)','후 평균(px)','전 중앙값','후 중앙값','≤5px→>10px 손상'],rows)
    lines += ['', '고정 N3 좌표를 before로 쓰는 동일 Base-phase 검산도 공개했다. N3와 Base의 참조 phase가 다른 2프레임은 [PHASE_ALIGNMENT_AUDIT.json](PHASE_ALIGNMENT_AUDIT.json)에 ID를 기록했으며 유리한 참조로 다시 고르지 않았다.', '']
    rows=[]
    for label in ('DIRECT_VISIBLE','SELF_OCCLUDED'):
        item=corner_n3['visibility_damage_vs_N3']['combined']['CORRECTED_IMAGE_ROLE'][label]
        rows.append([label,item['corners'],number(item['before_N3']['mean']),number(item['after']['mean']),
                     number(item['before_N3']['P90']),number(item['after']['P90']),item['good5_to_bad10']])
    lines += table(['집합','코너','N3 평균(px)','최종 평균(px)','N3 P90','최종 P90','≤5px→>10px 손상'],rows)
    actual=corner_n3['actual_reprojected_SELF_only']['combined']['CORRECTED_IMAGE_ROLE']
    lines += ['', f"사람 SELF 중 **실제로 새 자세로 재투영한 {actual['evaluable_reprojected_SELF_corners']}개**만 비교하면 Base 평균 {number(actual['before_BASE']['mean'])}px, 같은 phase N3 평균 {number(actual['before_N3_same_BASE_phase']['mean'])}px, 재투영 뒤 평균 {number(actual['after_reprojection']['mean'])}px다. 사람 SELF 335개 중 새 자세가 없던 37개와 예측 H로 선택되지 않은 51개는 실제 재투영 집합과 분리했다. 예측 H로 제외된 코너는 새 자세가 나오면 모두 재투영으로 대체하며, 인간 SELF를 예측에서 놓친 경우는 마스크 오판으로 기록한다.", '']
    lines += ['', 'SELF 중앙값7.303→5.943px와 H 중앙값6.974→5.009px만 보면 좋아 보이지만 평균/P90과 최종 자세는 악화됐다. 숨은 점 오차만 좋아졌다고 위치·회전 개선을 주장하지 않는다. 직접 가시 코너10개는≤5px에서>10px로 손상됐다. 참조가 없는 숨은 점과 적용되지 않은 재투영은 별도 NA로 남긴다.', '',
        '## 실제 영상에서1·2점 오판 진단', '',
        '같은245장·Base/N3 좌표로2개의 oracle 진단 family×7조건×2좌표×245장=6860경로를 한 번 실행했다. HUMAN_NOSELF는 사람 H만 제외한 정확한 마스크 대조이고, KNOWN_ACCURATE_OBSERVATIONS는 사람 DIRECT·유효 참조·8px 이내 점만 남긴 관측 상한 대조다. 두 family를 배포 성능으로 부르지 않는다.', '',
        '조건은 오류 없음, 올바른 점1/2개 제거, 잘못 남기는 점1/2개, 두 오류1+1/2+2다. 사람이 SELF라고 한 점도 좌표는 정확할 수 있으므로 첫 family의 SELF 잔류를 무조건 픽셀 이상치라고 부르지 않는다. 두 번째 family는 실제 참조 오차>8px인 유효 대응을 남겨 픽셀 이상치 진단을 수행한다. SHA 기반 ID 선택을 고정했고 손상/잔류 수가 부족하면 변형 불가로 기록한 뒤 원 anchor 경로를 계산했다. 변형 불가를 프레임 실패로 세지 않는다.', '',
        '[REAL_STRESS_PROTOCOL.json](REAL_STRESS_PROTOCOL.json), [REAL_STRESS_ROWS.jsonl.gz](REAL_STRESS_ROWS.jsonl.gz), [REAL_STRESS_EXECUTION.json](REAL_STRESS_EXECUTION.json), [REAL_STRESS_STATISTICS.json](REAL_STRESS_STATISTICS.json)에 실제 제거/잔류 ID, 남은 올바른 대응 수·2D/3D 배치, inlier 정확도, 위치·회전·ADDsym와 산출률이 있다.490개 동일 좌표 bank의 가설을14마스크 사이에 공유했고 재투영점을 관측으로 다시 넣지 않았다. 과거 실제490행의1점 오류 진단과 과거 합성128scene의2점 진단은 별도로 보존했다.', '',
        '실제6860경로 중 변형 불가1799행(HUMAN829/KNOWN970)은 그대로 남았다. KNOWN 정확 대응 상한의 N3 anchor는169새자세/76반환, 평균9.11758cm·9.75318°였으나 이는 참조로 올바른 점을 고른 oracle다. 실제 픽셀 이상치1/2개를 남기는 변형 가능 집합에서는 최종 잘못된 inlier가 평균0.62130/1.43885개 남았다. 강건 솔버가 항상 이상치를 제거한다는 가정은 맞지 않았다.1점/2점 제거 집합은 각각225/217장으로 서로 다르므로 서로의 평균만 비교해 제거 효과라고 부르지 않는다.', '',
        '## 수정 감독과 실제 학습', '',
        '원래 감독은 실제 물리 wire의 float32 seam과 무한 depth/no-match 처리에서 유효 경계를 NONE으로 만드는 문제가 있었다. 실제 USD의90° 공유 면 wire, 원래 마스크/depth 정책, 유한 첫 표면과 camera-Z 앞면 검사로 수정했다. 실제 geometry/support가 인증되지 않은 edge[1,3,5,7]은 IGNORE다. 원래 source feature/order/가중치/영상을 수정하지 않았고 다른 GLB의 물리 경계를 새로 인증했다고 주장하지 않는다.', '',
        '최종 READY train POS26049/NONE11780/IGNORE26683, calibration4465/1974/4313, source-test4497/1966/4289다. depth 복구13757ray는 이전 감독 준비 단계의 실제 실행이며 이번 학습에서 재실행하지 않았다. [수정 감독 보고서](../pallet_kp_supervision_repair_20261010_v1/RESULT_KO.md)의 보존된 실패·0-update 검증·triangle witnesses·입력 SHA를 연결한다. 기존 RGB에서 감독을 만들 수 있어 새128/1024scene RGB를 생성하지 않았다.', '',
        '모델은 같은5890-parameter 소형 대응 head다. GEOMETRY_ONLY는 영상 채널을0으로 하되 기하적 역할 정보를 유지하고, IMAGE_NO_ROLE은 역할 채널을0으로 하며, IMAGE_ROLE은 모두 사용한다. 감독은65offset bin+NONE의66-way soft CE이고 IGNORE gradient는0이다. 가림 종류 분류 정확도를 통과해야 후단 평가하는 gate는 없다. 원래66-way MAP→선 fit→코너 교점을 유지했고 match-mass decoder나 치수 prior를 추가하지 않았다.', '',
        f"이번 실제 formal 업데이트는{training['formal_updates']}, batch 노출{training['formal_RGB_exposures']}, source probe head call{training['source_probe_head_calls']}/이미지노출{training['source_probe_image_exposures']}, 전체 head forward{training['total_head_forward_calls']}회다. 학습 wall은{number(training['wall_seconds'])}초이며 전체 배포 latency가 아니다. [FORMAL_UPDATE_ROWS.jsonl.gz](FORMAL_UPDATE_ROWS.jsonl.gz)의9000행으로3모델의 같은 배치 순서·각3000회·학습률과 유한 loss를 검산할 수 있다. [TRAIN_LOGS.jsonl.gz](TRAIN_LOGS.jsonl.gz)는 실제 source 곡선이고 [CHECKPOINT_METADATA.json](CHECKPOINT_METADATA.json)은 마지막 가중치 SHA와 설정을 연결한다.", '',
        'IMAGE_ROLE source-test positive 채택률은97.1759%, 채택된 positive 평균 위치 오차0.80673px, NONE false acceptance1.62767%다. 이 source-test는 감독 결함 진단에 이미 사용했으므로 독립 holdout 성능이 아니다. 합성 대응 학습이 작동해도 실사 자세 개선을 보장하지 않는다는 결론을 이번245장 결과가 보여 준다.', '',
        '## 실제 전체 경로 시간', '',
        '쉬움/중간에 속하는 고정26장(13세션×2장)으로4경로를 각각 warmup20회+26장×5반복, 총600회 실행했다. 이 패널도 새 성능을 보기 전에 고정했다. GPU 동기화된 RAM 원본 BGR→초기 RGB 추정기→N3/cornerSubPix 또는 새 관측→모든 필요한 초기 자세→fresh finite-subset 강건 PnP→H 재투영/표시/metadata 반환까지 포함한다.', '',
        '모델·체크포인트 로드, RGB 파일 decode, 정답/기준 읽기, parity 검사, durable journal, 자원 snapshot은 구간 밖이다. 관측 cache replay나 기존 시간의 합산을 사용하지 않았다. 다른 연구/학습/시간 측정 process와 겹치지 않는 자원 검사를 수행했고 기존 desktop rustdesk만 명시적 예외로 보존했다. 각600출력은 저장된 자세/좌표와1e−7 parity, center/score/box/candidate metadata 보존, GT canary를 통과해야 공식 시간이 된다.', '']
    rows=[]
    for arm in runtime['arms']:
        item=runtime['summaries'][arm]['full']
        rows.append([arm,item['n']]+[number(item[k]) for k in ('mean_ms','sample_variance_ms2','sample_std_ms','median_ms','p90_ms')])
    lines += table(['전체 경로','측정 n','평균(ms)','표본분산(ms²)','표본SD(ms)','중앙값(ms)','P90(ms)'],rows)
    lines += ['', '[RUNTIME_ROWS.jsonl.gz](RUNTIME_ROWS.jsonl.gz)은 warmup을 포함한 실제600원행이며 [RUNTIME.json](RUNTIME.json)에는 단계별 실제 시간·forward/PnP 호출수·환경·자원 snapshot이 있다. [RUNTIME_ADAPTER_RECEIPT.json](RUNTIME_ADAPTER_RECEIPT.json)은 학습·735관측·1470geometry·245cohort·26panel의 SHA를 연결한다.', '',
        '## 실제 이미지와 수치 그림', '',
        '사례는 새 정확도를 보기 전에 쉬움/중간 각각3개, 정렬된 세션/ID 규칙으로 선정했다. 유리한 새 결과를 골라 대표 사례로 쓰지 않았다. [VISUAL_CASE_PROTOCOL.json](VISUAL_CASE_PROTOCOL.json)의6개 영상에 N3/새 IMAGE_ROLE을 각각 표시한12패널이다. 캡션의 위치/회전/상태와 좌표는 실제 원행에서 읽었다. 청록은 초기 좌표, 노랑은 선택한 관측, 초록은 최종 inlier, 보라 점선은 기존 proxy 참조, 주황 실선은 최종 자세 전체 투영, 주황 사각형은 실제 교체한 H 표시 좌표다. 전체 자세 투영선이 모든 native 출력 좌표를 대체했다는 뜻은 아니다. 원본 RGB는 수정하지 않았다.', '',
        '![245장 실제 자세 성능과 산출률](figures/01_pose_overview.png)', '',
        '![같은 프레임의 평균 차이와 세션95%구간](figures/02_paired_deltas.png)', '',
        '![올바른 대응점·inlier·가림 오판과 자세 성능](figures/03_correspondence_mask.png)', '',
        '![수정 source 감독의 실제 학습과 실사 전이](figures/04_learning_transfer.png)', '',
        '![직접 가시점 손상과 숨은 점 재투영 오차](figures/05_visible_hidden_damage.png)', '',
        '![쉬움과 중간의 실제 영상12패널](figures/06_real_cases.png)', '',
        '![기존1점 가림 오판 원행의245장 범위 재집계](figures/07_historical_mask_stress.png)', '',
        '![전체 경로600회 실측 분포](figures/08_runtime.png)', '',
        '![실사6860경로의1점2점및동시오류진단](figures/09_real_mask_stress.png)', '',
        '## 중단·수정·검산·게시', '',
        '관측735개와 geometry1470개는 정답을 읽기 전에 봉인했다. 첫 채점 시 cached frame ID `session__stem`과 공개 ID `session:stem`의 연결 검사에서 멈췄다. 모든 geometry는 이미 저장됐고 그 전 실제 cached reference/채점은0이었다. [EVALUATION_INTERRUPTION.json](EVALUATION_INTERRUPTION.json)에 실패를 남기고 [REFERENCE_ID_MAPPING.json](REFERENCE_ID_MAPPING.json)의 원본 image path+session319일대일 mapping으로 채점만 이어갔다. Detector/head/PnP/optimizer를 재실행하지 않았고 실패 전 geometry SHA는 그대로다.', '',
        '실패 직전 프로세스의 wall과 SciPy residual callback counter는 저장되지 않아 NA다. 이를0이나 추정 시간으로 바꾸지 않았다. 점 솔버 호출은 봉인된 per-solve operation_counts로 정확히 재구성했고, unit optimizer2회는 기존 PASS/source에서 도출한 값으로 실제 저장 counter와 구분했다. 시간 비교는 별도 완결된600경로 측정만 사용한다.', '',
        '[REVIEW_CHECKS.json](REVIEW_CHECKS.json)은 독립 public arithmetic 검산, [REVIEW_MANIFEST.json](REVIEW_MANIFEST.json)은 게시 파일 SHA/bytes, [BUILD_LEDGER.json](BUILD_LEDGER.json)은 실제 실행량과 중단·복구를 기록한다. public verifier는 Torch/OpenCV/모델/PnP/ray를 실행하지 않고 원행·배치·cohort·관측 decode·최종 자세/H projection·통계·bootstrap·실행량·시간 원행을 다시 계산한다. 이 검산이 비공개 가중치의 실제 GPU 실행 자체나 물리 GT를 독립 인증하는 것은 아니다.', '',
        '원본 source의 local main HEAD·status·사용자 수정6파일·diff와 기존331개 게시 파일을 SHA로 보존했다. 기존 마감 실험·잘못된 결과·원본 영상/가중치는 삭제하거나 바꾸지 않았다. 새 실사 촬영/수동 annotation/논문/PPT/LaTeX/PDF 수정은0이다. 성능이 나빠서 설정·seed·학습량을 바꾸거나 재학습하지 않았다.', '',
        '게시 대상은 `research/observation-refiner-robust-pnp-20261009`다. main push/자동 merge/force push는 하지 않는다. [PUBLICATION.json](PUBLICATION.json)에 payload commit과 보호된 원격 main 관찰 SHA가 있고, 마지막 게시 commit과 원격 research SHA 일치는 push 후 별도 확인한다. 원격 main은 다른 작업자가 바꿀 수 있으므로 저장소 전체에서 main이 절대 바뀌지 않았다고 주장하지 않는다.', '',
        'public 검산 명령과 비공개 데이터/가중치 의존성의 범위는 [REPRODUCE.md](REPRODUCE.md)에 있다. 결과는 source 학습 회복과 실제 pose 성공을 구분하며, 다음 학습 또는 설정 반복을 실행하지 않은 상태에서 이 고정 실험을 종료한다.', '']
    for figure in ('01_pose_overview.png','02_paired_deltas.png','03_correspondence_mask.png','04_learning_transfer.png','05_visible_hidden_damage.png','06_real_cases.png','07_historical_mask_stress.png','08_runtime.png','09_real_mask_stress.png'):
        assert (DOC/'figures'/figure).is_file(), figure
    mask_lines=['다음은 IMAGE_ROLE과 고정 N3→cornerSubPix의 마스크 교차표다. 사람 정보가 있는 코너에서 초기 예측 H가 일치하는지 판정하며, 정보가 없는 코너까지 완벽한 마스크라고 인증하지 않는다. 개선/악화/혼합은 **새 자세를 산출한 프레임만** 센다. 기본 반환은 별도로 남겼다.', '']
    rows=[]
    for key in ('wrong_mask','matching_known_mask'):
        item=mask_n3[key]; q=item['quality_new_only']
        rows.append([key,item['frames'],item['new_pose'],item['fallback'],item['no_pose'],
                     q.get('both_improved',0),q.get('both_worsened',0),q.get('mixed_or_equal',0)])
    mask_lines += table(['마스크','전체','새 자세','기본 반환','실패','위치·회전 모두 개선','둘 다 악화','혼합/동률'],rows)
    mask_lines += ['', '마스크가 틀린 60장에서도 새 자세 54장을 구했고 그중 10장은 위치·회전 모두 좋아졌다. 알려진 마스크가 맞는 185장에서는 새 자세 166장을 구했지만 그중 69장은 두 오차가 모두 커졌다. 가림 분류의 맞고 틀림으로 자세 성공을 대체할 수 없다.', '']
    lines[lines.index('## 직접 가시점 손상과 숨은 점 오차'):lines.index('## 직접 가시점 손상과 숨은 점 오차')] = mask_lines
    stress_lines=['아래 28조건은 변형 가능한 프레임만의 결과와 분모를 표시한다. 변형 불가는 원 anchor 산출을 전체 운용 집계에 남겼다. 올바른/틀린 inlier 평균은 새 자세가 없을 때의 빈 inlier 집합도 포함한 조건 분모로 계산했다. 자세 오차는 조건의 전체 운용 출력이다. 올바른 입력 개수별·2D/3D rank별 분포와 쉬움/중간의 평균·분산·SD·중앙값·P90은 원행과 통계 JSON에 있다.', '']
    rows=[]
    for coordinate in ('BASE','N3_SUBPIX'):
        for family in stress_statistics['families']:
            for condition in stress_statistics['conditions']:
                g=stress_statistics['strata']['combined'][f'{coordinate}::{family}::{condition}']
                item=g['transformable_only']; pose=item['pose']; value=pose['metrics']['operational']
                rows.append([coordinate,family,condition,g['transformation_possible'],g['nontransformable'],
                    pose['new_pose_estimated'],pose['fallback_used'],pose['no_pose'],
                    number(item['mean_correct_pool']),number(item['mean_correct_final_inliers']),number(item['mean_wrong_final_inliers']),
                    number(value['translation_cm']['mean']),number(value['rotation_deg']['mean']),number(value['ADDsym_cm']['mean'])])
    stress_lines += table(['좌표','oracle family','조건','변형 가능','변형 불가','새 자세','반환','실패','올바른 입력 평균','올바른 inlier 평균','틀린 inlier 평균','위치(cm)','회전(°)','ADDsym(cm)'],rows) + ['']
    lines[lines.index('## 수정 감독과 실제 학습'):lines.index('## 수정 감독과 실제 학습')] = stress_lines
    runtime_note='타이밍 경로 detector는 600회이고 로드 단계 내부 초기화 1회는 구간 밖이다. 이 benchmark process의 실제 detector forward 총수는 601회다. 고정 N3는 300회, 새 head는 150회이며 초기 자세 600회와 feature 역할 자세 150회는 타이밍 경로에 포함했다.'
    lines[lines.index('## 실제 이미지와 수치 그림'):lines.index('## 실제 이미지와 수치 그림')] = [runtime_note,'']
    (DOC/'RESULT_KO.md').write_text('\n'.join(lines), encoding='utf-8')
    print('DETAILED_REPORT_FROM_ACTUAL_COMPLETED_ROWS',len(lines),'lines')


if __name__ == '__main__':
    main()
