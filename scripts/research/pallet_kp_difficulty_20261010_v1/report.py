"""Reviewable Korean report and exact CSV from the completed sealed follow-up."""
import csv
from collections import Counter
from pathlib import Path
from ..pallet_observation_refiner_20261009_v1 import common as C
from .visual_review import LABELS

DOC=C.WORKTREE/'_docs/experiments/pallet_kp_difficulty_20261010_v1'

def main():
    assert not (DOC/'RESULT_KO.md').exists()
    metrics=C.read(DOC/'METRICS.json');runtime=C.read(DOC/'RUNTIME.json');assert runtime['complete']
    rows=list(C.iter_rows(DOC/'PREDICTIONS.jsonl.gz'))
    arms=metrics['fixed_comparators']+metrics['new_methods']
    labels=dict(LABELS,GEOMETRY_ONLY='기존 Geometry head',IMAGE_NO_ROLE='기존 Image/no-role head',IMAGE_ROLE='기존 Image/role head')
    with (DOC/'METRICS.csv').open('w',encoding='utf-8',newline='') as f:
        fields=['method','scope','metric','n','mean','sample_variance','sample_std','median','P90','max','unit','ddof']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for a in arms:
            for scope,group in metrics['methods'][a]['metrics'].items():
                for key,stat in group.items():writer.writerow(dict(method=a,scope=scope,metric=key,**stat))
    text=['“가림 판단이 일부 틀려도 다른 유효 대응점으로 자세를 구하고 자기 가림 코너를 재투영했을 때, 기존 단순 대조보다 실제 위치·회전이 좋아졌는가?”',
          '', '**아니요. 전체319장에서는 위치·회전을 함께 개선하지 못했다.** 새 자세가 나온 프레임에서는 예측 자기 가림 코너를 최종 자세의 재투영으로 대체했고, 그 초기 좌표나 재투영점을 최종 fit 관측으로 다시 넣지 않았다. 대응 존재 확률을 합산하면 새 자세가 더 나오지만 오차는 증가했다. 초기 치수 분기를 유지하는 절제는 일부 전역 분기 오류를 줄였으나 기존 N3→SubPix보다 나빴다.',
          '', '이는 [원래 실험](../pallet_observation_refiner_20261009_v1/RESULT_KO.md)에 대한 후속 어려움 진단이다. 원래 게시 commit `6e3e96bd22396aca47b09ab1216a4a18fb2aa5b2`의 코드·원행·문서·그림111개는 해시로 보존했다. 새8조건과 성공 기준을 [PROTOCOL.json](PROTOCOL.json)에 고정한 뒤 실행했다. 가설을 도출할 때 기존 DEV319 결과를 보았으므로 이번319장은 독립 holdout 검증이 아니다. 실사 위치·회전·ADDsym reference는 기존 기하 재구성값이며 독립 물리 측정 GT가 아니다.',
          '', '## 전체 결과와 분모', '', '수치는 원행의 운용 출력이며 fallback도 포함한다. 17조건 각각319장이 산출됐고 완전 운용 실패는0장이다. 기존 고정 대조는 새 자세0/기본 반환0으로 표시하며 과거319출력을 뜻한다. ADDsym 원행은 m이고 표·CSV는 cm다. 극단 오차를 삭제하거나 cap하지 않았다.', '',
          '| 조건 | 새 자세 | 기본 반환 | 위치 평균 cm | 회전 평균 ° | ADDsym 평균 cm |',
          '|---|---:|---:|---:|---:|---:|']
    for a in arms:
        r=metrics['methods'][a];s=r['metrics']['operational']
        text.append(f'| {labels[a]} | {r["new_pose_estimated"]} | {r["fallback_used"]} | {s["translation_cm"]["mean"]:.4f} | {s["rotation_deg"]["mean"]:.4f} | {s["ADDsym_cm"]["mean"]:.4f} |')
    text+=['','![전체 오차와 산출률](figures/01_pose_and_coverage.png)',
           '', '점은319개 모든 오차, 원은 중앙값, 마름모는 평균, 선은 P10–P90이다. 큰 오차 때문에 평균과 중앙값이 크게 다르다. 새 자세와 기본 반환은 오른쪽 막대에서 분리했다. [PREDICTIONS.jsonl.gz](PREDICTIONS.jsonl.gz)의 key는 `(method,id)`다.',
           '', '## 평균·분산·표준편차·중앙값·P90', '', '아래는 각319 운용 출력의 재집계다. 표본분산·SD는 ddof=1, 분위수는 선형 보간이다. 표본분산 단위는 cm²/°², SD는 cm/°다. 모든17조건의 운용/새 자세 조건부 통계102행은 [METRICS.csv](METRICS.csv), 원행 ID·비교 분모·CI는 [METRICS.json](METRICS.json)에 있다. 새 자세0이면 조건부 평균은 NA이며0으로 채우지 않았다.']
    focus=['BASE','N3_SUBPIX']+metrics['new_methods']
    for key,unit in [('translation_cm','위치 cm'),('rotation_deg','회전 °'),('ADDsym_cm','ADDsym cm')]:
        text+=['',f'### {unit} — 운용319장','', '| 조건 | 평균 | 표본분산 | SD | 중앙값 | P90 |', '|---|---:|---:|---:|---:|---:|']
        for a in focus:
            s=metrics['methods'][a]['metrics']['operational'][key]
            text.append('| '+labels[a]+' | '+' | '.join(f'{s[k]:.4f}' for k in ('mean','sample_variance','sample_std','median','P90'))+' |')
    text+=['', '## 같은 영상의 paired 비교', '', 'new−comparator가 음수일 때 개선이다. 기존과 동일한13세션·10,000개 세션 multiplicity 행렬을 재사용했고, [공개 행렬](BOOTSTRAP_SESSION_DRAWS.json.gz)에서95% CI를 독립 재계산할 수 있다. SD와 CI는 다른 값이다. 모든75쌍의 공통 운용/후보 새 자세/양쪽 새 자세 집합과 제외 ID를 공개했다. 과거 고정 대조에는 새 fit 개념이 없으므로 `both_new_pose`는0이며, 후보의 새 자세 집합에서 같은 대조 프레임을 비교하는 `candidate_new_pose`를 따로 쓴다.', '',
        '| 비교 | 집합/장수 | 위치 Δcm [95% CI] | 회전 Δ° [95% CI] |', '|---|---|---:|---:|']
    pairs=[('IMAGE_ROLE_MATCH_MASS','N3_SUBPIX'),('N3_SUBPIX_INITIAL_DIMENSION_PRIOR_GEOM_NOSELF','N3_SUBPIX'),('IMAGE_ROLE_MATCH_MASS','IMAGE_ROLE'),('N3_SUBPIX_INITIAL_DIMENSION_PRIOR_GEOM_NOSELF','N3_SUBPIX_GEOM_NOSELF_ROBUST')]
    for a,b in pairs:
        pair=metrics['contrasts'][a+'_minus_'+b]
        for scope in ['common_operational','candidate_new_pose']:
            r=pair[scope];vals=[]
            for key in ['translation_cm','rotation_deg']:
                s=r['metrics'][key];vals.append(f'{s["mean"]:+.4f} [{s["CI95"][0]:+.4f}, {s["CI95"][1]:+.4f}]' if s['mean'] is not None else 'NA')
            text.append(f'| {labels[a]} − {labels[b]} | {scope}/{r["common_frames"]} | '+' | '.join(vals)+' |')
    text+=['', '사전 지정 primary IMAGE_ROLE_MATCH_MASS와 secondary N3 치수 prior 모두 N3→SubPix보다 두 평균을 낮추지 못했다. primary의8개 새 자세만 보아도 같은8장의 대조보다 위치·회전이 나쁘다. 치수 prior와 이전 unlocked 솔버의 평균 차이는 음수지만 두 CI 모두0을 포함한다. 그 차이를 일반화된 성공으로 주장하지 않는다.',
           '', '## KP 보정에서 무엇이 어려웠는가', '',
           '| 어려움 | 관측 가능한 진단과 실제 증거 | 해석 |', '|---|---|---|',
           '| 정답 경계의 존재 | 실제 USD 메시·closest point·앞 표면 깊이·visible mask를 함께 검사 | 직육면체 모서리·hull 자체를 물리 경계 GT로 쓰면 안 된다. NONE와 IGNORE, 내부선·실제 외곽을 구분해야 한다. |',
           '| 후보 범위와 이동량 | source-test 양성 normal offset 평균1.13px/중앙값0.69/P902.53; 영상 내915코너 초기 오차 평균2.73px | source는 좌표 이동이 비교적 작다. 채택 점의0.2px 개선과 실사 큰 오차 해결을 같은 난도로 해석하면 안 된다. |',
           '| 대응 존재와 위치의 확률 | 경계 양성941개에서 Pmatch 중앙값은 약0.31, 내부 양성은0.82–0.97; 경계의 조건부 정답 두 bin 확률은0.59–0.64 | 위치 분포가 정답 근처에 있어도 존재 확률이 낮으면 관측을 잃는다. 66-way 최댓값과 존재 사건의 확률 합은 다르다. |',
           '| 경계→선→코너 조립 | 이상적인 기존 감독으로도 test128중 영상 안≥4코너12; frozen3모델은 원래 decoder에서 각각0/128 | 일부 점의 정밀도보다 대응 그래프를 닫아 PnP 입력을 만드는 문제가 먼저 막혔다. 합성에서도 발생하므로 실사 전이만의 문제라고 할 수 없다. |',
           '| 수치 경계 민감도 | 원래 라벨10,752개 재현, 정확한 경계 광선 NONE 중1,202개가 고정±0.05px에서 목표 깊이 실제 메시 교차 | 감독의 수치 민감도에 대한 구체적 증거다. 이웃 표면 hit를 자동으로 수정 GT나 RGB 경계 소유권 정답으로 승인하지 않았다. |',
           '| 교차의 조건과 외삽 | 기존 NO_ROLE 교차는 지지선 span의최대186.9배 밖으로 외삽 | 선 두 개가 있다는 것만으로 안정된 코너가 생기지 않는다. 각도·지지 길이·외삽을 연속 값으로 기록했다. |',
           '| 남은 대응의 정확도·배치 | 기존 N3 마스크 후 reference상 정확한 점≥4는192/319; 정확4·rank6에서도92.08cm/88.39° 사례 | 점 수와 국소 Jacobian rank는 정확한 전역 해를 보증하지 않는다. 실제 정확 점/최종inlier/배치를 함께 봐야 한다. |',
           '| 강건 합의의 잘못된 일관성 | 고정 사례6inlier 중 reference상 정확2; 새 자세도176.85cm | 서로 일관된 오답이나 치수·위상 다중해는 잔차 합의만으로 걸러지지 않을 수 있다. |',
           '| 숨은 점 복원과 자세 성능 | 숨은 재투영 오차와 직접 가시 손상을 별도 집계 | 재투영은 최종 R,t에서 파생되므로 자체가 fit 자세를 개선하지 않는다. 숨은 점 오차만으로 성공을 선언할 수 없다. |',
           '', '이 분류의 원행은 [LEARNED_DIFFICULTY_ROWS](LEARNED_DIFFICULTY_ROWS.jsonl.gz), [GEOMETRY_DIFFICULTY_ROWS](GEOMETRY_DIFFICULTY_ROWS.jsonl.gz), [SOURCE_CEILING_ROWS](SOURCE_CEILING_ROWS.jsonl.gz), [POSTHOC_CORRESPONDENCE_ROWS](POSTHOC_CORRESPONDENCE_ROWS.jsonl.gz)에서 확인할 수 있다. reference상 정확함은 고정된8px 판정이며 독립 물리 GT 판정이 아니다.',
           '', '![source와 실사 관측 병목](figures/02_source_and_real_bottlenecks.png)',
           '', '초기 모델·N3를 새로 학습하지 않았다. 동일한 기존3×3,000업데이트 모델의 last 가중치만 사용했다. source-test에서 채택한 동일 양성 집합의 zero-offset 대비 평균 개선은0.16/0.21/0.22px지만 세 모델 모두4코너 조립은0이었다. existence합산은 NONE·IGNORE 채택도 늘린다. 기존 POSITIVE만으로 조립하면 모든 모델·양쪽decoder/cal+test에서4코너는0이다. [SOURCE_LOGIT_DIFFICULTY](SOURCE_LOGIT_DIFFICULTY.json)는65조건부 분포·엔트로피·정답 두 bin 질량을 공개한다.',
           '', '## 실제 메시로 확인한 감독 문제', '',
           '이미 있는 P0 RGB·실제 USD 형상·전달된 visible/amodal mask·기존 feature/target 캐시를 사용했다. 새 RGB·장면·수동 어노테이션은0이다. 고정source-test128장에서31,875개의 실제 광선과21,377개의closest-point 질의를 실행했다. 원래 POSITIVE2,333/NONE4,414/IGNORE4,005가 정확히 재현됐다.',
           '', '실제 메시 위·영상 내·visible-mask 지지 NONE4,130개를 검사하면2,164개는 정확한 광선이 표면을 놓쳐 무한대였고1,966개는 목표보다 앞 표면에 맞았다. 후자는 실제 가림 증거가 있으므로 mask만으로 가시로 바꾸지 않았다. 무한대 중1,202개는 고정±0.05px에서 목표 깊이의 실제 표면 hit를 얻었다(경계 역할1,174). 원 코드의 tiny-inset 주석과 달리 실제 ray에는 이동이 없었다. 이것은 정확한 경계 광선의 민감도가 NONE 감독을 만드는 증거다. 해결되지 않은962개는 미해결로 남겼다.',
           '', '![실제 source 경계 민감도](figures/08_source_ray_sensitivity.png)',
           '', '![앞 표면 가림과 미해결 대조](figures/09_source_ray_controls.png)',
           '', '원래 positive에 위 증거만 추가한 **기계적 진단**은4코너가 raw13→33/128, 영상 내12→28/128이었다. 수정된 감독·배포 출력·학습 성능은 아니다. 이 진단에서도100/128장은 영상 안4코너가 부족하다. 수치 감독만 고치면 전체 아이디어가 성공한다는 근거는 없다. [SOURCE_FINDINGS_KO.md](SOURCE_FINDINGS_KO.md)에는 정확한source/query/primitive/깊이 사례와 한계를 자세히 적었다.',
           '', '## 실제 영상의 실패와 부분적 개선', '', '아래 사례는 성능 집계 후 사전 지정한 기전 사례와 primary가 새 자세를 구한8장 중 BASE 대비 최대 위치 감소/증가 ID를 선택한 설명 그림이다. 전체319 평균은 사례 선택과 독립이다. 노랑은 관측·선 지지, 초록은 최종inlier, 빨강은 예측 자기 가림, 주황은 최종 자세 전체 모델 투영, 분홍 점선은 기하 reference다. 주황 전체 투영은 검토용이며 native 출력에서 모든 점을 재투영으로 바꿨다는 뜻이 아니다.',
           '', '![실제 영상12패널 비교](figures/03_real_difficulty_cases.png)',
           '', '`plastic_night_01:038630`: 기존 N3→SubPix 3.28cm/2.64°→unlocked92.08cm/88.39°→치수 prior13.88cm/1.38°다. pool/inlier는[1,2,4,5]의4개 정확 대응이고 국소rank6이다. prior는 초기 선택한 cf_extents=[1.3,0.11,1.1] 분기를 유지한다. unlocked는[1.1,0.11,1.3]을 골랐다. prior가 큰 분기 오류를 줄였어도 위치는 원래 대조보다 나쁘다.',
           '', '`eval_pallet09:1778653806958839552`:6개 합의 중 reference상 정확2개로, 초기182.01cm/16.47°가176.85cm/16.30°로 조금 좋아져도 정확한 자세라고 할 수 없다. `wood_night_01:030607`은 primary 새 자세8장 중 BASE 대비 위치 감소가 최대인 사례이며6.01→4.87cm지만 회전3.01→5.50°로 악화한다. `wood_night_01:030825`는 같은8장 중 BASE 대비 위치 증가가 최대인 사례이며1.86cm/2.90°→79.82cm/102.02°다. [VISUAL_REVIEW.json](VISUAL_REVIEW.json)에 이미지 원본SHA·crop·ID·방법·rowSHA·selection이 있다.',
           '', '## 가림 오판, 최종 inlier와 실패를 분리', '', '원 실험 N3 마스크가 알려진 사람 상태와 틀린91장 중21장은 위치·회전이 모두 개선됐다. 알려진 상태가 맞는228장 중77장은 둘 다 악화됐다. 새 N3 prior에서도91장 중16개가 둘 다 개선되고,228장 중73개가 둘 다 악화된다. 완벽한 가림 분류를 필요조건으로 쓰지 않았다. 알려지지 않은 사람 상태는 UNKNOWN으로 남긴다.',
           '', 'correct/wrong/unknown pool과 final inlier, 남은 정확 점의 수·3D배치·영상 배치·국소rank, 오제외/오잔류 ID는 후행 참조 annotation으로 기록했다. **GT를 관측 선택이나 최종 fit에 넣지 않았다.** 새코드와 원행에서 no-match 초기 좌표 자동 채움은false, 재투영 관측 재사용은false다. H가 재추정 후 달라져도 프레임을 실패로 바꾸지 않았다.',
           '', '| 새 조건 | fit 실패 이유(그 뒤 기본 반환) | 다중 후보가 기록된 새 자세 |', '|---|---|---:|']
    for a in metrics['new_methods']:
        rs=[r for r in rows if r['method']==a]
        multiple=sum(bool(r['solver'].get('multiple_solutions')) for r in rs if r['new_pose_estimated'])
        text.append(f'| {labels[a]} | {metrics["methods"][a]["failure_reasons"] or "없음"} | {multiple} |')
    text+=['', '`insufficient_observations`는 실제 유효 입력<4, `insufficient_consensus`는 지지 합의 부족이다. 원 솔버의 `degenerate_or_numerical_generation_failure`는 생성 단계의 퇴화/수치 실패 묶음이며 여기서 임의로 하나의 원인으로 바꾸지 않았다. `multiple_solutions`는 기록된 대안 존재이며 자동 실패 또는 전역 유일성 증명이 아니다. 새 fit 실패와 운용 완전 실패, 기본 출력 반환을 구별했다.',
           '', '## 직접 가시점 손상과 숨은 점 오차', '', '| 경로 | 직접 가시1759점 평균px 전→후 / 악화점수 | 사람 자기 가림448점 평균px 전→후 | 실제 재투영 ID의 평균px 전→후 / 점수 |', '|---|---|---|---|']
    for a in ['N3_SUBPIX_INITIAL_DIMENSION_PRIOR_GEOM_NOSELF','IMAGE_NO_ROLE_MATCH_MASS','IMAGE_ROLE_MATCH_MASS']:
        d=metrics['visibility_damage'][a]['operational'];v,h,p=d['DIRECT_VISIBLE'],d['SELF_OCCLUDED'],d['ALGORITHM_REPROJECTED_IDS']
        text.append(f'| {labels[a]} | {v["before"]["mean"]:.4f}→{v["after"]["mean"]:.4f} / {v["worsened"]} | {h["before"]["mean"]:.4f}→{h["after"]["mean"]:.4f} | {p["before"]["mean"]:.4f}→{p["after"]["mean"]:.4f} / {p["corners"]} |')
    text+=['', '사람 자기 가림 집합과 알고리즘 재투영 집합은 다르므로 분모도 다르다. fallback으로 변하지 않은 점은 전/후 같은 집합으로 유지했다. ROLE의 실제8재투영점 평균은5.78→60.83px다. 다른 경로에서 숨은 오차가 내려도 위치·회전 개선을 대신하지 못한다.',
           '', '## 실제 전체 경로 처리시간과 실행량', '', '조용한 창에서26실사×5반복으로130회/경로를 측정하고20warmup/경로를 별도 실행했다. 총600회다. RAM BGR에서 fresh detector(공유neck)→좌표 보정/관측→fresh초기pose→새가설 bank→강건fit→숨은 재투영까지 포함했다. RGB 디코딩·모델 로드·참조검산·journal은 구간 밖이다. 기존 시간 합산이나 정확도 캐시 재생이 아니다.', '', '| 경로 | N | 평균ms | 분산ms² | SDms | 중앙값ms | P90ms |', '|---|---:|---:|---:|---:|---:|---:|']
    for a in runtime['arms']:
        s=runtime['summaries'][a]['full']
        text.append(f'| {labels[a]} | {s["n"]} | '+' | '.join(f'{s[k]:.4f}' for k in ['mean_ms','sample_variance_ms2','sample_std_ms','median_ms','p90_ms'])+' |')
    runtime_rows=list(C.iter_rows(DOC/'RUNTIME_ROWS.jsonl.gz'));role=[r for r in runtime_rows if r['arm']=='IMAGE_ROLE_MATCH_MASS' and r['phase']=='measured']
    text+=['', f'ROLE runtime panel의 운용 상태는 {dict(Counter(r["output_status"] for r in role))}다. 이 선택된26패널의 속도·산출 분포를 전체319 학습 성공률로 일반화하지 않았다. 모든600출력·initial/final pose·hidden replacement가 봉인된 평가 경로와1e-7 tolerance로 일치했다. 자원 상태7스냅샷에서 경쟁·온도 guard를 통과했다. [RUNTIME_ROWS](RUNTIME_ROWS.jsonl.gz), [RUNTIME.json](RUNTIME.json).',
           '', '새 정확도 평가:8×319=2,552경로. 기존 unlocked4×319=1,276경로를 실제 재실행해 좌표·R,t·T/R/ADDsym 최대차0을 확인했다. pilot26×4=104새경로+104기존재생은 GT 채점 전에 수행했다. 정확도 단계는 동일 입력의 초기 자세와 고정logits를 재사용했으며 detector/head forward0이다. 따라서 정확도 솔버 구간26.126초는 배포 latency로 쓰지 않았다.',
           '', 'source 고정head CPU 진단은24+48=72 batch16 forward,1,152 image-head exposures였다. 추가 학습0, 새RGB0, 새실사·수동annotation0이다. 실제 전체 경로에서는 detector600+내부초기화1, 기존 N3 forward300, 고정 ROLE head150, 초기historical pose600+feature-role초기pose150, 강건후단300을 실행했다. 원래9,000학습 update와 원 실험 실행량을 새 실행량으로 세지 않았다. 자세한 가설/LM/실패·검산·시도 기록은 [CAUSAL_POSE_EXECUTION](CAUSAL_POSE_EXECUTION.json), [SOURCE_DIAGNOSTIC_EXECUTION_COUNTS](SOURCE_DIAGNOSTIC_EXECUTION_COUNTS.json), [BUILD_LEDGER](BUILD_LEDGER.json)을 참고한다.',
           '', '## 어떤 요소가 효과를 냈고 무엇은 필요 없었는가', '',
           '| 요소 | 이번 증거로 말할 수 있는 결론 |', '|---|---|',
           '| 완벽한 가림 분류 | 성공 필요조건이 아니다. 오판이 있어도 실제 개선한 프레임이 있다. 반대로 알려진 mask가 맞아도 악화하므로 분류정확도만 성공 지표로 쓰지 않는다. |',
           '| mask없는 강건PnP | 기존 필수 대조에 포함했고 전체 평균에서는 단순 N3보다 나빴다. 강건만으로 충분하다는 근거가 없으며, 이 대조를 생략하고 새 판단기 필요성을 주장하지 않았다. |',
           '| 영상·역할 경계 학습 | 기존 source 양성 일부의 좌표를 조금 개선했으나 source graph와 실제 자세가 막혔다. 현재 role포함이 필요하거나 효과적이라는 후단 개선 근거는 없다. 새role classifier나 대형 backbone 필요성도 입증하지 않았다. |',
           '| 존재 확률 합산 | joint66-way와 존재 사건의 MAP 차이를 제거해 새 자세 산출을 늘렸지만 실제 오차를 악화했다. 후보 채택 증가를 성공으로 볼 수 없다. threshold를 조정해 다시 고르지 않았다. |',
           '| 초기 치수 분기 prior | 일부 큰 전역 branch오류를 줄일 수 있지만 잘못된 초기분기도 보존할 수 있다. 고정N3 단순대조를 이기지 못했다. 원래 prior-free 솔버 계약을 바꾸지 않고 새별도절제로 공개했다. |',
           '| 최종 숨은 재투영 | 요구된 출력처리를 수행했다. fitR,t를 고친 독립 정보가 아니므로 재투영 그 자체를 위치·회전 개선 원인으로 주장하지 않는다. |',
           '', '성공을 가능하게 하려면 먼저 물리적으로 관측 가능한 target과 경계NONE의 수치 판정을 검증하고, 남은 독립 관측이 실제 자세를 구할 만큼 충분한지 확인해야 한다. 이번 데이터에서 단순히 가림 정확도·채택률·픽셀loss를 높이는 것만으로 해결될 조건이 아니었다. 부분 선관측 경로는 원 실험의 동일 IMAGE_ROLE point+line 별도 절제를 그대로 유지했으며, 이번 point 경로에 선·그 선의 교차점을 중복 관측으로 추가하지 않았다.',
           '', '수정된 감독에 의한 새학습, 새로운 성공 모델, 새로운 holdout 확인은 **미실행**이다. 기존 감독이 정확하다고 가장해 새 모델을 학습하지 않았고, 부정적인 성능 때문에 설정·seed를 바꿔 반복하지 않았다. 원 실험의 E6 통제4변형 미실행·과거 FIXED_CONTROLS의 BASE293+SUBPIX293=586행 R/t 미저장도 해결했다고 보고하지 않는다. 이586행은 저장 metric의 재집계는 가능하지만 독립 pose 재계산은 불가능하며, 새2552행의 R/t 누락을 뜻하지 않는다. 진단이 완료됐다는 사실과 성능 목표를 달성했다는 주장은 구별한다.',
           '', '## 다른 사람이 확인하는 방법', '', '표의숫자는 [METRICS.csv](METRICS.csv), 모든2552새행은 [PREDICTIONS.jsonl.gz](PREDICTIONS.jsonl.gz), 해시·직접검산은 [REVIEW_CHECKS.json](REVIEW_CHECKS.json)과 [REVIEW_MANIFEST.json](REVIEW_MANIFEST.json)에서 연결한다. [README.md](README.md)의 명령은 private 영상·가중치·GPU 없이 공개파일만 검산한다. [REPRODUCE.md](REPRODUCE.md)는 검산과 실제재실행에 필요한 원본을 구분하고, 완성된 산출물을 덮어쓰지 않는 방법을 설명한다.',
           '', '전용브랜치 `research/observation-refiner-robust-pnp-20261009`에 정상commit/push하고 원격SHA를 검증한다. main 병합·push와 force push는 하지 않았다. 게시 전 source상태·가중치·원래111파일 보존 및 디스크공간 때문에 사용한 독립 tmpfs게시clone은 [PUBLICATION_PRECHECK.json](PUBLICATION_PRECHECK.json)에 기록한다. 게시SHA·원격일치 영수증은 [PUBLICATION.json](PUBLICATION.json)을 참고한다.']
    (DOC/'RESULT_KO.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    (DOC/'README.md').write_text('''# KP 보정의 어려움과 고정 인과 비교 — 검토 자료

전체319장의 새8조건에서 기존 N3→SubPix보다 위치·회전을 함께 개선하지 못했다. 성공한 척하지 않고 source 감독·대응 존재·코너 조립·전역 다중해의 병목을 실제로 검사했다.

먼저 [그림5개·실사12패널 포함 상세 보고서](RESULT_KO.md), [source 감독 세부 증거](SOURCE_FINDINGS_KO.md)를 읽는다. 기존 보고서와 원행111개는 변경하지 않았다.

공개 Git만 있으면 Python3.9+ 표준라이브러리로 다음 명령을 실행할 수 있다. 추론·학습·원본RGB·가중치·GPU·private환경변수는 필요하지 않다.

```bash
python3 scripts/research/pallet_kp_difficulty_20261010_v1/review_verify.py --require-manifest
```

`--root`는 repo 또는 이 문서 디렉터리, `--output`은 별도 검산 JSON이다. 오류는 nonzero exit로 표시한다. 출력 `REVIEW_CHECKS.json`은 재생성 가능한 검산 영수증이며 원행을 덮어쓰지 않는다.

| 파일 | 검토할 내용 |
|---|---|
| [PROTOCOL.json](PROTOCOL.json) | 사전고정8조건·성공기준·별도prior/decoder절제·DEV319한계 |
| [PREDICTIONS.jsonl.gz](PREDICTIONS.jsonl.gz), [CAUSAL_GEOMETRY_SEALED.jsonl.gz](CAUSAL_GEOMETRY_SEALED.jsonl.gz) | 채점 전 봉인과 채점 후2552행·상태·R,t·native출력 |
| [METRICS.csv](METRICS.csv), [METRICS.json](METRICS.json) | 5통계·산출률·75paired비교·집합ID·숨은/가시손상 |
| [BOOTSTRAP_SESSION_DRAWS.json.gz](BOOTSTRAP_SESSION_DRAWS.json.gz) | 기존과 같은13세션10000행을 직접CI재계산 |
| [POSTHOC_CORRESPONDENCE_ROWS.jsonl.gz](POSTHOC_CORRESPONDENCE_ROWS.jsonl.gz) | 참조정확pool·배치·finalinlier·오판과자세결과 분리 |
| [MATCH_MASS_OBSERVATIONS.jsonl.gz](MATCH_MASS_OBSERVATIONS.jsonl.gz) | 원래957logits에서별도고정decoder·없음자동채움false |
| [SOURCE_CEILING_ROWS.jsonl.gz](SOURCE_CEILING_ROWS.jsonl.gz), [SOURCE_LOGIT_DIFFICULTY.json](SOURCE_LOGIT_DIFFICULTY.json) | 실제기존target1024와존재/위치확률분리 |
| [SOURCE_RAY_VALIDATION_ROWS.jsonl.gz](SOURCE_RAY_VALIDATION_ROWS.jsonl.gz), [SOURCE_RAY_PROTOCOL.json](SOURCE_RAY_PROTOCOL.json) | 실제메시고정128광선검사·NONE수치민감도·앞표면대조 |
| [VISUAL_REVIEW.json](VISUAL_REVIEW.json), [SOURCE_RAY_VISUAL_CASES.json](SOURCE_RAY_VISUAL_CASES.json) | 실제이미지hash·ID·query·crop·사후선택근거 |
| [RUNTIME_ROWS.jsonl.gz](RUNTIME_ROWS.jsonl.gz), [RUNTIME.json](RUNTIME.json) | fresh전체경로600회·일치·경쟁guard·5통계 |
| [BUILD_LEDGER.json](BUILD_LEDGER.json), [REVIEW_MANIFEST.json](REVIEW_MANIFEST.json) | 실제실행량·보존·공개전체binding |
| [PUBLICATION_PRECHECK.json](PUBLICATION_PRECHECK.json), [PUBLICATION.json](PUBLICATION.json) | 보호상태·게시commit·원격SHA영수증 |

자세 원행 key는 `(method,id)`이며 gzip JSONL을 읽어 한 프레임을 추적할 수 있다. source ceiling/ray는 `id`, source logits는 `(arm,id)`, runtime은 `(arm,id,phase,repeat 또는 warmup_index)`이며 각 schema를 참고한다. `solver.used`는 scoring pool, `fit_input_ids`는 실제fit, `inliers`는 최종합의다. 서로 같다고 가정하지 않는다. 새R,t가 있으면 초기hidden을재투영교체하며 재투영점을 다시fit에 넣지 않는다. fallback은전체 초기출력이다. `GEOMETRIC_PROXY`는독립물리GT가아니다.

원본RGB와featurecache·체크포인트를 공개하지 않았다. 공개검산은저장결과의일관성·연산을 확인하며 실제촬영GT나원본학습실행을 독립인증하는것은아니다. 실제재실행요건은 [REPRODUCE.md](REPRODUCE.md)를 참고한다.
''',encoding='utf-8')
    print('REPORT_COMPLETE',len(text))

if __name__=='__main__':main()
