## 연구자가 검토할 수 있는 실행 방법과 그림

아래 그림은 저장된 원행을 읽어 만든 검토 자료다. 결과를 본 뒤 threshold·mask·checkpoint·pose를 바꾸지 않았다. 그래프의 모든 프레임 분모와 자세 상태는 본문의 표와 원행에 유지된다. 그림의 frame/source 선택·원본 binding·표시 좌표·파생 PNG 해시는 [VISUAL_REVIEW_CASES.json](VISUAL_REVIEW_CASES.json)에 있다. 원본 RGB 파일을 Git에 추가하지 않고 기존 영상의 주석 montage만 게시했다.

실제 후단은 `고정 RGB 키포인트 → 관측 후보 선택/경계 교점 → 예측 자기 가림 제외 → 유한 4점 부분집합 PnP → 최종 H 재투영`이다. masked/unmasked와 일반/강건 대조는 같은 native 좌표·K·치수 후보를 사용한다. 첫 robust 실행에서 생성한 가설은 마스크 사이에 재사용하지만, 각 arm은 허용 generator와 같은 자기 U에서 scoring한다. 최종 R,t가 없는 경우에는 고정 초기 자세 fallback을 별도 상태로 반환한다.

자세 산출률·평가 분모·axis·실제 source target·5,890 parameter head·28개 channel·loss·AdamW·동일 초기값/배치·원행 key와 frame drilldown은 [REVIEW_GUIDE_KO.md](REVIEW_GUIDE_KO.md)에 자세히 설명했다. 공개 원행만 검산하려면 저장소 루트에서 아래 명령을 실행한다. Python 3.9 이상 표준 라이브러리만 필요하고 모델 추론·private source·GPU가 필요하지 않다.

```bash
python3 scripts/research/pallet_observation_refiner_20261009_v1/review_verify.py
python3 scripts/research/pallet_observation_refiner_20261009_v1/review_verify.py \
  --frame-id eval_noapril:1775201415399297536
```

`--frame-id`는 설명을 추가하면서 전체319 검사도 유지한다. `--root`는 repo 또는 결과 디렉터리, `--output`은 별도 검산 JSON, `--require-manifest`는 게시 manifest 필수 검사 옵션이다. 공개 검산은 저장된 metric·투영·관측 decode·time의 일관성을 검사한다. 원본 RGB에서 detector/N3/head를 다시 실행하거나 물리 정답 자체를 인증하는 전체 재현과는 범위가 다르다. 전체 원본 의존 재현은 [REPRODUCE.md](REPRODUCE.md)에 있다.

### 그림 1 — 전체 자세 결과와 실제 새 산출

![전체319장의 자세 결과와 새 자세/fallback 구분](figures/01_pose_overview.png)

출처: [METRICS.json](METRICS.json), [METRICS.csv](METRICS.csv)의 전체 운용 통계와 `new_pose_estimated/fallback_used/no_pose`. T/ADDsym은 cm, R은 degree다. 큰 오차도 전체 원행에 남아 있다. 학습 모델0/3/0 새 산출과319/316/319 fallback을 구별해야 한다. IMAGE_ROLE의 BASE와 같은 평균은 새 자세 개선의 증거가 아니다. 원프레임 SD는 CI가 아니다.

### 그림 2 — 같은 프레임의 짝차이

![고정 대조에 대한 위치·회전 짝차이와 구간](figures/02_paired_deltas.png)

출처: [PAIRED_COMPARISONS.json](PAIRED_COMPARISONS.json). 부호는 새 방법−대조이며 음수가 오차 감소다. 본문/그림에 표시된 비교 scope와 공통 분모를 함께 읽는다. CI는13세션의 기존10,000 bootstrap draw이고 SD나 추가 seed 변동이 아니다. 단일 seed·반복 개발 DEV319·다중비교 보정 없음의 한계가 있다. 양쪽 방법의 새 산출 공통집합이0인 비교에서 신규 개선량을 주장하지 않았다.

### 그림 3 — 가림 마스크 오판과 기하 실패

![정답 오제외와 오답 잔류 스트레스의 실제 기록](figures/03_mask_stress.png)

출처: [GEOMETRY_STRESS_SUMMARY.json](GEOMETRY_STRESS_SUMMARY.json), [GEOMETRY_STRESS.jsonl.gz](GEOMETRY_STRESS.jsonl.gz), [REAL_MASK_STRESS_SUMMARY.json](REAL_MASK_STRESS_SUMMARY.json). analytic128×11×2 경로는 렌더0인 수학적 진단이다. 실사 사람 마스크의 한 점 오제외/오잔류638행과 구별한다. 정확한 점 수·배치·오답의 일관성이 합의의 한계를 만든다. 공선에 가까운 false consensus의 큰 T/R도 삭제하지 않았다. 수치 pose 반환은 정확한 pose 성공과 다르다.

### 그림 4 — source에서의 대응 점수와 real 관측 부족

![세 동일 모델의 source 성능과 실사 대응 충분성](figures/04_learning_observation_gap.png)

출처: [TRAIN_LOGS.jsonl](TRAIN_LOGS.jsonl)의 고정 source-test probe, [CORRESPONDENCE_COUNTS.json](CORRESPONDENCE_COUNTS.json), [METRICS.json](METRICS.json). source에서 채택된 양성 후보의 오차는 조건부 오차다. source 양성 채택률과 실사4개 이상 교점 수를 같은 성공률이라고 부르지 않는다. 실사 raw4교점 이상은0/6/0, 최종 새 point 자세는0/3/0이었다. 가림 분류 점수로 후단 평가를 생략하지 않았고 no-match를 초기 좌표로 채우지 않았다.

### 그림 5 — 실제 전체 경로 시간

![원영상부터 후단 자세와 재투영까지 실제 처리시간](figures/05_runtime.png)

출처: [RUNTIME_ROWS.jsonl.gz](RUNTIME_ROWS.jsonl.gz), [RUNTIME.json](RUNTIME.json). 각4경로의 warmup20회 및 본 측정130회가 실제 실행되어 총600행이다. 본 측정에는 고정 detector, 초기 자세, 정제/관측, 실제 새 PnP 시도와 hidden 재투영 경로를 포함했다. 캐시 재생 시간이나 과거 평균의 합산이 아니다. 모델 로딩·영상 decode·GT 채점·parity 검사는 측정 밖이다. ROLE17.61ms는 측정 패널에서도 point 자세 fallback인 경로의 시간이며, 빠른 새 기하 복원을 달성한 결과가 아니다.

### 그림 6 — 실사에서 한 행씩 확인하는 사례

![저장된 실사 입력·관측·inlier·숨은 투영의 검토 사례](figures/06_real_cases.png)

출처: 기존 실사 RGB와 봉인된 [PREDICTIONS.jsonl.gz](PREDICTIONS.jsonl.gz)/[LEARNED_PREDICTIONS.jsonl.gz](LEARNED_PREDICTIONS.jsonl.gz), 사후 [REAL_CORRESPONDENCE_ROWS.jsonl.gz](REAL_CORRESPONDENCE_ROWS.jsonl.gz). panel ID·method·선택 기준과 legend는 [VISUAL_REVIEW_CASES.json](VISUAL_REVIEW_CASES.json)에 기록한다. 저장된 초기점·최종 input/inlier·excluded hidden과 최종 투영을 구분한다. reference overlay는 GT 봉인 뒤 그린 평가용 표시이고 solver의 관측이었다는 뜻이 아니다.

그림의 여섯 사례는 유형별 위치오차/위치차이 극값과 frame ID tie-break로 정한 사후 설명용 선택이다. 고정 detector ROI를 crop하고 ROI 밖 투영은 clip/marker로 표시한다. 아래 본문의 ID-첫행 예시와 그림의 극값 사례 선택은 별개이며 대표성이나 평균 개선을 보장하지 않는다.

주황8코너 wireframe은 최종 자세 투영이며 출력 `native_points` 전체가 아니다. H의 초기 좌표만 투영으로 교체하고 nonhidden 관측을 투영으로 덮어쓰지 않았다. cyan 입력은 Base 또는 N3→SubPix, yellow는 관측/line support, green은 point consensus, magenta 점선은 review-only 재구성 reference다. 지역 point+line에는 point consensus를 주장하지 않았다.

| panel·유형 | method | 사후 선택 기준 |
|---|---|---|
| 1, wrong mask·T/R 개선 | `N3_SUBPIX_GEOM_NOSELF_ROBUST` | T 감소 최대; ID tie-break |
| 2, correct mask·T/R 악화 | `N3_SUBPIX_GEOM_NOSELF_ROBUST` | T 증가 최대; ID tie-break |
| 3, 저정확 pool·큰 오차 | `N3_SUBPIX_GEOM_NOSELF_ROBUST` | accurate pool≤2, wrong inlier≥2, T≥100cm 중 T 최대 |
| 4, ROLE fallback | `IMAGE_ROLE` | 선이 있으나 코너0인 fallback 중 ID 첫 행 |
| 5, ROLE point+line 악화 | `IMAGE_ROLE_POINT_LINE` | T/R 모두 악화한 새 지역 정제 중 T 증가 최대 |
| 6, 가시 oracle 개선 | `N3_SUBPIX_ORACLE_VISIBLE_ROBUST` | T/R 모두 개선 중 T 감소 최대 |

| panel | ID | pool 총/참조8px 정확 | 합의 총/참조8px 정확 | T cm 전→후 | R deg 전→후 |
|---|---|---:|---:|---:|---:|
| 1, wrong mask·개선 | `eval_pallet09:1778653832794714368` | 7/2 | 5/2 | 54.41→7.57 | 88.08→10.06 |
| 2, correct mask·악화 | `plastic_night_01:038630` | 4/4 | 4/4 | 3.28→92.08 | 2.64→88.39 |
| 3, 저정확 pool·큰 오차 | `eval_pallet09:1778653806958839552` | 6/2 | 6/2 | 182.01→176.85 | 16.47→16.30 |
| 4, ROLE fallback | `eval_cad:1778653033056483584` | 0/0 | 0/0 | 3.38→3.38 | 1.72→1.72 |
| 5, ROLE point+line 악화 | `eval_night09:1779449604823769344` | 0/0 | 해당 없음 | 1056.87→1486.51 | 64.96→84.08 |
| 6, 가시 oracle 개선 | `wood_night_01:031671` | 6/6 | 6/6 | 167.93→14.93 | 88.13→4.58 |

panel1–3은 N3_SUBPIX_GEOM_NOSELF_ROBUST, panel6은 N3_SUBPIX_ORACLE_VISIBLE_ROBUST다. panel2처럼 참조상 정확한4점과 맞는 mask여도 나쁜 자세가 나올 수 있다. panel4는 query14개/선2개가 있지만 코너0이고, panel5는 line rank6의 지역 정제를 point 합의로 표시하지 않았다.

wrong mask이지만 T/R 모두 개선된 `eval_noapril:1775201415399297536`과 알려진 mask는 맞지만 T/R 모두 악화된 `eval_noapril:1775201443822140928`의 원행도 직접 조회할 수 있다. 이 두 예는 정확한 pool4개 이상인 해당 유형에서 ID 순서 첫 행이라는 사후 설명용 선택이다. 사례를 잘 보여주는 그림은 전체319 성능의 증거가 아니며, 그림의 정확한 대응 수는 기존 참조8px 정의이지 독립 물리 측정이 아니다.

### 그림 7 — source의 실제 마스크와 감독 의미

![실제 source RGB·팔레트 마스크와 positive/no-match/ignore 감독 사례](figures/07_supervision_cases.png)

출처: 이미 존재하던 P0 RGB·delivered visible/amodal mask, 실제 scene.usd 삼각형과 공개 source audit/학습 규약. source ID·split·mask binding·target 의미를 manifest와 [REVIEW_GUIDE_KO.md](REVIEW_GUIDE_KO.md)에서 확인한다. 팔레트의 실제 형상 마스크와 projected virtual cuboid edge를 구별한다. 물리적으로 없는 수직 bounding edge는 ignore이며 box/hull을 실제 마스크로 사용하지 않았다. source의 형상·depth·마스크 감독을 설명하는 그림이 실사에서 같은 edge를 유효하게 관측했다는 증거는 아니다.

source-test P0 세 원본의 retained target bin을 사용하며, 표시 수리 전후3개 distinct source 프레임에서 수행한 auxiliary CPU ray pass6회는 검토 그림 작성량으로 따로 기록한다. 원 실험의 학습·감독 준비를 다시 실행한 것이 아니고 새 RGB도 생성하지 않았다. 60,000개 inventory 확인과 P0 128+G38 32/TEX32 정밀 패널 감사의 범위를 구별한다. 실제 edge 순서의 물리 ID는0/2/4/6/8/9/10/11, unsupported vertical ID는1/3/5/7이다.

| panel | source-test ID·선택 기준 | positive / no-match / ignore | 실제 메쉬/기존 amodal IoU |
|---|---|---:|---:|
| 1 | `P0__shard_00_f0136`, ID 순서 첫 eligible | 7 / 45 / 32 | 0.999927 |
| 2 | `P0__shard_04_f0743`, positive query 최다 | 36 / 20 / 28 | 1.000000 |
| 3 | `P0__shard_06_f0072`, no-match query 최다 | 1 / 55 / 28 | 0.999981 |

stripe patch는 기존 RGB를 표시용 sampling한 것이고 `lo/hi/weight/valid` target은 retained source cache에서 가져왔다. source-test 사례 선택으로 학습 모델을 재선택하지 않았다.

### 검토 범위와 남은 공백

추가 [REVIEW_LOSS_MASK_CHECK.json](REVIEW_LOSS_MASK_CHECK.json)은 실제 첫 정식 배치16장의 positive249/no-match635/ignore460 target에 대해 per-logit ignore gradient0을 확인한 CPU loss-only 검산이다. 원 학습의 all-invalid mask 검사보다 실제 ignore 분리를 직접 확인한다. 추가 detector/head forward나 update는0이며 원9000 업데이트와 구별한다. 실행 명령과 세부 loss 정의는 [REVIEW_GUIDE_KO.md](REVIEW_GUIDE_KO.md)에 있다.

고정 대조1,276행 중586행에는 과거 actual_pose R/t가 저장되지 않았다. 원래 metric과 좌표는 보존했고 저장 metric 재집계는 가능하지만, 누락한 과거 R/t를 새로 복원해 독립 검산한 것처럼 표시하지 않았다. source 특징 준비 내부 OpenCV primitive 호출 수와 point+line finite-difference까지 포함한 전체 residual 호출 수는 NA다. 네 통제 변형은 유효 기존 source 재사용·새 RGB0 우선에 따라 미실행 E6로 남았다. [CHECKS.json](CHECKS.json)의 item12는 기존 family 분리 PASS와 네 변형 NOT_EXECUTED를 별도 표시한다.

[PUBLICATION_PRECHECK.json](PUBLICATION_PRECHECK.json)/[PUBLICATION.json](PUBLICATION.json)은 원 실험의 이전 게시 스냅샷이다. 이번 문서·그림·공개 검산 확장은 [REVIEW_MANIFEST.json](REVIEW_MANIFEST.json)의 현 bundle binding으로 검토한다. 수치 원행은 변경하지 않았고 이번 추가 작업은 설명·이미지·검토 도구 작성이다.

공개 파일만 복사한 별도 폴더에서 원본 영상·가중치·private cache·GPU·PALLET 환경변수 없이 검사를 실제 실행했다. 정상 파일은 통과했고 평균값 변경, 예측 행 누락, 숨은 재투영 좌표 변경, 그림 파일 변경은 해당 검사에서 실패로 검출됐다. 명령의 expected/actual exit와 실패 key는 [REVIEW_VALIDATION_TESTS.json](REVIEW_VALIDATION_TESTS.json)에 있다. 원 실험86파일의 해시 보존과 이번 추가 실행량(loss-only CPU2회·기존3장면 mesh mask ray6회·detector/head/PnP/update0)은 [REVIEW_BUILD.json](REVIEW_BUILD.json)에 별도로 기록했다.
