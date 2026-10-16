# A 결과의 새 스키마와 분모

[확인] `newly_defined_handoff_A_*`는 이번 실행에서 새로 정의한 스키마야. 기존 결과 파일에 아래 열이 있다고 가정하지 않아. 코드의 읽기 지점은 기존 `TARGETS.json`, 원시 특징 cache, 실제 N3/PoseFix 예측 형식을 각각 검증하고 새 결과 행으로 변환해.

| 결과 위치/열 | 의미와 단위 |
|---|---|
| `A_protocol.json`, `results/A_ID_MANIFEST.json` | 첫 새 metric 이전에 고정한 seed·예산·후보·지원·학습/진입판단/개발진단 ID와 역할. 과거 `SYNTH_HELDOUT` 이름은 독립 확인을 뜻하지 않아. |
| `A_*_BASELINES.json`, `A_*_FROZEN_*_seed*.json`, `A_*_FIT_*_seed*.json`의 `rows` | 동일한 전체 frame ID 순서의 방법별 결과. `id`, `session`, `method`를 연결 키로 사용해. |
| `corner.evaluable` | 참조의 유효한 0–7 코너가 있는가. 추론 지원 마스크와 별개인 평가용 참조 마스크야. |
| `corner.detected`, `corner.matched` | 검출 존재와 기존 참조 instance 매칭을 분리해. 실제 DEV는 검출 319/319, 매칭 311/319이며 미매칭 8개를 검출 없음으로 쓰지 않아. |
| `corner.E_sym`, `corner.frame_mean_px` | 승인된 물체 전체 대칭 tuple 중 평가 branch의 평균 코너 거리/원본 영상 대각선, 원본 pixel 평균 거리. 누락·미매칭은 기존 평가 구현의 영상 대각선 penalty로 남아. |
| `corner.errors`, `corner.observed_errors` | 전자는 유효 참조 코너의 전체 penalty 포함 거리, 후자는 매칭되고 예측이 존재하는 코너만의 거리야. 둘의 분모를 혼합하지 않아. |
| `summary.corner.matched_pooled_corner8_*` | `observed_errors`를 모은 조건부 중앙값/90백분위. seed마다 계산하고 그 세 scalar를 평균할 때 seed 평균이라고 표시해. |
| `summary.corner.full_penalty_*`, `PCK`, `gross20` | 전체 평가 참조 코너를 분모로 하는 거리 중앙값/90백분위, 5/10/20px 이하 비율, 20px 초과 비율. PCK는 Percentage of Correct Keypoints(허용 거리 안의 코너 비율)이야. |
| `pose.available`, `translation_cm`, `rotation_deg`, `ADDsym_m` | 실제 기존 전체 W/D+PnP 함수 F(q)의 산출 여부와 참조 대비 오차. 이동 cm, 회전 degree, 승인된 proper rotation group에 따른 정준 8코너 평균 3차원 거리 m. ADD-S나 표면 mesh 점 지표와 동일하다고 쓰지 않아. |
| `selected_index` | J는 하나의 정수 action, I는 코너별 8개 정수 action. NoOp 통계는 J의 0 또는 I의 8개 index가 모두 0인 행을 세. 기존 RAW/N3/PoseFix의 `null`은 새 action readout이 적용되지 않았다는 뜻이므로 해당 방법의 0 counter를 NoOp 선택률로 해석하지 않아. |
| `final_hypothesis`, `generating_hypothesis` | 각각 실제 F의 최종 W/D 가설, 선택한 bank action을 만든 가설. PERM은 혼합된 결합이라 생성 가설을 `PERM`으로 표시해. I는 단일 생성 가설이 없어 `null`이야. |
| `raw_error_bin` | 기존 원시 frame 평균 거리의 `<=5`, `(5,10]`, `>10` pixel 구간. 최종 TSV와 STRATA는 RAW 행만으로 다시 산출하며 방법별 결과로 구간을 바꾸지 않아. |
| `reference_distance_m` | 참조 중심의 camera 원점까지 거리. 실제 DEV에서는 같은 2D 레이블 기반 재구성 참조이며 독립 거리 계측이 아니야. |
| `A_*_METRIC_ROWS.tsv` | 위 scalar를 합친 행 단위 표. `supervised_corners`는 평가 참조 코너 수이며 학습에 실제 사용한 횟수가 아니야. `observed_corners`는 조건부 관측 거리 수야. 산출하지 않은 `IoU3D`나 실패 오차는 빈칸으로 남겨. |
| `A_*_ORACLE.json`, `A_*_ORACLE_CORNERS.json` | 실제 F를 모든 action에 적용한 ADDsym 최소 단일 후보와 바로 그 선택의 이동·회전·2D 거리·손상. CORNERS 보완은 저장된 index/bank를 재사용하므로 새 F 호출은 0회야. |
| `A_*_ORACLE_GAPS.json` | 같은 frame·bank의 실제 선택 ADDsym minus oracle ADDsym. 실패는 `null`과 별도 실패 수로 남겨. J의 유한 gap은 잠긴 수치오차보다 음수일 수 없고, I는 더 큰 조합 공간이라 음수일 수 있어. |
| `A_*_COMPARISONS.json` | 각 seed 원결과의 paired 손상·자세 차이와 frame별 seed 평균 차이의 95% bootstrap CI. 실제는 session, synthetic 주 분석은 frame, 10,000회/seed 20260917을 공유해. synthetic scenario secondary는 미실행으로 명시해. |
| `A_manifest.json`, `A_fits/*.json`, `A_smoke/*.json` | 실제 입력·코드·bank·체크포인트 해시, 동일 초기화/순서, update·노출·제외·시간. 큰 bank/가중치는 저장소 밖 cache에 보관해. |
| `A_ANALYSIS_REAGGREGATION.json` | 실제 fresh 분석 명령/time·재사용·추가CNN/최종F/update0과 반복 realbank 생성의 초기PnP를 구분한 영수증. 초기PnP 개별호출은 이 분석에서 미계측이야. |
| `results/A_SAMPLING_MASK_RECEIPT.json` | 원실행GPU mask tensor가 보존되지 않은 한계를 표시하고 CPU FP32 좌표 연산으로2304frame의기존support/coverage를재구성해. 원시 validmask·affine·box/inputbounds·32점stencil·actioncounts만사용하며GT가시성은사용하지않아. 실제realbank재생성의초기PnP1276/RefineLM1276과CNN/최종F/update0을분리해. mask와bank대형배열은외부cache에보관하고parentmanifest가이보조영수증을바인딩해. |

[확인] 실제 DEV의 모든 방법은 전체 319개 frame에 F를 적용하고, 자세 산출 319/319와 2D 매칭 311/319를 따로 보고해. 이 319개 참조는 독립 계측이 아니야. source oracle의 실패 frame도 1,031개 전체 분모에 남아 있으며 원시/NoOp 실패를 유한한 0오차로 대체하지 않아.

[확인] 손상은 승인된 평가 branch 이후 원래 GT 코너 identity로 정렬해 비교해. `good5_to_bad10`은 기준선 <5px 코너가 새 결과 >10px가 된 개수, `bad20_to_good10`은 기준선 >20px가 새 결과 <10px가 된 개수야. 유효 관측·가시성 mask는 물리 가림 자체의 증거가 아니며 물리 occlusion 정도는 미측정으로 표시해.

[추정] GEO/PERM의 비교는 이 유한 bank에서 결합을 유지/깨뜨린 대조의 범위만 뒷받침해. I/J는 같은 점별 logits의 decoder 비교야. 새 방법과 기존 N3/PoseFix의 비교는 후보·손실·decoder가 함께 다른 방법 전체 비교이며 독립 센서 정확도의 확증이 아니야.
