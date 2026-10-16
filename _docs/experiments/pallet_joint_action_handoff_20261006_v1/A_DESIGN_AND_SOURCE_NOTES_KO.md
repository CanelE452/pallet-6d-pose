# A 실행 설계와 선행연구 연결

[확인] 새 실행은 기존 `DimensionConditionedPointRefiner`의 실제 YOLO 경로를 상속해. 기존 코드와 radial 222개/null-last 출력 경로는 수정하지 않았어. 동적 bank에서는 NoOp=0을 명시하며 기존 가중치와 학습 가능한 parameter 집합을 유지해. `test_a_contract.py`의 작은 random-weight fixture는 같은 radial 좌표를 NoOp-first로 바꿨을 때 logits의 재정렬 근접성을 검사해. 별도 실제 N3 checkpoint 세 seed QA에서 기존 상속 radial API의 logits와 decoded points는 byte 단위로 같았어. 동적 radial 좌표를 FP32 q에서 다시 빼 metadata를 만들면 trained logits의 최대 차이가 4.482269287109375e-5로 관측됐으므로 random fixture의 2e-6 허용오차를 실제 전체 모델의 exact parity라고 일반화하지 않아. GEO/PERM의 실제 코너별 logits permutation 차이는 9.5367431640625e-7 이내였고 support는 같았어.

[확인] `A_protocol.json`과 `results/A_ID_MANIFEST.json`은 첫 새 metric 이전에 고정했어. synthetic TRAIN의 기존 eligible 55,915개, 이전 source selection 1,031개(새 oracle 진입 판단용 개발 자료), 과거 사용된 SYNTH_HELDOUT 1,985개(이번에도 개발 진단), 실제 DEV 319개/13세션을 분리했어. source selection의 검출 없음 1개도 1,031개 분모에 남아. 과거 이름의 HELDOUT이 새 독립 시험을 뜻하지 않아.

[확인] 원래 `SOURCE_MANIFEST.json` 전체 60,000개와 cache의 60,000개 record index를 전량 대조했어. train55,980/calibration1,004/selection1,031/heldout1,985가 정확히 같아서 특징/no-detection으로 누락된 source 행은 0개야. 55,980 train 중 기존 매칭·유효 target 규칙에 맞는 55,915개를 같은 학습 순서로 사용하고 65개 제외를 별도로 기록해. calibration1,004개는 이번 추가 학습·oracle 진입·평가에 사용하지 않아.

[확인] 이번에 구현·실행한 GEO는 원시 8코너 잔차 위에 object-centred/camera-axis SE(3) 투영 변위를 더하는 후보야. centre8은 고정하므로 강체 9점 투영이 아니야. 같은 좌표·특징·역할·변위·치수 metadata의 코너별 주변 multiset을 고정하고 non-NoOp 대응만 깨뜨린 PERM은 유한 후보의 기전 대조야. 완전한 통계적 독립이나 팔레트 상관 오류의 주원인을 입증하지 않아. I는 각 코너에서 argmax를 따로 골라 bank보다 큰 조합 공간을 만들고 J는 한 action의 argmax야.

[확인] 회전은 camera axes에 평행한 축의 radians, 이동은 camera-frame metres야. 물체 중심 회전에서 `t`는 같이 회전하지 않아. K/q는 원본 pixel, network sampling은 원래 letterbox의 per-axis affine를 사용해. 최신 경로의 두 scale 축은 같지만 벡터로 적용하고 기존 `canvas_affine`/`branch_inputs`와 실제 source/real 16프레임에서 일치를 검사했어. 왜곡은 기존의 `None`/pinhole 계약을 이어받았으며 새 lens 교정 검증을 했다는 뜻이 아니야.

[확인] 첫 metric 전에 반경·방향·허용오차·반복 상한·NoOp 순서·permutation seed를 잠갔어. 원본 cap은 이미지 대각선의 1%, finite-difference sensitivity는 radian/metre 각각의 투영 민감도를 정규화하는 데만 사용해. GT pose/axis/branch는 생성과 점수/readout에 제공하지 않아. 최종 자세는 항상 기존 전체 W/D+SQPnP+LM 함수 F(q)이고 생성에 쓴 T를 채점하지 않아.

[확인] actual source 8개와 real 8개를 사용하는 `results/A_NUMERICAL_PARITY.json`에서 NoOp F 반복의 ADDsym_m 최대 차이는 0m였어. oracle headroom의 1e-7m numerical tie 허용오차는 그전에 protocol에 고정했고, 실용 성능 기준으로 쓰지 않아. 실제 final source/PERM oracle 최소는 한 후보의 ADDsym_m으로 골랐고 이동/회전도 바로 그 후보에서 읽어.

다음 원문은 저장소 밖 `/tmp/pallet-paper-primary`의 저자/공식 proceedings PDF를 읽었어. 해시는 C의 `PRIMARY_SOURCE_READ_NOTES_KO.md`와 다운로드 영수증에 연결돼. 검색 요약을 전문 확인으로 바꾸지 않았고 PDF를 저장소에 넣지 않았어.

| 논문과 확인 범위 | 이번 구현에 적용하는 범위 |
|---|---|
| DeepIM: Deep Iterative Matching for 6D Pose Estimation (2018/ECCV), §3.3 Eq.1–2/§3.5 | 물체 중심·camera-aligned 회전과 이동 분리, rendered/observed matching 입력 및 noisy initial pose 학습을 확인했어. 이번 코너 변위 bank는 rendered-image matching 또는 DeepIM reproduction이 아니야. |
| MegaPose: 6D Pose Estimation of Novel Objects via Render & Compare (2022/CoRL; proceedings 2023), §3/§3.1, 공식 프로젝트 Method | CAD와 관심 영역을 사용하는 coarse hypothesis classification, multi-view rendering refiner의 역할을 확인했어. 이번 알고리즘은 CAD RGB render/compare를 사용하지 않아. 프로젝트 설명과 논문 원문을 구별해. |
| EPro-PnP: Generalized End-to-End Probabilistic Perspective-n-Points for Monocular Object Pose Estimation (2022/CVPR), §1/§3.1–3.2 | correspondence로 매개화한 pose probability density와 target distribution 사이 KL 및 importance sampling 목적을 확인했어. 단순 PnP 미분과 같지 않으며 이번에는 이 loss를 추가하거나 재현하지 않아. |
| Disentangling by Factorising (2018/ICML), §3/Algorithm 1 | 차원별 permutation으로 주변 sample을 보존하며 대응을 깨는 원리를 확인했어. 팔레트 PERM은 이 원리의 새 유한 대조이고 FactorVAE가 팔레트 성능을 입증하지 않아. |

[확인] source oracle 1,031프레임에서 GEO가 raw보다 1e-7m 넘게 나은 후보를 가진 프레임은 954개였고 같은 후보의 ADDsym_m 여지 중앙값은 0.011915445475009378m였어. raw/oracle pose coverage는 둘 다 1,030/1,031이야. 이는 배포 성능이 아니라 고정 bank 안의 GT-assisted 개발 진단이야. GEO 207,031회, PERM 207,031회의 F 호출은 오프라인 비용으로 분리해. 이 비영 여지에 따라 원래 잠긴 두 arm×세 seed, fit당 6,000 update, batch16의 최대 36,000 update를 진행해. 결과가 작거나 음성이어도 예산·seed·cap을 늘리지 않아.

새 JSON/CSV는 `newly_defined_handoff_A_*` 스키마야. 각 metric 행의 `id`, `session`, `method`, `corner`, `pose`, `selected_index`, `final_hypothesis`, `generating_hypothesis`는 이번에 새로 정의한 열이며 기존 파일에 존재한다고 가정하지 않아. 실패한 pose의 유한 오차는 `null`이고 missing frame을 0오차로 채우지 않아. 실제 DEV 참조는 같은 2D 레이블과 치수에서 재구성한 자세이므로 독립적인 계측 정확도 주장을 하지 않아.

[확인] 인위적인 NaN missing fixture의 backward에서 발견한 비유한 gradient는 첫 optimizer update 전에 sampling 위치와 displacement metadata의 내부 안전 좌표/마스크를 적용해 수정했어. 원본 q/NoOp/누락 출력은 그대로 보존해. 실제 eligible TRAIN 55,915개에는 NaN/[-1,-1] 점이 없었어. J는 지원 코너의 score로 한 bank action을 선택하고 그 action의 모든 기존 유효 코너에 적용해 oracle 상한과 일치시켜. 지원이 전혀 없으면 NoOp이고, I의 지원 없는 코너는 원시 값으로 남아. 이 정합성 수정에 사용한 optimizer update는 0회야.
