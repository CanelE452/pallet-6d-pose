[확인] 이 문서는 1단계 진단 및 2·3단계 사전 계약입니다. 기존 예측 코너를 고정하고 W/D 가설 선택만 진단하며, 단계별 실제 결과는 [REPORT_KO.md](REPORT_KO.md)에 기록합니다.

# 방법과 사전 계약

N3 seed 1·2·3, cornerSubPix, 1% diagonal cap, 예측 코너·support·K·치수, SQPnP/RefineLM, proper symmetry metric은 기존 결과와 같습니다. native 차원 순서는 `[W,D,H]`, PnP 고정 차원은 `[W,H,D]`입니다. 원래 F를 실제 호출하여 S0를 복원하고 두 가설의 실제 최종 8점 fit도 저장합니다. [solver.py](../../../scripts/research/pallet_wd_hypothesis_diag_20261010/solver.py), [STAGE1_FINAL_SOURCE_LOCK.json](STAGE1_FINAL_SOURCE_LOCK.json)

S0는 원래 공식 selector입니다. 공식 weighted score의 대안−선택 차이는 penalty를 포함한 nominal px 단위이며 순수 9점 RMSE와 동일하지 않습니다. 단순 선택은 두 최종 fit의 전체 9점 RMSE 최소값만 사용하고 동률이면 고정된 긴 앞면/짧은 앞면 순서를 따릅니다. GT-parity oracle은 고정 참조가 제공하는 긴 축 parity를 선택합니다. 동일 qFinal의 후보 중 metric R이 최소인 후보를 탐색하지 않습니다.

각 모집단의 선택은 해당 참조값을 읽기 전에 qFinal·K·고정 치수·support·source flag만으로 계산하고 봉인합니다. REAL의 봉인·parity PASS가 SYNTH/AUX 계산보다 먼저입니다. 참조는 그 모집단의 봉인 이후 오차·고도·참조 margin·거리와 명시적인 oracle 진단에만 읽습니다. SYNTH 참조 margin은 렌더러 GT 코너의 두 8점 fit 잔차 차이입니다. REAL margin은 기존 GT-builder 값을 유지합니다. [inputs.py](../../../scripts/research/pallet_wd_hypothesis_diag_20261010/inputs.py), [STAGE1_SELECTION_SEAL.json](STAGE1_SELECTION_SEAL.json)

실제 최초 실행에서는 SYNTH GT 준비의 반복 NPZ 압축 해제로 메모리 문제가 발생해 중단했습니다. 봉인된 선택과 원래 core를 보존하고 별도 stage1_resume의 배열 1회 읽기로 scoring만 복구했습니다. 완료한 REAL/SYNTH F 선택 재실행은 0입니다. 중단 전 참조 margin fit 횟수는 NOT_MEASURED이고, 완료된 복구 참조 fit은 3,970회입니다. 이 provenance 제한을 숨기지 않습니다. [STAGE1_POSTSEAL_AMENDMENT.json](STAGE1_POSTSEAL_AMENDMENT.json)

혼동: proper symmetry 최소 R>45°와 |yaw|≥60°. 성공: strict T<5 cm와 proper symmetry 최소 R<5°. 오차는 영상별 seed 평균의 분포, 비율은 seed별 이진 판정 뒤 seed 평균입니다. REAL 319장 CI는 13 session cluster; SYNTH 1,985장 CI는 frame bootstrap입니다. 재표본 수 10,000, RNG seed 20260917, 표본 분산 ddof=1, P90은 NumPy linear quantile입니다. 전체 반복 seed를 957/5,955개 독립 영상으로 세지 않습니다. missing pose의 수치 오차는 available-only이고 비율의 전체 영상 분모에는 false로 남깁니다. [statistics.py](../../../scripts/research/pallet_wd_hypothesis_diag_20261010/statistics.py), [verdict.py](../../../scripts/research/pallet_wd_hypothesis_diag_20261010/verdict.py)

S1은 VIS_RULE_V1의 visible∩supported 코너가 6개 이상일 때 각 가설을 그 부분집합으로 fit하고 부분집합+center RMSE 최소값을 택합니다. 6개 미만은 S0 fallback입니다. S2는 full8와 8개의 leave-one-out7, 두 가설의 18 fit 중 각 부분집합+center RMSE 최소값을 택합니다. S3는 target frame을 제외한 session GT의 median normal·plane offset을 쓰는 사후 camera-fixed feasibility 진단입니다. SYNTH S3도 oracle 상한으로만 읽습니다. 새 코너 이동이나 score coefficient tuning은 하지 않습니다. [rules.py](../../../scripts/research/pallet_wd_hypothesis_diag_20261010/rules.py)

주 경로는 N3_THEN_SUBPIX입니다. 이후 SUPPORTED는 REAL 혼동 변화 CI 상한<0, SYNTH 혼동 평균 변화<0·2개 이상 seed 개선, 두 모집단 모두 성공 변화 CI 상한<0이 아님을 함께 만족해야 합니다. 어느 모집단이든 혼동 변화 CI 하한>0 또는 성공 변화 CI 상한<0이면 WORSENED입니다. 나머지는 UNRESOLVED이고 S3는 FEASIBILITY_ONLY입니다. 3규칙×2지표의 다중 비교 보정은 하지 않습니다. 1단계에는 이 판정을 적용하지 않습니다.

Metric3D-v2는 공식 기본 예제의 ViT-Small/RAFT-4만 미리 고정했습니다. [공식 hubconf](https://github.com/YvanYin/Metric3D/blob/eb5b6fac0dc155e4e52f576e304fbf11655ff339/hubconf.py#L18-L21)와 [Small 입력 설정](https://github.com/YvanYin/Metric3D/blob/eb5b6fac0dc155e4e52f576e304fbf11655ff339/mono/configs/HourglassDecoder/vit.raft5.small.py#L18-L31)의 616×1064 입력·canonical focal 1000을 사용합니다. 원본 RGB는 private에서 비율 유지 resize, RGB 평균값 padding, 공식 mean/std 정규화 후 추론하며, unpad·원래 해상도 보간·scaled fx/1000 곱으로 metric depth를 복원합니다. K는 fx/fy/cx/cy 전처리 계약으로 지원되며 모델에 K 전체 행렬을 직접 넣는 API라고 주장하지 않습니다. [depth.py](../../../scripts/research/pallet_wd_hypothesis_diag_20261010/depth.py)

공식 코드 commit은 `eb5b6fac0dc155e4e52f576e304fbf11655ff339`, HF revision은 `80d2d1410afb4b23cd9d18c6be9144483d4b70b6`, Small 가중치는 150,120,967 bytes/SHA256 `b34b2a2be9148054991cef7e417930e1320602ba7bc503b0ee4e7888543728f6`입니다. 공식 [BSD-2-Clause LICENSE](https://github.com/YvanYin/Metric3D/blob/eb5b6fac0dc155e4e52f576e304fbf11655ff339/LICENSE)와 [저자의 비상업 제한 제거 안내](https://github.com/YvanYin/Metric3D/issues/115#issuecomment-2245665482)를 근거로 연구 사용합니다. 공식 HF 모델카드/API에는 checkpoint별 별도 license 표시가 확인되지 않았다는 제한을 남깁니다. 연구 전용 라이선스라는 표현은 쓰지 않습니다.

[확인] Metric3D v2는 [arXiv 2404.15506](https://arxiv.org/abs/2404.15506)에 TPAMI 2024 게재 정보가 있고 DOI는 `10.1109/TPAMI.2024.3444912`입니다. 첨부의 “publication venue 미확인” 전제를 이 공식 정보로 정정합니다. 코드·checkpoint·K 지원·실제 준비 상태의 근거는 [DEPTH_PREPARATION.json](DEPTH_PREPARATION.json)에 함께 보존했습니다. 공식 최소 GPU VRAM은 확인되지 않았습니다. **준비 시점**의 GPU는 RTX 3080 10 GiB이며 source/hash/import·전처리 검산만 했습니다. 이 준비 근거만으로 실제 추론 가능 여부나 깊이 정확도를 검증했다고 표현하지 않습니다.

준비 시점에는 기존 env를 바꾸지 않고 private target에 mmcv1.7.2/addict2.4.0/yapf0.40.1만 보충했습니다. 기존 torch2.1.1/torchvision0.16.1/numpy1.26.4는 공식 requirements_v2의 torch2.0.1/torchvision0.15.2/numpy1.23.1과 다릅니다. xformers가 없으면 공식 소스가 제공하는 Torch Attention fallback을 사용하며 소스를 수정하지 않습니다. source/hash/import와 전처리 검산만 통과했고 모델 생성·forward는 0이었습니다. 실제 추론과 깊이 정확도는 이후 Stage3 실행 결과로 별도 보고합니다.

깊이는 예측 앞면 quad 0–3을 centroid 기준 0.85배로 축소한 영역의 positive finite depth 중앙값입니다. 모델·원본 RGB·adapter의 SHA는 모델 생성과 추론 전에 봉인합니다. SYNTH 1,985장과 final-test 4 session을 제외한 REAL 231장에서 **네 경로·세 seed별로** 앞면 평균 Z 상대오차의 coverage·절대오차 median/P90·signed bias를 모두 보고합니다.

S4 정확도 gate의 주 경로는 N3_THEN_SUBPIX입니다. REAL 231의 각 영상에서 세 seed의 **절대 상대오차를 먼저 구한 뒤 산술 평균**하고, 이 231개 영상 값의 median을 gate에 사용합니다. signed error를 먼저 평균하여 상쇄하지 않습니다. 주 경로의 어느 영상·seed라도 깊이 추정이 누락되면 ID를 공개하고 gate FAIL로 S4를 실행하지 않습니다. 유효한 subset만으로 통과시키지 않습니다. 완전 coverage에서 이 median>5%면 S4를 실행하지 않습니다.

통과하면 margin m={1,2,3} px 중 **SYNTH 주 경로 혼동률의 seed 평균** 최소값을 택하고 동률은 작은 m으로 고정하여 REAL에 한 번 적용합니다. final-test 4 session은 모델·threshold 선택에서 제외하며 과거 개발 용도를 새 test로 재봉인하지 않습니다. 이 기준은 실제 깊이 추론 전에 고정했습니다.
