# 실제 취득 절차와 필요한 의존성

[확인] 이번 상태는 취득 패킷 `DONE`, 실제 취득·독립 참조·독립 평가 `BLOCKED_DATA`야. 필요한 것은 같은 대상과 시각에 연결된 실제 clean/실물 가림 pair, 카메라 교정, 독립 자세 계측, 외부 변환 실측, 반복·불확실성 자료, 출처를 확인한 사람 검수야. 현재 미연결 126행이나 후속 탐색 72좌표를 새 자료량으로 요구하지 않아.

## 측정 목적과 조건 고정

[추정·미검증] 배포에서 사용하는 거리·방향·가림 종류를 먼저 정하고, raw 관측의 지원 범위와 기존 13세션 개발 결과의 잘됨/큰 오류/실패 조건을 근거로 취득표를 작성해. 결과를 본 뒤 유리한 조건만 남기면 안 돼. 거리·yaw는 센서 범위와 실제 운용 경로로 결정하고 기준서에 수치·이유·작성 시각을 남겨. 이번에는 보편적인 최적 거리나 임의 가림률을 새로 정하지 않았어.

[추정·미검증] 각 고정 조건에서 카메라·팔레트·계측 표지를 단단히 고정해. clean 이미지를 얻고 실제 사람/판/현장 장애물을 광선 경로에 넣은 occluded 이미지를 얻어. 가림 물체를 넣을 때 카메라·팔레트·표지를 건드리지 않아야 해. 합성 masking은 별도 진단 자료로 기록하고 이 manifest의 `physical_external`에 넣지 않아. 자기 가림·영상 밖 점은 외부 가림과 구분해. clean→occluded 순서만으로 조명/시간 효과가 고정될 수 있으므로 실제 취득 순서와 조명 변화를 기록하고, 허용되면 순서를 교차한 반복 취득을 기준서에 먼저 정해.

[추정·미검증] 같은 pair·같은 실제 자세·같은 녹화·인접 구간은 하나의 split에 넣어. 캘리브레이션이나 파일 복사가 녹화를 독립 세션으로 바꾸지 않아. 모델을 바꾸거나 고른 자료는 development로 남겨. independent_test는 각 모델의 학습/선택 녹화 및 실제 자세 ID와 겹치지 않아야 하고, 그 자료의 첫 모델 노출보다 모델/선택 규칙 잠금이 빨라야 해. 독립 확인용 결과를 읽은 뒤 수정한 모델은 같은 자료를 다시 독립 확증으로 쓰지 않아.

## 독립 참조 설치와 상대 자세 확인

[확인] 「Estimating the Pose of a Euro Pallet with an RGB Camera based on Synthetic Training Data (2022/Logistics Journal: Proceedings)」 §4.1, 인쇄 p.5–6은 별도의 optical motion capture(광학 동작 계측)로 카메라·팔레트 기준을 얻고 RGB 영상 시퀀스를 재생하여 비교했어. [공식 원문](https://proc.logistics-journal.de/article/download/1038/1009/8188)에서 확인한 입력·참조 설정이야. 그 장치의 명목 정밀도를 이번 데이터의 검증된 불확실성으로 가져오지 않아.

[추정·미검증] 사용할 수 있는 장치에 맞춰 optical motion capture, laser tracker(레이저 추적 계측), 또는 모델 입력 영상과 독립적인 surveyed tag rig(위치·축이 실측된 태그 계측 장치)를 설치해. 팔레트 기준 중심/축과 카메라 광학 중심/축까지의 외부 변환을 별도 교정해. 태그가 있다는 것만으로 독립 참조가 생기지 않아. 모델 점이나 같은 2차원 레이블의 Perspective-n-Point(PnP, 2차원-3차원 대응으로 자세를 푸는 방법)에서 만든 자세, 모델 출력으로 역산한 태그 장착값은 제외해.

[추정·미검증] `T_A_B`는 B 좌표를 A로 보내는 4×4 강체 변환으로 정의해. 두 물체를 world에서 측정한다면 `T_camera_object = inverse(T_world_camera) @ T_world_object`야. 카메라 표지→optical 및 팔레트 표지→정준 팔레트 중심/축 변환을 이 사슬에 포함해. 각 변환의 방향·실측 방법·단위·원문 파일·SHA-256·날짜·공분산을 교정 근거 파일에 적어. 변환을 계산하는 장치 adapter는 해당 계측 장치의 원문 좌표에서 이 계약으로 변환한 값을 저장해야 해.

[확인] 최종 계약은 OpenCV optical +X 오른쪽/+Y 아래/+Z 전방, 팔레트 직육면체 중심 원점, metre 단위야. 운영 치수 [W,D,H]는 pose xyz=[W,H,D]로 한 번만 매핑해. 기존 `pose.py`가 반환한 `R_physical`과 같은 정준 팔레트 축을 사용해. W/D 가설의 camera-facing `R_cf`에 정준 점을 곱하지 않아. 기하 담당의 계약 검토도 같은 매핑과 Y축 proper symmetry group을 확인했어. 기존 구현 출처는 `scripts/research/pallet_dim_conditioned_p_v1/pose.py`야.

[추정·미검증] clean 전부터 occluded 후까지 카메라와 팔레트를 독립 계측해 상대 자세 trace를 저장해. trace는 그 전체 구간을 덮어야 하고, 사전에 정한 계측 간격 상한을 지켜야 해. 각 샘플은 실제 원문 record ID와 연결돼야 해. 상대 이동/회전의 허용치는 목표 오차 정밀도와 참조 불확실성에 근거해 먼저 정하고 `same_pose_tolerance.basis`에 남겨. 끝점만 같거나 바퀴 명령이 정지인 것을 전체 구간 정지로 바꾸지 않아.

[확인] validator는 trace 구간 coverage·샘플 간격·동기·모든 기록의 상대 이동/회전 한계를 검사해. 측정값 drift가 기준 이하인 것과 참조 오차까지 고려한 실제 상대 자세 보존은 구분해야 해. 사람 검수에서 drift의 불확실성까지 포함한 보존 범위를 근거 파일에 확인하고, 미확인 구간은 이 독립 평가에 넣지 않아. 희망 공차에 맞춰 원문 값을 고치지 않아.

## 카메라·시간·레이블

[추정·미검증] 실제 해상도/초점/렌즈 설정에서 camera K와 왜곡을 교정해. calibration 사진, fitting 방법, 잔차, 반복성, 유효 조건과 계수 불확실성을 보관해. manifest는 pixel K와 `none` 또는 `opencv_brown_conrady` 모델을 명시해. 원영상이 undistorted이면 보정 후 K/좌표와 그 preprocessing을 기록하고 왜곡을 중복 적용하지 않아. 현재 기존 wrapper의 영 왜곡 계약과 맞지 않는 새 영상은 호환된 정상 inference 경로를 먼저 검증해야 해.

[추정·미검증] camera 센서 clock과 계측 clock을 교정된 하나의 time base로 변환해. 원문 timestamp, offset/drift 추정, trigger/동기 방식, 오차 상한, 보간 정책을 time_sync_calibration 근거에 남겨. nominal frames per second(초당 프레임 수)로 시각을 추측하지 않아. 각 영상 시각에 대응하는 계측 record ID와 변환 후 timestamp를 저장하고 허용 동기 오차를 검사해. timestamp는 float milliseconds 대신 integer nanoseconds로 기록해.

[추정·미검증] 사람 검수 파일은 팔레트 instance 일치, 실제 외부 가림 여부, 촬영 변동, 직접 가시 코너의 x/y, 검수자/시각·수정 이력을 기록해. 직접 본 좌표는 `direct-visible`, 가려진 점을 독립 pose에서 투영했으면 `occluded-projected`, 육안으로 추정한 숨은 점은 `hidden-estimated`, 모르면 `unknown`, 화면 밖은 `out-of-frame`이야. 투영점이나 추정점을 직접 관측 레이블의 정확도 분모로 섞지 않아. 이번 B CLI의 주 지표는 독립 6차원 참조의 최종 자세 오차이며 2차원 클릭 정확도를 새로 만들어 계산하지 않아.

[추정·미검증] 대칭은 실제 외관/구조/허용 pose 모호성을 검토한 근거 파일로 승인해. 기본 C1은 항등변환 하나, C2는 정준 Y축의 0°/180°, C4는 0°/90°/180°/270° proper 회전이야. [W,D]가 같다는 이유만으로 C4를 승인하지 않아. 물리 자세 보존 검사에는 symmetry를 적용해 실제 회전을 숨기지 않아.

## 불확실성과 자료량

[확인] 「Evaluation of measurement data — Guide to the expression of uncertainty in measurement (2008/JCGM 100)」 §4.2–4.3/§5.2/§7.2에서 반복 관측, 다른 근거의 불확실성, 공통 참조의 상관, 결과·공분산 보고 범위를 읽었어. 같은 교정값을 쓰는 clean/occluded 및 모델 비교에서 공통 오차를 독립 잡음으로 합산할 수 없어. [공식 원문](https://www.bipm.org/documents/20126/2071204/JCGM_100_2008_E.pdf)을 이번 취득 양식에 적용한 범위야.

[추정·미검증] uncertainty_budget.template.tsv에 참조 장치 교정·반복성·시간 동기·카메라/물체 표지 장착·치수·광학 교정 성분을 기록해. tangent 순서는 [tx,ty,tz,rx,ry,rz], translation=m, rotation=rad이고 회전은 카메라 축의 작은 좌측 perturbation으로 정의해. 공통 성분은 clean/occluded cross covariance에 반영해. `C_delta = C_clean + C_occluded - C_cross - transpose(C_cross)`를 사용하며 참조 불확실성을 0으로 채우지 않아. validator는 두 개별 공분산, 12×12 결합 공분산, 차이 공분산의 양의 준정부호를 검사해. 실제 오차 분포·비선형 변환에 따른 불확실성 전파는 장치 교정 담당의 근거가 추가로 필요해. 이번 evaluator는 참조 공분산을 보관·검증하고 지표 신뢰구간에 자동 전파하지 않아.

[확인] 기존 319장은 13세션의 반복 사용 개발 자료이고 119장은 단일 세션이야. 기존 `PAIRED_POSE_ANALYSIS.json`에 기록된 SHA-256과 현재 로컬 `YOLO_SCORES.json`이 일치하는 것을 확인한 뒤, 같은 frame/session의 N3−Base 이동·회전 차이를 seed별로 재집계했어. 각 세션에서 프레임 차이를 평균하고 그 뒤 3개 seed를 평균한 13개 세션 통계의 표준편차는 이동 5.0141cm/회전 3.4968°였고 범위는 각각 −2.5841…+16.7871cm/−9.6388…+4.4040°였어. [39개 세션×seed 집계 행](../results/B_REUSED_DEV319_SESSION_VARIATION.csv)과 [정의·해시·전체 값](../results/B_REUSED_DEV319_SESSION_VARIATION.json)을 남겼어. 새 PnP는 0회야.

[확인] 위 수치는 재구성된 2차원 레이블/기하 참조의 모델 간 개발 차이이고, 실제 동일 자세의 clean/실물 가림 오차 차이가 아니야. 이를 B pilot 분산으로 사용해 요구량을 계산하지 않았어. 현재 자료의 세션별 이질성이 있으므로 코너/인접 프레임/seed를 독립 n으로 늘리지 않는 근거로만 재사용해. 물리 clean/occluded pair의 이동·회전 차이 분산은 없어. 현재 결과로 B의 필요한 pair 수를 정확한 숫자로 확정할 수 없어.

[추정·미검증] 자료량은 실제 pilot에서 얻은 녹화 단위 평균 paired 오차 차이의 표준편차 `s_D`와 사전에 선택한 평균 오차 추정의 95% 반폭 `h`로 정해. 독립 녹화들의 정규 근사 설계식은 `n_recordings ≈ ceil((1.96*s_D/h)^2)`이고 작은 녹화 수에서는 t 계수/실제 bootstrap 및 실패율 정밀도도 별도 검토해. `h`는 업무상 필요한 cm/degree 정밀도로 지정해야 해. 같은 녹화의 코너·인접 프레임·3개 seed를 독립 n으로 늘리지 않아. 이 근사식을 중앙값/P90의 자료량 보장으로 쓰지 않아. pilot 결과를 모델 수정에 사용했다면 pilot은 development이고 이후 독립 확인 녹화를 별도로 취득해야 해. `precision_target.template.json`에 아직 없는 s_D·h·요구 녹화 수를 null로 남겼어.
