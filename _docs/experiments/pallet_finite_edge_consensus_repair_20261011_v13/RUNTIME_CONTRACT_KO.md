# V13 fresh 단일 FINITE_EDGE_CONSENSUS의 전체 경로 시간 계약

새 [deployment.py](../../../scripts/research/pallet_finite_edge_consensus_repair_20261011_v13/deployment.py)와 [runtime.py](../../../scripts/research/pallet_finite_edge_consensus_repair_20261011_v13/runtime.py)의 실행 전 계약이다. 이 문서를 작성한 단계에서 model·영상 decode·fit·시간 측정·GT·원행 산술 실행은0이며 아직 runtime 결과가 없다. Root가 전체 source와 peer 검토를 마친 뒤 별도로 protocol을 동결하고 실제 단독 실행한다. V13 정확도 kernel·관측·solver·runner를 고치지 않는다.

기존 corrected RUNTIME_PANEL의13세션에서 각2장, 총26장을 그대로 쓴다. 원 순서20warmup은 i%26, 공식 측정은 동일26장×5회=130회로 총150번 fresh predict다. 새 영상·학습·seed·성능으로 고른 case·추가 warmup·자동 retry가 없다. GPU가 필요하며 stage마다 CUDA synchronize와 perf_counter_ns를 쓴다. 설정/자원 조건이 다르면 첫 실패와 prefix를 남긴다.

## 한 실제 호출에 포함하는 것

V7의 변경 없는 detector/N3/IMAGE_ROLE 모델 lifecycle과 capture를 재사용한다. 호출마다 초기 N3 자세·자기 가림 H, 원 Base 예측에 대한 역할 특징의 초기 자세, 원 ROLE head/query decode를 새로 계산한다. 같은 sparse q의 강건 point PnP를 새로 구하고 그 결과는 finite LOCAL의 시작값에만 공급한다. 실제 native N3 endpoint LOO와 unused line 필터를 새로 수행한 뒤 한 번의 V13 FiniteEdgeConsensusBank.solve가 단일 출력만 만든다. 그 bank 안의 proposal·실제 inlier refit dispatch는 모두 시간에 포함하며 ledger로 각각 센다.

최종 fit에는 실제 sparse q와 코너에 소비되지 않은 실제 선만 쓴다. missing native 코너 fill0, 초기 H 2D fit0, initial-pose 잔차 prior0, 같은 선의 점/선 중복0이다. 승인 NEW이면 변경 없는 V2 assemble이 최종 R,t로 H를 재투영하며 그 투영을 다시 fit하지 않는다. typed 관측 부족·수치 실패·약한 관측·다중 branch는 기존 출력/fallback 상태로 남긴다. NEW가 전역해·물리 ownership·정확도 개선을 뜻하지 않는다.

V2 packet을 그대로 호출하되 새 deployment 모듈 안에서 id/session/method/K/xyz/raw_hw/selected_index/fixed_metadata의8개 예약 필드만 기대 metadata와 semantic SHA가 같은지 검증한 뒤 payload에서 제거한다. 기존 V11의 검증된 동일성 adapter와 같은 규칙이며, 수치 결과를 바꾸지 않는다. detector 후보의 선택·confidence/class·비선택 좌표·center8 보존과 최종 prediction metadata 조립도 실제 호출 안에 포함한다. 한 call은 한 finite bank·한 output이며 과거 V11의 두 view 조립 가정을 쓰지 않는다.

## 시계와 실제 계수

측정 입력은 RAM의 원 RGB·K·물리 치수·id/session/object_type뿐이다. synchronized t0 직후 predict부터 최종 반환 직후 synchronized t1까지를 full로 저장한다. 저장 기하나 endpoint 결정·pose receipt를 predict 입력으로 넘기지 않는다. 실제 초기 자세 계산, 강건 PnP, native endpoint 검증, 모든 finite proposal/refit와 진단·branch 처리·H 투영·출력 metadata·복사 비용이 full에 포함된다. 기존 시간의 합산이나 캐시 replay를 latency로 쓰지 않는다.

6개의 실제 stage tick은 detector/N3_correction/initial_pose/boundary_observation/endpoint_validation_and_filtered_local/final_pose_reprojection_and_metadata다. stage 동기화 비용도 full에 포함되고, 마지막 tick에서 return까지도 별도 저장한다. 130측정의 n/평균/표본분산(ddof1)/SD/중앙/P90/최대를 저장한다. 모델 로딩·26원영상 decode·file SHA·로그·반환 후 parity·자원/보호 probe는 t0–t1 밖이다. 전체 작업 wall은 latency가 아니다.

검출기의 최초 예측 warmup에는 기존 Ultralytics 내부 준비 forward1회가 더 들어간다. 따라서 첫 call은 detector2/N3 1, 나머지149call은 detector1/N3 1, 총151/150을 처음부터 검사한다. 모델 로딩 직후 detector/N3가0/0인지도 기록·검사하며 내부 준비를 로딩 비용으로 옮기지 않는다. artificial pre-forward나 별도 dummy 이미지가 없다. 첫 call의 준비 비용은 원래20warmup 구간 안에 포함된다.

각150call의 전체 output·원 detector 모든 candidate·point bank·finite repair ledger·execution event·t0/t1·6tick·실제 model/head/OpenCV/SciPy/경로 계수 delta를 원행에 저장한다. runtime wrapper는 V13 solver의 least_squares dispatch/return/exception/success 및 residual/J callback을 실제로 계수한다. proposal+refit 계수를 local 총계와 맞추고 entry=return+exception을 확인한다. 처리된 수치 예외를 없애거나 success로 바꾸지 않는다. OpenCV6 primitive도 같은 entry/return/exception 계약으로 센다.

P13의 native bank before/after에서 delta를 새 deployment에서 도출하며 모든 값이 음수가 아닌지 확인한다. 원 endpoint observer event는 그대로 복사한다. 그 observer가 local 이벤트를 내지 않으므로 새 deployment는 repair 호출 시작/반환과 local path 계수를 직접 기록하고 ledger의 실제 solve1·view1을 확인한다. 새 이벤트를 optimizer dispatch로 해석하지 않는다.

## 입력 권위, parity와 보존

runtime freeze 전에 V13 전체245 기하/245ledger seal·cleanup, 원 VALIDATION의 complete=True/passed=False 713,802검사·74실패를 그대로 보존하고, 별도 REVIEW_VALIDATION의 전체 독립 PASS와 후단245 scoring 완료를 evaluation_review.geometry_gate 및 receipt의 SHA로 확인한다. 이 supplement는 한 endpoint의 부동소수점 cusp에서 독립 검산의 한쪽 미분 branch만 검토하며 수치 solver·저장 자세·정책·성능 threshold를 바꾸지 않는다. 원 V11 실패 supplement를 새로운 성공 권위로 요구하지 않는다. 자체 RUNTIME_PROTOCOL은 R13.inputs, V7.inputs의 가중치/칼리브레이션 전체 closure와 새 runtime/deployment/계약/panel/normalized helper/geometry gate/원 checker/별도 review checker/원 FAIL3파일/실제 review3파일/cusp 계약/scoring authority/seal/ledger 및 원행 SHA를 묶는다. protocol을 끝에 다시 검증한다.

normalized_frames는 원 cohort identity와 canonical IMAGE input을 연결한다.26개 원 RGB는 파일 SHA/shape를 검사하고 한 번 decode해 측정 밖 RAM에 보관한다. 저장 V13 기하의26개 points/actual_pose/H/status/fallback/fixed_metadata는 반환 **후에만** parity에 쓴다. 원 Base/N3와 실제 fresh detector 후보·metadata도 반환 후 대조하며, 비교값을 fit이나 시작값으로 넣지 않는다. 원 전체245 기하의 단일 method와 끝 extra row 없음도 확인한다.

자체 freeze/preflight/measure의 출력은 새 V13 DOC에만 exclusive 생성한다. 첫 오류·실행량·PENDING/INTERRUPTED 원행·중단 capture를 보존하며 자동 재시도나 overwrite가 없다. 성공이면150원행이 close/flush/fsync/promote된 뒤 공식130summary를 인수한다. 첫 error와 cleanup/stream 오류는 따로 남긴다. 실패 prefix의 시간은 공식 성능 수치로 사용하지 않는다.

전/중/후 resource quiet와80°C 미만을 실제 probe로 확인하고 다른 측정과 겹치지 않게 한다. 모델 hook·OpenCV/SciPy wrapper·CV thread·pipeline·환경 변수 원상복구와 보호 snapshot 동일성을 완료 뒤 확인한다. probe 사이의 순간 경합이나 하드웨어 진위까지 증명하지 않는다. 원 영상·가중치·사용자 변경·V7~V12 코드/결과는 보존한다.

```bash
python -m scripts.research.pallet_finite_edge_consensus_repair_20261011_v13.runtime freeze
python -m scripts.research.pallet_finite_edge_consensus_repair_20261011_v13.runtime preflight
python -m scripts.research.pallet_finite_edge_consensus_repair_20261011_v13.runtime measure
```

필요한 source/baseline/기존 가중치/칼리브레이션 인자는 원 runner parser의 동일 옵션으로 명시한다. 원 정확도 PROTOCOL과 별도의 --runtime-protocol을 사용한다. 아직 실행하지 않은 이 계약이나 code-only 준비를 실제 runtime 완료로 보고하지 않는다.
