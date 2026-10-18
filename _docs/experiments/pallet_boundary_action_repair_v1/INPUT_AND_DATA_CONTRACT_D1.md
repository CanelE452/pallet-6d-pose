# D1 입력·좌표·참조 경계

`pipeline_contract.build_upstream`은 기존 prediction-only join의 raw RGB 경로·해시, N3/SubPix image keypoint, N0 F pose, native XYZ, K만 export한다. 그 파일에 함께 있던 역사적 E0·state·관측·candidate bank는 export하지 않는다. 합성도 원 pre-state cache의 같은 정보만 export한다. 원 GT sidecar는 열지 않는다.

각 입력은 닫힌 top-level schema를 사용한다. `reference`, `truth`, GT pose/visibility/box 등 미등록 payload는 RGB 접근 전에 거부한다. 생산 함수는 image/native prediction/known registration과 고정 모델만 받는다. 최종 특징은 원 점·선·마스크·고정 비용 및 N0 차이에서 계산하며 ID/session/material/성공/GT 오차/oracle 순위는 model input이 아니다.

- R_cf는 native cuboid→camera 회전이다. native cuboid는 중심 원점이며 m 단위다.
- R_physical=R_cf@Q이며 Q는 예측이 선택한 native W/D basis와 알려진 물리 model basis의 등록이다. 중심 centroid는 camera m다.
- image keypoint와 K는 저장된 원영상 좌표다. synthetic는 기존 provision의 100px padding 제거가 이미 반영됐으므로 다시 제거하지 않는다.
- crop/letterbox에서 온 좌표는 upstream source seal에 연결한다. 새 producer가 GT box로 영상을 자르거나 표시 crop를 측정 좌표에 반영하지 않는다.
- 실사 238와 합성 1003 모두 기존 N3/SubPix/F 캐시를 정확히 재사용한다. 이 캠페인에 새 detector/N3/SubPix forward는 필요하지 않았다.
- 모든 synthetic 학습 label은 별도 GT sidecar와 개별 source C1/C2로 만든다. synthetic C1/C2와 실사 C2를 바꾸거나 90°를 평가 대칭으로 허용하지 않는다.

실제 model descriptor·weights·CAL model·runtime config 해시, producer 및 원 inference helper source 해시는 실행 seal에 저장된다. 이 해시는 파일 동일성의 증거이며 물리 owner나 실제 6D GT의 인증이 아니다.

실행 명령은 `python -B -m scripts.research.pallet_boundary_action_repair_v1.d1_batch observe ... --root-GO`와 별도 `solve ... --root-GO`다. observe는 GPU 단일 작업이고 solve는 명시한 CPU worker 수를 사용한다. frame별 seal이 있어도 입력/소스/출력 hash가 다르면 완료 cache로 사용하지 않는다. 참조 scorer는 common feature/prediction seal 뒤의 별도 단계다.

최종 CPU 실행은 `d1_memoized_batch`의 shared dispatcher를 사용한다. 기존 V3 CPU wrapper와 3개 producer 파일은 수정하지 않는다. 추가 adapter와 dispatcher source, 원 143장 입력/단계 동일성 검사, 실제 1장 exact parity, 원 batch의 안전 drain receipt는 `D1_CACHE_TECHNICAL_AMENDMENT_LOCK.json`으로 연결한다. feature bank와 별도의 참조 없는 `d1_parity_audit`가 실제 E0 실행/재사용 경로, N0/E0 alias, 원 S 제외 및 전체 분모를 재검산한다. 재사용은 원 실행 비용이 0이었다는 뜻이 아니며, 이번 재사용 비용과 과거 실행 비용의 측정 범위를 분리한다.

중단 후에는 `D1_RECOVERY_TECHNICAL_AMENDMENT_LOCK.json`이 기존 source·입력·관측과 완료 692행을 묶는다. 원 dispatcher를 수정하지 않고 새 namespace에서 실행했으며 원 lock/journal/log는 그대로 보존했다. `D1_RECOVERY_FINALIZATION_RECEIPT.json`은 새 batch receipt의 canonical exact copy와 두 최종 feature seal을 연결한다. 독립 검산은 두 journal 1,201행의 중복 0·전체 분모·원 703개 source와 13개 실제 모델 파일 SHA를 확인했고 PASS했다. 공개된 분석 코드와 결과 원장은 입력 절차를 검증하며, 별도 참조 scorer가 계산하는 자세 정확도와 구분한다.
