# V17 hybrid 일반 PnP의 실제 전체 경로 시간 계약

고정 primary `HYBRID_STANDARD_PNP`만 fresh 추론한다. accuracy의 `NATIVE_STANDARD_PNP` paired arm은 이 측정 안에서 실행하지 않는다. 이미 저장한 q·H·LOO 결과·자세를 추론 입력으로 사용하지 않으며 저장 V17 기하는 호출 반환 후 parity 대조에만 사용한다. 알려진 DEV 운용 측정이며 새 물리 ground truth 또는 정확도 평가가 아니다.

1. 같은 RGB에서 원 detector, 고정 N3→cornerSubPix, 초기 N3 자세, Base feature 자세, 고정 IMAGE_ROLE head와 source-CAL decoder를 실행한다.
2. 변경 없는 V7 `outcomes(IMAGE_ROLE_CORNERWISE_HYBRID)`를 실행한다. 원 V4 LOO는 H∪{k}를 제외한 native 관측으로 실제 boundary 대 native 좌표를 비교한다. 선택·좌표·threshold를 바꾸지 않는다. 기존 point-only robust control도 이 원 함수 안에서 실제 실행하므로 전체 시간과 계수에 포함한다.
3. scoped constructor observer에서 실제 selected `PoseBank`를 보존하고 같은 q/K/치수/image-size/H의 `solve(robust=False)`를 한 번 실행한다. 이미 존재하는 같은 numeric problem은 실제 in-memory cache를 재사용한다. 아직 요청하지 않았던 whole-U·차원 문제는 새 Generic 호출을 허용하고 그 key와 실제 cache miss를 기록한다. 이를 무조건 추가 Generic 0이라고 주장하지 않는다.
4. 일반 솔버의 전체 U SSE·최소4관측·whole-U Jacobian rank·기존 numerical tie/multiple-root 정책을 그대로 사용한다. 8px inlier 수는 진단이며 강건 inlier4 게이트를 추가하지 않는다. endpoint/선 factor/LOCAL/SciPy 경로는 0회다.
5. NEW R,t가 있으면 H 초기 XY를 fit·점수에서 제외한 채 최종 재투영으로 한 번 바꾸고 다시 fit하지 않는다. unavailable이면 전체 원 N3를 기본 출력으로 반환한다. center/confidence/box/비선택 candidate를 보존한다.

기존 13 session×2의 26장 패널, warmup20+5×26 measured130=150 호출을 고정한다. 첫 detector 호출의 internal warmup2 forward, 나머지149×1, N3/head150을 별도로 기록한다. 파일 decode·모델 loading·자원/보호 검사·반환 후 parity·logging·GT scoring은 시간 밖이며 detector부터 초기 자세·head·LOO·point control·일반 PnP·H 대체·metadata까지 CUDA 동기화와 perf_counter_ns의 t0–t1에 포함한다. stage5는 LOO/point control/일반 PnP를 합한 단계이며 순수 solver 시간으로 해석하지 않는다. 반환 metadata의 마지막 부분은 return 구간에 포함한다.

원 V16 runtime의 schedule/protect/instruments/delta 함수 전체 AST를 변경 없이 재사용했음을 source SHA와 비교로 기록한다. 그 함수들의 dispatcher는 새 V17 모듈의 명시적 import가 결정하며 실제 추론 경로는 새 deployment 코드에 따로 결속한다. 원 V16 선 미분 검산 complete-but-FAIL564는 원 protocol/STARTED/checks와 그대로 보호한다. V17의 실제 V7 좌표 PASS, 자체 paired geometry seal/정리 완료/독립 ordinary scalar PASS 및 scoring490 완료를 측정 게이트로 사용한다.

새 own runtime protocol 이후 별도 quiet preflight, 실제 측정 중 quiet/thermal 관측, entry·return·exception, head/model/base/fresh/ordinary/bank/SVD와 원행별 cache 계보를 보존한다. 관측 사이의 일시적 경합과 receipt만으로 하드웨어 진위를 인증하지 않는다. 첫 오류·active packet·prefix·restore 오류를 보존하고 자동 retry하지 않는다. primitive·optimizer observer·SVD·constructor·hook·thread·환경을 복원하고 protected bytes를 검산한 뒤 full150 완료시에만 공식 시간 요약을 승인한다.

이 새 source 작성 단계에서는 모델·RGB·PnP·SVD·SciPy·GT·시간 측정이 0회다. root만 검토 후 freeze/run한다. accuracy replay 시간이나 기존 경로 시간 합산을 배포 latency로 사용하지 않는다.
