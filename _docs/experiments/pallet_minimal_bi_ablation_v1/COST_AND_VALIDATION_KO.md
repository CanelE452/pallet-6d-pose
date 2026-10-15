# 실제 비용·검산·보존 범위

판정은 **UPSTREAM_STILL_USEFUL**다. 단일 구성으로238장×4=952행, 별도384개 좌표 진단까지 완료했다. 모든 실패와 큰 오차를 남겼고 운영 e·main·논문을 변경하지 않았다.

## 실제 측정 경계

고정 CAL6장, batch1, 경로별2회 예열, 영상당5회다. 기존 통계 seed20260917로 경로 순서를 균형 배치했다. RGB·자기 q·자기 F는 메모리에 준비하고 상태/마스크/B/I/최종pose 전체를 실행했다. 파일 decode와 모델 loading은 측정 밖이며 첫 예열에 lazy 모델 loading이 들어갈 수 있다. PyTorch CUDA를 전후 동기화했다. NN float32, 기하/solver float64, TF32/autocast 없음, BLAS/OpenMP/OpenCV/Torch1 thread였다. 장비·라이브러리 실제 버전은 TIMING.json에 있다.

| 경로 | 측정수 | 평균±SD ms | 중앙값 ms | p95 ms | 측정 경계 |
|---|---:|---:|---:|---:|---|
| R0 | 30 | 0.181±0.178 | 0.171 | 0.628 | 준비된 F 복사 API만 |
| R1 | 30 | 2476.041±1727.372 | 3205.028 | 4384.796 | 실제 B/I 보정 블록 |
| N0 | 30 | 0.225±0.194 | 0.199 | 0.622 | 준비된 F 복사 API만 |
| N1 | 30 | 2472.183±1843.577 | 2691.297 | 4835.888 | 실제 B/I 보정 블록 |

R0/N0 숫자는 초기 F·YOLO/N3의 시간이 아니다. 전체 RGB→YOLO→앞단→F를 연결 실행하지 않아 **전체 추론시간/FPS/Jetson은NA**다. 저장 시간을 합산해 전체 실측으로 만들지 않았다. 현재 비교로 앞단 제거의 총 지연 감소를 확정하지 않는다. 고정6장은 전체192장의 시간 분포 보장이 아니다. N3와 SubPix의 개별 비용도 분리하지 않았다.

모든120회와 실제 단독 RAW CLI의 최종 pose·가설·수락/fallback은 정확도 결과와 일치했다. 최대 GPU allocated294.36MiB/reserved414MiB는 PyTorch 프로세스 계측이며 시스템 전체 VRAM 수치가 아니다. 기존 디스플레이/RustDesk 작업은 유지했다. 무거운 gallery render는 시간 측정 종료 뒤 실행했다.

메인 R1/N1은각238회 observe/상태head/ResNet/Tiny/SAM encoder·decoder/B/I/회복/pose를 실제 실행했다. N3/SubPix/O/T builder/U 표시 호출은 보정 블록에서0이다. N0/N1 앞단은 기존 캐시다. [실제 계수](MAIN_ACTUAL_CALL_COUNTS.json)와 [전체120회](TIMING.json)를 확인할 수 있다. 실제 RAW CLI에서도 N3 자산·모듈·SubPix 접근을 차단했다.

## 실행량과 자원

| 실행 | 실제량 |
|---|---:|
| 주/CAL 입력 | 238장·952경로 |
| 메인 새 B/I 관측/pose | 각476회 |
| 좌표만 복원한 조건부 pose | 384회 |
| 모든 실제 시도 관측 | 585회 |
| 실제 B/I pose 및 원본 solver 반복 | 959회 |
| 초기 F 입력 준비3시도 | 714회(기존102 대조×3 + 누락136 계산×3) |
| 고유 누락 RAW F | 136개 |
| 새 초기 YOLO/N3 forward·fit·optimizer update | 0 |

585회 관측에는 메인476, 선택10 parity20, 경로 수정 전 CAL12, 메모리probe12, 단독CLI1, 비용측정/예열64를 포함한다. solver959회는 메인476+조건부384+parity20+이전CAL12+CLI1+비용64+원본반복2다. 합성 계약 fixture 호출은 과학적 원영상 수와 분리했다. 세 입력 준비 모두 같은 INPUTS JSON 바이트였고 기존 시도를 지우지 않았다.

계측 GPU 관련 stage 합은 **426.890초(7.11분)**, CPU solver/analysis stage 합은 **776.154초(12.94분)**다. GPU 혼합 작업의 전 구간도 CPU 예산에 보수적으로 중복 분류하면 **17.03분**이다. 병렬 pose worker CPU 합은 **2152.681초(35.88 CPU분)**이며 경과시간과 다르다. 원장에는 메모리probe 결과의 중복 영수증을 실행이 아닌 메타데이터로 명시했다.

인터프리터 시작/종료와 일부 짧은 소스/실패probe는 별도 세부타이머가 없다. GPU 실행 전체가 포함된 최초GPU 로그 생성→마지막GPU 종료 후 coordinate 시작의 보수적 구간은 **28.91분**으로60분 상한 안이다. 상기 stage 합을 전체 프로세스 CPU시간이라고 부르지 않는다. 코드/복원/검산을 포함해 기록 가능한 초기화부터 게시 전 검산까지 벽시계는 **55.56분**이며 초기 파일 읽기의 일부·이후 commit/push 소요는 별도 로컬 마감 영수증에 남긴다. 모든 실제 모델 실행은 원장 범위 안이고 동시 연구GPU1개/pose worker최대3/BLAS1을 유지했다.

## 검산과 실패 이력

- 기존 solver 반복2회와 저장 pose 차이0을 먼저 확인해 pose1e−9/rtol0, audit scalar1e−8/rtol1e−8을 고정했다.
- 기존10장×2 실제 관측/pose, mask bitwise, 전체 설정60개와 이산 leaf153,775개 exact. 최종0/7/10/10 회귀도 일치했다.
- 최종 계약59개 PASS. 전체952행 통계/전환/6녹화 bootstrap 독립1,800개와 조건부384 결과/xy·비xy필드 독립12,860개 검사 모두0불일치다.
- 실제32개 과정 PNG, 링크109개와 byte/표기230검사 PASS. 기존 공개 RGB 바이트 inventory의238장 coverage를 확인하고 선택4개만 갤러리에 재사용했다.
- 메인/진단/시간 실행의 기술적 실패행은0이며 알고리즘 fallback은 결과에 그대로 포함했다. 초기 테스트 collection 상대 import, synthetic fixture index, Path.open guard, sparse 의존2개와 selector 누락, test monkeypatch 순서 오류는 실행 접합 문제로 수정 후 재검사했다. 입력 builder의 로컬 경로 의존과 중간 self-source-key 수정 전3시도, 독립 좌표검사의 사전 shape/schema probe2개도 보존했다. 이 오류를 알고리즘 실패로 바꾸지 않았고 임계값/가중치는 바꾸지 않았다.

원본 로그·소스 snapshot·실패·모델 관측·마스크·전체 solver 후보는1.3GB의 지속 로컬 ARTIFACT_ROOT에 보존했다. 이전 실행 alias도 유지했다. 공개물은 전체 지표/예측/작은 요약과 허용된4개 영상/32단계 그림이다. 원본 manual 좌표·가중치·대형 raw 배열을 새로 push하지 않는다. 사용자 dirty6개와 기존 실험·논문 SHA를 확인했고 index에 섞지 않았다.

[실행 영수증](EXECUTION_RECEIPT.json), [원장](RESOURCE_LEDGER.jsonl), [최종 테스트](FINAL_TESTS.txt), [독립 통계](INDEPENDENT_RECOUNT_KO.md), [좌표 검산](COORDINATE_INDEPENDENT_CHECK.json), [재현 명령](REPRODUCE.md)을 함께 보존한다.
