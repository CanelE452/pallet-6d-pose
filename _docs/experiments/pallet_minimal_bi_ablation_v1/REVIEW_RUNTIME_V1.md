# 최소 B/I 런타임 소스 검토

판정은 **계산 동등성 범위 PASS, 전체 정확도 판단 보류**다. [검토 JSON](REVIEW_RUNTIME_V1.json)에 검토 시점의 소스 SHA와 범위를 고정했다. 이 검토는 ROOT가 완료한 기존10장×두 가지 초기화의 실제 관측·자세 parity를 확인 근거로 사용하며, 그20회 검사를 다시 실행해 합산하지 않았다.

| 단계 | 유지한 계산 | 제거한 의존성 |
|---|---|---|
| 기하 | 각 입력 자신의 F 자세·CF 치수·등록 Q에서 S와 B/I 역할 결정 | 과거 E 자세/bank |
| 상태 | 기존 실제영상 상태 분류기·고정 RGB 특징·ImageNet ResNet 공간 특징 | N3 가중치·N3 특징 캐시·N3 출력 파일 |
| 마스크 | 각 초기화의 non-S 좌표로 새 Tiny와 MobileSAM 실행 | 이전 초기화의 상태/마스크 재사용 |
| B/I | 기존 v2 교정과 v4_3b B 경계 회복, 공분산·지지/거부 유지 | 표시 전용 U 모델·U 영역 계산 |
| 자세 | 기존 JOINT 전체 기본값과 analytic Jacobian, own-F 시작점·동일 fallback | O/T 영역 생성과 영역 비용, C, 보고서/점수 스택 |

[runtime.py](../../../scripts/research/pallet_minimal_bi_ablation_v1/runtime.py)는 관측과 자세 API를 분리한다. O/T는 자세 입력에서 U로 바뀌고 S/O/T/U는 2D PnP 대응점이 아니다. 보관한 B/I 측정 XY와 최종 모델 투영은 별도 필드다. 원측정 fallback, 실제 교정, 모델 투영을 같은 것으로 표시하지 않는다.

[계약 테스트](../../../scripts/research/pallet_minimal_bi_ablation_v1/test_runtime.py)는27 PASS/0 FAIL이다. S 프롬프트/영상 질의 차단, O/T/U의 대응점 제외, 입력 불변성, 빈 관측과 부족한 관측의 행 보존, 잘못된 공분산 거부, 고정 Q와 fallback, 파일 접근 차단, N3 없는 loader, 별도 관측/투영, 단일 CLI와 계측 복원성을 포함한다. 초기 수집 오류와 두 테스트/guard 수정 이력은 별도 로컬 구현 영수증에 보존했다. 실제 parity 시작 뒤 runtime 수식 변경은0이다.

ROOT의20개 parity는 관측·상태·ID 순서·분기 일치, 마스크 배열 bitwise 일치, 자세 atol=1e−9/rtol=0을 통과했다. 기존 관측 audit 실수 scalar는 사전에 고정한 atol=1e−8/rtol=1e−8이다. 사용하지 않는 `ot_domains`, `u_region_display`만 비교 대상에서 제외했다. 이 허용치를 결과를 본 뒤 완화하지 않았다.

`instrument(components)`는 실제 함수 호출과 실제 neural forward hook을 센다. lazy 모델을 미리 실행하지 않으며 컨텍스트 종료 시 wrapper와 hook을 복원한다. 고정 CAL6×두 초기화12관측에서 각 상태 head/ResNet/Tiny/SAM encoder·decoder/B/I/회복 호출이1회씩이었다. N3/SubPix/OT/U는0회이고 금지 자산 접근/금지 import 시도도0회였다. 이는 고정 조건에서 R1 후단이 N3 자산을 요구하지 않는 근거다. 기존 초기 F를 만들 때의 전체 upstream 계산을 실행하지 않았다는 점과 구별한다.

같은 CAL6×두 초기화로 별도 CUDA 메모리 probe를1회 실행했다. 새 PnP·참조 읽기는0회이며, 기존 파일럿의 상태·측정/공분산 및 두 마스크 배열은 전부 정확히 일치했다. 최대 allocated는308,659,200B(294.36MiB), reserved는434,110,464B(414MiB), 프로세스 경과는5.252초였다. 이는 resident 모델과 첫 lazy loading을 포함한 해당 프로세스의 PyTorch 메모리이며 GPU 전체 사용량이나 정확도 근거가 아니다. 결과 영수증 SHA는 `afcc6a916c99dcfce8d334cea0973280b9a4b373280a9ac4a63bbac0df2e9751`이다.

[batch.py](../../../scripts/research/pallet_minimal_bi_ablation_v1/batch.py)의 관측 캐시 키는 전체 frame·branch·고정 protocol SHA를 포함한다. 자세 캐시는 관측 binding과 좌표 대체 진단 여부를 확인한다. 각 분기에서 상태·프롬프트·마스크·측정을 다시 계산하며 모델 가중치 캐시만 공유한다. 초기 기준선의 accepted/fallback은 NA이며 알고리즘 fallback과 기술적 예외를 구분하고 행을 보존한다. 원238개의 ID/분할을 바꾸는 성공 기반 분기는 보이지 않았다.

[timing.py](../../../scripts/research/pallet_minimal_bi_ablation_v1/timing.py)는 같은 고정 CAL6에서 각 경로2회 예열과 영상당5회 측정을 사용한다. 파일 해독·해시 계산은 측정 밖이고 CUDA 동기화와 실제 호출 계측을 포함한다. R0/N0는 보관한 own-F 결과를 반환하는 API 비용만 측정한다. R1/N1은 메모리 RGB+own q+own F에서 B/I와 최종 자세까지다. 전체 YOLO→N3/SubPix→F 시간을 재지 않으므로 전체 추론 시간은 NA이며 이 계측으로 Jetson 실시간 성능을 주장하지 않는다.

남은 제한은 다음과 같다.

- S와 B/I 역할은 own-F에 조건부다. 초기 F 자체가 숨은 예측 채널의 영향을 받지 않았다고 주장하지 않는다.
- 상태 모델은 실제 CAL 학습 이력과 적은 외부 O 감독을 갖는다. RAW 출처에 대한 새 교정 범위 검증·재학습은 없다.
- 원좌표 fallback도 B/I 관측 목록에 남는다. `corrected=True`가 반드시 0보다 큰 이동 또는 정확한 보정을 뜻하지 않는다.
- resume는 메타 binding을 확인하지만 모든 기존 result/NPZ byte를 재해시하지 않는다. 입력 self-seal과 protocol의 입력 SHA를 매 단계 직접 비교하는 범위도 제한되어 있다. 통제된 불변 작업공간과 최종 owner의 byte 검증에 의존하며 실제 변조는 관찰하지 않았다.
- timing의 PRED seal gate는 존재 확인이다. 전체 예측 byte/행 재검증은 별도 owner/scorer가 담당한다.
- 이 경로는 B/I-only 대조다. 기존 V33의 유한선·O/S/C 전체 방법을 재실행한 결과가 아니다. 전체238장 정확도·채택 판단은 이 검토 범위에 없다.

직접 import된 legacy16개 소스, 과학 정본 protocol의26개 소스, 입력 ancestry7개 원소스가 기준과 일치했다. 사용자 보호 파일6개의 SHA도 모두 일치했다. 검토 중 frozen 소스·원자료·논문·main 변경은0이다. 공개 문서에 로컬 절대경로는 추가하지 않았다. 기존 입력 영수증의 경로 표시는 STATE가 별도 V2로 정리하고 V1은 로컬 이력으로 보존한다.
