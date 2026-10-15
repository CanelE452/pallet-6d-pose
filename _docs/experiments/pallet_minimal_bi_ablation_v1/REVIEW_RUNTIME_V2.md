# 최소 B/I 런타임 검토 V2 보충

**요청한 입력·캐시·계측 게이트 수정은 확인됐으며 수학 변경은 없다.** [V1 검토](REVIEW_RUNTIME_V1.md)는 그대로 보존했다. [V2 JSON](REVIEW_RUNTIME_V2.json)은 실행 구성 `54de240fa7fca0f38738a60c7957a24018461b5ed0ae828f35473cf91fa0c199`와 그 소스19개의 SHA를 확인한 보충 검토다. 기존 과학 정본 소스26개도 그대로이고 사용자 보호 파일6개는 검토 종료 시 모두 SHA가 일치했다.

| V1에서 남긴 운영상 제한 | V2 확인 |
|---|---|
| 입력 self-seal과 실행 protocol의 직접 연결 부족 | `_input(input_dir, config)`가 입력 payload와 seal의 SHA를 protocol과 비교한다. 관측·자세·좌표 교체 진단·계측이 이 경로를 사용한다. |
| 기존 관측/자세 캐시 내용 검증 부족 | 관측 result digest와 마스크 byte SHA를 재검산한다. 자세 단계도 관측 digest, 관측 binding, 진단 flag, 자세 result digest를 검사한다. |
| 계측 전 예측 봉인의 존재만 검사 | 압축 예측 전체 SHA, protocol SHA, 238개/952행을 봉인과 비교하고 그 뒤에만 계측 출력과 모델을 준비한다. |

기존 INPUTS와 sanitized V2 INPUTS의 전체 byte SHA는 `291449b15a8d1116e130ee23ed7e937791905bf339fce97e04b343d9d5f8d824`로 같다. V2 입력 seal은 `60f90ed1df90f97f0445eeb433b9aa54276d18c9758cabb7586c7f68d9c93807`이다. 이 확인은 opaque byte 해시만 사용했고 실제 입력 행이나 참조를 다시 해독하지 않았다.

수정한 네 함수의 원 AST로 합성 검사를 구성했다. 입력/봉인 불일치와 self-consistent 입력 변경, 관측 digest/마스크 byte/binding 오류, 자세 digest/관측 binding/진단 flag 오류, 계측의 예측 byte/protocol/행수 오류를 포함한 **18검사/0실패**다. 기존 batch 계약 테스트도 별도로4 PASS/0 FAIL(1.08초)이었다. 새 GPU·NN·PnP 실행과 실제 예측/참조 행 읽기는0이다.

[저장된 parity의 이산값 보충 검사](PARITY_DISCRETE_ADDENDUM.json)는 기존10장×두 초기화만 순차 재검산했다. solver configuration20개와 base configuration20개를 포함한 설정 객체60개가 전체 dict로 정확히 같았다. 상태·native ID와 순서·source·정수 rank/index/candidate 식별·수락/fallback 등 이산 leaf153,775개와 sequence 길이61,227개에 불일치는0이었다. 이는 기존20개 표본의 추가 검사이며 새20개 독립 표본으로 합산하지 않는다. 설정 이외 실수 결과에는 이미 먼저 잠근 pose/기존 audit 허용치를 유지한다.

계측의 전체 RGB 추론 시간은 계속 **NA**다. R0/N0는 own-F pass-through API 비용이고 R1/N1은 메모리 RGB·own q·own F에서 최소 B/I 관측과 같은 BI-only 자세 계산까지다. YOLO·N3/SubPix·초기 F 실행시간이나 캐시 시간을 더한 전체 시간으로 부르지 않는다. `float32 neural / float64 geometry`는 계산 dtype 설명이며 위치 정확도 주장과 다르다. 별도 실제 계측이 끝나기 전 속도/FPS/Jetson 성능이나 경로 간 속도 우위를 주장하지 않는다.

남은 범위 제한도 명시한다. 자체 result digest는 우발적 손상을 검사하며 payload와 digest를 함께 바꾸는 행위까지 인증하지 않는다. 통제된 append-only 소유권과 최종 불변 봉인이 필요하다. 계측의 byte gate가 예측 저널 전체를 직접 해독해 ID·arm을 재집계하는 것은 아니며 그 검증은 별도 owner/scorer의 책임이다. own-F에 조건부인 S, 초기 F의 기존 채널 영향, 상태 모델의 학습과 RAW 적용 범위, 원좌표 fallback과 실제 보정·모델 투영의 구분도 V1과 같다. 이 검토는 전체238장 정확도나 채택 권고를 확정하지 않는다.

검토 중 frozen Python 파일, V1 검토, 원자료, main, 논문은 수정하지 않았다. 공개 보충 문서에 로컬 절대경로는 추가하지 않았다.
