# 기존 주석과 인계 자료 연결 결과

완료했던 **12장·96점과 12개 실제 대상 확인을 그대로 재사용**했습니다. 다시 클릭할 작업을 열지 않았습니다. 원래 120장·반복 24장의 공식 가시 코너 평가 완료로 바꾸지는 않았습니다.

| 범위 | 완료 자료 재사용 | 사용자 제외 | 참조·실제 반복 기록 대기 | 합계 |
|---|---:|---:|---:|---:|
| 원래 1차 작업 | 12 | 5 | 103 | 120 |
| 원래 반복 작업 | 0 | 1 | 23 | 24 |
| 작업 합계 | 12 | 6 | 126 | 144 |
| 서로 다른 이미지 | 12 | 5 | 103 | 120 |

반복 24개는 24장의 새로운 이미지가 아닙니다. 완료 12장의 이미지 중 6장은 원래 반복 목록에도 들어 있지만, 그 1차 기록을 두 번째 실제 검수처럼 복사하지 않았습니다. 사용자 제외 5장 중 `174925:620`은 반복 목록에도 있어 작업 기준 제외는 6개, 이미지 기준 제외는 5장입니다.

기존 결과가 사용한 처음 저장한 점은 직접 클릭 66점 + PnP로 채운 30점입니다. 나중에 저장한 직접 좌표 72점 + 자체 가림 상태 24개는 **다른 버전**으로 보존했습니다. 13개 초안 중 이번 12장 밖의 초안 1개도 원래 평가에 넣지 않았습니다.

전달된 과거 주석 312개(같은 네 세션 이름의 67개, 다른 촬영의 245개)의 실제 JSON 크기·SHA와 저장된 점 출처를 검산했습니다. 같은 세션 이름의 67개는 로컬 Git blob과 바이트가 같습니다. 이번 수신 PC에서 실제 기존 ZIP 멤버와 바이트가 같았던 JSON은 245개, 인계 SHA와 같았던 원사진은 245개입니다. 로컬에 풀린 동일 JSON은 199개였습니다. 원래 수신 PC의 두 ZIP 본체와 245개 멤버를 모두 검증한 실행 기록은 CONNECTION.json에 개별 경로·SHA와 함께 남깁니다. 기존 인벤토리의 점 출처 1,271개 직접 클릭 / 1,225개 PnP도 실제 저장 필드와 맞습니다. 원래 120장에 연결 가능한 과거 픽셀·코너 대응 확정은 0개입니다.

`173507:2910`, `173507:3210`은 파일명 기반 후보로 유지합니다. 이전 도구의 `camera_dynamic_0123_v4` 번호와 정확한 원사진 연결이 아직 확인되지 않아 정답으로 승격하지 않았습니다. 다른 촬영 245장은 인계된 CPU 픽셀 감사가 원래 촬영과 다름을 기록합니다. 이번 재실행은 영상이나 PNG를 다시 디코딩한 픽셀 감사가 아니라 실제 저장 바이트의 동일성 검산이며, 원본 ZIP이 없는 다른 PC에서는 그 검산을 수행했다고 표시하지 않습니다.

## 실제 수신 기록에서 확인한 공식 참조 조건

수신 PC의 `review/LIFTER_REFERENCE_REVIEWED.json`과 실제 일괄 승인 `APPROVAL_RECEIPT.json`이 없습니다. 현재 원본 snapshot의 이번 12개 record는 `status=draft`, `reviewer=null`, 최상위 `evaluation_use=false`입니다. 기존 코드의 `reference_gate`와 `_validate_reference`를 직접 호출하면 각각 `WAITING_HUMAN_REFERENCE_SUBMISSION`과 미승인 bundle 거절을 반환합니다. 72개 좌표와 각 코너 상태가 저장되지 않았다는 뜻이 아니라, 저장한 입력을 현재 계약의 공식 참조로 제출한 기록이 없다는 뜻입니다. 기존 실제 사람 이력 답변과 대상 판정 12개는 보존했습니다. 새로운 맹검 조건을 추가하지 않았습니다.

## 나중 저장한 72점에 대한 별도 민감도 계산

고정 12장·같은 Base/N3 원시 예측·선택 객체·결측 마스크·고정 ID를 그대로 사용했습니다. 24개 자체 가림 상태에는 좌표를 만들어 넣지 않았습니다. 원래 직접 클릭 좌표 66개는 완전히 같고, 원래 PnP 좌표였던 6개가 나중 저장된 좌표로 바뀌었습니다. 정확도가 좋은 버전을 선택하지 않고 원래 96점과 현재 72점 두 버전을 모두 공개합니다.

**아래 표는 승인 미제출 초안 좌표를 사용한 탐색적 참조 버전 민감도입니다. 공식 가시 코너 정확도나 독립 정답이 아닙니다.** 원래 원고의 96점 핵심 표를 교체하지 않습니다.

| 방법 | 중앙값(px) | P90(px) | PCK≤10px 전체72점 | 유효 / 실패 |
|---|---:|---:|---:|---:|
| Base | 3.649195 | 8.878079 | 67/72 (93.0556%) | 72 / 0 |
| N3 | 3.990919 | 8.611929 | 69/72 (95.8333%) | 72 / 0 |

중앙값 차이 `median(N3)-median(Base)`는 0.341724px, 짝지은 차이 중앙값 `median(N3-Base)`는 0.169492px입니다. 개선 33 / 악화 39 / 동일 0점입니다. 이 버전에서도 중앙값 악화를 유지합니다. 원래 96점 수치 재계산은 원본 결과와 정확히 일치했습니다.

분모 변경 효과와 좌표 변경 효과를 구분하도록 같은 72개 ID에 원래 저장 좌표를 적용한 중간 패널도 JSON에 함께 저장했습니다. 72점과 96점의 결과 차이를 모델 성능 변화로 해석하지 않습니다.

원고에서 유지할 `x`: 원래 전체 120장·반복24장 공식 가시 코너 정확도, 반복 신뢰도, 독립 물리 T/R. 기존 12장 결과는 별도 보조 결과로 계속 쓸 수 있습니다.

- [144개 작업별 연결 CSV](../../../../data/pallet/results/pallet_remaining_evidence_connection_20261006_v1/annotations/FRAME_TASK_CONNECTION.csv)
- [312개 원본 주석별 연결 CSV](../../../../data/pallet/results/pallet_remaining_evidence_connection_20261006_v1/annotations/HANDOFF_SOURCE_CONNECTION.csv)
- [출처·버전·분모와 기존 수치](../../../../data/pallet/results/pallet_remaining_evidence_connection_20261006_v1/annotations/CONNECTION.json)
- [실행 검산 및 실제 CPU 비용](../../../../data/pallet/results/pallet_remaining_evidence_connection_20261006_v1/annotations/VALIDATION.json)
- [원래96점·같은72개ID 원래좌표·나중72개좌표 민감도](../../../../data/pallet/results/pallet_remaining_evidence_connection_20261006_v1/annotations/LATER_DRAFT72_SENSITIVITY.json)
- [나중72개 점별 오차와 원래 좌표 버전 비교](../../../../data/pallet/results/pallet_remaining_evidence_connection_20261006_v1/annotations/LATER_DRAFT72_POINT_ERRORS.csv)

재실행: 원본 인계 ZIP을 풀어 `--handoff`로 지정하고, 완료 자료가 있는 저장소를 `--receiver-root`로 지정합니다. 게시한 `annotations/inputs`도 동일 인계 입력으로 사용 가능합니다. 모델 실행과 새로운 주석 작성은 없습니다.

```bash
python3 scripts/research/pallet_remaining_evidence_connection_20261006_v1/annotation_connection.py \
  --handoff /tmp/pallet-remaining-evidence-handoff-20261006 \
  --receiver-root /home/minjae/Documents/github/pallet-pose \
  --output-root /tmp/pallet-github-publication-20261006-v1 \
  --archive-root /home/minjae/Downloads
```
