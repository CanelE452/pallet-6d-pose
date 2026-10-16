# 반환 메타데이터 연결 오류의 별도 수정

원 V11 runtime 첫 호출은 실제 Base/N3/IMAGE_ROLE과 sparse point·LOCAL 경로를 실행했으나, `deployment.py:132`에서 V2 `packet()`에 완성된 frame row를 전달해 `id` 등 동일 메타데이터가 중복됐다. `TypeError("dict() got multiple values for keyword argument 'id'")`로 반환 전에 멈췄다. 공식 latency는 없고 반환 완료0/serialized0이며, 첫 model forward·실행량·partial packet·cleanup·실패 receipt를 그대로 보존한다. 이 실패를 새로운 기하 성능 실패나 성공한 150회 측정으로 보고하지 않는다.

새 `deployment_review.Pipeline`은 원 Pipeline을 상속하고 해당 instance의 packet 연결만 감싼다. 원 V2 packet이 빈 결과에서 생성한 metadata와 입력 result의 id/session/method/K/xyz/raw_hw/selected_index/fixed_metadata가 정확히 같은지 semantic hash로 확인하고, 겹친 키만 result에서 제거한다. 원 packet이 동일 metadata를 한 번 붙이며 나머지 관측·후보·R,t·상태·H 재투영·기본 출력은 바꾸지 않는다. geometry core·원 deployment·원 runtime·원 protocol은 바이트 불변이다. 이 보완 코드도 fresh interval 안에서 실행하며 비교·복사 비용을 포함한다.

`runtime_review.py`는 원 runtime 함수에 동일150회(20warmup+26장×5), 동일 가중치·단계·수학·선택·숫자 조건을 적용한다. 원 runtime의 정확도 prerequisites/input closure는 원 accuracy 폴더에서 읽고, 새 runtime protocol/started/raw/resources/receipt는 `runtime_review/`에 독점 생성한다. 원 실패와 protocol·prefight·원행prefix·partial packet, 두 보완 코드와 이 계약을 별도 freeze 입력에 묶는다. 함수 호출 동안 별도 module namespace의 class/출력 연결만 대체하고 종료 후 원 identity를 복구한다. 수학 policy나 accuracy 입력을 수정하지 않는다.

반환 이후 좌표·자세·H·NEW/fallback·center/후보/점수/box metadata와 원245 geometry의26개 panel을 대조한다. 자체150회 완료·typed numerical exception 계수·모델/optimizer/primitive ledger·cleanup 성공 이후에만130개 원 측정 구간의 공식 latency를 기록한다. 원150설정 실패와 수정 후150 실제 완료 여부를 분리한다. 실패 artifact를 덮어쓰거나 자동 재시도하지 않는다. 이번 재실행은 확인된 인터페이스 오류 수정의 검증이며 성능에 맞춘 threshold/seed/학습 탐색이 아니다.

```bash
python -m scripts.research.pallet_endpoint_observability_repair_20261011_v11.runtime_review freeze --source-root /absolute/source --baseline-root /absolute/baseline
python -m scripts.research.pallet_endpoint_observability_repair_20261011_v11.runtime_review preflight --source-root /absolute/source --baseline-root /absolute/baseline
python -m scripts.research.pallet_endpoint_observability_repair_20261011_v11.runtime_review measure --source-root /absolute/source --baseline-root /absolute/baseline
```
