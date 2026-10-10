# V11: endpoint 관측 필터와 약한 기하의 안전 반환

**목표 미해결:** 같은 Clean153+Moderate92의245장에서 SAFE 평균 T/R은12.611230982495721cm/11.787036654409325°이며 N3의9.754548028679153cm/10.914841863166323°보다 나쁘다. 기존 LOCAL보다 오차는 줄었지만 새 방법의 성공으로 보고하지 않는다.

[결과·전체972moment/135CI·손상·그림](RESULT_KO.md), [실행 원장](BUILD_LEDGER.json), [수학/관측 계약](METHOD_CONTRACT_KO.md), [전체 경로 비용 계약](RUNTIME_CONTRACT_KO.md)을 함께 읽는다. 평가 참조는 기존 N3-phase GEOMETRIC_PROXY이며 independent physical GT/unseen test가 아니다.

[전체 기하490행](GEOMETRY_SEALED.jsonl.gz)과 [공유1fit245ledger](CONTROL_LEDGERS.jsonl.gz), [새 실제 scoring490행](PREDICTIONS.jsonl.gz), [1470행 집계](METRICS.json), [집계 CSV](METRICS.csv), [조건부 원행490](POSTHOC_ROWS.jsonl.gz)을 공개한다. 원 후보/초기화/J/상태/H와 출력 좌표를 함께 확인할 수 있다.

[원 검산 중단95](VALIDATION_CHECKS.json)와 [optional-field 한개만 보완한 별도 검산 PASS394452](REVIEW_VALIDATION_CHECKS.json)을 모두 보존했다. [통계 독립 검산](VERIFICATION.json)은972moment/135CI를 확인했다. 원 실패 receipt를 PASS로 바꾸지 않았다.

[8PNG/36panel 근거](FIGURE_BINDINGS.json), [실제 8개 이미지 검토](VISUAL_REVIEW.json), [case 원행](FIGURE_CASE_ROWS.jsonl.gz)이 저장한 영상 SHA/출력과 연결된다. [6case](figures/case_01.png)와 [전체 운용 그림](figures/01_all_operational.png)을 보고 NEW와 fallback 및 H를 구분할 수 있다.

원 SAFE runtime의 packet id 중복 [0행 실패](RUNTIME_RECEIPT.json), metadata 수정 후 첫 detector 계수 guard의 [1행 실패](runtime_review/RUNTIME_RECEIPT.json)를 모두 보존했다. 첫 warmup 내부 준비 forward를 정확히 계수한 별도 [성공 경로](runtime_forward_review/RUNTIME_RECEIPT.json)는 fresh150호출/공식130측정, 전체 mean149.6915392153846ms/P90229.74461220000003ms다. [독립 검산](FORWARD_RUNTIME_CHECKS.json)은33,905검사/56 timing n·moment PASS이며 수학·시계·순서를 바꾸지 않았다. 기존 가중치·원본 RGB가 필요한 새 실행과 공개 저장 증거 검토를 구분한다. 공개 fresh복원/최종 보호/게시 상태는 최종 receipt가 권위이며 여기서 완료를 미리 주장하지 않는다. 대상은 research/observation-refiner-robust-pnp-20261009이며 main을 바꾸지 않는다.

V9/V10의 코드·수치·실패 자료를 수정하지 않았다. 현재 새 학습·RGB·seed0이고, H 초기 좌표는 최종 fit에서 제외하며 승인된 NEW 후 최종 R,t 투영으로 교체한다. UNKNOWN endpoint는 유지하고 잘못된 mask 하나로 프레임을 실패시키지 않는다.

[후속 공개 자료 검토·fresh 복원·게시 권위](PUBLIC_REVIEW_KO.md)는 이 결과 snapshot과 별도 기록이다. 실제 완료 receipt가 생기기 전에는 해당 단계를 완료로 해석하지 않는다.
