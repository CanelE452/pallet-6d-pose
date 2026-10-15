# 최소 B/I 직접 연결: RAW 대 N3/SubPix

**완료 · UPSTREAM_STILL_USEFUL.** PRIMARY192 공동 성공은 R0 66, R1 82, N0 108, N1 108이다. RAW에 B/I를 붙인 이득은 있지만 N3/SubPix 제거를 권고할 근거는 부족하다. N1은14회복/14손상이고 연속 오차가 악화했다. 운영 e·main·논문은 변경하지 않았다.

- [최종 결과·4개 모집단·네 대비·여섯 답변](RESULTS_KO.md)
- [실제 과정 그림 32개](gallery/GALLERY_KO.md) · [로컬 HTML](gallery/index.html)
- [전체952행](ROW_METRICS.csv) · [모든 전환](CHANGED_CASES.csv)
- [조건부 좌표 교체384건](COORDINATE_DIAGNOSTICS_KO.md)
- [실제 비용·실패 이력·보존 범위](COST_AND_VALIDATION_KO.md)
- [독립1,800개 집계 검사](INDEPENDENT_RECOUNT_KO.md) · [좌표12,860개 검사](COORDINATE_INDEPENDENT_CHECK.json)
- [재현](REPRODUCE.md) · [사전 계약](PLAN_AND_CONTRACT.md) · [구성 결정](COMPONENT_DECISIONS.md)
- [실행 영수증](EXECUTION_RECEIPT.json) · [소스/모델 SHA](MODEL_AND_SOURCE_MANIFEST.json)

R0=RAW+F, R1=RAW+최소 B/I, N0=N3+SubPix+F, N1=N3+SubPix+같은 B/I다. 주 성공은 위치≤5cm AND 회전≤5°다. CAL46은 주192장과 분리했다. 원영상은238개이며952개의 새 영상이 아니다.

연구 브랜치는 `research/minimal-bi-ablation-20261016-v1`이다. 대형 관측/마스크/solver 기록과 모든 시도는 별도 지속 로컬 ARTIFACT_ROOT에 보존했다. 공개 예측은 최종 모델 출력이며 수동 정답 좌표를 포함하지 않는다. 전체 RGB 추론시간과 Jetson/FPS는NA다.
