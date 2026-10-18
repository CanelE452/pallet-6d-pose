# 채택 판단

이번 no-B와 H1 guarded switch는 채택하지 않는다. 기존 H1·상태별 처리·N3·SubPix·운영 기본값과 논문을 유지한다.

- no-B: PRIMARY192에서 같은 seed 성공120→113,117→113,121→116, 회복/손상3/10·3/7·3/8. seed19 최대 위치208.22cm를 포함해 모두 보존한다.
- guard: 내부 확인을 통과한 seed18도120→116, 회복2/손상6. 19/20은 확인 실패로 H1에 복귀했다. 미채택 trial의 실사 결과를 근거로 다시 선택하지 않는다.
- P2: 앵커 개선의 GT 진단은 유효하지만 자동 실제 성능이나 물리 소속 인증이 아니다. 원 mask encoding/complete scene/자동 typed 공급기에는 공백이 남는다.
- P4: 생성 가능한 인과적 coverage 공백과 원 입력이 미확정이며, 실제5시도 후 남은4회로 matched6fit을 실행할 수 없어 미실행했다.

다음 수정 한 가지는 **B 질의 앵커·검색 영역을 인증 가능한 target 유한 edge 대응 계약에 연결**하는 것이다. 원 관측이 실제 edge인지/다른 물체인지/복수 전환인지 모르면 위치 범위와 불확실성을 유지한다. 이번 실사로 가중치·문턱·seed를 재탐색하지 않는다.

[전체 근거](REPORT_KO.md) · [네 모집단 전체 표](FULL_RESULTS.md) · [원수치](COMPARISON_ALL_ARMS.csv) · [나쁜 seed 이미지](gallery/extremes/index.html)
