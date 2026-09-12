# 최종 권고 — 새 방법 승격 없음

`RECOMMENDED_PAPER_PATH = NO_NEW_METHOD_FREEZE_EXISTING_STORY`

이번 실행은 C의 구현 오류와 이후 GPU 자원 충돌 때문에 **제한적으로 종료**했다.
모든 트랙을 완전한 3-seed 학습으로 검증했다고 주장하지 않는다.

- A: 기존 image_line_only seed1을 고정했다. 기존 DEV 부분 개선은 보존하지만
  original overall gate 미통과와 joint-vs-line-only 미확정 결론도 보존한다.
  독립 confirmation population이 입증되지 않아 확인 실험은 NOT_RUN이다.
- C: 실제 forward/gradient 분리는 가능했고, 단회 갱신에서 frozen tensor와
  raw keypoint 불변성이 통과했다. C0/C1 seed1 학습·평가는 완료했다. 그러나
  제가 작성한 최종 BN 감사 코드 오류로 C2 seed1의900회 갱신 후 checkpoint를
  저장하지 못했다. C2 성능 실패가 아니라 **실행 실패**다. 재실행 시 예산을
  초과하므로 임의 재학습하지 않았다. 이후 다른 프로젝트가 GPU를 사용해
  남은 fit은 기다리지 않고 NOT_RUN으로 남겼다. C2-vs-C0/C1 결론은 보류다.
- D: teacher 미노출 synthetic1985장을1187/417/381로 나눴고 trust3x1500회를
  실행했다. 모든 seed에서 안전한 calibration threshold가 없었다. 평균 이득이
  양수여도 harmful edge 비율이25% 기준을 넘으므로 학생 학습을 하지 않았다.
- B: 중앙 Jacobian 오차는 작았지만 canonical W/D 선택 전환을 포함한
  catastrophic 비율6/256=2.34%가1% 기준을 넘었다. 후속 loss 학습은 없다.
- E: 이전 불량 reference 세션을 제외해도 depth boundary usable coverage가
  23.43%로80% 기준에 미달했다. depth teacher와 학생 학습은 없다.

따라서 지금 허용되는 논문 주장은 strong synthetic baseline, 이미 수행한
controlled adaptation, local-line의 development 결과, 이번 mechanism
진단이다. C의 한 seed C0/C1 관찰은 탐색적 관찰로만 추가할 수 있다.

금지: 독립 일반화 확인, 새로운 method 우월성, C2 실패 확정, novel loss,
geometry teacher의 학생 개선 입증, DHT/자기학습/RGB-D 자체의 불가능성.

다음 강한 주장에는 새 independent data가 필요하다. 수집 프로토콜은
CONFIRMATION_DATA_PROTOCOL.md에 분리했다. C 재개에는 GPU 가용성과 잃어버린
C2 fit의 추가900-update 예산 승인이 필요하다. 이 문서는 재개 승인을 대신하지
않으며 D3 adapter, 새 threshold, seed/epoch sweep은 실행하지 않았다.

실제 학생 optimizer 갱신2700회, 저장·평가된 학생 checkpoint2개, trust 갱신
4500회다. 고정 파라미터와 source 보존 감사는 범위를 명시해 기록했으며
`_docs/paper/final/`은 수정하지 않았다. Git 완료 여부는 최종 CLI SHA 확인으로
보고한다. 과거 PAUSE_AUDIT/EXECUTION_BLOCKER는 중간 사건 기록으로 보존한다.
