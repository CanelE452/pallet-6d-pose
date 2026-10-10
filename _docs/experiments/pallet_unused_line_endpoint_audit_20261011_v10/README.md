# V9 unused line의 native endpoint-heldout 대응 감사

이 V10은 최신 쉬움153+중간92, 총245장의 기존 V9 sparse IMAGE_ROLE 관측을 진단한다. 검사 선의 두 endpoint와 실제 자기 가림 H를 모두 제외한 native N3 대응으로 기존 강건 PnP를 구하고, 그 선과의 고정8px endpoint RMS 일치 여부를 기록한다. unavailable·ambiguous는 UNKNOWN이며 NONE 감독이나 전체 frame 실패가 아니다.

새 최종 pose·LOCAL filtering·GT score·모델/학습/RGB·runtime을 실행하는 블록이 아니다. V9의 관측과 실제 결과를 수정하지 않으며, REJECT라는 진단 label도 관측 제거를 뜻하지 않는다. V9의 실제 위치·회전 개선 목표는 아직 달성되지 않았다.

전체245장/370고유 unused line을 실제 한 번 감사했고 독립441191검사를 PASS했다.265 SUPPORTED/67 CONTRADICTED/38 UNVERIFIED를 보존했다. 이는 native 독립 관측과의 일치/모순/검증 불가이며 물리 ownership 정답이나 새 최종 자세 개선을 뜻하지 않는다. 독립 PASS 뒤 조건별 요약도 자체 고정/실제1회 완료했다. 공개 검토·게시 권위는 별도로 확인한다.

- [한국어 보고 구조와 V9 실패의 구분](RESULT_KO.md)
- [사전 고정 감사 계약](AUDIT_CONTRACT_KO.md), [재현 명령](REPRODUCE.md)
- [실행량/후단 상태](BUILD_LEDGER.json)
- [전체245 audit행](AUDIT_ROWS.jsonl.gz), [245ledger행](AUDIT_LEDGERS.jsonl.gz), [실제 계수와 cleanup](AUDIT_RECEIPT.json), [완료seal](AUDIT_SEAL.json)
- [15조건×3층의 저장 endpoint/status/outcome 요약](DIAGNOSTIC_SUMMARY.json), [245진단원행](DIAGNOSTIC_FRAME_ROWS.jsonl.gz)
- [독립 scalar441191검사](VALIDATION_CHECKS.json), [기존 V9 정상 게시 SHA](PARENT_V9_PUBLICATION_RECEIPT.json)
- [관측 packet auditor](../../../scripts/research/pallet_unused_line_endpoint_audit_20261011_v10/audit.py)
- [전체245 parent 검증·quiet guard·PnP streaming driver](../../../scripts/research/pallet_unused_line_endpoint_audit_20261011_v10/runner.py)
- [독립 stdlib scalar checker](../../../scripts/research/pallet_unused_line_endpoint_audit_20261011_v10/validation_checks.py)
- [이미 완료된 V9 결과·원행·648통계·81CI·600runtime](../pallet_sparse_local_line_20261010_v9/RESULT_KO.md)

V9 정상 게시와 remote SHA 확인 후 최신 보호 snapshot을 만들고 V10 자체 protocol/preflight/run을 실제 수행했다. 원행·bank 공유·실제 primitive/J/SVD 진입·partial 실패·cleanup을 보존하며 threshold나 solver를 결과에 맞춰 반복하지 않는다. native solve370·Generic24313·LM887·project332·Rodrigues114658·J332·scoped SVD74669의 실제 실행량을 기록했다. V9 LOCAL 저장 witness 복사는 새 LOCAL/J/SVD/objective0이며 wall39.8702초는 replay 실행량으로 배포 latency가 아니다.

모순 선이 있는54frame(23NEW)의 저장 N3 outcome은2개 모두 개선/12개 모두 악화다. 모든 선이 supported인122frame(71NEW)에서도36개가 T/R 모두 악화했다. observed weak3개는 모두 악화했으나 이 진단은 ownership 정답이나 인과 효과를 증명하지 않는다. 현재 V11의 최소 후속 수정은 코드 준비이며 새로운 filtered pose 성능을 얻은 것으로 쓰지 않는다.
