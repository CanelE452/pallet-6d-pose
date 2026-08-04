# PCR_SYNTHETIC_TRANSFER — NOT RUN

Gate B synthetic held-out screen 는 실행하지 않았다.

사유: **Gate A (32-frame capacity) FAIL**.
사전 규칙 "Gate A FAIL 이면 Gate B 실행 금지" 에 따른다.
따라서 이번 작업에서 N87 은 **한 번도 열지 않았다**(허용 2회 중 0회 사용).

판정과 수치: `PCR_CAPACITY_GATE.md`, `pcr_gate_a.json`

참고 — Gate B 를 실행했다면 적용했을 group split 설계(미실행, 기록용):
scene key 가 복원되는 5 root(v4_split_base / aug_squash·trunc·scale_v2 /
paper_4pallet_mask_v1)에서만 구성.  `aug_*` 는 파일명으로 v4 원본에 귀속(매핑 600/600 확인).
`mixed_v8_train` 은 scene/sequence 식별자가 전무해 group 정의 불가.
