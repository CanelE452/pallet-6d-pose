# PURPOSE — PCR-DOPE (corner-role discriminative feature)

[소비처]
"팔레트 내부 corner identity 를 feature 수준에서 분리하면 confident-wrong 이 줄어드는가"
의 판정.  `_docs/audits/PCR_GO_STOP_GATE.md`.  Gate 실패 시 base ep57 유지.

[문장]
"symmetry-aware corner-role objective 로 학습한 role feature 는 GT corner 와
팔레트 위 wrong-peak 를 구분하며, 이 feature 로 stage 4~6 입력을 modulate 하면
real F2 signed bias 와 PnP 실패가 감소한다."

## 판단 지표 (사전 고정)

Gate A(capacity): proto acc >=0.95 / structural >=0.95 / GT>student-wrong >=0.90 /
GT>teacher-wrong >=0.90 / corner error -50% / centroid +10% 이내 / NaN 0
Gate B(synthetic): proto acc >=0.80 / structural >=0.80 / GT>teacher-wrong >=0.70 /
C1 대비 corner median +10% 이내 / centroid +5% 이내 / collapse 없음
Gate B(real): session-fold 최소 AUC >=0.70 / GT>baseline-wrong >=0.70 /
GT>student-wrong >=0.65 / F2 far -5% 또는 >50px tail -5% / PnP >=70 /
reproj +5% 이내 / near +5% 이내 / 새 >100px 0

## 실행 범위

- belief/affinity/decoder/PnP 변경 없음.  FiLM 은 shared feature 만 modulate.
- FiLM zero-init 시 ep57 과 bit-exact identity.
- N87 접근 2회(Gate B real one-shot, Gate C final)만 허용.
- 기존 실패 경로(residual/proposal/graph/line/router/view/DiffPnP) 재도입 없음.

## ★ 사전 기록한 deviation

Gate B 의 group-disjoint split 은 scene key 가 복원되는 5 root 에서만 구성한다.
`mixed_v8_train` 은 JSON 에 scene/sequence 식별자가 전무하고 파일명도 flat 이라
group 을 정의할 수 없다(Phase I1 우선순위 1~3 모두 불가).  Gate C full canonical 에는
6 root 를 그대로 사용한다.  `aug_*` 는 파일명(`squash_b000_000006_v0`)으로
v4_split_base 원본에 귀속시켜 같은 group 으로 묶는다(매핑 600/600 확인).
