# PURPOSE — PAPER_S2 ep57 target-semantics audit

[소비처] 지도교수 보고 및 논문 실험 설계 — PAPER_S2 real 실패 원인을
"모델 용량 부족"이 아니라 target semantics / augmentation 분포 / geometry loss
적용범위 중 무엇인지 판정하는 근거. 이후 admissible한 수정 arm 선정에 사용.

[문장] ep57 학습 데이터에서 belief map 안에 중심이 있으나 Gaussian support가
경계를 넘는 keypoint는 all-zero target과 valid channel mask를 동시에 받아
border-positive가 background-negative로 supervise되었고, truncation 증강은
설계상 모든 코너를 중앙 [0.20,0.80] 밴드로 밀어넣어 real truncation 분포와
어긋났다 — 이 두 가지가 real rear/truncation 실패에 기여했다.

## 범위 / 금지
- 진단 전용. 성능 개선·신규 full training 금지 (게이트 통과 전).
- final-test 접근 금지 (open count 0 유지).
- 기존 데이터/JSON/PNG/체크포인트 읽기 전용, 덮어쓰기 금지.
- 평가 대상: synthetic training/val + strict filter-val. manual36은 별도 표.

## 판정 지표 (Gate)
- Gate A(H1): center_inside=True & belief_target_nonzero=False & mask=1 건수.
  1건 이상 → defect 후보. keypoint의 1% 이상 → smoke 수정 실험.
- Gate B(H2): aug_trunc_v2 vs real truncated의 border-distance / bbox-touch 분포 차이.
- Gate C(H3): aug_trunc_v2의 DiffPnP valid rate. <10% → "DiffPnP가 truncation을
  regularize했다"는 해석 금지.
- H5: train decoder(D0/D1) vs eval decoder(D2/D3) 좌표·missing 불일치.
