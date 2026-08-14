# PURPOSE — PFDR far-decoupled refinement (N1/N2/N3, 3 epoch)

[소비처]
"corner 개선을 pose 개선으로 전환할 수 있는가" 의 최종 판정.
`pfdr/PFDR_GO_STOP_GATE.md` → 최종 architecture 선택(base ep57 / PFDR belief-only /
PFDR + 9-point geometry).  FAIL 이면 base ep57 유지하고 이 계열을 종료한다.

[문장]
"near corner 와 centroid 를 ep57 H6 로 bit-exact 고정한 채 far corner 만 H5 anchor +
residual adapter 로 학습하면, static E2 가 얻은 far 개선(-15.5% / -16.2%)이
canonical eval56 과 unseen wood 두 셋 모두에서 **pose(reprojection·PnP)** 개선으로
전환된다."

## 판단 지표 (사전 고정, 결과 보고 조정 금지)

eval56: PnP >=50/56 / reproj <=10.402px(-10%) / far <=10.266px(-10%) /
        near exact 불변 / >50 <=45 / >100 <=17 / NaN <=119 /
        improved>worsened / P(improve) >=0.90 / 새 PnP failure 0
wood:   PnP >=44/45 / reproj <=8.820px(-5%) / far <=12.762px(-10%) /
        near exact 불변 / >50 <=40 / >100 <=36 / NaN <=51 /
        improved>worsened / P(improve) >=0.80 / 새 PnP failure 0
두 셋 중 하나라도 FAIL 이면 cross-pallet ACCEPT 금지.

arm 비교: N2 가 N1 보다 나아야 pose objective 유효, N2 가 N3(near control) 보다
far 에서 우수해야 far-specific claim 성립.

## 실행 범위

- base ep57 **전체 frozen**(trainable param 0 확인), adapter 136,004 param 만 학습.
- N1/N2 는 near·centroid 를 건드리지 않고, N3 는 far·centroid 를 건드리지 않는다
  (zero-init 시 bitwise 0.0 검증 완료).
- canonical Stage-B loader 29,308 samples, 정확히 3 epoch, checkpoint selection 없음.
- eval56/wood 는 학습에 사용하지 않는다.  final-test 미사용.
- canonical decoder·centroid 포함 canonical PnP 변경 없음.

## 한계 / 사전 기록한 리스크

- **anchor λ 가 상한 10 에 clamp됐다.**  zero-init 에서 Huber(0,0) 의 gradient 가
  정확히 0 이라 gradient-norm 측정이 성립하지 않는 지점이다.  규정상 결과를 보고
  조정하지 않지만, λ=10 이 residual 을 0 근처로 묶어 학습을 막을 수 있다.
- N2/N3 의 GT-pose seed 는 training 전용이며 inference 경로에 없다.
- eval56 F2 8 frame / wood 5 frame 으로 F2 지표는 단독 근거로 쓰지 않는다.
