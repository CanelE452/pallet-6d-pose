# PURPOSE — ralph_meas_v3 (self-training 측정-무결성 하네스)

[소비처] 논문 Method 비교표(Synthetic only / Naïve ST / Reproj+flip ST / Ours × yaw·ADD·det rate)
와 그 표의 신뢰성 근거. 현재 진행 중인 질문 = "2D 키포인트는 좋아지는데 yaw 는 왜 나빠지는가".

[문장] "loo+flip self-training 은 검출률과 2D/3D 위치를 개선하지만 yaw 는 개선하지 못한다 —
그리고 그 yaw 악화는 unpaired-N(모델별 검출집합 상이) 아티팩트가 아니라(혹은 아티팩트다)."
→ `yaw_paired_diagnosis.py` 가 이 문장의 괄호를 확정한다.

## 구성
- 측정 전용(처방·수정 없음). GT = manual GT (2D 클릭 → PnP pseudo-GT), floor nonzero.
- 하위 스크립트: measA(2D raw argmax) / measB(GT pose 불확실성 floor) / measC_yaw(yaw) /
  threeway·fourway_pose_metrics(모델×지표 표) / yaw_paired_diagnosis(페어링 진단).
- 판단 지표: self-domain median yaw(°)·centroid(cm)·ADD(m)·det rate(%), THRESH 0.3, N_DET_MIN 6,
  GT reproj>5px 프레임 제외.

## caveat
pseudo-GT depth 약제약(ADD floor outside 0.027 / night 0.028 / noapril 0.078 m),
noapril N≤18 소표본, 모델별 검출셋이 달라 unpaired(그래서 paired 진단이 필요).
