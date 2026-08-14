# A2. Squash 비율 강건성 Ablation  (논문 Table)

> 상태: **미시작** | convention: camera-facing 0123 | 의존: paper_base, paper_base_nosquash 학습
> 구분: **새로** (논문 핵심 — 처음 본 비율 일반화)

## 목적 (한 줄)
squash(찌부/늘림) 증강이 **학습에 없던 aspect ratio** 파렛트의 keypoint 추정 일반화를 실제로 높이는가.

## 판단 지표
비율이 다양한 테스트셋에서 **keypoint reproj(전체 9kp 평균) + 검출률**. squash 유 vs 무 비교.
(부차적 매력 주의: clean 정밀도만 보지 말고 unseen 비율에서의 일반화가 본질)

## 설정
- 모델: `paper_base`(squash O) vs `paper_base_nosquash`(squash X, 나머지 동일)
- 데이터: 학습=mixed_v8 camera-facing (±squash) / 평가=비율 다양한 파렛트 (TBD: 합성 비율 grid + real)
- convention: camera-facing 0123, 9kp

## 방법
1. squash 유/무 두 모델 동일 조건 학습 (squash만 차이)
2. aspect ratio 구간별(예: 0.6/0.8/1.0/1.3/1.6) 테스트셋에서 9kp reproj·검출
3. 학습 분포 밖 비율에서 격차 확인

## 결과 (2026-06-08, squash vs no-squash 실모델 비교)

paper_base(squash+scale) vs dope_cropaug_pretrain(squash 없음). 둘 다 mixed_v8 camfacing + trunc, VGG/sigma4.0/60ep 동일조건. order-free 9kp, GT셋.
```
model                domain   det%   9kp_med   good%   front  back   ctr
─────────────────────────────────────────────────────────────────────────
paper_base(squash)   indoor   26.6   13.0      3.4     15.4   9.6    12.3
pretrain(no-squash)  indoor   18.6   34.6 ↓↓   2.4     66.0   7.0    24.8   폭망
paper_base(squash)   outside  45.3   18.9      12.1    9.4    19.2   13.1
pretrain(no-squash)  outside  39.8   17.9      17.6    10.0   18.8   15.4   동급
paper_base(squash)   night    30.0   21.9      0.0     11.9   34.0   17.6
pretrain(no-squash)  night    46.7   19.5      2.4     10.3   21.6   12.4   동급~약간↑
```

## 결론 (가설 반증 — squash는 indoor에 해롭지 않고 오히려 도움)
- **squash 없는 모델이 indoor에서 9kp 13→35px 폭망**(front corner 66px, cuboid가 pallet 안쪽 collapse). 다양한 종횡비 미학습 → mixed_v8 정면 비율에 과적합 → real indoor 비율차에서 무너짐.
- squash 효과는 **indoor 특이적**(outside/night은 동급). 정면 top-down flat depth에서 squash가 extent 추정에 강함.
- → **squash 제거/완화는 답 아님(회귀).** 유지/강화가 데이터 방향. indoor 약점은 squash가 아니라 **back(4-7)·centroid·detection(26%)**.

## 산출물
- 스크립트: `scripts/data_prep/eval/squash_vs_nosquash.py`, `squash_indoor_overlay.py`
- 결과: `scripts/data_prep/eval/squash_vs_nosquash_results.json`
- indoor 비교 overlay: `data/pallet/eval_results/squash_vs_nosquash/indoor_overlay/_contact_sheet.jpg` (좌=squash, 우=no-squash)
