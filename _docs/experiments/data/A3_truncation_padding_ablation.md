# A3. Truncation Padding Ablation  (논문 Table)

> 상태: **미시작** (dope_cropaug에서 일부 검증됨 — 논문용 재정리) | 의존: paper_base(padding) vs nopadding
> 구분: **새로**

## 목적 (한 줄)
DOPE padding(잘린 코너의 belief를 padding 영역에 supervise)이 truncation 이미지의 keypoint/PnP 강건성을 높이는가.

## 판단 지표
crop level(0/1/2)별 **검출률 · PnP 성공률 · reproj(전체 9kp)**. padding 유 vs 무.

## 설정
- 모델: `paper_base`(padding O) vs padding 미적용 동일 모델
- 데이터: GT known 프레임을 crop해 truncation 합성 (L/R 측면 위주, `eval_ab_crop` 방식 재활용)
- 참고: 기존 YOLO/DOPE 검증 — padding이 심한 truncation에서 PnP 76% vs 45%, DOPE PnP 23→99% (memory)

## 방법
1. padding 유/무 모델 학습
2. crop level별 GT-보정 평가 (화면 밖 코너도 offset 보정)
3. truncation 강도↑에서 격차 확인

## 결과 (TBD)
```
lvl   pad?   det%   PnP%   reproj_9kp_med
──────────────────────────────────────────
0     no     TBD    TBD    TBD
0     yes    TBD    TBD    TBD
...
```

## 결론 (TBD)

## 산출물 (예정)
- `challenge/scripts/eval_ab_crop.py` 계열 재활용, level별 곡선
