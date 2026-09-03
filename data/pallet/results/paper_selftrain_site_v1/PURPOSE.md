# PURPOSE — paper_selftrain_site_v1

[소비처]
논문의 "post-hoc site-matched adaptation analysis" 절.  현 단계는 그 절을 쓸지
말지를 정하는 **preflight** 이며, 학습 결과가 아니라 pseudo-label 수급 진단이
소비처다.  full-site 학습을 실제로 할지는 이 진단을 사용자가 보고 결정한다.

[문장]
"SITE_A 전체 2,227장에 기존 R0 teacher 와 동결된 filter 를 그대로 적용하면
A8 의 500장/120 PL 대비 pseudo-label 수·recording 분산·기하 coverage 가
늘고 반복 노출이 줄어든다" — 또는 그 반증.

## 이번 단계 범위 (PREFLIGHT ONLY)

새 student training 0 · 새 teacher 0 · 새 filter 0 · 새 threshold 0 ·
결과를 보고 설정 변경 0.  teacher inference 는 기존 R0 체크포인트
(sha 970a0913...)를 기존 recipe 로 pool 에 한 번 돌리는 것뿐이다.

## 판단 지표 (결과 보기 전 고정)

네 질문에만 답한다.  성능 예측은 하지 않는다.

```
1  usable pseudo-label 이 늘어나는가        (F4 accepted 수)
2  recording 에 고르게 분산되는가            (accepted share by recording)
3  기하 coverage 가 넓어지는가               (bbox scale·위치·corner·score 분포)
4  반복 노출이 줄어드는가                     (1440 / N_unique per epoch)
```

A8 기준값: teacher input 500 · F4 accepted 120 · 반복 12.0회/epoch.

## 금지

- 평가 GT(PAPER_EVAL · SITE_A eval · 6D GT)로 pseudo-label 품질을 재지 않는다.
  이번 질문은 purity 가 아니라 quantity/diversity/exposure 다.
  `GT_USED_FOR_SELECTION = false`
- `paper_selftrain_v1/` 의 기존 artifact 를 수정하지 않는다.
- "N 이상이면 GO" 같은 새 임계값을 CLI 가 만들지 않는다.  보고까지만 하고 멈춘다.

## 누수 계약 (이미 검증됨)

adapt = capturepallet01/10/11 (2,227장), eval = SITE_A_EVAL_ELIGIBLE.csv (88장).
image SHA · source recording · underlying recording 겹침 모두 0.
