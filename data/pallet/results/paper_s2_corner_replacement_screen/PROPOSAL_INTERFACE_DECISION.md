# PROPOSAL INTERFACE DECISION

> 재학습 0 step.  epoch-5 checkpoint 고정.  이전 sigmoid+threshold 평가가 학습
> objective 와 불일치했음을 확인했고, 이전 replacement gate 는 g~1e-9 라 실제로
> 시험되지 않았다.  raw Q 로 다시 평가했다.  N87 은 mechanism screen, final-test 미사용.

## [현재 판정]

```
Shared base fine-tuning          REJECT   (C0 13.24 -> C1base 13.95px, PnP 70 -> 68)
Raw proposal branch              REJECT   (raw-Q 로도 median 80.1px = base 의 6배)
Proposal decoder                 LOCAL    (셋 중 최선이나 절대성능 미달; ARGMAX 와 동률)
Coordinate router                NOT RUN  (oracle 상한이 gate 미달)
Map-level replacement gate       REJECT   (g~1e-9, 시험되지 않음)
Final path                       base DOPE
```

## [지지 증거]

- [확인] interface 가설을 실제로 검증했다: raw-Q 세 decoder 전부 실행.
  far median 이 160.3(sigmoid) → 117.9(raw local) 로 나아졌으므로 interface 는
  **일부** 원인이었다.  단 base 22.1px 에 비하면 여전히 5배다.
- [확인] confident-wrong 부분집합에서 proposal 이 34.3% 이기고 22.7% 가 10px 이상 우세 =
  상보성 신호는 실재한다.
- [확인] margin oracle 이 tail -20.8%, reproj -17.6%, near 개선을 냈다.

## [반증 증거]

- [확인] P-ARGMAX(80.07) ≈ P-LOCAL(80.15) → 정제 문제가 아니라 top-1 위치가 틀렸다.
- [확인] P-DSNT near 109.9px → Q 가 뾰족하지 않다(학습 objective 자신의 read-out 이 최악).
- [확인] oracle 조차 PnP 를 70 → 70 으로 못 올린다.  좌표 선택으로는 pose 실패가 안 풀린다.
- [확인] 표본: corner 519(F2 268, confident-wrong 181), frame 87.  소표본 screen 이다.
- [확인] harness 는 지시문 규칙에 따라 centroid 를 PnP 에서 제외했으므로 C0 절대 reproj
  (24.91px)가 canonical(23.162px)과 다르다.  arm 간 비교는 공정하다.

## [다음 admissible experiment]

1. proposal branch 를 base 와 **분리해 재학습** — 이번 branch 는 λ_prop 3.67e-05 로
   사실상 가중되지 않은 채 학습됐다(raw loss 는 6.05→1.02 로 내려갔지만).
   같은 objective 를 제대로 가중해 단독 학습한 뒤 다시 이 harness 로 재평가.
2. confident-wrong 상보성(22.7%)이 유일한 양성 신호이므로, 이를 겨냥한 실험만 남긴다.
   전체 평균을 겨냥하는 설계는 이 데이터에서 근거가 없다.
3. PnP 구조는 좌표 선택이 아니라 다른 층위의 문제다.  좌표 router 로 재시도하지 않는다.
