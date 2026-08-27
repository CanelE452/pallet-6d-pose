# YOLO26m SERVER RUN — 체크리스트

> ⚠ **지금 실행하지 않는다.** 아래 4 조건이 전부 만족될 때만.

```
1. yolo26n seed42 STRONG_PASS
2. seed43 / seed44 까지 같은 방향
3. n 에서 capacity bottleneck 이라는 명확한 근거
4. server log / args / checkpoint provenance 를 전부 회수 가능
```

## 비교 금지

로컬 `stage_a_m_640_b8_seed42` 는 **batch 8 · 12 epoch 미완주**다.
서버 batch32 · 60 epoch 판과 **비교하지 않는다**.

## preflight

```
dataset manifest SHA 가 로컬과 일치하는지
VRAM >= (batch32 가 들어갈 만큼). OOM 후 자동 batch 축소 금지
expected_args.json 과 실제 args.yaml diff 0 (seed/name/path 제외)
업로드·다운로드 checksum
```

## 회수할 것

```
args.yaml / results.csv / weights(last.pt) / 학습 로그 / nvidia-smi 기록
```

## 판정

n 과 m 의 차이가 작으면 **n 을 final 로 선택**한다 — 더 작고, 더 빠르고,
Jetson 배포에 유리하고, 재현성이 높다.
