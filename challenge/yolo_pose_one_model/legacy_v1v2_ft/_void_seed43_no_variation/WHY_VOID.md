# 이 결과는 무효다 — seed 가 실제로 바뀌지 않았다

`seed=43` 으로 돌렸지만 seed 42 와 **비트 단위로 동일한 가중치**가 나왔다.

```
검증  torch.load 후 state_dict 879개 텐서 전수 비교
      different = 0 / 879 · max_abs_diff = 0.000e+00
      train_args.seed 는 42 / 43 으로 다르게 기록돼 있다(기록만 다름)
```

원인 (ultralytics 8.4.60 `data/build.py:346`)

```python
generator = torch.Generator()
generator.manual_seed(6148914691236517205 + RANK)   # args.seed 가 안 들어간다
```

이 FT 는 고정 가중치(G38 last.pt)에서 시작하므로 남은 무작위성은 데이터 순서와
worker 증강 RNG 뿐이고, 둘 다 위 고정 generator 에서 파생된다. 따라서 args.seed 를
바꿔도 학습이 완전히 동일하게 재현된다.

→ 이 산출물은 "seed 재현 확인" 이 **아니라** 같은 실행을 두 번 한 것이다.
   replicate 로 인용하지 말 것. 진짜 replicate 는 generator seed 에 args.seed 를
   더하는 패치를 넣고 다시 돌린다(lv_driver.py 참조).

★ 주의: 이 무효화는 이 FT 실행에 대한 확인이다. 프로젝트의 다른 seed 쌍이 전부
  무효라는 뜻은 아니다 — CF_DATA_C_V2_EARLY10K_STD_60EP_SEED42/43 은 실제로 달랐다
  (different 749/879). 왜 그쪽은 달랐는지는 미확인 [추정: 60ep 에서 비결정 CUDA
  커널이 누적, `torch.use_deterministic_algorithms(warn_only=True)` 라 허용됨].
