# OLD STAGE-A LINEAGE AUDIT

```
stage_a train      73916
  generic exposure 38002   (unique = 동일, repeat 없음)
  target exposure  35914   = unique 17957 × 2
  target alias     17957  (__rep1)
  T v1 base 8982   T v2 base 8975
```

- T__v1 vs T__v2 : 서로 다른 실제 렌더 (RGB 해시 교집합 0/40 표본). alias 는 __rep1 접미사
- alias 근거     : base vs __rep1  라벨 60/60 동일, RGB sha256 60/60 동일

## OLD_GENERIC 계보
```
source  data/pallet/training_data/paper_release/v2_prod40k_clean_merged
관계    SAME_SOURCE
  OG ⊂ broad40k        38002/38002 (100%)
  V1_CF_MATCHED10K ⊂ OG 9496/10000 (95.0%)
  V2_EARLY10K 와 겹침   0
```