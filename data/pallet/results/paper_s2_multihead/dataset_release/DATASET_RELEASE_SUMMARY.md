# DATASET RELEASE SUMMARY

물리 concat 없음. manifest 만 만들었고 원본 pool 은 그대로다.

```
PAPER_CORE_V1_corner_manifest.json          33,758
PAPER_CORE_V1_line_manifest.json            33,758
DEPLOYMENT_CANDIDATE_V1_corner_manifest.json 33,758
DEPLOYMENT_CANDIDATE_V1_line_manifest.json  43,758
SAMPLING_POLICY.json
checksums.sha256
```

## 학습에서 제외된 것

```
BROAD_40K MH_DEV                      6242
EDGE_HARD_TRUNC_DEV                   1000
EDGE_HARD_TRUNC_UNTOUCHED             1000
EDGE_HARD_CLEAN_UNTOUCHED             1000
CORNER_LA_Y15_30                      2500
CORNER_LA_Y30_PLUS                    2500
NEGATIVE_SYNTH_V1                    10000
```

## 누수

```
HARD_BLOCK        0
LEAKAGE_CLEAN     True
```

MH_DEV 6,242 는 어떤 training manifest 에도 들어가지 않는다 (manifest 생성 시
`mh_split == "MH_TRAIN"` 으로 필터). EDGE dev/untouched, NEGATIVE dev 도 동일.
real test 는 이 감사에서 아예 건드리지 않았다.
