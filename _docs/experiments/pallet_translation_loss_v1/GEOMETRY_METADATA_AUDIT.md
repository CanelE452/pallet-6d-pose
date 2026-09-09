# Stage 0B — geometry 항을 계산할 수 있는가

translation-aware 항에 필요한 네 가지 — 3D 치수 / 카메라 K / GT pose (R,t) /
8 cuboid corner 대응 — 를 55,980 train + 4,020 val 전수의 renderer label 에서
직접 세었다. 표본이 아니라 전수다.

산출: `data/pallet/results/pallet_translation_loss_v1/GEOMETRY_METADATA_AUDIT.json`

```
                          n      K    dims   pose  corners  perm_v4  all
ALL                  60,000  100%    100%   100%    100%    100%   100%
train                55,980  100%    100%   100%    100%    100%   100%
val                   4,020  100%    100%   100%    100%    100%   100%
G38:train            38,002  100%    100%   100%    100%    100%   100%
P0:train              8,989  100%    100%   100%    100%    100%   100%
TEX:train             8,989  100%    100%   100%    100%    100%   100%
asset 4 종 각각              100%    100%   100%    100%    100%   100%
```

읽지 못한/파싱 실패 label 파일 0 건.

사전등록 게이트 — 전체 >= 98% 이고 어떤 주요 subgroup 도 >= 90% —
**PASS (전부 100%)**.

## 그래도 남는 배선 문제 — YOLO label 은 이 정보를 담지 않는다 [확인]

ultralytics 가 읽는 것은 `datasets/.../labels/{split}/*.txt` 뿐이고,
거기에는 정규화된 box + 9 keypoint 만 있다. K·치수·pose 는 **renderer label
JSON 에만** 있고 dataloader 경로에 존재하지 않는다.

따라서 Stage A 는 `merged_stem -> {K, dims, R, t}` 부수 테이블을 만들어
loss 안에서 조회하는 배선이 필요하다. `PROBE_METADATA_60K.jsonl` 이 이미
`merged_stem` 을 키로 갖고 있으므로 새 데이터 생성은 필요 없다.
단 mosaic 의 `im_file` 함정(memory `yolo-pose-aux-loss-wiring-traps`)이
그대로 적용된다.
