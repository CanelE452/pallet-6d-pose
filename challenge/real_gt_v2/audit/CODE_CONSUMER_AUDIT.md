# Legacy real-GT and consumer audit

Status: **READ-ONLY AUDIT COMPLETE**
Scope: reviewed positive DEV 140 and the requested annotation/evaluation consumers
Source manifest: `/home/minjae/pallet_worker_transfer_20260821T105141Z/REAL_GT_QA_20260821T133405Z/REVIEWED_CLEAN_REALDEV_V2_MANIFEST.json`
Manifest SHA-256: `813ea97b6532591b4b5b0d1f688819c67bc173f59eceaaaf71048d80e698a490`
Declared role: `DEVELOPMENT — final test 아님`

This audit was completed before implementation changes. No source GT JSON was rewritten. The per-file SHA-256, byte size, and nanosecond mtime baseline needed to prove that claim after migration is in `LEGACY_GT_PER_FRAME.csv`.

## Data findings

- `DEV_POS140` contains 140 unique frames across `eval_cad=18`, `eval_noapril=12`, `eval_outside=22`, `eval_pallet07=27`, `eval_pallet09=33`, `eval_night08=12`, and `eval_night09=16`.
- `COMMON_DEV_POS128` is exactly `DEV_POS140` minus the 12 IDs in `FT_EVAL_LEAK.json`. All 12 removed frames are daytime `eval_outside` frames.
- `DEV_NEG2689` is `data/pallet/raw_data/negative_real_20260823/rgb`, with contiguous names `000000.png` through `002688.png`.
- There is no frozen untouched FINAL membership. `FINAL_POS=0` and `FINAL_NEG=0` mean membership is unavailable, not that no real data exists.
- Legacy camera-facing dimensions are `(width=1.10,height=.11,depth=1.30)` for 81 frames and `(width=1.30,height=.11,depth=1.10)` for 59 frames.
- `fix_swap` and `extrapolated_mask` are absent in all 140 labels. The 1,256 non-null `manual_kps` locations therefore do not carry recoverable manual-click versus auto-fill provenance. Four locations are null, all in frame `1778653804674198784` at indices 4, 6, 7, and 8.
- The projected cuboid has 39 out-of-frame corners across 19 frames. Object-level `visibility=1` occurs in all 140 labels, but it is not a per-keypoint visibility annotation.

## Geometry contract findings

`scripts/annotate/annotate_pnp.py` uses an untyped `(width, depth, height)` tuple. Its local camera-facing frame is centroid-origin, `+X=right`, `+Y=down`, `+Z=forward`; indices 0--3 are on the near face. `scripts/annotate/annotate_io.py` serializes the same tuple as JSON names `(width, height, depth)`. This is the source of two superficially conflicting dimension orders.

The stored `pose_transform` maps the per-frame camera-facing point frame to the camera. It is not a pose in a single physical pallet frame. Legacy W/D identifies yaw parity only:

- `(W,D)=(1.10,1.30)` permits physical yaw candidates 0 and 180 degrees.
- `(W,D)=(1.30,1.10)` permits physical yaw candidates 90 and 270 degrees.

Reprojection, cheirality, and the current camera-facing point identities cannot distinguish the two signs in either pair. A physical direction marker, manual review, or a frozen symmetry contract is required. Migration must not silently pick one.

The LR and FB permutations in `fix_manual_swap.py` correspond to reflections with determinant -1. Their combination is a proper 180-degree yaw, but neither reflection may be used as a canonical rotation. `_repnp_with_new_dims.py` instead inserts fixed dimensions at the same point indices and overwrites the solved pose; it is not a physical-frame conversion.

## Annotation consumers

| Consumer | Audited behavior | v2 implication |
|---|---|---|
| `annotate.py` | Uses legacy fields and has no per-keypoint visibility/reason state or signed physical-axis confirmation. | Add explicit v2 state and keep old JSON readable. |
| `annotate_pnp.py` | Solves as-given and W/D-swapped camera-facing hypotheses and can overwrite effective dimensions per frame. | Separate fixed physical XYZ dimensions from named camera-facing WHD hypotheses; record scores and assignment. |
| `annotate_io.py` | Atomically writes legacy pose, projected points, dimensions, and `manual_kps`. | Preserve those fields byte-for-byte in migrated copies while adding v2 fields. |
| `annotate_draw.py` | Does not distinguish visible, occluded, truncated, and unknown point states. | Add distinct markers and a legend. |
| `fix_manual_swap.py` | Searches identity/LR/FB/LR+FB and two W/D choices by reprojection, then mutates labels. | Deprecate for paper GT and require explicit legacy-mutation opt-in. |
| `_repnp_with_new_dims.py` | Re-solves fixed-size points at unchanged indices and directly overwrites JSON. | Deprecate for paper GT and route users to v2 migration. |
| `verify_kp_contract.py` | Loads per-frame GT dimensions; its near/far swap is a reflection, so the check does not establish a fixed physical pose frame. | Do not use it as the canonical migration proof. |

## Evaluation consumers and population mismatch

- Historical `cf_real_eval.py` reads all 140 reviewed positives and computes 2D keypoint diagnostics. Its pose path is explicitly blocked.
- Historical `neg_eval_one.py` removes the 12 leaked positives, so it uses 128 positives plus 2,689 negatives.
- Both scripts hard-code environment-specific paths and silently skip unavailable entries. They remain historical reproduction artifacts and must not be edited.
- A new paper comparison must use the same explicit `COMMON_DEV_POS128` membership for every method. `DEV_POS140` is reserved for geometry/selector diagnostics. Paper-final results require separately frozen `FINAL_POS` and `FINAL_NEG` manifests.
- Existing ALL161 pose numbers depended on per-frame GT dimensions and are not transferable to the v2 paper table.

## Local Ultralytics visibility semantics

The audited environment is Ultralytics 8.4.60 in `pallet-yolo26`. For a three-channel keypoint target, the local pose loss and metrics use `visibility != 0` as the mask. Values 1 and 2 are both supervised valid points; 0 is ignored. The v2 converter will therefore map `0 -> (0,0,0)`, `1 -> (x,y,1)`, and `2 -> (x,y,2)`, while documenting that 1 and 2 have identical loss masking locally.

The existing repository YOLO-label auditor accepts only 0/2. It is not the GT-v2 schema validator and must not be reused to reject valid v2 visibility 1.

## Paper pose gate

ADD(-S) AUC, rotation median, translation median, and yaw median remain null until all four independent contracts pass:

1. canonical migration;
2. GT-independent W/D selector;
3. frozen physical symmetry specification;
4. frozen untouched FINAL manifests.

At audit completion, none of these four gates is fully satisfied. Emitting zero, NaN, or legacy pose values would misrepresent blocked metrics as measurements.
