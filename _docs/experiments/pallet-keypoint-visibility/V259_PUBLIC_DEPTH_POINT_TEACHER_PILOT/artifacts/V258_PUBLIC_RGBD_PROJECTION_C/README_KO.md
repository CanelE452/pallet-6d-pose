# V258 public RGB-D projection diagnostic API

이 모듈은 classifier/학습기가 아닙니다. Root가 exact public frame bytes를 따로 승인·검증한 뒤 호출하는 순수 geometry/depth 진단입니다. 현재 실행은 invented-array toy만이며 실제 frame/OBJ 본문/weights/model 접근은 0입니다. 기존 허용 K, format QA와 공식 README 텍스트만 확인했습니다.

## 고정 API

```python
vertices, faces, parser_report = parse_obj(obj_path, binding=None)
# binding supplied: {path,sha256,size_bytes}; exact bytes checked before parse.
# Or pure bytes/text: parse_obj_text(verified_obj_bytes).

depth_f64, support_bool, render_report = render_depth(vertices, faces, T, K, (H, W))
stats, raw_observed_minus_rendered_f64 = describe_observed_depth(depth_m, depth_f64, sam_mask_bool)
arrays_by_interpretation, report = diagnostic_pair(vertices, faces, T, K, (H, W), depth_m, sam_mask_bool)
```

`parse_obj` returns float64 xyz vertices and int64 triangular vertex indices. OBJ slash/negative indices work. mtllib/usemtl values remain inert; no material/texture/path follow. Nontriangular faces or invalid references are rejected; there is no fan triangulation, welding, scale/axis correction. Root must supply the exact authorized absolute asset path, preferably with its binding. No automatic directory scan or network exists.

K must be finite3×3, positive focal lengths/nonsingular, last row[0,0,1]. T must be finite4×4 with homogeneous last row and proper rigid rotation (max orthogonality/determinant error≤1e-5, validation only; no matrix correction). Mesh coordinates are used at scale1, translation in meters as the Root rule. Depth PNG uint16/mm must be converted by the separately authorized decoder to floating meter depth. A raw uint16 depth input is rejected. SAM mask must already be bool; `true` is proposal support, never a native/point teacher.

Projection is OpenCV+Z pinhole. Integer `(column,row)` is a pixel center; no extra0.5 shift. Each triangle is rasterized two-sided at integer centers. Screen barycentric weights interpolate reciprocal camera depth `z=1/sum(lambda_i/z_i)`. The z-buffer stores minimum positive cameraZ, inclusive edges; exact depth ties retain the first face. No lens distortion/registration correction or surface smoothing is applied. Fully outside/behind/degenerate triangles are counted. If any triangle crosses the fixed numerical near guard1e-9m, the entire interpretation returns `(None,None,report)` with `UNSUPPORTED_NEAR_PLANE_CROSSING`; clipping is required under a separate explicit contract, not guessed here. Background depth is NaN, support bool.

`describe_observed_depth` returns a float64 H×W observed-minus-rendered map, NaN where either depth is invalid. Stats contain raw count/min/max/mean/std/linear quantiles and exact sign counts overall/within/outside SAM, plus SAM×render support2×2, IoU/Dice. Nonfinite/nonpositive observed depths are invalid, not E/SELF negatives. No D/E/occlusion threshold is applied; output signs and overlaps are descriptive and do not certify point visibility.

`diagnostic_pair` computes exactly T and inverse(T), keeps both reports and arrays, and returns no winner. The README pose section says camera-to-object while its3D Model section explicitly says ob_in_cam/T@p_obj; this module does not resolve that textual conflict through automatic selection. Neither hypothesis is a registration/native-gold certification.

## Root execution boundary

The pure APIs accept already-bound in-memory arrays. Root may SHA-check its one exact PNG bundle, decode via PIL under its separate true entry, convert depth mm→m and mask255→bool, and pass those arrays. This module contains no PNG codec/model imports. `run_bound_entry(entry_path)` is an optional five-input exact-binding reader for OBJ/K/T + already-admitted meter-depth NPY/bool-mask NPY; `ENTRY_FALSE_TEMPLATE.json` cannot execute. It returns arrays/report in memory and does not create output files or discover inputs.

Save both full render/depth-residual arrays, both interpretation reports and the exact decoder/input provenance. Any full-frame overlays belong to Root's actual driver. No native labeling, mask-negative E/SELF conversion, best-pose search, optimization/PnP, threshold fitting or training is authorized by these prepared files.

## Analytic toy proof

```sh
/home/minjae/anaconda3/bin/python -B test_geometry.py
```

`TOY_PROOF.json` was created once:10 groups PASS, terminal4d8e0e exit0, NumPy2.3.5. It covers known triangle centers/depth, reciprocal depth16/11 versus wrong affine2, min-z occlusion independent of face order, out-of-frame/partial coverage, T/inverse(T), crossing-near unsupported handling, finite K/SE3 rejection, continuous depth/support partitions and deterministic repeat bytes. One deliberately invalid uint16 rejection test casts a synthetic NaN and emitted a RuntimeWarning; assertions passed, no real payload was involved. Existing proof is write-once; use a fresh output context to repeat the toy.
