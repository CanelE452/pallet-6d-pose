# The path the forklift actually runs

Three entrypoints exist and all three converge on one decoder, so the
deployment path is not ambiguous and nothing had to be chosen by preference:

```
scripts/dope/run_dope_live.py:29                from detector import ModelData, ObjectDetector
challenge/scripts/run_live.py:38                from detector import ModelData, ObjectDetector
challenge/.../depth_cam/calib/dope_inference.py:36  from detector import ModelData, ObjectDetector
```

All three resolve `detector` to `Deep_Object_Pose/common/detector.py` (each
inserts `Deep_Object_Pose/common` on `sys.path`), and all three call
`ObjectDetector.find_object_poses`.  The forklift FSM
(`dope_inference.py`) carries an identical `_DopeCfg` to
`challenge/config/task.yaml`, so there is also one configuration.

## Call graph

```
image
  -> run_live.py:404-409   proc_scale = 400/h, cv2.resize(w*scale & ~7, 400),
                           K scaled by proc_scale        [aspect preserving]
  -> run_live_io.py:94     run_forward -> net(t) -> (belief stages, affinity stages)
                           returns out[-1][0], seg[-1][0]      = H6 (9,50,50), A6 (16,50,50)
  -> detector.py:544       find_object_poses -> find_objects
       detector.py:684     gaussian_filter(belief, sigma=config.sigma)      sigma = 3
       detector.py:686-703 4-neighbour NMS, peaks where  map > config.thresh_map   0.30
       detector.py:708-730 win = 11, weighted average over the RAW map
       detector.py:733     + 0.4395
       detector.py:805-821 objects built from all_peaks[-1] (centroid channel 8)
                           where the raw score > config.thresh_points        0.30
       detector.py:842-930 每 corner candidate matched to a centroid through the
                           affinity field: aff[2i], aff[2i+1] at the integer peak,
                           x10, normalised, compared with the unit vector to the
                           centroid; accepted when dist_angle < thresh_angle  0.50
       detector.py:558     points = 8 corners + centroid x scale_factor      9 points
       detector.py:561-563 valid_count < 4 -> skip
  -> cuboid_pnp_solver.py:89   cv2.solvePnP(..., flags=SOLVEPNP_EPNP)
       cuboid_pnp_solver.py:110-121  negative-z flip guard
  -> run_live.py:441       enforce_camera_facing (index swap only, R untouched)
  -> run_live_gates.py:48  evaluate_result: min_detected_keypoints 7,
                           z in [0.3, 5.0] m (reads location as centimetres),
                           reproj <= 8.0 px, cuboid edge ratio 0.3, depth agreement
  -> run_live.py:452       first hypothesis that clears every gate wins
```

## Two conventions this path carries

**The 3D model is the yaw-180 partner of the training convention.**
`Cuboid3d` (`cuboid.py:12-23, 94-104`) orders its corners
FrontTopRight, FrontTopLeft, ... while the network was trained against
`annotate_pnp.make_pallet_keypoints_3d` (camera-facing 0123).  Numerically
`M_cuboid = M_camfacing @ Ry^T` with `Ry = diag(-1, 1, -1)` to 2.8e-17, and the
index map is `(5, 4, 7, 6, 1, 0, 3, 2, 8)` -- exactly the `swap_map` that
`run_live.enforce_camera_facing:78` applies.  run_live swaps only the displayed
indices and leaves R alone; its own comment marks the R mapping as a TODO.
This audit converts P2's rotation by that constant so all three paths report in
one frame.

**The live loop's preprocessing does not match this checkpoint.**  run_live
resizes aspect-preserving to height 400; every evaluation in this programme,
and this checkpoint's training, squashes to 400x400.  Holding the input fixed
and swapping only the decoder is the point of the audit, so P2 receives the
squash tensors with a squash-space K (exact: scaling by (400/W, 400/H) scales
fx,cx and fy,cy by the same factors, leaving the pose unchanged).

## Provenance

```
HEAD                88d25c55be0a9ef9275781177b7eb248ba96f648
ep57 SHA256         c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896
S1 checkpoint       weights/paper_s2_stagewise_bias_screen/epoch_005.pth   (run_state 5/5)
C1 checkpoint       weights/paper_s2_corner_replacement_screen/epoch_005.pth (run_state 5/5)
N2 checkpoint       weights/paper_s2_pfdr/N2/epoch_003.pth                  (run_state 3/3)
N3 checkpoint       weights/paper_s2_pfdr/N3/epoch_003.pth                  (run_state 3/3)
deployment config   challenge/config/task.yaml inference.belief
                    threshold 0.30  thresh_map 0.30  thresh_points 0.30
                    thresh_angle 0.50  sigma 3
deployment gates    min_detected_keypoints 7  max_reproj_error_px 8.0
                    cuboid_edge_ratio_tol 0.3  z 0.3-5.0 m
```
