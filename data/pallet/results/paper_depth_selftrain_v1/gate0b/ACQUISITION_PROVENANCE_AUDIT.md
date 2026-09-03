# Acquisition provenance audit

The question is what produced the stored depth, not what reads it.

## What was searched

```text
paths   scripts/, challenge/, _docs/history/, data/pallet/raw_data/, /home/minjae/Documents/github/25y_automatic_lifter-master, /home/minjae/Documents/github/26y_automatic_lifter-master.zip
tokens  cam_K write, imwrite depth, rs.align, aligned_depth_frame, get_depth_scale, depth_units, rosbag, aligned_depth_to_color, literal session names
```

**No writer was found.** Nothing in this repository or in the lifter repository writes
the `rgb/<nanoseconds>.png` + `depth/<nanoseconds>.png` + `cam_K.txt` layout.
`save_rgbd.py` writes colour JPEG only, `logger.py` writes calibration logs into
raw/meta/model, and `main_rec.py` records forklift runs. The recorder that produced
these eight sequences is not present in anything available here. The timestamps are
nanosecond integers but no ROS artefact exists either.

## What was found


**_docs/history/2026-05-14.md** line 63  
> data/outside/capture* 형식 (cam_K.txt + rgb/{ts}.png + depth/{ts}.png, depth=uint16 mm)

- establishes: a project record stating the stored depth is uint16 millimetres for this exact layout
- does not establish: it describes the format a reader was written against, not a writer's contract

**_docs/history/2026-05-14.md** line 57  
> task.yaml의 camera intrinsic이 header.txt geo_fx/fy/cx/cy와 일치 (614.18/614.31/329.28/234.53)

- establishes: the day cam_K equals the DOPE baseline RGB header intrinsic, so it is the colour-stream K for the RGB pipeline
- does not establish: that the same file is the K the depth was aligned into at capture time

**_docs/history/2026-05-14.md** line data/outside 데이터 메모  
> 12개 시퀀스, 약 9,900 frames, intrinsic은 baseline header와 동일 (RealSense D435i)

- establishes: device family is RealSense D435i and the intrinsic matches the RGB baseline

**challenge/scripts/live/run_live_io.py**  
> numpy uint16(mm) depth; get_distance returns d/1000.0

- establishes: consumer contract for the scale

**/home/minjae/.../25y_automatic_lifter-master/depth_cam/logger.py** line 148  
> align = rs.align(rs.stream.color) ... aligned.get_depth_frame()

- establishes: the project's own capture-side code, when it captures depth, aligns to colour first
- does not establish: that this particular file produced these sequences; logger.py writes raw/meta/model, not rgb/ and depth/

## Grade per recording

```text
recording               writer                             scale            K role                 alignment             strength
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
capturepallet01      not found  DEPTH_SCALE_STRONG_REPO_CONTRACT    LIKELY_COLOR_K  ASSUMED_ALIGNED_UNPROVEN    C_COMPATIBLE_ONLY
capturepallet10      not found  DEPTH_SCALE_STRONG_REPO_CONTRACT    LIKELY_COLOR_K  ASSUMED_ALIGNED_UNPROVEN    C_COMPATIBLE_ONLY
capturepallet11      not found  DEPTH_SCALE_STRONG_REPO_CONTRACT    LIKELY_COLOR_K  ASSUMED_ALIGNED_UNPROVEN    C_COMPATIBLE_ONLY
capturenight01       not found  DEPTH_SCALE_STRONG_REPO_CONTRACT          UNPROVEN  ASSUMED_ALIGNED_UNPROVEN    C_COMPATIBLE_ONLY
capturenight02       not found  DEPTH_SCALE_STRONG_REPO_CONTRACT          UNPROVEN  ASSUMED_ALIGNED_UNPROVEN    C_COMPATIBLE_ONLY
capturenight03       not found  DEPTH_SCALE_STRONG_REPO_CONTRACT          UNPROVEN  ASSUMED_ALIGNED_UNPROVEN    C_COMPATIBLE_ONLY
capturenight04       not found  DEPTH_SCALE_STRONG_REPO_CONTRACT          UNPROVEN  ASSUMED_ALIGNED_UNPROVEN    C_COMPATIBLE_ONLY
capturenight10       not found  DEPTH_SCALE_STRONG_REPO_CONTRACT          UNPROVEN  ASSUMED_ALIGNED_UNPROVEN    C_COMPATIBLE_ONLY
```

Every recording lands on `C_COMPATIBLE_ONLY`. The reader treats the format as aligned
millimetre depth and the project's own capture-side code aligns to colour whenever it
touches depth, but no record ties these particular files to how they were made.

The scale is graded higher than the alignment because a project history entry states
the unit outright, which is more than a consumer convention.

