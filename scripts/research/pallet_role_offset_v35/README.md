# V35 physical boundary / virtual corner correspondence verification

From the repository root, with completed immutable V32 and V34 inputs:

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_role_offset_v35/run35.py
```

The driver resumes missing stages in this existing run. Use a new result directory
and protocol for a new experiment; never overwrite completed historical results.
GPU access is needed only for extraction. This diagnostic performs no neural
optimizer updates. It fits 24 train-only point/role offset calibrations with
NumPy; V34 checkpoints are frozen, three seeds are evaluated.

- `extract35.py`: physical visible support, frozen Hough candidates and point parents.
- `analyze35.py`: registered scalar offset / orthogonal projection comparisons.
- `followup35.py`: explicitly post-primary bounds, constant offsets, multiple lines.
- `quantization35.py`: best-of-all-bins GT physical-line oracle.
- `audit35.py`, `audit_grid35.py`: source/GT isolation, independent projection and grid arithmetic.
- `report35.py`, `deliver35.py`: MD/plots, verified HTML table and actual Chrome window.

Data are existing V32 600 rendered observations, 16 train and 8 check appearance
groups from the same two CAD assets. 436 detected observations, including 130/200
check observations; 820 supported check points form the main paired diagnostic.
No fresh real evaluation or manual GT edits. The renderer check is reused DEV.

GT physical role/visibility matching is an oracle correspondence; constant-offset
calibration is train-only but does not make GT line selection deployable. Exact
GT offsets, virtual lines, best-of4/full-grid and joint observable projections
are diagnostic controls. Fully GT-free confidence selection is reported separately.
The follow-up runs preserve the failed primary SUMMARY and use separate protocols.
Notifications are sent separately after all artifact checks and browser delivery.
