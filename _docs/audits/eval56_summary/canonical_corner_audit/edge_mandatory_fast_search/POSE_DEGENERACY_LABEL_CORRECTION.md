# The groups measure the image, not the pose

`d799101` stands unedited.  Its numbers are right and two of its names are not.

```
historical label        scientific interpretation
G1_pose_coplanar    ->  G1_PROJECTED_FACE_COLLAPSE
G2_pose_collinear   ->  G2_PROJECTED_LINE_LIKE
```

Both quantities are measured in the image: the projected separation of the top
and bottom faces, and the smaller singular value of the projected corners.  A
small value says the *projection* is poorly conditioned.  It does not prove that
PnP has more than one solution -- the 3D corners of a box with non-zero height
are never coplanar, and whether a particular view admits a second pose depends on
noise, on which corners are usable and on the solver, none of which was measured.

```
POSE_ONLY_DEGENERACY     kept as the historical label of d799101
UNIQUE_POSE_AMBIGUITY    UNRESOLVED
```

So `G1` and `G2` are **not** auto-excluded from pose evaluation.  `d799101`
proposed an `AMBIGUOUS_POSE` bucket; that is withdrawn as an automatic action.
It can only be decided by locating the provenance of the original 1,056-frame
audit or by a real pose-ambiguity audit that tests solutions rather than
conditioning.

What `d799101` does establish is unchanged: every frame has twelve well-defined
supporting lines, and degenerate frames carry loss in proportion to their size
(leverage 0.98 to 1.09), so nothing is filtered.
