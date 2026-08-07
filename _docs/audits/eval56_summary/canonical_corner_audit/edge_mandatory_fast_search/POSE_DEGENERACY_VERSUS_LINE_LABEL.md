# POSE_ONLY_DEGENERACY

Degenerate frames are harder for the model, and they carry no unusual training
leverage.  Nothing is deleted and no role is masked.

## The 1,056 figure could not be sourced

Before anything else, the provenance step failed.  The repository has no record
of 1,056 pose-degenerate frames or 1,692 tiny frames:

```
eligibility_audit.csv          20,000 frames, tiny=True 537, all eligible=True
                               no coplanar / collinear column
dataset safety_gates           G1..G5 and all_pass, True for all 20,000 frames
_docs/history                  no occurrence of 1056 / 1,056 / 1692 / 1,692
```

So those counts come from an audit outside this repository, and the instruction's
own warning applies -- they may describe a 40k pool rather than the 13,618-frame
`LINE_TRAIN`.  I did not adopt them.  Instead the degeneracy is derived here from
the data, and the two numbers are reported side by side so they can be compared
rather than conflated.

## The criterion, and why its threshold is not new

Per frame, from the projected cuboid in canonical50 cells:

```
thickness   median projected separation of the top and bottom faces over the
            four corresponding corner pairs (0,3) (1,2) (4,7) (5,6).
            As it goes to zero the eight corners collapse onto four and the PnP
            configuration approaches coplanar.
spread      smaller singular value of the centred eight projected corners,
            normalised.  As it goes to zero the projection approaches a line.
```

Both are compared against **0.75 canonical50 cell**, which is the label's own
tube sigma (1.5 MAP100 pixel) already locked in `cc82012`.  Two projected faces
closer than one sigma sit inside the blur of the supervision itself.  No new
threshold was introduced for this screen.

```
G2_pose_collinear   spread < sigma
G1_pose_coplanar    thickness < sigma, not G2
G0_nondegenerate    otherwise
G3_line_invalid     any role with ||p1 - p0|| < 1e-4 or non-finite theta/rho
```

## Intersections with the current pools

```
pool             frames    G0      G1     G2   G3    thickness p1 / p50
LINE_TRAIN       13,618  10,660  2,500   458    0        0.218 / 1.45
line_search2k     2,000   1,586    349    65    0        0.239 / 1.49
D0_SEEN512          512     405     84    23    0        0.274 / 1.42
```

By this criterion 21.7% of `LINE_TRAIN` is degenerate, far more than the 2.64%
quoted for the other pool.  That is a difference of definition and of population,
not a contradiction, and it is why the external figure was not reused.

## G3 is empty, everywhere

```
LINE_TRAIN     163,416 roles    LINE_COLLAPSED 0    LINE_NONFINITE 0
line_search2k   24,000 roles    0    0
D0_SEEN512       6,144 roles    0    0
valid-role histogram: every frame in every pool has 12 of 12
```

Every frame has twelve well-defined supporting lines.  Pose degeneracy does not
produce line-label degeneracy: `STRUCTURAL_LINE_LABEL_DEGENERACY` does not fire,
and no role masking is warranted.

This is the point the screen was built to test.  A coplanar PnP configuration
still projects eight distinct corners in general position on the image plane, and
the line through any two of them is perfectly well defined even when the pose
that produced them is not unique.

## M0 epoch 5, stratified on D0_SEEN512

Existing checkpoint, one forward pass, no training.

```
group                frames  roles   angle med  angle p90  offset med  loss mean  loss share  leverage
G0_nondegenerate        405   4,658     5.8343     50.659      2.2905    0.07776      77.75%      0.98
G1_pose_coplanar         84     997     9.0189     65.540      4.3063    0.08099      17.33%      1.06
G2_pose_collinear        23     273    12.8970     56.609      5.1943    0.08385       4.91%      1.09
G3_line_invalid           0
```

Degenerate frames really are harder -- G2 is 2.2 times G0 in angle and 2.3 times
in offset.  But **leverage is 0.98, 1.06 and 1.09**: each group contributes map
loss almost exactly in proportion to its size.  The threshold for
`POSE_DEGENERACY_POISONS_LINE_TRAINING` was 3.0.

That matters for the underfit result.  Removing every degenerate frame would
remove about 22% of the loss and about 22% of the data, so it cannot explain a
model sitting at 6.60 degree on frames it trained on.  G0 alone is 5.83 degree --
still 5.8 times outside the budget.

```
CAUSE              POSE_ONLY_DEGENERACY
TRAINING_FILTER    KEEP
```

## Policy

```
supporting-line training    keep every frame, mask no role
unique-6D pose evaluation   G1 and G2 to an AMBIGUOUS_POSE bucket, or excluded
                            and reported separately
detection / keypoint eval   no exclusion
```

A frame can be valid for the 2D task and ineligible for unique 6D pose at the
same time, and that is the distinction to carry forward rather than a deletion.

The 1,692 tiny frames were kept out of this filter by instruction; the repository's
nearest equivalent is `tiny=True` on 537 of 20,000 audited frames, and tininess is
intended difficulty, not degeneracy.

## Next

```
data x optimizer-step 2x2    UNCHANGED
```

No filtering rule is committed, so the pool and therefore the step counts for the
2x2 stay exactly as locked.  Nothing was trained, no frame was deleted, no role
masked.  No PnP, no CIGM, no dimensions.  `untouched`, `eval56`, `wood45` and
final-test remain unopened.
