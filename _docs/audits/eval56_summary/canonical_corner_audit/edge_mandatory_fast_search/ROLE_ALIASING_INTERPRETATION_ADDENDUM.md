# Redundant is not contradictory

`f3bab98` stands unedited.  Its measurement is right and one sentence in it is
not.

```
historical            STRUCTURAL_ROLE_TARGET_ALIASING_PRESENT
interpretation        NEAR_COINCIDENT_ROLE_TARGETS_PRESENT
                      STRUCTURAL_ROLE_REDUNDANCY_PRESENT
withdrawn             "the supervision cannot separate the roles"
```

The twelve channels are supervised independently.  When two of their targets
nearly coincide, each channel is still told exactly what to predict and both
answers are consistent -- the two maps are simply almost the same map.  That is
**redundancy**, not a conflict, and no gradient pulls a channel in two
directions.

The withdrawn phrasing implied the loss was self-contradictory on 75% of the
training set, which would have been a serious defect.  It is not one.  What
remains true, and is kept:

```
10,211 of 13,618 frames contain a near-coincident role pair
the four commonest are each a top-face edge against the bottom-face edge
beneath it, separated in 3D by the pallet's height alone
aliased frames score better, not worse, on the existing checkpoint
```

`TRAINING_FILTER` stays `KEEP` and the pool is unchanged.
