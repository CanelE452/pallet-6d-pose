# What the F50 adapter result is allowed to be called

`a6c987c` stands unchanged.  This narrows how it may be cited before the next
screen is built on top of it.

## Established

```
the constrained F50 adapter is active, not inert
    alpha 0 -> 0.07404, relative F50 change 29.7%, cosine 0.955988,
    parameter delta norm 21.63
it improves on F0
    angle median +17.42%, offset median +21.23%
    angle p90 +36.85%, offset p90 +24.60%
it misses the relative threshold
    40% required on both medians, neither reached
the large D0/D2 gap seen in the broad late-A1 arm does not reproduce here
    F2 1.0445 / 1.0766 against F1 1.4255 / 1.2994, at every mark
```

## Not established

```
that the broad unfreeze's parameter count itself caused the specialization
    it was observed in the broad late-A1 arm and not observed in F2.  Those are
    two configurations differing in what is adapted, where, and how much; the
    count is one of several differences and nothing isolated it
that specialization is solved
    F2 is worse than F1 on all four statistics.  A model that learned less has
    less to specialize with, and a small ratio between two populations is not a
    generalization guarantee
that adapter capacity is exhausted
    alpha stopped moving after 8,515 while the body's relative effect kept
    climbing to 29.7%.  A gate that plateaus while its body grows is not the
    signature of a component out of room
that a frozen A1 is sufficient
    nothing in this program has passed the task gate
that role-encoder capacity is the bottleneck
    no role-encoder screen has been run
```

The last one matters most here, because the next screen is about to test it.
It is being tested, not confirmed in advance.

## The next question

```
DOES_ONE_MORE_ROLE_CONDITIONED_NONLOCAL_BLOCK
CONVERT_THE_F2_SIGNAL ?
```

F2 stays exactly as it is -- the adapter is not removed, resized or retuned, and
A1 is not unfrozen again.  One additional role-query refinement block, gated by
a learnable scalar initialised to zero so that step 0 is the F2 function
exactly.  One factor.

A null result there will mean that *this recipe* -- one extra block at the
current width and head count -- did not convert the signal.  It will not mean
role-encoder capacity is not the bottleneck, and that sentence is forbidden in
advance.
