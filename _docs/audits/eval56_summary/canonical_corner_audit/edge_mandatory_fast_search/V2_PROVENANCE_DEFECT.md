# The V2 O0 result was recorded, not reproducible

`9dfa414` says the V2 screen runs "in a committed script".  It does not.  What
was committed is a helper module:

```
present   sha_file · line_rect_intersection · sample_strip · Refiner
          wrap_half_pi · budget_losses · raster_lines

absent    dataset loader · GT edge construction · jitter generation
          O0 training loop · optimizer · evaluation loop
          O1 · F50/F100/MULTI/RGB arms · checkpointing · CLI main
```

O0 itself ran from a scratchpad heredoc -- the same defect the V1 addendum names
in V1, repeated one commit after naming it.

```
O0_RESULT_RECORDED            true
O0_EXECUTION_PATH_REPRODUCIBLE  false
```

`9dfa414` is not edited.  Its numbers -- angle median 0.0149 degrees, offset
median 0.0415 cell, GT_LINE_COVERED 100% -- are kept as
`O0_PREVIOUS = HISTORICAL_RESULT`.  They are not treated as verified until the
committed runner reproduces the gate, and that reproduction is recorded
separately as `O0_REPRO_RUN` rather than merged into the old figures.

The claim I should have made in `9dfa414` was that the *sampler and loss* were
committed, not the screen.
