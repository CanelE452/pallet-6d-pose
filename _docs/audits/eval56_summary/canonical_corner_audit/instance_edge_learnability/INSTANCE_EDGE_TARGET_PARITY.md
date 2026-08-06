# Target generator parity

Ground-truth endpoints through the target generator and back out of the fixed O12 decoder, against the same decoder run on analytic geometry.

```
{
 "frames": 200,
 "parity_corners": 1599,
 "excluded_off_frame_corners": 1,
 "median_cells": 0.0,
 "max_cells": 1.4142135623730974,
 "p99_cells": 1.414213562373092,
 "clamp_violations": 0,
 "gate_median": 0.5,
 "gate_max": 1.5,
 "passed": true
}
```

The parity population was fixed before the run: a corner counts only when all three incident edges have an in-frame clipped segment and the corner itself projects inside the image.  A corner outside the frame cannot be represented in a rasterised field, so including it would measure clipping rather than the generator.
