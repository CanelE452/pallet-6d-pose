# V38 point-conditioned line uncertainty

Does prediction of uncertainty at the target point transfer to unused groups and permit beneficial conservative line corrections?

Frozen V36 grid/both candidates and V34anchor3seeds. Existing600renderobservations/436detected, same2CAD. New mean/scale fit12groups, calibration4groups, reused check8groups. No neural updates/newrealGT/manualedits. Existing upstreamV36 and V35offsets previously saw all16oldtrain groups: calibration is held out of NEW mean/scale fitting only, not independent of upstream. Empirical coverage, not a distribution-free conformal guarantee.
