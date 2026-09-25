# S1 preservation preflight

HEAD `4de998d9d8794a8da4a2ef5b2f6a007235587660` / main. Previous outputs hash-bound and immutable. GPU checked separately. Original S1/GEO_LINEAR frozen. One adapter architecture, one fixed320-step run, no DEV validation during fitting.

Unspecified clean coordinate regression is fixed before training as SmoothL1(beta1) in640-input pixel units at frozen top1 detection. The same units apply to distillation; each term is normalized by its valid xy scalar count. Cached original augmentations and teacher mask are reused.
