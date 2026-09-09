# High-resolution local point + Hough refinement

Active goal: achieve verified point/line accuracy improvement. This experiment replaces coarse cached-P4 line evidence with image-resolution local edge evidence while preserving existing points and semantic IDs. It does not claim to solve C4 ID ambiguity or large displacement. No real GT training, pseudo-label filtering, or GT modification. Prior runs remain immutable.
