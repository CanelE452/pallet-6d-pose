# Source audit

Corrected conclusion: 60,000 source records directly joined to unchanged P cache indices; no missing dimensions. G38 uses **fixed renderer XYZ** from immutable GEOMETRY_SIDETABLE, fully checked against the original fixed_dimensions builder for all40,000 rows. Legacy20,000 fixed-XYZ inputs are unchanged. The model receives WDH=XYZ[0,2,1]. Raw G38 dimensions_m is camera-facing and excluded from conditioning. See CANONICAL_DIMENSION_CORRECTION.json.

Offline provenance reconstruction uses original renderer index metadata to undo camera-facing encoding. Inference only reads the resulting fixed sidecar; it never consults GT pose, clicked geometry or a chosen symmetry phase.19345 exact input rows changed before any DIM main optimization. The six metadata-free N0/N1 runs remain valid. Pre-correction sidecar/normalization/protocol and discarded N4 smoke are preserved in history.

Registry dimensions match the requested values. Real C2 is the pre-existing benchmark convention (wood physically unreviewed); square C4 is the approved task contract. Unknown source assets remain C1 even if square.

Only 55,915 of 55,980 paper training records are matched/usable, exactly as OLD_P. All missing/unmatched rows remain in evaluation/audits. Original source image/cache files are not copied or rewritten.

Historical preliminary statements, INVALID and superseded: this file initially said raw G38 dimensions_m was canonical and sorted-extent equality was sufficient. A first zero-update assertion had been relaxed on that mistaken interpretation. Tracing the original builder exposed the camera-facing encoding; the full correction above supersedes both assertions. Equality of unordered physical extents does not certify fixed object-axis semantics.
