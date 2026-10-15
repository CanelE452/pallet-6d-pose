# Published research snapshot

The current implementation is [statewise joint PnP v3](nonsevere_v1/statewise_joint_ot_20261014_v2/README_KO.md), using `RUNTIME_CONFIG_V3.json`. It excludes S before image queries, freezes separate B/I image measurements, and applies every O/T region simultaneously in continuous SE(3). S is projected only from the resulting registered pose. Two B/I plus O/T produce an explicitly ambiguous pose family; infeasible O/T remain unresolved. Prior completion claims about `anchored_completion_v1/region_completion_v1` are withdrawn: those historical studies compared finite candidates and included S in image regions.

The corrected implementation passed 95 focused checks and fresh-RGB contract tests. The solver-only 98-record replay uses immutable prior RGB measurements and separate post-seal scoring. Reused clean/moderate TEST49 results remain exploratory: mean translation is 3.514cm versus original PnP 3.363cm, and 5cm/5deg success is 39/49 versus 40/49. Overall superiority and independent natural truncation recovery are not established. Current results, source snapshots, conditional-T failures and real-image pose-family figures are preserved; the earlier descriptions below refer to historical methods.

The optional refiner and clean/moderate experiment records are published on
`research/visibility-curriculum-20261013`. The main workspace and its unrelated
uncommitted changes are preserved.

Start with [nonsevere_v1/README_KO.md](nonsevere_v1/README_KO.md) and
`nonsevere_v1/BOUNDARY_PROGRESSIVE_CONFIG.json`. The small clean-source Tiny
checkpoint required by this API is included. Raw RGB datasets, generated masks,
feature arrays, vendor repositories, and large external checkpoints are not
part of the Git snapshot. Historical studies retain their recorded local paths
and hashes; their input datasets are needed for a full rerun.

Restore the public foundation dependency at its exact recorded commit:

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python \
  -m scripts.research.pallet_visibility_curriculum_20261013_v1.bootstrap_foundation
```

The bootstrap validates the pinned MobileSAM source and checkpoint hashes from
`MOBILESAM_ASSET_MANIFEST.json`. It verifies an existing installation and does
not overwrite one with different contents. Inference also needs the project's
Python environment with NumPy, SciPy, OpenCV, PyTorch, torchvision and timm.

The publication preserves the experimental status: the TEST49 images had been
seen before, external occlusion is an image-based hypothesis, and the final
ordering advantage over matched all-visible optimization was not established.

`PUBLICATION_MANIFEST_V1.json` identifies the prior completed work. New count
rules and the literal PoseFix-plus-SubPix comparison are subsequent work.

The subsequent count-rule implementation is documented in
[nonsevere_v1/SELECTION_RULES_KO.md](nonsevere_v1/SELECTION_RULES_KO.md).
Its explicit optional configuration is
`nonsevere_v1/BOUNDARY_RULE_CONFIG.json`; the API is
`boundary_rule_api.BoundaryRuleCorrector`. The fixed 5+/4/3/2 rules were checked
on CAL46, attached to both existing upstream pipelines on reused TEST49,
and verified with 161 software tests plus 190 subtests. The count means
image-supported boundary measurements, rather than initial geometric points.
The new API does not generate another-instance or depth evidence itself.

`PUBLICATION_MANIFEST_V2.json` records the additional source, comparison,
frozen count-study records, independent audits, and current selection guide.
The earlier snapshots and protocols retain their original hashes; validation
amendments are recorded separately. The observed tradeoffs do not establish
a universal winner, and the existing default model has not been promoted.

The subsequent statewise method is documented in
[the completed method and results](nonsevere_v1/region_completion_v1/README_KO.md)
and [the paper draft](nonsevere_v1/region_completion_v1/PAPER_DRAFT_EN.md).
Its explicit optional configuration is `nonsevere_v1/PROTECTED_COMPLETION_CONFIG.json`.
The implementation preserves image measurements in a mixed 2D readout, keeps
rigid projections separate, and compares finite pose proposals with empirical
coarse image regions. The soft region term changed zero poses relative to the
identical candidate pool without that term. Measurement preservation improved
the reported outline coordinates; superior hidden recovery is not established.

The new method passed 255 tests and 199 subtests, a 720-output independent
inference audit, and fresh-RGB parity on the predeclared CAL example. Paper
figures are available as PDF, SVG and PNG. `PUBLICATION_MANIFEST_V3.json` records
the added method sources, all frozen outcomes, paper artifacts, and the small
352KB fitted joint-head parameter archive. Restore its public backbone with:

```bash
python -m scripts.research.pallet_visibility_curriculum_20261013_v1.bootstrap_joint
```

The backbone is restored from the official PyTorch model endpoint and checked
against its full recorded SHA256. Input RGB datasets and upstream detector
weights remain external dependencies. Existing initial/native correspondences
must be supplied in the exact model basis expected by the selected configuration.
