# Published research snapshot

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
