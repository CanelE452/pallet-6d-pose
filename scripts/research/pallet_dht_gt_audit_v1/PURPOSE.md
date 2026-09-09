# Manual GT audit of the completed point–line decoder pilot

Requested 2026-09-09. Consumer: diagnosis for the project owner.

Question: does the reported baseline pooled-point P90 of 41.48732863482036 px
represent model error against visually valid manual labels, or is label uncertainty
material? Also check whether separating annotation provenance changes the evidence
for combining point and Hough information.

Inputs are the completed `pallet_dht_decoder_probe_v1` predictions/metrics and the
same 319-frame `PAPER_EVAL_ALL_POS` DEV manifest. No new training, inference, model
selection, GT overwrite, or relabelling to match predictions is authorized by this
audit. Preserve the original same-ID, visibility and match contract and distinguish
any diagnostic subset from the official full-population result.

Before numerical analysis: reconstruct the metric from saved prediction coordinates
and actual scored GT; inspect click/projection/unknown provenance, visibility,
centroid/corner, session and historical human-review strata. Record the two order
statistics interpolated at P90. For fixed membership/masks derive deterministic
P90 bounds under independently assumed GT location error at most 2/5/10 pixels;
these assumptions are not measured annotation repeatability.

Visual analysis: inspect raw images with GT and IDs first, without model overlays.
Use all-frame contact sheets only for triage, then detailed views for selected
large-error/boundary/prior-review examples and metadata-selected controls. Record
which images and points were actually inspected, visible support, occlusion,
semantic ambiguity and candidate annotation errors. Automated consistency checks,
small PnP residuals, and agreement with a prediction do not certify visual GT.

Deliverables: separate numerical audit, explicit review coverage/limitations,
interactive GT/raw/prediction report, concise conclusions and review candidates.
Any candidate correction must retain an independent image-based justification and
remain a sidecar; no corrected full-set accuracy is claimed without verified
coordinates. Open the final HTML visibly and send the already requested Discord
completion notice after checking the report.

Failure criteria: any source hash or original metric mismatch requires explanation
before conclusions. Ambiguous/occluded GT remains unresolved, not labelled correct
or automatically removed. The audit can establish numerical accuracy, examples of
true errors and limits of plausible small click noise without certifying all GT.
