# 011067 role-contract audit

No training, inference, pseudo-label creation, tuning or GT edits. Run from repository root in `pallet-yolo26`.

```bash
python -m scripts.research.pallet_verified_anchor_v1.complete_directive
python -m scripts.research.pallet_verified_anchor_v1.review_metadata_qa --smoke
# Actual metadata QA already saved by human; never automatically KEEP.
python -m scripts.research.pallet_verified_anchor_v1.finalize_metadata_qa
python -m scripts.research.pallet_011067_corner_contract_v1.preflight
python -m scripts.research.pallet_011067_corner_contract_v1.build_candidates
python -m scripts.research.pallet_011067_corner_contract_v1.geometry_audit
python -m scripts.research.pallet_011067_corner_contract_v1.pnp_audit
python -m scripts.research.pallet_011067_corner_contract_v1.inventory
python -m scripts.research.pallet_011067_corner_contract_v1.review_front_role --prepare
python -m scripts.research.pallet_011067_corner_contract_v1.review_front_role --smoke
python -m scripts.research.pallet_011067_corner_contract_v1.review_front_role
```

STOP until real user decision exists. Prior model exposure in the conversation is disclosed; this is not a retrospectively blind study. UI shows only neutral direct-click markers and explicitly inferred role text/outline, never model coordinates or current-GT label.

For this run, the user explicitly selected B in chat and asked the assistant to enter it. `record_chat_decision` transcribes those quoted statements, with confidence NOT_REPORTED (never fabricated), and refuses to overwrite an existing decision. The user separately clarified A/B canonical180-degree equivalence. The conversation disclosed that relation before final signoff; do not claim a pristine blind review. This script is run-specific, not an automatic default for future subjects.

```bash
# Guarded: refuses missing/invalid human lock before reading model payload.
python -m scripts.research.pallet_011067_corner_contract_v1.model_role_diag
python -m scripts.research.pallet_011067_corner_contract_v1.finalize
python -m scripts.research.pallet_011067_corner_contract_v1.report
python -m scripts.research.pallet_011067_corner_contract_v1.test_audit
```

`report` and `test_audit` can also run before the human decision; they report PENDING and do not create a final decision or figure07. Exact coordinates, UI mapping, decision and locks stay in the private `data/pallet/results/pallet_011067_corner_contract_v1` namespace. Public role permutations are the mathematical C4 candidates, not the private A/B/C/D presentation mapping.

Original artifacts and reference remain unchanged; stage only this code namespace, public audit documents/figures and anchor `_FINAL`/V2 files. No raw RGB, checkpoint, NPZ cache or private mapping.
