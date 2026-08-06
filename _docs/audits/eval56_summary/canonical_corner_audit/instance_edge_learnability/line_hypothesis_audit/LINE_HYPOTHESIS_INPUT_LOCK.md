# Audit input lock

Read-only: no training run, no optimizer, no checkpoint written.

```
{
 "created_utc": "2026-08-06T04:09:53Z",
 "head": "eff34b21f900c2ef50fb6b711e857116ed1c6a7b",
 "origin_main": "eff34b21f900c2ef50fb6b711e857116ed1c6a7b",
 "git_status": "M _docs/history/.last-compact-resume.md\n?? Deep_Object_Pose/common/instance_edge_hypotheses.py\n?? challenge/tests/test_instance_edge_line_audit.py\n?? scripts/stage0/instance_edge_line_audit.py",
 "audit_type": "read-only mechanism audit",
 "training_runs": 0,
 "optimizers_created": 0,
 "checkpoints_written": 0,
 "state_all_done": true,
 "complete_decision": "DIRECT_12EDGE_HEAD_NOT_LEARNABLE",
 "mechanism_labels": {
  "automatic": "DIRECT_12EDGE_HEAD_NOT_LEARNABLE",
  "primary": "DIRECT_12EDGE_FIELD_LOCALIZATION_FAIL",
  "secondary": "SYNTHETIC_TO_REAL_TRANSFER_COLLAPSE",
  "qualification": "The automatic label is the Phase G rule applied verbatim. It understates the result: channel identity is learned (12/12 active, min recall 0.997, Hungarian == fixed), the fixed incidence topology works (+38pp over shuffled), and what fails is precise line localization, which then collapses on real data."
 },
 "tests": {
  "passed": 491,
  "failed": 0,
  "skipped": 0
 },
 "a1_sha256": "00a0dcd8730e21d14b8a86e2f2a398650b78026006e4e358eabc438148fb9657",
 "a1_sha256_unchanged": true,
 "a1_training_steps": 0,
 "a1_parameter_delta": 0,
 "final_test_open_count": 0,
 "sealed_tokens": [
  "capturenight08",
  "capturenight09",
  "capturepallet07",
  "capturepallet09",
  "testset_full8_manifest",
  "handannot17"
 ],
 "arms": [
  "L12-F50",
  "L12-MS"
 ],
 "l5_excluded": "the five-class representation is structurally non-generative (O5C collapses eight corners onto two points)",
 "sets": [
  "val",
  "untouched",
  "eval56",
  "wood"
 ],
 "checkpoints": {
  "L12-F50|1": {
   "path": "weights/paper_s2_instance_edge/L12_F50/seed1/epoch_020.pth",
   "sha256": "29d8e90a558ebbcd176d8f40e7daa65f5fd9c7d7a23ed28771b4c41130e95983",
   "selected_epoch": 20,
   "selection_basis": "synthetic validation only"
  },
  "L12-F50|2": {
   "path": "weights/paper_s2_instance_edge/L12_F50/seed2/epoch_019.pth",
   "sha256": "0b179b2b25651c49e2d5850f25a45dd156b24e7f2331eba69472e78a21b48559",
   "selected_epoch": 19,
   "selection_basis": "synthetic validation only"
  },
  "L12-F50|3": {
   "path": "weights/paper_s2_instance_edge/L12_F50/seed3/epoch_018.pth",
   "sha256": "f8b9e6f0d1a13c922040ad6fe72bf8aa9814bbf7440a8b7f7860f6f58715c94f",
   "selected_epoch": 18,
   "selection_basis": "synthetic validation only"
  },
  "L12-F50|selected": {
   "seed": 3
  },
  "L12-MS|1": {
   "path": "weights/paper_s2_instance_edge/L12_MS/seed1/epoch_019.pth",
   "sha256": "b4fab6c2e9a4fb4d50b3a539de98c881e0724a1fe6d570e753ae1b29d78c05cd",
   "selected_epoch": 19,
   "selection_basis": "synthetic validation only"
  },
  "L12-MS|2": {
   "path": "weights/paper_s2_instance_edge/L12_MS/seed2/epoch_016.pth",
   "sha256": "2ef7ccf163b7e82f2d119e439ae4470a23bcf06d961b92ed9475b4c2096267c7",
   "selected_epoch": 16,
   "selection_basis": "synthetic validation only"
  },
  "L12-MS|3": {
   "path": "weights/paper_s2_instance_edge/L12_MS/seed3/epoch_018.pth",
   "sha256": "9f5d314b7837392636991730bd3011f3b832bee4a85716144e3b20a18d6d0ba7",
   "selected_epoch": 18,
   "selection_basis": "synthetic validation only"
  },
  "L12-MS|selected": {
   "seed": 2
  }
 }
}
```
