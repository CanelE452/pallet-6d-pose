# Stored-cost diagnosis: structured v2, seed 1

This is a read-only analysis of saved predictions and synthetic calibration archives. No threshold was selected on real images, no model ran, no ground truth changed, and no HTML was rendered. Candidate names were reconstructed with the frozen proposal generator and its original FP32 input/output convention; all selected coordinates and valid masks match the saved predictions.

The full point + segment + DHT verifier retains identity on all 319 real frames: identity-minus-best cost has median 0.02667, P90 0.11141, and maximum 0.40831, below the synthetic-selected margin 0.5. The lowest-cost candidate is nevertheless nonidentity in 312 frames. These are learned scores, not probabilities.

Synthetic calibration explains the conservative margin. Every registered margin below 0.5 violates the clean-data constraints. For the full model, margin 0.25 changes one clean frame and increases mean error 5.20101→5.22820px and P90 6.97947→7.01640px. Margin 0.5 retains all clean calibration frames and fixes most deliberately C4-permuted states. The latter have median gaps around 2.4, far above the real gap distribution. This distribution mismatch does not establish a unique causal explanation for failure.

| Frame | Stored arm / lowest-cost candidate | Identity − best | Actual result |
|---|---|---:|---|
| plastic_day_01:005838 | segment / 45: C4=3, corner5 half-snap | 0.85340 | Selected; observed-point mean 118.581→5.167px |
| plastic_day_01:005838 | full / 42: C4=3, corner2 half-snap | 0.06408 | Identity retained; candidate post-hoc mean 5.032px |
| eval_pallet09:1778653664407620608 | full / 43: C4=3, corner3 half-snap | 0.40831 | Identity retained; candidate post-hoc mean 167.681px vs baseline 11.147px |
| eval_pallet07:1778652166837872128 | full / 23: C4=0, corner7 half-snap | 0.30007 | Identity retained; candidate post-hoc mean 224.876px vs baseline 237.222px |

The user case also contains C4 slot1 with post-hoc same-ID mean error 19.863px, but the full model gives it cost 2.29447, worse than identity 2.22497. Thus the failure includes ranking the wrong layout, not only rejecting a good candidate at the margin. This fixed candidate diagnosis used GT only after cost ranking; it is not an automatic correction or real-data selection rule.

The apparent segment-arm aggregate improvement comes from exactly one observed frame; its other 308 observed frames remain unchanged. Point-only changes one different frame and worsens it. Full changes none. Ten frames have no observations under the unchanged original metric policy. A lower margin cannot be justified by the rescued example alone: the full model's largest real gap belongs to a harmful candidate.

Artifacts: COST_GAP_DIAGNOSIS.json contains all 319×3 rankings, proposal-kind mappings, representative records, all calibration rows, state-wise gap distributions, and input SHA bindings. The report source is reusable and its actual 319-frame input contract passed separately in ../report_input_validation_seed1.json; rendering remains deferred by root instruction.
