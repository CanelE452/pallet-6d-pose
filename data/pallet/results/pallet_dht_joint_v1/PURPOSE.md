# Purpose

Full-network, synthetic-supervised YOLO26n point + global Deep Hough / inverse-Hough feature fusion. This experiment answers the user's explicit architecture request and is separate from frozen point-conditioned line correction, pseudo-label filtering, and self-training. Protocol and results are developed in this folder only; original paper outputs and previous completed experiments remain intact.

Proposal: `_docs/notes/pallet_dht_joint.md`. Primary arm: `hough_joint`; controls: equally trained `point_only` and `hough_features`. Main train budget is not frozen until the synthetic plumbing and resource probe completes. Reused real DEV319 and negative2689 must be newly inferred for every final model. A diagram or smoke test is not task completion.
