# RUN PROVENANCE — paper_s2_micro_arch_screen

- created: 2026-07-29T10:21:06.084964+00:00
- git HEAD: 9045c69224dca86e0225b3b777ad720afa1aa1b3
- checkpoint: /home/minjae/Documents/github/pallet-pose/weights/paper_s2_stageB/net_epoch_0057.pth (SHA-256 c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896, unchanged)
- python 3.10.20 / torch 2.1.1+cu118 / cuda 11.8 / opencv 4.9.0
- gpu: NVIDIA GeForce RTX 3080
- mechanism-val membership SHA: 8d086cd7f8a20cf6fd76af4b26fd5b2ed01ed2a1173e441ae777909202260914
- final-test open count: 0

## Manifest hashes
- micro_train_B_manifest.json: 7cad403983165c8a0dd0a216f1d678e9123057f9f1f4e86176b4038089ef8298
- micro_train_A_manifest.json: e9acd08f6dfe3103d3f8e7eece3b23e9b146593a55a3ac62a15224777527d89b
- B_membership_hash: 99c2de1b9dc7087a86bbf06443cfe41b04c53eddfc0f8c4baffaa248500da436
- A_membership_hash: 6313b92fff193ca7a48e23eca5b8cf69628377a1332173f303b2730d8ed7e307
- mechanism_val_membership_sha256: 8d086cd7f8a20cf6fd76af4b26fd5b2ed01ed2a1173e441ae777909202260914

## Run configs
- M0_B: manifest micro_train_B, trainable 17673, epochs_run 2, best_epoch 0, sampler_order_hash 86caafb66a685b5d…
- B1: manifest micro_train_B, trainable 19897, epochs_run 6, best_epoch 4, sampler_order_hash 939053ea64cb41c8…
- M0_A: manifest micro_train_A, trainable 17673, epochs_run 10, best_epoch 10, sampler_order_hash fe74012152661ab2…
- A1: manifest micro_train_A, trainable 17673, epochs_run 10, best_epoch 10, sampler_order_hash fe74012152661ab2…

## Reused (not reimplemented)
- evaluator: `paper_s2_mechanism_diagnostic` (decode_all / FrameGeometry / metrics)
- loss: `heatmap_refinement.channel_masked_mse`
- target: `utils_belief.CreateBeliefMap(clip_at_border=)` + `spatial_keypoint_validity`
- decoder for the edge loss: `diffpnp3d_loss.LocalSoftArgmax2D`

