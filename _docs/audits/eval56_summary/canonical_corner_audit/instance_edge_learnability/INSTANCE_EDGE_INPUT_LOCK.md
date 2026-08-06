# Input lock

```
{
 "created_utc": "2026-08-05T20:45:13Z",
 "branch": "main",
 "head": "179585a013c6d530f8a2c904ecca6abef3eaf769",
 "origin_main": "179585a013c6d530f8a2c904ecca6abef3eaf769",
 "git_status": "M _docs/history/.last-compact-resume.md\n M _docs/history/2026-08-06.md\n?? Deep_Object_Pose/common/instance_edge_head.py\n?? Deep_Object_Pose/common/instance_edge_topology.py\n?? challenge/tests/test_instance_edge_learnability.py\n?? scripts/stage0/instance_edge_learnability.py",
 "python": "3.10.20",
 "torch": "2.1.1+cu118",
 "cuda": "11.8",
 "opencv": "4.9.0",
 "gpu": "NVIDIA GeForce RTX 3080",
 "a1_checkpoint": "weights/paper_s2_pdg/A1/epoch_003.pth",
 "a1_sha256": "00a0dcd8730e21d14b8a86e2f2a398650b78026006e4e358eabc438148fb9657",
 "a1_vgg_checksum": "700b7bef16fd7737522633b9a0f726a4573c035814639bc0504e617d5ae8d9cf",
 "ep57_vgg_checksum": "700b7bef16fd7737522633b9a0f726a4573c035814639bc0504e617d5ae8d9cf",
 "a1_vgg_equals_ep57": true,
 "a1_parameter_checksum_before": "5b8a3f651120648377327d33ae7089f458194212678f977a6d49df29d30c1c7f",
 "ep57_sha256": "c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896",
 "ep57_sha256_expected": "c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896",
 "feature_taps": {
  "F100": "vgg[17] ch=256",
  "F50": "net.vgg(x) ch=128"
 },
 "eval56": {
  "n": 56,
  "membership_sha256": "d4eb5ebe4f30d87bc23fe356482f9aa0db844f6cb984663e1137245ad096d347",
  "splits": [
   "eval"
  ],
  "per_domain": {
   "outside": 22,
   "noapril": 12,
   "cad": 22
  }
 },
 "wood": {
  "n": 45,
  "membership_sha256": "ebcc4164779593cf980459db98c0396941ead0b2e13a40d43afb1a979bb3a7a6",
  "per_domain": {
   "wood_183705": 25,
   "wood_184309": 20
  }
 },
 "synthetic_root": "data/pallet/training_data/paper_4pallet_mask_v1",
 "synthetic_manifests": {
  "train": {
   "n": 3039,
   "sha256": "a149475a65cf8ca86fdf1fa3cf030062d48d33415f43c2bcf76802e8b78c2772"
  },
  "val": {
   "n": 1045,
   "sha256": "ffb52c5c584d1978e90a548f81c8ae46bc009b5a6f6a48d46feb2878805d6f03"
  },
  "untouched": {
   "n": 5916,
   "sha256": "d26bda235a9a54dfb0f68247478c8dde70f3759350e1007a783fb64d1967951c"
  }
 },
 "ppd_checkpoints": {
  "L0": {
   "path": "weights/paper_s2_ppd_t2_screen/L0/last.pth",
   "sha256": "618911a97e4d18b9b427bc179064aa1946c14d1a0f4b4b06b874e87f9d6ef384"
  },
  "M0": {
   "path": "weights/paper_s2_ppd_t2_screen/M0/last.pth",
   "sha256": "7e68b02917ebf4b2b1e1ae38fae6210d251d95358ceda012a62d3992422e26c4"
  },
  "M1": {
   "path": "weights/paper_s2_ppd_t2_screen/M1/last.pth",
   "sha256": "9cc5678e0b0675c02d8cc8ff3904d06df377c7123a25e0f53c9f5d10abc8966c"
  }
 },
 "ppd_run_state": {
  "L0": {
   "arm": "L0",
   "epoch": 20,
   "completed": true,
   "head": "8f0a61fe5b506337c64bf72b42d821cc029c7dfc",
   "timestamp": 1785597536.7239118
  },
  "M0": {
   "arm": "M0",
   "epoch": 20,
   "completed": true,
   "head": "8f0a61fe5b506337c64bf72b42d821cc029c7dfc",
   "timestamp": 1785598811.6952538
  },
  "M1": {
   "arm": "M1",
   "epoch": 20,
   "completed": true,
   "head": "8f0a61fe5b506337c64bf72b42d821cc029c7dfc",
   "timestamp": 1785600087.2334623
  }
 },
 "final_test_prohibited_tokens": [
  "capturenight08",
  "capturenight09",
  "capturepallet07",
  "capturepallet09",
  "testset_full8_manifest",
  "handannot17"
 ],
 "final_test_open_count": 0,
 "a1_optimizer_creation_count": 0,
 "a1_training_step_count": 0,
 "checkpoint_mtimes": {
  "weights/paper_s2_pdg/A1/epoch_003.pth": 1785925240.8967078,
  "weights/paper_s2_stageB/net_epoch_0057.pth": 1783582660.0541291
 },
 "topology_sha256": "9c0aafa1292eba3e844f429bc59e852b7950f155d1eca725d9cd79fa37d8103d"
}
```
