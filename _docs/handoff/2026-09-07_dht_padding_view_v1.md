# DHT padding and viewpoint experiment setup

User request: configure padding experiments and evaluation across camera views. The user also wants generated HTML opened automatically. This turn prepares the reproducible main experiment and separate small pipeline checks; it does not start the eighteen main trainings.

- Code: `scripts/research/dht_padding_view_v1/`; protocol: `_docs/experiments/dht_padding_view_v1/README.md`.
- Results: `data/pallet/results/dht_padding_view_v1/`; run commands from that directory with its `PURPOSE.md`.
- Python: `/home/minjae/anaconda3/envs/pallet-pose/bin/python`.
- Design: reflect101 / black / normalized-mean constant padding, all100px, float32 preprocessing, same400px input and frozen synthetic DOPE backbone. Two2048-frame training mixtures, seeds1/2/3, fresh DHT8 heads,6000steps/batch12 each.
- Frozen manifest4148records: existing2740 plus low_train1024, low_val128, low_test256. Existing record order is preserved; use population indices rather than assuming appended order. Manifest SHA256: `3ed767b5605ef26208def32cc10684447dcf17ceef3409eaa508a445a3c57246`.
- DATA_AUDIT PASS: no selected image duplicates or new renderer-group crossover. Base train elevation:0 below15degrees. Low-balanced:492below5,532at5–15,412at15–30,612at30+. Data mixture also changes renderer/assets/backgrounds; it does not isolate camera view causally.
- Real evaluation uses inherited canonical manual DEV52 only. No real optimizer inputs, final-test access or real checkpoint selection. Synthetic backbone pretraining/holdout overlap remains unverified because the inherited full source manifest is unavailable.
- Model geometry and loss reuse frozen `deep_hough_side_v1/{dht.py,network.py,targets.py}`. Eight amodal structural side supporting lines, not physical visible-edge GT and not all12cuboid edges.
- `prepare.py --phase all`: immutable metadata, three full feature caches and separate24-synthetic-frame smoke caches. `train_eval.py` uses a fixed CONFIG budget and verifies cache bytes; main budget cannot silently become a short run.
- Smoke: `smoke/` has its own CONFIG(stage=smoke), manifest, caches and outputs;2mixtures×3padding×seed1,20steps/batch4. Never put these metrics in main performance tables.
- Driver defaults to plan-only; `--execute` runs prepare→18train/eval→verified report. HTML opens once by default; `--no-open` disables it. Resume requires identical config/manifest/cache/code identity.
- Evaluation: angle≤5degrees AND original-pixel GT endpoint-to-line mean distance≤8px. Report median/P90/success,nframes,nroles by seed, view and group. Zero cells NO_DATA; fewer20frames descriptive. No deployment pass-rate threshold invented.

Preparation status and browser QA are recorded in the result directory. Main status must stay NOT_RUN until actual final-step6000checkpoints and matching metrics exist.

Completed setup: allthree4148-frame caches and separate24-frame smoke caches exist;6smoke cells each reached20steps and evaluated60supported roles (360rows total). `SMOKE_CHECKS.json` PASS includes independently recomputed full batch chains, matched initializations and artifact provenance. No resume interruption test was performed. `SETUP_COMPLETION.json` records PREPARED, main NOT_RUN,0/18 main checkpoints. `dashboard_qa/BROWSER_QA.json` PASS: 6GT-only examples, low_test true yaw<5degrees14SMALL, realworldanglesunknown52, group split22/12/18, no JavaScript errors. `GALLERY_OPEN.json` records the final desktop browser handoff separately.
