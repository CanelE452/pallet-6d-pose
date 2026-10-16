# V38 target-point line uncertainty

Frozen V36 grid/both line candidates and V34point anchor,3seeds. New prototype modules predict signed normal target and residual uncertainty. No neural optimizer updates;36HistGradientBoosting fits x60iterations=2160boosting iterations. Three fits are repeated separately during audit.

Training12sourcegroups, calibration4, check8. OOF3fold mean residuals train uncertainty without in-sample mean residuals. Originalupstream V36 and V35median offsets previously saw all16oldtrain groups: heldout calibration is only for the NEW modules. Empirical90% candidate/pointmax coverage is not a formal conformal guarantee. All data is reused synthetic2CAD; no real or final-holdout evaluation.

Use `/home/minjae/anaconda3/envs/pallet-yolo26/bin/python` from repo root:

```bash
python scripts/research/pallet_line_uncertainty_v38/run38.py --phase all
```

Driver skips completed stage markers. Individual stages: train, eval, audit, probe, audit_probe, report, deliver. Desktop delivery needs Chrome and the actual X11 desktop. Existing explicitly authorized Discord completion is sent through the local hook after delivery verification.

Protocol fixes hyperparameters and partitions before evaluation. `SOURCE_BINDINGS` freezes training code, `PRIOR_BINDINGS` freezes previous results. `models/*.pkl` contains locally created scikit-learn models; `arrays` holds aligned predictions including OOF, scale, intervals, selection and point outputs.

Primary uncertainty modes share exactly the same mean predictor and candidate corrections. Constant vs appearance vs geometry scales compare uncertainty information. Pointwise24max calibration addresses candidate multiplicity at a point, not conditional safety after selecting the few accepted points.

`PROBE_PROTOCOL` is post-primary: exact mean/radius oracles and calibration-only quantile sensitivity .50/.70/.80/.90/.95. These do not alter the original90%decision. Check-selected thresholds and oracle scores are never reported as deployable improvements.

Audits:42point arrays,6finalmean/18OOF/12scale replays,3fits repeated with calibration/check corruption, exact normal squared-gain identity,18oracles and60calibration-threshold replays, unchanged previous files. Not all36models trained twice. See ENVIRONMENT.json for versions.
