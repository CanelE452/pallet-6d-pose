# Pallet DHT coupling v2 — LIVE statistics and final report

Latest snapshot: 2026-09-08T14:56:53.245853+00:00. **12/12 new main training cells complete; 12/12 new real evaluations. Current stage `aggregate`.** Read result STATUS.json before any action; this text is a snapshot. The driver is alive and authorized work must continue through actual results, visualization, visible HTML and Discord. Do not stop merely with a background launch. No new persistent goal requested; do not create one. No commit, no unrelated changes, preserve completed v1.

## Process and continuation

Repo /home/minjae/Documents/github/pallet-pose. Python /home/minjae/anaconda3/envs/pallet-yolo26/bin/python. Use login=false and PYTHONPATH=repo, OMP/MKL/OPENBLAS threads4. Owned driver unifiedexec session58650, driverPID1367168. Child PID/stage/log are in fresh STATUS.json. No other GPU work while main runs. All agents are currently available/idle; GPU work belongs solely to the root driver.

Read-only progress: python -m scripts.research.pallet_dht_coupling_v2.monitor --run-dir <absolute resultroot>. A functions store key `dht_v2_monitor_cmd` contains a compact version. Current wait pattern is functions.exec with pragma yield_time_ms60000, await setTimeout50000, then exec_command(cmd=load(storekey)). Keep concise user commentary about each minute. Monitor counts input batches started, not certified optimizer completion. Failure file/logs must be inspected, never overwritten silently. Driver stdout is DRIVER_STDOUT.log. Do not launch a duplicate or kill unrelated processes.

## Frozen design and sources

New source scripts/research/pallet_dht_coupling_v2; results data/pallet/results/pallet_dht_coupling_v2. Four arms balanced, pcgrad, balanced_pcgrad, incidence ×seeds1/2/3. Each same R0,55,980synthetictrain/4,020val,FP32B16/nbs16,AdamW1e-4,2full epochs/6,998updates,whole network continuation as v1. Last epoch EMA only. No self-training, filters, real tuning, test-time snapping, checkpoint selection or independentFINAL access. All new models require actual319positive+2689negative predictions; nine v1 same-budget controls(point_only/hough_features/hough_joint ×3seeds) reused only via REUSED_CONTROLS.json9cells/646bindings. Source all60,000image/label hashes revalidated.

TRAIN_PROTOCOL SHA `93cbe83285610dea8feed41bc0152afc47609c9088408ac64582fcd7040ddaaf`. TRAINING_PREFLIGHT SHA `552b63b71d4b29d09671a9bb3c74e00b2c8c7d8d218318e72f21175c83aa177b`. COMPLETION_CHAIN_BINDING SHA `3df4e015a86d1e1a5846344cd65b9f5a900741352d1f10578e9191887df71705`. Training, driver and reporting/statistic sources are now immutable unless a concrete failure requires an explicitly preserved reporting-only repair. The chain binds36sources including v1helpers, original statistics/evaluator and viz palette/style. Never silently loosen numerical criteria.

balanced lambda_line=.1*stock.o2m/.8 (epoch1 .1, epoch2 .0125). pcgrad fixedline.1, symmetric original-partner two-task SUM projection on actual structural parameter intersection, before original global clipping. balanced_pcgrad both. incidence fixedline.1 plus lambda_incidence=.1, actual stock positive assignment + predicted endpoints + image-predicted same-role full Hough mixture. Same forward as v1. GT support/assignment only in incidence, not a GT-bin restricted prediction. Physical edge visibility labels unavailable; frozen source singleton objects. Private gradient preservation applies before global clipping, not final private updates.

Statistics: all4new×2references(joint,point)×6metrics=48 registered family,100,000paired-session draws/Bonferroni percentile intervals;95%descriptive, matched3seed means,13 reusedDEV sessions. Overall superiority requires all6fullpopdirections+familyCI+perseed matching/posecoverage. Do not select winning arm. AP/negative FP separate. Main real evaluator unchanged.

## Completion chain

Driver executes all12train → TRAINING_AUDIT/MATCHED_CONTROL_AUDIT(exactfulltraces and initial tensors) →12actualevaluation → runtime21models×26frames×3repeats=1638timings → aggregate → report → independent audit → headless actual visual QA → waits for root VISUAL_REVIEW.json → finalize(visible browser + authorized Discord + completion). Distinct final model states are measured, not required; a null intervention may legally yield identical outputs.

Strict runtime1e-4/rtol0 remains; numeric failures are retained as RUNTIME.PASS=false with completed collection, never called a pass. Old v1 has7/702strict failures caused by diagnosed sparse/CUDA numerical variation; new runtime status must be observed independently. Structural/availability/nonfinite changes remain fatal. Final execution completion is not accuracy success or all-checks-pass.

At awaiting_actual_screenshot_review: inspect actual saved screenshots with view_image (especially original wrong-case and role7), then write VISUAL_REVIEW.json complete/PASS, html_sha256, actual_screenshots_inspected=true, and input_sha256 binding inspected PNG/currentHTML/ACTUAL_VISUAL_QA. Root then lets driver proceed to actual browser visibility confirmation and Discord200/204. This is internal artifact review, no user permission question. Final report title `Pallet DHT Coupling · 점·선 결합 학습 검증`, gallery319raw×21models(new12+ref9), defaultbalanced predeclared. Original wrong-case eval_pallet07:1778652166837872128, farleft GT4–7 is role7 rear_left_height; role3 is a different unsupported edge. Heatmaps are learned line evidence, not calibrated confidence/attention/causal attribution.

## Completed checks and interpretation

31CPU tests +fourGPUsmokes16updates each passed, identicalv1initial state/fullsmoketrace. GPU peak≤5.25GiB, batches223/280/280/312ms in smoke. Original fresh-resume(None) failure archived and fixed before any update. Actual resume trace/RNG/LR/updates passed; CUDA weights not bit-exact. Final source hashes are in TRAINING_PREFLIGHT.

Fixed16syntheticGPUdiagnosis3rawoldjoint checkpoints: globalstock/linecos +.04576/−.00381/+.02894, but seed2Hough-only−.2791 (pose+RLE-only−.4283). One-batch local evidence, not population or causal accuracy. Same-graph TF32 backward probe reduced gradient reconstruction relativeL2 about3.77e-4→1.06e-6; all4testedconflict decisions unchanged, mainprecision unchanged. Firstbalancedmain fiveepoch1probes likewise tinyglobalcos but stronger early Hough-only conflict; all5reference/efficient conflict signs matched.

CPU same-graph criterion_epoch1_equivalence: first7losscomponents and all518parameter gradients/None masks exact v1 vsnewbalanced. Scalar difference oneFP32ULP7.629e-6 is sum7vs8 with appended0; SUM_REDUCTION reproduces it. Does not prove identical fullCUDAtrajectories. Exact initial/data matching does not imply same numerical training trajectory.

Independent actual gradient_audit helper passed completedbalanced8row path and completedpcgrad6998row path; no further duplicate tests planned before finalfullaudit. PCGradseed1 applied2316/6998steps (epoch1 1140/3499). Read-only CPU predictor load/hook check oldpoint/features/joint/newbalanced passed0/1/1/1Houghhooks,0forwards; receipt provenance/predictor_loader_check/RESULTS.json. Visual source tests7+headless21combinations PASS; fixture never marked actualcomplete.

Geometry agent wrote provenance/architecture_limits/ARCHITECTURE_LIMITS.md and ANALYTIC_EXAMPLES.json: correct two nonparallel role lines constrain a corner; a single infinite line lacks tangent position/endpoint order; jointly wrong points+lines or jointly swapped roles can satisfy incidence. Nearparallel intersection ill-conditioning and coarse lattice are possibilities, not measured failure causes or an8px error floor. Existing point/line GT supervision still penalizes these wrong outputs in training. PCGrad cannot add missing visibility/semantic information. Link this analysis in final explanation as conditional mechanism, not causal verdict.

## Completed cells snapshot

```json
[
  {
    "name": "balanced_pcgrad_seed1",
    "optimizer_steps": 6998,
    "checkpoint_sha256": "d87d4158ccf8c3a293752fa8d99493bf2ebef39289bece246348934b2fd5fbe6",
    "batch_trace_sha256": "30d9e3137d31b4e52d757c222afa0840c45c0c1ae29488b3913f01d01f2bd3f1"
  },
  {
    "name": "balanced_pcgrad_seed2",
    "optimizer_steps": 6998,
    "checkpoint_sha256": "15688b26c371b636b32696d5b145514abd15b2086a94c1d72d8635af33bc2f4d",
    "batch_trace_sha256": "89d08b30eae65d48408befe5df8d190a51e64b211815e3feb8a5ff360671270f"
  },
  {
    "name": "balanced_pcgrad_seed3",
    "optimizer_steps": 6998,
    "checkpoint_sha256": "40be40e7ec8ee4f3be11bc450861e2a897a298884f5d579ad58158350a375eee",
    "batch_trace_sha256": "5b07386a24e8a6a684964dc93274ea52b27a8693284c665521440e9e48e94de3"
  },
  {
    "name": "balanced_seed1",
    "optimizer_steps": 6998,
    "checkpoint_sha256": "110fabdc43f8df1c573dc1dc305e0131f2dcddd79bb9f891861112a53a36eb2b",
    "batch_trace_sha256": "30d9e3137d31b4e52d757c222afa0840c45c0c1ae29488b3913f01d01f2bd3f1"
  },
  {
    "name": "balanced_seed2",
    "optimizer_steps": 6998,
    "checkpoint_sha256": "964b4edfb83764644218d32220c8e9c54675848e4a875f01dc5005e60b625bb4",
    "batch_trace_sha256": "89d08b30eae65d48408befe5df8d190a51e64b211815e3feb8a5ff360671270f"
  },
  {
    "name": "balanced_seed3",
    "optimizer_steps": 6998,
    "checkpoint_sha256": "a840ef34df551d93e9554cb9b4446bf7af575fe9bc94b7fb6c18b6d14c154db0",
    "batch_trace_sha256": "5b07386a24e8a6a684964dc93274ea52b27a8693284c665521440e9e48e94de3"
  },
  {
    "name": "incidence_seed1",
    "optimizer_steps": 6998,
    "checkpoint_sha256": "134bb477bcbaae1f4e4ec312d44b7eb682cdd6712dc00b675a84c0944afbc2e4",
    "batch_trace_sha256": "30d9e3137d31b4e52d757c222afa0840c45c0c1ae29488b3913f01d01f2bd3f1"
  },
  {
    "name": "incidence_seed2",
    "optimizer_steps": 6998,
    "checkpoint_sha256": "34d87f19d58b58e8410bd3869a775805f267740387af2ee0ba7397df2734d561",
    "batch_trace_sha256": "89d08b30eae65d48408befe5df8d190a51e64b211815e3feb8a5ff360671270f"
  },
  {
    "name": "incidence_seed3",
    "optimizer_steps": 6998,
    "checkpoint_sha256": "a79714cd902f84202deda47ea0b0db4b3674493624588972e079181b74145b58",
    "batch_trace_sha256": "5b07386a24e8a6a684964dc93274ea52b27a8693284c665521440e9e48e94de3"
  },
  {
    "name": "pcgrad_seed1",
    "optimizer_steps": 6998,
    "checkpoint_sha256": "ea48bf085e140d49805da87ba662eba80fc6442ceedd41b3d0ee6df54b7d1b6f",
    "batch_trace_sha256": "30d9e3137d31b4e52d757c222afa0840c45c0c1ae29488b3913f01d01f2bd3f1"
  },
  {
    "name": "pcgrad_seed2",
    "optimizer_steps": 6998,
    "checkpoint_sha256": "ee4d709a297f928299475b6f96a318369c29b44fafab2f2fa5f2973dfd41bb0f",
    "batch_trace_sha256": "89d08b30eae65d48408befe5df8d190a51e64b211815e3feb8a5ff360671270f"
  },
  {
    "name": "pcgrad_seed3",
    "optimizer_steps": 6998,
    "checkpoint_sha256": "fd4157ed32366a27fe2e15b6d0a92475876dd8d015fa3c461014ad441bbaf471",
    "batch_trace_sha256": "5b07386a24e8a6a684964dc93274ea52b27a8693284c665521440e9e48e94de3"
  }
]
```

The archived earlier handoff is provenance/handoff_history/pre_three_completed.md. Notes _docs/notes/pallet_dht_coupling.md and history2026-09-08 contain prior details. Update notes/history/index/final handoff after actual completion, then give a self-contained Korean result with report link and honest uncertainty. Do not end now.

## Two-seed gradient interpretation (actual, read-only)

Operator wrote provenance/two_seed_gradient_summary/{SUMMARY.json,INTERPRETATION.md,summarize.py}; SUMMARY SHA2d438fe2ff7cc37ae55dbae220ce9894fc724174819944126b6f24997566d42f. All8 checkpoints/internal EMA6998/source/protocol/fulltraces verified. Root independently checked all8 INITIAL_STATE.pt file SHAs exact against their v1 joint same-seed initial files (seed1 69d35089bfbf1dd16a07f8c4b2f6a2e14b2080f3d097770426121b418e4fd672; seed2 71754e5fedf223b000891c0ee94128cc48040ff600a5a9c768b8df1a86287a7f).

PCGrad applied2316/2060 and balanced-PCGrad2230/1998, each denominator6998. PCGrad median global weightedline/stock norm ratio E1→E2: seed1 .00621→.04470, seed2 .00610→.03998; balanced-PCGrad .00642→.00566 and .00604→.00499. These are preprojection gradient magnitudes, not real accuracy or causal claims. Local Hough/pose+RLE/global directions differ; only8fixedprobes/run,7validHoughcosines. Main incidence own separated gradient unmeasured. 48non-incidence fixedprobes reconstructed/reference conflict signs match, numericaldifferencespreserved. The reusable summarizer accepts --seeds 1 2 3 --output-dir <new absolute provenance dir> after all12complete; refuses partial/overwrite. Use then for final3seed summary, noGPU. Allagentsidle.

## Main training complete; actual evaluation started

All12main completed6998updates and passed TRAINING_AUDIT/MATCHED_CONTROL_AUDIT at2026-09-08T14:39:21UTC. Actual12finalEMA states distinct, allsame-seed full augmentation traces exact across4new and oldjoint, allinitialtensors exact. TRAINING_AUDIT SHA7d2fe853d998ede3a33fce16b6488bffbd2b72800a433b23895c16645d62c944. Driver now actual12evaluation in sequence, then runtime→aggregate→report→independentaudit→actualvisualQA→root screenshot review→visibleHTML→Discord. Operator currently writing final3seed gradient summary, visual agent reviewing actualfirstbalanced_seed1 outputs withoutnewforwards. Root must still inspect real screenshots and writeVISUAL_REVIEW; do not stop with launchedjobs. Actual evaluationfirstcell finished~1minute.

## All actual evaluations and runtime complete

All12new actual evaluations completed:3828positive and32268negative model-forwards (12×319/2689). Runtime21models/1638timings complete, strictparityPASS=false,7fails (balanced2,features4,balanced_PCGrad1). Maxpointdelta .006103515625px,box .0013427734375,score0,kpconf1.19209e-6. Preserve criterionatol1e-4/rtol0; do not relabelPASS. Meanofseedmedianms:point8.758,features10.852,joint10.670,balanced10.475,PCGrad10.788,balanced_PCGrad10.689,incidence10.489. Driver now100kpairedsession/48registered comparisons. Global accuracy preliminary means show mixed effects, do not claim stablegain; finalCI/verdictpending.

Operator final3seed summary provenance/three_seed_gradient_summary/SUMMARY.json SHA40cc8204fc4b04c7f1d73d16de05eb893b7a0ec0ccb682be108b7cc1011c8599:PCGrad6653/20994(31.69%),balanced_PCGrad6276/20994(29.89%). Fixedline ratioE2/E1medians6.40–7.20,balanced .827–.882. 72nonincidence independentgradientprobesconflict signs match; numericaldiffpreserved. Mainincidenceownseparategradientunmeasured.

Visualfirstactualbalanced_seed1review PASS, provenance/first_actual_evaluation_review,3008predictions/14NPZ allbindings,role7correctGT4–7 butpointIDerrorremains. Root actually viewed raw_case_role7.png. This is NOT final report screenshot review. Geometryagent now examining originalcase all21 savedpredictions into provenance/problem_case_final_review. NoGPU/newforwards. Root must still inspect actualfinalhome.png andproblem_case_left_height.png fromvisual_QA, thenwriteVISUAL_REVIEWboundcurrentHTMLandACTUAL_VISUAL_QA, letfinalizeropenvisibleHTML+Discord, verifyCOMPLETE, updatedocs. Do not stop now.
