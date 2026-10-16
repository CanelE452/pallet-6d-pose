# Ongoing bounded decisive point/line experiment

## Completion supersedes the running status below

Main session44750 completed exit0;9cells×2000=18000updates,18cal/val evaluations,6cell causal
interventions and report complete. PIPELINE_COMPLETION elapsed1759.6sec. Gate fails all three
Hough seeds (+0.95% / -7.94% / -5.85%, catastrophic0/2/8); direct also fails. Real evaluation0.
Crucial nuance: within-Hough line-on improves over line-off13.65/5.53/10.78%, and unordered
geometry versus newly trained point improves5.47/15.43/9.69%. StockYOLO4.18px versus newHough
30.88/27.47/32.14px: current architecture NO_GO, not universal fusion impossibility.
See `_docs/notes/point_line_fusion/02_local_and_identity/pallet_point_line_decisive_v6_results.md` and run INTERPRETATION_KO.md.
NumPy audit PASS all9cells, maxprimarydiff6.93e-8/rawdiff5.27e-5px.
Report regenerated with numbered corners and unordered geometry captions. Three actual panels
inspected before label-only amendment. Browser after xdg-open notconfirmed; directgoogle-chrome
opened and xprop/xwininfo confirmed0x62003e1 active/IsViewable with expected title. BROWSER_OPEN
htmlSHA af1925eb00692f97ea70ba110efc24b934ef33db989bd682f6099a7efbe90af5.
Latest user asks whether Hough cannot be used in YOLO. Already answered in commentary: it can;
this was YOLO26. Failure is preserving whole-model accuracy, not connectivity. Do not switch to
DOPE or claim global YOLO>DOPE benchmarks. Final labelled largest-loss panel inspected.
Discord completion-only notification delivered (DISCORD_NOTIFICATION.json). Automatic approval
review rejected the initial metrics-bearing payload; no detailed results were sent externally.
Latest user asks whether failure causes are being checked individually: explain completed controlled
diagnostics, distinguish confirmed weak new heads/initialization sensitivity from unproven visibility
and domain-gap explanations. Current candidate fails; fusion itself is not disproven.

User asks to assess shared conversation, predict failure reasons and verify whether fusion works.
Earlier request explicitly authorized running unfinished experiments. Latest asks for a decisive
experiment, not endless redesign. Do not claim universal impossibility from a failed bounded pilot.

## Current experiment

New standalone code `scripts/research/pallet_point_line_decisive_v6/` and output
`data/pallet/results/pallet_point_line_decisive_v6/`.
Shared conversation fetched by curl to `/tmp/pallet_shared_conversation.html` (HTTP200), decoded
React stream at `/tmp/pallet_shared_decoded.txt`; long relevant messages at
`/tmp/pallet_shared_messages.txt`. It argues for dimensions + valid symmetry + joint physical pose,
and comparing predicted versus oracle lines with identical points/starts.

Another session is independently running `pallet_point_line_v5_local`; never edit or stop it.
No new subagents spawned (developer forbids unless explicitly requested).

This v6 copies its local source and fixes:
1. theta round-to-pi target needs rho sign reversal (reproduced near179deg misplaced x=-.5→+.486).
2. Hough mode seam uses rho-reflected neighbors and sign-aligned homogeneous line averaging.
3. solver line energy/Hessian gets sigma²/3 to match point units and three incident edges per corner.
4. eval catastrophic failures use actual raw pixel errors, not a hardcoded900px image diagonal.
5. restores the7 excluded detector-match failures to val512; calibration253+3=256.
The original v5 files/results remain untouched. These new defects do not explain all historical failures.

17 geometry/gradient tests PASS. Real synthetic B4 Hough8-update CPU smoke complete. Preflight
hashes all2,533 images/features/targets, confirms cross-split byte SHA overlap0, internal projected
targets agree exactly. This does NOT independently certify original renderer labels or real manualGT.
Original object-frame audit1200 frames is inherited (SOURCE SHA in PREFLIGHT.json).

Main launched under escalation with env `/home/minjae/anaconda3/envs/pallet-yolo26/bin/python`:
`python -u .../scripts/research/pallet_point_line_decisive_v6/cli_driver.py`
workdir results v6. **Live unified exec session44750** (poll write_stdin). GPU accessible only
outside sandbox; readonly normal filesystem tools work normally. Training3arms×3seed×2000B4,
AdamW1e-4. All numerical core/CLI/intervention sources locked in PROTOCOL_LOCK.json; DO NOT EDIT.
Report-only `report.py` was developed after lock and can still be corrected, preserving numbers.
At last check point seeds1/2/3 complete, direct1complete/direct2ongoing. Driver continues training,
all calibration/val score+evaluate, comparison, CPU interventions, report, PIPELINE_COMPLETION.
Do not stop at training success. Verify actual2000 markers and final artifacts.

## Pre-training diagnostic results

ORACLE_PREFLIGHT.json: existing v5 point_seed1 frozen, synthetic calibration253. Same predicted
points/actual starts, corrected balanced solver:
none primary.09608091/rawmean83.85262px; GTlines primary.02701798/rawmean23.70340px.
GT-near starts: none.09065127, GTlines.02503929. All coverage1.
This is privileged oracle evidence, NOT actual performance and NOT a comparison to original YOLO.
User explicitly told this distinction, since new point head is weak.

STOCK_GEOMETRY_BASELINE.json: additional original stock R0 actual inference505synthval frames,
checkpoint970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7.
Matches existing cache boxes within.1px using observed bbox only; labels loaded after inference.
Scoring applies optimal one-to-one UNORDERED corner assignment to handle camera-dynamic vs
object-frame semantics; same scoring applied to new pose models by report.py. This is a geometric
backstop, NOT main same-ID metric or deployable ID correction.
Stock geometric rawframe mean4.17867/median2.34857/P906.87221px, n505.
User informed that beating a weak new baseline alone cannot count as improvement.

QUANTIZATION_DIAGNOSTIC.json: exactGTline at target bin centers, calibration253/3033validedges:
36x65 median1.09261/mean1.56374/P903.40049px; nonlearned180x257reference median.25336/P90.78476.
Not a hard precision bound because neural decode interpolates locally. No new setting selected.

## Remaining required work

- Monitor main44750 with meaningful updates (<60sec gaps). No final before results.
- Driver may fail at reporting/diagnostic runtime; preserve errors and frozen core, repair reporting
  or make an explicitly versioned diagnostic repair if necessary, do not silently alter lock.
- Inspect VERDICT, DIAGNOSTICS, DECISION, STRONG_BASELINE_COMPARISON, TRAINING_TREND,
  CONDITION_BREAKDOWN, LINE_ACCURACY. Gate: everyseedH>P >=1% mean improvement,
  median/P90/coverage nonworse, 0 original≤10px→>50px/posefailure. HvsD separately.
- report.py makes8overlay images (4SHAselected+2best/2worst) and HTML. Must view actual PNGs,
  check generated HTML, open actual desktop browser via escalated command and verify window
  (prior user says HTML must automatically open; previous false-open issue).
- If synthetic gate passes, do not call it final success; assess strong-stock comparison and real DEV
  contract before any actual-real claims. No FINAL data or GT relabeling. Existing reused DEV only.
- If negative, explain exact scope and failure mechanism. Need distinguish underfit new head vs
  domain transfer; realwasnotrun if gatefails. Actual training loss trends are in report artifact.
- Update notes (currently protocol_notes SHA frozen; add result to a separate RESULT note or preserve
  original lock provenance when appending), handoff and final report interpretation.
- Completion Discord was explicitly authorized earlier. Use existing `scripts/research/discord_notify.py`
  `notify(message,run_dir,evidence=DECISION.json)` under escalation. Do not print hook secrets.
  Send final verdict+numbers only after report finished; save receipt. Do not commit/push unless asked.

## Interpretation for user

Tentative: share's common geometry approach is plausible, not proven. Main candidate failures:
amodal-box lines vs physical image edges; wrong parallel-edge/role choice; inconsistent confidence
and weighting; weak newly trained point/pose heads. Sharing backbone does not preclude complementary
spatial evidence. Correct-line oracle improves but does not prove learned lines will transfer.
PCGrad cannot correct role/visibility/label/solver issues; historical surgery did not establish stable gain.
Original41.49 was a different checkpoint/population/pooledpointP90, NOT current frame-meanP90.
