"""Read-only campaign evidence snapshot and requirements/report draft writer.

Reads instructions, source and existing receipts only. It never imports a
scientific producer/scorer/trainer or opens per-frame numerical GT/RGB/weights.
Only a new report directory and a new private draft artifact are written.
"""
from __future__ import annotations
import argparse, ast, csv, hashlib, io, json, os, re, subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def write(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_text()!=text and 'DRAFT' not in path.name and path.name!='REPORT_SCAFFOLD_KO.txt':
        raise ValueError('Existing artifact differs; use a new snapshot directory: '+str(path))
    path.write_text(text)
def jsontext(value):return json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'


def git(repo,*args,raw=False):
    value=subprocess.check_output(['git','-C',str(repo),*args],env=dict(os.environ,GIT_OPTIONAL_LOCKS='0'))
    return value if raw else value.decode().strip()


EVIDENCE = {
 'plan':'CAMPAIGN_PLAN.json','start':'START.json','env':'ENVIRONMENT_INITIAL.json','p0':'P0_REUSE_PROOF.json',
 'baseline':'P0_BASELINE_REAGGREGATION/EVALUATION_SEAL.json','sets':'P0_FIXED_SETS/FIXED_SETS.json',
 's1':'S1_EXECUTION_SUMMARY.json','s1audit':'S1_REUSE_AND_ADMISSION_AUDIT.json',
 's1parity':'S1_CONSTRUCTED_AND_UNION_PARITY.json','s1pred':'s1_predictions/PREDICTION_SEAL.json',
 's1train':'s1_training/TRAINING_MANIFEST.json','s1eval':'S1_EVALUATION/EVALUATION_SEAL.json',
 's1ind':'S1_INDEPENDENT_FINAL/INDEPENDENT_VERIFICATION.json','s1paired':'S1_INDEPENDENT_FINAL/INDEPENDENT_PAIRED.json',
 's1cohorts':'S1_ANALYSIS/FIXED_COHORTS.json','s1gallery':'S1_GALLERY/GALLERY_MANIFEST.json',
 's1privacy':'S1_GALLERY/PIXEL_REVIEW.json','s1argv':'S1_EXECUTED_COMMANDS.json',
 's1inputs':'s1_prepared/INPUT_BINDINGS.json','s1normalizer':'s1_labels/TRAIN_ONLY_NORMALIZER.json',
 'dstrict':'D_STRICT/SUMMARY.json','dfixed':'D_FIXED_COMMON_GRAPH/SUMMARY.json',
 'deval':'D_POSTSEAL_EVALUATION/EVALUATION_SEAL.json','dpaired':'D_POSTSEAL_EVALUATION/PAIRED_DECOMPOSITION.json',
 'dgamma':'D_GAMMA_AUDIT.json','calprotocol':'S2_CALIBRATION/PROTOCOL.json',
 'dgammaexec':'D_FIXED_GAMMA_EXECUTION.json','dgammasummary':'D_FIXED_COMMON_GRAPH_FIXED_GAMMA/SUMMARY.json',
 'dgammaeval':'D_FIXED_GAMMA_POSTSEAL_EVALUATION/EVALUATION_SEAL.json','dgammasupp':'D_FIXED_GAMMA_POSTSEAL_EVALUATION/SUPPLEMENT_SEAL.json',
 'dgammacmp':'D_FIXED_GAMMA_POSTSEAL_EVALUATION/GAMMA_CORRECTION_COMPARISON.json',
 'calmodel':'S2_CALIBRATION/MODEL.json','calsummary':'S2_CALIBRATION/SUMMARY.json',
 'calpackets':'S2_CALIBRATION/PACKET_SEAL.json','s2quiesce':'S2_UNIT_WEIGHT_QUIESCE_START.json',
 's2lock':'S2_FULL_NORMAL/PRODUCER_LOCK.json','s2progress':'FULL_S2_PROGRESS.json',
 's2h1plan':'S2_UNIT_WEIGHT_H1_AUTOMATION_PLAN.json','s2go':'S2_UNIT_WEIGHT_FULL_GO.json',
 's2stop':'S2_UNIT_WEIGHT_QUIESCE_COMPLETE.json','s2newlock':'S2_FULL_UNIT_WEIGHT/PRODUCER_LOCK.json',
 's2newgo':'S2_UNIT_WEIGHT_FULL_GO.json','s2review':'S2_UNIT_WEIGHT_PREPRODUCTION_REVIEW.json',
 'asset':'s3_s4_production/MATERIAL_ASSET_AUDIT.json','structure':'s3_s4_production/STRUCTURAL_REVIEW_SELECTION24.json',
 'materiallock':'s3_s4_production/METHOD_LOCK_RESOURCE_LP4.json','materialpilot':'s3_s4_production/PILOT_RECEIPT.json',
 'materialinfo':'s3_s4_production/INFORMATION_TABLE.json','materialargv':'s3_s4_production/ACTUAL_ARGV.json',
 'resource':'s3_s4_production/RESOURCE_AMENDMENT_LP4.json','rasterparity':'S3_S4_RENDER_THREAD_PARITY_LP4.json',
 'materialtests':'s3_s4_production/CONSTRUCTED_TESTS.json','materialfull':'s3_s4_production/FULL_RECEIPT.json',
 'materialfinish':'s3_s4_postprocess/FINISH_RECEIPT.json','fpolicy':'F_OBSERVER_POLICY_LOCK_V2.json',
 'frule':'F_SELECTION_RULE_CLARIFICATION.json','fcore':'F_CORE_PRECOMPUTE_LOCK.json',
 'fgo':'F_PRECOMPUTE_FRONTEND_GO.json',
 'fobservations':'F_OBSERVATIONS/OBSERVATION_MANIFEST.json','fcalpackets':'F_CALIBRATION/PACKET_MANIFEST.json',
 'postbankparity':'POST_BANK_S1_VALIDATION/PARITY_WITH_INDEPENDENT_S1.json',
 'ftests':'F_CONSTRUCTED_IMAGE_TEST.json','fchoice':'F_SYSTEM_CHOICE.json',
 'fits':'GLOBAL_H1_FIT_ATTEMPTS.jsonl','hevents':'H1_EXECUTION_EVENTS.jsonl',
 'headtest':'H1_CONSTRUCTED_PRE_FIT_TESTS.json','headlock':'H1_PREFIT_SOURCE_LOCK.json',
 'adapter':'EXTRA_ADAPTER_HANDOFF_FINAL.json','checkpoint':'CAMPAIGN_LIVE_CHECKPOINT_01.json'}


# status E: evidence for the stated completed scope; P: pending or provisional;
# N: not applicable for a specified missing independent input. Never infer full
# campaign completion merely from source code, a plan, pilot, or partial rows.
REQUIREMENTS = '''
0|P|Execute all P0/S1/D/S2/S3/S4/F/R branches to actual results or concrete blockers|checkpoint|Finish all registered stages; this draft is not final completion.
0|E|Preserve authorized scope: isolated worktree, small adapters, no default deployment or new public raw data|start,plan|Final publication audit remains required.
0|E|Do not buy new paid resources, disclose new external raw data, force-update main or expand publication authorization|plan,start|No commit/push/deploy/resource purchase by this task; ordinary fast-forward publication only within parent authorization.
0|E|Search local manifests/configs/assets before declaring blockers; keep independent branches running|p0,s1inputs,asset|Only independent physical labels and exact CAD identity remain uncertified.
1|E|Keep pose improvement, role-policy effect and causal explanation as separate claims|s1,dfixed,fpolicy|S1 union result is not independent B/I evidence.
1|E|Separate role, visibility, physical ownership, precision and information; never force B=line/I=point|materialinfo,fpolicy|Physical material correspondence is an automatic hypothesis, not certified truth.
1|E|Use confirmed/inferred/unverified labels and never assign zero accuracy to unexecuted work|checkpoint|Retain these labels in final report.
1|E|Keep representative seed20261018 and original historical baselines distinct|plan,baseline,s1|Report 120/117/121, not the best seed as representative.
1|E|Distinguish generated upper bound151/154 from actual selector success and fixed31/41/38 cohorts|s1,s1cohorts|S1 selected119/118/120; none of three novel PF-good cases selected.
1|E|Do not relabel earlier107-path condition-bit experiment as original120 full B/I removal|fpolicy,frule|Final F will be a matched new backend, not original120 parity.
2|E|Freeze latest origin/main execution SHA and record original dirty HEAD separately|start,plan|Worktree HEAD/origin3553202; START original dirty repo HEAD7e92 has a different snapshot role.
2|E|Preserve original dirty status/index/HEAD/protected files and old run receipts|start|Current read-only preservation snapshot is included; verify again before publication.
2|P|AGENTS/research contracts read and no old authorization/gate copied or disabled|plan,headlock|Adapters are honest new code; final contract-read audit must be consolidated.
2|E|Reuse original numerical primitives through separate new adapters|adapter,headlock,s1parity|Protected old source/gates remain separate from new campaign.
2|E|Reuse frozen environment and forbid YOLO/N3/PF backbone retraining or unnecessary upgrades|env,s1inputs,materiallock|No new package installation or backbone fit evidenced.
2|E|Bind real/synthetic inputs, old observations/full banks/checkpoints, LOCAL/OBS/RISK/PF and canonical reference provenance|p0,s1inputs,asset|Private path closure in existing P0/S1 receipts; no raw assets copied to public draft.
2|E|Keep real238=PRIMARY192+CAL46 and synthetic1003=train768+VAL235 exact ordered IDs|plan,p0|All later stages must retain these denominators before COMPLETE.
2|E|Verify K, native XYZ/Q, units, image canvas/padding, actual consumed coordinates and hashes|p0,s1inputs,materialtests|S1 exact prepared-canvas crop parity and proxy pixel/registration tests documented.
2|E|Use original real C2 and per-row synthetic C1/C2; no invented90degree evaluation symmetry|plan,s1ind|Strict physical-pose dedup does not merge C2 phases.
2|E|Keep GT pose/box/visibility/mask out of production and CAL only in registered calibration|calprotocol,s1pred,materialpilot|Final campaign-wide access-receipt consolidation remains pending.
2|E|Reuse and canonically reaggregate all saved baseline outputs before any new baseline solve|baseline,p0|No old1241 pose regeneration required for S1.
2|E|Check fixed primitive parity plus full input/output contracts; predeclare numerical tolerance and bitwise result separately|s1parity,s1audit|Feature tolerance1e-10 was frozen prereference; old scalars/masks bitwise preserved.
2|E|Label nonparity fullnormal backend MATCHED rather than original E reproduction|s2lock,frule|See precise method-difference ledger.
2|P|Provide per-stage allowed status, hash and next action; do not treat diagnostic as full method|checkpoint|Draft matrix supplies snapshot; final unified STAGE_STATUS remains pending.
3|E|Optimize repeated production rather than weakening the three-H1-seed comparison|s1,s2h1plan,adapter|Reuse deterministic packets/costs; own-bank H1 still required.
3|E|Cache keys include actual geometry/images/masks/q/effectivecov/typedpartial/source and consumed closures|s1audit,s2lock,materiallock,adapter|Gamma is also a consumed input and must be disclosed/fixed in D.
3|E|Cache actual ordered starts, feasible samples, branches, solver settings, models/env/seed and downstream bank/train inputs|s2lock,materiallock,headlock|All active final row seals still require end-of-stage verification.
3|E|Preserve every visited endpoint, rejected proposal and source alias, not only winners|s1audit,adapter,materialinfo|Material full archive ancestry validation remains pending until complete receipt.
3|E|Invalidate affected descendants when q/cov/factors/starts change; use exact unchanged rows only|s2quiesce,adapter|S2 weight correction requires new descendant keys/rows, not old-result relabeling.
3|E|PF union recomputes five gaps, bank mean/max/count, all logits and own train normalizer|s1audit,s1train,adapter|Fullnormal explicitly masks absent B/edge costs/gaps in both matched arms.
3|E|Original G10/G13 start independently from P1/WD0; E0 reads G10-selected family G13|s1audit|New fullnormal intentionally collapses G13 to G10; do not apply original dependency claims to it.
3|E|Reuse stored family readouts without new solve when only selection changes|dpaired|Family readout and within-family effects are separate diagnostic components.
3|E|Require identical post-dedup partial/source factors for strict2x2 diagonal reuse|dstrict,dfixed|103 strict-compatible,135 source-changing; not all238 pure source-matched interventions.
3|P|Share COMMON/BI fits only after complete packet/features/labels/checkpoint equality proof|fpolicy,adapter|No final F equality or fit reuse is yet established.
3|E|Oracle upper bound/fixed-bank rescoring/pilot/one seed cannot replace full automatic method results|s1,checkpoint|All pending branches stay incomplete.
3|E|Use <=24 deterministic timing pilot without accuracy selection and retain original numeric caps|s2go,materialpilot,plan|Technical attempts preserved; summarize aggregate costs across attempts.
3|P|Report optimizer/forward/render calls, peak memory, wall tails and exact reuse; separate prior cost from new cost|s1,materialpilot,s2progress|Campaign-wide actual cost ledger and elapsed wall remain pending.
3|E|Start CPUworkers<=3,BLAS1,GPUjob1; document measured parallel/resource amendments truthfully|plan,resource,rasterparity|LP4 is four internal software graphics workers plus two solver processes, not hardware GPU rendering.
3|E|Do not regenerate candidates per H1 seed or call those seeds three entire pipeline repetitions|headlock,s1train|Own H1 initialization repetitions only.
4|E|S1 both PF+SubPix anchors cover all238/1003 frozen inputs; infer missing synthetic pose outputs without backbone fit|s1inputs,s1|Frozen tensor rows reused;2006 new synthetic F calls,7453 cornerSubPix calls,0 NN forwards.
4|E|S1 preserves all old D1 physical candidates/aliases and strict numerical pose equality|s1audit|No C2 or90degree pose collapse.
4|E|Freeze reference-free complete-action admission and common baseline consequences prereference|plan,s1audit|Old eligibility unchanged; old baseline retraining not required.
4|E|Missing features use explicit validity, not confident zero evidence|s1audit,adapter|New fullnormal B/edge channels are deliberately invalid.
4|E|S1 each union trains three own-distribution H1 seeds and seals238 outputs before scorer|s1train,s1pred,s1eval,s1normalizer|3 completed fits/1176 scientific steps; baseline three fits exact reused.
4|E|Compare original H1/PF anchors and actual union on192/49/143/CAL46 with continuous/tail metrics|s1eval,s1ind|Full exact independent verification completed.
4|E|Report generated/eligible/selected, fixed31/41/38, novel3 and old12 recovery/damage|s1cohorts,s1paired|Novel3 selected success0/0/0; old12 recovery0/0/1.
5|E|D uses representative observation seed20261018 with all three saved observation seeds summarized|dstrict|Only one observation-seed solve diagnosis, not total model-seed uncertainty.
5|E|D states/camera/partial graph/factor dimensions/source incompatibility explicitly tracked|dstrict,dfixed|Logical fixed graph is not physical source identity.
5|E|Do not rebuild FACE partial during a purported single-q intervention or forge old typed archive provenance|dfixed|Explicit counterfactual provenance carries changed q/cov donor source IDs.
5|E|D keeps both W/D family problems unless exact full-objective transformation equality|dfixed|Equal rendered initialization alone does not imply equal native factors/objectives.
5|E|D counts limited-family solve budget<=1904 distinct from full original producer costs|dfixed|786 actual optimizer calls,547 parsed numerical reuses; not H1/full hierarchy evidence.
5|E|D reports within-family effects and family-readout contribution separately|dpaired|Postseal diagnostic decomposition exists; conclusions remain conditional.
5|E|Pure covariance intervention holds COMMON reliability Gamma fixed;retain original Sigma+Gamma result separately|dgamma,dgammaexec,dgammasummary,dgammaeval|Corrected238 diagnostic completed:64actualoptimizercalls/72familydispatches/916reuses;still conditional on COMMON logical graph.
5|E|Distinguish103 physical-source-compatible/135 source-changing strata from all238 logical fixedgraph correction|dstrict,dgammaeval,dgammasupp,dgammacmp|ALL238 correction changes16poses in eachSigma1cell butjointsuccessdelta0;no claim of source-independent physical causality.
6|E|S2 freezes COMMON20261018 q/dimensions/typedpartial graph and three C0/CP/CB arms|calprotocol,calpackets|Actual corrected unit-reliability packets and descendant production still pending.
6|E|S2 uses same canonical CAL46, zero-centered80% ellipse recipe, no PRIMARY/bias tuning|calprotocol,calmodel|241 full factors:222B/19I across4 recording groups.
6|E|S2 states pooled/role formulas,SPD/fallback,scene scaling,shrinkage and parameter counts prereference|calprotocol,calmodel|CP1 effective floor;CB2 effective role floors from3 raw fits; pseudo-count20.
6|E|S2 CAL LOGO internal evaluation is development, not independent validation|calsummary|CP and CB heldout coverage both.738095;CP NLL26.9067<CB26.9860 in existing summary.
6|E|S2 corrected packet contract holds intended common unit reliability while varying covariance|s2review,s2newgo|Old135armrows/37820calls/0unknowninterruptions preserved;corrected producer active,final performance unvalidated.
6|P|S2 reconstructs full own C0/CP/CB candidate banks for238+1003 and three H1 fits each|s2newlock,s2newgo|Actual corrected root S2_FULL_UNIT_WEIGHT with2workers;whole banks/9fits/realpredictions/evaluation still pending.
6|E|S2 is new matched fullnormal hierarchy rather than smaller107 backend or original120 parity|s2lock|Exact changes and masked feature channels are listed separately.
6|P|Do not hide CB no-gain; choose pooled if no strict LOGO benefit without coverage loss|calsummary,fpolicy|Provisional CP recipe choice must be reconciled with corrected reliability contract.
6|E|Calibration of derived canonical corners does not certify physical owner probabilities|calmodel,asset|Owner physical accuracy remains NA.
7|E|Search actualCAD/model/texture/registration dependencies and honestly classify structured proxies|asset,materiallock|Real CAD identity/signed axes unverified;missing namedGLBs use fixed structuredwood proxy.
7|N|Independent physical owner accuracy/oracle and exact physical CAD identity|asset,structure|No independent labels or verified exact object identity; automatic proxy experiment continues.
7|E|Select fixed24 structural review without PRIMARY accuracy tuning, preserve six-cluster constraint honestly|structure|REC_007 has only1 PRIMARY row; deterministic extra fill;human confirmations0.
7|P|Complete structural24 source classification/object-vs-occluder-vs-background-vs-unknown and actual/virtual junction review|structure|Selection exists; full derived review/classification audit still pending.
7|E|Exclude coplanar triangulation edges; retain registered material support IDs and object coordinates|materialinfo,materialtests|Creases measured on pre-anisotropic source geometry.
7|E|Use automatic current-pose zbuffer visibility and rawRGB/mask support, never GT visibility|materiallock,materialinfo|Predicted visibility/support remain model hypotheses.
7|E|Generic object3D point/line support adapter must not force new material supports into8native slots|materialinfo,materialtests|Projection/Jacobian/registration constructed checks exist.
7|E|COMMON/BI receive identical supports/cues/masks/visibility/search; dimensions follow independent directions|materialinfo|COMMON joint normalized rank vs BI cue priority; 2D/1D/none not role-forced.
7|E|Use source dedup/normalized mixture-none and do not reinterpret four global permutations as material owners|materialinfo,materiallock|Raw shared evidence consumed once; unknown correlation explicitly retained.
7|P|Generate all238/1003 S3 COMMON/BI actual endpoints then own-bank normalization/labels/3H1 each|materialfull,materialfinish|Producer active; no completed full receipt/H1 outputs at snapshot.
7|E|No large new owner network campaign or fabricated human confirmation|materiallock,structure|Automatic registered-proxy branch only.
8|E|Search frozen/standard refiners and choose one compatible fixed automatic whole-image path|materiallock|No compatible local frozen tracker/refiner found; implemented nonlearned DIS path is not blocked.
8|E|S4 actual render->observation correspondence->metricdepth object3D->PnP/refine->rerender update|materialinfo,materialtests|Whole native grayscale DIS FAST bidirectional; current correspondences archived.
8|E|Keep object mask/crop/init automatic;handle occlusion/depth/OOB/too-few matches/fallback and preserve failed proposals|materialinfo,materiallock|Zero GT crop/init/temporal state;valid correspondence failure is an algorithm outcome.
8|E|Disclose actual grayscale/fixed representation and do not claim all RGB,DeepIM or SRT3D reproduction|materialinfo,materiallock|Render-to-real transfer is an unverified method hypothesis.
8|E|Run distinct S4_PL/CONTOUR/IMAGE arms with same known proxy/K/info access and actual consumed-information table|materialinfo|PL does not consume structural mesh residual;CONTOUR andIMAGE do.
8|E|Select <=2 genuinely distinct physical starts without WD basis duplication and <=4 updates each|materiallock,materialtests|N0 then distinct E0/PF priority; internal refinement bound10.
8|E|Report method-specific inner work/wall, not only outer update equality|materialpilot,resource|Calls and wall differ across arms;renderer CPU cost must be explicit.
8|P|Generate all238/1003 S4 poses, union D1/PF, refresh all bank features, train three own H1 seeds each|materialfull,materialfinish,adapter|Three arms/9 fits pending;single-arm or fixed-bank score cannot substitute.
8|P|S4 coverage and selected recovery on fixed41/38 plus continuous/tail changes|s1cohorts,materialfinish|Await all sealed predictions and separate canonical evaluation.
8|E|Reset each image/start and use no temporal observation or GT initialization|materialinfo,materiallock|Single-image cold start; no tracking performance claim.
8|E|Truthfully record software EGL implementation and metric pixel-centre depth convention|materiallock,rasterparity|Mesa llvmpipe CPU;nonMSAA visible-depth pass;renderer principal point+.5 with stored K unchanged.
9|P|F mandatory after registered comparisons even if all stages fail to exceed original120|frule,fcore,fgo,fobservations,fcalpackets|Actual1241observations and1241 ownCALpackets complete;finalcandidatebanks/choice/H1/PRIMARY scoring still pending.
9|E|Freeze/disclose VAL/CAL selection and mandated real-development all-negative fallback|frule|S1 was already evaluated before clarification; PRIMARY used only for mandatedfallback bit,not unseen test.
9|P|Freeze F_SYSTEM_CHOICE and selected optional shared proposal set before union/H1/scoring|fchoice,fcore|System choice absent at snapshot;no F fit authorization from core lock.
9|E|Final rows ORIGINAL_120/MATCHED_COMMON/MATCHED_BI and no unsupported original120 parity|plan,frule|Original baseline exact reuse;matched new fullnormal comparison scope.
9|E|F both arms shareRGB/K/nativeQ/N0/states,masks,cues,search,budgets,E0readout,proposal admission and H1 contract|fpolicy|Requires full execution identity audit before final completion.
9|E|Callable intervention table covers actual cue generation/adoption,recovery,dimension,calibration and downstream roles|fpolicy|Policies1-3 vary;4 tied to own pooled recipe;5 retained both arms.
9|E|Strong COMMON keeps both exterior/interior cues and all tight/full masks; not just condition-bit removal|fpolicy,ftests|Actual image-line generation and adoption were constructed-tested.
9|E|F no duplicated one-pixel source masquerading as independent two-direction2D evidence|fpolicy,ftests|PolicyV2 guard fixed before F scientific production.
9|P|Generate each F arm's actual observations/full candidates and own train normalizer/3H1,unless exact fit equality proven|fcore,fchoice|Core-only scheduling proof does not complete selection/union/training/evaluation.
9|E|Interpret F as policy bundle1-4 with downstream B/I summaries remaining,not removal of all B/I information|fpolicy|Virtual native corner is not independently certified physical material junction.
9|P|Stop after finite registered result even if BI loses; no extra seed/threshold/preset tournament|plan,frule|Audit final attempts and report null/opposite results without equivalence claim.
10|E|H1 seeds18/19/20,representative18,188->64->64->1,16321 parameters|headlock,headtest,s1train|Own initialization repetitions,not3new observation pipelines.
10|E|AdamW.001/decay.0001,batch32frames,max50epochs,patience5,own VALloss minimum checkpoint|headlock,s1train|CPU1;768train/235VAL;normalizer fits eligibleTRAIN only.
10|E|Global finite budget<=36fits;exact reused fits not counted as new;do not reapply obsolete18fit limit|fits,plan|Only3 S1 fit reservations exist at snapshot;future attempts/technical resumes must stay in ledger.
10|E|Stage budget S1<=6,S2<=9,S3<=6,S4<=9,F<=6 and exact overlap may share|plan|33if C0 reusable/36if matchedC0 required;ceilings are not targets.
10|P|Consolidate covariance fits/PF forwards/render/obs forwards/optimizer/H1/importIO/eval costs separately|s1,materialpilot,s2progress|No campaign total yet;H1 fit timer is not full elapsedwall.
10|E|Preserve technical attempts/reasons;do not count resource interruption as scientific negative|s2quiesce,resource,checkpoint|Corrections preserve old artifacts and restart affected descendants honestly.
10|E|Avoid invented +4 scientific threshold;report allseed recovery/damage/continuous/tail tradeoffs|s1ind,s1paired|S1 successdelta -1/+1/-1 with some continuous/tail improvements,not uniformly better.
10|E|Null is not equivalence and positive development result is not independent generalization|plan,s1|independent_real_test_n=0.
11|E|No GT per-frame pose/box/mask/visibility/success/oracle in producer;metadata IDs/session/material not model inputs|s1pred,materialpilot,headlock|Known model identity is an asset-selection metadata input,not H1 feature.
11|E|Synthetic GT labels only in registered heads.prepare;CAL only registered fit;real references after model/prediction seals|s1train,s1pred,s1eval,calprotocol|Draft reads no raw numerical reference file and runs no scorer.
11|E|Paired full192=49+143,CAL46 separate,one returned pose T<=5cm AND R<=5degrees|s1eval,s1ind|Future stages require same denominator/no unfinished row zero-fill.
11|E|Include algorithm no-pose/fallback;available-only continuous denominator explicit;no trimming or zero errors|s1ind|Unexecuted rows excluded from COMPLETE rather than assigned failures.
11|E|RealC2 only,mean/sampleSD(ddof1)/median/P90/P95/max andT>50/100,R>45 tails|s1ind|Independent quaternion/atan2 expression verifies canonical metrics.
11|E|All3seed paired recovery/damage/bothsuccess/bothfailure,deltaT/R,fixed31/41/38 andgenerated/eligible/selected|s1paired,s1cohorts|Never subtract changing failure cohorts as paired samples.
11|E|Reusable post-bank analysis independently matches completed S1 coverage/fixedcohort outputs|postbankparity|52checks passed across4populations/5cohorts/all3seeds;zero newsolves/fits.
11|E|Six recording clusters,10000 paired cluster bootstrap,named sessions,leave-one-cluster-out;no576 independent frames|s1paired|Development sensitivity analysis only;all later comparisons still pending.
11|N|Independent new real confirmation|plan,s1|independent_real_test_n=0;blocks independent confirmation only,not this authorized development campaign.
11|E|S1 gallery contains all recovery/damage plusdeterministic normal/largeerror/candidateabsent/selectionfail|s1gallery,s1privacy|13derived cases;all4unique transition IDs;reference and observation layers separated.
11|P|S2/S3/S4/F galleries match every recovery/damage ID and include q,cov/normal,source/family/cost/admission/H1/refmetrics|materialfinish,fchoice|Await sealed predictions/evaluation;no cherry-picked positive gallery.
11|E|Respect rights/privacy,redacted derived panels only;display transform never changes production coordinates|s1privacy,s1gallery,plan|No public rawRGB/caches/weights/references;final public artifact scan pending.
12|P|Publish execution SHA/changecommits/baseline preservation,input/env/code/model/calibrator bindings|start,p0|Draft has evidence hashes;local code commit/publication proof pending.
12|P|Consolidate stage questions/priorattempt differences/groups/budgets/status and cache/new/reuse/interrupted ledger|checkpoint,materiallock|This matrix/method ledger are draft scaffolding,not final result report.
12|P|Consolidate q/Sigma/Gamma/source/partial changes and full candidateadmission/features/labels/normalizer/checkpoints|dgamma,calpackets,s1audit|S2 correction and S3/S4/F full seals needed.
12|P|Consolidate full row errors/pairedmetrics/upperbounds/selectedresults/galleries|s1eval,s1cohorts|S1 complete;remaining stages pending.
12|P|Record actual argv/environment/resume for every executed phase and all technicalattempt costs|s1argv,materialargv|Full campaign reproduce commands and cost accounting incomplete.
12|P|Create local code commit/reproducible Korean finalreport within authorized publication scope|plan,start|No commit/push performed by this task;final allowed public file and git checks pending.
12|P|Final answer states completed/reused/diagnostic/incomplete,whatwasomitted,benefits/costs,BIeffect,causallimits,blockers,totalcompute|checkpoint|Answer all eight original questions after remaining registered stages.
'''


METHODS = [
 ('BASELINE','Original observations and e hierarchy','Frozen original D1 observations/full bank/H1; P1prior1,WD0prior0,G10 legacy outline and separately typed G13. E0 reads G10-selected G13 family.','Exact reused for S1; baseline scoring is reaggregation, not a new solve.','d1_producer.py:solve_e','s1audit'),
 ('S1','PF completion versus PF coordinate replacement','Original D1 bank gains two actual PF+SubPix completed SE3 actions; original geometry/observation packet is unchanged.','Not an independent B/I-policy intervention;realized119/118/120 versus120/117/121.','s1_union.py:union_frame','s1'),
 ('S1','Admission and dependent features','Proper finiteR/allnative positivedepth/produceravailability admits independent actions even if old B/I feasibility disagrees. Old baseline eligibility remains unchanged.','Old26scalars/31masks bitwise reused;5gaps/full188 summaries/ownTRAINnormalizer/H1 regenerated.','s1_union.py:union_frame','s1audit'),
 ('D','Fixed logical graph versus physical source identity','Strict source-compatible subset103;135 rows change physical source association. Fixed-COMMON counterfactual preserves logical factorIDs and partials, records q/cov donor sources.','Algorithmic sensitivity conditional on graph;not actual BI on all238,not original120 full method.','qcov.py:fixed_common_packet','dfixed'),
 ('D','Covariance donor also changes Gamma','Original COVARIANCE_FIELDS transports covariance_px2,fit_covariance_px2,fit_reliability_weight;19factors/18frames differ. Corrected diagnostic separately fixes COMMON Gamma acrossall4cells.','Corrective238run/64optimizercalls/72dispatches/916reuses andpostsealALL238comparisoncomplete;legacySigma+Gammaresultretained;physicalsourcecaveatremains.','qcov.py:fixed_common_packet','dgammasupp'),
 ('S2','Calibration target and degrees of freedom','Own COMMON20261018 derived corner CAL46 residuals;80%zero-centeredellipse,one pooled additive floor or two shrunk rolefloors.','Not physical owner certification. CP preferred to CB by existing LOGO summary;unitweight correction pending.','calibration.py:fit','calmodel'),
 ('S2','Reliability correction','Old C0 preserved fit_reliability_weight whileCP/CBset1;new common-unit packetadapter isregistered andcorrected2workerproduceractive.','Old135armrows/37820calls/0unknowninterruptionspreserved,nooldH1fits;finalcorrectedbanks/training/evaluationpending.','calibration.py:apply_selection','s2stop'),
 ('FULLNORMAL','New backend versus original E','LOCAL COMMON q/cov and typed normal-only factors replace original D1 observation distribution. New matched hierarchy uses original numeric caps and original scalar primitives,not exact old computational trace.','MATCHED_BI/MATCHED_COMMON only; no original120 parity.','fullnormal_backend.py:configuration','s2lock'),
 ('FULLNORMAL','Effective covariance and role loss weights','covariance_scale1,zero additional B/I total-errorfloor,state lossweights1;full-point effective covariance comes from supplied packet.','Different covariance regime from original historical calibrated floor;same regime within matched arms.','fullnormal_backend.py:configuration','calprotocol'),
 ('FULLNORMAL','Revoked archived2D seed removal','Archived full points at partial IDs are excluded from P3P/SQPNP,full-point seed objective,seed constraints,rank and geometricguards. Original typed parser still authenticates archived q/cov.','Original typed backend retained raw full-BI seeds/guards;removal is an explicit method change,not a proved old bug.','fullnormal_backend.py:_complete','s2lock'),
 ('FULLNORMAL','No-full-point initialization','If no complete point remains,the registeredN0center and finite proper rotation list replace undefined fullpoint-span depth initialization;no archived q used.','Not deletion of all global search;original globalrotation cap retained.','fullnormal_backend.py:_initial_poses','s2lock'),
 ('FULLNORMAL','Normal-only constraints','Retains original normal likelihood,OT union andU ellipse constraints,but drops partial tangent support intervals and revoked-point canvas/protection guards.','Intervals remain immutable audit/display archives;normal-only feasible set intentionally differs.','fullnormal_backend.py:_joint_constraint_pair','s2lock'),
 ('FULLNORMAL','Full finite search,not LOCAL short solver','SLSQP finite initializations/branches with original maxseed160/joint250;P1prior1 andWD0prior0 retained;all endpoints journaled.','Limited D backend must not be described as the same full candidateproducer.','fullnormal_backend.py:solve_boundary_u_region_joint_ot_pnp','s2lock'),
 ('FULLNORMAL','Rank and ambiguity readout','New _finish computes weighted completepoint+normal image Jacobian rank. Original typed backend also exposed active joint-constraint rank and alternative-family ambiguity.','Status semantics differ and need disclosure;success remains canonical returnedpose metric.','fullnormal_backend.py:_finish','s2lock'),
 ('FULLNORMAL','G10/G13 exact collapse in new graph','G10 already consumes the same typed normal-only graph/outline/P1WD0starts asG13,so G13 is an exact G10alias.','Original G10legacy andG13typed stages are not generally aliases;G10-family readout remains explicit.','fullnormal_batch.py:solve_frame','s2lock'),
 ('FULLNORMAL','Feature energy availability','compile_models uses bundle=None to avoid querying revoked oldprofile points;B/edge energy+gap indices2,4,28,30 and masks are0;ABreducesA.','Same62shape/188H1math,but oldprofile evidence is absent. Zero is missing value plusvalidity0,not zero-cost certainty.','fullnormal_batch.py:feature_frame','adapter'),
 ('FULLNORMAL','Other profile-derived feature values','Without a boundary bundle,B_supported_n/B_profile_n/B_consumed_fraction are defined zero counts/fraction with valid masks;B_entropy_mean is missing. AB ownerentropy equals A ownerentropy. Fixed/A/AB costs and all geometry/source/covariance summaries use the new ownpacket.','Only four energy/gap indices are forcibly masked;do not say all profile-derived channels are invalid or old26scalars match originalD1. Bank minima span all candidates/families;H1 pooling uses eligible actions.','fullnormal_batch.py:feature_frame','s2lock'),
 ('FULLNORMAL','Own packet costs for final shared actions','kernel_extra scores added PF/render physical poses on actual fullnormal packet;reuses old26/masks only after source/config/byte proof or fresh numericalaudit.','Activefixed/A/AB gaps refreshed;B/edge remain invalid;eacharm ownbankpooling/normalizer.','kernel_extra.py:union_frame','adapter'),
 ('S3S4','Original downstream cost contexts for auxiliary comparisons','extra_joint takes sealed S1D1+PF bank and originalD1packet/bundle. S3 new support changes generatedposes;producer support archive remains separate.','Both auxiliary COMMON/BI use identical old downstreamfeatures. ExactR/t costmemoization doesnot pool arms ormerge symmetries.','extra_joint.py:union_frames','adapter'),
 ('S3','Asset identity/proxy','Existing texturedscanOBJ/USD meshes registered by bboxcenter/fixedproper axisrank/known dimension scaling;missingGLBs usefixedstructuredwoodproxy.','ExactphysicalCAD/signedaxes/ownerlabels unverified. This is structuredproxy performance,not physical ownership accuracy.','material_assets.py','asset'),
 ('S3','Actual generic3D supports','Welded source-mesh crease/boundaryedges excludecoplanar triangulation;supportIDs/objectXYZ are independent of8nativecuboid corners.','Currentpose zbuffer visibility is automatichypothesis,notGTvisibility.','material_support.py','materialinfo'),
 ('S3','COMMON/BI actual adoption difference','Same RGB/mask evidence,proxyvisibility andsearch;COMMON minimum normalizedcost withRGBtie,BI role-specificcuepriority;two/one/noneindependentdirections determine2D/1D/none.','No role-forced observation dimension;physicalsupport remains uncertified.','material_support.py:_policy','materialinfo'),
 ('S3','Source correlation and likelihood','Junction replaces underlying1Dfactors;rawpixel sourceIDs consumedonce;nonoverlapping originaltyped evidence retained;unknowncorrelation explicit.','No reinterpretation ofglobalownerpermutations asphysical materialparts.','material_support.py','materialinfo'),
 ('S4_PL','Finite extra refinement control','Original native fullpoints/typednormals/cuboidoutline refined with genericGaussian/uniformnone residualadapter and <=10 internal evaluations/update.','New objectiveimplementation;not exact originalhierarchical E solver reuse.','render_refine.py','materialinfo'),
 ('S4_CONTOUR','Structured proxy silhouette','Visible registeredproxy silhouette/depth matches automatictargetmask,rerendered object3D-to-boundarynormal factors.','Different from cuboidoutline control;not entireappearance matching.','render_refine.py','materialinfo'),
 ('S4_IMAGE','Whole grayscale render-depth pose update','Nativegrayscale render/observation,bidirectional OpenCV4.9DISFAST,currentmetricdepth inversepose3Dsupport,RefineLM andrerender.','Actual posegeneration,not RGBbank rescoring;not allRGBchannels,DeepIM/SRT3D or temporaltracking reproduction.','render_refine.py','materialinfo'),
 ('S4','Shared start/update budget','Same <=2 distinctphysicalN0/E0/PFpriority starts and <=4updates with10inner evaluations/refineLMiterations.','WDbasis aliases are not second physicalstarts;equal outerbudget doesnot mean equalwall.','render_refine.py','materiallock'),
 ('RENDER','Software implementation truth','EGLdevice1 isMesa llvmpipeLLVM15 CPUsoftware rasterizer,notCUDA/hardwareGPU. TexturedRGB plusnonMSAA visibledepthpass.','Rendererworker cost must be counted asCPU;hardware GPUrenderjobs0.','material_assets.py','materiallock'),
 ('RENDER','Pixel-centre metric depth','Rendererprincipalpoint+.5 convertsOpenGLindex+.5 centres tostoredinteger pixelcentres;inputKunchanged;pose-derivednear/far;nonMSAAdepth.','Authored slantedplane/project-unproject tests verify metricray convention;MSAA depthresolve alone notused.','material_assets.py','materialtests'),
 ('RESOURCE','LP4 amendment','LP_NUM_THREADS1pilot thenmeasuredLP4 with12case RGB/depth/mask strictbyteparity;one rendererprocess alongside2solvers.','Four internalsoftwaregraphics workers are explicit exceptionwithin resourceamendment;BLAS1/datasetworkers3unchanged.','render_thread_parity.py','resource'),
 ('F','Full observation intervention','Bothcue typesand bothtight/full masks generatedatalllegalnative rays;COMMON sharedqualityrank,BIpriority;actualindependent2D/1D/fallbackq,not conditionbitonly.','Nativevirtual corners stillnotcertified materialjunctions;no-new-support retainsN0withdeclarednoise.','f_observer.py','fpolicy'),
 ('F','Covariance/downstream scope','Actual1241ownqCALpacketmanifestcomplete;eacharm ownqCALfit,samepooledrecipe;downstreamrole summaries/costs remainboth.','Policybundle1-4with4tied;notallB/Iinformation removed. Corebanks/finalchoice/H1/evaluationpending.','f_calibration.py','fcalpackets'),
 ('F','Selection and early computation','VALmedian3seed rule,tiesloss/wall plusmandated allnegativePRIMARYdevelopmentfallback;unconditionalcoremayprecomputeonly independentinputs.','Clarification afterS1scoring isexplicit. NoFsystemchoice,fit orindependenttest claim yet.','f_batch.py','frule'),
 ('H1','Honest new campaign training adapter','Original action_data/models/evaluate math with owntrainnormalizer/ownlabels,global36attemptledger anddurableresume;oldcampaign18limit/receipts unused.','No extra modelarchitecture claim;resumedoptimizerstepscountedaslogicalsteps,not new hypothesis.','heads.py','headlock'),
 ('R','Independent metrics and evidence limits','S1 quaternionHamilton/atan2geodesic versuscanonicaltrace/acos,fsumstats/bootstrap;allrows/classes verified afterseals.','PRIMARY remains development;independent_real_test_n0;postseal outputdoesnot createunseentest.','independent_check.py','s1ind')]


def build(run,repo,instruction,public):
    run,repo,instruction,public=map(Path,(run,repo,instruction,public))
    lines=instruction.read_text().splitlines();assert len(lines)==839
    plan=read(run/'CAMPAIGN_PLAN.json');assert sha(instruction)==plan['instruction_sha256']
    assert sha(run/'CAMPAIGN_PLAN.json')=='e5b965e312fc96e6ef245d868285be2bf48e7ba4ee5a7dc5a612368d9485c023'
    sections={}
    for i,line in enumerate(lines,1):
        if re.match(r'^\d+\. ',line) and int(line.split('.')[0])<=13:
            number=int(line.split('.')[0])
            # Numbered lists within a section have much shorter titles.
            if number==0 or any(marker in line for marker in ('목표·용어','P0 —','빠른 경로','S1 —','D_QCOV —','S2 —','S3 —','S4 —','F —','유한한 학습','참조 분리','완료 산출물','검토할 실제')):
                sections[number]=dict(title=line,start_line=i)
    for number,section in sections.items():section['end_line']=sections.get(number+1,dict(start_line=840))['start_line']-1
    evidence={}
    for key,name in EVIDENCE.items():
        path=run/name
        evidence[key]=dict(path='PRIVATE_RUN/'+name,exists=path.exists())
        if path.exists():evidence[key].update(sha256=sha(path),bytes=path.stat().st_size)
    original=Path(read(run/'START.json')['original_repo']);start=read(run/'START.json')
    status=git(original,'status','--porcelain=v1','-z',raw=True)
    index=Path(git(original,'rev-parse','--git-path','index'))
    if not index.is_absolute():index=original/index
    preservation=dict(original_HEAD=git(original,'rev-parse','HEAD'),captured_original_HEAD=start['head'],
        dirty_status_sha256=hashlib.sha256(status).hexdigest(),captured_dirty_status_sha256=start['status_sha256'],
        original_index_sha256=sha(index),captured_original_index_sha256=start['index_sha256'],
        original_protected={name:sha(original/name)==digest for name,digest in start['protected'].items()},
        isolated_worktree_HEAD=git(repo,'rev-parse','HEAD'),isolated_origin_main=git(repo,'rev-parse','origin/main'),
        local_commit_or_push_performed_by_this_task=False)
    preservation['original_HEAD_unchanged']=preservation['original_HEAD']==start['head']
    preservation['original_dirty_status_unchanged']=preservation['dirty_status_sha256']==start['status_sha256']
    preservation['original_index_bitwise_unchanged']=preservation['original_index_sha256']==start['index_sha256']
    requirements=[]
    for text in REQUIREMENTS.strip().splitlines():
        section,status,requirement,keys,next_action=text.split('|',4);number=int(section)
        requirements.append(dict(id=f'REQ-{len(requirements)+1:03d}',source_section=number,
            source_lines=[sections[number]['start_line'],sections[number]['end_line']],requirement=requirement,
            status={'E':'EVIDENCED_FOR_STATED_SCOPE','P':'PENDING','N':'NOT_APPLICABLE_WITH_REASON'}[status],
            evidence=keys.split(','),scope_or_next_action=next_action))
    for item in requirements:
        assert all(k in evidence for k in item['evidence'])
        if item['status']=='EVIDENCED_FOR_STATED_SCOPE' and not any(evidence[k]['exists'] for k in item['evidence']):
            raise ValueError('Claim without an existing evidence artifact: '+item['id'])
    fits=[json.loads(line)for line in (run/'GLOBAL_H1_FIT_ATTEMPTS.jsonl').read_text().splitlines()if line.strip()]
    s1=read(run/'S1_EXECUTION_SUMMARY.json');gamma=read(run/'D_GAMMA_AUDIT.json')
    matrix=dict(schema='PALLET_FULL_REQUIREMENTS_TRACEABILITY_DRAFT_V1',draft=True,final_campaign_complete=False,
        snapshot_UTC=datetime.now(timezone.utc).isoformat(),instruction=dict(path=instruction.name,sha256=sha(instruction),lines=len(lines),all_lines_read=True),
        campaign_plan_sha256=sha(run/'CAMPAIGN_PLAN.json'),execution_base_sha=plan['base_sha'],
        evidence_labels=['[확인]','[추정]','[추정·미검증]'],status_semantics='EVIDENCED applies only to the exact stated completed scope; plans/code/pilots do not establish final effectiveness.',
        instruction_sections=sections,requirements=requirements,status_counts=dict(Counter(r['status']for r in requirements)),evidence=evidence,
        preservation_snapshot=preservation,populations=plan['populations'],
        scientific_production_scoring_training_calls_by_this_task=0,numerical_reference_files_opened_by_this_task=0,
        draft_generator_source_sha256=sha(__file__),
        H1_fit_attempts_at_snapshot=len(fits),H1_fits_by_arm=dict(Counter(r['arm']for r in fits)),
        stage_snapshot={'P0':'COMPLETE_EXACT_REUSE','S1':'COMPLETE_NEW','D_QCOV':'DIAGNOSTIC_ONLY;fixedCOMMON-Gamma corrective238run/postsealALL238supplementcomplete',
            'S2':'INCOMPLETE;corrected S2_FULL_UNIT_WEIGHT active with2workers;old135armrows/37820calls/0unknownpreserved','S3':'INCOMPLETE;full producer active','S4':'INCOMPLETE;full producer active',
            'F':'INCOMPLETE;1241observations andownCALpacketscomplete;corebanks/finalchoice/H1/evaluationpending','R':'INCOMPLETE;S1 and correctedD diagnostic evaluated,remaining comparisons pending'},
        confirmed_S1=dict(original_PRIMARY=[120,117,121],union_PRIMARY=[119,118,120],novel_good_candidates=3,
            new_good_cases_selected_success=[0,0,0],generated_eligible_upper_bound=[151,154],
            per_seed_recovery_damage=[[0,1],[1,0],[1,2]],three_seed_fit_internal_seconds=s1['phase_wall_seconds']['H1_three_fit_internal_sum'],
            instrumented_phase_sum_seconds=s1['instrumented_phase_wall_sum_seconds'],phase_sum_is_campaign_elapsed_wall=False),
        D_Gamma=dict(changed_factors=gamma['changed_factors'],changed_frames=gamma['changed_frames'],
            corrected_diagnostic_complete=read(run/'D_FIXED_GAMMA_EXECUTION.json')['complete'],
            corrected_summary=read(run/'D_FIXED_COMMON_GRAPH_FIXED_GAMMA/SUMMARY.json'),
            conclusion='Original donor is Sigma+Gamma;completed correction holdsCOMMONGammafixed. All238 supplemental comparison has0successchange;physical-source/logical-graph caveats remain.'),
        limits=['PRIMARY is previously used development data;independent_real_test_n=0.',
            'Exact physical CAD identity and independent human material-owner labels are not available;automatic proxy work continues.',
            'No public rawRGB,reference data,weights or bulk private caches are included in this draft.',
            'This snapshot may become stale while authorized workers proceed;refresh final statuses from terminal seals,not running row counts.'])
    ledger=[]
    for idx,(stage,difference,current,limit,callable_name,key)in enumerate(METHODS,1):
        module,*function=callable_name.split(':');directory=repo/'scripts/research/pallet_bi_full_execution_v1'
        if module=='d1_producer.py':directory=repo/'scripts/research/pallet_boundary_action_repair_v1'
        path=directory/module;line=None
        tree=ast.parse(path.read_text())
        if function:
            line=next((n.lineno for n in ast.walk(tree)if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))and n.name==function[0]),None)
        ledger.append(dict(id=f'METHOD-{idx:02d}',stage=stage,difference=difference,actual_method=current,
            interpretation_or_limit=limit,source=str(path.relative_to(repo)),callable=function[0]if function else'MODULE',line=line,
            source_sha256=sha(path),evidence=key,evidence_sha256=evidence[key].get('sha256'),status='SOURCE_CONFIRMED;execution scope from evidence'))
    matrix['method_difference_ledger_rows']=len(ledger)
    write(public/'REQUIREMENTS_MATRIX_DRAFT.json',jsontext(matrix));write(run/'REQUIREMENTS_MATRIX_DRAFT.json',jsontext(matrix))
    write(public/'METHOD_DIFFERENCE_LEDGER_DRAFT.json',jsontext(ledger))
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(ledger[0]));writer.writeheader();writer.writerows(ledger)
    write(public/'METHOD_DIFFERENCE_LEDGER_DRAFT.csv',stream.getvalue())
    gaps='''현재 미완료 항목 — 전체 캠페인 완료 보고서 아님

1. S2의 공통 단위 reliability 계약은 교정됐고 S2_FULL_UNIT_WEIGHT에서 2개 worker가 실행 중이다. 전체 C0/CP/CB 후보,9 H1 fit,238x3 예측 seal과 별도 채점은 아직 필요하다. 구버전135 arm행/37820호출/불명 중단0은 보존됐고 H1 fit은 시작하지 않았다.
2. 기존 D donor는 Sigma와 Gamma를 함께 옮겼다.19 factor/18 frame의 차이를 확인한 뒤 COMMON Gamma를 고정한238행 진단을 완료했다:64 optimizer호출,72 family dispatch,916재사용. ALL238 postseal 보충 비교는 Sigma1 두 군에서16 pose씩 바뀌었으나 성공 변화는0이다.103 source-compatible/135 source-changing과 논리 graph 조건은 계속 명시한다.
3. S3/S4 full receipt,EXTRAS hash,모든 row archive ancestry 검증,실사/합성 feature union,각자 normalizer/labels와15 H1 fit,출력 seal이 남아 있다.원31/41/38의 generated/eligible/selected 변화와 전량 paired 지표를 채운다.
4. 구조검토24는 선택만 확인됐다.물체/가림물체/배경/unknown,재료support/가상목표,두방향/한방향/없음,출처를 실제 derived review로 연결한다.독립 사람확인0이며 물리 owner정확도는NA다.
5. F의1241 관측과1241 own-q CAL packet manifest는 완료됐다. F_SYSTEM_CHOICE,core 후보,최종 optional PF/render proposal공유,두 군의 H1/예측/채점은 미완료다. 독립적인 frontend 선계산을 최종 F 완료로 바꾸지 않는다.정책1-3은 실제 관측 개입이고4는 같은 pooled recipe,후단 B/I 요약은 양쪽에 남는다.
6. S2/S3/S4/F의 모든 recovery/damage gallery,6cluster/10000bootstrap/session/leave-one-cluster-out,전체 cost/argv/resume 원장,로컬commit/최종공개점검이 남아 있다.
7. 원120 대비 새fullnormal 차이는 단순adapter가 아니다:revoked2D seed/guard 제거,normal-only partial,tangent제약제거,G10/G13collapse,B/edge feature마스크,AB=A,새rank/readout 의미를 명시한다.
8. renderer는EGL이지만Mesa llvmpipe CPUsoftware다.LP4 내부graphicsworkers,2개solverprocess와의 자원amendment 및12case byteparity를 보고한다.하드웨어GPU렌더링으로 적지 않는다.

현재 확정 S1:원120/117/121 ->union119/118/120;회복/손상0/1,1/0,1/2.상한151->154이나 새3사례는세seed모두선택실패.일부평균/꼬리개선과성공의tradeoff이며독립B/I기여증거가아니다.
independent_real_test_n=0.이미개발에사용한PRIMARY는새출력seal후에도독립test가아니다.원dirtyworktree/oldreceipt/immutableplans는이작업에서변경하지않았다.
'''
    write(public/'GAPS_HANDOFF_DRAFT.txt',gaps)
    scaffold='''팔레트 B/I 전체 캠페인 결과 보고서 골격 [DRAFT]

[확인] 실행 기준:3553202a3682faeb7be2e7df051b230683965382.실사238=PRIMARY192(TEST49+REMAINDER143)+CAL46.합성1003=train768+VAL235.대표seed20261018.독립새실사0.

[확인] P0와 S1만 전체 결과로 보고할 수 있다. S1 PRIMARY:원H1 120/117/121,union119/118/120.151->154는 generated/eligible 상한이고 실제성공증가가 아니다.다른수치는 REQUIREMENTS_MATRIX_DRAFT와 기존S1평가/독립검증receipt를 인용한다.

[확인] D는 제한 solver의 조건부 진단이다.103 source-compatible와135 source-changing을 구분한다. 기존 Sigma donor에 Gamma도 포함되어19 factor/18 frame이 달랐다. COMMON Gamma를 고정한238행 교정 진단과 ALL238 postseal 비교가 완료됐다.64 optimizer호출/72 family dispatch/916재사용이며, 교정 후 성공 수 변화는0이다. 물리 source 및 COMMON 논리 graph 조건은 그대로 남는다.

[확인] S2 CAL46의241 point(222B/19I),zero-centered80% ellipse와 pooled/shrunk role floor는 등록돼 적합됐다. 공통 단위 reliability 입력 교정 후2 worker의 새 전량 후보 생산이 진행 중이다. [추정·미검증] 완성된 candidate/own-H1/채점 결과를 여기에 추가한다. 보존된 구버전135 arm행/37820호출을 과학적 음성으로 표시하지 않는다.

[확인] S3/S4는등록된structuredproxy,자동zbuffer/원영상지지,동일physicalstarts의PL/CONTOUR/wholegrayscaleDIS실험이다.정확CAD/owner인증없음,renderer는llvmpipeCPU. [추정·미검증]1241완료seal,5군자기분포H1,후보상한/실현선택/연속오차/pairedcluster/gallery가들어갈자리다.

[추정·미검증] F 최종3행:ORIGINAL_120 / MATCHED_COMMON / MATCHED_BI.설정선택과PRIMARY개발fallback사실,두군actual관측/후보/학습/평가와정책1-4범위를여기에명시한다.원120직접parity나모든B/I정보제거를주장하지않는다.

[PENDING] 최종표:모든seed×192/49/143/CAL46의성공,T/Rmean/sampleSD/median/P90/P95/max,T>50/100,R>45,no-pose/fallback;paired회복/손상/공동성공/공동실패와deltaT/R;generated/eligible/selected;고정31/41/38;6cluster/bootstrap10000/session/leave-one-out.
[PENDING] 모든전환ID+정상/큰오류/후보부재/선택실패derivedgallery.원관측q/Σ/normal/source와참조투영은분리한다.원RGB/weights/privatecaches/refs공개없음.
[PENDING] 총elapsedwall과phase별forward/render/optimizer/H1/calibration/importIO/eval비용,exactreuse/중단기술시도,실제argv/env/resume,변경commit/publicationproof.

최종답변8개질문:완료상태/정확생략/기준대비이득손상/seed촬영군BI차이/원인연결/남은식별불가/독립검증과자산한계/총실행비용.현재본문은그답변의완료본이아니다.
'''
    write(public/'REPORT_SCAFFOLD_KO.txt',scaffold)
    return dict(complete_draft=True,requirements=len(requirements),method_differences=len(ledger),
        public_directory=str(public),matrix_sha256=sha(public/'REQUIREMENTS_MATRIX_DRAFT.json'),preservation=preservation)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('run','repo','instruction','public'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args();print(jsontext(build(args.run,args.repo,args.instruction,args.public)))
