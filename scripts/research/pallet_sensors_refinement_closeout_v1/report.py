"""Generate manuscript tables directly from locked measured artifacts."""
import numpy as np
from env import *
def textfile(path,text):
    assert any(path.resolve().is_relative_to(r) for r in (DOC,PAPER))
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text.strip()+'\n')
def run():
    verify();d=read(DOC/'UNIFIED_DEV_RESULTS.json');m=d['methods'];pr=read(DOC/'P_VS_R0_PAIRED.json');pd=read(DOC/'P_VS_D_PAIRED.json');rt=read(DOC/'RUNTIME_PANEL.json');train=read(DOC/'D_TRAINING_AUDIT.json');pop=read(DOC/'POPULATION_AND_METRIC_LOCK.json');secondary=read(DOC/'EXPLORATORY_PAIRED.json')
    means={}
    for arm in ('P','L','D'):
        means[arm]={k:float(np.mean([m[f'{arm}{s}'][k] for s in (1,2,3)])) for k in ('median_px','p90_px','frame_mean_px','gross20','corner8_median_px')}
        means[arm]['ALL_GT_PCK']={t:float(np.mean([m[f'{arm}{s}']['ALL_GT_PCK'][t] for s in (1,2,3)])) for t in ('5','10','20')}
        means[arm]['pose']={k:float(np.mean([m[f'{arm}{s}']['pose'][k] for s in (1,2,3)])) for k in ('rotation_median_deg','yaw_median_deg','translation_median_cm','iou3d_median','add_sym_auc','coverage')}
    write(DOC/'SEED_MEAN_RESULTS.json',means)
    P=means['P'];ref=m['R0'];effects=pr['session'];dd=pd['session']
    harm=[]
    if P['p90_px']>ref['p90_px']:harm.append('P90')
    if P['gross20']>ref['gross20']:harm.append('gross20')
    if P['ALL_GT_PCK']['10']<ref['ALL_GT_PCK']['10']:harm.append('ALL_GT_PCK10')
    for key in ('rotation_median_deg','yaw_median_deg','translation_median_cm','iou3d_median','add_sym_auc','coverage'):
        delta=P['pose'][key]-ref['pose'][key]
        if (delta>0 if key.endswith(('_deg','_cm')) else delta<0):harm.append(key)
    evidence='P_DEV_HARM_OR_TRADEOFF' if harm else 'P_DEV_SUPPORTED' if effects['high']<0 else 'P_DEV_UNRESOLVED'
    # The requested prior full training and new physical data are outside this run;
    # official-network adapter readiness/resource measurement remains a technical gap.
    status=dict(EXECUTION='PARTIAL',completed_scope=['fixed P/R0/L audit','D3 fits18000','D synthetic selection+heldout','DEV10 models','paired statistics','desktop runtime+PnP','data adapter tests','confirmation tools','manuscript draft'],
        incomplete_scope=['official PoseFix network integration and actual GPU resource estimate','formal prior performance comparison','independent real confirmation'],
        EVIDENCE=evidence,PAPER='NEEDS_BOTH',observed_P_R0_harm_metrics=harm,publication_acceptance_claim=False,independent_confirmation=False,
        P_R0=effects,P_D=dd,D_actual_updates=18000,smoke_updates=train['smoke_updates'],D_resumes=sum(r['resumes'] for r in train['runs']),
        runtime='DESKTOP_ONLY_MEASURED',prior=read(DOC/'PRIOR_BASELINE_READINESS.json')['status'],confirmation='NEW_CONFIRMATION_DATA_REQUIRED')
    write(DOC/'FINAL_STATUS.json',status);write(DOC/'CURRENT_RESULT.json',dict(**status,stage='MEASUREMENTS_COMPLETE_WITH_DOCUMENTED_PRIOR_AND_CONFIRMATION_GAPS'))
    numbers={name:bound(DOC/name) for name in ('UNIFIED_DEV_RESULTS.json','SEED_MEAN_RESULTS.json','P_VS_R0_PAIRED.json','P_VS_D_PAIRED.json','EXPLORATORY_PAIRED.json','RUNTIME_PANEL.json','POPULATION_AND_METRIC_LOCK.json','D_TRAINING_AUDIT.json','D_SELECTION.json')}
    write(PAPER/'NUMBER_SOURCES.json',dict(files=numbers,tables=dict(T1='METHOD_FREEZE.json,P/D_PROTOCOL_LOCK.json',T2='UNIFIED_DEV_RESULTS.methods.<model>;SEED_MEAN_RESULTS.<arm>',T3='P_VS_R0_PAIRED.session;P_VS_D_PAIRED.session',T4='RUNTIME_PANEL.summary.<model>',T5='CONFIRMATION_READINESS.status'),membership='POPULATION_AND_METRIC_LOCK.frames',GT_denominator=pop['total_supervised'],numeric_tables_generated=True))
    headers='| Model | Median px | P90 px | Frame mean px | Gross20 % | PCK5 % | PCK10 % | PCK20 % | Matched / GT points |\n|---|---:|---:|---:|---:|---:|---:|---:|---|\n'
    table=headers
    for name in ['R0']+[f'{a}{s}' for a in ('P','L','D') for s in (1,2,3)]:
        v=m[name];table+=f"| {name} | {v['median_px']:.4f} | {v['p90_px']:.4f} | {v['frame_mean_px']:.4f} | {v['gross20']*100:.3f} | {v['ALL_GT_PCK']['5']*100:.3f} | {v['ALL_GT_PCK']['10']*100:.3f} | {v['ALL_GT_PCK']['20']*100:.3f} | {v['matched_frames']}/{pop['positive']}; {v['supervised_points']}/{v['gt_denominator']} |\n"
    for a,v in means.items():table+=f"| {a} seed mean | {v['median_px']:.4f} | {v['p90_px']:.4f} | {v['frame_mean_px']:.4f} | {v['gross20']*100:.3f} | {v['ALL_GT_PCK']['5']*100:.3f} | {v['ALL_GT_PCK']['10']*100:.3f} | {v['ALL_GT_PCK']['20']*100:.3f} | same fixed population |\n"
    pose='| Model | Rotation deg | Yaw deg | Translation cm | IoU3D | ADDsym AUC | Pose coverage |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for name in m:
        v=m[name]['pose'];pose+=f"| {name} | {v['rotation_median_deg']:.4f} | {v['yaw_median_deg']:.4f} | {v['translation_median_cm']:.4f} | {v['iou3d_median']:.5f} | {v['add_sym_auc']:.5f} | {v['n']}/{pop['positive']} |\n"
    timing='| Model | Image→2D median/mean/P90 ms | +PnP median/mean/P90 ms | Paired added2D median ms | Refiner parameters | Single-model peak allocated MiB |\n|---|---|---|---:|---:|---:|\n'
    for name,v in rt['summary'].items():
        a=v['image_to_2d_ms'];b=v['end_to_end_ms'];timing+=f"| {name} | {a['median']:.3f}/{a['mean']:.3f}/{a['p90']:.3f} | {b['median']:.3f}/{b['mean']:.3f}/{b['p90']:.3f} | {v['paired_added_2d_ms']['median']:.3f} | {rt['params'][name]['refiner_trainable']} | {rt['isolated_memory'][name]['peak_allocated_bytes']/1048576:.2f} |\n"
    interval='| Comparison | Delta median px | Session95% interval px | Frame95% interval px |\n|---|---:|---|---|\n'
    for name,value in [('P−R0',pr),('P−D',pd)]:
        a=value['session'];b=value['frame'];interval+=f"| {name} | {a['delta']:.5f} | [{a['low']:.5f}, {a['high']:.5f}] | [{b['low']:.5f}, {b['high']:.5f}] |\n"
    translation_mm=(ref['pose']['translation_median_cm']-P['pose']['translation_median_cm'])*10
    interpretation='P has a lower development median than this fixed direct control.' if dd['high']<0 else 'The direct control has a lower development median than P.' if dd['low']>0 else 'The interval includes zero; neither equivalence nor a P-specific advantage is established.'
    tail=secondary['comparisons']['P-D']['keypoint_location_p90_px']
    tradeoff=f"P−D P90 difference is {tail['delta']:.3f}px (session95% [{tail['low']:.3f},{tail['high']:.3f}]); positive means P has the larger P90. The observed D runtime is lower; neither all-metric dominance nor equivalence is claimed."
    textfile(PAPER/'RESULTS_DRAFT.md',f'''# Development results — not independent confirmation

T2a. Original-pixel point errors. Pooled9-point median is computed per seed, then seed means; no ensemble. All-GT PCK uses{pop['total_supervised']} supervised points, including failures. Each raw model has its own row.

{table}
T2b. Canonical MAIN pose against geometry-reconstructed reference; no external6D metrology claim.

{pose}
T3. Shared paired draws across model seeds,13 observed sessions,10,000 resamples,seed20260914. Negative favors P. Primary P−R0 is the only designated primary; P−D and secondary outputs are exploratory. Intervals do not establish independent generalization.

{interval}
P−R0 translation median reduction averaged over seeds is{translation_mm:.3f}mm. This is a difference of geometry-reference error statistics, not directly measured insertion improvement and not a pixel-to-mm conversion. Per-session and leave-one-session-out results are in the experiment CSV/JSON. Tail and pose intervals are in EXPLORATORY_PAIRED.json.

{interpretation} {tradeoff} D's fixed train/cal probe curves and complete update traces must accompany that claim. No saturation or global convergence guarantee follows from6,000 updates. Shared evidence and+2.57% parameters do not match FLOPs, output support or supervision. D regresses unbounded residuals with L1; P uses candidate-distribution supervision and expectation.

T4. RTX3080 desktop.26 prespecified images,20 warmups/model,5 balanced order blocks, batch1 and4 CPU threads. Decoding/model loading excluded. All samples and allocated/incremental memory are retained. Display/RustDesk stayed active.

{timing}
Feature sampling is included, not a cached-feature deployment benchmark. End-to-end adds the canonical prediction-only selector and selected-pose solver. Embedded export/Jetson performance was not measured. No fastest repeat was selected.

T5. Independent confirmation: NOT_YET_MEASURED. Formal prior-method performance: NOT_YET_MEASURED. Neither missing value is represented by zero. See the separate collection and prior protocols.
''')
    textfile(PAPER/'METHOD_DRAFT.md','''# Method draft

A frozen YOLO26n pose detector provides boxes, scores, nine predicted points and P3/P4 neck features. A small point refiner P processes the selected detection only. It adapts the two feature scales and samples a4×8 patch at each of221 candidate displacements around each corner.13 directions and17 radii cover a maximum0.08 predicted-box diagonal. The stencil fraction is0.1310373991727829, determined from historical source prediction geometry. A shared scorer and null score produce a222-way distribution per corner. The expected candidate displacement is scaled by frozen lambda and capped using0.01 original-image diagonal, transformed by the letterbox gain. Original coordinates are updated by delta/gain. Center8, nonselected detections, boxes and scores are preserved.

P contains18,962 trainable parameters. Its source-only soft target distribution and training/selection rules are historical and fixed; no new P fit is performed. Expectation readout is established prior art (Integral Regression and PoseFix), not new mathematics. The module depends on this detector's internal feature interface and is not demonstrated to be model-agnostic.

D uses the same initialized shared adapters, patch encoding, role context and candidate sampling, then a descriptor encoder64→40, mean/max set pooling and93→64→2 residual head. It contains19,450 trainable parameters. All parameters are fresh; none come from a trained P. Output residuals are bbox-diagonal normalized and trained with per-frame valid-point L1. Raw D predictions are unbounded, unlike P's candidate convex hull. The same final lambda/cap grid is allowed; D has no temperature. This compares readout/supervision packages, not softmax alone, and does not reproduce PoseFix.

For P, let d_b be the predicted box diagonal, v_j the fixed normalized displacement bank including v_null=0, and z_kj the corner scores. With frozen temperature T, the raw movement is Δ_k=λ d_b Σ_j softmax(z_k/T)_j v_j. If cap c is active, use Δ_k min(1,c/||Δ_k||). Here c is original-image diagonal times the frozen fraction and letterbox gain g. Restoration adds Δ_k/g once to the original corner. For D, replace d_b Σ probability×v_j by d_b times the learned normalized residual; retain the same restoration contract. These formulas specify the tested computation, not a new mathematical estimator.

Known physical dimensions and camera intrinsics are inputs to the unchanged downstream prediction-only PnP selector, not P/D neural inputs. Canonical MAIN uses the selected camera-facing cuboid and SQPnP/RefineLM. The current geometry reference is derived from manual points and known dimensions, so it is not independently instrumented ground truth.
''')
    textfile(PAPER/'DATA_EVALUATION_DRAFT.md',f'''# Data and evaluation draft

Source partitions: {pop['synthetic_partitions']}. Matched usable training rows:{pop['synthetic_matched_train']}. The new D fits use exactly the frozen P seed-specific order,3×6,000 actual updates,batch16 and96,000 nominal exposures per seed. AdamW LR0.001,betas(0.9,0.999),WD0.0001,100-step warmup,cosine final fraction0.1,clip5. No real image/feature/label or negative data enter the optimizer. The frozen detector itself and prior research have their own training/selection histories; no claim of no real annotation anywhere is made.

D selection uses the old synthetic1031-frame partition and all10 lambda/cap rules, averaging the old normalized capped8-corner full-GT frame score over seeds; ties prefer smaller lambda then stricter cap. Calibration is only a diagnostic probe for D; there is no fake temperature sweep. The1985-frame synthetic heldout is evaluated after selection. These are historically reused synthetic datasets, not universally untouched data.

Development: {pop['positive']} positives in{len(pop['sessions'])} sessions,{pop['negative']} negatives. Precision has{pop['matched']} matched frames and{pop['matched_supervised']} supervised points; all-GT PCK has{pop['total_supervised']} points. No keypoint confidence threshold alters the existing supervision contract. Negative session metadata is unavailable ({pop['negative_session_metadata']} populated records). Actual D negative inference checks candidate preservation, while negative points are not supervised localization endpoints.

Pose coverage follows the canonical pose evaluator, not the2D IoU matching gate: it uses the top prediction with sufficient finite corners and successful selector/solvers. Thus319 available poses do not imply319 correctly matched2D detections. Missing/nonfinite/unsolved pose IDs and matched2D denominators are reported separately. The primary conditional residual cannot replace complete-failure-aware PCK.

Full-precision predictions and canonical targets are used for residuals. CSV rounding differences are separately checked. R0 is a single baseline in each shared bootstrap draw; each seed's pooled statistic is computed before averaging. Session clusters are the primary uncertainty description and frame bootstrap is secondary.10,000 draws,seed20260914. Session/seed count does not guarantee out-of-session performance. No noninferiority margin or operational tolerance was invented.

Formal prior comparison and independent confirmation remain unfinished. No present DEV result is renamed as final test evidence. See NUMBER_SOURCES.json for every numerical table binding.
''')
    claims=f'''# Claim–evidence matrix

| Claim | Evidence | Estimand / N / units | Uncertainty | Novelty relation | Remaining check | Permissible manuscript wording |
|---|---|---|---|---|---|---|
| P lowers developed2D residual | P_VS_R0_PAIRED + full-precision predictions | mean seed pooled median difference,{pop['matched']} matched/{pop['positive']} total frames,{pop['matched_supervised']} points,px | {effects['delta']:.4f} [{effects['low']:.4f},{effects['high']:.4f}] session95% | empirical signal, not new expectation | independent sessions/QA | On reused development data the fixed refiner reduced pooled median error. |
| Detector outputs are preserved | original P inference audits; D actual3008-image checks; runtime parity | boxes/scores/order/top1/center | exact tested identity | implementation contract | portability beyond fixed R0 | The tested wrapper preserves the detector outputs. |
| Same-evidence direct control | D_PROTOCOL_LOCK,tests,D3 checkpoints |19,450 vs18,962 params;8×221×32×2 samples | P−D {dd['delta']:.4f} [{dd['low']:.4f},{dd['high']:.4f}] | bounded readout/supervision comparison | convergence; formal prior | {interpretation} |
| Geometry change | canonical MAIN per-frame outputs | translation median reduction{translation_mm:.3f}mm;319 poses per existing model | exploratory intervals separate | sensing evaluation | independent metrology | Geometry-reference errors changed by the reported amount. |
| Runtime | RUNTIME_PANEL | desktop actual images,26×5/model | sample median/mean/P90; no device-general CI | system cost | Jetson/export if required | Desktop overhead was measured under the stated conditions. |
| Formal prior superiority | none | NOT_YET_MEASURED | none | not established | PoseFix actual trained comparator | No claim currently permitted. |
| Independent generalization | none | NEW_CONFIRMATION_DATA_REQUIRED | none | not established | new captures and blinded labels | No independent confirmation has been conducted. |

Avoid SOTA, first-ever, arbitrary pallet, unseen-instance, model-agnostic, all-metric superiority, Jetson real-time and forklift safety. Related-method links and limits are in PRIOR_ART_MATRIX.md. No journal acceptance inference is made.
'''
    textfile(PAPER/'CLAIM_EVIDENCE_MATRIX.md',claims)
    textfile(PAPER/'ABSTRACT_DRAFT.md',f'''# Working abstract — development evidence only

We study synthetic-supervised local keypoint refinement for monocular pallet pose estimation using a fixed RGB estimator. A18,962-parameter module reuses detector features while preserving boxes, scores, instance selection and the center point. On319 reused development images, the three-seed mean pooled supervised keypoint median decreased by{-effects['delta']:.3f}px (paired-session95% interval for the decrease:[{-effects['high']:.3f},{-effects['low']:.3f}]px). Evaluation also reports complete-denominator PCK, geometry-reference pose errors and desktop runtime. A separately trained same-evidence direct residual control tests the readout/supervision design under the same source exposure budget. These observations are development evidence; a formal prior-method performance comparison and untouched real-session confirmation remain required. The current work does not establish a new expectation operator, state-of-the-art performance or operational handling reliability.
''')
    textfile(PAPER/'T1_RESOURCES.md','''# T1. Inputs and resources

| Model | Image evidence | New training in this run | Added parameters | Output / deployment assumptions |
|---|---|---|---:|---|
| R0 | RGB, fixed detector |0 |0 |9points, known-size PnP |
| P | existing P3/P4, predicted box/points |0; frozen3 historical6000-step heads |18,962 |candidate distribution→8corner movement; center fixed |
| L | same feature scales, predicted edges |0; frozen3 historical heads |19,810 |line readout/correction; original structural control |
| D |same candidate spatial evidence and context as P |3×6000 source-only updates |19,450 |direct residual L1, same final cap grid |
| PoseFix adaptation |separate RGB crop backbone+9pose maps |0 external full updates |NOT_YET_MEASURED |adapter tested; network integration not complete |

Total detector and allocated memory details are in RUNTIME_PANEL.json. Same sample counts do not imply equal runtime or FLOPs. ImageNet initialization for eventual PoseFix would be additional supervision and must be disclosed.
''')
    textfile(PAPER/'FIGURE_SPECIFICATIONS.md','''# Figure specifications

F1. Fixed R0→P3/P4 sampling→candidate scoring/null→expected displacement→original coordinates→PnP. Caption must label dimensions as PnP inputs only and identify unchanged detector outputs. No novelty label on expectation.

F2. All-session P−R0 and P−D median effects, plus leave-one-session-out; draw directly from paired JSON/CSV. Caption identifies reused DEV and conditional uncertainty, no favorable-session filtering.

F3. D fixed train/cal probe curves for all3 seeds, rendered from D_TRAINING_AUDIT. Caption distinguishes normalized L1 from P cross-entropy and does not assert convergence from loss-scale comparisons.

F4. Every runtime sample or distribution, including added latency and memory, from RUNTIME_PANEL. No fastest-run selection. Any example images must use a prespecified balanced sampling policy; no cherry-picked success montage is generated here.
''')
    textfile(PAPER/'REMAINING_WORK.md','''# Remaining work

1. Make the formal PoseFix pallet adaptation run in an isolated compatible environment; verify official-network loading, optimizer/loss semantics and coordinates, and measure actual GPU resources. Then separately authorize its full comparison budget. CRT-6D official executable source is unavailable at the recorded commit.
2. Complete formal prior-method performance comparison before confirmation unblinding. Do not use D as a PoseFix result.
3. Collect untouched sessions, independent blinded annotations and calibration/dimension metadata under CONFIRMATION_PROTOCOL. No such membership currently exists in the supplied manifests.
4. Assess annotation repeatability and the geometry-reference limitation. Physical6D metrology and handling trials need actual measurements and task requirements.
5. Review claims and journal fit with the advisor. This workspace is a draft for human review, not a submitted/final manuscript or a publication guarantee.
''')
    textfile(DOC/'REPORT_KO.md',f'''# 고정 P 원고 검증 결과

실행 상태 {status['EXECUTION']}; 개발 근거 {evidence}; 논문 상태 NEEDS_BOTH.

새 본 학습은 D만3seed×6000=18000 updates, disposable smoke{train['smoke_updates']}회, 재개{status['D_resumes']}회입니다. R0/P/L 재학습은0입니다. 모든 D 최종 checkpoint를 사용했고 성능 기반 재시도는 없습니다.

R0 pooled9 median {ref['median_px']:.4f}px → P seed평균 {P['median_px']:.4f}px. 차이 {effects['delta']:.4f}px, 세션95% [{effects['low']:.4f},{effects['high']:.4f}]. 같은311 matched frames/2756 points이며 전체GT PCK 분모는2818점입니다. R0 PCK10 {ref['ALL_GT_PCK']['10']*100:.3f}% → P {P['ALL_GT_PCK']['10']*100:.3f}%.13개 재사용 개발 세션의 관측입니다.

P−D median 차이 {dd['delta']:.4f}px, 세션95% [{dd['low']:.4f},{dd['high']:.4f}]. {interpretation} 같은 샘플링/optimizer/노출이지만 loss·출력범위·readout이 달라 softmax 하나의 인과분해가 아닙니다. D 고정train/cal probe는 결과와 함께 공개하며 일반 회귀 방식의 원리적 열세를 주장하지 않습니다.

P−D P90은{tail['delta']:.3f}px로 P가 관측상 약간 크며, 세션95% [{tail['low']:.3f},{tail['high']:.3f}]는0을 포함합니다. D가 런타임도 더 낮아 P의 모든 지표 우세라는 결론은 내리지 않습니다.

P의 geometry-reference translation median 감소는 seed평균{translation_mm:.3f}mm입니다. 독립 장치의 실측6D GT나 실제 삽입 성공 개선이 아닙니다. 모든2D/6D/시간 수치는 새 원고 RESULTS_DRAFT.md에 자동 연결했습니다.

GPU는 실제 RTX3080. 같은26영상×5repeat, 모델별20warmup, 균형순서로 BGR→2D와 PnP 포함 시간을 측정했습니다. Display/RustDesk 유지; Jetson 미측정. 기존 timing은 보존했습니다.

PoseFix 공식 소스/라이선스를 확인하고 실제 합성train8장 adapter를 검사했습니다. 공식 네트워크의TF1환경·9점연결·GPU자원 실측은 미완료이며 formal prior 성능 비교는 하지 않았습니다. CRT-6D 공식 저장소에는 실행 코드가 없습니다. 독립FINAL manifest는 모두 UNAVAILABLE이며 새 촬영·사람의 blinded annotation/QA가 필요합니다. 이 두 항목 때문에 PAPER_READY라고 하지 않습니다.

원고에 쓸 수 있는 범위: 고정 estimator의 특징을 재사용하는 소형 정제기, 검사된 wrapper 보존 계약, 현재DEV의 대응 정확도 변화와 실제 desktop 비용. 아직 쓸 수 없는 범위: SOTA/최초/독립일반화/정식선행우월/Jetson실시간/지게차안전보장.

시작 main {read(DOC/'SOURCE_BINDING.json')['start_sha']}. 최종 commit/push SHA는 Git 게시 확인 메시지와 기록을 따릅니다. 기존 scientific verdict와 _docs/paper/final은 보존되었습니다.
''')
    print('REPORT_READY',status['EXECUTION'],evidence,status['PAPER'],flush=True)
if __name__=='__main__':run()
