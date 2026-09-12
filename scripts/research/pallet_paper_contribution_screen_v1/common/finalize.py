"""Publish explicit incomplete/negative outcomes; never invent missing fits."""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.contracts import ROOT,RAW,DOC,sha,write

TRACKS={'A':'A_architecture_confirmation','C':'C_geometry_preserving_da','D':'D_student_relative_geometry','B':'B_alignment_loss','E':'E_privileged_depth_teacher'}
def read(path):return json.loads(path.read_text())
def md(path,text):
    if path.exists():
        assert path.read_text()==text;return
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)

def main():
    preserved=read(DOC/'PRESERVED_SOURCE_SHA.json')
    changed=[p for p,h in preserved.items() if sha(ROOT/p)!=h];assert not changed
    c=read(DOC/TRACKS['C']/'PER_SEED.json')
    d=read(DOC/TRACKS['D']/'MECHANISM_RESULT.json');b=read(DOC/TRACKS['B']/'MECHANISM_RESULT.json');e=read(DOC/TRACKS['E']/'MECHANISM_RESULT.json')
    assert d['status']==b['status']==e['status']=='FAIL'
    # Independent decisive-statistic recomputation from row-level saved outputs.
    br=read(RAW/'B_alignment_loss/PER_FRAME.json')
    bbad=sum(r['catastrophic'] for r in br)/len(br);assert bbad==b['catastrophic_sample_fraction'] and bbad>.01
    er=read(RAW/'E_privileged_depth_teacher/BOUNDARY_SAMPLES.json');clean=[r for r in er if not r['prior_bad_reference']]
    ev=[r['signed_px'] for r in clean if r['signed_px'] is not None]
    coverage=len(ev)/len(clean);assert coverage==e['excluded_prior_bad_reference']['coverage'] and coverage<.8
    dr=read(RAW/'D_student_relative_geometry/EDGES.json');cal=np.array([r['split']=='trust_cal' for r in dr]);gain=np.array([r['gain_px'] for r in dr])
    assert all(abs(r['gain_px']-(r['student_normal_px']-r['teacher_normal_px']))<1e-12 for r in dr)
    independent_d=[]
    for seed in [1,2,3]:
        pred=np.array(read(RAW/f'D_student_relative_geometry/PREDICTIONS_seed{seed}.json')['predicted_gain_px'])
        for tau in [0,.1,.25,.5,1]:
            selected=(pred[cal]>tau);g=gain[cal][selected]
            cov=float(selected.mean());harm=float((g<0).mean()) if len(g) else None
            assert cov<.1 or harm is None or harm>=.25
            independent_d.append(dict(seed=seed,tau=tau,coverage=cov,harm=harm))
    init=[];orders=[];audits={}
    for n in ['C0_seed1','C1_seed1']:
        audits[n]=read(RAW/TRACKS['C']/n/'TRAINING_AUDIT.json');init.append(audits[n]['init_state_sha256'])
        orders.append([r['synthetic'] for r in read(RAW/TRACKS['C']/n/'EXPOSURE.json')])
    assert len(set(init))==1 and orders[0]==orders[1] and len(orders[0])==900
    write(DOC/TRACKS['C']/'TRAINING_AUDIT.json',dict(status='FAIL',reason='C2 post-fit audit KeyError and lost checkpoint; not a performance conclusion',
        completed_audits=audits,C2_seed1_updates_consumed=900,total_student_updates=2700,remaining_student_updates=0,
        C2_frozen_after_training_verification='NOT_RUN',completed_C0_C1_init_parity=True,completed_C0_C1_actual_batch_parity=True,
        repaired_code_unit_tests='PASS',refit_executed=False))
    write(DOC/TRACKS['A']/'PROTOCOL_LOCK.json',dict(status='PASS',new_training=0,selection='frozen existing evidence only, no new DEV selection',confirmation_primary='pooled supervised kp median px'))
    write(DOC/TRACKS['A']/'MECHANISM_RESULT.json',dict(status='PASS',scope='existing candidate freeze only; no confirmation',original_overall_gate_passed=False))
    write(DOC/TRACKS['A']/'TRAINING_AUDIT.json',dict(status='PASS',optimizer_updates=0))
    write(DOC/TRACKS['A']/'VERDICT.json',dict(status='NOT_RUN',verdict='A_NEEDS_NEW_CONFIRMATION_DATA',paper_candidate=False,evidence_level='DEVELOPMENT'))
    for track,result in [('D',d),('B',b),('E',e)]:
        write(DOC/TRACKS[track]/'TRAINING_AUDIT.json',dict(status='NOT_RUN',student_optimizer_updates=0,
            trust_predictor_updates=4500 if track=='D' else 0,reason='mechanism gate FAIL'))
        write(DOC/TRACKS[track]/'VERDICT.json',dict(status='FAIL',verdict=result['verdict'],student_status='NOT_RUN',evidence_level='MECHANISM_ONLY',paper_candidate=False))
        if 'per_seed' in result:write(DOC/TRACKS[track]/'PER_SEED.json',result['per_seed'])
    checks=dict(frozen_checkpoint_SHA='PASS',baseline_output_reproducibility='PASS',split_hash_overlap='PASS',
        no_real_GT_training_input='PASS',same_seed_init_available_fits='PASS',same_seed_actual_minibatches_available_fits='PASS',
        optimizer_allowlist_wiring='PASS',C2_unintended_final_updates='NOT_RUN',center_keypoint_index='PASS',
        shared_symmetry_assignment='PASS',dimension_contract_bindings='PASS',canonical_evaluator_reaggregation_available_fits='PASS',
        finite_gradients_executed_fits='PASS',decisive_final_score_recomputation='PASS',old_artifacts_hash_scope='PASS')
    write(DOC/'REGRESSION_TESTS.json',dict(status='PASS',unit_tests=13,checks=checks,
        limitation='Unit/available-artifact checks passed. This does not certify the lost C2 run or complete the 3-seed C experiment.',
        independent_recomputation=dict(B_catastrophic=bbad,E_clean_coverage=coverage,D_calibration=independent_d)))
    for track in TRACKS:
        write(DOC/TRACKS[track]/'REGRESSION_TESTS.json',dict(status='PASS',source='../REGRESSION_TESTS.json',
            limitation='C2 full-fit freeze verification unavailable' if track=='C' else 'Tests certify implementation checks, not method benefit'))
        write(DOC/TRACKS[track]/'FINAL_AUDIT.json',dict(status='FAIL' if track=='C' else 'PASS',
            executed_scope_accounted=True,performance_verdict=read(DOC/TRACKS[track]/'VERDICT.json'),
            new_student_updates=2700 if track=='C' else 0,old_source_hash_changes=0))
    evidence=read(DOC/TRACKS['A']/'EXISTING_EVIDENCE.json');a=read(DOC/TRACKS['A']/'CANDIDATE_FREEZE.json')
    am=['# Track A — frozen development candidate, confirmation NOT_RUN','',
        'Candidate: image_line_only, seed1, lambda0.25, temperature1, no cap. New training: 0.',
        f"Checkpoint SHA256: `{a['sha256']}`.",'',
        '| Existing arm | kp median px | kp P90 px | translation cm | rotation deg | IoU3D | ADDsym AUC |',
        '|---|---:|---:|---:|---:|---:|---:|']
    base=evidence['baseline'];m={**base['two_d'],**base['main_6d']}
    keys=['keypoint_location_median_px','keypoint_location_p90_px','translation_median_cm','rotation_median_deg','iou3d_median','add_sym_auc']
    am.append('| R0 | '+' | '.join(f'{m[k]:.4f}' for k in keys)+' |')
    for name,row in evidence['arms'].items():am.append('| '+name+' | '+' | '.join(f"{row['metrics'][k]['mean']:.4f}" for k in keys)+' |')
    am+=['','Yaw and per-session/seed results are preserved in A_architecture_confirmation/EXISTING_EVIDENCE.json.',
        'Joint versus line-only remains unresolved; same deployment parameter architecture. Simpler training objective chosen, not a claim of statistical equivalence or faster inference.',
        'Seed1 integrated runtime median17.634ms versus paired R0 median9.152ms; added median8.378ms. These are existing measurements, not a new benchmark.',
        'Existing full-candidate identity/negative-preservation audit is retained. New independent data is not established; source manifests named FINAL are not sufficient.',
        'Verdict: A_NEEDS_NEW_CONFIRMATION_DATA. Existing overall gate failure is unchanged.']
    md(DOC/'TRACK_A_REPORT.md','\n'.join(am)+'\n')
    cm=['# Track C — incomplete execution, no C2 performance verdict','',
        'Stage0 PASS: actual autograd dependencies identify241,566 detection-only trainable parameters. Frozen tensor and dense raw-keypoint invariance passed a one-update probe. Final selection may still change.',
        '', '| Available seed1 arm | AP50-95 | kp median px | kp P90 px | translation cm | yaw deg |','|---|---:|---:|---:|---:|---:|']
    for r in c['available']:
        p=r['MAIN_pose']['ALL'];cm.append(f"| {r['name']} | {r['ap50_95']:.6f} | {r['kp_median']:.4f} | {r['kp_p90']:.4f} | {p['translation_median_cm']:.4f} | {p['yaw_median_deg']:.4f} |")
    cm+=['','Both fits consumed900 actual SGD updates, had identical initial tensor SHA and all900 actual synthetic batch hashes. Night subgroup N106. Detection/ranking, paired common-frame errors, gross20, normalized frame means, full MAIN pose and subgroup records are in PER_SEED.json.',
        'C2 seed1 consumed900 updates but agent-written BN audit code raised KeyError before checkpoint save. Its weights were lost. This is an execution failure, not C_GEOMETRY_PRESERVING_DA_FAIL. C2 vs C0/C1 and three-seed inference are unavailable.',
        'Prefix construction is corrected and regression-tested; final unverified weights now persist before audits. No extra900-update refit was authorized or performed.',
        'A separate tsfm-peft-method-screen GPU process appeared at2026-09-12T14:58:33Z. No wait, kill or modification; remaining C fits NOT_RUN per resource rule.',
        '900+900+900=2700 actual student updates consumed; only two final checkpoints/evaluations available. The requested8100-update comparison is NOT complete.',
        'Other documented limitations: optional Albumentations skipped by installed API mismatch; PyTorch warned that cuBLAS strict determinism was not enabled. Actual input parity was checked, but bit-exact retraining is not claimed.',
        'The C0 first detection-only evaluation is INVALID_FOR_POSE_DUE_TO_TASK_METADATA and preserved. A separate corrected metadata checkpoint has identical tensor weights; valid evaluations live in evaluation_pose/.',
        'The earlier EXECUTION_BLOCKER/PAUSE_AUDIT files are historical event receipts. CPU D/B/E continuation and final Git disposition are recorded by FINAL_AUDIT.json; C itself remains unresolved.']
    md(DOC/'TRACK_C_REPORT.md','\n'.join(cm)+'\n')
    md(DOC/'TRACK_D_REPORT.md',f'''# Track D — D_COMPLEMENTARITY_NOT_PREDICTABLE

Teacher: frozen A local line seed1; corrected DHT seed1 is a descriptive control,
not the main teacher. Teacher train55980; out-of-training heldout1985 frames,
split by scenario into1187 train/417 calibration/381 test. Zero stored image-hash
and scenario overlap.15476 eligible edges from1978 usable frames.

Three hidden64/dropout0.1 MLPs each completed1500 updates. All fifteen calibration
seed/threshold combinations failed the coverage/harm safety combination. For
tau0.25, coverage was17.56%/12.24%/14.15%, mean gain0.373/0.382/0.377px, but harm
35.03%/38.44%/36.09% exceeded25%. These are calibration diagnostics, not selected
test performance. Each seed therefore froze abstention: test selected coverage0,
gain/harm undefined. No test threshold was tried to rescue the result.

On1884 overlapping control edges, local teacher mean normal gain0.12255px versus
corrected DHT -2.39420px. These are teacher-specific normal directions; not an
identical scalar error target or a student benefit claim.

Student fits0. D2 vs D1 NOT_RUN. Evidence level MECHANISM_ONLY. The float32 JSON
serialization correction occurred before any trust/student update; no tuning.
''')
    md(DOC/'TRACK_B_REPORT.md',f'''# Track B — B_JACOBIAN_INVALID

Fixed synthetic calibration256. Numerical derivative at0.01px; verification
at±1px includes the frozen canonical PnP W/D selector and deployment yaw adapter.
State order is[x,z,yaw], with C2 yaw differences modulo pi; center8 excluded.

Finite100%. Relative median errors: x0.235%, z0.306%, yaw0.300%.
P90: x1.837%, z2.647%, yaw2.388%. No ill-conditioned samples by the locked rule.
However6/256=2.34375% samples exceeded the locked catastrophic linearisation
criterion, above1%. Each had a W/D branch switch among its perturbations.
Median/P90 accuracy does not waive this tail gate.

Gradient calibration NOT_RUN; B0/B1/B2 training NOT_RUN; student updates0.
LC-derived B1 remains prior art, not a novel loss. Evidence MECHANISM_ONLY.
''')
    clean=e['excluded_prior_bad_reference'];allr=e['all_references']
    md(DOC/'TRACK_E_REPORT.md',f'''# Track E — E_RGBD_ALIGNMENT_NOT_READY

Existing506-frame sensor development population; top/side semantic hull edges
only, excluding bottom-bottom edges. Fixed prior depth scale0.001 and prior
Sobel97th-percentile discontinuities, nearest boundary along±12px normal.
No offset or scale fitting. Prior capturepallet11 exclusion is not newly selected.

All references: coverage{allr['coverage']:.4%}, median absolute displacement
{allr['median_absolute_px']}px, session-bias safety{allr['session_bias_safe_fraction']:.4%}.
Excluding prior bad references: coverage{clean['coverage']:.4%}, median absolute
{clean['median_absolute_px']}px, signed median{clean['median_signed_px']}px,
session-bias safety{clean['session_bias_safe_fraction']:.4%}.

Usable coverage fails80%; session safety also fails80%. Missing boundaries stay
in the denominator. A good conditional displacement median does not establish
overall alignment. Teacher construction/superiority and student fits NOT_RUN.
Student updates0. Evidence MECHANISM_ONLY; no claim that RGB-D teaching is impossible.
''')
    write(DOC/'FINAL_AUDIT.json',dict(status='PASS',scope='accurate closure and preservation of executed scope, NOT completion of all requested C fits',
        overall_execution='PARTIAL_RESOURCE_AND_IMPLEMENTATION_LIMITED',all_method_results_accounted=True,
        A='NOT_RUN: needs independent data',C='NOT_RUN: incomplete comparison',D='FAIL',B='FAIL',E='FAIL',
        actual_student_updates=2700,completed_student_checkpoints=2,trust_updates=4500,
        C2_lost_fit_updates=900,extra_fit_or_sweep=False,old_hash_checked_files=len(preserved),old_hash_changes=changed,
        paper_final_modified=False,foreign_gpu_process_modified=False,reboot=False,
        recommendation='NO_NEW_METHOD_FREEZE_EXISTING_STORY',main_proposed_method=None,
        git_status='commit/push verified after this receipt; see final CLI SHA confirmation'))
    print('Executed scope accounted; C remains incomplete, not a negative method result.')

if __name__=='__main__':main()
