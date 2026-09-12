"""Close the authorized repair using complete evidence, retaining partial history."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.contracts import ROOT,RAW,DOC,R0,R0_SHA,sha,write

OUT=DOC/'C_geometry_preserving_da/RESUME_COMPLETE'

def read(path):return json.loads(path.read_text())

def text(path,value):
    if path.exists():
        assert path.read_text()==value
        return
    path.write_text(value)

def main():
    summary=read(OUT/'PER_SEED.json');verdict=read(OUT/'VERDICT.json')
    audit=read(OUT/'TRAINING_AUDIT.json');independent=read(OUT/'INDEPENDENT_SAVED_AUDIT.json')
    assert audit['status']==independent['status']=='PASS'
    assert len(summary['per_seed'])==9 and audit['total_optimizer_updates']==8100
    passed=verdict['status']=='PASS'
    assert passed==all(verdict['gates'].values())
    adapter_followup=(not verdict['gates']['recovery_70pct'] and all(
        verdict['gates'][k] for k in ('median_px_nonworse','p90_px_nonworse','gross20_nonworse')))
    preserved=read(DOC/'PRESERVED_SOURCE_SHA.json')
    assert all(sha(ROOT/p)==h for p,h in preserved.items())
    means=summary['seed_means']
    recommendation='C_GEOMETRY_PRESERVING_DA' if passed else 'NO_NEW_METHOD_FREEZE_EXISTING_STORY'
    rows=['# C complete three-seed screen — authorized repair','',
        f"Verdict: **{verdict['verdict']}**. Evidence: {verdict['evidence_level']}; independent confirmation NOT_RUN.",'',
        'All9 arm/seed fits have900 actual optimizer updates and final checkpoints. Each was evaluated on the same319 positive+2689 negative frames with unchanged canonical2D and MAIN6D evaluators.',
        'Accounting:8100 valid-comparison updates +900 consumed by the historical lost-checkpoint run =9000 cumulative. This authorized resume added6300 updates, including one900-update repair. No new hyperparameter, threshold, seed or architecture search.',
        '', '## Seed means','',
        '| Arm | AP50-95 | Det | paired kp median px | paired P90 px | paired gross20 | translation cm | yaw deg | IoU3D | ADDsym AUC |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ['C0','C1','C2']:
        r=means[arm];g=r['paired_geometry'];p=r['pose']
        rows.append(f"| {arm} | {r['ap50_95']:.6f} | {r['Det']:.6f} | {g['median_px']:.4f} | {g['p90_px']:.4f} | {g['gross20']:.6f} | {p['translation_median_cm']:.4f} | {p['yaw_median_deg']:.4f} | {p['iou3d_median']:.6f} | {p['add_sym_auc']:.6f} |")
    rows+=['','Paired geometry uses the3-arm common matched-frame set within each seed, not the union of separately detected frames. Seed means are means of per-seed statistics, not pooled independent seeds.',
        '', '## All seeds','',
        '| Seed | Arm | AP50 | AP50-95 | AUROC | FPR95 | Night N | Night Det | matched kp median/P90 px | rotation/yaw deg | translation cm | pose coverage |',
        '|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|']
    for r in summary['per_seed']:
        g=r['geometry'];p=r['pose']
        rows.append(f"| {r['seed']} | {r['arm']} | {r['ap50']:.6f} | {r['ap50_95']:.6f} | {r['auroc']:.6f} | {r['fpr95']:.6f} | {r['night_N']} | {r['night_Det']:.6f} | {g['median_px']:.4f}/{g['p90_px']:.4f} | {p['rotation_median_deg']:.4f}/{p['yaw_median_deg']:.4f} | {p['translation_median_cm']:.4f} | {r['pose_coverage']:.4f} |")
    rows+=['','## Locked gates','']
    rows += [f'- {k}: {"PASS" if v else "FAIL"}' for k,v in verdict['gates'].items()]
    rows+=['',f"C1−C0 AP50-95: {verdict['C1_minus_C0_detection']:.8f}; C2−C0: {verdict['C2_minus_C0_detection']:.8f}; recovery fraction: {verdict['recovery_fraction']}.",
        'Per-seed contrasts and safety flags are in VERDICT.json. The gate was frozen before the first fit, not chosen from these results.',
        '', '## Execution and deployment audit','',
        'All9 initial tensor hashes match. All900 actual synthetic batch hashes per seed match across3 arms; C1/C2 real batch hashes also match. Existing273 pseudo labels are reused; manual real GT never enters training.',
        'All3 saved C2 checkpoints independently preserve639 non-detection state keys bit-exactly versus R0. Raw dense pose tensors match on two CPU probe sizes. All9 canonicalCSV median/P90 values were independently recomputed.',
        'Deployment remains the stock3,043,704-parameter RGB Pose26 model. No additional branch or sensor is added; runtime in milliseconds was not newly benchmarked. Dense pose invariance does not imply unchanged final detection selection.',
        'The previous BN audit failure, old checkpoint/task-metadata correction and partial reports remain intact. C0/C1 seed1 were not refit. Existing optional-Albumentations and strict-cuBLAS-determinism limitations remain; actual input parity is verified, bit-exact retraining is not claimed.',
        '', '## Paper interpretation','',
        f'Recommended path: `{recommendation}`.',
        ('C qualifies as a development candidate under the frozen gate. It is not independently confirmed and branch-freezing/ST novelty remains high-risk.' if passed else 'C does not qualify under the frozen gate. Do not promote one improved metric into geometry-preserving DA success. Limit the paper to the strong synthetic baseline, controlled target-adaptation study, local-line development result and failure-mechanism analysis; do not add a new module to force a method claim.'),
        'A still lacks untouched confirmation data; D/B/E remain their original mechanism FAIL outcomes with zero student fits. No D3 adapter was implemented. No claim that DHT, self-training or depth teaching is impossible.']
    if adapter_followup:
        rows+=['', 'Optional future work only: because C2 preserves geometry but does not recover detection gain, the master permits recommending a detection-specific residual adapter (D3). It is not implemented, trained or selected here and requires a separately authorized experiment; it is not required for the recommended paper closeout.']
    text(OUT/'REPORT.md','\n'.join(rows)+'\n')
    candidate=None
    if passed:
        checkpoint=RAW/'C_geometry_preserving_da/C2_seed1/last.pt'
        candidate=dict(method='C_GEOMETRY_PRESERVING_DA',arm='C2',seed=1,selection='first numeric seed, not best DEV',
            checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=sha(checkpoint),
            gate_population='reused DEV319+NEG2689; not independent',evidence_level='DEVELOPMENT',
            manual_real_GT_in_training=False,extra_sensor_at_inference=False,novelty_claim=False)
    write(OUT/'PROPOSED_METHOD_FREEZE.json',dict(main_candidate=candidate,max_candidates=1,recommendation=recommendation))
    write(DOC/'RESUME_FINAL_AUDIT.json',dict(status='PASS',C_execution='COMPLETE',C_scientific_verdict=verdict['status'],
        valid_fits=9,updates_per_fit=900,valid_comparison_updates=8100,prior_lost_fit_updates=900,
        cumulative_student_updates=9000,new_updates_this_resume=6300,other_student_fits=0,prior_trust_updates=4500,
        checkpoint_independent_verification='PASS',augmented_exposure_parity='PASS',all_seed_final_only=True,
        original_source_hash_checks=len(preserved),original_source_changes=0,
        paper_final_modified=False,reboot=False,foreign_GPU_process_modified=False,
        recommendation=recommendation,independent_confirmation='NOT_RUN: no documented untouched population',
        optional_future_adapter_recommendation=adapter_followup,adapter_implemented=False,
        report='C_geometry_preserving_da/RESUME_COMPLETE/REPORT.md',
        historical_partial_record='FINAL_AUDIT.json and EXECUTION_BLOCKER.json retained as prior events'))
    matrix=['# Updated track decision after complete C comparison','',
        f'Current recommendation: `{recommendation}`. This supersedes the C-incomplete decision in the initial TRACK_DECISION_MATRIX.md; initial evidence is retained.',
        '', '| Track | Research question | Stage0 status | Stage1 executed? | Main comparison | Primary result | P90 safety | Pose safety | Runtime/deployment cost | Real labels in training? | Extra inference sensor? | Novelty risk | Closest prior art | Evidence level | Paper candidate? | Reason |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|',
        '| A | Independent local-line confirmation? | PASS freeze | No new training; confirmation NOT_RUN | R0 vs frozen local-line | Existing development gain only | Not independently confirmed | Existing partial gain | Existing added8.378ms seed1 | No | No | Medium/high | Deep Hough line priors | DEVELOPMENT | No main candidate | Needs new independent population |',
        f"| C | Detection-only DA preserves geometry? | PASS | Yes9x900 | C2 vs C0/C1 | C2−C0 AP50-95 {verdict['C2_minus_C0_detection']:.6f} | {'PASS' if verdict['gates']['p90_px_nonworse'] else 'FAIL'} | {'PASS' if verdict['gates']['pose_safe'] else 'FAIL'} | Stock RGB graph; no extra parameters | No manual real GT; frozen pseudo labels | No | High: freezing/ST established | Soft Teacher; task-specific freezing | {verdict['evidence_level']} | {'Development candidate' if passed else 'No'} | {verdict['verdict']} |",
        '| D | Predict useful teacher normal components? | FAIL | Trust3x1500; students0 | Selected realized normal gain | All calibration thresholds unsafe | Students NOT_RUN | Students NOT_RUN | Training-only proposal | No | No planned | Medium/high | Learning to Reweight Examples | MECHANISM_ONLY | No | Harm safety fails |',
        '| B | Reliable alignment Jacobian? | FAIL | No | Local derivative vs±1px PnP |6/256 catastrophic=2.34% | Training NOT_RUN | W/D branch switches | Loss-only proposal | No | No | High: LC-derived | Linear-Covariance Loss | MECHANISM_ONLY | No | Exceeds1% tail gate |',
        '| E | RGB-D boundaries ready? | FAIL | No teacher/student | Manual non-ground hull vs depth | Clean coverage23.43% | Teacher NOT_RUN | Teacher NOT_RUN | RGB-D offline planned | No training; manual GT diagnostic | No planned RGB-student depth | High | Learning Using Privileged Information | MECHANISM_ONLY | No | Coverage below80% |',
        '', 'Primary-source boundaries remain in PRIOR_ART_BOUNDARY.md. Reused DEV is not independent confirmation; no automatic novelty or state-of-the-art claim.']
    text(DOC/'RESUME_TRACK_DECISION_MATRIX.md','\n'.join(matrix)+'\n')
    ko=f'''# C 정정 재개 최종 결과

C0/C1/C2 × seed1/2/3의9개 fit과 동일 평가를 모두 완료했다.
판정: `{verdict['verdict']}`.
권고: `{recommendation}`.

유효 비교 학습8100회, 과거 저장 실패로 소모한900회, 누적9000회다.
이번 승인으로6300회를 추가했고 기존 C0/C1 seed1은 다시 학습하지 않았다.
모든 seed의 증강 입력 순서와 저장된 C2 frozen state를 검증했다.

{'C를 development 단계의 main 후보로만 고정한다. 독립 확인과 novelty 검토는 여전히 필요하다.' if passed else 'C는 사전 gate를 통과하지 못했으므로 main 방법으로 승격하지 않는다.'}
A/D/B/E의 기존 판정은 바뀌지 않는다. 새 독립 데이터는 아직 필요하며,
DHT·자기학습·RGB-D의 일반적 불가능성이나 독립 일반화를 주장하지 않는다.

현재 상세 결과: C_geometry_preserving_da/RESUME_COMPLETE/REPORT.md.
현재 의사결정표: RESUME_TRACK_DECISION_MATRIX.md.
초기 미완료 기록은 과거 사건으로 보존하며 현재 완료 상태와 혼동하지 않는다.
'''
    text(DOC/'RESUME_FINAL_RECOMMENDATION_KO.md',ko)
    write(DOC/'CURRENT_RESULT.json',dict(status='COMPLETE_C_SCREEN',recommended_paper_path=recommendation,
        audit='RESUME_FINAL_AUDIT.json',decision_matrix='RESUME_TRACK_DECISION_MATRIX.md',
        recommendation='RESUME_FINAL_RECOMMENDATION_KO.md',report='C_geometry_preserving_da/RESUME_COMPLETE/REPORT.md'))
    print('Complete report:',recommendation,flush=True)

if __name__=='__main__':main()
