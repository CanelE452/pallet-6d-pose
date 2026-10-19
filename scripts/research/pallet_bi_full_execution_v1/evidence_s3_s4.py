"""Draft reporting from sealed saved summary metadata only; no scientific work."""
from pathlib import Path
import argparse,csv,hashlib,json

ARMS=('S3_COMMON','S3_BI','S4_PL','S4_CONTOUR','S4_IMAGE')
SEEDS=(20261018,20261019,20261020)
POPS=('PRIMARY192','TEST49','REMAINDER143','CAL46')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def build(run,out):
    r,o=Path(run),Path(out);o.mkdir(parents=True,exist_ok=True)
    es=read(r/'S3_S4_EVALUATION/EVALUATION_SEAL.json');check=read(r/'S3_S4_INDEPENDENT/INDEPENDENT_VERIFICATION.json')
    if not es['complete']or not check['PASS']or check['canonical_evaluation_seal_sha256']!=sha(r/'S3_S4_EVALUATION/EVALUATION_SEAL.json'):raise ValueError('Completed verified evaluation required')
    for name,digest in es['artifacts'].items():
        if sha(r/'S3_S4_EVALUATION'/name)!=digest:raise ValueError('Canonical artifact differs')
    summary=read(r/'S3_S4_EVALUATION/SUMMARY.json');s1seal=read(r/'S1_EVALUATION/EVALUATION_SEAL.json')
    s1check=read(r/'S1_INDEPENDENT_FINAL/INDEPENDENT_VERIFICATION.json')
    if not s1seal['complete']or not s1check['PASS']or s1seal['reference_sha256']!=es['reference_sha256']or s1check['canonical_evaluation_seal_sha256']!=sha(r/'S1_EVALUATION/EVALUATION_SEAL.json')or sha(r/'S1_EVALUATION/SUMMARY.json')!=s1seal['artifacts']['SUMMARY.json']:raise ValueError('Completed same-reference S1 summary required')
    s1=read(r/'S1_EVALUATION/SUMMARY.json')
    bankseal=read(r/'S3_S4_BANK_ANALYSIS/ANALYSIS_SEAL.json')
    if not bankseal['complete']or sha(r/'S3_S4_BANK_ANALYSIS/SUMMARY.json')!=bankseal['artifacts']['SUMMARY.json']:raise ValueError('Complete bound bank summary required')
    bank=read(r/'S3_S4_BANK_ANALYSIS/SUMMARY.json');cs=read(r/'S3_S4_WITHIN_STAGE_CONTRASTS/CONTRASTS_SEAL.json')
    cp=r/'S3_S4_WITHIN_STAGE_CONTRASTS/INDEPENDENT_PAIRED_CONTRASTS.json'
    if not cs['complete']or not cs['PASS']or sha(cp)!=cs['artifacts'][cp.name]:raise ValueError('Completed paired contrasts required')
    contrasts=read(cp);validation=read(r/'s3_s4_postprocess/VALIDATION_SELECTION_INPUTS.json')
    exact={pop:{**summary[pop],**{k:v for k,v in s1[pop].items()if k.startswith('S1_UNION_')}}for pop in POPS}
    public=o/'S3_S4_FULL_PRECISION_METRICS.json';public.write_text(json.dumps(exact,indent=2,allow_nan=False)+'\n')
    fields=['population','arm','denominator','successes','success_rate','no_pose','T_gt50cm','T_gt100cm','R_gt45deg']
    stats=('n','missing','mean','sample_std','sample_variance','median','P90','P95','max')
    fields += [prefix+'_'+key for prefix in ('T_cm','R_deg')for key in stats]
    fields += [flag+'_'+kind for flag in ('accepted','fallback','technical_failure','solver_rejected','timeout')for kind in ('true_n','false_n','unknown_n')]
    with (o/'S3_S4_FULL_PRECISION_METRICS.csv').open('w',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for pop in POPS:
            for arm,v in exact[pop].items():
                flat={k:v[k]for k in fields[2:9]};flat.update(population=pop,arm=arm)
                for prefix,axis in [('T_cm','translation_cm'),('R_deg','rotation_deg')]:flat.update({prefix+'_'+k:v[axis][k]for k in stats})
                for flag in ('accepted','fallback','technical_failure','solver_rejected','timeout'):flat.update({flag+'_'+k:v['status_counts'][flag][k]for k in ('true_n','false_n','unknown_n')})
                writer.writerow(flat)
    md=['# S3/S4 completed evidence — draft','',
        'All five arms completed the original1241 production rows, own synthetic TRAIN/VAL preparation, three H1 fits and sealed238 real predictions before canonical scoring. This section reports completed S3/S4 evidence only; final S2/F results and the whole campaign remain pending. PRIMARY192 is TEST49+REMAINDER143 reused development data; CAL46 is separate and is not an independent test. Success is T≤5cm and C2 rotation≤5deg. Seeds below are20261018/19/20.','',
        f'Independent quaternion/atan2 verification agrees on every238×22 arm success classification; {check["compared_numeric_fields"]} summary numbers agree within {check["summary_numeric_max_abs"]!r}. The21 registered paired contrasts also pass independent verification ({cs["compared_numeric_fields"]} numbers, maximum difference {cs["maximum_absolute_difference"]!r}). No additional optimizer, pose call, fit or reference decode was used to assemble this draft.','',
        'The JSON and CSV companions preserve full Python round-trip numeric precision for all four populations, baseline arms, both PF anchors, S1 and all new seeds. Tables display six decimals. Variance is sample variance (ddof1), SD is sample SD; no trimming or outlier removal. Error moments use available poses, while success/no-pose retain the full denominator. Status unknown counts remain unknown rather than zero failures.','',
        '[Full precision metrics (JSON)](S3_S4_FULL_PRECISION_METRICS.json) · [Full precision metrics and status counts (CSV)](S3_S4_FULL_PRECISION_METRICS.csv)','',
        '## Success counts across all populations','', '| Arm | PRIMARY192 s18/19/20 | TEST49 | REMAINDER143 | CAL46 |','|---|---:|---:|---:|---:|']
    for arm in ('D1_H1_MLP','S1_UNION')+ARMS:
        md.append('| '+arm+' | '+' | '.join('/'.join(str(exact[p][f'{arm}_s{s}']['successes'])for s in SEEDS)for p in POPS)+' |')
    md += ['', '## Actual information consumed and interpretation','',
        'S3 COMMON and BI both generate generic material/interior proposals from the same admitted physical starts, image cues, two masks and accessible asset pool. Their support adoption policies differ. Both union banks retain all original D1 and PF actions and the same original downstream feature context; generic factors change the added proposals rather than replacing every old H1 feature. COMMON exceeds BI success counts on PRIMARY in all three seeds (BI−COMMON −1/−6/−2), so these runs do not demonstrate a BI policy gain over the strong common control.','',
        'S4 PL uses the original native point observations, typed normal factors, and pose-updated native-outline/full-mask factors. Its branch never invokes the proxy renderer. The three selected recoveries from fixed original41 use START/0/N0 updates3/1/4, as audited separately; they are genuine refinements rather than PF duplicates. PL improves over S1 by4/5/2 PRIMARY successes, but that gain is evidence for this PL proposal/union/head pipeline, not a gain attributable to BI material ownership, whole-image RGB matching, or rendered CAD evidence.','',
        'CONTOUR uses the visible rendered proxy contour. IMAGE uses whole native grayscale DIS correspondences and rendered depth with PnP/refinement; it is a finite cold-start single-image baseline, not a DeepIM/SRT3D or temporal reproduction. Against PL, CONTOUR changes PRIMARY successes by−6/−5/+2; IMAGE changes them by−6/−5/−4. IMAGE−CONTOUR is0/0/−6. Additional candidate coverage therefore does not imply improved selected accuracy.','',
        'The asset audit confirms all235 synthetic VAL rows lack their exact named GLB and use fixed structured proxies; present-named USD rows occur in TRAIN. This is a confirmed distribution/certification limitation for material, contour and image geometry, not a proven cause of their accuracy. PL does not render those assets. Real and synthetic instance ownership, signed-axis registration and exact physical correspondence remain uncertified, with zero human owner labels. Recorded EGL/Mesa llvmpipe rendering is software rasterization, not hardware GPU rendering.','',
        'Validation joint_success is a count out of235, not a probability or PRIMARY result. Median VAL count/loss is listed below only as the registered development input; PRIMARY outcomes are not used to tune the final system. Conditional noise-model diagnostics and pending S2/F outputs cannot be described as completed end-to-end gains.','',
        '| Arm | Median VAL joint success /235 | Median VAL loss |','|---|---:|---:|']
    for arm in ARMS:md.append(f'| {arm} | {validation[arm]["median_joint_success"]:g} | {validation[arm]["median_validation_loss"]!r} |')
    md += ['', '## Generated, admitted and selected coverage','',
        'All old151 PRIMARY good-bank IDs remain generated and eligible. The generated/eligible upper bounds below are retrospective coverage diagnostics, not selected performance. Fixed original41 and PF-union38 are preserved original cohorts, not redefined per arm/seed.','',
        '| Arm | PRIMARY generated / eligible | Fixed41 generated / eligible | Fixed41 selected s18/19/20 | Fixed38 generated / eligible | Fixed38 selected s18/19/20 |','|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        b=bank[arm];p=b['populations']['PRIMARY192'];a=b['fixed_cohorts']['original41_candidate_absent'];c=b['fixed_cohorts']['PF_union38_candidate_absent']
        selected=lambda q:'/'.join(str(q['seeds'][str(s)]['selected_good'])for s in SEEDS)
        md.append(f'| {arm} | {p["generated_good"]}/{p["eligible_good"]} | {a["generated_good"]}/{a["eligible_good"]} | {selected(a)} | {c["generated_good"]}/{c["eligible_good"]} | {selected(c)} |')
    md += ['', '## PRIMARY matched paired contrasts','',
        'Counts and95% frame-weighted recording-cluster percentile intervals are copied from the completed independent paired results: six PRIMARY recording clusters,10000 draws, seed20260917. Named-session sensitivity and every leave-one-recording-cluster result remain in the private contrast artifact; these intervals do not create new independent samples.','',
        '| Contrast | Seed | Recovery | Damage | Δ successes | Δ percentage points | Recording-cluster95% CI (percentage points) |','|---|---:|---:|---:|---:|---:|---:|']
    for name,v in contrasts['PRIMARY192'].items():
        ci=v['deltas']['success_rate']['recording_cluster_CI']['CI95'];md.append(f'| {name.rsplit("_s",1)[0]} | {name.rsplit("_s",1)[1]} | {v["counts"]["RECOVERY"]} | {v["counts"]["DAMAGE"]} | {v["success_difference_count"]} | {v["success_difference_percentage_points"]:.6f} | [{100*ci[0]:.6f}, {100*ci[1]:.6f}] |')
    for pop in POPS:
        md += ['',f'## {pop}: per-seed error moments and tails','',
            'T is cm and R is degrees. Each axis cell is **mean / sample SD / sample variance / median / P90 / P95 / maximum**. Full precision, all baselines and every status category are in the companions.','',
            '| Arm | Seed | Success / full denominator | T moments | R moments | T>50 / T>100 / R>45 | No pose | Fallback true/false/unknown |','|---|---:|---:|---|---|---:|---:|---:|']
        for arm in ARMS:
            for s in SEEDS:
                v=exact[pop][f'{arm}_s{s}'];fmt=lambda a:' / '.join(f'{v[a][k]:.6f}'for k in ('mean','sample_std','sample_variance','median','P90','P95','max'));fb=v['status_counts']['fallback']
                md.append(f'| {arm} | {s} | {v["successes"]}/{v["denominator"]} | {fmt("translation_cm")} | {fmt("rotation_deg")} | {v["T_gt50cm"]}/{v["T_gt100cm"]}/{v["R_gt45deg"]} | {v["no_pose"]} | {fb["true_n"]}/{fb["false_n"]}/{fb["unknown_n"]} |')
    evidence=['S3_S4_EVALUATION/EVALUATION_SEAL.json','S3_S4_EVALUATION/SUMMARY.json','S3_S4_INDEPENDENT/INDEPENDENT_VERIFICATION.json','S3_S4_BANK_ANALYSIS/ANALYSIS_SEAL.json','S3_S4_BANK_ANALYSIS/SUMMARY.json','S3_S4_WITHIN_STAGE_CONTRASTS/CONTRASTS_SEAL.json','S3_S4_WITHIN_STAGE_CONTRASTS/INDEPENDENT_PAIRED_CONTRASTS.json','s3_s4_production/FULL_RECEIPT.json','s3_s4_production/MATERIAL_ASSET_AUDIT.json','s3_s4_archive_report/PROXY_ASSET_DOMAIN_COUNTS.json','s3_s4_postprocess/VALIDATION_SELECTION_INPUTS.json','S3_S4_HANDOFF_SEAL.json','s3_s4_postseal_analysis/ACTUAL_COST_INVENTORY.json']
    md += ['', '## Evidence and limits','',
        'Literal executed canonical, independent, whole-bank, saved-rank and21-contrast commands are preserved in ROOT_S3_S4_POSTSEAL_EXECUTION/POSTPROCESS_LEDGER.jsonl. Its completion-prefix supplement authenticates both the original eight-event snapshot and current ten-event append-only journal. Completed fit checkpoints, last/resume/history/normalizers and source hashes remain private through the handoff closure. Instrumented generation, internal H1, and outer child CPU/wall have different scopes and must not be double counted. Old technical attempts and uninstrumented setup remain disclosed rather than silently zeroed.','',
        'Per-image transition galleries and full-axis T-horizontal/R-vertical plots are display-only companions; source/receipt versions and actual limited pixel checks must be reported separately. Neither visual agreement of a nearly square cuboid nor an automatic native ID certifies physical material ownership. C2 allows180° Y symmetry only;90° orientation errors remain errors.','',
        'This draft makes no accuracy claim for pending S2 or F and no whole-campaign completion claim.','', '| Private evidence path under RUN | SHA256 |','|---|---|']
    evidence += ['S1_EVALUATION/EVALUATION_SEAL.json','S1_EVALUATION/SUMMARY.json','S1_INDEPENDENT_FINAL/INDEPENDENT_VERIFICATION.json']
    bindings={n:sha(r/n)for n in evidence}
    for n,d in bindings.items():md.append(f'| {n} | {d} |')
    path=o/'S3_S4_EVIDENCE_DRAFT.md';path.write_text('\n'.join(md)+'\n')
    receipt={'complete':True,'scope':'draft report metadata assembly only; scientific stages remain separate','source_sha256':sha(__file__),'evidence_sha256':bindings,'public_artifacts':{p.name:sha(p)for p in [path,public,o/'S3_S4_FULL_PRECISION_METRICS.csv']},'new_pose_calls':0,'new_fits':0,'new_metric_evaluations':0,'reference_files_decoded':0,'whole_campaign_complete':False}
    receipt_path=r/'S3_S4_EVIDENCE_DRAFT_BINDINGS_V2.json'
    if receipt_path.exists():raise ValueError('Preserve prior report binding; use fresh revision')
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n');return receipt
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(build(a.run,a.output)))
