"""Audit all attempts and matched controls; never select the best repeat."""
import json
import numpy as np
from . import recovery_common as R
from .pseudo import top
C=R.C


def clustered_pck_contrast(a,b,metadata):
    assert len(a)==len(b) and len(a)>0
    aa={r['id']:r for r in a};bb={r['id']:r for r in b};assert set(aa)==set(bb)
    sessions=sorted({metadata[k] for k in aa});totals=[]
    for session in sessions:
        ids=[k for k in aa if metadata[k]==session]
        good_a=sum(sum(e<=20 for e in aa[k]['errors']) for k in ids)
        good_b=sum(sum(e<=20 for e in bb[k]['errors']) for k in ids)
        count=sum(len(aa[k]['errors']) for k in ids)
        assert count==sum(len(bb[k]['errors']) for k in ids)
        totals.append([good_a-good_b,count])
    totals=np.array(totals,float);rng=np.random.default_rng(20260920)
    w=rng.multinomial(len(sessions),np.full(len(sessions),1/len(sessions)),size=10000)
    deltas=100*(w@totals[:,0])/(w@totals[:,1])
    return dict(delta_pp=float(100*totals[:,0].sum()/totals[:,1].sum()),CI95_pp=np.quantile(deltas,[.025,.975]).tolist(),
                clusters=len(sessions),bootstrap_replicates=10000,exploratory=True,multiple_search_adjustment=False)


def detection_parity(phase,arm):
    rows=C.read(R.RAW/phase/f'EVAL_PREDICTIONS_{arm}.json')['records']
    old={r['id']:r for r in C.read(R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records']}
    max_box=max_score=0.
    for row in rows:
        a,b=row['prediction'],old[row['id']]['prediction'];assert a['selected_index']==b['selected_index'] and len(a['candidates'])==len(b['candidates'])
        for p,q in zip(a['candidates'],b['candidates']):
            max_box=max(max_box,float(np.max(np.abs(np.array(p['box_xyxy'])-q['box_xyxy']))));max_score=max(max_score,abs(p['score']-q['score']))
    assert max_box==max_score==0
    return dict(images=len(rows),max_box_difference_px=max_box,max_score_difference=max_score)


def main():
    baselines=C.read(R.BASE_RAW/'paper_metrics_plastic/SUMMARY.json')['arms']
    oldmetrics=C.read(R.BASE_RAW/'EVAL_METRICS.json');base_sym=C.read(R.BASE_DOC/'RESULTS.json')['summaries']['R0']['PLASTIC']
    metadata={r['id']:r['session'] for r in C.read(R.BASE_DOC/'EVAL_PROTOCOL.json')['records']}
    teacher_lock=C.ROOT/'_docs/experiments/pallet_posefix_replay_v1/INPUT_LOCK.json'
    teacher_ids=set(C.read(teacher_lock)['real_ids'])
    base_rows=[r for r in oldmetrics['R0'] if r['kind']=='PLASTIC']
    overlap=sorted(teacher_ids & {r['id'] for r in base_rows})
    teacher_sessions={metadata[k] for k in overlap}
    provenance=dict(teacher_lock=C.bound(teacher_lock),manual_teacher_overlap_ids=overlap,
        full194_status='EXPLORATORY_DEV_WITH_TEACHER_TRAINING_OVERLAP' if overlap else 'REUSED_DEV',
        exclusions_are_posthoc_sensitivity_not_new_holdout=True)
    grid={};all_bindings=[]
    for phase in ['bn_probe','pose_only','pose_repeat']:
        protocol=C.read(R.DOC/phase/'PROTOCOL.json')
        for b in protocol['sources']:C.verify(b)
        grid[phase]={}
        for arm in protocol['arms']:
            path=R.RAW/phase/f'RESULTS_{arm}.json'
            if not path.exists():path=R.RAW/phase/f'SCREEN_{arm}.json'
            result=C.read(path);fit=C.read(R.DOC/phase/f'FIT_{arm}.json');C.verify(fit['checkpoint'])
            grid[phase][arm]=dict(PCK20=100*result['symmetry']['PCK']['20'],median8=result['symmetry']['matched_pooled_corner8_median_px'],
                iou3d=result['pose']['iou3d_median'],t_cm=result['pose']['translation_median_cm'],ADDsym=result['pose']['add_sym_auc'])
            if phase!='bn_probe':
                assert fit['protected_state_exact'] and fit['optimizer_steps']==320
                grid[phase][arm]['detector_parity']=detection_parity(phase,arm)
            all_bindings.extend([C.bound(path),C.bound(R.DOC/phase/f'FIT_{arm}.json')])
    runs=[]
    for phase,suffix in [('pose_only','LR5'),('pose_repeat','ORDER43'),('pose_repeat','ORDER44')]:
        result=C.read(R.RAW/phase/f'RESULTS_REF_{suffix}.json')
        control=C.read(R.RAW/phase/f'SCREEN_SYN_{suffix}.json');raw=C.read(R.RAW/phase/f'SCREEN_RAW_{suffix}.json')
        d=result['detection']['report']['metrics']['box_and_keypoint_2d'];bd=baselines['R0']['detection']['report']['metrics']['box_and_keypoint_2d']
        flags=dict(pck20_above_R0=result['symmetry']['PCK']['20']>base_sym['PCK']['20'],
            keypoint_median_below_R0=d['keypoint_location_median_px']<bd['keypoint_location_median_px'],
            ap_preserved=d['box_ap50_95']>=bd['box_ap50_95'],iou3d_preserved=result['pose']['iou3d_median']>=baselines['R0']['pose']['iou3d_median'],
            pck20_above_SYN=result['symmetry']['PCK']['20']>control['symmetry']['PCK']['20'],
            pck20_above_RAW=result['symmetry']['PCK']['20']>raw['symmetry']['PCK']['20'])
        runs.append(dict(phase=phase,arm=f'REF_{suffix}',flags=flags,
            pck20=100*result['symmetry']['PCK']['20'],kp_median=d['keypoint_location_median_px'],kp_p90=d['keypoint_location_p90_px'],
            ap5095=d['box_ap50_95'],fpr95=result['detection']['ranking']['fpr95'],pose=result['pose'],
            contrasts={name:clustered_pck_contrast(result['metrics'],other,metadata) for name,other in [('R0',[r for r in oldmetrics['R0'] if r['kind']=='PLASTIC']),('SYN',control['metrics']),('RAW',raw['metrics'])]},
            source=C.bound(R.RAW/phase/f'RESULTS_REF_{suffix}.json')))
    traces={}
    for seed in [43,44]:
        a=C.read(R.DOC/'pose_repeat'/f'TRACE_RAW_ORDER{seed}.json');b=C.read(R.DOC/'pose_repeat'/f'TRACE_REF_ORDER{seed}.json')
        assert a['image_tensor_sha256']==b['image_tensor_sha256']
        traces[str(seed)]=b['image_tensor_sha256']
    assert traces['43']!=traces['44']
    checkpoints=[C.read(R.DOC/p/f'FIT_REF_{s}.json')['checkpoint']['sha256'] for p,s in [('pose_only','LR5'),('pose_repeat','ORDER43'),('pose_repeat','ORDER44')]]
    assert len(set(checkpoints))==3
    sensitivity={}
    for label,excluded in [('exclude_exact_teacher_images',set(overlap)),
                           ('exclude_teacher_overlap_sessions',{k for k,v in metadata.items() if v in teacher_sessions})]:
        base=[r for r in base_rows if r['id'] not in excluded]
        values={}
        for run in runs:
            rows=C.read(R.RAW/run['phase']/f"RESULTS_{run['arm']}.json")['metrics']
            rows=[r for r in rows if r['id'] not in excluded]
            errors=[e for r in rows for e in r['errors']]
            values[run['arm']]=dict(pck20=100*sum(e<=20 for e in errors)/len(errors),
                contrast_R0=clustered_pck_contrast(rows,base,metadata))
        errors=[e for r in base for e in r['errors']]
        sensitivity[label]=dict(images=len(base),R0_PCK20=100*sum(e<=20 for e in errors)/len(errors),runs=values)
    summary=dict(complete=True,goal_automatically_completed=False,attempts=grid,refined_repeats=runs,
                 teacher_provenance=provenance,posthoc_sensitivity=sensitivity,
                 all_repeat_primary_and_control_flags_pass=all(all(r['flags'].values()) for r in runs),
                 genuine_order_repeats=True,paired_raw_ref_same_images=True,trace_hashes=traces,
                 independent_holdout_confirmation=False,checkpoint_selected_by_real_eval=False,
                 configuration_selected_on_reused_DEV=True,tail_risk_must_be_reported=True,artifacts=all_bindings)
    C.freeze(R.DOC/'RECOVERY_AUDIT.json',summary)
    lines=['# Self-training recovery — frozen detector, pose-only adaptation','',
        'The user goal is improved self-training, not merely finding a favourable diagnostic. All completed attempts are retained below. No existing release is replaced.','',
        '## Fixed population and method','',
        'Ordinary plastic194 and NEG2689. R0 initialization; freeze backbone, neck, box/class branches and all running buffers; train only pose branches/flow. Same217 unique pseudo images from249 candidates,512 real+512 synthetic exposures per epoch,5epochs/320updates. No IoU selection, no manual review masks, no inference-time refiner. RAW/REF targets have matched trusted-keypoint masks. Existing Replay teacher had prior manual9-image/38-corner supervision; this is not a claim of never using real supervision anywhere in the teacher pipeline.','',
        'Six predeclared source/raw/refined × learning-rate screens were followed by two actual order replicates of the selected1e-5 setting, with matched source/raw controls. The loader sorts paths and uses a fixed RNG generator, so indexed aliases ensure orders43/44 truly differ. Same data membership/multiplicities; first-batch image hashes verify RAW/REF pairing and distinct repeats.','',
        '## All attempts','', '| Phase | Arm | PCK20 % | median8 px | IoU3D | t cm | ADDsym |','|---|---|---:|---:|---:|---:|---:|']
    for phase,arms in grid.items():
        for arm,m in arms.items():lines.append(f"| {phase} | {arm} | {m['PCK20']:.3f} | {m['median8']:.3f} | {m['iou3d']:.5f} | {m['t_cm']:.3f} | {m['ADDsym']:.5f} |")
    lines+=['','## Selected method: all three runs, not best repeat','',
        '| Model | PCK20 % | paper kp median px | paper kp P90 px | AP50–95 | FPR95 | IoU3D | ADDsym |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    b=baselines['R0'];d=b['detection']['report']['metrics']['box_and_keypoint_2d']
    lines.append(f"| R0 | {100*base_sym['PCK']['20']:.3f} | {d['keypoint_location_median_px']:.3f} | {d['keypoint_location_p90_px']:.3f} | {d['box_ap50_95']:.4f} | {b['detection']['ranking']['fpr95']:.4f} | {b['pose']['iou3d_median']:.5f} | {b['pose']['add_sym_auc']:.5f} |")
    for r in runs:lines.append(f"| {r['arm']} | {r['pck20']:.3f} | {r['kp_median']:.3f} | {r['kp_p90']:.3f} | {r['ap5095']:.4f} | {r['fpr95']:.4f} | {r['pose']['iou3d_median']:.5f} | {r['pose']['add_sym_auc']:.5f} |")
    lines+=['','## Uncertainty and limits','',
        'The194 images are repeatedly used DEV, not independent test data. Session-clustered bootstrap intervals below are exploratory and not adjusted for the search. Seed/order repeatability is not new image-level validation. MAIN6D uses the frozen geometry-reconstructed reference, not externally measured ground-truth pose. PCK20 uses whole-object symmetry and8corners, while paper median/P90 uses fixed9-keypoint supervision on matched detections.','']
    for r in runs:
        lines.append(f"- {r['arm']}: primary/control flags {r['flags']}; clustered PCK20 contrasts {r['contrasts']}")
    lines+=['','No assertion of uniformly improved tails or all viewpoints. P90 and subgroup harms must be disclosed even if primary central-accuracy metrics improve. No automatic final-model promotion.','']
    lines+=['## Evaluation provenance correction','',
        f'The full194 evaluation includes {len(overlap)} images previously used for manual supervision of the frozen Replay teacher: {overlap}. The student did not directly train on these labels, but the complete teacher/student pipeline is not independent of them. Preserve full194 results as exploratory development evidence, not independent test results.','',
        'Excluding exact teacher images or their entire overlapping session below is a posthoc sensitivity check, not a newly independent holdout and not a replacement of the original primary population.','',
        '```json',json.dumps(sensitivity,ensure_ascii=False,indent=2),'```','']
    C.write_text(R.RAW/'RECOVERY_REPORT.md','\n'.join(lines))
    print(json.dumps(dict(runs=runs,all_flags=summary['all_repeat_primary_and_control_flags_pass']),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
