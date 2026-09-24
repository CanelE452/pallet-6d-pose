"""Read-only regression checks of frozen inputs, metrics, and report delivery."""
import ast
import json
import re
from pathlib import Path
import numpy as np
import yaml
import cv2
from . import common as C

def main():
    tests={}
    def check(name,value):
        tests[name]=bool(value)
        assert value,name
    bindings=C.read(C.DOC/'INPUT_BINDINGS.json')
    check('test_current_head_recorded',len(bindings['head_before'])==40 and bindings['branch']=='main')
    count=C.immutable()
    check('test_input_hashes',count>5120)
    check('test_existing_artifacts_unchanged',True)  # all bound historical files/cache above re-hashed
    role=C.read(C.DOC/'H10_ROLE_PREVALENCE.json');pair=C.read(C.DOC/'PAIR_INTEGRITY.json');split=C.read(C.DOC/'RECORDING_DISJOINT_AUDIT.json')
    target=[r for r in C.read(C.E.RAW/'TARGETS.json') if r['role']=='H']
    check('test_H10_exact',{r['id'] for r in role['rows']}=={r['id'] for r in target} and len(target)==10)
    check('test_whole_C4_only',all(len(m['all_whole_C4'])==4 for r in role['rows'] for m in r['models'].values()))
    check('test_no_free_matching',role['no_free_matching'])
    # Verify every H annotation against the pre-existing immutable lock, not just a new snapshot.
    old_bind={b['path']:b for b in C.read(C.E.DOC/'INPUT_LOCK.json')['files']}
    auxiliary=[]
    for r in target:
        for k in ('image','annotation'):
            assert r[k]['sha256']==old_bind[r[k]['path']]['sha256'];C.verify(r[k]);auxiliary.append(r[k])
    check('test_no_annotation_write',True)
    for key in ('same_init','same_train_images','same_targets','same_masks','same_source_replay','same_order','same_optimizer','same_lr','same_seed','same_updates','same_base_RGB'):
        check('test_S0_S1_'+key,pair[key])
    check('test_only_random_occlusion_difference',pair['only_random_occlusion_difference'] and pair['actual_trace_cache_verified']==5120)
    plans=[r for r in map(json.loads,(C.E.C.DOC/'AUGMENTATION_PLAN.jsonl').read_text().splitlines()) if r['material']=='PLASTIC']
    targets=C.read(C.E.C.DOC/'TARGETS.json')
    real_indices={r['target_index'] for r in plans if r['real']}
    assert {targets[i]['id'] for i in real_indices}==set(split['train_ids'])
    # Resolve actual loader images to their native recording-bound RGB, not names alone.
    seen=set();max_coordinate_error=0.
    for r in plans:
        if not r['real']:continue
        t=targets[r['target_index']]
        if r['image'] not in seen:
            expected=cv2.copyMakeBorder(cv2.imread(str(C.ROOT/t['image']['path'])),100,100,100,100,cv2.BORDER_REFLECT_101)
            assert np.array_equal(cv2.imread(r['image']),expected)
            seen.add(r['image'])
        with np.load(C.ROOT/r['cache']['path']) as z: kp=z['keypoints'][0]
        transformed=(np.c_[np.array(t['target']),np.ones(9)]@np.array(r['native_to_model']).T)[:,:2]
        error=float(np.max(np.abs(np.clip(transformed,0,640)-kp[:,:2]*640)))
        assert error<.001
        max_coordinate_error=max(max_coordinate_error,error)
        mask=np.array(t['mask'],bool)
        assert (kp[~mask,2]==1).all() and not (kp[~mask,2]==2).any()
    check('test_actual_train_RGB_recording_binding',len(seen)==10)
    check('test_real_target_affine_and_ignore_mask',max_coordinate_error<.001)
    args={}
    for a in ('S0','S1'):
        p=C.E.C.RAW/f'runs/PLASTIC_{a}/args.yaml';args[a]=yaml.safe_load(p.read_text());auxiliary.append(C.bind(p))
    diff={k:[args['S0'].get(k),args['S1'].get(k)] for k in set(args['S0'])|set(args['S1']) if args['S0'].get(k)!=args['S1'].get(k)}
    check('test_runtime_training_args',set(diff)<={'name','save_dir'})
    for k,v in pair['args'].items():
        assert args['S0'][k]==v,(k,args['S0'][k],v)
    C.save(C.DOC/'PAIR_RUNTIME_ARGS.json',dict(passed=True,bindings=auxiliary[-2:],differences=diff,only_output_paths_differ=True,
        actual_train_RGB_matches_native_reflect100=True,real_indices=sorted(real_indices),max_affine_target_error_px=max_coordinate_error))
    # Same synthetic source manifest and untouched source references.
    sourceplan=C.read(C.E.C.DOC/'SOURCE_PROBE_PLAN.json')['records']
    assert len(sourceplan)==256
    for r in sourceplan:
        for k in ('image','label'):
            C.verify(r[k]);auxiliary.append(r[k])
    sourceids=[r['id'] for r in sourceplan]
    for a in ('S0','S1'):
        dd=C.read(C.E.C.RAW/f'DIAGNOSTICS_PLASTIC_{a}.json')
        assert [r['id'] for r in dd['source']]==sourceids
    check('test_source256_fixed',True)
    for prefix in ('train','heldout'):
        check(f'test_{prefix}_recordings_resolved',len(split[prefix+'_recordings'])>0)
    check('test_recording_intersection_empty',not split['recording_intersection'])
    check('test_image_sha_intersection_empty',not split['image_sha_intersection'])
    check('test_existing_near_duplicate_rule',not split['near_duplicate']['flagged'])
    check('test_heldout_exact',split['heldout_ids']==[r['id'] for r in C.records()] and split['heldout_frames']==128)
    raw=C.read(C.DOC/'PREDICTIONS_LOCK.json');pose=C.read(C.DOC/'POSE_PREDICTIONS_LOCK.json');start=C.read(C.RAW/'SCORING_START.json')
    for b in (raw['file'],pose['file'],pose['selector'],pose['solver'],start['raw_lock'],start['pose_lock']):C.verify(b)
    check('test_prediction_freeze_before_scoring',raw['created_at']<=pose['created_at']<start['created_at'])
    check('test_same_inference_args',raw['args']==dict(conf=.001,imgsz=640,rect=True,padding=100,padding_mode='BORDER_REFLECT_101',augment=False,half=False,selected='argmax confidence'))
    src=(Path(__file__).parent/'infer.py').read_text();tree=ast.parse(src)
    check('test_no_GT_in_inference',not raw['GT_input'] and not pose['GT_input'] and all(word not in src for word in ('TRUTH_FOR_DISPLAY','VERIFIED_LABELS','metadata(\'REAL_DEV\')','gt[')))
    check('test_production_D9_unchanged',pose['production_D9_unchanged'])
    results=C.read(C.DOC/'RESULTS.json');tt=C.read(C.DOC/'TRANSITIONS.json')
    expected={'ALL':128,**{k:split['counts'][s] for k,s in C.SEVS.items()}}
    for g,n in expected.items():
        for a in C.ARMS:
            r=results['groups'][g][a]
            assert r['current']['frames']==n and r['twoD']['total_frames']==n
            for k,c in r['twoD']['correct'].items():
                assert np.isclose(c/r['twoD']['corners'],r['twoD']['PCK'][k])
        a,b=results['groups'][g]['S0'],results['groups'][g]['S1'];t=tt[g]['canonical_GT_identity_aligned']
        assert b['twoD']['correct']['10']-a['twoD']['correct']['10']==t['gained_correct10']-t['lost_correct10']
        assert tt[g]['canonical_GT_identity_aligned']==tt[g]['native_fixed_ID_no_symmetry']
    check('test_severity_denominators',expected['ALL']==sum(expected[g] for g in C.SEVS))
    check('test_same_ID_transition_reconciliation',True)
    pm=C.read(C.RAW/'POSE_METRICS.json')
    for arm,rows in pm.items():
        for fid,r in rows.items():
            candidates=[h for h in r['hypotheses'] if h['metric']['available']]
            if candidates:
                chosen=min(candidates,key=lambda h:(h['metric']['ADDsym_normalized'],h['name']))
                assert chosen['name']==r['oracle_name'] and chosen['metric']==r['oracle']
    check('test_oracle_posthoc_only','NONDEPLOYABLE' in results['oracle_label'])
    anchor=C.read(C.DOC/'VERIFIED_ANCHOR_TRANSFER.json');C.verify(anchor['reference_binding'])
    check('test_verified_anchor_final_version',anchor['reference']=='VERIFIED_VISIBLE_ANCHOR_FINAL_V2' and anchor['points']==66)
    check('test_fixed_identity_verified',anchor['fixed_identity'] and not anchor['symmetry_remapping'])
    history=C.read(C.DOC/'HISTORICAL_DIRECTION.json')
    check('test_historical_not_mixed_with_new_denominators',history['historical_only'] and history['groups']['ALL']['historical_frames']==300 and history['groups']['ALL']['heldout_frames']==128)
    repro=C.read(C.DOC/'REPRODUCTION_STATUS.json')
    check('test_new_training_steps_zero',repro['new_training_steps']==repro['reproduction_fits']==0 and repro['checkpoint_reused'])
    case=C.read(C.DOC/'CASE_MANIFEST.json');check('test_case_counts',len(case['cases'])==22 and len(case['random_selected'])==6)
    import hashlib
    assert case['random_selected']==sorted(split['heldout_ids'],key=lambda i:hashlib.sha256(('recording-disjoint-v1|'+i).encode()).hexdigest())[:6]
    check('test_random_control_not_output_selected',True)
    md=(C.DOC/'REPORT_KO.md').read_text();images=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',md)
    check('test_report_figures_exist',len(images)>=33 and all((C.DOC/p).is_file() for p in images))
    check('test_single_next_design_only',C.read(C.DOC/'TRANSFER_DECISION.json')['next_one_experiment']['design_only'])
    C.save(C.DOC/'AUXILIARY_BINDINGS.json',dict(files=auxiliary,code=[C.bind(p) for p in sorted(Path(__file__).parent.glob('*.py'))]))
    C.save(C.DOC/'AUDIT.json',dict(status='PASS',passed=len(tests),tests=tests,input_hashes_checked=count,
        historical_unchanged=True,images=len(images),created_at=C.now(),source_training_zero=True))
    print('AUDIT_PASS',len(tests),'input_hashes',count,'images',len(images),flush=True)

if __name__=='__main__':main()
