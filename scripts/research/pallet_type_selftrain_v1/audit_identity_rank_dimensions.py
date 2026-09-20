"""Independent output-contract and metric parity audit for both linear screens."""
import re
import subprocess
import sys
import numpy as np
from . import identity_rank_dimensions as D
from scripts.research.pallet_dim_conditioned_p_v1 import eval_math as EM


def main():
    I=D.I;C=D.C
    tests=[str(C.ROOT/'scripts/research/pallet_type_selftrain_v1'/name) for name in
        ['test_identity_rank.py','test_identity_rank_calibration.py','test_identity_rank_hard.py',
         'test_identity_rank_linear.py','test_identity_rank_dimensions.py']]
    run=subprocess.run([sys.executable,'-m','pytest','-q',*tests],capture_output=True,text=True)
    print(run.stdout,flush=True);assert run.returncode==0,run.stderr
    count=int(re.search(r'(\d+) passed',run.stdout).group(1));assert count==18
    reference={r['id']:r for r in C.read(I.R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records'] if r['kind']=='PLASTIC'}
    baseline={r['id']:r for r in C.read(I.R.BASE_RAW/'EVAL_METRICS.json')['R0'] if r['kind']=='PLASTIC'}
    pe,pop=I.R.E.O.population_metadata()
    targets={item.frame_id:pe.E._legacy_forbidden_target(item) for item,meta in pop if item.frame_id in reference}
    groups=C.read(C.N.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects']
    perms=next(r['permutations'] for r in groups if r['object_type']==C.TYPES['PLASTIC'])
    for key,r in reference.items():
        t=targets[key];b=baseline[key]
        m=EM.measure(I.top(r['prediction'])['keypoints_xy'],t.keypoints_xy,t.keypoint_supervision_mask,perms,r['raw_hw'],b['matched'],b['detected'])
        np.testing.assert_allclose(np.asarray(m['canonical_errors'],float),np.asarray(b['canonical_errors'],float),atol=1e-9,rtol=0,equal_nan=True)
    for mod in [D.L,D]:
        p=mod.verify()
        for name in ['DECISION_LOCK.json','OUTPUTS_LOCK.json']:
            lock=C.read(mod.DOC/name)
            for binding in lock.get('artifacts',[]):C.verify(binding)
            for binding in lock.get('fits',{}).values():C.verify(binding)
        decisions={r['id']:r for r in C.read(mod.RAW/'DECISIONS.json')}
        result=C.read(mod.DOC/'RESULTS.json');checked=0;changed={}
        for arm in I.ARMS:
            rows=C.read(mod.RAW/f'EVAL_PREDICTIONS_{arm}.json')['records']
            saved={r['id']:r for r in C.read(mod.RAW/f'SCREEN_{arm}.json')['metrics']}
            assert len(rows)==194 and {r['id'] for r in rows}==set(reference)
            changed[arm]=0
            for r in rows:
                old=reference[r['id']];q=np.asarray(I.top(old['prediction'])['keypoints_xy'])
                n=np.asarray(I.top(r['prediction'])['keypoints_xy'])
                choice=decisions[r['id']]['choices'][arm]['choice']
                np.testing.assert_array_equal(n,q[I.F.QUARTER] if choice else q)
                # Same physical set; no claim that a new pixel location was found.
                assert sorted(map(tuple,q[:8]))==sorted(map(tuple,n[:8]))
                I.assert_preserved(old['prediction'],r['prediction'])
                t=targets[r['id']];b=baseline[r['id']];s=saved[r['id']]
                assert s['matched']==b['matched'] and s['detected']==b['detected']
                m=EM.measure(n,t.keypoints_xy,t.keypoint_supervision_mask,perms,r['raw_hw'],b['matched'],b['detected'])
                np.testing.assert_allclose(np.asarray(m['canonical_errors'],float),np.asarray(s['canonical_errors'],float),atol=1e-9,rtol=0,equal_nan=True)
                changed[arm]+=int(choice);checked+=1
        assert changed==result['changed_frames']
        assert not any(result['passed'].values()) and not result['goal_complete'] and not result['auto_promoted']
        C.freeze(mod.DOC/'COMPLETION_AUDIT.json',dict(experiment_complete=True,goal_complete=False,
            regression_tests_passed=count,baseline_metric_parity_frames=194,
            prediction_preservation_and_metric_parity_checks=checked,changed_frames=changed,
            physical_point_set_unchanged=True,center_boxes_scores_confidence_unchanged=True,
            source_bindings_verified=len(p['sources']),failed_real_screen=result['passed'],
            new_annotations=0,new_tags=0,auto_promoted=False,
            evidence=[C.bound(__file__),C.bound(mod.DOC/'RESULTS.json'),C.bound(mod.DOC/'PROTOCOL.json'),
                C.bound(mod.DOC/'OUTPUTS_LOCK.json')]+[C.bound(t) for t in tests]))
        print('OUTPUT_AUDIT_COMPLETE',mod.PHASE,checked,changed,flush=True)


if __name__=='__main__':main()
