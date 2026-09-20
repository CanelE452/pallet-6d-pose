"""Recompute camera features/decisions/metrics and audit convention transfer."""
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from . import facing_rank as V
from scripts.research.pallet_dim_conditioned_p_v1 import eval_math as EM

I=V.I; C=V.C


def main():
    tests=[C.HERE/p for p in ['test_facing_rank_features.py','test_identity_rank_calibration.py',
        'test_identity_rank_linear.py','test_identity_rank_hard.py']]
    run=subprocess.run([sys.executable,'-m','pytest','-q',*map(str,tests)],capture_output=True,text=True)
    print(run.stdout,flush=True);assert run.returncode==0,run.stderr
    passed=int(re.search(r'(\d+) passed',run.stdout).group(1));assert passed==10
    protocol=V.verify()
    for file in ['INPUTS_LOCK.json','OUTPUTS_LOCK.json']:
        for b in C.read(V.DOC/file)['artifacts']: C.verify(b)
    for b in C.read(V.DOC/'DECISION_LOCK.json')['fits'].values(): C.verify(b)
    data=I.SourceData().data; arr=data.arrays; affine=C.N.E.old('features').canvas_affine
    old=np.load(V.L.H.RAW/'source.npz');s=np.load(V.RAW/'source.npz')
    real=np.load(V.RAW/'real.npz');old_real=np.load(V.L.H.RAW/'real.npz')
    for a,b in [(old,s),(old_real,real)]:
        for key in ['errors','y','eligible','train']: np.testing.assert_array_equal(a[key],b[key])
    ideal=[];source_meta=C.read(V.RAW/'SOURCE_METADATA.json')
    with ExitStack() as stack:
        archives={}
        for idx,r in enumerate(source_meta):
            row=r['row'];meta=data.source['records'][int(data.indices[row])]
            assert r['id']==meta['id'] and r['points_sha256']==I.N.array_sha(arr['points'][row])
            gain,offset=affine(meta['prepared_shape_hw'],arr['input_shape'][row])
            q=(np.asarray(arr['points'][row],float)-offset)/gain-meta['reflect_pad_px']
            x,d=V.F.describe(q,r['K']);np.testing.assert_allclose(x,s['x'][idx],atol=1e-7)
            np.testing.assert_allclose(V.F.describe(q[I.F.HALF],r['K'])[0],x,atol=1e-5)
            np.testing.assert_allclose(V.F.describe(q[I.F.QUARTER],r['K'])[0],-x,atol=1e-5)
            loc=r['locator']
            if '::' in loc:
                a,m=loc.split('::',1)
                if a not in archives: archives[a]=stack.enter_context(zipfile.ZipFile(a))
                payload=archives[a].read(m)
            else: payload=(C.ROOT/loc).read_bytes()
            assert hashlib.sha256(payload).hexdigest()==r['raw_sha256']
            annotation=json.loads(payload)
            np.testing.assert_array_equal(V.F.camera_matrix(annotation['camera_data']['intrinsics']),r['K'])
            obj=annotation['objects'][0]
            points=np.vstack([obj['projected_cuboid'],obj['projected_cuboid_centroid']])
            _,diag=V.F.describe(points,r['K'])
            ideal.append(dict(id=r['id'],diagnostic=diag))
    assert len(ideal)==2062
    for arm in I.ARMS:
        fit=C.read(V.DOC/f'FIT_{arm}.json');C.verify(fit['checkpoint']);m=np.load(C.ROOT/fit['checkpoint']['path'])
        val=~s['train'];prob=V.L.probability(V.L.normalize(s['x'][val],m['scale']),m['weight'])
        assert I.calibrate(prob,s['errors'][val],s['y'][val],s['eligible'][val])==fit['calibration']
    cameras={r['id']:r['K'] for r in C.read(V.RAW/'EVAL_CAMERAS.json')}
    decisions={r['id']:r for r in C.read(V.RAW/'DECISIONS.json')}
    reference={r['id']:r for r in C.read(I.R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records'] if r['kind']=='PLASTIC'}
    baseline={r['id']:r for r in C.read(I.R.BASE_RAW/'EVAL_METRICS.json')['R0'] if r['kind']=='PLASTIC'}
    pe,pop=I.R.E.O.population_metadata()
    targets={item.frame_id:pe.E._legacy_forbidden_target(item) for item,meta in pop if item.frame_id in reference}
    groups=C.read(C.N.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects']
    perms=next(r['permutations'] for r in groups if r['object_type']==C.TYPES['PLASTIC'])
    checked=0;changed={}
    for arm in I.ARMS:
        f=C.read(V.DOC/f'FIT_{arm}.json');m=np.load(C.ROOT/f['checkpoint']['path'])
        records=C.read(V.RAW/f'EVAL_PREDICTIONS_{arm}.json')['records']
        metrics={r['id']:r for r in C.read(V.RAW/f'SCREEN_{arm}.json')['metrics']}
        assert len(records)==194 and {r['id'] for r in records}==set(reference)
        changed[arm]=0
        for r in records:
            key=r['id'];prior=reference[key];q=np.asarray(I.top(prior['prediction'])['keypoints_xy'])
            x,_=V.F.describe(q,cameras[key]);d=decisions[key]
            np.testing.assert_array_equal(x,np.asarray(d['features'],np.float32))
            prob=V.L.probability(V.L.normalize(x,m['scale']),m['weight'])
            np.testing.assert_allclose(prob,d['choices'][arm]['probabilities'],atol=1e-12)
            choice=I.F.choose(prob,f['calibration']['threshold']);assert choice==d['choices'][arm]['choice']
            n=np.asarray(I.top(r['prediction'])['keypoints_xy'])
            np.testing.assert_array_equal(n,q[I.F.QUARTER] if choice else q)
            assert sorted(map(tuple,n[:8]))==sorted(map(tuple,q[:8]))
            I.assert_preserved(prior['prediction'],r['prediction'])
            t=targets[key];b=baseline[key]
            measured=EM.measure(n,t.keypoints_xy,t.keypoint_supervision_mask,perms,r['raw_hw'],b['matched'],b['detected'])
            np.testing.assert_allclose(np.asarray(measured['canonical_errors'],float),np.asarray(metrics[key]['canonical_errors'],float),atol=1e-9,rtol=0,equal_nan=True)
            changed[arm]+=choice;checked+=1
    # GT is used below only AFTER locked output evaluation; no new decisions follow.
    real_diag=[]
    for key,prior in reference.items():
        t=targets[key];q=np.asarray(t.keypoints_xy);valid=np.asarray(t.keypoint_supervision_mask,bool)
        if not valid[:8].all():
            real_diag.append(dict(id=key,complete=False));continue
        _,gt=V.F.describe(q,cameras[key])
        raw=I.top(prior['prediction'])['keypoints_xy'];_,pred=V.F.describe(raw,cameras[key])
        errors=I.pair_errors(raw,q,valid)
        real_diag.append(dict(id=key,complete=True,GT_geometry=gt,R0_geometry=pred,
            GT_diagnostic_quarter_mean_improvement_px=float(np.nanmean(errors[0])-np.nanmean(errors[1]))))
    complete=[r for r in real_diag if r['complete'] and r['GT_geometry']['available']]
    strong=[r for r in complete if r['GT_diagnostic_quarter_mean_improvement_px']>20]
    diagnostic=dict(status='POSTHOC_GT_DIAGNOSTIC_NOT_SELECTOR_OR_PERFORMANCE',
        source_ideal_count=len(ideal),source_ideal_available=sum(r['diagnostic']['available'] for r in ideal),
        source_ideal_negative_margin=sum(r['diagnostic'].get('facing_margin',0)<-1e-6 for r in ideal),
        real_total=194,real_complete=len(complete),real_incomplete=194-len(complete),
        real_GT_negative_margin=sum(r['GT_geometry']['facing_margin']<0 for r in complete),
        quarter_improvement_over20_frames=len(strong),
        these_frames_GT_negative_margin=sum(r['GT_geometry']['facing_margin']<0 for r in strong),
        these_frames_R0_negative_margin=sum(r['R0_geometry']['facing_margin']<0 for r in strong),
        warning='2D GT projections and camera intrinsics are not independent physical3D truth. Negative margin may reflect localization/calibration/convention; do not relabel, remove frames or claim a proven annotation bug.',
        rows=real_diag,source_ideal_rows=ideal,GT_selected_outputs_for_training=False,
        evidence=[C.bound(V.DOC/'OUTPUTS_LOCK.json'),C.bound(__file__)])
    C.freeze(V.DOC/'CONVENTION_TRANSFER_DIAGNOSTIC.json',diagnostic)
    result=C.read(V.DOC/'RESULTS.json');assert result['changed_frames']==changed
    C.freeze(V.DOC/'COMPLETION_AUDIT.json',dict(experiment_complete=True,goal_complete=False,
        regression_tests_passed=passed,source_features_raw_provenance_group_checks=2062,
        source_and_real_supervision_identical_to_prior=True,source_calibration_recomputed=True,
        real_feature_decision_preservation_metric_checks=checked,changed_frames=changed,
        physical_point_set_unchanged=True,frame_count_unchanged=True,new_annotations=0,new_tags=0,
        real_GT_for_diagnostics_only=True,auto_promoted=False,
        evidence=[C.bound(V.DOC/p) for p in ['PROTOCOL.json','RESULTS.json','OUTPUTS_LOCK.json','CONVENTION_TRANSFER_DIAGNOSTIC.json']]
            +[C.bound(__file__)]+[C.bound(p) for p in tests]))
    print('FACING_AUDIT_COMPLETE',checked,changed,flush=True)
    print(json.dumps({k:v for k,v in diagnostic.items() if k not in ['rows','source_ideal_rows','evidence']},indent=2),flush=True)


if __name__=='__main__':main()
