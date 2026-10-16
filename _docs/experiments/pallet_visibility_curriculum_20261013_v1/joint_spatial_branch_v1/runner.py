"""Six CAL-only combinations of frozen joint RGB observations and W/D branches."""
from __future__ import annotations

import argparse
import builtins
from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy
import io
from itertools import product
from pathlib import Path
import time

import cv2
import numpy as np

from code_snapshot import common as C
from code_snapshot.branch_curriculum import solve_branch_curriculum
from code_snapshot.scoring import pose_metric, distribution, paired, rotation_y


HERE=Path(__file__).resolve().parent
SCRATCH=HERE.parent
JOINT=SCRATCH/'cal_joint_spatial_obs_v1/pose_probe_v1'
PRIMARY=SCRATCH/'cal_segmentation_v2'
FAMILIES=('RGB26_F4F8_PCA32_CORRECTED','RGB26_F4F8_PCA64_CORRECTED')
ARMS={f'{family}_WD_M{margin:g}':dict(source_joint_arm=family,selection_margin_px2=margin)
      for family,margin in product(FAMILIES,(.5,1.,2.))}


def binding(path):
    path=Path(path)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=C.sha(path))


@contextmanager
def no_scoring_reads():
    originals=builtins.open,io.open,Path.open
    blocked=('ORACLE','TARGETS.json','SCORING_METADATA','SCORED_ROWS','MODEL_PHASE_AUDIT',
             'OOF_OBSERVATIONS.json','LABEL_AUDIT.json','RESULTS.json','cal_branch_v1')
    def wrap(fn):
        def guard(path,*args,**kwargs):
            if any(token in str(path) for token in blocked):
                raise AssertionError('Inference attempted scoring read: '+str(path))
            return fn(path,*args,**kwargs)
        return guard
    builtins.open,io.open,Path.open=map(wrap,originals)
    try:yield
    finally:builtins.open,io.open,Path.open=originals


def joint_appearance(primary,head):
    app=deepcopy(primary['appearance']);threshold=head['threshold']
    for name,p in head['head_probabilities'].items():
        k=int(name)
        if app['states'][k] in ('OUT_OF_FRAME','SELF_OCCLUDED'):
            raise AssertionError('Joint score appeared at a fixed excluded state')
        app['visibility_score'][k]=p
        if p<threshold:app['states'][k]='UNKNOWN'
        else:
            app['states'][k]='VISIBLE_BOUNDARY' if app['boundary_score'][k]>=.35 else 'VISIBLE_INTERIOR'
            app['local_accuracy_score'][k]=max(app['local_accuracy_score'][k],.12)
    return app


def declare():
    if (HERE/'PROTOCOL.json').exists():raise FileExistsError('Preserve declared branch experiment')
    jointprotocol=C.read(JOINT/'PROTOCOL.json')
    selected={r['name']:r for r in jointprotocol['arms'] if r['name'] in FAMILIES}
    assert len(selected)==2 and all(r['threshold']==.45 for r in selected.values())
    paths=[Path(__file__),SCRATCH/'CAL_INPUTS.json',JOINT/'PREDICTIONS.jsonl.gz',
           JOINT/'PREDICTION_SEAL.json',JOINT/'PROTOCOL.json',PRIMARY/'PREDICTIONS.jsonl.gz',
           PRIMARY/'PREDICTION_SEAL.json']+sorted((HERE/'code_snapshot').glob('*.py'))
    C.write(HERE/'PROTOCOL.json',dict(schema='joint_spatial_OOF_WD_branch_CAL_v1',frames=76,arms=ARMS,
        source_joint_arms=selected,registered_dimensions='frame.xyz from isolated prediction inputs only',
        observations='Existing sealed fitted_measurements from joint-spatial OOF RGB coordinate regressor; original native channels',
        original_H_fixed=True,head_usable_probability_frozen=True,usable_threshold=.45,
        common_support='Same q, same H-excluded visible pool, same confidence/accuracy weights for both W/D models',
        alternate_initializer='Unchanged predecessor all finite predicted q SQPnP+LM, including self-hidden/out-of-frame predictions if finite; only initialization, never projections',
        final_fit_and_branch_cost_exclude_known_self_hidden=True,minimum_shared_visible_support=4,
        source_algorithm='Previous branch_curriculum.py unchanged except frozen local import adapter',
        source_branch_algorithm_binding=C.read(HERE/'code_snapshot/BRANCH_IMPORT_ADAPTER.json'),
        margin_grid_px2=[.5,1.,2.],threshold_or_regressor_retuning=0,
        initial_pose_physical_Q='Original R_cf.T@R_physical preserved if original family retained; alternate registered proper Q from predecessor',
        original_branch_expected='Exact parity with each source joint-spatial single-family solver',
        no_projected_corners_as_observations=True,no_GT_for_selection=True,labels_only_after456_predictions_sealed=True,
        scoring_reference='Postseal native physical model phase audit against isolated CAL76 annotation; expected initial63consistent/13inconsistent',
        current_default_API_unchanged=True,TEST_reads=0,FINAL_reads=0,
        limitations=['Joint heads/regressors selected on reused CAL LOGO; this combination is exploratory development.',
                    'The alternate initializer retains the predecessor all-finite detector observation contract.',
                    'Fixed initial H can itself be inaccurate; no oracle hidden-state replacement.',
                    'Branch common cost can prefer a false family when corrected observed coordinates are still inaccurate.',
                    'Reference pose is geometric reconstruction, not an independently measured sensor pose.'],
        bindings=[binding(path) for path in paths]))


def infer():
    if (HERE/'PREDICTIONS.jsonl.gz').exists():raise FileExistsError('Preserve completed branch predictions')
    protocol=C.read(HERE/'PROTOCOL.json')
    for row in protocol['bindings']:
        if binding(row['path'])!=row:raise AssertionError('Frozen prediction dependency differs')
    source_seal=C.read(JOINT/'PREDICTION_SEAL.json')
    if not source_seal['before_pose_scoring'] or C.sha(JOINT/'PREDICTIONS.jsonl.gz')!=source_seal['sha256']:
        raise AssertionError('Joint observations must already be sealed')
    primaryseal=C.read(PRIMARY/'PREDICTION_SEAL.json')
    if not primaryseal['before_scoring'] or C.sha(PRIMARY/'PREDICTIONS.jsonl.gz')!=primaryseal['sha256']:
        raise AssertionError('Primary appearance must be sealed')
    cv2.setNumThreads(1);begin=time.monotonic();records=[]
    with no_scoring_reads():
        frames=C.read(SCRATCH/'CAL_INPUTS.json')['frames']
        if len(frames)!=76 or any(r['partition']!='CAL' or r['population_role']!='DEV' for r in frames):
            raise AssertionError('Exact isolated CAL76 required')
        upstream={r['id']:r for r in C.rows(PRIMARY/'PREDICTIONS.jsonl.gz') if r['method']=='SEGMENTED_RGB_CURRICULUM'}
        joint={(r['id'],r['arm']):r for r in C.rows(JOINT/'PREDICTIONS.jsonl.gz') if r['arm'] in FAMILIES}
        assert len(upstream)==76 and len(joint)==152
        for index,frame in enumerate(frames):
            initial=frame['initial_poses']['N3_SUBPIX'];base=upstream[frame['id']]
            raw=np.asarray(frame['points']['N3_SUBPIX'],float)[:8]
            H=np.asarray(base['self_hidden'],bool)
            assert np.array_equal(H,C.self_hidden(C.cuboid(*initial['cf_extents']),initial['R_cf'],initial['centroid']))
            for arm,setting in ARMS.items():
                source=joint[(frame['id'],setting['source_joint_arm'])]
                assert np.array_equal(raw,np.asarray(source['original_points'])) and np.array_equal(H,source['initial_self_hidden'])
                q=np.asarray(source['fitted_measurements'],float);app=joint_appearance(base,source)
                config=dict(selection_margin_px2=setting['selection_margin_px2'],support_visibility_threshold=source['threshold'],
                            curriculum=dict(visible_threshold=source['threshold'],easy_visibility=max(.45,source['threshold'])))
                result=solve_branch_curriculum(q,np.asarray(frame['K']),np.asarray(frame['xyz']),initial,H,app,config)
                original=result['candidates'][0]
                for key in ['R','t']:
                    assert np.allclose(original['curriculum'][key],source['solution'][key],rtol=0,atol=1e-10),(frame['id'],arm,key)
                for candidate in result['candidates']:
                    if candidate['available']:
                        assert not any(H[k] for stage in candidate['curriculum']['stages'] for k in stage['used_ids'])
                        assert candidate['self_hidden_used']==H.tolist()
                assert np.array_equal(q,np.asarray(source['fitted_measurements']))
                records.append(dict(id=frame['id'],session=frame['session'],arm=arm,
                    source_joint_arm=setting['source_joint_arm'],source_joint_actual_pose=source['actual_pose'],
                    primary_actual_pose=base['actual_pose'],actual_pose=result['actual_pose'],solution=result['solution'],
                    branch_result=result,initial_pose=initial,registered_dimensions=frame['xyz'],
                    original_points=raw.tolist(),fitted_measurements=q.tolist(),initial_self_hidden=H.tolist(),
                    OOF_head_probabilities=source['head_probabilities'],original_family_parity=True,
                    common_pool_excludes_H=True,GT_input=False,TEST_accessed=False,FINAL_accessed=False))
            if (index+1)%16==0:print('JOINT_WD_CAL',index+1,round(time.monotonic()-begin,2),flush=True)
    assert len(records)==456
    for row in protocol['bindings']:
        if binding(row['path'])!=row:raise AssertionError('Frozen dependency changed during branch inference')
    C.save_rows(HERE/'PREDICTIONS.jsonl.gz',records)
    C.write(HERE/'PREDICTION_SEAL.json',dict(complete=True,before_scoring=True,rows=456,
        sha256=C.sha(HERE/'PREDICTIONS.jsonl.gz'),protocol_sha256=C.sha(HERE/'PROTOCOL.json'),
        GT_read_canary='PASS',TEST_reads=0,FINAL_reads=0,source_joint_parity_cases=456,
        original_q_H_unchanged=True,projected_points_used_as_observations=False,seconds=time.monotonic()-begin))


def phase_audit(frame,pose):
    mapping=np.asarray(frame['scoring_metadata']['native_to_reference_corner_ids'],int)
    truth=frame['target']['truth'];Q=np.asarray(pose['R_cf']).T@np.asarray(pose['R_physical'])
    # Right pose factor Q satisfies X_physical(row)=X_cf(row)@Q.
    physical=C.cuboid(*pose['cf_extents'])@Q
    referenceQ=np.asarray(truth['body_R']).T@np.asarray(truth['R'])
    reference=(C.cuboid(*truth['body_xyz'])@referenceQ)[mapping]
    gaps=[float(np.max(np.linalg.norm(physical@rotation_y(j*2*np.pi/truth['order']).T-reference,axis=1)))
          for j in range(truth['order'])]
    return dict(consistent=min(gaps)<1e-6,max_corner_gap_m=min(gaps),physical_from_cf_Q=Q.tolist(),
                cf_extents=pose['cf_extents'])


def summary(rows):
    return dict(frames=len(rows),accepted=sum(r['branch_result']['accepted'] for r in rows),
        flips=sum(r['branch_result']['branch_flip'] for r in rows),
        true_corrections=sum(not r['initial_phase']['consistent'] and r['selected_phase']['consistent'] for r in rows),
        false_flips=sum(r['initial_phase']['consistent'] and not r['selected_phase']['consistent'] for r in rows),
        initially_wrong=sum(not r['initial_phase']['consistent'] for r in rows),
        missed_wrong=sum(not r['initial_phase']['consistent'] and not r['selected_phase']['consistent'] for r in rows),
        missed_wrong_low_observation_count=sum(not r['initial_phase']['consistent'] and not r['selected_phase']['consistent']
            and len(r['branch_result']['common_visible_ids'])<4 for r in rows),
        selection_reasons=dict(Counter(r['branch_result']['selection_reason'] for r in rows)),
        common_pool_counts=dict(Counter(len(r['branch_result']['common_visible_ids']) for r in rows)),
        metrics={key:distribution([r['metric'][key] for r in rows]) for key in ('translation_cm','rotation_deg','ADDsym_cm')},
        success_count=sum(r['metric']['success_5cm_5deg'] for r in rows))


def score():
    if (HERE/'RESULTS.json').exists():raise FileExistsError('Preserve completed six-arm branch result')
    seal=C.read(HERE/'PREDICTION_SEAL.json')
    assert seal['before_scoring'] and C.sha(HERE/'PREDICTIONS.jsonl.gz')==seal['sha256']
    assert C.sha(HERE/'PROTOCOL.json')==seal['protocol_sha256']
    labels=C.read(SCRATCH/'ORACLE_CAL_INPUTS.json')['frames'];assert len(labels)==76 and all(r['partition']=='CAL' for r in labels)
    metadata={r['id']:r for r in labels};rows=list(C.rows(HERE/'PREDICTIONS.jsonl.gz'));groups=defaultdict(list)
    for row in rows:
        frame=metadata[row['id']];truth=frame['target']['truth']
        row['metric']=pose_metric(row['actual_pose'],truth)
        row['primary_metric']=pose_metric(row['primary_actual_pose'],truth)
        row['joint_metric']=pose_metric(row['source_joint_actual_pose'],truth)
        row['initial_phase']=phase_audit(frame,row['initial_pose']);row['selected_phase']=phase_audit(frame,row['actual_pose'])
        row['candidate_phase_audits']=[phase_audit(frame,dict(R_cf=c['R_cf'],R_physical=c['R_physical'],cf_extents=c['dimensions']))
            if c['available'] else None for c in row['branch_result']['candidates']]
        groups[row['arm']].append(row)
    assert len(groups)==6 and all(len(g)==76 for g in groups.values())
    initial_unique={r['id']:r['initial_phase'] for r in rows};counts=dict(Counter(v['consistent'] for v in initial_unique.values()))
    assert counts=={True:63,False:13},counts
    sessions=sorted({r['session'] for r in rows});reports={};comparisons={};per_session={}
    for name,group in groups.items():
        reports[name]=summary(group);per_session[name]={s:summary([r for r in group if r['session']==s]) for s in sessions}
        for reference_name,metric_field in [('PRIMARY_V1','primary_metric'),('OWN_JOINT_SINGLE_FAMILY','joint_metric')]:
            reference={r['id']:dict(id=r['id'],session=r['session'],metric=r[metric_field]) for r in group}
            comparisons[name+'/'+reference_name]=paired(group,reference)
    C.save_rows(HERE/'SCORED_ROWS.jsonl.gz',rows)
    result=dict(complete=True,schema='joint_spatial_WD_branch_CAL_results_v1',frames=76,arms=ARMS,
        initial_phase_reference=dict(consistent=63,inconsistent=13),summary=reports,per_session=per_session,
        paired=comparisons,phase_reference='Postseal native physical model corner consistency under proper registered symmetry; Q used as row-vector right factor',
        prediction_seal=binding(HERE/'PREDICTION_SEAL.json'),protocol=binding(HERE/'PROTOCOL.json'),
        original_single_family_parity=True,TEST_reads=0,FINAL_reads=0,integration_performed=False)
    C.write(HERE/'RESULTS.json',result)
    for name,item in reports.items():
        print(name,'flips',item['flips'],'correct',item['true_corrections'],'false',item['false_flips'],
            'missed',item['missed_wrong'],'lowObsMissed',item['missed_wrong_low_observation_count'],
            'T',round(item['metrics']['translation_cm']['mean'],4),'R',round(item['metrics']['rotation_deg']['mean'],4),
            'success',item['success_count'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('declare','infer','score'))
    stage=parser.parse_args().stage
    {'declare':declare,'infer':infer,'score':score}[stage]()
