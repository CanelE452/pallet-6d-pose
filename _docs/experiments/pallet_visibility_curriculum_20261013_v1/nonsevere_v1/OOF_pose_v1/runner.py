"""Current CAL46 RGB evidence with sealed fixed-parameter OOF head controls."""
import argparse
import builtins
from collections import Counter,defaultdict
from contextlib import contextmanager
from copy import deepcopy
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

import cv2
import numpy as np

ROOT=Path('/home/minjae/Documents/github/pallet-pose');sys.path.insert(0,str(ROOT))
from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C, joint_api as JA, segmentation as SG
from scripts.research.pallet_visibility_curriculum_20261013_v1.api import VisibilityCurriculum
from scripts.research.pallet_visibility_curriculum_20261013_v1.branch_curriculum import solve_branch_curriculum
from scripts.research.pallet_visibility_curriculum_20261013_v1.scoring import pose_metric
HERE=Path(__file__).resolve().parent;COHORT=HERE.parent;SOURCE=ROOT/'scripts/research/pallet_visibility_curriculum_20261013_v1'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(C.finite(value),indent=2,allow_nan=False)+'\n')
def rows(path):return [json.loads(l) for l in gzip.open(path,'rt') if l.strip()]


@contextmanager
def no_labels():
    originals=builtins.open,io.open,Path.open
    blocked=('ORACLE','SCORING_METADATA','TARGETS','TRAINING_LABELS','LABEL_AUDIT','OOF_DIAGNOSTICS','SCORED_ROWS','RESULTS.json','/annotations/')
    def wrap(fn):
        def guarded(path,*args,**kwargs):
            if any(v in str(path) for v in blocked):raise AssertionError('Inference scoring read '+str(path))
            return fn(path,*args,**kwargs)
        return guarded
    builtins.open,io.open,Path.open=map(wrap,originals)
    try:yield
    finally:builtins.open,io.open,Path.open=originals


def bindings(p):
    for path,digest in p['input_sha256'].items():assert sha(path)==digest,path
    for name,digest in p['source_sha256'].items():assert sha(SOURCE/name)==digest,name


def predictor(records,frame,no_delta):
    def call(image,q,H,initial,K,model):
        corners=np.asarray(q)[:8];h,w=image.shape[:2]
        eligible=np.isfinite(corners).all(1)&~np.all(corners==-1,axis=1)&~H&(corners[:,0]>=0)&(corners[:,0]<w)&(corners[:,1]>=0)&(corners[:,1]<h)
        probability=np.zeros(8);delta=np.zeros((8,2));output=np.array(q,copy=True);refined=[]
        for k in range(8):
            record=records.get((frame['id'],k));assert eligible[k]==(record is not None)
            if record is None:continue
            assert record['session']==frame['session'];probability[k]=record['P_usable'];delta[k]=record['RGB_delta_px']
            assert np.linalg.norm(delta[k])<=16+1e-8
            candidate=corners[k]+delta[k]
            if not no_delta and probability[k]>=.45 and 0<=candidate[0]<w and 0<=candidate[1]<h:output[k]=candidate;refined.append(k)
        return dict(P_usable=probability.tolist(),points=output.tolist(),RGB_delta_px=delta.tolist(),eligible=eligible.tolist(),
            threshold=.45,observed_refined_ids=refined,diagnostics=dict(source='fixed_CAL46_heldout_session_head',
                regression_applied=not no_delta,projected_points_used=False,GT_input=False))
    return call


def policy(together):
    def call(q,K,dimensions,initial,H,app,config):
        config=deepcopy(config)
        if together:config['curriculum']['staged_visible']=False
        return solve_branch_curriculum(q,K,dimensions,initial,H,app,config)
    return call


def infer():
    p=read(HERE/'PROTOCOL.json');assert not(HERE/'PREDICTIONS.jsonl.gz').exists();bindings(p)
    write(HERE/'RUNNER_BINDING.json',dict(runner_sha256=sha(__file__),protocol_sha256=sha(HERE/'PROTOCOL.json'),before_inference=True))
    result=[];cv2.setNumThreads(1)
    with no_labels():
        for name in ('ORACLE_CAL_INPUTS.json','CAL_SCORING_METADATA.json','joint_model/TRAINING_LABELS.json'):
            try:read(COHORT/name)
            except AssertionError:pass
            else:raise AssertionError('readcanary failed')
        frames=read(COHORT/'CAL_INPUTS.json')['frames'];assert len(frames)==46 and {f['id'] for f in frames}==set(p['frame_ids'])
        head_rows=read(COHORT/'joint_model/OOF_PREDICTIONS.json');assert all(set(r)=={'id','corner','session','P_usable','RGB_delta_px'} for r in head_rows)
        records={(r['id'],r['corner']):r for r in head_rows}
        assert read(COHORT/'joint_model/OOF_SEAL.json')['sha256']==sha(COHORT/'joint_model/OOF_PREDICTIONS.json')
        base=VisibilityCurriculum.from_config(COHORT/'CAL_VISIBILITY_CONFIG.json')
        candidate=JA.JointSpatialBranch.from_config(COHORT/'JOINT_CONFIG.json')
        real_segment=SG.segment
        for index,frame in enumerate(frames):
            assert frame['partition']=='CAL';image_path=ROOT/frame['image'];assert sha(image_path)==frame['image_sha256']
            image=cv2.imread(str(image_path));q=np.asarray(frame['points']['N3_SUBPIX'],float);q_before=q.copy()
            initial=deepcopy(frame['initial_poses']['N3_SUBPIX']);initial_before=deepcopy(initial);K=np.asarray(frame['K']);H=C.self_hidden(C.cuboid(*initial['cf_extents']),initial['R_cf'],initial['centroid'])
            probability=real_segment(image,candidate.segmentation_weights,candidate.device)
            with patch.object(SG,'segment',lambda *args,**kwargs:probability):
                visibility=base.refine(image,q,K,initial)
                result.append(dict(id=frame['id'],session=frame['session'],method='FRESH_CAL_VISIBILITY_CONTROL',
                    actual_pose=visibility['actual_pose'],solution=visibility['solution'],self_hidden=H.tolist(),
                    appearance=visibility['appearance'],observed_points=q.tolist(),corrected_measurements=q.tolist(),
                    fresh_visibility_classifier_scope='FULL_CAL46',GT_input=False))
                variants={}
                for method,together,no_delta in (('JOINT_OOF_STAGED',False,False),('JOINT_OOF_VISIBLE_TOGETHER',True,False),('JOINT_OOF_NO_DELTA',False,True)):
                    with patch.object(JA,'_predict_joint',predictor(records,frame,no_delta)),patch.object(JA,'solve_branch_curriculum',policy(together)):
                        value=candidate.refine(image,q,K,initial,frame['xyz'])
                    assert value['self_hidden']==H.tolist();assert all(not H[k] for c in value['branch_result']['candidates'] if c['available'] for s in c['curriculum']['stages'] for k in s['used_ids'])
                    assert np.array_equal(np.asarray(value['corrected_measurements'])[8:],q[8:])
                    if no_delta:assert np.array_equal(np.asarray(value['corrected_measurements']),q)
                    variants[method]=value
                    original=next(c for c in value['branch_result']['candidates'] if c['original'])
                    original_pose=dict(initial,R_cf=original['R_cf'],R_physical=original['R_physical'],centroid=original['centroid'],cf_extents=original['dimensions'])
                    result.append(dict(id=frame['id'],session=frame['session'],method=method,actual_pose=value['actual_pose'],
                        original_family_pose=original_pose,solution=value['solution'],branch_result=value['branch_result'],self_hidden=H.tolist(),
                        appearance=value['appearance'],observed_points=q.tolist(),corrected_measurements=value['corrected_measurements'],
                        joint_observation=value['joint_observation'],OOF=True,GT_input=False))
                a,b=variants['JOINT_OOF_STAGED'],variants['JOINT_OOF_VISIBLE_TOGETHER']
                assert a['appearance']==b['appearance'] and a['corrected_measurements']==b['corrected_measurements']
                assert a['appearance']==variants['JOINT_OOF_NO_DELTA']['appearance']
            assert np.array_equal(q,q_before) and initial==initial_before
            result.append(dict(id=frame['id'],session=frame['session'],method='INITIAL',actual_pose=initial,
                solution=dict(accepted=False,reason='supplied_initial'),self_hidden=H.tolist(),observed_points=q.tolist(),GT_input=False))
            if (index+1)%10==0:print('OOF_CAL46_INFER',index+1,flush=True)
    bindings(p);assert len(result)==230
    C.save_rows(HERE/'PREDICTIONS.jsonl.gz',result)
    write(HERE/'PREDICTION_SEAL.json',dict(complete=True,before_scoring=True,rows=len(result),sha256=sha(HERE/'PREDICTIONS.jsonl.gz'),
        protocol_sha256=sha(HERE/'PROTOCOL.json'),runner_sha256=sha(__file__),GT_read_canary='PASS',
        identical_evidence_staged_together=True,identical_evidence_nodelta=True,new_source_CNN_image_forwards=46,
        new_detector_forwards=0,new_backbone_forwards=0,cached_OOF_features=True,severe_frames=0,TEST_rows_read=0,FINAL_rows_read=0))
    print('OOF_CAL46_SEALED',len(result),flush=True)


def distribution(values):
    x=np.asarray(values,float);return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),P90=float(np.quantile(x,.9)))


def score():
    p,seal=read(HERE/'PROTOCOL.json'),read(HERE/'PREDICTION_SEAL.json');assert not(HERE/'RESULTS.json').exists()
    assert seal['before_scoring'] and seal['sha256']==sha(HERE/'PREDICTIONS.jsonl.gz')
    # Container filtering is allowed here only after inference seal; targets
    # outside exacteligibleCAL46 are discarded before any scoring use.
    source=read('/dev/shm/pallet-visibility-curriculum-20261013-v1/ORACLE_CAL_INPUTS.json')['frames']
    allowed=set(p['frame_ids']);isolated={f['id']:dict(target=f['target'],metadata=f['scoring_metadata']) for f in source if f['id'] in allowed}
    assert len(isolated)==46 and set(isolated)==allowed
    write(HERE/'ISOLATED_CAL_SCORING.json',dict(schema='CAL46_ONLY_POSTSEAL_SCORING',rows=isolated,severe_rows=0,TEST_rows=0,FINAL_rows=0))
    scored=rows(HERE/'PREDICTIONS.jsonl.gz');groups=defaultdict(list)
    for row in scored:
        truth=isolated[row['id']]['target']['truth'];row['metric']=pose_metric(row['actual_pose'],truth)
        if 'original_family_pose' in row:row['original_family_metric']=pose_metric(row['original_family_pose'],truth)
        groups[row['method']].append(row)
    summary={}
    for method,value in groups.items():
        acceptance=[r.get('branch_result',{}).get('accepted',r['solution']['accepted']) for r in value]
        report=dict(frames=len(value),accepted=sum(acceptance),local_curriculum_accepted=sum(r['solution']['accepted'] for r in value),
            flips=sum(r.get('branch_result',{}).get('branch_flip',False) for r in value),
            success_count=sum(r['metric']['success_5cm_5deg'] for r in value),
            metrics={k:distribution([r['metric'][k] for r in value]) for k in ('translation_cm','rotation_deg','ADDsym_cm')})
        if 'original_family_metric' in value[0]:report['original_family_metrics']={k:distribution([r['original_family_metric'][k] for r in value]) for k in ('translation_cm','rotation_deg','ADDsym_cm')}
        summary[method]=report
        print(method,report,flush=True)
    sessions=sorted({r['session'] for r in scored});rng=np.random.default_rng(20261013);draw=rng.integers(0,len(sessions),(10000,len(sessions)));lookup={(r['id'],r['method']):r for r in scored};paired={}
    for a,b in (('JOINT_OOF_STAGED','FRESH_CAL_VISIBILITY_CONTROL'),('JOINT_OOF_STAGED','INITIAL'),('JOINT_OOF_STAGED','JOINT_OOF_VISIBLE_TOGETHER'),('JOINT_OOF_STAGED','JOINT_OOF_NO_DELTA')):
        values=groups[a];comparison={}
        for field in ('metric','original_family_metric'):
            if field=='original_family_metric' and field not in lookup[values[0]['id'],b]:continue
            comparison[field]={}
            for k in ('translation_cm','rotation_deg','ADDsym_cm','success_5cm_5deg'):
                delta=np.asarray([float(r[field][k])-float(lookup[r['id'],b][field][k]) for r in values])
                sums=np.asarray([sum(d for d,r in zip(delta,values) if r['session']==s) for s in sessions]);counts=np.asarray([sum(r['session']==s for r in values) for s in sessions]);bootstrap=sums[draw].sum(1)/counts[draw].sum(1)
                comparison[field][k]=dict(mean_difference=float(delta.mean()),session_bootstrap_CI95=np.quantile(bootstrap,[.025,.975]).tolist())
        paired[a+'-'+b]=comparison
    C.save_rows(HERE/'SCORED_ROWS.jsonl.gz',scored)
    write(HERE/'RESULTS.json',dict(schema=p['schema'],summary=summary,paired=paired,sessions=sessions,
        prediction_sha256=seal['sha256'],CAL46_only=True,severe_rows=0,TEST_rows_read=0,FINAL_rows_read=0,
        policy_retuning=0,identical_evidence_controls=True,limits=p['limits']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('infer','score'));args=parser.parse_args()
    infer() if args.phase=='infer' else score()
