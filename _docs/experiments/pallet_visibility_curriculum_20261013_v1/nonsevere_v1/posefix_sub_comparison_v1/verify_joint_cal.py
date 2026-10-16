"""No-label pure-wrapper parity for the FULL-CAL cached joint control."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path('/home/minjae/Documents/github/pallet-pose')
sys.path.insert(0, str(ROOT))
from scripts.research.pallet_visibility_curriculum_20261013_v1 import joint_api as J
from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C

HERE = Path(__file__).resolve().parent
COHORT = HERE.parent


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in gzip.open(path, 'rt')]


def main():
    out = HERE/'JOINT_CAL_CONTROL_PARITY.json'
    assert not out.exists()
    assert read(HERE/'PREDICTION_SEAL.json')['predictions_sha256'] == C.sha(HERE/'PREDICTIONS.jsonl.gz')
    references = {r['id']:r for r in rows(HERE/'PREDICTIONS.jsonl.gz')
                  if r['partition']=='CAL' and r['method']=='JOINT_SINGLE_FAMILY'}
    evidence = {r['id']:r for r in rows(COHORT/'SAM_boundary_uncertainty_v1/PREDICTIONS.jsonl.gz') if r['method']=='RAW_STAGED'}
    full = {(r['id'],r['corner']):r for r in read(COHORT/'joint_model/FULL_PREDICTION_CACHE.json')}
    frames = [x['frame'] for x in read(HERE/'BASELINE_INPUTS.json')['frames'] if x['frame']['partition']=='CAL']
    engine = J.JointSpatialBranch({'injected_frozen_prediction_cache':True}, 'injected_clean_source_mask',
            COHORT/'visibility_logistic_CAL46_v2.json', branch_policy='preserve_original_family')
    maximum = 0.
    for frame in frames:
        fid=frame['id'];q=np.asarray(frame['points']['N3_SUBPIX']);h,w=frame['raw_hw']
        H=np.asarray(evidence[fid]['self_hidden'],bool)
        valid=np.isfinite(q[:8]).all(1)&~np.all(q[:8]==-1,axis=1)
        inside=valid&(q[:8,0]>=0)&(q[:8,0]<w)&(q[:8,1]>=0)&(q[:8,1]<h)
        p=np.zeros(8);delta=np.zeros((8,2))
        for k in np.flatnonzero(inside&~H):
            p[k]=full[fid,k]['P_usable'];delta[k]=full[fid,k]['RGB_delta_px']
        head=dict(P_usable=p.tolist(),eligible=(inside&~H).tolist(),threshold=.45,
            RGB_delta_px=delta.tolist(),points=references[fid]['native_points'],
            observed_refined_ids=np.flatnonzero(~np.isclose(np.asarray(references[fid]['native_points'])[:8],q[:8],rtol=0,atol=1e-12,equal_nan=True).all(1)).tolist())
        # Current RGB extraction/head already occurred in frozen caches. These
        # injected values test wrapper algebra, not a new image forward.
        with patch.object(J,'_predict_joint',return_value=head), \
             patch.object(J,'_segment',return_value=np.zeros((h,w))), \
             patch.object(J,'_segmented_appearance',return_value=deepcopy(evidence[fid]['appearance'])):
            actual=engine.refine(np.zeros((h,w,3),np.uint8),q,frame['K'],frame['initial_poses']['N3_SUBPIX'],frame['xyz'])
        reference=references[fid]
        for field in ('R_cf','R_physical','centroid','cf_extents'):
            maximum=max(maximum,float(np.max(np.abs(np.asarray(actual['actual_pose'][field])-np.asarray(reference['actual_pose'][field])))))
        assert actual['solution']['used_ids']==reference['solution']['used_ids']
        assert actual['solution']['accepted']==reference['solution']['accepted']
        np.testing.assert_allclose(actual['corrected_measurements'],reference['native_points'],rtol=0,atol=0)
    print('FULLCAL_JOINT_MAXIMUM',maximum,flush=True)
    assert maximum < 1e-12 and len(frames)==46
    out.write_text(json.dumps(dict(PASS=True,frames=46,max_pose_difference=maximum,exact_used_ids_and_acceptance=True,
        pure_production_wrapper_with_frozen_cached_heads_and_evidence=True,new_RGB_or_model_forwards=0,labels_read=0,
        current_joint_api_sha256=C.sha(Path(J.__file__)),prediction_sha256=C.sha(HERE/'PREDICTIONS.jsonl.gz')),indent=2)+'\n')
    print('FULLCAL_JOINT_WRAPPER_PARITY',len(frames),maximum,flush=True)


if __name__=='__main__':
    main()
