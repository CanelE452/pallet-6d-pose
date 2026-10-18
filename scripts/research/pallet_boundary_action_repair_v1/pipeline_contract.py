"""Prediction-only data contract for the one fixed D1 producer.

No reference is accepted by this module. Historical caches supply only already
executed N3/SubPix image predictions, known calibration, and registered sizes.
"""
from __future__ import annotations

from copy import deepcopy
from collections import Counter
from pathlib import Path
import numpy as np

from scripts.research.pallet_statewise_end_to_end_recovery_v1.common import rows, read, sha, save_rows, write

SCHEMA = 'BOUNDARY_ACTION_D1_UPSTREAM_INPUT_V1'
ALLOWED = frozenset(('schema','id','partition','split','session','cluster_id','image',
    'image_sha256','raw_hw','K','native_xyz','initial_pose','initial_points',
    'known_physical_dimensions_xyz_m','prediction_object_type','crop_transform',
    'upstream_provenance'))


def validate_input(frame):
    if set(frame)-ALLOWED:
        raise ValueError('Unrecognized producer input fields: '+str(sorted(set(frame)-ALLOWED)))
    if frame.get('schema') != SCHEMA:
        raise ValueError('Explicit prediction-only schema required')
    for key,shape in [('K',(3,3)),('native_xyz',(8,3)),('initial_points',(9,2))]:
        a=np.asarray(frame[key],float)
        if a.shape!=shape or not np.isfinite(a).all():
            raise ValueError('Finite '+key+' with shape '+str(shape)+' required')
    p=frame['initial_pose']
    if not p.get('available'):
        raise ValueError('Unavailable upstream needs an explicit retained failure row')
    for key,shape in [('R_cf',(3,3)),('R_physical',(3,3)),('centroid',(3,))]:
        a=np.asarray(p[key],float)
        if a.shape!=shape or not np.isfinite(a).all():raise ValueError('Invalid initial '+key)
    if not np.allclose(np.asarray(p['R_cf']).T@p['R_cf'],np.eye(3),atol=1e-6,rtol=0):
        raise ValueError('Proper predicted initial rotation required')
    a,b,c=np.asarray(p['cf_extents'],float)/2
    expected=np.array([[-a,-b,-c],[a,-b,-c],[a,b,-c],[-a,b,-c],[-a,-b,c],[a,-b,c],[a,b,c],[-a,b,c]])
    if not np.allclose(expected,frame['native_xyz'],atol=1e-10,rtol=0):
        raise ValueError('Native cuboid and upstream predicted pose registration disagree')
    if sha(frame['image'])!=frame['image_sha256']:raise ValueError('Original RGB changed')
    return frame


def build_upstream(old_run,output):
    """Export sanitized existing forward outputs without opening any target.

    Real input is the sealed predictor-only v1 join, but its historical banks,
    old states, old E0, and all observations are deliberately not exported.
    Synthetic input is the sealed pre-state production cache, not its GT sidecar.
    """
    old=Path(old_run); out=Path(output);out.mkdir(parents=True,exist_ok=True)
    realpath=old/'inputs_all_available_v1/PREDICTION_INPUTS238.jsonl.gz'
    synthpath=old/'synthetic_v1/EXPANSION1003_NO_GT_PACKETS.jsonl.gz'
    real_seal=read(realpath.parent/'INPUT_BINDING.json')
    synth_seal=read(synthpath.with_name('EXPANSION1003_NO_GT_PACKETS_SEAL.json'))
    if sha(realpath)!=real_seal['output_sha256'] or sha(synthpath)!=synth_seal['predictions_sha256']:
        raise ValueError('Frozen upstream input bytes changed')
    real_digest=sha(realpath);synth_digest=sha(synthpath)
    real=[];synthetic=[]
    for r in rows(realpath):
        trace=r['trace'];p=deepcopy(trace['initial_pose'])
        xyz=deepcopy(trace['native_xyz'])
        Q=np.asarray(p['R_cf']).T@np.asarray(p['R_physical'])
        dimensions=np.ptp(np.asarray(xyz)@Q,axis=0).tolist()
        f=dict(schema=SCHEMA,id=r['id'],partition=r['partition'],split=None,
            session=r['session'],cluster_id=r['cluster_id'],image=r['image'],
            image_sha256=r['image_sha256'],raw_hw=trace['raw_hw'],K=trace['K'],
            native_xyz=xyz,initial_pose=p,initial_points=trace['initial_points'],
            known_physical_dimensions_xyz_m=dimensions,prediction_object_type='REGISTERED_KNOWN_PALLET',
            crop_transform=None,upstream_provenance=dict(source='SEALED_ACTUAL_N3_SUBPIX_F',
                source_journal_sha256=real_digest,old_state_observation_candidate_bank_exported=False,
                GT_inputs=False))
        validate_input(f);real.append(f)
    for r in rows(synthpath):
        p=deepcopy(r['initial_pose']);xyz=deepcopy(r['solver_input']['xyz'])
        f=dict(schema=SCHEMA,id=r['id'],partition='SYNTH_TRAIN' if r['split']=='train' else 'SYNTH_VAL',
            split=r['split'],session='SYNTHETIC_OPAQUE_METADATA',cluster_id='SYNTHETIC_OPAQUE_METADATA',
            image=r['image'],image_sha256=r['image_sha256'],raw_hw=r['raw_hw'],K=r['K'],
            native_xyz=xyz,initial_pose=p,initial_points=r['initial_points'],
            known_physical_dimensions_xyz_m=r['known_physical_dimensions_xyz_m'],
            prediction_object_type='REGISTERED_KNOWN_PALLET',crop_transform=None,
            upstream_provenance=dict(source='SEALED_ACTUAL_N3_SUBPIX_F',
                source_journal_sha256=synth_digest,raw_coordinates_already_unpadded=True,
                known_camera_and_registered_asset_axes_from_prior_sanitized_provision=True,
                per_frame_GT_pose_box_visibility_or_mask_inputs=False,GT_inputs=False))
        validate_input(f);synthetic.append(f)
    if len(real)!=238 or Counter(r['partition']for r in real)!=Counter(TEST=49,REMAINDER=143,CAL=46):
        raise ValueError('Exact real238 required')
    if len(synthetic)!=1003 or Counter(r['split']for r in synthetic)!=Counter(train=768,validation=235):
        raise ValueError('Exact original synthetic768/235 required')
    for name,data in [('REAL238',real),('SYNTHETIC1003',synthetic)]:
        save_rows(out/(name+'_UPSTREAM.jsonl.gz'),data)
    receipt=dict(schema='D1_UPSTREAM_SANITIZATION_SEAL_V1',complete=True,
        real_rows=238,synthetic_rows=1003,source_bindings={str(realpath):sha(realpath),str(synthpath):sha(synthpath)},
        artifacts={p.name:sha(p)for p in out.glob('*_UPSTREAM.jsonl.gz')},
        source_sha256=sha(__file__),GT_reads=0,new_backbone_forwards=0,
        new_SubPix_calls=0,N0_reused_exactly=True,old_E0_or_history_only_bank_used=False,
        source_basis_limit='Existing synthetic provision used known per-asset model-axis registration; no per-frame GT pose is read by this producer.')
    write(out/'UPSTREAM_SEAL.json',receipt);return receipt
