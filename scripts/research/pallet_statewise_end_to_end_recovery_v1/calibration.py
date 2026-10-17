"""One predeclared synthetic-validation choice for A's none prior.

This separate supervision process never reads real PRIMARY/CAL references.
It evaluates normalized conditional image densities at the renderer pose,
not final PRIMARY success, and preserves every catalog value and frame.
"""
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp

from .common import rows,sha,write
from .solver import compile_frame,_native_pose
from . import owner_model as A
from scripts.research.pallet_visibility_curriculum_20261013_v1 import joint_ot_jacobian_v2 as J


def choose_none_prior(frames_path,gt_path,output):
    start=time.monotonic()
    catalog=(.01,.05,.1)
    frames=[f for f in rows(frames_path) if f['split']=='validation']
    if not frames:
        raise ValueError('Explicit synthetic validation population is required')
    gt={r['id']:r for r in rows(gt_path)}
    results=[]
    for eta in catalog:
        losses=[]
        for frame in frames:
            truth=gt[frame['id']]
            pose=dict(R_physical=truth['R_fixed'],centroid=truth['t'])
            family_losses=[]
            for model in compile_frame(frame,dict(none_prior=eta)).values():
                c=model.ctx
                p=_native_pose(pose,c)
                q,_,_,_=J.projection_and_jacobian(p,c.xyz,c.K,c.scale)
                conditional=A.energy(model.owner,q)
                # Undo the candidate-constant peak subtraction. Catalog
                # selection must use the actual normalized density, whose
                # covariance volume and eta-dependent constants matter.
                peak=sum(g['reliability']*g['log_mixture_peak'] for g in model.owner['groups'])
                family_losses.append(conditional-2*peak)
            total=float(-2*logsumexp(-.5*np.asarray(family_losses))+2*np.log(len(family_losses)))
            losses.append(dict(id=frame['id'],negative_twice_log_density=total))
        results.append(dict(none_prior=eta,mean_negative_twice_log_density=float(np.mean(
            [r['negative_twice_log_density']for r in losses])),rows=losses))
    selected=min(results,key=lambda r:(r['mean_negative_twice_log_density'],catalog.index(r['none_prior'])))
    result=dict(schema='SYNTHETIC_ONLY_NONE_PRIOR_CALIBRATION_V1',catalog=list(catalog),
        selected_none_prior=selected['none_prior'],validation_frames=len(frames),
        selection='lowest mean normalized observation negative-twice-log-density at existing renderer pose; no success labels',
        global_family_prior='equal across both fixed registered WD families; no GT family selection',
        equal_global_owner_prior=True,source_symmetry_not_relabelled=True,all_catalog_results=results,
        inputs_sha256={str(frames_path):sha(frames_path),str(gt_path):sha(gt_path)},
        PRIMARY_reference_reads=0,real_reference_reads=0,source_sha256=sha(__file__),
        elapsed_seconds=time.monotonic()-start)
    write(output,result)
    return result
