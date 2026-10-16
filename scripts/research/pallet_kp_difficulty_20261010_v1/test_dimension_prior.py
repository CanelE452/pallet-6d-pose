"""Numerical checks for the new dimension-only prior, not research scores."""
import argparse
from pathlib import Path
import time
import cv2
import numpy as np
from ..pallet_observation_refiner_20261009_v1 import common as C
from ..pallet_observation_refiner_20261009_v1.solver import HypothesisBank, project
from ..pallet_observation_refiner_20261009_v1.inference import finish
from .dimension_prior import DimensionPriorBank


def run(output):
    assert not output.exists(), 'Preserve completed numerical checks'
    cv2.setNumThreads(1)
    begin = time.monotonic()
    K = np.array([[610.,0.,320.],[0.,608.,240.],[0.,0.,1.]])
    dims = np.array([1.2,.12,.8])
    R = cv2.Rodrigues(np.array([.25,-.2,.08]))[0]
    t = np.array([.03,.02,2.8])
    records = []
    total = {}
    for branch in (0,1):
        model = HypothesisBank(np.zeros((9,2)),K,dims).models[branch]
        q = np.vstack([project(model,R,t,K),[320.5,239.5]])
        extent = dims if branch==0 else dims[[2,1,0]]
        initial = dict(available=True,cf_extents=extent.tolist(),selected_hypothesis='known_toy_branch')
        for name, ids in [('plane_four',[0,1,4,5]),('nonplane_four',[0,1,2,4]),
                          ('five',[0,1,2,4,5]),('eight',list(range(8)))]:
            points=q.copy()
            points[[i for i in range(8) if i not in ids]]=np.nan
            bank=HypothesisBank(points,K,dims)
            prior=DimensionPriorBank(bank,initial)
            answer=prior.solve()
            assert answer['available'],(branch,name,answer['reason'])
            assert answer['initial_dimension_prior']['dimension_index']==branch
            assert np.allclose(answer['cf_extents'],extent,rtol=0,atol=1e-9)
            assert max(answer['residuals_used_px'])<1e-5
            assert bank.hypotheses is not None
            assert {c.dim for c in bank.hypotheses}=={0,1}
            records.append(dict(case=name,branch=branch,available=True,inliers=answer['inliers'],
                                free_pose_refit=True,hypothesis_bank_restored=True,
                                max_input_residual_px=max(answer['residuals_used_px']),
                                multiple_solutions=answer['multiple_solutions']))
            for k,v in bank.ledger.items():total[k]=total.get(k,0)+v
        bank=HypothesisBank(q,K,dims)
        result=finish(DimensionPriorBank(bank,initial),q,initial,excluded=[6,7],hidden=[6,7])
        assert result['new_pose_estimated']
        assert not set(result['solver']['fit_input_ids'])&{6,7}
        assert np.array_equal(result['native_points'][8],q[8])
        assert np.allclose(result['native_points'][[6,7]],np.asarray(result['actual_pose']['projected'])[[6,7]])
        assert result['reprojections_reused_as_observations'] is False
        records.append(dict(case='hidden_replacement',branch=branch,fit_excludes_hidden=True,center_unchanged=True))
        for k,v in bank.ledger.items():total[k]=total.get(k,0)+v
    q=np.full((9,2),np.nan);q[:3]=[[100,100],[200,100],[100,200]];q[8]=[320,240]
    bank=HypothesisBank(q,K,dims)
    result=DimensionPriorBank(bank,dict(available=True,cf_extents=dims.tolist())).solve()
    assert not result['available'] and result['reason']=='insufficient_observations'
    assert bank.ledger['generic_calls']==0
    records.append(dict(case='three_observations',available=False,generic_calls=0))
    bank=HypothesisBank(np.vstack([project(bank.models[0],R,t,K),[320,240]]),K,dims)
    result=DimensionPriorBank(bank,dict(available=False)).solve()
    assert not result['available'] and result['reason']=='PRIOR_UNAVAILABLE'
    records.append(dict(case='missing_prior',available=False,reason=result['reason']))
    value=dict(schema='dimension_prior_numeric_checks_v1',passed=True,cases=records,counts=total,
               new_detector_forwards=0,new_head_forwards=0,training_updates=0,
               geometric_toy_pose_paths=len(records),wall_seconds=time.monotonic()-begin,
               not_real_accuracy_evidence=True,code=[C.binding(Path(__file__)),C.binding(Path(__file__).with_name('dimension_prior.py'))])
    C.write(output,value)
    print('DIMENSION_PRIOR_TESTS',len(records),'PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
