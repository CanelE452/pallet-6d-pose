"""Constructed union tests plus two fixed old-cache feature parity checks."""
from __future__ import annotations
import argparse,time
from copy import deepcopy
from pathlib import Path
import numpy as np
from . import s1_union as U

def constructed():
    names=U.MODES
    def c(energy,gap):
        f=[0.]*62
        for j in range(5):f[j]=U.encode(energy+j);f[26+j]=U.encode(gap);f[31+j]=f[57+j]=1.
        return dict(features=f)
    old=[c(8.,3.),c(5.,0.)];mins,spread=U.recover_minima(old)
    assert np.allclose([mins[m]for m in names],[5.,6.,7.,8.,9.],atol=1e-12,rtol=0)
    new=U.update_gaps(old,mins,dict(zip(names,[2.,3.,4.,5.,6.])))
    assert all(abs(U.decode(new[0]['features'][26+j])-6.)<1e-12 for j in range(5))
    assert all(abs(U.decode(new[1]['features'][26+j])-3.)<1e-12 for j in range(5))
    assert all(a['features'][:26]==b['features'][:26]and a['features'][31:]==b['features'][31:]for a,b in zip(old,new))
    assert U.update_gaps(old,mins,dict(zip(names,[50.]*5)))==old
    family=dict(native_xyz=[[-1,-1,-1],[1,1,1]],physical_registration_Q=np.eye(3).tolist())
    pose=dict(available=True,R_physical=np.eye(3).tolist(),centroid=[0,0,3])
    assert U.pose_valid(pose,family)[0]
    bad=deepcopy(pose);bad['centroid'][2]=.1;assert not U.pose_valid(bad,family)[0]
    bad=deepcopy(pose);bad['R_physical'][0][0]=-1;assert not U.pose_valid(bad,family)[0]
    assert not U.pose_valid(dict(available=False),family)[0]
    from scripts.research.pallet_boundary_action_repair_v1.action_data import same_pose,inputs,fit_normalizer
    phase=deepcopy(pose);phase['R_physical']=np.diag([-1,1,-1]).tolist()
    assert not same_pose(pose,phase,1.)
    fixture=dict(id='constructed',candidates=[dict(candidate_id='N0',features=old[0]['features'],eligible=True,actual_pose=pose,source_baseline_aliases=['N0']),
        dict(candidate_id='OTHER',features=old[1]['features'],eligible=True,actual_pose=dict(pose,centroid=[.1,0,3]),source_baseline_aliases=[])],fallback=pose)
    norm=fit_normalizer([fixture]);x,entries=inputs(fixture,norm,'H1');assert x.shape==(2,188)
    assert np.array_equal(x[:,62:124],np.tile(x[:,:62].mean(0),(2,1)))
    assert np.array_equal(x[:,124:186],np.tile(x[:,:62].max(0),(2,1)))
    assert np.allclose(x[:,186],np.log1p(2))
    return dict(constructed_pass=True,checks=['five_gap_refresh','individual26_bitwise_reuse','missing_masks_preserved',
        'proper_rotation','positive_depth','F_availability','no_C2_collapse','188D_eligible_bank_mean_max_count'])

def parity(d1,old_run):
    from scripts.research.pallet_statewise_end_to_end_recovery_v1 import solver as S
    result=[];config=U.read(Path(old_run)/'CONFIG_FROZEN.json')
    for domain in ('real','synthetic'):
        tick=time.monotonic();ip,bankdir=U.paths(d1,domain);frame=next(U.rows(ip));src=bankdir/'frames/0000'
        erun=next(U.rows(src/'E_RUN.jsonl.gz'));old=next(U.rows(src/'FEATURES.jsonl.gz'));bundle=next(U.rows(src/'BOUNDARY_BUNDLE.jsonl.gz'))
        inp=dict(frame,packet=erun['packet'],families=erun['families'],fallback=frame['initial_pose']);models=S.compile_frame(inp,config,bundle=bundle)
        bank=[]
        for c in old['candidates']:
            bank.append(dict(candidate_id=c['candidate_id'],source='N0'if 'N0'in c.get('source_baseline_aliases',[])else'E0'if'E0'in c.get('source_baseline_aliases',[])else'OTHER',
                R_physical=c['actual_pose']['R_physical'],centroid=c['actual_pose']['centroid']))
        scores=[S.candidate_scores(c,models)for c in bank];fresh=S.features_for_bank(inp,models,bank,scores)
        a=np.asarray([c['features']for c in old['candidates']]);b=np.asarray([c['features']for c in fresh]);delta=float(np.max(np.abs(a-b)))
        if delta>U.FEATURE_ATOL:raise AssertionError(('Original scalar feature parity failed',domain,delta))
        mins,spread=U.recover_minima(old['candidates']);exactmins={}
        for mode in U.MODES:
            vals=[s[f]['scores'][mode]for s in scores for f in models if S._finite_feature(s[f]['scores'][mode])is not None]
            exactmins[mode]=min(vals)if vals else None
        md=max(abs(mins[m]-exactmins[m])for m in U.MODES if mins[m]is not None)
        if md>U.FEATURE_ATOL:raise AssertionError(('Cached minimum recovery parity failed',domain,md))
        constructed_pose=deepcopy(frame['initial_pose']);constructed_pose['centroid'][0]+=.01
        union,_=U.union_frame(frame,erun,old,bundle,[dict(method=U.METHODS[0],actual_pose=constructed_pose)],config)
        ubank=bank+[dict(candidate_id=U.METHODS[0],source='PF_COMPLETE_ACTION',R_physical=constructed_pose['R_physical'],centroid=constructed_pose['centroid'])]
        uscores=scores+[S.candidate_scores(ubank[-1],models)]
        ufresh=S.features_for_bank(inp,models,ubank,uscores)
        ua=np.asarray([c['features']for c in union['candidates']]);ub=np.asarray([c['features']for c in ufresh]);udelta=float(np.max(np.abs(ua-ub)))
        if udelta>U.FEATURE_ATOL:raise AssertionError(('Constructed PF union feature parity failed',domain,udelta))
        result.append(dict(domain=domain,fixed_input_order_index=0,id=frame['id'],candidates=len(bank),
            old_features_bitwise_equal=bool(np.array_equal(a,b)),old_features_max_abs=delta,
            constructed_PF_union_features_max_abs=udelta,
            recovered_bank_minima_max_abs=md,recovered_minima_spread=spread,feature_atol=U.FEATURE_ATOL,wall_seconds=time.monotonic()-tick))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--d1',type=Path);p.add_argument('--old-run',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    out=constructed()
    if a.d1:out['fixed_old_cache_parity']=parity(a.d1,a.old_run)
    if a.output:U.write(a.output,out)
    print(U.canonical(out),flush=True)
if __name__=='__main__':main()
