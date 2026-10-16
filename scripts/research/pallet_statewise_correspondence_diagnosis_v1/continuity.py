"""Actual consumer-factor preservation checks without fitting or references."""
from copy import deepcopy
import argparse
from pathlib import Path
import time

import numpy as np

from .common import canonical, digest, finite, read, rows, write, record_stage
from .typed_replay import _has_new_typed, _module


def prepared(packet, *, stage=None, family_name=None):
    spec=packet['solver_input']
    family=(packet['original_e_stage1_family_seed_bank'][family_name]['family']
            if family_name else packet['registered_family'])
    typed=_has_new_typed(packet) if stage is None else stage=='G13'
    backend=_module('statewise_partial_BI_outline_solver_v4_13' if typed else
                    'statewise_global_outline_solver_v4_10')
    boundary=spec['boundary_observations'] if typed else packet['original_e_legacy_boundary_observations']
    args=backend._prepare(family['native_xyz'],spec['K'],spec['states'],spec['visible'],spec['domains'],
                         spec['image_hw'],family['coarse_pose'],spec['config'],boundary,spec['u_regions'])
    x,K,states,measured,domains,hw,cfg,boundary,regions,Q,prior,requested=args
    scale=max(float(np.linalg.norm(np.ptp(x,axis=0))),1e-3)
    v=backend.V._pose_vector(packet['e0_pose']['R_cf'],packet['e0_pose']['centroid'],scale)
    return dict(backend=backend,xyz=x,K=K,states=states,measured=measured,domains=domains,hw=hw,
                cfg=cfg,boundary=boundary,regions=regions,prior=prior,scale=scale,v=v,
                outline=spec['outline_observation'])


def support_for_candidate(candidate, packet):
    """Posthoc decomposition with original guard predicates, no new cutoff."""
    try:
        p=prepared(packet,stage=candidate['stage'],family_name=candidate['family'])
        b=p['backend'];R=np.asarray(candidate['R'],float);t=np.asarray(candidate['t'],float)
        vector=b.V._pose_vector(R,t,p['scale'])
        q,_=b.V.project_native(p['xyz'],R,t,p['K'])
        visible=b.P.BASE._visible_audit(q,p['measured'],-R.T@t,p['hw'],p['cfg'])
        boundary=b.P._boundary_audit(vector,p['xyz'],p['K'],p['boundary'],p['hw'],p['scale'],p['cfg'],1.)
        return dict(passed=bool(all(v['visible_support_guards_pass'] for v in visible)
                               and all(v['feasible'] for v in boundary)),
                    predicate='original visible geometry/canvas and original finite partial tangent support',
                    sequential_pruning_claimed=False,new_thresholds=0)
    except (ValueError,KeyError,np.linalg.LinAlgError,FloatingPointError) as exc:
        return dict(passed=None,reason=type(exc).__name__+': '+str(exc))


def values(p):
    b=p['backend']
    objective=b.P._joint_objective_pair(p['v'],p['xyz'],p['K'],p['measured'],p['scale'],
                                      p['prior'],p['cfg'],p['boundary'],1.)
    constraints=b.P._joint_constraint_pair(p['v'],p['xyz'],p['K'],p['measured'],p['domains'],
                    tuple(None for _ in p['domains']),p['hw'],p['scale'],p['cfg'],p['boundary'],p['regions'])
    # Unsupported outlines have no active shape term in the original consumer.
    outline=b.outline_value_and_gradient(p['v'],p['xyz'],p['K'],p['scale'],p['outline'],
                  b._outline_config(p['outline']['configuration']))[:2] if p['outline'].get('supported') else (0.,np.zeros(6))
    return dict(objective=objective[0],gradient=objective[1],constraints=constraints[0],
                constraint_jacobian=constraints[1],outline=outline[0],outline_gradient=outline[1])


def maximum_difference(a,b):
    return {k:float(np.max(np.abs(np.asarray(a[k])-np.asarray(b[k]))))
            if np.asarray(a[k]).size else 0. for k in a}


def check(packet):
    import json
    p=prepared(packet); original=values(p)
    serial=values(prepared(json.loads(canonical(packet))))
    changed=deepcopy(packet)
    changed['display_only']={'S_projected_readout':[[1e9,-1e9]]*8,'internal_ID':'POISON'}
    display=values(prepared(changed))
    # Directly perturb already parsed archived xy of partial factors. This
    # isolates their mathematical activity, bypassing immutable-input guards;
    # it does not propose changing an input archive in production.
    partial_ids={x['native_corner'] for x in p['boundary']}
    altered=dict(p,measured=deepcopy(p['measured']))
    for m in altered['measured']:
        if m['native_corner'] in partial_ids:m['xy']=np.asarray(m['xy'])+[123.,-99.]
    partial=values(altered)
    tests=dict(serialization=maximum_difference(original,serial),
               unused_display_and_S_readout=maximum_difference(original,display),
               archived_partial_xy_mathematical_activity=maximum_difference(original,partial))
    return dict(id=packet['id'],packet_sha256=digest(packet),partial_ids=sorted(partial_ids),
        deltas=tests,serialization_and_display_pass=all(v==0 for k in ('serialization','unused_display_and_S_readout') for v in tests[k].values()),
        partial_xy_is_absent_from_point_loss=tests['archived_partial_xy_mathematical_activity']['objective']==0.,
        archived_xy_constraint_activity_reported_separately=True,
        no_solve=True,no_GT=True)


def audit(folder):
    out=Path(folder); start=time.monotonic();cpu=time.process_time()
    ids=set(read(out/'REPLAY_PROTOCOL.json')['pilot_ids'])
    result=[check(p) for p in rows(out/'e_trace_v2/TYPED_PACKETS.jsonl.gz') if p['id'] in ids]
    assert len(result)==6
    write(out/'CONTINUITY_AUDIT.json',dict(rows=result,all_serialization_display_pass=all(r['serialization_and_display_pass'] for r in result),
        objective_invariance_of_partial_xy=all(r['partial_xy_is_absent_from_point_loss'] for r in result),
        precision='actual original consumer factor calls; scalar/analytic backend retained; no optimization or GT',
        valid_permutation_check='test_reobs.py moves XYZ and owner IDs together; semantic-only relabel changes residual',
        pose_serialization_check='strict pilot replay parity; no additional fits'))
    record_stage(out,'actual_factor_continuity',start,worker_cpu_seconds=time.process_time()-cpu,new_pose_calls=0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();audit(a.folder)
