"""GT-free unit-free task spread and discrete failure aggregation."""
import math
import numpy as np

DEFINITION = dict(formula='max(midrank(spread_x),midrank(spread_z),midrank(spread_yaw),candidate_switch,axis_switch,pnp_failure,detection_failure)',
    midrank='(count(value<v)+0.5*count(value==v))/N on finite values; missing assigned1',
    spread='90th percentile absolute deviations from coordinatewise median, NumPy linear quantile',
    insufficient='Fewer than2 valid poses -> all spreads missing; invalid P0 -> yaw spread missing',
    yaw='P0 yaw unwrap anchor, wrap[-pi,pi), median unwrapped center, absolute circular deviations',
    candidate_switch='Among pose-valid views including P0; compare recovered dense grid index to P0; absent P0 detection ->1',
    axis_switch='Among pose-valid views including P0; absent P0 pose ->1',
    pnp_failure='Number without valid pose /8, includes detection failures',
    detection_failure='Number without above-floor candidate /8',
    scalar_units='Only empirical ranks and rates; never add meters to degrees', range=[0,1])


def wrap(angle):
    return (np.asarray(angle)+np.pi)%(2*np.pi)-np.pi


def midranks(values):
    values = np.asarray([np.nan if v is None else v for v in values], float)
    finite = np.isfinite(values)
    out = np.ones(len(values))
    a = values[finite]
    if len(a):
        out[finite] = [(np.sum(a<v)+.5*np.sum(a==v))/len(a) for v in a]
    return out


def summarize(views):
    assert len(views) == 8
    original = views[0]
    valid = [v for v in views if v['pose_valid']]
    sx = sz = sy = None
    if len(valid) >= 2:
        x,z = np.array([v['x_m'] for v in valid]), np.array([v['z_m'] for v in valid])
        sx,sz = [float(np.quantile(abs(a-np.median(a)),.9)) for a in (x,z)]
        if original['pose_valid']:
            anchor = original['yaw_rad']
            angles = anchor + wrap(np.array([v['yaw_rad'] for v in valid])-anchor)
            center = np.median(angles)
            sy = float(np.quantile(abs(wrap(angles-center)),.9))
    candidate = float(np.mean([v['candidate_index'] != original['candidate_index'] for v in valid])) if valid and original['detection'] else 1.
    axis = float(np.mean([v['axis_id'] != original['axis_id'] for v in valid])) if valid and original['pose_valid'] else 1.
    return dict(spread_x_m=sx,spread_z_m=sz,spread_yaw_rad=sy,spread_yaw_deg=None if sy is None else math.degrees(sy),
        candidate_switch_rate=candidate,axis_switch_rate=axis,pnp_failure_rate=1-len(valid)/8,
        detection_failure_rate=sum(not v['detection'] for v in views)/8,valid_poses=len(valid),original_pose_valid=original['pose_valid'])


def calculate(rows):
    result = [dict(frame_id=r['frame_id'], **summarize(r['views'])) for r in rows]
    ranks = [midranks([r[k] for r in result]) for k in ('spread_x_m','spread_z_m','spread_yaw_rad')]
    for i,r in enumerate(result):
        r.update(rank_x=float(ranks[0][i]),rank_z=float(ranks[1][i]),rank_yaw=float(ranks[2][i]))
        r['R_task'] = max(r[k] for k in ('rank_x','rank_z','rank_yaw','candidate_switch_rate','axis_switch_rate','pnp_failure_rate','detection_failure_rate'))
        assert 0 <= r['R_task'] <= 1
    return result


def cvar90(values):
    a=np.asarray(values,float)
    assert len(a) and np.isfinite(a).all()
    return float(np.sort(a)[-math.ceil(.1*len(a)):].mean())
