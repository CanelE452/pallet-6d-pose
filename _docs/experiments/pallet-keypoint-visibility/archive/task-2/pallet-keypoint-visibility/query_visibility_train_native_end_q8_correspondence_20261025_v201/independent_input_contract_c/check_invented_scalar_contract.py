"""Invented scalar algebra and schema fixtures only; no real data/model read."""
from pathlib import Path
import copy
import hashlib
import inspect
import json
import math
import original_other7_scalar_contract as c

H = Path(__file__).absolute().parent
Q8 = [(0., 0.), (10., 0.), (10., 10.), (0., 10.),
      (2., 3.), (12., 3.), (12., 13.), (2., 13.)]
GROUP = 'INVENTED_ORIGINAL_GROUP'
ROLE = 'ORIGINAL_SCENE_CONTEXT'


def records(q8=Q8, available=None):
    if available is None:
        available = [True] * 8
    return [dict(context_group_id=GROUP, context_role=ROLE, native_slot=k,
                 query_xy=list(q), query_available=available[k]) for k, q in enumerate(q8)]


def active(rr, q, L=20., i=0, av=True):
    return c.active_crop_relative_other7(rr, expected_group_id=GROUP,
            expected_role=ROLE, native_slot=i, active_q=q, active_L=L,
            active_available=av)


def shape(rr, i=0, av=True):
    return c.other7_layout_only(rr, expected_group_id=GROUP, expected_role=ROLE,
                              native_slot=i, active_available=av)


def nearly(a, b):
    return math.isclose(a, b, abs_tol=1e-13, rel_tol=0)


def run():
    rr = records()
    checks, examples = [], {}
    def add(name, passed):
        assert passed, name
        checks.append(dict(name=name, PASS=True))
    r0, r1 = active(rr, (0., 0.)), active(rr, (1.5, -.5))
    add('self coordinate and mask exactly zero', r0['other_dxdy8'][0] == (0., 0.) and not r0['other_mask8'][0])
    add('all other7 encode common active displacement', all(nearly(r1['other_dxdy8'][k][j]-r0['other_dxdy8'][k][j], -[1.5, -.5][j]/20) for k in range(1, 8) for j in range(2)))
    # Rectangle relation: q0=q1+q3-q2, without accessing original active q0.
    inferred = tuple(r1['other_dxdy8'][1][j]+r1['other_dxdy8'][3][j]-r1['other_dxdy8'][2][j] for j in range(2))
    add('self0 does not block indirect exact anchor recovery in ideal fixture', all(nearly(inferred[j], -[1.5, -.5][j]/20) for j in range(2)))
    examples['ideal_rectangle_self0_indirect_offset'] = dict(active_jitter=[1.5, -.5], inferred_old_anchor_minus_active_over_L=inferred, formula='r1+r3-r2; omitted q0 never read')
    changed_self = copy.deepcopy(rr);changed_self[0]['query_xy'] = [999999., -888888.]
    add('neither contract reads omitted original active xy', active(changed_self, (1.5, -.5)) == r1 and shape(changed_self) == shape(rr))
    # Pairwise differences remove common crop translation, but active-L scaling remains.
    pair0 = tuple(r0['other_dxdy8'][1][j]-r0['other_dxdy8'][2][j] for j in range(2))
    pair1 = tuple(r1['other_dxdy8'][1][j]-r1['other_dxdy8'][2][j] for j in range(2))
    add('other7 pair difference cancels active translation', all(nearly(a,b) for a,b in zip(pair0,pair1)))
    rscale = active(rr, (0., 0.), L=40.)
    add('active L still encodes scale through pair spacing', nearly(rscale['other_dxdy8'][1][0], r0['other_dxdy8'][1][0]/2))
    s0 = shape(rr)
    add('layout-only has no active q or L argument', not {'active_q','active_L'} & set(inspect.signature(c.other7_layout_only).parameters))
    transformed = [(3*x+41, 3*y-17) for x,y in Q8]
    s1 = shape(records(transformed))
    add('layout-only invariant under common original-coordinate shift/positive scale', all(nearly(a,b)for aa,bb in zip(s0['other_shape_dxdy8'],s1['other_shape_dxdy8'])for a,b in zip(aa,bb)))
    ar = active(records(transformed), (3*1.5+41,3*(-.5)-17), L=60.)
    add('aligned context invariant under joint image-coordinate positive scale/translation', all(nearly(a,b)for aa,bb in zip(r1['other_dxdy8'],ar['other_dxdy8'])for a,b in zip(aa,bb)))
    add('all native0..7 retained with exactly7 others and self0', all(sum(active(rr,Q8[i],i=i)['other_mask8'])==7 and not active(rr,Q8[i],i=i)['other_mask8'][i] for i in range(8)))
    add('different active crops have identical layout-only but different aligned context', shape(rr) == shape(rr) and r0 != r1)
    examples['information_loss'] = dict(active_crop_positions=[[0,0],[1.5,-.5]], layout_only_equal=True, original_target_normalized_location=[[0,0],[-1.5/20,.5/20]], conclusion='layout invariance also removes target-to-active-crop spatial binding')
    # Missing context remains masked; factual availability is not visibility.
    missing = records(available=[True, False, True, True, False, True, True, True]);missing[1]['query_xy']=None
    rm = active(missing, (0., 0.));add('missing other points zero+masked without completion', not rm['other_mask8'][1] and rm['other_dxdy8'][1] == (0.,0.) and not rm['other_mask8'][4])
    unavailable=records(available=[False,True,True,True,True,True,True,True])
    add('inputU does not produce a new E label or forward', not active(unavailable, None, av=False)['forward_allowed'] and not shape(unavailable,av=False)['forward_allowed'])
    try:active(rr,None,av=False);ok=False
    except ValueError:ok=True
    add('factual active availability cannot silently be overridden',ok)
    sparse=records(available=[True,True,False,False,False,False,False,False])
    add('one usable other point means unavailable normalized context', shape(sparse)['context_available'] is False)
    add('record order irrelevant; explicit native membership join', active(list(reversed(rr)),(1.5,-.5)) == r1)
    # Repeated external/global IDs are deliberately absent, never uniqueness keys.
    for name, mutate in [
        ('duplicate native', lambda x:x[7].update(native_slot=6)),
        ('mixed origin', lambda x:x[7].update(context_group_id='OTHER')),
        ('mixed original role', lambda x:x[7].update(context_role='ORIGINAL_BEFORE_CONTEXT')),
        ('fabricated jitter role', lambda x:x[7].update(context_role='SIMULTANEOUS_JITTER8')),
        ('nested GT input', lambda x:x[7].update(GT_endpoint=[1,2])),
        ('implicit numeric mask', lambda x:x[7].update(query_available=1)),
        ('available nonfinite point', lambda x:x[7].update(query_xy=[float('nan'),0]))]:
        bad=copy.deepcopy(rr);mutate(bad)
        try:active(bad,(0,0));ok=False
        except ValueError:ok=True
        add(name+' rejected',ok)
    # Valid input geometry is compatible with different semantic worlds. No label
    # argument exists: the function must not create or "certify" a visibility.
    add('no target/owner/quality/reference/pose/native-gold output', not any(any(s in k for s in ['target','label','owner','quality','reference','pose','gold'])for k in r0))
    examples['semantic_nonidentifiability'] = dict(same_original_q8=True, same_active_qL=True, possible_semantic_states=['intended end visible','intended end hidden by foreign object','wrong selected instance/part correspondence'], scalar_output_same=True, meaning='Geometry alone cannot certify these alternatives; not newly labeled real examples')
    out=dict(completed=True,PASS=True,checks=checks,count=len(checks),examples=examples,
             fixture_kind='INVENTED_SCALAR_ALGEBRA_ONLY',actual_source_metadata_arrays_headers_images_weights_models_metrics=0,
             proves='Information/invariance properties of these two declared formulas only',
             does_not_prove=['current model shortcut use','general invalidity of q8','winning input/architecture','real target identity or visibility','cause or performance effect'])
    with (H/'INVENTED_FIXTURES.json').open('x')as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(completed=True,PASS=True,count=len(checks),sha256=hashlib.sha256((H/'INVENTED_FIXTURES.json').read_bytes()).hexdigest())))


if __name__ == '__main__':run()
