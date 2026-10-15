"""Exact stable ranking/cache equivalence, synthetic only; no RGB or labels."""
from copy import deepcopy
import hashlib
from pathlib import Path
import numpy as np
import pytest

from . import bounded_geometry_graph_rank_v4_29 as B
from .test_joint_geometry_guided_lines_v4_28 import fixture,line


def old_step(beam,choices,score,size,explored=0,truncated=False):
    expanded=[]
    for selected,_ in beam:
        for node in [*choices,None]:
            path=selected+[node];expanded.append((path,score(path)));explored+=1
    expanded.sort(key=B._rank_key)
    return expanded[:size],explored,truncated or len(expanded)>size


@pytest.mark.parametrize('size',[1,3,8,256])
def test_streaming_heap_exact_stable_prefix_ties_infinities_and_generation_order(size):
    beam=[([dict(name='same',energy=i%2,token=i)],{})for i in range(5)]
    choices=[dict(name=str(i%3),energy=float('inf')if i%7==0 else float(i%4),token=i)for i in range(73)]
    visited=[]
    def score(path):
        visited.append(tuple(None if n is None else n['token']for n in path))
        return dict(total_engineering_energy=sum(9. if n is None else n['energy']for n in path),
                    unmodified_record=[None if n is None else n['token']for n in path])
    expected=old_step(beam,choices,score,size,7,False);old_order=visited[:];visited.clear()
    actual=B.bounded_beam_step(beam,choices,score,size,7,False)
    assert actual==expected and visited==old_order


def test_heap_empty_and_truncated_flags_match_existing_loop():
    assert B.bounded_beam_step([],[],lambda x:None,2,13,True)==([],13,True)
    score=lambda path:dict(total_engineering_energy=0.)
    assert B.bounded_beam_step([([],{})],[],score,4,0,False)==old_step([([],{})],[],score,4)


def test_bounded_cache_eviction_recomputes_exact_values_and_keeps_capacity():
    cache=B.BoundedIntersectionCache(2);plain={};recomputations=0
    for a in [1,2,3,1,3,2,4,3,1,4]:
        def calculate():return(np.array([a/7.,a*a/11.]),float(a)/13.)
        if a not in plain:plain[a]=calculate()
        if a not in cache:cache[a]=calculate();recomputations+=1
        first,second=cache[a],plain[a]
        assert np.array_equal(first[0],second[0])and first[1]==second[1]
        assert len(cache)<=2
    assert recomputations>len(plain)


def observations():
    args,kw=fixture();args[3][0]='B';args[4][0]['state']='B'
    # Several alternatives and two named incident edges force beam truncation,
    # intersection reuse/eviction, ordered support and rounded diagnostics.
    a=line([[x,60.]for x in range(35,70)],[0.,1.],-60.)
    b=line([[60.,y]for y in range(35,70)],[1.,0.],-60.)
    rays=[]
    for mate,base in [(1,a),(4,b)]:
        if args[2]['initial_self_hidden'][mate]:continue
        alternatives=[]
        for i in range(4):
            item=deepcopy(base);item['offset']+=i*.01
            item['inlier_rms_px']=i*.01
            alternatives.append(item)
        rays.append(dict(native_corner=0,native_neighbor=mate,supported=True,chains=alternatives,rejected_chains=[]))
    return args,kw,dict(edge_records=rays)


def gated_snapshot(args,kw,obs):
    # The inherited S slots are Poison sentinels. Snapshot only permitted
    # fields; check the excluded projection container and slots by identity.
    geometry={k:v for k,v in args[2].items()if k!='initial_projection_xy'}
    return deepcopy((args[:2],geometry,args[3:],kw,obs))


@pytest.mark.parametrize('cache_limit',[1,2,4096])
def test_entire_parent_graph_output_exact_with_tiny_LRU_and_bounded_expansion(cache_limit):
    args,kw,obs=observations()
    configuration=dict(graph_configuration=dict(maximum_beam_states=3,reported_joint_alternatives=3))
    before=gated_snapshot(args,kw,obs);parent=B.G.rank_joint_geometry_line_graph
    projection=args[2]['initial_projection_xy'];projection_slots=tuple(map(id,projection))
    parent_globals={k:id(v)for k,v in parent.__globals__.items()}
    expected=parent(obs,args[2],args[3],args[4],**kw,configuration=configuration,
        pose_guides=None,visibility_context=None,pallet_mask=args[1])
    actual=B.rank_joint_geometry_line_graph_bounded(obs,args[2],args[3],args[4],**kw,
        configuration=configuration,pose_guides=None,visibility_context=None,
        pallet_mask=args[1],intersection_cache_limit=cache_limit)
    assert actual==expected
    assert B.G.NMS.G.C.finite(gated_snapshot(args,kw,obs))==B.G.NMS.G.C.finite(before)
    assert args[2]['initial_projection_xy']is projection
    assert tuple(map(id,projection))==projection_slots
    assert parent_globals=={k:id(v)for k,v in parent.__globals__.items()}


def test_fresh_v29_facade_entire_output_and_source_bytes_exact():
    args,kw,obs=observations();paths=[Path(B.G.__file__),Path(B.PRODUCER.__file__)]
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}
    expected=B.PRODUCER._rank_fresh(obs,args[2],args[3],args[4],**kw,
        configuration=None,pose_guides=None,visibility_context=None,pallet_mask=args[1])
    actual=B.rank_fresh_v29_bounded(obs,args[2],args[3],args[4],**kw,
        configuration=None,pose_guides=None,visibility_context=None,
        pallet_mask=args[1],intersection_cache_limit=1)
    assert actual==expected
    assert before=={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}


def test_invalid_cache_or_beam_budget_fails_without_scoring():
    with pytest.raises(ValueError):B.BoundedIntersectionCache(0)
    with pytest.raises(ValueError):B.bounded_beam_step([],[],lambda p:None,0,0,False)
