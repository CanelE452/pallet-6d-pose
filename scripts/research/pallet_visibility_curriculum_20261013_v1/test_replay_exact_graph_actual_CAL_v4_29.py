"""Metadata selection/guide/ref contracts; no actual rows, RGB or mask IO."""
from copy import deepcopy
import hashlib
import json
import numpy as np
import pytest

from . import replay_exact_graph_actual_CAL_v4_29 as R
from .test_geometry_graph_exact_memo_v4_29 import graph_fixture


def test_case_selection_fixed_original_order_not_candidate_or_error_magnitude():
    ids=['plastic:0','wood_a:0','wood_b:0','wood_c:0']
    host=dict(partial_observation=dict(completed_CAL_ids_original_order=ids,
        spools=[dict(id=k,summary=dict(actual_chain_hypotheses=n))for k,n in zip(ids,[1,2000,200000,0])]))
    selected=R.choose_cases(host,dict(ids=ids))
    assert [r['id']for r in selected]==ids[:2]
    assert selected[1]['summary']['actual_chain_hypotheses']==2000


def test_saved_guide_geometry_and_order_restore_without_pose_or_start_search():
    args,_,_=graph_fixture();g=args[2];states=args[3]
    saved=R.P._geometry_record(g,states)
    row=dict(states=states,initial_geometry=g,observation=dict(guide_geometries={
        'ORIGINAL_INITIAL':saved,'ORIGINAL_WD/OWN_INITIAL_COARSE':deepcopy(saved),
        'SWAPPED_WD/CACHED_17':deepcopy(saved)}))
    guides=R.restore_pose_guides(row)
    assert [r['guide_id']for r in guides]==['ORIGINAL_WD/OWN_INITIAL_COARSE','SWAPPED_WD/CACHED_17']
    assert [r['source_kind']for r in guides]==['ORIGINAL_TASK_COARSE','UNIFORM_INITIAL_PROPOSAL_BANK']
    assert all(R.G._digest(R.P._geometry_record(r['geometry'],states))==R.G._digest(saved)for r in guides)
    changed=deepcopy(row);changed['initial_geometry']['centroid'][0]+=.001
    with pytest.raises(ValueError,match='Original stored'):R.restore_pose_guides(changed)


def test_stable_reference_decoration_removes_only_redundant_line_record():
    ref=dict(guide_id='ORIGINAL_INITIAL',seed_direction_xy=[1.,0],guide_reference_id='abc')
    line=dict(guide_reference=ref,guide_reference_id='abc',chain_id='chain')
    observation=dict(edge_records=[dict(native_corner=0,native_neighbor=1,chains=[line])])
    graph=dict(rounded_BI_junctions=[],line_candidates=[dict(name='n',provenance=['0:0'],line_record=line,
        actual_inliers=[[1.,2.],[2.,2.]])],joint_alternatives=[dict(total_engineering_energy=1.)])
    expected=deepcopy(graph);expected['line_candidates'][0].pop('line_record')
    expected['line_candidates'][0]['chain_references']=[dict(ray_index=0,chain_index=0,native_corner=0,
        native_neighbor=1,guide_reference=ref,guide_reference_id='abc',chain_id='chain')]
    assert R.decorate_graph(graph,observation,{})==expected


def test_canonical_entire_graph_covers_diagnostics_and_alternatives():
    value=dict(selected=['same'],alternatives=[dict(cost=np.float64(1.25),unavailable=float('inf'))],
        diagnostics=dict(actual=np.array([[1.,2.]])))
    expected=hashlib.sha256(json.dumps(R.C.finite(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    assert R._canonical_sha(value)==expected
    changed=deepcopy(value);changed['alternatives'][0]['cost']+=1e-9
    assert R._canonical_sha(changed)!=R._canonical_sha(value)


def test_no_execution_without_explicit_root_replay_GO(tmp_path):
    with pytest.raises(ValueError,match='Separate root'):R.run(tmp_path)
    assert not list(tmp_path.iterdir())
    assert R.MODULE.endswith('.replay_exact_graph_actual_CAL_v4_29')and R.MODULE!='__main__'
