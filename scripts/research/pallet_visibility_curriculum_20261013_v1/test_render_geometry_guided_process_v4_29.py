"""Display gate/state/ref tests; no actual images, scores or journals."""
import gzip
import json
import numpy as np
import pytest

from . import render_geometry_guided_process_v4_29 as R
from .test_joint_geometry_guided_lines_v4_28 import Poison


def test_render_without_explicit_root_GO_reads_nothing(monkeypatch):
    def blocked(*a,**kw):raise AssertionError('No IO before explicit root GO')
    monkeypatch.setattr(R.C,'read',blocked)
    with pytest.raises(ValueError,match='explicit root'):
        R.render()


def test_S_U_markers_are_gated_before_coordinates_or_source():
    row=dict(states=['I','U','S','S','S','S','S','S'],initial_geometry=dict(initial_self_hidden=[False,False,True,True,True,True,True,True]),
        visible_measurements=[dict(native_corner=0,state='I',xy=[10.,20.],source='VISIBLE_BACKBONE_FALLBACK'),
            dict(native_corner=1,state='U',xy=Poison(),source=Poison()),dict(native_corner=2,state='S',xy=Poison(),source=Poison())])
    assert R._gated_visible(row)==[row['visible_measurements'][0]]


def test_selected_S_anchor_rejects_before_chain_fields():
    row=dict(states=['S']*8,initial_geometry=dict(initial_self_hidden=[True]*8),
        observation=dict(selected_chain_refs=[dict(native_corner=0,native_neighbor=1)],chain_observation=Poison()))
    with pytest.raises(ValueError,match='permanent state gates'):R._selected(row)


def test_only_exact_requested_rows_decode_without_whole_journal_dictionary(tmp_path):
    path=tmp_path/'rows.jsonl.gz'
    with gzip.open(path,'wt')as stream:
        stream.write('unrelated line deliberately not JSON\n')
        for i,ident in enumerate(R.CASES):stream.write(json.dumps(dict(id=ident,value=i))+'\n')
    rows,digests=R._exact_rows(path)
    assert list(rows)==list(R.CASES)and len(digests)==2
    assert rows[R.CASES[1]]['value']==1


def test_selected_full_and_partial_support_is_exact_with_no_corner_creation():
    chains=[dict(chain_id='full',observed_points=[[1.,2.],[3.,2.],[5.,2.]],inlier_mask=[True,False,True],
        normal=[0.,1.],offset=-2.,proposal_grade=R.P.FULL_GRADE,actual_finite_inlier_span_px=4.),
        dict(chain_id='partial',observed_points=[[7.,8.],[9.,8.]],inlier_mask=[True,True],
        normal=[0.,1.],offset=-8.,proposal_grade=R.P.PARTIAL_GRADE,actual_finite_inlier_span_px=2.)]
    refs=[dict(native_corner=0,native_neighbor=1,ray_index=0,chain_index=i,chain_id=c['chain_id'])for i,c in enumerate(chains)]
    row=dict(states=['I','U','S','S','S','S','S','S'],initial_geometry=dict(initial_self_hidden=[False,False,True,True,True,True,True,True]),
        observation=dict(selected_chain_refs=refs,chain_observation=dict(edge_records=[dict(chains=chains)])))
    lines=R._selected(row)
    assert np.array_equal(lines[0]['xy'],[[1.,2.],[5.,2.]])and not lines[0]['partial']and lines[1]['partial']


def test_final_overlay_uses_only_actual_policy_active_plan_and_e_has_no_new_lines():
    row=dict(states=['I','U','S','S','S','S','S','S'],initial_geometry=dict(initial_self_hidden=[False,False,True,True,True,True,True,True]))
    group=dict(native_corner=0,native_neighbor=1,actual_inlier_xy=[[10.,20.],[22.,20.]],normal_unit_xy=[0.,1.],
        offset_px=-20.,proposal_grade=R.P.PARTIAL_GRADE,physical_native_edge=[0,1])
    plan=dict(active_groups=[group],new_owner_ids=[0]);result=dict(solution=dict(v29_cost_plan=plan))
    actual,lines=R._active_plan_lines(result,row,R.ARMS[1])
    assert actual is plan and len(lines)==1 and np.array_equal(lines[0]['xy'],group['actual_inlier_xy'])
    assert R._active_plan_lines(Poison(),row,'SAME_FROZEN_E')==({},[])


def test_fallback_overlay_is_labeled_plan_not_an_invented_feasible_correction():
    row=dict(states=['I','U','S','S','S','S','S','S'],initial_geometry=dict(initial_self_hidden=[False,False,True,True,True,True,True,True]))
    result=dict(solution={},WD_family='ORIGINAL_WD',new_geometry_edge_cost_plans=dict(ORIGINAL_WD=dict(active_groups=[],new_owner_ids=[])))
    plan,lines=R._active_plan_lines(result,row,R.ARMS[1])
    assert not lines and plan['new_owner_ids']==[]
