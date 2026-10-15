"""Cached-only runner/crop contracts; no real images, scores or pose fitting."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from . import run_directed_geometry_line_cases_v4_28b as R


class Poison(dict):
    def __getitem__(self, key):
        if key == 'native_corner':
            return super().__getitem__(key)
        raise AssertionError('Forbidden S/U payload read')


def test_roi_uses_original_coordinate_model_and_BI_support_not_U_point():
    row = dict(geometry=dict(initial_projection_xy=[[100., 110.], [120., 130.]],
                            image_hw=[480, 640], initial_self_hidden=[False, False, True]),
               states=['B', 'U', 'S'], visible_measurements=[
                   dict(native_corner=0, xy=[95., 105.]), Poison(native_corner=1), Poison(native_corner=2)],
               u_regions=Poison())
    graph = dict(line_candidates=[dict(native_corner=0, actual_inliers=[[90., 100.], [125., 140.]]),
                                  Poison(native_corner=1), Poison(native_corner=2)])
    assert R._roi(row, graph) == [40., 175., 50., 190.]


def test_roi_clips_display_canvas_only_and_does_not_change_support():
    row = dict(geometry=dict(initial_projection_xy=[[-30., -20.], [690., 510.]],
                            image_hw=[480, 640], initial_self_hidden=[False]*2),
               states=['B', 'B'], visible_measurements=[])
    graph = dict(line_candidates=[dict(native_corner=0, actual_inliers=[[-10., 5.], [650., 490.]])])
    before = deepcopy(graph)
    assert R._roi(row, graph) == [0., 640., 0., 480.]
    assert graph == before


def test_combined_keeps_all_cached_accepted_rejected_and_extra_paths():
    base = dict(schema='IMAGE_SUPPORTED_EDGE_CHAINS_V4_17C', edge_records=[
        dict(native_corner=2, native_neighbor=3, chains=[dict(id='full')],
             rejected_chains=[dict(id='partial', reason='insufficient_span')])])
    extra = [dict(native_corner=2, native_neighbor=1, chains=[dict(id='guided')])]
    observed = dict(result=dict(base_observation=base, additional_guided_edge_records=extra))
    original = deepcopy(observed)
    combined = R._combined(observed)
    assert len(combined['edge_records']) == 2
    assert combined['edge_records'][0]['rejected_chains'][0]['id'] == 'partial'
    assert combined['edge_records'][1]['chains'][0]['id'] == 'guided'
    assert observed == original


def test_changed_frozen_input_stops_before_directed_code_mask_or_image(tmp_path, monkeypatch):
    def read(path):
        if Path(path).name == 'INPUT_SEAL.json':
            return dict(plan_sha256='expected', inputs_sha256='expected')
        raise AssertionError('No downstream input may be opened')
    def forbidden(*args, **kwargs):
        raise AssertionError('No measurement or re-ranking before source gate')
    monkeypatch.setattr(R.C, 'read', read)
    monkeypatch.setattr(R.C, 'sha', lambda p:'changed')
    monkeypatch.setattr(R, '_directed', forbidden)
    monkeypatch.setattr(R.np, 'load', forbidden)
    monkeypatch.setattr(R.cv2, 'imread', forbidden)
    with pytest.raises(ValueError, match='Frozen diagnostic'):
        R.rerank(tmp_path)


def test_rerank_never_calls_new_pixel_search_or_NN(tmp_path, monkeypatch):
    monkeypatch.setattr(R, '_verify_run', lambda p:dict(source_sha256={}))
    row = dict(id='B:case', mask_binding=dict(path='mask.npz', key='pallet_mask'),
               image_sha256='a'*64, geometry=dict(image_hw=[20, 30]), states=['B'],
               visible_measurements=[], pose_guides=[])
    previous = dict(result=dict(base_observation=dict(edge_records=[]),
                                additional_guided_edge_records=[], joint_graph=dict(old=True)),
                    expected_edge_visibility={})
    def read(path):
        name = Path(path).name
        if name == 'PLAN.json':return dict(parent='saved_parent')
        if name == 'INPUTS.json':return dict(cases=[row])
        if name == 'B__case.json':return previous
        raise AssertionError(name)
    class Arrays:
        def __enter__(self):return dict(pallet_mask=np.zeros((20, 30)))
        def __exit__(self, *args):return False
    class Directed:
        def rank_ordered_geometry_line_graph(self, chains, geometry, states, visible, **kwargs):
            assert chains['edge_records'] == []
            return dict(selected_line_names=[], line_candidates=[])
    def forbidden(*args, **kwargs):raise AssertionError('New pixel search forbidden')
    written = {}
    monkeypatch.setattr(R.C, 'read', read)
    monkeypatch.setattr(R.C, 'sha', lambda p:'digest')
    monkeypatch.setattr(R.C, 'write', lambda p,v:written.__setitem__(Path(p).name,v))
    monkeypatch.setattr(R.np, 'load', lambda *args,**kw:Arrays())
    monkeypatch.setattr(R.cv2, 'imread', forbidden)
    monkeypatch.setattr(R.OLD.JOINT, 'observe_joint_geometry_guided_lines', forbidden)
    monkeypatch.setattr(R.OLD.VISIBILITY, 'prepare_edge_visibility', lambda *a,**kw:object())
    monkeypatch.setattr(R, '_expected_visibility', lambda *a:dict(counts={}))
    monkeypatch.setattr(R, '_directed', lambda:Directed())
    R.rerank(tmp_path)
    assert written['B__case.json']['cached_actual_chains_unchanged']
    assert written['OBSERVATION_SEAL.json']['new_pixel_searches'] == 0


def test_span_only_partial_missing_RGB_context_is_not_invented():
    class Context:
        def classify_expected_edge(self,k,mate,*,observed_chain_points,mask_context_kind):
            assert mask_context_kind is None
            assert len(observed_chain_points) in (0,2)
            return dict(samples=[],counts=dict(observed_support=0,occluder_supported_gap=0,unsupported_unknown=1))
    row=dict(states=['B','U','S','S','S','S','S','S'],geometry=dict(initial_self_hidden=[False,False]+[True]*6))
    graph=dict(selected_line_names=['short'],line_candidates=[dict(name='short',native_corner=0,native_edge=[0,1],
        actual_inliers=[[10.,20.],[11.,20.]],proposal_grade='PARTIAL_LINE_PROPOSAL',line_record={})])
    result=R._expected_visibility(row,graph,Context())
    assert result['records'][0]['actual_line_context_missing_or_incomplete']
    assert result['partial_line_RGB_context_not_fabricated']
