from copy import deepcopy
import pytest
from . import rerank_observed_bi_edge_population_v4_30 as R


class Poison:
    def __getitem__(self,key):raise AssertionError('Excluded chain payload read')


def fixture():
    line=dict(normal=[0.,2.],offset=-38.,observed_points=[[1.,19.],[9.,19.]],inlier_mask=[True,True],
        guide_reference=dict(geometry_ref_id='original'),guide_reference_id='g',chain_id='c')
    obs=dict(edge_records=[dict(native_corner=0,native_neighbor=1,chains=[line],neighbor_image_coordinate_used=False)])
    node=dict(name='0-1/0:0',native_edge=[0,1],provenance=['0:0'],normal=[0.,1.],offset=-19.,
        actual_inliers=[[1.,19.],[9.,19.]],proposal_grade='ACTUAL')
    return obs,dict(selected_line_names=[node['name']],line_candidates=[node])


def test_explicit_root_GOs_precede_all_IO(monkeypatch):
    monkeypatch.setattr(R.C,'read',lambda p:pytest.fail('No IO before root GO'))
    with pytest.raises(PermissionError):R.prepare()
    with pytest.raises(PermissionError):R.rerank()


def test_exact_raw_generation_and_normal43_population_separate():
    assert R.RAW_OBSERVATION_GENERATION=='FRESH_GUIDED_NMS_TLS_PARTIAL_DIRECTED_GRAPH_WINNER_V29'
    assert R.SELECTION_PROFILE==R.F.SELECTION_PROFILE
    assert not R.F.RECIPE['old_V29_noise_validated_for_new_population']


def test_same_exact_refs_with_both_BI_owners_and_unknown_mate_no_newpoint():
    obs,graph=fixture();before=deepcopy((obs,graph));states=['B','I']+['S']*6;hidden=[False,False]+[True]*6
    refs=R.rebuild_selected_references(graph,obs,states,hidden)
    assert refs[0]['eligible_BI_owner_ids']==[0,1]
    assert refs[0]['endpoint_provenance'][1]['model_only_other_BI_endpoint']is True
    assert refs[0]['endpoint_provenance'][1]['full_2D_point_observation_created']is False
    states[1]='U';refs=R.rebuild_selected_references(graph,obs,states,hidden)
    assert refs[0]['eligible_BI_owner_ids']==[0]
    assert (obs,graph)==before


def test_initial_S_gate_precedes_actual_chain_payload_and_zero_preserved():
    obs,graph=fixture();obs['edge_records'][0]['chains']=Poison()
    with pytest.raises(ValueError):R.rebuild_selected_references(graph,obs,['B','S']+['U']*6,[False]*8)
    assert R.rebuild_selected_references(dict(selected_line_names=[],line_candidates=[]),Poison(),['S']*8,[True]*8)==[]


def test_raw_TLS_or_membership_changes_fail_closed():
    obs,graph=fixture();obs['edge_records'][0]['chains'][0]['offset']=-37.
    with pytest.raises(ValueError):R.rebuild_selected_references(graph,obs,['B','I']+['U']*6,[False]*8)
    obs,graph=fixture();graph['selected_line_names'].append('absent')
    with pytest.raises(ValueError):R.rebuild_selected_references(graph,obs,['B','I']+['U']*6,[False]*8)


def test_source_union_does_not_overwrite_historical_hashes():
    mapping={'old':'a'}
    with pytest.raises(ValueError):R._union(mapping,{'old':'b'})
    assert mapping=={'old':'a'}


def test_gzip1_writer_same_JSON_values_and_original_reader_roundtrip(tmp_path):
    data=[dict(id='CAL0',zero=[],none=None,v=[1.,2.]),dict(id='CAL1',unicode='한글')]
    path=tmp_path/'rows.jsonl.gz';R.save_rows_fast(path,iter(data))
    assert list(R.C.rows(path))==data
    with pytest.raises(FileExistsError):R.save_rows_fast(path,data)


def test_central_source_input_and_protection_maps_verified_then_unioned(monkeypatch):
    verified=[];monkeypatch.setattr(R.D.R,'_verify',lambda mapping:verified.append(dict(mapping)))
    central=dict(source_sha256={'core':'a'},input_sha256={'planinput':'b'},protected_user_sha256={'user':'c'})
    source={};inputs={};protected={};R.bind_central_maps(source,inputs,protected,central)
    assert verified==list(central.values())
    assert source==central['source_sha256']and inputs==central['input_sha256']and protected==central['protected_user_sha256']
    with pytest.raises(ValueError,match='conflict'):R.bind_central_maps({'core':'different'}, {}, {}, central)


def test_node_native_ids_must_match_ray_before_actual_chain_payload():
    obs,graph=fixture();graph['line_candidates'][0]['native_edge']=[0,4]
    obs['edge_records'][0]['chains']=Poison()
    with pytest.raises(ValueError,match='original ray'):R.rebuild_selected_references(graph,obs,['B','I']+['U']*6,[False]*8)


def test_worker_releases_large_rows_and_mask_before_roundtrip(monkeypatch,tmp_path):
    import weakref
    import numpy as np
    class WeakDict(dict):pass
    refs={};calls=[]
    def read(path,entry):
        calls.append(True)
        if len(calls)==1:
            row=WeakDict(id='CAL',mask_binding=dict(path='mask',sha256='x'));refs['row']=weakref.ref(row);return row
        assert all(value()is None for value in refs.values())
        return dict(selection_profile=R.SELECTION_PROFILE)
    def rerank(row,cfg,mask):
        refs['mask']=weakref.ref(mask);new=WeakDict(selection_profile=R.SELECTION_PROFILE);refs['new']=weakref.ref(new)
        return new,dict(id='CAL')
    class Context(dict):
        def __enter__(self):return self
        def __exit__(self,*args):pass
    monkeypatch.setattr(R.OLD,'_read_single',read);monkeypatch.setattr(R.C,'sha',lambda p:'x')
    monkeypatch.setattr(R.np,'load',lambda *a,**kw:Context(pallet_mask_clean=np.zeros((2,2))))
    monkeypatch.setattr(R,'rerank_cached_row',rerank);monkeypatch.setattr(R,'save_rows_fast',lambda *a:None)
    monkeypatch.setattr(R.OLD,'_index_row',lambda row,path:dict(id='CAL'))
    entry,record=R._worker(dict(id='CAL'),{},tmp_path)
    assert entry['id']==record['id']=='CAL'and len(calls)==2
