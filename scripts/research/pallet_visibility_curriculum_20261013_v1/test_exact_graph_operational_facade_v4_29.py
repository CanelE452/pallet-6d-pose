"""Private hook equivalence on synthetic graphs; no actual image or model IO."""
from copy import deepcopy
import json
import pytest
from . import exact_graph_operational_facade_v4_29 as A
from .test_geometry_graph_exact_memo_v4_29 import graph_fixture


@pytest.mark.parametrize('limit',[1,4096])
def test_new_rank_context_each_call_and_full_graph_exact(limit):
    args,kw,obs=graph_fixture();cfg=dict(graph_configuration=dict(maximum_beam_states=5,reported_joint_alternatives=5))
    options=dict(**kw,configuration=cfg,pose_guides=None,visibility_context=None,pallet_mask=args[1])
    expected=A.P._rank_fresh(obs,args[2],args[3],args[4],**options);diagnostics=[]
    rank=A.make_rank_adapter(maximum_cache_entries=limit,intersection_cache_limit=1,diagnostics=diagnostics)
    for _ in range(2):
        actual=rank(obs,args[2],args[3],args[4],**options)
        assert json.dumps(actual,allow_nan=True)==json.dumps(expected,allow_nan=True)
    assert len(diagnostics)==2 and all('cache_info'in r for r in diagnostics)


def test_private_observer_changes_only_rank_global_and_preserves_keyword_defaults():
    parent=A.P.observe_joint_geometry_edge_evidence;before={k:id(v)for k,v in parent.__globals__.items()}
    observer=A.make_observer()
    assert observer.__code__ is parent.__code__ and observer.__globals__ is not parent.__globals__
    assert observer.__kwdefaults__==parent.__kwdefaults__
    assert observer.__globals__['_rank_fresh']is not parent.__globals__['_rank_fresh']
    for k,v in parent.__globals__.items():
        if k!='_rank_fresh':assert observer.__globals__[k]is v
    assert before=={k:id(v)for k,v in parent.__globals__.items()}


def test_original_observe_one_code_and_initialized_model_captured_without_calls(monkeypatch):
    from . import observe_geometry_guided_edges_v4_29 as V
    model=object();monkeypatch.setattr(V,'_MODEL',model)
    before={k:id(v)for k,v in V.observe_one.__globals__.items()}
    observer=A.make_observe_one()
    assert observer.__code__ is V.observe_one.__code__
    assert observer.__globals__['_MODEL']is model
    assert observer.__globals__['P']is not V.P
    assert observer.__globals__['P'].observe_joint_geometry_edge_evidence.__code__ is V.P.observe_joint_geometry_edge_evidence.__code__
    assert before=={k:id(v)for k,v in V.observe_one.__globals__.items()}
