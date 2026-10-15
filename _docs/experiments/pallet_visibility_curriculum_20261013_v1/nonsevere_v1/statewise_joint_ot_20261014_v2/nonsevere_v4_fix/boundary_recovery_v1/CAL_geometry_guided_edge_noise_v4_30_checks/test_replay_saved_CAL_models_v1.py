"""Synthetic audit-code invariance, with no model/error/reference IO."""
import ast
import importlib.util
from pathlib import Path
import numpy as np
import pytest

HERE=Path(__file__).resolve().parent

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

NEW=load(HERE/'replay_saved_CAL_models_v1.py','new_saved_error_audit')
OLD=load(NEW.OLD_SCRIPT,'old_saved_error_audit')


def test_original_numeric_functions_AST_exact():
    def functions(path):
        return {f.name:ast.dump(f,include_attributes=False)for f in ast.parse(path.read_text()).body if isinstance(f,ast.FunctionDef)}
    old,new=functions(NEW.OLD_SCRIPT),functions(HERE/'replay_saved_CAL_models_v1.py')
    for name in ('balanced','evaluate','near','digest'):assert old[name]==new[name]


def test_same_synthetic_equalweights_and_scalar_gradient():
    component=dict(normal_error_px=3.,angular_sine_error=.1,normal_floor_px=1.,actual_inlier_span_px=50.,
        conditional_variance_floor=[1.,2./2502.],mixture_component_prior=1.)
    records=[dict(id=str(i),partition='CAL',compatible=True,session='a'if i<2 else'b',groups=[dict(components=[component])],tangents=[])for i in range(4)]
    a=OLD.balanced(records,'line',None);b=NEW.balanced(records,'line',None)
    assert a[:3]==b[:3]and np.array_equal(a[3],b[3])
    assert np.array_equal(a[3],[.25]*4)
    av,ag=OLD.evaluate({},a[2],a[3],'line',[4.,.01]);bv,bg=NEW.evaluate({},b[2],b[3],'line',[4.,.01])
    assert av==bv and np.array_equal(ag,bg)


def test_new_family_profile_and_root_GO_before_all_IO(monkeypatch):
    assert NEW.FAMILY.endswith('_V30')and NEW.POPULATION.startswith('V30_')
    assert NEW.RAW=='FRESH_GUIDED_NMS_TLS_PARTIAL_DIRECTED_GRAPH_WINNER_V29'
    assert NEW.PROFILE.startswith('ALL_ELIGIBLE_NAMED_BI_ENDPOINTS')
    monkeypatch.setattr(NEW,'read',lambda p:pytest.fail('IO before separate root GO'))
    with pytest.raises(PermissionError):NEW.main()
