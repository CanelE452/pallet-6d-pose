"""Pure and toy-only regression checks for the saved native audit driver."""
import importlib.util
from pathlib import Path

import pytest

path=Path(__file__).with_name('audit_saved_fixed_CAL_smoke_v1.py')
spec=importlib.util.spec_from_file_location('saved_native_audit_driver',path)
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)


def test_GO_and_exact_seal_format_precede_data_IO(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('data IO before explicit GO/seal')
    monkeypatch.setattr(A,'read',forbidden);monkeypatch.setattr(A,'sha',forbidden)
    with pytest.raises(PermissionError):A.main()
    with pytest.raises(ValueError):A.main(root_saved_native_audit_GO=True)


def test_original_BI_binding_does_not_copy_S_U_O_T_payloads():
    class Poison(dict):
        def __getitem__(self,key):
            if key=='native_corner':return dict.__getitem__(self,key)
            raise AssertionError('excluded payload')
    source=dict(states=['B','S','U','O','T','I','B','I'],
        initial_self_hidden=[False,True,False,False,False,False,False,False],
        visible_measurements=[Poison(native_corner=k) for k in (1,2,3,4)]+[
            dict(native_corner=0,xy=[2.,3.],source='SYNTHETIC',fit_covariance_px2=[[1.,0.],[0.,1.]],fit_reliability_weight=.5)])
    value=A.BI_archive(source)
    assert len(value)==1 and value[0]['native_corner']==0 and value[0]['state']=='B'


def test_genuine_toy_smoke_bindings_fresh_origins_starts_and_native_equations():
    from scripts.research.pallet_visibility_curriculum_20261013_v1.test_smoke_clean_fresh_BI_runtime_v4_32 import packet,P
    value=packet();result=P.smoke_worker_packet(value)
    audit=A.audit_values(value,result)
    assert audit['failure_count']==0,audit['failures']
    assert audit['static_probes']>=8 and len(audit['fresh_banks'])==2
