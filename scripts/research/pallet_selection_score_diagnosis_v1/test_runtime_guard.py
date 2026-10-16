import sys
from types import SimpleNamespace

import pytest

from .runtime_guard import FixedPoseOnly, ForbiddenScientificCall


def test_prohibited_optimizer_body_never_runs_and_profile_restores():
    namespace={}
    exec(compile('def minimize():\n    raise AssertionError("body entered")\n',
                 '/fixture/scipy/optimize/_minimize.py','exec'),namespace)
    previous=sys.getprofile();guard=FixedPoseOnly()
    with pytest.raises(ForbiddenScientificCall):
        with guard:namespace['minimize']()
    assert sys.getprofile() is previous
    assert guard.receipt()['forbidden_function_bodies_executed']==0
    assert guard.blocked_attempts['numeric_optimization:minimize']==1


def test_native_PnP_event_is_blocked_without_calling_PnP():
    guard=FixedPoseOnly()
    with pytest.raises(ForbiddenScientificCall):
        guard._profile(None,'c_call',SimpleNamespace(__name__='solvePnP'))


def test_fixed_residual_function_is_allowed():
    namespace={}
    exec(compile('def _joint_objective_pair(x):\n    return x*x,2*x\n',
                 '/fixture/pallet_visibility_curriculum_20261013_v1/example.py','exec'),namespace)
    with FixedPoseOnly() as guard:
        assert namespace['_joint_objective_pair'](2)==(4,4)
    assert not guard.blocked_attempts
