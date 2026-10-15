"""Independent scalar reconstruction against analytic runtime kernels."""
import numpy as np
import pytest

from . import audit_geometry_guided_edges_v4_29 as A
from . import geometry_guided_edge_problem_v4_29 as RUNTIME
from .test_geometry_guided_edge_problem_v4_29 import fixture


@pytest.mark.parametrize('partial,two,policy',[
    (False,False,'ALL_BI_REPLACE'),(False,True,'ALL_BI_REPLACE'),
    (True,False,'ALL_BI_REPLACE'),(True,False,'RGB_NONFALLBACK_PRESERVE'),
    (False,True,'RGB_NONFALLBACK_PRESERVE')])
def test_manual_factor_sum_and_filtered_guards(partial,two,policy):
    base,bundle,_,R,t=fixture(partial=partial,two=two)
    problem=RUNTIME.GeometryGuidedEdgeProblem(base,bundle,source_policy=policy,tangent_gamma=.2)
    task=dict(states=base.states,initial_self_hidden=[s=='S' for s in base.states],
        spec=dict(visible=[a['source_record'] for a in base.prepartial_bound_source_archives.values()]))
    for perturb in ([0.]*6,[.01,-.02,.005,.008,-.011,.025]):
        p=RUNTIME.V._pose_vector(R,t,base.scale)+perturb
        trial_R,trial_t=RUNTIME.V._pose(p,base.scale)
        checked=problem.evaluate(trial_R,trial_t)
        independent=A.recompute_native(task,{},trial_R,trial_t,problem.plan,.2,raw=base.guard_base)
        assert all(passed for passed,_ in independent['checks'])
        assert independent['objective']==pytest.approx(checked['objective'],abs=1e-9)
        assert independent['accepted']==checked['accepted']
        first=A.cost_buckets(independent['native_costs']);second=A.cost_buckets(checked['native_costs'])
        assert first.keys()==second.keys()
        assert all(first[k]==pytest.approx(second[k],abs=1e-9) for k in first)


def test_scalar_reconstruction_detects_modified_source_archive():
    base,bundle,_,R,t=fixture(partial=True)
    problem=RUNTIME.GeometryGuidedEdgeProblem(base,bundle,source_policy='ALL_BI_REPLACE',tangent_gamma=.2)
    task=dict(states=base.states,initial_self_hidden=[s=='S' for s in base.states],
        spec=dict(visible=[a['source_record'] for a in base.prepartial_bound_source_archives.values()]))
    problem.plan['corner_plans'][0]['source_archive']['original_observed_xy'][0]+=1.
    independent=A.recompute_native(task,{},R,t,problem.plan,.2,raw=base.guard_base)
    assert not all(passed for passed,_ in independent['checks'])
