"""Independent complete-BI core reconstruction using the v22 audit kernel."""
from pathlib import Path
import types

from . import audit_bounded_geometric_priors_control_v4_22 as KERNEL
from . import evaluate_complete_bi_bounded_priors_v4_23 as V
from . import refine_grazing_image_chain_mixtures_v4_18c as ORIGINAL

_PLAN_DIGESTS={}


def _problem(task,family,models):
    # Rebuild directly from the pure core, not the experiment's problem
    # factory or subclass. Original measurements still own the hard guards.
    spec=task['spec']
    base=ORIGINAL.BASE.NATIVE.prepare_native_edge_problem(family['native_xyz'],spec['K'],spec['states'],
        spec['visible'],spec['domains'],spec['image_hw'],family['coarse_pose'],
        config={**spec['config'],'prior_weight':0.},boundary_observations=task['boundary_merged'],
        u_regions=task['u_regions'],edge_bundle=dict(edges=[],replaces_normal_factor_refs=[]))
    bundle=V.COMP.authenticate_complete_bi_image_chain_groups(base,task['new_chain_observation'],
        junction_proposals=task['complete_BI_evidence']['junction_proposals'],
        visible_measurements=spec['visible'],initial_self_hidden=task['initial_self_hidden'],
        image_binding=dict(sha256=task['image_sha256'],image_hw=spec['image_hw']),
        initial_projection_xy=task['complete_BI_evidence']['initial_projection_xy'],
        **V.CAL.mixture_noise_parameters(models['COMPLETE']))
    problem=V.COMP.CompleteBIImageChainProblem(base,bundle,overlap_policy=V.COMP.PROTOTYPE_OVERLAP_POLICY)
    _PLAN_DIGESTS[(task['id'],family['name'])]=V.E._digest(problem.plan)
    return problem,bundle


def main():
    C,E=V.C,V.E
    _PLAN_DIGESTS.clear()
    facade=types.SimpleNamespace(**{**V.__dict__,'V':V.OLD.V})
    core=types.SimpleNamespace(**{**ORIGINAL.__dict__,'_problem':_problem,
        '_verified_model':lambda:V._verified_models()['NMS']})
    runner=types.FunctionType(KERNEL.main.__code__,{**KERNEL.main.__globals__,'V':facade,'NATIVE':core,
        '__file__':__file__},'audit_complete_BI_bounded_priors')
    runner(V.OUTPUT)
    checks=0;failures=[]
    for row in C.rows(V.OUTPUT/'PREDICTIONS.jsonl.gz'):
        for arm in V.ARMS[1:]:
            result=row['arms'][arm]
            if not result['accepted']:continue
            plan=result['solution']['complete_BI_replacement_audit'];readout=result['keypoint_readout']
            checks+=1
            if E._digest(plan)!=_PLAN_DIGESTS[(row['id'],result['WD_family'])]:
                failures.append(dict(id=row['id'],arm=arm,reason='saved replacement plan differs from independently rebuilt core'))
            for corner in plan['corner_plans']:
                if not corner['point_cost_revoked']:continue
                k=corner['native_corner'];checks+=1
                if readout['reported_points'][k]is not None:
                    failures.append(dict(id=row['id'],arm=arm,native_corner=k,reason='revoked q reported as precise point'))
            checks+=1
            if result['solution']['replaced_old_normal_factor_count']!=len(plan['replaces_normal_factor_refs']):
                failures.append(dict(id=row['id'],arm=arm,reason='normal replacement count differs'))
    dest=V.OUTPUT/'independent_root_audit_v1'
    C.write(dest/'COMPLETE_BI_READOUT_AND_KERNEL_RECEIPT.json',dict(schema='INDEPENDENT_COMPLETE_BI_READOUT_AND_KERNEL_V23',
        checks=checks,failure_count=len(failures),failures=failures,
        source=E._binding(Path(__file__)),kernel_source=E._binding(Path(KERNEL.__file__)),
        selected_pose_audit=E._binding(dest/'AUDIT.json'),core_problem_factory_independent_of_experiment_driver=True,
        new_image_pose_fits_NN_or_reference_decode=0))
    print(dict(complete_readout_checks=checks,failures=len(failures)),flush=True)
    if failures:raise ValueError('Complete-BI readout/accounting audit failed')


if __name__=='__main__':main()
