"""Rebuild all selected O-abstention guards/costs and control invariants."""
from pathlib import Path
import types
import numpy as np

from . import audit_image_chain_mixture_control_v4_18 as AUDIT
from . import evaluate_o_localization_abstention_v4_19 as V


def main():
    facade=types.SimpleNamespace(**{**V.PARENT.OLD.__dict__,**V.__dict__,'MIX':V.PARENT.MIX})
    run=types.FunctionType(AUDIT.main.__code__,{**AUDIT.main.__globals__,'V':facade},AUDIT.main.__name__)
    run(V.OUTPUT)
    C,E=V.C,V.E;output=V.OUTPUT;dest=output/'independent_root_audit_v1'
    parents={t['id']:t for t in C.rows(V.PARENT.OUTPUT/'INPUTS.jsonl.gz')}
    oldrows={r['id']:r for r in C.rows(V.PARENT.OUTPUT/'PREDICTIONS.jsonl.gz')}
    inputs={t['id']:t for t in C.rows(output/'INPUTS.jsonl.gz')}
    model=V._verified_model();checks=0;failures=[];unchanged=0;nested=0;maxcost=0.
    def check(ok,what,ident):
        nonlocal checks
        checks+=1
        if not ok:failures.append(dict(id=ident,check=what))
    for row in C.rows(output/'PREDICTIONS.jsonl.gz'):
        ident=row['id'];task=inputs[ident];source=parents[ident];previous=oldrows[ident]
        expected=V.adapt_task(source)
        check(E._digest(task)==E._digest(expected),'exact pure domain-only input adaptation',ident)
        changed=task['O_localization_abstention']['changed_native_ids']
        for oldarm,newarm in zip(V.PARENT.ARMS[1:],V.ARMS[1:]):
            old,new=previous['arms'][oldarm],row['arms'][newarm]
            if not changed:
                keys=('R_cf','R_physical','centroid','physical_registration_Q','cf_extents')
                check(all(old['actual_pose'][k]==new['actual_pose'][k]for k in keys),
                    'no-O exact pose parity',ident)
                check(old['accepted']==new['accepted']and old['WD_family']==new['WD_family'],
                    'no-O family/feasibility parity',ident)
                unchanged+=1
            # Constraint relaxation preserves this parent's pose feasibility,
            # even when the bounded new optimization chooses another basin.
            if old['accepted']:
                family=task['families'][old['WD_family']]
                problem,_=V.PARENT._problem(task,family,model)
                pose=old['actual_pose'];fresh=problem.evaluate(pose['R_cf'],pose['centroid'])
                check(fresh['accepted'],'parent feasible pose stays feasible after O relaxation',ident)
                delta=abs(fresh['objective']-old['solution']['objective']);maxcost=max(maxcost,delta)
                check(delta<1e-8,'O change adds no likelihood at fixed parent pose',ident)
                nested+=1
            if new['accepted']:
                for k in changed:
                    check(new['keypoint_readout']['point_sources'][k]==
                        'JOINT_CANVAS_FEASIBLE_PROJECTION_O_LOCALIZATION_ABSTAINED_NOT_UNIQUE',
                        'O model hypothesis not precise point',ident)
    for key in('source_sha256','input_sha256','protected_user_sha256'):
        E._verify_sources(C.read(output/'PROTOCOL.json')[key])
    C.write(dest/'CONTROL_INVARIANTS_V1.json',dict(schema='O_LOCALIZATION_ABSTENTION_INDEPENDENT_INVARIANTS_V19',
        checks=checks,failures=failures,failure_count=len(failures),
        no_O_arm_pose_parity_count=unchanged,parent_nested_pose_checks=nested,
        maximum_fixed_pose_likelihood_difference=maxcost,all238_kept=True,
        no_new_pose_fits_or_RGB_NN_or_reference_decodes=True,
        note='Same six-start policy; relaxed feasibility may change the selected optimization starts.',
        source=E._binding(Path(__file__)),kernel_audit_source=E._binding(Path(AUDIT.__file__)),
        selected_pose_audit=E._binding(dest/'AUDIT.json'),prediction_seal=E._binding(output/'PREDICTION_SEAL.json')))
    C.write(dest/'CONTROL_INVARIANTS_SEAL_V1.json',dict(receipt_sha256=C.sha(dest/'CONTROL_INVARIANTS_V1.json')))
    print(dict(extra_checks=checks,failures=len(failures),no_O_pose_parity=unchanged,
        nested_parent_poses=nested,maximum_fixed_pose_likelihood_difference=maxcost),flush=True)
    if failures:raise ValueError('O ablation invariants failed')


if __name__=='__main__':main()
