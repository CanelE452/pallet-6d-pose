"""Seal fixed all95 saved-family tangent re-ranking; no fits/labels/images.

This is an objective-at-saved-poses diagnostic, not continuous optimization.
Lambda0/.25/1 are fixed by the root protocol before computation. All95 and
every fallback remain. TEST was already reused and consulted in earlier
experiments; this seal precedes only the new selection's metric computation.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import numpy as np
from . import common as C
from . import joint_ot_pnp as V
from . import statewise_evaluation_v4 as E
from . import partial_bi_tangent_budget_diagnosis_v4_13 as T

FIX = C.DOC/'nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix'
STRENGTHS = (0., .25, 1.)


def run(version='c'):
    source=FIX/'boundary_recovery_v1'/('face_partial_BI_outline_eval_v4_13'+version)
    output=FIX/'independent_review'/('tangent_budget_family_rerank_v4_13'+version)
    if output.exists():raise FileExistsError('Preserve prior diagnostic '+str(output))
    protocol=C.read(source/'PROTOCOL.json');seal=C.read(source/'PREDICTION_SEAL.json')
    journal=source/'PREDICTIONS.jsonl.gz'
    if (seal['rows']!=95 or not seal['before_new_reference_metric_read']
            or not seal['all2arms_complete_before_score'] or C.sha(journal)!=seal['predictions_sha256']
            or C.sha(source/'PROTOCOL.json')!=seal['protocol_sha256']):
        raise ValueError('Complete sealed95 predictions required')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):
        E._verify_sources(protocol[key])
    rows=list(C.rows(journal))
    if [r['id'] for r in rows]!=protocol['ids'] or sum(r['partition']=='CAL' for r in rows)!=46 or sum(r['partition']=='TEST' for r in rows)!=49:
        raise ValueError('Exact ordered CAL46/TEST49 with all fallbacks required')
    sources={str(Path(p).relative_to(C.ROOT)):C.sha(p) for p in (Path(__file__),Path(T.__file__),Path(V.__file__),Path(C.__file__))}
    inputs={str(p.relative_to(C.ROOT)):C.sha(p) for p in (source/'PROTOCOL.json',source/'PREDICTION_SEAL.json',journal)}
    output.mkdir(parents=True)
    plan=dict(schema='FIXED_PARTIAL_BI_TANGENT_SAVED_FAMILY_RERANK_V4_13',version=version,
        inverse_variance_weights=list(STRENGTHS),huber_scale=protocol['solver_configuration']['huber_scale'],
        source_sha256=sources,input_sha256=inputs,source_population={'CAL':46,'TEST':49,'severe':0},
        recipe='same old effective marginal tangent variance / lambda, reliability fixed, same pseudo-Huber',
        tangent_source='archived old observed B/I xy, preserving exact source; may be RGB-corrected or backbone fallback',
        no_old_normal_likelihood_reintroduced=True,all_OT_U_physical_feasibility_kept_by_saved_pose_admission=True,
        no_runtime_GT_or_ID_routing=True,TEST_reused_iterative=True,parameters_chosen_by_GT=False,
        new_labels_opened=False,new_RGB_reads=0,new_pose_fits=0,new_CAL_fits=0,
        comparison_is_saved_two_family_winners_only_not_global_reoptimization=True,
        all95_selection_seal_before_any_new_selected_pose_metrics=True)
    C.write(output/'PLAN.json',plan)
    for p,h in sources.items():
        dest=output/'code_snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(C.ROOT/p,dest)
        if C.sha(dest)!=h:raise ValueError('Source snapshot changed')
    choices=[];counts={part:{str(w):0 for w in STRENGTHS} for part in ('CAL','TEST')}
    source_counts={}
    for row in rows:
        arm=row['arms']['FACE_AWARE_PARTIAL_BI_OLD_Q'];typed=row['face_observation']['boundary_observations'];evaluated=[]
        for record in typed:
            k=record['native_corner']
            if row['states'][k] not in ('B','I') or row['initial_geometry']['initial_self_hidden'][k]:
                raise ValueError('Only gated actual B/I archives allowed')
            source_name=record['association_provenance']['original_measurement_source']
            source_counts[source_name]=source_counts.get(source_name,0)+1
        for family in arm['families']:
            solution=family['solution'];entry=dict(name=family['name'],accepted=solution['accepted'],original_objective=family['objective'],supplements={})
            evaluated.append(entry)
            if not solution['accepted']:continue
            R=np.asarray(solution['R']);translation=np.asarray(solution['t']);Q=np.asarray(family['physical_registration_Q'])
            if not V._valid_rotation(R) or not V._valid_rotation(Q) or not np.allclose(R@Q,solution['R_physical'],atol=1e-7,rtol=0):
                raise ValueError('Saved native/physical family binding mismatch')
            q,depth=V.project_native(np.asarray(family['native_xyz']),R,translation,np.asarray(row['initial_geometry']['K']))
            if not np.allclose(q,solution['projected_native'],atol=1e-8,rtol=0) or np.any(depth<=0):
                raise ValueError('Actual family model projection mismatch')
            objective=float(family['objective'])
            if not np.isfinite(objective):raise ValueError('Finite admitted objective required')
            for strength in STRENGTHS:
                extra=T.typed_tangent_supplement(q,row['states'],typed,
                    inverse_variance_weight=strength,huber_scale=plan['huber_scale'])
                entry['supplements'][str(strength)]={**extra,'augmented_objective_at_saved_pose':objective+extra['cost']}
        selected={}
        for strength in STRENGTHS:
            candidates=[(f['supplements'][str(strength)]['augmented_objective_at_saved_pose'],index,f)
                        for index,f in enumerate(evaluated) if f['accepted']]
            if candidates:
                value,index,f=min(candidates,key=lambda item:(item[0],item[1]));name=f['name']
            else:
                value=None;name=arm['selected_family'];index=None
            if strength==0 and name!=arm['selected_family']:
                raise ValueError('Lambda0 must preserve parent stable objective selector exactly')
            changed=name!=arm['selected_family'];counts[row['partition']][str(strength)]+=int(changed)
            pose=arm['arms'][name]['actual_pose']
            selected[str(strength)]=dict(family=name,augmented_objective_at_saved_pose=value,
                changed_from_parent=changed,accepted_family_index=index,source_pose_sha256=E._digest(pose),
                no_new_pose_fit=True,unique_solution_claimed=False)
        choices.append(dict(id=row['id'],partition=row['partition'],states=row['states'],
            typed_native_ids=[r['native_corner'] for r in typed],parent_selected_family=arm['selected_family'],
            family_evaluations=evaluated,choices=selected))
    for key in ('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    E._verify_sources(sources);E._verify_sources(inputs)
    C.save_rows(output/'SELECTIONS.jsonl.gz',choices)
    C.write(output/'SELECTION_SEAL.json',dict(schema='FIXED_TANGENT_SAVED_FAMILY_SELECTION_SEAL_V4_13',
        rows=95,CAL=46,TEST=49,severe=0,ordered_ids=protocol['ids'],selections_sha256=C.sha(output/'SELECTIONS.jsonl.gz'),
        plan_sha256=C.sha(output/'PLAN.json'),source_sha256=sources,input_sha256=inputs,
        before_any_new_selection_metrics=True,new_GT_labels_read=False,new_RGB_reads=0,new_pose_fits=0,
        all_fallbacks_and_native_incompatible_frames_kept=True))
    C.write(output/'SUMMARY.json',dict(changed_family_counts=counts,archived_measurement_source_counts=source_counts,
        controls_fixed_before_new_metric_reads=True,no_lambda_selected=True,no_WD_solution_or_pose_improvement_claim=True,
        current_all49_max_translation_le5cm_goal_complete=False,selection_seal_sha256=C.sha(output/'SELECTION_SEAL.json')))
    print('FIXED_ALL95_TANGENT_RE_RANK_SEALED',version,counts,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--version',choices=('c','d'),default='c')
    run(parser.parse_args().version)
