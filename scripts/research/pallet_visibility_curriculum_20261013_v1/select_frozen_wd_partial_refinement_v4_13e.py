"""Fix W/D correspondence from sealed OLD G10, then read same-family d pose.

No image, GT, line fitting, pose fitting or metric enters prediction. OLD G10
uses its full B/I and pre-existing legacy partial evidence plus global outline.
The d normal-localization stage may revoke further complete 2D likelihoods;
it cannot reselect the already inferred W/D correspondence in this control.
Every95 frame, failed pose, ambiguous branch and original native flag remains.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import shutil

import numpy as np

from . import common as C
from . import statewise_evaluation_v4 as E
from . import evaluate_face_partial_BI_outline_v4_13d as D

OLD = D.BASELINE
PARTIAL = D.OUTPUT
OUTPUT = OLD.parent / 'frozen_wd_partial_refinement_eval_v4_13e'
ARMS = ('OLD_G10', 'D_JOINT_RESELECT', 'FROZEN_WD_FACE_AWARE_PARTIAL_BI')
RULE = 'OLD_G10.selected_family fixed for every frame; exact d nested same-family refined arm, including its fallback'


def select_same_family(old, partial):
    """Pure predicted-correspondence routing; no performance/GT inputs."""
    for key in ('id', 'partition', 'session', 'image_sha256', 'states'):
        if old[key] != partial[key]:
            raise ValueError('Exact old/d frame binding required: ' + key)
    new = partial['arms'][D.ARMS[1]]
    for key in ('image_sha256', 'states', 'frozen_BI_fit', 'u_region_observation'):
        if old[key] != new[key]:
            raise ValueError('Unchanged archived observation binding required: ' + key)
    name = old['selected_family']
    if name not in ('ORIGINAL_WD', 'SWAPPED_WD'):
        raise ValueError('Explicit original or swapped family required')
    old_family = next(f for f in old['families'] if f['name'] == name)
    family = next(f for f in new['families'] if f['name'] == name)
    for key in ('native_xyz', 'cf_extents', 'physical_registration_Q'):
        if not np.allclose(old_family[key], family[key], rtol=0, atol=1e-12):
            raise ValueError('Selected family native geometry/registration differs: ' + key)
    selected = deepcopy(new['arms'][name])
    if selected['WD_family'] != name:
        raise ValueError('Nested refined arm belongs to another W/D family')
    pose = selected['actual_pose']
    Q = np.asarray(family['physical_registration_Q'])
    if (not np.allclose(pose['cf_extents'], family['cf_extents'], atol=1e-12, rtol=0)
            or not np.allclose(pose['physical_registration_Q'], Q, atol=1e-12, rtol=0)
            or not np.allclose(np.asarray(pose['R_cf']) @ Q, pose['R_physical'], atol=1e-7, rtol=0)):
        raise ValueError('Same-family refined physical pose registration differs')
    # Preserve numeric pose/objective/observations. Carry initial family
    # uncertainty forward; fixing a correspondence never proves uniqueness.
    initial_ambiguity = deepcopy(old['family_ambiguity'])
    within_ambiguity = deepcopy(selected['ambiguity_audit'])
    ambiguous = bool(initial_ambiguity.get('detected') or within_ambiguity.get('material_ambiguity'))
    selected['method'] = ARMS[2]
    selected['hierarchical_correspondence_audit'] = dict(rule=RULE, frozen_family=name,
        initial_family_ambiguity=initial_ambiguity, refined_within_family_ambiguity=within_ambiguity,
        d_joint_reselection_ambiguity=deepcopy(new['family_ambiguity']),
        initial_family_or_refined_pose_globally_unique_proved=False,
        old_pose_used_as_output=False, refined_d_same_family_numeric_pose_exact=True,
        all_failed_refined_families_kept_as_own_explicit_fallback=True,
        new_pose_fits=0, GT_inputs=False, frame_ID_or_metric_parameter_routing=False)
    if ambiguous:
        selected['ambiguity_audit']['material_ambiguity'] = True
        selected['ambiguity_audit']['frozen_family_choice_does_not_prove_uniqueness'] = True
        selected['solution'].setdefault('ambiguity', {})['detected'] = True
        selected['solution']['ambiguity']['initial_family_ambiguity_preserved'] = bool(initial_ambiguity.get('detected'))
        selected['solution']['OT_solved'] = False
        selected['solution']['OT_unique_solution_claimed'] = False
        readout = selected['keypoint_readout']
        readout['OT_solved'] = False
        readout['OT_unique_solution_claimed'] = False
        for k, state in enumerate(selected['states']):
            if state in ('O', 'T') and readout['reported_points'][k] is not None:
                readout['point_sources'][k] = 'AMBIGUOUS_FROZEN_FAMILY_JOINT_PROJECTION'
    return selected


def _load():
    sources, inputs, protected, result = {}, {}, None, {}
    protocols = {}
    for key, folder in (('old', OLD), ('partial', PARTIAL)):
        protocol = C.read(folder / 'PROTOCOL.json'); seal = C.read(folder / 'PREDICTION_SEAL.json')
        journal = folder / 'PREDICTIONS.jsonl.gz'
        if (seal['rows'] != 95 or not seal['before_new_reference_metric_read']
                or C.sha(journal) != seal['predictions_sha256']
                or C.sha(folder / 'PROTOCOL.json') != seal['protocol_sha256']):
            raise ValueError('Complete sealed original95 source required')
        for field in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
            E._verify_sources(protocol[field])
        if protected is not None and protected != protocol['protected_user_sha256']:
            raise ValueError('Protected user-file contracts differ')
        protected = protocol['protected_user_sha256']
        sources.update(protocol['source_sha256']); inputs.update(protocol['input_sha256'])
        rows = list(C.rows(journal))
        if [r['id'] for r in rows] != protocol['ids']:
            raise ValueError('Exact ordered original95 IDs required')
        result[key] = rows; protocols[key] = protocol
        for path in (journal, folder / 'PREDICTION_SEAL.json', folder / 'PROTOCOL.json'):
            inputs[str(path.relative_to(C.ROOT))] = C.sha(path)
    if [r['id'] for r in result['old']] != [r['id'] for r in result['partial']]:
        raise ValueError('Same ordered old/d whole95 population required')
    if sum(r['partition'] == 'CAL' for r in result['old']) != 46 or sum(r['partition'] == 'TEST' for r in result['old']) != 49:
        raise ValueError('Keep original CAL46/TEST49, no failure/native filtering')
    obs_seal = C.read(PARTIAL / 'OBSERVATION_SEAL.json')
    dseal = C.read(PARTIAL / 'PREDICTION_SEAL.json')
    if (obs_seal['rows'] != 95 or obs_seal['pose_solve_calls'] != 0 or obs_seal['GT_inputs']
            or C.sha(PARTIAL / 'OBSERVATION_SEAL.json') != dseal['observation_seal_sha256']
            or C.sha(PARTIAL / 'OBSERVATIONS.jsonl.gz') != obs_seal['observations_sha256']):
        raise ValueError('Independent sealed95 actual partial observation contract required')
    for path in (PARTIAL / 'OBSERVATION_SEAL.json', PARTIAL / 'OBSERVATIONS.jsonl.gz'):
        inputs[str(path.relative_to(C.ROOT))] = C.sha(path)
    return result['old'], result['partial'], sources, inputs, protected, protocols


def predict(output=OUTPUT):
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError('Preserve every earlier selection experiment')
    old, partial, sources, inputs, protected, parents = _load()
    for module in (__file__, C.__file__, E.__file__, D.__file__):
        path = Path(module); sources[str(path.relative_to(C.ROOT))] = C.sha(path)
    for hashes in (sources, inputs, protected): E._verify_sources(hashes)
    output.mkdir(parents=True)
    C.write(output / 'PROTOCOL.json', dict(schema='FROZEN_WD_THEN_PARTIAL_LOCALIZATION_ALL95_V4_13E',
        arms=list(ARMS), rule=RULE, ids=[r['id'] for r in old], primary_population=dict(CAL=46, TEST=49, severe=0),
        source_sha256=sources, input_sha256=inputs, protected_user_sha256=protected,
        OLD_complete_BI_and_legacy_partial_plus_outline_family_choice=True,
        d_same_family_numeric_pose_not_OLD_numeric_pose=True, old_or_new_GT_metric_selector=False,
        initial_family_choice_ambiguity_preserved=True, refined_family_failures_not_replaced_by_other_family=True,
        no_new_RGB_line_state_model_or_pose_fit=True, TEST_reused_iterative_exploratory=True,
        hypothesis_motivated_posthoc=True, selector_fixed_before_own_GT_metric_read=True,
        global_optimality_or_uniqueness_claimed=False, all95_fallback_native_flagged_rows_kept=True))
    for relative, digest in sources.items():
        dest = output / 'code_snapshot' / relative; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(C.ROOT / relative, dest)
        if C.sha(dest) != digest: raise ValueError('Snapshot changed')
    selected = []
    for a, b in zip(old, partial):
        if b['arms'][D.ARMS[0]] != a: raise ValueError('D exact unchanged old archive differs')
        frozen = select_same_family(a, b)
        new = b['arms'][D.ARMS[1]]
        selected.append(dict(id=a['id'], partition=a['partition'], session=a['session'], image=a['image'],
            image_sha256=a['image_sha256'], states=a['states'], frozen_family=a['selected_family'],
            d_reselected_family=new['selected_family'], selected_model_geometry={k: deepcopy(next(f for f in new['families'] if f['name'] == a['selected_family'])[k])
                for k in ('native_xyz', 'cf_extents', 'physical_registration_Q')},
            arms={ARMS[0]:deepcopy(a['arms']['SELECTED_JOINT_OUTLINE']),
                  ARMS[1]:deepcopy(new['arms']['SELECTED_JOINT_OUTLINE']), ARMS[2]:frozen},
            source_partial_observation=deepcopy(b['merged_partial_observation']),
            d_exact_whole_old_numeric_cache_reuse=b['exact_whole_old_numeric_cache_reuse'],
            initial_family_ambiguity=deepcopy(a['family_ambiguity']),
            contract=dict(new_pose_fits=0,GT_inputs=False,frozen_BI_archives_OT_U_states_unchanged=True,
                exact_d_refined_same_family_numeric_pose=True,no_OLD_pose_fallback_selector=True)))
    for hashes in (sources, inputs, protected): E._verify_sources(hashes)
    C.save_rows(output / 'PREDICTIONS.jsonl.gz', selected)
    C.write(output / 'PREDICTION_SEAL.json', dict(rows=95, CAL=46, TEST=49, severe=0,
        predictions_sha256=C.sha(output / 'PREDICTIONS.jsonl.gz'), protocol_sha256=C.sha(output / 'PROTOCOL.json'),
        source_sha256=sources, input_sha256=inputs, before_new_reference_metric_read=True,
        selection_rule=RULE,new_pose_fits=0,all_failed_ambiguous_and_native_flagged_frames_kept=True))
    print(dict(output=str(output),sealed=95,family_reselection_prevented=sum(r['frozen_family']!=r['d_reselected_family'] for r in selected)),flush=True)


def score(output=OUTPUT):
    from .scoring import pose_metric, distribution
    output=Path(output).resolve(); protocol=C.read(output / 'PROTOCOL.json'); seal=C.read(output / 'PREDICTION_SEAL.json')
    if (seal['rows']!=95 or not seal['before_new_reference_metric_read']
            or C.sha(output / 'PREDICTIONS.jsonl.gz')!=seal['predictions_sha256']
            or C.sha(output / 'PROTOCOL.json')!=seal['protocol_sha256']): raise ValueError('Seal all95 before metric read')
    for field in ('source_sha256','input_sha256','protected_user_sha256'): E._verify_sources(protocol[field])
    rows=list(C.rows(output / 'PREDICTIONS.jsonl.gz'))
    if [r['id'] for r in rows]!=protocol['ids']: raise ValueError('Ordered95 required')
    targets={}
    for part,count in (('CAL',46),('TEST',49)):
        path=E.COHORT / ('OOF_pose_v1/ISOLATED_CAL_SCORING.json' if part=='CAL' else 'test_reused_v1/ISOLATED_TEST_SCORING.json')
        labels=C.read(path)['rows']; population=[r for r in rows if r['partition']==part]
        if len(population)!=count: raise ValueError('Keep every original frame')
        values=[dict(id=r['id'],frozen_family=r['frozen_family'],d_reselected_family=r['d_reselected_family'],
            metrics={a:pose_metric(r['arms'][a]['actual_pose'],labels[r['id']]['target']['truth']) for a in ARMS},
            accepted={a:r['arms'][a]['accepted'] for a in ARMS}) for r in population]
        summary={a:dict(translation_cm=distribution([r['metrics'][a]['translation_cm'] for r in values]),
            rotation_deg=distribution([r['metrics'][a]['rotation_deg'] for r in values]),
            translation_le5=sum(r['metrics'][a]['translation_cm']<=5 for r in values),
            successes_5cm_5deg=sum(r['metrics'][a]['success_5cm_5deg'] for r in values),
            fallback_count=sum(not r['accepted'][a] for r in values),
            translation_over5_rows=[r for r in values if r['metrics'][a]['translation_cm']>5]) for a in ARMS}
        C.write(output / f'{part}_SCORE.json',dict(partition=part,count=count,rows=values,summary=summary,
            target_binding=E._binding(path),all_failed_native_flagged_and_ambiguous_kept=True,
            TEST_reused_exploratory=True,no_runtime_GT_or_metric_routing=True))
        targets[part]=E._binding(path)
        print(part,{a:{k:v for k,v in info.items() if k!='translation_over5_rows'}
                    for a,info in summary.items()},flush=True)
    for field in ('source_sha256','input_sha256','protected_user_sha256'): E._verify_sources(protocol[field])
    C.write(output / 'SCORE_SEAL.json',dict(prediction_seal=E._binding(output / 'PREDICTION_SEAL.json'),
        score_sha256={p:C.sha(output / f'{p}_SCORE.json') for p in ('CAL','TEST')},target_bindings=targets))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('command',choices=('predict','score'))
    parser.add_argument('--output',type=Path,default=OUTPUT); args=parser.parse_args()
    predict(args.output) if args.command=='predict' else score(args.output)
