"""Independent post-seal verification of v17 native-evidence controls.

No fitting, RGB query, reference-label decoding or per-frame method routing.
The frozen input's observations, CAL model and current guards are rebuilt;
candidate energy arithmetic, complete population, fallback and S/U readout
contracts are checked separately from the evaluator's claimed acceptance.
"""
from pathlib import Path
from collections import Counter
import argparse
import math
import numpy as np

from . import common as C
from . import statewise_evaluation_v4 as E
from . import evaluate_cached_native_edges_v4_17 as CACHE
from . import native_edge_problem_v4_17 as N


def audit(output, destination=None):
    output = Path(output).resolve()
    destination = Path(destination).resolve() if destination else output/'independent_root_audit_v1'
    if destination.exists():
        raise FileExistsError('Preserve completed prior audit')
    protocol = C.read(output/'PROTOCOL.json'); seal = C.read(output/'PREDICTION_SEAL.json')
    failures = []; checks = 0; max_difference = 0.; fallback_counts = Counter()

    def check(value, label, ident=None, arm=None):
        nonlocal checks
        checks += 1
        if not bool(value):
            failures.append(dict(check=label, id=ident, arm=arm))

    check(seal['rows'] == 238, 'complete238 seal')
    check(C.sha(output/'PREDICTIONS.jsonl.gz') == seal['predictions_sha256'], 'sealed prediction bytes')
    check(C.sha(output/'PROTOCOL.json') == seal['protocol_sha256'], 'sealed protocol bytes')
    for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
        E._verify_sources(protocol[key]); check(True, 'all bound '+key+' unchanged')
    input_path = output/'INPUTS.jsonl.gz'
    if not input_path.exists():
        input_path = CACHE.OUTPUT/'INPUTS.jsonl.gz'
    tasks = {t['id']: t for t in C.rows(input_path)}
    model = CACHE._verified_model(); angle = CACHE.ANGLE.authentication_angular_parameters(model)
    rows = list(C.rows(output/'PREDICTIONS.jsonl.gz'))
    check([r['id'] for r in rows] == protocol['ids'], 'all original IDs in order')
    check(Counter(r['partition'] for r in rows) == Counter(CAL=46, TEST=49, REMAINDER=143), '46+49+143 severe0')
    for row in rows:
        task = tasks[row['id']]; ident = row['id']
        check(E._digest(row['arms']['SAME_FROZEN_E']) == E._digest(task['baseline']), 'exact unchanged e baseline', ident)
        check(row['states'] == task['states'], 'all fixed states preserved', ident)
        for arm, value in row['arms'].items():
            if arm == 'SAME_FROZEN_E':
                continue
            key = 'boundary_original' if 'ORIGINAL' in arm else 'boundary_merged'
            family = value['solution']['family']; name = family['name']; family = task['families'][name]
            pose = value['actual_pose']; R = np.asarray(pose['R_cf']); t = np.asarray(pose['centroid'])
            check(np.allclose(R @ np.asarray(family['physical_registration_Q']), pose['R_physical'], atol=1e-10, rtol=0), 'physical Q unchanged', ident, arm)
            visible = task['spec']['visible']; records = task[key]; spec = task['spec']
            bundle = CACHE.AUTH.authenticate_supported_native_edges(family['native_xyz'], task['states'], visible,
                records, spec['image_hw'], initial_self_hidden=task['initial_self_hidden'],
                image_binding=dict(sha256=task['image_sha256'], image_hw=spec['image_hw']), config=spec['config'], **angle)
            problem = N.prepare_native_edge_problem(family['native_xyz'], spec['K'], task['states'], visible,
                spec['domains'], spec['image_hw'], family['coarse_pose'], config=spec['config'],
                boundary_observations=records, u_regions=task['u_regions'], edge_bundle=bundle)
            fresh = problem.evaluate(R, t)
            if value['accepted']:
                check(fresh['accepted'], 'selected pose passes rebuilt current guards', ident, arm)
                difference = abs(fresh['objective'] - value['solution']['objective'])
                max_difference = max(max_difference, difference)
                check(difference <= 1e-8, 'native objective recomputed', ident, arm)
                check(fresh['S_image_terms'] == fresh['U_image_terms'] == 0, 'S U no image point costs', ident, arm)
            else:
                fallback_counts[arm] += 1
                coarse = task['families']['ORIGINAL_WD']['coarse_pose']
                check(name == 'ORIGINAL_WD' and np.array_equal(R, coarse['R']) and np.array_equal(t, coarse['t']), 'fallback own original coarse only', ident, arm)
            readout = value['keypoint_readout']
            for k, state in enumerate(task['states']):
                if state == 'S':
                    check(readout['point_sources'][k] == 'POSE_ONLY_MODEL_PROJECTION', 'S model projection only', ident, arm)
                elif state == 'U':
                    check(readout['reported_points'][k] is None, 'U region only no point', ident, arm)
            selection = value['candidate_selection']; contenders = selection['candidates']
            for candidate in contenders:
                check(abs(math.fsum(candidate['native_cost_vector']) - candidate['native_energy']) <= 1e-10,
                    'native components sum without mask', ident, arm)
            chosen = next((c for c in contenders if c['name'] == selection['selected_name']), None)
            if chosen is not None:
                minimum = min(c['native_energy'] for c in contenders)
                check(chosen['native_energy'] <= minimum + 1e-10, 'global native minimum over feasible contenders', ident, arm)
                if value['accepted']:
                    check(abs(chosen['native_energy'] - fresh['objective']) <= 1e-8, 'selection uses same likelihood as reported pose', ident, arm)
    for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
        E._verify_sources(protocol[key])
    result = dict(schema='INDEPENDENT_NATIVE_EDGE_CONTROL_ROOT_AUDIT_V4_17', checks=checks,
        failures=failures, failure_count=len(failures), rows=len(rows), primary_test192=192,
        CAL46_not_primary=True, severe=0, maximum_recomputed_objective_difference=max_difference,
        fallback_counts=dict(fallback_counts), no_reference_labels_decoded=True,
        no_new_fits_RGB_or_NN_calls=True, audit_source=E._binding(Path(__file__)),
        prediction_seal=E._binding(output/'PREDICTION_SEAL.json'),
        complete_primary_accuracy_goal_not_claimed=True)
    destination.mkdir(parents=True)
    C.write(destination/'AUDIT.json', result)
    C.write(destination/'SEAL.json', dict(audit_sha256=C.sha(destination/'AUDIT.json')))
    print(dict(checks=checks, failures=len(failures), max_objective_difference=max_difference), flush=True)
    if failures:
        raise ValueError('Independent current native-edge audit failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--destination', type=Path)
    a = parser.parse_args(); audit(a.output, a.destination)
