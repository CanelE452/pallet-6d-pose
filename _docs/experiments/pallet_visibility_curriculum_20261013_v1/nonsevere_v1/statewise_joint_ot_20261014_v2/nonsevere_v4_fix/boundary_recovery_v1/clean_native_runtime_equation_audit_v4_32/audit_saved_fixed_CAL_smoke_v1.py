"""Posthoc independent query/seed/start/native replay of one sealed CAL smoke."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path('/home/minjae/Documents/github/pallet-pose')
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).parent
B = OUT.parent
RUN = B/'clean_fresh_BI_actual_runtime_smoke_v4_32'
SMOKE_PLAN = B/'clean_fresh_BI_actual_runtime_smoke_plan_v4_32/PREDECLARATION.json'
spec = importlib.util.spec_from_file_location('independent_positive_native', OUT/'independent_positive_native_replay_v1.py')
REPLAY = importlib.util.module_from_spec(spec)
spec.loader.exec_module(REPLAY)
POLICIES = ('ALL_BI_REPLACE', 'RGB_NONFALLBACK_PRESERVE')
FAMILIES = ('ORIGINAL_WD', 'SWAPPED_WD')
COARSE = 'INDEPENDENT_INITIAL_COARSE_PREDICTOR'
FRESH = 'FRESH_OBSERVED_BI_SQPNP_SUBSET'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    return value


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def BI_archive(source):
    result = []
    for record in source['visible_measurements']:
        k = record['native_corner']
        if source['initial_self_hidden'][k] or source['states'][k] not in ('B', 'I'):
            continue
        item = dict(native_corner=k, state=source['states'][k], xy=plain(record['xy']), source=record['source'])
        for key in ('covariance_px2', 'fit_covariance_px2', 'fit_reliability_weight'):
            if key in record: item[key] = plain(record[key])
        result.append(item)
    return sorted(result, key=lambda item:item['native_corner'])


def named_probe_pose_matches(family, starts, sample, scale):
    by_name = {start['name']:start for start in starts['starts']}
    by_name['INDEPENDENT_COARSE_STATIC'] = family['coarse_pose']
    expected = by_name.get(sample['name'])
    if expected is None:
        return False
    R,t,p = np.asarray(sample['R']),np.asarray(sample['native_t']),np.asarray(sample['pose_vector'])
    return bool(p.shape == (6,) and np.allclose(R,expected['R'],atol=2e-12,rtol=0)
        and np.allclose(t,expected['t'],atol=2e-12*scale,rtol=0)
        and np.allclose(REPLAY.Rotation.from_rotvec(p[:3]).as_matrix(),R,atol=1e-10,rtol=0)
        and np.allclose(p[3:]*scale,t,atol=2e-12*scale,rtol=0))


def audit_values(packet, result):
    checks, samples, banks = [], [], []
    def check(condition, reason):
        checks.append(dict(passed=bool(condition), reason=reason))
    source = packet['clean_input']
    check(set(packet) == REPLAY.WORKER_KEYS and packet['schema'] == REPLAY.WORKER_SCHEMA, 'positive seven-key worker boundary')
    check(result['ready'] is True and source['identifiers'] == result['identifiers'], 'ready fixed CAL identity')
    check(source['identifiers']['partition'] == 'CAL', 'CAL only')
    check(result['packet_sha256'] == digest(packet), 'exact serialized positive packet')
    clean = result['clean']
    check(clean['clean_task_sha256'] == digest({k:v for k,v in clean.items() if k != 'clean_task_sha256'}), 'immutable clean task digest')
    archive = BI_archive(source)
    check(set(result['records']) == set(POLICIES) and set(clean['families']) == set(FAMILIES), 'both policies and WD families')
    calls = result['observed_call_counts']
    for key in ('continuous_refiner','full_solver','scipy_optimizer','legacy_evaluator_run','legacy_refiner_wrapper'):
        check(calls[key] == 0, 'profile forbidden path zero:'+key)
    check(calls['fresh_BI_task'] == 1 and calls['fresh_PnP_bank'] == 2, 'one clean task and two fresh geometric banks')
    SQPNP_calls = 0
    for name in FAMILIES:
        bank, family = clean['fresh_seed_banks'][name], clean['families'][name]
        camera = np.asarray(source['K'], float);camera = camera/camera[2,2]
        binding = digest(dict(native_xyz=plain(family['native_xyz']), K=camera.tolist(),
            states=source['states'], initial_self_hidden=source['initial_self_hidden'], admitted_BI=archive,
            initial_coarse_R=family['coarse_pose']['R'], initial_coarse_t=family['coarse_pose']['t'],
            physical_Q=family['physical_registration_Q']))
        check(bank['input_binding_sha256'] == binding and bank['admitted_BI_archive'] == archive,
              'fresh bank original BI covariance/source/weight input binding:'+name)
        proposals = {p['name']:p for p in bank['proposals']}
        check(len(proposals) == len(bank['proposals']), 'unique fresh bank names:'+name)
        check(bank['admitted_BI_ids'] == [m['native_corner'] for m in archive], 'only gated sorted BI IDs:'+name)
        check(bank['fresh_seed_count'] == sum(p['source'] == FRESH for p in proposals.values()), 'fresh seed count:'+name)
        check(bank['SQPNP_calls'] <= bank['subset_count'] <= 127, 'fixed SQPnP subset budget:'+name)
        SQPNP_calls += bank['SQPNP_calls']
        body = np.asarray(family['native_xyz'], float)
        Q = np.asarray(family['physical_registration_Q'], float)
        scale = max(float(np.linalg.norm(np.ptp(body,axis=0))),1e-3)
        for proposal in proposals.values():
            R,t = np.asarray(proposal['R']),np.asarray(proposal['t'])
            check(proposal['source'] in (COARSE,FRESH) and proposal['provenance']['input_binding_sha256'] == binding,
                  'only own coarse or newly bound BI subset source')
            check(np.allclose(R.T@R,np.eye(3),atol=1e-7,rtol=0) and np.isclose(np.linalg.det(R),1.,atol=1e-7,rtol=0),
                  'fresh proper rotation')
            check(np.all((body@R.T+t)[:,2] > 1e-9*scale), 'all fresh model vertices positive depth')
            check(np.allclose(proposal['centroid'],R@body.mean(axis=0)+t,atol=1e-12,rtol=0)
                  and np.allclose(proposal['R_physical'],R@Q,atol=1e-12,rtol=0), 'fresh native translation/center/Q')
            if proposal['source'] == COARSE:
                check(np.array_equal(R,family['coarse_pose']['R']) and np.array_equal(t,family['coarse_pose']['t']), 'own independent coarse exact')
            else:
                subset = proposal['provenance']['subset_native_ids']
                check(set(subset) <= set(bank['admitted_BI_ids']) and len(subset) >= 3
                      and proposal['provenance']['extrinsic_guess_used'] is False, 'fresh BI subset without extrinsic guess')
        banks.append(dict(family=name,proposals=len(proposals),fresh_seeds=bank['fresh_seed_count'],SQPNP_calls=bank['SQPNP_calls']))
        for policy in POLICIES:
            record = result['records'][policy][name]
            check(record['family'] == family and record['fresh_seed_bank'] == bank, 'same clean family/bank shared across policies')
            plan = record['current_cost_plan']
            check(record['current_cost_plan_sha256'] == digest(plan), 'complete current cost plan digest')
            check(plan['source_policy'] == policy, 'exact current native source policy')
            check([p['name'] for p in record['all_fresh_seed_evaluations']] == list(proposals), 'all seed statuses retained in order')
            starts = record['start_packet']
            check(starts['start_packet_sha256'] == digest({k:v for k,v in starts.items() if k != 'start_packet_sha256'}), 'native start packet digest')
            check(starts['fresh_PnP_input_sha256'] == binding and starts['clean_task_sha256'] == clean['clean_task_sha256'], 'starts bound to fresh input and clean geometry')
            check(len(starts['starts']) <= 7 and starts['selection']['selected_start_names'] == [s['name'] for s in starts['starts']], 'fixed native6 plus coarse budget')
            for start in starts['starts']:
                expected = family['coarse_pose'] if start['name'] == 'OWN_INITIAL_COARSE' else proposals.get(start['name'])
                check(expected is not None and np.allclose(start['R'],expected['R'],atol=2e-12,rtol=0)
                      and np.allclose(start['t'],expected['t'],atol=2e-12*scale,rtol=0), 'start from generated fresh bank or own coarse only')
            check([s['name'] for s in record['samples']] == ['INDEPENDENT_COARSE_STATIC']+starts['selection']['selected_start_names'], 'all fixed native probes retained')
            check(np.isclose(record['pose_vector_scale'],scale,atol=1e-12,rtol=0), 'independent pose vector scale')
            for sample in record['samples']:
                check(named_probe_pose_matches(family,starts,sample,scale), 'static probe pose exactly bound to its named own coarse or native start')
                numeric = REPLAY.replay(packet,family,plan,sample['R'],sample['native_t'],
                    expected=sample['evaluation'],expected_gradient=sample['objective_gradient'])
                checks.extend(numeric['checks'])
                check(np.isclose(numeric['objective'],sample['objective'],atol=1e-8,rtol=1e-8), 'independent static objective snapshot')
                samples.append(dict(policy=policy,family=name,name=sample['name'],**numeric))
    check(calls['SQPNP_Generic'] == SQPNP_calls, 'observed fresh SQPnP calls equal the two bank inventories')
    return dict(checks=len(checks),failure_count=sum(not c['passed'] for c in checks),
        failures=[c for c in checks if not c['passed']],fresh_banks=banks,samples=samples,
        static_probes=len(samples),only_static_native_start_probes_replayed=True,
        all_seed_evaluation_rows_retained_but_acceptance_and_costs_not_independently_replayed=True,
        no_new_PnP_optimizer_RGB_NN_or_reference_calls_by_audit=True,
        physical_edge_identity_native_index_mapping_global_optimum_or_pose_accuracy_certified=False)


def main(*,root_saved_native_audit_GO=False,expected_smoke_seal_sha256=None):
    if root_saved_native_audit_GO is not True:
        raise PermissionError('Separate root saved-native smoke audit GO before data IO')
    if (not isinstance(expected_smoke_seal_sha256,str) or len(expected_smoke_seal_sha256) != 64
            or any(c not in '0123456789abcdef' for c in expected_smoke_seal_sha256)):
        raise ValueError('Explicit same smoke seal SHA before data IO')
    if (OUT/'ACTUAL_AUDIT_PLAN_V1.json').exists(): raise FileExistsError('Preserve every audit attempt')
    if sha(RUN/'SEAL.json') != expected_smoke_seal_sha256: raise ValueError('Same approved smoke seal required')
    seal = read(RUN/'SEAL.json');plan = read(SMOKE_PLAN)
    sources = {str(Path(__file__).relative_to(ROOT)):sha(__file__),
        str((OUT/'independent_positive_native_replay_v1.py').relative_to(ROOT)):sha(OUT/'independent_positive_native_replay_v1.py')}
    declaration = dict(UTC=datetime.now(timezone.utc).isoformat(),root_saved_native_audit_GO=True,
        source_sha256=sources,smoke_seal_sha256=expected_smoke_seal_sha256,
        saved_static_native_probes_and_fresh_input_provenance_only=True,actual_numeric_packet_result_decodes_before_this_plan=0)
    (OUT/'ACTUAL_AUDIT_PLAN_V1.json').write_text(json.dumps(declaration,indent=2)+'\n')
    if sha(SMOKE_PLAN) != seal['plan_sha256'] or plan['first_CAL_id'] != seal['case']:
        raise ValueError('Exact first canonical CAL plan/case required')
    for key in ('source_sha256','input_sha256','protected_user_sha256','original_CAL_file_sha256'):
        for relative,value in plan[key].items():
            if sha(ROOT/relative) != value: raise ValueError('Frozen saved smoke binding changed:'+relative)
    for relative,value in seal['assets'].items():
        if sha(RUN/relative) != value: raise ValueError('Saved packet/result changed:'+relative)
    packet,result = read(RUN/'WORKER_PACKET.json'),read(RUN/'RESULT.json')
    audit = audit_values(packet,result)
    audit.update(UTC=datetime.now(timezone.utc).isoformat(),source_sha256=sources,
        first_CAL_id=seal['case'],smoke_seal_sha256=expected_smoke_seal_sha256,
        plan_sha256=sha(OUT/'ACTUAL_AUDIT_PLAN_V1.json'))
    target = OUT/'ACTUAL_AUDIT_V1.json';target.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    (OUT/'ACTUAL_AUDIT_SEAL_V1.json').write_text(json.dumps(dict(audit_sha256=sha(target),
        plan_sha256=audit['plan_sha256'],source_sha256=sources,smoke_seal_sha256=expected_smoke_seal_sha256),indent=2)+'\n')
    print(dict(checks=audit['checks'],failures=audit['failure_count'],static_probes=audit['static_probes'],audit_sha256=sha(target)),flush=True)
    if audit['failure_count']: raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-saved-native-audit-GO',action='store_true')
    parser.add_argument('--expected-smoke-seal-sha256')
    args = parser.parse_args()
    main(root_saved_native_audit_GO=args.root_saved_native_audit_GO,
        expected_smoke_seal_sha256=args.expected_smoke_seal_sha256)
