"""Independently replay selected CAL46 query equations from sealed saved rows."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/home/minjae/Documents/github/pallet-pose')
B = ROOT/'_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1'
RUN = B/'initial_only_observed_ray_observations_v4_32'
OUT = Path(__file__).parent
OWNER_SHA = '18240967686493694afb9f01afa1b2e6c1f5ac64b5461186915b6480f1dd8678'
SEAL_SHA = 'dfef446eb778fae816bce74733b5c15c1c1f81a8df6ca3b1064cad0301750e31'
PROFILE = 'INITIAL_ONLY_ORIGINAL17C_OR_MEASURED_BI_QUERY_FIXED_NATIVE_RAY_V32'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main(*, GO=False):
    if GO is not True:
        raise PermissionError('Root cached CAL46 equation audit GO required before IO')
    if (OUT/'AUDIT_V1.json').exists() or (OUT/'PLAN_V1.json').exists():
        raise FileExistsError('Preserve existing audit/attempt')
    source = sha(__file__)
    write(OUT/'PLAN_V1.json', dict(schema='INDEPENDENT_CAL46_EQUATION_AUDIT_PLAN_V32',
        UTC=datetime.now(timezone.utc).isoformat(), root_cached_audit_GO=True,
        source_sha256=source, CAL_owner_sha256=OWNER_SHA, CAL_observation_seal_sha256=SEAL_SHA,
        population='all 46 sealed CAL rows, retained without error/pose selection',
        NN_RGB_selector_PnP_optimization_noise_fit_reference_or_primary_reads=0,
        method='standalone numpy equations and saved byte/canonical identities; no research module imports',
        actual_physical_edge_identity_or_pose_accuracy_certified=False))
    checks, results = [], []

    def check(name, value):
        checks.append((name, bool(value)))
        if not value:
            write(OUT/'FAILED_ATTEMPT_V1.json', dict(check=name, completed_checks=len(checks),
                completed_rows=results, source_sha256=source))
            raise ValueError(name)

    check('owner exact', sha(RUN/'CAL_OWNER_COMPLETION_RECEIPT_V32_V1.json') == OWNER_SHA)
    check('CAL seal exact', sha(RUN/'CAL_OBSERVATION_SEAL.json') == SEAL_SHA)
    owner, seal = read(RUN/'CAL_OWNER_COMPLETION_RECEIPT_V32_V1.json'), read(RUN/'CAL_OBSERVATION_SEAL.json')
    check('owner PASS', owner['status'] == 'PASS' and owner['failure_count'] == 0)
    for rel, digest in owner['original_CAL_file_sha256'].items():
        check('original CAL53:'+rel, sha(ROOT/rel) == digest)
    for rel, digest in seal['source_sha256'].items():
        check('current sealed source:'+rel, sha(ROOT/rel) == digest)
    index = read(RUN/'CAL_ROW_INDEX.json')['entries']
    summary = read(RUN/'CAL_SUMMARY.json')['records']
    check('46 ordered canonical metadata', len(index) == len(summary) == 46
        and [r['id'] for r in index] == [r['id'] for r in summary] == seal['ordered_ids']
        and len(set(seal['ordered_ids'])) == 46)
    for position, item in enumerate(index):
        path = RUN/'row_spool'/item['path']
        check('spool bytes:'+item['id'], sha(path) == item['sha256'])
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            row = json.loads(next(f))
            check('one saved row:'+item['id'], not any(line.strip() for line in f))
        check('canonical saved row:'+item['id'], canonical(row) == item['canonical_row_sha256'])
        check('uniform CAL identity:'+item['id'], row['partition'] == item['partition'] == 'CAL' and row['id'] == item['id'])
        obs, geometry, states = row['observation'], row['initial_geometry'], row['states']
        check('strict current profile:'+item['id'], row['selection_profile'] == obs['selection_profile'] == PROFILE
            and obs['current_graph_geometry_source'] == 'ORIGINAL_INITIAL_ONLY'
            and obs['external_pose_guides_consumed'] == 0
            and not obs['preserved_old_guide_metadata_used_as_current_scoring_input'])
        graph = obs['joint_graph']
        points = {m['native_corner']: np.asarray(m['xy'], float) for m in row['visible_measurements']
                  if not geometry['initial_self_hidden'][m['native_corner']] and states[m['native_corner']] in ('B', 'I')}
        selected = set(graph['selected_line_names'])
        nodes = [n for n in graph['line_candidates'] if n['name'] in selected]
        check('one line per physical edge:'+item['id'], len(nodes) == len(selected)
            == len({tuple(n['native_edge']) for n in nodes}))
        records, kinds = [], Counter()
        for node in nodes:
            normal = np.asarray(node['normal'], float)
            length = np.linalg.norm(normal)
            check('finite TLS', np.isfinite(length) and length > 0)
            n, offset = normal/length, float(node['offset'])/length
            for reference in node['chain_references']:
                ri, ci = reference['ray_index'], reference['chain_index']
                ray = obs['chain_observation']['edge_records'][ri]
                chain = ray['chains'][ci]
                k, mate = ray['native_corner'], ray['native_neighbor']
                check('native edge identity', tuple(sorted((k, mate))) == tuple(sorted(node['native_edge'])))
                check('nonS original BI anchor', not geometry['initial_self_hidden'][k]
                    and not geometry['initial_self_hidden'][mate] and k in points and states[mate] != 'S')
                ref = chain['guide_reference']
                kind = ref['guide_kind']
                allowed = {'FROZEN17C_ORIGINAL_MODEL_SEED': 'ORIGINAL_INITIAL',
                           'MEASURED_BI_ENDPOINT_DIRECTION': 'MEASURED_BI_ENDPOINTS'}
                check('allowed actual query kind', kind in allowed and ref['guide_id'] == allowed[kind])
                kinds[kind] += 1
                check('all query source aliases', reference['guide_reference'] == ref
                    and reference['guide_reference_id'] == chain['guide_reference_id']
                    == chain['generating_guide_reference_id'] == ref['guide_reference_id'])
                check('query digest', canonical({a: b for a, b in ref.items() if a != 'guide_reference_id'}) == ref['guide_reference_id'])
                error = float(np.linalg.norm(np.asarray(ref['actual_search_anchor_xy'])-points[k]))
                check('actual query original q anchor', error <= 1e-6)
                if mate in points:
                    delta = points[mate]-points[k]
                else:
                    delta = np.asarray(geometry['initial_projection_xy'][mate])-geometry['initial_projection_xy'][k]
                span = float(np.linalg.norm(delta))
                check('native direction span at least10', span >= 10.)
                fixed = delta/span
                actual = np.asarray(ref['actual_search_direction_xy'], float)
                actual /= np.linalg.norm(actual)
                cosine = float(actual@fixed)
                check('fixed signed native direction', cosine > 0 and np.allclose(actual, fixed, atol=1e-8, rtol=0))
                distances = [abs(float(n@points[j])+offset) for j in (k, mate) if j in points]
                check('original BI normal reach43', max(distances) <= 43.+1e-8)
                records.append(dict(line_name=node['name'], raw_reference=f'{ri}:{ci}', kind=kind,
                    original_q_anchor_error_px=error, signed_cosine=cosine,
                    maximum_original_BI_normal_distance_px=max(distances)))
        check('summary count:'+item['id'], len(selected) == summary[position]['selected']
            == item['summary']['selected_physical_edges'] == obs['selected_physical_edges'])
        results.append(dict(id=item['id'], selected_count=len(nodes), selected_ref_count=len(records),
            kinds=dict(kinds), selected_checks=records))
        if (position+1) % 10 == 0:
            print(dict(audited_CAL_rows=position+1, checks=len(checks)), flush=True)
        del row, obs, graph, geometry, nodes
    check('same source after read', sha(__file__) == source)
    check('same CAL seal after read', sha(RUN/'CAL_OBSERVATION_SEAL.json') == SEAL_SHA)
    value = dict(schema='INDEPENDENT_ALL_CAL46_SELECTED_QUERY_EQUATION_REPLAY_V32',
        UTC=datetime.now(timezone.utc).isoformat(), checks=len(checks), failures=[], cases=results,
        selected_lines=sum(r['selected_count'] for r in results),
        selected_references=sum(r['selected_ref_count'] for r in results),
        source_sha256=source, CAL_owner_sha256=OWNER_SHA, CAL_seal_sha256=SEAL_SHA,
        all_53_CAL_file_bytes_and_284_source_pins_checked=True,
        full_1237_input_map_rehashed=False,
        no_new_selector_optimizer_RGB_NN_reference_or_primary_metric_reads=True,
        graph_optimality_finite_edge_identity_or_pose_accuracy_certified=False)
    write(OUT/'AUDIT_V1.json', value)
    write(OUT/'AUDIT_SEAL_V1.json', dict(audit_sha256=sha(OUT/'AUDIT_V1.json'),
        source_sha256=source, plan_sha256=sha(OUT/'PLAN_V1.json'), CAL_seal_sha256=SEAL_SHA))
    print(dict(checks=len(checks), failures=0, CAL_rows=len(results), selected_lines=value['selected_lines'],
        audit_sha256=sha(OUT/'AUDIT_V1.json')), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-CAL46-audit-GO', action='store_true')
    main(GO=parser.parse_args().root_CAL46_audit_GO)
