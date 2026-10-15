"""Independent saved CAL3 query equations; no selector/optimizer imports."""
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
OUT = Path(__file__).parent
RUN = B/'initial_only_observed_ray_CAL_cases_v4_32'
PLAN = B/'initial_only_observed_ray_CAL_plan_v4_32/PREDECLARATION.json'
PLAN_SHA = '0318f505b17f0d1c6d1529477d02ccbd9354373fc1b8c97b2b7fcead25f9731f'
SEAL_SHA = 'eac238f222a469f0d1eee3263b359dbd4eb09292a91781afc0b5de34644b6fb0'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def one(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        value = json.loads(next(f))
        if any(line.strip() for line in f):
            raise ValueError('One original row required')
    return value


def main(*, GO=False):
    if GO is not True:
        raise PermissionError('Explicit cached CAL3 audit GO before IO')
    if (OUT/'AUDIT_V1.json').exists():
        raise FileExistsError('Preserve the independent audit')
    checks, rows = [], []

    def check(name, value):
        checks.append((name, bool(value)))
        if not value:
            raise ValueError(name)

    check('plan exact', sha(PLAN) == PLAN_SHA)
    check('execution seal exact', sha(RUN/'RERANK_SEAL.json') == SEAL_SHA)
    plan, seal = read(PLAN), read(RUN/'RERANK_SEAL.json')
    check('external guides absent', plan['external_pose_guides'] is None and not plan['reconstruct_old_guide_bank'])
    for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
        for rel, digest in plan[key].items():
            check(key+':'+rel, sha(ROOT/rel) == digest)
    for rel, digest in seal['assets'].items():
        check('asset:'+rel, sha(RUN/rel) == digest)
    result = read(RUN/'RESULTS.json')
    check('same ordered three', [r['id'] for r in result['cases']] == [c['id'] for c in plan['cases']])
    for index, case in enumerate(plan['cases']):
        raw = one(ROOT/case['spool_relative'])
        check('canonical original row', canonical(raw) == case['canonical_row_sha256'])
        check('CAL only', raw['partition'] == 'CAL' and raw['id'] == case['id'])
        saved = one(RUN/'graphs'/f'case{index}.jsonl.gz')
        graph, geometry, states = saved['graph'], raw['initial_geometry'], raw['states']
        points = {}
        for m in raw['visible_measurements']:
            k = m['native_corner']
            if geometry['initial_self_hidden'][k] or states[k] not in ('B', 'I'):
                continue
            points[k] = np.asarray(m['xy'], float)
        selected = set(graph['selected_line_names'])
        check('strict current graph source', saved['graph_geometry_source'] == 'ORIGINAL_INITIAL_ONLY'
              and saved['external_pose_guides_consumed'] == 0)
        selected_nodes = [n for n in graph['line_candidates'] if n['name'] in selected]
        check('one node per physical edge', len(selected_nodes) == len(selected)
              == len({tuple(n['native_edge']) for n in selected_nodes}))
        observed = []
        kinds = Counter()
        for node in selected_nodes:
            normal = np.asarray(node['normal'], float)
            length = np.linalg.norm(normal)
            check('finite nonzero TLS normal', np.isfinite(length) and length > 0)
            n, offset = normal/length, float(node['offset'])/length
            for reference in node['chain_references']:
                ri, ci = reference['ray_index'], reference['chain_index']
                ray = raw['observation']['chain_observation']['edge_records'][ri]
                chain = ray['chains'][ci]
                k, mate = ray['native_corner'], ray['native_neighbor']
                check('initial S and anchor BI', not geometry['initial_self_hidden'][k]
                      and not geometry['initial_self_hidden'][mate] and states[k] in ('B', 'I')
                      and states[mate] != 'S')
                ref = chain['guide_reference']
                kind = ref['guide_kind']
                kinds[kind] += 1
                allowed = {'FROZEN17C_ORIGINAL_MODEL_SEED': 'ORIGINAL_INITIAL',
                           'MEASURED_BI_ENDPOINT_DIRECTION': 'MEASURED_BI_ENDPOINTS'}
                check('strict query kind and source', kind in allowed and ref['guide_id'] == allowed[kind])
                check('raw exact reference', reference['guide_reference'] == ref
                      and reference['guide_reference_id'] == chain['guide_reference_id']
                      == chain['generating_guide_reference_id'] == ref['guide_reference_id'])
                check('reference content digest', canonical({a: b for a, b in ref.items() if a != 'guide_reference_id'})
                      == ref['guide_reference_id'])
                anchor_error = float(np.linalg.norm(np.asarray(ref['actual_search_anchor_xy'])-points[k]))
                check('anchor original q within metadata tolerance', anchor_error <= 1e-6)
                if states[mate] in ('B', 'I'):
                    delta = points[mate]-points[k]
                else:
                    delta = np.asarray(geometry['initial_projection_xy'][mate])-geometry['initial_projection_xy'][k]
                check('native reference at least10px', np.linalg.norm(delta) >= 10.)
                fixed = delta/np.linalg.norm(delta)
                actual = np.asarray(ref['actual_search_direction_xy'], float)
                actual /= np.linalg.norm(actual)
                cosine = float(actual@fixed)
                check('fixed signed ray exact', cosine > 0 and np.allclose(actual, fixed, atol=1e-8, rtol=0))
                distances = [abs(float(n@points[j])+offset) for j in (k, mate) if j in points]
                check('normal reach43px', max(distances) <= 43.+1e-8)
                observed.append(dict(line_name=node['name'], raw_reference=f'{ri}:{ci}',
                    kind=kind, anchor_error_px=anchor_error, signed_cosine=cosine,
                    maximum_original_BI_normal_distance_px=max(distances)))
        check('summary selection count', len(selected) == result['cases'][index]['selected_count'])
        rows.append(dict(id=case['id'], selected_count=len(selected),
            selected_ref_count=len(observed), kinds=dict(kinds), selected_checks=observed))
        del raw, saved
    value = dict(schema='INDEPENDENT_CAL3_SELECTED_QUERY_EQUATION_REPLAY_V32',
        UTC=datetime.now(timezone.utc).isoformat(), checks=len(checks), failures=[], cases=rows,
        plan_sha256=PLAN_SHA, execution_seal_sha256=SEAL_SHA, source_sha256=sha(__file__),
        method='standalone numpy equations; no association/ranker/optimizer module imported',
        no_new_selector_optimizer_RGB_NN_reference_or_metric_reads=True,
        graph_optimality_finite_edge_identity_or_accuracy_certified=False)
    path = OUT/'AUDIT_V1.json'
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    (OUT/'AUDIT_SEAL_V1.json').write_text(json.dumps(dict(audit_sha256=sha(path),
        source_sha256=sha(__file__), plan_sha256=PLAN_SHA, execution_seal_sha256=SEAL_SHA), indent=2)+'\n')
    print(dict(checks=len(checks), failures=0, audit_sha256=sha(path)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-CAL3-audit-GO', action='store_true')
    main(GO=parser.parse_args().root_CAL3_audit_GO)
