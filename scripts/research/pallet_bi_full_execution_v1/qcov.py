"""Source-preserving q/covariance diagnostic using immutable saved diagonals.

The four cells are meaningful only when the two diagonals have identical
post-deduplication factors AND physical image-source association. Ineligible
cross cells remain explicit missing interventions, not algorithm failures.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from scripts.research.pallet_bi_conditioned_observation_v1 import predict as P
from scripts.research.pallet_bi_conditioned_observation_v1 import solver as S

SEEDS = (20261018, 20261019, 20261020)
REPRESENTATIVE = 20261018
COVARIANCE_FIELDS = ('covariance_px2', 'fit_covariance_px2', 'fit_reliability_weight')


def read(path):
    path = Path(path)
    with (gzip.open if path.suffix == '.gz' else open)(path, 'rt') as stream:
        return json.load(stream)


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.pending')
    with (gzip.open if path.suffix == '.gz' else open)(pending, 'wt') as stream:
        json.dump(S._json(value), stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
    pending.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): digest.update(block)
    return digest.hexdigest()


def effective(selection):
    visible, partial, dedup = S._deduplicate(selection['full_points'], selection['partial_lines'])
    partial_ids = {r['native_corner'] for r in partial}
    full = {r['native_corner']: r for r in visible if r['native_corner'] not in partial_ids}
    return full, partial, dedup


def covariance(record):
    return np.asarray(record.get('fit_covariance_px2',
        S.SOLVER_CONFIG['covariance_scale'] * np.asarray(record['covariance_px2']) +
        np.asarray(S.SOLVER_CONFIG['state_BI_total_error_floor_px2'][record['state']])), float)


def comparable(first, second):
    """Return all incompatibilities, with no coordinate-dependent admission."""
    a, pa, da = effective(first); b, pb, db = effective(second)
    reasons = []
    if first['states'] != second['states']: reasons.append('AUTOMATIC_STATES_DIFFER')
    if first['partial_lines'] != second['partial_lines']: reasons.append('SUBMITTED_TYPED_PARTIALS_DIFFER')
    if pa != pb: reasons.append('EFFECTIVE_TYPED_PARTIAL_FACTORS_DIFFER')
    if a.keys() != b.keys(): reasons.append('EFFECTIVE_FULL_FACTOR_IDS_DIFFER')
    if da != db: reasons.append('PHYSICAL_SOURCE_OWNERSHIP_OR_SUPPRESSION_DIFFER')
    changed_sources = [k for k in sorted(a.keys() & b.keys())
        if any(a[k].get(key) != b[k].get(key) for key in ('state', 'source', 'physical_source_ids'))]
    if changed_sources: reasons.append('FULL_FACTOR_PHYSICAL_SOURCE_ASSOCIATION_DIFFERS')
    # Even suppressed submitted points are protected: q substitution must not
    # alter ownership and revive a formerly suppressed factor.
    aa = {r['native_corner']: r for r in first['full_points']}
    bb = {r['native_corner']: r for r in second['full_points']}
    if aa.keys() != bb.keys(): reasons.append('SUBMITTED_VISIBLE_IDS_DIFFER')
    return dict(eligible=not reasons, reasons=reasons, changed_source_corner_ids=changed_sources,
        common_effective_full_ids=sorted(a), bi_effective_full_ids=sorted(b),
        common_deduplication=da, bi_deduplication=db)


def mix(q_selection, covariance_selection):
    """Honest counterfactual; the original typed partial/archive is unmodified."""
    eligibility = comparable(q_selection, covariance_selection)
    if not eligibility['eligible']: raise ValueError('Cross intervention invalid: ' + ','.join(eligibility['reasons']))
    result = deepcopy(q_selection)
    donor = {r['native_corner']: r for r in covariance_selection['full_points']}
    partial_ids = {r['native_corner'] for r in result['partial_lines']}
    for row in result['full_points']:
        if row['native_corner'] in partial_ids: continue
        source = donor[row['native_corner']]
        for field in COVARIANCE_FIELDS:
            row.pop(field, None)
            if field in source: row[field] = deepcopy(source[field])
    if result['partial_lines'] != q_selection['partial_lines']: raise AssertionError('Partial changed')
    if effective(result)[2] != effective(q_selection)[2]: raise AssertionError('Source dedup changed')
    return result


def differences(first, second):
    a, pa, da = effective(first); b, pb, db = effective(second)
    shared = sorted(a.keys() & b.keys())
    return dict(full_factor_ids_same=a.keys() == b.keys(), shared_full_factor_n=len(shared),
        q_changed_corner_ids=[k for k in shared if a[k]['xy'] != b[k]['xy']],
        effective_covariance_changed_corner_ids=[k for k in shared if not np.array_equal(covariance(a[k]), covariance(b[k]))],
        conditional_covariance_changed_corner_ids=[k for k in shared if a[k]['covariance_px2'] != b[k]['covariance_px2']],
        effective_partials_same=pa == pb, source_deduplication_same=da == db,
        q_distance_px=[float(np.linalg.norm(np.asarray(a[k]['xy']) - b[k]['xy'])) for k in shared],
        effective_covariance_frobenius_difference=[float(np.linalg.norm(covariance(a[k])-covariance(b[k]))) for k in shared])


def fixed_common_packet(frame, common, bi, q_bit, covariance_bit):
    """Transport to a fixed *logical* graph, with explicit source divergence.

    Every typed partial and its original revoked 2D archive is retained byte
    for byte. Deduplication happens once on COMMON before logical factor IDs
    are assigned. BI donors are existing effective complete 2D observations
    of the same native 3D vertex; a missing donor leaves COMMON unchanged.
    """
    if common['states'] != bi['states']: raise ValueError('Immutable automatic states differ')
    visible, partial, dedup = S._deduplicate(common['full_points'],common['partial_lines'])
    complete, _, _ = effective(common); donors, _, _ = effective(bi)
    result=dict(states=deepcopy(common['states']),full_points=deepcopy(visible),partial_lines=deepcopy(partial))
    factors=[]
    for row in result['full_points']:
        k=row['native_corner']
        if k not in complete:continue
        source=deepcopy(complete[k]);donor=donors.get(k)
        reason=None if donor is not None else 'BI_HAS_NO_EFFECTIVE_COMPLETE_2D_FACTOR_AT_SAME_NATIVE_VERTEX'
        if donor is not None and donor['state']!=source['state']:raise ValueError('Donor automatic role differs')
        q_donor=donor if q_bit and donor is not None else source
        cov_donor=donor if covariance_bit and donor is not None else source
        row['xy']=deepcopy(q_donor['xy'])
        for field in COVARIANCE_FIELDS:
            row.pop(field,None)
            if field in cov_donor:row[field]=deepcopy(cov_donor[field])
        # This legacy parser field explicitly carries logical graph identity
        # in this adapter, never a claim about the new coordinate's RGB owner.
        row['physical_source_ids']=['CANONICAL_LOGICAL_FULL_FACTOR:'+str(k)]
        factors.append(dict(native_corner=k,object_xyz=deepcopy(frame['native_xyz'][k]),dimension=2,
            factor_identity_scope='CANONICAL_LOGICAL_FACTOR_GRAPH_NOT_PHYSICAL_RGB_SOURCE',
            canonical_logical_factor_ids=deepcopy(row['physical_source_ids']),
            canonical_original_source_ids=deepcopy(source.get('physical_source_ids',[])),
            q_donor_source_ids=deepcopy(q_donor.get('physical_source_ids',[])),
            cov_donor_source_ids=deepcopy(cov_donor.get('physical_source_ids',[])),
            q_donor='BI'if q_bit and donor is not None else 'COMMON',
            covariance_donor='BI'if covariance_bit and donor is not None else 'COMMON',
            changed_source_association=(q_donor.get('physical_source_ids')!=source.get('physical_source_ids')or
                cov_donor.get('physical_source_ids')!=source.get('physical_source_ids')),
            missing_BI_donor_fallback_reason=reason))
    provenance=dict(schema='COUNTERFACTUAL_FIXED_COMMON_FACTOR_GRAPH_V1',q_bit=int(q_bit),covariance_bit=int(covariance_bit),
        canonical_COMMON_deduplication=dedup,canonical_COMMON_effective_full_ids=sorted(complete),
        typed_partials_and_revoked_archives_unchanged=True,factors=factors,
        source_identity_scope='logical association fixed; transported q/covariance need not share physical RGB source',
        interpretation='algorithmic q/covariance sensitivity conditional on COMMON graph; not pure physical-source intervention')
    validate_fixed_packet(common,result,provenance)
    return result,provenance


def validate_fixed_packet(common,selection,provenance):
    original_visible,original_partial,_=S._deduplicate(common['full_points'],common['partial_lines'])
    original_full={r['native_corner']:r for r in original_visible};partial_ids={r['native_corner']for r in original_partial}
    if selection['states']!=common['states']or selection['partial_lines']!=original_partial:raise ValueError('Fixed partial/state graph changed')
    if {r['native_corner']for r in selection['full_points']}!=set(original_full):raise ValueError('Canonical native factor set changed')
    actual_full,actual_partial,_=S._deduplicate(selection['full_points'],selection['partial_lines'])
    if len(actual_full)!=len(original_visible)or actual_partial!=original_partial:raise ValueError('Logical-ID pass changed effective graph')
    factors={r['native_corner']:r for r in provenance['factors']}
    for row in selection['full_points']:
        k=row['native_corner']
        if k in partial_ids:
            if row!=original_full[k]:raise ValueError('Original typed revoked archive changed')
        elif k not in factors or row['physical_source_ids']!=factors[k]['canonical_logical_factor_ids']:
            raise ValueError('Explicit logical-factor provenance required')
    return True


def parsed_solver_payload(frame,selection):
    """Exact actual typed-parser output plus solver/geometry/start closure."""
    visible,partial,_=S._deduplicate(selection['full_points'],selection['partial_lines'])
    xyz,K,measurements,hw,boundary=S._prepare(frame['native_xyz'],frame['K'],selection['states'],
        visible,frame['raw_hw'],partial,S.SOLVER_CONFIG)
    # Physical source IDs have already served the deduplication stage. They
    # do not enter the numerical objective. All parser outputs, including the
    # complete original partial archive/provenance, remain in this exact key.
    return S._json(dict(frame={k:frame[k]for k in ('K','native_xyz','raw_hw','initial_pose')},
        states=selection['states'],xyz=xyz,K=K,measurements=measurements,image_hw=hw,boundary=boundary,
        backend=S.contract()))


def run_fixed_common(prior_run,output,plan_path,plan_sha256,*,root_GO=False):
    if root_GO is not True:raise PermissionError('Root GO after frozen counterfactual plan required')
    if sha(plan_path)!=plan_sha256:raise ValueError('Frozen plan changed')
    start=time.monotonic();prior_run,output=Path(prior_run),Path(output);output.mkdir(parents=True,exist_ok=True)
    inventory=read(prior_run/'P0_INPUT_CONTRACT.json')['real']['inventory'];rows=[];new_calls=0;reuses=0
    write(output/'PROTOCOL.json',dict(schema='COUNTERFACTUAL_FIXED_COMMON_FACTOR_GRAPH_V1',frozen_plan_sha256=plan_sha256,
        representative_observation_seed=REPRESENTATIVE,full_denominator=238,family_call_upper_bound=1904,
        no_numeric_reference_inputs=True,source_sha256=sha(__file__),written_before_new_solves=True))
    for ordinal,frame in enumerate(inventory):
        path=output/'frames'/f'{ordinal:04d}.json.gz'
        if path.exists():
            record=read(path)
            if record['frozen_plan_sha256']!=plan_sha256:raise ValueError('Checkpoint plan mismatch')
            new_calls+=record['actual_new_family_optimizer_calls'];reuses+=record['exact_numerical_packet_reuses'];rows.append(record);continue
        saved=read(prior_run/'prediction_frames/real'/f'{ordinal:04d}'/'FRAME_PREDICTIONS.json.gz')
        details=saved['current_observation_and_solver_records'];common=details['OBS_COMMON_s20261018'];bi=details['OBS_BI_s20261018']
        original_payload=parsed_solver_payload(frame,common['selection']);cache={P.digest(original_payload):(original_payload,common['solver_result'],'ACTUAL_COMMON_DIAGONAL')}
        cells={};calls_here=0;reuses_here=0;raw_packets={}
        for name,qbit,cbit in [('Q0S0',0,0),('Q1S0',1,0),('Q0S1',0,1),('Q1S1',1,1)]:
            selection,provenance=fixed_common_packet(frame,common['selection'],bi['selection'],qbit,cbit)
            payload=parsed_solver_payload(frame,selection);signature=P.digest(payload)
            if signature in cache:
                old,result,origin=cache[signature]
                if old!=payload:raise AssertionError('Hash collision')
                calls=0;reuses_here+=1;reused=True
            else:
                # The new adapter validates logical/source transport above;
                # the unchanged normal typed parser and revocation validator
                # are then used by the original numerical solve primitive.
                result=S.solve_frame(frame,selection);calls=result['optimizer_calls'];origin=name;reused=False
                cache[signature]=(payload,result,name)
            calls_here+=calls
            cells[name]=dict(available=True,actual_pose=result['actual_pose'],accepted=result['accepted'],fallback=result['fallback'],
                status=result['status'],selected_family=(result.get('solution')or{}).get('selected_family'),
                family_readouts=family_readouts(result),exact_numerical_solver_payload_sha256=signature,
                exact_after_typed_parse_cache_reused=reused,cache_origin=origin,actual_new_family_optimizer_calls=calls,
                source_transport_provenance=provenance)
            raw_packets[name]=dict(selection=selection,solver_result=result)
        actual_bi_payload=parsed_solver_payload(frame,bi['selection'])
        transported_bi_payload=parsed_solver_payload(frame,raw_packets['Q1S1']['selection'])
        # Equality here is conservative because full original per-record
        # provenance remains in the parsed key. Raw graph/source differences
        # are independently reported even when numerical solves coincide.
        record=dict(id=frame['id'],partition=frame['partition'],session=saved['ledger']['session'],cluster_id=saved['ledger']['cluster_id'],
            ordinal=ordinal,frozen_plan_sha256=plan_sha256,cells=cells,actual_BI_diagonal=bi['solver_result'],
            actual_COMMON_diagonal=common['solver_result'],strict_source_graph_compatibility=comparable(common['selection'],bi['selection']),
            transported_Q1S1_matches_actual_BI_parsed_payload=transported_bi_payload==actual_bi_payload,
            actual_new_family_optimizer_calls=calls_here,exact_numerical_packet_reuses=reuses_here,
            interpretation='algorithmic sensitivity conditional on COMMON logical graph; Q1S1 is not actual BI by default')
        write(output/'packets'/f'{ordinal:04d}.json.gz',raw_packets);write(path,record)
        rows.append(record);new_calls+=calls_here;reuses+=reuses_here
        if new_calls>1904:raise RuntimeError('Preregistered family-call upper bound exceeded')
        if (ordinal+1)%24==0:print(json.dumps(dict(stage='D_FIXED_COMMON_GRAPH',done=ordinal+1,total=238,new_family_calls=new_calls)),flush=True)
    summary=dict(schema='COUNTERFACTUAL_FIXED_COMMON_FACTOR_GRAPH_V1',status='DIAGNOSTIC_ONLY',full_denominator=len(rows),
        actual_new_family_optimizer_calls=new_calls,exact_numerical_packet_reuses=reuses,family_call_upper_bound=1904,
        transported_Q1S1_matches_actual_BI_parsed_payload_frames=sum(r['transported_Q1S1_matches_actual_BI_parsed_payload']for r in rows),
        strict_source_graph_compatible_frames=sum(r['strict_source_graph_compatibility']['eligible']for r in rows),
        source_changing_frames=sum(any(f['changed_source_association']for f in r['cells']['Q1S1']['source_transport_provenance']['factors'])for r in rows),
        missing_BI_donor_frames=sum(any(f['missing_BI_donor_fallback_reason']is not None for f in r['cells']['Q1S1']['source_transport_provenance']['factors'])for r in rows),
        missing_BI_donor_factors=sum(f['missing_BI_donor_fallback_reason']is not None for r in rows for f in r['cells']['Q1S1']['source_transport_provenance']['factors']),
        frozen_plan_sha256=plan_sha256,source_sha256=sha(__file__),wall_seconds=time.monotonic()-start,
        numeric_reference_inputs=False,physical_source_intervention_claimed=False,all_family_endpoints_preserved=True)
    write(output/'SUMMARY.json',summary);return summary


def family_readouts(result):
    """Keep rejected family status and its N0 fallback, plus all saved endpoints."""
    packet = result.get('solution')
    if packet is None: return {}
    output = {}
    for family in packet['families']:
        solution = family['solution']
        pose = dict(available=True, R_cf=solution['R'], R_physical=solution['R_physical'],
            centroid=solution['t'], cf_extents=family['cf_extents'],
            physical_registration_Q=family['physical_registration_Q'], source='QCOV_FIXED_FAMILY') if family['accepted'] else dict(
                available=True, R_cf=family['coarse_pose']['R'],
                R_physical=family['coarse_pose']['R_physical'], centroid=family['coarse_pose']['t'],
                cf_extents=family['cf_extents'], physical_registration_Q=family['physical_registration_Q'],
                source='QCOV_FIXED_FAMILY_N0_FALLBACK')
        output[family['name']] = dict(actual_pose=pose, accepted=family['accepted'],
            fallback=not family['accepted'], status=solution['status'], objective=family['objective'],
            all_visited_endpoints=deepcopy(solution.get('candidates', [])),
            original_solution=deepcopy(solution))
    return output


def run(prior_run, output):
    start = time.monotonic(); prior_run, output = Path(prior_run), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    inventory = read(prior_run/'P0_INPUT_CONTRACT.json')['real']['inventory']
    if len(inventory) != 238: raise ValueError('Full real238 required')
    rows = []; new_calls = 0; packet_reuse = 0; seed_stats = {s: [] for s in SEEDS}
    for ordinal, frame in enumerate(inventory):
        source_path = prior_run/'prediction_frames/real'/f'{ordinal:04d}'/'FRAME_PREDICTIONS.json.gz'
        saved = read(source_path); details = saved['current_observation_and_solver_records']
        for seed in SEEDS:
            a,b = [details[f'OBS_{arm}_s{seed}']['selection'] for arm in ('COMMON','BI')]
            seed_stats[seed].append(dict(id=frame['id'], partition=frame['partition'],
                compatibility=comparable(a,b), differences=differences(a,b)))
        common, bi = [details[f'OBS_{arm}_s{REPRESENTATIVE}'] for arm in ('COMMON','BI')]
        eligibility = comparable(common['selection'], bi['selection'])
        cache = {}; cells = {}; cell_inputs = {'Q0S0':common, 'Q1S1':bi}
        for name, item in cell_inputs.items():
            signature = P.exact_solve_key(frame,item['selection'])
            if signature != item['exact_observation_signature']: raise ValueError('Immutable diagonal dependency closure differs')
            payload = {k:item['selection'][k] for k in ('states','full_points','partial_lines')}
            cache[signature] = (payload, item['solver_result'], name)
            cells[name] = dict(available=True, actual_pose=item['solver_result']['actual_pose'],
                accepted=item['solver_result']['accepted'], fallback=item['solver_result']['fallback'],
                status=item['solver_result']['status'], family_readouts=family_readouts(item['solver_result']),
                selected_family=(item['solver_result'].get('solution') or {}).get('selected_family'),
                exact_solve_key=signature, original_diagonal_exact_reuse=True, actual_new_family_optimizer_calls=0)
        for name, q, cov in [('Q1S0',bi,common), ('Q0S1',common,bi)]:
            if not eligibility['eligible']:
                cells[name] = dict(available=False, excluded_cross_intervention=True,
                    exclusion_reasons=eligibility['reasons'], algorithm_failure=False,
                    actual_new_family_optimizer_calls=0)
                continue
            selection = mix(q['selection'],cov['selection']); signature = P.exact_solve_key(frame,selection)
            payload = {k:selection[k] for k in ('states','full_points','partial_lines')}
            if signature in cache:
                old, result, origin = cache[signature]
                if old != payload: raise AssertionError('Hash collision')
                calls=0; reused=True; packet_reuse+=1
            else:
                result=S.solve_frame(frame,selection); calls=result['optimizer_calls']; reused=False;origin=name
                cache[signature]=(payload,result,name);new_calls+=calls
            cells[name]=dict(available=True,actual_pose=result['actual_pose'],accepted=result['accepted'],
                fallback=result['fallback'],status=result['status'],family_readouts=family_readouts(result),
                selected_family=(result.get('solution') or {}).get('selected_family'), exact_solve_key=signature,
                exact_packet_cache_reused=reused, cache_origin=origin, actual_new_family_optimizer_calls=calls,
                counterfactual_provenance=dict(q_donor='BI' if name=='Q1S0' else 'COMMON',
                    covariance_donor='COMMON' if name=='Q1S0' else 'BI', typed_partial_and_source_association_fixed=True,
                    original_typed_archives_modified=False))
            if not reused: write(output/'new_counterfactual_packets'/f'{ordinal:04d}_{name}.json.gz',dict(selection=selection,solver_result=result))
        record=dict(id=frame['id'],partition=frame['partition'],session=saved['ledger']['session'],
            cluster_id=saved['ledger']['cluster_id'],ordinal=ordinal,cross_eligibility=eligibility,cells=cells,
            stored_source_path=str(source_path),stored_source_sha256=sha(source_path),
            factor_decomposition_scope='within fixed family versus automatic minimum-cost family readout; one observation seed')
        write(output/'frames'/f'{ordinal:04d}.json.gz',record);rows.append(record)
    summary=dict(schema='BI_FULL_QCOV_DIAGNOSTIC_V1',status='DIAGNOSTIC_ONLY',full_real_denominator=238,
        representative_observation_seed=REPRESENTATIVE,full_diagonals_exact_reused=476,
        cross_eligible_frames=sum(r['cross_eligibility']['eligible'] for r in rows),
        cross_excluded_frames=sum(not r['cross_eligibility']['eligible'] for r in rows),
        cross_exclusion_ids=[dict(id=r['id'],partition=r['partition'],reasons=r['cross_eligibility']['reasons']) for r in rows if not r['cross_eligibility']['eligible']],
        reason_counts=dict(Counter(reason for r in rows for reason in r['cross_eligibility']['reasons'])),
        actual_new_family_optimizer_calls=new_calls,actual_new_family_optimizer_call_upper_bound=1904,
        cross_packet_exact_reuses=packet_reuse,
        seed_observation_summaries={str(s):dict(frames=len(ss),q_changed_frames=sum(bool(r['differences']['q_changed_corner_ids']) for r in ss),
            effective_covariance_changed_frames=sum(bool(r['differences']['effective_covariance_changed_corner_ids']) for r in ss),
            cross_eligible_frames=sum(r['compatibility']['eligible']for r in ss),
            source_or_factor_divergence_frames=sum(not r['compatibility']['eligible']for r in ss))for s,ss in seed_stats.items()},
        seed_observation_differences=seed_stats,
        reference_numeric_inputs=False,wall_seconds=time.monotonic()-start,source_sha256=sha(__file__),
        limitations=['Source-changing strata do not identify separate q and covariance causal effects.',
            'Excluded cross cells remain in the full238 inventory and are not zero-score algorithm failures.',
            'One observation seed diagnostic is not a full observation-model initialization uncertainty study.',
            'Limited two-family solver is not original D1 full candidate producer.'])
    if new_calls>1904:raise RuntimeError('Preregistered solve budget exceeded')
    write(output/'SUMMARY.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prior-run',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--fixed-common-graph',action='store_true');parser.add_argument('--plan');parser.add_argument('--plan-sha256');parser.add_argument('--root-go',action='store_true')
    args=parser.parse_args()
    out=run_fixed_common(args.prior_run,args.output,args.plan,args.plan_sha256,root_GO=args.root_go)if args.fixed_common_graph else run(args.prior_run,args.output)
    print(json.dumps({k:v for k,v in out.items()if isinstance(v,(str,int,float,bool))},sort_keys=True))


if __name__=='__main__':main()
