"""Fixed CAL3 initial-only raw-query rerank, no inherited pose guide scoring.

This is an observation-selection diagnosis, not a pose accuracy experiment.
The same original three CAL raw spools and target masks are retained. No
cached guide bank is reconstructed or supplied to the numerical ranker.
Declaration hashes bytes only; execution needs explicit opt-in before IO.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np

from . import initial_only_observed_ray_association_v4_32 as F
from . import run_observed_anchor_native_ray_CAL_v4_31 as OLD

C, R, OBS, B = OLD.C, OLD.R, OLD.OBS, OLD.B
CASES = OLD.CASES
PLAN = B/'initial_only_observed_ray_CAL_plan_v4_32/PREDECLARATION.json'
OUTPUT = B/'initial_only_observed_ray_CAL_cases_v4_32'
ANCESTRY = B/'initial_only_raw_query_source_audit_v4_32/AUDIT_V1.json'
SOURCE_CLOSURE = B/'frozen_e_zero_edge_native_source_audit_v4_31/COARSE_Q_GUIDE_SOURCE_CLOSURE_V1.json'
PEER = B/'initial_only_observed_ray_association_peer_review_v4_32/SOURCE_REVIEW_V1.json'
ADMISSION_SHA256 = 'd4ab01e8ffb0d3ee00e18730fdd7976ca9debf84e7c432c216df6cc648931674'
PEER_SHA256 = 'fc5ed589894fa89312b1924e9c86b4e38b45c0e672e6543bc6716798f60383ce'
ANCESTRY_SHA256 = 'e0db2eab479fe2527aa394465f7650c67999de385f4be8cae5c2b26e6b908620'


def _pins():
    if C.sha(F.__file__) != ADMISSION_SHA256 or C.sha(PEER) != PEER_SHA256:
        raise ValueError('Exact independently reviewed V32 admission required')
    if C.sha(SOURCE_CLOSURE) != F.INITIAL_SOURCE_CLOSURE_SHA256:
        raise ValueError('Traced original coarse/q ancestry required')


def _cases(entries):
    if tuple(e['id'] for e in entries) != CASES or any(e['partition'] != 'CAL' for e in entries):
        raise ValueError('Same ordered three CAL cases required; no primary or reselection')


def _sources():
    result = OLD._sources()
    paths = (Path(__file__), Path(__file__).with_name('test_run_initial_only_observed_ray_CAL_v4_32.py'),
             Path(F.__file__), Path(F.__file__).with_name('test_initial_only_observed_ray_association_v4_32.py'))
    result.update({str(p.resolve().relative_to(C.ROOT)): C.sha(p) for p in paths})
    return result


def declare(*, root_metadata_GO=False):
    if root_metadata_GO is not True:
        raise PermissionError('Metadata GO before IO')
    if PLAN.exists():
        raise FileExistsError('Preserve all fixed-CAL3 declarations')
    _pins()
    previous = C.read(OLD.PLAN)
    if C.sha(OLD.PLAN) != C.read(OLD.PLAN.parent/'PLAN_SEAL.json')['plan_sha256']:
        raise ValueError('Original fixed CAL3 declaration changed')
    _cases(previous['cases'])
    protocol = C.read(OBS.OUTPUT/'PROTOCOL.json')
    index = {e['id']: e for e in C.read(OBS.OUTPUT/'CAL_ROW_INDEX.json')['entries']}
    if previous['configuration'] != protocol['configuration']:
        raise ValueError('Original query/graph configuration required')
    # The independent source audit is required before any cached numerical row.
    if C.sha(ANCESTRY) != ANCESTRY_SHA256:
        raise ValueError('Exact independently traced raw-query ancestry required')
    ancestry = C.read(ANCESTRY)
    if (ancestry.get('verdict') != 'NO_CROSS_QUERY_NMS_OR_PREGRAPH_CACHED_GUIDE_QUOTA_FOUND'
            or ancestry.get('actual_rows_or_images_or_arrays_read') != 0):
        raise ValueError('Source-only independence of original/measured raw query budgets required')
    R._verify(ancestry['source_sha256'])
    paths = [OLD.PLAN, OLD.PLAN.parent/'PLAN_SEAL.json', SOURCE_CLOSURE, PEER, ANCESTRY,
             OBS.OUTPUT/'PROTOCOL.json', OBS.OUTPUT/'CAL_ROW_INDEX.json', OBS.OUTPUT/'CAL_OBSERVATION_SEAL.json']
    cases = []
    for number, ident in enumerate(CASES):
        entry = index[ident]
        path = OBS._row_path(OBS.OUTPUT, ident)
        if (entry['partition'] != 'CAL' or C.sha(path) != entry['sha256']
                or entry['sha256'] != previous['cases'][number]['sha256']):
            raise ValueError('Same immutable original raw CAL spool required')
        cases.append({**entry, 'spool_relative': str(path.relative_to(C.ROOT))})
        paths.append(path)
    sources = _sources()
    inputs = {str(p.relative_to(C.ROOT)): C.sha(p) for p in paths}
    for mapping in (sources, inputs, protocol['protected_user_sha256']):
        R._verify(mapping)
    value = dict(schema='FIXED_CAL3_INITIAL_ONLY_ACTUAL_QUERY_SELECTION_V32_PLAN',
        UTC=datetime.now(timezone.utc).isoformat(), cases=cases, selection_profile=F.SELECTION_PROFILE,
        configuration=protocol['configuration'], recipe=F.RECIPE,
        source_sha256=sources, input_sha256=inputs, protected_user_sha256=protocol['protected_user_sha256'],
        original_fixed_CAL3_selection_preserved=True, raw_chain_indices_pixels_TLS_q_covariance_immutable=True,
        raw_archive_contains_extra_queries=True, extra_query_numeric_payload_or_guide_scoring_consumed=False,
        external_pose_guides=None, reconstruct_old_guide_bank=False,
        rank_graph_internal_geometry='ORIGINAL_INITIAL_ONLY',
        raw_query_ancestry_bound_to_source_only_audit=True,
        old_graphs_or_metrics_used_as_numeric_baselines=False,
        reference_labels_RGB_NN_posefit_decodes=0,
        actual_cached_rows_or_masks_decoded_in_declare=0,
        target_mask_used_as_existing_boundary_context_not_occluder=True,
        independent_occluder_mask=None, separate_automatic_instance_experiment_not_consumed=True,
        S_image_or_local_2D_queries=0, U_only_approximate_regions=True,
        all_three_rows_and_zero_selection_retained=True,
        population_has_new_noise=False, old_noise_transfer=False,
        CAL46_fitting_and_uniform_primary_evaluation_required=True,
        adopted_method=False, pose_accuracy_claim=False)
    PLAN.parent.mkdir(parents=True)
    C.write(PLAN, value)
    for relative, digest in sources.items():
        target = PLAN.parent/'code_snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((C.ROOT/relative).read_bytes())
        if C.sha(target) != digest:
            raise ValueError('Source snapshot mismatch')
    C.write(PLAN.parent/'PLAN_SEAL.json', dict(plan_sha256=C.sha(PLAN), source_sha256=sources))
    return dict(plan_sha256=C.sha(PLAN), actual_rows_masks_or_pose_reads=0)


def _rank_row(row, mask, configuration):
    """Only raw observation, independent geometry, gated BI and target mask."""
    if row['partition'] != 'CAL' or row['id'] not in CASES:
        raise ValueError('Declared CAL row required before coordinate payload')
    observation, geometry = row['observation'], row['initial_geometry']
    if observation['observation_generation'] != F.P.OBSERVATION_GENERATION:
        raise ValueError('Exact original raw generation required')
    before = R._canonical_sha(OLD._immutable_observation_view(row))
    if F.P.VIS.mask_pixel_sha256(mask) != observation['target_mask_pixel_sha256']:
        raise ValueError('Frozen target mask pixel binding required')
    binding = dict(sha256=row['image_sha256'], image_hw=geometry['image_hw'])
    context = F.P.VIS.prepare_edge_visibility(mask, geometry, row['states'], row['visible_measurements'],
        image_binding=binding, occluder_mask=None, occluder_binding=None)
    started = time.monotonic()
    result = F.rank_initial_only_observed_ray_graph(observation['chain_observation'], geometry,
        row['states'], row['visible_measurements'], image_binding=binding,
        configuration=configuration, pose_guides=None, pallet_mask=mask, visibility_context=context)
    seconds = time.monotonic()-started
    if R._canonical_sha(OLD._immutable_observation_view(row)) != before:
        raise ValueError('Original raw arrays, guides, states, geometry or q mutated')
    graph = result['graph']
    cfg, _ = F.G._config(configuration)
    R.decorate_graph(graph, observation['chain_observation'], cfg)
    selected = set(graph['selected_line_names'])
    audits = [r for r in result['node_association_ledger'] if r['line_name'] in selected]
    if (any(not r['passes'] for r in audits)
            or len(audits) != len(selected) or result['external_pose_guides_consumed'] != 0):
        raise ValueError('Every selected line must have one fresh strict-source admission')
    summary = dict(id=row['id'], partition='CAL', rank_seconds=seconds,
        candidate_nodes_checked=result['candidate_nodes_checked'], rejected=result['rejected_node_count'],
        rejection_reasons=dict(Counter(r['reason'] for r in result['node_association_ledger'] if not r['passes'])),
        selected_line_names=graph['selected_line_names'], selected_count=len(selected),
        unique_admitted_line_candidates=graph['actual_unique_line_candidates'],
        selected_admission_audits=audits, zero_selected_edges=not selected,
        immutable_raw_view_sha256=before, raw_indices_arrays_q_covariances_unchanged=True,
        source_profile=F.SELECTION_PROFILE, external_pose_guides_consumed=0,
        graph_geometry_source=result['graph_geometry_source'],
        other_object_mask_confirmed_or_consumed=False, target_mask_absence_is_occlusion=False,
        new_RGB_NMS_candidate_TLS_NN_labels_posefit_or_KPs=0,
        cached_support_rounded_diagnostic_TLS_refits_allowed=True,
        performance_or_noise_validity_claim=False)
    return result, summary


def run(*, expected_plan_sha256, root_actual_rerank_GO=False):
    if root_actual_rerank_GO is not True:
        raise PermissionError('Separate actual CAL3 GO before IO')
    if not isinstance(expected_plan_sha256, str) or len(expected_plan_sha256) != 64:
        raise ValueError('Explicit exact plan SHA required')
    _pins()
    if OUTPUT.exists():
        raise FileExistsError('Preserve every successful/failed attempt')
    if C.sha(PLAN) != expected_plan_sha256 or C.read(PLAN.parent/'PLAN_SEAL.json')['plan_sha256'] != expected_plan_sha256:
        raise ValueError('Exact declared V32 source/row plan required')
    plan = C.read(PLAN)
    _cases(plan['cases'])
    for mapping in (plan['source_sha256'], plan['input_sha256'], plan['protected_user_sha256']):
        R._verify(mapping)
    OUTPUT.mkdir(parents=True)
    (OUTPUT/'graphs').mkdir()
    C.write(OUTPUT/'EXECUTION_START.json', dict(plan_sha256=expected_plan_sha256, UTC=datetime.now(timezone.utc).isoformat()))
    try:
        summaries = []
        for number, case in enumerate(plan['cases']):
            row = OBS._read_single(C.ROOT/case['spool_relative'], case)
            path = C.ROOT/row['mask_binding']['path']
            if C.sha(path) != row['mask_binding']['sha256']:
                raise ValueError('Frozen target mask file changed')
            with np.load(path, allow_pickle=False) as archive:
                mask = archive['pallet_mask_clean'].copy()
            result, summary = _rank_row(row, mask, plan['configuration'])
            C.save_rows(OUTPUT/'graphs'/f'case{number}.jsonl.gz', [result])
            C.write(OUTPUT/f'case{number}_SUMMARY.json', summary)
            summaries.append(summary)
            print(dict(id=summary['id'], selected=summary['selected_count'], rejected=summary['rejected'], seconds=summary['rank_seconds']), flush=True)
            del row, mask, result
        _cases(summaries)
        C.write(OUTPUT/'RESULTS.json', dict(cases=summaries, all_three_rows_kept=True,
            selection_profile=F.SELECTION_PROFILE, pose_or_primary_accuracy_claim=False, new_noise_available=False))
        text = '# V32 원 초기/관측 B/I 질의만 사용: 고정 CAL3 저장 선 진단\n\n'
        text += '원래 세 CAL 사례와 실제 raw chain을 유지한다. 과거 자세 후보에서 만든 질의를 제외한 뒤 그래프를 다시 계산하며, 외부 pose guide는 점수에도 사용하지 않는다. 새 PnP·포즈 점수·정확도·CAL 잡음 적합 결과가 아니다.\n\n'
        text += '| 고정 CAL | 허용 선 후보 | 제외 raw 후보 | 최종 선택 | 시간(s) |\n|---|---:|---:|---:|---:|\n'
        for row in summaries:
            text += f"| {row['id']} | {row['unique_admitted_line_candidates']} | {row['rejected']} | {row['selected_count']} | {row['rank_seconds']:.3f} |\n"
        text += '\n원 q·covariance·상태·geometry·pixel/TLS·raw index는 그대로 보존했다. 선택0도 남겼다. target mask 바깥을 다른 물체 가림으로 확정하지 않았고 별도 자동 instance 후보도 공급하지 않았다. 새 모집단 CAL46 적합과 전체192장 균일 평가가 필요하다.\n'
        (OUTPUT/'REPORT_KO.md').write_text(text, encoding='utf-8')
        for mapping in (plan['source_sha256'], plan['input_sha256'], plan['protected_user_sha256']):
            R._verify(mapping)
        C.write(OUTPUT/'RERANK_SEAL.json', dict(plan_sha256=expected_plan_sha256,
            selection_profile=F.SELECTION_PROFILE, rows=3, ordered_ids=list(CASES),
            source_sha256=plan['source_sha256'], input_sha256=plan['input_sha256'],
            protected_user_sha256=plan['protected_user_sha256'],
            assets={str(p.relative_to(OUTPUT)): C.sha(p) for p in OUTPUT.rglob('*') if p.is_file()},
            all_three_rows_and_zero_selections_retained=True, external_pose_guides_consumed=0,
            labels_RGB_NN_posefit_or_primary_metrics=0, old_sources_models_or_outputs_changed=False))
    except BaseException as error:
        C.write(OUTPUT/'FAILURE_RECEIPT.json', dict(plan_sha256=expected_plan_sha256,
            error_type=type(error).__name__, error=str(error), attempt_preserved=True))
        raise
    return dict(output=str(OUTPUT.relative_to(C.ROOT)), seal_sha256=C.sha(OUTPUT/'RERANK_SEAL.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('declare', 'run'))
    parser.add_argument('--root-metadata-GO', action='store_true')
    parser.add_argument('--root-actual-rerank-GO', action='store_true')
    parser.add_argument('--expected-plan-sha256')
    args = parser.parse_args()
    result = declare(root_metadata_GO=args.root_metadata_GO) if args.stage == 'declare' else run(
        expected_plan_sha256=args.expected_plan_sha256, root_actual_rerank_GO=args.root_actual_rerank_GO)
    print(result, flush=True)
