"""Three fixed CAL cached-chain V31 admission diagnoses, opt-in only.

Declaration reads metadata and hashes only, before any observation row/mask is
decoded. Execution reuses immutable raw V29 chains, original guides and mask;
no RGB, labels, reference projection, pose or metric is read. Current V30 is
unchanged. The new selected population has no validated noise model yet.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import time
import numpy as np

from . import observed_anchor_native_ray_association_v4_31 as F
from . import run_observed_bi_line_association_CAL_v4_30b as OLD

C, R, OBS, B = OLD.C, OLD.R, OLD.OBS, OLD.B
OUTPUT = B / 'observed_anchor_native_ray_CAL_cases_v4_31'
PLAN = B / 'observed_anchor_native_ray_CAL_plan_v4_31/PREDECLARATION.json'
CASES = ('wood_184309:000925', 'wood_184309:000804', 'eval_cad:1778653017736058368')
ADMISSION_SOURCE_SHA256 = 'ccdb311a4468825919a6731620e6836a0b52bffc775cc594d90c6bea8d2babad'
ORIGINAL_RUNNER_SOURCE_SHA256 = '65b92a6fc9198f7ff5ded709ecc4b3d26907fd5888940b4602fe4e9d64ab8e5c'


def _sources():
    sources = OLD._sources()
    paths = (Path(__file__), Path(__file__).with_name('test_run_observed_anchor_native_ray_CAL_v4_31.py'),
        Path(F.__file__), Path(F.__file__).with_name('test_observed_anchor_native_ray_association_v4_31.py'),
        Path(OLD.__file__))
    sources.update({str(p.resolve().relative_to(C.ROOT)): C.sha(p) for p in paths})
    return sources


def _fixed_cases(entries):
    if tuple(e['id'] for e in entries) != CASES or any(e['partition'] != 'CAL' for e in entries):
        raise ValueError('Keep the same ordered three CAL cases; no reselection or primary rows')
    return CASES


def _source_pins():
    if (C.sha(F.__file__) != ADMISSION_SOURCE_SHA256
            or C.sha(OLD.__file__) != ORIGINAL_RUNNER_SOURCE_SHA256):
        raise ValueError('Exact frozen admission and original runner required')


def declare():
    """Bind fixed case/index/source bytes; never decode a CAL row or mask."""
    if PLAN.exists():
        raise FileExistsError('Preserve every V31 declaration')
    _source_pins()
    previous, previousseal = C.read(OLD.PLAN), C.read(OLD.PLAN.parent / 'PLAN_SEAL.json')
    if C.sha(OLD.PLAN) != previousseal['plan_sha256']:
        raise ValueError('Original fixed three-case declaration changed')
    _fixed_cases(previous['cases'])
    protocol = C.read(OBS.OUTPUT / 'PROTOCOL.json')
    if previous['configuration'] != protocol['configuration']:
        raise ValueError('Same original geometry/line configuration required')
    index = {e['id']: e for e in C.read(OBS.OUTPUT / 'CAL_ROW_INDEX.json')['entries']}
    baseline_seal = C.read(OLD.OUTPUT / 'RERANK_SEAL.json')
    cases, paths = [], [OLD.PLAN, OLD.PLAN.parent / 'PLAN_SEAL.json',
        OBS.OUTPUT / 'PROTOCOL.json', OBS.OUTPUT / 'PREPARATION_SEAL.json',
        OBS.OUTPUT / 'CAL_ROW_INDEX.json', OBS.OUTPUT / 'CAL_OBSERVATION_SEAL.json',
        OLD.OUTPUT / 'RERANK_SEAL.json']
    for number, ident in enumerate(CASES):
        entry = index[ident]
        source = OBS._row_path(OBS.OUTPUT, ident)
        baseline = OLD.OUTPUT / 'graphs' / ('case' + str(number) + '.jsonl.gz')
        if (entry['partition'] != 'CAL' or C.sha(source) != entry['sha256']
                or entry['sha256'] != previous['cases'][number]['sha256']
                or C.sha(baseline) != baseline_seal['assets'][str(baseline.relative_to(OLD.OUTPUT))]):
            raise ValueError('Same frozen raw CAL spool and prior cached V30 graph required')
        cases.append({**entry, 'spool_relative': str(source.relative_to(C.ROOT)),
            'previous_V30_graph': R._binding(baseline)})
        paths.extend((source, baseline))
    _fixed_cases(cases)
    sources = _sources()
    inputs = {str(p.relative_to(C.ROOT)): C.sha(p) for p in paths}
    for mapping in (sources, inputs, protocol['protected_user_sha256']):
        R._verify(mapping)
    PLAN.parent.mkdir(parents=True)
    C.write(PLAN, dict(schema='FIXED_THREE_CAL_OBSERVED_ANCHOR_NATIVE_RAY_CACHED_PLAN_V31',
        UTC=datetime.now(timezone.utc).isoformat(), cases=cases,
        configuration=protocol['configuration'], selection_profile=F.SELECTION_PROFILE, recipe=F.RECIPE,
        source_sha256=sources, input_sha256=inputs,
        protected_user_sha256=protocol['protected_user_sha256'],
        fixed_case_origin='same earlier declared CAL top3 diagnosis; no new error/metric/ID selection',
        admission='original immutable BI q actual query anchor + fixed signed native ray + unchanged V30 normal reach',
        unknown_mate='only original model projection direction; S/U/O/T observed points are never queried',
        guides_raw_indices_masks_q_covariance_and_states_unchanged=True,
        original_V29_and_previous_V30_graphs_are_diagnostic_baselines_only=True,
        parent_globals_inference_models_and_outputs_changed=False,
        all_three_rows_and_zero_candidates_or_selected_abstentions_retained=True,
        original_full_raw_archive_retained_by_exact_spool_and_canonical_bindings=True,
        no_actual_row_mask_RGB_reference_or_score_decode_in_declaration=True,
        new_RGB_NMS_candidate_TLS_NN_labels_posefit_or_KP_adoption=0,
        cached_support_rounded_diagnostic_TLS_refits_allowed=True,
        no_reference_or_error_terms_in_ranking_or_report=True,
        validated_noise_model_available=False, old_noise_transfer_or_relabel=False,
        new_CAL46_noise_and_uniform_evaluation_required_before_method_adoption=True,
        fixed_three_CAL_selection_diagnosis_is_not_full_performance_evidence=True,
        actual_cached_rerank_requires_separate_root_GO=True))
    for relative, digest in sources.items():
        target = PLAN.parent / 'code_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((C.ROOT / relative).read_bytes())
        if C.sha(target) != digest:
            raise ValueError('Exact frozen source snapshot required')
    C.write(PLAN.parent / 'PLAN_SEAL.json', dict(plan_sha256=C.sha(PLAN), source_sha256=sources,
        input_sha256=inputs, actual_rows_masks_RGB_labels_or_scores_decoded=0))
    print(dict(plan_sha256=C.sha(PLAN), cases=CASES, actual_cached_rows_decoded=0), flush=True)


def _single_graph(path):
    records = iter(C.rows(path))
    result = next(records, None)
    if result is None or next(records, None) is not None:
        raise ValueError('One exact previous cached graph required')
    if result['selection_profile'] != F.F.SELECTION_PROFILE:
        raise ValueError('Previous V30 graph is not the declared baseline')
    return result['graph']


def _immutable_observation_view(row):
    observation = row['observation']
    return dict(chain_observation=observation['chain_observation'], guide_bank=observation['guide_bank'],
        guide_geometries=observation['guide_geometries'], initial_geometry=row['initial_geometry'],
        visible_measurements=row['visible_measurements'], states=row['states'], mask_binding=row['mask_binding'])


def _selected_audits(result):
    names = set(result['graph']['selected_line_names'])
    return [entry for entry in result['node_association_ledger'] if entry['line_name'] in names]


def run(*, root_actual_rerank_GO=False):
    if root_actual_rerank_GO is not True:
        raise PermissionError('Separate root cached three-CAL GO before any IO')
    if OUTPUT.exists():
        raise FileExistsError('Preserve every cached rerank attempt')
    _source_pins()
    plan, pseal = C.read(PLAN), C.read(PLAN.parent / 'PLAN_SEAL.json')
    if C.sha(PLAN) != pseal['plan_sha256'] or plan['selection_profile'] != F.SELECTION_PROFILE:
        raise ValueError('Exact frozen new-population plan required')
    _fixed_cases(plan['cases'])
    for mapping in (plan['source_sha256'], plan['input_sha256'], plan['protected_user_sha256']):
        R._verify(mapping)
    OUTPUT.mkdir(parents=True)
    (OUTPUT / 'graphs').mkdir()
    records = []
    for number, case in enumerate(plan['cases']):
        path = C.ROOT / case['spool_relative']
        row = OBS._read_single(path, case)
        if row['id'] != case['id'] or row['partition'] != 'CAL':
            raise ValueError('Only the same exact CAL source row is permitted')
        observation, geometry = row['observation'], row['initial_geometry']
        if observation['observation_generation'] != F.P.OBSERVATION_GENERATION:
            raise ValueError('Unchanged raw V29 generation required')
        before = R._canonical_sha(_immutable_observation_view(row))
        guides = R.restore_pose_guides(row)
        binding = dict(sha256=row['image_sha256'], image_hw=geometry['image_hw'])
        regenerated = F.G.prepare_joint_geometry_guides(geometry, row['states'], row['visible_measurements'],
            image_binding=binding, configuration=plan['configuration'], pose_guides=guides)
        if R._canonical_sha(regenerated) != R._canonical_sha(observation['guide_bank']):
            raise ValueError('Exact original guide reconstruction required')
        mask_path = C.ROOT / row['mask_binding']['path']
        if C.sha(mask_path) != row['mask_binding']['sha256']:
            raise ValueError('Original cached target-mask bytes changed')
        with np.load(mask_path, allow_pickle=False) as archive:
            mask = archive['pallet_mask_clean'].copy()
        if F.P.VIS.mask_pixel_sha256(mask) != observation['target_mask_pixel_sha256']:
            raise ValueError('Exact original mask pixels required')
        context = F.P.VIS.prepare_edge_visibility(mask, geometry, row['states'], row['visible_measurements'],
            image_binding=binding, occluder_mask=None, occluder_binding=None)
        started = time.monotonic()
        result = F.rank_observed_anchor_native_ray_graph(observation['chain_observation'], geometry,
            row['states'], row['visible_measurements'], image_binding=binding,
            configuration=plan['configuration'], pose_guides=guides, pallet_mask=mask, visibility_context=context)
        seconds = time.monotonic() - started
        if R._canonical_sha(_immutable_observation_view(row)) != before:
            raise ValueError('Raw chains, q, states, geometry, guide or mask bindings mutated')
        graph = result['graph']
        cfg, _ = F.G._config(plan['configuration'])
        R.decorate_graph(graph, observation['chain_observation'], cfg)
        C.save_rows(OUTPUT / 'graphs' / ('case' + str(number) + '.jsonl.gz'), [result])
        prior = _single_graph(C.ROOT / case['previous_V30_graph']['path'])
        raw = observation['joint_graph']
        prior_names = set(prior['selected_line_names'])
        record = dict(id=row['id'], rank_seconds=seconds,
            original_V29_candidates=raw['actual_unique_line_candidates'],
            previous_V30_candidates=prior['actual_unique_line_candidates'],
            new_V31_candidates=graph['actual_unique_line_candidates'],
            original_V29_selected=raw['selected_line_names'], previous_V30_selected=prior['selected_line_names'],
            new_V31_selected=graph['selected_line_names'], checked=result['candidate_nodes_checked'],
            rejected=result['rejected_node_count'], rejection_reasons=dict(Counter(
                e['reason'] for e in result['node_association_ledger'] if not e['passes'])),
            previous_selected_rejections=[e for e in result['node_association_ledger']
                if e['line_name'] in prior_names and not e['passes']],
            all_new_selected_admission_audits=_selected_audits(result),
            zero_candidate_abstention=graph['actual_unique_line_candidates'] == 0,
            zero_selected_edges=len(graph['selected_line_names']) == 0,
            joint_states_evaluated=graph['joint_states_evaluated'],
            same_original_spool=R._binding(path), same_cached_target_mask=R._binding(mask_path),
            original_raw_guide_q_states_binding_sha256=before,
            original_raw_arrays_indices_and_guide_geometry_unchanged=True,
            original_chain_refs_preserved=True, independent_occluder_mask_available=False,
            target_mask_absence_is_unknown=True, selection_profile=F.SELECTION_PROFILE,
            validated_noise_model_available=False, old_noise_transfer=False,
            no_reference_projection_or_metric_comparison=True,
            new_RGB_NMS_candidate_TLS_NN_labels_posefit_or_KP_adoption=0,
            cached_support_rounded_diagnostic_TLS_refits_allowed=True)
        C.write(OUTPUT / ('case' + str(number) + '_SUMMARY.json'), record)
        records.append(record)
        print(dict(id=row['id'], candidates=record['new_V31_candidates'], rejected=record['rejected'],
            selected=record['new_V31_selected'], seconds=seconds), flush=True)
        del row, observation, graph, result, prior, raw, mask
    if tuple(r['id'] for r in records) != CASES:
        raise ValueError('All three fixed rows including empty selections required')
    C.write(OUTPUT / 'RESULTS.json', dict(selection_profile=F.SELECTION_PROFILE, cases=records,
        all_three_rows_kept=True, validated_noise_model_available=False, primary_performance_claim=False))
    text = ('# v31 관측 q 앵커·고정 native 방향: 고정 CAL3 저장 선 진단\n\n'
        '원래 3개 CAL 사례를 그대로 유지하고 저장된 실제 선만 재순위한다. 원 q에 실제로 고정돼 탐색된 후보인지, '
        '고정 native 진행 방향과 부호가 맞는지, 기존 43px 법선 범위를 만족하는지 검사한다. 모델 앵커 탐색을 '
        'q 앵커 측정으로 바꾸거나 원 픽셀·TLS 계수를 이동하지 않는다.\n\n'
        '| 고정 CAL 사례 | 원 v29 / v30 / 새 v31 후보 수 | v31 제외 | 원 v30 / 새 v31 선택 수 | 시간(s) |\n'
        '|---|---:|---:|---:|---:|\n')
    for item in records:
        text += (f'| {item["id"]} | {item["original_V29_candidates"]} / {item["previous_V30_candidates"]} / '
            f'{item["new_V31_candidates"]} | {item["rejected"]} | {len(item["previous_V30_selected"])} / '
            f'{len(item["new_V31_selected"])} | {item["rank_seconds"]:.3f} |\n')
    text += ('\n선택 0도 abstain 결과로 남긴다. 모든 원 raw chain·인덱스·guide·마스크는 바인딩된 원 archive에 그대로 '
        '보존하며 새 선택 graph와 제외 사유만 별도로 쓴다. 별도 다른 물체 가림 마스크는 없어 target mask 부재를 '
        'external O로 단정하지 않는다. S/U 국소 점은 질의하지 않는다.\n\n'
        '새 영상·NMS·후보 TLS·NN·라벨·참조·PnP·정밀 KP 채택은 없다. 고정 worker의 cached-support rounded '
        '진단 TLS 재적합은 허용되며 새 픽셀 측정과 구별한다. 새 population에 검증된 noise는 아직 없고 기존 모델을 '
        '전이하지 않는다. 별도 CAL46 잡음 적합과 균일 전체 평가 전에는 방법 채택·정확도·전체 성능 개선 증거가 아니다.\n')
    (OUTPUT / 'REPORT_KO.md').write_text(text, encoding='utf-8')
    for mapping in (plan['source_sha256'], plan['input_sha256'], plan['protected_user_sha256']):
        R._verify(mapping)
    C.write(OUTPUT / 'RERANK_SEAL.json', dict(plan_sha256=C.sha(PLAN), selection_profile=F.SELECTION_PROFILE,
        source_sha256=plan['source_sha256'], input_sha256=plan['input_sha256'],
        protected_user_sha256=plan['protected_user_sha256'], rows=3, all_three_rows_and_empty_selections_retained=True,
        assets={str(p.relative_to(OUTPUT)): C.sha(p) for p in OUTPUT.rglob('*') if p.is_file()},
        original_V29_V30_sources_models_plans_or_observations_changed=False,
        new_RGB_NMS_candidate_TLS_NN_labels_posefit_or_KP_adoption=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('declare', 'run'))
    parser.add_argument('--root-actual-rerank-GO', action='store_true')
    arguments = parser.parse_args()
    declare() if arguments.stage == 'declare' else run(root_actual_rerank_GO=arguments.root_actual_rerank_GO)
