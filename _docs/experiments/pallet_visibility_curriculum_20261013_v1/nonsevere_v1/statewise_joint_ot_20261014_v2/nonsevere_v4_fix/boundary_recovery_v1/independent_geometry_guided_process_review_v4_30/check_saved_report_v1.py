"""Small-artifact QA only; never opens INPUTS/PREDICTIONS/OBSERVATIONS journals."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[8]
BASE = Path(__file__).resolve().parent.parent
OUTPUT = Path(__file__).resolve().parent
REPORT = BASE / 'geometry_guided_process_v4_30_v1'
PRED = BASE / 'geometry_guided_edge_eval_v4_30'
PLAN = BASE / 'geometry_guided_process_plan_v4_30/PREDECLARATION.json'
SOURCE = ROOT / 'scripts/research/pallet_visibility_curriculum_20261013_v1/render_geometry_guided_process_v4_30.py'
CASES = ['wood_night_01:031426', 'eval_pallet09:1778653661653195264']
ARMS = ['SAME_FROZEN_E', 'ASSOCIATED_EDGE_ALL_BI_NATIVE', 'ASSOCIATED_EDGE_ALL_BI_CONTAINMENT',
        'ASSOCIATED_EDGE_SOURCE_NATIVE', 'ASSOCIATED_EDGE_SOURCE_CONTAINMENT']
IMAGES = [
 ('data/evaluation/pallet_eval_v1/final/positive/sessions/wood_night_01/rgb/031426.png',
  '782ca888145e1996031d525477def722df85216db6f534910e46acc43b88defa'),
 ('data/evaluation/pallet_eval_v1/dev_existing/sessions/eval_pallet09/rgb/1778653661653195264.png',
  '9e143682538abfe5a95cb5f77319319e395b7fd6f9e916b55ba4fbe3d6fd8b29')]
MASKS = [
 ('_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/remainder_after_test49_v1/fresh_observations_v4_14/masks/wood_night_01__031426.npz',
  '542a1f4ed28eda60ac48b8b4891f08183a353e56ae278fcad6fe8e427092de79'),
 (str(BASE.relative_to(ROOT) / 'image_evidence_v4_6/eval_pallet09__1778653661653195264.npz'),
  'ea40cfb5538d877b9af7971ad66fd4823de6bce47c178efe726fceb84ae3d424')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links = []; self.images = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ('href', 'src'):
                self.links.append(value)
                if tag == 'img': self.images.append(value)


def main(*, root_report_QA_GO=False):
    if not root_report_QA_GO:
        raise ValueError('Explicit root report-QA GO is required before IO')
    target = OUTPUT / 'REVIEW.json'
    if target.exists():
        raise FileExistsError('Preserve every QA attempt')
    checks = []; failures = []

    def check(condition, label):
        checks.append(label)
        if not condition: failures.append(label)

    seal = read(REPORT / 'RENDER_SEAL.json'); data = read(REPORT / 'PROCESS_DATA.json'); plan = read(PLAN)
    original_seal = sha(REPORT / 'RENDER_SEAL.json')
    check(original_seal == '3dc3fddf4ffe71248c016b2539fdb02f307c6147a2d43e66dfbb1a0c71ec5b03', 'exact original render seal')
    check(sha(SOURCE) == seal['source_sha256'], 'renderer bytes')
    check(sha(PLAN) == seal['plan_sha256'] == data['source_plan_sha256'], 'pre-score plan bytes')
    for rel, digest in plan['source_sha256'].items(): check(sha(ROOT / rel) == digest, 'declared source ' + rel)
    check(seal['cases'] == plan['cases'] == [c['id'] for c in data['cases']] == CASES, 'fixed two cases')
    check(seal['arms'] == plan['arms'] == ARMS, 'uniform five arms')
    check(len(seal['assets']) == 22, '22 sealed assets')
    pngs = [name for name in seal['assets'] if name.endswith('.png')]
    check(len(pngs) == 20, '18 stages and two montages')
    dimensions = {}
    for name, digest in seal['assets'].items():
        check(sha(REPORT / name) == digest, 'asset SHA ' + name)
        if name.endswith('.png'):
            with Image.open(REPORT / name) as image:
                dimensions[name] = list(image.size); image.verify()
            check(dimensions[name] == ([3150, 2394] if 'montage' in name else [1050, 798]), 'PNG dimensions ' + name)
    page = (REPORT / 'index.html').read_text(); links = Links(); links.feed(page)
    check(len(links.images) == 20, 'HTML 20 image references')
    for link in links.links: check((REPORT / link).is_file(), 'HTML local link ' + link)
    for phrase in ['서로 다른 최종 방법 비교', '아직 채택', 'fallback은 활성계획', '기존 e 기본값은 유지', '독립 계측이 아닙니다']:
        check(phrase in page or phrase == '아직 채택' and '그래프 선정 후보' in page, 'HTML scope ' + phrase)
    check(data['default_remains_e'] and data['new_GT_projection'] == 0, 'default and no new GT projection')
    check(data['selection_profile'] == plan['selection_profile'], 'V30 selection profile binding')
    check(data['raw_observation_generation'] == 'FRESH_GUIDED_NMS_TLS_PARTIAL_DIRECTED_GRAPH_WINNER_V29', 'unchanged raw V29 measurement generation')
    check(data['observation_family'] == 'OBSERVED_BI_ASSOCIATED_MULTIGUIDE_ACTUAL_NATIVE_EDGES_V30', 'new V30 CAL family')
    check(data['population'] == 'V30_ASSOCIATION_GATED_RUNTIME_ACTIVE_DETERMINISTIC_ONE_LINE_PER_PHYSICAL_EDGE', 'new V30 active population')
    check(data['legacy_v29_cost_plan_is_internal_alias_only'] is True, 'legacy dictionary alias scope')
    binding=data['final_observation_protocol'];check(sha(ROOT / binding['path']) == binding['sha256'], 'final OBS protocol binding')
    check(data['policy_specific_CAL_model_seal_sha256'] == '5681a7bf36bc6ce4d7146df48f94ca9791be44c49a77954b7ad2132129b13adc', 'current V30 CAL model seal')
    for name in ['S_U_point_markers', 'GT_wire', 'new_measurement_NN_posefit_calls']:
        check(seal[name] == 0, 'render no-call/marker ' + name)
    for key in ['observation_seal', 'prediction_seal', 'score_seal']:
        binding = data[key]; check(sha(ROOT / binding['path']) == binding['sha256'], 'small upstream seal ' + key)
    score_seal = read(PRED / 'SCORE_SEAL.json'); protocol = read(PRED / 'PROTOCOL.json')
    scores = {}
    for partition in ['TEST', 'REMAINDER']:
        path = PRED / (partition + '_SCORE.json')
        check(sha(path) == score_seal['score_sha256'][partition], 'fixed score SHA ' + partition)
        scores[partition] = read(path)
    for path, digest in IMAGES + MASKS:
        check(protocol['input_sha256'][path] == digest, 'source input binding ' + path)
        check(sha(ROOT / path) == digest, 'actual RGB/mask SHA ' + path)
    for i, case in enumerate(data['cases']):
        ident = case['id']; partition = 'REMAINDER' if i == 0 else 'TEST'
        matches = [r for r in scores[partition]['rows'] if r['id'] == ident]
        check(len(matches) == 1, 'one fixed score row ' + ident)
        stored = matches[0]
        check(case['images'] == [f'case{i}_stage{s:02d}.png' for s in range(1, 10)], 'nine ordered stages ' + ident)
        check(case['montage'] == f'case{i}_montage.png', 'fixed montage ' + ident)
        check([m['arm'] for m in case['final_methods']] == ARMS, 'five final methods ' + ident)
        for key in ['S_U_point_markers', 'GT_wire', 'new_measurement_or_posefit_calls']:
            check(case[key] == 0, ident + ' ' + key)
        for key in ['stages1_to4_are_common_saved_evidence_not_new_PnP_steps', 'stages5_to9_are_separate_final_methods_not_optimizer_iterations']:
            check(case[key] is True, ident + ' ' + key)
        check(case['other_object_occlusion_mask_available'] is False, 'other mask absent ' + ident)
        check(case['initialS_model_only_ids'] == [k for k, state in enumerate(case['states']) if state == 'S'], 'S model only IDs ' + ident)
        check(all(case['states'][m['native_corner']] in ('B', 'I') and m['state'] == case['states'][m['native_corner']] for m in case['visible_sources']), 'BI-only q markers ' + ident)
        check(len(case['selected_lines']) == (3 if i == 0 else 7), 'candidate count ' + ident)
        for item in case['selected_lines']:
            ref = item['reference']; k = ref['native_corner']; mate = ref['native_neighbor']
            check(case['states'][k] in ('B', 'I') and case['states'][mate] != 'S', 'selected source gate ' + ref['line_name'] + ident)
            check(item['inlier_count'] > 0 and item['actual_span_px'] > 0, 'stored actual support ' + ref['line_name'] + ident)
        visible_ids = {m['native_corner'] for m in case['visible_sources']}
        for j, method in enumerate(case['final_methods']):
            arm = method['arm']; label = ident + '/' + arm; cost = method['source_cost_plan']
            check(method['metric'] == stored['metrics'][arm], 'exact full stored metric ' + label)
            check(method['WD_family'] == stored['fixed_family'][arm], 'family ' + label)
            check(method['joint_feasible'] == stored['accepted'][arm], 'accepted/joint ' + label)
            check(method['fallback'] == (not stored['accepted'][arm]), 'fallback ' + label)
            check(len(method['model_wire_projection_xy']) == 8 and method['outside_crop_model_vertices'] == 0, 'stored whole model projection ' + label)
            check(method['legacy_centroid_field_is_translation_parameter'] and method['centered_native_model_center_equals_translation'], 'centered native t scope ' + label)
            if j == 0:
                check(cost == {} and method['active_line_count'] == 0 and method['retained_observation_ids'] is None, 'e no new plan ' + label)
                continue
            groups = cost['active_groups']; owners = cost['new_owner_ids']
            check(len(groups) == method['active_line_count'] == cost['ledger']['physical_edge_factors_once'], 'line factor once ' + label)
            check(owners == method['new_owner_ids'] and len(set(owners)) == len(owners), 'owner ledger ' + label)
            check(set(method['retained_observation_ids']) == visible_ids - set(owners), 'retained observation IDs ' + label)
            check(cost['ledger']['new_precise_2D_KPs'] == 0 and cost['old_q_source_covariance_reliability_immutable'], 'no precise new q ' + label)
            check(method['active_line_count'] == (3 if i == 1 and j in (1, 2) else 0), 'active versus planned case count ' + label)
            for group in groups:
                check(group['full_2D_corner_measurement'] is False, 'line not point ' + label)
                check(group['selected_chain_ref']['line_name'] in {line['reference']['line_name'] for line in case['selected_lines']}, 'active selected ref ' + label)
    for arm, expected in [('SAME_FROZEN_E', 49), ('ASSOCIATED_EDGE_ALL_BI_NATIVE', 37), ('ASSOCIATED_EDGE_ALL_BI_CONTAINMENT', 37), ('ASSOCIATED_EDGE_SOURCE_NATIVE', 36), ('ASSOCIATED_EDGE_SOURCE_CONTAINMENT', 38)]:
        check(sum(row['metrics'][arm]['translation_cm'] <= 5 for row in scores['TEST']['rows']) == expected, 'TEST translation within5 ' + arm)
    md = (REPORT / 'REPORT_KO.md').read_text()
    for phrase in ['176.5351', 'V24', '208.2242', '실제 활성 선 3개', '활성 선 요인은 모두 0개', 'bounded S', '기본값 e는 유지']:
        check(phrase in md, 'MD scope ' + phrase)
    for link in ['index.html', 'PROCESS_DATA.json', 'RENDER_SEAL.json', 'case0_montage.png', 'case1_montage.png']:
        check((REPORT / link).is_file(), 'MD local artifact ' + link)
    check(sha(REPORT / 'RENDER_SEAL.json') == original_seal, 'original seal unchanged after QA')
    receipt = dict(schema='SMALL_ACTUAL_PROCESS_REPORT_REVIEW_V30_V1',
        UTC=datetime.now(timezone.utc).isoformat(), checks=len(checks), failure_count=len(failures), failures=failures,
        source={'path': str(Path(__file__).relative_to(ROOT)), 'sha256': sha(__file__)},
        renderer_source_sha256=sha(SOURCE), original_render_seal_sha256=original_seal,
        report_markdown={'path': str((REPORT / 'REPORT_KO.md').relative_to(ROOT)), 'sha256': sha(REPORT / 'REPORT_KO.md')},
        cases=CASES, uniform_methods=5, sealed_assets=22, PNGs=20, exact_metric_rows=10,
        viewed_images=['case0_montage.png', 'case1_montage.png'],
        direct_visual_findings=['Original RGB brightness preserved; common evidence and separate final methods shown.',
            'No S/U point markers or GT wire seen; ranges and model wires distinguished.',
            'Wood graph3/new active0 in all new methods; do not attribute result to new lines.',
            'P09 ALL active3 and joint accepted but translation237.2541cm; SOURCE active0.'],
        notes=['Historical176.5351cm is a V24 different-method result, not frozen e; root-confirmed source clarification.',
            'S image queries/point costs are zero; frozen coarse-derived bounded S regularizer remains active.',
            'No full journal parsing; exact row provenance is inherited from sealed renderer output and root full audit.',
            'No recomputation of RGB gradients/TLS, references, PnP, native objective or global model projection in QA.'],
        scope=dict(owner_QA_not_a_new_independent_native_objective_audit=True, new_RGB_NN_NMS_TLS_labels_posefits=0, full_input_prediction_observation_journal_decodes=0,
            prior_render_source_plan_assets_modified=0, root_full_audit_checks_not_repeated=99385),
        default_remains_e=True, new_method_adopted=False)
    target.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    seal_out = dict(schema='SMALL_ACTUAL_PROCESS_REPORT_REVIEW_SEAL_V30_V1',
        receipt_sha256=sha(target), review_source_sha256=sha(__file__), original_render_seal_sha256=original_seal,
        appended_report_sha256=sha(REPORT / 'REPORT_KO.md'))
    (OUTPUT / 'REVIEW_SEAL.json').write_text(json.dumps(seal_out, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(checks=len(checks), failure_count=len(failures), failures=failures,
        receipt_sha256=sha(target), seal_sha256=sha(OUTPUT / 'REVIEW_SEAL.json')), ensure_ascii=False))
    if failures: raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root-report-QA-GO', action='store_true')
    main(root_report_QA_GO=parser.parse_args().root_report_QA_GO)
