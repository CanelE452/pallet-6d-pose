"""이전 217장 RAW 학생의 누락된 동일 선택기 대조. 새 fit/GPU 추론 없음.

freeze는 저장된 원예측/최종 후보와 두 frozen scorer만 읽는다.
score는 모든 이름을 잠근 다음 기존 후보 metric을 연결한다.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from . import common as C

PRIVATE = C.RAW/'historical_raw'
LOCK = C.DOC/'HISTORICAL_RAW_LOCK.json'
RESULT = C.DOC/'HISTORICAL_RAW_RESULTS.json'
OLD_ORACLE = C.ROOT/'data/pallet/results/pallet_oracle_mechanism_followup_v1/pose_oracle'
PAPER_RAW = C.ROOT/'data/pallet/results/pallet_selftraining_paper_closure_v1'
PAPER_DOC = C.ROOT/'_docs/experiments/pallet_selftraining_paper_closure_v1'
SELECTORS = ('D9', 'GEO', 'NEWGEO')


def verify():
    value = C.read(LOCK)
    assert not value['reference_coordinates_read'] and value['scorers_are_existing_common_models']
    for binding in value['files']+value['sources']:
        C.verify(binding)
    return value


def candidate_by_name(record, name):
    matches = [row for row in record['hypotheses'] if row['name'] == name]
    assert len(matches) <= 1
    return matches[0]['pose'] if matches else record['current']


def fit_contract():
    import yaml
    original = C.ROOT/'_docs/experiments/pallet_type_selftrain_v1/selftrain_recovery_v1/pose_only'
    data = C.ROOT/'data/pallet/results/pallet_type_selftrain_v1/selftrain_recovery_v1/pose_only'
    fitted, actual_args, inputs = {}, {}, []
    for target in ('RAW', 'REF'):
        path = original/f'FIT_{target}_LR5.json'
        fit = C.read(path)
        assert fit['complete'] and fit['optimizer_steps'] == 320 and fit['epochs'] == 5
        assert fit['exact_R0_initialization'] and fit['protected_state_exact']
        C.verify(fit['checkpoint']); C.verify(fit['protocol'])
        argspath = data/f'runs/{target}_LR5/args.yaml'
        args = yaml.safe_load(argspath.read_text())
        assert args['lr0'] == 1e-5 and args['epochs'] == 5 and args['optimizer'] == 'AdamW'
        assert args['seed'] == 42 and args['batch'] == args['nbs'] == 16
        fitted[target] = fit['checkpoint']; actual_args[target] = args
        inputs += [C.bind(path), fit['checkpoint'], fit['protocol'], C.bind(argspath)]
    assert C.read(original/'FIT_RAW_LR5.json')['protocol'] == C.read(original/'FIT_REF_LR5.json')['protocol']
    # 이름·출력·RAW/REF 데이터 참조 외 설정은 같아야 한다.
    allowed = {'data', 'name', 'project', 'save_dir', 'model'}
    assert actual_args['RAW']['model'] == actual_args['REF']['model']
    assert {k: v for k, v in actual_args['RAW'].items() if k not in allowed} == {k: v for k, v in actual_args['REF'].items() if k not in allowed}
    audit = C.read(PAPER_DOC/'CORE_COMPARABILITY_AUDIT.json')
    assert audit['real_unique'] == 217 and audit['updates'] == 320
    assert audit['identical_order_images_boxes_masks'] and audit['paired_targets'] == 'EXACT_MATCH except supervised xy'
    assert audit['train_eval_RGB_overlap'] == audit['train_eval_recording_overlap'] == []
    inputs += [C.bind(PAPER_DOC/'CORE_COMPARABILITY_AUDIT.json')]
    return dict(checkpoints=fitted, real_unique=217, real_slots=512, synthetic_slots=512,
        updates=320, epochs=5, lr0=1e-5, seed=42, pose_only=True,
        image_box_support_export_parity=True, original_audit_reused=True,
        limitation='과거 audit의 공통 args.lr0는 LR4 기본값이다. 본 LR5는 각 실제 runs/args.yaml의 1e-5를 검증했다. 기존 first-batch/설정 감사를 전체 320 tensor 전수 일치로 승격하지 않는다.',
        versus_clean78='clean78은 회원/노출 조성 및 post-affine common support 처리가 달라, 217↔78을 clean성 하나의 인과 대조라고 하지 않는다.'), inputs


def freeze():
    if LOCK.exists():
        verify(); print('HISTORICAL_RAW_ALREADY_LOCKED', flush=True); return
    from scripts.research.pallet_clean_to_pose_transfer_v1.selector_compat import guard_reference_reads
    reads = guard_reference_reads()
    import torch
    from scripts.research.pallet_selector_recovery_v1 import features as F, models as M, common as U
    from scripts.research.pallet_clean_to_pose_transfer_v1 import eval_student as E
    from .audit_controls import close
    torch.set_num_threads(2)
    contract, sources = fit_contract()
    historical_lock = C.read(OLD_ORACLE/'CANDIDATES_LOCK.json')
    assert historical_lock['no_reference_coordinates_read']
    for binding in historical_lock['files']+historical_lock['sources']:
        C.verify(binding)
    predicted_lock = C.read(PAPER_DOC/'PREDICTIONS_LOCK.json')
    assert not predicted_lock['reference_coordinates_read']
    for binding in predicted_lock['files']:
        C.verify(binding)
    base = E.paths(42)
    rows = C.read(base['metadata']); E.validate_membership(rows)
    existing_predictions = C.read(base['predictions'])
    existing_candidates = C.read(base['candidates'])
    source_candidates = C.read(OLD_ORACLE/'PLASTIC_CANDIDATES.json')
    source_predictions = C.read(PAPER_RAW/'PREDICTIONS.json')
    source_poses = C.read(PAPER_RAW/'POSE_PREDICTIONS.json')
    assert source_candidates['ids'] == [row['id'] for row in rows] and source_candidates['GT_input'] is False
    for source, current in (('R0', 'R0'), ('REF_LR5', 'OLD_REF')):
        close(source_predictions[source], existing_predictions[current])
        close(source_candidates['arms'][source], existing_candidates[current])
    metadata = {row['id']: row for row in C.read(C.ROOT/'data/pallet/results/pallet_visible_refine_hidden_pnp_v1/INFERENCE_METADATA.json')}
    for row in rows:
        for key in ('K', 'xyz', 'hw'):
            close(row[key], metadata[row['id']][key])
        assert source_candidates['groups'][row['id']] == dict(recording=row['recording'], severity=row['severity'])
    old_checkpoint = C.read(C.DOC/'SAME_GEO_CONTROLS_LOCK.json')['same_frozen_scorer']
    new_checkpoint = C.read(C.DOC/'SELECTOR_CALIBRATION_FIT.json')['checkpoint']
    scorer_bindings = dict(GEO=old_checkpoint, NEWGEO=new_checkpoint)
    scorers = {}
    for name, binding in scorer_bindings.items():
        C.verify(binding)
        scorers[name] = torch.load(C.ROOT/binding['path'], map_location='cpu', weights_only=False)
        assert scorers[name]['d'] == 94 and scorers[name]['variant'] == 'GEO_LINEAR'
    raw_prediction = source_predictions['RAW_LR5']; raw_candidate = source_candidates['arms']['RAW_LR5']
    assert set(raw_prediction) == set(raw_candidate) == {row['id'] for row in rows}
    close(source_poses['RAW_LR5'], {fid: record['current'] for fid, record in raw_candidate.items()})
    features, decisions = {}, {f'OLD_RAW_{name}': {} for name in scorers}
    poses = {'OLD_RAW_D9': source_poses['RAW_LR5'], **{f'OLD_RAW_{name}': {} for name in scorers}}
    counts = {name: Counter() for name in scorers}
    for row in rows:
        fid = row['id']; record = raw_candidate[fid]
        feature = F.extract(raw_prediction[fid], row['K'], row['xyz'], row['hw'])
        assert feature['selection'] == record['selected_name']
        assert [h['name'] for h in feature['hypotheses']] == sorted(h['name'] for h in record['hypotheses'])
        features[fid] = feature
        for name, scorer in scorers.items():
            chosen, scores = record['selected_name'], None
            if feature['valid']:
                scores = M.scores(scorer, np.asarray(feature['features'], np.float32)[None])[0]
                index = int(M.selection(scores[None], U.HYP)[0])
                assert index == 1-int(M.selection(scores[None, ::-1], U.HYP[::-1])[0])
                chosen = U.HYP[index]
            arm = f'OLD_RAW_{name}'
            poses[arm][fid] = candidate_by_name(record, chosen)
            decisions[arm][fid] = dict(selected=chosen, D9_selected=record['selected_name'],
                changed=chosen != record['selected_name'], valid_pair=bool(feature['valid']),
                scores=scores.tolist() if scores is not None else None)
            counts[name].update(frames=1, changed=int(chosen != record['selected_name']), valid_pair=int(feature['valid']))
    files = []
    for name, value in [('METADATA', rows), ('PREDICTIONS', {'RAW_LR5': raw_prediction}),
                        ('CANDIDATES', {'RAW_LR5': raw_candidate}), ('FEATURES', features),
                        ('DECISIONS', decisions), ('POSES', poses)]:
        path = PRIVATE/(name+'.json'); C.save(path, value, True); files.append(C.bind(path))
    sources += [C.bind(path) for path in (OLD_ORACLE/'CANDIDATES_LOCK.json', OLD_ORACLE/'PLASTIC_CANDIDATES.json',
        PAPER_DOC/'PREDICTIONS_LOCK.json', PAPER_RAW/'PREDICTIONS.json', PAPER_RAW/'POSE_PREDICTIONS.json',
        base['metadata'], base['predictions'], base['candidates'], C.DOC/'SAME_GEO_CONTROLS_LOCK.json',
        C.DOC/'SELECTOR_CALIBRATION_FIT.json', Path(__file__), Path(F.__file__), Path(M.__file__))]
    sources += list(scorer_bindings.values())
    C.save(LOCK, dict(created_at=C.now(), files=files, sources=sources, contract=contract,
        same_R0_REF128_prediction_and_candidate_parity=True, same128_metadata_and_population=True,
        reference_coordinates_read=False, read_guard_active=True, read_paths=sorted(set(reads)),
        scorers_are_existing_common_models=True, scorers=scorer_bindings,
        final_pose_from_unchanged_cached_candidates=True, counts={k: dict(v) for k, v in counts.items()},
        new_student_fits=0, new_selector_fits=0, optimizer_updates=0, GPU_seconds=0,
        scope='DEV 결과를 본 뒤 누락된 matched OLD_RAW217 대조 추가; 사전등록으로 소급하지 않음'), True)
    print('HISTORICAL_RAW_LOCKED', {k: dict(v) for k, v in counts.items()}, flush=True)


def score():
    lock = verify()
    if RESULT.exists():
        value = C.read(RESULT)
        for binding in value['sources']+value['private_artifacts']:
            C.verify(binding)
        print('HISTORICAL_RAW_ALREADY_SCORED', flush=True); return
    from scripts.research.pallet_clean_to_pose_transfer_v1 import eval_student as E
    from scripts.research.pallet_pose_objective_followup_v2 import metric_baseline as M
    from scripts.research.pallet_oracle_mechanism_followup_v1 import pose_oracle as O
    from .audit_controls import close
    C.save(PRIVATE/'SCORING_START.json', dict(created_at=C.now(), lock=C.bind(LOCK)), True)
    historical = C.read(OLD_ORACLE/'PLASTIC_METRICS.json')['arms']['RAW_LR5']
    rows = C.read(PRIVATE/'METADATA.json'); groups = E.group_ids(rows)
    poses = C.read(PRIVATE/'POSES.json'); decisions = C.read(PRIVATE/'DECISIONS.json')
    _, truth = O.D.Pose.metadata('REAL_DEV')
    candidates = C.read(PRIVATE/'CANDIDATES.json')['RAW_LR5']
    metrics = {f'OLD_RAW_{name}': {} for name in SELECTORS}
    options_private = {}
    for row in rows:
        fid = row['id']; entry = historical[fid]; options_private[fid] = []
        for option in entry['hypotheses']:
            pose = candidate_by_name(candidates[fid], option['name'])
            metric = M.extend_metric(option['metric'], pose, truth[fid])
            options_private[fid].append(dict(name=option['name'], metric=metric))
        for selector in SELECTORS:
            arm = f'OLD_RAW_{selector}'
            name = candidates[fid]['selected_name'] if selector == 'D9' else decisions[arm][fid]['selected']
            chosen = [option['metric'] for option in options_private[fid] if option['name'] == name]
            assert len(chosen) == 1
            metrics[arm][fid] = chosen[0]
            M.extend_metric(chosen[0], poses[arm][fid], truth[fid])
    prior_raw = C.read(C.ROOT/'data/pallet/results/pallet_pose_objective_followup_v2/metric_baseline/FRAME_METRICS_PRIVATE.json')['PLASTIC']['OLD_RAW']
    close(prior_raw, metrics['OLD_RAW_D9'])
    current_result = C.read(C.DOC/'CURRENT_GEO_RESULTS.json')
    for binding in current_result['sources']+current_result['private_artifacts']:
        C.verify(binding)
    all_metrics = C.read(C.RAW/'CURRENT_GEO_FRAME_METRICS_PRIVATE.json')
    all_metrics.update(metrics)
    comparisons = [(f'OLD_RAW_{s}', f'OLD_REF_{s}') for s in SELECTORS]
    comparisons += [(f'R0_{s}', f'OLD_RAW_{s}') for s in SELECTORS]
    comparisons += [('OLD_RAW_D9', 'OLD_RAW_GEO'), ('OLD_RAW_D9', 'OLD_RAW_NEWGEO'), ('OLD_RAW_GEO', 'OLD_RAW_NEWGEO')]
    for student in ('RAW_CLEAR_S42', 'REF_CLEAR_S42', 'RAW_OCC_S42', 'REF_OCC_S42', 'RAW_OCC_S43', 'REF_OCC_S43'):
        comparisons += [(f'OLD_RAW_{s}', f'{student}_{s}') for s in SELECTORS]
    summaries = {g: {a: M.summarize(values[f] for f in ids) for a, values in all_metrics.items()} for g, ids in groups.items()}
    paired = {g: {b+'-minus-'+a: M.paired(all_metrics[a], all_metrics[b], ids) for a, b in comparisons} for g, ids in groups.items()}
    loto = {rec: {b+'-minus-'+a: M.paired(all_metrics[a], all_metrics[b],
        [r['id'] for r in rows if r['severity'] != 'CLEAN' and r['recording'] != rec]) for a, b in comparisons}
        for rec in sorted({r['recording'] for r in rows})}
    association = {}
    for selector in SELECTORS:
        arm = f'OLD_RAW_{selector}'
        association[arm] = dict(predictions=C.bind(PRIVATE/'PREDICTIONS.json'), prediction_arm='RAW_LR5',
            candidates=C.bind(PRIVATE/'CANDIDATES.json'), candidate_arm='RAW_LR5', metadata=C.bind(PRIVATE/'METADATA.json'),
            decisions=C.bind(PRIVATE/'DECISIONS.json') if selector != 'D9' else None, decisions_arm=arm,
            poses=C.bind(PRIVATE/'POSES.json'), poses_arm=arm)
    artifacts = []
    for name, value in [('FRAME_METRICS_PRIVATE', metrics), ('ALL_FRAME_METRICS_PRIVATE', all_metrics),
                        ('CASE_BINDINGS_PRIVATE', association),
                        ('ORACLE_METRICS_SELECTIONS_PRIVATE', dict(candidate_metrics={'RAW_LR5': options_private}, posthoc_selections={}))]:
        path = PRIVATE/(name+'.json'); C.save(path, M.clean(value), True); artifacts.append(C.bind(path))
    # 최종 결합 보고서는 새 세 팔만 root alias에서 읽는다. 기존 24팔은 덮어쓰지 않는다.
    for name, value in [('FRAME_METRICS', metrics), ('POSES', poses), ('CASE_BINDINGS', association)]:
        path = C.RAW/f'HISTORICAL_RAW_{name}_PRIVATE.json'
        C.save(path, M.clean(value), True); artifacts.append(C.bind(path))
    result = dict(created_at=C.now(), groups=summaries, paired=paired, leave_one_recording_out_NATURAL99=loto,
        available_models=list(all_metrics), added_controls=['OLD_RAW_'+s for s in SELECTORS],
        matched217_pair={s: M.classify_candidate(summaries['NATURAL99']['OLD_REF_'+s], summaries['NATURAL99']['OLD_RAW_'+s]) for s in SELECTORS},
        contract=lock['contract'], unchanged_original_D9_OLD_RAW128=True,
        sources=[C.bind(path) for path in (LOCK, PRIVATE/'SCORING_START.json', OLD_ORACLE/'PLASTIC_METRICS.json',
            C.DOC/'CURRENT_GEO_RESULTS.json', Path(__file__), Path(M.__file__),
            C.ROOT/'data/pallet/results/pallet_pose_objective_followup_v2/metric_baseline/FRAME_METRICS_PRIVATE.json')],
        private_artifacts=artifacts, new_student_fits=0, new_selector_fits=0, optimizer_updates=0, GPU_seconds=0,
        evidence='같은217 RAW/REF, 같은 frozen 선택기와128/99. reusedDEV/geometry-derived6D. 새로운 seed 반복이 아님.',
        correction_comparison='matched217 RAW-vs-REF를 같은 선택기 안에서 비교한다. 217↔78의 회원/노출/support 변경과 섞지 않는다.')
    C.save(RESULT, M.clean(result), True)
    natural = summaries['NATURAL99']; lines = ['# 이전 217장 RAW 대조 보완', '',
        '기존 217장 RAW 학생을 추가 학습하지 않고 동일 D9·기존 GEO·새 공통 GEO로 평가했다. 원표는 보존했다.', '',
        '| 선택기 | RAW T 중앙값 cm | REF T 중앙값 cm | RAW R 중앙값 ° | REF R 중앙값 ° |', '|---|---:|---:|---:|---:|']
    for selector in SELECTORS:
        a, b = [natural[f'OLD_{target}_{selector}']['full_population'] for target in ('RAW', 'REF')]
        lines.append(f"| {selector} | {a['translation_cm']['median']:.4f} | {b['translation_cm']['median']:.4f} | {a['rotation_deg']['median']:.4f} | {b['rotation_deg']['median']:.4f} |")
    lines += ['', '두 학생은 실제 LR 1e-5/320 update/217장 계약이며, checkpoint와 실제 args.yaml을 확인했다. 과거 공통 audit args의 LR4 기본값을 이 LR5 실행값으로 오해하지 않는다.', '',
        'R0/REF 예측·후보128개가 현재 캐시와 각각 일치하고, OLD_RAW D9의 위치·회전·yaw는 기존 metric과도 일치했다.', '',
        '표의 주 비교는 동일217 RAW→REF다. clean78과 비교하면 회원/노출 조성과 common support 변화가 함께 있어 clean성 하나의 효과로 해석할 수 없다.', '',
        '실사 평가를 본 뒤 추가한 대조다. 독립 TEST나 새로운 seed 반복이 아니며, 새 fit/GPU 추론은 0회다.', '', '[전체 난도·기록·짝지은 결과](HISTORICAL_RAW_RESULTS.json)', '']
    C.save(C.DOC/'HISTORICAL_RAW_REPORT_KO.md', '\n'.join(lines), True)
    print('HISTORICAL_RAW_RESULTS', {s: {key: natural['OLD_RAW_'+s]['full_population'][key]['median'] for key in ('translation_cm', 'rotation_deg')} for s in SELECTORS}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['freeze', 'score'])
    args = parser.parse_args(); globals()[args.phase]()
