"""CPU-only, GT-dependent coordinate/expert diagnostics on frozen outputs.

No generated target, weight, prediction, or selection is an inference artifact.
Run ``freeze`` before ``score``. Private point errors/choices stay in data/.
"""
from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path
import time

import numpy as np
from . import common as C
from scripts.research.pallet_dim_conditioned_p_v1 import eval_math as EM
from scripts.paper.pose_metric_closure_v1.symmetry_aware_pose_metrics import pose_auc

RAW = C.RAW / 'coordinate_oracle'
ARMS = {'PLASTIC': ('R0', 'TEACHER', 'RAW_LR5', 'REF_LR5'),
        'WOOD': ('R0', 'TEACHER', 'WOOD_RAW_LR5', 'WOOD_REF_LR5')}
RADII = (0, 2, 4, 8, 12)
THRESHOLDS = (5, 10, 20)
FLAGS = dict(is_oracle=True, GT_DEPENDENT=True, DIAGNOSTIC_ONLY=True,
             production_training_or_inference_input=False)


def paths(material):
    directory = C.P.RAW if material == 'PLASTIC' else C.M.RAW
    return directory


def freeze():
    dst = C.DOC / 'ORACLE_COORDINATE_PROTOCOL.json'
    if dst.exists():
        for binding in C.read(dst)['inputs']:
            C.verify(binding)
        print('COORDINATE_PROTOCOL_ALREADY_FROZEN')
        return
    inputs = [C.P.TRUTH, C.P.FINAL, C.P.SPLIT, C.P.RAW / 'ANCHOR_POINTS.json',
              C.P.DOC / 'PSEUDO_LABEL_QUALITY.json', C.P.DOC / 'CORE_RESULTS.json',
              C.M.DOC / 'WOOD_RESULTS.json', C.M.RAW / 'EVAL_METADATA.json',
              Path(__file__), Path(EM.__file__)]
    for material in ARMS:
        directory = paths(material)
        lock = (C.P.DOC / 'PREDICTIONS_LOCK.json' if material == 'PLASTIC'
                else C.M.DOC / 'WOOD_PREDICTIONS_LOCK.json')
        for binding in C.read(lock)['files']:
            C.verify(binding)
        inputs += [lock, directory / 'PREDICTIONS.json', directory / 'FRAME_METRICS.json',
                   directory / 'FIXED_ID_METRICS.json',
                   C.RAW / 'pose_oracle' / f'{material}_METRICS.json']
    C.save(dst, dict(**FLAGS, utc=C.now(), pool=ARMS, radii_px=RADII,
                    radii_status='Descriptive grid fixed before this computation; old local search radius12 included',
                    coordinate_unit='native original image pixels',
                    legacy_primary='Strict native corner0..7 identity, legacy valid mask and unchanged selected-detection match gate; full failure denominator',
                    legacy_supplement='Separate original whole-object allowed-symmetry metric; no per-point oracle with independent symmetry choices',
                    verified='Existing FINAL_V2 66 DIRECT_VISIBLE fixed-ID points; no box-match gate, same historical visible metric',
                    whole_output='One complete existing output per frame, maximize correct count separately for each threshold; lexicographic arm-name tie',
                    per_point='Minimum error at exactly the same native corner ID; no free/Hungarian/nearest-corner matching; never PnP input',
                    whole_pose='Choose among four already production-selected poses; minimize original ADDsym_normalized; not W/D union candidate oracle',
                    detection='Only originally box-mismatched frames; compare stored-candidate best box IoU and best PCK10 independently, original match gate and whole-object symmetry retained',
                    failure='Keep native diagonal penalty; local radius cannot repair missing or box-mismatched points',
                    local='max(e-r,0) only at available metric-valid points; image/rigidity/semantic evidence ignored',
                    roles='All reused DEV; no independent confirmation and no deployment claim',
                    inputs=[C.bind(p) for p in sorted(set(inputs))]), True)
    print('COORDINATE_PROTOCOL_FROZEN')


def point(prediction, corner):
    index = prediction.get('selected_index')
    if index is None:
        return None
    q = np.asarray(prediction['candidates'][index]['keypoints_xy'][corner], float)
    return q if q.shape == (2,) and np.isfinite(q).all() and not np.all(q == -1) else None


def fixed_error(prediction, corner, target, penalty, matched=True):
    q = point(prediction, corner)
    missing = q is None or not matched
    return (float(penalty) if missing else float(np.linalg.norm(q - target))), missing


def stats(values):
    e = np.asarray(values, float)
    assert e.ndim == 1 and np.isfinite(e).all() and (e >= 0).all()
    return dict(points=len(e), mean_px=float(e.mean()) if len(e) else None,
                median_px=float(np.median(e)) if len(e) else None,
                P90_px=float(np.quantile(e, .9)) if len(e) else None,
                above20=int((e > 20).sum()),
                PCK={str(t): dict(correct=int((e <= t).sum()), total=len(e),
                                 fraction=float((e <= t).mean()) if len(e) else None)
                     for t in THRESHOLDS})


def choose_whole(frame_points, arms, threshold):
    """One existing whole output. Tie depends on arm name, not pool ordering."""
    assert frame_points and arms
    return min(arms, key=lambda a: (-sum(r['errors'][a] <= threshold for r in frame_points), a))


def cross_tab(points, teacher, student, threshold):
    counts = Counter()
    for r in points:
        t = not r['missing'][teacher] and r['errors'][teacher] <= threshold
        s = not r['missing'][student] and r['errors'][student] <= threshold
        counts['both_correct' if t and s else 'teacher_only' if t else 'student_only' if s else 'both_wrong'] += 1
    return dict(points=len(points), threshold_px=threshold,
                **{k: counts[k] for k in ('both_correct', 'teacher_only', 'student_only', 'both_wrong')},
                teacher_missing=sum(r['missing'][teacher] for r in points),
                student_missing=sum(r['missing'][student] for r in points))


def local_errors(points, arm, radius):
    assert radius >= 0
    return [r['errors'][arm] if r['missing'][arm] else max(0., r['errors'][arm] - radius) for r in points]


def coordinate_summary(points, arms):
    assert points
    frame_points = {}
    for p in points:
        frame_points.setdefault(p['frame_id'], []).append(p)
    baselines = {a: dict(**stats([r['errors'][a] for r in points]),
                         available_points=sum(not r['missing'][a] for r in points)) for a in arms}
    whole = {}
    choices = {}
    for threshold in THRESHOLDS:
        chosen = {fid: choose_whole(rr, arms, threshold) for fid, rr in frame_points.items()}
        values = [p['errors'][chosen[p['frame_id']]] for p in points]
        entry = stats(values)
        entry['objective'] = f'frame correct count at {threshold}px; other columns describe that same whole output choice'
        entry['selected_arm_counts'] = dict(Counter(chosen.values()))
        entry['available_points'] = sum(not r['missing'][chosen[r['frame_id']]] for r in points)
        entry['gain_correct_vs_each'] = {a: entry['PCK'][str(threshold)]['correct'] - baselines[a]['PCK'][str(threshold)]['correct'] for a in arms}
        assert all(v >= 0 for v in entry['gain_correct_vs_each'].values())
        whole[str(threshold)] = entry
        choices[str(threshold)] = chosen
    best_arms = [min(arms, key=lambda a: (p['errors'][a], a)) for p in points]
    per = stats([p['errors'][a] for p, a in zip(points, best_arms)])
    per['selected_arm_counts'] = dict(Counter(best_arms))
    per['available_points'] = sum(any(not p['missing'][a] for a in arms) for p in points)
    per['not_a_rigid_pose_or_attainable_student'] = True
    for threshold in THRESHOLDS:
        assert per['PCK'][str(threshold)]['correct'] >= whole[str(threshold)]['PCK'][str(threshold)]['correct']
    local = {a: {str(r): stats(local_errors(points, a, r)) for r in RADII} for a in arms}
    cross = {a: {str(t): cross_tab(points, 'TEACHER', a, t) for t in (10, 20)} for a in arms if a != 'TEACHER'}
    return dict(frames=len(frame_points), points=len(points), baselines=baselines,
                whole_output_by_objective=whole, per_point_fixed_identity=per,
                teacher_student_cross_tabs=cross, local_move_optimistic=local), dict(whole_output_choices=choices, per_point_choices=best_arms)


def group_points(points):
    groups = {'ALL': points}
    for key in ('recording', 'severity'):
        for value in sorted({r[key] for r in points}):
            groups[f'{key}:{value}'] = [r for r in points if r[key] == value]
    return groups


def legacy(material, predictions, truth):
    directory = paths(material)
    old_fixed = C.read(directory / 'FIXED_ID_METRICS.json')
    old_sym = C.read(directory / 'FRAME_METRICS.json')
    metadata = (C.P.records() if material == 'PLASTIC' else C.read(C.M.RAW / 'EVAL_METADATA.json'))
    ids = [r['id'] for r in metadata]
    assert len(ids) == (128 if material == 'PLASTIC' else 45)
    assert all(set(predictions[a]) == set(ids) for a in ARMS[material])
    all_points, symmetry = [], {a: {} for a in ARMS[material]}
    parity = 0
    for m in metadata:
        fid = m['id']; gt = truth[fid]
        matched = old_fixed['R0'][fid]['matched']
        target = np.asarray(gt['gt'], float)
        valid = np.asarray(gt['valid'], bool) & np.isfinite(target).all(-1) & ~(target == -1).all(-1)
        assert valid[:8].tolist() == old_fixed['R0'][fid]['canonical_valid']
        for arm in ARMS[material]:
            p = predictions[arm][fid]
            selected = C.P.selected(p); reference = C.P.selected(predictions['R0'][fid])
            assert p['selected_index'] == predictions['R0'][fid]['selected_index']
            assert (selected or {}).get('box_xyxy') == (reference or {}).get('box_xyxy')
            q = np.full((9, 2), np.nan) if selected is None else selected['keypoints_xy']
            symmetry[arm][fid] = dict(id=fid, **EM.measure(q, gt['gt'], gt['valid'], gt['permutations'], gt['hw'], matched, selected is not None))
            if arm in old_sym:
                np.testing.assert_allclose(symmetry[arm][fid]['errors'], old_sym[arm][fid]['errors'], atol=1e-10, rtol=0)
        for corner in np.flatnonzero(valid[:8]):
            errors, missing = {}, {}
            for arm in ARMS[material]:
                errors[arm], missing[arm] = fixed_error(predictions[arm][fid], corner, target[corner], math.hypot(*gt['hw']), matched)
                if arm in old_fixed:
                    assert abs(errors[arm] - old_fixed[arm][fid]['canonical_errors'][corner]) < 1e-10
                    parity += 1
            all_points.append(dict(frame_id=fid, corner_id=int(corner),
                                   recording=m.get('recording', m.get('recording_group')),
                                   severity=m['severity'], errors=errors, missing=missing))
    assert len(all_points) == (985 if material == 'PLASTIC' else 346)
    sym_groups = {}
    for name, pp in group_points(all_points).items():
        frame_ids = sorted({p['frame_id'] for p in pp})
        baseline = {a: EM.summary([symmetry[a][i] for i in frame_ids]) for a in ARMS[material]}
        whole = {}
        for threshold in THRESHOLDS:
            selected = {fid: min(ARMS[material], key=lambda a: (-sum(e <= threshold for e in symmetry[a][fid]['errors']), a)) for fid in frame_ids}
            rows = [symmetry[selected[i]][i] for i in frame_ids]
            whole[str(threshold)] = dict(summary=EM.summary(rows), selected_arm_counts=dict(Counter(selected.values())))
        sym_groups[name] = dict(baselines=baseline, whole_output_by_objective=whole,
                               contract='Each whole output gets only original allowed whole-object symmetry; no pointwise mixing after independent branch choices')
    return all_points, sym_groups, parity


def verified(predictions, truth):
    saved = C.read(C.P.RAW / 'ANCHOR_POINTS.json')
    reference = C.read(C.P.FINAL)
    assert reference['reference_version'] == 'VERIFIED_VISIBLE_ANCHOR_FINAL_V2'
    expected = {}
    for fi, ci in reference['review_queue']:
        f = reference['frames'][fi]; c = f['corners'][ci]
        if f['frame_id'] in predictions['R0'] and c['status'] == 'DIRECT_VISIBLE':
            assert c['coordinate_source'] == 'manual_click' and ci < 8
            expected[f['frame_id'], ci] = c['xy']
    assert len(saved) == len(expected) == 66
    result = []
    for p in saved:
        fid, corner = p['frame_id'], p['corner_id']
        assert expected[fid, corner] == p['verified_xy']
        errors, missing = {}, {}
        for arm in ARMS['PLASTIC']:
            errors[arm], missing[arm] = fixed_error(predictions[arm][fid], corner, np.asarray(expected[fid, corner]), math.hypot(*truth[fid]['hw']))
            assert abs(errors[arm] - p['errors'][arm]) < 1e-10
            assert missing[arm] == p['missing'][arm]
        result.append({k: p[k] for k in ('frame_id', 'corner_id', 'recording', 'severity')} | dict(errors=errors, missing=missing))
    assert len({r['frame_id'] for r in result}) == 16
    return result


def pose_summary(material):
    source = C.RAW / 'pose_oracle' / f'{material}_METRICS.json'
    data = C.read(source)['arms']
    ids = sorted(data['R0'])
    arms = ARMS[material]
    baseline = {}
    for arm in arms:
        rows = [data[arm][i]['current'] for i in ids]
        baseline[arm] = dict(ADDsym_AUC=pose_auc([r['ADDsym_normalized'] if r['available'] else float('inf') for r in rows], 1.),
                             frames=len(ids), available=sum(r['available'] for r in rows))
    chosen = {}; rows = []
    for fid in ids:
        options = [(data[a][fid]['current']['ADDsym_normalized'], a) for a in arms if data[a][fid]['current']['available']]
        arm = min(options)[1] if options else None
        chosen[fid] = arm
        rows.append(data[arm][fid]['current'] if arm else dict(id=fid, available=False))
    auc = pose_auc([r['ADDsym_normalized'] if r['available'] else float('inf') for r in rows], 1.)
    assert all(auc >= b['ADDsym_AUC'] - 1e-12 for b in baseline.values())
    return dict(**FLAGS, baselines=baseline, whole_output_oracle_AUC=auc,
                gaps={a: auc - b['ADDsym_AUC'] for a, b in baseline.items()},
                frames=len(ids), available=sum(r['available'] for r in rows),
                selected_arm_counts=dict(Counter(chosen.values())),
                source=C.bind(source), candidates='Only each model original production-selected pose; no alternative-WD or per-point mix',
                objective='ADDsym only; not simultaneous axis/R/yaw/t optimum'), chosen


def box_iou(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    intersection = float(np.maximum(0., np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod())
    union = float(np.maximum(0., a[2:] - a[:2]).prod() + np.maximum(0., b[2:] - b[:2]).prod() - intersection)
    return intersection / union if union > 0 else 0.


def detection_diagnostic(material, predictions, truth):
    old = C.read(paths(material) / 'FRAME_METRICS.json')['R0']
    ids = sorted(old)
    mismatches = [fid for fid in ids if not old[fid]['matched']]
    if not mismatches:
        return dict(status='NOT_NEEDED_ALL_SELECTED_DETECTIONS_MATCH', frames=len(ids), mismatches=0,
                    extra_inference=0, alternative_candidate_scoring=0), {}
    public, private = {}, {}
    for arm in ARMS[material]:
        current_rows = {}; alternatives = {}
        for fid in ids:
            t = truth[fid]; p = predictions[arm][fid]; c = C.P.selected(p)
            q = c['keypoints_xy'] if c else np.full((9, 2), np.nan)
            current_rows[fid] = dict(id=fid, **EM.measure(q, t['gt'], t['valid'], t['permutations'], t['hw'], old[fid]['matched'], c is not None))
        for fid in mismatches:
            t = truth[fid]; p = predictions[arm][fid]; rows = []
            for index, candidate in enumerate(p['candidates']):
                iou = box_iou(candidate['box_xyxy'], t['box'])
                measured = dict(id=fid, **EM.measure(candidate['keypoints_xy'], t['gt'], t['valid'], t['permutations'], t['hw'], iou >= .5, True))
                rows.append(dict(index=index, box_iou=iou, matched=iou >= .5,
                                 correct10=sum(e <= 10 for e in measured['errors']), metric=measured))
            best_box = min(rows, key=lambda r: (-r['box_iou'], r['index'])) if rows else None
            best_kp = min(rows, key=lambda r: (-r['correct10'], r['index'])) if rows else None
            alternatives[fid] = dict(candidates=rows, selected_index=p['selected_index'], best_box_index=best_box['index'] if best_box else None,
                                     best_keypoint_index=best_kp['index'] if best_kp else None)
        summary = dict(status='GT_ASSISTED_STORED_CANDIDATES_ONLY', frames=len(ids), mismatches=len(mismatches),
                       mismatches_with_multiple_candidates=sum(len(r['candidates']) > 1 for r in alternatives.values()),
                       any_box_match_recoverable=sum(any(c['matched'] for c in r['candidates']) for r in alternatives.values()),
                       main_current=EM.summary(list(current_rows.values())), methods={})
        for label, key in [('best_box_IoU', 'best_box_index'), ('best_PCK10', 'best_keypoint_index')]:
            new_rows = dict(current_rows)
            for fid, r in alternatives.items():
                if r[key] is not None:
                    new_rows[fid] = r['candidates'][r[key]]['metric']
            summary['methods'][label] = dict(full_denominator=EM.summary([new_rows[i] for i in ids]),
                                             originally_mismatched_only=EM.summary([new_rows[i] for i in mismatches]),
                                             selection_objective='box IoU' if label == 'best_box_IoU' else 'PCK10 with original box-match gate',
                                             matched_recoveries=sum(new_rows[i]['matched'] for i in mismatches))
        public[arm] = summary; private[arm] = alternatives
    return dict(**FLAGS, arms=public, inference_runs=0, normal_matched_frames_changed=0,
                scope='Only 8 previously mismatched Plastic frames; teacher alternative unselected outputs retain their actual stored values; no GT-box recrop or new inference'), private


def score():
    dst = C.DOC / 'ORACLE_COORDINATE_RESULTS.json'
    protocol_path = C.DOC / 'ORACLE_COORDINATE_PROTOCOL.json'
    protocol = C.read(protocol_path)
    for binding in protocol['inputs']:
        C.verify(binding)
    if dst.exists():
        print('COORDINATE_RESULTS_ALREADY_FROZEN')
        return
    start = time.perf_counter(), time.process_time()
    # This new scoring operation reads coordinates only after the frozen contract.
    truth = C.read(C.P.TRUTH)
    materials = {}; raw_details = {}; parity = {}
    for material, arms in ARMS.items():
        predictions = C.read(paths(material) / 'PREDICTIONS.json')
        points, sym, count = legacy(material, predictions, truth)
        groups, choices = {}, {}
        for name, pp in group_points(points).items():
            groups[name], choices[name] = coordinate_summary(pp, arms)
        pose, pose_choices = pose_summary(material)
        detection, detection_private = detection_diagnostic(material, predictions, truth)
        materials[material] = dict(reference='LEGACY_GEOMETRY_DERIVED_MIXED_PROVENANCE',
                                   strict_fixed_ID=groups, legacy_whole_symmetry_supplement=sym,
                                   whole_production_pose_expert_oracle=pose,
                                   conditional_detection_oracle=detection,
                                   verified_visible='NA_REFERENCE_NOT_VERIFIED' if material == 'WOOD' else 'Separate PLASTIC_VERIFIED66 block')
        raw_details[material] = dict(points=points, choices=choices, whole_production_pose_choices=pose_choices,
                                    conditional_detection_choices=detection_private)
        parity[material] = count
        if material == 'PLASTIC':
            anchor = verified(predictions, truth)
            anchor_groups = {}; anchor_choices = {}
            for name, pp in group_points(anchor).items():
                anchor_groups[name], anchor_choices[name] = coordinate_summary(pp, arms)
            materials['PLASTIC_VERIFIED66'] = dict(reference='FINAL_V2 DIRECT_VISIBLE; fixed native identity, no box gate', groups=anchor_groups)
            raw_details['PLASTIC_VERIFIED66'] = dict(points=anchor, choices=anchor_choices)
    result = dict(**FLAGS, protocol=C.bind(protocol_path), materials=materials,
                  numeric_parity=dict(legacy_fixed_ID_point_comparisons=parity, verified_point_comparisons=264,
                                      legacy_whole_symmetry_recomputed=True, denominator_match=True),
                  uncertainty='Descriptive repeated DEV; recording blocks retained, no corner-independent hypothesis tests',
                  timing=dict(wall_seconds=time.perf_counter()-start[0], cpu_seconds=time.process_time()-start[1], gpu_seconds=0, fits=0, optimizer_updates=0),
                  limitations=['No oracle decision, point weights, or coordinates are consumed by production training or inference.',
                               'Per-point mixtures need not form a projected rigid cuboid; they are not sent to PnP.',
                               'Local move bound ignores image evidence and rigidity; it is not learnability.',
                               'Teacher-only-correct DEV points were not student training points.',
                               'Coordinate expert, pose expert, WD candidate and local-radius gaps cannot be added.'])
    for binding in protocol['inputs']:
        C.verify(binding)
    C.save(RAW / 'POINT_ERRORS_AND_ORACLE_CHOICES_PRIVATE.json', dict(**FLAGS, materials=raw_details), True)
    C.save(dst, result, True)
    report(result)
    print('COORDINATE_ORACLE_COMPLETE', {m: result['materials'][m]['whole_production_pose_expert_oracle']['whole_output_oracle_AUC'] for m in ARMS})


def report(result):
    rows = []
    for material in ARMS:
        g = result['materials'][material]['strict_fixed_ID']['ALL']
        ref = ARMS[material][-1]
        for name, s in [(a, g['baselines'][a]) for a in ARMS[material]] + [('whole-output PCK10 oracle', g['whole_output_by_objective']['10']), ('per-point fixed-ID oracle', g['per_point_fixed_identity'])]:
            p = s['PCK']['10']
            rows.append([material, name, f"{p['correct']}/{p['total']}", f"{100*p['fraction']:.3f}", f"{s['median_px']:.3f}", f"{s['P90_px']:.3f}"])
    text = '# 현재 frozen pool의 좌표·whole-output oracle\n\n'
    text += '[확인] R0 / 같은 Replay9장38점 teacher / 현재320-update RAW / REF만 사용했다. 신규 fit·GPU inference·평가 기반 target 생성은0이다. 아래 모든 oracle는 GT_DEPENDENT / DIAGNOSTIC_ONLY이며 배포 성능이 아니다. raw 선택/오류는 private data namespace에만 저장했다.\n\n'
    text += '## Legacy reference의 strict native identity\n\n'
    text += '같은 corner0..7 ID끼리 비교하고 original selected-detection match gate와 전체 분모를 유지했다. teacher의 좋은 점을 다른 번호로 붙이거나 점별 symmetry를 고르지 않았다. 아래 median/P90은 실패 penalty 포함 전체점이며 기존 main의 matched-only median/P90과 구분한다. 기존 whole-object symmetry 지표는 JSON의 별도 supplement다.\n\n'
    text += C.table(['Material', 'output / oracle', 'PCK10', '%', 'full median px', 'full P90 px'], rows)
    text += '\n## 검수66점\n\n'
    g = result['materials']['PLASTIC_VERIFIED66']['groups']['ALL']
    rows = []
    for name, s in [(a, g['baselines'][a]) for a in ARMS['PLASTIC']] + [('whole-output PCK10 oracle', g['whole_output_by_objective']['10']), ('per-point fixed-ID oracle', g['per_point_fixed_identity'])]:
        rows.append([name, f"{s['PCK']['10']['correct']}/66", f"{s['median_px']:.3f}", f"{s['P90_px']:.3f}"])
    text += C.table(['output / oracle', 'PCK10', 'median px', 'P90 px'], rows)
    text += '\n16 reused DEV frames의 직접 검수점이며 reference selection/PnP-assisted first pass 이력이 있다. 학생TRAIN217의 정답으로 부르지 않는다. Wood strict verified는 NA_REFERENCE_NOT_VERIFIED다.\n\n'
    cross = []
    for arm, tt in g['teacher_student_cross_tabs'].items():
        for threshold, r in tt.items():
            cross.append([arm, threshold, r['both_correct'], r['teacher_only'], r['student_only'], r['both_wrong']])
    text += C.table(['student/reference output', 'px', '둘다정확', '교사만', '학생만', '둘다오류'], cross)
    text += '\n이 교사만정확한 DEV점들이 학생학습에 들어갔는데도 못배웠다는 의미는 아니다. recording별 같은 교차표와 raw 오류를 함께 보존했다.\n\n## Whole production-pose expert oracle\n\n'
    pp = []
    for material in ARMS:
        p = result['materials'][material]['whole_production_pose_expert_oracle']; ref = ARMS[material][-1]
        pp.append([material, f"{p['baselines'][ref]['ADDsym_AUC']:.8f}", f"{p['whole_output_oracle_AUC']:.8f}", f"{p['gaps'][ref]:.8f}", f"{p['available']}/{p['frames']}"])
    text += C.table(['material', 'REF production AUC', 'whole-pose oracle AUC', 'gap', 'coverage'], pp)
    text += '\n각 모델이 원래 선택한 D9 pose 하나씩, frame마다 하나만 선택했다. W/D 대안 후보의 union oracle와 다르다. ADD 목적의 GT선택이며 PCK/R/yaw/axis 동시최적이 아니다. 기존 동일 normalized ADDsym 적분(0–0.1,1001 trapezoidal,실패분모포함)을 재사용했다.\n\n## 조건부 검출 후보 선택\n\n'
    dd = result['materials']['PLASTIC']['conditional_detection_oracle']; rows = []
    for arm, d in dd['arms'].items():
        for method, v in d['methods'].items():
            rows.append([arm, method, d['mismatches_with_multiple_candidates'], d['any_box_match_recoverable'],
                         f"{100*v['full_denominator']['PCK']['10']:.3f}", v['matched_recoveries']])
    text += C.table(['arm', 'GT objective', '8실패 중 복수후보', 'box회수가능', '전체985점PCK10%', '매칭회수'], rows)
    text += '\nPlastic 원래120/128 매칭의 실패8장만 저장된 검출후보를 검사했다. best-box IoU와 best-PCK10 선택을 따로 계산했고 PCK에는 원래IoU≥.5 gate·whole-object symmetry·전체분모를 유지했다. 정상매칭120장을 새로선택하지 않았고 정답crop 재추론도 없다. Wood45는 모두매칭되어 이진단NA/불필요다. 추가GT선택정보에 따른 이득을배포성능으로쓰지않는다.\n\n## 반경 내 이상적 이동\n\n'
    rr = []
    for material in ARMS:
        g = result['materials'][material]['strict_fixed_ID']['ALL']; arm = ARMS[material][-1]
        for radius, s in g['local_move_optimistic'][arm].items():
            rr.append([material, radius, f"{s['PCK']['10']['correct']}/{s['points']}", f"{s['P90_px']:.3f}"])
    text += C.table(['REF material', 'radius native px', 'optimistic PCK10', 'full P90 px'], rr)
    text += '\n사전에고정한 descriptive r=0/2/4/8/12px, max(e−r,0)이다. 결측/box mismatch에는 적용하지 않았다. 모든 점을GT방향으로 움직일 수 있다고 가정하므로 edge/color cue로회수가능하거나 rigid pose가된다는 증거가아니다. teacher학생선택·후보선택·국소이동 gap을더하지않는다.\n\n'
    text += '검증: 기존 nativefixed-ID와whole-symmetry 수치재현, verified66 same teacher/input parity, whole-output≤per-point 및 기존arm≤oracle, arm순서/tie 불변, missing분모·radius불변을 검사했다. 자세한 난도/recording/임계5·10·20·손익은 [ORACLE_COORDINATE_RESULTS.json](ORACLE_COORDINATE_RESULTS.json), 입력과 고정 규칙은 [ORACLE_COORDINATE_PROTOCOL.json](ORACLE_COORDINATE_PROTOCOL.json)에 있다.\n'
    C.save(C.DOC / 'ORACLE_COORDINATE_REPORT_KO.md', text, True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'score'])
    {'freeze': freeze, 'score': score}[parser.parse_args().action]()
