"""Merge sealed control extensions without overwriting any earlier results."""
from pathlib import Path
from . import common as C
from .robustness import run as robustness


def merge_equal(destination, source):
    for name, value in source.items():
        if name in destination:
            assert destination[name] == value, ('Conflicting historical output', name)
        else:
            destination[name] = value


def pairs_for(arms):
    pairs = set()
    for suffix in ('D9', 'GEO', 'NEWGEO'):
        for seed in (42, 43):
            for condition in ('CLEAR', 'OCC'):
                pairs.add((f'RAW_{condition}_S{seed}_{suffix}', f'REF_{condition}_S{seed}_{suffix}'))
            for target in ('RAW', 'REF'):
                pairs.add((f'{target}_CLEAR_S{seed}_{suffix}', f'{target}_OCC_S{seed}_{suffix}'))
        pairs.add((f'OLD_RAW_{suffix}', f'OLD_REF_{suffix}'))
        for arm in arms:
            if arm.endswith('_' + suffix) and not arm.startswith('R0_'):
                pairs.add((f'R0_{suffix}', arm))
        for arm in arms:
            if arm.endswith('_' + suffix) and '_S' in arm:
                pairs.add((f'OLD_REF_{suffix}', arm))
    for prefix in sorted({a.rsplit('_', 1)[0] for a in arms}):
        pairs.add((prefix + '_D9', prefix + '_GEO'))
        pairs.add((prefix + '_GEO', prefix + '_NEWGEO'))
    # Every new candidate is also compared to the fixed historical strong reference.
    for arm in arms:
        if arm != 'OLD_REF_GEO':
            pairs.add(('OLD_REF_GEO', arm))
    assert all(a in arms and b in arms and a != b for a, b in pairs)
    return sorted(pairs)


def main():
    from scripts.research.pallet_clean_to_pose_transfer_v1 import eval_student as E
    from scripts.research.pallet_pose_objective_followup_v2 import metric_baseline as M
    result_path = C.DOC / 'FINAL_RESULTS.json'
    if result_path.exists():
        for b in C.read(result_path)['inputs'] + C.read(result_path)['private_artifacts']:
            C.verify(b)
        print('FINAL_RESULTS_REUSED'); return
    # Both sources have prediction-before-scoring locks and preserve previous outputs.
    source_sets = ('CLEAR43', 'HISTORICAL_RAW')
    metrics, poses, bindings, inputs = {}, {}, {}, []
    for tag in source_sets:
        for kind, target in (('FRAME_METRICS', metrics), ('POSES', poses), ('CASE_BINDINGS', bindings)):
            path = C.RAW / f'{tag}_{kind}_PRIVATE.json'
            merge_equal(target, C.read(path)); inputs.append(C.bind(path))
        inputs.append(C.bind(C.DOC / f'{tag}_RESULTS.json'))
    assert len(metrics) == len(poses) == len(bindings) == 33
    metadata = C.OLD.RAW / 'evaluation/S42/METADATA.json'
    rows = C.read(metadata); groups = E.group_ids(rows)
    assert all(set(data) == set(groups['FULL128']) for data in metrics.values())
    assert all(set(data) == set(groups['FULL128']) for data in poses.values())
    pairs = pairs_for(metrics)
    summaries = {g: {a: M.summarize(v[i] for i in ids) for a, v in metrics.items()} for g, ids in groups.items()}
    paired = {g: {b + '-minus-' + a: M.paired(metrics[a], metrics[b], ids) for a, b in pairs} for g, ids in groups.items()}
    natural = [r for r in rows if r['severity'] != 'CLEAN']
    loro = {rec: {b + '-minus-' + a: M.paired(metrics[a], metrics[b], [r['id'] for r in natural if r['recording'] != rec])
                  for a, b in pairs} for rec in sorted({r['recording'] for r in natural})}
    private = []
    for kind, value in (('FRAME_METRICS', metrics), ('POSES', poses), ('CASE_BINDINGS', bindings)):
        p = C.RAW / f'FINAL_{kind}_PRIVATE.json'; C.save(p, value, True); private.append(C.bind(p))
    C.save(C.DOC / 'FINAL_PAIRS.json', pairs, True)
    inputs += [C.bind(metadata), C.bind(Path(__file__)), C.bind(C.DOC / 'FINAL_PAIRS.json')]
    C.save(result_path, M.clean(dict(created_at=C.now(), available_models=list(metrics), groups=summaries,
        paired=paired, leave_one_recording_out_NATURAL99=loro, inputs=inputs, private_artifacts=private,
        same_evaluation=True, reference='Repeated DEV / geometry-derived 6D; no independent physical truth',
        repeat_scope='Actual paired data/augmentation streams42/43 for CLEAR and OCC; same R0 weights, not independent initializations',
        no_new_fit_in_analysis=True, no_oracle_selection=True, no_best_seed_or_selector_mix=True)), True)
    print('FINAL_RESULTS_COMPLETE', len(metrics), len(pairs), flush=True)
    robust_pairs = [('R0_GEO', 'OLD_REF_GEO'), ('OLD_RAW_GEO', 'OLD_REF_GEO'),
                    ('OLD_REF_GEO', 'R0_NEWGEO')]
    for seed in (42, 43):
        for selector in ('GEO', 'NEWGEO'):
            for condition in ('CLEAR', 'OCC'):
                robust_pairs.append((f'RAW_{condition}_S{seed}_{selector}', f'REF_{condition}_S{seed}_{selector}'))
            robust_pairs.append((f'REF_CLEAR_S{seed}_{selector}', f'REF_OCC_S{seed}_{selector}'))
            robust_pairs.append(('OLD_REF_GEO', f'REF_CLEAR_S{seed}_{selector}'))
            robust_pairs.append(('OLD_REF_GEO', f'REF_OCC_S{seed}_{selector}'))
    rp = C.DOC / 'FINAL_ROBUSTNESS_PAIRS.json'; C.save(rp, robust_pairs, True)
    robustness(C.RAW / 'FINAL_FRAME_METRICS_PRIVATE.json', metadata, rp, 'FINAL_ROBUSTNESS.json')


if __name__ == '__main__':
    main()
