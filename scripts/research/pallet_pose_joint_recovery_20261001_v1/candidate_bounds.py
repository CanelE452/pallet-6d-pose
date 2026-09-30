"""Pool-limited impossibility bounds; all GT choices are diagnostic only.

For each scalar error, its pointwise minimum over the frozen candidates is a
lower bound for any selector that keeps coordinates and boxes unchanged.
Each oracle still selects one complete pose; T-best and R-best are separate.
No oracle prediction is exported for runtime use.
"""
import time
from pathlib import Path
import numpy as np
import torch

from .selector_attribution import ROOT, DOC, PREV, RAW, read, bind, save, verify
from scripts.research.pallet_pose_stable_improvement_20261001_v1 import evaluate as E


def main():
    start = time.monotonic()
    torch.set_num_threads(2)
    original_result = bind(PREV / 'RESULTS.json')
    lock = read(PREV / 'POSE_PREDICTIONS_LOCK.json')
    for b in lock['files']:
        verify(b)
    input_lock, rows, _, protocol = E.locked_inputs()
    refs = E.reference_bindings(protocol, rows)
    assert refs == read(PREV / 'REFERENCE_BINDINGS.json')
    poses = read(ROOT / 'data/pallet/results/pallet_pose_stable_improvement_20261001_v1/POSE_CANDIDATES.json')
    op_path = ROOT / 'data/pallet/results/pallet_pose_stable_improvement_20261001_v1/POSE_METRICS.json'
    operational = read(op_path)
    primary = read(PREV / 'RESULTS.json')
    for b in primary['artifacts']:
        E.verify_file(b)
    _, gt = E.C.D.O.D.Pose.metadata('REAL_DEV')
    all_metrics = {}
    frame_rows = []
    for model in E.model_names():
        all_metrics[model] = {r['id']: [dict(name=h['name'], metric=E.C.D.metric(r['id'], h['pose'], gt[r['id']]))
            for h in poses[model][r['id']]['hypotheses']] for r in rows}
    groups = E.groups(rows)
    summaries, selected = {}, {}
    for model in E.model_names():
        selected[model] = {}
        for oracle, metric_key in [('T_best', 'translation_cm'), ('R_best', 'rotation_deg')]:
            chosen = {}
            for row in rows:
                fid = row['id']
                pool = [h for h in all_metrics[model][fid] if h['metric']['available']]
                h = min(pool, key=lambda h:(h['metric'][metric_key],h['name'])) if pool else None
                chosen[fid] = h['metric'] if h else dict(id=fid, available=False)
                if operational[model][fid]['available']:
                    assert h is not None
                    assert h['metric'][metric_key] <= operational[model][fid][metric_key] + 1e-7
                frame_rows.append(dict(model=model,id=fid,recording=row['recording'],oracle=oracle,
                    diagnostic_only=True,selected_whole_pose=h['name'] if h else None,
                    T_cm=h['metric'].get('translation_cm') if h else None,
                    R_deg=h['metric'].get('rotation_deg') if h else None))
            selected[model][oracle] = chosen
        summaries[model] = {pop:{oracle:E.C.D.summarize(metrics[i] for i in ids)
            for oracle,metrics in selected[model].items()} for pop,ids in groups.items() if pop in ('NATURAL99','CLEAN29','WOOD45')}
    # Independent parity with the previously published four-model oracle audit.
    old = read(ROOT / '_docs/experiments/pallet_pose_diagnosis_20260930_v1/E2_ORACLE_SUMMARY.json')
    parity_count = 0
    for model, oldname in [('R0','identity'),('PRIOR1','PRIOR1'),('FULL125','FULL125')]:
        for pop in ('NATURAL99','CLEAN29'):
            for oracle in ('T_best','R_best'):
                for metric in E.METRICS:
                    for statistic in ('median','P90'):
                        a = summaries[model][pop][oracle]['conditional'][metric][statistic]
                        b = old['models'][oldname][pop][oracle]['summary']['conditional'][metric][statistic]
                        assert np.isclose(a,b,atol=1e-7,rtol=0.), (model,pop,oracle,metric,statistic,a,b)
                        parity_count += 1
    r0_clean = E.C.D.summarize(operational['R0'][i] for i in groups['CLEAN29'])
    r0_nat = E.C.D.summarize(operational['R0'][i] for i in groups['NATURAL99'])
    clean_limit = r0_clean['conditional']['translation_cm']['median'] * E.GUARD_RATIO
    tail_limit = r0_nat['conditional']['translation_cm']['P90'] * E.GUARD_RATIO
    def bound(models):
        clean = float(np.mean([summaries[m]['CLEAN29']['T_best']['conditional']['translation_cm']['median'] for m in models]))
        tail = float(np.mean([summaries[m]['NATURAL99']['T_best']['conditional']['translation_cm']['P90'] for m in models]))
        return dict(models=models,clean_T_median_lower_bound_cm=clean,clean_T_limit_cm=clean_limit,
            clean_T_unavoidable_excess_cm=clean-clean_limit,
            clean_T_guard_impossible=clean>clean_limit,
            natural_T_P90_lower_bound_cm=tail,natural_T_P90_limit_cm=tail_limit,
            natural_T_tail_guard_impossible=tail>tail_limit,
            full_goal_excluded_with_this_fixed_pool=clean>clean_limit or tail>tail_limit)
    bounds = {m:bound([m]) for m in E.model_names()}
    bounds.update({arm+'_mean3seeds':bound([f'{arm}_s{s}' for s in (1,2,3)]) for arm in E.C.ARMS})
    csv_path = DOC / 'CANDIDATE_BOUND_ROWS.csv'
    # The previous experiment's immutable writer must not own this namespace.
    import csv
    with csv_path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(frame_rows[0]));writer.writeheader();writer.writerows(frame_rows)
    out = dict(type='Post-result exploratory pool bounds, not deployable accuracy',
        original_primary_result=original_result,reference_lock=bind(PREV/'REFERENCE_BINDINGS.json'),
        pose_lock=bind(PREV/'POSE_PREDICTIONS_LOCK.json'),codes=[bind(Path(__file__))],
        formula='For fixed candidate set C_i: min_c T(i,c) <= T(i,s(i)). Empirical median and P90, and mean of seed quantiles, preserve this componentwise ordering.',
        scope='Only changes choosing existing W/D poses, fixed coordinates, fixed selected detection, and identical metric/failure contracts. New points, detectors, solvers, or cross-model candidate unions are outside this bound.',
        no_combined_T_R_oracle=True,oracle_choices_for_runtime=False,
        old_oracle_numeric_parity_checks=parity_count,guard_ratio=E.GUARD_RATIO,
        bounds=bounds,summaries=summaries,artifacts=[bind(csv_path)],
        interpretation='A bound above a required ceiling rules out that selector-only solution. A bound below it does not establish attainability, joint feasibility, generalization, or success.',
        fits=0,image_forwards=0,changed_primary_results=0,wall_seconds=time.monotonic()-start)
    assert bind(PREV/'RESULTS.json') == original_result
    save(DOC/'CANDIDATE_BOUNDS.json',out)
    print('COMPLETE',bounds,flush=True)


if __name__ == '__main__':
    main()
