"""Saved-row matched 2D/6D target analysis; no model, pose solver or manuscript I/O."""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from .baseline import BASELINE_ROOT

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = Path('_docs/experiments/pallet_pose_target_6d_20261006_v1')
DOC = ROOT / NAMESPACE
OLD = BASELINE_ROOT / '_docs/experiments/pallet_joint_action_handoff_20261006_v1'
SPEC = importlib.util.spec_from_file_location('pose_target_existing_eval_math', BASELINE_ROOT / 'scripts/research/pallet_dim_conditioned_p_v1/eval_math.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
METRICS = ['translation_cm', 'rotation_deg', 'ADDsym_m']
TOLERANCES = dict(translation_cm=1e-9, rotation_deg=1e-9, ADDsym_m=1e-7)
BOOTSTRAP_SEED = 20260917
RESAMPLES = 10000


def configure(root: Path):
    global ROOT, DOC, OLD
    ROOT = Path(root).resolve()
    DOC = ROOT / NAMESPACE
    OLD = BASELINE_ROOT / '_docs/experiments/pallet_joint_action_handoff_20261006_v1'
    if DOC.resolve().relative_to(ROOT) != NAMESPACE:
        raise ValueError('new result namespace must not be redirected')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def binding(path):
    path=Path(path)
    return dict(path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), sha256=sha(path), bytes=path.stat().st_size)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, 'tolist'):
        return clean(value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write(path, value):
    path = Path(path)
    if not path.resolve().is_relative_to(DOC.resolve()):
        raise ValueError('report writes must remain in the new namespace')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def ordered(rows, ids):
    lookup = {r['id']: r for r in rows}
    if len(lookup) != len(rows) or set(lookup) != set(ids) or len(set(ids)) != len(ids):
        raise ValueError('duplicate, missing or unexpected row IDs')
    return [lookup[i] for i in ids]


def load_methods(split, seeds):
    manifest_path = OLD / 'results/A_ID_MANIFEST.json'
    ids = read(manifest_path)['IDs']['synthetic_evaluation' if split == 'SYNTH_HELDOUT' else 'real_evaluation']
    if len(ids) != (1985 if split == 'SYNTH_HELDOUT' else 319):
        raise ValueError('full evaluation registry changed')
    base_path = OLD / f'results/A_{split}_BASELINES.json'
    baseline = read(base_path)
    methods = {name: ordered(rows, ids) for name, rows in baseline['rows'].items()}
    files = [manifest_path, base_path]
    for seed in (1, 2, 3):
        path = OLD / f'results/A_{split}_FIT_GEO_seed{seed}.json'
        old = read(path)
        assert old['trained'] and old['GT_inference_access'] is False
        assert old['bank_binding'] == baseline['bank_binding']
        methods[f'FIT_GEO_J_seed{seed}'] = ordered(old['rows']['GEO_J'], ids)
        files.append(path)
    oracle_path = OLD / f'results/A_{split}_ORACLE.json'
    corner_path = OLD / f'results/A_{split}_ORACLE_CORNERS.json'
    oracle_payload = read(oracle_path)
    assert oracle_payload['bank_binding'] == baseline['bank_binding']
    oracle = ordered([r for r in oracle_payload['rows'] if r['arm'] == 'GEO'], ids)
    corners = ordered([r for r in read(corner_path)['rows'] if r['arm'] == 'GEO'], ids)
    methods['GEO_6D_ORACLE'] = []
    for row, corner in zip(oracle, corners):
        assert row['index'] == corner['selected_index'] and row['oracle'] == corner['oracle_pose']
        methods['GEO_6D_ORACLE'].append(dict(id=row['id'], session=row['session'], corner=corner['oracle_corner'],
            pose=row['oracle'], selected_index=row['index'], final_hypothesis=row['final_hypothesis'],
            generating_hypothesis=row['generating_hypothesis'], action_count=row['actions']))
    files.extend([oracle_path, corner_path])
    for seed in seeds:
        path = DOC / f'results/{split}_POSE_TARGET_GEO_J_seed{seed}.json'
        result = read(path)
        assert result['status']=='DONE' and result['complete'] and result['readout']=='unchanged J'
        assert result['full_denominator'] == len(ids) and result['GT_inference_access'] is False
        assert result['bank_binding'] == baseline['bank_binding'], 'OLD/NEW bank binding differs'
        assert result['target_sha256']==baseline['target_sha256'], 'OLD/NEW evaluation GT target differs'
        if split=='SYNTH_HELDOUT':
            expected_bank=next(x['sha256'] for x in read(OLD/'A_manifest.json')['cache_files'] if x['name']=='source_banks.npy')
        else:
            expected_bank=next(x['sha256'] for x in read(OLD/'results/A_SAMPLING_MASK_RECEIPT.json')['external_mask_files'] if Path(x['path']).name=='REAL_DEV_generated_banks.npz')
        assert result['bank_artifact']['sha256']==expected_bank, 'OLD/NEW native evaluation bank bytes differ'
        fit=read(DOC/f'fits/POSE_TARGET_GEO_seed{seed}.json')
        assert fit['complete'] and fit['updates']==6000 and fit['final_checkpoint_only']
        assert result['checkpoint_sha256']==fit['checkpoint_sha256'], 'prediction/checkpoint mismatch'
        rows = ordered(result['rows'], ids)
        for row, bound in zip(rows, oracle):
            assert isinstance(row['selected_index'], int) and 0 <= row['selected_index'] < bound['actions']
            if 'action_count' in row:
                assert row['action_count'] == bound['actions']
            if 'oracle_selected_index' in row:
                assert row['oracle_selected_index'] == bound['index']
        methods[f'POSE_TARGET_GEO_J_seed{seed}'] = rows
        files.append(path)
    for rows in methods.values():
        for row, raw in zip(rows, methods['RAW']):
            assert row['session'] == raw['session'], 'original scenario/session mismatch'
    return ids, methods, oracle, [binding(p) for p in files]


def summarize(rows):
    corner = M.summary([r['corner'] for r in rows])
    available = [r['pose'] for r in rows if r['pose']['available']]
    pose = dict(total_frames=len(rows), available=len(available), failures=len(rows)-len(available),
        coverage=len(available)/len(rows) if rows else None)
    for key in METRICS:
        values = [r[key] for r in available]
        assert all(np.isfinite(values)), 'available pose has nonfinite metric'
        pose[key] = distribution(values)
    has_indices = any(r.get('selected_index') is not None for r in rows)
    return dict(corner=corner, pose=pose,
        NoOp=sum(r.get('selected_index') == 0 for r in rows) if has_indices else None,
        selected_action_distribution=dict(collections.Counter(str(r.get('selected_index')) for r in rows)) if has_indices else None,
        NoOp_scope='GEO bank index0; baseline N3/PoseFix has no GEO action index',
        final_hypothesis_counts=dict(collections.Counter(r.get('final_hypothesis') or 'UNAVAILABLE' for r in rows)))


def distribution(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    return dict(count=len(a), mean=float(a.mean()) if len(a) else None,
        median=float(np.median(a)) if len(a) else None, P90=float(np.quantile(a, .9)) if len(a) else None,
        min=float(a.min()) if len(a) else None, max=float(a.max()) if len(a) else None,
        percentiles={str(p): float(np.quantile(a, p/100)) for p in (1, 5, 25, 50, 75, 95, 99)} if len(a) else {})


class SharedBootstrap:
    """Same original-ID multinomial draws for every method, seed and valid mask."""
    def __init__(self, ids, groups=None, resamples=RESAMPLES):
        self.ids = list(ids)
        assert len(ids) > 0 and len(ids) == len(set(ids))
        self.units, self.inverse = np.unique(ids if groups is None else groups, return_inverse=True)
        if groups is None:
            self.units = np.asarray(ids)
            self.inverse = np.arange(len(ids))
        assert len(self.inverse) == len(ids)
        self.level = 'frame' if groups is None else 'original_scenario_or_session'
        self.counts = np.bincount(self.inverse, minlength=len(self.units))
        self.resamples = resamples
        n = len(self.units)
        weights = np.random.default_rng(BOOTSTRAP_SEED).multinomial(n, np.full(n, 1/n), size=resamples).astype('uint16')
        self.draw_sha256 = hashlib.sha256(weights.tobytes()).hexdigest()
        self.weights = weights.astype(float)
        self.denominators = {}

    def meta(self):
        return dict(level=self.level, units=len(self.units), full_master_frames=len(self.ids), seed=BOOTSTRAP_SEED,
            resamples=self.resamples, draw_sha256=self.draw_sha256,
            group_size_histogram=dict(collections.Counter(str(c) for c in self.counts)),
            group_sizes=dict(zip(self.units.tolist(), self.counts.tolist())),
            weighting='uniform original-unit draws; retain every frame of each drawn group; pooled eligible-frame mean; all seed/method draws shared',
            multiplicity_adjusted=False)

    def contrast(self, delta):
        d = np.asarray(delta, float)
        assert d.shape == (len(self.ids),)
        valid = np.isfinite(d)
        count = np.bincount(self.inverse, weights=valid.astype(float), minlength=len(self.units))
        total = np.bincount(self.inverse, weights=np.where(valid, d, 0), minlength=len(self.units))
        key = valid.tobytes()
        if key not in self.denominators:
            self.denominators[key] = self.weights @ count
        denominator = self.denominators[key]
        numerator = self.weights @ total
        keep = denominator > 0
        samples = numerator[keep]/denominator[keep]
        return dict(mean_paired_difference=float(d[valid].mean()) if valid.any() else None,
            median_paired_difference=float(np.median(d[valid])) if valid.any() else None,
            CI95=np.quantile(samples, [.025, .975]).tolist() if len(samples) and np.count_nonzero(count)>1 else None,
            full_master_frames=len(d), common_eligible_frames=int(valid.sum()), eligible_units=int(np.count_nonzero(count)),
            valid_resamples=int(keep.sum()), empty_resamples=int((~keep).sum()), draw_sha256=self.draw_sha256,
            seed=BOOTSTRAP_SEED, resamples=self.resamples,
            scope='paired same-ID repeated-use development diagnostic; no independent confirmation or multiplicity correction')


def coverage(new, base):
    assert [r['id'] for r in new] == [r['id'] for r in base]
    a = np.array([r['pose']['available'] for r in new], bool)
    b = np.array([r['pose']['available'] for r in base], bool)
    return dict(full_frames=len(new), common_success=int((a & b).sum()), new_only_success=int((a & ~b).sum()),
        base_only_success=int((~a & b).sum()), both_failed=int((~a & ~b).sum()))


def delta_vector(new, base, kind, key):
    assert [r['id'] for r in new] == [r['id'] for r in base]
    result = []
    for a, b in zip(new, base):
        valid = a[kind]['available' if kind == 'pose' else 'evaluable'] and b[kind]['available' if kind == 'pose' else 'evaluable']
        result.append(a[kind][key] - b[kind][key] if valid else np.nan)
    return np.asarray(result, float)


def compare(new, base, bootstraps):
    result = dict(coverage=coverage(new, base), corner_damage=M.damage([r['corner'] for r in base], [r['corner'] for r in new]), statistics={})
    for kind, key in [('corner', 'E_sym'), *[('pose', k) for k in METRICS]]:
        delta = delta_vector(new, base, kind, key)
        result['statistics'][key] = {name: bootstrap.contrast(delta) for name, bootstrap in bootstraps.items()}
    return result


def seed_mean_compare(news, bases, bootstraps):
    assert len(news) == len(bases) and len(news) in (1, 3)
    result = dict(seeds=len(news), statistics={}, aggregation='per-ID mean of paired differences across executed fixed seeds; common metric eligibility across those seeds; same original-ID bootstrap')
    for kind, key in [('corner', 'E_sym'), *[('pose', k) for k in METRICS]]:
        matrix = np.asarray([delta_vector(a, b, kind, key) for a, b in zip(news, bases)])
        valid = np.isfinite(matrix).all(0)
        delta = np.where(valid, np.where(np.isfinite(matrix), matrix, 0).mean(0), np.nan)
        result['statistics'][key] = {name: bootstrap.contrast(delta) for name, bootstrap in bootstraps.items()}
    return result


def face_state(raw, other):
    if not raw['pose']['available'] or not other['pose']['available'] or not raw.get('final_hypothesis') or not other.get('final_hypothesis'):
        return 'UNAVAILABLE'
    return 'SWITCH' if raw['final_hypothesis'] != other['final_hypothesis'] else 'NO_SWITCH'


def recovery(raw, old, new, oracle, tolerance=1e-7):
    assert all([r['id'] for r in rows] == [r['id'] for r in raw] for rows in (old, new, oracle))
    rows = []
    for a, b, c, o in zip(raw, old, new, oracle):
        record = dict(id=a['id'], common_success=all(r['pose']['available'] for r in (a, b, c, o)))
        if record['common_success']:
            rc, oc, nc, bound = [r['pose']['ADDsym_m'] for r in (a, b, c, o)]
            headroom, old_gain, new_gain = rc-bound, rc-oc, rc-nc
            valid_ratio = headroom > tolerance
            record.update(raw_cost_m=rc, old_cost_m=oc, new_cost_m=nc, oracle_cost_m=bound,
                headroom_m=headroom, old_recovered_m=old_gain, new_recovered_m=new_gain,
                old_gap_m=oc-bound, new_gap_m=nc-bound, ratio_eligible=valid_ratio,
                old_ratio=old_gain/headroom if valid_ratio else None, new_ratio=new_gain/headroom if valid_ratio else None)
        rows.append(record)
    valid = [r for r in rows if r['common_success']]
    ratios = [r for r in valid if r['ratio_eligible']]
    result = dict(full_frames=len(rows), common_success=len(valid), excluded_missing_or_F_failure=len(rows)-len(valid),
        ratio_eligible_frames=len(ratios), excluded_headroom_le_tolerance=len(valid)-len(ratios), tolerance_m=tolerance,
        distributions={k: distribution([r[k] for r in valid]) for k in ['headroom_m', 'old_recovered_m', 'new_recovered_m', 'old_gap_m', 'new_gap_m']},
        ratios={method: dict(distribution=distribution([r[method+'_ratio'] for r in ratios]),
            negative=sum(r[method+'_ratio']<0 for r in ratios), over_one=sum(r[method+'_ratio']>1 for r in ratios),
            oracle_bound_violations_beyond_tolerance=sum(r[method+'_gap_m'] < -tolerance for r in valid)) for method in ('old', 'new')},
        exact_oracle_action={method:dict(count=sum(o['pose']['available'] and x.get('selected_index')==o.get('selected_index') for x,o in zip(values,oracle)),
            denominator=sum(o['pose']['available'] for o in oracle),rate=sum(o['pose']['available'] and x.get('selected_index')==o.get('selected_index') for x,o in zip(values,oracle))/sum(o['pose']['available'] for o in oracle) if any(o['pose']['available'] for o in oracle) else None)
            for method,values in [('old',old),('new',new)]},
        rows=rows, scope='development diagnostic at current references and same GEO bank; ratios are not clamped; >1 may reflect numeric oracle discrepancy and is counted separately')
    return result


def hypothesis_analysis(raw, old, new, oracle):
    output = {}
    states = [face_state(a, b) for a, b in zip(raw, oracle)]
    for state in ['NO_SWITCH', 'SWITCH', 'UNAVAILABLE']:
        indices = [i for i, v in enumerate(states) if v == state]
        oo = [oracle[i] for i in indices]
        group = dict(frames=len(indices), ids=[raw[i]['id'] for i in indices], summaries={name: summarize([rows[i] for i in indices]) for name, rows in [('RAW', raw), ('OLD', old), ('NEW', new), ('ORACLE', oracle)]}, recovery={})
        for name, rows in [('OLD', old), ('NEW', new)]:
            chosen = [rows[i] for i in indices]
            denominator=sum(b['pose']['available'] for b in oo)
            exact=sum(b['pose']['available'] and a.get('selected_index')==b.get('selected_index') for a,b in zip(chosen,oo))
            group['recovery'][name] = dict(exact_oracle_action=exact,exact_oracle_action_denominator=denominator,exact_oracle_action_rate=exact/denominator if denominator else None,
                final_hypothesis_recovered=sum(a['pose']['available'] and b['pose']['available'] and a.get('final_hypothesis')==b.get('final_hypothesis') for a, b in zip(chosen, oo)),
                switches_from_RAW=sum(face_state(raw[i], rows[i])=='SWITCH' for i in indices))
        output[state] = group
    return dict(groups=output, rows=[dict(id=a['id'], raw=a.get('final_hypothesis'), old=b.get('final_hypothesis'), new=c.get('final_hypothesis'), oracle=o.get('final_hypothesis'), oracle_group=state) for a, b, c, o, state in zip(raw, old, new, oracle, states)], scope='actual final F hypothesis, never generating hypothesis or GT-selected inference rule')


def quadrant(delta_pose, delta_2d, pose_tolerance=1e-7, corner_tolerance=1e-9):
    if not np.isfinite(delta_pose) or not np.isfinite(delta_2d):
        return 'UNAVAILABLE'
    p = -1 if delta_pose < -pose_tolerance else 1 if delta_pose > pose_tolerance else 0
    c = -1 if delta_2d < -corner_tolerance else 1 if delta_2d > corner_tolerance else 0
    return {(-1, -1):'SIX_D_IMPROVES_TWO_D_IMPROVES',(-1, 1):'SIX_D_IMPROVES_TWO_D_WORSENS',(1, -1):'SIX_D_WORSENS_TWO_D_IMPROVES',(1, 1):'BOTH_WORSEN'}.get((p, c), 'HAS_NUMERIC_TIE')


def tradeoff(new, base):
    frame = []; corners = collections.Counter(); valid_corner_count = 0
    for a, b in zip(new, base):
        assert a['id'] == b['id']
        d = a['pose']['ADDsym_m']-b['pose']['ADDsym_m'] if a['pose']['available'] and b['pose']['available'] else np.nan
        e = a['corner']['frame_mean_px']-b['corner']['frame_mean_px'] if a['corner']['evaluable'] and b['corner']['evaluable'] else np.nan
        frame.append(dict(id=a['id'], ADDsym_delta_m=d, native_mean_corner_delta_px=e, quadrant=quadrant(d, e)))
        if a['corner']['evaluable'] and b['corner']['evaluable']:
            assert a['corner']['canonical_valid'] == b['corner']['canonical_valid']
            for i, valid in enumerate(a['corner']['canonical_valid']):
                if valid:
                    valid_corner_count += 1
                    corners[quadrant(d, a['corner']['canonical_errors'][i]-b['corner']['canonical_errors'][i])] += 1
    return dict(frame_counts=dict(collections.Counter(r['quadrant'] for r in frame)), canonical_corner_counts=dict(corners), canonical_corner_denominator=valid_corner_count,
        corner_damage=M.damage([r['corner'] for r in base], [r['corner'] for r in new]), frame_rows=frame,
        scope='6D means same-frame ADDsym; 2D means native evaluation mean-L2 or same canonical GT corner; separate T/R metrics also reported; neither alone proves whole sensing improvement')


def screening_decision(comparison, raw_old_damage, raw_new_damage, old_summary, new_summary):
    """Pre-result root operationalization; accepts synthetic-only statistics."""
    stats = {k: comparison['statistics'][k]['frame'] for k in METRICS}
    means = {k: stats[k]['mean_paired_difference'] for k in METRICS}
    if any(v is None for v in means.values()):
        return dict(decision='STOP_MIXED', continue_seeds=False, reason='missing common-success pose metrics; no added fits justified')
    worsening = all(means[k] > TOLERANCES[k] for k in METRICS)
    improving = [k for k in METRICS if means[k] < -TOLERANCES[k]]
    opposing = [k for k in METRICS if means[k] > TOLERANCES[k]]
    supported = [k for k in improving if stats[k]['CI95'] is not None and stats[k]['CI95'][1] < 0]
    catastrophic_increase = raw_new_damage['good5_to_bad10'] > raw_old_damage['good5_to_bad10']
    failure_increase = new_summary['pose']['failures'] > old_summary['pose']['failures']
    gross_increase = new_summary['corner']['gross20'] > old_summary['corner']['gross20']
    safe = not (catastrophic_increase or failure_increase or gross_increase)
    if worsening and catastrophic_increase:
        decision, run, reason = 'STOP', False, 'all three paired pose means worsen and RAW canonical good5-to-bad10 damage increases'
    elif supported and safe:
        decision, run, reason = 'CONTINUE', True, 'at least one improved paired pose mean has frame CI upper<0; failures and both fixed damage controls do not increase'
    elif improving and opposing and safe:
        decision, run, reason = 'MIXED_CONTINUE', True, 'confirm directional reproducibility of unchanged hard ADD target across three fixed seeds; at least one mean improves and failure/damage controls do not increase'
    else:
        decision, run, reason = 'STOP_MIXED', False, 'mixed direction or uncertainty with no preregistered safe continuation basis; no additional fits'
    return dict(decision=decision, continue_seeds=run, reason=reason, paired_means_NEW_minus_OLD=means,
        improved_pose_metrics=improving, negative_frame_CI_metrics=supported,
        opposing_pose_metrics=opposing,
        failure_increase=failure_increase, canonical_catastrophic_increase=catastrophic_increase, gross20_increase=gross_increase,
        root_operationalization_frozen_before_metrics=True, criterion_population='SYNTH_HELDOUT_ONLY', REAL_used=False,
        numerical_tolerances=TOLERANCES, arbitrary_percentage_thresholds=False, hyperparameter_changes=False)


def analyze_split(split, seeds):
    ids, methods, oracle, inputs = load_methods(split, seeds)
    raw = methods['RAW']
    secondary_name = 'scenario' if split == 'SYNTH_HELDOUT' else 'session'
    bootstraps = dict(frame=SharedBootstrap(ids))
    bootstraps[secondary_name] = SharedBootstrap(ids, [r['session'] for r in raw])
    summaries = {name: summarize(rows) for name, rows in methods.items()}
    comparisons = {}; recoveries = {}; hypotheses = {}; tradeoffs = {}
    for seed in seeds:
        new_name = f'POSE_TARGET_GEO_J_seed{seed}'
        old_name = f'FIT_GEO_J_seed{seed}'
        new = methods[new_name]
        for base in [old_name, f'N3_seed{seed}', f'PoseFix_seed{seed}', 'RAW', 'GEO_6D_ORACLE']:
            comparisons[f'{new_name}_minus_{base}'] = compare(new, methods[base], bootstraps)
            tradeoffs[f'{new_name}_minus_{base}'] = tradeoff(new, methods[base])
        recoveries[str(seed)] = recovery(raw, methods[old_name], new, methods['GEO_6D_ORACLE'])
        hypotheses[str(seed)] = hypothesis_analysis(raw, methods[old_name], new, methods['GEO_6D_ORACLE'])
    seedmeans = {}
    for family in ['FIT_GEO_J', 'N3', 'PoseFix', 'RAW']:
        seedmeans[f'POSE_TARGET_GEO_J_minus_{family}'] = seed_mean_compare([methods[f'POSE_TARGET_GEO_J_seed{s}'] for s in seeds],
            [methods[family if family == 'RAW' else f'{family}_seed{s}'] for s in seeds], bootstraps)
    result = dict(schema='pose_target_6d_saved_row_results_v1', status='DONE', split=split, full_denominator=len(ids),
        executed_new_seeds=list(seeds), IDs=ids, summaries=summaries, comparisons=comparisons, seed_mean_comparisons=seedmeans,
        bootstrap={name: value.meta() for name, value in bootstraps.items()}, source_bindings=inputs,
        original_eval_math_sha256=sha(BASELINE_ROOT/'scripts/research/pallet_dim_conditioned_p_v1/eval_math.py'),
        damage_vs_RAW={name:M.damage([r['corner'] for r in raw],[r['corner'] for r in rows]) for name,rows in methods.items()},
        scope='repeated-use synthetic development diagnostic' if split == 'SYNTH_HELDOUT' else '319 repeated REAL_DEV / 13 sessions; same2D+geometry reconstructed reference; exploratory, no decision or independent metrology',
        new_model_calls=0, new_F_calls=0, new_optimizer_updates=0)
    if split == 'REAL_DEV':
        assert len(bootstraps['session'].units) == 13
        result['matching_groups'] = dict(matched=sum(r['corner']['matched'] for r in raw), unmatched=sum(not r['corner']['matched'] for r in raw), pose_total=len(raw))
    return result, recoveries, hypotheses, tradeoffs


def screen(root=ROOT):
    configure(root)
    begin=time.monotonic()
    result, recovery_result, hypothesis_result, tradeoff_result = analyze_split('SYNTH_HELDOUT', [1])
    old = result['summaries']['FIT_GEO_J_seed1']; new = result['summaries']['POSE_TARGET_GEO_J_seed1']
    raw_old = result['damage_vs_RAW']['FIT_GEO_J_seed1']
    raw_new = result['comparisons']['POSE_TARGET_GEO_J_seed1_minus_RAW']['corner_damage']
    decision = screening_decision(result['comparisons']['POSE_TARGET_GEO_J_seed1_minus_FIT_GEO_J_seed1'], raw_old, raw_new, old, new)
    decision.update(schema='pose_target_seed1_screen_v1', source_bindings=result['source_bindings'], paired_result=result['comparisons']['POSE_TARGET_GEO_J_seed1_minus_FIT_GEO_J_seed1'], old_summary=old, new_summary=new, RAW_old_damage=raw_old, RAW_new_damage=raw_new)
    write(DOC/'SEED1_SCREENING.json', decision)
    write(DOC/'SYNTH_RESULTS.json', result)
    write(DOC/'SEED1_SCREEN_EXECUTION.json',dict(status='DONE',code_sha256=sha(Path(__file__)),seconds_wall=time.monotonic()-begin,
        source_bindings=result['source_bindings'],read_populations=['SYNTH_HELDOUT'],REAL_read=False,new_model_calls=0,new_F_calls=0,new_optimizer_updates=0))
    return decision


def matched_comparison(seeds):
    from . import training as training_module,evaluation as evaluation_module
    from scripts.research.pallet_joint_action_handoff_20261006_v1.a_data import Data as original_data
    from scripts.research.pallet_joint_action_handoff_20261006_v1.scorer import action_scores as original_scores,decode_bank as original_decode
    protocol=read(OLD/'A_protocol.json')
    manifest=read(OLD/'A_manifest.json')
    expected_bank=next(r['sha256'] for r in manifest['cache_files'] if r['name']=='source_banks.npy')
    preflight=read(DOC/'PREFLIGHT.json')
    preflight_current=preflight['status']=='PASS' and preflight['input_bindings_sha256']==sha(DOC/'INPUT_BINDINGS.json') and preflight['protocol_sha256']==sha(DOC/'PROTOCOL.json')
    rows = []
    for seed in seeds:
        old_path = OLD/f'A_fits/GEO_seed{seed}.json'
        new_path = DOC/f'fits/POSE_TARGET_GEO_seed{seed}.json'
        old, new = read(old_path), read(new_path)
        contract = new['matched_contract']
        checks=dict(candidate_bank=contract['candidate_bank_sha256']==expected_bank,
            bank_binding=contract['bank_binding']==manifest['bank_binding'],
            architecture=contract['architecture_sha256']==sha(BASELINE_ROOT/'scripts/research/pallet_joint_action_handoff_20261006_v1/scorer.py'),
            parameters=contract['parameter_count']==old['params']==20259,
            config=contract['config']==protocol['config'],
            optimizer=contract['optimizer']=={k:protocol['optimizer'][k] for k in ['name','lr','weight_decay','betas']},
            schedule=contract['schedule']==dict(warmup=protocol['optimizer']['warmup_steps'],cosine_steps=5900,final_lr_fraction=protocol['optimizer']['cosine_final_lr_fraction']),
            gradient_clip=contract['gradient_clipping']==protocol['optimizer']['gradient_clip_norm'],
            batch=contract['batch']==protocol['batch']==16,updates=contract['updates']==new['updates']==old['updates']==6000,
            initializer=contract['initial_state_sha256']==old['first_step']['initial_state_sha256'],
            order=contract['order_sha256']==old['order_sha256'],
            numeric=contract['numeric']['TF32_matmul']==old['TF32_matmul'] and contract['numeric']['TF32_cudnn']==old['TF32_cudnn'] and contract['numeric']['dtype']=='FP32' and contract['numeric']['cudnn_benchmark'] is False,
            final_F=contract['final_F_code_sha256']==sha(BASELINE_ROOT/'scripts/research/pallet_dim_conditioned_p_v1/pose.py'),
            original_feature_dimension_loader=training_module.Data is original_data and evaluation_module.Data is original_data,
            original_scorer_reducer=training_module.action_scores is original_scores,
            original_J_decoder=evaluation_module.decode_bank is original_decode,
            frozen_input_preflight=preflight_current,
            changed_components=contract['changed_components']==['training_target'],
            nested_only_target_diff=sorted(k for k in set(contract['OLD'])|set(contract['NEW']) if contract['OLD'].get(k)!=contract['NEW'].get(k))==['training_target'],
            final_checkpoint_only=new['final_checkpoint_only'] and old['final_checkpoint_only'],real_training_zero=new['real_training']==old['real_training']==0)
        rows.append(dict(seed=seed, OLD=old, NEW=new, matched_contract=contract, independent_checks=checks,matched_contract_passed=all(checks.values()),source_bindings=[binding(old_path),binding(new_path)]))
    return dict(schema='pose_target_matched_comparison_v1', rows=rows, only_intended_changed_component='training_target',
        intended_OLD='2D raw-phase candidate squared-pixel soft target cross entropy', intended_NEW='hard first-argmin final ADDsym cost action cross entropy',
        unchanged_inference='same JointActionScorer action_scores -> hard J; no GT; exact native bank -> same prediction-only final F',
        no_new_PERM_fit=True,source_bindings=[binding(p) for p in [OLD/'A_protocol.json',OLD/'A_manifest.json',DOC/'PREFLIGHT.json',DOC/'INPUT_BINDINGS.json',DOC/'PROTOCOL.json']])


def n(value, digits=6):
    return 'NA' if value is None else f'{value:.{digits}f}'


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|',*['| '+' | '.join(map(str,row))+' |' for row in rows]])


def verdict(synthetic, seeds, matched, tradeoffs):
    if any(not r['matched_contract_passed'] for r in matched['rows']):
        return 'BLOCKED_INTEGRITY'
    return comparison_question(synthetic,seeds,'FIT_GEO_J')['verdict']


def comparison_question(synthetic,seeds,family):
    paired=synthetic['seed_mean_comparisons'][f'POSE_TARGET_GEO_J_minus_{family}']['statistics']
    improved = [k for k in METRICS if paired[k]['frame']['mean_paired_difference'] is not None and paired[k]['frame']['mean_paired_difference'] < -TOLERANCES[k]]
    supported=[]
    per_seed={}
    for key in METRICS:
        per_seed[key]=[synthetic['comparisons'][f'POSE_TARGET_GEO_J_seed{s}_minus_{family}_seed{s}']['statistics'][key]['frame']['mean_paired_difference'] for s in seeds]
        if key in improved and paired[key]['frame']['CI95'] is not None and paired[key]['frame']['CI95'][1]<0 and sum(v is not None and v < -TOLERANCES[key] for v in per_seed[key])>=2:
            supported.append(key)
    harms = []
    for seed in seeds:
        a=synthetic['summaries'][f'POSE_TARGET_GEO_J_seed{seed}'];b=synthetic['summaries'][f'{family}_seed{seed}']
        raw_new=synthetic['damage_vs_RAW'][f'POSE_TARGET_GEO_J_seed{seed}'];raw_base=synthetic['damage_vs_RAW'][f'{family}_seed{seed}']
        harms.append(a['pose']['failures']>b['pose']['failures'] or a['corner']['gross20']>b['corner']['gross20'] or raw_new['good5_to_bad10']>raw_base['good5_to_bad10'])
    worsens = [k for k in METRICS if paired[k]['frame']['mean_paired_difference'] is not None and paired[k]['frame']['mean_paired_difference'] > TOLERANCES[k]]
    if supported and len(seeds)==3 and not any(harms) and not worsens:
        label='POSE_TARGET_SUPPORTED'
    elif improved and (any(harms) or worsens):
        label='POSE_TARGET_TRADEOFF'
    else:
        label='POSE_TARGET_NOT_SUPPORTED'
    return dict(verdict=label,reference_family=family,improved_pointestimate_metrics=improved,negative_CI_and_reproduced_metrics=supported,
        per_seed_paired_means=per_seed,opposing_pose_metrics=worsens,per_seed_damage_or_coverage_increase=harms,
        stable_evidence_requires_three_seeds=True,REAL_used=False)


def render_reports(synthetic, real, matched, recoveries, hypotheses, tradeoffs, decision, label):
    seeds = synthetic['executed_new_seeds']
    receipts = [r['NEW'] for r in matched['rows']]
    count_path = DOC/'EXECUTION_COUNTS.json'; counts = read(count_path) if count_path.exists() else {}
    cache_path = DOC/'POSE_COST_CACHE_MANIFEST.json'; cache = read(cache_path) if cache_path.exists() else {}
    auxiliary_path=DOC/'TRAIN_2D_6D_TARGET_COMPARISON.json'
    auxiliary=read(auxiliary_path) if auxiliary_path.exists() else None
    cache_counts={key:cache.get(key) for key in ['status','full_TRAIN_rows','completed_rows','available_targets','excluded_all_F_invalid',
        'candidate_F_attempted','candidate_F_completed','candidate_F_available','candidate_F_failures','PnP_counts','chunk_seconds_sum','seconds_wall','workers']}
    parts = ['# 최종 6D target matched 실험 결과',
        '[확인] 후보를 만드는 6D 기하와 원 2D 감독/최종 6D 평가 사이의 차이에서, 이번에는 감독만 hard final ADDsym 첫 argmin action CE로 바꿨어. 후보·특징·치수·모델·초기값·순서·최종6,000 update checkpoint·hard J·prediction-only F를 실제 receipt로 대조해. 논문·LaTeX·PDF·참고문헌을 쓰거나 빌드하지 않았어.',
        '[확인] 요청된 단일 target 교체는 soft 2D CE→hard 6D CE야. target component 안에서 비용 정렬과 label hardening이 함께 달라지므로 둘의 효과는 이 비교로 따로 식별하지 않아. 이를 나누는 추가 대조를 실행하지 않았어.',
        f"[확인] 실행한 정식 fit은 {len(receipts)}개, optimizer update는 {sum(r['updates'] for r in receipts):,}회야. 추가 PERM/hyperparameter/후속 모델 실험은0이야.",
        '## 실제 실행과 seed1 screen',
        f"[확인] seed1 screen은 {decision['decision']}이야. {decision['reason']} REAL_DEV는 판단에 쓰지 않았어. seed2/3 실행은 {'동일 잠금 설정으로 완료' if len(seeds)==3 else '미실행'}야.",
        '[확인] CPU pose-cost와 GPU fit/evaluation의 실제 수는 EXECUTION_COUNTS/POSE_COST_CACHE_MANIFEST/TRAIN_RECEIPTS 및 per-seed 원행 receipt가 기준이야. 아래는 현재 영수증의 그대로인 실행 내역이야.',
        '```json\n'+json.dumps(clean(dict(execution_counts=counts,pose_cache_execution=cache_counts,fit_seconds=[dict(seed=r['seed'],updates=r['updates'],seconds=r['seconds']) for r in receipts])),ensure_ascii=False,indent=2)+'\n```']
    if auxiliary is not None:
        parts.extend(['## TRAIN 2D/6D target index 차이',
            '[확인] source TRAIN 감독에서만 계산한 부수 진단이야. 새 학습·feature forward·F 호출을 추가하지 않고 현재 고정 target을 서로 비교했어. 정의 없는 2D target과 모든 F 실패인 6D target은 별도 분모로 남겼어.',
            table(['항목','실제 수'],[[key,auxiliary[key]] for key in ['full_usable_TRAIN','common_defined','undefined_2d','all_F_invalid_6d','exact_same_index','both_NoOp','two_d_NoOp_six_d_move','two_d_move_six_d_NoOp','both_move_different_action']]+[['정확히 같은 index % (공동정의 분모)',n(auxiliary['exact_same_index_fraction']*100,4)]] )])
    parts.append('## matched OLD/NEW 계약의 실제 나란한 값')
    for receipt in matched['rows']:
        contract=receipt['matched_contract']
        keys=['candidate_bank_sha256','architecture_sha256','parameter_count','feature_source','dimension_input','optimizer','schedule',
            'gradient_clipping','batch','updates','order_sha256','initial_state_sha256','inference_readout','final_F_code_sha256','numeric','training_target']
        parts.extend([f"### seed{receipt['seed']}",table(['고정/변경 항목','OLD soft2D','NEW hard6D'],
            [[key,json.dumps(contract['OLD'][key],ensure_ascii=False),json.dumps(contract['NEW'][key],ensure_ascii=False)] for key in keys]),
            '[확인] 실제 원 영수증과 별도 대조한 계약 검산: '+json.dumps(receipt['independent_checks'],ensure_ascii=False)])
    for result in (synthetic, real):
        split = result['split'];parts.extend(['## '+split, '[확인] '+result['scope']])
        rows=[]
        for name,value in result['summaries'].items():
            c,p=value['corner'],value['pose']
            rows.append([name,n(c['matched_pooled_corner8_median_px'],4)+' / '+n(c['matched_pooled_corner8_P90_px'],4),n(c['PCK']['10'],6),n(c['gross20'],6),n(p['translation_cm']['median'],4)+' / '+n(p['translation_cm']['P90'],4),n(p['rotation_deg']['median'],4)+' / '+n(p['rotation_deg']['P90'],4),n(p['ADDsym_m']['median'],6)+' / '+n(p['ADDsym_m']['P90'],6),f"{p['available']}/{p['total_frames']}",value['NoOp'] if value['NoOp'] is not None else 'NA'])
        parts.append(table(['방법/seed','코너 med/P90 px','PCK10 분율','gross20 분율','T med/P90 cm','R med/P90 deg','ADD med/P90 m','F coverage','NoOp'],rows))
        paired_rows=[]
        for seed in seeds:
            for family in ['FIT_GEO_J','N3','PoseFix','RAW']:
                key=f'POSE_TARGET_GEO_J_seed{seed}_minus_'+(family if family=='RAW' else f'{family}_seed{seed}')
                value=result['comparisons'][key]
                for metric in METRICS:
                    stat=value['statistics'][metric];frame=stat['frame'];secondary=stat['scenario' if split=='SYNTH_HELDOUT' else 'session']
                    paired_rows.append([key,metric,n(frame['mean_paired_difference'],8),n(frame['median_paired_difference'],8),frame['CI95'],secondary['CI95'],frame['common_eligible_frames']])
        parts.extend(['### matched OLD→NEW가 우선인 paired 비교',table(['NEW−control','지표','paired 평균','paired 중앙값','frame95%CI','scenario/session95%CI','공동성공'],paired_rows),
            '[확인] 동일ID의 차이를 먼저 계산하고 seed평균도 같은ID에서 수행했어. 10,000회/seed20260917 draw를 모든 seed·방법·지표에 공유했어. synthetic은 frame이 주, 원 scenario가 보조야. REAL은 session과 frame을 둘 다 보고했고 독립 실험/다중비교 교정이 아니야. 자세 실패는4분모에 남기고 conditional pose값의 공동성공과 전체 분모를 구분했어.'])
        recovery_rows=[];wd_rows=[];damage_rows=[]
        for seed in seeds:
            recovery_value=recoveries[split][str(seed)];dist=recovery_value['distributions'];wd=hypotheses[split][str(seed)]['groups']['SWITCH'];name=f'POSE_TARGET_GEO_J_seed{seed}'
            recovery_rows.append([seed,recovery_value['common_success'],n(dist['headroom_m']['median'],8),n(dist['old_recovered_m']['median'],8),n(dist['new_recovered_m']['median'],8),
                n(dist['old_gap_m']['median'],8)+' / '+n(dist['old_gap_m']['P90'],8),n(dist['new_gap_m']['median'],8)+' / '+n(dist['new_gap_m']['P90'],8),
                n(recovery_value['ratios']['old']['distribution']['median'],6),n(recovery_value['ratios']['new']['distribution']['median'],6),recovery_value['excluded_headroom_le_tolerance'],
                recovery_value['ratios']['new']['negative'],recovery_value['ratios']['new']['over_one'],n(recovery_value['exact_oracle_action']['old']['rate'],6),n(recovery_value['exact_oracle_action']['new']['rate'],6)])
            wd_rows.append([seed,wd['frames'],wd['recovery']['OLD']['final_hypothesis_recovered'],wd['recovery']['NEW']['final_hypothesis_recovered'],wd['recovery']['OLD']['exact_oracle_action'],wd['recovery']['NEW']['exact_oracle_action']])
            for ref in ('RAW',f'FIT_GEO_J_seed{seed}',f'N3_seed{seed}',f'PoseFix_seed{seed}'):
                value=tradeoffs[split][name+'_minus_'+ref];damage=value['corner_damage'];damage_rows.append([name+'−'+ref,damage['good5_to_bad10'],damage['bad20_to_good10'],value['canonical_corner_denominator'],str(value['frame_counts']),str(value['canonical_corner_counts'])])
        parts.extend(['### oracle gap 회수',table(['seed','공동성공','headroom med m','OLD회수 med m','NEW회수 med m','OLD gap med/P90 m','NEW gap med/P90 m','OLDratio med','NEWratio med','headroom≤1e−7 제외','NEW 음수ratio','NEWratio>1','OLD exactaction분율','NEW exactaction분율'],recovery_rows),
            '[확인] ratio는 같은 reference/bank의 개발 진단이고 음수·1초과를 clamp하지 않았어. 전체 분포·별도 결측/F실패 제외·oracle보다 좋은 수치오차 범위는 ORACLE_RECOVERY에 남겼어.',
            '### 최종 W/D hypothesis correction',table(['seed','oracle전환 frame','OLD최종hyp 회수','NEW최종hyp 회수','OLD exactaction','NEW exactaction'],wd_rows),
            '[확인] 생성 perturbation 이름과 최종 F 선택을 구분했어. oracle가 전환하지 않는 집단/전환하는 집단/불가 집단의 RAW·OLD·NEW·oracle T/R/ADD와 frame수도 WD_HYPOTHESIS_ANALYSIS에 모두 있어. REAL GT는 이 사후 표에만 사용돼.',
            '### 2D/6D tradeoff와 canonical damage',table(['NEW−control','good<5→bad>10 코너','bad>20→good<10 코너','canonical코너 분모','frame사분면','corner사분면'],damage_rows),
            '[확인] 손상 수는 frame 수가 아니라 같은 canonical GT 코너 수야. ADDsym와 native mean-L2/각 canonical코너의 방향을 결합했고 수치동률·결측/F실패도 별도 남겼어. 2D 또는 ADD 한 지표만으로 전체 센싱 개선을 주장하지 않아.'])
    parts.extend(['## 고정 사항과 supervision 변경', '[확인] MATCHED_COMPARISON.json의 OLD/NEW 영수증과 matched_contract에 bank/architecture/params/features/dimensions/optimizer/LR/batch/updates/order/initializer/readout/F/numerical 설정을 나란히 남겼어. initializer 검사는 첫 optimizer 전 원 digest와 같아야 해. hard 6D target의 분모 제외가 있으면 원 55,915 및 order 노출과 분리해 밝혀. 새 checkpoint를 매 update 저장한 운영 차이는 원500 update 간격보다 촘촘한 중단 계수 보존이며 모델/optimizer 산술 차이가 아니다. 따라서 wall시간 자체를 원 방법의 속도 효과로 해석하지 않았어.',
        '## 테스트', '[확인] 실제 계약 테스트 명령·PASS 수는 CONTRACT_TEST_RESULTS.json에 기록하고 아래 최종 재집계에서 연결해. 테스트는 임의 재학습이나 새 모델/F 실행이 아니야.',
        '## 판정',label,
        '[확인] OLD 감독 변경의 판정과 N3보다 6D를 더 잘 예측하는 질문은 별도야. N3_6D_question: '+json.dumps(synthetic['N3_6D_question'],ensure_ascii=False),
        '[확인] 이 판정은 고정 bank·참조·모델·예산·반복 개발 자료에 한정돼. 실제 DEV의 reference는 같은2D 주석+치수 재구성이며 독립 physical pair는0/BLOCKED_DATA야. 이후 실험을 자동 실행하지 않았어.'])
    test_path=DOC/'CONTRACT_TEST_RESULTS.json'
    if test_path.exists():
        tests=read(test_path);parts.extend([f"[확인] 실제 테스트: `{tests.get('command','see receipt')}` — {tests.get('passed','NA')} PASS / {tests.get('failed','NA')} FAIL."])
    (DOC/'RESULT_KO.md').write_text('\n\n'.join(parts)+'\n')
    next_parts=['# 다음 판단', '[확인] 이번 판정은 '+label+'야. source TRAIN final ADDsym hard target만 바꾼 고정 실험이고, 다음 실험은 실행하지 않아.',
        '[확인] matched target 교체는 soft2D→hard6D 전체야. 비용 정렬과 label hardness 각각의 기여는 별도로 식별되지 않았어.',
        '[확인] seed1 synthetic screen: '+decision['decision']+' — '+decision['reason']+' REAL_DEV는 screen 조건에 포함되지 않았어.',
        '[추정·미검증] 원인 후보는 서로 구분해야 해. TRAIN target exact-match가 낮으면 후보 간 구별/표현·최적화의 가능성이 남고, TRAIN 회수가 높지만 synthetic에서 낮으면 일반화 가능성이 남아. synthetic 개선과 REAL만의 악화는 synthetic-to-real transfer 가능성과도 맞지만 독립 실사 참조가 없어 원인을 확정하지 않아. 6D 개선과 canonical 2D 손상이 함께 늘면 preservation 목적의 충돌 가능성이 있어. 현재 관찰만으로 모델 크기·최적화·특징 부재를 원인으로 단정하지 않아.',
        '[확인] TRAIN target/기존2Dteacher의 index 일치·NoOp/이동 구분은 TRAIN_2D_6D_TARGET_COMPARISON의 실제 요약에 있고, 이번 loss·checkpoint를 선택하는 데 쓰지 않았어. TRAIN 새로운 추가 점수 추론이 없으면 학습 전후 exact-match의 전 모집단 평가를 완료했다고 말하지 않아.',
        '[추정·미검증] 후속 후보는 이 원행을 바탕으로 사전 고정할 수 있지만 현재 실행하지 않아. 새 backbone·cap·LR·temperature·seed·candidate 탐색은 이번 결과에서 정당화되지 않았어.']
    observations=[dict(seed=r['seed'],cumulative=r['observed_preupdate_training_action_fit'],
        last_window=r['final_observed_preupdate_training_window']) for r in receipts
        if 'observed_preupdate_training_action_fit' in r]
    if observations:
        next_parts.extend(['[확인] 기존 학습 forward에서 optimizer update 직전에 관측한 action 적중/CE야. 누적과 마지막100 update(5901–6000)의 eligible/제외·이동/NoOp exposure 분모를 구분해 아래 실제 receipt 값을 옮겼어. 추가 model forward/optimizer update는0이야. checkpoint가 변하며 같은 TRAIN 행을 반복 노출한 관측이고 최종 checkpoint의 고유 TRAIN 전체 정확도가 아니야.',
            '```json\n'+json.dumps(clean(observations),ensure_ascii=False,indent=2)+'\n```',
            '[추정·미검증] 이 관측 pre-update exposure 적중률과 heldout oracle-action/gap 회수는 범위가 달라. 관측 적중이 낮으면 표현·최적화 가능성, 관측 적중이 높고 heldout 회수가 낮으면 일반화 가능성이 남지만, 해당 수치만으로 underfitting 또는 어느 원인을 확정하지 않아.'])
    if label=='POSE_TARGET_SUPPORTED':
        next_parts.append('[확인] 비용 목표가 일치하는 감독에서 회수 증가가 관찰된 범위와 남은 oracle gap을 분리해. 후보/모델/학습량이 같다는 계약 안에서 목표 변경의 효과이며 독립 실사 성능의 인과 확증은 아니야.')
    (DOC/'NEXT_DECISION_KO.md').write_text('\n\n'.join(next_parts)+'\n')


def final(root=ROOT, seeds=(1,)):
    configure(root)
    seeds=list(seeds)
    assert seeds in ([1],[1,2,3])
    decision=read(DOC/'SEED1_SCREENING.json')
    if len(seeds)==3:
        assert decision['continue_seeds'], 'closed seed1 gate cannot authorize seeds2/3'
    begin=time.monotonic();synthetic, sr, sw, st=analyze_split('SYNTH_HELDOUT',seeds);real, rr, rw, rt=analyze_split('REAL_DEV',seeds)
    matched=matched_comparison(seeds);recoveries={'SYNTH_HELDOUT':sr,'REAL_DEV':rr};hypotheses={'SYNTH_HELDOUT':sw,'REAL_DEV':rw};tradeoffs={'SYNTH_HELDOUT':st,'REAL_DEV':rt}
    synthetic['OLD_target_question']=comparison_question(synthetic,seeds,'FIT_GEO_J')
    synthetic['N3_6D_question']=comparison_question(synthetic,seeds,'N3')
    label=verdict(synthetic,seeds,matched,tradeoffs)
    write(DOC/'SYNTH_RESULTS.json',synthetic);write(DOC/'REAL_DEV_RESULTS.json',real);write(DOC/'MATCHED_COMPARISON.json',matched)
    write(DOC/'ORACLE_RECOVERY.json',dict(schema='pose_target_oracle_recovery_v1',results=recoveries));write(DOC/'WD_HYPOTHESIS_ANALYSIS.json',dict(schema='pose_target_WD_analysis_v1',results=hypotheses));write(DOC/'TWO_D_SIX_D_TRADEOFF.json',dict(schema='pose_target_2D_6D_tradeoff_v1',results=tradeoffs))
    render_reports(synthetic,real,matched,recoveries,hypotheses,tradeoffs,decision,label)
    result=dict(schema='pose_target_reporting_execution_v1',status='DONE',verdict=label,seeds=seeds,seconds_wall=time.monotonic()-begin,
        code_sha256=sha(Path(__file__)),source_bindings=synthetic['source_bindings']+real['source_bindings'],
        new_model_calls=0,new_final_F_calls=0,new_PnP_calls=0,new_fit_calls=0,new_optimizer_updates=0,manuscript_writes=0)
    write(DOC/'REPORT_EXECUTION.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['screen','final'],required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[1])
    parser.add_argument('--repo-root',type=Path,default=ROOT)
    args=parser.parse_args(argv)
    result=screen(args.repo_root) if args.stage=='screen' else final(args.repo_root,args.seeds)
    print(json.dumps(clean(result),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
