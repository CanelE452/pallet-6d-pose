"""Read-only A/E/F/G diagnostics, with an explicit source and output contract.

Historical files are never changed. References are read only for scoring after
the stored prediction-only choices. Protected physical FINAL is excluded.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / '_docs/experiments/pallet_wd_info_fast_20261011'
NAMES = ('long-face-front', 'short-face-front')
UNCLEAR = {'eval_night08:1779449470423201536', 'eval_night09:1779449638581035008',
           'eval_outside:1778653367706938112', 'plastic_night_01:038567'}
WD_REL = '_docs/experiments/pallet_wd_hypothesis_diag_20261010'
GEOMETRY_REL = 'challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def digest(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(path):
    with Path(path).open(encoding='utf-8') as f: return json.load(f)


def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f: yield json.loads(line)


def clean(v):
    if isinstance(v, dict): return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [clean(x) for x in v]
    if isinstance(v, np.ndarray): return clean(v.tolist())
    if isinstance(v, np.generic): return clean(v.item())
    if isinstance(v, float) and not math.isfinite(v): return None
    return v


def write(name, value):
    path = DOC / name
    assert path.is_relative_to(DOC) and not path.exists(), f'Preserve {path.name}'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_rows(name, values):
    path = DOC / name
    assert not path.exists()
    with gzip.open(path, 'wt', encoding='utf-8', compresslevel=6) as f:
        for v in values: f.write(json.dumps(clean(v), ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')


def pose_api(source):
    # Historical module resolves its original environment without loading real RGB.
    os.environ['PALLET_SOURCE_ROOT'] = str(source)
    from scripts.research.pallet_wd_hypothesis_diag_20261010 import common as c
    c.SOURCE = source
    return c.pose_api()


def contract(source, published):
    manifest_path = source / 'challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json'
    manifest = read(manifest_path)
    assert manifest['expected_count'] == 319 and manifest['role'] == 'DEV'
    assert manifest['provenance']['held_out_final'] is False
    allowed, excluded, role_rows = [], [], []
    for r in manifest['items']:
        session_path = (source / r['image_path']).parent.parent / 'session.json'
        meta = read(session_path)
        role = meta['population_role']
        assert role in ('DEV', 'FINAL')
        (allowed if role == 'DEV' else excluded).append(r['frame_id'])
        role_rows.append(dict(id=r['frame_id'], session=r['session_id'],
                             manifest_role=r['population_role'], physical_session_role=role,
                             session_metadata= str(session_path.relative_to(source)),
                             session_metadata_sha256=sha(session_path)))
    assert len(allowed) == 173 and len(excluded) == 146
    allow = set(allowed)
    real = [r for r in rows(published / WD_REL / 'STAGE1_ROWS_REAL.jsonl.gz')
            if r['id'] in allow and r['seed'] == 1 and r['method'] == 'BASE']
    assert len(real) == 173
    intersection = sorted(r['id'] for r in real if r['grade'] in ('clean', 'moderate'))
    references = read(source / 'data/evaluation/pallet_eval_v1/DATASET_CONTRACT.json')
    assert references['roles']['FINAL'] == 'untouched final evaluation population'
    packet = dict(status='LOCKED_CONSERVATIVE_SOURCE_ROLE_INTERSECTION',
        manifest=dict(path=str(manifest_path.relative_to(source)), sha256=sha(manifest_path),
                      declared_role='DEV', declared_held_out_final=False),
        physical_role_source='data/evaluation/pallet_eval_v1/DATASET_CONTRACT.json:9-13',
        reason='PAPER manifest relabels physical FINAL as DEV, without changing active frame roles. Exclude physical FINAL for all new scoring/features.',
        allowed_ids=sorted(allowed), allowed_frames=173,
        allowed_id_sha256=digest(sorted(allowed)),
        allowed_session_counts=dict(sorted(Counter(i.split(':')[0] for i in allowed).items())),
        excluded_protected_ids=sorted(excluded), excluded_frames=146,
        excluded_session_counts=dict(sorted(Counter(i.split(':')[0] for i in excluded).items())),
        rect245_intersection_ids=intersection, rect245_intersection_frames=len(intersection),
        rect245_post_development_selection=True,
        grade_counts=dict(Counter(r['grade'] for r in real)),
        unclear_allowed_ids=sorted(allow & UNCLEAR),
        axis_accuracy_evaluable_frames=len(allow - UNCLEAR),
        full_pose_success_denominator=173,
        prior_final4_role='Previously opened DEV; no re-sealing; forbidden for model/threshold selection',
        currently_sealed_names=['testset_full8_manifest', 'handannot17'],
        square=dict(frames=118, sessions=1, physical_symmetry='C4', long_short_axis='NOT_APPLICABLE', merged_with_rectangle=False),
        historical_rect319_rect245='Public aggregate reuse only; no new protected frame scoring',
        source_roles=role_rows)
    write('INPUT_CONTRACT.json', packet)
    return packet


def truth_real(source, reference_rows):
    raw = read(source / 'data/pallet/results/paper_pose_metric_closure_v1/GEOMETRY_RESOLVED_POSE_GT.json')['frames']
    out = {}
    for fid, r in reference_rows.items():
        g = raw[fid.replace(':', '__', 1)]
        xyz = np.asarray(r['fixed_metadata']['dimensions_pnp_WH_D_m'], float)
        d = g['physical_dimensions_m']; cf = np.array([d['across'], d['height'], d['along']])
        Rcf = np.asarray(g['R_gt_representative'], float)
        a = math.pi / 2
        q = np.eye(3) if abs(cf[0] - xyz[0]) < 1e-6 else np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
        assert g['physical_long_axis'] in ('CF_WIDTH', 'CF_DEPTH')
        expected = 'long-face-front' if g['physical_long_axis'] == 'CF_WIDTH' else 'short-face-front'
        assert expected == r['hypOracle']
        assert abs(cf[0] - (max(xyz[0], xyz[2]) if expected == NAMES[0] else min(xyz[0], xyz[2]))) < 1e-6
        out[fid] = dict(R=Rcf @ q, t=np.asarray(g['t_gt']), xyz=xyz,
                        body_R=Rcf, body_xyz=cf, order=int(r['pose_symmetry_order']))
    return out


def good(p):
    return bool(p.get('available') and p['translation_cm'] < 5 and p['rotation_deg'] < 5)


def axis_wrong(name, oracle, fid):
    return None if fid in UNCLEAR or name is None else int(name != oracle)


def basic(values):
    a = np.asarray([x for x in values if x is not None], float)
    a = a[np.isfinite(a)]
    if not len(a): return dict(n=0, mean=None, SD=None, variance=None, median=None, P90=None, P95=None)
    v = float(a.var(ddof=1)) if len(a) > 1 else None
    return dict(n=len(a), mean=float(a.mean()), SD=None if v is None else math.sqrt(v),
                variance=v, median=float(np.median(a)), P90=float(np.quantile(a, .9)),
                P95=float(np.quantile(a, .95)), variance_ddof=1)


def clustered_interval(values, sessions, seed=20260917):
    v = np.asarray(values, float); unique, group = np.unique(sessions, return_inverse=True)
    n = len(unique); valid = np.isfinite(v)
    if not valid.any(): return None
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n, np.full(n, 1 / n), size=10000)
    sums = np.bincount(group, weights=np.where(valid, v, 0), minlength=n)
    counts = np.bincount(group, weights=valid, minlength=n)
    denominator = draws @ counts
    means = np.divide(draws @ sums, denominator, out=np.full(len(draws), np.nan), where=denominator > 0)
    return np.nanquantile(means, [.025, .975]).tolist()


def summarize(records):
    grouped = defaultdict(list)
    for r in records: grouped[(r['population'], r['backbone'], r['method'], r['arm'])].append(r)
    result = []
    for group, rr in sorted(grouped.items()):
        by = defaultdict(list)
        for r in rr: by[r['id']].append(r)
        ids = sorted(by); sessions = [by[i][0]['session'] for i in ids]
        frame = {}
        for k in ('translation_cm', 'rotation_deg', 'ADDsym_m', 'IoU3D', 'success', 'available', 'axis_wrong', 'wd_confusion'):
            values = []
            for fid in ids:
                val = [r[k] for r in by[fid] if r.get(k) is not None]
                values.append(float(np.mean(val)) if val else np.nan)
            frame[k] = values
        stats = {k: basic(v) for k, v in frame.items()}
        if group[0] == 'REAL173':
            for k, v in frame.items(): stats[k]['CI95_mean_session_cluster'] = clustered_interval(v, sessions)
        for k in ('success', 'available', 'axis_wrong', 'wd_confusion'):
            stats[k]['rate'] = stats[k]['mean']; stats[k]['full_frames'] = len(ids)
        result.append(dict(population=group[0], backbone=group[1], method=group[2], arm=group[3],
                           frames=len(ids), sessions=len(set(sessions)), seeds=sorted({r['seed'] for r in rr}),
                           aggregation='Per-frame mean over fixed seeds, then frame mean/quantiles. Missing success=0; axis UNCLEAR=NA.',
                           metrics=stats))
    return result


def a2(source, published):
    c = read(DOC / 'INPUT_CONTRACT.json'); allowed = set(c['allowed_ids'])
    selected_rows = []
    for name in ('STAGE1_ROWS_REAL.jsonl.gz', 'STAGE1_ROWS_AUX.jsonl.gz'):
        selected_rows.extend(r for r in rows(published / WD_REL / name) if r['id'] in allowed)
    assert len(selected_rows) == 173 * 24
    references = {r['id']: r for r in selected_rows if r['backbone'] == 'YOLO' and r['method'] == 'BASE' and r['seed'] == 1}
    truth = truth_real(source, references); pose = pose_api(source)
    output, candidate_output = [], []
    common_pairs = defaultdict(list); counts = Counter()
    for r in selected_rows:
        candidates = {}
        for h in NAMES:
            if r['hypS0'] == h: p = r['pose']['S0']
            elif r['hypOracle'] == h: p = r['pose']['ORACLE']
            else: p = pose.metric((r['id'], r['candidates'][h]['actual_pose'], truth[r['id']]))
            candidates[h] = p
        valid = [h for h in NAMES if candidates[h].get('available')]
        current = r['hypS0']; parity = r['hypOracle'] if r['id'] not in UNCLEAR else current
        tmin = min(valid, key=lambda h: (candidates[h]['translation_cm'], NAMES.index(h))) if valid else None
        rmin = min(valid, key=lambda h: (candidates[h]['rotation_deg'], NAMES.index(h))) if valid else None
        sm = min(valid, key=lambda h: (not good(candidates[h]), h != current, NAMES.index(h))) if valid else None
        choices = dict(CURRENT=current, GT_PARITY_DIAGNOSTIC=parity, BEST_T_POSTHOC=tmin,
                       BEST_R_POSTHOC=rmin, BEST_SUCCESS_POSTHOC=sm)
        support = dict(population='REAL173', id=r['id'], session=r['session'], backbone=r['backbone'],
                       method=r['method'], seed=r['seed'], grade=r['grade'],
                       axis_reference_status='UNCLEAR' if r['id'] in UNCLEAR else 'CONFIRMED',
                       hyp_current=current, hyp_reference=None if r['id'] in UNCLEAR else r['hypOracle'],
                       formal_score_gap=r['formal_score_gap_px'], reference_margin_px=r['reference_margin_px'],
                       elevation_deg=r['elevation_deg'], distance_m=r['distance_m'], material=r['material'],
                       candidate_metrics=candidates,
                       candidate_scores={h:r['candidates'][h]['formal_score'] for h in NAMES})
        candidate_output.append(support)
        for arm, h in choices.items():
            p = dict(available=False) if h is None else candidates[h]
            out = dict(**{k:v for k,v in support.items() if k not in ('candidate_metrics', 'candidate_scores')},
                       arm=arm, selected_hypothesis=h, available=int(bool(p.get('available'))),
                       success=int(good(p)), axis_wrong=axis_wrong(h, r['hypOracle'], r['id']),
                       wd_confusion=int(bool(p.get('available') and p['rotation_deg'] > 45 and abs(p['yaw_deg']) >= 60)))
            out.update({k:p.get(k) for k in ('translation_cm','rotation_deg','ADDsym_m','IoU3D')})
            output.append(out)
            if candidates.get(current, {}).get('available') and p.get('available'):
                common_pairs[(r['backbone'],r['method'],arm)].append(dict(id=r['id'],seed=r['seed'],
                    delta_T_cm=p['translation_cm']-candidates[current]['translation_cm'],
                    delta_R_deg=p['rotation_deg']-candidates[current]['rotation_deg']))
        counts[(r['backbone'],r['method'],'both_candidates_fail')] += not any(good(p) for p in candidates.values())
        counts[(r['backbone'],r['method'],'correct_parity_pose_threshold_fail')] += not good(candidates[r['hypOracle']])
    write_rows('AE_CANDIDATE_METRICS.jsonl.gz', candidate_output)
    write_rows('A2_ORACLE_ROWS.jsonl.gz', output)
    write('A2_ORACLE_METRICS.json', dict(status='DONE', new_PnP_calls=0, summaries=summarize(output),
        population='REAL173', full_pose_denominator=173, UNCLEAR_not_overridden=True,
        failure_counts=[dict(backbone=b,method=m,event=e,seed_frame_count=v,
                             frame_seed_denominator=173*3) for (b,m,e),v in sorted(counts.items())],
        common_valid_paired=[dict(backbone=b,method=m,arm=a,rows=len(rr),
            delta_T_cm=basic(x['delta_T_cm'] for x in rr),delta_R_deg=basic(x['delta_R_deg'] for x in rr))
            for (b,m,a),rr in sorted(common_pairs.items())],
        reference='Existing manual2D geometry PnP; not independent physical metrology',
        posthoc_oracles_are_deployable=False))
    return candidate_output


def e_curve(rr, confidence_name, confidence):
    n = len(rr); conf = np.asarray(confidence, float)
    finite = np.isfinite(conf); levels = np.unique(conf[finite])[::-1]
    curve = []
    total_success = sum(good(r['candidate_metrics'].get(r['hyp_current'], {})) for r in rr)
    known = [i for i,r in enumerate(rr) if r['hyp_reference'] is not None and r['hyp_current'] is not None]
    err = [int(rr[i]['hyp_current'] != rr[i]['hyp_reference']) for i in known]
    from scripts.research.pallet_wd_hypothesis_diag_20261010.statistics import auc
    detector = auc(-conf[known], err)
    for threshold in levels:
        accepted = finite & (conf >= threshold); kk = [i for i in known if accepted[i]]
        success = sum(good(rr[i]['candidate_metrics'].get(rr[i]['hyp_current'], {})) for i in np.flatnonzero(accepted))
        count = int(accepted.sum())
        wrong = sum(rr[i]['hyp_current'] != rr[i]['hyp_reference'] for i in kk)
        curve.append(dict(threshold=float(threshold), accepted=count, coverage=count/n,
            axis_evaluable_accepted=len(kk), wrong_axis_accepted=wrong,
            accepted_axis_error=None if not kk else wrong/len(kk),
            accepted_6D_failure=1-success/count, accepted_success_over_all=success/n,
            abstained=n-count, invalid_confidence=int((~finite).sum()),
            unclear_axis_accepted=sum(accepted[i] and rr[i]['hyp_reference'] is None for i in range(n))))
    cov=np.array([x['coverage'] for x in curve]); risk=np.array([x['accepted_6D_failure'] for x in curve])
    order=np.argsort(cov);aurc=float(np.trapz(np.r_[risk[order][0],risk[order]],np.r_[0,cov[order]])) if len(curve) else None
    fixed=[]
    for target in (1,.95,.9,.8,.5):
        options=[x for x in curve if x['coverage']>=target]
        chosen=min(options,key=lambda x:x['coverage']) if options else (curve[0] if curve else None)
        fixed.append(dict(target_coverage=target, actual=chosen))
    return dict(confidence=confidence_name,frames=n,curve=curve,fixed_points=fixed,
        AURC_6D_failure=aurc,AURC_integration='Trapezoid over attained coverage; risk at smallest coverage extended to zero',
        error_detection_ROC_AUC=detector,ROC_positive='wrong long/short axis; UNCLEAR and no-choice excluded',
        ROC_score='negative confidence',random_rejection_expected_6D_risk=1-total_success/n,
        random_rejection_scope='Label-independent uniform random rejection expectation; no GT ranking',
        deployment_threshold_selected=False)


def e(published):
    # READ the runtime box from historical prediction rows, never a GT box.
    cached=list(rows(DOC/'AE_CANDIDATE_METRICS.jsonl.gz'))
    boxes={}; allowed=set(read(DOC/'INPUT_CONTRACT.json')['allowed_ids'])
    for r in rows(published/WD_REL/'STAGE1_ROWS_REAL.jsonl.gz'):
        if r['id'] not in allowed: continue
        m=r['fixed_metadata']['candidate_metadata']
        if isinstance(m,list):m=m[r['fixed_metadata']['selected_index']]
        box=m.get('box_xyxy',m.get('bbox'))
        boxes[(r['id'],r['method'],r['seed'])]=None if box is None else float(np.linalg.norm(np.asarray(box[2:])-box[:2]))
    groups=defaultdict(list)
    for r in cached:
        if r['backbone']=='YOLO':groups[(r['method'],r['seed'])].append(r)
    curves=[]
    for (method,seed),rr in sorted(groups.items()):
        gaps=[np.nan if r['formal_score_gap'] is None else r['formal_score_gap'] for r in rr]
        normalized=[np.nan if boxes[(r['id'],method,seed)] in (None,0) else gaps[i]/boxes[(r['id'],method,seed)] for i,r in enumerate(rr)]
        for name,values in [('official_weighted_score_gap',gaps),('score_gap_over_predicted_box_diagonal',normalized)]:
            curves.append(dict(method=method,seed=seed,**e_curve(rr,name,values)))
    write('E_RISK_COVERAGE.json',dict(status='DONE',population='REAL173',frames=173,
        new_PnP_calls=0,confidence_gap_unit='Weighted official score (nominal pixel scale includes penalties), not pure residual pixels',
        curves=curves,hypotheses_preserved=2,fallback='If only one candidate valid preserve it; zero valid candidates remain no-pose',
        real_threshold_curves='Descriptive only; no deployed threshold or robot safety claim',
        C_probability_analysis='Added only after sealed C predictions, separate output'))


def f_g1(source, published):
    path=source/'data/pallet/results/pallet_remaining_evidence_connection_20261006_v1/time_physical/VALIDATION.json'
    old=read(path); timeline=old['timeline'];physical=old['physical_reference']
    assert timeline['frames']==8910 and physical['accepted_independent_reference_rows']==0
    assert all(not r['nominal_extrinsics_measured'] for r in old['geometry']['camera_metadata'])
    square=read(published/'_docs/experiments/pallet_posefix_target_data_diagnosis_v1/TEMPORAL_DIVERSITY_AUDIT.json')
    write('F_G_INPUT_AUDIT.json',dict(F=dict(status='BLOCKED_CONTRACT',
        actually_existing_timed_sequence_frames=8910, timed_sessions=4,
        duplicate_sensor_timestamps=timeline['original_duplicate_observations_retained'],
        continuous_timestamps_exist=True,nominal_extrinsics_measured=False,
        unknown_camera_motion_or_fixed_camera_contract=True,
        no_face_switch_independent_contract=False,
        causal_windows_executed=[],oracle_motion_trajectory_reuse='No verified current matching trajectory found',
        raw_incoming_sequences_exist=True, raw_incoming_order_is_not_independent_motion=True,
        square_display_indices_are_timestamps=not square['frame_indices_are_display_order_not_timestamps'],
        reason='Real sequences and timestamps exist, but legal inter-frame physical transforms / stationary camera and object association are not verified. Do not connect arbitrary frame order.'),
        G1=dict(status='BLOCKED_CALIBRATION',independent_floor_calibrations=0,
            accepted_independent_reference_rows=physical['accepted_independent_reference_rows'],
            inspected_existing_reference_join_rows=physical['fixed_reference_join_rows'],
            pallet_floor_contact_condition='UNKNOWN; not silently ineligible',
            S3_reference_information='Other same-session manual-PnP GT planes (REAL), own exact renderer plane (SYNTH); privileged feasibility',
            reasons=physical['reasons']),
        reused_audit_source=dict(path=str(path.relative_to(source)),sha256=sha(path)),
        new_RGB_decoding=0,new_PnP_calls=0,new_sensor_actions=0))


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',required=True,type=Path)
    p.add_argument('--published-root',required=True,type=Path)
    p.add_argument('--task',required=True,choices=['contract','a2','e','fg-audit'])
    a=p.parse_args();start=time.monotonic()
    if a.task=='contract':result=contract(a.source_root,a.published_root);print('CONTRACT',result['allowed_frames'],result['rect245_intersection_frames'],flush=True)
    elif a.task=='a2':a2(a.source_root,a.published_root);print('A2 DONE',flush=True)
    elif a.task=='e':e(a.published_root);print('E DONE',flush=True)
    else:f_g1(a.source_root,a.published_root);print('F/G1 AUDIT DONE',flush=True)
    print('WALL_SECONDS',round(time.monotonic()-start,6),flush=True)


if __name__=='__main__':main()
