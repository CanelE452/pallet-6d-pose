"""Consolidate completed C/D and reused candidates without rerunning inference."""
import csv
import gzip
import importlib.util
from collections import defaultdict
import numpy as np
from . import common as C
from . import paired


def available_metric(p, key):
    if not p.get('available'):
        return None
    aliases = {'translation_cm': 'T_cm', 'rotation_deg': 'R_deg'}
    return p.get(key, p.get(aliases.get(key, '')))


def normalize(r, kind):
    p = r['pose']; available = bool(p.get('available'))
    success = int(available and available_metric(p, 'translation_cm') < 5
                  and available_metric(p, 'rotation_deg') < 5)
    if 'legacy_confusion' in r:
        confusion = int(r['legacy_confusion'])
    else:
        assert not available or available_metric(p, 'yaw_deg') is not None
        confusion = int(available and available_metric(p, 'rotation_deg') > 45
                        and abs(available_metric(p, 'yaw_deg')) >= 60)
    if kind == 'C':
        assert success == int(r['success'])
    elif kind == 'D':
        assert success == int(r['strict_success'])
    return dict(id=r['id'], seed=int(r['seed']), session=r['session'],
        backbone=r.get('backbone', 'YOLO'), method=r['method'], grade=r['grade'],
        source_kind=kind, available=int(available), pose_success=success,
        wd_confusion=confusion, axis_wrong=r.get('axis_error', r.get('axis_wrong')),
        selected_hypothesis=r.get('hyp', r.get('hypothesis', r.get('hyp_current'))),
        p_long=r.get('p_long'),
        **{key: available_metric(p, key) for key in ['translation_cm', 'rotation_deg',
          'yaw_deg', 'ADDsym_m', 'ADDsym_normalized', 'IoU3D']})


def moments(v):
    v = np.asarray(v, dtype=float); v = v[np.isfinite(v)]
    return dict(n=len(v), mean=float(v.mean()) if len(v) else None,
        variance=float(v.var(ddof=1)) if len(v)>1 else None,
        SD=float(v.std(ddof=1)) if len(v)>1 else None,
        median=float(np.median(v)) if len(v) else None,
        P90=float(np.quantile(v, .9)) if len(v) else None,
        P95=float(np.quantile(v, .95)) if len(v) else None,
        maximum=float(v.max()) if len(v) else None)


def run():
    contract = C.read(C.DOC/'INPUT_CONTRACT.json')
    ids = sorted(contract['allowed_ids']); sessions = [i.split(':')[0] for i in ids]
    names = sorted(set(sessions))
    if (C.DOC/'BOOTSTRAP_DRAWS.npy').exists():
        frozen = C.read(C.DOC/'BOOTSTRAP_LOCK.json')
        assert frozen['sessions'] == names
        assert frozen['sha256'] == C.sha(C.DOC/'BOOTSTRAP_DRAWS.npy')
        draw = np.load(C.DOC/'BOOTSTRAP_DRAWS.npy', allow_pickle=False)
        assert draw.shape == (10000, 9)
    else:
        draw = paired.save_draws(sessions)
    weights = np.stack([(draw == i).sum(1) for i in range(len(names))], axis=1)
    source_files = ['C_FRAME_METRICS.jsonl.gz', 'D_REPAIRED_ROWS.jsonl.gz',
                    'AE_CANDIDATE_METRICS.jsonl.gz']
    records = [normalize(r, 'C') for r in C.rows(C.DOC/source_files[0])]
    records += [normalize(r, 'D') for r in C.rows(C.DOC/source_files[1])]
    for r in C.rows(C.DOC/source_files[2]):
        p = r['candidate_metrics'].get(r['hyp_current'], dict(available=False))
        record = dict(r, pose=p, axis_wrong=None if r['hyp_reference'] is None or r['hyp_current'] is None
                      else int(r['hyp_reference'] != r['hyp_current']))
        records.append(normalize(record, 'historical_candidate'))
    # N3 baseline seed1 is present in both D and historical sources: keep each
    # source in provenance, but never count duplicated rows in an aggregate.
    unique = {}
    for r in records:
        key = (r['backbone'], r['method'], r['id'], r['seed'])
        if key in unique:
            old = unique[key]
            for field in ['available', 'pose_success', 'wd_confusion', 'translation_cm', 'rotation_deg']:
                a, b = old[field], r[field]
                assert (a is None and b is None) or abs(a-b) < 1e-10, (key, field)
        else:
            unique[key] = r
    records = sorted(unique.values(), key=lambda r: (r['backbone'],r['method'],r['id'],r['seed']))
    assert all(r['id'] in set(ids) for r in records)
    out = C.DOC/'FRAME_METRICS.jsonl.gz'
    assert not out.exists()
    with gzip.open(out, 'wt') as f:
        for r in records:
            import json
            f.write(json.dumps(C.clean(r), ensure_ascii=False, allow_nan=False, separators=(',',':'))+'\n')
    grouped = defaultdict(list)
    for r in records:
        grouped[r['backbone'],r['method']].append(r)
    spec = importlib.util.spec_from_file_location('fast_frozen_auc', C.SOURCE/'scripts/paper/pose_metric_closure_v1/symmetry_aware_pose_metrics.py')
    auc = importlib.util.module_from_spec(spec); spec.loader.exec_module(auc)
    summary = []
    for (backbone, method), rr in grouped.items():
        for population, subset in [('REAL173',set(ids)),('RECT245_INTERSECTION110',set(contract['rect245_intersection_ids']))]:
            by = defaultdict(list)
            for r in rr:
                if r['id'] in subset: by[r['id']].append(r)
            assert set(by) == subset
            metrics = {}
            for key in ['translation_cm','rotation_deg','yaw_deg','ADDsym_m','IoU3D',
                        'available','pose_success','wd_confusion','axis_wrong']:
                frame_values = []; cluster = []
                for fid in sorted(by):
                    values = [r[key] for r in by[fid] if r[key] is not None]
                    frame_values.append(float(np.mean(values)) if values else np.nan)
                    cluster.append(fid.split(':')[0])
                metrics[key] = moments(frame_values)
                total = np.array([sum(v for v,s in zip(frame_values,cluster) if s==name and np.isfinite(v)) for name in names])
                count = np.array([sum(s==name and np.isfinite(v) for v,s in zip(frame_values,cluster)) for name in names])
                denominator = weights @ count; valid = denominator > 0
                values = (weights[valid] @ total) / denominator[valid]
                metrics[key]['CI95_mean_session_cluster'] = np.quantile(values,[.025,.975]).tolist() if len(values) else None
                metrics[key]['empty_bootstrap_draws'] = int((~valid).sum())
            adds = [r['ADDsym_normalized'] if r['ADDsym_normalized'] is not None else np.inf for r in rr if r['id'] in subset]
            summary.append(dict(population=population,backbone=backbone,method=method,
                frames=len(by), prediction_rows=sum(map(len,by.values())),seeds=sorted({r['seed'] for r in rr}),
                metrics=metrics, ADDsym_AUC_full=auc.pose_auc(adds,1.0),
                ADDsym_AUC_seed_then_average=True, aggregation='seed mean within each frame; sample SD across frames, ddof1; missing binary=0; available-only continuous'))
    primary = []
    for after, before in paired.CONTRASTS:
        a = {(r['id'],r['seed']):r for r in grouped['YOLO',after]}
        b = {(r['id'],r['seed']):r for r in grouped['YOLO',before]}
        assert set(a)==set(b)
        seeds = sorted({seed for fid,seed in a})
        for metric in paired.METRICS:
            aa = [[a[fid,seed][metric] for seed in seeds] for fid in ids]
            bb = [[b[fid,seed][metric] for seed in seeds] for fid in ids]
            effect = paired.bootstrap_effect(bb,aa,sessions,draw)
            primary.append(dict(after=after,before=before,metric=metric,**effect))
    adjusted = paired.holm([r['p_raw'] for r in primary])
    for r,p in zip(primary,adjusted): r['p_Holm']=p
    C.write(C.DOC/'PAIRED.json',dict(status='DONE_EXPLORATORY_FIXED_EIGHT_HYPOTHESES',results=primary,
        family_size=8,bootstrap=C.binding(C.DOC/'BOOTSTRAP_LOCK.json'),all_methods_reported=True))
    C.write(C.DOC/'METRICS.json',dict(status='DONE', summaries=summary,raw_rows=C.binding(out),
        inputs=[C.binding(C.DOC/name) for name in source_files],frames=173,axis_reference_frames=170,
        uncertainty='9 sessions, repeatedly exposed development references; no independent final claim',
        first_failed_D_rows='Preserved D_ROWS.jsonl.gz; excluded from scientific metrics as adapter type error',
        raw_rows_provenance_deduplicated=True))
    print('AGGREGATED',len(records),'rows',len(summary),'summaries',len(primary),'primary effects')


if __name__=='__main__': run()
