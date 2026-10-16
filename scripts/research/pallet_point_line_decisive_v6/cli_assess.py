"""Score a split with a frozen checkpoint, then evaluate and compare.

The inference rule is the configuration itself: no calibration margin is chosen
after the fact. Ground truth is opened only in `evaluate`, never in `score`.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch

KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT))
from plpose_local.heads import PoseNet                                  # noqa: E402
from plpose_local import solver                                          # noqa: E402
from plpose_local.geometry import corners, project, symmetry_permutations, yaw  # noqa: E402
from cli_train import Split, sha                                         # noqa: E402


def score(args):
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f'refusing to overwrite scores: {out}')
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = ck['config']
    model = PoseNet(arm=ck['arm'], channels=cfg['channels'], width=cfg['width'],
                    starts=cfg['starts'], thetas=cfg['thetas'], rhos=cfg['rhos'],
                    modes=cfg['modes'])
    model.load_state_dict(ck['state_dict'], strict=True)
    model.to(args.device).eval()
    data = Split(args.manifest)
    rows = []
    loads = []
    original_load = torch.load
    def audited_load(path, *a, **kw):
        loads.append(str(Path(path).resolve()))
        return original_load(path, *a, **kw)
    torch.load = audited_load
    with torch.inference_mode():
        for i in range(len(data.records)):
            O, _, _ = data.batch([i], args.device, targets=False)   # GT is never opened here
            o = model(O)
            diag = torch.linalg.vector_norm(O['image_hw'].flip(-1), dim=-1)
            b, S = o['R0'].shape[0], o['R0'].shape[1]
            rep = lambda x: x[:, None].expand(b, S, *x.shape[1:]).reshape(b * S, *x.shape[1:])
            flat = lambda x: x.reshape(b * S, *x.shape[2:])
            lines = o['lines']
            Rr, tr = solver.refine(rep(O['K']), flat(o['R0']), flat(o['t0']), rep(O['dims']),
                                   rep(o['points']), rep(O['point_mask']),
                                   rep(lines) if lines is not None else None,
                                   rep(o['mass']) if lines is not None else None,
                                   rep(O['line_mask']) if lines is not None else None,
                                   rep(diag), iters=cfg['iters'])
            Rs, ts = Rr.reshape(b, S, 3, 3), tr.reshape(b, S, 3)
            Rb, tb, e, best, valid = solver.select(
                O['K'], Rs, ts, O['dims'], o['points'], O['point_mask'],
                lines, o['mass'], O['line_mask'], diag)
            rows.append(dict(id=data.records[i]['id'], session=data.records[i]['session'],
                             asset=data.records[i]['asset'], order=data.records[i]['symmetry']['order'],
                             points=o['points'][0].cpu().tolist(),
                             R=Rb[0].cpu().tolist(), t=tb[0].cpu().tolist(),
                             candidate_energy=e[0].cpu().tolist(),
                             candidate_selected=int(best[0]), pose_valid=bool(valid[0])))
    torch.load = original_load
    forbidden = {str((data.root/r['supervision']).resolve()) for r in data.records}
    opened = len([p for p in loads if p in forbidden])
    if opened:
        raise RuntimeError('GT supervision opened during scoring')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(schema='plpose_local_predictions_1', arm=ck['arm'], seed=ck['seed'],
                                   steps=ck['steps'], checkpoint_sha256=sha(args.checkpoint),
                                   manifest_sha256=sha(args.manifest), supervision_files_opened=opened,
                                   observation_loads=len(loads), load_audit_sha256=hashlib.sha256('\n'.join(loads).encode()).hexdigest(),
                                   rows=rows), ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(f'{ck["arm"]} seed{ck["seed"]}: scored {len(rows)}')


def evaluate(args):
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f'refusing to overwrite evaluation: {out}')
    data = Split(args.manifest)
    pred = json.loads(Path(args.predictions).read_text())
    if pred['manifest_sha256'] != sha(args.manifest):
        raise SystemExit('predictions refer to another population')
    if len(pred['rows']) != len(data.records):
        raise SystemExit('prediction count differs; refusing zip truncation')
    per, failures = [], 0
    for rec, row in zip(data.records, pred['rows']):
        if rec['id'] != row['id']:
            raise SystemExit('prediction/manifest order mismatch')
        o = torch.load(data.root / rec['observation'], map_location='cpu', weights_only=True)
        s = torch.load(data.root / rec['supervision'], map_location='cpu', weights_only=True)
        diag = float(torch.linalg.vector_norm(o['image_hw'].flip(-1), dim=-1))
        gt_uv, _ = project(o['K'], s['R'], s['t'], corners(o['dims']))
        if not row['pose_valid']:
            failures += 1
            per.append(dict(id=rec['id'], session=rec['session'], asset=rec['asset'],
                            pose_valid=False, normalized=1.0, raw_px=None)); continue
        R = torch.tensor(row['R'], dtype=torch.float32)[None]
        t = torch.tensor(row['t'], dtype=torch.float32)[None]
        best = None
        for g, perm in enumerate(symmetry_permutations(rec['symmetry']['order'])):
            Rg = R @ yaw(g * (4 // rec['symmetry']['order'])).T
            uv, _ = project(o['K'], Rg, t, corners(o['dims']))
            err = torch.linalg.vector_norm(uv[0, :8] - gt_uv[0, :8], dim=-1)
            m = float((err / diag).clamp(max=1.).mean())
            if best is None or m < best[0]:
                best = (m, float(err.mean()))
        per.append(dict(id=rec['id'], session=rec['session'], asset=rec['asset'],
                        pose_valid=True, normalized=best[0], raw_px=best[1]))
    conditional_per = list(per)
    ledger = json.loads((data.root/'DETECTION_LEDGER.json').read_text())
    split = data.doc['split']
    for missing in ledger['failures']:
        if missing['split'] == split:
            failures += 1
            per.append(dict(id=missing['id'],session='MISSING_DETECTION',asset='UNKNOWN',
                pose_valid=False,normalized=1.,raw_px=None,reason=missing['reason']))
    norm = np.array([p['normalized'] for p in per])
    px = np.array([p['raw_px'] for p in per if p['raw_px'] is not None])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        schema='plpose_local_evaluation_1', arm=pred['arm'], seed=pred['seed'],
        manifest_sha256=pred['manifest_sha256'], predictions_sha256=sha(args.predictions),
        frames=len(per), pose_failures=failures, coverage=float((len(per) - failures) / max(1, len(per))),
        primary_normalized_mean=float(norm.mean()),
        detection_conditioned_frames=len(conditional_per),
        detection_conditioned_normalized_mean=float(np.mean([p['normalized'] for p in conditional_per])),
        conditional_raw_px=dict(median=float(np.median(px)) if len(px) else None,
                                p90=float(np.quantile(px, .9)) if len(px) else None,
                                mean=float(px.mean()) if len(px) else None, n=int(len(px))),
        metric_note='One effective symmetry element per frame, 8 corners, mean 2D error divided by the '
                    'raw image diagonal, capped at 1 per point; a missing pose counts as 1. This is a '
                    'transparent pilot metric and is not the repository MAIN6D or a BOP score.',
        per_frame=per), ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(f'{pred["arm"]} seed{pred["seed"]}: primary={norm.mean():.6f} coverage={1-failures/len(per):.4f}')


def compare(args):
    ref = [json.loads(Path(p).read_text()) for p in args.reference]
    met = [json.loads(Path(p).read_text()) for p in args.method]
    ids = [f['id'] for f in ref[0]['per_frame']]
    for d in ref + met:
        if [f['id'] for f in d['per_frame']] != ids:
            raise SystemExit('populations differ')
    R = np.array([[f['normalized'] for f in d['per_frame']] for d in ref])
    M = np.array([[f['normalized'] for f in d['per_frame']] for d in met])
    checks = []
    for j in range(len(ref)):
        rm, mm = R[j].mean(), M[j].mean()
        rc, mc = ref[j]['coverage'], met[j]['coverage']
        rp, mp = ref[j]['conditional_raw_px'], met[j]['conditional_raw_px']
        new_cat = sum(bool(r['pose_valid'] and r['raw_px'] <= 10. and
                           (not m['pose_valid'] or m['raw_px'] > 50.))
                      for r, m in zip(ref[j]['per_frame'], met[j]['per_frame']))
        checks.append(dict(seed_slot=j + 1, reference_primary=float(rm), method_primary=float(mm),
                           relative_gain=float((rm - mm) / rm) if rm > 0 else 0.,
                           beats_by_1pct=bool(mm <= .99 * rm),
                           median_nonworse=bool(mp['median'] is not None and rp['median'] is not None
                                                and mp['median'] <= rp['median'] + 1e-9),
                           p90_nonworse=bool(mp['p90'] is not None and rp['p90'] is not None
                                             and mp['p90'] <= rp['p90'] + 1e-9),
                           coverage_nonworse=bool(mc >= rc - 1e-12),
                           new_catastrophic_frames=new_cat))
    diff = M.mean(0) - R.mean(0)
    sess = np.array([f['session'] for f in ref[0]['per_frame']])
    names = np.unique(sess)
    rng = np.random.default_rng(20260909)
    sums = np.array([diff[sess == g].sum() for g in names])
    cnts = np.array([(sess == g).sum() for g in names])
    boot = []
    for start in range(0, 20000, 1000):
        idx = rng.integers(0, len(names), (1000, len(names)))
        boot.extend((sums[idx].sum(1) / cnts[idx].sum(1)).tolist())
    lo, hi = np.quantile(boot, [.025, .975])
    numeric = all(all(v for k, v in c.items() if isinstance(v, bool)) and c['new_catastrophic_frames'] == 0
                  for c in checks)
    Path(args.output).write_text(json.dumps(dict(
        schema='plpose_local_comparison_1', reference_arm=ref[0]['arm'], method_arm=met[0]['arm'],
        checks=checks, seed_averaged_difference_mean=float(diff.mean()),
        session_bootstrap=dict(lower=float(lo), upper=float(hi), sessions=int(len(names)),
                               draws=20000, seed=20260909,
                               caveat='Synthetic renderer groups were never shown to be independent '
                                      'units; this interval is descriptive, not a generalisation claim.'),
        numeric_gate_passed=bool(numeric), scientific_success=None,
        note='All three seeds must satisfy every check for the numeric gate. A passed gate is a candidate '
             'for further review, not an adoption decision.'), ensure_ascii=False, indent=2) + '\n',
        encoding='utf8')
    print(json.dumps(dict(reference=ref[0]['arm'], method=met[0]['arm'], numeric_gate_passed=numeric,
                          mean_difference=float(diff.mean()), ci=[float(lo), float(hi)])))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('score'); s.add_argument('--manifest', required=True)
    s.add_argument('--checkpoint', required=True); s.add_argument('--output', required=True)
    s.add_argument('--device', default='cuda:0'); s.set_defaults(fn=score)
    e = sub.add_parser('evaluate'); e.add_argument('--manifest', required=True)
    e.add_argument('--predictions', required=True); e.add_argument('--output', required=True)
    e.set_defaults(fn=evaluate)
    c = sub.add_parser('compare'); c.add_argument('--reference', nargs=3, required=True)
    c.add_argument('--method', nargs=3, required=True); c.add_argument('--output', required=True)
    c.set_defaults(fn=compare)
    a = ap.parse_args(); a.fn(a)
