"""Synthetic-only matched Direct/DHT side-outline experiment and artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'hough_attention_transfer_v1'))
import core as C
from dht import SparseDHT
from network import DirectSide, DeepHoughSide, target_distribution, line_loss, decode
import targets as T


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


class Data:
    def __init__(self, source, device):
        cache = json.loads((source / 'CACHE.json').read_text())
        expected = dict(config_sha256=C.sha(source / 'CONFIG.json'),
                        manifest_sha256=C.sha(source / 'manifest.json'))
        assert cache['signature'] == expected, 'Source cache provenance mismatch'
        self.records, self.populations = [], {}
        for pop, rows in json.loads((source / 'manifest.json').read_text())['populations'].items():
            self.populations[pop] = []
            for row in rows:
                i = len(self.records)
                self.records.append(dict(row, population=pop, index=i))
                self.populations[pop].append(i)
        self.features = np.load(source / 'features.npy', mmap_mode='r')
        assert self.features.shape == (len(self.records), 128, 50, 50)
        grids = np.load(source / 'targets.npz')['grids']
        theta, rho, support = T.make_targets(self.records, grids)
        self.facing = T.facing_side_support(self.records) & support
        self.theta, self.rho, self.support = [torch.from_numpy(a).to(device) for a in (theta, rho, support)]
        self.device = device

    def batch(self, indices):
        features = torch.from_numpy(np.asarray(self.features[indices])).to(self.device, dtype=torch.float32)
        return features, self.theta[indices], self.rho[indices], self.support[indices]


def model_for(arm, seed, lattice, dh, device):
    return (DirectSide(dh, lattice, seed) if arm == 'Direct' else DeepHoughSide(lattice, seed)).to(device)


def summary(rows):
    out = {'n_roles': len(rows), 'n_frames': len({r['id'] for r in rows})}
    for key in ('angle_deg', 'distance_px', 'distance_diagonal'):
        vals = [r[key] for r in rows]
        if vals:
            out[key] = dict(median=float(np.median(vals)), p90=float(np.percentile(vals, 90)), mean=float(np.mean(vals)))
    if rows:
        out['success_5deg_8px'] = float(np.mean([r['angle_deg'] <= 5 and r['distance_px'] <= 8 for r in rows]))
    return out


@torch.no_grad()
def evaluate(data, model, indices, cfg, lattice, arm, seed):
    model.eval()
    rows = []
    for start in range(0, len(indices), cfg['batch']):
        idx = indices[start:start + cfg['batch']]
        features, _, _, support = data.batch(idx)
        theta, rho = [v.cpu().numpy() for v in decode(model(features), lattice)]
        for b, i in enumerate(idx):
            rec = data.records[i]
            lines = T.line_pixels(theta[b], rho[b], rec['width'], rec['height'])
            angle, distance = T.pixel_errors(lines, rec['gt_points'])
            for role in np.flatnonzero(support[b].cpu().numpy()):
                rows.append(dict(id=rec['id'], population=rec['population'], group=rec.get('group', ''),
                                 arm=arm, seed=seed, role=int(role), role_name=T.ROLE_NAMES[role],
                                 camera_facing=bool(data.facing[i, role]), angle_deg=float(angle[role]),
                                 distance_px=float(distance[role]), distance_diagonal=float(distance[role] / np.hypot(rec['width'], rec['height']))))
    return rows


def check(data, cfg, lattice, dh, out):
    features, theta, rho, support = data.batch(data.populations['synth_train'][:2])
    target = target_distribution(theta, rho, lattice)
    assert torch.allclose(target.sum(-1), torch.ones_like(theta), atol=1e-5)
    info = {}
    for arm in ('Direct', 'DHT'):
        model = model_for(arm, 101, lattice, dh, data.device)
        scores = model(features)
        assert scores.shape == (2, 8, lattice.theta_bins * lattice.rho_bins)
        loss = line_loss(scores, target, support, lattice)
        loss.backward()
        assert torch.isfinite(loss)
        norms = [float(p.grad.norm()) for p in model.parameters() if p.grad is not None]
        assert max(norms) > 0 and all(np.isfinite(norms))
        info[arm] = {'parameters': sum(p.numel() for p in model.parameters()),
                     'loss': float(loss), 'max_gradient_norm': max(norms)}
    # Verify geometry discretization independently of learned weights.
    oracle_rows = []
    for start in range(0, len(data.records), 12):
        idx = list(range(start, min(start + 12, len(data.records))))
        target = target_distribution(data.theta[idx], data.rho[idx], lattice)
        assert torch.allclose(target.sum(-1)[data.support[idx]], torch.ones_like(data.theta[idx][data.support[idx]]), atol=1e-5)
        t, r = [v.cpu().numpy() for v in decode(target, lattice)]
        for b, i in enumerate(idx):
            rec = data.records[i]
            angle, dist = T.pixel_errors(T.line_pixels(t[b], r[b], rec['width'], rec['height']), rec['gt_points'])
            for role in np.flatnonzero(data.support[i].cpu().numpy()):
                oracle_rows.append(dict(id=rec['id'], population=rec['population'], angle_deg=float(angle[role]),
                                        distance_px=float(dist[role]), distance_diagonal=float(dist[role] / np.hypot(rec['width'], rec['height']))))
    info['target_peak_oracle'] = {p: summary([r for r in oracle_rows if r['population'] == p]) for p in data.populations}
    info['PASS'] = True
    write(out / 'CHECKS.json', info)
    log(f'Checks passed: {info["Direct"]}, {info["DHT"]}')


def train(data, cfg, lattice, dh, out, arm, seed, sanity=False):
    name = f'{arm}_seed{seed}' if not sanity else 'DHT_sanity32_seed101'
    path = out / 'checkpoints' / f'{name}_final.pth'
    if path.exists():
        saved = torch.load(path, map_location='cpu')
        assert saved['config_sha256'] == C.sha(out / 'CONFIG.json')
        log(f'Reusing completed {name}')
        return
    model = model_for(arm, seed, lattice, dh, data.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    rng = np.random.default_rng(seed)
    indices = data.populations['synth_train'][:32] if sanity else data.populations['synth_train']
    steps = 1500 if sanity else cfg['steps']
    history, losses = [], []
    start = time.monotonic()
    model.train()
    for step in range(1, steps + 1):
        idx = rng.choice(indices, cfg['batch'], replace=False).tolist()
        features, theta, rho, support = data.batch(idx)
        target = target_distribution(theta, rho, lattice)
        loss = line_loss(model(features), target, support, lattice)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        if not np.isfinite(value):
            raise RuntimeError(f'Nonfinite loss at {name} step {step}')
        losses.append(value)
        if step == 1 or step % 500 == 0 or step == steps:
            row = dict(step=step, loss_mean=float(np.mean(losses)), elapsed_sec=time.monotonic() - start)
            history.append(row)
            log(f'{name} step {step}/{steps} loss {row["loss_mean"]:.4f} elapsed {row["elapsed_sec"]:.1f}s')
            write(out / f'{name}_history.json', history)
            losses.clear()
    torch.save(dict(model=model.state_dict(), arm=arm, seed=seed, steps=steps,
                    config_sha256=C.sha(out / 'CONFIG.json'), config=cfg,
                    script_sha256={p.name: C.sha(p) for p in HERE.glob('*.py')}), path)
    if sanity:
        rows = evaluate(data, model, indices, cfg, lattice, arm, seed)
        report = summary(rows)
        report['loss_first'] = history[0]['loss_mean']
        report['loss_last'] = history[-1]['loss_mean']
        report['loss_decreased'] = history[-1]['loss_mean'] < history[0]['loss_mean']
        report['PASS'] = (report['loss_decreased'] and report['distance_px']['median'] <= 8
                          and report['angle_deg']['median'] <= 5)
        write(out / 'SANITY.json', report)
        log(f'Sanity {report}')
        assert report['PASS'], 'Sanity does not fit its 32 training frames'


def load_model(out, arm, seed, lattice, dh, device):
    saved = torch.load(out / 'checkpoints' / f'{arm}_seed{seed}_final.pth', map_location='cpu')
    assert saved['config_sha256'] == C.sha(out / 'CONFIG.json')
    model = model_for(arm, seed, lattice, dh, device)
    model.load_state_dict(saved['model'], strict=True)
    return model.eval()


def evaluate_all(data, cfg, lattice, dh, out):
    rows, by_run = [], {}
    for seed in cfg['seeds']:
        for arm in ('Direct', 'DHT'):
            model = load_model(out, arm, seed, lattice, dh, data.device)
            result = {}
            for pop, indices in data.populations.items():
                these = evaluate(data, model, indices, cfg, lattice, arm, seed)
                rows.extend(these)
                result[pop] = summary(these)
                result[pop]['camera_facing'] = summary([r for r in these if r['camera_facing']])
                result[pop]['by_role'] = {name: summary([r for r in these if r['role'] == role]) for role, name in enumerate(T.ROLE_NAMES)}
                log(f'{arm} seed{seed} {pop} median {result[pop]["distance_px"]["median"]:.2f}px')
            by_run[f'{arm}_seed{seed}'] = result
            del model
    with (out / 'PER_ROLE.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    seed_summary = {}
    for arm in ('Direct', 'DHT'):
        seed_summary[arm] = {}
        for pop in data.populations:
            entries = [by_run[f'{arm}_seed{s}'][pop] for s in cfg['seeds']]
            merged = {}
            for metric in ('angle_deg', 'distance_px'):
                for stat in ('median', 'p90'):
                    vals = [entry[metric][stat] for entry in entries]
                    merged[f'{metric}_{stat}_mean_across_seeds'] = float(np.mean(vals))
                    merged[f'{metric}_{stat}_range_across_seeds'] = [min(vals), max(vals)]
                    merged[f'camera_facing_{metric}_{stat}_mean_across_seeds'] = float(np.mean([entry['camera_facing'][metric][stat] for entry in entries]))
            merged['success_5deg_8px_mean_across_seeds'] = float(np.mean([entry['success_5deg_8px'] for entry in entries]))
            seed_summary[arm][pop] = merged
    write(out / 'RESULTS.json', {'config': cfg, 'by_run': by_run, 'seed_summary': seed_summary,
                               'aggregation': 'Each seed pools supported lines within population; reported seed summary averages those statistics.'})


def select_samples(data):
    def ordered(indices):
        return sorted(indices, key=lambda i: hashlib.sha256(('dht-side-v1-gallery:' + data.records[i]['id']).encode()).hexdigest())
    chosen = []
    for pop in ('synth_test', 'cross_v4'):
        chosen.extend(ordered(data.populations[pop])[:3])
    groups = sorted({data.records[i].get('group', '') for i in data.populations['real_dev']})
    for group in groups:
        chosen.extend(ordered([i for i in data.populations['real_dev'] if data.records[i].get('group', '') == group])[:2])
    return chosen


def export_visuals(data, cfg, lattice, dh, out):
    direct = load_model(out, 'Direct', 1, lattice, dh, data.device)
    dht = load_model(out, 'DHT', 1, lattice, dh, data.device)
    samples = []
    dest = out / 'artifacts'
    dest.mkdir(exist_ok=True)
    for i in select_samples(data):
        rec = data.records[i]
        features, theta, rho, support = data.batch([i])
        with torch.no_grad():
            dt, dr = [v[0].cpu().numpy() for v in decode(direct(features), lattice)]
        features.requires_grad_(True)
        scores = dht(features).masked_fill(~lattice.valid.flatten()[None, None], -1e9)
        ht, hr = [v[0].detach().cpu().numpy() for v in decode(scores, lattice)]
        probability = scores.softmax(-1)[0].detach().cpu().numpy().reshape(8, lattice.theta_bins, lattice.rho_bins)
        sensitivity = []
        for role in range(8):
            scalar = scores[0, role, scores[0, role].detach().argmax()]
            grad = torch.autograd.grad(scalar, features, retain_graph=role < 7)[0]
            sensitivity.append((grad * features).abs().sum(1)[0].detach().cpu().numpy())
        dl = T.line_pixels(dt, dr, rec['width'], rec['height'])
        hl = T.line_pixels(ht, hr, rec['width'], rec['height'])
        da, dd = T.pixel_errors(dl, rec['gt_points'])
        ha, hd = T.pixel_errors(hl, rec['gt_points'])
        stem = f'{rec["population"]}_{i:04d}'
        picture = cv2.imread(rec['image'])
        assert picture is not None
        picture = cv2.copyMakeBorder(picture, 100, 100, 100, 100, cv2.BORDER_REFLECT_101)
        assert cv2.imwrite(str(dest / f'{stem}.png'), picture)
        np.savez_compressed(dest / f'{stem}.npz', gt_points=np.asarray(rec['gt_points']) + 100,
                            support=support[0].cpu().numpy(), facing=data.facing[i],
                            direct_lines=dl + 100, dht_lines=hl + 100,
                            direct_angle=da, dht_angle=ha, direct_distance=dd, dht_distance=hd,
                            dht_probability=probability, theta_degrees=lattice.theta_degrees.cpu().numpy(),
                            rho_values=lattice.rho_values.cpu().numpy(), gt_theta=theta[0].cpu().numpy(),
                            gt_rho=rho[0].cpu().numpy(), pred_theta=ht, pred_rho=hr,
                            feature_sensitivity=np.stack(sensitivity))
        samples.append(dict(id=rec['id'], population=rec['population'], image=str(dest / f'{stem}.png'),
                            original_image=rec['image'], pad_px=100, artifact=f'artifacts/{stem}.npz'))
        log(f'Visualization exported {stem}')
    write(out / 'visualization_manifest.json', dict(edges=T.SIDE_EDGES, role_names=T.ROLE_NAMES, samples=samples,
          summary='RESULTS.json', selection='SHA256 fixed prefix dht-side-v1-gallery, 3 synth_test, 3 cross_v4, 2 per real group; independent of predictions',
          sensitivity='Absolute gradient times input VGG feature, summed across channels, for winning predicted pre-softmax logit of each role; feature-level sensitivity proxy, not attention or causal attribution.'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=C.ROOT / 'data/pallet/results/hough_attention_transfer_v1')
    parser.add_argument('--run-dir', type=Path, default=C.ROOT / 'data/pallet/results/deep_hough_side_v1')
    parser.add_argument('--phase', choices=['check', 'sanity', 'train', 'evaluate', 'export', 'all'], default='all')
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--batch', type=int, default=12)
    args = parser.parse_args()
    out, source = args.run_dir.resolve(), args.source.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'checkpoints').mkdir(exist_ok=True)
    source_cfg = json.loads((source / 'CONFIG.json').read_text())
    cfg = dict(schema='deep_hough_side_v1', steps=args.steps, seeds=args.seeds, batch=args.batch, lr=.001, weight_decay=.0001,
               source_cache=str(source), source_manifest_sha256=C.sha(source / 'manifest.json'),
               backbone=source_cfg['backbone'], backbone_sha256=source_cfg['backbone_sha256'],
               backbone_frozen=True, real_used_for_training=False, pad_px=100, input_size=400, features=[128, 50, 50],
               coordinate_mapping='feature=(original+pad+.5)*50/padded_size-.5; VGG stride8 centers',
               edges=T.SIDE_EDGES, role_names=T.ROLE_NAMES, theta_bins=180, rho_step=.5, origin=[24.5, 24.5],
               dht_channels=16, dht_normalize=True, target_angle_sigma_deg=1., target_rho_sigma_cell=.5,
               selection='fixed final checkpoint, no real selection', target='eight structural side-outline supporting lines, includes hidden edges',
               method='task-adapted feature DHT; not exact published multiscale ResNet-FPN reproduction',
               evaluation='Existing manual real DEV52, not independent final test; facing subset means geometry, not physical visibility',
               pretrain_overlap='Inherited synthetic backbone pretraining manifest unavailable; overlap with synthetic holdout unverified')
    cfg = json.loads(json.dumps(cfg))
    if (out / 'CONFIG.json').exists():
        assert json.loads((out / 'CONFIG.json').read_text()) == cfg, 'Existing configuration differs'
    else:
        write(out / 'CONFIG.json', cfg)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    dh = C.load_dh()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lattice = SparseDHT().to(device)
    data = Data(source, device)
    write(out / 'PROVENANCE.json', dict(scripts={p.name: C.sha(p) for p in HERE.glob('*.py')},
          source_cache=json.loads((source / 'CACHE.json').read_text()), torch=torch.__version__, device=str(device),
          gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
          populations={p: len(v) for p, v in data.populations.items()}))
    phase = args.phase
    if phase in ('check', 'all'):
        check(data, cfg, lattice, dh, out)
    if phase in ('sanity', 'all'):
        train(data, cfg, lattice, dh, out, 'DHT', 101, sanity=True)
    if phase in ('train', 'all'):
        assert json.loads((out / 'SANITY.json').read_text())['PASS'], 'Passing sanity required'
        for seed in cfg['seeds']:
            for arm in ('Direct', 'DHT'):
                train(data, cfg, lattice, dh, out, arm, seed)
    if phase in ('evaluate', 'all'):
        evaluate_all(data, cfg, lattice, dh, out)
    if phase in ('export', 'all'):
        export_visuals(data, cfg, lattice, dh, out)


if __name__ == '__main__':
    main()
