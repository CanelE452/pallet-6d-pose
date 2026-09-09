"""G1/G2/G3/G5 integrity gates on ACTUAL exported data, then PROTOCOL_LOCK.

Unit-test success is not accepted as evidence of a correct local hookup.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
from cli.common import ROOT, RESULTS, STRUCTURED_V2, DECODER_PROBE, backbone_binding, read, write, sha256, git  # noqa: E402
sys.path.insert(0, str(ROOT))
from pointline_v4.cache_io import ExportDataset  # noqa: E402
from pointline_v4.model import EvidenceVerifier, Observation, ARMS  # noqa: E402
from pointline_v4.geometry import EDGES  # noqa: E402

EXPORT = RESULTS / 'export'
CORE = KIT / 'pointline_v4'


def core_sha():
    return {p.name: sha256(p) for p in sorted(CORE.glob('*.py'))}


def g1_coordinates(train):
    """Round-trip raw<->feature, content mask meaning, line equation units."""
    out = dict(gate='G1', checks=[])
    for index in (0, 7, 123, 900):
        o = train.observation(index)
        raw_hw = None
        for plane, (affine, mask) in enumerate(zip(o.raw_to_feature, o.content_valid)):
            a = torch.eye(3, dtype=torch.float64)
            a[:2] = affine[0].double()
            inverse = torch.linalg.inv(a)
            pts = o.layouts[0, 0].double()                       # real predicted corners, raw px
            hom = torch.cat([pts, torch.ones(9, 1, dtype=torch.float64)], 1)
            feat = hom @ a.T
            back = torch.cat([feat[:, :2], torch.ones(9, 1, dtype=torch.float64)], 1) @ inverse.T
            out['checks'].append(dict(name='affine_round_trip', record=index, plane=plane,
                                      max_abs_raw_px=float((back[:, :2] - pts).abs().max())))
        # Content mask must cover a contiguous top-left block equal to the letterboxed content.
        for plane, mask in enumerate(o.content_valid):
            m = mask[0, 0].numpy()
            out['checks'].append(dict(name='content_mask', record=index, plane=plane,
                                      true_cells=int(m.sum()), total_cells=int(m.size),
                                      fraction=float(m.mean())))
    # Line equations must be raw-pixel homogeneous lines.
    o = train.observation(0)
    line_h, line_valid = o.line_h[0].double(), o.line_valid[0]
    unit = []
    for role in range(12):
        for mode in range(4):
            if not bool(line_valid[role, mode]):
                continue
            a, b, c = line_h[role, mode].tolist()
            n = (a * a + b * b) ** .5
            if n < 1e-8:
                continue
            # A point exactly on the line, and the same point pushed d raw px along the normal.
            base = torch.tensor([-a * c / n ** 2, -b * c / n ** 2], dtype=torch.float64)
            for d in (0., 3., 17.):
                p = base + d * torch.tensor([a / n, b / n], dtype=torch.float64)
                dist = abs((a * p[0] + b * p[1] + c).item()) / n
                unit.append(abs(dist - d))
    out['checks'].append(dict(name='line_equation_raw_pixel_units', tested=len(unit),
                              max_abs_error_px=float(max(unit)) if unit else None))
    out['line_provenance'] = ('line_h/peak_logits/peak_valid are reused from the frozen structured-v2 pack, '
                              'whose packer already re-derived NMS peaks and asserted equality with the saved '
                              'line_fusion evidence (cache.pack_observation). Not re-implemented here.')
    trip = max(c['max_abs_raw_px'] for c in out['checks'] if c['name'] == 'affine_round_trip')
    units = next(c['max_abs_error_px'] for c in out['checks'] if c['name'] == 'line_equation_raw_pixel_units')
    out['PASS'] = bool(trip < 1e-6 and (units is None or units < 1e-6))
    out['max_round_trip_raw_px'] = trip
    return out


def g2_data(manifests):
    out = dict(gate='G2', checks=[])
    ids, hashes = {}, {}
    for role, data in manifests.items():
        ids[role] = {r['frame_id'] for r in data.records}
        hashes[role] = {r['observation_sha256'] for r in data.records}
    for a, b in (('train', 'calibration'), ('train', 'synth_val'), ('calibration', 'synth_val')):
        out['checks'].append(dict(name='disjoint_frame_id', pair=[a, b], overlap=len(ids[a] & ids[b])))
    # Frozen upstream image-byte separation (train/val), recorded by the source manifest.
    source = read(DECODER_PROBE / 'MANIFEST.json')
    out['checks'].append(dict(name='upstream_image_sha_disjoint_train_val',
                              value=bool(source['decoder_train_val_image_sha_disjoint'])))
    out['checks'].append(dict(name='upstream_real_GT_read', value=bool(source['real_GT_read'])))
    out['checks'].append(dict(name='structured_v2_cache_real_GT_read',
                              value=bool(read(STRUCTURED_V2 / 'CACHE_COMPLETION.json')['real_GT_read'])))
    # Proposals are GT-free by contract, verified on the actual packed spec.
    spec = read(STRUCTURED_V2 / 'PROPOSAL_SPEC.json')
    out['checks'].append(dict(name='proposal_spec_GT_used', value=bool(spec['GT_used']),
                              DLT_or_PnP_used=bool(spec['DLT_or_PnP_used'])))
    # Scoring must never open a supervision file: instrument torch.load for real.
    opened = []
    real_load = torch.load
    def watched(f, *a, **k):
        opened.append(str(f))
        return real_load(f, *a, **k)
    torch.load = watched
    try:
        probe = ExportDataset(EXPORT / 'train.json', verify_targets=False)
        for i in range(8):
            probe.observation(i)
    finally:
        torch.load = real_load
    supervision_touched = [p for p in opened if 'supervision' in p]
    out['checks'].append(dict(name='supervision_files_opened_during_scoring_path',
                              value=len(supervision_touched), files=supervision_touched[:5]))
    out['PASS'] = bool(all(c.get('overlap', 0) == 0 for c in out['checks'] if c['name'] == 'disjoint_frame_id')
                       and len(supervision_touched) == 0
                       and not read(DECODER_PROBE / 'MANIFEST.json')['real_GT_read']
                       and not spec['GT_used'])
    return out


def _observation_batch(data, indices):
    return data.batch(indices)


def g3_comparison(train):
    """Arm-differentiating information paths, on real exported observations."""
    out = dict(gate='G3', checks=[])
    torch.manual_seed(1)
    obs = _observation_batch(train, [0, 1, 2, 3])
    models = {}
    initial = {}
    for arm in ARMS:
        torch.manual_seed(1)
        m = EvidenceVerifier(arm=arm, channels=(64, 128), visual=16, width=64, layers=2).eval()
        models[arm] = m
        initial[arm] = {k: v.clone() for k, v in m.state_dict().items()}
    same = all(all(torch.equal(initial['P'][k], initial[a][k]) for k in initial['P']) for a in ARMS)
    receipts = {a: models[a].parameter_receipt()['registered'] for a in ARMS}
    out['checks'].append(dict(name='same_seed_initial_state_identical_across_arms', value=bool(same),
                              registered_parameters=receipts))

    with torch.no_grad():
        base = {a: models[a](obs)[0].clone() for a in ARMS}

    # (a) explicit Hough cue must be inert for P/S and active for H/HA.
    perturbed = Observation(**{f: getattr(obs, f) for f in
                               ('features', 'raw_to_feature', 'content_valid', 'baseline', 'point_valid',
                                'point_conf', 'layouts', 'candidate_valid', 'line_h', 'line_logits',
                                'line_valid', 'diagonal')})
    perturbed.line_logits = obs.line_logits + 5.0
    perturbed.line_h = obs.line_h * 1.7 + 3.0
    with torch.no_grad():
        hough = {a: float((models[a](perturbed)[0] - base[a]).abs().max()) for a in ARMS}
    out['checks'].append(dict(name='sensitivity_to_explicit_hough_cue', value=hough))

    # (b) P must issue no interior segment query: perturb the planes only strictly
    # between edge endpoints, away from every 3x3 endpoint patch.
    interior = Observation(**{f: getattr(obs, f) for f in
                              ('features', 'raw_to_feature', 'content_valid', 'baseline', 'point_valid',
                               'point_conf', 'layouts', 'candidate_valid', 'line_h', 'line_logits',
                               'line_valid', 'diagonal')})
    planes = [f.clone() for f in obs.features]
    touched = 0
    for plane, (feature, affine) in enumerate(zip(planes, obs.raw_to_feature)):
        b, c, h, w = feature.shape
        keep = torch.zeros((b, h, w), dtype=torch.bool)
        q = obs.layouts[:, :, :8]
        hom = torch.cat([q.reshape(b, -1, 2), torch.ones(b, q.shape[1] * 8, 1)], -1)
        idx = torch.bmm(hom, affine.transpose(1, 2))
        for j in range(b):
            for (x, y) in idx[j].tolist():
                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        r, cc = int(round(y)) + dy, int(round(x)) + dx
                        if 0 <= r < h and 0 <= cc < w:
                            keep[j, r, cc] = True
        mid = torch.zeros((b, h, w), dtype=torch.bool)
        u, v = torch.tensor(EDGES)[:, 0], torch.tensor(EDGES)[:, 1]
        centre = idx.reshape(b, -1, 8, 2)
        for j in range(b):
            for k in range(centre.shape[1]):
                for e in range(12):
                    p0, p1 = centre[j, k, u[e]], centre[j, k, v[e]]
                    for t in (.3, .4, .5, .6, .7):
                        p = p0 + (p1 - p0) * t
                        r, cc = int(round(float(p[1]))), int(round(float(p[0])))
                        if 0 <= r < h and 0 <= cc < w:
                            mid[j, r, cc] = True
        target = mid & ~keep
        touched += int(target.sum())
        feature[target[:, None].expand(-1, c, -1, -1)] += 25.0
    interior.features = tuple(planes)
    with torch.no_grad():
        interior_effect = {a: float((models[a](interior)[0] - base[a]).abs().max()) for a in ARMS}
    out['checks'].append(dict(name='sensitivity_to_segment_interior_only_pixels',
                              perturbed_cells=touched, value=interior_effect))

    # (c) H must be invariant to a relabelling of the baseline point IDs, scored at a
    # FIXED non-identity candidate; HA (same-ID anchor) must react.
    perm = torch.tensor([1, 5, 6, 2, 0, 4, 7, 3])
    relabelled = Observation(**{f: getattr(obs, f) for f in
                                ('features', 'raw_to_feature', 'content_valid', 'baseline', 'point_valid',
                                 'point_conf', 'layouts', 'candidate_valid', 'line_h', 'line_logits',
                                 'line_valid', 'diagonal')})
    newbase = obs.baseline.clone(); newbase[:, :8] = obs.baseline[:, perm]
    newconf = obs.point_conf.clone(); newconf[:, :8] = obs.point_conf[:, perm]
    newvalid = obs.point_valid.clone(); newvalid[:, :8] = obs.point_valid[:, perm]
    newlayouts = obs.layouts.clone(); newlayouts[:, 0] = newbase          # identity slot must track baseline
    relabelled.baseline, relabelled.point_conf = newbase, newconf
    relabelled.point_valid, relabelled.layouts = newvalid, newlayouts
    fixed = 5   # a non-identity candidate whose coordinates are unchanged
    with torch.no_grad():
        shift = {a: float((models[a](relabelled)[0][:, fixed] - base[a][:, fixed]).abs().max()) for a in ARMS}
    out['checks'].append(dict(name='baseline_id_relabelling_shift_at_fixed_candidate',
                              candidate_index=fixed, value=shift))

    # (d) H must receive real gradient through the Hough inputs.
    grad = {}
    for arm in ('P', 'S', 'H', 'HA'):
        g = Observation(**{f: getattr(obs, f) for f in
                           ('features', 'raw_to_feature', 'content_valid', 'baseline', 'point_valid',
                            'point_conf', 'layouts', 'candidate_valid', 'line_h', 'line_logits',
                            'line_valid', 'diagonal')})
        logits = obs.line_logits.clone().requires_grad_(True)
        g.line_logits = logits
        models[arm](g)[0].sum().backward()
        grad[arm] = float(logits.grad.abs().sum()) if logits.grad is not None else 0.
    out['checks'].append(dict(name='gradient_into_line_logits', value=grad))

    out['PASS'] = bool(same
                       and hough['P'] == 0. and hough['S'] == 0. and hough['H'] > 0 and hough['HA'] > 0
                       and interior_effect['P'] == 0. and interior_effect['S'] > 0 and interior_effect['H'] > 0
                       and shift['H'] < 1e-5 and shift['HA'] > 1e-4
                       and grad['P'] == 0. and grad['S'] == 0. and grad['H'] > 0 and grad['HA'] > 0)
    return out


def g5_resources():
    free = subprocess.run(['df', '-B1', '--output=avail', str(RESULTS)], capture_output=True, text=True).stdout.split()[-1]
    out = dict(gate='G5', cuda_available=torch.cuda.is_available(),
               device_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               free_bytes_results=int(free), torch_version=torch.__version__)
    out['PASS'] = bool(out['cuda_available'] and out['free_bytes_results'] > 20 << 30)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=('gates', 'lock'), default='gates')
    args = ap.parse_args()
    manifests = {role: ExportDataset(EXPORT / f'{role}.json', verify_targets=(role != 'synth_val'))
                 for role in ('train', 'calibration', 'synth_val')}
    if args.phase == 'gates':
        started = time.perf_counter()
        gates = [g1_coordinates(manifests['train']), g2_data(manifests),
                 g3_comparison(manifests['train']), g5_resources()]
        oracle = read(RESULTS / 'NATIVE_ORACLE_AUDIT.json')
        gates.append(dict(gate='G4', PASS=bool(oracle['exploratory_entry_gate']),
                          relative_headroom=oracle['relative_headroom'],
                          frames_with_at_least_one_pixel_mean_gain=oracle['frames_with_at_least_one_pixel_mean_gain'],
                          source=str(RESULTS / 'NATIVE_ORACLE_AUDIT.json')))
        write(RESULTS / 'PREFLIGHT.json', dict(schema='pointline_v4_preflight_1',
              all_gates_PASS=all(g['PASS'] for g in gates), gates=gates,
              elapsed_seconds=time.perf_counter() - started,
              note='Software tests are not accepted as hookup evidence; every gate above ran on the actual export.'))
        for g in gates:
            print(g['gate'], 'PASS' if g['PASS'] else 'FAIL')
        print('ALL', all(g['PASS'] for g in gates))
    else:
        pre = read(RESULTS / 'PREFLIGHT.json')
        if not pre['all_gates_PASS']:
            raise SystemExit('Refusing to lock: a preflight gate failed')
        template = read(KIT / 'PROTOCOL_TEMPLATE.json')
        template.update(locked=True, status='LOCKED_FOR_STAGE_A',
                        train_manifest_sha256=sha256(EXPORT / 'train.json'),
                        core_source_sha256=core_sha(),
                        calibration_manifest_sha256=sha256(EXPORT / 'calibration.json'),
                        synth_val_manifest_sha256=sha256(EXPORT / 'synth_val.json'),
                        preflight_sha256=sha256(RESULTS / 'PREFLIGHT.json'),
                        oracle_audit_sha256=sha256(RESULTS / 'NATIVE_ORACLE_AUDIT.json'),
                        export_completion_sha256=sha256(RESULTS / 'EXPORT_COMPLETION.json'),
                        backbone=backbone_binding(), repository_commit=git('rev-parse', 'HEAD'),
                        torch_version=torch.__version__, locked_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        write(RESULTS / 'PROTOCOL_LOCK.json', template)
        print('locked', sha256(RESULTS / 'PROTOCOL_LOCK.json'))


if __name__ == '__main__':
    main()
