#!/usr/bin/env python3
"""Conditional paired REAL source evidence from supplied poses only.

This emits positive-support V/E teachers and UNKNOWN elsewhere. It never emits
outside/full-negative/self labels, fits a transform, or consults model scores.
An immutable, externally frozen protocol must exist before invocation.
"""
from pathlib import Path
from collections import Counter
import argparse, csv, hashlib, importlib.util, json, sys, time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
cv2.setNumThreads(1)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def stats(a):
    a = np.asarray(a, np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if not a.size:
        return {'N': 0}
    return dict(N=int(a.size), median=float(np.median(a)), mean=float(a.mean()),
                p05=float(np.percentile(a, 5)), p95=float(np.percentile(a, 95)),
                p99=float(np.percentile(a, 99)), min=float(a.min()), max=float(a.max()))

def ratio(num, den):
    return float(num) / int(den) if den else None

def erode2(mask):
    return cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0

def read_mask(path):
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert a.shape == (480, 640) and set(np.unique(a)) <= {0, 255}
    return a > 0

def read_depth(path):
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert a.shape == (480, 640) and a.dtype == np.uint16
    return a.astype(np.float64)

def load_mesh(path):
    vv, ff = [], []
    for line in Path(path).open():
        if line.startswith('v '):
            vv.append([float(x) for x in line.split()[1:4]])
        elif line.startswith('f '):
            ids = [int(t.split('/')[0]) - 1 for t in line.split()[1:]]
            for j in range(1, len(ids) - 1):
                ff.append([ids[0], ids[j], ids[j + 1]])
    return np.asarray(vv, np.float64), np.asarray(ff, np.int64)

def source_file_ref(path):
    p = DATA / path
    return dict(path=str(p), relative_path=path, sha256=sha(p), bytes=p.stat().st_size)

def font(size):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)

def make_sheet(records, out):
    fixed = [r for r in records if r['fixed_six_selection']]
    assert len(fixed) == 6
    canvas = Image.new('RGB', (1140, 100 + 315 * len(fixed)), 'white')
    d = ImageDraw.Draw(canvas)
    d.text((12, 8), 'Fixed public RGBD pairs: conditional TEACHER evidence V2', font=font(22), fill='black')
    d.text((12, 38), 'A / B / green=stable V; orange=A visible, B closer; cyan=B visible, A closer.', font=font(18), fill='black')
    d.text((12, 66), 'Source SAM+depth+supplied mesh. Native alignment uncertified. All other pixels UNKNOWN; no O/self GT.', font=font(16), fill='black')
    for i, r in enumerate(fixed):
        z = np.load(r['evidence']['path'])
        a = np.array(Image.open(r['files']['A']['image']['path']).convert('RGB'))
        b = np.array(Image.open(r['files']['B']['image']['path']).convert('RGB'))
        ev = np.full_like(a, 35)
        ev[z['stable_visible_both']] = [0, 220, 70]
        ev[z['external_B']] = [255, 110, 0]
        ev[z['external_A']] = [0, 200, 255]
        y = 100 + 315 * i
        d.text((12, y), r['pair_id'] + '   neutral filename order; no direction selection', font=font(18), fill='black')
        for j, image in enumerate([a, b, ev]):
            canvas.paste(Image.fromarray(image).resize((320, 240), Image.Resampling.NEAREST), (12 + 380 * j, y + 26))
        c = r['counts']
        d.text((12, y + 273), f"stable {c['stable_visible_both']}   A->B E50 {c['external_B_50']}   B->A E50 {c['external_A_50']}   remainder U", font=font(17), fill='black')
    canvas.save(out)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'SYMMETRIC_PAIRS_V2')
    parser.add_argument('--fixed-only', action='store_true')
    args = parser.parse_args()
    assert args.protocol.is_file() and sha(args.protocol) == args.protocol_sha256
    protocol = json.loads(args.protocol.read_text())
    # The exact final protocol schema/authority is also preserved verbatim below.
    # A nonempty approved protocol is mandatory, not silently manufactured here.
    assert protocol
    assert not args.output.exists(), 'Immutable output: use a new directory for another audit'
    args.output.mkdir(parents=True)
    (args.output / 'FROZEN_PROTOCOL_BYTES.json').write_bytes(args.protocol.read_bytes())
    selection_path = ROOT / 'FIXED_SIX_PAIR_SELECTION.json'
    fixed_ids = set(json.loads(selection_path.read_text())['pair_ids'])
    manifest_path = ROOT / 'EXACT_POSE_PAIR_MANIFEST.json'
    pairs = json.loads(manifest_path.read_text())['pairs']
    assert len(pairs) == 50 and Counter(p['split'] for p in pairs) == Counter(TRAIN_LAB=22, VAL_HALL=28)
    if args.fixed_only:
        pairs = [p for p in pairs if p['pair_id'] in fixed_ids]
        assert len(pairs) == 6
    renderer_path = ROOT.parent / 'source_audit/mesh_depth_render.py'
    spec = importlib.util.spec_from_file_location('frozen_source_renderer_v66', renderer_path)
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    assert renderer.smoke()['PASS']
    vertices, faces = load_mesh(DATA / 'assets/mesh/pallox.obj')
    started = time.monotonic()
    records = []
    manifest_out = args.output / 'PAIR_EVIDENCE_MANIFEST.jsonl'
    for p in pairs:
        a, b = p['before'], p['after']
        assert a['session'] == b['session'] == p['session']
        assert a['split'] == b['split'] == p['split']
        Ta, Tb = np.loadtxt(DATA / a['pose']), np.loadtxt(DATA / b['pose'])
        Ka, Kb = np.loadtxt(DATA / a['K']), np.loadtxt(DATA / b['K'])
        assert np.array_equal(Ta, Tb) and np.array_equal(Ka, Kb)
        assert sha(DATA / a['pose']) == sha(DATA / b['pose']) == p['supplied_pose_sha256']
        Ma, Mb = read_mask(DATA / a['mask_teacher']), read_mask(DATA / b['mask_teacher'])
        Da, Db = read_depth(DATA / a['depth']), read_depth(DATA / b['depth'])
        Ca, Cb = erode2(Ma), erode2(Mb)
        mesh_m, render_info = renderer.render_depth(vertices, faces, Ka, Ta, [480, 640])
        assert render_info['positive_depth_vertices'] == render_info['vertices']
        assert render_info['near_plane_crossing_triangles'] == 0
        M = mesh_m.astype(np.float64) * 1000.0
        finite = np.isfinite(M)
        ta, tb = np.maximum(20.0, .01 * Da), np.maximum(20.0, .01 * Db)
        agreeA = finite & (Da > 0) & (np.abs(M - Da) <= ta)
        agreeB = finite & (Db > 0) & (np.abs(M - Db) <= tb)
        Va, Vb = Ca & agreeA, Cb & agreeB
        def evidence(delta):
            # External class is named by OCCLUDED destination, not reference.
            Ea = Vb & (Da > 0) & ((Db - Da) >= delta) & ~Va & ((M - Da) >= np.maximum(50., 2. * tb))
            Eb = Va & (Db > 0) & ((Da - Db) >= delta) & ~Vb & ((M - Db) >= np.maximum(50., 2. * ta))
            return Ea, Eb
        Ea, Eb = evidence(50.)
        Ea100, Eb100 = evidence(100.)
        Ea200, Eb200 = evidence(200.)
        stable = Va & Vb & (np.abs(Da - Db) <= 20.)
        common = Ca & Cb & (Da > 0) & (Db > 0)
        assert not (Va & Ea).any() and not (Vb & Eb).any()
        assert np.all(Ea100 <= Ea) and np.all(Ea200 <= Ea100)
        assert np.all(Eb100 <= Eb) and np.all(Eb200 <= Eb100)
        exactKnownA, exactKnownB = Va | Ea, Vb | Eb
        # Under the conditional shared-camera/static-target assumption, an
        # observed visible target in either view establishes positive target
        # support in both. The destination can still have UNKNOWN visibility.
        fullPositive = Va | Vb
        zero = np.zeros((480, 640), bool)
        # UNKNOWN is explicit; no implicit assignment to outside/full-negative.
        clsA, clsB = np.zeros((480, 640), np.uint8), np.zeros((480, 640), np.uint8)
        clsA[Va], clsA[Ea], clsB[Vb], clsB[Eb] = 1, 2, 1, 2
        evidence_path = args.output / (p['pair_id'].replace(':', '__') + '.npz')
        np.savez_compressed(evidence_path, visible_A=Va, visible_B=Vb, external_A=Ea, external_B=Eb,
                            stable_visible_both=stable, target_known_positive_A=fullPositive,
                            target_known_positive_B=fullPositive, full_positive_A=fullPositive,
                            full_positive_B=fullPositive, full_negative_A=zero, full_negative_B=zero,
                            shared_full_positive=fullPositive, full_unknown_A=~fullPositive,
                            full_unknown_B=~fullPositive, class_known_A=exactKnownA, class_known_B=exactKnownB,
                            visible_positive_A=Va, visible_positive_B=Vb, visible_negative_A=Ea,
                            visible_negative_B=Eb, class_A=clsA, class_B=clsB,
                            teacher_core_A=Ca, teacher_core_B=Cb, mesh_agree_A=agreeA,
                            mesh_agree_B=agreeB, source_mesh_depth_m=mesh_m)
        counts = dict(teacher_A=int(Ma.sum()), teacher_B=int(Mb.sum()), teacher_core_A=int(Ca.sum()),
                      teacher_core_B=int(Cb.sum()), teacher_core_depth_valid_A=int((Ca & (Da > 0)).sum()),
                      teacher_core_depth_valid_B=int((Cb & (Db > 0)).sum()), visible_A=int(Va.sum()),
                      visible_B=int(Vb.sum()), external_A_50=int(Ea.sum()), external_B_50=int(Eb.sum()),
                      external_A_100=int(Ea100.sum()), external_B_100=int(Eb100.sum()),
                      external_A_200=int(Ea200.sum()), external_B_200=int(Eb200.sum()),
                      external_A_eroded_5x5=int(erode2(Ea).sum()), external_B_eroded_5x5=int(erode2(Eb).sum()),
                      external_A_teacher_positive=int((Ea & Ma).sum()), external_B_teacher_positive=int((Eb & Mb).sum()),
                      stable_visible_both=int(stable.sum()), common_teacher_core_both_depth=int(common.sum()),
                      common_teacher_core_depth_stable_20mm=int((common & (np.abs(Da - Db) <= 20.)).sum()),
                      target_known_positive_A=int(fullPositive.sum()), target_known_positive_B=int(fullPositive.sum()),
                      UNKNOWN_A=int((~exactKnownA).sum()), UNKNOWN_B=int((~exactKnownB).sum()),
                      full_UNKNOWN_A=int((~fullPositive).sum()), full_UNKNOWN_B=int((~fullPositive).sum()),
                      outside_labels=0, semantic_self_labels=0)
        # Registration evidence is DESCRIPTIVE: never select/drop a pair here.
        rgbA = np.array(Image.open(DATA / a['image']).convert('RGB'))
        rgbB = np.array(Image.open(DATA / b['image']).convert('RGB'))
        relative = Tb @ np.linalg.inv(Ta)
        yy, xx = np.where(common)
        step = max(1, len(xx) // 2000)
        xx, yy = xx[::step], yy[::step]
        dz = Da[yy, xx] / 1000.
        points = np.stack([(xx - Ka[0, 2]) * dz / Ka[0, 0], (yy - Ka[1, 2]) * dz / Ka[1, 1], dz], axis=1)
        warped = points @ relative[:3, :3].T + relative[:3, 3]
        hproj = warped @ Kb.T
        reproj = hproj[:, :2] / hproj[:, 2, None]
        errs = np.linalg.norm(reproj - np.stack([xx, yy], axis=1), axis=1)
        files = {}
        for name, r in [('A', a), ('B', b)]:
            files[name] = {key: source_file_ref(r[key]) for key in ['image', 'depth', 'mask_teacher', 'pose', 'K']}
        fractions = dict(teacher_core_valid_depth_A=ratio((Ca & (Da > 0)).sum(), Ca.sum()),
                         teacher_core_valid_depth_B=ratio((Cb & (Db > 0)).sum(), Cb.sum()),
                         mesh_agree_given_teacher_core_valid_A=ratio(Va.sum(), (Ca & (Da > 0)).sum()),
                         mesh_agree_given_teacher_core_valid_B=ratio(Vb.sum(), (Cb & (Db > 0)).sum()),
                         common_depth_stable20=ratio((common & (np.abs(Da - Db) <= 20.)).sum(), common.sum()))
        r = dict(pair_id=p['pair_id'], session=p['session'], split=p['split'], A_frame=a['frame_id'],
                 B_frame=b['frame_id'], fixed_six_selection=p['pair_id'] in fixed_ids,
                 files=files, counts=counts, coverage_descriptive=fractions,
                 common_teacher_depth_B_minus_A_mm=stats((Db - Da)[common]),
                 common_teacher_depth_abs_residual_mm=stats(np.abs(Db - Da)[common]),
                 common_teacher_RGB_abs_channel_mean=stats(np.abs(rgbB.astype(float) - rgbA.astype(float)).mean(2)[common]),
                 mesh_minus_depth_A_teacher_core_mm=stats((M - Da)[Ca & (Da > 0) & finite]),
                 mesh_minus_depth_B_teacher_core_mm=stats((M - Db)[Cb & (Db > 0) & finite]),
                 supplied_pose_reprojection_px=stats(errs),
                 supplied_pose_relative_identity_max_abs=float(np.abs(relative - np.eye(4)).max()),
                 supplied_pose_and_K_identical=True, renderer=render_info,
                 evidence=dict(path=str(evidence_path), sha256=sha(evidence_path)),
                 protocol_sha256=args.protocol_sha256, tier='CONDITIONAL_TEACHER_NOT_GOLD',
                 pixel_RGB_depth_alignment_certified=False, native_depth_is_assumed_RGB_grid=True,
                 no_threshold_selection_no_pair_quality_filter=True,
                 admission='HOLD_FOR_ROOT_AND_CONTRACT_LABEL_REVIEW; NO_FIT',
                 unknown_policy='Everything except explicit Va/Vb/Ea/Eb remains UNKNOWN. No O/self/full-negative.')
        records.append(r)
        with manifest_out.open('a') as f:
            f.write(json.dumps(r, allow_nan=False) + '\n')
        print('PAIR_V2', len(records), '/', len(pairs), p['pair_id'],
              'V', counts['visible_A'], counts['visible_B'], 'E', counts['external_A_50'], counts['external_B_50'],
              'stable20', fractions['common_depth_stable20'], flush=True)
    make_sheet(records, args.output / 'FIXED_SIX_SYMMETRIC_TEACHER_V2.png')
    flat_path = args.output / 'PAIR_QUALITY_COUNTS.csv'
    flat_rows = []
    for r in records:
        flat_rows.append(dict(pair_id=r['pair_id'], session=r['session'], split=r['split'],
                              fixed_six_selection=r['fixed_six_selection'], **r['counts'],
                              **r['coverage_descriptive'],
                              common_abs_depth_median_mm=r['common_teacher_depth_abs_residual_mm'].get('median'),
                              common_abs_depth_p95_mm=r['common_teacher_depth_abs_residual_mm'].get('p95'),
                              common_abs_depth_p99_mm=r['common_teacher_depth_abs_residual_mm'].get('p99')))
    with flat_path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(flat_rows[0])); w.writeheader(); w.writerows(flat_rows)
    split_counts = {}
    for split in ['TRAIN_LAB', 'VAL_HALL']:
        rr = [r for r in records if r['split'] == split]
        split_counts[split] = dict(pairs=len(rr), frame_count=2 * len(rr),
                                   total_native_pixels=2 * len(rr) * 480 * 640,
                                   counts={k: sum(r['counts'][k] for r in rr) for k in rr[0]['counts']})
    report = dict(schema='PUBLIC_PALLOX_SYMMETRIC_SOURCE_EVIDENCE_V2', protocol=str(args.protocol),
                  protocol_sha256=args.protocol_sha256, source_pair_manifest_sha256=sha(manifest_path),
                  fixed_selection_sha256=sha(selection_path), renderer_sha256=sha(renderer_path),
                  code_sha256=sha(__file__), all_requested_pairs_written=len(records) == len(pairs),
                  pairs=len(records), fixed_six_only=args.fixed_only, split_counts=split_counts,
                  native_class_encoding={'0': 'UNKNOWN', '1': 'CONDITIONAL_VISIBLE_TEACHER', '2': 'CONDITIONAL_EXTERNAL_TEACHER'},
                  caveat='SAM prompt teacher, supplied approximate mesh/poses, native RGB/depth grid assumption not calibration-certified; no independent human per-pixel truth.',
                  external_reference='External_A uses visible B tauB; external_B uses visible A tauA. mesh margin max(50,2*tau_reference) is required.',
                  thresholds_not_selected_by_model_score_or_current_data=True, pairs_not_filtered_by_diagnostic_quality=True,
                  private_RGB_GT_scores_read=False, source_only=True, GPU_used=False,
                  training=False, model_inference=False, PnP_calls=0, pose_or_alignment_optimization=False,
                  fit_admission='HOLD_FOR_ROOT_AND_CONTRACT_REVIEW', seconds=time.monotonic() - started,
                  evidence_manifest_sha256=sha(manifest_out), pair_quality_csv_sha256=sha(flat_path),
                  cases=records)
    write(args.output / 'SYMMETRIC_SOURCE_EVIDENCE_REPORT.json', report)
    print('DONE', len(records), 'pairs', report['seconds'], 'seconds', flush=True)

if __name__ == '__main__':
    main()
