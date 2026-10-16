"""Export ROI features and object-frame supervision from the stock R0 detector.

One real forward per frame supplies both the detections and the native P3/P4
that the ROI crop is sampled from. Ground truth is built by projecting the fixed
object-frame corners with the renderer pose -- never by reading the stored
camera-facing corner order, which re-picks the front face every frame.
"""
from __future__ import annotations
import argparse, hashlib, json, math, sys, time, zipfile
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT)); sys.path.insert(0, str(ROOT))
from plpose_local.geometry import corners, project           # noqa: E402

R0 = ROOT / 'challenge/yolo_pose_one_model/spatial_concat_scratch/runs/YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt'
R0_SHA = '970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7'
PROBE = ROOT / 'challenge/yolo_pose_one_model/spatial_concat_scratch/PROBE_METADATA_60K.jsonl'
STRUCTURED = ROOT / 'data/pallet/results/pallet_dht_structured_v2'
DECODER = ROOT / 'data/pallet/results/pallet_dht_decoder_probe_v1'
SIZE, PAD = 24, 100
CONTRACT = ROOT / '_docs/experiments/pallet_translation_loss_v1/LOSS_SYMMETRY_CONTRACT.json'


def symmetry_table():
    """Effective upright-yaw order per asset, straight from the frozen contract.

    C4 requires a confirmed 90 degree equivalence AND an exact square footprint.
    No asset carries the former, so no frame here can be C4; shape alone never
    promotes an order.
    """
    c = json.loads(CONTRACT.read_text())
    order = {k: v['max_valid_order'] for k, v in c['assets'].items()}
    legacy = {k: v for k, v in c['legacy_stream_asset_resolution'].items() if k != 'basis'}
    return order, legacy


_ZIPS: dict = {}


def read_label(locator: str) -> dict:
    """Renderer labels live either on disk or inside a frozen legacy archive."""
    if '::' not in locator:
        return json.loads((ROOT / locator).read_text())
    archive, inner = locator.split('::', 1)
    zf = _ZIPS.get(archive)
    if zf is None:
        zf = _ZIPS[archive] = zipfile.ZipFile(archive)
    return json.loads(zf.read(inner))


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_population():
    """Frame ids and split, reusing the frozen structured-v2 lineage."""
    from scripts.research.pallet_dht_structured_v2.cache import CachedData
    data = CachedData(STRUCTURED, verify=False)
    ids = {}
    for split, key in (('train', 'train'), ('calibration', 'calibration'), ('synth_val', 'synth_val')):
        ids[split] = [data.records[i]['id'] for i in data.populations[key]]
    return ids


def load_probe():
    out = {}
    with open(PROBE) as f:
        for line in f:
            r = json.loads(line)
            out[r['merged_stem']] = r
    return out


class Capture:
    """Hook the stock Pose26 head input: [P3, P4, P5] before detection."""
    def __init__(self, model):
        self.features = None
        head = model.model.model[-1]
        if type(head).__name__ != 'Pose26':
            raise ValueError(f'expected stock Pose26 head, found {type(head).__name__}')
        self.handle = head.register_forward_pre_hook(self._hook)

    def _hook(self, _m, args):
        if len(args) != 1 or len(args[0]) != 3:
            raise ValueError('expected a single P3/P4/P5 list')
        self.features = tuple(f.detach().float() for f in args[0][:2])


def roi_sample(plane, box_raw, affine, size=SIZE):
    """Sample a size x size grid over the predicted raw box from one plane."""
    _, c, h, w = plane.shape
    xs = torch.linspace(0, 1, size, device=plane.device)
    gy, gx = torch.meshgrid(xs, xs, indexing='ij')
    x = box_raw[0] + gx * (box_raw[2] - box_raw[0])
    y = box_raw[1] + gy * (box_raw[3] - box_raw[1])
    ones = torch.ones_like(x)
    pts = torch.stack((x, y, ones), -1) @ torch.as_tensor(affine, dtype=torch.float32,
                                                          device=plane.device).T
    grid = torch.stack((2 * (pts[..., 0] + .5) / w - 1, 2 * (pts[..., 1] + .5) / h - 1), -1)
    val = torch.nn.functional.grid_sample(plane, grid[None], mode='bilinear',
                                          padding_mode='zeros', align_corners=False)
    inside = ((pts[..., 0] >= 0) & (pts[..., 0] <= w - 1) &
              (pts[..., 1] >= 0) & (pts[..., 1] <= h - 1))
    return val[0], inside


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., x2 - x1) * max(0., y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    import cv2
    from ultralytics import YOLO
    from scripts.research.pallet_line_pose_v1 import source_data as SD

    if sha(R0) != R0_SHA:
        raise SystemExit('R0 checkpoint SHA mismatch')
    torch.set_num_threads(2); cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'export exists: {out}')
    population, probe = load_population(), load_probe()
    order_by_asset, legacy_asset = symmetry_table()
    model = YOLO(str(R0), task='pose'); model.model.float().eval()
    cap = Capture(model)

    ledger, manifests, started = [], {}, time.perf_counter()
    for split, ids in population.items():
        if args.limit:
            ids = ids[:args.limit]
        records = []
        for n, fid in enumerate(ids):
            meta = probe[fid]
            label = read_label(meta['renderer_label_locator'])
            obj = label['objects'][0]
            ci = label['camera_data']['intrinsics']
            K = torch.tensor([[ci['fx'], 0, ci['cx']], [0, ci['fy'], ci['cy']], [0, 0, 1.]])
            T = torch.tensor(obj['pose_transform'], dtype=torch.float32)
            fixed = meta['fixed_renderer_dimensions_m_xyz_model_input']
            dims = torch.tensor([[fixed[0], fixed[2], fixed[1]]])        # (W, D, H)
            Rg, tg = T[None, :3, :3], T[None, :3, 3]
            gt_uv, gt_z = project(K[None], Rg, tg, corners(dims))
            raw_h, raw_w = meta['raw_shape_height_width']
            gt_box = [float(gt_uv[0, :8, 0].min()), float(gt_uv[0, :8, 1].min()),
                      float(gt_uv[0, :8, 0].max()), float(gt_uv[0, :8, 1].max())]

            image = cv2.imread(meta['image_path'])
            if image is None:
                ledger.append(dict(id=fid, split=split, reason='image_unreadable')); continue
            cap.features = None
            pred = model.predict(image, imgsz=640, conf=.001, iou=.7, max_det=300,
                                 verbose=False, half=False)[0]
            if cap.features is None:
                raise SystemExit('feature hook did not fire')
            boxes = pred.boxes.xyxy.detach().cpu().numpy()
            confs = pred.boxes.conf.detach().cpu().numpy()
            if len(boxes) == 0:
                ledger.append(dict(id=fid, split=split, reason='no_detection')); continue
            cand = boxes - PAD                                            # padded -> raw
            scores = np.array([iou(b, gt_box) for b in cand])
            best = int(scores.argmax())
            if scores[best] < .5:
                ledger.append(dict(id=fid, split=split, reason='unmatched_detection',
                                   best_iou=float(scores[best]))); continue
            box = torch.tensor(cand[best], dtype=torch.float32)

            fh, fw = cap.features[1].shape[-2:]
            tf = SD.transform((raw_h + 2 * PAD, raw_w + 2 * PAD), (int(fh) * 16, int(fw) * 16))
            aff_in = np.array([[tf.gain, 0, PAD * tf.gain + tf.pad_ltrb[0]],
                               [0, tf.gain, PAD * tf.gain + tf.pad_ltrb[1]]], dtype=np.float64)
            planes, masks = [], []
            for stride, plane in zip((8, 16), cap.features):
                a = aff_in / stride
                a[:, 2] -= .5
                v, ins = roi_sample(plane, box, a)
                planes.append(v); masks.append(ins)
            feat = torch.cat(planes, 0)[None]
            content = (masks[0] & masks[1])[None, None]

            stem = fid.replace('/', '_')
            obs = dict(features=feat, content=content, box=box[None],
                       K=K[None], dims=dims, image_hw=torch.tensor([[float(raw_h), float(raw_w)]]))
            sup = dict(points=gt_uv[:, :9].contiguous(),
                       valid=torch.ones(1, 9, dtype=torch.bool),
                       R=Rg.contiguous(), t=tg.contiguous())
            op = out / split / 'observations' / f'{stem}.pt'
            sp = out / split / 'supervision' / f'{stem}.pt'
            op.parent.mkdir(parents=True, exist_ok=True); sp.parent.mkdir(parents=True, exist_ok=True)
            torch.save(obs, op); torch.save(sup, sp)
            asset = meta.get('source_asset') or legacy_asset[meta['source']]
            records.append(dict(id=fid, session=meta['source'], asset=asset,
                                asset_resolution='direct' if 'source_asset' in meta else
                                'legacy_stream_asset_resolution (frozen contract)',
                                observation=str(op.relative_to(out)), observation_sha256=sha(op),
                                supervision=str(sp.relative_to(out)), supervision_sha256=sha(sp),
                                source_image_sha256=sha(meta['image_path']),
                                original_image=meta['image_path'],
                                detection_iou=float(scores[best]), detection_conf=float(confs[best]),
                                symmetry=dict(order=order_by_asset[asset], verified=False,
                                              basis='LOSS_SYMMETRY_CONTRACT.json max_valid_order; '
                                                    'C4 never confirmed for any asset')))
            if (n + 1) % 200 == 0:
                print(f'{split} {n+1}/{len(ids)} ({time.perf_counter()-started:.0f}s)', flush=True)
        doc = dict(schema='plpose_local_instance_manifest_1',
                   coordinate_contract='object_x_width_y_up_z_depth_v1  (A=identity vs renderer fixed axes)',
                   split=split, population_scope='DETECTION_CONDITIONED: frames whose stock-R0 top-IoU '
                   'detection matched the projected GT box at IoU>=0.5; failures are kept in DETECTION_LEDGER.json',
                   feature_source='stock R0 Pose26 head input P3(stride8)+P4(stride16), 16+16 channels after ROI sample',
                   records=records)
        p = out / f'{split}.json'
        p.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        manifests[split] = dict(records=len(records), sha256=sha(p))
        print(f'{split}: {len(records)} records', flush=True)
    (out / 'DETECTION_LEDGER.json').write_text(json.dumps(dict(
        schema='plpose_local_ledger_1', requested={k: len(v) for k, v in population.items()},
        exported={k: v['records'] for k, v in manifests.items()}, failures=ledger,
        failure_counts={r: sum(1 for x in ledger if x['reason'] == r)
                        for r in {x['reason'] for x in ledger}} if ledger else {},
        note='A missing detection is a pose failure in deployment; these frames are excluded from the '
             'manifests but counted here so the conditioning is never hidden.'),
        ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(manifests, indent=1))


if __name__ == '__main__':
    main()
