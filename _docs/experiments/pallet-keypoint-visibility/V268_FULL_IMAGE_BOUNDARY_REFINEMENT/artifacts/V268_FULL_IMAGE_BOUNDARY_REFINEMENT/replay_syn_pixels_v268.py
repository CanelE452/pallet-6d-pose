"""Portable CPU replay: 235 SYN original-M4 pixels, no raw dataset/model needed.

Run from any directory with Python, NumPy, SciPy and PyTorch installed:
  python -B /path/to/V268_FULL_IMAGE_BOUNDARY_REFINEMENT/replay_syn_pixels_v268.py
The two saved V268 DEV logit files must remain in syn_training_v1/.
"""
from pathlib import Path
import hashlib
import io
import json
import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage import binary_erosion, distance_transform_edt

HERE = Path(__file__).absolute().parent
ARMS = ('V267_CS32', 'HIGHRES_REFINE', 'BOUNDARY_REFINE')


def safe(value):
    p = Path(value)
    assert p.is_absolute() and not any(s in str(p).lower() for s in ('v112', 'cache', '.pyc', '..')), str(p)
    for a in reversed((p, *p.parents)):
        assert not a.is_symlink(), str(a)
    return p


def digest(path):
    h = hashlib.sha256()
    with safe(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def checked(bound):
    rel = bound['relative_path']
    assert not Path(rel).is_absolute() and not any(x in rel.lower() for x in ('v112', 'cache', '.pyc', '..'))
    path = safe(HERE / rel)
    assert path.stat().st_size == bound['size_bytes'] and digest(path) == bound['sha256']
    return path


def boundary(mask):
    return mask & ~binary_erosion(mask, structure=np.ones((3,3), bool), border_value=0)


def score(pred, truth):
    tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum()); tn = int((~pred & ~truth).sum())
    out = dict(TP=tp, FP=fp, FN=fn, TN=tn,
               IoU=tp/(tp+fp+fn) if tp+fp+fn else 1.,
               Dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.)
    pb, gb = boundary(pred), boundary(truth)
    npred, ngold = int(pb.sum()), int(gb.sum())
    dg = distance_transform_edt(~gb) if ngold else None
    dp = distance_transform_edt(~pb) if npred else None
    for tol in (1, 3):
        pm = int((pb & (dg <= tol)).sum()) if dg is not None else 0
        gm = int((gb & (dp <= tol)).sum()) if dp is not None else 0
        precision, recall = pm/npred if npred else 0., gm/ngold if ngold else 0.
        out[f'boundary{tol}px'] = dict(predicted_boundary_pixels=npred, GT_boundary_pixels=ngold,
            matched_predicted_boundary_pixels=pm, matched_GT_boundary_pixels=gm,
            precision=precision, recall=recall, F1=2*precision*recall/(precision+recall) if precision+recall else 0.)
    return out


def aggregate(rows):
    tp, fp, fn, tn = (sum(r[k] for r in rows) for k in ('TP', 'FP', 'FN', 'TN'))
    out = dict(frames=len(rows), CM_TP_FP_FN_TN=[tp,fp,fn,tn],
        micro_IoU=tp/max(tp+fp+fn,1), micro_Dice=2*tp/max(2*tp+fp+fn,1),
        mean_frame_IoU=float(np.mean([r['IoU'] for r in rows])),
        mean_frame_Dice=float(np.mean([r['Dice'] for r in rows])))
    for tol in (1,3):
        name = f'boundary{tol}px'
        sums = {k:sum(r[name][k] for r in rows) for k in ('predicted_boundary_pixels','GT_boundary_pixels','matched_predicted_boundary_pixels','matched_GT_boundary_pixels')}
        p = sums['matched_predicted_boundary_pixels']/max(sums['predicted_boundary_pixels'],1)
        r = sums['matched_GT_boundary_pixels']/max(sums['GT_boundary_pixels'],1)
        out[name] = dict(precision=p, recall=r, F1=2*p*r/(p+r) if p+r else 0., **sums)
    return out


def compare(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b)
        for k in a: compare(a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x,y in zip(a,b): compare(x,y)
    elif isinstance(a, float):
        assert abs(a-b) < 1e-12, (a,b)
    else:
        assert a == b, (a,b)


def main():
    torch.set_num_threads(1)
    folder = safe(HERE/'syn_pixel_replay_v1')
    manifest = json.loads(safe(folder/'REPLAY_INPUTS.json').read_bytes())
    with np.load(checked(manifest['packed_M4']), allow_pickle=False) as z:
        packed = z['packed']; offsets = z['byte_offsets']; hw = z['raw_hw']; frame_ids = z['frame_ids']
    assert len(frame_ids) == 235 and np.array_equal(frame_ids, np.arange(768,1003))
    logits = {}
    for arm in ARMS:
        with np.load(checked(manifest['saved_logits'][arm]), allow_pickle=False) as z:
            assert np.array_equal(z['frame_ids'], frame_ids)
            logits[arm] = np.array(z['mask_logits'], copy=True)
    rows = {a:[] for a in ARMS}
    for j, (h,w) in enumerate(hw):
        h,w = int(h),int(w)
        truth = np.unpackbits(packed[offsets[j]:offsets[j+1]], bitorder='little', count=h*w).reshape(h,w).astype(bool)
        for arm in ARMS:
            field = torch.from_numpy(logits[arm][j:j+1])
            pred = F.interpolate(field, (h,w), mode='bilinear', align_corners=False)[0,0].numpy() >= 0
            rows[arm].append(score(pred, truth))
    results = {a:aggregate(rows[a]) for a in ARMS}
    compare(results, manifest['expected_results'])
    receipt = dict(completed=True, PASS=True, all235_counts_and_metrics_match_frozen_report=True,
        source_RGB_or_M4_files_read=0, current_data_reads=0, optimizer_updates=0,
        model_or_encoder_forwards=0, raw_pixel_threshold='M4 uint8>127; prediction raw logit>=0',
        restoration='bilinear raw logits to original H/W; align_corners=False; no antialias/postprocess',
        boundary='mask minus3x3 erosion, outside0; fixed original-pixel tolerances1/3', results=results)
    p = safe(folder/'REPLAY_RESULT.json')
    if p.exists():
        assert json.loads(p.read_bytes()) == receipt
    else:
        p.write_text(json.dumps(receipt,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    print(json.dumps(receipt,ensure_ascii=False,sort_keys=True))


if __name__ == '__main__':
    main()
