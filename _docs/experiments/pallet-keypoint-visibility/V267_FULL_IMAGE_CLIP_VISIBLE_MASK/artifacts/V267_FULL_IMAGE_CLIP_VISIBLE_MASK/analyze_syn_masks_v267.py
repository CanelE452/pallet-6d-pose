"""Saved SYN DEV full-frame mask diagnostics at original pixel resolution.

No model/encoder forward, fitting, threshold search, current data or GT crop.
M4 query presence is evaluated separately from native visibility/occlusion.
"""
from pathlib import Path
import collections
import csv
import hashlib
import html
import io
import json
import shutil
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.nn import functional as F

HERE = Path(__file__).absolute().parent
ARMS = ('PLAIN_CLIP', 'CLIP_SURGERY_ARCH')
FREEZE_SHA = '7da54bae6f02e70e33641ff78782325e981c7755a3f69327e85aa51ec9895b84'


def safe(path):
    p = Path(path)
    assert p.is_absolute() and not any(s in str(p).lower() for s in ('v112', 'cache', '.pyc', '..')), str(p)
    for a in reversed((p, *p.parents)):
        assert not a.is_symlink(), str(a)
    return p


def binding(path, expected=None):
    p = safe(path)
    b = p.read_bytes()
    result = dict(path=str(p), sha256=hashlib.sha256(b).hexdigest(), size_bytes=len(b))
    assert expected is None or result['sha256'] == expected
    return result


def checked(bound):
    assert binding(bound['path']) == bound
    return safe(bound['path'])


def read_json(path):
    return json.loads(safe(path).read_bytes())


def write_json(path, obj):
    with safe(path).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def write_csv(path, rows):
    with safe(path).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pixel_counts(truth, pred):
    return [int((truth & pred).sum()), int((~truth & pred).sum()),
            int((truth & ~pred).sum()), int((~truth & ~pred).sum())]


def count_metrics(counts):
    tp, fp, fn, tn = counts
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, IoU=tp/(tp+fp+fn) if tp+fp+fn else 1.,
                Dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.)


def aggregate(records):
    counts = np.asarray([[r[k] for k in ('TP', 'FP', 'FN', 'TN')] for r in records], dtype=np.int64).sum(0)
    return dict(frames=len(records), **count_metrics(counts.tolist()),
                mean_frame_IoU=float(np.mean([r['IoU'] for r in records])),
                mean_frame_Dice=float(np.mean([r['Dice'] for r in records])))


def presence_summary(y, pred):
    cm = [[int(((y == a) & (pred == b)).sum()) for b in (0, 1, -1)] for a in (0, 1)]
    result = dict(N=len(y), CM_rows_NON_TARGET_TARGET_cols_NON_TARGET_TARGET_U=cm)
    for c, name in enumerate(('NON_TARGET', 'TARGET')):
        tp, fp, n = cm[c][c], cm[1-c][c], sum(cm[c])
        result[name] = dict(precision=tp/(tp+fp) if tp+fp else None,
                            recall=tp/n if n else None,
                            F1=2*tp/(n+tp+fp) if n+tp+fp else None)
    result['U_rate'] = sum(r[2] for r in cm)/len(y) if len(y) else None
    return result


def original_logits(logit, raw_hw):
    return F.interpolate(torch.from_numpy(logit[None]), size=tuple(raw_hw),
                         mode='bilinear', align_corners=False)[0, 0].numpy()


def sample_queries(logit, xy, hw):
    grid = 2*(np.asarray(xy, np.float32)+.5)/np.asarray(hw[::-1], np.float32)-1
    return F.grid_sample(torch.from_numpy(logit[None]), torch.from_numpy(grid[None, None]),
                         mode='bilinear', padding_mode='border', align_corners=False)[0, 0, 0].numpy()


def area_group(fraction):
    for edge, name in ((.01, 'lt1pct'), (.05, '1to_lt5pct'), (.1, '5to_lt10pct')):
        if fraction < edge:
            return name
    return 'ge10pct'


def color_probability(p):
    stops = np.asarray([[12, 25, 90], [40, 155, 205], [240, 220, 70], [220, 45, 30]], np.float64)
    x = np.clip(p, 0, 1)*3
    low = np.minimum(np.floor(x).astype(int), 2)
    f = x-low
    return np.clip(stops[low]*(1-f[..., None])+stops[low+1]*f[..., None], 0, 255).astype(np.uint8)


def error_overlay(rgb, truth, pred):
    out = rgb.astype(np.float64)
    for mask, color in ((truth & pred, (35, 210, 90)), (~truth & pred, (225, 40, 220)),
                        (truth & ~pred, (240, 65, 45))):
        out[mask] = out[mask]*.45+np.asarray(color)*.55
    return out.astype(np.uint8)


def panel(image, title, size=(640, 400)):
    p = Image.new('RGB', (size[0], size[1]+38), 'white')
    draw = ImageDraw.Draw(p)
    draw.text((6, 4), title, font=FONT, fill='black')
    source = Image.fromarray(image).convert('RGB')
    source.thumbnail(size, Image.Resampling.LANCZOS)
    p.paste(source, ((size[0]-source.width)//2, 38+(size[1]-source.height)//2))
    return p


def main():
    started = time.monotonic()
    torch.set_num_threads(1)
    freeze_bound = binding(HERE/'NEW_CONFIG_FROZEN_BEFORE_CURRENT.json', FREEZE_SHA)
    freeze = read_json(freeze_bound['path'])
    complete = read_json(checked(freeze['training_complete']))
    audit = read_json(HERE/'data_audit/COMPLETE.json')
    audit_summary = read_json(checked(audit['outputs']['AUDIT_SUMMARY.json']))
    inputs_bound = audit['outputs']['FULL_FRAME_RGB_INPUTS.jsonl']
    targets_bound = audit['outputs']['VISIBLE_MASK_TARGETS.jsonl']
    queries_bound = audit['outputs']['ORIGINAL_QUERY_SIDECAR.jsonl']
    inputs = [json.loads(x) for x in checked(inputs_bound).read_bytes().splitlines()]
    targets = [json.loads(x) for x in checked(targets_bound).read_bytes().splitlines()]
    queries = [json.loads(x) for x in checked(queries_bound).read_bytes().splitlines()]
    old_bound = audit_summary['dependencies']['PRESENCE_QUERY_LABELS.csv']
    old = list(csv.DictReader(io.StringIO(checked(old_bound).read_text())))
    assert len(old) == 8024 and {r['id'] for r in old} == {r['id'] for r in inputs}
    old = {(r['id'], int(r['native_corner'])): r for r in old}
    packets = {}
    source_bindings = dict(freeze=freeze_bound, inputs=inputs_bound, targets=targets_bound,
                           queries=queries_bound, old_SYN_presence_labels=old_bound)
    for arm in ARMS:
        bound = complete['outputs'][arm+'_SYN_DEV_MASKS.npz']
        with np.load(checked(bound), allow_pickle=False) as z:
            packets[arm] = {key: z[key].copy() for key in z.files}
        source_bindings[arm] = bound
    ids = packets[ARMS[0]]['frame_ids']
    assert np.array_equal(ids, packets[ARMS[1]]['frame_ids']) and len(ids) == 235
    assert all(inputs[int(i)]['split'] == 'validation' for i in ids)
    out = safe(HERE/'syn_mask_diagnostic_v1')
    if out.exists():
        assert {p.name for p in out.iterdir()} == {
            'DIAGNOSTIC_PROTOCOL_BEFORE_RGB.json', 'PREPARATION_FAILURE_BEFORE_RGB.json'}
        previous = read_json(out/'DIAGNOSTIC_PROTOCOL_BEFORE_RGB.json')
        assert previous['sources'] == source_bindings
    else:
        out.mkdir()
    write_json(out/'DIAGNOSTIC_PROTOCOL_FINAL_BEFORE_RGB.json', dict(completed=True,
        code=binding(Path(__file__).absolute()), sources=source_bindings,
        metric='RAW32 logits bilinear to original HxW align_corners=False; positive logit>=0; M4>127',
        selection='First frame_index from each rank-based M4 area-fraction quartile; then unused CS lowest IoU and unused CS highest IoU, ties frame_index.',
        selection_saved_before_any_RGB_decode=True, query_target='M4 target presence, not native E',
        area_bins=['lt1pct', '1to_lt5pct', '5to_lt10pct', 'ge10pct'], new_fit_or_model_forward=0))
    per_frame, per_query = [], []
    fractions = {}
    point_truth, point_valid, point_oldvalid = [], [], []
    point_pred = {arm: [] for arm in ARMS}
    original_masks = {}
    for j, index0 in enumerate(ids):
        index = int(index0)
        row, target_row, query = inputs[index], targets[index], queries[index]
        assert row['id'] == target_row['id'] == query['id']
        with Image.open(io.BytesIO(checked(target_row['M4']).read_bytes())) as im:
            truth = np.array(im) > 127
        assert truth.shape == tuple(row['raw_hw']) and int(truth.sum()) == target_row['M4_area']
        original_masks[index] = truth
        fraction = float(truth.mean())
        fractions[index] = fraction
        xy = np.asarray(query['initial_native8_xy'], np.float64)
        nearest = np.floor(xy+.5).astype(np.int64)
        h, w = truth.shape
        valid = np.isfinite(xy).all(1) & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        valid &= (nearest[:, 0] >= 0) & (nearest[:, 0] < w) & (nearest[:, 1] >= 0) & (nearest[:, 1] < h)
        assert np.array_equal(valid, query['original_coordinate_valid8_no_ROI_rule'])
        labels = np.full(8, -1, np.int8)
        labels[valid] = truth[nearest[valid, 1], nearest[valid, 0]].astype(np.int8)
        stored_labels = np.asarray([-1 if value is None else int(value)
                                    for value in query['M4_target_at_query8']], np.int8)
        assert np.array_equal(labels, stored_labels)
        oldvalid = np.asarray([old[(row['id'], k)]['valid'].lower() == 'true' for k in range(8)])
        assert np.all(~oldvalid | valid)
        for k in np.flatnonzero(oldvalid):
            assert int(old[(row['id'], int(k))]['presence_target']) == int(labels[k])
        scores, predictions = {}, {}
        for arm in ARMS:
            logit = packets[arm]['mask_logits'][j]
            pixel_logit = original_logits(logit, truth.shape)
            metrics = count_metrics(pixel_counts(truth, pixel_logit >= 0))
            per_frame.append(dict(frame_index=index, id=row['id'], arm=arm, H=h, W=w,
                GT_pixels=int(truth.sum()), GT_fraction=fraction, area_group=area_group(fraction), **metrics))
            scores[arm] = sample_queries(logit, xy, truth.shape)
            predictions[arm] = np.where(valid, (scores[arm] >= 0).astype(np.int8), -1)
            point_pred[arm].extend(predictions[arm].tolist())
        for k in range(8):
            per_query.append(dict(frame_index=index, id=row['id'], native=k, qx=xy[k, 0], qy=xy[k, 1],
                coordinate_valid=bool(valid[k]), v61_source_valid=bool(oldvalid[k]), M4_presence=int(labels[k]),
                plain_logit=float(scores[ARMS[0]][k]), plain_pred=int(predictions[ARMS[0]][k]),
                CS_logit=float(scores[ARMS[1]][k]), CS_pred=int(predictions[ARMS[1]][k])))
        point_truth.extend(labels.tolist())
        point_valid.extend(valid.tolist())
        point_oldvalid.extend(oldvalid.tolist())
    y = np.asarray(point_truth)
    valid = np.asarray(point_valid)
    oldvalid = np.asarray(point_oldvalid)
    assert int(valid.sum()) == 1742
    metrics = {arm: aggregate([r for r in per_frame if r['arm'] == arm]) for arm in ARMS}
    bins = {arm: {name: aggregate([r for r in per_frame if r['arm'] == arm and r['area_group'] == name])
                  for name in ('lt1pct', '1to_lt5pct', '5to_lt10pct', 'ge10pct')
                  if any(r['arm'] == arm and r['area_group'] == name for r in per_frame)} for arm in ARMS}
    point_reports = {arm: {name: presence_summary(y[mask], np.asarray(point_pred[arm])[mask])
                         for name, mask in [('ORIGINAL_COORDINATE_VALID', valid), ('V61_SOURCE_VALID_PARITY', oldvalid)]}
                     for arm in ARMS}
    order = sorted(ids.tolist(), key=lambda i: (fractions[i], i))
    selected = []
    for q, indices in enumerate(np.array_split(np.asarray(order), 4), 1):
        selected.append(dict(frame_index=int(min(indices)), reason='FIRST_FRAME_IN_AREA_FRACTION_RANK_QUARTILE_'+str(q)))
    cs_records = [r for r in per_frame if r['arm'] == ARMS[1]]
    for reason, reverse in [('CS_LOWEST_UNUSED_IOU', False), ('CS_HIGHEST_UNUSED_IOU', True)]:
        available = [r for r in cs_records if r['frame_index'] not in {s['frame_index'] for s in selected}]
        r = min(available, key=lambda r: (-r['IoU'] if reverse else r['IoU'], r['frame_index']))
        selected.append(dict(frame_index=r['frame_index'], reason=reason))
    assert len({r['frame_index'] for r in selected}) == 6
    write_json(out/'SELECTION_BEFORE_RGB.json', dict(completed=True, RGB_decoded_so_far=0,
        rule='Four first-index area rank quartiles + unused CS IoU low/high, fixed before RGB inspection',
        selected=selected, diagnostic_examples_only_not_estimator=True))
    write_csv(out/'ORIGINAL_RESOLUTION_PER_FRAME.csv', per_frame)
    write_csv(out/'SYN_DEV_QUERY_PRESENCE.csv', per_query)
    coarse = complete['SYN_DEV_results']
    results = dict(completed=True, PASS=True, frames=235,
        original_resolution_metrics=metrics, original_resolution_area_groups=bins,
        frozen_coarse32_metrics=coarse, point_presence=point_reports,
        query_counts=dict(all_DEV_queries=len(y), coordinate_valid=int(valid.sum()),
                          original_invalid_U=int((~valid).sum()), v61_source_valid=int(oldvalid.sum()),
                          coordinate_valid_target=int((y[valid] == 1).sum()),
                          coordinate_valid_non_target=int((y[valid] == 0).sum())),
        thresholds=dict(M4_uint8_gt=127, predicted_logit_ge=0),
        semantic_limit='TARGET/NON_TARGET pixel presence, not native visible/external cause. Background, holes, foreign objects and query error can all be NON_TARGET.',
        original_mask_sources='Original user SYN1003 M4 subset DEV235, not internal V129440',
        selection_before_RGB=binding(out/'SELECTION_BEFORE_RGB.json'), current_or_real_reads=0,
        model_forwards=0, optimizer_updates=0)
    write_json(out/'RESULTS.json', results)
    global FONT
    font_path = safe(Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    FONT = ImageFont.truetype(str(font_path), 19)
    overview = Image.new('RGB', (1640, 6*300+65), 'white')
    od = ImageDraw.Draw(overview)
    od.text((12, 8), 'Full SYN DEV frames: RGB | original M4 | plain errors | CS architecture errors', font=FONT, fill='black')
    gallery = []
    for number, chosen in enumerate(selected):
        index = chosen['frame_index']
        row = inputs[index]
        with Image.open(io.BytesIO(checked(row['RGB']).read_bytes())) as im:
            rgb = np.array(im.convert('RGB'))
        truth = original_masks[index]
        j = int(np.flatnonzero(ids == index)[0])
        probs, overlays = {}, {}
        for arm in ARMS:
            full = original_logits(packets[arm]['mask_logits'][j], truth.shape)
            probs[arm] = 1/(1+np.exp(-np.clip(full.astype(np.float64), -80, 80)))
            overlays[arm] = error_overlay(rgb, truth, full >= 0)
        titles = ['Full RGB (no crop)', 'Original visible M4 GT', 'Plain CLIP target probability [0,1]',
                  'CS architecture target probability [0,1]', 'Plain: green TP / purple FP / red FN',
                  'CS: green TP / purple FP / red FN']
        images = [rgb, np.repeat((truth*255).astype(np.uint8)[..., None], 3, 2),
                  color_probability(probs[ARMS[0]]), color_probability(probs[ARMS[1]]), overlays[ARMS[0]], overlays[ARMS[1]]]
        board = Image.new('RGB', (1320, 1420), 'white')
        draw = ImageDraw.Draw(board)
        draw.text((10, 8), f'DEV frame {index} | {chosen["reason"]}', font=FONT, fill='black')
        stat = {r['arm']: r for r in per_frame if r['frame_index'] == index}
        draw.text((10, 35), f'M4 area {fractions[index]*100:.2f}% | original pixel IoU plain {stat[ARMS[0]]["IoU"]:.3f} / CS {stat[ARMS[1]]["IoU"]:.3f}', font=FONT, fill='black')
        for k, (image, title) in enumerate(zip(images, titles)):
            board.paste(panel(image, title), (10+(k%2)*650, 75+(k//2)*445))
        name = f'SYN_DEV_{index:04d}_FULL_FRAME.png'
        board.save(safe(out/name))
        shutil.copyfile(checked(row['RGB']), safe(out/f'SYN_DEV_{index:04d}_RGB.png'))
        shutil.copyfile(checked(targets[index]['M4']), safe(out/f'SYN_DEV_{index:04d}_M4.png'))
        for k, image in enumerate((images[0], images[1], images[4], images[5])):
            overview.paste(panel(image, ['RGB', 'Original M4', 'Plain errors', 'CS errors'][k], (400, 245)), (10+k*410, 60+number*300))
        gallery.append(dict(frame_index=index, reason=chosen['reason'], image=name))
    overview.save(safe(out/'SYN_FULL_FRAME_OVERVIEW_FIXED6.png'))
    rows_html = ''.join('<tr><td>'+html.escape(arm)+'</td><td>'+f'{metrics[arm]["IoU"]*100:.2f}'+'</td><td>'+f'{metrics[arm]["Dice"]*100:.2f}'+'</td><td>'+f'{metrics[arm]["mean_frame_IoU"]*100:.2f}'+'</td></tr>' for arm in ARMS)
    cards = ''.join(f'<h2>DEV {r["frame_index"]} · {html.escape(r["reason"])}</h2><a href="{r["image"]}"><img src="{r["image"]}"></a>' for r in gallery)
    page = '<!doctype html><html lang="ko"><meta charset="utf-8"><title>V267 합성 전체 영상 검증</title><style>body{font-family:sans-serif;max-width:1500px;margin:30px auto;line-height:1.7}img{max-width:100%}td,th{border:1px solid #bbb;padding:8px}table{border-collapse:collapse}</style><h1>원래 합성 DEV235장 · 전체 영상 M4 분할</h1><p>예측32×32 logit을 원래 영상 크기로 bilinear 복원하고 logit≥0으로 이진화했습니다. 원래 M4의 uint8&gt;127 정답과 비교합니다. 키포인트 crop·재학습·문턱 선택은 없습니다.</p><table><tr><th>방식</th><th>원해상도 micro IoU%</th><th>micro Dice%</th><th>평균 영상 IoU%</th></tr>'+rows_html+'</table><p>확률 색상은 모든 영상에서 고정0…1입니다. 초록=TP, 보라=FP, 빨강=FN. 그림은 전체 영상을 비율을 유지해 표시합니다. 6개 사례는 RGB를 읽기 전에 면적 사분위와 IoU 규칙으로 고정했습니다. 전체235장 지표가 성능 근거입니다.</p><p><a href="RESULTS.json">지표 JSON·32격자 대비</a> · <a href="ORIGINAL_RESOLUTION_PER_FRAME.csv">235장 두 방식 표</a> · <a href="SYN_DEV_QUERY_PRESENCE.csv">원래 query의 M4 존재 평가</a> · <a href="SELECTION_BEFORE_RGB.json">사례 선택 기록</a></p><p>Query 평가는 대상 픽셀 존재/비존재이며 native 보임/외부 가림 정답이 아닙니다. 비대상에는 바닥·구멍·다른 물체·초기 좌표 오류가 함께 포함됩니다.</p><img src="SYN_FULL_FRAME_OVERVIEW_FIXED6.png">'+cards+'</html>'
    safe(out/'INDEX_KO.html').write_text(page)
    files = [p for p in out.iterdir() if p.is_file()]
    write_json(out/'COMPLETE.json', dict(completed=True, PASS=True, code=binding(Path(__file__).absolute()),
        sources=source_bindings, outputs={p.name: binding(p) for p in files}, frames=235,
        selected_full_RGB_frames=6, RGB_decodes_after_selection_only=True,
        current_or_real_reads=0, model_forwards=0, optimizer_updates=0, elapsed_seconds=time.monotonic()-started))
    print(json.dumps(dict(PASS=True, completed=True, original_resolution=metrics,
                         point_presence=point_reports, complete=binding(out/'COMPLETE.json'))), flush=True)


if __name__ == '__main__':
    main()
