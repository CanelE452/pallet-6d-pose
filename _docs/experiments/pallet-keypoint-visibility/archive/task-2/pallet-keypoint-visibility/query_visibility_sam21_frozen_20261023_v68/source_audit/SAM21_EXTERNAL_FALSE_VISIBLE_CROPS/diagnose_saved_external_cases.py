"""Source-only saved-mask/RGB intervention diagnosis; no models or features."""
from pathlib import Path
import csv
import hashlib
import json
import math
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V66 = ROOT.parent / 'query_visibility_controlled_real_20261022_v66'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))

def require(condition, message):
    if not condition:
        raise ValueError(message)

def load_json(path):
    return json.loads(Path(path).read_text())

def rows(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))

def boolstr(value):
    require(value in ('True', 'False'), 'Explicit saved Boolean')
    return value == 'True'

def checked(binding):
    require(sha(binding['path']) == binding['sha256'], 'Changed saved file: ' + binding['path'])
    return Path(binding['path'])

def rgb(binding):
    image = cv2.imread(str(checked(binding)), cv2.IMREAD_COLOR)
    require(image is not None, 'Source RGB decode failed')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def binary_mask(binding):
    mask = cv2.imread(str(checked(binding)), cv2.IMREAD_GRAYSCALE)
    require(mask is not None, 'Source mask decode failed')
    return mask > 0

def strict25(mask, x, y):
    return bool(x >= 2 and y >= 2 and x + 2 < mask.shape[1] and y + 2 < mask.shape[0]
                and mask[y - 2:y + 3, x - 2:x + 3].all())

def replay_paste(composite, original, donor_rgb, donor_visible):
    tr = composite['transform']; x0, y0, x1, y1 = tr['donor_visible_bbox_xyxy']
    yy, xx = np.nonzero(donor_visible)
    require([int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1] == [x0, y0, x1, y1], 'Saved donor visible bounds')
    height, width = tr['resized_hw']
    resized_rgb = cv2.resize(donor_rgb[y0:y1, x0:x1], (width, height), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(donor_visible[y0:y1, x0:x1].astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    left, top = tr['translation_xy']; h, w = original.shape[:2]
    dx0, dy0, dx1, dy1 = max(0, left), max(0, top), min(w, left + width), min(h, top + height)
    sx, sy = dx0 - left, dy0 - top
    paste = np.zeros((h, w), bool); donor_canvas = np.zeros_like(original)
    paste[dy0:dy1, dx0:dx1] = resized_mask[sy:sy + dy1 - dy0, sx:sx + dx1 - dx0]
    donor_canvas[dy0:dy1, dx0:dx1] = resized_rgb[sy:sy + dy1 - dy0, sx:sx + dx1 - dx0]
    replay = original.copy(); replay[paste] = donor_canvas[paste]
    return paste, donor_canvas, replay

def panel(ax, image, window, q, pixel, box, title, *, original_visible=None, paste=None, chosen=None):
    ax.imshow(image, interpolation='nearest')
    if chosen is not None:
        overlay = np.zeros((*chosen.shape, 4), np.float32)
        overlay[chosen] = [0., .75, 1., .32]
        ax.imshow(overlay, interpolation='nearest')
    if original_visible is not None:
        ax.contour(original_visible.astype(np.uint8), levels=[.5], colors=['#1bb149'], linewidths=.9)
    if paste is not None:
        ax.contour(paste.astype(np.uint8), levels=[.5], colors=['#ff8a00'], linewidths=1.)
    ax.add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor='#d6a900', linewidth=.8))
    px, py = pixel
    ax.add_patch(Rectangle((px - 2.5, py - 2.5), 5, 5, fill=False, edgecolor='#ffff00', linewidth=1.3))
    ax.scatter([q[0]], [q[1]], marker='+', c='#ff00bf', s=85, linewidths=1.4)
    ax.add_patch(Rectangle((px - .5, py - .5), 1, 1, fill=False, edgecolor='white', linewidth=1.1))
    x0, y0, x1, y1 = window
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

def render(path, case, original, composed, visible, paste, mask, q, pixel, box, side):
    h, w = paste.shape
    padding = max(box[2] - box[0], box[3] - box[1]) * .12
    context = [max(0., min(box[0] - padding, q[0] - 6)), max(0., min(box[1] - padding, q[1] - 6)),
               min(w - 1., max(box[2] + padding, q[0] + 6)), min(h - 1., max(box[3] + padding, q[1] + 6))]
    length = max(9, int(math.ceil(side)))
    native = [q[0] - length / 2, q[1] - length / 2, q[0] + length / 2, q[1] + length / 2]
    fig, ax = plt.subplots(2, 3, figsize=(15, 9), gridspec_kw={'height_ratios': [1.2, 1.]})
    for i, window in enumerate((context, native)):
        prefix = 'Prompt context' if i == 0 else f'Native crop, side={length}px'
        panel(ax[i, 0], original, window, q, pixel, box, prefix + ': original target', original_visible=visible)
        panel(ax[i, 1], composed, window, q, pixel, box, prefix + ': opaque other-object paste', paste=paste)
        panel(ax[i, 2], composed, window, q, pixel, box, prefix + ': saved SAM2 mask (cyan)', paste=paste, chosen=mask)
    fig.suptitle('Saved SAM2 false VISIBLE on source-certified external cutout\n' + case['sample_id'], fontsize=13, y=.985)
    fig.text(.5, .045, f"q=({q[0]:.3f}, {q[1]:.3f}); native pixel={tuple(pixel)}; chosen logit={case['saved_exact_logit']:.4f}; "
             f"branch={case['selected_mask_index']}; strict25E={case['strict25_E_verified']}; mask-positive5x5={case['mask_positive_5x5']}/25", ha='center', fontsize=10)
    fig.text(.5, .02, 'Green: original visible target; orange: pasted object edge; magenta: unchanged query; yellow: native5x5. '
             'BOP/cutout evidence, not human semantic-corner truth.', ha='center', fontsize=9)
    fig.subplots_adjust(left=.015, right=.985, top=.92, bottom=.08, wspace=.04, hspace=.12)
    fig.savefig(path, dpi=150); plt.close(fig)
    return dict(prompt_context_window_xyxy=context, query_display_window_xyxy=native, display_crop_side=length,
                display_resampling='nearest; source RGB native pixels preserved')

def main():
    output = HERE / 'DIAGNOSIS.json'; require(not output.exists(), 'Preserve previous diagnosis')
    recipe = ref(HERE / 'RECIPE.json')
    input_bindings = [recipe, ref(__file__)]
    query_path = ROOT / 'source_outputs/SAM21_BASE_PLUS/PER_QUERY.csv'
    target_path = ROOT / 'source_outputs/SAM21_BASE_PLUS/TARGET_OUTPUTS.json'
    extraction_seal_path = ROOT / 'source_outputs/SAM21_BASE_PLUS/EXTRACTION_SEAL.json'
    seal = load_json(extraction_seal_path)
    require(seal['completed'] and seal['source_only'] and not seal['current_RGB_GT_scores_inputs_read'], 'Sealed source extraction required')
    sealed_outputs = {Path(b['path']).name: b for b in seal['outputs']}
    checked(sealed_outputs['PER_QUERY.csv']); checked(sealed_outputs['TARGET_OUTPUTS.json'])
    source_input = ROOT / 'source_inputs/INPUTS_RGB_ONLY.jsonl'
    manifest = {r['global_id']: r for r in map(json.loads, source_input.open())}
    score = {r['sample_id']: r for r in rows(query_path)}
    target = {r['global_id']: r for r in load_json(target_path)}
    composite_path = V66 / 'controlled/COMPOSITES.jsonl'
    composites = {r['id']: r for r in map(json.loads, composite_path.open()) if r['split'] == 'VALIDATION'}
    boundary_path = V66 / 'source_audit/SOURCE_BOUNDARY_FAILURE_DIAGNOSIS_QUERY_ROWS.csv'
    boundary = {r['sample_id']: r for r in rows(boundary_path) if r['model'] == 'V65_BASELINE' and r['source'] == 'CONTROLLED_YCB'}
    gtpath = V66 / 'label_cache_v2/CONTROLLED_VAL.npz'
    keys = ['sample_id', 'global_id', 'target_index', 'native_corner', 'query', 'side', 'image_sha256', 'dense_allowed', 'dense_valid', 'dense_exact_state', 'dense_radius2_state']
    with np.load(gtpath, allow_pickle=False) as z:
        gt = {k: z[k] for k in keys}
    require(len(gt['sample_id']) == len(boundary) == 3200, 'Fixed controlled source query cohort')
    eligible = []
    known_E = 0
    for i, sid in enumerate(gt['sample_id'].astype(str)):
        r = score[sid]; b = boundary[sid]; gid = str(gt['global_id'][gt['target_index'][i]])
        require(r['source'] == 'CONTROLLED_YCB' and r['global_id'] == gid, 'Same source/target identity')
        valid = bool(gt['dense_valid'][i, 4, 4]); e = int(gt['dense_exact_state'][i, 4, 4]) == 1
        if not (valid and e):
            continue
        known_E += 1
        require(int(b['exact_GT']) == 1 and int(b['allowed']) == int(gt['dense_allowed'][i, 4, 4]), 'Frozen E reference')
        if int(r['exact_visible']) == 1:
            require(float(r['exact_logit']) > 0, 'Strict saved positive')
            eligible.append(dict(index=i, sample_id=sid, global_id=gid, strict25_E=boolstr(b['strict25_E']),
                                 selection_sha256=hashlib.sha256(sid.encode()).hexdigest()))
    eligible.sort(key=lambda r: (not r['strict25_E'], r['selection_sha256']))
    selected = []; used = set()
    for r in eligible:
        if r['global_id'] in used:
            continue
        selected.append(r); used.add(r['global_id'])
        if len(selected) == 4:
            break
    require(len(selected) == 4, 'Four distinct controlled false-visible targets required')
    selection_path = HERE / 'SELECTION.json'
    selection = dict(recipe=recipe, known_E_queries=known_E, false_visible_queries=len(eligible),
        strict25_E_false_visible_queries=sum(r['strict25_E'] for r in eligible), selected=selected,
        confidence_or_logit_magnitude_ranked=False, mask_branches_changed=False)
    selection_path.write_text(json.dumps(selection, indent=2) + '\n')
    for p in (query_path, target_path, extraction_seal_path, source_input, composite_path, boundary_path, gtpath):
        input_bindings.append(ref(p))
    cases = []
    for rank, selected_case in enumerate(selected, 1):
        i = selected_case['index']; sid = selected_case['sample_id']; gid = selected_case['global_id']
        s, m, t, c = score[sid], manifest[gid], target[gid], composites[gid]
        k = int(gt['native_corner'][i]); q = gt['query'][i]; px, py = np.floor(q + .5).astype(int)
        require(q.tolist() == c['query'][k] and m['box'] == c['box'] == t['box'], 'Unchanged query/box')
        require([float(s['query_x']), float(s['query_y'])] == q.tolist() and int(s['native_corner']) == k, 'Saved query/slot')
        require(c['image']['sha256'] == m['image_sha256'] == t['image_sha256'] == s['image_sha256'] == str(gt['image_sha256'][i]), 'Source RGB SHA identity')
        require(c['target_original']['id'] != c['occluder_original']['id'] and c['target_original']['image']['sha256'] != c['occluder_original']['image']['sha256'], 'Independent other-object RGB')
        original = rgb(c['target_original']['image']); composed = rgb(c['image'])
        donor_rgb = rgb(c['occluder_original']['image']); donor_visible = binary_mask(c['occluder_original']['visible'])
        original_visible_file = binary_mask(c['target_original']['visible'])
        with np.load(checked(c['labels']), allow_pickle=False) as z:
            cert = {name: z[name] for name in ('original_visible', 'paste_mask', 'visible_positive', 'controlled_E_pixel')}
        visible = cert['original_visible'].astype(bool); paste = cert['paste_mask'].astype(bool)
        require(np.array_equal(visible, original_visible_file), 'Original visible ownership matches source mask')
        require(np.array_equal(cert['visible_positive'], visible & ~paste) and np.array_equal(cert['controlled_E_pixel'], visible & paste), 'Controlled E/V certificate replay')
        reconstructed_paste, donor_canvas, replay = replay_paste(c, original, donor_rgb, donor_visible)
        require(np.array_equal(reconstructed_paste, paste) and np.array_equal(replay, composed), 'Opaque donor RGB/support must replay bit-exact')
        require(visible[py, px] and paste[py, px] and not cert['visible_positive'][py, px], 'Exact native point is certified newly occluded target')
        new_E = visible & paste; strict = strict25(new_E, px, py)
        require(strict == selected_case['strict25_E'], 'Strict25E boundary metadata replay')
        saved_mask_path = checked(t['mask_output'])
        with np.load(saved_mask_path, allow_pickle=False) as z:
            logit, quality, branch = z['chosen_logit'], z['quality_scores'], int(z['selected_index'])
        require(logit.dtype == np.float32 and logit.shape == paste.shape and np.isfinite(logit).all(), 'Native saved mask logits')
        require(branch == int(t['selected_mask_index']) == int(s['selected_mask_index']) == int(np.argmax(quality)), 'Previously chosen predicted-quality argmax only')
        exact, near = float(logit[py, px]), float(logit[py - 2:py + 3, px - 2:px + 3].max())
        require(exact == float(s['exact_logit']) > 0 and near == float(s['radius2_logit']), 'Saved exact/radius2 replay')
        positive = logit > 0; remain = visible & ~paste
        _, components = cv2.connectedComponents(positive.astype(np.uint8), connectivity=8)
        component = components == components[py, px]
        distance = cv2.distanceTransform(np.pad(paste.astype(np.uint8), 1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
        qmask = positive[py - 2:py + 3, px - 2:px + 3]
        case = dict(rank=rank, **selected_case, native_corner=k, query=q.tolist(), pixel=[int(px), int(py)],
            source='CONTROLLED_YCB', original_target_id=c['target_original']['id'], donor_id=c['occluder_original']['id'],
            original_target_object_id=c['target_original']['object_id'], donor_object_id=c['occluder_original']['object_id'],
            RGB_bindings=dict(original=c['target_original']['image'], controlled=c['image'], donor=c['occluder_original']['image']),
            labels=c['labels'], saved_mask=t['mask_output'], source_input_box=c['box'], side=float(gt['side'][i]),
            target_original_visible_at_query=True, new_occluder_at_query=True, target_visible_after_at_query=False,
            strict25_E_verified=strict, opaque_RGB_support_full_replay_bit_exact=True,
            RGB_outside_paste_bit_exact=bool(np.array_equal(composed[~paste], original[~paste])),
            RGB_inside_paste_matches_resized_donor=bool(np.array_equal(composed[paste], donor_canvas[paste])),
            query_RGB_original=original[py, px].tolist(), query_RGB_controlled=composed[py, px].tolist(), query_RGB_resized_donor=donor_canvas[py, px].tolist(),
            query_to_paste_exterior_distance_px=float(distance[py, px]),
            selected_mask_index=branch, predicted_quality=quality.astype(float).tolist(), saved_exact_logit=exact, saved_radius2_logit=near,
            mask_positive_5x5=int(qmask.sum()),
            all_controlled_E_pixels=int(new_E.sum()), SAM_positive_on_controlled_E_pixels=int((positive & new_E).sum()),
            all_paste_pixels=int(paste.sum()), SAM_positive_on_paste_pixels=int((positive & paste).sum()),
            remaining_target_visible_pixels=int(remain.sum()), SAM_positive_on_remaining_target_visible_pixels=int((positive & remain).sum()),
            query_component_pixels=int(component.sum()), query_component_controlled_E_pixels=int((component & new_E).sum()),
            query_component_remaining_target_visible_pixels=int((component & remain).sum()),
            interpretation_status='Mechanical proof complete; visual ownership interpretation pending',
            evidence_limit='Known controlled-source intervention using BOP original-visible ownership, not human semantic-corner gold.')
        imagepath = HERE / f'{rank:02d}_{selected_case["selection_sha256"][:12]}.png'
        case['display'] = render(imagepath, case, original, composed, visible, paste, positive, q, [int(px), int(py)], c['box'], float(gt['side'][i]))
        case['PNG'] = ref(imagepath)
        (HERE / f'{rank:02d}_CASE.json').write_text(json.dumps(case, indent=2, allow_nan=False) + '\n')
        cases.append(case)
        for b in (c['target_original']['image'], c['image'], c['occluder_original']['image'], c['target_original']['visible'], c['occluder_original']['visible'], c['labels'], t['mask_output']):
            input_bindings.append(b)
    thumbs = []
    for case in cases:
        image = Image.open(case['PNG']['path']).convert('RGB')
        image.thumbnail((1500, 900), Image.Resampling.LANCZOS); thumbs.append(image)
    canvas = Image.new('RGB', (3000, 1800), 'white')
    for i, image in enumerate(thumbs):
        canvas.paste(image, ((i % 2) * 1500, (i // 2) * 900))
    contact = HERE / 'CONTACT_SHEET.png'; canvas.save(contact)
    for binding in input_bindings:
        require(sha(binding['path']) == binding['sha256'], 'Input state changed during diagnosis')
    report = dict(schema='V68_SOURCE_SAM21_EXTERNAL_FALSE_VISIBLE_CROP_DIAGNOSIS_V1', completed=True,
        selection=ref(selection_path), recipe=recipe, script=ref(__file__), cases=cases,
        contact_sheet=ref(contact), input_bindings=input_bindings,
        source_RGB_reads_authorized_for_saved_output_diagnosis=True, visual_review_pending=True,
        model_or_GT_branch_reselection=False, confidence_ranked_selection=False,
        new_model_forward_calls=0, new_training=0, weights_or_features_reads=0, GPU_calls=0,
        current_RGB_GT_scores_inputs_reads=0, threshold_or_radius_changes=0, modified_prior_artifacts=False)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(completed=True, diagnosis=ref(output), known_E=known_E, eligible=len(eligible),
        strict_eligible=sum(r['strict25_E'] for r in eligible), selected=[c['sample_id'] for c in cases], contact_sheet=ref(contact))))

if __name__ == '__main__':
    main()
