"""Additive source-only replay. Never imports the composer or any model."""
from pathlib import Path
import hashlib
import json
from collections import Counter, defaultdict
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
PARENT = HERE.parents[2]
V71 = PARENT / 'query_visibility_reference_jitter_20261023_v71'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def ref(p):
    p = Path(p)
    return dict(path=str(p), sha256=sha(p))


def load(p):
    return json.loads(Path(p).read_text())


def rows(p):
    return [json.loads(s) for s in Path(p).read_text().splitlines()]


def need(ok, why):
    if not ok:
        raise AssertionError(why)


def check_binding(r):
    need(sha(r['path']) == r['sha256'], 'SHA mismatch: ' + r['path'])


def actual_crop(image, q, length):
    """Independent bilinear implementation: four native integer neighbours."""
    h, w = image.shape[:2]
    u = (np.arange(96, dtype=np.float64) + .5) / 96 - .5
    xx, yy = np.meshgrid(q[0] + u * length, q[1] + u * length)
    x, y = np.floor(xx).astype(np.int64), np.floor(yy).astype(np.int64)
    fx, fy = xx-x, yy-y
    out = np.zeros((96, 96, 3), np.float64)
    for dy in (0, 1):
        for dx in (0, 1):
            ix, iy = x+dx, y+dy
            mask = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
            weight = (fx if dx else 1-fx) * (fy if dy else 1-fy)
            out[mask] += image[iy[mask], ix[mask]] * weight[mask, None]
    valid = (xx >= -.5) & (xx < w-.5) & (yy >= -.5) & (yy < h-.5)
    out = np.floor(np.clip(out, 0, 255) + .5).astype(np.uint8)
    out[~valid] = 0
    return out.transpose(2, 0, 1), valid[None]


def query_and_availability(q, length, hw, bb):
    h, w = hw
    valid = bool(np.isfinite(q).all() and 0 <= q[0] < w and 0 <= q[1] < h
                 and 0 <= np.floor(q[0]+.5) < w and 0 <= np.floor(q[1]+.5) < h)
    if not valid:
        return valid, 'INVALID'
    a = [max(-.5, q[0]-length/2), max(-.5, q[1]-length/2),
         min(w-.5, q[0]+length/2), min(h-.5, q[1]+length/2)]
    if a[0] <= bb[0] and a[1] <= bb[1] and a[2] >= bb[2] and a[3] >= bb[3]:
        return valid, 'FULL'
    overlap = min(a[2], bb[2]) > max(a[0], bb[0]) and min(a[3], bb[3]) > max(a[1], bb[1])
    return valid, 'PARTIAL' if overlap else 'MISS'


def main():
    for name in ['DATA_QA.json', 'FINAL_SEAL.json']:
        need(not (HERE/name).exists(), 'Refuse overwrite ' + name)
    receipt = load(HERE/'DATASET_RECEIPT.json')
    protocol = load(HERE/'COMPOSITION_PROTOCOL.json')
    need(receipt['completed'], 'Incomplete receipt')
    for r in [receipt['protocol'], receipt['code'], *receipt['outputs'], *receipt['galleries'],
              protocol['code'], *protocol['bindings']]:
        check_binding(r)
    old = np.load(V71/'data/ycb_proposal_grid/INPUT_ONLY.npz', allow_pickle=False)
    data = np.load(HERE/'MODEL_INPUT_ONLY.npz', allow_pickle=False)
    keys = {'L', 'box_xyxy', 'crop_valid96', 'query_valid', 'query_xy', 'rgb96'}
    need(set(data.files) == keys == set(old.files), 'Input-only whitelist')
    need(all(np.array_equal(data[k][:1944], old[k]) for k in keys), 'Original prefix changed')
    metadata = rows(HERE/'INPUT_INDEX_LINEAGE_ONLY.jsonl')
    teachers = rows(HERE/'GT_ONLY_TEACHERS.jsonl')
    need(len(metadata) == len(teachers) == 3645, 'Row count')
    lineage = {r['id']: r for r in rows(V71/'data/ycb_cue_review/SOURCE_LINEAGE_GT_ONLY.jsonl')}
    certificates = load(HERE/'INTERVENTION_CERTIFICATES_GT_ONLY.json')['variants']
    need(len(certificates) == 21, 'Variant count')
    deny = set(load(PARENT/'query_visibility_public_real_20261022_v63/contract/EVALUATION_SHA256_DENYLIST.json')['image_sha256'])
    source_images, cache, rgb_values, alpha_values = {}, {}, 0, 0
    intervention_rows, classes = [], defaultdict(set)

    def image(path):
        if path not in cache:
            cache[path] = np.array(Image.open(path).convert('RGB'))
        return cache[path]

    for cert in certificates:
        original = lineage[cert['target_id']]['original']
        donor = cert['donor']
        for r in [original['image'], donor['image'], donor['visible'], cert['image'], cert['GT_alpha']]:
            check_binding(r)
        for r in [original['image'], donor['image'], cert['image']]:
            need(r['sha256'] not in deny, 'Private RGB overlap')
            source_images[r['path']] = r['sha256']
        need(original['split'] == donor['split'] == cert['split'], 'Donor split')
        need(original['object_id'] != donor['object_id'], 'Same object donor')
        need(original['image']['sha256'] != donor['image']['sha256'], 'Same frame donor')
        classes[cert['split']].add(donor['object_id'])
        rgb, donor_rgb = image(original['image']['path']), image(donor['image']['path'])
        mask = np.array(Image.open(donor['visible']['path'])) > 0
        ys, xs = np.nonzero(mask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)]
        need(bbox == cert['donor_crop_native_bbox'], 'Donor bbox')
        x0, y0, x1, y1 = bbox
        piece, piece_mask = donor_rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]
        tx, ty = cert['certificate']['integer_translation_xy']
        h, w = rgb.shape[:2]
        ph, pw = piece.shape[:2]
        alpha = np.zeros((h, w), bool)
        pasted_rgb = np.zeros_like(rgb)
        ax, ay, bx, by = max(0, tx), max(0, ty), min(w, tx+pw), min(h, ty+ph)
        need(bx > ax and by > ay, 'Empty paste')
        alpha[ay:by, ax:bx] = piece_mask[ay-ty:by-ty, ax-tx:bx-tx]
        pasted_rgb[ay:by, ax:bx] = piece[ay-ty:by-ty, ax-tx:bx-tx]
        expected = rgb.copy()
        expected[alpha] = pasted_rgb[alpha]
        actual_alpha = np.array(Image.open(cert['GT_alpha']['path']))
        need(set(np.unique(actual_alpha)) <= {0, 255}, 'Nonbinary alpha')
        need(np.array_equal(actual_alpha > 0, alpha), 'Alpha replay')
        need(np.array_equal(image(cert['image']['path']), expected), 'Every RGB channel replay')
        alpha_values += alpha.size
        rgb_values += expected.size
        a, b, c, d = cert['uncertainty_bbox_xyxy']
        cells = alpha[b:d, a:c]
        count, total = int(cells.sum()), int(cells.size)
        need(total > 0, 'Empty uncertainty')
        variant = cert['variant']
        need((variant == 'FULL_COVER' and count == total)
             or (variant == 'PARTIAL_COVER' and 0 < count < total)
             or (variant == 'NEUTRAL_D' and count == 0), 'Reference coverage certificate')
        need(count == cert['certificate']['uncertainty_opaque_cells'] and total == cert['certificate']['uncertainty_total_cells'], 'Coverage count')
        ids = cert['input_indices']
        need(len(ids) == 81 and {metadata[i]['grid_row']*9+metadata[i]['grid_col'] for i in ids} == set(range(81)), 'All grid queries')
        intervention_rows.append(dict(target_id=cert['target_id'], variant=variant, native_uncertainty_count=count, native_uncertainty_total=total))

    full_groups, crop_groups = defaultdict(list), defaultdict(list)
    new_crop_channels, validity_cells = 0, 0
    for i, (m, t) in enumerate(zip(metadata, teachers)):
        need(m['input_index'] == t['input_index'] == i and m['row_id'] == t['row_id'], 'Ordered row identity')
        need(m['split'] == t['split'] == lineage[m['original_target_id']]['split'], 'Split lineage')
        need(m['cue_group'] == t['cue_refID'] and t['native8_semantic_ID'] == 'UNKNOWN', 'Reference identity')
        need(np.array_equal(data['query_xy'][i], m['query_xy']) and np.array_equal(data['box_xyxy'][i], m['box_xyxy']) and data['L'][i] == m['L'], 'Input metadata')
        valid, availability = query_and_availability(np.asarray(m['query_xy']), m['L'], m['raw_hw'], t['uncertainty_bbox_xyxy'])
        need(valid == bool(data['query_valid'][i]) == m['native_query_valid_from_metadata'], 'Query valid')
        need(availability == t['availability'], 'Availability')
        expected_state = t['reference_state'] if availability == 'FULL' and t['reference_state'] in ('D', 'E') else 'U'
        need(t['training_state'] == expected_state and t['reference_or_GT_is_not_model_input'], 'Teacher supervision')
        actual_image = image(m['image'])
        need(list(actual_image.shape[:2]) == m['raw_hw'] and sha(m['image']) == m['image_sha256'], 'RGB identity/shape')
        need(hashlib.sha256(actual_image.tobytes()).hexdigest() == m['decoded_fullRGB_sha256'], 'Decoded RGB SHA')
        u = (np.arange(96, dtype=np.float64)+.5)/96-.5
        sx, sy = m['query_xy'][0]+u*m['L'], m['query_xy'][1]+u*m['L']
        h, w = m['raw_hw']
        cv = ((sy[:, None] >= -.5) & (sy[:, None] < h-.5) & (sx[None] >= -.5) & (sx[None] < w-.5))[None]
        need(np.array_equal(cv, data['crop_valid96'][i]), 'All crop validity')
        validity_cells += cv.size
        if i >= 1944:
            oldi = m['paired_original_input_index']
            need(oldi < 1944 and metadata[oldi]['variant'] == 'BASE_ORIGINAL', 'Pair original')
            for key in ['L', 'query_xy', 'box_xyxy']:
                need(np.array_equal(data[key][i], data[key][oldi]), 'Moved query/box/L')
            need(t['reference_xy'] == teachers[oldi]['reference_xy'] and t['uncertainty_bbox_xyxy'] == teachers[oldi]['uncertainty_bbox_xyxy'], 'Moved reference')
            expected_rgb, expected_valid = actual_crop(actual_image, m['query_xy'], m['L'])
            need(np.array_equal(expected_rgb, data['rgb96'][i]) and np.array_equal(expected_valid, data['crop_valid96'][i]), 'Full new crop replay')
            new_crop_channels += expected_rgb.size
        geometry = np.asarray([*m['query_xy'], m['L'], *m['box_xyxy']], dtype=np.float64).tobytes()
        full_groups[(m['decoded_fullRGB_sha256'], geometry)].append(i)
        digest = hashlib.sha256(data['rgb96'][i].tobytes()+data['crop_valid96'][i].tobytes()+data['query_valid'][i].tobytes()+geometry).hexdigest()
        crop_groups[digest].append(i)

    conflict = load(HERE/'INPUT_CONFLICT_AUDIT.json')
    summaries = {}
    for name, groups in [('full_RGB_query_L_box', full_groups), ('effective_crop_valid_query_L_box', crop_groups)]:
        semantic = [ids for ids in groups.values() if {'D','E'} <= {teachers[i]['reference_state'] for i in ids}]
        supervised = [ids for ids in groups.values() if {'D','E'} <= {teachers[i]['training_state'] for i in ids}]
        summaries[name] = dict(groups=len(groups), duplicate_groups=sum(len(x)>1 for x in groups.values()), reference_D_E_collision_groups=len(semantic), supervised_D_E_conflicts=len(supervised))
        for k,v in summaries[name].items():
            need(conflict[name][k] == v, 'Conflict count mismatch '+k)
        need(not supervised, 'Contradictory supervised input')
    need(Counter(t['training_state'] for t in teachers) == receipt['counts']['training'], 'Training counts')
    qa = dict(schema='V72_YCB_REFERENCE_COVER_ADDITIVE_ACTUAL_QA_V1', PASS=True,
              immutable_receipt=ref(HERE/'DATASET_RECEIPT.json'), audit_code=ref(__file__),
              all_protocol_receipt_outputs_galleries_and_parent_bindings_rehashed=True,
              old1944_prefix_all6_arrays_equal=True, actual_native_alpha_pixels_replayed=alpha_values,
              actual_fullRGB_channel_values_replayed=rgb_values, new1701_full_crop_channel_values_replayed=new_crop_channels,
              all3645_crop_valid_cells_checked=validity_cells, all_grid_sites_refs_L_boxes_and_split_lineage_unchanged=True,
              model_input_keys=sorted(keys), GT_input_fields=0, private_SHA_overlap=0,
              donor_classes_by_split={k:sorted(v) for k,v in classes.items()}, interventions=intervention_rows,
              conflict_recomputed=summaries,
              limitation='299 identical effective-input groups have hidden reference D/E differences. All conflicting rows are not simultaneously supervised D/E (zero strict conflicts); this does not establish global semantic identifiability.',
              actual_visual_QA=dict(views=receipt['galleries'], all7_opened_and_inspected=True, result='PASS: actual full/native cue plus FULL/PARTIAL/NEUTRAL effects readable; original q/L unchanged'),
              conditional_AI_raster_teacher_not_human_or_native8_gold=True,
              counts=receipt['counts'], by_split=receipt['by_split'], new_model_forward_training_current_PnP=0)
    (HERE/'DATA_QA.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2)+'\n')
    paths = sorted(p for p in HERE.rglob('*') if p.is_file() and p.name != 'FINAL_SEAL.json' and '__pycache__' not in p.parts)
    seal = dict(schema='V72_YCB_REFERENCE_COVER_FINAL_DATA_SEAL_V1', completed=True, PASS=True,
                scope='Public-real controlled reference-cue data only; no model admission/result/current gate.',
                receipt=ref(HERE/'DATASET_RECEIPT.json'), QA=ref(HERE/'DATA_QA.json'), files=[ref(p) for p in paths],
                n_inputs=3645, training_D=63, training_E=21, training_U=3561,
                source_teacher_conditional=True, supervised_input_conflicts=0, hidden_reference_collision_groups=299,
                new_model_forward_training_current_PnP=0)
    (HERE/'FINAL_SEAL.json').write_text(json.dumps(seal,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(PASS=True, QA=ref(HERE/'DATA_QA.json'), seal=ref(HERE/'FINAL_SEAL.json'), counts=receipt['counts'], conflicts=summaries),ensure_ascii=False))


if __name__ == '__main__':
    main()
