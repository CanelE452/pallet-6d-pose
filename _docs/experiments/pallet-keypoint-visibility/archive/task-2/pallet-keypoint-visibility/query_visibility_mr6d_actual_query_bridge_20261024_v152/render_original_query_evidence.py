"""Render factual original-query crops after the frozen source prediction seal.

No model, GT role assignment, or semantic labels. Images are evidence figures.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

V = Path(__file__).resolve().parent
P = V.parent
CROP = P / 'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'
CROP_SHA = '6a6c2543e28715e3d47e15c543c25f75ff7d3efa3b881e523bd3451817fc004b'
POLICY = V / 'ROOT_POLICY_BEFORE_MODEL_OR_REFERENCE_BRIDGE.json'
POLICY_SHA = 'a941442d93e0a49554be86716fd324a4ad3bed01d629abb2d2ed77212bce7270'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def check(binding):
    assert sha(binding['path']) == binding['sha256']
    return Path(binding['path'])


def font(size):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {key: clean(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(val) for val in value]
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prediction-seal', required=True, type=Path)
    args = ap.parse_args()
    assert sha(POLICY) == POLICY_SHA and sha(CROP) == CROP_SHA
    seal = json.loads(args.prediction_seal.read_text())
    assert seal['completed'] and seal['PASS'] and seal['frames'] == 10
    assert seal['rows'] == 80 and seal['actual_original_initializer']
    files = {Path(item['path']).name: check(item) for item in seal['outputs']}
    propfile = next(path for name, path in files.items() if name.startswith('FROZEN_PROPOSALS_'))
    queryfile = next(path for name, path in files.items() if name.endswith('_AND_VALID9X9_INPUT_ONLY.npz'))
    props = [json.loads(line) for line in propfile.read_text().splitlines() if line]
    with np.load(queryfile, allow_pickle=False) as packet:
        arrays = {name: packet[name].copy() for name in packet.files}
    assert len(props) == 10 and arrays['query_xy'].shape == (80, 2)
    nodes = [node for node in ast.parse(CROP.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name == 'crop']
    assert len(nodes) == 1
    scope = {'np': np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(CROP), 'exec'), scope)
    crop = scope['crop']
    out = V / 'original_query_evidence'
    out.mkdir(exist_ok=False)
    facts, sheets, thumbnails = [], [], []
    for i, prop in enumerate(props):
        assert i == prop['frame_index']
        rgb = np.load(check(prop['canonical_RGB_NPY']), allow_pickle=False)
        assert rgb.dtype == np.uint8 and list(rgb.shape) == prop['raw_hw'] + [3]
        assert hashlib.sha256(rgb.tobytes()).hexdigest() == prop['decoded_RGB_sha256']
        assert sha(prop['image']) == prop['image_sha256']
        raw = Image.fromarray(rgb)
        marked = raw.copy()
        draw = ImageDraw.Draw(marked)
        hh, ww = rgb.shape[:2]
        selected = prop['selected_index']
        expected = None if not prop['candidates'] else int(np.argmax([c['score'] for c in prop['candidates']]))
        assert selected == expected
        # Every candidate box remains visible; selected candidate is orange.
        for candidate in prop['candidates']:
            box = candidate['box_xyxy']
            color = (255, 150, 0) if candidate['candidate_index'] == selected else (130, 130, 130)
            draw.rectangle(box, outline=color, width=3 if candidate['candidate_index'] == selected else 1)
            draw.text((max(0, box[0]), max(0, box[1] - 18)),
                      f"c{candidate['candidate_index']} {candidate['score']:.3f}",
                      fill=color, font=font(16), stroke_width=1, stroke_fill='black')
        crops, validities = [], []
        for slot in range(8):
            j = i * 8 + slot
            q, side = arrays['query_xy'][j], arrays['L'][j]
            construct = bool(arrays['finite_non_sentinel'][j] and np.isfinite(side) and side > 0)
            if construct:
                cr, valid = crop(rgb, q, side, 96)
                cr, valid = cr.transpose(1, 2, 0), valid[0]
            else:
                cr = np.zeros((96, 96, 3), np.uint8)
                valid = np.zeros((96, 96), bool)
            crops.append(cr)
            validities.append(valid)
            if np.isfinite(q).all() and 0 <= q[0] < ww and 0 <= q[1] < hh:
                x, y = q
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=(255, 0, 210), width=2)
                draw.text((min(max(x + 6, 0), ww - 25), min(max(y - 20, 0), hh - 24)),
                          str(slot), font=font(20), fill=(255, 0, 210), stroke_width=1, stroke_fill='black')
            facts.append(clean(dict(global_index=j, frame_index=i, frame_id=prop['id'],
                source_id=prop['source_id'], native_slot=slot, original_query=q, original_L=side,
                selected_index=selected, no_detection=bool(arrays['no_detection'][j]),
                finite_non_sentinel=bool(arrays['finite_non_sentinel'][j]),
                continuous_valid=bool(arrays['continuous_valid'][j]),
                nearest_valid=bool(arrays['nearest_valid'][j]), crop_constructed=construct,
                crop_valid_fraction=float(valid.mean()), rgb96_SHA256=hashlib.sha256(cr.tobytes()).hexdigest(),
                valid96_SHA256=hashlib.sha256(valid.tobytes()).hexdigest(),
                semantic_state='UNREVIEWED', local_query_evidence='UNREVIEWED', human_gold=False)))
        width, height = 800, round(hh * 800 / ww)
        canvas = Image.new('RGB', (1600, 125 + height + 270), 'white')
        cd = ImageDraw.Draw(canvas)
        cd.text((12, 8), f"MR6D {prop['source_id']} | original firstargmax={selected} | candidates={len(prop['candidates'])}", font=font(22), fill='black')
        cd.text((12, 40), 'Left: original RGB. Right: unchanged q8 magenta / selected box orange / other candidates gray.', font=font(20), fill='black')
        cd.text((12, 72), 'Below: original q/L factual96 crops, nearest x2. No semantic labels or coordinate correction.', font=font(20), fill='black')
        cd.text((12, 101), 'Gouda et al., MR6D, CC-BY-4.0. Crop display only; no learned feature or classification call.', font=font(16), fill='black')
        canvas.paste(raw.resize((width, height), Image.Resampling.NEAREST), (0, 125))
        canvas.paste(marked.resize((width, height), Image.Resampling.NEAREST), (width, 125))
        for slot in range(8):
            item = facts[i * 8 + slot]
            x, y = slot * 200, 135 + height
            cd.text((x + 5, y), f"native{slot} valid={item['nearest_valid']}", font=font(16), fill='black')
            cd.text((x + 5, y + 22), f"L={item['original_L']:.2f}" if item['original_L'] is not None else 'no query', font=font(16), fill='black')
            canvas.paste(Image.fromarray(crops[slot]).resize((192, 192), Image.Resampling.NEAREST), (x + 4, y + 48))
        fig = out / f'FRAME_{i:02d}_ORIGINAL_Q_AND_CROPS.png'
        canvas.save(fig)
        packet = out / f'FRAME_{i:02d}_FACTUAL96.npz'
        np.savez_compressed(packet, rgb96=np.stack(crops), valid96=np.stack(validities), global_index=np.arange(i * 8, i * 8 + 8))
        sheets.append(dict(frame_index=i, source_id=prop['source_id'], sheet=bind(fig), crop_packet=bind(packet)))
        thumbnails.append(marked.resize((640, 360), Image.Resampling.NEAREST))
    assert len(facts) == 80
    factpath = out / 'ALL80_ORIGINAL_QUERY_CROP_FACTS.jsonl'
    factpath.write_text(''.join(json.dumps(f, ensure_ascii=False, allow_nan=False) + '\n' for f in facts))
    overview_paths = []
    for start in range(0, 10, 4):
        ov = Image.new('RGB', (1280, 840), 'white')
        od = ImageDraw.Draw(ov)
        for off in range(min(4, 10 - start)):
            x, y = (off % 2) * 640, (off // 2) * 420
            od.text((x + 5, y + 8), f"#{start+off:02d} {props[start+off]['source_id']}", font=font(21), fill='black')
            ov.paste(thumbnails[start + off], (x, y + 48))
        path = out / f'OVERVIEW_{start//4:02d}.png'
        ov.save(path)
        overview_paths.append(bind(path))
    result = dict(completed=True, PASS=True, code=bind(__file__), policy=bind(POLICY),
        prediction_seal=bind(args.prediction_seal), crop_operator=bind(CROP), frames=10, slots=80,
        sheets=sheets, overviews=overview_paths, crop_facts=bind(factpath), actual_visual_review_pending=True,
        original_q_L_candidate_selection_unchanged=True, semantic_labels_created=0, human_gold=False,
        new_model_training_private_PnP_calls=0)
    path = out / 'EVIDENCE_RENDER_COMPLETE.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(bind(path)))


if __name__ == '__main__':
    main()
