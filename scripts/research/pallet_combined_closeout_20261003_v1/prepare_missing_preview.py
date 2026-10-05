"""Render five outstanding visibility questions with PnP position hints only.

The source annotations are read-only. Hollow markers never constitute a label or
a manually clicked reference coordinate. All generated assets stay in images/.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
REVIEW = ROOT / 'data/pallet/results/pallet_lifter_case_review_20261003_v1/review'
OUTPUT = ROOT / ('_docs/experiments/pallet_combined_closeout_20261003_v1/'
                 'human_review_20261004_v1/completion_20261005_v1/images')
MISSING = [('174126:13', 3), ('174126:419', 5), ('174342:1190', 5),
           ('174925:32', 5), ('174925:1002', 5)]
FONT = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
YELLOW = '#ffe269'
LANCZOS = getattr(Image, 'Resampling', Image).LANCZOS


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def font(size):
    return ImageFont.truetype(str(FONT), size)


def crop_box(points, image_size):
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    dx, dy = max(xs) - min(xs), max(ys) - min(ys)
    pad_x, pad_y = max(26., .16 * dx), max(30., .65 * dy)
    width, height = image_size
    return (max(0, math.floor(min(xs) - pad_x)),
            max(0, math.floor(min(ys) - pad_y)),
            min(width, math.ceil(max(xs) + pad_x)),
            min(height, math.ceil(max(ys) + pad_y)))


def render_preview(raw, box, xy, fid, index):
    panel = Image.new('RGB', (840, 440), '#172027')
    draw = ImageDraw.Draw(panel)
    draw.text((24, 13), f'{fid}   확인할 점 {index} · 위치 참고(PnP)',
              font=font(23), fill='#eef2f4')
    crop = raw.crop(box)
    scale = min(792 / crop.width, 336 / crop.height)
    display_size = (round(crop.width * scale), round(crop.height * scale))
    display = crop.resize(display_size, LANCZOS)
    offset = ((panel.width - display.width) // 2,
              68 + (336 - display.height) // 2)
    panel.paste(display, offset)
    # Per-axis display scaling accounts for rounding; original coordinates stay intact.
    marker_x = offset[0] + (xy[0] - box[0]) * display.width / crop.width
    marker_y = offset[1] + (xy[1] - box[1]) * display.height / crop.height
    radius = 16
    draw.ellipse((marker_x-radius, marker_y-radius, marker_x+radius, marker_y+radius),
                 outline=YELLOW, width=3)
    question = f'{index}?'
    text_font = font(28)
    text_box = draw.textbbox((0, 0), question, font=text_font)
    label_w = text_box[2] - text_box[0] + 14
    label_h = text_box[3] - text_box[1] + 12
    tx = min(panel.width-label_w-18, max(18, marker_x+radius+7))
    ty = max(64, marker_y-radius-label_h-5)
    draw.rounded_rectangle((tx-5, ty-2, tx+label_w, ty+label_h), radius=4,
                           fill='#172027', outline=YELLOW, width=1)
    draw.text((tx+2, ty-text_box[1]+4), question, font=text_font, fill=YELLOW)
    draw.text((24, 408), '원 안의 점이 실제로 보이는지 확인해 주세요.',
              font=font(18), fill='#c4d0d7')
    return panel, dict(display_size=list(display_size), display_offset=list(offset),
                       display_scale_xy=[display.width/crop.width, display.height/crop.height])


def prepare(output=OUTPUT):
    output = Path(output).resolve()
    manifest_path = REVIEW / 'MANIFEST.json'
    recovery_path = REVIEW / 'VIEWER_DRAFT_RECOVERY.json'
    manifest, recovery = read(manifest_path), read(recovery_path)
    frames = {f['frame_id']: f for f in manifest['frames']}
    inputs = {str(manifest_path): sha256(manifest_path),
              str(recovery_path): sha256(recovery_path)}
    panels, entries = [], []
    # Validate every input before producing any new files.
    for fid, index in MISSING:
        frame = frames[fid]
        draft = recovery['drafts'][fid+'|primary']['record']
        actual_missing = [c['id'] for c in draft['corners'] if c['visibility'] is None]
        if actual_missing != [index]:
            raise ValueError(f'{fid}: current missing corners changed: {actual_missing}')
        session, frame_index = fid.split(':')
        source_path = REVIEW / 'native_annotations/primary' / (
            f'{session}_{int(frame_index):05d}.PNP_ASSISTED.json')
        pnp = read(source_path)
        if (pnp['frame_id'] != fid or pnp['review_pass'] != 'primary'
                or pnp['bindings'] != recovery['bindings']
                or pnp['image_sha256'] != frame['image_sha256']):
            raise ValueError(f'{fid}: PnP source bindings differ')
        point = pnp['editor_keypoint_annotations'][index]
        manual = pnp['manual_reference_corners'][index]
        xy = pnp['editor_kps_2d'][index]
        if (point['source'] != 'pnp_projected' or point['xy'] != xy
                or manual['visibility'] is not None or manual['x'] is not None
                or manual['y'] is not None):
            raise ValueError(f'{fid}: hint must be projected, with no actual human label')
        image_path = REVIEW / frame['image_path']
        if sha256(image_path) != frame['image_sha256']:
            raise ValueError(f'{fid}: original image hash mismatch')
        inputs[str(source_path)] = sha256(source_path)
        inputs[str(image_path)] = sha256(image_path)
        raw = Image.open(image_path).convert('RGB')
        points = pnp['editor_kps_2d'][:8]
        if not all(isinstance(p, list) and len(p) == 2 and
                   all(isinstance(v, (int, float)) and math.isfinite(v) for v in p)
                   for p in points):
            raise ValueError(f'{fid}: incomplete PnP position box')
        box = crop_box(points, raw.size)
        panel, display = render_preview(raw, box, xy, fid, index)
        filename = f'{fid.replace(":", "_")}_corner_{index}_question.png'
        panels.append((filename, panel))
        entries.append(dict(frame_id=fid, review_pass='primary', missing_corner_id=index,
                            source_image=str(image_path), image_sha256=frame['image_sha256'],
                            source_pnp=str(source_path), source_pnp_sha256=inputs[str(source_path)],
                            position_hint_source='pnp_projected', position_hint_xy=xy,
                            crop_box_original_pixels=list(box), display=display,
                            preview_filename=filename, labels_inferred=False,
                            coordinate_is_human_reference=False))
    output.mkdir(parents=True, exist_ok=True)
    for filename, panel in panels:
        target = output / filename
        panel.save(target)
        next(e for e in entries if e['preview_filename'] == filename)['preview_sha256'] = sha256(target)
    sheet = Image.new('RGB', (1248, 1074), '#111a20')
    draw = ImageDraw.Draw(sheet)
    draw.text((25, 17), '남은 5개 점 확인', font=font(30), fill='#eef2f4')
    draw.text((25, 65), '노란 원: 확인할 위치 참고(PnP)', font=font(22), fill=YELLOW)
    for i, (_, panel) in enumerate(panels):
        thumb = panel.resize((600, 315), LANCZOS)
        sheet.paste(thumb, (16+(i % 2)*616, 112+(i // 2)*320))
    sheet_path = output / 'missing_corner_contact_sheet.png'
    sheet.save(sheet_path)
    for path, expected in inputs.items():
        if sha256(Path(path)) != expected:
            raise RuntimeError(f'Source changed while preview was being prepared: {path}')
    result = dict(schema='lifter_missing_corner_preview_v1',
                  source_kind='machine_assistance_not_annotation',
                  created_at=datetime.now(timezone.utc).isoformat(),
                  bindings=recovery['bindings'], labels_inferred=False,
                  raw_annotation_files_modified=False, human_actions_synthesized=False,
                  evaluated_model_predictions_used=False, evaluation_use=False,
                  independent_reference=False, input_sha256=inputs, records=entries,
                  contact_sheet=str(sheet_path), contact_sheet_sha256=sha256(sheet_path))
    (output/'preview_manifest.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = prepare(args.output)
    print(json.dumps(dict(contact_sheet=result['contact_sheet'], preview_count=len(result['records']),
                          manifest=str(args.output/'preview_manifest.json')), ensure_ascii=False))
