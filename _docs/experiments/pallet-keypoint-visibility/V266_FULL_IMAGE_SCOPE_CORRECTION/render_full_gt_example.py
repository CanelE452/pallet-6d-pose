"""Demonstrate existing uncropped SYN raster GT; no model prediction or fit."""
from pathlib import Path
import hashlib
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).absolute().parent
ROWS = HERE.parent / 'V266_SEGMENTATION_CONTRASTIVE/data_v1/ROW_MANIFEST.jsonl'
BUFFERS = HERE.parent / 'V266_SEGMENTATION_CONTRASTIVE/data_v1/SOURCE_BUFFER_BINDINGS.json'


def safe(value):
    p = Path(value)
    if not p.is_absolute() or any(s in str(p).lower() for s in ('v112', 'cache', '.pyc', '..')):
        raise ValueError(str(p))
    for a in reversed((p, *p.parents)):
        if a.is_symlink():
            raise ValueError(str(a))
    return p


def binding(path):
    p = safe(path)
    raw = p.read_bytes()
    return {'path': str(p), 'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}


def main():
    image_out = safe(HERE / 'FULL_FRAME_GT_EXAMPLE.png')
    source_out = safe(HERE / 'SOURCE.json')
    assert not image_out.exists() and not source_out.exists()
    rows = [json.loads(line) for line in safe(ROWS).read_text().splitlines()]
    bound = {b['path']: b for b in json.loads(safe(BUFFERS).read_text())['buffers']}
    chosen = None
    # Select the lowest source base with both BEFORE/COVER and nonzero FOREIGN
    # in COVER; inspect ownership only, not RGB or any model result.
    for base in sorted({r['source_base_index'] for r in rows}):
        pair = {}
        for variant in ('BEFORE', 'COVER'):
            candidates = [r for r in rows if r['source_base_index'] == base and r['variant'] == variant]
            if candidates:
                pair[variant] = min(candidates, key=lambda r: r['source_index'])
        if len(pair) != 2:
            continue
        for r in pair.values():
            b = binding(r['image']['path'])
            assert b == bound[r['image']['path']] and b['sha256'] == r['image']['sha256']
        with np.load(safe(pair['COVER']['image']['path']), allow_pickle=False) as z:
            foreign_pixels = int((z['part_id'] == 999).sum())
        if foreign_pixels:
            chosen = pair
            break
    assert chosen is not None
    fontpath = safe(Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'))
    titlefont = ImageFont.truetype(str(fontpath), 24)
    font = ImageFont.truetype(str(fontpath), 18)
    canvas = Image.new('RGB', (1096, 1270), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 12), '실제로 저장되어 있는 합성 전체 이미지와 픽셀 분할 정답', font=titlefont, fill='#16324a')
    draw.text((20, 50), '512×512 원본 전체 프레임 · crop 아님 · 모델 예측 아님 · renderer 소유 ID 정답', font=font, fill='#16324a')
    draw.text((20, 80), '초록=대상 팔레트 · 주황=외부 물체 · 파랑=배경 · 검정=모호함/동률', font=font, fill='#16324a')
    cases = []
    palette = np.asarray([[40, 185, 85], [240, 100, 30], [85, 110, 185]], np.uint8)
    for panel, variant in enumerate(('BEFORE', 'COVER')):
        r = chosen[variant]
        with np.load(safe(r['image']['path']), allow_pickle=False) as z:
            rgb = np.array(z['RGB_uint8'], copy=True)
            owner = np.array(z['part_id'], copy=True)
            amb = np.array(z['ambiguity'], copy=True).astype(bool)
            ties = np.array(z['tie_count'], copy=True)
        assert rgb.shape == (512, 512, 3) and owner.shape == amb.shape == ties.shape == (512, 512)
        classes = np.where(owner < 0, 2, np.where(owner == 999, 1, 0)).astype(np.uint8)
        gt = palette[classes]
        unknown = amb | (ties >= 2)
        gt[unknown] = 0
        y = 154 + panel * 550
        draw.text((20, y-34), f'{variant} · base{r["source_base_index"]:03d} 원본 RGB', font=font, fill='#16324a')
        draw.text((564, y-34), f'{variant} · 정확한 전체 renderer 분할 정답', font=font, fill='#16324a')
        canvas.paste(Image.fromarray(rgb), (20, y))
        canvas.paste(Image.fromarray(gt), (564, y))
        cases.append({'variant': variant, 'source_base_index': r['source_base_index'],
            'source': bound[r['image']['path']], 'original_shape': [512, 512],
            'RGB_uint8_HWC_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
            'part_id_array_sha256': hashlib.sha256(np.ascontiguousarray(owner).tobytes()).hexdigest(),
            'class_pixel_counts_TARGET_FOREIGN_BACKGROUND': [int(((classes == c) & ~unknown).sum()) for c in range(3)],
            'unknown_pixels': int(unknown.sum()), 'source_native_row_not_used_to_crop': r['source_index']})
    draw.text((20, 1220), 'V266 학습은 이 전체 정답을 KP crop으로 잘라 사용했습니다. 전체 이미지 CLIP/SCLIP 분할 실험은 아닙니다.', font=font, fill='#9d3516')
    canvas.save(image_out)
    source = {'completed': True, 'PASS': True,
        'purpose': 'Show existing full512 SYN ownership supervision, clarify V266 crop scope; no new experiment.',
        'selection': 'Lowest source_base_index with BEFORE and COVER, nonzero storedFOREIGN in COVER; selected by metadata/owner only before RGB decoding.',
        'selected_before_RGB_decode': True, 'row_manifest': binding(ROWS), 'buffer_manifest': binding(BUFFERS),
        'cases': cases, 'label_codebook': {'TARGET0': 'part_id0..18', 'FOREIGN1': 'part_id999', 'BACKGROUND2': 'part_id<0', 'UNKNOWN': 'ambiguity or tie>=2'},
        'GT_provenance': 'Stored synthetic renderer first-hit raster part_id; not manual real annotation, not model segmentation.',
        'interpretation_limits': ['Foreign surface is not automatically intended native keypoint E.', 'This source contains one designated target pallet plus synthetic foreign geometry; it is not a semantic cone category annotation.', 'Real223 has keypoint visibility annotation, not real pixel segmentation GT.', 'V266 used frozen DINOv2 KP crop tokens, not whole-image CLIP/SCLIP features.'],
        'code': binding(Path(__file__).absolute()), 'output': binding(image_out),
        'model_forwards': 0, 'optimizer_updates': 0, 'CLIP_or_SAM_downloads': 0, 'Git_changes': 0}
    source_out.write_text(json.dumps(source, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'completed': True, 'image': binding(image_out), 'source': binding(source_out)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
