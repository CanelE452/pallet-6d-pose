"""Caption-only repair of generated scientific cards; no original RGB or scores."""
from pathlib import Path
import json, hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

V = Path(__file__).resolve().parents[1]
SRC = V / 'fixed_current_crop_review_v1'
OUT = SRC / 'caption_aligned_v2'

def binding(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

receipt = SRC / 'CROP_RENDER_COMPLETE.json'
assert binding(receipt)['sha256'] == '9f47236d47af91e3667cd18589106deed2e324072cf29291b96c6f1286b219cd'
source = json.loads(receipt.read_text())
assert source['PASS'] and source['selected_cases'] == 19 and not OUT.exists()
OUT.mkdir()
font = ImageFont.truetype('/usr/share/fonts/truetype/nanum/NanumGothic.ttf', 25)
rows = []
for row in source['panels']:
    path = Path(row['card']['path'])
    assert binding(path) == row['card']
    old = np.array(Image.open(path).convert('RGB'))
    assert old.shape == (1400, 2080, 3)
    board = Image.fromarray(old.copy())
    draw = ImageDraw.Draw(board)
    draw.rectangle((0, 840, 2079, 897), fill='#f5f8fb')
    for x, label in [(24, '조건'), (465, '예측'), (590, '원래 raw visibility logit'),
                     (1025, '고정 threshold64'), (1510, 'decision_score64')]:
        draw.text((x, 850), label, font=font, fill='#20364c')
    new = np.array(board)
    assert np.array_equal(old[:840], new[:840])
    assert np.array_equal(old[898:], new[898:])
    out = OUT / path.name
    board.save(out)
    assert np.array_equal(np.array(Image.open(out)), new)
    rows.append({'global_index': row['global_index'], 'original': row['card'],
                 'caption_aligned': binding(out), 'all_nonheader_pixels_exact': True})

index = OUT / 'index.html'
body = (SRC / 'index.html').read_text()
body = body.replace('<h1>4 고정조건 × WOOD/SQUARE strata</h1>',
    '<h1>4 고정조건 × WOOD/SQUARE strata</h1><p>표의 제목 위치만 바로잡은 v2입니다. '
    '원래 선택·수치·원본/실제 crop 픽셀은 모두 같습니다. '
    '<a href="../index.html">보존한 원래 v1</a></p>')
index.write_text(body)
result = {'completed': True, 'PASS': True, 'code': binding(Path(__file__)),
          'source_render_receipt': binding(receipt), 'cards': rows, 'index': binding(index),
          'patch_box_xyxy_halfopen': [0, 840, 2080, 898],
          'original_card_count': 19, 'same_selection_and_values': True,
          'all_nonheader_pixels_byte_exact': True,
          'original_RGB_cache_features_labels_scores_models_read': 0,
          'new_forward_fit_threshold_choice_or_PnP': 0,
          'original_v1_preserved': True, 'browser_rendered': False,
          'actual_v2_visual_QA_pending': True}
out = OUT / 'CAPTION_ALIGNMENT_COMPLETE.json'
out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(binding(out)))
