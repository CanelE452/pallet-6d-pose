"""Six metadata-locked public LOCO TRAIN takes; no model, labels or query creation."""
from pathlib import Path
import argparse, collections, datetime, hashlib, io, json, zipfile

P = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
HERE = Path(__file__).resolve().parent
V73 = P / 'query_visibility_public_pallet_loco_20261023_v73/data/intake'
V74 = P / 'query_visibility_loco_native_context_20261023_v74'
SOURCE = V74 / 'source_groups/INPUT_ONLY_SOURCE_GROUP_MANIFEST.jsonl'
FINDINGS = P / 'query_visibility_training_duration_control_20261023_v124/public_intake/FINDINGS.json'

def bind(path):
    path = Path(path)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]

def write_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def select():
    # Only the authorized official TRAIN subset is parsed; other subset records are skipped.
    source_rows = []
    for line in SOURCE.open():
        if '"subset": 2,' in line:
            r = json.loads(line)
            assert r['subset'] == 2 and r['official_split'] == 'TRAIN'
            source_rows.append(r)
    assert len(source_rows) == 1109
    # Exclusion uses original image metadata only; no previous labels, scores or payloads.
    f20 = V73 / 'FIXED_20_RGB_MANIFEST.jsonl'
    e40 = V74 / 'public_reference_review/EXTENSION_40/EXTENSION_40_RGB_REVIEW_MANIFEST.jsonl'
    prior = read_jsonl(f20) + read_jsonl(e40)
    used = {r['archive_member'] for r in prior if r['source_subset'] == 2}
    assert len(used) == 12
    groups = collections.defaultdict(list)
    for r in source_rows:
        groups[r['path_capture_date_take']].append(r)
    assert sorted(groups) == ['2019-11-25_0', '2019-11-25_1', '2019-11-25_2',
                              '2019-11-25_3', '2019-11-25_5', '2019-11-25_6']
    selected = []
    for index, (take, rows) in enumerate(sorted(groups.items())):
        streams = collections.defaultdict(list)
        for r in rows:
            streams[r['path_stream_key']].append(r)
        stream = min(streams, key=lambda k: (-len(streams[k]), k))
        seq = streams[stream]
        kinds = {r['timestamp_kind'] for r in seq}
        assert len(kinds) == 1
        lo = min(r['timestamp_filename_number'] for r in seq)
        hi = max(r['timestamp_filename_number'] for r in seq)
        midpoint = (lo + hi) / 2
        unused = [r for r in seq if r['archive_member'] not in used]
        assert unused
        r = min(unused, key=lambda x: (abs(x['timestamp_filename_number'] - midpoint), x['archive_member']))
        selected.append({'selection_index': index, 'take': take, 'take_count': len(rows),
                         'stream_counts': {k: len(v) for k, v in sorted(streams.items())},
                         'selected_stream': stream, 'stream_count': len(seq),
                         'stream_timestamp_min': lo, 'stream_timestamp_max': hi,
                         'stream_timestamp_midpoint': midpoint,
                         'timestamp_kind': next(iter(kinds)),
                         'normalized_temporal_position': (r['timestamp_filename_number'] - lo) / (hi - lo),
                         'excluded_existing_fixed20_extension40_count_in_stream': len(seq) - len(unused),
                         'original_row': r})
    assert len({r['original_row']['archive_member'] for r in selected}) == 6
    lock = {'completed_before_selected_RGB_access': True,
            'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'selection_rule': 'Each of six takes once. Choose largest metadata-only stream (tie stream path), then unused original closest to that stream timestamp min/max midpoint (tie original path). Never compare incompatible camera clocks.',
            'source': bind(SOURCE), 'public_intake_findings': bind(FINDINGS),
            'code': bind(__file__), 'prior_fixed20': bind(f20), 'prior_extension40': bind(e40),
            'exclusion_rule': 'All12 subset2 originals in fixed20+extension40; no score/label/image screening.',
            'archive': {'path': str(V73 / 'dataset.zip'), 'bytes': 769055500,
                        'sha256': 'f3629d989071b824edc63e000f45ccc5826cfbb201a338f16ed87ab3e10bcdcf',
                        'prior_whole_archive_audit': bind(V73 / 'ARCHIVE_CRC_AND_SHA_AUDIT.json')},
            'rows': selected, 'all_original_fold': 'TRAIN', 'license': 'CC0-1.0',
            'no_pixels_models_scores_current_V112_VAL_payloads_read': True,
            'new_labels_or_queries': False,
            'limits': ['All six takes are one official environment and one date, not six independent sites.',
                       'Sensor-clock units are unverified; midpoint is used only within a fixed stream.',
                       'Selection is new relative to fixed20+extension40, not never decoded: V74 inventoried all5593 original images.']}
    write_new(HERE / 'SELECTION_LOCK_BEFORE_PIXELS.json', lock)
    print(json.dumps({'selection_lock': bind(HERE / 'SELECTION_LOCK_BEFORE_PIXELS.json'),
                      'selected': [r['original_row']['archive_member'] for r in selected]}), flush=True)

def render():
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    lock_path = HERE / 'SELECTION_LOCK_BEFORE_PIXELS.json'
    lock = json.loads(lock_path.read_text())
    assert lock['code'] == bind(__file__) and lock['source'] == bind(SOURCE)
    original_dir = HERE / 'RGB_ORIGINAL'
    original_dir.mkdir()
    fontpath = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    title = ImageFont.truetype(fontpath, 29)
    normal = ImageFont.truetype(fontpath, 21)
    small = ImageFont.truetype(fontpath, 17)
    W, H = 1120, 755
    contact = Image.new('RGB', (W * 2, H * 3), '#e7edf3')
    receipts = []
    with zipfile.ZipFile(lock['archive']['path']) as z:
        for selected in lock['rows']:
            i = selected['selection_index']; r = selected['original_row']
            info = z.getinfo(r['archive_member'])
            raw = z.read(info)  # ZipFile verifies stored CRC as well as subsequent SHA checks.
            assert hashlib.sha256(raw).hexdigest() == r['jpeg_file_SHA256']
            a = np.array(Image.open(io.BytesIO(raw)).convert('RGB'), dtype=np.uint8, copy=True)
            assert list(a.shape[:2]) == r['raw_hw']
            assert hashlib.sha256(a.tobytes()).hexdigest() == r['decoded_RGB_SHA256']
            f = original_dir / f'{i:02d}_{selected["take"]}.jpg'
            with f.open('xb') as out: out.write(raw)
            panel = Image.new('RGB', (W, H), '#f8fafc'); d = ImageDraw.Draw(panel)
            d.text((18, 12), f'{i + 1}/6  {selected["take"]}  |  원 TRAIN 유지', font=title, fill='#123753')
            d.text((18, 51), selected['selected_stream'].replace('dataset/subset-2/', ''), font=normal, fill='#294d61')
            im = Image.fromarray(a); im.thumbnail((1080, 607), Image.Resampling.LANCZOS)
            panel.paste(im, (20 + (1080 - im.width) // 2, 86 + (607 - im.height) // 2))
            d.text((18, 704), f'무표시 원본 축소표시 | native {a.shape[1]}×{a.shape[0]} | 시간 위치 {selected["normalized_temporal_position"]:.3f}', font=normal, fill='#294d61')
            d.text((18, 735), '구조 cue 표시 가능성 검토만 · 새 D/E 라벨/native 대응/모델 추론 없음', font=small, fill='#7b5521')
            card = HERE / f'CASE_{i:02d}.png'; panel.save(card)
            contact.paste(panel, ((i % 2) * W, (i // 2) * H))
            receipts.append({'selection_index': i, 'take': selected['take'],
                             'original_member': r['archive_member'], 'original_JPEG': bind(f),
                             'native_hw': list(a.shape[:2]), 'decoded_RGB_sha256': r['decoded_RGB_SHA256'],
                             'ZIP_CRC32_verified': True, 'JPEG_and_decoded_SHA_exact_V74': True,
                             'card': bind(card)})
    sheet = HERE / 'CONTACT_SIX_TAKES.png'; contact.save(sheet)
    write_new(HERE / 'RENDER_RECEIPTS.json',
              {'completed': True, 'PASS': True, 'selection_lock': bind(lock_path), 'code': bind(__file__),
               'originals': receipts, 'contact': bind(sheet), 'all6_originals_decoded': True,
               'old_files_modified': False, 'model_current_V112_VAL_payload_fit_label_calls': 0})
    print(json.dumps({'completed': True, 'receipt': bind(HERE / 'RENDER_RECEIPTS.json'),
                      'contact': bind(sheet)}), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['select', 'render'], required=True)
    args = parser.parse_args()
    (select if args.stage == 'select' else render)()
