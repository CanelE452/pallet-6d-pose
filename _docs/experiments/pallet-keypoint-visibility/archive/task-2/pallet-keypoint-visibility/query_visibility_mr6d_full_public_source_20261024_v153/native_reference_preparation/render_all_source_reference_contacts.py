"""Render all sealed public source frames, no model/query/semantic input."""
from pathlib import Path
from datetime import datetime, timezone
import ast, hashlib, json, math, html
import numpy as np
from PIL import Image, ImageDraw, ImageFont, __version__ as PIL_VERSION

H = Path(__file__).resolve().parent
V = H.parent
P = V.parent
F = V / 'full_source_v1'
R = H / 'provided_reference_v1'
O = H / 'source_reference_contact_v1'
OLD = P / 'query_visibility_mr6d_actual_query_bridge_20261024_v152/native_reference_audit/project_original_native_references.py'
POLICY = '46b3b077edfc78d113dcd0b38771bb3df96053a3741002ac774bc1c9866e2334'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p): return {'path': str(p), 'sha256': sha(p)}
def read(p): return json.loads(Path(p).read_text())
def write(p, x):
    with Path(p).open('x') as f:
        json.dump(x, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def get_draw():
    nodes = [n for n in ast.parse(OLD.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in {'dashed', 'draw_instance'}]
    assert len(nodes) == 2
    ns = {'np': np, 'math': math, 'ImageDraw': ImageDraw, 'ImageFont': ImageFont}
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(OLD), 'exec'), ns)
    return ns['draw_instance'], {'source': bind(OLD), 'selected_AST_sha256': hashlib.sha256(ast.dump(module, include_attributes=False).encode()).hexdigest(), 'display_math_unchanged': True}

def main():
    assert sha(V / 'ROOT_POLICY_BEFORE_NEW_SOURCE_BODIES.json') == POLICY
    dp = F / 'FULL_SOURCE_DOWNLOAD_COMPLETE.json'
    download = read(dp)
    assert download['completed'] and download['PASS']
    source_selection_path = F / 'FULL_SOURCE_SELECTION_BEFORE_NEW_PAYLOADS.json'
    source = read(source_selection_path)['frames']
    projection_path = R / 'REFERENCE_PROJECTION_COMPLETE.json'
    projection = read(projection_path)
    assert projection['completed'] and projection['PASS']
    geometry_path = Path(projection['frames_geometry']['path'])
    assert sha(geometry_path) == projection['frames_geometry']['sha256']
    frames = read(geometry_path)['frames']
    assert len(source) == len(frames) == 230
    keys = ['frame_index', 'id', 'source', 'source_id', 'split', 'official_split', 'scene', 'frame_key', 'obj1_gt_instance_indices']
    for a, b in zip(source, frames):
        assert all(a[k] == b[k] for k in keys)
        assert a['source_files']['RGB'] == b['source_files']['RGB']
    files = {x['relative_path']: x for x in download['files']}
    assert len(files) == 1067
    draw_instance, drawproof = get_draw()
    O.mkdir(exist_ok=False)
    full = O / 'frames'
    full.mkdir()
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 19)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    rendered = []
    for s in frames:
        fi = s['frame_index']
        relative = s['source_files']['RGB']
        f = files[relative]
        assert f['completed'] and f['source_content_ID_verified']
        rgb_path = Path(f['local_path'])
        assert rgb_path == F / 'original' / relative
        assert sha(rgb_path) == f['sha256']
        with Image.open(rgb_path) as src:
            raw = src.convert('RGB')
        assert [raw.height, raw.width] == s['raw_hw'] == [720, 1280]
        marked = draw_instance(raw, s['instances'])
        name = f"FRAME_{fi:03d}_{s['scene']}_{int(s['frame_key']):06d}"
        marked_path = full / (name + '_ALL_INSTANCES_NATIVE.png')
        marked.save(marked_path, compress_level=1)
        rendered.append({k: s[k] for k in keys} | {
            'raw_RGB': {'path': str(rgb_path), 'sha256': f['sha256']},
            'decoded_RGB_sha256': hashlib.sha256(np.asarray(raw).tobytes()).hexdigest(),
            'decoder': 'PIL' + PIL_VERSION + ' diagnostic display only; not R0 canonical transport',
            'raw_hw_verified': [720, 1280], 'all_instance_overlay': bind(marked_path),
            'raw_original_link_retained_without_duplicate_PNG': True})
    pages = []
    # Validation appears first for review. Original frame indices and rows never move.
    for split in ['SOURCE_VALIDATION', 'SOURCE_DEVELOPMENT']:
        items = [x for x in rendered if x['split'] == split]
        for start in range(0, len(items), 8):
            batch = items[start:start + 8]
            sheet = Image.new('RGB', (1280, 1688), '#edf2f6')
            dr = ImageDraw.Draw(sheet)
            dr.text((16, 5), f'{split} page {start // 8 + 1}: provided references only / all GT pallet instances retained', font=font, fill='#17324b')
            dr.text((16, 31), 'Original RGB files unchanged. NEAREST thumbnail is diagnostic; use full-resolution links for structural adjudication.', font=small, fill='#526477')
            for j, x in enumerate(batch):
                col, row = j % 2, j // 2
                xx, yy = col * 640, 74 + row * 402
                dr.text((xx + 8, yy), f"{x['frame_index']:03d} {x['source_id']} | GT {x['obj1_gt_instance_indices']}", font=small, fill='#17324b')
                with Image.open(x['all_instance_overlay']['path']) as im:
                    thumb = im.convert('RGB').resize((640, 360), Image.Resampling.NEAREST)
                sheet.paste(thumb, (xx, yy + 28))
            page_path = O / f'{split}_CONTACT_{start // 8 + 1:02d}.png'
            sheet.save(page_path, compress_level=1)
            pages.append({'split': split, 'frame_indices': [x['frame_index'] for x in batch], 'contact': bind(page_path)})
    write(O / 'ALL230_SOURCE_REFERENCE_CONTACT_INDEX.json', {'completed': True, 'frames': rendered, 'pages': pages})
    lines = ['<!doctype html><meta charset="utf-8"><title>MR6D source native references</title>',
             '<style>body{font:17px sans-serif;max-width:1500px;margin:24px auto;color:#17324b}img{width:100%;height:auto}section{margin:28px 0}small{color:#526477}</style>',
             '<h1>MR6D 공개 실사 전체 native reference 검토</h1>',
             '<p>230장·303개 관측·2,424개 슬롯 전체를 보존했습니다. 제공 pose/K로 투영한 명목 reference이며 정밀 픽셀 정답이 아닙니다. 원본 RGB는 변경하지 않았고, D/E/S 라벨을 자동 생성하지 않았습니다.</p>',
             '<p>별도 source validation 45장을 먼저 표시합니다. 모델·query·가시성 점수는 이 자료에 사용하지 않았습니다.</p>']
    for split in ['SOURCE_VALIDATION', 'SOURCE_DEVELOPMENT']:
        lines.append('<h2>' + split + '</h2>')
        for x in [x for x in rendered if x['split'] == split]:
            pp = x['all_instance_overlay']['path']
            raw_path = x['raw_RGB']['path']
            lines.append(f'<section><b>{x["frame_index"]:03d} {html.escape(x["source_id"])}</b> — GT {x["obj1_gt_instance_indices"]} · <a href="{html.escape(raw_path)}">원본 JPEG</a><br><a href="{html.escape(pp)}"><img loading="lazy" src="{html.escape(pp)}"></a></section>')
    with (O / 'INDEX.html').open('x') as f:
        f.write('\n'.join(lines))
    write(O / 'CONTACT_RENDER_COMPLETE.json', {
        'completed': True, 'PASS': True, 'UTC': datetime.now(timezone.utc).isoformat(),
        'code': bind(Path(__file__)), 'root_policy': bind(V / 'ROOT_POLICY_BEFORE_NEW_SOURCE_BODIES.json'),
        'download': bind(dp), 'source_selection': bind(source_selection_path),
        'reference_projection': bind(projection_path), 'draw_helpers': drawproof,
        'index': bind(O / 'ALL230_SOURCE_REFERENCE_CONTACT_INDEX.json'), 'HTML': bind(O / 'INDEX.html'),
        'counts': {'frames': 230, 'source_validation_frames': 45, 'source_development_frames': 185,
                   'instances': 303, 'native_references': 2424, 'all_original_RGB_shapes_verified': 230},
        'pages': pages, 'source_all_raw_RGB_sha_verified': True, 'original_RGB_or_pose_changed': False,
        'presentation_storage': '230 full native overlays + 29 contact pages; original JPEG links; no duplicate full raw/paired/single-instance raster copies',
        'semantics_input_or_new_labels': False, 'predictions_or_queries_read': False,
        'model_fit_PnP_private_download': 0, 'actual_visual_review_pending': True})
    print(json.dumps({'seal': bind(O / 'CONTACT_RENDER_COMPLETE.json'), 'pages': len(pages)}))

if __name__ == '__main__': main()
