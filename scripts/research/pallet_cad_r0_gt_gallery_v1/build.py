"""Display all eval_cad frames: existing GT and unchanged cached raw R0 output.

No inference, correction, channel permutation, filtering, training or split edits.
"""
from pathlib import Path
import hashlib
import html
import json
import os

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs/pallet_cad_r0_gt_gallery_v1'
PROTOCOL = ROOT / '_docs/experiments/pallet_type_selftrain_v1/EVAL_PROTOCOL.json'
PRED = ROOT / 'data/pallet/results/pallet_type_selftrain_v1/EVAL_PREDICTIONS_R0.json'
SCORES = ROOT / 'data/pallet/results/pallet_type_selftrain_v1/paper_metrics_plastic/F0_R0.json'
EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
GREEN = (70, 245, 85)  # OpenCV BGR
BLUE = (255, 170, 35)


def read(path):
    return json.loads(path.read_text())


def binding(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def verify(record):
    assert binding(ROOT / record['path'])['sha256'] == record['sha256'], record['path']


def write(path, value):
    content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    if path.exists():
        assert path.read_text() == content, f'Refusing to overwrite different artifact: {path}'
    else:
        with path.open('x') as stream:
            stream.write(content)


def image_write(path, pixels):
    if path.exists():
        assert np.array_equal(cv2.imread(str(path)), pixels), path
    else:
        assert cv2.imwrite(str(path), pixels), path


def text(canvas, label, point, color=(245,245,245), scale=.48):
    cv2.putText(canvas, label, point, cv2.FONT_HERSHEY_SIMPLEX, scale, (15,15,15), 3, cv2.LINE_AA)
    cv2.putText(canvas, label, point, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw(canvas, points, color, prefix=None, filled=False):
    xy = np.rint(points).astype(int)
    for a,b in EDGES:
        cv2.line(canvas, tuple(xy[a]), tuple(xy[b]), color, 1, cv2.LINE_AA)
    height,width = canvas.shape[:2]
    for j,point in enumerate(xy):
        # Do not move out-of-frame predictions/GT to the image boundary.
        if not (0 <= point[0] < width and 0 <= point[1] < height):
            continue
        if j == 8:
            cv2.drawMarker(canvas, tuple(point), color, cv2.MARKER_CROSS, 9, 1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, tuple(point), 3, color, -1 if filled else 1, cv2.LINE_AA)
        if prefix:
            label_at = (min(width-28, max(2, int(point[0])+5)), max(13,int(point[1])-6))
            text(canvas, prefix+str(j), label_at, color, .40)


def panel(rgb, gt, pred, number, median, maximum, overlay):
    canvas = rgb.copy()
    draw(canvas, gt, GREEN, None if overlay else 'G')
    if overlay:
        draw(canvas, pred, BLUE, 'P', True)
    header = np.full((56, rgb.shape[1], 3), (30,25,19), np.uint8)
    text(header, f'{number:02d}/18  '+('GT (green) + R0 raw (blue)' if overlay else 'GT / existing annotation'), (12,20))
    text(header, f'Keypoint med {median:.2f} px | max {maximum:.2f} px' if overlay else 'G0-G7: corners | G8: center', (12,43))
    return np.concatenate([header, canvas], axis=0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = [r for r in read(PROTOCOL)['records'] if r['session'] == 'eval_cad']
    payload = read(PRED)
    predictions = {r['id']:r for r in payload['records'] if r['id'].startswith('eval_cad:')}
    archived = {r['frame_id']:r for r in read(SCORES)['records']}
    assert len(records) == len(predictions) == 18
    assert {r['id'] for r in records} == set(predictions)
    verify(payload['checkpoint'])
    cards, receipt, previews, pooled = [], [], [], []
    for number,r in enumerate(records,1):
        verify(r['image']); verify(r['annotation'])
        rgb = cv2.imread(str(ROOT / r['image']['path']))
        assert rgb is not None and rgb.shape[:2] == (480,640)
        obj = read(ROOT / r['annotation']['path'])['objects'][0]
        annotations = obj['keypoint_annotations']
        gt = np.asarray([a['xy'] for a in annotations], float)
        mask = np.asarray([a['visibility'] > 0 and a.get('in_frame',True) for a in annotations])
        top = max(predictions[r['id']]['prediction']['candidates'], key=lambda p:p['score'])
        pred = np.asarray(top['keypoints_xy'], float)
        assert gt.shape == pred.shape == (9,2) and np.isfinite(gt).all() and np.isfinite(pred).all()
        error = np.linalg.norm(pred-gt, axis=1)
        assert np.allclose(error[mask], archived[r['id']]['errors_px'], atol=1e-9, rtol=0)
        pooled.extend(error[mask].tolist())
        med, maximum = float(np.median(error[mask])), float(error[mask].max())
        left = panel(rgb, gt, pred, number, med, maximum, False)
        right = panel(rgb, gt, pred, number, med, maximum, True)
        compare = OUT / f'{number:02d}_compare.png'
        overlay = OUT / f'{number:02d}_overlay.png'
        image_write(compare, np.concatenate([left,right], axis=1))
        image_write(overlay, right)
        previews.append(cv2.resize(right, (480,402), interpolation=cv2.INTER_AREA))
        truncated = bool(obj.get('truncation',{}).get('is_truncated'))
        details = ' · '.join(f'P{j}: {error[j]:.2f}px'+(' (평가 제외)' if not mask[j] else '') for j in range(9))
        original = html.escape(os.path.relpath(ROOT/r['image']['path'], OUT), quote=True)
        cards.append(f'''<section class="card" id="frame-{number}">
<h2>{number:02d} / 18 · {html.escape(r['id'])}</h2>
<p>점 오차 중앙값 <b>{med:.2f}px</b> · 최대 <b>{maximum:.2f}px</b> · 평가 대상 {int(mask.sum())}/9점{' · <span class="warn">잘림 있음</span>' if truncated else ''}</p>
<a href="{compare.name}" target="_blank"><img src="{compare.name}" width="1280" height="536" alt="{number}번 왼쪽 정답, 오른쪽 정답과 R0 추론 겹침"></a>
<p><a href="{overlay.name}" target="_blank">겹쳐 보기 확대</a> · <a href="{original}" target="_blank">표시 없는 원본</a></p>
<details><summary>각 점의 오차</summary><p>{details}</p></details></section>''')
        receipt.append(dict(number=number, id=r['id'], image=r['image'], annotation=r['annotation'],
            raw_prediction_xy=pred.tolist(), gt_xy=gt.tolist(), supervised=mask.tolist(), errors_px=error.tolist(),
            median_px=med, max_px=maximum, truncated=truncated, compare=binding(compare), overlay=binding(overlay)))
    for page in range(3):
        tiles = previews[page*6:page*6+6]
        image_write(OUT / f'preview_{page+1}.png', np.concatenate([np.concatenate(tiles[i:i+2],axis=1) for i in range(0,6,2)],axis=0))
    nav = ' '.join(f'<a href="#frame-{n}">{n:02d}</a>' for n in range(1,19))
    page = '''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>eval_cad 18장 — 정답 vs R0 원본 추론</title><style>
body{margin:0;background:#101820;color:#edf2f7;font:16px/1.65 system-ui,sans-serif}main{max-width:1320px;margin:auto;padding:20px}
h1{font-size:26px}h2{font-size:19px;overflow-wrap:anywhere}a{color:#8fcbff}img{display:block;width:100%;height:auto;border-radius:5px}
header{background:#192630;padding:18px;border-radius:8px}.card{padding:25px 0;border-bottom:1px solid #415260}p{margin:9px 0}
.legend{position:sticky;top:0;background:#101820f5;padding:9px 18px;z-index:1;border-bottom:1px solid #415260;text-align:center}
.gt{color:#55f546}.pred{color:#23aaff}.warn{color:#ffc570}nav a{display:inline-block;padding:5px 10px}details{color:#bdcbd7}
</style></head><body><div class="legend"><span class="gt">초록 G = 기존 정답</span> · <span class="pred">파랑 P = 보정 없는 R0 추론</span> · 왼쪽: 정답 / 오른쪽: 겹쳐 보기</div><main>
<header><h1>eval_cad · 전체 18장</h1><p>합성학습 R0의 저장된 원본 출력을 표시했습니다. Self-training 학생 또는 보정기 출력이 아닙니다.</p>
<p>GT는 표시·오차 계산에만 사용했습니다. 점 이동·대칭 재배열·이미지 선별·학습·데이터 분할 변경은 하지 않았습니다.</p>
<p>0–7은 코너, 8은 중심점입니다. 선은 기존 라벨의 cuboid 연결입니다. 화면 밖 좌표는 경계로 옮기지 않았으며 잘려 보입니다.
숫자는 원본 640×480 좌표계에서 평가 가능한 점의 오차입니다. 가려진 점의 GT도 기존 라벨을 그대로 표시합니다.</p>
<p>전체 점 오차 중앙값 11.66px · P90 20.31px · 잘림 포함 4장도 빠짐없이 표시.</p><nav>'''+nav+'</nav></header>'+''.join(cards)+'</main></body></html>'
    write(OUT/'index.html', page)
    write(OUT/'GALLERY_RECEIPT.json', dict(session='eval_cad', frames=18, all_frames_included=True,
        checkpoint=payload['checkpoint'], sources=[binding(p) for p in [PROTOCOL,PRED,SCORES,Path(__file__)]],
        cached_error_parity='PASS', new_inference=False, refinement=False, coordinate_permutation=False,
        GT_used_for_display_and_scoring_only=True, training_updates=0, split_changed=False,
        median_px=float(np.median(pooled)), p90_px=float(np.percentile(pooled,90)),
        truncated_frames=sum(r['truncated'] for r in receipt), records=receipt))
    print(json.dumps(dict(gallery=str(OUT/'index.html'), frames=len(receipt), median_px=float(np.median(pooled)),
                         p90_px=float(np.percentile(pooled,90)), status='PASS'), ensure_ascii=False))


if __name__ == '__main__':
    main()
