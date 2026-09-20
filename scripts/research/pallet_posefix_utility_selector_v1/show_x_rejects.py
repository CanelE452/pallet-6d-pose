"""Display exactly the non-green X-only rejects without altering predictions."""
import base64
import html
import json

import numpy as np
from PIL import Image

from . import x_crossing_filter as X


def main():
    P = X.P
    lock = P.read(X.DOC / 'OUTPUTS_LOCK.json')
    for binding in lock['artifacts'] + [lock['protocol']]:
        P.verify_binding(binding)
    decisions = P.read(X.RAW / 'DECISIONS.json')['DEV72']
    source = {r['id']: r for r in P.read(X.A.RAW / 'ACCEPTED_UNCHANGED_PREDICTIONS.json')['DEV72']}
    cards = []
    ids = []
    for decision in decisions:
        if decision['keep']:
            continue
        row = source[decision['id']]
        P.verify_binding(row['image'])
        path = P.ROOT / row['image']['path']
        with Image.open(path) as im:
            width, height = im.size
        uri = 'data:image/' + ('png' if path.suffix.lower() == '.png' else 'jpeg') + ';base64,' + base64.b64encode(path.read_bytes()).decode()
        q = np.asarray(X.A.N.top(row['prediction'])['keypoints_xy'], float)[:8]
        assert X.inspect(q) == {k: v for k, v in decision.items() if k != 'id'}
        edges = sorted({tuple(sorted((face[i], face[(i+1)%4]))) for face in X.FACES for i in range(4)})
        shapes = [f'<line x1="{q[a,0]}" y1="{q[a,1]}" x2="{q[b,0]}" y2="{q[b,1]}" stroke="#ffe34f" stroke-width="1.5"/>' for a,b in edges]
        crossing_indices = set()
        for face in decision['faces']:
            if not face['x_crossing']:
                continue
            idx = face['corners']; crossing_indices.update(idx)
            pts = ' '.join(f'{x},{y}' for x,y in q[idx])
            shapes.append(f'<polygon points="{pts}" fill="none" stroke="#ff4848" stroke-width="2.5"/>')
        for i, (x,y) in enumerate(q):
            shapes.append(f'<circle cx="{x}" cy="{y}" r="2.5" fill="#ffe34f"/><text x="{x+4}" y="{y-5}" font-size="10" fill="white" stroke="#101820" stroke-width="2" paint-order="stroke">P{i}</text>')
        lo = np.maximum(q[list(crossing_indices)].min(0)-25, 0)
        hi = np.minimum(q[list(crossing_indices)].max(0)+25, [width,height])
        crop = f'{lo[0]} {lo[1]} {hi[0]-lo[0]} {hi[1]-lo[1]}'
        def svg(viewbox):
            return f'<svg viewBox="{viewbox}" role="img"><image width="{width}" height="{height}" href="{uri}"/><g class="marks">'+''.join(shapes)+'</g></svg>'
        ids.append(decision['id'])
        cards.append('<article><h2>'+html.escape(decision['id'])+'</h2><p>교차 면: '+html.escape(', '.join(decision['crossing_faces']))+'</p>'+svg(f'0 0 {width} {height}')+'<h3>X 교차 면 확대</h3>'+svg(crop)+'</article>')
    assert len(ids) == 2
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>비초록 X 교차 탈락 2장</title><style>body{background:#101820;color:#eaf1f8;font:16px/1.6 system-ui;margin:0}main{max-width:1050px;margin:auto;padding:20px}article{background:#172737;margin:30px 0;padding:14px}h2{font-size:18px;overflow-wrap:anywhere}svg{width:100%;display:block;max-height:650px}button{padding:10px;background:#29475e;color:white;border:1px solid #7095ad;cursor:pointer}.raw .marks{display:none}</style><main><h1>비초록: X 교차로 탈락한 2장</h1><p>노랑 = 변경하지 않은 Replay 보정 좌표 · 빨강 = 같은 면 내부에서 변이 교차한 면.<br>각 사진 아래에 해당 면을 확대했습니다. GT를 사용한 탈락 판정이 아닙니다.</p><p>주의: 두 장의 기존 평가 기준점 오차는 모두 10px 이내였습니다. X 교차 검출이 큰 위치 오차를 뜻하지는 않습니다.</p><button onclick="document.body.classList.toggle('raw')">원본 / 표시 전환</button>'''+''.join(cards)+'</main></html>'
    out = P.OUT / 'x_crossing_filter'
    out.mkdir(parents=True, exist_ok=True)
    target = out / 'NON_GREEN_REJECTED.html'
    target.write_text(page, encoding='utf-8')
    print(json.dumps(dict(page=str(target), ids=ids, coordinates_changed=False), ensure_ascii=False))


if __name__ == '__main__':
    main()
