"""Build and open a local explanation of target scope, padding and viewpoint."""
from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
from pathlib import Path

import markdown


def read(path):
    return json.loads(path.read_text())


def report_section(path, output_dir):
    rendered = markdown.markdown(path.read_text(), extensions=['tables', 'fenced_code'])
    def fix(match):
        attribute, value = match.groups()
        if re.match(r'^(?:https?:|data:|file:|#)', value):
            return match.group(0)
        relative = os.path.relpath((path.parent / value).resolve(), output_dir)
        return f'{attribute}="{html.escape(relative, quote=True)}"'
    return re.sub(r'(href|src)="([^"]+)"', fix, rendered)


def build(run_dir):
    out = run_dir / 'diagnosis_followup'
    out.mkdir(exist_ok=True)
    cfg = read(run_dir / 'CONFIG.json')
    manifest = read(Path(cfg['source_cache']) / 'manifest.json')
    records = {r['id']: r for r in manifest['populations']['real_dev']}
    ids = ['eval_cad__1778653055734035712', 'eval_outside__1778653345465966336']
    examples = []
    for sample_id in ids:
        record = records[sample_id]
        path = Path(record['image'])
        mime = 'image/png' if path.suffix.lower() == '.png' else 'image/jpeg'
        examples.append(dict(id=sample_id, width=record['width'], height=record['height'],
                             points=record['gt_points'], image=f'data:{mime};base64,' + base64.b64encode(path.read_bytes()).decode()))
    reports = []
    for title, relative in [('카메라 구도와 학습 데이터 분포', 'diagnosis_viewpoint/VIEWPOINT_FINDINGS.md'),
                            ('패딩을 바꾸어 본 추가 진단', 'diagnosis_padding/PAD_FINDINGS.md')]:
        path = run_dir / relative
        if not path.exists():
            raise FileNotFoundError(path)
        figure = ('<img src="../diagnosis_viewpoint/coverage_and_errors.png" '
                  'alt="학습과 실제 이미지의 면적비 분포 및 DHT 오차">'
                  if relative.startswith('diagnosis_viewpoint/') else '')
        reports.append(f'<section><h2>{title}</h2>{report_section(path, out)}{figure}</section>')
    body = '''<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>팔레트 DHT: 양쪽 면, 패딩, 정면 실패 진단</title>
<style>
body{max-width:1120px;margin:auto;padding:24px;background:#f1f4f8;color:#142238;font-family:system-ui,sans-serif;line-height:1.7}
section{background:white;border:1px solid #d5ddeb;border-radius:12px;padding:24px;margin:20px 0}
h1{font-size:26px}h2{font-size:22px}h3{font-size:18px}p{margin:12px 0}a{color:#155ab6}
.controls{display:flex;gap:18px;flex-wrap:wrap;align-items:center;margin:12px 0}select{padding:8px;font:inherit}
svg{display:block;background:#e6ebf3;width:100%;max-height:620px}img{max-width:100%;height:auto}
table{border-collapse:collapse;display:block;max-width:100%;overflow:auto;font-size:14px}th,td{padding:9px 12px;border-bottom:1px solid #dde3ec;text-align:left}
pre{overflow:auto;padding:12px;background:#f4f6fa}.note{color:#536278;font-size:14px}.takeaway{border-left:4px solid #2e76bb;padding-left:16px}
</style><body><h1>양쪽 면·패딩·정면 실패를 다시 확인했습니다</h1>
<p class="takeaway">현재 모델은 좌우 측면의 8개 선을 모두 예측합니다. 앞·뒤 면의 폭 방향 가로선 4개는 빠져 있습니다.
학습 데이터에는 낮게 보는 구도가 부족하며, 패딩이 미치는 영향은 같은 모델의 입력을 바꾸어 별도로 확인했습니다.</p>
<p><a href="../deep_hough_gallery.html">기존 예측·민감도 갤러리</a> · <a href="../REPORT.md">원래 학습 실험 보고서</a></p>
<section><h2>어떤 선을 학습했는지 직접 켜고 끄기</h2>
<p>아래는 <strong>정답 코너로 그린 학습 대상 설명</strong>입니다. 모델 예측이나 새 학습 결과가 아닙니다.
양쪽 측면을 켜면 현재의 8개 선을, 주황 점선을 켜면 이번 출력에서 제외한 나머지 4개 선을 볼 수 있습니다.</p>
<div class="controls"><select id="example"><option value="0">윗면이 넓게 보이는 실제 이미지</option><option value="1">낮은 구도의 실제 이미지</option></select>
<label style="color:#008c58"><input id="left" type="checkbox" checked>왼쪽 측면 4개</label>
<label style="color:#1478e6"><input id="right" type="checkbox" checked>오른쪽 측면 4개</label>
<label style="color:#b55d00"><input id="width" type="checkbox">제외된 앞·뒤 폭 방향 4개</label></div>
<svg id="scope" role="img" aria-label="실제 이미지 위에 표시한 측면 8개 선과 제외된 4개 선"></svg>
<p id="scope-count" class="note"></p>
<p>기존 갤러리 상단은 전체 8개 예측을 보여주고, 하단은 선택한 선 하나를 자세히 보여줍니다.
두 측면을 예측해도 12개 선이 일관된 하나의 입체로 연결되도록 강제하는 단계는 현재 없습니다.
가려진 선에는 영상 증거가 없을 수 있으며, 추가 기하 제약이 필요합니다.</p></section>
<section><h2>왜 반사 패딩을 썼는가</h2>
<p>이번 선택의 근거는 기존 Direct 실험과 입력 조건을 유지하는 것이었습니다. 반사 패딩의 성능 우위를 사전에 검증한 것은 아닙니다.
반사는 주변 무늬를 뒤집어 연장하고, 공백 패딩은 일정한 색으로 채웁니다.
<a href="https://docs.opencv.org/4.13.0/d2/de8/group__core__array.html">OpenCV 패딩 정의</a></p>
<p>반사는 단색 테두리와 원영상 사이의 갑작스러운 밝기 차이를 줄일 수 있지만, 팔레트와 배경의 가짜 복사본도 만듭니다.
후보 선을 따라 특징을 모으는 이 DHT에서는 그 복사본이 추가 단서나 혼동 요인이 될 수 있습니다. 현재 집계에 원영상 유효 영역 마스크는 없습니다.</p>
<p>640×480에 사방 100px을 더하면 840×680이 되며, <strong>전체 면적의 46.2%가 패딩</strong>입니다.
50×50 특징에서 원영상이 차지하는 크기는 약 38.1×35.3칸입니다. 같은 두께의 공백으로 바꾸어도 이 해상도 손실은 남습니다.</p></section>
__REPORTS__
<section><h2>정면을 개선하기 위한 다음 비교</h2>
<ol><li><strong>앞·뒤 폭 방향 가로선까지 포함한 12개 구조선</strong>을 학습하고, 앞면과 옆면의 오차를 따로 평가합니다. 정면에서 남는 긴 경계도 학습 신호로 사용합니다.</li>
<li><strong>낮은 높이·정면·원거리·작은 팔레트</strong>의 조합을 합성 학습에 보강합니다. 수평 방향만 늘리지 말고 영상 속 면적과 경계 길이 분포도 맞춥니다.</li>
<li><strong>팔레트 검출 후 확대 입력 또는 높은 해상도의 특징</strong>을 비교합니다. 공백/반사 패딩의 재학습 비교와 유효 영역을 사용하는 DHT 집계도 별도 요인으로 확인합니다.</li>
<li><strong>보이는 코너와 팔레트 치수·카메라 보정값으로 포즈를 추정</strong>하고 숨은 외곽을 투영하는 방법을 비교합니다. 평면·대칭으로 여러 해가 가능하므로 다른 코너, 깊이 또는 시간 정보를 이용한 검증이 필요합니다.
<a href="https://docs.opencv.org/4.13.0/d5/d1f/calib3d_solvePnP.html">OpenCV PnP의 입력 조건과 다중 해</a></li></ol>
<p class="note">위 항목은 아직 수행하지 않은 다음 학습 비교 제안입니다. 이번 추가 진단은 기존 개발셋과 고정된 모델을 사용했고 학습 가중치를 바꾸지 않았습니다.</p></section>
<script id="examples" type="application/json">__EXAMPLES__</script>
<script>
const samples=JSON.parse(document.getElementById('examples').textContent);
const groups={left:{edges:[[0,4],[4,7],[7,3],[3,0]],color:'#27ff9c'},right:{edges:[[1,5],[5,6],[6,2],[2,1]],color:'#36b4ff'},width:{edges:[[0,1],[2,3],[4,5],[6,7]],color:'#ffaf35'}};
const ns='http://www.w3.org/2000/svg';
function make(name,attrs){const n=document.createElementNS(ns,name);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,String(v));return n;}
function refresh(){const sample=samples[Number(document.getElementById('example').value)],svg=document.getElementById('scope');
svg.replaceChildren();svg.setAttribute('viewBox',`0 0 ${sample.width} ${sample.height}`);
svg.append(make('image',{href:sample.image,width:sample.width,height:sample.height}));let count=0;
for(const [key,g] of Object.entries(groups)){if(!document.getElementById(key).checked)continue;for(const [a,b] of g.edges){const p=sample.points[a],q=sample.points[b];
svg.append(make('line',{x1:p[0],y1:p[1],x2:q[0],y2:q[1],stroke:g.color,'stroke-width':3,'stroke-dasharray':key==='width'?'8 5':'none','data-group':key}));count++;}}
document.getElementById('scope-count').textContent=`표시한 정답 선: ${count}개. 양쪽 측면 합계 8개가 현재 학습 대상입니다. 주황 점선은 학습에서 빠진 선입니다. 원본 이미지: ${sample.id}`;}
for(const id of ['example','left','right','width'])document.getElementById(id).addEventListener('change',refresh);refresh();
</script></body></html>'''
    body = body.replace('__REPORTS__', '\n'.join(reports))
    body = body.replace('__EXAMPLES__', json.dumps(examples, ensure_ascii=False).replace('<', '\\u003c'))
    path = out / 'viewpoint_padding_diagnosis.html'
    path.write_text(body, encoding='utf-8')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--no-open', action='store_true')
    args = parser.parse_args()
    path = build(args.run_dir.resolve())
    print(path)
    if not args.no_open:
        from visualize import open_gallery
        open_gallery(path)


if __name__ == '__main__':
    main()
