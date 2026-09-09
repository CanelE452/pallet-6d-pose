"""Render completed real DEV results only; no model forward or selection."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path

import numpy as np

from . import cache as C


ROOT = Path(__file__).resolve().parents[3]
TITLE = 'Structured V2 · 학습된 점·선 배치 검증'
CASE = 'eval_pallet07:1778652166837872128'
ARMS = ('baseline', 'point_only', 'point_segment', 'point_segment_hough')
LABELS = dict(baseline='Original joint points', point_only='Point patches',
    point_segment='Point + segment', point_segment_hough='Point + segment + DHT')
EDGES = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))


class Inputs:
    def __init__(self):
        self.hashes = {}

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        digest = C.sha(path)
        if expected is not None and digest != expected:
            raise ValueError(f'Report input changed: {path}')
        self.hashes[str(path)] = digest
        return path

    def read(self, path, expected=None):
        return C.read(self.bind(path, expected))

    def verify(self):
        for path, digest in self.hashes.items():
            if C.sha(path) != digest:
                raise ValueError(f'Report input changed during rendering: {path}')


def number(value, digits=2):
    return '—' if value is None else f'{float(value):.{digits}f}'


def table(headers, rows):
    escaped = lambda value: html.escape(str(value))
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>'+escaped(x)+'</th>' for x in headers) + \
        '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+escaped(x)+'</td>' for x in row)+'</tr>' for row in rows) + '</tbody></table></div>'


def transitions(frames, arm):
    """Descriptive per-frame mean changes; never used to choose a model/image."""
    out = dict(improved=0, worsened=0, unchanged=0, unobserved=0)
    for frame in frames:
        pairs = [(a,b) for a,b in zip(frame['metrics']['baseline']['errors_px'],
            frame['metrics'][arm]['errors_px']) if a is not None and b is not None]
        if not pairs:
            out['unobserved'] += 1
            continue
        delta = float(np.mean([b-a for a,b in pairs]))
        out['improved' if delta < -1e-9 else 'worsened' if delta > 1e-9 else 'unchanged'] += 1
    return out


def load_actual(run, inputs, seed):
    """Fail before writing HTML when actual real evaluation is unavailable."""
    names = [f'REAL_RESULTS_seed{seed}.json', f'REAL_FRAME_METRICS_seed{seed}.json',
             f'REAL_PREDICTIONS_seed{seed}.json']
    missing = [name for name in names if not (run/name).is_file()]
    if missing:
        raise ValueError('Actual completed real results required before rendering: '+', '.join(missing))
    results, metrics, predictions = [inputs.read(run/name) for name in names]
    if not (results.get('complete') and results.get('PASS') and metrics.get('complete')
            and predictions.get('complete') and predictions.get('PASS')):
        raise ValueError('Incomplete actual real evaluation')
    if results['seed'] != seed or predictions['seed'] != seed:
        raise ValueError('Result seed mismatch')
    if not results['GT_unchanged'] or not results['same_ID'] or predictions['selection_uses_GT']:
        raise ValueError('Unchanged-GT/same-ID/GT-free selection contract differs')
    for artifact in (results, predictions):
        for mapping in ('input_sha256', 'source_sha256'):
            for path, digest in artifact[mapping].items():
                inputs.bind(path, digest)
    protocol = inputs.read(run/'PROTOCOL.json')
    if tuple(protocol['architecture']['arms']) != ARMS[1:]:
        raise ValueError('Expected all three registered verifier arms')
    metric_map = {row['id']: row for row in metrics['records']}
    ids = [row['id'] for row in predictions['records']]
    if len(ids) != 319 or len(set(ids)) != 319 or set(ids) != set(metric_map) or CASE not in ids:
        raise ValueError('All original319 frames and the required case must remain')
    canonical_path = ROOT/'challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json'
    canonical = inputs.read(canonical_path, results['input_sha256'][str(canonical_path)])
    items = {row['frame_id']: row for row in canonical['items']}
    if set(items) != set(ids) or canonical['role'] != 'DEV':
        raise ValueError('Expected the exact reused DEV manifest')
    from PIL import Image
    frames = []
    for row in predictions['records']:
        metric, item = metric_map[row['id']], items[row['id']]
        source = inputs.bind(row['image'], row['image_sha256'])
        if source != (ROOT/item['image_path']).resolve():
            raise ValueError('Raw image locator changed')
        with Image.open(source) as image:
            w, h = image.size
        annotation_path = (ROOT/item['gt_v2_path']).resolve()
        annotation = inputs.read(annotation_path, results['input_sha256'][str(annotation_path)])['objects'][0]['keypoint_annotations']
        gt_points = [x['xy'] for x in annotation]
        visibility = [x['visibility'] for x in annotation]
        if gt_points != metric['gt_points'] or [v>0 for v in visibility] != metric['gt_supervised']:
            raise ValueError('GT coordinates/mask differ from official metrics')
        for arm in ARMS:
            arm_row = row['baseline'] if arm == 'baseline' else row['arms'][arm]
            if arm_row['points'] != metric['arms'][arm]['points']:
                raise ValueError('Displayed and evaluated prediction coordinates differ')
            if arm_row['point_valid'] != row['baseline']['point_valid'] or arm_row['points'][8] != row['baseline']['points'][8]:
                raise ValueError('Centroid or predicted-point coverage changed')
        frames.append(dict(id=row['id'], session_id=row['session_id'], image_uri=source.as_uri(),
            image_path=str(source), image_sha256=row['image_sha256'], width=w, height=h,
            baseline=row['baseline'], arms=row['arms'], metrics=metric['arms'],
            gt=dict(points=gt_points, visibility=visibility,
                sources=[x.get('source','unknown') for x in annotation]),
            baseline_match_iou50=metric['baseline_match_iou50']))
    real = {row['arm']: row for row in results['summaries']}
    if set(real) != set(ARMS) or real['baseline']['n_observed_points'] != 2738:
        raise ValueError('Canonical comparison/2738 observed points missing')
    if abs(real['baseline']['p90_px']-41.48732863482036) > 1e-10:
        raise ValueError('Original baseline P90 differs')
    selections = {}
    for arm in ARMS[1:]:
        selected = inputs.read(run/f'SELECTION_{arm}_seed{seed}.json')
        if not (selected['complete'] and selected['PASS'] and selected['no_real_selection']
                and selected['arm'] == arm and selected['seed'] == seed):
            raise ValueError('Expected frozen synthetic-only margin selection')
        if any(frame['arms'][arm]['margin'] != selected['margin'] for frame in frames):
            raise ValueError('Displayed margin differs from frozen selection')
        selections[arm] = selected['margin']
    return frames, results, selections, protocol


def render(run_dir, seed=1):
    run = Path(run_dir).resolve()
    if not (run/'PURPOSE.md').is_file():
        raise ValueError('Result PURPOSE.md required')
    inputs = Inputs()
    inputs.bind(__file__)
    inputs.bind(C.__file__)
    frames, result, margins, protocol = load_actual(run, inputs, seed)
    summary = {row['arm']: row for row in result['summaries']}
    before, after = summary['baseline'], summary['point_segment_hough']
    direction = '감소' if after['p90_px'] < before['p90_px'] else '증가' if after['p90_px'] > before['p90_px'] else '동일'
    headline = f"전체 점·선 모델의 실사 P90은 {before['p90_px']:.2f} → {after['p90_px']:.2f}px로 {direction}했습니다. Median {before['median_px']:.2f} → {after['median_px']:.2f}px. 한 학습 seed와 재사용 DEV 결과이며 안정적 향상이나 최종 일반화 검증으로 해석하지 않습니다."
    metrics = table(['Method','Median px','P90 px','Mean px','PCK10 %','>20 px %','Observed / GT','Margin'],
        [[LABELS[arm],number(row['median_px']),number(row['p90_px']),number(row['mean_px']),
          number(100*row['pck10_observed']),number(100*row['above20_fraction']),
          f"{row['n_observed_points']} / {row['n_gt_points']}",margins.get(arm,'—')]
         for arm,row in summary.items()])
    changes = {arm: transitions(frames,arm) for arm in ARMS[1:]}
    change_table = table(['Method','Improved frames','Worsened frames','Unchanged frames','Unobserved frames'],
        [[LABELS[arm],row['improved'],row['worsened'],row['unchanged'],row['unobserved']] for arm,row in changes.items()])
    paired = table(['Paired comparison','Mean frame Δ px','Session bootstrap 95% CI','Frames / sessions'],
        [[LABELS[row['left']]+' − '+LABELS[row['right']],number(row['mean_frame_delta_px']),
          '['+', '.join(number(v) for v in row['ci95'])+']',f"{row['n_frames']} / {row['n_sessions']}"]
         for row in result['paired']])
    difficulty = table(['Method','Baseline error bin','Points','Before mean px','After mean px','≤10 → >10 %'],
        [[LABELS[row['arm']],row['difficulty'],row['n_points'],number(row['baseline_mean_px']),number(row['after_mean_px']),
          number(None if row['easy_to_over10_rate'] is None else 100*row['easy_to_over10_rate'])]
         for row in result['difficulty'] if row['arm'] != 'baseline'])
    data = dict(frames=frames, labels=LABELS, arms=ARMS, edges=EDGES, required_case=CASE, margins=margins)
    replacements = dict(TITLE=TITLE,HEADLINE=html.escape(headline),METRICS=metrics,CHANGES=change_table,
        PAIRED=paired,DIFFICULTY=difficulty,DATA=json.dumps(data,ensure_ascii=False,allow_nan=False).replace('</','<\\/'),
        BUDGET=html.escape(f"train {protocol['split']['train_count']} · calibration {protocol['split']['calibration_count']} · synthetic validation {protocol['split']['synthetic_validation_count']} · {protocol['training']['steps']} updates / arm · seed {seed}"))
    document = TEMPLATE
    for key,value in replacements.items():
        document = document.replace('@@'+key+'@@',value)
    if '@@' in document or '<script src=' in document:
        raise ValueError('Unresolved or external report dependency')
    inputs.verify()
    page = run/'index.html'
    page.write_text(document)
    receipt = dict(complete=True,PASS=True,experiment_complete=True,
        scope='Completed single-seed real evaluation report, not completion of the active improvement goal.',
        active_goal_complete=False,title=TITLE,html=str(page),html_sha256=C.sha(page),
        seed=seed,n_real_frames=319,n_displayed_methods=4,input_sha256=inputs.hashes,
        external_dependencies=False,local_original_images=True,no_model_forward=True,no_desktop_open=True,
        created_at_utc=datetime.now(timezone.utc).isoformat())
    C.write(run/'REPORT_RENDER.json',receipt)
    print(json.dumps({key:receipt[key] for key in ('complete','PASS','html','html_sha256')},ensure_ascii=False))
    return receipt


TEMPLATE = '''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>@@TITLE@@</title><style>
:root{font:15px Arial,sans-serif;color:#e7eef5;background:#10202d;color-scheme:dark}*{box-sizing:border-box}main{max-width:1580px;margin:auto;padding:22px}h1{font-size:27px}h2{font-size:20px}p{line-height:1.7}.notice,section{padding:18px;background:#183044;border:1px solid #405b70;border-radius:8px;margin:18px 0}.notice{border-left:4px solid #70bfff}.muted{font-size:13px;color:#bacbd9}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{font-weight:normal}thead{border-top:1px solid #526e83;border-bottom:1px solid #526e83}tbody{border-bottom:1px solid #526e83}.controls{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}label{font-size:13px}input,select,button{background:#24445c;color:inherit;padding:8px;border:1px solid #648199;border-radius:4px}select{max-width:380px;display:block;margin-top:5px}.panels{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.panel{min-width:0}.panel h3{font-size:14px;font-weight:normal}canvas{display:block;width:100%;height:auto;background:#10202d}.details-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.swatch{display:inline-block;width:13px;height:9px;margin:0 6px 0 15px}.legend{font-size:13px}.cost svg{width:100%;height:185px}details{margin:16px 0}summary{cursor:pointer}a{color:#91d2ff}@media(max-width:1050px){.panels{grid-template-columns:1fr}.details-grid{grid-template-columns:1fr}}
</style></head><body><main><h1>@@TITLE@@</h1><p class="notice">@@HEADLINE@@</p><p>기존 hough_joint seed1 모델을 고정하고, 8개 점·12개 유한 선분·DHT 선 모드로 전체 배치 비용을 학습했습니다. <b>Point patches 대조군도 같은 DHT 교점 후보 풀을 사용합니다.</b> 이 비교는 비용을 계산할 때 선분 내부와 DHT 증거를 추가한 효과이며, DHT를 완전히 제거한 CNN 대조가 아닙니다.</p><p class="muted">@@BUDGET@@. 실사 GT로 학습하거나 margin을 선택하지 않았습니다. 모든 GT·번호·관측 분모를 유지했고 중심점 ID8은 복사했습니다. 아래 그림은 실제 좌표와 선택 비용이며 attention 또는 인과 설명이 아닙니다.</p>
<section id="quantitative"><h2>실제 319장 / 13세션 결과</h2>@@METRICS@@<p class="muted">오차는 같은-ID 9점의 관측값 기준입니다. 2,738개 관측점과 2,818개 GT 감독점의 차이는 기존 결측·검출 매칭 정책을 그대로 반영합니다. PCK10은 관측점 기준입니다.</p>@@CHANGES@@<p class="muted">프레임 평균 점오차의 감소/증가/동일 횟수입니다(동일 허용오차 10⁻⁹px). 개선 횟수가 많아도 큰 실패 때문에 평균·P90이 악화할 수 있습니다.</p><details id="bootstrap"><summary>대응 세션 bootstrap와 원래 점의 난이도</summary>@@PAIRED@@<p class="muted">실제 저장된 95% 구간입니다. 평균 프레임 오차 차이의 구간이며 median·P90의 구간으로 바꾸어 읽으면 안 됩니다. 13개 기존 세션을 재표집한 단일 학습 seed의 탐색 결과입니다.</p>@@DIFFICULTY@@<p class="muted">GT 오류로 나눈 사후 난이도이며 학습·선택 규칙으로 사용하지 않았습니다.</p></details></section>
<section id="gallery"><h2>원본과 실제 선택 비교</h2><div class="controls"><label>Search ID / session<input id="search" type="search" placeholder="pallet07, 037376 …"></label><label>Image <span id="found"></span><select id="frame"></select></label><label>Verifier<select id="arm"><option value="point_segment_hough">Point + segment + DHT</option><option value="point_segment">Point + segment</option><option value="point_only">Point patches</option></select></label><label><input id="showgt" type="checkbox" checked>GT</label><label><input id="labels" type="checkbox" checked>Point IDs</label><label><input id="edges" type="checkbox" checked>12 structural edges</label><button id="originalcase">Original user case</button></div><p class="legend"><span class="swatch" style="background:#32dc72"></span>GT · circle v2 / diamond v1 / cross v0<span class="swatch" style="background:#42baff"></span>Original prediction<span class="swatch" style="background:#ffd24a"></span>Selected prediction</p><p id="frameinfo" class="notice"></p><div class="panels"><div class="panel"><h3>Original raw image</h3><canvas id="raw"></canvas></div><div class="panel"><h3>Original frozen joint points</h3><canvas id="before"></canvas></div><div class="panel"><h3 id="aftertitle">Selected learned layout</h3><canvas id="after"></canvas></div></div><div class="details-grid"><div id="pointtable"></div><div><h3>Actual C4 candidate costs − identity</h3><div id="cost" class="cost"></div><p id="selectioninfo" class="muted"></p><p class="muted">0/1/2/3은 저장된 원래/C4 90°/180°/270° 후보입니다. 음수는 identity보다 낮은 모델 비용이며 정답 확률이 아닙니다. 최종 선택은 전체 유효 후보 중에서 이루어집니다.</p></div></div></section>
<section><h2>해석 범위</h2><p>GT에는 직접 클릭·투영·가림·출처 미기록 점이 섞여 있습니다. v&gt;0는 좌표 감독 여부이며 실제 물리 경계의 가시성을 보증하지 않습니다. 기존 정성 검토도 모든 GT의 픽셀 정확도 인증은 아닙니다. 이 불확실성을 이유로 이번 평가의 GT나 분모를 수정하지 않았습니다.</p><p>합성 훈련 1,792장과 calibration 256장은 기존 2,048장 안에서 분리했습니다. 별도 합성 validation 512장으로 진행 여부를 확인했으며, 기존 backbone이 본 합성 계보와 재사용 실사 DEV의 한계는 남습니다. 다중 학습 seed의 안정성, 독립 FINAL 일반화, 6D 개선을 이 표만으로 주장하지 않습니다.</p><p><a href="REAL_RESULTS_seed1.json">Real results</a> · <a href="REAL_PREDICTIONS_seed1.json">Actual predictions</a> · <a href="REAL_FRAME_METRICS_seed1.json">Point errors</a> · <a href="PROTOCOL.json">Protocol</a> · <a href="REPORT_RENDER.json">Report bindings</a></p></section></main><script id="report-data" type="application/json">@@DATA@@</script><script>
const DATA=JSON.parse(document.getElementById('report-data').textContent),$=id=>document.getElementById(id);let current=DATA.required_case,token=0;const images=new Map(),colors={gt:'#32dc72',base:'#42baff',selected:'#ffd24a'};
function option(select,value,label){const o=document.createElement('option');o.value=value;o.textContent=label;select.append(o)}
function average(values){const a=values.filter(x=>x!==null);return a.length?a.reduce((s,x)=>s+x,0)/a.length:null}
function fmt(v){return v===null?'—':Number(v).toFixed(2)}
function line(ctx,a,b,color,width=1.5){if(!a||!b)return;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke()}
function marker(ctx,p,index,color,visibility=2){if(!p)return;ctx.save();ctx.lineWidth=2;ctx.strokeStyle='#10202d';ctx.fillStyle=color;ctx.beginPath();if(visibility===1){ctx.moveTo(p[0],p[1]-5);ctx.lineTo(p[0]+5,p[1]);ctx.lineTo(p[0],p[1]+5);ctx.lineTo(p[0]-5,p[1]);ctx.closePath()}else{ctx.arc(p[0],p[1],4,0,Math.PI*2)}ctx.fill();ctx.stroke();if(visibility===0){line(ctx,[p[0]-5,p[1]-5],[p[0]+5,p[1]+5],color,2);line(ctx,[p[0]-5,p[1]+5],[p[0]+5,p[1]-5],color,2)}if($('labels').checked){ctx.font='bold 12px Arial';ctx.lineWidth=3;ctx.strokeStyle='#10202d';ctx.strokeText(index,p[0]+6,p[1]-5);ctx.fillStyle=color;ctx.fillText(index,p[0]+6,p[1]-5)}ctx.restore()}
function layout(ctx,points,valid,color,visibility=null){if($('edges').checked)for(const [u,v]of DATA.edges)if(valid[u]&&valid[v])line(ctx,points[u],points[v],color);points.forEach((p,i)=>{if(valid[i]||visibility)marker(ctx,p,i,color,visibility?visibility[i]:2)})}
function costplot(row){const values=row.costs.slice(0,4).map(v=>v===null?null:v-row.identity_cost),range=Math.max(.01,...values.filter(v=>v!==null).map(Math.abs)),center=250,scale=210/range;let svg='<svg viewBox="0 0 550 185" role="img" aria-label="Actual C4 costs minus identity"><line x1="250" y1="5" x2="250" y2="175" stroke="#9fb9ca"/>';values.forEach((v,i)=>{const y=22+i*42;svg+='<text x="3" y="'+(y+5)+'" fill="#e7eef5" font-size="13">C4 '+(i*90)+'°</text>';if(v!==null){const end=center+v*scale;svg+='<rect x="'+Math.min(center,end)+'" y="'+(y-9)+'" width="'+Math.max(1,Math.abs(end-center))+'" height="18" fill="#ffd24a"/><text x="475" y="'+(y+5)+'" fill="#e7eef5" font-size="13">'+v.toFixed(3)+'</text>'}else svg+='<text x="475" y="'+(y+5)+'" fill="#b9cad7" font-size="13">invalid</text>'});$('cost').innerHTML=svg+'</svg>'}
function pointtable(f,arm){const rows=f.gt.points.map((p,i)=>'<tr><td>'+i+(i===8?' · center':'')+'</td><td>'+f.gt.visibility[i]+'</td><td>'+fmt(f.metrics.baseline.errors_px[i])+'</td><td>'+fmt(f.metrics[arm].errors_px[i])+'</td><td>'+fmt(f.metrics[arm].move_px[i])+'</td><td></td></tr>').join('');$('pointtable').innerHTML='<h3>Same-ID point errors</h3><div class="scroll"><table><thead><tr><th>ID</th><th>GT v</th><th>Before px</th><th>After px</th><th>Move px</th><th>GT source</th></tr></thead><tbody>'+rows+'</tbody></table></div>';Array.from($('pointtable').querySelectorAll('tbody tr')).forEach((r,i)=>r.lastChild.textContent=f.gt.sources[i])}
async function render(){window.REPORT_READY=false;const ticket=++token,f=DATA.frames.find(f=>f.id===current);if(!f){window.REPORT_READY=true;return}const arm=$('arm').value,row=f.arms[arm];let image=images.get(f.id);if(!image){image=new Image();image.src=f.image_uri;await image.decode();images.set(f.id,image)}if(ticket!==token)return;for(const id of ['raw','before','after']){const cv=$(id);cv.width=f.width;cv.height=f.height;const ctx=cv.getContext('2d');ctx.drawImage(image,0,0);if(id==='raw')continue;if($('showgt').checked)layout(ctx,f.gt.points,f.gt.visibility.map(v=>v>0),colors.gt,f.gt.visibility);const q=id==='before'?f.baseline:row;layout(ctx,q.points,q.point_valid,id==='before'?colors.base:colors.selected)}$('aftertitle').textContent=DATA.labels[arm]+' · selected layout';$('frameinfo').textContent=f.id+' | '+f.width+'×'+f.height+' | original mean '+fmt(average(f.metrics.baseline.errors_px))+'px → selected mean '+fmt(average(f.metrics[arm].errors_px))+'px | baseline IoU match: '+f.baseline_match_iou50;pointtable(f,arm);costplot(row);$('selectioninfo').textContent='Selected index '+row.selected_index+' / margin '+row.margin+' / identity cost '+Number(row.identity_cost).toFixed(4)+' / selected cost '+Number(row.selected_cost).toFixed(4)+' / cost reduction '+Number(row.identity_cost-row.selected_cost).toFixed(4)+'. GT errors above are evaluation-only. ID8 is copied.';window.REPORT_CURRENT_FRAME=f.id;window.REPORT_READY=true}
function filter(){const query=$('search').value.toLowerCase().trim(),rows=DATA.frames.filter(f=>(f.id+' '+f.session_id).toLowerCase().includes(query));$('frame').replaceChildren();rows.forEach(f=>option($('frame'),f.id,f.id));$('found').textContent=rows.length+'/319';if(!rows.length){$('frameinfo').textContent='No matching frame';window.REPORT_READY=true;return}if(!rows.some(f=>f.id===current))current=rows[0].id;$('frame').value=current;render()}
$('search').oninput=filter;$('frame').onchange=()=>{current=$('frame').value;render()};for(const key of ['arm','showgt','labels','edges'])$(key).onchange=render;$('originalcase').onclick=()=>{$('search').value='';current=DATA.required_case;filter()};window.REPORT_FRAME_COUNT=DATA.frames.length;filter();
</script></body></html>'''


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=1)
    args=parser.parse_args()
    render(args.run_dir,args.seed)
