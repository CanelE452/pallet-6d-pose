"""Actual photograph and requested lower line tolerance; no pose modification."""
import json
import shutil

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from . import common as C
from . import statewise_evaluation_v4 as V
from .render_keypoint_process_gallery_v1 import setup_font

BASE = V.FIX / 'boundary_recovery_v1'
OUT = BASE / 'lower_band_case_v1'


def main():
    if OUT.exists():
        raise FileExistsError('Preserve completed line comparison')
    diagnostic_path = BASE / 'kp3_bottom_lower_band_v1/DIAGNOSTIC.json'
    diag = C.read(diagnostic_path)
    prior_path = BASE / 'kp3_bottom_piecewise_v1/DIAGNOSTIC.json'
    if C.sha(prior_path) != diag['bindings']['original_piecewise_diagnostic_sha256']:
        raise ValueError('Original fit changed')
    prior = C.read(prior_path)
    image_path = C.ROOT / diag['bindings']['image_path']
    if C.sha(image_path) != diag['bindings']['image_sha256']:
        raise ValueError('Actual photograph changed')
    selected = prior['selected_representative']
    original = diag['variants'][0]
    envelope = next(v for v in diag['variants'] if v['name'] == 'OUTER_INLIER_ENVELOPE')
    support_intervals = []
    for name in ('left', 'right'):
        ray = selected[name]
        points = np.asarray(ray['observed_points'])[np.asarray(ray['inlier_mask'], bool)]
        n = np.asarray(ray['normal'])
        values = points @ np.array([-n[1], n[0]])
        support_intervals.append([float(values.min()), float(values.max())])
    contour = np.asarray(prior['actual_contour'])
    rgb = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    OUT.mkdir(parents=True)
    shutil.copyfile(image_path, OUT / 'actual.png')
    setup_font()
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), gridspec_kw={'height_ratios': [1, 1.7]})
    colors = ['#eda126', '#cb7ee7']

    def draw_line(ax, line, interval, intersection, color, alpha=1., lw=1.8):
        n = np.asarray(line['normal_unit_xy'])
        t = np.array([-n[1], n[0]])
        lo, hi = interval
        corner_t = float(np.asarray(intersection) @ t)
        span = [min(lo, corner_t) - 3., max(hi, corner_t) + 3.]
        segment = -line['offset_px'] * n + np.asarray(span)[:, None] * t
        ax.plot(segment[:, 0], segment[:, 1], color=color, alpha=alpha, lw=lw, ls='--')
        solid = -line['offset_px'] * n + np.asarray(interval)[:, None] * t
        ax.plot(solid[:, 0], solid[:, 1], color=color, alpha=alpha, lw=lw)

    for col, variant in enumerate((original, envelope)):
        for row in (0, 1):
            ax = axes[row, col]
            ax.imshow(rgb, interpolation='nearest')
            ax.plot(contour[:, 0], contour[:, 1], color='#00bf88', lw=1.)
            for branch, color in enumerate(colors):
                if col:
                    draw_line(ax, original['lines'][branch], support_intervals[branch],
                              original['geometry_intersection_only_xy'], '#c6ced9', .8, 1.)
                draw_line(ax, variant['lines'][branch], support_intervals[branch],
                          variant['geometry_intersection_only_xy'], color)
            point = variant['geometry_intersection_only_xy']
            ax.scatter(*point, marker='x', s=70, c='white', linewidths=2.)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_xlim(205, 502)
                ax.set_ylim(294, 240)
            else:
                ax.set_xlim(345, 399)
                ax.set_ylim(289, 271)
                ax.set_title(f'선의 교점 후보 ({point[0]:.2f}, {point[1]:.2f})', fontsize=11)
        axes[0, col].set_title('기존 중심선' if col == 0 else '관측점에서 계산한 아래쪽 허용선', fontsize=13)
    fig.suptitle('두 선을 약 1.2px 낮춰, 기존 선과의 사이를 허용 범위로 봅니다', fontsize=15)
    fig.text(.03, .04,
             '왼쪽 +1.16px · 오른쪽 +1.25px (원본 이미지 기준). 초록=SAM 윤곽, 회색=기존 선, 주황/보라=현재 선.\n'
             '실선=표본을 관측한 구간, 점선=교점까지의 연장. 흰 ×는 두 직선의 교점 후보이며, 실제 KP로 확정한 점이 아닙니다.',
             fontsize=10)
    fig.subplots_adjust(left=.035, right=.98, bottom=.15, top=.87, wspace=.10, hspace=.25)
    figure = OUT / 'LOWER_LINES_ACTUAL_KO.png'
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    data = dict(contour=contour.tolist(), support_intervals=support_intervals, variants=diag['variants'])
    document = '''<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>하단 두 직선 — 아래쪽 허용 범위</title>
<style>body{margin:0;background:#f4f6f8;color:#192838;font:16px/1.65 system-ui,sans-serif}
main{max-width:1140px;margin:24px auto;padding:0 18px 36px}h1{font-size:27px;margin-bottom:8px}
.card{background:white;border:1px solid #dce2e9;border-radius:12px;padding:18px;margin:18px 0}
.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center}button{font:inherit;padding:7px 12px;border:1px solid #bac7d5;border-radius:7px;background:white;cursor:pointer}
button[aria-pressed=true]{background:#e8f3ff;border-color:#2278be}canvas{width:100%;display:block;background:#242a30;max-height:360px}
.zoom{max-width:730px;margin:16px auto 0}.small{font-size:14px;color:#526273}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:8px;border-bottom:1px solid #dce2e9}
a{color:#1767a8}input[type=range]{width:240px;max-width:100%}output{min-width:58px}img{max-width:100%}
@media(max-width:600px){h1{font-size:23px}main{padding:0 10px}.card{padding:12px}th,td{font-size:14px;padding:6px}}</style></head>
<body><main><h1>두 선에 아래쪽 여유를 주었습니다</h1>
<p>이 사진에서는 약 1px 낮춘 선이 영상 경계를 더 잘 따라갑니다. 기본 화면은 관측점에서 계산한 <b>왼쪽 +1.16px / 오른쪽 +1.25px</b>의 아래쪽 허용선입니다.</p>
<div class="card"><div class="controls" role="group" aria-label="선 위치 비교">
<button data-mode="CENTER">기존 선</button><button data-mode="DOWN_1PX">아래로 1px</button>
<button data-mode="DOWN_2PX">아래로 2px</button><button data-mode="OUTER_INLIER_ENVELOPE" aria-pressed="true">관측점의 아래쪽 허용선</button></div>
<div class="controls" style="margin:14px 0"><label for="shift">두 선 함께 내리기</label><input id="shift" type="range" min="0" max="2" step="0.1" value="0"><output id="shiftValue">개별 이동</output>
<label><input type="checkbox" id="outline" checked> SAM 윤곽 표시</label></div>
<p id="status" aria-live="polite"></p><canvas id="wide" width="1600" height="330" aria-label="실제 사진의 하단 경계"></canvas>
<div class="zoom"><canvas id="zoom" width="1300" height="430" aria-label="두 직선이 만나는 부분 확대"></canvas></div>
<p class="small">주황=왼쪽 선 · 보라=오른쪽 선 · 회색=기존 선 · 초록=SAM 윤곽. 옅은 띠는 기존 선부터 현재 선까지의 이동 범위입니다. 실선은 관측 구간, 점선은 교점까지의 연장입니다. 흰 ×는 교점 후보입니다.</p></div>
<div class="card"><table><thead><tr><th>선 위치</th><th>왼쪽</th><th>오른쪽</th></tr></thead><tbody>
<tr><td>기존 선</td><td>97%</td><td>81%</td></tr><tr><td>1px 아래</td><td>98%</td><td>100%</td></tr>
<tr><td>관측점의 아래쪽 허용선</td><td>98%</td><td>100%</td></tr><tr><td>2px 아래</td><td>84%</td><td>73%</td></tr></tbody></table>
<p class="small">비율은 동일한 표본 위치에서 영상과 마스크의 경계 기준을 모두 통과한 비율입니다. KP의 정답률은 아닙니다. 왼쪽 147개·오른쪽 118개 표본으로 비교했습니다.</p></div>
<p>따라서 이번 예시에서는 기존 선과 약 1.2px 낮춘 선 사이를 허용 띠로 보는 것이 적절합니다. 2px까지 내리면 경계를 지지하는 표본이 줄어듭니다. 선을 내려도 두 선의 각도는 같아서, 교점의 좌우 위치가 민감한 문제는 남습니다.</p>
<p class="small">이번 변경은 선과 허용 범위의 표시입니다. 실제 KP나 자세를 새로 확정하지 않았으며 U 상태는 유지합니다. 표시하신 점·원래 KP3·정답 주석은 이동량 계산에 사용하지 않았습니다.</p>
<details class="card"><summary>고정된 비교 이미지</summary><a href="LOWER_LINES_ACTUAL_KO.png"><img src="LOWER_LINES_ACTUAL_KO.png" alt="실제 사진에서 기존 두 직선과 아래쪽 허용선 비교"></a></details>
<p><a href="LOWER_LINES_ACTUAL_KO.png">비교 이미지</a> · <a href="../kp3_bottom_lower_band_v1/DIAGNOSTIC.json">표본별 계산 요약</a></p>
</main><script>
const data=__DATA__, byName=Object.fromEntries(data.variants.map(v=>[v.name,v]));
const photograph=new Image();let current=byName.OUTER_INLIER_ENVELOPE, mode=current.name;
function intersect(lines){const [a,b]=lines, [nx,ny]=a.normal_unit_xy,[mx,my]=b.normal_unit_xy,d=nx*my-ny*mx;
return [(-a.offset_px*my+ny*b.offset_px)/d,(-nx*b.offset_px+a.offset_px*mx)/d];}
function at(line,t){const [nx,ny]=line.normal_unit_xy;return [-line.offset_px*nx-ny*t,-line.offset_px*ny+nx*t];}
function segment(line,interval){return interval.map(t=>at(line,t));}
function draw(canvas,view){const ctx=canvas.getContext('2d'),[x0,y0,x1,y1]=view,sx=canvas.width/(x1-x0),sy=canvas.height/(y1-y0);
const xy=p=>[(p[0]-x0)*sx,(p[1]-y0)*sy];ctx.clearRect(0,0,canvas.width,canvas.height);ctx.imageSmoothingEnabled=false;
ctx.drawImage(photograph,x0,y0,x1-x0,y1-y0,0,0,canvas.width,canvas.height);
function path(points,color,width,dash=[]){ctx.beginPath();points.map(xy).forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.stroke();ctx.setLineDash([]);}
const old=byName.CENTER;current.lines.forEach((line,i)=>{const interval=data.support_intervals[i],orig=old.lines[i],color=['#ffb531','#d697f3'][i];
const [nx,ny]=line.normal_unit_xy, tCorner=-ny*current.geometry_intersection_only_xy[0]+nx*current.geometry_intersection_only_xy[1];
const extended=[Math.min(interval[0],tCorner)-3,Math.max(interval[1],tCorner)+3],p=segment(line,extended),q=segment(orig,extended);
ctx.beginPath();[...q,...p.slice().reverse()].map(xy).forEach((v,j)=>j?ctx.lineTo(...v):ctx.moveTo(...v));ctx.closePath();ctx.fillStyle=i?'#cb7ee733':'#eda12633';ctx.fill();
if(mode!=='CENTER')path(q,'#d4dbe5',2,[7,7]);path(p,color,3,[8,6]);path(segment(line,interval),color,3.5);});
if(document.getElementById('outline').checked)path(data.contour,'#00c490',2);
const p=xy(current.geometry_intersection_only_xy);ctx.lineWidth=3.5;ctx.strokeStyle='white';ctx.beginPath();ctx.moveTo(p[0]-7,p[1]-7);ctx.lineTo(p[0]+7,p[1]+7);ctx.moveTo(p[0]+7,p[1]-7);ctx.lineTo(p[0]-7,p[1]+7);ctx.stroke();}
function render(){if(!photograph.complete||!photograph.naturalWidth)return;
draw(document.getElementById('wide'),[205,238,502,299]);draw(document.getElementById('zoom'),[343,270,401,289]);
const shifts=current.lines.map(l=>l.positive_image_y_shift_px),p=current.geometry_intersection_only_xy;
const tested=current.lines.every(l=>l.image_support);document.getElementById('status').textContent=`이동 왼쪽 ${shifts[0].toFixed(2)}px / 오른쪽 ${shifts[1].toFixed(2)}px · 교점 후보 (${p[0].toFixed(2)}, ${p[1].toFixed(2)})`+(tested?` · 표본 지지 ${current.lines.map(l=>Math.round(l.image_support.joint_supported_fraction*100)+'%').join(' / ')}`:' · 중간 이동량은 표시 비교용입니다.');
document.querySelectorAll('[data-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.mode===mode)));window.renderedMode=mode;}
function choose(name){mode=name;current=byName[name];const equal=current.lines[0].positive_image_y_shift_px===current.lines[1].positive_image_y_shift_px;
document.getElementById('shiftValue').textContent=equal?current.lines[0].positive_image_y_shift_px.toFixed(1)+'px':'개별 이동';
if(equal)document.getElementById('shift').value=current.lines[0].positive_image_y_shift_px;render();}
document.querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>choose(b.dataset.mode)));
document.getElementById('shift').addEventListener('input',event=>{const dy=Number(event.target.value),named=dy===0?'CENTER':dy===1?'DOWN_1PX':dy===2?'DOWN_2PX':null;
if(named){choose(named);return;}mode='CUSTOM';const lines=byName.CENTER.lines.map(l=>({normal_unit_xy:l.normal_unit_xy,offset_px:l.offset_px-l.normal_unit_xy[1]*dy,positive_image_y_shift_px:dy}));
current={lines,geometry_intersection_only_xy:intersect(lines)};document.getElementById('shiftValue').textContent=dy.toFixed(1)+'px';render();});
document.getElementById('outline').addEventListener('change',render);photograph.onload=render;photograph.src='actual.png';
</script></body></html>'''
    document = document.replace('__DATA__', json.dumps(data, ensure_ascii=False, allow_nan=False))
    (OUT / 'index.html').write_text(document, encoding='utf-8')
    shutil.copyfile(__file__, OUT / 'render_lower_band_case_v1.py')
    C.write(OUT / 'SEAL.json', dict(
        diagnostic_sha256=C.sha(diagnostic_path), original_fit_sha256=C.sha(prior_path),
        image_sha256=C.sha(OUT / 'actual.png'), html_sha256=C.sha(OUT / 'index.html'),
        figure_sha256=C.sha(figure), renderer_sha256=C.sha(__file__),
        default_display_variant=envelope['name'], exact_native_KP_adopted=False,
        state_changed=False, pose_solver_changed=False, GT_used_to_choose_shift=False))


if __name__ == '__main__':
    main()
