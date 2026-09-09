"""Offline, saved-output-only report for the explicit decoder pilot."""
from __future__ import annotations

import argparse
import base64
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

from scripts.research.pallet_dht_decoder_probe_v1 import evaluate as E
from scripts.research.pallet_line_pose_v1 import report as H
from scripts.research.pallet_dht_joint_v1.line_targets import EDGES, ROLE_NAMES

TITLE='Point–Line Decoder Probe · 점·선 후보 결합'
LABELS=dict(baseline='Frozen joint · baseline',point_only='Learned point control',line_fusion='Learned point + line',wrong_image_line='Wrong-image line ablation')
ASSETS=Path.home()/'.claude/agents/viz-expert/assets'


def style(inputs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.style.use(inputs.bind(ASSETS/'analysis.mplstyle'))
    spec=importlib.util.spec_from_file_location('decoder_palette',inputs.bind(ASSETS/'palette.py'))
    palette=importlib.util.module_from_spec(spec);spec.loader.exec_module(palette)
    colors={k:palette.color_for(v) for k,v in dict(gt='gt',baseline='baseline',control='before',line='after',candidate='pred',movement='pred').items()}
    colors['gt_halo']=palette.CV['text']
    return colors


def n(value,unit=''):
    return H.number(value,unit)


def table_summary(results):
    return H.table(['Population','Model','Frames','Observed / GT','Median (px)','P90 (px)','Mean (px)','PCK10 (%)','>50 px (%)'],
        [[r['population'],LABELS[r['arm']],r['n_frames'],f"{r['n_observed_points']} / {r['n_gt_points']}",n(r['median_px']),n(r['p90_px']),n(r['mean_px']),n(None if r['pck10_observed'] is None else 100*r['pck10_observed']),n(None if r['above50_fraction'] is None else 100*r['above50_fraction'])] for r in results['summaries']])


def render(run_dir):
    run_dir=Path(run_dir).resolve();inputs=E.Inputs()
    done=inputs.read(run_dir/'EVALUATION_COMPLETION.json')
    E.require(done.get('complete') and done.get('PASS'),'Completed actual decoder evaluation is required')
    for path,digest in done['output_sha256'].items():inputs.bind(run_dir/path,digest)
    for path,digest in done['input_sha256'].items():inputs.bind(path,digest)
    results=inputs.read(run_dir/'RESULTS.json');predictions=inputs.read(run_dir/'PREDICTIONS.json')
    metrics=inputs.read(run_dir/'FRAME_METRICS.json');protocol=inputs.read(run_dir/'TRAIN_PROTOCOL.json')
    E.require(results['complete'] and results['n_new_training_seeds']==1,'Actual single-training-seed pilot required')
    per_id={r['id']:r for r in metrics['records']};frames=[]
    for row in predictions['records']:
        if row['population']!='real_dev':continue
        m=per_id[row['id']];path=inputs.bind(row['image'],row['image_sha256'])
        bgr=cv2.imread(str(path));E.require(bgr is not None,'Original image decode failed')
        E.require(list(bgr.shape[:2])==[row['height'],row['width']],'Original image shape differs')
        frames.append(dict(id=row['id'],session=row['session_id'],width=row['width'],height=row['height'],
            image=H.data_uri(bgr),gt=[p if v else None for p,v in zip(m['gt_points'],m['gt_supervised'])],
            baseline=row['baseline'],arms=row['arms'],geometry=row.get('geometry'),metrics=m['arms'],view=m.get('view',{})))
    E.require(len(frames)==319 and len({f['id'] for f in frames})==319 and E.CASE in {f['id'] for f in frames},'Real319/problem case missing')
    colors=style(inputs)
    data=dict(frames=frames,colors=colors,roles=list(ROLE_NAMES),edges=[list(e) for e in EDGES],required_case=E.CASE,labels=LABELS)
    pairtable=H.table(['Population','Comparison','Common frames','Sessions','Mean frame delta (px)','Session 95% CI (px)'],
        [[r['population'],f"{r['left']} − {r['right']}",r['n_frames'],r['n_sessions'],n(r['mean_frame_delta_px']),f"[{n(r['ci95'][0])}, {n(r['ci95'][1])}]"] for r in results['paired']])
    difficult=H.table(['Population','Model','Original GT error bin','Points','Frames','Before mean (px)','After mean (px)','Mean delta (px)','Easy → >10px (%)'],
        [[r['population'],LABELS[r['arm']],r['difficulty'],r['n_points'],r['n_frames'],n(r['baseline_mean_px']),n(r['after_mean_px']),n(r['mean_delta_px']),n(None if r['easy_to_over10_rate'] is None else 100*r['easy_to_over10_rate'])] for r in results['difficulty']])
    views=H.table(['View axis','Stratum','Model','Frames','Median (px)','P90 (px)','Mean (px)'],
        [[r['axis'],r['group'],LABELS[r['arm']],r['n_frames'],n(r['median_px']),n(r['p90_px']),n(r['mean_px'])] for r in results.get('views',[])])
    diagnostic_path=run_dir/'DIAGNOSTIC.json'
    diagnostic=inputs.read(diagnostic_path) if diagnostic_path.is_file() else None
    diagnostic_text=('기존 joint 3 seed의 고정 14장 무학습 후보 진단은 별도 파일입니다. 이는 새 결합부를 3 seed로 학습한 결과가 아닙니다. GT oracle는 배포할 수 없는 사후 상한입니다.' if diagnostic else '별도 무학습 진단은 이 보고서 입력에 없습니다.')
    if diagnostic:
        E.require(diagnostic['complete'] and diagnostic['PASS'] and diagnostic['selection_uses_gt'] is False,'Actual GT-free diagnostic required')
        drows=[]
        for seed,entry in diagnostic['per_seed'].items():
            for method,value in entry['nine_points']['metrics'].items():
                drows.append([seed,method,entry['nine_points']['n_frames'],value['point_count'],n(value['median_px']),n(value['p90_px']),n(value['mean_px'])])
        dt=H.table(['Existing backbone seed','Method','Fixed frames','Observed points','Median (px)','P90 (px)','Mean (px)'],drows)
        case_errors=[r['evaluation']['oracle']['errors_px'][0] for r in diagnostic['records'] if r['frame_id']==E.CASE]
        diagnostic_text+=f" 문제 사례의 원래 corner0은 GT oracle로 후보를 골라도 {min(case_errors):.1f}–{max(case_errors):.1f}px 오류가 남습니다. 모든 점의 정답 후보가 확보된 것은 아닙니다.</p><details><summary>고정14장 · 기존3 seed · GT oracle 별도 진단</summary>{dt}<p>Oracle는 원래 ID별 GT에 가장 가까운 후보를 사후 선택한 상한입니다. 학습결합부 또는 GT 없는 고정 선택의 성능과 동일시하지 않습니다.</p></details><p>"
    payload=json.dumps(data,ensure_ascii=False,allow_nan=False).replace('</','<\\/')
    replacements=dict(TITLE=TITLE,HEADLINE=results['continuation']['headline_ko'],RESULTS=table_summary(results),PAIRED=pairtable,
        DIFFICULTY=difficult,VIEWS=views,DIAGNOSTIC=diagnostic_text,CRITERIA=json.dumps(results['continuation'],ensure_ascii=False,indent=2),
        PROTOCOL=json.dumps(protocol,ensure_ascii=False,indent=2),DATA=payload)
    text=TEMPLATE
    for key,value in replacements.items():text=text.replace('@@'+key+'@@',value)
    E.require('@@' not in text and '<script src=' not in text,'Unresolved/external report source')
    page=run_dir/'index.html';page.write_text(text)
    inputs.bind(Path(__file__));inputs.bind(Path(H.__file__));inputs.bind(Path(E.__file__))
    inputs.verify()
    receipt=dict(complete=True,PASS=True,experiment_complete=True,title=TITLE,html=str(page),html_sha256=E.sha(page),
        input_sha256=inputs.hashes,n_real_frames=319,n_new_training_seeds=1,
        scope='Saved-output rendering only; browser and independent coordinate QA are separate.',no_inference=True,no_auto_open=True)
    E.write(run_dir/'REPORT_RENDER.json',receipt)
    print(json.dumps({k:receipt[k] for k in ('complete','html','html_sha256')},ensure_ascii=False))
    return receipt


TEMPLATE='''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>@@TITLE@@</title><style>
:root{font:15px Arial,sans-serif;color:#e4edf5;background:#0c1825;color-scheme:dark}*{box-sizing:border-box}main{max-width:1550px;margin:auto;padding:24px}h1{font-size:29px}h2{font-size:21px}p{line-height:1.7}section{padding:20px;background:#132639;border:1px solid #344b61;border-radius:10px;margin:20px 0}.notice{padding:18px;background:#25394d;border-left:4px solid #e9b16c}.muted,small{color:#adc0cf}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:10px 8px;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}thead{border-top:1px solid #526679;border-bottom:1px solid #526679}tbody{border-bottom:1px solid #526679}th{font-weight:normal}details{margin:16px 0}summary{cursor:pointer;color:#9ad5ed}pre{white-space:pre-wrap;max-height:400px;overflow:auto}.controls{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}select,button{background:#203951;color:inherit;padding:8px;border:1px solid #6b8195;border-radius:5px}label{font-size:12px}select{display:block;max-width:360px;margin-top:4px}.panels{display:grid;grid-template-columns:1fr 1fr;gap:12px}.panel{min-width:0;padding:12px;background:#0b1927}canvas{display:block;width:100%;height:auto}.panel h3{font-size:14px;font-weight:normal}.zoom .panels{grid-template-columns:1fr}a{color:#9ad5ed}.legend{line-height:1.8;font-size:13px}@media(max-width:750px){.panels{grid-template-columns:1fr}main{padding:10px}}
</style></head><body><main><h1>@@TITLE@@</h1><p class="notice">@@HEADLINE@@</p><p>기존 joint seed1을 고정하고 작은 결합부만 합성 데이터로 학습한 단일 seed 탐색 실험입니다. 점 전용 대조군과 실제 선 후보 결합을 같은 예산으로 비교합니다. 실사 319장·13세션은 재사용 DEV이며 독립 final 또는 안정적 성능 향상을 입증하지 않습니다.</p>
<section><h2>원래 점 번호로 평가한 실제 결과</h2><p class="muted">Original semantic IDs · supervised nine points · same baseline IoU matching. 결측·미매칭은 coverage에 남기며, 표의 PCK는 관측된 점 기준입니다. 전체 GT 분모의 PCK도 RESULTS.json에 저장합니다.</p>@@RESULTS@@<details><summary>탐색적 후속 진행 기준</summary><pre>@@CRITERIA@@</pre></details><details><summary>대응 세션 bootstrap · 20,000 draws</summary><p>같은 원본 프레임의 평균 점오차 차이를 비교합니다. 13개 재사용 촬영 세션에 조건부인 95% 구간이며 한 번 학습한 결합부의 탐색 결과입니다. Wrong-image line은 선 의존성 진단이지만 입력 분포 변화도 함께 일어납니다.</p>@@PAIRED@@</details></section>
<section><h2>원래 양호한 점과 큰 오류</h2><p class="muted">GT 오류로 나눈 사후 점 그룹입니다. 이 난이도 구분은 후보 선택이나 모델 입력에 사용하지 않습니다. Centroid ID8을 포함한 공식 9점과 모서리 8점의 별도 결과는 JSON에 보존합니다.</p>@@DIFFICULTY@@<details><summary>미리 고정한 관점별 진단</summary><p>재구성 앙각·2D 면적비는 설명용 구도 그룹이며 실제 yaw·가시성 실측 또는 추가 개선 판정이 아닙니다.</p>@@VIEWS@@</details></section>
<section id="gallery"><h2>원본 점·선·교점 후보를 직접 확인</h2><p class="muted">기본은 사용자가 지적한 점 번호 오류 사례입니다. 원본좌표와 번호를 그대로 유지합니다. 도움/악화 정렬은 GT를 본 사후 탐색이며 자동 선택 규칙이 아닙니다.</p><div class="controls"><label>Image<select id="frame"></select></label><label>Sort<select id="rank"><option value="case">Required case / original order</option><option value="best">Most improved (GT diagnostic)</option><option value="worst">Most harmed (GT diagnostic)</option></select></label><label>Line input<select id="arm"><option value="line_fusion">Actual image lines</option><option value="wrong_image_line">Wrong-image lines</option></select></label><label>Corner ID<select id="corner"></select></label><label><input type="checkbox" id="showgt" checked>GT overlay</label><label><input type="checkbox" id="showcandidates" checked>Candidate lines / intersections</label><button id="zoom">2× view</button></div><div id="frameinfo" class="notice"></div><p class="legend" id="legend"></p><div class="panels"><div class="panel"><h3>① Original + GT + frozen baseline IDs</h3><canvas id="baseline"></canvas></div><div class="panel"><h3>② Learned point control</h3><canvas id="control"></canvas></div><div class="panel"><h3>③ Learned line fusion + same-ID displacement</h3><canvas id="fusion"></canvas></div><div class="panel"><h3>④ Actual candidate lines / intersections</h3><canvas id="candidates"></canvas></div></div><p id="candidateinfo" class="muted"></p><p class="muted">교점은 영상에서 예측한 역할 선으로 생성했습니다. 표시된 후보 가중치·gate는 실제 결합부 출력이며 attention·인과 설명·보정된 정답 확률이 아닙니다. Signed gate는 음수도 가능하므로 후보 평균의 반대 방향으로 움직일 수 있고, 별도 residual은 원본 대각선의 0.05배 tanh 범위입니다. 총 이동량에 같은 한도를 적용한 모델은 아닙니다. GT는 그림과 평가에만 사용합니다.</p></section>
<section><h2>무학습 진단과 한계</h2><p>@@DIAGNOSTIC@@</p><p>이번 단계는 2D 점 위치에 한정합니다. 6D 성능, negative 검출 변화, 전체 시스템 지연은 검증하지 않았습니다. GT oracle가 일부 점의 더 가까운 후보를 고를 수 있어도, 모든 점의 정답 후보가 존재하거나 모델이 GT 없이 올바른 후보를 선택했다는 의미는 아닙니다.</p><details><summary>고정 protocol</summary><pre>@@PROTOCOL@@</pre></details><p><a href="RESULTS.json">RESULTS</a> · <a href="PREDICTIONS.json">Actual predictions</a> · <a href="FRAME_METRICS.json">Frame metrics</a> · <a href="DIAGNOSTIC.json">Separate no-training diagnostic</a> · <a href="EVALUATION_COMPLETION.json">Evaluation bindings</a></p></section>
</main><script id="report-data" type="application/json">@@DATA@@</script><script>
const DATA=JSON.parse(document.getElementById('report-data').textContent),$=id=>document.getElementById(id);let available=[],token=0;const images=new Map();
const c=DATA.colors;function option(el,v,t){const o=document.createElement('option');o.value=v;o.textContent=t;el.appendChild(o)}
function points(ctx,p,color,numbers=true){if(!p)return;ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=2;ctx.font='12px Arial';p.forEach((q,i)=>{if(!q)return;ctx.beginPath();ctx.arc(q[0],q[1],4,0,7);ctx.stroke();if(numbers)ctx.fillText(String(i),q[0]+5,q[1]-5)});}
function segment(ctx,a,b,color,width=1){if(!a||!b)return;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke()}
function structure(ctx,p,color){if(!p)return;DATA.edges.forEach(([a,b])=>segment(ctx,p[a],p[b],color));points(ctx,p,color)}
function groundTruth(ctx,p){if(!p)return;ctx.save();DATA.edges.forEach(([a,b])=>segment(ctx,p[a],p[b],c.gt_halo,4.5));ctx.strokeStyle=c.gt_halo;ctx.lineWidth=5;ctx.lineJoin='round';ctx.font='12px Arial';p.forEach((q,i)=>{if(!q)return;ctx.beginPath();ctx.arc(q[0],q[1],4,0,7);ctx.stroke();ctx.strokeText(String(i),q[0]+5,q[1]-5)});ctx.restore();structure(ctx,p,c.gt)}
function infinite(ctx,h,w,height,color){const [nx,ny,k]=h,p=[];if(Math.abs(ny)>1e-9){for(const x of [0,w]){const y=-(nx*x+k)/ny;if(y>=0&&y<=height)p.push([x,y])}}if(Math.abs(nx)>1e-9){for(const y of [0,height]){const x=-(ny*y+k)/nx;if(x>=0&&x<=w)p.push([x,y])}}if(p.length>=2)segment(ctx,p[0],p[1],color,1.5)}
function vals(row,arm){return arm==='baseline'?row.baseline:row.arms[arm]}
function mean(row,arm){const a=row.metrics[arm].errors_px.filter(x=>x!=null);return a.length?a.reduce((x,y)=>x+y)/a.length:null}
function delta(row){const a=mean(row,$('arm').value),b=mean(row,'baseline');return a==null||b==null?null:a-b}
function reset(){available=DATA.frames.slice();const rank=$('rank').value;if(rank!=='case')available=available.filter(x=>delta(x)!=null);available.sort((a,b)=>rank==='best'?delta(a)-delta(b):rank==='worst'?delta(b)-delta(a):a.id===DATA.required_case?-1:b.id===DATA.required_case?1:a.id.localeCompare(b.id));$('frame').replaceChildren();available.forEach((f,i)=>option($('frame'),i,f.id));render()}
async function render(){window.REPORT_READY=false;const current=++token,f=available[Number($('frame').value)||0];if(!f){window.REPORT_READY=true;return}let im=images.get(f.id);if(!im){im=new Image();im.src=f.image;await im.decode();images.set(f.id,im)}if(current!==token)return;const chosen=$('arm').value,actual=vals(f,chosen),baseline=f.baseline.points,corner=Number($('corner').value),geo=actual.geometry||f.geometry,decoder=actual.decoder||{};
for(const id of ['baseline','control','fusion','candidates']){const cv=$(id);cv.width=f.width;cv.height=f.height;const x=cv.getContext('2d');x.drawImage(im,0,0,f.width,f.height);if($('showgt').checked)groundTruth(x,f.gt);if(id==='baseline')structure(x,baseline,c.baseline);if(id==='control')structure(x,f.arms.point_only.points,c.control);if(id==='fusion'){structure(x,actual.points,c.line);if(baseline&&actual.points)actual.points.forEach((q,i)=>segment(x,baseline[i],q,c.movement));}if(id==='candidates'){if($('showcandidates').checked&&geo){const roles=geo.incident_roles?.[corner]||[];roles.forEach(r=>(geo.line_peaks_h_raw?.[r]||[]).forEach((h,j)=>{if(geo.line_peak_valid?.[r]?.[j])infinite(x,h,f.width,f.height,c.candidate)}));const weights=decoder.candidate_weights?.[corner]||[];(geo.candidates_xy?.[corner]||[]).forEach((q,j)=>{if(!geo.candidate_valid?.[corner]?.[j])return;x.fillStyle=c.candidate;x.globalAlpha=.3+.7*(weights[j]||0);x.beginPath();x.arc(q[0],q[1],2+8*(weights[j]||0),0,7);x.fill()});x.globalAlpha=1;}if(baseline)points(x,[baseline[corner]],c.baseline,false);if(actual.points)points(x,[actual.points[corner]],c.line,false);}}
const show=v=>v==null?'missing':v.toFixed(2)+' px';$('frameinfo').textContent=f.id+' | '+f.session+' | original '+f.width+'×'+f.height+' | mean: baseline '+show(mean(f,'baseline'))+' → point control '+show(mean(f,'point_only'))+' → '+chosen+' '+show(mean(f,chosen));$('candidateinfo').textContent='Corner '+corner+' | roles '+((geo?.incident_roles?.[corner]||[]).map(r=>r+':'+DATA.roles[r]).join(', '))+' | candidate geometry '+(geo?'saved from actual input':'unavailable')+' | gate '+(decoder.gate?.[corner]??'not saved')+' | donor '+(actual.donor_id??'same image')+'. Candidate slot order is fixed without GT; centroid ID8 is separate.';window.REPORT_CURRENT_FRAME=f.id;window.REPORT_READY=true;}
for(let i=0;i<8;i++)option($('corner'),i,String(i));$('corner').value='4';for(const [key,label] of [['gt','GT (white halo)'],['baseline','Baseline'],['control','Point control'],['line','Line fusion'],['candidate','Candidates / displacement']]){const span=document.createElement('span'),swatch=document.createElement('span');swatch.style.cssText='display:inline-block;width:16px;height:10px;margin:0 6px 0 12px;background:'+c[key]+';border:1px solid '+c.gt_halo;span.append(swatch,document.createTextNode(label));$('legend').append(span)}['rank','arm'].forEach(k=>$(k).addEventListener('change',reset));['frame','corner','showgt','showcandidates'].forEach(k=>$(k).addEventListener('change',render));$('zoom').onclick=()=>$('gallery').classList.toggle('zoom');window.REPORT_FRAME_COUNT=DATA.frames.length;reset();
</script></body></html>'''


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    render(parser.parse_args().run_dir)
