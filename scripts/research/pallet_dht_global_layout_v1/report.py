"""Offline report of actual saved whole-layout selection; no inference or GT edits."""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
AUDIT=ROOT/'data/pallet/results/pallet_dht_gt_audit_v1'
PROBE=ROOT/'data/pallet/results/pallet_dht_decoder_probe_v1'
TITLE='Global Layout · 8점·12선 공동 선택'
CASE='eval_pallet07:1778652166837872128'
ARMS=('baseline','independent','global','global_shared_only')
LABELS=dict(baseline='Frozen baseline',independent='Independent corners',
            global_='Global layout',global_shared_only='Shared lines · no DLT')
LABELS['global']=LABELS.pop('global_')
EDGES=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
ROLES=('front_top_width','front_right_height','front_bottom_width','front_left_height',
       'rear_top_width','rear_right_height','rear_bottom_width','rear_left_height',
       'left_top_depth','right_top_depth','right_bottom_depth','left_bottom_depth')
ASSETS=Path.home()/'.claude/agents/viz-expert/assets'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


class Inputs:
    def __init__(self):self.hashes={}
    def bind(self,path,expected=None):
        path=Path(path).resolve();digest=sha(path)
        if expected is not None:assert digest==expected,f'Input SHA mismatch: {path}'
        self.hashes[str(path)]=digest;return path
    def read(self,path,expected=None):return json.loads(self.bind(path,expected).read_text())
    def verify(self):
        for p,s in self.hashes.items():assert sha(p)==s,f'Input changed during rendering: {p}'


def colors(inputs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.style.use(inputs.bind(ASSETS/'analysis.mplstyle'))
    spec=importlib.util.spec_from_file_location('global_layout_palette',inputs.bind(ASSETS/'palette.py'))
    p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
    return dict(gt=p.color_for('gt'),halo=p.CV['text'],baseline=p.color_for('baseline'),
                independent=p.color_for('before'),global_=p.color_for('after'),
                alternative=p.color_for('pred'),evidence=p.color_for('pred'))


def gt_context(inputs,frame_ids):
    """Join qualitative review records without turning them into a GT whitelist."""
    canonical=inputs.read(ROOT/'challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json')
    contexts={}
    for item in canonical['items']:
        key=item['frame_id']
        if key not in frame_ids:continue
        path=inputs.bind(ROOT/item['gt_v2_path']);obj=inputs.read(path)['objects'][0]
        a=obj['keypoint_annotations'];assert len(a)==9
        contexts[key]=dict(annotation=str(path),annotation_sha256=sha(path),
            points=[k['xy'] for k in a],visibility=[k['visibility'] for k in a],
            sources=[k.get('source','unknown') for k in a],reasons=[k.get('reason','unknown') for k in a],
            in_frame=[k.get('in_frame') for k in a],reviews=[[] for _ in range(9)],queue=[])
    assert set(contexts)==set(frame_ids)
    specialist=inputs.read(AUDIT/'specialist_gt_review.json')
    assert specialist['complete'] and specialist['gt_modified'] is False
    for frame in specialist['reviewed_frames']:
        if frame['frame_id'] not in contexts:continue
        context=contexts[frame['frame_id']];assert frame['gt_sha256']==context['annotation_sha256']
        for point in frame['points']:
            index=point['index'];assert point['xy']==context['points'][index]
            context['reviews'][index].append(dict(reviewer='specialist',review_class=point['review_class'],
                note=frame.get('observation_ko'),pixel_accuracy_certified=False))
    review=inputs.read(AUDIT/'ROOT_GT_VISUAL_REVIEW.json')
    assert review['complete'] and review['all_319_gt_certified'] is False
    for row in review['records']:
        if row['id'] not in contexts:continue
        context=contexts[row['id']];assert row['annotation_sha256']==context['annotation_sha256']
        context['reviews'][row['point']].append(dict(reviewer='root',review_class=row['selected_point_assessment'],
            note=row.get('note'),pixel_accuracy_certified=False))
    queue=inputs.read(AUDIT/'GT_REVIEW_QUEUE.json');assert queue['canonical_GT_edited'] is False
    for row in queue['items']:
        if row['frame_id'] in contexts:contexts[row['frame_id']]['queue'].append(row)
    return contexts


def number(value):
    return '—' if value is None else f'{float(value):.2f}'


def table(headers,rows):
    cell=lambda v:html.escape(str(v))
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+cell(v)+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+cell(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def metric_table(rows):
    return table(['Population','Method','Frames','Observed / GT','Median px','P90 px','Mean px','PCK10 %'],
        [[r.get('population','real_dev subset'),LABELS[r['arm']],r['n_frames'],str(r['n_observed_points'])+' / '+str(r['n_gt_points']),
          number(r.get('median_px')),number(r.get('p90_px')),number(r.get('mean_px')),
          number(None if r.get('pck10_observed') is None else 100*r['pck10_observed'])] for r in rows])


def cohort_html(cohorts):
    blocks=[]
    for key,c in cohorts.items():
        blocks.append('<details><summary>'+html.escape(key)+'</summary><p>'+html.escape(c['description'])+'</p>'+metric_table(c['summaries'])+'</details>')
    return ''.join(blocks)


def evidence_view(evidence):
    """Keep actual top-mode geometry; never invent hard line choices for mixtures."""
    if evidence is None:return None
    keys=('candidate_xy','candidate_valid','candidate_sources','candidate_source_slot','line_peaks_h_raw','line_peak_valid',
          'line_peak_theta_rho','line_peak_log_probability','edges','supporting_modes','score_definition',
          'lines','line_weights','available_line_roles','explicit_seed_hypotheses')
    out={key:evidence[key] for key in keys if key in evidence}
    if 'line_peaks_h_raw' in out:
        h=np.asarray(out['line_peaks_h_raw']);valid=np.asarray(out['line_peak_valid'])
        assert h.ndim==3 and h.shape[0]==12 and h.shape[-1]==3 and valid.shape==h.shape[:2]
        assert np.isfinite(h).all()
    return out


def build(run_dir,inputs):
    done=inputs.read(run_dir/'EVALUATION_COMPLETION.json')
    assert done.get('complete') and done.get('PASS'),'Actual completed evaluation required'
    for p,s in done.get('output_sha256',{}).items():inputs.bind(Path(p) if Path(p).is_absolute() else run_dir/p,s)
    results=inputs.read(run_dir/'RESULTS.json');pred=inputs.read(run_dir/'PREDICTIONS.json',done['predictions_sha256'])
    metric=inputs.read(run_dir/'FRAME_METRICS.json');protocol=inputs.read(run_dir/'PROTOCOL.json',done['protocol_sha256'])
    selection=inputs.read(run_dir/'CALIBRATION_SELECTION.json',done['calibration_selection_sha256'])
    assert pred.get('complete') and pred.get('PASS') and results.get('complete')
    assert pred['protocol_sha256']==sha(run_dir/'PROTOCOL.json')
    assert pred['calibration_selection_sha256']==sha(run_dir/'CALIBRATION_SELECTION.json')
    assert protocol['real_gt_tuning'] is False and protocol['gt_modifications']==0
    actual=[f for f in pred['records'] if f['population']=='real_dev']
    assert len(actual)==319 and len({f['id'] for f in actual})==319 and CASE in {f['id'] for f in actual}
    metrics={r['id']:r for r in metric['records'] if r['population']=='real_dev'}
    assert set(metrics)=={f['id'] for f in actual}
    context=gt_context(inputs,set(metrics));frames=[]
    for row in actual:
        g=context[row['id']];m=metrics[row['id']]
        assert g['points']==m['gt_points'] and [v>0 for v in g['visibility']]==m['gt_supervised']
        p=inputs.bind(row['image'],row['image_sha256'])
        for arm in ARMS[1:]:
            assert row['arms'][arm]['point_valid']==row['baseline']['point_valid']
            assert row['arms'][arm]['points'][8]==row['baseline']['points'][8]
        frames.append(dict(id=row['id'],session=row['session_id'],width=row['width'],height=row['height'],
            image='data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode(),image_path=str(p),image_sha256=row['image_sha256'],
            baseline=row['baseline'],arms=row['arms'],evidence=evidence_view(row.get('evidence')),gt=g,
            metrics=m['arms'],view=m.get('view',{}),layout_diagnostic=m.get('layout_diagnostic'),
            candidate_diagnostic=m.get('candidate_diagnostic')))
    baseline=next(r for r in results['summaries'] if r['population']=='real_dev' and r['arm']=='baseline')
    assert baseline['n_frames']==319 and baseline['n_observed_points']==2738
    assert abs(baseline['p90_px']-protocol['evaluation']['baseline_p90_px'])<1e-10
    return frames,results,protocol,selection,pred.get('score_definition',{})


def render(run_dir):
    run_dir=Path(run_dir).resolve();assert (run_dir/'PURPOSE.md').is_file()
    inputs=Inputs();inputs.bind(__file__);frames,results,protocol,selection,score_definition=build(run_dir,inputs)
    data=dict(frames=frames,colors=colors(inputs),edges=EDGES,roles=ROLES,labels=LABELS,required_case=CASE,
              selection=selection,score_definition=score_definition)
    paired=table(['Population','Comparison','Frames','Sessions','Mean frame Δ px','Session 95% CI'],
        [[r.get('population'),str(r.get('left'))+' − '+str(r.get('right')),r.get('n_frames'),r.get('n_sessions'),
          number(r.get('mean_frame_delta_px')),str(r.get('ci95'))] for r in results.get('paired',[])])
    difficulty=table(['Population','Method','Baseline error bin','Points','Before mean px','After mean px','Mean Δ px','≤10 → >10 %'],
        [[r.get('population'),LABELS[r['arm']],r.get('difficulty'),r.get('n_points'),number(r.get('baseline_mean_px')),
          number(r.get('after_mean_px')),number(r.get('mean_delta_px')),number(None if r.get('easy_to_over10_rate') is None else 100*r['easy_to_over10_rate'])] for r in results.get('difficulty',[])])
    real={r['arm']:r for r in results['summaries'] if r['population']=='real_dev'}
    signal=results.get('continuation',{}).get('continuation_signal',False)
    easy=next(r for r in results['difficulty'] if r['population']=='real_dev' and r['arm']=='global' and r['difficulty']=='easy_le10')
    crossed=round(easy['n_points']*easy['easy_to_over10_rate'])
    outcome='전체 배치 선택은 기존 예측보다 악화했습니다.' if real['global']['median_px']>real['baseline']['median_px'] and real['global']['p90_px']>real['baseline']['p90_px'] else '전체 배치 선택의 실제 결과입니다.'
    headline=f"{outcome} 실사 P90 {real['baseline']['p90_px']:.2f} → {real['global']['p90_px']:.2f}px. 원래 오류 ≤10px였던 점 {crossed}/{easy['n_points']}개가 10px 밖으로 이동했습니다. 사전 진행 기준은 {'충족' if signal else '미충족'}했습니다."
    breakdown=''.join('<details><summary>'+html.escape(label)+'</summary>'+metric_table([dict(population=group,**row) for group,rows in results.get(key,{}).items() for row in rows])+'</details>' for key,label in [('by_session','촬영 세션별 실제 결과'),('by_image_resolution','원래 영상 해상도별 실제 결과')])
    view_path=run_dir/'VIEW_DIAGNOSTIC.json'
    if view_path.is_file():
        view=inputs.read(view_path);assert view['complete'] and view['PASS'] and view['used_for_selection'] is False
        breakdown+='<details id="viewdiagnostic"><summary>기존 관점 그룹의 사후 설명용 결과</summary><p>'+html.escape(view['warning_ko'])+'</p>'+metric_table([dict(population=g['dimension']+' / '+g['group'],**r) for g in view['groups'] for r in g['summaries']])+'</details>'
    calibration_note=''
    calibration_path=run_dir/'CALIBRATION_BASELINE_DIAGNOSTIC.json'
    if calibration_path.is_file():
        calibration=inputs.read(calibration_path)
        assert calibration['complete'] and calibration['audit_integrity_PASS'] and calibration['selection_changed'] is False
        identity=calibration['baseline_unconditional_identity']['mean_normalized_corner_error']
        selected=calibration['selected_config']['mean_normalized_corner_error']
        ratio=calibration['selected_config']['ratio_to_unconditional_identity']
        calibration_note=f'<p class="notice" id="calibrationfinding">합성 calibration에서도 개선이 확인된 것은 아닙니다. 같은 {calibration["n_frames"]}장·{calibration["n_observed_corners"]}개 코너의 영상 대각선 정규화 평균 오차는 원래 점 유지 {identity:.6f} → 선택 설정 {selected:.6f} ({ratio:.2f}배)였습니다. 9가지 가중치 중 최소 오차를 골랐지만 모두 원래 점 유지보다 나빴습니다. 각 후보 풀에 원래 배치는 있었으나, 항상 원래 점을 반환하는 정책은 9개 설정에 포함되지 않았습니다. 이는 선택을 바꾸지 않은 <a href="CALIBRATION_BASELINE_DIAGNOSTIC.json">사후 기준선 대조</a>입니다.</p>'
    behavior=table(['Method','Frames','Unchanged','Input fallback','Invalid baseline available','Invalid baseline selected'],
        [[LABELS[arm],r.get('n_frames'),r.get('n_unchanged_frames'),r.get('n_input_fallback_frames'),r.get('n_baseline_invalid_geometry_exception_available'),r.get('n_baseline_invalid_geometry_exception_selected')] for arm,r in results.get('selection_behavior',{}).items()])
    text=TEMPLATE
    values=dict(TITLE=TITLE,HEADLINE=html.escape(headline),METRICS=metric_table(results['summaries']),PAIRED=paired,
        DIFFICULTY=difficulty,COHORTS=cohort_html(results.get('cohorts',{})),BREAKDOWN=breakdown,BEHAVIOR=behavior,
        SELECTION=html.escape(json.dumps(selection,ensure_ascii=False,indent=2)),CALIBRATION_NOTE=calibration_note,
        VERDICT=html.escape(json.dumps(results.get('continuation',results.get('verdict',{})),ensure_ascii=False,indent=2)),
        DATA=json.dumps(data,ensure_ascii=False,allow_nan=False).replace('</','<\\/'))
    for key,value in values.items():text=text.replace('@@'+key+'@@',value)
    assert '@@' not in text and '<script src=' not in text
    page=run_dir/'index.html';page.write_text(text);inputs.verify()
    receipt=dict(complete=True,PASS=True,title=TITLE,html=str(page),html_sha256=sha(page),n_real_frames=319,
        n_displayed_methods=4,experiment_complete=True,input_sha256=inputs.hashes,
        scope='Saved-output rendering only. Approximate global search, not a proven global optimum.',
        no_inference=True,no_gt_edits=True,no_desktop_open=True,no_notification=True)
    write(run_dir/'REPORT_RENDER.json',receipt);print(json.dumps({k:receipt[k] for k in ('complete','html','html_sha256')}))
    return receipt


TEMPLATE='''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>@@TITLE@@</title><style>
:root{font:15px Arial,sans-serif;color:#e7edf5;background:#0c1a27;color-scheme:dark}*{box-sizing:border-box}main{max-width:1540px;margin:auto;padding:22px}h1{font-size:28px}h2{font-size:21px;font-weight:normal}p{line-height:1.7}section,.notice{padding:18px;margin:18px 0;background:#162b3e;border:1px solid #465e72;border-radius:8px}.notice{border-left:4px solid #9bd3e8}.muted,small{color:#b9cad7;font-size:13px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:9px;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{font-weight:normal}thead{border-top:1px solid #60798c;border-bottom:1px solid #60798c}tbody{border-bottom:1px solid #60798c}details{margin:16px 0}summary{cursor:pointer}pre{white-space:pre-wrap;max-height:450px;overflow:auto;font-size:12px}.controls{display:flex;flex-wrap:wrap;gap:12px;margin:14px 0}select,button{background:#294459;color:inherit;padding:8px;border:1px solid #708da0;border-radius:4px}label{font-size:12px}select{display:block;max-width:380px;margin-top:5px}.panels{display:grid;grid-template-columns:1fr 1fr;gap:12px}.panel{min-width:0;background:#0e1e2c;padding:10px}.panel h3{font-size:14px;font-weight:normal}canvas{display:block;width:100%;height:auto}.zoom .panels{grid-template-columns:1fr}a{color:#a0daf0}.legend{font-size:13px;line-height:1.9}@media(max-width:800px){.panels{grid-template-columns:1fr}main{padding:10px}}
</style></head><body><main><h1>@@TITLE@@</h1><p class="notice">@@HEADLINE@@</p><p>기존 hough_joint_seed1 EMA의 원래 점 출력과 저장된 선으로 8개 꼭짓점과 12개 연결선을 함께 선택한 실험입니다. Independent는 새 학습 모듈이 아니라 이번 동일 16후보의 독립 선택입니다. 이전 decoder probe의 학습된 point_only/line_fusion과 구분합니다. 같은 후보 풀을 쓴 독립점 선택과 비교하며, 전체 배치 점수는 beam 64·고정 순서 4개의 근사 탐색으로 최소화합니다. 전역 최적해를 증명한 것은 아닙니다. 추가 CNN 학습·추론은 없고 실사 GT로 가중치를 조정하지 않았습니다.</p><section id="metrics"><h2>원래 번호로 평가한 실제 2D 결과</h2><p class="muted">Same semantic IDs · 319 reused DEV images / 13 sessions · 2,738 observed points. 기준 P90 41.4873px를 그대로 유지합니다. 중심점 ID8과 결측 마스크는 복사합니다. 결과 완성과 향상 판정은 구분하며 독립 FINAL이나 반복 seed 검증은 아닙니다.</p>@@METRICS@@<details><summary>합성 calibration 선택과 실제 조건</summary><p>동결 모델이 이미 학습에 사용한 합성 훈련 풀에서 256장을 고른 calibration입니다. 독립 테스트 성능으로 해석하지 않습니다. Shared-only 대조군은 projective consistency 가중치를 0으로 두지만 DLT 퇴화·깊이 수치검사는 유지합니다. 원래 baseline은 퇴화해도 identity fallback으로 남기며 이 경우 geometry cost 0은 유효한 배치라는 뜻이 아닙니다.</p>@@CALIBRATION_NOTE@@<pre>@@SELECTION@@</pre></details><details><summary>대응 세션 bootstrap와 사전 진행 기준</summary>@@PAIRED@@<pre>@@VERDICT@@</pre></details></section><section><h2>큰 오류와 원래 양호한 점</h2><p class="muted">GT 오류로 나눈 사후 난이도입니다. 원래 오류 ≤10px 점이 >10px로 넘어가는 비율과 >20px 점의 변화를 함께 확인합니다. 이 그룹은 선택 입력이 아닙니다.</p>@@DIFFICULTY@@<details><summary>GT 출처·가시성·정성 검토별 설명용 점 집합</summary><p>직접 클릭 기록·visible 표기·정성적으로 외곽 부근이라는 검토는 각각 다른 정보이며 픽셀 정확도 인증이 아닙니다. 검토가 없는 점을 정확하다고 가정하지 않습니다. 부분집합으로 공식 분모를 교체하지 않습니다.</p>@@COHORTS@@</details>@@BREAKDOWN@@<details><summary>실제 배치 유지·fallback 비율</summary>@@BEHAVIOR@@</details></section><section id="gallery"><h2>전체 배치와 실제 선택 점수</h2><p class="muted">기본은 사용자가 지적한 원래 사례입니다. 대안 배치 순서는 실제 저장된 선택 점수 순서이며, GT로 좋은 후보를 고르지 않습니다. 도움/악화 정렬만 GT를 본 사후 탐색입니다.</p><div class="controls"><label>Image<select id="frame"></select></label><label>Sort<select id="rank"><option value="case">Required case / original order</option><option value="best">Most improved · GT diagnostic</option><option value="worst">Most harmed · GT diagnostic</option></select></label><label>Joint score<select id="arm"><option value="global">Global · shared lines + DLT</option><option value="global_shared_only">Shared lines only · DLT zero</option></select></label><label>Role<select id="role"></select></label><label>Line mode<select id="mode"><option value="all">All Top-4 modes</option><option value="0">Mode 1</option><option value="1">Mode 2</option><option value="2">Mode 3</option><option value="3">Mode 4</option></select></label><label>Alternative saved layout<select id="alternative"></select></label><label>GT point<select id="point"></select></label><label><input id="showgt" type="checkbox" checked>GT IDs / visibility</label><label><input id="showevidence" type="checkbox" checked>Role top-mode lines</label><label><input id="showmovement" type="checkbox" checked>Baseline → selected</label><button id="yaw90">Inspect explicit yaw90</button><button id="raw">Clean raw / overlays</button><button id="zoom">Larger panels</button></div><p class="notice" id="frameinfo"></p><p class="legend" id="legend"></p><div class="panels"><div class="panel"><h3>① Frozen hough_joint seed1 EMA · original point IDs</h3><canvas id="baseline"></canvas></div><div class="panel"><h3>② Independent corner selection · same candidate pool</h3><canvas id="independent"></canvas></div><div class="panel"><h3 id="selectedtitle">③ Global selected layout · all 12 connections</h3><canvas id="selected"></canvas></div><div class="panel"><h3 id="alternativetitle">④ Runner-up global layout</h3><canvas id="runnerup"></canvas></div></div><p id="evidenceinfo" class="muted"></p><p id="casefinding" class="notice"></p><div id="scores"></div><p class="muted">표는 실제 저장된 점수의 분해입니다. 선 항은 저장된 Top-4 역할 선 안에서 재정규화한 혼합 가중치로 두 끝점을 함께 설명하는 비용이며, 표시된 Top-4 선 하나를 반드시 hard 선택했다는 뜻이 아닙니다. 서로 다른 후보 index가 동일 좌표 배치를 만들 수도 있습니다. score gap 0이 서로 다른 배치의 불확실성을 뜻한다고 단정하지 않습니다. 가중치·점수 gap은 정답 확률이나 attention·인과 설명이 아닙니다. DLT 항은 사영 구조 일관성이고 독립 6D 정확도 검증은 아닙니다.</p><details><summary>선택한 GT 점의 출처와 이전 정성 검토</summary><p id="gtinfo"></p><pre id="reviewinfo"></pre></details></section><section><h2>해석과 보존 범위</h2><p>모든 배치는 원래 점 번호로 채점합니다. 점 위치 후보에서 번호를 재배치한 효과, 선 교점의 위치 보정, C4 회전 배치 후보의 효과는 실제 기록된 후보 출처가 있는 경우에만 구분합니다. GT 순열 oracle는 실제 선택으로 표시하지 않습니다. 화면 밖·가려진 GT 및 검토 대상을 이유로 기존 결과를 자동 제외하지 않습니다.</p><p><a href="RESULTS.json">RESULTS</a> · <a href="PREDICTIONS.json">Saved layouts</a> · <a href="FRAME_METRICS.json">Same-ID frame errors</a> · <a href="CALIBRATION_SELECTION.json">Synthetic calibration</a> · <a href="PROTOCOL.json">Fixed protocol</a> · <a href="REPORT_RENDER.json">Report bindings</a></p></section></main><script id="report-data" type="application/json">@@DATA@@</script><script>
const DATA=JSON.parse(document.getElementById('report-data').textContent),$=id=>document.getElementById(id),c=DATA.colors,images=new Map();let frames=[],token=0,clean=false;
function option(el,value,label){const o=document.createElement('option');o.value=value;o.textContent=label;el.append(o)}
function segment(x,a,b,color,width=1,dashed=false){if(!a||!b)return;x.save();x.strokeStyle=color;x.lineWidth=width;x.setLineDash(dashed?[7,5]:[]);x.beginPath();x.moveTo(...a);x.lineTo(...b);x.stroke();x.restore()}
function labels(x,points,color){if(!points)return;x.font='12px Arial';points.forEach((p,i)=>{if(!p)return;x.beginPath();x.arc(...p,4,0,7);x.strokeStyle=c.halo;x.lineWidth=5;x.stroke();x.strokeStyle=color;x.lineWidth=2;x.stroke();x.strokeStyle=c.halo;x.lineWidth=3;x.strokeText(String(i),p[0]+6,p[1]-5);x.fillStyle=color;x.fillText(String(i),p[0]+6,p[1]-5)})}
function visiblePoints(row){return row.points.map((p,i)=>row.point_valid[i]?p:null)}
function layout(x,p,color){if(!p)return;DATA.edges.forEach(([a,b])=>segment(x,p[a],p[b],color,1.5));labels(x,p,color)}
function gt(x,f){x.font='12px Arial';f.gt.points.forEach((p,i)=>{const v=f.gt.visibility[i];if(v===0)return;const [a,b]=p;x.beginPath();if(v===2)x.arc(a,b,4,0,7);else{x.moveTo(a,b-5);x.lineTo(a+5,b);x.lineTo(a,b+5);x.lineTo(a-5,b);x.closePath()}x.strokeStyle=c.halo;x.lineWidth=6;x.stroke();x.strokeStyle=c.gt;x.lineWidth=2;x.stroke();x.strokeStyle=c.halo;x.lineWidth=4;x.strokeText(String(i),a+6,b+13);x.fillStyle=c.gt;x.fillText(String(i),a+6,b+13)})}
function infinite(x,h,w,height,color){const [nx,ny,k]=h,p=[];if(Math.abs(ny)>1e-9)for(const a of [0,w]){const b=-(nx*a+k)/ny;if(b>=0&&b<=height)p.push([a,b])}if(Math.abs(nx)>1e-9)for(const b of [0,height]){const a=-(ny*b+k)/nx;if(a>=0&&a<=w)p.push([a,b])}if(p.length>=2)segment(x,p[0],p[1],color,1.5,true)}
function mean(f,arm){const e=f.metrics[arm].errors_px.filter(v=>v!=null);return e.length?e.reduce((a,b)=>a+b)/e.length:null}
function delta(f){const a=mean(f,$('arm').value),b=mean(f,'baseline');return a==null||b==null?null:a-b}
function reset(){frames=DATA.frames.slice();const mode=$('rank').value;if(mode!=='case')frames=frames.filter(f=>delta(f)!=null);frames.sort((a,b)=>mode==='best'?delta(a)-delta(b):mode==='worst'?delta(b)-delta(a):a.id===DATA.required_case?-1:b.id===DATA.required_case?1:DATA.frames.indexOf(a)-DATA.frames.indexOf(b));$('frame').replaceChildren();frames.forEach((f,i)=>option($('frame'),i,f.id));render(true)}
function optionsFor(row){return [...(row.top_hypotheses||[]).map((h,i)=>({key:'top:'+i,label:'Saved rank '+(i+1),hypothesis:h,diagnostic:'top'+(i+1)})),...Object.entries(row.explicit_hypotheses||{}).map(([name,h])=>({key:'explicit:'+name,label:'Explicit '+name,hypothesis:h,diagnostic:'explicit:'+name}))]}
function scores(f,arm,alternative){const root=$('scores');root.replaceChildren();const ranked=f.arms[arm].top_hypotheses||[],rows=ranked.slice();if(alternative?.key.startsWith('explicit:'))rows.push(alternative.hypothesis);const table=document.createElement('table'),head=document.createElement('thead'),tr=document.createElement('tr');['Saved rank','Total cost','Shared weighted','Point weighted','DLT weighted','Geometry check','Origin','Candidate indices (IDs 0–7)'].forEach(s=>{const th=document.createElement('th');th.textContent=s;tr.append(th)});head.append(tr);table.append(head);const body=document.createElement('tbody');rows.forEach((r,i)=>{const tr=document.createElement('tr');[i<ranked.length?i+1:alternative.label,r.total,r.weighted_terms?.shared_line,r.weighted_terms?.point_prior,r.weighted_terms?.geometry,r.geometry_valid?'valid':r.baseline_invalid_geometry_zero_cost_exception?'identity exception':r.geometry_reason,r.kind??r.origin??'not recorded',JSON.stringify(r.indices)].forEach(v=>{const td=document.createElement('td');td.textContent=typeof v==='number'?v.toFixed(5):v??'—';tr.append(td)});body.append(tr)});table.append(body);const wrap=document.createElement('div');wrap.className='scroll';wrap.append(table);root.append(wrap);const p=document.createElement('p');p.className='muted';const row=f.arms[arm];p.textContent='Weights: '+JSON.stringify(rows[0]?.weights??{})+' | fallback: '+(row.fallback_reason??'none')+' | baseline geometry exception selected: '+Boolean(row.baseline_exception_selected)+'. Raw independent_line is diagnostic and is not an additional global-score term.';root.append(p);const d=document.createElement('details'),label=document.createElement('summary'),pre=document.createElement('pre');label.textContent='Actual raw score components / search provenance';pre.textContent=JSON.stringify({definition:DATA.score_definition,top3:rows,explicit_whole_layout_seeds:row.explicit_hypotheses},null,2);d.append(label,pre);root.append(d);const gd=document.createElement('details'),gs=document.createElement('summary'),gp=document.createElement('p'),gt=document.createElement('table'),gh=document.createElement('thead'),gb=document.createElement('tbody');gs.textContent='GT-only posthoc diagnostics · not used for selection';gp.textContent='Nearest candidate errors do not imply that eight independently nearest points form a valid whole layout. Explicit C4 seeds reassign existing predicted locations without correcting them geometrically.';const hr=document.createElement('tr');['Saved hypothesis','Cost','Same-ID mean px','Same-ID median px','Origin'].forEach(s=>{const th=document.createElement('th');th.textContent=s;hr.append(th)});gh.append(hr);(f.layout_diagnostic?.[arm]||[]).forEach(r=>{const tr=document.createElement('tr');[r.name,r.total,r.observed_mean_px,r.observed_median_px,r.kind].forEach(v=>{const td=document.createElement('td');td.textContent=typeof v==='number'?v.toFixed(3):v??'—';tr.append(td)});gb.append(tr)});gt.append(gh,gb);const cp=document.createElement('pre');cp.textContent=JSON.stringify(f.candidate_diagnostic,null,2);gd.append(gs,gp,gt,cp);root.append(gd)}
async function render(resetAlternative=false){window.REPORT_READY=false;const current=++token,f=frames[Number($('frame').value)||0];if(!f)return;const arm=$('arm').value,row=f.arms[arm],ranked=row.top_hypotheses||[],alternatives=optionsFor(row);if(resetAlternative||$('alternative').options.length!==alternatives.length){$('alternative').replaceChildren();alternatives.forEach(r=>option($('alternative'),r.key,r.label+' · '+(r.hypothesis.total==null?'ineligible':'cost '+Number(r.hypothesis.total).toFixed(4))));if(ranked.length>1)$('alternative').value='top:1'}const altEntry=alternatives.find(r=>r.key===$('alternative').value)||alternatives[0],alt=altEntry?.hypothesis;let im=images.get(f.id);if(!im){im=new Image();im.src=f.image;await im.decode();images.set(f.id,im)}if(current!==token)return;const role=Number($('role').value);
for(const id of ['baseline','independent','selected','runnerup']){const cv=$(id);cv.width=f.width;cv.height=f.height;const x=cv.getContext('2d');x.drawImage(im,0,0);if(clean)continue;if($('showgt').checked)gt(x,f);if(id==='baseline')layout(x,visiblePoints(f.baseline),c.baseline);if(id==='independent')layout(x,visiblePoints(f.arms.independent),c.independent);if(id==='selected'){layout(x,visiblePoints(row),c.global_);if($('showmovement').checked)row.points.forEach((p,i)=>{if(row.point_valid[i])segment(x,f.baseline.points[i],p,c.evidence,1)})}if(id==='runnerup'&&alt)layout(x,alt.points_xy.map((p,i)=>row.point_valid[i]?p:null),c.alternative);if((id==='selected'||id==='runnerup')&&$('showevidence').checked&&f.evidence)(f.evidence.line_peaks_h_raw?.[role]||[]).forEach((h,i)=>{if(f.evidence.line_peak_valid?.[role]?.[i]&&($('mode').value==='all'||Number($('mode').value)===i))infinite(x,h,f.width,f.height,c.evidence)})}
const fmt=v=>v==null?'not observed':v.toFixed(2)+' px';$('frameinfo').textContent=f.id+' | '+f.session+' | '+f.width+'×'+f.height+' | mean baseline '+fmt(mean(f,'baseline'))+' → independent '+fmt(mean(f,'independent'))+' → '+arm+' '+fmt(mean(f,arm));$('selectedtitle').textContent='③ '+DATA.labels[arm]+' · selected whole layout';$('alternativetitle').textContent='④ '+(alt?altEntry.label+' · alternative whole layout':'No alternative layout saved');$('evidenceinfo').textContent='Role '+role+' '+DATA.roles[role]+' · endpoints '+DATA.edges[role].join('–')+' | dashed lines: actual saved top modes, not observed physical-edge guarantees. '+(ranked.length>1?'Saved rank1/rank2 cost gap '+(ranked[1].total-ranked[0].total).toFixed(6):'Runner-up unavailable.')+' Selected hypothesis index: '+row.selected_hypothesis_index+' | weights renormalized within retained Top-4 modes: '+JSON.stringify(f.evidence?.line_weights?.[role]||[]);scores(f,arm,altEntry);const yaw=(f.layout_diagnostic?.[arm]||[]).find(r=>r.name==='explicit:baseline_yaw90'),top=(f.layout_diagnostic?.[arm]||[]).find(r=>r.name==='top1'),altDiag=(f.layout_diagnostic?.[arm]||[]).find(r=>r.name===altEntry?.diagnostic);$('casefinding').textContent=f.id===DATA.required_case&&yaw&&top?'이 사례의 고정 yaw90 후보는 존재합니다. 사후 같은-ID median은 '+Number(yaw.observed_median_px).toFixed(2)+'px이지만 실제 비용 '+Number(yaw.total).toFixed(3)+'으로, 선택 배치의 비용 '+Number(top.total).toFixed(3)+'보다 높아 선택되지 않았습니다. 선택 배치의 사후 median은 '+Number(top.observed_median_px).toFixed(2)+'px입니다. Alternative saved layout의 explicit yaw90은 실제 저장된 후보이며 GT로 재선택한 출력이 아닙니다.':altDiag?'Displayed hypothesis: same-ID median '+(altDiag.observed_median_px==null?'not observed':Number(altDiag.observed_median_px).toFixed(2)+' px')+' (posthoc GT diagnostic; not used for selection).':'No GT diagnostic for this hypothesis.';const p=Number($('point').value);$('gtinfo').textContent='ID '+p+' | v='+f.gt.visibility[p]+' | source='+f.gt.sources[p]+' | reason='+f.gt.reasons[p]+' | in_frame='+f.gt.in_frame[p]+'. Pixel accuracy is not certified.';$('reviewinfo').textContent=JSON.stringify({point_reviews:f.gt.reviews[p].length?f.gt.reviews[p]:'not_reviewed',frame_review_queue:f.gt.queue},null,2);window.REPORT_CURRENT_FRAME=f.id;window.REPORT_READY=true;}
DATA.roles.forEach((s,i)=>option($('role'),i,i+' · '+s));$('role').value='7';for(let i=0;i<9;i++)option($('point'),i,i===8?'8 · centroid':i);$('point').value='4';for(const [key,label] of [['gt','GT (white halo; diamond = v1)'],['baseline','Baseline'],['independent','Independent'],['global_','Selected global'],['alternative','Alternative / dashed line evidence']]){const span=document.createElement('span');span.style.marginRight='18px';const sw=document.createElement('span');sw.style.cssText='display:inline-block;width:14px;height:9px;margin-right:5px;background:'+c[key]+';border:1px solid '+c.halo;span.append(sw,document.createTextNode(label));$('legend').append(span)}['rank','arm'].forEach(k=>$(k).onchange=reset);$('frame').onchange=()=>render(true);['alternative','role','mode','point','showgt','showevidence','showmovement'].forEach(k=>$(k).onchange=()=>render());$('yaw90').onclick=()=>{if(optionsFor(frames[Number($('frame').value)||0].arms[$('arm').value]).some(o=>o.key==='explicit:baseline_yaw90')){$('alternative').value='explicit:baseline_yaw90';render()}};$('raw').onclick=()=>{clean=!clean;render()};$('zoom').onclick=()=>$('gallery').classList.toggle('zoom');window.REPORT_FRAME_COUNT=DATA.frames.length;reset();
</script></body></html>'''


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    render(parser.parse_args().run_dir)
