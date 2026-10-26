"""Join saved KP classifications and saved segmentation pictures; no model forward."""
from pathlib import Path
import csv, hashlib, html, json, math

W=Path(__file__).absolute().parent
def safe(p):
    p=Path(p)
    assert p.is_absolute() and not any(t in str(p).lower() for t in ('v112','cache','.pyc','..'))
    for a in reversed((p,*p.parents)):assert not a.is_symlink(),str(a)
    return p
def read(rel):return json.loads(safe(W/rel).read_text())
def write(rel,s):
    with safe(W/rel).open('x',encoding='utf-8',newline='\n')as f:f.write(s)
def bind(rel):
    p=safe(W/rel);return dict(file=rel,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
def main():
    points=read('kp_comparison/POINT_MODEL_COMPARISON.json');metrics=read('kp_comparison/METRICS_OVERALL_MATERIAL_NATIVE.json')
    cases=read('segmentation_atlas/CASES.json');assert len(cases)==10 and len(points['rows'])==1784
    source=W.parent/'V270_INITIAL_KP_JOINT_VISIBILITY_QUICKCHECK/evaluation_v4/ALL1784_CLASSIFICATION_COORDINATE_JOIN.csv'
    with safe(source).open()as f:old={int(r['global_index']):r for r in csv.DictReader(f)}
    selected=[]
    for c in cases:
        rr=[dict(r)for r in points['rows']if r['frame_index']==c['frame_index']];assert len(rr)==8
        for r in rr:
            o=old[r['global_index']];ref=None
            if o['manual_D']=='True':
                x,y=float(o['manual_reference_x']),float(o['manual_reference_y'])
                if math.isfinite(x)and math.isfinite(y):ref=[x,y]
            r['manual_D_reference']=ref
        selected.append(dict(c,point_rows=rr))
    names={'CAL_LEGACY':'기존 CAL · 실사 학습','V61_PRESENCE_PROXY':'v61 · KP crop presence','HEATMAP_LOCAL':'V264 · heatmap 주변 특징','FULL_CLIP':'일반 Full CLIP','FULL_CLIP_SURGERY_ARCH':'V267 · CLIP Surgery 구조','EVA_STANDARD':'V269 · 일반 EVA CLIP','EVA_CSA':'V269 · 같은 EVA + CSA','EVA_DECLIP':'V269 · DeCLIP','VIS_ONLY':'V270 · 가림만','JOINT':'V270 · 위치+가림 공동'}
    order=['CAL_LEGACY','V61_PRESENCE_PROXY','HEATMAP_LOCAL','VIS_ONLY','JOINT','FULL_CLIP','FULL_CLIP_SURGERY_ARCH','EVA_STANDARD','EVA_CSA','EVA_DECLIP']
    labels={0:'목재 · 외부 물체 없는 장면',25:'목재 · 기존 query와 실제 모서리 차이',46:'목재 · 앞의 콘 / 분할 방식의 차이',100:'목재 야간 · 콘 가림과 JOINT 누락',105:'정사각형 · 콘은 제외, 보이는 모서리 오판',106:'정사각형 · 보임 점 판정 비교',113:'정사각형 · 외부 가림과 주변 모서리',124:'정사각형 · 이전 사용자 이견이 있는 경계 사례',189:'정사각형 · 작은 대상과 가림',83:'목재 야간 · 작은 팔레트 영역 누락'}
    findings={0:'외부 물체가 없는 장면입니다. JOINT는 KP2/3/6/7을 E로 냅니다. DeCLIP도 KP2/3을 E로 내지만 마스크·기존 query 경계의 문제와 구별해야 합니다.',25:'사람 주석은 native 보임을 뜻합니다. 일부 기존 query는 실제 모서리 밖 배경에 있습니다. 그 query에서 non-target을 낸 CLIP을 곧바로 픽셀 분할 오류라고 부르면 안 됩니다.',46:'Surgery 분할에는 콘 아래 조각이 남고 DeCLIP은 그 조각을 제외합니다. 다만 팔레트의 보이는 모서리까지 마스크 밖에 놓이는 문제가 남습니다.',100:'DeCLIP은 등록된 보임·외부 가림을 이 장면에서 잘 구분합니다. JOINT는 콘에 가린 native0을 D로 냅니다. 같은 장면에서 두 방식의 문제가 다르게 나타납니다.',105:'콘 대부분을 제외하는 팔레트 분할은 가능합니다. 그러나 얇은 팔레트와 모서리 주변 경계가 정확하지 않아 사람 D 주석을 가진 KP1/2를 DeCLIP이 E로 판단합니다.',106:'팔레트의 큰 영역과 개별 모서리의 가림 정답은 다릅니다. 경계가 잘린 점과 native 번호에 의존하는 JOINT 결과를 나란히 보세요.',113:'앞의 물체를 제외한 영역과 점의 위치를 함께 보세요. 한 점의 presence를 native 가림 정답과 같은 뜻으로 해석하지 않습니다.',124:'native4(G996)는 이전 사용자가 실제로 보이는 점처럼 보인다고 지적한 경계 사례입니다. 원래 주석을 보존했으며 확실한 가림 성공·실패 대표로 단정하지 않습니다.',189:'팔레트가 작게 보이는 장면에서는 낮은 해상도 마스크가 모서리까지 보존하지 못할 수 있습니다. 주석 없는 점은 성능에 넣지 않습니다.',83:'이 야간 장면에서 DeCLIP의 등록된 D점이 모두 E로 나옵니다. 실제 마스크가 팔레트 영역을 충분히 남기는지 확인할 수 있습니다.'}
    payload=dict(cases=selected,metrics=metrics['rows'],names=names,order=order,labels=labels,findings=findings)
    data=json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    template='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>팔레트 분할과 KP 가림 판정 전체 비교</title>
<style>body{margin:0;background:#eef2f7;color:#162b43;font:17px/1.65 system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:24px 22px 60px}h1{font-size:30px;line-height:1.4}h2{font-size:23px}section{background:#fff;border-radius:12px;padding:22px;margin:20px 0}.lead{border-left:6px solid #cf7524}nav{display:flex;flex-wrap:wrap;gap:16px}a{color:#135caa}button,select{font:inherit;padding:8px;border:1px solid #9aadc2;border-radius:6px;background:#fff;color:inherit}button{cursor:pointer}label{display:inline-flex;align-items:center;gap:7px}.controls{display:flex;flex-wrap:wrap;align-items:center;gap:12px;margin:12px 0}.controls select{max-width:100%}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.cell{background:#f5f7fa;border:1px solid #d5dfeb;border-radius:7px;overflow:hidden}.cell h3{font-size:16px;margin:8px 12px}.pic{position:relative;line-height:0}.pic img{display:block;width:100%;height:auto}.pic svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.cell p{font-size:14px;padding:0 12px}.note{color:#4c6179;font-size:15px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:15px}th,td{padding:8px;border:1px solid #cbd5e1;text-align:center;min-width:52px}th{background:#eaf0f7}td:first-child,th:first-child{text-align:left;min-width:160px}.yes{background:#daf0e3;color:#145534}.no{background:#fae0e3;color:#842836}.neutral{background:#eef1f6;color:#4c5e72}.focus{outline:3px solid #e6a318;outline-offset:-3px}.state{display:inline-block;min-width:23px;font-weight:700}.wideimg{display:block;width:100%;height:auto}.two{display:grid;grid-template-columns:1fr 1fr;gap:20px}.flow{padding:15px;background:#f0f4f8;border-radius:8px}details{margin:12px 0}summary{cursor:pointer;font-weight:650}.findings{padding:12px;border-left:4px solid #ca8b2c;background:#fff8e9}.stat{font-variant-numeric:tabular-nums}#status{font-size:14px}#point-results button{padding:3px 9px;font-size:15px}.warn{font-weight:700;color:#9b4f20}@media(max-width:950px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}.two{grid-template-columns:1fr}}@media(max-width:520px){main{padding:12px}section{padding:13px}.grid{grid-template-columns:1fr}h1{font-size:25px}}</style>
<main><h1>팔레트 segmentation과 키포인트 보임·가림 판정, 전체 비교</h1>
<section class="lead"><p><strong>안 되는 부분이 서로 다릅니다.</strong> KP 공동 모델은 일부 native 번호를 항상 가림으로 냅니다. CLIP 분할은 팔레트의 큰 영역은 찾지만 경계·작은 대상·가리는 물체를 정확하게 구분하지 못하는 경우가 있습니다. 또 원래 query가 실제 모서리를 벗어나 있으면 분할이 맞아도 native 가림 평가에서는 오답이 될 수 있습니다.</p><p>아래는 <strong>저장된 출력</strong>입니다. 새 학습·새 모델 추론·좌표 보정·PnP·문턱 변경은 없습니다. 실사223장×8점의10방법 결과를 묶고, 기존부터 확인한 대표10장에 실제 전체 이미지 마스크를 붙였습니다. 대표10장은 무작위 표본이나 전체 성공률의 근거가 아닙니다.</p><nav><a href="#masks">실제 팔레트 분할</a><a href="#point-results">각 KP의 정답·오판</a><a href="#all-results">전체·재질별 성능</a><a href="#labels">합성 정답 대조</a><a href="kp_comparison/POINT_MODEL_COMPARISON.html">223장 모든 점 검색</a></nav></section>
<section id="masks"><h2>1. 팔레트만 남긴 영상 · 실제 예측 마스크</h2><div class="controls"><button id="previous" type="button">이전</button><label>장면 <select id="scene"></select></label><button id="next" type="button">다음</button></div><div class="controls"><label>표시 <select id="mode"><option value="cutout_png">팔레트로 분할된 RGB만</option><option value="binary_png">흑백 마스크 자체</option><option value="overlay_png">원본 위 녹색 마스크</option><option value="probability_png">0~1 팔레트 확률 지도</option></select></label><label>비교군 <select id="family"><option value="historical">일반 CLIP / Surgery / DeCLIP</option><option value="matched">동일 EVA / EVA+CSA / DeCLIP</option></select></label></div><div class="controls"><label><input type="checkbox" id="show-query">기존 query 점 표시</label><label><input type="checkbox" id="show-manual">수동 보임 참조 표시</label><label>강조할 KP <select id="kp"><option value="all">전체</option><option>0</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option><option>6</option><option>7</option></select></label></div><div class="grid" id="images"></div><p id="scene-finding" class="findings"></p><p class="note">흰색/초록=모델이 대상 팔레트로 분할한 부분, 검정/회색=나머지. 나머지에는 배경·콘·구멍이 함께 포함되며 외부 가림 물체를 직접 분류한 영역이 아닙니다. 확률 지도는 장면마다 색 범위를 바꾸지 않고0~1입니다. query는 파란 원, 수동 D 참조는 자홍 십자이며 표시만 합니다. 실사 pixel segmentation GT는 없습니다.</p><p id="case-links"></p></section>
<section id="point-results"><h2>2. 위 장면의 각 KP: 어떤 방법이 보임·가림을 냈나</h2><div class="scroll" id="point-table"></div><p class="note">D=보임, E=외부 가림으로 평가한 출력, U=방법의 보류. 초록/빨강 배경은 사람 native D/E와 일치/불일치입니다. <strong>CLIP의 빨간 칸이 모두 픽셀 분할 오류라는 뜻은 아닙니다.</strong> SELF·영상 밖·미주석·대응 불명은 D/E 정답으로 합치지 않습니다. *는 D/E 주석은 있지만 공통 PRIMARY1237점에서는 제외된 행입니다.</p><p class="warn">그림의 query는 이전 저장 좌표입니다. VIS_ONLY/JOINT는 전체 이미지에서 새 native 좌표와 E를 함께 출력한 모델이며 이 query 픽셀을 입력으로 읽은 방식이 아닙니다. 같은 native 번호로 비교했으며 좌표를 새로 재매칭하지 않았습니다.</p></section>
<section><h2>3. 두 방식이 실제로 하는 일</h2><div class="two"><div class="flow"><strong>KP별 직접 판정</strong><p>전체 이미지 또는 KP crop 특징 → 공유 모델 → 각 native 점의 D/E</p><p>V270은 위치·가림 공동 출력입니다. 점별 모델8개를 따로 학습한 것이 아니며 팔레트 분할 마스크도 만들지 않습니다. strict D=0 교사 선택으로 아래쪽2/3/6/7의 보임 학습 표본이0개가 됐고 실사223장 모두 이 네 점을 E로 냈습니다.</p></div><div class="flow"><strong>CLIP 기반 분할 후 점 판정</strong><p>전체 이미지 → CLIP 계열 특징 → 합성 M4로 학습한 분할 head → 기존 query의 target/non-target</p><p>이 그림은 text prompt만 쓴 zero-shot segmentation이 아닙니다. 저장 raw32×32 지도를 원본 크기로 보간했습니다. 큰 영역이 맞아도 얇은 모서리·구멍·가리는 물체가 흐려질 수 있습니다. non-target을 native E와 비교하는 proxy입니다.</p></div></div><p><a href="METHOD_AND_MASK_SEMANTICS_KO.md">방법별 입력·학습 정답·출력 의미·공식 코드와의 실행 범위</a></p></section>
<section id="all-results"><h2>4. 같은 PRIMARY1237점 전체 비교</h2><div class="controls"><label>자료 <select id="group"><option value="ALL">전체 · D1147 / E90</option><option value="WOOD_PRIMARY105">목재105장</option><option value="SQUARE118">정사각형118장</option></select></label><label>KP <select id="metric-slot"><option value="ALL">모든 번호</option><option>0</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option><option>6</option><option>7</option></select></label></div><div class="scroll" id="metric-table"></div><p class="note">각 클래스의 분모가 다릅니다. 보임이 많아서 전체 정확도 하나로 우열을 고르지 않습니다. 기존 CAL은 실사 학습 이력이 있고, 다른 방법들은 서로 다른 입력·교사·백본·학습 예산입니다. 방법 간 역사적 참고 비교이며 통제된 동일 구조 ablation이라고 부르지 않습니다. 자기 가림은 별도입니다.223장은 반복 개발 평가이며 새 독립 평가셋은 아닙니다.</p><p><a href="kp_comparison/METRICS_OVERALL_MATERIAL_NATIVE.csv">전체·재질·native번호별540지표 행</a> · <a href="kp_comparison/POINT_MODEL_COMPARISON.csv">223장×8점 원표</a> · <a href="kp_comparison/POINT_MODEL_COMPARISON.html">1784점 검색·필터</a></p></section>
<section id="labels"><h2>5. 합성에서는 진짜 팔레트 마스크와 직접 비교</h2><p>왼쪽의 M4는 원래 합성 GT입니다. 예측은 판재 틈을 메우거나, 두 번째 장면에서 팔레트 위 상자 영역까지 포함합니다. 이 두 사례에서는 DeCLIP의 영역 IoU가 더 높지만 모서리·구멍·가림 물체까지 정확하다는 뜻은 아닙니다. 아래 두 합성 사례는 전체 실사 성능의 증거가 아닙니다.</p><a href="SYN_GT_VS_PRED.png"><img class="wideimg" loading="lazy" src="SYN_GT_VS_PRED.png" alt="실제 합성 M4 정답과 CLIP,Surgery,DeCLIP 예측 마스크 비교"></a><p><a href="SYN_GT_VS_PRED.json">원본 픽셀 GT·IoU·변환·SHA</a></p></section>
<section><h2>현재 확인된 공통 문제와 고칠 순서</h2><ol><li><strong>학습 감독:</strong> 같은 KP 번호에서 보임·외부 가림이 둘 다 있어야 합니다. 번호만으로 답하는 문제를 먼저 막아야 합니다.</li><li><strong>분할의 세부 정보:</strong>32×32 예측은 큰 영역을 보여주지만 세밀한 visible-pallet mask와 다릅니다. 원래 합성 M4의 모서리·구멍·가림 물체와 예측을 원본 픽셀에서 대조해야 합니다.</li><li><strong>점의 의미와 위치:</strong> query가 배경이면 정확한 분할도 non-target을 냅니다. 해당 native의 가림 여부와 query 픽셀 소속을 별도로 확인해야 합니다.</li></ol><p>이번 시각화는 세 문제가 동시에 존재함을 보여줍니다. 새 모델이나 새 규칙은 적용하지 않았습니다.</p><p><a href="segmentation_atlas/INDEX_KO.html">10장×5모델 분할 원본 전체</a> · <a href="kp_comparison/INDEX_KO.html">같은 장면 KP 번호·결과 패널</a> · <a href="VISUAL_COMPARISON_PLAN_BEFORE_RENDER.json">표시 범위·선정 근거</a> · <a href="ROOT_VISUAL_AND_NUMERIC_QA.json">원표·지도·화면 검증</a></p></section></main>
<script id="saved-data" type="application/json">__DATA__</script><script>
(()=>{'use strict';const D=JSON.parse(document.getElementById('saved-data').textContent),$=id=>document.getElementById(id);const scene=$('scene'),kp=$('kp');
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const asset=s=>'segmentation_atlas/'+s;const pct=x=>x==null?'—':(100*x).toFixed(2)+'%';
D.cases.forEach((c,i)=>{const o=document.createElement('option');o.value=i;o.textContent='F'+String(c.frame_index).padStart(3,'0')+' · '+D.labels[c.frame_index];scene.appendChild(o)});
function marks(c){const q=$('show-query').checked,manual=$('show-manual').checked;if(!q&&!manual)return '';const [h,w]=c.raw_hw,chosen=kp.value;let content='';
for(const r of c.point_rows){if(chosen!=='all'&&String(r.native_slot)!==chosen)continue;const x=r.original_q_x,y=r.original_q_y;if(q&&Number.isFinite(x)&&Number.isFinite(y)&&x>=0&&x<w&&y>=0&&y<h)content+=`<circle cx="${x}" cy="${y}" r="${chosen==='all'?10:18}" fill="none" stroke="${chosen==='all'?'#18c9ff':'#ffe23a'}" stroke-width="6"/><text x="${x+14}" y="${y-13}" font-size="${Math.max(26,w*.031)}" fill="white" stroke="black" stroke-width="4" paint-order="stroke">${r.native_slot}</text>`;
if(manual&&r.manual_D_reference){const [a,b]=r.manual_D_reference;content+=`<path d="M${a-14},${b}H${a+14}M${a},${b-14}V${b+14}" stroke="#ff36c9" stroke-width="6"/>`;}}
return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="xMidYMid meet" aria-hidden="true">${content}</svg>`;}
function renderScene(){const c=D.cases[Number(scene.value)],set=$('family').value==='matched'?['EVA_STANDARD','EVA_CSA','EVA_DECLIP']:['FULL_CLIP','FULL_CLIP_SURGERY_ARCH','EVA_DECLIP'],mode=$('mode').value;
const img=(file,title,sub)=>`<article class="cell"><h3>${esc(title)}</h3><div class="pic"><img src="${asset(file)}" alt="${esc(title)}">${marks(c)}</div><p>${sub}</p></article>`;
$('images').innerHTML=img(c.rgb_display_png,'원본 전체 이미지',esc(c.frame_id))+set.map(m=>img(c.models[m][mode],D.names[m],'target 분할 면적 '+pct(c.models[m].target_fraction)+' · 예측')).join('');
$('scene-finding').textContent=D.findings[c.frame_index];$('case-links').innerHTML=`<a href="${asset(c.comparison_png)}">전체 해상도 분할 비교</a> · <a href="${asset(c.matched_eva_png)}">백본 조건을 맞춘 EVA 비교</a> · <a href="kp_comparison/FRAME_${String(c.frame_index).padStart(3,'0')}_KP_COMPARISON.png">KP 위치와 판정표</a>`;renderPoints(c);}
function renderPoints(c){const selected=kp.value;let s='<table><thead><tr><th>방법</th>'+c.point_rows.map(r=>`<th class="${String(r.native_slot)===selected?'focus':''}"><button type="button" data-kp="${r.native_slot}">KP ${r.native_slot}</button></th>`).join('')+'</tr></thead><tbody><tr><th>사람 native 정답</th>'+c.point_rows.map(r=>`<td>${esc(r.truth)}${!r.primary&&['D','E'].includes(r.truth)?'*':''}</td>`).join('')+'</tr>';
for(const m of D.order){s+=`<tr><th>${esc(D.names[m])}</th>`;for(const r of c.point_rows){const score=r.primary&&['D','E'].includes(r.truth);const cl=score?(r[m]===r.truth?'yes':'no'):'neutral';s+=`<td class="${cl} ${String(r.native_slot)===selected?'focus':''}"><span class="state">${r[m]}</span></td>`;}s+='</tr>';}s+='</tbody></table>';$('point-table').innerHTML=s;document.querySelectorAll('[data-kp]').forEach(b=>b.onclick=()=>{kp.value=b.dataset.kp;$('show-query').checked=true;renderScene()});}
function renderMetrics(){const group=$('group').value,slot=$('metric-slot').value,rr=D.metrics.filter(r=>r.cohort==='PRIMARY_PAIRED1237'&&r.group===group&&String(r.native_slot)===slot);let s='<table><thead><tr><th>방법</th><th>보임 정답 / 전체</th><th>보임→가림 / 보류</th><th>외부 가림 정답 / 전체</th><th>가림→보임 / 보류</th><th>보임 P / R</th><th>가림 P / R</th><th>보류율</th></tr></thead><tbody>';
for(const m of D.order){const r=rr.find(x=>x.method===m);if(!r)continue;s+=`<tr><td>${esc(D.names[m])}</td><td>${r.D_D} / ${r.actual_D}</td><td>${r.D_E} / ${r.D_U}</td><td>${r.E_E} / ${r.actual_E}</td><td>${r.E_D} / ${r.E_U}</td><td>${pct(r.D_precision)} / ${pct(r.D_recall)}</td><td>${pct(r.E_precision)} / ${pct(r.E_recall)}</td><td>${pct(r.U_rate)}</td></tr>`;}s+='</tbody></table>';$('metric-table').innerHTML=s;}
for(const id of ['scene','mode','family','show-query','show-manual','kp'])$(id).addEventListener('change',renderScene);for(const id of ['group','metric-slot'])$(id).addEventListener('change',renderMetrics);$('previous').onclick=()=>{scene.value=(Number(scene.value)+D.cases.length-1)%D.cases.length;renderScene()};$('next').onclick=()=>{scene.value=(Number(scene.value)+1)%D.cases.length;renderScene()};renderScene();renderMetrics();
})();</script></html>'''
    write('ROOT_READOUT_KO.html',template.replace('__DATA__',data))
    summary=dict(PASS=True,completed=True,task='Overall visual comparison of per-KP classifiers and full-image CLIP segmentation',
                 existing_saved_outputs_only=True,new_training=0,new_model_forwards=0,new_PnP=0,threshold_search=0,annotation_changes=0,
                 frames=223,native_rows=1784,methods=10,classification_metric_rows=len(metrics['rows']),real_visual_frames=10,mask_models=5,
                 real_pixel_GT=False,SYN_original_M4_GT_examples=2,primary_N=1237,primary_D=1147,primary_E=90,
                 current223_repeated_development_evaluation=True,conclusion_KO='KP slot shortcut, coarse/incorrect segmentation and query/native semantic or position mismatch are different failures. No new method adopted.',
                 root_readout=bind('ROOT_READOUT_KO.html'),source_tables=[bind('kp_comparison/POINT_MODEL_COMPARISON.json'),bind('kp_comparison/METRICS_OVERALL_MATERIAL_NATIVE.json'),bind('segmentation_atlas/CASES.json')])
    write('FINAL_VISUAL_COMPARISON_SUMMARY.json',json.dumps(summary,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    md='''# V271 KP별 가림 판정과 CLIP 팔레트 분할 전체 시각 비교

저장 출력만으로 실사223장×8점의10가지 방법을 묶었습니다. 새 학습·모델 추론·PnP·문턱 변경은 없습니다. 같은 대표10장의 실제 팔레트 분할 마스크와 분할된 RGB, 각 native KP의 D/E/U를 볼 수 있습니다.

**문제는 세 가지입니다:** KP 공동 모델의 번호 사전값 의존, CLIP 마스크의 거친 경계·가림 물체 오포함, 기존 query와 사람 native 가림 정답의 위치·의미 차이. 빨간 CLIP 판정을 모두 픽셀 분할 실패로 해석하면 안 됩니다. real pixel GT는 없고 합성 M4 GT와는 별도로 직접 대조했습니다.

![원본과 Surgery·DeCLIP이 팔레트로 분할한 RGB](segmentation_atlas/SEGMENTATION_OVERVIEW.png)

- [장면·모델·마스크 종류·KP 번호를 바꿔 보는 전체 페이지](ROOT_READOUT_KO.html)
- [흑백 팔레트 마스크 자체](segmentation_atlas/SEGMENTATION_MASK_OVERVIEW.png)
- [10장×5분할 모델 전체](segmentation_atlas/INDEX_KO.html)
- [같은 장면 KP 위치와 정답·오판](kp_comparison/INDEX_KO.html)
- [223장1784점10방법 전체 검색](kp_comparison/POINT_MODEL_COMPARISON.html)
- [재질·번호별540행 지표](kp_comparison/METRICS_OVERALL_MATERIAL_NATIVE.csv)
- [실제 합성 M4 GT와 예측 비교](SYN_GT_VS_PRED.png)
- [방법별 학습·입력·출력 의미](METHOD_AND_MASK_SEMANTICS_KO.md)
- [시각·수치 검산](ROOT_VISUAL_AND_NUMERIC_QA.json)

![외부 물체 없는 목재의 KP별 결과](kp_comparison/FRAME_000_KP_COMPARISON.png)

대표10장은 무작위 성능 표본이 아닙니다. 수치는 고정 PRIMARY1237(D1147/E90)에서 집계하며 자기 가림·영상 밖·미주석·대응 불명은 별도입니다. 기존 CAL은 실사 학습 이력이 있고 각 방법은 입력·교사·백본·예산이 달라 역사적 참고 비교입니다. 이번 그림의 마스크는 전체 이미지를 사용한 CLIP 계열 특징+합성 mask head의 저장 결과이며 text-only zero-shot 분할은 실행하지 않았습니다.223장은 반복 개발 평가입니다.
'''
    write('PUBLICATION_README_KO.md',md)
    print(json.dumps(dict(PASS=True,visual_frames=len(cases),point_rows=len(points['rows']),metric_rows=len(metrics['rows']),new_model_forward=0),ensure_ascii=False))
if __name__=='__main__':main()
