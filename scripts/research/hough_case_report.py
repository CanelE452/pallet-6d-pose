"""Render the frozen one-frame Hough recheck; no inference or desktop actions."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TITLE = "Hough · 필터 오통과 사례 재검증"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(run_dir):
    run_dir = Path(run_dir).resolve()
    sources = {}

    def read(name):
        path = run_dir / name
        sources[str(path)] = sha(path)
        return json.loads(path.read_text())

    result = read("RESULTS.json")
    provenance = read("provenance/provenance.json")
    geometry = read("GEOMETRY_AUDIT.json")
    read("provenance/CURRENT_LEARNED_CASE.json")
    read("PROTOCOL.json")
    assert all(x.get("complete") and x.get("PASS") for x in [result, provenance, geometry])
    assert result["shape_hw"] == [480, 640] and result["no_gt_in_hough_or_fusion"]
    assert len(result["fusion"]) == 4 and len(result["learned"]) == 3
    assert result["baseline"]["indexed"]["n"] == 8

    def embedded(path):
        path = Path(path)
        sources[str(path)] = sha(path)
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()

    assets = {"raw": embedded(run_dir / "raw.png"),
              "canny": embedded(run_dir / "canny.png"),
              "reference": embedded(ROOT / provenance["screenshot"])}
    names = {"baseline": "원본 YOLO26n R0 / M4", "long_side8": "고전 Hough long · 측면 8선",
             "long_cuboid12": "고전 Hough long · 전체 12선", "short_side8": "고전 Hough short · 측면 8선",
             "short_cuboid12": "고전 Hough short · 전체 12선", **{f"learned_seed{s}": f"기존 학습형 image_joint · seed {s}" for s in [1, 2, 3]}}
    rows = [{"id": "baseline", **result["baseline"]}] + result["fusion"] + result["learned"]
    metric_rows = "".join(f'<tr><td>{names[r["id"]]}</td><td>{r["indexed"]["median_px"]:.2f}</td><td>{r["indexed"]["mean_px"]:.2f}</td><td>{r["indexed"]["max_px"]:.2f}</td><td>{r["indexed"]["median_px"]-result["baseline"]["indexed"]["median_px"]:+.2f}</td></tr>' for r in rows)
    oracle_rows = "".join(f'<tr><td>{names[r["id"]]}</td><td>{r["oracle_reindexed"]["median_px"]:.2f}</td><td>{r["oracle_reindexed"]["max_px"]:.2f}</td></tr>' for r in rows)
    support_rows = "".join(f'<tr><td>{name}</td><td>{x["count"]}</td><td>{x["settings"]["minLineLength"]}px</td><td>{x["gt_oracle_matched_roles"]}/{x["gt_oracle_supported_roles"]}</td><td>{100*x["baseline_evidence"]["oriented_hough_support_fraction_mean"]:.1f}%</td></tr>' for name,x in result["hough"].items())
    f = provenance["filter"]
    m4 = provenance["M4_record"]
    filter_rows = "".join(f'<tr><td>{label}</td><td>{value}</td><td>{criterion}</td><td>PASS</td></tr>' for label,value,criterion in [
        ("Box confidence", f'{m4["box_conf"]:.5f}', f'≥ {f["box_confidence_minimum"]}'),
        ("Valid corners", str(m4["valid_corners"]), f'≥ {f["minimum_valid_corners"]}'),
        ("Remove consistency", f'{m4["s_remove"]:.5f}', f'≤ {f["s_remove_maximum"]}'),
        ("Flip consistency", f'{m4["s_flip"]:.5f}', f'≤ {f["s_flip_maximum"]}')])
    payload = json.dumps({"result": result, "assets": assets, "names": names}, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    document = TEMPLATE.replace("@@TITLE@@", TITLE).replace("@@DATA@@", payload)
    document = document.replace("@@METRICS@@", metric_rows).replace("@@ORACLE_ROWS@@", oracle_rows)
    document = document.replace("@@SUPPORT@@", support_rows).replace("@@FILTER@@", filter_rows)
    document = document.replace("@@REPROJ@@", f'{m4["s_reproj"]:.5f}')
    assert "@@" not in document
    assert '<script src=' not in document and '<link ' not in document
    output = run_dir / "index.html"
    output.write_text(document)
    sources[str(Path(__file__).resolve())] = sha(__file__)
    receipt = {"schema": "hough_case_report_render_v1", "complete": True, "PASS": True,
               "title": TITLE, "html": str(output), "html_sha256": sha(output),
               "input_sha256": sources, "n_embedded_images": 3, "external_dependencies": 0,
               "n_classical_settings": 4, "n_learned_seeds": 3,
               "new_inference": False, "new_training": False, "opened": False,
               "scope": "Renders actual frozen RESULTS; browser/visual QA is recorded separately."}
    (run_dir / "REPORT_RENDER.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ["complete", "PASS", "html", "html_sha256"]}))


TEMPLATE = r'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>@@TITLE@@</title>
<style>
:root{color-scheme:dark;--bg:#07131e;--panel:#102131;--border:#30485e;--text:#e5edf5;--muted:#adc0d0;--gt:#63e6a0;--pred:#5cd9ef;--new:#f68cdb;--line:#ffb454}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:16px/1.65 system-ui,"Noto Sans CJK KR",sans-serif}main{max-width:1480px;margin:auto;padding:32px}h1{font-size:32px;line-height:1.3;margin:12px 0}h2{font-size:23px;margin:0 0 12px}h3{font-size:17px;margin:0 0 10px}p{margin:10px 0;color:var(--muted)}section,details.card{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:22px;margin:20px 0}header{margin-bottom:22px}.tag{font-size:13px;color:var(--line)}.verdict{border-left:4px solid var(--line);padding:16px 20px;background:#202d3b;color:var(--text)}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:18px 0}.stat{background:var(--panel);border:1px solid var(--border);border-radius:9px;padding:15px 20px}.stat strong{display:block;font-size:27px}.stat span{font-size:13px;color:var(--muted)}.tablewrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px 12px;text-align:right;border-bottom:1px solid var(--border);white-space:nowrap}th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:500}.small,small{font-size:13px;color:var(--muted)}.controls{display:flex;gap:12px;align-items:end;flex-wrap:wrap;margin:14px 0}.controls label{font-size:13px;color:var(--muted)}select,button{background:#071724;color:var(--text);border:1px solid #4c667f;border-radius:6px;padding:9px 11px;font:inherit;cursor:pointer}select{display:block;max-width:100%}input{accent-color:#76c2ff}button:hover{background:#244258}.checks{display:flex;gap:20px;flex-wrap:wrap}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.panel{background:#091724;border:1px solid var(--border);padding:14px;border-radius:8px;min-width:0}.panel canvas{display:block;width:100%;height:auto;background:#111}.canvaswrap{overflow:auto;max-height:1000px}.zoomed canvas{width:1280px;max-width:none}.axis{display:flex;justify-content:space-between;color:var(--muted);font:12px/1.6 monospace;margin-top:5px}.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:13px;margin:12px 0}.legend b{font-weight:500}.info{padding:12px 16px;border-left:3px solid #80c4ff;background:#213446;margin:12px 0}.reference{display:block;max-width:100%;height:auto;margin:14px auto}.oracle{border:2px solid var(--line)!important}.oracle summary{color:#ffd193;font-weight:600;cursor:pointer}.twocol{display:grid;grid-template-columns:1fr 1fr;gap:24px}a{color:#91cbff}.pill{display:inline-block;background:#253f55;padding:2px 8px;border-radius:8px;font-size:12px}summary{cursor:pointer}.scrollnote{margin-top:8px}.fullscreen{background:var(--bg);overflow:auto;padding:22px}.fullscreen .canvaswrap{max-height:none}.fullscreen canvas{width:min(100%,1280px)}@media(max-width:900px){main{padding:14px}.grid,.twocol{grid-template-columns:1fr}.stats{gap:6px}.stat{padding:12px}.stat strong{font-size:21px}h1{font-size:26px}}
</style></head><body><main>
<header><span class="tag">실제 원본 한 장 · 고정 설정 · 새 학습/모델 추론 없음</span><h1>@@TITLE@@</h1><p>eval_pallet07:1778652166837872128 · 원본 640 × 480 px</p>
<div class="verdict"><strong>일부 실제 경계는 잡지만, 이번 점·선 결합은 잘못 붙은 점 번호를 해결하지 못했습니다.</strong><br>고전 Hough 4설정과 기존 학습형 3개 seed 모두 원래 번호 기준 큰 오차가 남습니다. 번호 대응 오류와 수십 픽셀의 위치 오차가 함께 있습니다.</div></header>
<div class="stats"><div class="stat"><strong>267.93 px</strong><span>원본 점 오차 중앙값 · 최대 315.69 px</span></div><div class="stat"><strong>36 / 146</strong><span>long / short Hough 유한 선분 후보</span></div><div class="stat"><strong>267.92–269.40 px</strong><span>고전 점·선 보정 4설정의 중앙값</span></div></div>
<section><h2>자동 보정 결과를 먼저 비교합니다</h2><p>첨부 그림과 같은 감독 마스크: visibility &gt; 0인 <strong>7개 코너 + 중심점(8), 총 8점</strong>입니다. 코너 3은 영상 밖에 있어 평가에서 제외됩니다. 기존 변수 이름이 corner여도 이 수치에는 중심점이 포함됩니다.</p><div class="tablewrap"><table><thead><tr><th>방법</th><th>중앙값 px ↓</th><th>평균 px ↓</th><th>최대 px ↓</th><th>중앙값 변화 px</th></tr></thead><tbody>@@METRICS@@</tbody></table></div><p class="small">모든 사전 고정 설정과 학습형 image_joint 3개 seed를 표시했습니다. 정답으로 가장 좋은 설정을 선택하지 않았습니다. 학습형은 이미 저장된 동일 원본의 결과입니다.</p></section>
<section id="gallery"><h2>원본 위에서 점 번호와 실제 선을 확인합니다</h2><p>①과 ③은 같은 원본·같은 좌표입니다. ②의 주황색은 검출된 유한 선분이고, ④는 선택한 역할에 대응된 선만 강조합니다. 코너가 영상 밖에 있으면 좌표를 경계로 옮기지 않고 그대로 자릅니다.</p>
<div class="controls"><label>보정 방법<select id="method"></select></label><label>Hough 후보<select id="preset"><option value="long">long · 36 segments · min 64px</option><option value="short">short · 146 segments · min 16px</option></select></label><label>역할 · 현재 방법의 그래프<select id="role"></select></label><label>② 배경<select id="background"><option value="raw">Raw image</option><option value="canny">Canny edges</option></select></label></div>
<div class="controls checks"><label><input type="checkbox" id="showgt" checked> GT</label><label><input type="checkbox" id="showpred" checked> R0 points</label><label><input type="checkbox" id="shownew" checked> Refined points</label><label><input type="checkbox" id="showlines" checked> Hough segments</label><label><input type="checkbox" id="shownumbers" checked> Point IDs</label><label><input type="checkbox" id="zoom"> 2× 확대 · 가로 스크롤</label><button id="clean">원본만 보기</button><button id="restore">겹쳐 보기 복원</button><button id="full">갤러리 전체 화면</button></div>
<div class="legend"><b style="color:var(--gt)">● GT (hollow = visibility 1)</b><b style="color:var(--pred)">● R0</b><b style="color:var(--new)">● Refined / learned line</b><b style="color:var(--line)">━ Observed Hough segment</b><b>↗ Point displacement</b></div><div id="caseinfo" class="info"></div>
<div class="grid">
<div class="panel"><h3>① Original · GT + R0 point IDs</h3><div class="canvaswrap"><canvas id="baseline" width="640" height="480"></canvas></div><div class="axis"><span>x: 0 → 639 px · y: 0 ↓ 479 px</span><span class="cursor"></span></div></div>
<div class="panel"><h3>② Raw / Canny · all Hough segments</h3><div class="canvaswrap"><canvas id="edges" width="640" height="480"></canvas></div><div class="axis"><span>x: 0 → 639 px · y: 0 ↓ 479 px</span><span class="cursor"></span></div></div>
<div class="panel"><h3>③ Same image · refined points + movement</h3><div class="canvaswrap"><canvas id="refined" width="640" height="480"></canvas></div><div class="axis"><span>x: 0 → 639 px · y: 0 ↓ 479 px</span><span class="cursor"></span></div></div>
<div class="panel"><h3>④ Selected role · actual line evidence</h3><div class="canvaswrap"><canvas id="association" width="640" height="480"></canvas></div><div class="axis"><span>x: 0 → 639 px · y: 0 ↓ 479 px</span><span class="cursor"></span></div></div>
</div><div class="info" id="associationinfo"></div>
<p class="small">역할 번호는 현재 방법의 8선/12선 그래프에 따릅니다. Height(높이), Depth(깊이), Width(너비)는 정해진 코너 인덱스의 의미입니다. 실선 길이·샘플 누적·역할 확률은 attention이나 인과적 설명이 아닙니다. visibility 1은 감독 가능한 가려진/추정 점이며, 그 점을 잇는 amodal 선이 실제 보이는 경계라는 보장은 없습니다.</p>
<details><summary>현재 방법의 9개 점 좌표·개별 오차</summary><div id="pointtable" class="tablewrap"></div></details></section>
<section><h2>경계 후보가 존재하는 것과 번호가 맞는 것은 다릅니다</h2><div class="tablewrap"><table><thead><tr><th>고정 검출 설정</th><th>전체 후보</th><th>최소 길이</th><th>GT oracle 대응 / 평가 가능 12선</th><th>R0 12선의 방향 일치 Hough 지지율</th></tr></thead><tbody>@@SUPPORT@@</tbody></table></div><p>GT oracle 대응은 정답으로 후보를 찾은 사후 진단입니다. 각도 ≤5°, 양 끝점의 평균 수직거리 ≤8px, 영상 안 GT 선분의 투영 겹침 ≥25%를 모두 요구합니다. 분모 9는 양 끝점이 감독되는 전체 12선 중 평가 가능한 선입니다. <strong>보이는 물리적 edge의 개수나 검출 재현율이 아닙니다.</strong></p><p>원본에는 팔레트 외곽 외에도 내부 격자, 의자, 차량, 배경 경계가 있습니다. 짧은 선분을 허용해 후보가 146개로 늘어도 이 기준의 GT 대응은 3/9입니다. 가려지거나 잘린 경계는 증거가 약할 수 있으므로 선이 없다는 이유만으로 해당 점이 틀렸다고 판정할 수 없습니다.</p><details><summary>이번 보정의 고정 규칙</summary><p>영상 전체에서 Gaussian 5×5, σ=1.4 → Canny 60/160 → Hough(ρ=1px, θ=0.5°). long은 투표 60·최소 64px·간격 12px, short는 투표 20·최소 16px·간격 4px입니다. 검출 단계는 GT와 예측 점을 입력받지 않습니다.</p><p>그 다음 원본 예측선에 대해 각도 ≤12°, 법선 거리 ≤box 대각선의 3%, 투영 겹침 ≥20%인 후보를 비용순 일대일 연결합니다. GT를 쓰지 않는 점 anchor + 선 WLS, λ=1, 원본 이동 한도 8px, 중심점 유지입니다. 초기 역할이 틀리면 그 역할에 잘못된 영상 선을 연결할 수 있습니다. 학습형은 합성 데이터로 선택된 기존 규칙을 그대로 사용하며 이 고전 8px 한도와 별개입니다.</p></details></section>
<details class="card oracle" id="oracle"><summary>GT를 본 사후 번호 변경 진단 — 자동 보정이 아닙니다</summary><p><strong>원본 267.93 → 25.71px</strong>는 같은 예측 좌표의 번호만 정답을 이용해 재배열한 결과입니다. 이를 자동 모델의 개선값으로 사용할 수 없습니다. 번호를 맞춘 뒤에도 최대 40.87px의 위치 오차가 남아, 예전 그림의 “위치 오류가 아니다”라는 설명은 너무 강했습니다.</p><div class="twocol"><div><div class="canvaswrap"><canvas id="oraclecanvas" width="640" height="480"></canvas></div><p class="small">GT ORACLE RELABEL · NOT AUTOMATIC<br>GT index i ← original prediction [1,5,6,2,0,4,7,3,8][i]</p></div><div class="tablewrap"><table><thead><tr><th>방법 + 같은 GT 재번호</th><th>중앙값 px</th><th>최대 px</th></tr></thead><tbody>@@ORACLE_ROWS@@</tbody></table></div></div><p>전체 12개 무방향 선의 집합은 이 번호 교환 전후에 정확히 같습니다. 따라서 단순히 “선 위 영상 증거가 있는가”만 모은 점수는 이런 번호 교환을 구별할 수 없습니다. 높이 역할은 유지되지만 깊이·너비 역할은 바뀝니다. 이는 실제 팔레트의 90° 회전이 허용되는 대칭이라는 주장도, 역할을 판단하는 모델까지 해결 불가능하다는 주장도 아닙니다.</p><p>같은 실물 코너끼리 사후 대응시키면 short·측면 8선에서 원래 예측 0 ↔ GT 4는 12.64→6.79px, 원래 예측 3 ↔ GT 7은 9.61→5.25px로 국소 개선됩니다. 이 대응도 GT를 보고 알아낸 것이며, 자동 출력의 점 번호는 그대로 틀립니다. 학습형의 사후 중앙값도 25.34–25.42px이지만 원래 번호의 큰 오류는 남습니다.</p></details>
<section><h2>왜 기존 필터는 통과시켰나요?</h2><p>첨부 그림은 새 선 모델이 아니라 기존 R0 점 예측에 F4_PROPOSED를 적용한 결과입니다. 아래 네 조건을 모두 통과했습니다. 자기 일관성이 있어도 GT의 점 번호와 일치한다는 보장은 없습니다.</p><div class="tablewrap"><table><thead><tr><th>F4 실제 조건</th><th>이 프레임</th><th>임계값</th><th>결정</th></tr></thead><tbody>@@FILTER@@</tbody></table></div><p>Reprojection score <strong>@@REPROJ@@</strong>는 그림에 표시된 진단값이며 <strong>F4 통과 조건에는 포함되지 않습니다.</strong> Flip 수치는 저장값과 계산 코드를 확인했으며, 원래 뒤집기 예측점이 저장되지 않아 새 추론 없는 독립 수치 재현은 하지 않았습니다.</p><details><summary>사용자가 본 참조 그림</summary><img class="reference" id="reference" alt="Original M4 false acceptance diagnostic screenshot"><p class="small">F4를 통과한 프레임 중 GT 최대 오차로 고른 사례입니다. 실제 학습에 편입된 pseudo-label이라는 뜻은 아닙니다.</p></details></section>
<section><h2>다음 점·선 결합에서 검증할 것</h2><p><strong>이번에 확인한 것:</strong> 원본 edge 후보와 작은 위치 보정은 존재하지만, 이 한 장의 번호 오류나 기존 필터 오통과를 해결하지는 못했습니다. 새 거절 필터를 추가하지 않았으므로 오통과율 개선 결과도 없습니다.</p><p><strong>다음 실험 제안:</strong> 영상에서 점과 독립적으로 얻은 후보를 쓰고, 높이·깊이·너비의 역할과 가림 상태를 구별하는 신뢰도를 검증한 뒤, 점 anchor를 유지하는 robust fusion을 적용합니다. 역할이 충돌하거나 불확실하면 보정을 보류하는 규칙도 비교할 수 있습니다. 임계값은 별도 합성 calibration에서 고정하고, 여러 실제 세션에서 누락·coverage와 필터 precision/recall을 함께 평가해야 합니다.</p><p class="small">범위: 사용자가 지정한 이미 본 DEV 한 장입니다. 일반화나 전체 모델 우월성의 근거가 아닙니다. 원본·GT·기존 모델·이전 아티팩트는 수정하지 않았습니다. 좌표와 비교 수치는 RESULTS.json, 계보는 provenance/provenance.json, 기하 검증은 GEOMETRY_AUDIT.json에 저장되어 있습니다.</p></section>
</main><script id="report-data" type="application/json">@@DATA@@</script><script>
'use strict';const DATA=JSON.parse(document.getElementById('report-data').textContent),R=DATA.result,$=id=>document.getElementById(id),C={gt:'#63e6a0',pred:'#5cd9ef',new:'#f68cdb',line:'#ffb454'},IM={};window.REPORT_READY=false;window.REPORT_ERRORS=[];
const methods=[{id:'baseline',...R.baseline},...R.fusion,...R.learned];function opt(el,v,t){let o=document.createElement('option');o.value=v;o.textContent=t;el.append(o)}for(const m of methods)opt($('method'),m.id,DATA.names[m.id]);$('method').value='long_side8';
function current(){return methods.find(m=>m.id===$('method').value)}function graph(m){return m.graph==='cuboid12'?R.cuboid_edges:R.side_edges}function roleName(e){let k=e.join(','),height=[[1,2],[3,0],[5,6],[7,4]].some(x=>x.join(',')===k),depth=[[0,4],[1,5],[2,6],[3,7]].some(x=>x.join(',')===k);return(height?'Height':depth?'Depth':'Width')+' '+e.join('–')}
function roles(){const prev=$('role').value;$('role').replaceChildren();graph(current()).forEach((e,i)=>opt($('role'),i,`${i}: ${roleName(e)}`));if([...$('role').options].some(o=>o.value===prev))$('role').value=prev;else $('role').value='0'}
function line(ctx,a,b,color,width=1.4,alpha=1,dash=[]){if(!a||!b||![...a,...b].every(Number.isFinite))return;ctx.save();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.globalAlpha=alpha;ctx.setLineDash(dash);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();ctx.restore()}
function points(ctx,ps,color,gt=false,alpha=1){if(!ps)return;ctx.save();ctx.globalAlpha=alpha;const numbers=$('shownumbers').checked;ps.forEach((p,i)=>{if(!p||!p.every(Number.isFinite)||(gt&&!R.gt_supervised[i]))return;ctx.strokeStyle='#051019';ctx.lineWidth=2;ctx.fillStyle=color;ctx.beginPath();ctx.arc(p[0],p[1],3.6,0,Math.PI*2);if(gt&&R.gt_visibility[i]===1){ctx.strokeStyle=color;ctx.stroke()}else{ctx.fill();ctx.stroke()}if(numbers){ctx.font='bold 12px monospace';const x=p[0]+5,y=p[1]-6;ctx.strokeStyle='#00111d';ctx.lineWidth=3;ctx.strokeText(String(i),x,y);ctx.fillText(String(i),x,y)}});ctx.restore()}
function structure(ctx,ps,color,gt=false,alpha=.8){for(const[a,b]of R.cuboid_edges){if(gt&&(!R.gt_supervised[a]||!R.gt_supervised[b]))continue;line(ctx,ps[a],ps[b],color,1.1,alpha,gt&&(R.gt_visibility[a]===1||R.gt_visibility[b]===1)?[4,4]:[])}}
function drawGT(ctx){if($('showgt').checked){structure(ctx,R.gt_xy,C.gt,true,.7);points(ctx,R.gt_xy,C.gt,true)}}function drawR0(ctx,alpha=1){if($('showpred').checked){structure(ctx,R.baseline.points_xy,C.pred,false,alpha*.65);points(ctx,R.baseline.points_xy,C.pred,false,alpha)}}
function base(id,image='raw'){const cv=$(id),ctx=cv.getContext('2d');ctx.clearRect(0,0,640,480);ctx.drawImage(IM[image],0,0,640,480);return ctx}function segments(ctx,preset,alpha=.65){if($('showlines').checked)for(const s of R.hough[preset].segments_xy)line(ctx,...s,C.line,1.4,alpha)}
function arrows(ctx,ps){ps.forEach((q,i)=>{const p=R.baseline.points_xy[i],dx=q[0]-p[0],dy=q[1]-p[1],len=Math.hypot(dx,dy);if(len<.05)return;line(ctx,p,q,C.new,1.2);let t=Math.atan2(dy,dx),a=Math.min(4,len);line(ctx,q,[q[0]-a*Math.cos(t-.5),q[1]-a*Math.sin(t-.5)],C.new,1.2);line(ctx,q,[q[0]-a*Math.cos(t+.5),q[1]-a*Math.sin(t+.5)],C.new,1.2)})}
function infinite(h){const [a,b,c]=h,pts=[];if(Math.abs(b)>1e-9)for(const x of[0,639]){const y=-(a*x+c)/b;if(y>=0&&y<=479)pts.push([x,y])}if(Math.abs(a)>1e-9)for(const y of[0,479]){const x=-(b*y+c)/a;if(x>=0&&x<=639)pts.push([x,y])}return pts.length>=2?[pts[0],pts.at(-1)]:null}
function render(){if(!IM.raw||!IM.canny)return;window.REPORT_READY=false;const m=current(),preset=$('preset').value,r=Number($('role').value),e=graph(m)[r],ps=m.points_xy;let ctx=base('baseline');drawGT(ctx);drawR0(ctx);ctx=base('edges',$('background').value);segments(ctx,preset,.8);drawGT(ctx);ctx=base('refined');drawGT(ctx);drawR0(ctx,.35);if($('shownew').checked){structure(ctx,ps,C.new);points(ctx,ps,C.new);arrows(ctx,ps)}ctx=base('association');drawGT(ctx);if($('showpred').checked){points(ctx,R.baseline.points_xy,C.pred);line(ctx,R.baseline.points_xy[e[0]],R.baseline.points_xy[e[1]],C.pred,2)}let info='';
if(m.associations){const a=m.associations[r];info=`Role ${r} · ${roleName(e)} · ${a.matched?'MATCHED':'NO MATCH'}`;if(a.matched){if($('showlines').checked)line(ctx,...a.observed_segment_clipped_xy,C.line,4);info+=` · ${m.preset} segment #${a.segment_index} · angle ${a.angle_deg.toFixed(2)}° · normal distance ${a.normal_distance_px.toFixed(2)}px · overlap ${(100*a.overlap_fraction).toFixed(1)}%`;const s=a.observed_segment_clipped_xy;ctx.font='bold 12px monospace';ctx.fillStyle=C.line;ctx.strokeStyle='#00111d';ctx.lineWidth=3;const tx=(s[0][0]+s[1][0])/2,ty=(s[0][1]+s[1][1])/2-7;if($('showlines').checked){ctx.strokeText('segment #'+a.segment_index,tx,ty);ctx.fillText('segment #'+a.segment_index,tx,ty)}}else info+=' · 고정 각도·거리·겹침 / 일대일 연결 조건을 통과한 유한 선분이 없습니다.';info+=' · GT-free association; role correctness is not guaranteed.'}
else if(m.seed){const d=m.diagnostics,seg=infinite(d.line_h_supplied_image[r]);if(seg&&d.line_valid[r]&&$('showlines').checked)line(ctx,...seg,C.new,2.6,1,[7,4]);info=`Role ${r} · ${roleName(e)} · saved learned supporting line (dashed, NOT an observed finite Hough segment) · null ${(100*d.null_probability[r]).toFixed(1)}% · entropy ${d.entropy_normalized[r].toFixed(3)} · conditional peak ${(100*d.peak_probability_conditional[r]).toFixed(1)}%. These are model outputs, not calibrated edge correctness.`}
else{info=`Role ${r} · ${roleName(e)} · baseline points only; no line correction.`}
if($('shownew').checked&&m.id!=='baseline'){points(ctx,ps,C.new);arrows(ctx,ps)}$('associationinfo').textContent=info;$('caseinfo').textContent=DATA.names[m.id]+` · indexed median ${m.indexed.median_px.toFixed(2)}px · max ${m.indexed.max_px.toFixed(2)}px · change ${(m.indexed.median_px-R.baseline.indexed.median_px).toFixed(2)}px · no relabel`;
let table='<table><thead><tr><th>ID</th><th>GT / visibility</th><th>R0 xy</th><th>Selected xy</th><th>R0 error px</th><th>Selected error px</th></tr></thead><tbody>';for(let i=0;i<9;i++)table+=`<tr><td>${i===8?'8 (center)':i}</td><td>${R.gt_xy[i].map(v=>v.toFixed(2)).join(', ')} / ${R.gt_visibility[i]}</td><td>${R.baseline.points_xy[i].map(v=>v.toFixed(2)).join(', ')}</td><td>${ps[i].map(v=>v.toFixed(2)).join(', ')}</td><td>${R.baseline.indexed.per_point_px[i]?.toFixed(2)??'excluded'}</td><td>${m.indexed.per_point_px[i]?.toFixed(2)??'excluded'}</td></tr>`;$('pointtable').innerHTML=table+'</tbody></table>';
ctx=base('oraclecanvas');structure(ctx,R.gt_xy,C.gt,true);points(ctx,R.gt_xy,C.gt,true);const reordered=R.oracle_reindex.map(i=>R.baseline.points_xy[i]);structure(ctx,reordered,C.new);points(ctx,reordered,C.new);ctx.fillStyle='#181000d9';ctx.fillRect(0,0,640,29);ctx.fillStyle=C.line;ctx.font='bold 14px monospace';ctx.fillText('GT ORACLE RELABEL — NOT AUTOMATIC',10,20);window.REPORT_READY=true}
$('method').onchange=()=>{let m=current();if(m.preset)$('preset').value=m.preset;roles();render()};$('preset').onchange=render;for(const id of['role','background','showgt','showpred','shownew','showlines','shownumbers'])$(id).onchange=render;$('zoom').onchange=()=>{$('gallery').classList.toggle('zoomed',$('zoom').checked)};$('clean').onclick=()=>{for(const id of['showgt','showpred','shownew','showlines'])$(id).checked=false;$('background').value='raw';render()};$('restore').onclick=()=>{for(const id of['showgt','showpred','shownew','showlines','shownumbers'])$(id).checked=true;render()};$('full').onclick=async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await $('gallery').requestFullscreen()}catch(e){$('caseinfo').textContent+=' · Fullscreen unavailable: use browser zoom.'}};document.addEventListener('fullscreenchange',()=>$('gallery').classList.toggle('fullscreen',!!document.fullscreenElement));for(const cv of document.querySelectorAll('canvas')){cv.addEventListener('mousemove',e=>{const r=cv.getBoundingClientRect(),label=cv.closest('.panel')?.querySelector('.cursor');if(label)label.textContent=`x ${((e.clientX-r.left)*640/r.width).toFixed(1)}, y ${((e.clientY-r.top)*480/r.height).toFixed(1)}`})}
Promise.all(['raw','canny'].map(k=>new Promise((resolve,reject)=>{let im=new Image();im.onload=()=>{IM[k]=im;resolve()};im.onerror=reject;im.src=DATA.assets[k]}))).then(()=>{roles();$('role').value='4';$('reference').src=DATA.assets.reference;render()}).catch(e=>{window.REPORT_ERRORS.push(String(e));$('caseinfo').textContent='Image load failed'});
</script></body></html>'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    render(parser.parse_args().run_dir)
