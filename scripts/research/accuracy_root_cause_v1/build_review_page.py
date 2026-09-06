"""human_review 패킷을 브라우저에서 클릭으로 끝낼 수 있는 한 페이지로 만든다.

목적 : CSV 를 손으로 채우게 하면 리뷰가 실제로 굴러가지 않는다.
지표 : 프레임당 두 질문(축 가설 / 가림 코너)을 클릭 두세 번으로 답하고,
       끝나면 CSV 두 개를 내려받아 그대로 폼 파일에 넣을 수 있는가.

★anchoring 방지 — 모델 예측·재투영 수치·채택 표시·정답키를 **페이지에 넣지 않는다.**
  A/B 어느 쪽이 저장본인지는 _ANSWER_KEY.csv 에만 있고 이 스크립트는 그 파일을 읽지 않는다.

패킷 폴더를 매번 새로 훑으므로, 프레임이 추가되면 다시 돌리면 된다.
"""
import csv, html, json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
PACK = ROOT / "_docs/audits/accuracy_root_cause_v1/human_review"
RES = ROOT / "data/pallet/results/accuracy_root_cause_v1"

# ★role·note 를 페이지에 싣지 않는다.
# phase-2 의 role 은 "GROSS_KEYPOINT_ERROR" 같은 **모델 실패 라벨**이고 note 는
# "identity_max=425.2px" 처럼 모델 오차 수치다.  지시문 §3.3 이 금지한
# "이전 failure label" 그 자체라, 보이면 리뷰어가 그쪽으로 끌려간다.
# 목록은 분석용으로 CSV 에만 남고, 페이지는 프레임 id 만 보여준다.

frames = []
for d in sorted(x for x in PACK.iterdir() if x.is_dir()):
    imgs = {n: (d / f"{n}.png") for n in ("01_raw", "02_gt_only",
                                          "03_geometry_A", "03_geometry_B")}
    if not imgs["01_raw"].exists():
        continue
    frames.append(dict(id=d.name, dir=d.name,
                       has_geom=imgs["03_geometry_A"].exists() and imgs["03_geometry_B"].exists()))

# phase-1(축 모호) 과 phase-2(실패 농축) 가 순서로 묶여 보이면 그것도 단서가 된다.
# 프레임 id 로 결정적 셔플 — 매번 같은 순서라 중간 저장이 유지된다.
import hashlib
frames.sort(key=lambda f: hashlib.md5(("20260906:" + f["id"]).encode()).hexdigest())

print(f"프레임 {len(frames)} · 그중 A/B 가설 있음 {sum(f['has_geom'] for f in frames)}")

DATA = json.dumps(frames, ensure_ascii=False)
page = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<title>GT review — accuracy_root_cause_v1</title>
<style>
:root{--bg:#111;--fg:#eee;--mut:#999;--line:#333;--acc:#4da3ff;--ok:#3ddc84}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);
 font:14px/1.5 system-ui,-apple-system,"Noto Sans KR",sans-serif}
header{position:sticky;top:0;background:#0b0b0bee;backdrop-filter:blur(6px);
 border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:16px;align-items:center;z-index:9}
header b{font-size:15px} .sp{flex:1}
button{background:#222;color:var(--fg);border:1px solid var(--line);border-radius:6px;
 padding:7px 12px;cursor:pointer;font:inherit} button:hover{border-color:var(--acc)}
button.primary{background:var(--acc);color:#001;border-color:var(--acc);font-weight:600}
main{padding:16px;max-width:1500px;margin:0 auto}
.frame{border:1px solid var(--line);border-radius:10px;margin-bottom:22px;overflow:hidden}
.fh{padding:10px 14px;background:#181818;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.tag{font-size:11px;padding:2px 8px;border-radius:99px;background:#243;color:#9f9;border:1px solid #365}
.mut{color:var(--mut);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;padding:12px}
.cell{background:#000;border:1px solid var(--line);border-radius:8px;overflow:hidden}
.cell h4{margin:0;padding:6px 10px;font-size:12px;font-weight:600;background:#1c1c1c;color:#bbb}
.cell img{width:100%;display:block;cursor:zoom-in}
.q{padding:12px 14px;border-top:1px solid var(--line);display:flex;gap:10px;
 align-items:center;flex-wrap:wrap}
.q label{display:flex;gap:6px;align-items:center;cursor:pointer;
 border:1px solid var(--line);border-radius:6px;padding:6px 10px}
.q label:hover{border-color:var(--acc)} .q input{accent-color:var(--acc)}
.done{outline:2px solid var(--ok)}
textarea{width:100%;min-height:40px;background:#0d0d0d;color:var(--fg);
 border:1px solid var(--line);border-radius:6px;padding:6px;font:inherit}
#lb{position:fixed;inset:0;background:#000e;display:none;align-items:center;
 justify-content:center;z-index:50;cursor:zoom-out} #lb img{max-width:98vw;max-height:98vh}
.help{background:#161616;border:1px solid var(--line);border-radius:10px;padding:14px 18px;margin-bottom:20px}
.help h2{margin:0 0 8px;font-size:16px} .help li{margin:3px 0}
kbd{background:#222;border:1px solid var(--line);border-radius:4px;padding:1px 6px;font-size:12px}
</style></head><body>
<header><b>GT review</b><span class="mut" id="prog"></span><span class="sp"></span>
<button onclick="jumpNext()">다음 미완료 &darr;</button>
<button class="primary" onclick="dl()">CSV 내려받기</button>
<button onclick="if(confirm('입력을 전부 지웁니다'))reset()">초기화</button></header>
<main>
<div class="help"><h2>무엇을 판정하나</h2>
<ol>
<li><b>축 가설</b> — 아래 <b>Hypothesis A</b> 와 <b>B</b> 는 팔레트의 가로/세로(width/depth)를
서로 바꾼 두 해석이다. 사진의 팔레트 윤곽과 <b>더 잘 맞는 쪽</b>을 고른다.
<b>둘 다 사진과 안 맞으면 「둘 다 틀림」</b> — 그건 저장된 pose 자체가 틀렸다는 뜻이라
「모르겠음」과 다른 정보다.
어느 쪽이 저장본인지는 일부러 감췄고, 순서도 프레임마다 무작위다.</li>
<li><b>가림 코너</b> — <code>02 GT keypoints</code> 에서 <b>빈 사각형</b>은 사람이 찍은 점이 아니라
PnP 로 채운 점이다. 그 점이 <b>사진 속 실제 코너 자리</b>에 있으면 OK, 엉뚱하면 BAD 를 고른다.</li>
</ol>
<p class="mut">모르겠으면 <b>모르겠음</b>을 고르는 편이 찍는 것보다 낫다 — 그게 이 리뷰의 핵심 정보다.
입력은 브라우저에 자동 저장되니 중간에 닫아도 된다. 이미지를 클릭하면 크게 볼 수 있다.</p></div>
<div id="app"></div></main>
<div id="lb" onclick="this.style.display='none'"><img id="lbi"></div>
<script>
const F=__DATA__, K='gtreview_v1';
let S=JSON.parse(localStorage.getItem(K)||'{}');
const save=()=>localStorage.setItem(K,JSON.stringify(S));
function done(f){const s=S[f.id]||{};return (!f.has_geom||s.hyp)&&s.extrap}
function prog(){const d=F.filter(done).length;
 document.getElementById('prog').textContent=` ${d} / ${F.length} 완료`;}
function set(id,k,v){S[id]=S[id]||{};S[id][k]=v;save();
 document.getElementById('c_'+id).classList.toggle('done',done(F.find(x=>x.id===id)));prog();}
function zoom(src){document.getElementById('lbi').src=src;document.getElementById('lb').style.display='flex';}
function jumpNext(){const n=F.find(f=>!done(f));if(n)document.getElementById('c_'+n.id).scrollIntoView({behavior:'smooth',block:'start'});else alert('전부 완료했습니다.');}
function reset(){S={};save();render();}
function radio(id,k,val,label){const s=S[id]||{};const on=s[k]===val;
 return `<label><input type="radio" name="${k}_${id}" ${on?'checked':''} onchange="set('${id}','${k}','${val}')">${label}</label>`;}
function render(){
 document.getElementById('app').innerHTML=F.map((f,i)=>{
  const s=S[f.id]||{};const d=f.dir;
  const geom=f.has_geom?`
   <div class="cell"><h4>Hypothesis A</h4><img loading="lazy" src="${d}/03_geometry_A.png" onclick="zoom(this.src)"></div>
   <div class="cell"><h4>Hypothesis B</h4><img loading="lazy" src="${d}/03_geometry_B.png" onclick="zoom(this.src)"></div>`:'';
  const qgeom=f.has_geom?`<div class="q"><b>1. 어느 쪽이 사진과 맞나</b>
   ${radio(f.id,'hyp','A','A 가 맞다')}${radio(f.id,'hyp','B','B 가 맞다')}${radio(f.id,'hyp','both_wrong','둘 다 틀림')}${radio(f.id,'hyp','cannot_tell','모르겠음')}
   <span class="mut">확신도</span>${[1,2,3,4,5].map(n=>radio(f.id,'conf',String(n),String(n))).join('')}</div>`
   :`<div class="q mut">클릭 코너가 부족해 두 가설을 그릴 수 없는 프레임이다 — 2번만 답한다.</div>`;
  return `<section class="frame ${done(f)?'done':''}" id="c_${f.id}">
   <div class="fh"><b>${f.id}</b><span class="mut">${i+1} / ${F.length}</span></div>
   <div class="grid">
    <div class="cell"><h4>01 raw</h4><img loading="lazy" src="${d}/01_raw.png" onclick="zoom(this.src)"></div>
    <div class="cell"><h4>02 GT keypoints — 채운 원 = 사람이 찍음, 빈 사각 = PnP 로 채움</h4><img loading="lazy" src="${d}/02_gt_only.png" onclick="zoom(this.src)"></div>
    ${geom}</div>
   ${qgeom}
   <div class="q"><b>2. 빈 사각(외삽) 점들이 실제 코너 자리에 있나</b>
    ${radio(f.id,'extrap','ok','전부 맞다')}${radio(f.id,'extrap','some','일부 어긋난다')}
    ${radio(f.id,'extrap','bad','대체로 엉뚱하다')}${radio(f.id,'extrap','none','외삽 점이 없다')}
    ${radio(f.id,'extrap','cannot_tell','모르겠음')}</div>
   <div class="q" style="display:block"><span class="mut">메모(선택)</span>
    <textarea oninput="set('${f.id}','note',this.value)">${(s.note||'').replace(/</g,'&lt;')}</textarea></div>
  </section>`}).join('');
 prog();}
function dl(){
 const fr=[['frame_id','hypothesis_A_better','hypothesis_B_better','cannot_tell','both_wrong','confidence_1to5','extrapolated_points_verdict','note']];
 F.forEach(f=>{const s=S[f.id]||{};
  // ★답이 하나라도 있으면 내보낸다.  예전엔 has_geom 인 프레임만 내보내서,
  //   A/B 가설을 못 그린 4장의 답과 메모가 조용히 버려졌다.
  if(s.hyp||s.extrap||(s.note||'').trim())
    fr.push([f.id,s.hyp==='A'?1:0,s.hyp==='B'?1:0,s.hyp==='cannot_tell'?1:0,s.hyp==='both_wrong'?1:0,
             s.conf||'',s.extrap||'',(s.note||'').replace(/[\\n,]/g,' ')]);});
 const g=(rows,n)=>{const b=new Blob(['\ufeff'+rows.map(r=>r.join(',')).join('\\n')],{type:'text/csv'});
  const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download=n;a.click();};
 // 한 파일로 합친다 — 연속 다운로드 두 번째를 브라우저가 막아 지난번에 유실됐다.
 g(fr,'REVIEW_FILLED_frame.csv');}
render();
</script></body></html>"""
(PACK / "REVIEW.html").write_text(page.replace("__DATA__", DATA), encoding="utf-8")
print("wrote", PACK / "REVIEW.html")
