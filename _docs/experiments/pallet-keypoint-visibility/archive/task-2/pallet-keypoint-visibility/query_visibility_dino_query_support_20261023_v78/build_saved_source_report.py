"""Source-only tables, fixed RGB gallery and plots from completed saved outputs."""
from pathlib import Path
import json,hashlib,csv,html
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
R=Path(__file__).resolve().parent;P=R.parent
V77=P/'query_visibility_controlled_pairs_20261023_v77'
RUN=R/'runs/SOURCE_DINO_V1'
def rd(p):return json.loads(Path(p).read_text())
def b(p):
 p=Path(p).resolve();return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def check(x):assert b(x['path'])['sha256']==x['sha256']
def write(p,x):
 assert not p.exists(),p;p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def pct(x):return f'{100*x:.1f}%'
def sigmoid(x):return float(np.exp(-np.logaddexp(0.,-float(x))))
def font(n,bold=False):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans'+('-Bold'if bold else'')+'.ttf',n)
def main():
 assert rd(RUN/'FIT_COMPLETE.json')['completed']
 gate=rd(R/'SOURCE_ADMISSION_DECISION.json')
 audits=R/'independent_saved_audit/INDEPENDENT_SELECTED_SAVED_AUDIT.json'
 assert rd(audits)['PASS']
 versions={
  'V76':rd(V77/'frozen_V76_source_baseline/SOURCE_RESULTS.json')['results'],
  'V77':rd(V77/'runs/PAIRED_SUPPORT_V1/SOURCE_RESULTS.json')['results'],
  'V78':rd(RUN/'SOURCE_RESULTS.json')['results']}
 freeze=rd(RUN/'MODEL_FREEZE.json')
 # Four source cases were fixed before V77 training and before V78 existed.
 fixedmeta=V77/'fixed_saved_score_gallery/SAVED_SCORE_GALLERY_METADATA.json'
 cases=rd(fixedmeta)['cases'];assert len(cases)==4
 out=R/'fixed_saved_score_gallery';out.mkdir(exist_ok=False)
 write(out/'FIXED_CASE_LOCK.json',dict(original_gallery=b(fixedmeta),case_ids=[(c['original_id'],c['anchor_k'])for c in cases],new_fit=b(RUN/'FIT_COMPLETE.json'),code=b(Path(__file__)),selection='Same4metadata-firstV77cases, no newscoreselection',new_forward_current_fit_PnP=0))
 with np.load(RUN/'SELECTED_SOURCE_SCORES.npz',allow_pickle=False)as z:
  scores={k:z[k]for k in ['id','support_logit','primary_loss_mask','after_original_row_index','after_anchor_k']}
 panels=[];case_records=[]
 for i,c in enumerate(cases):
  oi=c['original_row_index'];ai=c['after_row_index'];k=c['anchor_k'];fi=5203+ai
  assert str(scores['id'][oi])==c['original_id'] and scores['after_original_row_index'][ai]==oi and scores['after_anchor_k'][ai]==k
  assert scores['primary_loss_mask'][oi,k]and scores['primary_loss_mask'][fi,k]
  panel=Image.new('RGB',(1600,660),'#111820');d=ImageDraw.Draw(panel)
  d.text((16,12),f'{i+1}. {c["source"]}/{c["split"]}: {c["original_id"]} native{k}',font=font(23,True),fill='white')
  d.text((16,49),'Fixed before V77/V78 fits; weak local-support teacher BEFORE V -> AFTER C; same q/r/L.',font=font(20),fill='#cbd3dd')
  q=np.asarray(c['q']);ref=np.asarray(c['r']);L=c['L']
  for x,key,title in [(16,'raw_before_RGB96','BEFORE: same original query crop'),(420,'raw_after_RGB96','AFTER: opaque public-object paste')]:
   check(c[key]);im=Image.open(c[key]['path']).convert('RGB').resize((384,384),Image.Resampling.NEAREST)
   di=ImageDraw.Draw(im)
   for pt,color in [([191.5,191.5],'#00efff'),(((ref-q)*96/L+48)*4-.5,'#ff00ee')]:
    a1,a2=map(float,pt);di.line((a1-8,a2,a1+8,a2),fill=color,width=2);di.line((a1,a2-8,a1,a2+8),fill=color,width=2)
   panel.paste(im,(x,118));d.text((x,87),title,font=font(19),fill='#d9dce2')
  vals=[]
  for name,key in [('V76','V76 frozen source-selected'),('V77','V77 selected naturalVAL')]:
   z=c['scores'][key];vals.append((name,z['before_logit'],z['after_logit']))
  vals.append(('V78 frozen DINO',float(scores['support_logit'][oi,k]),float(scores['support_logit'][fi,k])))
  for j,(name,bz,az)in enumerate(vals):
   y=97+j*132;color='#ffe091'if j==2 else'#cbd7eb';d.text((840,y),name,font=font(23,True),fill=color)
   d.text((840,y+36),f'Before pV={sigmoid(bz):.6f}  {"VISIBLE"if bz>=0 else"COVERED"}',font=font(22),fill=color)
   d.text((840,y+70),f'After  pV={sigmoid(az):.6f}  {"VISIBLE"if az>=0 else"COVERED"}',font=font(22),fill=color)
   d.text((840,y+102),f'both correct={bool(bz>=0 and az<0)}',font=font(19),fill=color)
  d.text((16,534),f'q cyan={q.round(3).tolist()} | r magenta={ref.round(3).tolist()} | q-r={np.linalg.norm(q-ref):.3f}px | L={L:.3f}px',font=font(20),fill='#d0d5dc')
  d.text((16,574),'Shown RGB96 is the entire local L crop. V78 encodes its bilinear224 resize plus a separate predicted target crop.',font=font(19),fill='#d0d5dc')
  d.text((16,608),'Public q is source GT-derived emulation. Four pre-exposed illustrations do not certify semantic D/E/S or generalization.',font=font(19),fill='#d0d5dc')
  png=out/f'PAIR_{ai:06d}.png';panel.save(png);panels.append(panel)
  case_records.append(dict(original_id=c['original_id'],after_row_index=ai,native_k=k,models=[dict(model=name,before_pV=sigmoid(bz),after_pV=sigmoid(az),both_correct=bool(bz>=0 and az<0))for name,bz,az in vals],before=c['raw_before_RGB96'],after=c['raw_after_RGB96'],panel=b(png)))
 sheet=Image.new('RGB',(1600,660*4))
 for i,panel in enumerate(panels):sheet.paste(panel,(0,660*i))
 sheet.save(out/'CONTACT_SHEET.png')
 write(out/'GALLERY_METADATA.json',dict(cases=case_records,lock=b(out/'FIXED_CASE_LOCK.json'),scores=b(RUN/'SELECTED_SOURCE_SCORES.npz'),contact_sheet=b(out/'CONTACT_SHEET.png'),new_forward_current_fit_PnP=0))
 # Full source VAL tables; source split/GT admission is identical for all models.
 rows=[]
 for domain in ['NATURAL_ORIGINAL','CONTROLLED_AFTER']:
  for source in ['SYNTHETIC','YCB_PUBLIC_REAL']:
   for version,result in versions.items():
    cohort=result['natural_and_after'][f'{domain}|{source}|val'];rows.append(dict(domain=domain,source=source,model=version,**cohort['metrics']))
 write(R/'SAVED_SOURCE_COMPARISON.json',dict(rows=rows,source_gate=b(R/'SOURCE_ADMISSION_DECISION.json'),bindings={v:b(path)for v,path in [('V76',V77/'frozen_V76_source_baseline/BASELINE_SEAL.json'),('V77',V77/'REPORT_FINAL_SEAL.json'),('V78',RUN/'FIT_COMPLETE.json')]},labels='C0/V1/U-1 weak reference-local support; not humansemantic.',fully_solved=False,current_reads=0,PnP=0))
 with open(R/'SOURCE_VAL_METRICS.csv','x')as f:
  keys=['domain','source','model','N','visible_precision','visible_recall','covered_precision','covered_recall','mean_BCE_FP32','confusion_rows_GT_cols_pred_COVERED_VISIBLE'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
 colors=['#939fb0','#4188af','#cf6b27'];fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
 for j,(model,result)in enumerate(versions.items()):
  m=result['natural_and_after']['NATURAL_ORIGINAL|YCB_PUBLIC_REAL|val']['metrics']
  vals=[m[f]for f in ['visible_recall','covered_recall','covered_precision']]
  axes[0].bar(np.arange(3)+(j-1)*.24,vals,width=.23,label=model,color=colors[j])
  ps=[result['paired_anchors'][s+'|val']['both_correct_rate']for s in ['SYNTHETIC','YCB_PUBLIC_REAL']]
  axes[1].bar(np.arange(2)+(j-1)*.24,ps,width=.23,label=model,color=colors[j])
 axes[0].set(title='Natural public YCB VAL',xticks=range(3),xticklabels=['V recall','C recall','C precision'],ylim=(0,1.05));axes[0].legend(fontsize=9)
 axes[1].set(title='Controlled pair: both correct',xticks=range(2),xticklabels=['Synthetic / 230','YCB / 731'],ylim=(0,1.05))
 hist=rd(RUN/'HISTORY.json');axes[2].plot([h['update']for h in hist],[h['naturalVAL4strata_macroBCE']for h in hist],color=colors[2]);axes[2].set(title='V78 natural VAL selector',xlabel='Update',ylabel='4 strata macro BCE')
 for ax in axes[:2]:ax.yaxis.set_major_formatter(PercentFormatter(1))
 for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
 fig.savefig(R/'SOURCE_COMPARISON.png',dpi=150);plt.close(fig)
 def table(domain):
  s='<table><tr><th>자료 / model</th><th>N</th><th>CM 행/열 C,V</th><th>V precision</th><th>V recall</th><th>C precision</th><th>C recall</th><th>BCE</th></tr>'
  for v in rows:
   if v['domain']==domain:s+='<tr>'+''.join(f'<td>{x}</td>'for x in [v['source']+' / '+v['model'],v['N'],v['confusion_rows_GT_cols_pred_COVERED_VISIBLE'],*[pct(v[f])for f in ['visible_precision','visible_recall','covered_precision','covered_recall']],f"{v['mean_BCE_FP32']:.4f}"])+'</tr>'
  return s+'</table>'
 checks='<table><tr><th>학습 전 고정한 source 기준</th><th>V78</th><th>기준</th><th>판정</th></tr>'+''.join('<tr>'+''.join(f'<td>{x}</td>'for x in [html.escape(k),pct(v['value']),pct(v['minimum']),'PASS'if v['PASS']else'FAIL'])+'</tr>'for k,v in gate['checks'].items())+'</table>'
 lead='공개 source 기준을 충족했습니다. 현재 팔레트의 사람 가림 평가와 새 촬영 일반화는 아직 확인하지 않았습니다.'if gate['source_candidate_pass']else'범용 DINO 특징도 고정한 공개 source 기준을 모두 충족하지 못했습니다. 가림 판별을 해결한 모델로 채택하지 않습니다.'
 public=versions['V78']['natural_and_after']['NATURAL_ORIGINAL|YCB_PUBLIC_REAL|val']['metrics']
 body='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>V78 DINO source 비교</title><style>body{font:16px/1.7 NanumGothic,sans-serif;max-width:1430px;margin:35px auto;padding:0 25px;color:#203047;background:#f5f8fc}h1{font-size:32px}h2{margin-top:42px}.lead{font-size:22px;font-weight:bold;background:#fff0d8;border-left:6px solid #cf6b27;padding:22px}table{border-collapse:collapse;background:white;width:100%;font-size:14px;margin:20px 0}td,th{border:1px solid #d1ddeb;padding:9px;text-align:left}th{background:#e4edf8}img{max-width:100%;height:auto}a{color:#155ead}</style><h1>V78 — 범용 동결 DINO와 query 중심 가림 판별</h1>'''
 body+=f'<p class="lead">{lead}</p><p>공개 자연 YCB VAL에서 V precision {pct(public["visible_precision"])}, V recall {pct(public["visible_recall"])}, C precision {pct(public["covered_precision"])}, C recall {pct(public["covered_recall"])}입니다. 아래 V/C는 사람의 D/E/S가 아닌 참조 주변 support 약한 교사입니다.</p>'
 body+=f'<p>사전학습 DINOv2 ViT-S/14의 가중치를 동결하고 48,929개 파라미터 head만 학습 1회·2,000 update 실행했습니다. 자연 VAL source×class 4집단 평균 BCE 최저 {freeze["best_update"]} update를 선택했고 문턱은 0.5입니다. V77과 원본·가림 후 이미지, q/L/box, 라벨, 분할, query admission, 32+32+32 sampler, 최적화 조건을 유지했습니다. 현재 평가 223장 접근·추가 추론·학습·문턱 선택은 0회이며 PnP도 0회입니다.</p>'
 body+='''<h2>무엇을 바꿨는가</h2><p>합성 원본 1,003개와 공개 객체 4,200개(고유 원본 RGB 1,000장), 가림 후 4,799개로 구성된 10,002개 대상 단위 입력·80,016 query를 보존했습니다. 10,002개의 독립 촬영 영상을 뜻하지 않습니다. 같은 query 중심 RGB96과 예측 8점으로 정한 대상 crop128을 각각 bilinear224로 변환합니다. 마지막 16×16 DINO patch token에서 query 중앙 2×2와 local 전체의 유효 영역 가중 평균, 대상 crop 전체의 가중 평균을 읽습니다. head는 이 특징과 원래 42차원 query 배치·유효 영역 비율을 받습니다. 참조점·정답 mask·가림 anchor·source ID·사람 가림 주석은 forward에 제공하지 않습니다.</p><p>대상 crop은 초기 query box이며 정답 대상의 분리 mask가 아닙니다. 배경과 다른 물체가 섞일 수 있습니다. 새 표현·pooling·head·초기화를 함께 바꾼 후보이므로 DINO만의 인과 효과로 해석하지 않습니다. 작은 입력 crop에서 사라진 세부를 업샘플링으로 복원한 것도 아닙니다.</p><h2>원래 영상의 자연 VAL</h2>'''+table('NATURAL_ORIGINAL')
 body+='''<p>자연 VAL 전체는 합성 1,880점·공개 6,400점입니다. 같은 known/input/ref-in-crop 기준으로 합성 1,625점(C173/V1452), 공개 2,018점(C287/V1731)을 평가합니다. 교사 U는 합성187·공개4,299점으로 남겼고 음성으로 바꾸지 않았습니다. binary head의 유효 입력에는 학습된 confidence 보류가 없으며 U·입력 제외율은 모델의 신뢰도 보류율과 다릅니다.</p><h2>학습 전에 고정한 source 채택 기준</h2>'''+checks
 body+='''<p>이 기준은 checkpoint 선택 규칙이 아닙니다. checkpoint는 자연 VAL의 고정 BCE 규칙 하나로 골랐고, controlled pair는 진단과 사전에 명시한 채택 기준에만 씁니다. 기준 통과 여부에 따라 문턱·seed·checkpoint를 다시 고르지 않았습니다. 통과하더라도 반복 공개 검증의 약한 라벨에서 성립하는 후보 조건이며 사람 semantic 가림 성능의 보증은 아닙니다.</p><h2>가린 후 전체 known query VAL</h2>'''+table('CONTROLLED_AFTER')
 body+='<img src="SOURCE_COMPARISON.png" alt="V76 V77 V78 공개 source 검증 비교"><h2>미리 고정한 실제 crop 4쌍</h2><p>가리기 전후 q·참조점·크기가 같습니다. V77 전에 metadata로 고른 합성 TRAIN/VAL·공개 TRAIN/VAL 4사례를 그대로 사용했습니다. source 학습 사례도 포함하며 그림 네 개로 전체 정확도를 추정하지 않습니다.</p><img src="fixed_saved_score_gallery/CONTACT_SHEET.png" alt="고정 source RGB crop과 V76 V77 V78 저장 점수">'
 body+='''<h2>해석 범위</h2><p>목표는 native 가상 cuboid 참조점 주변에서 대상 support가 남았는지를 판별하는 것입니다. M0에 예상 support가 없는 경우는 U이며 외부 가림 C로 바꾸지 않습니다. native 참조점을 물리 mesh 코너로 이동하지 않습니다. 외부 가림·자기 가림·위치 오류의 원인 분류는 별도이며 이 binary head가 자기 가림 원인까지 학습했다는 주장은 하지 않습니다.</p><p>공개 query는 합성 TRAIN의 초기 오차를 적용한 source augmentation이며 실제 팔레트 초기 추론기 출력이 아닙니다. source VAL은 반복 개발 자료이고 DINO의 범용 사전학습 자료 전체와의 중복은 이번 실험에서 인증하지 않았습니다. 현재 223장은 직접 촬영했고 KU Leuven 자료와 무관하다는 사용자 확인을 보존했습니다. 현재 자료로 후보를 선택하지 않았습니다.</p><h2>검산과 저장 파일</h2><ul>'''
 links={'고정 실험 조건':'EXPERIMENT_DECISIONS_BEFORE_FEATURES.json','채택 기준 판정':'SOURCE_ADMISSION_DECISION.json','전체 수치':'SAVED_SOURCE_COMPARISON.json','체크포인트 동결':'runs/SOURCE_DINO_V1/MODEL_FREEZE.json','학습 완료 기록':'runs/SOURCE_DINO_V1/FIT_COMPLETE.json','저장 결과 독립 검산':'independent_saved_audit/INDEPENDENT_SELECTED_SAVED_AUDIT.json','같은 입력의 V77 결과':'../query_visibility_controlled_pairs_20261023_v77/RESULTS_KO.html','마지막 현재 팔레트 평가 V76':'../query_visibility_reference_local_support_20261023_v76/RESULTS_KO.html'}
 body+=''.join(f'<li><a href="{(R/p).resolve()}">{label}</a></li>'for label,p in links.items())+'</ul></html>'
 (R/'RESULTS_KO.html').write_text(body)
 print(json.dumps(dict(report=str(R/'RESULTS_KO.html'),gallery=str(out/'CONTACT_SHEET.png'),source_candidate_pass=gate['source_candidate_pass']),ensure_ascii=False))
if __name__=='__main__':main()
