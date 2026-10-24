"""Render closed SOURCE-only diagnostic results. No model/input-cache access."""
from pathlib import Path
import argparse, hashlib, html, json, os
V=Path(__file__).absolute().parent
O=V/'source_train_modality_diagnosis_v2'
F=V/'saved_readout_root'
ARMS=('C62_DENSE_POINT_RAW_Q8','C63_DENSE_POINT_WEAK_BRIDGE_RAW_Q8')
MODES=('FACTUAL_UNCHANGED','CONSTANT_RGB127_KEEP_VALID_XY_CONTEXT','ZERO_OTHER_Q8_XY_ONLY','ZERO_OTHER_Q8_MASK_AND_XY','ZERO_NATIVE_ONEHOT_BRANCH_ONLY','ZERO_ALL_CONDITION_BRANCHES')
SHORT=('Factual','RGB127','No q8 XY','No q8','No native','No condition')

def bind(p):
 p=Path(p);assert 'v112' not in str(p).casefold() and V in p.parents and not p.is_symlink()
 return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def rel(p):return str(Path(p).relative_to(V))
def pct(x):return '—'if x is None else '%.2f%%'%(100*x)

def figures():
 c=read(O/'COMPLETE.json');assert c['completed'] and c['PASS'] and c['scalar_rows']==73680
 s=read(O/'SUMMARY.json');assert s['conditions']==list(MODES)
 F.mkdir(exist_ok=False)
 os.environ['MPLCONFIGDIR']=str(V.parent/'publication_root_preparation/V214_MATPLOTLIB_CACHE')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 fig,axs=plt.subplots(2,2,figsize=(15,9),sharey=True)
 for ai,arm in enumerate(ARMS):
  for pi,pop in enumerate(('POINT','WEAK')):
   ax=axs[pi,ai];rows=s['models'][arm][pop];x=np.arange(6)
   ax.bar(x-.19,[100*rows[m]['D_recall']for m in MODES],.38,label='D recall: visible',color='#247ba0')
   ax.bar(x+.19,[100*rows[m]['E_recall']for m in MODES],.38,label='E recall: external/weak cover',color='#ed6a5a')
   ax.set_xticks(x,SHORT,rotation=18,ha='right');ax.set_ylim(0,105);ax.grid(axis='y',alpha=.25);ax.set_axisbelow(True)
   ax.set_title(arm[:3]+' / '+('SOURCE point TRAIN N6096 (D5853/E243)'if pop=='POINT'else'SOURCE local-feature weak TRAIN N44 (D26/E18)'))
   ax.set_ylabel('Recall (%)');ax.legend(loc='lower left',fontsize=9)
 fig.suptitle('Frozen head SOURCE TRAIN sensitivity — OOD interventions; not real-test performance',fontsize=14)
 fig.tight_layout();fig.savefig(F/'SOURCE_TRAIN_MODALITY_RECALL.png',dpi=140);plt.close(fig)
 from PIL import Image,ImageDraw,ImageFont
 manifest=read(O/'FIXED8_DISPLAY_IMAGE_MANIFEST.json');images=manifest['images'];assert len(images)==8
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
 canvas=Image.new('RGB',(1240,860),'white');d=ImageDraw.Draw(canvas)
 d.text((20,10),'Fixed SOURCE inputs chosen before scores. Display enlargement only; original clean96 unchanged.',font=font,fill='black')
 for k in range(4):
  left,constant=images[2*k:2*k+2];assert left['row_id']==constant['row_id']and left['mode']==MODES[0]and constant['mode']==MODES[1]
  x=20+620*(k%2);y=55+400*(k//2)
  d.text((x,y),left['population']+' / '+left['row_id'][:49],font=font,fill='black')
  for j,z in enumerate((left,constant)):
   p=Path(z['path']);assert bind(p)['sha256']==z['sha256']
   with Image.open(p)as im:
    assert im.mode=='RGB'and im.size==(96,96);im=im.resize((288,288),Image.Resampling.NEAREST);canvas.paste(im,(x+300*j,y+55))
   d.text((x+300*j,y+30),'Original RGB96'if j==0 else'RGB127 (valid/grid kept)',font=font,fill='black')
 canvas.save(F/'FIXED4_SOURCE_INPUT_DISPLAY.png')
 write(F/'FIGURES_COMPLETE.json',dict(completed=True,PASS=True,code=bind(__file__),diagnostic_complete=bind(O/'COMPLETE.json'),summary=bind(O/'SUMMARY.json'),figures=[bind(F/'SOURCE_TRAIN_MODALITY_RECALL.png'),bind(F/'FIXED4_SOURCE_INPUT_DISPLAY.png')],actual_visual_view_pending=True,raw_input_cache_or_checkpoint_opens=0,new_training=0))

def report():
 s=read(O/'SUMMARY.json');c=read(O/'COMPLETE.json');fq=read(F/'FIGURES_COMPLETE.json')
 qa=V/'independent_saved_result_review_c/REVIEW.json';vis=V/'ROOT_ACTUAL_FIXED4_SOURCE_CROPS_VISUAL_QA.json'
 assert c['completed']and c['PASS']and read(qa)['completed']and read(qa)['PASS']and read(vis)['completed']and read(vis)['PASS']
 rows=[]
 for arm in ARMS:
  for pop in ('POINT','WEAK'):
   for mode,label in zip(MODES,SHORT):
    z=s['models'][arm][pop][mode];rows.append([arm[:3],pop,label,str(z['CM_D_E_by_D_E_U']),pct(z['D_recall']),pct(z['E_recall']),pct(z['E_precision']),str(z['sign_flips']),str(z['U'])])
 headers=['모델','SOURCE 집합','조건','CM D/E행 × D/E/U열','D recall','E recall','E precision','원응답과 다른 수','U']
 table='|'+'|'.join(headers)+'|\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('|'+'|'.join(r)+'|'for r in rows)
 intro='동결 C62/C63의 SOURCE TRAIN 의존도만 검사했다. 모델2개×6조건×(point6096+weak44)=73,680개 scalar를 저장했다. 원 모델 출력을 먼저 재현하고, 가중치·원 q/L/native ID를 유지한 채 RGB 또는 condition branch 입력 일부만 바꿨다. 현재 실사·공개 HOLDOUT·SOURCE HOLDOUT의 새 추론이나 점수 읽기는 없다. 과거 공개 요약에 노출된 이력은 별도 기록했다.'
 limits='point7040 중 known6096(D5853/E243)만 forward했다. UNKNOWN944와 원 inputU64는 metadata에 유지하고 다시 판정하지 않았다. point는 own-image procedural point 상태이고 weak44는 공개3frame의 visible-feature 덮기 감독이다. 이 집합은 이미 학습에 쓴 자료다. constantRGB/조건제거는 OOD 민감도 검사로, 일반화 성능이나 실사 실패의 단일 원인을 증명하지 않는다. RGB127에서도 validity와 고정 XY grid가 남는다. 보류0은 이진 규칙의 결과이며 confidence가 아니다.'
 links=[('native8 전체 표',rel(O/'NATIVE8_METRICS.csv')),('모든 조건 full 표',rel(O/'FULL_AND_NATIVE8_METRICS.csv')),('73,680 scalar 원장',rel(O/'ALL73680_SOURCE_TRAIN_CONDITION_SCALARS.jsonl')),('동일 non-RGB 입력 pair',rel(O/'PAIR_INPUT_MATCHES_BEFORE_SCORE_READS.json')),('pair별 응답 차이',rel(O/'PAIRED_SOURCE_VARIANT_CONDITION_CONTRASTS.jsonl')),('감독 의미·공개 실사 coverage 감사','target_coverage_review_b/KO.md'),('기존 pretrained 시도와 코드 readiness','pretrained_source_readiness_c/KO.md'),('독립 저장 결과 감사',rel(qa)),('실행 전 metadata gate 보완 기록','ROOT_PRE_EXECUTION_PAYLOAD_ANNOTATION_IDENTITY_FINDING_AND_BOUNDED_V2_REPAIR.json')]
 doc=['# V214 SOURCE TRAIN 영상·조건 의존 진단','',intro,'',limits,'','![SOURCE TRAIN recall](saved_readout_root/SOURCE_TRAIN_MODALITY_RECALL.png)','',table,'','## 결과 전에 고정한 네 입력','','![고정 네 SOURCE 입력](saved_readout_root/FIXED4_SOURCE_INPUT_DISPLAY.png)','','네 사례는 대표 표본이 아니다. actual visual receipt는 표시 가독성과 원crop 관찰만 확인하며 물리 native 정답·실제 차폐 원인을 새로 인증하지 않는다.','','## SOURCE 감독 범위','','PUBLIC SOURCE TRAIN155frame 중 weak44는3frame/8case/native0..4뿐이며 native5/6/7은 없다. 나머지152frame은 RGB가 없는 것이 아니다. 기존 public owner239행/81frame은 query 중심 소유자 보조 정답으로 native D/E와 다르다. Point known의 native6/7 E입력은2/4개이며 독립 장면 수가 아니다. 다음 SOURCE-only 작업은 새 프레임의 intended-part 대응을 확인하고 UNKNOWN을 보존하며 weak 감독 범위를 넓히는 것이다.','','## 표·코드·관찰','',*['- [%s](%s)'%(a,b)for a,b in links],'','새 학습·문턱 조정·모델 채택·PnP 변경·전체 해결을 주장하지 않는다.','']
 text='\n'.join(doc)
 for name in ('ROOT_REPORT_KO.md','PORTABLE_REPORT_KO.md'):
  with(V/name).open('x')as f:f.write(text)
 page='<!doctype html><meta charset="utf-8"><style>body{font:17px sans-serif;margin:24px}table{border-collapse:collapse}th,td{padding:6px;border:1px solid #bbb}img{max-width:100%}</style><h1>V214 SOURCE TRAIN diagnosis</h1><p>'+html.escape(intro)+'</p><p>'+html.escape(limits)+'</p><img src="saved_readout_root/SOURCE_TRAIN_MODALITY_RECALL.png"><table><tr>'+''.join('<th>'+html.escape(x)+'</th>'for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>'for x in row)+'</tr>'for row in rows)+'</table><img src="saved_readout_root/FIXED4_SOURCE_INPUT_DISPLAY.png"><ul>'+''.join('<li><a href="'+b+'">'+html.escape(a)+'</a></li>'for a,b in links)+'</ul>'
 with(V/'ROOT_READOUT_KO.html').open('x')as f:f.write(page)
 write(V/'ROOT_RESULTS_KO.json',dict(completed=True,PASS=True,phase='V214',code=bind(__file__),diagnostic_complete=bind(O/'COMPLETE.json'),summary=bind(O/'SUMMARY.json'),numeric_QA=bind(qa),visual_QA=bind(vis),figures_complete=bind(F/'FIGURES_COMPLETE.json'),models=ARMS,conditions=MODES,scalar_rows=73680,main_rows=rows,source_apparent_OOD_not_generalization=True,protected_new_image_row_score_or_label_artifacts=0,training=0,PnP=0,selected_model=None,adopted=False,fully_solved=False))
 write(F/'REPORT_BUILD_COMPLETE.json',dict(completed=True,PASS=True,code=bind(__file__),outputs=[bind(V/n)for n in ('ROOT_RESULTS_KO.json','ROOT_REPORT_KO.md','PORTABLE_REPORT_KO.md','ROOT_READOUT_KO.html')],new_metric_fit_or_array_model_access=0))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--figures',action='store_true');ap.add_argument('--report',action='store_true');a=ap.parse_args();assert a.figures!=a.report
 figures()if a.figures else report()
