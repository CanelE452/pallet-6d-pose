"""Visualize fixed saved Source CF responses without any new inference or selection."""
from pathlib import Path
import hashlib,html,json,os,stat
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).absolute().parent
H=R/'V254_ROOT_SAVED_CF_RESPONSE'
D=H/'saved_cf_response_v1'
FAMS=('SELF_SCENE15','BEFORE_ANCHOR4')
SHAMS=('NEAR_SHAM','FAR_SHAM')
MODELS=('BASE','CONTROL','TREAT')

def safe(p):
 p=Path(p)
 if not p.is_absolute() or '..' in p.parts or any(w in str(p).lower() for w in ('v112','cache','.pyc')):raise ValueError(str(p))
 for a in [*reversed(p.parents),p]:
  try:t=os.lstat(a)
  except FileNotFoundError:continue
  if stat.S_ISLNK(t.st_mode):raise ValueError('Artifact symlink')
 return p
def raw(p):return safe(p).read_bytes()
def load(p):return json.loads(raw(p))
def write(p,s):
 with safe(p).open('xb') as f:f.write(s.encode())
def mdtable(headers,rows):
 return '| '+' | '.join(headers)+' |\n| '+' | '.join('---'for _ in headers)+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)
def htable(headers,rows):
 return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>'for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>'for x in r)+'</tr>'for r in rows)+'</table></div>'
def rate(a,n):return 'NA' if not n else f'{a}/{n} ({100*a/n:.1f}%)'
def num(x):return 'NA' if x is None else f'{x:.4f}'

def run():
 c=load(D/'COMPLETE.json');assert c['completed'] and c['PASS']
 b=c['outputs']['SUMMARY.json'];sb=raw(D/'SUMMARY.json')
 assert hashlib.sha256(sb).hexdigest()==b['sha256'] and len(sb)==b['size_bytes']
 s=json.loads(sb);ov=s['overview']
 def row(f,sh,m):return next(r for r in ov if r['augmentation_family']==f and r['sham']==sh and r['model']==m)
 support=[];responses=[]
 for f in FAMS:
  for sh in SHAMS:
   a=row(f,sh,'BASE')
   support.append([f,sh,a['all_candidates'],a['lineage_PASS'],a['input_exact_PASS'],a['original_D_E_D_labels'],a['exact_scored_triplets_any_role'],a['primary_FIT_D_E_D']])
   for m in MODELS:
    a=row(f,sh,m);n=a['primary_FIT_D_E_D']
    responses.append([f,sh,m,n,rate(a['predicted_D_E_D'],n),rate(a['cover_sham_positive'],n),num(a['cover_before_median']),num(a['sham_before_median']),num(a['cover_sham_median'])])
 fig,ax=plt.subplots(figsize=(12,5.6),constrained_layout=True)
 labels=[f+' / '+sh for f in FAMS for sh in SHAMS];x=np.arange(4);width=.14
 cols=[('all_candidates','All candidate triples'),('lineage_PASS','Declared lineage'),('input_exact_PASS','Exact original inputs'),('original_D_E_D_labels','Original D/E/D labels'),('exact_scored_triplets_any_role','Exact + saved scores, all roles'),('primary_FIT_D_E_D','Primary FIT D/E/D')]
 for i,(key,label) in enumerate(cols):
  vals=[row(f,sh,'BASE')[key] for f in FAMS for sh in SHAMS]
  bars=ax.bar(x+(i-2.5)*width,vals,width,label=label)
  ax.bar_label(bars,fontsize=7,padding=2)
 ax.set_xticks(x,labels,rotation=10,ha='right');ax.set_ylabel('Number of candidate triplets')
 ax.set_title('V254 support axes: separate checks, not a sequential funnel\nNEAR/FAR share BEFORE and COVER; counts are correlated')
 ax.legend(fontsize=8,ncol=3);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
 fig.savefig(safe(H/'PAIR_SUPPORT.png'),dpi=160);plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(13.5,8.5),constrained_layout=True)
 for ax,(f,sh) in zip(axes.flat,[(f,sh)for f in FAMS for sh in SHAMS]):
  boxes=[]
  for m in MODELS:
   rr=row(f,sh,m)
   for key,label in [('cover_before','C-B'),('sham_before','S-B'),('cover_sham','C-S')]:
    if rr['primary_FIT_D_E_D']:
     boxes.append({'label':m+'\n'+label,'whislo':rr[key+'_p10'],'q1':rr[key+'_p25'],'med':rr[key+'_median'],'q3':rr[key+'_p75'],'whishi':rr[key+'_p90'],'fliers':[]})
  if boxes:ax.bxp(boxes,showfliers=False)
  else:ax.text(.5,.5,'No eligible primary FIT triplets',transform=ax.transAxes,ha='center')
  ax.axhline(0,color='black',linewidth=.7);ax.grid(axis='y',alpha=.15)
  ax.set_title(f+' / '+sh+' / n='+str(row(f,sh,'BASE')['primary_FIT_D_E_D']));ax.set_ylabel('Within-model selected E-logit difference')
 fig.suptitle('Saved frozen responses: p25/median/p75 boxes; p10/p90 whiskers\nRaw logit scales differ across models; this is not a calibrated cross-model score ranking')
 fig.savefig(safe(H/'SAVED_CF_RESPONSE.png'),dpi=160);plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
 patterns=('D/D/D','D/D/E','D/E/D','D/E/E','E/D/D','E/D/E','E/E/D','E/E/E')
 colors=plt.get_cmap('tab20').colors
 for ax,(f,sh) in zip(axes.flat,[(f,sh)for f in FAMS for sh in SHAMS]):
  bottom=np.zeros(3)
  for i,p in enumerate(patterns):
   vals=np.array([row(f,sh,m)['primary_prediction_sequences'].get(p,0)for m in MODELS])
   ax.bar(MODELS,vals,bottom=bottom,label=p,color=colors[(2*i)%len(colors)]);bottom+=vals
  ax.set_title(f+' / '+sh);ax.set_ylabel('Primary FIT triplets');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
  if not bottom.any():ax.text(.5,.5,'No eligible primary FIT triplets',transform=ax.transAxes,ha='center')
 handles,lab=axes.flat[0].get_legend_handles_labels();fig.legend(handles,lab,loc='lower center',ncol=8,fontsize=9)
 fig.suptitle('Fixed threshold 0: BEFORE / COVER / SHAM prediction patterns\nD/E/D is the complete expected pattern; training-set behavior only')
 fig.savefig(safe(H/'FIXED_PREDICTION_PATTERNS.png'),dpi=160,bbox_inches='tight');plt.close(fig)
 census=s['census']
 paras=[
  '첨부 V241은 V240까지의 검토입니다. 이후 분기·참조·생성 쌍 관련 진단과 학습이 추가됐으므로, 문서의 미실행 목록을 현재 상태처럼 다시 쓰지 않습니다. 이번 V254는 새 모델을 학습하지 않고 기존 V129의 전체 장면 가림 대조와 V252 세 모델의 저장 점수를 연결해 행동을 확인한 결과입니다.',
  f"원 Source {census['all_original_rows']:,}행과 {census['all_original_groups']:,}개 입력 그룹을 먼저 검산했습니다. 모든 그룹의 원 8개 slot을 확인했고, 별칭 행을 제거하지 않았습니다. SYN 7,040행의 masked {census['SYN_masked_rows_retained']}행과 전체 masked {census['masked_rows_retained']:,}행은 E나 성공 보류로 바꾸지 않았습니다. 후보 triplet {s['candidate_count_including_missing']:,}개를 누락·입력 불일치·라벨 상태와 함께 보존했습니다.",
  'SELF_SCENE15는 각 장면 자체의 초기 추론 좌표를 사용하고, BEFORE_ANCHOR4는 원 BEFORE 초기 좌표를 다른 variant에도 유지합니다. 두 가족을 섞지 않았습니다. q/L·가용성뿐 아니라 원 q8/L8/availability8·box·known condition32의 정확한 일치를 요구했습니다. 좌표를 보정하거나 근처 사례로 대체하지 않았습니다.',
  '표의 primary는 모든 구성원이 FIT이며, 원 정답이 BEFORE D·COVER E·SHAM D이고, 선언된 대상 관계·원 입력·저장 점수 가용성을 모두 만족한 경우입니다. 실제 판정 D/E/D와 COVER−SHAM 점수 양의 비율을 각각 표시합니다. 방향만 맞아도 원 보임과 대조 보임을 함께 맞힌 것은 아닙니다. NEAR와 FAR는 BEFORE/COVER를 공유하므로 두 수를 독립 사건으로 더하지 않습니다.',
  '합성 정답은 선택된 native record의 첫 witness에 대한 상태입니다. 전체 대상 팔레트가 단독으로 보일 때와 foreign BOX가 앞을 막을 때를 구분하며, 자기 기하가 가린 S는 E에 합치지 않습니다. 이 검사는 q 한 픽셀의 소유자·부위 전체 가시성·실사 팔레트의 사람 정답을 새로 증명하지 않습니다. 공개 YCB의 associated-r 보조 정답도 이번 합성 triplet 성능에 합치지 않았습니다.',
  '이는 학습에 쓰인 Source FIT 장면에서의 저장 행동 진단입니다. 성능 개선 실험이나 독립 일반화 검증이 아니며, 최신 모델을 채택하거나 현재 평가에 진입한 결과도 아닙니다. 세 모델의 raw logit 크기는 서로 보정된 확률이 아니므로, 크기 차이만으로 모델 순위를 정하지 않습니다.',
  '새 학습·모델/encoder/체크포인트 forward·RGB/원 GT 해독·bank 적합·문턱 선택·PnP는 없습니다. 여섯 Source 컨테이너 전체를 읽고 hash했으므로 DEV 메타데이터와 저장 출력도 포함됩니다. 이는 고정 census와 진단용이며 새 설정 선택에 사용하지 않았습니다. 사용자가 첨부한 검토문 전체 바이트는 읽고 hash했지만, 그 안의 과거 현재/held 평가 집계를 새 선택 근거로 사용하지 않았습니다.',
 ]
 sh=['원 입력 가족','SHAM','전체 후보','선언 관계','입력 exact','원 D/E/D','exact+점수(모든 분할)','primary FIT']
 rh=['원 입력 가족','SHAM','모델','FIT n','예측 D/E/D','COVER>SHAM','median C−B','median S−B','median C−S']
 links=[('V241 검토와 후속 기록의 대조','artifacts/V254_REVIEW_RECONCILIATION_A/RECONCILIATION_DRAFT_KO.md'),('사전 계약','artifacts/V254_ROOT_SAVED_CF_RESPONSE/PROTOCOL_BEFORE_SOURCE_PAIR_ROWS.json'),('모든 후보와 누락','artifacts/V254_ROOT_SAVED_CF_RESPONSE/saved_cf_response_v1/CF_TRIPLETS.csv'),('세 모델의 전체 저장 반응','artifacts/V254_ROOT_SAVED_CF_RESPONSE/saved_cf_response_v1/RESPONSE_MODEL_ROWS.csv'),('원 주소 0–7 포함 전체 집계','artifacts/V254_ROOT_SAVED_CF_RESPONSE/saved_cf_response_v1/RESPONSE_SUMMARY.csv'),('원 SYN 7,040행과 masked 상태','artifacts/V254_ROOT_SAVED_CF_RESPONSE/saved_cf_response_v1/SYN7040_ROWS.csv'),('기계 판독 요약','artifacts/V254_ROOT_SAVED_CF_RESPONSE/saved_cf_response_v1/SUMMARY.json'),('독립 산술 QA','artifacts/V254_PREVIOUS_PROGRESS_AND_CF_QA_C/actual_review_v1/REVIEW.json'),('공개 저장 입력 재현 코드','artifacts/V254_PUBLIC_REPLAY_ROOT/verify_saved_cf_response.py'),('공개 재현 입력 매핑','runtime_dependencies.json')]
 images=[('PAIR_SUPPORT.png','대조 가능한 사례의 분모와 제외 축'),('SAVED_CF_RESPONSE.png','동일 query의 가림 점수 변화'),('FIXED_PREDICTION_PATTERNS.png','원 보임·실제 가림·대조 보임의 판정 패턴')]
 md='# V254 — 저장된 동일 query의 COVER·SHAM 반응\n\n'+'\n\n'.join(paras)+'\n\n## 전체 후보와 primary 분모\n\n'+mdtable(sh,support)+'\n## 고정 세 모델의 FIT 행동\n\n'+mdtable(rh,responses)+'\n'+''.join('\n!['+a+']('+p+')\n'for p,a in images)+'\n## 원 행·코드·재현 자료\n\n'+'\n'.join('- ['+a+']('+p+')'for a,p in links)+'\n'
 write(H/'PUBLIC_README_KO.md',md)
 page='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V254 저장 동일 query 가림 대조 반응</title><style>body{font:16px/1.7 system-ui;max-width:1280px;margin:24px auto;padding:20px;color:#18313c;background:#f5f8fa}.scroll{overflow:auto}table{border-collapse:collapse;font-size:12px;background:white}td,th{border:1px solid #cedce2;padding:7px;white-space:nowrap}img{max-width:100%}a{color:#116685}</style><h1>V254: 저장된 동일 query의 가림 대조 반응</h1>'+''.join('<p>'+html.escape(x)+'</p>'for x in paras)+'<h2>분모와 제외 축</h2>'+htable(sh,support)+'<h2>고정 세 모델의 FIT 행동</h2>'+htable(rh,responses)+''.join('<h2>'+a+'</h2><img src="'+p+'" alt="'+a+'">'for p,a in images)+'<h2>원 행과 재현 자료</h2><ul>'+''.join('<li><a href="'+p+'">'+html.escape(a)+'</a></li>'for a,p in links)+'</ul></html>\n'
 write(H/'ROOT_READOUT_KO.html',page)
 write(H/'PUBLIC_READOUT_TABLES.json',json.dumps({'support_headers':sh,'support_rows':support,'response_headers':rh,'response_rows':responses,'links':links,'new_model_training':0},ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'completed':True,'PASS':True,'support_rows':len(support),'response_rows':len(responses),'charts':3}))

if __name__=='__main__':run()
