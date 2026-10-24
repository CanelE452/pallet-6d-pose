"""Root-entry saved-only two-baseline evaluation and fixed source visualization.

No model/encoder imports. GT and crop/prediction payloads are read only after
both completed prediction seals have been validated inside the actual entry.
"""
from pathlib import Path
import argparse,csv,hashlib,html,json,sys,traceback
import numpy as np
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;V=H.parent
PROTOCOL=H/'SAVED_EVALUATION_PROTOCOL_V2.json';OUT=V/'saved_source_evaluation_v2'
MODELS=('v61','C16');VARIANTS=('BEFORE','COVER','SHAM')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def check(b):assert bind(b['path'])==b,b['path'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def arrsha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def ratio(a,b):return float(a/b)if b else None
def metrics(pred,y,mask,states=None):
 mask=np.asarray(mask,bool);known=mask&(y>=0)
 cm=np.array([[np.sum(known&(y==r)&(pred==c))for c in (0,1,-1)]for r in (0,1)],np.int64)
 out={'CM':cm.tolist(),'rows':['D','EXTERNAL_E'],'columns':['D','E','prediction_U'],'all_rows':int(mask.sum()),'known_DE':int(known.sum()),'D_precision':ratio(cm[0,0],cm[:,0].sum()),'D_recall':ratio(cm[0,0],cm[0].sum()),'E_precision':ratio(cm[1,1],cm[:,1].sum()),'E_recall':ratio(cm[1,1],cm[1].sum()),'known_prediction_U':int(cm[:,2].sum()),'known_prediction_U_rate':ratio(cm[:,2].sum(),cm.sum())}
 if states is not None:out['out_of_external_task_typed_predictions']={s:{'N':int((mask&(states==s)).sum()),'D':int((mask&(states==s)&(pred==0)).sum()),'E':int((mask&(states==s)&(pred==1)).sum()),'prediction_U':int((mask&(states==s)&(pred==-1)).sum())}for s in ('S','U')}
 return out
def report(pred,y,states,rows,gt):
 allmask=np.ones(len(rows),bool);style=np.array([r['style']for r in rows]);variant=np.array([r['variant']for r in rows]);slot=np.array([r['logical_slot']for r in rows]);error=np.array([np.nan if g['error_px']is None else g['error_px']for g in gt])
 bins={'ERROR_ZERO':np.isfinite(error)&(np.abs(error)<1e-9),'ERROR_3PX':np.isfinite(error)&(np.abs(error-3)<1e-9),'ERROR_GT3PX':np.isfinite(error)&(error>3+1e-9),'ERROR_MISSING':~np.isfinite(error)}
 assert np.array(list(bins.values())).sum(0).tolist()==[1]*len(rows)
 return {'ALL':metrics(pred,y,allmask,states),'by_style':{s:metrics(pred,y,style==s,states)for s in sorted(set(style))},'by_variant':{s:metrics(pred,y,variant==s,states)for s in VARIANTS},'by_error':{s:metrics(pred,y,m,states)for s,m in bins.items()},'by_slot':{str(s):metrics(pred,y,slot==s,states)for s in range(8)},'style_variant':{s:{v:metrics(pred,y,(style==s)&(variant==v),states)for v in VARIANTS}for s in sorted(set(style))}}
def triplets_and_flips(pred,hidden,rows,states):
 groups={}
 for i,r in enumerate(rows):groups.setdefault(r['query_group_id'],[]).append(i)
 native1=[]
 for gid,ix in groups.items():
  rr={rows[i]['variant']:i for i in ix};assert set(rr)==set(VARIANTS)
  ii=[rr[v]for v in VARIANTS]
  for k in ('query_xy','initial_points','L','context_box_xyxy'):assert all(rows[i][k]==rows[ii[0]][k]for i in ii)
  if rows[ii[0]]['logical_slot']!=1:continue
  assert states[ii].tolist()==['D','E','D']
  zz=hidden[ii]
  native1.append({'query_group_id':gid,'style':rows[ii[0]]['style'],'jitter_index':rows[ii[0]]['jitter_index'],'offset_xy':rows[ii[0]]['offset_xy'],'occurrences':ii,'prediction':pred[ii].tolist(),'all3_D_E_D':bool(np.array_equal(pred[ii],[0,1,0])),'hidden_direction_logit_BEFORE_COVER_SHAM':zz.tolist(),'cover_minus_before':float(zz[1]-zz[0]),'sham_minus_before':float(zz[2]-zz[0]),'cover_minus_sham':float(zz[1]-zz[2])})
 assert len(native1)==18
 flips=[]
 for style in sorted({r['style']for r in rows}):
  for variant in VARIANTS:
   for slot in range(8):
    ix=sorted([i for i,r in enumerate(rows)if r['style']==style and r['variant']==variant and r['logical_slot']==slot],key=lambda i:rows[i]['jitter_index']);assert len(ix)==9
    p=pred[ix].reshape(3,3);z=hidden[ix];edges=[(p[y,x],p[y,x+1])for y in range(3)for x in range(2)]+[(p[y,x],p[y+1,x])for y in range(2)for x in range(3)]
    flips.append({'style':style,'variant':variant,'slot':slot,'occurrences':ix,'predictions_jitter_order':p.ravel().tolist(),'count_D':int((p==0).sum()),'count_E':int((p==1).sum()),'count_prediction_U':int((p==-1).sum()),'different_from_zero_offset':int((p!=p[1,1]).sum()),'DE_flip_edges_of_12_horizontal_vertical':int(sum(a>=0 and b>=0 and a!=b for a,b in edges)),'U_transition_edges_of_12':int(sum((a==-1)!=(b==-1)for a,b in edges)),'hidden_direction_logit_min':float(z.min()),'hidden_direction_logit_max':float(z.max())})
 return {'N':18,'all3_correct':sum(r['all3_D_E_D']for r in native1),'by_style':{s:{'N':9,'all3_correct':sum(r['all3_D_E_D']for r in native1 if r['style']==s)}for s in sorted({r['style']for r in rows})},'rows':native1,'jitter_flips':flips}
def authorize(entry):
 e=read(entry);assert e['approved_saved_source_evaluation']is True and e['source_only']is True
 assert all(e[k]is False for k in ('model_forward','fit','current223','V112','PnP','automatic_retry'))
 for k in ('code','protocol','generated','v61_prediction_seal','C16_prediction_seal'):check(e[k])
 assert e['code']==bind(__file__)and e['protocol']==bind(PROTOCOL)
 p=read(PROTOCOL);assert p['code']==e['code']and p['generated']==e['generated']and read(e['generated']['path'])['PASS']
 s61=read(e['v61_prediction_seal']['path']);s16=read(e['C16_prediction_seal']['path'])
 assert s61['completed']and s61['PASS']and s16['completed']and s16['PASS']
 assert s61['input_rows']==s16['input_manifest']==e['input_rows'] and s61['input_crops']==e['input_crops']
 assert s61['input_complete']==s16['crop_complete']==e['crop_complete']and s61['render_complete']==s16['render_complete']
 # Both prediction seals passed before any GT hash/body or data payload access.
 for k in ('input_rows','input_crops','crop_complete','GT_rows'):check(e[k])
 assert s61['occurrences']==s16['rows']==432
 for s in (s61,s16):check(s['predictions'])
 assert not OUT.exists(),'Preserve earlier attempt; no automatic rerun.'
 return e,p,s61,s16
def load_predictions(s61,s16,rows):
 with np.load(s61['predictions']['path'],allow_pickle=False)as f:a={k:np.array(f[k],copy=True)for k in ('raw_visible_logit','prediction_D0_E1_Uminus1','query_available','occurrence_index','query_id','query_xy','L')}
 with np.load(s16['predictions']['path'],allow_pickle=False)as f:b={k:np.array(f[k],copy=True)for k in ('query_logit','prediction','query_available','global_occurrence_index','occurrence_id')}
 ids=np.array([r['id']for r in rows]);q=np.array([r['query_xy']for r in rows]);L=np.array([r['L']for r in rows]);order=np.arange(432)
 assert np.array_equal(a['occurrence_index'],order)and np.array_equal(b['global_occurrence_index'],order)and np.array_equal(a['query_id'],ids)and np.array_equal(b['occurrence_id'],ids)
 assert np.array_equal(a['query_xy'],q)and np.array_equal(a['L'],L)
 pred={'v61':a['prediction_D0_E1_Uminus1'],'C16':b['prediction']};z={'v61':a['raw_visible_logit'],'C16':b['query_logit']};available={'v61':a['query_available'],'C16':b['query_available']}
 for m in MODELS:assert pred[m].shape==z[m].shape==available[m].shape==(432,)and np.isfinite(z[m]).all()
 assert np.array_equal(pred['v61'],np.where(available['v61'],np.where(z['v61']>=0,0,1),-1))
 assert np.array_equal(pred['C16'],np.where(available['C16'],np.where(z['C16']>=0,1,0),-1))
 for i,r in enumerate(rows):
  av=r['original_input_availability'];assert available['v61'][i]==(av['continuous']and av['nearest']and av['ROI_any_valid'])and available['C16'][i]==r['query_available']
 return pred,z,available
def font(size):
 from PIL import ImageFont
 return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)
def crop_panel(crop,r,g,pr,z,width=404,height=270):
 from PIL import Image,ImageDraw
 im=Image.new('RGB',(width,height),'white');d=ImageDraw.Draw(im);raw=Image.fromarray(crop).resize((192,192),Image.Resampling.NEAREST);marked=Image.fromarray(crop.copy()).resize((192,192),Image.Resampling.NEAREST);md=ImageDraw.Draw(marked)
 def cross(x,y,color):
  if 0<=x<96 and 0<=y<96:
   xx=(x+.5)*2-.5;yy=(y+.5)*2-.5;md.line((xx-6,yy,xx+6,yy),fill=color,width=2);md.line((xx,yy-6,xx,yy+6),fill=color,width=2)
 cross(47.5,47.5,'yellow')
 if g['reference_xy']is not None:
  ref=(np.array(g['reference_xy'])-r['query_xy'])*96/r['L']+47.5;cross(*ref,'cyan')
 im.paste(raw,(8,22));im.paste(marked,(204,22))
 d.text((8,3),'RAW saved RGB96',font=font(12),fill='black');d.text((204,3),'DISPLAY: q yellow / r cyan',font=font(11),fill='black')
 own=g['q_nearest_pixel_owner_diagnostic'];pix='U'if own['ambiguity']or not own['in_frame']else('BG'if own['part_owner_id']<0 else('FOREIGN'if own['part_owner_id']==999 else'TARGET'))
 text=[f"row{r['occurrence_index']} slot{r['logical_slot']} {g['raw_state']} / qpixel {pix}",f"L={r['L']:.3f} native px; error={g['error_px']:.3f}px",f"v61 {pr['v61']} (visible z={z['v61']:+.3f}) | C16 {pr['C16']} (hidden z={z['C16']:+.3f})"]
 for j,t in enumerate(text):d.text((8,219+16*j),t,font=font(12),fill='#152f43')
 return im
def visuals(rows,gt,crops,pred,z):
 from PIL import Image,ImageDraw
 root=OUT/'figures';root.mkdir();styles=sorted({r['style']for r in rows});palette={0:'D',1:'E',-1:'U'};selected=[]
 for style in styles:
  selected += sorted([i for i,r in enumerate(rows)if r['style']==style and r['variant']=='BEFORE'and r['jitter_index']==4],key=lambda i:rows[i]['logical_slot'])
 assert len(selected)==16
 board=Image.new('RGB',(4*412,70+4*286),'#eaf0f5');d=ImageDraw.Draw(board)
 d.text((12,8),'Fixed zero-offset BEFORE: all8 corners x2 NEW registered assets | RAW and marked-copy panels',font=font(18),fill='#102f43');d.text((12,34),'Source emulation, one camera. Region state is not centre-pixel material. Marks never enter models.',font=font(15),fill='#334e60')
 for k,i in enumerate(selected):board.paste(crop_panel(crops[i],rows[i],gt[i],{m:palette[int(pred[m][i])]for m in MODELS},{m:float(z[m][i])for m in MODELS}),(6+(k%4)*412,65+(k//4)*286))
 path=root/'ZERO_OFFSET_BEFORE_ALL16.png';board.save(path);files=[bind(path)]
 for style in styles:
  board=Image.new('RGB',(3*412,68+9*286),'#eaf0f5');d=ImageDraw.Draw(board);d.text((10,8),style+' native1: all9 fixed jitters, columns BEFORE / COVER / SHAM',font=font(17),fill='#102f43');d.text((10,34),'Same q/L across columns. Conditional registered geometry D/E/D; q/r marks display only.',font=font(14),fill='#334e60')
  for j in range(9):
   for k,v in enumerate(VARIANTS):
    ix=[i for i,r in enumerate(rows)if r['style']==style and r['logical_slot']==1 and r['jitter_index']==j and r['variant']==v];assert len(ix)==1;i=ix[0]
    board.paste(crop_panel(crops[i],rows[i],gt[i],{m:palette[int(pred[m][i])]for m in MODELS},{m:float(z[m][i])for m in MODELS}),(6+k*412,64+j*286))
  path=root/(style+'_NATIVE1_ALL9_TRIPLETS.png');board.save(path);files.append(bind(path))
 return files,selected
def charts(reports):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(12,3.5))
 for ax,m in zip(axes,('v61','C16','Q_PIXEL_ORACLE')):
  cm=np.array(reports[m]['ALL']['CM']);ax.imshow(cm,cmap='Blues',vmin=0,vmax=max(1,cm.max()));ax.set_xticks(range(3),['D','E','U']);ax.set_yticks(range(2),['D','external E']);ax.set_title(m+' / structural-region target')
  for i,j in np.ndindex(cm.shape):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='black')
  ax.set_xlabel('Prediction');ax.set_ylabel('Registered region teacher')
 fig.tight_layout();path=OUT/'figures/SEMANTIC_DE_CONFUSION.png';fig.savefig(path,dpi=150);plt.close(fig);return bind(path)
def html_report(result,figures,seals):
 def esc(s):return html.escape(str(s))
 def cell(x):return '—'if x is None else f'{100*x:.2f}%'
 table=''
 for m in ('v61','C16','Q_PIXEL_ORACLE'):
  v=result['semantic_reports'][m]['ALL'];table+=f"<tr><td>{m}</td><td><code>{esc(v['CM'])}</code></td><td>{cell(v['D_precision'])}</td><td>{cell(v['D_recall'])}</td><td>{cell(v['E_precision'])}</td><td>{cell(v['E_recall'])}</td><td>{cell(v['known_prediction_U_rate'])}</td></tr>"
 imgs=''.join(f'<figure><a href="figures/{Path(b["path"]).name}"><img src="figures/{Path(b["path"]).name}"></a><figcaption>{esc(Path(b["path"]).stem)}</figcaption></figure>'for b in figures)
 links=''.join(f'<li><a href="{html.escape(Path(b["path"]).as_uri())}">{esc(k)}</a> <code>{b["sha256"]}</code></li>'for k,b in seals.items())
 diag=result['semantic_D_false_E_by_q_pixel']
 text=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>V128 고정 합성 source 진단</title><style>body{{font-family:Arial,"Noto Sans KR",sans-serif;max-width:1450px;margin:28px auto;padding:0 24px;color:#153043;line-height:1.6}}h1{{font-size:30px}}.note{{background:#fff0d5;padding:16px;border-left:5px solid #b46a10}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border:1px solid #cdd9e2;text-align:left}}img{{max-width:100%;height:auto}}figure{{margin:25px 0}}code{{overflow-wrap:anywhere}}li{{margin:8px 0}}a{{color:#075b92}}</style>
 <h1>새 등록 기하 2종·카메라 1개: 고정 모델 source 진단</h1><p class="note">이 결과는 실제 팔레트 전이 성능의 증거가 아닙니다. 두 새 절차적 자산, 같은 시점, 고정 외부 물체의 6개 장면과 432개 query를 평가했습니다. 학습·모델 선택·임계값 변경은 없습니다. 기존 실사 C16 집계는 D 1137/1147, 외부 E 35/90으로 그대로이며, 이 보고서는 현재 실사 파일을 읽지 않았습니다.</p>
 <p>정답은 등록된 모서리와 인접 구조 패치의 연속 ray witness에 대한 D/E/S/U입니다. q 픽셀에 팔레트 물질이 있는지는 별도 진단입니다. 자기 가림 S와 teacher U는 외부 E 음성/양성으로 바꾸지 않습니다. 원본 RGB96과 표시용 q/r 마크를 분리했습니다. q는 그대로이며 r은 이 새 합성 자산의 알려진 투영 좌표입니다. 실제 initializer 또는 실사 좌표 정확도 인증을 뜻하지 않습니다.</p>
 <h2>모든 432개 보존</h2><p>teacher 상태: {esc(result['teacher_counts'])}. 입력 가용성: {esc(result['availability'])}. 두 모델의 실제 입력은 다릅니다. v61은 같은 전체 RGB에서 새로 추출한 P3 ROI65, C16은 DINO T/Q/F/C/P1928을 받습니다. RGB96의 시각화가 같다는 이유로 네트워크 입력이 같다고 주장하지 않습니다.</p>
 <h2>구조 cue D/E 평가</h2><p>행 D/E, 열 예측 D/E/U. recall 분모에 예측 U를 포함하며, S/U typed 판정은 JSON에 별도로 보존했습니다.</p><table><tr><th>모델</th><th>CM</th><th>D precision</th><th>D recall</th><th>E precision</th><th>E recall</th><th>known 예측 U</th></tr>{table}</table>
 <p>Q_PIXEL_ORACLE은 q의 가장 가까운 픽셀이 target이면 D, foreign/background면 E인 oracle입니다. 구조 cue 평가에서 기대 CM [[84,114,0],[0,18,0]]이며, 중심 물질과 기대 구조 cue의 의미 차이를 보여줍니다. 별도의 q-pixel-objective 표는 중심 픽셀 목적에 대한 설명용이며 구조 teacher를 대체하지 않습니다.</p>
 <p>구조 D를 E로 판단한 경우를 q 픽셀로 분해하면 {esc(diag)}입니다. 배경 q에서 E가 나오는 패턴은 두 목적 정의의 차이와 부합할 수 있습니다. 그러나 q에 target이 있는 D 누락과 외부 E 누락도 별도로 남으므로 v61의 모든 실패를 목표 정의 차이 하나로 설명할 수 없습니다. 이는 저장 결과의 기술적 분해이며 인과 증명이나 실사 비율 추정이 아닙니다.</p>
 <h2>고정 native1 D→E→D triplet 18개</h2><p>{esc({m:{k:v for k,v in result['native1_triplets'][m].items()if k not in ('rows','jitter_flips')}for m in MODELS})}</p><p>모든 jitter의 before/cover/sham logit과 판정, 12개 수평·수직 이웃 edge의 판정 flip 수, zero-offset과 다른 판정 수를 JSON에 남겼습니다. 좋은 예시를 재선택하지 않았습니다.</p>
 {imgs}<h2>해석 범위</h2><ul><li>q를 옮겨도 동일 cue identity에 대한 region teacher는 고정되며, source 입력만 달라집니다. 주변 팔레트 물질과 실제 구조 cue 관측은 같지 않을 수 있습니다.</li><li>동일 카메라·2종 자산·작은 등록 패치·외부 cuboid라는 강한 상관과 제한이 있습니다. 432개는 독립 실사 432장의 표본이 아닙니다.</li><li>연속 37 witness와 반올림 픽셀은 서로 다른 ray입니다. 픽셀 aliasing을 라벨 수정이나 학습 승인으로 사용하지 않았습니다.</li><li>이 고정 평가로 checkpoint/threshold/정규화/학습 자료를 선택하지 않습니다. 실사 외부 가림 목표는 미해결입니다.</li></ul>
 <h2>전체 저장 결과·근거</h2><p><a href="RESULTS.json">전체 집계 JSON</a> · <a href="ALL432_SCORES.csv">모든 432개 scores</a> · <a href="ALL_FAILURE_IDS.json">모든 D/E 실패 ID</a></p><ul>{links}</ul></html>'''
 (OUT/'READOUT_KO.html').write_text(text)
def run(entry):
 e,p,s61,s16=authorize(entry)
 packet=read(e['input_rows']['path']);truth=read(e['GT_rows']['path']);rows=packet['rows'];gt=truth['rows']
 assert packet['completed']and truth['completed']and truth['input_row_identity']==e['input_rows']and len(rows)==len(gt)==432
 assert [r['occurrence_index']for r in rows]==[g['occurrence_index']for g in gt]==[r['rgb96_row']for r in rows]==list(range(432))and [r['id']for r in rows]==[g['id']for g in gt]
 pred,z,available=load_predictions(s61,s16,rows);states=np.array([g['raw_state']for g in gt]);y=np.array([g['external_target_D0_E1_else_minus1']for g in gt],np.int8)
 assert np.array_equal(y,np.where(states=='D',0,np.where(states=='E',1,-1)))and {s:int((states==s).sum())for s in 'DESU'}=={'D':198,'E':18,'S':54,'U':162}
 pixel=[]
 for g in gt:
  x=g['q_nearest_pixel_owner_diagnostic'];pixel.append(-1 if not x['in_frame']or x['ambiguity']else(0 if 0<=x['part_owner_id']<999 else 1))
 pixel=np.array(pixel,np.int8)
 reports={m:report(pred[m],y,states,rows,gt)for m in MODELS};reports['Q_PIXEL_ORACLE']=report(pixel,y,states,rows,gt)
 assert reports['Q_PIXEL_ORACLE']['ALL']['CM']==[[84,114,0],[0,18,0]]
 hidden={'v61':-z['v61'],'C16':z['C16']};triples={m:triplets_and_flips(pred[m],hidden[m],rows,states)for m in MODELS}
 pixelreports={m:report(pred[m],pixel,states,rows,gt)for m in MODELS}
 OUT.mkdir();write(OUT/'STARTED.json',{'entry':bind(entry),'saved_only':True})
 try:
  with np.load(e['input_crops']['path'],allow_pickle=False)as f:crops=np.array(f['rgb96'],copy=True);valid=np.array(f['valid96'],copy=True);assert np.array_equal(f['occurrence_index'],np.arange(432))
  assert crops.shape==(432,96,96,3)and crops.dtype==np.uint8 and valid.shape==(432,96,96)and valid.dtype==bool
  for i,r in enumerate(rows):assert arrsha(crops[r['rgb96_row']])==r['rgb96_sha256']and arrsha(valid[r['rgb96_row']])==r['valid96_sha256']
  own=np.array([g['q_nearest_pixel_owner_diagnostic']['part_owner_id']if g['q_nearest_pixel_owner_diagnostic']['part_owner_id']is not None else -999 for g in gt])
  diag={m:{'all_semantic_D_predicted_E':int(((states=='D')&(pred[m]==1)).sum()),'q_background':int(((states=='D')&(pred[m]==1)&(own==-1)).sum()),'q_target':int(((states=='D')&(pred[m]==1)&(own>=0)&(own<999)).sum()),'q_foreign':int(((states=='D')&(pred[m]==1)&(own==999)).sum()),'external_E_predicted_D':int(((states=='E')&(pred[m]==0)).sum())}for m in MODELS}
  result={'completed':True,'source_only':True,'code':e['code'],'entry':bind(entry),'teacher_counts':{s:int((states==s).sum())for s in 'DESU'},'rows':432,'availability':{m:{'available':int(available[m].sum()),'prediction_U':int((~available[m]).sum())}for m in MODELS},'semantic_reports':reports,'q_pixel_objective_reports':pixelreports,'q_pixel_objective_scope':'Nearest target presence D versus foreign/background E, not expected structural cue identity; out-of-frame/tie U. This table is distinct from the semantic D/E task.','semantic_D_false_E_by_q_pixel':diag,'native1_triplets':triples,'zero_offset_expected_source_inputs16':True,'current_aggregate_reference_only':{'D_correct':1137,'D_N':1147,'external_E_correct':35,'external_E_N':90,'current_file_access':False},'new_model_forward_fit_current_V112_PnP':0,'no_training_admission_or_selection':True}
  failures={m:[{'occurrence_index':i,'id':rows[i]['id'],'state':str(states[i]),'prediction':int(pred[m][i]),'query_xy':rows[i]['query_xy'],'L':rows[i]['L']}for i in range(432)if y[i]>=0 and pred[m][i]!=y[i]]for m in MODELS};write(OUT/'ALL_FAILURE_IDS.json',{'completed':True,'all_DE_failures':failures,'S_U_preserved_elsewhere_not_failures_for_DE':True})
  table=[]
  for i,r in enumerate(rows):
   table.append({'occurrence_index':i,'id':r['id'],'query_group_id':r['query_group_id'],'style':r['style'],'variant':r['variant'],'slot':r['logical_slot'],'jitter_index':r['jitter_index'],'dx':r['offset_xy'][0],'dy':r['offset_xy'][1],'qx':r['query_xy'][0],'qy':r['query_xy'][1],'L':r['L'],'reference_x':gt[i]['reference_xy'][0]if gt[i]['reference_xy']else None,'reference_y':gt[i]['reference_xy'][1]if gt[i]['reference_xy']else None,'error_px':gt[i]['error_px'],'raw_teacher_state':states[i],'external_target':int(y[i]),'q_pixel_D0_elseE1_Uminus1':int(pixel[i]),**{f'{m}_{k}':v for m in MODELS for k,v in [('raw_logit',float(z[m][i])),('prediction',int(pred[m][i])),('available',bool(available[m][i]))]}})
  with (OUT/'ALL432_SCORES.csv').open('x',newline='')as f:w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
  figs,selected=visuals(rows,gt,crops,pred,z);result['fixed_zero_offset_visual_indices']=selected;figs.insert(0,charts(reports));result['figures']=figs;write(OUT/'RESULTS.json',result)
  evidence={k:e[k]for k in ('input_rows','GT_rows','input_crops','crop_complete','v61_prediction_seal','C16_prediction_seal')};html_report(result,figs,evidence)
  import re,urllib.parse
  links=re.findall(r'(?:href|src)="([^"]+)"',(OUT/'READOUT_KO.html').read_text())
  for x in links:
   path=Path(urllib.parse.unquote(urllib.parse.urlparse(x).path))if x.startswith('file:')else OUT/x
   assert path.exists(),str(path)
  outputs=[bind(f)for f in sorted(OUT.rglob('*'))if f.is_file()]
  write(OUT/'EVALUATION_SEAL.json',{'completed':True,'PASS':True,'entry':bind(entry),'code':e['code'],'protocol':e['protocol'],'both_prediction_seals_completed_before_GT_access':True,'input_rows':e['input_rows'],'GT_rows':e['GT_rows'],'input_crops':e['input_crops'],'v61_prediction_seal':e['v61_prediction_seal'],'C16_prediction_seal':e['C16_prediction_seal'],'outputs':outputs,'all432_retained':True,'all_canonical_savedcrop_hashes_exact':True,'local_links_exist':len(links),'new_model_encoder_fit_current_V112_PnP':0,'human_visual_review_not_performed_by_script':True})
  print(json.dumps({'completed':True,'seal':bind(OUT/'EVALUATION_SEAL.json'),'semantic_CM':{m:reports[m]['ALL']['CM']for m in MODELS},'native1_all3':{m:triples[m]['all3_correct']for m in MODELS}}),flush=True)
 except BaseException:write(OUT/'FAILED_NO_RESTART.json',{'traceback':traceback.format_exc(),'automatic_retry':False});raise
def generated():
 pred=np.array([0,1,-1,0,1,1,-1,0]);y=np.array([0,0,0,1,1,1,-1,-1]);s=np.array(['D','D','D','E','E','E','S','U']);m=metrics(pred,y,np.ones(8,bool),s)
 assert m['CM']==[[1,1,1],[1,2,0]]and m['D_recall']==1/3 and m['E_precision']==2/3 and m['known_prediction_U_rate']==1/6 and m['out_of_external_task_typed_predictions']['S']['prediction_U']==1
 assert np.where(np.array([-0.,0.,-1.,1.])>=0,0,1).tolist()==[0,0,1,0]and np.where(np.array([-0.,0.,-1.,1.])>=0,1,0).tolist()==[1,1,0,1]
 rr=[];gg=[];pp=[];ss=[]
 for style in ('WOOD','POLY'):
  for v,st,p in zip(VARIANTS,('D','E','D'),(0,1,0)):
   for slot in range(8):
    for j in range(9):
     rr.append({'query_group_id':style+str(slot)+str(j),'style':style,'variant':v,'logical_slot':slot,'jitter_index':j,'offset_xy':[j%3-1,j//3-1],'query_xy':[1.,2.],'initial_points':[[1.,2.]]*8,'L':16.,'context_box_xyxy':[0.,0.,3.,4.]});pp.append(p);ss.append(st)
 t=triplets_and_flips(np.array(pp),np.array(pp,float),rr,np.array(ss));assert t['N']==18 and t['all3_correct']==18 and len(t['jitter_flips'])==48;json.dumps(t,allow_nan=False)
 checks={'CM_U_denominators_and_typed_S_U':True,'no_teacher_U_negative':True,'raw_sign_rules_explicit_opposite_polarities':True,'sameqL_triplet_join_by_group_not_rowoffset':True,'all432_and_both_seals_required_before_GT_hash':True,'no_model_encoder_import':True}
 write(H/'GENERATED_EVALUATION_CHECKS_V2.json',{'completed':True,'PASS':True,'code':bind(__file__),'checks':checks,'actual_GT_crop_predictions_model_current_V112_read':0})
def prepare():
 proof=H/'GENERATED_EVALUATION_CHECKS_V2.json';assert read(proof)['PASS']and read(proof)['code']==bind(__file__)
 write(PROTOCOL,{'completed':True,'code':bind(__file__),'generated':bind(proof),'input_schema':'Root-bound INPUT_ROWS.json/GT_ROWS.json rows432 and RGB96_INPUT_ONLY rgb96HWC/validHW. Exact schemas read from builder code only.','prediction_keys':{'v61':['raw_visible_logit','prediction_D0_E1_Uminus1','query_available'],'C16':['query_logit','prediction','query_available']},'GT_boundary':'Both completed/PASS prediction seals verified before GT hash/body or prediction/crop payload reads. Explicit root saved-eval entry.','output':str(OUT),'metrics':'ALL/style/variant/error0,error3,error>3/missing/slot; D/E CM2x3/P/R/U; S/U typed separately; centre-pixel objective separate; native1all18triples/logitdelta/jitterflips.','fixed_visuals':'zero-offset BEFORE all8x2=16 RAW+marked-copy panels; native1all9jitters x3variants for eachstyle. No case selection from model success.','coordinates':'Saved RGB96 unchanged. q at47.5; reference=(r-q)*96/L+47.5 display-only.','source_limits':'One camera,2new proceduralassets; correlated synthetic source and finitepatch teacher; not real transfer or coordinate correctness. CurrentC16 aggregate1137/1147 and35/90 referenced from root instruction only; no current access.','no_model_fit_current_V112_PnP':True})
 write(H/'ROOT_SAVED_EVAL_ENTRY_TEMPLATE_V2.json',{'approved_saved_source_evaluation':False,'source_only':True,'model_forward':False,'fit':False,'current223':False,'V112':False,'PnP':False,'automatic_retry':False,'code':bind(__file__),'protocol':bind(PROTOCOL),'generated':bind(proof),'input_rows':None,'GT_rows':None,'input_crops':None,'crop_complete':None,'v61_prediction_seal':None,'C16_prediction_seal':None})
 print(json.dumps({'code':bind(__file__),'protocol':bind(PROTOCOL),'template':bind(H/'ROOT_SAVED_EVAL_ENTRY_TEMPLATE_V2.json'),'command':'/home/minjae/anaconda3/bin/python '+str(Path(__file__).resolve())+' --entry ROOT_ACTUAL_SAVED_EVAL_ENTRY.json','actual_payload_or_model_read':0}),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--generated',action='store_true');a.add_argument('--prepare',action='store_true');a.add_argument('--entry',type=Path);o=a.parse_args()
 if o.generated:generated()
 elif o.prepare:prepare()
 elif o.entry:run(o.entry)
 else:raise SystemExit('Use --generated / --prepare / --entry root.json')
