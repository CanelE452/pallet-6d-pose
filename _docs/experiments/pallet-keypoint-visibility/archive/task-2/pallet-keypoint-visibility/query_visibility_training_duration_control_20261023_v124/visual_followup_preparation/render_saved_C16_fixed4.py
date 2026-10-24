"""Inert fixed4 C16 report helper. Run only after root announces completed current evaluation."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,hashlib,json,math
from PIL import Image,ImageDraw,ImageFont
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
V=P/'query_visibility_training_duration_control_20261023_v124'
OLD=P/'query_visibility_success_failure_visual_review_20261023_v117'
EVAL=V/'private_evaluation/C16_FROZEN_CURRENT223'
FONT='/usr/share/fonts/truetype/nanum/NanumGothic.ttf';BOLD='/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf'
INK='#182d3b';MUTED='#4a6171';GREEN='#24654a';RED='#a43e35'
FIXED=((0,200),(3,1416),(5,325),(8,1021))

def require(x,s):
 if not x:raise ValueError(s)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(p),'sha256':sha(p)}
def check(b):
 require(sha(b['path'])==b['sha256'],'Changed binding: '+b['path']);return Path(b['path'])
def readbound(b):return json.loads(check(b).read_text())
def readcsv(p):
 with Path(p).open()as f:return list(csv.DictReader(f))
def txt(d,xy,s,n=20,color=INK,bold=False):d.text(xy,s,font=ImageFont.truetype(BOLD if bold else FONT,n),fill=color)
def decision(r):
 a=r['valid']=='True';z=float(r['raw_query_aux_hidden_logit']);require(math.isfinite(z),'Nonfinite saved logit')
 pred='U'if not a else('D'if z<0 else'E');require(r['authoritative_decision']in(pred,'UNKNOWN'if pred=='U'else pred),'Saved sign/U mismatch');return pred,z,a

def metrics_rows(tables,model):
 out=[]
 for group,denom in [('WOOD_PRIMARY105',(616,29)),('SQUARE',(531,61))]:
  rows=[r for r in tables['metrics']if r['cohort']=='paired_same_query_RGB'and r['model']==model and r['task']=='D_vs_E'and(r['group']==group or(group=='SQUARE'and r['group'].startswith('SQUARE')))]
  require(len(rows)==1,'Missing fixed paired metric');r=rows[0]
  require((int(r['visible_support']),int(r['occluded_support']))==denom,'Cohort changed')
  out.append(dict(group=r['group'],model=model,human_D_N=denom[0],human_E_N=denom[1],CM_rows_D_E_cols_D_E_U=[[int(r['TP']),int(r['FN_decided']),int(r['U_visible'])],[int(r['FP']),int(r['TN']),int(r['U_occluded'])]],D_correct=int(r['TP']),E_correct=int(r['TN']),D_recall=float(r['visible_recall_all_GT']),E_recall=float(r['occluded_recall_all_GT']),D_precision=float(r['visible_precision']),E_precision=float(r['occluded_precision']),U_rate=float(r['abstention_rate'])))
 return out

def render(args):
 # Only an externally supplied completed evaluation binding can open output bodies/pixels.
 ep=Path(args.evaluation_seal);require(ep.resolve()==(EVAL/'evaluation/EVALUATION_SEAL.json').resolve(),'Fixed C16 evaluation path')
 require(sha(ep)==args.expected_evaluation_sha256,'Wrong completed C16 evaluation SHA')
 ev=json.loads(ep.read_text());require(ev.get('completed')is True and ev.get('PASS')is True,'Evaluation incomplete')
 inf=readbound(ev['inference']);require(inf.get('completed')is True and inf.get('PASS')is True and inf['model_unchanged']and inf['slots']==1784 and inf['valid']==1739 and inf['U']==45 and inf['additional_U']==0 and inf['D_rule']=='z<0','Frozen C16 predictions incomplete')
 require(inf.get('current_used_for_selection')is False,'Current selected the model')
 require(inf['source_model_freeze']==ev['source_freeze'],'Sourcefreeze mismatch')
 freeze=readbound(ev['source_freeze']);require(freeze['completed']and freeze['threshold_logit']==0 and freeze.get('current_used_for_fit_or_selection')is False,'Unfrozen source selection');step=int(freeze['selected_update']);require(step in range(0,8001,100),'Unexpected source step')
 scoreb=next(x for x in inf['outputs']if Path(x['path']).name=='HEAD_ONLY_SCORES.csv');scores=readcsv(check(scoreb));require(len(scores)==1784,'Original slots changed')
 metricb=next(x for x in ev['files']if Path(x['path']).name=='CURRENT_PRIMARY_RESULTS.json');result=readbound(metricb);require(result['primary_model']=='V124_C16_RAW','Wrong actual current model')
 entry=readbound(ev['entry']);oldb=entry['bindings']['baseline_C15_scores'];oldscores=readcsv(check(oldb));require(len(oldscores)==1784,'C15 slots changed')
 out=Path(args.output);require(out.resolve()==(V/'visual_followup').resolve()and not out.exists(),'Do not overwrite report outputs');out.mkdir()
 lockp=OLD/'SELECTION_LOCK_BEFORE_PIXELS.json';receiptp=OLD/'RENDER_RECEIPTS.json';lock=json.loads(lockp.read_text());prior=json.loads(receiptp.read_text())
 m16=metrics_rows(result['tables'],'V124_C16_RAW');m15=metrics_rows(result['tables'],'V123_C15_RAW')
 descriptions=[('노출된 테두리·모서리 구조가 보이는 사례.','보임 판단은 초기 q의 좌표 정확성까지 입증하지 않음.'),('콘이 질의 부근을 차지하는 사례.','외부 가림 r은 기존 추정 좌표이며 좌표 gold가 아님.'),('콘과 남아 있는 목재가 함께 보이는 경계.','주변 재질의 존재와 기대한 구조 단서의 가시성은 다름.'),('콘이 crop 중심을 크게 차지하는 사례.','그림만으로 context·재질·위치의 단일 원인을 확정하지 않음.')]
 cards=[];cases=[]
 for n,(ix,row)in enumerate(FIXED):
  item=lock['selected'][ix];rec=prior['cases'][ix];r=scores[row];old=oldscores[row]
  require(item['original_native_row']==rec['original_native_row']==row and item['case_index']==rec['case_index']==ix,'Fixed selection changed')
  require(r['id']==old['id']==item['id']==rec['id']and int(r['native_corner'])==int(old['native_corner'])==item['native_slot']==rec['native_slot'],'Row/slot mismatch')
  q=[float(r['query_x']),float(r['query_y'])];L=float(r['roi_side'])
  require(q==item['query_xy']==[float(old['query_x']),float(old['query_y'])]and L==item['original_L']==float(old['roi_side']),'Original q/L changed')
  pred,z,a=decision(r);oldpred,oldz,olda=decision(old);require(a==olda,'Factual availability changed')
  human='D'if item['human_state']=='DIRECT_VISIBLE'else'E';correct=pred==human
  fullp=check(item['full_RGB_binding']);rawp=check(rec['raw_png']);markp=check(rec['marked_png'])
  with Image.open(fullp)as x:full=x.convert('RGB')
  with Image.open(rawp)as x:raw=x.convert('RGB')
  with Image.open(markp)as x:marked=x.convert('RGB')
  require(raw.size==marked.size==(96,96)and hashlib.sha256(full.tobytes()).hexdigest()==item['decoded_RGB_sha256_expected'],'Bound display pixels changed')
  title=('보임'if human=='D'else'외부 가림')+(' 성공'if correct else' 실패')+' · '+('목재'if n in(0,2)else'정사각형')
  card=Image.new('RGB',(800,635),'white');d=ImageDraw.Draw(card);color=GREEN if correct else RED;d.rectangle((0,0,799,7),fill=color)
  txt(d,(20,22),title,26,color,True);txt(d,(20,65),f"사람 {human} / C15 {oldpred} → C16 {pred}  · source step{step}",22,INK,True)
  txt(d,(20,102),f'원행 {row} · 슬롯 {item["native_slot"]} · 원 L={L:.2f} native px',17,MUTED)
  txt(d,(20,134),f'C15 zE={oldz:.4f} → C16 zE={z:.4f} (raw0 고정)',17,MUTED)
  txt(d,(20,170),'전체 원 RGB · 무표시',18);txt(d,(356,170),'저장 RGB96 · 무표시',18);txt(d,(568,170),'같은 crop · 표시 별도',18)
  card.paste(full.resize((320,240),Image.Resampling.BILINEAR),(20,200));card.paste(raw.resize((192,192),Image.Resampling.NEAREST),(356,200));card.paste(marked.resize((192,192),Image.Resampling.NEAREST),(568,200))
  txt(d,(356,410),'노랑 + : 원 q / 하늘 원 : 기존 r',17,MUTED)
  obs,limit=descriptions[n];txt(d,(20,468),obs,21,INK,True);txt(d,(20,505),limit,18,MUTED);txt(d,(20,545),'ID: '+item['id'],17,MUTED);txt(d,(20,579),'원 q/L·사람 라벨 유지 · 새 모델 호출/학습/좌표 교정/PnP 없음',17,MUTED);txt(d,(20,607),'local RGB96는 입력 일부; 별도 frozen context/features도 사용',16,MUTED)
  cp=out/f'CASE_{ix:02d}_C16.png';card.save(cp);cards.append(card)
  cases.append(dict(original_native_row=row,original_case_index=ix,id=item['id'],native_slot=item['native_slot'],q=q,L_native_pixels=L,human_state=item['human_state'],C15_prediction=oldpred,C15_raw_zE=oldz,C16_prediction=pred,C16_raw_zE=z,query_available=a,correct_against_existing_human_state=correct,E_reference_is_not_coordinate_gold=item['E_reference_is_occluded_estimated_not_coordinate_gold'],full_RGB=bind(fullp),raw_RGB96=bind(rawp),marked_RGB96=bind(markp),card=bind(cp),visual_description_KO=obs))
 board=Image.new('RGB',(1600,1500),'#edf2f5');d=ImageDraw.Draw(board)
 txt(d,(24,17),'현재 C16 · 같은 고정 4사례와 C15 저장 비교',31,INK,True)
 txt(d,(24,65),f"paired cohort: 목재 D {m16[0]['D_correct']}/616 · E {m16[0]['E_correct']}/29 | 정사각형 D {m16[1]['D_correct']}/531 · E {m16[1]['E_correct']}/61",23)
 txt(d,(24,105),f'공개 source-selected step{step} · 원 고정 행 [200,1416,325,1021] 유지 · 새 current 선택 없음',20,MUTED)
 for n,card in enumerate(cards):board.paste(card,((n%2)*800,155+(n//2)*635))
 txt(d,(24,1445),'고정4는 빈도 표본이 아님. E r은 추정값; 가시성과 좌표 검증·기하 self는 별개. 단일 원인 증명 아님.',19,MUTED)
 bp=out/'C16_FIXED4_SUCCESS_FAILURE.png';board.save(bp)
 summary=dict(completed=True,model='C16',source_selected_update=step,source_freeze=ev['source_freeze'],evaluation_seal=bind(ep),inference_seal=ev['inference'],scores=scoreb,C15_scores=oldb,metric_source=metricb,selection_lock=bind(lockp),prior_render_receipts=bind(receiptp),original_native_row_order=[x[1]for x in FIXED],all4_exact_id_slot_q_L=True,C15_metrics=m15,C16_metrics=m16,cases=cases,board=bind(bp),renderer=bind(Path(__file__)),fixed_before_C16_outcome_selection=True,current_used_for_fit_or_setting_selection=False,new_model_encoder_fit_initializer_PnP=0,fully_solved=False,actual_board_visual_QA_pending=True,root_render_QA_complete=False)
 sp=out/'C16_FIXED4_MAPPING_AND_METRICS.json';sp.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(completed=True,board=bind(bp),mapping=bind(sp)),ensure_ascii=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--evaluation-seal',type=Path,default=EVAL/'evaluation/EVALUATION_SEAL.json');p.add_argument('--expected-evaluation-sha256',required=True);p.add_argument('--output',type=Path,default=V/'visual_followup');render(p.parse_args())
