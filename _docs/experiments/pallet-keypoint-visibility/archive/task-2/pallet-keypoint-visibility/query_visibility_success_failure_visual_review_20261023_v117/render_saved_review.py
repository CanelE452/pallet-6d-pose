"""Locked stored current visual diagnosis only. No model/feature/fit imports."""
from pathlib import Path
from datetime import datetime,timezone
from PIL import Image,ImageDraw,ImageFont
import PIL,numpy as np
import json,hashlib,html,sys,textwrap
O=Path(__file__).resolve().parent;P=O.parent
FONT='/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
def ft(n):return ImageFont.truetype(FONT,n)
F=ft(23);S=ft(19);B=ft(29)
def bind(p):return {'path':str(p),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest(),'bytes':Path(p).stat().st_size}
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write(n,d):
 with (O/n).open('x')as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def wrap(draw,text,xy,width,font=F,fill='#182635',gap=8):
 x,y=xy;line=''
 for ch in text:
  if ch=='\n' or (line and draw.textlength(line+ch,font=font)>width):
   draw.text((x,y),line,font=font,fill=fill);y+=font.size+gap;line='' if ch=='\n' else ch
  else:line+=ch
 if line:draw.text((x,y),line,font=font,fill=fill);y+=font.size+gap
 return y
roles={'D_SUCCESS_WOOD_PRIMARY105':'D 성공 · 목재 · 두 head 모두 보임','D_SUCCESS_SQUARE118':'D 성공 · 정사 · 두 head 모두 보임','E_SUCCESS_WOOD_PRIMARY105':'외부 E 성공 · 목재 · C12 숨김','E_SUCCESS_SQUARE118':'외부 E 성공 · 정사 · C12 숨김','WOOD_696_D_FIXED':'D 변화 · 목재 · C11 숨김 → C12 보임','WOOD_325_E_FALSEVISIBLE':'외부 E 실패 · 목재 · 둘 다 보임','SQUARE_996_E_FALSEVISIBLE':'외부 E 실패 · 정사 · 둘 다 보임','E_REGRESSION_WOOD_PRIMARY105':'외부 E 퇴행 · 목재 · C11 숨김 → C12 보임','E_REGRESSION_SQUARE118':'외부 E 퇴행 · 정사 · C11 숨김 → C12 보임'}
def response(v):return {'VISIBLE':'D','ABSENT_NONVISIBLE':'H','D':'D','E':'H','ABSTAIN':'U','OUT_OF_TASK_GEOMETRY_S':'기하 S','OTHER_UNRESOLVED':'U'}.get(v,v)
def responses(c):return ' · '.join(f'{k} {response(v)}'for k,v in c['decisions'].items())
def human(c):return '인간 D: 보임'if c['human_state']=='DIRECT_VISIBLE'else'인간 E: 외부가림'
def marker(im,c,crop=False):
 m=im.copy();d=ImageDraw.Draw(m);q=c['query_xy'];r=c['reference_xy'];L=c['original_L']
 if crop:
  q=[47.5,47.5];r=None if None in r else [(c['reference_xy'][i]-c['query_xy'][i])/L*96+47.5 for i in [0,1]];sz=3;lw=1
 else:
  sz=5;lw=2;box=c['frozen_context_box'];d.rectangle(box,outline='#ed62ff',width=2);d.rectangle([q[0]-L/2,q[1]-L/2,q[0]+L/2,q[1]+L/2],outline='white',width=1)
 d.line([q[0]-sz,q[1],q[0]+sz,q[1]],fill='#fff500',width=lw);d.line([q[0],q[1]-sz,q[0],q[1]+sz],fill='#fff500',width=lw)
 if r is not None and None not in r:d.ellipse([r[0]-sz,r[1]-sz,r[0]+sz,r[1]+sz],outline='#00eaff',width=lw)
 return m
lockp=O/'SELECTION_LOCK_BEFORE_PIXELS.json';lock=json.loads(lockp.read_text());assert lock['completed']and lock['selection_not_based_on_new_pixels']
lockhash=bind(lockp);pack=lock['current_original_RGB96_packet_recorded_binding'];assert bind(Path(pack['path']))['sha256']==pack['sha256']
# Only these two existing model-input display fields; no P3/features/labels/model output arrays read.
with np.load(pack['path'],allow_pickle=False)as z:RGB=z['rgb96'];VALID=z['crop_valid96']
assert RGB.shape==(1784,3,96,96) and RGB.dtype==np.uint8 and VALID.shape==(1784,1,96,96)
figdir=O/'figures';figdir.mkdir(exist_ok=False);rawdir=O/'exact_saved_crops';rawdir.mkdir(exist_ok=False)
cache={};cards=[];receipts=[];selected=[]
for c in lock['selected']:
 ix=c['case_index']
 if c['missing']:
  card=Image.new('RGB',(1960,850),'white');ImageDraw.Draw(card).text((30,30),roles[c['role']]+' — 해당 사례 없음',font=B,fill='black');cards.append(card);continue
 n=c['original_native_row'];path=Path(c['full_RGB_binding']['path']);assert bind(path)['sha256']==c['full_RGB_binding']['sha256']
 if str(path)not in cache:cache[str(path)]=np.array(Image.open(path).convert('RGB'),copy=True)
 full=cache[str(path)];assert ah(full)==c['decoded_RGB_sha256_expected']
 raw=Image.fromarray(RGB[n].transpose(1,2,0));marked=marker(raw,c,True);fullraw=Image.fromarray(full);fullmarked=marker(fullraw,c)
 rawp=rawdir/f'case_{ix:02d}_RAW_RGB96.png';raw.save(rawp);markedp=rawdir/f'case_{ix:02d}_MARKED_RGB96.png';marked.save(markedp)
 validim=Image.fromarray(np.floor(np.clip(VALID[n,0],0,1)*255+.5).astype(np.uint8));validp=rawdir/f'case_{ix:02d}_VALID96.png';validim.save(validp)
 # Output raw PNG round-trips to the exact saved array, not a regenerated/recentered crop.
 assert np.array_equal(np.array(Image.open(rawp)),RGB[n].transpose(1,2,0))
 card=Image.new('RGB',(1960,850),'#f6f8fa');d=ImageDraw.Draw(card)
 d.text((16,12),f'사례 {ix+1:02d} | {roles[c["role"]]}',font=B,fill='#143f5c')
 d.text((16,54),f'원행 {n} · {c["id"]} · native 슬롯 {c["native_slot"]} · {human(c)}',font=F,fill='#182635')
 d.text((16,88),'원 전체 RGB (무표시)',font=S,fill='black');d.text((672,88),'표시 별도: q 노랑 / r 하늘 / 원 L 흰색 / 고정 context 자홍',font=S,fill='black')
 # Native640x480 currently, display-only fit if a recorded image has another shape.
 fr=fullraw.copy();fr.thumbnail((640,480),Image.Resampling.BILINEAR);fm=fullmarked.copy();fm.thumbnail((640,480),Image.Resampling.BILINEAR)
 card.paste(fr,(16,122));card.paste(fm,(672,122))
 d.text((1328,88),'원 q의 RGB96 · 무표시 3배',font=S,fill='black');d.text((1632,88),'표시만 추가한 동일 crop',font=S,fill='black')
 card.paste(raw.resize((288,288),Image.Resampling.NEAREST),(1328,122));card.paste(marked.resize((288,288),Image.Resampling.NEAREST),(1632,122))
 yy=430;yy=wrap(d,responses(c),(1328,yy),590,F)
 yy=wrap(d,f'C11 z={c["C11_raw_zE"]:.6f} / C12 z={c["C12_raw_zE"]:.6f}',(1328,yy),590,F)
 yy=wrap(d,f'L={c["original_L"]:.3f} native px · q-r={c["initial_reference_error_px"]:.3f}px',(1328,yy),590,F)
 yy=wrap(d,'r 출처: '+c['reference_annotation_coordinate_source'],(1328,yy),590,S)
 yy=wrap(d,'E의 r은 가려진 지점의 기존 추정/미검증 좌표입니다.'if c['human_state']=='EXTERNAL_OCCLUDED'else'D의 r은 기존 manual-click reference입니다. 좌표 검증기 출력이 아닙니다.',(1328,yy),590,S,fill='#a33827')
 d.text((16,626),f'q=({c["query_xy"][0]:.5f}, {c["query_xy"][1]:.5f})  L={c["original_L"]:.5f}  | valid96={float(VALID[n].mean()):.4f}',font=F,fill='black')
 d.text((16,661),f'고정 context box: '+str([round(x,3)for x in c['frozen_context_box']])+'  | 원 geometryS='+c['geometry_self_hidden_original'],font=F,fill='black')
 wrap(d,'D=보임 응답 / H=binary hidden / U=미해결. 같은 초기 q·RGB96 표시가 네 모델의 입력까지 같다는 뜻은 아닙니다. v61은 P3, legacy는 자체 RGB/guidance, C11/C12는 동일 저장 DINO features/operator입니다.',(16,705),1915,S)
 wrap(d,'팔레트 재질이 q 근처에 보이는 것과 기대한 부품 단서가 보이는 것은 다릅니다. 표시 r은 display-only이며 crop·q/L·라벨·모델 판단을 바꾸지 않았습니다.',(16,778),1915,S)
 fp=figdir/f'CASE_{ix:02d}.png';card.save(fp);cards.append(card)
 receipts.append({'case_index':ix,'original_native_row':n,'id':c['id'],'native_slot':c['native_slot'],'full_RGB':c['full_RGB_binding'],'decoded_RGB_sha256':ah(full),'RGB96_CHW_sha256':ah(RGB[n]),'valid96_sha256':ah(VALID[n]),'raw_png':bind(rawp),'marked_png':bind(markedp),'valid_png':bind(validp),'figure':bind(fp),'raw_png_exact_saved_array':True,'model_decisions_or_labels_q_L_unchanged':True})
 selected.append((c,raw,fullmarked,marked))
contacts=[]
for j in range(3):
 sheet=Image.new('RGB',(1960,2550),'white')
 for k in range(3):sheet.paste(cards[3*j+k],(0,850*k))
 p=figdir/f'CONTACT_{j+1:02d}.png';sheet.save(p);contacts.append(bind(p))
# Fixed four-role user preview, exclusively a subset of the locked nine cases.
preview=Image.new('RGB',(1600,1300),'#e4ebf0');chosen=[]
for j,role in enumerate(lock['inline_four_roles']):
 panel=Image.new('RGB',(800,650),'white');d=ImageDraw.Draw(panel);c=next(x for x in lock['selected']if x['role']==role);chosen.append(c['case_index'])
 d.text((16,16),roles[role],font=ft(25),fill='#173f59')
 if c['missing']:d.text((16,70),'해당 고정 조건 사례 없음',font=B,fill='black')
 else:
  _,raw,fm,marked=next(x for x in selected if x[0]['role']==role)
  d.text((16,60),human(c)+f'  · 원행 {c["original_native_row"]}',font=ft(25),fill='black')
  d.text((16,102),responses(c),font=ft(22),fill='black')
  d.text((16,139),f'L {c["original_L"]:.2f}px · q-r {c["initial_reference_error_px"]:.2f}px',font=ft(22),fill='black')
  d.text((16,177),'실제 원 RGB96 (무표시)',font=S,fill='black');panel.paste(raw.resize((288,288),Image.Resampling.NEAREST),(16,210))
  thumb=fm.copy();thumb.thumbnail((300,245),Image.Resampling.BILINEAR);d.text((330,177),'전체 영상의 위치 표시',font=S,fill='black');panel.paste(thumb,(330,210))
  d.text((650,177),'q / r 별도표시',font=ft(17),fill='black');panel.paste(marked.resize((128,128),Image.Resampling.NEAREST),(650,210))
  wrap(d,'r: 기존 추정/출처 미지정. 작은 오차도 가려진 좌표의 정답 인증이 아닙니다.'if c['human_state']=='EXTERNAL_OCCLUDED'else'r: 기존 manual click. D 성공은 원 q의 좌표 정답 여부를 모델이 검증한 뜻이 아닙니다.',(16,517),765,ft(20),fill='#9c352a')
  wrap(d,'원 q/L·라벨 유지 · D 보임 / H 숨김 / U 미해결\nC11/C12 입력 동일, 네 모델 전체 입력은 서로 다릅니다.',(16,588),765,ft(17),gap=4)
 preview.paste(panel,((j%2)*800,(j//2)*650))
pp=O/'INLINE_FIXED4_SUMMARY.png';preview.save(pp)
receipt={'completed':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'selection_lock':lockhash,'renderer':bind(Path(__file__)),'command':f'python {Path(__file__)}','runtime':{'Python':sys.version.split()[0],'Pillow':PIL.__version__,'NumPy':np.__version__,'Korean_font':FONT},'current_RGB96_packet_recorded_binding':pack,'packet_fields_read_only':['rgb96','crop_valid96'],'packet_container_rows':1784,'locked_rows_extracted_or_viewed':9,'unique_full_RGBs_decoded':len(cache),'all9_rawPNG_exact_saved_RGB96':True,'all_full_RGB_file_and_decoded_hashes_exact':True,'no_new_crop_or_recenter':True,'cases':receipts,'contacts':contacts,'inline4':bind(pp),'inline4_case_indices':chosen,'raw_not_painted':True,'markers_only_on_separate_panels':True,'no_GT_best_crop_or_reference_model_input':True,'no_new_model_inference_fit_threshold_or_coordinate_change_PnP':True,'current_data_used_for_training_or_model_selection':False,'root_has_viewed_new_outputs_at_write_time':False}
write('RENDER_RECEIPTS.json',receipt)
# Saved metric context only, no new evaluation or model choice.
res=json.loads((P/'query_visibility_material_augmented_query_head_20261023_v114/private_evaluation/C12_FROZEN_CURRENT223/evaluation/CURRENT_PRIMARY_RESULTS.json').read_text());mx=[r for r in res['tables']['D_vs_external_E_2x3']if r['cohort']=='paired_same_query_RGB'and r['model']in['v61','legacy','V111_C11_RAW','V114_C12_RAW']]
def esc(s):return html.escape(str(s),quote=True)
def rel(p):return esc(str(Path(p).relative_to(O)))
def pct(v):return f'{v*100:.1f}%'
htmltext='''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>보임·외부가림 성공/실패 실제 이미지 비교</title><style>body{margin:0;background:#eef2f6;color:#182a39;font:17px/1.75 system-ui,'NanumGothic',sans-serif}main{max-width:1200px;margin:auto;padding:30px 24px}header,section{background:white;padding:26px;margin:20px 0;border-radius:12px;border:1px solid #d4dfe7}h1{font-size:30px;line-height:1.4}h2{font-size:23px}p{margin:12px 0}.warn{padding:14px 18px;background:#fff0e7;border-left:5px solid #b66537}.note{padding:14px 18px;background:#eaf3f9;border-left:5px solid #356e98}img{width:100%;height:auto;display:block}figure{margin:14px 0}figcaption{font-size:14px;color:#526778}a{color:#155d90;overflow-wrap:anywhere}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;border-bottom:1px solid #d9e2e9;text-align:right}th:first-child,td:first-child{text-align:left}code{font-size:13px}.table{overflow:auto}.tag{font-size:14px;color:#536b7e}</style></head><body><main><header><div class="tag">V117 · 저장된 현재 평가의 사례 진단 · 새 모델 호출0</div><h1>어디서 맞고, 어디서 틀렸나?<br>원래 질의를 그대로 둔 실제 이미지 비교</h1><p>보이는 D 성공과 외부가림 E 성공을 따로 보여 주고, D를 고친 경우·외부 E를 놓친 경우·C11에서 C12로 퇴행한 경우를 함께 확인합니다. 9사례는 이미지 보기 전에 조건별 가장 앞 원행과 기존 고정행으로 잠갔습니다.</p><p class="warn">같은 초기 q와 원 RGB96을 표시해도 네 모델의 입력이 모두 같았다는 뜻은 아닙니다. v61은 P3, legacy는 자체 RGB/guidance 경로이며, C11/C12만 정확히 같은 저장 DINO features와 operator를 공유합니다. 이 페이지는 평가 진단용이지 학습·모델 선택 자료가 아닙니다.</p></header>'''
htmltext+=f'<section><h2>한눈에 보는 고정4사례</h2><p>D 성공 · E 성공 · E 실패 · D 변화의 순서입니다. 원 crop은 무표시이며 q/r는 별도 표시 패널에만 추가했습니다.</p><a href="{rel(pp)}"><img src="{rel(pp)}" alt="실제 저장 crop의 고정4 성공실패 사례"></a><p><a href="SELECTION_LOCK_BEFORE_PIXELS.json">선택 lock</a> · <a href="RENDER_RECEIPTS.json">원본/렌더 hash·명령 기록</a></p></section>'
htmltext+='<section><h2>전체 지표와 사례의 관계</h2><p>현재223프레임/1,784슬롯 전체는 그대로 유지했습니다. factual U45, 추가 U0입니다. 아래는 동일 paired D/E 코호트의 저장 지표이며, 9사례의 빈도로 전체 성능을 추정하지 않습니다.</p><div class="table"><table><tr><th>그룹 / 모델</th><th>D 정답 / 수</th><th>E 정답 / 수</th><th>E 정밀도</th><th>E 재현율</th><th>U율</th></tr>'
for r in mx:
 name={'v61':'v61','legacy':'실사 legacy','V111_C11_RAW':'C11','V114_C12_RAW':'C12'}[r['model']];group='목재'if r['group'].startswith('WOOD')else'정사'
 htmltext+=f'<tr><td>{group} / {name}</td><td>{r["D_pred_D"]}/{r["human_D_N"]}</td><td>{r["E_pred_hidden"]}/{r["human_external_E_N"]}</td><td>{pct(r["external_E_response_precision"])}</td><td>{pct(r["external_E_response_recall"])}</td><td>{pct(r["unresolved_rate"])}</td></tr>'
htmltext+='</table></div><p class="note">가시성 판단과 좌표 정답 검증은 별개입니다. E의 기존 reference는 가려진 지점의 추정값이고 일부는 원 manifest에서 출처가 미지정입니다. 작은 q-r 오차도 정확한 위치 gold를 의미하지 않으며, 큰 오차에서 E를 맞힌 사례 역시 localization 성공은 아닙니다. 재질이 보이는 것과 기대한 부품 단서가 보이는 것도 다릅니다.</p></section>'
for c in lock['selected']:
 htmltext+=f'<section id="case{c["case_index"]}"><h2>{c["case_index"]+1:02d}. {roles[c["role"]]}</h2>'
 if c['missing']:htmltext+='<p>해당 고정 조건의 사례 없음. 다른 사례로 교체하지 않았습니다.</p></section>';continue
 i=c['case_index'];htmltext+=f'<p>원행 <b>{c["original_native_row"]}</b> · {esc(c["id"])} · native 슬롯{c["native_slot"]} · {human(c)} · L {c["original_L"]:.3f}px · q-r {c["initial_reference_error_px"]:.3f}px</p>'
 htmltext+=f'<p><b>{esc(responses(c))}</b> / C11 z={c["C11_raw_zE"]:.6f}, C12 z={c["C12_raw_zE"]:.6f}. D iff z&lt;0; H iff z≥0, factual U 우선.</p>'
 if c['human_state']=='EXTERNAL_OCCLUDED':htmltext+='<p class="warn">E reference: 가려진 지점의 기존 추정/미검증 r. 원 기록 출처: '+esc(c['reference_annotation_coordinate_source'])+'. 표시 r로 새 crop을 만들거나 원 좌표를 교정하지 않았습니다. 작은 거리·보이는 주변 팔레트 재질로 해당 기대 단서의 위치나 가시성을 새로 확정하지 않습니다.</p>'
 else:htmltext+='<p>기존 manual-click D reference의3px 이내 부분집합입니다. 원 q의 좌표 정답 여부를 모델이 검증한 출력은 아닙니다.</p>'
 fp=figdir/f'CASE_{i:02d}.png';htmltext+=f'<figure><a href="{rel(fp)}"><img src="{rel(fp)}" alt="사례{i+1} 원 전체RGB와 원 q RGB96의 무표시·별도표시 비교"></a><figcaption>클릭하면 원 크기. raw96은 저장 배열과 byte-exact. 노랑q / 하늘r / 흰L범위 / 자홍 frozen context. 원 RGB에는 표시를 넣지 않았습니다.</figcaption></figure>'
 htmltext+=f'<p><a href="{rel(rawdir/f"case_{i:02d}_RAW_RGB96.png")}">무표시 원 RGB96</a> · <a href="{rel(rawdir/f"case_{i:02d}_MARKED_RGB96.png")}">표시 패널</a> · <a href="{rel(rawdir/f"case_{i:02d}_VALID96.png")}">원 valid96</a></p></section>'
htmltext+='''<section><h2>이 사례로 말할 수 있는 범위</h2><p>성공과 실패가 모두 실제로 존재합니다. 다만 이9개는 결과 조건별로 고른 진단 표본이며, 공통 모양을 원인이나 전체 빈도로 단정할 수 없습니다. 현재223자료는 과거 개발에 노출되었고 이번 그림으로 head·문턱·checkpoint·arm·학습 자료를 선택하지 않습니다.</p><p>현재 모델은 주어진 q에 대한 조건부 보임/hidden 응답을 출력합니다. 초기 좌표가 정확한지, 보이는 주변 구조가 기대한 native 부품인지까지 동시에 검증한 모델은 아닙니다. 초기 기하 self 경로도 별도이며 raw hidden은 학습된S/가림 원인 분류가 아닙니다. 이 차이를 분리해서 실제 이미지를 검토해야 합니다.</p><p>원 human GT·q/L·native순서·reference·기존 모델 출력은 바꾸지 않았습니다. 새 모델·encoder·학습·GT recrop·좌표 이동·PnP0. 전체 사례의 관찰과 목표 범위에 관한 최종 해석은 root가 원 이미지를 확인한 뒤 별도 기록합니다.</p></section></main></body></html>'''
(O/'ROOT_READOUT_KO.html').write_text(htmltext)
result={'completed':True,'scope':'Stored current report-only visual diagnosis after deterministic lock; no fitting/inference/private-driven design.','selection_lock':lockhash,'render_receipt':bind(O/'RENDER_RECEIPTS.json'),'HTML':bind(O/'ROOT_READOUT_KO.html'),'inline4':bind(pp),'selected_count':9,'unique_full_RGBs':len(cache),'all9_rawPNG_exact':True,'all4_models_inputs_identical':False,'C11_C12_features_operator_identical':True,'E_reference_not_coordinate_gold':True,'new_head_encoder_fit_PnP':0,'root_actual_visual_review_pending':True}
write('RESULT.json',result)
print(json.dumps({'result':bind(O/'RESULT.json'),'HTML':bind(O/'ROOT_READOUT_KO.html'),'inline4':bind(pp),'render_receipt':bind(O/'RENDER_RECEIPTS.json'),'figures':9,'fullRGB':len(cache)},ensure_ascii=False))
