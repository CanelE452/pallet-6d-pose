"""Root-authorized saved-crop diagnostic, no recrop/model/GT label creation."""
from pathlib import Path
from collections import Counter
import csv,hashlib,html,io,json,math,os,stat,textwrap
import numpy as np
from PIL import Image,ImageDraw,ImageFont
H=Path(__file__).parent;V=H.parent;P=V.parent
LEDGER={'path':str(V/'saved_real_head_evaluation_v1/current1784/ALL1784_CURRENT_NATIVE_LEDGER.jsonl'),'sha256':'2c63bc193fb7ace8c82873f6299ec3b26406ca4a9bdd105b37fe01174de531df'}
INPUT={'path':str(P/'query_visibility_jitter_heads_frozen_real_benchmark_20261024_v184/real_raw10_SAM96_inputs_v1/CURRENT223_1784_INPUTS.npz'),'sha256':'b2d4723167f1e384ca3fed9dbe22ebbf180e7f9671bbb1d630c2f4eea6a96d08'}
MODELS=('v61','legacy','C55','C63','C64','C65');GROUPS=('WOOD_PRIMARY105','SQUARE118')
LABELS=('D ref<=3: C63 E -> C65 D','D ref<=3: C63 D -> C65 E','D ref<=3: both C63/C65 E','Human E: legacy E, C65 D','Human E: C65 E','Human D: legacy E, C65 D')
META=('row_id','original_global_index','native_slot','query_xy','L','original_query_available')
def need(v,m):
 if not bool(v):raise ValueError(m)
def raw(p):
 p=Path(p);need(p.is_absolute()and '..'not in p.parts and 'v112'not in str(p).casefold(),'literal nonV112 before FS')
 fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for part in p.parts[1:-1]:
   n=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
  leaf=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 finally:os.close(fd)
 with os.fdopen(leaf,'rb')as f:need(stat.S_ISREG(os.fstat(f.fileno()).st_mode),'regular file');return f.read()
def sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def binding(p):z=raw(p);return dict(path=str(p),sha256=hashlib.sha256(z).hexdigest(),size_bytes=len(z))
def checked(b):
 z=raw(b['path']);need(hashlib.sha256(z).hexdigest()==b['sha256'],'exact saved artifact SHA');return z

def write(p,x):
 with p.open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 return binding(p)
def num(v):
 try:z=float(v)
 except (ValueError,TypeError):return None
 return z if math.isfinite(z)else None
def yes(v):
 need(type(v)is bool or v in ('True','False'),'saved paired bool spelling');return v is True or v=='True'
def eligible(r):return yes(r['paired_same_query_RGB_D_vs_E'])and r['query_available']is True and r['human_state']in('DIRECT_VISIBLE','EXTERNAL_OCCLUDED')
def predicate(r,k):
 d=r['human_state']=='DIRECT_VISIBLE';e=r['human_state']=='EXTERNAL_OCCLUDED';er=num(r['initial_reference_error_px']);near=er is not None and er<=3
 a,b,l=r['C63_D_E_U'],r['C65_D_E_U'],r['legacy_D_E_U']
 return (d and near and a=='E'and b=='D',d and near and a=='D'and b=='E',d and near and a==b=='E',e and l=='E'and b=='D',e and b=='E',d and l=='E'and b=='D')[k]
def select(rows):
 out=[]
 for group in GROUPS:
  pool=[r for r in rows if r['group']==group and eligible(r)]
  for k,label in enumerate(LABELS):
   candidates=sorted([r for r in pool if predicate(r,k)],key=lambda r:r['global_index']);r=candidates[0]if candidates else None
   out.append(dict(position=len(out),group=group,stratum=k+1,predicate=label,matching_count=len(candidates),global_index=None if r is None else r['global_index'],missing=r is None))
 seen=Counter(x['global_index']for x in out if not x['missing'])
 for x in out:x['duplicate_across_strata_count']=0 if x['missing']else seen[x['global_index']]
 return out

def ref_address(r):
 q=[num(r['query_x']),num(r['query_y'])];ref=[num(r['human_reference_x']),num(r['human_reference_y'])];L=num(r['roi_side'])
 if any(x is None for x in q+ref)or L is None or L<=0:return None
 return [47.5+96*(ref[j]-q[j])/L for j in range(2)]
def font(size):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)
def text_lines(draw,text,xy,width=62,size=17,fill=(35,45,55)):
 x,y=xy
 for line in textwrap.wrap(str(text),width=width):draw.text((x,y),line,font=font(size),fill=fill);y+=size+5
 return y

def draw_panel(cell,row,rgb=None):
 im=Image.new('RGB',(620,760),'white');d=ImageDraw.Draw(im);d.rectangle((0,0,619,759),outline=(170,180,190),width=2)
 d.text((16,12),f"{cell['position']+1:02d} {cell['group']} / type{cell['stratum']}",font=font(21),fill=(15,35,65));text_lines(d,cell['predicate'],(16,44),55,17)
 if row is None:d.text((18,125),'MISSING - no replacement',font=font(23),fill=(150,50,50));return im
 i=row['global_index'];r=ref_address(row);x0,y0,scale=16,98,3
 crop=Image.fromarray(rgb,'RGB').resize((288,288),Image.Resampling.NEAREST);im.paste(crop,(x0,y0));cx=x0+scale*47.5+1.;cy=y0+scale*47.5+1.
 d=ImageDraw.Draw(im);d.line((cx-11,cy,cx+11,cy),fill=(0,255,255),width=2);d.line((cx,cy-11,cx,cy+11),fill=(0,255,255),width=2)
 if r is not None:
  rx,ry=r;px=x0+(max(0.,min(95.,rx))+.5)*scale;py=y0+(max(0.,min(95.,ry))+.5)*scale
  if -.5<=rx<95.5 and -.5<=ry<95.5:
   d.line((px-8,py-8,px+8,py+8),fill=(255,0,220),width=2);d.line((px-8,py+8,px+8,py-8),fill=(255,0,220),width=2)
  else:
   d.ellipse((px-6,py-6,px+6,py+6),outline=(255,0,220),width=3);d.text((x0,y0+294),'Reference OUTSIDE; edge marker is display only',font=font(12),fill=(150,0,120))
 yy=100
 info=[f"g{i} native{row['native_corner']}",f"human: {row['human_state']}",f"query: ({row['query_x']:.2f},{row['query_y']:.2f})",f"L: {row['roi_side']:.3f}",f"existing ref error: {row['initial_reference_error_px']} px",f"duplicate cells: {cell['duplicate_across_strata_count']}"]
 for t in info:yy=text_lines(d,t,(320,yy),25,16)+5
 y=423;d.text((18,y),'MODEL       D/E/U     SAVED VALUE',font=font(17),fill=(15,35,65));y+=28
 for m in MODELS:
  pr=row[m+'_D_E_U'];col=(25,120,65)if pr=='D'else(180,40,35)if pr=='E'else(100,100,100)
  if m=='v61':score=num(row.get('v61_saved_p_present'));kind='p_present'
  elif m=='legacy':score=num(row.get('legacy_saved_p_DIRECT'));kind='p_DIRECT'
  else:score=num(row.get(m+'_raw_direct_end_external_logit'));kind='raw z'
  value='NA'if score is None else f'{score:.6g}'
  d.text((18,y),m,font=font(18),fill=(25,35,45));d.text((161,y),pr,font=font(19),fill=col);d.text((232,y),f'{kind}: {value}',font=font(17),fill=(35,45,55));y+=29
 if r is not None:y=text_lines(d,f'Existing reference raw96 address: ({r[0]:.2f}, {r[1]:.2f})',(18,637),60,15)+3
 else:y=text_lines(d,'Existing reference unavailable',(18,637),60,15)+3
 text_lines(d,'Cyan: ORIGINAL q=(47.5,47.5). Magenta: existing numeric reference proxy. Different model calls do not certify native-part identity or physical cause.',(18,y),70,14)
 return im

def fixtures():
 r=dict(global_index=4,group=GROUPS[0],paired_same_query_RGB_D_vs_E='True',query_available=True,human_state='DIRECT_VISIBLE',initial_reference_error_px='2',C63_D_E_U='E',C65_D_E_U='D',legacy_D_E_U='E')
 a=select([r,dict(r,global_index=2)]);need(a[0]['global_index']==2 and a[5]['global_index']==2 and a[0]['duplicate_across_strata_count']==2,'minimum-index and duplicate fixture');need(a[1]['missing']and all(x['missing']for x in a[6:]),'missing no fallback')
 need(not eligible(dict(r,query_available=False))and not eligible(dict(r,paired_same_query_RGB_D_vs_E='False')),'primary available filter fixture')
 rr=dict(query_x=10.,query_y=20.,roi_side=96.,human_reference_x='10',human_reference_y='20');need(ref_address(rr)==[47.5,47.5]and ref_address(dict(rr,human_reference_x='106'))==[143.5,47.5],'original reference address algebra center/outside')
 need(ref_address(dict(rr,human_reference_x='nan'))is None,'missing reference not invented')
 return dict(completed=True,PASS=True,count=5,invented_only=True,checks=['minimum_index','duplicate_kept','missing_no_substitute','primary_available_preserved','unchanged_ref_address_and_missing'])

def main():
 protocol=json.loads(raw(H/'PROTOCOL_AND_ROOT_AUTHORITY_BEFORE_LEDGER_OR_PIXELS.json'));write(H/'PURE_SELECTION_AND_ADDRESS_CHECKS.json',fixtures())
 rows=[json.loads(x)for x in checked(LEDGER).splitlines()if x.strip()];need(len(rows)==1784 and [r['global_index']for r in rows]==list(range(1784)),'same saved1784 order');need(set(r['group']for r in rows)==set(GROUPS),'same WOOD/SQUARE');need(sum(eligible(r)for r in rows)==1237,'same original paired1237 available')
 cells=select(rows);unique=sorted({x['global_index']for x in cells if not x['missing']});selection=dict(completed=True,PASS=True,selected_before_any_NPZ_pixel_read=True,ledger=LEDGER,root_predicates=protocol['predicates'],cells12=cells,unique_global_indices=unique,unique_examples=len(unique),missing_cells=sum(x['missing']for x in cells),not_representative=True,no_replacement=True)
 selected=write(H/'SELECTED12_STRATA_BEFORE_RAW_INPUT_PIXELS.json',selection)
 # Actual permission covers only current RAW10 RGB, valid and META6; SAM is not decoded.
 z=checked(INPUT)
 with np.load(io.BytesIO(z),allow_pickle=False)as pack:x={k:pack[k]for k in ('rgb96','valid96',*META)}
 need(x['rgb96'].dtype==np.uint8 and x['rgb96'].shape==(1784,96,96,3)and x['valid96'].dtype==bool and x['valid96'].shape==(1784,96,96),'original RAW96 dtypes')
 for k,dt,shape in [('original_global_index',np.int64,(1784,)),('native_slot',np.int8,(1784,)),('query_xy',np.float64,(1784,2)),('L',np.float64,(1784,)),('original_query_available',np.bool_,(1784,))]:need(x[k].dtype==dt and x[k].shape==shape,'original META dtype '+k)
 need(x['row_id'].dtype.kind=='U'and x['row_id'].shape==(1784,)and np.array_equal(x['original_global_index'],np.arange(1784))and np.array_equal(x['native_slot'],np.arange(1784)%8)and (~x['original_query_available']).sum()==45,'original allnative/U')
 for i,r in enumerate(rows):
  q=[np.nan,np.nan]if r['query_x']is None or r['query_y']is None else [r['query_x'],r['query_y']];L=np.nan if r['roi_side']is None else r['roi_side']
  need(str(x['row_id'][i])==r['id']+':native'+str(r['native_corner'])and int(x['native_slot'][i])==r['native_corner']and bool(x['original_query_available'][i])==r['query_available']and np.array_equal(x['query_xy'][i],q,equal_nan=True)and np.array_equal(x['L'][i],L,equal_nan=True),'all1784 META6 saved ledger exact')
 before={i:(sha(x['rgb96'][i]),sha(x['valid96'][i]))for i in unique};root=H/'unique_cases';root.mkdir();clean={};manifest=[]
 for i in unique:
  f=root/('CASE_%04d'%i);f.mkdir();rgb=x['rgb96'][i];valid=x['valid96'][i];c=f/'CLEAN96.png';v=f/'VALID96.png';Image.fromarray(rgb,'RGB').save(c);Image.fromarray(valid.astype(np.uint8)*255,'L').save(v)
  need(np.array_equal(np.asarray(Image.open(c)),rgb)and np.array_equal(np.asarray(Image.open(v))==255,valid),'derived clean96/valid96 exact decoded parity');clean[i]=dict(CLEAN96=binding(c),VALID96=binding(v),decoded_RGB_sha256=before[i][0],decoded_bool_valid_sha256=before[i][1],valid_fraction=float(valid.mean()))
 overview=Image.new('RGB',(2480,2280),(235,239,243));panels=[];csvrows=[]
 for cell in cells:
  i=cell['global_index'];r=None if i is None else rows[i];panel=draw_panel(cell,r,None if r is None else x['rgb96'][i]);p=H/('PANEL_%02d.png'%cell['position']);panel.save(p);overview.paste(panel,((cell['position']%4)*620,(cell['position']//4)*760));rec=dict(**cell,panel=binding(p),clean96=None if i is None else clean[i],row=None if r is None else r,existing_reference_raw96_address=None if r is None else ref_address(r),new_visibility_label=None,physical_native_gold_unknown=True);panels.append(rec)
  record={k:cell[k]for k in['position','group','stratum','predicate','matching_count','global_index','missing','duplicate_across_strata_count']}
  if r is not None:
   for k in['id','native_corner','human_state','query_x','query_y','roi_side','query_available','initial_reference_error_px','human_reference_x','human_reference_y']:record[k]=r[k]
   for m in MODELS:record[m+'_prediction']=r[m+'_D_E_U'];record[m+'_raw_logit_or_saved_probability']=r.get(m+'_raw_direct_end_external_logit')if m not in('v61','legacy')else r.get('v61_saved_p_present'if m=='v61'else'legacy_saved_p_DIRECT')
  csvrows.append(record)
 overview.save(H/'PANELS_12.png');fields=list(dict.fromkeys(k for r in csvrows for k in r))
 with (H/'ALL12_SELECTED_STRATA.csv').open('x',newline='')as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(csvrows)
 for i in unique:need(before[i]==(sha(x['rgb96'][i]),sha(x['valid96'][i])),'source arrays unchanged')
 checked(INPUT);checked(LEDGER)
 payload=dict(completed=True,PASS=True,selection=selected,panels12=panels,unique_examples=len(unique),missing_cells=selection['missing_cells'],input_NPZ=INPUT,input_NPZ_not_published=True,decoded_members=list(x),SAM_members_decoded=0,all1784_original_META_exact=True,original_U45_preserved=True,all_selected_CLEAN96_and_VALID96_decoded_parity_exact=True,no_new_crop_initializer_model_or_coordinate_changes=True,root_authority=binding(H/'PROTOCOL_AND_ROOT_AUTHORITY_BEFORE_LEDGER_OR_PIXELS.json'))
 write(H/'IMAGE_MANIFEST.json',payload)
 intro=f'기존 paired D/E 1237 중 사전 규칙 12칸 · 고유 예시 {len(unique)}개 · 빈 조건 {selection["missing_cells"]}칸. 중복은 유지했다. Cyan은 원 q, magenta는 기존 reference proxy이며 GT/보정점이 아니다. 실제 native 부위·물리적 가림 원인을 이 crop만으로 재인증하지 않는다. v61/legacy는 저장 확률, C55/C63/C64/C65는 저장 raw logit으로 값의 의미가 다르다.'
 md=['# 저장 crop 조건별 진단',intro,'[12패널 PNG](PANELS_12.png) · [행 CSV](ALL12_SELECTED_STRATA.csv) · [사전 선택](SELECTED12_STRATA_BEFORE_RAW_INPUT_PIXELS.json) · [manifest](IMAGE_MANIFEST.json) · [완료](COMPLETE.json)','![12패널](PANELS_12.png)']
 cards=[]
 for r in panels:
  cell=r;caption=f"{cell['group']} 조건{cell['stratum']}: {cell['predicate']} / "+('MISSING'if cell['missing']else f"g{cell['global_index']} (해당조건 {cell['matching_count']}개, 중복 {cell['duplicate_across_strata_count']}칸)")
  rel=Path(cell['panel']['path']).name;md.extend(['',f'## {caption}',f'![{caption}]({rel})']);cards.append(f'<section><h2>{html.escape(caption)}</h2><img src="{rel}" alt="{html.escape(caption)}"></section>')
 (H/'PORTABLE_GALLERY_KO.md').write_text('\n\n'.join(md)+'\n')
 (H/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>V220 저장 crop 진단</title><style>body{font:16px sans-serif;max-width:1300px;margin:30px auto;padding:0 16px;color:#25354a}img{max-width:100%;height:auto}section{border-top:1px solid #ccd5df;margin-top:30px}h2{font-size:19px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:16px}</style><h1>원 저장 crop 조건별 진단</h1><p>'+html.escape(intro)+'</p><p><a href="ALL12_SELECTED_STRATA.csv">행 CSV</a> · <a href="SELECTED12_STRATA_BEFORE_RAW_INPUT_PIXELS.json">사전선택</a> · <a href="IMAGE_MANIFEST.json">manifest</a> · <a href="PORTABLE_GALLERY_KO.md">Markdown</a> · <a href="COMPLETE.json">완료</a></p><a href="PANELS_12.png"><img src="PANELS_12.png" alt="12조건 overview"></a><div class="cards">'+''.join(cards)+'</div></html>')
 complete=dict(completed=True,PASS=True,code=binding(Path(__file__).absolute()),protocol=payload['root_authority'],selection=selected,manifest=binding(H/'IMAGE_MANIFEST.json'),row_CSV=binding(H/'ALL12_SELECTED_STRATA.csv'),overview=binding(H/'PANELS_12.png'),portable_MD=binding(H/'PORTABLE_GALLERY_KO.md'),HTML=binding(H/'index.html'),unique_global_indices=unique,unique_examples=len(unique),selected_cells=12,missing_cells=selection['missing_cells'],duplicate_cases_preserved=True,all_original_META_and_saved_pixels_unchanged=True,original_INPUT_NPZ_rehash_exact=True,RAW_NPZ_copy_or_publish=False,model_forward_fit_initializer_recrop_PnP_or_GTchanges=0,source_or_protected_bodies_outside_explicit_current_scope=0,no_new_visibility_or_native_gold_labels=True,not_representative_or_causal_proof=True,actual_image_tool_views_by_builder=0)
 cb=write(H/'COMPLETE.json',complete);print(json.dumps(dict(completed=True,PASS=True,complete=cb,unique_global_indices=unique,selected_cells=[{k:r[k]for k in['group','stratum','global_index','missing','matching_count']}for r in cells]),ensure_ascii=False))
if __name__=='__main__':main()
