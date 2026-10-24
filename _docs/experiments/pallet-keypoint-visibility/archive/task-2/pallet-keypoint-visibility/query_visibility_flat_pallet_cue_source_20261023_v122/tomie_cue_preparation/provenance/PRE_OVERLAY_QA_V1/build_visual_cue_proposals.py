"""PUBLIC TOMIE fixed4 visual cue proposals; no model or teacher construction."""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import json,csv,hashlib,ast,html,collections
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
S=P/'query_visibility_public_flat_pallet_pilot_20261023_v103/raw_pilot/selected_members'
ROOT=V/'ROOT_SOURCE_SCOPE_BEFORE_NEW_LABELS.json'
EXT=P/'query_visibility_public_flat_pallet_pilot_20261023_v103/raw_pilot/EXTRACTION_COMPLETE_V2.json'
GEN=P/'query_visibility_external_only_bounded_prior_20261023_v102/public_real_extension_intake/TOMIE_generator_py.txt'
FON='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT=ImageFont.truetype(FON,18);SMALL=ImageFont.truetype(FON,14);BIG=ImageFont.truetype(FON,23)
CLUSTER='TOMIE:scenario_3_without_dist_LVL_0:ALL_CAMERAS_ALL_TIME_ONE_EPISODE'
D={
0:[('BLUE',[178,160,320,252],False,[(259,164,'FAR_DECK_RIM_CORNER',5),(312,177,'RIGHT_DECK_RIM_CORNER',5),(231,235,'NEAR_DECK_RIM_CORNER',5),(182,218,'LEFT_DECK_RIM_CORNER',5)]),('PALE',[272,106,404,187],False,[(398,121,'RIGHT_FAR_DECK_RIM_CORNER',5),(328,166,'NEAR_DECK_RIM_SUPPORT_TURN',5),(276,154,'LEFT_DECK_RIM_CORNER',5)])],
1:[('BLUE',[178,159,322,253],False,[(260,163,'FAR_DECK_RIM_CORNER',5),(313,177,'RIGHT_DECK_RIM_CORNER',5),(231,235,'NEAR_DECK_RIM_CORNER',5),(182,218,'LEFT_DECK_RIM_CORNER',5)]),('PALE',[1150,350,1296,475],True,[(1168,353,'UNCUT_UPPER_LEFT_DECK_RIM_CORNER',6),(1159,412,'UNCUT_LOWER_LEFT_DECK_RIM_CORNER',6)])],
2:[('BLUE',[842,147,974,234],False,[(849,154,'LEFT_FAR_DECK_RIM_CORNER',5),(911,149,'RIGHT_FAR_DECK_RIM_CORNER',5)]),('PALE',[886,205,1023,306],False,[(891,216,'LEFT_FAR_DECK_RIM_CORNER',5),(962,208,'RIGHT_FAR_DECK_RIM_CORNER',5),(943,287,'NEAR_LEFT_DECK_RIM_SUPPORT_TURN',6)])],
3:[('BLUE',[842,147,974,234],False,[(848,154,'LEFT_FAR_DECK_RIM_CORNER',5),(912,149,'RIGHT_FAR_DECK_RIM_CORNER',5),(967,211,'RIGHT_NEAR_DECK_RIM_CORNER',5),(900,216,'LEFT_NEAR_DECK_RIM_CORNER',5)])]}
FRAME=[('camera_0','0.jpg','04.csv','05.csv'),('camera_0','1101.jpg','04.csv','05.csv'),('camera_7','0.jpg','06.csv','07.csv'),('camera_7','1100.jpg','06.csv','07.csv')]
def bind(p):
 p=Path(p);return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def jl(p,rs):
 with Path(p).open('x')as f:
  for r in rs:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
def iou(a,b):
 z=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));u=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-z
 return z/u if u>0 else 0.
def matched(fn,camera,name):
 with (S/fn).open()as f:
  return [dict(csv_data_row_index=i,raw_fields=r) for i,r in enumerate(csv.DictReader(f)) if r['fileName']==camera+'/images/'+name]
def marked(im,inst,cues):
 out=im.copy();d=ImageDraw.Draw(out)
 for a in inst:d.rectangle(a['visual_bbox_xyxy_native_half_open'],outline=(255,0,220),width=2);d.text((a['visual_bbox_xyxy_native_half_open'][0],max(0,a['visual_bbox_xyxy_native_half_open'][1]-18)),a['appearance_key'],font=SMALL,fill=(255,0,220))
 for c in cues:
  x,y=c['reference_xy_native'];bb=c['uncertainty_bbox_xyxy_closed'];d.rectangle(bb,outline=(255,170,0),width=1);d.line((x-4,y,x+4,y),fill=(0,255,255),width=1);d.line((x,y-4,x,y+4),fill=(0,255,255),width=1);d.text((x+4,y-16),c['cue_id'].split(':')[-1],font=SMALL,fill=(0,255,255))
 return out

def main():
 assert not(H/'AI_WEAK_CUE_PROPOSALS22.jsonl').exists()
 ext=json.loads(EXT.read_text());selected=ext['members'][:4]
 instances=[];cues=[];csvs=[];images={};frame_bind=[]
 for fi,(camera,name,dat,raw)in enumerate(FRAME):
  p=S/f'{fi:02d}.jpg';assert bind(p)['sha256']==selected[fi]['extracted_file']['sha256'];im=Image.open(p).convert('RGB');assert im.size==(1296,1024);images[fi]=im
  dr=matched(dat,camera,name);rr=matched(raw,camera,name);assert len(rr)==8 and len(dr)in(0,8)
  parsed=[]
  for row in dr:
   r=row['raw_fields'];bb=ast.literal_eval(r['BoundingBox']);positive=bb!=[-1,-1,-1,-1] and bb[2]>0 and bb[3]>0
   parsed.append({'ObjectName':r['ObjectName'],'raw_BoundingBox':bb,'bbox_xyxy_from_author_rowcol_heightwidth':[bb[1],bb[0],bb[1]+bb[3],bb[0]+bb[2]]if positive else None,'Visible_raw_unchanged':r['Visible'],'Occlusion_raw_unchanged':r['Occlusion'],'csv_data_row_index':row['csv_data_row_index']})
  cf={'frame_index':fi,'source_frame':camera+'/images/'+name,'RGB':bind(p),'data_csv':bind(S/dat),'raw_csv':bind(S/raw),'matching_data_rows':dr,'matching_raw_rows':rr,'bbox_interpretation':'Author calc_2d_bbox iterates array rows first, columns second; drawing uses (bb[1],bb[0]) to (bb[1]+bb[3],bb[0]+bb[2]). Interpret as [row_min,col_min,row_span,col_span]; no camera/flip/scale correction applied.','parsed_data_rows':parsed,'no_data_rows_is_missing_not_visibility':not dr};csvs.append(cf)
  for appearance,box,clipped,pts in D[fi]:
   vi=f'TOMIE_VISUAL_F{fi:02d}_{appearance}';candidate=[]
   for z in parsed:
    if z['ObjectName'].startswith('PALLET') and z['bbox_xyxy_from_author_rowcol_heightwidth']is not None:candidate.append({'entity':z['ObjectName'],'csv_bbox_xyxy':z['bbox_xyxy_from_author_rowcol_heightwidth'],'visual_bbox_IoU_diagnostic_only':iou(box,z['bbox_xyxy_from_author_rowcol_heightwidth'])})
   ir={'visual_instance_id':vi,'frame_index':fi,'camera':camera,'source_frame_name':name,'capture_cluster':CLUSTER,'partition':'PUBLIC_TOMIE_DEVELOPMENT_EPISODE_GROUP_NO_TRAIN_ADMISSION','RGB':bind(p),'native_hw':[1024,1296],'appearance_key':appearance,'shape':'LOW_PROFILE_PERFORATED_FLAT_PALLET','material_appearance':'BLUE_MOLDED_PLASTIC_LOOKING'if appearance=='BLUE'else'PALE_MINT_CREAM_MOLDED_PLASTIC_LOOKING_NOT_CERTIFIED_GREEN','material_annotation':'Appearance only; manufacturer/material composition not certified.','visual_bbox_xyxy_native_half_open':box,'bbox_provenance':'Manual visual envelope on raw RGB; not official bbox, not amodal/owner mask, not initializer output.','image_clipped_instance':clipped,'official_entity_id':None,'official_entity_association':'UNKNOWN','official_bbox_overlap_candidates_diagnostic_only':candidate,'official_bbox_max_IoU_to_visual_envelope':max([x['visual_bbox_IoU_diagnostic_only']for x in candidate],default=None),'association_reason':'No matching data.csv rows for camera0/frame0.'if not dr else'Author-format saved entity boxes do not overlap this visible envelope; no correction or entity assignment was inferred.','visible_cues':len(pts),'same_color_across_frames':'May depict same physical pallet, but no new entity/part correspondence is certified; all share one episode group.','initializer_target_or_q_association':'NOT_INSPECTED_OR_CERTIFIED','native_mapping':'UNKNOWN','human_gold':False,'official_cue_gold':False,'training_admission':False};instances.append(ir)
   for ci,(x,y,kind,u)in enumerate(pts):
    assert 0<=x-u and x+u<1296 and 0<=y-u and y+u<1024
    cues.append({'cue_id':vi+f':C{ci+1}','visual_instance_id':vi,'frame_index':fi,'camera':camera,'source_frame_name':name,'capture_cluster':CLUSTER,'partition':ir['partition'],'RGB':ir['RGB'],'native_hw':[1024,1296],'reference_xy_native':[x,y],'uncertainty_radius_native_px':u,'uncertainty_bbox_xyxy_closed':[x-u,y-u,x+u,y+u],'uncertainty_native_cell_footprint_xyxy_half_open':[x-u-.5,y-u-.5,x+u+.5,y+u+.5],'visual_target_bbox_xyxy_native_half_open':box,'shape':ir['shape'],'material_appearance':ir['material_appearance'],'cue_kind':kind,'cue_purpose':'Visible 2D outer-deck/rim/support direction-change of this visual pallet, not arbitrary interior material or hole centre.','weak_state_proposal':'D_VISIBLE_STRUCTURAL_CUE_CONDITIONAL_AI','evidence':'Actual raw full scene plus nearest enlarged unmarked crop and separate marked copy were inspected by reviewer C; no model/segmentation/depth used.','uncertainty_reason':'Small/soft native edges, molded rim rounding and nearby ribs/adjacent pallet; range is localization uncertainty, not certified positive material pixels.','cut_or_ambiguous_point_selected':False,'whole_pallet_visible_claim':False,'source_annotation_tier':'AI_WEAK_PROPOSAL_PENDING_ROOT_VISUAL_REVIEW','official_entity_id':None,'official_entity_association':'UNKNOWN_MANUAL_VISUAL_INSTANCE_USED','initializer_query_or_L_generated':False,'native8_semantic_id':None,'native_mapping':'UNKNOWN','human_gold':False,'training_admission':False})
  frame_bind.append({'frame_index':fi,'source_frame':camera+'/images/'+name,'RGB':bind(p),'matching_data_rows':len(dr),'raw_entity_rows':len(rr),'visual_instances':[x['visual_instance_id']for x in instances if x['frame_index']==fi]})
 jl(H/'MATCHING_CSV_ROWS4_UNCHANGED.jsonl',csvs);jl(H/'VISUAL_INSTANCES7_AI_WEAK.jsonl',instances);jl(H/'AI_WEAK_CUE_PROPOSALS22.jsonl',cues)
 gallery=[]
 for fi,im in images.items():
  ii=[x for x in instances if x['frame_index']==fi];cc=[x for x in cues if x['frame_index']==fi]
  out=Image.new('RGB',(1480,710+len(ii)*370),(248,250,252));d=ImageDraw.Draw(out)
  d.text((18,12),f'TOMIE {fi:02d} | SOURCE-ONLY AI-WEAK STRUCTURAL CUE PROPOSALS',font=BIG,fill=(10,35,60))
  d.text((18,47),'One episode group | manual visual identity | official entity UNKNOWN | no initializer/native labels',font=FONT,fill=(100,45,10))
  raw=im.resize((700,553),Image.Resampling.BILINEAR);mk=marked(im,ii,cc).resize((700,553),Image.Resampling.BILINEAR)
  out.paste(raw,(18,90));out.paste(mk,(760,90));d.text((18,66),'RAW full RGB - display resize only',font=SMALL,fill='black');d.text((760,66),'Separate marks: visual bbox magenta; reference cyan; uncertainty orange',font=SMALL,fill='black')
  for j,inst in enumerate(ii):
   b=inst['visual_bbox_xyxy_native_half_open'];x0=max(0,b[0]-18);y0=max(0,b[1]-18);x1=min(1296,b[2]+18);y1=min(1024,b[3]+18);crop=im.crop((x0,y0,x1,y1));cm=marked(im,[inst],[x for x in cc if x['visual_instance_id']==inst['visual_instance_id']]).crop((x0,y0,x1,y1))
   sc=min(3.,660/crop.width,300/crop.height);sz=(round(crop.width*sc),round(crop.height*sc));yy=710+j*370
   d.text((18,yy-40),inst['visual_instance_id']+(' | clipped instance: uncut cues only'if inst['image_clipped_instance']else''),font=FONT,fill=(10,35,60))
   out.paste(crop.resize(sz,Image.Resampling.NEAREST),(18,yy));out.paste(cm.resize(sz,Image.Resampling.NEAREST),(760,yy));d.text((18,yy+sz[1]+5),'RAW original crop / nearest display; new pixels are not inferred',font=SMALL,fill='black');d.text((760,yy+sz[1]+5),'Marks are proposals, not official/native gold',font=SMALL,fill='black')
  f=H/f'FRAME_{fi:02d}_CUE_PROPOSALS.png';assert not f.exists();out.save(f);gallery.append(bind(f))
 # Exact author-box display only: original saved format, no registration tuning.
 out=Image.new('RGB',(1480,1280),(248,250,252));d=ImageDraw.Draw(out)
 d.text((16,8),'Saved CSV entity bbox diagnostic - no flips/scales/calibration corrections',font=BIG,fill='black')
 for fi,im in images.items():
  z=im.copy();dd=ImageDraw.Draw(z)
  for r in csvs[fi]['parsed_data_rows']:
   bb=r['bbox_xyxy_from_author_rowcol_heightwidth']
   if bb is not None:
    dd.rectangle(bb,outline=(255,120,0),width=3);dd.text((bb[0],bb[1]),r['ObjectName'],font=FONT,fill=(255,120,0))
  for a in instances:
   if a['frame_index']==fi:dd.rectangle(a['visual_bbox_xyxy_native_half_open'],outline=(0,255,255),width=3)
  xx=16+(fi%2)*735;yy=70+(fi//2)*600;out.paste(z.resize((710,561)),(xx,yy));d.text((xx,yy-22),f'{fi:02d}: orange saved CSV boxes; cyan manual visual pallets',font=SMALL,fill='black')
 f=H/'CSV_ENTITY_BBOX_DIAGNOSTIC.png';out.save(f)
 report={'completed':True,'root_scope':bind(ROOT),'source_extraction':bind(EXT),'author_bbox_code':bind(GEN),'code':bind(__file__),'raw_images_inspected':4,'capture_clusters':1,'frames':frame_bind,'visual_instances':7,'weak_visible_cue_proposals':22,'by_appearance_instances':dict(collections.Counter(x['appearance_key']for x in instances)),'source_partition':'All TOMIE camera/time neighbors remain one PUBLIC development episode group; no TRAIN admission.','official_entity_association':'UNKNOWN for all7 manual visual instances. Camera0/frame0 data.csv absent; remaining raw saved author-format bbox maxIoU=0 for all selected visual envelopes. This does not prove metadata invalid: exact exporter/calibration/raw-coordinate alignment remains unverified.','raw_entity_bboxes_unmodified':True,'entity_reassignment_or_registration_tuning':False,'material':'Visual blue/pale molded-plastic-looking flat pallets; pale color not certified saturated green.','omissions':'No pale instance in frame03; far image-clipped corners of frame01 pale excluded. Frame02 blue/pale touching borders and pale near-right truck contact were not proposed as visible cues. No external E or S states inferred from bbox/Visible/Occlusion.','cue_semantics':'Component/rim direction-change visibility, not pixel-material gold. Closed uncertainty±5/6px and inclusive native-cell footprint both saved explicitly.','actual_original_initializer_q_L_changed':False,'new_query_or_L_generated':False,'native_mapping':'UNKNOWN','human_gold':False,'training_admission':False,'new_composite_model_fit_current_V112_PnP':0,'outputs':{'instances':bind(H/'VISUAL_INSTANCES7_AI_WEAK.jsonl'),'cues':bind(H/'AI_WEAK_CUE_PROPOSALS22.jsonl'),'matching_csv':bind(H/'MATCHING_CSV_ROWS4_UNCHANGED.jsonl'),'gallery':gallery,'CSV_bbox_diagnostic':bind(f)},'review_status':'Actual raw/coordinate inspection complete; final proposal overlays awaiting reviewer C display check and root final weak-source review.'}
 write(H/'RESULT.json',report)
 page='<!doctype html><html lang="ko"><meta charset="utf-8"><title>TOMIE fixed4 구조 cue 제안</title><style>body{font:16px sans-serif;max-width:1480px;margin:25px auto;color:#123}img{width:100%}p{line-height:1.6}section{margin:30px 0}</style><h1>공개 TOMIE flat-pallet 구조 cue 제안</h1><p>원본4장·시각 인스턴스7개·약한 visible 단서22개. 파란색 및 밝은색 성형형 외관이며, 공식 entity 연결과 native 대응은 UNKNOWN입니다. 동일 episode 전체는 한 development 그룹입니다. 원래 initializer q/L을 바꾸거나 학습을 실행하지 않았습니다.</p><p>색/재질 자체를 D로 부른 것이 아니라 외곽 rim/support 방향 전환이 실제 보이는 점만 제안했습니다. 잘린 끝·접촉/가림이 불명확한 점은 제외했습니다. 전체 물체가 보인다는 주장은 아닙니다.</p>'
 for f in gallery:page+='<section><img src="'+html.escape(Path(f['path']).name)+'"><p>위: 원본 전체 장면 / 별도 표시. 아래: 원래 native 좌표 관찰용 crop / 별도 uncertainty 표시. 모델 입력이나 공식 정답이 아닙니다.</p></section>'
 page+='<details><summary>공식 CSV bbox 대응 한계</summary><p>저자 코드의 row/column/span 해석을 그대로 표시했습니다. 보이는 blue/pale 인스턴스와 이 raw 좌표의 대응을 인증할 수 없어 manual visual instance ID를 사용합니다. 좌표 반전·이동·entity 재배정은 하지 않았습니다.</p><img src="CSV_ENTITY_BBOX_DIAGNOSTIC.png"></details><p><a href="AI_WEAK_CUE_PROPOSALS22.jsonl">22 cue JSONL</a> · <a href="VISUAL_INSTANCES7_AI_WEAK.jsonl">7 instance JSONL</a> · <a href="RESULT.json">결과/출처</a></p></html>'
 (H/'GALLERY_KO.html').write_text(page)
 print(json.dumps({'result':bind(H/'RESULT.json'),'cues':bind(H/'AI_WEAK_CUE_PROPOSALS22.jsonl'),'instances':bind(H/'VISUAL_INSTANCES7_AI_WEAK.jsonl'),'gallery':gallery},ensure_ascii=False))
if __name__=='__main__':main()
