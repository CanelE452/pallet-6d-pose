"""V206 sealed weak visible-feature ROI join and fixed8 display; no fit or new labels."""
from pathlib import Path
from collections import Counter
import argparse,copy,csv,hashlib,html,io,json,os,stat,traceback
from PIL import Image,ImageDraw,ImageFont
H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
SCOPE=V/'ROOT_PROSPECTIVE_SOURCE_TRAIN_VISIBLE_PART_WITNESS_SCOPE.json';SCOPE_SHA='2d90d7c29994cb02bb0c24fd2556e97493f0dfee2b7b84ffadefeac24cc6c4b4'
RULE=V/'ROOT_PROSPECTIVE_WITNESS_JOIN_AND_FIXED8_VISUAL_RULE.json';RULE_SHA='2e5ce8883e804f642abcbe0f8da60a0583138798aaffce5b6482c0dceb52d774'
SELECTION=V/'SOURCE8_WITNESS_CASES_BEFORE_NEW_ANNOTATION.json';SELECTION_SHA='7600afe3cd0039ec2055f9cd094b99b6b4a73344c28d41baf0c457a53843a297'
A_READY=V/'observer_a/PREPARATION_COMPLETE.json';A_SHA='cd723bea2134c078a7452245170c1ccf9988431f0e6cbde091a59d9027b1efa0'
C_READY=V/'observer_c/FINAL_READY.json';C_SHA='d0cc2ab36f7da6958148de2ae7f3cd45698915e170f49396661ca27ae2e6b40d'
PROTOCOL=H/'PROTOCOL_BEFORE_SEALED_ROI_VALUES_OR_SELECTED_PNGS.json';OUT=V/'joint_witness_display_v1'
STATE='visible_intended_part_identified_YES_UNKNOWN';ROI='witness_ROI_raw96_xyxy_or_null'
EDITABLE={STATE,ROI,'visible_feature_description','positive_local_visual_evidence','ambiguity_note','actually_viewed_image_bindings'}
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def need(x,msg):
 if not x:raise AssertionError(msg)
def safe(path):
 path=Path(path);need(path.is_absolute()and P in path.parents and '..'not in path.parts and 'v112'not in str(path).casefold(),'Forbidden/outside path before filesystem');return path
def opened(path,write=False):
 path=safe(path);fd=os.open(path.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for part in path.parts[1:-1]:n=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
  n=os.open(path.name,(os.O_WRONLY|os.O_CREAT|os.O_EXCL if write else os.O_RDONLY)|os.O_NOFOLLOW,0o644,dir_fd=fd)
 finally:os.close(fd)
 need(stat.S_ISREG(os.fstat(n).st_mode),'Regular file/no symlink');return os.fdopen(n,'wb'if write else'rb')
def rb(p):
 with opened(p)as f:return f.read()
def digest(z):return hashlib.sha256(z).hexdigest()
def bind(p):return dict(path=str(safe(p)),sha256=digest(rb(p)))
def get(b):
 path=safe(b['path']);need(V in path.parents and path.suffix=='.json','Ownphase JSON metadata only')
 data=rb(path);need(digest(data)==b['sha256'],'Bound metadata SHA drift');return json.loads(data)
def write(p,d):
 with opened(p,True)as f:f.write((json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
def text(p,s):
 with opened(p,True)as f:f.write(s.encode())
def pinned(p,sha):
 b=bind(p);need(b['sha256']==sha,'Pinned code/preparation metadata');return b

def roi_valid(box):
 need(isinstance(box,list)and len(box)==4 and all(type(x)is int for x in box),'Inclusive integer ROI, bool/float forbidden')
 x0,y0,x1,y1=box;need(0<=x0<=x1<=95 and 0<=y0<=y1<=95,'Raw96 inclusive0..95 order')
 return (x1-x0+1)*(y1-y0+1)
def geometry(a,c):
 if a is None or c is None:return dict(category='BOTH_UNKNOWN'if a is None and c is None else'ONE_UNKNOWN',area_A=None if a is None else roi_valid(a),area_C=None if c is None else roi_valid(c),intersection_area=None,union_area=None,IoU=None,enclosing_union_bbox=None)
 aa=roi_valid(a);ac=roi_valid(c);inter=max(0,min(a[2],c[2])-max(a[0],c[0])+1)*max(0,min(a[3],c[3])-max(a[1],c[1])+1);union=aa+ac-inter
 return dict(category='BOTH_YES_OVERLAP'if inter else'BOTH_YES_DISJOINT',area_A=aa,area_C=ac,intersection_area=inter,union_area=union,IoU=inter/union,enclosing_union_bbox=[min(a[0],c[0]),min(a[1],c[1]),max(a[2],c[2]),max(a[3],c[3])])
def allowed_images(case):
 s=case['original_slot'];f=case['original_frame_inputs'];return dict(clean96=s['clean96'],valid96=s['valid96'],fulloriginal=f['fulloriginal'],overlay=f['overlay'],all8contact=f['all8contact'])
def validate_rows(rows,template,cases,observer):
 need(len(rows)==len(template)==len(cases)==8 and len({r['row_id']for r in rows})==8,'All fixed8 unique rows retained')
 for r,t,case in zip(rows,template,cases):
  need(set(r)==set(t)and all(r[k]==t[k]for k in set(t)-EDITABLE),'Only editable6 fields; ordered identity/qL/native/priorflags/fitfalse immutable')
  s=case['original_slot'];need(r['row_id']==s['row_id']and r['observer']==observer and r['training_admission']is False,'Fixed case/observer/no fit')
  if observer=='A':
   ri=r['original_input'];need(all(ri[k]==s[k]for k in ('frame_key','original_frame_index','native_slot','query_xy','L','original_query_available','crop_constructed')),'A original input snapshot exact')
   need(r['original_image_bindings']==allowed_images(case),'A image binding snapshot exact')
  else:
   need(all(r[k]==s[k]for k in ('frame_key','original_frame_index','native_slot','original_query_available'))and r['original_query_xy']==s['query_xy']and r['original_L']==s['L'],'C original input snapshot exact')
   need(r['prior_relation_unchanged']==case['prior_relation']and r['prior_location_flags_unchanged']==case['prior_location_flags']and r['corrected_query_or_localization_output']is False and r['physical_native_gold']is False,'C original prior/authority flags immutable')
  need(r[STATE]in ('YES','UNKNOWN'),'No unreviewed rows in actual join')
  if r[STATE]=='YES':roi_valid(r[ROI]);need(s['original_query_available']is True and r['visible_feature_description'].strip()and r['positive_local_visual_evidence'].strip(),'YES needs original available and positive identified visual feature')
  else:need(r[ROI]is None and r['ambiguity_note'].strip(),'UNKNOWN null+reason, never imputed')
  imgs=allowed_images(case);views=r['actually_viewed_image_bindings'];need(isinstance(views,list)and len({(b['path'],b['sha256'])for b in views})==len(views)and all(b in imgs.values()for b in views),'Samecase exact actual view bindings only')
  need(imgs['clean96']in views and any(imgs[k]in views for k in ('fulloriginal','overlay')),'Actual samecase CLEAN96+full/q8 context required')
 return True

def source_metadata():
 scope=get(pinned(SCOPE,SCOPE_SHA));rule=get(pinned(RULE,RULE_SHA));selection=get(pinned(SELECTION,SELECTION_SHA));a=get(pinned(A_READY,A_SHA));c=get(pinned(C_READY,C_SHA));at=get(a['template'])['rows'];ct=get(c['template'])['rows'];cases=selection['source_cases8']
 need(scope['fixed_case_selection']==pinned(SELECTION,SELECTION_SHA)and len(cases)==8 and len({x['original_slot']['frame_key']for x in cases})==3,'Fixed original8/3sourceframes')
 need(rule['all8_fixed_source_cases_retained']and rule['no_automatic_externalE_or_training_admission'],'Same prospective descriptive rule/no labels')
 return cases,at,ct,dict(scope=pinned(SCOPE,SCOPE_SHA),rule=pinned(RULE,RULE_SHA),selection=pinned(SELECTION,SELECTION_SHA),observer_A_ready=pinned(A_READY,A_SHA),observer_C_ready=pinned(C_READY,C_SHA),observer_A_template=a['template'],observer_C_template=c['template'])
def require_seals(entry):
 # Validate BOTH seals before opening either observer's newly annotated values.
 for observer,opposite in (('A','C'),('C','A')):
  d=entry['observers'][observer];complete=get(d['complete']);validation=get(d['validation']);ind=get(d['independence'])
  need(complete['completed']is True and complete['PASS']is True and validation['completed']is True and validation['PASS']is True,'Both complete and validator PASS required before ROI reads')
  key=d['independence_zero_field'];name=key.casefold();need('roi'in name and 'read'in name and ('other'in name or ('new_'+opposite.casefold())in name),'Explicit OTHER new ROI read field')
  need(type(ind[key])in(int,bool)and ind[key]==0,'Other new ROI values not read before own seal')
 return True

def png(b,whitelist,mode):
 need(b in whitelist and Path(b['path']).suffix=='.png','Only selected16 CLEAN96/VALID96 generated PNGs')
 data=rb(b['path']);need(digest(data)==b['sha256'],'Selected PNG SHA')
 with Image.open(io.BytesIO(data))as im:need(im.mode==mode and im.size==(96,96),'Original96 PNG mode/size');return im.copy(),data

def card(row,clean,valid):
 sheet=Image.new('RGB',(1330,530),'white');d=ImageDraw.Draw(sheet);f=lambda n:ImageFont.truetype(FONT,n)
 d.text((16,10),row['row_id'],font=f(19),fill='black');d.text((16,39),'Weak visible-feature rectangles only; not corrected q / keypoint / physical gold. No E or fit admission.',font=f(16),fill='black')
 for x,title in [(16,'Original CLEAN96'),(430,'Separate A green / C magenta + original q'),(844,'Enclosing union bbox (display only)')]:
  panel=clean.copy().resize((384,384),Image.Resampling.NEAREST);pd=ImageDraw.Draw(panel)
  if x!=16:
   q=(47.5+.5)*4-.5;pd.line([(q-10,q),(q+10,q)],fill='cyan',width=2);pd.line([(q,q-10),(q,q+10)],fill='cyan',width=2)
   pairs=[(row['ROI_A'],'lime'),(row['ROI_C'],'magenta')]if x==430 else[(row['enclosing_union_bbox'],'orange')]
   for box,color in pairs:
    if box is not None:pd.rectangle((box[0]*4,box[1]*4,(box[2]+1)*4-1,(box[3]+1)*4-1),outline=color,width=3)
  sheet.paste(panel,(x,100));d.text((x,72),title,font=f(14),fill='black')
 sheet.paste(valid.convert('RGB'),(1230,100));d.text((1230,201),'valid96',font=f(13),fill='black')
 d.text((16,498),row['category']+' | A='+row['state_A']+' C='+row['state_C']+' | IoU='+str(row['IoU'])+' | UNKNOWN kept; no ROI from q center.',font=f(16),fill='black');return sheet

def tests():
 checks=[]
 def add(n,v):need(v,n);checks.append(n)
 g=geometry([0,0,0,0],[0,0,0,0]);add('Inclusive_onepixel_inter1_union1_IoU1',g['intersection_area']==g['union_area']==1 and g['IoU']==1)
 g=geometry([0,0,1,1],[1,1,2,2]);add('Inclusive_fractional_IoU1over7_unionbbox',g['intersection_area']==1 and g['union_area']==7 and g['IoU']==1/7 and g['enclosing_union_bbox']==[0,0,2,2])
 g=geometry([0,0,0,0],[95,95,95,95]);add('Disjoint_no_query_or_minarea_gate',g['category']=='BOTH_YES_DISJOINT'and g['IoU']==0 and g['union_area']==2)
 add('UNKNOWN_no_union_imputation',geometry(None,[1,2,3,4])['enclosing_union_bbox']is None and geometry(None,None)['category']=='BOTH_UNKNOWN')
 for box in ([-1,0,1,1],[0,0,96,95],[2,0,1,1],[False,0,1,1],[0.,0,1,1]):
  try:roi_valid(box);ok=False
  except AssertionError:ok=True
  add('Invalid_inclusive_ROI_rejected_'+str(len(checks)),ok)
 # Contract drift and cross-case view tests use invented rows, not real ROI values.
 def case(i):return dict(original_slot=dict(row_id='invented'+str(i),frame_key='f'+str(i),original_frame_index=i,native_slot=i,query_xy=[17.125,22.5],L=200.25,original_query_available=True,crop_constructed=True,clean96=dict(path='clean'+str(i),sha256='a'*64),valid96=dict(path='valid'+str(i),sha256='b'*64)),original_frame_inputs=dict(fulloriginal=dict(path='full'+str(i),sha256='c'*64),overlay=dict(path='overlay'+str(i),sha256='d'*64),all8contact=dict(path='contact'+str(i),sha256='e'*64)),prior_relation='NEAR',prior_location_flags=dict(A='NO',C='YES'))
 cases=[case(i)for i in range(8)];template=[]
 for x in cases:
  s=x['original_slot'];template.append(dict(row_id=s['row_id'],observer='C',frame_key=s['frame_key'],original_frame_index=s['original_frame_index'],native_slot=s['native_slot'],original_query_xy=s['query_xy'],original_L=s['L'],original_query_available=True,prior_relation_unchanged='NEAR',prior_location_flags_unchanged=x['prior_location_flags'],corrected_query_or_localization_output=False,physical_native_gold=False,training_admission=False,**{STATE:'UNREVIEWED',ROI:None,'visible_feature_description':'','positive_local_visual_evidence':'','ambiguity_note':'','actually_viewed_image_bindings':[]}))
 rows=copy.deepcopy(template)
 for r,x in zip(rows,cases):r.update({STATE:'YES',ROI:[95,95,95,95],'visible_feature_description':'invented feature','positive_local_visual_evidence':'invented local edge','actually_viewed_image_bindings':[allowed_images(x)['clean96'],allowed_images(x)['fulloriginal']]})
 add('Fixed8_original_meta_and_samecase_views_ACCEPT',validate_rows(rows,template,cases,'C'))
 for name,mutate in [('qdrift',lambda x:x[0].update(original_query_xy=[0.,0.])),('fittrue',lambda x:x[0].update(training_admission=True)),('wrongcaseview',lambda x:x[0].update(actually_viewed_image_bindings=rows[1]['actually_viewed_image_bindings'])),('reorder',lambda x:x.reverse())]:
  bad=copy.deepcopy(rows);mutate(bad)
  try:validate_rows(bad,template,cases,'C');ok=False
  except AssertionError:ok=True
  add(name+'_rejected',ok)
 return dict(completed=True,PASS=True,invented_only=True,checks=checks,count=len(checks),actual_ROI_or_image_reads=0)

def prepare():
 cases,at,ct,inputs=source_metadata();need(all(x[STATE]=='UNREVIEWED'and x[ROI]is None for x in at+ct),'Empty templates only during prepare')
 checks=tests();write(H/'INVENTED_CHECKS.json',checks);pngs=[b for x in cases for b in (x['original_slot']['clean96'],x['original_slot']['valid96'])];need(len(pngs)==16 and len({x['path']for x in pngs})==16,'Exact selected16PNG bindings')
 protocol=dict(completed=True,prepared_only=True,code=bind(__file__),inputs=inputs,checks=bind(H/'INVENTED_CHECKS.json'),source_png_whitelist=pngs,output=str(OUT),editable_fields6=sorted(EDITABLE),sealing='Both observer COMPLETE+validator completed/PASS and other-new-ROI-read0 exactbound receipts before either new observation JSON read. All8/template immutable fields/META/priorflags/fitfalse and samecase CLEAN96+full/q8context actual viewbinding enforced again.',geometry='Integer inclusive0..95. Keep both original ROIs; intersection/union area and IoU are descriptive. Enclosing bbox is not a union mask. Any UNKNOWN -> no union/IoU imputation. No threshold/q containment/center/minarea/locationflag gate.',pixels='Rootentry only selected16 preexisting CLEAN96+VALID96 PNG bytes; no new source NPZ/RGB/GT/weights/model/eval. Original PNGs never written. Display nearest4x, separate boxes+union and cyan47.5 query display only.',authority='Independent assistant SOURCE TRAIN weak visible-feature witness, not corrected query/keypoint/physicalnative gold. No E labels/fitadmission/modelselection. All8 actual semantic overlay views remain pending root review.',actual_new_ROI_values_or_image_reads=0)
 write(PROTOCOL,protocol)
 observer={k:dict(observations=dict(path=None,sha256=None),complete=dict(path=None,sha256=None),validation=dict(path=None,sha256=None),independence=dict(path=None,sha256=None),independence_zero_field=None)for k in ('A','C')}
 write(H/'ROOT_ACTUAL_ENTRY_TEMPLATE_APPROVED_FALSE.json',dict(approved_sealed_joint_witness_display=False,code=protocol['code'],protocol=bind(PROTOCOL),inputs=inputs,source_png_whitelist=pngs,observers=observer,training_admission=False,new_E_labels=False,automatic_retry=False))
 write(H/'FINAL_READY.json',dict(completed=True,PASS=True,prepared_only=True,code=protocol['code'],protocol=bind(PROTOCOL),checks=protocol['checks'],false_entry=bind(H/'ROOT_ACTUAL_ENTRY_TEMPLATE_APPROVED_FALSE.json'),actual_new_ROI_values_or_images_read=0));print(json.dumps(dict(completed=True,PASS=True,ready=bind(H/'FINAL_READY.json'),checks=checks['count'])))

def run(ep):
 entry=json.loads(rb(ep));need(entry['approved_sealed_joint_witness_display']is True and entry['training_admission']is False and entry['new_E_labels']is False and entry['automatic_retry']is False,'Root actual sealed-only display authorization')
 p=get(entry['protocol']);need(entry['code']==p['code']==bind(__file__)and entry['protocol']==bind(PROTOCOL),'Frozen join code/protocol')
 cases,at,ct,inputs=source_metadata();need(inputs==p['inputs']==entry['inputs']and entry['source_png_whitelist']==p['source_png_whitelist'],'Frozen selection/templates/PNG whitelist')
 require_seals(entry)
 docs={k:get(entry['observers'][k]['observations'])for k in ('A','C')};ar=docs['A']['rows'];cr=docs['C']['rows'];validate_rows(ar,at,cases,'A');validate_rows(cr,ct,cases,'C')
 need(not OUT.exists(),'Exclusive actual output/no retry');OUT.mkdir()
 try:
  rows=[];pngbindings=[];cards=[]
  for i,(case,a,c)in enumerate(zip(cases,ar,cr)):
   s=case['original_slot'];g=geometry(a[ROI],c[ROI]);row=dict(row_id=s['row_id'],frame_key=s['frame_key'],native_slot=s['native_slot'],original_query_xy=s['query_xy'],original_L=s['L'],original_query_available=s['original_query_available'],state_A=a[STATE],state_C=c[STATE],ROI_A=copy.deepcopy(a[ROI]),ROI_C=copy.deepcopy(c[ROI]),feature_A=a['visible_feature_description'],feature_C=c['visible_feature_description'],positive_evidence_A=a['positive_local_visual_evidence'],positive_evidence_C=c['positive_local_visual_evidence'],ambiguity_A=a['ambiguity_note'],ambiguity_C=c['ambiguity_note'],**g,semantic_feature_agreement='PENDING_FIXED8_ACTUAL_VISUAL_REVIEW',corrected_keypoint_or_new_E_label=False,training_admission=False)
   clean,data=png(s['clean96'],p['source_png_whitelist'],'RGB');valid,vdata=png(s['valid96'],p['source_png_whitelist'],'L');folder=OUT/('CASE_%02d'%i);folder.mkdir()
   with opened(folder/'ORIGINAL_CLEAN96.png',True)as f:f.write(data)
   with opened(folder/'ORIGINAL_VALID96.png',True)as f:f.write(vdata)
   image=card(row,clean,valid);cardpath=folder/'WITNESS_A_C_UNION_DISPLAY.png'
   with opened(cardpath,True)as f:image.save(f,format='PNG')
   images={k:bind(folder/name)for k,name in [('clean96','ORIGINAL_CLEAN96.png'),('valid96','ORIGINAL_VALID96.png'),('card','WITNESS_A_C_UNION_DISPLAY.png')]};pngbindings.extend(images.values());cards.append(dict(position=i,row_id=row['row_id'],images=images,original_source_images=allowed_images(case)));rows.append(row)
  write(OUT/'JOINED_WITNESS8.json',dict(completed=True,PASS=True,rows=rows,observer_A_original=entry['observers']['A']['observations'],observer_C_original=entry['observers']['C']['observations'],root_entry=bind(ep),original8_order_preserved=True,no_new_E_or_training_admission=True))
  with opened(OUT/'JOINED_WITNESS8.csv',True)as f:
   buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);f.write(buf.getvalue().encode())
  write(OUT/'FIXED8_DISPLAY_MANIFEST.json',dict(completed=True,PASS=True,cards=cards,source_png_whitelist=p['source_png_whitelist'],actual_fixed8_toolviews='PENDING_ROOT_REVIEW'))
  md=['# V206 SOURCE TRAIN visible-feature witness8','','두 assistant 원 ROI를 각각 보존한다. 사각형은 weak visible-feature/향후개입 근거이며 corrected q·keypoint·정답위치·physical gold가 아니다. 청록47.5십자는 원 query 표시일 뿐이다. UNKNOWN과 불일치를 보존하며 새 E·fit admission은 없다.','', 'union bbox는 두 사각형을 둘러싼 표시용 상자이며 segmentation mask가 아니다. overlap/IoU는 기술 통계이고 문턱/모델선택 기준이 아니다. 고정8 모두의 의미적 실제 검토는 root가 별도로 기록한다.','']
  h=['<!doctype html><meta charset="utf-8"><h1>V206 fixed8 weak witness displays</h1><p>Separate original A/C boxes, no corrected keypoint, E label or fit admission. Unknown kept; semantic tool review pending.</p>']
  for r,c in zip(rows,cards):
   rel=Path(c['images']['card']['path']).relative_to(OUT).as_posix();md+=['## '+r['row_id'],'',f"{r['category']} | A={r['ROI_A']} / C={r['ROI_C']} | IoU={r['IoU']} (descriptive only)",'',f'![고정 weak witness 표시]({rel})',''];h+=['<h2>'+html.escape(r['row_id'])+'</h2><p>'+html.escape(str({k:r[k]for k in ('category','ROI_A','ROI_C','IoU')}))+'</p><img style="max-width:100%" src="'+rel+'">']
  text(OUT/'PORTABLE_GALLERY_KO.md','\n'.join(md));text(OUT/'index.html','\n'.join(h))
  for b in p['source_png_whitelist']:need(bind(Path(b['path']))==b,'Original selected PNG remains unchanged')
  complete=dict(completed=True,PASS=True,code=p['code'],protocol=entry['protocol'],entry=bind(ep),inputs=inputs,observer_seals=entry['observers'],joined=bind(OUT/'JOINED_WITNESS8.json'),CSV=bind(OUT/'JOINED_WITNESS8.csv'),display_manifest=bind(OUT/'FIXED8_DISPLAY_MANIFEST.json'),gallery=bind(OUT/'PORTABLE_GALLERY_KO.md'),HTML=bind(OUT/'index.html'),fixed8_rows_preserved=True,categories=dict(Counter(r['category']for r in rows)),actual_semantic_tool_review='PENDING_ROOT_FIXED8_VIEW',source_images_unchanged=True,source_PNGs_read=16,source_array_or_GT_model_eval_reads=0,new_E_labels=False,training_admission=False,automatic_retry=False)
  write(OUT/'COMPLETE.json',complete);print(json.dumps(dict(completed=True,PASS=True,complete=bind(OUT/'COMPLETE.json'),categories=complete['categories'])))
 except BaseException:
  write(OUT/'FAILED_NO_AUTOMATIC_RETRY.json',dict(completed=False,PASS=False,traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':
 ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);args=ap.parse_args();prepare()if args.prepare else run(args.entry)
