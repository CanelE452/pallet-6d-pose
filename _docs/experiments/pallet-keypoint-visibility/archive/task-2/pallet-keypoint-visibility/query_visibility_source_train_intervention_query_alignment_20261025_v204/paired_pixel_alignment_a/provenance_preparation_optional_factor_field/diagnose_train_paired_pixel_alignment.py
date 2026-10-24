"""V204 TRAIN-only saved paired RGB diagnostic. Never a model, teacher or relabeler."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,csv,hashlib,html,json,math,os,stat,traceback
import numpy as np
from PIL import Image,ImageDraw,ImageFont
H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
SCOPE=V/'ROOT_PROSPECTIVE_SOURCE_TRAIN_ONLY_DIAGNOSTIC_SCOPE.json';SCOPE_SHA='bf5076237e75ebacfd83335bcba91f387153eef61c3d419a8a1376c334bf1aa8'
PROTO=H/'PROTOCOL_BEFORE_TRAIN_PIXEL_ARRAY_DIAGNOSTIC.json';OUT=H/'actual_train_pixel_alignment_v1'
HALF=(.5,1.5,3.,6.,12.);THRESH=(1,8,32);VARIANTS=('BEFORE','COVER','SHAM')
META=('row_id','original_global_index','native_slot','query_xy','L','original_query_available')
DOMAINS=('SYNTH_FINITE_END','REAL_SEMANTIC_END_WEAK');FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def need(v,m):
 if not bool(v):raise AssertionError(m)
def safe(p):
 p=Path(p).absolute();need(P in p.parents and '..'not in p.parts and 'v112'not in str(p).casefold(),'Exact project path; forbidden lexical refusal before filesystem');return p
def opened(p,write=False):
 p=safe(p);fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for part in p.parts[1:-1]:n=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
  last=os.open(p.name,(os.O_WRONLY|os.O_CREAT|os.O_EXCL if write else os.O_RDONLY)|os.O_NOFOLLOW,0o644,dir_fd=fd)
 finally:os.close(fd)
 need(stat.S_ISREG(os.fstat(last).st_mode),'Regular file/no symlink');return os.fdopen(last,'wb'if write else'rb')
def sha(p):
 h=hashlib.sha256()
 with opened(p)as f:
  for z in iter(lambda:f.read(1<<20),b''):h.update(z)
 return h.hexdigest()
def bind(p):return dict(path=str(safe(p)),sha256=sha(p))
def checked(b):need(bind(b['path'])==b,'Exact frozen binding');return Path(b['path'])
def read(p):
 with opened(p)as f:return json.load(f)
def lines(p):
 with opened(p)as f:return[json.loads(z)for z in f if z.strip()]
def text(p,s):
 with opened(p,True)as f:f.write(s.encode())
def write(p,d):text(p,json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def writecsv(p,rows):
 import io
 if not rows:text(p,'EMPTY\n');return
 buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);text(p,buf.getvalue())
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def key(r):
 k=r['control_join_key'];need(len(k)==4 and k[0]in DOMAINS and k[2]in VARIANTS and type(k[3])is int and 0<=k[3]<8,'Exact control key domain/attempt/canonicalvariant/native');return tuple(k)
def sameq(a,b):return np.array_equal(np.array([np.nan,np.nan]if a is None else a,np.float64),np.array([np.nan,np.nan]if b is None else b,np.float64),equal_nan=True)
def join_train(rows,roles):
 rm={key(r):r for r in roles};need(len(rm)==len(roles),'Unique immutable role keys');need(len({key(r)for r in rows})==len(rows)and set(rm)=={key(r)for r in rows},'All current cache rows exact key set')
 train=[]
 for cache_index,r in enumerate(rows):
  z=rm[key(r)];need(z['role']in('TRAIN','HOLDOUT'),'Original component role enum')
  if z['role']!='TRAIN':continue # No HOLDOUT features/targets/statistics escape this boundary.
  need(r['row_id']==z['row_id']and r['original_global_index']==z['original_global_index']and r['native_slot']==z['native_slot']and r['original_query_available']==z['original_query_available']and r['source_packet']==z['source_packet']and r['authority']==z['authority']and sameq(r['query_xy'],z['query_xy'])and r['L']==z['L'],'TRAIN current control key/META/source authority exact; stale cache_index ignored')
  t=int(z['stored_weak_direct_target']);need(t in(-1,0,1),'Existing weak target only');d={**r,'actual_cache_index':cache_index,'role':'TRAIN','component_id':z['component_id'],'stored_weak_direct_target':t,'selected_native_slot':z['selected_native_slot'],'factors':z['factors']};train.append(d)
 return train

def population(train):
 allgroups=defaultdict(list);known=defaultdict(dict)
 for r in train:
  k=key(r);allgroups[(k[0],k[1],k[3])].append(r)
  if r['stored_weak_direct_target']>=0:
   need(r['native_slot']==r['selected_native_slot']and r['root_admitted']is True,'Only root-admitted requested native weak target');known[(k[0],k[1],k[3])][k[2]]=r
 triplets=[]
 for k,rr in sorted(known.items()):
  need(set(rr)==set(VARIANTS)and[rr[v]['stored_weak_direct_target']for v in VARIANTS]==[0,1,0],'Known TRAIN complete D/E/D triplet')
  before=rr['BEFORE'];need(all(sameq(before['query_xy'],rr[v]['query_xy'])and before['L']==rr[v]['L']and before['original_query_available']==rr[v]['original_query_available']for v in VARIANTS),'Same original q/L/native/availability across paired triplet')
  triplets.append(dict(domain=k[0],attempt_id=k[1],native_slot=k[2],role='TRAIN',component_id=before['component_id'],authority=before['authority'],source_base_index=before['source_base_index'],source_frame_index=before['source_frame_index'],source_donor_id=before['source_donor_id'],query_xy=before['query_xy'],L=before['L'],original_query_available=before['original_query_available'],factors=before['factors'],variant_rows={v:rr[v]['actual_cache_index']for v in VARIANTS},control_join_keys={v:list(key(rr[v]))for v in VARIANTS}))
 counts=[]
 for d in DOMAINS:
  for n in range(8):
   rr=[r for r in train if r['domain']==d and r['native_slot']==n];cc=Counter(r['stored_weak_direct_target']for r in rr)
   counts.append(dict(domain=d,native_slot=n,TRAIN_rows=len(rr),known_D=cc[0],known_E=cc[1],target_UNKNOWN=cc[-1],original_input_U=sum(not r['original_query_available']for r in rr),native_attempt_groups=sum(k[0]==d and k[2]==n for k in allgroups),partial_UNKNOWN_attempt_groups=sum(k[0]==d and k[2]==n and len(v)<3 and all(r['stored_weak_direct_target']==-1 for r in v)for k,v in allgroups.items())))
 return triplets,counts

def stable_selection(triplets):
 result=[]
 for d in DOMAINS:
  rr=[dict(t,stable_attempt_SHA256=hashlib.sha256(t['attempt_id'].encode()).hexdigest())for t in triplets if t['domain']==d]
  result.extend(sorted(rr,key=lambda t:(t['stable_attempt_SHA256'],t['attempt_id']))[:6])
 need(len(result)==12 and len({(r['domain'],r['attempt_id'])for r in result})==12,'Fixed first6 by attempt SHA in each domain')
 return result

def paired_measure(before,after,vbefore,vafter):
 need(before.dtype==after.dtype==np.uint8 and before.shape==after.shape==(96,96,3)and vbefore.dtype==vafter.dtype==bool and vbefore.shape==vafter.shape==(96,96),'Exact original RGB96/valid96')
 valid=vbefore&vafter;delta=np.abs(after.astype(np.int16)-before.astype(np.int16));mag=delta.max(2);yy,xx=np.indices((96,96));dist=np.hypot(xx-47.5,yy-47.5);whole=[];windows=[]
 for threshold in THRESH:
  changed=(mag>=threshold)&valid;num=int(changed.sum());den=int(valid.sum());nearest=float(dist[changed].min())if num else None
  whole.append(dict(absRGB_max_channel_threshold=threshold,valid_intersection_pixels=den,changed_pixels=num,changed_fraction=num/den if den else None,nearest_changed_pixel_distance_to_q96=nearest,mean_max_abs_channel_over_valid=float(mag[valid].mean())if den else None,mean_abs_RGB_channel_over_valid=float(delta[valid].mean())if den else None,max_abs_channel_over_valid=int(mag[valid].max())if den else None,valid_masks_equal=bool(np.array_equal(vbefore,vafter))))
  for half in HALF:
   region=(np.maximum(abs(xx-47.5),abs(yy-47.5))<=half)&valid;nd=int(region.sum());nc=int((changed&region).sum())
   windows.append(dict(absRGB_max_channel_threshold=threshold,query_window_halfwidth96=half,valid_window_pixels=nd,changed_pixels=nc,changed_fraction=nc/nd if nd else None,any_changed_pixel=bool(nc),mean_max_abs_channel_over_valid_window=float(mag[region].mean())if nd else None,mean_abs_RGB_channel_over_valid_window=float(delta[region].mean())if nd else None,max_abs_channel_in_valid_window=int(mag[region].max())if nd else None))
 return whole,windows,mag,valid

def summarize_windows(rows):
 out=[]
 for domain in ('ALL_TRAIN',*DOMAINS):
  for variant in ('COVER','SHAM'):
   for threshold in THRESH:
    for half in HALF:
     rr=[r for r in rows if(domain=='ALL_TRAIN'or r['domain']==domain)and r['variant']==variant and r['absRGB_max_channel_threshold']==threshold and r['query_window_halfwidth96']==half];den=sum(r['valid_window_pixels']for r in rr);num=sum(r['changed_pixels']for r in rr)
     out.append(dict(domain=domain,variant=variant,threshold=threshold,halfwidth96=half,analyzed_TRAIN_triplets=len(rr),window_opportunity_triplets=sum(r['valid_window_pixels']>0 for r in rr),triplets_with_changed_query_window=sum(r['any_changed_pixel']for r in rr),triplets_without_changed_query_window=sum(not r['any_changed_pixel']for r in rr),pooled_valid_pixels=den,pooled_changed_pixels=num,pooled_changed_fraction=num/den if den else None,mean_triplet_changed_fraction=float(np.mean([r['changed_fraction']for r in rr if r['changed_fraction']is not None]))if any(r['changed_fraction']is not None for r in rr)else None))
 return out

def make_card(t,rgbs,valids):
 im=Image.new('RGB',(1500,1060),'white');dr=ImageDraw.Draw(im);font=lambda n:ImageFont.truetype(FONT,n)
 def label(x,y,s,n=19):dr.text((x,y),str(s),fill='black',font=font(n))
 label(20,10,'V204 TRAIN saved pair | '+t['domain']+' | '+t['attempt_id'],24);label(20,48,'native=%d | q=%s | L=%.8g | weak D/E/D, not physical-native gold'%(t['native_slot'],t['query_xy'],t['L']),20)
 label(20,80,'Top: clean original96 x4 DISPLAY, cyan query=(47.5,47.5). Bottom: max abs RGB difference; invalid overlap gray.',18)
 for j,v in enumerate(VARIANTS):
  x=20+j*490;label(x,118,v+' | original label '+str([0,1,0][j]),22);raw=Image.fromarray(rgbs[v]);zoom=raw.resize((384,384),Image.Resampling.NEAREST);zd=ImageDraw.Draw(zoom);q=(47.5+.5)*4-.5;zd.line([(q-12,q),(q+12,q)],fill='cyan',width=2);zd.line([(q,q-12),(q,q+12)],fill='cyan',width=2);im.paste(zoom,(x,154));im.paste(raw,(x+390,154));label(x+390,254,'clean96',15)
 for j,v in enumerate(('COVER','SHAM')):
  x=20+j*730;_,_,mag,val=paired_measure(rgbs['BEFORE'],rgbs[v],valids['BEFORE'],valids[v]);heat=np.zeros((96,96,3),np.uint8);heat[:,:,0]=mag.astype(np.uint8);heat[~val]=[100,100,100];z=Image.fromarray(heat).resize((384,384),Image.Resampling.NEAREST);zd=ImageDraw.Draw(z);q=191.5;zd.line([(q-12,q),(q+12,q)],fill='cyan',width=2);zd.line([(q,q-12),(q,q+12)],fill='cyan',width=2)
  for half in HALF:zd.rectangle([q-4*half,q-4*half,q+4*half,q+4*half],outline=(0,180,255),width=1)
  im.paste(z,(x,594));label(x,560,'BEFORE vs '+v+' | red intensity=abs change; fixed query squares',19)
 label(20,1000,'No recrop, moved query, new label, model output or causal/physical part certification. Stable hash examples are not random prevalence samples.',18)
 return im

def tests():
 checks=[]
 def test(n,v):need(v,n);checks.append(dict(name=n,PASS=True))
 z=np.zeros((96,96,3),np.uint8);a=z.copy();a[47,47,0]=7;v=np.ones((96,96),bool);whole,win,_,_=paired_measure(z,a,v,v)
 test('Closed_halfpixel_window_0p5_has4_centers_and_threshold1_only',win[0]['valid_window_pixels']==4 and win[0]['changed_pixels']==1 and win[5]['changed_pixels']==0 and whole[0]['nearest_changed_pixel_distance_to_q96']==math.sqrt(.5))
 test('Fixed_windows_counts_4_16_36_144_576',[r['valid_window_pixels']for r in win[:5]]==[4,16,36,144,576])
 a=z.copy();a[0,0]=255;w,_,_,_=paired_measure(z,a,v,v);test('UInt8_subtraction_does_not_wrap_and_distance_is_raw96_Euclidean',w[2]['max_abs_channel_over_valid']==255 and abs(w[0]['nearest_changed_pixel_distance_to_q96']-math.sqrt(2)*47.5)<1e-12)
 vv=v.copy();vv[0,0]=False;w,_,_,_=paired_measure(z,a,v,vv);test('Only_valid_intersection_missing_edit_not_zero_distance',w[0]['changed_pixels']==0 and w[0]['nearest_changed_pixel_distance_to_q96']is None and w[0]['valid_intersection_pixels']==9215)
 r=dict(control_join_key=[DOMAINS[0],'INVENTED','BEFORE',0],row_id='invented',original_global_index=0,native_slot=0,query_xy=[1.,2.],L=10.,original_query_available=False,source_packet={'path':'metadata-only'},authority='weak',root_admitted=False,domain=DOMAINS[0],attempt_id='INVENTED',variant='BEFORE',source_base_index=0,source_frame_index=None,source_donor_id=None,actual_cache_index=0)
 role={**r,'role':'TRAIN','stored_weak_direct_target':-1,'selected_native_slot':0,'component_id':'authored','factors':{},'cache_index':999};hold={**r,'control_join_key':[DOMAINS[0],'H','BEFORE',0],'row_id':'hold','attempt_id':'H'};hrole={**role,**hold,'role':'HOLDOUT'}
 train=join_train([hold,r],[role,hrole]);test('Immutable_TRAIN_keyjoin_ignores_stale_index_and_filters_H_before_analysis',len(train)==1 and train[0]['actual_cache_index']==1 and train[0]['original_query_available']is False)
 trip,cc=population(train);test('Partial_UNKNOWN_U_preserved_without_fabricated_DED',not trip and sum(x['target_UNKNOWN']for x in cc)==1 and sum(x['partial_UNKNOWN_attempt_groups']for x in cc)==1)
 broken={**role,'query_xy':[9.,2.]}
 try:join_train([r],[broken]);ok=False
 except AssertionError:ok=True
 test('META_q_drift_rejected_before_pixels',ok)
 fake=[dict(domain=d,attempt_id=str(i))for d in DOMAINS for i in range(8)];sel=stable_selection(fake);test('Stable_first6_per_domain_unchanged_by_input_order',sel==stable_selection(fake[::-1])and len(sel)==12)
 try:safe(P/'blocked_v112_before_stat');ok=False
 except AssertionError:ok=True
 test('Forbidden_name_lexical_refusal',ok)
 return dict(completed=True,PASS=True,invented_only=True,count=len(checks),checks=checks,actual_input_NPZ_or_pixel_reads=0)

def metadata(scope):
 cache=read(checked(scope['direct_cache_complete']));split=read(checked(scope['component_split_complete']));need(cache['completed']and cache['PASS']and split['completed']and split['PASS'],'Closed source cache/split')
 rows=lines(checked(cache['metadata']));rb=split['outputs']['DIRECT_CONTROL_SPLIT_ROWS.jsonl'];roles=lines(checked(rb));need(len(rows)==len(roles)==3144,'Expanded current3144 metadata, no stale V190 row pointer')
 train=join_train(rows,roles);triplets,counts=population(train);need(len(train)==2784 and len(triplets)==109 and Counter(t['domain']for t in triplets)=={DOMAINS[0]:82,DOMAINS[1]:27},'Expected immutable TRAIN109knownDED82synthetic27public, fullTRAIN2784')
 bindings=dict(cache_complete=scope['direct_cache_complete'],split_complete=scope['component_split_complete'],rows=cache['metadata'],roles=rb,inputs=cache['inputs'])
 return cache,train,triplets,counts,bindings

def prepare():
 need(bind(SCOPE)['sha256']==SCOPE_SHA,'Exact root TRAIN-only prospective scope');s=read(SCOPE);need(s['prospective_source_TRAIN_diagnostic_scope']is True,'Source-only scope');cache,train,triplets,counts,bindings=metadata(s)
 selection=stable_selection(triplets);write(H/'FIXED12_BEFORE_PIXEL_READS.json',dict(completed=True,selection_rule='Per domain sort SHA256(attempt_id UTF8), then attempt_id; first6, no pixel/model/outcome used.',selections=selection,count=12,actual_pixels_or_NPZ_headers_read=0));write(H/'TRAIN_POPULATION_BEFORE_PIXEL_READS.json',dict(completed=True,physical_metadata_rows_read=3144,original_TRAIN_rows=2784,known_TRAIN_triplets=109,known_by_domain=dict(Counter(t['domain']for t in triplets)),allnative_status=counts,known_triplets=triplets,UNKNOWN_or_U_not_relabelled=True,HOLDOUT_feature_or_label_aggregate_performed=False));write(H/'INVENTED_CHECKS.json',tests())
 p=dict(completed=True,prepared_only=True,code=bind(__file__),scope=bind(SCOPE),inputs=bindings,selection=bind(H/'FIXED12_BEFORE_PIXEL_READS.json'),population=bind(H/'TRAIN_POPULATION_BEFORE_PIXEL_READS.json'),checks=bind(H/'INVENTED_CHECKS.json'),output=str(OUT),measure=dict(center96=[47.5,47.5],halfwidths96=HALF,thresholds_max_abs_RGB=THRESH,changed='max_c abs(int16(after)-int16(before)) >= threshold, original valid96 intersection only',window='Closed max(abs(x-47.5),abs(y-47.5)) <= halfwidth on integer RAW96 pixel centers',distance='Minimum Euclidean raw96 distance from(47.5,47.5) to thresholded changed valid pixel; None if no changed pixel',magnitude='Mean maxabs and mean absolute RGB channel over all valid window pixels including unchanged; not only changed pixels',comparison='BEFORE->COVER and BEFORE->canonicalSHAM for same current control_join_key domain/attempt/selectednative; qL/availability exact, valid intersection, no source_packet follow',no_threshold_tuning=True),limits=['Only saved known TRAIN109 triplets; full TRAIN allnative UNKNOWN/U administrative census retained.','Opaque container verification/unpack physically includes originalHOLDOUT rows; only TRAIN takes are compared or reported.','Weak finite-end D/E/D authority is unchanged, not human native or physical point gold.','This measures intervention pixel location relative to the originalquery, not endpoint identity or cause of model errors.','Changed pixels away from query can coexist with a valid finite-end intervention; no target/label correction or admission decision.','All inputs copied from immutable saved cache, no resampling/newquery/model/fit/weights/protected evaluation.'],source_or_target_packet_bodies_followed=0,selected_model=None,fully_solved=False)
 write(PROTO,p);write(H/'ROOT_ACTUAL_TRAIN_PIXEL_DIAGNOSTIC_ENTRY_TEMPLATE_APPROVED_FALSE.json',dict(approved_TRAIN_paired_pixel_diagnostic=False,code=p['code'],protocol=bind(PROTO),scope=p['scope'],source_containers_may_physically_include_HOLDOUT=True,analyze_or_report_HOLDOUT=False,weights_model_forward_or_protected_eval=False,automatic_retry=False));write(H/'FINAL_READY.json',dict(completed=True,PASS=True,prepared_only=True,code=p['code'],protocol=bind(PROTO),selection=p['selection'],population=p['population'],checks=p['checks'],false_entry=bind(H/'ROOT_ACTUAL_TRAIN_PIXEL_DIAGNOSTIC_ENTRY_TEMPLATE_APPROVED_FALSE.json'),actual_pixel_NPZ_header_or_weight_reads=0));print(json.dumps(dict(completed=True,PASS=True,ready=bind(H/'FINAL_READY.json'),fixed12=[r['attempt_id']for r in selection])))

def run(ep):
 e=read(ep);need(e['approved_TRAIN_paired_pixel_diagnostic']is True and e['source_containers_may_physically_include_HOLDOUT']is True and e['analyze_or_report_HOLDOUT']is False and e['weights_model_forward_or_protected_eval']is False and e['automatic_retry']is False,'Root authorized TRAIN-only diagnostic entry');p=read(checked(e['protocol']));need(e['code']==p['code']==bind(__file__)and e['protocol']==bind(PROTO)and e['scope']==p['scope']==bind(SCOPE),'Frozen small diagnostic protocol')
 s=read(SCOPE);cache,train,triplets,counts,bindings=metadata(s);need(bindings==p['inputs']and read(checked(p['population']))['known_triplets']==triplets,'Frozen current metadata population');selection=read(checked(p['selection']))['selections'];need(selection==stable_selection(triplets),'Fixed selection unchanged before pixels');need(not OUT.exists(),'Exclusive new actual output/no retry');OUT.mkdir()
 try:
  # Shared container verification/unpack is allowed; take only original TRAIN rows immediately.
  with opened(checked(bindings['inputs']))as f:
   with np.load(f,allow_pickle=False)as z:full={k:z[k]for k in ('rgb96','valid96',*META)}
  take=np.array([r['actual_cache_index']for r in train],np.int64);x={k:a[take].copy()for k,a in full.items()};del full
  need(x['rgb96'].dtype==np.uint8 and x['rgb96'].shape==(2784,96,96,3)and x['valid96'].dtype==bool and x['valid96'].shape==(2784,96,96),'Only TRAIN sliced original96 schema')
  lookup={r['actual_cache_index']:i for i,r in enumerate(train)}
  for i,r in enumerate(train):
   need(str(x['row_id'][i])==r['row_id']and int(x['original_global_index'][i])==r['original_global_index']and int(x['native_slot'][i])==r['native_slot']and bool(x['original_query_available'][i])==r['original_query_available']and sameq(x['query_xy'][i],r['query_xy'])and np.array_equal(x['L'][i],np.array(np.nan if r['L']is None else r['L'],np.float64),equal_nan=True),'All TRAIN rawMETA exact currentkey join')
   need(ash(x['rgb96'][i])==r['raw_RGB96_sha256']and ash(x['valid96'][i])==r['raw_valid96_sha256'],'All TRAIN original RGB/valid byteSHA exact')
  whole=[];windows=[];skipped=[];image_records=[];gallery=[]
  for t in triplets:
   if not t['original_query_available']:skipped.append(dict(domain=t['domain'],attempt_id=t['attempt_id'],reason='ORIGINAL_INPUT_U_NOT_E'));continue
   ix={v:lookup[t['variant_rows'][v]]for v in VARIANTS};rgbs={v:x['rgb96'][i]for v,i in ix.items()};valids={v:x['valid96'][i]for v,i in ix.items()};base={k:t[k]for k in ('domain','attempt_id','native_slot','component_id','authority','source_base_index','source_frame_index','source_donor_id','L')}
   for v in ('COVER','SHAM'):
    ww,win,_,_=paired_measure(rgbs['BEFORE'],rgbs[v],valids['BEFORE'],valids[v]);whole.extend(dict(**base,variant=v,**r)for r in ww);windows.extend(dict(**base,variant=v,**r)for r in win)
   if any(t['domain']==r['domain']and t['attempt_id']==r['attempt_id']for r in selection):
    d=OUT/t['attempt_id'];d.mkdir();card=make_card(t,rgbs,valids);card.save(d/'CARD.png');clean={}
    for v in VARIANTS:
     path=d/(v+'_CLEAN96.png');Image.fromarray(rgbs[v]).save(path)
     with Image.open(path)as im:need(np.array_equal(np.asarray(im),rgbs[v]),'Saved original clean96 PNG pixel exact')
     clean[v]=bind(path)
    rec=dict(domain=t['domain'],attempt_id=t['attempt_id'],native_slot=t['native_slot'],source_base_index=t['source_base_index'],source_frame_index=t['source_frame_index'],card=bind(d/'CARD.png'),clean96=clean,originalq=t['query_xy'],L=t['L'],no_query_or_crop_change=True);image_records.append(rec);gallery.append((rec['attempt_id'],Path(rec['card']['path']).relative_to(OUT).as_posix()))
  need(len(whole)==109*2*3 and len(windows)==109*2*3*5 and not skipped and len(image_records)==12,'All109 knownavailable comparisons/fixed12 preserved')
  summary=summarize_windows(windows);writecsv(OUT/'TRAIN_ALL_NATIVE_TARGET_STATUS.csv',counts);writecsv(OUT/'PAIR_WHOLE96_THRESHOLDS.csv',whole);writecsv(OUT/'QUERY_CENTER_FIXED_WINDOWS.csv',windows);writecsv(OUT/'QUERY_CENTER_WINDOW_SUMMARY.csv',summary)
  cover=[r for r in whole if r['variant']=='COVER'and r['absRGB_max_channel_threshold']==1];distances={d:[r['nearest_changed_pixel_distance_to_q96']for r in cover if r['domain']==d and r['nearest_changed_pixel_distance_to_q96']is not None]for d in DOMAINS}
  summary_json=dict(completed=True,PASS=True,scope=p['scope'],original_TRAIN_rows=2784,known_TRAIN_DED_triplets=109,known_by_domain=dict(Counter(t['domain']for t in triplets)),weak_authority_by_domain={d:sorted({t['authority']for t in triplets if t['domain']==d})for d in DOMAINS},source_coverage={d:dict(triplets=sum(t['domain']==d for t in triplets),bases=sorted({t['source_base_index']for t in triplets if t['domain']==d and t['source_base_index']is not None}),frames=sorted({t['source_frame_index']for t in triplets if t['domain']==d and t['source_frame_index']is not None}),components=len({t['component_id']for t in triplets if t['domain']==d}),native_known_counts=dict(Counter(t['native_slot']for t in triplets if t['domain']==d)))for d in DOMAINS},allnative_TRAIN_counts=counts,window_summary=summary,cover_threshold1_nearest_distance96={d:dict(N=len(a),min=min(a)if a else None,median=float(np.median(a))if a else None,max=max(a)if a else None)for d,a in distances.items()},skipped_preserved=skipped,physically_unpacked_shared_container_rows=3144,HOLDOUT_rows_analyzed_or_reported=0,selection=p['selection'],limits=p['limits'],new_labels_model_fit_or_data_threshold_selection=False,selected_model=None,fully_solved=False)
  write(OUT/'SUMMARY.json',summary_json);write(OUT/'IMAGE_MANIFEST.json',dict(completed=True,images=image_records,actual_toolviews='PENDING_SEPARATE_RECORD',all12_or_all109_semantic_view_claim=False))
  md=['# V204 TRAIN paired pixel / 원 query 정렬 진단','','TRAIN109 known D/E/D(합성82·공개실사 weak27)를 동일 원 q/L/native에서 비교했다. 두 source 권위는 서로 다르며 human native gold가 아니다. threshold는고정 maxabsRGB≥1/8/32이고valid intersection만 센다.','', '[수치요약](SUMMARY.json) · [window 전3270행](QUERY_CENTER_FIXED_WINDOWS.csv) · [window 집계90행](QUERY_CENTER_WINDOW_SUMMARY.csv) · [whole96 전654행](PAIR_WHOLE96_THRESHOLDS.csv) · [allnative TRAIN UNKNOWN/U 분모](TRAIN_ALL_NATIVE_TARGET_STATUS.csv)','', '그림은 입력 변경이 없는 원 crop의 표시 사본과 차이 heatmap이다. q가 아닌 곳의 edit가 원 finite-end를 가렸는지, 모델이 그 정보를 사용했는지까지 인증하지 않는다. stablehash 각source6은 무작위 대표표본이 아니다.','']
  body='<h1>V204 fixed TRAIN paired RGB / original query</h1><p>109 known D/E/D, 82 synthetic /27 public weak. Original q/L/native/crop unchanged. No model or relabeling.</p>'
  for name,path in gallery:md+=['## '+name,'',f'![고정 TRAIN {name}]({path})',''];body+='<h2>'+html.escape(name)+'</h2><img style="max-width:100%" src="'+path+'">'
  md+=['## 한계','',*['- '+q for q in p['limits']],''];text(OUT/'PORTABLE_GALLERY_KO.md','\n'.join(md));text(OUT/'index.html','<!doctype html><meta charset="utf-8">'+body)
  checked(bindings['inputs'])
  for b in (p['scope'],p['selection'],p['population'],bindings['cache_complete'],bindings['split_complete'],bindings['rows'],bindings['roles']):checked(b)
  outputs={str(q.relative_to(OUT)):bind(q)for q in sorted(OUT.rglob('*'))if q.is_file()};write(OUT/'COMPLETE.json',dict(completed=True,PASS=True,code=p['code'],protocol=bind(PROTO),entry=bind(ep),scope=p['scope'],inputs=bindings,selection=p['selection'],outputs=outputs,TRAIN_only=True,known_TRAIN_triplets109=True,TRAINrows2784_verified=True,all_original_qL_native_RGB_valid_hashes_preserved=True,shared_HOLDOUT_rows_physically_unpacked_but_not_analyzed=True,physical_shared_container_rows3144=True,source_packet_or_target_NPZ_bodies_followed=0,weights_models_forward_fit_protected_eval_reads=0,newlabels_or_admission_or_selection=False,automatic_retry=False));print(json.dumps(dict(completed=True,PASS=True,complete=bind(OUT/'COMPLETE.json'))))
 except BaseException:
  write(OUT/'FAILED_NO_AUTOMATIC_RETRY.json',dict(completed=False,PASS=False,traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':
 ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);a=ap.parse_args();prepare()if a.prepare else run(a.entry)
