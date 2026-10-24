"""Independent saved-CSV arithmetic/metadata QA; does not open input NPZ or pixels."""
from pathlib import Path
from collections import defaultdict,Counter
import json,csv,io,hashlib,math,statistics,os,stat
H=Path(__file__).absolute().parent;V=H.parent;P=V.parent;A=V/'paired_pixel_alignment_a';O=A/'actual_train_pixel_alignment_v1';num=0

def need(x,m):
 global num
 if not x:raise AssertionError(m)
 num+=1

def raw(p):
 p=Path(p).absolute();need(P in p.parents and '..'not in p.parts and all('v112'not in s.casefold()for s in p.parts),'Finite name before filesystem');fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for s in p.parts[1:-1]:n=os.open(s,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
  n=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 finally:os.close(fd)
 with os.fdopen(n,'rb')as f:need(stat.S_ISREG(os.fstat(f.fileno()).st_mode),'Regular file no symlink');return f.read()
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(raw(p)).hexdigest())
def bound(b):
 z=raw(b['path']);need(hashlib.sha256(z).hexdigest()==b['sha256'],'Exact bound bytes');return z

def js(p):return json.loads(raw(p))
def parse(b):return list(csv.DictReader(io.StringIO(b.decode())))
def f(x):return None if x is None or x==''else float(x)
def equal(a,b):
 if a in(None,'')or b in(None,''):return a in(None,'')and b in(None,'')
 try:return math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-12)
 except(ValueError,TypeError):return a==b

done=js(O/'COMPLETE.json');need(bind(O/'COMPLETE.json')['sha256']=='3d197c2a328b27a93feebc019e650ab935d2c9407976f1885b5e5eef5efc4df3'and done['completed']and done['PASS'],'Exact completed A actual')
proto=json.loads(bound(done['protocol']));bound(done['code']);readout=js(A/'READOUT_FROM_SAVED_NUMERICS.json');need(readout['actual_complete']==bind(O/'COMPLETE.json'),'Readout exact completion');summary=json.loads(bound(done['outputs']['SUMMARY.json']))
whole=parse(bound(done['outputs']['PAIR_WHOLE96_THRESHOLDS.csv']));windows=parse(bound(done['outputs']['QUERY_CENTER_FIXED_WINDOWS.csv']));aggregated=parse(bound(done['outputs']['QUERY_CENTER_WINDOW_SUMMARY.csv']));native=parse(bound(done['outputs']['TRAIN_ALL_NATIVE_TARGET_STATUS.csv']))
train=[json.loads(t)for t in raw(V/'label_loss_contract_c/TRAIN_ONLY_CONTROL_LABEL_JOIN2784.jsonl').splitlines()if t];need(len(train)==2784 and all(r['role']=='TRAIN'for r in train),'Independent original TRAIN boundary')
known=[r for r in train if r['direct_target']>=0];known3={(r['domain'],r['attempt_id'],r['native_slot'])for r in known};need(len(known)==327 and len(known3)==109,'Independent selected known327/109')
original={tuple(r['control_join_key']):r for r in [json.loads(t)for t in bound(done['inputs']['rows']).splitlines()if t]if tuple(r['control_join_key'])in{tuple(x['control_join_key'])for x in train}}
need(len(original)==2784,'All original TRAIN input metadata exact key set')
def pairkey(r):return r['domain'],r['attempt_id'],int(r['native_slot'])
def key(r):return (*pairkey(r),r['variant'],int(r['absRGB_max_channel_threshold']))
wb={key(r):r for r in whole};xb={(*key(r),float(r['query_window_halfwidth96'])):r for r in windows};need(len(wb)==len(whole)==654 and len(xb)==len(windows)==3270 and len(aggregated)==90 and len(native)==16,'Exact saved row counts/no duplicate keys')
need({pairkey(r)for r in whole}==known3 and {pairkey(r)for r in windows}==known3,'Only all109 original TRAIN selected natives')
halves=[.5,1.5,3.,6.,12.];thresholds=[1,8,32];areas={h:sum(max(abs(x-47.5),abs(y-47.5))<=h for x in range(96)for y in range(96))for h in halves};need(list(areas.values())==[4,16,36,144,576],'Independent fixed integer-center square geometry')
need(proto['measure']['center96']==[47.5,47.5]and proto['measure']['halfwidths96']==halves and proto['measure']['thresholds_max_abs_RGB']==thresholds,'Fixed prospective thresholds/windows')
for row in whole+windows:
 k=key(row);need(k[:3]in known3 and k[3]in('COVER','SHAM')and k[4]in thresholds,'Finite all109 x2x3 key closure');o=original[(*k[:2],k[3],k[2])]
 for x in('authority','component_id','L'):
  value=next(t[x]for t in train if tuple(t['control_join_key'])==(*k[:2],k[3],k[2]))if x=='component_id'else o[x]
  need(equal(row[x],value),'Original authority/component/L join')
 for x in('source_base_index','source_frame_index','source_donor_id'):need(equal(row[x],o[x]),'Original source provenance join')
 for r in [row]:
  window='valid_window_pixels'in r;den=int(r['valid_window_pixels']if window else r['valid_intersection_pixels']);n=int(r['changed_pixels']);mean=f(r['mean_max_abs_channel_over_valid_window']if window else r['mean_max_abs_channel_over_valid']);rgbmean=f(r['mean_abs_RGB_channel_over_valid_window']if window else r['mean_abs_RGB_channel_over_valid']);mx=f(r['max_abs_channel_in_valid_window']if window else r['max_abs_channel_over_valid'])
  need(0<=n<=den<=9216 and equal(r['changed_fraction'],n/den if den else None),'Pixel denominator/fraction conservation')
  if den:need(0<=rgbmean<=mean+1e-12 and mean<=min(mx,3*rgbmean)+1e-12 and mx<=255 and mean+1e-12>=n*k[4]/den,'Saved magnitude/count bounds including unchanged pixels')
  if den:need((mx>=k[4])==(n>0),'Max threshold and changed flag consistency')
  if window:
   h=float(r['query_window_halfwidth96']);need(h in halves and den<=areas[h]and (r['any_changed_pixel']=='True')==(n>0),'Exact fixed-window opportunity and boolean');w=wb[k];need(den<=int(w['valid_intersection_pixels'])and n<=int(w['changed_pixels']),'Window subset of whole96');d=f(w['nearest_changed_pixel_distance_to_q96'])
   if n:need(d is not None and d<=math.sqrt(2)*h+1e-12,'Window changed implies bounded Euclidean nearest')
   if d is not None and d<=h:need(n>0,'Euclidean-near changed pixel lies inside square')
  else:
   d=f(r['nearest_changed_pixel_distance_to_q96']);need((d is None)==(n==0),'No-change nearest null never zero');need(r['valid_masks_equal']=='True','Original pair valid masks equal')
   if d is not None:need(math.sqrt(.5)-1e-12<=d<=math.sqrt(2)*47.5+1e-12 and equal(2*d*d,round(2*d*d)),'Nearest integer pixel-center distance lattice')
for pk in known3:
 for v in('COVER','SHAM'):
  ww=[wb[(*pk,v,t)]for t in thresholds];need([int(r['changed_pixels'])for r in ww]==sorted([int(r['changed_pixels'])for r in ww],reverse=True),'Threshold monotone changed counts')
  need(len({r['valid_intersection_pixels']for r in ww})==1 and len({r['mean_max_abs_channel_over_valid']for r in ww})==1,'Threshold-invariant denominator/magnitude')
  for t in thresholds:
   rr=[xb[(*pk,v,t,h)]for h in halves];need([int(r['changed_pixels'])for r in rr]==sorted(int(r['changed_pixels'])for r in rr),'Nested square changed counts monotone')
for s in aggregated:
 d,v,t,h=s['domain'],s['variant'],int(s['threshold']),float(s['halfwidth96']);rr=[r for r in windows if(d=='ALL_TRAIN'or r['domain']==d)and r['variant']==v and int(r['absRGB_max_channel_threshold'])==t and float(r['query_window_halfwidth96'])==h];den=sum(int(r['valid_window_pixels'])for r in rr);n=sum(int(r['changed_pixels'])for r in rr);anycount=sum(int(r['changed_pixels'])>0 for r in rr)
 ex=dict(analyzed_TRAIN_triplets=len(rr),window_opportunity_triplets=sum(int(r['valid_window_pixels'])>0 for r in rr),triplets_with_changed_query_window=anycount,triplets_without_changed_query_window=len(rr)-anycount,pooled_valid_pixels=den,pooled_changed_pixels=n,pooled_changed_fraction=n/den if den else None,mean_triplet_changed_fraction=statistics.mean(float(r['changed_fraction'])for r in rr if r['changed_fraction']!=''))
 need(all(equal(s[k],x)for k,x in ex.items()),'Independent 90 summary rows from3270 CSV rows')
need(all(all(equal(s[k],j[k])for k in s)for s,j in zip(aggregated,summary['window_summary'])),'CSV90 versus JSON90 summary exact')
groups=defaultdict(list)
for r in train:groups[(r['domain'],r['attempt_id'],r['native_slot'])].append(r)
for s in native:
 d,n=s['domain'],int(s['native_slot']);rr=[r for r in train if r['domain']==d and r['native_slot']==n];ex=dict(TRAIN_rows=len(rr),known_D=sum(r['direct_target']==0for r in rr),known_E=sum(r['direct_target']==1for r in rr),target_UNKNOWN=sum(r['direct_target']<0for r in rr),original_input_U=sum(not r['original_query_available']for r in rr),native_attempt_groups=sum(k[0]==d and k[2]==n for k in groups),partial_UNKNOWN_attempt_groups=sum(k[0]==d and k[2]==n and len(rs)<3 and all(r['direct_target']==-1for r in rs)for k,rs in groups.items()));need(all(equal(s[k],v)for k,v in ex.items()),'Independent native0..7 UNKNOWN/U/partial denominator')
need(sum(int(s['known_D'])for s in native)==218 and sum(int(s['known_E'])for s in native)==109 and sum(int(s['target_UNKNOWN'])for s in native)==2457 and sum(int(s['original_input_U'])for s in native)==112 and sum(int(s['partial_UNKNOWN_attempt_groups'])for s in native)==168,'Original TRAIN whole/native denominator totals')
for d in('SYNTH_FINITE_END','REAL_SEMANTIC_END_WEAK'):
 rr=[r for r in whole if r['domain']==d and r['variant']=='COVER'and int(r['absRGB_max_channel_threshold'])==1];coverage=summary['source_coverage'][d];need(coverage['triplets']==len(rr)and coverage['components']==len({r['component_id']for r in rr})and coverage['bases']==sorted({int(r['source_base_index'])for r in rr if r['source_base_index']!=''})and coverage['frames']==sorted({int(r['source_frame_index'])for r in rr if r['source_frame_index']!=''}),'Independent source frame/base/component coverage')
 distances=[float(r['nearest_changed_pixel_distance_to_q96'])for r in rr if r['nearest_changed_pixel_distance_to_q96']!=''];got=summary['cover_threshold1_nearest_distance96'][d];need(all(equal(got[k],v)for k,v in dict(N=len(distances),min=min(distances),median=statistics.median(distances),max=max(distances)).items()),'Independent nearest min/median/max saved scalar')
for rr in readout['threshold1_query_windows']:
 for h in halves:
  found=[r for r in aggregated if r['domain']==rr['domain']and r['variant']==rr['variant']and int(r['threshold'])==1 and float(r['halfwidth96'])==h];need(len(found)==1 and int(found[0]['triplets_with_changed_query_window'])==rr['h'+str(h)]and int(found[0]['analyzed_TRAIN_triplets'])==rr['N'],'Narrative readout fixed-threshold window counts')
for rr in readout['fixed_threshold1_8_32_centers']:
 s=next(r for r in aggregated if r['domain']==rr['domain']and r['variant']==rr['variant']and int(r['threshold'])==rr['threshold']and float(r['halfwidth96'])==rr['halfwidth96']);need(all(equal(v,s[k])for k,v in rr.items()),'Narrative selected threshold rows exact')
lookup={(r['domain'],r['variant'],int(r['threshold']),float(r['halfwidth96'])):r for r in aggregated};center={v:int(lookup['ALL_TRAIN',v,1,.5]['triplets_with_changed_query_window'])for v in('COVER','SHAM')};need(center==dict(COVER=45,SHAM=6),'Central2x2 COVER45/109 SHAM6/109')
result=dict(completed=True,PASS=True,code=bind(__file__),actual_A_complete=bind(O/'COMPLETE.json'),readout=bind(A/'READOUT_FROM_SAVED_NUMERICS.json'),checks=num,whole_rows654=len(whole),window_rows3270=len(windows),summary_rows90=len(aggregated),allnative16=len(native),known_TRAIN109=len(known3),central2x2_threshold1=center,original_TRAIN_rows2784=True,unknown2457_inputU112_partial168_preserved=True,all_source_weak_authorities_unchanged=True,validation='Independent scalar aggregation/identity/bounds consistency of saved tables and original TRAIN metadata, not raw pixel difference recomputation',original_NPZ_header_target_pixel_weight_model_or_HOLDOUT_score_reads=0)
with(H/'NUMERICAL_QA.json').open('x')as f_:json.dump(result,f_,ensure_ascii=False,indent=2);f_.write('\n')
print(json.dumps(dict(completed=True,PASS=True,qa=bind(H/'NUMERICAL_QA.json'),checks=num),ensure_ascii=False))
