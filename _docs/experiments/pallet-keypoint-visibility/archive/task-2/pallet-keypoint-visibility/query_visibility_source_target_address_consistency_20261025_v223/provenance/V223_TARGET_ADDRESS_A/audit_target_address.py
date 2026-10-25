"""V223 SOURCE JSON-only target-address audit. Never follows image/GT/model pointers."""
import argparse, collections, csv, hashlib, io, json, math, os, stat
from pathlib import Path
import numpy as np

P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
H=P/'publication_root_preparation/V223_TARGET_ADDRESS_A';OUT=H/'actual_audit_v1'
CF=P/'query_visibility_expected_part_counterfactual_source_bridge_20261025_v222/expected_part_counterfactual_pairs_v1'
INPUTS={
 'attempts':{'path':str(CF/'ALL48_ATTEMPTS.json'),'sha256':'eb0820002ab3d520f5eec9b26588ee4759196c6f17c99d17857417be3439f58f'},
 'variants':{'path':str(CF/'ALL144_VARIANT_ROWS.jsonl'),'sha256':'4097805a0e2fe2f7a68b196348284ad3c62b5721d56751d49dd5eb48d161e641'},
 'FIT':{'path':str(CF/'SOURCE_FIT_ROWS_AND_EXACT_PNG_WHITELIST.json'),'sha256':'195c78666c4fb912f90c750a5f3a6dc22edffa1cd3a02fd6d9cda562de5f98fc'},
 'DEV':{'path':str(CF/'SOURCE_DEV_ROWS_AND_EXACT_PNG_WHITELIST.json'),'sha256':'42ebcbd30ea287a627712f376f1b8adf6560f5e9e601d5633888d32a854bb22c'},
 'weak360':{'path':str(P/'query_visibility_public_train_intended_part_query_targets_20261025_v218/joined_weak_intended_part_targets_v1/ALL360_CHECKED_WEAK_TARGET_CANDIDATES.jsonl'),'sha256':'52ac5bb8d0fb19a1329b9bdb7f7b56964fed954a2a4d50489e3dcb7d598d139d'},
}

def need(x,m):
 if not bool(x):raise ValueError(m)
def safe(p):
 s=str(p);need('v112'not in s.casefold(),'V112 rejected before FS');p=Path(s)
 need(p.is_absolute()and'..'not in p.parts and P in p.parents,'Exact project path');return p
def parentfd(p):
 p=safe(p);fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for part in p.parts[1:-1]:n=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
  return p,fd
 except BaseException:os.close(fd);raise
def raw(p):
 p,fd=parentfd(p)
 try:n=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 finally:os.close(fd)
 with os.fdopen(n,'rb')as f:need(stat.S_ISREG(os.fstat(f.fileno()).st_mode),'Regular file');return f.read()
def sha(x):return hashlib.sha256(x).hexdigest()
def enc(v):return(json.dumps(v,ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode()
def save(name,b):
 p,fd=parentfd(OUT/name)
 try:n=os.open(p.name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o644,dir_fd=fd)
 finally:os.close(fd)
 with os.fdopen(n,'wb')as f:f.write(b)
 return {'path':str(p),'sha256':sha(b),'size_bytes':len(b)}
def savej(name,v):return save(name,enc(v))
def mask(rle):
 a=np.zeros(9216,bool);end=0
 for start,length in rle:
  need(type(start)is int and type(length)is int and start>=end and length>0 and start+length<=9216,'Canonical RLE');a[start:start+length]=True;end=start+length
 return a.reshape(96,96)
def rect(box):
 need(len(box)==4 and all(type(x)is int and 0<=x<96 for x in box),'Inclusive rectangle');x0,y0,x1,y1=box;need(x0<=x1 and y0<=y1,'Nonempty rectangle');a=np.zeros((96,96),bool);a[y0:y1+1,x0:x1+1]=True;return a
def center4(a):return int(a[47:49,47:49].sum())
def condition_key(c):return sha(np.asarray(c,dtype='<f4').tobytes())
def declared_input_key(rgb,valid,condition,available):
 return None if not available or not isinstance(rgb,dict)else(rgb['sha256'],sha(mask(valid).tobytes()),condition_key(condition))
def collision_groups(rows):
 groups={}
 for r in rows:
  if r['key']is not None and r['target']in(0,1):groups.setdefault(r['key'],[]).append(r)
 conflicts=[{'recorded_RGB_sha256':k[0],'valid_mask_sha256':k[1],'condition32_sha256':k[2],'labels':sorted({r['target']for r in rr}),'rows':[{x:r[x]for x in('family','row_id','target')}for r in rr]}for k,rr in groups.items()if len({r['target']for r in rr})>1]
 return {'known_rows':sum(r['target']in(0,1)for r in rows),'known_available_rows_with_signature':sum(len(v)for v in groups.values()),'distinct_recorded_input_signatures':len(groups),'duplicate_signature_groups':sum(len(v)>1 for v in groups.values()),'opposite_label_signature_groups':len(conflicts),'conflicts':conflicts,'new_image_byte_or_decoded_identity_verified':False}
def distribution(rows,key):
 a=np.asarray([r[key]for r in rows],np.float64)
 return None if not len(a)else{'min':float(a.min()),'median':float(np.median(a)),'max':float(a.max()),'mean':float(a.mean())}
def parent_stats(rr):
 return {'parents':len(rr),'target_central4_any':sum(r['target_central4_pixels']>0 for r in rr),'target_central4_all':sum(r['target_central4_pixels']==4 for r in rr),'COVER_central4_clear':sum(r['COVER_central4_pixels']==0 for r in rr),'COVER_central4_any':sum(r['COVER_central4_pixels']>0 for r in rr),'SHAM_central4_any':sum(r['SHAM_central4_pixels']>0 for r in rr),'SHAM_central4_all':sum(r['SHAM_central4_pixels']==4 for r in rr),'target_centroid_distance_px':distribution(rr,'centroid_distance_px'),'target_area_pixels':distribution(rr,'target_pixels'),'target_bbox_width':distribution(rr,'bbox_width'),'target_bbox_height':distribution(rr,'bbox_height')}
def condition_groups(rows,with_valid=False):
 groups={}
 for r in rows:groups.setdefault((r['condition32_sha256'],r['valid_mask_sha256'])if with_valid else(r['condition32_sha256'],),[]).append(r)
 result=[]
 for key,rr in groups.items():
  if len(rr)<2:continue
  result.append({'condition32_sha256':key[0],'valid_mask_sha256':key[1]if with_valid else None,'parents':len(rr),'distinct_ROI_masks':len({r['target_mask_sha256']for r in rr}),'distinct_recorded_RGB_sha256':len({r['recorded_RGB_sha256']for r in rr}),'distinct_absolute_queries':len({tuple(r['query_xy'])for r in rr}),'roles':dict(collections.Counter(r['source_role']for r in rr)),'native_slots':sorted({r['native_slot']for r in rr}),'parent_rows':[r['row_id']for r in rr]})
 return {'unique_keys':len(groups),'repeated_groups':len(result),'repeated_groups_with_different_ROI':sum(r['distinct_ROI_masks']>1 for r in result),'groups':result}
def csv_bytes(rows):
 b=io.StringIO();w=csv.DictWriter(b,fieldnames=list(rows[0]));w.writeheader()
 for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False,separators=(',',':'))if isinstance(v,(list,dict))else v for k,v in r.items()})
 return b.getvalue().encode()

def run():
 inputs={}
 for name,b in INPUTS.items():
  need(Path(b['path']).suffix in('.json','.jsonl'),'Only five SOURCE texts');z=raw(b['path']);need(sha(z)==b['sha256'],'Pinned SOURCE text SHA');inputs[name]=[json.loads(x)for x in z.splitlines()if x]if Path(b['path']).suffix=='.jsonl'else json.loads(z)
 attempts=inputs['attempts']['attempts48'];variants=inputs['variants'];weak=inputs['weak360'];need(len(attempts)==48 and len(variants)==144 and len(weak)==360,'48/144/360 retained')
 vmap={r['row_id']:r for r in variants};need(len(vmap)==144,'Unique CF row IDs')
 for role,key in [('SOURCE_FIT','FIT'),('SOURCE_DEV','DEV')]:
  m=inputs[key];rr=[r for r in variants if r['source_role']==role];need(m['source_role']==role and m['CF_variant_rows']==[r for r in rr if r['constructed']]and m['all_parent_BEFORE_rows']==[r for r in rr if r['variant']=='BEFORE'],'Role manifest closure')
 parent_rows=[];variant_rows=[];records=[];role_by_component={}
 for a in attempts:
  r=a['parent_metadata'];pid=r['row_id'];target=mask(r['exact_union_RLE_rowmajor96']);valid=mask(r['valid96_RLE']);union=np.zeros_like(target)
  for box in r['ROI96_by_observer'].values():union|=rect(box)
  need(np.array_equal(union,target)and target.any()and np.all(valid[target]),'Exact OR/valid');ca=rect(a['geometry']['COVER_rectangle']);sa=rect(a['geometry']['SHAM_rectangle']);need(ca[target].all()and not(sa&target).any()and int(ca.sum())==int(sa.sum()),'Declared geometry proof')
  need(role_by_component.setdefault(r['component_id'],r['source_role'])==r['source_role'],'Immutable component role')
  q=np.asarray(r['q8_xy'],np.float64);av=np.asarray(r['q8_available'],bool);k=r['native_slot'];need(q.shape==(8,2)and r['generated_query_xy']==r['q8_xy'][k]and av[k],'Active original named-family query')
  mm=av.copy();mm[k]=False;xy=np.zeros((8,2),np.float32);xy[mm]=np.clip((q[mm]-q[k])/r['L'],-8,8).astype(np.float32);one=np.zeros(8,np.float32);one[k]=1;cc=np.concatenate([one,mm.astype(np.float32),xy.ravel()]);need(np.array_equal(cc,np.asarray(r['condition32'],np.float32)),'Fullq8 F64 clipped FP32 condition exact')
  yy,xx=np.nonzero(target);cx=float(xx.mean());cy=float(yy.mean());dx=cx-47.5;dy=cy-47.5;raw_other=47.5+96*(q[mm]-q[k])/r['L'];other_ids=np.flatnonzero(mm)
  nearest=None if not len(other_ids)else int(other_ids[np.argmin(np.sum((raw_other-[cx,cy])**2,axis=1))])
  row={'row_id':pid,'nominal_row_id':r['nominal_row_id'],'source_role':r['source_role'],'component_id':r['component_id'],'frame_position':r['frame_position'],'candidate_gt_list_index':r['candidate_gt_list_index'],'native_slot':k,'anatomical_family':r['anatomical_family'],'normalized_offset':r['normalized_offset'],'query_xy':r['generated_query_xy'],'L':r['L'],'q8_origin':r['q8_origin'],'target_pixels':int(target.sum()),'target_bbox':[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())],'bbox_width':int(xx.max()-xx.min()+1),'bbox_height':int(yy.max()-yy.min()+1),'centroid_raw96_xy':[cx,cy],'centroid_delta_from_querycenter_px':[dx,dy],'centroid_delta_div96':[dx/96,dy/96],'centroid_distance_px':float(math.hypot(dx,dy)),'target_central4_pixels':center4(target),'COVER_central4_pixels':center4(ca),'SHAM_central4_pixels':center4(sa),'other_available_count':int(mm.sum()),'nearest_other_query_to_ROI_centroid_native':nearest,'condition32':cc.tolist(),'condition_other_mask8':mm.tolist(),'condition_other_dxdy16':xy.ravel().tolist(),'available_other_query_raw96_xy':raw_other.tolist(),'condition32_sha256':condition_key(cc),'valid_mask_sha256':sha(valid.tobytes()),'target_mask_sha256':sha(target.tobytes()),'recorded_RGB_sha256':r['RGB96_binding']['sha256'],'original_detector_to_GT_instance':r['original_detector_to_GT_instance'],'native_point_gold':False}
  parent_rows.append(row)
  for variant in('BEFORE','COVER','SHAM'):
   v=vmap[pid+'__'+variant];need(v['condition32']==r['condition32']and v['valid96_RLE']==r['valid96_RLE']and v['native_point_DE_target']==-1,'Same triplet conditioning/native-1')
   alpha=np.zeros_like(ca)if variant=='BEFORE'else ca if variant=='COVER'else sa
   variant_rows.append({'row_id':v['row_id'],'parent_row_id':pid,'source_role':v['source_role'],'native_slot':k,'variant':variant,'weak_target_D0_E1':v['weak_finite_part_target_D0_E1'],'native_point_DE_target':-1,'input_available':v['input_available'],'declared_alpha_central4_pixels':center4(alpha),'declared_alpha_central4_clear':center4(alpha)==0,'target_centroid_distance_px':row['centroid_distance_px'],'condition32_sha256':row['condition32_sha256'],'recorded_RGB_sha256':None if v['RGB96']is None else v['RGB96']['sha256']})
   records.append({'family':'CF','row_id':v['row_id'],'target':v['weak_finite_part_target_D0_E1'],'key':declared_input_key(v['RGB96'],v['valid96_RLE'],v['condition32'],v['input_available'])})
 for r in weak:
  g=r['generated_input'];t=r['weak_target_E_binary']if r['weak_loss_candidate']is True else -1
  need(t in(-1,0,1),'Unknown not negative/E');records.append({'family':'NATURAL_GT_QUERY_WEAK','row_id':r['row_id'],'target':t,'key':declared_input_key(g['RGB96'],g['valid96_RLE'],r['deployable_condition32'],g['generated_query_available'])})
 need(collections.Counter(r['source_role']for r in parent_rows)=={'SOURCE_FIT':39,'SOURCE_DEV':9},'Original48 split')
 groups={'condition32_only':condition_groups(parent_rows),'condition32_and_valid':condition_groups(parent_rows,True),'declared_input_label_collisions':collision_groups(records)}
 summary={'completed':True,'PASS':True,'source_only':True,'parent_rows48':48,'variant_rows144':144,'natural_family_rows360':360,'canonical_weak_known57_unknown303':dict(collections.Counter(r['target']for r in records if r['family']=='NATURAL_GT_QUERY_WEAK')),'parents_all':parent_stats(parent_rows),'by_role':{role:parent_stats([r for r in parent_rows if r['source_role']==role])for role in('SOURCE_FIT','SOURCE_DEV')},'by_native':{str(k):parent_stats([r for r in parent_rows if r['native_slot']==k])for k in range(8)},'by_role_native':{role:{str(k):parent_stats([r for r in parent_rows if r['native_slot']==k])for k in range(8)}for role in('SOURCE_FIT','SOURCE_DEV')},'components_by_role':dict(collections.Counter(role_by_component.values())),'repeated_condition_groups_different_ROI':groups['condition32_only']['repeated_groups_with_different_ROI'],'repeated_condition_valid_groups_different_ROI':groups['condition32_and_valid']['repeated_groups_with_different_ROI'],'declared_opposite_label_input_signature_groups':groups['declared_input_label_collisions']['opposite_label_signature_groups'],'center_definition':'RAW96 query=(47.5,47.5); central4 means pixel centers [47,48]x[47,48]. CLEAR=zero declared alpha in all4; this is geometry, not new RGB decode or query/native visibility.','addressability_KO':{'metadata_determined':'각 saved row에서 GT후보/native/q8/condition32와 관찰 finite ROI OR가 정확히 join된다. ROI의 centroid/bbox/central4 관계는 대수적으로 정해진다.','not_established':'ROI는 head feature가 아니라 weak bag 감독이다. 같은 condition에 다른 ROI가 있어도 RGB가 다르면 전체 model input 정보부족/라벨모순을 증명하지 않는다. centroid는 material point/GT q-error가 아니다.','physical_semantic_limit':'q8는 GT-derived training family이며 원 detector→GT instance는 UNKNOWN이다. 두 reader ROI와 named expectedpart linkage는 weak authority이고 exact physical native-point gold가 아니다.','collision_limit':'JSON에 기록된 encoded PNG SHA+valid+condition 동일성 후보를 검사했다. 이번에 PNG byte/decoded equality를 새로 확인하지 않았으며 exact duplicate 충돌이 없으면 입력 불충분을 단정하지 않는다.'},'new_image_NPZ_PT_GT_model_protected_or_training_reads':0}
 p,fd=parentfd(OUT)
 try:os.mkdir(p.name,dir_fd=fd)
 finally:os.close(fd)
 artifacts={'parents48':save('PARENTS48_TARGET_ADDRESS.csv',csv_bytes(parent_rows)),'variants144':save('VARIANTS144_CENTER_AND_CONDITION.csv',csv_bytes(variant_rows)),'groups':savej('COLLISION_AND_CONDITION_GROUPS.json',groups),'summary':savej('SUMMARY.json',summary)}
 lines=['# SOURCE finite-part 타깃 주소 진단','','허용된 SOURCE JSON 5개만 읽었으며 새 PNG·모델·GT·보호 데이터·학습 접근은 0이다. ROI는 학습입력이 아니라 weak bag/region 타깃이다.','', '| 범위 | 부모 | target central4 접촉 | COVER central4 clear | SHAM central4 접촉 | centroid거리 min/median/max(px) |','|---|---:|---:|---:|---:|---|']
 for name,rr in [('전체',parent_rows)]+[(role,[r for r in parent_rows if r['source_role']==role])for role in('SOURCE_FIT','SOURCE_DEV')]:
  st=parent_stats(rr);d=st['target_centroid_distance_px'];lines.append(f"| {name} | {st['parents']} | {st['target_central4_any']} | {st['COVER_central4_clear']} | {st['SHAM_central4_any']} | {d['min']:.3f}/{d['median']:.3f}/{d['max']:.3f} |")
 lines+=['','중심은 원 RAW96의 (47.5,47.5), central4는 index47/48의 네 pixel이다. CLEAR는 선언된 alpha support가 네 pixel을 모두 비운다는 뜻이며 실제 RGB·query support·native D/E를 새 판정한 것이 아니다.', '',f"48부모를 먼저 접은 뒤 동일 condition32에서 다른 ROI mask가 있는 group은 {summary['repeated_condition_groups_different_ROI']}개이고, valid까지 같은 group은 {summary['repeated_condition_valid_groups_different_ROI']}개다. 같은 conditioning의 D/E/D triplet은 의도된 RGB 개입이다. 다른 ROI/라벨만으로 전체 입력이 동일하다고 간주하지 않는다.",'',f"기록된 PNG SHA+valid+condition을 함께 묶은 알려진 D/E 라벨 충돌 group은 {summary['declared_opposite_label_input_signature_groups']}개다. 이것은 SOURCE JSON의 선언 identity 검사이며 새 PNG byte/decoded duplicate proof가 아니다. 동일 condition 아래 다른 ROI는 witness 주소의 변화를 보여주지만 RGB가 부위를 주소화할 가능성은 남는다.",'','각 row→GT candidate/native/q8→finite ROI의 metadata 연결과 위치는 정해져 있다. 그러나 ROI는 배경도 포함하는 weak bag이며 centroid는 물리 부위점/정답 q나 좌표오차가 아니다. 원 detector→GT instance는 UNKNOWN이고 부모 query는 GT-derived training family다. 원 초기 query나 배포에서 native exact point가 유일하게 주소화된다는 결론은 이 자료만으로 확정할 수 없다.','','COVER가 중심을 비워도 remote finite target을 가렸을 수 있고, SHAM이 중심을 덮어도 target ROI는 보존될 수 있다. 이를 absence=E 또는 query/native label로 복사하지 않는다. 입력 정보부족·잘못된 annotation·모델 원인·실사 개선은 이번 표에서 증명하지 않는다.','','[48부모/native/role/centroid/conditioning 표](PARENTS48_TARGET_ADDRESS.csv) · [144variant 중심 표](VARIANTS144_CENTER_AND_CONDITION.csv) · [condition/충돌 원장](COLLISION_AND_CONDITION_GROUPS.json) · [전체 role×native 요약](SUMMARY.json)','']
 artifacts['report']=save('KO.md','\n'.join(lines).encode())
 for b in INPUTS.values():need(sha(raw(b['path']))==b['sha256'],'Final SOURCE text hashes unchanged')
 code=raw(Path(__file__).absolute());complete=savej('COMPLETE.json',{'completed':True,'PASS':True,'frozen_no_further_edits':True,'authority':'Root authorized V223 bounded JSON-only audit prepare/execute; five exact SOURCE text files only.','code':{'path':str(Path(__file__).absolute()),'sha256':sha(code)},'inputs':INPUTS,'artifacts':artifacts,'source_text_files_read':5,'pointer_follows':0,'PNG_NPZ_PT_GT_payload_model_protected_or_training_access':0,'actual_runs':1})
 print(json.dumps({'completed':True,'PASS':True,'COMPLETE':complete,'summary':artifacts['summary'],'parents_all':summary['parents_all'],'condition_different_ROI_groups':summary['repeated_condition_groups_different_ROI'],'condition_valid_different_ROI_groups':summary['repeated_condition_valid_groups_different_ROI'],'opposite_label_signature_groups':summary['declared_opposite_label_input_signature_groups']},ensure_ascii=False))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--run-approved',action='store_true');args=a.parse_args();need(args.run_approved,'Explicit root-approved JSON audit flag');run()
