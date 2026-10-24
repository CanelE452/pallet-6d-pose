"""V201 SOURCE TRAIN109 finite-end/query correspondence; stdlib metadata only."""
from pathlib import Path
from collections import Counter,defaultdict
import ast,csv,hashlib,io,json,math,os,stat,statistics
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility');H=Path(__file__).absolute().parent
V197=P/'query_visibility_train_only_target_contract_readout_20261024_v197';F=V197/'fixed_train_target_contract_viewer_v1'
INPUTS={
'scope':(H.parent/'ROOT_TRAIN_ONLY_CORRESPONDENCE_SCOPE.json','4db9e8d14c7d0af2641a903191dbe9ef1779364c38a003d4fd7f61bb7480bc38'),
'complete':(F/'COMPLETE.json','4a9efcd72b20b13f7814396d5f02535af90f5a009fea5791aa989e124791a316'),
'all109':(F/'ALL109_TRAIN_TRIPLET_CONTRACTS_METADATA_BEFORE_PIXELS.json','9e1ca2fd482a38aa596421edc7aba99a6e1051f7736ba128f5509ab894802bde'),
'mapping':(F/'SELECTED_TRAIN_ALL8_TARGET_MAPPING.csv','b62ea42f3b222a282bed6f58a7cd82247f2e1d0c9d4a2a76c8217c49ccafdcc9'),
'selection':(F/'SELECTION_BEFORE_ANY_TARGET_NPZ_HEADER_OR_PIXEL_READ.json','29b1ca6e4981d650076493ed39a9943d467f5e356f4ed5af77b3f87929113e34'),
'image_manifest':(F/'IMAGE_MANIFEST.json','5e03e28f4f4631edbac22807693340d6e0256bb11276d9c0bd4825410f7bd3e5'),
'viewer_code':(V197/'viewer_preparation_a_v2/build_fixed_train_target_contract_viewer.py','86543bb80cb50c7675834f1c309b4fbfb39b83791b867143b6da8f2872ca903f'),
'synthetic_candidate_code':(P/'query_visibility_sampled_domain_direct_end_20261024_v189/direct_end_training_preparation_c/synthetic_direct_end_candidate.py','1668550225434f609ed70d2d62093e35bea6bd8471b9c1f5692f7cfb0b358341'),
'synthetic_producer_code':(P/'query_visibility_sampled_domain_direct_end_20261024_v189/direct_end_training_preparation_c/prepare_and_generate_synthetic_sampled_end.py','8ccdda9e839938c95970d04bb39da1bf0554d39d7723e9c15a03cb3c4a603e81'),
'constructive_code':(P/'query_visibility_spatial_stencil_control_20261023_v127/registered_native_teacher_feasibility/build_registered_pallet.py','1ac00f53f037ceeebe3d67d6f76f69cbbdf3781461c514ae6b483256f3e7c01d'),
'real_producer_code':(P/'query_visibility_source_end_pairs_opposite_sham_20261024_v193/real_end_opposite_sham_preparation_c/build_real_end_opposite_sham_pairs.py','7375e34f3710806d56584b8261c96dc551ba0459144c8e7b9256ab984dcc00e5')}
def need(v,m):
 if not v:raise AssertionError(m)
def safe(p):
 p=Path(p);need(p.is_absolute()and'..'not in p.parts and all('v112'not in s.casefold()for s in p.parts),'Forbidden name before filesystem access');return p
def raw(p):
 p=safe(p);fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for s in p.parts[1:-1]:n=os.open(s,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
  n=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 finally:os.close(fd)
 need(stat.S_ISREG(os.fstat(n).st_mode),'Regular only')
 with os.fdopen(n,'rb')as f:return f.read()
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(raw(p)).hexdigest())
def read(key):
 p,h=INPUTS[key];z=raw(p);need(hashlib.sha256(z).hexdigest()==h,'Fixed binding '+key);return z
def write(name,obj):
 p=H/name
 with p.open('x')as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 return bind(p)
def csvwrite(name,rows):
 with(H/name).open('x',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 return bind(H/name)
def geom(q,L,boxes):
 need(len(q)==2 and all(math.isfinite(float(x))for x in q)and math.isfinite(L)and L>0,'Original finite F64 q/L')
 need(bool(boxes),'All candidate boxes retained')
 out=[]
 for bb in boxes:
  need(len(bb)==4 and all(math.isfinite(x)for x in bb)and bb[0]<=bb[2]and bb[1]<=bb[3],'Closed finite candidate enclosure')
  x0,y0,x1,y1=bb;crop=[96*(x0-q[0])/L+47.5,96*(y0-q[1])/L+47.5,96*(x1-q[0])/L+47.5,96*(y1-q[1])/L+47.5]
  dx=(x0+x1)/2-q[0];dy=(y0+y1)/2-q[1]
  rx=max(x0-q[0],0,q[0]-x1);ry=max(y0-q[1],0,q[1]-y1)
  out.append(dict(box_crop96=crop,inside=x0<=q[0]<=x1 and y0<=q[1]<=y1,region_distance_crop_px=math.hypot(rx,ry)*96/L,center_distance_crop_px=math.hypot(dx,dy)*96/L,center_dx_crop_px=dx*96/L,center_dy_crop_px=dy*96/L,width_crop_px=(x1-x0)*96/L,height_crop_px=(y1-y0)*96/L,inside_sample_coordinate_extent=all(0<=x<=95 for x in crop)))
 return out
def stats(v):return dict(N=len(v),min=min(v),median=statistics.median(v),max=max(v))if v else dict(N=0,min=None,median=None,max=None)
def fixtures():
 tests=[]
 def add(n,v):tests.append(dict(name=n,PASS=bool(v)));need(v,n)
 g=geom([100.,200.],96.,[[99.,199.,101.,201.]])[0];add('qcenter47p5_inside_closed_box_distance0',g['box_crop96']==[46.5,46.5,48.5,48.5]and g['inside']and g['region_distance_crop_px']==0)
 g=geom([0.,0.],192.,[[6.,8.,10.,12.]])[0];add('region_distance5_not_center_distance',g['region_distance_crop_px']==5 and g['center_distance_crop_px']>5 and not g['inside'])
 g=geom([0.,0.],96.,[[0.,0.,0.,0.]])[0];add('degenerate_closed_point',g['inside']and g['width_crop_px']==0)
 g=geom([0.,0.],96.,[[-2.,-1.,1.,2.],[10.,0.,12.,1.]]);add('all_alternatives_preserved',len(g)==2 and g[0]['inside']and not g[1]['inside'])
 g=geom([0.,0.],96.,[[48.,0.,49.,1.]])[0];add('outside_sample_extent_separate',not g['inside_sample_coordinate_extent'])
 for n,q,L in [('missing_q',[None,None],96.),('zero_L',[0.,0.],0.)]:
  try:geom(q,L,[[0.,0.,1.,1.]]);ok=False
  except(TypeError,AssertionError):ok=True
  add(n+'_not_imputed',ok)
 return dict(completed=True,PASS=True,invented_only=True,checks=tests)
def run():
 tests=fixtures();write('GENERATED_INVENTED_GEOMETRY_CHECKS.json',tests)
 z=json.loads(read('all109'));rows=z['triplets109'];complete=json.loads(read('complete'));selection=json.loads(read('selection'));manifest=json.loads(read('image_manifest'));mapping=list(csv.DictReader(io.StringIO(read('mapping').decode())))
 need(len(rows)==109 and len(mapping)==288 and complete['all327_TRAIN_known_labels_preserved']is True,'Closed TRAIN109 and selected288')
 index={(x['domain'],x['attempt_id']):x for x in rows};need(len(index)==109,'Unique triplet keys');summaryrows=[];by=defaultdict(list);canon=Counter();groups=Counter()
 for x in rows:
  a=x['association'];need(a['selected_native_slot']==x['selected_native_slot']and a['all_other_native_targets_UNKNOWN']is True and a['association_only_not_nativephysicalgold']is True,'Native/authority contract')
  need(x['selected_query_available']is True and a['proof_status']=='CONTROLLED_DIRECT_END_WEAK_PAIR_SUPPORTED','Known declared controls only')
  q=a['original_selected_query_xy'];L=a['original_selected_L'];gg=geom(q,L,a['candidate_enclosures_fullRGB']);stored=a['candidate_enclosures_original96_diagnostic'];need(len(gg)==len(stored),'All candidate projections retained')
  need(max(abs(v-w)for g,bb in zip(gg,stored)for v,w in zip(g['box_crop96'],bb))<1e-11,'Independent original96 formula agrees')
  group='BASE%03d'%a['source_base_index']if a['source_base_index']is not None else 'FRAME%04d'%a['source_frame_index'];groups[(x['domain'],group)]+=1;canon[(x['domain'],x['selected_native_slot'],str(a['canonical_index_metadata']))]+=1
  r=dict(domain=x['domain'],attempt_id=x['attempt_id'],source_group=group,source_base_index=a['source_base_index'],source_frame_index=a['source_frame_index'],source_case_id=a['source_case_id'],source_donor_id=a['source_donor_id'],component_id=x['component_id'],selected_native_slot=x['selected_native_slot'],canonical_index_metadata=a['canonical_index_metadata'],canonical_authority=a['canonical_authority'],weak_authority=a['weak_authority'],end_locality_id=a['end_locality_id'],original_q_x=q[0],original_q_y=q[1],original_L=L,candidate_count=len(gg),candidate_boxes_fullRGB_json=json.dumps(a['candidate_enclosures_fullRGB']),candidate_boxes_crop96_json=json.dumps([g['box_crop96']for g in gg]),query_crop_x=47.5,query_crop_y=47.5,query_inside_any_candidate=any(g['inside']for g in gg),min_query_distance_to_candidate_region_crop_px=min(g['region_distance_crop_px']for g in gg),min_candidate_center_distance_crop_px=min(g['center_distance_crop_px']for g in gg),max_candidate_center_distance_crop_px=max(g['center_distance_crop_px']for g in gg),candidate_center_dxy_crop_px_json=json.dumps([[g['center_dx_crop_px'],g['center_dy_crop_px']]for g in gg]),candidate_width_height_crop_px_json=json.dumps([[g['width_crop_px'],g['height_crop_px']]for g in gg]),all_candidates_in_original_sample_coordinate_extent=all(g['inside_sample_coordinate_extent']for g in gg),stored_proof_status=a['proof_status'],declared_target_contract='selected slot BEFORE0/COVER1/SHAM0;other7 UNKNOWN',physical_native_certified=False,training_admission=False)
  summaryrows.append(r);by[x['domain']].append(r)
 # Selected all8 table validates same q/L/native/U and D/E/D without opening any target container.
 mgrp=defaultdict(list)
 for r in mapping:need(r['role']=='TRAIN','Only selected TRAIN rows');mgrp[(r['domain'],r['attempt_id'])].append(r)
 selected_status=[]
 for key,rr in mgrp.items():
  need(key in index and len(rr)==24,'Selected all3/all8');slot=index[key]['selected_native_slot'];need({(r['variant'],int(r['native_slot']))for r in rr}=={(v,n)for v in('BEFORE','COVER','SHAM')for n in range(8)},'Exact all8 cube')
  for n in range(8):
   r8=[r for r in rr if int(r['native_slot'])==n];need(len({(r['original_query_xy'],r['original_L'],r['original_query_available'],r['original_global_index'])for r in r8})==1,'Original qL/global/U unchanged across variants')
  for r in rr:
   target=int(r['direct_weak_target']);need(target==({'BEFORE':0,'COVER':1,'SHAM':0}[r['variant']]if int(r['native_slot'])==slot else -1),'Only selected native known')
   if r['original_query_available']=='False':need(target==-1,'OriginalU never weak known')
 selected_keys={(x['domain'],x['attempt_id'])for x in selection['selected']};need(selected_keys==set(mgrp),'Original fixed12 unchanged')
 domain_summary={}
 for d,rr in by.items():
  originals=[index[(r['domain'],r['attempt_id'])]['association']for r in rr];domain_summary[d]=dict(triplets=len(rr),declared_known_weak_labels=3*len(rr),all8_variant_rows=24*len(rr),other7_variant_targets_UNKNOWN=21*len(rr),native_known_triplets={str(n):sum(r['selected_native_slot']==n for r in rr)for n in range(8)},unique_source_groups=len({r['source_group']for r in rr}),unique_source_components=len({r['component_id']for r in rr}),unique_cases=len({r['source_case_id']for r in rr if r['source_case_id']is not None}),query_inside_any_candidate=sum(r['query_inside_any_candidate']for r in rr),query_outside_all_candidates=sum(not r['query_inside_any_candidate']for r in rr),region_distance_crop_px=stats([r['min_query_distance_to_candidate_region_crop_px']for r in rr]),center_distance_crop_px=stats([r['min_candidate_center_distance_crop_px']for r in rr]),candidate_width_crop_px=stats([v[0]for r in rr for v in json.loads(r['candidate_width_height_crop_px_json'])]),candidate_height_crop_px=stats([v[1]for r in rr for v in json.loads(r['candidate_width_height_crop_px_json'])]),canonical_unknown=sum(r['canonical_index_metadata']is None for r in rr))
 n67={str(n):dict(selected_native_known_in109=sum(x['selected_native_slot']==n for x in rows),selected12_all8_rows=sum(int(r['native_slot'])==n for r in mapping),selected12_all8_inputU_rows=sum(int(r['native_slot'])==n and r['original_query_available']=='False'for r in mapping),selected12_all8_direct_targets_UNKNOWN=sum(int(r['native_slot'])==n and int(r['direct_weak_target'])==-1 for r in mapping),original_semantic_SELF_D_E_OOF_reason='UNKNOWN_IN_ALLOWED_TRAIN109_AND_SELECTED_MAPPING; other-slot direct UNKNOWN is by design; do not infer absence of relevant D/E coverage')for n in(6,7)}
 outcsv=csvwrite('ALL109_TRAIN_FINITE_END_QUERY_CORRESPONDENCE.csv',summaryrows);groupcsv=csvwrite('SOURCE_GROUP_COUNTS.csv',[dict(domain=d,source_group=g,triplets=n)for(d,g),n in sorted(groups.items())]);cancsv=csvwrite('NATIVE_CANONICAL_COUNTS.csv',[dict(domain=d,native_slot=n,canonical_index=c,triplets=k)for(d,n,c),k in sorted(canon.items())])
 # Freeze a tiny deterministic re-view subset of the original fixed selection before any PNG read.
 first=[]
 for item in selection['selected']:
  if item['domain']not in {x['domain']for x in first}:first.append(item)
 views=[]
 for x in first:
  contact=next(y for y in manifest['contacts']if y['attempt_id']==x['attempt_id']);ctx=next(y for y in manifest['fullcontexts']if y['attempt_id']==x['attempt_id']and y['variant']=='BEFORE');need(ctx['available']is True,'Only existing exact context/no fallback')
  views.extend([dict(domain=x['domain'],attempt_id=x['attempt_id'],kind='CONTACT',binding=contact['contact']),dict(domain=x['domain'],attempt_id=x['attempt_id'],kind='BEFORE_CONTEXT_DISPLAY',binding=ctx['display'])])
 vp=write('FIXED_PREVIOUS_TRAIN_VIEW_PLAN_BEFORE_PNG_READS.json',dict(completed=True,original_selection=bind(INPUTS['selection'][0]),rule='first already-selected triplet per domain in unchanged V197 selection order; no score/error or new training selection',images=views,original_training_selection_unchanged=True,new_view_claim='PENDING_ACTUAL_TOOLS_VIEW'))
 evidence=[]
 for key in ('viewer_code','synthetic_candidate_code','synthetic_producer_code','constructive_code','real_producer_code'):
  s=read(key).decode();nodes=[dict(name=n.name,line=n.lineno,end_line=n.end_lineno)for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)];evidence.append(dict(role=key,binding=bind(INPUTS[key][0]),functions=nodes,execution=False))
 summary=dict(completed=True,PASS=True,scope=bind(INPUTS['scope'][0]),inputs={k:bind(v[0])for k,v in INPUTS.items()},domain_summary=domain_summary,total_triplets=109,declared_known_weak_labels=327,all8_variant_rows=2616,other7_variant_UNKNOWN_rows=2289,selected_mapping_rows=len(mapping),selected_mapping_known=sum(int(x['direct_weak_target'])>=0 for x in mapping),selected_mapping_inputU=sum(x['original_query_available']=='False'for x in mapping),selected_native_counts={str(n):sum(x['selected_native_slot']==n for x in rows)for n in range(8)},native6_7_caveat=n67,all_candidate_projection_parity=True,selected_original_qL_native_U_3variant_parity=True,source_code_evidence=evidence,outputs=dict(per109=outcsv,source_groups=groupcsv,native_canonical=cancsv,view_plan=vp),distance_authority='Closed weak candidate-region distance in original96 coordinates; enclosure center is diagnostic only, not physical point coordinate error. Any candidate containment counted, no candidate shrinking or remapping.',actual_source_NPZ_RGB_weight_model_score_HOLDOUT_target_reads=0,new_labels_or_training_admission=False,selected_model=None,fully_solved=False)
 write('SUMMARY.json',summary);print(json.dumps(dict(completed=True,PASS=True,summary=bind(H/'SUMMARY.json'),per109=outcsv,domain_summary=domain_summary,native_counts=summary['selected_native_counts']),ensure_ascii=False))
if __name__=='__main__':run()
