"""Independent V180 saved audit: own four-tap scalar sampler + whole-pixel counts.
No evaluator/helper import, torch/model/weights, private/held data or 96grid resampling.
"""
import argparse,hashlib,json,math,sys,time
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import numpy as np
V=Path(__file__).resolve().parents[1];D=Path(__file__).resolve().parent
E=V/'saved_sam_allTRAIN4096_evaluation_c_v3';MAN=V/'ALL4096_TRAIN_INPUTS_METADATA_ONLY.jsonl'
CP=E/'COMPLETE.json';CP_SHA='4b6f778cf54304f59b082f486e4e955df796bb8fbbf0adca2502d07e0175378f'
TRUTH=['TARGET','FOREIGN','BG','AMBIGUOUS','INVALID_SUPPORT'];PRED=['TARGET','NON_TARGET','U']

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def checked(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def put(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def finite(q):return q is not None and len(q)==2 and all(x is not None and math.isfinite(x) for x in q) and q!=[-1.,-1.]
def category(part,amb,x,y):
 if x<0 or y<0 or y>=part.shape[0] or x>=part.shape[1]:return 4
 if amb[y,x]:return 3
 p=int(part[y,x]);return 2 if p<0 else (1 if p==999 else 0)
def four_taps(q):
 x,y=q;ix=math.floor(x);iy=math.floor(y);dx=x-ix;dy=y-iy
 return [(ix,iy,(1-dx)*(1-dy)),(ix+1,iy,dx*(1-dy)),(ix,iy+1,(1-dx)*dy),(ix+1,iy+1,dx*dy)]
def sample_point(part,amb,logits,q,available):
 mass=[[] for _ in range(5)];terms=[];support=0.
 for x,y,w in four_taps(q):
  c=category(part,amb,x,y);mass[c].append(w)
  if 0<=x<512 and 0<=y<512:
   support+=w
   if logits is not None:terms.append(float(logits[y,x])*w)
 weights=[math.fsum(a) for a in mass];nx=math.floor(q[0]+.5);ny=math.floor(q[1]+.5);nclass=category(part,amb,nx,ny)
 near_good=0<=nx<512 and 0<=ny<512;bil_good=-.5<=q[0]<511.5 and -.5<=q[1]<511.5
 nearval=float(logits[ny,nx]) if available and near_good else None
 bilval=sum(terms) if available and bil_good else None
 return weights,nclass,(2 if nearval is None else (0 if nearval>0 else 1)),(2 if bilval is None else (0 if bilval>0 else 1)),nearval,bilval,support

def region(logits,part,amb,available):
 # Independent Boolean partition and flat bincount, ambiguity removed before classification.
 valid=np.logical_not(amb);target=np.logical_and(part>=0,part!=999);foreign=part==999;bg=part<0
 v=valid.ravel();t=target.ravel();counts={'pixels':int(part.size),'ambiguity_excluded':int(np.count_nonzero(amb)),'valid_pixels':int(np.count_nonzero(v)),'target_pixels':int(np.count_nonzero(v&t)),'foreign_pixels':int(np.count_nonzero(valid&foreign)),'BG_pixels':int(np.count_nonzero(valid&bg))}
 if not available:return dict(**counts,prediction_U_pixels=counts['valid_pixels'],TP=None,FP=None,FN=None,TN=None,IoU=None,Dice=None)
 pred=(logits>0).ravel();codes=t[v].astype(np.int8)*2+pred[v].astype(np.int8);table=np.bincount(codes,minlength=4);tn,fp,fn,tp=map(int,table)
 ratio=lambda a,b:None if not b else a/b
 return dict(**counts,prediction_U_pixels=0,TP=tp,FP=fp,FN=fn,TN=tn,FP_on_FOREIGN=int(np.count_nonzero(valid&foreign&(logits>0))),FP_on_BG=int(np.count_nonzero(valid&bg&(logits>0))),IoU=ratio(tp,tp+fp+fn),Dice=ratio(2*tp,2*tp+fp+fn),TARGET_precision=ratio(tp,tp+fp),TARGET_recall=ratio(tp,tp+fn))

comparisons=0;max_float_delta=0.
def same(a,b,path='',tol=2e-10):
 global comparisons,max_float_delta
 if isinstance(a,dict):
  assert isinstance(b,dict) and set(a)==set(b),(path,'keys',set(a)^set(b))
  for k in a:same(a[k],b[k],path+'/'+str(k),tol)
 elif isinstance(a,(list,tuple)):
  assert isinstance(b,list) and len(a)==len(b),(path,'length')
  for i,(x,y) in enumerate(zip(a,b)):same(x,y,path+'/'+str(i),tol)
 elif a is None:assert b is None,(path,a,b);comparisons+=1
 elif isinstance(a,(int,bool,str)):assert a==b,(path,a,b);comparisons+=1
 else:
  assert isinstance(b,(int,float)) and math.isfinite(a) and math.isfinite(b),(path,a,b)
  delta=abs(a-b);max_float_delta=max(max_float_delta,delta);assert delta<=tol,(path,a,b,delta);comparisons+=1

def describe(cm):
 t=cm[0];n=[cm[1][j]+cm[2][j] for j in range(3)];tot=math.fsum(math.fsum(x) for x in cm);tp=t[0];fp=n[0];tn=n[1];ratio=lambda a,b:None if not b else a/b
 return {'truth_order':TRUTH,'prediction_order':PRED,'soft_confusion':cm,'sample_equivalent_mass':tot,'TARGET_mass':math.fsum(t),'NON_TARGET_known_mass':math.fsum(n),'AMBIGUOUS_mass':math.fsum(cm[3]),'INVALID_SUPPORT_mass':math.fsum(cm[4]),'prediction_U_mass':math.fsum(x[2] for x in cm),'known_truth_prediction_U_mass':t[2]+n[2],'TARGET_precision_known_truth':ratio(tp,tp+fp),'TARGET_recall_including_prediction_U':ratio(tp,math.fsum(t)),'NON_TARGET_recall_including_prediction_U':ratio(tn,math.fsum(n)),'binary_known_confusion_T_nonT_U':[t,n],'denominator_note':'Fractional raster point/sample mass; missing/ambiguous/invalid are retained separately, not physical native D/E labels.'}

def invented():
 p=np.array([[0,999],[7,-1]],np.int32);a=np.array([[False,True],[False,False]])
 w=sample_point(p,a,None,[.5,.5],False)[0];assert w==[.5,0,.25,.25,0]
 w=sample_point(p,a,None,[-.25,0],False)[0];assert w==[.75,0,0,0,.25]
 l=np.zeros((512,512),np.float32);q=[.49,.49];pbig=np.zeros((512,512),np.int32);abig=np.zeros((512,512),bool);l[0,0]=1;l[0,1]=l[1,0]=l[1,1]=-1
 z=sample_point(pbig,abig,l,q,True);assert z[2:4]==(0,1)
 assert sample_point(pbig,abig,np.zeros((512,512),np.float32),[0,0],True)[2:4]==(1,1)
 assert sample_point(pbig,abig,l,[0,0],False)[2:4]==(2,2)
 assert not finite(None) and not finite([-1.,-1.]) and not finite([None,None])
 m=region(np.ones((2,2),np.float32),p,a,True);assert m['TP']==2 and m['FP']==1 and m['IoU']==2/3 and m['ambiguity_excluded']==1
 u=region(None,p,a,False);assert u['prediction_U_pixels']==3 and u['TP'] is None
 return {'PASS':True,'cases':8,'method':'Own scalar four-tap / nearest / zero tie / U / ambiguity / border mass and flat confusion bincount toy checks.','source_or_SAM_output_arrays_read':0}

def prepare():
 assert sys.executable=='/home/minjae/anaconda3/envs/pallet-yolo26/bin/python' and np.__version__=='1.26.4'
 checks=invented();put(D/'PREPARATION_COMPLETE.json',{'completed':True,'PASS':True,'code':bind(Path(__file__)),'checks':checks,'authorization':'Root actual evaluation terminal completion trigger required; no helper/model rerun.','expected_evaluation_complete':{'path':str(CP),'sha256':CP_SHA},'manifest':bind(MAN),'scope':'qpoint4096 and wholepixel512 direct source part/ambiguity/SAM-logit audit,96grid sample-equivalent recomputation excluded; crop parity evidence bindings verified.','weights_RGB_model_private_held_reads':0})
 print(json.dumps({'prepared':True,'code':bind(Path(__file__)),'preparation':bind(D/'PREPARATION_COMPLETE.json')}))

def run():
 assert sys.executable=='/home/minjae/anaconda3/envs/pallet-yolo26/bin/python' and np.__version__=='1.26.4'
 assert sha(CP)==CP_SHA;complete=read(CP);assert complete['completed'] and complete['PASS'] and complete['query_N']==4096 and complete['image_N']==512
 for k in ['rows','images','summary','manifest','prediction_complete','source_TRAIN_sidecar']:checked(complete[k])
 input_before={k:bind(complete[k]['path']) for k in ['rows','images','summary','manifest','prediction_complete','source_TRAIN_sidecar']}
 manifest=lines(MAN);rows=lines(complete['rows']['path']);images=lines(complete['images']['path']);summary=read(complete['summary']['path']);prediction=read(complete['prediction_complete']['path'])
 assert len(manifest)==len(rows)==4096 and len(images)==len(prediction['scenes'])==512 and all(r['split']=='TRAIN' for r in manifest)
 assert [r['original_global_index'] for r in rows]==[r['original_global_index'] for r in manifest] and len({r['original_global_index'] for r in rows})==4096
 img_by={r['scene_index']:r for r in images};scene_by={r['scene_index']:r for r in prediction['scenes']};source_by=defaultdict(list);manifest_by=defaultdict(list)
 for r in rows:source_by[r['scene_index']].append(r)
 for r in manifest:manifest_by[r['scene_index']].append(r)
 keys=['original_global_index','original_occurrence_id','scene_index','base_index','native_slot','query_xy','L','split','scene_owner_point']+list(manifest[0]['original_input_flags'])
 with np.load(complete['source_TRAIN_sidecar']['path'],allow_pickle=False) as f:train={k:f[k].copy() for k in keys}
 tidx={int(i):j for j,i in enumerate(train['original_global_index'])};assert len(tidx)==4096
 facts=[];assets={};crop_chunks={};measured=missing=originalU=predU=0;available_scenes=0;whole_tot=defaultdict(int)
 for scene,s in scene_by.items():
  rr=source_by[scene];mm=manifest_by[scene];im=img_by[scene];assert s['original_input_metadata8']==mm and [r['native_slot'] for r in rr]==list(range(8))
  sb=im['source_buffer'];checked(sb);assets[sb['path']]=sb
  assert sb==mm[0]['full_RGB_packet']
  with np.load(sb['path'],allow_pickle=False) as f:part=f['part_id'].copy();amb=f['ambiguity'].copy()
  assert part.dtype==np.int32 and amb.dtype==bool and part.shape==amb.shape==(512,512)
  available=bool(s['prompt_available'] and s['runtime_available']);assert available==bool(im['prompt_available'] and im['runtime_available']);logits=None
  if available:
   checked(s['outputs']);assets[s['outputs']['path']]=s['outputs'];assert im['prediction_output']==s['outputs']
   with np.load(s['outputs']['path'],allow_pickle=False) as f:
    logits=f['image_logits'].copy();mask=f['mask_logits_gt0'].copy();meta={k:f[k].copy() for k in ['original_query_xy','original_L','native_slot','original_global_index','original_occurrence_id','original_query_available','prediction_available','box_xyxy']}
   assert logits.dtype==np.float32 and logits.shape==(512,512) and np.isfinite(logits).all() and mask.dtype==bool and np.array_equal(mask,logits>0)
   assert meta['native_slot'].tolist()==list(range(8)) and meta['original_global_index'].tolist()==[r['original_global_index'] for r in mm]
   assert meta['original_occurrence_id'].tolist()==[r['original_occurrence_id'] for r in mm]
   assert np.array_equal(meta['box_xyxy'],np.array(mm[0]['original_q8_context_box_xyxy'],np.float64))
   available_scenes+=1
  else:assert s['outputs'] is None and not any(r['prediction_available'] for r in rr)
  region_result=region(logits,part,amb,available);same(region_result,im['metrics'],'scene'+str(scene)+'/region',tol=0)
  for k,v in region_result.items():
   if isinstance(v,int):whole_tot[k]+=v
  for k,(r,m) in enumerate(zip(rr,mm)):
   for key,value in m.items():same(value,r[key],'original_metadata/'+str(m['original_global_index'])+'/'+key,tol=0)
   i=m['original_global_index'];j=tidx[i];q=m['original_query_xy'];L=m['original_L'];oq=m['original_input_flags']['query_available'];av=bool(oq and available)
   assert r['original_input_available']==oq and r['prediction_available']==av
   assert int(train['scene_index'][j])==scene and int(train['base_index'][j])==m['base_index'] and int(train['native_slot'][j])==k and str(train['original_occurrence_id'][j])==m['original_occurrence_id'] and str(train['split'][j])=='TRAIN'
   qnum=np.asarray([np.nan,np.nan] if q is None else q,np.float64);lnum=np.nan if L is None else L
   assert np.array_equal(train['query_xy'][j],qnum,equal_nan=True) and (float(train['L'][j])==lnum or (np.isnan(train['L'][j]) and np.isnan(lnum)))
   for flag,value in m['original_input_flags'].items():assert bool(train[flag][j])==value
   if available:
    assert np.array_equal(meta['original_query_xy'][k],qnum,equal_nan=True) and (meta['original_L'][k]==lnum or (np.isnan(meta['original_L'][k]) and np.isnan(lnum)))
    assert bool(meta['original_query_available'][k])==oq and bool(meta['prediction_available'][k])==av
   originalU+=int(not oq);predU+=int(not av)
   f={'metadata':m,'weight':None,'near':2,'bil':2,'hard':None}
   if finite(q):
    w,h,n,b,nv,bv,support=sample_point(part,amb,logits,q,av);a=train['scene_owner_point'][j];sidecar=[float(a[0]+a[1]),float(a[2]),float(a[3]),float(a[4]),float(a[5])]
    same(w,sidecar,'pointtruth_sidecar/'+str(i),tol=2e-13);same(w,r['point_owner_soft5'],'pointtruth_saved/'+str(i),tol=2e-13)
    assert not r['point_truth_missing'] and r['nearest_raster_class_index']==h and r['nearest_raster_class']==TRUTH[h]
    assert r['prediction_nearest_index']==n and r['prediction_bilinear_index']==b and r['prediction_nearest']==PRED[n] and r['prediction_bilinear']==PRED[b]
    same(nv,r['nearest_logit'],'nearestvalue/'+str(i),tol=0);same(bv,r['bilinear_logit'],'bilinearvalue/'+str(i),tol=2e-11);same(support,r['bilinear_inbounds_weight'],'support/'+str(i),tol=2e-15)
    f.update(weight=w,near=n,bil=b,hard=h);measured+=1
   else:
    assert r['point_truth_missing'] and r['point_owner_soft5'] is None and np.isnan(train['scene_owner_point'][j]).all() and r['prediction_nearest_index']==r['prediction_bilinear_index']==2;missing+=1
   if finite(q) and L is not None and math.isfinite(L) and L>0:
    assert r['rgb96_original_byteparity'] is True and r['valid96_original_byteparity'] is True and len(r['grid96_xy_sha256'])==64
    cb=m['saved_crop_chunk'];crop_chunks[cb['path']]=cb
   else:assert r['rgb96_original_byteparity'] is None and r['valid96_original_byteparity'] is None
   facts.append(f)
 assert measured==4048 and missing==48 and originalU==predU==80 and available_scenes==506
 for b in crop_chunks.values():checked(b)
 group_names=['ALL']
 for key in ['native_slot','variant','material','aspect','geometry_family']:
  vals=range(8) if key=='native_slot' else sorted({m.get(key,m['factors'].get(key)) for m in manifest})
  group_names += [key+'::'+str(v) for v in vals]
 savedgroups={g['group']:g for g in summary['groups']};assert set(savedgroups)==set(group_names)
 for name in group_names:
  key,val=(name.split('::',1) if name!='ALL' else (None,None));ff=[f for f in facts if key is None or str(f['metadata'].get(key,f['metadata']['factors'].get(key)))==val];g=savedgroups[name]
  count={'N':len(ff),'original_inputU':sum(not f['metadata']['original_input_flags']['query_available'] for f in ff),'predictionU':sum(not (f['metadata']['original_input_flags']['query_available'] and scene_by[f['metadata']['scene_index']]['prompt_available'] and scene_by[f['metadata']['scene_index']]['runtime_available']) for f in ff),'point_truth_missing_N':sum(f['weight'] is None for f in ff),'point_truth_measured_N':sum(f['weight'] is not None for f in ff)}
  for k,v in count.items():same(v,g[k],name+'/'+k,tol=0)
  for mode,keypred in [('nearest','near'),('bilinear','bil')]:
   cm=[[math.fsum(f['weight'][cl] for f in ff if f['weight'] is not None and f[keypred]==pr) for pr in range(3)] for cl in range(5)]
   same(describe(cm),g['point'][mode],name+'/point/'+mode)
   hard=[[0]*3 for _ in range(5)]
   for f in ff:
    if f['weight'] is not None:hard[f['hard']][f[keypred]]+=1
   same({'truth_order':TRUTH,'prediction_order':PRED,'confusion':hard,'N':sum(sum(x) for x in hard),'note':'Nearest categorical raster diagnostic; separate from fractional point truth.'},g['nearest_raster'][mode],name+'/nearest_raster/'+mode,tol=0)
 # Renderer clean-pixel evidence: verify original inherited SHA against actual saved clean128 PNG bytes, no original RGB/cache arrays loaded.
 from PIL import Image
 cleanbind=complete['previews']['clean128'];checked(cleanbind);cleans=lines(cleanbind['path']);assert len(cleans)==128
 manifest_by_id={r['original_global_index']:r for r in manifest}
 for r in cleans:
  m=manifest_by_id[r['original_global_index']];checked(r['clean96'])
  with Image.open(r['clean96']['path']) as im:assert im.mode=='RGB' and im.size==(96,96);decoded=hashlib.sha256(im.tobytes()).hexdigest()
  assert decoded==r['decoded_rgb96_sha256']==r['original_rgb96_sha256']==m['saved_rgb96_sha256_inherited'] and r['clean_pixels_unchanged']
 for k,b in input_before.items():assert bind(b['path'])==b
 for b in assets.values():checked(b)
 assert sha(CP)==CP_SHA
 result={'completed':True,'PASS':True,'UTC':datetime.now(timezone.utc).isoformat(),'code':bind(Path(__file__)),'preparation':bind(D/'PREPARATION_COMPLETE.json'),'evaluation_complete':bind(CP),'inputs':input_before,'method':'Own scalar floor-half-up + four-tap zero-border onehot truth and float64 logit sampling; own scalar hard/soft grouped confusion sums via math.fsum; independent whole-raster Boolean partition+bincount. No evaluator/helper imports.','query_rows':4096,'scenes':512,'measured_point_rows':4048,'missing_original_q_rows':48,'original_inputU':80,'predictionU':80,'available_scene_outputs':506,'unavailable_nullq8_scene_rows':48,'strata_groups_audited':len(group_names),'point_hard_soft_sampler_metric_blocks':len(group_names)*4,'field_comparisons':comparisons,'maximum_observed_float_delta':max_float_delta,'soft_sum_tolerance':2e-10,'point_truth_tolerance':2e-13,'bilinear_logit_tolerance':2e-11,'whole_pixel_counts_and_IoU_Dice_exact':True,'whole_pixel_count_sums_all512':dict(whole_tot),'source_part_ambiguity_and_SAM_output_files_verified':len(assets),'original_crop_chunk_SHA_bindings_verified':len(crop_chunks),'actual_clean96_PNG_decoded_original_hashes_verified':128,'original_q_L_ID_native_split_flags_all_preserved':True,'all_saved_input_SHA_bindings_unchanged':True,'sampling_model_prompt_threshold_or_label_changes':False,'scope_limits':['Did not re-sample all37M full96 points or independently rebuild crop96 CM/central4/32. Verified all crop-parity row attestations and original chunk SHA bindings; independently decoded fixed128 clean96 PNGs to original inherited pixel SHA.','Original source NPZ containers were streamed for SHA and only part_id/ambiguity arrays decoded; RGB/depth/pose/teacher keys were not loaded. Original crop NPZ containers were streamed for SHA only, not decoded.','Numerical integrity of source literal raster ownership/proposal overlap, not physical native identity, native visibility, external/self gold or full-goal success.','Source TRAIN apparent diagnostics only; no held/private/V112/model/weights/forward/train or selection.'],'selected_model':None,'adopted':False,'training_admission':False,'fully_solved':False}
 put(D/'REVIEW.json',result);print(json.dumps({'PASS':True,'review':bind(D/'REVIEW.json'),'groups':len(group_names),'max_delta':max_float_delta,'comparisons':comparisons},ensure_ascii=False))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--run',action='store_true');a=ap.parse_args();assert a.prepare!=a.run
 if a.prepare:prepare()
 else:run()
