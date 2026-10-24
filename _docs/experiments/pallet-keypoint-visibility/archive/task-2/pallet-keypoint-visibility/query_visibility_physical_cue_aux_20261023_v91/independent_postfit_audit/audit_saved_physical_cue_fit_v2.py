"""Independent saved-only V91 audit. Never imports model/trainer or runs a head."""
from pathlib import Path
from collections import Counter
import argparse,ast,json,hashlib,datetime,math,sys
import numpy as np
import torch
from torch.nn import functional as F
O=Path(__file__).resolve().parent;R=O.parent
FIELDS=('p3_roi','local_rgb96','local_valid96')

def bind(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb')as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return {'path':str(p.resolve()),'sha256':h.hexdigest(),'bytes':p.stat().st_size}
def check(x):
 a=bind(x['path']);assert a['sha256']==x['sha256'],x['path']
 if 'bytes'in x:assert a['bytes']==x['bytes']
def read(p):return json.loads(Path(p).read_text())
def jl(p):return[json.loads(s)for s in Path(p).read_text().splitlines()if s.strip()]
def save(p,x):
 with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def ratio(a,b):return float(a/b)if b else None
def cm(y,z,av,use):
 p=np.where(av,z>=0,2).astype(np.int8);m=np.array([[np.count_nonzero(use&(y==a)&(p==b))for b in(0,1,2)]for a in(0,1)],np.int64)
 return {'N':int(use.sum()),'rows_D_E_cols_D_E_U':m.tolist(),'D_precision':ratio(m[0,0],m[:,0].sum()),'D_recall':ratio(m[0,0],m[0].sum()),'E_precision':ratio(m[1,1],m[:,1].sum()),'E_recall':ratio(m[1,1],m[1].sum()),'U_rate':ratio(m[:,2].sum(),m.sum())}
def metrics(rows,s):
 y=s['labels'];split=np.array([r['split']for r in rows]);ans={}
 for part in('TRAIN_LAB','VAL_HALL'):
  base=split==part
  if not base.any():continue
  known=base&np.isin(y,(0,1));d={'all_views':int(base.sum()),'known_views':int(known.sum()),'teacher_U':int((base&(y==-1)).sum())}
  for kind in('region','reference'):
   z=s[kind+'_logit'];av=s[kind+'_available'];d[kind]=cm(y,z,av,known);p=np.where(av,z>=0,2)
   d[kind+'_teacher_U_prediction_distribution']={str(k):int((base&(y==-1)&(p==k)).sum())for k in(0,1,2)}
  lookup={(r['cue_id'],r['view']):i for i,r in enumerate(rows)if r['split']==part};pairs={}
  for aft,name in[(1,'D_to_E'),(0,'D_to_D')]:
   ix=[]
   for j,r in enumerate(rows):
    i=lookup.get((r['cue_id'],'before'))
    if r['split']==part and r['view']=='after'and y[j]==aft and i is not None and y[i]==0:ix.append((i,j))
   q={'N':len(ix),'effective_pair_weight':float(sum(rows[j]['duplicate_weight']for i,j in ix))}
   for kind in('region','reference'):
    z=s[kind+'_logit'];av=s[kind+'_available'];n=sum(bool(av[i]and av[j]and z[i]<0 and(z[j]>=0 if aft else z[j]<0))for i,j in ix);q[kind]={'both_correct':n,'rate':ratio(n,len(ix))}
   pairs[name]=q
  d['paired_views']=pairs;ans[part]=d
 return ans

def gates(m):
 v=m['VAL_HALL'];g={}
 for kind in('region','reference'):
  for key,t,op in[('D_recall',.8,'ge'),('E_recall',.8,'ge'),('E_precision',.5,'ge'),('U_rate',.1,'le')]:
   n=v[kind][key];g[kind+'_'+key+'_'+op]=n is not None and(n>=t if op=='ge'else n<=t)
 n=v['paired_views']['D_to_E']['region']['rate'];g['region_D_to_E_pair_both_correct_ge_0p7']=n is not None and n>=.7
 return g

def macro(loss,y,w,use):
 d={}
 for c in(0,1):
  m=use&(y==c);ww=w[m].astype(np.float64);assert m.any()and ww.sum()>0;d[str(c)]={'N':int(m.sum()),'weight':float(ww.sum()),'weighted_mean':float(np.sum(loss[m].astype(np.float64)*ww)/ww.sum())}
 return float(np.mean([d[str(c)]['weighted_mean']for c in(0,1)])),d

def bce(z,av,y):
 z=np.where(av,z,0).astype(np.float64);yy=np.clip(y,0,1).astype(np.float64);return np.maximum(z,0)-z*yy+np.log1p(np.exp(-np.abs(z)))
def history_replay(h):
 assert len(h)==20 and[r['update']for r in h]==list(range(100,2001,100));best=float('inf');bu=None
 for row in h:
  pc=row['per_class_BCE'];assert set(pc)=={'0','1'};n=float(np.mean([pc[str(c)]['weighted_mean']for c in(0,1)]));assert n==row['heldout_macroBCE']and math.isfinite(n)
  selected=n<best
  if selected:best=n;bu=row['update']
  assert row['saved']is selected and row['best_update']==bu
  assert row['known_VAL_regions']==170 and row['validation_frames']==56 and row['validation_tiles']==7224 and row['mosaic_before_scoring_GT']is True
  assert row['cumulative_sampled_nominal']==row['update']*96 and 0<=row['cumulative_sampled_factual_available']<=row['cumulative_sampled_nominal']
 return {'best_update':bu,'selected_history_macroBCE':best,'eligible_checkpoints':20,'update0_excluded':True,'strict_earliest_min':True}

def interp9(a,x,y):
 x=np.clip(np.asarray(x,np.float64),0,8);y=np.clip(np.asarray(y,np.float64),0,8);x0=np.floor(x).astype(int);y0=np.floor(y).astype(int);x1=np.minimum(x0+1,8);y1=np.minimum(y0+1,8);dx=x-x0;dy=y-y0
 return(a[y0[:,None],x0[None,:]]*(1-dx)[None,:]*(1-dy)[:,None]+a[y0[:,None],x1[None,:]]*dx[None,:]*(1-dy)[:,None]+a[y1[:,None],x0[None,:]]*(1-dx)[None,:]*dy[:,None]+a[y1[:,None],x1[None,:]]*dx[None,:]*dy[:,None])
def mosaic64(logits,valid,queries,sides,hw):
 H,W=hw;num=np.zeros((H,W),np.float64);den=np.zeros_like(num)
 for a,v,q,L in zip(logits,valid,queries,sides):
  L=int(L);lo=np.asarray(q)-L/2;x0,y0=map(int,lo);xs=np.arange(max(0,x0),min(W,x0+L));ys=np.arange(max(0,y0),min(H,y0+L))
  if not len(xs)or not len(ys):continue
  ux=2*(xs-q[0])/L;uy=2*(ys-q[1])/L;mx=(ux+1)*9/2-.5;my=(uy+1)*9/2-.5
  tent=np.maximum(0,1-abs(uy))[:,None]*np.maximum(0,1-abs(ux))[None,:]
  vv=v.astype(np.float64);value=interp9(a.astype(np.float64)*vv,mx,my);support=interp9(vv,mx,my);sl=np.ix_(ys,xs)
  num[sl]+=tent*value;den[sl]+=tent*support
 av=den>0;out=np.zeros_like(num);out[av]=num[av]/np.maximum(den[av],1e-12);return out,av,den

def sample_native_cpu(a,v,points):
 # Independent expression of the frozen CPUF32 scoring recipe, no model import.
 H,W=a.shape;p=np.asarray(points,np.float64);finite=np.isfinite(p).all(1);inside=finite&(p[:,0]>=0)&(p[:,0]<=W-1)&(p[:,1]>=0)&(p[:,1]<=H-1);safe=np.where(np.isfinite(p),p,0)
 norm=np.empty_like(safe);norm[:,0]=2*(safe[:,0]+.5)/W-1;norm[:,1]=2*(safe[:,1]+.5)/H-1
 g=torch.tensor(norm,dtype=torch.float32)[None,None];vv=torch.from_numpy(v.astype(np.float32))[None,None];aa=torch.from_numpy(a.astype(np.float32))[None,None]*vv
 with torch.inference_mode():
  d=F.grid_sample(vv,g,mode='bilinear',padding_mode='border',align_corners=False)[0,0,0];n=F.grid_sample(aa,g,mode='bilinear',padding_mode='border',align_corners=False)[0,0,0];z=(n/d.clamp_min(1e-12)).numpy()
 return z,inside&(d.numpy()>0)

def expected_factual(tiles):
 out=np.zeros((len(tiles),9,9),bool);off=(np.arange(9)+.5)/9-.5;u=(np.arange(96)+.5)/96-.5
 for i,t in enumerate(tiles):
  x,y=t['query_xy'];L=t['roi_side'];H,W=t['raw_hw'];gx=x+off*L;gy=y+off*L;nx=(gx>=0)&(gx<W);ny=(gy>=0)&(gy<H)
  vx=(x+u*L>=-.5)&(x+u*L<W-.5);vy=(y+u*L>=-.5)&(y+u*L<H-.5)
  px=np.array([vx[math.floor(j*96/9):math.ceil((j+1)*96/9)].any()for j in range(9)]);py=np.array([vy[math.floor(j*96/9):math.ceil((j+1)*96/9)].any()for j in range(9)])
  out[i]=ny[:,None]&nx[None,:]&py[:,None]&px[None,:]
 return out

def nominal_available(valid,terms):
 ans=np.zeros(len(terms['tile_index']),bool)
 for i in np.flatnonzero(terms['loss_admitted']):
  grid=terms['uncertainty9_normalized_xy_safe'][i].astype(np.float64);xy=np.clip((grid+1)*4.5-.5,0,8);vv=valid[terms['tile_index'][i]].astype(np.float64);ok=True
  for x,y in xy:
   ix,iy=math.floor(x),math.floor(y);jx,jy=min(ix+1,8),min(iy+1,8);dx,dy=x-ix,y-iy
   value=vv[iy,ix]*(1-dx)*(1-dy)+vv[iy,jx]*dx*(1-dy)+vv[jy,ix]*(1-dx)*dy+vv[jy,jx]*dx*dy
   ok=ok and value>0
  ans[i]=ok
 return ans

def static_boundary(protocol,trainer_path):
 tree=ast.parse(Path(trainer_path).read_text());defs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
 pred=defs['predict_mosaics'];assert[n.arg for n in pred.args.args]==['model','bank','meta','frame_indices','mosaic_module','device']
 names={n.id for n in ast.walk(pred)if isinstance(n,ast.Name)};assert not({'teacher','loss_bank'}&names)
 calls=[n for n in ast.walk(pred)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)];assert not any(n.func.id in {'read','lines','open'}for n in calls)
 validate=defs['validate'];src=ast.get_source_segment(Path(trainer_path).read_text(),validate);assert src.index('predict_mosaics(')<src.index('score_completed_mosaics(')
 modeltree=ast.parse(Path(protocol['bindings']['model_code']['path']).read_text());fwd=next(n for n in ast.walk(modeltree)if isinstance(n,ast.FunctionDef)and n.name=='forward');assert tuple(n.arg for n in fwd.args.args[1:])==FIELDS
 return {'PASS':True,'three_forward_fields':list(FIELDS),'prediction_no_GTargument':True,'alltiles_mosaic_before_scoring':True,'static_not_dynamic_taint_proof':True}

def generated():
 y=np.array([0,0,1,1,-1]);z=np.array([-2,2,2,-2,100],np.float32);av=np.array([1,0,1,1,1],bool);known=y!=-1
 assert cm(y,z,av,known)['rows_D_E_cols_D_E_U']==[[1,0,1],[1,1,0]]
 assert bce(np.array([20,-20]),np.array([0,0],bool),np.array([0,1])).tolist()==[math.log(2),math.log(2)]
 a=np.array([.1,.3,.5,.9,1e99]);m,_=macro(a,y,np.ones(5),known);a[-1]=-1e99;assert macro(a,y,np.ones(5),known)[0]==m
 rows=[{'split':'VAL_HALL','cue_id':str(i//2),'view':'before'if i%2==0 else'after','duplicate_weight':1.}for i in range(4)]
 toy={'labels':np.array([0,1,0,0]),'region_logit':np.array([-2,2,-2,-2]),'reference_logit':np.array([-2,2,-2,-2]),'region_available':np.ones(4,bool),'reference_available':np.ones(4,bool)};r=metrics(rows,toy);assert r['VAL_HALL']['paired_views']['D_to_E']['region']['rate']==1 and r['VAL_HALL']['paired_views']['D_to_D']['N']==1 and all(gates(r).values())
 hist=[];best=float('inf');bu=None
 for i,update in enumerate(range(100,2001,100)):
  score=.8 if i<1 else .4 if i<3 else .6;saved=score<best
  if saved:best=score;bu=update
  hist.append({'update':update,'per_class_BCE':{'0':{'weighted_mean':score},'1':{'weighted_mean':score}},'heldout_macroBCE':score,'saved':saved,'best_update':bu,'known_VAL_regions':170,'validation_frames':56,'validation_tiles':7224,'mosaic_before_scoring_GT':True,'cumulative_sampled_nominal':update*96,'cumulative_sampled_factual_available':update*96})
 assert history_replay(hist)['best_update']==200
 tile=np.full((1,9,9),2,np.float32);vv=np.ones_like(tile,bool);vv[0,:,0]=False
 a,av,d=mosaic64(tile,vv,np.array([[8,8]]),np.array([16]),[17,17]);assert np.all(a[av]==2)and not av[16].any()
 z,v=sample_native_cpu(a.astype(np.float32),av,[[8.,8.],[6.,6.]]);assert v.all()and np.array_equal(z,[2,2])
 proof={'completed':True,'PASS':True,'code':bind(__file__),'scope':bind(O/'SCOPE_AND_NUMERIC_POLICY_BEFORE_OUTPUTS.json'),'CM_U_known_only_macro':True,'all9gates_and_paired':True,'earlieststrictmin_update0excluded':True,'mosaic_constant_with_invalid_support_not_diluted':True,'mosaic_boundary_halfopen':True,'CPU_scoring_bilinear_border':True,'no_model_import_or_forward':True,'actual_run_outputs_GT_feature_RGB_weight_reads':0}
 save(O/'GENERATED_SAVED_AUDIT_TESTS_V2.json',proof);print(json.dumps(proof,indent=2))

def actual(completed_sha):
 torch.set_num_threads(1);assert str(torch.__version__)=='2.5.1+cu121'
 prep=read(O/'PREPARATION_SEAL_V2.json');assert prep['PASS']and prep['completed']
 for k in('scope','code','generated'):check(prep[k])
 scope=read(prep['scope']['path'])
 for z in scope['bindings'].values():check(z)
 p=read(scope['bindings']['protocol']['path']);run=Path(p['run_directory']);donepath=run/'FIT_COMPLETE.json';assert bind(donepath)['sha256']==completed_sha
 done=read(donepath);assert done['completed']and done['optimizer_updates']==2000 and done['current_PnP']==0
 for k in('model_freeze','history','scores','tilemaps','mosaics','source_results'):check(done[k])
 # Bind all protocolmetadata/code. Old source checkpointparameter neverloaded.
 for k,z in p['bindings'].items():
  if not str(z['path']).endswith('.pt'):check(z)
 out=O/'completed_audit_v2';out.mkdir(exist_ok=False)
 save(out/'STARTED.json',{'parent_exactFIT_COMPLETE':bind(donepath),'preparation':bind(O/'PREPARATION_SEAL_V2.json'),'new_forward_fit_current_PnP':0})
 freeze=read(done['model_freeze']['path']);assert freeze['completed']and freeze['optimizer_updates']==2000 and freeze['external_threshold']==.5 and freeze['current_used_for_fit_selection']is False and freeze['PnP']==0
 check(freeze['checkpoint']);assert freeze['trainer']['sha256']==scope['bindings']['trainer_preparation']['sha256']and freeze['protocol']['sha256']==scope['bindings']['protocol']['sha256']
 history=read(done['history']['path']);ha=history_replay(history);assert done['best_update']==freeze['best_update']==ha['best_update']and freeze['selection_macroBCE']==ha['selected_history_macroBCE']
 source=read(done['source_results']['path']);rows=jl(p['bindings']['teacher_views']['path']);tiles=jl(p['bindings']['source_grid']['path']);frames=jl(p['bindings']['source_frames']['path'])
 assert len(rows)==604 and len(tiles)==12900 and len(frames)==100
 with np.load(done['scores']['path'],allow_pickle=False)as z:s={k:z[k]for k in z.files}
 keys={'region_logit','reference_logit','region_available','reference_available','labels','weights','region_NLL_FP32_DIAGNOSTIC_U_CLAMPED','reference_NLL_FP32_DIAGNOSTIC_U_CLAMPED','cue_view_index','cue_id','pair_id','frame_input_id','split','view','reference_defined'};assert set(s)==keys
 for k in keys:assert s[k].shape==(604,)
 for k in('region_logit','reference_logit','region_NLL_FP32_DIAGNOSTIC_U_CLAMPED','reference_NLL_FP32_DIAGNOSTIC_U_CLAMPED'):assert s[k].dtype==np.float32 and np.isfinite(s[k]).all()
 for k in('region_available','reference_available','reference_defined'):assert s[k].dtype==bool
 assert np.array_equal(s['cue_view_index'],np.arange(604,dtype=np.int32))
 for k in('cue_id','pair_id','frame_input_id','split','view'):assert np.array_equal(s[k],[r[k]for r in rows])
 assert np.array_equal(s['labels'],[r['teacher_label']for r in rows])and np.array_equal(s['weights'],[r['duplicate_weight']for r in rows])and np.array_equal(s['reference_defined'],[r['reference_xy_unchanged']is not None for r in rows])
 assert Counter(s['labels'].tolist())=={0:261,1:37,-1:306}
 with np.load(done['tilemaps']['path'],allow_pickle=False)as z:
  assert set(z.files)=={'tile_index','tile_id','physical_external_logit','map_valid'};tm={k:z[k]for k in z.files}
 assert np.array_equal(tm['tile_index'],np.arange(12900))and np.array_equal(tm['tile_id'],[t['tile_id']for t in tiles])
 maps=tm['physical_external_logit'];valid=tm['map_valid'];assert maps.dtype==np.float32 and maps.shape==(12900,9,9)and np.isfinite(maps).all()and valid.dtype==bool and valid.shape==maps.shape
 ev=expected_factual(tiles);assert np.array_equal(valid,ev)
 mm=read(done['mosaics']['path']);assert mm['completed']and mm['frame_count']==100 and mm['tile_count']==12900 and mm['GT_used_for_construction']is False
 qs=np.array([t['query_xy']for t in tiles]);Ls=np.array([t['roi_side']for t in tiles]);fi_lookup={f['frame_input_id']:i for i,f in enumerate(frames)};tis=np.array([fi_lookup[t['frame_input_id']]for t in tiles]);M=float(np.max(np.abs(maps)));gamma=scope['mosaic_replay_numeric_policy']['gamma4096'];bound=4*gamma*max(1,M)/(1-gamma)
 reconstructed={k:np.zeros(604,np.float32)for k in('region_logit','reference_logit')};reconstructed.update({k:np.zeros(604,bool)for k in('region_available','reference_available')});numeric=[]
 for fi,(frame,m)in enumerate(zip(frames,mm['frames'])):
  assert m['frame_index']==fi and m['frame_input_id']==frame['frame_input_id']and m['image_sha256']==frame['image_sha256']and m['raw_hw']==frame['raw_hw'];check(m['file'])
  with np.load(m['file']['path'],allow_pickle=False)as z:
   assert set(z.files)=={'physical_external_logit','native_valid','input_support'};native=z['physical_external_logit'];nv=z['native_valid'];support=z['input_support']
  assert native.shape==nv.shape==support.shape==(480,640)and native.dtype==support.dtype==np.float32 and nv.dtype==bool and np.isfinite(native).all()and np.isfinite(support).all()and(support>=0).all()
  assert np.array_equal(nv,support>0)and not native[~nv].any()
  use=np.flatnonzero(tis==fi);assert len(use)==129
  rebuilt,rv,rd=mosaic64(maps[use],valid[use],qs[use],Ls[use],frame['raw_hw']);assert np.array_equal(rv,nv)
  delta=np.abs(rebuilt-native.astype(np.float64));mx=float(delta.max());den_delta=np.abs(rd-support.astype(np.float64));positive=rd>0
  drel=float(np.max(den_delta[positive]/rd[positive]))if positive.any()else 0.
  entry={'frame_index':fi,'logit_max_abs_error':mx,'theoretical_bound_before_outputs':bound,'support_max_relative_error':drel,'support_relative_bound':gamma/(1-gamma),'hard_sign_differences':int(np.count_nonzero(rv&((rebuilt>=0)!=(native>=0)))),'saved_nearzero_within_engineering_bound':int(np.count_nonzero(nv&(abs(native)<=bound)))};numeric.append(entry)
  if not(mx<=bound and drel<=gamma/(1-gamma)):
   save(out/'FAILED_MOSAIC_NUMERIC_FRAME.json',entry);raise AssertionError('Independentmosaic arithmeticbound')
  for ri,row in enumerate(rows):
   if row['frame_input_id']!=frame['frame_input_id']or row['reference_xy_unchanged']is None:continue
   zz,aa=sample_native_cpu(native,nv,row['uncertainty9_xy_GT_only']);zr,ar=sample_native_cpu(native,nv,[row['reference_xy_unchanged']]);ok=bool(aa.all());rok=bool(ar[0]);reconstructed['region_available'][ri]=ok;reconstructed['reference_available'][ri]=rok
   reconstructed['region_logit'][ri]=float(zz.mean())if ok else 0.;reconstructed['reference_logit'][ri]=float(zr[0])if rok else 0.
 for k,z in reconstructed.items():assert np.array_equal(s[k],z),k
 mr=metrics(rows,s);assert mr==source['source_metrics'];g=gates(mr);assert len(g)==9 and g==source['predeclared_source_gate']and source['all_nine_pass']==all(g.values())
 assert mr['VAL_HALL']['known_views']==170 and mr['VAL_HALL']['all_views']==336 and mr['VAL_HALL']['teacher_U']==166 and mr['VAL_HALL']['paired_views']['D_to_E']['N']==22 and mr['VAL_HALL']['paired_views']['D_to_D']['N']==63
 knownval=(s['split']=='VAL_HALL')&np.isin(s['labels'],(0,1));storedmacro,storedpc=macro(s['region_NLL_FP32_DIAGNOSTIC_U_CLAMPED'],s['labels'],s['weights'],knownval);assert storedmacro==source['final_alltile_macroBCE']and storedpc==source['final_alltile_per_class']
 analytic={};nll_numeric={}
 for kind in('region','reference'):
  a=bce(s[kind+'_logit'],s[kind+'_available'],s['labels']);sm=s[kind+'_NLL_FP32_DIAGNOSTIC_U_CLAMPED'];known=np.isin(s['labels'],(0,1));analytic[kind]=dict(zip(('macro','classes'),macro(a,s['labels'],s['weights'],knownval)));d=sm[known].astype(np.float64)-a[known];nll_numeric[kind]={'known_only_N':int(known.sum()),'max_abs_storedFP32_minus_analyticFP64':float(abs(d).max()),'mean_signed':float(d.mean()),'teacher_U_placeholder_excluded':True}
 with np.load(p['bindings']['teacher_units']['path'],allow_pickle=False)as z:terms={k:z[k]for k in('tile_index','training_label','loss_admitted','loss_weight','uncertainty9_normalized_xy_safe')}
 probe=run/'NOMINAL_GT_TERMS_FACTUAL_SUPPORT_DIAGNOSTIC.npz'
 with np.load(probe,allow_pickle=False)as z:assert np.array_equal(z['unit_index'],np.arange(77916))and np.array_equal(z['loss_admitted'],terms['loss_admitted']);factual=z['factual_available']
 assert np.array_equal(factual,nominal_available(valid,terms))
 split=np.array([t['split']for t in tiles])[terms['tile_index']];pools={c:np.flatnonzero((split=='TRAIN_LAB')&terms['loss_admitted']&(terms['training_label']==c))for c in(0,1)};prob={c:terms['loss_weight'][pools[c]]/terms['loss_weight'][pools[c]].sum()for c in(0,1)}
 preflight=read(run/'SOURCE_GPU_PREFLIGHT.json');assert preflight['PASS']and preflight['optimizer_updates']==0 and preflight['training_term_counts']=={str(c):len(pools[c])for c in(0,1)}
 assert preflight['training_class_effective_mass']=={str(c):float(terms['loss_weight'][pools[c]].sum())for c in(0,1)}
 rng=np.random.default_rng(20261023);cum=0
 for update in range(1,2001):
  ix=np.concatenate([rng.choice(pools[c],48,replace=True,p=prob[c])for c in(0,1)]);cum+=int(factual[ix].sum())
  if update%100==0:assert history[update//100-1]['cumulative_sampled_factual_available']==cum
 started=read(run/'RUN_STARTED.json');first=read(run/'FIRST_UPDATE.json');assert started['optimizer_updates']==0 and first['update']==1 and first['restart_allowed']is False and started['current_PnP']==first['current_PnP']==0
 baseline=read(run/'BASELINE0_RESULTS.json');assert baseline['selection_eligible']is False
 boundary=static_boundary(p,scope['bindings']['trainer_preparation']['path'])
 report={'schema':'V91_INDEPENDENT_SELECTED_SAVED_AUDIT','completed':True,'PASS':True,'blockers':[],'created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'failure_preserved_addendum':bind(O/'STATIC_GUARD_V2_REPAIR_ADDENDUM.json'),'scope':prep['scope'],'code':bind(__file__),'preparation':bind(O/'PREPARATION_SEAL_V2.json'),'fit_complete':bind(donepath),'model_freeze':done['model_freeze'],'checkpoint_SHA_only':freeze['checkpoint'],'history':done['history'],'scores':done['scores'],'tilemaps':done['tilemaps'],'mosaics':done['mosaics'],'published_results':done['source_results'],'exact_checks':{'all604_identity_label_U_duplicateweights':True,'all12900_inputonly_tileIDs_F32map_shapes_finite':True,'all12900_factual_validity_reconstructedfrominputgeometry':True,'all100native_mosaics_all129tiles_independent_F64_replay':True,'all604completedmosaic_CPUF32score_replay_bitexact':True,'D_E_U_CM_P_R_U_and_pair22_63':True,'all9_fixedgates_exact':True,'earlieststrict_history20_selected_checkpoint':True,'unknown_placeholder_NLL_not_summed':True,'nominal_term_availability_and2000_expected_sampler_cumulative_counts':True},'source_metrics':mr,'source_gate':g,'all_nine_pass':all(g.values()),'history_selection':ha,'final_macroBCE_storedFP32':storedmacro,'final_minus_selected_history_macroBCE':storedmacro-ha['selected_history_macroBCE'],'independent_known_VAL_analyticBCE_FP64':analytic,'NLL_numerical_diagnostic':nll_numeric,'mosaic_numeric_policy':scope['mosaic_replay_numeric_policy'],'mosaic_numeric_actual_all100':numeric,'static_forward_mosaic_GTscoring_boundary':boundary,'sampler_scope':'Expected2000RNGdraws reconstructed from frozen pools/weights/seed and reconciled cumulative factual counts; per-update actual sampledindex arrays notsaved, not an independent observation ofevery index.','execution_limits':'History perstep logits notsaved; classNLL/historyselection values internally reconciled. No newmodel/headforward or checkpointparameterload. Sourceinitialization and optimizer execution evidencedby frozen code/certificates only.','RGB_P3_feature_weightparameter_model_forward_fit_current_PnP':0,'native8_human_semantic_gold_or_goalsolved':False,'current_evaluation_admission':False}
 save(out/'REVIEW.json',report);save(O/'REVIEW.json',report);save(out/'SEAL.json',{'completed':True,'PASS':True,'review':bind(O/'REVIEW.json'),'code':bind(__file__),'preparation':bind(O/'PREPARATION_SEAL_V2.json'),'actualFIT_COMPLETE':bind(donepath),'new_forward_fit_current_PnP':0});print(json.dumps({'review':bind(O/'REVIEW.json'),'seal':bind(out/'SEAL.json'),'gates':g},indent=2))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['generated','actual']);ap.add_argument('--fit-complete-sha');a=ap.parse_args()
 if a.stage=='generated':generated()
 else:
  if not a.fit_complete_sha:ap.error('--fit-complete-sha required')
  actual(a.fit_complete_sha)
