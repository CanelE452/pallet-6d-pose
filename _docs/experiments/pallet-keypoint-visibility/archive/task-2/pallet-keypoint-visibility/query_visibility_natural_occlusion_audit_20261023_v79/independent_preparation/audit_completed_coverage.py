"""Independent saved coverage aggregates + frozen8-row scalar source replay."""
from pathlib import Path
from functools import lru_cache
import argparse,hashlib,json,sys
import numpy as np
from scalar_payload_replay import replay,centers
from independent_grid_math import roi_inverse_float32_phase
R=Path(__file__).resolve().parent;V=R.parent;P=V.parent;C=V/'coverage';OLD=P/'query_visibility_reference_local_support_20261023_v76';V77=P/'query_visibility_controlled_pairs_20261023_v77'
sys.path.insert(0,str(V77/'independent_preparation'))
from audit_completed_source_experiment import pure

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def load(p):return json.loads(Path(p).read_text())
def check(b):assert sha(b['path'])==b['sha256']
def lines(p):return[json.loads(x)for x in Path(p).read_text().splitlines()]
def save(p,d):
 with p.open('x')as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
def equal(a,b):return np.array_equal(a,b,equal_nan=True)if a.dtype.kind in 'fc'else np.array_equal(a,b)
def quant(a):
 a=np.asarray(a,float);a=a[np.isfinite(a)]
 return None if not len(a)else dict(zip(['min','p25','p50','p75','p90','p95','p99','max'],map(float,np.quantile(a,[0,.25,.5,.75,.9,.95,.99,1]))))

def verify_summary(pub,a,selected,weights=None):
 m=np.asarray(selected,bool);w=np.ones(m.sum())if weights is None else np.asarray(weights,float)[m]
 assert pub['query_rows']==m.sum()and pub['query_draw_weight']==float(w.sum())
 flat=a['class_counts'].reshape(-1,2,3,3);valid=a['valid_counts'].reshape(-1,2,7);unique=a['unique_native_pixels'].reshape(-1,2,3)
 for gi,n in enumerate([9,96]):
  pg=pub['grids'][gi];assert pg['N']==n and pg['valid_count_sums']==valid[m,gi].sum(0,dtype=np.int64).tolist()and pg['cached_canvas_but_NN_unknown_positions']==int((valid[m,gi,4]-valid[m,gi,5]).sum())
  for si,s in enumerate(pg['subsets']):
   cnt=flat[m,gi,si];den=cnt.sum(1);wc=(cnt*w[:,None]).sum(0);total=wc.sum()
   assert s['V_C_O_position_counts']==cnt.sum(0,dtype=np.int64).tolist()and s['V_C_O_weighted_counts']==wc.tolist()and s['weighted_position_fraction']==(None if total==0 else(wc/total).tolist())
   assert s['queries_no_valid']==(den==0).sum()and s['queries_with_V']==(cnt[:,0]>0).sum()and s['queries_with_C']==(cnt[:,1]>0).sum()and s['queries_with_O']==(cnt[:,2]>0).sum()and s['queries_mixed_V_C']==((cnt[:,0]>0)&(cnt[:,1]>0)).sum()
   for k,cl in enumerate(['M4_VISIBLE','M0_AND_NOT_M4_COVERED_SUPPORT','NOT_M0_OUTSIDE_SUPPORT']):assert s['per_query_fraction_quantiles'][cl]==quant(np.divide(cnt[:,k],den,out=np.full(len(cnt),np.nan),where=den>0))
   assert s['unique_native_pixel_count_quantiles']==quant(unique[m,gi,si])and s['valid_position_count_quantiles']==quant(den)and s['native_pixel_repetition_factor_quantiles']==quant(np.divide(den,unique[m,gi,si],out=np.full(len(cnt),np.nan),where=unique[m,gi,si]>0))

def main(output):
 output.mkdir(parents=True,exist_ok=True);policy=load(R/'INDEPENDENT_REPLAY_POLICY_BEFORE_OUTPUTS.json');seal=load(C/'results/COVERAGE_SEAL.json');assert seal['completed']and seal['PASS']
 for k in ['protocol','code','math','generated','per_query','summary','sampler']:check(seal[k])
 prot=load(C/'COVERAGE_PROTOCOL.json');assert sha(C/'COVERAGE_PROTOCOL.json')=='db2b3d371c22b3459710e58573f40e52a8204bd1ea8145684e9c5aa876dad4fb'
 with np.load(C/'results/COVERAGE_PER_QUERY.npz',allow_pickle=False)as z:a={k:z[k]for k in z.files}
 assert a['class_counts'].shape==(10002,8,2,3,3)and a['valid_counts'].shape==(10002,8,2,7)and a['unique_native_pixels'].shape==(10002,8,2,3)
 assert np.array_equal(a['class_counts'].sum(-1),a['valid_counts'][...,[3,5,6]])and(a['class_counts']>=0).all()and(a['unique_native_pixels']>=0).all()and(a['unique_native_pixels']<=a['class_counts'].sum(-1)).all()
 gt=[]
 for path in [OLD/'gt_corpus_build/GT_ONLY_TEACHER_CORPUS.npz',V77/'pair_generation/AFTER_GT_ONLY.npz']:
  with np.load(path,allow_pickle=False)as z:gt.append({k:z[k]for k in ['id','source','split','label8','rule8','native_GTref8','convex_self_proxy8_separate']})
 g={k:np.concatenate([x[k]for x in gt])for k in gt[0]}
 for k in ['id','source','split']:assert np.array_equal(a[k],g[k])
 for field,key in [('teacher_label8','label8'),('rule8','rule8'),('self_proxy8_separate','convex_self_proxy8_separate')]:assert np.array_equal(a[field],g[key])
 with np.load(V77/'runs/PAIRED_SUPPORT_V1/SELECTED_SOURCE_SCORES.npz',allow_pickle=False)as z:
  meta={k:z[k]for k in ['primary_loss_mask','input_available','reference_in_local_square_GT_only','domain','after_original_row_index','after_anchor_k','pair_good']}
 for k in ['primary_loss_mask','input_available','reference_in_local_square_GT_only','domain']:assert np.array_equal(a[k],meta[k])
 orig=meta['after_original_row_index'];assert np.array_equal(a['original_row_index'],np.r_[np.arange(5203),orig])
 assert np.array_equal(g['native_GTref8'][5203:],g['native_GTref8'][orig],equal_nan=True)
 off=(g['native_GTref8']-a['query8_xy'])/a['L'][:,None,None];assert equal(a['reference_offset_L_GT_only'],off)and equal(a['q_reference_error_px_GT_only'],np.linalg.norm(g['native_GTref8']-a['query8_xy'],axis=-1))
 for k in ['query8_xy','L','raw_hw','input_hw','gain','offset','query_missing','valid_counts','unique_native_pixels']:assert equal(a[k][5203:],a[k][orig]),k
 before=a['class_counts'][orig];after=a['class_counts'][5203:];assert np.array_equal(before[...,0]+before[...,1],after[...,0]+after[...,1])and np.array_equal(before[...,2],after[...,2])and(after[...,0]<=before[...,0]).all()and(after[...,1]>=before[...,1]).all()
 assert np.array_equal(a['class_counts'][...,0,:],a['class_counts'][...,2,:]) # native is cached intersection at this exact literal phase
 y=a['teacher_label8'].reshape(-1);pm=a['primary_loss_mask'].reshape(-1);src=np.repeat(a['source'],8);spl=np.repeat(a['split'],8);dom=np.repeat(a['domain'],8);rule=a['rule8'].reshape(-1);selfp=a['self_proxy8_separate'].reshape(-1);offset=np.max(abs(off),axis=-1).reshape(-1);err=a['q_reference_error_px_GT_only'].reshape(-1);state=a['q_center_support_state'].reshape(-1)
 def condition(dim,s):
  if dim=='ALL':return np.ones(len(y),bool)
  if dim=='teacher':return y==int(s)
  if dim=='q_center_state':return state==int(s)
  if dim=='branch':return np.char.startswith(rule,'STRONG')if s=='STRONG'else~np.char.startswith(rule,'STRONG')
  if dim=='self_proxy':return selfp==int(s)
  if dim=='actual_reference_square':return~a['reference_in_local_square_GT_only'].reshape(-1)
  if dim=='q_ref_error':return err<=3 if s=='LE3PX'else err>3
  return {'CENTER_LE1_OVER18':offset<=1/18,'LE025_EXCLUDE_CENTER':(offset>1/18)&(offset<=.25),'LE05_EXCLUDE025':(offset>.25)&(offset<=.5),'LINF_GT05':offset>.5,'NONFINITE':~np.isfinite(offset)}[s]
 summary=load(C/'results/COVERAGE_SUMMARY.json')
 for row in summary['strata']:
  m=(dom==row['domain'])&(src==row['source'])&(spl==row['split'])&condition(row['dimension'],row['stratum'])
  if row['population']=='UNCHANGED_PRIMARY':m&=pm
  verify_summary(row['coverage'],a,m)
 # Actual frozen sampler replay; no model/optimizer calls.
 ak=meta['after_anchor_k'];pb=orig*8+ak;pa=(5203+np.arange(4799))*8+ak;anchor=np.zeros(len(y),bool);anchor[pa]=1
 pools=dict(natural_pools={(s,c):np.flatnonzero((dom==0)&(spl=='train')&(src==ss)&(y==c)&pm)for s,ss in enumerate(['SYNTHETIC','YCB_PUBLIC_REAL'])for c in [0,1]},nonanchor_pools={(s,c):np.flatnonzero((dom==1)&~anchor&(spl=='train')&(src==ss)&(y==c)&pm)for s,ss in enumerate(['SYNTHETIC','YCB_PUBLIC_REAL'])for c in [0,1]},pair_train_pools={s:np.flatnonzero(meta['pair_good']&(a['split'][5203:]=='train')&(a['source'][5203:]==ss))for s,ss in enumerate(['SYNTHETIC','YCB_PUBLIC_REAL'])},pair_before_query=pb,pair_after_query=pa)
 draw=pure(V77/'train_paired_once.py','sample');rng=np.random.default_rng(20261023);freq=np.zeros(len(y),np.int64);dh=hashlib.sha256()
 for _ in range(2000):
  ix=draw(pools,rng);assert len(ix)==96 and pm[ix].all()and(spl[ix]=='train').all();np.add.at(freq,ix,1);dh.update(ix.astype('<i8').tobytes())
 assert freq.sum()==192000 and np.array_equal(freq.reshape(10002,8),a['sampler_replay_query_draw_count'])
 exposures=load(C/'results/SAMPLER_EXPOSURE_SUMMARY.json');poolmap={}
 for name in ['natural_pools','nonanchor_pools']:
  for(s,c),ix in pools[name].items():poolmap[f'{name}|{s}|{c}']=ix
 for s,ix in pools['pair_train_pools'].items():poolmap[f'pair_before|{s}']=pb[ix];poolmap[f'pair_after|{s}']=pa[ix]
 for row in exposures['pools']:
  ix=poolmap[row['pool']];m=np.zeros(len(y),bool);m[ix]=1;assert row['rows']==len(ix)and row['per_batch_draws']==8;verify_summary(row['coverage'],a,m)
 verify_summary(exposures['frequency_weighted_all_TRAIN'],a,freq>0,freq)
 # Fixed representative mask payload replay, chosen before outputs by metadata SHA.
 original_masks=lines(OLD/'gt_corpus_build/PER_ROW_TEACHER_GT_ONLY.jsonl');after_masks=lines(V77/'pair_generation/PAIR_LINEAGE_GT_ONLY.jsonl');scalar=[]
 for chosen in policy['selection']['rows']:
  i=chosen['frame_index'];bank=next(b for b in prot['banks']if b['start']<=i<b['end']);j=i-bank['start'];man=load(bank['manifest']['path']);ch=next(x for x in man['chunks']if x['start']<=j<x['end']);check(ch['file'])
  with np.load(ch['file']['path'],allow_pickle=False)as z:q=z['query8_xy'][j-ch['start']];L=float(z['L'][j-ch['start']]);cv9=z['p3_roi'][j-ch['start'],:,64].astype(bool);cv96=z['crop_valid96'][j-ch['start'],:,0]
  assert equal(q,a['query8_xy'][i])and L==a['L'][i]
  from PIL import Image
  if i<5203:pair=original_masks[i]['mask_pair'];b0=pair['M0'];b4=pair['M4'];threshold=pair['threshold_strict_gt']
  else:pair=after_masks[i-5203];b0=pair['M0_unchanged'];b4=pair['M4_after'];threshold=127 if a['source'][i]=='SYNTHETIC'else 0
  check(b0);check(b4);f=np.asarray(Image.open(b0['path']))>threshold;v=np.asarray(Image.open(b4['path']))>threshold;assert f.shape==v.shape==tuple(a['raw_hw'][i])and not(v&~f).any()
  center=centers(f,v,q)
  for k,vv in center.items():assert np.array_equal(vv,a[k][i]),(i,k)
  for gi,(n,cache)in enumerate([(9,cv9),(96,cv96)]):
   rr,grid,nv,maps=replay(f,v,q,L,n,cache)
   for k,vv in rr.items():assert np.array_equal(vv,a[k][i,:,gi]),(i,n,k)
   low=0 if n==9 else-.5;h,w=f.shape;upperx=w if n==9 else w-.5;uppery=h if n==9 else h-.5;good=np.isfinite(q).all(1)&~(q==-1).all(1);expected=(grid[...,0]>=low)&(grid[...,0]<upperx)&(grid[...,1]>=low)&(grid[...,1]<uppery)&good[:,None,None];assert np.array_equal(cache,expected)
   if n==9:
    inv,bound=roi_inverse_float32_phase(grid,a['gain'][i],a['offset'][i]);assert np.all(abs(inv-grid)<=bound)
    actualmax=np.max(abs(inv-grid),axis=(1,2,3));assert np.allclose(actualmax,a['ROI_float32_norm_max_native_abs_error'][i],rtol=0,atol=1e-12)
    oldpix=np.floor(grid+.5).astype(np.int64);newpix=np.floor(inv+.5).astype(np.int64);inside=(newpix[...,0]>=0)&(newpix[...,0]<w)&(newpix[...,1]>=0)&(newpix[...,1]<h)&nv
    pixelchanges=((oldpix!=newpix).any(-1)&nv).sum((1,2));oldstate=np.stack([ff.astype(np.uint8)+vv.astype(np.uint8)for ff,vv,_ in maps]);newstate=np.zeros(nv.shape,np.uint8);newstate[inside]=f[newpix[...,1][inside],newpix[...,0][inside]].astype(np.uint8)+v[newpix[...,1][inside],newpix[...,0][inside]].astype(np.uint8)
    assert np.array_equal(pixelchanges,a['ROI_float32_norm_NN_pixel_changes'][i])and np.array_equal(((oldstate!=newstate)&inside).sum((1,2)),a['ROI_float32_norm_mask_class_changes'][i])and np.array_equal((nv&~inside).sum((1,2)),a['ROI_float32_norm_recovered_outside'][i])
  scalar.append(dict(frame_index=i,id=chosen['id'],source=chosen['source'],split=chosen['split'],domain=chosen['domain'],all8queries_9_and96_count_unique_fingerprint_qcenter_exact=True,actual_cached_canvas_and_literal_formula_exact=True,FP32_inverse_phase_within_derived_bound=True))
 report=dict(PASS=True,code=bind(Path(__file__)),frozen_policy=bind(R/'INDEPENDENT_REPLAY_POLICY_BEFORE_OUTPUTS.json'),scalar_math=bind(R/'scalar_payload_replay.py'),coverage_seal=bind(C/'results/COVERAGE_SEAL.json'),frames=10002,queries=80016,original_GT_and_primary_all_exact=True,all4799_pairs_q_L_affine_validity_fullsupport_unchanged=True,M4_visible_count_monotonic_all_pairs=True,summary_strata_recomputed=len(summary['strata']),all_class_counts_denominators_quantiles_and_12pool_weighted_exposure_exact=True,sampler_batches_replayed=2000,sampler_draws=192000,sampler_draws_SHA256=dh.hexdigest(),fixed_scalar_rows=scalar,scalar_cells_in_both_grids=8*8*(9*9+96*96),source_native_mask_bodies_decoded=16,cached_P3_used_only_for_validity_channel64=True,RGB_model_newfit_current_PnP=0,semantics='V/C/O are mask components. O not externalE/selfS; original reference-local labels/U unchanged. NN/float32 phase audits do not certify physical RFcenters or RGB semantic gold.')
 save(output/'COVERAGE_AGGREGATE_AND_SCALAR_AUDIT.json',report);print(json.dumps(dict(PASS=True,report=bind(output/'COVERAGE_AGGREGATE_AND_SCALAR_AUDIT.json'))))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args();main(a.output)
