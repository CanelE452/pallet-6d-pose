"""All unchanged source crop mask coverage. No RGB/model/trainer execution."""
from pathlib import Path
from collections import Counter,OrderedDict
import argparse,ast,hashlib,json,time
import numpy as np
from PIL import Image
from coverage_math import coordinates,cached_formula,count_grid,roi_float_phase,query_center,VALID_NAMES,CLASS_NAMES,SUBSET_NAMES

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;P=ROOT.parent
V76=P/'query_visibility_reference_local_support_20261023_v76';V77=P/'query_visibility_controlled_pairs_20261023_v77';V78=P/'query_visibility_dino_query_support_20261023_v78'
ENTRY_SHA='bdefc2af54e4e9e38eb1dc312116ad230421a9e47eadfa4810b85b5b2b3987b6'
OUT=HERE/'results'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert sha(b['path'])==b['sha256'],b['path']
def load(p):return json.loads(Path(p).read_text())
def rows(p):return[json.loads(x)for x in Path(p).read_text().splitlines()]
def write(p,d):
 with open(p,'x')as f:f.write(json.dumps(d,indent=2,allow_nan=False)+'\n')
def keys(p,names):
 with np.load(p,allow_pickle=False)as z:return{k:z[k]for k in names}
def setup_banks():
 b=load(V77/'COMPLETED_SOURCE_INPUT_BINDINGS.json');o=b['bindings']['original_input_bindings'];check(o);old=load(o['path']);banks=[];start=0
 for pack in old['packs']:
  for k in ('manifest','lineage','seal'):check(pack[k])
  m=load(pack['manifest']['path']);banks.append(dict(name='ORIGINAL_'+pack['source'],domain=0,start=start,end=start+m['frame_count'],**{k:pack[k]for k in ('manifest','lineage','seal')}));start+=m['frame_count']
 pack={k:b['bindings']['after_input_'+k]for k in ('manifest','lineage','seal')}
 for v in pack.values():check(v)
 m=load(pack['manifest']['path']);banks.append(dict(name='CONTROLLED_AFTER',domain=1,start=start,end=start+m['frame_count'],**pack));assert banks[-1]['end']==10002
 return banks
def freeze():
 assert sha(ROOT/'AUDIT_ENTRY.json')==ENTRY_SHA;e=load(ROOT/'AUDIT_ENTRY.json');check(e['decisions']);assert load(HERE/'GENERATED_COVERAGE_TESTS.json')['PASS']
 assert load(HERE/'GENERATED_COVERAGE_TESTS.json')['coverage_math_sha256']==sha(HERE/'coverage_math.py')
 bs=setup_banks();paths=[ROOT/'AUDIT_ENTRY.json',Path(e['decisions']['path']),HERE/'coverage_math.py',HERE/'test_generated_coverage.py',HERE/'GENERATED_COVERAGE_TESTS.json',Path(__file__),
 V76/'gt_corpus_build/PER_ROW_TEACHER_GT_ONLY.jsonl',V76/'gt_corpus_build/GT_ONLY_TEACHER_CORPUS.npz',V77/'pair_generation/PAIR_LINEAGE_GT_ONLY.jsonl',V77/'pair_generation/AFTER_GT_ONLY.npz',V77/'pair_generation/SEAL.json',V77/'runs/PAIRED_SUPPORT_V1/SELECTED_SOURCE_SCORES.npz',V77/'train_paired_once.py',
 V76/'source_inventory/UNCHANGED_MODEL_INPUT_LINKS.jsonl',V76/'public_input_preparation/IMAGES_AND_EXISTING_P3_INPUT_ONLY.jsonl',
 P.parent/'external_ray_proxy_head_20261022_v60/head_and_roi.py',P/'query_visibility_loco_native_context_20261023_v74/native_context_cache/compile_input_pack.py',V76/'synth1003_input_pack_code/build_synth1003_input_pack.py',V77/'after_input_preparation/build_after_input_pack.py',V77/'after_input_preparation/BEFORE_GUARDS_PASS.json']
 bindings=[bind(p)for p in paths]
 for b in bs:
  bindings.extend(b[k]for k in ('manifest','lineage','seal'));bindings.extend(c['file']for c in load(b['manifest']['path'])['chunks'])
 protocol=dict(frozen=True,scope='All original5203+after4799 source mask coverage only. No input/label/admission/threshold/model changes.',bindings=bindings,banks=bs,
  shape=[10002,8],grid_axis=[9,96],valid_count_axis=VALID_NAMES,class_axis=CLASS_NAMES,class_subset_axis=SUBSET_NAMES,
  mask_rules='SYNTH M0/M4 raw>127; PUBLIC raw>0. afterM4 sealed PNG same source threshold. Assert M4subsetM0. Outside-native/NN invalid ignored, not C or O.',
  coordinate_rules='Exact old native_grid: q+(((j+.5)/9)-.5)*L. RGB96 exact q+(((j+.5)/96)-.5)*L; masks nearestfloor+.5. CachedROI9 nativecontinuous; cachedRGB96[-.5,W-.5), both actualcompiler goodfinite andnot(-1,-1).',
  sampling_validity='native=good&continuous&NNbounds; cached_known=cachedcanvas&NNbounds (negative-half RGB can differ); intersection=native&cached. Cached-valid NN-unknown is reported separately and excluded from class counts.',
  missing='Preserve all rawq/missing/primary. Purecrop finite-sentinel behavior is not the actualcompiler: compile_frame skips sentinel. No arbitrary query movement or missing relabel.',
  q_center='q_center_mask_valid requiresgood&continuous&NNbounds; M0/M4=-1 invalid. support_state=-1INVALID,0Cmask,1Vmask,2O. raw3x3aroundroundq hasperpixelNNbounds+querygood, preservingneighbors evenwhenquerycontinuousinvalid. Diagnostic only, no newprimary labels.',
  alignment='All ROI channel65/RGB96 cached-valid maps checked against exactoldliteral. All source affine/gain/offset rederived. P3 norm float32 inverse phase/ties measured, not physical RFcenter gold. RGBbilinear vsmaskNN differs at boundaries; no photometric segmentation-gold claim.',
  fingerprints=dict(names=['gt_map_native_sha256','gt_map_intersection_sha256'],shape=[10002,8,2],dtype='S64',header='V79_MAP_V1|grid{N}',payload='np.packbits(little,Corder [valid,M0&valid,M4&valid])',validity=['native','intersection']),
  forward_collisions='Root owns actual forward-input fingerprints/collision grouping separately. This audit exposes only compact GT spatial fingerprints, never used as forward inputs.',
  sampler='Exact V77 pure sample AST, seed20261023 replay2000×96 without model/optimizer; report12component pools and effective frequency-weighted mask class prevalence. Does not create dense training targets or newweights.',
  outputs=['COVERAGE_PER_QUERY.npz','COVERAGE_SUMMARY.json','SAMPLER_EXPOSURE_SUMMARY.json','COVERAGE_SEAL.json'],RGB_model_current_forward_fit_PnP=0)
 write(HERE/'COVERAGE_PROTOCOL.json',protocol);print(json.dumps(bind(HERE/'COVERAGE_PROTOCOL.json')))

def quant(x):
 a=np.asarray(x,float);a=a[np.isfinite(a)]
 return None if not len(a)else dict(zip(['min','p25','p50','p75','p90','p95','p99','max'],map(float,np.quantile(a,[0,.25,.5,.75,.9,.95,.99,1]))))
def summary(a,selected,weights=None):
 flat=a['class_counts'].reshape(-1,2,3,3);vc=a['valid_counts'].reshape(-1,2,7);unique=a['unique_native_pixels'].reshape(-1,2,3)
 sel=np.asarray(selected,bool);w=np.ones(sel.sum(),np.float64)if weights is None else np.asarray(weights,np.float64)[sel]
 out=dict(query_rows=int(sel.sum()),query_draw_weight=float(w.sum()),grids=[])
 for gi,n in enumerate((9,96)):
  sets=[]
  for si,name in enumerate(SUBSET_NAMES):
   c=flat[sel,gi,si];den=c.sum(1);weighted=(c*w[:,None]).sum(0);total=weighted.sum()
   sets.append(dict(validity=name,V_C_O_position_counts=c.sum(0,dtype=np.int64).tolist(),V_C_O_weighted_counts=weighted.tolist(),weighted_position_fraction=None if total==0 else(weighted/total).tolist(),
    queries_no_valid=int((den==0).sum()),queries_with_V=int((c[:,0]>0).sum()),queries_with_C=int((c[:,1]>0).sum()),queries_with_O=int((c[:,2]>0).sum()),queries_mixed_V_C=int(((c[:,0]>0)&(c[:,1]>0)).sum()),
    per_query_fraction_quantiles={cl:quant(np.divide(c[:,j],den,out=np.full(len(c),np.nan),where=den>0))for j,cl in enumerate(CLASS_NAMES)},unique_native_pixel_count_quantiles=quant(unique[sel,gi,si]),
    valid_position_count_quantiles=quant(den),native_pixel_repetition_factor_quantiles=quant(np.divide(den,unique[sel,gi,si],out=np.full(len(c),np.nan),where=unique[sel,gi,si]>0))))
  out['grids'].append(dict(N=n,valid_count_sums=vc[sel,gi].sum(0,dtype=np.int64).tolist(),cached_canvas_but_NN_unknown_positions=int((vc[sel,gi,4]-vc[sel,gi,5]).sum()),subsets=sets))
 return out

def run():
 prot=load(HERE/'COVERAGE_PROTOCOL.json');assert prot['frozen']and not OUT.exists()
 for b in prot['bindings']:check(b)
 OUT.mkdir();start=time.monotonic()
 base=rows(V76/'gt_corpus_build/PER_ROW_TEACHER_GT_ONLY.jsonl');after=rows(V77/'pair_generation/PAIR_LINEAGE_GT_ONLY.jsonl');assert len(base)==5203 and len(after)==4799
 gt0=keys(V76/'gt_corpus_build/GT_ONLY_TEACHER_CORPUS.npz',['id','source','split','label8','rule8','native_GTref8','convex_self_proxy8_separate'])
 gt1=keys(V77/'pair_generation/AFTER_GT_ONLY.npz',['id','source','split','label8','rule8','native_GTref8','convex_self_proxy8_separate'])
 # Standardize only metadata key spelling, never original teacher values.
 for b in (gt0,gt1):
  if 'convex_self_proxy8_separate'not in b:raise AssertionError('selfproxy metadata missing')
 g={k:np.concatenate([gt0[k],gt1[k]])for k in gt0}
 meta=keys(V77/'runs/PAIRED_SUPPORT_V1/SELECTED_SOURCE_SCORES.npz',['id','source','split','primary_loss_mask','input_available','reference_in_local_square_GT_only','domain','after_original_row_index','after_anchor_k','pair_good'])
 for k in ('id','source','split'):assert np.array_equal(meta[k],g[k]),k
 syn=rows(V76/'source_inventory/UNCHANGED_MODEL_INPUT_LINKS.jsonl');pubinv={r['P3_cache']['path']:r for r in rows(V76/'public_input_preparation/IMAGES_AND_EXISTING_P3_INPUT_ONLY.jsonl')}
 a=dict(class_counts=np.zeros((10002,8,2,3,3),np.int32),valid_counts=np.zeros((10002,8,2,7),np.int32),unique_native_pixels=np.zeros((10002,8,2,3),np.int32),
  gt_map_native_sha256=np.empty((10002,8,2),dtype='S64'),gt_map_intersection_sha256=np.empty((10002,8,2),dtype='S64'),near_NN_half_tie_counts=np.zeros((10002,8,2),np.int32),literal_vs_algebra_NN_changes=np.zeros((10002,8,2),np.int32),
  ROI_float32_norm_max_native_abs_error=np.zeros((10002,8),np.float64),ROI_float32_norm_NN_pixel_changes=np.zeros((10002,8),np.int32),ROI_float32_norm_mask_class_changes=np.zeros((10002,8),np.int32),ROI_float32_norm_recovered_outside=np.zeros((10002,8),np.int32),query8_xy=np.zeros((10002,8,2),np.float64),L=np.zeros(10002,np.float64),raw_hw=np.zeros((10002,2),np.int32),input_hw=np.zeros((10002,2),np.int32),gain=np.zeros(10002,np.float64),offset=np.zeros((10002,2),np.float64),query_missing=np.zeros((10002,8),bool))
 mask_cache=OrderedDict();mask_files=0;checked=0;allnative=0;cache_mismatch9=cache_mismatch96=0
 def mask(b,threshold):
  nonlocal mask_files
  key=(b['path'],b['sha256'],threshold)
  if key not in mask_cache:
   check(b);v=np.array(Image.open(b['path']));assert v.ndim==2;mask_cache[key]=v>threshold;mask_files+=1
   if len(mask_cache)>40:mask_cache.popitem(last=False)
  else:mask_cache.move_to_end(key)
  return mask_cache[key]
 for bank in prot['banks']:
  man=load(bank['manifest']['path']);line=rows(bank['lineage']['path'])
  for chunk in man['chunks']:
   check(chunk['file'])
   with np.load(chunk['file']['path'],allow_pickle=False)as z:
    q=z['query8_xy'];ls=z['L'];cached9=z['p3_roi'][:,:,64].astype(bool);cached96=z['crop_valid96'][:,:,0];qmissing=z['query_missing']
   for j in range(len(q)):
    i=bank['start']+chunk['start']+j;assert i==checked;rec=line[chunk['start']+j];assert rec['id']==g['id'][i]
    if i<5203:
     pair=base[i]['mask_pair'];M0,M4=pair['M0'],pair['M4'];threshold=pair['threshold_strict_gt']
     if i<1003:aff=syn[i];HW=aff['raw_RGB']['raw_hw'];ih=aff['backbone_input_shape'];gain=aff['gain'];off=np.asarray(aff['offset'],float)
     else:aff=pubinv[rec['P3_cache']['path']];HW=aff['raw_hw'];ih=aff['input_shape'];gain=aff['gain'];off=np.asarray(aff['offset'],float)
    else:
    pair=after[i-5203];M0,M4=pair['M0_unchanged'],pair['M4_after'];threshold=127 if g['source'][i]=='SYNTHETIC' else 0
     original=int(meta['after_original_row_index'][i-5203]);HW=a['raw_hw'][original].tolist();ih=rec['input_HW'];gain=rec['gain'];off=np.asarray(rec['offset'],float)
     assert np.array_equal(q[j],a['query8_xy'][original])and ls[j]==a['L'][original]and np.array_equal(ih,a['input_hw'][original])and gain==a['gain'][original]and np.array_equal(off,a['offset'][original])
    m0=mask(M0,threshold);m4=mask(M4,threshold);assert list(m0.shape)==list(HW)and m0.shape==m4.shape and not(m4&~m0).any()
    allnative+=m0.size;h,w=HW;canvas=np.asarray([h+200,w+200]);derived=float(np.min(np.asarray(ih)/canvas));resized=np.rint(canvas*derived);offset=np.asarray([round((ih[1]-resized[1])/2-.1),round((ih[0]-resized[0])/2-.1)],float)
    assert gain==derived and np.array_equal(off,offset)
    good=np.isfinite(q[j]).all(1)&~(q[j]==-1).all(1);assert np.array_equal(qmissing[j],~good)
    finite=q[j][good];expectedL=float(np.clip(.25*np.max(np.ptp(finite,axis=0)),16,max(HW)))if len(finite)>=2 else 16.
    assert expectedL==float(ls[j])
    a['query8_xy'][i]=q[j];a['L'][i]=ls[j];a['raw_hw'][i]=HW;a['input_hw'][i]=ih;a['gain'][i]=gain;a['offset'][i]=off;a['query_missing'][i]=qmissing[j]
    for k,v in query_center(m0,m4,q[j]).items():
     if k not in a:a[k]=np.zeros((10002,)+v.shape,dtype=v.dtype)
     a[k][i]=v
    for gi,n in enumerate((9,96)):
     cached=cached9[j]if n==9 else cached96[j]
     grid=coordinates(q[j],float(ls[j]),n);expected=cached_formula(grid,HW,n,q[j]);mis=int((cached!=expected).sum())
     if n==9:cache_mismatch9+=mis
     else:cache_mismatch96+=mis
     assert mis==0,(i,n,mis)
     result,grid,pix,nv=count_grid(m0,m4,q[j],float(ls[j]),n,cached)
     for k,v in result.items():a[k][i,:,gi]=v
     if n==9:
      phase,_=roi_float_phase(grid,gain,off,HW,nv,m0,m4)
      for k,v in phase.items():a[k][i]=v
    checked+=1
   print(json.dumps(dict(frames=checked,total=10002,mask_decodes=mask_files,seconds=time.monotonic()-start)),flush=True)
 assert checked==10002 and cache_mismatch9==cache_mismatch96==0
 # Preserve original teacher/admission and expose the existing sampler, not new labels.
 a.update(id=g['id'],source=g['source'],split=g['split'],domain=meta['domain'],original_row_index=np.r_[np.arange(5203),meta['after_original_row_index']],teacher_label8=g['label8'],rule8=g['rule8'],self_proxy8_separate=g['convex_self_proxy8_separate'],primary_loss_mask=meta['primary_loss_mask'],input_available=meta['input_available'],reference_in_local_square_GT_only=meta['reference_in_local_square_GT_only'],reference_offset_L_GT_only=(g['native_GTref8']-a['query8_xy'])/a['L'][:,None,None],q_reference_error_px_GT_only=np.linalg.norm(g['native_GTref8']-a['query8_xy'],axis=-1))
 y=a['teacher_label8'].reshape(-1);primary=a['primary_loss_mask'].reshape(-1);src=np.repeat(a['source'],8);split=np.repeat(a['split'],8);domain=np.repeat(a['domain'],8);selfp=a['self_proxy8_separate'].reshape(-1);rule=a['rule8'].reshape(-1);offset=np.max(abs(a['reference_offset_L_GT_only']),axis=-1).reshape(-1);error=a['q_reference_error_px_GT_only'].reshape(-1)
 strata=[]
 conditions=[('ALL','ALL',np.ones(len(y),bool))]+[('teacher',str(c),y==c)for c in(-1,0,1)]+[('q_center_state',str(c),a['q_center_support_state'].reshape(-1)==c)for c in(-1,0,1,2)]+[('branch','STRONG',np.char.startswith(rule,'STRONG')),('branch','FALLBACK_OR_UNKNOWN',~np.char.startswith(rule,'STRONG'))]+[('self_proxy',str(c),selfp==c)for c in(-1,0,1)]+[('offset_Linf','CENTER_LE1_OVER18',offset<=1/18),('offset_Linf','LE025_EXCLUDE_CENTER',(offset>1/18)&(offset<=.25)),('offset_Linf','LE05_EXCLUDE025',(offset>.25)&(offset<=.5)),('offset_Linf','GT_REFERENCE_OUTSIDE_SQUARE',offset>.5),('offset_Linf','NONFINITE',~np.isfinite(offset)),('q_ref_error','LE3PX',error<=3),('q_ref_error','GT3PX',error>3)]
 for dom in(0,1):
  for source in('SYNTHETIC','YCB_PUBLIC_REAL'):
   for part in('train','val'):
    base=(domain==dom)&(src==source)&(split==part)
    for population,pop in [('ALL_QUERIES',np.ones(len(y),bool)),('UNCHANGED_PRIMARY',primary)]:
     for dim,name,condition in conditions:
      m=base&pop&condition
      strata.append(dict(domain=dom,source=source,split=part,population=population,dimension=dim,stratum=name,coverage=summary(a,m)))
 is_anchor=np.zeros((10002,8),bool);af=np.arange(4799);ak=meta['after_anchor_k'];is_anchor[5203+af,ak]=True;is_anchor=is_anchor.reshape(-1)
 pb=meta['after_original_row_index']*8+ak;pa=(5203+af)*8+ak;after_src=a['source'][5203:];after_train=a['split'][5203:]=='train'
 samplemeta=dict(natural_pools={(s,c):np.flatnonzero((domain==0)&(split=='train')&(src==source)&(y==c)&primary)for s,source in enumerate(('SYNTHETIC','YCB_PUBLIC_REAL'))for c in(0,1)},nonanchor_pools={(s,c):np.flatnonzero((domain==1)&~is_anchor&(split=='train')&(src==source)&(y==c)&primary)for s,source in enumerate(('SYNTHETIC','YCB_PUBLIC_REAL'))for c in(0,1)},pair_train_pools={s:np.flatnonzero(meta['pair_good']&after_train&(after_src==source))for s,source in enumerate(('SYNTHETIC','YCB_PUBLIC_REAL'))},pair_before_query=pb,pair_after_query=pa)
 pools=[]
 for name in('natural_pools','nonanchor_pools'):
  for(s,c),ix in samplemeta[name].items():pools.append((f'{name}|{s}|{c}',ix))
 for s,ix in samplemeta['pair_train_pools'].items():pools.extend([(f'pair_before|{s}',pb[ix]),(f'pair_after|{s}',pa[ix])])
 assert len(pools)==12 and all(len(ix)for _,ix in pools)
 node=next(n for n in ast.parse((V77/'train_paired_once.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='sample');env={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),'frozenV77sampleAST','exec'),env)
 rng=np.random.default_rng(20261023);frequency=np.zeros(len(y),np.int64)
 for _ in range(2000):
  ix=env['sample'](samplemeta,rng);assert len(ix)==96 and primary[ix].all()and(split[ix]=='train').all();np.add.at(frequency,ix,1)
 assert frequency.sum()==192000
 a['sampler_replay_query_draw_count']=frequency.reshape(10002,8).astype(np.int32)
 np.savez_compressed(OUT/'COVERAGE_PER_QUERY.npz',**a)
 exposure=[]
 for name,ix in pools:
  selected=np.zeros(len(y),bool);selected[ix]=True
  exposure.append(dict(pool=name,rows=len(ix),per_batch_draws=8,expected_batch_fraction=1/12,coverage=summary(a,selected)))
 write(OUT/'SAMPLER_EXPOSURE_SUMMARY.json',dict(completed=True,seed=20261023,updates_replayed_without_model=2000,batch_size=96,total_query_draws=192000,pool_count=12,pools=exposure,frequency_weighted_all_TRAIN=summary(a,frequency>0,frequency),scope='Audit of current sparse-query sampler exposure only; not newdense supervision, lossweights orlabel filters.',forward_fit_PnP=0))
 write(OUT/'COVERAGE_SUMMARY.json',dict(completed=True,PASS=True,frames=10002,queries=80016,all_original_and_after_kept=True,grid_axis=[9,96],valid_count_axis=VALID_NAMES,class_axis=CLASS_NAMES,class_subset_axis=SUBSET_NAMES,
  cachedROI9_valid_formula_mismatches=cache_mismatch9,cachedRGB96_valid_formula_mismatches=cache_mismatch96,all_source_affines_rederived_exact=True,all_L_rederived_exact=True,
  ROI_float32_normalized_grid_max_native_abs_phase=float(a['ROI_float32_norm_max_native_abs_error'].max()),ROI_NN_pixel_changes_from_float32_norm=int(a['ROI_float32_norm_NN_pixel_changes'].sum()),ROI_mask_class_changes_from_float32_norm=int(a['ROI_float32_norm_mask_class_changes'].sum()),literal_vs_algebra_NN_pixel_changes_by_grid=a['literal_vs_algebra_NN_changes'].sum((0,1)).tolist(),NN_half_ties_le1e6_by_grid=a['near_NN_half_tie_counts'].sum((0,1)).tolist(),
  mask_decodes=mask_files,native_mask_pair_pixels_checked=allnative,M4_subset_M0_all=True,strata=strata,
  q_labels_primary_threshold_inputs_unchanged=True,GT_fingerprints='fixedS64 headers/packbits exactly COVERAGE_PROTOCOL; rootinput-collision audit separate',RGB96_repeated_native_samples_not_independent_observations=True,
  alignment_limit='Nominal mask NN maps, cached-valid phase and P3 normalized-grid arithmetic verified. Not physical P3 receptive-field pixel centers or RGBbilinear semantic segmentation gold.',RGB_model_current_forward_fit_PnP=0,seconds=time.monotonic()-start))
 write(OUT/'COVERAGE_SEAL.json',dict(completed=True,PASS=True,protocol=bind(HERE/'COVERAGE_PROTOCOL.json'),code=bind(Path(__file__)),math=bind(HERE/'coverage_math.py'),generated=bind(HERE/'GENERATED_COVERAGE_TESTS.json'),per_query=bind(OUT/'COVERAGE_PER_QUERY.npz'),summary=bind(OUT/'COVERAGE_SUMMARY.json'),sampler=bind(OUT/'SAMPLER_EXPOSURE_SUMMARY.json'),new_RGB_model_current_forward_fit_PnP=0))
 print(json.dumps(bind(OUT/'COVERAGE_SEAL.json')),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','run']);a=p.parse_args();dict(freeze=freeze,run=run)[a.stage]()
