"""Existing source TRAIN only. No VAL/held/private/pixel/model/weight reads or forwards."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,itertools,json
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
V181=P/'query_visibility_local_support_raw_head_20261024_v181';V180=P/'query_visibility_frozen_sam_source_baseline_20261024_v180';V173=P/'query_visibility_full_source_query_causal_census_20261024_v173'
SCOPE=V/'ROOT_SCOPE_BEFORE_NEW_COMMON_QUERY_OR_PUBLIC_PIXELS.json'
ARMS=['C50_RAW96_DUAL_SUPPORT_OWNER_AUX','C51_RAW96_PLUS_FROZEN_SAM_DUAL_SUPPORT_OWNER_AUX']
def b(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(s) for s in p.read_text().splitlines() if s]
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def counts(a):return {str(k):int(v) for k,v in Counter(a.tolist()).items()}
def quantile(a):return np.quantile(a,[0,.1,.25,.5,.75,.9,1]).tolist() if len(a) else None
def main():
 assert b(SCOPE)['sha256']=='142ba2fd62dfa16c79bda690bf90814d0a9cdd330898722b8f896edf76b396a1'
 manifest=V180/'ALL4096_TRAIN_INPUTS_METADATA_ONLY.jsonl';rows=lines(manifest)
 ownerpath=V173/'literal_raster_census_v1/TRAIN4096_LITERAL_RASTER_SOFT_SIDECAR.npz'
 targetpath=V181/'source_raw96_cache_v1/SYNTH_TRAIN4096_TARGETS.npz';endpath=V180/'source_end_supervision_train4096_v1/TRAIN4096_END_SUPPORT_ONLY.npz'
 with np.load(ownerpath,allow_pickle=False) as z:a={k:z[k].copy() for k in ['original_global_index','scene_index','base_index','native_slot','query_xy','L','split','query_available','scene_owner_point','scene_owner_central4','scene_owner_central32','variant','material','aspect','geometry_family','foreign_family','original_semantic_state']}
 with np.load(targetpath,allow_pickle=False) as z:t={k:z[k].copy() for k in z.files}
 with np.load(endpath,allow_pickle=False) as z:mass=z['jointmass7'].copy()
 assert len(rows)==4096 and (a['split']=='TRAIN').all() and np.array_equal(a['original_global_index'],np.arange(4096))
 av=a['query_available'];vt=t['support_target'][:,0];et=t['support_target'][:,1]
 for i,r in enumerate(rows):assert r['original_global_index']==i and r['original_query_xy']==(None if np.isnan(a['query_xy'][i]).all() else a['query_xy'][i].tolist()) and r['original_L']==(None if np.isnan(a['L'][i]) else a['L'][i]) and r['original_input_flags']['query_available']==bool(av[i])
 pairrows=[];samegroups=0;validsamegroups=0
 for base in range(128):
  for k in range(8):
   ids=np.flatnonzero((a['base_index']==base)&(a['native_slot']==k));assert len(ids)==4;i0=next(int(i) for i in ids if a['variant'][i]=='BEFORE')
   sameall=all(np.array_equal(a['query_xy'][i],a['query_xy'][i0],equal_nan=True) and (a['L'][i]==a['L'][i0] or np.isnan(a['L'][i]) and np.isnan(a['L'][i0])) for i in ids);samegroups+=sameall;validsamegroups+=sameall and av[ids].all()
   for i in ids:
    if i==i0:continue
    finite=np.isfinite(a['query_xy'][i]).all() and np.isfinite(a['query_xy'][i0]).all();shift=float(np.linalg.norm(a['query_xy'][i]-a['query_xy'][i0])) if finite else None
    pairrows.append(dict(base_index=base,native_slot=k,BEFORE_global_index=i0,variant_global_index=int(i),variant=str(a['variant'][i]),both_available=bool(av[i]&av[i0]),q_equal_including_null=bool(np.array_equal(a['query_xy'][i],a['query_xy'][i0],equal_nan=True)),L_equal_including_null=bool(a['L'][i]==a['L'][i0] or np.isnan(a['L'][i]) and np.isnan(a['L'][i0])),source_pixel_q_shift=shift,BEFORE96_sampleunit_q_shift=shift*96/float(a['L'][i0]) if finite and np.isfinite(a['L'][i0]) and a['L'][i0]>0 else None,L_ratio=float(a['L'][i]/a['L'][i0]) if np.isfinite(a['L'][i]) and np.isfinite(a['L'][i0]) and a['L'][i0]>0 else None))
 pf=H/'ALL3072_BEFORE_VARIANT_ORIGINAL_QUERY_COMPARISONS.jsonl'
 with pf.open('x') as f:
  for r in pairrows:f.write(json.dumps(r,allow_nan=False)+'\n')
 pairsummary={v:dict(N=sum(r['variant']==v for r in pairrows),both_available=sum(r['variant']==v and r['both_available'] for r in pairrows),exact_qL_same_valid=sum(r['variant']==v and r['both_available'] and r['q_equal_including_null'] and r['L_equal_including_null'] for r in pairrows),source_shift_valid_quantiles=quantile([r['source_pixel_q_shift'] for r in pairrows if r['variant']==v and r['both_available']]),crop96_shift_valid_quantiles=quantile([r['BEFORE96_sampleunit_q_shift'] for r in pairrows if r['variant']==v and r['both_available']]),L_ratio_valid_quantiles=quantile([r['L_ratio'] for r in pairrows if r['variant']==v and r['both_available']])) for v in ['COVER','NEAR_SHAM','FAR_SHAM']}
 coverage={}
 for field in ['variant','native_slot','material','aspect','geometry_family','foreign_family','original_semantic_state']:
  coverage[field]={}
  for val in sorted(set(a[field].tolist()),key=str):
   ix=a[field]==val;coverage[field][str(val)]=dict(N=int(ix.sum()),original_inputU=int((ix&~av).sum()),visible={str(y):int((ix&(vt==y)).sum()) for y in [-1,0,1]},foreign={str(y):int((ix&(et==y)).sum()) for y in [-1,0,1]})
 fullcross={}
 for g,m,asp in sorted(set(zip(a['geometry_family'],a['material'],a['aspect']))):
  ix=(a['geometry_family']==g)&(a['material']==m)&(a['aspect']==asp);fullcross[g+'|'+m+'|'+asp]=dict(N=int(ix.sum()),visible={str(y):int((ix&(vt==y)).sum()) for y in [-1,0,1]},foreign={str(y):int((ix&(et==y)).sum()) for y in [-1,0,1]})
 factors={r['base_index']:r['factors'] for r in rows};keys=['geometry_family','material','aspect','foreign_family','signed_quarter'];cells=Counter(tuple(str(r[k]) for k in keys) for r in factors.values());values=[sorted({str(r[k]) for r in factors.values()}) for k in keys]
 factorial=dict(keys=keys,base_count=128,cartesian_count=128,unique_cells=len(cells),missing_cells=[list(x) for x in itertools.product(*values) if x not in cells],duplicated_cells=[dict(cell=list(x),count=n) for x,n in cells.items() if n>1],pairwise_counts={})
 for i,k in enumerate(keys):
  for l in keys[i+1:]:
   c=Counter((str(r[k]),str(r[l])) for r in factors.values());factorial['pairwise_counts'][k+'__'+l]=dict(observed_cells=len(c),counts=sorted(set(c.values())))
 positive={}
 for name,col,mi in [('visible',0,1),('foreign',1,2)]:
  xx=mass[av&(t['support_target'][:,col]==1),mi];positive[name]=dict(N=len(xx),fraction_quantiles=quantile(xx),sample_equivalent_of9216_quantiles=quantile(xx*9216),below4samples=int((xx*9216<4).sum()),below16samples=int((xx*9216<16).sum()),not_binary_mask_or_RGB_observability_proof=True)
 mixture={}
 for key in ['scene_owner_point','scene_owner_central4','scene_owner_central32']:
  f=a[key][:,2];mixture[key]={str(y):dict(N=int((av&(et==y)).sum()),Fmass0=int((av&(et==y)&(f==0)).sum()),Fmass1=int((av&(et==y)&(f==1)).sum()),mixedF=int((av&(et==y)&(f>0)&(f<1)).sum()),rawFmasssum=float(np.sum(f[av&(et==y)]))) for y in [-1,0,1]}
 modelsummary={};bindings=dict(scope=b(SCOPE),TRAIN_manifest=b(manifest),TRAIN_owner6=b(ownerpath),TRAIN_targets=b(targetpath),TRAIN_endmass=b(endpath))
 for arm in ARMS:
  opath=V181/'runs/C50_C51_LOCAL_SUPPORT_FIXED8000_v2'/arm/'FINAL_SYNTH_TRAIN4096_OUTPUTS.npz';bindings[arm+'_TRAINoutputs']=b(opath)
  with np.load(opath,allow_pickle=False) as z:
   assert np.array_equal(z['original_global_index'],np.arange(4096)) and np.array_equal(z['query_xy'],a['query_xy'],equal_nan=True) and np.array_equal(z['L'],a['L'],equal_nan=True) and np.array_equal(z['original_query_available'],av);sp=z['support_prediction'];op=z['owner_prediction']
   modelsummary[arm]=dict(known_available_support_error_counts=[int((av&(t['support_target'][:,j]>=0)&(sp[:,j]!=t['support_target'][:,j])).sum()) for j in range(2)],foreignYES_on_unknown_target=int((av&(et<0)&(sp[:,1]==1)).sum()),visibleYES_on_unknown_target=int((av&(vt<0)&(sp[:,0]==1)).sum()),unknown_is_not_a_false_positive_label=True)
 publiccf={}
 for domain in ['PUBLIC_TRAIN1480','CF3816']:
  meta=lines(V181/'source_raw96_cache_v1'/(domain+'_ROWS.jsonl'))
  with np.load(V181/'source_raw96_cache_v1'/(domain+'_TARGETS.npz'),allow_pickle=False) as z:y=z['owner_target'].copy();assert (z['support_target']==-1).all()
  publiccf[domain]=dict(rows=len(meta),weak_owner_counts=counts(y),new_end_supervision_known0=True,models={})
  for arm in ARMS:
   opath=V181/'runs/C50_C51_LOCAL_SUPPORT_FIXED8000_v2'/arm/('FINAL_'+domain+'_OUTPUTS.npz');bindings[arm+'_'+domain+'_outputs']=b(opath)
   with np.load(opath,allow_pickle=False) as z:
    assert z['row_id'].tolist()==[r['row_id'] for r in meta];sp=z['support_prediction'];op=z['owner_prediction'];valid=z['original_query_available']
    publiccf[domain]['models'][arm]={str(c):dict(N=int((y==c).sum()),visible_yes=int(((y==c)&valid&(sp[:,0]==1)).sum()),foreign_yes=int(((y==c)&valid&(sp[:,1]==1)).sum()),owner_correct=int(((y==c)&valid&(op==c)).sum())) for c in [0,1,2]}
  if domain=='CF3816':
   by=defaultdict(list)
   for r in meta:by[(r['original']['pair_id'],r['native_slot'])].append(r)
   assert len(by)==1272 and all(len(v)==3 for v in by.values())
   same=sum(all(r['query_xy']==vv[0]['query_xy'] and r['L']==vv[0]['L'] and r['original_query_available']==vv[0]['original_query_available'] for r in vv) for vv in by.values());publiccf[domain]['all1272_pair_native_original_qL_flags_fixed']=same==1272
 cf=H/'SOURCE_TRAIN_COUNTS_AND_COVERAGE.json';write(cf,dict(rows4096=4096,original_inputU80=80,all4_qL_identical_groups_including_missing=int(samegroups),all4_qL_identical_all_available_groups=int(validsamegroups),pair_comparisons=pairsummary,conditional_coverage=coverage,all8_geometry_material_aspect=fullcross,factorial=factorial,positive_support_mass=positive,literal_ownerF_vs_endforeign=mixture,sourceTRAIN_apparent_model_association=modelsummary,publicTRAIN_CF_weak_owner_and_support_response=publiccf))
 source=P/'query_visibility_registered_native_diversity_20261023_v129/render_preparation';bindings['source_variant_math']=b(source/'corpus_math.py');bindings['source_renderer']=b(source/'render_corpus_once.py');bindings['original_input_geometry_code']=b(P/'query_visibility_actual_initializer_source_20261024_v132/frozen_visibility_preparation/run_frozen_visibility.py')
 findings=[
 'Original image-query replay is faithful, but valid BEFORE-COVER and BEFORE-NEAR exactsameqL comparisons are0; FAR32. Fourvariant allsame8 groups are missing, validallsame0. These original differentquery crops cannot isolate samequery occlusion causality.',
 'Nominal factor marginals and all pairwise5factor cells are balanced; 5wayCartesian has96/128unique,32missing and32double. This deterministic design limitation is not asserted to be a generation bug.',
 'Native6/7 foreign support labels are each512UNKNOWN; visibleNO113 occurs onlynative0..3. Full originalslots remain, and selfunknown is not automatically erroneous or grounds to drop slots.',
 'Positive endclass average mass is tiny in96grid (visiblemedian12.52/9216,foreign18.11/9216). This is fractional source support, not RGB contrast or physical corner gold. FullTRAIN contrast remains to be measured under separate authorization.',
 'Literal point/central4 ownerF differs from endforeign: central4 Fmass0 occurs21of138endYES; Fmass1 occurs7of1507endNO. Point counterparts24and9. Independent supervision heads answer different questions.',
 'Known sourceTRAINsupport errors0 in both frozenheads is apparentfit. ForeignUNKNOWN predictionsYES51vs127 are unconstrained diagnostic outputs, not false positives or new labels. Public/CF weakowner response associations have no endtruth and do not establish natural occlusion accuracy.',
 'V129core49-54 nearclearance is for the selected patch only, not every native end. Renderer61-75 keeps target/camera/source pixels unchanged, while V132input_geometry98-126 uses each scene original initializer proposal, so qL may differ. NEAR14endforeignYES is not itself a code-error proof.'
 ]
 report=dict(completed=True,PASS=True,source_TRAIN_only=True,code=b(Path(__file__)),bindings=bindings,counts=b(cf),original_qL_pair_ledger=b(pf),findings=findings,
  actual_code_or_metadata_error_detected=False,remaining_unknowns=['No newRGBcontrast/allTRAINpixelvisibility measurement in this audit.','No physicalnative/pointquality certification or actual transfer cause inferred.'],
  next_safe_action_under_frozen_scope='Prepare all128base COVER-originalq8/L/flags common-query diagnostic across4savedvariants. Preserve image-original andanchororiginal fields; require all originalCOVERraw96/valid byteparity before admission. SameV180end/owner targets are separate diagnostic-only GT, never modelinputs; noBEFOREanchorcomparison selection in this scope.',
  old_new_VAL_private_publicheld_V112_reads=0,new_pixels_model_forward_fit_qL_or_label_changes=0,selected_model=None,fully_solved=False)
 write(H/'REVIEW.json',report);print(json.dumps(dict(completed=True,PASS=True,review=b(H/'REVIEW.json'),counts=b(cf))))
if __name__=='__main__':main()
