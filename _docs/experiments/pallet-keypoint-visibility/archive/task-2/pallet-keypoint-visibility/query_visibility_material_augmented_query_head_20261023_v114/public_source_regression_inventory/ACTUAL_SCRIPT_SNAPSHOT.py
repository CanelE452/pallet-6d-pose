from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json,csv,hashlib
import numpy as np
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility');V=P/'query_visibility_material_augmented_query_head_20261023_v114';R=P/'query_visibility_query_anchored_cue_20261023_v96';T=V/'public_source_regression_inventory'
def rd(p):return json.loads(p.read_text())
def jl(p):return [json.loads(x) for x in p.read_text().splitlines()]
def bind(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
files={'scope':T/'SCOPE_BEFORE_SCORE_ARRAYS.json','C11_scores':P/'query_visibility_spatial_target_query_features_20261023_v111/runs/C11_DINOV2_SPATIAL_TARGET_QUERY/SELECTED_ALL41984_SOURCE_SCORES.npz','C12_scores':V/'runs/C12_MATERIAL_AUGMENTED_SPATIAL_QUERY/SELECTED_ALL45184_SOURCE_SCORES.npz','base_lineage':R/'actual_query_anchored_image_intake/SOURCE960_DRIVER_LINEAGE.jsonl','old_GT':R/'actual_query_anchored_image_intake/OCCURRENCES31680_GT_ONLY.jsonl','assignments':R/'query_anchored_pair_preparation/DONOR_ASSIGNMENT960_K16_BEFORE_PIXELS.jsonl','feasibility':R/'actual_query_anchored_image_intake/FEASIBILITY15360_GT_ONLY.jsonl','C11_source_results':P/'query_visibility_spatial_target_query_features_20261023_v111/runs/C11_DINOV2_SPATIAL_TARGET_QUERY/SOURCE_RESULTS.json','C12_source_results':V/'runs/C12_MATERIAL_AUGMENTED_SPATIAL_QUERY/SOURCE_RESULTS.json','C12_source_review':V/'saved_source_review/REVIEW.json','fixed_first_regression_selection':V/'source_public_regression_diagnosis/SELECTION_LOCK_BEFORE_PIXELS.json','frozen_C11_material_baseline':P/'query_visibility_ycb_wood_foam_source_20261023_v113/c11_frozen_material_baseline/SOURCE_RESULTS.json'}
assert files['scope'].exists()
with np.load(files['C11_scores'],allow_pickle=False) as a:C11={k:a[k] for k in a.files}
with np.load(files['C12_scores'],allow_pickle=False) as a:C12={k:a[k] for k in a.files}
base=jl(files['base_lineage']);gt=jl(files['old_GT']);assign=jl(files['assignments']);feas=jl(files['feasibility']);assert (len(base),len(gt),len(assign),len(feas))==(960,31680,960,15360)
N=41984
samekeys=['query_available','factual_query_available','context_available','context_Q_inside','input_unit_index','raw_teacher_state','external_target','split','domain','source','variant','query_xy_raw','L_raw','global_occurrence_index','old_accepted_triplets','new_selected_triplets']
for k in samekeys:
 a=C11[k];b=C12[k] if k.endswith('triplets') else C12[k][:N]
 assert np.array_equal(a,b,equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a,b),k
for X in [C11,C12]:
 p=np.where(X['query_available'],(X['query_logit']>=0).astype(np.int8),-1).astype(np.int8);assert np.array_equal(p,X['prediction'])
assert np.array_equal(C11['global_occurrence_index'],np.arange(N))
triples=C11['old_accepted_triplets'];assert triples.shape==(194,3)
labels=np.array([g['candidate_aux_hidden_label_GT_only'] for g in gt]);assert np.array_equal(labels,C11['external_target'][:31680])

def cm_metrics(y,p):
 m=[[int(np.sum((y==r)&(p==c))) for c in [0,1,-1]] for r in [0,1]]
 def div(a,b):return a/b if b else None
 dn=sum(m[0]);en=sum(m[1]);return {'CM_D_E_by_pred_D_E_U':m,'known_D':dn,'known_E':en,'D_recall':div(m[0][0],dn),'E_recall':div(m[1][1],en),'D_precision':div(m[0][0],m[0][0]+m[1][0]),'E_precision':div(m[1][1],m[0][1]+m[1][1]),'known_U':m[0][2]+m[1][2],'known_U_rate':div(m[0][2]+m[1][2],dn+en)}
def transition(ix):
 p=C11['prediction'][ix];q=C12['prediction'][ix]
 return {'N':len(ix),'matrix_C11_D_E_U_by_C12_D_E_U':[[int(np.sum((p==a)&(q==b))) for b in [0,1,-1]] for a in [0,1,-1]],'C11_correctE_to_C12_wrongD':int(np.sum((p==1)&(q==0))),'C11_wrongD_to_C12_correctE':int(np.sum((p==0)&(q==1)))}
def info_before(b):
 x=base[b];return {'base_row_index':b,'input_id':x['input_id'],'original_query_id':x.get('original_query_id'),'base_image':x['fullRGB_original_image'],'base_image_sha256':x['original_image_sha256'],'raw_split':x['split'],'source':x['source'],'source_family':x['source_family'],'frame_index':x['frame_index'],'pair_id':x.get('pair_id'),'view':x.get('view'),'native_slot_logical_only':x['native_slot_logical_only'],'query_xy':x['query_xy'],'L':x['L']}

pairs=[]
for tr in triples:
 b,c,s=map(int,tr);g=gt[c];bi=g['base_row_index'];rank=g['donor_rank'];a=assign[bi];f=feas[16*bi+rank];d=a['ranks'][rank];x=base[bi]
 assert b==bi*33 and c==b+1+2*rank and s==c+1
 assert f['base_row_index']==bi and f['donor_rank']==rank and f['pair_ready'] and g['pair_ready']
 assert f['assigned_donor_id']==d['donor_id'] and g['variant']=='OPAQUE_COVER'
 assert np.array_equal(labels[[b,c,s]],[0,1,0]) and all(C11['query_available'][[b,c,s]])
 assert np.array_equal(C11['query_xy_raw'][[b,c,s]],np.repeat(np.array(x['query_xy'])[None,:],3,axis=0)) and np.all(C11['L_raw'][[b,c,s]]==x['L'])
 z1=[float(C11['query_logit'][i]) for i in [b,c,s]];z2=[float(C12['query_logit'][i]) for i in [b,c,s]]
 p1=[int(C11['prediction'][i]) for i in [b,c,s]];p2=[int(C12['prediction'][i]) for i in [b,c,s]]
 def zinfo(z, p):
  margin=z[1]-z[2];margin32=float(np.float32(np.float32(z[1])-np.float32(z[2])))
  return {'BEFORE':z[0],'COVER':z[1],'SHAM':z[2],'pred_BEFORE_COVER_SHAM':p,'cover_minus_sham_F64':margin,'cover_minus_sham_F32':margin32,'cover_minus_before_F64':z[1]-z[0],'margin_ge2':margin>=2,'margin_F32_ge2':margin32>=2,'absolute_cover_ge0':z[1]>=0,'absolute_cover_gt0':z[1]>0,'margin_ge2_but_cover_rawD':margin>=2 and z[1]<0,'all3_correct':p==[0,1,0]}
 r={**info_before(bi),'split':str(C11['split'][c]),'donor_rank':rank,'donor_id':d['donor_id'],'donor_object_id':d['object_id'],'donor_scene_id':d['scene_id'],'donor_frame_id':d['frame_id'],'donor_instance_id':d['instance_id'],'donor_image_sha256':d['image_sha256'],'donor_source_family':'PUBLIC_YCB_REAL_VISIBLE_CUTOUT','same_added_wood_object16':d['object_id']==16,'triple_global_indices':[b,c,s],'occurrence_id':g['occurrence_id'],'C11':zinfo(z1,p1),'C12':zinfo(z2,p2),'C11correctE_C12wrongD':p1[1]==1 and p2[1]==0,'C11wrongD_C12correctE':p1[1]==0 and p2[1]==1,'certificate':{'feasibility_row':16*bi+rank,'protected_pixels':f['protected_pixels'],'protected_fraction_L2':f['protected_area_over_L_squared'],'protected_cover_all1':f['protected_cover_all1'],'protected_sham_all0':f['protected_sham_all0'],'same_full_RGBA_transform':f['same_full_RGBA_transform'],'mass_PASS':f['mass_recheck']['PASS'],'exclusive_core_no_known_other':f['exclusive_alpha']['every_eroded_native_alpha_pixel_selected_and_not_other'],'complete_other_instance_masks':f['exclusive_alpha']['complete_other_instance_masks']}}
 assert all(r['certificate'][k] for k in ['protected_cover_all1','protected_sham_all0','same_full_RGBA_transform','mass_PASS','exclusive_core_no_known_other','complete_other_instance_masks'])
 pairs.append(r)
assert len(pairs)==194

def summarize_pairs(ps):
 n=len(ps);out={'accepted_covers':n,'unique_base_queries':len({p['base_row_index'] for p in ps}),'unique_base_images':len({p['base_image_sha256'] for p in ps}),'unique_donors':len({p['donor_id'] for p in ps}),'regressions_C11E_to_C12D':sum(p['C11correctE_C12wrongD'] for p in ps),'fixes_C11D_to_C12E':sum(p['C11wrongD_C12correctE'] for p in ps)}
 for name in ['C11','C12']:
  q=[p[name] for p in ps];out[name]={'correct_cover_E':sum(x['absolute_cover_ge0'] for x in q),'cover_D':sum(not x['absolute_cover_ge0'] for x in q),'cover_U':0,'all3_correct':sum(x['all3_correct'] for x in q),'cover_minus_sham_ge2':sum(x['margin_ge2'] for x in q),'margin_ge2_but_cover_rawD':sum(x['margin_ge2_but_cover_rawD'] for x in q),'margin_lt2_and_cover_rawD':sum((not x['margin_ge2']) and not x['absolute_cover_ge0'] for x in q),'F32_vs_F64_margin_ge2_disagreements':sum(x['margin_ge2']!=x['margin_F32_ge2'] for x in q)}
 return out
oldmetrics={};transitions={}
for sp in ['TRAIN','VALIDATION']:
 m=(np.arange(N)<31680)&(C11['split']==sp)
 oldmetrics[sp]={}
 for var in ['ALL','BEFORE','OPAQUE_COVER','CUE_PRESERVING_SHAM']:
  ix=np.flatnonzero(m & ((C11['variant']==var) if var!='ALL' else np.ones(N,dtype=bool)))
  oldmetrics[sp][var]={'all_rows':len(ix),'teacher_U_retained':int(np.sum(C11['external_target'][ix]==-1)),'C11':cm_metrics(C11['external_target'][ix],C11['prediction'][ix]),'C12':cm_metrics(C12['external_target'][ix],C12['prediction'][ix])}
  ei=ix[C11['external_target'][ix]==1];transitions[sp+'_'+var]=transition(ei)
by_object=[]
for sp in ['TRAIN','VALIDATION']:
 ps=[p for p in pairs if p['split']==sp]
 for ob in sorted({p['donor_object_id'] for p in ps}):
  q=[p for p in ps if p['donor_object_id']==ob];by_object.append({'split':sp,'object_id':ob,'same_added_material_object16':ob==16,**summarize_pairs(q)})
by_rank=[{'split':sp,'donor_rank':rank,**summarize_pairs([p for p in pairs if p['split']==sp and p['donor_rank']==rank])} for sp in ['TRAIN','VALIDATION'] for rank in range(16)]
reg=[p for p in pairs if p['C11correctE_C12wrongD']];fix=[p for p in pairs if p['C11wrongD_C12correctE']]
assert sum(p['split']=='VALIDATION' for p in reg)==5
# Include eligibility/geometry denominator for attempted D bases: no RGB/body opens.
attempted=[]
for f in feas:
 if f['reason']=='QUERY_ANCHORED_BEFORE_NOT_D_NO_EDIT':continue
 a=assign[f['base_row_index']]['ranks'][f['donor_rank']]
 assert a['donor_id']==f['assigned_donor_id']
 sp='TRAIN' if f['raw_split'] in ['TRAIN','TRAIN_LAB'] else 'VALIDATION'
 attempted.append({'split':sp,'object_id':a['object_id'],'source':f['source'],'accepted':f['pair_ready'],'reason':f['reason']})
assert len(attempted)==976
attempts_by_object=[]
for sp in ['TRAIN','VALIDATION']:
 for ob in sorted({x['object_id'] for x in attempted if x['split']==sp}):
  xs=[x for x in attempted if x['split']==sp and x['object_id']==ob];attempts_by_object.append({'split':sp,'object_id':ob,'locked_eligible_D_attempts':len(xs),'accepted_covers':sum(x['accepted'] for x in xs),'failure_reasons':dict(Counter(x['reason'] for x in xs if not x['accepted']))})
# Other source groups: paired synthetic selected D/E/D only, and unpaired natural E summaries.
new_pairs=[]
for tr in C11['new_selected_triplets']:
 b,c,s=map(int,tr);z1=[float(C11['query_logit'][i]) for i in [b,c,s]];z2=[float(C12['query_logit'][i]) for i in [b,c,s]]
 new_pairs.append({'split':str(C11['split'][c]),'C11':zinfo(z1,[int(C11['prediction'][i]) for i in [b,c,s]]),'C12':zinfo(z2,[int(C12['prediction'][i]) for i in [b,c,s]])})
other={}
for name,mask in [('synthetic_all', (np.arange(N)>=31680)&(np.arange(N)<33984)),('YCB_natural_surface_TRAIN',(np.arange(N)>=33984))]:
 for sp in ['TRAIN','VALIDATION']:
  ix=np.flatnonzero(mask&(C11['split']==sp))
  if len(ix):other[name+'_'+sp]={'all_rows':len(ix),'C11':cm_metrics(C11['external_target'][ix],C11['prediction'][ix]),'C12':cm_metrics(C12['external_target'][ix],C12['prediction'][ix]),'known_E_transitions':transition(ix[C11['external_target'][ix]==1])}
for sp in ['TRAIN','VALIDATION']:
 xs=[x for x in new_pairs if x['split']==sp];other['synthetic_selected_triplets_'+sp]={'N':len(xs)}
 for name in ['C11','C12']:
  q=[p[name] for p in xs];other['synthetic_selected_triplets_'+sp][name]={'correct_cover_E':sum(x['absolute_cover_ge0'] for x in q),'cover_D':sum(not x['absolute_cover_ge0'] for x in q),'all3_correct':sum(x['all3_correct'] for x in q),'margin_ge2':sum(x['margin_ge2'] for x in q),'margin_ge2_but_cover_rawD':sum(x['margin_ge2_but_cover_rawD'] for x in q),'F32_vs_F64_margin_ge2_disagreements':sum(x['margin_ge2']!=x['margin_F32_ge2'] for x in q)}
# Natural old four before E entries retain original source identities; no forced donor join.
natural=[]
for i in np.flatnonzero((np.arange(N)<31680)&(C11['variant']=='BEFORE')&(C11['external_target']==1)):
 g=gt[int(i)];x=info_before(g['base_row_index']);x.update({'global_occurrence_index':int(i),'split':str(C11['split'][i]),'C11_z':float(C11['query_logit'][i]),'C12_z':float(C12['query_logit'][i]),'C11_prediction':int(C11['prediction'][i]),'C12_prediction':int(C12['prediction'][i]),'donor':'NONE_NATURAL_BEFORE'});natural.append(x)
assert len(natural)==4
summary={'completed':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'analysis':'Saved public source only, no new model/RGB/current/V112 access.','C11_selected_update':1800,'C12_selected_update':1200,'fixed_threshold_logit':0,'match_checks':{'all41984_common_teacher_query_L_fold_availability_unit_variant_exact':True,'all194accepted_oldtriplets_and960base_queries_joined':True,'all194construction_certificates_consistent':True,'no_features_or_masks_or_RGB_opened':True},'old_public_original_occurrences':31680,'accepted_controlled_covers_total':194,'eligible_D_locked_attempts':976,'all_locked_assignments':15360,'LOCO_accepted_covers':0,'PALLOX_accepted_covers':194,'old_public_confusion':oldmetrics,'old_E_transitions':transitions,'accepted_covers_summary':{sp:summarize_pairs([p for p in pairs if p['split']==sp]) for sp in ['TRAIN','VALIDATION']},'accepted_covers_by_donor_object':by_object,'accepted_covers_by_rank':by_rank,'attempt_geometry_denominator_by_object':attempts_by_object,'regressions':reg,'counter_direction_fixes':fix,'natural_BEFORE_E4':natural,'other_source_groups':other,'all5_VAL_regression_count':5,'VAL_regression_object_ids':dict(Counter(str(x['donor_object_id']) for x in reg if x['split']=='VALIDATION')),'VAL_regression_unique_base_queries':len({x['base_row_index'] for x in reg if x['split']=='VALIDATION'}),'VAL_regression_unique_base_images':len({x['base_image_sha256'] for x in reg if x['split']=='VALIDATION'}),'limits':['Conditionally labeled query-anchored physical-cue AUX, not native/human gold. All accepted old covers are Pallox and share one physical bin; LOCO supplies no accepted covers.','Ranks/donors repeat on clustered base queries/images. Object/family concentration has geometry-admission and fold-pool covariates; no population or isolated causal conclusion.','C11/C12 material composition, batch, selector and fitted parameters co-changed. This is a postfit descriptive source diagnosis, no model/threshold or training design selected.','No color-derived donor identification. Donor object IDs and scene/frame/instance IDs come exclusively from prelocked assignment and saved certificates. Object16 is the same identity included as added wood material source.','Relative cover-sham margin is independent of absolute sign; zero is E per authoritative decoder. Reported margin uses exact difference of saved FP32 values in F64, with separate F32 threshold agreement.','Unpaired natural surface rows have no cover/sham correspondence; no artificial relative-margin pairing is introduced.'],'new_model_RGB_fit_private_current_V112_PnP':0,'bindings':{k:bind(p) for k,p in files.items()},'own_code':bind(Path(__file__))}
(T/'RESULT.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
# Compact rows, preserving all accepted denominators, not merely regressions.
cols=['split','base_row_index','input_id','pair_id','view','native_slot_logical_only','query_xy','L','donor_rank','donor_id','donor_object_id','donor_scene_id','donor_frame_id','donor_instance_id','source','source_family','base_image','base_image_sha256','triple_global_indices','C11_BEFORE','C11_COVER','C11_SHAM','C11_margin','C11_margin_ge2_coverD','C12_BEFORE','C12_COVER','C12_SHAM','C12_margin','C12_margin_ge2_coverD','C11correctE_C12wrongD','C11wrongD_C12correctE']
def csvrow(x):
 r={k:x.get(k) for k in cols}
 for k in ['query_xy','triple_global_indices']:r[k]=json.dumps(r[k])
 for a in ['C11','C12']:
  for st in ['BEFORE','COVER','SHAM']:r[a+'_'+st]=x[a][st]
  r[a+'_margin']=x[a]['cover_minus_sham_F64'];r[a+'_margin_ge2_coverD']=x[a]['margin_ge2_but_cover_rawD']
 return r
for name,xs in [('ALL194_ACCEPTED_COVERS.csv',pairs),('ALL_REGRESSIONS.csv',reg),('ALL_COUNTER_DIRECTION_FIXES.csv',fix)]:
 with (T/name).open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=cols);w.writeheader();w.writerows(csvrow(x) for x in xs)
(T/'ALL194_ACCEPTED_TRIPLETS.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in pairs))
seal={'completed':True,'PASS':'Saved metadata and arithmetic consistency; not causal or training admission','scope':bind(files['scope']),'outputs':[bind(T/n) for n in ['RESULT.json','ALL194_ACCEPTED_COVERS.csv','ALL_REGRESSIONS.csv','ALL_COUNTER_DIRECTION_FIXES.csv','ALL194_ACCEPTED_TRIPLETS.jsonl']],'source_model_or_data_changes':0,'private_current_V112_RGB_model_forward_fit_PnP':0}
(T/'SEAL.json').write_text(json.dumps(seal,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'accepted':summary['accepted_covers_summary'],'object':by_object,'all5VAL_objectIDs':summary['VAL_regression_object_ids'],'regressionVAL_unique_basequeries':summary['VAL_regression_unique_base_queries'],'regressionVAL_unique_images':summary['VAL_regression_unique_base_images'],'regression_records':[{'base':x['base_row_index'],'rank':x['donor_rank'],'object':x['donor_object_id'],'split':x['split'],'C11cover':x['C11']['COVER'],'C12cover':x['C12']['COVER'],'C12margin':x['C12']['cover_minus_sham_F64']} for x in reg],'fixes':len(fix),'other':other,'result':bind(T/'RESULT.json'),'seal':bind(T/'SEAL.json')},ensure_ascii=False))
