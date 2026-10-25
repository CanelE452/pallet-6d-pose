"""Root-delegated V244 scalar SOURCE join only; no pixels/arrays/models."""
import csv,hashlib,io,json,math,os,statistics,stat,traceback
from collections import Counter
from pathlib import Path
R=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation');H=R/'V244_ROOT_FROZEN_BEHAVIOR';OUT=H/'ycb_pair_join_v1'
INPUTS={
 'pair_complete':(H/'v235_pair_replay_v1/COMPLETE.json','c68b19c4780929d5f0381fd625525711069f8d90d4b4c889df62a7f9dcbaf2f7'),
 'pair_CSV':(H/'v235_pair_replay_v1/ALL5406_OLD_PAIR_BEHAVIOR.csv','ed3bef4f0d4b526b62f171a3b6b3820fd9c4b5ec71c6844aa7eb1d18b8f3e19c'),
 'pair_summary':(H/'v235_pair_replay_v1/SUMMARY.json','256b73ea650d5d23de02ae5f35c3464a4381177362384a8b6768552e3818a7db'),
 'object_aux_metadata':(R/'V236_PUBLICATION_PACKAGE/artifacts/V234_ROOT_LABEL_DIAG/YCB_KNOWN1521_LABEL_QUERY_MODEL_DIAGNOSTIC.csv','8eab4fe1f9c58ba55ab308437605cc64a2c30dc92f07d21416759b455a342b08'),
 'draw_ledger':(R/'V235_ROOT_PUBLIC_COUNTERFACTUAL/data_only_source_fit_v1/TREATMENT_DRAW_SOURCE1000.jsonl','96566d42a74d574a45e926c0a0f99afd176a04d12cbf3cdd3b24d745f0565dee'),
 'plan':(R/'V244_YCB_PAIR_JOIN_A/JOIN_PLAN.json','da891812c9e5639fc78b6383804efc98bafdefc10aaeda0f7ed07cd393fac935')}

def safe(p):
 p=Path(p)
 if any(t in str(p).casefold() for t in ('v112','__pycache__','.pyc','current223','publicheld','oldval','v177','v178','source_holdout','v182','private_evaluation')) or '..' in p.parts or not p.is_absolute():raise ValueError('Forbidden lexical path before FS')
 for a in reversed((p,*p.parents)):
  try:mode=os.lstat(a).st_mode
  except FileNotFoundError:continue
  if stat.S_ISLNK(mode):raise ValueError('Symlink ancestor')
 return p

def read(p):
 with os.fdopen(os.open(safe(p),os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:return f.read()

def bind(p):
 raw=read(p);return {'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)}

def checked(pair):
 p,sha=pair;raw=read(p)
 if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('Explicit metadata bytes changed')
 return raw

def write_json(name,v):
 p=safe(OUT/name)
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 return bind(p)

def write_csv(name,rows):
 p=safe(OUT/name)
 with p.open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 return bind(p)

def isflag(r):return (r['object_id']==7 and r['aux_slot'] in (3,7)) or (r['object_id']==2 and r['aux_slot']==2)

def metrics(rows,exposure):
 use=[r for r in rows if r['success']];n=len(use);orig=[r for r in use if r['original_D_prediction']=='D'];direction=[r['COVER_minus_SHAM'] for r in use]
 out={'attempts':len(rows),'successful_pairs':n,'failed_U':len(rows)-n,'COVER_E':sum(r['COVER_prediction']=='E' for r in use),'SHAM_D':sum(r['SHAM_prediction']=='D' for r in use),'COVER_E_and_SHAM_D':sum(r['COVER_prediction']=='E' and r['SHAM_prediction']=='D' for r in use),'COVER_gt_SHAM':sum(v>0 for v in direction),'COVER_eq_SHAM':sum(v==0 for v in direction),'COVER_lt_SHAM':sum(v<0 for v in direction),'COVER_minus_SHAM_mean':math.fsum(direction)/n if n else None,'COVER_minus_SHAM_median':statistics.median(direction) if n else None,'original_saved_D_correct':len(orig),'original_D_to_COVER_E':sum(r['COVER_prediction']=='E' for r in orig),'SHAM_preserves_original_correct_D':sum(r['SHAM_prediction']=='D' for r in orig),'SHAM_preservation_denominator':len(orig)}
 for role,target in [('COVER','E'),('SHAM','D')]:
  weights={r['pair_index']:exposure[(r['pair_index'],role)] for r in use};total=sum(weights.values());correct=sum(weights[r['pair_index']] for r in use if r[role+'_prediction']==target)
  out[role+'_E_rate' if role=='COVER' else 'SHAM_D_rate']=out['COVER_E' if role=='COVER' else 'SHAM_D']/n if n else None
  out[role+'_actual_gradient_draws']=total;out[role+'_unique_pairs_drawn']=sum(w>0 for w in weights.values());out[role+'_pool_successful_pairs_never_drawn']=sum(w==0 for w in weights.values());out[role+'_draw_weighted_correct']=correct;out[role+'_draw_weighted_correct_rate']=correct/total if total else None
 out['paired_crossing_rate']=out['COVER_E_and_SHAM_D']/n if n else None
 out['unique_pairs_ever_drawn_either_role']=sum(exposure[(r['pair_index'],'COVER')]+exposure[(r['pair_index'],'SHAM')]>0 for r in use)
 out['unique_pairs_both_roles_drawn']=sum(exposure[(r['pair_index'],'COVER')]>0 and exposure[(r['pair_index'],'SHAM')]>0 for r in use)
 return out

def run():
 if safe(OUT).exists():raise ValueError('Fresh output only; no retry')
 OUT.mkdir()
 try:
  raw={k:checked(v) for k,v in INPUTS.items()};inputbindings={k:{'path':str(INPUTS[k][0]),'sha256':INPUTS[k][1],'size_bytes':len(v)} for k,v in raw.items()}
  (OUT/'JOIN_PLAN.json').write_bytes(raw['plan']);(OUT/'join_saved_YCB_pair_metadata.py').write_bytes(read(Path(__file__).absolute()))
  complete=json.loads(raw['pair_complete']);saved_summary=json.loads(raw['pair_summary'])
  if not all(v.get('completed') is True and v.get('PASS') is True for v in (complete,saved_summary)):raise ValueError('Completed actual pair replay required')
  for key,name in [('pair_CSV','ALL5406_OLD_PAIR_BEHAVIOR.csv'),('pair_summary','SUMMARY.json')]:
   if (complete['outputs'][name]['path'],complete['outputs'][name]['sha256'])!=(str(INPUTS[key][0]),INPUTS[key][1]):raise ValueError('Exact replay output binding graph')
  mapping={};samples=set()
  for v in csv.DictReader(io.StringIO(raw['object_aux_metadata'].decode())):
   rowid=v['row_id'];sample=int(v['sample_index'])
   if rowid in mapping or sample in samples:raise ValueError('Unique original YCB row IDs/sampleindices required')
   mapping[rowid]={k:v[k] for k in ('row_id','role','associated_r_state')};mapping[rowid].update({'sample_index':sample,'object_id':int(v['object_id']),'aux_slot':int(v['aux_slot'])});samples.add(sample)
  if len(mapping)!=1521:raise ValueError('Existing knownYCB1521 retained')
  allrows=[];parentset=set();ycb=[]
  for i,v in enumerate(csv.DictReader(io.StringIO(raw['pair_CSV'].decode()))):
   p=int(v['parent_sample_index']);f=int(v['family']);ok=v['success']=='True'
   if int(v['pair_index'])!=i or p in parentset or f not in (0,1) or v['source_role']!='FIT' or v['success'] not in ('True','False') or v['original_teacher_D']!='True':raise ValueError('Original complete FITD pair ordering/identity')
   parentset.add(p);v.update({'pair_index':i,'parent_sample_index':p,'family':f,'success':ok})
   if ok:
    c=float(v['COVER_selected_logit']);s=float(v['SHAM_selected_logit']);d=float(v['COVER_minus_SHAM'])
    if not all(map(math.isfinite,(c,s,d))) or d!=c-s or int(v['COVER_target'])!=1 or int(v['SHAM_target'])!=0 or v['COVER_prediction']!=('E' if c>=0 else 'D') or v['SHAM_prediction']!=('E' if s>=0 else 'D'):raise ValueError('Saved pair fixed zero-threshold and weak targets')
    v['COVER_minus_SHAM']=d
   elif any(v[k]!='' for k in ('COVER_selected_logit','SHAM_selected_logit','COVER_minus_SHAM')) or v['COVER_prediction']!='U' or v['SHAM_prediction']!='U' or int(v['COVER_target'])!=-1 or int(v['SHAM_target'])!=-1:raise ValueError('Failed39 U/null retained')
   allrows.append(v)
   if f==1:
    meta=mapping[v['row_id']]
    if meta['sample_index']!=p or meta['role']!='FIT' or meta['associated_r_state']!='D':raise ValueError('Exact original associated-r FITD rowid/sample join; no fallback')
    v.update({'object_id':meta['object_id'],'aux_slot':meta['aux_slot'],'associated_r_state':meta['associated_r_state']});v['fixed_review_group']='FLAGGED_OBJ7_AUX3_7_OR_OBJ2_AUX2' if isflag(v) else 'OTHER_YCB';ycb.append(v)
  if len(allrows)!=5406 or len(ycb)!=841 or sum(v['success'] for v in ycb)!=830:raise ValueError('All5406 and YCB841/830/11 exact denominators')
  exposure=Counter();natural=Counter();ledgercount=0
  for update,line in enumerate(raw['draw_ledger'].splitlines(),1):
   v=json.loads(line);ledgercount+=1
   keys=('original_sample_indices','actual_parent_indices','augmentation_attempt_indices','augmentation_variant_minus1Original_0COVER_1SHAM','family','target')
   if v['update']!=update or any(len(v[k])!=32 for k in keys) or v['family']!=[0]*16+[1]*16 or v['target']!=([0]*8+[1]*8)*2:raise ValueError('All1000 fixed32 family8D8E draw records')
   for j,(old,parent,attempt,variant,f,target) in enumerate(zip(*(v[k] for k in keys))):
    if attempt==-1:
     if variant!=-1 or parent!=old:raise ValueError('Natural draw unchanged')
     natural[(f,target)]+=1;continue
    if not 0<=attempt<5406 or variant not in (0,1):raise ValueError('Exact compact attempt/variant IDs')
    row=allrows[attempt]
    if not row['success'] or parent!=row['parent_sample_index'] or f!=row['family'] or target!=(1 if variant==0 else 0):raise ValueError('Actually drawn CF must successfulFIT same declared family/sourceparent')
    if (variant==1 and j%16 not in range(4,8)) or (variant==0 and j%16 not in range(12,16)):raise ValueError('Preserved old last4 replacement positions')
    exposure[(attempt,'COVER' if variant==0 else 'SHAM')]+=1
  if ledgercount!=1000:raise ValueError('Original1000 ledger records')
  overall=metrics(ycb,exposure);ref=saved_summary['family_metrics']['YCB_ASSOCIATED_R_AUX']
  for key,value in ref.items():
   if key not in overall:continue
   if isinstance(value,int):
    if overall[key]!=value:raise ValueError('Independent YCB count mismatch:'+key)
   elif value is not None and not math.isclose(overall[key],value,rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Independent YCB saved scalar mismatch:'+key)
  groups={'ALL_YCB':ycb,'FIXED_REVIEW_FLAGGED':[v for v in ycb if isflag(v)],'OTHER_YCB':[v for v in ycb if not isflag(v)]}
  for obj,aux in sorted({(v['object_id'],v['aux_slot']) for v in ycb}):groups[f'object{obj}_aux{aux}']=[v for v in ycb if (v['object_id'],v['aux_slot'])==(obj,aux)]
  groupmetrics={name:metrics(rows,exposure) for name,rows in groups.items()}
  exposure_rows=[]
  for v in allrows:
   row={'pair_index':v['pair_index'],'parent_sample_index':v['parent_sample_index'],'row_id':v['row_id'],'family':v['family'],'success':v['success'],'object_id':v.get('object_id'),'aux_slot':v.get('aux_slot'),'fixed_review_group':v.get('fixed_review_group'),'COVER_gradient_draws':exposure[(v['pair_index'],'COVER')],'SHAM_gradient_draws':exposure[(v['pair_index'],'SHAM')],'COVER_prediction':v['COVER_prediction'],'SHAM_prediction':v['SHAM_prediction']};exposure_rows.append(row)
  for v in ycb:v['COVER_gradient_draws']=exposure[(v['pair_index'],'COVER')];v['SHAM_gradient_draws']=exposure[(v['pair_index'],'SHAM')]
  joined=write_csv('ALL841_YCB_PAIR_ROWS.csv',ycb);drawcsv=write_csv('ALL5406_ACTUAL_DRAW_EXPOSURE.csv',exposure_rows)
  gm=write_csv('GROUP_METRICS.csv',[{'group':name,**value} for name,value in groupmetrics.items()])
  family_exposure={str(f):{'COVER_pool':sum(v['family']==f and v['success'] for v in allrows),'SHAM_pool':sum(v['family']==f and v['success'] for v in allrows),'COVER_draws':sum(exposure[(v['pair_index'],'COVER')] for v in allrows if v['family']==f),'SHAM_draws':sum(exposure[(v['pair_index'],'SHAM')] for v in allrows if v['family']==f),'COVER_unique_drawn':sum(v['family']==f and exposure[(v['pair_index'],'COVER')]>0 for v in allrows),'SHAM_unique_drawn':sum(v['family']==f and exposure[(v['pair_index'],'SHAM')]>0 for v in allrows)} for f in (0,1)}
  summary=write_json('SUMMARY.json',{'completed':True,'PASS':True,'YCB_rows':841,'YCB_success':830,'YCB_failed_U':11,'groups':groupmetrics,'draw_updates':1000,'family_CF_exposure':family_exposure,'natural_draw_counts':{str(k):v for k,v in natural.items()},'metadata_join':'Exact unique row_id and original sample index, roleFIT/associated-rD; no unknown mapping fallback','no_new_model_or_threshold_selection':True,'limits':['This is old constructed SOURCE FIT pool behavior and actual optimizer draw exposure, not independent natural SOURCE DEV/current performance or causal proof.','Augmentation pool feature extraction and actual gradient use are different; per-role weighted metrics retain repetitions and unique-pair denominators separately.','COVER/SHAM were independently drawn in the old V235 recipe; no matched-pair draw co-occurrence is asserted.','The existing YCB CSV contributes ONLY original row identity/object/aux/role/associated-state mapping. Its historical model-score/reference/owner columns are not used.','Original SOURCE DEV natural metrics are not recomputed; this join does not prove why their failure occurs.','YCB native remains UNKNOWN; associated-r weak targets are not literal query-center/native physical gold.']})
  top=groupmetrics['FIXED_REVIEW_FLAGGED'];text=f'''# 既存 YCB SOURCE FIT CF 行動・draw join\n\n新しい画像・配列・モデル実行はなく、既存の row_id/sample_index と object/aux の scalar metadata を結合しました。全YCB成功830組: COVER E {overall['COVER_E']}/830、SHAM D {overall['SHAM_D']}/830、両方正解 {overall['COVER_E_and_SHAM_D']}/830。\n\n固定注目群 obj7 aux3/7 または obj2 aux2 は成功{top['successful_pairs']}組です。実際の学習drawは COVER {top['COVER_actual_gradient_draws']}回、SHAM {top['SHAM_actual_gradient_draws']}回です。poolと反復drawを別分母にしました。\n\nこれは既存 SOURCE FIT 合成pair機能の診断です。自然SOURCE DEVやcurrent性能、失敗原因・因果関係の証明ではありません。YCB native は UNKNOWN のままです。\n\n[SUMMARY](SUMMARY.json) · [group metrics](GROUP_METRICS.csv) · [YCB rows](ALL841_YCB_PAIR_ROWS.csv) · [actual draw exposure](ALL5406_ACTUAL_DRAW_EXPOSURE.csv)\n'''
  # The report is Korean; preserve scalar quantities with the same scope.
  text=text.replace('既存 YCB SOURCE FIT CF 行動・draw join','기존 YCB SOURCE FIT CF 행동·draw join').replace('新しい画像・配列・モデル実行はなく、既存の row_id/sample_index と object/aux の scalar metadata を結合しました。全YCB成功830組:','새 이미지·배열·모델 실행 없이 기존 row_id/sample_index와 object/aux scalar metadata만 결합했습니다. 전체 YCB 성공830쌍:').replace('固定注目群','고정 주목 그룹').replace(' は成功','은 성공').replace('組です。実際の学習drawは','쌍입니다. 실제 학습 draw는').replace('回、','회, ').replace('回です。poolと反復drawを別分母にしました。','회입니다. pool과 반복 draw는 별도 분모입니다.').replace('これは既存 SOURCE FIT 合成pair機能の診断です。自然SOURCE DEVやcurrent性能、失敗原因・因果関係の証明ではありません。YCB native は UNKNOWN のままです。','기존 SOURCE FIT 합성 pair 기능의 진단입니다. 자연 SOURCE DEV나 current 성능, 실패 원인·인과관계의 증명이 아닙니다. YCB native는 UNKNOWN을 유지합니다.').replace('または','또는')
  with (OUT/'KO.md').open('x') as f:f.write(text)
  complete_out=write_json('COMPLETE.json',{'completed':True,'PASS':True,'delegated_Root_actual_savedmetadata_join':True,'entry_authority':'Root direct task: admit exact completed B3 files + existing YCB mapping CSV; subsequent exact1000drawledger admission.','inputs':inputbindings,'plan_exact_copy':bind(OUT/'JOIN_PLAN.json'),'code_exact_copy':bind(OUT/'join_saved_YCB_pair_metadata.py'),'summary':summary,'group_metrics':gm,'joined_YCB':joined,'draw_exposure':drawcsv,'report':bind(OUT/'KO.md'),'source_rows':{'all_attempts':5406,'YCB_attempts':841,'YCB_success':830,'YCB_failure':11},'new_RGB_array_weight_model_GT_or_currentheld_access':0,'model_or_threshold_selection':False,'automatic_retry':False})
  print(json.dumps({'completed':True,'PASS':True,'COMPLETE':complete_out,'ALL_YCB':overall,'FIXED_REVIEW_FLAGGED':top,'OTHER_YCB':groupmetrics['OTHER_YCB'],'family_exposure':family_exposure},ensure_ascii=False),flush=True)
 except BaseException:
  write_json('FAILED_EXECUTION.json',{'completed':False,'PASS':False,'automatic_retry':False,'traceback':traceback.format_exc()});raise

if __name__=='__main__':run()
