"""Independent V214 saved scalar/CSV/pair arithmetic only. No payload/model imports."""
from pathlib import Path
from collections import defaultdict,Counter
import csv,hashlib,json,math,statistics
import numpy as np
V=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_source_train_modality_reliance_diagnosis_20261025_v214')
H=V/'source_train_modality_diagnosis_v2';O=Path(__file__).parent
def read(name):
 p=H/name;assert 'v112'not in str(p).lower()
 for q in [p,*p.parents]:assert not q.is_symlink()
 assert p.parent==H and p.suffix in{'.json','.jsonl','.csv'}
 return p.read_bytes()
def bind(name):return dict(path=str(H/name),sha256=hashlib.sha256(read(name)).hexdigest())
def load(name):return json.loads(read(name))
def close(a,b):
 if isinstance(a,dict):assert set(a)==set(b);[close(a[k],b[k])for k in a]
 elif isinstance(a,list):assert len(a)==len(b);[close(x,y)for x,y in zip(a,b)]
 elif a is None or isinstance(a,(str,bool,int)):assert a==b,(a,b)
 else:assert b is not None and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-10),(a,b)
def label(x):return -1 if x is None or not math.isfinite(x)else int(x>=0)
def cm(rows):
 m=[[0,0,0],[0,0,0]]
 for r in rows:m[r['known_target_D0_E1']][2 if r['prediction_D0_E1_Uminus1']==-1 else r['prediction_D0_E1_Uminus1']]+=1
 def div(a,b):return a/b if b else None
 return dict(N=len(rows),D=sum(m[0]),E=sum(m[1]),CM_D_E_by_D_E_U=m,D_precision=div(m[0][0],sum(x[0]for x in m)),D_recall=div(m[0][0],sum(m[0])),E_precision=div(m[1][1],sum(x[1]for x in m)),E_recall=div(m[1][1],sum(m[1])),U=m[0][2]+m[1][2],U_rate=div(m[0][2]+m[1][2],len(rows)))
def delta(rows,base):
 differences=[];tr=[[0]*3 for _ in range(3)];flips=loss=gain=0
 for r in rows:
  f=base[r['row_id']];a=f['prediction_D0_E1_Uminus1'];b=r['prediction_D0_E1_Uminus1'];t=r['known_target_D0_E1'];tr[a if a>=0 else 2][b if b>=0 else 2]+=1
  flips+=a!=b;loss+=(a==t and b!=t);gain+=(a!=t and b==t)
  if r['raw_logit']is not None and f['raw_logit']is not None:differences.append(r['raw_logit']-f['raw_logit'])
 absolute=sorted(abs(x)for x in differences)
 def quantile(p):
  k=(len(absolute)-1)*p;i=int(k);return absolute[i]+(absolute[min(i+1,len(absolute)-1)]-absolute[i])*(k-i)
 return dict(N=len(rows),finite_paired=len(differences),mean_delta=statistics.fmean(differences)if differences else None,mean_abs_delta=statistics.fmean(absolute)if absolute else None,median_abs_delta=statistics.median(absolute)if absolute else None,p95_abs_delta=quantile(.95)if absolute else None,max_abs_delta=max(absolute)if absolute else None,sign_flips=flips,transition_D_E_U=tr,factual_correct_changed_wrong=loss,factual_wrong_changed_correct=gain)
def main():
 complete=load('COMPLETE.json');assert bind('COMPLETE.json')['sha256']=='29872dc341ea656cc9525a256ba2c70392a9149282bfae5fe34cf788251cbd13'and complete['completed']and complete['PASS']
 names=['SUMMARY.json','ALL73680_SOURCE_TRAIN_CONDITION_SCALARS.jsonl','FULL_AND_NATIVE8_METRICS.csv','NATIVE8_METRICS.csv','PAIR_INPUT_MATCHES_BEFORE_SCORE_READS.json','PAIRED_SOURCE_VARIANT_CONDITION_CONTRASTS.jsonl']
 for name in names:assert bind(name)==complete['outputs'][name]
 summary=load('SUMMARY.json');arms=tuple(summary['models']);modes=tuple(summary['conditions']);assert len(arms)==2 and len(modes)==6
 rows=[json.loads(x)for x in read(names[1]).splitlines()];assert len(rows)==73680==complete['scalar_rows'];groups=defaultdict(list);identity={};lookup={}
 meta=('original_global_index','native_slot','query_xy','L','original_query_available','component_role','known_target_D0_E1')
 for r in rows:
  a,p,m,i=r['arm'],r['population'],r['condition'],r['row_id'];assert a in arms and p in('POINT','WEAK')and m in modes and r['component_role']=='TRAIN'and r['original_query_available']is True and 0<=r['native_slot']<8 and r['known_target_D0_E1']in(0,1)
  key=(a,p,m,i);assert key not in lookup;lookup[key]=r;groups[a,p,m].append(r)
  fields={k:r[k]for k in meta};assert identity.setdefault((p,i),fields)==fields
  assert r['prediction_D0_E1_Uminus1']==label(r['raw_logit'])
 assert len(identity)==6140 and Counter(p for p,i in identity)=={'POINT':6096,'WEAK':44}
 computed={};native={}
 for a in arms:
  for p,n in [('POINT',6096),('WEAK',44)]:
   factual=groups[a,p,modes[0]];base={r['row_id']:r for r in factual};assert len(base)==n
   assert Counter(r['known_target_D0_E1']for r in factual)==({0:5853,1:243}if p=='POINT'else{0:26,1:18})
   for m in modes:
    rr=groups[a,p,m];assert len(rr)==n and{r['row_id']for r in rr}==set(base)
    for r in rr:
     f=base[r['row_id']];assert r['factual_raw_logit']==f['raw_logit'];expected=None if r['raw_logit']is None or f['raw_logit']is None else float(np.float32(r['raw_logit'])-np.float32(f['raw_logit']));assert r['delta_from_factual']==expected
    z=cm(rr);z.update(delta(rr,base));z['by_native']={str(k):cm([r for r in rr if r['native_slot']==k])for k in range(8)};close(z,summary['models'][a][p][m]);computed[a,p,m]=z
    for k in range(8):native[a,p,m,k]=z['by_native'][str(k)]
 # All CSV cells independently match recomputed scalar metrics, including null P/R.
 for name,expected,key_fields in [('FULL_AND_NATIVE8_METRICS.csv',computed,('arm','population','condition')),('NATIVE8_METRICS.csv',native,('arm','population','condition','native_slot'))]:
  seen=set();rr=list(csv.DictReader(read(name).decode().splitlines()));assert len(rr)==len(expected)
  for r in rr:
   key=tuple(int(r[k])if k=='native_slot'else r[k]for k in key_fields);assert key not in seen;seen.add(key)
   values={k:(json.loads(v)if v else None)for k,v in r.items()if k not in key_fields};close(expected[key],values)
 assert summary['factual_replay']==complete['factual_replay'];replay_max=0
 for a in arms:
  for p,n in [('POINT',6096),('WEAK',44)]:
   r=summary['factual_replay'][a][p];assert r['rows']==n and r['signs_exact']is True and 0<=r['max_abs_logit_difference']<=1e-4;replay_max=max(replay_max,r['max_abs_logit_difference'])
 pairmeta=load('PAIR_INPUT_MATCHES_BEFORE_SCORE_READS.json');assert pairmeta['completed']and pairmeta['checkpoint_or_prediction_bodies_read_so_far']is False;pairs=pairmeta['pairs'];pairmap={}
 def pairkey(r):return r['population'],r['left_row_id'],r['right_row_id']
 paircounts={p:dict(all_pairs=0,exact_nonRGB_matched=0,matched_different_RGB=0,matched_same_RGB=0)for p in('POINT','WEAK')}
 for r in pairs:
  key=pairkey(r);assert key not in pairmap;pairmap[key]=r;p=r['population'];paircounts[p]['all_pairs']+=1
  assert r['exact_nonRGB_model_inputs_matched']==(r['exact_qL_native_othercontext']and r['exact_validity'])and r['same_target']==(r['left_target']==r['right_target'])
  for a in arms:
   left=groups[a,p,modes[0]][r['left_index']];right=groups[a,p,modes[0]][r['right_index']];assert (left['row_id'],right['row_id'])==(r['left_row_id'],r['right_row_id'])and (left['known_target_D0_E1'],right['known_target_D0_E1'])==(r['left_target'],r['right_target'])
   if r['exact_qL_native_othercontext']:assert all(left[k]==right[k]for k in('query_xy','L','native_slot'))
  if r['exact_nonRGB_model_inputs_matched']:paircounts[p]['exact_nonRGB_matched']+=1;paircounts[p]['matched_same_RGB'if r['exact_RGB']else'matched_different_RGB']+=1
 assert {p:{k:paircounts[p][k]for k in('all_pairs','exact_nonRGB_matched')}for p in paircounts}==pairmeta['counts']=={'POINT':{'all_pairs':4570,'exact_nonRGB_matched':2310},'WEAK':{'all_pairs':18,'exact_nonRGB_matched':18}}
 evidence=Counter();contrasts=[json.loads(x)for x in read('PAIRED_SOURCE_VARIANT_CONDITION_CONTRASTS.jsonl').splitlines()];assert len(contrasts)==len(pairs)*12==55056==complete['pair_contrast_rows'];seen=set()
 for r in contrasts:
  k=pairkey(r);a=r['arm'];m=r['condition'];assert (a,m,k)not in seen;seen.add((a,m,k));assert all(r[x]==v for x,v in pairmap[k].items())
  p=r['population'];left=lookup[a,p,m,r['left_row_id']];right=lookup[a,p,m,r['right_row_id']];assert r['left_logit']==left['raw_logit']and r['right_logit']==right['raw_logit'];difference=right['raw_logit']-left['raw_logit'];assert r['right_minus_left']==difference
  expected=m==modes[0]and r['exact_nonRGB_model_inputs_matched']and not r['exact_RGB']and abs(difference)>1e-4;assert r['source_RGB_dependence_evidence']==expected
  if expected:evidence[a,p]+=1
 assert all(computed[a,p,modes[1]]['CM_D_E_by_D_E_U']==([[0,5853,0],[0,243,0]]if p=='POINT'else[[0,26,0],[0,18,0]])for a in arms for p in('POINT','WEAK'))
 zero={a:{p:{k:computed[a,p,modes[-1]][k]for k in('sign_flips','mean_abs_delta','max_abs_delta','CM_D_E_by_D_E_U')}for p in('POINT','WEAK')}for a in arms};assert zero[arms[0]]['POINT']['sign_flips']==2 and zero[arms[0]]['WEAK']['sign_flips']==0 and all(zero[arms[1]][p]['sign_flips']==0 for p in('POINT','WEAK'))
 assert complete['NPY_interpreted_SOURCE_HOLDOUT_or_protected_rows']==0 and complete['model_optimizer_training_calls']==0 and summary['point_unknown944_not_forwarded']and summary['point_originalU64_not_forwarded']and summary['all7040_point_metadata_retained']and summary['weak44_separate']
 result=dict(completed=True,PASS=True,complete=bind('COMPLETE.json'),diagnostic_complete=bind('COMPLETE.json'),summary=bind('SUMMARY.json'),code=dict(path=str(Path(__file__)),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),saved_text_bindings=[bind(n)for n in names],checks=dict(scalar_rows=73680,full_metric_groups=24,native_metric_groups=192,pair_contrast_rows=55056,all_scalar_original_metadata_and_TRAIN_known_identity_consistent=True,scalar_fixedzero_decoder_and_F32_delta_exact=True,all_CM_P_R_U_delta_quantile_transition_and_CSV_cells_recomputed=True,all_mode_rowsets_same=True,zero_NONFINITE_U_predictions=sum(r['prediction_D0_E1_Uminus1']==-1 for r in rows)==0,factual_replay_saved_receipt_shape_signs_bound_checked=True,factual_replay_reported_max_abs=replay_max,pair_join_qL_native_and_contrast_values_exact=True,RGB127_both_models_all_E=True),pair_counts_from_saved_input_matching_flags=paircounts,factual_matched_different_RGB_logit_dependence_counts={a:{p:evidence[a,p]for p in('POINT','WEAK')}for a in arms},zero_all_conditions=zero,limits=['Only owned saved scalars/summary/CSV/pair texts were read; no source input, GT, old score arrays, checkpoint/model or image read, no forward/fit.','Factual replay maxima/signs are validated as matching runtime COMPLETE/SUMMARY receipts; source saved-logit arrays were not reopened or replayed here.','RGB/valid/context byte-equality flags come from saved pre-score input-matching metadata; this review checks their logic, row joins and counts, not source pixels independently.','6096 point and44 weak are SOURCE apparent-fit diagnoses, not independent validation; UNKNOWN944 and originalU64 were not forwarded and are retained as separate metadata statuses.','RGB127 all-E is sensitivity to an OOD intervention, not proof of real-transfer failure cause or image-only reasoning. Valid/grid and remaining paths persist.','No flips after condition-branch zero does not mean zero numerical branch contribution: raw-logit deltas are preserved. Native information may remain indirectly in RGB/context.','Matched-input RGB-difference/logit-difference evidence is bounded computational dependence on these SOURCE inputs, not semantic native gold, new labels, model adoption, threshold choice or goal completion.'],model_input_or_weights_or_arrays_or_images_opened=0,neural_forward_or_fit=0,selected_model=None,adopted=False,fully_solved=False)
 with(O/'REVIEW.json').open('x')as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(dict(completed=True,PASS=True,review=dict(path=str(O/'REVIEW.json'),sha256=hashlib.sha256((O/'REVIEW.json').read_bytes()).hexdigest()),pair_counts=paircounts,zero_all=zero)))
if __name__=='__main__':main()
