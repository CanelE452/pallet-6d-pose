"""Saved-feature values only. No encoder/head/weights/optimizer/image access."""
from pathlib import Path
import json,hashlib,collections,time
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;RUN=V/'runs/C20_C21_FIXED1000_SOURCE_CONTROL'
def read(p):return json.loads(Path(p).read_text())
def bind(p):return dict(path=str(Path(p).resolve()),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def quant(x):
 a=np.asarray(x,np.float64);return dict(count=len(a),quantile_0_25_50_75_95_100=np.quantile(a,[0,.25,.5,.75,.95,1]).tolist())if len(a)else dict(count=0)
def clean(x):
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,dict):return{k:clean(v)for k,v in x.items()}
 if isinstance(x,(tuple,list)):return[clean(v)for v in x]
 return x
def main():
 start=time.monotonic();entry=read(V/'ROOT_ACTUAL_C20_C21_ENTRY.json');p=read(entry['protocol']['path'])
 with np.load(p['old_features']['path'],allow_pickle=False)as z:old=z['features'];omap=z['occurrence_unit']
 with np.load(entry['new_features']['path'],allow_pickle=False)as z:new=z['features'];names=z['corpus'];sourceindex=z['source_occurrence_index']
 with np.load(RUN/'C21_PUBLIC_PLUS_NATIVE/FINAL_ALL61110_SOURCE_SCORES.npz',allow_pickle=False)as z:
  sc={k:z[k]for k in('external_target','raw_state','split','corpus','query_available','source_occurrence_index','feature_unit_index','prediction','query_logit','occurrence_id')}
 assert old.shape==(4428,1928)and new.shape==(56214,1928)and old.dtype==new.dtype==np.float32
 assert np.isfinite(old).all()and np.isfinite(new).all()
 assert np.array_equal(sc['feature_unit_index'],np.r_[omap,4428+np.arange(56214)])
 assert np.array_equal(names,sc['corpus'][4896:])and np.array_equal(sourceindex,sc['source_occurrence_index'][4896:])
 y=sc['external_target'];train=(sc['split']=='TRAIN')&sc['query_available']&(y>=0)
 def f(i):return old[omap[i]]if i<4896 else new[i-4896]
 groups={}
 for i in np.flatnonzero(train):
  a=f(i);key=hashlib.sha256(a.tobytes()).hexdigest();g=groups.setdefault(key,dict(D=[],E=[]));g['D'if y[i]==0 else'E'].append(int(i))
 cross=[dict(feature_SHA256=k,**v)for k,v in groups.items()if v['D']and v['E']]
 lb=sum(min(len(g['D']),len(g['E']))for g in cross)
 # Source-native formula is frozen: sourcei=(((base*4+variant)*9+condition)*8+slot).
 native_new=np.flatnonzero(names=='NATIVE');inv=np.empty(55296,np.int64);inv[sourceindex[native_new]]=native_new
 pairs=[];blocks={k:[]for k in('T','Q','F','C','P')}
 targets=np.flatnonzero(train&(sc['corpus']=='NATIVE')&(y==1))
 for i in targets:
  s=int(sc['source_occurrence_index'][i]);vi=(s//72)%4;before=s-72*vi;j=int(4896+inv[before])
  if y[j]!=0 or not sc['query_available'][j]:continue
  a=f(int(i)).astype(np.float64);b=f(j).astype(np.float64);na=np.linalg.norm(a);nb=np.linalg.norm(b)
  cos=float(np.dot(a,b)/(na*nb))if na and nb else None
  rec=dict(E_global_index=int(i),before_D_global_index=j,E_source_occurrence=s,before_source_occurrence=before,
   base_index=s//288,condition_index=(s%72)//8,CF_slot=s%8,variant_index=vi,
   whole_feature_cosine=cos,whole_feature_l2=float(np.linalg.norm(a-b)),maximum_absolute_difference=float(np.max(np.abs(a-b))),
   exact_feature_equal=bool(np.array_equal(a,b)),C21_E_logit=float(sc['query_logit'][i]),C21_before_D_logit=float(sc['query_logit'][j]),
   E_Q_inside=bool(a[1927]),before_Q_inside=bool(b[1927]))
  for k,lo in zip(('T','Q','F','C','P'),range(0,1920,384)):
   aa=a[lo:lo+384];bb=b[lo:lo+384];n1=np.linalg.norm(aa);n2=np.linalg.norm(bb)
   if n1 and n2:blocks[k].append(float(np.dot(aa,bb)/(n1*n2)))
  pairs.append(rec)
 norms={}
 for corpus in('OLD_PUBLIC','PUBLIC_AUX','NATIVE'):
  for label in(0,1):
   ix=np.flatnonzero(train&(sc['corpus']==corpus)&(y==label));block={}
   for name,lo in zip(('T','Q','F','C','P'),range(0,1920,384)):
    ns=[float(np.linalg.norm(f(int(i))[lo:lo+384].astype(np.float64)))for i in ix]
    block[name]=quant(ns)
   norms[corpus+'::'+('D'if label==0 else'E')]=dict(rows=len(ix),block_norms=block)
 output=dict(completed=True,PASS=True,scope='Saved public features and saved labels/scores only; no model, parameter, image, private or V112 access.',code=bind(__file__),
  feature_inputs=dict(old=p['old_features'],new=entry['new_features']),TRAIN_known_available_rows=int(train.sum()),
  exact_F32_feature_collision=dict(unique_hashes=len(groups),opposite_D_E_groups=len(cross),minimum_errors_for_any_deterministic_classifier_due_to_these_collisions=lb,
   groups=cross[:12],all_groups_listed=len(cross)<=12,method='SHA256 of complete FP32[1928] input bytes, no rounded/quantized similarity treated as identity'),
  native_TRAIN_E_vs_same_base_condition_slot_BEFORE_D=dict(pairs=len(pairs),exact_equal=sum(r['exact_feature_equal']for r in pairs),
   cosine=quant([r['whole_feature_cosine']for r in pairs if r['whole_feature_cosine']is not None]),
   feature_l2=quant([r['whole_feature_l2']for r in pairs]),maximum_absolute_difference=quant([r['maximum_absolute_difference']for r in pairs]),
   per_block_cosine={k:quant(v)for k,v in blocks.items()},
   closest_first12=sorted(pairs,key=lambda r:(-r['whole_feature_cosine'],r['E_global_index']))[:12]),
  feature_block_norms=norms,
  limits=['Exact-collision absence does not prove this head can separate all labels, and high cosine does not prove a bottleneck or impossibility.',
   'Near-pair analysis is restricted to corresponding native E versus BEFORE D at identical source base/condition/slot; it is not a global nearest-neighbor classifier or new fitted model.',
   'Normalized block norms and saved producer parity are input integrity evidence, not independent encoder replay or semantic-gold certification.'],
  seconds=time.monotonic()-start,model_weight_image_current_V112_calls=0)
 out=H/'SAVED_FEATURE_COLLISION_CHECK.json'
 with out.open('x')as z:json.dump(clean(output),z,ensure_ascii=False,indent=2,allow_nan=False);z.write('\n')
 print(json.dumps(dict(output=bind(out),exact_collisions=output['exact_F32_feature_collision'],paired=output['native_TRAIN_E_vs_same_base_condition_slot_BEFORE_D'],seconds=output['seconds'])))
if __name__=='__main__':main()
