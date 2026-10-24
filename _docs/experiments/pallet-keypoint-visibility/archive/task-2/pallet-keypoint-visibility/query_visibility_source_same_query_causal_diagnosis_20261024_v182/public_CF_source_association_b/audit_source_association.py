"""V182 saved sourceTRAIN-only association; no pixels/features/weights/forward.
Owner training response is not native/end-support accuracy.
"""
from pathlib import Path
from collections import Counter
import hashlib,json,sys
import numpy as np
H=Path(__file__).resolve().parent; V=H.parent; P=V.parent
SCOPE=V/'ROOT_SCOPE_BEFORE_NEW_COMMON_QUERY_OR_PUBLIC_PIXELS.json'
OLD=P/'query_visibility_local_support_raw_head_20261024_v181'
FIT=OLD/'runs/C50_C51_LOCAL_SUPPORT_FIXED8000_v2/COMPLETE.json'
CACHE=OLD/'source_raw96_cache_v1/COMPLETE.json'
PROOFS=P/'query_visibility_public_train_owner_counterfactual_20261024_v161/source_pixels_v1/ALL159_PIXEL_INPUT_PROOFS.jsonl'
ARMS=('C50_RAW96_DUAL_SUPPORT_OWNER_AUX','C51_RAW96_PLUS_FROZEN_SAM_DUAL_SUPPORT_OWNER_AUX')
DOMAINS={'PUBLIC_TRAIN1480':1480,'CF3816':3816}
VARIANTS=('BEFORE','COVER','ADJACENT_SHAM')
def sha(p):
 p=Path(p).resolve(); assert not any(s in str(p).lower() for s in ('_v112','_v169','_v177','_v178','private','current223','synthetic_val','final_synth_val'))
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def check(b):assert bind(b['path'])==b;return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def load(b,keys):
 with np.load(check(b),allow_pickle=False) as z:return {k:z[k].copy() for k in keys}
def same(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def association(y,a):
 result={}
 for c,name in enumerate(('TARGET','FOREIGN','BACKGROUND')):
  ii=np.flatnonzero(y==c);j=Counter();p=Counter()
  for i in ii:
   s=a['support_prediction'][i];j['U' if (s<0).any() else str(int(s[0]))+str(int(s[1]))]+=1
   p[str(int(a['regional_support_proxy_D0_E1_Uminus1'][i]))]+=1
  result[name]={'N':len(ii),'joint_order_00_10_01_11_U':[j[k] for k in ('00','10','01','11','U')],'proxy_order_D_E_U':[p[k] for k in ('0','1','-1')]}
 return result
def main():
 assert sha(SCOPE)=='142ba2fd62dfa16c79bda690bf90814d0a9cdd330898722b8f896edf76b396a1'
 assert not (H/'REVIEW.json').exists()
 fit=read(FIT);cache=read(CACHE);assert fit['completed'] and fit['PASS'] and fit['all2_frozen_before_outputs_or_VAL_access'];assert fit['training_cache_complete']==bind(CACHE)
 proofs=lines(PROOFS);assert len(proofs)==159 and len({r['pair_id'] for r in proofs})==159
 assert all(r['split']=='SOURCE_DEVELOPMENT' and r['attempted_once'] and not r['replacement_or_retry'] for r in proofs)
 pr={r['pair_id']:r for r in proofs};targets={};meta={};outputs={};bindings=[bind(SCOPE),bind(FIT),bind(CACHE),bind(PROOFS)];summary={}
 keys=('raw_support_logits','raw_owner_logits','support_prediction','owner_prediction','regional_support_proxy_D0_E1_Uminus1','row_id','original_global_index','native_slot','query_xy','L','original_query_available')
 for d,n in DOMAINS.items():
  dd=cache['domains'][d];targets[d]=load(dd['targets'],('support_target','owner_target'));meta[d]=lines(check(dd['metadata']));bindings.extend((dd['targets'],dd['metadata']))
  y=targets[d]['owner_target'];assert y.shape==(n,) and len(meta[d])==n and (targets[d]['support_target']==-1).all()
  assert all(r['split']=='SOURCE_DEVELOPMENT' and r['new_native_visibility_target']==r['new_point_quality_target']==-1 for r in meta[d])
  summary[d]={'rows':n,'input_U':sum(not r['original_query_available'] for r in meta[d]),'owner_counts_T_F_BG_U':[int((y==c).sum()) for c in (0,1,2,-1)],'regional_support_GT_known':0,'arms':{}}
  for arm in ARMS:
   b=fit['arms'][arm]['outputs'][d];a=load(b,keys);bindings.append(b);outputs[arm,d]=a
   assert len(a['row_id'])==n and a['row_id'].tolist()==[r['row_id'] for r in meta[d]]
   for k,dtype in (('query_xy',np.float64),('L',np.float64),('original_global_index',np.int64),('native_slot',np.int8),('original_query_available',bool)):
    m=np.asarray([r[k] for r in meta[d]],dtype=dtype);assert same(a[k],m),k
   available=a['original_query_available'];assert (a['owner_prediction'][~available]==-1).all() and (a['support_prediction'][~available]==-1).all()
   assert np.isnan(a['raw_owner_logits'][~available]).all() and np.isnan(a['raw_support_logits'][~available]).all()
   assert np.array_equal(a['owner_prediction'][available],a['raw_owner_logits'][available].argmax(1))
   assert np.array_equal(a['support_prediction'][available],(a['raw_support_logits'][available]>0).astype(np.int8))
   pred=np.full(n,-1,np.int8);pred[available & (a['support_prediction'][:,0]==1)]=0;pred[available & (a['support_prediction'][:,1]==1)]=1
   assert same(pred,a['regional_support_proxy_D0_E1_Uminus1'])
   cm=np.zeros((3,4),np.int64)
   for i in np.flatnonzero(y>=0):cm[y[i],int(a['owner_prediction'][i]) if a['owner_prediction'][i]>=0 else 3]+=1
   summary[d]['arms'][arm]={'owner_TRAIN_apparent_CM_T_F_BG_to_T_F_BG_U':cm.tolist(),'owner_to_support_association_NOT_support_accuracy':association(y,a)}
 groups={}
 for i,r in enumerate(meta['CF3816']):
  o=r['original'];groups.setdefault(o['pair_id'],{}).setdefault(o['variant'],[]).append(i)
 assert len(groups)==159 and set(groups)==set(pr)
 transitions={a:{'PASS':Counter(),'FAILED_RETAINED':Counter()} for a in ARMS};deltas={a:{'PASS':[],'FAILED_RETAINED':[]} for a in ARMS};owner_transitions={a:Counter() for a in ARMS}
 fkeys=('feature_constructed','query_available','nearest_valid','central4_available','context_available')
 native=Counter();frames=set();anchors=set();failed=[]
 for pair,g in groups.items():
  p=pr[pair];assert set(g)==set(VARIANTS)
  for v,ii in g.items():
   ii.sort(key=lambda i:meta['CF3816'][i]['native_slot']);assert [meta['CF3816'][i]['native_slot'] for i in ii]==list(range(8))
   assert all(meta['CF3816'][i]['original']['variant_payload']==p['variants'][v] for i in ii)
  bb=g['BEFORE']
  for v in VARIANTS[1:]:
   for k in ('query_xy','L','native_slot','original_global_index','original_query_available'):
    a=np.asarray([meta['CF3816'][i][k] for i in bb]);b=np.asarray([meta['CF3816'][i][k] for i in g[v]]);assert same(a,b),(pair,k)
   for k in fkeys:
    a=np.asarray([meta['CF3816'][i]['original'][k] for i in bb],bool);b=np.asarray([meta['CF3816'][i]['original'][k] for i in g[v]],bool);assert same(a,b)
   for arm in ARMS:
    for k in ('query_xy','L','native_slot','original_global_index','original_query_available'):assert same(outputs[arm,'CF3816'][k][bb],outputs[arm,'CF3816'][k][g[v]])
  slot=p['native_slot'];ii=[g[v][slot] for v in VARIANTS];o=meta['CF3816'][ii[0]]['original'];assert o['original_query_id']==p['anchor_query_id'] and o['query_xy']==p['original_q'] and o['L']==p['original_L']
  if p['PASS']:
   assert all(p['proof'].values());assert targets['CF3816']['owner_target'][ii].tolist()==[-1,1,0]
   native[slot]+=2;frames.add(o['original_frame_index']);anchors.add(p['anchor_query_id'])
  else:
   assert (targets['CF3816']['owner_target'][sum(g.values(),[])]==-1).all();failed.append({'pair_id':pair,'reason':p['reason'],'failed_proof_keys':[k for k,v in p['proof'].items() if not v]})
  bucket='PASS' if p['PASS'] else 'FAILED_RETAINED'
  for arm in ARMS:
   a=outputs[arm,'CF3816'];ss=a['support_prediction'][ii];transitions[arm][bucket]['|'.join(''.join(str(int(x)) for x in row) for row in ss)]+=1
   owner_transitions[arm][bucket+':'+','.join(str(int(x)) for x in a['owner_prediction'][ii])]+=1
   raw=a['raw_support_logits'][ii].astype(np.float64);deltas[arm][bucket].append([raw[1,0]-raw[0,0],raw[1,1]-raw[0,1],raw[2,0]-raw[0,0],raw[2,1]-raw[0,1]])
 assert len(anchors)==53 and len(frames)==46 and len(failed)==2 and summary['PUBLIC_TRAIN1480']['input_U']==194 and summary['CF3816']['input_U']==360
 result=dict(completed=True,PASS=True,sourceTRAIN_only=True,code=bind(__file__),scope=bind(SCOPE),inputs=bindings,domains=summary,CF_same_q_invariants=dict(all159_pairs_all8_slots_all3_variants=True,original_and_saved_head_q_L_native_global_available_byte_equal=True,five_original_input_flags_byte_equal=True,all477_payloads_retained=True,failed_pairs=failed),CF_selected_known314=dict(original_anchors53=len(anchors),original_frames46=len(frames),native_slot_known_counts={str(k):native[k] for k in range(8)},before_owner_label_remains_UNKNOWN=True),selected_anchor_response_NOT_native_or_regional_accuracy={a:dict(owner_transition_counts=dict(owner_transitions[a]),support_joint_transition_counts={b:dict(c) for b,c in transitions[a].items()},mean_raw_logit_deltas_COVERminusBEFORE_V_F_SHAMminusBEFORE_V_F={b:np.mean(x,axis=0).tolist() for b,x in deltas[a].items()}) for a in ARMS},interpretation_KO=['공개 TRAIN owner253과 CF owner314는 두 head 모두 학습된 owner 라벨에 완전 일치한다. 이는 TRAIN apparent 응답이며 전이 성공이 아니다.','같은 q의 COVER/SHAM에서 owner와 support 출력의 연관성은 확인되지만 public/CF regional support 정답은 모두 UNKNOWN이다. FOREIGN을 native 외부가림, TARGET을 native 보임으로 대체할 수 없다.','CF는53원anchor·46frame의 상관된 복제이며 selected native2/3은 없다. 전체 WOOD/SQUARE/native8 목표와 실패·UNKNOWN은 보존한다.','다음 sourceTRAIN 작업은 동일 q의 의도한 끝 부위 지지 근거를 분리해야 하며 owner 학습 반복만으로 native 문제 해결을 주장할 수 없다.'],read_boundary=dict(opened='V182 scope, V181 completed fit/cache JSON metadata, public/CF saved outputs+owner/support targets+rows, V161 proof JSONL only',not_opened='No RGB/crop/raw input NPZ, features, weights, model forward, sourceVAL results/arrays, held/private/current/V112'),new_labels=0,new_fit_or_forward=0,training_admission=False,selected_model=None,fully_solved=False)
 with (H/'REVIEW.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'completed':True,'PASS':True,'review':bind(H/'REVIEW.json')},ensure_ascii=False))
if __name__=='__main__':main()
