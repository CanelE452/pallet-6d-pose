"""独立replayの小さいmemory fixturesのみ。actualSOURCE/payload/producer未実行。"""
import ast
import hashlib
from pathlib import Path
import json
import types
import numpy as np
from independent_scene_math import lift_scene,footprints,teacher_proof,sample_all,sha
from qa_io import LRU,lexical

P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')

def pure(path,pin,names):
    b=path.read_bytes();assert hashlib.sha256(b).hexdigest()==pin
    nodes=[x for x in ast.parse(b).body if isinstance(x,ast.FunctionDef) and x.name in names]
    env={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    return types.SimpleNamespace(**{x:env[x] for x in names})

crop=pure(P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py',
 '6a6c2543e28715e3d47e15c543c25f75ff7d3efa3b881e523bd3451817fc004b',['crop']).crop
context=pure(P/'query_visibility_target_context_rgb_20261023_v107/target_context_math.py',
 'aa7cd8e2568c8b7d30827e8681b4b87899944bd23cfa1b27948ceda5b55d07f3',
 ['query_extent','valid_box','target_context_grid','sample_target_context'])
base=(np.arange(80*100*3).reshape(80,100,3)%251).astype(np.uint8)
carrier=np.zeros((3,96,96),np.uint8);carrier[:,40:56,40:56]=233
alpha=np.zeros((96,96),bool);alpha[40:56,40:56]=True
q=np.array([40.5,40.5],np.float64);L=np.float64(48)
scene,native=lift_scene(base,carrier,alpha,q,L)
# Scalar inverse-address oracle catches exact nearest-cell boundaries/order.
oracle=base.copy();oa=np.zeros(base.shape[:2],bool)
for y in range(len(base)):
 for x in range(base.shape[1]):
  u=(x-q[0])*96./L+47.5;v=(y-q[1])*96./L+47.5
  if -.5<=u<95.5 and -.5<=v<95.5:
   xx=int(np.floor(u+.5));yy=int(np.floor(v+.5))
   if alpha[yy,xx]:oracle[y,x]=carrier[:,yy,xx];oa[y,x]=True
assert np.array_equal(scene,oracle) and np.array_equal(native,oa)
sham_rgb=np.zeros_like(carrier);sham_rgb[:,5:21,5:21]=233
sham_a=np.zeros_like(alpha);sham_a[5:21,5:21]=True
sham,sa=lift_scene(base,sham_rgb,sham_a,q,L)
proof=teacher_proof(base,scene,sham,native,sa,[40.5,40.5]);assert proof['PASS']
assert not teacher_proof(base,base,sham,np.zeros_like(native),sa,[40.5,40.5])['PASS']
edge=footprints([99.,79.],base.shape[:2])
assert edge['continuous_reference']['supported'] and len(edge['continuous_reference']['pixels'])==1
assert not edge['nearest_half_up_3x3']['supported']
assert not footprints([99.1,79.],base.shape[:2])['continuous_reference']['supported']
queries=np.array([[36.+i,35.+i/2] for i in range(8)],np.float64)
group=dict(q8=queries.tolist(),L8=[48.]*8,original_available8=[True]*7+[False],box_xyxy=context.query_extent(queries))
b=sample_all(base,group,crop,context);c=sample_all(scene,group,crop,context)
assert np.array_equal(b['node_valid'],c['node_valid']) and np.array_equal(b['context_valid'],c['context_valid'])
assert not b['node_support'][-1] and np.count_nonzero(b['nodes'][-1])>0 # finite unavailable q not deleted
assert sha(c['nodes'])!=sha(b['nodes']) and sha(c['context'])!=sha(b['context'])
cache=LRU(32);cache.get('a',lambda:np.ones(20,np.uint8));cache.get('b',lambda:np.ones(20,np.uint8))
assert list(cache.items)==['b'] and cache.peak<=32
for bad in ['/invented/V112/x','/invented/__pycache__/a.pyc']:
 try:lexical(bad)
 except ValueError:pass
 else:raise AssertionError('bad lexical accepted')
print(json.dumps(dict(completed=True,PASS=True,invented_groups=5,
 checks=['scalar nearest-cell lift','positive-only taps/halfup3x3 +edgeUNKNOWN','renewed teacher proof positive/negative',
 'all8/V90+V107 same support/finite unavailable node retained','bounded LRU/lexical rejection'],
 actual_SOURCE_RGB_array_GT_weight_model_or_dataset_reads=0)))
