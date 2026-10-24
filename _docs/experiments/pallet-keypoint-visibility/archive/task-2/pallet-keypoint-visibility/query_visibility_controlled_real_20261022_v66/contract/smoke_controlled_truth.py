"""Exhaustive tiny-array completion proof; no real images/masks/private data."""
import itertools,json,hashlib
from pathlib import Path
import numpy as np
from controlled_truth import U,V,E,O,JOINT_PAIRS,construct_certainty,query_truth

cases=0;completions=0
for shape in[(1,1),(2,2)]:
 n=int(np.prod(shape))
 # Original states V=observedvisible, E=hiddenfullproxy(unknowncontrolled), O=outside.
 for original_states in itertools.product([V,E,O],repeat=n):
  original=np.array(original_states).reshape(shape);visible=original==V;full=original!=O
  for pasted in itertools.product([False,True],repeat=n):
   paste=np.array(pasted,bool).reshape(shape);cert=construct_certainty(visible,full,paste)
   assert np.all(cert['exact_state'][original==E]==U)
   assert np.array_equal(cert['controlled_E_pixel'],visible&paste)
   unknown=np.flatnonzero((original==E).ravel());expected=set()
   for bits in itertools.product([False,True],repeat=len(unknown)):
    factual_full=visible.copy().ravel();factual_full[unknown]=bits;factual_full=factual_full.reshape(shape)
    factual_visible=visible&~paste
    exact=V if factual_visible[0,0]else E if factual_full[0,0]else O
    near=V if factual_visible.any()else E if factual_full.any()else O
    expected.add(JOINT_PAIRS.index((exact,near)));completions+=1
   result=query_truth(cert,[0.,0.]);assert result['joint_compatible_codes']==sorted(expected),(original,paste,result,expected)
   assert result['exact_state']==(next(iter({JOINT_PAIRS[x][0]for x in expected}))if len({JOINT_PAIRS[x][0]for x in expected})==1 else U)
   assert result['radius2_state']==(next(iter({JOINT_PAIRS[x][1]for x in expected}))if len({JOINT_PAIRS[x][1]for x in expected})==1 else U)
   cases+=1
# Unknown-full center alone cannot be promoted to O; exact and near share it.
v=np.zeros((5,5),bool);f=v.copy();f[2,2]=True;p=v.copy()
r=query_truth(construct_certainty(v,f,p),[2,2]);assert r['exact_state']==r['radius2_state']==U and r['joint_compatible_codes']==[2,5]
# Known newly occluded pixel proves full-positive; unknown-full neighbors block
# O but do not block E when all visibility bits are source-known negatives.
v[2,1]=True;f[2,1]=True;p[2,1]=True
r=query_truth(construct_certainty(v,f,p),[2,2]);assert r['exact_state']==U and r['radius2_state']==E and r['joint_compatible_codes']==[2,4]
strict=query_truth(construct_certainty(v,f,p,visible_negative_trusted=False),[2,2])
assert strict['radius2_state']==U # untrusted visible negatives propagate uncertainty
v[:]=True;f[:]=True;p[:]=True
r=query_truth(construct_certainty(v,f,p),[2,2]);assert r['radius2_strict_new_controlled_E_core']and r['joint_state']==2
assert not query_truth(construct_certainty(v,f,p),[0,0])['radius2_strict_new_controlled_E_core']
assert query_truth(construct_certainty(v,f,p),[4.6,0])['valid']is False
out=dict(schema='V66_CONTROLLED_PARTIAL_TRUTH_EXHAUSTIVE_MOCK_PROOF_V1',PASS=True,
         tiny_array_patterns=cases,factual_full_completions=completions,
         checks=['all hidden-full proxies remain exactU','all possible joint completions exactly matched',
                 'unknown is not negative/full0','shared exact/near correlation respected','knownpositive existential E versus unknownfull',
                 'untrusted visibility alternative staysU','strict25pixel newlyoccludedcore','clipping and nearest validity'],
         source_or_private_RGB_masks_labels_scores_reads=0,new_models_GPU_training_PnP_calls=0,
         code_sha256=hashlib.sha256((Path(__file__).parent/'controlled_truth.py').read_bytes()).hexdigest())
(Path(__file__).parent/'CONTROLLED_TRUTH_MOCK_PROOF.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
