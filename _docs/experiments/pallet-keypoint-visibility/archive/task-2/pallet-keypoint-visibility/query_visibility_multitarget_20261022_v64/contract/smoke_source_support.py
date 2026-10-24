"""Generated mask fixtures only; no current private input or score is opened."""
import hashlib,json
from pathlib import Path
import numpy as np
from source_support_contract import *

HERE=Path(__file__).resolve().parent
for j,(a,b)in enumerate(JOINT_STATES):
 f=np.zeros((5,5),bool);v=f.copy()
 if a!=O:f[2,2]=True
 if a==V:v[2,2]=True
 if b==V:f[2,3]=True;v[2,3]=True
 if b==E:f[2,3]=True
 lab=query_labels(f,v,[2.,2.]);assert lab['classes']==[a,b]and lab['joint']==j
 try:joint_codes(np.array([V]),np.array([O]));raise AssertionError('Impossible joint accepted')
 except ValueError:pass
assert np.array_equal(joint_codes(np.array([9]),np.array([9]),valid=[False]),[-1])
assert not query_labels(np.zeros((5,5)),np.zeros((5,5)),[4.7,0.])['valid']
assert not query_labels(np.zeros((5,5)),np.zeros((5,5)),[-.1,0.])['valid']
assert np.array_equal(decode_mask(np.array([[0,1,2,254,255]],np.uint8),'SYNTH'),[[False,False,False,True,True]])
assert np.array_equal(decode_mask(np.array([[0,255]],np.uint8),'YCB'),[[False,True]])
f=np.ones((9,9),bool);v=np.zeros_like(f);other=v.copy();other[4,4]=True
assert external_evidence(f,v,[4.,4.],other_visible=other)['evidence']=='EXTERNAL_OTHER_VISIBLE_EVIDENCE'
assert external_evidence(f,v,[4.,4.],measured_depth=np.ones((9,9)))['evidence']=='EXTERNAL_RENDER_ONLY_INTERIOR_UNVERIFIED'
assert external_evidence(f,v,[4.,4.],measured_depth=np.ones((9,9)),target_render_depth=np.full((9,9),2.),depth_tolerance=.1)['evidence']=='EXTERNAL_DEPTH_FRONT_EVIDENCE'
assert external_evidence(f,v,[4.,4.],measured_depth=np.full((9,9),3.),target_render_depth=np.full((9,9),2.),depth_tolerance=.1)['evidence']=='EXTERNAL_CONFLICTING_DEPTH'
f[4,5]=False
assert external_evidence(f,v,[4.,4.])['evidence']=='EXTERNAL_AMBIGUOUS_RENDER_BOUNDARY'
rng=np.random.default_rng(20261022);p=rng.dirichlet(np.ones(6),size=1000);m=joint_marginals(p)
assert np.all(m[:,0,V]<=m[:,1,V]+1e-12)
assert np.all(m[:,0,V]+m[:,0,E]<=m[:,1,V]+m[:,1,E]+1e-12)
assert np.allclose(m.sum(-1),1)
result=dict(schema='V64_SOURCE_SUPPORT_RULE_MOCK_SMOKE_V1',six_joint_state_fixtures_pass=True,impossible_pairs_rejected=True,native_continuous_and_nearest_validity_pass=True,source_codec_rules_pass=True,other_visible_and_paired_depth_evidence_pass=True,measured_depth_alone_not_promoted=True,no_ambiguous_label_relabeling=True,random_softmax_marginal_checks=1000,semantic_self_generated=False,current_private_inputs_or_scores_read=0,new_PnP_or_training_or_GPU=0,code_sha256=hashlib.sha256((HERE/'source_support_contract.py').read_bytes()).hexdigest())
(HERE/'SOURCE_SUPPORT_SMOKE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
