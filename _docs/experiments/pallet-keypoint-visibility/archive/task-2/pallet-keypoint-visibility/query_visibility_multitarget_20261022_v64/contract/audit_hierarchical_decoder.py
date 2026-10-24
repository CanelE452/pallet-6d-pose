"""One million generated joint posteriors; no source/private data are opened."""
import hashlib,json
from pathlib import Path
import numpy as np
from source_support_contract import *

HERE=Path(__file__).resolve().parent
counter=np.array([.36,.01,.32,0,.31,0])
m=joint_marginals(counter)
assert tuple(m.argmax(-1))==(V,E)
assert tuple(hierarchical_states(counter))==(E,E)
assert JOINT_STATES[int(counter.argmax())]==(V,V)
rng=np.random.default_rng(20261022);counts=np.zeros(6,np.int64);total=0;accepted_total=0
for alpha in[.1,1.,10.,100.]:
 p=rng.dirichlet(np.full(6,alpha),size=250000)
 h=hierarchical_states(p);j=joint_codes(h[:,0],h[:,1])
 assert(j>=0).all();counts+=np.bincount(j,minlength=6);total+=len(p)
 cm=confident_joint_map(p);accepted_total+=int(cm['accepted'].sum())
 assert np.all(cm['classes'][~cm['accepted']]==-1)
# Exact .5 ties and pure boundary posteriors use the registered >= convention.
edges=np.array([[.5,0,0,0,0,.5],[0,.5,0,0,0,.5],[0,0,.5,0,0,.5],[0,0,0,.5,0,.5],[0,0,0,0,.5,.5],*np.eye(6)])
h=hierarchical_states(edges);assert(joint_codes(h[:,0],h[:,1])>=0).all()
assert tuple(h[0])==(V,V)and tuple(h[1])==(E,V)and tuple(h[2])==(E,E)
cm=confident_joint_map(edges);assert cm['accepted'].all()and cm['joint'][0]==0
assert confident_joint_map(counter)['joint']==-1
logical_low_E=np.array([.43,.12,0,0,0,.45])
assert tuple(hierarchical_states(logical_low_E))==(E,V)and confident_joint_map(logical_low_E)['joint']==-1
result=dict(schema='V64_HIERARCHICAL_MECHANISM_DECODER_AUDIT_V2',generated_joint_draws=total,dirichlet_alphas=[.1,1.,10.,100.],all_pairs_in_six_valid_states=True,decoded_joint_counts=counts.tolist(),half_threshold_edge_cases=len(edges),counterexample=dict(joint=counter.tolist(),independent_marginal_argmax=[V,E],registered_hierarchical=[E,E],joint_MAP=[V,V]),logical_support_decoder='V ifpVisible>=.5; elseE ifpFull>=.5; elseO at eachscale, using nested sums',joint_MAP='unqualified six-state diagnostic; user-facing mechanism additionally requires maxjoint>=.5',confidence='User-facing jointMAP ifmaxjoint>=.5,elsepairU; source confidence heuristic, not calibrated current-pallet reliability',accepted_jointMAP_draws=accepted_total,accepted_jointMAP_scale_marginals_at_least_half=True,exact_half_joint_ties_accept_first_index=True,low_pE_logical_state_counterexample_withheld_by_joint_confidence=True,primary_binary='pVisible>=.5 unchanged for visible-vs-nonvisible transfer comparison',logical_hard_decoder_role='paired support diagnostic only',current_private_or_source_data_reads=0,new_GPU_training_or_PnP=0,code_sha256=hashlib.sha256((HERE/'source_support_contract.py').read_bytes()).hexdigest())
(HERE/'HIERARCHICAL_DECODER_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
