"""Generated rows only; never opens current scores, inputs or GT."""
import json
from pathlib import Path
import tempfile
import numpy as np
from evaluate_sealed_current import complete_gate,validate_scores
from source_support_contract import joint_marginals,confident_joint_map

rows=[]
for i,p in enumerate([*np.eye(6),np.full(6,1/6),np.array([.5,0,0,0,0,.5])]):
    p=np.asarray(p);m=joint_marginals(p);c=confident_joint_map(p)
    row=dict(id='MOCK',native_corner=str(i),valid='True',p_visible=str(p[0]),confidence_U=str(not bool(c['accepted'])))
    row.update({'joint_p'+str(j):str(x)for j,x in enumerate(p)})
    for scale,name in enumerate(['exact','radius2']):
        row.update({'p_visible_'+name:str(m[scale,0]),'p_external_support_'+name:str(m[scale,1]),
                    'p_outside_full_'+name:str(m[scale,2]),'p_full_'+name:str(m[scale,:2].sum())})
    rows.append(row)
rows.append(dict(id='MOCK',native_corner='8',valid='False',p_visible=''))
assert validate_scores(rows)==dict(rows=9,valid=8,invalid=1)
bad=[dict(rows[-1],p_visible='.3')]
try:validate_scores(bad);raise AssertionError('Invalid main score accepted')
except AssertionError as e:assert str(e)!='Invalid main score accepted'
with tempfile.TemporaryDirectory(dir=Path(__file__).parent)as tmp:
    p=Path(tmp)/'gate.json';p.write_text(json.dumps(dict(approved=False)))
    try:complete_gate(p,Path(tmp)/'NO_SEAL',Path(tmp)/'NO_SCORES');raise AssertionError('Unapproved gate accepted')
    except PermissionError:pass
result=dict(PASS=True,checks=['six joint probability/marginal fixtures','uniform U','half tie accepted',
                              'invalid main score rejected','unapproved gate rejected before missing seal/score access'],
            current_private_GT_inputs_scores_model_GPU_training_PnP_reads_or_calls=0)
(Path(__file__).parent/'SEALED_CURRENT_EVALUATOR_MOCK_SMOKE.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
