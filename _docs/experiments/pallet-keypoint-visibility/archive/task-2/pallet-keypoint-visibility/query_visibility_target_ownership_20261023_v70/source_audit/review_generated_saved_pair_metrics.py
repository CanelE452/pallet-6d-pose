"""Exercise V70 pair metrics on generated metadata only; no files of source data."""
from pathlib import Path
import hashlib,importlib.util,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
code=ROOT/'source_evaluation/evaluate_saved_ownership.py'
digest=hashlib.sha256(code.read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('saved_evaluator_generated_review',code)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ids=np.array([f'MOCK_ORIG:{i}'for i in range(3200)]+[f'MOCK_CTRL:{i}'for i in range(3200)])
source=np.array(['ORIGINAL_YCB']*3200+['CONTROLLED_YCB']*3200)
state=np.full((6400,2),2,np.int8);state[:1571]=0;state[3200:4326]=0;state[4326:4771]=1
gt=dict(sample_id=ids,source=source,states=state)
rr=[]
for i in range(3200):
    kind='V_TO_V'if i<1126 else('V_TO_E'if i<1571 else'NON_V_UNCHANGED')
    rr.append(dict(split='VAL',original_index=str(i),controlled_index=str(i+3200),
      original_sample_id=ids[i],controlled_sample_id=ids[i+3200],pair_type=kind,
      original_exact_state=str(state[i,0]),controlled_exact_state=str(state[i+3200,0])))
original_reader=m.rows;m.rows=lambda unused:rr
perfect=(state[:,0]==0).astype(np.int8)
mixed=perfect.copy();mixed[3200:3210]=0;mixed[:4]=0;mixed[20]=0;mixed[4326:4338]=1
with_u=perfect.copy();with_u[3230]=-1
result=m.pair_tables(dict(pair_path='GENERATED_NO_FILE_READ'),gt,
  dict(PERFECT=dict(prediction=perfect),MIXED=dict(prediction=mixed),WITH_U=dict(prediction=with_u)))
assert result['PERFECT']['V_TO_V']['joint_correct']==1126
assert result['PERFECT']['V_TO_E']['joint_correct']==445
assert result['MIXED']['V_TO_V']['original_visible']==1121
assert result['MIXED']['V_TO_V']['controlled_visible']==1116
assert result['MIXED']['V_TO_V']['joint_correct']==1115
assert result['MIXED']['V_TO_V']['original_visible_to_controlled_nonvisible']==6
assert result['MIXED']['V_TO_V']['original_nonvisible_to_controlled_visible']==1
assert result['MIXED']['V_TO_E']['controlled_binary_nonvisible']==433
assert result['MIXED']['V_TO_E']['joint_correct']==433
assert result['WITH_U']['V_TO_V']['joint_correct']==1125
assert result['WITH_U']['V_TO_V']['prediction_transition_matrix']['counts'][0][2]==1
assert sum(map(sum,result['WITH_U']['V_TO_V']['prediction_transition_matrix']['counts']))==1126
# GT metadata cannot silently become training or mismatched identity in eval.
bad=rr[0].copy();rr[0]['split']='TRAIN'
try:m.pair_tables(dict(pair_path='GENERATED'),gt,dict(M=dict(prediction=perfect)))
except ValueError:pass
else:raise AssertionError('TRAIN pair accepted')
rr[0]=bad;rr[0]['original_sample_id']='OTHER_TARGET'
try:m.pair_tables(dict(pair_path='GENERATED'),gt,dict(M=dict(prediction=perfect)))
except ValueError:pass
else:raise AssertionError('Wrong target pair accepted')
m.rows=original_reader
assert digest==hashlib.sha256(code.read_bytes()).hexdigest()
out=ROOT/'source_audit/GENERATED_SAVED_PAIR_METRICS_REVIEW.json';assert not out.exists()
r=dict(completed=True,status='PASS',evaluator=dict(path=str(code),sha256=digest),
 generated_pairs=3200,preservedV=1126,changedVE=445,perfect_mixed_U_transition_checks=True,
 TRAIN_or_wrong_target_rejected=True,actual_source_GT_scores_csv_RGB_features_weights_current_reads=0,
 model_forward_calls=0,fit=0)
out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(path=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest())))
