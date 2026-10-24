"""Mock-only metric invariants; no source/private files, models, GPU or pose."""
import json
from pathlib import Path
import numpy as np
from evaluate_frozen_source import confusion, summarize, binary_metrics, train_prior

# U contributes a false negative for recall_all_GT, never disappears from GT.
y = np.array([0, 1, 2, 2])
pred = np.array([0, -1, 1, 2])
assert confusion(y, pred, 3).tolist() == [[1,0,0,0],[0,0,0,1],[0,1,1,0]]
prior = np.full((4, 6), 1/6)
r = summarize(np.array([0,1,2,5]), prior)
assert r['confident_joint_MAP_U']['coverage'] == 0
assert r['confident_joint_MAP_U']['withheld'] == 4
assert all(c['recall_all_GT'] == 0 for c in r['confident_joint_MAP_U']['per_class'] if c['support'])
assert np.isclose(r['joint_NLL'], np.log(6))
# Exact half accepted, first argmax tie, while low-pE logical E remains U.
p = np.array([[.5,0,0,0,0,.5],[.43,.12,0,0,0,.45]])
r = summarize(np.array([0,1]), p)
assert r['confident_joint_MAP_U']['coverage'] == .5
assert r['scales']['exact']['logical_support_diagnostic']['confusion_truth_rows_prediction_columns_U_last'][1][1] == 1
assert r['scales']['exact']['confident_joint_MAP_U']['per_class'][1]['recall_all_GT'] == 0
# Binary fixed >=.5 and prethresholded rawSAM strict >0 stay distinct.
assert binary_metrics([1,0], [.5,.499])['tp'] == 1
assert binary_metrics([1,0], (np.array([0., .001]) > 0).astype(float))['fn'] == 1
train = dict(dense_valid=np.ones((3,9,9),bool), joint=np.zeros((3,9,9),int), source=np.array(['SYNTH']*3))
prob, info = train_prior(train, np.array(['SYNTH','YCB']))
assert np.array_equal(prob, np.array([[1,0,0,0,0,0]]*2))
assert info['YCB']['missing_domain_uses_global_TRAIN']
result = dict(schema='V64_INDEPENDENT_EVALUATOR_MOCK_SMOKE_V1', PASS=True,
              checks=['U all-GT denominators','uniform NLL','half tie acceptance','low-pE logical call kept separate',
                      'head >=.5 vs rawSAM >0','TRAIN-only missing-domain prior fallback'],
              source_private_model_GPU_PnP_reads_or_calls=0)
(Path(__file__).parent/'FROZEN_SOURCE_EVALUATOR_SMOKE.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
