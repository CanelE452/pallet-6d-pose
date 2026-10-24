"""Score-rule fixtures only. No current-evaluation CSV/labels are opened."""
import copy,hashlib,json
from pathlib import Path
from evaluate_saved_scores import call,calculate,D,E,S

HERE=Path(__file__).resolve().parent
row=dict(id='MOCK',native_corner='0',present_at_fixed_0_5='True',legacy_stored_state='B')
p={('MOCK','0'):.5}
assert call(row,'RAW_SAM','D_vs_E',p,.5,{'threshold_comparison':'>'})=='ABSENT_NONVISIBLE'
assert call(row,'HEAD','D_vs_E',p,.5)=='VISIBLE'
assert call(row,'v61','D_vs_E',p,.5,{'threshold_comparison':'>'})=='VISIBLE'
assert call(row,'legacy','D_vs_E',p,.5,{'threshold_comparison':'>'})=='VISIBLE'
rows=[];probs={};flags={}
for i,(state,prob,flag)in enumerate([(D,.8,True),(D,.2,False),(E,.8,True),(S,.8,True),(D,None,None)]):
 rows.append(dict(id='MOCK',native_corner=str(i),human_state=state));probs['MOCK',str(i)]=prob;flags['MOCK',str(i)]=flag
original=copy.deepcopy(probs)
rules=dict(threshold_comparison='>=',geometry_model='HEAD+GEOMETRY',geometry_flags=flags)
a=calculate(rows,'HEAD','D_vs_ES',probs,.5,rules);b=calculate(rows,'HEAD+GEOMETRY','D_vs_ES',probs,.5,rules)
assert(a['TP'],a['FN_decided'],a['FP'],a['TN'],a['other_unresolved_visible'])==(1,1,2,0,1)
assert(b['TP'],b['FN_decided'],b['FP'],b['TN'],b['other_unresolved_visible'])==(0,2,0,2,1)
assert probs==original
result=dict(schema='SAVED_SCORE_RULE_MOCK_SMOKE_V1',private_inputs_or_labels_opened=0,raw_SAM_strict_tie_absent=True,default_head_v61_legacy_tie_behavior_preserved=True,geometry_override_separate_model_tag=True,raw_score_dictionary_unchanged=True,geometry_can_help_occluded_and_damage_visible=True,unknown_raw_score_preserved_unresolved=True,mock_raw_metrics=a,mock_hybrid_metrics=b,script_sha256=hashlib.sha256((HERE/'evaluate_saved_scores.py').read_bytes()).hexdigest())
(HERE/'SCORE_RULE_MOCK_SMOKE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
