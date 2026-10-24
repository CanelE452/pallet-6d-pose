"""Root's authorized post-freeze diagnostic gates, never source/model selection."""
from pathlib import Path
import argparse,json,hashlib,datetime
R=Path(__file__).resolve().parents[1];P=R.parent;OLD=P/'query_visibility_cue_occlusion_20261023_v72'
F=R/'mixed_fit/runs/MIXED_SUPPORT_V1/MODEL_FREEZE.json'
def b(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def load(p):return json.loads(p.read_text())
def save(p,v):assert not p.exists();p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['infer','evaluate']);a=ap.parse_args()
 f=load(F);assert f['completed']and f['optimizer_updates']==2000 and not f['current_used_for_fit_selection']and f['PnP']==0 and f['visible_probability_threshold']==.5
 audit=R/'mixed_fit_independent_saved_audit/INDEPENDENT_SAVED_RESULT_AUDIT.json';proof=load(audit);assert proof['PASS']and proof['checkpoint']==f['checkpoint']and proof['primary_admission_reconstructed_exact']
 if a.stage=='infer':
  code=R/'private_evaluation/prepare_and_infer_after_freeze.py';qa=R/'private_adapter_preparation_audit/OUTPUT_GEOMETRY_V3_ADDENDUM.json';q=load(qa);assert q['PASS']and q['bindings']['V3']==b(code)
  deps=[P/'query_visibility_loco_native_context_20261023_v74/native_context_cache/compile_input_pack.py',OLD/'private_evaluation/current_input_adapter.py',P/'query_visibility_loco_native_context_20261023_v74/source_contract/native_context_contract_v2.py',OLD/'private_evaluation/CURRENT_INPUT_ADAPTER_GATE.json',OLD/'private_evaluation/CUE_ATTENTION_V3/PREPARATION_SEAL.json']
  save(R/'private_evaluation/POST_SOURCE_FREEZE_GATE.json',dict(schema='V76_POST_SOURCE_FREEZE_ONE_CURRENT_DIAGNOSTIC',approved=True,source_selection_complete=True,source_model_freeze=b(F),source_independent_audit=b(audit),adapter_code=b(code),adapter_code_audit=b(qa),code_and_prior_input_dependencies=[b(x)for x in deps],current_used_for_selection=False,threshold=.5,PnP=0,authorization='Existing user goal: source-only fit then fixed current223 evaluation, no current selection.',created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 else:
  g=load(R/'contract/EVALUATOR_GATE_TEMPLATE.json');g.update(approved=True,source_frozen=True,current_outputs_complete=True,evaluation_authorized_after_source_freeze=True,source_model_freeze=b(F),checkpoint=f['checkpoint'],inference_seal=b(R/'private_evaluation/MIXED_SUPPORT_V1/INFERENCE_SEAL.json'),scores=b(R/'private_evaluation/MIXED_SUPPORT_V1/REAL_QUERY_SCORES.csv'),historical_v72=dict(scores=b(OLD/'private_evaluation/CUE_ATTENTION_V3/REAL_QUERY_SCORES.csv'),model_freeze=b(OLD/'runs/CUE_ATTENTION_V3/MODEL_FREEZE.json'),inference_seal=b(OLD/'private_evaluation/CUE_ATTENTION_V3/INFERENCE_SEAL.json')),scope='Authorized fixed-model saved-score evaluation only; root does not choose model/threshold from current outcomes.')
  save(R/'contract/POST_INFERENCE_EVALUATION_GATE.json',g)
 print(a.stage+'_GATE_SAVED')
if __name__=='__main__':main()
