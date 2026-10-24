"""After independent source review only. Creates root entries; no current body read."""
from pathlib import Path
import argparse,hashlib,json
from datetime import datetime,timezone
V=Path(__file__).resolve().parent;P=V.parent
OLD=P/'query_visibility_material_augmented_query_head_20261023_v114'
RUN=V/'runs/C13_SAME_MATERIAL_FOREGROUND_SPATIAL_QUERY'
def b(p):p=Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def r(p):return json.loads(Path(p).read_text())
def w(p,d):
 with Path(p).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
def prepare(stage):
 review=V/'saved_source_review/REVIEW.json';rr=r(review)
 assert rr['completed'] and rr['PASS'] and not rr.get('blockers',[])
 freeze=RUN/'MODEL_FREEZE.json';ff=r(freeze);fit=RUN/'FIT_COMPLETE.json';gates=RUN/'SOURCE_GATE_RESULTS.json'
 assert r(fit)['completed'] and r(fit)['optimizer_updates']==2000 and r(gates)['all_gates_pass'] and ff['selected_update']==1000 and ff['threshold_logit']==0
 admission=V/'ROOT_SOURCE_ADMISSION_FOR_FROZEN_CURRENT_EVAL.json'
 if not admission.exists():
  w(admission,dict(completed=True,created_UTC=datetime.now(timezone.utc).isoformat(),frozen_model_admitted_for_once_current_comparison=True,source_all15_gates_pass=True,source_model_freeze=b(freeze),source_saved_review=b(review),source_public_comparison=b(V/'public_source_comparison/RESULT.json'),current_for_training_normalizer_threshold_checkpoint_arm_selection=False,V112_allowed=False,PnP=0,limitation='Source gates and controlled public developmentVAL improvement do not prove current transfer. Selected checkpoint1000 and raw0 were fixed on source before current.'))
 oldname='ROOT_ACTUAL_C12_FROZEN_CURRENT_HEAD_ONLY_ENTRY.json' if stage=='INFER' else 'ROOT_ACTUAL_C12_SAVED_CURRENT_EVALUATION_ENTRY.json'
 e=r(OLD/oldname)
 e.update(approved=True,stage=stage,code=b(V/'current_evaluation_preparation/frozen_head_current.py'),operator=b(V/'ROOT_CURRENT_SAVED_FEATURE_OPERATOR_BEFORE_C13_FIT.json'),generated=b(V/'current_evaluation_preparation/GENERATED_CURRENT_TESTS.json'),source_trainer=ff['trainer'],source_freeze=b(freeze),source_fit_complete=b(fit),source_gates=b(gates),source_review=b(review),source_admission=b(admission),checkpoint=ff['checkpoint'],root_created_UTC=datetime.now(timezone.utc).isoformat())
 out=V/'private_evaluation/C13_FROZEN_CURRENT223'
 if stage=='EVALUATE_SAVED':
  assert r(out/'INFERENCE_SEAL.json')['PASS'];e['inference_seal']=b(out/'INFERENCE_SEAL.json');out=out/'evaluation'
 e['output_directory']=str(out)
 name='ROOT_ACTUAL_C13_FROZEN_CURRENT_HEAD_ONLY_ENTRY.json' if stage=='INFER' else 'ROOT_ACTUAL_C13_SAVED_CURRENT_EVALUATION_ENTRY.json'
 w(V/name,e);print(json.dumps(b(V/name)))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['INFER','EVALUATE_SAVED']);prepare(a.parse_args().stage)
