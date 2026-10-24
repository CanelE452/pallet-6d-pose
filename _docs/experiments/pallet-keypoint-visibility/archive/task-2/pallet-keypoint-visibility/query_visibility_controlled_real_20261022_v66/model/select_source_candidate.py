"""Select exactly once from two completed source-only frozen fits."""
from pathlib import Path
import hashlib,json,math,time
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def main():
 out=ROOT/'SOURCE_SELECTION_SEAL.json';assert not out.exists()
 protocol=ROOT/'FIT_PROTOCOL_V2.json';proto=json.loads(protocol.read_text());lockp=ROOT/'provenance/FIT_INPUT_LOCK.json';lock=json.loads(lockp.read_text())
 for b in lock['bindings']:assert sha(b['path'])==b['sha256']
 candidates=[]
 for name in proto['candidates']:
  d=ROOT/'runs'/name;freeze=d/'MODEL_FREEZE.json';f=json.loads(freeze.read_text());checkpoint=d/'BEST_PARTIAL_HEAD.pt'
  assert f['completed']is True and f['eval_data_used']is False and f['PnP_used']is False
  assert f['current223_used']is False and f['source_only']is True
  assert f['optimizer_updates']==proto['updates'] and f['parameters']==95508
  assert f['protocol_sha256']==sha(protocol) and f['input_lock_sha256']==sha(lockp)
  assert f['checkpoint_sha256']==sha(checkpoint) and f['threshold']==.5
  assert set(f['validation_sources'])==set(proto['validation_sources'])
  assert math.isfinite(f['best_source_partial_NLL'])
  h=json.loads((d/'HISTORY.json').read_text());best=min(h,key=lambda row:row['VAL_partial_NLL'])
  assert best['update']==f['best_update']and best['VAL_partial_NLL']==f['best_source_partial_NLL']
  assert sha(d/'PROTOCOL_AT_START.json')==sha(protocol)
  candidates.append(dict(candidate=name,checkpoint=bind(checkpoint),freeze=bind(freeze),source_macro_partial_NLL=f['best_source_partial_NLL'],best_update=f['best_update']))
 selected=min(candidates,key=lambda c:c['source_macro_partial_NLL'])
 audits={}
 for role,p in [('joined_cache',ROOT/'source_audit/JOINED_CACHE_AUDIT.json'),('partial_trainer',ROOT/'source_audit/INDEPENDENT_PARTIAL_TRAINER_AUDIT.json'),('source_protocol',protocol),('selection_code',Path(__file__))]:audits[role]=bind(p)
 result=dict(schema='V66_SOURCE_ONLY_CANDIDATE_SELECTION_SEAL',completed=True,created_unix=time.time(),primary=selected['candidate'],
  current_eval_used_for_selection=False,private_inference_before_selection=0,fit_input_lock=bind(lockp),fit_protocol=bind(protocol),
  thresholds=dict(binary_visible=.5,logical_full=.5,joint_confidence=.5),candidates=candidates,source_audits=audits,
  selection='Common4source macro informativecenter partialNLL; strict minimum, exacttie listed CONTROLLED_ONLY first. GPU validation is frozen selection criterion; later CPU audits cannot reselect.',
  new_head_optimizer_updates=2*proto['updates'],new_backbone_optimizer_updates=0,new_PnP=0,
  limitations=['Controlled cutouts and conditional photographic teachers are separate evidence tiers.','No real semanticcorner/self pixelGT created.','Current223 remains evaluation-only and repeatedly exposed development cohort.'])
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
