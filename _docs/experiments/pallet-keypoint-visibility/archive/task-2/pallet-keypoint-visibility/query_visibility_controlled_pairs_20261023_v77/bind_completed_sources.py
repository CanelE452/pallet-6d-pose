"""Bind completed source-only pairs and recomputed feature/input QA."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent;V=R.parent/'query_visibility_reference_local_support_20261023_v76'
def b(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return dict(path=str(p),sha256=h.hexdigest())
def load(p):return json.loads(p.read_text())
def write(p,x):assert not p.exists();p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 qa=load(R/'after_input_pack/INPUT_PACK_QA.json');assert qa['PASS']
 data=load(R/'pair_generation/SEAL.json');assert data['completed']
 independent=load(R/'independent_data_audit/DATA_AUDIT.json');assert independent['PASS']
 paths=dict(after_input_manifest=R/'after_input_pack/PACK_MANIFEST.json',after_input_lineage=R/'after_input_pack/INDEX_LINEAGE_INPUT_ONLY.jsonl',after_input_seal=R/'after_input_pack/PACK_SEAL.json',after_input_QA=R/'after_input_pack/INPUT_PACK_QA.json',after_teacher_GT=R/'pair_generation/AFTER_GT_ONLY.npz',pair_data_seal=R/'pair_generation/SEAL.json',pair_build_protocol=R/'pair_generation/PAIR_BUILD_PROTOCOL_BEFORE_COMPOSITION.json',source_independent_data_audit=R/'independent_data_audit/DATA_AUDIT.json',original_loader_code=V/'mixed_fit/train_mixed_once.py',original_input_bindings=V/'mixed_fit/SOURCE_INPUT_BINDINGS.json',original_protocol=V/'mixed_fit/MIXED_FIT_PROTOCOL.json',root_decisions=R/'EXPERIMENT_DECISIONS_BEFORE_PAIR_BUILD.json')
 write(R/'COMPLETED_SOURCE_INPUT_BINDINGS.json',dict(schema='V77_COMPLETE_PAIRED_SOURCE_BINDINGS',bindings={k:b(v)for k,v in paths.items()},frozen_choices_unchanged=True,current_reads=0,PnP=0))
 f=load(V/'mixed_fit/runs/MIXED_SUPPORT_V1/MODEL_FREEZE.json');baseline=dict(code=b(R/'evaluate_previous_frozen_model.py'),protocol=b(R/'FROZEN_V76_BASELINE_PROTOCOL.json'),source_inputs=b(R/'COMPLETED_SOURCE_INPUT_BINDINGS.json'),loader_code=b(R/'train_paired_once.py'),checkpoint=f['checkpoint'],model_code=f['model_code'],old_original_scores=b(V/'mixed_fit/runs/MIXED_SUPPORT_V1/SELECTED_SOURCE_SCORES.npz'))
 write(R/'FROZEN_V76_BASELINE_ENTRY.json',dict(schema='V77_LOCKED_V76_SOURCE_BASELINE_INFERENCE_ENTRY',bindings=baseline,optimizer_updates=0,current_reads=0,PnP=0))
 print(json.dumps(b(R/'COMPLETED_SOURCE_INPUT_BINDINGS.json')))
if __name__=='__main__':main()
