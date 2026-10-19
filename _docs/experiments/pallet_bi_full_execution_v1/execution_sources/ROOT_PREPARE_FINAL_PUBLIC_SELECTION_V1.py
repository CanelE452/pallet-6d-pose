"""ROOT exact metadata selection after all actual terminal gates; no science/Git."""
from pathlib import Path
import hashlib,json,time
R=Path('/home/minjae/Documents/pallet-research/pallet-bi-full-execution-20261019-v1')
W=Path('/tmp/pallet-bi-full-execution-20261019-v1')
P=W/'_docs/experiments/pallet_bi_full_execution_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def ref(path):
 p=R/path
 assert p.is_file(),str(p)
 return dict(path=path,sha256=sha(p))
spec=read(R/'FINAL_PUBLICATION_SUPPORT_V4/FINAL_SELECTION_TEMPLATE.json')
spec['template_only']=False
spec['code_filenames']=sorted(set(spec['code_filenames'])|{'FINAL_REPORT_README.md'})
docs=['FINAL_REPORT_KO.md','METHODOLOGY_KO.md','STRUCTURE_REVIEW_NOTES_KO.md','REPRODUCE.md',
 'S1_D_EVIDENCE.md','S3_S4_EVIDENCE.md','METHOD_DIFFERENCE_LEDGER.json','METHOD_DIFFERENCE_LEDGER.csv',
 'STAGE_STATUS.json','STAGE_STATUS.csv','D_PUBLIC_SCALARS_README.md',
 'REQUIREMENTS_MATRIX.json','REQUIREMENTS_MATRIX.csv','CACHE_DEPENDENCIES.json','CACHE_DEPENDENCIES.csv',
 'EXECUTION_INDEX.json','EXECUTION_INDEX.csv','PROVENANCE_INDEX.json']
spec['documents']=[dict(name=n,sha256=sha(P/n))for n in docs]
spec['stage_statuses']={x['stage']:x['status']for x in read(P/'STAGE_STATUS.json')['stages']}
bindings={'size_policy':'ROOT_FINAL_PUBLIC_SIZE_POLICY.json',
 'final_sequence':'FINAL_SEQUENCE_RECOVERY_V1/FINAL_SEQUENCE_SEAL.json',
 'derived_completion':'DERIVED_COMPLETION_V1/DERIVED_COMPLETION_SEAL.json',
 'final_sequence_GO':'FINAL_SEQUENCE_RECOVERY_ROOT_GO_V1.json',
 'derived_completion_GO':'DERIVED_COMPLETION_ROOT_GO_V1.json',
 'cost_summary':'MASTER_COSTS_FINAL_V1/PUBLIC_COST_SUMMARY.json',
 'derived_service_cost':'ROOT_DERIVED_SERVICE_FINAL_CAPTURE_PUBLIC.json',
 'error_budget40':'ERROR_BUDGET40_V1/ERROR_BUDGET_SEAL.json',
 'report_appendices':'FINAL_REPORT_MATERIALIZED_V1/REPORT_SEAL.json',
 'final_visual_review':'ROOT_LIMITED_FINAL_PUBLIC_VISUAL_REVIEW.json'}
for k,v in bindings.items():spec[k]=ref(v)
execution=[x['path']for x in spec['execution_sources']]+[
 'ROOT_EXECUTE_FINAL_REPORT_V1.py','FINAL_REPORT_SUPPORT_V1/PREPARE_REPORT_SPEC.py',
 'ROOT_PREPARE_FINAL_PUBLIC_SELECTION_V1.py']
assert len({Path(n).name for n in execution})==len(execution)
spec['execution_sources']=[ref(n)for n in execution]
receipt_specs=[
 ('origin-and-original-checkout','provenance','START.json','Original dirty checkout, instruction SHA and protected-byte bindings'),
 ('plan','provenance','CAMPAIGN_PLAN.json','Preregistered stages,fit budgets and source/selection contracts'),
 ('P0-reuse','cache','P0_REUSE_PROOF.json','Exact full-input and source contracts;reuse instead of another backbone forward'),
 ('S1-reuse','cache','S1_REUSE_AND_ADMISSION_AUDIT.json','Endpoint preservation,typed admission and strict-Rt dedup'),
 ('S2-recovery','cache','S2_FOUR_CPU_RECOVERY_INVOCATION_V1/RECOVERY_COMPLETION.json','Actual1241 completed rows;current/reused/cumulative optimizer units'),
 ('F-recovery','cache','F_FOUR_CPU_RECOVERY_INVOCATION_V1/RECOVERY_COMPLETION.json','Actual1241 completed rows;current/reused/cumulative optimizer units'),
 ('full-costs','calls','MASTER_COSTS_FINAL_V1/PUBLIC_COST_SUMMARY.json','Recovery-aware lowerbounds,33fits,7854realheadforwards and unknown costs'),
 ('derived-service','calls','ROOT_DERIVED_SERVICE_FINAL_CAPTURE_PUBLIC.json','Separate fifth inclusive service CPU/elapsed,not additive to nested children'),
 ('fit-attempts','model','GLOBAL_H1_FIT_ATTEMPTS.jsonl','All33 registered own-bank tiny H1 fit reservations;neural payloads remain private'),
 ('F-training','model','F_FINALIZE/FINISH_RECEIPT.json','Actual6 finalfits+own pre-reference prediction seals'),
 ('versions','environment','ENVIRONMENT_INITIAL.json','Measured Python,numerical-library and host versions;no whole environment dump'),
 ('CPU-placement','environment','ROOT_CPU_PLACEMENT_AMENDMENT_V1.json','ActualCPU placement only;no numerical/kernel change or speedup claim'),
 ('final-sequence-argv','argv','FINAL_SEQUENCE_RECOVERY_V1/ACTUAL_COMMAND_LEDGER.jsonl','Literal12stage childcommands and completion outcomes'),
 ('derived-argv','argv','DERIVED_COMPLETION_V1/ACTUAL_COMMAND_LEDGER.jsonl','Literal8stage derived childcommands and completion outcomes'),
 ('report-argv','argv','ROOT_FINAL_REPORT_ACTUAL_EXECUTION_V1.json','Actuallyexecuted savedscalar report command,CPU/wall and exactspec'),
 ('S2-calibration','calibration','S2_CALIBRATION/SUMMARY.json','Registered CAL46 common/role covariance calibration;not independent test'),
 ('F-calibration','calibration','F_CALIBRATION/MODELS.json','EachF arm ownCAL calibration;measurement covariance not neural weights'),
 ('frozen-final-choice','provenance','F_SYSTEM_CHOICE.json','ActualS2_CP ownVALchoice,empty auxiliary actions,no PRIMARYranking')]
spec['public_receipt_index']=[dict(alias=a,role=b,scope=d,**ref(c))for a,b,c,d in receipt_specs]
spec['notes']=['Actualclosed33fit/11pred/40arm/30contrast/11gallery/99scatter campaign. Scientificandderivedcomplete;publicationisaseparateROOTaction.',
 'Allfailures,outliers and transitions preserved. RawsourceRGB,CALRGB,poses,GT,weights,fullbanks andprivateassetsnotselected.',
 'SelectedS2_CP finalFhas noPF/material/S4auxiliary;eacharm ownsitsobservation/CAL/fullnormal/H1.',
 'Wholecampaigncompletion andactualremoteSHA recordedafterpush,outsidecommittedprepublicationmetadata.']
out=R/'ROOT_FINAL_SELECTION_V1.json'
with out.open('x')as f:json.dump(spec,f,ensure_ascii=False,indent=2);f.write('\n')
receipt=dict(schema='ROOT_FINAL_PUBLIC_SELECTION_PREPARATION_V1',complete=True,epoch=time.time(),
 selection_sha256=sha(out),source_sha256=sha(__file__),code_files=len(spec['code_filenames']),
 documents=len(docs),execution_sources=len(execution),new_scientific_calls=0,Git_actions=0)
with(R/'ROOT_FINAL_SELECTION_ACTUAL_PREPARATION_V1.json').open('x')as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt))
