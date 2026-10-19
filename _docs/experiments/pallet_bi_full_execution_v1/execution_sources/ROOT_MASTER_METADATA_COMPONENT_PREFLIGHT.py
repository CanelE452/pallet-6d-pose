from pathlib import Path
import json,time,resource,hashlib
from scripts.research.pallet_bi_full_execution_v1 import master_costs as M
R=Path("/home/minjae/Documents/pallet-research/pallet-bi-full-execution-20261019-v1"); cp=R/'MASTER_COSTS_METADATA_CATALOG_V3.json'
r=M.Reader(); c=r.read(cp); start=time.monotonic();cpu=time.process_time()
closed=M.closed_attempts(r,c['closed'])
heads=M.head_accounts(r,R,c['predictions'],33)
ledgers=[M.command_ledger(r,x)for x in c['ledgers'][:2]]
r.stable()
out=dict(schema='ROOT_MASTER_METADATA_COMPONENT_PREFLIGHT_V1',scope='closed historical cost audit + presentlycompleteH1cost/prediction receipts + closedS3ledgers only;notfullcampaignaudit',complete_all_scientific_stages=False,master_source_sha256=M.sha(M.__file__),catalog_sha256=M.sha(cp),closed_known_calls=closed['known_calls'],completed_H1_fits=heads['completed_new_fits'],reservations=heads['reservations'],tiny_head_real_prediction_forwards=heads['tiny_head_real_prediction_forwards'],prediction_status=[dict(arm=x['arm'],complete=x['complete'])for x in heads['predictions']],closed_ledgers=ledgers,input_binding_count=len(r.bindings),input_bindings=r.bindings,wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,new_scientific_calls=0,new_fits=0,new_model_forwards=0,dataset_or_reference_decodes=0,epoch=time.time())
p=R/'ROOT_MASTER_METADATA_COMPONENT_PREFLIGHT_V1.json'
with p.open('x')as f:json.dump(out,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({k:out[k]for k in ['closed_known_calls','completed_H1_fits','reservations','tiny_head_real_prediction_forwards','input_binding_count','wall_seconds','CPU_seconds']},indent=2))

