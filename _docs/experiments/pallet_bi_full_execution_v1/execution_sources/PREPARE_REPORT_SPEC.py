"""Metadata-only spec preparation;ROOT executes only after actual gates."""
import argparse,hashlib,json,pathlib,sys,time
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
p=argparse.ArgumentParser();p.add_argument('--template',required=True);p.add_argument('--output',required=True);a=p.parse_args()
s=read(a.template);assert s['template_only'] is True
for path,digest in s['source_bindings'].items():assert sha(path)==digest,(path,'reviewed code changed')
for key,terminal in [('metrics','EXPORT_SEAL.json'),('paired','PUBLIC_CONTRASTS_SEAL.json'),('budget','ERROR_BUDGET_SEAL.json'),('D','EXPORT_SEAL.json')]:
    path=pathlib.Path(s[key]['path'])/terminal;value=read(path);assert value['complete'] is True
    s[key]['sha256']=sha(path)
for key in ['cost','cost_waiter','fifth_service','f_choice','derived','sequence','method_ledger_draft']:
    path=pathlib.Path(s[key]['path']);s[key]['sha256']=sha(path)
    if key not in ['method_ledger_draft','cost']:assert read(path)['complete'] is True
cost=read(s['cost']['path']);assert cost['all_declared_H1_fits_complete'] and cost['all_declared_H1_prediction_seals_complete']
assert cost['H1']['completed_new_fits']==cost['H1']['reservations']==33 and cost['H1']['tiny_head_real_prediction_forwards']==7854
s['cohorts']['fixed_sets']['sha256']=sha(s['cohorts']['fixed_sets']['path'])
for item in s['cohorts']['banks']:
    value=read(item['seal']);assert value['complete'] is True and value['actual_new_pose_calls']==0
    item['sha256']=sha(item['seal']);assert sha(item['summary'])==value['artifacts']['SUMMARY.json']
item=s['saved_diagnostic'];assert read(item['seal'])['complete'] is True;item['sha256']=sha(item['seal'])
s['template_only']=False;s['preparation']={'source_sha256':sha(__file__),'actual_argv':sys.argv,'epoch':time.time(),'new_scientific_calls':0}
with pathlib.Path(a.output).open('x')as f:json.dump(s,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps({'prepared_spec':a.output,'sha256':sha(a.output),'new_scientific_calls':0,'report_materialized':False}))
