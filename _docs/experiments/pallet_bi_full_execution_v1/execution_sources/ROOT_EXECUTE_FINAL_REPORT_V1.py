"""ROOT-reviewed two-child saved-scalar report execution after real terminal gates."""
from pathlib import Path
import hashlib,json,os,resource,subprocess,time
R=Path('/home/minjae/Documents/pallet-research/pallet-bi-full-execution-20261019-v1');W=Path('/tmp/pallet-bi-full-execution-20261019-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def put(p,v):
 with Path(p).open('x')as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n')
hp=R/'FINAL_REPORT_SUPPORT_V1/IMPLEMENTATION_HANDOFF.json'
assert sha(hp)=='07f98ff368962b24a5a72749401f738390f25c86fb42be47964e973e3919921c'
h=read(hp)
assert all(sha(p)==s for p,s in h['artifacts'].items())
for rel in ['DERIVED_COMPLETION_V1/DERIVED_COMPLETION_SEAL.json','ERROR_BUDGET40_WAITER_V1/ERROR40_COMPLETION.json','MASTER_COSTS_WAIT_V2/COMPLETION.json']:
 p=R/rel;assert p.exists() and read(p)['complete']is True,('Required real terminal missing',rel)
for p in [R/'FINAL_REPORT_SPEC_V1.json',R/'FINAL_REPORT_ROOT_GO_V1.json',R/'FINAL_REPORT_MATERIALIZED_V1',R/'ROOT_FINAL_REPORT_ACTUAL_SPEC_PREPARE_V1.json',R/'ROOT_FINAL_REPORT_ACTUAL_EXECUTION_V1.json']:
 assert not p.exists(),('Fresh report/receipt namespace required',str(p))
env=dict(os.environ,**h['execute']['environment']);environment={k:env[k]for k in h['execute']['environment']}
def call(argv,receipt_name,log_name,extra=None):
 before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.monotonic();epoch=time.time()
 with(R/log_name).open('xb')as log:p=subprocess.run(argv,cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT)
 after=resource.getrusage(resource.RUSAGE_CHILDREN)
 value=dict(schema='ROOT_FINAL_REPORT_ACTUAL_CHILD_V1',actual_argv=argv,cwd=str(W),outer_environment=environment,returncode=p.returncode,start_epoch=epoch,finish_epoch=time.time(),wall_seconds=time.monotonic()-start,child_CPU_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,log_sha256=sha(R/log_name),implementation_handoff_sha256=sha(hp),report_source_sha256=h['source_sha256'],new_scientific_calls=0,whole_campaign_complete=False,timing_scope='One external child CPU/wall including import/output/teardown;parent receipt write CPU uninstrumented UNKNOWN_NOT_ZERO')
 if extra:value.update(extra)
 put(R/receipt_name,value);assert p.returncode==0,value
 return value
call(h['prepare']['future_argv'],'ROOT_FINAL_REPORT_ACTUAL_SPEC_PREPARE_V1.json','ROOT_FINAL_REPORT_SPEC_PREPARE_V1.log')
sp=R/'FINAL_REPORT_SPEC_V1.json';spec=read(sp);assert spec['template_only']is False and all(sha(p)==s for p,s in spec['source_bindings'].items())
go=dict(approved=True,metadata_only=True,source_sha256=h['source_sha256'],spec_sha256=sha(sp),epoch=time.time(),authority='ROOT complete reviewed source/template + independent review;existing user fullcampaign authority;savedscalar formatting only after real terminal gates;no new hypothesis or selection')
gp=R/'FINAL_REPORT_ROOT_GO_V1.json';put(gp,go)
call(h['execute']['future_argv'],'ROOT_FINAL_REPORT_ACTUAL_EXECUTION_V1.json','ROOT_FINAL_REPORT_MATERIALIZATION_V1.log',dict(spec_sha256=sha(sp),ROOT_GO_sha256=sha(gp)))
seal=R/'FINAL_REPORT_MATERIALIZED_V1/REPORT_SEAL.json';v=read(seal);assert v['complete']is True and v['whole_campaign_complete']is False
put(R/'ROOT_FINAL_REPORT_PARENT_COMPLETION_V1.json',dict(schema='ROOT_FINAL_REPORT_PARENT_COMPLETION_V1',complete=True,epoch=time.time(),report_source_sha256=h['source_sha256'],report_seal_sha256=sha(seal),prepare_receipt_sha256=sha(R/'ROOT_FINAL_REPORT_ACTUAL_SPEC_PREPARE_V1.json'),execution_receipt_sha256=sha(R/'ROOT_FINAL_REPORT_ACTUAL_EXECUTION_V1.json'),new_scientific_calls=0,public_copy_or_Git_actions=0,whole_campaign_complete=False))
print(json.dumps(dict(report_ready=True,report_seal=str(seal),report_seal_sha256=sha(seal),whole_campaign_complete=False)))
