import pathlib,json,time,subprocess,hashlib
R=pathlib.Path("/home/minjae/Documents/pallet-research/pallet-bi-full-execution-20261019-v1");W=pathlib.Path("/tmp/pallet-bi-full-execution-20261019-v1")
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def put(p,v):
 with p.open('x')as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n')
h=json.loads((R/'COST_WAITER_IMPLEMENTATION_HANDOFF.json').read_text())
source=W/'scripts/research/pallet_bi_full_execution_v1/cost_waiter.py'
assert sha(source)==h['source_sha256']=='3e3ab0d9ac5e7b5d3eaa2b0fd0d695df039b24c3990b962e8fa78fbd99216cce'
for p,s in h['artifact_bindings'].items():assert sha(p)==s
pp=pathlib.Path(h['plan']);plan=json.loads(pp.read_text())
assert sha(pp)==h['plan_sha256']=='6d69c8dbf2dd6192260c194ddb4708cbccea91bf791f7467e994febe073af5e1'
assert all(sha(p)==s for p,s in plan['sources'].items())
assert all(sha(p)==s for p,s in plan['fixed_inputs'].items())
g=dict(h['ROOT_GO_fields'],epoch=time.time(),authority='User fullcampaign;ROOT metadata-only closure review inclfreshV2catalog guard+21tests;no new scientificcalls')
gp=pathlib.Path(h['ROOT_GO_required_path']);put(gp,g)
unit='pallet-bi-master-cost-waiter-20261019-v2';log=R/'MASTER_COSTS_WAITER_SYSTEMD_V2.log'
argv=['systemd-run','--user','--unit='+unit,'--property=WorkingDirectory='+str(W),'--property=Restart=no','--property=RemainAfterExit=yes','--property=CPUAccounting=yes','--property=MemoryAccounting=yes','--property=StandardOutput=append:'+str(log),'--property=StandardError=append:'+str(log)]
for key,value in h['outer_env'].items():argv.append('--setenv='+key+'='+value)
argv+=h['future_argv'];began=time.time();p=subprocess.run(argv,cwd=W,text=True,capture_output=True)
v=dict(schema='ROOT_COST_WAITER_ACTUAL_LAUNCH_V2',epoch=time.time(),launch_start_epoch=began,actual_service_argv=argv,unit=unit,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,source_sha256=sha(source),plan_sha256=sha(pp),GO_sha256=sha(gp),catalog_sha256=sha(plan['catalog']),implementation_handoff_sha256=sha(R/'COST_WAITER_IMPLEMENTATION_HANDOFF.json'),new_scientific_calls=0,one_future_metadata_audit_child=True,campaign_complete=False)
put(R/'ROOT_COST_WAITER_ACTUAL_LAUNCH_V2.json',v)
assert p.returncode==0,v
print(json.dumps(v,indent=2))

