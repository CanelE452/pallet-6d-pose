import json,pathlib,hashlib,time,subprocess,os
R=pathlib.Path("/home/minjae/Documents/pallet-research/pallet-bi-full-execution-20261019-v1"); W=pathlib.Path("/tmp/pallet-bi-full-execution-20261019-v1"); PY="/home/minjae/anaconda3/envs/pallet-pose/bin/python"
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def put(p,v):
 with pathlib.Path(p).open('x')as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n')
old=R/'FINAL_SEQUENCE/PLAN.json'; p=json.loads(old.read_text())
assert all(sha(k)==v for k,v in p['sources'].items())
assert len(p['sources'])==738
adapter=W/'scripts/research/pallet_bi_full_execution_v1/fullnormal_recovery.py'
assert sha(adapter)=='5c5914b4f7dbcaea027b34cb195c2eee79437f4f25c74149e16cceec70c39eb1'
output=R/'FINAL_SEQUENCE_RECOVERY_V1';output.mkdir(exist_ok=False)
p['output']=str(output);p['F_original_controller_source_sha256']=p['F_controller_source_sha256'];p['F_controller_source_sha256']=sha(adapter)
p['sources'][str(adapter)]=sha(adapter)
for path in [R/'FULLNORMAL_RECOVERY_ROOT_GO.json',R/'ROOT_FULLNORMAL_RECOVERY_SERVICE_LAUNCH.json',R/'FULLNORMAL_RECOVERY_IMPLEMENTATION_HANDOFF.json',R/'FINAL_SEQUENCE_IDLE_CONTROLLER_STOPPED_FOR_RECOVERY.json',R/'SERVICE_ENVIRONMENT_PARITY_CHECK.json']:p['fixed_inputs'][str(path)]=sha(path)
p['technical_recovery_amendment']=dict(original_plan_sha256=sha(old),original_go_sha256=sha(R/'FINAL_SEQUENCE_ROOT_GO.json'),original_frozen_source_count=738,original_source_hashes_unchanged=True,scientific_method_changes=False,unchanged_controller_source=True,reason='Original D4 owners disappeared with unknown in-memory cost; ROOT transient services recover immutable complete ROWs and continue exact same numerical code. F completion records actual new adapter source rather than fabricating original controller ownership.',epoch=time.time())
pp=output/'PLAN.json';put(pp,p)
g=json.loads((R/'FINAL_SEQUENCE_ROOT_GO.json').read_text());g['plan_sha256']=sha(pp);g['epoch']=time.time();g['original_GO_sha256']=sha(R/'FINAL_SEQUENCE_ROOT_GO.json');g['technical_recovery_only']=True
gp=R/'FINAL_SEQUENCE_RECOVERY_ROOT_GO_V1.json';put(gp,g)
heads=R/'S2_UNIT_WEIGHT_FOUR_CPU_H1'
assert not (heads/'PREFIT_SEAL.json').exists()
assert not any('S2_' in x.get('arm','') or x.get('arm','').startswith('F_') for x in map(json.loads,(R/'GLOBAL_H1_FIT_ATTEMPTS.jsonl').read_text().splitlines()))
tasks={
 'S2_H1':('pallet-bi-full-s2-h1-20261019-v1',[PY,'-B',str(R/'S2_UNIT_WEIGHT_FOUR_CPU_FINISH_H1.py')]),
 'FINAL_SEQUENCE':('pallet-bi-full-final-sequence-20261019-v1',[PY,'-B','-m','scripts.research.pallet_bi_full_execution_v1.final_sequence','run','--plan',str(pp),'--root-GO',str(gp)])
}
receipt=dict(schema='ROOT_POST_CORE_SERVICES_LAUNCH_V1',epoch=time.time(),fresh_plan_sha256=sha(pp),fresh_GO_sha256=sha(gp),services={})
for key,(unit,argv) in tasks.items():
 log=R/(key+'_RECOVERY_SERVICE.log')
 command=['systemd-run','--user','--unit='+unit,'--property=WorkingDirectory='+str(W),'--property=Restart=no','--property=RemainAfterExit=yes','--property=CPUAccounting=yes','--property=MemoryAccounting=yes','--property=StandardOutput=append:'+str(log),'--property=StandardError=append:'+str(log),'--setenv=OMP_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1','--setenv=OPENBLAS_NUM_THREADS=1','--setenv=PYTHONPATH='+str(W)]+argv
 started=time.time();out=subprocess.run(command,text=True,capture_output=True)
 receipt['services'][key]=dict(unit=unit,actual_argv=command,scientific_argv=argv,log=str(log),launch_start_epoch=started,launch_end_epoch=time.time(),returncode=out.returncode,stdout=out.stdout,stderr=out.stderr)
 assert out.returncode==0,receipt
put(R/'ROOT_POST_CORE_RECOVERY_SERVICES_LAUNCH.json',receipt)
print(json.dumps(receipt,indent=2))

