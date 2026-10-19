import json, pathlib, hashlib, os, subprocess, time
R=pathlib.Path("/home/minjae/Documents/pallet-research/pallet-bi-full-execution-20261019-v1")
W=pathlib.Path("/tmp/pallet-bi-full-execution-20261019-v1")
PY="/home/minjae/anaconda3/envs/pallet-pose/bin/python"
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
h=json.loads((R/'FULLNORMAL_RECOVERY_IMPLEMENTATION_HANDOFF.json').read_text())
assert sha(h['source'])==h['source_sha256']=='5c5914b4f7dbcaea027b34cb195c2eee79437f4f25c74149e16cceec70c39eb1'
active=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  a=[v.decode(errors='replace') for v in (p/'cmdline').read_bytes().split(b'\0') if v]
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 if any(v.endswith('/S2_UNIT_WEIGHT_FOUR_CPU_FINISH_H1.py') or v.endswith('/F_CORE_FOUR_CPU_CONTROLLER.py') or v in ('scripts.research.pallet_bi_full_execution_v1.fullnormal_dispatch_four','scripts.research.pallet_bi_full_execution_v1.fullnormal_recovery') for v in a):active.append(dict(pid=int(p.name),argv=a))
assert not active,active
plan=json.loads((R/'FINAL_SEQUENCE/PLAN.json').read_text())
assert all(sha(p)==v for p,v in plan['sources'].items())
go=dict(h['ROOT_GO_required_fields'],epoch=time.time(),author='ROOT',scope='technical recovery of same frozen scientific1241 rows;4+4workers;no new fits or Fchoice',old_process_owner_check=active,original_interrupted_inmemory_calls='UNKNOWN_NOT_ZERO',implementation_handoff_sha256=sha(R/'FULLNORMAL_RECOVERY_IMPLEMENTATION_HANDOFF.json'),service_parity_sha256=sha(R/'SERVICE_ENVIRONMENT_PARITY_CHECK.json'))
gp=R/'FULLNORMAL_RECOVERY_ROOT_GO.json'
with gp.open('x')as f:json.dump(go,f,sort_keys=True,indent=2);f.write('\n')
gs=sha(gp)
out=dict(schema='ROOT_FULLNORMAL_RECOVERY_SERVICE_LAUNCH_V1',GO_sha256=gs,epoch=time.time(),cwd=str(W),old_owner_absent=True,services={})
for arm in ('S2','F'):
 argv=[gs if s=='ROOT_MUST_INSERT_ACTUAL_GO_SHA256' else s for s in h['proposed_actual_argv'][arm]]
 unit='pallet-bi-full-'+arm.lower()+'-recovery-20261019-v1'
 log=R/(arm+'_FOUR_CPU_RECOVERY_SYSTEMD.log')
 command=['systemd-run','--user','--unit='+unit,'--property=WorkingDirectory='+str(W),'--property=Restart=no','--property=RemainAfterExit=yes','--property=CPUAccounting=yes','--property=MemoryAccounting=yes','--property=StandardOutput=append:'+str(log),'--property=StandardError=append:'+str(log)]
 for key,value in h['environment'].items():command.append('--setenv='+key+'='+value)
 command+=argv
 began=time.time()
 p=subprocess.run(command,text=True,capture_output=True)
 out['services'][arm]=dict(unit=unit,actual_argv=command,scientific_argv=argv,log=str(log),launch_start_epoch=began,launch_end_epoch=time.time(),returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
 assert p.returncode==0,out
with (R/'ROOT_FULLNORMAL_RECOVERY_SERVICE_LAUNCH.json').open('x')as f:json.dump(out,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps(out,indent=2))

