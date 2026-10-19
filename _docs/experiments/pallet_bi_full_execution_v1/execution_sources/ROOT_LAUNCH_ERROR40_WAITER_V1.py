import pathlib,json,time,subprocess,hashlib,resource,os
R=pathlib.Path('/home/minjae/Documents/pallet-research/pallet-bi-full-execution-20261019-v1');W=pathlib.Path('/tmp/pallet-bi-full-execution-20261019-v1')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def put(p,v):
 with p.open('x') as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n')
hp=R/'ERROR40_WAITER_IMPLEMENTATION_HANDOFF.json';h=json.loads(hp.read_text())
assert sha(hp)=='54fea2aa14b30a40990eebc0d355610326246fbe5d344187d931edcfe478270e'
assert sha(h['source'])==h['source_sha256']=='b6aa65d27ff68a86dce4009344a2a26bb01374ddb3880b5f3a1a327143c266dd'
assert sha(h['test_source'])==h['test_source_sha256']
assert not (R/'ERROR_BUDGET40_WAITER_V1').exists() and not (R/'ERROR_BUDGET40_V1').exists()
for key in ('sequence','derived'):assert not h['active_closures'][key]['changed']
env=dict(os.environ,PYTHONPATH=str(W),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
b=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.monotonic();before=time.time()
p=subprocess.run(h['prepare_argv'],cwd=W,env=env,text=True,capture_output=True)
a=resource.getrusage(resource.RUSAGE_CHILDREN)
receipt=dict(schema='ROOT_ERROR40_WAITER_ACTUAL_PREPARE_V1',actual_argv=h['prepare_argv'],outer_environment={k:env[k] for k in ('PYTHONPATH','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},cwd=str(W),returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,epoch=time.time(),launch_start_epoch=before,wall_seconds=time.monotonic()-start,child_CPU_seconds=a.ru_utime+a.ru_stime-b.ru_utime-b.ru_stime,implementation_handoff_sha256=sha(hp),new_scientific_calls=0)
put(R/'ROOT_ERROR40_WAITER_ACTUAL_PREPARE_V1.json',receipt);assert p.returncode==0,receipt
pp=R/'ERROR_BUDGET40_WAITER_V1/PLAN.json';plan=json.loads(pp.read_text())
assert all(sha(p)==s for p,s in plan['sources'].items()) and all(sha(p)==s for p,s in plan['fixed_inputs'].items())
g=dict(h['ROOT_GO_required_fields']);g['plan_sha256']=sha(pp);g.update(epoch=time.time(),authority='ROOT full-source review and independent review;user-authorized scalar-only final40 audit,one child after closed derived output;no new fit/solve')
gp=pathlib.Path(h['ROOT_GO_path']);put(gp,g)
argv=h['proposed_ROOT_systemd_argv'];began=time.time();p=subprocess.run(argv,cwd=W,text=True,capture_output=True)
v=dict(schema='ROOT_ERROR40_WAITER_ACTUAL_LAUNCH_V1',epoch=time.time(),launch_start_epoch=began,actual_service_argv=argv,unit='pallet-bi-error40-waiter-20261019-v1',returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,source_sha256=sha(h['source']),plan_sha256=sha(pp),GO_sha256=sha(gp),implementation_handoff_sha256=sha(hp),new_scientific_calls=0,one_future_scalar_only_child=True,campaign_complete=False)
put(R/'ROOT_ERROR40_WAITER_ACTUAL_LAUNCH.json',v);assert p.returncode==0,v
print(json.dumps(v,indent=2))
