import json,pathlib,time,subprocess,hashlib,resource
R=pathlib.Path("/home/minjae/Documents/pallet-research/pallet-bi-full-execution-20261019-v1");W=pathlib.Path("/tmp/pallet-bi-full-execution-20261019-v1")
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def put(p,v):
 with p.open('x')as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n')
h=json.loads((R/'DERIVED_COMPLETION_IMPLEMENTATION_HANDOFF.json').read_text())
assert sha(h['source'])==h['source_sha256']=='42fd83e31b47e89aaa94c4b7bf6b5573c7eaecec1c90a6251fcaf72a5f69a03d'
record=dict(schema='ROOT_DERIVED_COMPLETION_ACTUAL_LAUNCH_V1',epoch=time.time(),handoff_sha256=sha(R/'DERIVED_COMPLETION_IMPLEMENTATION_HANDOFF.json'),source_sha256=h['source_sha256'],root_review='fullcontroller+testsource+existingchildAPIs read;18fixturetests;no blockingissue',prepare_argv=h['prepare_argv'])
b=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.monotonic();p=subprocess.run(h['prepare_argv'],cwd=W,text=True,capture_output=True,env=None);a=resource.getrusage(resource.RUSAGE_CHILDREN)
record['prepare']=dict(returncode=p.returncode,wall_seconds=time.monotonic()-start,child_CPU_seconds=a.ru_utime+a.ru_stime-b.ru_utime-b.ru_stime,stdout=p.stdout,stderr=p.stderr)
assert p.returncode==0,record
pp=pathlib.Path(h['output'])/'PLAN.json';plan=json.loads(pp.read_text())
assert len(plan['scatter_pairs'])==23 and len(plan['requested_unique_arms'])==40
assert all(sha(p)==s for p,s in plan['sources'].items())
g=h['ROOT_GO_required_fields'];g['plan_sha256']=sha(pp);g['epoch']=time.time();g['authority']='user full campaign;ROOT reviewed existing PRIMARY rights/privacy and serialpostsealderivedscope';g['publication_by_this_GO']=False
gp=pathlib.Path(h['ROOT_GO_path']);put(gp,g)
argv=h['proposed_ROOT_user_service_argv'];p=subprocess.run(argv,cwd=W,text=True,capture_output=True)
record.update(actual_service_argv=argv,service_returncode=p.returncode,service_stdout=p.stdout,service_stderr=p.stderr,plan_sha256=sha(pp),GO_sha256=sha(gp),actual_service_start_request_epoch=time.time(),new_fits=0,new_solves=0)
put(R/'ROOT_DERIVED_COMPLETION_ACTUAL_LAUNCH.json',record)
assert p.returncode==0,record
print(json.dumps(record,indent=2))

