import pathlib,json,hashlib,subprocess,time,resource,os,csv
R=pathlib.Path("/home/minjae/Documents/pallet-research/pallet-bi-full-execution-20261019-v1");W=pathlib.Path("/tmp/pallet-bi-full-execution-20261019-v1");PY="/home/minjae/anaconda3/envs/pallet-pose/bin/python"
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
source=W/'scripts/research/pallet_bi_full_execution_v1/error_budget.py'
assert sha(source)=='b534ecce4426121d3d0c7a773e310f7453bdd1f8ac6f1145c1af7863074c62a2'
export=R/'S1_S3_S4_PUBLIC_METRICS';seal=sha(export/'EXPORT_SEAL.json')
prior=json.loads((export/'EXPORT_SEAL.json').read_text());assert prior['complete'] and len(prior['arms'])==25
for n,s in prior['artifacts'].items():assert sha(export/n)==s
out=R/'ERROR_BUDGET25_V1';receipt=R/'ROOT_ERROR_BUDGET25_EXECUTION.json'
assert not out.exists() and not receipt.exists()
argv=[PY,'-B','-m','scripts.research.pallet_bi_full_execution_v1.error_budget','--export-dir',str(export),'--export-seal-sha256',seal,'--output',str(out),'--expected-arm-count','25']
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(W))
start=time.monotonic();b=resource.getrusage(resource.RUSAGE_CHILDREN)
with (R/'ROOT_ERROR_BUDGET25.log').open('x')as f:p=subprocess.run(argv,cwd=W,env=env,stdout=f,stderr=subprocess.STDOUT)
a=resource.getrusage(resource.RUSAGE_CHILDREN)
v=dict(schema='ROOT_ERROR_BUDGET25_ACTUAL_EXECUTION_V1',scope='Explicit initial25 completed arms only;notS2/Ffinalresults',actual_argv=argv,worktree=str(W),environment={k:env[k]for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','PYTHONPATH']},source_sha256=sha(source),trusted_export_seal_sha256=seal,returncode=p.returncode,wall_seconds=time.monotonic()-start,child_CPU_seconds=a.ru_utime+a.ru_stime-b.ru_utime-b.ru_stime,epoch=time.time(),new_pose_metric_calls=0,new_solves=0,new_fits=0,posthoc_descriptive=True)
if p.returncode==0:v['output_seal_sha256']=sha(out/'ERROR_BUDGET_SEAL.json')
with receipt.open('x')as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n')
assert p.returncode==0,v
d=json.loads((out/'ERROR_BUDGET_SEAL.json').read_text())
print(json.dumps(dict(execution=v,conservation={k:d[k]for k in ['campaign_scope','joint_rows','band_rows','conservation_rows','maximum_mean_absolute_residual','maximum_variance_absolute_residual']}),indent=2))
rows=list(csv.DictReader((out/'ERROR_BANDS.csv').open()))
print('REPRESENTATIVE_ORIGINAL_PRIMARY_BANDS')
print(json.dumps([x for x in rows if x['population']=='PRIMARY192' and x['arm']=='D1_H1_MLP_s20261018'],indent=2))

