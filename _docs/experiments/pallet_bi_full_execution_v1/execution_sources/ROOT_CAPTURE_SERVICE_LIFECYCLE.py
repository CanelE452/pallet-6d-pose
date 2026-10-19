"""Metadata-only lifecycle capture for the four ROOT transient research units."""
import json,pathlib,time,subprocess,hashlib,os
R=pathlib.Path("/home/minjae/Documents/pallet-research/pallet-bi-full-execution-20261019-v1")
O=R/'ROOT_SERVICE_LIFECYCLE_V1'
O.mkdir(exist_ok=False)
UNITS=['pallet-bi-full-s2-recovery-20261019-v1.service','pallet-bi-full-f-recovery-20261019-v1.service','pallet-bi-full-s2-h1-20261019-v1.service','pallet-bi-full-final-sequence-20261019-v1.service']
PROP=['Id','ActiveState','SubState','MainPID','Result','ExecMainStatus','ExecMainCode','CPUUsageNSec','MemoryCurrent','MemoryPeak','ExecMainStartTimestampMonotonic','ExecMainExitTimestampMonotonic','ActiveEnterTimestampMonotonic','StateChangeTimestampMonotonic','NRestarts','TasksCurrent']
def put(p,v):
 with p.open('x')as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n')
put(O/'LOCK.json',dict(schema='ROOT_TRANSIENT_SERVICE_LIFECYCLE_V1',source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),units=UNITS,properties=PROP,new_scientific_calls=0,new_reference_reads=0,epoch=time.time()))
while True:
 records=[]
 for unit in UNITS:
  cmd=['systemctl','--user','show',unit,'--property='+','.join(PROP)]
  p=subprocess.run(cmd,text=True,capture_output=True)
  d=dict(x.split('=',1)for x in p.stdout.splitlines()if '='in x)
  d['unit']=unit;d['returncode']=p.returncode;d['stderr']=p.stderr
  if d.get('MainPID','0')!='0':
   pp=pathlib.Path('/proc')/d['MainPID']
   try:
    d['actual_process_argv']=[v.decode()for v in (pp/'cmdline').read_bytes().split(b'\0')if v]
    d['process_startticks']=(pp/'stat').read_text().rsplit(')',1)[1].split()[19]
    d['children']=(pp/'task'/d['MainPID']/'children').read_text().split()
   except(FileNotFoundError,ProcessLookupError):d['proc_disappeared_during_snapshot']=True
  records.append(d)
 snapshot=dict(epoch=time.time(),units=records,new_scientific_calls=0)
 with (O/'SNAPSHOTS.jsonl').open('a')as f:f.write(json.dumps(snapshot,sort_keys=True)+'\n')
 done=all(x.get('SubState')in('exited','failed','dead')for x in records)
 if done:
  put(O/'COMPLETION.json',dict(complete=True,scientific_completion_claim=False,last_snapshot=snapshot,snapshot_sha256=hashlib.sha256((O/'SNAPSHOTS.jsonl').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()))
  break
 time.sleep(30)

