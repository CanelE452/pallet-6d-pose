import json,hashlib,os,time,datetime,urllib.request,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent; V=P.parent
D=Path('/dev/shm/pallet_v92_qwen3vl2b_89644892_v1')
PLAN=V/'model_feasibility/QWEN3VL2B_NATIVE_ALLOWLIST_PLAN.json'
REVIEW=V/'independent_qwen3_pilot_review/METADATA_PRECHECK.json'
ENTRY=P/'ROOT_WEIGHT_ONLY_INTAKE_ENTRY_V1.json'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def wr(p,d): p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
if not ENTRY.exists():
 wr(ENTRY,dict(schema='V92_ROOT_WEIGHT_ONLY_INTAKE_AUTHORIZATION',approved=True,created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_SHA256=sha(PLAN),independent_metadata_precheck_SHA256=sha(REVIEW),model='Qwen/Qwen3-VL-2B-Instruct',revision='89644892e4d85e24eaac8bacfd4f463576704203',weight_bytes=4255140312,weight_SHA256='7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0',destination=str(D),scope='weight-only byte intake, no tensor deserialization, processor/constructor/forward/current/fit/PnP0; inference still needs separate frozen protocol/generated/review/rootentry',minimum_available_RAM_before_download=12000000000,minimum_tmpfs_after_weight=2000000000,minimum_disk_reserve=500000000,minimum_free_GPU_MB=7000,maximum_download_seconds=600,one_weight_copy_only=True))
 print('ENTRY_READY',sha(ENTRY),flush=True);raise SystemExit(0)
e=json.loads(ENTRY.read_text());assert e['approved'] and sha(PLAN)==e['plan_SHA256'] and sha(REVIEW)==e['independent_metadata_precheck_SHA256']
assert not (P/'ROOT_WEIGHT_ONLY_INTAKE_COMPLETE_V1.json').exists()
plan=json.loads(PLAN.read_text());w=next(x for x in plan['files'] if x['name']=='model.safetensors');assert w['bytes']==e['weight_bytes'] and w['lfs_sha256']==e['weight_SHA256']
assert not D.is_symlink();D.mkdir(exist_ok=True)
f=D/'model.safetensors';part=D/'model.safetensors.part';assert not f.exists() and not part.exists()
m={x.split(':')[0]:int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')}
sv=os.statvfs(D);free=sv.f_bavail*sv.f_frsize;dv=os.statvfs(V);disk=dv.f_bavail*dv.f_frsize
gpu=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
assert m['MemAvailable']>=e['minimum_available_RAM_before_download'] and free-e['weight_bytes']>=e['minimum_tmpfs_after_weight'] and disk>=e['minimum_disk_reserve'] and gpu>=e['minimum_free_GPU_MB'],(m,free,disk,gpu)
t=time.monotonic();n=0;h=hashlib.sha256();last=0
wr(P/'ROOT_WEIGHT_ONLY_RESOURCE_BEFORE_V1.json',dict(memory_available=m['MemAvailable'],tmpfs_available=free,rootdisk_available=disk,GPU_free_MB=gpu,entry_SHA256=sha(ENTRY),downloader_SHA256=sha(Path(__file__))))
url='https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/'+e['revision']+'/model.safetensors'
try:
 req=urllib.request.Request(url,headers={'User-Agent':'pallet-v92-source-only-intake/1'})
 with urllib.request.urlopen(req,timeout=30) as r,part.open('xb') as out:
  assert r.status==200
  while True:
   b=r.read(4*1024*1024)
   if not b:break
   n+=len(b);assert n<=e['weight_bytes'];out.write(b);h.update(b)
   assert time.monotonic()-t<=e['maximum_download_seconds']
   if n-last>=256*1024*1024:print(json.dumps(dict(bytes=n,expected=e['weight_bytes'],seconds=round(time.monotonic()-t,2))),flush=True);last=n
  out.flush();os.fsync(out.fileno())
 assert n==e['weight_bytes'] and h.hexdigest()==e['weight_SHA256'],(n,h.hexdigest())
 part.rename(f)
 wr(P/'ROOT_WEIGHT_ONLY_INTAKE_COMPLETE_V1.json',dict(completed=True,entry_SHA256=sha(ENTRY),downloader_SHA256=sha(Path(__file__)),file=str(f),bytes=n,sha256=h.hexdigest(),seconds=time.monotonic()-t,source_model=e['model'],revision=e['revision'],new_model_deserialization_or_forward_or_fit_or_current_or_PnP=0,ephemeral_tmpfs=True))
 print('WEIGHT_ONLY_COMPLETE',h.hexdigest(),flush=True)
except BaseException as x:
 wr(P/'ROOT_WEIGHT_ONLY_INTAKE_FAILURE_V1.json',dict(completed=False,error_type=type(x).__name__,error_message=str(x)[:200] if not isinstance(x,urllib.error.HTTPError) else 'HTTP status '+str(x.code),partial=str(part),partial_bytes=n,partial_SHA256=h.hexdigest(),seconds=time.monotonic()-t,retained=True));raise
