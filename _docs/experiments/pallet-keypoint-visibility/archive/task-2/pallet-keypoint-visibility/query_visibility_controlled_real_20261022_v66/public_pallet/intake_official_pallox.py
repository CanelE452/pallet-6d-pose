#!/usr/bin/env python3
"""Official fixed-version Pallox quarantine intake. No training/inference/PnP."""
from pathlib import Path,PurePosixPath
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import defaultdict,Counter
import argparse,csv,datetime,hashlib,json,threading,time
import requests

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data';DOI='doi:10.48804/DJLDFS'
API='https://rdr.kuleuven.be/api/access/datafile/'
threadstate=threading.local()
def sha(data):return hashlib.sha256(data).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
def metadata():
 raw=json.loads((ROOT/'DATASET_V1_OFFICIAL.json').read_text());assert raw['status']=='OK';d=raw['data'];assert d['versionNumber']==1 and d['versionMinorNumber']==0
 records=[]
 for entry in d['files']:
  f=entry['dataFile'];rel=PurePosixPath(entry.get('directoryLabel',''))/f['filename'];assert not rel.is_absolute()and'..'not in rel.parts
  checksum=f['checksum'];assert checksum['type']=='MD5';assert not entry.get('restricted',False)
  records.append(dict(file_id=f['id'],relative_path=str(rel),bytes=f['filesize'],MD5=checksum['value'],url=API+str(f['id']),dataset_version='1.0',restricted=False))
 assert len(records)==1234 and sum(r['bytes']for r in records)==193208876
 return d,records
def session_frames(records):
 files={r['relative_path']:r for r in records};sessions=sorted({r['relative_path'].split('/')[0]for r in records if'/rgb/'in r['relative_path']});joined=[];missing=[]
 for session in sessions:
  rgbs=sorted(k for k in files if k.startswith(session+'/rgb/'))
  for image in rgbs:
   stem=Path(image).stem;paths=dict(image=image,depth=f'{session}/depth/{stem}.png',mask_teacher=f'{session}/masks/{stem}_sam1_mask.png',pose=f'{session}/annotated_poses/{stem}.txt',K=f'{session}/cam_K.txt')
   row=dict(id=session+':'+stem,session=session,frame_id=stem,split='TRAIN_LAB'if session.endswith('_lab')else'VAL_HALL',**paths)
   absent=[k for k,p in paths.items()if p not in files]
   if absent:row['missing']=absent;missing.append(row)
   else:joined.append(row)
 return joined,missing
def initialize():
 d,records=metadata();joined,missing=session_frames(records);p=ROOT/'SPLIT_LOCK_BEFORE_FIT.json';assert not p.exists()
 split=dict(schema='PUBLISHED_PALLOX_SESSION_SPLIT_QUARANTINE_V1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),dataset_DOI=DOI,version='1.0',API_metadata_sha256=sha((ROOT/'DATASET_V1_OFFICIAL.json').read_bytes()),fit_has_started=False,admission='HOLD_PENDING_SOURCE_AND_FAMILY_AUDITS',assignment='all lab sessions TRAIN; all hall sessions VAL. Whole environment/session split, no per-frame random split.',counts=dict(Counter(r['split']for r in joined)),fully_joined=len(joined),RGB_without_complete_annotation=len(missing),frames=joined,excluded_incomplete=missing)
 assert split['counts']=={'TRAIN_LAB':168,'VAL_HALL':116};write(p,split);write(ROOT/'LICENSE_OFFICIAL.json',dict(dataset_license=d['license'],version='1.0',metadata_sha256=split['API_metadata_sha256'],dataset_DOI=DOI,paper_license_is_separate=True))
 (ROOT/'LICENSE.txt').write_text('Dataset: Replication Data for:6D Pose Estimation of Pallet Bins for Autonomous Logistics Using a Synthetically Trained Object Detector and FoundationPose\nDOI: https://doi.org/10.48804/DJLDFS\nVersion:1.0\nLicense: Creative Commons Attribution4.0 International (CC-BY-4.0)\nOfficial license: https://creativecommons.org/licenses/by/4.0/\nAuthors: Yanming Wu; Eric Demeester (Associatie KU Leuven)\nSource: fixed-version KU Leuven RDR API metadata, preserved in DATASET_V1_OFFICIAL.json. The publication license does not replace this dataset license.\n')
 with(ROOT/'OFFICIAL_CATALOG.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 selected=set()
 for session in sorted({r['session']for r in joined}):
  r=next(x for x in joined if x['session']==session)
  selected.update(r[k]for k in ['image','depth','mask_teacher','pose','K'])
 selected.update(r['relative_path']for r in records if r['relative_path']=='README.md'or r['relative_path'].startswith('assets/mesh/'))
 fixed=[r for r in records if r['relative_path']in selected]
 write(ROOT/'INITIAL_FIXED_SAMPLE_LOCK.json',dict(selection='First fully joined frame in each of six sessions, plus shared mesh/material/texture, K and README. Chosen before RGB inspection.',files=fixed,bytes=sum(r['bytes']for r in fixed),training=False,current_model_inference=False,PnP=False))
 print('LOCKED',len(joined),split['counts'],'initial_files',len(fixed),'bytes',sum(r['bytes']for r in fixed),flush=True)
def fetch(row):
 path=DATA/row['relative_path'];path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():
  body=path.read_bytes();assert len(body)==row['bytes']and hashlib.md5(body).hexdigest()==row['MD5'];return dict(**row,sha256=sha(body),downloaded_bytes=0,verified_existing=True)
 if not hasattr(threadstate,'session'):
  threadstate.session=requests.Session();threadstate.session.headers['User-Agent']='PublicDatasetIntegrityAudit/1.0 (official-repository-replication)'
 for attempt in range(6):
  try:
   response=threadstate.session.get(row['url'],timeout=(15,90))
   if response.status_code in [429,500,502,503,504]:
    retry=response.headers.get('Retry-After','');delay=min(60,float(retry))if retry.replace('.','',1).isdigit()else min(30,2**attempt);time.sleep(delay);continue
   response.raise_for_status();body=response.content
   assert len(body)==row['bytes'],('length',row['relative_path'],len(body),row['bytes'])
   assert hashlib.md5(body).hexdigest()==row['MD5'],('MD5',row['relative_path'])
   temp=path.with_name(path.name+'.partial');temp.write_bytes(body);temp.replace(path)
   return dict(**row,sha256=sha(body),downloaded_bytes=len(body),verified_existing=False)
  except(requests.RequestException,AssertionError):
   if attempt==5:raise
   time.sleep(min(30,2**attempt))
 raise RuntimeError('retry budget exhausted '+row['relative_path'])
def download(stage,workers):
 d,records=metadata();assert(ROOT/'SPLIT_LOCK_BEFORE_FIT.json').exists()
 if stage=='sample':todo=json.loads((ROOT/'INITIAL_FIXED_SAMPLE_LOCK.json').read_text())['files']
 else:todo=records
 output=ROOT/f'{stage.upper()}_DOWNLOAD_INVENTORY.jsonl';assert not output.exists()
 total=sum(r['bytes']for r in todo);done=0;downloaded=0;start=time.monotonic();results=[]
 with output.open('w')as stream,ThreadPoolExecutor(max_workers=workers)as executor:
  tasks={executor.submit(fetch,r):r for r in todo}
  for job in as_completed(tasks):
   result=job.result();results.append(result);stream.write(json.dumps(result)+'\n');stream.flush();done+=1;downloaded+=result['downloaded_bytes']
   if done%50==0 or done==len(todo):print(stage,'files',done,'/',len(todo),'verified_MB',round(sum(r['bytes']for r in results)/1e6,2),'new_MB',round(downloaded/1e6,2),'seconds',round(time.monotonic()-start,1),flush=True)
 write(ROOT/f'{stage.upper()}_DOWNLOAD_SEAL.json',dict(stage=stage,files=len(results),catalog_bytes=total,new_downloaded_bytes=downloaded,MD5_all_pass=True,SHA256_recorded=True,inventory_sha256=sha(output.read_bytes()),version='1.0',DOI=DOI,source_only=True,training=False,current_model_inference=False,PnP=False,seconds=time.monotonic()-start))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['initialize','sample','all']);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
 if a.command=='initialize':initialize()
 else:download(a.command,a.workers)
