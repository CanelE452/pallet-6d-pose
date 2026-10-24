"""Pinned full MR6D source intake. Metadata seal first, then exact bounded GETs.

No model, optimizer, private data, pose fitting, query correction or labels.
"""
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import argparse,hashlib,http.client,json,shutil,socket,sys,threading,time,urllib.error,urllib.request
import numpy as np

H=Path(__file__).resolve().parent;V=H.parent;P=V.parent;O=V/'full_source_v1';OLD=P/'query_visibility_mr6d_public_native_intake_20261024_v151'
POLICY=V/'ROOT_POLICY_BEFORE_NEW_SOURCE_BODIES.json';POLICY_SHA='46b3b077edfc78d113dcd0b38771bb3df96053a3741002ac774bc1c9866e2334'
COMMIT='3c4ed8453312655499c3d3a3cf38b4929fae77ee';PYTHON='/home/minjae/anaconda3/envs/pallet-yolo26/bin/python'
SELECTION=O/'FULL_SOURCE_SELECTION_BEFORE_NEW_PAYLOADS.json';PLAN=O/'PINNED_CONTENT_PLAN_BEFORE_NEW_PAYLOADS.json'
LOCK=threading.Lock()
class IntegrityFailure(RuntimeError):pass
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def write(p,a):
 with Path(p).open('x')as f:json.dump(a,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def journal(path,row):
 with LOCK:
  with path.open('a')as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
def retryable(exc):
 if isinstance(exc,urllib.error.HTTPError):return exc.code==429 or 500<=exc.code<=599
 return isinstance(exc,(urllib.error.URLError,TimeoutError,ConnectionError,socket.timeout,http.client.IncompleteRead))

def prepare():
 assert sha(POLICY)==POLICY_SHA and not O.exists();policy=read(POLICY);O.mkdir();(H/'directory_metadata').mkdir()
 pin=read(OLD/'metadata/PINNED_REMOTE_SOURCE.json');info=read(OLD/'metadata/HF_DATASET_INFO.json');assert pin['commit']==info['sha']==COMMIT and sha(OLD/'metadata/HF_DATASET_INFO.json')==pin['metadata_sha256']
 split=read(check(policy['fixed_split']));files={x['rfilename']for x in info['siblings']};frames=[];payload=['models/obj_000001.ply'];GTbindings=[]
 for gtpath in sorted((p for p in files if p.endswith('/scene_gt.json')),key=lambda p:(p.split('/')[0],int(p.split('/')[1]))):
  release,scene,_=gtpath.split('/');gt=read(OLD/'provided_metadata'/gtpath);GTbindings.append(bind(OLD/'provided_metadata'/gtpath))
  for key in sorted(gt,key=int):
   indices=[i for i,x in enumerate(gt[key])if int(x['obj_id'])==1]
   if not indices:continue
   assert release=='val'and scene in policy['source_development_scenes']+policy['source_validation_scenes'];stem=f'{int(key):06d}';prefix=f'{release}/{scene}'
   rgb=[x for x in files if x.startswith(prefix+'/rgb/')and Path(x).stem==stem];assert len(rgb)==1
   sf=dict(RGB=rgb[0],depth=f'{prefix}/depth/{stem}.png',full_masks=[],visible_masks=[]);payload.extend([sf['RGB'],sf['depth']])
   for i in indices:
    for folder,field in [('mask','full_masks'),('mask_visib','visible_masks')]:
     path=f'{prefix}/{folder}/{stem}_{i:06d}.png';assert path in files;payload.append(path);sf[field].append(dict(gt_instance_index=i,path=path))
   frames.append(dict(frame_index=len(frames),id=f'V153_MR6D_{release}_{scene}_{stem}',source='MR6D',source_id=f'{release}/{scene}:{key}',
    split='SOURCE_VALIDATION'if scene in policy['source_validation_scenes']else'SOURCE_DEVELOPMENT',official_split=release,
    scene=scene,frame_key=key,frame_filename_stem=stem,obj1_gt_instance_indices=indices,source_files=sf))
 assert len(frames)==230 and len(payload)==len(set(payload))==1067 and sum(len(x['obj1_gt_instance_indices'])for x in frames)==303
 assert Counter(x['split']for x in frames)==Counter(SOURCE_DEVELOPMENT=185,SOURCE_VALIDATION=45)
 write(SELECTION,dict(completed=True,sealed_before_any_new_payload_GET=True,UTC=now(),code=bind(__file__),policy=bind(POLICY),commit=COMMIT,
  fixed_split=policy['fixed_split'],HF_inventory=bind(OLD/'metadata/HF_DATASET_INFO.json'),GT_metadata_bindings=GTbindings,frames=frames,payload_paths=payload,
  ordering='Numeric scene then unmodified numeric JSON frame key; every obj1 GT list index retained.',native_labels_created=0,model_calls=0))
 # Reuse pinned V151 directory metadata, fetching only the four new scene1 directories.
 prior=read(OLD/'factual_sample_v1/XET_ETAG_CONTENT_ID_CORRECTION_BEFORE_REMAINING_GET.json');dirs={r['directory']:r for r in prior['source_metadata']}
 folders=sorted({str(Path(x).parent)for x in payload});metadata=[];meta_journal=H/'METADATA_NETWORK_ATTEMPTS.jsonl'
 def directory(folder):
  if folder in dirs:
   r=dirs[folder];path=check(r['body']);return dict(directory=folder,body=bind(path),bytes=path.stat().st_size,reused_V151=True)
  url=f'https://huggingface.co/api/datasets/anas-gouda/mr6d/tree/{COMMIT}/{folder}';dest=H/'directory_metadata'/(folder.replace('/','_')+'.json')
  for attempt in range(3):
   rec=dict(directory=folder,pinned_url=url,attempt=attempt+1,UTC=now())
   try:
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'MR6D-V153-authorized-full-source/1.0'}),timeout=40)as r:
     assert r.status==200;body=r.read(15000001)
     if len(body)>15000000:raise IntegrityFailure('Directory metadata size cap exceeded')
     if r.headers.get('Link')and'rel="next"'in r.headers['Link']:raise IntegrityFailure('Unexpected paginated directory; stop before payload')
    json.loads(body);dest.write_bytes(body);rec.update(completed=True,bytes=len(body));journal(meta_journal,rec)
    return dict(directory=folder,body=bind(dest),bytes=len(body),reused_V151=False,pinned_url=url)
   except BaseException as exc:
    rec.update(completed=False,error=type(exc).__name__+': '+str(exc),retry=retryable(exc)and attempt<2);journal(meta_journal,rec)
    if not rec['retry']:raise
    time.sleep(attempt+1)
 with ThreadPoolExecutor(max_workers=4)as pool:metadata=list(pool.map(directory,folders))
 assert sum(r['bytes']for r in metadata if not r['reused_V151'])<=policy['intake']['new_metadata_byte_cap']
 entries={}
 for r in metadata:
  for x in read(check(r['body'])):
   if x['type']=='file':entries[x['path']]=x
 assert all(p in entries for p in payload)
 records=[]
 old_receipt=read(OLD/'factual_sample_v1/TINY_SOURCE_DOWNLOAD_COMPLETE.json');reuse={r['relative_path']:r for r in old_receipt['files']};assert len(reuse)==47
 for rel in payload:
  e=entries[rel];size=int(e['size']);assert 0<size<=policy['intake']['max_file_bytes']
  identity=e['lfs']['oid']if e.get('lfs')else e['oid'];kind='LFS_SHA256'if e.get('lfs')else'GIT_BLOB_SHA1'
  assert len(identity)==(64 if e.get('lfs')else 40)
  if e.get('lfs'):assert e['lfs']['size']==size
  records.append(dict(index=len(records),relative_path=rel,pinned_url=f'https://huggingface.co/datasets/anas-gouda/mr6d/resolve/{COMMIT}/{rel}',expected_bytes=size,
   source_content_ID=identity,source_content_ID_kind=kind,opaque_Xet_hash=e.get('xetHash'),reuse_V151=reuse.get(rel)))
 total=sum(r['expected_bytes']for r in records);assert total<=policy['intake']['original_payload_byte_cap']
 assert sum(r['reuse_V151']is not None for r in records)==47
 write(PLAN,dict(completed=True,sealed_before_any_new_payload_GET=True,selection=bind(SELECTION),policy=bind(POLICY),code=bind(__file__),commit=COMMIT,
  directory_metadata=metadata,new_metadata_bytes=sum(r['bytes']for r in metadata if not r['reused_V151']),payloads=records,payload_bytes=total,
  unique_payload_cap=policy['intake']['original_payload_byte_cap'],reuse_files=47,new_payload_GETs=1020,max_concurrency=4,max_network_retries=2,
  checksum_failure_retry=False,automatic_source_replacement=False))
 print(json.dumps(dict(prepared=True,selection=bind(SELECTION),plan=bind(PLAN),frames=230,instances=303,files=1067,
  payload_bytes=total,new_metadata_bytes=sum(r['bytes']for r in metadata if not r['reused_V151']),reuse_files=47,new_GETs=1020)),flush=True)

def verify(path,row):
 raw=Path(path).read_bytes()
 if len(raw)!=row['expected_bytes']:raise IntegrityFailure('File size mismatch '+row['relative_path'])
 content=hashlib.sha256(raw).hexdigest()
 actual=content if row['source_content_ID_kind']=='LFS_SHA256'else hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
 if actual!=row['source_content_ID']:raise IntegrityFailure('Pinned content checksum mismatch '+row['relative_path'])
 return content

def download():
 assert sha(POLICY)==POLICY_SHA;plan=read(PLAN);check(plan['selection']);assert plan['code']==bind(__file__)and plan['policy']==bind(POLICY)
 assert not(O/'PAYLOAD_DOWNLOAD_STARTED.json').exists();write(O/'PAYLOAD_DOWNLOAD_STARTED.json',dict(completed=True,UTC=now(),plan=bind(PLAN),policy=bind(POLICY),network_retry_policy='2 identical pinned retries only for eligible network errors',model_calls=0))
 journal_dir=O/'download_receipts';journal_dir.mkdir();log=O/'PAYLOAD_NETWORK_ATTEMPTS.jsonl';done=[]
 def fetch(row):
  dest=O/'original'/row['relative_path'];dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists()
  if row['reuse_V151']is not None:
   prior=row['reuse_V151'];source=Path(prior['local_path']);assert sha(source)==prior['sha256'];value=verify(source,row);shutil.copyfile(source,dest);assert verify(dest,row)==value
   rec=dict(row,completed=True,local_path=str(dest),bytes=dest.stat().st_size,sha256=value,source_content_ID_verified=True,GET_performed=False,reused_verified_original=True,attempts=0)
  else:
   for attempt in range(3):
    tmp=dest.with_name(dest.name+f'.attempt{attempt+1}.partial');record=dict(index=row['index'],relative_path=row['relative_path'],pinned_url=row['pinned_url'],attempt=attempt+1,UTC=now())
    try:
     with urllib.request.urlopen(urllib.request.Request(row['pinned_url'],headers={'User-Agent':'MR6D-V153-authorized-full-source/1.0'}),timeout=40)as response:
      assert response.status==200;total=0
      with tmp.open('xb')as stream:
       while True:
        block=response.read(262144)
        if not block:break
        total+=len(block)
        if total>row['expected_bytes']:raise IntegrityFailure('Source size exceeded pinned metadata')
        stream.write(block)
      value=verify(tmp,row);ETag=response.headers.get('ETag')
     tmp.rename(dest);record.update(completed=True,bytes=total,opaque_HTTP_ETag=ETag);journal(log,record)
     rec=dict(row,completed=True,local_path=str(dest),bytes=total,sha256=value,source_content_ID_verified=True,GET_performed=True,reused_verified_original=False,attempts=attempt+1,opaque_HTTP_ETag=ETag);break
    except BaseException as exc:
     record.update(completed=False,error=type(exc).__name__+': '+str(exc),partial_path=str(tmp),partial_bytes=tmp.stat().st_size if tmp.exists()else 0,
      retry=not isinstance(exc,IntegrityFailure)and retryable(exc)and attempt<2);journal(log,record)
     if not record['retry']:raise
     time.sleep(attempt+1)
  write(journal_dir/f"{row['index']:04d}.json",rec);return rec
 try:
  with ThreadPoolExecutor(max_workers=4)as pool:
   futures=[pool.submit(fetch,r)for r in plan['payloads']]
   for future in as_completed(futures):
    done.append(future.result())
    if len(done)%25==0 or len(done)==1067:print('verified',len(done),1067,'bytes',sum(r['bytes']for r in done),flush=True)
  done.sort(key=lambda r:r['index']);assert len(done)==1067 and sum(r['GET_performed']for r in done)==1020
  assert sum(r['bytes']for r in done)==plan['payload_bytes']and sum(not r['GET_performed']for r in done)==47
  receipt=O/'FULL_SOURCE_DOWNLOAD_COMPLETE.json';write(receipt,dict(completed=True,PASS=True,UTC=now(),code=bind(__file__),policy=bind(POLICY),plan=bind(PLAN),selection=plan['selection'],
   files=done,unique_payload_bytes=sum(r['bytes']for r in done),new_metadata_bytes=plan['new_metadata_bytes'],source_content_ID_all1067_exact=True,
   original47_reused=True,new_payload_GETs=1020,max_concurrency=4,max_identical_network_retries=2,checksum_failure_retry=False,source_replacement=False,model_or_fit_calls=0))
  print(json.dumps(dict(download_complete=bind(receipt),unique_payload_bytes=plan['payload_bytes'])),flush=True)
 except BaseException as exc:
  write(O/'FULL_SOURCE_DOWNLOAD_FAILURE_PRESERVED.json',dict(completed=False,UTC=now(),exception=type(exc).__name__+': '+str(exc),successful_receipts=len(list(journal_dir.glob('*.json'))),
   plan=bind(PLAN),all_partials_and_failure_attempts_preserved=True,automatic_restart=False));raise

def factual():
 import cv2,PIL
 from PIL import Image,ImageDraw,ImageFont
 assert Path(sys.executable).resolve()==Path(PYTHON).resolve()and cv2.__version__=='4.9.0'and np.__version__=='1.26.4';cv2.setNumThreads(1)
 download=read(O/'FULL_SOURCE_DOWNLOAD_COMPLETE.json');assert download['completed']and download['PASS'];selection=read(check(download['selection']));frames=selection['frames']
 bypath={r['relative_path']:r for r in download['files']};canonical=O/'canonical_RGB';canonical.mkdir();records=[];RGBrows=[];thumbs={};scenecache={};PIL_equal=0
 for r in frames:
  i=r['frame_index'];sf=r['source_files'];ip=O/'original'/sf['RGB'];dp=O/'original'/sf['depth'];assert sha(ip)==bypath[sf['RGB']]['sha256']and sha(dp)==bypath[sf['depth']]['sha256']
  bgr=cv2.imread(str(ip),cv2.IMREAD_COLOR);rgb=np.ascontiguousarray(bgr[:,:,::-1]);pil=np.array(Image.open(ip).convert('RGB'),np.uint8);assert rgb.shape==pil.shape and rgb.dtype==np.uint8
  dep=np.array(Image.open(dp));assert dep.dtype==np.uint16 and dep.shape==rgb.shape[:2];delta=rgb.astype(np.int16)-pil.astype(np.int16);same=not np.any(delta);PIL_equal+=same
  npy=canonical/f'RGB_{i:03d}.npy';np.save(npy,rgb,allow_pickle=False);assert np.load(npy,allow_pickle=False).tobytes()==rgb.tobytes()
  RGBrows.append(dict(frame_index=i,id=r['id'],source='MR6D',source_id=r['source_id'],split=r['split'],image=str(ip),image_sha256=bypath[sf['RGB']]['sha256'],
   decoded_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),raw_hw=list(rgb.shape[:2]),decoder='PINNED_CV2_IMREAD_COLOR_BGR_TO_RGB',canonical_RGB_NPY=bind(npy)))
  key=(r['official_split'],r['scene'])
  if key not in scenecache:
   pref=OLD/'provided_metadata'/r['official_split']/r['scene'];scenecache[key]=(read(pref/'scene_gt.json'),read(pref/'scene_gt_info.json'),read(pref/'scene_camera.json'))
  gt,gi,cam=scenecache[key];instances=[];visible=[];full=[]
  for ix in r['obj1_gt_instance_indices']:
   assert gt[r['frame_key']][ix]['obj_id']==1
   fp=next(x['path']for x in sf['full_masks']if x['gt_instance_index']==ix);vp=next(x['path']for x in sf['visible_masks']if x['gt_instance_index']==ix)
   assert sha(O/'original'/fp)==bypath[fp]['sha256']and sha(O/'original'/vp)==bypath[vp]['sha256']
   fm=np.array(Image.open(O/'original'/fp))!=0;vm=np.array(Image.open(O/'original'/vp))!=0;assert fm.shape==vm.shape==dep.shape
   full.append(fm);visible.append(vm);instances.append(dict(gt_instance_index=ix,obj_id=1,full_pixels=int(fm.sum()),visible_pixels=int(vm.sum()),visible_outside_full=int((vm&~fm).sum()),
    provided_GT_info=gi[r['frame_key']][ix],semantic_native_labels='UNASSIGNED'))
  overlaps=[dict(GT_indices=[r['obj1_gt_instance_indices'][a],r['obj1_gt_instance_indices'][b]],full_overlap=int((full[a]&full[b]).sum()),visible_overlap=int((visible[a]&visible[b]).sum()))for a in range(len(visible))for b in range(a+1,len(visible))]
  records.append(dict(frame_index=i,id=r['id'],source_id=r['source_id'],split=r['split'],official_split=r['official_split'],scene=r['scene'],frame_key=r['frame_key'],raw_RGB_HW=list(rgb.shape[:2]),
   decoded_RGB_sha256=RGBrows[-1]['decoded_RGB_sha256'],PIL_diagnostic_equal=same,PIL_changed_pixels=int(np.any(delta!=0,axis=2).sum()),PIL_max_channel_difference=int(np.abs(delta).max()),
   depth_dtype=str(dep.dtype),depth_nonzero=int((dep>0).sum()),depth_scale_to_mm=cam[r['frame_key']]['depth_scale'],instances=instances,instance_pair_overlaps=overlaps))
  image=Image.fromarray(rgb);image.thumbnail((316,178),Image.Resampling.NEAREST);thumbs[i]=image
  if(i+1)%25==0:print('decoded',i+1,230,flush=True)
 assert len(records)==230 and sum(len(r['instances'])for r in records)==303
 manifest=O/'RGB230_INPUT_ONLY_MANIFEST.json';write(manifest,dict(completed=True,rows=RGBrows,source_selection=download['selection'],
  original_R0_decoder='Pinnedcv2 exact V152 callsite; canonicalNPY explicit for future C111, PIL diagnostic only.',GT_pose_mask_depth_or_label_fields_copied=0))
 facts=O/'FRAME230_FACTUAL_SUMMARY.json';write(facts,dict(completed=True,frames=records,decoder_runtime=dict(python=sys.executable,Python=sys.version,NumPy=np.__version__,cv2=cv2.__version__,Pillow=PIL.__version__),
  PIL_equal_frames=PIL_equal,all303_instance_indices_retained=True,nonexclusive_masks_not_ownership_gold=True,native_semantic_labels_created=0))
 contactdir=O/'raw_overview_contacts';contactdir.mkdir();boards=[];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
 for scene in sorted({r['scene']for r in frames},key=int):
  group=[r for r in frames if r['scene']==scene]
  for start in range(0,len(group),24):
   subset=group[start:start+24];sheet=Image.new('RGB',(1312,1320),'#edf2f7');draw=ImageDraw.Draw(sheet)
   draw.text((14,10),f"MR6D raw factual RGB | scene{scene} | {group[0]['split']} | all frames preserved; native semantics unassigned",font=font,fill='#18334c')
   for pos,r in enumerate(subset):
    x=8+(pos%4)*328;y=42+(pos//4)*211;draw.text((x,y),f"#{r['frame_index']:03d} frame{r['frame_key']} / obj1 instances:{len(r['obj1_gt_instance_indices'])}",font=font,fill='#18334c');sheet.paste(thumbs[r['frame_index']],(x,y+23))
   path=contactdir/f'SCENE_{scene}_PAGE_{start//24+1:02d}.png';sheet.save(path);boards.append(bind(path))
 seal=O/'FULL_SOURCE_INTAKE_COMPLETE.json';write(seal,dict(completed=True,PASS=True,UTC=now(),code=bind(__file__),policy=bind(POLICY),selection=download['selection'],download=bind(O/'FULL_SOURCE_DOWNLOAD_COMPLETE.json'),
  RGB_manifest=bind(manifest),source_download_seal=bind(O/'FULL_SOURCE_DOWNLOAD_COMPLETE.json'),canonical_decoder='PINNED_CV2_IMREAD_COLOR_BGR_TO_RGB',
  factual_summary=bind(facts),raw_contact_pages=boards,frames=230,rows=1840,development_frames=185,validation_frames=45,GT_instance_observations=303,
  original_payload_files=1067,unique_original_payload_bytes=download['unique_payload_bytes'],canonical_RGB_arrays=230,lossless_original_decoder_transport=True,PIL_equal_frames=PIL_equal,
  all_source_split_and_instance_inventory_unchanged=True,new_native_labels_created=0,model_initializer_encoder_fit_private_PnP_calls=0,source_validation_not_used_for_training=True,
  limits=['Public released val is development source, not an official blind benchmark.','Instance masks can overlap; no exclusive owner/native visibility auto-labels.','Source nominal model/pose has centimeter uncertainty; no exact3px gold.','These Euro wood scenes do not fill real square-pallet supervision.']))
 print(json.dumps(dict(completed=True,PASS=True,intake_seal=bind(seal),RGB_manifest=bind(manifest),frames=230,instances=303,contact_pages=len(boards),PIL_equal=PIL_equal)),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--download-and-factual',action='store_true');a=ap.parse_args();assert a.prepare!=a.download_and_factual
 if a.prepare:prepare()
 else:download();factual()
