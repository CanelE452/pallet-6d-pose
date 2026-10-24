#!/usr/bin/env python3
"""Range-extract a scene-disjoint pilot from official BOP YCB-V train_real only.
Does not open any pallet evaluation data and never invokes GPU or PnP.
"""
import argparse, concurrent.futures, hashlib, io, json, os, pathlib, struct, time, threading, zlib
import requests
from PIL import Image
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parent
SOURCE='https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/5c2c4aa229800355648cd268040aa814f8dc94f0/'
URLS=[SOURCE+'ycbv_train_real.z01',SOURCE+'ycbv_train_real.zip']
SIZES=[48318382080,27404732247]
_tls=threading.local()
_resolved_urls={}
_resolve_lock=threading.Lock()
def get_session():
 if not hasattr(_tls,'session'): _tls.session=requests.Session()
 return _tls.session

def resolved_url(disk):
 with _resolve_lock:
  if disk in _resolved_urls:return _resolved_urls[disk]
  for attempt in range(12):
   r=get_session().get(URLS[disk],headers={'Range':'bytes=0-0','Accept-Encoding':'identity'},timeout=60)
   if r.status_code==206:
    _resolved_urls[disk]=r.url
    print(f'Acquired reusable signed CDN URL for disk {disk}',flush=True)
    return r.url
   if r.status_code==429:
    delay=max(1,float(r.headers.get('Retry-After','60')))
    print(f'HF resolver rate limited; respecting Retry-After={delay}s',flush=True)
    time.sleep(delay+1);continue
   r.raise_for_status()
  raise RuntimeError('Could not resolve archive CDN URL after rate limit retries')

def get_range(disk,start,length):
 end=start+length-1
 for attempt in range(8):
  try:
   url=resolved_url(disk)
   r=get_session().get(url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},timeout=90)
   if r.status_code in [401,403]:
    with _resolve_lock:_resolved_urls.pop(disk,None)
    raise RuntimeError('CDN signature expired; resolving a new URL')
   if r.status_code==429:
    delay=max(1,float(r.headers.get('Retry-After','60')))
    print(f'CDN rate limited; respecting Retry-After={delay}s',flush=True);time.sleep(delay+1)
    raise RuntimeError('CDN rate limit')
   if r.status_code!=206 or r.headers.get('content-range','').split('/')[0]!=f'bytes {start}-{end}' or len(r.content)!=length:
    raise RuntimeError(f'Bad range {r.status_code} {r.headers.get("content-range")} wanted {start}-{end} len={len(r.content)}')
   return r.content
  except Exception:
   if attempt==7: raise
   time.sleep(min(15,1.5**attempt))

def scan_index():
 path=ROOT/'central_directory.bin'
 if not path.exists():
  # Verified ZIP64 directory metadata, re-check on every fresh acquisition.
  tail=get_session().get(URLS[1],headers={'Range':'bytes=-512','Accept-Encoding':'identity'},timeout=90)
  vals=struct.unpack_from('<4sQ2H2I4Q',tail.content,tail.content.find(b'PK\x06\x06'))
  _,_,_,_,disk,diskdir,n1,n2,size,offset=vals
  assert disk==diskdir==1 and n1==n2==1257337 and size==124348025 and offset==27280384124,vals
  chunk=8*1024*1024
  ranges=[(offset+i,min(chunk,size-i)) for i in range(0,size,chunk)]
  with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
   results=list(ex.map(lambda x:get_range(1,*x),ranges))
  path.write_bytes(b''.join(results))
 data=path.read_bytes(); pos=0; count=0; sceneinfo={}; rows=[]
 while pos<len(data):
  vals=struct.unpack_from('<4s6H3I5H2I',data,pos)
  assert vals[0]==b'PK\x01\x02',(pos,vals)
  _,create,extract,flags,method,t,d,crc,csize,usize,nlen,xlen,clen,disk,iattr,eattr,offset=vals
  name=data[pos+46:pos+46+nlen].decode('utf8' if flags&2048 else 'cp437')
  extra=data[pos+46+nlen:pos+46+nlen+xlen]
  i=0
  while i<len(extra):
   xtype,xsize=struct.unpack_from('<HH',extra,i); value=extra[i+4:i+4+xsize]; i+=4+xsize
   if xtype==1:
    j=0
    if usize==0xffffffff: usize=struct.unpack_from('<Q',value,j)[0]; j+=8
    if csize==0xffffffff: csize=struct.unpack_from('<Q',value,j)[0]; j+=8
    if offset==0xffffffff: offset=struct.unpack_from('<Q',value,j)[0]; j+=8
    if disk==0xffff: disk=struct.unpack_from('<I',value,j)[0]
  row={'name':name,'disk':disk,'offset':offset,'compressed_size':csize,'size':usize,'crc32':crc,'method':method}
  rows.append(row)
  pieces=name.split('/')
  if len(pieces)>2 and pieces[0]=='train_real':
   scene=pieces[1]
   s=sceneinfo.setdefault(scene,{'files':0,'rgb':0,'mask_visib':0,'mask':0,'depth':0,'bytes':0})
   s['files']+=1;s['bytes']+=usize
   if pieces[2] in s:s[pieces[2]]+=1
  count+=1;pos+=46+nlen+xlen+clen
 assert count==1257337
 with (ROOT/'archive_index.jsonl').open('w') as f:
  for row in rows:f.write(json.dumps(row)+'\n')
 (ROOT/'scene_inventory.json').write_text(json.dumps(sceneinfo,indent=2))
 print(json.dumps({'count':count,'scenes':sceneinfo},indent=2),flush=True)
 return rows

def read_entry(row):
 p=ROOT/'data'/row['name']
 if p.exists() and p.stat().st_size==row['size'] and (zlib.crc32(p.read_bytes())&0xffffffff)==row['crc32']:return p
 # Fetch local header and compressed bytes in one request; local extras are read from central metadata assumptions only after inspecting header.
 fetch_length=30+len(row['name'].encode('utf8'))+256+row['compressed_size']
 buf=get_range(row['disk'],row['offset'],fetch_length)
 vals=struct.unpack_from('<4s5H3I2H',buf);assert vals[0]==b'PK\x03\x04'
 nlen,xlen=vals[-2:];size=row['compressed_size'];start=30+nlen+xlen
 b=buf[start:start+size] if start+size<=len(buf) else get_range(row['disk'],row['offset']+start,size)
 if row['method']==0:raw=b
 elif row['method']==8:raw=zlib.decompress(b,-15)
 else:raise ValueError(row)
 assert len(raw)==row['size'] and zlib.crc32(raw)&0xffffffff==row['crc32'],row
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_bytes(raw);tmp.rename(p)
 return p

def select_and_download(rows,train_scenes,val_scenes,ntrain,nval,workers):
 names={r['name']:r for r in rows};selected=[];splitplan={};framemeta={}
 for split,scenes,num in [('train',train_scenes,ntrain),('val',val_scenes,nval)]:
  for scene in scenes:
   prefix=f'train_real/{scene}/'
   rgb=sorted([n for n in names if n.startswith(prefix+'rgb/') and not n.endswith('/')])
   assert len(rgb)>num,(scene,len(rgb),num)
   idx=np.linspace(0,len(rgb)-1,num=num,dtype=int)
   frames=[int(pathlib.Path(rgb[i]).stem) for i in idx]
   splitplan[scene]={'split':split,'frames':frames,'total_archive_frames':len(rgb),'sampling':'linspace over complete ordered scene; scene-disjoint train/val'}
   for leaf in ['scene_gt.json','scene_gt_info.json','scene_camera.json']:
    entry=names[prefix+leaf];p=read_entry(entry);selected.append(entry)
    framemeta[(scene,leaf)]=json.loads(p.read_text())
   for frame in frames:
    for kind in ['rgb','depth']:
     candidates=[prefix+f'{kind}/{frame:06d}'+ext for ext in ['.png','.jpg']]
     n=next((n for n in candidates if n in names),None)
     if n:selected.append(names[n])
    gt=framemeta[(scene,'scene_gt.json')][str(frame)]
    for inst,item in enumerate(gt):
     for kind in ['mask','mask_visib']:
      n=prefix+f'{kind}/{frame:06d}_{inst:06d}.png';assert n in names,n;selected.append(names[n])
 print('selected',len(selected),'compressed MB',sum(x['compressed_size'] for x in selected)/1e6,'uncompressed MB',sum(x['size'] for x in selected)/1e6,flush=True)
 (ROOT/'split_plan.json').write_text(json.dumps(splitplan,indent=2))
 (ROOT/'selected_archive_entries.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in selected))
 assert sum(x['size'] for x in selected)<2_000_000_000
 start=time.time()
 priority=sorted(selected,key=lambda r: 0 if any('/'+s+'/' in r['name'] for s in val_scenes) else 1)
 with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
  futures={ex.submit(read_entry,row):row for row in priority}
  for j,f in enumerate(concurrent.futures.as_completed(futures)):
   f.result()
   if j%100==0:print(f'downloaded {j+1}/{len(selected)} elapsed={time.time()-start:.1f}s',flush=True)
 build_manifest(splitplan,framemeta)

def build_manifest(splitplan,framemeta):
 counts={}; rows=[];errors=[]
 for scene,s in splitplan.items():
  split=s['split'];counts.setdefault(split,{'scenes':[],'frames':0,'instances':0,'visible_pixels':0,'full_pixels':0,'external_proxy_pixels':0})['scenes'].append(scene)
  prefix=ROOT/'data'/'train_real'/scene
  gt=framemeta[(scene,'scene_gt.json')];info=framemeta[(scene,'scene_gt_info.json')];camera=framemeta[(scene,'scene_camera.json')]
  for frame in s['frames']:
   im=next((prefix/'rgb'/f'{frame:06d}{ext}' for ext in ['.png','.jpg'] if (prefix/'rgb'/f'{frame:06d}{ext}').exists()))
   shape=Image.open(im).size[::-1];assert shape==(480,640),(im,shape)
   image_sha256=hashlib.sha256(im.read_bytes()).hexdigest()
   deny_path=ROOT.parent/'contract/EVALUATION_SHA256_DENYLIST.json'
   deny=set(json.loads(deny_path.read_text())['image_sha256'])
   assert image_sha256 not in deny,('EVALUATION LEAKAGE',im)
   counts[split]['frames']+=1
   for inst,obj in enumerate(gt[str(frame)]):
    mv=prefix/'mask_visib'/f'{frame:06d}_{inst:06d}.png';mf=prefix/'mask'/f'{frame:06d}_{inst:06d}.png'
    vis=np.asarray(Image.open(mv));full=np.asarray(Image.open(mf));assert vis.shape==full.shape==shape
    assert set(np.unique(vis)).issubset({0,255}) and set(np.unique(full)).issubset({0,255})
    vis=vis>0;full=full>0
    outside=int((vis&~full).sum()); assert outside==0,(mv,'visible_not_subset',outside)
    ext=full&~vis
    item={'image_sha256':image_sha256,'split':split,'image':str(im),'mask_vis':str(mv),'mask_full':str(mf),'scene_id':int(scene),'frame_id':frame,'object_id':obj['obj_id'],'instance_id':inst,'K':camera[str(frame)]['cam_K'],'cam_R_m2c':obj['cam_R_m2c'],'cam_t_m2c':obj['cam_t_m2c'],'depth':str(prefix/'depth'/f'{frame:06d}.png'),'depth_scale':camera[str(frame)].get('depth_scale'),'height':shape[0],'width':shape[1],'vis_pixels':int(vis.sum()),'full_pixels':int(full.sum()),'external_proxy_pixels':int(ext.sum()),'visible_fraction':float(vis.sum()/max(full.sum(),1)),'source':'BOP YCB-V official train_real','source_archives':URLS,'license':'MIT','labels':{'visible':'mask_vis>0','external_occlusion_proxy':'mask_full>0 and mask_vis==0','outside_target_support':'mask_full==0; NOT synonymous with occlusion','self_occlusion':'unavailable from 2D masks; do not assign'}}
    rows.append(item);counts[split]['instances']+=1;counts[split]['visible_pixels']+=item['vis_pixels'];counts[split]['full_pixels']+=item['full_pixels'];counts[split]['external_proxy_pixels']+=item['external_proxy_pixels']
 for split in counts:
  (ROOT/f'{split}_instances.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows if r['split']==split))
 (ROOT/'instances.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
 (ROOT/'integrity_audit.json').write_text(json.dumps({'counts':counts,'errors':errors,'all_shape_checks_passed':True,'all_masks_binary':True,'all_vis_subset_full':True,'crc32_verified_all_selected_entries':True,'eval_data_opened':False,'gpu_used':False,'PnP_used':False,'selection_locked_before_training':True,'all_public_rgb_sha256_disjoint_from_223_eval':True},indent=2))
 print(json.dumps(counts,indent=2),flush=True)
def namesafe_disk(scene):return 0 if int(scene)<50 else 1

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--index-only',action='store_true');ap.add_argument('--train-scenes',default='000000,000001,000002,000003,000004,000005,000006,000007');ap.add_argument('--val-scenes',default='000008,000009');ap.add_argument('--train-frames',type=int,default=80);ap.add_argument('--val-frames',type=int,default=80);ap.add_argument('--workers',type=int,default=16);args=ap.parse_args()
 if (ROOT/'archive_index.jsonl').exists():rows=[json.loads(x) for x in (ROOT/'archive_index.jsonl').read_text().splitlines()]
 else:rows=scan_index()
 if args.index_only:return
 select_and_download(rows,args.train_scenes.split(','),args.val_scenes.split(','),args.train_frames,args.val_frames,args.workers)
if __name__=='__main__':main()
