"""Bounded selected scene_gt JSON ranges only; no images/depth/masks/archive body."""
from pathlib import Path
import hashlib,json,struct,zlib
import requests
H=Path(__file__).resolve().parent;P=H.parent.parent
CACHE=P/'query_visibility_public_real_20261022_v63/ycb'
REV='5c2c4aa229800355648cd268040aa814f8dc94f0'
URL=f'https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/{REV}/ycbv_train_real.z01'
def bind(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def write(p,x):
 with p.open('x')as f:json.dump(x,f,indent=2);f.write('\n')
def main():
 candidates={}
 with(CACHE/'archive_index.jsonl').open()as f:
  for line in f:
   if '/scene_gt.json' in line:
    r=json.loads(line);scene=int(r['name'].split('/')[1])
    if 10<=scene<=14:candidates[scene]=r
 assert len(candidates)==5 and all(r['disk']==0 for r in candidates.values())
 write(H/'CONTROL_SELECTION_BEFORE_METADATA.json',dict(completed=True,rule='First five numerically ordered train_real scenes not cached0..9:10..14, scene_gt JSON only.',original_archive_index=bind(CACHE/'archive_index.jsonl'),official_pinned_URL=URL,selected=[candidates[i]for i in sorted(candidates)],compressed_payload_bytes=sum(r['compressed_size']for r in candidates.values()),new_network_body_budget=5_000_000,no_image_depth_mask_payload=True))
 session=requests.Session();used=0;requests_log=[];receipts=[]
 def get_range(start,length,path):
  nonlocal used
  assert used+length<=5_000_000
  response=session.get(URL+f'?download=true&v111_range_start={start}',headers={'Range':f'bytes={start}-{start+length-1}','Accept-Encoding':'identity','Cache-Control':'no-cache'},stream=True,timeout=45)
  log=dict(start=start,length=length,status=response.status_code,content_range=response.headers.get('Content-Range'),body_bytes=0)
  if response.status_code!=206 or response.headers.get('Content-Range','').split('/')[0]!=f'bytes {start}-{start+length-1}':
   response.close();requests_log.append(log);write(H/'FAILED_CONTROL_REQUEST.json',dict(request=log,bytes_already_read=used,no_full_body_fallback=True));raise RuntimeError(log)
  data=bytearray()
  for chunk in response.iter_content(65536):
   used+=len(chunk);data.extend(chunk)
   if len(data)>length or used>5_000_000:response.close();raise RuntimeError('Unexpected body over budget')
  response.close();assert len(data)==length;path.write_bytes(data);log['body_bytes']=len(data);log['file']=bind(path);requests_log.append(log);return bytes(data)
 for scene,r in sorted(candidates.items()):
  folder=H/f'scene_{scene:06d}_metadata';folder.mkdir()
  header=get_range(r['offset'],30,folder/'LOCAL_HEADER30.bin');values=struct.unpack('<4s5H3I2H',header);assert values[0]==b'PK\x03\x04';nlen,xlen=values[-2:]
  payload=get_range(r['offset']+30,nlen+xlen+r['compressed_size'],folder/'NAME_EXTRA_COMPRESSED.bin');assert payload[:nlen].decode()==r['name']
  compressed=payload[nlen+xlen:];raw=zlib.decompress(compressed,-15)if r['method']==8 else compressed
  assert len(raw)==r['size']and(zlib.crc32(raw)&0xffffffff)==r['crc32'];out=folder/'scene_gt.json';out.write_bytes(raw);gt=json.loads(raw)
  object_ids=sorted({o['obj_id']for values in gt.values()for o in values});counts={str(i):sum(o['obj_id']==i for values in gt.values()for o in values)for i in(2,3,7,8)}
  receipts.append(dict(scene_id=scene,metadata=bind(out),archive_entry=r,CRC_verified=True,frames=len(gt),all_object_ids=object_ids,target_instance_counts=counts))
  print(json.dumps(dict(scene_id=scene,frames=len(gt),object_ids=object_ids,target_counts=counts,control_body_bytes=used)),flush=True)
 write(H/'CONTROL_METADATA_COMPLETE.json',dict(completed=True,selected=receipts,request_log=requests_log,actual_new_network_body_bytes=used,budget=5_000_000,whole_archive_SHA_verified=False,no_RGB_depth_mask_or_models=True))
 print(json.dumps(dict(completed=True,receipt=bind(H/'CONTROL_METADATA_COMPLETE.json'),bytes=used)),flush=True)
if __name__=='__main__':main()
