"""V151 explicitly authorized source-only tiny intake. No models or native labels."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,collections,math,urllib.request,struct,sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont,__version__ as PIL_VERSION,features
V=Path(__file__).resolve().parents[1];O=V/'factual_sample_v1';M=V/'metadata';D=V/'provided_metadata'
POLICY='3b23bd1ef0c48819b8c84f07a8f0a53c372620457ca972878e47cfad21ebdfba'
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(p),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def h(s):return hashlib.sha256(s.encode()).hexdigest()
def root_inputs():
 assert sha(V/'ROOT_PUBLIC_INTAKE_POLICY_BEFORE_REMOTE_SAMPLE_BODIES.json')==POLICY
 policy=read(V/'ROOT_PUBLIC_INTAKE_POLICY_BEFORE_REMOTE_SAMPLE_BODIES.json');clar=read(V/'ROOT_MULTI_INSTANCE_SELECTION_CLARIFICATION_BEFORE_SAMPLES.json')
 pin=read(M/'PINNED_REMOTE_SOURCE.json');info=read(M/'HF_DATASET_INFO.json');receipt=read(M/'PROVIDED_METADATA_DOWNLOAD_RECEIPT.json')
 assert pin['completed'] and info['sha']==pin['commit']==receipt['commit'] and not pin['gated'] and not pin['private']
 assert sha(M/'HF_DATASET_INFO.json')==pin['metadata_sha256'] and receipt['completed'] and receipt['PASS']
 assert receipt['image_depth_mask_mesh_bodies_downloaded']==0
 for x in receipt['files']:
  assert sha(x['local_path'])==x['sha256'] and Path(x['local_path']).stat().st_size==x['bytes']
 return policy,clar,pin,info,receipt

def select():
 policy,clar,pin,info,receipt=root_inputs();assert not O.exists();O.mkdir()
 files={x['rfilename']for x in info['siblings']};scenes=[];bindings=[]
 for path in sorted(files):
  if not path.endswith('/scene_gt.json'):continue
  split,scene,_=path.split('/');gt=read(D/path);ip=D/split/scene/'scene_gt_info.json';cp=D/split/scene/'scene_camera.json'
  gi=read(ip)if ip.exists()else None;camera=read(cp);fr=[];nobj=0;bins={x:[]for x in['LOW_LT_0.5','MID_0.5_TO_LT_0.9','HIGH_GE_0.9']}
  bindings.extend([bind(D/path),bind(cp)]);
  if gi is not None:bindings.append(bind(ip))
  for key in sorted(gt,key=int):
   idx=[i for i,x in enumerate(gt[key])if int(x['obj_id'])==1]
   if not idx:continue
   assert key in camera
   nobj+=len(idx);fractions=[];qual=[]
   for i in idx:
    if gi is None:continue
    assert key in gi and len(gi[key])==len(gt[key]);f=float(gi[key][i]['visib_fract']);assert math.isfinite(f)and 0<=f<=1
    fractions.append({'gt_instance_index':i,'visib_fract':f})
    b='LOW_LT_0.5'if f<.5 else'MID_0.5_TO_LT_0.9'if f<.9 else'HIGH_GE_0.9'
    if b not in qual:qual.append(b)
   r={'split':split,'scene':scene,'frame_key':key,'frame_filename_stem':f'{int(key):06d}','frame_hash_input':f'{split}|{scene}|{key}',
      'frame_hash':h(f'{split}|{scene}|{key}'),'obj1_gt_instance_indices':idx,'instance_visibility_fractions':fractions,'qualifying_bins':qual}
   fr.append(r)
   for b in qual:bins[b].append(r)
  if fr:
   scenes.append({'split':split,'scene':scene,'scene_hash_input':f'MR6D_V151_SOURCE_SAMPLE_V1|{split}|{scene}',
      'scene_hash':h(f'MR6D_V151_SOURCE_SAMPLE_V1|{split}|{scene}'),'obj1_frames':len(fr),'obj1_instances':nobj,
      'GTinfo_present':gi is not None,'bin_eligible_frame_counts':{b:len(v)for b,v in bins.items()},'frames':fr})
 groups={s:sorted([x for x in scenes if x['split']==s],key=lambda x:(x['scene_hash'],x['scene']))for s in sorted(set(x['split']for x in scenes))}
 selected=[]
 while len(selected)<policy['max_selected_scenes'] and any(groups.values()):
  for split in sorted(groups):
   if groups[split]and len(selected)<policy['max_selected_scenes']:selected.append(groups[split].pop(0))
 chosen=[];missing=[]
 for scene in selected:
  dedup={}
  if scene['GTinfo_present']:
   for b in['LOW_LT_0.5','MID_0.5_TO_LT_0.9','HIGH_GE_0.9']:
    candidates=[r for r in scene['frames']if b in r['qualifying_bins']]
    if not candidates:missing.append({'split':scene['split'],'scene':scene['scene'],'bin':b,'reason':'NO_OBJ1_INSTANCE_QUALIFIES_METADATA_BIN'});continue
    r=min(candidates,key=lambda x:(x['frame_hash'],int(x['frame_key'])))
    if r['frame_key']not in dedup:dedup[r['frame_key']]=dict(r,selection_reasons=[])
    dedup[r['frame_key']]['selection_reasons'].append({'visibility_bin':b,'selection':'MIN_SHA256_FRAME_AMONG_ANY_OBJ1_QUALIFYING'})
  else:
   for r in sorted(scene['frames'],key=lambda x:(x['frame_hash'],int(x['frame_key'])))[:3]:dedup[r['frame_key']]=dict(r,selection_reasons=[{'visibility_bin':None,'selection':'MIN_HASH_FALLBACK_GTINFO_ABSENT'}])
  chosen.extend(sorted(dedup.values(),key=lambda x:(x['frame_hash'],int(x['frame_key']))))
 assert len(selected)<=4 and len(chosen)<=12 and len({(x['split'],x['scene'],x['frame_key'])for x in chosen})==len(chosen)
 payload=['models/obj_000001.ply'];
 for r in chosen:
  folder=f"{r['split']}/{r['scene']}";stem=r['frame_filename_stem'];rgb=[p for p in files if p.startswith(folder+'/rgb/')and Path(p).stem==stem]
  assert len(rgb)==1;entries={'RGB':rgb[0],'depth':folder+'/depth/'+stem+'.png','full_masks':[],'visible_masks':[]}
  payload.extend([rgb[0],entries['depth']])
  for i in r['obj1_gt_instance_indices']:
   for mod,k in [('mask','full_masks'),('mask_visib','visible_masks')]:
    p=f'{folder}/{mod}/{stem}_{i:06d}.png';assert p in files;entries[k].append({'gt_instance_index':i,'path':p});payload.append(p)
  r['source_files']=entries
 assert all(p in files for p in payload)and len(payload)==len(set(payload))
 artifact={'completed':True,'sealed_before_new_RGB_depth_mask_mesh_GET':True,'UTC':now(),'policy':bind(V/'ROOT_PUBLIC_INTAKE_POLICY_BEFORE_REMOTE_SAMPLE_BODIES.json'),
 'clarification':bind(V/'ROOT_MULTI_INSTANCE_SELECTION_CLARIFICATION_BEFORE_SAMPLES.json'),'code':bind(Path(__file__).resolve()),'commit':pin['commit'],
 'metadata_receipt':bind(M/'PROVIDED_METADATA_DOWNLOAD_RECEIPT.json'),'HF_info':bind(M/'HF_DATASET_INFO.json'),'metadata_bindings':bindings,
 'all_obj1_scenes':[dict((k,v)for k,v in s.items()if k!='frames')for s in scenes],
 'selected_scenes':[dict((k,v)for k,v in s.items()if k!='frames')for s in selected],
 'selected_frames':chosen,'missing_bins':missing,'payload_paths':payload,
 'frame_hash_key_contract':'Use unmodified decimal JSON frame key, e.g.val|000000|0; filename is separately zero padded6. No image-based replacement.',
 'metadata_global_enum':{'GT_scene_count':sum(p.endswith('/scene_gt.json')for p in files),'obj1_scene_count':len(scenes)},
 'native_D_E_S_labels_created':0,'new_model_calls':0}
 write(O/'FIXED_SOURCE_SAMPLE_SELECTION_BEFORE_PIXELS.json',artifact)
 return artifact

def metadata_plan(selection):
 policy,_,pin,_,receipt=root_inputs();base='https://huggingface.co/datasets/anas-gouda/mr6d/resolve/'+pin['commit']+'/'
 out=[];sumbytes=0
 for rel in selection['payload_paths']:
  url=base+rel;request=urllib.request.Request(url,method='HEAD',headers={'User-Agent':'MR6D-V151-authorized-tiny-source-intake/1.0'})
  with urllib.request.urlopen(request,timeout=30)as r:
   assert r.status==200;size=int(r.headers['Content-Length']);assert 0<size<=10000000
   row={'relative_path':rel,'pinned_url':url,'HEAD_status':r.status,'HEAD_UTC':now(),'expected_bytes':size,
        'ETag':r.headers.get('ETag'),'Content_Type':r.headers.get('Content-Type')}
  sumbytes+=size;out.append(row)
 existing_remote=Path(M/'HF_DATASET_INFO.json').stat().st_size+receipt['total_body_bytes']
 assert existing_remote+sumbytes<=policy['max_body_download_bytes']
 plan={'completed':True,'sealed_before_new_payload_GET':True,'selection':bind(O/'FIXED_SOURCE_SAMPLE_SELECTION_BEFORE_PIXELS.json'),
 'payloads':out,'payload_bytes':sumbytes,'existing_HF_and_provided_metadata_bytes':existing_remote,
 'total_remote_metadata_plus_payload_bytes':existing_remote+sumbytes,'cap':policy['max_body_download_bytes']}
 write(O/'PINNED_TINY_DOWNLOAD_PLAN_BEFORE_GET.json',plan);return plan

def download(plan):
 out=[]
 for row in plan['payloads']:
  p=O/'original'/row['relative_path'];p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists()
  rec=dict(row,requested_UTC=now());tmp=p.with_suffix(p.suffix+'.partial');assert not tmp.exists()
  req=urllib.request.Request(row['pinned_url'],headers={'User-Agent':'MR6D-V151-authorized-tiny-source-intake/1.0'})
  try:
   with urllib.request.urlopen(req,timeout=40)as r:
    assert r.status==200
    with tmp.open('xb')as f:
     total=0
     while True:
      b=r.read(1024*256)
      if not b:break
      total+=len(b);assert total<=row['expected_bytes'];f.write(b)
    assert total==row['expected_bytes'];rec.update(GET_status=r.status,GET_ETag=r.headers.get('ETag'),Content_Type=r.headers.get('Content-Type'))
   assert rec['GET_ETag']==row['ETag'];tmp.rename(p);rec.update(completed=True,bytes=p.stat().st_size,local_path=str(p),sha256=sha(p))
   etag=(row.get('ETag')or'').strip('"')
   if len(etag)==40 and all(c in'0123456789abcdef'for c in etag):
    rec['git_blob_SHA1_verified']=hashlib.sha1(b'blob '+str(p.stat().st_size).encode()+b'\0'+p.read_bytes()).hexdigest()==etag;assert rec['git_blob_SHA1_verified']
   elif len(etag)==64 and all(c in'0123456789abcdef'for c in etag):
    rec['SHA256_ETag_verified']=rec['sha256']==etag;assert rec['SHA256_ETag_verified']
   write(O/(f'DOWNLOAD_RECEIPT_{len(out):03d}.json'),rec);out.append(rec)
   print('download',len(out),len(plan['payloads']),row['relative_path'],rec['bytes'],flush=True)
  except Exception as e:
   write(O/'DOWNLOAD_FAILURE_PRESERVED.json',dict(completed=False,failed=row,exception=type(e).__name__+': '+str(e),partial_preserved=tmp.exists(),successful_files=len(out),automatic_retry=0));raise
 final={'completed':True,'PASS':True,'UTC':now(),'plan':bind(O/'PINNED_TINY_DOWNLOAD_PLAN_BEFORE_GET.json'),'files':out,
 'downloaded_payload_bytes':sum(x['bytes']for x in out),'all_metadata_plus_payload_bytes':plan['existing_HF_and_provided_metadata_bytes']+sum(x['bytes']for x in out),
 'automatic_retries':0,'full_archive_download':False,'new_model_calls':0}
 write(O/'TINY_SOURCE_DOWNLOAD_COMPLETE.json',final);return final

def ply_vertices(p):
 data=p.read_bytes();end=data.index(b'end_header')+len(b'end_header');end=data.index(b'\n',end)+1;header=data[:end].decode('ascii');lines=header.splitlines();nv=None;props=[];invertex=False;fmt=None
 for line in lines:
  t=line.split()
  if not t:continue
  if t[0]=='format':fmt=t[1]
  if t[0]=='element':invertex=t[1]=='vertex';nv=int(t[2])if invertex else nv
  if t[0]=='property'and invertex:assert t[1]!='list';props.append((t[1],t[2]))
 assert nv and all(c in[n for _,n in props]for c in['x','y','z'])
 if fmt=='ascii':
  vals=np.array([[float(v)for v in x.split()]for x in data[end:].decode('ascii').splitlines()[:nv]],dtype=np.float64)
 else:
  codes={'char':'b','uchar':'B','int8':'b','uint8':'B','short':'h','ushort':'H','int16':'h','uint16':'H','int':'i','uint':'I','int32':'i','uint32':'I','float':'f','float32':'f','double':'d','float64':'d'}
  st=struct.Struct(('<'if fmt=='binary_little_endian'else'>')+''.join(codes[t]for t,_ in props));vals=np.array([st.unpack_from(data,end+i*st.size)for i in range(nv)],dtype=np.float64)
 a=vals[:,[[n for _,n in props].index(c)for c in['x','y','z']]];assert np.isfinite(a).all();return a,header

def validate_and_render(selection,receipt):
 records=[];mask_fail=0;se3_fail=0;pair_overlap=[];mesh=O/'original/models/obj_000001.ply';v,header=ply_vertices(mesh);mi=read(D/'models/models_info.json')['1'];mn=v.min(0);mx=v.max(0);assert np.allclose(mn,[mi['min_x'],mi['min_y'],mi['min_z']],rtol=0,atol=1e-6);assert np.allclose(mx-mn,[mi['size_x'],mi['size_y'],mi['size_z']],rtol=0,atol=1e-6)
 for j,r in enumerate(selection['selected_frames']):
  prefix=D/r['split']/r['scene'];gt=read(prefix/'scene_gt.json')[r['frame_key']];gi=read(prefix/'scene_gt_info.json')[r['frame_key']];ca=read(prefix/'scene_camera.json')[r['frame_key']]
  rgbp=O/'original'/r['source_files']['RGB'];depp=O/'original'/r['source_files']['depth'];rgb=np.array(Image.open(rgbp).convert('RGB'));dep=np.array(Image.open(depp));assert dep.shape==rgb.shape[:2];K=np.asarray(ca['cam_K'],dtype=np.float64).reshape(3,3);assert np.isfinite(K).all()and K[0,0]>0 and K[1,1]>0;assert float(ca['depth_scale'])>0
  instances=[];vm=[];fm=[]
  for i in r['obj1_gt_instance_indices']:
   assert gt[i]['obj_id']==1;R=np.asarray(gt[i]['cam_R_m2c'],dtype=np.float64).reshape(3,3);t=np.asarray(gt[i]['cam_t_m2c'],dtype=np.float64).reshape(3);finite=bool(np.isfinite(R).all()and np.isfinite(t).all());ortho=float(np.max(np.abs(R.T@R-np.eye(3))));det=float(np.linalg.det(R));se3=finite and ortho<=1e-3 and abs(det-1)<=1e-3;se3_fail+=not se3
   fullp=O/'original'/next(x['path']for x in r['source_files']['full_masks']if x['gt_instance_index']==i);visp=O/'original'/next(x['path']for x in r['source_files']['visible_masks']if x['gt_instance_index']==i)
   fullraw=np.array(Image.open(fullp));visraw=np.array(Image.open(visp));assert fullraw.shape==visraw.shape==rgb.shape[:2];full=fullraw!=0;vis=visraw!=0;outside=int((vis&~full).sum());mask_fail+=outside>0;vm.append((i,vis));fm.append((i,full))
   instances.append({'gt_instance_index':i,'object_id':1,'provided_T':{'finite':finite,'R_orthogonality_maxabs':ortho,'R_determinant':det,'fixed_1e_minus3_SE3_check':se3,'cam_t_m2c_unchanged':t.tolist()},
    'mask_full_unique_values':np.unique(fullraw).tolist(),'mask_visible_unique_values':np.unique(visraw).tolist(),'full_pixels':int(full.sum()),'visible_pixels':int(vis.sum()),'visible_outside_full_pixels':outside,
    'visible_subset_of_full':outside==0,'provided_GT_info_unchanged':gi[i],'full_mask':bind(fullp),'visible_mask':bind(visp),'native_visibility':'UNASSIGNED'})
  overlaps=[]
  for a in range(len(vm)):
   for b in range(a):overlaps.append({'instance_indices':[vm[b][0],vm[a][0]],'full_mask_overlap_pixels':int((fm[b][1]&fm[a][1]).sum()),'visible_mask_overlap_pixels':int((vm[b][1]&vm[a][1]).sum())})
  pair_overlap.extend(overlaps)
  records.append(dict(selection_index=j,split=r['split'],scene=r['scene'],frame_key=r['frame_key'],selection_reasons=r['selection_reasons'],
   RGB=bind(rgbp),decoded_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),raw_RGB_HW=list(rgb.shape[:2]),original_image_file_and_pixels_not_modified=True,
   depth=bind(depp),depth_dtype=str(dep.dtype),depth_shape=list(dep.shape),depth_nonzero=int((dep!=0).sum()),depth_scale=ca['depth_scale'],
   depth_min_nonzero=int(dep[dep!=0].min())if np.any(dep)else None,depth_max=int(dep.max()),K_unchanged=K.tolist(),instances=instances,instance_pair_mask_overlaps=overlaps))
 metrics={'completed':True,'PASS_read_preservation_and_schema':True,'mesh':dict(file=bind(mesh),format_header=header,vertex_count=len(v),bounds_min=mn.tolist(),bounds_max=mx.tolist(),extent=(mx-mn).tolist(),models_info_match_1e_minus6=True,
  units_status='BOP metadata/convention millimeters; no surface/up/native binding certified by bbox alone.'),
  'source_frames':len(records),'retained_pallet_instance_records':sum(len(x['instances'])for x in records),'numeric_observations':{'SE3_checks_failed_fixed1e_minus3':se3_fail,'instance_masks_with_visible_outside_full':mask_fail,'instance_pair_overlap_records':len(pair_overlap)},
  'preservation':'Every selected original source image and every obj1 instance retained, including numerical or mask inconsistencies. No pose/mask/value repair, point snapping, native labels or model outputs.',
  'decoder':{'Python':sys.version,'Pillow':PIL_VERSION,'NumPy':np.__version__,'libjpeg':features.version('jpg'),'libjpeg_turbo':features.version('libjpeg_turbo')},'frames':records}
 write(O/'SOURCE_GEOMETRY_AND_INSTANCE_MASK_VALIDATION.json',metrics)
 # Full frame resized only for a raw diagnostic montage. No pixels annotated or cropped.
 fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';font=ImageFont.truetype(fontpath,19);small=ImageFont.truetype(fontpath,15);boards=[]
 for start in range(0,len(records),4):
  subset=records[start:start+4];sheet=Image.new('RGB',(1320,930),'#eaf0f6');draw=ImageDraw.Draw(sheet)
  draw.text((18,10),'MR6D public factual RGB | supplied visibility is whole-instance, native D/E unassigned',font=font,fill='#192c40')
  for pos,x in enumerate(subset):
   xx=16+(pos%2)*654;yy=52+(pos//2)*432;im=Image.open(x['RGB']['path']).convert('RGB');im=im.resize((640,360),resample=Image.Resampling.NEAREST);sheet.paste(im,(xx,yy+60))
   draw.text((xx,yy),f"#{x['selection_index']:02d} {x['split']}/{x['scene']} frame{x['frame_key']}",font=font,fill='#142d47')
   st=' | '.join(f"obj1 index{i['gt_instance_index']}: {i['provided_GT_info_unchanged']['visib_fract']:.3f}"for i in x['instances']);draw.text((xx,yy+27),st,font=small,fill='#344f69')
  bp=O/(f'RAW_FACTUAL_RGB_CONTACT_{start//4+1:02d}.png');sheet.save(bp);boards.append(bind(bp))
 final={'completed':True,'PASS':True,'selection':bind(O/'FIXED_SOURCE_SAMPLE_SELECTION_BEFORE_PIXELS.json'),'download_receipt':bind(O/'TINY_SOURCE_DOWNLOAD_COMPLETE.json'),'validation':bind(O/'SOURCE_GEOMETRY_AND_INSTANCE_MASK_VALIDATION.json'),
  'contact_boards':boards,'original_frames':len(records),'all_obj1_instances_retained':sum(len(x['instances'])for x in records),
  'actual_pixel_view_completed':False,'actual_pixel_view_receipt_pending':True,'native_D_E_S_created':0,'new_model_fit_PnP_private_access':0}
 write(O/'FACTUAL_SAMPLE_INTAKE_COMPLETE.json',final)
 print(json.dumps({'complete':bind(O/'FACTUAL_SAMPLE_INTAKE_COMPLETE.json'),'boards':boards,'numerical_observations':metrics['numeric_observations']}),flush=True)
if __name__=='__main__':
 s=select();p=metadata_plan(s);r=download(p);validate_and_render(s,r)
