#!/usr/bin/env python3
"""Closed saved gallery check: no source array/model access."""
import hashlib,io,json,math,stat
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlparse,unquote
import numpy as np
from PIL import Image
R=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
S=R/'V233_ROOT_CROP_ADAPTATION'
G=S/'fixed8_source_dev_gallery_v2'
H=R/'V233_GALLERY_QA_B'
access=[]
def safe(p):
 p=Path(p);assert p.is_absolute() and '..' not in p.parts
 assert 'v112' not in str(p).lower() and '__pycache__' not in str(p).lower()
 assert p.is_relative_to(R)
 for q in [p]+list(p.parents): assert not q.is_symlink(),str(q)
 return p
def read(p,binding=None):
 p=safe(p);raw=p.read_bytes();sha=hashlib.sha256(raw).hexdigest()
 if binding:
  assert sha==binding['sha256']; assert len(raw)==binding['size_bytes']
 access.append({'path':str(p),'sha256':sha,'size_bytes':len(raw)})
 return raw
complete_raw=read(G/'COMPLETE.json');comp=json.loads(complete_raw)
assert comp['PASS'] is True and comp['actual_crop_rows']==7 and comp['empty_buckets']==1
assert all(comp[k]==0 for k in ('model_forward','new_fit','current_or_held_input'))
policy_raw=read(S/'FIXED8_DEV_CASE_POLICY_BEFORE_PREDICTION_READ.json')
assert hashlib.sha256(policy_raw).hexdigest()=='dcd313caf192764124e92a794aee1a7de2c5d1bdfa2eb5fec923315a7bc5c11f'
policy=json.loads(policy_raw)
white={}
for b in comp['outputs']:
 rel=Path(b['path']);assert not rel.is_absolute() and len(rel.parts)==1 and rel.suffix in ('.png','.html','.json')
 assert b['path'] not in white;white[b['path']]=b
payload={name:read(G/name,b) for name,b in white.items()}
ids=json.loads(payload['CASE_IDS_BEFORE_PIXEL_READ.json']);data=json.loads(payload['CASE_DATA.json'])
assert ids['policy_sha256']==data['policy_sha256']==comp['policy_sha256']
assert data['fit_complete_sha256']=='dd662d0a466e38972a386104f9a26f235f955b060c20da4a3b678283f70ed114'
assert data['crop_complete_sha256']=='45b294ea7e89ff1a9e7b1b4765a24c88b2b0b402a79aaa3e5e77b65f4ac80535'
expected_counts=[1273,15,51,5,333,7,0,57];expected_ids=[0,382,148,7,7221,7283,None,7220]
assert len(ids['selection'])==len(data['cases'])==len(policy['buckets'])==8
pnginfo=[]
for i,(selected,case,bucket) in enumerate(zip(ids['selection'],data['cases'],policy['buckets'])):
 assert selected['bucket_index']==case['bucket_index']==i
 assert selected['population']==case['population']==expected_counts[i]
 assert selected['sample_index']==case['sample_index']==expected_ids[i]
 for k,v in bucket.items():assert selected[k]==case[k]==v
 if case['sample_index'] is None:
  assert i==6 and case['population']==0 and set(case)==set(selected);continue
 d=case['diagnostic'];assert d['fit'] is False and d['internal_source_role']=='INTERNAL_SOURCE_DEV'
 assert d['sample_index']==case['sample_index'] and d['target']==case['target']
 assert d['family_code']==case['family'] and d['available'] is True
 for arm in ('control','treatment'):
  a=case[arm];assert int(a['sample_index'])==case['sample_index'] and a['row_id']==d['row_id']
  assert int(a['family'])==case['family'] and int(a['target'])==case['target'] and a['role']=='DEV' and a['available']=='True'
  logits=[float(a['logit0']),float(a['logit1'])];assert all(map(math.isfinite,logits))
  assert float(a['selected_family_logit'])==logits[case['family']]
  assert a['prediction']==('E' if logits[case['family']]>=0 else 'D')
 truth='DE'[case['target']];assert (case['treatment']['prediction']==truth)==case['treatment_correct']
 if case['family']==1:assert d['native_slot'] is None and d['native_mapping']=='UNKNOWN' and d['reference_xy_GT_diagnostic_only'] is None
 with Image.open(io.BytesIO(payload[case['raw_png']])) as im:
  assert im.mode=='RGB' and im.size==(96,96);raw=np.array(im)
 chw=np.ascontiguousarray(raw.transpose(2,0,1));sha=hashlib.sha256(chw.tobytes()).hexdigest();assert sha==d['pixel_sha256']
 with Image.open(io.BytesIO(payload[case['overlay_png']])) as im:
  assert im.mode=='RGB' and im.size==(288,288);overlay=np.array(im)
 cyan=(overlay[:,:,0]<30)&(overlay[:,:,1]>220)&(overlay[:,:,2]>220);ys,xs=np.nonzero(cyan)
 assert len(xs)>0;cx=(int(xs.min())+int(xs.max()))/2;cy=(int(ys.min())+int(ys.max()))/2
 assert abs(cx-143.5)<=.5 and abs(cy-143.5)<=.5
 pnginfo.append({'bucket':i,'sample_index':case['sample_index'],'raw_size':[96,96],'restored_CHW_SHA256':sha,'source_diagnostic_SHA_match':True,'overlay_size':[288,288],'cyan_center_raster':[cx,cy],'query_center_RAW96':[47.5,47.5],'raster_center_tolerance_overlay_px':.5})
with Image.open(io.BytesIO(payload['SOURCE_DEV_FIXED8.png'])) as im:assert im.size==(1600,1140);im.load()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[];self.articles=0;self.text=[]
 def handle_starttag(self,t,a):
  if t=='article':self.articles+=1
  for k,v in a:
   if k in ('href','src'):self.links.append(v)
 def handle_data(self,d):self.text.append(d)
p=Links();p.feed(payload['ROOT_READOUT_KO.html'].decode());assert p.articles==8
for link in p.links:
 u=urlparse(link);assert not u.scheme and not u.netloc
 rel=unquote(u.path)
 if rel:assert Path(rel).name==rel and rel in white
text=' '.join(p.text)
for phrase in ('YCB E 정답 0개','진단 전용','YCB native는 UNKNOWN','현재 실사 성공으로 해석할 수 없습니다'):assert phrase in text
result={'completed':True,'PASS':True,'complete':{'path':str(G/'COMPLETE.json'),'sha256':hashlib.sha256(complete_raw).hexdigest()},'bucket_counts':expected_counts,'fixed_sample_indices':expected_ids,'empty_YCB_E_correct_bucket_explicit':True,'raw_png_count':7,'overlay_png_count':7,'board_png_count':1,'PNG_CHW_roundtrip_checks':pnginfo,'html_articles':p.articles,'closed_relative_link_count':len(p.links),'all_declared_outputs_SHA_exact':True,'access':access,'limits':['Global minimum/population are checked against sealed policy/CASE_IDS and authorized fixed expectations; no full score CSV reranking in this gallery-only task.','PNG HWC decoded then transposed to contiguous CHW matches case diagnostic pixel SHA; original NPY/source RGB/valid bitmap not reopened. Prior crop QA supplied source-byte correspondence.','Cyan marker is checked as rasterized image centre equivalent to original RAW96 (47.5,47.5) within half overlay pixel; no coordinate correction or new semantic visibility label.','No actual tool image view or browser render; numerical saved-PNG/hash/HTML-link QA only.','No model/forward/fit/pretrained/prefix/GT/protected access; source_RGB_binding strings within CASE_DATA were never followed.','Initial gallery CHW-vs-HWC assertion failure is a Root-disclosed producer history, not independently rerun or inspected by this saved-output QA.']}
out=H/'REVIEW.json';assert not out.exists();out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'completed':True,'PASS':True,'review':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'raw_roundtrips':len(pnginfo),'relative_links':len(p.links)}))
