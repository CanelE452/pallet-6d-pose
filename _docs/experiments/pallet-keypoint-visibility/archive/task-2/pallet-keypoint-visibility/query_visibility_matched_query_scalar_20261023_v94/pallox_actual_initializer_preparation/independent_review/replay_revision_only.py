"""Pure generated V2 revision check; no torch/imported source module or pixels."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
PREP=HERE.parent
BASE=HERE.parents[2]
ROI=BASE.parent/'external_ray_proxy_head_20261022_v60/head_and_roi.py'
BACK=BASE/'query_visibility_public_real_20261022_v63/model/deployment_assets/frozen_yolo_features.py'
CODE=PREP/'infer_fixed_pallox_queries.py'
OLD=PREP/'provenance/PRE_BOUNDARY_AND_AFFINE_GUARD_V1/infer_fixed_pallox_queries.py'

def bind(p):
 p=Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)

def extracted(p,names,ns):
 nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)and n.name in names]
 assert len(nodes)==len(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
 return ns

def main():
 new=extracted(CODE,['native_function','selected_queries','serial'],{'np':np,'ast':ast,'ROI':ROI})
 old=extracted(OLD,['native_function','selected_queries'],{'np':np,'ast':ast,'ROI':ROI})
 rng=np.random.default_rng(20261023)
 tests=[]
 for h,w in [(80,100),(480,640),(1080,1920),(32,32)]:
  for xy in [(-.5,10),(-.25,10),(0,10),(w-.51,10),(w-.5,10),(w-.25,10),(w,10),(10,-.25),(10,h-.25),(10,h)]:
   q=np.array([[10.+i,20.+i]for i in range(9)],np.float64);q[0]=xy
   tests.append((q,h,w))
  for _ in range(50):tests.append((rng.uniform(-100,max(h,w)+100,(9,2)),h,w))
 for q,h,w in tests:
  z=dict(candidates=[dict(score=.8,keypoints_xy=q.copy()),dict(score=.8,keypoints_xy=q.copy()+30)],selected_index=0)
  a=new['selected_queries'](z,[h,w]);b=old['selected_queries'](z,[h,w])
  good=np.isfinite(q[:8]).all(-1)&~(q[:8]==-1).all(-1)
  L=float(np.clip(.25*np.max(np.ptp(q[:8][good],axis=0)),16,max(h,w)))if good.sum()>=2 else 16.
  for k,r in enumerate(a):
   x,y=q[k]
   assert r['query_xy']==q[k].tolist()==b[k]['query_xy'] and r['roi_side']==L==b[k]['roi_side']
   assert r['original_query_continuous_valid']==bool(0<=x<w and 0<=y<h)
   assert r['query_nearest_pixel_valid']==bool(0<=np.floor(x+.5)<w and 0<=np.floor(y+.5)<h)
   assert r['query_rgb96_centre_domain_valid']==r['query_nearest_pixel_valid']
  alt=q.copy();alt[8]=[1e8,-1e8]
  assert new['selected_queries'](dict(candidates=[dict(score=.8,keypoints_xy=alt)],selected_index=0),[h,w])==a
 empty=new['selected_queries'](dict(candidates=[],selected_index=None),[80,100])
 assert len(empty)==8 and all(r['query_xy']is None and r['roi_side']is None and not any(r[k]for k in ['original_query_continuous_valid','query_nearest_pixel_valid','query_rgb96_centre_domain_valid'])for r in empty)
 affine=extracted(BACK,['canvas_affine'],{'np':np})['canvas_affine']
 affinecases=0
 for ch,cw,ih,iw in [(680,840,544,640),(1280,2120,416,640),(840,840,640,640),(1000,700,640,448)]:
  g,o=affine([ch,cw],[ih,iw]);expectedg=min(640/ch,640/cw)
  expected=np.array([round((iw-round(cw*expectedg))/2-.1),round((ih-round(ch*expectedg))/2-.1)],np.float64)
  assert g==expectedg and np.array_equal(o,expected);affinecases+=1
 p=json.loads((PREP/'PROTOCOL_BEFORE_SOURCE_FORWARD.json').read_text())
 proof=json.loads((PREP/'GENERATED_QUERY_SERIALIZATION_TESTS.json').read_text())
 assert p['code']['sha256']==proof['code']['sha256']==bind(CODE)['sha256'] and proof['PASS']is True
 assert bind(PREP/'GENERATED_QUERY_SERIALIZATION_TESTS.json')['sha256']==p['generated']['sha256']
 assert bind(ROI)['sha256']==p['native_grid']['sha256']
 assert bind(BACK)['sha256']==p['assets']['backbone_code']['sha256']
 payload=dict(completed=True,PASS=True,code=bind(CODE),old_code=bind(OLD),protocol=bind(PREP/'PROTOCOL_BEFORE_SOURCE_FORWARD.json'),producer_generated=bind(PREP/'GENERATED_QUERY_SERIALIZATION_TESTS.json'),case_count=len(tests),native_slots_checked=8*len(tests),boundary_and_random_hw_count=4,continuous_nearest_and_RGBcentre_exact=True,old_and_new_q_L_exact=True,centroid8_excluded=True,first_argmax_ties=True,no_detection8_preserved=True,independent_affine_cases=affinecases,no_torch_source_module_model_pixel_feature_weight_array_current=True)
 out=HERE/'GENERATED_INDEPENDENT_QUERY_REVISION_REPLAY_V2.json';assert not out.exists();out.write_text(json.dumps(payload,indent=2)+'\n')
 print(json.dumps(bind(out)))

if __name__=='__main__':main()
