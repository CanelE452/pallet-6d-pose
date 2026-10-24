"""Bounded pure-AST query serializer review. Never imports/runs source model."""
from pathlib import Path
import ast
import hashlib
import json
import numpy as np

HERE=Path(__file__).resolve().parent
CODE=HERE.parent/'infer_fixed_pallox_queries.py'
ROI=HERE.parents[3]/'external_ray_proxy_head_20261022_v60/head_and_roi.py'

def bind(p):
 p=Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)

def main():
 ns={'np':np,'ast':ast,'ROI':ROI}
 nodes=[n for n in ast.parse(CODE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ['native_function','selected_queries','serial']]
 assert len(nodes)==3
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(CODE),'exec'),ns)
 choose=ns['selected_queries']
 q=np.array([[10.+i,20.+i]for i in range(9)],np.float64)
 def z(pts, scores=(.8,.8),selected=0):
  return dict(candidates=[dict(score=s,keypoints_xy=pts.copy()+i*20)for i,s in enumerate(scores)],selected_index=selected)
 a=choose(z(q),[80,100]);assert len(a)==8 and a[0]['query_xy']==q[0].tolist()
 q2=q.copy();q2[8]=1e8;assert choose(z(q2,scores=(.8,)),[80,100])==choose(z(q,scores=(.8,)),[80,100])
 empty=choose(dict(candidates=[],selected_index=None),[80,100]);assert len(empty)==8 and all(r['query_xy']is None for r in empty)
 boundary=[]
 for x,y in [(-.5,10),(-.25,10),(0,10),(99.49,10),(99.5,10),(99.75,10),(100,10),(10,-.25),(10,79.75),(10,80)]:
  p=q.copy();p[0]=[x,y];out=choose(z(p,scores=(.8,)),[80,100])[0]
  expected=bool(0<=x<100 and 0<=y<80)
  nearest=bool(0<=np.floor(x+.5)<100 and 0<=np.floor(y+.5)<80)
  boundary.append(dict(q=[x,y],serialized=out,continuous_expected=expected,nearest_expected=nearest,disagrees_with_continuous=out['original_query_continuous_valid']!=expected))
 payload=dict(completed=True,code=bind(CODE),roi=bind(ROI),pure_AST_only=True,
  firstargmax_tie=True,centroid8_excluded=True,no_detection8_retained=True,
  boundary_cases=boundary,continuous_field_mismatches=sum(x['disagrees_with_continuous']for x in boundary),
  no_actual_RGB_feature_weight_array_model_forward_current=True)
 p=HERE/'GENERATED_INDEPENDENT_QUERY_BOUNDARY_REPLAY_V1.json';assert not p.exists();p.write_text(json.dumps(payload,indent=2)+'\n')
 print(json.dumps(dict(output=bind(p),mismatches=payload['continuous_field_mismatches'])))

if __name__=='__main__':main()
