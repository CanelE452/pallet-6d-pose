"""Generated-only geometry, crop and entry checks; no actual source render."""
from pathlib import Path
import importlib.util,json,math,tempfile
import numpy as np
import matched_render_math as m
import render_matched_triplet as r
H=Path(__file__).resolve().parent

def main():
 proof=m.generated();crop,ash=r.crop_operator()
 # Independent scalar bilinear edge fixture agrees with pinned crop AST.
 rgb=np.arange(11*13*3,dtype=np.uint16).reshape(11,13,3)%256;rgb=rgb.astype(np.uint8);q=np.array([.125,10.25]);L=9.
 a,v=crop(rgb,q,L,96)
 for y,x in[(0,0),(47,47),(48,48),(95,95),(13,83),(83,13)]:
  xx=q[0]+((x+.5)/96-.5)*L;yy=q[1]+((y+.5)/96-.5)*L;good=(-.5<=xx<12.5 and-.5<=yy<10.5);z=np.zeros(3);ix=math.floor(xx);iy=math.floor(yy)
  for dx in(0,1):
   for dy in(0,1):
    px=ix+dx;py=iy+dy;w=(xx-ix if dx else 1-xx+ix)*(yy-iy if dy else 1-yy+iy)
    if 0<=px<13 and 0<=py<11:z+=rgb[py,px]*w
  expected=np.floor(np.clip(z,0,255)+.5).astype(np.uint8)if good else np.zeros(3,np.uint8)
  assert np.array_equal(a[:,y,x],expected)and bool(v[0,y,x])==good
 # Ownership ties become explicit U; immutable target buffers survive overlay.
 d=np.array([[3.,4.],[np.inf,2.]]);second=np.full_like(d,np.inf);o=np.array([[0,1],[-1,2]],np.int16);saved=(d.copy(),second.copy(),o.copy())
 f=np.array([[1.,4.],[2.,6.]])
 aa=d.copy();bb=second.copy();oo=o.copy();m.update_nearest_two(aa,bb,oo,f,308);ids,amb=m.final_ownership(aa,bb,oo)
 assert ids.tolist()==[[308,-2],[308,2]]and amb[0,1]
 assert all(np.array_equal(x,y)for x,y in zip((d,second,o),saved))
 try:r.authorize({'not':'a filename'})
 except(TypeError,AssertionError):pass
 # Root guard rejects before source geometry/pixels for unapproved stub entry.
 with tempfile.TemporaryDirectory()as td:
  p=Path(td)/'entry.json';p.write_text(json.dumps(dict(root_approved_actual_rgb_triplet=False)))
  try:r.authorize(p)
  except AssertionError:pass
  else:raise AssertionError('Unapproved entry accepted')
 proof.update(completed=True,code=r.bind(H/'render_matched_triplet.py'),math=r.bind(H/'matched_render_math.py'),test=r.bind(__file__),pinned_crop_code=r.bind(r.CROP_CODE),crop_AST_sha256=ash,scalar_crop_points=6,ownership_differentpart_tie_U=True,immutable_target_layers=True,entry_first_guard=True,actual_source_pixels_geometryarrays_render_model_fit_current_PnP=0)
 r.write(H/'GENERATED_RENDER_TESTS.json',proof);print(json.dumps(proof))
if __name__=='__main__':main()
