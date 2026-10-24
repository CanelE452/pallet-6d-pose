"""Replay the six public supplied poses with a second CPU rasterizer."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from mesh_depth_render import render_depth
ROOT=Path(__file__).resolve().parents[1];PUBLIC=ROOT/'public_pallet';DATA=PUBLIC/'data'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 start=time.monotonic();out=ROOT/'source_audit/INDEPENDENT_FIXED_SIX_GEOMETRY.json';assert not out.exists()
 vertices=[];faces=[]
 for line in (DATA/'assets/mesh/pallox.obj').open():
  if line.startswith('v '):vertices.append(list(map(float,line.split()[1:4])))
  elif line.startswith('f '):
   face=[int(t.split('/')[0])-1 for t in line.split()[1:]]
   faces.extend([[face[0],face[k],face[k+1]]for k in range(1,len(face)-1)])
 vertices=np.asarray(vertices);faces=np.asarray(faces)
 selected=json.loads((PUBLIC/'FIXED_SIX_PAIR_SELECTION.json').read_text())['pair_ids']
 manifest=json.loads((PUBLIC/'EXACT_POSE_PAIR_MANIFEST.json').read_text())['pairs']
 results=[]
 for p in manifest:
  if p['pair_id']not in selected:continue
  r=p['before'];K=np.loadtxt(DATA/r['K']);T=np.loadtxt(DATA/r['pose'])
  actual,meta=render_depth(vertices,faces,K,T,(480,640))
  with np.load(PUBLIC/f"FIXED_PAIR_{p['session']}.npz")as z:other=z['render_depth_before_m']
  both=np.isfinite(actual)&np.isfinite(other);maskdiff=int(np.count_nonzero(np.isfinite(actual)!=np.isfinite(other)))
  dz=np.abs(actual[both].astype(float)-other[both].astype(float))
  # Inclusive edge tolerances are deliberately different, so preserve rather
  # than hide boundary-only discrepancies between the two implementations.
  results.append(dict(pair_id=p['pair_id'],meta=meta,common_pixels=int(both.sum()),
    finite_mask_difference=maskdiff,max_depth_difference_m=float(dz.max()),p99_depth_difference_m=float(np.percentile(dz,99)),
    differences_over_1mm=int((dz>.001).sum())))
  print(results[-1],flush=True)
 result=dict(schema='V66_INDEPENDENT_FIXED_PUBLIC_MESH_REPLAY',PASS=all(r['differences_over_1mm']==0 and r['finite_mask_difference']<=2 and r['meta']['near_plane_crossing_triangles']==0 for r in results),
  renderer1=sha(ROOT/'source_audit/mesh_depth_render.py'),renderer2=sha(PUBLIC/'audit_fixed_real_pairs.py'),
  pair_selection_sha256=sha(PUBLIC/'FIXED_SIX_PAIR_SELECTION.json'),results=results,
  source_only=True,current223_read=False,new_PnP=0,pose_or_calibration_fitting=False,
  scope='Checks implementation agreement only; supplied mesh/poses and RGB-depth registration are not human pixel truth.',seconds=time.monotonic()-start)
 out.write_text(json.dumps(result,indent=2)+'\n');print('PASS',result['PASS'],flush=True)
if __name__=='__main__':main()
