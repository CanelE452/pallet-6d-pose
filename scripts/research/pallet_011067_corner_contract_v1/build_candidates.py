import re
import numpy as np
from . import common as C

def main():
    obj=C.object_data();d=obj['physical_dimensions_m'];base=C.xyz(d['x'],d['y'],d['z'])
    edges={tuple(sorted(e)) for e in C.contract()['edges'] if max(e)<8}
    frame_text=(C.ROOT/'_docs/archive/paper_support_20260830/real_gt_v2/FRAME_CONVENTION.md').read_text()
    rows=[]
    for angle in (0,90,180,270):
        a=np.deg2rad(angle);r=np.rint([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]]).astype(float)
        dims=[d['x'],d['y'],d['z']] if angle%180==0 else [d['z'],d['y'],d['x']]
        cf=C.xyz(*dims);rot=base@r.T
        dist=np.linalg.norm(cf[:,None,:]-rot[None,:,:],axis=2)
        perm=dist.argmin(axis=1).tolist()
        assert np.max(np.min(dist,axis=1))<1e-10 and sorted(perm)==list(range(9)) and perm[8]==8
        assert np.isclose(np.linalg.det(r),1) and np.allclose(r.T@r,np.eye(3))
        assert {tuple(sorted((perm[a],perm[b]))) for a,b in edges}==edges
        assert {perm[i] for i in (0,1,4,5)}=={0,1,4,5}
        line=next(x for x in frame_text.splitlines() if x.startswith(f'YAW_{angle} '))
        table=list(map(int,re.findall(r'\d+',line.split('[')[-1])))
        assert perm==table, (angle,perm,table)
        rows.append(dict(c4=f'YAW_{angle}',rotation=r.tolist(),perm_new_to_stored=perm,
            dimensions_xyz=dims,det=1,edge_preserved=True,top_bottom_preserved=True,
            interpretation='Role-coordinate change; 90/270 swaps W/D, not physical rectangular symmetry'))
    C.put(C.DOC/'C4_CANDIDATES.json',dict(candidates=rows,count=len(rows),P8_fixed=True,
        generation='3D proper rotation + exact coordinate-set matching; FRAME_CONVENTION table checked',free_matching=False))
    print('C4 four proper permutations PASS')

if __name__=='__main__':main()
