import cv2
import numpy as np
from . import common as C

def main():
    geometry=C.read(C.RAW/'GEOMETRY_LOCK.json');assert C.sha(C.DOC/'GEOMETRY_ONLY_CANDIDATES.json')==geometry['geometry']['sha256']
    intr=C.read(C.ANN)['camera_data']['intrinsics'];K=np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1]],float)
    q=C.points();direct=C.direct_mask();rows=[]
    for c in C.read(C.DOC/'C4_CANDIDATES.json')['candidates']:
        p=c['perm_new_to_stored'];m=direct[p];v=q[p];xyz=C.xyz(*c['dimensions_xyz'])
        if m[:8].sum()<4:
            rows.append(dict(c4=c['c4'],status='INSUFFICIENT_DIRECT_MANUAL'));continue
        ok,rvecs,tvecs,_=cv2.solvePnPGeneric(xyz[m],v[m],K,None,flags=cv2.SOLVEPNP_SQPNP)
        solutions=[]
        for rv,tv in zip(rvecs,tvecs):
            rv,tv=cv2.solvePnPRefineLM(xyz[m],v[m],K,None,rv,tv)
            R=cv2.Rodrigues(rv)[0];cam=xyz@R.T+tv.reshape(3)
            proj=cv2.projectPoints(xyz,rv,tv,K,None)[0].reshape(-1,2)
            n=R@np.array([0.,0.,-1.]);fcenter=cam[:4].mean(0);direction=-fcenter/np.linalg.norm(fcenter)
            errors=np.linalg.norm(proj[m]-v[m],axis=1)
            solutions.append(dict(mean_reprojection_px=float(errors.mean()),rms_reprojection_px=float(np.sqrt(np.mean(errors**2))),
                cheirality=bool((cam[:8,2]>0).all()),depth_min_m=float(cam[:8,2].min()),
                LR_violations=sum(int(proj[a,0]>=proj[b,0]) for a,b in C.LR),
                TB_violations=sum(int(proj[a,1]>=proj[b,1]) for a,b in C.TB),
                FR_violations=sum(int(cam[a,2]>=cam[b,2]) for a,b in C.FR),
                upright_abs_R11=float(abs(R[1,1])),WHD=[c['dimensions_xyz'][i] for i in (0,1,2)],
                front_area_px2=C.area(proj[:4]),front_normal_camera_cos=float(n@direction),
                pose_R=R.tolist(),pose_t=tv.reshape(3).tolist()))
        rows.append(dict(c4=c['c4'],status='DIRECT_MANUAL_ONLY' if ok else 'SOLVE_UNAVAILABLE',
                         n=int(m[:8].sum()),solutions=solutions))
    C.put(C.DOC/'PNP_CANDIDATE_DIAGNOSTIC.json',dict(candidates=rows,semantic_choice=None,
        use='DIAGNOSTIC_ONLY',new_model_inference=0,new_GT_pose=False,
        limitation='C4 is a proper role-coordinate change with W/D parity swap; equal fit need not identify semantic front. No projected/extrapolated points used.'))
    C.figure_table(C.FIG/'05_pnp_candidate_diagnostic.png','Direct-manual-only PnP: fit is NOT semantic truth',
        ['C4','Manual n','Reprojection mean px','LR / TB / FR violations','Front normal cos'],
        [[r['c4'],r['n'],f'{r["solutions"][0]["mean_reprojection_px"]:.4f}',
          ' / '.join(str(r['solutions'][0][k+'_violations']) for k in ('LR','TB','FR')),
          f'{r["solutions"][0]["front_normal_camera_cos"]:.4f}'] for r in rows if r.get('solutions')])
    print([(r['c4'],[(s['mean_reprojection_px'],s['LR_violations'],s['FR_violations']) for s in r['solutions']]) for r in rows])

if __name__=='__main__':main()
