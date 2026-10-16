from common import *
import numpy as np
def main():
    old=read(PREVIOUS/'RENDER_COMPLETE.json')['rows'];hashes={r['sha256'] for r in old};errors=[];count=0
    for folder in [OUT,OUT/'mean_risk_followup',OUT/'foreground_matched']:
        rows=read(folder/'RENDER_COMPLETE.json')['rows'];new={r['sha256'] for r in rows};assert not new&hashes;hashes|=new
        for r in rows:
            assert sha(ROOT/r['image'])==r['sha256'],r['id']
            X=np.array(r['Xcf']);R=np.array(r['R']);t=np.array(r['t']);K=np.array(r['K'])
            q=(X@R.T+t)@K.T;xy=q[:,:2]/q[:,2:];errors.append(float(np.linalg.norm(xy-np.array(r['gt']),axis=-1).max()));count+=1
    assert max(errors)<1e-8
    write(OUT/'DATA_REPLAY_AUDIT.json',dict(PASS=True,new_observations=count,all_image_bytes_match=True,
        image_hashes_disjoint_across_stages_and_parent=True,independent_GT_projection_max_px=max(errors)))
    print('Image/GT replay PASS',count,max(errors))
if __name__=='__main__':main()
