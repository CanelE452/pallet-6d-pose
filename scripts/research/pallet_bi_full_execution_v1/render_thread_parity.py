"""Authored-pose strict raster parity for software worker-count restriction."""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from . import material_assets as A


def run(root,output):
    assets=A.AssetRepository(root);renderer=A.VisibleRenderer();records=[]
    poses=[(None,[1.1,.11,1.3]),(None,[.8,.14,.59]),('scene.usd',[1.,.15,1.2]),
        ('scene.usd',[1.2,.15,1.]),('scene_1.usd',[1.,.14,1.2]),('scene_1.usd',[1.2,.14,1.])]
    R=Rotation.from_euler('xyz',[60.,10.,5.],degrees=True).as_matrix();t=np.array([0.,0.,3.])
    try:
        for name,dims in poses:
            model,registration=assets.get(dims,name)
            for h,w in [(480,640),(540,960)]:
                K=np.array([[608.,0,w/2],[0,608.,h/2],[0,0,1.]])
                rgb,depth,mask=renderer.render(model,dims,R,t,K,[h,w])
                records.append(dict(source=name,dimensions=dims,hw=[h,w],
                    rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                    depth_sha256=hashlib.sha256(depth.tobytes()).hexdigest(),
                    mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest()))
    finally:renderer.close()
    result=dict(authored_known_poses_only=True,current_real_GT_reads=0,LP_NUM_THREADS=os.environ.get('LP_NUM_THREADS'),
        GL_implementation=A.hardware_info(),records=records,source_sha256=A.sha(__file__))
    path=Path(output);path.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--asset-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.asset_root,a.output)
