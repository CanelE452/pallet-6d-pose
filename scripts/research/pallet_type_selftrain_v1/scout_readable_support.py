"""Read-only RGB scout for easier manual annotation; not a training split."""
import cv2
import numpy as np
from . import common as C
from .real_support_review import closure

OUT=C.OUT/'large_corner_recovery_v1/readability_scout'


def main():
    pool=C.read(C.DOC/'POOL.json')
    groups=C.read(C.ROOT/'data/pallet/results/site_environment_audit_v1/SOURCE_RECORDING_GROUPS.json')
    dev=C.read(C.DOC/'EVAL_PROTOCOL.json')['records']
    seeds={str((C.ROOT/r['image']['path']).parent.parent.relative_to(C.ROOT)) for r in dev}
    seeds|={s['session_key'] for g in groups['groups'] if g['recording_id'] in pool['evaluation_recording_ids'] for s in g['sessions']}
    excluded=closure(seeds,groups);hashes={r['image']['sha256'] for r in dev}
    roots=['data/pallet/raw_data/'+n for n in ['capture02','capture03','capture0403middle','real_data','vdoframes']]
    rows=[];tiles=[]
    for root in roots:
        if root in excluded:continue
        directory=C.ROOT/root;rgb=directory/'rgb' if (directory/'rgb').is_dir() else directory
        paths=sorted(p for p in rgb.iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg'})
        for i in range(4):
            p=paths[((2*i+1)*len(paths))//8];binding=C.bound(p)
            assert binding['sha256'] not in hashes
            im=cv2.imread(str(p));assert im is not None
            tile=cv2.resize(im,(480,360));banner=np.zeros((30,480,3),np.uint8)
            cv2.putText(banner,f'{len(rows)+1:02d} {directory.name} / sample{i+1}',(8,21),0,.6,(255,255,255),1)
            tiles.append(np.concatenate([banner,tile],axis=0))
            rows.append(dict(index=len(rows)+1,image=binding,session=root,has_K=(directory/'cam_K.txt').exists()))
    OUT.mkdir(parents=True,exist_ok=True)
    contact=np.concatenate([np.concatenate(tiles[i:i+4],axis=1) for i in range(0,len(tiles),4)],axis=0)
    dest=OUT/'contact.jpg'
    if not dest.exists():assert cv2.imwrite(str(dest),contact)
    C.freeze(OUT/'SCOUT.json',dict(records=rows,GT_used=False,training=False,
        warning='RGB inspection only; object type/calibration/training history need separate confirmation. No annotations replaced.',
        sources=[C.bound(__file__),C.bound(C.DOC/'POOL.json'),C.bound(C.DOC/'EVAL_PROTOCOL.json')]))
    print(dest);print([(r['index'],r['session'],r['has_K']) for r in rows])


if __name__=='__main__':main()
