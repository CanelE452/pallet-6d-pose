"""User-authorized RGB-only extreme-low subset; preserve full evaluation."""
import argparse
from collections import Counter
import cv2
import numpy as np
from . import common as C
from . import elevation_review as V

DOC=C.DOC/'large_corner_recovery_v1/extreme_low_subset_v1'
OUT=C.OUT/'large_corner_recovery_v1/extreme_low_subset_v1'


def prepare():
    rows=C.read(V.OUT/'FRAMES.json')
    for b in C.read(V.DOC/'PREPARED.json')['sources']:C.verify(b)
    C.freeze(DOC/'PROTOCOL.json',dict(status='AUTHORIZED_CRITERION_REVIEW_NOT_YET_APPLIED',
        scope='PLASTIC194 only. GREEN150,WOOD125 and negatives unchanged.',
        user_confirmation='User replied 엉 to excluding only nearly invisible upper-face extreme-low scenes.',
        criterion='Exclude only when RGB shows the upper face nearly edge-on and its corners cannot be distinguished reliably. Do not exclude solely for a LOW tag, blur, darkness, occlusion, small object size, prediction error, or missing detection. Keep uncertain/borderline cases.',
        process='Assistant qualitative review of all194 original RGBs, session/id order, no GT/prediction/metric overlay. Freeze list before subset scoring. Past development exposure remains; not a new independent test or preregistered original benchmark.',
        output='New immutable subset manifest only; original files/labels/splits/reports remain unchanged. Excluded eval images remain prohibited as training data.',
        full_set_remains_reported=True,recovery_goal_not_satisfied_by_exclusion=True,
        sources=[C.bound(V.OUT/'FRAMES.json'),C.bound(__file__)]))
    OUT.mkdir(parents=True,exist_ok=True)
    for start in range(0,len(rows),12):
        tiles=[]
        for i,row in enumerate(rows[start:start+12],start):
            C.verify(row['image']);im=cv2.imread(str(C.ROOT/row['image']['path']));assert im is not None
            tile=cv2.resize(im,(640,480));banner=np.zeros((30,640,3),np.uint8)
            cv2.putText(banner,f'{i:03d} {row["session"]} / {row["elevation"]}',(8,22),0,.65,(255,255,255),1)
            tiles.append(np.concatenate([banner,tile]))
        while len(tiles)<12:tiles.append(np.zeros_like(tiles[0]))
        sheet=np.concatenate([np.concatenate(tiles[j:j+3],1) for j in range(0,12,3)])
        p=OUT/f'sheet_{start//12:02d}.jpg'
        if not p.exists():assert cv2.imwrite(str(p),sheet,[cv2.IMWRITE_JPEG_QUALITY,94])
    print('Prepared',len(rows),'original RGBs in17 contact sheets; no exclusions applied')


if __name__=='__main__':prepare()
