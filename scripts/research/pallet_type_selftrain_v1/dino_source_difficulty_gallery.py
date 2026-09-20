"""First six existing train-tail examples by row; diagnostic, no sample selection."""
import cv2
import numpy as np
from . import dino_mid_feature as U


def main():
    doc=U.C.read(U.DOC/'FULL_SOURCE_DIFFICULTY_AUDIT.json');source=U.P.SourceData()
    chosen=sorted([r for r in doc['records'] if r['partition']=='train' and r['far40']>0],key=lambda r:r['row'])[:6]
    panels=[];bindings=[]
    for r in chosen:
        x=source.item(r['row']);U.C.verify(x['image_binding']);image=cv2.imread(str(U.C.ROOT/x['image_binding']['path']))
        h,w=image.shape[:2];scale=min(600/w,470/h);canvas=np.full((520,600,3),25,np.uint8)
        im=cv2.resize(image,(round(w*scale),round(h*scale)));canvas[50:50+len(im),:im.shape[1]]=im
        for points,valid,color,prefix in [(x['original_gt'][:8],x['original_gt_valid'][:8],(0,240,0),'G'),
            (x['original_points'][:8],x['valid'][:8],(255,220,0),'P')]:
            for j,(q,v) in enumerate(zip(points,valid)):
                if not v or not np.isfinite(q).all():continue
                xy=(round(q[0]*scale),round(q[1]*scale)+50)
                cv2.circle(canvas,xy,4,color,1,cv2.LINE_AA);cv2.putText(canvas,prefix+str(j),(xy[0]+4,xy[1]-4),cv2.FONT_HERSHEY_SIMPLEX,.35,color,1,cv2.LINE_AA)
        cv2.putText(canvas,r['id']+' far40='+str(r['far40']),(8,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(240,240,240),1)
        cv2.putText(canvas,'Existing synthetic TRAIN only; green GT / cyan R0',(8,39),cv2.FONT_HERSHEY_SIMPLEX,.4,(240,240,240),1)
        panels.append(canvas);bindings.append(dict(row=r['row'],image=x['image_binding']))
    out=U.C.OUT/'selftrain_recovery_v1'/U.PHASE;out.mkdir(parents=True,exist_ok=True);path=out/'source_tail_contact.jpg'
    assert not path.exists();assert cv2.imwrite(str(path),np.concatenate([np.concatenate(panels[:3],1),np.concatenate(panels[3:],1)],0),[cv2.IMWRITE_JPEG_QUALITY,95])
    U.C.freeze(out/'SOURCE_TAIL_CONTACT_RECEIPT.json',dict(source=U.C.bound(__file__),image=U.C.bound(path),rows=bindings,
        selection='First six rows in existing source TRAIN with far40>0; no real GT and no training changes.'))
    print(path,flush=True)


if __name__=='__main__':main()
