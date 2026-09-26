"""Freeze user-approved PnP association boxes, never PnP point supervision."""
from collections import Counter
import numpy as np
from PIL import Image
from . import common as C
from .open_existing_annotation import WORK, rows
from .labels import validate_frame, coverage


def main():
    with C.exclusive('annotating'):
        if (C.DOC/'HARD_LABEL_LOCK.json').exists():
            lock=C.read(C.DOC/'HARD_LABEL_LOCK.json')
            C.verify(lock['labels']);C.verify(lock['protocol_amendment'])
            return
        approval=C.read(C.DOC/'PNP_BOX_USER_APPROVAL.json')
        assert approval['approved'] and approval['direct_clicks_only_supervision']
        confirmation=C.read(C.DOC/'DIRECT_CLICK_USER_CONFIRMATION.json')
        assert confirmation['confidence']=='PRETTY_CONFIDENT_USER_REPORTED'
        selected=rows();frames={};bindings=[];qa=[]
        for r in selected:
            C.verify(r['image'])
            p=WORK/'annotations'/((C.ROOT/r['image']['path']).stem+'.json')
            d=C.read(p);bindings.append(C.bind(p))
            assert d['image']==r['image'] and d['selection_sha256']==C.sha(C.DOC/'HARD_SELECTION_LOCK.json')
            w,h=Image.open(C.ROOT/r['image']['path']).size
            projected=np.asarray(d['pose']['projected_all'],float)[:8]
            assert projected.shape==(8,2) and np.isfinite(projected).all()
            box=np.clip(np.r_[projected.min(0),projected.max(0)],0,[w,h,w,h]).tolist()
            corners=[]
            for a in d['keypoint_annotations'][:8]:
                manual=a['source']=='manual_click'
                corners.append(dict(status='DIRECT_VISIBLE' if manual else 'UNCERTAIN',
                                    xy=a['xy'] if manual else None,
                                    source=a['source'],reason='user-confirmed direct click' if manual else 'not supervised; physical visibility not inferred'))
            f=dict(frame_id=r['frame_id'],image=r['image'],image_sha256=r['image']['sha256'],
                   recording=r['recording'],tag=r['tag'],size=[w,h],bbox=box,
                   bbox_source='PNP_PROJECTED_CORNERS_CLIPPED',role='ROLE_CONFIDENT',
                   role_confidence='pretty confident, user reported',corners=corners,complete=True,
                   camera_K=d['camera_K'],physical_dimensions_m=d['pose']['_physical_dimensions_m'])
            errors,warnings=validate_frame(f)
            qa.append(dict(frame_id=r['frame_id'],errors=errors,warnings=warnings))
            assert not errors and not warnings,(r['frame_id'],errors,warnings)
            frames[r['frame_id']]=f
        cov=coverage(frames,selected);assert cov['passed'],cov
        # Preserve the incomplete historical custom-GUI file; use a new immutable snapshot.
        path=C.RAW/'HARD_LABELS_PNP_BOX_PRIVATE.json'
        C.save(path,dict(frames=frames,source_files=bindings,selection_sha256=C.sha(C.DOC/'HARD_SELECTION_LOCK.json')),immutable=True)
        amendment=dict(created_at=C.now(),approval=C.bind(C.DOC/'PNP_BOX_USER_APPROVAL.json'),
                       point_confirmation=C.bind(C.DOC/'DIRECT_CLICK_USER_CONFIRMATION.json'),
                       reference_protocol='HUMAN_PNP_ASSISTED_NOT_BLIND',
                       bbox='axis-aligned envelope of saved PnP projected corners0..7, clipped to image; association only; identical both arms; not manual visible-envelope',
                       support='36 direct-click corners only; other corners ignored without inferring physical visibility; P8 ignored',
                       uncertainty='User says pretty confident, not independently verified exact GT',
                       hard_losses='xy only; no box/cls/dfl/visibility supervision',
                       unchanged='selection, budget, seed, remaining clean/synthetic slots, frozen selector')
        C.save(C.DOC/'ANNOTATION_PROTOCOL_AMENDMENT.json',amendment,immutable=True)
        C.save(C.DOC/'HARD_COVERAGE_PUBLIC.json',cov,immutable=True)
        C.save(C.DOC/'HARD_PROVENANCE_PUBLIC.json',dict(coverage=cov,source_files=bindings,
               reference_protocol=amendment['reference_protocol'],bbox=amendment['bbox'],
               automatic_corner_supervision=False,P8_supervised=False,QA=qa,
               model_outputs_opened_before_lock=0,recordings=dict(Counter(r['recording'] for r in selected))),immutable=True)
        C.save(C.DOC/'HARD_LABEL_LOCK.json',dict(created_at=C.now(),labels=C.bind(path),
               selection=C.bind(C.DOC/'HARD_SELECTION_LOCK.json'),coverage=cov,
               protocol_amendment=C.bind(C.DOC/'ANNOTATION_PROTOCOL_AMENDMENT.json'),model_outputs_opened=0),immutable=True)
        C.set_state('HARD_LABELS_LOCKED_TRAINING_PENDING',training='NOT_RUN',
                    metadata_confirmation='USER_CONFIRMED',active_frames=len(frames),
                    next='Frozen teacher supplement and matched-budget training integrity checks')
        print('HARD_LABELS_LOCKED',cov)


if __name__=='__main__':main()
