"""Post-seal reference/basis/silhouette integrity check; no fits or selection."""
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import common as C
from . import evaluate_two_wd_families_v4_9 as W
from . import statewise_evaluation_v4 as E
from .global_pallet_silhouette_v4_9 import observe_mask_outline,score_projected_outline
from .diagnose_wd_outline_v4_9 import IDS,MASKS

OUTPUT=W.OUTPUT/'reference_silhouette_diagnosis_v1'


def gap(a,b):
    d=np.linalg.norm(np.asarray(a)[:,None]-np.asarray(b)[None],axis=-1)
    return float(max(d.min(0).max(),d.min(1).max()))


def main():
    assert not OUTPUT.exists()
    complete=C.read(W.OUTPUT/'PREDICTION_COMPLETE.json');scoreseal=C.read(W.OUTPUT/'SCORE_SEAL.json')
    assert complete['rows']==95
    assert C.sha(W.OUTPUT/'TEST_POSE_SCORE.json')==scoreseal['TEST_score_sha256']
    predseal=C.read(W.OUTPUT/'TEST_PREDICTION_SEAL.json')
    assert C.sha(W.OUTPUT/'TEST_PREDICTIONS.jsonl.gz')==predseal['predictions_sha256']
    rows={r['id']:r for r in C.rows(W.OUTPUT/'TEST_PREDICTIONS.jsonl.gz')}
    target=E.COHORT/'test_reused_v1/ISOLATED_TEST_SCORING.json'
    labels=C.read(target)['rows']
    records={r['id']:r for r in C.read(MASKS/'RECORDS.json')}
    OUTPUT.mkdir(parents=True)
    C.write(OUTPUT/'PROTOCOL.json',dict(ids=list(IDS),labels_used_postseal_for_integrity_only=True,
        source_sha256=C.sha(Path(__file__)),target_binding=E._binding(target),
        WD_score_seal_sha256=C.sha(W.OUTPUT/'SCORE_SEAL.json'),GT_pose_fits=0,
        target_changes=0,frame_exclusions=0,new_selection_rule=False))
    fig,axes=plt.subplots(1,3,figsize=(17,5));values=[]
    for ax,key in zip(axes,IDS):
        row=rows[key];g=row['runtime']['initial_geometry'];truth=labels[key]['target']['truth']
        record=records[key];maskpath=MASKS/record['mask_npz'];imagepath=C.ROOT/row['image']
        assert C.sha(maskpath)==record['mask_file_sha256'] and C.sha(imagepath)==row['image_sha256']
        with np.load(maskpath) as a:mask=a['pallet_mask_clean'].copy()
        observation=observe_mask_outline(mask,g['image_hw'])
        X=np.asarray(g['native_xyz']);Q=np.asarray(g['physical_registration_Q']);K=np.asarray(g['K'])
        B=C.cuboid(*truth['body_xyz']);G=C.cuboid(*truth['xyz'])
        bodyR=np.asarray(truth['body_R']);R=np.asarray(truth['R']);t=np.asarray(truth['t'])
        Qref=bodyR.T@R;registered=X@Q;bodyphysical=B@Qref
        pa=C.project(B,bodyR,t,K);pb=C.project(registered,R,t,K);pc=C.project(G,R,t,K)
        first=score_projected_outline(observation,B,K,bodyR,t)
        second=score_projected_outline(observation,registered,K,R,t)
        third=score_projected_outline(observation,G,K,R,t)
        value=dict(id=key,original_cf_extents=g['cf_extents'],reference_body_extents=truth['body_xyz'],
            registered_physical_extents=np.ptp(registered,axis=0).tolist(),reference_physical_extents=truth['xyz'],
            reference_body_to_physical_Q=Qref.tolist(),reference_body_to_physical_det=float(np.linalg.det(Qref)),
            unregistered_body_vs_registered_physical_gap_m=gap(B,registered),
            unregistered_comparison_is_different_axes_not_shape_error=True,
            reference_body_in_physical_axes_vs_registered_gap_m=gap(bodyphysical,registered),
            reference_canonical_vs_registered_gap_m=gap(G,registered),
            centering_origin_gap_m=float(np.linalg.norm(B.mean(0)-registered.mean(0))),
            body_vs_registered_reference_projection_unordered_gap_px=gap(pa,pb),
            canonical_vs_registered_reference_projection_unordered_gap_px=gap(pc,pb),
            reference_body_projection_xy=pa.tolist(),registered_reference_projection_xy=pb.tolist(),
            reference_body_outline=first,registered_reference_outline=second,canonical_reference_outline=third,
            physical_reference_center_m=t.tolist(),GT_postseal_integrity_only=True)
        values.append(value)
        image=cv2.imread(str(imagepath));assert image is not None
        ax.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));ax.axis('off')
        p=observation['interior_contour_xy'];ax.plot(*p.T,'.',color='yellow',ms=1,label='observed mask envelope')
        for p,color,style,label in [(pa,'red','-','reference body cube'),(pb,'cyan','--','registered physical cube at reference pose')]:
            h=cv2.convexHull(p.astype('float32')).reshape(-1,2);ax.plot(*np.vstack([h,h[:1]]).T,color=color,ls=style,lw=1.8,label=label)
        ax.set_xlim(0,image.shape[1]);ax.set_ylim(image.shape[0],0)
        ax.set_title(key+'\nRef IoU %.4f; basis pixel gap %.2g'%(first['hull_IoU'],value['body_vs_registered_reference_projection_unordered_gap_px']),fontsize=9)
        ax.legend(loc='upper right',fontsize=6)
    fig.suptitle('Post-seal integrity only: identical registered and reference-body geometry; no pose correction',fontsize=12)
    fig.tight_layout();fig.savefig(OUTPUT/'REFERENCE_ACTUAL_IMAGE.png',dpi=140);plt.close(fig)
    C.write(OUTPUT/'DIAGNOSIS.json',dict(schema='POSTSEAL_REFERENCE_CUBE_INTEGRITY_V4_9',rows=values,
        GT_used_for_fit=False,GT_used_for_selector=False,reference_changed=False,
        source_sha256=C.sha(Path(__file__)),figure_sha256=C.sha(OUTPUT/'REFERENCE_ACTUAL_IMAGE.png'),
        protocol_sha256=C.sha(OUTPUT/'PROTOCOL.json')))
    print(str(OUTPUT),flush=True)


if __name__=='__main__':main()
