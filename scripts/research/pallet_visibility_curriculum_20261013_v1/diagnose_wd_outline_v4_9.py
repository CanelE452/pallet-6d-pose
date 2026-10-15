"""Prediction-only diagnosis of three sealed W/D mask comparisons.

No pose fitting, GT/annotation reads, per-S local query or parameter selection.
Raw versus clean masks and all saved swapped-family samples are evaluated
using exactly the existing whole-mask envelope score.
"""
from pathlib import Path
from copy import deepcopy

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from . import common as C
from . import evaluate_two_wd_families_v4_9 as WD
from . import statewise_evaluation_v4 as E
from .global_pallet_silhouette_v4_9 import observe_mask_outline, score_projected_outline

IDS=('eval_pallet07:1778652140531310080','eval_pallet07:1778652142480077056',
     'eval_pallet09:1778653661653195264')
OUTPUT=WD.OUTPUT/'outline_diagnosis_v1'
MASKS=WD.OUTPUT.parent/'image_evidence_v4_6'


def main():
    if OUTPUT.exists():
        raise FileExistsError('Preserve prior diagnostic')
    protocol=C.read(WD.OUTPUT/'PROTOCOL.json')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):
        E._verify_sources(protocol[key])
    complete=C.read(WD.OUTPUT/'PREDICTION_COMPLETE.json')
    assert complete['rows']==95
    seal=C.read(WD.OUTPUT/'TEST_PREDICTION_SEAL.json')
    assert C.sha(WD.OUTPUT/'TEST_PREDICTIONS.jsonl.gz')==seal['predictions_sha256']
    wd={r['id']:r for r in C.rows(WD.OUTPUT/'TEST_PREDICTIONS.jsonl.gz')}
    prior={r['id']:r for r in C.rows(WD.PREVIOUS.OUTPUT/'TEST_PREDICTIONS.jsonl.gz')}
    maskseal=C.read(MASKS/'SEAL.json')
    assert C.sha(MASKS/'RECORDS.json')==maskseal['records_sha256']
    records={r['id']:r for r in C.read(MASKS/'RECORDS.json')}
    OUTPUT.mkdir(parents=True)
    source_binding={str(Path(__file__).relative_to(C.ROOT)):C.sha(Path(__file__)),
        str(Path(WD.__file__).relative_to(C.ROOT)):C.sha(Path(WD.__file__)),
        'scripts/research/pallet_visibility_curriculum_20261013_v1/global_pallet_silhouette_v4_9.py':
            C.sha(Path(__file__).parent/'global_pallet_silhouette_v4_9.py')}
    C.write(OUTPUT/'PROTOCOL.json',dict(ids=list(IDS),GT_inputs=False,new_pose_fits=0,
        source_sha256=source_binding,WD_TEST_prediction_seal_sha256=C.sha(WD.OUTPUT/'TEST_PREDICTION_SEAL.json'),
        mask_records_sha256=C.sha(MASKS/'RECORDS.json'),all_choices_diagnostic_only=True))
    fig,axes=plt.subplots(3,3,figsize=(17,11))
    rows=[]
    colors={'INITIAL_REGULARIZED':'#dc52ff','ORIGINAL_NO_PRIOR':'#00b6ff','SWAPPED_WD_NO_PRIOR':'#00ed77'}
    for index,key in enumerate(IDS):
        row=wd[key];base=prior[key];g=row['runtime']['initial_geometry'];record=records[key]
        imagepath=C.ROOT/row['image'];maskpath=MASKS/record['mask_npz']
        assert C.sha(imagepath)==row['image_sha256']==record['image_sha256']
        assert C.sha(maskpath)==record['mask_file_sha256']
        image=cv2.imread(str(imagepath));assert image is not None
        with np.load(maskpath) as a:
            raw=a['pallet_mask_raw'].copy();clean=a['pallet_mask_clean'].copy()
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        observations={name:observe_mask_outline(mask,g['image_hw']) for name,mask in [('raw',raw),('clean',clean)]}
        candidates=[]
        for name,arm,geometry in [
            ('INITIAL_REGULARIZED',base['arms'][WD.PREVIOUS.ARMS[0]],g),
            ('ORIGINAL_NO_PRIOR',row['arms'][WD.ARMS[0]],g),
            ('SWAPPED_WD_NO_PRIOR',row['arms'][WD.ARMS[1]],row['families'][1])]:
            pose=arm['actual_pose'];solution=arm['solution']
            assert np.allclose(np.asarray(pose['R_cf'])@np.asarray(geometry['physical_registration_Q']),pose['R_physical'],atol=1e-9,rtol=0)
            score={n:score_projected_outline(o,geometry['native_xyz'],g['K'],pose['R_cf'],pose['centroid']) for n,o in observations.items()}
            projection=C.project(geometry['native_xyz'],pose['R_cf'],pose['centroid'],g['K'])
            hull=cv2.convexHull(projection.astype(np.float32)).reshape(-1,2)
            candidates.append(dict(name=name,accepted=arm['accepted'],objective=solution.get('objective'),
                image_objective_components=solution.get('objective_components'),cf_extents=geometry['cf_extents'],
                projection_xy=projection.tolist(),native_hull_xy=hull.tolist(),outline=score))
        samples=[]
        for family in row['families']:
            for sample_index,sample in enumerate(family['solution'].get('feasible_pose_samples',[])):
                samples.append(dict(family=family['name'],sample_index=sample_index,objective=sample['objective'],
                    outline={n:score_projected_outline(o,family['native_xyz'],g['K'],sample['R'],sample['t']) for n,o in observations.items()}))
        outputrow=dict(id=key,image=row['image'],image_sha256=row['image_sha256'],mask_file_sha256=record['mask_file_sha256'],
            changed_cleanup_pixels=int(np.count_nonzero(raw!=clean)),
            observed={n:{k:deepcopy(o[k]) for k in ('mask_area_px','hull_area_px','reference_diagonal_px','frame_margin_px')} for n,o in observations.items()},
            candidates=candidates,all_feasible_samples=samples,GT_inputs=False)
        rows.append(outputrow)
        for column in range(3):
            ax=axes[index,column];ax.imshow(rgb);ax.set_xlim(0,image.shape[1]);ax.set_ylim(image.shape[0],0);ax.axis('off')
        overlay=np.zeros((*raw.shape,4));overlay[raw>=.5]=[1,.85,0,.21]
        axes[index,0].imshow(overlay)
        contours,_=cv2.findContours((clean>=.5).astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            p=c.reshape(-1,2);axes[index,0].plot(*np.vstack([p,p[:1]]).T,color='yellow',lw=1.1)
        axes[index,0].set_title(key+'\nRaw mask fill / clean mask boundary',fontsize=9)
        for column in (1,2):
            p=observations['clean']['interior_contour_xy'];axes[index,column].plot(*p.T,'.',color='#ffd700',ms=.8,label='observed convex envelope')
        for c in candidates:
            p=np.asarray(c['native_hull_xy']);axes[index,1].plot(*np.vstack([p,p[:1]]).T,color=colors[c['name']],lw=1.7,
                label=c['name']+' loss %.4f'%c['outline']['clean']['outline_loss'])
        axes[index,1].legend(fontsize=6,loc='upper right');axes[index,1].set_title('Whole-mask fit; no physical reference',fontsize=10)
        f=row['families'][1]
        for j,s in enumerate(f['solution'].get('feasible_pose_samples',[])):
            p=C.project(f['native_xyz'],s['R'],s['t'],g['K']);hull=cv2.convexHull(p.astype('float32')).reshape(-1,2)
            axes[index,2].plot(*np.vstack([hull,hull[:1]]).T,lw=1.5,label='SWAP sample%d obj %.3f'%(j,s['objective']))
        axes[index,2].legend(fontsize=6,loc='upper right');axes[index,2].set_title('All saved SWAP samples; no new fit',fontsize=10)
    fig.suptitle('Observed mask envelope is a geometric cue, not a physical-axis or corner certificate',fontsize=13)
    fig.tight_layout();fig.savefig(OUTPUT/'ACTUAL_OUTLINE_COMPARISON.png',dpi=140);plt.close(fig)
    C.write(OUTPUT/'DIAGNOSIS.json',dict(schema='PREDICTION_ONLY_WD_MASK_DIAGNOSIS_V4_9',rows=rows,
        GT_inputs=False,new_pose_fits=0,new_image_forwards=0,global_RGB_file_reads=3,S_local_pixel_queries=0,
        mask_hull_is_explicit_approximation=True,physical_native_GT_phase_not_used=True,
        source_sha256=source_binding,figure_sha256=C.sha(OUTPUT/'ACTUAL_OUTLINE_COMPARISON.png')))
    E._verify_sources(source_binding)
    print(str(OUTPUT),flush=True)


if __name__=='__main__':
    main()
