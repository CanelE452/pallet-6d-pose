"""User-requested postscore display of the exact ten TEST49 threshold gains.

Stored predictions only. No inference, PnP, SubPix, target, mask or fitting.
Native CF cuboids use stored cf_extents/R_cf/centroid, not R_physical.
"""
from pathlib import Path

import numpy as np
from PIL import Image

from . import common as C

B=C.DOC/'nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1'
PAIRS=B/'raw_yolo_vs_N3_subpix_TEST49_paired_v1/PAIR_METRICS.json'
OLD=C.DOC/'nonsevere_v1/posefix_sub_comparison_v1'
OUTPUT=C.DOC/'raw_n3_subpix_gain10_contact_v1'
METHODS=('NATIVE_BASE','INITIAL_N3_SUBPIX')
EDGES=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))


def projection(pose,K):
    if pose['available'] is not True:
        raise ValueError('All requested ten cases have saved poses')
    return C.project(C.cuboid(*pose['cf_extents']),pose['R_cf'],pose['centroid'],K)


def render():
    if OUTPUT.exists():raise FileExistsError('Preserve every displayed comparison')
    pairs=C.read(PAIRS)
    ids=pairs['groups']['gain']
    if len(ids)!=10 or pairs['groups']['loss'] or pairs['group_counts']['gain']!=10:
        raise ValueError('Exactly the user-requested ten improvements, zero losses')
    frames={r['frame']['id']:r['frame'] for r in C.read(OLD/'BASELINE_INPUTS.json')['frames'] if r['frame']['id'] in ids}
    saved={}
    for row in C.rows(OLD/'SCORED_ROWS.jsonl.gz'):
        if row['id'] in ids and row['method'] in METHODS:
            key=(row['id'],row['method'])
            if key in saved:raise ValueError('Duplicate saved case/method')
            saved[key]=row
    if len(frames)!=10 or len(saved)!=20:raise ValueError('All ten actual image/prediction pairs required')
    inputs={str(p.relative_to(C.ROOT)):C.sha(p) for p in (PAIRS,OLD/'BASELINE_INPUTS.json',
        OLD/'SCORED_ROWS.jsonl.gz',OLD/'PREDICTION_SEAL.json')}
    OUTPUT.mkdir()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results=[]
    for page in range(2):
        fig,axes=plt.subplots(5,2,figsize=(13,15),layout='constrained')
        fig.suptitle('Same images: YOLO only (left) / N3 + SubPix + total cap (right)\nDots: predicted keypoints; wire: saved PnP prediction. No ground-truth overlay.',fontsize=13)
        for line,fid in enumerate(ids[page*5:page*5+5]):
            f=frames[fid];path=C.ROOT/f['image']
            if C.sha(path)!=f['image_sha256']:raise ValueError('Exact original RGB required')
            inputs[f['image']]=f['image_sha256']
            photo=np.asarray(Image.open(path).convert('RGB'));h,w=photo.shape[:2]
            if [h,w]!=f['raw_hw']:raise ValueError('Original canvas required')
            q=np.asarray(f['points']['BASE'])[:8]
            valid=np.isfinite(q).all(1)&~(q==-1).all(1)
            lo=q[valid].min(0);hi=q[valid].max(0)
            margin=np.maximum((hi-lo)*.08,22.)
            x0,y0=np.maximum(lo-margin,0);x1,y1=np.minimum(hi+margin,[w-1,h-1])
            rowproof=dict(id=fid,image_sha256=f['image_sha256'],viewport_from_raw_YOLO_only=[x0,x1,y0,y1],methods={})
            for col,(method,color) in enumerate(zip(METHODS,('#ff963e','#00e4ff'))):
                r=saved[(fid,method)];points=np.asarray(r['native_points'])[:8]
                expected=np.asarray(f['points']['BASE' if col==0 else 'N3_SUBPIX'])[:8]
                if not np.array_equal(points,expected,equal_nan=True):raise ValueError('Exact archived native KP stage required')
                xy=projection(r['actual_pose'],f['K']);ax=axes[line,col]
                ax.imshow(photo);ax.set_xlim(x0,x1);ax.set_ylim(y1,y0);ax.axis('off')
                for a,b in EDGES:ax.plot(xy[[a,b],0],xy[[a,b],1],color=color,linewidth=1.4)
                good=np.isfinite(points).all(1)&~(points==-1).all(1)
                ax.scatter(points[good,0],points[good,1],s=18,c='white',edgecolors=color,linewidths=.8,zorder=4)
                for k in np.flatnonzero(good):
                    ax.annotate(str(k),points[k],xytext=(3,-9),textcoords='offset points',color='white',fontsize=7,
                        bbox=dict(facecolor='black',alpha=.45,pad=.3,edgecolor='none'))
                m=r['metric'];status='PASS' if m['success_5cm_5deg'] else 'FAIL'
                ax.set_title(f"{page*5+line+1:02d} {fid}\n{status}   T={m['translation_cm']:.2f} cm / R={m['rotation_deg']:.2f} deg",fontsize=10)
                rowproof['methods'][method]=dict(saved_metric=m,projection_xy=xy.tolist(),native_points=points.tolist(),pose=r['actual_pose'])
            results.append(rowproof)
        fig.savefig(OUTPUT/f'cases_{page*5+1:02d}_{page*5+5:02d}.png',dpi=145);plt.close(fig)
    C.write(OUTPUT/'DISPLAY_DATA.json',dict(rows=results,exact_case_count=10,GT_reference_or_new_pose_computation=0,
        S_U_visibility_classification_applied=False,scope='Initial model predictions before statewise refinement; points are not verified observations'))
    (OUTPUT/'REPORT_KO.md').write_text('# 원 YOLO와 N3+SubPix의 실제 개선 10장\n\n동일 TEST49 중 개선 10장, 퇴행 0장이다. 각 행 왼쪽은 원 YOLO, 오른쪽은 N3+SubPix+총 이동 제한 후의 결과다. 점은 저장된 예측 KP, 선은 저장된 PnP 자세이며 정답 오버레이가 아니다. 현재 상태별 후처리를 적용한 비교는 아니다.\n\n![1–5](cases_01_05.png)\n\n![6–10](cases_06_10.png)\n\n전체49의 나머지 실패도 원 비교에 포함돼 있다. 이 그림은 개선된 10장만 설명하며, 모든 사진이나 최대오차가 좋아졌다는 뜻은 아니다.\n')
    C.write(OUTPUT/'PROVENANCE.json',dict(schema='POSTHOC_EXACT_TEN_INITIAL_PREDICTION_CONTACT_DISPLAY_V1',
        assets={p.name:C.sha(p) for p in OUTPUT.iterdir() if p.is_file()},
        source_binding=dict(path=str(Path(__file__).resolve().relative_to(C.ROOT)),sha256=C.sha(__file__)),
        input_sha256=inputs,ids=ids,new_NN_SubPix_PnP_fit_or_reference_reads=0,
        ROI='Raw YOLO points only; never corrected points/final pose or reference',all_dots_are_predictions_not_observed_GT=True))
    return OUTPUT


if __name__=='__main__':print(render())
