"""Actual-image comparison of two already sealed B5 prior controls."""
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

from . import common as C
from . import evaluate_shared_edge_partial_b_v4_7 as V47


def main():
    inputs=V47.OUTPUT.parent/'b5_pnp_diagnosis_v1'
    output=V47.OUTPUT.parent/'b5_pnp_diagnosis_visual_v1'
    if output.exists():raise FileExistsError('Preserve previous factual visualization')
    source=C.read(V47.OUTPUT/'TEST_frames/eval_pallet07__1778652166837872128.json')
    path=C.ROOT/source['image']
    if C.sha(path)!=source['image_sha256']:raise ValueError('Actual RGB binding differs')
    picture=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
    score=C.read(inputs/'POSTSEAL_SCORE.json')
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams['font.family']=font.get_name()
    arms=('EXACT_CURRENT_REPLAY','NO_PRIOR')
    fig,axes=plt.subplots(2,2,figsize=(12,7.5),gridspec_kw={'height_ratios':[1.65,1.]})
    records=[]
    for column,arm in enumerate(arms):
        saved=C.read(inputs/(arm+'.json'));spec=saved['spec'];pose=saved['solution']
        observed=np.array(next(m['xy'] for m in spec['visible'] if m['native_corner']==5))
        projected=C.project(spec['xyz'],pose['R'],pose['t'],spec['K'])[5]
        error=float(np.linalg.norm(observed-projected))
        if not np.isclose(error,score['arms'][arm]['B5_residual_px']):raise ValueError('Saved residual differs')
        title='기존: 초기 자세 제약 유지' if column==0 else '비교 실험: 초기 자세 제약만 제거'
        for row in range(2):
            ax=axes[row,column];ax.imshow(picture)
            ax.scatter(*observed,c='#00c6ee',s=160,marker='x',linewidths=3,zorder=4,label='5번 보정 관측')
            ax.scatter(*projected,edgecolors='#ff7b00',facecolors='none',s=160,linewidths=2.5,zorder=5,label='PnP 재투영')
            ax.plot([observed[0],projected[0]],[observed[1],projected[1]],color='#fff600',linewidth=2)
            ax.set_xticks([]);ax.set_yticks([])
            if row==0:
                ax.set_title(title+'\n위치 오차 '+f"{score['arms'][arm]['translation_cm']:.2f}cm",fontproperties=font,fontsize=14)
                ax.legend(prop=font,loc='upper left',framealpha=.95)
            else:
                ax.set_xlim(observed[0]-45,observed[0]+35)
                ax.set_ylim(observed[1]+35,observed[1]-30)
                ax.set_title(f'5번 관측–재투영 차이: {error:.2f}px',fontproperties=font,fontsize=13)
        records.append(dict(arm=arm,observed_xy=observed.tolist(),projected_xy=projected.tolist(),residual_px=error,
            centroid_translation_error_cm=score['arms'][arm]['translation_cm']))
    fig.suptitle('같은 이미지 · 같은 보정 좌표: PnP의 초기 자세 제약 영향',fontproperties=font,fontsize=17)
    fig.tight_layout(rect=[0,0,1,.95]);output.mkdir(parents=True)
    fig.savefig(output/'B5_PRIOR_CONTROL.png',dpi=150);plt.close(fig)
    C.write(output/'MANIFEST.json',dict(image=source['image'],image_sha256=C.sha(path),
        source_sha256=C.sha(__file__),input_bindings={n:C.sha(inputs/n) for n in
            ('EXACT_CURRENT_REPLAY.json','NO_PRIOR.json','POSTSEAL_SCORE.json')},
        records=records,new_inference=False,new_GT_read=False,no_synthetic_photograph=True))
    print(output/'B5_PRIOR_CONTROL.png',flush=True)


if __name__=='__main__':main()
