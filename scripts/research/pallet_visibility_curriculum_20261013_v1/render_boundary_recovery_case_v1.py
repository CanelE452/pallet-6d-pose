"""Actual B1/B2 search bands and optional recovery; annotations are posthoc only."""
import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from . import common as C
from . import evaluation_interval_v4_2b as E
from . import statewise_evaluation_v4 as V
from .render_keypoint_process_gallery_v1 import setup_font, supported_lines

FID='eval_pallet09:1778653661653195264'
ROOT=V.FIX/'boundary_recovery_v1'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path)
    p.add_argument('--evidence',type=Path,default=ROOT/'image_evidence_v1')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output=a.output.resolve();a.evidence=a.evidence.resolve()
    if a.candidate:a.candidate=a.candidate.resolve()
    if a.output.exists():raise FileExistsError('Preserve existing figure')
    row=next(r for r in C.rows(E.OUTPUT/'PREDICTIONS.jsonl.gz') if r['id']==FID)
    rt=row['runtime'];g=rt['initial_geometry'];raw=np.asarray(rt['original_raw_points'],float)
    image=cv2.imread(str(C.ROOT/row['image']));rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    capture=a.evidence;stem=FID.replace(':','__')
    meta=C.read(capture/(stem+'.json'))
    if C.sha(capture/(stem+'.npz'))!=meta['mask_file_sha256'] or C.sha(C.ROOT/row['image'])!=row['image_sha256']:
        raise ValueError('Actual image/mask evidence changed')
    mask=np.load(capture/(stem+'.npz'))['pallet_mask']
    contours,_=cv2.findContours((mask>=.5).astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    score=C.read(E.OUTPUT/'POSTSEAL_SCORING.json')
    label_binding=score['label_bindings']['TEST']
    label_path=C.ROOT/label_binding['path']
    if C.sha(label_path)!=label_binding['sha256']:raise ValueError('Historical annotations changed')
    labels=C.read(label_path)
    label=labels['rows'][FID]
    mapping=label['metadata']['native_to_reference_corner_ids']
    truth=np.asarray(label['target']['gt'],float)[mapping]
    valid=np.asarray(label['target']['valid'],bool)[mapping]
    old={r['native_corner']:r for r in rt['visible_correction']['records']}
    candidate=C.read(a.candidate) if a.candidate else None
    measured={m['native_corner']:m for m in candidate['visible_measurements']} if candidate else {}
    setup_font();fig,axes=plt.subplots(2,2,figsize=(12,8.8))
    for col,k in enumerate((1,2)):
        for rr in (0,1):
            ax=axes[rr,col];ax.imshow(rgb);ax.set_xlim(raw[k,0]-43,raw[k,0]+20);ax.set_ylim(raw[k,1]+30,raw[k,1]-30)
            ax.set_xticks([]);ax.set_yticks([])
            for contour in contours:
                q=contour[:,0,:];ax.plot(q[:,0],q[:,1],color='#05c979',lw=1.4)
            ax.scatter(*raw[k],color='#007bea',marker='o',s=65,facecolors='none',linewidths=2,label='원예측')
            if valid[k]:ax.scatter(*truth[k],color='black',marker='D',s=35,label='사후 주석')
        mate=2 if k==1 else 1;P=np.asarray(g['initial_projection_xy'])
        guide=P[mate]-P[k];length=np.linalg.norm(guide);d=guide/length;n=np.array([-d[1],d[0]])
        if n@(P.mean(0)-raw[k])<0:n=-n
        cfg=rt['visible_correction']['configuration'];lo=cfg['along_start_px'];hi=min(cfg['along_end_px'],.8*length);radius=cfg['normal_search_radius_px']
        corners=np.array([raw[k]+s*d+t*n for s,t in ((lo,-radius),(hi,-radius),(hi,radius),(lo,radius))])
        axes[0,col].add_patch(Polygon(corners,facecolor='#ed5555',alpha=.25,edgecolor='#c71e1e',lw=2))
        error=np.linalg.norm(raw[k]-truth[k])
        axes[0,col].set_title(f'KP{k} 기존: 세로 경계 후보 ±12px\n원예측 주석 오차 {error:.2f}px · 유효 샘플 0/15',fontsize=12)
        supported_lines(axes[0,col],old[k])
        ax=axes[1,col]
        if k in measured:
            m=measured[k];xy=np.asarray(m['xy']);supported_lines(ax,next(r for r in candidate['records'] if r['native_corner']==k))
            ax.scatter(*xy,color='#eb8110',s=55,label='수정 관측')
            ax.annotate('',xy,xytext=raw[k],arrowprops={'arrowstyle':'->','lw':2,'color':'#eb8110'})
            improved=np.linalg.norm(xy-truth[k])
            status='영상 교점' if m['source']!='VISIBLE_BACKBONE_FALLBACK' else '원예측 유지'
            ax.set_title(f'KP{k} 수정: {status}\n주석 오차 {error:.2f} → {improved:.2f}px',fontsize=12)
        else:
            ax.set_title(f'KP{k}: 실제 SAM 외곽 경계와 영상 근거',fontsize=12)
        ax.legend(loc='lower left',fontsize=8,framealpha=.9)
    fig.suptitle('B라고 판정해도 경계 탐색이 보정을 못 하는 이유 — 실제 KP1·KP2',fontsize=14)
    fig.text(.025,.022,'녹색=실제 SAM 외곽. 빨강=기존 선 후보 중심의 탐색 범위. 파랑=원예측, 검정=사후 주석.\n'
             '주석은 비교에만 사용하며 보정 입력이 아닙니다. 새 점은 실제 영상 선의 교점이어야 합니다.\n'
             '이 사진에는 별도의 축·코너 등록 불일치가 있으므로 KP 오차 감소를 전체 자세 복구와 동일시하지 않습니다.',fontsize=9)
    fig.subplots_adjust(left=.035,right=.98,top=.89,bottom=.14,wspace=.12,hspace=.30)
    a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=175);plt.close(fig)
    context=None
    if candidate:
        context=a.output.with_name(a.output.stem+'_context.png')
        fig,axes=plt.subplots(1,2,figsize=(12,4.8))
        for col,ax in enumerate(axes):
            ax.imshow(rgb);ax.set_xlim(170,545);ax.set_ylim(308,150)
            ax.set_xticks([]);ax.set_yticks([])
            for k in (1,2,4,7):
                ax.scatter(*raw[k],color='#007bea',s=48,facecolors='none',linewidths=1.8)
                if col:
                    xy=np.asarray(measured[k]['xy'])
                    ax.scatter(*xy,color='#eb8110',s=27)
                    if k in (1,2):
                        ax.annotate('',xy,xytext=raw[k],arrowprops={'arrowstyle':'->','lw':1.6,'color':'#eb8110'})
                        ax.text(xy[0]-28,xy[1]-4 if k==1 else xy[1]+11,f'KP{k}',color='#a84e00',fontsize=11,
                                bbox={'facecolor':'white','alpha':.8,'edgecolor':'none','pad':1})
                elif k in (4,7):
                    ax.scatter(*np.asarray(measured[k]['xy']),color='#eb8110',s=27)
                elif k in (1,2):
                    ax.text(raw[k,0]+4,raw[k,1]-4 if k==1 else raw[k,1]+11,f'KP{k}',color='#007bea',fontsize=11,
                            bbox={'facecolor':'white','alpha':.8,'edgecolor':'none','pad':1})
            ax.set_title('기존: KP1·KP2 원좌표 유지' if col==0 else '수정: 실제 영상 선의 교점으로 보정',fontsize=14)
        fig.text(.04,.035,'파랑=원예측 · 주황=영상 경계로 보정한 점. KP4·KP7의 기존 보정값은 그대로 보존했습니다.\n'
                 '이 비교는 B 키포인트의 영상 좌표입니다. 최종 PnP 자세의 성능은 별도로 검증해야 합니다.',fontsize=10)
        fig.subplots_adjust(left=.025,right=.985,top=.91,bottom=.15,wspace=.08)
        fig.savefig(context,dpi=175);plt.close(fig)
    C.write(a.output.with_suffix('.json'),{'actual_image_sha256':row['image_sha256'],'actual_mask_file_sha256':meta['mask_file_sha256'],
            'historical_annotations_used_only_for_posthoc_overlay':True,'GT_inputs_to_recovery':False,
            'candidate_sha256':C.sha(a.candidate) if a.candidate else None,'figure_sha256':C.sha(a.output),
            'renderer_sha256':C.sha(__file__),'context_figure':str(context) if context else None,
            'context_figure_sha256':C.sha(context) if context else None})


if __name__=='__main__':main()
