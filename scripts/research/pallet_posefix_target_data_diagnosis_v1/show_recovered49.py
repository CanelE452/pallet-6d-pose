"""Render every natural hard TRAIN corner recovered by frozen FULL125."""
import copy
import html
from pathlib import Path
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from . import common as C

def main():
    out=C.DOC/'recovered49';assert not out.exists();out.mkdir();(out/'figures').mkdir()
    torch.set_num_threads(4);cv2.setNumThreads(1)
    binding=C.read(C.DOC/'PREDICTIONS_FROZEN.json')['models']['FULL125'];C.verify(binding)
    predictions=C.read(C.ROOT/binding['path']);census=C.read(C.RAW/'TRAIN_CORNER_CENSUS.json');lookup={(r['index'],r['corner_id']):r for r in census}
    hard=[r for r in predictions if r['mode']=='NATURAL_OCC' and lookup[(r['index'],r['corner_id'])]['occ_error_px']>20]
    selected=sorted([r for r in hard if r['output_error']<=10],key=lambda r:(lookup[(r['index'],r['corner_id'])]['display_index'],r['corner_id']))
    assert len(hard)==50 and len(selected)==49
    assert len({(r['index'],r['corner_id']) for r in selected})==49
    public=[];bindings=[];cache={}
    for number,r in enumerate(selected,1):
        idx=r['index'];j=r['corner_id'];meta=C.pair(idx)['metadata'];target=np.array(meta['target_original'][j]);before=np.array(r['input_xy']);after=np.array(r['output_xy'])
        np.testing.assert_allclose(np.linalg.norm(before-target),r['input_error'],atol=1e-9);np.testing.assert_allclose(np.linalg.norm(after-target),r['output_error'],atol=1e-9)
        assert r['input_error']>20 and r['output_error']<=10
        if idx not in cache:
            C.verify(meta['image']);bindings.append(meta['image']);bgr=cv2.imread(str(C.ROOT/meta['image']['path']));assert C.B.P.array_sha(bgr)==meta['clean_RGB_sha']
            tensor=torch.from_numpy(bgr[:,:,::-1].copy().transpose(2,0,1)).float()[None]/255
            C.D.apply_occlusion(tensor,copy.deepcopy(meta['plans']))
            occ=np.clip(np.rint(tensor[0].numpy().transpose(1,2,0)[:,:,::-1]*255),0,255).astype(np.uint8);assert C.B.P.array_sha(occ)==meta['occluded_RGB_sha']
            cache[idx]=(bgr[:,:,::-1],occ[:,:,::-1])
        clean,occ=cache[idx];allpts=np.stack([target,before,after]);low=allpts.min(0)-35;high=allpts.max(0)+35
        low=np.maximum(low,[0,0]);high=np.minimum(high,[640,480])
        fig,axs=plt.subplots(1,3,figsize=(15,4.8),facecolor='#11222d')
        titles=['CLEAN RGB: location only (not the OCC model input)',f'BEFORE: OCC R0 | error {r["input_error"]:.2f}px',f'AFTER: FULL125 | error {r["output_error"]:.2f}px']
        for k,ax in enumerate(axs):
            ax.imshow(clean if k==0 else occ,interpolation='nearest');ax.scatter(*target,marker='x',s=120,c='#65ff76',linewidths=2,label='fixed pseudo target (NOT GT)')
            if k==0:
                ax.add_patch(Rectangle(low,*(high-low),fill=False,edgecolor='white',linewidth=1.5));ax.text(target[0]+6,target[1]-7,f'P{j}',color='#65ff76',fontsize=12)
                ax.set_xlim(0,640);ax.set_ylim(480,0)
            else:
                xy=before if k==1 else after;col='#ffa92f' if k==1 else '#38e8ff'
                ax.scatter(*xy,facecolors='none',edgecolors=col,s=140,linewidths=2,label='R0 input' if k==1 else 'FULL125 output')
                ax.plot([xy[0],target[0]],[xy[1],target[1]],'--',color=col,alpha=.8)
                ax.set_xlim(low[0],high[0]);ax.set_ylim(high[1],low[1]);ax.legend(fontsize=8,loc='best')
            ax.set_title(titles[k],color='white',fontsize=10);ax.tick_params(colors='white',labelsize=8)
        fig.suptitle(f'{number:02d}/49 | P{j} | {r["input_error"]:.2f}px -> {r["output_error"]:.2f}px | TRAIN pseudo-target recovery, NOT independent evaluation',color='white',fontsize=12)
        fig.tight_layout();path=out/'figures'/f'corner_{number:02d}.png';fig.savefig(path,dpi=110,facecolor=fig.get_facecolor());plt.close(fig)
        public.append(dict(case=number,corner=j,before_px=r['input_error'],after_px=r['output_error'],path=f'figures/{path.name}'))
        if number%10==0:print('RENDERED',number,flush=True)
    for b in bindings:C.verify(b)
    C.verify(binding)
    frames=len({r['index'] for r in selected});intro=f'49개는 이미지49장이 아니라 {frames}개 프레임에 포함된 코너49개입니다. 같은 사진이라도 확인하는 코너가 다르면 별도 표시합니다. 실제 인공가림 RGB→R0 입력→FULL125 출력의 저장 결과만 사용했습니다. 이번에 추론·학습을 다시 하지 않았습니다.'
    legend='왼쪽은 위치를 알아보기 위한 가림 전 원본, 가운데는 인공가림 후 R0 입력, 오른쪽은 같은 가림 영상의 FULL125 출력입니다. 가운데/오른쪽 확대 범위는 같습니다. 초록 ×는 고정 수도레이블이며 실제 GT가 아닙니다. 숫자는 원영상 좌표의 픽셀 거리입니다.'
    md=['# 실제 hard TRAIN 복구 49개 — 코너별 전후 비교','',intro,'',legend,'','[전체 HTML 갤러리](GALLERY.html)','']
    page=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>복구49개 각각 보기</title><style>body{background:#11222d;color:#edf5f8;font:19px system-ui;margin:24px auto;max-width:1700px;padding:0 16px}p{line-height:1.7}img{width:100%;display:block}section{margin:35px 0}a{color:#76d7ff}.note{padding:18px;background:#233a46;position:relative}</style><h1>49개 코너: 보정 전 → 후</h1>',f'<div class="note"><p>{intro}</p><p>{legend}</p><p>50개 hard 중 복구49개만 모은 갤러리이므로 전체 성능의 무작위 표본이 아닙니다. 남은1개 실패는 기존 전체 보고서에 보존했습니다.</p></div>']
    for r in public:
        title=f'{r["case"]:02d}/49 · P{r["corner"]} · {r["before_px"]:.2f}px → {r["after_px"]:.2f}px'
        md+=['',f'## {title}','',f'![{title}]({r["path"]})']
        page.append(f'<section id="case{r["case"]}"><h2>{html.escape(title)}</h2><a href="{r["path"]}" target="_blank"><img loading="lazy" src="{r["path"]}" alt="{html.escape(title)}"></a></section>')
    C.save(out/'REPORT_KO.md','\n'.join(md)+'\n');C.save(out/'GALLERY.html','\n'.join(page)+'</html>\n');C.save(out/'SUMMARY.json',dict(corners=49,unique_frames=frames,hard_total=50,new_inference=False,new_training=False,reference='frozen pseudo target, not physical GT',cases=public))
    files=[out/'REPORT_KO.md',out/'GALLERY.html',out/'SUMMARY.json',Path(__file__).resolve()]+sorted((out/'figures').glob('*.png'))
    C.save(out/'PUBLICATION_MANIFEST.json',dict(files=[C.bind(p) for p in files],input_hashes_verified=True))
    print('READY',49,frames,flush=True)

if __name__=='__main__':main()
