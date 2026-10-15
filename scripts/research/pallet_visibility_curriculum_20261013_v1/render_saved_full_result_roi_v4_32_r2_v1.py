"""Readable crops of saved render data only; no fresh reference/pose reads."""
from pathlib import Path
import numpy as np
from . import common as C
from . import render_clean_fresh_BI_full_result_process_v4_32_r2_v1 as R

BASE=R.OUTPUT
OUTPUT=BASE/'saved_roi_v1'
PARENT_SEAL='78bc8404d5bf4c1a6de85c60b549d1a2e32ab16979f80671463506cc1cb28a9a'


def observed_model_ROI(data,image_hw):
    """Viewport only: archived BI + own model + actual support; no GT/score."""
    h,w=image_hw;points=[np.asarray(data['coarse_projection_xy'])]
    points += [np.asarray([v['xy']for v in data['original_BI_archives']]).reshape(-1,2)]
    points += [np.asarray(g['actual_inlier_xy'])for p in data['active_cost_plans'].values()for g in p['active_groups']]
    xy=np.concatenate(points);xy=xy[np.isfinite(xy).all(1)]
    xy=xy[(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)]
    if not len(xy):return [0.,float(w),0.,float(h)]
    lo=xy.min(0);hi=xy.max(0);center=(lo+hi)/2
    span=np.maximum(hi-lo,[100.,65.]);half=span/2+np.maximum(24.,span*.25)
    return [max(0.,center[0]-half[0]),min(float(w),center[0]+half[0]),
            max(0.,center[1]-half[1]),min(float(h),center[1]+half[1])]


def main(*,root_saved_plot_GO=False):
    if not root_saved_plot_GO:raise PermissionError('Root saved-overlay visualization GO before IO')
    if OUTPUT.exists():raise FileExistsError('Preserve saved ROI render')
    if C.sha(BASE/'RENDER_SEAL.json')!=PARENT_SEAL:raise ValueError('Exact completed parent renderer')
    OUTPUT.mkdir()
    sources={str(Path(__file__).relative_to(C.ROOT)):C.sha(__file__),str(Path(R.DRAW.__file__).relative_to(C.ROOT)):C.sha(R.DRAW.__file__)}
    C.write(OUTPUT/'SOURCE_PLAN.json',dict(source_sha256=sources,parent_render_seal_sha256=PARENT_SEAL,
        scope='saved PROCESS_DATA and copied RGB only; no PRED/reference reopen, no feature query or pose solve',
        ROI_rule='BI + own coarse model + actual support bounds, 25percent or24px margin, min100x65 span; no GT/score/selected final pose',
        same_ROI_for_all_panels=True,full_original_panels_preserved=True))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    parent=C.read(BASE/'RENDER_SEAL.json');inputs={str((BASE/'RENDER_SEAL.json').relative_to(C.ROOT)):PARENT_SEAL};info=[]
    md=['# 확대 보기: 저장된 전체 결과 과정','',
        '원본 전체 프레임과 최초 렌더는 보존했다. 이 확대는 보관된 B/I 점·초기 coarse 모델·실제 선 지지만으로 정한 동일 ROI를 사용하며 참조·최종오차·선택된 최종 자세로 ROI를 고르지 않았다. 좌표는 원본 px다.','',
        '점은 raw YOLO가 아니라 기존 N3/SubPix/BI 보정 archive다. 보라선은 예측이다. O/U는 영역만, S는 모델만 사용한다. 선0인 패널은 비워 두고 active=0을 표시했다. 마지막 초록 패널만 고정 평가 참조이며 탐색·보정·fit과 무관하다. 저장되지 않은 시작 자세·연속 중간 궤적은 만들지 않았다.','']
    for ci,case in enumerate(parent['cases']):
        folder=BASE/case['folder'];data=C.read(folder/'PROCESS_DATA.json');image=np.asarray(Image.open(folder/'source_rgb.png').convert('RGB'))
        for name in('PROCESS_DATA.json','source_rgb.png'):inputs[str((folder/name).relative_to(C.ROOT))]=C.sha(folder/name)
        roi=observed_model_ROI(data,image.shape[:2]);x0,x1,y0,y1=roi
        def photo(ax,title):
            R.DRAW._photo(ax,image,title);ax.set_xlim(x0,x1);ax.set_ylim(y1,y0)
        def figure():
            f,a=plt.subplots(3,2,figsize=(16,15));f.subplots_adjust(top=.92,bottom=.06,hspace=.58,wspace=.22)
            return f,a.ravel()
        fig,axes=figure();photo(axes[0],'Actual RGB — same observed/model ROI')
        photo(axes[1],'Own coarse PREDICTION + archived B/I')
        R.DRAW._wire(axes[1],np.asarray(data['coarse_projection_xy']),color=R.DRAW.PURPLE,label='predicted own coarse')
        R.DRAW._BI(axes[1],data['original_BI_archives'])
        from matplotlib.patches import Ellipse
        for region in data['U_ranges']:
            if data['states'][region['native_corner']]!='U':continue
            if region.get('shape')=='ellipse':axes[1].add_patch(Ellipse(region['center_xy'],2*region['half_axes_px'][0],2*region['half_axes_px'][1],angle=region['angle_deg'],fill=False,edgecolor='#ff975a',linestyle=':'))
        for region in data['O_regions']:
            if data['states'][region['native_corner']]!='O':continue
            basis=np.asarray(region['basis']);axes[1].add_patch(Ellipse(region['center_xy'],2*region['axes_px'][0],2*region['axes_px'][1],angle=np.rad2deg(np.arctan2(basis[1,0],basis[0,0])),fill=False,edgecolor='#fa5ab5',linestyle=':'))
        for ax,arm in zip(axes[2:],R.M.ARMS[1:]):
            photo(ax,arm.replace('CLEAN_FRESH_BI_',''));p=data['active_cost_plans'].get(arm)
            if p is not None:R.DRAW._support(ax,p['active_groups'])
            ax.text(.01,-.19,'active='+str(0 if p is None else len(p['active_groups']))+'; owners='+str([]if p is None else p['new_owner_ids'])+'\nsaved actual finite support; no precise new KP',transform=ax.transAxes,va='top',fontsize=10)
        fig.suptitle(data['id']+' — stored observation/active-line crop',fontsize=15)
        first=f'case{ci}_observations_ROI.png';fig.savefig(OUTPUT/first,dpi=145);plt.close(fig)
        fig,axes=figure()
        for ax,arm,color in zip(axes[:5],R.M.ARMS,R.COLORS):
            photo(ax,arm.replace('CLEAN_FRESH_BI_','')+' — PREDICTED');xy=data['final_projection_xy'][arm]
            if xy is not None:R.DRAW._wire(ax,np.asarray(xy),color=color,label='stored prediction',style='-',width=1.8)
            metric=data['saved_metrics'][arm];pred=data['final_predictions'][arm]
            ax.text(.01,-.19,f"T={metric['translation_cm']:.3f} cm; R={metric['rotation_deg'] if metric['rotation_deg'] is None else round(metric['rotation_deg'],3)} deg\naccepted={pred['accepted']}; WD={None if pred['actual_pose']is None else pred['actual_pose'].get('WD_family')}",transform=ax.transAxes,va='top',fontsize=10)
        photo(axes[-1],'FIXED EVALUATION REFERENCE ONLY')
        xy=data['fixed_reference_projection_xy']
        if xy is not None:R.DRAW._wire(axes[-1],np.asarray(xy),color='#5bff77',label='evaluation reference, never search/fit',style='-',width=1.8)
        else:axes[-1].text(.03,.1,'Reference R unresolved; no wire',transform=axes[-1].transAxes)
        axes[-1].text(.01,-.19,'Last panel only; no GT/reference-guided query or pose fit.\nDerived reference is not independent physical metrology.',transform=axes[-1].transAxes,va='top',fontsize=10)
        fig.suptitle(data['id']+' — stored final predictions and separate reference',fontsize=15)
        second=f'case{ci}_final_and_reference_ROI.png';fig.savefig(OUTPUT/second,dpi=145);plt.close(fig)
        info.append(dict(id=data['id'],ROI_original_xy=roi,parent_process_sha256=C.sha(folder/'PROCESS_DATA.json'),images=[first,second]))
        md+=['## '+data['id'],'',f'![관측과 활성 선 확대]({first})','',f'![최종 예측·별도 평가 참조 확대]({second})','',
            f'[전체 원본/보관값](../{case["folder"]}/PROCESS_DATA.json) · [원본 사진](../{case["folder"]}/source_rgb.png)','']
    (OUTPUT/'REPORT_KO.md').write_text('\n'.join(md));C.write(OUTPUT/'ROI_DATA.json',info)
    C.write(OUTPUT/'RENDER_SEAL.json',dict(schema='SAVED_FULL_R2_PROCESS_ROI_ADDITIVE_SEAL',source_sha256=sources,input_sha256=inputs,
        assets={p.name:C.sha(p)for p in OUTPUT.iterdir()if p.is_file()},parent_render_seal_sha256=PARENT_SEAL,
        numerical_data_changes=0,new_reference_PRED_PnP_or_optimizer_reads_or_calls=0,visual_QA_pending=True))
    (BASE/'LATEST_REPORT_ROI_V1.md').write_text('# 전체 결과 과정 최신 확대 보기\n\n[확대 과정·최종 예측·별도 평가 참조](saved_roi_v1/REPORT_KO.md) · [최초 전체 프레임 보고서](REPORT_KO.md)\n\n동일 원본 좌표에서 저장된 관측/초기모델/실제지지로 viewport만 확대했다. 최초 full-frame/봉인은 보존했다. 모델·관측·선·평가·방법 선택을 바꾸지 않았다.\n')
    print(str(OUTPUT),C.sha(OUTPUT/'RENDER_SEAL.json'))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root-saved-plot-GO',action='store_true');a=p.parse_args();main(root_saved_plot_GO=a.root_saved_plot_GO)
