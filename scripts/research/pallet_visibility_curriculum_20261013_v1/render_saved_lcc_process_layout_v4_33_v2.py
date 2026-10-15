"""Add readable caption cells to saved V33 overlays; no runtime or mask IO."""
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

from . import render_largest_component_global_outline_process_v4_33 as R

C=R.C
BASE=R.OUTPUT
OUTPUT=BASE/'display_v2'
PARENT_SEAL_SHA256='037e2b7f684edcdccf418aa56263f361c143b7574f49a1e02fdd04d3bbdbb7cd'


def main(*,root_saved_display_GO=False):
    if root_saved_display_GO is not True:
        raise PermissionError('ROOT saved-image display correction GO before IO')
    if OUTPUT.exists():raise FileExistsError('Preserve every saved display correction')
    if C.sha(BASE/'RENDER_SEAL.json')!=PARENT_SEAL_SHA256:
        raise ValueError('Exact completed parent render required')
    parent=C.read(BASE/'RENDER_SEAL.json')
    cases={case['id']:case['folder'] for case in parent['cases']}
    if set(cases)!=set(R.FIXED_IDS):raise ValueError('Same fixed three cases required')
    inputs={R.M._relative(BASE/'RENDER_SEAL.json'):PARENT_SEAL_SHA256}
    for identifier in R.FIXED_IDS:
        folder=BASE/cases[identifier]
        for filename in ('PROCESS_DATA.json','source_rgb.png'):
            relative=cases[identifier]+'/'+filename
            if C.sha(folder/filename)!=parent['assets'][relative]:raise ValueError('Saved parent asset changed')
            inputs[R.M._relative(folder/filename)]=parent['assets'][relative]
    if C.sha(R.PARENT/'RENDER_SEAL.json')!=R.PARENT_RENDER_SEAL_SHA256:
        raise ValueError('Original native observation display seal changed')
    oldseal=C.read(R.PARENT/'RENDER_SEAL.json')
    oldfolders={case['id']:case['folder'] for case in oldseal['cases']}
    inputs[R.M._relative(R.PARENT/'RENDER_SEAL.json')]=R.PARENT_RENDER_SEAL_SHA256
    for identifier in R.FIXED_IDS:
        relative=oldfolders[identifier]+'/PROCESS_DATA.json';path=R.PARENT/relative
        if C.sha(path)!=oldseal['assets'][relative]:raise ValueError('Archived native display data changed')
        inputs[R.M._relative(path)]=oldseal['assets'][relative]
    sources={R.M._relative(path):C.sha(path) for path in (Path(__file__),Path(R.__file__),Path(R.DRAW.__file__))}
    OUTPUT.mkdir()
    C.write(OUTPUT/'SOURCE_PLAN.json',dict(schema='SAVED_LCC_PROCESS_CAPTION_LAYOUT_PLAN_V33_V2',
        UTC=datetime.now(timezone.utc).isoformat(),source_sha256=sources,input_sha256=inputs,
        parent_render_seal_sha256=PARENT_SEAL_SHA256,
        change='Separate caption axes under each image; existing stored coordinates/poses/ROI unchanged; no numeric correction',
        same_case_and_ROI=True,mask_PRED_INPUTS_model_or_original_reference_document_reads=0,
        RGB_reads='three already-copied source RGBs only',reference='saved projection last separate panel',
        new_NN_PnP_TLS_optimizer_or_measurement_calls=0))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from PIL import Image
    md=['# 고정 세 사례: 설명글 공간을 분리한 확대판','',
        '첫 렌더의 설명글 일부가 축 라벨과 겹쳐, 이미지와 설명글을 별도 칸으로 나눴다. 원본 렌더·봉인은 보존했고 좌표·자세·동일 ROI·저장 오차를 바꾸지 않았다. 전수 결과의 성능이나 마스크 의미를 새로 검증하는 그림이 아니다.','',
        '보라선은 초기 예측, 노란 점은 N3/SubPix/BI 보정 archive다. S는 영상 점을 그리지 않으며 U/O는 영역만 표시한다. 실제 유한 native 선과 전경 마스크 경계는 별개다. C 패널의 회색 점선은 이전 V32 union 예측, 실선은 V33 LCC 예측으로 중간 최적화 과정이 아니다. 마지막 초록 패널만 기존 저장 평가 참조다.','',
        '[원본 전체 프레임·union/LCC 픽셀 비교](../REPORT_KO.md)','']
    information=[]
    for identifier in R.FIXED_IDS:
        folder=BASE/cases[identifier];data=C.read(folder/'PROCESS_DATA.json')
        oldpath=C.ROOT/data['parent_process_binding']['path']
        if C.sha(oldpath)!=data['parent_process_binding']['sha256']:raise ValueError('Saved old observation archive changed')
        inputs[R.M._relative(oldpath)]=C.sha(oldpath);old=C.read(oldpath)
        image=np.asarray(Image.open(folder/'source_rgb.png').convert('RGB'))
        x0,x1,y0,y1=data['ROI_original_xy']
        def grid():
            fig=plt.figure(figsize=(16,17));outer=fig.add_gridspec(3,2,left=.065,right=.98,bottom=.025,top=.94,hspace=.25,wspace=.20)
            panels=[]
            for cell in outer:
                inner=cell.subgridspec(2,1,height_ratios=(8,1.8),hspace=.34)
                ax=fig.add_subplot(inner[0]);caption=fig.add_subplot(inner[1]);caption.axis('off')
                panels.append((ax,caption))
            return fig,panels
        def photo(ax,title):
            R.DRAW._photo(ax,image,title);ax.set_xlim(x0,x1);ax.set_ylim(y1,y0)
        def note(ax,text):ax.text(.01,.92,text,transform=ax.transAxes,va='top',fontsize=10)
        ci=R.FIXED_IDS.index(identifier)
        fig,panels=grid()
        photo(panels[0][0],'Actual original RGB — saved observed/model ROI')
        note(panels[0][1],'Same original-pixel viewport; no GT/final-pose/error ROI selection.')
        ax,caption=panels[1];photo(ax,'Predicted own coarse + archived B/I + U/O ranges')
        R.DRAW._wire(ax,np.asarray(data['coarse_projection_xy']),color=R.DRAW.PURPLE,label='own coarse PREDICTION')
        context=data['archived_context'];R.DRAW._BI(ax,context['original_BI_archives'])
        for region in context['U_ranges']:
            if context['states'][region['native_corner']]!='U' or region.get('shape')!='ellipse':continue
            ax.add_patch(Ellipse(region['center_xy'],2*region['half_axes_px'][0],2*region['half_axes_px'][1],angle=region['angle_deg'],fill=False,edgecolor='#ff975a',linestyle=':'))
        for region in context['O_regions']:
            if context['states'][region['native_corner']]!='O':continue
            basis=np.asarray(region['basis'])
            ax.add_patch(Ellipse(region['center_xy'],2*region['axes_px'][0],2*region['axes_px'][1],angle=np.rad2deg(np.arctan2(basis[1,0],basis[0,0])),fill=False,edgecolor='#fa5ab5',linestyle=':'))
        note(caption,'N3/SubPix/BI archives, not raw YOLO.\nU/O regions only; no S/U/O observed-point markers.')
        for index,key,title,color in ((2,'original_union_observation','Original union — stored global-mask boundary','#ffbd45'),
            (3,'LCC_observation','LCC8 — stored global-mask boundary','#00ffff')):
            ax,caption=panels[index];photo(ax,title);R._outline(ax,data[key],color,'stored mask boundary')
            note(caption,'Convex foreground observation, not a native physical-edge identity proof.\nNo mask/observer recomputation in this display correction.')
        for (ax,caption),arm in zip(panels[4:],R.M.NATIVE_ARMS):
            photo(ax,arm.replace('CLEAN_FRESH_BI_','')+' — actual finite support')
            active=old['active_cost_plans'].get(arm)
            if active is not None:R.DRAW._support(ax,active['active_groups'])
            count=data['native_active_counts'][arm]
            if count!=(0 if active is None else len(active['active_groups'])):raise ValueError('Unchanged native support count differs')
            note(caption,f'active={count}; owners={[] if active is None else active["new_owner_ids"]}\nC retains this policy population; its selected WD may differ.')
        fig.suptitle(identifier+' — saved observations and mask-boundary control',fontsize=15)
        first=f'case{ci}_observations_ROI.png';fig.savefig(OUTPUT/first,dpi=145);plt.close(fig)
        fig,panels=grid()
        for (ax,caption),arm,color in zip(panels[:5],R.ARMS,R.OLD.COLORS):
            photo(ax,arm.replace('CLEAN_FRESH_BI_','')+' — PREDICTED')
            if arm in data['archived_union_C_projection_xy']:
                previous=data['archived_union_C_projection_xy'][arm]
                if previous is not None:R.DRAW._wire(ax,np.asarray(previous),color='#c4c4c4',label='V32 union prediction',style=':',width=1.4)
            current=data['final_projection_xy'][arm]
            if current is not None:R.DRAW._wire(ax,np.asarray(current),color=color,label='V33 stored prediction',style='-',width=1.8)
            if arm in data['archived_union_C_projection_xy']:ax.legend(loc='upper right',fontsize=8)
            pred=data['final_predictions'][arm];metric=data['saved_metrics'][arm]
            def n(value):return 'undefined' if value is None else f'{value:.4f}'
            note(caption,f'T={n(metric["translation_cm"])} cm; R={n(metric["rotation_deg"])} deg\naccepted={pred["accepted"]}; WD={pred["WD_family"]}; new active={data["native_active_counts"].get(arm,"comparison only")}')
        ax,caption=panels[-1];photo(ax,'ARCHIVED FIXED EVALUATION REFERENCE ONLY')
        reference=data['archived_reference_projection_xy']
        if reference is not None:R.DRAW._wire(ax,np.asarray(reference),color='#5bff77',label='saved reference, never search/fit',style='-',width=1.8)
        note(caption,'Last separate panel; no original reference document reopened.\nStored physical-body projection, no native-channel identity equivalence claim.')
        fig.suptitle(identifier+' — stored final predictions; no intermediate trajectory',fontsize=15)
        second=f'case{ci}_final_and_reference_ROI.png';fig.savefig(OUTPUT/second,dpi=145);plt.close(fig)
        information.append(dict(id=identifier,parent_process_sha256=C.sha(folder/'PROCESS_DATA.json'),ROI_original_xy=data['ROI_original_xy'],images=[first,second]))
        md+=['## '+identifier,'',f'![보관 관측과 mask 경계]({first})','',f'![최종 예측과 별도 보관 참조]({second})','',f'[원본 전체 mask 비교](../{cases[identifier]}/union_LCC_full_canvas.png) · [저장 수치](../{cases[identifier]}/PROCESS_DATA.json)','']
    C.write(OUTPUT/'DISPLAY_DATA.json',information)
    (OUTPUT/'REPORT_KO.md').write_text('\n'.join(md))
    for path,digest in inputs.items():
        if C.sha(C.ROOT/path)!=digest:raise ValueError('Saved display input changed')
    C.write(OUTPUT/'RENDER_SEAL.json',dict(schema='SAVED_LCC_PROCESS_CAPTION_LAYOUT_SEAL_V33_V2',source_sha256=sources,input_sha256=inputs,
        assets={path.name:C.sha(path) for path in OUTPUT.iterdir() if path.is_file()},parent_render_seal_sha256=PARENT_SEAL_SHA256,
        numerical_data_changes=0,new_PRED_INPUTS_mask_reference_NN_TLS_PnP_optimizer_reads_or_calls=0,visual_QA_pending=True))
    (BASE/'LATEST_REPORT_DISPLAY_V2.md').write_text('# V33 고정 사례 최신 확대판\n\n[설명글을 분리한 확대판](display_v2/REPORT_KO.md) · [원본 전체 union/LCC 비교와 최초 렌더](REPORT_KO.md)\n\n저장된 값·ROI·모델·관측을 바꾸지 않고 설명글 공간만 분리했다. 원본 렌더·봉인은 보존했다.\n')
    print(C.sha(OUTPUT/'RENDER_SEAL.json'))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root-saved-display-GO',action='store_true')
    a=p.parse_args();main(root_saved_display_GO=a.root_saved_display_GO)
