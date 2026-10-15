"""Actual-image process comparison of sealed v17 native-edge controls."""
from pathlib import Path
from html import escape
import argparse
import os
import numpy as np
import cv2

from . import common as C
from . import statewise_evaluation_v4 as E
from . import evaluate_cached_native_edges_v4_17 as CACHE
from . import evaluate_refined_native_edges_v4_17 as REFINE

OUTPUT = CACHE.OUTPUT.parent/'native_edge_actual_process_report_v4_17'
CASES = ('wood_night_01:031344', 'wood_night_01:031426', 'eval_pallet07:1778652150610404864')
EDGES = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
COLORS = dict(B='#00aaff', I='#a778ff', O='#ed9a26', T='#d24c75', U='#a0a0a0')


def _wire(ax, q, color, *, label=None, linestyle='-', alpha=1.):
    q = np.asarray(q)
    for i,(k,m) in enumerate(EDGES):
        ax.plot(q[[k,m],0], q[[k,m],1], color=color, lw=1.5,
            linestyle=linestyle, alpha=alpha, label=label if i==0 else None)


def _project(task, arm):
    value = arm['actual_pose']; family = arm.get('WD_family')
    if family is None:
        family = arm.get('solution',{}).get('family',{}).get('name')
    if family is None:
        # The archived e model extents exactly identify its registered family.
        family = next(k for k,v in task['families'].items()
            if np.allclose(v['cf_extents'], value['cf_extents'], atol=1e-12, rtol=0))
    return CACHE.NATIVE.V.project_native(np.asarray(task['families'][family]['native_xyz']),
        np.asarray(value['R_cf']), np.asarray(value['centroid']), np.asarray(task['spec']['K']))[0]


def main(output=OUTPUT):
    output=Path(output).resolve()
    if output.exists(): raise FileExistsError('Preserve prior report')
    folders=(CACHE.OUTPUT, REFINE.OUTPUT); rows_by=[]; scores=[]; source_bindings=[]
    for folder in folders:
        seal=C.read(folder/'PREDICTION_SEAL.json'); scoring=C.read(folder/'SCORE_SEAL.json')
        if (seal['rows']!=238 or C.sha(folder/'PREDICTIONS.jsonl.gz')!=seal['predictions_sha256']
                or C.sha(folder/'PROTOCOL.json')!=seal['protocol_sha256']):
            raise ValueError('Complete source predictions before reports')
        for partition,digest in scoring['score_sha256'].items():
            if C.sha(folder/f'{partition}_SCORE.json')!=digest:raise ValueError('Frozen score binding mismatch')
        rows_by.append({r['id']:r for r in C.rows(folder/'PREDICTIONS.jsonl.gz')})
        scores.append({p:C.read(folder/f'{p}_SCORE.json')for p in ('CAL','TEST','REMAINDER')})
        source_bindings.extend(E._binding(folder/n)for n in ('PREDICTION_SEAL.json','SCORE_SEAL.json','PROTOCOL.json'))
    tasks={t['id']:t for t in C.rows(CACHE.OUTPUT/'INPUTS.jsonl.gz') if t['id']in CASES}
    original={t['id']:t for t in C.rows(CACHE.ARCHIVE.OUTPUT/'INPUTS.jsonl.gz') if t['id']in CASES}
    output.mkdir(parents=True); artifacts=[]; case_html=[]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    for ident in CASES:
        task=tasks[ident]; old=original[ident]['finish_input']['_original_source']
        image_path=C.ROOT/task['image']
        if C.sha(image_path)!=task['image_sha256']:raise ValueError('Exact bound actual RGB required')
        image=cv2.cvtColor(cv2.imread(str(image_path)),cv2.COLOR_BGR2RGB)
        fig,axes=plt.subplots(2,3,figsize=(17,8),constrained_layout=True)
        for ax in axes.flat:
            ax.imshow(image); ax.set_xlim(0,image.shape[1]); ax.set_ylim(image.shape[0],0)
            ax.set_aspect('equal'); ax.set_axis_off()
        axes[0,0].set_title('1. Actual RGB')
        geom=old['initial_geometry']; raw=np.asarray(old['original_raw_points'])
        _wire(axes[0,1],geom['initial_projection_xy'],'#ffb13b',label='initial model projection')
        for k,state in enumerate(task['states']):
            if state=='S':continue
            if raw.shape==(8,2)and np.isfinite(raw[k]).all()and np.min(raw[k])>=0:
                axes[0,1].plot(*raw[k],marker='x',c=COLORS[state],ms=6)
                axes[0,1].annotate(str(k)+':'+state,raw[k],xytext=(3,-3),textcoords='offset points',color=COLORS[state],fontsize=8)
        axes[0,1].set_title('2. Initial inference and fixed states; S=model only')
        for m in task['spec']['visible']:
            k=m['native_corner'];xy=np.asarray(m['xy'])
            axes[0,2].plot(*xy,marker='o',mfc='none',mec=COLORS[task['states'][k]],ms=5)
            axes[0,2].annotate(str(k)+':'+task['states'][k],xy,xytext=(3,-3),textcoords='offset points',fontsize=8,color=COLORS[task['states'][k]])
        for d in task['spec']['domains']+task['u_regions']:
            if d.get('shape')=='ellipse':
                axes[0,2].add_patch(Ellipse(d['center_xy'],2*d['half_axes_px'][0],2*d['half_axes_px'][1],
                    angle=d.get('angle_deg',0.),fill=False,ec=COLORS[d.get('state','U')],lw=1.))
        axes[0,2].set_title('3. Frozen visible B/I and O/T/U regions')
        baseline=rows_by[0][ident]['arms']['SAME_FROZEN_E']
        for ax,source,armname,title in (
            (axes[1,0],rows_by[0],CACHE.ARMS[2],'4. Cached candidates: native evidence first'),
            (axes[1,1],rows_by[1],REFINE.ARMS[2],'5. Continuous native-edge refinement')):
            arm=source[ident]['arms'][armname]
            _wire(ax,_project(task,baseline),'#1592ff',label='previous e',linestyle='--',alpha=.8)
            _wire(ax,_project(task,arm),'#ff9e00',label='new candidate')
            ax.legend(loc='lower left',fontsize=7);ax.set_title(title)
        axes[1,2].set_axis_on();axes[1,2].cla();axes[1,2].axis('off')
        partition=task['partition'];lines=[ident,'All rows retained; no GT pose used for fitting.','Blue dashed: previous e; orange: new method.','Partial line statistics are not precise keypoints.','U remains a region; S comes from pose projection.','']
        for i,names in ((0,('SAME_FROZEN_E',CACHE.ARMS[1],CACHE.ARMS[2])),(1,(REFINE.ARMS[1],REFINE.ARMS[2]))):
            metricrow=next(r for r in scores[i][partition]['rows'] if r['id']==ident)
            for name in names:
                m=metricrow['metrics'][name]
                lines.extend([name,f'  translation {m["translation_cm"]:.3f} cm; rotation {m["rotation_deg"]:.3f} deg'])
        axes[1,2].text(0.,1.,'\n'.join(lines),va='top',fontsize=9,transform=axes[1,2].transAxes)
        fig.suptitle(ident+' — actual-image process; exploratory comparison',fontsize=13)
        filename=ident.replace(':','__')+'.png';fig.savefig(output/filename,dpi=150);plt.close(fig)
        artifacts.append(dict(id=ident,path=filename,image=E._binding(image_path)))
        case_html.append(f'<article><h2>{escape(ident)}</h2><a href="{filename}"><img src="{filename}" alt="Actual process {escape(ident)}"></a></article>')
    summary_rows=[]
    for index,arms in ((0,('SAME_FROZEN_E',CACHE.ARMS[1],CACHE.ARMS[2])),(1,(REFINE.ARMS[1],REFINE.ARMS[2]))):
        for arm in arms:
            values=[]
            for p in ('TEST','REMAINDER'):
                v=scores[index][p]['summary'][arm];d=v['translation_cm']
                values.append(f'<td>{v["translation_le5"]}/{scores[index][p]["count"]}</td><td>{d["mean"]:.3f}</td><td>{d["max"]:.3f}</td>')
            summary_rows.append('<tr><th>'+escape(arm)+'</th>'+''.join(values)+'</tr>')
    html='<!doctype html><html lang="ko"><meta charset="utf-8"><title>Native-edge 실제 과정 비교</title><style>body{font:15px sans-serif;margin:24px;background:#f4f7fa;color:#182634}article{background:white;padding:16px;margin:16px 0}img{width:100%;height:auto}table{border-collapse:collapse;background:white}th,td{padding:9px;border:1px solid #cbd3dc}p{max-width:1100px}</style><h1>Native-edge 실제 영상 비교</h1><p>기존 49장과 추가 143장에 동일한 방법을 적용한 탐색 실험입니다. CAL 46장은 주 평가에 합치지 않았습니다. 추가 143장의 기준 자세는 annotation/PnP로 유도되어 독립 실측 정확도와 구분합니다. 모든 실패·fallback을 포함하며 S는 자세 투영, U는 영역으로 유지합니다.</p><table><tr><th>방법</th><th>49장 ≤5cm</th><th>평균 cm</th><th>최대 cm</th><th>143장 ≤5cm</th><th>평균 cm</th><th>최대 cm</th></tr>'+''.join(summary_rows)+'</table>'+''.join(case_html)+'</html>'
    (output/'index.html').write_text(html,encoding='utf-8')
    C.write(output/'RECEIPT.json',dict(schema='NATIVE_EDGE_ACTUAL_IMAGE_PROCESS_REPORT_V4_17',cases=artifacts,
        sources=source_bindings+[E._binding(Path(__file__))],reference_wire_drawn=False,
        actual_RGB_bound=True,new_pose_or_point_fits=0,new_method_default_adoption=False))
    C.write(output/'SEAL.json',dict(artifacts=[E._binding(p)for p in sorted(output.iterdir())if p.is_file()]))
    print(str(output/'index.html'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUTPUT)
    a=parser.parse_args();main(a.output)
