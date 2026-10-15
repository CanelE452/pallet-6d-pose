"""Actual RGB, all NMS alternatives and matched OFF/ON pose comparison."""
from pathlib import Path
from html import escape
import argparse
import numpy as np
import cv2

from . import common as C
from . import statewise_evaluation_v4 as E
from . import evaluate_image_chain_mixtures_v4_18 as ON
from . import evaluate_old_normal_t4_only_v4_18 as OFF
from .build_native_edge_actual_report_v4_17 import _wire,_project,COLORS

OUTPUT=ON.OUTPUT.parent/'NMS_ON_OFF_actual_process_report_v4_18'
CASES=('wood_night_01:031344','wood_night_01:031426','eval_noapril:1775201428535961600')


def main(output=OUTPUT):
    output=Path(output)
    if output.exists():raise FileExistsError('Preserve prior report')
    sources=[];methods=[];scores=[]
    for module in(OFF,ON):
        folder=module.OUTPUT;seal=C.read(folder/'PREDICTION_SEAL.json');scoreseal=C.read(folder/'SCORE_SEAL.json')
        if seal['rows']!=238 or C.sha(folder/'PREDICTIONS.jsonl.gz')!=seal['predictions_sha256']:raise ValueError('Complete238 prediction seal required')
        for p,h in scoreseal['score_sha256'].items():
            if C.sha(folder/(p+'_SCORE.json'))!=h:raise ValueError('Scored source bytes changed')
        methods.append({r['id']:r for r in C.rows(folder/'PREDICTIONS.jsonl.gz')if r['id']in CASES})
        scores.append({p:C.read(folder/(p+'_SCORE.json'))for p in('TEST','REMAINDER')})
        sources.extend(E._binding(folder/n)for n in('PROTOCOL.json','PREDICTION_SEAL.json','SCORE_SEAL.json'))
    tasks={r['id']:r for r in C.rows(ON.OUTPUT/'INPUTS.jsonl.gz')if r['id']in CASES}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True);html=[];cases=[]
    for ident in CASES:
        task=tasks[ident];imagepath=C.ROOT/task['image']
        if C.sha(imagepath)!=task['image_sha256']:raise ValueError('Bound actual RGB required')
        image=cv2.cvtColor(cv2.imread(str(imagepath)),cv2.COLOR_BGR2RGB)
        fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
        for ax in axes.flat:ax.imshow(image);ax.set_axis_off()
        axes[0,0].set_title('1. Actual source RGB')
        points=[]
        for m in task['spec']['visible']:
            k=m['native_corner'];xy=np.array(m['xy']);points.append(xy[None])
            axes[0,1].plot(*xy,marker='o',ms=5,mfc='none',mec=COLORS[task['states'][k]])
            axes[0,1].annotate(str(k)+':'+task['states'][k],xy,xytext=(3,-4),textcoords='offset points',fontsize=9,color=COLORS[task['states'][k]])
        for ray in task['new_chain_observation']['edge_records']:
            for chain in ray.get('chains',[]):
                xy=np.array(chain['observed_points'])[chain['inlier_mask']];points.append(xy)
                axes[0,1].scatter(xy[:,0],xy[:,1],s=3,c='#00dbed',alpha=.65)
        axes[0,1].set_title('2. Frozen B/I (circles) and ALL actual NMS chains (cyan)')
        before=methods[0][ident]['arms'][OFF.ARMS[1]];after=methods[1][ident]['arms'][ON.ARMS[1]]
        qb,qa=_project(task,before),_project(task,after);points.extend([qb,qa])
        _wire(axes[1,0],qb,'#0784ff',label='OFF: old normals + T4',linestyle='--')
        _wire(axes[1,0],qa,'#ff9700',label='ON: new NMS mixtures')
        axes[1,0].legend(loc='lower left',fontsize=8);axes[1,0].set_title('3. Same proposal bank and hard guards; native-energy selector')
        xy=np.vstack(points);lo=xy.min(axis=0)-18;hi=xy.max(axis=0)+18
        for ax in(axes[0,1],axes[1,0]):
            ax.set_xlim(max(0,lo[0]),min(image.shape[1],hi[0]));ax.set_ylim(min(image.shape[0],hi[1]),max(0,lo[1]));ax.set_aspect('equal')
        axes[1,1].clear();axes[1,1].axis('off');lines=[ident,'Matched OFF/ON controls, no new pose optimization.','Every parallel line alternative is retained.','Line support anchors are statistics, not precise keypoints.','S: model projection only. U: broad regions only.','']
        for i,(module,label)in enumerate(((OFF,'OFF'),(ON,'ON'))):
            r=next(r for r in scores[i][task['partition']]['rows']if r['id']==ident);m=r['metrics'][module.ARMS[1]]
            lines.extend([label+': '+module.ARMS[1],f'  {m["translation_cm"]:.3f} cm / {m["rotation_deg"]:.3f} deg',
                '  family '+methods[i][ident]['arms'][module.ARMS[1]]['WD_family'],''])
        lines.extend(['Additional143 reference: annotation/PnP derived.','This is not independent physical metrology.','These controls do not preserve all original49 successes.'])
        axes[1,1].text(0,1,'\n'.join(lines),va='top',fontsize=10,transform=axes[1,1].transAxes)
        fig.suptitle('Actual-image process: '+ident,fontsize=14)
        filename=ident.replace(':','__')+'.png';fig.savefig(output/filename,dpi=155);plt.close(fig)
        html.append('<article><h2>'+escape(ident)+'</h2><a href="'+filename+'"><img src="'+filename+'"></a></article>')
        cases.append(dict(id=ident,image=E._binding(imagepath),artifact=filename))
    summary=[]
    for i,module in enumerate((OFF,ON)):
        for arm in module.ARMS[1:]:
            cells=[]
            for part in('TEST','REMAINDER'):
                s=scores[i][part]['summary'][arm];d=s['translation_cm'];cells.extend([str(s['translation_le5'])+'/'+str(scores[i][part]['count']),f'{d["mean"]:.3f}',f'{d["max"]:.3f}'])
            summary.append('<tr><th>'+arm+'</th>'+''.join('<td>'+x+'</td>'for x in cells)+'</tr>')
    doc='<!doctype html><html lang="ko"><meta charset="utf-8"><title>NMS 실제 과정 비교</title><style>body{font:15px sans-serif;background:#f2f6f9;color:#162435;margin:22px}article{background:white;padding:15px;margin:16px 0}img{width:100%;height:auto}table{border-collapse:collapse;background:white}th,td{border:1px solid #bbc8d3;padding:9px}</style><h1>새 경계선 정보: 실제 과정 비교</h1><p>동일한 저장 자세 후보·O/T/U 제약·기존 B/I 관측을 사용하고, 기존 부분 법선만 사용할 때와 새 NMS 혼합 경계선을 적용할 때를 비교했습니다. CAL 46장은 주 평가에서 분리했고, severe는 제외했습니다. 추가 143장 기준은 annotation/PnP 유도값입니다. 기존 49장 모두 통과한 기본 방법은 유지합니다.</p><table><tr><th>방법</th><th>49장 ≤5cm</th><th>평균 cm</th><th>최대 cm</th><th>143장 ≤5cm</th><th>평균 cm</th><th>최대 cm</th></tr>'+''.join(summary)+'</table>'+''.join(html)+'</html>'
    (output/'index.html').write_text(doc,encoding='utf-8')
    C.write(output/'RECEIPT.json',dict(cases=cases,sources=sources+[E._binding(Path(__file__))],actual_pixels_only=True,
        all_NMS_alternatives_displayed=True,precise_hidden_or_U_KPs_created=0,new_pose_fits=0,default_changed=False))
    C.write(output/'SEAL.json',dict(artifacts=[E._binding(p)for p in sorted(output.iterdir())if p.is_file()]))
    print(str(output/'index.html'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args();main(a.output)
