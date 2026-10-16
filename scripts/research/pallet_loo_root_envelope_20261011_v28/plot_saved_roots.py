"""Plot all133 frozen adopted-point root preference envelopes."""
import gzip
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent


def save(p,r):
    with p.open('x') as f:json.dump(r,f,indent=2);f.write('\n')


def main():
    path=ROOT/'output_corrected/DECISION_ROWS.jsonl.gz';out=ROOT/'figures';out.mkdir()
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    inputs={str(p):sha(p) for p in (path,Path(__file__))}
    save(out/'PROTOCOL.json',dict(schema='frozen_root_preference_figures_v28',inputs=inputs,
        matplotlib=matplotlib.__version__,new_fitting=0,new_PnP=0,new_projection=0,new_RGB=0))
    with gzip.open(path,'rt') as f:rr=[json.loads(s) for s in f]
    adopted=[r for r in rr if r['original_adopted']];assert len(adopted)==133
    fig,axes=plt.subplots(1,2,figsize=(15,6))
    for ax,name,title in [(axes[0],'DIM_ALLOWED_GE4','All dimension-allowed roots with >=4 inliers'),
        (axes[1],'DIM_ALLOWED_MAX_INLIER_COUNT','Roots with maximal inlier count')]:
        for outcome,label,color in [('BETTER','Coordinate improved (80)','#1764ad'),('WORSE','Coordinate worsened (53)','#c63227')]:
            these=[r for r in adopted if r['coordinate_outcome']==outcome]
            ax.scatter([r['pools'][name]['gain_min_px2'] for r in these],
                [r['roots'][r['original_selected_root_index']]['gain_px2'] for r in these],c=color,label=label,s=36,alpha=.8,
                edgecolors='white',linewidths=.4)
        stable=[r for r in adopted if r['pools'][name]['all_candidate_preferred']]
        worse=sum(r['coordinate_outcome']=='WORSE' for r in stable)
        ax.axvline(0,color='gray',lw=1);ax.axhline(0,color='gray',lw=1)
        ax.set_xscale('symlog',linthresh=1)
        ax.set(xlabel='Minimum gain across saved root pool (px², symlog)',ylabel='Original selected-root gain (px²)',title=title)
        ax.text(.03,.96,f'All roots prefer candidate: {len(stable)}/133\nCoordinate worsened in this subset: {worse}/{len(stable)}',
            transform=ax.transAxes,ha='left',va='top',fontsize=10,bbox=dict(facecolor='white',alpha=.85,edgecolor='lightgray'))
        ax.grid(alpha=.2);ax.legend(loc='lower right',fontsize=9)
    fig.suptitle('All 133 adopted points: multiple LOO roots disagree, and agreement can still be wrong',fontsize=13)
    fig.text(.5,.02,'Existing DEV pixel proxy. Pool chosen without GT; alternate-root local rank not revalidated. No new PnP/projection/policy/pose scoring.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.065,1,.94));p=out/'ALL133_ROOT_PREFERENCE_ENVELOPES.png';fig.savefig(p,dpi=150);plt.close(fig)
    assert inputs=={str(p):sha(p) for p in (path,Path(__file__))}
    save(out/'CHECKS.json',dict(passed=True,all133_adopted_points_plotted=True,inputs_unchanged=True,
        file=dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size),new_fitting=0,new_PnP=0,new_projection=0,new_RGB=0))
    print(json.dumps(dict(passed=True,all133_plotted=True,matplotlib=matplotlib.__version__)))


if __name__=='__main__':main()
