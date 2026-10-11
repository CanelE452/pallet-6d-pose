"""Render saved 2D geometric evidence; no RGB asset or pose truth access."""
import argparse
import hashlib
import json
from pathlib import Path
import math


def main():
    p=argparse.ArgumentParser();p.add_argument('--diagnostic',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('fresh absent figure output required')
    before=hashlib.sha256(a.diagnostic.read_bytes()).hexdigest()
    data=json.loads(a.diagnostic.read_bytes())
    matches=[r for r in data['rows'] if r['context']=='26.HYBRID_FACTOR_SUBSET_FINITE_EDGE_CONSENSUS.proposal.153' and r['edge']==2]
    if len(matches)!=1:raise ValueError('one fixed preselected stored witness required')
    r=matches[0];A,B=r['stored_projected_endpoints'];support=r['actual_support_points']
    normal=r['observed_normal'];tangent=[-normal[1],normal[0]]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    ax=axes[0];ax.scatter(*zip(*support),s=32,label='Actual query support',color='#2463bf')
    ax.plot([A[0],B[0]],[A[1],B[1]],color='#ba2634',linewidth=3,label='Saved projected model edge')
    ax.scatter([A[0],B[0]],[A[1],B[1]],s=40,color='#ba2634')
    lo=min(p[0] for p in support+[A,B])-15;hi=max(p[0] for p in support+[A,B])+15
    ax.plot([lo,hi],[(r['observed_offset']-normal[0]*x)/normal[1] for x in (lo,hi)],'--',color='#2463bf',alpha=.7,label='Stored observed TLS line')
    ax.set(title='Same stored observations and projection',xlabel='Native image x (px)',ylabel='Native image y (px)')
    ax.set_aspect('equal');ax.invert_yaxis();ax.legend(fontsize=8,loc='upper left',bbox_to_anchor=(0.,-1.5))
    ax=axes[1];mid=[(A[i]+B[i])/2 for i in range(2)]
    ax.plot([A[0],B[0]],[A[1],B[1]],'o-',color='#ba2634',linewidth=3,label='Projected model edge')
    ax.annotate('',xy=[mid[i]+.025*tangent[i] for i in range(2)],xytext=[mid[i]-.025*tangent[i] for i in range(2)],arrowprops={'arrowstyle':'<->','color':'#2463bf'})
    ax.text(mid[0]-.025,mid[1]+.03,'Arrow: observed line direction',fontsize=8,color='#2463bf')
    ax.set_xlim(mid[0]-.04,mid[0]+.04);ax.set_ylim(mid[1]-.04,mid[1]+.04);ax.invert_yaxis();ax.set_aspect('equal')
    ax.set(title=f"Model edge zoom: {r['projected_segment_length_px']:.6f} px",xlabel='Native image x (px)',ylabel='Native image y (px)')
    ax.set_xticks([mid[0]-.03,mid[0],mid[0]+.03]);ax.ticklabel_format(useOffset=False);ax.legend(fontsize=8)
    ax=axes[2];labels=['Old\nblock','Observed-frame\nblock','Closed-segment\nsupport RMS']
    values=[r['original_edge_RMS_px'],r['repaired_edge_RMS_px'],r['actual_support_closed_segment_RMS_px']]
    ax.bar(labels,values,color=['#bf842e','#267345','#555555'])
    ax.axhline(8.,color='#ba2634',linestyle='--',label='Existing 8 px edge gate')
    for i,v in enumerate(values):ax.text(i,v+3,f'{v:.3f}',ha='center',fontsize=9)
    ax.set(title='Geometric constraint discrepancy',ylabel='Pixel norm / distance (distinct definitions)');ax.legend(fontsize=8)
    fig.suptitle('Finite-support failure: model edge shrinks across the observed line\nSaved 21-context diagnostic; no new fit or pose-accuracy claim',fontsize=14)
    a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=130);plt.close(fig)
    if hashlib.sha256(a.diagnostic.read_bytes()).hexdigest()!=before:raise ValueError('input diagnostic changed')
    print(json.dumps(dict(figure=str(a.output),bytes=a.output.stat().st_size,sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),
                         diagnostic_sha256=before,rendered_saved_context=r['context'],new_dataset_RGB=0,new_model_PnP_GT_calls=0)))


if __name__=='__main__':main()
