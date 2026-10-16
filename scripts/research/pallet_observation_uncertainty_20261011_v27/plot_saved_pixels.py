"""Standalone figures from frozen V27 pixel rows; no fitting or projections."""
import gzip
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT=Path(__file__).resolve().parent


def rows(p):
    with gzip.open(p,'rt') as f:return [json.loads(s) for s in f]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p,r):
    with p.open('x') as f:json.dump(r,f,indent=2);f.write('\n')


def main():
    rr=ROOT/'output/REAL_ROWS.jsonl.gz';sr=ROOT/'output/SOURCE_ROWS.jsonl.gz'
    dest=ROOT/'figures';dest.mkdir()
    inputs={str(p):sha(p) for p in (rr,sr,Path(__file__))}
    save(dest/'PROTOCOL.json',dict(schema='saved_pixel_figures_v27',inputs=inputs,matplotlib=matplotlib.__version__,
        new_fits=0,new_projection=0,new_PnP=0,new_RGB=0,
        plots='all133; CAL conditional role348 and existing admitted177; all13 BAD local XY'))
    real=rows(rr);source=rows(sr);assert len(real)==133 and len(source)==573
    good=[r for r in real if r['candidate_proxy_quality']=='GOOD']
    bad=[r for r in real if r['candidate_proxy_quality']=='BAD'];assert len(bad)==13
    purple=[r for r in bad if r['LOO_fit_all_known_GOOD']];assert len(purple)==3
    fig,axes=plt.subplots(1,3,figsize=(19,6))
    groups=[('GOOD candidate (120)',good,'#1764ad','o'),('BAD candidate (13)',bad,'#c63227','o'),
        ('BAD; all proxy-GOOD LOO fit (3)',purple,'#772ca2','D')]
    for name,rrr,color,marker in groups:
        axes[0].scatter([r['LOO_error_px'] for r in rrr],[r['candidate_error_px'] for r in rrr],
            s=34 if marker=='o' else 65,c=color,marker=marker,label=name,alpha=.8,edgecolors='white',linewidths=.35)
        axes[1].scatter([r['reference_gain_px2'] for r in rrr],[r['LOO_gain_px2'] for r in rrr],
            s=34 if marker=='o' else 65,c=color,marker=marker,alpha=.8,edgecolors='white',linewidths=.35)
    axes[0].axvline(8,c='gray',ls='--',lw=1);axes[0].axhline(8,c='gray',ls='--',lw=1)
    axes[0].set(xlabel='Saved LOO projection error to reference (px)',ylabel='Adopted candidate error to reference (px)',
        title='All 133 adopted points / 85 frames',xlim=(0,21),ylim=(0,21))
    axes[0].legend(loc='upper left',fontsize=8)
    axes[1].axvline(0,c='gray',lw=1);axes[1].axhline(0,c='gray',lw=1)
    axes[1].set(xlabel='Reference gain: native error² - candidate error² (px²)',
        ylabel='LOO gain: native residual² - candidate residual² (px²)',
        title='53/133 worsened but passed LOO')
    pools=[('SOURCE CAL ROLE conditional (348)',[r for r in source if r['arm']=='IMAGE_ROLE'],'#269449','-'),
        ('SOURCE CAL ROLE point admitted (177)',[r for r in source if r['arm']=='IMAGE_ROLE' and r['existing_point_admission_accepted']],'#e79121','--'),
        ('REAL adopted (133)',real,'#1764ad','-')]
    for name,pool,color,style in pools:
        xx=sorted(r['candidate_error_over_radius'] for r in pool)
        axes[2].step(xx,[(i+1)/len(xx) for i in range(len(xx))],where='post',c=color,ls=style,label=name)
    axes[2].axvline(1,c='gray',ls=':',lw=1)
    axes[2].set(xlabel='Candidate reference error / recorded radius',ylabel='Empirical cumulative fraction',
        title='Different conditional populations; diagnostic only',xlim=(0,6),ylim=(0,1.03))
    axes[2].legend(loc='lower right',fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Frozen pixel audit: LOO validation error and radius transfer',fontsize=15,y=.98)
    fig.text(.5,.01,'Existing DEV proxy / source CAL. No new PnP, projection, inference or pose scoring. Radius is not LOO-pose uncertainty.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.055,1,.94));fig.savefig(dest/'ALL133_LOO_GAIN_AND_RADIUS.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(4,4,figsize=(16,15));flat=list(axes.flat)
    for ax,r in zip(flat,bad):
        g=r['reference_xy'];coordinates={key:[r[key][j]-g[j] for j in range(2)] for key in ('native_N3_xy','candidate_xy','saved_LOO_xy')}
        ax.scatter([0],[0],c='#269449',s=70,marker='*',label='Reference')
        for key,name,color,marker in [('native_N3_xy','Native N3','#e79121','o'),('candidate_xy','Candidate','#c63227','x'),('saved_LOO_xy','Saved LOO','#1764ad','+')]:
            xy=coordinates[key];ax.scatter([xy[0]],[xy[1]],c=color,s=48,marker=marker,label=name)
        c=coordinates['candidate_xy'];ax.add_patch(Circle(c,r['calibrated_candidate_radius_px'],fill=False,color='#c63227',alpha=.5))
        n=coordinates['native_N3_xy'];v=coordinates['saved_LOO_xy']
        ax.plot([n[0],c[0]],[n[1],c[1]],c='gray',lw=.7,ls='--')
        ax.plot([v[0],c[0]],[v[1],c[1]],c='#1764ad',lw=.7,ls=':')
        span=max(8,max(abs(t) for xy in coordinates.values() for t in xy)+r['calibrated_candidate_radius_px']+2)
        ax.set(xlim=(-span,span),ylim=(span,-span),aspect='equal')
        session,fid=r['frame_id'].split(':');label=session.replace('eval_','')+' / '+fid[-7:]+f' / k{r["corner"]}'
        ax.set_title(label+'\n'+f'candidate {r["candidate_error_px"]:.2f}px; LOO {r["LOO_error_px"]:.2f}px'+
            ('\nAll proxy-GOOD LOO fit' if r['LOO_fit_all_known_GOOD'] else ''),fontsize=9,
            color='#772ca2' if r['LOO_fit_all_known_GOOD'] else 'black')
        ax.grid(alpha=.2);ax.tick_params(labelsize=8)
    for ax in flat[len(bad):]:ax.axis('off')
    handles,labels=flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4)
    fig.suptitle('All 13 adopted BAD points: coordinates relative to saved DEV reference',fontsize=15,y=.995)
    fig.text(.5,.026,'Circle: recorded candidate radius. Dashed: N3 to candidate. Dotted: candidate to LOO. Image y points downward.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.07,1,.96));fig.savefig(dest/'ALL13_BAD_SAVED_PIXEL_GEOMETRY.png',dpi=150);plt.close(fig)
    assert inputs=={str(p):sha(p) for p in (rr,sr,Path(__file__))}
    save(dest/'CHECKS.json',dict(passed=True,all133_plotted=True,all13_BAD_plotted=True,inputs_unchanged=True,
        files={n:dict(sha256=sha(dest/n),bytes=(dest/n).stat().st_size) for n in ('ALL133_LOO_GAIN_AND_RADIUS.png','ALL13_BAD_SAVED_PIXEL_GEOMETRY.png')},
        new_fits=0,new_projection=0,new_PnP=0,new_RGB=0))
    print(json.dumps(dict(passed=True,all133_plotted=True,all13_BAD_plotted=True,matplotlib=matplotlib.__version__)))


if __name__=='__main__':main()
