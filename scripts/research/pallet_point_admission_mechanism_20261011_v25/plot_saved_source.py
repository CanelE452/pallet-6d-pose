"""Visualize all four already identified saved source corner cases; no fits."""
import gzip
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT=Path(__file__).resolve().parent
READY='_docs/experiments/pallet_kp_supervision_repair_20261010_v1/READY_SOURCE_TARGET_ROWS.jsonl.gz'
OLD='_docs/experiments/pallet_source_corner_admission_20261011_v24'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def rows(p):
    with gzip.open(p,'rt') as f:return [json.loads(s) for s in f if s.strip()]


def save(p,r):
    with p.open('x') as f:json.dump(r,f,ensure_ascii=False,indent=2);f.write('\n')


def draw_line(ax,l,color,lim):
    n=l['normal'];b=l['offset']
    if abs(n[1])>=abs(n[0]):
        x=lim[0];y=[(b-n[0]*v)/n[1] for v in x]
    else:
        y=lim[1];x=[(b-n[1]*v)/n[0] for v in y]
    ax.plot(x,y,color=color,lw=1.2)


def main():
    out=ROOT/'source_plot';out.mkdir()
    subset=ROOT/'git_blob_subset'
    inputs={'source_ready':subset/READY,'source_decoded':subset/OLD/'decoder/DECODED_FRAMES.jsonl.gz',
        'audited_corners':ROOT/'output/CORNER_ROWS.jsonl.gz','code':Path(__file__)}
    before={k:dict(sha256=sha(p),bytes=p.stat().st_size) for k,p in inputs.items()}
    save(out/'PROTOCOL.json',dict(inputs=before,selection='All four IMAGE_ROLE source decoder-selected cases with saved reference error >8px.',
        point_admission='Existing cap8, not a changed threshold.',
        coordinates='Saved predicted query/line/intersection and certified source POS wire targets; not independently measured real R,t.',
        new_line_fit=0,new_PnP=0,new_RGB=0,new_model=0))
    ready={r['id']:r for r in rows(inputs['source_ready']) if r['partition']=='calibration'}
    frames={(r['arm'],r['id']):r for r in rows(inputs['source_decoded'])}
    selected=[r for r in rows(inputs['audited_corners']) if r['arm']=='IMAGE_ROLE' and r['decoder_selected'] and r['saved_corner_error_px']>8]
    assert len(selected)==4
    fig,axes=plt.subplots(2,4,figsize=(17.6,9.2),dpi=160)
    for col,r in enumerate(selected):
        frame=frames[(r['arm'],r['id'])];ls={l['edge']:l for l in frame['lines']}
        source=ready[r['id']];p=r['predicted_xy'];g=r['reference_xy'];rad=r['recorded_radius_px']
        allpts=[g,p]
        for edge in r['edges']:
            l=ls[edge];allpts.extend(l['support_points'])
            allpts.extend(source['queries'][q]['actual_target_uv'] for q in l['queries'])
        mins=[min(v[j] for v in allpts) for j in (0,1)];maxs=[max(v[j] for v in allpts) for j in (0,1)]
        span=max(maxs[j]-mins[j] for j in (0,1));center=[(mins[j]+maxs[j])/2 for j in (0,1)]
        overview=[(v-.56*span-4,v+.56*span+4) for v in center]
        midpoint=[(p[j]+g[j])/2 for j in (0,1)];zoom=[(v-35,v+35) for v in midpoint]
        for row,lim in enumerate((overview,zoom)):
            ax=axes[row,col]
            for edge,color in zip(r['edges'],('#006ba4','#ff800e')):
                l=ls[edge];draw_line(ax,l,color,lim)
                targets=[source['queries'][q]['actual_target_uv'] for q in l['queries']]
                ax.scatter(*zip(*l['support_points']),s=19,c=color,marker='o',label='predicted query' if edge==r['edges'][0] else None)
                ax.scatter(*zip(*targets),s=32,c=color,marker='+',label='certified POS target' if edge==r['edges'][0] else None)
            ax.add_patch(Circle(p,rad,fill=False,color='#a020f0',lw=1.1,label='calibrated radius' if row==1 else None))
            ax.scatter(*p,s=62,c='#b60000',marker='x',lw=2,label='predicted intersection')
            ax.scatter(*g,s=58,c='#006400',marker='*',label='saved wire reference')
            ax.plot([p[0],g[0]],[p[1],g[1]],'k:',lw=1)
            ax.set_xlim(*lim[0]);ax.set_ylim(lim[1][1],lim[1][0]);ax.set_aspect('equal');ax.grid(alpha=.2)
            ax.set_xlabel('image x (px)');ax.set_ylabel('image y (px)')
            if row==0:
                status='cap8 REJECTED' if not r['existing_point_admission_accepted'] else 'cap8 admitted; LOO UNKNOWN'
                ax.set_title(f"{r['id']} k{r['corner']}\n{status}",fontsize=9)
            else:
                ax.set_title(f"error {r['saved_corner_error_px']:.3f}px / radius {rad:.3f}px\nnormal condition {r['normal_matrix_condition']:.3f}",fontsize=9)
    handles,labels=axes[1,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=5,fontsize=9)
    fig.suptitle('All four source decoder-selected >8px cases: saved boundaries and corner construction\nTop: retained query support. Bottom: 70px corner zoom. No new fit; source physical pose and native N3/H/LOO unverified.',fontsize=12)
    fig.subplots_adjust(top=.86,bottom=.1,hspace=.42,wspace=.32)
    p=out/'ALL4_SOURCE_BOUNDARY_AND_CORNER.png';fig.savefig(p);plt.close(fig)
    assert before=={k:dict(sha256=sha(p),bytes=p.stat().st_size) for k,p in inputs.items()}
    save(out/'CHECKS.json',dict(passed=True,cases=4,all_four_cases_included=True,inputs_unchanged=True,
        image_sha256=sha(p),image_bytes=p.stat().st_size,matplotlib=matplotlib.__version__,
        new_line_fit=0,new_PnP=0,new_RGB=0))
    print(json.dumps(dict(passed=True,cases=4,image_bytes=p.stat().st_size)))


if __name__=='__main__':main()
