"""Plot every saved conditional candidate; never fit or infer."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    rows_path=OUT/'ROWS.jsonl'
    protocol=dict(schema='frozen_saved_candidate_plot_v24',rows_sha256=sha(rows_path),
        code_sha256=sha(Path(__file__)),all_rows=573,selection='All rows, all outcomes, all three arms.',
        new_inference=0,new_pose_scoring=0,thresholds_changed=False)
    (OUT/'PLOT_PROTOCOL.json').write_text(json.dumps(protocol,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[json.loads(s) for s in rows_path.read_text().splitlines()]
    fig,axs=plt.subplots(1,3,figsize=(14,5),sharey=True)
    for ax,arm in zip(axs,('GEOMETRY_ONLY','IMAGE_NO_ROLE','IMAGE_ROLE')):
        rr=[r for r in rows if r['arm']==arm]
        for key,color,label in (
                ('NECESSARY_GATES_PASS_FINAL_UNKNOWN','#237aa6','Necessary gates pass; final unknown'),
                ('DEFINITELY_REJECTED_RADIUS','#a4a4a4','Rejected by existing radius gate'),
                ('DEFINITELY_REJECTED_IMAGE','#d34e3e','Rejected by image bounds')):
            ss=[r for r in rr if r['necessary_gate_state']==key]
            if ss:ax.scatter([r['normalized_uncertainty'] for r in ss],
                             [r['error_px'] for r in ss],s=20,c=color,alpha=.72,label=f'{label} ({len(ss)})')
        ax.axvline(1,c='black',ls='--',lw=1)
        ax.axhline(8,c='#66588e',ls=':',lw=1)
        ax.set_xscale('log');ax.set_yscale('log')
        ax.set_title(f'{arm}: {len(rr)} conditional candidates')
        ax.set_xlabel('Recorded radius / initial query spacing')
        ax.grid(True,which='major',alpha=.2)
        ax.legend(fontsize=7,loc='upper left')
    axs[0].set_ylabel('Saved candidate vs actual-wire reference error (px)')
    fig.suptitle('Saved source CAL candidates before final admission',fontsize=14)
    fig.text(.5,.01,'Dashed x=1: existing admission gate. Dotted y=8 px: descriptive error boundary only.\n'
             'Passing two necessary gates is not final acceptance. No new inference, line fit, PnP or real pose evaluation.',
             ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.09,1,.94))
    image=OUT/'ALL573_SAVED_CANDIDATES.png'
    fig.savefig(image,dpi=180);plt.close(fig)
    assert sha(rows_path)==protocol['rows_sha256']
    (OUT/'PLOT_CHECKS.json').write_text(json.dumps(dict(passed=True,all_rows_plotted=len(rows),
        rows_unchanged=True,image_sha256=sha(image),image_bytes=image.stat().st_size,
        matplotlib_version=matplotlib.__version__,new_inference=0,new_pose_scoring=0),indent=2)+'\n')
    print(json.dumps(dict(passed=True,all_rows_plotted=len(rows),image_bytes=image.stat().st_size)))


if __name__=='__main__':
    main()
