"""Plot all 24 recorded cases with >=4 annotation-proxy GOOD points."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    inp = ROOT / 'ROWS.jsonl'
    bindings = {str(p): sha(p) for p in (inp, Path(__file__))}
    protocol = dict(schema='saved_all24_cause_plot_v23', inputs=bindings,
        selection='all rows in rejected59 with saved used_proxy_good_count>=4; sorted ID',
        outcome_subset_selected=False, errors_copied_from_saved_rows=True,
        proxy_truth_is_not_independent_physical_pose=True, new_PnP=0,
        new_projection=0, new_pose_accuracy_evaluations=0, new_moments_or_CIs=0)
    (ROOT / 'PLOT_PROTOCOL.json').write_text(json.dumps(protocol, indent=2) + '\n')
    rows = sorted((json.loads(l) for l in inp.read_text().splitlines()), key=lambda r:r['id'])
    rows = [r for r in rows if r['used_proxy_good_count'] >= 4]
    assert len(rows) == 24
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(17, 15), sharey=True, layout='constrained')
    y = list(range(24))
    methods = ('N3_SUBPIX', 'HYBRID_POINT_CONSENSUS_WHOLE_U_CLOSURE')
    for ax, key, unit in zip(axes, ('translation_cm','rotation_deg'), ('cm','degree')):
        for method, marker, color, label in zip(methods, ('o','x'), ('#617a9c','#cb493f'), ('Fixed N3 + SubPix','V21 actual selected output')):
            x = [r['saved_errors'][method][key] for r in rows]
            ax.scatter(x, y, marker=marker, color=color, label=label, s=40)
        for i,r in enumerate(rows):
            values = [r['saved_errors'][m][key] for m in methods]
            ax.plot(values,[i,i],color='#aaaaaa',linewidth=.8,zorder=0)
        ax.set_xscale('symlog',linthresh=1)
        ax.set_xlabel(key+' ['+unit+']; symlog axis')
        ax.grid(alpha=.2)
        ax.legend(loc='lower right',fontsize=9)
    labels = [r['id']+' | goodU='+str(r['used_proxy_good_count'])+' | boundary='+str(len(r['selected_boundary_corner_ids'])) for r in rows]
    axes[0].set_yticks(y, labels,fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle('All24 cases: whole-U standard candidates rejected, >=4 saved proxy-GOOD observations\nGOOD = existing annotation error <=8px; it does not prove physical ownership, exact correspondence, or pose observability.\nEvery outcome included. These 24 cases are a diagnosis subset of the full245 evaluation; no new fits or scores.',fontsize=12)
    png=ROOT/'ALL24_SAVED_ERRORS.png'; svg=ROOT/'ALL24_SAVED_ERRORS.svg'
    fig.savefig(png,dpi=140);fig.savefig(svg);plt.close(fig)
    assert bindings == {p:sha(Path(p)) for p in bindings}
    check = dict(passed=True, cases=24, IDs=[r['id'] for r in rows],
        input_bindings=bindings, PNG_sha256=sha(png), SVG_sha256=sha(svg),
        new_PnP=0, new_projection=0, new_pose_accuracy_evaluations=0,
        visually_reviewed=False)
    (ROOT/'PLOT_CHECKS.json').write_text(json.dumps(check,indent=2)+'\n')
    print(json.dumps(dict(complete=True,cases=24,PNG_bytes=png.stat().st_size,SVG_bytes=svg.stat().st_size)))


if __name__ == '__main__':
    main()
