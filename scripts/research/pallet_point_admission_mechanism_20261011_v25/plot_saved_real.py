"""Plot all 133 saved adopted observations; no outcome selection or inference."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    p=ROOT/'real_loo/ROWS.jsonl.gz';source=p.read_bytes();h=hashlib.sha256(source).hexdigest()
    out=ROOT/'plot';out.mkdir()
    (out/'PROTOCOL.json').write_text(json.dumps(dict(input_sha256=h,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        selected_set='all existing133 adopted corners, all85 frames, all outcomes',
        physical_truth='existing annotation pixel proxy',new_inference=0,new_PnP=0),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[json.loads(s) for s in gzip.decompress(source).decode().splitlines()];assert len(rows)==133
    fig,axes=plt.subplots(1,2,figsize=(12,5.5))
    for quality,color,label in [('GOOD','#338e69','Within 8 px proxy (120)'),('BAD','#c95043','Over 8 px proxy (13)')]:
        rr=[r for r in rows if r['candidate_proxy_quality']==quality]
        axes[0].scatter([r['saved_candidate_LOO_residual_px'] for r in rr],
            [r['selected_point']['candidate_error_px'] for r in rr],c=color,s=24,alpha=.8,label=label)
        axes[1].scatter([r['selected_point']['N3_error_px'] for r in rr],
            [r['selected_point']['candidate_error_px'] for r in rr],c=color,s=24,alpha=.8,label=label)
    all_good_bad=[r for r in rows if r['candidate_proxy_quality']=='BAD' and not r['LOO_fit_quality_ids']['BAD'] and not r['LOO_fit_quality_ids']['UNKNOWN']]
    assert len(all_good_bad)==3
    for r in all_good_bad:
        axes[0].scatter(r['saved_candidate_LOO_residual_px'],r['selected_point']['candidate_error_px'],
                        s=120,facecolors='none',edgecolors='#322f78',linewidth=1.5)
    axes[0].axvline(8,color='black',ls='--',lw=1);axes[0].axhline(8,color='black',ls=':',lw=1)
    axes[0].set_xlim(-.1,8.5);axes[0].set_yscale('log')
    axes[0].set_xlabel('Saved candidate vs held-out pose residual (px)')
    axes[0].set_ylabel('Candidate vs annotation reference error (px)')
    axes[0].set_title('LOO acceptance does not certify point correctness')
    maxerr=max(r['selected_point']['candidate_error_px'] for r in rows)
    maxnative=max(r['selected_point']['N3_error_px'] for r in rows)
    upper=max(maxerr,maxnative)*1.07
    axes[1].plot([.1,upper],[.1,upper],color='black',ls='--',lw=1)
    axes[1].set_xscale('log');axes[1].set_yscale('log')
    axes[1].set_xlabel('Native N3 vs annotation reference error (px)')
    axes[1].set_ylabel('Adopted boundary vs reference error (px)')
    axes[1].set_title('80 improve coordinate error; 53 worsen it')
    for ax in axes:ax.grid(True,which='major',alpha=.2);ax.legend(fontsize=8,loc='lower right')
    fig.suptitle('All 133 actually adopted real corner observations',fontsize=14)
    fig.text(.5,.015,'Purple rings: the 3 bad candidates whose LOO fit uses only known-within8px native points.\n'
             'Saved DEV pixel proxy; no new LOO, PnP or pose evaluation. Physical ownership remains unknown.',
             ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.095,1,.94));image=out/'ALL133_SAVED_ADOPTED_POINTS.png';fig.savefig(image,dpi=180);plt.close(fig)
    assert h==hashlib.sha256(p.read_bytes()).hexdigest()
    (out/'CHECKS.json').write_text(json.dumps(dict(passed=True,rows=133,all_outcomes=True,
        input_unchanged=True,image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
        image_bytes=image.stat().st_size,matplotlib=matplotlib.__version__,new_inference=0,new_PnP=0),indent=2)+'\n')
    print(json.dumps(dict(passed=True,rows=133,image_bytes=image.stat().st_size)))


if __name__=='__main__':
    main()
