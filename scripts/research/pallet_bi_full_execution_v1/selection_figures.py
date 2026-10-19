"""Standalone scientific figure from saved, posthoc action-rank diagnostics."""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import s1_union as U


def run(diagnostic,output,arm='S1_UNION',cohort='PF_new_good'):
    diagnostic,output=map(Path,(diagnostic,output));seal=U.read(diagnostic/'SELECTION_DIAGNOSTIC_SEAL.json')
    if not seal['complete']:raise ValueError('Complete rank diagnosis required')
    for name,digest in seal['artifacts'].items():
        if U.sha(diagnostic/name)!=digest:raise ValueError('Saved diagnostic changed')
    data=U.read(diagnostic/'FIXED_COHORT_ACTIONS.json')[arm][cohort]
    ids=sorted({r['id']for r in data});seeds=(20261018,20261019,20261020)
    figure,axes=plt.subplots(1,len(ids),figsize=(4.2*len(ids),4.4),squeeze=False,sharey=True)
    for ax,fid in zip(axes.flat,ids):
        rows={r['seed']:r for r in data if r['id']==fid};ranks=[rows[s]['best_good_action_rank']for s in seeds]
        ax.plot(range(3),ranks,'o-',color='#df7b18',lw=1.7,markersize=7)
        ax.axhline(1,color='#168647',ls='--',lw=1,label='selected top rank')
        for i,value in enumerate(ranks):
            if value is not None:ax.annotate(str(value),(i,value),textcoords='offset points',xytext=(0,-15),ha='center')
        ax.set_xticks(range(3),[str(s)for s in seeds],rotation=15);ax.set_xlabel('H1 initialization seed')
        ax.set_title(fid,fontsize=10);ax.grid(axis='y',alpha=.25)
        ax.set_ylim(max(v for r in data for v in [r['best_good_action_rank']]if v is not None)+3,0)
    axes[0,0].set_ylabel('Best actually successful action rank (smaller is better)')
    figure.suptitle('Three newly available successful poses: admitted, then ranked below the selected action',fontsize=12)
    figure.text(.5,.015,'PRIMARY development data. Every successful pose passed admission; selected outputs fail in all three seeds.\nRanks use saved logits only; logits are not calibrated probabilities. No new forward, fit, or pose solve.',ha='center',fontsize=9)
    figure.tight_layout(rect=(0,.11,1,.93));output.mkdir(parents=True,exist_ok=True)
    for extension in ('png','svg'):figure.savefig(output/('NEW_GOOD_ACTION_RANKS.'+extension),dpi=160)
    plt.close(figure)
    U.write(output/'FIGURE_SOURCE.json',dict(arm=arm,cohort=cohort,cases=[dict(id=fid,seeds=[{k:r[k]for k in ('seed','best_good_action_rank','best_good_logit_minus_selected','selected_candidate_id')}for r in data if r['id']==fid])for fid in ids],
        diagnostic_seal_sha256=U.sha(diagnostic/'SELECTION_DIAGNOSTIC_SEAL.json'),source_sha256=U.sha(__file__),new_forward_or_fit_or_solve=0,
        artifacts={p.name:U.sha(p)for p in output.iterdir()if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--diagnostic',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.diagnostic,a.output)
