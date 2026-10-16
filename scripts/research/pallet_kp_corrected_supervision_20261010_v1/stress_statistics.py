"""Saved oracle stress reaggregation; no pose calls or additional hypotheses."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .statistics import binding,distribution,summary,write

NAME='pallet_kp_corrected_supervision_20261010_v1'
def read(p):return json.loads(p.read_text())
def rows(p):
    with gzip.open(p,'rt')as f:return [json.loads(l)for l in f]
def aggregate(rr):
    def quality(r):
        delta=r['delta_vs_corresponding_simple'];t=delta['translation_cm'];rot=delta['rotation_deg']
        if not r['pose']['available']:return 'no_pose'
        if t is None or rot is None:return 'no_common_comparator'
        if t < -1e-9 and rot < -1e-9:return 'both_improved'
        if t > 1e-9 and rot > 1e-9:return 'both_worsened'
        return 'mixed_or_equal'
    def s(rr):
        return dict(pose=summary(rr),correct_pool_count_histogram=dict(Counter(str(r['correct_pool_count'])for r in rr)),
            direct_correct_pool_count_histogram=dict(Counter(str(len(r['remaining_direct_correct_ids']))for r in rr)),
            correct_final_inlier_count_histogram=dict(Counter(str(len(r['correct_final_inlier_ids']))for r in rr)),
            wrong_final_inlier_count_histogram=dict(Counter(str(len(r['wrong_final_inlier_ids']))for r in rr)),
            correct_pool_registry_rank_counts=dict(Counter(str(r['correct_pool_registry_shape']['numerical_rank'])for r in rr)),
            correct_pool_image_rank_counts=dict(Counter(str(r['correct_pool_image_shape']['numerical_rank'])for r in rr)),
            pose_quality_operational=dict(Counter(quality(r)for r in rr)),
            pose_quality_new_only=dict(Counter(quality(r)for r in rr if r['new_pose_estimated'])),
            mean_correct_pool=float(np.mean([r['correct_pool_count']for r in rr]))if rr else None,
            mean_wrong_pool=float(np.mean([len(r['wrong_pool_ids'])for r in rr]))if rr else None,
            mean_correct_final_inliers=float(np.mean([len(r['correct_final_inlier_ids'])for r in rr]))if rr else None,
            mean_wrong_final_inliers=float(np.mean([len(r['wrong_final_inlier_ids'])for r in rr]))if rr else None,
            correct_pool_ge4=sum(r['correct_pool_count']>=4 for r in rr))
    return dict(rows=len(rr),transformation_possible=sum(r['transformation_possible']for r in rr),
        nontransformable=sum(not r['transformation_possible']for r in rr),
        nontransformable_reasons=dict(Counter(json.dumps(r['nontransformable_reason'],sort_keys=True,separators=(',',':'))
                                              for r in rr if not r['transformation_possible'])),
        all_operational=s(rr),transformable_only=s([r for r in rr if r['transformation_possible']]),
        nontransformable_anchor_retained=s([r for r in rr if not r['transformation_possible']]),
        mask_groups={kind:s([r for r in rr if r['mask_wrong_on_known']==wrong])
                     for kind,wrong in [('wrong_known_mask',True),('matching_known_mask',False)]})

def chart(book,doc):
    conditions=book['conditions'];coordinates=book['coordinates'];families=book['families']
    labels=['Anchor','Drop1 good','Drop2 good','Retain1 bad','Retain2 bad','Drop1 + retain1','Drop2 + retain2']
    fig,axes=plt.subplots(2,3,figsize=(17,10))
    for i,family in enumerate(families):
        for coordinate,color,dx in [('BASE','#a77843',-.09),('N3_SUBPIX','#4383ad',.09)]:
            groups=[book['strata']['combined'][coordinate+'::'+family+'::'+c]['transformable_only']for c in conditions]
            x=np.arange(len(conditions))+dx
            for ax,metric in zip(axes[i,:2],('translation_cm','rotation_deg')):
                means=[g['pose']['metrics']['operational'][metric]['mean']for g in groups]
                p90=[g['pose']['metrics']['operational'][metric]['P90']for g in groups]
                ax.plot(x,means,'o-',c=color,label=coordinate+' mean');ax.plot(x,p90,'x--',c=color,label=coordinate+' P90')
                ax.set_yscale('symlog',linthresh=1);ax.grid(axis='y',alpha=.2)
            count=[g['pose']['new_pose_estimated']for g in groups]
            eligible=[g['pose']['denominator']for g in groups]
            axes[i,2].bar(x,[100*n/d if d else 0 for n,d in zip(count,eligible)],.18,color=color,label=coordinate)
            for j,(n,d)in enumerate(zip(count,eligible)):
                axes[i,2].text(x[j],min(101,100*n/d+2 if d else 2),f'{n}/{d}',ha='center',fontsize=7,rotation=90)
        axes[i,0].set_ylabel(family+'\nPosition error (cm; symlog)');axes[i,1].set_ylabel('Rotation error (degree; symlog)')
        axes[i,2].set_ylabel('New pose / transformable frames (%)');axes[i,2].set_ylim(0,116)
        for ax in axes[i]:ax.set_xticks(range(len(conditions)),labels,rotation=28,ha='right')
        axes[i,0].legend(fontsize=8,ncol=2);axes[i,2].legend(fontsize=9)
    fig.suptitle('09 | Actual245-frame bounded mask stress: oracle diagnostics, no deployment claim',y=1.01)
    fig.text(.5,-.025,'Only condition-specific transformable subsets shown; all6860 rows over245 frames, including1799 nontransformable anchor rows, remain summarized separately.\nHUMAN_NOSELF retainedSELF can be proxy-accurate; KNOWN_ACCURATE uses fixed8px proxy-good DIRECT versus known>8px bad points.\nSame coordinate/K/dimension/4-ID hypotheses reused over14 masks per frame/coordinate. New pose alone is not reference-accurate success.',ha='center',fontsize=9)
    fig.tight_layout();p=doc/'figures/09_real_mask_stress.png';fig.savefig(p,dpi=120,bbox_inches='tight',facecolor='white');plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3]);a=p.parse_args()
    root=a.root.resolve();doc=root/'_docs/experiments'/NAME;cohort=read(doc/'COHORT.json');allowed=set(cohort['ids'])
    labels={r['id']:r['label']for r in cohort['frames']};protocol=read(doc/'REAL_STRESS_PROTOCOL.json')
    path=doc/'REAL_STRESS_ROWS.jsonl.gz';raw=rows(path);execution=read(doc/'REAL_STRESS_EXECUTION.json')
    assert execution['complete']and len(raw)==6860 and len({r['method']for r in raw})==28
    assert all(r['id']in allowed and r['oracle_diagnostic_not_deployment']for r in raw)
    assert all(not r['initial_coordinates_changed']and not r['GT_pose_or_coordinates_input_to_solver']for r in raw)
    methods=sorted({r['method']for r in raw})
    for method in methods:assert {r['id']for r in raw if r['method']==method}==allowed
    scopes={'combined':cohort['ids'],'easy':[i for i in cohort['ids']if labels[i]=='clean'],
            'medium':[i for i in cohort['ids']if labels[i]=='moderate']}
    book=dict(schema='actual_eligible_real_mask_stress_saved_statistics_v1',complete=True,frames=245,rows=6860,
        conditions=[r['name']for r in protocol['conditions']],coordinates=protocol['coordinates'],families=protocol['families'],
        strata={s:{m:aggregate([r for r in raw if r['id']in set(ii)and r['method']==m])for m in methods}for s,ii in scopes.items()},
        nontransformable_total=sum(not r['transformation_possible']for r in raw),
        family_nontransformable={f:sum(not r['transformation_possible']for r in raw if r['family']==f)for f in protocol['families']},
        bindings=[binding(p,root)for p in (path,doc/'REAL_STRESS_EXECUTION.json',doc/'REAL_STRESS_PROTOCOL.json',doc/'COHORT.json')],
        new_statistical_pose_calls=0,oracle_diagnostic_not_deployment=True,geometric_proxy_not_independent_physical_GT=True,
        absent_transformations_not_counted_as_solver_failures=True)
    write(doc/'REAL_STRESS_STATISTICS.json',book);chart(book,doc)
    print(json.dumps(dict(complete=True,rows=6860,nontransformable=book['nontransformable_total'],new_fits=0)))

if __name__=='__main__':main()
