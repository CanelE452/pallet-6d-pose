"""Render the recorded source-only corrected-supervision proposal counts.

Only READ_ONLY_WIRE_REVIEW.json is consumed. No RGB, mesh, model, ray, target
cache, pose solver or training component is loaded.
"""
import argparse
import hashlib
import json
from pathlib import Path


def binding(path, root):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(root)) if path.is_relative_to(root) else path.name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3])
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    root=args.root.resolve()
    doc=root/'_docs/experiments/pallet_kp_supervision_gate_20261010_v1'
    output=(args.output_dir or doc).resolve()
    figure_path=output/'figures/02_corrected_supervision.png'
    receipt_path=output/'CORRECTED_SUPERVISION_FIGURE_BINDING.json'
    assert not figure_path.exists() and not receipt_path.exists(),'Preserve completed review figures'
    source_path=doc/'READ_ONLY_WIRE_REVIEW.json'
    review=json.loads(source_path.read_text())
    assert review['schema']=='independent_read_only_actual_wire_review_v1' and review['complete']
    assert review['passed_narrow_physical_correspondence_claim']
    new=review['target_cache_compatibility']['source_test_target_counts_if_proposal_applied']
    changed=review['counts']['proposed_label_changed']
    old=dict(POSITIVE=review['counts']['original_positive_to_corrected_physical_positive'],
             NONE=new['NONE']+changed,IGNORE=new['IGNORE'])
    graph=review['ideal_graph'];families=graph['families'];queries=families*84
    assert families==128 and sum(old.values())==sum(new.values())==queries==10752
    assert old==dict(POSITIVE=2333,NONE=4414,IGNORE=4005)
    assert new==dict(POSITIVE=4497,NONE=2250,IGNORE=4005)
    graph_old=[graph['old_raw_ge4_corners'],graph['old_in_frame_ge4_corners']]
    graph_new=[graph['corrected_raw_ge4_corners'],graph['corrected_in_frame_ge4_corners']]
    assert graph_old==[13,12] and graph_new==[110,95]

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(13.4,7.))
    axis=axes[0];bottom=np.zeros(2)
    colors={'POSITIVE':'#16a34a','NONE':'#64748b','IGNORE':'#cbd5e1'}
    for category in ('POSITIVE','NONE','IGNORE'):
        values=np.array([old[category],new[category]])
        axis.bar([0,1],values,width=.6,bottom=bottom,color=colors[category],label=category)
        for x,value,base in zip([0,1],values,bottom):
            axis.text(x,base+value/2,f'{value:,}',ha='center',va='center',fontsize=12,
                      color='black' if category=='IGNORE' else 'white',fontweight='bold')
        bottom+=values
    axis.set_xticks([0,1],['Original supplied targets','Owned-wire proposal'])
    axis.set_ylabel('Query count');axis.set_ylim(0,queries*1.06)
    axis.set_title('Partial proposal: 6,463 of 10,752 queries audited',fontsize=12)
    axis.legend(loc='upper center',bbox_to_anchor=(.5,-.105),ncol=3,frameon=False)
    axis.grid(axis='y',alpha=.15);axis.set_axisbelow(True)
    for x in [0,1]:axis.text(x,queries+110,f'n={queries:,}',ha='center',fontsize=10)

    axis=axes[1];x=np.arange(2)
    axis.bar(x-.18,graph_old,width=.35,color='#94a3b8',label='Original ideal target graph')
    axis.bar(x+.18,graph_new,width=.35,color='#2563eb',label='Proposed ideal target graph')
    for xx,values in [(x-.18,graph_old),(x+.18,graph_new)]:
        for pos,value in zip(xx,values):axis.text(pos,value+2,str(value),ha='center',fontsize=12,fontweight='bold')
    axis.axhline(families,color='#64748b',ls='--',lw=1)
    axis.text(1.42,families+1,'128 families',ha='right',va='bottom',fontsize=9,color='#475569')
    axis.set_xticks(x,['Raw decoded graph','In-frame decoded graph'])
    axis.set_ylim(0,145);axis.set_ylabel('Families with ≥4 decoded corners')
    axis.set_title('Ideal observation upper bound — no learned or pose result',fontsize=12)
    axis.legend(loc='upper center',bbox_to_anchor=(.5,-.105),frameon=False,fontsize=10)
    axis.grid(axis='y',alpha=.15);axis.set_axisbelow(True)

    fig.suptitle('Source-test 128: physical-wire supervision proposal, original cache preserved',fontsize=15,y=.98)
    fig.text(.035,.06,'Proposed NONE = 1,966 certified front-surface cases + 284 other NONE unchanged and unvalidated.\n'
             'This partial proposal is separate from full prepared targets, which IGNORE the 284 unresolved queries.\n'
             'Original approximate own-mesh depth tolerance + unchanged supplied visible-mask kernel; no source-target mutation or new learning.\n'
             'Ideal graph counts do not establish independent corners, unique pose, or real-image improvement.',
             fontsize=10,ha='left',va='bottom')
    fig.tight_layout(rect=[0,.20,1,.94])
    figure_path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(figure_path,dpi=150);plt.close(fig)
    receipt=dict(schema='corrected_source_supervision_scientific_figure_v1',complete=True,
        file=binding(figure_path,root),source=binding(source_path,root),code=binding(Path(__file__),root),
        plotted_values=dict(source_test_families=families,queries=queries,original=old,proposed=new,
            original_ideal_raw_inframe_ge4=graph_old,proposed_ideal_raw_inframe_ge4=graph_new,
            audited_queries=6463,proposed_NONE_certified=1966,other_NONE_unchanged_unvalidated=284),
        presentation_only=True,new_numeric_inference_or_fitting=False,new_source_target_mutations=0,
        new_models_rays_training_PnP_RGB_reads=0,
        interpretation='Partial fixed source-test proposal: only6463queries audited;284 otherNONE unchanged/unvalidated, separated from full prepared targets that IGNORE these284. Original approximate ownmesh depth and suppliedmask policy. Ideal target graph upper bound, not trained prediction, independent corner proof, pose success or fulltrain supervision readiness.')
    with receipt_path.open('x') as stream:json.dump(receipt,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
