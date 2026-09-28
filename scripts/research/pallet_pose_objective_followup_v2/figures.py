"""CPU-only measured pose plots and already-public RGB overlays.

All outputs remain in V2. Posthoc example selection never changes predictions,
pose candidates, targets, recipe selection, or either scored denominator.
"""
from __future__ import annotations
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-objective-v2-mpl')
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from . import common as C
from .metric_baseline import PRIMARY, SEVERITIES

TOL=1e-7 # Existing metric-lock numerical parity tolerance, not practical significance.
COLORS=['#26828e','#d8802e','#588f46','#785ca6','#bb5f7e','#617eac','#826344']
BASE_COLORS={'R0':'#333333','OLD_RAW':'#a47c36','OLD_REF':'#7853a2','SYN':'#888888'}
REPLAY_CYCLES={'BASELINE_REPEAT','RECIPE_REPEAT'}
SOURCES=set(); FIGURES=[]; EXAMPLES=[]


def source(path):
    path=Path(path); SOURCES.add(path); return C.read(path)


def register_retained():
    """Register committed A/B examples without rendering or selecting anything."""
    path=C.DOC/'RETAINED_FIGURE_MANIFEST.json'
    if not path.exists():return None
    previous=source(path)
    assert previous['role']=='RETAINED_HISTORICAL_AB_EXAMPLES_NOT_FINAL_SELECTION'
    assert len(previous['figures'])==len(previous['examples'])==4
    current_figures={r['file']['path']:r for r in FIGURES}
    current_examples={r['figure']['path'] for r in EXAMPLES if r.get('figure')}
    for row in previous['figures']:
        C.verify(row['file'])
        if row['file']['path'] in current_figures:
            assert row['file']==current_figures[row['file']['path']]['file']
            continue
        FIGURES.append(dict(row,historical_role=previous['role'],retained_manifest=C.bind(path)))
    for row in previous['examples']:
        C.verify(row['figure']);C.verify(row['image']);C.verify(row['previous_publication'])
        approval=C.read(C.ROOT/row['previous_publication']['path'])
        assert row['id'] in {e.get('id',e.get('frame_id')) for e in approval['examples']}
        assert row['native_image_bounds_clipped'] and row['selected_recipe'] is False
        if row['figure']['path'] not in current_examples:
            EXAMPLES.append(dict(row,retained_historical=True,historical_role=previous['role'],retained_manifest=C.bind(path)))
    return C.bind(path)


def register_retained_only():
    """Repair registration only: no PNG bytes, plot values, or choices change."""
    SOURCES.clear();FIGURES.clear();EXAMPLES.clear()
    path=C.DOC/'FIGURE_MANIFEST.json';value=C.read(path)
    FIGURES.extend(value['figures']);EXAMPLES.extend(value['examples'])
    retained=register_retained();assert retained is not None
    value.update(figures=FIGURES,examples=EXAMPLES,retained_historical_manifest=retained,
        retained_historical_figures=sum('historical_role' in f for f in FIGURES),
        total_png_bytes=sum(f['file']['bytes'] for f in FIGURES))
    bindings={b['path']:b for b in value['sources']}
    for file in [Path(__file__),C.ROOT/retained['path']]:bindings[C.bind(file)['path']]=C.bind(file)
    value['sources']=[bindings[k] for k in sorted(bindings)]
    C.save(path,value)
    print('RETAINED_PUBLIC_FIGURES_REGISTERED',value['retained_historical_figures'],flush=True)


def direction(dt,dr):
    if abs(dt)<=TOL and abs(dr)<=TOL:return 'unchanged'
    if dt<-TOL and dr<-TOL:return 'improved'
    if dt>TOL and dr>TOL:return 'worsened'
    return 'tradeoff_or_single_axis'


def select_examples(allowed,ids,before,after):
    """Same strict two-axis rule in both directions; no cherry-picked fallback."""
    candidates=[]
    for fid in sorted(set(allowed)&set(ids)):
        if not before[fid]['available'] or not after[fid]['available']:continue
        dt=after[fid]['translation_cm']-before[fid]['translation_cm']
        dr=after[fid]['rotation_deg']-before[fid]['rotation_deg']
        candidates.append(dict(id=fid,delta_translation_cm=dt,delta_rotation_deg=dr,category=direction(dt,dr)))
    result=[]
    for category in ('improved','worsened','unchanged'):
        eligible=[r for r in candidates if r['category']==category]
        # Symmetric ordering: largest abs translation change, then rotation, then ID.
        eligible=sorted(eligible,key=lambda r:(-abs(r['delta_translation_cm']),-abs(r['delta_rotation_deg']),r['id']))
        if eligible:result.append(dict(status='AVAILABLE',**eligible[0]))
        else:result.append(dict(status='NA',category=category,reason='No already-public, common-valid frame in the requested population meets the same two-axis rule.'))
    return result,Counter(r['category'] for r in candidates)


def save(fig,name,caption,population=None):
    folder=C.DOC/'figures';folder.mkdir(parents=True,exist_ok=True)
    path=folder/(name+'.png');assert path.is_relative_to(C.DOC)
    fig.savefig(path,dpi=145,bbox_inches='tight',facecolor='white');plt.close(fig)
    with Image.open(path) as im:size=list(im.size)
    FIGURES.append(dict(name=name,file=C.bind(path),pixels=size,caption=caption,population=population))
    return path


def read_cycles():
    cycles=[]
    for path in sorted((C.DOC/'cycles').glob('*/RESULTS_*_S*.json')):
        result=source(path)
        if not {'cycle','material','seed','groups'}<=set(result):continue
        for binding in result['private_artifacts']:C.verify(binding)
        C.verify(result['prediction_lock'])
        raw=C.RAW/'cycles'/result['cycle'];tag=f"{result['material']}_S{result['seed']}"
        metrics=source(raw/f'POSE_METRICS_{tag}.json')
        meta=source(raw/f'METADATA_{tag}.json')
        expected=128 if result['material']=='PLASTIC' else 45
        assert len(meta)==expected and all(len(metrics[a])==expected for a in metrics)
        label=f"{result['cycle'].split('_')[0]} S{result['seed']}"
        if result['cycle'] in REPLAY_CYCLES:label+=' [deterministic replay]'
        cycles.append(dict(result=result,path=path,raw=raw,tag=tag,metrics=metrics,metadata={r['id']:r for r in meta},
            label=label,independent_replication=False if result['cycle'] in REPLAY_CYCLES else None,
            slug=f"{result['cycle'].lower()}_{result['material'].lower()}_s{result['seed']}"))
    return cycles


def metric(row,key,quantile='median'):
    return row['conditional'][key][quantile]


def verify_summaries(groups,metrics,members):
    checked=0
    for group,ids in members.items():
        for arm,summary in groups[group].items():
            assert len(ids)==summary['frames']
            valid=[metrics[arm][fid] for fid in ids if metrics[arm][fid]['available']]
            assert len(valid)==summary['valid_pose'] and summary['failed_pose']==len(ids)-len(valid)
            for key in ('translation_cm','rotation_deg'):
                for name,q in [('median',50),('P90',90)]:
                    actual=summary['conditional'][key][name]
                    if valid:assert np.isclose(actual,np.percentile([r[key] for r in valid],q),atol=1e-9,rtol=1e-9)
                    else:assert actual is None
                    checked+=1
    return checked


def objective_plot(baseline,cycles):
    cards=[]
    for arm in ('R0','OLD_RAW','OLD_REF','SYN'):
        row=baseline['materials']['PLASTIC']['groups'][PRIMARY][arm]
        cards.append(dict(label=arm,row=row,color=BASE_COLORS[arm],marker={'R0':'*','OLD_RAW':'s','OLD_REF':'D','SYN':'X'}[arm],fill=True))
    overview=[c for c in cycles if c['result']['material']=='PLASTIC' and c['result']['cycle'] not in REPLAY_CYCLES]
    for i,cycle in enumerate(overview):
        for target in ('RAW','REF'):
            cards.append(dict(label=cycle['label']+' '+target,row=cycle['result']['groups'][PRIMARY]['NEW_'+target],
                color=COLORS[i%len(COLORS)],marker='o' if target=='RAW' else '^',fill=target=='REF'))
    old=cards[2]['row'];fig,axs=plt.subplots(1,2,figsize=(12,4.4))
    for j,card in enumerate(cards):
        row=card['row'];assert row['frames']==99
        t,r=metric(row,'translation_cm'),metric(row,'rotation_deg')
        if t is None or r is None:continue
        label=f"{card['label']} ({row['valid_pose']}/99)"
        for ax,xy in zip(axs,[(t,r),(t-metric(old,'translation_cm'),r-metric(old,'rotation_deg'))]):
            ax.scatter(*xy,s=80,marker=card['marker'],edgecolors=card['color'],
                facecolors=card['color'] if card['fill'] else 'none',linewidths=1.3,label=label,zorder=3)
            offset=(-13,-15) if j==2 else (10,-4) if j>3 and card['label'].endswith('REF') else (4,4)
            ax.annotate(str(j+1),xy,xytext=offset,textcoords='offset points',fontsize=8,color=card['color'])
    axs[0].set(xlabel='Median translation error (cm)',ylabel='Median C2 rotation error (deg)',title='Absolute conditional medians')
    axs[1].axhline(0,color='#bbb',lw=.8);axs[1].axvline(0,color='#bbb',lw=.8)
    axs[1].set(xlabel='Delta median T vs OLD_REF (cm)',ylabel='Delta median R vs OLD_REF (deg)',title='Difference of medians; lower-left improves both')
    for ax in axs:ax.grid(alpha=.17);ax.spines[['top','right']].set_visible(False)
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,[f'{j+1}. '+label for j,label in enumerate(labels)],loc='lower center',ncol=min(4,len(labels)),fontsize=8,bbox_to_anchor=(.5,-.04))
    fig.suptitle('Plastic natural Moderate + Severe: fixed 99 reused-DEV frames',fontsize=12)
    fig.tight_layout(rect=(0,.12,1,.95))
    fig.text(.5,-.09,'Nominal seed43 fits are deterministic replays, not independent replications; duplicate points omitted.',ha='center',va='top',fontsize=8)
    save(fig,'objective99_tr_scatter','Lower T and R are better. Conditional medians with full99 pose coverage in legend; no T/R weighted sum, no independent-test claim. Right panel is difference of medians, not median frame delta. Seed43 has identical streams/state: duplicate reruns omitted from overview; see separate replay plots and REPLICATION_VALIDITY_CORRECTION.',dict(material='PLASTIC',frames=99))


def severity_plot(baseline,cycles):
    for material in ('PLASTIC','WOOD'):
        current=[c for c in cycles if c['result']['material']==material and c['result']['cycle'] not in REPLAY_CYCLES]
        # Wood has no new-cycle claim when no Wood fit exists; still retain baseline context.
        cards=[(a,baseline['materials'][material]['groups'],a,BASE_COLORS[a],'-') for a in ('R0','OLD_REF')]
        for i,c in enumerate(current):
            for target in ('RAW','REF'):
                cards.append((c['label']+' '+target,c['result']['groups'],'NEW_'+target,COLORS[i%len(COLORS)],'--' if target=='RAW' else '-'))
        fig,axs=plt.subplots(2,2,figsize=(11.5,7))
        counts=[baseline['materials'][material]['groups']['severity:'+s]['R0']['frames'] for s in SEVERITIES]
        labels=[f'{s}\nN={n}' for s,n in zip(('Clean','Moderate','Severe'),counts)]
        for ax,key,quantile in [(axs[0,0],'translation_cm','median'),(axs[0,1],'rotation_deg','median'),
                                (axs[1,0],'translation_cm','P90'),(axs[1,1],'rotation_deg','P90')]:
            for label,groups,arm,color,style in cards:
                values=[metric(groups['severity:'+s][arm],key,quantile) for s in SEVERITIES]
                ax.plot(range(3),[np.nan if v is None else v for v in values],marker='o',color=color,ls=style,label=label,lw=1.3,ms=4)
            ax.set_xticks(range(3),labels);ax.set_title(quantile+' '+('T (cm)' if key=='translation_cm' else 'R (deg)'))
            ax.set_ylabel('Conditional error; lower is better');ax.grid(alpha=.16);ax.spines[['top','right']].set_visible(False)
            if counts[-1]==0:ax.text(2,.45,'NA: no frames',transform=ax.get_xaxis_transform(),ha='center',fontsize=8)
        handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=min(5,len(labels)),fontsize=8)
        fig.suptitle(f'{material.title()} severity strata | medians and tails shown separately',fontsize=12)
        fig.tight_layout(rect=(0,.065,1,.96))
        save(fig,material.lower()+'_severity_tr','Natural severity labels fixed before fits. P90 is not median. Missing Wood Severe is NA, not zero. Valid-pose conditional summaries retain full-stratum denominators in bound result JSON. Identical seed43 replay curves omitted; no independent robustness inferred.',dict(material=material,counts=counts))


def paired_delta(cycle,populations):
    material=cycle['result']['material'];group=PRIMARY if material=='PLASTIC' else 'ALL'
    ids=populations[material]['groups'][group]
    fig,axs=plt.subplots(1,3,figsize=(15.5,4.2));counts=[]
    palette={'improved':'#288c72','worsened':'#c45858','unchanged':'#777777','tradeoff_or_single_axis':'#b68c32'}
    for ax,(after_arm,before_arm) in zip(axs,(('NEW_REF','OLD_REF'),('NEW_RAW','OLD_RAW'),('NEW_REF','NEW_RAW'))):
        before=cycle['metrics'][before_arm];after=cycle['metrics'][after_arm]
        common=[fid for fid in ids if before[fid]['available'] and after[fid]['available']]
        rows=[(after[i]['translation_cm']-before[i]['translation_cm'],after[i]['rotation_deg']-before[i]['rotation_deg']) for i in common]
        c=Counter(direction(dt,dr) for dt,dr in rows);counts.append(dict(after=after_arm,before=before_arm,common_valid=len(common),full=len(ids),categories=dict(c)))
        for category,color in palette.items():
            values=np.asarray([(dt,dr) for dt,dr in rows if direction(dt,dr)==category])
            if len(values):ax.scatter(values[:,0],values[:,1],s=16,color=color,alpha=.8,label=f"{category.replace('_',' ')} ({len(values)})")
        ax.axhline(0,color='#999',lw=.8);ax.axvline(0,color='#999',lw=.8)
        from matplotlib.ticker import MaxNLocator
        xs='symlog' if rows and max(abs(r[0]) for r in rows)>1 else 'linear'
        ys='symlog' if rows and max(abs(r[1]) for r in rows)>1 else 'linear'
        if xs=='symlog':ax.set_xscale('symlog',linthresh=1)
        else:ax.xaxis.set_major_locator(MaxNLocator(5))
        if ys=='symlog':ax.set_yscale('symlog',linthresh=1)
        else:ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set(xlabel=f'Per-frame delta T (cm, {xs})',ylabel=f'Per-frame delta R (deg, {ys})',
               title=f"{after_arm} - {before_arm} | paired {len(common)}/{len(ids)}")
        ax.legend(fontsize=7,loc='best');ax.grid(alpha=.12);ax.spines[['top','right']].set_visible(False)
    is_replay=cycle['result']['cycle'] in REPLAY_CYCLES
    note=' | NOT an independent seed confirmation' if is_replay else ' | every common-valid frame; no outlier trimming'
    fig.suptitle(cycle['label']+' '+material.title()+note,fontsize=11)
    if is_replay:fig.text(.5,.005,'Fixed loader streams and all 879 checkpoint tensors match the original run. Numerical replay only; no variance estimate.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.045 if is_replay else 0,1,.94))
    save(fig,cycle['slug']+'_paired_frame_tr_delta','Posthoc descriptive frame deltas. Symlog linear interval +/-1 in each axis is a display setting, not an improvement threshold. Numeric parity tolerance1e-7; no pose failures silently become zero.'+(' Deterministic replay only, NOT independent training variation or evidence of robustness.' if is_replay else ''),dict(group=group,paired=counts,independent_replication=False if is_replay else None))


def overlays(cycle,populations,selected=False,target='REF',comparison='OLD_REF'):
    from scripts.research.pallet_selftraining_paper_closure_v1.report import overlay
    material=cycle['result']['material'];previous=C.P if material=='PLASTIC' else C.M
    approved_path=previous.DOC/'FIGURE_MANIFEST.json';approved=source(approved_path)
    key='frame_id' if material=='PLASTIC' else 'id';allowed={r[key] for r in approved['examples']}
    group=PRIMARY if material=='PLASTIC' else 'ALL';ids=populations[material]['groups'][group]
    chosen,counts=select_examples(allowed,ids,cycle['metrics'][comparison],cycle['metrics']['NEW_'+target])
    predictions=source(cycle['raw']/f"PREDICTIONS_{cycle['tag']}.json")
    truth=source(C.P.TRUTH)
    for entry in chosen:
        entry.update(cycle=cycle['result']['cycle'],material=material,seed=cycle['result']['seed'],target=target,
            comparison=comparison,population=group,eligible_approved=len(allowed&set(ids)),all_eligible_direction_counts=dict(counts),
            previous_publication=C.bind(approved_path),selected_recipe=selected,
            selection='Within previously published IDs only: both signed deltas beyond +/-1e-7; same ranking for improvement/deterioration = abs(delta T) descending, abs(delta R) descending, ID. Numeric ties need both within1e-7. No fallback relabeling.')
        if entry['status']=='NA':EXAMPLES.append(entry);continue
        fid=entry['id'];metadata=cycle['metadata'][fid];C.verify(metadata['image'])
        with Image.open(C.ROOT/metadata['image']['path']) as source_image:im=source_image.convert('RGB')
        assert list(im.size)==list(reversed(metadata['hw']))
        gt={j:xy for j,xy in enumerate(truth[fid]['gt'][:8]) if truth[fid]['valid'][j]}
        arms=['R0',comparison,'NEW_RAW','NEW_REF'];fig,axs=plt.subplots(1,4,figsize=(16,3.5))
        colors=['#009fcb','#b64cb7','#d0a13b','#e67626']
        panel_metrics=[]
        for ax,arm,color in zip(axs,arms,colors):
            row=cycle['metrics'][arm][fid]
            caption=f"{arm}\nT={row['translation_cm']:.3f} cm, R={row['rotation_deg']:.3f} deg" if row['available'] else arm+'\nPose unavailable'
            overlay(ax,im,predictions[arm][fid],gt,caption,color)
            assert tuple(ax.get_xlim())==(0.,float(im.width)) and tuple(ax.get_ylim())==(float(im.height),0.)
            panel_metrics.append(dict(arm=arm,available=row['available'],translation_cm=row.get('translation_cm'),rotation_deg=row.get('rotation_deg')))
        display_category={'improved':'joint numeric decrease','worsened':'joint numeric increase','unchanged':'numeric parity'}[entry['category']]
        title=f"{cycle['label']} {display_category}: {fid} | NEW_{target} - {comparison}: T {entry['delta_translation_cm']:+.3f} cm, R {entry['delta_rotation_deg']:+.3f} deg"
        fig.suptitle(title+'\nGreen crosses: legacy 2D reference. Colored corners/edges: native 2D prediction, NOT PnP reprojection.',fontsize=9)
        fig.text(.5,.02,'Panel T/R are separate D9 pose metrics against geometry-derived reference; physical signed-axis truth is unresolved.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.045,1,.84))
        name=('selected_' if selected else '')+cycle['slug']+'_'+entry['category']
        path=save(fig,name,'Already-public RGB only. Joint numeric T/R change against OLD_REF is not a practical-significance claim and is not native2D visual improvement. R0 panels are separate descriptive baseline, not the example selection comparator.',dict(material=material,frame_id=fid))
        entry.update(figure=C.bind(path),image=metadata['image'],image_dimensions_wh=list(im.size),
            native_image_bounds_clipped=True,green_reference='legacy supported corner2D; unknown independent physical provenance',
            colored_output='native detector keypoints, no symmetry remapping or PnP reprojection',panel_pose_metrics=panel_metrics)
        EXAMPLES.append(entry)


def main():
    SOURCES.clear();FIGURES.clear();EXAMPLES.clear()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
    SOURCES.add(Path(__file__)); baseline=source(C.DOC/'BASELINE_POSE_RESULTS.json')
    C.verify(baseline['private_metric_artifact'])
    source(C.DOC/'METRIC_AND_SELECTION_LOCK.json')
    correction_path=C.DOC/'REPLICATION_VALIDITY_CORRECTION.json'
    correction=source(correction_path) if correction_path.exists() else None
    populations=source(C.RAW/'metric_baseline/POPULATION_LOCK_PRIVATE.json')
    baseline_frames=source(C.ROOT/baseline['private_metric_artifact']['path'])
    checked=sum(verify_summaries(baseline['materials'][mat]['groups'],baseline_frames[mat],populations[mat]['groups']) for mat in populations)
    cycles=read_cycles()
    if any(c['result']['cycle'] in REPLAY_CYCLES for c in cycles):
        assert correction and correction['status']=='NOT_RUN_EFFECTIVE_TRAINING_VARIATION'
        assert correction['deterministic_reexecution_verified'] and not correction['effective_data_variation']
    checked+=sum(verify_summaries(c['result']['groups'],c['metrics'],populations[c['result']['material']]['groups']) for c in cycles)
    objective_plot(baseline,cycles);severity_plot(baseline,cycles)
    for cycle in cycles:paired_delta(cycle,populations)
    selection_path=C.DOC/'FINAL_SELECTION.json';selection=source(selection_path) if selection_path.exists() else None
    selected=None
    if selection and selection.get('selected_cycle'):
        selected=next((c for c in cycles if c['result']['cycle']==selection['selected_cycle'] and
            c['result']['material']==selection.get('selected_material','PLASTIC') and c['result']['seed']==selection.get('selected_seed',42)),None)
        assert selected is not None,'Selected result missing; do not silently substitute another recipe'
        overlays(selected,populations,True,selection.get('selected_target','REF'),selection.get('comparison','OLD_REF'))
        for cycle in cycles:
            if cycle['result']['material']=='WOOD':overlays(cycle,populations)
    else:
        for cycle in cycles:overlays(cycle,populations)
    retained=register_retained()
    C.save(C.DOC/'FIGURE_MANIFEST.json',dict(kind='Measured aggregate plots and pre-approved RGB only',generated_utc=C.now(),
        completed_cycles=[dict(cycle=c['result']['cycle'],material=c['result']['material'],seed=c['result']['seed'],result=C.bind(c['path'])) for c in cycles],
        selected_recipe=selection,selection_status='HISTORICAL_SELECTION_WITH_REPLICATION_CORRECTION' if selected and correction else 'FINAL_SELECTION_AVAILABLE' if selected else 'NO_FINAL_SELECTION_PER_CYCLE_EXAMPLES',
        replication_correction=C.bind(correction_path) if correction else None,
        effective_independent_replication_status=correction['status'] if correction else 'NO_CORRECTION_AVAILABLE',
        interpretation='PARTIAL_BUDGET: seed43 counts as spent fits but not independent replication; historical FINAL_SELECTION pending/seed wording is superseded by correction.' if correction else 'Reused DEV only.',
        overview_omitted_duplicate_replays=[c['result']['cycle'] for c in cycles if c['result']['cycle'] in REPLAY_CYCLES],
        retained_historical_manifest=retained,retained_historical_figures=sum('historical_role' in f for f in FIGURES),
        figures=FIGURES,examples=EXAMPLES,sources=[C.bind(path) for path in sorted(SOURCES)],
        total_png_bytes=sum(f['file']['bytes'] for f in FIGURES),T_R_median_P90_values_verified_from_private_frames=checked,
        image_policy='No new RGB IDs; image hash and old publication manifest verified; native image bounds preserved. No coordinate arrays published.',
        invariants=dict(old_outputs_written=False,inference=False,training=False,evaluation_predictions_modified=False,
            GT_for_posthoc_example_selection_only=True,all_full_population_denominators_preserved=True)))
    lines=['# Measured figure index','',
        'All labels/captions are English. These are reused-DEV diagnostics, not independent generalization or physical signed-axis validation.','']
    if correction:
        lines+=['Nominal seed43 fits reproduced identical loader streams and all879 tensors. They are numerical replays, **not independent training-seed replications**. Effective replication is NOT_RUN; final scope remains PARTIAL_BUDGET. Duplicate replay points/curves are omitted from the objective/severity overviews, while separately labeled replay plots remain visible. Historical FINAL_SELECTION wording does not override [the correction](REPLICATION_VALIDITY_CORRECTION.md).','']
    for figure in FIGURES:
        path=Path(figure['file']['path']).relative_to(C.DOC.relative_to(C.ROOT))
        heading=('Retained historical A/B: ' if figure.get('historical_role') else '')+figure['name']
        lines += [f"## {heading}",'',f"![{figure['name']}]({path})",'',figure['caption'],'']
    missing=[r for r in EXAMPLES if r['status']=='NA']
    if missing:
        lines+=['## Unavailable example categories','']
        lines+=[f"- {r['cycle']} {r['material']} S{r['seed']} {r['category']}: {r['reason']}" for r in missing]
    C.save(C.DOC/'FIGURE_INDEX.md','\n'.join(lines)+'\n')
    print('V2_FIGURES_READY',len(FIGURES),'plots',len([r for r in EXAMPLES if r['status']=='AVAILABLE']),'RGB_examples',flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--register-retained-only',action='store_true')
    if parser.parse_args().register_retained_only:register_retained_only()
    else:main()
