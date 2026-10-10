"""Regenerate public figures from published rows/tables, without RGB or inference."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / '_docs/experiments/pallet_n3_subpix_final_20261010'
METHODS = ('BASE', 'N3_DIM_SYM', 'SUBPIX', 'N3_THEN_SUBPIX')
LABEL = dict(zip(METHODS, ('Base', 'N3', 'SubPix', 'N3 → SubPix')))
COLORS = dict(zip(METHODS, ('#727272', '#2070b4', '#cf7c14', '#18845b')))
METRICS = ('corner_px', 'translation_cm', 'rotation_deg', 'ADDsym_cm')
TITLES = dict(zip(METRICS, ('Corner error (px)', 'Translation error (cm)',
                           'Rotation error (°)', 'ADDsym error (cm)')))
FIGURES = (
    ('01_method_overview.png', 'Fixed integrated refinement pipeline'),
    ('02_pose_accuracy_by_seed.png', 'Pose mean by seed and mean across seeds'),
    ('03_paired_improvement_ci.png', 'Paired session cluster confidence intervals'),
    ('04_grade_breakdown.png', 'All annotation difficulty grades retained'),
    ('05_error_distribution.png', 'Untrimmed error distributions'),
    ('06_corner_damage_and_hypothesis.png', 'Corner quality, movement cap and hypothesis diagnostics'),
    ('07_qualitative_examples.png', 'Post-hoc coordinate-only improvement and failure examples'),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_rows(doc):
    with gzip.open(doc / 'PREDICTIONS.jsonl.gz', 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def mean_rows(rows, method):
    """Keep one arithmetic mean per original image/canonical corner, not 957 images."""
    groups = {}
    for row in rows:
        if row['method'] == method:
            groups.setdefault(row['id'], []).append(row)
    return [groups[key] for key in sorted(groups)]


def values(rows, method, metric):
    result = []
    for group in mean_rows(rows, method):
        if metric == 'corner_px':
            errors = np.asarray([r['corner']['canonical_errors'] for r in group], float)
            observed = np.asarray([r['canonical_observed'] for r in group], bool).all(0)
            result.extend(errors[:, observed].mean(0).tolist())
        elif all(r['pose']['available'] for r in group):
            field = 'ADDsym_m' if metric == 'ADDsym_cm' else metric
            scale = 100 if metric == 'ADDsym_cm' else 1
            result.append(scale * np.mean([r['pose'][field] for r in group]))
    return np.asarray(result, float)


def polish(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=.18, axis='x')
    ax.set_axisbelow(True)


def save(fig, out):
    fig.savefig(out, dpi=145, facecolor='white', bbox_inches='tight', pil_kwargs={'compress_level':9})
    plt.close(fig)


def overview(out):
    fig = plt.figure(figsize=(15, 5.3))
    ax = fig.add_axes([.025, .08, .95, .8]); ax.set_xlim(0, 15); ax.set_ylim(0, 4); ax.axis('off')
    def box(x, y, w, h, text, color, fontsize=11):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.09',
                     facecolor=color,edgecolor='#496075',linewidth=1.2))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fontsize)
    box(.1, 1.4, 2.1, 1.2, 'Original RGB\nfixed detector\nBase keypoints q₀', '#e9eef3')
    ax.add_patch(FancyBboxPatch((2.9,.95),8.5,2.2,boxstyle='round,pad=.12',
                 facecolor='#f2faf6',edgecolor='#18845b',linewidth=2))
    ax.text(7.15,2.95,'One keypoint refinement block',ha='center',fontsize=13,weight='bold')
    box(3.15,1.3,2.25,1.15,'N3 learned initialization\nfixed seed checkpoint\nq₀ → qN', '#dceaf7')
    box(6.05,1.3,2.25,1.15,'cornerSubPix\noriginal gray gradients\nqN → qS', '#fff0d9')
    box(8.95,1.3,2.2,1.15,'Final total cap\nanchor: original q₀\n≤ 1% image diagonal', '#d9efe3')
    box(12.1,1.4,2.6,1.2,'Existing final F\nW/D hypotheses\nSQPnP + LM → R, t', '#e9eef3')
    for a,b in ((2.25,3.02),(5.5,5.93),(8.4,8.84),(11.45,11.98)):
        ax.annotate('',xy=(b,1.9),xytext=(a,1.9),arrowprops=dict(arrowstyle='->',lw=1.8,color='#364659'))
    ax.annotate('original gray',xy=(7.15,2.65),xytext=(1.15,3.5),ha='center',fontsize=10,
                arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90',color='#a26b16'))
    ax.text(7.5,.4,'Fixed object selection, K, dimensions, support mask and center point 8.\n'
            'N3 was trained earlier on synthetic corner offsets; no joint training or hidden-corner recovery.',
            ha='center',va='center',fontsize=11)
    fig.suptitle('Feature-Initialized Image-Gradient Keypoint Refinement',fontsize=16,weight='bold')
    save(fig,out)


def accuracy(metrics, out):
    fig, axes = plt.subplots(1,2,figsize=(15,8.4))
    scopes = [('seed 1',metrics['by_seed']['1']['ALL']),('seed 2',metrics['by_seed']['2']['ALL']),
              ('seed 3',metrics['by_seed']['3']['ALL']),('seed-mean',metrics['seed_mean']['ALL'])]
    for ax, metric in zip(axes, ('translation_cm','rotation_deg')):
        for group, (scope, summary) in enumerate(scopes):
            for mindex, method in enumerate(METHODS):
                stat=summary[method]['metrics'][metric]; y=group*5+mindex
                ax.scatter(stat['mean'],y,color=COLORS[method],s=65,zorder=3)
                ax.annotate(f"{stat['mean']:.2f} ± {stat['sample_std']:.2f}  n={stat['n']}",
                            xy=(stat['mean'],y),xytext=(9,-3),textcoords='offset points',
                            fontsize=9,color=COLORS[method],ha='left',va='center')
            ax.axhline(group*5+3.8,color='#c9c9c9',lw=.6)
        ax.set_yticks([g*5+mi for g in range(4) for mi in range(4)],
                      [f'{scope}: {LABEL[m]}' for scope,_ in scopes for m in METHODS])
        ax.invert_yaxis(); ax.set_xlabel(TITLES[metric]); polish(ax)
        ax.set_xlim(0,max(s[method]['metrics'][metric]['mean'] for _,s in scopes for method in METHODS)*1.8)
    fig.suptitle('Pose accuracy: every seed retained',fontsize=17,weight='bold')
    fig.text(.5,.025,'Points show means; labels show mean ± sample SD (ddof=1), with untrimmed errors.\n'
             'Every scope and method: successful poses 319/319, no_pose 0; seed-mean averages three errors per original image.',
             ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.065,1,.96]);save(fig,out)


def paired_plot(paired, out):
    fig,axes=plt.subplots(1,2,figsize=(15,6.5))
    labels=[]
    for scope in ('1','2','3','seed_mean'):
        for comp in ('N3_DIM_SYM','SUBPIX'):
            labels.append((scope,comp))
    for ax,metric in zip(axes,('translation_cm','rotation_deg')):
        for y,(scope,comp) in enumerate(labels):
            source=paired['seed_mean'] if scope=='seed_mean' else paired['by_seed'][scope]
            stat=source['N3_THEN_SUBPIX_minus_'+comp]['statistics'][metric]
            avg=stat['mean_paired_difference']; ci=stat['CI95']
            ax.plot(ci,[y,y],color=COLORS[comp],lw=2)
            ax.scatter(avg,y,color=COLORS[comp],s=55,zorder=4)
            ax.text(ci[1],y-.17,f' {avg:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]',fontsize=8)
        ax.axvline(0,color='#333',lw=1,ls='--');ax.set_yticks(range(len(labels)),
            [f"{'seed '+s if s!='seed_mean' else 'seed-mean'}: combination − {LABEL[c]}" for s,c in labels])
        ax.invert_yaxis();ax.set_xlabel('Δ '+TITLES[metric]);polish(ax)
        lo,hi=ax.get_xlim();ax.set_xlim(lo,hi+(hi-lo)*.35)
    fig.suptitle('Paired additional effect: negative = lower error',fontsize=16,weight='bold')
    fig.text(.5,.015,'13-session paired cluster bootstrap, 10,000 shared draws, seed 20260917; 95% percentile CI.\n'
             'A CI crossing zero leaves the direction uncertain; seed-mean does not count 957 independent images.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.06,1,.96]);save(fig,out)


def grades(metrics, out):
    fig,axes=plt.subplots(2,2,figsize=(14,8))
    for ax,metric in zip(axes.flat,('translation_cm','rotation_deg','corner_px','ADDsym_cm')):
        for mi,method in enumerate(METHODS):
            ys=[metrics['seed_mean'][grade][method]['metrics'][metric]['mean'] for grade in ('clean','moderate','severe')]
            ax.plot(np.arange(3)+(mi-1.5)*.07,ys,'o-',color=COLORS[method],label=LABEL[method],lw=1.5)
        ax.set_xticks(range(3),['clean\n153 images','moderate\n92 images','severe\n74 images'])
        ax.set_ylabel(TITLES[metric]);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0,0].legend(fontsize=9,ncol=2)
    fig.suptitle('Difficulty annotation grades: seed-mean, all 319 images',fontsize=16,weight='bold')
    fig.text(.5,.015,'These are existing annotation difficulty grades, not measured physical occlusion rates.\n'
             'Means retain all finite eligible errors, including severe outliers; successful pose and corner denominators differ.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.06,1,.95]);save(fig,out)


def ecdf(rows, metrics, out):
    fig,axes=plt.subplots(2,2,figsize=(15,9))
    for ax,metric in zip(axes.flat,('translation_cm','rotation_deg','corner_px','ADDsym_cm')):
        for method in METHODS:
            data=np.sort(values(rows,method,metric));stat=metrics['seed_mean']['ALL'][method]['metrics'][metric]
            assert len(data)==stat['n']
            ax.step(data,np.arange(1,len(data)+1)/len(data),where='post',color=COLORS[method],
                label=f"{LABEL[method]} n={len(data)} | mean {stat['mean']:.2f} | med {stat['median']:.2f} | P90 {stat['P90']:.2f}",lw=1.6)
        ax.set_xscale('symlog',linthresh=1);ax.set_xlabel(TITLES[metric]+' (symlog scale, linear ≤ 1)')
        ax.set_ylabel('Empirical cumulative fraction');ax.set_ylim(0,1.03)
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
        ax.legend(fontsize=7,loc='lower right')
    fig.suptitle('Untrimmed distributions of per-image / per-corner seed means',fontsize=16,weight='bold')
    fig.text(.5,.01,'Every finite eligible error is retained, including the maximum. No clipping or winsorization.\n'
             'Pose: common successful original images; corners: original observed canonical corners. Failures remain separate in the report.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.06,1,.95]);save(fig,out)


def diagnostics_plot(rows, metrics, out):
    fig,axes=plt.subplots(2,2,figsize=(14,8))
    for mi,method in enumerate(METHODS):
        vals=[metrics['by_seed'][str(s)]['ALL'][method]['corner']['PCK']['10'] for s in (1,2,3)]
        axes[0,0].plot((1,2,3),vals,'o-',label=LABEL[method],color=COLORS[method])
    axes[0,0].set_ylabel('PCK@10 (all reference corners)');axes[0,0].set_ylim(.5,.75)
    axes[0,0].legend(fontsize=9,ncol=2)
    harms=[];recover=[];capc=[];capf=[];hn=[];hb=[]
    for seed in (1,2,3):
        bymethod={m:{r['id']:r for r in rows if r['seed']==seed and r['method']==m} for m in METHODS}
        bad=good=cc=cf=sn=sb=0
        for fid,row in bymethod['N3_THEN_SUBPIX'].items():
            n3=bymethod['N3_DIM_SYM'][fid];base=bymethod['BASE'][fid]
            valid=np.asarray(row['corner']['canonical_valid'],bool)
            before=np.asarray(n3['corner']['canonical_errors'],float)[valid]
            after=np.asarray(row['corner']['canonical_errors'],float)[valid]
            bad+=np.sum((before<5)&(after>10));good+=np.sum((before>20)&(after<=10))
            active=np.asarray(row['correction']['cap_active8'],bool);cc+=active.sum();cf+=active.any()
            if row['pose']['available'] and n3['pose']['available']:
                sn+=row['final_hypothesis']!=n3['final_hypothesis']
            if row['pose']['available'] and base['pose']['available']:
                sb+=row['final_hypothesis']!=base['final_hypothesis']
        harms.append(bad);recover.append(good);capc.append(cc);capf.append(cf);hn.append(sn);hb.append(sb)
    for ax, series, labs, title in (
        (axes[0,1],(harms,recover),('good<5 → bad>10','bad>20 → good≤10'),'N3 → combination corner harm / recovery'),
        (axes[1,0],(capc,capf),('corners capped','frames with ≥1 cap'),'Final cap activation: original Base anchor'),
        (axes[1,1],(hn,hb),('vs N3','vs Base'),'Final W/D hypothesis switches')):
        for offset, (vals,lab) in enumerate(zip(series,labs)):
            x=np.arange(3)+(offset-.5)*.3
            bars=ax.bar(x,vals,width=.28,label=lab,color=('#c75c50','#529b91')[offset])
            ax.bar_label(bars,padding=3,fontsize=10)
        ax.set_title(title,fontsize=11);ax.set_ylabel('Count');ax.legend(fontsize=9)
    for ax in axes.flat:
        ax.set_xticks(range(3),['seed 1','seed 2','seed 3']) if ax!=axes[0,0] else ax.set_xticks((1,2,3),['seed 1','seed 2','seed 3'])
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Independent diagnostics: benefits and harms remain visible',fontsize=16,weight='bold')
    fig.text(.5,.015,'Harm / recovery use the same canonical reference identities; cap and hypothesis panels are post-hoc diagnostics.\n'
             'Counts refer to repeated executions of the same 319 images per seed, not additional independent images.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.055,1,.95]);save(fig,out)


def qualitative(rows, out):
    seed1={m:{r['id']:r for r in rows if r['seed']==1 and r['method']==m} for m in METHODS}
    ids=[fid for fid,r in seed1['N3_THEN_SUBPIX'].items() if r['pose']['available'] and seed1['N3_DIM_SYM'][fid]['pose']['available']]
    def dt(fid):return seed1['N3_THEN_SUBPIX'][fid]['pose']['translation_cm']-seed1['N3_DIM_SYM'][fid]['pose']['translation_cm']
    def dr(fid):return seed1['N3_THEN_SUBPIX'][fid]['pose']['rotation_deg']-seed1['N3_DIM_SYM'][fid]['pose']['rotation_deg']
    selections=[('largest T improvement',min(ids,key=lambda fid:(dt(fid),fid)),'minimum combined − N3 translation error'),
                ('largest T worsening',min(ids,key=lambda fid:(-dt(fid),fid)),'maximum combined − N3 translation error'),
                ('near-zero joint change',min(ids,key=lambda fid:(abs(dt(fid))+abs(dr(fid)),fid)),'minimum |ΔT (cm)| + |ΔR (degree)|; descriptive only; not necessarily exactly zero'),
                ('largest absolute T error',min(ids,key=lambda fid:(-seed1['N3_THEN_SUBPIX'][fid]['pose']['translation_cm'],fid)),'maximum combined translation error')]
    fig,axes=plt.subplots(4,2,figsize=(14,15))
    details=[]
    markers={'BASE':'o','N3_DIM_SYM':'s','SUBPIX':'^','N3_THEN_SUBPIX':'x'}
    for rowindex,(kind,fid,rule) in enumerate(selections):
        row=seed1['N3_THEN_SUBPIX'][fid];h,w=row['raw_hw']; scale=np.array([w,h])
        ax,zoom=axes[rowindex]
        for method in METHODS:
            q=np.asarray(seed1[method][fid]['qFinal'],float)[:8]/scale
            support=np.asarray(row['prediction_support'][:8],bool)
            ax.scatter(q[support,0],q[support,1],s=35,marker=markers[method],color=COLORS[method],label=LABEL[method],alpha=.85)
        reference=np.asarray(row.get('evaluation_reference_points',[]),float)
        if reference.size:
            ref=reference[np.asarray(row['evaluation_permutation'],int)][:8]/scale
            valid=np.asarray(row['evaluation_reference_valid'],bool)[np.asarray(row['evaluation_permutation'],int)][:8]
            ax.scatter(ref[valid,0],ref[valid,1],marker='+',s=65,color='#7c4db0',label='geometric reference')
        q0=np.asarray(row['q0'],float)[:8];qn=np.asarray(row['qN'],float)[:8]
        qs=np.asarray(row['qS'],float)[:8];qf=np.asarray(row['qFinal'],float)[:8]
        moved=np.linalg.norm(qf-qn,axis=-1);support=np.asarray(row['prediction_support'][:8],bool)
        moved[~support]=-1;k=int(np.argmax(moved))
        for name,coords,color,marker in (('q₀',q0,'#727272','o'),('qN',qn,'#2070b4','s'),
                                       ('qS before final cap',qs,'#cf7c14','^'),('qFinal',qf,'#18845b','x')):
            point=(coords[k]-q0[k])/scale*1000
            zoom.scatter(*point,s=85,c=color,marker=marker,label=name)
        path=np.array([q0[k],qn[k],qs[k],qf[k]])
        for start,end in zip(path[:-1],path[1:]):
            zoom.annotate('',xy=(end-q0[k])/scale*1000,xytext=(start-q0[k])/scale*1000,
                arrowprops=dict(arrowstyle='->',color='#8b8b8b',lw=1.4))
        ax.set_title(f'{kind}\n{fid} | seed 1 | ΔT={dt(fid):+.2f} cm, ΔR={dr(fid):+.2f}°',fontsize=10)
        ax.set_xlabel('x / original image width');ax.set_ylabel('y / original image height');ax.invert_yaxis()
        ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
        if rowindex==0:ax.legend(fontsize=8,ncol=3,loc='upper center')
        zoom.set_title(f'Corner {k}: normalized movement from Base (×1000)\nT={row["pose"]["translation_cm"]:.2f} cm, R={row["pose"]["rotation_deg"]:.2f}°',fontsize=10)
        zoom.set_xlabel('Δx / width ×1000');zoom.set_ylabel('Δy / height ×1000');zoom.invert_yaxis()
        zoom.grid(alpha=.2);zoom.set_aspect('equal',adjustable='datalim');zoom.margins(.22)
        zoom.legend(loc='upper right',fontsize=8,framealpha=.95)
        details.append(dict(category=kind,id=fid,seed=1,selection_rule=rule,tie_break='lexicographic ID',
            translation_delta_cm=dt(fid),rotation_delta_deg=dr(fid),zoom_corner=k,
            zoom_rule='maximum combined-minus-N3 displacement over supported native corners',
            normalization='x/width,y/height; movement inset multiplied by1000',
            reference='evaluation-only geometric reconstruction; native ordering uses evaluation_permutation',
            editing='no RGB; scatter overlays and arrows from saved points only'))
    fig.suptitle('Post-hoc coordinate examples: improvements, harms and extreme failure',fontsize=16,weight='bold')
    fig.text(.5,.015,'Private RGB redistribution rights are unconfirmed; no original image or workplace is published.\n'
             'Selection is descriptive after evaluation. Reference points are for scoring/illustration only and never enter inference.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.055,1,.97]);save(fig,out)
    return details


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--doc',type=Path,default=DOC)
    args=parser.parse_args();doc=args.doc.resolve();out=doc/'figures';out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12})
    rows=load_rows(doc);metrics=read(doc/'METRICS.json');paired=read(doc/'PAIRED_COMPARISONS.json')
    overview(out/FIGURES[0][0]);accuracy(metrics,out/FIGURES[1][0]);paired_plot(paired,out/FIGURES[2][0])
    grades(metrics,out/FIGURES[3][0]);ecdf(rows,metrics,out/FIGURES[4][0])
    diagnostics_plot(rows,metrics,out/FIGURES[5][0]);examples=qualitative(rows,out/FIGURES[6][0])
    sources=[dict(path=name,sha256=sha(doc/name)) for name in ('PREDICTIONS.jsonl.gz','METRICS.json','PAIRED_COMPARISONS.json','RESULTS.csv')]
    figures=[]
    for i,(name,title) in enumerate(FIGURES,1):
        figures.append(dict(path='figures/'+name,title=title,sha256=sha(out/name),bytes=(out/name).stat().st_size,
            seeds=[1,2,3] if i not in (1,7) else ([1] if i==7 else []),
            original_images=319 if i!=1 else None,independent_sessions=13 if i!=1 else None,
            evidence=sources if i!=1 else [dict(path='INPUT_AND_METHOD_LOCK.json',sha256=sha(doc/'INPUT_AND_METHOD_LOCK.json'))],
            generated_by='scripts/research/pallet_n3_subpix_final_20261010/figures.py',
            creator='repository experiment script; original matplotlib graphics',
            rights_status='OWN_GENERATED_COORDINATE_OR_AGGREGATE_CHART; NO_RGB',
            content='coordinate-only post-hoc examples' if i==7 else 'aggregate scientific chart' if i!=1 else 'method diagram'))
    index=dict(schema='pallet_n3_subpix_final_figure_index_v1',figures=figures,qualitative_examples=examples,
        RGB_publication='PRIVATE_NOT_PUBLISHED: original redistribution rights not established',
        seed_mean_definition='three seed errors averaged within each original image/canonical corner; never 957 independent images',
        statistics_script='scripts/research/pallet_n3_subpix_final_20261010/summarize.py',
        visual_inspection='pending external review; PNG decode and visual inspection recorded in EXECUTION_AND_VERIFICATION.json')
    (doc/'FIGURE_INDEX.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(figures=len(figures),paths=[f['path'] for f in figures],raw_rgb_used=False)))


if __name__=='__main__':main()
