"""Six public joint-refinement figures from numeric rows only; never load RGB."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch
import numpy as np

from scripts.research.pallet_n3_subpix_final_20261010.figures import load_rows, values, sha, save

ROOT=Path(__file__).resolve().parents[3]
DOC=ROOT/'_docs/experiments/pallet_feature_gradient_joint_20261010'
METHODS=('BASE','N3_DIM_SYM','SUBPIX','N3_THEN_SUBPIX','JOINT_FIXED_ISOTROPIC','FG_JOINT_POSTERIOR')
LABEL=dict(zip(METHODS,('Base','N3','SubPix','Sequential','Isotropic','Joint posterior')))
COLOR=dict(zip(METHODS,('#797979','#2479bc','#ca8015','#178266','#9870bb','#c84050')))
COMPARATORS=('N3_THEN_SUBPIX','N3_DIM_SYM','SUBPIX','JOINT_FIXED_ISOTROPIC')
TITLE={'translation_cm':'Translation (cm)','rotation_deg':'Rotation (°)','ADDsym_cm':'ADDsym (cm)','corner_px':'Corner (px)'}
FIGURES=(('01_joint_method.png','One joint posterior/image equation versus existing sequential route'),
         ('02_seed_pose_comparison.png','All6 methods, three seeds and per-image seed mean'),
         ('03_paired_ci.png','Joint minus sequential / single methods / isotropic: session95%CI'),
         ('04_per_grade_damage.png','Every difficulty grade, corner harm and fallback'),
         ('05_qualitative.png','Rule-selected coordinate-only improvement/worsening/nearzero/gross cases'),
         ('06_covariance_vs_gradient.png','Actual prior/gradient axes and output movements'))


def polish(ax):
    ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.18);ax.set_axisbelow(True)


def overview(out):
    fig,ax=plt.subplots(figsize=(16,7));ax.axis('off');ax.set_xlim(0,16);ax.set_ylim(0,7)
    def box(x,y,w,h,text,c):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.1',facecolor=c,edgecolor='#4b5969',lw=1.3))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11)
    def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1.8,color='#455364'))
    ax.text(.2,6.1,'Existing sequential baseline',fontsize=14,weight='bold')
    box(.2,4.8,2.3,1,'Base q₀\noriginal RGB detector','#eef0f3')
    box(3.2,4.8,2.8,1,'Frozen N3 location qN\nlearned initialization','#e1edf7')
    box(6.7,4.8,3.1,1,'OpenCV cornerSubPix\niterative gradient refinement','#fff0db')
    box(10.5,4.8,2.2,1,'Final Base cap\n1% diagonal','#e1f1e9')
    box(13.4,4.8,2.3,1,'Frozen final F\nW/D + SQPnP + LM','#eef0f3')
    for a,b in ((2.6,3.05),(6.1,6.55),(9.9,10.35),(12.8,13.25)):arrow((a,5.3),(b,5.3))
    ax.text(.2,4.1,'New inference-only joint location calculation',fontsize=14,weight='bold')
    box(.2,2.45,3.7,1.25,'N3: all222 logits + native candidates\nμ = existing frozen qN\nΣ = stabilized covariance proxy','#e1edf7')
    box(.2,.75,3.7,1.25,'Original gray [0,1], Sobel3\nfixed qN-centered11×11 window\nnormalized image equation A, b','#fff0db')
    box(4.7,1.15,5.45,2.2,'ONE joint solve\nE(q) = (q−μ)ᵀΣ⁻¹(q−μ) + (qᵀAq−2qᵀb+const)/16\nq* = solve(Σ⁻¹ + A/16, Σ⁻¹μ + b/16)\nIsotropic ablation changes only Σ → 16 I₂','#f9e6e9')
    box(10.8,1.75,1.95,1,'Final Base cap\n1% diagonal','#e1f1e9')
    box(13.4,1.75,2.3,1,'Same frozen F\nR, t + same scoring','#eef0f3')
    arrow((4,3.05),(4.55,2.8));arrow((4,1.4),(4.55,1.75));arrow((10.3,2.25),(10.65,2.25));arrow((12.85,2.25),(13.25,2.25))
    ax.text(8,.2,'No joint-path cornerSubPix call; no extra training, GT inputs, coordinate averaging or DEV tuning.',ha='center',fontsize=11)
    fig.suptitle('Feature-Guided Joint Keypoint Refinement: actual computational difference',fontsize=16,weight='bold')
    save(fig,out)


def pose_plot(metrics,out):
    fig,axes=plt.subplots(1,3,figsize=(19,11))
    scopes=[(s,metrics['by_seed'][s]['ALL']) for s in ('1','2','3')]+[('seed-mean',metrics['seed_mean']['ALL'])]
    for ax,metric in zip(axes,('translation_cm','rotation_deg','ADDsym_cm')):
        for si,(scope,summaries) in enumerate(scopes):
            for mi,m in enumerate(METHODS):
                stat=summaries[m]['metrics'][metric];y=si*7+mi
                ax.scatter(stat['mean'],y,color=COLOR[m],s=48)
                ax.annotate(f"{stat['mean']:.2f} ± {stat['sample_std']:.2f}",xy=(stat['mean'],y),
                    xytext=(8,-2),textcoords='offset points',fontsize=8,color=COLOR[m],va='center')
            ax.axhline(si*7+5.7,color='#cccccc',lw=.7)
        ax.set_yticks([si*7+mi for si in range(4) for mi in range(6)],
            [f'{scope}: {LABEL[m]}' for scope,_ in scopes for m in METHODS],fontsize=8)
        ax.invert_yaxis();ax.set_xlabel(TITLE[metric]);polish(ax)
        ax.set_xlim(0,max(s[m]['metrics'][metric]['mean'] for _,s in scopes for m in METHODS)*1.85)
    success=[s[m]['pose']['available'] for _,s in scopes for m in METHODS]
    fail=[s[m]['pose']['failures'] for _,s in scopes for m in METHODS]
    fig.suptitle('All fixed methods and seeds: pose mean with sample SD',fontsize=17,weight='bold')
    fig.text(.5,.015,f'Labels: mean ± sample SD (ddof=1); every finite error retained. Successful poses: {min(success)}–{max(success)}/319; no_pose: {min(fail)}–{max(fail)}.\n'
        'Seed-mean averages three scalar errors per original image: 319 images, never957 independent images.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.055,1,.95]);save(fig,out)


def paired_plot(paired,out):
    fig,axes=plt.subplots(1,3,figsize=(20,10))
    scopes=['1','2','3','seed_mean']; entries=[(s,m) for s in scopes for m in COMPARATORS]
    for ax,metric in zip(axes,('translation_cm','rotation_deg','ADDsym_cm')):
        for y,(scope,m) in enumerate(entries):
            source=paired['seed_mean']['ALL'] if scope=='seed_mean' else paired['by_seed'][scope]['ALL']
            stat=source['FG_JOINT_POSTERIOR_minus_'+m]['statistics'][metric];mu=stat['mean_paired_difference'];ci=stat['CI95']
            ax.plot(ci,[y,y],color=COLOR[m],lw=1.8);ax.scatter(mu,y,c=COLOR[m],s=40,zorder=3)
            ax.annotate(f'{mu:+.2f} [{ci[0]:+.2f},{ci[1]:+.2f}]',xy=(ci[1],y),xytext=(4,-8),
                textcoords='offset points',fontsize=7)
        ax.set_yticks(range(len(entries)),[f"{'seed '+s if s!='seed_mean' else 'seed-mean'} − {LABEL[m]}" for s,m in entries],fontsize=8)
        ax.invert_yaxis();ax.axvline(0,c='#333333',ls='--',lw=1);ax.set_xlabel('Joint Δ '+TITLE[metric]);polish(ax)
        lo,hi=ax.get_xlim();ax.set_xlim(lo-(hi-lo)*.04,hi+(hi-lo)*.45)
    fig.suptitle('Joint paired effects: primary comparison is Sequential',fontsize=17,weight='bold')
    fig.text(.5,.015,'Negative = lower error;13-session paired cluster bootstrap10000 identical draws, seed20260917,95%CI.\n'
        'Zero crossing remains visible. Seed-mean fixes three seeds and resamples original sessions only.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.055,1,.95]);save(fig,out)


def grades_damage(metrics,out):
    fig,axes=plt.subplots(2,3,figsize=(18,9));grades=('clean','moderate','severe')
    for ax,metric in zip(axes[0],('translation_cm','rotation_deg','ADDsym_cm')):
        for mi,m in enumerate(METHODS):
            y=[metrics['seed_mean'][g][m]['metrics'][metric]['mean'] for g in grades]
            ax.plot(np.arange(3)+(mi-2.5)*.04,y,'o-',color=COLOR[m],label=LABEL[m],lw=1.3)
        ax.set_ylabel(TITLE[metric]);ax.set_xticks(range(3),['clean153','moderate92','severe74']);polish(ax)
    axes[0,0].legend(fontsize=8,ncol=2)
    ax=axes[1,0]
    for mi,m in enumerate(METHODS):
        y=[metrics['seed_mean'][g][m]['corner']['gross20_rate'] for g in grades]
        ax.plot(np.arange(3)+(mi-2.5)*.04,y,'o-',color=COLOR[m],label=LABEL[m],lw=1.3)
    ax.set_ylabel('gross20 fraction (all reference corners)');ax.set_xticks(range(3),grades);polish(ax)
    diag=metrics['diagnostics']['by_seed'];ax=axes[1,1]
    for offset,(key,label) in enumerate((('damage_BASE','vs Base'),('damage_N3','vs N3'),('damage_SEQUENTIAL','vs Sequential'))):
        y=[diag[str(s)][key]['FG_JOINT_POSTERIOR']['good5_to_bad10'] for s in (1,2,3)]
        bars=ax.bar(np.arange(3)+(offset-1)*.25,y,width=.23,label=label)
        ax.bar_label(bars,fontsize=9,padding=2)
    ax.set_title('Joint: good<5px → bad>10px');ax.set_ylabel('Damaged corners');ax.set_xticks(range(3),['seed1','seed2','seed3']);ax.legend(fontsize=8);polish(ax)
    ax=axes[1,2]
    for offset,(kind,label) in enumerate((('flat_gradient','flat fallback'),('all','all fallback'),('cap','capped corners'))):
        y=[]
        for seed in ('1','2','3'):
            j=diag[seed]['joint']['FG_JOINT_POSTERIOR']
            y.append(j['cap_active_corners'] if kind=='cap' else sum(j['fallback_counts'].values()) if kind=='all' else j['fallback_counts'].get(kind,0))
        bars=ax.bar(np.arange(3)+(offset-1)*.25,y,width=.23,label=label)
        ax.bar_label(bars,fontsize=8,padding=2)
    ax.set_title('Joint corner fallback / final total cap');ax.set_xticks(range(3),['seed1','seed2','seed3']);ax.set_ylabel('Corner count');ax.legend(fontsize=8);polish(ax)
    fig.suptitle('Every difficulty grade and negative outcome retained',fontsize=16,weight='bold')
    fig.text(.5,.01,'Grades are existing annotation difficulty, not measured occlusion fractions. Finite gross failures remain in means.\n'
        'Seed-mean threshold rates average the three seed rates. Fallback corners and no_pose images are different counts.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.06,1,.95]);save(fig,out)


def selected_cases(rows):
    by={m:{r['id']:r for r in rows if r['seed']==1 and r['method']==m} for m in METHODS}
    joint=by['FG_JOINT_POSTERIOR'];seq=by['N3_THEN_SUBPIX']
    ids=[fid for fid in joint if joint[fid]['pose']['available'] and seq[fid]['pose']['available']]
    def dt(fid):return joint[fid]['pose']['translation_cm']-seq[fid]['pose']['translation_cm']
    def dr(fid):return joint[fid]['pose']['rotation_deg']-seq[fid]['pose']['rotation_deg']
    selectors=[('good',min(ids,key=lambda fid:(dt(fid),fid))),('bad',min(ids,key=lambda fid:(-dt(fid),fid))),
               ('near-zero',min(ids,key=lambda fid:(abs(dt(fid))+abs(dr(fid)),fid))),
               ('gross',min(ids,key=lambda fid:(-joint[fid]['pose']['translation_cm'],fid)))]
    return by,[dict(category=kind,id=fid,seed=1,deltaT_cm=dt(fid),deltaR_deg=dr(fid)) for kind,fid in selectors]


def qualitative(rows,out):
    by,cases=selected_cases(rows);fig,axes=plt.subplots(2,2,figsize=(14,10))
    for ax,case in zip(axes.flat,cases):
        fid=case['id'];row=by['FG_JOINT_POSTERIOR'][fid];h,w=row['raw_hw'];scale=np.array([w,h])
        support=np.asarray(row['prediction_support'][:8],bool)
        for m,marker in zip(('BASE','N3_DIM_SYM','N3_THEN_SUBPIX','JOINT_FIXED_ISOTROPIC','FG_JOINT_POSTERIOR'),('o','s','^','D','x')):
            q=np.asarray(by[m][fid]['qFinal'],float)[:8]/scale
            ax.scatter(q[support,0],q[support,1],s=40,color=COLOR[m],marker=marker,label=LABEL[m],alpha=.8)
        ref=np.asarray(row['evaluation_reference_points'],float)[np.asarray(row['evaluation_permutation'],int)][:8]/scale
        valid=np.asarray(row['evaluation_reference_valid'],bool)[np.asarray(row['evaluation_permutation'],int)][:8]
        ax.scatter(ref[valid,0],ref[valid,1],s=60,c='#562f88',marker='+',label='geometric reference')
        ax.set_title(f"{case['category']}: {fid}\nJoint−Sequential ΔT={case['deltaT_cm']:+.3f}cm, ΔR={case['deltaR_deg']:+.3f}°\nJoint T={row['pose']['translation_cm']:.2f}cm, R={row['pose']['rotation_deg']:.2f}°",fontsize=10)
        ax.set_xlabel('x / original width');ax.set_ylabel('y / original height');ax.invert_yaxis();ax.set_aspect('equal',adjustable='datalim');polish(ax)
        ax.legend(fontsize=7,ncol=2,loc='best')
    fig.suptitle('Locked selection rules, coordinate-only examples',fontsize=16,weight='bold')
    fig.text(.5,.015,'Rules fixed before new scoring: seed1 min/max Joint−Sequential T; min |ΔTcm|+|ΔR°|; maximum absolute Joint T.\n'
        'Tie: lexicographic ID. Geometric reference is scoring-only. No RGB, workplace or private source pathname.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.065,1,.94]);save(fig,out)
    return cases


def covariance(rows,cases,out):
    by,_=selected_cases(rows);fig,axes=plt.subplots(2,2,figsize=(14,10));details=[]
    for ax,case in zip(axes.flat,cases):
        fid=case['id'];row=by['FG_JOINT_POSTERIOR'][fid]
        qn=np.asarray(row['qN'],float);qf=np.asarray(row['qFinal'],float);qu=np.asarray(row['qS'],float)
        records=row['correction']['diagnostics']['corner_records']
        choices=[r['corner'] for r in records if r.get('Sigma') is not None and r.get('A') is not None]
        if not choices:
            ax.text(.5,.5,'No valid covariance/image record',ha='center',transform=ax.transAxes);continue
        k=max(choices,key=lambda k:(np.linalg.norm(qf[k]-qn[k]),-k));record=records[k]
        Sigma=np.asarray(record['Sigma']);A=np.asarray(record['A']);se,sv=np.linalg.eigh(Sigma);ae,av=np.linalg.eigh(A)
        angle=np.degrees(np.arctan2(sv[1,1],sv[0,1]))
        ax.add_patch(Ellipse((0,0),2*np.sqrt(se[1]),2*np.sqrt(se[0]),angle=angle,fill=False,ec=COLOR['N3_DIM_SYM'],lw=1.6,label='Σ proxy ellipse'))
        for j in range(2):
            direction=av[:,j]*4*np.sqrt(max(ae[j],0))
            ax.plot([-direction[0],direction[0]],[-direction[1],direction[1]],c='#ca8015',lw=2,
                    label='A principal directions' if j==1 else None)
        for point,label,color,marker in ((np.zeros(2),'N3 μ','#2479bc','s'),(qu[k]-qn[k],'Joint pre-cap','#c84050','o'),
            (qf[k]-qn[k],'Joint final','#c84050','x'),(np.asarray(by['N3_THEN_SUBPIX'][fid]['qFinal'])[k]-qn[k],'Sequential','#178266','^')):
            ax.scatter(*point,c=color,marker=marker,s=60,label=label)
        ax.annotate('',xy=qf[k]-qn[k],xytext=(0,0),arrowprops=dict(arrowstyle='->',color='#c84050',lw=1.4))
        extent=max(np.sqrt(se[1])+1,np.max(np.abs(qu[k]-qn[k]))+1,np.max(np.abs(qf[k]-qn[k]))+1)
        ax.set_xlim(-extent,extent);ax.set_ylim(extent,-extent);ax.set_aspect('equal');polish(ax)
        ax.set_title(f"{case['category']}: {fid}, corner{k}\nΣ eig={se[0]:.2f},{se[1]:.2f}px² | A eig={ae[0]:.3f},{ae[1]:.3f} | S={record['S']:.3g}",fontsize=9)
        ax.set_xlabel('x − N3 x (native px)');ax.set_ylabel('y − N3 y (native px)');ax.legend(fontsize=7,loc='best')
        details.append(dict(id=fid,seed=1,corner=k,selection='largest final Joint-minus-N3 movement among saved valid Sigma/A corners in each locked qualitative frame'))
    fig.suptitle('Actual covariance / gradient axes and movement',fontsize=16,weight='bold')
    fig.text(.5,.015,'Ellipse: one sqrt(eigenvalue) radius of stabilized Σ, not a calibrated confidence region.\n'
        'Orange A axes use schematic4sqrt(eigenvalue) lengths; directions are actual. Same selected frames; explanatory, not causal proof.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.065,1,.95]);save(fig,out)
    return details


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--doc',type=Path,default=DOC);args=parser.parse_args();doc=args.doc
    out=doc/'figures';out.mkdir(parents=True,exist_ok=True);rows=load_rows(doc)
    metrics=json.loads((doc/'METRICS.json').read_text());paired=json.loads((doc/'PAIRED.json').read_text())
    overview(out/FIGURES[0][0]);pose_plot(metrics,out/FIGURES[1][0]);paired_plot(paired,out/FIGURES[2][0]);grades_damage(metrics,out/FIGURES[3][0])
    cases=qualitative(rows,out/FIGURES[4][0]);cov_cases=covariance(rows,cases,out/FIGURES[5][0])
    evidence=[dict(path=f,sha256=sha(doc/f)) for f in ('PREDICTIONS.jsonl.gz','METRICS.json','METRICS.csv','PAIRED.json','FUSION_METHOD_LOCK.json','NEW_COORDINATES_SEALED.jsonl.gz')]
    index=dict(schema='feature_gradient_joint_figures_v1',figures=[dict(path='figures/'+name,title=title,sha256=sha(out/name),bytes=(out/name).stat().st_size,
        evidence=evidence,seeds=[1,2,3] if i not in (1,5,6) else [1] if i in (5,6) else [],images=319 if i not in (1,5,6) else 4 if i in (5,6) else None,
        sessions=13 if i not in (1,5,6) else None,rights_status='OWN_GENERATED_COORDINATE_OR_AGGREGATE_CHART; NO_RGB',
        generated_by='scripts/research/pallet_feature_gradient_joint_20261010/figures.py') for i,(name,title) in enumerate(FIGURES,1)],
        examples=cases,covariance_examples=cov_cases,RGB='PRIVATE_NOT_PUBLISHED; rights unconfirmed; no image read by figure script',
        selection_rules='FUSION_METHOD_LOCK.json qualitative_selection_rules sealed before new scoring; tie lexicographic ID',
        visual_inspection='pending external review in EXECUTION_LEDGER.json')
    (doc/'FIGURE_INDEX.json').write_text(json.dumps(index,indent=2)+'\n');print(json.dumps(dict(figures=6,RGB_used=False)))


if __name__=='__main__':main()
