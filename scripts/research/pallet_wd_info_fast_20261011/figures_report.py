"""Render exact, public derived diagnostics. No RGB, reference coordinates or fits.

PNG only: the user excludes PDF/manuscript changes. Existing figure style and
palette are vendored unchanged so reproduction does not depend on a home folder.
"""
import argparse
import gzip
import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from . import common as C
from .assets.palette import color_for, colors_for, SEMANTIC, LINESTYLE, ALPHA

ASSETS = Path(__file__).parent / 'assets'
plt.style.use(ASSETS / 'ieee.mplstyle')
FIG = C.DOC / 'FIGURES'
INPUTS = {}


def read(name):
    path = C.DOC / name
    INPUTS[name] = C.binding(path)
    return C.read(path)


def rows(name):
    path = C.DOC / name
    INPUTS[name] = C.binding(path)
    return list(C.rows(path))


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / (name + '.png'), format='png', dpi=300)
    plt.close(fig)


def panel(ax, text):
    ax.text(0.02, 1.03, text, transform=ax.transAxes, va='bottom')


def flow():
    fig, ax = plt.subplots(figsize=(7.16, 4.15))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
    def box(x, y, w, h, text, label='baseline'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.6',
            linewidth=0.7, edgecolor=color_for(label), facecolor=color_for(label),
            alpha=ALPHA['range']))
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=7.5)
    def arrow(a, b, text=None):
        ax.annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='->',
            color=SEMANTIC['annotation'], lw=.7))
        if text:
            ax.text((a[0]+b[0])/2, (a[1]+b[1])/2+2, text,
                ha='center', va='bottom', fontsize=6.5)
    box(1, 78, 28, 19, 'Original RGB + runtime K/dimensions\nFrozen YOLO / fixed N3 seeds 1–3')
    box(37, 80, 29, 15, 'G: two fitted pose candidates\nV: frozen ROI features\nD: registered dimensions')
    box(74, 80, 24, 15, 'C: eight fixed classifier fits\nSynthetic groups only', 'ours')
    arrow((29, 87), (37, 87)); arrow((66, 87), (74, 87))
    box(1, 46, 28, 20, 'D: two matched local scorers\nSame initialization / batches\n1,000 updates each', 'ours')
    box(37, 48, 29, 17, 'Current or new selector\nSame cached candidates for C\nSame solver, new corners for D')
    box(74, 46, 24, 20, 'Save predictions first\nThen score REAL173\n9 sessions; 3 UNCLEAR')
    arrow((15, 78), (15, 66)); arrow((29, 56), (37, 56))
    arrow((87, 80), (87, 66)); arrow((66, 56), (74, 56))
    box(37, 18, 29, 17, 'Reference-only diagnostics\nA: oracle / residual / noise\nG2: GT-plane feasibility', 'gt')
    arrow((51, 35), (86, 46), 'Scoring only; never a predictor input')
    box(1, 18, 28, 17, 'F: motion association absent\nG1: independent plane absent\nBLOCKED; no fabricated sequence', 'gt')
    box(74, 18, 24, 17, 'Physical FINAL146 excluded\nNo new features or scoring\nNo independent final claim', 'gt')
    ax.text(50, 4, 'C/D synthetic splits differ: compare within each locked pair; do not pool their TEST results.',
        ha='center', fontsize=7, color=SEMANTIC['annotation'])
    save(fig, '01_information_flow')


def margin():
    rr = rows('A1_PAIRED_MARGIN_RESIDUAL_ROWS.jsonl.gz')
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 3.15), sharex=True, sharey=True)
    for ax, method, p in zip(axs, ['BASE', 'N3_THEN_SUBPIX'], ['(a) Base', '(b) N3 then cornerSubPix']):
        grouped = defaultdict(list)
        for r in rr:
            if r['method'] == method: grouped[r['id']].append(r)
        points = []
        for fid, rs in sorted(grouped.items()):
            errors = [r['observed_fixed_index_corner_error']['mean'] for r in rs
                      if r['observed_fixed_index_corner_error']['mean'] is not None]
            if not errors: continue
            wrong = [r['axis_wrong'] for r in rs if r['axis_wrong'] is not None]
            points.append((fid, rs[0]['reference_margin_px'], float(np.mean(errors)),
                           None if not wrong else float(np.mean(wrong))))
        for label, predicate, marker, col in [
            ('All seeds correct', lambda v:v==0, '.', color_for('baseline')),
            ('At least one seed wrong', lambda v:v is not None and v>0, 'o', color_for('ours')),
            ('UNCLEAR axis', lambda v:v is None, 'x', color_for('gt'))]:
            pp = [r for r in points if predicate(r[3])]
            ax.scatter([r[1] for r in pp], [r[2] for r in pp], s=12 if marker=='.' else 15,
                       marker=marker, color=col, alpha=.8, linewidths=.6, label=label)
        ax.set_yscale('log'); ax.set_xlabel('Reference reprojection margin (px)')
        ax.axvline(0, color=SEMANTIC['zero'], ls=LINESTYLE['zero'], lw=.6)
        highest = max(points, key=lambda r:r[2])
        ax.annotate(highest[0].replace(':', ':\n'), xy=highest[1:3],
            xytext=(.48, .84), textcoords='axes fraction', fontsize=6,
            arrowprops=dict(arrowstyle='-', color=SEMANTIC['annotation'], lw=.5))
        panel(ax, p)
    axs[0].set_ylabel('Mean visible-corner error (px; log scale)')
    axs[1].legend(loc='lower right', fontsize=6)
    save(fig, '02_margin_and_corner_residual')


def oracle():
    obj = read('A2_ORACLE_METRICS.json')
    rr = {r['arm']:r for r in obj['summaries']
          if r['backbone']=='YOLO' and r['method']=='N3_THEN_SUBPIX'}
    arms = ['CURRENT','GT_PARITY_DIAGNOSTIC','BEST_T_POSTHOC','BEST_R_POSTHOC','BEST_SUCCESS_POSTHOC']
    labels = ['Current','GT parity','Best T','Best R','Best success']
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 3.25))
    for ax, key, ylabel, p in zip(axs, ['translation_cm','rotation_deg','success'],
            ['Translation error (cm)','Rotation error (deg)','5 cm / 5 deg success (%)'],['(a)','(b)','(c)']):
        scale = 100 if key=='success' else 1
        means = np.array([rr[a]['metrics'][key]['mean']*scale for a in arms])
        ci = np.array([rr[a]['metrics'][key]['CI95_mean_session_cluster'] for a in arms])*scale
        ax.errorbar(means, np.arange(len(arms)), xerr=[means-ci[:,0],ci[:,1]-means],
            fmt='o', color=color_for('baseline'), capsize=2, markersize=3)
        if key!='success':
            ax.scatter([rr[a]['metrics'][key]['median'] for a in arms], np.arange(len(arms)),
                marker='x', s=15, color=color_for('ours'), label='Median')
        ax.set_yticks(range(len(arms)), labels if ax is axs[0] else ['']*len(arms))
        ax.invert_yaxis(); ax.set_xlabel(ylabel); panel(ax, p)
    axs[0].legend(loc='lower right', fontsize=6)
    save(fig, '03_candidate_oracles')


def risk():
    obj = read('E_RISK_COVERAGE.json')
    labels = ['official_weighted_score_gap','score_gap_over_predicted_box_diagonal']
    titles = ['Weighted score gap','Gap / predicted\nbox diagonal']
    sources = [obj['curves'],obj['curves']]
    methods = ['N3_THEN_SUBPIX','N3_THEN_SUBPIX']
    if (C.DOC/'E_PROBABILITY_RISK.json').exists():
        sources.append(read('E_PROBABILITY_RISK.json')['curves'])
        labels.append('abs(2p_long-1)');titles.append('G+V+D probability\nconfidence')
        methods.append('G_V_D_MLP')
    fig, axs = plt.subplots(2, len(labels), figsize=(7.16, 4.75), sharex=True)
    cols = colors_for(['seed1','seed2','seed3'])
    for j, (conf, text) in enumerate(zip(labels,titles)):
        for seed, col in zip([1,2,3],cols):
            rr = next(r for r in sources[j] if r['method']==methods[j]
                      and r['seed']==seed and r['confidence']==conf)
            cov = [r['coverage'] for r in rr['curve']]
            axs[0,j].plot(cov,[r['accepted_6D_failure'] for r in rr['curve']],
                color=col,lw=.8,label=f'Seed {seed}')
            axs[1,j].plot(cov,[r['accepted_success_over_all'] for r in rr['curve']],color=col,lw=.8)
            baseline = rr['random_rejection_expected_6D_risk']
            axs[0,j].axhline(baseline,color=col,lw=.5,ls=LINESTYLE['baseline'],alpha=ALPHA['overlay'])
        panel(axs[0,j],f'({chr(97+j)}) {text}')
        panel(axs[1,j],f'({chr(97+len(labels)+j)}) All-frame success')
        axs[1,j].set_xlabel('Coverage (accepted / 173)')
        for ax in axs[:,j]: ax.set_xlim(0,1); ax.set_ylim(0,1)
    axs[0,0].set_ylabel('6D failure among accepted')
    axs[1,0].set_ylabel('Accepted successes / all 173')
    axs[0,-1].legend(loc='lower right',fontsize=6)
    save(fig,'04_risk_coverage')


def noise():
    obj = read('A3_GEOMETRY_DIAGNOSTIC.json')
    audit = read('A3_PHYSICAL_ROUNDTRIP_AUDIT.json')
    fig, axs = plt.subplots(1,3,figsize=(7.16,2.95))
    for ax, condition, xlabel, p in zip(axs[:2],['gaussian','correlated_residual'],
            ['Gaussian noise SD (px)','Correlated residual multiplier'],['(a)','(b)']):
        rr = [r for r in obj['summaries'] if condition in r['condition'].lower()]
        rr.sort(key=lambda r:r['scale'])
        for key,label,col in [('parity_error','Long/short error',color_for('ours')),
                              ('success','Raw 5 cm / 5 deg success',color_for('baseline'))]:
            ax.plot([r['scale'] for r in rr],[100*r[key] for r in rr],'-o',
                lw=.8,ms=3,label=label,color=col)
        ax.set_ylim(0,100);ax.set_xlabel(xlabel);panel(ax,p)
    axs[0].set_ylabel('Rate (%)');axs[0].legend(loc='center right',fontsize=6)
    ax=axs[2];ax.axis('off');panel(ax,'(c) Physical round-trip gate')
    text = ('Long/short parity: 16 / 16 pass\nPhysical pose: 2 / 16 fail\n\n'
            'C1 failures at exact corners:\nG38__G__f30814\nG38__G__f30131\n\n'
            'Residual = diag(-1, +1, -1)\nSigned front lost: yaw 180 deg\n'
            'Forbidden under C1\n\nRaw success is not a noise ceiling.')
    ax.text(.03,.89,text,va='top',fontsize=7,color=SEMANTIC['annotation'])
    assert len(audit['failure_ids'])==2
    save(fig,'05_noise_and_signed_front_failure')


def floor():
    obj=read('G2_PLANE_SENSITIVITY.json')
    fig,axs=plt.subplots(2,4,figsize=(7.16,4.25),sharey=True)
    cases=[('offset_cm',None,'Offset error (cm)'),('normal_deg',0,'Normal X error (deg)'),
           ('normal_deg',1,'Normal Y error (deg)'),('normal_deg',2,'Normal Z error (deg)')]
    for i,pop in enumerate(['SYNTH64','REAL61_GT_PEER_FEASIBILITY']):
        for j,(kind,axis,xlabel) in enumerate(cases):
            rr=sorted([r for r in obj['conditions'] if r['population']==pop
                and r['perturbation']==kind and r['camera_axis']==axis],key=lambda r:r['delta'])
            ax=axs[i,j]
            ax.plot([r['delta'] for r in rr],[100*r['success_rate'] for r in rr],'-o',
                color=color_for('baseline' if i==0 else 'ours'),ms=3,lw=.8)
            ax.axvline(0,color=SEMANTIC['zero'],lw=.5,ls=LINESTYLE['zero'])
            ax.set_xlabel(xlabel);ax.set_ylim(0,60)
            panel(ax,f'({chr(97+i*4+j)}) '+('SYNTH64' if i==0 else 'REAL61 GT peer'))
        axs[i,0].set_ylabel('5 cm / 5 deg success (%)')
    save(fig,'06_plane_input_sensitivity')


def primary():
    """All classifier results and the eight predeclared paired effects, when sealed."""
    metrics=read('METRICS.json');paired=read('PAIRED.json')
    wanted=[r for r in metrics['summaries'] if r['population']=='REAL173' and r['backbone']=='YOLO']
    model_names = set(read('C_RESULTS.json')['summaries'])
    names=sorted(model_names|{'OFFICIAL','D_CONTROL_SUBPIX','D_JOINT_DIM_SUBPIX'})
    lookup={r['method']:r for r in wanted}
    fig,axs=plt.subplots(1,3,figsize=(7.16,4.6),gridspec_kw=dict(width_ratios=[1,1,1.65]))
    for ax,key,label,p in zip(axs[:2],['wd_confusion','pose_success'],
                            ['W/D confusion (%)','5 cm / 5 deg success (%)'],['(a)','(b)']):
        y=np.arange(len(names));values=[lookup[n]['metrics'][key]['mean']*100 for n in names]
        ax.barh(y,values,color=colors_for(names),height=.68)
        ax.set_yticks(y,[n.replace('C_','').replace('_',' ') for n in names] if ax is axs[0] else ['']*len(names))
        ax.invert_yaxis();ax.set_xlabel(label);panel(ax,p)
    ax=axs[2]
    for i,r in enumerate(paired['results']):
        # paired.py's own exact effect schema is verified rather than guessed.
        value=r['effect_pp']
        ci=np.array(r['CI95'])
        col=color_for('ours' if r['metric']=='pose_success' else 'baseline')
        ax.errorbar(value,i,xerr=[[value-ci[0]],[ci[1]-value]],fmt='o',color=col,ms=3,lw=.8,capsize=2)
    ax.axvline(0,color=SEMANTIC['zero'],ls=LINESTYLE['zero'],lw=.7)
    ax.set_yticks(range(8),[f"{i//2+1}: {'Success' if r['metric']=='pose_success' else 'W/D'}" for i,r in enumerate(paired['results'])])
    ax.invert_yaxis();ax.set_xlabel('Paired change (percentage points)');panel(ax,'(c) Nine-session bootstrap 95% CI')
    save(fig,'07_models_and_primary_paired_effects')


def cases():
    rr=rows('C_FRAME_METRICS.jsonl.gz')
    official={r['id']:r for r in rr if r['method']=='OFFICIAL' and r['seed']==1}
    new=[r for r in rr if r['method']=='G_V_D_MLP' and r['seed']==1]
    recover=[r for r in new if r['success']==1 and official[r['id']]['success']==0]
    harm=[r for r in new if r['axis_error']==1 and official[r['id']]['axis_error']==0]
    selected=[max(recover,key=lambda r:(official[r['id']]['pose']['rotation_deg']-r['pose']['rotation_deg'],r['id'])),
              max(harm,key=lambda r:(r['pose']['rotation_deg']-official[r['id']]['pose']['rotation_deg'],r['id'])),
              max(new,key=lambda r:(official[r['id']]['pose']['translation_cm'],r['id']))]
    fig,axs=plt.subplots(1,4,figsize=(7.16,3.2))
    selected_public=[]
    for ax,r,label in zip(axs[:3],selected,['Strict recovery','Axis damage','Largest official T']):
        old=official[r['id']]
        for hyp,marker in [('long-face-front','o'),('short-face-front','s')]:
            p=r['candidates'][hyp]
            ax.scatter(p['translation_cm'],p['rotation_deg'],marker=marker,s=28,
                       color=color_for('baseline' if hyp==old['hyp'] else 'ours'))
            who=[]
            if hyp==old['hyp']:who.append('Official')
            if hyp==r['hyp']:who.append('GVD')
            ax.annotate(('Long' if hyp.startswith('long') else 'Short')+'\n'+','.join(who),
                (p['translation_cm'],p['rotation_deg']),xytext=(4,4),textcoords='offset points',fontsize=6)
        ax.axvline(5,color=SEMANTIC['threshold'],ls=LINESTYLE['threshold'],lw=.6)
        ax.axhline(5,color=SEMANTIC['threshold'],ls=LINESTYLE['threshold'],lw=.6)
        vals=[p['translation_cm'] for p in r['candidates'].values() if p.get('available')]
        ax.set_xscale('log');ax.set_xlim(min(.5,min(vals)*.5),max(12,max(vals)*4))
        ax.set_ylim(0,max(105,1.2*max(p['rotation_deg'] for p in r['candidates'].values() if p.get('available'))))
        ax.set_xlabel('Translation error (cm; log)')
        panel(ax,label)
        ax.text(.02,-.29,r['id'].replace(':',':\n'),transform=ax.transAxes,fontsize=6)
        selected_public.append(dict(label=label,id=r['id'],seed=1,official_hyp=old['hyp'],
            new_hyp=r['hyp'],official_success=old['success'],new_success=r['success'],
            official_axis_error=old['axis_error'],new_axis_error=r['axis_error'],candidate_metrics=r['candidates']))
    axs[0].set_ylabel('Rotation error (deg)')
    damage=read('D_VERIFICATION.json')['extreme_corner_transitions'][0]
    ax=axs[3]
    ax.bar([0,1],[damage['control_px'],damage['joint_px']],color=colors_for(['baseline','ours']),width=.6)
    ax.set_xticks([0,1],['Control','Joint']);ax.set_ylabel('Corner 3 error (px)')
    ax.axhline(5,color=SEMANTIC['threshold'],ls=LINESTYLE['threshold'],lw=.6)
    ax.axhline(10,color=SEMANTIC['threshold'],ls=LINESTYLE['threshold'],lw=.6)
    panel(ax,'SubPix corner damage')
    ax.text(.02,-.29,damage['id'].replace(':',':\n'),transform=ax.transAxes,fontsize=6)
    save(fig,'08_real_recovery_damage_and_tail_cases')
    C.write(C.DOC/'FIGURE_CASES.json',dict(seed_fixed_before_case_selection=1,
        selection_rules=['Largest R reduction among strict recoveries','Largest R increase among direct-axis damages',
                         'Largest official T among all frames','All detected good5-to-bad10 D corner damages'],
        classifier_cases=selected_public,D_damage=damage,raw_RGB_published=False,reference_coordinates_published=False),replace=True)


def run():
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true')
    args=parser.parse_args();start=time.perf_counter()
    flow();margin();oracle();risk();noise();floor()
    if args.final:primary();cases()
    C.write(C.DOC/'FIGURE_INPUTS.json',dict(inputs=INPUTS,contains_reference_coordinates=False,
        contains_RGB=False,raw_seed_curves_retained=True,protected_frames_processed=0,
        style=C.binding(ASSETS/'ieee.mplstyle'),palette=C.binding(ASSETS/'palette.py'),
        final_primary_figure_rendered=args.final),replace=True)
    C.write(C.DOC/'FIGURE_RECEIPT.json',dict(wall_seconds=time.perf_counter()-start,
        figures=sorted(p.name for p in FIG.glob('*.png')),format='PNG',dpi=300,
        rendered_by_this_command=sorted(p.name for p in FIG.glob('[0-9]*.png')),
        retained_other_command_figures={'D_LOCAL_METHOD.png': {'dpi':160,'producer':'d_diagram.py'}},
        output_bindings=[C.binding(p) for p in sorted(FIG.glob('*.png'))],
        code=C.binding(Path(__file__)),
        new_PnP_calls=0,new_fits=0,new_predictions=0),replace=True)
    print('FIGURES',len(list(FIG.glob('*.png'))),'seconds',time.perf_counter()-start)


if __name__=='__main__':run()
