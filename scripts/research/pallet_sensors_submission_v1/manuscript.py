"""Build all numerical text/tables from bound JSON, compile and render real PDFs."""
import re,shutil
import numpy as np
from env import *
def put(path,text):
    path=Path(path);assert path.is_relative_to(PAPER) or path.is_relative_to(DOC)
    if path.parent==PAPER/'generated_tables' and text.startswith(r'\begin{longtable}') and r'\endhead' not in text:
        text=text.replace(r'\midrule',r'\midrule\endhead',1)
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
def tex(value):return str(value).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
def fmt(x,n=3):return 'NM' if x is None else f'{x:.{n}f}'
def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dst=PAPER/'figures';dst.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(7,3.6));ax.set_xlim(0,10);ax.set_ylim(0,6);ax.axis('off')
    def box(x,y,label,color='#edf2f7'):
        ax.text(x,y,label,ha='center',va='center',fontsize=10,bbox=dict(boxstyle='round,pad=.45',facecolor=color,edgecolor='#374151'))
    def arrow(a,b,**kwargs):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#374151',**kwargs))
    box(1,3.6,'RGB image');box(3.4,3.6,'Frozen R0\npoints + box');box(6.5,3.6,'P local\nrefinement','#dbeafe');box(9,3.6,'PnP\npose')
    arrow((1.9,3.6),(2.35,3.6));arrow((4.5,3.6),(5.55,3.6));arrow((7.5,3.6),(8.4,3.6))
    box(4.8,5.2,'Reused P3/P4 features');arrow((3.4,4.3),(4.1,4.85));arrow((5.65,4.85),(6.5,4.3))
    box(6.5,1.5,'Synthetic supervision\n(training only)','#fef3c7');arrow((6.5,2.2),(6.5,2.9),linestyle='dashed')
    box(9,1.4,'Intrinsics\nknown dimensions');arrow((9,2.2),(9,3))
    ax.text(.3,.1,'Preserved: boxes, scores, selection, center8.  Updated: supported corners0–7.',fontsize=9)
    fig.tight_layout();fig.savefig(dst/'pipeline.pdf',bbox_inches='tight');fig.savefig(dst/'graphical_abstract.png',dpi=220,bbox_inches='tight');plt.close(fig)
    shutil.copyfile(ROOT/'_docs/paper/sensors_refinement_closeout_v1/FIG_SESSION_EFFECTS.pdf',dst/'session_effects.pdf')
    shutil.copyfile(ROOT/'_docs/paper/sensors_refinement_closeout_v1/FIG_D_PROBES.pdf',dst/'D_probes.pdf')
    if (DOC/'TRAINING_AUDIT.json').exists():
        train=read(DOC/'TRAINING_AUDIT.json');fig,axes=plt.subplots(1,2,figsize=(8,3))
        for r in train['runs']:
            for ax,part in zip(axes,('train','cal')):
                ax.plot([p['step'] for p in r['probes']],[p[part]['heatmap']+p[part]['coordinate'] for p in r['probes']],label='seed'+str(r['seed']));ax.set_title('Fixed '+part+' probe');ax.set_xlabel('Optimizer updates');ax.set_ylabel('CE + coordinate L1');ax.legend()
        fig.tight_layout();fig.savefig(dst/'prior_probes.pdf');plt.close(fig)
    manifest=read(PAPER/'FIGURE_MANIFEST.json')
    for name,paths,purpose,limitation in [
        ('pipeline.pdf',[HERE/'manuscript.py',ROOT/'scripts/research/pallet_final_ml_contribution_test_v1/generic_point_refiner.py'], 'Explain frozen detector, learned refiner, training-only supervision and geometry inputs', 'Architecture diagram; not measured accuracy or proof of harmless geometry'),
        ('graphical_abstract.png',[HERE/'manuscript.py'], 'Reuse the implemented pipeline as the graphical abstract', 'Internal-draft diagram; no independent-confirmation or safety claim'),
        ('session_effects.pdf',[ROOT/'_docs/paper/sensors_refinement_closeout_v1/FIG_SESSION_EFFECTS.pdf',OLD_DOC/'P_VS_R0_PAIRED.json'], 'Show all observed P-minus-R0 session effects, including the harmful session', 'Reused DEV descriptive session effects, not independent test intervals'),
        ('D_probes.pdf',[ROOT/'_docs/paper/sensors_refinement_closeout_v1/FIG_D_PROBES.pdf',OLD_DOC/'D1_COMPLETION.json'], 'Retain existing direct-control probe behavior', 'Not convergence certification; losses are not comparable across objectives')]:
        manifest.append(dict(figure=name,output=bound(dst/name),sources=[bound(p) for p in paths],purpose=purpose,limitation=limitation))
    if (DOC/'TRAINING_AUDIT.json').exists():
        manifest.append(dict(figure='prior_probes.pdf',output=bound(dst/'prior_probes.pdf'),sources=[bound(DOC/'TRAINING_AUDIT.json')],purpose='Show fixed train/cal probes for all completed prior seeds',limitation='Diagnostic only, no checkpoint selection or convergence guarantee'))
    write(PAPER/'FIGURE_MANIFEST.json',manifest)

def generate():
    (PAPER/'generated_tables').mkdir(parents=True,exist_ok=True)
    source=DOC/'UNIFIED_DEV_RESULTS.json' if (DOC/'UNIFIED_DEV_RESULTS.json').exists() else OLD_DOC/'UNIFIED_DEV_RESULTS.json'
    results=read(source)['methods'];pmean=read(OLD_DOC/'SEED_MEAN_RESULTS.json')['P'];pr=read(OLD_DOC/'P_VS_R0_PAIRED.json')['session'];pd=read(OLD_DOC/'P_VS_D_PAIRED.json')['session'];pop=read(OLD_DOC/'POPULATION_AND_METRIC_LOCK.json')
    rtpath=DOC/'RUNTIME_PANEL.json' if (DOC/'RUNTIME_PANEL.json').exists() else OLD_DOC/'RUNTIME_PANEL.json';runtime=read(rtpath);rts=runtime['summary']
    prior_ready=(DOC/'TRAINING_AUDIT.json').exists() and read(DOC/'TRAINING_AUDIT.json').get('complete',False)
    evaluated='PRIOR1' in results
    confirmation=(DOC/'CONFIRMATION_RESULTS.json').exists()
    values=dict(Pparams=f"{runtime['params']['P1']['refiner_trainable']:,}",Dparams=f"{runtime['params']['D1']['refiner_trainable']:,}",DevN=pop['positive'],NegN=pop['negative'],SessionN=len(pop['sessions']),MatchedN=pop['matched'],MatchedPoints=f"{pop['matched_supervised']:,}",AllPoints=f"{pop['total_supervised']:,}",TrainN=f"{pop['synthetic_partitions']['train']:,}",UsableTrainN=f"{pop['synthetic_matched_train']:,}",CalN=f"{pop['synthetic_partitions']['calibration']:,}",SelectionN=f"{pop['synthetic_partitions']['selection']:,}",HeldoutN=f"{pop['synthetic_partitions']['heldout']:,}",Rmedian=fmt(results['R0']['median_px']),Pmedian=fmt(pmean['median_px']),Improve=fmt(-pr['delta']),ImproveLow=fmt(-pr['high']),ImproveHigh=fmt(-pr['low']),RPCK=fmt(results['R0']['ALL_GT_PCK']['10']*100),PPCK=fmt(pmean['ALL_GT_PCK']['10']*100),PDdelta=fmt(pd['delta']),PDlow=fmt(pd['low']),PDhigh=fmt(pd['high']),TranslationImprove=fmt((results['R0']['pose']['translation_median_cm']-pmean['pose']['translation_median_cm'])*10),Ptime=fmt(rts['P1']['image_to_2d_ms']['median']),Pfulltime=fmt(rts['P1']['end_to_end_ms']['median']),Rtime=fmt(rts['R0']['image_to_2d_ms']['median']),Rfulltime=fmt(rts['R0']['end_to_end_ms']['median']))
    prior_abstract='The fixed-budget prior experiment is complete; it tests matched exposure rather than full-method convergence.' if evaluated else 'The prior comparison is not yet complete and no prior performance is inferred.'
    if evaluated:
        contrast=read(DOC/'P_VS_PRIOR_PAIRED.json')
        if 'session' in contrast:
            q=contrast['session'];direction='The interval favors the prior, not P.' if q['low']>0 else 'The interval favors P.' if q['high']<0 else 'The interval includes zero and does not establish equivalence.'
            prior_abstract=f"The P--prior median difference is {fmt(q['delta'])} pixels (paired-session interval [{fmt(q['low'])},{fmt(q['high'])}]). {direction} This is a fixed-budget development comparison."
    values.update(PriorAbstract=prior_abstract,PriorTrainingStatus='All three matched-budget fits were completed; the learning curves are diagnostic, not a best-checkpoint selector.' if prior_ready else 'The approved fits are pending or in progress; this draft does not represent them as completed.',ConfirmationAbstract='Independent confirmation is reported separately under its frozen protocol.' if confirmation else 'Independent real-session confirmation is not yet available.',ConfirmationProtocol='Independent data are evaluated only after the complete model panel, capture lineage, and blinded annotation QA have been frozen. The supplied confirmation results are separate from DEV.' if confirmation else 'Independent confirmation is awaiting new data. The collection plan starts with 12 separate sessions and 15 prespecified frames per session, with at least 30 frames independently double annotated before blind adjudication. This is not a power guarantee. Predictions are hidden during annotation and the full panel must be frozen before evaluation. Existing FINAL-named manifests have unavailable membership and are not treated as independent observations.',ConclusionPrior='The source-derived prior comparison quantifies a fixed-budget tradeoff, not converged-method superiority.' if evaluated else 'No conclusion against a formally trained prior is drawn until its execution and evaluation finish.',ConfirmationConclusion='The independently collected confirmation evidence is reported separately with its limitations.' if confirmation else 'Independent real-session confirmation remains required.')
    values['RuntimePanelStatus']='The expanded panel was measured in the same session for all comparators.' if rtpath.parent==DOC else 'The expanded panel including the prior is deferred because a separate GPU training task was active. Only the previous R0/P/D/L session is reported here as historical timing; prior latency and a same-session prior speed comparison are not yet measured.'
    put(PAPER/'generated_tables/numbers.tex','\n'.join('\\newcommand{\\'+k+'}{'+str(v)+'}' for k,v in values.items())+'\n')
    # Main rows are seed means, never ensemble outputs; raw and wrapped prior distinguished.
    families=['P','D','L']+(['PRIOR'] if evaluated else [])
    def mean(fam,field,sub=None):return float(np.mean([results[f'{fam}{s}'][field][sub] if sub else results[f'{fam}{s}'][field] for s in (1,2,3)]))
    rows=[['R0',fmt(results['R0']['median_px']),fmt(results['R0']['p90_px']),fmt(results['R0']['ALL_GT_PCK']['10']*100,2),fmt(results['R0']['pose']['translation_median_cm'])]]
    for fam in families:rows.append([fam+' mean',fmt(mean(fam,'median_px')),fmt(mean(fam,'p90_px')),fmt(mean(fam,'ALL_GT_PCK','10')*100,2),fmt(mean(fam,'pose','translation_median_cm'))])
    if not evaluated:rows.append(['Prior','NM','NM','NM','NM'])
    accuracy=r'''\begin{table}[t]\centering\caption{Reused DEV: seed-mean statistics, not an ensemble. Translation is against the geometry reference. NM means not measured in this build.}\label{tab:accuracy}
\small\begin{tabular}{lrrrr}\toprule
Model & Median & P90 & PCK10 & Trans.\\
 & px & px & \% & cm\\\midrule
'''+''.join(' & '.join(r)+r' \\'+'\n' for r in rows)+r'\bottomrule\end{tabular}\end{table}'+'\n'
    put(PAPER/'generated_tables/accuracy.tex',accuracy)
    prior_text='The actual prior DEV comparison has not yet completed in this build. Its result cells are not filled using D, a scaffold, or a proxy network. The implementation and resource receipts distinguish completed validation from any remaining training or evaluation.\n'
    if evaluated:
        q=read(DOC/'P_VS_PRIOR_PAIRED.json');sel=read(DOC/'PRIOR_SELECTION.json')['selected_rule'];prior_text='The prior checkpoints were fixed at the final training update; only the wrapper rule was selected on synthetic data. The selected wrapper has lambda '+fmt(sel['lam'])+' and cap fraction '+str(sel['max_move_image_diagonal_fraction'])+'. Raw and wrapped outputs are separate supplementary rows. '
        if q['common_support_exact']:
            prior_text+='All wrapped prior seeds retain the baseline matched supervision support, with no nonfinite supervised coordinates. '
        if 'session' in q:
            prior_text+=f"The P--prior paired median difference is {fmt(q['session']['delta'])} px, with session interval [{fmt(q['session']['low'])}, {fmt(q['session']['high'])}] px. "
            prior_text+='This interval favors the prior, so P is not the most accurate comparator under this central-error criterion. ' if q['session']['low']>0 else 'This interval favors P under this central-error criterion. ' if q['session']['high']<0 else 'The interval includes zero; neither equivalence nor a direction of improvement is established. '
        else:prior_text+='Precision supports differ, so no favorable-intersection primary contrast is substituted. '
        prior_text+='A negative contrast favors P; a positive contrast favors the prior. The complete-denominator PCK, tail and pose rows must be considered alongside the median. Six thousand updates per seed do not certify saturation or reproduce the original 140-epoch protocol.\n'
    put(PAPER/'generated_tables/prior_results.tex',tex(prior_text))
    rr=[]
    for name in ('R0','P1','D1','L1','PRIOR1'):
        a=rts.get(name);rr.append([name]+([fmt(a[k]['median']) for k in ('image_to_2d_ms','pnp_ms','end_to_end_ms')] if a else ['NM']*3))
    runtime_tex=r'''\begin{table}[t]\centering\caption{Desktop median latency in ms, representative seed1. Training/scoring excluded from timing; all repeats retained.}\label{tab:runtime}\small
\begin{tabular}{lrrr}\toprule Model & BGR-to-2D & PnP & Total\\\midrule
'''+''.join(' & '.join(r)+r' \\'+'\n' for r in rr)+r'\bottomrule\end{tabular}\end{table}'+'\n'
    if rtpath.parent!=DOC:
        runtime_tex=runtime_tex.replace('Desktop median latency in ms, representative seed1.', 'Historical R0/P/D/L desktop median latency in ms, representative seed1. Expanded panel deferred; prior NM.')
    put(PAPER/'generated_tables/runtime.tex',runtime_tex)
    confirm_text='Not yet measured: no independently reviewed capture/annotation panel is available. The proposed collection count is a plan, not a completed experiment. Development observations are not relabeled as confirmation results.\n'
    if confirmation:
        c=read(DOC/'CONFIRMATION_RESULTS.json');q=c['P_R0'];confirm_text=f"The independent 2D panel has {c['metrics']['R0']['total_frames']} frames. The P--R0 median difference is {fmt(q['delta'])} px, interval [{fmt(q['low'])}, {fmt(q['high'])}]. New-frame pose status is {c['pose_status']}. No independent physical 6D measurement is inferred.\n"
    put(PAPER/'generated_tables/confirmation_results.tex',tex(confirm_text))
    # Full per-model tables, not just the headline contrasts.
    head=r'\begin{longtable}{lrrrrrrrr}\toprule Model & Med & P90 & Mean & Gross20 & PCK5 & PCK10 & PCK20 & N\\\midrule'+'\n'
    table=head
    for name,a in results.items():
        table+=' & '.join([tex(name),fmt(a['median_px']),fmt(a['p90_px']),fmt(a['frame_mean_px']),fmt(a['gross20']*100,2)]+[fmt(a['ALL_GT_PCK'][str(t)]*100,2) for t in (5,10,20)]+[str(a['supervised_points'])])+r' \\'+'\n'
    table+=r'\bottomrule\end{longtable}'+'\n';put(PAPER/'generated_tables/full_accuracy.tex',table)
    table=r'\begin{longtable}{lrrrrrr}\toprule Model & Rot(deg) & Yaw(deg) & Trans(cm) & IoU3D & ADD AUC & Coverage\\\midrule'+'\n'
    for name,a in results.items():
        p=a['pose'];table+=' & '.join([tex(name)]+[fmt(p[k]) for k in ('rotation_median_deg','yaw_median_deg','translation_median_cm','iou3d_median','add_sym_auc','coverage')])+r' \\'+'\n'
    table+=r'\bottomrule\end{longtable}'+'\n';put(PAPER/'generated_tables/full_pose.tex',table)
    table=r'\begin{longtable}{lrrrrrrr}\toprule Model & 2D med & mean & P90 & Total med & mean & P90 & Add2D med\\\midrule'+'\n'
    for name,a in rts.items():table+=' & '.join([tex(name)]+[fmt(a[k][v]) for k in ('image_to_2d_ms','end_to_end_ms') for v in ('median','mean','p90')]+[fmt(a['paired_added_2d_ms']['median'])])+r' \\'+'\n'
    table+=r'\bottomrule\end{longtable}'+'\n';put(PAPER/'generated_tables/full_runtime.tex',table)
    write(PAPER/'NUMBER_SOURCES.json',dict(sources=[bound(p) for p in (source,OLD_DOC/'P_VS_R0_PAIRED.json',OLD_DOC/'P_VS_D_PAIRED.json',OLD_DOC/'SEED_MEAN_RESULTS.json',OLD_DOC/'POPULATION_AND_METRIC_LOCK.json',rtpath)],macro_values=values,prior_evaluated=evaluated,confirmation_evaluated=confirmation,units='px,cm,mm,deg,ms; PCK fractions multiplied by100 exactly once',seed_policy='per-seed statistic then mean, not ensemble'))
    from paper_assets import run as assets
    assets(results,runtime,source,rtpath)
    from numbers_manifest import run as inventory
    inventory(source,rtpath,values)
    return evaluated,confirmation

def run():
    previous=read(DOC/'PDF_CHECKS.json') if (DOC/'PDF_CHECKS.json').exists() else {}
    prior_exists=(DOC/'UNIFIED_DEV_RESULTS.json').exists()
    trained=(DOC/'TRAINING_AUDIT.json').exists() and read(DOC/'TRAINING_AUDIT.json').get('complete',False)
    confirmation_exists=(DOC/'CONFIRMATION_RESULTS.json').exists()
    try:
        if complete('MANUSCRIPT_COMPLETE') and previous.get('prior_evaluated')==prior_exists and previous.get('prior_training_complete')==trained and previous.get('independent_gap_declared')==(not confirmation_exists):
            for b in read(PAPER/'NUMBER_SOURCES.json')['sources']:assert sha(ROOT/b['path'])==b['sha256']
            verify();print('MANUSCRIPT_REUSED_HASH_MATCH',flush=True);return
    except (AssertionError,OSError):pass
    start=now();verify();evaluated,confirmation=generate();figures()
    compiler=RAW/'tectonic';assert compiler.exists(),'Fetch official standalone Tectonic release into task raw directory'
    cache=RAW/'tex_cache';cache.mkdir(exist_ok=True)
    for name in ('manuscript','supplementary'):
        p=subprocess.run([str(compiler),'--keep-logs','--keep-intermediates',str(PAPER/(name+'.tex'))],cwd=PAPER,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env={**os.environ,'XDG_CACHE_HOME':str(cache)})
        put(DOC/(name.upper()+'_BUILD.log'),p.stdout)
        print(p.stdout[-3000:],flush=True);assert p.returncode==0,(name,p.returncode)
        pdf=PAPER/(name+'.pdf');assert pdf.exists()
        renders=RAW/'pdf_render';renders.mkdir(exist_ok=True)
        subprocess.run(['pdftoppm','-r','100','-png',str(pdf),str(renders/name)],check=True)
        text=subprocess.check_output(['pdftotext','-layout',str(pdf),'-'],text=True)
        for token in ('??','TODO','Lorem ipsum','PLACEHOLDER'):assert token not in text,(name,token)
        put(DOC/(name.upper()+'_EXTRACTED.txt'),text)
    write(DOC/'PDF_CHECKS.json',dict(compile=True,rendered=True,automated_text_checks=True,visual_review='PENDING_AGENT_PAGE_INSPECTION',sources=[bound(PAPER/f) for f in ('manuscript.tex','supplementary.tex','refs.bib','manuscript.pdf','supplementary.pdf')],independent_gap_declared=not confirmation,prior_evaluated=evaluated,prior_training_complete=trained))
    receipt('MANUSCRIPT_COMPLETE',[HERE/'manuscript.py',HERE/'paper_assets.py',HERE/'numbers_manifest.py',PAPER/'manuscript.tex',PAPER/'supplementary.tex',PAPER/'refs.bib',PAPER/'NUMBER_SOURCES.json',PAPER/'NUMBERS_MANIFEST.json',PAPER/'FIGURE_MANIFEST.json'],[PAPER/'manuscript.pdf',PAPER/'supplementary.pdf'],start,status='FULL_DRAFT_WITH_DECLARED_GAPS',not_submitted=True)
    print('PDF_COMPILED_AND_RENDERED_REQUIRES_VISUAL_REVIEW',flush=True)
