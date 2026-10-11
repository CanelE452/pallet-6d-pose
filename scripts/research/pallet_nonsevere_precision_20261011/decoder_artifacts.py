"""Publish decoder evidence, authoritative-artifact selection, and CPU receipts."""
import argparse,gzip,json,time
from pathlib import Path
import numpy as np
import geometry as G

def load(p):
    with gzip.open(p,'rt') as f:return [json.loads(x) for x in f]

def execution_provenance(doc, drafts):
    """Describe only locally present attempts, without inventing their outcome."""
    doc=Path(doc)
    geometry_locks=[]
    for path in sorted(doc.glob('GEOMETRY*LOCK*.json')):
        value=json.loads(path.read_text())
        if value.get('status')=='LOCKED_BEFORE_GEOMETRY_ARITHMETIC':
            geometry_locks.append((path,value))
    operative_geometry=[p.name for p,v in geometry_locks if v.get('code_sha256')==G.sha(G.__file__)]
    assert len(operative_geometry)==1
    import decoder_audit as D
    decoder_locks=[]
    for path in sorted(doc.glob('DECODER*LOCK*.json')):
        value=json.loads(path.read_text())
        if value.get('status')=='LOCKED_BEFORE_DECODER_ARITHMETIC':
            decoder_locks.append((path,value))
    operative_decoder=[p.name for p,v in decoder_locks if v.get('code_sha256')==G.sha(D.__file__)]
    assert len(operative_decoder)==1
    prior_geometry=[p.name for p,v in geometry_locks if p.name!=operative_geometry[0]]
    prior_decoder=[p.name for p,v in decoder_locks if p.name!=operative_decoder[0]]
    partial=[dict(filename=r['filename'],rows=r['rows']) for r in drafts if not r['complete']]
    prior=bool(prior_geometry or prior_decoder or partial)
    return dict(status='COMPLETED_WITH_PRESERVED_PRIOR_ATTEMPTS' if prior else 'COMPLETED_CLEAN_ARITHMETIC',
        conservative_extra_draft_cpu_wall_seconds=300 if prior else 0,
        extra_draft_time_type='conservative allowance for locally preserved prior attempts, not an exact stopwatch' if prior else 'no local earlier attempt; no draft allowance invented',
        operative_geometry_lock=operative_geometry[0],operative_decoder_lock=operative_decoder[0],
        preserved_prior_geometry_locks=prior_geometry,preserved_prior_decoder_locks=prior_decoder,
        preserved_partial_decoder_rows=partial,clean_decoder_wrapper_present=(doc/'DECODER_REPLAY_LOCK.json').exists(),
        implementation_failures=[],prior_execution_outcomes_inferred_from_lock_presence=False,
        historical_original_failure_receipts='Original publication receipts remain immutable; lock presence alone is not used to fabricate failure causes in this execution')

def run(a):
    start=time.monotonic();doc=Path(a.public_dir);audit=json.loads((doc/'DECODER_AUDIT.json').read_text());authoritative=audit['complete_target_rows_artifact']
    drafts=[]
    for p in sorted(doc.glob('DECODER_FRAME_SCALARS*.jsonl.gz')):
        rows=load(p);drafts.append(dict(filename=p.name,sha256=G.sha(p),rows=len(rows),complete=len(rows)==60000,publication='INCLUDE_AUTHORITATIVE' if p.name==authoritative else 'EXCLUDE_PARTIAL_DRAFT_PRESERVE_LOCALLY',strict_json_all_rows=True))
    if not a.figure_only:G.write(doc/'PUBLIC_ARTIFACT_SELECTION.json',dict(status='LOCKED_COMPLETE_RESULTS_INCLUDED',authoritative_decoder_rows=authoritative,decoder_row_artifacts=drafts,excluded_publication=[x['filename'] for x in drafts if x['filename']!=authoritative],adverse_completed_experiment_results_excluded=False,excluded_reason='Interrupted implementation diagnostics; complete dataset and all adverse scientific results remain included',raw_rgb_published=False,raw_reference_coordinates_published=False,private_absolute_paths_published=False))
    rows=load(doc/authoritative);prob=load(doc/'DECODER_PROBABILITY_SCALARS.jsonl.gz');train=load(doc/'DECODER_TRAIN_PROBABILITY_SCALARS.jsonl.gz')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
    examples=audit['source_target_small_displacement_examples'];x=[r['desired_dx_native_px'] for r in examples];y=[r['expected_dx_native_px'] for r in examples]
    ax=axes[0,0];ax.plot([0,3.2],[0,3.2],color='gray',ls='--',label='Desired displacement');ax.plot(x,y,'o-',color='#0072B2',label='Actual target expectation')
    for xx,yy in zip(x,y):ax.annotate(f'{xx:g} -> {yy:.3f}',(xx,yy),xytext=(4,-16),textcoords='offset points')
    ax.set(xlabel='Desired native dx (px)',ylabel='Decoded native dx (px)',title='Source 222-candidate decoder\nBox diagonal 340 px');ax.legend(fontsize=8)
    good=[r for r in rows if r['desired_native_displacement_mean_px'] is not None];desired=np.array([r['desired_native_displacement_mean_px'] for r in good]);target=np.array([r['T1_gain1_uncapped_movement_mean_px'] for r in good]);capped=np.array([r['T1_gain1_capped_movement_mean_px'] for r in good])
    edges=np.r_[0,np.geomspace(.05,max(desired.max(),.051)*1.001,17)];bins=np.digitize(desired,edges)-1;bx=[];by=[];bc=[];counts=[]
    for i in range(len(edges)-1):
        m=bins==i
        if m.any():bx.append(np.median(desired[m]));by.append(np.median(target[m]));bc.append(np.median(capped[m]));counts.append(int(m.sum()))
    ax=axes[0,1];ax.plot(bx,by,'o-',label='Target expectation, uncapped',color='#0072B2');ax.plot(bx,bc,'s-',label='Target expectation, final cap',color='#D55E00');ax.plot([.03,desired.max()],[.03,desired.max()],ls='--',color='gray',lw=1);ax.set(xscale='symlog',yscale='symlog',xlabel='Initial desired displacement, frame mean (px)',ylabel='Decoded displacement, bin median (px)',title=f'All {len(good):,} supported synthetic frames; tails retained');ax.legend(fontsize=8)
    ax=axes[1,0];groups=[('TRAIN seed1, first512',train),('VAL seed1',[r for r in prob if r['seed']==1]),('VAL seed2',[r for r in prob if r['seed']==2])]
    for label,rs in groups:
        values=sorted(r['mean_expected_move_native_px'] for r in rs if r['mean_expected_move_native_px'] is not None);ax.plot(values,np.arange(1,len(values)+1)/len(values),label=f'{label} (n={len(values)})')
    ax.set(xlabel='Actual network expected displacement, frame mean (px)',ylabel='Empirical cumulative fraction',title='Actual frozen OLD_P network probabilities\nAll tails retained');ax.legend(fontsize=8)
    c=audit['corner_counts'];ax=axes[1,1];labels=['Supported','Aux eligible','Outside hull','Beyond cap'];values=[c['supported'],c['auxiliary_eligible'],c['outside_hull'],c['beyond_native_cap']];bars=ax.bar(labels,values,color=['#777777','#009E73','#D55E00','#CC79A7']);ax.set(ylabel='Synthetic corners',title='Fixed auxiliary eligibility\nExcluded corners retain original CE')
    for b,v in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,b.get_height(),f'{v:,}',ha='center',va='bottom',fontsize=8)
    figdir=doc/'FIGURES';figdir.mkdir(exist_ok=True);p=figdir/'decoder_evidence.png';assert a.figure_only or not p.exists();fig.savefig(p);plt.close(fig)
    if not a.figure_only:G.write(doc/'DECODER_FIGURE_INPUTS.json',dict(authoritative_rows=authoritative,source_rows_sha256=G.sha(doc/authoritative),examples=examples,binned_all_supported_frames=dict(desired_median_px=bx,target_uncapped_median_px=by,target_capped_median_px=bc,counts=counts),corner_counts=c,figure=p.name,raw_coordinates=False,raw_rgb=False))
    if a.figure_only:
        G.write(doc/'DECODER_FIGURE_VISUAL_REPAIR.json',dict(status='TITLE_WRAPPING_ONLY',figure=p.name,source_data_unchanged=True,raw_rgb_reads=0,new_inferences=0,new_solver_calls=0,wall_seconds=time.monotonic()-start));return
    geom=json.loads((doc/'GEOMETRY_AUDIT.json').read_text());contract=json.loads((doc/'GEOMETRY_CONTRACT_AUDIT.json').read_text())
    G.write(doc/'DECODER_EXECUTION_RECEIPT.json',dict(**execution_provenance(doc,drafts),geometry_completed_wall_seconds=geom['actual_cpu_wall_seconds'],geometry_contract_completed_wall_seconds=contract['actual_cpu_wall_seconds'],decoder_finalizer_wall_seconds=audit['actual_cpu_wall_seconds'],figure_wall_seconds=time.monotonic()-start,source_changes=0,optimizer_updates=0,new_network_inferences=0,new_pose_solver_calls=0,actual_train_forward_capture_reused=512,actual_validation_forward_cache_rows_reused=8040,publication_selection='PUBLIC_ARTIFACT_SELECTION.json',clean_replay_commands=['geometry.py --build [source/private/public arguments]','train.py calibrate [source/private/public arguments; ROOT GPU owner]','decoder_replay.py [source/private/public arguments]','decoder_completion.py [source/private/public arguments]'],new_condition_or_hyperparameter_search=False))
    print('DECODER_FIGURE_COMPLETE',p,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--public-dir',required=True);p.add_argument('--figure-only',action='store_true');run(p.parse_args())
