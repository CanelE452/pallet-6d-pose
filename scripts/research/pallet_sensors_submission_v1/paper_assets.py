"""Exact coordinate diagrams, declared case selection and measured resource tables."""
import numpy as np
from env import *

def run(results,runtime,source,rtpath):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from generic_point_refiner import GenericPointRefiner
    manifest=[];dst=PAPER/'figures';dst.mkdir(exist_ok=True)
    # Draw the real candidate lattice; schematic weights are explicitly not data.
    config=read(B/'P_SEED1_COMPLETION.json')['config'];model=GenericPointRefiner(**config)
    offsets=model.displacements.numpy();stencil=model.stencil.numpy()*model.stencil_fraction
    fig,ax=plt.subplots(figsize=(4.6,3.7))
    ax.scatter(*offsets[:-1].T,s=5,c='#64748b',label='221 candidates')
    ax.scatter(0,0,c='black',marker='+',s=70,label='input / null')
    k=105;ax.scatter(*(offsets[k]+stencil).T,s=9,facecolors='none',edgecolors='#2563eb',label='one actual 4 x 8 stencil')
    ax.scatter(*offsets[k],s=35,c='#2563eb');ax.set_aspect('equal');ax.set_xlabel('x displacement / box diagonal');ax.set_ylabel('y displacement / box diagonal');ax.legend(fontsize=7,loc='upper left');fig.tight_layout();fig.savefig(dst/'candidate_sampling.pdf');plt.close(fig)
    manifest.append(dict(figure='candidate_sampling.pdf',source=bound(ROOT/'scripts/research/pallet_final_ml_contribution_test_v1/generic_point_refiner.py'),config=bound(B/'P_SEED1_COMPLETION.json'),purpose='Explain actual candidate/stencil geometry, not invent a measured probability map',candidate_index=k,weights='not plotted; expectation is the weighted sum in Eq1'))
    if 'PRIOR1' in results and all(f'PRIOR{s}' in runtime['summary'] for s in (1,2,3)):
        fig,ax=plt.subplots(figsize=(4.5,3.5))
        for fam,color in [('R0','#6b7280'),('P','#2563eb'),('D','#f59e0b'),('L','#059669'),('PRIOR','#dc2626')]:
            names=['R0'] if fam=='R0' else [f'{fam}{s}' for s in (1,2,3)]
            x=[runtime['summary'][n]['end_to_end_ms']['median'] for n in names];y=[results[n]['median_px'] for n in names]
            ax.scatter(x,y,color=color,label=fam,s=28)
        ax.set_xlabel('BGR-to-pose median latency (ms)');ax.set_ylabel('DEV pooled median error (px)');ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(dst/'accuracy_latency.pdf');plt.close(fig)
        manifest.append(dict(figure='accuracy_latency.pdf',sources=[bound(source),bound(rtpath)],purpose='Observed per-seed accuracy-cost tradeoff, not an ensemble or unseen-session guarantee',x='summary.MODEL.end_to_end_ms.median',y='methods.MODEL.median_px'))
    # Rights-safe coordinate panels: actual GT and predictions, no private RGB export.
    pe=old('paper_evaluation');pop=pe.population();paths={'R0':LINE/'baseline/FULL_CANDIDATES.json','P1':BRAW/'evaluation/P1/PREDICTIONS.json','D1':OLD_RAW/'evaluation/D1/PREDICTIONS.json'}
    if 'PRIOR1' in results:paths['PRIOR1']=RAW/'evaluation/PRIOR1/PREDICTIONS.json'
    predictions={n:read(p)['frames'] for n,p in paths.items()};cases=[]
    for item in pop.positive.items:
        key=pe.canonical_key(item.image);gt=pe.E._legacy_forbidden_target(item);m=gt.keypoint_supervision_mask
        pp={n:max(f[key],key=lambda c:c['score']) if f[key] else None for n,f in predictions.items()}
        if any(p is None or p['keypoints_xy'] is None for p in pp.values()):continue
        if pe.E._box_iou(np.asarray(pp['R0']['box_xyxy']),gt.box_xyxy)<.5:continue
        delta=float(np.median(np.linalg.norm(np.asarray(pp['P1']['keypoints_xy'])[m]-gt.keypoints_xy[m],axis=-1))-np.median(np.linalg.norm(np.asarray(pp['R0']['keypoints_xy'])[m]-gt.keypoints_xy[m],axis=-1)))
        cases.append((delta,item.frame_id,key,gt,pp))
    cases.sort(key=lambda r:(r[0],r[1]));chosen=[cases[0],cases[-1]]
    fig,axes=plt.subplots(2,len(paths),figsize=(3*len(paths),6),squeeze=False)
    members=[]
    for row,(delta,fid,key,gt,pp) in enumerate(chosen):
        allxy=np.concatenate([np.asarray(p['keypoints_xy']) for p in pp.values()]+[gt.keypoints_xy[gt.keypoint_supervision_mask]])
        lo=allxy.min(0)-12;hi=allxy.max(0)+12
        for ax,(name,p) in zip(axes[row],pp.items()):
            xy=np.asarray(p['keypoints_xy']);mask=gt.keypoint_supervision_mask;target=gt.keypoints_xy
            ax.scatter(*target[mask].T,c='black',marker='x',s=23,label='GT');ax.scatter(*xy[mask].T,facecolors='none',edgecolors='#2563eb',s=28,label=name)
            for k in np.flatnonzero(mask):ax.plot([target[k,0],xy[k,0]],[target[k,1],xy[k,1]],color='#9ca3af',lw=.7)
            ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1]);ax.set_aspect('equal');ax.set_title(name);ax.set_xlabel('original x (px)');ax.set_ylabel('original y (px)');ax.legend(fontsize=6)
        axes[row,0].text(0,1.14,('Largest gain' if row==0 else 'Largest harm')+f': P1-R0 {delta:+.2f}px',transform=axes[row,0].transAxes,fontsize=8)
        members.append(dict(frame_id=fid,image_key=key,delta_frame_median_px=delta))
    fig.tight_layout();fig.savefig(dst/'qualitative_coordinates.pdf');plt.close(fig)
    manifest.append(dict(figure='qualitative_coordinates.pdf',sources={n:bound(p) for n,p in paths.items()},population='DEV matched positive images',selection='min and max of frame median(P1)-frame median(R0); lexical frame_id tie; not primary pooled statistic',members=members,purpose='Retain actual improving and harming examples with all required models; no coordinates edited',limitation='Coordinate-only; no raw RGB published because image redistribution permission is unverified. Does not establish visual boundary correctness.'))
    write(PAPER/'FIGURE_MANIFEST.json',manifest)
    resource_table(runtime,rtpath)
    measurement_table()

def measurement_table():
    from collections import Counter
    contract_path=C.POSE/'POSE_EVAL_OBJECT_CONTRACT.json';contract=read(contract_path);manifest_path=C.POSE/'AXIS_REVIEW_MANIFEST.json';frames=read(manifest_path)['frames_list'];groups=Counter();sources=[]
    for f in frames:
        path=ROOT/f['annotation'];camera=read(path)['camera_data'];k=camera['intrinsics'];groups[(camera['width'],camera['height'],*(k[n] for n in ('fx','fy','cx','cy')))]+=1;sources.append(bound(path))
    values=[dict(width=k[0],height=k[1],fx=k[2],fy=k[3],cx=k[4],cy=k[5],frames=n) for k,n in sorted(groups.items())]
    objects={k:v for k,v in contract.items() if isinstance(v,dict) and 'physical_dimensions_m' in v}
    write(DOC/'MEASUREMENT_INPUT_AUDIT.json',dict(contract=bound(contract_path),frame_manifest=bound(manifest_path),annotations=sources,intrinsics_groups=values,objects=objects,population=contract['population'],camera_fields_only=['width','height','intrinsics'],distortion='canonical solver passes None; no newly estimated distortion or independently verified calibration uncertainty',reference_claim='geometry-reconstructed from manual9-point annotations under declared dimensions/axes; no independent physical inspection or6D instrument is established by the object contract',symmetry='declared long-axis0/180 equivalence;90/270 remain distinct; wood visual180symmetry is unverified'))
    table=r'\begin{longtable}{lrrrr}\toprule Object & Long(m) & Short(m) & Height(m) & Frames\\\midrule'+'\n'
    for name,o in objects.items():
        d=o['physical_dimensions_m'];label='Plastic' if name.startswith('plastic') else 'Wood'
        table+=' & '.join([label]+[f"{d[k]:.2f}" for k in ('long','short','height')]+[str(contract['population'][name])])+r' \\'+'\n'
    table+=r'\bottomrule\end{longtable}'+'\n'+r'\begin{longtable}{rrrrrrr}\toprule Width & Height & $f_x$ & $f_y$ & $c_x$ & $c_y$ & Frames\\\midrule'+'\n'
    for g in values:table+=' & '.join([str(g['width']),str(g['height'])]+[f"{g[k]:.3f}" for k in ('fx','fy','cx','cy')]+[str(g['frames'])])+r' \\'+'\n'
    table+=r'\bottomrule\end{longtable}'+'\n'
    table+='Intrinsics above are annotation records in pixels, not newly measured calibration accuracy. The canonical solver passes no distortion coefficients. The records inspected here provide resolution and intrinsics, not an independently verified lens/calibration uncertainty report. Object dimensions are the unchanged registered geometry; the evaluation contract does not claim new physical inspection. The declared long-axis equivalence allows a half turn, but not a quarter turn. The wood object is not established as visually symmetric under the half turn. These benchmark conventions do not certify interchangeable physical fork-entry geometry.\n'
    (PAPER/'generated_tables/measurement_inputs.tex').write_text(table)

def resource_table(runtime,rtpath):
    params=runtime['params'];cache=read(LINE/'cache/CACHE_COMPLETE.json');arrays=read(DOC/'SOURCE_BINDING.json')['cache_arrays'];rows=[]
    for name in ('R0','P1','D1','L1','PRIOR1'):
        p=params.get(name);parameter_source=rtpath
        if p is None and name=='PRIOR1' and (DOC/'PRIOR1_COMPLETE.json').exists():
            parameter_source=DOC/'PRIOR1_COMPLETE.json';count=read(parameter_source)['parameters'];p=dict(refiner_trainable=count,total=params['R0']['total']+count)
        record=dict(model=name,parameter_source=bound(parameter_source),real_refiner_training=0,input='RGB / predicted pose' if name=='PRIOR1' else 'reused R0 P3/P4 / predicted pose' if name!='R0' else 'RGB',extra_pretraining='ImageNet classification' if name=='PRIOR1' else 'none beyond R0',known_geometry='PnP only, not refiner input',trainable_parameters=p['refiner_trainable'] if p else None,total_parameters=p['total'] if p else None,shared_R0_parameters=params['R0']['total'],additional_parameter_ratio=p['refiner_trainable']/params['R0']['total'] if p else None,inference_peak_MiB=runtime['isolated_memory'].get(name,{}).get('peak_allocated_bytes',0)/2**20 if name in runtime['isolated_memory'] else None,train_peak_MiB=None,seconds_per_update_including_probes_checkpointing=None)
        if name=='PRIOR1' and (DOC/'PRIOR1_COMPLETE.json').exists():
            t=read(DOC/'PRIOR1_COMPLETE.json');record.update(train_peak_MiB=t['peak_allocated_MiB'],seconds_per_update_including_probes_checkpointing=t['elapsed_seconds']/6000,training_source=bound(DOC/'PRIOR1_COMPLETE.json'))
        elif name in ('P1','D1'):
            path=B/'P_SEED1_COMPLETION.json' if name=='P1' else OLD_DOC/'D1_COMPLETION.json';t=read(path);record.update(seconds_per_update_including_probes_checkpointing=t['elapsed_seconds']/6000,training_source=bound(path))
        rows.append(record)
    write(DOC/'RESOURCE_COMPARISON.json',dict(models=rows,runtime_source=bound(rtpath),runtime_threads=runtime.get('threads'),cpu_hardware=runtime.get('cpu_hardware'),cache_source=bound(LINE/'cache/CACHE_COMPLETE.json'),cache_bytes=sum(r['bytes'] for r in arrays),first_cache_extraction_seconds=cache['elapsed_seconds_this_process'],first_cache_rows=cache['rows_extracted_this_process'],cache_reused=True,scope='P/D/L head fits use precomputed features; add one-time cache extraction. Prior consumes actual RGB plus cached initial pose. Old missing allocated training peaks remain unmeasured, not zero.',R0_pretraining=bound(DOC/'R0_PRETRAINING_PROVENANCE.json')))
    f=lambda x,n=2:'NM' if x is None else f'{x:.{n}f}'
    table=r'\begin{longtable}{lrrrrrr}\toprule Model & Learned & Total & Add/R0(\%) & Inf.MiB & Train MiB & s/update\\\midrule'+'\n'
    for r in rows:table+=' & '.join([r['model'],str(r['trainable_parameters']) if r['trainable_parameters'] is not None else 'NM',str(r['total_parameters']) if r['total_parameters'] is not None else 'NM',f(r['additional_parameter_ratio']*100 if r['additional_parameter_ratio'] is not None else None),f(r['inference_peak_MiB']),f(r['train_peak_MiB']),f(r['seconds_per_update_including_probes_checkpointing'],4)])+r' \\'+'\n'
    table+=r'\bottomrule\end{longtable}'+'\n'
    table+=f"The shared R0 has {params['R0']['total']:,} parameters. The recorded first cache pass processed {cache['rows_extracted_this_process']:,} records in {cache['elapsed_seconds_this_process']:.2f} seconds; bound cache arrays occupy {sum(r['bytes'] for r in arrays)/2**30:.3f} GiB. Head fit time excludes that extraction. NM indicates an unavailable historical allocation measurement, not zero cost. R0 is not newly trained.\n"
    if 'threads' in runtime:
        t=runtime['threads'];cpu=next((x['data'] for x in runtime['cpu_hardware']['lscpu'] if x['field']=='Model name:'),'see runtime hardware record')
        table+=f"CPU: {cpu}. Timed code fixes Torch intra-operation threads to {t['torch_intraop']}. "
        if 'memory_phase_threads' in runtime:
            z=runtime['memory_phase_threads']['R0'];table+=f"Inter-operation and OpenCV thread observations were not durably captured in the timed phase. In the separate fresh-process memory phase, observed settings were Torch intra-operation {z['torch_intraop']}, inter-operation {z['torch_interop']}, OpenCV {z['opencv']} (R0 process; all models are in the runtime JSON). These are not retrospectively labeled timed-phase observations.\n"
        else:table+=f"Recorded inter-operation {t['torch_interop']}, OpenCV {t['opencv']}. These are library settings, not a universal process thread limit.\n"
    (PAPER/'generated_tables/resources.tex').write_text(table)

if __name__=='__main__':
    p=DOC/'UNIFIED_DEV_RESULTS.json' if (DOC/'UNIFIED_DEV_RESULTS.json').exists() else OLD_DOC/'UNIFIED_DEV_RESULTS.json';rt=DOC/'RUNTIME_PANEL.json' if (DOC/'RUNTIME_PANEL.json').exists() else OLD_DOC/'RUNTIME_PANEL.json'
    run(read(p)['methods'],read(rt),p,rt)
