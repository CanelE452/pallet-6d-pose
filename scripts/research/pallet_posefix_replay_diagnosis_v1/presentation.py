"""Deterministic figures and complete pass tables, not a model selector."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scripts.research.pallet_posefix_replay_diagnosis_v1 import run as R
from analysis import BINS

def fmt(x,d=3): return '—' if x is None else f'{x:.{d}f}'

def build():
    real=R.read(R.DOC/'PASS_METRICS_REAL_DEV.json')['splits']
    synth=R.read(R.DOC/'PASS_METRICS_SYNTH.json')['splits']
    lines=['# PoseFix iterative replay diagnosis — all passes', '',
        'Historical Replay means synthetic training-data rehearsal, NOT iterative inference. Here: original synthetic-only PRIOR1/2/3, 0 training updates.', '',
        'Values are arithmetic means of three seed-specific summaries, not medians after pooling seeds. RAW and CAPPED are independent feedback paths. Cap is applied relative to each pass input. No pass, cap or checkpoint is selected on these results.', '',
        'Median/P90: observed matched corner8. PCK: full fixed GT denominator including missing/mismatch penalties. Pose reference on real/square is geometry reconstructed, not independently measured physical GT. All real sets are reused development diagnostics.', '']
    for split,panel in {**real,**synth}.items():
        lines += [f'## {split}', '']
        for mode in R.CHAINS:
            data=panel['mean_seed'][mode]
            lines += [f'### {mode} feedback','',
                '| pass | median px | P90 | PCK10 % | rot deg | trans cm | IoU3D | cap-hit % | reversal % |',
                '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
            for p in range(4):
                m=data['passes'][str(p)]; q=data['pose'][str(p)]
                cap=None if p==0 else 100*data['movement'][str(p)]['cap_hit_corner_fraction']
                reversal=None if p<2 else data['oscillation']['pairs']['d1_d2' if p==2 else 'd2_d3']['reversal']
                lines.append(f'| PASS{p} | {fmt(m["matched_pooled_corner8_median_px"])} | {fmt(m["matched_pooled_corner8_P90_px"])} | {fmt(100*m["PCK"]["10"])} | {fmt(q["rotation_deg"]["median"])} | {fmt(q["translation_cm"]["median"])} | {fmt(q["IoU3D"]["median"])} | {fmt(cap)} | {fmt(None if reversal is None else 100*reversal)} |')
            lines += ['', 'Cap-hit in RAW rows is the hypothetical same-input cap; RAW feedback itself remains uncapped.', '',
                '| R0 error | n corners / seed | PASS1 Δ px | PASS2 Δ px | PASS1 cap-hit % | PASS2 regression % |',
                '|---|---:|---:|---:|---:|---:|']
            for label in BINS:
                b=data['error_strata'][label]; a,c=b['passes']['1'],b['passes']['2']
                lines.append(f'| {label} | {b["n_corners"]:.0f} | {fmt(a["delta_from_R0"]["mean"])} | {fmt(c["delta_from_R0"]["mean"])} | {fmt(None if a["cap_hit_fraction"] is None else 100*a["cap_hit_fraction"])} | {fmt(None if c["regression_fraction"] is None else 100*c["regression_fraction"])} |')
            lines += ['', 'Δ = output error minus initial error, negative is improvement. These strata use matched/input-valid native corners against the locked PASS0 whole-object branch, not pointwise GT reassignment.', '']
    R.write(R.DOC/'REPLAY_SUMMARY.md','\n'.join(lines).rstrip()+'\n')
    rows=[r for r in R.read(R.RAW/'INPUTS.json') if r['split']=='REAL_DEV']
    z=np.load(R.RAW/'predictions/seed1_REAL_DEV.npz'); a=np.load(R.RAW/'scores/seed1_REAL_DEV.npz')
    prefix='CAPPED_'; mask=a[prefix+'GT_support']; targets=a[prefix+'locked_gt']
    choices=[]
    for label,key in [('Monotonic improvement','corner_monotonic_improve'),('Reversal / oscillation','corner_oscillation'),('Repeated cap','corner_repeated_cap')]:
        eligible=np.argwhere(a[prefix+key]&mask)
        choices.append((label,None if not len(eligible) else tuple(map(int,eligible[0]))))
    fig,axes=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
    chosen=[]
    for ax,(label,index) in zip(axes,choices):
        if index is None: ax.text(.5,.5,label+'\nNo eligible example',ha='center');ax.axis('off');continue
        i,k=index;r=rows[i];im=cv2.imread(str(R.ROOT/r['image']))[:,:,::-1]
        if r['pad']: im=im[r['pad']:-r['pad'],r['pad']:-r['pad']]
        ax.imshow(im); traj=z['points'][i,1,:,k]; raw=z['points'][i,0,:,k];gt=targets[i,k]
        ax.plot(raw[:,0],raw[:,1],'o--',color='#ff5ccf',label='RAW feedback',lw=1.4)
        ax.plot(traj[:,0],traj[:,1],'o-',color='#ffd23f',label='CAPPED feedback',lw=2)
        ax.scatter([gt[0]],[gt[1]],marker='x',s=130,c='#46ff61',label='GT, locked R0 branch')
        offsets=[(-24,14),(15,25),(22,-16),(-24,-23)]
        for p,xy in enumerate(traj):
            ax.annotate(str(p),xy,xytext=offsets[p],textcoords='offset points',color='#fff',
                bbox=dict(facecolor='black',alpha=.55,edgecolor='none',pad=1.5),
                arrowprops=dict(arrowstyle='-',color='white',lw=.7))
        both=np.vstack([traj,raw,gt]); lo=both.min(0)-35; hi=both.max(0)+35
        ax.set_xlim(max(0,lo[0]),min(im.shape[1],hi[0]));ax.set_ylim(min(im.shape[0],hi[1]),max(0,lo[1]))
        errors=a[prefix+'errors_native_PASS0_branch'][i,:,k]
        ax.set_title(f'{label}\n{r["id"]}\nnative corner{k}; error '+ ' / '.join(f'{v:.2f}' for v in errors)+' px',fontsize=8);ax.legend(fontsize=7,loc='lower left')
        chosen.append(dict(category=label,id=r['id'],corner=k,rule='first lexicographic eligible frame then smallest native index'))
    fig.savefig(R.DOC/'FIG_REPLAY_TRAJECTORIES.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,mode in zip(axes,R.CHAINS):
        data=real['REAL_DEV']['mean_seed'][mode]
        for label in BINS:
            b=data['error_strata'][label]['passes']; vals=[0]+[b[str(p)]['delta_from_R0']['mean'] for p in (1,2,3)]
            ax.plot(range(4),vals,'o-',label=label+' px')
        ax.axhline(0,color='k',lw=.6);ax.set_xticks(range(4));ax.set_xlabel('Pass');ax.set_ylabel('Mean error change from R0 (px)')
        ax.set_title('DEV319 / '+mode+' feedback');ax.legend(fontsize=8)
    fig.savefig(R.DOC/'FIG_ERROR_BY_INITIAL_BIN.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    x=a['CAPPED_raw_displacement'][:,0][mask]; y=a['CAPPED_capped_displacement'][:,0][mask]
    e=a['CAPPED_errors_native_PASS0_branch'][:,0][mask];colors=np.digitize(e,[5,10,20],right=True)
    for i,label in enumerate(BINS):
        ix=colors==i;axes[0].scatter(x[ix],y[ix],s=10,alpha=.5,label=label)
    axes[0].plot([0,max(x)],[0,max(x)],'k--',lw=.7);axes[0].set_xlabel('RAW movement (px)');axes[0].set_ylabel('Same-input capped movement (px)')
    axes[0].set_title('DEV319 seed1 PASS1');axes[0].legend(fontsize=8)
    data=real['REAL_DEV']['mean_seed']['CAPPED']['error_strata']
    vals=[100*data[k]['passes']['1']['cap_hit_fraction'] for k in BINS]
    axes[1].bar(BINS,vals);axes[1].set_ylabel('Cap-hit corners (%)');axes[1].set_xlabel('Initial R0 error (px)');axes[1].set_title('DEV319 PASS1 / mean of seeds')
    fig.savefig(R.DOC/'FIG_RAW_VS_CAPPED.png',dpi=180);plt.close(fig)
    R.write(R.DOC/'FIGURE_MANIFEST.json',dict(training_updates=0,seed=1,split='REAL_DEV',chosen=chosen,
        selection='Preregistered first eligible identity, not largest gain. Categories are descriptive, overlapping, not representative prevalence estimates.',
        error_plot='Mean of3seed matched-support errors against locked R0 whole-object branch',files=[R.bound(R.DOC/n) for n in ['FIG_REPLAY_TRAJECTORIES.png','FIG_ERROR_BY_INITIAL_BIN.png','FIG_RAW_VS_CAPPED.png']]))

if __name__=='__main__': build()
