"""Saved scientific plots; --charts-only needs no private RGB or model files.

Full mode draws derived annotated crops of six eligible cases frozen before
corrected scores. No inference, fits, rays, new RGB or annotation are run.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

NAME='pallet_kp_corrected_supervision_20261010_v1'
OLD='pallet_observation_refiner_20261009_v1'
HEADS=('GEOMETRY_ONLY','IMAGE_NO_ROLE','IMAGE_ROLE')
EDGES=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))

def read(p):return json.loads(Path(p).read_text())
def rows(p):
    with gzip.open(p,'rt') as f:return [json.loads(l) for l in f]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p,root):return dict(path=str(p.relative_to(root)),sha256=sha(p),bytes=p.stat().st_size)
def semantic(row):return hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def label(a):
    if a=='BASE':return 'Frozen Base'
    if a=='N3_SUBPIX':return 'Frozen N3 + SubPix'
    heads={'GEOMETRY_ONLY':'geometry only','IMAGE_NO_ROLE':'image, no role','IMAGE_ROLE':'image + role',
           'IMAGE_ROLE_NO_MASK_ROBUST':'image + role / no mask','IMAGE_ROLE_STANDARD':'image + role / ordinary',
           'IMAGE_ROLE_POINT_LINE':'image + role / point + line'}
    for prefix,name in [('CORRECTED_','Repaired: '),('OLD_','Old source: ')]:
        if a.startswith(prefix):return name+heads[a[len(prefix):]]
    arm='N3 + SubPix' if a.startswith('N3_SUBPIX_') else 'Base'
    suffix=a[len('N3_SUBPIX_'):] if a.startswith('N3_SUBPIX_') else a[len('BASE_'):]
    tails={'NO_MASK_STANDARD':'no mask / ordinary','NO_MASK_ROBUST':'no mask / robust',
        'GEOM_NOSELF_STANDARD':'predicted self-mask / ordinary','GEOM_NOSELF_ROBUST':'predicted self-mask / robust',
        'ORACLE_NOSELF_ROBUST':'human self-mask / robust [oracle]','ORACLE_VISIBLE_ROBUST':'human DIRECT only / robust [oracle]'}
    return arm+': '+tails.get(suffix,suffix)

def color(a):
    if a in ('BASE','N3_SUBPIX'):return '#6c7783'
    if a.endswith('POINT_LINE'):return '#a85c50'
    return '#895c9c' if a.startswith('CORRECTED_') else '#4383ad'
def save(fig,d,name):
    fig.savefig(d/name,dpi=120,bbox_inches='tight',facecolor='white');plt.close(fig)

def pose(book,data,d):
    arms=['BASE','N3_SUBPIX']+[a+'_'+s for a in ('BASE','N3_SUBPIX') for s in
        ('NO_MASK_STANDARD','NO_MASK_ROBUST','GEOM_NOSELF_STANDARD','GEOM_NOSELF_ROBUST','ORACLE_NOSELF_ROBUST','ORACLE_VISIBLE_ROBUST')]
    arms+=['CORRECTED_'+a for a in HEADS+('IMAGE_ROLE_NO_MASK_ROBUST','IMAGE_ROLE_STANDARD','IMAGE_ROLE_POINT_LINE')]
    fig,axes=plt.subplots(1,3,figsize=(19,12.5),gridspec_kw={'width_ratios':[1.55,1.35,1.05]})
    y=np.arange(len(arms));N=book['population']['frames']
    for ax,key,unit in zip(axes[:2],('translation_cm','rotation_deg'),('Position (cm)','Rotation (degree)')):
        for i,a in enumerate(arms):
            vals=np.array([r['pose'][key] for r in data[a] if r['pose']['available']])
            if not len(vals):continue
            lo,med,hi=np.quantile(vals,[.1,.5,.9]);mean=vals.mean()
            ax.scatter(vals,np.full(len(vals),i),s=3,c=color(a),alpha=.13,rasterized=True)
            ax.plot([lo,hi],[i,i],c=color(a),lw=3);ax.scatter(med,i,s=28,c=color(a),zorder=3)
            ax.scatter(mean,i,s=34,marker='D',facecolor='white',edgecolor='black',zorder=4)
            ax.text(.99,i+.24,f'{mean:.2f} / {med:.2f} / {hi:.2f}',transform=ax.get_yaxis_transform(),ha='right',fontsize=8)
        ax.set_yticks(y,[label(a) for a in arms] if ax is axes[0] else []);ax.invert_yaxis()
        ax.set_xscale('symlog',linthresh=1);ax.set_xlim(left=0);ax.grid(axis='x',alpha=.2);ax.set_xlabel(unit+'; every available output retained')
        ax.set_title('Mean / median / P90; large errors remain in the dot cloud')
    for i,a in enumerate(arms):
        s=book['methods'][a];n=s['new_pose_estimated'];b=s['fallback_used'];f=s['fixed_control_outputs'];bad=s['no_pose']
        axes[2].barh(i,n,color='#4383ad');axes[2].barh(i,b,left=n,color='#dba065')
        axes[2].barh(i,f,left=n+b,color='#959ea6');axes[2].barh(i,bad,left=n+b+f,color='#ba5656')
        axes[2].text(N/2,i,f'{n} new / {b} fallback / {bad} fail' if not f else f'{f} fixed outputs',ha='center',va='center',fontsize=8)
    axes[2].set_xlim(0,N);axes[2].set_yticks(y,[]);axes[2].invert_yaxis();axes[2].set_xlabel(f'Frames / eligible {N}');axes[2].set_title('Output status is part of the result')
    fig.suptitle('01 | Existing easy153 + medium92: saved pose errors and coverage',y=1.01)
    fig.text(.5,-.02,'Circles: median; hollow diamonds: mean; segments: P10-P90; symlog axes. Status colors: new blue, fallback orange, fixed gray, failure red.\nOracle arms are diagnostic. References are geometric proxies. Severe74 excluded by frozen human labels independently of pose errors.',ha='center',fontsize=9)
    fig.tight_layout();save(fig,d,'01_pose_overview.png')

def paired(book,d):
    cs=[('CORRECTED_IMAGE_ROLE_minus_N3_SUBPIX','Repaired image + role minus N3 + SubPix [primary]'),
        ('CORRECTED_GEOMETRY_ONLY_minus_N3_SUBPIX','Repaired geometry only minus N3 + SubPix'),
        ('CORRECTED_IMAGE_NO_ROLE_minus_N3_SUBPIX','Repaired image, no role minus N3 + SubPix'),
        ('CORRECTED_IMAGE_ROLE_POINT_LINE_minus_N3_SUBPIX','Repaired point + line minus N3 + SubPix'),
        ('CORRECTED_IMAGE_ROLE_NO_MASK_ROBUST_minus_CORRECTED_IMAGE_ROLE','Same image + role: no mask minus self-mask'),
        ('CORRECTED_IMAGE_ROLE_STANDARD_minus_CORRECTED_IMAGE_ROLE','Same image + role: ordinary minus robust'),
        ('CORRECTED_IMAGE_ROLE_minus_OLD_IMAGE_ROLE','Repaired targets minus old targets [same head architecture]'),
        ('N3_SUBPIX_GEOM_NOSELF_ROBUST_minus_N3_SUBPIX_NO_MASK_ROBUST','N3 + SubPix: self-mask minus no mask [same robust solver]'),
        ('BASE_GEOM_NOSELF_ROBUST_minus_BASE_NO_MASK_ROBUST','Base: self-mask minus no mask [same robust solver]')]
    fig,axes=plt.subplots(1,2,figsize=(16,8.5))
    for ax,key,unit in zip(axes,('translation_cm','rotation_deg'),('cm','degree')):
        for i,(c,_) in enumerate(cs):
            for scope,dy,col,m in [('combined',-.17,'#233f5d','o'),('easy',0,'#438e68','s'),('medium',.17,'#b47d37','^')]:
                v=book['strata'][scope]['contrasts'][c]['common_operational']['metrics'][key]
                if v['CI95'] is None:continue
                lo,hi=v['CI95'];ax.plot([lo,hi],[i+dy,i+dy],c=col,lw=1.7);ax.scatter(v['mean_delta'],i+dy,c=col,s=27,marker=m,zorder=3)
        ax.axvline(0,c='#666',ls='--');ax.set_yticks(np.arange(len(cs)),[r[1]for r in cs]if ax is axes[0]else [])
        ax.invert_yaxis();ax.set_xlabel(f'Paired mean error difference ({unit}); negative favors candidate');ax.grid(axis='x',alpha=.2)
        ax.set_title('Position'if key=='translation_cm'else 'Rotation')
    axes[1].legend(handles=[Line2D([],[],color=c,marker=m,label=l)for c,m,l in [('#233f5d','o','Combined245'),('#438e68','s','Easy153'),('#b47d37','^','Medium92')]],fontsize=9)
    fig.suptitle('02 | Paired operational deltas: recorded 10,000 session draws',y=1.02)
    fig.text(.5,-.02,'95% CI of paired mean, not error SD; fallback included. Common available IDs are bound in METRICS.json.\nThe exact13-session multiplicity matrix is reused with eligible-frame weights; zero new random draws.',ha='center',fontsize=9)
    fig.tight_layout();save(fig,d,'02_paired_deltas.png')

def correspondence(book,post,d):
    arms=['CORRECTED_'+a for a in HEADS+('IMAGE_ROLE_POINT_LINE',)];names=['Geometry only','Image\nno role','Image + role','Image + role\npoint + line']
    fig,axes=plt.subplots(2,2,figsize=(15,10));N=book['population']['frames']
    for i,a in enumerate(arms):
        rr=[r for r in post if r['method']==a];p=np.array([r['correct_pool_count']for r in rr]);direct=np.array([len(r['human_direct_correct_pool_ids'])for r in rr])
        for dx,v,c in [(-.18,int((p>=4).sum()),'#4383ad'),(.18,int((direct>=4).sum()),'#438e68')]:
            axes[0,0].bar(i+dx,v,.36,color=c);axes[0,0].text(i+dx,v+2,str(v),ha='center',fontsize=9)
        if all(r['point_PnP_inliers_applicable']for r in rr):
            v=sum(r['correct_final_inlier_count']>=4 for r in rr);axes[0,1].bar(i,v,.6,color='#895c9c');axes[0,1].text(i,v+2,str(v),ha='center')
        else:axes[0,1].text(i,N*.15,'No thresholded\npoint consensus',ha='center',fontsize=10)
    for ax in axes[0]:ax.set_xticks(range(4),names);ax.set_ylim(0,N*1.09);ax.set_ylabel(f'Frames / eligible {N}');ax.grid(axis='y',alpha=.2)
    axes[0,0].set_title('Retained proxy-accurate points (<=8px) before fit')
    axes[0,0].legend(handles=[Line2D([],[],color='#4383ad',lw=8,label='All valid reference points >=4'),Line2D([],[],color='#438e68',lw=8,label='Human DIRECT accurate points >=4')],fontsize=9)
    axes[0,1].set_title('Final proxy-accurate point-consensus inliers >=4')
    role=[r for r in post if r['method']=='CORRECTED_IMAGE_ROLE'];ks=['both_improved','both_worsened','mixed_or_equal','fallback','no_pose'];cols=['#438e68','#be5a56','#8b739b','#dba065','#333']
    for i,wrong in enumerate([False,True]):
        rr=[r for r in role if r['mask_wrong_on_known']==wrong];counts=Counter(r['paired_pose_outcome_vs_same_coordinate_initial']for r in rr);left=0
        for k,c in zip(ks,cols):
            axes[1,0].barh(i,counts[k],left=left,color=c)
            if counts[k]:axes[1,0].text(left+counts[k]/2,i,str(counts[k]),ha='center',va='center')
            left+=counts[k]
        axes[1,0].text(left+2,i,f'n={len(rr)}',va='center')
    axes[1,0].set_yticks([0,1],['Matching known mask','Wrong known mask']);axes[1,0].set_xlim(0,N*1.1)
    axes[1,0].set_xlabel('Frames; T/R both compared with frozen Base initial pose');axes[1,0].set_title('Mask disagreement is separate from pose quality')
    axes[1,0].legend(handles=[Line2D([],[],color=c,lw=8,label=k.replace('_',' '))for k,c in zip(ks,cols)],fontsize=8,ncol=2)
    for wrong,c in [(False,'#438e68'),(True,'#be5a56')]:
        rr=[r for r in role if r['mask_wrong_on_known']==wrong and r['new_pose_estimated']and r['translation_cm']is not None]
        axes[1,1].scatter([r['correct_pool_count']for r in rr],[r['translation_cm']for r in rr],s=25,alpha=.5,c=c,label='Wrong mask'if wrong else 'Matching mask')
    axes[1,1].set_yscale('symlog',linthresh=1);axes[1,1].set_xlabel('Retained proxy-accurate point count');axes[1,1].set_ylabel('Saved position error (cm; new poses)')
    axes[1,1].set_title('Point count does not certify geometry or pose');axes[1,1].legend(fontsize=9);axes[1,1].grid(alpha=.2)
    fig.suptitle('03 | Observation sufficiency, inlier accuracy and mask/pose distinction',y=1.02)
    fig.text(.5,-.02,'Accuracy is a frozen geometric proxy, not an independent physical observation certificate.\nObject/image rank, layouts, DIRECT IDs and actual inlier IDs remain recorded per frame.',ha='center',fontsize=9)
    fig.tight_layout();save(fig,d,'03_correspondence_mask.png')

def learning(curves,book,data,d):
    fig,axes=plt.subplots(2,2,figsize=(14.5,9));cols=['#4383ad','#a87743','#895c9c']
    for a,c in zip(HEADS,cols):
        curve=sorted([r for r in curves['source_curves']if r['arm']==a],key=lambda r:r['step'])
        steps=[0]+[r['step']for r in curve];test=[curves['initial_probes'][a]]+[r['source_test']for r in curve]
        axes[0,0].plot(steps,[v['loss_image_average']for v in test],'o-',c=c,label=label('CORRECTED_'+a).replace('Repaired: ',''))
        axes[0,1].plot(steps,[100*v['positive_adoption_rate']for v in test],'o-',c=c)
        axes[1,0].plot(steps[1:],[v['positive_candidate_error_px_mean_conditional_accepted']for v in test[1:]],'o-',c=c)
        axes[1,0].text(3000,test[-1]['positive_candidate_error_px_mean_conditional_accepted'],f'  NONE false accept {100*test[-1]["no_match_false_acceptance"]:.1f}%',color=c,fontsize=9)
    axes[0,0].set_ylabel('Heldout-source image-average CE');axes[0,0].set_title('Same source-test128;3000 updates, seed1');axes[0,0].legend(fontsize=9)
    axes[0,1].set_ylabel('Positive-target adoption (%)');axes[0,1].set_ylim(0,105);axes[0,1].set_title('Position accuracy is conditional on adoption')
    axes[1,0].set_ylabel('Accepted positive-candidate error (px)');axes[1,0].set_title('Final false NONE acceptances annotated separately')
    for ax in list(axes.flat)[:3]:ax.set_xlabel('Formal update');ax.grid(alpha=.2)
    N=book['population']['frames']
    for i,a in enumerate(HEADS):
        alias='CORRECTED_'+a;rr=data[alias];corner=sum(len(r.get('selected_corner_ids',[]))>=4 for r in rr);new=book['methods'][alias]['new_pose_estimated']
        for dx,v,c in [(-.18,corner,cols[i]),(.18,new,'#438e68')]:
            axes[1,1].bar(i+dx,v,.36,color=c);axes[1,1].text(i+dx,v+2,str(v),ha='center')
    axes[1,1].set_xticks(range(3),['Geometry only','Image\nno role','Image + role']);axes[1,1].set_ylim(0,N*1.1);axes[1,1].set_ylabel(f'Real easy/medium frames / {N}')
    axes[1,1].set_title('Actual assembled closure versus new-pose output');axes[1,1].grid(axis='y',alpha=.2)
    axes[1,1].legend(handles=[Line2D([],[],color='#895c9c',lw=8,label='>=4 assembled corners before eligibility/mask'),Line2D([],[],color='#438e68',lw=8,label='New robust point-PnP pose')],fontsize=8,loc='lower left')
    fig.suptitle('04 | Source learning and real pose transfer are separate outcomes',y=1.02)
    fig.text(.5,-.02,'Fixed original66-way MAP; no source-test model selection or match-mass decoder. Geometry-only retains geometric role channels.\nCorrected-source loss denominators differ from old mistargeted labels; lower loss alone is not proof of improved correction.',ha='center',fontsize=9)
    fig.tight_layout();save(fig,d,'04_learning_transfer.png')

def damage(book,post,d):
    arms=['CORRECTED_'+a for a in HEADS+('IMAGE_ROLE_POINT_LINE',)];fig,axes=plt.subplots(1,3,figsize=(16,5.2))
    for ax,category,title in zip(axes[:2],('DIRECT_VISIBLE','SELF_OCCLUDED'),('DIRECT: protect observed corners','SELF: hidden reprojection is separate')):
        for i,a in enumerate(arms):
            rr=[r for r in post if r['method']==a];delta=[]
            for r in rr:
                delta.extend(r['reference_error_output_native_px'][k]-r['reference_error_initial_native_px'][k]
                    for k in r['paired_corner_valid_ids']if r['human_states_native'][k]==category)
            if not delta:continue
            q=np.quantile(delta,[.1,.5,.9]);mean=np.mean(delta);ax.scatter(delta,np.full(len(delta),i),s=3,alpha=.1,c=color(a))
            ax.plot(q[[0,2]],[i,i],c=color(a),lw=3);ax.scatter(q[1],i,c=color(a),s=28);ax.scatter(mean,i,marker='D',facecolor='white',edgecolor='black')
            ax.text(.99,i+.2,f'n={len(delta)}; mean {mean:+.2f}; P90 {q[2]:+.2f}',transform=ax.get_yaxis_transform(),ha='right',fontsize=8)
        ax.set_yticks(range(4),[label(a).replace('Repaired: ','')for a in arms]if ax is axes[0]else [])
        ax.set_ylim(3.5,-.5);ax.axvline(0,c='#666',ls='--');ax.set_xscale('symlog',linthresh=1);ax.grid(axis='x',alpha=.2)
        ax.set_xlabel('Output minus frozen Base error (px)');ax.set_title(title)
    for i,a in enumerate(arms):
        v=book['visibility_damage'][a]['operational'].get('DIRECT_VISIBLE',{}).get('good5_to_bad10',0)
        axes[2].bar(i,v,color=color(a));axes[2].text(i,v+.2,str(v),ha='center')
    axes[2].set_xticks(range(4),['Geometry','Image\nno role','Image +\nrole','Image + role\npoint + line']);axes[2].set_ylabel('DIRECT count: input<5px to output>10px');axes[2].set_title('Visible damage explicitly reported');axes[2].grid(axis='y',alpha=.2)
    fig.suptitle('05 | Visible damage and hidden reprojection do not substitute for pose evaluation',y=1.03)
    fig.text(.5,-.035,'Paired operational errors including fallback; frozen Base permutation and valid reference IDs.\nSELF references can be reconstructed. Symbols: circle median, diamond mean, segments P10-P90.',ha='center',fontsize=9)
    fig.tight_layout();save(fig,d,'05_visible_hidden_damage.png')

def stress(doc,d):
    book=read(doc/'REAL_MASK_STRESS_SUMMARY.json');fig,axes=plt.subplots(1,3,figsize=(15,5.2))
    aliases=['N3_SUBPIX_DROP_ONE_VISIBLE','N3_SUBPIX_KEEP_ONE_HIDDEN']
    for i,(scope,c)in enumerate([('combined','#233f5d'),('easy','#438e68'),('medium','#b47d37')]):
        for j,a in enumerate(aliases):
            s=book['strata'][scope][a]['summary'];x=j+(i-1)*.2;axes[0].bar(x,s['new_pose_estimated']/s['denominator']*100,.19,color=c)
            for ax,k in zip(axes[1:],('translation_cm','rotation_deg')):
                v=s['metrics']['operational'][k];ax.plot(x,v['mean'],'o',c=c);ax.plot(x,v['P90'],'x',c=c)
    for ax in axes:ax.set_xticks([0,1],['Drop1 human DIRECT','Retain1 human SELF']);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('New pose rate (%)');axes[0].set_ylim(0,105)
    axes[1].set_ylabel('Position error (cm)');axes[2].set_ylabel('Rotation error (degree)')
    axes[0].legend(handles=[Line2D([],[],color=c,lw=8,label=l)for l,c in [('Combined245','#233f5d'),('Easy153','#438e68'),('Medium92','#b47d37')]],fontsize=9)
    for ax in axes[1:]:ax.set_yscale('symlog',linthresh=1);ax.set_title('Circle mean; cross P90')
    fig.suptitle('07 | Historical real mask stress, eligible IDs only',y=1.02)
    fig.text(.5,-.03,'Zero new fits in this historical figure. Hidden retained coordinates may still be accurate.\nThe prior real study lacks2/both variants;128-scene analytical stress is separate, not245 real coverage.',ha='center',fontsize=9)
    fig.tight_layout();save(fig,d,'07_historical_mask_stress.png')

def runtime(doc,d):
    p=doc/'RUNTIME.json'
    if not p.exists()or not read(p).get('complete'):return False
    book=read(p);measured=[r for r in rows(doc/'RUNTIME_ROWS.jsonl.gz')if r['phase']=='measured']
    fig,ax=plt.subplots(figsize=(11,5.5));maximum_latency=0
    for i,a in enumerate(book['arms']):
        rr=[r for r in measured if r['arm']==a]
        vals=np.array([r['full_ms']for r in rr],float)
        assert np.isfinite(vals).all()
        maximum_latency=max(maximum_latency,float(vals.max()))
        mean=vals.mean();med=np.median(vals);p90=np.quantile(vals,.9)
        ax.scatter(np.full(len(vals),i),vals,c=color('CORRECTED_'+a),s=12,alpha=.25);ax.plot([i-.2,i+.2],[mean,mean],c='black',lw=2)
        ax.scatter(i,med,c='#4383ad',s=55);ax.scatter(i,p90,c='#be5a56',marker='x',s=55)
        ax.text(i,vals.max()+1,f'n={len(vals)}\nmean {mean:.2f}\nmedian {med:.2f}\nP90 {p90:.2f}',ha='center',fontsize=9)
    ax.set_xticks(range(len(book['arms'])),[label('CORRECTED_'+a)if a=='IMAGE_ROLE'else label(a)for a in book['arms']])
    ax.set_ylim(0,maximum_latency*1.22+3)
    ax.set_ylabel('Actual synchronized whole-path wall time (ms)');ax.grid(axis='y',alpha=.2)
    fig.suptitle('08 | Fresh eligible whole paths include detector, initial pose and final fit',y=1.04)
    fig.text(.5,-.025,'Actual RAM-RGB calls on frozen26 eligible frames: warmup20 + measured130 per arm. Black line: mean; blue circle: median; red cross: P90.\nNo cache replay, sum of old stage timings or omitted solver cost. Every measured latency, including outliers, is shown.',ha='center',fontsize=9)
    fig.tight_layout();save(fig,d,'08_runtime.png');return True

def cube(dims):
    a,b,c=np.asarray(dims,float)/2
    return np.array([[-a,-b,-c],[a,-b,-c],[a,b,-c],[-a,b,-c],[-a,-b,c],[a,-b,c],[a,b,c],[-a,b,c]])
def projection(pose,K):
    if not pose or not pose.get('available'):return np.full((8,2),np.nan)
    camera=cube(pose['cf_extents'])@np.asarray(pose['R_cf']).T+np.asarray(pose['centroid']);uv=camera@np.asarray(K).T
    q=np.full((8,2),np.nan);valid=camera[:,2]>1e-9;q[valid]=uv[valid,:2]/uv[valid,2:];return q
def crop(frame):
    h,w=frame['raw_hw'];l,t,r,b=frame['candidate_metadata']['box_xyxy'];pad=max(r-l,b-t)*.07
    return [max(0,int(l-pad)),max(0,int(t-pad)),min(w,int(np.ceil(r+pad))),min(h,int(np.ceil(b+pad)))]
def network(ax,q,c,lw=1,ls='-',valid=None,alpha=1):
    q=np.asarray(q,float)
    for a,b in EDGES:
        if valid is not None and(a not in valid or b not in valid):continue
        if np.isfinite(q[[a,b]]).all():ax.plot(q[[a,b],0],q[[a,b],1],c=c,lw=lw,ls=ls,alpha=alpha)
def markers(ax,q,ids,roi,c,m,size,labels=False):
    l,t,r,b=roi;q=np.asarray(q,float);off=[]
    for k in ids:
        p=q[k]
        if not np.isfinite(p).all():off.append(k);continue
        inside=l<=p[0]<r and t<=p[1]<b;x=np.clip(p[0],l+3,r-3);y=np.clip(p[1],t+3,b-3)
        ax.scatter(x,y,c=c,marker=m if inside else '>',s=size,linewidths=.5,zorder=5)
        if not inside:off.append(k)
        if labels and inside:ax.text(x+3,y+3,str(k),color=c,fontsize=7,bbox=dict(facecolor='black',alpha=.65,pad=.3),zorder=6)
    return off

def cases(root,doc,source,data,post,d):
    import cv2
    protocol=read(doc/'VISUAL_CASE_PROTOCOL.json');allowed=set(read(doc/'COHORT.json')['ids'])
    frames={r['id']:r for r in read(root/'_docs/experiments'/OLD/'INPUTS.json')['frames']}
    obs_path=doc/'LEARNED_OBSERVATIONS.jsonl.gz';observations={r['id']:r for r in rows(obs_path)if r['method']=='IMAGE_ROLE'}
    ann={(r['method'],r['id']):r for r in post};lookup={a:{r['id']:r for r in rr}for a,rr in data.items()}
    historical_lines={(r['alias'],r['id']):(i,r['method'])for i,r in enumerate(rows(doc/'HISTORICAL_FILTERED_ROWS.jsonl.gz'),1)}
    corrected_lines={('CORRECTED_'+r['method'],r['id']):(i,r['method'])for i,r in enumerate(rows(doc/'LEARNED_PREDICTIONS.jsonl.gz'),1)}
    posthoc_lines={(r['method'],r['id']):i for i,r in enumerate(post,1)}
    fig,axes=plt.subplots(6,2,figsize=(15,28));manifest=[]
    for i,case in enumerate(protocol['cases']):
        fid=case['id'];assert fid in allowed;f=frames[fid];p=source/f['image'];assert sha(p)==f['image_sha256']
        image=cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB);roi=crop(f);l,t,r,b=roi
        for j,a in enumerate(protocol['panel_methods']):
            ax=axes[i,j];row=lookup[a][fid];ar=ann[(a,fid)]
            ax.imshow(image[t:b,l:r],extent=[l,r,b,t]);ax.set_xlim(l,r);ax.set_ylim(b,t);ax.set_aspect('equal');ax.axis('off')
            initial_arm='N3_SUBPIX'if a=='N3_SUBPIX'else 'BASE';initial=np.asarray(f['points'][initial_arm],float)[:8]
            q=np.asarray(row.get('input_points',initial),float)[:8];network(ax,initial,'#30d3de',alpha=.55);markers(ax,initial,range(8),roi,'#30d3de','.',20,True)
            pool=ar['pool_ids'];inliers=ar['final_inlier_ids'];hidden=ar['initial_hidden_ids']
            observed_off=markers(ax,q,pool,roi,'#ffdf45','o',28,True);inlier_off=markers(ax,q,inliers,roi,'#72ee85','o',65)
            hidden_off=markers(ax,initial,hidden,roi,'#fa5358','x',65)
            projected=projection(row.get('actual_pose'),f['K']);network(ax,projected,'#ff902b',lw=1.5)
            projection_off=markers(ax,projected,range(8),roi,'#ff902b','.',19)
            reference=np.asarray(ar['reference_native_points_px'],float);valid=ar['reference_valid_native_ids'];network(ax,reference,'#f294dc',ls='--',valid=valid,alpha=.8)
            reference_off=markers(ax,reference,valid,roi,'#f294dc','+',30)
            if ar['hidden_reprojected_ids']:markers(ax,np.asarray(row['native_points'])[:8],ar['hidden_reprojected_ids'],roi,'#ff902b','s',48)
            if a=='CORRECTED_IMAGE_ROLE':
                for line in observations[fid]['lines']:
                    support=np.asarray(line['support_points'],float)
                    if len(support):ax.scatter(support[:,0],support[:,1],s=5,c='#ffdf45',alpha=.6,zorder=3)
            pose=row['pose'];status=row['output_status'];error=f'T {pose["translation_cm"]:.2f}cm | R {pose["rotation_deg"]:.2f}deg'if pose['available']else 'No pose output'
            ax.set_title(f'{case["label"].upper()} | {fid}\n{label(a)}\n{error} | {status}',fontsize=9,loc='left')
            note=f'Observed {pool}; inliers {inliers}; hidden excluded {hidden}\nAccurate pool {ar["correct_pool_count"]}; accurate inliers {ar["correct_final_inlier_count"]}; off-crop pose IDs {projection_off}'
            if a=='N3_SUBPIX':note=f'Fixed control: no new fit pool or inlier selection\nOff-crop saved-pose projection IDs {projection_off}'
            if a=='CORRECTED_IMAGE_ROLE'and row.get('fallback_used'):note+='\nInsufficient/invalid observations: baseline fallback'
            ax.text(.01,.015,note,transform=ax.transAxes,fontsize=7.5,color='white',bbox=dict(facecolor='black',alpha=.7,pad=2),va='bottom')
            rawname='HISTORICAL_FILTERED_ROWS.jsonl.gz'if a=='N3_SUBPIX'else 'LEARNED_PREDICTIONS.jsonl.gz'
            line,rawmethod=(historical_lines if a=='N3_SUBPIX'else corrected_lines)[(a,fid)]
            manifest.append(dict(id=fid,method=a,initial_input_arm=initial_arm,label=case['label'],source_image=f['image'],
                source_image_sha256=f['image_sha256'],crop_xyxy=roi,selected_before_new_accuracy=True,raw_status=status,pose_metrics=pose,
                observed_ids=pool,final_inlier_ids=inliers,predicted_hidden_exclusion_ids=hidden,native_hidden_replaced_ids=ar['hidden_reprojected_ids'],
                full_projection_off_crop_ids=projection_off,observed_off_crop_ids=observed_off,inlier_off_crop_ids=inlier_off,
                hidden_input_off_crop_ids=hidden_off,reference_off_crop_ids=reference_off,row_semantic_sha256=semantic(row),
                posthoc_semantic_sha256=semantic(ar),orange_wireframe_is_full_pose_projection=True,orange_wireframe_is_not_full_native_output=True,
                raw_record=dict(path=str((doc/rawname).relative_to(root)),line_1_based=line,method=rawmethod,id=fid,
                    alias_field_removed_for_semantic_hash=a=='N3_SUBPIX'),
                posthoc_record=dict(path=str((doc/'POSTHOC_CORRESPONDENCE_ROWS.jsonl.gz').relative_to(root)),
                    line_1_based=posthoc_lines[(a,fid)],method=a,id=fid),
                reference_kind='GEOMETRIC_PROXY valid points only',independent_reference_truth_certified=False))
    fig.legend(handles=[
        Line2D([],[],c='#30d3de',marker='.',label='Frozen input (Base or N3+SubPix)'),
        Line2D([],[],c='#ffdf45',marker='o',ls='',label='Selected observations / edge support'),
        Line2D([],[],c='#72ee85',marker='o',ls='',label='Final point-consensus inliers'),
        Line2D([],[],c='#fa5358',marker='x',ls='',label='Predicted self-hidden exclusion'),
        Line2D([],[],c='#ff902b',label='Full saved-pose projection (not all native output)'),
        Line2D([],[],c='#ff902b',marker='s',ls='',label='Actual replaced hidden native output'),
        Line2D([],[],c='#f294dc',ls='--',label='Valid geometric-proxy reference'),
        Line2D([],[],c='white',marker='>',markeredgecolor='black',ls='',label='Off-crop point clipped to border')],
        loc='lower center',bbox_to_anchor=(.5,.003),ncol=2,fontsize=9)
    fig.suptitle('06 | Twelve actual eligible panels: six IDs fixed before corrected scores',y=.998,fontsize=17)
    fig.text(.5,.053,'Same original RGB and frozen detector ROI in each pair. Zero new annotation, image generation or model calls.\nLarge-error projections remain in statistics and are marked off crop. Fixed illustrations, not a performance-selected success set.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.04,right=.985,top=.962,bottom=.11,hspace=.5,wspace=.04)
    save(fig,d,'06_real_cases.png')
    path=doc/'VISUAL_REVIEW_CASES.json'
    value=dict(schema='corrected_eligible_real_visual_review_v1',complete=True,
        source_protocol=bind(doc/'VISUAL_CASE_PROTOCOL.json',root),cohort=bind(doc/'COHORT.json',root),
        source_bindings=[bind(p,root)for p in (doc/'LEARNED_PREDICTIONS.jsonl.gz',doc/'POSTHOC_CORRESPONDENCE_ROWS.jsonl.gz',obs_path,
                                             root/'_docs/experiments'/OLD/'INPUTS.json',doc/'HISTORICAL_FILTERED_ROWS.jsonl.gz')],
        cases=manifest,case_count=6,panel_count=12,derived_rgb_crops=True,raw_rgb_publication=False,
        new_models_rays_fits_RGB_annotations=0,figure=bind(d/'06_real_cases.png',root),
        provenance_extension='Added exact raw/posthoc line locations; all case coordinates, metrics, selection and figure unchanged.')
    if path.exists():
        previous=read(path);assert previous['cases']==value['cases'],'Numerical case records must remain unchanged'
        value['render_revision']=dict(reason='Increase row spacing and clarify fixed-control display; numerical cases unchanged',
                                      previous_figure=previous['figure'],new_scientific_calls=0)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3])
    p.add_argument('--source-root',type=Path);p.add_argument('--charts-only',action='store_true')
    p.add_argument('--real-cases-only',action='store_true');p.add_argument('--runtime-only',action='store_true');args=p.parse_args()
    assert sum((args.charts_only,args.real_cases_only,args.runtime_only))<=1
    root=args.root.resolve();doc=root/'_docs/experiments'/NAME;d=doc/'figures';d.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.pad_inches':.14})
    if args.runtime_only:
        assert runtime(doc,d),'Complete actual runtime receipt required'
        print('Runtime figure generated from recorded full_ms; new scientific calls0');return
    book=read(doc/'METRICS.json');allowed=set(read(doc/'COHORT.json')['ids']);assert len(allowed)==245
    data={a:[]for a in book['methods']}
    for row in rows(doc/'HISTORICAL_FILTERED_ROWS.jsonl.gz'):
        if row.get('historical_stress'):continue
        data[row['alias']].append({k:v for k,v in row.items()if k!='alias'})
    for row in rows(doc/'LEARNED_PREDICTIONS.jsonl.gz'):data['CORRECTED_'+row['method']].append(row)
    for a,rr in data.items():assert len(rr)==245 and{r['id']for r in rr}==allowed,a
    post=rows(doc/'POSTHOC_CORRESPONDENCE_ROWS.jsonl.gz');assert all(r['id']in allowed for r in post)
    if not args.real_cases_only:
        pose(book,data,d);paired(book,d);correspondence(book,post,d);learning(read(doc/'SOURCE_CURVES.json'),book,data,d);damage(book,post,d);stress(doc,d)
        hasruntime=runtime(doc,d)
    else:hasruntime=False
    if not args.charts_only:
        assert args.source_root,'Full review needs existing original RGB via --source-root'
        cases(root,doc,args.source_root.resolve(),data,post,d)
    print(json.dumps(dict(charts=True,real_panels=not args.charts_only,runtime=hasruntime,figure_bytes=sum(p.stat().st_size for p in d.glob('*.png')),new_scientific_calls=0)))

if __name__=='__main__':main()
