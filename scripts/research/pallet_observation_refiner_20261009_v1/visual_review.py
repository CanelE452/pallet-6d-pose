"""Review figures from sealed measurements; no inference, optimizer or pose fit.

--charts-only needs only the published numeric experiment files. Full mode also
reads original real/P0 images and the retained source target cache. Three actual
mesh silhouette ray passes are illustration-only and counted in the manifest.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from . import common as C

FIG = C.DOC / 'figures'
ARMS = ['BASE','SUBPIX','N3','N3_SUBPIX','BASE_NO_MASK_ROBUST',
        'BASE_GEOM_NOSELF_ROBUST','N3_SUBPIX_NO_MASK_ROBUST',
        'N3_SUBPIX_GEOM_NOSELF_ROBUST','GEOMETRY_ONLY','IMAGE_NO_ROLE',
        'IMAGE_ROLE','IMAGE_ROLE_POINT_LINE']
NAMES = {'BASE':'Base (old)','SUBPIX':'SubPix (old)','N3':'N3 (old)',
         'N3_SUBPIX':'N3 + SubPix (old)','BASE_NO_MASK_ROBUST':'Base / no mask / robust',
         'BASE_GEOM_NOSELF_ROBUST':'Base / geometry mask / robust',
         'N3_SUBPIX_NO_MASK_ROBUST':'N3+SubPix / no mask / robust',
         'N3_SUBPIX_GEOM_NOSELF_ROBUST':'N3+SubPix / geometry mask / robust',
         'GEOMETRY_ONLY':'Learned geometry only','IMAGE_NO_ROLE':'Learned image, no role',
         'IMAGE_ROLE':'Learned image + role','IMAGE_ROLE_POINT_LINE':'Image + role / point + line'}
COLORS = ['#576271']*4+['#2f78aa']*4+['#c7792d','#9b5aaa','#4d9565','#aa4d4d']
EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]


def rows(name):
    return list(C.iter_rows(C.DOC/name))


def indexed_rows(name):
    return {(r['id'],r['method']):(i,r) for i,r in enumerate(C.iter_rows(C.DOC/name),1)}


def finish(fig, name):
    fig.savefig(FIG/name,dpi=140,facecolor='white',bbox_inches='tight')
    plt.close(fig)


def style():
    plt.rcParams.update({'font.size':10,'axes.titlesize':12,'axes.labelsize':10,
                         'figure.titlesize':16,'axes.spines.top':False,
                         'axes.spines.right':False,'savefig.pad_inches':.13})


def chart_pose():
    raw=rows('FIXED_CONTROLS.jsonl.gz')+rows('POSE_DIAGNOSTICS.jsonl.gz')+rows('LEARNED_PREDICTIONS.jsonl.gz')
    data={m:[r for r in raw if r['method']==m] for m in ARMS}
    fig,axs=plt.subplots(1,3,figsize=(17.8,8.7),gridspec_kw={'width_ratios':[1.7,1.3,1.05]})
    y=np.arange(len(ARMS))
    for ax,key,unit in zip(axs[:2],['translation_cm','rotation_deg'],['Position error (cm)','Rotation error (deg)']):
        for i,m in enumerate(ARMS):
            a=np.array([r['pose'][key] for r in data[m] if r['pose']['available']],float)
            assert len(a)==319
            a.sort();q=np.quantile(a,[.1,.5,.9]);mean=a.mean()
            ax.scatter(a,np.full(len(a),i),s=4,alpha=.12,color=COLORS[i],rasterized=True)
            ax.plot(q[[0,2]],[i,i],color=COLORS[i],lw=3)
            ax.scatter(q[1],i,s=38,marker='o',c=COLORS[i],edgecolor='white',zorder=4)
            ax.scatter(mean,i,s=48,marker='D',facecolor='white',edgecolor='black',zorder=5)
            ax.text(.99,i+.24,f'mean {mean:.2f} | med {q[1]:.2f} | P90 {q[2]:.2f}',
                    transform=ax.get_yaxis_transform(),ha='right',fontsize=8,color='#333333')
        ax.set_xscale('symlog',linthresh=1);ax.grid(axis='x',alpha=.25);ax.invert_yaxis()
        ax.set_xlabel(unit+' (symlog; every available frame retained)')
        ax.set_yticks(y);ax.set_yticklabels([NAMES[m] for m in ARMS] if ax is axs[0] else [])
    axs[0].set_title('All 319 operational outputs, including large errors')
    axs[1].set_title('Circles: median; diamonds: mean; bar: P10–P90')
    new=np.array([sum(r['new_pose_estimated'] for r in data[m]) for m in ARMS])
    fallback=np.array([sum(r['fallback_used'] for r in data[m]) for m in ARMS])
    old=np.array([319 if m in C.CONTROLS else 0 for m in ARMS])
    axs[2].barh(y,new,color='#2f78aa',label='New pose')
    axs[2].barh(y,fallback,left=new,color='#e4ae6d',label='Base fallback')
    axs[2].barh(y,old,color='#9b9fa6',label='Historical fixed control')
    for i in y:axs[2].text(160,i,f'{new[i]} new / {fallback[i]} fallback' if old[i]==0 else '319 historical outputs',ha='center',va='center',fontsize=8)
    axs[2].set_xlim(0,319);axs[2].set_yticks(y,[]);axs[2].invert_yaxis();axs[2].set_xlabel('Frames (no complete operational failures)');axs[2].set_title('Production status is part of the result')
    axs[2].legend(loc='upper center',bbox_to_anchor=(.5,-.1),fontsize=9)
    fig.suptitle('01 | Actual pose error and output coverage',y=1.015)
    fig.text(.37,-.02,'Human reference poses are reconstructed evaluation references. Learned point + line is a separate local ablation.',ha='center',fontsize=9)
    fig.tight_layout();finish(fig,'01_pose_overview.png')


def chart_paired():
    book=C.read(C.DOC/'PAIRED_COMPARISONS.json')
    contrasts=[('N3_SUBPIX_NO_MASK_STANDARD_minus_N3_SUBPIX','No-mask standard − old N3+SubPix'),
        ('N3_SUBPIX_NO_MASK_ROBUST_minus_N3_SUBPIX_NO_MASK_STANDARD','No-mask robust − no-mask standard (same points)'),
        ('N3_SUBPIX_GEOM_NOSELF_ROBUST_minus_N3_SUBPIX','Geometry mask robust − old N3+SubPix'),
        ('N3_SUBPIX_GEOM_NOSELF_ROBUST_minus_N3_SUBPIX_NO_MASK_ROBUST','Geometry mask − no mask (same robust solver)'),
        ('BASE_GEOM_NOSELF_ROBUST_minus_BASE_NO_MASK_ROBUST','Geometry mask − no mask (Base; same solver)'),
        ('GEOMETRY_ONLY_minus_N3_SUBPIX','Learned geometry only − old N3+SubPix'),
        ('IMAGE_NO_ROLE_minus_N3_SUBPIX','Learned image, no role − old N3+SubPix'),
        ('IMAGE_ROLE_minus_N3_SUBPIX','Learned image + role − old N3+SubPix'),
        ('IMAGE_ROLE_NO_MASK_ROBUST_minus_IMAGE_ROLE','No-mask ablation − learned image + role'),
        ('IMAGE_ROLE_POINT_LINE_minus_IMAGE_ROLE','Point + line − same image + role observations')]
    fig,axs=plt.subplots(1,2,figsize=(15,6.8));y=np.arange(len(contrasts))
    for ax,key,unit in zip(axs,['translation_cm','rotation_deg'],['cm','deg']):
        for i,(name,label) in enumerate(contrasts):
            a=book['contrasts'][name]['operational']['metrics'][key];mean=a['mean'];lo,hi=a['CI95']
            ax.plot([lo,hi],[i,i],lw=2.5,color='#4b6983');ax.scatter(mean,i,c='#173e5c',s=38,zorder=3)
            ax.text(.99,i+.2,f'{mean:+.3f} [{lo:+.3f}, {hi:+.3f}]',transform=ax.get_yaxis_transform(),ha='right',fontsize=9)
        ax.axvline(0,color='#777777',ls='--');ax.grid(axis='x',alpha=.2);ax.invert_yaxis()
        ax.set_yticks(y,[r[1] for r in contrasts] if ax is axs[0] else [])
        ax.set_xlabel(f'Paired mean error difference ({unit}); negative favors left method')
        ax.set_title('Position' if key=='translation_cm' else 'Rotation')
    fig.suptitle('02 | Paired operational deltas with the recorded session bootstrap 95% CI',y=1.015)
    fig.text(.5,-.035,'All contrasts use the common 319-frame operational set. CI is from the sealed session-resampling draw; no new bootstrap was run.',ha='center',fontsize=9)
    fig.tight_layout();finish(fig,'02_paired_deltas.png')


def chart_stress():
    summary=C.read(C.DOC/'GEOMETRY_STRESS_SUMMARY.json');raw=rows('GEOMETRY_STRESS.jsonl.gz');conditions=summary['conditions']
    labs=['All valid','Drop 1 correct','Drop 2 correct','Keep 1 wrong','Keep 2 wrong','Drop/keep 1 each','Drop/keep 2 each','2 coherent wrong','4 valid','5 valid','Near collinear']
    x=np.arange(len(conditions));fig,axs=plt.subplots(2,2,figsize=(15.5,8.5),sharex=True)
    for solver,color,dx in [('STANDARD','#bb7941',-.12),('ROBUST','#397cac',.12)]:
        groups=[[r for r in raw if r['condition']==cond and r['solver']==solver] for cond in conditions]
        for ax,key in zip(axs[0],['translation_cm','rotation_deg']):
            med=[summary['summary'][c][solver][key]['median'] for c in conditions]
            p90=[summary['summary'][c][solver][key]['P90'] for c in conditions]
            ax.plot(x+dx,med,'o-',c=color,label=solver.title()+' median')
            ax.plot(x+dx,p90,'x--',c=color,alpha=.65,label=solver.title()+' P90')
            ax.set_yscale('symlog',linthresh=1);ax.grid(axis='y',alpha=.2)
        counts=[sum(r['pose_available'] for r in g) for g in groups]
        axs[1,0].bar(x+dx,counts,width=.24,color=color,label=solver.title())
        wrong=[np.mean([r['inlier_wrong_count'] for r in g]) for g in groups]
        right=[np.mean([r['inlier_correct_count'] for r in g]) for g in groups]
        axs[1,1].plot(x+dx,right,'o-',c=color,label=solver.title()+' correct inliers')
        axs[1,1].plot(x+dx,wrong,'x--',c=color,label=solver.title()+' wrong inliers')
    correct=[np.mean([r['correct_remaining_count'] for r in raw if r['condition']==c and r['solver']=='ROBUST']) for c in conditions]
    for i,c in enumerate(conditions):
        standard=summary['summary'][c]['STANDARD']['available'];robust=summary['summary'][c]['ROBUST']['available']
        axs[1,0].text(i,max(standard,robust)+2,f'{standard} / {robust}',ha='center',fontsize=8)
    axs[1,0].set_title('Output counts above bars: standard / robust',fontsize=10)
    axs[1,1].plot(x,correct,'s:',color='#222222',label='Correct points left before fit')
    axs[0,0].set_ylabel('Position error (cm; symlog)');axs[0,1].set_ylabel('Rotation error (deg; symlog)')
    axs[1,0].set_ylabel('New pose outputs / 128 scenes');axs[1,0].set_ylim(0,143);axs[1,1].set_ylabel('Mean correspondence count');axs[1,1].set_ylim(-.1,8.5)
    for ax in axs.flat:ax.set_xticks(x,labs,rotation=35,ha='right');ax.grid(axis='y',alpha=.18)
    axs[0,0].legend(fontsize=8,ncol=2);axs[1,0].legend(fontsize=8);axs[1,1].legend(fontsize=8)
    fig.suptitle('03 | Wrong exclusion / wrong retention: 11 fixed analytical conditions',y=1.01)
    fig.text(.5,-.035,'128 scenes per condition, both solvers, no pose prior and no baseline fallback. This is an analytical geometry diagnostic, not rendered RGB or a physical visibility claim.',ha='center',fontsize=9)
    fig.tight_layout();finish(fig,'03_mask_stress.png')


def chart_learning():
    logs=[json.loads(s) for s in (C.DOC/'TRAIN_LOGS.jsonl').read_text().splitlines() if s.strip()]
    counts=C.read(C.DOC/'CORRESPONDENCE_COUNTS.json');raw=rows('LEARNED_PREDICTIONS.jsonl.gz');arms=ARMS[8:11];colors=['#c7792d','#9b5aaa','#4d9565']
    fig,axs=plt.subplots(2,2,figsize=(14,8))
    for arm,color in zip(arms,colors):
        train=[r for r in logs if r['kind']=='formal' and r['arm']==arm]
        test=[r for r in logs if r['kind']=='source_curve' and r['arm']==arm]
        axs[0,0].plot([r['step'] for r in train],[r['loss_image_average'] for r in train],c=color,label=NAMES[arm])
        axs[0,1].plot([r['step'] for r in test],[100*r['source_test']['positive_adoption_rate'] for r in test],'o-',c=color,label=NAMES[arm])
        h=counts[arm]['corners_histogram'];axs[1,0].plot(np.arange(9),[int(h.get(str(i),0)) for i in range(9)],'o-',c=color,label=NAMES[arm])
    labels=['Geometry\nonly','Image\nno role','Image + role','Image + role\npoint + line']
    new=[sum(r['new_pose_estimated'] for r in raw if r['method']==a) for a in arms+['IMAGE_ROLE_POINT_LINE']]
    fallback=[sum(r['fallback_used'] for r in raw if r['method']==a) for a in arms+['IMAGE_ROLE_POINT_LINE']]
    x=np.arange(4);axs[1,1].bar(x,new,color='#397cac',label='New pose');axs[1,1].bar(x,fallback,bottom=new,color='#e4ae6d',label='Base fallback')
    for i,(n,b) in enumerate(zip(new,fallback)):axs[1,1].text(i,170,f'{n} new\n{b} fallback',ha='center',fontsize=10)
    axs[0,0].set_xlabel('Formal updates (same seed / order / batch16)');axs[0,0].set_ylabel('Training loss, image average');axs[0,0].legend(fontsize=8)
    axs[0,1].set_xlabel('Formal updates');axs[0,1].set_ylabel('Positive target adopted on source test (%)');axs[0,1].set_ylim(0,100)
    axs[1,0].axvline(4,color='#777777',ls='--');axs[1,0].set_xticks(range(9));axs[1,0].set_xlabel('Observed corner intersections per real frame');axs[1,0].set_ylabel('Real frames / 319');axs[1,0].legend(fontsize=8)
    axs[1,1].set_xticks(x,labels,fontsize=9);axs[1,1].set_ylabel('Operational frames / 319');axs[1,1].legend(fontsize=8,ncol=2,loc='upper center',bbox_to_anchor=(.5,1.16))
    for ax in axs.flat:ax.grid(axis='y',alpha=.2)
    fig.suptitle('04 | Source learning worked; real observation coverage was insufficient',y=1.025)
    fig.text(.5,-.035,'Source test: 128 existing P0 families / 2,333 positive queries. Real: 319 frames. Positive adoption is not pose success; source scores did not select a model or setting.',ha='center',fontsize=9)
    fig.tight_layout();finish(fig,'04_learning_observation_gap.png')


def chart_runtime():
    runtime=C.read(C.DOC/'RUNTIME.json');arms=runtime['arms'];fig,ax=plt.subplots(figsize=(11,5.4));y=np.arange(4)
    mean=[runtime['summaries'][a]['full']['mean_ms'] for a in arms]
    median=[runtime['summaries'][a]['full']['median_ms'] for a in arms]
    p90=[runtime['summaries'][a]['full']['p90_ms'] for a in arms]
    ax.barh(y,mean,color=['#687485','#397cac','#247b65','#a47140'],height=.55,label='Mean')
    ax.scatter(median,y,marker='o',facecolor='white',edgecolor='black',s=58,label='Median',zorder=4)
    ax.scatter(p90,y,marker='|',c='black',s=330,label='P90',zorder=4)
    for i in y:ax.text(max(mean[i],p90[i])+1,i,f'{mean[i]:.2f} / {median[i]:.2f} / {p90[i]:.2f} ms',va='center')
    ax.set_yticks(y,[NAMES[a] for a in arms]);ax.invert_yaxis();ax.set_xlim(0,62);ax.grid(axis='x',alpha=.2)
    ax.set_xlabel('Measured whole path (ms): mean / median / P90');ax.legend(loc='lower right',fontsize=9)
    ax.set_title('05 | Actual detector → initial pose → correction → final pose / reprojection')
    fig.text(.5,-.02,'26 real frames × 5 repeats = 130 measured calls per arm; 20 warmups per arm. Quiet window, no coordinate replay. RGB decode and model load excluded.\nImage + role returned Base fallback on this panel; its speed does not demonstrate successful learned PnP.',ha='center',fontsize=9)
    fig.tight_layout();finish(fig,'05_runtime.png')


def crop_from_box(image,frame):
    h,w=image.shape[:2];box=np.asarray(frame['candidate_metadata']['box_xyxy'],float)
    l,t,r,b=box;pad=max(12,.06*max(r-l,b-t))
    return [max(0,int(l-pad)),max(0,int(t-pad)),min(w,int(np.ceil(r+pad))),min(h,int(np.ceil(b+pad)))]


def plot_network(ax,q,color,style='-',alpha=1,lw=1.25):
    q=np.asarray(q,float)
    for a,b in EDGES:
        if np.isfinite(q[[a,b]]).all():ax.plot(q[[a,b],0],q[[a,b],1],style,color=color,alpha=alpha,lw=lw)


def pose_projection(pose,K):
    from .solver import cuboid,project
    if not pose or not pose.get('available'):return np.full((8,2),np.nan)
    return project(cuboid(*pose['cf_extents']),pose['R_cf'],pose['centroid'],K)


def clipped_points(ax,q,crop,color,marker,label=None,size=50):
    q=np.asarray(q,float).reshape(-1,2);good=np.isfinite(q).all(1);q=q[good]
    l,t,r,b=crop;inside=(q[:,0]>=l)&(q[:,0]<r)&(q[:,1]>=t)&(q[:,1]<b)
    if inside.any():ax.scatter(q[inside,0],q[inside,1],s=size,c=color,marker=marker,label=label,zorder=6)
    for p in q[~inside]:
        a=np.clip(p,[l+5,t+5],[r-5,b-5]);ax.scatter(*a,s=size+12,c=color,marker='>',zorder=7)
    return int((~inside).sum())


def chart_cases():
    import cv2
    inputs={r['id']:r for r in C.read(C.DOC/'INPUTS.json')['frames']}
    fixed=indexed_rows('POSE_DIAGNOSTICS.jsonl.gz');learned=indexed_rows('LEARNED_PREDICTIONS.jsonl.gz');allrows={**fixed,**learned}
    corr=indexed_rows('REAL_CORRESPONDENCE_ROWS.jsonl.gz')
    target_path=C.ROOT/'data/pallet/results/pallet_posefix_replay_diagnosis_v1/TARGETS.json';targets=C.read(target_path)
    def select(predicate,score):
        pool=[v for v in corr.values() if predicate(v[1])];assert pool
        return sorted(pool,key=lambda a:(-score(a[1]),a[1]['id']))[0][1]
    selections=[('Wrong mask; both errors improve',select(lambda r:r['method']=='N3_SUBPIX_GEOM_NOSELF_ROBUST' and r['mask_pose_outcome']=='wrong_mask:both_improved',lambda r:-r['translation_delta_cm']),
         'N3+SubPix geometry robust; known-human mask wrong; both pose deltas <0; greatest position reduction, ID tie-break'),
       ('Correct known mask; both errors worsen',select(lambda r:r['method']=='N3_SUBPIX_GEOM_NOSELF_ROBUST' and r['mask_pose_outcome']=='correct_mask_on_known:both_worsened',lambda r:r['translation_delta_cm']),
         'N3+SubPix geometry robust; known-human mask correct; both pose deltas >0; greatest position increase, ID tie-break'),
       ('Low correct support; large pose error',select(lambda r:r['method']=='N3_SUBPIX_GEOM_NOSELF_ROBUST' and r['new_pose_estimated'] and r['correct_pool_count']<=2 and len(r['wrong_final_inlier_ids'])>=2 and r['translation_cm']>=100,lambda r:r['translation_cm']),
         'N3+SubPix geometry robust; new pose; <=2 reference-accurate pool points, >=2 wrong consensus inliers, position>=100cm; largest position error, ID tie-break'),
       ('Image + role: insufficient corners / fallback',None,
         'IMAGE_ROLE baseline fallback; non-all-no-match with 0 selected corners; smallest frame ID; observations exist but no corner intersection'),
       ('Same observations; point + line worsens',select(lambda r:r['method']=='IMAGE_ROLE_POINT_LINE' and r['new_pose_estimated'] and r['translation_delta_cm']>0 and r['rotation_delta_deg']>0,lambda r:r['translation_delta_cm']),
         'IMAGE_ROLE point+line new pose; both errors worse than Base; largest position increase, ID tie-break'),
       ('Human visible oracle improves (diagnostic)',select(lambda r:r['method']=='N3_SUBPIX_ORACLE_VISIBLE_ROBUST' and r['new_pose_estimated'] and r['translation_delta_cm']<0 and r['rotation_delta_deg']<0,lambda r:-r['translation_delta_cm']),
         'N3+SubPix human direct-visible oracle; both errors improve; largest position reduction, ID tie-break')]
    fallback=sorted([r for _,r in learned.values() if r['method']=='IMAGE_ROLE' and r['fallback_used'] and not r['all_no_match'] and not r['selected_corner_ids']],key=lambda r:r['id'])[0]
    selections[3]=(selections[3][0],corr[(fallback['id'],fallback['method'])][1],selections[3][2])
    observations={}
    # Lines are retained in public learned observations; no head is loaded here.
    selected_ids={r['id'] for _,r,_ in selections if r['method'].startswith('IMAGE_ROLE')}
    for i,r in enumerate(C.iter_rows(C.DOC/'LEARNED_OBSERVATIONS.jsonl.gz'),1):
        if r['id'] in selected_ids and r['method']=='IMAGE_ROLE':observations[r['id']]=(i,r)
    fig,axs=plt.subplots(3,2,figsize=(14.8,17.7));cases=[]
    for ax,(title,a,criterion) in zip(axs.flat,selections):
        key=(a['id'],a['method']);line,row=allrows[key];frame=inputs[a['id']];imagepath=C.ROOT/frame['image'];image=cv2.cvtColor(cv2.imread(str(imagepath)),cv2.COLOR_BGR2RGB)
        assert C.sha(imagepath)==frame['image_sha256'];crop=crop_from_box(image,frame);l,t,r,b=crop
        ax.imshow(image[t:b,l:r],extent=[l,r,b,t]);ax.set_xlim(l,r);ax.set_ylim(b,t);ax.set_aspect('equal');ax.axis('off')
        q=np.asarray(row['input_points'],float)[:8];arm='N3_SUBPIX' if row['method'].startswith('N3_SUBPIX') else 'BASE';initial=np.asarray(frame['points'][arm],float)[:8]
        plot_network(ax,initial,'#27c8e2',alpha=.6,lw=1)
        off_initial=clipped_points(ax,initial,crop,'#27c8e2','+',size=35)
        for i,p in enumerate(initial):
            if np.isfinite(p).all() and l<=p[0]<r and t<=p[1]<b:ax.text(*p,str(i),color='white',fontsize=9,weight='bold',bbox=dict(facecolor='black',alpha=.5,pad=.6,edgecolor='none'),zorder=8)
        hidden=row['hidden_initial'];excluded=row['excluded'];pool=a['pool_ids'];inlier=a['final_inlier_ids'];qvalid=np.isfinite(q).all(1)
        observed_ids=[i for i in pool if qvalid[i]];off_obs=clipped_points(ax,q[observed_ids],crop,'#ffde50','o',size=25)
        off_in=clipped_points(ax,q[[i for i in inlier if qvalid[i]]],crop,'#66ef8c','o',size=48)
        off_hidden=clipped_points(ax,initial[hidden],crop,'#ff585d','x',size=78)
        final=pose_projection(row['actual_pose'],frame['K']);plot_network(ax,final,'#ff7e25',lw=1.7);off_proj=clipped_points(ax,final,crop,'#ff7e25','.',size=22)
        reference=targets[a['id']]['truth'];from .solver import project,cuboid
        refq=project(cuboid(*reference['xyz']),reference['R'],reference['t'],frame['K']);plot_network(ax,refq,'#f589ef',style='--',lw=1.2)
        line_edges=[];observation_binding=None
        if a['id'] in observations:
            obsline,obs=observations[a['id']];packet=obs.get('observation',obs)
            for o in packet.get('lines',[]):
                support=np.asarray(o.get('support_points',[]),float)
                if support.size:ax.scatter(support[:,0],support[:,1],s=8,c='#ffde50',zorder=4)
                line_edges.append(o['edge'])
            observation_binding=dict(file='LEARNED_OBSERVATIONS.jsonl.gz',line=obsline,key_sha256=C.digest(obs))
        baseline=row['baseline_pose'];new=row['pose'];metric=f'T {baseline["translation_cm"]:.2f} → {new["translation_cm"]:.2f} cm | R {baseline["rotation_deg"]:.2f} → {new["rotation_deg"]:.2f} deg'
        status='NEW_POSE' if row['new_pose_estimated'] else 'BASE_FALLBACK' if row['fallback_used'] else 'NO_POSE'
        support=f'pool {pool}; inliers {inlier}; H {hidden}; excluded {excluded}'
        if row.get('local_point_line_refinement'):support=f'point IDs {row["solver"].get("point_ids",[])}; line edges {row["solver"].get("line_edges",[])}; H {hidden}'
        titletext=title+'\n'+a['id']+'\n'+metric+' | '+status
        ax.set_title(titletext,fontsize=11,loc='left',pad=8)
        ax.text(.01,.015,support+'\n'+f'correct pool {a["correct_pool_count"]}; correct/wrong inliers {a["correct_final_inlier_count"]}/{len(a["wrong_final_inlier_ids"])}; off-ROI final {off_proj}/8',
                transform=ax.transAxes,fontsize=8,color='white',va='bottom',bbox=dict(facecolor='black',alpha=.75,pad=3,edgecolor='none'))
        cases.append(dict(panel=len(cases)+1,title=title,id=a['id'],method=a['method'],selection=criterion,posthoc_illustration_not_performance_evidence=True,
            image=C.binding(imagepath),crop_xyxy=crop,frozen_input_coordinate_arm=arm,
            orange_overlay='complete actual_pose projection for review; final native output replaces only excluded hidden coordinates after a new fit',
            raw_pose=dict(file='LEARNED_PREDICTIONS.jsonl.gz' if key in learned else 'POSE_DIAGNOSTICS.jsonl.gz',line=line,key_sha256=C.digest(row)),
            correspondence=dict(file='REAL_CORRESPONDENCE_ROWS.jsonl.gz',line=corr[key][0],key_sha256=C.digest(a)),
            observation=observation_binding,reference_key_sha256=C.digest(targets[a['id']]),reference='dashed reconstructed evaluation reference; not a new independent 6D measurement',
            metric_baseline=baseline,metric_final=new,status=row['output_status'],display_status=status,hidden_ids=hidden,excluded_ids=excluded,observed_pool_ids=pool,final_inlier_ids=inlier,
            off_crop_counts=dict(initial=off_initial,observed=off_obs,inlier=off_in,hidden=off_hidden,final_projection=off_proj),
            point_PnP_inliers_applicable=a['point_PnP_inliers_applicable'],selected_query_count=row.get('selected_queries'),all_no_match=row.get('all_no_match'),line_edges=line_edges))
    legend=[Line2D([],[],color='#27c8e2',marker='+',label='Frozen input (Base or N3+SubPix) + native IDs'),Line2D([],[],color='#ffde50',marker='o',label='Selected observations / line support'),Line2D([],[],color='#66ef8c',marker='o',ls='',label='Final point consensus inliers'),Line2D([],[],color='#ff585d',marker='x',ls='',label='Hidden set (oracle panel uses human labels)'),Line2D([],[],color='#ff7e25',label='Full final pose projection (review overlay)'),Line2D([],[],color='#f589ef',ls='--',label='Reconstructed GT pose (review only)')]
    fig.legend(handles=legend,ncol=3,loc='lower center',bbox_to_anchor=(.5,.005),fontsize=9)
    fig.suptitle('06 | Deterministic posthoc real-image cases: observations, consensus and pose',y=.999)
    fig.text(.5,.048,'Crops use the frozen detector box. Boundary > marks an off-ROI point. Orange is the full pose projection; actual native output replaces only excluded hidden coordinates after a new fit.\nAll large errors remain in the 319-frame statistics. A fallback leaves the frozen Base output unchanged.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.025,right=.99,top=.94,bottom=.095,hspace=.43,wspace=.045);finish(fig,'06_real_cases.png')
    return cases,C.binding(target_path)


def query_geometry(points):
    centers=[];normals=[]
    for a,b in EDGES:
        d=points[b]-points[a];n=np.array([-d[1],d[0]])/max(np.linalg.norm(d),1e-30)
        for u in np.arange(1,8)/8:centers.append((1-u)*points[a]+u*points[b]);normals.append(n)
    return np.asarray(centers),np.asarray(normals)


def chart_supervision():
    import cv2
    from . import source_audit as S
    cache=C.SCRATCH/'learned_cache';families=S.selected_families()
    if (cache/'CACHE_MANIFEST.json').exists():
        manifest=C.read(cache/'CACHE_MANIFEST.json');records=manifest['records']
        arrays={name:np.load(cache/(name+'.npy'),mmap_mode='r') for name in ['lo','hi','weight','valid']}
        cachebindings=[C.binding(cache/'CACHE_MANIFEST.json')]+[C.binding(cache/(k+'.npy')) for k in arrays]
    else:
        # The published three target rows are sufficient for these review
        # images; the large frozen detector feature cache is unnecessary.
        previous=C.read(C.DOC/'VISUAL_REVIEW_CASES.json');records=[r['retained_cache_record'] for r in previous['source_cases']]
        arrays={k:{r['index']:np.asarray(r['retained_target_row'][k]) for r in previous['source_cases']} for k in ['lo','hi','weight','valid']}
        cachebindings=previous['private_cache_bindings']
    test=[r for r in records if r['partition']=='source_test' and r['target_counts'].get('positive',0)>0]
    selection=[('First eligible source-test family',min(test,key=lambda r:r['id']))]
    selection.append(('Most positive queries in source-test',sorted([r for r in test if r!=selection[0][1]],key=lambda r:(-r['target_counts']['positive'],r['id']))[0]))
    selection.append(('Most no-match queries in source-test',sorted([r for r in test if r not in [x[1] for x in selection]],key=lambda r:(-r['target_counts'].get('no_match',0),r['id']))[0]))
    vertices,tri,mesh=S.mesh_normalized();audit={r['id']:r for r in C.read(C.DOC/'SYNTH_SUPERVISION_AUDIT.json')['panel']}
    maskcache=C.SCRATCH/'visual_review_auxiliary_masks';maskcache.mkdir(exist_ok=True)
    journal=maskcache/'EXECUTED_PASSES.json'
    if journal.exists():execution=C.read(journal)
    else:
        previous=C.read(C.DOC/'VISUAL_REVIEW_CASES.json')['review_auxiliary_actual_mesh_mask_ray_passes'] if (C.DOC/'VISUAL_REVIEW_CASES.json').exists() else 0
        execution=dict(actual_auxiliary_mask_ray_passes=previous,previous_unpersisted_passes=previous,new_training_RGB=0)
    newpasses=0
    fig,axs=plt.subplots(3,4,figsize=(16.2,11.8),gridspec_kw={'width_ratios':[1.35,1.35,1.35,1]});result=[]
    for axis,(criterion,record) in zip(axs,selection):
        i=record['index'];family=families[i];paths=S.locate(family);g=S.geometry(family);rgb=cv2.cvtColor(cv2.imread(str(paths['rgb'])),cv2.COLOR_BGR2RGB)
        visible=cv2.imread(str(paths['visible']),0)>127;amodal=cv2.imread(str(paths['amodal']),0)>127
        maskpath=maskcache/(record['id']+'.npy')
        if maskpath.exists():actual=np.load(maskpath)
        else:
            scene=S.ray_scene(vertices,tri,g['dims']);actual=S.render_mask(scene,g['K'],g['R'],g['t'],g['hw']);np.save(maskpath,actual)
            execution['actual_auxiliary_mask_ray_passes']+=1;newpasses+=1;C.write(journal,execution)
        iou=float((actual&amodal).sum()/max(1,(actual|amodal).sum()))
        assert iou>=.995
        y,x=np.where(amodal);pad=20;l=max(0,int(x.min())-pad);r=min(rgb.shape[1],int(x.max())+pad+1);t=max(0,int(y.min())-pad);b=min(rgb.shape[0],int(y.max())+pad+1);crop=[l,t,r,b]
        points=np.asarray(record['selected_points'],float);centers,normals=query_geometry(points)
        lo=arrays['lo'][i];hi=arrays['hi'][i];weight=arrays['weight'][i];valid=arrays['valid'][i]
        positive=valid&(lo!=65);none=valid&(lo==65);ignore=~valid
        from .solver import project
        sourceuv=project(g['X'],g['R'],g['t'],g['K'])
        for ax in axis[:3]:ax.imshow(rgb[t:b,l:r],extent=[l,r,b,t]);ax.set_xlim(l,r);ax.set_ylim(b,t);ax.axis('off')
        plot_network(axis[0],points[:8],'#27c8e2',lw=.9)
        for k,color in [(positive,'#66ef8c'),(none,'#ffbf48'),(ignore,'#ff585d')]:
            ps=centers[k];axis[0].scatter(ps[:,0],ps[:,1],s=10,c=color,alpha=.8)
        axis[0].set_title(record['id']+'\nExisting original P0 RGB + cached query labels',fontsize=9,loc='left')
        axis[0].text(.01,.02,f'positive {positive.sum()} | no match {none.sum()} | ignore {ignore.sum()}\n1024 families split 768 / 128 / 128; this is source test',transform=axis[0].transAxes,fontsize=8,color='white',bbox=dict(facecolor='black',alpha=.7,pad=2))
        for mask,color in [(amodal,'#fd9cff'),(visible,'#60f198')]:
            axis[1].contour(np.arange(l,r),np.arange(t,b),mask[t:b,l:r],levels=[.5],colors=[color],linewidths=1.3)
        axis[1].set_title('Delivered masks: visible (green), amodal (pink)\nPhysical holes / rounded shape are preserved',fontsize=9)
        axis[2].contour(np.arange(l,r),np.arange(t,b),actual[t:b,l:r],levels=[.5],colors=['#ffde50'],linewidths=1.3)
        # Native corner ordering changes the edge IDs. Physical top/bottom
        # footprint edges have equal canonical height; rounded vertical hull
        # corners do not exist. This mapping was checked against actual mesh.
        physical_edges=[e for e,(aa,bb) in enumerate(EDGES) if np.isclose(g['X'][aa,1],g['X'][bb,1])]
        ignored_edges=sorted(set(range(12))-set(physical_edges));assert len(physical_edges)==8 and len(ignored_edges)==4
        for e,(aa,bb) in enumerate(EDGES):axis[2].plot(sourceuv[[aa,bb],0],sourceuv[[aa,bb],1],color='#66ef8c' if e in physical_edges else '#ff585d',ls='-' if e in physical_edges else '--',lw=1.25)
        axis[2].set_title(f'Actual USD mesh silhouette (yellow); IoU {iou:.5f}\n8 physical footprint edges; 4 nonphysical verticals IGNORE',fontsize=9)
        axis[3].axis('off');patchrecords=[]
        # Three unmodified-RGB resampled stripes: stored positive, none, ignored.
        for j,(kind,mask) in enumerate([('positive',positive),('no match',none),('IGNORE',ignore)]):
            ids=np.flatnonzero(mask)
            if kind=='IGNORE':ids=np.asarray([q for q in ids if q//7 in ignored_edges]);assert len(ids)
            qid=int(ids[len(ids)//2]);normal=normals[qid];tangent=np.array([normal[1],-normal[0]])
            offset=np.arange(-32,33);cross=np.arange(-10,11);grid=centers[qid]+offset[None,:,None]*normal+cross[:,None,None]*tangent
            patch=cv2.remap(rgb,grid[:,:,0].astype(np.float32),grid[:,:,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            inset=axis[3].inset_axes([.03,.72-j*.31,.94,.18]);inset.imshow(patch,extent=[-32,32,10,-10]);inset.set_yticks([]);inset.set_xticks([-32,0,32]);inset.tick_params(labelsize=7)
            label=f'Q{qid}: {kind}'
            targetoffset=None
            if kind=='positive':
                targetoffset=float(lo[qid]+weight[qid]-32);inset.axvline(targetoffset,color='#66ef8c',lw=2);label+=f' / target {targetoffset:+.2f}px'
            elif kind=='no match':label+=' / class 65 (no coordinate)'
            else:label+=' / no training target'
            inset.set_title(label,fontsize=8,loc='left');patchrecords.append(dict(query_id=qid,kind=kind,center_xy=centers[qid].tolist(),normal=normal.tolist(),target_offset_px=targetoffset,lo=int(lo[qid]),hi=int(hi[qid]),weight=float(weight[qid]),valid=bool(valid[qid])))
        result.append(dict(panel=len(result)+1,id=record['id'],family=record['family'],partition=record['partition'],index=i,selection=criterion,
            RGB=C.binding(paths['rgb']),label=C.binding(paths['label']),visible_mask=C.binding(paths['visible']),amodal_mask=C.binding(paths['amodal']),crop_xyxy=crop,
            cache_record_sha256=C.digest(record),query_target_row_sha256=C.digest({k:np.asarray(v[i]).tolist() for k,v in arrays.items()}),
            retained_cache_record=record,retained_target_row={k:np.asarray(v[i]).tolist() for k,v in arrays.items()},
            target_counts=record['target_counts'],patches=patchrecords,actual_mesh_amodal_IoU=iou,mesh=mesh['asset'],
            physical_edge_ids=physical_edges,unsupported_vertical_edge_ids=ignored_edges,
            illustration_only_distinct_auxiliary_mask=1,new_RGB_created=0,prior_128_audit_record=audit.get(record['id'])))
    fig.suptitle('07 | Exact existing source supervision: actual pallet geometry, masks and candidate targets',y=.995)
    fig.text(.5,.01,'Labels are read from retained source-only targets. Green = positive; amber = no match; red = IGNORE. No detector, head or pose solver was rerun.\nThree actual mesh silhouettes, for review only: six mask ray passes executed during figure preparation; no training RGB. Dashed red verticals are physically unsupported and never positive labels.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.045,1,.97]);finish(fig,'07_supervision_cases.png')
    return result,cachebindings,execution,newpasses


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--charts-only',action='store_true');args=parser.parse_args()
    style();FIG.mkdir(parents=True,exist_ok=True)
    for fn in [chart_pose,chart_paired,chart_stress,chart_learning,chart_runtime]:fn()
    if args.charts_only:
        print('CHARTS_ONLY_DONE',sum(p.stat().st_size for p in FIG.glob('0[1-5]_*.png')),flush=True);return
    real,reference=chart_cases();source,cache,visualcounts,newpasses=chart_supervision()
    bindingnames=['INPUTS.json','FIXED_CONTROLS.jsonl.gz','POSE_DIAGNOSTICS.jsonl.gz','LEARNED_PREDICTIONS.jsonl.gz','LEARNED_OBSERVATIONS.jsonl.gz','REAL_CORRESPONDENCE_ROWS.jsonl.gz','PAIRED_COMPARISONS.json','GEOMETRY_STRESS_SUMMARY.json','GEOMETRY_STRESS.jsonl.gz','TRAIN_LOGS.jsonl','CORRESPONDENCE_COUNTS.json','RUNTIME.json','SOURCE_FAMILY_SPLIT.json','SYNTH_SUPERVISION_AUDIT.json']
    figures=[dict(file=p.name,binding=C.binding(p)) for p in sorted(FIG.glob('0[1-7]_*.png'))]
    size=sum(p.stat().st_size for p in FIG.glob('0[1-7]_*.png'));assert len(figures)==7 and size<8_000_000
    C.write(C.DOC/'VISUAL_REVIEW_CASES.json',dict(schema='sealed_measurements_scientific_review_v1',
        full_population=319,not_a_new_experiment=True,no_detector_calls=0,no_head_forward_calls=0,no_training_updates=0,no_PnP_calls=0,
        review_auxiliary_actual_mesh_mask_ray_passes=visualcounts['actual_auxiliary_mask_ray_passes'],review_distinct_actual_mesh_masks=3,
        review_auxiliary_passes_current_run=newpasses,review_unpersisted_first_render_passes=visualcounts['previous_unpersisted_passes'],
        new_training_RGB=0,no_hull_used_as_pallet_mask=True,
        derivative_PNG_total_bytes=size,figures=figures,real_cases=real,source_cases=source,
        raw_bindings=[C.binding(C.DOC/name) for name in bindingnames],reference_binding=reference,private_cache_bindings=cache,
        script=C.binding(Path(__file__)),limitations=['Posthoc examples illustrate recorded mechanisms; they do not replace all319 statistics.',
         'Reconstructed evaluation GT pose is dashed; no new independent real 6D measurement was obtained.',
         'Visible human corners and coordinates within8px are distinct tests; unknown/unannotated corners remain unknown.',
         'Actual mesh support is established for the audited P0 scene.usd subset; no new RGB or controlled four-variant claim.',
         'Candidate stripes resample existing RGB for display only; targets are copied from the retained source-only cache.']))
    print('FULL_VISUAL_REVIEW_DONE',size,flush=True)


if __name__=='__main__':main()
