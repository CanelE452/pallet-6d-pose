"""Derived scientific figures from sealed rows; no fit, ray or inference."""
from pathlib import Path
from collections import Counter
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ..pallet_observation_refiner_20261009_v1 import common as C
from ..pallet_observation_refiner_20261009_v1.visual_review import crop_from_box,plot_network,clipped_points,pose_projection

DOC=C.WORKTREE/'_docs/experiments/pallet_kp_difficulty_20261010_v1'
FIG=DOC/'figures'
LABELS={'BASE':'Base','N3_SUBPIX':'N3 + SubPix',
 'BASE_NO_MASK_ROBUST':'Base / unlocked / no mask','BASE_GEOM_NOSELF_ROBUST':'Base / unlocked / mask',
 'N3_SUBPIX_NO_MASK_ROBUST':'N3+SubPix / unlocked / no mask','N3_SUBPIX_GEOM_NOSELF_ROBUST':'N3+SubPix / unlocked / mask',
 'BASE_INITIAL_DIMENSION_PRIOR_NO_MASK':'Base / dimension prior / no mask',
 'BASE_INITIAL_DIMENSION_PRIOR_GEOM_NOSELF':'Base / dimension prior / mask',
 'N3_SUBPIX_INITIAL_DIMENSION_PRIOR_NO_MASK':'N3+SubPix / dimension prior / no mask',
 'N3_SUBPIX_INITIAL_DIMENSION_PRIOR_GEOM_NOSELF':'N3+SubPix / dimension prior / mask',
 'GEOMETRY_ONLY_MATCH_MASS':'Geometry / match mass','IMAGE_NO_ROLE_MATCH_MASS':'Image, no role / match mass',
 'IMAGE_ROLE_MATCH_MASS':'Image + role / match mass','IMAGE_ROLE_MATCH_MASS_NO_MASK':'Image + role / match mass / no mask'}

def save(fig,name):
    fig.savefig(FIG/name,dpi=130,bbox_inches='tight',facecolor='white');plt.close(fig)

def main():
    assert not (DOC/'VISUAL_REVIEW.json').exists()
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    new=list(C.iter_rows(DOC/'PREDICTIONS.jsonl.gz'))
    old=list(C.iter_rows(C.DOC/'FIXED_CONTROLS.jsonl.gz'))+list(C.iter_rows(C.DOC/'PREDICTIONS.jsonl.gz'))+list(C.iter_rows(C.DOC/'LEARNED_PREDICTIONS.jsonl.gz'))
    data={(r['method'],r['id']):r for r in old+new}
    arms=['BASE','N3_SUBPIX','N3_SUBPIX_NO_MASK_ROBUST','N3_SUBPIX_GEOM_NOSELF_ROBUST']+C.read(DOC/'METRICS.json')['new_methods']
    fig,axes=plt.subplots(1,3,figsize=(17,8),gridspec_kw={'width_ratios':[1.6,1.3,1]});y=np.arange(len(arms))
    for ax,key,unit in zip(axes[:2],['translation_cm','rotation_deg'],['Position error (cm)','Rotation error (deg)']):
        for i,arm in enumerate(arms):
            rs=[r for (m,_),r in data.items() if m==arm]
            vals=np.asarray([r['pose'][key] for r in rs if r['pose']['available']]);assert len(vals)==319
            lo,med,hi=np.quantile(vals,[.1,.5,.9]);mean=vals.mean();color='#316f99' if i<4 else '#8a5c99'
            ax.scatter(vals,np.full(len(vals),i),s=3,c=color,alpha=.10)
            ax.plot([lo,hi],[i,i],c=color,lw=3);ax.scatter(med,i,c=color,s=30);ax.scatter(mean,i,marker='D',facecolor='white',edgecolor='black',zorder=4)
            ax.text(.99,i+.24,f'{mean:.2f} / {med:.2f} / {hi:.2f}',transform=ax.get_yaxis_transform(),ha='right',fontsize=8)
        ax.set_xscale('symlog',linthresh=1);ax.set_xlabel(unit+'; mean / median / P90');ax.set_yticks(y,[LABELS[a] for a in arms] if ax is axes[0] else []);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    for i,arm in enumerate(arms):
        rs=[r for (m,_),r in data.items() if m==arm];n=sum(r['new_pose_estimated'] for r in rs);b=sum(r['fallback_used'] for r in rs)
        axes[2].barh(i,n,color='#316f99');axes[2].barh(i,b,left=n,color='#dfaa6b')
        if n+b==0:axes[2].barh(i,319,color='#9ba2ab')
        axes[2].text(159,i,f'{n} new / {b} fallback' if n+b else '319 fixed outputs',ha='center',va='center',fontsize=8)
    axes[2].set_xlim(0,319);axes[2].set_yticks(y,[]);axes[2].invert_yaxis();axes[2].set_xlabel('All319 frames; zero complete failures')
    fig.suptitle('Whole operational population: increased output coverage did not improve pose',y=1.02)
    fig.tight_layout();save(fig,'01_pose_and_coverage.png')
    ceiling=C.read(DOC/'SOURCE_CEILING.json');source=C.read(DOC/'SOURCE_MATCH_MASS_ANALYSIS.json')['summary'];mass=C.read(DOC/'MATCH_MASS_DECODE_CHECKS.json')['counts']
    fig,axs=plt.subplots(1,3,figsize=(15.8,5.5));heads=['GEOMETRY_ONLY','IMAGE_NO_ROLE','IMAGE_ROLE'];x=np.arange(3)
    a=[source[h]['source_test']['ORIGINAL_66_CLASS_MAP']['before_mask_ge4'] for h in heads]
    b=[source[h]['source_test']['FIXED_BINARY_EXISTENCE_MAP']['before_mask_ge4'] for h in heads]
    axs[0].bar(x-.17,a,.34,label='Original66-way');axs[0].bar(x+.17,b,.34,label='Fixed match mass');axs[0].axhline(13,c='#555',ls='--',label='Recorded ideal target ceiling13/128')
    axs[0].set_ylim(0,18);axs[0].set_ylabel('Source-test frames with >=4 raw corners /128');axs[0].legend(fontsize=8)
    for i,h in enumerate(heads):
        for dec,dx,col in [('ORIGINAL_66_CLASS_MAP',-.17,'#316f99'),('FIXED_BINARY_EXISTENCE_MAP',.17,'#98619c')]:
            s=source[h]['source_test'][dec];axs[1].bar(i+dx,100*s['positive_adoption_rate'],.34,color=col)
            axs[1].text(i+dx,100*s['positive_adoption_rate']+1,f"{100*s['no_match_false_acceptance']:.1f}%",ha='center',fontsize=9)
    axs[1].set_ylim(0,80);axs[1].set_ylabel('Recorded positive adoption (%)');axs[1].set_title('Text above bars: recorded NONE false acceptance')
    axs[2].bar(x-.17,[mass[h]['old']['four_raw_corner_frames'] for h in heads],.34,label='Original66-way')
    axs[2].bar(x+.17,[mass[h]['new']['four_raw_corner_frames'] for h in heads],.34,label='Fixed match mass')
    axs[2].set_ylabel('Real frames with >=4 raw corners /319');axs[2].legend(fontsize=8)
    for ax in axs:ax.set_xticks(x,['Geometry','Image no role','Image + role']);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Corner closure already fails on source; more mass is not proof of valid observations',y=1.05)
    fig.text(.5,-.06,'Source positive-label-only closure:0/128 for every model and both decoders. Recorded targets themselves are under separate actual-mesh ray audit.\nCounts are before image eligibility, predicted self-mask and final consensus; they are not pose success rates.',ha='center',fontsize=9)
    fig.tight_layout();save(fig,'02_source_and_real_bottlenecks.png')
    inputs={f['id']:f for f in C.read(C.DOC/'INPUTS.json')['frames']}
    post={(r['method'],r['id']):r for r in C.iter_rows(DOC/'POSTHOC_CORRESPONDENCE_ROWS.jsonl.gz')}
    targetpath=C.ROOT/'data/pallet/results/pallet_posefix_replay_diagnosis_v1/TARGETS.json';targets=C.read(targetpath)
    primary=[r for r in new if r['method']=='IMAGE_ROLE_MATCH_MASS' and r['new_pose_estimated']]
    best=min(primary,key=lambda r:(post[(r['method'],r['id'])]['translation_delta_cm'],r['id']))
    worst=max(primary,key=lambda r:(post[(r['method'],r['id'])]['translation_delta_cm'],r['id']))
    selections=[('Fixed: global branch ambiguity','plastic_night_01:038630','N3_SUBPIX'),
                ('Fixed: coherent inaccurate consensus','eval_pallet09:1778653806958839552','N3_SUBPIX'),
                ('Primary learned path: largest position reduction',best['id'],'BASE'),
                ('Primary learned path: largest position increase',worst['id'],'BASE')]
    mass_obs={r['id']:r for r in C.iter_rows(DOC/'MATCH_MASS_OBSERVATIONS.jsonl.gz') if r['method']=='IMAGE_ROLE_MATCH_MASS'}
    fig,axs=plt.subplots(4,3,figsize=(17,18));cases=[]
    for i,(title,fid,branch) in enumerate(selections):
        methods=[branch,branch+'_GEOM_NOSELF_ROBUST',branch+'_INITIAL_DIMENSION_PRIOR_GEOM_NOSELF'] if branch=='N3_SUBPIX' else ['BASE','IMAGE_ROLE','IMAGE_ROLE_MATCH_MASS']
        f=inputs[fid];imagepath=C.ROOT/f['image'];assert C.sha(imagepath)==f['image_sha256'];image=cv2.cvtColor(cv2.imread(str(imagepath)),cv2.COLOR_BGR2RGB)
        crop=crop_from_box(image,f);l,t,r,b=crop
        truth=targets[fid]['truth'];from ..pallet_observation_refiner_20261009_v1.solver import cuboid,project
        reference=project(cuboid(*truth['xyz']),truth['R'],truth['t'],f['K'])
        for j,m in enumerate(methods):
            row=data[(m,fid)];ax=axs[i,j];ax.imshow(image[t:b,l:r],extent=[l,r,b,t]);ax.set_xlim(l,r);ax.set_ylim(b,t);ax.set_aspect('equal');ax.axis('off')
            initial=np.asarray(f['points'][branch],float)[:8];plot_network(ax,initial,'#29c7dd',alpha=.6)
            for k,p in enumerate(initial):
                if np.isfinite(p).all() and l<=p[0]<r and t<=p[1]<b:ax.text(*p,str(k),color='white',fontsize=8,bbox=dict(facecolor='black',alpha=.5,pad=.5))
            q=np.asarray(row.get('input_points',f['points'][branch]),float)[:8];solver=row.get('solver',{});pool=solver.get('used',[]);inliers=solver.get('inliers',[]);hidden=row.get('hidden_initial',[])
            if pool:clipped_points(ax,q[pool],crop,'#ffdd42','o',size=20)
            if inliers:clipped_points(ax,q[inliers],crop,'#65ef8b','o',size=44)
            if hidden:clipped_points(ax,initial[hidden],crop,'#ff545b','x',size=65)
            pose=row.get('actual_pose',row.get('geometry_pose'));final=pose_projection(pose,f['K']) if pose else np.full((8,2),np.nan)
            plot_network(ax,final,'#ff8425',lw=1.6);clipped_points(ax,final,crop,'#ff8425','.',size=18);plot_network(ax,reference,'#f286ec',style='--')
            if m=='IMAGE_ROLE_MATCH_MASS':
                for line in mass_obs[fid]['lines']:
                    points=np.asarray(line['support_points']);ax.scatter(points[:,0],points[:,1],s=8,c='#ffdd42')
            p=row['pose'];status=row.get('output_status','FIXED_CONTROL');ax.set_title(f'{title if j==0 else ""}\n{m.replace("_INITIAL_DIMENSION_PRIOR_GEOM_NOSELF"," / dimension prior").replace("_GEOM_NOSELF_ROBUST"," / unlocked mask")}\nT {p["translation_cm"]:.2f}cm | R {p["rotation_deg"]:.2f}deg | {status}',fontsize=9,loc='left')
            a=post.get((m,fid));accuracy=f'; accurate pool {a["correct_pool_count"]}; accurate inliers {a["correct_final_inlier_count"]}' if a else ''
            ax.text(.01,.015,f'{fid}\npool{pool}; inliers{inliers}; H{hidden}{accuracy}',transform=ax.transAxes,color='white',fontsize=7,bbox=dict(facecolor='black',alpha=.75,pad=2))
            cases.append(dict(id=fid,method=m,selection=title,panel=[i+1,j+1],image=C.binding(imagepath),crop_xyxy=crop,
                row_sha256=C.digest(row),pose_metrics=row['pose'],status=status,posthoc_only=True,
                orange='full actual pose projection for review; native output replaces only excluded hidden after NEW_POSE',reference='GEOMETRIC_PROXY, not independent physical GT'))
    legend=[Line2D([],[],c='#29c7dd',label='Initial keypoints'),Line2D([],[],c='#ffdd42',marker='o',ls='',label='Selected observations'),Line2D([],[],c='#65ef8b',marker='o',ls='',label='Final inliers'),Line2D([],[],c='#ff545b',marker='x',ls='',label='Excluded hidden'),Line2D([],[],c='#ff8425',label='Full final pose'),Line2D([],[],c='#f286ec',ls='--',label='Reconstructed reference')]
    fig.legend(handles=legend,loc='lower center',ncol=3,fontsize=9)
    fig.subplots_adjust(top=.98,bottom=.06,hspace=.48,wspace=.08);save(fig,'03_real_difficulty_cases.png')
    C.write(DOC/'VISUAL_REVIEW.json',dict(complete=True,cases=cases,reference_binding=C.binding(targetpath),
        new_model_forwards=0,new_pose_fits=0,new_rays=0,new_RGB=0,source_RGB_reads_for_real_derivatives=4,
        images=[C.binding(p) for p in sorted(FIG.glob('0[123]_*.png'))],code=C.binding(Path(__file__))))
    print('VISUAL_REVIEW_COMPLETE',len(cases))

if __name__=='__main__':main()
