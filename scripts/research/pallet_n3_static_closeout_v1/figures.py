"""Figures from measured data and approved static RGB only; no generated imagery."""
import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from .compute import C,R,M,DOC,RAW,SOURCE,write,bind
from scripts.research.pallet_n3_completion_v3 import evaluation as E

def main():
    out=DOC/'figures';out.mkdir(exist_ok=True);provenance={}
    core=C.read(DOC/'CORE_RESULTS.json')['methods'];datasets={};datasets['YOLO']=[core['R0']['mean']['headline'],core['N3_DIM_SYM']['mean']['headline']]
    for b,name in [('dope','DOPE'),('resnet18','ResNet18\n10ep CONSTANT-fold')]:
        d=C.read(C.RAW/f'evaluation/{b}.json');datasets[name]=[d['methods']['base']['headline'],d['seed_aggregate']['mean_of_seed_metrics']]
    fields=[('matched_pooled_corner8_median_px','Conditional corner median (px)',1),('matched_pooled_corner8_P90_px','Conditional corner P90 (px)',1),('full_PCK10_fraction','Full corner PCK10 (%)',100),('pose_translation_cm_median','Translation median (cm)',1),('pose_rotation_deg_median','Rotation median (deg)',1),('full_penalty_P90_px','Full-penalty corner P90 (px)',1)]
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained');x=np.arange(3)
    for ax,(field,label,mul) in zip(axes.flat,fields):
        for i,(name,color) in enumerate([('Base','#536579'),('N3','#158579')]):ax.bar(x+(i-.5)*.32,[h[i][field]*mul for h in datasets.values()],.32,label=name,color=color)
        ax.set_xticks(x,list(datasets));ax.set_ylabel(label);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2)
    axes[0,0].legend();fig.suptitle('DEV319: within-backbone correction; three-seed statistic means')
    fig.savefig(out/'backbone_results.png',dpi=160);plt.close(fig)
    provenance['backbone_results.png']={'sources':[bind(DOC/'CORE_RESULTS.json'),*[bind(C.RAW/f'evaluation/{b}.json') for b in ['dope','resnet18']]],'selection':'all three backbones; means of all three seeds'}
    paired=C.read(DOC/'PAIRED_POSE_ANALYSIS.json');fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for ax,key in zip(axes.flat,['translation_cm_median','rotation_deg_median','yaw_deg_median','translation_cm_P90','rotation_deg_P90','yaw_deg_P90']):
        for i,(b,d) in enumerate(paired['backbones'].items()):
            h=d['mean_of_seed_statistics'][key];lo,hi=h['CI95'];ax.plot([lo,hi],[i,i],color='#158579',lw=2);ax.plot(h['mean_seed_delta'],i,'o',color='#18374b')
        ax.axvline(0,color='gray',ls='--');ax.set_yticks(range(3),['YOLO','DOPE','ResNet18']);ax.set_title(key.replace('_',' '));ax.set_xlabel('N3 minus Base');ax.grid(alpha=.2)
    fig.suptitle('Session-paired 95% intervals (10,000 draws); posthoc reused DEV')
    fig.savefig(out/'pose_uncertainty.png',dpi=160);plt.close(fig);provenance['pose_uncertainty.png']={'sources':[bind(DOC/'PAIRED_POSE_ANALYSIS.json')],'selection':'all six pose contrasts, no clipped CI limits'}
    y=C.read(RAW/'YOLO_SCORES.json');fig,axes=plt.subplots(1,4,figsize=(15,3.8),layout='constrained');thresholds=np.arange(0,801)
    for ax,group in zip(axes,['clean','moderate','severe','unclassified']):
        for name,method,color in [('Base','R0','#536579'),('N3','N3_DIM_SYM','#158579')]:
            seeds=[method] if method=='R0' else [method+f'_seed{s}' for s in (1,2,3)];curves=[]
            for key in seeds:
                err=np.array([e for r in y[key]['corner_scores'] if r['occlusion']==group for e in r['errors']]);curves.append((err[:,None]<=thresholds).mean(0)*100)
            ax.plot(thresholds,np.mean(curves,axis=0),color=color,label=name)
        ax.set_title(group);ax.set_xlabel('Error threshold (px)');ax.set_ylim(0,101);ax.set_xlim(0,800);ax.grid(alpha=.2)
    axes[0].set_ylabel('Full-reference corners within threshold (%)');axes[0].legend();fig.suptitle('YOLO: recorded occlusion strata, including unknown labels and full tails')
    fig.savefig(out/'occlusion_pck.png',dpi=160);plt.close(fig);provenance['occlusion_pck.png']={'sources':[bind(RAW/'YOLO_SCORES.json')],'threshold_axis':'full0..800 image-diagonal range; no tail cropped','mode':'full-reference PCK'}
    # Deterministic posthoc illustration: first lexicographic approved-label ID in
    # each signed seed1 change category, all categories retained including absence.
    truth,_=R.load_dev_context(SOURCE,include_pose=False);by={r['id']:r for r in truth};R.attach_dcp(truth,SOURCE,'N3_DIM_SYM',1)
    normalized={}
    for b in ['dope','resnet18']:normalized[b]=E.normalize_prediction_payload(C.read(C.RAW/f'predictions/{b}_DEV319.json'))['methods']
    fig,axes=plt.subplots(3,3,figsize=(13,10),layout='constrained');selection=[]
    for bi,b in enumerate(['yolo','dope','resnet18']):
        if b=='yolo':bs=y['R0']['corner_scores'];ns=y['N3_DIM_SYM_seed1']['corner_scores']
        else:
            d=C.read(C.RAW/f'evaluation/{b}.json');bs=d['methods']['base']['result']['corner_rows'];ns=d['methods']['n3_seed1']['result']['corner_rows']
        candidates={k:[] for k in ['improved','unchanged','worsened']}
        for before,after in zip(bs,ns):
            assert before['id']==after['id'];fid=before['id']
            if by[fid]['occlusion']=='unclassified' or not before['evaluable']:continue
            delta=after['frame_mean_px']-before['frame_mean_px'];key='improved' if delta< -1e-9 else 'worsened' if delta>1e-9 else 'unchanged';candidates[key].append((fid,delta))
        for gi,g in enumerate(['improved','unchanged','worsened']):
            ax=axes[bi,gi];ax.axis('off')
            if not candidates[g]:ax.text(.5,.5,f'{b}: {g}\nNo qualifying frame',ha='center',va='center');selection.append({'backbone':b,'category':g,'id':None});continue
            fid,delta=sorted(candidates[g])[0];r=by[fid];img=cv2.imread(str(SOURCE/r['_image_path']));ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            if b=='yolo':p=r['predictions']['R0'];a=r['predictions']['N3_DIM_SYM_seed1']
            else:p=normalized[b]['base'][fid]['points'];a=normalized[b]['n3_seed1'][fid]['points']
            for points,marker,color,label in [(r['gt'],'+','#fff04c','Reference'),(p,'o','#00ceff','Base'),(a,'x','#ff4576','N3 seed1')]:
                if points is None:continue
                points=np.array(points);ok=np.isfinite(points[:8]).all(-1)
                if label=='Reference':ok&=np.array(r['valid'][:8])
                ax.scatter(points[:8][ok,0],points[:8][ok,1],s=20,marker=marker,c=color,label=label,linewidths=.8)
            ax.set_xlim(0,img.shape[1]);ax.set_ylim(img.shape[0],0);ax.set_title(f'{b} | {g} | delta {delta:+.3g}px\n{fid}',fontsize=8)
            selection.append({'backbone':b,'category':g,'id':fid,'delta_frame_mean_px':delta,'approved_occlusion':r['occlusion'],'image':bind(SOURCE/r['_image_path'])})
    handles,legends=axes[0,0].get_legend_handles_labels();fig.legend(handles,legends,loc='outside lower center',ncol=3)
    fig.suptitle('Static examples: deterministic posthoc selection; predicted corners, not PnP reprojections')
    fig.savefig(out/'static_examples.png',dpi=160);plt.close(fig)
    write(DOC/'EXAMPLE_SELECTION.json',{'rule':'seed1, approved direct-review IDs only; signed full frame-mean corner change with locked1e-9 tolerance; first lexicographic ID per category; no favorable filtering','posthoc':True,'rows':selection,'qualitative_only':True})
    provenance['static_examples.png']={'sources':[bind(DOC/'EXAMPLE_SELECTION.json')],'reference':'locked image/geometry-derived corners; no independent physical truth','not_PnP_reprojected':True}
    write(DOC/'FIGURE_PROVENANCE.json',provenance)

if __name__=='__main__':main()
