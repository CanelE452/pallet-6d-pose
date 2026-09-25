import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import common as C
def finish(name):
    p=C.DOC/'figures';p.mkdir(parents=True,exist_ok=True);plt.tight_layout();plt.savefig(p/name,dpi=150);plt.close()
def main():
    p=C.read(C.DOC/'ZERO_INIT_PARITY.json');fit=C.read(C.DOC/'FIT.json');r=C.read(C.DOC/'RESULTS.json')['groups'];a=C.read(C.DOC/'VERIFIED_VISIBLE.json')['groups'];s=C.read(C.DOC/'SOURCE_PRESERVATION.json')['groups'];cost=C.read(C.DOC/'COMPUTE_COST.json');d=C.read(C.DOC/'DECISION.json')
    fig,ax=plt.subplots(figsize=(8,3));ax.plot([v['xy_max_px'] for v in p['rows']],marker='o');ax.set(title='Zero-init S1 parity / 32 fixed inputs / max difference 0 px',xlabel='Input',ylabel='Max xy difference (px)');finish('01_zero_init_parity.png')
    tr=C.read(C.RAW/'TRACE.json');fig,axs=plt.subplots(1,2,figsize=(11,4));axs[0].plot([v['L_clean'] for v in tr],label='Clean SmoothL1');axs[0].plot([v['L_occ_preserve'] for v in tr],label='Occluded preserve SmoothL1');axs[0].legend();axs[0].set(xlabel='Step / fixed320',ylabel='Loss (640px coordinate units)');axs[1].plot([v['residual_xy_mean_px'] for v in tr]);axs[1].set(xlabel='Step',ylabel='Mean xy residual norm (px)');finish('02_training_losses.png')
    for num,g in enumerate(('CLEAN','MODERATE','SEVERE'),3):
        fig,axs=plt.subplots(1,2,figsize=(11,4));arms=['BASE','PRES1'];x=np.arange(2)
        for k,field in enumerate(('current','oracle')):axs[0].bar(x+(k-.5)*.3,[r[g][b][field]['ADDsym_AUC'] for b in arms],.3,label=field)
        axs[0].set(xticks=x,xticklabels=['S1+GEO','PRES1+GEO'],ylabel='ADDsym AUC',title=g);axs[0].legend();axs[1].bar(['S1','PRES1'],[r[g][b]['twoD']['PCK']['10'] for b in arms]);axs[1].set(ylabel='PCK10',title='Same raw xy metric contract');finish(f'{num:02d}_{g.lower()}_'+('recovery.png' if g=='CLEAN' else 'preservation.png'))
    fig,ax=plt.subplots(figsize=(9,4));gg=['CLEAN','MODERATE','SEVERE','HARD','ALL'];x=np.arange(len(gg))
    for k,b in enumerate(('BASE','PRES1')):ax.bar(x+(k-.5)*.35,[a[g][b]['PCK']['10']['fraction'] for g in gg],.35,label=b)
    ax.set(xticks=x,xticklabels=gg,ylabel='Fixed-ID visible anchor PCK10');ax.legend();finish('06_verified_visible.png')
    fig,axs=plt.subplots(1,2,figsize=(10,4));arms=['BASE','PRES1'];axs[0].bar(arms,[s[b]['twoD']['PCK']['10']['fraction'] for b in arms]);axs[0].set(ylabel='Source256 PCK10');axs[1].bar(arms,[s[b]['pose']['ADDsym_AUC'] for b in arms]);axs[1].set(ylabel='Exact source256 ADDsym AUC');finish('07_source_preservation.png')
    fig,axs=plt.subplots(1,2,figsize=(10,4));axs[0].bar(arms,[cost['timings_ms'][b]['median'] for b in arms]);axs[0].set(ylabel='Median RGB inference ms / RTX3080');axs[1].bar(['Frozen base','New adapter'],[cost['model_params'],cost['additional_params']]);axs[1].set(yscale='log',ylabel='Parameters (log scale)');finish('08_latency_params.png')
    fig,ax=plt.subplots(figsize=(10,4));ax.bar(['CLEAN','MODERATE','SEVERE'],[d['delta_AUC'][g] for g in ('CLEAN','MODERATE','SEVERE')]);ax.axhline(0,color='black');ax.set(ylabel='PRES1 AUC - frozen S1+GEO AUC',title=d['primary']);finish('09_decision.png')
    cases()

def cases():
    from scripts.research.pallet_existing_data_transfer_v1.report import crop,tile
    from scripts.research.pallet_clean19_pose_mismatch_v1.render import draw_edges,projection
    from scripts.research.pallet_recording_disjoint_transfer_v1 import common as V
    rr={r['id']:r for r in V.records()};truth=C.read(V.E.P.RAW/'TRUTH_FOR_DISPLAY_ONLY.json');md={r['id']:r for r in C.read(V.E.V.RAW/'INFERENCE_METADATA.json')};pred=C.read(C.RAW/'RAW_PREDICTIONS.json')['real'];pm=C.read(C.RAW/'POSE_METRICS.json');poses=C.read(C.RAW/'POSE_DECISIONS.json');selected=[]
    for sev in ('CLEAN','MODERATE_OCCLUSION','SEVERE_OCCLUSION'):
        ids=[i for i,r in rr.items() if r['severity']==sev];order=sorted(ids,key=lambda i:pm['PRES1'][i]['current']['ADDsym_normalized']-pm['BASE'][i]['current']['ADDsym_normalized']);selected.extend([order[0],order[-1]])
    panels=[];manifest=[]
    for fid in dict.fromkeys(selected):
        base,off,scale=crop(rr[fid],truth[fid]['box']);pair=[]
        for a in ('BASE','PRES1'):
            im=base.copy();q=np.array(C.selected(pred[a][fid])['keypoints_xy']);pose=poses[a][fid]['current']
            draw_edges(im,(np.array(truth[fid]['gt'])-off)*scale,(50,220,50));draw_edges(im,(q-off)*scale,(0,220,255));draw_edges(im,(projection(pose,np.array(md[fid]['K']))-off)*scale,(20,30,255))
            pair.append(tile(im,[a+' '+fid,rr[fid]['severity'],f"ADDnorm {pm[a][fid]['current']['ADDsym_normalized']:.4f}",'Green GT / yellow raw xy / red selected PnP'],height=300))
        panels.append(np.hstack(pair));manifest.append(dict(id=fid,severity=rr[fid]['severity'],delta_ADDnorm=pm['PRES1'][fid]['current']['ADDsym_normalized']-pm['BASE'][fid]['current']['ADDsym_normalized']))
    cv2.imwrite(str(C.DOC/'figures/10_improved_and_failed_cases.jpg'),np.vstack(panels));C.save(C.DOC/'FIGURE_CASES.json',dict(selection='Posthoc min/max delta ADDnorm within each severity, illustrative extremes only; no fitting/selection use',cases=manifest))
if __name__=='__main__':main()
