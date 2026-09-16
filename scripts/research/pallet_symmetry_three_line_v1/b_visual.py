"""Fixed-rule post-hoc inspection, including separate GT explanation panels."""
import csv,math
import numpy as np
import torch,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
import env as E
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.hough import Lattice
from scripts.research.pallet_symmetry_dht_local_v2.symdht_local_v2.geometry import decode_single_mode
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.constants import EDGES

def main():
    torch.set_num_threads(4);metrics=E.read(E.RAW/'B/SYNTH_FULL_METRICS.json')
    base={r['frame_id']:r for r in metrics['B0_P']['1']};new={r['frame_id']:r for r in metrics['B3_THREE_AMBIG']['1']}
    delta={fid:np.mean([next(r['frame_mean_px'] for r in metrics['B3_THREE_AMBIG'][str(s)] if r['frame_id']==fid)-next(r['frame_mean_px'] for r in metrics['B0_P'][str(s)] if r['frame_id']==fid) for s in [1,2,3]]) for fid in base}
    line=list(csv.DictReader((E.DOC/'B/line_error_and_ambiguity.csv').open()));ambiguity={};height={}
    for row in line:
        fid=row['frame_id'];ambiguity.setdefault(fid,[]).append(float(row['ambiguity']))
        if row['role_3D']=='height_axis_Y' and row['MAP_endpoint_distance_px']:height[fid]=max(height.get(fid,0),float(row['MAP_endpoint_distance_px']))
    groups=dict(improvement=sorted(delta,key=lambda k:(delta[k],k))[:5],harm=sorted(delta,key=lambda k:(-delta[k],k))[:5],
      high_ambiguity=sorted(base,key=lambda k:(-np.mean(ambiguity[k]),k))[:5],height_edge_failure=sorted(height,key=lambda k:(-height[k],k))[:5])
    E.freeze(E.DOC/'B/VISUAL_SELECTION.json',dict(groups=groups,seed_view=1,ranking='three-seed mean frame error delta; mean edge ambiguity; maximum height edge normal-incidence error',GT_for_posthoc_selection_only=True))
    records=E.read(E.EXPORT/'synth_val.json')['records'];lookup={r['frame_id']:i for i,r in enumerate(records)};lattice=Lattice()
    p0=torch.load(E.RAW/'B/synth_val_predictions/B0_P_seed1.pt',map_location='cpu',weights_only=False)['points']
    p3=torch.load(E.RAW/'B/synth_val_predictions/B3_THREE_AMBIG_seed1.pt',map_location='cpu',weights_only=False)['points']
    eta=E.read(E.DOC/'B/selection_lock.json')['selected_eta']['B3_THREE_AMBIG'];T=E.read(E.C.B/'P_SELECTION.json')['temperatures']['1']['temperature']
    thumbs=[];files=[]
    for group,ids in groups.items():
        for rank,fid in enumerate(ids):
            i=lookup[fid];rec=records[i];c=torch.load(E.RAW/f'B/cache/synth_val/{i:04d}.pt',map_location='cpu',weights_only=False)
            im=cv2.imread(rec['source_image'])[100:-100,100:-100,::-1].copy();h,w=im.shape[:2]
            mapping=c['P'][1]['alignment']['mapping'];lines=decode_single_mode(c['line_logits'][mapping][None],c['line_valid'][mapping][None],lattice,c['box_raw'][None])['raw_line'][0].numpy()
            fig=plt.figure(figsize=(14,10));grid=fig.add_gridspec(4,4,height_ratios=[2,2,1,1]);left=fig.add_subplot(grid[:2,:2]);right=fig.add_subplot(grid[:2,2:])
            for ax in [left,right]:ax.imshow(im);ax.set_xlim(0,w);ax.set_ylim(h,0);ax.axis('off')
            for e,((a,b),linecoef) in enumerate(zip(EDGES,lines)):
                nx,ny,cc=linecoef
                if abs(ny)>1e-6:xx=np.array([0,w]);yy=-(nx*xx+cc)/ny
                else:yy=np.array([0,h]);xx=-(ny*yy+cc)/nx
                left.plot(xx,yy,'--',lw=.6,alpha=.4,color=plt.cm.tab20(e/12))
                mid=(c['base_raw'][a].numpy()+c['base_raw'][b].numpy())/2
                left.text(*mid,f'e{e} A={float(c["P"][1]["ambiguity"][e]):.2f}',fontsize=5,color='white',bbox=dict(facecolor='black',alpha=.45,pad=.1))
            for pts,marker,col,label in [(c['base_raw'].numpy(),'x','cyan','R0'),(p0[i],'o','lime','P'),(p3[i],'+','magenta','B3')]:
                left.scatter(pts[:8,0],pts[:8,1],s=22,marker=marker,color=col,label=label)
            for k,xy in enumerate(p3[i,:8]):left.text(*xy,str(k),fontsize=7,color='yellow')
            left.legend(fontsize=7);left.set_title('Inference evidence: R0 / P / B3; MAP lines for display only')
            gt=np.array(base[fid]['target'],float);valid=np.array(base[fid]['gt_mask'],bool)
            right.scatter(gt[valid,0],gt[valid,1],c='yellow',marker='x',label='GT (post-hoc only)')
            right.scatter(p3[i,:8,0],p3[i,:8,1],c='magenta',marker='+',label='B3');right.legend(fontsize=7)
            right.set_title(f'Post-hoc explanation: mean-seed delta={delta[fid]:+.5f}px')
            logits=c['P'][1]['output']['logits'][0]/T;bias=c['P'][1]['biases']['B3_THREE_AMBIG']
            before=logits.softmax(-1).numpy();after=(logits+eta*bias).softmax(-1).numpy()
            for k in range(8):
                ax=fig.add_subplot(grid[2+k//4,k%4]);ax.plot(before[k],color='green',lw=.65);ax.plot(after[k],color='magenta',lw=.65);ax.axvline(221,color='gray',lw=.5)
                incident=[e for e,(a,b) in enumerate(EDGES) if k in [a,b]]
                ax.set_title(f'k{k}: edges {incident}; null=221',fontsize=7);ax.tick_params(labelsize=6)
            fig.suptitle(f'{group} #{rank+1} — {fid}\nStructural cuboid lines need not be visible image boundaries. Entropy is not visibility.',fontsize=10)
            fig.tight_layout();path=E.RAW/f'B/figures/{group}_{rank+1}.png';path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=130);plt.close(fig)
            files.append(E.bound(path));thumb=Image.open(path).convert('RGB');thumb.thumbnail((500,360));thumbs.append(thumb.copy())
    contact=Image.new('RGB',(2000,1800),'white')
    for i,thumb in enumerate(thumbs):contact.paste(thumb,((i%4)*500,(i//4)*360))
    contact.save(E.DOC/'B/VISUAL_CONTACT.png')
    E.write(E.DOC/'B/VISUAL_MANIFEST.json',dict(files=files,panels=20,GT_inference_input=False,contact=E.bound(E.DOC/'B/VISUAL_CONTACT.png')))
    print('VISUAL PANELS COMPLETE',len(files),flush=True)
if __name__=='__main__':main()
