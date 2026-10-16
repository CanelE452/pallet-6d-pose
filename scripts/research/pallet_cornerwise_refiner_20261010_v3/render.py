"""Render fixed illustrative images and saved-row plots; no inference or fitting."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from . import common as C

def run(args):
    C.verify_protocol(args);C.protect(args)
    output=Path(args.output);folder=output/'figures'
    C.require(not folder.exists(),'preserve completed figure directory')
    folder.mkdir()
    scored=list(C.rows(output/'PREDICTIONS.jsonl.gz'))
    fixed=list(C.rows(output/'FIXED_PREDICTIONS.jsonl.gz'))
    data={(r['method'],r['id']):r for r in scored+fixed}
    methods=['N3_SUBPIX',*C.METHODS]
    fig,axes=plt.subplots(1,2,figsize=(13,6))
    for axis,field,unit in zip(axes,['translation_cm','rotation_deg'],['cm','degrees']):
        for i,method in enumerate(methods):
            values=np.array([r['pose'][field] for r in data.values() if r['method']==method and r['pose']['available']])
            axis.scatter(values,np.full(len(values),i),s=8,alpha=.22)
            if values.size:
                q=np.quantile(values,[.1,.5,.9]);axis.plot([q[0],q[2]],[i,i],linewidth=3)
                axis.scatter([values.mean()],[i],marker='D',facecolor='white',edgecolor='black',zorder=5)
        axis.set_yticks(range(len(methods)),methods if axis is axes[0] else [])
        axis.invert_yaxis();axis.set_xscale('symlog',linthresh=1);axis.set_xlabel(field+' ('+unit+')');axis.grid(alpha=.2)
    fig.suptitle('All operational outputs retained: Clean153 + Moderate92\nDiamond=mean; segment=P10-P90. Existing geometric proxy DEV references.')
    fig.tight_layout();fig.savefig(folder/'01_all_operational.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for axis,field in zip(axes,['translation_cm','rotation_deg']):
        pair=np.array([(data['N3_SUBPIX',r['id']]['pose'][field],r['pose'][field]) for r in scored
            if r['method']==C.PRIMARY and r['pose']['available'] and data['N3_SUBPIX',r['id']]['pose']['available']])
        axis.scatter(pair[:,0],pair[:,1],s=13,alpha=.5)
        upper=max(1,pair.max());axis.plot([0,upper],[0,upper],'k--')
        axis.set_xscale('symlog',linthresh=1);axis.set_yscale('symlog',linthresh=1)
        axis.set_xlabel('Fixed N3 '+field);axis.set_ylabel('Cornerwise '+field);axis.grid(alpha=.2)
    fig.suptitle('Same245 IDs; below diagonal favors cornerwise selection')
    fig.tight_layout();fig.savefig(folder/'02_same_frame_pairs.png',dpi=140);plt.close(fig)
    protocol=C.read(C.V.DOC/'VISUAL_CASE_PROTOCOL.json');case_rows=[]
    for index,case in enumerate(protocol['cases']):
        path=Path(args.source_root)/case['image']['path'];C.bound(path,case['image'],'fixed illustrative RGB')
        rgb=np.asarray(Image.open(path).convert('RGB'))
        fig,axes=plt.subplots(1,3,figsize=(16,5))
        for axis,method in zip(axes,['N3_SUBPIX','N3_VALIDATED_ROLE',C.PRIMARY]):
            row=data[method,case['id']];axis.imshow(rgb)
            q=np.asarray(row['native_points'],float);valid=np.isfinite(q[:8]).all(1)
            axis.scatter(q[:8][valid,0],q[:8][valid,1],s=32,facecolor='none',edgecolor='#00dce6',linewidth=1.4)
            for k in np.flatnonzero(valid):axis.text(q[k,0]+3,q[k,1]+3,str(k),color='cyan',fontsize=8)
            for k in row.get('reprojected_ids',[]):axis.scatter(*q[k],s=68,marker='s',facecolor='none',edgecolor='#ff8c24')
            contract=row.get('observation_contract',{})
            for c in contract.get('per_validated_corner',[]):
                p=np.asarray(c['validated_xy']);accepted=c['id'] in contract.get('hybrid_boundary_corner_ids',[])
                axis.scatter(*p,s=35,marker='+' if accepted else 'x',color='#42ff50' if accepted else '#ff62e5')
            ref=row.get('evaluation_reference',{});gt=np.asarray(ref.get('native_points_px',[]),float)
            ids=ref.get('valid_native_ids',[])
            if len(gt):axis.scatter(gt[ids,0],gt[ids,1],s=19,marker='.',color='white',alpha=.65)
            score=row['pose'];T=score.get('translation_cm');R=score.get('rotation_deg')
            axis.set_title(method+'\n'+row['output_status']+' / T='+str(round(T,4) if T is not None else None)+'cm / R='+str(round(R,4) if R is not None else None)+'deg',fontsize=8)
            axis.axis('off')
            case_rows.append(dict(id=case['id'],method=method,pose=row['pose'],output_status=row['output_status'],
                selected_boundary_ids=contract.get('hybrid_boundary_corner_ids',[]),reprojected_ids=row.get('reprojected_ids',[])))
        fig.suptitle(case['label']+' / '+case['id']+'\ncyan=output; green+=accepted boundary; magenta x=rejected; orange square=H reprojection; white=proxy reference',fontsize=9)
        fig.tight_layout();fig.savefig(folder/f'case_{index+1:02}.png',dpi=140,bbox_inches='tight');plt.close(fig)
    C.save_rows(output/'FIGURE_CASE_ROWS.jsonl.gz',case_rows)
    C.write_new(output/'FIGURE_BINDINGS.json',dict(figures=[C.binding(p) for p in sorted(folder.glob('*.png'))],
        fixed_case_protocol=C.binding(C.V.DOC/'VISUAL_CASE_PROTOCOL.json'),case_rows=C.binding(output/'FIGURE_CASE_ROWS.jsonl.gz'),
        new_detector_calls=0,new_pose_fits=0,new_RGB=0,selection_uses_new_scores=False))
    print('CORNERWISE_FIGURES',len(list(folder.glob('*.png'))),flush=True)

if __name__=='__main__':run(C.parser(__doc__).parse_args())
