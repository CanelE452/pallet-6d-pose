"""One authenticated private postseal model-orientation display, without scoring."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from . import stage_gallery as G
from .s1_analysis import wire,project

def render(run,d1,reference_dir,output,GO=False):
    if not GO:raise PermissionError('Explicit root display GO required')
    r,d,o=map(Path,(run,d1,output));fid='eval_pallet07:1778652140531310080';seed=20261018
    plan,source,baseline,scores,coverage,bindings,es=G.validate_inputs(r/'S1_STAGE_GALLERY_MANIFEST.json',d,r/'S1_EVALUATION',r/'POST_BANK_S1_VALIDATION_V2',r/'S1_INDEPENDENT_FINAL')
    rights=G.U.read(d/'RIGHTS_SCOPE_BINDING.json');privacy=G.U.read(d/'DISPLAY_PRIVACY_PLAN.json')
    if fid not in rights['approved_ids']or G.U.sha(d/'DISPLAY_PRIVACY_PLAN.json')!=rights['display_privacy_plan_sha256']or G.U.sha(d/'RIGHTS_RECEIPT.json')!=rights['rights_receipt_sha256']:raise ValueError('PRIMARY privacy scope differs')
    ref=Path(reference_dir)/'FIXED_REFERENCES.json';seal=G.U.read(Path(reference_dir)/'SCORE_SEAL.json')
    if G.U.sha(ref)!=es['reference_sha256']or G.U.sha(ref)!=seal['artifacts'][ref.name]:raise ValueError('Postseal display reference differs')
    entry=source['S1_UNION'];rows=list(G.U.rows(G.U.paths(d,'real')[0]));ordinal=next(i for i,f in enumerate(rows)if f['id']==fid);frame=rows[ordinal]
    if frame['partition']not in G.PRIMARY:raise ValueError('CAL RGB prohibited')
    result,rb=G.core_result('S1_UNION',entry,fid,ordinal,d)
    p0=baseline[fid]['predictions'][f'D1_H1_MLP_s{seed}'];p1=entry['predictions'][fid]['predictions'][f'S1_UNION_s{seed}']
    cv=next(c for c in coverage if c['arm']=='S1_UNION'and c['id']==fid)
    rank,best=G.saved_action_rank(cv,p1,entry['features'][fid])
    if best is None or not best['available']:raise ValueError('Actual saved eligible best-good action required')
    refs=G.U.read(ref)['references'];rr=refs[frame['partition']][fid]
    reference=dict(available=True,R_physical=rr['R'],centroid=rr['t'])
    roi=G.display_roi(frame,result);x0,y0,x1,y1=roi['xyxy_raw']
    from scripts.research.pallet_boundary_action_repair_v1.display_privacy import display_image
    image=display_image(frame,privacy)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,6.5));fam=result['families']['ORIGINAL_WD'];xyz=np.asarray(fam['native_xyz']);center=xyz.mean(0)
    variants=[('Original H1 selected',p0['actual_pose'],'#32a8ff'),('Private reference projection',reference,'#1ba448'),('Saved eligible best-good action',best,'#e29800')]
    for ax,(title,pose,color)in zip(axes,variants):
        ax.imshow(image);ax.set_xlim(x0,x1);ax.set_ylim(y1,y0);ax.axis('off');wire(ax,pose,frame,result,color)
        pixels,z=project(pose,frame,result)
        for i,(uv,depth)in enumerate(zip(pixels,z)):
            if np.isfinite(uv).all()and depth>0:ax.annotate(str(i),uv,xytext=(2,2),textcoords='offset points',fontsize=11,color=color,bbox=dict(boxstyle='round,pad=.13',fc='white',alpha=.7,ec='none'),clip_on=True)
        R=np.asarray(pose['R_physical'])@np.asarray(fam['physical_registration_Q']).T
        axispoints=np.vstack([center,center+np.diag(.4*np.ptp(xyz,axis=0))])
        camera=(axispoints-center)@R.T+np.asarray(pose['centroid']);p=camera@np.asarray(frame['K']).T;uv=p[:,:2]/p[:,2:]
        if (camera[:,2]>0).all()and np.isfinite(uv).all():
            for k,(label,c)in enumerate([('native X','#e52d27'),('native Y','#20aa3d'),('native Z','#246ee9')],1):
                ax.annotate('',uv[k],xytext=uv[0],arrowprops=dict(arrowstyle='->',color=c,lw=2.3));ax.annotate(label,uv[k],fontsize=9,color=c,bbox=dict(fc='white',alpha=.7,ec='none'),clip_on=True)
        if title.startswith('Original'):metric=scores[fid]['metrics'][f'D1_H1_MLP_s{seed}'];detail=f'{metric["translation_cm"]:.2f}cm / {metric["rotation_deg"]:.2f}deg'
        elif title.startswith('Saved'):detail='posthoc rank '+str(rank['best_good_action_rank'])+'; no new forward'
        else:detail='display projection only; no numeric GT export'
        ax.set_title(title+'\n'+detail,fontsize=12)
    fig.suptitle(fid+' | seed20261018 | Why a nearly square cuboid can look similar at a large orientation error',fontsize=13)
    fig.text(.5,.035,'Corner numbers and native X/Y/Z arrows are projected MODEL coordinates, including S corners; they are not image observations or certified physical owners.\nOnly C2 (180° about object Y) is allowed by the saved metric; a 90° turn is not equivalent. Same saved native XYZ/Q projection as gallery wire.\nPrediction-only ROI; derived PRIMARY privacy-redacted image. Source pose/GT matrices and numeric GT projections are not exported.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.02,right=.98,top=.86,bottom=.17,wspace=.08);o.mkdir(parents=True,exist_ok=False)
    p=o/'ORIENTATION_NATIVE_AXES.jpg';fig.savefig(p,dpi=120,pil_kwargs=dict(quality=88));plt.close(fig)
    receipt=dict(complete=True,id=fid,seed=seed,input_bindings=bindings,core_binding=rb,display_source_closure=G.display_sources(),supplement_source_sha256=G.U.sha(__file__),
        original_selected_candidate_id=p0['selected_candidate_id'],best_good_saved_action_id=rank['best_good_action_id'],best_good_saved_rank=rank['best_good_action_rank'],
        rights_scope_sha256=G.U.sha(d/'RIGHTS_SCOPE_BINDING.json'),privacy_plan_sha256=G.U.sha(d/'DISPLAY_PRIVACY_PLAN.json'),
        reference_display_only=True,GT_numeric_JSON_exported=False,projected_S_are_model_coordinates_not_observations=True,physical_owner_certified=False,raw_RGB_copied=False,CAL_RGB_opened=0,new_pose_calls=0,new_metric_evaluations=0,new_fits=0,artifact_sha256={p.name:G.U.sha(p)})
    G.U.write(o/'DISPLAY_SEAL.json',receipt);return dict(complete=True,panel=str(p))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('run','d1','reference-dir','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--root-go',action='store_true');a=p.parse_args();print(json.dumps(render(a.run,a.d1,a.reference_dir,a.output,a.root_go)))
