#!/usr/bin/env python3
"""Fixed label-transition examples; source RGB only, no model-based selection."""
from pathlib import Path
import csv,json
import numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import target_identity_swap as audit

OUT=Path(__file__).resolve().parent
labels=audit.loadnp(OUT/'PAIR_LABELS.npz');probs=audit.loadnp(OUT/'EVAL_MIXED_MULTITARGET_UNWEIGHTED/OUTPUTS.npz')
rows=audit.readrows(audit.ROOT/'source_audit/YCB_VALIDATION_QUERY_EVIDENCE.csv')
inputs={r['id']:r for r in [json.loads(s)for s in(audit.V63/'sam_source/INPUTS.jsonl').read_text().splitlines()]if r['source']=='YCB'}
metadata={f"YCB:{r['scene_id']:06d}:{r['frame_id']:06d}:{r['instance_id']:06d}":r for r in [json.loads(s)for s in(audit.V63/'ycb/val_instances.jsonl').read_text().splitlines()]}
old=labels['classes_original'][:,0,4,4];new=labels['classes_other'][:,0,4,4]
transitions=[(0,2),(1,0),(1,2),(2,0)]
ix=[int(np.flatnonzero((old==a)&(new==b))[0])for a,b in transitions]
fig,axes=plt.subplots(4,5,figsize=(16,11));records=[];names=['VISIBLE','EXTERNAL_PROXY','OUTSIDE']
for ri,i in enumerate(ix):
    r=rows[i];ident=r['id'];other=str(labels['conditioned_id_other'][i]);src=inputs[ident]
    rgb=Image.open(audit.check_source(src['image'])).convert('RGB');scene=rgb.copy();draw=ImageDraw.Draw(scene)
    for tid,color in [(ident,'deepskyblue'),(other,'orange')]:draw.rectangle(inputs[tid]['box'],outline=color,width=3)
    qx,qy=float(r['query_x']),float(r['query_y']);px,py=int(r['pixel_x']),int(r['pixel_y']);L=float(r['roi_side']);side=int(np.ceil(L));left=int(np.floor(qx-side/2));top=int(np.floor(qy-side/2));cropbox=(left,top,left+side,top+side)
    draw.ellipse((qx-4,qy-4,qx+4,qy+4),outline='yellow',width=2)
    axes[ri,0].imshow(scene);axes[ri,0].set_title(f"scene{r['scene_id']} frame{r['frame_id']}  q{r['query_index']}\nblue target{r['instance_id']} / orange target{metadata[other]['instance_id']}",fontsize=9)
    def crop_with_marker(image):
        result=image.crop(cropbox);d=ImageDraw.Draw(result);x,y=px-left,py-top
        d.rectangle((x-2,y-2,x+2,y+2),outline='yellow',width=1)
        d.ellipse((x-.7,y-.7,x+.7,y+.7),fill='white');return result
    axes[ri,1].imshow(crop_with_marker(rgb),interpolation='nearest');axes[ri,1].set_title(f"Fixed RGB / q / ROI L={L:.1f}px\n5x5 yellow box; white nearest pixel",fontsize=9)
    for col,tid in [(2,ident),(3,other)]:
        m=metadata[tid];v=audit.mask(m['mask_vis']);f=audit.mask(m['mask_full']);overlay=np.asarray(rgb).astype(float)
        overlay[v]=.45*overlay[v]+.55*np.array([0,255,80]);overlay[f&~v]=.45*overlay[f&~v]+.55*np.array([255,0,180])
        axes[ri,col].imshow(crop_with_marker(Image.fromarray(np.uint8(overlay))),interpolation='nearest');axes[ri,col].set_title(('Original'if tid==ident else'Other')+f" target support\nGT {names[int(old[i]if tid==ident else new[i])]}",fontsize=9)
    text=[];case=dict(sample_id=r['sample_id'],other_id=other,selection='first cache row matching prespecified GT transition; no prediction selection',original_state=names[int(old[i])],other_state=names[int(new[i])],query=[qx,qy],roi_side=L,arms={})
    for arm,label in [('ORIGINAL','Original'),('CONTEXT_ONLY_SWAP','Context only'),('FULL_CONDITION_SWAP','Full condition')]:
        sp,v,f,h=audit.marginals(probs[arm][i:i+1]);u=bool(probs[arm][i].max()<.5)
        text.append(f"{label}\npV={v[0,0]:.3f}, pFull={f[0,0]:.3f}\nlogical={names[int(h[0,0])]} {' U'if u else''}")
        case['arms'][arm]=dict(p_visible=float(v[0,0]),p_full=float(f[0,0]),logical_state=names[int(h[0,0])],confidence_U=u)
    axes[ri,4].text(.02,.95,'\n\n'.join(text),ha='left',va='top',fontsize=10,transform=axes[ri,4].transAxes);records.append(case)
    for ax in axes[ri]:ax.axis('off')
fig.suptitle('Public YCB fixed target-swap probes: same RGB and native query, different conditioned object\nGreen=official target-visible; magenta=full minus visible proxy. Context-only deliberately leaves original SAM/box fixed.',fontsize=12)
fig.tight_layout(rect=[0,0,1,.94]);image=OUT/'FIXED_TARGET_SWAP_CASES.png';assert not image.exists();fig.savefig(image,dpi=150);plt.close(fig)
(OUT/'FIXED_TARGET_SWAP_CASES.json').write_text(json.dumps(dict(source_only=True,private_data_read=False,prediction_selection=False,cases=records),indent=2)+'\n')
print(image)
