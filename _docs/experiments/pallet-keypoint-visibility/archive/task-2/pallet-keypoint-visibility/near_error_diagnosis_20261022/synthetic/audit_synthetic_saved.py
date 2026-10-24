from pathlib import Path
import csv, gzip, json, hashlib
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT=Path(__file__).resolve().parent
TASK=OUT.parents[1]
V61=TASK.parent/'target_presence_classifier_20261022_v61'
V60=TASK.parent/'external_ray_proxy_head_20261022_v60'
GT=Path('/home/minjae/Documents/pallet-research/pallet-statewise-end-to-end-recovery-v1/synthetic_v1/SYNTH_GT_LABELS.jsonl.gz')
def read(p):
    with (gzip.open if str(p).endswith('.gz') else open)(p,'rt') as f: return json.load(f)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for z in iter(lambda:f.read(4*1024*1024),b''):h.update(z)
    return h.hexdigest()
def save(name,x):
    (OUT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
def csvsave(name,rr):
    if not rr:return
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rr for k in r)));w.writeheader();w.writerows(rr)
u=read(V60/'PREPARED_INPUTS_SYNTH.json.gz');by={x['id']:x for x in u}
pred=list(csv.DictReader((V61/'SYNTH_QUERY_PREDICTIONS.csv').open()))
mask_audit={x['id']:x for x in csv.DictReader((V61/'MASK_CONTRACT_AUDIT.csv').open())}
ray={(x['id'],int(x['native_corner'])):x for x in csv.DictReader((V60/'SYNTH_QUERY_OUTPUTS.csv').open())}
with gzip.open(GT,'rt') as f:gt={x['id']:x for x in map(json.loads,f)}
rows=[];case_masks={};mapping_max=0
for x in u:
    ident=x['id'];image=Path(x['image']);ann=image.parent.parent/'labels'/(image.name.replace('_rgb.png','')+'_label.json')
    obj=read(ann)['objects'][0];ref=np.array(obj['projected_cuboid'],float)[:8];stored=np.array(gt[ident]['projected_GT_native_xy'],float)
    gap=float(np.max(np.abs(ref-stored)));assert gap<1e-10;mapping_max=max(mapping_max,gap)
    a=mask_audit[ident];m4=np.array(Image.open(a['visible_path']))>127;m0=np.array(Image.open(a['amodal_path']))>127;h,w=m4.shape
    q=np.array(x['initial_points'],float)[:8];good=np.isfinite(q).all(1)&~(q==-1).all(1);finite=q[good]
    L=float(np.clip(.25*np.max(np.ptp(finite,axis=0)),16,max(x['raw_hw']))) if len(finite)>=2 else 16.
    assert obj['keypoint_convention']=='camera_dynamic_0123_v4'
    for r0 in [r for r in pred if r['id']==ident]:
        r=dict(r0);k=int(r['native_corner']);px,py=int(r['pixel_x']),int(r['pixel_y']);qx,qy=q[k];gx,gy=ref[k];gpx,gpy=np.floor(ref[k]+.5).astype(int)
        gvalid=bool(0<=gx<w and 0<=gy<h and 0<=gpx<w and 0<=gpy<h)
        valid=r['valid']=='True';target=int(r['presence_target']);p=float(r['presence_probability']);guess=p>=.5
        if valid:assert int(m4[py,px])==target
        distance=float(np.linalg.norm(q[k]-ref[k]));gmask=bool(m4[gpy,gpx]) if gvalid else None
        block=ray[(ident,k)];r.update(native_corner=k,pixel_x=px,pixel_y=py,valid=valid,presence_target=target,presence_probability=p,predicted_present=guess,
            gt_x=float(gx),gt_y=float(gy),gt_nearest_pixel_x=int(gpx),gt_nearest_pixel_y=int(gpy),gt_coordinate_valid=gvalid,gt_pixel_M4=gmask,
            query_to_native_renderer_gt_px=distance,near_le3=valid and gvalid and distance<=3,near_le05=valid and gvalid and distance<=.5,
            M4_query_vs_GT_different=valid and gvalid and bool(target)!=gmask,query_crop_side_native_px=L,
            stored_external_ray_proxy_blocked=int(block['target']),stored_external_ray_loss_valid=block['loss_valid']=='True',
            synthetic_semantic_DIRECT_truth='UNAVAILABLE',material_variant=obj.get('v2_labels',{}).get('material_variant_actual'),source_asset=obj.get('source_asset'),
            width_m=obj['dimensions_m']['width'],height_m=obj['dimensions_m']['height'],depth_m=obj['dimensions_m']['depth'],
            final_luma_pallet=obj.get('v2_labels',{}).get('luma_pallet_final'))
        if valid:
            r['M4_stencil3_sum']=int(m4[max(0,py-1):min(h,py+2),max(0,px-1):min(w,px+2)].sum())
            r['M4_stencil7_sum']=int(m4[max(0,py-3):min(h,py+4),max(0,px-3):min(w,px+4)].sum())
            r['M0_stencil3_sum']=int(m0[max(0,py-1):min(h,py+2),max(0,px-1):min(w,px+2)].sum())
            lo=q[k]-L/2;hi=q[k]+L/2
            x0=max(0,int(np.ceil(lo[0])));x1=min(w,int(np.ceil(hi[0])));y0=max(0,int(np.ceil(lo[1])));y1=min(h,int(np.ceil(hi[1])))
            region=m4[y0:y1,x0:x1]
            r['M4_crop_fraction_native_pixel_centers']=float(region.mean()) if region.size else None
            offs=((np.arange(9)+.5)/9-.5)*L;xx,yy=np.meshgrid(offs,offs);grid=q[k]+np.stack([xx,yy],-1);ixy=np.floor(grid+.5).astype(int)
            vg=(grid[...,0]>=0)&(grid[...,0]<w)&(grid[...,1]>=0)&(grid[...,1]<h)&(ixy[...,0]>=0)&(ixy[...,0]<w)&(ixy[...,1]>=0)&(ixy[...,1]<h)
            r['M4_at_9x9_nominal_grid_fraction']=float(m4[ixy[...,1][vg],ixy[...,0][vg]].mean()) if vg.any() else None
            ylo=max(0,py-3);yhi=min(h,py+4);xlo=max(0,px-3);xhi=min(w,px+4);local=m4[ylo:yhi,xlo:xhi];oy,ox=np.mgrid[ylo:yhi,xlo:xhi]
            if local.any():
                dx=float((ox[local]-qx).mean());dy=float((oy[local]-qy).mean());norm=float(np.hypot(dx,dy))
                r['M4_local7_support_centroid_dx']=dx;r['M4_local7_support_centroid_dy']=dy
                r['M4_local7_support_direction_deg']=float(np.degrees(np.arctan2(dy,dx))) if norm>1e-9 else None
                r['query_minus_GT_dot_support_direction']=float(np.dot(q[k]-ref[k],[dx,dy])/norm) if norm>1e-9 else None
        rows.append(r)
csvsave('SYNTH_ENRICHED_QUERIES.csv',rows)
def metric(rr,split,stratum):
    y=np.array([r['presence_target'] for r in rr],bool);p=np.array([r['presence_probability'] for r in rr]);z=p>=.5
    tp=int((y&z).sum());fp=int((~y&z).sum());fn=int((y&~z).sum());tn=int((~y&~z).sum())
    return dict(split=split,stratum=stratum,n=len(rr),positive=int(y.sum()),negative=int((~y).sum()),TP=tp,FP=fp,FN=fn,TN=tn,
        precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
        false_positive_rate=fp/(fp+tn) if fp+tn else None,median_probability=float(np.median(p)) if len(p) else None)
stats=[];distinction=[]
for split in ['train','validation']:
    a=[r for r in rows if r['split']==split and r['valid']]
    filters={'ALL':lambda r:True,'ERROR_LE_3':lambda r:r['near_le3'],'ERROR_LE_05':lambda r:r['near_le05'],
        'NOT_ELIGIBLE_NEAR':lambda r:not r['near_le3'],'ERROR_LE_3_Q_AND_GT_M4_1':lambda r:r['near_le3'] and r['presence_target']==1 and r['gt_pixel_M4'],
        'ERROR_LE_3_GT_M4_1_Q_M4_0':lambda r:r['near_le3'] and r['presence_target']==0 and r['gt_pixel_M4'],
        'ERROR_LE_3_GT_M4_0_Q_M4_1':lambda r:r['near_le3'] and r['presence_target']==1 and not r['gt_pixel_M4']}
    for band in ['CLEAR_INTERIOR','CLEAR_EXTERIOR','BOUNDARY_UNCERTAIN','IMAGE_BORDER_UNCERTAIN']:
        filters['ERROR_LE_3_'+band]=lambda r,b=band:r['near_le3'] and r['boundary']==b
    for role in ['B','I']:
        filters['ERROR_LE_3_INITIAL_ROLE_'+role]=lambda r,k=role:r['near_le3'] and r['initial_geometric_role']==k
    for label,fun in filters.items():stats.append(metric([r for r in a if fun(r)],split,label))
    for scope,sub in [('ALL_VALID',a),('ERROR_LE3',[r for r in a if r['near_le3']]),('ERROR_LE05',[r for r in a if r['near_le05']])]:
        distinction.append(dict(split=split,scope=scope,n=len(sub),gt_M4_positive=sum(r['gt_pixel_M4'] is True for r in sub),
            q_M4_positive=sum(r['presence_target']==1 for r in sub),gt1_q0=sum(r['gt_pixel_M4'] is True and r['presence_target']==0 for r in sub),
            gt0_q1=sum(r['gt_pixel_M4'] is False and r['presence_target']==1 for r in sub),
            gt1_q0_pred0=sum(r['gt_pixel_M4'] is True and r['presence_target']==0 and not r['predicted_present'] for r in sub),
            external_ray_notblocked_q_M4_0=sum(r['stored_external_ray_loss_valid'] and r['stored_external_ray_proxy_blocked']==0 and r['presence_target']==0 for r in sub),
            note='GT M4 membership is projected cuboid pixel support; neither it nor ray proxy is certified human-like semantic DIRECT truth'))
csvsave('SYNTH_NEAR_METRICS.csv',stats);csvsave('SYNTH_TARGET_DISTINCTION.csv',distinction)
fraction_rows=[];correlations=[]
for split in ['train','validation']:
    for scope in ['ALL','ERROR_LE3','ERROR_LE3_BOUNDARY']:
        a=[r for r in rows if r['valid'] and r['split']==split and (scope=='ALL' or r['near_le3']) and (scope!='ERROR_LE3_BOUNDARY' or r['boundary']=='BOUNDARY_UNCERTAIN')]
        for target in [None,0,1]:
            b=[r for r in a if target is None or r['presence_target']==target]
            if not b:continue
            xx=np.array([r['M4_crop_fraction_native_pixel_centers'] for r in b]);pp=np.array([r['presence_probability'] for r in b]);yy=np.array([r['presence_target'] for r in b])
            correlations.append(dict(split=split,scope=scope,target='ALL' if target is None else target,n=len(b),
                pearson_probability_vs_crop_fraction=float(np.corrcoef(xx,pp)[0,1]) if np.std(xx)>0 and np.std(pp)>0 else None,
                median_crop_fraction=float(np.median(xx)),median_probability=float(np.median(pp))))
        for lo,hi in [(0,.1),(.1,.25),(.25,.5),(.5,.75),(.75,1.00001)]:
            b=[r for r in a if lo<=r['M4_crop_fraction_native_pixel_centers']<hi]
            if b:fraction_rows.append(dict(scope=scope,crop_fraction_min=lo,crop_fraction_max=min(1.,hi),**metric(b,split,'M4_crop_fraction')))
csvsave('SYNTH_CROP_FRACTION_METRICS.csv',fraction_rows);csvsave('SYNTH_CROP_FRACTION_CORRELATIONS.csv',correlations)
va=[r for r in rows if r['split']=='validation' and r['near_le3']]
spec=[
    ('Near target mismatch: GT M4=1, query M4=0; predicted absent',lambda r:r['gt_pixel_M4'] and not r['presence_target'] and not r['predicted_present']),
    ('Boundary target-positive miss at nearly exact coordinate',lambda r:r['presence_target']==1 and not r['predicted_present'] and r['boundary']=='BOUNDARY_UNCERTAIN'),
    ('Clear-interior target-positive miss',lambda r:r['presence_target']==1 and not r['predicted_present'] and r['boundary']=='CLEAR_INTERIOR'),
    ('Clear-interior correct presence control',lambda r:r['presence_target']==1 and r['predicted_present'] and r['boundary']=='CLEAR_INTERIOR'),
    ('Ray-not-blocked but query has no target pixel',lambda r:r['stored_external_ray_loss_valid'] and r['stored_external_ray_proxy_blocked']==0 and not r['presence_target'] and not r['predicted_present']),
    ('Boundary correct presence control',lambda r:r['presence_target']==1 and r['predicted_present'] and r['boundary']=='BOUNDARY_UNCERTAIN')]
selection=[]
for title,fun in spec:
    candidates=[r for r in va if fun(r)]
    if candidates:
        r=min(candidates,key=lambda r:(r['query_to_native_renderer_gt_px'],r['id'],r['native_corner']));selection.append(dict(case=title,**r))
save('SYNTH_CROP_CASES.json',selection)
fig,axs=plt.subplots(len(selection),4,figsize=(14,3.8*len(selection)),layout='constrained')
for i,r in enumerate(selection):
    x=by[r['id']];rgb=np.array(Image.open(x['image']).convert('RGB'));a=mask_audit[r['id']];m4=np.array(Image.open(a['visible_path']))>127
    q=np.array([r['query_x'],r['query_y']],float);g=np.array([r['gt_x'],r['gt_y']]);L=r['query_crop_side_native_px'];lo=q-L/2;hi=q+L/2
    for j in range(4):
        ax=axs[i,j];ax.set_axis_off();ax.imshow(rgb if j!=2 else m4,cmap='gray' if j==2 else None,interpolation='nearest')
        if j==0:
            ax.add_patch(Rectangle(lo,L,L,fill=False,edgecolor='cyan',lw=1));ax.set_title(r['id']+f' / native k{r["native_corner"]}',fontsize=9)
        else:
            if j==3:ax.set_xlim(q[0]-6.5,q[0]+6.5);ax.set_ylim(q[1]+6.5,q[1]-6.5)
            else:ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1])
            ax.plot(*q,marker='+',color='cyan',ms=10,mew=1.2);ax.plot(*g,marker='x',color='magenta',ms=8,mew=1.3)
            ax.add_patch(Rectangle((r['pixel_x']-.5,r['pixel_y']-.5),1,1,fill=False,edgecolor='yellow',lw=1))
            ax.set_title(['','RGB: same native crop','M4: white = visible target','13 px RGB detail'][j],fontsize=9)
    axs[i,0].text(0,-.08,r['case'],ha='left',va='top',transform=axs[i,0].transAxes,fontsize=8,wrap=True)
    axs[i,1].text(0,-.08,f'Error={r["query_to_native_renderer_gt_px"]:.3f} px; crop side={L:.1f} px\np(present)={r["presence_probability"]:.3f}; target={r["presence_target"]}; GT-pixel M4={int(r["gt_pixel_M4"])}',ha='left',va='top',transform=axs[i,1].transAxes,fontsize=8)
    axs[i,2].text(0,-.08,r['boundary']+'; 3x3 sum='+str(r.get('M4_stencil3_sum')),ha='left',va='top',transform=axs[i,2].transAxes,fontsize=8)
fig.suptitle('Saved synthetic validation: target definition and near-query failures',fontsize=15)
fig.supxlabel('Cyan +: unchanged initial query; magenta x: saved native renderer GT projection; yellow square: one labeled nearest pixel.\nM4 center-pixel presence is distinct from semantic visibility. No model call, training, PnP, crop-center change, or threshold tuning.',fontsize=9)
fig.savefig(OUT/'SYNTH_NEAR_CROP_DIAGNOSIS.png',dpi=140);plt.close(fig)
save('SYNTH_AUDIT_SUMMARY.json',dict(no_training=True,no_model_forward=True,no_PnP=True,threshold=.5,query_coordinate_changes=0,
    frames=len(u),queries=len(rows),renderer_gt_coordinate_mapping_max_abs_px=mapping_max,
    near_eligibility='valid query AND native renderer GT continuous coordinate in image AND its nearest pixel in image AND Euclidean query-to-GT error <=3px',
    error_le3_but_GT_invalid_exclusions=[dict(id=r['id'],split=r['split'],native_corner=r['native_corner'],gt_xy=[r['gt_x'],r['gt_y']],gt_nearest_pixel_xy=[r['gt_nearest_pixel_x'],r['gt_nearest_pixel_y']],error_px=r['query_to_native_renderer_gt_px']) for r in rows if r['valid'] and r['query_to_native_renderer_gt_px']<=3 and not r['gt_coordinate_valid']],
    nonnear_stratum='NOT_ELIGIBLE_NEAR includes error>3 and GT-invalid references; it is not an error>3-only stratum',
    saved_renderer_GT_binding=str(GT),certified_synthetic_semantic_DIRECT_labels_available=False,
    metrics=stats,objective_distinction=distinction,
    sources=[dict(path=str(p),sha256=sha(p)) for p in [V61/'PROTOCOL.json',V61/'SYNTH_QUERY_PREDICTIONS.csv',V61/'MASK_CONTRACT_AUDIT.csv',V60/'PREPARED_INPUTS_SYNTH.json.gz',GT]],
    case_selection='validation only, error<=3 px; deterministic minimum error then ID/corner within diagnostic categories; qualitative cases, not representative prevalence'))
print(json.dumps(dict(metrics=[r for r in stats if r['split']=='validation'],distinction=[r for r in distinction if r['split']=='validation'],cases=[dict(id=r['id'],k=r['native_corner'],error=r['query_to_native_renderer_gt_px'],title=r['case']) for r in selection]),indent=2))
