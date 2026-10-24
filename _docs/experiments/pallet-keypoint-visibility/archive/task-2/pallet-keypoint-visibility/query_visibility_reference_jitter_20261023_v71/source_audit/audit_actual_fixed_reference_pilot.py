"""Independent source raster/GT audit; no builder/contract/model imports."""
from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib,math
import numpy as np
from PIL import Image

V=Path(__file__).resolve().parents[1];O=V/'data/synth_reference_jitter/BUILD_V2'
P=V.parent;S=P/'query_visibility_reference_semantic_20261023_v69'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def image(p):return np.array(Image.open(p).convert('RGB'))
def mask(p):
    a=np.array(Image.open(p));assert a.ndim==2
    return a>0
result=read(O/'PILOT_RESULT.json');protocol=read(result['protocol']['path'])
for b in result['outputs']+[result['protocol'],result['code']]:assert sha(b['path'])==b['sha256']
rows=[json.loads(s)for s in(O/'ROWS.jsonl').read_text().splitlines()]
cert=read(O/'INTERVENTION_CERTIFICATES.json')
ledger=read(S/'source_cue_reference_proposals/AI_OPERATIONAL_CUE_REFERENCE_PROPOSAL_LEDGER.json')
proposals={r['selection_rank']:r for r in ledger['proposals']}
with np.load(O/'MODEL_INPUTS_ONLY.npz',allow_pickle=False)as z:
    assert set(z.files)=={'rgb','valid','initial_query_xy','input_crop_L','target_box_xyxy'}
    x={k:z[k]for k in z.files}
with np.load(O/'GT_ONLY.npz',allow_pickle=False)as z:
    assert set(z.files)=={'reference_state','training_target','availability'}
    g={k:z[k]for k in z.files}
assert len(rows)==260 and x['rgb'].shape==(260,96,96,3)and x['valid'].shape==(260,96,96)
assert x['rgb'].dtype==x['valid'].dtype==np.uint8
assert all(len(a)==260 for a in list(x.values())+list(g.values()))
assert len({r['row_id']for r in rows})==260
accepted=set(protocol['root_actual_RGB_review']['conditional_D_approved_ranks'])
variants={(v['rank'],v['variant']):v for v in cert['variants']};assert len(variants)==20
grids=[('PRIMARY',dx,dy)for dx,dy in protocol['jitter']['primary_grid_L']]+[('LOCATION_STRESS',dx,dy)for dx,dy in protocol['jitter']['location_stress_grid_L']]
originals={};visible={};raster={};alpha={};region_stats=[];donor_images={}
checks=Counter();bindings=[]
for key,v in variants.items():
    rank,kind=key;p=proposals[rank]
    for b in [v['image'],v['alpha_GT_only']]:assert sha(b['path'])==b['sha256']
    if rank not in originals:
        assert sha(p['original_image'])==p['original_image_sha256']
        originals[rank]=image(p['original_image'])
        mp=Path(p['original_image']).parent.parent/'mask_visible'/Path(p['original_image']).name.replace('_rgb.png','.png')
        visible[rank]=np.array(Image.open(mp))>127
    im=image(v['image']['path']);a=mask(v['alpha_GT_only']['path']);orig=originals[rank]
    assert np.array_equal(im[~a],orig[~a]);checks['raster_outside_alpha_bit_exact']+=1
    if kind=='ORIGINAL':assert not a.any()and np.array_equal(im,orig)
    else:
        m=v['intervention_GT_only'];d=m['donor'];assert d['split']=='TRAIN'and m['distinct_instance']is True
        for b in [d['image'],d['visible']]:assert sha(b['path'])==b['sha256']
        if d['id']not in donor_images:donor_images[d['id']]=(image(d['image']['path']),mask(d['visible']['path']))
        di,dm=donor_images[d['id']];bx=m['original_donor_bbox'];rw,rh=m['resized_wh'];left,top=m['integer_offset_xy']
        dc=di[bx[1]:bx[3],bx[0]:bx[2]];mc=dm[bx[1]:bx[3],bx[0]:bx[2]]
        dc=np.array(Image.fromarray(dc).resize((rw,rh),Image.Resampling.BILINEAR))
        mc=np.array(Image.fromarray(mc.astype(np.uint8)*255).resize((rw,rh),Image.Resampling.NEAREST))>0
        replay_alpha=np.zeros_like(a);replay=orig.copy();H,W=a.shape
        for yy in range(rh):
            for xx in range(rw):
                x0,y0=left+xx,top+yy
                if 0<=x0<W and 0<=y0<H and mc[yy,xx]:replay_alpha[y0,x0]=True;replay[y0,x0]=dc[yy,xx]
        assert np.array_equal(a,replay_alpha)and np.array_equal(im,replay)
        checks['donor_alpha_and_RGB_full_replay_bit_exact']+=1
    if rank in accepted:
        u=p['uncertainty_bbox_half_open_xyxy'];bounds=[math.floor(u[0]+.5),math.floor(u[1]+.5),math.ceil(u[2]+.5),math.ceil(u[3]+.5)]
        cut=a[bounds[1]:bounds[3],bounds[0]:bounds[2]];n=int(cut.sum());area=cut.size
        state='D_teacher'if n==0 else('E_teacher'if n==area else'U')
        assert state==v['semantic_GT_only']['reference_state']
        assert state=={'ORIGINAL':'D_teacher','FULL_COVER':'E_teacher','PARTIAL_COVER':'U','QUERY_ONLY_CONTROL':'D_teacher'}[kind]
        assert v['reference_GT_only']['reference_xy']==p['associated_cue_xy']and v['reference_GT_only']['uncertainty_bbox']==u
        region_stats.append(dict(rank=rank,variant=kind,region_cells=area,opaque_cells=n,reference_state=state))
    else:assert v['reference_GT_only']is None and v['semantic_GT_only']['reference_state']=='U'
    raster[key]=im;alpha[key]=a

def recrop(rgb,q,L):
    # Independent Cartesian pixel-center formula and bilinear interpolation.
    H,W=rgb.shape[:2]
    xs=q[0]+(np.arange(96)+.5-48)*(L/96)
    ys=q[1]+(np.arange(96)+.5-48)*(L/96)
    xx=np.repeat(xs[None,:],96,axis=0);yy=np.repeat(ys[:,None],96,axis=1)
    valid=(xx>=-.5)&(xx<W-.5)&(yy>=-.5)&(yy<H-.5)
    xx=np.minimum(np.maximum(xx,0),W-1);yy=np.minimum(np.maximum(yy,0),H-1)
    xa=np.floor(xx).astype(int);ya=np.floor(yy).astype(int);xb=np.minimum(xa+1,W-1);yb=np.minimum(ya+1,H-1)
    fx=(xx-xa)[:,:,None];fy=(yy-ya)[:,:,None]
    val=rgb[ya,xa]*(1-fx)*(1-fy)+rgb[ya,xb]*fx*(1-fy)+rgb[yb,xa]*(1-fx)*fy+rgb[yb,xb]*fx*fy
    val=np.floor(val+.5).clip(0,255).astype(np.uint8);val[~valid]=0
    return val,valid

target_code={'D_teacher':1,'E_teacher':0,'U':-1};av_code={'FULL':2,'PARTIAL':1,'MISS':0,'INVALID':-1}
counts=Counter();counterexamples=Counter();raw_groups=defaultdict(list);crop_groups=defaultdict(list)
crop_max_error=0;crop_nonidentical_channels=0;index={}
for i,r in enumerate(rows):
    rank=r['cue_rank'];p=proposals[rank];kind=r['variant'];im=raster[(rank,kind)];a=alpha[(rank,kind)]
    jitter=int(r['row_id'].rsplit('_J',1)[1]);role,dx,dy=grids[jitter]
    q=np.array(p['original_query_xy'])+p['original_side_L']*np.array([dx,dy]);L=p['original_side_L'];box=p['original_target_box_xyxy']
    assert role==r['jitter_role_GT_only']and r['delta_L_GT_only']==[dx,dy]
    assert np.array_equal(x['initial_query_xy'][i],q)and x['input_crop_L'][i]==L and np.array_equal(x['target_box_xyxy'][i],box)
    assert np.array_equal(q,r['model_inputs']['initial_query_xy'])and r['model_inputs']['input_crop_L']==L and r['model_inputs']['target_box_xyxy']==box
    assert sha(r['crop']['path'])==r['crop']['sha256']and np.array_equal(image(r['crop']['path']),x['rgb'][i])
    if rank in accepted:
        u=p['uncertainty_bbox_half_open_xyxy'];H,W=im.shape[:2]
        c=[max(0,q[0]-L/2),max(0,q[1]-L/2),min(W,q[0]+L/2),min(H,q[1]+L/2)]
        overlap=max(0,min(c[2],u[2])-max(c[0],u[0]))*max(0,min(c[3],u[3])-max(c[1],u[1]))
        full=u[0]>=c[0]and u[1]>=c[1]and u[2]<=c[2]and u[3]<=c[3]
        av='FULL'if full else('PARTIAL'if overlap>0 else'MISS')
        st=next(s['reference_state']for s in region_stats if s['rank']==rank and s['variant']==kind)
    else:st='U';av='INVALID'
    target=st if av=='FULL'and role=='PRIMARY'and st in ['D_teacher','E_teacher']else'U'
    assert st==r['reference_state_GT_only']and av==r['availability_GT_only']and target==r['training_target_GT_only']
    assert g['reference_state'][i]==target_code[st]and g['training_target'][i]==target_code[target]and g['availability'][i]==av_code[av]
    crop,va=recrop(im,q,L);assert np.array_equal(va,x['valid'][i].astype(bool))
    error=np.abs(crop.astype(int)-x['rgb'][i].astype(int));crop_max_error=max(crop_max_error,int(error.max()));crop_nonidentical_channels+=int((error!=0).sum())
    assert error.max()<=1  # Algebraically equivalent floating summation may round an exact half by one gray level.
    counts[(kind,role,st,av,target)]+=1;checks['rows_GT_input_qLbox_and_crop_verified']+=1
    H,W=im.shape[:2];px,py=math.floor(q[0]+.5),math.floor(q[1]+.5)
    native_valid=bool(0<=q[0]<W and 0<=q[1]<H and 0<=px<W and 0<=py<H)
    after=visible[rank]&~a
    exact=bool(after[py,px])if native_valid else None
    near=bool(after[max(0,py-2):min(H,py+3),max(0,px-2):min(W,px+3)].any())if native_valid else None
    saved=r['native_query_support_GT_only'];assert saved['valid']==native_valid and saved['exact']==exact and saved['radius2']==near
    if av=='FULL'and role=='PRIMARY':
        if st=='D_teacher'and native_valid and not exact:counterexamples['D_but_query_background']+=1
        if st=='E_teacher'and native_valid and exact:counterexamples['E_but_query_visible_target']+=1
    # Group actual exposed input bytes independently, including actual numeric input dtypes.
    common=q.astype(np.float64).tobytes()+np.array([L],np.float64).tobytes()+np.asarray(box,np.float64).tobytes()
    rawkey=hashlib.sha256(im.tobytes()+common).hexdigest();cropkey=hashlib.sha256(x['rgb'][i].tobytes()+x['valid'][i].tobytes()+common).hexdigest()
    raw_groups[rawkey].append(i);crop_groups[cropkey].append(i);index[(rank,kind,jitter)]=i

reported_counts=Counter({(r['variant'],r['jitter_role'],r['reference_state'],r['availability'],r['training_target']):r['N']for r in result['counts']})
assert counts==reported_counts and dict(counterexamples)==result['query_reference_counterexamples']
collisions=[];supervised=[]
for scope,groups in [('full_permitted',raw_groups),('effective_crop',crop_groups)]:
    for key,ix in groups.items():
        states={int(g['reference_state'][i])for i in ix if g['reference_state'][i]>=0}
        targets={int(g['training_target'][i])for i in ix if g['training_target'][i]>=0}
        if len(states)>1:
            assert all(g['training_target'][i]==-1 for i in ix)
            collisions.append(dict(scope=scope,rows=[rows[i]['row_id']for i in ix],all_training_U=True))
        if len(targets)>1:supervised.append(dict(scope=scope,rows=ix))
assert len(collisions)==result['semantic_state_collisions']==21 and not supervised

neutral=[]
for i,r in enumerate(rows):
    if r['variant']=='QUERY_ONLY_CONTROL'and r['availability_GT_only']=='FULL'and r['jitter_role_GT_only']=='PRIMARY':
        j=int(r['row_id'].rsplit('_J',1)[1]);oi=index[(r['cue_rank'],'ORIGINAL',j)]
        change=(x['rgb'][i]!=x['rgb'][oi]).any(2);channel=(x['rgb'][i]!=x['rgb'][oi])
        neutral.append(dict(row_id=r['row_id'],rank=r['cue_rank'],original_row=rows[oi]['row_id'],
            RGB_changed_pixels=int(change.sum()),RGB_changed_channels=int(channel.sum()),
            fraction_changed_RGB_pixels=float(change.mean()),observable_RGB_paste_difference=bool(change.any()),
            reference_D_training_preserved=bool(g['reference_state'][i]==1 and g['training_target'][i]==1)))
assert len(neutral)==27
parity=read(O/'V1_RASTER_REPLAY_PARITY.json')
audit=dict(schema='V71_ACTUAL_FIXED_REFERENCE_PILOT_INDEPENDENT_AUDIT_V1',completed=True,status='PASS',
 builder_or_contract_functions_imported=False,actual_rows=260,actual_rasters=20,checks=dict(checks),
 model_input_keys=sorted(x),GT_only_keys=sorted(g),GT_input_leakage_arrays=0,
 labels_recomputed_from_fixed_reference_and_actual_alpha=True,q_L_box_jitter_verbatim=True,
 independent_crop_reconstruction=dict(maximum_RGB_channel_difference=crop_max_error,nonidentical_RGB_channels=crop_nonidentical_channels,all_observed_valid_masks_exact=True,all_saved_PNG_to_model_RGB_arrays_exact=True),
 actual_alpha_and_RGB_replay=dict(all20_outside_alpha_bit_exact=True,all12_nonoriginal_full_RGB_and_alpha_replay_bit_exact=True,distinct_TRAIN_donors=len(donor_images),region_stats=region_stats),
 training_target_counts={str(k):int(v)for k,v in Counter(g['training_target'].tolist()).items()},
 reference_state_counts={str(k):int(v)for k,v in Counter(g['reference_state'].tolist()).items()},
 query_reference_counterexamples=dict(counterexamples),semantic_state_collisions=collisions,
 expected_U_observation_collision_groups=21,contradictory_supervised_D_E_collisions=supervised,
 collision_interpretation='Underlying D/E may differ at identical observed MISS input. These21 groups are all trainingU; they demonstrate non-observability, not supervised label repair or global identifiability.',
 neutral_D_full_primary=dict(N=27,observable_RGB_paste_difference=sum(n['observable_RGB_paste_difference']for n in neutral),identical_to_original=sum(not n['observable_RGB_paste_difference']for n in neutral),rows=neutral),
 model_fit_ready=False,independent_VAL=0,semantic_native8_mapping='UNKNOWN',human_gold=0,
 V1_V2_raster_parity_report=bind(O/'V1_RASTER_REPLAY_PARITY.json'),
 limitations=['RGB differences only prove intervention pixels enter the model crop; they do not certify an occluder is visually recognizable.','Only3 supervised FULL references in2source RGBframes; no generalization/goal-completion claim.','Legacy helper evidence_tier/admission flags remain unadmitted; root pilot authorizes construction, not model fit.'],
 bindings=[bind(O/n)for n in ['PILOT_RESULT.json','MODEL_INPUTS_ONLY.npz','GT_ONLY.npz','ROWS.jsonl','INTERVENTION_CERTIFICATES.json','INPUT_COLLISION_AUDIT.json']]+[bind(result['code']['path']),bind(result['protocol']['path']),bind(__file__)],
 source_RGB_and_alpha_reads_for_authorized_audit=True,P3_feature_or_weights_or_current_reads=0,new_model_forward=0,fit=0,PnP=0,prior_artifacts_modified=False)
out=V/'source_audit/ACTUAL_PILOT_INDEPENDENT_AUDIT.json';assert not out.exists()
out.write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(dict(status='PASS',audit=bind(out),checks=dict(checks),crop_error=crop_max_error,counterexamples=dict(counterexamples),neutral_observable=audit['neutral_D_full_primary']['observable_RGB_paste_difference'],neutral_N=27,training_counts=audit['training_target_counts'])))
