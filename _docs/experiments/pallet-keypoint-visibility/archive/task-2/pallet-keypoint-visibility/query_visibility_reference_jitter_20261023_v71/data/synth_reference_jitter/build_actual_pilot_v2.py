"""Build actual source-only fixed-reference/jitter rasters; no model or pose."""
from pathlib import Path
import collections, hashlib, importlib.util, json, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE=Path(__file__).resolve().parent
O=BASE/'BUILD_V2'
V=BASE.parent.parent
P=V.parent
S=P/'query_visibility_reference_semantic_20261023_v69'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p): return {'path':str(p),'sha256':sha(p)}
def savejson(path,d):
    assert not path.exists(), path
    path.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def largest_square(mask):
    """Largest all-opaque native square, central tie first; integral image."""
    h,w=mask.shape
    ii=np.pad(mask.astype(np.int64),((1,0),(1,0))).cumsum(0).cumsum(1)
    def candidates(k):
        counts=ii[k:,k:]-ii[:-k,k:]-ii[k:,:-k]+ii[:-k,:-k]
        return np.argwhere(counts==k*k)
    lo,hi=1,min(h,w);best=None
    while lo<=hi:
        k=(lo+hi)//2;c=candidates(k)
        if len(c):best=(k,c);lo=k+1
        else:hi=k-1
    if best is None:raise ValueError('Donor has no visible native pixel')
    k,c=best
    distances=((c+(k-1)/2-np.array([(h-1)/2,(w-1)/2]))**2).sum(1)
    y,x=c[int(np.argmin(distances))]
    return int(x),int(y),int(k)

def paste_layer(shape, donor_rgb, donor_mask, center):
    """Known opaque other-instance front layer, no blending/antialias alpha."""
    h,w=shape;dh,dw=donor_mask.shape
    left=int(round(float(center[0])-(dw-1)/2));top=int(round(float(center[1])-(dh-1)/2))
    alpha=np.zeros((h,w),np.uint8);color=np.zeros((h,w,3),np.uint8)
    x0,y0=max(0,left),max(0,top);x1,y1=min(w,left+dw),min(h,top+dh)
    if x1>x0 and y1>y0:
        dx,dy=x0-left,y0-top
        alpha[y0:y1,x0:x1]=donor_mask[dy:dy+y1-y0,dx:dx+x1-x0]
        color[y0:y1,x0:x1]=donor_rgb[dy:dy+y1-y0,dx:dx+x1-x0]
    return alpha,color,[left,top]

def sample_crop(rgb,q,L,N=96):
    h,w=rgb.shape[:2]
    yy,xx=np.meshgrid((np.arange(N)-(N-1)/2)*(L/N)+q[1],(np.arange(N)-(N-1)/2)*(L/N)+q[0],indexing='ij')
    valid=(xx>=-.5)&(xx<w-.5)&(yy>=-.5)&(yy<h-.5)
    x=np.clip(xx,0,w-1);y=np.clip(yy,0,h-1)
    x0=np.floor(x).astype(int);y0=np.floor(y).astype(int);x1=np.minimum(x0+1,w-1);y1=np.minimum(y0+1,h-1)
    wx=(x-x0)[...,None];wy=(y-y0)[...,None]
    z=(1-wy)*((1-wx)*rgb[y0,x0]+wx*rgb[y0,x1])+wy*((1-wx)*rgb[y1,x0]+wx*rgb[y1,x1])
    z=np.clip(np.floor(z+.5),0,255).astype(np.uint8);z[~valid]=0
    return z,valid.astype(np.uint8)

def signature(*arrays):
    h=hashlib.sha256()
    for arr in arrays:
        a=np.ascontiguousarray(arr);h.update(str(a.shape).encode()+str(a.dtype).encode()+a.tobytes())
    return h.hexdigest()

def main():
    protocol=BASE/'PILOT_PROTOCOL.json'
    assert sha(protocol)=='e5c8699f1fd330b3d12bb2366f4d3ed93a51fb92fd2a45430d0ef69e691e81db'
    spec=json.loads(protocol.read_text())
    for b in spec['bindings']:assert sha(b['path'])==b['sha256'],b
    ledger_path=S/'source_cue_reference_proposals/AI_OPERATIONAL_CUE_REFERENCE_PROPOSAL_LEDGER.json'
    proposals=json.loads(ledger_path.read_text())['proposals']
    accepted=set(spec['root_actual_RGB_review']['conditional_D_approved_ranks'])
    module=importlib.util.spec_from_file_location('v69_reference_contract',S/'reference_visibility_contract.py')
    contract=importlib.util.module_from_spec(module)
    import sys
    sys.modules[module.name]=contract;module.loader.exec_module(contract)
    assert largest_square(np.ones((5,7),bool))==(1,0,5)
    targets_path=P/'query_visibility_controlled_real_20261022_v66/provenance/TARGETS.jsonl'
    pool=[json.loads(x) for x in targets_path.read_text().splitlines() if json.loads(x)['split']=='TRAIN']
    donors=sorted(pool,key=lambda x:hashlib.sha256(x['id'].encode()).hexdigest())[:4]
    O.mkdir()
    savejson(O/'EXECUTION_ENTRYLOCK.json',{'protocol':bind(protocol),'code':bind(Path(__file__)),'donors':[{'id':d['id'],'split':d['split'],'image':d['image'],'visible':d['visible']} for d in donors],'parent_proposals':bind(ledger_path),'root_approved_conditional_D':sorted(accepted),'training':False,'current_access':False})
    for name in ['rasters','alpha_GT_only','crops_INPUT_only']: (O/name).mkdir()
    rows=[];variants=[];inputs=[];valids=[];qs=[];boxes=[];Ls=[];gtstates=[];gttargets=[];availability=[]
    source_rgb_cache={};source_mask_cache={};donor_cache={};skip=[]
    raw_hash_groups=collections.defaultdict(list);effective_hash_groups=collections.defaultdict(list)
    grids=[('PRIMARY',x,y) for x,y in spec['jitter']['primary_grid_L']]+[('LOCATION_STRESS',x,y) for x,y in spec['jitter']['location_stress_grid_L']]
    observed=[]
    for proposal in proposals:
        rank=proposal['selection_rank'];path=Path(proposal['original_image'])
        if str(path) not in source_rgb_cache:
            assert sha(path)==proposal['original_image_sha256']
            source_rgb_cache[str(path)]=np.array(Image.open(path).convert('RGB'))
            mpath=path.parent.parent/'mask_visible'/path.name.replace('_rgb.png','.png')
            source_mask_cache[str(path)]=np.array(Image.open(mpath))>127
            observed.append({'image':bind(path),'M4_GT_only':bind(mpath)})
        rgb=source_rgb_cache[str(path)];mask=source_mask_cache[str(path)];h,w=mask.shape
        assert list(rgb.shape[:2])==proposal['raw_hw']
        q0=np.array(proposal['original_query_xy'],float);L=float(proposal['original_side_L']);box=np.array(proposal['original_target_box_xyxy'],float)
        ref=None
        if rank in accepted:
            ref=contract.ReferenceGT(proposal['frame_id'],proposal['NEW_cue_refID'],tuple(proposal['associated_cue_xy']),tuple(proposal['uncertainty_bbox_half_open_xyxy']),original_state='D_teacher',original_D_teacher_evidence='V71 root actual RGB exposed outline/corner-region review, AI conditional teacher; not human/native8 gold',reference_provenance=str(ledger_path),source_name='SYNTH_AI_OPERATIONAL_CUE')
        local=[('ORIGINAL',rgb,np.zeros((h,w),np.uint8),None,{})]
        if ref:
            donor=donors[rank%4];did=donor['id']
            if did not in donor_cache:
                assert sha(donor['image']['path'])==donor['image']['sha256']
                assert sha(donor['visible']['path'])==donor['visible']['sha256']
                drgb=np.array(Image.open(donor['image']['path']).convert('RGB'));dm=np.array(Image.open(donor['visible']['path']))>0
                yy,xx=np.where(dm);db=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
                donor_cache[did]=(drgb[db[1]:db[3],db[0]:db[2]],dm[db[1]:db[3],db[0]:db[2]],db)
            drgb,dm,db=donor_cache[did];cx,cy,core=largest_square(dm)
            geom=contract.reference_geometry(ref,(h,w));ux0,uy0,ux1,uy1=geom['region_bounds'];uh,uw=uy1-uy0,ux1-ux0
            scale=max(.25*L/max(dm.shape),(max(uw,uh)+4)/core)
            resized_hw=(max(1,math.ceil(dm.shape[0]*scale)),max(1,math.ceil(dm.shape[1]*scale)))
            rh,rw=resized_hw
            rr=np.array(Image.fromarray(drgb).resize((rw,rh),Image.Resampling.BILINEAR))
            rm=np.array(Image.fromarray(dm.astype(np.uint8)*255).resize((rw,rh),Image.Resampling.NEAREST))>0
            # Recompute actual resized opaque core, not assumed scaled mask geometry.
            rx,ry,rk=largest_square(rm)
            if rk<max(uw,uh): raise ValueError('Resized donor core insufficient; preserve failure rather than shrink uncertainty')
            core_center=np.array([rx+(rk-1)/2,ry+(rk-1)/2]);unc_center=np.array([(ux0+ux1-1)/2,(uy0+uy1-1)/2])
            center=unc_center-core_center+np.array([(rw-1)/2,(rh-1)/2])
            def layer(c):return paste_layer((h,w),rr,rm.astype(np.uint8),c)
            alpha,color,offset=layer(center)
            assert np.all(alpha[uy0:uy1,ux0:ux1]==1)
            meta={'donor':{'id':did,'image':donor['image'],'visible':donor['visible'],'split':'TRAIN'},'original_donor_bbox':db,'resized_wh':[rw,rh],'integer_offset_xy':offset,'distinct_instance':True,'front_certificate':'Explicit opaque compositing after the complete target RGB; not inferred depth/PnP','reference_region_cells':int(uw*uh)}
            def combined(a,c):return np.where(a[...,None]>0,c,rgb).astype(np.uint8)
            local.append(('FULL_COVER',combined(alpha,color),alpha,contract.OccluderGT(alpha,did,'Explicit distinct opaque composited front layer'),meta))
            partial=None
            # All candidates determined by fixed native geometry, never scores or q/r relocation.
            for displacement in range(1,rw+uw+2):
                aa,cc,off=layer(center+np.array([displacement,0]));n=int(aa[uy0:uy1,ux0:ux1].sum())
                if 0<n<uw*uh:
                    partial=(aa,cc,off,n);break
            assert partial is not None,'No partial-cover variant; do not fabricate coverage'
            aa,cc,off,n=partial
            local.append(('PARTIAL_COVER',combined(aa,cc),aa,contract.OccluderGT(aa,did,'Explicit distinct opaque composited front layer'),dict(meta,integer_offset_xy=off,opaque_reference_cells=n)))
            neutral=None
            for group,dx,dy in grids:
                qq=q0+L*np.array([dx,dy]);xi,yi=np.floor(qq+.5).astype(int)
                if not(0<=xi<w and 0<=yi<h):continue
                c=qq-core_center+np.array([(rw-1)/2,(rh-1)/2]);aa,cc,off=layer(c)
                if aa[yi,xi] and not aa[uy0:uy1,ux0:ux1].any(): neutral=(aa,cc,off,qq.tolist());break
            if neutral:
                aa,cc,off,qq=neutral
                local.append(('QUERY_ONLY_CONTROL',combined(aa,cc),aa,contract.OccluderGT(aa,did,'Explicit distinct opaque composited front layer'),dict(meta,integer_offset_xy=off,covered_preexisting_grid_q=qq,opaque_reference_cells=0)))
            else:skip.append({'rank':rank,'variant':'QUERY_ONLY_CONTROL','reason':'No preexisting fixed-grid inframe q with an opaque cutout footprint fully clear of reference uncertainty'})
        for variant,vrgb,alpha,occluder,meta in local:
            stem=f'{rank:02d}_{variant}';raster=O/'rasters'/(stem+'.png');apath=O/'alpha_GT_only'/(stem+'.png')
            Image.fromarray(vrgb).save(raster);Image.fromarray(alpha*255).save(apath)
            assert np.array_equal(vrgb[alpha==0],rgb[alpha==0])
            semantic=contract.controlled_reference_state(ref,(h,w),occluder) if ref else {'reference_state':'U','reason':'Unadmitted source cue; no semantic state inferred from masks'}
            variants.append({'rank':rank,'frame_id':proposal['frame_id'],'variant':variant,'image':bind(raster),'alpha_GT_only':bind(apath),'reference_GT_only':None if not ref else {'reference_id':ref.reference_id,'reference_xy':list(ref.reference_xy),'uncertainty_bbox':list(ref.uncertainty_bbox)},'semantic_GT_only':semantic,'intervention_GT_only':meta,'outside_alpha_RGB_bit_exact':True})
            after_mask=mask & (alpha==0)
            for jitter_i,(role,dx,dy) in enumerate(grids):
                q=q0+L*np.array([dx,dy]);crop,valid=sample_crop(vrgb,q,L)
                av=contract.crop_availability(ref,q,L,(h,w)) if ref else {'availability':'INVALID','reason':'Unadmitted/null cue cannot define evidence availability'}
                state=semantic['reference_state'];target=state if av['availability']=='FULL' and role=='PRIMARY' and state in ('D_teacher','E_teacher') else 'U'
                rowid=f'{stem}_J{jitter_i:02d}';cp=O/'crops_INPUT_only'/(rowid+'.png');Image.fromarray(crop).save(cp)
                fullsig=signature(vrgb,q.astype(np.float64),np.array([L],np.float64),box.astype(np.float64))
                cropsig=signature(crop,valid,q.astype(np.float64),np.array([L],np.float64),box.astype(np.float64))
                native=contract.native_support_audit(after_mask,q)
                row={'row_id':rowid,'split':'TRAIN','source_group':proposal['frame_id'],'cue_rank':rank,'reference_GT_only':proposal['NEW_cue_refID'],'variant':variant,'jitter_role_GT_only':role,'delta_L_GT_only':[dx,dy],'model_inputs':{'rgb':bind(raster),'initial_query_xy':q.tolist(),'input_crop_L':L,'target_box_xyxy':box.tolist()},'crop':bind(cp),'model_input_full_sha256':fullsig,'model_input_effective_crop_sha256':cropsig,'reference_state_GT_only':state,'availability_GT_only':av['availability'],'training_target_GT_only':target,'native_query_support_GT_only':native,'native8_semantic_mapping':'UNKNOWN','semantic_gold':False,'human_verified':False}
                collision_record={'row_id':rowid,'reference_state':state,'training_target':target,'availability':av['availability'],'jitter_role':role}
                raw_hash_groups[fullsig].append(collision_record);effective_hash_groups[cropsig].append(collision_record)
                rows.append(row);inputs.append(crop);valids.append(valid);qs.append(q);boxes.append(box);Ls.append(L);gtstates.append({'D_teacher':1,'E_teacher':0,'U':-1}[state]);gttargets.append({'D_teacher':1,'E_teacher':0,'U':-1}[target]);availability.append({'FULL':2,'PARTIAL':1,'MISS':0,'INVALID':-1}[av['availability']])
    conflicts=[];training_conflicts=[]
    for kind,groups in [('full_permitted',raw_hash_groups),('effective_crop',effective_hash_groups)]:
        for sig,rr in groups.items():
            states={r['reference_state'] for r in rr if r['reference_state'] in ('D_teacher','E_teacher')}
            targets={r['training_target'] for r in rr if r['training_target'] in ('D_teacher','E_teacher')}
            if len(states)>1:conflicts.append({'scope':kind,'hash':sig,'rows':rr})
            if len(targets)>1:training_conflicts.append({'scope':kind,'hash':sig,'rows':rr})
    savejson(O/'INPUT_COLLISION_AUDIT.json',{'semantic_state_collisions':conflicts,'contradictory_supervised_D_E_targets':training_conflicts,'reason':'MISS/stress U observations can legitimately hide different reference states. Only identical observed inputs with conflicting supervised D/E targets block training.','global_identifiability_proved':False})
    assert not training_conflicts,'Conflicting supervised D/E labels for identical input bytes'
    parity=[]
    for raster in sorted((O/'rasters').glob('*.png')):
        old=BASE/'rasters'/raster.name
        parity.append({'name':raster.name,'same_PNG_bytes':sha(raster)==sha(old)})
    assert all(x['same_PNG_bytes'] for x in parity)
    savejson(O/'V1_RASTER_REPLAY_PARITY.json',{'unchanged_rasters':parity,'coordinate_mask_label_recipe_changed':False,'validation_logic_repaired_only':True})
    np.savez_compressed(O/'MODEL_INPUTS_ONLY.npz',rgb=np.stack(inputs),valid=np.stack(valids),initial_query_xy=np.array(qs,np.float64),input_crop_L=np.array(Ls,np.float64),target_box_xyxy=np.array(boxes,np.float64))
    np.savez_compressed(O/'GT_ONLY.npz',reference_state=np.array(gtstates,np.int8),training_target=np.array(gttargets,np.int8),availability=np.array(availability,np.int8))
    (O/'ROWS.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n' for x in rows))
    savejson(O/'INTERVENTION_CERTIFICATES.json',{'variants':variants,'skipped':skip,'observed_original_source_files':observed,'all_siblings_TRAIN':True,'new_PnP':0,'GT_masks_are_not_input':True})
    counts=collections.Counter((r['variant'],r['jitter_role_GT_only'],r['reference_state_GT_only'],r['availability_GT_only'],r['training_target_GT_only']) for r in rows)
    source_examples=collections.Counter()
    for r in rows:
        if r['availability_GT_only']=='FULL' and r['jitter_role_GT_only']=='PRIMARY':
            native=r['native_query_support_GT_only'];st=r['reference_state_GT_only']
            if native.get('valid') and st=='D_teacher' and native['exact'] is False:source_examples['D_but_query_background']+=1
            if native.get('valid') and st=='E_teacher' and native['exact'] is True:source_examples['E_but_query_visible_target']+=1
    summary={'schema':'V71_ACTUAL_FIXED_REFERENCE_JITTER_PILOT','status':'ACTUAL_RASTERS_AND_INPUT_CACHE_BUILT_NOT_MODEL_FIT','completed':True,'protocol':bind(protocol),'code':bind(Path(__file__)),'rows':len(rows),'source_original_proposals':8,'conditional_D_reference_count':4,'approved_source_RGB_frames':3,'primary_full_learning_refs':3,'primary_full_learning_RGB_frames':2,'independent_VAL':0,'source_RGB_decoded':len(source_rgb_cache),'distinct_donor_instances_decoded':len(donor_cache),'counts':[{'variant':k[0],'jitter_role':k[1],'reference_state':k[2],'availability':k[3],'training_target':k[4],'N':n} for k,n in sorted(counts.items())],'query_reference_counterexamples':dict(source_examples),'semantic_state_collisions':len(conflicts),'contradictory_supervised_D_E_target_collisions':training_conflicts,'global_semantic_identifiability_proved':False,'model_fit_authorized_by_this_pilot':False,'source_human_gold':0,'native8_mapping':'UNKNOWN','current223_access':0,'model_forwards':0,'optimizer_updates':0,'PnP':0,'parent_ledger_unchanged':sha(ledger_path)==spec['bindings'][0]['sha256'],'outputs':[bind(O/x) for x in ['MODEL_INPUTS_ONLY.npz','GT_ONLY.npz','ROWS.jsonl','INTERVENTION_CERTIFICATES.json','INPUT_COLLISION_AUDIT.json','V1_RASTER_REPLAY_PARITY.json']]}
    savejson(O/'PILOT_RESULT.json',summary)
    print(json.dumps({'rows':len(rows),'variants':len(variants),'counterexamples':dict(source_examples),'result':bind(O/'PILOT_RESULT.json')}))

if __name__=='__main__':main()
