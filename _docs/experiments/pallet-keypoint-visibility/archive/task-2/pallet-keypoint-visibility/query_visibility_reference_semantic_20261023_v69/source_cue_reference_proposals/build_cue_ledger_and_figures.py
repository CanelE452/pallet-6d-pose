"""Geometry+figure proposal ledger. Existing review rasters only, no models."""
from pathlib import Path
import collections,csv,hashlib,importlib.util,json,math,sys
from PIL import Image,ImageDraw,ImageFont
OUT=Path(__file__).resolve().parent;V69=OUT.parent
F='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font=lambda n:ImageFont.truetype(F,n)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def native_from_sheet(path,k):
    x=8+(k%4)*304;y=45+(k//4)*345+28
    im=Image.open(path).convert('RGB').crop((x,y,x+288,y+288))
    native=im.resize((96,96),Image.Resampling.NEAREST)
    assert native.resize((288,288),Image.Resampling.NEAREST).tobytes()==im.tobytes()
    return native
def recover_query_crop(candidate,frame):
    q=candidate['original_query_xy'];half=candidate['original_side_L']/2
    b=[q[0]-half,q[1]-half,q[0]+half,q[1]+half]
    x0,y0=[math.floor(x+.5) for x in b[:2]];x1,y1=[math.ceil(x+.5) for x in b[2:]]
    im=Image.new('RGB',(x1-x0,y1-y0),(70,0,70));seen=bytearray(im.width*im.height)
    for k,ar in enumerate(frame['observations8']):
        patch=native_from_sheet(frame['raw_crop_sheet']['path'],k)
        cx,cy=[math.floor(x+.5) for x in ar['reference_xy']];ox,oy=cx-48,cy-48
        l,t,r,bot=max(x0,ox),max(y0,oy),min(x1,ox+96),min(y1,oy+96)
        h,w=candidate['raw_hw'];l,t,r,bot=max(l,0),max(t,0),min(r,w),min(bot,h)
        if l>=r or t>=bot:continue
        for yy in range(t,bot):
            for xx in range(l,r):
                dest=(xx-x0,yy-y0);src=(xx-ox,yy-oy);idx=dest[1]*im.width+dest[0]
                pix=patch.getpixel(src)
                if seen[idx]:assert im.getpixel(dest)==pix
                im.putpixel(dest,pix);seen[idx]=1
    return im,(x0,y0),b,{'native_cells':len(seen),'covered_cells_from_existing_review':sum(seen),'unavailable_cells':len(seen)-sum(seen)}
def independent_availability(xy,b,q,L,hw):
    crop=[q[0]-L/2,q[1]-L/2,q[0]+L/2,q[1]+L/2];h,w=hw
    obs=[max(crop[0],-.5),max(crop[1],-.5),min(crop[2],w-.5),min(crop[3],h-.5)]
    iw=max(0,min(b[2],obs[2])-max(b[0],obs[0]));ih=max(0,min(b[3],obs[3])-max(b[1],obs[1]))
    state='FULL' if obs[0]<=b[0] and obs[1]<=b[1] and obs[2]>=b[2] and obs[3]>=b[3] else ('PARTIAL' if iw*ih>0 else 'MISS')
    return state,crop,obs,iw*ih
def main():
    target=OUT/'AI_OPERATIONAL_CUE_REFERENCE_PROPOSAL_LEDGER.json';assert not target.exists()
    selection=json.loads((OUT/'CUE_PROPOSAL_SELECTION_LOCK.json').read_text())
    observations=json.loads((OUT/'VISUAL_LOCALIZATION_OBSERVATIONS_LOCK.json').read_text())
    review=json.loads((V69/'source_anchor_review/EIGHT_FRAME_REFERENCE_VISUAL_REVIEW.json').read_text());frames={x['frame_id']:x for x in review['frames']}
    spec=importlib.util.spec_from_file_location('v69_math_cue_audit',V69/'reference_visibility_contract.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    assert sha(OUT/'CUE_PROPOSAL_SELECTION_LOCK.json')=='93b766611576286196442f4262677843af9b2df4cb340ed2f6d9700cf922c81c'
    assert sha(OUT/'VISUAL_LOCALIZATION_OBSERVATIONS_LOCK.json')=='b458d8cec869d64589de110b98865d8ebf198709a064243b4561f16c05bcb8b4'
    dest=OUT/'CUE_REVIEW_FIGURES';dest.mkdir(exist_ok=True)
    rows=[];csvrows=[];thumbs=[]
    for c,loc in zip(selection['candidates'],observations['observations']):
        assert c['candidate_point_ID']==loc['candidate_point_ID'];f=frames[c['frame_id']];k=c['original_native_index'];q=c['original_query_xy'];r=c['original_renderer_r_xy'];L=c['original_side_L']
        native=native_from_sheet(c['raw96_sheet']['path'],k);cx,cy=[math.floor(v+.5) for v in r];origin=(cx-48,cy-48)
        xy=loc['associated_cue_xy'];b=loc['uncertainty_bbox_half_open_xyxy']
        if xy is None:
            av={'availability':'INVALID','reason':'U_UNRESOLVED_CUE_NO_COORDINATE_OR_REGION_INVENTED','crop_bounds':[q[0]-L/2,q[1]-L/2,q[0]+L/2,q[1]+L/2]};state='UNRESOLVED'
        else:
            # original_state U prevents accidental original-D admission. Only pure geometry is used.
            ref=module.ReferenceGT(c['frame_id'],loc['NEW_cue_refID'],tuple(xy),tuple(b),original_state='U',source_name='SYNTH_AI_CUE_PROPOSAL')
            av=module.crop_availability(ref,q,L,c['raw_hw']);ind=independent_availability(xy,b,q,L,c['raw_hw'])
            assert av['availability']==ind[0] and av['crop_bounds']==ind[1] and av['observed_crop_bounds']==ind[2]
            assert av['observed_uncertainty_intersection_area']==ind[3]
            state=av['availability']
        qpatch,qorigin,qbounds,qcoverage=recover_query_crop(c,f)
        # Existing raw96 views need not cover every original-q review pixel.
        # Missing review-only pixels stay purple and explicitly counted; do not
        # reopen original RGB or invent pixels. This does not change geometry.
        im8=native.resize((768,768),Image.Resampling.NEAREST)
        panel=Image.new('RGB',(1680,1380),(24,26,29));panel.paste(im8,(70,90));panel.paste(im8,(890,90));d=ImageDraw.Draw(panel)
        d.text((12,8),f"{c['selection_rank']} {c['candidate_point_ID']}  {loc['proposal_status']}  crop:{state}",font=font(20),fill='white')
        d.text((12,42),'Green box/open cross=NEW cue proposal; cyan open cross=OLD renderer r. Right raw RGB unchanged.',font=font(18),fill=(220,220,220))
        # Coordinate guides at true native pixel centers. The right raster is unmarked.
        for i in range(96):
            if i%4:continue
            xx=70+(i+.5)*8;yy=90+(i+.5)*8;col=(80,80,80) if i%8 else (135,135,135)
            d.line((xx,90,xx,858),fill=col,width=1);d.line((70,yy,838,yy),fill=col,width=1)
            if i%8==0:d.text((xx-12,70),str(origin[0]+i),font=font(12),fill='white');d.text((25,yy-5),str(origin[1]+i),font=font(12),fill='white')
        transform=lambda p:(70+(p[0]-origin[0]+.5)*8,90+(p[1]-origin[1]+.5)*8)
        def cross(point,col):
            xx,yy=transform(point)
            for line in [(xx-25,yy,xx-11,yy),(xx+11,yy,xx+25,yy),(xx,yy-25,xx,yy-11),(xx,yy+11,xx,yy+25)]:d.line(line,fill=col,width=2)
        cross(r,(0,255,255))
        if b is not None:
            aa=transform(b[:2]);bb=transform(b[2:]);d.rectangle((*aa,*bb),outline=(80,255,80),width=3);cross(xy,(80,255,80))
        else:d.text((90,110),'U: NO POINT/UNCERTAINTY BOX INVENTED',font=font(20),fill=(255,190,80),stroke_width=1,stroke_fill='black')
        d.text((12,875),'Upper guides:4-native-pixel center lines; labels8px. AI observational proposals only, not human/gold.',font=font(17),fill='white')
        # Lower left is a q-centered original-L review crop from already saved raw96 union.
        qp=qpatch.resize((384,384),Image.Resampling.NEAREST);panel.paste(qp,(26,945))
        d.text((26,918),'Original q/L review; purple=unrecorded',font=font(16),fill=(255,150,235))
        d.text((450,930),f"old renderer r={r}; old q={q}; L={L:.6f}",font=font(16),fill='white')
        d.text((450,963),f"NEW associated cue xy={xy}; uncertainty bbox={b}",font=font(17),fill=(80,255,80))
        d.text((450,998),f"Crop availability of ENTIRE cue uncertainty: {state}",font=font(22),fill=(255,220,100))
        d.text((450,1038),'r/q/L/native8 are preserved lineage. New cue ID is separate from old native8 GT.',font=font(17),fill='white')
        d.text((450,1071),'FULL/PARTIAL/MISS is geometry, never E/O. All training targets stay U/unadmitted.',font=font(17),fill='white')
        d.text((450,1104),'No new composites, jitter, masks, model, fit, PnP or current data.',font=font(17),fill='white')
        d.text((450,1137),f"Source pixel reconstruction from existing raw96 sheets: {qcoverage['covered_cells_from_existing_review']}/{qcoverage['native_cells']} cells",font=font(17),fill='white')
        d.text((450,1203),f"Missing review-only cells: {qcoverage['unavailable_cells']}; purple cells are NOT source RGB or model input.",font=font(17),fill=(255,150,235))
        if xy is not None:
            d.text((450,1170),'Integer cue xy represents a visual corner/edge region; uncertainty includes blur/bevel.',font=font(17),fill=(210,210,210))
        else:d.text((450,1170),'Prior coarse D_candidate remains recorded; finer inspection did not certify a location.',font=font(17),fill=(255,190,80))
        path=dest/f'{c["selection_rank"]:02d}_{c["candidate_point_ID"].replace(":","_k")}_CUE_REVIEW.png';panel.save(path)
        proposal={**c,**loc,'source_observed_reference_state_proposal':'AI_VISIBLE_CUE_D_PROPOSAL' if xy is not None else 'U',
            'crop_availability_original_q_L':av,'availability_category_for_summary':state,
            'proposed_training_target':'U_UNADMITTED','training_authorized':False,
            'renderer_r_to_NEW_cue_displacement_xy':None if xy is None else [xy[0]-r[0],xy[1]-r[1]],
            'new_cue_is_not_old_renderer_coordinate_correction':True,'native8_semantic_gold_mapping_admitted':False,
            'q_crop_review_pixel_source':'Existing8 native96 raw crop union; no original source RGB reopened; unavailable review-only pixels explicitly purple, never model inputs',
            'q_crop_review_native_bounds':[qorigin[0],qorigin[1],qorigin[0]+qpatch.width,qorigin[1]+qpatch.height],
            'q_crop_review_reconstruction':qcoverage,'figure':{'path':str(path),'sha256':sha(path)}}
        rows.append(proposal);csvrows.append({'candidate_point_ID':c['candidate_point_ID'],'frame_id':c['frame_id'],'native_index_lineage':k,'NEW_cue_refID':loc['NEW_cue_refID'],'proposal_status':loc['proposal_status'],'renderer_r_json':json.dumps(r),'original_q_json':json.dumps(q),'original_L':L,'associated_cue_xy_json':json.dumps(xy),'uncertainty_bbox_json':json.dumps(b),'original_crop_availability':state,'human_verified':False,'training_admitted':False})
        thumb=Image.new('RGB',(840,500),(24,26,29));td=ImageDraw.Draw(thumb)
        td.text((10,7),f"{c['selection_rank']} {c['candidate_point_ID']}  {state}",font=font(20),fill='white')
        left=panel.crop((70,90,838,858)).resize((384,384),Image.Resampling.NEAREST);right=native.resize((384,384),Image.Resampling.NEAREST)
        thumb.paste(left,(12,45));thumb.paste(right,(428,45));td.text((12,445),f"NEW cue {xy} / AI only / root review pending",font=font(17),fill=(90,255,90) if xy else (255,190,80));thumbs.append(thumb)
    contact=Image.new('RGB',(1680,2000),(18,20,23))
    for j,t in enumerate(thumbs):contact.paste(t,((j%2)*840,(j//2)*500))
    contactpath=OUT/'FIXED8_CUE_REFERENCE_PROPOSAL_CONTACT_SHEET.png';contact.save(contactpath)
    result={'schema':'AI_OPERATIONAL_CUE_REFERENCE_PROPOSAL_LEDGER_V1','status':'CONDITIONAL_AI_PROPOSALS_ROOT_REVIEW_PENDING_NO_GT_ADMISSION',
        'source':'SYNTH TRAIN only; same deterministic8 candidate IDs among prior D30,5 frames',
        'observational_definition':'A NEW operational2D reference is a visually localized associated pallet outer corner/edge turning region. Its point is an integer representative inside an explicit visual-uncertainty rectangle. Renderer cuboid r, old q/L/native8 remain separate immutable lineage. No arbitrary nearest-foreground reassignment.',
        'counts':{'candidate_points':8,'distinct_frames':5,'localized_AI_visible_cue_proposals':6,'U_no_point_invented':2,'availability':dict(collections.Counter(r['availability_category_for_summary'] for r in rows)),'human_verified':0,'gold':0,'training_admitted':0},
        'future_controlled_truth_plan_only':'After independent/root admission and separate composition freeze, hold each NEW cue xy/region/state fixed under q jitter. Only a distinct certified opaque front layer covering every uncertainty cell may create conditional E. Natural E/S not admitted. No composition or jitter generated here.',
        'coverage_semantics':'V69 bound draft crop_availability with original q/L unchanged, independently recomputed rectangle intersection and parity checked. FULL requires whole cue region; PARTIAL/MISS imply evidence/location limits, never semantic E/O. U candidates without coordinates have UNRESOLVED summary/INVALID geometry.',
        'review_only_preview_limitation':'One original-q review raster has84 cells outside union of prior saved raw96 sheets. They remain purple and counted, never fabricated or used as inputs. Every proposed cue uncertainty region is present in its source reference raw96; crop availability is pure original q/L geometry.',
        'reference_identity_limit':'Named old native index is conditional lineage association, not a newly admitted canonical8 landmark dictionary or model token. This ledger does not resolve general RGB/q/box-only hidden-reference identifiability.',
        'read_scope':{'existing_review_RGB_rasters_only':True,'new_original_source_RGB_decodes':0,'source_VAL_RGB':0,'GT_masks_depth_features_weights_or_performance':0,'current_data':0,'new_model_forward_fit_render_PnP':0,'RGB_composites_or_actual_q_jitters_generated':0,'prior_artifacts_modified':False},
        'native_grid_phase_note':'Pre-localization grid labels identify displayed pixel-block indices. Final review guides are explicitly aligned to native pixel centers. Half-integer uncertainty bounds include complete native cells; no point/bbox was changed after availability computation.',
        'bindings':[{'path':str(p),'sha256':sha(p)} for p in [OUT/'CUE_PROPOSAL_SELECTION_LOCK.json',OUT/'VISUAL_LOCALIZATION_OBSERVATIONS_LOCK.json',OUT/'LOCALIZATION_GRID_OUTPUTS.json',V69/'DRAFT_PROTOCOL.json',V69/'reference_visibility_contract.py',Path(__file__)]],
        'contact_sheet':{'path':str(contactpath),'sha256':sha(contactpath)},'proposals':rows}
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (OUT/'CUE_REFERENCE_PROPOSAL_LEDGER.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    files=[p for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='CUE_PROPOSAL_SEAL.json']
    seal={'schema':'CUE_REFERENCE_PROPOSAL_SEAL_V1','status':result['status'],'ledger_sha256':sha(target),'counts':result['counts'],'all_original_r_q_L_native_index_preserved':True,'geometry_independent_parity':True,'source_training_current_inference_authorized':False,'bindings':[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in files]}
    (OUT/'CUE_PROPOSAL_SEAL.json').write_text(json.dumps(seal,indent=2)+'\n')
    print(json.dumps({'ledger':str(target),'ledger_sha256':sha(target),'seal_sha256':sha(OUT/'CUE_PROPOSAL_SEAL.json'),'counts':result['counts'],'coverage':[(r['candidate_point_ID'],r['availability_category_for_summary']) for r in rows]},ensure_ascii=False))
if __name__=='__main__':main()
