"""Fixed6 cue ownership sanity only. No point/radius/model changes."""
from pathlib import Path
import argparse,collections,csv,hashlib,json,math
import numpy as np
from PIL import Image,ImageDraw,ImageFont
OUT=Path(__file__).resolve().parent;V69=OUT.parent;BASE=V69.parent
AUDIT=BASE.parent/'target_presence_classifier_20261022_v61/MASK_CONTRACT_AUDIT.csv'
F='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font=lambda n:ImageFont.truetype(F,n)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(p),'sha256':sha(p)}
def write(p,a):
    assert not p.exists(),p
    p.write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n')
def metadata():
    ledger=OUT/'AI_OPERATIONAL_CUE_REFERENCE_PROPOSAL_LEDGER.json'
    a=json.loads(ledger.read_text());assert sha(ledger)=='c9758ee265882753616f43ebbbc9a2da7c23bbca6b2ea78e36e54032af7e2d0b'
    with AUDIT.open() as f:paths={r['id']:r for r in csv.DictReader(f)}
    for r in a['proposals']:
        if r['associated_cue_xy'] is not None:
            old=paths[r['frame_id']];assert old['split']=='train' and old['annotation_sha256']==r['original_annotation_sha256']
    return a,paths
def prepare():
    a,paths=metadata();resolved=[r for r in a['proposals'] if r['associated_cue_xy'] is not None]
    rows=[{'candidate_point_ID':r['candidate_point_ID'],'frame_id':r['frame_id'],'associated_cue_xy':r['associated_cue_xy'],
           'uncertainty_bbox':r['uncertainty_bbox_half_open_xyxy'],'raw_hw':r['raw_hw'],
           'expected_visible_path':paths[r['frame_id']]['visible_path'],'expected_visible_sha256':paths[r['frame_id']]['visible_sha256'],
           'expected_amodal_path':paths[r['frame_id']]['amodal_path'],'expected_amodal_sha256':paths[r['frame_id']]['amodal_sha256']} for r in resolved]
    p={'schema':'FROZEN_CUE_OWNERSHIP_SANITY_PROTOCOL_V1','scope':'Root authorized all6 existing localized AI cue proposals; original8 selection and6 xy/bbox frozen. No relocation, uncertainty change, candidate replacement, semantic D certification or admission.',
       'target_records':rows,'two_U':'Keep null associated coordinate/bbox, no fabricated statistics. Existing MISS stays MISS, never E/O.',
       'threshold':'M0=mask_amodal>127; M4=mask_visible>127, native dtype/shape preserved. No resize/dilation/rethreshold for counts.',
       'uncertainty_cells':'Positive-area native cell intersection: floor(lo+.5),ceil(hi+.5), half-open integer bounds; keep outside count if any.',
       'boundary':'Inner boundary B(mask)=mask AND NOT(all8neighbors true), outside-image neighbors false. Outer boundary=(~mask) AND(any8neighbor true). Counts on fixed cue cells. Also count visible M0 outline M4 & B(M0) and internal-only visibility boundary B(M4) & ~B(M0).',
       'exact_cue_pixel':'Separate nearest native floor(xy+.5), not a semantic D target. Bbox can straddle a true visible edge while its representative cue point is outside material.',
       'predeclared_sanity_flags':'No M4-positive cue cells => HOLD_NO_VISIBLE_TARGET_OWNERSHIP. M4-positive but no M4 & M0-inner-boundary cells => HOLD_NO_VISIBLE_SHAPE_EDGE. Otherwise VISIBLE_TARGET_OUTLINE_IN_FIXED_BOX_REVIEW_ONLY; never automatic semantic D.',
       'lineage':'Verify same RGBfile SHA, annotation SHA/projected_cuboid native index/sourceasset/convention, original input q/L and source mask audit SHAs. No source mask positivity used to relocate or shrink cue region.',
       'outputs':'FROZEN_CUE_OWNERSHIP_SANITY.json + overlay contact/per-casePNG; previous ledger/seal unchanged',
       'execution_scope':{'TRAIN_unique_resolved_frames':len({r['frame_id'] for r in resolved}),'source_mask_arrays_allowed':8,'original_RGB_file_hash_allowed':True,'original_RGB_decode_needed':False,'current_model_feature_weight_fit_forward_PnP':False},
       'bindings':[bind(p) for p in [OUT/'AI_OPERATIONAL_CUE_REFERENCE_PROPOSAL_LEDGER.json',OUT/'CUE_PROPOSAL_SEAL.json',OUT/'VISUAL_LOCALIZATION_OBSERVATIONS_LOCK.json',AUDIT,BASE/'query_visibility_amodal_global_20261022_v65/build_global_cache.py',Path(__file__)]]}
    dest=OUT/'FROZEN_CUE_OWNERSHIP_SANITY_PROTOCOL_V2.json';write(dest,p);print(json.dumps({'path':str(dest),'sha256':sha(dest)}))
def bounds(mask):
    p=np.pad(mask,1,constant_values=False);neighbors=[p[1+dy:1+dy+mask.shape[0],1+dx:1+dx+mask.shape[1]] for dy in [-1,0,1] for dx in [-1,0,1] if dx or dy]
    inner=mask & ~np.logical_and.reduce(neighbors);outer=~mask & np.logical_or.reduce(neighbors)
    return inner,outer
def native_raw(r):
    p=Path(r['raw96_sheet']['path']);assert sha(p)==r['raw96_sheet']['sha256'];k=r['original_native_index'];x=8+(k%4)*304;y=45+(k//4)*345+28
    crop=Image.open(p).convert('RGB').crop((x,y,x+288,y+288)).resize((96,96),Image.Resampling.NEAREST)
    cx,cy=[math.floor(v+.5) for v in r['original_renderer_r_xy']]
    return crop,(cx-48,cy-48)
def run():
    protocol=OUT/'FROZEN_CUE_OWNERSHIP_SANITY_PROTOCOL_V2.json';p=json.loads(protocol.read_text())
    for b in p['bindings']:assert sha(b['path'])==b['sha256'],b['path']
    a,paths=metadata();cache={};outputs=[];figures=[];thumbs=[];dest=OUT/'OWNERSHIP_SANITY_OVERLAYS';dest.mkdir(exist_ok=True)
    for r in a['proposals']:
        raw,origin=native_raw(r);panel=Image.new('RGB',(1240,550),(24,26,29));d=ImageDraw.Draw(panel)
        d.text((12,7),r['candidate_point_ID']+' / FROZEN xy+bbox / ownership sanity only',font=font(18),fill='white')
        panel.paste(raw.resize((384,384),Image.Resampling.NEAREST),(20,70))
        d.text((20,42),'Raw RGB untouched',font=font(17),fill='white')
        row={'candidate_point_ID':r['candidate_point_ID'],'NEW_cue_refID':r['NEW_cue_refID'],'original_renderer_r_xy':r['original_renderer_r_xy'],'original_query_xy':r['original_query_xy'],'original_L':r['original_side_L'],'original_native_index':r['original_native_index'],'associated_cue_xy':r['associated_cue_xy'],'fixed_uncertainty_bbox':r['uncertainty_bbox_half_open_xyxy'],'original_crop_availability':r['availability_category_for_summary'],'human_verified':False,'semantic_D_certified':False,'training_admitted':False}
        if r['associated_cue_xy'] is None:
            row.update(sanity_status='U_UNRESOLVED_NO_POINT_NO_REGION',mask_statistics=None,exact_cue_support=None,source_mask_binding=None)
            d.text((430,100),'U: coordinate/bbox remain null',font=font(22),fill=(255,190,80));d.text((430,145),'No mask-derived replacement cue.',font=font(20),fill='white')
        else:
            fid=r['frame_id']
            if fid not in cache:
                md=paths[fid];assert sha(r['original_image'])==r['original_image_sha256'];assert sha(r['original_annotation'])==r['original_annotation_sha256']
                an=json.loads(Path(r['original_annotation']).read_text());ob=an['objects'][0]
                assert ob['source_asset']==r['source_asset'] and ob['keypoint_convention']=='camera_dynamic_0123_v4'
                assert ob['projected_cuboid'][r['original_native_index']]==r['original_renderer_r_xy']
                arrays={};mb={}
                for short,name in [('M0','amodal'),('M4','visible')]:
                    path=Path(md[name+'_path']);assert sha(path)==md[name+'_sha256'];im=Image.open(path);arr=np.asarray(im)
                    assert list(arr.shape)==r['raw_hw'] and arr.ndim==2
                    val,cnt=np.unique(arr,return_counts=True);assert val.tolist()==json.loads(md[name+'_values']) and cnt.tolist()==json.loads(md[name+'_counts'])
                    arrays[short]=arr>127;mb[short]={'path':str(path),'sha256':sha(path),'PIL_mode':im.mode,'dtype':str(arr.dtype),'shape':list(arr.shape),'unique_values':val.tolist(),'native_area_gt127':int((arr>127).sum())}
                assert not (arrays['M4'] & ~arrays['M0']).any()
                cache[fid]=(arrays,mb,{'RGB_SHA_match':True,'annotation_SHA_match':True,'native_r_and_asset_match':True,'mask_audit_annotation_SHA_match':md['annotation_sha256']==r['original_annotation_sha256']})
            ar,mb,lineage=cache[fid];m0=ar['M0'];m4=ar['M4'];h,w=m0.shape
            # Recheck each native index even when sharing a frame.
            ob=json.loads(Path(r['original_annotation']).read_text())['objects'][0];assert ob['projected_cuboid'][r['original_native_index']]==r['original_renderer_r_xy']
            b=r['uncertainty_bbox_half_open_xyxy'];x0,y0=[math.floor(v+.5) for v in b[:2]];x1,y1=[math.ceil(v+.5) for v in b[2:]]
            assert 0<=x0<x1<=w and 0<=y0<y1<=h
            i0,o0=bounds(m0);i4,o4=bounds(m4);sl=(slice(y0,y1),slice(x0,x1))
            stat={'region_bounds_half_open_integer':[x0,y0,x1,y1],'region_cells':(x1-x0)*(y1-y0),'outside_image_cells':0,'M0_gt127_cells':int(m0[sl].sum()),'M4_gt127_cells':int(m4[sl].sum()),'M0_and_not_M4_cells':int((m0&~m4)[sl].sum()),'outside_M0_cells':int((~m0)[sl].sum()),'M0_inner_boundary_cells':int(i0[sl].sum()),'M0_outer_boundary_cells':int(o0[sl].sum()),'M4_inner_boundary_cells':int(i4[sl].sum()),'M4_outer_boundary_cells':int(o4[sl].sum()),'visible_M0_inner_boundary_cells':int((m4&i0)[sl].sum()),'M4_boundary_not_M0_inner_boundary_cells':int((i4&~i0)[sl].sum())}
            xy=r['associated_cue_xy'];x,y=[math.floor(v+.5) for v in xy]
            exact={'native_xy':[x,y],'M0_gt127':bool(m0[y,x]),'M4_gt127':bool(m4[y,x]),'M0_inner_boundary':bool(i0[y,x]),'M4_inner_boundary':bool(i4[y,x]),'semantic_D_inferred':False}
            status='HOLD_NO_VISIBLE_TARGET_OWNERSHIP' if stat['M4_gt127_cells']==0 else ('HOLD_NO_VISIBLE_SHAPE_EDGE' if stat['visible_M0_inner_boundary_cells']==0 else 'VISIBLE_TARGET_OUTLINE_IN_FIXED_BOX_REVIEW_ONLY')
            row.update(sanity_status=status,mask_statistics=stat,exact_cue_support=exact,source_mask_binding=mb,lineage=lineage)
            overlay=np.array(raw).copy();binary=np.zeros((96,96,3),np.uint8)
            ox,oy=origin
            for yy in range(96):
                for xx in range(96):
                    gx,gy=ox+xx,oy+yy
                    if not(0<=gx<w and 0<=gy<h):continue
                    if m0[gy,gx]:binary[yy,xx]=(100,65,0)
                    if m4[gy,gx]:binary[yy,xx]=(30,105,40)
                    if i0[gy,gx]:overlay[yy,xx]=(255,155,20);binary[yy,xx]=(255,155,20)
                    if i4[gy,gx]:overlay[yy,xx]=(0,255,255);binary[yy,xx]=(0,255,255)
            panel.paste(Image.fromarray(overlay).resize((384,384),Image.Resampling.NEAREST),(430,70));panel.paste(Image.fromarray(binary).resize((384,384),Image.Resampling.NEAREST),(840,70))
            d.text((430,42),'Native boundary overlay + fixed cue box',font=font(16),fill='white');d.text((840,42),'M0 orange / M4 cyan boundary',font=font(16),fill='white')
            for off in [430,840]:
                trans=lambda pt:(off+(pt[0]-ox+.5)*4,70+(pt[1]-oy+.5)*4)
                aa=trans(b[:2]);bb=trans(b[2:]);d.rectangle((*aa,*bb),outline=(255,0,255),width=2)
                cx,cy=trans(xy)
                for ln in [(cx-12,cy,cx-5,cy),(cx+5,cy,cx+12,cy),(cx,cy-12,cx,cy-5),(cx,cy+5,cx,cy+12)]:d.line(ln,fill=(255,0,255),width=2)
            d.text((18,470),f"Fixedbox cells={stat['region_cells']} M4={stat['M4_gt127_cells']} M0={stat['M0_gt127_cells']} visibleM0edge={stat['visible_M0_inner_boundary_cells']} | exact M4={exact['M4_gt127']}",font=font(17),fill='white')
            d.text((18,500),status+' / originalcrop:'+r['availability_category_for_summary']+' / no point moved / not semantic D',font=font(16),fill=(255,200,100))
        path=dest/f'{r["selection_rank"]:02d}_{r["candidate_point_ID"].replace(":","_k")}_OWNERSHIP.png';panel.save(path);row['overlay']={'path':str(path),'sha256':sha(path)};outputs.append(row)
        thumb=panel.copy();thumb.thumbnail((992,440));thumbs.append(thumb)
    contact=Image.new('RGB',(1984,1760),(18,20,23))
    for j,t in enumerate(thumbs):contact.paste(t,((j%2)*992,(j//2)*440))
    cp=OUT/'FROZEN_CUE_OWNERSHIP_SANITY_OVERLAY.png';contact.save(cp)
    result={'schema':'FROZEN_CUE_OWNERSHIP_SANITY_V1','status':'COMPLETED_SANITY_ONLY_NO_D_ADMISSION','all6_frozen_resolved_cues_checked':True,'all8_original_proposals_preserved':True,
        'counts':{'resolved':6,'U_null':2,'unique_native_mask_frames':len(cache),'native_masks_decoded':len(cache)*2,'sanity_flags':dict(collections.Counter(r['sanity_status'] for r in outputs))},
        'interpretation':'Mask boundary/ownership consistency can reject/hold a fixed visual cue association but does not certify semantic D, native8 physical identity or human truth. Exact cue material support is separate from an uncertainty region straddling a visible edge. M0 is target-only rendered silhouette, not proof of a solid cuboid corner; M4 is final visible support.',
        'no_selection_or_relocation':'ALL6 selected localized proposals tested, same xy/bbox as sealed. No mask-driven coordinate/radius adjustment or replacement. TwoU remain null; MISS remains MISS, never E/O.',
        'original_input_lineage':'Original q/L/box inherited verbatim from bound input JSONL selection lock and previous ledger. This sanity opens only masks/RGBfileSHA/annotation for already selected TRAIN candidates; GT never becomes model input.',
        'read_scope':{'TRAIN_RGB_hashed_not_decoded':len(cache),'existing_review_rasters_for_overlay_only':True,'TRAIN_annotation_JSON':len(cache),'source_native_masks':len(cache)*2,'VAL_or_current_arrays_or_scores':0,'model_features_weights_forward_training_PnP':0,'composites_or_jitter':0},
        'parent_ledger_sha256_unchanged':sha(OUT/'AI_OPERATIONAL_CUE_REFERENCE_PROPOSAL_LEDGER.json'),'parent_seal_sha256_unchanged':sha(OUT/'CUE_PROPOSAL_SEAL.json'),
        'protocol':bind(protocol),'audit_code':bind(Path(__file__)),'overlay_contact':bind(cp),'records':outputs}
    for b in p['bindings']:assert sha(b['path'])==b['sha256']
    destjson=OUT/'FROZEN_CUE_OWNERSHIP_SANITY.json';write(destjson,result)
    print(json.dumps({'report':str(destjson),'sha256':sha(destjson),'counts':result['counts'],'records':[{k:r.get(k) for k in ['candidate_point_ID','sanity_status','mask_statistics','exact_cue_support']} for r in outputs]},ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.prepare:prepare()
    elif a.run:run()
    else:p.error('choose --prepare or --run')
