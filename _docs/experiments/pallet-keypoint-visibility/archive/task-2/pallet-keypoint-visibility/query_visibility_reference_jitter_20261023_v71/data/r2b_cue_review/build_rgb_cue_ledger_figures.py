"""Final actual public RGB cue ledger+figures; no training/admission."""
from pathlib import Path
import collections,csv,hashlib,json,math
from PIL import Image,ImageDraw,ImageFont
OUT=Path(__file__).resolve().parent;F='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';font=lambda n:ImageFont.truetype(F,n)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(p),'sha256':sha(p)}
def main():
    dest=OUT/'R2B_WOOD_CUE_PROPOSAL_LEDGER.json';assert not dest.exists()
    lp=OUT/'RGB_CUE_OBSERVATIONS_LOCK.json';assert sha(lp)=='fb4fb58982bb465822efde02b2b42867d08f550d03fc50b1f9ac8c548752e082'
    lock=json.loads(lp.read_text());regions=json.loads((OUT/'CANDIDATE_FRAME_VISUAL_ORDER.json').read_text());regions={r['record_id']:r for r in regions['records']}
    for b in lock['bindings']:assert sha(b['path'])==b['sha256']
    figdir=OUT/'REVIEW_CARDS';figdir.mkdir(exist_ok=True);rawdir=OUT/'RAW_NATIVE_CROPS';rawdir.mkdir(exist_ok=True)
    rows=[];csvrows=[];thumbs=[]
    for j,r in enumerate(lock['records']):
        fr=r['source_frame'];assert sha(fr['image'])==fr['image_sha256'];im=Image.open(fr['image']).convert('RGB');assert im.size==(fr['width'],fr['height'])
        reg=regions[r['record_id']]['detail_region'];raw=im.crop(reg);assert raw.size==(160,160)
        rawpath=rawdir/f'{j:02d}_{r["record_id"]}_NATIVE160_RGB.png';raw.save(rawpath)
        panel=Image.new('RGB',(1520,680),(24,26,29));d=ImageDraw.Draw(panel);xy=r['associated_cue_xy'];b=r['uncertainty_bbox_half_open_xyxy']
        d.text((12,7),f"{j} {r['record_id']}  {fr['stream']} slot{fr['sensor_slot']:02d} idx{fr['index_in_sorted_topic']}  {r['proposal_state']}",font=font(21),fill='white')
        d.text((12,40),f"fixed xy={xy}  visual uncertainty={b}  native8=UNKNOWN  AI/nonhuman/conditional only",font=font(17),fill=(170,235,170))
        context=im.copy();cd=ImageDraw.Draw(context)
        cd.rectangle(reg,outline=(255,200,40),width=3)
        if xy is not None:
            xx,yy=xy
            for ln in [(xx-12,yy,xx-4,yy),(xx+4,yy,xx+12,yy),(xx,yy-12,xx,yy-4),(xx,yy+4,xx,yy+12)]:cd.line(ln,fill=(0,255,255),width=3)
        context.thumbnail((480,360),Image.Resampling.LANCZOS);panel.paste(context,(12,120));d.text((12,89),'Actual source context; yellow=reviewROI',font=font(17),fill='white')
        left=raw.resize((480,480),Image.Resampling.NEAREST);right=left.copy();ld=ImageDraw.Draw(left)
        # Guides use true native pixel-center coordinates. Raw copy stays untouched.
        for k in range(0,160,16):
            q=(k+.5)*3;ld.line((q,0,q,480),fill=(100,100,100),width=1);ld.line((0,q,480,q),fill=(100,100,100),width=1)
        if b is not None:
            tr=lambda pt:((pt[0]-reg[0]+.5)*3,(pt[1]-reg[1]+.5)*3)
            aa=tr(b[:2]);bb=tr(b[2:]);ld.rectangle((*aa,*bb),outline=(255,0,220),width=2);xx,yy=tr(xy)
            for ln in [(xx-13,yy,xx-5,yy),(xx+5,yy,xx+13,yy),(xx,yy-13,xx,yy-5),(xx,yy+5,xx,yy+13)]:ld.line(ln,fill=(80,255,80),width=2)
        else:ld.text((9,9),'U: no point invented',font=font(20),fill=(255,190,80),stroke_width=2,stroke_fill='black')
        panel.paste(left,(516,120));panel.paste(right,(1024,120));d.text((516,89),'Marked native160 x3; fixed cue/uncertainty',font=font(17),fill='white');d.text((1024,89),'RAW native160 x3, unchanged pixels',font=font(17),fill='white')
        d.text((12,526),'One unresolved coarsefamily.',font=font(18),fill=(255,210,130));d.text((12,557),'No independent TRAIN/VAL split.',font=font(18),fill=(255,210,130));d.text((12,588),'No E/S or canonical8GT.',font=font(18),fill=(255,210,130))
        d.text((12,632),'Visible wood-cue proposal only. Region straddles physical edges; exact representative pixel need not be material. No mask/model-based relocation.',font=font(16),fill='white')
        fp=figdir/f'{j:02d}_{r["record_id"]}_REVIEW.png';panel.save(fp)
        integerbounds=None
        if b is not None:
            integerbounds=[math.floor(b[0]+.5),math.floor(b[1]+.5),math.ceil(b[2]+.5),math.ceil(b[3]+.5)]
            assert 0<=integerbounds[0]<integerbounds[2]<=im.width and 0<=integerbounds[1]<integerbounds[3]<=im.height
        if r['legacy_candidate'] is not None:assert r['associated_cue_xy']==r['legacy_candidate']['xy']
        row={**r,'image':fr['image'],'image_sha256':fr['image_sha256'],'raw_hw':[fr['height'],fr['width']],
            'target_instance_id_AI_local_only':'AI_TARGET:'+fr['stream']+':'+str(fr['sensor_slot']).zfill(2)+':LOW_WOOD_PALLET_LOCAL',
            'crossframe_physical_target_ID':'UNKNOWN','reference_xy':xy,'uncertainty_bbox':b,
            'uncertainty_native_cell_bounds_half_open':integerbounds,
            'uncertainty_native_cells':None if b is None else (integerbounds[2]-integerbounds[0])*(integerbounds[3]-integerbounds[1]),
            'initial_query_xy':None,'initial_crop_L':None,'query_generation_status':'NOT_GENERATED; future separate input-only jitter/crop protocol needed',
            'source_grouping_for_future_split':'NVIDIA_R2B_STORAGE_GALILEO_FAMILY_UNRESOLVED',
            'split':'UNASSIGNED_GROUP_HOLD','original_D_teacher_evidence_admitted':False,
            'diagnostic_native_crop_bounds_xyxy':reg,'raw_native_crop':bind(rawpath),'review_card':bind(fp)}
        rows.append(row);csvrows.append({'cue_refID':r['cue_refID'],'record_id':r['record_id'],'stream':fr['stream'],'sensor_slot':fr['sensor_slot'],'sorted_topic_index':fr['index_in_sorted_topic'],'image':fr['image'],'image_sha256':fr['image_sha256'],'state_proposal':r['proposal_state'],'reference_xy_json':json.dumps(xy),'uncertainty_bbox_json':json.dumps(b),'native8_mapping':'UNKNOWN','human_verified':False,'GT_admitted':False,'split':'UNASSIGNED_GROUP_HOLD'})
        thumb=Image.new('RGB',(1000,500),(24,26,29));td=ImageDraw.Draw(thumb)
        td.text((10,7),f"{j} {r['record_id']} {fr['stream']} slot{fr['sensor_slot']}  xy={xy}",font=font(19),fill='white')
        thumb.paste(left.resize((440,440),Image.Resampling.NEAREST),(20,45));thumb.paste(right.resize((440,440),Image.Resampling.NEAREST),(520,45));thumbs.append(thumb)
    contact=Image.new('RGB',(2000,2000),(18,20,23))
    for j,t in enumerate(thumbs):contact.paste(t,((j%2)*1000,(j//2)*500))
    cp=OUT/'EIGHT_RECORD_WOOD_CUE_RAW_MARKED_CONTACT_SHEET.png';contact.save(cp)
    result={'schema':'V71_PUBLIC_R2B_WOOD_CUE_PROPOSAL_LEDGER_V1','status':'ACTUAL_RGB_CANDIDATE_DATA_PREPARED_AI_CONDITIONAL_NOT_ADMITTED',
        'purpose':'Practical public real flat woodpallet2D cue evidence for the original crop/reference visibility goal, without pose improvement or a physicalmesh/PnP requirement.',
        'counts':lock['counts'],'actual_review':{'fixed_RGB_frames_viewed_in4overviewpages':48,'native_localization_regions_viewed':8,'existing3_rechecked':True,'new_additional_visible_cue_records':4,'new_U_null_records':1},
        'definition':lock['reference_definition'],'legacy_recheck_findings':'P0 wooddeck/endface is exposed; priorxy retained. P8 woodblock twofaces/lowerlayer are exposed; old4pxguess widened to layerededge uncertainty. P15 woodtop/endboard cue is exposed but blurred/rounded; retainxy with widerregion rather than pretending4pxprecision. OldV67files unchanged.',
        'source_evidence_tier':'AI_VISUAL_NONHUMAN_CONDITIONAL_TEACHER_PROPOSAL; human gold0, official point labels0, GTadmitted0, native8mappingUNKNOWN',
        'candidate_data_is_not_training':'No images composed, no qjitter/initialcoordinates manufactured, no model inputs constructed and no teacherGT/fit admitted. Root can review fixed candidate data under a separate protocol.',
        'family_and_split':'All48frames and these8records stay under NVIDIA_R2B_STORAGE_GALILEO_FAMILY_UNRESOLVED. Adjacentframes, sensors and likely repeatedphysicalpallets are near-duplicate observations. No independent TRAIN/VAL/sourcefamily claim; splitunassigned.',
        'E_S_policy':'No E or S labels generated. Movingcamera/frame similarity does not certify identicalreference correspondence or same-pixel before/after visibility.',
        'limitations':['Only3old+4new resolvedAIreferences; no claim of7independentphysicalcorners/families.','Visiblelocalwood cue regions do not establish a complete pallet cuboid/canonical8identity.','Material/edge association comes from actual RGB observation only; no ownershipmask/depthcertificate exists here.','U Galileo1 attempted upperdeck/carton seam remainsnull; later favorableframes were not substituted.','No imagecontent/modelscore optimization of crop/L/threshold/radius; per-cue uncertainty derivesfromvisualblur/bevel.'],
        'read_scope':{'existing_public_RGB_unique_images':48,'new_source_download':0,'SAM_mask_depth_feature_weight_or_modelscore_reads':0,'current223_RGB_GT_scores':0,'new_forward_fit_PnP':0,'prior_snapshots_modified':False},
        'bindings':[bind(p) for p in [OUT/'RGB_REVIEW_PROTOCOL.json',lp,OUT/'CANDIDATE_FRAME_VISUAL_ORDER.json',OUT/'FIXED48_RGB_OVERVIEW_OUTPUTS.json',Path(__file__)]],
        'contact_sheet':bind(cp),'records':rows}
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    jsonl=OUT/'R2B_WOOD_CUE_PROPOSAL_LEDGER.jsonl';assert not jsonl.exists();jsonl.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    with (OUT/'R2B_WOOD_CUE_PROPOSAL_LEDGER.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    files=[p for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='CUE_REVIEW_SEAL.json']
    seal={'schema':'V71_PUBLIC_R2B_CUE_REVIEW_SEAL_V1','status':result['status'],'ledger_sha256':sha(dest),'counts':result['counts'],'allcoords_bboxes_frozen_before_downstream_use':True,'one_unresolved_coarsefamily_no_independent_split':True,'teacher_GT_fit_current_forward_PnP_admission':False,'bindings':[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in files]}
    sp=OUT/'CUE_REVIEW_SEAL.json';assert not sp.exists();sp.write_text(json.dumps(seal,indent=2)+'\n')
    print(json.dumps({'ledger':str(dest),'ledger_sha256':sha(dest),'seal_sha256':sha(sp),'contact_sha256':sha(cp),'counts':result['counts']}))
if __name__=='__main__':main()
