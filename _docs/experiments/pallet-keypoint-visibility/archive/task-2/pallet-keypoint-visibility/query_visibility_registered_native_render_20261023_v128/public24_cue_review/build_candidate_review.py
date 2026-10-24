"""Public, image-only manual AUX cue proposal rendering; no model or training admission."""
from pathlib import Path
import json, hashlib, textwrap
from PIL import Image,ImageDraw,ImageFont
H=Path(__file__).resolve().parent;P=H.parent.parent
S=P/'query_visibility_spatial_stencil_control_20261023_v127/public_diversity_intake'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def ash(im):return hashlib.sha256(im.tobytes()).hexdigest()
def put(p,o):
 with Path(p).open('x')as f:json.dump(o,f,ensure_ascii=False,indent=2);f.write('\n')
receipt=json.loads((S/'EXTRACTION_AND_RENDER_COMPLETE.json').read_text());source={r['selection_index']:r for r in receipt['rows']}
rulers=json.loads((S/'coordinate_rulers/COORDINATE_RULER_MANIFEST_V2.json').read_text());rm={r['selection_index']:r for r in rulers['rows']}
# These are actual visual proposals after viewing all nine locked rulers and native detail displays.
props={
4:([817,813],6,'Outer deck/near support corner','The long pale front rail and short right-return rail meet above the visible near support; the descending support edge is visible.','Pale smooth surface; polymer versus coated wood is not certified.'),
6:([376,816],7,'Fork opening upper-left inside corner','The underside of the lower front rail meets the left edge of its support beside a dark fork opening; both adjacent structural boundaries are visible.','Brown wood-like texture; material not certified.'),
9:([225,339],8,'Beam/support junction beside left fork opening','The lower beam edge and the upper-left boundary of the large support meet. The paper label is below this proposed junction.','Brown rough wood-like texture; stacked structures and load are present.'),
10:([465,866],7,'Fork opening upper-right support corner','The diagonal front rail underside ends at the left edge of the middle support; the dark opening and descending support edge give local structure.','Brown wood-like texture; loaded pallet.'),
12:([1109,832],8,'Lower outer corner of near support block','The front and right-side support faces terminate at the visible lower outer corner against the floor. Frayed wood and blur widen the location region.','Brown wood-like support under dark wrapped load.'),
16:([1444,631],7,'Outer deck/support turn','The long front deck edge, short right deck return, and descending near support edge meet in a visible turn.','Pale brown wood-like surface; material not certified.'),
17:([451,762],7,'Near deck/support three-edge turn','Two pale horizontal deck edges turn above the nearest upright support; its vertical corner remains visible below the load.','Pale or white-coated surface; polymer versus coated wood is not certified.'),
22:(None,None,'UNRESOLVED','The dark rail/opening/support boundaries remain blurred and low contrast at native resolution. No single structural endpoint is precise enough for this proposal.','Dark/black pallet-like base; polymer is plausible but not certified.'),
23:([568,603],6,'Near deck/support corner','The long pale deck rail turns into the right-return rail above a brown/pale support. The near vertical edge is visible; the adjacent load is not the cue.','Mixed pale/brown appearance; material not certified.')}
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
(H/'figures').mkdir(exist_ok=False);rows=[];cards=[];rawdetailcontact=Image.new('RGB',(1680,1860),'#f5f7fa');markcontact=rawdetailcontact.copy();overview=Image.new('RGB',(1680,1260),'#f5f7fa')
for z,(case,(xy,u,kind,obs,appearance))in enumerate(props.items()):
 r=source[case];m=rm[case]
 for b in [r['original_JPEG'],m['ruler'],m['source_raw_crop']]:assert sha(b['path'])==b['sha256']
 im=Image.open(r['original_JPEG']['path']).convert('RGB');assert ash(im)==r['decoded_RGB_sha256']
 bounds=m['native_bounds_half_open'];native=im.crop(bounds);old=Image.open(m['source_raw_crop']['path']).convert('RGB');assert old.size==native.size and old.tobytes()==native.tobytes()
 if xy is not None:assert bounds[0]<=xy[0]<bounds[2]and bounds[1]<=xy[1]<bounds[3]
 center=xy or [615,640];db=[center[0]-64,center[1]-64,center[0]+64,center[1]+64];raw=im.crop(db);detail=raw.resize((512,512),Image.Resampling.NEAREST);marked=detail.copy();dr=ImageDraw.Draw(marked)
 unc=None if xy is None else [xy[0]-u,xy[1]-u,xy[0]+u,xy[1]+u]
 if xy is not None:
  at=[(xy[0]-db[0])*4+1.5,(xy[1]-db[1])*4+1.5];dr.line((at[0]-15,at[1],at[0]+15,at[1]),fill='#ff00b4',width=2);dr.line((at[0],at[1]-15,at[0],at[1]+15),fill='#ff00b4',width=2)
  dr.rectangle(((unc[0]-db[0])*4,(unc[1]-db[1])*4,(unc[2]-db[0]+1)*4-1,(unc[3]-db[1]+1)*4-1),outline='#ffbe00',width=2)
 else:dr.text((12,12),'U: no cue coordinate proposed',font=font,fill='yellow')
 card=Image.new('RGB',(1500,1080),'#f5f7fa');d=ImageDraw.Draw(card)
 d.text((20,12),f'CASE {case:02d} | ann {r["representative_bbox_annotation"]["id"]} | original TRAIN | '+('AI AUX cue proposal'if xy else'U / no point'),font=font,fill='#17354d')
 d.text((20,40),'Source visual proposal only; native8 UNKNOWN; human gold false; training admission false',font=small,fill='#6e421a')
 full=im.copy();fd=ImageDraw.Draw(full);ann=r['representative_bbox_annotation'];bb=ann['bbox'];fd.rectangle((bb[0],bb[1],bb[0]+bb[2],bb[1]+bb[3]),outline='#00ff90',width=4)
 full.thumbnail((400,290));card.paste(full,(20,100));d.text((20,78),'Full context: separate official bbox overlay',font=small,fill='black')
 card.paste(native,(450,100));d.text((450,78),f'Unmodified native bbox crop, exactly 1x | bounds {bounds}',font=small,fill='black')
 d.text((20,480),f'Raw native128 detail -> exact 4x | bounds {db}',font=small,fill='black');d.text((560,480),'Separate marked copy: magenta r / yellow uncertainty',font=small,fill='black');card.paste(detail,(20,510));card.paste(marked,(560,510))
 notes=[f'Cue: {kind}',f'xy: {xy}; uncertainty: {unc}',obs,appearance,'BBox is annotation identity evidence, not segmentation or exact instance ownership.']
 yy=515
 for note in notes:
  for line in textwrap.wrap(note,42):d.text((1090,yy),line,font=small,fill='#17354d');yy+=21
  yy+=15
 f=H/'figures'/f'CASE_{case:02d}_CUE_REVIEW.png';card.save(f);cards.append(bind(f))
 col=z%3;line=z//3;pos=(col*560+20,line*620+65)
 rawdetailcontact.paste(detail,pos);markcontact.paste(marked,pos)
 for b in(rawdetailcontact,markcontact):
  bd=ImageDraw.Draw(b);bd.text((col*560+20,line*620+8),f'CASE {case:02d} ann{ann["id"]} TRAIN',font=font,fill='black');bd.text((col*560+20,line*620+34),f'4x native detail; '+(f'r={xy} +/-{u}px'if xy else'U, no coordinate'),font=small,fill='black')
 full2=im.copy();full2.thumbnail((520,345));overview.paste(full2,(col*560+20,line*420+50));od=ImageDraw.Draw(overview);od.text((col*560+20,line*420+10),f'CASE {case:02d} | {r["take"]} | raw full context',font=small,fill='black')
 rows.append(dict(selection_index=case,source='LOCO_OFFICIAL_TRAIN_SUBSET2',original_fold='TRAIN',take=r['take'],source_frame_group='LOCO:subset-2:'+r['take'],camera_stream=r['camera_stream'],timestamp_raw=r['timestamp_raw'],timestamp_kind=r['timestamp_kind'],image_id=r['image_id'],original_member=r['original_member'],full_RGB=r['original_JPEG'],decoded_RGB_sha256=r['decoded_RGB_sha256'],raw_hw=r['native_hw'],target_identity=dict(official_annotation=ann,designated_bbox_xywh=bb,native_bbox_half_open=bounds,scope='The already designated annotation bbox is preserved. A structural cue inside it is proposed; bbox does not certify pixel ownership or isolate stacked instances.'),ruler=m['ruler'],unmodified_bbox_crop=m['source_raw_crop'],visual_status='VISIBLE_STRUCTURAL_CUE_PROPOSAL'if xy else'UNRESOLVED_U',cue_xy_original_RGB=xy,uncertainty_bbox_xyxy_closed=unc,uncertainty_radius_px=u,cue_kind=kind,visible_neighbor_edges=obs,material_appearance=appearance,physical_instance_ambiguity='Stacked pallet components may share the annotation bbox; intended component is described explicitly.'if case in(6,9)else'Official bbox designated; per-pixel instance ownership is not certified.',detail_bounds_half_open=db,detail_zoom=4,card=bind(f),human_gold=False,native8_mapping='UNKNOWN',semantic_D_E_S_label=None,training_admission=False))
for name,im in [('CONTACT_RAW_NATIVE_DETAILS_4X.png',rawdetailcontact),('CONTACT_MARKED_CANDIDATES_4X.png',markcontact),('CONTACT_ORIGINAL_FULL9.png',overview)]:im.save(H/name)
result=dict(completed=True,scope='Actual visual review of fixed nine root-screened PUBLIC TRAIN originals; one structural cue maximum per frame, no model-based selection.',source_root_screen=bind(S.parent/'ROOT_PUBLIC24_ACTUAL_VISUAL_SCREEN.json'),source_ruler_manifest=bind(S/'coordinate_rulers/COORDINATE_RULER_MANIFEST_V2.json'),source_intake=bind(S/'EXTRACTION_AND_RENDER_COMPLETE.json'),code=bind(__file__),rows=rows,counts=dict(reviewed_frames=9,visible_structural_cue_proposals=sum(r['cue_xy_original_RGB']is not None for r in rows),unresolved_no_coordinate=sum(r['cue_xy_original_RGB']is None for r in rows),original_TRAIN=9,capture_takes=len(set(r['take']for r in rows)),new_training_admission=0),contacts={k:bind(H/k)for k in ['CONTACT_RAW_NATIVE_DETAILS_4X.png','CONTACT_MARKED_CANDIDATES_4X.png','CONTACT_ORIGINAL_FULL9.png']},limits=['All original TRAIN assignments and annotation identities are preserved; no original VAL/TOMIE is moved into TRAIN.','These are prospective AI visual AUX region proposals for root review, not human gold or native8 correspondences.','The closed uncertainty box is a location proposal region, not a certified all-visible pixel mask; blur, layered rails, loads and stacked pallets can broaden it.','A visible cue proposal supplies no natural external E/S teacher; absence or blur is unresolved, not hidden evidence.','Nine frames belong to five takes in the same documented environment/date; these are correlated diversity candidates, not a new independent holdout.','Native raw full RGB and original bbox crops remain unchanged; only separate display copies carry marks.'],new_model_encoder_fit_current_V112_PnP=0,root_validation_pending=True)
put(H/'CANDIDATE_CUES9.json',result)
print(json.dumps({'candidate':bind(H/'CANDIDATE_CUES9.json'),'counts':result['counts'],'contacts':result['contacts']}))
