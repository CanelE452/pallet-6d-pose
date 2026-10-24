from pathlib import Path
import json,hashlib,datetime
from PIL import Image,ImageDraw
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility');R=P/'query_visibility_source_cue_corpus_20261023_v85';D=R/'annotator_b/batch03'
def bind(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
fmt=R/'ANNOTATION_RECORD_FORMAT_BEFORE_LABELS.json';assert bind(fmt)['sha256']=='170eaa7413db74101060564f0f99b3aa19f6b0e30a7c267858f456a67d55c6de';assignment=R/'ANNOTATOR_B_BATCH03_BEFORE_NEW_RGB.json';rows=json.loads(assignment.read_text())['rows'];slots=json.loads(fmt.read_text())['part_slots'];assert len(rows)==5
# Coordinates are independent before-only RGB observations, not GT/model-derived.
settings=[{'coords': [[272, 173], [375, 172], [273, 219], [375, 219], None, None],
  'notes': 'Object is small at native resolution. Plank corner localization is regional across several pixels; lower front lip '
           'and support overlap retain uncertainty. Thin runner end shapes are unresolved rather than forced from floor shadow.',
  'radii': [4, 4, 4, 4, None, None],
  'unknown': ['', '', '', '', 'UNKNOWN_U', 'UNKNOWN_U'],
  'wall': 'Small near/front broad horizontal-plank wall at the center of the lab aisle. Back opening/rim is visible above; the '
          'four selected points refer to the near/front wall, not the back rim.',
  'why_unknown': ['',
                  '',
                  '',
                  '',
                  'The thin dark runner end is not separable confidently from the foot/base contact shadow at this small native '
                  'size.',
                  'The runner upper protrusion junction is not resolved separately from dark support/floor silhouette.']},
 {'coords': [[547, 254], None, None, None, None, None],
  'notes': 'One observed front-board contour turn only; do not substitute the high back-rim corner or infer lower references '
           'behind cardboard.',
  'radii': [5, None, None, None, None, None],
  'unknown': ['', 'OUT_OF_FRAME_U', 'SELF_OR_UNRESOLVED_U', 'OUT_OF_FRAME_U', 'SELF_OR_UNRESOLVED_U', 'OUT_OF_FRAME_U'],
  'wall': 'Only an oblique right-edge portion of the near/front horizontal-board wall is visible. Its upper-left corner is '
          'exposed; the right wall extends out of frame and a pre-existing cardboard carton hides the lower/front structure.',
  'why_unknown': ['',
                  'Right principal wall endpoint is outside the image.',
                  'Lower wall junction is hidden by the existing cardboard/cropped structure and not localizable.',
                  'Right lower wall endpoint is outside the image or hidden.',
                  'Runner end is not observed behind foreground/cropped structure.',
                  'Right runner end is outside the image or hidden.']},
 {'coords': [[278, 163], [370, 163], [280, 206], [369, 206], None, None],
  'notes': 'Very small object; board corners denote uncertain image regions, not exact material pixels. Separate thin '
           'runner/foot junctions are unresolved and remain null.',
  'radii': [4, 4, 4, 4, None, None],
  'unknown': ['', '', '', '', 'UNKNOWN_U', 'UNKNOWN_U'],
  'wall': 'Small central frontal broad-plank wall in the lab. The near/front upper horizontal edge and lowest wide front-board '
          'lip are observed, with the back/opening rim above.',
  'why_unknown': ['',
                  '',
                  '',
                  '',
                  'The thin dark runner end is not separable confidently from the foot/base contact shadow at this small native '
                  'size.',
                  'The runner upper protrusion junction is not resolved separately from dark support/floor silhouette.']},
 {'coords': [[262, 131], [386, 131], [264, 189], [384, 189], None, None],
  'notes': 'Small native apparent size and blur. The lower board corners are region-level proposals; thin base-runner ends '
           'cannot be confidently separated from support/contact shadow.',
  'radii': [4, 4, 4, 4, None, None],
  'unknown': ['', '', '', '', 'UNKNOWN_U', 'UNKNOWN_U'],
  'wall': 'Central near/front broad wooden wall in the hall, viewed slightly from above; the near/front upper edge is lower than '
          'the back rim. Four outer front-board corner regions are observed.',
  'why_unknown': ['',
                  '',
                  '',
                  '',
                  'The thin dark runner end is not separable confidently from the foot/base contact shadow at this small native '
                  'size.',
                  'The runner upper protrusion junction is not resolved separately from dark support/floor silhouette.']},
 {'coords': [[553, 255], None, None, None, None, None],
  'notes': 'One exposed front-board corner above existing cardboard. The receding/back rim has a different identity and does not '
           'replace requested wall slots.',
  'radii': [5, None, None, None, None, None],
  'unknown': ['', 'OUT_OF_FRAME_U', 'SELF_OR_UNRESOLVED_U', 'OUT_OF_FRAME_U', 'SELF_OR_UNRESOLVED_U', 'OUT_OF_FRAME_U'],
  'wall': 'The near/front horizontal-board wall is truncated at the far right. Its upper-left corner is visible above the '
          'foreground cardboard, while the right end and lower structures are not observable.',
  'why_unknown': ['',
                  'Right principal wall endpoint is outside the image.',
                  'Lower wall junction is hidden by the existing cardboard/cropped structure and not localizable.',
                  'Right lower wall endpoint is outside the image or hidden.',
                  'Runner end is not observed behind foreground/cropped structure.',
                  'Right runner end is outside the image or hidden.']}]
descriptions={'UPPER_WALL_LEFT':'Upper top contour meets left outer edge of the principal broad front wall/top plank.','UPPER_WALL_RIGHT':'Upper top contour meets right outer edge of the principal broad front wall/top plank.','LOWEST_WIDE_WALL_LEFT':'Left outer edge meets the bottom lip of the lowest wide plank above the narrower feet.','LOWEST_WIDE_WALL_RIGHT':'Right outer edge meets the bottom lip of the lowest wide plank above the narrower feet.','BASE_RUNNER_LEFT':'Left protruding end of the thin base runner: upper runner edge joins its short outer end contour.','BASE_RUNNER_RIGHT':'Right protruding end of the thin base runner: upper runner edge joins its short outer end contour.'}
manifest=[]
for bi,(row,setting) in enumerate(zip(rows,settings)):
 pd=D/'pairs'/row['pair_id'].replace(':','__');pd.mkdir(parents=True,exist_ok=True);src=Path(row['before_RGB']['path']);assert bind(src)['sha256']==row['before_RGB']['sha256'];im=Image.open(src).convert('RGB');assert im.size==(640,480);entries=[]
 for k,(slot,q,rad) in enumerate(zip(slots,setting['coords'],setting['radii'])):
  if q is None:
   state=setting['unknown'][k];box=None;evidence=setting['why_unknown'][k]
  else:
   state='APPARENT_VISIBLE_STRUCTURAL_CUE_AI';box=[q[0]-rad,q[1]-rad,q[0]+rad,q[1]+rad];evidence='The named adjoining wood contours and the adjacent wood face are directly observed in the before fullscene/native detail; no coordinate was inferred from a mask, pose, model query or after view.'
  entries.append({'part_slot':slot,'reference_xy':q,'uncertainty_bbox_xyxy_half_open':box,'before_state':state,'cue_description':descriptions[slot],'evidence':evidence,'ambiguity':setting['notes'],'native8_semantic_id':None,'native8_status':'UNKNOWN','centerpixel_material_certified':False,'human_gold':False,'training_admitted':False,'uncertainty_is_calibrated':False})
 r={'schema':'V85_PRIMARY_ANNOTATOR_B_BEFORE_CUES_V1','pair_id':row['pair_id'],'source_split':row['split'],'source_before_RGB':{**row['before_RGB'],'raw_hw':[480,640]},'principal_wall_description':setting['wall'],'before_only_frozen':True,'slots':entries,'policy':bind(D/'READER_POLICY_BEFORE_BATCH_RGB.json'),'root_scope':bind(R/'SOURCE_CUE_CORPUS_SCOPE_BEFORE_NEW_RGB.json'),'format':bind(fmt),'assignment':bind(assignment),'selection':'Allsix predeclared physical view-relative slots retained; missing cues explicitly null. No after view used.','before_foreground_caution':'Nominal even before may already contain cardboard; visibility follows actual RGB, not parity.','code':bind(Path(__file__)),'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 f=pd/'BEFORE_CUES.json';assert not f.exists();f.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
 # Scientific visual receipt; original raw RGB stays at its original path.
 contact=Image.new('RGB',(660,560),(24,24,24));dr=ImageDraw.Draw(contact);cropbindings=[]
 for k,e in enumerate(entries):
  ox=8+(k%3)*220;oy=8+(k//3)*280;dr.text((ox,oy),e['part_slot'],fill=(255,255,255));q=e['reference_xy']
  if q is None:dr.multiline_text((ox,oy+36),'NULL / '+e['before_state']+'\nNo observed before reference',fill=(240,220,100),spacing=6);continue
  xy=[q[0]-32,q[1]-32,q[0]+32,q[1]+32];crop=im.crop(xy);rp=pd/(e['part_slot']+'_BEFORE_UNMARKED_NATIVE64.png');crop.save(rp);z=crop.resize((192,192),Image.Resampling.NEAREST);dd=ImageDraw.Draw(z);box=e['uncertainty_bbox_xyxy_half_open'];bb=[(box[0]-xy[0])*3,(box[1]-xy[1])*3,(box[2]-xy[0])*3,(box[3]-xy[1])*3];dd.rectangle(bb,outline=(255,235,0),width=2);contact.paste(z,(ox,oy+26));dr.text((ox,oy+224),f'r={q}; unc={box}',fill=(240,240,240));cropbindings.append({'part_slot':e['part_slot'],'native_crop_xyxy':xy,'raw_crop':bind(rp)})
 cp=pd/'BEFORE_SIX_SLOT_CONTACT.png';contact.save(cp);insp=D/(f'{bi:02d}_'+row['pair_id'].replace(':','__'))/'BEFORE_INSPECTION_METADATA.json'
 vr={'completed':True,'pair_id':row['pair_id'],'before_annotation':bind(f),'fullscene_before_and_native_grid_actually_viewed':True,'before_inspection':bind(insp),'contact':bind(cp),'raw_crops':cropbindings,'after_RGB_read':False,'reference_and_uncertainty_chosen_before_after':True,'root_policy_allows_after_only_after_before_seal':True}
 vp=pd/'BEFORE_VISUAL_RECEIPT.json';vp.write_text(json.dumps(vr,indent=2)+'\n');manifest.append({'pair_id':row['pair_id'],'pair_dir':str(pd),'before_annotation':bind(f),'before_contact':str(cp),'visual_receipt':bind(vp)})
f=D/'UNSEALED_BEFORE_METADATA_INDEX.json';assert not f.exists();f.write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
