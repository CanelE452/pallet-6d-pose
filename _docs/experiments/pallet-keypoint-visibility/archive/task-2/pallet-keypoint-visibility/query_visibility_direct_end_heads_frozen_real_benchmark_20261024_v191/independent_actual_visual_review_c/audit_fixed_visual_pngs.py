"""PNG-only parity + actual C12 tool-view observation seal; no source arrays."""
from pathlib import Path
from PIL import Image
import datetime,hashlib,json,os,re,stat
H=Path(__file__).absolute().parent;V=H.parent;OUT=V/'fixed_direct_end_real_visual_v1'
IDS=[0,2,16,56,256,300,840,844,848,908,914,996]
def need(v,m):
 if not v:raise AssertionError(m)
def raw(path):
 p=Path(path);need(V in p.parents and '..' not in p.parts and all('v112' not in s.casefold() for s in p.parts),'Explicit allowed path before access')
 fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for part in p.parts[1:-1]:
   nxt=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nxt
  final=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 finally:os.close(fd)
 need(stat.S_ISREG(os.fstat(final).st_mode),'Regular saved artifact')
 with os.fdopen(final,'rb') as f:return f.read()
def sha(b):return hashlib.sha256(b).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(raw(p)))
def checked(b):
 d=raw(b['path']);need(sha(d)==b['sha256'],'Manifest-bound saved PNG/text SHA');return d
def write(p,o):
 with p.open('x') as f:json.dump(o,f,ensure_ascii=False,indent=2);f.write('\n')
planpath=V/'ROOT_FIXED_REAL_VISUAL_QA_PLAN_BEFORE_PNG_TOOL_VIEWS.json'
need(sha(raw(planpath))=='260320841af72b6188a70b15e464a98859fb4ed4f6b5ed5272a1b9c879f1ee53','Exact root prospective view plan')
plan=json.loads(raw(planpath));need(plan['approved_actual_fixed_saved_crop_QA'] is True,'Root actual PNG review entry')
done=json.loads(checked(plan['visual_complete']));need(done['completed'] is True and done['PASS'] is True and done['selected_unique_cases']==68,'Terminal renderer completion;68selected not allviewed')
manifest=json.loads(checked(plan['image_manifest']));chosen=plan['independent_C_fixed_cases'];need([x['original_global_index'] for x in chosen]==IDS,'Exactly fixed C12, no replacement')
lookup={x['original_global_index']:x for x in manifest['images']};need(len(lookup)==68,'Full manifest contains68 unique selectedcases')
pixelchecks=[]
for x in chosen:
 i=x['original_global_index'];need(lookup[i]==x,'Root plan exact image-manifest record')
 for name in ['card','clean','valid']:checked(x[name])
 with Image.open(x['clean']['path']) as image:rgb=image.convert('RGB');need(rgb.size==(96,96),'Clean native96');rb=rgb.tobytes()
 with Image.open(x['valid']['path']) as image:
  mask=image.convert('L');need(mask.size==(96,96) and set(mask.getdata())<={0,255},'Valid96 binary raster');vb=bytes(1 if a else 0 for a in mask.getdata())
 with Image.open(x['card']['path']) as image:
  card=image.convert('RGB');need(card.size==(1600,1120),'Readable card dimensions');need(card.crop((980,202,1076,298)).tobytes()==rb,'Clean96 ROI has no overlay or resampling')
 need(sha(rb)==x['original_RGB96_array_sha256'] and sha(vb)==x['original_valid96_array_sha256'],'Decoded PNGs match renderer-bound original-array hashes')
 pixelchecks.append(dict(original_global_index=i,card=x['card'],clean=x['clean'],valid=x['valid'],card_clean96_ROI_exact=True,decoded_RGB_valid_hashes_exact=True))
observed={
0:('DIRECT_VISIBLE','E','D','0.823053','-8.67266','Wooden surfaces and a dark corner/edge junction are visible around the displayed q. Full validity. C54 rejects the stored visible label while C55 agrees. Blurring limits precise endpoint or selected-instance identity inference.'),
2:('DIRECT_VISIBLE','D','E','-3.81328','9.1483','Wood edge against grey floor; displayed q lies close to that image boundary. C55 rejects the stored visible label even with existing error2.00706px. No new semantic label or coordinate-cause attribution follows from the crop.'),
16:('DIRECT_VISIBLE','D','D','-8.16998','-12.6195','Wood block/plank corner is visible; both direct routes agree with stored D. Existing error6.21816px is separately shown. This is one selected successful example, not prevalence evidence.'),
56:('DIRECT_VISIBLE','U','U','UNKNOWN','UNKNOWN','Large lower black padding and the black portion of valid96 align; q falls in the invalid portion. inputavailableFalse and all saved routes U are clearly shown separately from the stored human D. Not counted as a newly visible/background annotation.'),
256:('EXTERNAL_OCCLUDED','E','D','8.43844','-3.16382','Red foreground surface next to grey floor occupies much of the crop. Stored human E is displayed; C54 agrees and C55 misses. This crop alone does not establish which selected endpoint lies behind the red object.'),
300:('EXTERNAL_OCCLUDED','E','E','5.79996','2.13196','Grey floor dominates around q with wooden slats lower in the crop. Both direct routes agree with stored E; existing error103.572px and pnp_projected reference authority are explicit. Local image cannot certify intended-end presence or a physical occlusion cause.'),
840:('DIRECT_VISIBLE','E','D','6.43195','-4.15729','Green rim/side and a red cone-like foreground object are visible; q is on the green boundary region. C54 rejects stored D despite existing error2.70108px, C55 agrees. Do not relabel the point or claim precise endpoint certification.'),
844:('DIRECT_VISIBLE','D','D','-3.87416','-6.15501','Green rim corner against grey background is visible near q. Both direct routes agree with stored D; error1.52622px is an existing association. Crop appearance supports readability, not a new instance/native correspondence proof.'),
848:('DIRECT_VISIBLE','E','D','0.953212','-10.3676','Green rim with a red cone-like object above it; q lies near the highlighted rim boundary. C54 rejects stored D, C55 agrees; existing error5.10185px is shown separately. No failure mechanism is inferred.'),
908:('EXTERNAL_OCCLUDED','E','E','6.89831','0.328205','Red cone-like surface/base fills the query neighborhood with green rim below. Both direct routes match stored E. This is a selected agreement case, not independent physical-native proof or general success.'),
914:('UNANNOTATED','U','U','UNKNOWN','UNKNOWN','Lower black padding aligns with invalid valid96; inputavailableFalse and all model routes U are legible. Human UNANNOTATED and missing reference/error remain UNKNOWN, not filled from the image.'),
996:('EXTERNAL_OCCLUDED','D','D','-9.48973','-12.9658','Green rim and adjacent red cone-like base are visible with grey background near q. Both direct routes miss stored human E although existing manual-click association error1.13822px is small. The local crop cannot verify the exact occluded endpoint or distinguish cause, so original label remains unchanged.')}
rows=[]
for sequence,x in enumerate(chosen,1):
 i=x['original_global_index'];human,c54,c55,z54,z55,note=observed[i]
 rows.append(dict(original_global_index=i,id=x['id'],native_slot=x['native_slot'],memberships=x['memberships'],actually_viewed=dict(tool='functions.exec -> tools.view_image',tool_call_sequence=sequence,call_batch=(sequence+1)//2,path=x['card']['path'],sha256=x['card']['sha256'],detail='high',successful_image_return_seen_by_reviewer=True,tool_output='The bound CARD.png pixels were returned into reviewer context; no separate opaque tool output ID was exposed.'),displayed_facts_only=dict(human_state=human,C54=c54,C55=c55,C54_rounded_raw=z54,C55_rounded_raw=z55),observation=note,layout_readable=True,no_title_table_footer_overlap=True,new_ground_truth_annotation=False))
with (H/'ALL12_ACTUAL_VISUAL_OBSERVATIONS.jsonl').open('x') as f:
 for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
gallery=checked(done['gallery']).decode();refs=re.findall(r'\]\(([^)]+)\)',gallery)
for ref in refs:
 need(not ref.startswith('/') and not re.match(r'^[A-Za-z]+:',ref) and '..' not in Path(ref).parts and all('v112' not in a.casefold() for a in Path(ref).parts),'Portable gallery pathguard')
 need(stat.S_ISREG((OUT/ref).lstat().st_mode),'Portable literal target exists and is not symlink')
receipt=dict(completed=True,PASS=True,review_type='Independent actual fixed-C12 saved-card visual QA and PNG-only consistency',recorded_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),root_plan=bind(planpath),renderer_complete=plan['visual_complete'],image_manifest=plan['image_manifest'],renderer_code=done['code'],renderer_protocol=done['protocol'],QA_code=bind(__file__),observations=bind(H/'ALL12_ACTUAL_VISUAL_OBSERVATIONS.jsonl'),actual_tool_viewed_card_count=12,actual_tool_viewed_original_indices=IDS,actual_tool_viewed_WOOD=6,actual_tool_viewed_SQUARE=6,selected_gallery_cases=68,full_original_current_slots=1784,all68_or_all1784_actually_viewed=False,tool_view_calls=12,tool_view_batches=6,additional_clean_valid_tool_views=0,PNG_only_automatic_checks=dict(cards_hashed=12,clean_PNG_hashed_and_decoded=12,valid_PNG_hashed_and_decoded=12,total_PNG_hashed=36,raw_NPZ_cache_or_model_or_original_RGB_reads=0,card_native_clean96_ROI_byte_parity=12,original_decoded_array_hash_parity=12,checks=pixelchecks),portable_gallery=done['gallery'],portable_local_refs_existing=len(refs),HTML_browser_rendered=False,layout_or_readability_blockers=[],findings=['Selected saved-card agreements and failures both displayed; small existing reference error does not eliminate all classification disagreement. These are fixed post-result strata, not random representative performance samples.','Input-U/padding, human UNANNOTATED/reference UNKNOWN, and model U are preserved separately. Black invalid pixels are not newly called background.','A local crop cannot establish intended semantic endpoint/selected-instance identity or the true cause of every disagreement; no annotation, q/L or decoder is changed.'],limits=['Only these12 CARD images actually viewed via image tools; no claim all68selected or all1784 inspected.','Numerical CM/metrics independence belongs to separate saved numeric audit; this visual review does not rerun model/evaluation.','Stored human semantic authority retained; existing coordinate error is an association/proxy, not newly certified physical point gold.','Repeated real-set exposure; not blind/fresh validation. No model/threshold/data/architecture/arm selection.'],model_forward_fit_encoder_PnP_or_crop_sampling=0,new_label_or_admission_or_model_choice=0,selected_model=None,adopted=False,fully_solved=False)
write(H/'REVIEW.json',receipt);print(json.dumps(dict(completed=True,PASS=True,review=bind(H/'REVIEW.json'),observations=bind(H/'ALL12_ACTUAL_VISUAL_OBSERVATIONS.jsonl'),actual_views=12,PNG_hashchecks=36,portable_links=len(refs))))
