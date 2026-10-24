from pathlib import Path
import json,hashlib,datetime
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parent; O=R/'REFERENCE_PASS_ROOT_A'; O.mkdir(exist_ok=True)
rows=[json.loads(l) for l in (R/'EXTENSION_40_RGB_REVIEW_MANIFEST.jsonl').read_text().splitlines()]
# Coordinates are RGB-only observations made on the fixed native-coordinate grids.
chosen={
4:[([79,454],[75,450,84,459],'UPPER_LEFT_SUPPORT_EDGE_TURN','Exposed left wooden support meets the thin deck below the loaded carton; the lower foot is near the frame cut and is not completed.'),([169,435],[165,431,174,440],'UPPER_MIDDLE_SUPPORT_EDGE_TURN','Visible deck-to-central-support turn below the same carton; floor/gray wrap is excluded.')],
5:[([124,437],[119,432,130,443],'LOWER_LEFT_SUPPORT_END','Left visible wooden foot meets the floor below the open fork slot; the foreground gray item is outside this reference.'),([306,417],[301,412,312,423],'LOWER_RIGHT_SUPPORT_END','Image-right wooden foot meets the floor, along the same exposed front pallet base.')],
11:[([921,657],[914,650,929,665],'LOWER_LEFT_SUPPORT_END','Exposed near-left wooden support meets the floor; the packing roll occludes the right end and is not used.'),([919,630],[912,623,927,638],'UPPER_LEFT_DECK_END','Visible near-left wooden deck end under the wrapped load, before the vertical support.')],
17:[([1145,600],[1139,594,1152,607],'LOWER_NEAR_BASE_TURN','Near lower wooden base corner where the two exposed pallet faces meet beneath blue containers.'),([1258,570],[1252,564,1265,577],'LOWER_RIGHT_BASE_END','Far image-right exposed base end under the blue containers; a small uncertainty region preserves the soft wood/floor transition.')],
18:[([302,620],[296,614,309,627],'LOWER_FRONT_SUPPORT_END','Visible right end of the near wooden foot; yellow post hides the left part and no hidden point is completed.'),([300,592],[294,586,307,599],'UPPER_FRONT_SUPPORT_TURN','Visible upper deck-to-support junction on the same near wooden base, to the right of the yellow post.')]
}
reasons={0:'Top-truncated blurred elevated pallet; stable individual endpoint not localized.',1:'Loaded pallet on blue cart is visible, but pallet/cart support ownership and endpoint localization remain ambiguous under the fixed rule.',2:'Elevated small rack pallet; unresolved individual endpoint.',3:'Elevated rack pallet is small and partly cut.',6:'Loaded wooden base is visible but blurred support ends cannot be localized consistently.',7:'Small low-resolution wooden base; endpoint identity not established.',8:'Black-wrapped load/base is cut at left and bottom; individual pallet association ambiguous.',9:'Small elevated rack pallet; no reliable exposed endpoint.',10:'Distant loaded structure at image-right; visible pallet cue not established.',12:'Pallet/load is crossed by steel rail; wood endpoint ownership uncertain.',13:'Wrap-covered base under white load; endpoint material and localization uncertain.',14:'Blurred green pallet-like base; physical endpoint ownership uncertain.',15:'Distant cage/wrapped structure; no reliable pallet endpoint.',16:'Only 28 by 7 pixel annotated extent; no localizable reference.',19:'Distant stack/group; individual pallet reference not established.'}
frames=[];refs=[]
for x in rows[:20]:
 i=x['extension_index'];fr=dict(x);pr=[]
 for j,(xy,ub,role,evidence) in enumerate(chosen.get(i,[])):
  rr={'cue_refID':f"{x['review_id']}:ROOT_RGB_REF{j}",'extension_index':i,'image':x['image'],'image_sha256':x['image_sha256'],'reference_xy':xy,'uncertainty_bbox_half_open_xyxy':ub,'reference_role_2D':role,'actual_RGB_evidence':evidence,'state_proposal':'VISIBLE_REFERENCE_CONDITIONAL_AI','native8_mapping':'UNKNOWN','human_verified':False,'official_corner_annotation':False,'admitted_training_teacher':False,'natural_E_or_S_generated':False,'initial_query_xy':None,'initializer_detection_or_raw_index_used':None,'source_mask_or_model_used_for_geometry':False,'target_instance_association':'AI RGB observation of the frozen representative entity; conditional, not human gold','localization_uncertainty_reason':'Resolution, blur, perspective and material-edge softness; range is preserved, no exact pixel-support truth.'}
  pr.append(rr);refs.append(rr)
 fr.update(reference_proposals=pr,frame_reference_state='CONDITIONAL_VISIBLE_REFERENCE_CANDIDATES' if pr else 'U',U_reason=None if pr else reasons[i]);frames.append(fr)
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
ledger=O/'RGB_REFERENCE_PROPOSAL_LEDGER.json';write(ledger,{'schema':'V74_EXTENSION_ROOT_A_RGB_REFERENCE_LEDGER','created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'indices':list(range(20)),'protocol':'../EXTENSION_PROTOCOL.json and ../../REFERENCE_REVIEW_PROTOCOL.json','model_mask_or_predicted_query_read':False,'current223_read':False,'native8_gold_added':0,'human_gold_added':0,'natural_E_S_added':0,'frames':frames})
rl=O/'RGB_REFERENCE_PROPOSALS.jsonl';rl.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in refs))
write(O/'RGB_REFERENCE_GEOMETRY_LOCK.json',{'schema':'V74_EXTENSION_ROOT_A_GEOMETRY_LOCK','frozen_before_masks_or_predictions':True,'conditional_reference_count':len(refs),'frames_with_reference':len(chosen),'unknown_frames':20-len(chosen),'bindings':[{'path':str(p),'sha256':sha(p)} for p in [ledger,rl,Path(__file__).resolve(),R/'EXTENSION_40_SELECTION_LOCK.json',R/'EXTENSION_40_RGB_REVIEW_MANIFEST.jsonl']]})
# Display-only marked copies after the geometry lock. Raw sources stay unchanged.
font=ImageFont.truetype('/usr/share/fonts/truetype/nanum/NanumGothic.ttf',19)
for page in range(5):
 can=Image.new('RGB',(1100,1080),'white');d=ImageDraw.Draw(can)
 for k,fr in enumerate(frames[page*4:page*4+4]):
  raw=Image.open(fr['raw_native_crop']).convert('RGB');marked=raw.copy();dm=ImageDraw.Draw(marked);b=fr['expanded_bbox_native_xyxy']
  for j,ref in enumerate(fr['reference_proposals']):
   x,y=ref['reference_xy'];u=ref['uncertainty_bbox_half_open_xyxy'];xx=x-b[0];yy=y-b[1]
   dm.rectangle((u[0]-b[0],u[1]-b[1],u[2]-b[0],u[3]-b[1]),outline='yellow',width=2);dm.line((xx-5,yy,xx+5,yy),fill='red',width=2);dm.line((xx,yy-5,xx,yy+5),fill='red',width=2)
  raw.thumbnail((490,210));marked.thumbnail((490,210));y=k*270
  can.paste(raw,(15,y+36));can.paste(marked,(565,y+36));d.text((15,y+5),f"EXT {fr['extension_index']:02d} {fr['frame_reference_state']} ({len(fr['reference_proposals'])})",font=font,fill='black')
 can.save(O/f'RAW_MARKED_PAGE_{page+1:02d}.png')
print(json.dumps({'frames':20,'reference_frames':len(chosen),'conditional_refs':len(refs),'U_frames':20-len(chosen),'ledger_sha':sha(ledger)},ensure_ascii=False))
