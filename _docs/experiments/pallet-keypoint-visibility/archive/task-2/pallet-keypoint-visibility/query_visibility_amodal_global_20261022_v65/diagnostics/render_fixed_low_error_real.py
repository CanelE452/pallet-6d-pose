#!/usr/bin/env python3
"""Post-seal diagnostics of preexisting fixed real cases; no fit/inference."""
from pathlib import Path
import csv,hashlib,json,math
import numpy as np
from PIL import Image,ImageDraw,ImageFont

OUT=Path(__file__).resolve().parent;ROOT=OUT.parent;PALLET=ROOT.parent;V63=PALLET/'query_visibility_public_real_20261022_v63'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
 with Path(p).open(newline='')as f:return list(csv.DictReader(f))
def keyed(p):return {(r['id'],r['native_corner']):r for r in rows(p)}
def font(size):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)
def sealed_score(path):
 sealpath=path.with_name('INFERENCE_SEAL.json');seal=json.loads(sealpath.read_text())
 if 'completed'in seal:assert seal['completed']
 else:assert seal['candidate']=='MIXED_CONTEXT_RGB_SAM_V4'and seal['frames']==223 and seal['new_optimizer_updates']==seal['new_PnP_calls']==0
 matching=[r for r in seal['outputs']if Path(r['path']).resolve()==path.resolve()];assert len(matching)==1 and sha(path)==matching[0]['sha256']
 return sealpath,seal
def mark_crop(rgb,bounds,q,reference,size):
 crop=rgb.crop(tuple(bounds));scale=min(size[0]/crop.width,size[1]/crop.height);newhw=(round(crop.width*scale),round(crop.height*scale));crop=crop.resize(newhw,Image.Resampling.NEAREST);result=Image.new('RGB',size,'#eeeeee');ox=(size[0]-newhw[0])//2;oy=(size[1]-newhw[1])//2;result.paste(crop,(ox,oy));d=ImageDraw.Draw(result)
 point=lambda xy:(ox+(xy[0]-bounds[0])*scale,oy+(xy[1]-bounds[1])*scale)
 qp=point(q);rp=point(reference);x,y=qp;d.ellipse((x-7,y-7,x+7,y+7),outline='#ffdb00',width=2)
 x,y=rp;d.line((x-6,y-6,x+6,y+6),fill='#00e5ff',width=2);d.line((x-6,y+6,x+6,y-6),fill='#00e5ff',width=2)
 return result
def render(cases,path,title):
 width,rowh,header=1900,310,140;image=Image.new('RGB',(width,header+rowh*len(cases)+80),'white');d=ImageDraw.Draw(image)
 d.text((15,10),title,font=font(27),fill='black');d.text((15,49),'Fixed before v65 outputs: yellow circle=unchanged query; cyan X=existing semantic reference. RGB shown without image enhancement.',font=font(19),fill='black')
 d.text((15,80),'v65 Eproxy=full minus visible source support; O=outside source full support. These are not current human external/self labels.',font=font(18),fill='#555555')
 for x,text in [(15,'Image context'),(335,'Same initial query ROI'),(645,'Native24px zoom'),(960,'Saved model outputs / fixed0.5 decisions')]:d.text((x,112),text,font=font(18),fill='black')
 for i,c in enumerate(cases):
  y=header+i*rowh;rgb=Image.open(c['image']).convert('RGB');assert sha(c['image'])==c['image_sha256'];q=c['query'];ref=c['reference'];L=c['roi_side'];full=rgb.resize((300,225),Image.Resampling.BILINEAR);fd=ImageDraw.Draw(full);sx,sy=300/rgb.width,225/rgb.height
  fd.rectangle(((q[0]-L/2)*sx,(q[1]-L/2)*sy,(q[0]+L/2)*sx,(q[1]+L/2)*sy),outline='#ffdb00',width=2);image.paste(full,(15,y+35))
  bounds=[math.floor(q[0]-L/2),math.floor(q[1]-L/2),math.ceil(q[0]+L/2),math.ceil(q[1]+L/2)];image.paste(mark_crop(rgb,bounds,q,ref,(280,225)),(335,y+35))
  px,py=c['nearest_query_pixel'];zoom=[px-12,py-12,px+12,py+12];image.paste(mark_crop(rgb,zoom,q,ref,(280,225)),(645,y+35))
  d.text((15,y+3),f"{c['id']} / k{c['native_corner']}   human semantic={c['human_state_short']}",font=font(20),fill='black')
  v=c['v65'];lines=[f"v61 pV {c['v61_p_visible']:.3f}     v63 pV {c['v63_p_visible']:.3f}",f"v65 exact: pV {v['p_visible_exact']:.3f}, Eproxy {v['p_external_support_exact']:.3f}, full {v['p_full_exact']:.3f}",f"v65 near2: pV {v['p_visible_radius2']:.3f}, Eproxy {v['p_external_support_radius2']:.3f}, full {v['p_full_radius2']:.3f}",f"logical exact/near: {v['logical_exact_short']} / {v['logical_radius2_short']}",f"confidence exact/near: {v['confident_exact_short']} / {v['confident_radius2_short']}; U={v['confidence_U']}",f"initial error {c['error_px']:.3f}px; same native pixel={c['same_nearest_pixel']}",f"reference source: {c['annotation_coordinate_source']}",f"R2 square contains reference pixel: {c['reference_nearest_pixel_in_query_radius2']}"]
  for j,line in enumerate(lines):d.text((960,y+36+26*j),line,font=font(19 if j<5 else 17),fill='black'if j<5 else'#444444')
  d.text((335,y+269),f"q=({q[0]:.3f},{q[1]:.3f}); ref=({ref[0]:.3f},{ref[1]:.3f}); ROI={L:.1f}px",font=font(16),fill='#444444')
  d.line((15,y+rowh-2,width-15,y+rowh-2),fill='#cccccc')
 d.text((15,header+rowh*len(cases)+10),'Current pixel ownership/full-mask GT is unavailable. Existing human labels refer to semantic native corners/reference points.',font=font(18),fill='#333333')
 d.text((15,header+rowh*len(cases)+39),'These examples diagnose transfer after freezing; they do not select a model, threshold, radius, or new training data.',font=font(18),fill='#333333');image.save(path)

if __name__=='__main__':
 assert not(OUT/'FIXED_LOW_ERROR_REAL_CROPS.json').exists()
 selectionpath=V63/'diagnostics/CROP_SELECTION_AND_SCORES.json';selection=json.loads(selectionpath.read_text());fixed=selection['fixed_preexisting_cases'];assert len(fixed)==4
 scorepath=ROOT/'private_evaluation/MIXED_GLOBAL_UNWEIGHTED/REAL_QUERY_SCORES.csv';sealpath,seal=sealed_score(scorepath);assert seal['source_selection_seal']['sha256']=='6d54028ea2247571f4c2dbc4beb4380b741c06633400f8c74b11e8396eef6261'and seal['PnP_calls']==0 and seal['optimizer_updates']==0
 oldpath=V63/'private_evaluation/MIXED_CONTEXT_RGB_SAM_V4/REAL_QUERY_SCORES.csv';oldseal,_=sealed_score(oldpath);new=keyed(scorepath);old=keyed(oldpath);manifestpath=PALLET/'evaluation_preparation_20261022/PREPARED_QUERY_MANIFEST.csv';truth=keyed(manifestpath);provenancepath=PALLET/'near_error_diagnosis_20261022/real/NEAR_DIRECT_ROWS_WITH_PROVENANCE.csv';provenance=keyed(provenancepath)
 keys=[(r['id'],str(r['native_corner']))for r in fixed]+[('wood_day_01:005249','5'),('SQUARE_YOLO:030931','4')];cases=[]
 short=lambda s:{'VISIBLE_TARGET':'V','EXTERNAL_SUPPORT_PROXY':'Eproxy','OUTSIDE_FULL_SUPPORT':'O','UNKNOWN':'U'}[s]
 bindings={str(p):sha(p)for p in[selectionpath,scorepath,sealpath,oldpath,oldseal,manifestpath,provenancepath]}
 for position,key in enumerate(keys):
  r=truth[key];s=new[key];o=old[key];assert s['valid'].lower()=='true';q=[float(r['query_x']),float(r['query_y'])];reference=[float(r['human_reference_x']),float(r['human_reference_y'])];assert np.array_equal(q,[float(s['query_x']),float(s['query_y'])]);assert r['image_sha256']==s['image_sha256']==o['image_sha256'];qp=np.floor(np.array(q)+.5).astype(int);rp=np.floor(np.array(reference)+.5).astype(int);error=float(np.linalg.norm(np.array(q)-reference));assert abs(error-float(r['initial_reference_error_px']))<1e-9
  prov=provenance.get(key,{});v={k:float(s[k])for k in ['p_visible_exact','p_visible_radius2','p_external_support_exact','p_external_support_radius2','p_full_exact','p_full_radius2','p_outside_full_exact','p_outside_full_radius2','max_joint_probability']};v.update({k:s[k]for k in ['logical_exact','logical_radius2','confident_exact','confident_radius2']});v.update({k+'_short':short(s[k])for k in ['logical_exact','logical_radius2','confident_exact','confident_radius2']});v['confidence_U']=s['confidence_U'].lower()=='true'
  case=dict(id=key[0],native_corner=int(key[1]),fixed_case_role='PREEXISTING_NEAR_DIRECT'if position<4 else'PREEXISTING_NEAR_EXTERNAL_TRADEOFF',image=r['image'],image_sha256=r['image_sha256'],query=q,reference=reference,nearest_query_pixel=qp.tolist(),nearest_reference_pixel=rp.tolist(),same_nearest_pixel=bool(np.array_equal(qp,rp)),reference_nearest_pixel_in_query_radius2=bool((np.abs(qp-rp)<=2).all()),error_px=error,roi_side=float(s['roi_side']),human_state=r['human_state'],human_state_short='D'if r['human_state']=='DIRECT_VISIBLE'else'E',human_label_source=r['human_label_source'],annotation_coordinate_source=prov.get('annotation_coordinate_source','NOT_RECORDED_IN_FIXED_NEAR_D_PROVENANCE'),reference_source=prov.get('reference_source','EXISTING_PREPARED_REFERENCE'),reference_coordinate_origin=prov.get('reference_coordinate_origin','existing_frozen_reference; no new point truth created'),reference_overlay_exposure=prov.get('reference_overlay_exposure','UNKNOWN'),model_prediction_exposure=prov.get('model_prediction_exposure','UNKNOWN'),v61_p_visible=float(r['presence_probability']),v63_p_visible=float(o['p_visible']),v63_support_scales='Only original binary p_visible was learned/exported; exact/near/full/Eproxy are unavailable.',v65=v,geometry_self_hidden=s['geometry_self_hidden'],current_exact_pixel_GT='UNAVAILABLE',semantic_corner_GT_must_not_be_substituted_for_pixel_GT=True)
  if position<4:
   original=fixed[position];assert key==(original['id'],str(original['native_corner']));assert abs(case['v61_p_visible']-original['v61_p'])<1e-12 and abs(case['v63_p_visible']-original['new_p'])<1e-12;assert error<=1 and case['human_state_short']=='D';case['semantic_binary_visible_at_fixed_threshold']=v['p_visible_exact']>=.5
  cases.append(case)
 render(cases[:4],OUT/'FIXED_LOW_ERROR_REAL_CROPS.png','Same four preexisting near-visible real cases: v61 / v63 / frozen v65')
 render(cases[4:],OUT/'FIXED_NEAR_EXTERNAL_TRADEOFF_CROPS.png','Same two preexisting near-external real cases: honest semantic tradeoffs')
 summary=dict(schema='POSTSEALED_FIXED_REAL_CROP_TRANSFER_DIAGNOSTIC_V65_V1',selection='Four cases copied unchanged and in-order from preexisting v63 selection file; two external cases explicitly named by parent before reading v65 results. No search for favorable v65 examples.',diagnostic_only=True,current_evaluation_opened_only_after_v65_complete_inference_seal=True,source_caches_models_gate_or_audits_modified=False,new_model_inference=False,GPU_used=False,training=False,threshold_or_model_choice=False,PnP=False,threshold=.5,source_selection_seal_sha256=seal['source_selection_seal']['sha256'],private_inference_seal_sha256=sha(sealpath),input_bindings=bindings,
              four_D_semantic_visible_counts=dict(v61=sum(c['v61_p_visible']>=.5 for c in cases[:4]),v63=sum(c['v63_p_visible']>=.5 for c in cases[:4]),v65_exact=sum(c['v65']['p_visible_exact']>=.5 for c in cases[:4]),v65_radius2_diagnostic=sum(c['v65']['p_visible_radius2']>=.5 for c in cases[:4])),
              caution=['Human semantic D/E/S labels are attached to existing reference native corners, not exact initial-query pixel ownership/full-mask truth. No new pixel GT is created.',
                       'All four D cases use existing manual-click references and query/reference round to the same native pixel. Larger coordinate misses cannot explain these selected remaining errors, but manual boundary uncertainty and strict support-target mismatch remain unresolved.',
                       'Radius2 is separately learned native clipped5x5 support. It is not a replacement semantic threshold/radius selected from current performance.',
                       'O prediction on semantic E is binary nonvisible but does not establish external mechanism. Eproxy is not the human EXTERNAL_OCCLUDED label.',
                       'The six fixed cases are explanatory examples, not an independent holdout estimate. Existing references had overlay exposure; model-prediction exposure was not confirmed.'],
              fixed_preexisting_D_cases=cases[:4],fixed_semantic_E_tradeoff_cases=cases[4:],code_sha256=sha(__file__))
 for p,h in bindings.items():assert sha(p)==h
 (OUT/'FIXED_LOW_ERROR_REAL_CROPS.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(dict(counts=summary['four_D_semantic_visible_counts'],cases=[dict(id=c['id'],k=c['native_corner'],human=c['human_state_short'],v65V=c['v65']['p_visible_exact'],v65near=c['v65']['p_visible_radius2'],confidence=c['v65']['confident_exact_short'])for c in cases]),indent=2))
