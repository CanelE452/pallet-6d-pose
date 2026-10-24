"""Same six preselected real cases, saved outputs only after complete seal."""
from pathlib import Path
import copy,csv,hashlib,json,math
import numpy as np
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;BASE=ROOT.parent
V63=BASE/'query_visibility_public_real_20261022_v63';V65=BASE/'query_visibility_amodal_global_20261022_v65'
KEYS=[('wood_night_01:032494',5),('SQUARE_YOLO:025747',5),('wood_night_01:032376',0),('SQUARE_YOLO:027620',1),('wood_day_01:005249',5),('SQUARE_YOLO:030931',4)]
WINNER_SHA='92ad8e897fd6524d00e62695645394fe3639973b347e2b30cd957ace5d594885'
PAIRS=[('V','V'),('E','V'),('E','E'),('O','V'),('O','E'),('O','O')]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):
 if sha(b['path'])!=b['sha256']:raise AssertionError('Binding changed: '+b['path'])
def keyed(p):
 with Path(p).open(newline='')as f:return{(r['id'],int(r['native_corner'])):r for r in csv.DictReader(f)}
def font(size,bold=False):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans'+('-Bold'if bold else'')+'.ttf',size)
def sealed_scores(path):
 sealpath=path.with_name('INFERENCE_SEAL.json');seal=json.loads(sealpath.read_text())
 if 'completed'in seal:assert seal['completed']is True
 found=[b for b in seal['outputs']if Path(b['path']).resolve()==path.resolve()];assert len(found)==1;check(found[0])
 return sealpath,seal,keyed(path)
def short(s):return dict(VISIBLE_TARGET='V',EXTERNAL_SUPPORT_PROXY='E',OUTSIDE_FULL_SUPPORT='O',UNKNOWN='U')[s]
def extract_model(s):
 fields=['p_visible_exact','p_visible_radius2','p_external_support_exact','p_external_support_radius2','p_outside_full_exact','p_outside_full_radius2','p_full_exact','p_full_radius2','max_joint_probability']
 out={k:float(s[k])for k in fields};out.update({k:s[k]for k in ['logical_exact','logical_radius2','confident_exact','confident_radius2']})
 out['confidence_U']=s['confidence_U'].lower()=='true';out['joint6']=[float(s['joint_p'+str(j)])for j in range(6)];assert abs(sum(out['joint6'])-1)<1e-10
 j=int(np.argmax(out['joint6']));out['joint_MAP_pair']=list(PAIRS[j]);out['joint_MAP_code']=j;out['confident_pair']=[short(s['confident_exact']),short(s['confident_radius2'])]
 return out

def marked(rgb,bounds,q,ref,size,roi=None):
 raw=rgb.crop(tuple(bounds));scale=min(size[0]/raw.width,size[1]/raw.height);hw=(round(raw.width*scale),round(raw.height*scale));a=raw.resize(hw,Image.Resampling.NEAREST);out=Image.new('RGB',size,'#eeeeee');ox=(size[0]-hw[0])//2;oy=(size[1]-hw[1])//2;out.paste(a,(ox,oy));d=ImageDraw.Draw(out)
 point=lambda p:(ox+(p[0]-bounds[0])*scale,oy+(p[1]-bounds[1])*scale)
 if roi is not None:
  x0,y0=point(roi[:2]);x1,y1=point(roi[2:]);d.rectangle((x0,y0,x1,y1),outline='#51e17c',width=3)
 x,y=point(q);d.ellipse((x-8,y-8,x+8,y+8),outline='#ffe100',width=3)
 x,y=point(ref);d.line((x-6,y-6,x+6,y+6),fill='#02dcff',width=2);d.line((x-6,y+6,x+6,y-6),fill='#02dcff',width=2)
 return out

def render(cases,path,title):
 width,rowh,header=2280,452,190;canvas=Image.new('RGB',(width,header+len(cases)*rowh+132),'white');d=ImageDraw.Draw(canvas)
 d.text((20,12),title,font=font(29,True),fill='#192f3e')
 d.text((20,54),'Yellow circle = unchanged initial query; cyan X = existing semantic reference; green rectangle = nominal local ROI side L.',font=font(21),fill='#222222')
 d.text((20,85),'96px is a diagnostic view. Local nine-grid span is 8L/9; global target context is also used by v65/v66 and is not shown here.',font=font(20),fill='#444444')
 d.text((20,116),'Saved support probabilities are not current pixel GT. Human D/E is a semantic-reference label; E/O model states do not prove a physical cause.',font=font(20),fill='#8a4b17')
 for x,text in [(20,'Image context'),(280,'Native96 diagnostic (3x)'),(592,'Local ROI L (rescaled)'),(915,'Saved probabilities / fixed confidence decoder')]:d.text((x,151),text,font=font(20,True),fill='#263d4d')
 for i,c in enumerate(cases):
  y=header+i*rowh;q=c['query'];ref=c['reference'];L=c['roi_side'];rgb=Image.open(c['image']).convert('RGB');assert sha(c['image'])==c['image_sha256']
  d.text((20,y+5),c['id']+' / k'+str(c['native_corner'])+'   human semantic='+c['human_state_short'],font=font(23,True),fill='#172733')
  thumb=rgb.resize((238,178),Image.Resampling.BILINEAR);td=ImageDraw.Draw(thumb);sx,sy=238/rgb.width,178/rgb.height
  rb=c['local_nominal_roi_xyxy'];db=c['diagnostic96_bounds_xyxy']
  td.rectangle((db[0]*sx,db[1]*sy,db[2]*sx,db[3]*sy),outline='#ef61dd',width=2)
  td.rectangle((rb[0]*sx,rb[1]*sy,rb[2]*sx,rb[3]*sy),outline='#51e17c',width=2);canvas.paste(thumb,(20,y+62))
  canvas.paste(marked(rgb,db,q,ref,(288,288),rb if c['local_roi_fits_diagnostic96']else None),(280,y+53))
  canvas.paste(marked(rgb,c['local_roi_integer_crop_xyxy'],q,ref,(288,288),rb),(592,y+53))
  d.text((20,y+254),'magenta = native96',font=font(17),fill='#9a3288');d.text((20,y+280),'green = nominal L',font=font(17),fill='#246d3b')
  d.text((20,y+311),'manual error '+f"{c['error_px']:.3f}px",font=font(18),fill='#444444')
  d.text((20,y+339),'same rounded pixel: '+str(c['same_nearest_pixel']),font=font(16),fill='#444444')
  d.text((280,y+351),'96 native pixels; L='+f'{L:.2f}'+'px',font=font(18),fill='#444444')
  d.text((592,y+351),'integer extent '+str(c['local_roi_integer_crop_xyxy'][2]-c['local_roi_integer_crop_xyxy'][0])+' x '+str(c['local_roi_integer_crop_xyxy'][3]-c['local_roi_integer_crop_xyxy'][1])+'px',font=font(18),fill='#444444')
  if not c['local_roi_fits_diagnostic96']:d.text((280,y+379),'L exceeds96; complete L shown right',font=font(16),fill='#a35719')
  else:d.text((280,y+379),'complete local L rectangle fits here',font=font(16),fill='#246d3b')
  x=915;top=y+57;col=[x,x+96,x+232,x+355,x+581,x+809,x+980]
  headers=['Model','pV exact*','pV r2','pE exact / r2','pO exact / r2','max joint','conf pair']
  for xx,t in zip(col,headers):d.text((xx,top),t,font=font(18,True),fill='#334a59')
  d.line((x,top+33,width-20,top+33),fill='#cdd9e0')
  for ri,version in enumerate(('v61','v63','v65','v66')):
   yy=top+45+ri*48;color='#164f87'if version=='v66'else'#222222';bold=version=='v66';m=c['models'][version]
   vals=[version,f"{m['p_visible_saved_binary']:.3f}"if version in ('v61','v63')else f"{m['p_visible_exact']:.3f}"]
   if version in ('v61','v63'):vals+=['N/A','N/A','N/A','N/A','N/A']
   else:vals+=[f"{m['p_visible_radius2']:.3f}",f"{m['p_external_support_exact']:.3f} / {m['p_external_support_radius2']:.3f}",f"{m['p_outside_full_exact']:.3f} / {m['p_outside_full_radius2']:.3f}",f"{m['max_joint_probability']:.3f}",'/'.join(m['confident_pair'])]
   for xx,t in zip(col,vals):d.text((xx,yy),t,font=font(22,bold),fill=color)
  d.text((x,top+249),'* v61/v63: only saved binary pV; no exported r2/E/O/joint.',font=font(18),fill='#555555')
  for j,version in enumerate(('v65','v66')):
   m=c['models'][version];text=version+' joint MAP '+('/'.join(m['joint_MAP_pair']))+'; max '+f"{m['max_joint_probability']:.3f}"+'; confidence U='+str(m['confidence_U'])
   d.text((x,top+282+j*28),text,font=font(18),fill='#164f87'if version=='v66'else'#444444')
  d.text((915,y+407),'q=('+', '.join(f'{v:.3f}'for v in q)+'); reference=('+', '.join(f'{v:.3f}'for v in ref)+')',font=font(17),fill='#444444')
  d.line((20,y+rowh-2,width-20,y+rowh-2),fill='#cccccc')
 footer=header+len(cases)*rowh
 for j,line in enumerate(['No current pixel-visible/full-support GT is available. Existing reference coordinates and human semantic labels are displayed separately.',
  'O/V or exact/r2 differences do not establish coordinate error or physical target absence. Boundary/target mismatch and transfer remain unresolved.',
  'Same six cases fixed before v66 scores. Saved outputs only; no new model, GPU, PnP, fit, threshold or candidate selection.']):d.text((20,footer+16+32*j),line,font=font(20),fill='#3e4e59')
 canvas.save(path)

def main():
 out=HERE/'FIXED_REAL_CROPS_V66.json'
 if out.exists():raise FileExistsError('Preserve completed fixed case diagnostic')
 score=ROOT/'private_evaluation/CONTROLLED_PLUS_PALLOX/REAL_QUERY_SCORES.csv';sealpath,seal,new=sealed_scores(score)
 assert seal['candidate']=='CONTROLLED_PLUS_PALLOX'and seal['source_selection_seal']['sha256']==WINNER_SHA;check(seal['source_selection_seal'])
 for key in ('new_optimizer_updates','new_PnP_calls','new_backbone_forward','SAM_forwards','real_RGB_images_opened'):assert seal[key]==0
 previous=V65/'diagnostics/FIXED_LOW_ERROR_REAL_CROPS.json';old=json.loads(previous.read_text());fixed=old['fixed_preexisting_D_cases']+old['fixed_semantic_E_tradeoff_cases'];assert[(r['id'],r['native_corner'])for r in fixed]==KEYS
 oldscore=V65/'private_evaluation/MIXED_GLOBAL_UNWEIGHTED/REAL_QUERY_SCORES.csv';oldsealpath,oldseal,old65=sealed_scores(oldscore)
 legacy=V63/'private_evaluation/MIXED_CONTEXT_RGB_SAM_V4/REAL_QUERY_SCORES.csv';legacyseal,_,old63=sealed_scores(legacy)
 for p,h in old['input_bindings'].items():assert sha(p)==h
 (HERE/'fixed_native96').mkdir(exist_ok=False);cases=[]
 for oldcase in fixed:
  c=copy.deepcopy(oldcase);key=(c['id'],c['native_corner']);s=new[key];s65=old65[key];s63=old63[key]
  assert s['valid'].lower()=='true';assert np.array_equal(c['query'],[float(s['query_x']),float(s['query_y'])]);assert c['image_sha256']==s['image_sha256']==s65['image_sha256']==s63['image_sha256'];assert c['roi_side']==float(s['roi_side'])==float(s65['roi_side'])
  q=c['query'];px,py=c['nearest_query_pixel'];L=c['roi_side'];bounds=[px-48,py-48,px+48,py+48];roi=[q[0]-L/2,q[1]-L/2,q[0]+L/2,q[1]+L/2]
  c.update(diagnostic96_bounds_xyxy=bounds,diagnostic96_native_pixels=96,local_nominal_roi_xyxy=roi,local_roi_integer_crop_xyxy=[math.floor(roi[0]),math.floor(roi[1]),math.ceil(roi[2]),math.ceil(roi[3])],local_nine_grid_center_span_px=8*L/9,
   local_roi_fits_diagnostic96=bool(roi[0]>=bounds[0]and roi[1]>=bounds[1]and roi[2]<=bounds[2]and roi[3]<=bounds[3]),global_context_also_input_not_shown_in_local_crop=True)
  c['models']=dict(v61=dict(p_visible_saved_binary=c['v61_p_visible'],p_visible_radius2=None,p_external_support_exact=None,p_outside_full_exact=None,max_joint_probability=None,unavailable='Only original saved binary pV; r2/E/O/joint not learned/exported'),
   v63=dict(p_visible_saved_binary=float(s63['p_visible']),p_visible_radius2=None,p_external_support_exact=None,p_outside_full_exact=None,max_joint_probability=None,unavailable='Only original saved binary pV; r2/E/O/joint not learned/exported'),v65=extract_model(s65),v66=extract_model(s))
  assert c['models']['v63']['p_visible_saved_binary']==c['v63_p_visible'];assert c['models']['v65']['p_visible_exact']==c['v65']['p_visible_exact']
  assert sha(c['image'])==c['image_sha256'];rgb=Image.open(c['image']).convert('RGB');raw=HERE/'fixed_native96'/(c['id'].replace(':','_')+'_k'+str(c['native_corner'])+'_RAW96.png');rgb.crop(tuple(bounds)).save(raw)
  c['raw_native96']=bind(raw);c['raw_native96_has_overlaid_markers']=False;cases.append(c)
 render(cases[:4],HERE/'FIXED_D_REAL_CROPS_V66.png','Same four preexisting near-visible cases: v61 / v63 / v65 / v66')
 render(cases[4:],HERE/'FIXED_E_REAL_CROPS_V66.png','Same two preexisting semantic external cases: support-state tradeoffs')
 bindings=[bind(p)for p in [previous,score,sealpath,oldscore,oldsealpath,legacy,legacyseal]]
 for b in bindings:check(b)
 report=dict(schema='V66_POSTSEALED_FIXED_SAME_REAL_CASE_CROP_DIAGNOSTIC',selection='Exact same ordered four semanticD/two semanticE cases as v65, named by parent before v66 diagnostic read; no score-based reselection',
  current_evaluation_read_only_after_complete_inference_seal=True,input_bindings=bindings,source_selection_seal=seal['source_selection_seal'],fixed_threshold=.5,
  new_model_inference_calls=0,GPU_used=False,new_training=False,new_PnP_calls=0,model_or_threshold_selection=False,old_artifacts_modified=False,
  current_exact_pixel_visible_full_GT='UNAVAILABLE',human_D_E_reference_is_semantic_not_pixel_state=True,reference_provenance='Unchanged existing manual/reference coordinate metadata from bound v65 diagnostic; prior overlay exposure retained, model-prediction exposure not confirmed.',
  crop_semantics='Raw96x96 centered at floor(query+.5), shown3xnearest in figures. Green nominallocalROI rectangle only in96whenfullyinside; separatecomplete L crop alwaysshown. Actual nine-gridcenter span8L/9. RGB5x5 neighborhoods/global target context extend modelinputs beyonda single center sample.',
  interpretations_prohibited=['Exact O versus radius2 V is not verified coordinate-error diagnosis.','Model O is source-full support category, not verified physical target absence.','Model E/source-support probability is not human semantic external cause GT.','The six explanatory fixed cases are not a fresh holdout performance estimate.'],
  four_D_binary_visible_counts=dict(v61=sum(c['models']['v61']['p_visible_saved_binary']>=.5 for c in cases[:4]),v63=sum(c['models']['v63']['p_visible_saved_binary']>=.5 for c in cases[:4]),
   v65_exact=sum(c['models']['v65']['p_visible_exact']>=.5 for c in cases[:4]),v65_radius2=sum(c['models']['v65']['p_visible_radius2']>=.5 for c in cases[:4]),v66_exact=sum(c['models']['v66']['p_visible_exact']>=.5 for c in cases[:4]),v66_radius2=sum(c['models']['v66']['p_visible_radius2']>=.5 for c in cases[:4])),
  fixed_semantic_D_cases=cases[:4],fixed_semantic_E_cases=cases[4:],figures=[bind(HERE/n)for n in ['FIXED_D_REAL_CROPS_V66.png','FIXED_E_REAL_CROPS_V66.png']],code=bind(__file__))
 out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(report=bind(out),counts=report['four_D_binary_visible_counts'],cases=[dict(id=c['id'],k=c['native_corner'],v66=c['models']['v66'])for c in cases]),indent=2))

if __name__=='__main__':main()
