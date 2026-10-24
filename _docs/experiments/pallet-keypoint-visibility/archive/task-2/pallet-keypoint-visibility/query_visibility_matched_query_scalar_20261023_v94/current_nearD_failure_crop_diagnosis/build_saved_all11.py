"""All11 fixed manual-near humanD misses. Saved RGB96 only, no fullRGB/P3/model."""
from pathlib import Path
import csv,hashlib,json,math
import numpy as np
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
V94=HERE.parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def binding(p):return dict(path=str(p),sha256=sha(p))
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def readcsv(p):
 with Path(p).open(newline='')as f:return list(csv.DictReader(f))
def key(r):return r['id'],int(r['native_corner'])
def yes(v):return v is True or str(v).lower()=='true'
def save(n,o):(HERE/n).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def xy_to_crop(xy,q,L):return (np.asarray(xy,np.float64)-np.asarray(q,np.float64))/float(L)*96+47.5

def generated():
 q=np.array([123.125,45.25]);L=25.77
 for j in [0,47.5,95]:
  xy=q+(j-47.5)*L/96;assert np.allclose(xy_to_crop(xy,q,L),j,rtol=0,atol=1e-12)
 assert np.array_equal(xy_to_crop(q,q,L),[47.5,47.5])
 a=np.arange(96*96*3,dtype=np.uint32).reshape(96,96,3).astype(np.uint8)
 two=np.asarray(Image.fromarray(a).resize((192,192),Image.Resampling.NEAREST))
 assert np.array_equal(two,np.repeat(np.repeat(a,2,axis=0),2,axis=1))
 save('GENERATED_MARKER_AND_DISPLAY_CHECK.json',dict(PASS=True,code=binding(__file__),crop_phase='q+((j+.5)/96-.5)*L',marker_inverse_exact=True,nearest2x_exact=True,source_pixel_or_model_reads=0))

def font(size):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)
def make_panel(case,rgb):
 p=Image.new('RGB',(420,390),'#ffffff');d=ImageDraw.Draw(p)
 def text(y,s,size=14,color='#151b26'):d.text((12,y),s,font=font(size),fill=color)
 text(8,f"{case['case_number']:02d} {case['group']} k{case['native_corner']}",16)
 # Keep original identifier readable on two deterministic lines.
 ident=case['id'];text(31,ident[:45],12)
 if len(ident)>45:text(47,ident[45:],12)
 text(66,f"human D | error {case['q_error_px']:.3f}px | L {case['original_L']:.2f}px")
 a,b,c=case['saved_models']['v61'],case['saved_models']['V91'],case['saved_models']['C1']
 text(87,f"v61 {a['decision']} pV={a['p_visible']:.3f} | V91 {b['decision']} z={b['raw_hidden_logit']:.3f}",12)
 text(105,f"C1 HIDDEN z={c['raw_hidden_logit']:.3f} | crop valid {case['crop_valid_fraction']:.3f}",13)
 text(131,'Saved RGB96 (unmarked)',11);text(131,'2x NEAREST, q/r markers',11) if False else None
 d.text((138,131),'2x NEAREST, q/r markers',font=font(11),fill='#151b26')
 p.paste(Image.fromarray(rgb),(12,151));big=Image.fromarray(rgb).resize((192,192),Image.Resampling.NEAREST);bd=ImageDraw.Draw(big)
 q=2*np.asarray(case['query_crop_xy'])+.5;r=2*np.asarray(case['reference_crop_xy'])+.5
 # q cyan cross, manual r magenta ring; no masks or inferred parts.
 bd.line((q[0]-6,q[1],q[0]+6,q[1]),fill='#00dfff',width=2);bd.line((q[0],q[1]-6,q[0],q[1]+6),fill='#00dfff',width=2)
 bd.ellipse((r[0]-5,r[1]-5,r[0]+5,r[1]+5),outline='#ff37d4',width=2)
 p.paste(big,(138,151))
 text(351,'cyan + q | magenta o existing manual r',12)
 text(370,'D/HIDDEN = saved binary response, no cause inference.',11)
 return p

def run():
 generated()
 packet_path=V94/'CURRENT_ARTIFACT_BINDINGS_AFTER_C1_SOURCE_AUDIT.json';packet=json.loads(packet_path.read_text());b=packet['bindings']
 inf=V94/'private_evaluation/C1_NATURAL_KNOWNSET_ONE_SHOT';ev=V94/'current_evaluation/C1_NATURAL_KNOWNSET_ONE_SHOT'
 seal=json.loads((inf/'INFERENCE_SEAL.json').read_text());es=json.loads((ev/'EVALUATION_SEAL.json').read_text());assert seal['completed']and es['completed']and seal['model_unchanged']
 current=readcsv(inf/'REAL_QUERY_SCORES.csv');manual=readcsv(b['manual_provenance']['path']);v91={key(r):r for r in readcsv(b['baseline_v91_scores']['path'])}
 eligible={key(r):r for r in manual if r['annotation_coordinate_source']=='manual_click'and r['human_state']=='DIRECT_VISIBLE'and yes(r['query_valid'])and float(r['initial_reference_error_px'])<=3}
 assert len(eligible)==153+223
 selections=[]
 for index,r in enumerate(current):
  k=key(r)
  if k not in eligible or not yes(r['valid'])or float(r['raw_query_aux_hidden_logit'])<0:continue
  m=eligible[k];old=v91[k]
  assert r['authoritative_decision']=='E'and m['image_sha256']==r['image_sha256']==old['image_sha256']
  for f in ['query_x','query_y']:assert float(m[f])==float(r[f])==float(old[f])
  q=[float(r['query_x']),float(r['query_y'])];ref=[float(m['human_reference_x']),float(m['human_reference_y'])];L=float(r['roi_side'])
  err=math.dist(q,ref);assert abs(err-float(m['initial_reference_error_px']))<1e-8
  assert L==float(old['roi_side'])
  c=dict(case_number=len(selections)+1,original_row_index=index,id=r['id'],native_corner=int(r['native_corner']),group=m['group'],image_sha256=r['image_sha256'],original_q=q,original_L=L,manual_reference_xy=ref,q_error_px=err,
   query_crop_xy=xy_to_crop(q,q,L).tolist(),reference_crop_xy=xy_to_crop(ref,q,L).tolist(),human_state='DIRECT_VISIBLE',annotation_coordinate_source=m['annotation_coordinate_source'],reference_source=m['reference_source'],reference_coordinate_origin=m['reference_coordinate_origin'],annotation_uncertainty_bbox=None,annotation_uncertainty_note='No explicit uncertainty interval/bbox in bound manual provenance; none invented.',
   saved_models={'v61':dict(decision='D'if yes(m['present_at_fixed_0_5'])else'HIDDEN',p_visible=float(m['presence_probability']),raw_decision_source='stored present_at_fixed_0_5; no invented rawlogit'),
    'V91':dict(decision='D'if float(old['raw_physical_external_logit'])<0 else'HIDDEN',raw_hidden_logit=float(old['raw_physical_external_logit']),p_visible=float(old['p_visible'])),
    'C1':dict(decision='HIDDEN',raw_hidden_logit=float(r['raw_query_aux_hidden_logit']),p_visible=float(r['p_visible']))})
  selections.append(c)
 assert len(selections)==11 and sum(c['group']=='WOOD_PRIMARY105'for c in selections)==4 and sum(c['group']=='SQUARE118'for c in selections)==7
 save('ALL11_SELECTION_BEFORE_CROP_BODY.json',dict(completed=True,selection='All manual_click valid humanD error<=3 with C1 valid rawlogit>=0, original output order, no replacement.',all11=selections,bindings=[binding(packet_path),binding(inf/'INFERENCE_SEAL.json'),binding(ev/'EVALUATION_SEAL.json'),binding(inf/'REAL_QUERY_SCORES.csv'),b['manual_provenance'],b['baseline_v91_scores']],crop_body_read_at_this_selection=False))
 # Only these four members are deserialized. p3_roi/box/checkpoint/fullRGB never loaded.
 with np.load(b['current_model_input']['path'],allow_pickle=False)as z:
  rgb=z['rgb96'];valid=z['crop_valid96'];q=z['query_xy'];L=z['side_L']
 assert rgb.shape==(1784,3,96,96)and rgb.dtype==np.uint8 and valid.shape==(1784,1,96,96)and valid.dtype==np.bool_
 images=[];selected_rgb=[];selected_valid=[]
 for c in selections:
  i=c['original_row_index'];assert np.array_equal(q[i],np.asarray(c['original_q'],np.float32))and L[i]==np.float32(c['original_L'])
  image=np.transpose(rgb[i],(1,2,0)).copy();images.append(image);selected_rgb.append(rgb[i]);selected_valid.append(valid[i])
  c['model_rgb96_CHW_uint8_sha256']=array_sha(rgb[i]);c['display_rgb96_HWC_uint8_sha256']=array_sha(image);c['crop_valid96_sha256']=array_sha(valid[i]);c['crop_valid_fraction']=float(valid[i].mean());c['original_query_and_L_unchanged']=True
  name=f"case_{c['case_number']:02d}_ORIGINAL_UNMARKED_RGB96.png";Image.fromarray(image).save(HERE/name);c['original96_file']=binding(HERE/name)
 np.savez_compressed(HERE/'ALL11_SAVED_RGB96_AND_METADATA.npz',rgb96=np.stack(selected_rgb),crop_valid96=np.stack(selected_valid),original_row_index=np.array([c['original_row_index']for c in selections]),query_xy=np.array([c['original_q']for c in selections]),side_L=np.array([c['original_L']for c in selections]),manual_reference_xy=np.array([c['manual_reference_xy']for c in selections]))
 figurepaths=[]
 for no,start in enumerate([0,8],1):
  end=min(start+8,11);rows=math.ceil((end-start)/4);fig=Image.new('RGB',(4*420,100+rows*390),'#e9edf3');d=ImageDraw.Draw(fig)
  d.text((14,10),f'V94 C1: all 11 manual-near humanD misses | fixed cases {start+1}-{end}',font=font(24),fill='#101727')
  d.text((14,43),'Original cached RGB96 and fixed 2x display; q/L/r unchanged. No fullRGB decode, recenter or model call.',font=font(16),fill='#202a38')
  d.text((14,69),'Human manual D evidence; C1 auxiliary hidden is not a learned occlusion cause. Root has not viewed this new gallery yet.',font=font(14),fill='#202a38')
  for j in range(start,end):fig.paste(make_panel(selections[j],images[j]),((j-start)%4*420,100+(j-start)//4*390))
  path=HERE/('CONTACT_01_ALL8.png'if no==1 else'CONTACT_02_REMAINING3.png');fig.save(path);figurepaths.append(binding(path))
 save('ALL11_CASES_AND_PROVENANCE.json',dict(completed=True,all11=selections,contacts=figurepaths,model_input_npz_binding=b['current_model_input'],selected_members_only=['rgb96','crop_valid96','query_xy','side_L'],metadata_transform='u=96*((x-q)/L+.5)-.5; display2x=2*(u+.5)-.5',source8_public3_vs_current11='Different task/label definitions. These are actual human manualD/current unchangedq errors; source C1 had weak conditional auxiliary cue knownset/native association UNKNOWN. No same-label or equivalent evaluation-unit claim.',new_RGB_decode_or_P3_body_or_model_or_forward_fit_PnP=0,root_viewed_this_new_gallery=False))
 print(json.dumps({'completed':True,'all11':[(c['case_number'],c['id'],c['native_corner'],round(c['q_error_px'],6))for c in selections],'contacts':figurepaths},indent=2))

if __name__=='__main__':run()
