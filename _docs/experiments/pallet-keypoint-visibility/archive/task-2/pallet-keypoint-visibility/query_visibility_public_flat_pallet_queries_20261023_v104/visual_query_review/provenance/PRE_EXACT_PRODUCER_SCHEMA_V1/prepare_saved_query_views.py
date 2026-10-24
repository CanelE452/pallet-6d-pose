"""Inert score-free saved-query visualizer. No detector/model/GT imports or calls."""
import argparse,ast,hashlib,json,pathlib,sys,tempfile
sys.dont_write_bytecode=True
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=pathlib.Path(__file__).resolve().parents[1];HERE=pathlib.Path(__file__).resolve().parent
P=ROOT.parent
MATH=P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'
FORBIDDEN={'score','scores','confidence','conf','selected_detection_score','raw_score','prediction_score','gt','labels','reference_xy','mask','pose'}
FRAME_KEYS={'frame_index','input_id','original_image','raw_hw','canonical_RGB','decoded_RGB_sha256','selected_bbox_xyxy','initial_points','L8','no_detection'}
# Binding final input-only names may be updated before preparation freeze; no actual manifest read now.
SLOT_CORE={'frame_index','input_id','native_slot','query_xy','roi_side','original_input_valid','reason'}
SLOT_EXTRA={'query_native_continuous_valid','query_native_nearest_valid','query_rgb96_centre_domain_valid','native_continuous_valid','native_nearest_valid','continuous_valid','nearest_valid','rgb96_centre_valid','query_continuous_valid','query_nearest_valid','query_rgb96_center_domain_valid'}
def sha(b):return hashlib.sha256(b).hexdigest()
def bind(f):return {'path':str(f.resolve()),'sha256':sha(f.read_bytes()),'bytes':f.stat().st_size}
def check(b):
 f=pathlib.Path(b['path']);assert sha(f.read_bytes())==b['sha256'];return f
def write(f,x):assert not f.exists(),str(f);f.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def pure_crop():
 tree=ast.parse(MATH.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='crop')
 ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(MATH)+'::crop_only','exec'),ns)
 return ns['crop'],sha(ast.dump(fn,include_attributes=False).encode())
def no_scores(x):
 if isinstance(x,dict):
  for k,v in x.items():
   assert k.lower() not in FORBIDDEN,f'Forbidden score/GT key: {k}'
   no_scores(v)
 elif isinstance(x,list):
  for v in x:no_scores(v)
def seal_contains(x,b):
 if isinstance(x,dict):
  if x.get('path')==b['path'] and x.get('sha256')==b['sha256']:return True
  return any(seal_contains(v,b) for v in x.values())
 if isinstance(x,list):return any(seal_contains(v,b) for v in x)
 return False
def load_rows(framefile,slotfile):
 frames=[json.loads(x) for x in framefile.read_text().splitlines() if x.strip()];slots=[json.loads(x) for x in slotfile.read_text().splitlines() if x.strip()]
 assert len(frames)==4 and len(slots)==32
 for i,f in enumerate(frames):
  no_scores(f);assert set(f)==FRAME_KEYS and f['frame_index']==i
  assert len(f['initial_points'])==len(f['L8'])==8
  assert len(f['raw_hw'])==2
  for k in range(8):
   s=slots[8*i+k];no_scores(s);assert SLOT_CORE<=set(s) and set(s)<=SLOT_CORE|SLOT_EXTRA
   assert s['frame_index']==i and s['input_id']==f['input_id'] and s['native_slot']==k
   assert s['query_xy']==f['initial_points'][k] and s['roi_side']==f['L8'][k]
   q,L=s['query_xy'],s['roi_side']
   assert (q is None)==(L is None)
   if q is None:assert not s['original_input_valid']
   else:assert np.asarray(q).shape==(2,) and np.isfinite(q).all() and np.isfinite(L) and L>0
 return frames,slots
def mark96(im):
 # q lies at native crop array coordinate(47.5,47.5), not GT/reference location.
 out=im.copy();d=ImageDraw.Draw(out);c=47.5
 d.line((c-6,c,c+6,c),fill=(255,40,40),width=1);d.line((c,c-6,c,c+6),fill=(255,40,40),width=1)
 return out
def contact(crops,slots,frame_index,out,marked=False):
 iw,ih=324,256;canvas=Image.new('RGB',(4*iw,2*ih+68),'white');d=ImageDraw.Draw(canvas)
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
 d.text((10,8),f'Frame{frame_index}: all8 original q/L | '+('CROSSHAIR DISPLAY ONLY' if marked else 'UNMARKED 96 + 2x NEAREST'),fill='black',font=font)
 for k in range(8):
  x=(k%4)*iw;y=36+(k//4)*ih;s=slots[k];a=Image.fromarray(crops[k].transpose(1,2,0))
  if marked:a=mark96(a)
  d.text((x+6,y),f'logical slot{k} | valid={s["original_input_valid"]}',fill='black',font=small)
  if s['query_xy'] is not None:
   q=s['query_xy'];d.text((x+6,y+18),f'q=({q[0]:.2f},{q[1]:.2f}) L={s["roi_side"]:.2f}',fill='black',font=small)
  else:d.text((x+6,y+18),'NO QUERY | placeholder only',fill='black',font=small)
  canvas.paste(a,(x+6,y+45));canvas.paste(a.resize((192,192),Image.Resampling.NEAREST),(x+112,y+45))
 d.text((10,2*ih+43),'Logical slots are initializer order, NOT verified physical/native cue IDs. No score or D/E label displayed.',fill='#56616f',font=small)
 canvas.save(out)
def render(frames,slots,out):
 out.mkdir(exist_ok=False);crop,_=pure_crop();allrgb=np.zeros((4,8,3,96,96),np.uint8);allvalid=np.zeros((4,8,1,96,96),bool);qraw=np.full((4,8,2),np.nan,np.float64);Lraw=np.full((4,8),np.nan,np.float64);ledger=[]
 for i,f in enumerate(frames):
  cf=check(f['canonical_RGB']);rgb=np.load(cf,allow_pickle=False);assert rgb.dtype==np.uint8 and rgb.shape==(*f['raw_hw'],3)
  assert sha(rgb.tobytes())==f['decoded_RGB_sha256'];im=Image.fromarray(rgb);im.save(out/f'FRAME_{i:02d}_UNMARKED.png')
  marked=im.copy();d=ImageDraw.Draw(marked);box=f['selected_bbox_xyxy']
  if box is not None:
   assert len(box)==4 and np.isfinite(box).all();d.rectangle(box,outline=(40,170,255),width=2)
  for k,s in enumerate(slots[i*8:i*8+8]):
   q,L=s['query_xy'],s['roi_side']
   if q is not None:
    qraw[i,k]=q;Lraw[i,k]=L;allrgb[i,k],allvalid[i,k]=crop(rgb,q,L,96)
    # finite invalid-centre queries keep their factual intersecting crop.
    d.line((q[0]-4,q[1],q[0]+4,q[1]),fill=(255,50,50),width=1);d.line((q[0],q[1]-4,q[0],q[1]+4),fill=(255,50,50),width=1);d.text((q[0]+5,q[1]+2),str(k),fill=(255,255,255))
   ledger.append({**s,'slot_order_index':i*8+k,'source_frame':{'frame_index':i,'canonical_RGB':f['canonical_RGB'],'decoded_RGB_sha256':f['decoded_RGB_sha256']},'crop_RGB96_bytes_sha256':sha(allrgb[i,k].tobytes()),'crop_valid96_bytes_sha256':sha(allvalid[i,k].tobytes()),'crop_valid_fraction':float(allvalid[i,k].mean()),'semantic_label':None,'native_association':'UNKNOWN','evidence_review_status':'NOT_YET_VISUALLY_REVIEWED','physical_structural_cue_near_q':None,'potential_foreign_overlap':None,'floor_or_wrong_object':None,'target_identity_ambiguity':None,'observation':None})
  marked.save(out/f'FRAME_{i:02d}_BBOX_AND_Q_DISPLAY_ONLY.png')
  contact(allrgb[i],slots[i*8:i*8+8],i,out/f'FRAME_{i:02d}_CROPS8_UNMARKED.png')
  contact(allrgb[i],slots[i*8:i*8+8],i,out/f'FRAME_{i:02d}_CROPS8_CROSSHAIR_DISPLAY_ONLY.png',True)
 np.savez_compressed(out/'ALL32_EXACT_RGB96_AND_VALID.npz',rgb96=allrgb,crop_valid96=allvalid,query_xy_raw=qraw,L_raw=Lraw)
 write(out/'ALL32_VIEW_PROVENANCE_AND_REVIEW_TEMPLATE.json',{'completed':True,'rows':ledger,'no_observation_or_labels_auto_generated':True,'model_score_reads':False,'q_L_unchanged':True,'crop_phase':'Native q+(j+.5)/96*L-.5*L; q array-centre47.5. Bilinear uint8 round/zero padding and native RGB valid[-.5,W-.5).','fullframe_markers_and_crop_crosshair':'Display-only; unmarked versions saved separately.','no_model_GTinference_fit_current_PnP':True})
 return ledger

def generated(output):
 crop,ash=pure_crop();a=np.zeros((12,15,3),np.uint8);a[...,0]=np.arange(15);a[...,1]=np.arange(12)[:,None]
 tests=0
 for q in [[7.5,5.5],[-1.,5.],[15.2,7.2],[-.4,0.1]]:
  for L in [4.,16.]:
   b,v=crop(a,q,L,96)
   for y,x in [(0,0),(47,47),(48,48),(95,95),(0,95),(95,0)]:
    sx=q[0]+((x+.5)/96-.5)*L;sy=q[1]+((y+.5)/96-.5)*L;valid=(-.5<=sx<14.5 and -.5<=sy<11.5)
    val=np.zeros(3)
    for xx in [int(np.floor(sx)),int(np.floor(sx))+1]:
     for yy in [int(np.floor(sy)),int(np.floor(sy))+1]:
      if 0<=xx<15 and 0<=yy<12:val+=a[yy,xx].astype(float)*max(0,1-abs(sx-xx))*max(0,1-abs(sy-yy))
    val=np.floor(np.clip(val,0,255)+.5).astype(np.uint8) if valid else np.zeros(3,np.uint8)
    assert bool(v[0,y,x])==valid and np.array_equal(b[:,y,x],val);tests+=1
 b,v=crop(a,[-1.,5.],16.,96);assert v.any() and b.any(),'finite OOB may not be zeroed'
 try:no_scores({'scores':[.9]})
 except AssertionError:pass
 else:raise AssertionError('score key was accepted')
 with tempfile.TemporaryDirectory() as td:
  td=pathlib.Path(td);cf=td/'generated.npy';np.save(cf,a);frames=[];slots=[]
  for i in range(4):
   qs=[[7.5,5.5],[-1.,5.]]+[None]*6;Ls=[4.,16.]+[None]*6
   frames.append({'frame_index':i,'input_id':f'generated{i}','original_image':{'path':'GENERATED_ONLY','sha256':'0'*64},'raw_hw':[12,15],'canonical_RGB':bind(cf),'decoded_RGB_sha256':sha(a.tobytes()),'selected_bbox_xyxy':[1,1,10,10],'initial_points':qs,'L8':Ls,'no_detection':False})
   for k in range(8):slots.append({'frame_index':i,'input_id':f'generated{i}','native_slot':k,'query_xy':qs[k],'roi_side':Ls[k],'original_input_valid':k==0,'reason':'GENERATED'})
  ff=td/'frames.jsonl';ss=td/'slots.jsonl';ff.write_text(''.join(json.dumps(x)+'\n' for x in frames));ss.write_text(''.join(json.dumps(x)+'\n' for x in slots));frames,slots=load_rows(ff,ss);render(frames,slots,td/'views');z=np.load(td/'views/ALL32_EXACT_RGB96_AND_VALID.npz');assert z['rgb96'].shape==(4,8,3,96,96);assert z['crop_valid96'][:,1].any() and not z['crop_valid96'][:,2:].any()
 write(output,{'PASS':True,'code':bind(pathlib.Path(__file__).resolve()),'math_code':bind(MATH),'crop_AST_sha256':ash,'scalar_sample_checks':tests,'generated_render_all32':True,'finite_invalid_crop_retained':True,'absent_query_placeholder_only':True,'score_rejection':True,'actual_source_RGB_q_scores_model_current_reads':0})
def actual(args):
 sealfile=check({'path':args.seal,'sha256':args.seal_sha});s=json.loads(sealfile.read_text());assert s.get('completed') is True
 fb={'path':str(pathlib.Path(args.frames).resolve()),'sha256':args.frames_sha};sb={'path':str(pathlib.Path(args.queries).resolve()),'sha256':args.queries_sha}
 assert seal_contains(s,fb) and seal_contains(s,sb),'Both score-free manifests must be exact bound by completed initializer seal'
 framesfile=check(fb);slotsfile=check(sb);frames,slots=load_rows(framesfile,slotsfile)
 out=pathlib.Path(args.output);render(frames,slots,out)
 write(out/'VISUALIZATION_COMPLETE.json',{'completed':True,'code':bind(pathlib.Path(__file__).resolve()),'source_seal':bind(sealfile),'frames':bind(framesfile),'queries':bind(slotsfile),'generated':bind(HERE/'GENERATED_VISUALIZER_TESTS.json'),'files':[bind(f) for f in sorted(out.iterdir()) if f.is_file()],'score_body_or_GT_reads':False,'actual_visual_observations_written':False,'no_model_fit_current_PnP':True})
if __name__=='__main__':
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True);g=sub.add_parser('generated');g.add_argument('--output',type=pathlib.Path,default=HERE/'GENERATED_VISUALIZER_TESTS.json');r=sub.add_parser('actual')
 for n in ['seal','seal-sha','frames','frames-sha','queries','queries-sha','output']:r.add_argument('--'+n,required=True)
 a=p.parse_args();generated(a.output) if a.mode=='generated' else actual(a)
