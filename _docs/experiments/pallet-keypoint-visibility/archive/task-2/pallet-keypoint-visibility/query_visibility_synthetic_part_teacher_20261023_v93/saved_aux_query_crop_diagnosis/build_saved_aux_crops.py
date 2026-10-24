"""Posthoc saved diagnostic palette crops; not training data, native query or semantic labels."""
from pathlib import Path
import argparse,ast,hashlib,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent
V93=ROOT.parent
RASTER=V93/'raster_prototype/FIXED_PART152_SIX_CAMERAS'
SEAL=RASTER/'RASTER_OUTPUT_SEAL.json'
EXPECTED_SEAL='7f44b42583d4a5cc912eb78f6d8e4fae62a20d387b6823310085b207e40d0c3e'
CROP_CODE=V93.parent/'query_visibility_loco_native_context_20261023_v74/native_context_cache/compile_input_pack.py'
CROP_SHA='802f2e17f1ee98ae09d8468b8ab749653a50054a85f24ef0af16385a7c262b66'
ORDER=['plusX','minusX','plusY','minusY','plusZ','minusZ']
OUT=ROOT/'ACTUAL_SAVED_AUX_CROPS'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(p),'sha256':sha(p),'bytes':Path(p).stat().st_size}
def write(p,o):
 assert not Path(p).exists();Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def check(d):assert sha(d['path'])==d['sha256'];return Path(d['path'])
def js(p):return json.loads(Path(p).read_text())
def crop_operator():
 assert sha(CROP_CODE)==CROP_SHA
 tree=ast.parse(CROP_CODE.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='crop');env={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(CROP_CODE),'exec'),env)
 return env['crop'],hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest()
def mask_nearest(ids,amb,q,L,n=96):
 h,w=ids.shape;off=((np.arange(n,dtype=np.float64)+.5)/n-.5)*L
 x=q[0]+off[None,:];y=q[1]+off[:,None];x,y=np.broadcast_arrays(x,y);px=np.floor(x+.5).astype(np.int64);py=np.floor(y+.5).astype(np.int64)
 continuous=(x>=0)&(x<w)&(y>=0)&(y<h);nearest=(px>=0)&(px<w)&(py>=0)&(py<h);valid=continuous&nearest
 sx=np.clip(px,0,w-1);sy=np.clip(py,0,h-1);out=np.where(valid,ids[sy,sx],-2).astype(np.int32);a=np.where(valid,amb[sy,sx],True)
 return out,a,valid,continuous,nearest

def scalar_rgb(rgb,q,L,iy,ix,n=96):
 h,w=rgb.shape[:2];x=q[0]+((ix+.5)/n-.5)*L;y=q[1]+((iy+.5)/n-.5)*L
 ok=-.5<=x<w-.5 and -.5<=y<h-.5
 x0=int(np.floor(x));y0=int(np.floor(y));v=np.zeros(3,np.float64)
 for xx,wx in [(x0,1-(x-x0)),(x0+1,x-x0)]:
  for yy,wy in [(y0,1-(y-y0)),(y0+1,y-y0)]:
   if 0<=xx<w and 0<=yy<h:v+=rgb[yy,xx]*wx*wy
 return (np.clip(np.floor(v+.5),0,255).astype(np.uint8) if ok else np.zeros(3,np.uint8)),ok

def generated():
 crop,astsha=crop_operator();rgb=np.arange(9*11*3,dtype=np.uint8).reshape(9,11,3);count=0
 for q,L in [(np.array([5.,4.]),64.),(np.array([-.49,8.49]),3.),(np.array([10.4,.2]),9.)]:
  c,v=crop(rgb,q,L,96)
  for iy,ix in [(0,0),(0,95),(95,0),(95,95),(47,47),(48,48),(21,72),(75,12)]:
   expected,valid=scalar_rgb(rgb,q,L,iy,ix);assert np.array_equal(c[:,iy,ix],expected)and bool(v[0,iy,ix])==valid;count+=1
  ids=np.arange(99,dtype=np.int32).reshape(9,11);amb=np.zeros_like(ids,bool);cat,a,valid,cont,nn=mask_nearest(ids,amb,q,L)
  for iy,ix in [(0,0),(95,95),(47,47),(48,48),(21,72),(75,12)]:
   x=q[0]+((ix+.5)/96-.5)*L;y=q[1]+((iy+.5)/96-.5)*L;px=int(np.floor(x+.5));py=int(np.floor(y+.5));ok=0<=x<11 and 0<=y<9 and 0<=px<11 and 0<=py<9
   assert bool(valid[iy,ix])==ok and cat[iy,ix]==(ids[py,px]if ok else -2)
 assert set(np.unique(cat[valid])).issubset(set(ids.ravel()))
 write(ROOT/'GENERATED_CROP_SANITY.json',{'completed':True,'PASS':True,'code':bind(__file__),'crop_AST_source':bind(CROP_CODE),'crop_AST_sha256':astsha,'scalar_RGB_exact_points':count,'scalar_categorical_nearest_exact_points':18,'no_bilinear_categorical_IDs':True,'RGBvalid_vs_nativecontinuous_NN_separate':True,'source_saved_RGB_mask_or_mesh_body_reads':0,'new_raster_model_current_fit_PnP':0})

def freeze():
 assert sha(SEAL)==EXPECTED_SEAL;seal=js(SEAL);assert seal['completed']and seal['PASS']and seal['cameras']==6
 resultdesc=next(x for x in seal['files']if Path(x['path']).name=='GEOMETRY_DIAGNOSTIC_RESULTS.json');result=js(check(resultdesc));assert [x['camera']for x in result['results']]==ORDER
 crop,astsha=crop_operator();frames=[]
 for item in result['results']:
  cam=item['camera_matrices'];center=np.asarray(item['selected_world_AABB_m'],np.float64).mean(0);pc=center[None]@np.asarray(cam['R']).T+np.asarray(cam['t']);pc=pc[0];assert pc[2]>0
  K=np.asarray(cam['K']);q=pc[:2]/pc[2]*np.array([K[0,0],K[1,1]])+K[:2,2]
  frames.append({'camera':item['camera'],'aux_q_xy':q.tolist(),'L':64.,'selected_world_AABB_center_m':center.tolist(),'selected_center_camera':pc.tolist(),'camera_matrices':cam,'source_NPZ':item['NPZ'],'query_semantic_DES':'UNKNOWN','native8_association':'UNKNOWN'})
 qfile=ROOT/'FIXED_AUX_QUERIES_BEFORE_CROPS.json';write(qfile,{'completed':True,'source_raster_seal':bind(SEAL),'query_rule':'POSTHOC projection of unchanged part152 worldAABBcenter, not initializer or corner cue. FixedL64, no querysearch/filtering; all6 retained.','frames':frames})
 write(ROOT/'PROTOCOL_BEFORE_SAVED_CROPS.json',{'completed':True,'code':bind(__file__),'generated':bind(ROOT/'GENERATED_CROP_SANITY.json'),'source_raster_seal':bind(SEAL),'source_results':resultdesc,'fixed_queries':bind(qfile),'crop_AST_source':bind(CROP_CODE),'crop_AST_sha256':astsha,'RGB_operator':'ExactV74bilinear q+((j+.5)/96-.5)*64, uint8floor(v+.5), RGBvalid[-.5,W-.5). Sameq/L eachPA/PS/PV; qnotrounded.','mask_operator':'Native categorical IDs/ambiguity nearest floor(x+.5), continuous+NNboundsvalid separately; invalidID-2/U, nativebackgroundID-1. No bilinear categorical labels.','colors':'Saved selectedgreen/foreignred diagnosticpalette only. NEVER trainingRGB/modelinput/physicalmaterial gold.','scope':'Posthoc operator illustration, auxiliary midboard centres NOT corner/native8/actual initialq. QuerysemanticDES UNKNOWN. No new raster/model/current/fit/PnP.','output':str(OUT),'output_budget_bytes':10000000})

def contact(rgb_native,crops,ids,amb,maskvalid,q,L,names):
 font=ImageFont.truetype('/usr/share/fonts/truetype/nanum/NanumGothic.ttf',23);small=ImageFont.truetype('/usr/share/fonts/truetype/nanum/NanumGothic.ttf',19)
 canvas=Image.new('RGB',(1920,2260),'#f5f7fa');draw=ImageDraw.Draw(canvas)
 draw.text((20,15),'POSTHOC auxiliary midpoint queries — not corners / native8 / saved initializer coordinates',font=font,fill='#132339')
 draw.text((20,50),'고정 L=64 / exact V74 RGB96 bilinear / categorical masks nearest only / 모든 6개 유지',font=font,fill='#132339')
 draw.text((20,84),'선택 green·foreign red는 diagnostic palette입니다. 절대 학습 입력이나 semantic D/E/S gold로 쓰지 않습니다.',font=small,fill='#b02624')
 labels=['Native PS + auxiliary q/L','PA RGB96','PS RGB96','PV RGB96','PA selected / foreign','PS selected / foreign','PV selected / foreign']
 for i,name in enumerate(names):
  y=130+i*340;draw.text((20,y),f'{name} | auxiliary q=({q[i,0]:.5f}, {q[i,1]:.5f})  L={L[i]:.0f}px  | query semantic state UNKNOWN',font=font,fill='#132339')
  native=Image.fromarray(rgb_native[i].copy());d=ImageDraw.Draw(native);x0,y0=q[i]-L[i]/2;x1,y1=q[i]+L[i]/2;d.rectangle((x0,y0,x1,y1),outline=(255,220,0),width=2);x,yq=q[i];d.ellipse((x-3,yq-3,x+3,yq+3),outline=(255,220,0),width=1)
  panels=[native]+[Image.fromarray(crops[i,j].transpose(1,2,0))for j in range(3)]
  for j in range(3):
   m=np.full((96,96,3),235,np.uint8);m[(ids[i,j]>=0)&maskvalid[i]]=[120,125,135];m[(ids[i,j]==152)&maskvalid[i]&~amb[i,j]]=[35,185,95];m[(ids[i,j]==308)&maskvalid[i]&~amb[i,j]]=[225,55,45];m[amb[i,j]|~maskvalid[i]]=[225,40,210];panels.append(Image.fromarray(m))
  for j,(im,label)in enumerate(zip(panels,labels)):
   x=20+270*j;draw.text((x,y+35),label,font=small,fill='#243348');canvas.paste(im.resize((256,256),Image.Resampling.NEAREST),(x,y+66))
 draw.text((20,2180),'RGB96 확대는 표시용이며 mask색은 별도 nearest categorical sample입니다. RGB valid와 native continuous/NN valid는 별도 저장합니다.',font=small,fill='#243348')
 draw.text((20,2214),'이 그림은 저장된 geometry 진단 이미지의 crop 연산 예시일 뿐, 새 raster/학습/초기좌표 개선/실제 가림 라벨을 만들지 않습니다.',font=small,fill='#243348')
 return canvas

def run():
 p=js(ROOT/'PROTOCOL_BEFORE_SAVED_CROPS.json');assert p['code']['sha256']==sha(__file__)and sha(SEAL)==EXPECTED_SEAL and js(check(p['generated']))['PASS']
 assert not OUT.exists();OUT.mkdir();fixed=js(check(p['fixed_queries']))['frames'];assert len(fixed)==6 and [x['camera']for x in fixed]==ORDER
 crop,astsha=crop_operator();assert astsha==p['crop_AST_sha256']
 q=np.array([x['aux_q_xy']for x in fixed],np.float64);L=np.full(6,64.,np.float64);rgbc=[];rgbv=[];cats=[];ambs=[];maskv=[];conts=[];nns=[];native=[];records=[]
 for i,f in enumerate(fixed):
  z=np.load(check(f['source_NPZ']),allow_pickle=False);cam=f['camera_matrices']
  for key,ref in [('camera_K','K'),('camera_R_world_to_cv','R'),('camera_t_world_to_cv','t')]:assert np.array_equal(z[key],np.asarray(cam[ref]))
  perRGB=[];perValid=[];perIDs=[];perAmb=[];perMask=[]
  for tag in ['PA','PS','PV']:
   rgb=z[tag+'_RGB_diagnostic_only'];assert rgb.shape==(256,256,3)and rgb.dtype==np.uint8
   c,v=crop(rgb,q[i],L[i],96);ci,ca,cv,co,nn=mask_nearest(z[tag+'_part_id'],z[tag+'_ambiguity'],q[i],L[i])
   perRGB.append(c);perValid.append(v);perIDs.append(ci);perAmb.append(ca);perMask.append(cv)
  assert all(np.array_equal(perValid[0],x)for x in perValid)and all(np.array_equal(perMask[0],x)for x in perMask)
  rgbc.append(perRGB);rgbv.append(perValid);cats.append(perIDs);ambs.append(perAmb);maskv.append(perMask[0]);conts.append(co);nns.append(nn);native.append(z['PS_RGB_diagnostic_only'])
  records.append({'camera':f['camera'],'aux_q_xy':q[i].tolist(),'L':64.,'input_NPZ':f['source_NPZ'],'RGB_valid_fraction':float(np.mean(perValid[0])),'categorical_mask_valid_fraction':float(np.mean(perMask[0])),'nearest_selected_pixels':[int(np.sum((x==152)&perMask[0]&~a))for x,a in zip(perIDs,perAmb)],'nearest_foreign_pixels':[int(np.sum((x==308)&perMask[0]&~a))for x,a in zip(perIDs,perAmb)],'query_semantic_DES':'UNKNOWN','q_is_saved_initializer_or_corner':False})
 rgbc=np.asarray(rgbc);rgbv=np.asarray(rgbv);cats=np.asarray(cats);ambs=np.asarray(ambs);maskv=np.asarray(maskv)
 np.savez_compressed(OUT/'ALL6_AUXILIARY_SAVED_CROPS_NOT_TRAINING.npz',rgb96_diagnostic_only=rgbc,RGB_valid96=rgbv,categorical_part_id96_nearest=cats,ownership_ambiguity96_nearest=ambs,mask_valid96=maskv,mask_continuous_valid96=np.asarray(conts),mask_nearest_inbounds96=np.asarray(nns),selected_part96_nearest=(cats==152)&maskv[:,None]&~ambs,foreign_part96_nearest=(cats==308)&maskv[:,None]&~ambs,aux_query_xy=q,L=L,camera_name=np.array(ORDER))
 contact(native,rgbc,cats,ambs,maskv,q,L,ORDER).save(OUT/'ALL6_AUXILIARY_QUERY_CROP_CONTACT.png')
 write(OUT/'CROP_PROVENANCE_AND_COUNTS.json',{'completed':True,'protocol':bind(ROOT/'PROTOCOL_BEFORE_SAVED_CROPS.json'),'records':records,'all6_kept':True,'RGB_or_q_L_relocation':False,'training_or_native_labels_created':False,'querysemanticDES_all_UNKNOWN':True,'new_raster_model_current_fit_PnP':0})
 size=sum(x.stat().st_size for x in OUT.iterdir()if x.is_file());assert size<10000000
 write(OUT/'SAVED_AUX_CROP_SEAL.json',{'completed':True,'PASS':True,'code':bind(__file__),'protocol':bind(ROOT/'PROTOCOL_BEFORE_SAVED_CROPS.json'),'raster_seal':bind(SEAL),'crop_operator_source':bind(CROP_CODE),'crop_AST_sha256':astsha,'cameras':6,'stages':3,'aux_L':64.,'all6_kept':True,'files':[bind(x)for x in sorted(OUT.iterdir())if x.is_file()],'bytes_before_seal':size,'training_input_admitted':False,'query_semantic_DES':'UNKNOWN','new_raster_model_current_fit_PnP':0})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['generated','freeze','run']);a=p.parse_args()
 if a.stage=='generated':generated()
 elif a.stage=='freeze':freeze()
 else:run()
