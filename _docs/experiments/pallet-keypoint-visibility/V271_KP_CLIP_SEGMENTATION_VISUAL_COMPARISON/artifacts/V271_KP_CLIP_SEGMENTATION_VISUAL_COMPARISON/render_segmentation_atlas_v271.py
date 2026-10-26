"""Saved whole-frame pallet mask atlas. No neural/model/optimizer/GT reads.

Exact geometry from V269 evaluate_fullframe_v269.py:258-273: bilinear raw32
logit -> original HW pixel centres, align_corners=False/border; then sigmoid.
Raw-map threshold is fixed at0 (probability.5), uniformly for all five models.
"""
from pathlib import Path
import hashlib,html,json,shutil,sys,time
import numpy as np
import PIL
from PIL import Image,ImageDraw,ImageFont
W=Path(__file__).absolute().parent
R=W.parent
FRAMES=(0,25,46,100,105,106,113,124,189,83)
ARMS=('FULL_CLIP','FULL_CLIP_SURGERY_ARCH','EVA_STANDARD','EVA_CSA','EVA_DECLIP')
NAMES={'FULL_CLIP':'OpenAI CLIP + SYN mask head','FULL_CLIP_SURGERY_ARCH':'CLIP Surgery arch + SYN head','EVA_STANDARD':'EVA original + SYN mask head','EVA_CSA':'EVA original + CSA + SYN head','EVA_DECLIP':'DeCLIP adapted + SYN mask head'}


def safe(p):
 p=Path(p);assert p.is_absolute()and not any(x in str(p).lower()for x in ('v112','cache','.pyc','..')),str(p)
 for a in [p,*p.parents]:assert not a.is_symlink(),str(a)
 return p


def binding(p):
 p=safe(p);h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return dict(path=str(p),sha256=h.hexdigest(),size_bytes=p.stat().st_size)


def verify(b):
 actual=binding(b['path']);assert actual['sha256']==b['sha256']
 for k in ('size_bytes','size','bytes'):
  if k in b:assert actual['size_bytes']==b[k]
 return safe(b['path'])


def read(p):return json.loads(safe(p).read_bytes())


def dump(p,d):safe(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n')


def upsample_raw_logits(logits,raw_hw):
 # Exact formula of the already-used V269 evaluator, not a new display warp.
 h,w=map(int,raw_hw);gh,gw=logits.shape
 x=np.clip((np.arange(w,dtype=np.float64)+.5)/w*gw-.5,0,gw-1)
 y=np.clip((np.arange(h,dtype=np.float64)+.5)/h*gh-.5,0,gh-1)
 x0=np.floor(x).astype(np.int64);y0=np.floor(y).astype(np.int64)
 x1=np.minimum(x0+1,gw-1);y1=np.minimum(y0+1,gh-1)
 dx=x-x0;dy=(y-y0)[:,None]
 a=logits[y0[:,None],x0[None,:]]*(1-dx)+logits[y0[:,None],x1[None,:]]*dx
 b=logits[y1[:,None],x0[None,:]]*(1-dx)+logits[y1[:,None],x1[None,:]]*dx
 return(a*(1-dy)+b*dy).astype(np.float32)

FONT=safe(Path('/usr/share/fonts/truetype/nanum/NanumGothic.ttf'))
def font(n):return ImageFont.truetype(str(FONT),n)


def fit(im,w,h,bg='#eef2f6'):
 result=Image.new('RGB',(w,h),bg);scale=min(w/im.width,h/im.height);size=(round(im.width*scale),round(im.height*scale));im=im.resize(size,Image.Resampling.LANCZOS);result.paste(im,((w-size[0])//2,(h-size[1])//2));return result


def save_image(im,path):safe(path);im.save(path)


def make_comparison(rgb,panels,frame,arms,out):
 pw,ph=430,290;cw=450;width=cw*4;height=1475
 can=Image.new('RGB',(width,height),'#edf2f7');d=ImageDraw.Draw(can)
 d.text((20,12),f'Frame {frame["frame_index"]:03d} · {frame["frame_id"]} · 예측 파렛트 segmentation',font=font(29),fill='#183651')
 d.text((20,57),'전체 이미지 · 실사 픽셀 GT 없음 · 고정 p(target)≥0.5 · 추가 추론/학습 없음',font=font(21),fill='#40546b')
 labels=('RGB + predicted mask overlay','P(target), fixed 0–1 scale','Predicted binary mask (white=target)','Predicted RGB cutout (gray=non-target)')
 keys=('overlay','probability','binary','cutout')
 for col,arm in enumerate(('ORIGINAL',*arms)):
  x=col*cw+10;title='원본 전체 RGB'if arm=='ORIGINAL'else NAMES[arm]
  d.text((x,105),title,font=font(20),fill='#173650')
  for row,key in enumerate(keys):
   y=145+row*325
   if arm=='ORIGINAL':
    if row==0:im=rgb
    else:
     im=Image.new('RGB',(pw,ph),'#f8fafc');z=ImageDraw.Draw(im)
     if row==1:
      z.text((12,15),'확률 색 범위는 모든 모델에 동일',font=font(22),fill='#294862')
      z.text((12,50),'어두움 0 → 밝은 초록 1',font=font(20),fill='#294862')
      p=np.linspace(0,1,360,dtype=np.float32)[None,:].repeat(36,axis=0);cm=np.stack((20+20*p,30+220*p,25+65*p),-1).astype(np.uint8);im.paste(Image.fromarray(cm),(18,95));z.text((15,142),'0                                      1',font=font(21),fill='#294862')
     elif row==2:
      for yy,t in enumerate(['흰색 = 모델이 찾은 파렛트 영역','검정 = 해당 영역 아님','검정을 외부 가림 E로 해석하지 않음','배경/콘/그림자/내부 구멍은 별도 구분 없음']):z.text((12,20+yy*44),t,font=font(20),fill='#294862')
     else:
      for yy,t in enumerate(['RGB는 예측 mask 안에서만 유지','나머지는 회색으로 표시','분할 모델: 전체 RGB512 → raw32 map','KP 주변 crop이나 좌표를 입력하지 않음']):z.text((12,20+yy*44),t,font=font(20),fill='#294862')
   else:im=panels[arm][key]
   can.paste(fit(im,pw,ph),(x,y));d=ImageDraw.Draw(can)
   d.text((x,y+ph+4),labels[row]if arm!='ORIGINAL'or row==0 else '공통 표시/해석',font=font(17),fill='#40546b')
 d.text((20,1450),'합성 M4로 학습한 binary visible-pallet-mask head의 저장 예측입니다. 실제 가림 원인 분할/실사 IoU 결과가 아닙니다.',font=font(20),fill='#65421b')
 save_image(can,out)


def make_overview(cases,panel_by_frame,rgb_by_frame,out,mode):
 width,height=1800,1245;can=Image.new('RGB',(width,height),'#edf2f7');d=ImageDraw.Draw(can)
 title='파렛트 분할된 RGB: 예측 mask 안의 색만 유지'if mode=='cutout'else'예측 파렛트 binary mask: 흰색=파렛트, 검정=나머지'
 d.text((20,12),title,font=font(32),fill='#183651');d.text((20,59),'실사 픽셀 GT 없음 · p(target)≥0.5 고정 · 전체 영상 입력 · 저장 출력만 시각화',font=font(22),fill='#40546b')
 for col,name in enumerate(('원본 전체 RGB','CLIP Surgery arch + SYN mask head','DeCLIP adapted + SYN mask head')):d.text((col*600+20,104),name,font=font(24),fill='#183651')
 for row,fi in enumerate((0,46,105)):
  c=next(x for x in cases if x['frame_index']==fi);y=150+row*340
  for col,arm in enumerate(('ORIGINAL','FULL_CLIP_SURGERY_ARCH','EVA_DECLIP')):
   im=rgb_by_frame[fi]if arm=='ORIGINAL'else panel_by_frame[fi][arm][mode]
   can.paste(fit(im,570,300),(col*600+15,y));d=ImageDraw.Draw(can)
   d.text((col*600+20,y+302),f'F{fi:03d} · {c["frame_id"]}',font=font(20),fill='#294862')
 d.text((20,1188),'회색 cutout / 검정 mask 영역은 배경·콘·내부 구멍을 포함할 수 있습니다. 외부 가림 E를 직접 분류한 영역이 아닙니다.',font=font(22),fill='#65421b')
 save_image(can,out)


def main():
 start=time.time();out=safe(W/'segmentation_atlas');assert not out.exists();out.mkdir()
 f267=safe(R/'V267_FULL_IMAGE_CLIP_VISIBLE_MASK/CURRENT_PREDICTIONS_FROZEN.json');f269=safe(R/'V269_FULL_IMAGE_DECLIP_MATCHED_BACKBONE/CURRENT_PREDICTIONS_FROZEN.json')
 a,b=read(f267),read(f269);assert a['PASS']and b['PASS'];assert a['fixed_threshold']==b['fixed_threshold']==.5
 m267=verify(a['frame_masks']);m269=verify(b['frame_masks'])
 complete269=read(R/'V269_FULL_IMAGE_DECLIP_MATCHED_BACKBONE/evaluation_v1/COMPLETE.json');compactpath=verify(complete269['visuals']['compact_visual_inputs']);compact=read(compactpath)
 image_records={r['frame_index']:r for r in compact['full_RGBs']};metadata={r['frame_index']:r for r in compact['frame_metadata']}
 assert all(i in image_records and i in metadata for i in FRAMES)
 dump(out/'SOURCE_RGB_AVAILABILITY.json',dict(PASS=True,input_manifest=binding(compactpath),available_frame_indices=sorted(image_records),count=len(image_records),selected=list(FRAMES)))
 with np.load(m267,allow_pickle=False)as z:
  assert np.array_equal(z['frame_index'],np.arange(223));maps={arm:np.array(z[arm+'_logit'],copy=True)for arm in ARMS[:2]}
 with np.load(m269,allow_pickle=False)as z:
  assert np.array_equal(z['frame_index'],np.arange(223));maps.update({arm:np.array(z[arm+'_logit'],copy=True)for arm in ARMS[2:]})
 for arm,v in maps.items():assert v.shape==(223,32,32)and v.dtype==np.float32 and np.isfinite(v).all()
 source_eval=safe(R/'V269_FULL_IMAGE_DECLIP_MATCHED_BACKBONE/evaluate_fullframe_v269.py')
 dump(out/'SELECTION_BEFORE_RGB.json',dict(PASS=True,completed=True,source_freezes=[binding(f267),binding(f269)],raw_maps=[binding(m267),binding(m269)],RGB_manifest=binding(compactpath),fixed_frames=list(FRAMES),current_RGB_reads=0,new_model_forwards=0,new_fits=0,pixel_GT_reads=0,transform_source=binding(source_eval),transform_source_lines='258-273',transform='raw32 logits bilinear pixel-center originalHW; align_corners=False,border, then sigmoid; binary logit>=0'))
 selected={arm:maps[arm][list(FRAMES)]for arm in ARMS}
 np.savez_compressed(safe(out/'SELECTED_RAW32_LOGITS.npz'),frame_index=np.array(FRAMES,np.int32),**{arm+'_logit':v for arm,v in selected.items()})
 dump(out/'SELECTED_RAW32_LOGITS.json',dict(frame_index=list(FRAMES),dtype='float32',shape=[10,32,32],models={arm:selected[arm].tolist()for arm in ARMS},meaning='target-pallet raw logit, not external-occluder class',threshold_logit=0))
 panels_by_frame={};rgbs={};cases=[];outputbindings={};decodeproofs={}
 for fi in FRAMES:
  record=image_records[fi];path=verify(record['file']);rgb=Image.open(path).convert('RGB');arr=np.asarray(rgb);raw_hw=list(arr.shape[:2]);meta=metadata[fi]
  assert hashlib.sha256(arr.tobytes()).hexdigest()==record['decoded_RGB_sha256'];assert raw_hw==list(meta['raw_hw'])
  rgbs[fi]=rgb;rawname=f'FRAME_{fi:03d}_RGB.png';shutil.copyfile(path,safe(out/rawname))
  display=rgb.copy();display.thumbnail((768,768),Image.Resampling.LANCZOS);displayname=f'FRAME_{fi:03d}_RGB_DISPLAY.png';save_image(display,out/displayname)
  panels={};model_records={}
  for arm in ARMS:
   dense=upsample_raw_logits(maps[arm][fi],raw_hw);probability=1/(1+np.exp(-np.clip(dense,-80,80)));mask=dense>=0
   color=np.stack((20+20*probability,30+220*probability,25+65*probability),-1).clip(0,255).astype(np.uint8)
   overlay=arr.copy();overlay[mask]=(overlay[mask]*.5+np.array([25,230,80])*.5).astype(np.uint8)
   cutout=np.full_like(arr,48);cutout[mask]=arr[mask]
   ps=dict(probability=Image.fromarray(color),binary=Image.fromarray((mask*255).astype(np.uint8)).convert('RGB'),overlay=Image.fromarray(overlay),cutout=Image.fromarray(cutout));panels[arm]=ps;mr={}
   for kind,im in ps.items():
    name=f'FRAME_{fi:03d}_{arm}_{kind.upper()}.png';save_image(im,out/name);mr[kind+'_png']=name;outputbindings[name]=binding(out/name)
   mr.update(target_fraction=float(mask.mean()),raw_logit_min=float(maps[arm][fi].min()),raw_logit_max=float(maps[arm][fi].max()));model_records[arm]=mr
  panels_by_frame[fi]=panels
  comparison=f'FRAME_{fi:03d}_COMPARE.png';matched=f'FRAME_{fi:03d}_MATCHED_EVA.png'
  frame=dict(frame_index=fi,frame_id=meta['frame_id'])
  make_comparison(rgb,panels,frame,('FULL_CLIP','FULL_CLIP_SURGERY_ARCH','EVA_DECLIP'),out/comparison)
  make_comparison(rgb,panels,frame,('EVA_STANDARD','EVA_CSA','EVA_DECLIP'),out/matched)
  c=dict(frame_index=fi,frame_id=meta['frame_id'],raw_hw=raw_hw,comparison_png=comparison,matched_eva_png=matched,rgb_png=rawname,rgb_display_png=displayname,models=model_records,real_pixel_GT_available=False)
  cases.append(c);decodeproofs[str(fi)]=dict(source=record['file'],decoded_RGB_sha256=record['decoded_RGB_sha256'])
  print('SAVED_FRAME',fi,'elapsed',round(time.time()-start,1),flush=True)
  for name in [comparison,matched,rawname,displayname]:outputbindings[name]=binding(out/name)
 dump(out/'CASES.json',cases)
 make_overview(cases,panels_by_frame,rgbs,out/'SEGMENTATION_OVERVIEW.png','cutout');make_overview(cases,panels_by_frame,rgbs,out/'SEGMENTATION_MASK_OVERVIEW.png','binary')
 links=[]
 for c in cases:
  links.append('<article><h2>F'+str(c['frame_index']).zfill(3)+' · '+html.escape(c['frame_id'])+'</h2><p><a href="'+c['comparison_png']+'">CLIP / Surgery / DeCLIP 비교</a> · <a href="'+c['matched_eva_png']+'">EVA 동일 backbone 대조</a> · <a href="'+c['rgb_png']+'">원본 전체 RGB</a></p><img loading="lazy" src="'+c['comparison_png']+'"><table><tr><th>모델</th><th>soft P(target)</th><th>진짜 흑백 mask</th><th>RGB cutout</th><th>alpha overlay</th></tr>'+''.join('<tr><td>'+html.escape(NAMES[a])+'</td>'+''.join('<td><a href="'+c['models'][a][k+'_png']+'">'+k+'</a></td>'for k in ('probability','binary','cutout','overlay'))+'</tr>'for a in ARMS)+'</table></article>')
 page='<!doctype html><meta charset="utf-8"><title>V271 전체 파렛트 segmentation 비교</title><style>body{max-width:1800px;margin:24px auto;font:18px sans-serif;background:#edf2f7;color:#173650}article{background:white;padding:18px;margin:28px 0;border-radius:12px}img{max-width:100%;height:auto}td,th{padding:10px;border-bottom:1px solid #ccd8e4}p{line-height:1.6}a{color:#1565a8}</style><h1>전체 영상의 파렛트 segmentation: 기존 저장 출력 비교</h1><p>모든 그림은 예측값입니다. 실사 pixelGT가 없어 IoU/정답 mask라고 주장하지 않습니다. 전체 RGB512 입력에서 만든 raw32 logit을 원본 HW로 pixel-center bilinear(border, align_corners=False) 보간한 뒤 sigmoid, 고정 p(target)≥0.5로 표시했습니다. 회색/검정 영역은 배경·콘·내부 구멍을 포함할 수 있으며 외부 가림 E와 동일하지 않습니다. 새 추론·학습·threshold 조정은 없습니다.</p><p>CLIP/Surgery/DeCLIP 비교는 backbone과 사전학습 차이가 함께 있습니다. 동일 EVA original→CSA→DeCLIP 대조 그림을 별도로 제공합니다. 모든 head는 기존 합성 M4 binary supervision으로 학습한 head이며 text-prompt zero-shot 분할을 새로 실행한 결과가 아닙니다.</p><h2>실제 RGB cutout</h2><a href="SEGMENTATION_OVERVIEW.png"><img src="SEGMENTATION_OVERVIEW.png"></a><h2>예측 binary mask 자체</h2><a href="SEGMENTATION_MASK_OVERVIEW.png"><img src="SEGMENTATION_MASK_OVERVIEW.png"></a><p><a href="CASES.json">사례/이미지 경로</a> · <a href="SELECTED_RAW32_LOGITS.npz">정확한 선택10 raw32 FP32 NPZ</a> · <a href="SELECTED_RAW32_LOGITS.json">브라우저용 raw32 JSON</a> · <a href="SOURCE_RGB_AVAILABILITY.json">기존84 RGB 목록</a></p>'+''.join(links)
 safe(out/'INDEX_KO.html').write_text(page)
 for name in ['CASES.json','SELECTED_RAW32_LOGITS.npz','SELECTED_RAW32_LOGITS.json','SEGMENTATION_OVERVIEW.png','SEGMENTATION_MASK_OVERVIEW.png','INDEX_KO.html','SOURCE_RGB_AVAILABILITY.json','SELECTION_BEFORE_RGB.json']:outputbindings[name]=binding(out/name)
 dump(out/'COMPLETE.json',dict(PASS=True,completed=True,code=binding(Path(__file__).absolute()),fixed_frames=list(FRAMES),frame_count=len(FRAMES),models=list(ARMS),map_bindings=[binding(m267),binding(m269)],source_RGB_manifest=binding(compactpath),RGB_decode_proofs=decodeproofs,source_transform=binding(source_eval),source_transform_lines='258-273',raw_map_to_original='logit bilinear align_corners=False,border then sigmoid',probability_range=[0,1],threshold=.5,overlay_target_RGB=[25,230,80],overlay_alpha=.5,cutout_non_target_RGB=[48,48,48],hard_mask_values=[0,255],real_pixel_GT_available=False,pixel_GT_reads=0,current_annotation_reads=0,new_model_forwards=0,new_fits=0,new_PnP=0,font=binding(FONT),runtime=dict(python=sys.executable,Pillow=PIL.__version__),outputs=outputbindings,elapsed_seconds=time.time()-start))
 print('PASS COMPLETE',len(cases),len(outputbindings),'elapsed',round(time.time()-start,1),flush=True)

if __name__=='__main__':main()
