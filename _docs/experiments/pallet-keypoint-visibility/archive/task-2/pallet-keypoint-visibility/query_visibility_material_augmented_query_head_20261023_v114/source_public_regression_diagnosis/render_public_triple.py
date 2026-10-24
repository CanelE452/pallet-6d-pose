import os,sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,importlib.util,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
O=Path(__file__).resolve().parent;V=O.parent;P=V.parent;C11=P/'query_visibility_spatial_target_query_features_20261023_v111'
def read(p):return json.loads(Path(p).read_text())
def rows(p):return[json.loads(s)for s in Path(p).read_text().splitlines()if s]
def b(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def arrsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def mod(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
lock=read(O/'SELECTION_LOCK_BEFORE_PIXELS.json');assert len(lock['rows'])==3
protocol=read(C11/'feature_preparation/FEATURE_PROTOCOL_BEFORE_SOURCE_FORWARD.json');refs=rows(protocol['old_RGB_refs']['path']);contexts=rows(protocol['old_contexts']['path']);cert=rows(C11/'spatial_query_features/INPUT_RGB_AND_SPATIAL_FACTS_INPUT_ONLY.jsonl')
spec=dict(completed=True,selection=b(O/'SELECTION_LOCK_BEFORE_PIXELS.json'),operator='Exact cached RGB96/valid via pinned C11 RGBrefs; verify native fullRGB and V90 crop at unchanged q/L. Context=exact V107 sample_target_context at original saved initial box. No new encoder/weights/P3.',literal_regions={'central4':'RGB96[:,47:49,47:49] four pixels (2x2), no semantic/mask weighting','central32':'RGB96[:,32:64,32:64] 32x32'},code=b(Path(__file__)),source_feature_protocol=b(C11/'feature_preparation/FEATURE_PROTOCOL_BEFORE_SOURCE_FORWARD.json'),new_point_or_image_selection=False)
with(O/'RENDER_SPEC_BEFORE_PIXELS.json').open('x')as f:json.dump(spec,f,indent=2)
producer=mod(protocol['code']['path'],'public_triple_input_only');crop=producer.pure_crop();tm=mod(protocol['target_crop_code']['path'],'public_triple_context');cache=producer.InputCache()
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14);bold=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',17);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
rgbs=[];valids=[];targets=[];tvs=[];data=[];cards=[]
for n,s in enumerate(lock['rows']):
 u=s['input_unit_index'];ref=refs[u];ctx=contexts[u];assert ref['input_unit_index']==ctx['input_unit_index']==cert[u]['input_unit_index']==u
 q=np.asarray(s['query_xy'],np.float64);L=s['L'];assert ctx['query_xy']==s['query_xy']and ctx['L']==L
 rgb,_=cache.image(ctx['full_RGB'],ctx['full_RGB_format'],ctx['raw_hw'],ctx['decoded_RGB_sha256']);a=cache.array(ref['rgb96_ref']);valid=cache.array(ref['valid96_ref']);native_a,native_valid=crop(rgb,q,L,96)
 assert np.array_equal(a,native_a)and np.array_equal(valid,native_valid)and arrsha(a)==ref['rgb96_sha256']==cert[u]['RGB96_sha256']and arrsha(valid)==ref['valid96_sha256']==cert[u]['valid96_sha256']
 t,tv=tm.sample_target_context(rgb,ctx['box_xyxy']);assert arrsha(t.transpose(2,0,1))==cert[u]['target_RGB224_float32_sha256']
 rgbs.append(a);valids.append(valid);targets.append(t);tvs.append(tv)
 stage=['BEFORE','COVER','SHAM'][n];Image.fromarray(a.transpose(1,2,0)).save(O/f'{stage}_RGB96_UNMARKED.png')
 card=Image.new('RGB',(1280,390),'#edf2f5');d=ImageDraw.Draw(card)
 d.text((14,9),f'{stage} | row{s["global_occurrence_index"]} / unit{u} | AUX {s["typed_state"]} | C11 z={s["C11_raw_logit"]:+.6f}; C12 z={s["C12_raw_logit"]:+.6f}',font=bold,fill='#142b3b')
 d.text((14,38),f'q=({q[0]:.4f},{q[1]:.4f}), L={L:.4f}px; same source target/context/q/L. C11 step1800 / C12 step1200, hidden iff z>=0.',font=font,fill='#294355')
 d.text((14,62),'PUBLIC controlled augmentation; conditional AUX cue support, native UNKNOWN, not natural foreign/human junction gold.',font=font,fill='#78491a')
 positions=[14,351,553,755,957];labels=['Full source RGB + q/L/context marks','Exact RGB96 unmarked x2','Separate q marker + central32','Original valid96 x2','Frozen context224 unmarked']
 for x,label in zip(positions,labels):d.text((x,91),label,font=small,fill='#294355')
 full=Image.fromarray(rgb).resize((320,240),Image.Resampling.BILINEAR);fd=ImageDraw.Draw(full);qq=q/2;box=np.asarray(ctx['box_xyxy'])/2;fd.rectangle(tuple(box),outline='#f2cc36',width=1);fd.rectangle((qq[0]-L/4,qq[1]-L/4,qq[0]+L/4,qq[1]+L/4),outline='#00ffff',width=1);fd.line((qq[0]-4,qq[1],qq[0]+4,qq[1]),fill='#00ffff',width=2);fd.line((qq[0],qq[1]-4,qq[0],qq[1]+4),fill='#00ffff',width=2);card.paste(full,(14,111))
 im=Image.fromarray(a.transpose(1,2,0)).resize((192,192),Image.Resampling.NEAREST);card.paste(im,(351,111));marked=im.copy();md=ImageDraw.Draw(marked);md.line((87.5,95.5,103.5,95.5),fill='#00ffff',width=2);md.line((95.5,87.5,95.5,103.5),fill='#00ffff',width=2);md.rectangle((63.5,63.5,127.5,127.5),outline='#f4cf32',width=1);card.paste(marked,(553,111))
 card.paste(Image.fromarray((valid[0]*255).astype(np.uint8)).convert('RGB').resize((192,192),Image.Resampling.NEAREST),(755,111));card.paste(Image.fromarray(np.floor(np.clip(t,0,255)+.5).astype(np.uint8)),(957,111))
 d.text((351,316),'Cyan=q; yellow=central32. Markers are display-only.',font=small,fill='#294355');d.text((351,336),f'RGB96 SHA {arrsha(a)[:20]}... | valid={valid.mean():.4f}',font=small,fill='#294355')
 d.text((14,365),'All crops/context are byte-verified against frozen C11 input facts. No P3/encoder/head/fit/current/PnP. No point or label change.',font=small,fill='#78491a')
 card.save(O/f'{stage}_CARD.png');cards.append(card);data.append(dict(**s,stage=stage,RGB96_sha256=arrsha(a),valid96_sha256=arrsha(valid),context224_CHW_F32_sha256=arrsha(t.transpose(2,0,1)),full_source=ctx['full_RGB'],decoded_RGB_sha256=ctx['decoded_RGB_sha256'],frozen_context_box=ctx['box_xyxy'],source_context_id=ctx['context_id'],RGB96_ref=ref['rgb96_ref'],valid96_ref=ref['valid96_ref'],valid_fraction=float(valid.mean()),all_cached_RGB_native_crop_and_context_hashes_exact=True))
def difference(after,before,sl):
 delta=after[:,sl,sl].astype(np.int16)-before[:,sl,sl].astype(np.int16);changed=(delta!=0).any(0)
 return dict(shape_CHW=list(delta.shape),changed_pixels=int(changed.sum()),total_pixels=int(changed.size),changed_channels=int((delta!=0).sum()),mean_absolute_channel_difference=float(np.abs(delta).mean()),max_absolute_channel_difference=int(np.abs(delta).max()),identical=not bool(delta.any()))
diffs={stage:{name:difference(rgbs[n],rgbs[0],sl)for name,sl in [('central4',slice(47,49)),('central32',slice(32,64)),('all96',slice(0,96))]}for n,stage in [(1,'COVER_vs_BEFORE'),(2,'SHAM_vs_BEFORE')]}
canvas=Image.new('RGB',(1280,1320),'#e2eaf0');d=ImageDraw.Draw(canvas);d.text((14,10),'First fixed old-public VAL regression: C11 correct external cover, C12 false visible',font=bold,fill='#142b3b');d.text((14,40),'Pallox 1.23m_hall:0016_0017 BEFORE base817, donor rank6. Image selection frozen before pixels; all3 retained.',font=font,fill='#294355')
for n,card in enumerate(cards):canvas.paste(card,(0,70+390*n))
d.text((14,1250),'Literal RGB differences vs BEFORE (four center pixels / central32):',font=bold,fill='#142b3b')
for k,stage in enumerate(diffs):
 v=diffs[stage];d.text((14,1279+18*k),f'{stage}: central4 changed{v["central4"]["changed_pixels"]}/4, meanAbs={v["central4"]["mean_absolute_channel_difference"]:.3f}; central32 changed{v["central32"]["changed_pixels"]}/1024, meanAbs={v["central32"]["mean_absolute_channel_difference"]:.3f}',font=font,fill='#294355')
canvas.save(O/'CONTACT_FIXED_PUBLIC_REGRESSION.png');np.savez_compressed(O/'EXACT_TRIPLE_CROPS_AND_CONTEXT.npz',RGB96=np.stack(rgbs),valid96=np.stack(valids),context224_F32=np.stack(targets),context_valid224=np.stack(tvs),q_xy=np.asarray([x['query_xy']for x in lock['rows']]),L=np.asarray([x['L']for x in lock['rows']]))
with(O/'FIGURE_AND_INPUT_PROVENANCE.json').open('x')as f:json.dump(dict(completed=True,selection=b(O/'SELECTION_LOCK_BEFORE_PIXELS.json'),renderer=b(Path(__file__)),source_feature_protocol=b(C11/'feature_preparation/FEATURE_PROTOCOL_BEFORE_SOURCE_FORWARD.json'),rows=data,literal_RGB_differences=diffs,region_definitions=spec['literal_regions'],contact=b(O/'CONTACT_FIXED_PUBLIC_REGRESSION.png'),arrays=b(O/'EXACT_TRIPLE_CROPS_AND_CONTEXT.npz'),all3_native_cached_context_hashes_exact=True,originalqL_box_sourceGT_model_threshold_unchanged=True,encoder_head_P3_fit_private_current_PnP=0),f,indent=2)
print(json.dumps(dict(contact=b(O/'CONTACT_FIXED_PUBLIC_REGRESSION.png'),RGB_differences=diffs),indent=2))
