"""Saved-only six preselected public query views. No model import, weights or inference."""
from pathlib import Path
import argparse,ast,hashlib,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent
V92=ROOT.parent
RUNNER=V92/'qwen3_source_pilot_v2_schema/run_qwen_source_once.py'
RUNNER_SHA='298f456f8c8b28a8919307ec78c1224ddb72c519227287a0fca61ffd0d02d7b4'
SELECT=V92/'independent_qwen3_saved_output_audit/FIXED6_QUALITATIVE_SELECTION_BEFORE_OUTPUTS.json'
SELECT_SHA='8b910c2b7b60faf8965ea50e2e3fba8696ef927ea65630ea2510c15f2117ff2d'
FONT=Path('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
OUT=ROOT/'FIXED6_ACTUAL_SAVED'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while True:
   b=f.read(1<<20)
   if not b:break
   h.update(b)
 return h.hexdigest()
def bind(p):return {'path':str(p),'sha256':sha(p),'bytes':Path(p).stat().st_size}
def write(p,o):
 assert not Path(p).exists();Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def check(d):assert sha(d['path'])==d['sha256'];return Path(d['path'])
def js(p):return json.loads(Path(p).read_text())
def context_function():
 assert sha(RUNNER)==RUNNER_SHA
 t=ast.parse(RUNNER.read_text());fn=next(n for n in t.body if isinstance(n,ast.FunctionDef)and n.name=='make_context')
 env={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(RUNNER),'exec'),env)
 return env['make_context']
def wrapped(draw,text,maxwidth,font):
 """All characters kept; only insert display line breaks, never truncate raw output."""
 lines=[]
 for para in str(text).split('\n'):
  line=''
  for ch in para:
   if line and draw.textlength(line+ch,font=font)>maxwidth:lines.append(line);line=ch
   else:line+=ch
  lines.append(line)
 return lines
def block(draw,text,x,y,width,font,color='#202833',spacing=7):
 for line in wrapped(draw,text,width,font):
  draw.text((x,y),line,font=font,fill=color);y+=font.size+spacing
 return y

def figure(context,rgb96,vlm256,meta,response,weak):
 f=ImageFont.truetype(str(FONT),24);small=ImageFont.truetype(str(FONT),21);title=ImageFont.truetype(str(FONT),32)
 raw=response.get('raw_text');raw=raw if raw is not None else 'None (모델 호출 없음)'
 parsed=json.dumps(response.get('parsed_observation'),ensure_ascii=False,indent=2)
 dummy=ImageDraw.Draw(Image.new('RGB',(10,10)))
 rawlines=wrapped(dummy,raw,1720,f);parsedlines=wrapped(dummy,parsed,1720,small)
 h=1070+(len(rawlines)*(f.size+7))+(len(parsedlines)*(small.size+6))
 im=Image.new('RGB',(1800,h),'#f3f5f8');d=ImageDraw.Draw(im)
 y=block(d,f"고정 사례 {meta['case_index']:02d} | frame{meta['frame_index']} · logical slot{meta['native_slot']} | {meta['split']}",35,25,1730,title)
 y=block(d,f"{meta['input_id']}  |  q=({meta['q'][0]:.4f}, {meta['q'][1]:.4f})  L={meta['L']:.6f}px",35,y+6,1730,f)
 y=block(d,'사전 AI 관찰 유형: '+weak+'  (weak observation, 정답·semantic D/E/S 아님)',35,y+4,1730,small)
 top=y+28
 panels=[(context,(35,top,815,top+510),'1. full-frame context (cyan: saved predicted box / yellow: q)'),
         (Image.fromarray(rgb96.transpose(1,2,0)),(850,top,1285,top+435),'2. 원본 unmarked RGB96'),
         (vlm256,(1325,top,1760,top+435),'3. 별도 VLM용 256×256')]
 for pic,box,label in panels:
  x0,y0,x1,y1=box;d.rectangle((x0-1,y0-1,x1+1,y1+1),outline='#a4acb6',width=2)
  scale=min((x1-x0)/pic.width,(y1-y0)/pic.height);copy=pic.resize((max(1,round(pic.width*scale)),max(1,round(pic.height*scale))),Image.Resampling.NEAREST if pic is not context else Image.Resampling.BILINEAR)
  im.paste(copy,(x0+((x1-x0)-copy.width)//2,y0+((y1-y0)-copy.height)//2))
  block(d,label,x0,y1+10,x1-x0,small)
 y=top+590
 y=block(d,'표시 영상은 official processor 이전 이미지입니다. full context는 배치상 축소 표시; 실제 GPU tensor dump가 아닙니다.',35,y,1730,small)
 y=block(d,'96→256 bicubic 확대는 새 픽셀 정보를 추가하지 않습니다. 원래 q/L/crop/valid는 그대로이며, native8 대응은 UNKNOWN입니다.',35,y+5,1730,small)
 y=block(d,f"저장 상태: {response['status']} | generated tokens={response.get('generated_tokens')} | grid={response.get('image_grid_thw')}",35,y+14,1730,f)
 y=block(d,'원응답 (수정·JSON 보정·재호출 없음):',35,y+15,1730,f)
 y=block(d,raw,35,y+8,1730,f,color='#152e51')
 y=block(d,'저장 parsed fields (원 parser 결과; 없으면 null):',35,y+18,1730,f)
 y=block(d,parsed,35,y+8,1730,small,color='#44515e')
 assert y<=im.height-20,(y,im.height)
 return im

def generated():
 make=context_function();rgb=np.zeros((180,320,3),np.uint8);rgb[:,160:,0]=160
 ctx=make(rgb,[30,40,290,170],[160,100]);assert np.array_equal(rgb[:,0,0],np.zeros(180,np.uint8))
 crop=np.tile(np.arange(96,dtype=np.uint8)[None,None,:],(3,96,1));crop0=crop.copy()
 vlm=Image.fromarray(crop.transpose(1,2,0)).resize((256,256),Image.Resampling.BICUBIC)
 meta={'case_index':1,'frame_index':17,'native_slot':4,'split':'TRAIN','input_id':'GENERATED_LONG_IDENTIFIER_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789','q':[160.,100.],'L':128.}
 response={'status':'UNPARSED_UNCERTAIN','raw_text':'```json\n'+('{"note":"raw text with no repair"} '*10)+'\n```','parsed_observation':None,'generated_tokens':96,'image_grid_thw':[[1,24,42],[1,16,16]]}
 image=figure(ctx,crop,vlm,meta,response,'VISIBLE_DISTINCT_CORNER_EDGE_FOOT')
 assert np.array_equal(crop,crop0) and image.width==1800
 image.save(ROOT/'GENERATED_LAYOUT_ONLY.png')
 write(ROOT/'GENERATED_RENDERER_TESTS.json',{'completed':True,'PASS':True,'code':bind(__file__),'runner':bind(RUNNER),'generated_layout':bind(ROOT/'GENERATED_LAYOUT_ONLY.png'),'actual_source_or_rawresponse_reads':0,'model_weights_forward_fit_current_PnP':0,'fullrawtext_not_truncated_or_repaired':True,'input_crop_not_modified':True})

def freeze():
 assert sha(SELECT)==SELECT_SHA and js(ROOT/'GENERATED_RENDERER_TESTS.json')['PASS']
 write(ROOT/'RENDER_PROTOCOL_BEFORE_OUTPUTS.json',{'completed':True,'code':bind(__file__),'runner':bind(RUNNER),'selection':bind(SELECT),'generated':bind(ROOT/'GENERATED_RENDERER_TESTS.json'),'output':str(OUT),'actual_stage_guard':'root completion notification + exact supplied SOURCE_PILOT_SEAL SHA, completed/PASS and all160/122/38 checks, then output/NPZ/canonical RGB hashes before reads','case_selection_unchanged':True,'same_q_L':True,'exact_runner_make_context_AST':True,'both_model_input_context_and_crop96_hash_reconstruction_required':True,'GPU_tensor_dump':False,'weak_AI_category_is_gold':False,'native_semantic_mapping':'UNKNOWN','new_model_forward_fit_current_PnP':0})

def render(seal_path,expected):
 assert sha(seal_path)==expected
 seal=js(seal_path);assert seal['completed'] and seal['PASS'] and seal['frames']==20 and seal['slots']==160 and seal['inference_slots']==122 and seal['U_without_model']==38
 assert sha(SELECT)==SELECT_SHA and sha(RUNNER)==RUNNER_SHA
 protocol=js(ROOT/'RENDER_PROTOCOL_BEFORE_OUTPUTS.json');assert protocol['code']['sha256']==sha(__file__)
 assert not OUT.exists();OUT.mkdir()
 rawdesc=next(x for x in seal['outputs'] if Path(x['path']).name=='ALL160_SOURCE_OBSERVATIONS.jsonl')
 responses=[json.loads(x) for x in check(rawdesc).read_text().splitlines()];assert len(responses)==160
 inputseal=js(check(js(check(seal['entry']))['source_input_seal']))
 packdesc=next(x for x in inputseal['files'] if Path(x['path']).name=='INPUT_RGB96_ONLY.npz');pack=np.load(check(packdesc),allow_pickle=False)
 framefile=check(js(check(inputseal['protocol']))['input_manifest']);frames=[json.loads(x) for x in framefile.read_text().splitlines()];assert len(frames)==20
 selected=js(SELECT)['fixed6'];assert len(selected)==6;make=context_function();records=[];images=[]
 for n,c in enumerate(selected,1):
  i=c['slot_index'];fi=c['frame_index'];k=c['native_slot_logical_only'];row=responses[i];frame=frames[fi]
  assert i==fi*8+k and row['slot_index']==i and row['frame_index']==fi and row['native_slot_logical_only']==k and row['input_id']==c['input_id']==frame['input_id']
  assert row['query_center_valid'] is True and row['model_calls_for_slot']==1 and bool(pack['query_center_valid'][i])
  q=pack['query_xy'][i];L=float(pack['L'][i]);assert np.array_equal(q,np.asarray(c['original_q'])) and np.array_equal(q,np.asarray(row['q'])) and L==c['original_L']==row['L']
  crop=pack['rgb96'][i];assert hashlib.sha256(crop.tobytes()).hexdigest()==row['crop_rgb96_sha256']
  rawpath=check(frame['decoded_RGB_input']);rgb=np.load(rawpath,allow_pickle=False)
  assert rgb.dtype==np.uint8 and list(rgb.shape[:2])==frame['raw_hw'] and hashlib.sha256(rgb.tobytes()).hexdigest()==frame['RGB_pixels_sha256']
  ctx=make(rgb,frame['selected_predictor_box_xyxy'],q);assert hashlib.sha256(np.asarray(ctx).tobytes()).hexdigest()==row['context_overlay_pixels_sha256']
  vlm=Image.fromarray(crop.transpose(1,2,0)).resize((256,256),Image.Resampling.BICUBIC)
  meta={'case_index':n,'slot_index':i,'frame_index':fi,'native_slot':k,'split':c['split'],'input_id':c['input_id'],'q':q.tolist(),'L':L}
  im=figure(ctx,crop,vlm,meta,row,c['weak_whole_crop_evidence']);png=OUT/f'case_{n:02d}_frame{fi:02d}_slot{k}.png';im.save(png);images.append(im)
  records.append({**meta,'weak_whole_crop_evidence':c['weak_whole_crop_evidence'],'weak_evidence_is_not_gold':True,'image':bind(rawpath),'RGB_pixels_sha256':frame['RGB_pixels_sha256'],'crop96_sha256':row['crop_rgb96_sha256'],'context_reconstructed_exact_sha256':row['context_overlay_pixels_sha256'],'vlm256_pixels_sha256':hashlib.sha256(np.asarray(vlm).tobytes()).hexdigest(),'raw_output_row':row,'figure':bind(png),'native8':'UNKNOWN','images_are_preprocessor_views_not_GPU_tensor_dump':True})
 contact=Image.new('RGB',(1800*2,sum(max(images[j].height,images[j+1].height) for j in range(0,6,2))),'white');y=0
 for j in range(0,6,2):
  contact.paste(images[j],(0,y));contact.paste(images[j+1],(1800,y));y+=max(images[j].height,images[j+1].height)
 contact.save(OUT/'ALL_FIXED6_CONTACT.png')
 write(OUT/'ALL_FIXED6_SAVED_VIEW_AND_RESPONSE_PROVENANCE.json',{'completed':True,'selection':bind(SELECT),'inference_seal':bind(seal_path),'records':records,'raw_text_repair':False,'new_model_forward_fit_current_PnP':0})
 write(OUT/'RENDER_SEAL.json',{'completed':True,'PASS':True,'code':bind(__file__),'protocol':bind(ROOT/'RENDER_PROTOCOL_BEFORE_OUTPUTS.json'),'inference_seal':bind(seal_path),'selection':bind(SELECT),'all6_context_and_crop_hashes_exact':True,'files':[bind(x) for x in sorted(OUT.iterdir()) if x.is_file()],'new_model_forward_fit_current_PnP':0})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['generated','freeze','render']);p.add_argument('--seal',type=Path);p.add_argument('--expected-seal-sha');a=p.parse_args()
 if a.stage=='generated':generated()
 elif a.stage=='freeze':freeze()
 else:assert a.seal and a.expected_seal_sha;render(a.seal,a.expected_seal_sha)
