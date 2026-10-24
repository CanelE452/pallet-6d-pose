from pathlib import Path
import hashlib,importlib.util,json,textwrap
import numpy as np
from PIL import Image,ImageDraw,ImageFont
H=Path(__file__).resolve().parent;P=H.parent.parent
OLD=P/'query_visibility_query_anchored_cue_20261023_v96'
CONTEXT=P/'query_visibility_target_context_rgb_20261023_v107'
def bind(p):return dict(path=str(Path(p).resolve()),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def load(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x)for x in Path(p).read_text().splitlines()]
def font(n):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',n)
def read_RGB(c):
 assert bind(c['full_RGB']['path'])==c['full_RGB']
 if c['full_RGB_format']=='CANONICAL_RGB_UINT8_HWC_NPY':a=np.load(c['full_RGB']['path'],allow_pickle=False)
 else:a=np.array(Image.open(c['full_RGB']['path']).convert('RGB'))
 assert a.dtype==np.uint8 and list(a.shape[:2])==c['raw_hw']and hashlib.sha256(a.tobytes()).hexdigest()==c['decoded_RGB_sha256'];return a
def fit(im,size):
 a=im.copy();a.thumbnail(size,Image.Resampling.LANCZOS);out=Image.new('RGB',size,'#18212b');out.paste(a,((size[0]-a.width)//2,(size[1]-a.height)//2));return out
def prediction(z,a):return 'U'if not a else('E'if z>=0 else'D')
def main():
 lock=load(H/'FIXED_FAILURE_SELECTION_BEFORE_RGB.json');s9=np.load(lock['C9_scores']['path'],allow_pickle=False);s10=np.load(lock['C10_scores']['path'],allow_pickle=False)
 lineage=rows(OLD/'actual_source_pair_input_pack/ALL31680_INPUT_LINEAGE.jsonl');base=rows(OLD/'actual_query_anchored_image_intake/SOURCE960_DRIVER_LINEAGE.jsonl');ctx=rows(CONTEXT/'OCCURRENCES33984_TARGET_CONTEXT_INPUT_ONLY.jsonl');defs={r['context_id']:r for r in rows(CONTEXT/'TARGET_CONTEXT_DEFINITIONS216_INPUT_ONLY.jsonl')};targets=rows(OLD/'query_anchored_pair_preparation/MATCHED_TARGET960_GT_ONLY.jsonl')
 spec=importlib.util.spec_from_file_location('fixed_context_math',CONTEXT/'target_context_math.py');math=importlib.util.module_from_spec(spec);spec.loader.exec_module(math)
 file_cache={};results=[];cards=[]
 for ci,case in enumerate(lock['selected'],1):
  folder=H/f'CASE{ci:02d}_{case["category"]}';folder.mkdir();indices=case['display_occurrence_indices'];width=450*len(indices);card=Image.new('RGB',(width,1020),'#101821');d=ImageDraw.Draw(card);d.text((10,10),case['category'],font=font(20),fill='white');d.text((10,40),'FIXED SOURCE FAILURE | C9/C10 saved logits only',font=font(13),fill='#ffdf70');d.text((10,58),'Conditional AUX; native UNKNOWN; model/label changes0',font=font(12),fill='#ffdf70');panelrows=[]
  for j,ix in enumerate(indices):
   line=lineage[ix];c=ctx[ix];definition=defs[c['context_id']];target=targets[line['base_row_index']];q=np.array(line['query_xy']);L=line['L'];assert np.array_equal(q,s10['query_xy_raw'][ix])and L==s10['L_raw'][ix]
   ref=line['input_ref']
   if ref['kind']=='BASE_ROW':rr=base[ref['slot_index']]['before_input_refs']['rgb96'];fpath=rr['file'];key=rr['key'];ri=rr['index']
   else:fpath=ref['file'];key='rgb96';ri=ref['index']
   if fpath['path']not in file_cache:
    assert bind(fpath['path'])=={k:fpath[k]for k in ('path','sha256')}
    if 'bytes'in fpath:assert Path(fpath['path']).stat().st_size==fpath['bytes']
    with np.load(fpath['path'],allow_pickle=False)as packed:file_cache[fpath['path']]=np.array(packed[key],copy=True)
   crop=file_cache[fpath['path']][ri];assert crop.dtype==np.uint8 and crop.shape==(3,96,96)and hashlib.sha256(crop.tobytes()).hexdigest()==line['tensor_value_sha256']['rgb96'];raw=Image.fromarray(crop.transpose(1,2,0));raw.save(folder/f'{j}_QUERY96_RAW.png')
   rgb=read_RGB(c);Image.fromarray(rgb).save(folder/f'{j}_FULL_RGB.png');context,valid=math.sample_target_context(rgb,definition['target_box_xyxy'],(224,224));context_u8=np.floor(np.clip(context,0,255)+.5).astype(np.uint8);ct=Image.fromarray(context_u8);ct.save(folder/f'{j}_TARGET_CONTEXT224_RAW.png')
   overlay=raw.copy();dd=ImageDraw.Draw(overlay);dd.line((43.5,47.5,51.5,47.5),fill='#ff4357');dd.line((47.5,43.5,47.5,51.5),fill='#ff4357')
   for box in target['original_regions']:
    bb=np.array(box['bbox']).reshape(2,2);mapped=(bb-q)/L*96+47.5;dd.rectangle(tuple(mapped.ravel()),outline='#38ff90',width=1)
   overlay.save(folder/f'{j}_QUERY96_Q_SUPPORT_OVERLAY.png')
   full=Image.fromarray(rgb);dd=ImageDraw.Draw(full);box=definition['target_box_xyxy'];dd.rectangle(tuple(box),outline='yellow',width=2);dd.rectangle((q[0]-L/2,q[1]-L/2,q[0]+L/2,q[1]+L/2),outline='#ff4357',width=1);dd.ellipse((q[0]-3,q[1]-3,q[0]+3,q[1]+3),outline='#ff4357',width=2);full.save(folder/f'{j}_FULL_Q_CONTEXT_OVERLAY.png')
   xx=j*450;stage=line['variant'];targetlabel='D'if s10['external_target'][ix]==0 else'E';z9=float(s9['query_logit'][ix]);z10=float(s10['query_logit'][ix]);a9=bool(s9['query_available'][ix]);a10=bool(s10['query_available'][ix]);p9=prediction(z9,a9);p10=prediction(z10,a10)
   d.text((xx+10,72),f'{stage} | row {ix} | GT {targetlabel}',font=font(15),fill='white');card.paste(fit(full,(430,300)),(xx+10,100));d.text((xx+10,407),'Exact fixed target context224',font=font(13),fill='white');d.text((xx+244,407),'Exact saved RGB96',font=font(13),fill='white');card.paste(ct,(xx+10,430));card.paste(raw.resize((192,192),Image.Resampling.NEAREST),(xx+244,430));d.text((xx+10,671),f'C9 z={z9:+.5f} -> {p9}',font=font(17),fill='white');d.text((xx+10,700),f'C10 z={z10:+.5f} -> {p10}',font=font(17),fill='#ffdf70');card.paste(overlay.resize((192,192),Image.Resampling.NEAREST),(xx+244,650));d.text((xx+245,845),'SEPARATE q/support overlay',font=font(11),fill='white')
   lines=[f'q=({q[0]:.3f},{q[1]:.3f}), L={L:.3f}px',f'raw split={line["raw_split"]}',f'context origin={definition["box_origin"]}',f'base={line["base_row_index"]}, rank={line["donor_rank"]}',f'context valid={valid.mean():.4f}',f'expected native part=UNKNOWN']
   yy=865
   for text in lines:
    for part in textwrap.wrap(text,width=55):d.text((xx+10,yy),part,font=font(12),fill='white');yy+=20
   panelrows.append(dict(occurrence_index=ix,occurrence_id=line['occurrence_id'],base_row_index=line['base_row_index'],donor_rank=line['donor_rank'],variant=stage,raw_split=line['raw_split'],q=q.tolist(),L=L,teacher_label=targetlabel,C9_logit=z9,C9_available=a9,C9_prediction=p9,C10_logit=z10,C10_available=a10,C10_prediction=p10,original_support_GT_only=target['original_regions'],context_definition=definition,full_RGB_input_binding=c['full_RGB'],raw_query96=bind(folder/f'{j}_QUERY96_RAW.png'),target_context224=bind(folder/f'{j}_TARGET_CONTEXT224_RAW.png'),support_overlay=bind(folder/f'{j}_QUERY96_Q_SUPPORT_OVERLAY.png'),query96_saved_tensor_SHA_exact=True))
  path=folder/'FIXED_COMPARISON_CARD.png';card.save(path);cards.append(bind(path));results.append(dict(category=case['category'],failure_occurrence_index=case['failure_occurrence_index'],panels=panelrows,card=bind(path)))
 result=dict(completed=True,selection=bind(H/'FIXED_FAILURE_SELECTION_BEFORE_RGB.json'),code=bind(Path(__file__)),context_math=bind(CONTEXT/'target_context_math.py'),cases=results,cards=cards,total_displayed_occurrences=sum(len(c['panels'])for c in results),rawq96_saved_input_hashes_all_exact=True,no_models_or_fit_or_current_or_PnP=True,labels_q_L_unmodified=True)
 with(H/'RENDER_RECEIPTS.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(completed=True,cards=cards,receipt=bind(H/'RENDER_RECEIPTS.json'))))
if __name__=='__main__':main()
