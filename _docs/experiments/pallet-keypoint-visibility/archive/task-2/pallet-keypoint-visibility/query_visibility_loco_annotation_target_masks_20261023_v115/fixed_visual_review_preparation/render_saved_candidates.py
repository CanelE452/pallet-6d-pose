"""V115 fixed saved-only display. Import and generated tests never open source files."""
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import sys

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
V = HERE.parent
OUT = V / 'fixed_visual_review'
SAM = V / 'source_sam_candidates_v1'
STATIC = {
    'renderer': Path(__file__).resolve(), 'generated': HERE/'GENERATED_RENDER_FORMAT_TESTS.json',
    'root_policy': V/'ROOT_SAM_CANDIDATE_POLICY_BEFORE_PIXELS.json',
    'metadata_review': V/'independent_metadata_review/REVIEW.json',
    'frame_inputs': V/'FRAME20_ORIGINAL_QUERY_INPUTS_ONLY.jsonl',
    'query_inputs': V/'QUERY160_ORIGINAL_INPUTS_ONLY.jsonl',
    'target_entities': V/'TARGET20_EXISTING_ANNOTATION_ENTITIES_GT_ONLY.jsonl',
    'request_plan': V/'SAM_REQUEST_PLAN40_METADATA_ONLY.jsonl',
}
CYAN, MAGENTA, YELLOW, ORANGE = (0,220,230), (240,40,220), (255,220,20), (255,135,30)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for b in iter(lambda: stream.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def bind(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def font(size):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)


def wrapped(draw, xy, text, width, size=16, fill=(30,35,42)):
    x, y = xy; f = font(size)
    for paragraph in str(text).split('\n'):
        line = ''
        for word in paragraph.split():
            trial = (line+' '+word).strip()
            if line and draw.textbbox((0,0), trial, font=f)[2] > width:
                draw.text((x,y), line, font=f, fill=fill); y += size+5; line = word
            else:
                line = trial
        draw.text((x,y), line, font=f, fill=fill); y += size+5
    return y


def context_bounds(box, hw):
    """Fixed display-only25% per-side padding, minimum16 native pixels."""
    x0,y0,x1,y1 = box; h,w = hw
    px,py = max(16.,.25*(x1-x0)),max(16.,.25*(y1-y0))
    return [max(0,math.floor(x0-px)), max(0,math.floor(y0-py)),
            min(w,math.ceil(x1+px)), min(h,math.ceil(y1+py))]


def marked(rgb, target, competitor, boxes, queries, roles=('TARGET','COMPETITOR')):
    require(rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3, 'Native RGB')
    result = rgb.copy()
    for role,mask,color in [('TARGET',target,CYAN),('COMPETITOR',competitor,MAGENTA)]:
        if role not in roles or mask is None:
            continue
        require(mask.dtype == np.bool_ and mask.shape == rgb.shape[:2], 'Native bool display mask')
        result[mask] = np.rint(.74*result[mask]+.26*np.asarray(color)).astype(np.uint8)
        inside = mask.copy()
        inside[1:,:] &= mask[:-1,:]; inside[:-1,:] &= mask[1:,:]
        inside[:,1:] &= mask[:,:-1]; inside[:,:-1] &= mask[:,1:]
        result[mask & ~inside] = color
    image = Image.fromarray(result); draw = ImageDraw.Draw(image)
    for name,box,color in [('TARGET',boxes.get('TARGET'),CYAN),('COMPETITOR',boxes.get('COMPETITOR'),MAGENTA),('INITIAL',boxes.get('INITIAL'),YELLOW)]:
        if box is not None and (name in roles or name == 'INITIAL'):
            draw.rectangle(tuple(box), outline=color, width=max(2,round(rgb.shape[1]/500)))
    for s in queries:
        if s['query_xy'] is None:
            continue
        x,y = s['query_xy']
        # Draw original coordinates; Pillow naturally clips display marks, never moves an OOB query.
        if not (-20 <= x <= rgb.shape[1]+20 and -20 <= y <= rgb.shape[0]+20):
            continue
        draw.line((x-6,y,x+6,y),fill=ORANGE,width=2); draw.line((x,y-6,x,y+6),fill=ORANGE,width=2)
        draw.text((x+7,y+3),str(s['native_slot']),font=font(max(13,round(rgb.shape[1]/70))),fill=ORANGE,stroke_width=1,stroke_fill=(0,0,0))
    return image


def panel(canvas, image, bounds, title, nearest=False):
    x,y,w,h = bounds; draw = ImageDraw.Draw(canvas)
    draw.rectangle((x,y,x+w,y+h), fill=(230,233,237), outline=(180,185,190))
    wrapped(draw,(x+10,y+7),title,w-20,16)
    if image is None:
        wrapped(draw,(x+15,y+65),'NULL: missing competitor; no model call and no zero-mask substitute.',w-30,20)
        return
    scale = min((w-12)/image.width,(h-38)/image.height)
    size = (max(1,round(image.width*scale)),max(1,round(image.height*scale)))
    shown = image.resize(size,Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS)
    canvas.paste(shown,(x+(w-size[0])//2,y+33+(h-38-size[1])//2))


def facts(frame, queries):
    noquery = [s['native_slot'] for s in queries if not s['query_exists']]
    oob = [s['native_slot'] for s in queries if s['query_exists'] and not s['query_center_valid']]
    return f"{frame['split']} | native HxW={frame['raw_hw']} | no-query slots={noquery} | OOB/invalid slots={oob} | original L={frame['L']}"


def make_card(frame, queries, entity, rgb, target, competitor, detail=False):
    boxes = dict(TARGET=entity['box_xyxy_raw'], COMPETITOR=None if entity['competitor'] is None else entity['competitor']['box_xyxy_raw'], INITIAL=frame['selected_predictor_box_xyxy'])
    raw = Image.fromarray(rgb)
    marked_all = marked(rgb,target,competitor,boxes,queries)
    width,height = (1600,1300) if detail else (1280,660)
    card = Image.new('RGB',(width,height),(248,249,251)); draw = ImageDraw.Draw(card)
    y = wrapped(draw,(20,14),f"FRAME {frame['frame_index']:02d} | {frame['input_id']} | representative ann {entity['existing_target_ann_id']} | competitor ann {None if entity['competitor'] is None else entity['competitor']['ann_id']}",width-40,22)
    y = wrapped(draw,(20,y+2),facts(frame,queries),width-40,16)
    wrapped(draw,(20,y+2),'Cyan: TARGET candidate + official target box | Magenta: COMPETITOR candidate + official box | Yellow: initializer-selected predictor box | Orange: original q/slot. Association UNKNOWN; masks are predictions, not D/E/S.',width-40,16)
    top = 155
    panel(card,raw,(14,top,width//2-21,420),'RAW native RGB, unmarked (display resize only)')
    panel(card,marked_all,(width//2+7,top,width//2-21,420),'Marked display: separate candidate masks, raw boxes, original q')
    if detail:
        cb = context_bounds(entity['box_xyxy_raw'],frame['raw_hw'])
        panel(card,raw.crop(tuple(cb)),(14,595,515,420),'RAW target-box context; fixed25%/min16 padding',True)
        panel(card,marked(rgb,target,None,boxes,queries,('TARGET',)).crop(tuple(cb)),(542,595,515,420),'TARGET candidate context (cyan); initial box yellow',True)
        cm = None if competitor is None else marked(rgb,None,competitor,boxes,queries,('COMPETITOR',)).crop(tuple(cb))
        panel(card,cm,(1070,595,515,420),'COMPETITOR candidate context (magenta)',True)
        y = wrapped(draw,(20,1030),f"Display-only context native XYXY={cb}. Unchanged q/L; no reassociation, mask-based query correction, semantic labels or training admission. Mask holes show underlying original RGB.",width-40,16)
        for s in queries:
            y = wrapped(draw,(20,y+1),f"slot{s['native_slot']}: q={s['query_xy']} | {s['input_status']} | continuous={s['continuous_valid']} nearest={s['nearest_valid']} | L={s['L']}",width-40,14)
    else:
        wrapped(draw,(20,590),'Candidate overlap/absence does not identify physical ownership, depth order, external occlusion, self occlusion or coordinate correctness. Missing/outside queries remain factual U; no query was moved.',width-40,16)
    return card


def validate_structure(e):
    require(e.get('schema') == 'V115_ROOT_SAVED_VISUAL_RENDER_ENTRY_V1' and e.get('approved') is True and e.get('allow_saved_public_render') is True,'Explicit approved saved-render entry required')
    require(e.get('source_only') is True and e.get('current223') is False and e.get('V112') is False and e.get('model_calls') == 0 and e.get('fit') is False and e.get('PnP') == 0,'Saved public-only scope')
    require(e.get('expected_frame_indices') == list(range(20)) and e.get('detailed_frame_indices') == list(range(6)),'Fixed20/first6 order')
    require(Path(e['output_directory']).resolve() == OUT,'Fixed output directory')
    require(set(e['bindings']) == set(STATIC)|{'sam_output_seal'},'Exact binding keys')
    for name,path in STATIC.items():
        require(set(e['bindings'][name]) == {'path','sha256'} and Path(e['bindings'][name]['path']).resolve() == path,'Static binding path '+name)
    require(Path(e['bindings']['sam_output_seal']['path']).resolve() == SAM/'SAM_CANDIDATE_OUTPUT_SEAL.json','Fixed SAM output seal')


def run(entry_path):
    e = read(entry_path); validate_structure(e)
    require(not OUT.exists(),'No overwrite/retry; preserve any existing rendering')
    for b in e['bindings'].values():
        require(sha(b['path']) == b['sha256'],'Changed bound artifact '+b['path'])
    proof = read(STATIC['generated'])
    require(proof['PASS'] is True and proof['renderer'] == e['bindings']['renderer'] and proof['source_payload_reads'] == 0,'Exact generated format proof')
    require(read(STATIC['metadata_review'])['PASS'] is True,'Completed metadata review')
    seal = read(e['bindings']['sam_output_seal']['path'])
    require(seal['completed'] is True and seal['PASS'] is True and seal['source_only'] is True and seal['training_admission'] is False and seal['current223'] is False and seal['V112'] is False and seal['fit'] is False and seal['PnP'] == 0,'Completed source-only SAM seal')
    for name in ['root_policy','frame_inputs','query_inputs','target_entities','request_plan']:
        require(seal['bindings'][name] == e['bindings'][name],'SAM/original metadata exact '+name)
    # First possible payload byte access occurs only after the approved entry + completed seal above.
    for b in seal['outputs']:
        require(Path(b['path']).resolve().is_relative_to(SAM),'SAM output stays in its fixed directory')
        require(sha(b['path']) == b['sha256'],'SAM saved output hash')
    require(seal['output_requests']['path'] == str(SAM/'OUTPUT_REQUESTS40.jsonl') and seal['query_readouts']['path'] == str(SAM/'QUERY_READOUTS160.jsonl'),'Fixed output filenames')
    require(sha(seal['output_requests']['path']) == seal['output_requests']['sha256'] and sha(seal['query_readouts']['path']) == seal['query_readouts']['sha256'],'Output metadata hash')
    frames,queries,entities,plan = [rows(STATIC[k]) for k in ['frame_inputs','query_inputs','target_entities','request_plan']]
    requests,readouts = rows(seal['output_requests']['path']),rows(seal['query_readouts']['path'])
    require((len(frames),len(queries),len(entities),len(plan),len(requests),len(readouts)) == (20,160,20,40,40,160),'All retained counts')
    for a,b in zip(readouts,queries):
        require(all(a[k] == value for k,value in b.items()),'Original query readout metadata exact')
    for a,b in zip(requests,plan):
        require(a['metadata_request'] == b and a['request_id'] == b['request_id'] and a['executed'] == b['call_eligible_metadata'],'Primary request provenance')
        require((a['mask'] is not None) == b['call_eligible_metadata'],'Missing competitor remains null')
    OUT.mkdir(); (OUT/'overview_cards').mkdir(); (OUT/'detail_cards').mkdir()
    artifacts,frame_receipts = [],[]; thumbs = []
    write(OUT/'RUN_STARTED.json',dict(entry=bind(entry_path),sam_output_seal=e['bindings']['sam_output_seal'],scope='Saved render only; no model calls'))
    output_shas = {b['path']:b['sha256'] for b in seal['outputs']}
    for i,frame in enumerate(frames):
        require(frame['frame_index'] == i and entities[i]['frame_index'] == i,'Original frame order')
        b = frame['decoded_RGB_input']; require(sha(b['path']) == b['sha256'],'Canonical NPY byte hash')
        rgb = np.load(b['path'],allow_pickle=False)
        require(rgb.dtype == np.uint8 and rgb.shape == tuple(frame['raw_hw'])+(3,),'Canonical native RGB uint8 HWC')
        require(hashlib.sha256(rgb.tobytes()).hexdigest() == frame['RGB_pixels_sha256'],'Exact initializer pixel SHA')
        masks=[]
        for req in requests[2*i:2*i+2]:
            if req['mask'] is None:
                masks.append(None); continue
            mb=req['mask']; require(output_shas.get(mb['path']) == mb['sha256'],'Primary mask bound in output seal')
            with np.load(mb['path'],allow_pickle=False) as z:
                require(z.files == ['visible_mask'],'Exact native mask key'); mask=z['visible_mask']
            require(mask.dtype == np.bool_ and mask.shape == tuple(frame['raw_hw']),'Native bool mask shape, no coercion/resampling')
            masks.append(mask)
        card=make_card(frame,queries[8*i:8*i+8],entities[i],rgb,*masks)
        path=OUT/'overview_cards'/f'FRAME_{i:02d}_RAW_AND_MARKED.png'; card.save(path); artifacts.append(bind(path))
        thumbs.append(card.resize((640,330),Image.Resampling.LANCZOS))
        if i < 6:
            detail=make_card(frame,queries[8*i:8*i+8],entities[i],rgb,*masks,detail=True)
            path=OUT/'detail_cards'/f'FRAME_{i:02d}_DETAIL.png'; detail.save(path); artifacts.append(bind(path))
        frame_receipts.append(dict(frame_index=i,input_id=frame['input_id'],canonical_RGB=b,verified_pixel_sha256=frame['RGB_pixels_sha256'],request_ids=[r['request_id'] for r in requests[2*i:2*i+2]],mask_bindings=[r['mask'] for r in requests[2*i:2*i+2]],native_mask_true_areas=[None if m is None else int(m.sum()) for m in masks],factual_query_status=[s['input_status'] for s in queries[8*i:8*i+8]],display_context_xyxy=context_bounds(entities[i]['box_xyxy_raw'],frame['raw_hw'])))
    for sheet_idx in range(2):
        sheet=Image.new('RGB',(1280,1650),(255,255,255))
        for local in range(10):
            sheet.paste(thumbs[10*sheet_idx+local],((local%2)*640,(local//2)*330))
        path=OUT/f'OVERVIEW_{sheet_idx+1:02d}_FRAMES_{10*sheet_idx:02d}_{10*sheet_idx+9:02d}.png';sheet.save(path);artifacts.append(bind(path))
    receipt=dict(completed=True,PASS=True,entry=bind(entry_path),bindings=e['bindings'],frames=20,overview_cards=20,overview_sheets=2,detailed_frames=list(range(6)),original_queries=160,frame_receipts=frame_receipts,figures=artifacts,runtime=dict(Python=platform.python_version(),NumPy=np.__version__,Pillow=PIL.__version__),display=dict(context_padding='25% per axis per side,min16px; native clipping for display only',full_resample='LANCZOS',context_resample='NEAREST',target_color=CYAN,competitor_color=MAGENTA,initial_box_color=YELLOW,q_color=ORANGE),actual_root_visual_review_completed=False,model_calls=0,fit=False,current223=False,V112=False,PnP=0,labels_created=False,query_updates=0,native_mapping='UNKNOWN',training_admission=False)
    write(OUT/'RENDERING_RECEIPT.json',receipt)
    write(OUT/'RENDERING_SEAL.json',dict(completed=True,PASS=True,receipt=bind(OUT/'RENDERING_RECEIPT.json'),figures=artifacts,entry=bind(entry_path),source_only=True,model_calls=0,current223=False,V112=False,fit=False,PnP=0))
    print(json.dumps(dict(completed=True,receipt=bind(OUT/'RENDERING_RECEIPT.json'),seal=bind(OUT/'RENDERING_SEAL.json'),figures=len(artifacts))))


def template(path):
    b={name:bind(p) for name,p in STATIC.items()}
    b['sam_output_seal']=dict(path=str(SAM/'SAM_CANDIDATE_OUTPUT_SEAL.json'),sha256='ROOT_FILL_COMPLETED_SAM_OUTPUT_SEAL_SHA256')
    write(path,dict(schema='V115_ROOT_SAVED_VISUAL_RENDER_ENTRY_V1',approved=False,allow_saved_public_render=False,source_only=True,current223=False,V112=False,model_calls=0,fit=False,PnP=0,expected_frame_indices=list(range(20)),detailed_frame_indices=list(range(6)),output_directory=str(OUT),bindings=b))


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['run','template']);ap.add_argument('--entry');ap.add_argument('--output');args=ap.parse_args()
    if args.mode == 'run':
        require(args.entry is not None,'--entry required');run(args.entry)
    else:
        require(args.output is not None,'--output required');template(args.output)
