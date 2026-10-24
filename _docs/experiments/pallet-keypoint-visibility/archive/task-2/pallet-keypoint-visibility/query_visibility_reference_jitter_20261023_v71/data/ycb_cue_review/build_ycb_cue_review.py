"""Prepare fixed YCB RGB cue-review material without semantic-label invention.

freeze: metadata only, then render: selected source RGB/masks only. No scores/models.
"""
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import json
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
PARENT = HERE.parents[2]
V63 = PARENT / 'query_visibility_public_real_20261022_v63'
V66 = PARENT / 'query_visibility_controlled_real_20261022_v66'
V69 = PARENT / 'query_visibility_reference_semantic_20261023_v69'
POOL = V66 / 'provenance/TARGETS.jsonl'
COMPOSITES = V66 / 'controlled/COMPOSITES.jsonl'
OLDLOCK = V66 / 'provenance/GENERATION_INPUT_LOCK.json'
OLDSEAL = V66 / 'controlled/SEAL.json'
DENY = V63 / 'contract/EVALUATION_SHA256_DENYLIST.json'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
INPUT_KEYS = {'id','source','split','image','image_sha256','raw_hw','query8','side','box'}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def read(path):
    return json.loads(Path(path).read_text())


def records(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def checked(binding):
    require(sha(binding['path']) == binding['sha256'], 'Binding changed ' + binding['path'])
    return Path(binding['path'])


def write_json(path, value):
    require(not path.exists(), 'Preserve ' + str(path))
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_lines(path, values):
    require(not path.exists(), 'Preserve ' + str(path))
    path.write_text(''.join(json.dumps(x, ensure_ascii=False, allow_nan=False)+'\n' for x in values))


def rank(record):
    return hashlib.sha256((record['image']['sha256']+':'+record['id']).encode()).hexdigest()


def select(pool, split, number):
    selected = []; seen = set()
    for row in sorted((r for r in pool if r['split'] == split), key=rank):
        if row['image']['sha256'] in seen:
            continue
        seen.add(row['image']['sha256']); selected.append(row)
        if len(selected) == number:
            break
    require(len(selected) == number, 'Insufficient source frames')
    return selected


def input_row(record, composite=False):
    result = dict(id=record['id'], source='YCB_CONTROLLED' if composite else 'YCB_ORIGINAL',
                  split=record['split'], image=record['image']['path'], image_sha256=record['image']['sha256'],
                  raw_hw=record['target_original']['raw_hw'] if composite else record['raw_hw'],
                  query8=record['query'], side=record['side'], box=record['box'])
    require(set(result) == INPUT_KEYS and np.asarray(result['query8']).shape == (8,2), 'Input whitelist and qset')
    require(np.isfinite(np.asarray(result['query8'])).all() and result['side'] > 0, 'Finite source inputs')
    return result


def boundary(mask):
    m = np.asarray(mask, bool)
    padded = np.pad(m, 1)
    inside = np.ones_like(m)
    for y in range(3):
        for x in range(3):
            inside &= padded[y:y+m.shape[0], x:x+m.shape[1]]
    return m & ~inside


def boundary_seeds(mask, count=8):
    yy, xx = np.nonzero(boundary(mask))
    if len(xx) == 0:
        return []
    indices = np.unique(np.floor((np.arange(count)+.5)*len(xx)/count).astype(int))
    return [[int(xx[i]),int(yy[i])] for i in indices]


def native_valid(q, hw):
    h,w = hw; q = np.asarray(q, float)
    if not np.isfinite(q).all():
        return False
    x,y = np.floor(q+.5).astype(int)
    return bool(0 <= q[0] < w and 0 <= q[1] < h and 0 <= x < w and 0 <= y < h)


def freeze(out, train_count, val_count):
    out.mkdir(parents=True, exist_ok=True)
    require(not (out/'SELECTION_LOCK.json').exists(), 'Selection already frozen')
    prior = read(OLDLOCK)
    require(ref(POOL) == prior['targets'], 'Frozen source target pool')
    deny = set(read(DENY)['image_sha256']); require(len(deny) == 223, 'Complete exclusion SHA set')
    pool = records(POOL); comp = records(COMPOSITES)
    require(Counter(r['split'] for r in pool) == {'TRAIN':1600,'VALIDATION':400}, 'Original source pool')
    require(Counter(r['split'] for r in comp) == {'TRAIN':1599,'VALIDATION':400}, 'Existing controlled pool')
    lookup = {r['target_original']['id']:r for r in comp}
    require(len(lookup) == len(comp), 'One existing controlled image per target')
    tr = {r['image']['sha256'] for r in pool if r['split']=='TRAIN'}
    va = {r['image']['sha256'] for r in pool if r['split']=='VALIDATION'}
    require(not tr&va and not (tr|va)&deny, 'Source frame split and evaluation exclusions')
    for c in comp:
        t,d = c['target_original'],c['occluder_original']
        require(t['split']==d['split']==c['split'], 'Donor stays in source split')
        require(t['image']['sha256']!=d['image']['sha256'] and t['object_id']!=d['object_id'], 'Distinct donor frame/object')
        require(c['query']==t['query'] and c['side']==t['side'] and c['box']==t['box'], 'Original input recipe unchanged')
        require(not {t['image']['sha256'],d['image']['sha256'],c['image']['sha256']}&deny, 'Forbidden evaluation image SHA')
        allowed_scenes = set(range(8)) if c['split']=='TRAIN' else {8,9}
        require(t['scene_id'] in allowed_scenes and d['scene_id'] in allowed_scenes, 'Scene-disjoint donor/target lineage')
    chosen = select(pool,'TRAIN',train_count)+select(pool,'VALIDATION',val_count)
    protocol = dict(schema='V71_FIXED_YCB_REAL_CUE_INTAKE_PROTOCOL_V1',
      purpose='Prepare actual RGB for reference-cue association review; no semantic reference is invented from masks.',
      selection='SHA256(UTF8(original image SHA256+":"+targetID)) ascending; first distinct original RGB SHA per split. Prefix-stable count expansion; no replacement based on ambiguity, mask geometry, composite availability, or scores.',
      counts=dict(TRAIN=train_count,VALIDATION=val_count),
      splits='Original source scenes0..7 TRAIN and8..9 VALIDATION; composites/donors inherit same split, never reassign by output.',
      model_input_fields=sorted(INPUT_KEYS),
      input_queryset_semantics='Eight preexisting available initial-query rows, not canonical/native semantic8 IDs. All GT r/uncertainty/state/masks and offset remain label-only.',
      GT_review='Keep original target mask proxies and controlled causal-paste lineage separate. Every query site and boundary inspection seed starts semantic U with null reference_xy and null uncertainty. Only independent actual RGB review can propose a new physical cue reference.',
      boundary_seed_recipe='8 evenly spaced quantile indices in row-major visible-inside8-neighbor boundary; inspection points only, not semantic corners/anchors/labels.',
      rendering='Same original RGB and existing composite. Raw context, visible/full/paste raster boundaries, unchanged q8/roughbox/inputL and raw/overlay native96 diagnostic crops. Mask boundaries are not semantic proof.',
      future_builder='After reviewed target-specific cue association, freeze new GT refID/r/visual uncertainty and D_teacher evidence separately. Existing opaque paste can certify E_teacher only if entire fixed reference uncertainty is covered; partial/unknown staysU. Jitter changes only initialq in a separately frozen recipe; no such jitters are generated here.',
      association_limit='Hidden reference identity may be unidentifiable from RGB+q+box alone. Multi-query input is available but no semantic slot token is silently added; this intake does not resolve hidden-ref collisions.',
      proxy_limit='BOP visible/full are rendered/depth-derived with raster/pose alignment caveats. Originalfull-only is not automatically causalE/self; mask0 is not physical absence.',
      box_provenance='Preexisting source annotation-derived jittered roughbox and L are retained; no mask-extrema prompt replacement.',
      source_initializer_evidence='V69 TRAIN residuals describe renderer-cuboid proxy vs saved initializer, not physical GT or real-source jitter calibration; no radius/offset policy selected.',
      labels_admitted_for_semantic_training=0,new_jitters_or_composites=False,
      model_forward_calls=0,training_calls=0,PnP_calls=0,current_data_reads=0)
    write_json(out/'PROTOCOL.json', protocol)
    inputs=[]; lineage=[]
    for r in chosen:
        c=lookup.get(r['id']); inputs.append(input_row(r))
        if c is not None:inputs.append(input_row(c,True))
        lineage.append(dict(id=r['id'],split=r['split'],selection_sha256=rank(r),original=r,
                            controlled=c,controlled_missing=c is None,
                            cue_review_status='PENDING_ACTUAL_RGB_REVIEW',semantic_reference_state='U'))
    write_lines(out/'INPUTS_RGB_ONLY.jsonl', inputs)
    write_lines(out/'SOURCE_LINEAGE_GT_ONLY.jsonl', lineage)
    write_json(out/'SELECTION_LOCK.json', dict(schema='V71_YCB_REVIEW_BEFORE_RGB_SELECTION_LOCK_V1',
        completed=True,source_only=True,selected_RGB_reads_before_lock=0,code=ref(__file__),protocol=ref(out/'PROTOCOL.json'),
        inputs=[ref(x) for x in [POOL,COMPOSITES,OLDLOCK,OLDSEAL,DENY]],
        earlier_metadata_evidence=[ref(V69/'SOURCE_INITIALIZER_RESIDUAL_DIAGNOSIS.json'),ref(V69/'source_anchor_review/SELECTION_AND_REVIEW_PROTOCOL.json'),ref(V69/'source_cue_reference_proposals/CUE_PROPOSAL_SELECTION_LOCK.json')],
        outputs=[ref(out/'INPUTS_RGB_ONLY.jsonl'),ref(out/'SOURCE_LINEAGE_GT_ONLY.jsonl')],
        selected=[dict(id=r['id'],split=r['split'],original_RGB=r['image'],selection_sha256=rank(r))for r in chosen],
        source_frame_overlap=0,source_donor_split_mismatches=0,source_vs_current_SHA_overlap=0,
        scores_or_model_or_current_used_for_selection=False))
    print(json.dumps(ref(out/'SELECTION_LOCK.json')))


def rgb(binding):
    return np.asarray(Image.open(checked(binding)).convert('RGB')).copy()


def label(draw,xy,value,size=22,color='#172536'):
    draw.text(xy,str(value),font=ImageFont.truetype(FONT,size),fill=color)


def overlay(array,vis,full,paste=None):
    a=array.copy();a[boundary(full)]=[65,160,255];a[boundary(vis)]=[0,255,115]
    if paste is not None:a[boundary(paste)]=[255,156,0]
    return a


def panel(canvas,array,window,xy,wh,queries,side,box,seeds=(),title='',site=None):
    x0,y0,x1,y1=window;w,h=wh
    scale=min(w/(x1-x0),h/(y1-y0));rw=max(1,int((x1-x0)*scale));rh=max(1,int((y1-y0)*scale))
    xx=np.floor(x0+(np.arange(rw)+.5)/rw*(x1-x0)+.5).astype(int)
    yy=np.floor(y0+(np.arange(rh)+.5)/rh*(y1-y0)+.5).astype(int)
    inside=(yy[:,None]>=0)&(yy[:,None]<array.shape[0])&(xx[None,:]>=0)&(xx[None,:]<array.shape[1])
    a=array[np.clip(yy,0,array.shape[0]-1)[:,None],np.clip(xx,0,array.shape[1]-1)[None,:]].copy();a[~inside]=0
    im=Image.new('RGB',(w,h),'#e1e6ec');im.paste(Image.fromarray(a),((w-rw)//2,(h-rh)//2));d=ImageDraw.Draw(im)
    def point(q):return ((q[0]-x0)/(x1-x0)*rw+(w-rw)/2,(q[1]-y0)/(y1-y0)*rh+(h-rh)/2)
    if box is not None:
        a,b=point(box[:2]);c,e=point(box[2:]);d.rectangle((a,b,c,e),outline='#00cce9',width=2)
    for k,q in enumerate(queries):
        if site is not None and k!=site:continue
        px,py=point(q)
        if site is not None:
            lo=point([q[0]-side/2,q[1]-side/2]);hi=point([q[0]+side/2,q[1]+side/2]);d.rectangle((*lo,*hi),outline='#ffb037',width=2)
        d.line((px-6,py,px+6,py),fill='#ff00b5',width=2);d.line((px,py-6,px,py+6),fill='#ff00b5',width=2)
        if site is None:d.text((px+4,py+2),'q'+str(k),font=ImageFont.truetype(FONT,17),fill='#ff00b5',stroke_width=1,stroke_fill='#ffffff')
    for k,seed in enumerate(seeds):
        px,py=point(seed);d.ellipse((px-3,py-3,px+3,py+3),outline='#ffea00',width=2)
        d.text((px+3,py-15),'b'+str(k),font=ImageFont.truetype(FONT,14),fill='#ffff00',stroke_width=1,stroke_fill='#000000')
    canvas.paste(im,xy);d=ImageDraw.Draw(canvas);label(d,(xy[0],xy[1]-29),title,21)
    return hashlib.sha256(a.tobytes()).hexdigest()


def render(out):
    lock=read(out/'SELECTION_LOCK.json');checked(lock['code']);require(lock['code']==ref(__file__),'Frozen builder code')
    checked(lock['protocol'])
    for b in lock['inputs']+lock['outputs']:checked(b)
    require(not (out/'REVIEW_MATERIAL_SEAL.json').exists(),'Preserve rendered review')
    lineage=records(out/'SOURCE_LINEAGE_GT_ONLY.jsonl');inp=records(out/'INPUTS_RGB_ONLY.jsonl')
    require(all(set(r)==INPUT_KEYS for r in inp),'Input GT/reference whitelist')
    sheets=out/'TARGET_SHEETS';sheets.mkdir(exist_ok=False)
    templates=[];summaries=[];overviews={'TRAIN':[],'VALIDATION':[]}
    for index,item in enumerate(lineage):
        r=item['original'];c=item['controlled'];a=rgb(r['image']);vis=np.asarray(Image.open(checked(r['visible'])))>0;full=np.asarray(Image.open(checked(r['full_proxy'])))>0
        require(list(a.shape[:2])==r['raw_hw'] and vis.shape==full.shape==a.shape[:2],'Native RGB/mask shape')
        require(not (vis&~full).any(),'Visible subset of full raster proxy')
        seeds=boundary_seeds(vis);comp=None;paste=None
        if c is not None:
            comp=rgb(c['image'])
            with np.load(checked(c['labels']),allow_pickle=False)as z:
                paste=z['paste_mask'];newvis=z['visible_positive']
                require(np.array_equal(z['original_visible'],vis) and np.array_equal(z['original_full'],full),'Original-mask lineage parity')
            require(np.array_equal(a[~paste],comp[~paste]),'Existing composite outside mask unchanged')
        queries=r['query'];box=r['box'];h,w=vis.shape
        lo=np.minimum(np.asarray(box[:2]),np.min(queries,axis=0));hi=np.maximum(np.asarray(box[2:]),np.max(queries,axis=0))
        context=[max(-.5,float(lo[0])-16),max(-.5,float(lo[1])-16),min(w-.5,float(hi[0])+16),min(h-.5,float(hi[1])+16)]
        sheet=Image.new('RGB',(1800,1290),'#f5f7fa');d=ImageDraw.Draw(sheet)
        label(d,(24,18),f"{r['split']} {index+1:02d} · {r['id']} · object {r['object_id']} · L={r['side']:.2f}px",29)
        label(d,(24,62),'선택: source SHA만 사용 | q0..7=기존 query 행, semantic 코너 ID 아님 | 모든 cue 후보 U / RGB 검토 대기',22)
        label(d,(24,102),'초록 visible raster 경계 · 파랑 full proxy 경계 · 주황 paste 경계/입력 L · 노랑 b*=inspection seed, semantic cue 증명 아님',20)
        panel(sheet,a,context,(24,180),(560,420),[],r['side'],None,title='원본 RGB (경계 overlay 없음)')
        panel(sheet,overlay(a,vis,full),context,(620,180),(560,420),queries,r['side'],box,seeds,title='원본 + raster 경계 / 기존 q / roughbox')
        if comp is not None:panel(sheet,overlay(comp,newvis,full,paste),context,(1216,180),(560,420),queries,r['side'],box,title='기존 controlled + 동일 q/box/L')
        else:label(d,(1216,250),'controlled 없음 — 배제하지 않고 U 유지',23)
        query_summary=[]
        for k,q in enumerate(queries):
            x=24+(k%4)*444;y=660+(k//4)*276;px,py=np.floor(np.asarray(q)+.5).astype(int)
            crop=[px-48-.5,py-48-.5,px+48-.5,py+48-.5]
            valid=native_valid(q,[h,w]);before=bool(vis[py,px])if valid else None;covered=bool(paste[py,px])if valid and paste is not None else None
            label(d,(x,y-32),f'q{k} ({q[0]:.1f},{q[1]:.1f}) | semantic U',20)
            rawhash=panel(sheet,a,crop,(x,y),(204,204),queries,r['side'],None,title='',site=k)
            overlayhash=panel(sheet,overlay(a,vis,full),crop,(x+212,y),(204,204),queries,r['side'],None,title='',site=k)
            label(d,(x,y+210),f'raw96 / overlay96  valid={valid}  maskV={before} paste={covered}',15)
            qrow=dict(candidate_id=r['id']+':QUERY_SITE:'+str(k),target_id=r['id'],split=r['split'],candidate_kind='EXISTING_QUERY_INSPECTION_SITE',
                      native_query_row_lineage=k,initial_query_xy=q,inspection_crop_native96=crop,
                      native_valid=valid,mask_visible_at_query=before,existing_paste_at_query=covered,
                      semantic_reference_xy=None,semantic_reference_uncertainty_bbox=None,semantic_reference_state='U',
                      AI_actual_cue_review='PENDING',semantic_or_native8_GT_admitted=False,
                      raw96_display_RGB_bytes_sha256=rawhash,overlay96_display_RGB_bytes_sha256=overlayhash)
            templates.append(qrow);query_summary.append(qrow)
        for k,seed in enumerate(seeds):
            templates.append(dict(candidate_id=r['id']+':BOUNDARY_INSPECTION:'+str(k),target_id=r['id'],split=r['split'],
                              candidate_kind='MASK_BOUNDARY_INSPECTION_SEED_NOT_SEMANTIC_CUE',inspection_seed_xy=seed,
                              semantic_reference_xy=None,semantic_reference_uncertainty_bbox=None,semantic_reference_state='U',
                              AI_actual_cue_review='PENDING',semantic_or_native8_GT_admitted=False))
        label(d,(24,1220),'native96은 진단 창이며 실제 model ROI side=L과 다름. mask 경계/target membership은 물리적 corner/edge cue의 semantic 증명이 아닙니다.',20)
        label(d,(24,1250),'실제 cue를 RGB로 고정·연결한 뒤 GT-only ref/uncertainty를 별도 동결합니다. 모호함·crop MISS·partial cover를 E/O/S로 바꾸지 않습니다.',20)
        name=f"{index+1:02d}_{r['split']}_{r['id'].replace(':','_')}.png";f=sheets/name;sheet.save(f)
        mini=Image.new('RGB',(576,498),'#ffffff');md=ImageDraw.Draw(mini);label(md,(8,4),r['id'],20);label(md,(8,35),r['split']+' · 원본 + query/raster 경계',19)
        panel(mini,overlay(a,vis,full),context,(8,82),(560,400),queries,r['side'],box,seeds,title='')
        overviews[r['split']].append(mini)
        summaries.append(dict(id=r['id'],split=r['split'],sheet=ref(f),original_RGB=r['image'],visible_mask=r['visible'],full_proxy_mask=r['full_proxy'],controlled_RGB=None if c is None else c['image'],controlled_labels=None if c is None else c['labels'],donor=None if c is None else c['occluder_original'],context_window=context,query_sites=query_summary,boundary_inspection_seeds=seeds,actual_semantic_cue_GT_count=0))
    figures=[]
    for split,pages in overviews.items():
        cols=4 if split=='TRAIN' else 2;rows=(len(pages)+cols-1)//cols
        sheet=Image.new('RGB',(cols*576,rows*498+80),'#eef2f6');d=ImageDraw.Draw(sheet)
        label(d,(16,12),split+' 고정 YCB source target · 초록 visible/파랑 full 경계는 semantic cue 증명 아님',26)
        for k,page in enumerate(pages):sheet.paste(page,((k%cols)*576,80+(k//cols)*498))
        f=out/(split+'_SOURCE_OVERVIEW.png');require(not f.exists(),'Preserve overview');sheet.save(f);figures.append(ref(f))
    write_lines(out/'CUE_REVIEW_TEMPLATE_GT_ONLY.jsonl',templates)
    write_json(out/'ACTUAL_SOURCE_REVIEW_MATERIAL.json',dict(completed=True,source_only=True,selection_lock=ref(out/'SELECTION_LOCK.json'),targets=summaries,figures=figures,query_inspection_sites=sum(r['candidate_kind']=='EXISTING_QUERY_INSPECTION_SITE'for r in templates),boundary_inspection_seeds=sum(r['candidate_kind'].startswith('MASK_BOUNDARY')for r in templates),all_candidates_semantic_U=True,semantic_GT_admitted=0,scores_used=0,current_data_reads=0,new_model_forward_calls=0,training_calls=0,PnP_calls=0))
    write_json(out/'REVIEW_MATERIAL_SEAL.json',dict(completed=True,source_only=True,selection_lock=ref(out/'SELECTION_LOCK.json'),code=ref(__file__),runtime=dict(executable=sys.executable,numpy=np.__version__),outputs=[ref(out/'ACTUAL_SOURCE_REVIEW_MATERIAL.json'),ref(out/'CUE_REVIEW_TEMPLATE_GT_ONLY.jsonl')]+figures+[s['sheet']for s in summaries],RGB_targets_read=len(lineage),GT_masks_do_not_enter_model_inputs=True,source_queries_boxes_L_and_old_outputs_unchanged=True,new_model_forward_calls=0,training_calls=0,PnP_calls=0,current_data_reads=0))
    print(json.dumps(ref(out/'REVIEW_MATERIAL_SEAL.json')))


def mock(path):
    m=np.zeros((7,9),bool);m[1:6,2:7]=True;b=boundary(m)
    require(int(b.sum())==16 and not b[3,4] and b[1,2],'Boundary geometry only')
    require(boundary_seeds(m)==boundary_seeds(m),'Deterministic boundary seeds')
    require(native_valid([0.,0.],[7,9]) and not native_valid([8.9,3.],[7,9]) and not native_valid([-0.1,0.],[7,9]),'Continuous+rounded bounds')
    rows=[dict(id='T'+str(i),split='TRAIN',image=dict(sha256=str(i//2)))for i in range(40)]
    chosen=select(rows,'TRAIN',8);require(len({r['image']['sha256']for r in chosen})==8,'Distinct source frames')
    require(select(rows[::-1],'TRAIN',8)==chosen,'Input order independent')
    write_json(Path(path),dict(completed=True,status='PASS',code=ref(__file__),generated_only=True,boundary_not_semantic=True,deterministic_distinct_frame_selection=True,invalid_rows_not_relabelled_absent=True,source_RGB_masks_or_metadata_read=0,model_forward_training_current_PnP=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['mock','freeze','render']);parser.add_argument('--out',type=Path,default=HERE);parser.add_argument('--train-count',type=int,default=8);parser.add_argument('--val-count',type=int,default=4)
    args=parser.parse_args()
    if args.mode=='mock':mock(args.out)
    elif args.mode=='freeze':freeze(args.out,args.train_count,args.val_count)
    else:render(args.out)
