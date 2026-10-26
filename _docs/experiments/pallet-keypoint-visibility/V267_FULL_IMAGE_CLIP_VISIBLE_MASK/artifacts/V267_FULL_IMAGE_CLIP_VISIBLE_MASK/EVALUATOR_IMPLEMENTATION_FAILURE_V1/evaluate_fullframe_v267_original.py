#!/usr/bin/env python3
"""Frozen full-image visible-mask evaluation, explicit point-presence proxy only.

Run after CURRENT_PREDICTIONS_FROZEN.json. Portable metrics replay reads compact
local tables and prediction JSONL, without RGB, models, checkpoints or solvers.
"""
from pathlib import Path
import argparse
from collections import Counter
import csv
import datetime as dt
import hashlib
import html
import io
import json
import math
import os
import stat
import sys
import types

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W = Path(__file__).absolute().parent
R = W.parent
NEW_ARMS = ('FULL_CLIP', 'FULL_CLIP_SURGERY_ARCH')
MODELS = (*NEW_ARMS, 'V264_HEATMAP_LOCAL', 'v61', 'CAL_LEGACY')
GROUPS = ('ALL', 'WOOD_PRIMARY105', 'SQUARE118')
PROXY = 'MASK_PRESENCE_PROXY_E_NOT_CAUSE'
EVALUATOR_PATH = R / 'V264_HEATMAP_LOCAL_VISIBILITY/evaluate_and_visualize_v264.py'
EVALUATOR_SHA = 'e9408144ba952789e487a02f1d4e4d34a9b39c202f25113fc32224e569e2ffeb'
PRIOR_PATH = R / 'V264_HEATMAP_LOCAL_VISIBILITY/CURRENT_ALL1784_PREDICTIONS.jsonl'
PRIOR_SHA = '46fc19cfa3aee116075869382073e7e488c5e604538d233d1070682f2d4c2736'
INPUT_COMPLETE_PATH = W / 'current_fullframe_inputs_v2/COMPLETE.json'
INPUT_COMPLETE_SHA = '83913adc35316e6dbf226d7c593f1307c06ca0d33399264fb0bb0f7109705e10'
IO_HELPER_PATH = W / 'probe_current_fullframes_v267.py'
IO_HELPER_SHA = 'c3463ef12ef245fdf156d77c8e3f6c911e3d343d64db1ca0244b67739726b84a'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def need(v, message):
    if not bool(v):
        raise ValueError(message)


def safe(value, fresh=False):
    p = Path(value)
    need(p.is_absolute() and '..' not in p.parts, 'Absolute nontraversal path required')
    need(not any(x in str(p).casefold() for x in ('v112','cache','.pyc')), 'Forbidden lexical path')
    for a in [*reversed(p.parents), p]:
        try:
            mode = os.lstat(a).st_mode
        except FileNotFoundError:
            continue
        need(not stat.S_ISLNK(mode), 'Symlink ancestor forbidden: ' + str(a))
    if fresh:
        need(not p.exists(), 'Immutable output exists: ' + str(p))
    return p


def raw(path):
    return safe(path).read_bytes()


def bind(path):
    p = safe(path)
    b = p.read_bytes()
    return dict(path=str(p), sha256=hashlib.sha256(b).hexdigest(), size_bytes=len(b))


def checked(b):
    need(isinstance(b,dict) and all(k in b for k in ('path','sha256','size_bytes')), 'Complete binding required')
    data = raw(b['path'])
    need(len(data)==b['size_bytes'] and hashlib.sha256(data).hexdigest()==b['sha256'], 'SHA/size mismatch: '+b['path'])
    return data


def dump(path, value):
    p = safe(path, True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return bind(p)


def table(path, rows):
    need(bool(rows), 'Nonempty output table')
    p = safe(path, True)
    with p.open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    return bind(p)


def jsonl(b):
    return [json.loads(line) for line in b.splitlines() if line.strip()]


def module(path, expected_sha, name):
    p=safe(path)
    b=p.read_bytes()
    need(hashlib.sha256(b).hexdigest()==expected_sha,'Pinned CODE mismatch: '+str(p))
    m=types.ModuleType(name)
    m.__file__=str(p)
    sys.modules[name]=m
    exec(compile(b,str(p),'exec'),m.__dict__)
    return m


def sigmoid(x):
    return 1/(1+math.exp(-max(-700,min(700,float(x)))))


def sample_border(array, q, raw_hw):
    """NumPy equivalent of grid_sample align_corners=False, padding_mode=border."""
    h,w=raw_hw
    x=np.clip((float(q[0])+.5)/w*array.shape[1]-.5,0,array.shape[1]-1)
    y=np.clip((float(q[1])+.5)/h*array.shape[0]-.5,0,array.shape[0]-1)
    x0,y0=int(math.floor(x)),int(math.floor(y))
    x1,y1=min(x0+1,array.shape[1]-1),min(y0+1,array.shape[0]-1)
    dx,dy=x-x0,y-y0
    return float(array[y0,x0]*(1-dx)*(1-dy)+array[y0,x1]*dx*(1-dy)+array[y1,x0]*(1-dx)*dy+array[y1,x1]*dx*dy)


def validate_rows(predictions, queries=None):
    need(len(predictions)==1784,'Exactly1784 full-image query predictions')
    for i,p in enumerate(predictions):
        need(p['global_index']==i and p['native_slot']==i%8,'Fixed global/native order')
        need(p['row_id']==p['frame_id']+':native'+str(i%8),'Fixed row identity')
        need(type(p['available']) is bool,'Boolean original availability')
        if queries is not None:
            q=queries[i]
            need(q['global_index']==i and q['frame_id']==p['frame_id'] and q['native_slot']==p['native_slot'],'Full-image input identity')
            need(q['available']==p['available'],'Original45 unavailable inputs preserved')
        for arm in NEW_ARMS:
            a=p[arm]
            if not p['available']:
                need(a['pred']=='U','Original unavailable input remains U')
                continue
            z,prob,pt=float(a['E_logit']),float(a['p_E']),float(a['p_target'])
            need(all(math.isfinite(x) for x in (z,prob,pt)) and 0<=pt<=1 and 0<=prob<=1,'Finite sampled proxy')
            need(abs(prob-sigmoid(z))<=2e-6 and abs(pt-sigmoid(-z))<=2e-6,'Sample raw masklogit then sigmoid')
            need(a['pred']==('E' if z>0 else 'D'),'Fixed Ptarget>=.5 D / <.5 E proxy')
    need(sum(p['available'] for p in predictions)==1739,'1739 supported /45 original U')


def freeze_inputs():
    """Only CODE, model outputs and original input metadata; no late annotations."""
    seal_path=W/'CURRENT_PREDICTIONS_FROZEN.json'
    seal=json.loads(raw(seal_path))
    need(seal.get('completed') is True and seal.get('PASS',True) is True,'Prediction freeze incomplete')
    config=json.loads(checked(seal['configuration']))
    need(config.get('completed') is True or config.get('PASS') is True,'Configuration freeze incomplete')
    need(seal.get('sample_padding_mode','border')=='border','Explicit fixed border sampling')
    pred_b=seal['predictions']
    map_b=seal['frame_masks']
    need(Path(pred_b['path'])==W/'CURRENT_ALL1784_PREDICTIONS.jsonl','Exact current prediction path')
    need(Path(map_b['path'])==W/'CURRENT_FRAME_MASKS.npz','Exact full-frame map path')
    complete_bytes=raw(INPUT_COMPLETE_PATH)
    need(hashlib.sha256(complete_bytes).hexdigest()==INPUT_COMPLETE_SHA,'Pinned input-only fullframe probe')
    input_complete=json.loads(complete_bytes)
    need(input_complete['PASS'] and input_complete['accessible_verified_frames']==223,'Full223 input verified')
    frame_b=seal['current_frame_inputs']
    need(frame_b==input_complete['frames'],'Inference frame bindings match exact canonical fullRGB223 inputs')
    frame_bytes=checked(frame_b)
    query_bytes=checked(input_complete['queries_manifest'])
    frames=jsonl(frame_bytes)
    queries=jsonl(query_bytes)
    need(len(frames)==223 and len(queries)==1784,'Fixed223frames/1784queries')
    predictions=jsonl(checked(pred_b))
    validate_rows(predictions,queries)
    with np.load(io.BytesIO(checked(map_b)),allow_pickle=False) as z:
        need('frame_index' in z.files and np.array_equal(z['frame_index'],np.arange(223)),'Map223 exact input frame indices')
        maps={a:np.array(z[a],dtype=float,copy=True) for a in NEW_ARMS}
        logits={a:np.array(z[a+'_logit'],dtype=float,copy=True) for a in NEW_ARMS}
    max_sample_diff=0.
    for arm in NEW_ARMS:
        need(maps[arm].shape==logits[arm].shape==(223,32,32),'Full-frame probability/logit grids223x32x32')
        need(np.isfinite(maps[arm]).all() and np.isfinite(logits[arm]).all(),'Finite full maps')
        need((maps[arm]>=0).all() and (maps[arm]<=1).all(),'Probability maps in0..1')
        recomputed=1/(1+np.exp(-np.clip(logits[arm],-700,700)))
        need(np.max(np.abs(recomputed-maps[arm]))<=2e-6,'Stored map is sigmoid(rawlogit)')
        for i,p in enumerate(predictions):
            if not p['available']:
                continue
            q=queries[i]
            sample=sample_border(logits[arm][i//8],q['original_q0'],q['raw_hw'])
            diff=abs(sample+float(p[arm]['E_logit']))
            max_sample_diff=max(max_sample_diff,diff)
            need(diff<=.002,'Query sampled logit inconsistent with full-image map: '+str(i))
    return predictions,frames,queries,maps,logits,dict(
        prediction_freeze=bind(seal_path),configuration=seal['configuration'],
        predictions=pred_b,frame_masks=map_b,current_frame_inputs=frame_b,
        input_complete=bind(INPUT_COMPLETE_PATH),query_inputs=input_complete['queries_manifest'],
        max_abs_numpy_vs_saved_sample_logit_difference=max_sample_diff)


def compute(e, ledger, new, prior):
    validate_rows(new)
    need(len(ledger)==len(prior)==1784,'All fixed1784 ledger/prior')
    for i,(row,p,b) in enumerate(zip(ledger,new,prior)):
        need(int(row['global_index'])==p['global_index']==b['global_index']==i,'Fixed evaluation order')
        need(row['row_id']==p['row_id']==b['row_id'],'Exact evaluation row identity')
        need((row['original_query_valid']=='True')==p['available']==b['available'],'Same original availability')
    pp={a:[p[a]['pred'] for p in new] for a in NEW_ARMS}
    pp.update(V264_HEATMAP_LOCAL=[p['HEATMAP_LOCAL']['pred'] for p in prior],
              v61=[r['v61_D_E_U'] for r in ledger],CAL_LEGACY=[r['legacy_D_E_U'] for r in ledger])
    cohorts={
        'PRIMARY':[r for r in ledger if e.primary(r)],
        'ALL_REGISTERED_D_E':[r for r in ledger if r['human_state'] in (e.D,e.E)],
        'MANUAL_CLICK_D_ALL':[r for r in ledger if e.manual_d(r)],
        'MANUAL_CLICK_D_LE3_VALID_INPUT':[r for r in ledger if e.near_d(r)]}
    need(len(cohorts['PRIMARY'])==1237 and sum(r['human_state']==e.D for r in cohorts['PRIMARY'])==1147,'PRIMARY1237 D1147/E90')
    classification,self_rows,transitions=[],[],[]
    for group in GROUPS:
        for cohort,rows in cohorts.items():
            take=[r for r in rows if group=='ALL' or r['group']==group]
            for model in MODELS:
                classification.append(dict(group=group,cohort=cohort,model=model,
                    prediction_semantics=PROXY if model in NEW_ARMS else 'PRIOR_VISIBILITY_CLASSIFIER',
                    **e.cm(take,pp[model])))
            for arm in NEW_ARMS:
                transitions.append(dict(group=group,cohort=cohort,model=arm,
                    **e.transition(take,pp['V264_HEATMAP_LOCAL'],pp[arm])))
        take=[r for r in ledger if r['human_state']==e.S and (group=='ALL' or r['group']==group)]
        for model in MODELS:
            count=Counter(pp[model][int(r['global_index'])] for r in take)
            self_rows.append(dict(group=group,model=model,N=len(take),response_D=count['D'],
                response_E=count['E'],response_U=count['U'],
                interpretation='SELF is separate. Binary mask target/non-target proxy is not learned self-occlusion correctness.'))
    external=[]
    for row in cohorts['PRIMARY']:
        if row['human_state']!=e.E:
            continue
        i=int(row['global_index'])
        prior_miss=pp['V264_HEATMAP_LOCAL'][i]=='D'
        r=dict(global_index=i,row_id=row['row_id'],group=row['group'],
            coordinate_source=row['annotation_coordinate_source'],prior_missed=prior_miss,
            user_disputed_annotation=i==996,V264_HEATMAP_LOCAL=pp['V264_HEATMAP_LOCAL'][i],
            interpretation='E output from mask means NON_TARGET_AT_QUERY, not identified foreign occluder.')
        for arm in NEW_ARMS:
            pred=pp[arm][i]
            r[arm]=pred
            r[arm+'_transition']=('RECOVERED_PRIOR_MISS' if pred=='E' else 'REMAINED_PRIOR_MISS') if prior_miss else ('KEPT_PRIOR_CORRECT' if pred=='E' else 'REGRESSED_PRIOR_CORRECT')
        external.append(r)
    need(len(external)==90 and sum(r['prior_missed'] for r in external)==67,'Same90 E/prior67misses')
    ext_summary=[]
    for group in GROUPS:
        rows=[r for r in external if group=='ALL' or r['group']==group]
        for arm in NEW_ARMS:
            count=Counter(r[arm+'_transition'] for r in rows)
            ext_summary.append(dict(group=group,model=arm,N=len(rows),
                prior_misses=sum(r['prior_missed'] for r in rows),
                **{k:count[k] for k in ('RECOVERED_PRIOR_MISS','REMAINED_PRIOR_MISS','KEPT_PRIOR_CORRECT','REGRESSED_PRIOR_CORRECT')}))
    selection=[]
    pattern_models=('V264_HEATMAP_LOCAL',*NEW_ARMS)
    for group in GROUPS[1:]:
        for truth in ('D','E'):
            for bits in ('000','001','010','011','100','101','110','111'):
                ids=[]
                for r in cohorts['PRIMARY']:
                    i=int(r['global_index'])
                    if r['group']==group and e.truth(r)==truth and ''.join('1' if pp[a][i]==truth else '0' for a in pattern_models)==bits:
                        ids.append(i)
                selection.append(dict(group=group,truth=truth,correctness_pattern=bits,census_N=len(ids),
                    selected_global_index=min(ids) if ids else None,empty=not ids))
    return dict(primary_counts=dict(N=1237,D=1147,E=90),classification=classification,
        human_SELF=self_rows,paired_transitions=transitions,external90_rows=external,
        external90_transition_summary=ext_summary,fixed_pattern_selection=selection,
        new_prediction_semantics=PROXY,real_pixel_mask_GT_available=False)


def metric_outputs(out,result,ledger,new,prior):
    joined=[]
    for i,(row,p) in enumerate(zip(ledger,new)):
        r={k:row[k] for k in ('global_index','row_id','id','native_corner','group','human_state',
             'annotation_coordinate_source','initial_reference_error_px','original_query_valid')}
        r['PRIMARY']=row['paired_same_query_RGB_D_vs_E']=='True' and row['human_state'] in ('DIRECT_VISIBLE','EXTERNAL_OCCLUDED')
        for a in NEW_ARMS:
            for key in ('pred','p_target','E_logit','p_E'):
                r[a+'_'+key]=p[a].get(key)
        r['V264_HEATMAP_LOCAL_pred']=prior[i]['HEATMAP_LOCAL']['pred']
        r['v61_pred']=row['v61_D_E_U']
        r['CAL_LEGACY_pred']=row['legacy_D_E_U']
        joined.append(r)
    return dict(classification=table(out/'CLASSIFICATION.csv',result['classification']),
        SELF=table(out/'SELF_RESPONSES.csv',result['human_SELF']),
        external90=table(out/'EXTERNAL90_TRANSITIONS.csv',result['external90_rows']),
        external_summary=table(out/'EXTERNAL90_TRANSITION_SUMMARY.csv',result['external90_transition_summary']),
        paired_transitions=dump(out/'PAIRED_TRANSITIONS.json',result['paired_transitions']),
        all1784=table(out/'ALL1784_COMPARISON.csv',joined))


def compact(out,ledger_b,new_b,prior_b):
    dest=safe(out/'compact_inputs',True)
    dest.mkdir()
    payloads={'LEDGER.csv':ledger_b,'NEW_PREDICTIONS.jsonl':new_b,
              'V264_PRIOR_PREDICTIONS.jsonl':prior_b,'V264_EVALUATOR_CODE.py':raw(EVALUATOR_PATH)}
    manifest={}
    for name,b in payloads.items():
        p=safe(dest/name,True)
        p.write_bytes(b)
        manifest[name]=dict(sha256=hashlib.sha256(b).hexdigest(),size_bytes=len(b))
    return dump(dest/'INTEGRITY.json',dict(completed=True,files=manifest,
        no_RGB_models_or_fullmap_required=True,prediction_semantics=PROXY))


def replay(inputs,out):
    integrity=json.loads(raw(inputs/'INTEGRITY.json'))
    data={}
    for name in ('LEDGER.csv','NEW_PREDICTIONS.jsonl','V264_PRIOR_PREDICTIONS.jsonl','V264_EVALUATOR_CODE.py'):
        b=raw(inputs/name)
        need(hashlib.sha256(b).hexdigest()==integrity['files'][name]['sha256'] and len(b)==integrity['files'][name]['size_bytes'],'Compact input integrity')
        data[name]=b
    e=module(inputs/'V264_EVALUATOR_CODE.py',EVALUATOR_SHA,'v267_replay_metric_api')
    ledger=list(csv.DictReader(io.StringIO(data['LEDGER.csv'].decode('utf-8-sig'))))
    new=jsonl(data['NEW_PREDICTIONS.jsonl'])
    prior=jsonl(data['V264_PRIOR_PREDICTIONS.jsonl'])
    result=compute(e,ledger,new,prior)
    outputs=metric_outputs(out,result,ledger,new,prior)
    summary=dump(out/'SUMMARY.json',dict(completed=True,PASS=True,mode='COMPACT_METRICS_REPLAY',
        **result,outputs=outputs,new_RGB_or_model_reads=0,new_optimizer_or_PnP=0))
    return dump(out/'COMPLETE.json',dict(completed=True,PASS=True,summary=summary,
        integrity=bind(inputs/'INTEGRITY.json'),code=bind(Path(__file__).absolute())))


def font(size):
    return ImageFont.truetype(FONT,size)


def text(draw,string,xy,size=20,color='#172238'):
    draw.text(xy,string,font=font(size),fill=color)


def fit_panel(image,size=(448,336)):
    canvas=Image.new('RGB',size,'#17212e')
    thumb=image.copy()
    thumb.thumbnail(size,Image.Resampling.BILINEAR)
    canvas.paste(thumb,((size[0]-thumb.width)//2,(size[1]-thumb.height)//2))
    return canvas


def annotated_image(image,frame,selected):
    """Initial eight native queries only; no GT marker or recenter."""
    image=image.copy()
    d=ImageDraw.Draw(image)
    scale=max(image.width,image.height)/640
    for slot,(q,available) in enumerate(zip(frame['original_query_xy8'],frame['original_query_available8'])):
        if not available:
            continue
        x,y=map(float,q)
        rr=max(3,round(4*scale))
        color='#fff43b' if slot==selected%8 else '#19dcff'
        d.ellipse((x-rr,y-rr,x+rr,y+rr),outline=color,width=max(1,round(2*scale)))
        d.text((x+rr+1,y-rr-1),str(slot),font=font(max(11,round(12*scale))),fill=color)
    return image


def panels_for_frame(rgb,frame,index,maps):
    full=Image.fromarray(rgb,'RGB')
    original_size=full.size
    panels=[('전체 RGB · 모델 입력은 이 전체 영상',annotated_image(full,frame,index))]
    prob_panels=[]
    overlay_panels=[]
    for arm in NEW_ARMS:
        p=maps[arm][frame['frame_index']]
        # Fixed [0,1] color scale; map first warped back to full RGB geometry.
        pbig=np.asarray(Image.fromarray(p.astype('float32'),'F').resize(original_size,Image.Resampling.BILINEAR))
        color=np.stack((20+20*pbig,30+220*pbig,25+65*pbig),-1).clip(0,255).astype('uint8')
        pm=annotated_image(Image.fromarray(color,'RGB'),frame,index)
        prob_panels.append((arm+' · 전체 P(target) 0–1',pm))
        mask=pbig>=.5
        tint=np.array(full,copy=True)
        tint[mask]=(tint[mask]*.50+np.array([25,230,80])*.50).astype('uint8')
        overlay_panels.append((arm+' · 고정 0.5 마스크',annotated_image(Image.fromarray(tint),frame,index)))
    return panels+prob_panels+overlay_panels


def card(index,ledger,new,prior,frame,rgb,maps,zoom,out):
    row,p=ledger[index],new[index]
    image=Image.new('RGB',(1440,1060),'white')
    d=ImageDraw.Draw(image)
    text(d,f"G{index} · {row['row_id']} · {row['human_state']}",(20,12),25)
    text(d,'새 모델의 E는 query 위치의 non-target proxy이며 외부 가림 원인을 분류한 출력이 아닙니다.',(20,49),19)
    text(d,f"기존 V264: {prior[index]['HEATMAP_LOCAL']['pred']} · v61: {row['v61_D_E_U']} · CAL: {row['legacy_D_E_U']}",(20,78),19)
    for j,a in enumerate(NEW_ARMS):
        z=p[a]
        pt='—' if z.get('p_target') is None else f"{float(z['p_target']):.3f}"
        text(d,f"{a}: {z['pred']} · P(target@원래q)={pt}",(20,105+j*27),19)
    panels=panels_for_frame(rgb,frame,index,maps)
    zoom_image=Image.fromarray(zoom,'RGB').resize((336,336),Image.Resampling.NEAREST)
    zd=ImageDraw.Draw(zoom_image)
    zd.ellipse((162,162,174,174),outline='#fff43b',width=3)
    panels.append(('확대 crop · 표시 전용 · 모델 입력 아님',zoom_image))
    for j,(title,panel) in enumerate(panels):
        x=20+(j%3)*480
        y=206+(j//3)*406
        text(d,title,(x,y-31),17)
        image.paste(fit_panel(panel),(x,y))
    text(d,'노랑○ 선택한 초기 native 점 · 파랑○ 나머지 초기 점(번호0–7) · 색 범위는 모든 영상에서 고정0–1',(20,965),17)
    caution='G996: 기존 E 주석과 사용자 해석 이견 유지; 명백한 실패로 단정하지 않습니다.' if index==996 else '실사 픽셀 마스크 정답이 없으므로 이 그림은 모델 출력의 시각 점검입니다. 실사 IoU를 주장하지 않습니다.'
    text(d,caution,(20,995),17,'#8b421e')
    text(d,'숨은 점의 참조 좌표로 다시 자르거나 점을 옮기지 않았습니다. 오른쪽 아래 crop만 표시를 위해 확대했습니다.',(20,1025),16)
    path=safe(out/f'CASE_G{index:04d}.png',True)
    image.save(path)
    return dict(global_index=index,row_id=row['row_id'],frame_index=index//8,
        frame_id=frame['frame_id'],human_state=row['human_state'],
        original_q0=frame['original_query_xy8'][index%8],user_disputed_annotation=index==996,
        new_outputs={a:p[a] for a in NEW_ARMS},V264_output=prior[index]['HEATMAP_LOCAL'],
        full_RGB_decoded_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
        full_RGB_encoded_binding=frame['image_binding'],
        display_only_zoom_RGB96_sha256=hashlib.sha256(zoom.tobytes()).hexdigest(),card=bind(path))


def visualize(e,out,ledger,new,prior,frames,maps,result):
    selected=[r['global_index'] for r in result['external90_rows']]
    selected += [s['selected_global_index'] for s in result['fixed_pattern_selection'] if not s['empty']]
    selected += [370,908,1515,996]
    selected=list(dict.fromkeys(selected))
    selection=dump(out/'CASE_SELECTION_BEFORE_RGB.json',dict(completed=True,
        external90_all=[r['global_index'] for r in result['external90_rows']],
        fixed_patterns=result['fixed_pattern_selection'],predefined=[370,908,1515,996],
        selected_indices=selected,prior_RGB_exposure=True,
        policy='All90 PRIMARY E, minimum G per material x D/E x three-arm correctness pattern; fixed cone-related and disputed examples. Prior RGB exposure is explicit; not blinded.',
        cone_membership='Not all external-E or prior67 misses are asserted to be cone occlusion.'))
    io_api=module(IO_HELPER_PATH,IO_HELPER_SHA,'v267_exact_fullRGB_io')
    # Saved96 crop is DISPLAY ONLY; no new model reads it.
    old_complete=json.loads(e.checked(e.PINNED['v263_current_complete']))
    crop_b=old_complete['crop_cache']['binding']
    with np.load(io.BytesIO(checked(crop_b)),allow_pickle=False) as z:
        zooms=z['RGB96_initial_geometry']
    rgb_by_frame={}
    for fi in sorted({i//8 for i in selected}):
        rgb_by_frame[fi]=io_api.read_rgb(frames[fi])
    cards=[]
    for i in selected:
        cards.append(card(i,ledger,new,prior,frames[i//8],rgb_by_frame[i//8],
                          maps,e.rgb_array(zooms[i]),out))
    # Compact full RGB selection supports independent visual recreation.
    dest=safe(out/'compact_visual_inputs',True)
    dest.mkdir()
    frame_outputs=[]
    for fi,rgb in rgb_by_frame.items():
        path=safe(dest/f'FULL_RGB_FRAME{fi:03d}.png',True)
        Image.fromarray(rgb,'RGB').save(path)
        frame_outputs.append(dict(frame_index=fi,frame_id=frames[fi]['frame_id'],
            decoded_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),file=bind(path),
            original_encoded_binding=frames[fi]['image_binding']))
    ids=np.array(selected,np.int64)
    frame_ids=np.array(sorted(rgb_by_frame),np.int64)
    np.savez_compressed(safe(dest/'SELECTED_MASKS_AND_DISPLAY_ZOOMS.npz',True),
        global_index=ids,frame_index=frame_ids,
        display_only_zoom_RGB96_HWC=np.stack([e.rgb_array(zooms[i]) for i in selected]),
        **{a:maps[a][frame_ids].astype('float32') for a in NEW_ARMS})
    inputs=dump(dest/'INPUTS.json',dict(completed=True,full_RGBs=frame_outputs,
        selected_masks_and_display_zooms=bind(dest/'SELECTED_MASKS_AND_DISPLAY_ZOOMS.npz'),
        frame_metadata=[frames[fi] for fi in frame_ids],selection=selection,
        local_zoom_model_input=False,real_mask_ground_truth=False))
    case_data=dump(out/'VISUAL_CASE_DATA.json',dict(completed=True,cases=cards,
        selection=selection,compact_visual_inputs=inputs,
        full_image_was_model_input=True,local_zoom_display_only=True,
        target_probability_fixed_color_scale=[0,1],real_pixel_mask_GT_available=False))
    return dict(selection=selection,case_data=case_data,compact_visual_inputs=inputs),cards


def report(out,result,cards):
    esc=lambda s:html.escape(str(s))
    pct=lambda x:'—' if x is None else f'{100*x:.1f}'
    parts=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>V267 전체 영상 CLIP 보이는 팔레트 분할</title>',
      '<style>body{font:17px/1.65 sans-serif;max-width:1500px;margin:32px auto;padding:0 20px;color:#172238}table{border-collapse:collapse}td,th{border:1px solid #cbd5e2;padding:8px}img{max-width:100%}.note{background:#fff0cf;padding:18px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.case{background:#f3f6fa;padding:10px}a{color:#155ec2}</style>',
      '<h1>전체 영상 CLIP · 보이는 팔레트 픽셀 분할</h1>',
      '<p class="note">학습 정답은 사용자 원본 합성 M4의 픽셀 마스크입니다. 실사223장은 이번 fit·설정 선택에 쓰지 않았지만 이전에도 확인한 개발 평가입니다. 모델 입력은 전체 영상이며, 아래 crop은 표시 전용입니다.</p>',
      '<p><b>새 점 단위 결과는 MASK_PRESENCE_PROXY_E_NOT_CAUSE입니다.</b> 원래 초기 query 위치의 P(target)≥0.5이면 D, 미만이면 E로 대응해 기존 D/E 주석과 비교했습니다. non-target는 배경·좌표 오차·다른 물체일 수도 있어 외부 가림 원인을 학습한 분류기라고 주장하지 않습니다. 실사 픽셀 정답이 없어 실사 IoU를 제시하지 않습니다.</p>',
      '<p>원래 native0–7·초기 좌표·입력 불가45점(U)을 보존했습니다. 자기 가림은 별도입니다. CAL은 과거 실사 학습 이력이 있는 기존 모델이며 새 모델과 훈련 조건이 다릅니다.</p>']
    for group in GROUPS:
        parts.append('<h2>'+esc(group)+' · 동일 PRIMARY 점 평가</h2><table><tr><th>방법</th><th>D→D/E/U</th><th>E→D/E/U</th><th>D P/R</th><th>E P/R/F1</th><th>U%</th></tr>')
        for r in result['classification']:
            if r['group']!=group or r['cohort']!='PRIMARY':
                continue
            cells=[r['model'],f"{r['D_D']}/{r['D_E']}/{r['D_U']}",f"{r['E_D']}/{r['E_E']}/{r['E_U']}",
                '/'.join(pct(r[k]) for k in ('D_precision','D_recall')),
                '/'.join(pct(r[k]) for k in ('E_precision','E_recall','E_F1')),pct(r['U_rate'])]
            parts.append('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in cells)+'</tr>')
        parts.append('</table>')
    parts.append('<h2>기존 외부 가림 누락67점의 proxy 변화</h2><table><tr><th>유형</th><th>방법</th><th>67중 E로 변화</th><th>D 유지</th><th>기존 E 유지</th><th>새 D 누락</th></tr>')
    for r in result['external90_transition_summary']:
        cells=[r['group'],r['model'],r['RECOVERED_PRIOR_MISS'],r['REMAINED_PRIOR_MISS'],r['KEPT_PRIOR_CORRECT'],r['REGRESSED_PRIOR_CORRECT']]
        parts.append('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in cells)+'</tr>')
    parts.append('</table><p><a href="CLASSIFICATION.csv">전체60행 혼동행렬/P/R/F1/U</a> · <a href="SELF_RESPONSES.csv">자기 가림 별도</a> · <a href="ALL1784_COMPARISON.csv">1784점 결과</a> · <a href="EXTERNAL90_TRANSITIONS.csv">외부 가림90점 변화</a> · <a href="SUMMARY.json">전체 JSON</a> · <a href="CASE_SELECTION_BEFORE_RGB.json">사례 선정·빈 셀·과거 노출</a></p>',
        '<h2>전체 RGB / 전체 P(target) / 마스크 / 표시 전용 확대</h2><p>외부 가림90점 전체와 고정 패턴별 최소 G 사례를 표시합니다. G370·908·1515 및 사용자 이견 G996을 유지했습니다. E 주석만으로 모든 사례를 콘 가림으로 부르지 않습니다. P(target)의 색은 모든 영상·방법에서 고정0–1이며 min/max를 따로 늘리지 않았습니다.</p><div class="grid">')
    for c in cards:
        name=Path(c['card']['path']).name
        parts.append('<div class="case"><p>'+esc('G'+str(c['global_index'])+' · '+c['row_id'])+'</p><a href="'+name+'"><img loading="lazy" src="'+name+'"></a></div>')
    parts.append('</div><p>이 평가·시각화에는 새 학습·모델 추론·PnP·threshold 선택이 없습니다. 위치 보정도 적용하지 않았습니다.</p></html>')
    path=safe(out/'INDEX_KO.html',True)
    path.write_text('\n'.join(parts),encoding='utf-8')
    return bind(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default='evaluation_v1')
    parser.add_argument('--replay-metrics',action='store_true')
    parser.add_argument('--inputs',default='evaluation_v1/compact_inputs')
    parser.add_argument('--metrics-only',action='store_true')
    args=parser.parse_args()
    need(Path(args.out).name==args.out and '..' not in args.out,'Single output directory name')
    out=safe(W/args.out,True)
    if args.replay_metrics:
        p=Path(args.inputs)
        inputs=safe(p if p.is_absolute() else W/p)
        out.mkdir()
        complete=replay(inputs,out)
    else:
        e=module(EVALUATOR_PATH,EVALUATOR_SHA,'v267_pinned_metric_api')
        new,frames,queries,maps,logits,seals=freeze_inputs()
        out.mkdir()
        consumer=dump(out/'CONSUMER_SEAL_BEFORE_LATE_GT_AND_RGB.json',dict(completed=True,PASS=True,
            seals=seals,predictions1784_and_fullmap223_input_identity_validated=True,
            late_annotation_or_GT_or_RGB_reads=0,created_UTC=dt.datetime.now(dt.timezone.utc).isoformat()))
        # Late current annotation join occurs strictly after the above output freeze checks.
        ledger_bytes=e.checked(e.PINNED['join'])
        ledger=list(csv.DictReader(io.StringIO(ledger_bytes.decode('utf-8-sig'))))
        prior_bytes=raw(PRIOR_PATH)
        need(hashlib.sha256(prior_bytes).hexdigest()==PRIOR_SHA,'Pinned saved V264 baseline')
        prior=jsonl(prior_bytes)
        result=compute(e,ledger,new,prior)
        outputs=metric_outputs(out,result,ledger,new,prior)
        compact_binding=compact(out,ledger_bytes,checked(seals['predictions']),prior_bytes)
        visuals,cards=({},[]) if args.metrics_only else visualize(e,out,ledger,new,prior,frames,maps,result)
        summary=dump(out/'SUMMARY.json',dict(completed=True,PASS=True,rows=1784,frames=223,**result,
            current223_is_repeated_development_evaluation=True,fresh_blind_evaluation=False,
            no_current_annotations_used_for_training=True,no_current_threshold_selection=True,
            no_real_pixel_mask_GT_or_real_IoU=True,new_model_checkpoint_encoder_reads=0,
            new_optimizer_updates=0,new_PnP_or_location_correction=0,
            query_sampling='Bilinear RAWlogit at (q+.5)/originalW,H; align_corners=False,border padding, then sigmoid.',
            threshold='P(target)>=0.5 D, otherwise E proxy; original unavailable45 remain U.',
            confusion_order=dict(rows=['D','E'],columns=['D','E','U']),
            consumer_pre_late_GT_seal=consumer,seals=seals,outputs=outputs,visuals=visuals,
            compact_metrics_inputs=compact_binding,pinned_metric_api=bind(EVALUATOR_PATH)))
        report_binding=report(out,result,cards)
        complete=dump(out/'COMPLETE.json',dict(completed=True,PASS=True,summary=summary,report=report_binding,
            outputs=outputs,visuals=visuals,compact_metrics_inputs=compact_binding,
            code=bind(Path(__file__).absolute()),consumer_pre_late_GT_seal=consumer))
    print(json.dumps(dict(completed=True,PASS=True,COMPLETE=complete),ensure_ascii=False))


if __name__=='__main__':
    main()
