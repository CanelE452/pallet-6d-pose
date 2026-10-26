#!/usr/bin/env python3
"""SYN-only full-frame boundary-refinement evaluation after all predictions seal.

The new mask-at-original-query outputs remain presence proxies, not learned
external-occlusion-cause classifiers. No fits, thresholds, relabels or PnP here.
"""
from pathlib import Path
import argparse
from collections import Counter
import csv
import hashlib
import html
import io
import json
import os
import stat
import sys
import types

import numpy as np
from PIL import Image, ImageDraw

W=Path(__file__).absolute().parent
R=W.parent
OLD_W=R/'V267_FULL_IMAGE_CLIP_VISIBLE_MASK'
BASE_CODE=OLD_W/'evaluate_fullframe_v267.py'
BASE_SHA='8d431e68a4616452b978c9ca8b48da6a0cb0d49076fe3b398acf927fa32b66b5'
REFERENCE_PATH=OLD_W/'CURRENT_ALL1784_PREDICTIONS.jsonl'
REFERENCE_SHA='4782cf6f8913602f77ba1e7866e44278f33177ab77c28ebc9febdc150f58a83c'
REFERENCE_FREEZE=OLD_W/'CURRENT_PREDICTIONS_FROZEN.json'
REFERENCE_FREEZE_SHA='2eada54f6138f711e4f9cf8fa7b2c6a50e00cad70e96e5a9fa25bed5aa62ff0d'
NEW_ARMS=('HIGHRES_REFINE','BOUNDARY_REFINE')
MODELS=(*NEW_ARMS,'V267_CS','V264_HEATMAP_LOCAL','v61','CAL_LEGACY')
FIXED_CASES=(370,908,1515,200,224,852,314,332,665,996)
GROUPS=('ALL','WOOD_PRIMARY105','SQUARE118')
PROXY='MASK_PRESENCE_PROXY_E_NOT_CAUSE'
B=None


def safe(value,fresh=False):
    p=Path(value)
    if not p.is_absolute() or '..' in p.parts or any(t in str(p).casefold() for t in ('v112','cache','.pyc')):
        raise ValueError(str(p))
    for a in [*reversed(p.parents),p]:
        try:mode=os.lstat(a).st_mode
        except FileNotFoundError:continue
        if stat.S_ISLNK(mode):raise ValueError('Symlink: '+str(a))
    if fresh and p.exists():raise FileExistsError(str(p))
    return p


def load_base(path):
    global B
    p=safe(path);data=p.read_bytes()
    if hashlib.sha256(data).hexdigest()!=BASE_SHA:raise ValueError('Pinned V267 evaluator CODE mismatch')
    B=types.ModuleType('v268_pinned_v267_evaluation_api')
    B.__file__=str(p);sys.modules[B.__name__]=B
    exec(compile(data,str(p),'exec'),B.__dict__)
    B.NEW_ARMS=NEW_ARMS
    B.MODELS=MODELS
    return B


def freeze_inputs():
    seal_path=W/'CURRENT_PREDICTIONS_FROZEN.json'
    seal=json.loads(B.raw(seal_path))
    B.need(seal.get('completed') is True and seal.get('PASS',True) is True,'New prediction freeze complete')
    config=json.loads(B.checked(seal['configuration']))
    B.need(config.get('completed') is True or config.get('PASS') is True,'New configuration freeze complete')
    B.need(seal.get('sample_padding_mode','border')=='border','Fixed border sampling')
    old_input_bytes=B.raw(OLD_W/'current_fullframe_inputs_v2/COMPLETE.json')
    B.need(hashlib.sha256(old_input_bytes).hexdigest()==B.INPUT_COMPLETE_SHA,'Exact old fullframe input identities')
    old_input=json.loads(old_input_bytes)
    B.need(seal['current_frame_inputs']==old_input['frames'],'Exact original223 frames')
    B.need(seal['current_query_inputs']==old_input['queries_manifest'],'Exact original1784 queries')
    frames=B.jsonl(B.checked(seal['current_frame_inputs']))
    queries=B.jsonl(B.checked(seal['current_query_inputs']))
    B.need(len(frames)==223 and len(queries)==1784,'223 full frames and1784 queries')
    B.need(Path(seal['predictions']['path'])==W/'CURRENT_ALL1784_PREDICTIONS.jsonl','Exact prediction payload')
    B.need(Path(seal['frame_masks']['path'])==W/'CURRENT_FRAME_MASKS.json','Exact map index payload')
    new=B.jsonl(B.checked(seal['predictions']))
    B.validate_rows(new,queries)
    index=json.loads(B.checked(seal['frame_masks']))
    B.need(index.get('completed') is True and index.get('PASS',True) is True and index['resolution']==256,'Frozen raw-map index')
    maps,logits={},{}
    for arm in NEW_ARMS:
        with np.load(io.BytesIO(B.checked(index['arms'][arm])),allow_pickle=False) as z:
            B.need(np.array_equal(z['frame_index'],np.arange(223)),'Map frame indices')
            logits[arm]=np.array(z['logits'],dtype=np.float32,copy=True)
        maps[arm]=1/(1+np.exp(-np.clip(logits[arm],-80,80)))
    maxdiff=0.
    for arm in NEW_ARMS:
        B.need(maps[arm].shape==logits[arm].shape==(223,256,256),'Fixed full-frame map256x256')
        B.need(np.isfinite(maps[arm]).all() and np.isfinite(logits[arm]).all(),'Finite new full maps')
        B.need((maps[arm]>=0).all() and (maps[arm]<=1).all(),'Target probabilities0..1')
        for first in range(0,223,16):
            rawlogit=logits[arm][first:first+16]
            recomputed=1/(1+np.exp(-np.clip(rawlogit,-80,80)))
            B.need(np.max(np.abs(recomputed-maps[arm][first:first+16]))<=2e-6,'Probability=sigmoid rawlogit')
        for i,p in enumerate(new):
            if not p['available']:continue
            q=queries[i]
            sample=B.sample_border(logits[arm][i//8],q['original_q0'],q['raw_hw'])
            diff=abs(sample+float(p[arm]['E_logit']))
            maxdiff=max(maxdiff,diff)
            B.need(diff<=.003,'Unchanged originalquery rawmap sampling: '+str(i))
    return new,frames,queries,maps,dict(
        prediction_freeze=B.bind(seal_path),configuration=seal['configuration'],
        predictions=seal['predictions'],frame_masks=seal['frame_masks'],
        current_frame_inputs=seal['current_frame_inputs'],current_query_inputs=seal['current_query_inputs'],
        forward_replay_inputs=seal.get('forward_replay_inputs'),
        max_abs_numpy_vs_saved_logit_difference=maxdiff)


def compute(e,ledger,new,prior,reference):
    B.validate_rows(new)
    B.need(len(ledger)==len(prior)==len(reference)==1784,'Same1784 original rows')
    for i,(r,p,old,ref) in enumerate(zip(ledger,new,prior,reference)):
        B.need(int(r['global_index'])==p['global_index']==old['global_index']==ref['global_index']==i,'Global order')
        B.need(r['row_id']==p['row_id']==old['row_id']==ref['row_id'],'Native/frame identity')
        B.need((r['original_query_valid']=='True')==p['available']==old['available']==ref['available'],'Unchanged original availability')
    pp={a:[p[a]['pred'] for p in new] for a in NEW_ARMS}
    pp.update(V267_CS=[p['FULL_CLIP_SURGERY_ARCH']['pred'] for p in reference],
        V264_HEATMAP_LOCAL=[p['HEATMAP_LOCAL']['pred'] for p in prior],
        v61=[r['v61_D_E_U'] for r in ledger],CAL_LEGACY=[r['legacy_D_E_U'] for r in ledger])
    cohorts={'PRIMARY':[r for r in ledger if e.primary(r)],
        'ALL_REGISTERED_D_E':[r for r in ledger if r['human_state'] in (e.D,e.E)],
        'MANUAL_CLICK_D_ALL':[r for r in ledger if e.manual_d(r)],
        'MANUAL_CLICK_D_LE3_VALID_INPUT':[r for r in ledger if e.near_d(r)]}
    B.need(len(cohorts['PRIMARY'])==1237 and sum(r['human_state']==e.D for r in cohorts['PRIMARY'])==1147,'PRIMARY1237 D1147/E90')
    classification,self_rows,transitions=[],[],[]
    for group in GROUPS:
        for cohort,rows in cohorts.items():
            take=[r for r in rows if group=='ALL' or r['group']==group]
            for model in MODELS:
                classification.append(dict(group=group,cohort=cohort,model=model,
                    prediction_semantics=PROXY if model in (*NEW_ARMS,'V267_CS') else 'PRIOR_VISIBILITY_CLASSIFIER',
                    **e.cm(take,pp[model])))
            for baseline in ('V267_CS','V264_HEATMAP_LOCAL'):
                for arm in NEW_ARMS:
                    transitions.append(dict(group=group,cohort=cohort,baseline=baseline,model=arm,
                        **e.transition(take,pp[baseline],pp[arm])))
        rows=[r for r in ledger if r['human_state']==e.S and (group=='ALL' or r['group']==group)]
        for model in MODELS:
            c=Counter(pp[model][int(r['global_index'])] for r in rows)
            self_rows.append(dict(group=group,model=model,N=len(rows),response_D=c['D'],
                response_E=c['E'],response_U=c['U'],interpretation='Human SELF separate; binary non-target mask response is not learned SELF correctness.'))
    external=[]
    for row in cohorts['PRIMARY']:
        if row['human_state']!=e.E:continue
        i=int(row['global_index'])
        r=dict(global_index=i,row_id=row['row_id'],group=row['group'],
            coordinate_source=row['annotation_coordinate_source'],user_disputed_annotation=i==996,
            V267_CS=pp['V267_CS'][i],V264_HEATMAP_LOCAL=pp['V264_HEATMAP_LOCAL'][i])
        for baseline in ('V267_CS','V264_HEATMAP_LOCAL'):
            missed=pp[baseline][i]=='D'
            r[baseline+'_prior_missed']=missed
            for arm in NEW_ARMS:
                p=pp[arm][i];r[arm]=p
                r[baseline+'__'+arm+'_transition']=('RECOVERED_PRIOR_MISS' if p=='E' else 'REMAINED_PRIOR_MISS') if missed else ('KEPT_PRIOR_CORRECT' if p=='E' else 'REGRESSED_PRIOR_CORRECT')
        external.append(r)
    B.need(len(external)==90,'Same90 primary external labels')
    ext_summary=[]
    for group in GROUPS:
        rows=[r for r in external if group=='ALL' or r['group']==group]
        for baseline in ('V267_CS','V264_HEATMAP_LOCAL'):
            for arm in NEW_ARMS:
                c=Counter(r[baseline+'__'+arm+'_transition'] for r in rows)
                ext_summary.append(dict(group=group,baseline=baseline,model=arm,N=len(rows),
                    prior_misses=sum(r[baseline+'_prior_missed'] for r in rows),
                    **{k:c[k] for k in ('RECOVERED_PRIOR_MISS','REMAINED_PRIOR_MISS','KEPT_PRIOR_CORRECT','REGRESSED_PRIOR_CORRECT')}))
    selection=[]
    for group in GROUPS[1:]:
        for truth in ('D','E'):
            for bits in ('000','001','010','011','100','101','110','111'):
                ids=[]
                for row in cohorts['PRIMARY']:
                    i=int(row['global_index'])
                    if row['group']==group and e.truth(row)==truth and ''.join('1' if pp[a][i]==truth else '0' for a in ('V267_CS',*NEW_ARMS))==bits:ids.append(i)
                selection.append(dict(group=group,truth=truth,correctness_pattern=bits,
                    census_N=len(ids),selected_global_index=min(ids) if ids else None,empty=not ids))
    return dict(primary_counts=dict(N=1237,D=1147,E=90),classification=classification,
        human_SELF=self_rows,paired_transitions=transitions,external90_rows=external,
        external90_transition_summary=ext_summary,fixed_pattern_selection=selection,
        new_prediction_semantics=PROXY,real_pixel_mask_GT_available=False,
        V267_CS_alias='V267 FULL_CLIP_SURGERY_ARCH; not an official complete CLIP Surgery pipeline.')


def metric_outputs(out,result,ledger,new,prior,reference):
    rows=[]
    for i,(r,p) in enumerate(zip(ledger,new)):
        x={k:r[k] for k in ('global_index','row_id','id','native_corner','group','human_state',
            'annotation_coordinate_source','initial_reference_error_px','original_query_valid')}
        x['PRIMARY']=r['paired_same_query_RGB_D_vs_E']=='True' and r['human_state'] in ('DIRECT_VISIBLE','EXTERNAL_OCCLUDED')
        for a in NEW_ARMS:
            for key in ('pred','p_target','E_logit','p_E'):x[a+'_'+key]=p[a].get(key)
        for key in ('pred','p_target','E_logit','p_E'):x['V267_CS_'+key]=reference[i]['FULL_CLIP_SURGERY_ARCH'].get(key)
        x['V264_HEATMAP_LOCAL_pred']=prior[i]['HEATMAP_LOCAL']['pred']
        x['v61_pred']=r['v61_D_E_U'];x['CAL_LEGACY_pred']=r['legacy_D_E_U']
        rows.append(x)
    return dict(classification=B.table(out/'CLASSIFICATION.csv',result['classification']),
        SELF=B.table(out/'SELF_RESPONSES.csv',result['human_SELF']),
        external90=B.table(out/'EXTERNAL90_TRANSITIONS.csv',result['external90_rows']),
        external_summary=B.table(out/'EXTERNAL90_TRANSITION_SUMMARY.csv',result['external90_transition_summary']),
        paired_transitions=B.dump(out/'PAIRED_TRANSITIONS.json',result['paired_transitions']),
        all1784=B.table(out/'ALL1784_COMPARISON.csv',rows))


def compact(out,ledger_b,new_b,prior_b,reference_b):
    dest=safe(out/'compact_inputs',True);dest.mkdir()
    payloads={'LEDGER.csv':ledger_b,'NEW_PREDICTIONS.jsonl':new_b,
        'V264_PRIOR_PREDICTIONS.jsonl':prior_b,'V267_REFERENCE_PREDICTIONS.jsonl':reference_b,
        'V267_EVALUATOR_CODE.py':B.raw(BASE_CODE),'V264_EVALUATOR_CODE.py':B.raw(B.EVALUATOR_PATH)}
    files={}
    for name,data in payloads.items():
        safe(dest/name,True).write_bytes(data)
        files[name]=dict(sha256=hashlib.sha256(data).hexdigest(),size_bytes=len(data))
    return B.dump(dest/'INTEGRITY.json',dict(completed=True,files=files,
        new_arms=list(NEW_ARMS),models=list(MODELS),no_RGB_or_models_required=True))


def replay(inputs,out):
    integrity=json.loads(B.raw(inputs/'INTEGRITY.json'))
    data={}
    for name,b in integrity['files'].items():
        value=B.raw(inputs/name)
        B.need(hashlib.sha256(value).hexdigest()==b['sha256'] and len(value)==b['size_bytes'],'Compact integrity')
        data[name]=value
    e=B.module(inputs/'V264_EVALUATOR_CODE.py',B.EVALUATOR_SHA,'v268_replay_metric_api')
    ledger=list(csv.DictReader(io.StringIO(data['LEDGER.csv'].decode('utf-8-sig'))))
    new=B.jsonl(data['NEW_PREDICTIONS.jsonl']);prior=B.jsonl(data['V264_PRIOR_PREDICTIONS.jsonl'])
    ref=B.jsonl(data['V267_REFERENCE_PREDICTIONS.jsonl'])
    result=compute(e,ledger,new,prior,ref);outputs=metric_outputs(out,result,ledger,new,prior,ref)
    summary=B.dump(out/'SUMMARY.json',dict(completed=True,PASS=True,mode='COMPACT_METRICS_REPLAY',
        **result,outputs=outputs,new_RGB_or_model_reads=0,new_optimizer_or_PnP=0))
    return B.dump(out/'COMPLETE.json',dict(completed=True,PASS=True,summary=summary,
        integrity=B.bind(inputs/'INTEGRITY.json'),code=B.bind(Path(__file__).absolute())))


def make_card(i,ledger,new,prior,reference,frame,rgb,maps,old_map,zoom,out):
    row=ledger[i]
    im=Image.new('RGB',(1440,1480),'white');d=ImageDraw.Draw(im)
    B.text(d,f"G{i} · {row['row_id']} · {row['human_state']}",(20,12),24)
    B.text(d,'원래 초기 query·native 그대로 / 전체 영상 입력 / E는 non-target proxy이며 외부 가림 원인 출력 아님',(20,46),18)
    old=reference[i]['FULL_CLIP_SURGERY_ARCH']
    B.text(d,f"V267_CS {old['pred']} P(T)={float(old['p_target']):.3f} · V264 {prior[i]['HEATMAP_LOCAL']['pred']} · CAL {row['legacy_D_E_U']}",(20,77),18)
    for j,a in enumerate(NEW_ARMS):
        p=new[i][a];value='—' if p.get('p_target') is None else f"{float(p['p_target']):.3f}"
        B.text(d,f"{a}: {p['pred']} · P(target@초기q)={value}",(20,105+j*25),18)
    # Fixed layout: full RGB + old probability/overlay; each new probability/overlay.
    new_panels=B.panels_for_frame(rgb,frame,i,maps)
    full=Image.fromarray(rgb,'RGB');p=old_map[frame['frame_index']]
    pb=np.asarray(Image.fromarray(p.astype('float32'),'F').resize(full.size,Image.Resampling.BILINEAR))
    color=np.stack((20+20*pb,30+220*pb,25+65*pb),-1).clip(0,255).astype('uint8')
    old_prob=B.annotated_image(Image.fromarray(color),frame,i)
    tint=np.array(full,copy=True);mask=pb>=.5
    tint[mask]=(tint[mask]*.5+np.array([25,230,80])*.5).astype('uint8')
    old_overlay=B.annotated_image(Image.fromarray(tint),frame,i)
    zoom_im=Image.fromarray(zoom).resize((336,336),Image.Resampling.NEAREST)
    zd=ImageDraw.Draw(zoom_im);zd.ellipse((162,162,174,174),outline='#fff43b',width=3)
    panels=[new_panels[0],('이전 V267_CS · 전체 P(T) 0–1',old_prob),('이전 V267_CS · 고정 0.5 mask',old_overlay),
            new_panels[1],new_panels[3],('crop 확대 · 표시 전용 · 모델 입력 아님',zoom_im),
            new_panels[2],new_panels[4]]
    for j,(title,panel) in enumerate(panels):
        x=20+(j%3)*480;y=206+(j//3)*406
        B.text(d,title,(x,y-31),17);im.paste(B.fit_panel(panel),(x,y))
    B.text(d,'노랑○ 선택한 초기 점 · 파랑○ 나머지 초기 점 · 전 방식 고정 P(T)0–1 · 자기 가림은 별도 평가',(20,1380),17)
    caution='G996: 사용자와 기존 E 주석 해석 이견 유지.' if i==996 else '실사 픽셀 정답이 없으므로 실사 IoU를 주장하지 않습니다. 마스크의 모서리 샘플은 native 가시성과 다릅니다.'
    B.text(d,caution,(20,1410),17,'#8b421e')
    B.text(d,'GT로 재중심 crop·좌표 보정·threshold 재선택 없음. 오른쪽 가운데 crop만 표시를 위해 확대했습니다.',(20,1440),16)
    path=safe(out/f'CASE_G{i:04d}.png',True);im.save(path)
    return dict(global_index=i,row_id=row['row_id'],frame_index=i//8,human_state=row['human_state'],
        original_q0=frame['original_query_xy8'][i%8],user_disputed_annotation=i==996,
        V267_CS=old,new_outputs={a:new[i][a] for a in NEW_ARMS},
        full_RGB_decoded_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),card=B.bind(path))


def visualize(e,out,ledger,new,prior,reference,frames,maps,result):
    ids=[r['global_index'] for r in result['external90_rows']]
    ids += [r['selected_global_index'] for r in result['fixed_pattern_selection'] if not r['empty']]
    ids += list(FIXED_CASES);ids=list(dict.fromkeys(ids))
    selection=B.dump(out/'CASE_SELECTION_BEFORE_RGB.json',dict(completed=True,
        external90_all=[r['global_index'] for r in result['external90_rows']],
        fixed_patterns=result['fixed_pattern_selection'],predefined=list(FIXED_CASES),
        selected_indices=ids,prior_RGB_exposure=True,
        policy='All90E plus minimum G per material/truth/V267+new2 correctness pattern; fixed historical examples preserved. Not blinded.'))
    io_api=B.module(B.IO_HELPER_PATH,B.IO_HELPER_SHA,'v268_exact_RGB_io')
    metadata_bytes=B.raw(REFERENCE_FREEZE)
    B.need(hashlib.sha256(metadata_bytes).hexdigest()==REFERENCE_FREEZE_SHA,'Old CS map provenance')
    old_seal=json.loads(metadata_bytes)
    with np.load(io.BytesIO(B.checked(old_seal['frame_masks'])),allow_pickle=False) as z:
        old_map=np.array(z['FULL_CLIP_SURGERY_ARCH'],copy=True)
    source=json.loads(e.checked(e.PINNED['v263_current_complete']))
    with np.load(io.BytesIO(B.checked(source['crop_cache']['binding'])),allow_pickle=False) as z:
        zooms=z['RGB96_initial_geometry']
    rgbs={fi:io_api.read_rgb(frames[fi]) for fi in sorted({i//8 for i in ids})}
    cards=[make_card(i,ledger,new,prior,reference,frames[i//8],rgbs[i//8],maps,old_map,e.rgb_array(zooms[i]),out) for i in ids]
    dest=safe(out/'compact_visual_inputs',True);dest.mkdir()
    frame_payloads=[]
    for fi,rgb in rgbs.items():
        p=safe(dest/f'FULL_RGB_FRAME{fi:03d}.png',True);Image.fromarray(rgb).save(p)
        frame_payloads.append(dict(frame_index=fi,file=B.bind(p),
            decoded_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),original_encoded_binding=frames[fi]['image_binding']))
    fis=np.array(sorted(rgbs),np.int64)
    np.savez_compressed(safe(dest/'SELECTED_MASKS_AND_DISPLAY_ZOOMS.npz',True),
        frame_index=fis,global_index=np.array(ids,np.int64),V267_CS=old_map[fis],
        display_only_zoom_RGB96_HWC=np.stack([e.rgb_array(zooms[i]) for i in ids]),
        **{a:maps[a][fis] for a in NEW_ARMS})
    compact_b=B.dump(dest/'INPUTS.json',dict(completed=True,full_RGBs=frame_payloads,
        masks=B.bind(dest/'SELECTED_MASKS_AND_DISPLAY_ZOOMS.npz'),frame_metadata=[frames[fi] for fi in fis],
        local_zoom_model_input=False,real_mask_ground_truth=False))
    cases=B.dump(out/'VISUAL_CASE_DATA.json',dict(completed=True,cases=cards,selection=selection,
        compact_visual_inputs=compact_b,fixed_color_range=[0,1],local_zoom_display_only=True))
    return dict(selection=selection,case_data=cases,compact_visual_inputs=compact_b),cards


def report(out,result,cards):
    esc=lambda x:html.escape(str(x))
    pct=lambda x:'—' if x is None else f'{100*x:.1f}'
    parts=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>V268 전체 영상 경계 보정</title>',
        '<style>body{font:17px/1.65 sans-serif;max-width:1500px;margin:30px auto;padding:0 20px;color:#172238}table{border-collapse:collapse}td,th{border:1px solid #cbd4df;padding:7px}img{max-width:100%}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.note{background:#fff0cb;padding:16px}.case{background:#f3f6fa;padding:10px}</style>',
        '<h1>전체 영상 경계 보정 · 두 합성 학습 방식과 고정 이전 모델 비교</h1>',
        '<p class="note">새 학습·설정 선택은 합성만 사용했습니다. 실사223장은 기존에도 확인한 개발 평가입니다. 원래 초기 좌표·native·입력불가45점(U)을 유지했습니다. 모델 입력은 전체 영상이고 crop은 표시 전용입니다.</p>',
        '<p>HIGHRES_REFINE과 BOUNDARY_REFINE을 모두 평가했으며 실사 결과로 하나를 선택하거나 threshold를 바꾸지 않았습니다. 새 E는 MASK_PRESENCE_PROXY_E_NOT_CAUSE입니다. P(target@원래q)≥0.5이면 D, 미만이면 E proxy로 기존 주석과 비교합니다. 배경·좌표 오차·다른 물체도 non-target이므로 외부 가림 원인을 학습한 분류기로 해석하지 않습니다.</p>',
        '<p>V267_CS는 FULL_CLIP_SURGERY_ARCH의 별칭이며 공식 CLIP Surgery 전체 방법을 재현했다는 뜻이 아닙니다. CAL은 과거 실사 학습 이력이 있습니다. 자기 가림은 별도로 집계합니다. 실사 픽셀 정답이 없어 실사 IoU를 주장하지 않습니다.</p>']
    for group in GROUPS:
        parts.append('<h2>'+esc(group)+' · PRIMARY</h2><table><tr><th>모델</th><th>D→D/E/U</th><th>E→D/E/U</th><th>D P/R</th><th>E P/R/F1</th><th>U%</th></tr>')
        for r in result['classification']:
            if r['group']!=group or r['cohort']!='PRIMARY':continue
            cells=[r['model'],f"{r['D_D']}/{r['D_E']}/{r['D_U']}",f"{r['E_D']}/{r['E_E']}/{r['E_U']}",
                '/'.join(pct(r[k]) for k in ('D_precision','D_recall')),
                '/'.join(pct(r[k]) for k in ('E_precision','E_recall','E_F1')),pct(r['U_rate'])]
            parts.append('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in cells)+'</tr>')
        parts.append('</table>')
    parts.append('<h2>외부 가림90점 · 이전 V267/V264에 대한 변화</h2><table><tr><th>재질</th><th>기준 모델</th><th>새 모델</th><th>기존 누락→E</th><th>누락 유지</th><th>정답 유지</th><th>새 누락</th></tr>')
    for r in result['external90_transition_summary']:
        vals=[r['group'],r['baseline'],r['model'],r['RECOVERED_PRIOR_MISS'],r['REMAINED_PRIOR_MISS'],r['KEPT_PRIOR_CORRECT'],r['REGRESSED_PRIOR_CORRECT']]
        parts.append('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in vals)+'</tr>')
    parts.extend(['</table><p><a href="CLASSIFICATION.csv">72행 전체 평가</a> · <a href="SELF_RESPONSES.csv">자기 가림</a> · <a href="ALL1784_COMPARISON.csv">1784점</a> · <a href="EXTERNAL90_TRANSITIONS.csv">외부 가림90점 변화</a> · <a href="SUMMARY.json">전체 JSON</a> · <a href="CASE_SELECTION_BEFORE_RGB.json">선정·빈 셀·과거 노출</a></p>',
        '<h2>전체 RGB · 이전/새 마스크와 P(target) · 표시 전용 확대</h2><p>외부 가림90점 전체와 고정 패턴 최소 G, 지정한 과거 사례를 보존했습니다. G996의 사용자 주석 이견도 유지합니다. E 주석만으로 모든 사례를 콘 가림으로 부르지 않습니다. 모든 확률 지도는 고정0–1 색입니다.</p><div class="grid">'])
    for c in cards:
        name=Path(c['card']['path']).name
        parts.append('<div class="case"><p>'+esc('G'+str(c['global_index'])+' · '+c['row_id'])+'</p><a href="'+name+'"><img loading="lazy" src="'+name+'"></a></div>')
    parts.append('</div><p>이 평가 스크립트는 새 모델 추론·학습·PnP·좌표 보정·운영 규칙 선택을 하지 않습니다.</p></html>')
    p=safe(out/'INDEX_KO.html',True);p.write_text('\n'.join(parts),encoding='utf-8')
    return B.bind(p)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default='evaluation_v1')
    parser.add_argument('--replay-metrics',action='store_true')
    parser.add_argument('--inputs',default='evaluation_v1/compact_inputs')
    parser.add_argument('--metrics-only',action='store_true')
    args=parser.parse_args()
    if Path(args.out).name!=args.out or '..' in args.out:raise ValueError('Single output directory')
    out=safe(W/args.out,True)
    if args.replay_metrics:
        p=Path(args.inputs);inputs=safe(p if p.is_absolute() else W/p)
        load_base(inputs/'V267_EVALUATOR_CODE.py');out.mkdir();complete=replay(inputs,out)
    else:
        load_base(BASE_CODE)
        e=B.module(B.EVALUATOR_PATH,B.EVALUATOR_SHA,'v268_pinned_metric_api')
        new,frames,queries,maps,seals=freeze_inputs();out.mkdir()
        consumer=B.dump(out/'CONSUMER_SEAL_BEFORE_LATE_GT_AND_RGB.json',dict(completed=True,PASS=True,
            seals=seals,late_annotation_GT_or_RGB_reads=0,all1784_input_order_and_fullmap_sampling_validated=True))
        # No current annotations or image pixels were read before this barrier.
        ledger_bytes=e.checked(e.PINNED['join'])
        ledger=list(csv.DictReader(io.StringIO(ledger_bytes.decode('utf-8-sig'))))
        prior_bytes=B.raw(B.PRIOR_PATH);B.need(hashlib.sha256(prior_bytes).hexdigest()==B.PRIOR_SHA,'Pinned V264')
        ref_bytes=B.raw(REFERENCE_PATH);B.need(hashlib.sha256(ref_bytes).hexdigest()==REFERENCE_SHA,'Pinned V267 CS reference')
        prior=B.jsonl(prior_bytes);reference=B.jsonl(ref_bytes)
        result=compute(e,ledger,new,prior,reference)
        outputs=metric_outputs(out,result,ledger,new,prior,reference)
        compact_b=compact(out,ledger_bytes,B.checked(seals['predictions']),prior_bytes,ref_bytes)
        visuals,cards=({},[]) if args.metrics_only else visualize(e,out,ledger,new,prior,reference,frames,maps,result)
        summary=B.dump(out/'SUMMARY.json',dict(completed=True,PASS=True,rows=1784,frames=223,**result,
            new_current_annotations_used_for_fit_or_setting_selection=False,
            current223_repeated_development_evaluation=True,fresh_blind_evaluation=False,
            new_model_or_checkpoint_reads=0,new_fit_or_PnP_or_location_changes=0,
            threshold='Fixed rawlogit>=0 D / <0 E proxy; original45 input U retained.',
            query_sampling='Original(q+.5)/HW, bilinear RAWlogit, align_corners=False,border, then sigmoid.',
            seals=seals,consumer_pre_late_GT_seal=consumer,outputs=outputs,visuals=visuals,
            compact_metrics_inputs=compact_b,pinned_V267_evaluator=B.bind(BASE_CODE)))
        report_b=report(out,result,cards)
        complete=B.dump(out/'COMPLETE.json',dict(completed=True,PASS=True,summary=summary,report=report_b,
            outputs=outputs,visuals=visuals,compact_metrics_inputs=compact_b,
            code=B.bind(Path(__file__).absolute()),consumer_pre_late_GT_seal=consumer))
    print(json.dumps(dict(completed=True,PASS=True,COMPLETE=complete),ensure_ascii=False))


if __name__=='__main__':main()
