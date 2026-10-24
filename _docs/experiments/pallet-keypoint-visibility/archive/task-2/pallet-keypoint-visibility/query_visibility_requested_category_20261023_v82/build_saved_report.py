"""V82 sealed saved-mask report/gallery. Code preparation does not read outputs. No model or feature imports.

The four source IDs and q/L come from V79. Source RGB is decoded only for the
category-provided paired detector overlay, byte-checked against both prior crop metadata
and the detector's saved RGB digest. No crop/query/detection is selected by GT.
"""
from pathlib import Path
import argparse, hashlib, html, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

R = Path(__file__).resolve().parent
P = R.parent
V79 = P / 'query_visibility_natural_occlusion_audit_20261023_v79'
V80 = P / 'query_visibility_dense_support_20261023_v80'
V81 = P / 'query_visibility_grounded_target_20261023_v81'
G79 = V79 / 'natural_failure_gallery'
OUT = R / 'fixed_natural_gallery'
EXPECTED = [('YCB:000009:000241:000000', 6), ('YCB:000008:001872:000004', 6),
            ('YCB:000008:001299:000002', 1), ('YCB:000009:000225:000002', 6)]

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for x in iter(lambda: f.read(1 << 20), b''): h.update(x)
    return h.hexdigest()
def bind(p): return {'path': str(Path(p).resolve()), 'sha256': sha(p)}
def check(b): assert sha(b['path']) == b['sha256'], b['path']
def read(p): return json.loads(Path(p).read_text())
def lines(p): return [json.loads(x) for x in Path(p).read_text().splitlines()]
def arrays(p):
    with np.load(p, allow_pickle=False) as z: return {k: z[k] for k in z.files}
def write(p, obj):
    with open(p, 'x') as f: json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
def pct(x): return '—' if x is None else f'{100*x:.1f}%'
def decimal(x): return '—' if x is None else f'{x:.3f}'
def esc(x): return html.escape(str(x))
def link(p, text): return f'<a href="{esc(p)}">{esc(text)}</a>'

def sample(mask, q, side, n, hw):
    """Literal old native9/RGB96 coordinates; NN validity kept separately."""
    h, w = hw
    offsets = ((np.arange(n) + .5) / n - .5) * side
    xx, yy = np.meshgrid(q[0] + offsets, q[1] + offsets)
    px, py = np.floor(xx + .5).astype(np.int64), np.floor(yy + .5).astype(np.int64)
    valid = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h) & (px >= 0) & (px < w) & (py >= 0) & (py < h)
    if mask is None: return None, valid
    assert mask.shape == (h, w) and mask.dtype == np.bool_
    out = np.zeros((n, n), bool); out[valid] = mask[py[valid], px[valid]]
    return out, valid

def grid(mask, valid, green=True):
    if mask is None:
        im = Image.new('RGB', (192, 192), '#28283c'); d = ImageDraw.Draw(im)
        for j in range(-180, 384, 20): d.line((j, 0, j+192, 192), fill='#54516e', width=5)
        d.rectangle((38, 75, 154, 115), fill='#222235')
        d.text((47, 85), 'U / NO MASK', fill='white', font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14))
        return im
    a = np.zeros((*mask.shape, 3), np.uint8); a[:] = [63, 69, 79]
    a[mask] = [76, 205, 138] if green else [232, 234, 239]
    a[~valid] = [8, 10, 14]
    im = Image.fromarray(a).resize((192, 192), Image.Resampling.NEAREST)
    if mask.shape == (9, 9):
        d = ImageDraw.Draw(im)
        for j in range(1, 9):
            v = round(j*192/9); d.line((v, 0, v, 192), fill='#1b2730'); d.line((0, v, 192, v), fill='#1b2730')
        d.rectangle((85, 85, 106, 106), outline='white', width=2)
    return im

def marker(d, xy, color, radius=5):
    x, y = map(float, xy); d.line((x-radius, y, x+radius, y), fill=color, width=2); d.line((x, y-radius, x, y+radius), fill=color, width=2)

def table(headers, rows):
    return '<table><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'

def render_case(n, case, association, old_association, detection, old_detection,
                maps, source_rgb, scores, prediction, old_prediction, iou, old_iou):
    bold=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',22)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    im=Image.new('RGB',(1840,710),'#111c28');d=ImageDraw.Draw(im)
    names=('NOT_VISIBLE_MASK','VISIBLE_MASK','U')
    d.text((20,12),f'{n+1}. {case["id"]} / native{case["native_k"]} / reference support {"V" if case["GT_label"] else "C"}',fill='white',font=bold)
    d.text((20,47),f'V81 q={names[old_prediction]}, M4 IoU={old_iou:.3f}  ->  V82 q={names[prediction]}, M4 IoU={iou:.3f} | q-r={case["error_px"]:.3f}px / L={case["L"]:.3f}px',fill='#d4dfea',font=font)
    phrase = str(detection['text_labels'][association['selected_detection_index']]) if association['status']=='ASSOCIATED' else None
    old_phrase = str(old_detection['text_labels'][old_association['selected_detection_index']]) if old_association['status']=='ASSOCIATED' else None
    d.text((20,76),f'V82 PROVIDED category="{association["requested_category"]}" / predicted phrase={phrase!r} | V81 phrase={old_phrase!r}',fill='#d4dfea',font=font)
    detail = f'V82 input-only box IoU={association["association_iou"]:.3f}, score={association["selected_detection_score"]:.3f}' if association['status']=='ASSOCIATED' else 'V82 U / '+association['reason']+' (no mask)'
    d.text((20,103),detail,fill='#d4dfea',font=font)
    h,w=source_rgb.shape[:2];scene=Image.fromarray(source_rgb).resize((440,330),Image.Resampling.BILINEAR)
    sd=ImageDraw.Draw(scene);scale=np.array([440/w,330/h])
    sd.rectangle(tuple((np.asarray(association['query_box_xyxy']).reshape(2,2)*scale).ravel()),outline='#53a6ff',width=3)
    for a,color in ((old_association,'#ff9b58'),(association,'#75ed98')):
        if a['status']=='ASSOCIATED':sd.rectangle(tuple((np.asarray(a['selected_box_xyxy_prompt']).reshape(2,2)*scale).ravel()),outline=color,width=3)
    marker(sd,np.asarray(case['q'])*scale,'#ffdf52');marker(sd,np.asarray(case['r'])*scale,'#3be6fa')
    d.text((20,139),'Blue input box / orange V81 / green V82',fill='white',font=small);im.paste(scene,(20,164))
    for key in ('raw_local_RGB96','raw_target_RGB128'):check(case[key])
    with Image.open(case['raw_local_RGB96']['path']) as a:local=a.convert('RGB').resize((192,192),Image.Resampling.NEAREST)
    with Image.open(case['raw_target_RGB128']['path']) as a:target=a.convert('RGB').resize((192,192),Image.Resampling.NEAREST)
    ld=ImageDraw.Draw(local);marker(ld,[95.5,95.5],'#ffdf52',4)
    marker(ld,((np.asarray(case['r'])-np.asarray(case['q']))/case['L']+.5)*192-.5,'#3be6fa',4)
    panels=[(480,'Same RGB96 / q,r',local),(694,'Same target RGB128',target),
        (908,'GT M0 / nominal9',grid(maps['M0_9'],maps['valid9'],False)),
        (1122,'GT M4 / nominal9',grid(maps['M4_9'],maps['valid9'])),
        (1336,'V81 SAM / nominal9',grid(maps['V81_SAM_9'],maps['valid9'])),
        (1550,'V82 SAM / nominal9',grid(maps['V82_SAM_9'],maps['valid9']))]
    for x,title,tile in panels:d.text((x,139),title,fill='white',font=small);im.paste(tile,(x,164))
    for j,text in enumerate(('Green=visible segmentation; gray=mask0; hatch=U/no mask. Mask0 is not causal E/S.',
            'q yellow / fixed GT-only r cyan. Same source images/q/L; no GT-best detector or mask.',
            'V82 adds PROVIDED category information. This is not a same-input ablation.')):
        d.text((480,380+28*j),text,fill='#d4dfea',font=font)
    d.text((20,522),'Stored learned-head outputs (fixed 0.5); SAM above is a binary mask sample, not a learned probability:',fill='#d4dfea',font=font)
    for j,name in enumerate(('V76','V77','V78','V80')):
        s=scores[name];color='#76e0aa' if s['prediction']==('VISIBLE' if case['GT_label'] else 'COVERED') else '#ffa28e'
        d.text((20+j*450,559),f'{name} pV={s["p_visible"]:.6f}',fill=color,font=bold)
        d.text((20+j*450,593),s['prediction'],fill=color,font=font)
    d.text((20,655),'Same four V79 cases fixed before V81/V82. Source support proxy, not human corner-cue D/E/S gold.',fill='#d4dfea',font=font)
    path=OUT/f'{n+1}_FIXED_REQUESTED_MASK.png';im.save(path)
    return path,phrase,old_phrase

def guard(sam_path,sam_sha,eval_path,eval_sha,comparison_path,comparison_sha):
    """No results, GT, RGB or NPZ body may be read before complete seals bind."""
    assert sha(sam_path)==sam_sha and sha(eval_path)==eval_sha and sha(comparison_path)==comparison_sha
    sam,ev,comparison=read(sam_path),read(eval_path),read(comparison_path)
    assert all(x['completed'] is True and x['PASS'] is True for x in (sam,ev,comparison))
    assert comparison['source_evaluation_seal']==bind(eval_path)
    for x in sam['outputs']:check(x)
    for x in ev['files']:check(x)
    for x in comparison['files']:check(x)
    for key in ('detector_output_seal','output_masks','summary','per_request'):check(sam[key])
    return sam,ev,comparison

def mask(row):
    if row['mask_status']=='U':
        assert row['status']=='U' and row['mask'] is None
        return None
    assert row['mask_status']=='OK' and row['status']=='ASSOCIATED';check(row['mask'])
    value=arrays(row['mask']['path'])['visible_mask']
    assert value.dtype==np.bool_ and value.shape==tuple(row['raw_hw'])
    return value

def cvtable(old_benchmarks,new_benchmarks,key):
    rows=[]
    assert [r['source'] for r in old_benchmarks]==[r['source'] for r in new_benchmarks]
    for baseline,new in zip(old_benchmarks,new_benchmarks):
        for name,record in (('V81',baseline),('V82',new)):
            m=record[key]
            rows.append([record['source'],name,m['N'],str(m['CM_rows_GT_CV_cols_pred_CVU']),pct(m['C_precision_known_CV_conditional']),pct(m['C_end_to_end_recall_U_is_miss']),pct(m['V_end_to_end_recall_U_is_miss']),f'{m["hold_N"]} ({pct(m["hold_rate"])})'])
    return table(['Source','조건','N','GT C/V × pred C/V/U','C precision*','C recall(U miss)','V recall(U miss)','보류'],rows)

def build(sam_path,sam_sha,eval_path,eval_sha,comparison_path,comparison_sha):
    samseal,evseal,cmpseal=guard(sam_path,sam_sha,eval_path,eval_sha,comparison_path,comparison_sha)
    assert not OUT.exists() and not (R/'RESULTS_KO.html').exists() and not (R/'RESULTS_KO.json').exists()
    evaldir,cmpdir=eval_path.parent,comparison_path.parent
    results=read(evaldir/'SOURCE_MASK_RESULTS.json');assert results['SAM_output_seal']==bind(sam_path)
    compare=read(cmpdir/'SAME_IDS_V81_V82_COMPARISON_KO.json')
    screen=read(cmpdir/'PROSPECTIVE_COMPONENT_SCREEN_KO.json');ownership=read(cmpdir/'OWNERSHIP_DIAGNOSTIC_KO.json')
    assert compare['completed'] and compare['PASS'] and compare['V82_eval_seal']==bind(eval_path)
    assert compare['all1035_IDs_GT_primary_qstates_strict_masks_exact'] and compare['not_same_input_ablation']
    assert screen['completed'] and screen['자동현재223평가승인'] is False
    old_eval=compare['V81_eval_seal'];check(old_eval);old_evseal=read(old_eval['path'])
    assert old_evseal['completed'] and old_evseal['PASS']
    for b in old_evseal['files']:check(b)
    oldresults=read(Path(old_eval['path']).parent/'SOURCE_MASK_RESULTS.json')
    oldsam=oldresults['SAM_output_seal'];check(oldsam);oldseal=read(oldsam['path'])
    assert oldseal['completed'] and oldseal['PASS']
    for b in oldseal['outputs']:check(b)
    for key in ('output_masks','per_RGB','summary'):check(oldseal[key])
    check(samseal['detector_output_seal']);dseal=read(samseal['detector_output_seal']['path']);check(dseal['summary'])
    detector_summary=read(dseal['summary']['path']);sam_summary=read(samseal['summary']['path'])
    previous=read(G79/'GALLERY_METADATA.json');cases=previous['cases']
    assert [(x['id'],x['native_k']) for x in cases]==EXPECTED
    prior=read(V80/'fixed_natural_gallery/GALLERY_METADATA.json')
    assert [x['id'] for x in prior['cases']]==[x['id'] for x in cases]
    associations=lines(samseal['output_masks']['path']);requests=lines(samseal['per_request']['path'])
    old_associations=lines(oldseal['output_masks']['path']);oldrgbs=lines(oldseal['per_RGB']['path'])
    assert len(associations)==len(requests)==len(old_associations)==1035 and len(oldrgbs)==435
    byrequest={x['request_id']:x for x in requests};byoldrgb={x['rgb_input_id']:x for x in oldrgbs}
    evaluation=arrays(evaldir/'SAVED_MASK_EVALUATION_PER_QUERY.npz')
    oldevaluation=arrays(Path(old_eval['path']).parent/'SAVED_MASK_EVALUATION_PER_QUERY.npz')
    for key in ('input_id','original_id','original_primary8','reference_local_label8','q_pixel_state8','q_pixel_valid8','strict535_mask8'):
        assert np.array_equal(evaluation[key],oldevaluation[key]),key
    byid={str(x):i for i,x in enumerate(evaluation['original_id'])}
    fixedcompare={(x['original_id'],x['native_k']):x for x in compare['same_fixed_four_cases']}
    assert list(fixedcompare)==EXPECTED
    OUT.mkdir();panels=[];records=[]
    for n,case in enumerate(cases):
        i=byid[case['id']];k=case['native_k'];a=associations[i];old_a=old_associations[i]
        rgb=byrequest[a['request_id']];oldrgb=byoldrgb[old_a['rgb_input_id']]
        assert evaluation['input_id'][i]==a['input_id']==old_a['input_id']
        assert a['initial_points']==old_a['initial_points'] and a['roi_side']==old_a['roi_side']==case['L']
        assert a['initial_points'][k]==case['q'] and np.array_equal(np.asarray(a['query_box_xyxy']),case['box'])
        check(rgb['detections']);check(oldrgb['detections'])
        det,old_det=arrays(rgb['detections']['path']),arrays(oldrgb['detections']['path'])
        check(case['source_RGB']);assert rgb['image_sha256']==oldrgb['image_sha256']==case['source_RGB']['sha256']
        with Image.open(case['source_RGB']['path']) as im:source_rgb=np.asarray(im.convert('RGB')).copy()
        assert list(source_rgb.shape)==[*a['raw_hw'],3]
        assert hashlib.sha256(source_rgb.tobytes()).hexdigest()==rgb['decoded_RGB_SHA256']==oldrgb['decoded_RGB_SHA256']
        gts=[]
        for key in ('M0','M4'):
            check(case['mask_pair'][key])
            with Image.open(case['mask_pair'][key]['path']) as im:gts.append(np.asarray(im)>0)
        assert not (gts[1]&~gts[0]).any()
        q,side=np.asarray(case['q']),case['L'];m0,v9=sample(gts[0],q,side,9,a['raw_hw']);m4,_=sample(gts[1],q,side,9,a['raw_hw'])
        oldmask,newmask=mask(old_a),mask(a)
        old9,_=sample(oldmask,q,side,9,a['raw_hw']);new9,_=sample(newmask,q,side,9,a['raw_hw']);new96,v96=sample(newmask,q,side,96,a['raw_hw'])
        pred=int(evaluation['prediction_CVU'][i,k]);oldpred=int(oldevaluation['prediction_CVU'][i,k])
        assert pred==(2 if newmask is None else int(new9[4,4]))
        assert oldpred==(2 if oldmask is None else int(old9[4,4]))
        iou=float(evaluation['IoU_allrows_nomask0'][i]);oldiou=float(oldevaluation['IoU_allrows_nomask0'][i])
        f=fixedcompare[(case['id'],k)]
        assert f['V81_prediction_CVU']==oldpred and f['V82_prediction_CVU']==pred and f['V81_whole_mask_IoU_no_mask0']==oldiou and f['V82_whole_mask_IoU_no_mask0']==iou
        maps=dict(M0_9=m0,M4_9=m4,V81_SAM_9=old9,V82_SAM_9=new9,valid9=v9)
        scores=prior['cases'][n]['scores']
        path,phrase,oldphrase=render_case(n,case,a,old_a,det,old_det,maps,source_rgb,scores,pred,oldpred,iou,oldiou)
        panels.append(bind(path));mp=OUT/f'{n+1}_FIXED_SAMPLED_MAPS.npz'
        u9=np.full((9,9),2,np.int8);u96=np.full((96,96),2,np.int8)
        np.savez_compressed(mp,M0_9=m0,M4_9=m4,native_valid9=v9,V81_predicted_mask9=u9 if old9 is None else old9.astype(np.int8),V82_predicted_mask9=u9 if new9 is None else new9.astype(np.int8),V82_predicted_mask96=u96 if new96 is None else new96.astype(np.int8),native_valid96=v96,q=q,L=np.asarray(side))
        records.append(dict(id=case['id'],input_id=a['input_id'],native_k=k,fixed_q=case['q'],fixed_r_GT_only=case['r'],L=side,requested_category=a['requested_category'],V82_predicted_phrase=phrase,V81_predicted_phrase=oldphrase,V81_mask_at_q=['NOT_VISIBLE_MASK','VISIBLE_MASK','U'][oldpred],V82_mask_at_q=['NOT_VISIBLE_MASK','VISIBLE_MASK','U'][pred],V81_whole_target_M4_IoU_nomask0=oldiou,V82_whole_target_M4_IoU_nomask0=iou,GT_reference_support=['C','V'][case['GT_label']],GT_query_support='O' if not m0[4,4] else 'V' if m4[4,4] else 'C',V82_association=a,V81_association=old_a,figure=bind(path),sampled_maps=bind(mp),same_old_RGB96=case['raw_local_RGB96'],same_old_target128=case['raw_target_RGB128'],source_RGB=case['source_RGB'],source_decode_exact_with_both_detectors=True,prior_scores=scores))
    contact=Image.new('RGB',(1840,710*4),'#111c28')
    for n,b in enumerate(panels):
        with Image.open(b['path']) as im:contact.paste(im,(0,n*710))
    contact.save(OUT/'CONTACT_SHEET.png')
    oldbench,newbench=oldresults['benchmarks'],results['benchmarks']
    ownershiprows=[]
    for a,b in zip(oldbench,newbench):
        for name,record in (('V81',a),('V82',b)):
            m=record['mask_IoU'];ownershiprows.append([record['source'],name,record['target_instances'],m['usable_selected_mask_N'],m['nomatch_N'],pct(m['mask_hold_rate']),decimal(m['all_rows_no_mask0_mean']),decimal(m['usable_selected_mask_conditional_mean'])])
    criteria=table(['사전 기준','실제 값','고정 한계','판정'],[[key,pct(x['value']),('≥' if x['comparison']=='ge' else '≤')+' '+pct(x['limit']),'PASS' if x['PASS'] else 'FAIL'] for key,x in screen['checks'].items()])
    ownerbins=table(['all800 YCB: GT-only global mask-IoU 순위 bin','N'],sorted(ownership['all800_bins'].items()))
    report=dict(completed=True,stage='PROVIDED_CATEGORY_SOURCE_COMPONENT',fully_solved=False,not_same_input_ablation=True,source_requested_category=dict(SYNTHETIC='fixed pallet constant',YCB_PUBLIC_REAL='source GT-metadata-provided canonical class; not predicted category/instance identity'),V81_benchmarks=oldbench,V82_benchmarks=newbench,prospective_component_screen=screen,ownership_diagnostic=ownership,paired_comparison=compare,fixed_four=records,figures=panels,contact_sheet=bind(OUT/'CONTACT_SHEET.png'),no_automatic_current_evaluation=True,current223=0,fit=0,PnP=0,report_builder_model_forward=0,source_RGB_figure_only_decodes=4,original_inputs_and_labels_unchanged=True,execution=dict(detector_unique_requests=detector_summary['unique_RGB_caption_requests'],detector_actual_RGB=detector_summary['unique_RGB'],detector_forward_calls=detector_summary['source_forward_calls'],SAM_embedding_calls=sam_summary['embedding_calls'],SAM_mask_predict_calls=sam_summary['associated'],detector_state_unchanged=detector_summary['state_unchanged'],SAM_state_unchanged=sam_summary['state_before']==sam_summary['state_after']),bindings=dict(SAM=bind(sam_path),evaluation=bind(eval_path),comparison=bind(comparison_path),old_V81_evaluation=old_eval,prior_gallery=bind(G79/'GALLERY_METADATA.json'),prior_selection=previous['selection_lock'],prior_scores=bind(V80/'fixed_natural_gallery/GALLERY_METADATA.json'),code=bind(__file__)))
    write(R/'RESULTS_KO.json',report)
    style='body{font-family:Arial,"NanumGothic",sans-serif;max-width:1480px;margin:32px auto;padding:0 24px;color:#17222b;line-height:1.65}h1{font-size:30px}h2{margin-top:34px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;border:1px solid #bdcad3;text-align:left}th{background:#e8f0f5}.notice{padding:18px;background:#edf4f8;border-left:5px solid #527a90}.warn{background:#fff1e9;border-color:#bb653f}img{width:100%;height:auto}a{color:#175575}'
    body=f'<!doctype html><html lang="ko"><meta charset="utf-8"><title>V82 요청 category 제공 source component</title><style>{style}</style><h1>V82: 대상 종류를 제공한 frozen grounding + SAM source 진단</h1>'
    body+='<p class="notice warn">사람의 corner cue D/E/S 판별 해결을 뜻하지 않습니다. 현재 223장 평가·학습·PnP는 0입니다. V82는 요청 category를 추가한 더 강한 입력 조건이며 V81과 동일 입력의 ablation이 아닙니다. 합성은 고정 pallet, 공개 YCB는 source GT metadata에서 제공된 category를 사용했습니다.</p>'
    body+=f'<p>모델·문턱·기존 q/L/box/native 순서를 고정했습니다. 각 caption은 category+마침표 하나이며 0.4/0.3 strict 문턱, 입력 q-box 최대 IoU≥0.1·동률 score/원 ordinal만으로 대응했습니다. GT mask·GT instance로 검출 후보나 SAM mask를 선택하지 않았습니다. 실제 detector는 원 RGB {detector_summary["unique_RGB"]}개에 대한 image/category 요청 {detector_summary["unique_RGB_caption_requests"]}개, 첫 반복 포함 {detector_summary["source_forward_calls"]}회입니다. SAM embedding {sam_summary["embedding_calls"]}회·mask {sam_summary["associated"]}회이며 상태 digest는 실행 전후 동일합니다. 보고서 생성은 저장 출력만 사용합니다.</p>'
    body+='<h2>전체 분모와 대상 mask IoU</h2>'+table(['Source','조건','전체 instance','mask 있음','U','U율','전체 mIoU(U=0)','mask 조건부 mIoU'],ownershiprows)
    body+='<p>U/no-match는 분모에 남기고 전체 mIoU에서는 0으로 포함합니다. U는 실제 all-zero mask가 아니므로 그림에서 별도 hatch로 표시합니다. mask가 있는 경우의 조건부 수치도 함께 표시합니다.</p>'
    for key,title in (('REFERENCE_LOCAL_PRIMARY_UNCHANGED_mask_at_q_component','원래 reference-local primary를 보존한 mask-at-q'),('SEPARATE_QUERY_PIXEL_CV_given_GT_M0_and_native_valid','별도 q-pixel GT M0 내부 C/V'),('strict535_same_frozen_IDs','같은 strict535 — C76 / V459')):
        body+='<h2>'+title+'</h2>'+cvtable(oldbench,newbench,key)
    body+='<p>*Precision은 해당 GT C/V-known subset 조건부 값입니다. Recall은 U도 miss로 포함합니다. GT O/unknown을 C로 합치지 않았습니다. q pixel의 mask0는 semantic external/self occlusion의 원인 판별이 아니며 reference/corner의 영상 단서 가시성도 직접 보증하지 않습니다. strict535는 이전에 고정한 같은 ID이며 새 점수에 맞춰 바꾸지 않았습니다.</p>'
    body+='<h2>실행 전 고정한 네 component 기준</h2>'+criteria
    body+='<p class="notice">전체 component screen: '+('PASS' if screen['component_screen_PASS'] else 'FAIL')+'. 이는 supplied-category source component의 공학적 screen이며, 통과하더라도 사람의 D/E/S·실제 initializer 오차 전이·같은 종류 다중 instance의 해결이나 자동 현재 평가 승인을 뜻하지 않습니다.</p>'
    body+='<h2>독립 GT-only 소유권 진단</h2>'+ownerbins
    body+=f'<p>같은 strict C76의 pred C/V/U={esc(ownership["strict_C76_prediction_CVU"])}. 새 false-visible {ownership["new_strict_C_falsevisible_N"]}개 중 다른 annotated instance가 q를 차지한 근거가 있는 수는 {ownership["new_falsevisible_other_visible_at_query_N"]}개입니다. 이 진단은 저장 mask의 intended-vs-other IoU를 사후 측정한 것이며 검출 선택에 사용하지 않았습니다. 범위가 다른 global IoU 순위와 local q ownership을 같은 원인으로 단정하지 않습니다.</p>'
    body+='<h2>같은 V79 네 실제 사례 — 성공 여부로 재선택하지 않음</h2><p>원본 RGB96·target128, q/L과 GT-only r marker를 그대로 유지합니다. 파랑 input box / 주황 V81 / 초록 V82 detector prompt입니다. 예측 phrase는 category 정답이나 instance identity와 다릅니다. nominal9는 실제 같은 q/L 좌표의 mask 표본이며 96 map도 별도 NPZ에 보존합니다.</p>'
    for record in records:body+=f'<figure><img src="{esc(record["figure"]["path"])}" alt="고정 사례 {esc(record["id"])}"><figcaption>{esc(record["id"])} / native{record["native_k"]}: requested {esc(record["requested_category"])}, V81 {esc(record["V81_mask_at_q"])} → V82 {esc(record["V82_mask_at_q"])}. 사례·query·L 선택은 결과로 바꾸지 않았습니다.</figcaption></figure>'
    body+='<h2>남는 범위</h2><p>요청 category는 instance identity가 아닙니다. 이 source에서 image/category 키가 모두 달라 같은 종류 다중 instance 분리 능력은 검증하지 못합니다. 공개 YCB q는 GT-derived source geometry와 synthetic residual을 이용한 emulated proposal이며 실제 배포 initializer 성공이 아닙니다. M0/M4는 source support proxy이고 virtual cuboid corner의 공중 위치·구멍·부분선 단서와 사람 semantic D/E/S는 다른 목표입니다. 반복 사용한 source VAL 개발 진단이며 foundation pretraining overlap도 알려져 있지 않습니다. 새 모델·동의어·문턱 탐색이나 현재 자료를 통한 재선택은 없습니다.</p>'
    body+='<p>활성 토큰 전체의 raw FP32 logits와 같은 postprocess tensor의 FP32 probabilities, token IDs/indices/attention mask를 저장해 .4/.3 판정을 재생할 수 있습니다. padded −inf는 실제 실행 guard certificate로 보존합니다. CPU sigmoid의 ULP 차이를 허용해 판정을 바꾸는 방식은 사용하지 않습니다.</p><p>'
    refs=[(eval_path,'전체 평가 seal'),(comparison_path,'동일 ID 비교·screen seal'),(cmpdir/'OWNERSHIP_DIAGNOSTIC_KO.json','소유권 진단'),(sam_path,'SAM seal'),(R/'EXPERIMENT_DECISIONS_BEFORE_INPUTS_AND_FORWARD.json','사전 결정'),(R/'SOURCE_COMPONENT_CRITERIA_BEFORE_FORWARD.json','사전 네 기준'),(R/'RESULTS_KO.json','보고서 JSON')]
    body+=' · '.join(link(p,t) for p,t in refs)+'</p></html>'
    with open(R/'RESULTS_KO.html','x') as f:f.write(body)
    write(OUT/'GALLERY_METADATA.json',dict(completed=True,cases=records,figures=panels,contact_sheet=bind(OUT/'CONTACT_SHEET.png'),same_fixed_V79_four=True,not_same_input_ablation=True,no_GT_detection_or_mask_choice=True,source_RGB_figure_only_decodes=4,model_forward_fit_current_PnP=0))
    write(OUT/'BUILD_SEAL.json',dict(completed=True,PASS=True,code=bind(__file__),report=bind(R/'RESULTS_KO.html'),report_JSON=bind(R/'RESULTS_KO.json'),gallery=bind(OUT/'GALLERY_METADATA.json'),figures=panels,contact=bind(OUT/'CONTACT_SHEET.png'),SAM=bind(sam_path),evaluation=bind(eval_path),comparison=bind(comparison_path),model_forward_fit_current_PnP=0))
    print(json.dumps(bind(OUT/'BUILD_SEAL.json')),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--sam-seal',type=Path,required=True);ap.add_argument('--sam-sha',required=True);ap.add_argument('--evaluation-seal',type=Path,required=True);ap.add_argument('--evaluation-sha',required=True);ap.add_argument('--comparison-seal',type=Path,required=True);ap.add_argument('--comparison-sha',required=True)
    a=ap.parse_args();build(a.sam_seal,a.sam_sha,a.evaluation_seal,a.evaluation_sha,a.comparison_seal,a.comparison_sha)

