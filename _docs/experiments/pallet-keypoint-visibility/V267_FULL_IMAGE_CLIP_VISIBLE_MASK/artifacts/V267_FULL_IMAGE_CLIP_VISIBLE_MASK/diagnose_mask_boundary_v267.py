"""Saved-map boundary diagnostic, not a new predictor or threshold policy."""
from pathlib import Path
import collections
import csv
import hashlib
import io
import json
import sys
import time
import types

import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage import distance_transform_edt, map_coordinates
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).absolute().parent
ARM = 'FULL_CLIP_SURGERY_ARCH'
BINS = ('LE3', 'GT3_LE6', 'GT6_LE16', 'GT16', 'EMPTYMASK')


def safe(value):
    p = Path(value)
    if not p.is_absolute() or any(x in str(p).lower() for x in ('v112', 'cache', '.pyc', '..')):
        raise ValueError(str(p))
    for a in reversed((p, *p.parents)):
        if a.is_symlink():
            raise ValueError(str(a))
    return p


def binding(path):
    p = safe(path)
    raw = p.read_bytes()
    return {'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)}


def checked(bound):
    assert binding(bound['path']) == bound
    return safe(bound['path'])


def write_json(path, value):
    p = safe(path); assert not p.exists()
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+'\n')


def write_csv(path, rows):
    p = safe(path); assert not p.exists()
    with p.open('x') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def raw_mask(logit, hw):
    x = torch.from_numpy(np.asarray(logit, np.float32))[None,None]
    return F.interpolate(x, size=tuple(hw), mode='bilinear', align_corners=False)[0,0].numpy() >= 0


def distance_bin(value):
    if value is None: return 'EMPTYMASK'
    if value <= 3: return 'LE3'
    if value <= 6: return 'GT3_LE6'
    if value <= 16: return 'GT6_LE16'
    return 'GT16'


def main():
    started = time.time(); torch.set_num_threads(1)
    out = safe(HERE/'mask_boundary_diagnostic_v1'); assert not out.exists()
    freeze_path = safe(HERE/'CURRENT_PREDICTIONS_FROZEN.json')
    freeze = json.loads(freeze_path.read_text()); assert freeze['PASS'] and freeze['completed']
    evaluation_path = safe(HERE/'evaluation_v2/COMPLETE.json')
    evaluation = json.loads(evaluation_path.read_text()); assert evaluation['PASS']
    joined_bound = evaluation['outputs']['all1784']
    rows = list(csv.DictReader(checked(joined_bound).open()))
    assert len(rows) == 1784 and [int(r['global_index']) for r in rows] == list(range(1784))
    queries = [json.loads(l) for l in checked(freeze['current_query_inputs']).read_text().splitlines()]
    frames = [json.loads(l) for l in checked(freeze['current_frame_inputs']).read_text().splitlines()]
    assert len(queries) == 1784 and len(frames) == 223
    with np.load(checked(freeze['frame_masks']), allow_pickle=False) as z:
        logits = np.array(z[ARM+'_logit'], copy=True)
    assert logits.shape == (223,32,32)
    false_indices = []
    all_visible = []
    for index, r in enumerate(rows):
        if r['PRIMARY'] == 'True' and r['human_state'] == 'DIRECT_VISIBLE':
            all_visible.append(index)
            if r[ARM+'_pred'] == 'E': false_indices.append(index)
    assert len(all_visible) == 1147 and len(false_indices) == 552
    wanted = set(false_indices)
    by_frame = collections.defaultdict(list)
    for index in false_indices: by_frame[int(queries[index]['frame_index'])].append(index)
    details = []
    frame_metrics = []
    for frame_index, frame in enumerate(frames):
        mask = raw_mask(logits[frame_index], frame['raw_hw'])
        area = int(mask.sum()); empty = area == 0
        frame_metrics.append({'frame_index':frame_index,'frame_id':frame['frame_id'],
             'predicted_target_pixels':area,'image_pixels':int(mask.size),'predicted_target_fraction':area/mask.size,
             'mask_empty':empty,'primary_D_false_non_target_queries':len(by_frame[frame_index])})
        if not by_frame[frame_index]: continue
        distance = None if empty else distance_transform_edt(~mask)
        for index in by_frame[frame_index]:
            r, q = rows[index], queries[index]
            assert q['available'] and int(q['global_index']) == index
            xx, yy = q['original_q0']; h,w = q['raw_hw']
            assert 0 <= xx < w and 0 <= yy < h
            value = None if empty else float(map_coordinates(distance, np.asarray([[yy],[xx]]), order=1, mode='nearest')[0])
            try: reference_error = float(r['initial_reference_error_px'])
            except ValueError: reference_error = float('nan')
            if not np.isfinite(reference_error): reference_error = None
            manual_near = (r['annotation_coordinate_source'] == 'manual_click' and reference_error is not None and reference_error <= 3)
            details.append({'global_index':index,'frame_index':int(q['frame_index']),'row_id':r['row_id'],
                'material':'WOOD' if r['group']=='WOOD_PRIMARY105' else 'SQUARE',
                'human_state':r['human_state'],'annotation_coordinate_source':r['annotation_coordinate_source'],
                'initial_reference_error_px':reference_error,'manual_D_initial_LE3':manual_near,
                'query_x':xx,'query_y':yy,'p_target':float(r[ARM+'_p_target']),
                'nearest_predicted_target_distance_px':value,'distance_bin':distance_bin(value),
                'frame_predicted_target_fraction':area/mask.size,'frame_mask_empty':empty})
    details.sort(key=lambda r:r['global_index'])
    assert {r['global_index'] for r in details} == wanted
    def is_manual_near(index):
        r=rows[index]
        try: e=float(r['initial_reference_error_px'])
        except ValueError: return False
        return r['annotation_coordinate_source']=='manual_click' and np.isfinite(e) and e <= 3
    manual_count=sum(is_manual_near(i) for i in all_visible)
    assert manual_count==376 and sum(r['manual_D_initial_LE3'] for r in details)==182
    table=[]
    for material in ('ALL','WOOD','SQUARE'):
        for cohort in ('ALL_PRIMARY_D','MANUAL_D_INITIAL_LE3','OTHER_PRIMARY_D'):
            def match_index(index):
                r=rows[index];m='WOOD' if r['group']=='WOOD_PRIMARY105' else 'SQUARE'
                return (material=='ALL' or material==m) and (cohort=='ALL_PRIMARY_D' or is_manual_near(index)==(cohort=='MANUAL_D_INITIAL_LE3'))
            denominator=sum(match_index(i) for i in all_visible)
            chosen=[r for r in details if match_index(r['global_index'])]
            table.append({'material':material,'cohort':cohort,'D_total':denominator,
                 'D_predicted_target':denominator-len(chosen),'D_false_non_target':len(chosen),
                 **{name:sum(r['distance_bin']==name for r in chosen) for name in BINS}})
    # Fixed six diagnostic cells. Prefer the location-accurate manual cohort;
    # fallback to other PRIMARY D errors only within the same bin/material.
    cells=[('WOOD','LE3'),('SQUARE','LE3'),('ALL','GT3_LE6'),('ALL','GT6_LE16'),('ALL','GT16'),('ALL','EMPTYMASK')]
    selection=[]
    for material, name in cells:
        pool=[r for r in details if r['distance_bin']==name and (material=='ALL' or r['material']==material)]
        preferred=[r for r in pool if r['manual_D_initial_LE3']]
        use=preferred or pool
        selection.append({'material':material,'distance_bin':name,'census':len(pool),'manual_near3_census':len(preferred),
                          'global_index':use[0]['global_index'] if use else None,
                          'selection_cohort':'MANUAL_D_INITIAL_LE3' if preferred else 'OTHER_PRIMARY_D' if use else 'EMPTY_CELL'})
    assert 200 in wanted
    gallery_selection=list(selection)
    if 200 not in {c['global_index'] for c in selection}:
        gallery_selection.append({'material':'WOOD','distance_bin':next(r['distance_bin'] for r in details if r['global_index']==200),
             'global_index':200,'selection_cohort':'EXPLICIT_PRIOR_USER_FEEDBACK_CASE','reason':'User previously judged the small position change acceptable; Root explicitly requested this visible accurate-query case.'})
    out.mkdir()
    summary={'completed':True,'PASS':True,'arm':ARM,'primary_D_total':1147,'false_non_target':552,
        'manual_D_initial_LE3_total':376,'manual_near3_false_non_target':182,
        'method':'Saved32logits ->originalHW bilinear align_cornersFalse ->logit>=0 mask. EDT(~mask) in raw-pixel units, bilinear EDT sampling at original continuous q.',
        'bins':{'LE3':'distance<=3px','GT3_LE6':'3<distance<=6px','GT6_LE16':'6<distance<=16px','GT16':'distance>16px','EMPTYMASK':'No predicted target pixel in this full frame'},
        'tables':table,'predicted_empty_frames':sum(r['mask_empty'] for r in frame_metrics),
        'scope':'Post-hoc boundary proximity diagnostic only; no predictor, threshold/policy, morphology or coordinate changes.',
        'limitations':['Real pixel segmentation GT is absent. Distance to the predicted mask is not a measured segmentation boundary error.','A point near a predicted boundary may reflect coarse output, location error, missed pallet regions or wrong-instance segmentation; these counts do not establish cause.','manualDinitial<=3 reduces initial-location confounding but is not a new independent evaluation cohort.','No dilation/nearest-mask coordinate rescue is adopted or scored as a new policy.'],
        'inputs':{'freeze':binding(freeze_path),'evaluation':binding(evaluation_path),'joined':joined_bound,'masks':freeze['frame_masks'],'queries':freeze['current_query_inputs'],'frames':freeze['current_frame_inputs']},
        'full_RGB_reads_before_metrics_and_selection_save':0,'optimizer_updates':0,'new_model_forwards':0}
    write_json(out/'RESULTS_BEFORE_RGB.json',summary)
    write_csv(out/'DISTANCE_BINS.csv',table);write_csv(out/'D_FALSE_NON_TARGET552_DETAILS.csv',details);write_csv(out/'FRAME_MASK_CENSUS223.csv',frame_metrics)
    write_json(out/'CASE_SELECTION_BEFORE_RGB.json',{'completed':True,'rule':'Six material/bin cells, lowest originalglobalindex in manualDinitial<=3 cohort when present; else lowest same cell PRIMARY D error. Empty cells retained.','fixed_six_selection':selection,'explicit_prior_user_feedback_case':200,'gallery_selection':gallery_selection})
    # Selected RGB decoding only after the frozen diagnostic records above.
    protocol=json.loads(checked(json.loads(safe(HERE/'NEW_CONFIG_FROZEN_BEFORE_CURRENT.json').read_text())['protocol']).read_text())
    bound=protocol['current_input_helper'];p=checked(bound);m=types.ModuleType('v267_boundary_input_only');m.__file__=str(p);sys.modules[m.__name__]=m;exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__)
    fontpath=safe(Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'));font=ImageFont.truetype(str(fontpath),20);small=ImageFont.truetype(str(fontpath),17)
    by_id={r['global_index']:r for r in details};cards=[];rgb_bindings=[]
    for cell in gallery_selection:
        index=cell['global_index']
        if index is None:continue
        r=by_id[index];q=queries[index];frame=frames[r['frame_index']];rgb=m.read_rgb(frame);mask=raw_mask(logits[r['frame_index']],frame['raw_hw'])
        rgb_bindings.append(frame['image_binding'])
        overlay=rgb.astype(np.float32);overlay[mask]=.55*overlay[mask]+.45*np.asarray([30,235,100],np.float32)
        h,w=rgb.shape[:2];display_w=640;display_h=round(h*display_w/w);card=Image.new('RGB',(1320,display_h+140+(350 if index==200 else 0)),'white');draw=ImageDraw.Draw(card)
        dist='EMPTYMASK' if r['nearest_predicted_target_distance_px'] is None else f'{r["nearest_predicted_target_distance_px"]:.2f}px'
        draw.text((16,8),f'G{index} {r["material"]} · 기존 보임 D / 예측 non-target · 거리 {dist}',font=font,fill='#17324a')
        draw.text((16,40),f'수동 클릭·초기≤3px={r["manual_D_initial_LE3"]}, 초기참조오차={r["initial_reference_error_px"]}px, P(target)={r["p_target"]:.4f}',font=small,fill='#17324a')
        for col,array in enumerate((rgb,np.clip(overlay,0,255).astype(np.uint8))):
            tile=Image.fromarray(array).resize((display_w,display_h),Image.Resampling.LANCZOS);td=ImageDraw.Draw(tile);x=q['original_q0'][0]*display_w/w;y=q['original_q0'][1]*display_h/h;td.line((x-10,y,x+10,y),fill='#ff00d4',width=3);td.line((x,y-10,x,y+10),fill='#ff00d4',width=3);card.paste(tile,(16+col*656,82))
        draw.text((16,display_h+92),'왼쪽 원본 전체 이미지 · 오른쪽 초록=예측 target mask · 분홍+=초기 q. 표시용이며 입력·판정은 변경하지 않음.',font=small,fill='#9d3516')
        if index==200:
            x0,y0=np.floor(np.asarray(q['original_q0'])-48).astype(int);box=(int(x0),int(y0),int(x0+96),int(y0+96))
            yy0=display_h+160
            draw.text((16,display_h+126),f'G200 별도 확대: 원 CLIP P(target)={float(rows[index]["FULL_CLIP_p_target"]):.3f}, CS={r["p_target"]:.3f}, 이전 V264={rows[index]["V264_HEATMAP_LOCAL_pred"]}',font=small,fill='#17324a')
            for col,array in enumerate((rgb,np.clip(overlay,0,255).astype(np.uint8))):
                crop=Image.fromarray(array).crop(box).resize((288,288),Image.Resampling.LANCZOS);cd=ImageDraw.Draw(crop);cx=(q['original_q0'][0]-x0)*3;cy=(q['original_q0'][1]-y0)*3;cd.line((cx-9,cy,cx+9,cy),fill='#ff00d4',width=3);cd.line((cx,cy-9,cx,cy+9),fill='#ff00d4',width=3);card.paste(crop,(16+col*320,yy0))
            draw.text((672,yy0+8),'표시 전용 96px 창 확대\n모델은 이 crop을 입력받지 않았습니다.\n초기 q와 예측 map은 그대로입니다.\n실사 pixel GT가 없어 분할 IoU는 미측정입니다.',font=small,fill='#17324a')
        path=safe(out/f'CASE_G{index:04d}.png');card.save(path);cards.append({'global_index':index,'cell':cell,'detail':r,'image':binding(path)})
    html='<!doctype html><meta charset="utf-8"><title>V267 마스크 경계 진단</title><style>body{font:18px sans-serif;max-width:1360px;margin:30px auto;background:#f5f7f9;color:#123}img{max-width:100%}p{line-height:1.6}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccd}</style><h1>보이는 점을 놓친 552점: 예측 마스크에서 얼마나 떨어졌는가</h1><p>원본 예측과 0.5 문턱을 유지한 저장 출력 진단입니다. 실제 픽셀 분할 정답이 없어 경계 오류의 원인을 확정하지 않습니다. 가까운 점을 새로 보임으로 바꾸거나 좌표를 이동하지 않았습니다.</p><table><tr><th>유형/부분집합</th><th>보임 전체</th><th>누락</th>'+''.join('<th>'+x+'</th>' for x in BINS)+'</tr>'
    for r in table:html+='<tr><td>'+r['material']+'/'+r['cohort']+'</td><td>'+str(r['D_total'])+'</td><td>'+str(r['D_false_non_target'])+'</td>'+''.join('<td>'+str(r[x])+'</td>' for x in BINS)+'</tr>'
    html+='</table><p>거리 단위는 원영상 px입니다. LE3=3이하, GT3_LE6=3초과6이하, GT6_LE16=6초과16이하, GT16=16초과, EMPTYMASK=프레임 전체에 예측 target 없음.</p><p><a href="../evaluation_v2/INDEX_KO.html">전체 성능·보임 성공 사례</a> · <a href="D_FALSE_NON_TARGET552_DETAILS.csv">552점 상세</a></p>'
    for cell in gallery_selection:
        if cell['global_index'] is None:html+='<p>'+str(cell)+' — 빈 셀 유지</p>'
        else:html+='<img src="CASE_G%04d.png">'%cell['global_index']
    safe(out/'INDEX_KO.html').write_text(html)
    write_json(out/'VISUAL_CASE_DATA.json',{'completed':True,'cards':cards,'selected_original_RGB_files_read':rgb_bindings,'no_pixel_GT_claim':True})
    names=['RESULTS_BEFORE_RGB.json','DISTANCE_BINS.csv','D_FALSE_NON_TARGET552_DETAILS.csv','FRAME_MASK_CENSUS223.csv','CASE_SELECTION_BEFORE_RGB.json','VISUAL_CASE_DATA.json','INDEX_KO.html']+[Path(c['image']['path']).name for c in cards]
    write_json(out/'COMPLETE.json',{'completed':True,'PASS':True,'code':binding(Path(__file__).absolute()),'outputs':{n:binding(out/n) for n in names},'selected_RGB_decodes':len(cards),'new_encoder_or_head_forwards':0,'optimizer_updates':0,'policy_label_or_threshold_changes':0,'elapsed_seconds':time.time()-started})
    print(json.dumps({'completed':True,'PASS':True,'complete':binding(out/'COMPLETE.json'),'tables':table,'selection':selection},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
