"""Independent saved public-output audit and fixed-order source RGB overlays."""
from pathlib import Path
from collections import Counter
import csv, hashlib, json, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parents[1]
AUDIT=ROOT/'OUTPUT_AUDIT'
VIS=ROOT/'VISUAL_REVIEW'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
COLORS=['#ff3838','#ff9d00','#ffe000','#39ff14','#00eaff','#2d81ff','#aa64ff','#ff54cf','#ffffff']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def load(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def write(p,obj):
    assert not p.exists(),str(p)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def quant(a):return {str(p):float(np.percentile(a,p)) for p in [0,25,50,75,100]} if a else None
def text(draw,xy,s,size=16,fill='white'):
    draw.text(xy,s,font=ImageFont.truetype(FONT,size),fill=fill,stroke_width=1,stroke_fill='#000000')
def mark_rgb(im,out,all_candidates=False):
    result=im.copy();d=ImageDraw.Draw(result);w,h=im.size
    scale=max(1,round(w/640));r=max(3,round(w/250));font=max(14,round(w/80))
    candidates=out['candidates'] if all_candidates else ([] if out['selected_index'] is None else [out['candidates'][out['selected_index']]])
    for c in candidates:
        selected=c['candidate_index']==out['selected_index']
        color='#00ffff' if selected else '#cc66cc'
        d.rectangle(c['box_xyxy'],outline=color,width=scale*(2 if selected else 1))
        for k,p in enumerate(c['keypoints_xy']):
            x,y=p
            d.ellipse([x-r,y-r,x+r,y+r],fill=COLORS[k] if selected else '#cc66cc',outline='black',width=1)
            if selected:text(d,(x+r,y-r),str(k),font,COLORS[k])
        if selected:text(d,(max(0,c['box_xyxy'][0]),max(0,c['box_xyxy'][1]-font-2)),f"sel {c['candidate_index']} conf={c['score']:.5f}",font)
    if out['no_detection']:text(d,(12,12),'NO DETECTION (retained)',max(22,font),'#ff5555')
    return result
def thumbnail(im,size):
    result=Image.new('RGB',size,'#181c24');im=im.copy();im.thumbnail(size,Image.Resampling.LANCZOS)
    result.paste(im,((size[0]-im.width)//2,(size[1]-im.height)//2));return result

def main():
    AUDIT.mkdir(exist_ok=False);VIS.mkdir(exist_ok=False)
    (VIS/'SELECTED_RAW9_OVERLAYS').mkdir();(VIS/'ALL_CANDIDATES_RAW9_OVERLAYS').mkdir();(VIS/'CUE_RAW_MARKED_CARDS').mkdir()
    inp=rows(ROOT/'INPUTS_RGB_ONLY.jsonl');outs=rows(ROOT/'FROZEN_PROPOSALS.jsonl')
    receipt=load(ROOT/'INFERENCE_RESULT.json');started=load(ROOT/'RUN_STARTED.json')
    assert sha(ROOT/'INFERENCE_RESULT.json')=='09a2669110d73e52aab3e1ac5c6c91f5156b9e3c89ac6e46cbcb0f415c2461cd'
    for filename in ['INPUT_LOCK.json','RUN_ENTRYLOCK.json']:
        for b in load(ROOT/filename)['bindings']:assert sha(b['path'])==b['sha256']
    for b in receipt['outputs']+[receipt['code'],receipt['entry'],started['entry']]:assert sha(b['path'])==b['sha256']
    assert len(inp)==len(outs)==48 and [a['id'] for a in inp]==[b['input_id'] for b in outs]
    assert receipt['runtime']['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    assert receipt['inference_initialization_state_digest']==receipt['after_forward_state_digest']==started['frozen_before_forward_state_digest']
    assert receipt['loaded_before_standard_inference_fusion_digest']==started['loaded_before_standard_fusion_state_digest']
    assert all(receipt[k] is True for k in ['completed','source_only','all_candidates_retained','standard_fusion_before_forward_snapshot','trained_parameters_and_registered_buffers_unchanged_during_all_forward_calls','checkpoint_unchanged','model_eval_and_no_grad','prediction_first_repeat_bit_exact','P3_first_repeat_bit_exact'])
    assert receipt['explicit_frame_predict_calls']==49 and receipt['main_frame_predict_calls']==48 and receipt['repeat_first_predict_calls']==1
    assert sum(o['forward_hooks'] for o in outs)+receipt['repeat_forward_hooks']==receipt['total_module_forward_hooks']==50
    with np.load(ROOT/'P3_INPUT_FEATURES_ONLY.npz',allow_pickle=False) as z:
        assert z.files==['p3'];p3=z['p3']
    assert p3.shape==(48,64,80,80) and p3.dtype==np.float16 and np.isfinite(p3).all()
    assert p3.nbytes==receipt['P3_uncompressed_bytes']==39321600 and p3.nbytes<=100000000
    per=[];images=[];all_points=[];selected_points=[];selected_scores=[];tie_frames=[]
    for n,(i,o) in enumerate(zip(inp,outs)):
        assert i['image_sha256']==o['image_sha256']==sha(i['image'])
        im=Image.open(i['image']).convert('RGB');a=np.asarray(im)
        assert list(a.shape[:2])==i['raw_hw'] and sha_bytes(a.tobytes())==i['RGB_pixels_sha256']
        images.append(im)
        h,w=i['raw_hw'];assert o['canvas_shape']==[h+200,w+200] and o['added_border']==100
        ih,iw=o['input_shape'];assert max(ih,iw)==640 and ih%32==iw%32==0
        assert np.all(p3[n,:,ih//8:,:]==0) and np.all(p3[n,:,:,iw//8:]==0)
        cc=o['candidates'];assert [c['candidate_index'] for c in cc]==list(range(len(cc)))
        for c in cc:
            b=np.asarray(c['box_xyxy']);q=np.asarray(c['keypoints_xy']);v=np.asarray(c['keypoints_conf'])
            assert b.shape==(4,) and q.shape==(9,2) and v.shape==(9,)
            assert np.isfinite(b).all() and np.isfinite(q).all() and np.isfinite(v).all() and math.isfinite(c['score'])
            assert b[2]>=b[0] and b[3]>=b[1] and (v>=0).all() and (v<=1).all() and .001<=c['score']<=1
            all_points.extend([(k,not(0<=x<w and 0<=y<h)) for k,(x,y) in enumerate(q)])
        if cc:
            scores=[c['score'] for c in cc];sel=int(np.argmax(scores));assert o['selected_index']==sel and not o['no_detection']
            ties=sum(s==scores[sel] for s in scores)
            if ties>1:tie_frames.append(i['id'])
            sc=cc[sel];selected_scores.append(sc['score'])
            selected_points.extend([(k,not(0<=x<w and 0<=y<h)) for k,(x,y) in enumerate(sc['keypoints_xy'])])
        else:assert o['selected_index'] is None and o['no_detection']
        stem=f"{n:02d}_{i['source_clip']}_{i['sensor']}_{i['sensor_slot']:02d}.png"
        selected=mark_rgb(im,o);allmark=mark_rgb(im,o,True)
        selected.save(VIS/'SELECTED_RAW9_OVERLAYS'/stem);allmark.save(VIS/'ALL_CANDIDATES_RAW9_OVERLAYS'/stem)
        per.append(dict(input_id=i['id'],source_clip=i['source_clip'],sensor=i['sensor'],sensor_slot=i['sensor_slot'],candidate_count=len(cc),no_detection=o['no_detection'],selected_index=o['selected_index'],selected_score=None if not cc else cc[o['selected_index']]['score'],input_shape=o['input_shape'],selected_overlay=str(VIS/'SELECTED_RAW9_OVERLAYS'/stem),all_candidates_overlay=str(VIS/'ALL_CANDIDATES_RAW9_OVERLAYS'/stem)))
    assert sum(x['no_detection'] for x in per)==receipt['no_detection_frames']==8
    assert sum(x['candidate_count'] for x in per)==receipt['candidate_count']==220
    stream={}
    for clip,sensor in [('r2b_storage_v3','D455_COLOR'),('r2b_storage_v3','HAWK_LEFT'),('r2b_galileo2_v1','D455_COLOR')]:
        pp=[r for r in per if r['source_clip']==clip and r['sensor']==sensor]
        stream[f'{clip}:{sensor}']=dict(frames=len(pp),no_detection=sum(r['no_detection'] for r in pp),candidates=sum(r['candidate_count'] for r in pp),selected_confidence_quantiles=quant([r['selected_score'] for r in pp if r['selected_score'] is not None]))
    pages=[]
    for pg in range(3):
        canvas=Image.new('RGB',(2240,1480),'#101722');d=ImageDraw.Draw(canvas)
        text(d,(16,12),f'FIXED INPUT ORDER {pg*16:02d}-{pg*16+15:02d}: raw RGB | confidence-selected RAW9; no cue/GT selection',22)
        text(d,(16,43),'Point colors/labels are raw indices0..8, not certified native8 corner IDs. All candidates retained separately. No detection stays visible.',16)
        for local in range(16):
            n=pg*16+local;x=(local%4)*560;y=80+(local//4)*350;o=outs[n];i=inp[n]
            text(d,(x+8,y+3),f"{n:02d} {i['sensor']} slot{i['sensor_slot']:02d} N={len(o['candidates'])}",14)
            selected=mark_rgb(images[n],o)
            canvas.paste(thumbnail(images[n],(272,275)),(x+4,y+30));canvas.paste(thumbnail(selected,(272,275)),(x+282,y+30))
            s='NO DETECTION' if o['no_detection'] else f"selected={o['selected_index']} conf={o['candidates'][o['selected_index']]['score']:.5f}"
            text(d,(x+8,y+312),s,14,'#ff8888' if o['no_detection'] else 'white')
        page=VIS/f'FIXED_ORDER_PAGE_{pg+1}.png';canvas.save(page);pages.append(bind(page))

    ledger_path=BASE/'query_visibility_reference_jitter_20261023_v71/data/r2b_cue_review/R2B_WOOD_CUE_PROPOSAL_LEDGER.json'
    ledger=load(ledger_path);assert sha(ledger_path)=='57abc23bebdf5ed7841e45de15358359807bf7c5a9f03c3495309f451a7ef282'
    by_hash={i['image_sha256']:n for n,i in enumerate(inp)};cue_rows=[];cards=[]
    for k,c in enumerate(ledger['records']):
        n=by_hash[c['image_sha256']];o=outs[n];i=inp[n];xy=c['associated_cue_xy'];unc=c['uncertainty_bbox']
        record=dict(record_id=c['record_id'],cue_refID=c['cue_refID'],input_id=i['id'],input_row=n,image_sha256=i['image_sha256'],fixed_cue_xy=xy,fixed_uncertainty_bbox=unc,cue_proposal_state=c['proposal_state'],native8_mapping='UNKNOWN',human_verified=False,GT_admitted=False,selected_index=o['selected_index'],no_detection=o['no_detection'],selected_score=None,nearest_raw9_geometric_query=None,nearest_first8_geometric_query=None,raw9_points_inside_fixed_uncertainty=None,selected_box_contains_cue_xy=None,analysis_is_GT_only_not_model_input=True,selected_detection_changed=False)
        if o['selected_index'] is not None:
            candidate=o['candidates'][o['selected_index']];record['selected_score']=candidate['score']
            record['selected_box_xyxy']=candidate['box_xyxy'];record['selected_raw9_xy']=candidate['keypoints_xy'];record['selected_raw9_conf']=candidate['keypoints_conf']
            if xy is not None:
                q=np.asarray(candidate['keypoints_xy']);dist=np.linalg.norm(q-xy,axis=1)
                for nm,limit in [('nearest_raw9_geometric_query',9),('nearest_first8_geometric_query',8)]:
                    j=int(np.argmin(dist[:limit]));record[nm]=dict(raw_index=j,xy=q[j].tolist(),distance_native_px=float(dist[j]),point_confidence=candidate['keypoints_conf'][j])
                x0,y0,x1,y1=unc;record['raw9_points_inside_fixed_uncertainty']=[j for j,(x,y) in enumerate(q) if x0<=x<x1 and y0<=y<y1]
                b=candidate['box_xyxy'];record['selected_box_contains_cue_xy']=b[0]<=xy[0]<b[2] and b[1]<=xy[1]<b[3]
        # 160-native-pixel cue crop (same fixed earlier bounds) with raw shown beside marks.
        bounds=c['diagnostic_native_crop_bounds_xyxy'];raw=images[n].crop(bounds);marked=mark_rgb(images[n],o)
        if xy is not None:
            dr=ImageDraw.Draw(marked);dr.rectangle(unc,outline='#ffff00',width=3);x,y=xy
            dr.line((x-12,y,x+12,y),fill='#ffff00',width=2);dr.line((x,y-12,x,y+12),fill='#ffff00',width=2)
        marked=marked.crop(bounds)
        card=Image.new('RGB',(1000,560),'#101722');dr=ImageDraw.Draw(card)
        text(dr,(12,10),f"{c['record_id']} | frame{n:02d} | raw cue crop / SAME confidence-selected prediction",20)
        text(dr,(12,43),'Yellow=fixed AI cue/uncertainty; colored0..8=raw queries; no relocation / no GT detection choice',15)
        card.paste(raw.resize((420,420),Image.Resampling.NEAREST),(35,80));card.paste(marked.resize((420,420),Image.Resampling.NEAREST),(540,80))
        if xy is None:desc='U/null reference: no point association metric is created.'
        elif o['no_detection']:desc='No detection: fixed visible-cue proposal remains source data; no initializer query available.'
        else:desc=f"selected conf={record['selected_score']:.5f}; nearest raw9 index{record['nearest_raw9_geometric_query']['raw_index']} distance={record['nearest_raw9_geometric_query']['distance_native_px']:.2f}px"
        text(dr,(12,520),desc,16)
        cp=VIS/'CUE_RAW_MARKED_CARDS'/f'{k:02d}_{c["record_id"]}.png';card.save(cp);record['review_card']=bind(cp);cards.append(card);cue_rows.append(record)
    combined=Image.new('RGB',(2000,2240),'#101722')
    for k,card in enumerate(cards):combined.paste(card,((k%2)*1000,(k//2)*560))
    cue_sheet=VIS/'FIXED_CUE_ASSOCIATION_RAW_MARKED.png';combined.save(cue_sheet)
    association=dict(schema='V72_FIXED_AI_CUE_TO_UNCHANGED_INITIALIZER_GT_ONLY_ASSOCIATION',records=cue_rows,source_cue_ledger=bind(ledger_path),resolved_visible_AI_cue_proposals=7,U_null=1,cue_frames_with_detection=6,cue_frames_without_detection=1,geometry_only='Nearest point is an analysis distance among the already selected detection raw9, not a semantic correspondence or a permitted detection-selection rule. No semantic correctness threshold is chosen.',native8_mapping='UNKNOWN',family='NVIDIA_R2B_STORAGE_GALILEO_FAMILY_UNRESOLVED',split='UNASSIGNED_GROUP_HOLD',labels='AI nonhuman conditional proposals only; no target mask, human gold, naturalE/S or training admission generated.',visual_review_pending=True,contact_sheet=bind(cue_sheet))
    write(AUDIT/'CUE_ASSOCIATION_ANALYSIS.json',association)
    report=dict(schema='V72_INDEPENDENT_SAVED_PUBLIC_INITIALIZER_OUTPUT_AUDIT',PASS=True,frames=48,no_detection_frames=8,candidates=220,selected_frames=40,all_candidates_raw9_points=1980,selected_raw9_points=360,by_stream=stream,first_argmax_tie_frames=tie_frames,all_candidate_out_of_native_bounds_by_raw_index=dict(Counter(k for k,b in all_points if b)),selected_out_of_native_bounds_by_raw_index=dict(Counter(k for k,b in selected_points if b)),P3=dict(shape=list(p3.shape),dtype=str(p3.dtype),finite=True,padded_region_exact_zero=True,uncompressed_bytes=p3.nbytes),state_and_replay=dict(receipt_digest_consistency=True,post_initialization_state_unchanged_recorded=True,checkpoint_hash_recomputed_unchanged=True,first_frame_exact_prediction_and_P3_repeat_receipt_PASS=True,explicit_calls=49,module_forward_hooks=50,internal_warmup_hooks=1,independent_reexecution=False),scope_limits=['ALL-candidate retention follows bound code list serialization plus sealed output/receipt; transient model results are not regenerated here.','Model state/repeat assertions are independently checked for receipt/start digest consistency and reviewed code. No fresh model load/replay or independent tensor state recomputation.','PNG RGB hashes, actualP3 contents, output IDs/shapes/argmax/coordinates/counters and zero padding are independently recalculated.','Raw9 includes ninthcenter; geometriccue proximity does not identify semanticnative8 or prove visibility.'],runtime_CUBLAS_WORKSPACE_CONFIG=receipt['runtime']['CUBLAS_WORKSPACE_CONFIG'],source_only=True,current223_model_forward_fit_PnP=0,all48_visuals_fixed_manifest_order=True,source_bindings=[bind(ROOT/x) for x in ['INPUTS_RGB_ONLY.jsonl','PROTOCOL.json','INPUT_LOCK.json','RUN_ENTRYLOCK.json','INFERENCE_RESULT.json','RUN_STARTED.json','FROZEN_PROPOSALS.jsonl','P3_INPUT_FEATURES_ONLY.npz','EXPECTED_OUTPUT_AUDIT_PROTOCOL.json']],audit_code=bind(Path(__file__)),figures=pages+[bind(cue_sheet)],cue_analysis=bind(AUDIT/'CUE_ASSOCIATION_ANALYSIS.json'),per_frame=per)
    write(AUDIT/'INDEPENDENT_SAVED_OUTPUT_AUDIT.json',report)
    with (AUDIT/'PER_FRAME_AUDIT.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(per[0]));writer.writeheader();writer.writerows(per)
    print(json.dumps(dict(PASS=True,audit=bind(AUDIT/'INDEPENDENT_SAVED_OUTPUT_AUDIT.json'),cue_analysis=bind(AUDIT/'CUE_ASSOCIATION_ANALYSIS.json'),figures=pages+[bind(cue_sheet)]),ensure_ascii=False))

def sha_bytes(data):return hashlib.sha256(data).hexdigest()

if __name__=='__main__':main()
