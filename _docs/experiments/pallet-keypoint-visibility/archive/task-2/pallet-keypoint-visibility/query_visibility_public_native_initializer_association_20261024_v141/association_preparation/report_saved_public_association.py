"""Saved-input source association audit; inert until an explicit root entry.

No detector/head execution or target selection. Source proposal states stay
attached to their original entity/role; actual query semantic labels stay U.
"""
from pathlib import Path
import argparse
import hashlib
import html
import json
import math
import numpy as np

H = Path(__file__).resolve().parent
V = H.parent
P = V.parent
POLICY = V / 'ROOT_SOURCE_INITIALIZER_ASSOCIATION_POLICY_BEFORE_FORWARD.json'
POLICY_SHA = '99e3efb5693da7ecb0866bdb0b931d6c21d1f56b31351ceaf7e2db50c6866d41'
INPUTS = V / 'initializer_preparation/INPUTS16_RGB_ONLY.jsonl'
PRED = V / 'actual_public_initializer_canonical_transport_v2/PREDICTION_SEAL.json'
PROTOCOL = H / 'PROTOCOL_BEFORE_SAVED_ASSOCIATION.json'
GENERATED = H / 'GENERATED_ASSOCIATION_CHECKS.json'
OUT = V / 'saved_public_association_v1'
FORBIDDEN = ('model_forward', 'head_forward', 'fit', 'private_current', 'V112',
             'PnP', 'pose_optimization', 'q_L_correction', 'GT_permutation',
             'detection_reselection', 'automatic_label_transfer', 'automatic_retry')
LEGACY_FIXED = {
    '0.55m_lab:0017': 'PALLOX50:0.55m_lab:0016_0017:after',
    '1.23m_hall:0012': 'PALLOX50:1.23m_hall:0012_0013:before',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]


def check(binding):
    path = Path(binding['path'])
    assert '_v112' not in str(path).lower()
    assert sha(path) == binding['sha256'], str(path)
    return path


def write(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def clean(x):
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        return clean(x.item())
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x


def xyxy(box, xywh=False):
    if box is None:
        return None
    a = np.asarray(box, np.float64)
    assert a.shape == (4,)
    if not np.isfinite(a).all():
        return None
    if xywh:
        assert a[2] > 0 and a[3] > 0
        a = np.array([a[0], a[1], a[0]+a[2], a[1]+a[3]])
    if a[2] <= a[0] or a[3] <= a[1]:
        return None
    return a


def boxfacts(a, b):
    if a is None or b is None:
        return None
    assert a[2] > a[0] and a[3] > a[1] and b[2] > b[0] and b[3] > b[1]
    area_a = float(np.prod(a[2:]-a[:2]))
    area_b = float(np.prod(b[2:]-b[:2]))
    overlap = float(np.prod(np.maximum(0., np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2]))))
    return dict(IoU=overlap/(area_a+area_b-overlap), intersection_area=overlap,
                selected_box_area=area_a, source_box_area=area_b,
                intersection_over_selected=overlap/area_a,
                intersection_over_source=overlap/area_b,
                selected_over_source_width=float((a[2]-a[0])/(b[2]-b[0])),
                selected_over_source_height=float((a[3]-a[1])/(b[3]-b[1])),
                selected_over_source_area=area_a/area_b)


def contains(points, box):
    points = np.asarray(points, np.float64)
    assert points.ndim == 2 and points.shape[1] == 2
    if box is None:
        return [None]*len(points)
    return [bool(np.isfinite(q).all() and box[0] <= q[0] < box[2] and box[1] <= q[1] < box[3]) for q in points]


def envelope(points):
    a = np.asarray(points, np.float64)
    good = np.isfinite(a).all(1)
    if not good.any():
        return None
    return xyxy(np.r_[a[good].min(0), a[good].max(0)])


def all_instance_diagnostics(candidate_boxes, selected_index, q8, annotations):
    # Original annotation/candidate order is preserved. There is no argmax,
    # threshold, selected annotation, label transfer or query permutation here.
    boxes = [xyxy(b) for b in candidate_boxes]
    selected = None if selected_index is None else boxes[selected_index]
    out = []
    for ann in annotations:
        box = xyxy(ann['bbox'], xywh=True)
        out.append(dict(annotation_id=ann['id'], official_bbox_xywh=ann['bbox'],
                        official_bbox_xyxy=box.tolist(),
                        selected_bbox_diagnostics=boxfacts(selected, box),
                        all_candidate_bbox_diagnostics=[boxfacts(b, box) for b in boxes],
                        original_q8_inside_official_bbox=contains(q8, box)))
    return out


def exact_numeric(a, b):
    aa, bb = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return aa.shape == bb.shape and aa.tobytes() == bb.tobytes()


def exact_value(a, b):
    if isinstance(a, dict) or isinstance(b, dict):
        return isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys() and all(exact_value(a[k], b[k]) for k in a)
    if isinstance(a, list) or isinstance(b, list):
        return isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and all(exact_value(x,y) for x,y in zip(a,b))
    if isinstance(a, (int,float)) and not isinstance(a,bool) and isinstance(b,(int,float)) and not isinstance(b,bool):
        return exact_numeric(a,b)
    return type(a) is type(b) and a == b


def generated():
    b = xyxy([0,0,4,4]); c = xyxy([1,1,4,4])
    f = boxfacts(b,c)
    assert f['IoU'] == 9/16 and boxfacts(b,xyxy([2,0,6,4]))['IoU'] == 1/3
    q = np.array([[-2.,1.],[0.,0.],[4.,2.],[2.,2.]])
    anns = [dict(id=9,bbox=[0,0,4,4]), dict(id=2,bbox=[20,20,2,3])]
    d = all_instance_diagnostics([[0,0,4,4]],0,q,anns)
    assert [a['annotation_id'] for a in d] == [9,2]
    assert d[0]['original_q8_inside_official_bbox'] == [False,True,False,True]
    null = all_instance_diagnostics([],None,q,anns)
    assert len(null)==2 and all(a['selected_bbox_diagnostics'] is None for a in null)
    assert np.array_equal(q, [[-2.,1.],[0.,0.],[4.,2.],[2.,2.]])
    assert not exact_numeric([0.],[-0.]) and exact_numeric([.1],[.1])
    assert not exact_value([{'x':1.}], [{'x':2.}])
    assert envelope(np.full((8,2),np.nan)) is None
    assert xyxy([1,1,1,2]) is None
    write(GENERATED, dict(completed=True,PASS=True,invented_only=True,code=bind(__file__),
        exact_IoU_and_halfopen_bounds=True,all_annotation_order_and_zeros_retained=True,
        no_detection_keeps_all_annotation_records=True,OOB_query_not_clipped=True,
        no_instance_argmax_or_label_transfer=True,byte_sensitive_saved_numeric_comparison=True,
        actual_RGB_q_annotation_weight_or_model_reads=0))


def prepare():
    assert sha(POLICY) == POLICY_SHA
    policy = read(POLICY)
    readiness = read(check(policy['readiness']))  # metadata only, no predictions
    generated()
    p = dict(completed=True,prepared_only=True,code=bind(__file__),policy=bind(POLICY),
        generated=bind(GENERATED),inputs=bind(INPUTS),source_inputs=policy['source_inputs_fixed_before_new_initialization'],
        source_semantic_proposals=policy['source_semantic_proposals_frozen'],readiness=policy['readiness'],
        expected_prediction_seal=str(PRED),expected_frames=16,expected_queries=128,
        expected_all_LOCO_official_instances=424,expected_named_Pallox_slots=32,
        fixed_LOCO_proposal_target=dict(all_COCO_image_id=34786,sub2_TRAIN_image_id=14793,target_annotation_id=371955),
        legacy2_expected_external_review=str(V/'independent_saved_input_review/REVIEW.json'),output=str(OUT),
        source_body_reads_during_prepare=0,
        labels='All128 actual query semantic states remain U. Existing source proposal states retain original instance/native role only; no automatic match, label copy or native permutation.',
        boxes='Selected detection is original firstargmax. Full official424 box and all candidate IoUs in original order; no threshold/ranking/instance reselection.',
        Pallox='Same-slot q-reference deltas and uncut virtual8 envelope diagnostics only. Inspect whole high crate vs flat base association visually; fullcrate upperrim D never automatically maps to base-corner D.',
        RGB='Read only canonical transport uint8 RGB after completed initializer. Raw RGB unchanged; overlay on separate copy; no GT-centered crop/model/encoder.',
        legacy='Separate initializer-agent saved-only review owns original V94 two-frame exact comparison. Association producer never opens historic proposal/query bodies.',
        prohibited={k:True for k in FORBIDDEN})
    write(PROTOCOL,p)
    template = dict(approved_saved_public_association=False,code=p['code'],policy=p['policy'],protocol=bind(PROTOCOL),generated=p['generated'],inputs=p['inputs'],
                    prediction_seal=dict(path=str(PRED),sha256='FILL_COMPLETED_NEW_PREDICTION_SEAL_SHA256'),
                    legacy_input_review=dict(path=str(V/'independent_saved_input_review/REVIEW.json'),sha256='FILL_COMPLETED_EXTERNAL_REVIEW_SHA256'),**{k:False for k in FORBIDDEN})
    path=H/'ROOT_SAVED_ASSOCIATION_ENTRY_TEMPLATE.json';write(path,template)
    print(json.dumps(dict(prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),generated=bind(GENERATED),template=bind(path),actual_q_RGB_annotation_reads=0)))


def source_proposals(policy, inputs, loco_manifest, named):
    loco = read(check(policy['source_semantic_proposals_frozen']['LOCO']))
    identity = loco['source_image_identity']
    assert all(str(x) in identity for x in (34786,14793,371955))
    proposals=loco['adjudicated_proposals'];assert len(proposals)==2
    assert len({r['native_slot'] for r in proposals})==2
    target_frames=[r for r in loco_manifest if r['all_COCO_image_id']==34786 and r['official_sub2_train_image_id']==14793]
    assert len(target_frames)==1
    target=target_frames[0];assert any(a['id']==371955 for a in target['official_pallet_instances'])
    byframe={r['id']:{} for r in inputs}
    frame=inputs[target['selection_index']];assert frame['source']=='LOCO' and frame['source_id']==target['source_id']
    for r in proposals:
        k=r['native_slot'];assert k in range(8) and r['semantic_state']=='D'
        xy=np.asarray(r['xy'],np.float64);assert xy.shape==(2,)and np.isfinite(xy).all()
        byframe[frame['id']][k]=dict(source_reference_xy=xy.tolist(),source_reference_state=r['semantic_state'],
            source_target_annotation_id=371955,uncertainty=r.get('uncertainty'),
            correspondence='FROZEN_SOURCE_NATIVE_ROLE_ONLY; actual detection entity not certified',
            human_gold=False,trainready=False)
    pallox=read(check(policy['source_semantic_proposals_frozen']['Pallox']))
    rr=pallox['rows'];assert len(rr)==32
    refs={(r['source_frame_id'],r['native_slot']):r for r in named['rows']};assert len(refs)==32
    sourceframes={r['source_id']:r for r in inputs if r['source']=='PALLOX'}
    assert len(sourceframes)==4
    for r in rr:
        key=(r['source_frame_id'],r['native_slot']);ref=refs[key]
        assert exact_value(r['frozen_reference_xy'],ref['reference_xy']) and r['continuous_inframe']==ref['continuous_inframe']
        assert r['root_review_state'] in ('D','E','S','U') and r['human_gold'] is False and r['trainready'] is False
        if not r['continuous_inframe']:assert r['root_review_state']=='U'
        byframe[sourceframes[r['source_frame_id']]['id']][r['native_slot']]=dict(
            source_reference_xy=ref['reference_xy'],source_reference_state=r['root_review_state'],
            proposal_state=r['proposal_state'],continuous_inframe=r['continuous_inframe'],
            native_name=r['native_name'],source_role_status=r['source_role_status'],
            source_reason_KO=r['root_review_reason_KO'],source_target_annotation_id=None,
            human_gold=False,trainready=False,correspondence='PROVISIONAL_SOURCE_HIGH_CRATE_VIRTUAL_ROLE; actual detection entity/extent not certified')
    return byframe


def font(size):
    from PIL import ImageFont
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)


def panel(rgb,row,framefacts,source_refs,path):
    from PIL import Image,ImageDraw
    raw=Image.fromarray(rgb,'RGB');overlay=raw.copy();d=ImageDraw.Draw(overlay)
    hh,ww=rgb.shape[:2];line=max(2,round(ww/640));fs=max(14,round(ww/80));ft=font(fs)
    for ann in framefacts.get('official_instance_diagnostics',[]):
        x0,y0,x1,y1=ann['official_bbox_xyxy'];d.rectangle((x0,y0,x1,y1),outline=(55,180,75),width=max(1,line//2))
        d.text((max(0,min(x0,ww-100)),max(0,min(y0,hh-fs))),str(ann['annotation_id']),font=ft,fill=(55,230,75),stroke_width=1,stroke_fill='black')
    box=framefacts['selected_bbox_xyxy']
    if box is not None:d.rectangle(box,outline=(255,160,10),width=line*2)
    for q in framefacts['queries']:
        k=q['native_slot'];point=q['query_xy']
        if point is not None and all(v is not None and math.isfinite(v)for v in point)and 0<=point[0]<ww and 0<=point[1]<hh:
            x,y=point;rad=line*2;d.ellipse((x-rad,y-rad,x+rad,y+rad),outline=(255,15,220),width=line)
            d.text((max(0,min(x+rad,ww-90)),max(0,min(y,hh-fs))),f'Q{k}',font=ft,fill=(255,15,220),stroke_width=1,stroke_fill='black')
    for k,r in source_refs.items():
        point=r['source_reference_xy']
        if point is None or any(v is None or not math.isfinite(v)for v in point):continue
        x,y=point
        if 0<=x<ww and 0<=y<hh:
            rad=line*3;d.line((x-rad,y,x+rad,y),fill=(0,235,255),width=line);d.line((x,y-rad,x,y+rad),fill=(0,235,255),width=line)
            d.text((max(0,min(x+rad,ww-110)),max(0,min(y-fs,hh-fs))),f'R{k}:{r["source_reference_state"]}',font=ft,fill=(0,235,255),stroke_width=1,stroke_fill='black')
    # Overlay is drawn at native resolution on its own image; no marks can
    # spill into the separate unchanged raw pane when q/reference is OOB.
    width=960;scale=width/ww;height=round(hh*scale)
    canvas=Image.new('RGB',(width*2,160+height),'white');cd=ImageDraw.Draw(canvas)
    cd.text((12,8),row['id']+' | public source only | original firstargmax kept',font=font(22),fill='black')
    cd.text((12,42),'Left raw RGB; right overlay: orange selectedbox, magenta Q0..7, cyan frozen source R/state, green official boxes.',font=font(19),fill='black')
    cd.text((12,72),'All actual query semantic labels remain U. Source refs/labels are NOT copied to a selected different entity or flat base.',font=font(19),fill='black')
    invalid=[q['native_slot']for q in framefacts['queries']if not q['native_continuous_valid']]
    cd.text((12,102),f'Original candidates={framefacts["candidate_count"]}; selected={framefacts["selected_index"]}; input invalid/OOB slots={invalid}; entity/extent review=PENDING',font=font(19),fill='black')
    canvas.paste(raw.resize((width,height),Image.Resampling.LANCZOS),(0,150))
    canvas.paste(overlay.resize((width,height),Image.Resampling.LANCZOS),(width,150))
    canvas.save(path)


def run(entry):
    e=read(entry);assert e['approved_saved_public_association']is True
    assert all(e[k]is False for k in FORBIDDEN)
    assert sha(POLICY)==POLICY_SHA and e['code']==bind(__file__) and e['protocol']==bind(PROTOCOL)
    p=read(PROTOCOL)
    for k in ('code','policy','generated','inputs'):assert e[k]==p[k];check(p[k])
    assert read(GENERATED)['PASS'] and not OUT.exists()
    assert Path(e['prediction_seal']['path'])==PRED
    seal=read(check(e['prediction_seal']));assert seal['completed']and seal['PASS']and seal['frames']==16 and seal['rows']==128
    assert seal['inputs']==p['inputs']and seal['GT_read']is False and seal['historical_predictions_read']is False and seal['actual_original_initializer']is True
    outputs={Path(b['path']).name:b for b in seal['outputs']}
    names=['FROZEN_PROPOSALS_ALL16_INPUT_ONLY.jsonl','ORIGINAL_NATIVE_QUERY128_INPUT_ONLY.jsonl','ORIGINAL_NATIVE_QUERY128_AND_VALID9X9_INPUT_ONLY.npz','RESULT.json']
    assert all(n in outputs and Path(outputs[n]['path']).parent==PRED.parent for n in names)
    OUT.mkdir()
    write(OUT/'INPUT_BINDINGS_BEFORE_SOURCE_BODY_READS.json',dict(completed=True,entry=bind(entry),protocol=bind(PROTOCOL),prediction_seal=e['prediction_seal'],source_inputs=p['source_inputs'],source_proposals=p['source_semantic_proposals'],RGB_q_annotation_body_reads_so_far=0))
    inputs=lines(check(p['inputs']));assert len(inputs)==16
    proposals=lines(check(outputs[names[0]]));queries=lines(check(outputs[names[1]]));result=read(check(outputs['RESULT.json']))
    assert len(proposals)==16 and len(queries)==128 and result['completed']and result['PASS']
    with np.load(check(outputs[names[2]]),allow_pickle=False)as packet:qa={k:packet[k].copy()for k in packet.files}
    assert np.array_equal(qa['global_index'],np.arange(128))and np.array_equal(qa['frame_index'],np.repeat(np.arange(16),8))
    for i,(a,b)in enumerate(zip(inputs,proposals)):
        assert a['frame_index']==i
        for k,v in a.items():assert b[k]==v
        c=b['candidates'];selected=None if not c else int(np.argmax([v['score']for v in c]));assert b['selected_index']==selected
    for j,r in enumerate(queries):
        i,k=divmod(j,8);assert r['frame_index']==i and r['native_slot']==k and r['frame_id']==inputs[i]['id']and r['query_id']==inputs[i]['id']+':native'+str(k)
        assert r['query_xy']==clean(qa['query_xy'][j])and r['L']==clean(qa['L'][j])and r['no_detection']==bool(qa['no_detection'][j])
        for a,b in [('finite_non_sentinel','finite_non_sentinel'),('native_continuous_valid','continuous_valid'),('query_nearest_pixel_valid','nearest_valid'),('query_RGB96_centre_domain_valid','RGBcentre_valid')]:assert r[a]==bool(qa[b][j])
        assert r['native_grid_any_valid']==bool(qa['native_grid_valid'][j].any())
    policy=read(POLICY)
    loco=read(check(p['source_inputs']['LOCO12']));lrows=loco['rows'];assert len(lrows)==12 and sum(len(r['official_pallet_instances'])for r in lrows)==424
    named=read(check(p['source_inputs']['Pallox4']));assert named['completed']and named['PASS']and len(named['rows'])==32 and len(named['frames'])==4
    assert all(r['semantic_state']=='U'for r in named['rows'])
    source_refs=source_proposals(policy,inputs,lrows,named)
    transport=read(check(result['canonical_transport_seal']));assert transport['completed']and transport['PASS']and transport['frames']==16
    manifest=read(check(transport['manifest']));tr=manifest['rows'];assert len(tr)==16
    frames=[];allrows=[];figures=[]
    for i,(row,prop)in enumerate(zip(inputs,proposals)):
        q8=qa['query_xy'][i*8:(i+1)*8];c=prop['candidates'];selected=prop['selected_index']
        boxes=[v['box_xyxy']for v in c];sbox=None if selected is None else xyxy(boxes[selected])
        ff=dict(frame_index=i,id=row['id'],source=row['source'],source_id=row['source_id'],split=row['split'],raw_hw=row['raw_hw'],
                candidate_count=len(c),selected_index=selected,selected_bbox_xyxy=None if sbox is None else sbox.tolist(),
                selected_candidate_unchanged=None if selected is None else c[selected],all_candidate_indices_retained=list(range(len(c))),
                input_q8_envelope_xyxy=clean(envelope(q8)),original_input_L=clean(qa['L'][i*8:(i+1)*8]),
                entity_association='PENDING_VISUAL_REVIEW_NO_AUTOMATIC_INSTANCE_SELECTION',
                extent_role_association='PENDING_WHOLE_CRATE_VS_FLAT_BASE_REVIEW'if row['source']=='PALLOX'else'PENDING_NATIVE_ROLE_AND_ENTITY_REVIEW',queries=[])
        if row['source']=='LOCO':
            lr=lrows[i];assert lr['selection_index']==i and lr['source_id']==row['source_id']
            ff['all_COCO_image_id']=lr['all_COCO_image_id'];ff['official_sub2_train_image_id']=lr['official_sub2_train_image_id']
            ff['official_instance_diagnostics']=all_instance_diagnostics(boxes,selected,q8,lr['official_pallet_instances'])
            ff['frozen_proposal_target_annotation_id']=371955 if lr['all_COCO_image_id']==34786 else None
        else:
            rr=sorted([r for r in named['rows']if r['source_frame_id']==row['source_id']],key=lambda x:x['native_slot']);assert [r['native_slot']for r in rr]==list(range(8))
            refxy=np.asarray([[np.nan if v is None else v for v in r['reference_xy']]for r in rr],np.float64)
            env=envelope(refxy)
            ff['Pallox_high_crate_virtual8_envelope_unclipped_xyxy']=clean(env)
            ff['selected_box_vs_virtual8_envelope_diagnostic']=boxfacts(sbox,env)
            ff['input_q8_envelope_vs_virtual8_envelope_diagnostic']=boxfacts(envelope(q8),env)
            ff['original_named_roles']=[{k:r[k]for k in ['native_slot','native_name','reference_xy','camera_Z_m','continuous_inframe','role_status']}for r in rr]
            ff['provisional_registration_scope']=named['role_scope']
        for k in range(8):
            j=i*8+k;qr=dict(queries[j]);ref=source_refs[row['id']].get(k)
            qnp=q8[k];rxy=None if ref is None else ref['source_reference_xy']
            delta=None;distance=None
            if rxy is not None and all(v is not None for v in rxy)and np.isfinite(qnp).all()and bool(qa['finite_non_sentinel'][j]):
                delta=(qnp-np.asarray(rxy)).tolist();distance=float(np.linalg.norm(delta))
            qr.update(global_index=j,actual_query_semantic_state='U',association_status='UNRESOLVED_NO_LABEL_TRANSFER',
                frozen_source_proposal=ref,same_native_slot_delta_query_minus_source_reference=delta,
                same_native_slot_distance_px_diagnostic=distance,
                distance_scope='NOT a certified keypoint error until same entity, full-object extent and native role are adjudicated',
                human_gold=False,trainready=False)
            ff['queries'].append(qr);allrows.append(qr)
        a=tr[i]
        for k in ('frame_index','id','source','source_id','split','raw_hw'):assert a[k]==row[k]
        rgb=np.load(check(a['canonical_RGB_NPY']),allow_pickle=False)
        assert rgb.dtype==np.uint8 and list(rgb.shape)==row['raw_hw']+[3]and hashlib.sha256(rgb.tobytes()).hexdigest()==row['decoded_RGB_sha256']==a['decoded_RGB_sha256']
        fig=OUT/f'FRAME_{i:02d}_RAW_AND_SAVED_ASSOCIATION.png';panel(rgb,row,ff,source_refs[row['id']],fig)
        ff['figure']=bind(fig);figures.append(ff['figure']);frames.append(ff)
    assert len(allrows)==128 and sum(len(f.get('official_instance_diagnostics',[]))for f in frames)==424
    assert all(r['actual_query_semantic_state']=='U'for r in allrows)
    assert Path(e['legacy_input_review']['path'])==Path(p['legacy2_expected_external_review'])
    check(e['legacy_input_review'])  # External audit binding only; no historic body reads.
    ledger=OUT/'ALL128_ORIGINAL_QUERIES_AND_UNTRANSFERRED_SOURCE_REFERENCES.jsonl'
    with ledger.open('x')as f:
        for r in allrows:f.write(json.dumps(clean(r),ensure_ascii=False,allow_nan=False)+'\n')
    summary=dict(completed=True,PASS=True,source_only=True,frames=16,original_queries=128,original_query_semantic_counts=dict(U=128),
        LOCO_official_instances=424,LOCO_source_proposals=2,LOCO_source_proposal_target_annotation_id=371955,
        Pallox_frames=4,Pallox_named_reference_slots=32,no_detection_frames=result['no_detection_frames'],
        original_input_validity={k:result[k]for k in ['finite_non_sentinel_slots','continuous_valid_slots','nearest_valid_slots','finite_OOB_slots']},
        frame_diagnostics=frames,legacy2_external_input_review=e['legacy_input_review'],historical_proposal_query_body_reads=0,
        source_proposal_labels_copied_to_actual_queries=0,instance_or_detection_reselection=0,GT_permutation=0,
        caveat_KO='IoU와 같은 슬롯 좌표 차이는 대상·구조 범위·역할의 일치를 자동 확정하지 않습니다. 큰 상자 윗테두리 보임 라벨을 바닥 팔레트의 점에 복사하지 않았습니다. 실제128 쿼리 라벨은 U이며 수동 확인이 필요합니다.',
        training_ready=False,human_gold=False,model_head_fit_private_V112_PnP_calls_or_reads=0)
    report=OUT/'SOURCE_ASSOCIATION_DIAGNOSTICS.json';write(report,clean(summary))
    parts=['<!doctype html><meta charset="utf-8"><title>Public saved initializer association</title><style>body{font:18px sans-serif;margin:24px;background:#f7f8fa}img{max-width:100%}</style>',
           '<h1>공개 실사 초기 추론: 대상과 구조 범위 확인</h1><p>저장된 첫 argmax 추론16장·128점을 그대로 표시했습니다. 모든 공식 박스 IoU를 남겼고 대상을 다시 선택하지 않았습니다. 기존 source 라벨은 원래 대상과 역할에 붙어 있으며 실제 쿼리128개의 의미 라벨은 U입니다. 큰 상자 전체와 바닥 팔레트의 구조 범위를 별도로 확인해야 합니다.</p>']
    for f in frames:parts.append('<h2>'+html.escape(f['id'])+'</h2><img src="'+html.escape(Path(f['figure']['path']).name)+'">')
    index=OUT/'INDEX.html';index.write_text('\n'.join(parts)+'\n')
    sealout=OUT/'ASSOCIATION_COMPLETE.json';write(sealout,dict(completed=True,PASS=True,code=bind(__file__),policy=e['policy'],protocol=bind(PROTOCOL),entry=bind(entry),prediction_seal=e['prediction_seal'],
        inputs=p['inputs'],source_inputs=p['source_inputs'],source_proposals=p['source_semantic_proposals'],results=bind(report),queries=bind(ledger),legacy2_external_input_review=e['legacy_input_review'],
        figures=figures,index=bind(index),all16_and128_preserved=True,all424_official_instances_preserved=True,semantic_query_U128=True,
        actual_query_or_label_mutations=0,automatic_association_or_GT_permutation=0,model_head_fit_private_V112_PnP=0))
    print(json.dumps(dict(completed=True,PASS=True,seal=bind(sealout),results=bind(report))))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);args=ap.parse_args()
    if args.prepare and args.entry is None:prepare()
    elif args.entry and not args.prepare:run(args.entry)
    else:raise SystemExit('Use inert --prepare or explicit root --entry after completed source initializer.')
