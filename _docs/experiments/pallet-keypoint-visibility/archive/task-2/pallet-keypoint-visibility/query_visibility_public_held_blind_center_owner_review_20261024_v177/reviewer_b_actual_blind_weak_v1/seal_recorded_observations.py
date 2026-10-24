"""Seal only B's manually recorded tool-viewed observations; no model/GT/image decoding."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
D=Path(__file__).resolve().parent; V=D.parent

def bind(p):
    p=Path(p);return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

def put(p,obj):
    with p.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
started=json.loads((D/'STARTED.json').read_text())
for k in ['definition','target_semantics_review','evidence_complete','input_deck','input_rows']:
    b=started[k];assert bind(b['path'])['sha256']==b['sha256'],k
inputs=[json.loads(s) for s in Path(started['input_rows']['path']).read_text().splitlines()]
by={r['original_global_index']:r for r in inputs}
batches=sorted(D.glob('BATCH_*_OBSERVATIONS.jsonl'))
assert len(batches)==9
rows=sorted([json.loads(s) for p in batches for s in p.read_text().splitlines()],key=lambda r:r['original_global_index'])
assert [r['original_global_index'] for r in rows]==list(range(584,944))
views=sorted([r for p in sorted(D.glob('BATCH_*_ACTUAL_VIEWS.json')) for r in json.loads(p.read_text())['actually_viewed_sheets']],key=lambda r:r['frame_index'])
assert [v['frame_index'] for v in views]==list(range(73,118))
assert len({v['path'] for v in views})==45
vb={v['frame_index']:v for v in views}
for v in views:
    assert v['actual_tool_view']=='tools.view_image' and v['all8slots_actually_viewed'] is True
    assert bind(v['path'])['sha256']==v['sha256']
for r in rows:
    q=by[r['original_global_index']];v=vb[r['frame_index']]
    assert (r['frame_index'],r['native_slot'],r['query_id'])==(q['frame_index'],q['native_slot'],q['query_id'])
    assert r['sheet_path']==v['path'] and r['sheet_sha256']==v['sha256']
    assert r['original_query_available']==q['query_available'] and r['center4_valid_bits']==q['center4_valid_bits']
    assert r['actually_viewed'] is True and r['training_admission'] is False
    assert r['center_owner'] in ['TARGET_VISIBLE_SUPPORTED','FOREIGN_VISIBLE_SUPPORTED','BACKGROUND_SUPPORTED','UNKNOWN']
    assert r['crop_presence'] in ['YES','NO','UNKNOWN']
    assert all(isinstance(r[k],str) and r[k] for k in ['center_reason','crop_reason','observation'])
    if q['forced_center_unknown_reason']:
        assert r['center_owner']=='UNKNOWN' and r['center_reason']==q['forced_center_unknown_reason']
    if not q['feature_constructed']:
        assert r['crop_presence']=='UNKNOWN' and r['crop_reason']=='INPUT_U'
assert sum(not q['query_available'] for q in inputs)==108
assert sum(q['forced_center_unknown_reason']=='CENTER4_INVALID' for q in inputs)==4
assert sum(not q['feature_constructed'] for q in inputs)==24
assert Counter(r['frame_index'] for r in rows)==Counter({i:8 for i in range(73,118)})
assert Counter(r['native_slot'] for r in rows)==Counter({i:45 for i in range(8)})
op=D/'ALL360_OBSERVATIONS.jsonl'
with op.open('x') as f:
    for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
vp=D/'ACTUAL_VIEWED45_SHEETS.json'
put(vp,{'schema':'V177_B_ACTUALLY_TOOL_VIEWED_SHEETS_v1','frames':45,'all_360_slots_actually_viewed':True,'actual_viewed_sheets':views,'scope':'Full sheets displayed using tools.view_image; no browser rendering claim.'})
seal={'schema':'V177_B_BLIND_WEAK_REVIEW_SEAL_v1','UTC':datetime.now(timezone.utc).isoformat(),'completed':True,'PASS':True,'rows':360,'frames_actually_viewed':45,'original_global_index_range_inclusive':[584,943],'frame_index_range_inclusive':[73,117],'observations':bind(op),'actual_viewed_sheets':views,'actual_view_manifest':bind(vp),'inputs':{k:started[k] for k in ['definition','target_semantics_review','evidence_complete','input_deck','input_rows']},'recorded_batches':[bind(p) for p in batches],'manual_observation_serializer':bind(D/'save_manual_batch.py'),'seal_script':bind(Path(__file__)),'integrity_checks':{'exact_once_all360_original_IDs':True,'all45_original_frames_and_native8':True,'all_sheet_SHA_bindings_unchanged':True,'all_metadata_ID_availability_center_bits_match':True,'original_inputU108_center_UNKNOWN':True,'additional_center4_invalid4_center_UNKNOWN':True,'unconstructed24_crop_UNKNOWN':True,'no_rows_filtered_or_replaced':True},'center_counts':dict(Counter(r['center_owner'] for r in rows)),'crop_presence_counts':dict(Counter(r['crop_presence'] for r in rows)),'no_model_output_or_GT_or_other_reviewer_read':True,'blinding_scope':'During this V177 preparation and actual review: no new model logits/predictions, GT pose/mask/native labels, A/root/other reviewer observations were read. Historical source/native evaluations had prior exposure as predeclared by the frozen definition.','labels_or_findings_shared_before_seal':False,'observation_authority':'Independent assistant visual weak observations only; not human/pixel/physical-native/coordinate gold. Official nominal Euro pallet/local obj1 crosswalk and per-query physical instance identity are not certified; unclear center/crop relations remain UNKNOWN.','center_and_crop_presence_distinct':True,'q8_box_is_not_GT_instance_authority':True,'inputU_and_padding_never_background_evidence':True,'all_original_q_L_native_and_labels_unchanged':True,'training_admission':False,'new_fit_calibration_model_selection_or_definition_changes':False,'private_current223_V112_access':False,'HTML_browser_rendered':False}
sp=D/'REVIEW_SEAL.json';put(sp,seal)
print(json.dumps({'completed':True,'rows':360,'frames_actually_viewed':45,'observations':bind(op),'actual_view_manifest':bind(vp),'seal':bind(sp)},ensure_ascii=False))
