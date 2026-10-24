"""V177 fixed saved-only weak center-owner evaluation; prepare uses invented data only."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
V = HERE.parent
P = V.parent
SCOPE = V / 'ROOT_SCOPE_BEFORE_PUBLIC_HELD_PIXELS_OR_NEW_OWNER_ANNOTATIONS.json'
DEFINITION = V / 'ROOT_WEAK_LABEL_DEFINITION_BEFORE_HELD_PIXELS.json'
INPUTS = V / 'metadata_readiness_c/ALL360_HELD_INPUT_ONLY.jsonl'
RENDER = V / 'blind_review_evidence_v1/ALL360_BLINDED_REVIEW_INPUTS.jsonl'
V61_COMPLETE = V / 'v61_public_held_baseline_v1/COMPLETE.json'
PROTOCOL = HERE / 'PROTOCOL_BEFORE_HELD_OBSERVATION_OR_MODEL_SCORE_BODIES.json'
CHECKS = HERE / 'GENERATED_INVENTED_CHECKS.json'
PREP = HERE / 'PREPARATION_COMPLETE.json'
OUT = V / 'saved_public_held_owner_evaluation_v1'
EXPECTED = {
    str(SCOPE): 'c2018965ace4b5cbe4d4ef9a0ddc816d10514c73f4db3a0ff13c9f8a76ca9c70',
    str(DEFINITION): 'fa9c00935a4c93d1af091eccba5aacda30bc98be8dcfaf205148eb2a3484df97',
    str(INPUTS): '9c962418c6ba63589e9aaf8a64417bc26924ae4c8a6f448c7ad75ff00fba605f',
    str(RENDER): 'd36712ced46a59d250d06421701178a412b12c8042b07f17e092a62fad576454',
    str(V61_COMPLETE): '37f119f24fbc5832ee3f5407ad12abe27461d11356a211ca4ca1efbed300fec2',
}
OWNER = ['TARGET_VISIBLE_SUPPORTED', 'FOREIGN_VISIBLE_SUPPORTED', 'BACKGROUND_SUPPORTED']
UNKNOWN = 'UNKNOWN'
CROP = ['YES', 'NO']
ARMS = ['C47_PUBLIC_WEAK253_ONLY', 'C48_PUBLIC_WEAK253_PLUS_LITERAL_SYNTHETIC',
        'C49_LITERAL_SYNTHETIC_PLUS_PUBLIC_YCB_CF314']
TAGS = ['C47', 'C48', 'C49', 'v61']
ROW_KEYS = ('original_global_index', 'frame_index', 'native_slot', 'center_owner', 'center_reason',
            'crop_presence', 'crop_reason', 'observation', 'sheet_path')


def require(value, message):
    if not value:
        raise AssertionError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def bind(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def check(binding):
    require(sha(binding['path']) == binding['sha256'], 'SHA binding mismatch')
    return Path(binding['path'])


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]


def meta(path):
    require(Path(path).suffix in ('.json', '.jsonl', '.py'), 'Metadata-only path required')
    if str(path) in EXPECTED:
        require(sha(path) == EXPECTED[str(path)], 'Frozen metadata changed')
    return lines(path) if Path(path).suffix == '.jsonl' else read(path)


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def inputs():
    source, rendered = meta(INPUTS), meta(RENDER)
    require(len(source) == len(rendered) == 360, '360 original rows required')
    identity = ('original_global_index', 'original_feature_index', 'occurrence_unit', 'query_id',
                'frame_index', 'frame_id', 'source_id', 'split', 'native_slot', 'query_xy', 'L',
                'no_detection', 'original_query_available', 'native_continuous_valid', 'feature_constructed',
                'nearest_valid', 'local_central4_available', 'context_available', 'Q_inside')
    sheets = {}
    for j, (s, r) in enumerate(zip(source, rendered)):
        require(all(s[k] == r[k] for k in identity), 'Render/source identity drift')
        require(s['original_global_index'] == 584+j == s['original_feature_index'], 'Original feature index')
        require(s['split'] == 'SOURCE_VALIDATION' and s['source_id'].split(':')[0] == 'val/000001', 'Held split')
        require(s['original_global_index'] == s['frame_index']*8+s['native_slot'], 'Input native order')
        require(bool(np.asarray(r['center4_valid_bits'], bool).all()) == s['local_central4_available'], 'Center support metadata')
        force = not s['original_query_available'] or not s['local_central4_available']
        require(bool(r['forced_center_unknown_reason']) == force, 'Forced center U metadata')
        require(r['crop_presence_forced_unknown'] == (not s['feature_constructed']), 'Forced crop U metadata')
        sheet = r['frame_sheet']
        if s['frame_index'] in sheets:
            require(sheets[s['frame_index']] == sheet, 'Frame sheet mismatch')
        sheets[s['frame_index']] = sheet
    require(len(sheets) == 45 and sorted(sheets) == list(range(73,118)), 'All45 frame sheets')
    require(sum(not r['original_query_available'] for r in source) == 108, 'Original inputU108')
    require(sum(not r['local_central4_available'] for r in source) == 112, 'Center invalid112')
    require(sum(not r['feature_constructed'] for r in source) == 24, 'No constructed crop24')
    return source, rendered, sheets


def validate_seal(seal, sheets, n):
    require(seal['completed'] is True and seal['PASS'] is True, 'Reviewer incomplete')
    require(seal['rows'] == n and seal['frames_actually_viewed'] == len(sheets), 'Reviewer denominator')
    require(seal['no_model_output_or_GT_or_other_reviewer_read'] is True, 'Reviewer blinding attestation')
    actual = [(x['path'], x['sha256']) for x in seal['actual_viewed_sheets']]
    expected = [(x['path'], x['sha256']) for x in sheets.values()]
    require(len(actual) == len(set(actual)) == len(expected) and sorted(actual) == sorted(expected), 'Actualview sheet bindings')
    require(set(seal['observations']) >= {'path','sha256'}, 'Observation binding')


def normalize_observations(observations, source, rendered):
    require(len(observations) == len(source), 'Observation rows missing/extra')
    indexed = {}
    for row in observations:
        require(all(k in row for k in ROW_KEYS), 'Common observation fields required')
        require(all(type(row[k]) is int for k in ('original_global_index','frame_index','native_slot')), 'Integer identity schema')
        g = row['original_global_index']
        require(g not in indexed, 'Duplicate observation ID')
        require(row['center_owner'] in OWNER+[UNKNOWN] and row['crop_presence'] in CROP+[UNKNOWN], 'Exact label enums required')
        require(all(isinstance(row[k], str) for k in ('center_reason','crop_reason','observation','sheet_path'))
                and bool(row['sheet_path'].strip()), 'Evidence/reason string schema')
        indexed[g] = {k: row[k] for k in ROW_KEYS}
    require(set(indexed) == {s['original_global_index'] for s in source}, 'Observation ID census')
    normalized = []
    for s, r in zip(source, rendered):
        a = indexed[s['original_global_index']]
        require(a['frame_index'] == s['frame_index'] and a['native_slot'] == s['native_slot'], 'Observation input identity')
        require(a['sheet_path'] == r['frame_sheet']['path'], 'Observation actualview sheet')
        if not s['original_query_available'] or not s['local_central4_available']:
            require(a['center_owner'] == UNKNOWN, 'Invalid/inputU center must remain UNKNOWN')
        if not s['feature_constructed']:
            require(a['crop_presence'] == UNKNOWN, 'No crop must remain UNKNOWN')
        normalized.append(a)
    return normalized


def consensus_value(a, b, known):
    return a if a == b and a in known else UNKNOWN


def consensus(a, b, source):
    rows = []
    for aa, bb, s in zip(a,b,source):
        require(aa['original_global_index'] == bb['original_global_index'] == s['original_global_index'], 'Consensus identity')
        center = consensus_value(aa['center_owner'],bb['center_owner'],OWNER)
        crop = consensus_value(aa['crop_presence'],bb['crop_presence'],CROP)
        require((s['original_query_available'] and s['local_central4_available']) or center == UNKNOWN, 'Forced centerU consensus')
        require(s['feature_constructed'] or crop == UNKNOWN, 'Forced cropU consensus')
        rows.append(dict(original_global_index=s['original_global_index'],query_id=s['query_id'],frame_index=s['frame_index'],
                         native_slot=s['native_slot'],split=s['split'],query_xy=s['query_xy'],L=s['L'],
                         source_original_query_available=s['original_query_available'],
                         source_center4_allvalid=s['local_central4_available'],source_crop_constructed=s['feature_constructed'],
                         center_owner=center,crop_presence=crop,reviewer_A=aa,reviewer_B=bb,
                         training_admission=False,human_pixel_physical_native_gold=False))
    return rows


def owner_decode(raw, available):
    require(raw.dtype == np.float32 and raw.shape == (len(available),3), 'FP32 owner3 shape')
    require(np.isfinite(raw[available]).all() and np.isnan(raw[~available]).all(), 'Owner U/NaN policy')
    result = np.full(len(available),-1,np.int8)
    result[available] = np.argmax(raw[available],axis=1).astype(np.int8)
    return result


def v61_decode(raw, available):
    require(raw.dtype == np.float32 and raw.shape == available.shape and np.isfinite(raw[available]).all(), 'v61 FP32/schema')
    result = np.full(len(raw),-1,np.int8)
    result[available] = (raw[available] < 0).astype(np.int8)
    return result


def hard_metrics(truth, prediction, classes):
    k = len(classes)
    require(len(truth) == len(prediction), 'Metric census')
    require(np.isin(truth,list(range(k))+[-1]).all() and np.isin(prediction,list(range(k))+[-1]).all(), 'Class range')
    cm = np.zeros((k+1,k+1),np.int64)
    for t,p in zip(truth,prediction):
        cm[int(t) if t >= 0 else k,int(p) if p >= 0 else k] += 1
    details = {}
    for j,name in enumerate(classes):
        tp, predicted, support = int(cm[j,j]), int(cm[:k,j].sum()), int(cm[j].sum())
        details[name] = dict(TP=tp,predicted_on_known_truth=predicted,known_truth_support=support,
                             precision_on_known_truth=tp/predicted if predicted else None,
                             recall_including_prediction_U=tp/support if support else None)
    return dict(N=len(truth),truth_class_order=classes+[UNKNOWN],prediction_class_order=classes+['U'],
                full_confusion=cm.tolist(),known_truth_confusion=cm[:k].tolist(),
                known_truth=int(cm[:k].sum()),unknown_truth=int(cm[k].sum()),
                prediction_U_all=int(cm[:,k].sum()),prediction_U_on_known_truth=int(cm[:k,k].sum()),
                predictions_on_unknown_truth=cm[k].tolist(),per_class=details,
                precision_excludes_unknown_truth=True,weak_reference_only=True)


def binary_labels(owner):
    x = np.full(len(owner),-1,np.int8)
    x[owner == 0] = 0
    x[(owner == 1)|(owner == 2)] = 1
    return x


def cohorts(truth, predictions):
    paired = np.logical_and.reduce([predictions[t] >= 0 for t in TAGS])
    return dict(FULL360=np.ones(len(truth),bool),KNOWN_CENTER_TRUTH=truth >= 0,
                ALL4_MODEL_PAIRED_AVAILABLE=paired,KNOWN_TRUTH_AND_ALL4_PAIRED_AVAILABLE=(truth>=0)&paired)


def count_table(a,b,na,nb):
    out=np.zeros((na,nb),np.int64)
    for x,y in zip(a,b):
        out[int(x),int(y)] += 1
    return out.tolist()


def packet_metadata(packet, index, source, total, v61=False):
    def take(key):
        a = packet[key]
        require(a.shape[0] == total, 'Packet metadata row count')
        return a[index]
    require(np.array_equal(take('global_index'),[r['original_global_index'] for r in source]), 'Packet global ID')
    for key in ('query_id','frame_index','frame_id','source_id','native_slot','split','no_detection'):
        require(np.array_equal(take(key),np.asarray([r[key] for r in source])), 'Packet identity '+key)
    require(np.array_equal(take('occurrence_unit'),[r['occurrence_unit'] for r in source]),'Original occurrence index')
    require(np.array_equal(take('query_xy'),np.asarray([r['query_xy'] if r['query_xy'] is not None else [np.nan,np.nan] for r in source],np.float64),equal_nan=True), 'Original q changed')
    require(np.array_equal(take('L'),np.asarray([r['L'] for r in source],np.float64),equal_nan=True), 'Original L changed')
    for key,source_key in [('native_continuous_valid','native_continuous_valid'),('native_nearest_valid' if v61 else 'nearest_valid','nearest_valid')]:
        a=take(key); require(a.dtype==bool and np.array_equal(a,[r[source_key] for r in source]), 'Original input flag '+key)
    if v61:
        mappings={'source_original_query_available':'original_query_available','source_original_feature_constructed':'feature_constructed',
                  'source_original_RGBcentral4_available':'local_central4_available','source_original_context_available':'context_available',
                  'source_original_Q_inside':'Q_inside'}
    else:
        mappings={'query_available':'original_query_available','feature_constructed':'feature_constructed',
                  'local_central4_available':'local_central4_available','context_available':'context_available','unit_Q_inside':'Q_inside'}
    for key,source_key in mappings.items():
        a=take(key); require(a.dtype==bool and np.array_equal(a,[r[source_key] for r in source]), 'Original source flag '+key)


def load_owner(binding, source):
    # Shared1840 NPZ container is decompressed only for required keys; retain/analyze held360 only.
    with np.load(check(binding),allow_pickle=False) as z:
        require(np.array_equal(z['global_index'],np.arange(1840)), 'MR1840 original index order')
        ix=np.asarray([r['original_global_index'] for r in source],np.int64)
        packet_metadata(z,ix,source,1840)
        require(z['owner_class_order'].astype(str).tolist()==OWNER, 'Frozen owner class order')
        available=z['query_available'][ix].copy()
        raw_full=z['raw_owner_three_logits'];saved_full=z['owner_prediction']
        require(raw_full.shape==(1840,3) and saved_full.shape==(1840,),'Original MR1840 score shape')
        raw=raw_full[ix].copy();saved=saved_full[ix].copy()
        require(saved.dtype==np.int8 and np.array_equal(saved,owner_decode(raw,available)), 'Original argmax/U prediction parity')
    return saved


def load_v61(binding, source):
    with np.load(check(binding),allow_pickle=False) as z:
        ix=np.arange(360,dtype=np.int64)
        packet_metadata(z,ix,source,360,True)
        require(np.array_equal(z['evaluation_index'],ix), 'v61 local evaluation index')
        raw=z['raw_visible_logit']; available=z['query_available']; saved=z['prediction_D0_E1_Uminus1']
        require(raw.dtype==np.float32 and available.dtype==bool and saved.dtype==np.int8, 'v61 types')
        for key in ['head_forwarded','native_query_valid','ROI_any_valid','RGB_central4_available']:
            require(z[key].dtype==bool and z[key].shape==(360,), 'v61 input flag schema')
        require(np.array_equal(available,z['head_forwarded']&z['native_query_valid']&z['ROI_any_valid']), 'v61 original availability')
        require(np.array_equal(z['native_query_valid'],z['native_continuous_valid']&z['native_nearest_valid']), 'v61 native rule')
        require(np.array_equal(z['RGB_central4_available'],[r['local_central4_available'] for r in source]), 'v61 center4 recorded only')
        require(np.array_equal(z['raw_hidden_logit'],-raw,equal_nan=True), 'v61 raw sign preservation')
        require(np.isfinite(raw[z['head_forwarded']]).all() and np.isnan(raw[~z['head_forwarded']]).all(), 'v61 no-query raw policy')
        require(np.array_equal(saved,v61_decode(raw,available)), 'v61 threshold0/U parity')
        require(int(available.sum())==251,'Frozen v61 availability251')
    return saved.copy()


def model_bindings(scope):
    fit = []
    for b in scope['models_frozen_before_future_annotation']:
        x = read(check(b))
        require(x['completed'] and x['PASS'] and x['selected_model'] is None, 'Frozen source fit receipt')
        fit.append(x)
    require(fit[0]['all2_frozen_before_outputs'] is True and fit[1]['fixed_final_frozen_before_outputs'] is True, 'Frozen before outputs')
    result={}
    for tag,arm,x in [('C47',ARMS[0],fit[0]),('C48',ARMS[1],fit[0]),('C49',ARMS[2],fit[1])]:
        require(x['arms'][arm]['updates']==8000,'Fixed final source update')
        result[tag]=x['arms'][arm]['outputs']['MR1840']
    v61=meta(V61_COMPLETE)
    require(v61['completed'] and v61['PASS'] and v61['score_body_blinded_until_held_annotation_seals'],'v61 completed seal')
    result['v61']=v61['outputs']['predictions']
    return result


def generated_checks():
    passed=[]
    require(consensus_value(OWNER[0],OWNER[0],OWNER)==OWNER[0] and consensus_value(OWNER[0],OWNER[1],OWNER)==UNKNOWN,'Known exact consensus')
    require(consensus_value(UNKNOWN,UNKNOWN,OWNER)==UNKNOWN and consensus_value(UNKNOWN,OWNER[0],OWNER)==UNKNOWN,'U consensus')
    require(consensus_value('YES','YES',CROP)=='YES' and consensus_value('YES','NO',CROP)==UNKNOWN,'Independent crop consensus')
    passed.append('Exact_known_consensus_each_field_and_U_disagreement')
    source=[dict(original_global_index=1,frame_index=2,native_slot=0,original_query_available=True,local_central4_available=True,feature_constructed=True)]
    rendered=[dict(frame_sheet=dict(path='/invented/SHEET.png',sha256='x'))]
    row=dict(original_global_index=1,frame_index=2,native_slot=0,center_owner=OWNER[0],center_reason='CLEAR_TARGET',crop_presence='YES',crop_reason='CLEAR',observation='invented',sheet_path='/invented/SHEET.png')
    normalize_observations([row],source,rendered)
    rejected=0
    for changes,obs in [({'original_query_available':False},row),({'local_central4_available':False},row),
                        ({'feature_constructed':False},row)]:
        try: normalize_observations([obs],[dict(source[0],**changes)],rendered)
        except AssertionError: rejected+=1
    require(rejected==3,'Forced U assertions')
    passed.append('Original_inputU_center4invalid_and_nocrop_known_labels_rejected')
    rejected=0
    for obs in [[row,row],[dict(row,original_global_index=2)],[dict(row,native_slot=1)],[dict(row,sheet_path='/wrong')]]:
        try: normalize_observations(obs,source,rendered)
        except AssertionError: rejected+=1
    require(rejected==4,'Observation identity drift')
    passed.append('Missing_duplicate_ID_native_and_sheet_drift_fail_closed')
    sheet={2:dict(path='/invented/SHEET.png',sha256='x')}
    seal=dict(completed=True,PASS=True,rows=1,frames_actually_viewed=1,observations=dict(path='/invented/rows.jsonl',sha256='x'),actual_viewed_sheets=list(sheet.values()),no_model_output_or_GT_or_other_reviewer_read=True)
    validate_seal(seal,sheet,1)
    try: validate_seal(dict(seal,no_model_output_or_GT_or_other_reviewer_read=False),sheet,1)
    except AssertionError: pass
    else: raise AssertionError('Blinding attestation accepted false')
    passed.append('Reviewer_barrier_and_actual_view_binding_guard')
    raw=np.array([[0,0,0],[2,2,1],[np.nan,np.nan,np.nan]],np.float32)
    require(owner_decode(raw,np.array([True,True,False])).tolist()==[0,0,-1],'Argmax first tie/U')
    try: owner_decode(raw,np.array([True,True,True]))
    except AssertionError: pass
    else: raise AssertionError('NaN available accepted')
    passed.append('FP32_argmax_first_tie_and_U_first_NaN_policy')
    require(v61_decode(np.array([0.,-1.,np.nan],np.float32),np.array([True,True,False])).tolist()==[0,1,-1],'Original v61 sign')
    passed.append('Original_v61_zero_visible_sign_and_U_first')
    truth=np.array([0,1,2,-1,0,1],np.int8);pred=np.array([0,2,-1,1,-1,1],np.int8)
    m=hard_metrics(truth,pred,['T','F','BG'])
    require(m['full_confusion']==[[1,0,0,1],[0,1,1,0],[0,0,0,1],[0,1,0,0]],'Full4x4 CM')
    require(m['per_class']['T']['precision_on_known_truth']==1 and m['per_class']['T']['recall_including_prediction_U']==.5,'P/R/U denominators')
    require(m['per_class']['F']['precision_on_known_truth']==1 and m['unknown_truth']==1 and m['prediction_U_on_known_truth']==2,'Unknown excluded precision')
    passed.append('Independent_integer_confusion_P_R_unknowntruth_and_abstention_denominators')
    bm=hard_metrics(binary_labels(truth),binary_labels(pred),['T','NON_T'])
    require(bm['full_confusion']==[[1,0,1],[0,2,1],[0,1,0]],'Same truth binary collapse')
    passed.append('Threeway_to_same_truth_binary_collapse_preserves_U')
    ps={t:pred.copy() for t in TAGS};ps['v61'][1]=-1
    cc=cohorts(truth,ps)
    require(int(cc['FULL360'].sum())==6 and int(cc['KNOWN_CENTER_TRUTH'].sum())==5,'Full vs known')
    require(int(cc['ALL4_MODEL_PAIRED_AVAILABLE'].sum())==3 and int(cc['KNOWN_TRUTH_AND_ALL4_PAIRED_AVAILABLE'].sum())==2,'Paired scope')
    passed.append('Full_known_and_all4_paired_masks_distinct')
    cross=count_table([0,0,3],[0,1,2],4,3)
    require(cross==[[1,1,0],[0,0,0],[0,0,0],[0,0,1]],'Crop association only')
    passed.append('Center_owner_crop_presence_cross_table_not_classifier_substitution')
    return dict(completed=True,PASS=True,invented_only=True,checks=passed,check_count=len(passed),
                actual_reviewer_score_GT_pixel_feature_weight_body_or_hash_reads=0,model_forward_fit_threshold_selection=0)


def prepare():
    scope,definition=meta(SCOPE),meta(DEFINITION)
    require(definition['center_owner_values']==OWNER+[UNKNOWN] and definition['crop_presence_values']==CROP+[UNKNOWN],'Frozen definitions')
    source,rendered,sheets=inputs()
    scores=model_bindings(scope)  # Small COMPLETE metadata only; never checks/opens score packets.
    write(CHECKS,generated_checks())
    p=dict(completed=True,prepared_only=True,code=bind(__file__),scope=bind(SCOPE),definition=bind(DEFINITION),
           inputs=bind(INPUTS),render_inputs=bind(RENDER),model_fit_receipts=scope['models_frozen_before_future_annotation'],
           v61_complete=bind(V61_COMPLETE),score_bindings_inherited_not_opened_or_hashed=scores,checks=bind(CHECKS),
           expected_reviewer_seals={t:str(V/f'reviewer_{t.lower()}_actual_blind_weak_v1/REVIEW_SEAL.json') for t in ['A','B']},
           reviewer_row_schema={k:('int' if k in ROW_KEYS[:3] else 'str') for k in ROW_KEYS},
           reviewer_seal_schema='completed/PASS true,rows360,frames_actually_viewed45,observations{path,sha256},actual_viewed_sheets45{path,sha256},no_model_output_or_GT_or_other_reviewer_read true',
           strict_enum_normalization=True,center_owner_classes=OWNER,crop_presence_classes=CROP,
           consensus='Each semantic field separately: exact equal known enum from A/B only, else UNKNOWN. Reasons/observation strings preserved but not votes.',
           force_validation='Each reviewer centerUNKNOWN on originalinputU108 or center4invalid112; each reviewer cropUNKNOWN on no constructed crop24. Contradiction fails closed, never relabeled silently.',
           model_rule='C47/C48/C49 FP32 raw owner3 first-index argmax and saved U first, exact original input metadata; v61 original threshold0/own availability, no extra center gate',
           shared_MR1840_container='Only required input metadata and raw score keys decompressed; retain/analyze held global584..943 only. No TRAIN score/label analysis, no features or checkpoints.',
           cohorts=['FULL360','KNOWN_CENTER_TRUTH','ALL4_MODEL_PAIRED_AVAILABLE','KNOWN_TRUTH_AND_ALL4_PAIRED_AVAILABLE'],
           metrics='Hard weak-reference center3truth x4prediction(U) CM plus full4x4 inclunknowntruth; per-class P excludesunknowntruth, R includespredictionU. Same centertruth binaryT/nonT for all4 models.',
           crop_presence='Observation consensus/agreements and center/prediction association tables only; never ownerT-as-crop-presence precision/recall.',
           strata='ALL and each native0..7, all45frames one public val/000001 video; fixed original inputU108 vscenterinvalid112 vstruthU vsmodelabstention kept separate',
           no_label_model_data_threshold_selection=True,no_training_admission=True,selected_model=None,adopted=False,fully_solved=False,
           output=str(OUT),automatic_retry=False,actual_reviewer_or_score_body_read_during_prepare=False)
    write(PROTOCOL,p)
    write(PREP,dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),checks=bind(CHECKS),
                    actual_reviewer_or_score_GT_RGB_NPZ_feature_weight_body_or_hash_reads=0,forward_fit_threshold_calls=0))
    write(HERE/'ROOT_ACTUAL_SAVED_EVALUATION_ENTRY_TEMPLATE.json',dict(
        approved_actual_saved_public_held_owner_evaluation=False,source_only=True,both_reviewer_seals_complete_before_score_read=True,
        no_model_threshold_or_data_selection=True,no_private_current_V112_PnP_qL_or_training=True,
        code=bind(__file__),protocol=bind(PROTOCOL),preparation=bind(PREP),
        reviewer_seals={t:dict(path=p['expected_reviewer_seals'][t],sha256='ROOT_FILL_AFTER_SEAL') for t in ['A','B']},output=str(OUT)))
    print(json.dumps(dict(prepared_only=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),checks=bind(CHECKS),preparation=bind(PREP))))


def run(entrypath):
    entry=read(entrypath)
    require(entry['approved_actual_saved_public_held_owner_evaluation'] is True and entry['source_only'] is True,'Actual root entry required')
    require(entry['both_reviewer_seals_complete_before_score_read'] is True and entry['no_model_threshold_or_data_selection'] is True,'Review barrier/no selection')
    require(entry['no_private_current_V112_PnP_qL_or_training'] is True,'Source-only restrictions')
    require(entry['code']==bind(__file__) and entry['protocol']==bind(PROTOCOL) and entry['preparation']==bind(PREP),'Frozen code/protocol/prepare')
    require(entry['output']==str(OUT),'Fixed new output folder')
    p=read(PROTOCOL);require(read(PREP)['PASS'] and read(check(p['checks']))['PASS'],'Preparation passed')
    require(p['scope']==bind(SCOPE) and p['definition']==bind(DEFINITION),'Scope/definition unchanged')
    require(p['inputs']==bind(INPUTS) and p['render_inputs']==bind(RENDER),'Input metadata unchanged')
    source,rendered,sheets=inputs()
    # Complete both reviewer seals and their originalview bindings before opening any observations/scores.
    seals={t:read(check(entry['reviewer_seals'][t])) for t in ['A','B']}
    for t in ['A','B']:
        require(entry['reviewer_seals'][t]['path']==p['expected_reviewer_seals'][t],'Fixed reviewer seal path')
        validate_seal(seals[t],sheets,len(source))
    OUT.mkdir(exist_ok=False)
    write(OUT/'STARTED.json',dict(started=True,entry=bind(entrypath),code=bind(__file__),protocol=bind(PROTOCOL),both_reviewers_sealed_before_any_score_read=True))
    try:
        reviews={t:normalize_observations(lines(check(seals[t]['observations'])),source,rendered) for t in ['A','B']}
        joined=consensus(reviews['A'],reviews['B'],source)
        truth=np.array([OWNER.index(r['center_owner']) if r['center_owner'] in OWNER else -1 for r in joined],np.int8)
        crop=np.array([CROP.index(r['crop_presence']) if r['crop_presence'] in CROP else 2 for r in joined],np.int8)
        scores=model_bindings(meta(SCOPE))
        require(scores==p['score_bindings_inherited_not_opened_or_hashed'] and p['v61_complete']==bind(V61_COMPLETE),'Frozen score bindings')
        predictions={t:load_owner(scores[t],source) for t in TAGS[:3]}
        predictions['v61']=load_v61(scores['v61'],source)
        masks=cohorts(truth,predictions);native=np.array([r['native_slot'] for r in source])
        report=dict(completed=True,weak_reference_only=True,selected_model=None,adopted=False,fully_solved=False,
                    rows=360,frames=45,sequence='val/000001',cohorts={},reviewer_agreement={},crop_associations={})
        for cohort,mask in masks.items():
            report['cohorts'][cohort]={}
            for group in ['ALL']+list(range(8)):
                ix=mask if group=='ALL' else mask&(native==group)
                info=dict(N=int(ix.sum()),original_input_U=sum(not source[i]['original_query_available'] for i in np.flatnonzero(ix)),
                          center4_invalid=sum(not source[i]['local_central4_available'] for i in np.flatnonzero(ix)),
                          no_constructed_crop=sum(not source[i]['feature_constructed'] for i in np.flatnonzero(ix)),models={})
                for tag in TAGS:
                    pred=predictions[tag][ix]
                    binary=pred if tag=='v61' else binary_labels(pred)
                    info['models'][tag]=dict(binary_T_nonT=hard_metrics(binary_labels(truth[ix]),binary,['T','NON_T']))
                    if tag!='v61':info['models'][tag]['owner3']=hard_metrics(truth[ix],pred,OWNER)
                report['cohorts'][cohort][str(group)]=info
        for field,known in [('center_owner',OWNER),('crop_presence',CROP)]:
            vals=known+[UNKNOWN]
            a=[vals.index(r[field]) for r in reviews['A']];b=[vals.index(r[field]) for r in reviews['B']]
            report['reviewer_agreement'][field]=dict(class_order=vals,table=count_table(a,b,len(vals),len(vals)),
                exact_all=sum(x==y for x,y in zip(a,b)),exact_known=sum(x==y and x<len(known) for x,y in zip(a,b)),
                disagreement=sum(x!=y for x,y in zip(a,b)))
        center=np.where(truth>=0,truth,3)
        report['crop_associations']['consensus_center_vs_crop']=dict(center_order=OWNER+[UNKNOWN],crop_order=CROP+[UNKNOWN],table=count_table(center,crop,4,3))
        for t in TAGS:
            pr=predictions[t];cat=np.where(pr>=0,pr,3 if t!='v61' else 2)
            report['crop_associations'][t]=dict(prediction_order=(OWNER+['U'] if t!='v61' else ['T','NON_T','U']),
                crop_order=CROP+[UNKNOWN],table=count_table(cat,crop,4 if t!='v61' else 3,3),association_only_not_crop_classifier=True)
        report['coverage']=dict(original_input_U=108,center4_invalid=112,input_available_center4_invalid=4,no_constructed_crop=24,
            consensus_center_known=int((truth>=0).sum()),consensus_center_unknown=int((truth<0).sum()),
            consensus_center_unknown_beyond_forced112=int((truth<0).sum())-112,
            consensus_crop_known=int((crop<2).sum()),consensus_crop_unknown=int((crop==2).sum()),
            model_prediction_U={t:int((predictions[t]<0).sum()) for t in TAGS},
            all4_model_paired_available=int(masks['ALL4_MODEL_PAIRED_AVAILABLE'].sum()))
        report['limitations']=[
            'Model-output-blind assistant weak visual consensus, not human/pixel/physical-native/coordinate gold; disagreement/U remains UNKNOWN.',
            'One previously exposed public validation video45frames, native0..7 input slots, correlated360queries; no pristine independent/external validation claim.',
            'Center literal material ownership differs from coarse crop target presence and native external/self visibility. v61 only binary reference, no3way ability.',
            'No threshold/model/data/architecture selection or new training admission; prior full native source qualification is not changed by this diagnostic.',
        ]
        jf=OUT/'ALL360_CENTER_OWNER_AND_CROP_CONSENSUS.jsonl'
        with jf.open('x') as f:
            for row in joined:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        ledger=OUT/'ALL360_FIXED_PREDICTIONS_AND_WEAK_OBSERVATIONS.csv'
        fields=['original_global_index','query_id','frame_index','native_slot','source_original_query_available','source_center4_allvalid','source_crop_constructed','center_owner','crop_presence']+[f'{t}_prediction' for t in TAGS]
        with ledger.open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for i,row in enumerate(joined):
                item={k:row[k] for k in fields if k in row}
                item.update({f'{t}_prediction':int(predictions[t][i]) for t in TAGS});w.writerow(item)
        rf=OUT/'FIXED_WEAK_CONFUSIONS_AND_STRATA.json';write(rf,report)
        for t in ['A','B']:check(entry['reviewer_seals'][t]);check(seals[t]['observations'])
        for b in scores.values():check(b)
        complete=dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),entry=bind(entrypath),
            reviewer_seals=entry['reviewer_seals'],observation_bindings={t:seals[t]['observations'] for t in ['A','B']},
            frozen_score_bindings=scores,all360_preserved=True,all45_frames_preserved=True,both_reviewers_sealed_before_any_score_read=True,
            outputs=dict(consensus=bind(jf),ledger=bind(ledger),report=bind(rf)),coverage=report['coverage'],
            model_forward_fit_threshold_data_selection_calls=0,private_current_V112_PnP_qL_changes=False,
            evaluation_only=True,new_training_admission=False,selected_model=None,adopted=False,fully_solved=False)
        write(OUT/'COMPLETE.json',complete)
        print(json.dumps(dict(completed=True,PASS=True,rows=360,complete=bind(OUT/'COMPLETE.json'))))
    except Exception:
        write(OUT/'FAILED.json',dict(completed=False,PASS=False,traceback=traceback.format_exc(),automatic_retry=False))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--entry',type=Path)
    args=parser.parse_args()
    prepare() if args.prepare else run(args.entry)
