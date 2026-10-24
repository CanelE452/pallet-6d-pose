"""Saved-only V184 evaluation. Preparation touches source/contract JSON + invented data.

Real annotation/baseline/new prediction bodies are opened only after a separately
bound root entry and BOTH fixed real prediction seals. No model imports/forwards.
"""
from pathlib import Path
from types import SimpleNamespace, ModuleType
from collections import Counter
import argparse
import ast
import csv
import hashlib
import json
import math
import platform
import traceback

import numpy as np

H = Path(__file__).resolve().parent
V = H.parent
P = V.parent
SCOPE = V/'ROOT_SCOPE_BEFORE_NEW_REAL_OUTPUTS_OR_V183_SOURCE_RESULTS.json'
CLARIFY = V/'ROOT_REFERENCE_AUTHORITY_CLARIFICATION_BEFORE_REAL_BODIES.json'
READINESS = P/'query_visibility_query_jitter_paired_source_training_20261024_v183/real_evaluation_readiness_c/REVIEW.json'
RAW_COMPLETE = V/'original_real_raw8_inputs_v1/COMPLETE.json'
PRED = V/'frozen_C52_C53_real_predictions_v1/PREDICTION_SEAL.json'
PRED_CODE = V/'frozen_prediction_preparation_c/compose_SAM96_and_infer_both_frozen_heads.py'
PRED_PROTOCOL = V/'frozen_prediction_preparation_c/PROTOCOL_BEFORE_REAL_RAW_SAM_OR_HEAD_WEIGHT_BODIES.json'
PRED_PREP = V/'frozen_prediction_preparation_c/PREPARATION_COMPLETE.json'
PRIOR_CURRENT = P/'query_visibility_direct_owner_frozen_current_benchmark_20261024_v178/evaluation_preparation'
PRIOR_PUBLIC = P/'query_visibility_public_held_blind_center_owner_review_20261024_v177'
CURRENT_CODE = PRIOR_CURRENT/'evaluate_saved_five_models.py'
CURRENT_PROTOCOL = PRIOR_CURRENT/'PROTOCOL_BEFORE_SAVED_CURRENT_SCORE_OR_LABEL_BODIES.json'
PUBLIC_CODE = PRIOR_PUBLIC/'saved_evaluation_preparation/evaluate_saved_public_held_owner.py'
PUBLIC_PROTOCOL = PRIOR_PUBLIC/'saved_evaluation_preparation/PROTOCOL_BEFORE_HELD_OBSERVATION_OR_MODEL_SCORE_BODIES.json'
PUBLIC_COMPLETE = PRIOR_PUBLIC/'saved_public_held_owner_evaluation_v1/COMPLETE.json'
CORE = P/'query_visibility_matched_query_scalar_20261023_v94/current_evaluation_preparation/evaluate_saved_only.py'
DECODER = CORE.parent/'infer_same_query_once.py'
METRIC = P/'query_visibility_cue_occlusion_20261023_v72/contract/saved_evaluate_current_scores.py'
REPORT = P/'query_visibility_spatial_target_query_features_20261023_v111/current_evaluation_preparation/evaluate_current_saved.py'
HELPER = P/'query_visibility_public_native_frozen_current_benchmark_20261024_v149/saved_evaluation_preparation/evaluate_saved_benchmark.py'
PROTOCOL = H/'PROTOCOL_BEFORE_REAL_ANNOTATION_OR_SAVED_SCORE_BODIES.json'
CHECKS = H/'GENERATED_MEANINGFUL_CHECKS.json'
PREP = H/'PREPARATION_COMPLETE.json'
OUT = V/'saved_real_head_evaluation_v1'
ARMS = ('C52_RAW96_JITTER_PAIRED_SOURCE', 'C53_RAW96_SAM_JITTER_PAIRED_SOURCE')
TAGS = ('C52', 'C53')
BASELINES = ('v61', 'legacy')
MODELS = BASELINES+TAGS
DOMAINS = {'CURRENT223_1784': (1784, 0, 45), 'PUBLIC_HELD45_360': (360, 584, 108)}
META = ('row_id', 'original_global_index', 'native_slot', 'query_xy', 'L', 'original_query_available')
OWNER = ['TARGET_VISIBLE_SUPPORTED', 'FOREIGN_VISIBLE_SUPPORTED', 'BACKGROUND_SUPPORTED']
OWNER_ORDER = ['TARGET', 'FOREIGN', 'BACKGROUND']
SUPPORT_ORDER = ['visible_end_support', 'foreign_front_support']
UNKNOWN = 'UNKNOWN'
PAIRS = tuple((a,b) for b in TAGS for a in BASELINES)+(('C52','C53'),)
SOURCE_HASHES = {
    SCOPE: '05c628ad960a5715fedd96fb4e3cde7b2a67ee3ce188e49eb644867f4e3e0b25',
    CLARIFY: '3e60202c1a000b2c5837f1fe826f26d05bc1201edc019e8e3ecf37f709f33fe6',
    RAW_COMPLETE: '2d90a7863fe80cfce95e3d54ee1ce529b2b800bcc85373cd7b1f0551c5e68a37',
    CURRENT_CODE: '5ff180a8e0924d43a0e10c5c5c0c25b65408e2903f7005ab78ff571cf7b33c07',
    CURRENT_PROTOCOL: '82afab189772a70f7ae358e512e956271b4ce73febec1f4988ca3f72c1010046',
    PUBLIC_CODE: 'b554fb6836f2d94ee8f3bbfbbcb1640662d46412c9db5cfb5c6e535a389f5f9b',
    PUBLIC_PROTOCOL: '18069829b30978be825f07c28e8bb7ac47a53961658aea7e723247cbd093af34',
    PUBLIC_COMPLETE: 'ad7e8d329a9dcffd136a481e978285840fd356b4134fb12d56c02aff5ba0dc24',
    PRED_CODE: 'ead5654961db0bca0dedb457a9212adc22a348a8b98be9b0abd83c7a15fbeaa0',
    PRED_PROTOCOL: '26f0aa63062b976d50c01a5797777960fc57ff33a8c204e92d22f5bbb819bafe',
    PRED_PREP: 'f19a6265e876772c8ad749b7a107a6c6f2fefa6d9497a76a50b9093ea29654f8',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    require('_v112' not in str(path).lower(), 'V112 prohibited')
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def bind(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def check(binding):
    require(bind(binding['path']) == binding, 'Frozen binding changed')
    return Path(binding['path'])


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def write_lines(path, rows):
    with Path(path).open('x') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')


def equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def extract(path, names, namespace):
    nodes = [n for n in ast.parse(Path(path).read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes} == set(names), 'Pinned pure function coverage')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path)+'::pure', 'exec'), namespace)


def kernels():
    # Original scalar metric/joins only, no top-level ancestor imports or I/O.
    ns = dict(np=np, math=math, Counter=Counter, c=SimpleNamespace(require=require), Path=Path, csv=csv,
              require=require, NEW=TAGS[0], OLD=TAGS[1], NEW_GEO='UNREPORTED_GEOMETRY0',
              OLD_GEO='UNREPORTED_GEOMETRY1', MODELS=MODELS)
    extract(CORE, {'readcsv','writecsv','key','boolean','numeric_same','score_index','response','join_all','calculate_all','historical_replay'}, ns)
    extract(DECODER, {'decode_rows'}, ns)
    # These helpers need their own names visible to one another in the same namespace.
    ns.update(NEW_MODELS=TAGS, PAIRS=PAIRS)
    extract(CURRENT_CODE, {'decision_state','assert_row_order','decision_rows','error_bucket','paired_transitions','owner_native_cross'}, ns)
    core = SimpleNamespace(**ns)
    base = ModuleType('fixed_original_human_semantic_metric')
    base.__dict__.update(math=math, Counter=Counter, D='DIRECT_VISIBLE', E='EXTERNAL_OCCLUDED', S='SELF_OCCLUDED',
        TASKS={'D_vs_E':['DIRECT_VISIBLE','EXTERNAL_OCCLUDED'], 'D_vs_S':['DIRECT_VISIBLE','SELF_OCCLUDED'],
               'D_vs_ES':['DIRECT_VISIBLE','EXTERNAL_OCCLUDED','SELF_OCCLUDED']}, GROUPS=['WOOD_PRIMARY105','SQUARE118'])
    extract(METRIC, {'yes','ratio','call','calculate'}, base.__dict__)
    ex = dict(np=np, math=math, Counter=Counter, c=SimpleNamespace(NEW=TAGS[0], OLD=TAGS[1],
              NEW_GEO='UNREPORTED_GEOMETRY0', OLD_GEO='UNREPORTED_GEOMETRY1', MODELS=MODELS))
    extract(REPORT, {'error_bins','matrices'}, ex)
    helper = dict(require=require)
    extract(HELPER, {'exact_csv_rows'}, helper)
    public = dict(np=np, require=require, check=check, OWNER=OWNER, UNKNOWN=UNKNOWN)
    extract(PUBLIC_CODE, {'hard_metrics','binary_labels','count_table','v61_decode','packet_metadata','load_v61'}, public)
    return core, base, SimpleNamespace(**ex), helper['exact_csv_rows'], SimpleNamespace(**public)


def decode(raw_support, raw_owner, available):
    n = len(available)
    require(available.dtype == bool and available.shape == (n,), 'Original availability bool')
    require(raw_support.dtype == raw_owner.dtype == np.float32 and raw_support.shape == (n,2) and raw_owner.shape == (n,3), 'FP32 support2/owner3')
    require(np.isfinite(raw_support[available]).all() and np.isfinite(raw_owner[available]).all(), 'Available finite raw outputs')
    require(np.isnan(raw_support[~available]).all() and np.isnan(raw_owner[~available]).all(), 'Original U raw NaN preserved')
    support = np.full((n,2), -1, np.int8)
    owner = np.full(n, -1, np.int8)
    proxy = np.full(n, -1, np.int8)
    support[available] = (raw_support[available] > 0).astype(np.int8)
    owner[available] = np.argmax(raw_owner[available], axis=1).astype(np.int8)
    proxy[available & (support[:,1] == 1)] = 1
    proxy[available & (support[:,1] == 0) & (support[:,0] == 1)] = 0
    return support, owner, proxy


def joint(support):
    result = np.full(len(support), 4, np.int8)
    good = (support >= 0).all(1)
    result[good] = support[good,0]+2*support[good,1]
    return result


def qarray(value):
    return np.array([np.nan, np.nan] if value is None else value, np.float64)


def packet_schema(packet, n):
    require(set(packet) == set(META)|{'raw_support_logits','raw_owner_logits','support_prediction','owner_prediction',
            'regional_support_proxy_D0_E1_Uminus1','support_order','owner_order'}, 'Exact target-free 13-key saved schema')
    for k, dtype, shape in [('original_global_index',np.int64,(n,)),('native_slot',np.int8,(n,)),
            ('query_xy',np.float64,(n,2)),('L',np.float64,(n,)),('original_query_available',np.bool_,(n,)),
            ('support_prediction',np.int8,(n,2)),('owner_prediction',np.int8,(n,)),
            ('regional_support_proxy_D0_E1_Uminus1',np.int8,(n,))]:
        require(packet[k].dtype == dtype and packet[k].shape == shape, 'Exact saved schema '+k)
    require(packet['row_id'].dtype.kind == 'U' and packet['row_id'].shape == (n,) and len(set(packet['row_id'].tolist())) == n, 'Unique original occurrence IDs')
    require(packet['support_order'].tolist() == SUPPORT_ORDER and packet['owner_order'].tolist() == OWNER_ORDER, 'Frozen head class orders')
    decoded = decode(packet['raw_support_logits'], packet['raw_owner_logits'], packet['original_query_available'])
    for key, value in zip(('support_prediction','owner_prediction','regional_support_proxy_D0_E1_Uminus1'), decoded):
        require(equal(packet[key], value), 'Saved strict>0/first-argmax/E-first/U decoder '+key)
    require(np.isin(packet['native_slot'], np.arange(8)).all(), 'Original native slot0..7')


def load_packet(binding, rows, domain, reference=None):
    with np.load(check(binding), allow_pickle=False) as handle:
        packet = {k: handle[k].copy() for k in handle.files}
    n, start, original_u = DOMAINS[domain]
    packet_schema(packet, n)
    require(len(rows) == n and np.array_equal(packet['original_global_index'], np.arange(start,start+n)), 'Original domain IDs, no concat alias')
    require(int((~packet['original_query_available']).sum()) == original_u, 'Original factual U preserved')
    for i, row in enumerate(rows):
        require(row['domain'] == domain and row['domain_index'] == i and row['original_global_index'] == start+i, 'Ordered original metadata domain')
        require(str(packet['row_id'][i]) == row['row_id'] == row['id']+':native'+str(row['native_slot']), 'Original row id/native occurrence spelling')
        require(int(packet['native_slot'][i]) == row['native_slot'] and bool(packet['original_query_available'][i]) == row['original_query_available'], 'Original native and U')
        require(np.array_equal(packet['query_xy'][i], qarray(row['query_xy']), equal_nan=True)
                and np.array_equal(packet['L'][i], np.array(row['L'],np.float64), equal_nan=True), 'Original F64 q/L')
        require(np.array_equal(qarray(row['actual_crop_grid_query_xy']), qarray(row['query_xy']), equal_nan=True)
                and np.array_equal(np.array(row['actual_crop_grid_L'],np.float64), np.array(row['L'],np.float64),equal_nan=True), 'Actual original crop grid, no FP32 alias substitution')
    if reference is not None:
        require(all(equal(packet[k], reference[k]) for k in META), 'Both heads exact same input META6 bytes')
    return packet


def current_associations(core, base, truth, packets):
    output=[]
    for group in ('ALL',*base.GROUPS):
        for bucket in ('ALL','LE3','GT3','MISSING_REFERENCE_ERROR'):
            for state in (base.D,base.E,base.S,'UNKNOWN'):
                ids=[i for i,r in enumerate(truth) if (group=='ALL' or r['group']==group)
                     and (bucket=='ALL' or core.error_bucket(r)==bucket)
                     and (r['human_state'] if r['human_state'] in (base.D,base.E,base.S) else 'UNKNOWN')==state]
                for tag,packet in packets.items():
                    cc=Counter(int(joint(packet['support_prediction'])[i]) for i in ids)
                    own=Counter(int(packet['owner_prediction'][i]) for i in ids)
                    pr=Counter(int(packet['regional_support_proxy_D0_E1_Uminus1'][i]) for i in ids)
                    output.append(dict(group=group,error_bin=bucket,human_semantic_state=state,model=tag,N=len(ids),
                        support_NONE=cc[0],support_VISIBLE_ONLY=cc[1],support_FOREIGN_ONLY=cc[2],support_BOTH=cc[3],support_U=cc[4],
                        proxy_D=pr[0],proxy_E=pr[1],proxy_U=pr[-1],owner_T=own[0],owner_F=own[1],owner_BG=own[2],owner_U=own[-1],
                        original_input_U=sum(not bool(packet['original_query_available'][i]) for i in ids),
                        scope='Association only: owner F/BG and support FOREIGN do not certify physical native external/self cause; primary evaluation separately uses existing human semantic authority.'))
    return output


def evaluate_current(p, packets, input_rows, out):
    core,base,extra,exact,_ = kernels()
    truth=core.readcsv(check(p['truth'])); manual=core.readcsv(check(p['manual'])); prior=lines(check(p['prior']))
    core.assert_row_order(core,truth,prior)
    for i,(t,r) in enumerate(zip(truth,prior)):
        for tag,packet in packets.items():
            require(core.key(r)==(input_rows[i]['id'],str(int(packet['native_slot'][i]))), 'Current original native mapping/order')
            require(core.numeric_same(r['query_x'],packet['query_xy'][i,0]) and core.numeric_same(r['query_y'],packet['query_xy'][i,1])
                    and core.numeric_same(r['roi_side'],packet['L'][i]), 'Original production q/L, no reference crop')
            require(core.boolean(r['valid']) == bool(packet['original_query_available'][i]), 'Original factual input45U')
    lookup={tag:{core.key(r):int(packets[tag]['regional_support_proxy_D0_E1_Uminus1'][i]) for i,r in enumerate(truth)} for tag in TAGS}
    original=base.call
    def call(row,model,task,probabilities,threshold,rules=None):
        return core.decision_state(lookup[model][core.key(row)]) if model in lookup else original(row,model,task,probabilities,threshold,rules)
    base.call=call
    adapted={tag:core.decision_rows(core,prior,packets[tag]['regional_support_proxy_D0_E1_Uminus1']) for tag in TAGS}
    old=[dict(r,raw_physical_external_logit=r['raw_query_aux_hidden_logit']) for r in adapted[TAGS[1]]]
    tables=core.calculate_all(base,truth,prior,adapted[TAGS[0]],old,manual)
    tables['coordinate_error_bins']=extra.error_bins(core,base,truth,manual,adapted[TAGS[0]],old)
    tables['D_vs_external_E_2x3']=extra.matrices(tables['metrics'])
    for av in [tables['availability'],*tables['availability_by_group']]:
        av['C52_regional_total_U']=av.pop('new_total_U'); av['C53_regional_total_U']=av.pop('V91_total_U')
    model_call=lambda r,m:call(r,m,'D_vs_E',{},.5)
    tables['paired_recovery_and_damage']=core.paired_transitions(base,truth,model_call)
    tables['support_joint_owner_and_reference_associations']=current_associations(core,base,truth,packets)
    eligible=[r for r in truth if base.yes(r,'paired_same_query_RGB_D_vs_E') and r['human_state'] in (base.D,base.E)]
    require(len(eligible)==1237 and sum(r['human_state']==base.D for r in eligible)==1147
            and sum(r['human_state']==base.E for r in eligible)==90, 'Fixed primary1237 D1147 E90')
    require(sum(base.yes(r,'same_slot_pipeline_D_vs_E') and r['human_state'] in (base.D,base.E) for r in truth)==1248, 'Same-slot1248')
    require({r['N'] for r in tables['manual_near_D']}=={153,223}, 'Original frozen manualnear376')
    replay=core.historical_replay(tables['metrics'],core.readcsv(check(p['historical_original'])),set(BASELINES))
    quality=core.readcsv(check(p['frozen_annotation_ledger']));require(len(quality)==1784,'Frozen reference authority ledger')
    ledger=[]
    for i,(r,qr) in enumerate(zip(truth,quality)):
        require(core.key(r)==core.key(qr), 'Quality association order, no primary filtering')
        d=dict(global_index=i,id=r['id'],native_corner=int(r['native_corner']),group=r['group'],human_state=r['human_state'],mapping_status=r['mapping_status'],
               query_x=float(packets[TAGS[0]]['query_xy'][i,0]) if np.isfinite(packets[TAGS[0]]['query_xy'][i,0]) else None,
               query_y=float(packets[TAGS[0]]['query_xy'][i,1]) if np.isfinite(packets[TAGS[0]]['query_xy'][i,1]) else None,
               roi_side=float(packets[TAGS[0]]['L'][i]) if np.isfinite(packets[TAGS[0]]['L'][i]) else None,
               query_available=bool(packets[TAGS[0]]['original_query_available'][i]),human_reference_x=r['human_reference_x'],human_reference_y=r['human_reference_y'],
               paired_same_query_RGB_D_vs_E=r['paired_same_query_RGB_D_vs_E'],same_slot_pipeline_D_vs_E=r['same_slot_pipeline_D_vs_E'],
               initial_reference_error_px=r['initial_reference_error_px'],original_query_valid=r['query_valid'],legacy_original_state=r['legacy_stored_state'])
        for tag in MODELS:
            response='OTHER_UNRESOLVED' if tag=='v61' and (r['query_valid']!='True' or r.get('present_at_fixed_0_5') not in ('True','False')) else model_call(r,tag)
            d[tag+'_original_response']=response;d[tag+'_D_E_U']='D' if response=='VISIBLE' else 'E' if response=='ABSENT_NONVISIBLE' else 'U'
        for tag,packet in packets.items():
            for k in ('owner_prediction','regional_support_proxy_D0_E1_Uminus1'):d[tag+'_'+k]=int(packet[k][i])
            d[tag+'_support_prediction']=packet['support_prediction'][i].tolist()
            d[tag+'_raw_support_logits']=[float(x) if np.isfinite(x) else None for x in packet['raw_support_logits'][i]]
            d[tag+'_raw_owner_logits']=[float(x) if np.isfinite(x) else None for x in packet['raw_owner_logits'][i]]
        d['v61_saved_p_present']=r.get('presence_probability','');d['legacy_saved_p_DIRECT']=r.get('legacy_p_DIRECT',r.get('legacy_direct_probability',''))
        for key in ('annotation_coordinate_source','raw_annotation_traceback','new_primary_quality_gold_target','secondary_error_px','secondary_fixed_reference_target'):d[key]=qr.get(key,'UNKNOWN_UNCERTIFIED')
        d['reference_scope']='Existing frozen reference/manual3px proxy association; existing human semantic visibility authority is valid. No new coordinate gold or primary quality gate.'
        ledger.append(d)
    inventory=[]
    for group in ('ALL',*base.GROUPS):
        rr=ledger if group=='ALL' else [r for r in ledger if r['group']==group]
        for tag in MODELS:
            cc=Counter(r[tag+'_D_E_U'] for r in rr); responses=Counter(r[tag+'_original_response'] for r in rr)
            inventory.append(dict(group=group,model=tag,all_registered_slots=len(rr),D=cc['D'],E_binary_response=cc['E'],U=cc['U'],U_rate=cc['U']/len(rr),
                legacy_abstain=responses['ABSTAIN'],out_of_task_geometry_S=responses['OUT_OF_TASK_GEOMETRY_S'],other_unresolved=responses['OTHER_UNRESOLVED'],
                human_state_inventory=json.dumps(dict(Counter(r['human_state'] for r in rr)),sort_keys=True)))
    tables['all1784_model_inventory']=inventory
    historical={}
    for name in ('METRICS','CONFUSION','HUMAN_SELF_BINARY_RESPONSE','MANUAL_NEAR_D','COORDINATE_ERROR_BINS','D_VS_EXTERNAL_E_2X3','ALL1784_MODEL_INVENTORY'):
        hist=core.readcsv(check(p['historical_three_tables'][name]));key=next(k for k in tables if k.upper()==name)
        historical[name]=exact([r for r in tables[key] if r.get('model') in BASELINES],[r for r in hist if r.get('model') in BASELINES])
    oldledger=core.readcsv(check(p['historical_three_tables']['ALL1784_NATIVE_LEDGER']))
    fields=('id','native_corner','v61_original_response','v61_D_E_U','legacy_original_response','legacy_D_E_U')
    historical['ALL1784_NATIVE_LEDGER_BASELINE_RESPONSES']=exact([{k:r[k] for k in fields} for r in ledger],[{k:r[k] for k in fields} for r in oldledger])
    for name,value in tables.items():
        if isinstance(value,list):core.writecsv(out/(name.upper()+'.csv'),value)
    write_lines(out/'ALL1784_CURRENT_NATIVE_LEDGER.jsonl',ledger)
    return dict(completed=True,original_slots=1784,primary_D_E_N=1237,primary_D=1147,primary_E=90,same_slot_N=1248,
        original_input_U=45,original_available=1739,WOOD_frames=105,SQUARE_frames=118,tables=tables,historical_original_metric_rows_exact=replay,
        historical_v61_legacy_tables_exact=historical,primary_quality_owner_reference_error_filter=False,
        human_semantic_authority='Existing human D/E/S with unchanged original mapping, missing/ambiguous mapping UNKNOWN; no new mesh recertification blocker.',
        prediction_scope='Frozen regional diagnostic E-first; independent supports and owner3 separately reported; neither owner F/BG nor nonvisible establishes native external/self physical cause.',
        reference_scope='Existing <=3px/manualnear/missing-reference association only; no coordinate correction or new point-quality gold.',
        repeated_development_exposure=True,fresh_independent_validation=False,legacy_CAL_overlap_preserved=True)


def evaluate_public(p, packets, input_rows, out):
    _,_,_,_,k=kernels()
    source=lines(check(p['inputs']));joined=lines(check(p['consensus']))
    require(len(source)==len(joined)==360,'All original held360, no label drop')
    for i,(s,r,m) in enumerate(zip(source,joined,input_rows)):
        require(s['original_global_index']==r['original_global_index']==m['original_global_index']==584+i
                and s['query_id']==r['query_id']==m['row_id'] and s['native_slot']==r['native_slot']==m['native_slot'], 'Held consensus/source/original row join')
        require(r['source_original_query_available']==s['original_query_available']==m['original_query_available']
                and r['source_center4_allvalid']==s['local_central4_available'] and r['source_crop_constructed']==s['feature_constructed'], 'Original held flags')
        require(np.array_equal(qarray(r['query_xy']),qarray(s['query_xy']),equal_nan=True)
                and np.array_equal(qarray(m['query_xy']),qarray(s['query_xy']),equal_nan=True)
                and np.array_equal(np.array(r['L'],np.float64),np.array(s['L'],np.float64),equal_nan=True)
                and np.array_equal(np.array(m['L'],np.float64),np.array(s['L'],np.float64),equal_nan=True), 'Held original q/L unchanged')
        require(r['center_owner'] in OWNER+[UNKNOWN] and r['crop_presence'] in ['YES','NO',UNKNOWN], 'Frozen weak enums')
        for field,known in [('center_owner',OWNER),('crop_presence',['YES','NO'])]:
            a,b=r['reviewer_A'][field],r['reviewer_B'][field]
            require(r[field]==(a if a==b and a in known else UNKNOWN),'Frozen A/B exact-known consensus, no relabel')
        require((s['original_query_available'] and s['local_central4_available']) or r['center_owner']==UNKNOWN,'Forced center U')
        require(s['feature_constructed'] or r['crop_presence']==UNKNOWN,'No crop forced cropU')
    truth=np.array([OWNER.index(r['center_owner']) if r['center_owner'] in OWNER else -1 for r in joined],np.int8)
    crop=np.array([['YES','NO',UNKNOWN].index(r['crop_presence']) for r in joined],np.int8)
    native=np.array([r['native_slot'] for r in source],np.int8)
    require(int((truth>=0).sum())==204 and int((truth<0).sum())==156,'Original known204/U156 weakconsensus unchanged')
    require(sum(not r['original_query_available'] for r in source)==108
            and sum(not r['local_central4_available'] for r in source)==112
            and sum(not r['feature_constructed'] for r in source)==24,'Original held input/center/crop U contracts')
    v61=k.load_v61(p['v61'],source)
    predictions={tag:packets[tag]['owner_prediction'] for tag in TAGS};predictions['v61']=v61
    paired=np.logical_and.reduce([predictions[tag]>=0 for tag in (*TAGS,'v61')])
    masks=dict(FULL360=np.ones(360,bool),KNOWN_CENTER_TRUTH=truth>=0,ALL3_MODEL_PAIRED_AVAILABLE=paired,KNOWN_TRUTH_AND_ALL3_PAIRED_AVAILABLE=(truth>=0)&paired)
    cohorts={}
    for cohort,mask in masks.items():
        cohorts[cohort]={}
        for group in ('ALL',*range(8)):
            ix=mask if group=='ALL' else mask&(native==group)
            models={}
            for tag,pr in predictions.items():
                binary=pr if tag=='v61' else k.binary_labels(pr)
                models[tag]=dict(binary_T_nonT=k.hard_metrics(k.binary_labels(truth[ix]),binary[ix],['T','NON_T']))
                if tag!='v61':models[tag]['owner3']=k.hard_metrics(truth[ix],pr[ix],OWNER)
            cohorts[cohort][str(group)]=dict(N=int(ix.sum()),original_input_U=sum(not source[i]['original_query_available'] for i in np.flatnonzero(ix)),
                center4_invalid=sum(not source[i]['local_central4_available'] for i in np.flatnonzero(ix)),
                no_constructed_crop=sum(not source[i]['feature_constructed'] for i in np.flatnonzero(ix)),models=models)
    associations={}
    center=np.where(truth>=0,truth,3)
    for tag,packet in packets.items():
        js=joint(packet['support_prediction']);pr=packet['regional_support_proxy_D0_E1_Uminus1'];own=packet['owner_prediction']
        associations[tag]=dict(support_joint_order=['NONE','VISIBLE_ONLY','FOREIGN_ONLY','BOTH','U'],crop_order=['YES','NO',UNKNOWN],
            center_order=OWNER+[UNKNOWN],support_joint_vs_crop=k.count_table(js,crop,5,3),support_joint_vs_center=k.count_table(js,center,5,4),
            proxy_order=['D','E','U'],proxy_vs_crop=k.count_table(np.where(pr>=0,pr,2),crop,3,3),
            owner_order=OWNER+['U'],owner_vs_crop=k.count_table(np.where(own>=0,own,3),crop,4,3),
            association_only=True,no_support_or_native_external_gold=True,no_ownerT_as_crop_classifier_metric=True)
    ledger=[]
    for i,r in enumerate(joined):
        row={k:r[k] for k in ('original_global_index','query_id','frame_index','native_slot','source_original_query_available','source_center4_allvalid','source_crop_constructed','center_owner','crop_presence')}
        row['v61_binary_T_nonT_prediction']=int(v61[i])
        for tag,packet in packets.items():
            row[tag+'_owner_prediction']=int(packet['owner_prediction'][i]);row[tag+'_support_prediction']=packet['support_prediction'][i].tolist()
            row[tag+'_regional_support_proxy_D0_E1_Uminus1']=int(packet['regional_support_proxy_D0_E1_Uminus1'][i])
        ledger.append(row)
    write_lines(out/'ALL360_PUBLIC_WEAK_LEDGER.jsonl',ledger)
    return dict(completed=True,rows=360,frames=45,sequence='val/000001',cohorts=cohorts,support_crop_center_associations=associations,
        coverage=dict(original_input_U=108,source_available=252,center4_invalid=112,otherwise_available_center4_invalid=4,no_constructed_crop=24,
            known_center204=int((truth>=0).sum()),unknown_center156=int((truth<0).sum()),v61_own_available=int((v61>=0).sum()),
            v61_own_U=int((v61<0).sum()),new_owner_U={tag:int((packets[tag]['owner_prediction']<0).sum()) for tag in TAGS},
            new_regional_U={tag:int((packets[tag]['regional_support_proxy_D0_E1_Uminus1']<0).sum()) for tag in TAGS}),
        legacy_baseline=dict(status='MISSING_IN_REVIEWED_LINEAGE',new_forward=False,not_imputed=True),
        reference_authority='Previously sealed prediction-blind assistant weak center-owner/crop-presence consensus only; no support or native external/self gold.',
        repeated_single_sequence_development_exposure=True,fresh_independent_validation=False,training_admission=False)


def validate_prediction_seal(seal, p):
    require(seal.get('completed') is True and seal.get('PASS') is True,'Both real predictions completed')
    require(seal.get('all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads') is True,'Both real predictions closed before evaluation bodies')
    require(seal['scope']==p['scope'] and seal['authority_clarification']==p['authority_clarification']
            and seal['source_fit_complete']==p['source_fit_complete'] and seal['original_raw_complete']==p['original_raw_complete'],'Exact frozen scope/fit/original inputs')
    require(seal['code']==p['prediction_producer_code'] and seal['protocol']==p['prediction_producer_protocol'], 'Frozen real prediction producer code/protocol')
    require(set(seal['arms'])==set(ARMS) and seal['original_rows2144']==2144 and seal['model_domain_rows4288']==4288,'Both fixed model/domain full census')
    require(seal['annotations_reference_GT_oldreal_scores_or_fullRGB_read'] is False
            and seal['all_batch_repeat2_state_and_input_bytes_exact'] is True,'Prediction exposure and repeat/state seal')
    require(seal['selected_model'] is None and seal['adopted'] is False,'No model choice')
    for arm in ARMS:
        require(seal['arms'][arm]['updates']==8000 and set(seal['arms'][arm]['outputs'])==set(DOMAINS),'Both fixed final8000, both full domains')
    return {tag:seal['arms'][arm]['outputs'] for tag,arm in zip(TAGS,ARMS)}


def generated_checks():
    passed=[]
    rs=np.array([[0,0],[1,0],[0,1],[1,1],[np.nan,np.nan]],np.float32)
    ro=np.array([[0,0,0],[1,2,2],[1,0,2],[3,2,1],[np.nan]*3],np.float32);av=np.array([1,1,1,1,0],bool)
    sp,op,proxy=decode(rs,ro,av)
    require(sp.tolist()==[[0,0],[1,0],[0,1],[1,1],[-1,-1]] and op.tolist()==[0,1,2,0,-1]
            and proxy.tolist()==[-1,0,1,1,-1] and joint(sp).tolist()==[0,1,2,3,4],'Independent support zero/BOTH/NONE, first tie and E-first/U')
    passed.append('Strict_zero_negative_binary_support_BOTH_NONE_Efirst_owner_firsttie_and_inputU')
    bad=rs.copy();bad[-1]=0
    try:decode(bad,ro,av)
    except ValueError:pass
    else:raise AssertionError('InputU finite raw accepted')
    passed.append('InputU_NaN_immutable_no_quality_or_owner_gate')
    toy=dict(row_id=np.array(['i'+str(i) for i in range(5)]),original_global_index=np.arange(5,dtype=np.int64),native_slot=np.array([7,2,6,0,1],np.int8),
        query_xy=np.ones((5,2),np.float64),L=np.ones(5,np.float64),original_query_available=av,raw_support_logits=rs,raw_owner_logits=ro,
        support_prediction=sp,owner_prediction=op,regional_support_proxy_D0_E1_Uminus1=proxy,
        support_order=np.array(SUPPORT_ORDER),owner_order=np.array(OWNER_ORDER))
    packet_schema(toy,5);rejected=0
    for changes in [dict(row_id=np.array(['same']*5)),dict(query_xy=toy['query_xy'].astype(np.float32)),dict(owner_order=np.array(OWNER_ORDER[::-1])),
                    dict(owner_prediction=np.array([1,1,2,0,-1],np.int8)),dict(regional_support_proxy_D0_E1_Uminus1=np.array([-1,0,1,0,-1],np.int8)),dict(features=np.zeros(1))]:
        try:packet_schema(dict(toy,**changes),5)
        except ValueError:rejected+=1
    require(rejected==6,'Malformed target-free schema/order/decoder rejected')
    passed.append('Exact13keys_dtypes_original_native_slots_unique_occurrences_and_saved_decoder_failclosed')
    core,base,_,_,public=kernels()
    prior=[dict(id=f'I{i//8}',native_corner=i%8,valid=i<1739,geometry_self_hidden=(i%2==0),reason='' if i<1739 else 'INPUT_U') for i in range(1784)]
    core.assert_row_order(core,[dict(r) for r in prior],prior)
    swapped=[dict(r) for r in prior];swapped[0],swapped[1]=swapped[1],swapped[0]
    try:core.assert_row_order(core,swapped,prior)
    except ValueError:pass
    else:raise AssertionError('Current native order drift accepted')
    pp=np.tile(np.array([0,1],np.int8),892);pp[1739:]=-1;pp[1]=-1
    rr=core.decision_rows(core,prior,pp)
    require(sum(not r['valid'] for r in rr)==46 and rr[0]['authoritative_decision']=='D' and rr[1]['authoritative_decision']=='UNKNOWN','Full1784 sign adapter and NONE extraU')
    passed.append('Same_actual_current_adapter1784_order_and45factualU_plus_regionalNONE_U')
    legacy=[dict(human_state=base.D,id='i',legacy_stored_state='U'),dict(human_state=base.E,id='j',legacy_stored_state='S')]
    met=base.calculate(legacy,'legacy','D_vs_E',{},.5)
    require(met['N']==2 and met['U_visible']==1 and met['out_of_task_S_occluded']==1,'Legacy abstention and S ownroute')
    passed.append('Frozen_legacy_ABSTAIN_and_outoftaskS_not_recast_as_new_support')
    toy_truth=np.array([0,1,2,-1,0,1],np.int8);toy_pred=np.array([0,2,-1,1,-1,1],np.int8)
    met=public.hard_metrics(toy_truth,toy_pred,['T','F','BG'])
    require(met['full_confusion']==[[1,0,0,1],[0,1,1,0],[0,0,0,1],[0,1,0,0]]
            and met['per_class']['F']['precision_on_known_truth']==1 and met['per_class']['T']['recall_including_prediction_U']==.5,'Weak known denominator/unknown/U')
    require(public.hard_metrics(public.binary_labels(toy_truth),public.binary_labels(toy_pred),['T','NON_T'])['full_confusion']==[[1,0,1],[0,2,1],[0,1,0]],'Weak same truth binary collapse')
    passed.append('Weak_CM_precision_excludes_unknown_truth_recall_includes_U_and_binary_collapse')
    require(public.v61_decode(np.array([0,-1,np.nan],np.float32),np.array([1,1,0],bool)).tolist()==[0,1,-1],'v61 >=0 ownavailability')
    passed.append('Original_v61_zero_visible_sign_and_ownU_not_new_sourceavailability')
    require(core.error_bucket(dict(initial_reference_error_px='3'))=='LE3' and core.error_bucket(dict(initial_reference_error_px='3.00001'))=='GT3'
            and core.error_bucket({})=='MISSING_REFERENCE_ERROR','Existing <=3px/missing no primary filter')
    passed.append('Frozen3px_boundary_and_missing_reference_kept_separate')
    tt=[dict(id=g+s,group=g,human_state=s,paired_same_query_RGB_D_vs_E='True',same_slot_pipeline_D_vs_E='True') for g in base.GROUPS for s in (base.D,base.E)]
    transitions=core.paired_transitions(base,tt,lambda r,m:'VISIBLE' if r['human_state']==base.D else 'ABSENT_NONVISIBLE')
    require(len(transitions)==60 and all(x['both_correct']==x['N'] for x in transitions),'Fixed paired bothnew/baselines panels')
    passed.append('Both_fixed_new_models_and_v61_legacy_transitions_N_conservation')
    require(public.count_table([0,1,2,3,4],[0,1,2,0,1],5,3)==[[1,0,0],[0,1,0],[0,0,1],[1,0,0],[0,1,0]],'Separate support/crop association')
    passed.append('Joint_BOTH_NONE_U_vs_crop_only_association_no_native_gold_conversion')
    b=dict(path='/invented/metadata.json',sha256='invented')
    p=dict(scope=b,authority_clarification=b,source_fit_complete=b,original_raw_complete=b,prediction_producer_code=b,prediction_producer_protocol=b)
    sealed=dict(completed=True,PASS=True,scope=b,authority_clarification=b,source_fit_complete=b,original_raw_complete=b,code=b,protocol=b,
        all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads=True,original_rows2144=2144,model_domain_rows4288=4288,
        annotations_reference_GT_oldreal_scores_or_fullRGB_read=False,all_batch_repeat2_state_and_input_bytes_exact=True,selected_model=None,adopted=False,
        arms={arm:dict(updates=8000,outputs={domain:b for domain in DOMAINS}) for arm in ARMS})
    validate_prediction_seal(sealed,p);rejected=0
    for delta in [dict(all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads=False),
                  dict(annotations_reference_GT_oldreal_scores_or_fullRGB_read=True),dict(code=dict(path='/invented/other.py',sha256='other')),
                  dict(arms={ARMS[0]:sealed['arms'][ARMS[0]]})]:
        try:validate_prediction_seal(dict(sealed,**delta),p)
        except ValueError:rejected+=1
    require(rejected==4,'Before-annotation bothmodels/provenance/closed-body barrier')
    passed.append('Both_real_predictions_and_frozen_producer_exact_binding_barrier_before_any_annotation_or_scores')
    return dict(completed=True,PASS=True,invented_only=True,checks=passed,count=len(passed),
                actual_real_annotation_old_or_new_scores_RGB_features_weights_body_or_hash_reads=0,model_forward_fit_threshold_selection=0)


def prepare():
    require(platform.python_version()=='3.10.20' and np.__version__=='1.26.4','Pinned CPU metric runtime')
    for path,expected in SOURCE_HASHES.items():require(sha(path)==expected,'Frozen source/contract metadata '+str(path))
    scope=read(SCOPE);raw=read(RAW_COMPLETE);old=read(CURRENT_PROTOCOL);pub=read(PUBLIC_PROTOCOL);complete=read(PUBLIC_COMPLETE)
    require(raw['completed'] and raw['PASS'] and complete['completed'] and complete['PASS'],'Existing contract COMPLETE metadata')
    codes=[CORE,DECODER,METRIC,REPORT,HELPER,CURRENT_CODE,PUBLIC_CODE]
    for binding in old['operator_codes']+[old['code']]:check(binding)
    checks=generated_checks();write(CHECKS,checks)
    keep=('prior','truth','manual','historical_original','historical_three_tables','annotation_authority','frozen_annotation_ledger')
    p=dict(completed=True,prepared_only=True,code=bind(__file__),scope=bind(SCOPE),authority_clarification=bind(CLARIFY),readiness=bind(READINESS),
        original_raw_complete=bind(RAW_COMPLETE),source_fit_complete=raw['fit_complete'],checks=bind(CHECKS),operator_codes=[bind(x) for x in codes],
        prediction_producer_code=bind(PRED_CODE),prediction_producer_protocol=bind(PRED_PROTOCOL),prediction_producer_preparation=bind(PRED_PREP),
        prior_current_protocol=bind(CURRENT_PROTOCOL),prior_public_protocol=bind(PUBLIC_PROTOCOL),public_prior_complete=bind(PUBLIC_COMPLETE),
        current={k:old[k] for k in keep},
        public=dict(inputs=pub['inputs'],definition=pub['definition'],consensus=complete['outputs']['consensus'],v61=complete['frozen_score_bindings']['v61'],
            prior_ledger=complete['outputs']['ledger'],prior_report=complete['outputs']['report'],legacy='MISSING_IN_REVIEWED_LINEAGE_NO_NEW_FORWARD'),
        domains={name:dict(rows=DOMAINS[name][0],original_inputU=DOMAINS[name][2],metadata=raw['domains'][name]['metadata']) for name in DOMAINS},
        prediction_seal_path=str(PRED),prediction_seal_binding='ROOT_FILL_AFTER_BOTH_COMPLETE',arms=list(ARMS),models=list(MODELS),output=str(OUT),
        packet_contract=dict(keys13=list(META)+['raw_support_logits','raw_owner_logits','support_prediction','owner_prediction','regional_support_proxy_D0_E1_Uminus1','support_order','owner_order'],
            support_decoder='FP32 logit>0 independently; BOTH/NONE retained; inputU rawNaN/pred-1 first',owner_decoder='First-tie argmax3 separate from support and human semantic',
            regional_proxy='foreignYES -> E; else visibleYES -> D; elseU; no quality/owner/error gate',metadata='OriginalF64 q/L, originalsource IDs/current0..1783/public584..943, original availability, exactsameMETA6 allarms'),
        current_contract='All1784 primary1237(D1147,E90),same-slot1248,WOOD105/SQUARE118,self separate,manualnear376 and fixed<=3px/missing-reference associations; human semantic authority valid, original mappingUNKNOWN preserved.',
        current_baselines='Historical v61/legacy kernel and own operators/availability/legacyCAL exposure unchanged; exact prior baseline tables and responses replayed.',
        public_contract='Held45/360 sourceU108 vs centerinvalid112 vs consensus204known/156U vs v61own251available/109U distinguished. Owner3 and same-center T/nonT weak CM/P/R only; support/crop association not native/support gold.',
        public_cohorts=['FULL360','KNOWN_CENTER_TRUTH','ALL3_MODEL_PAIRED_AVAILABLE','KNOWN_TRUTH_AND_ALL3_PAIRED_AVAILABLE'],public_native_groups=['ALL',*range(8)],
        exposure='Current223 and publicheld45 have repeated development exposure, not fresh independent validation. Neither real result may select head, decoder, threshold, architecture or fitting data.',
        primary_quality_error_owner_filter=False,new_training_admission=False,selected_model=None,adopted=False,fully_solved=False,automatic_retry=False,
        annotation_score_baseline_RGB_feature_checkpoint_body_or_hash_reads_during_prepare=0)
    write(PROTOCOL,p)
    template=H/'ROOT_ACTUAL_SAVED_REAL_EVALUATION_ENTRY_TEMPLATE_APPROVED_FALSE.json'
    write(template,dict(approved_saved_real_evaluation=False,all_two_real_predictions_frozen=False,evaluation_only=True,
        no_model_data_architecture_threshold_selection=True,no_forward_fit_RGB_features_weights_PnP_qL_or_V112=True,
        code=bind(__file__),protocol=bind(PROTOCOL),scope=bind(SCOPE),authority_clarification=bind(CLARIFY),prediction_seal=None,output=str(OUT)))
    write(PREP,dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),checks=bind(CHECKS),entry_template=bind(template),
        no_actual_real_annotation_or_score_bodies_or_hashes=True,new_model_fit_forward_threshold_calls=0))
    print(json.dumps(dict(prepared_only=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),checks=bind(CHECKS),preparation=bind(PREP))))


def run(entry_path):
    e=read(entry_path)
    require(all(e.get(k) is True for k in ('approved_saved_real_evaluation','all_two_real_predictions_frozen','evaluation_only',
            'no_model_data_architecture_threshold_selection','no_forward_fit_RGB_features_weights_PnP_qL_or_V112')),'Separate root saved-only evaluation entry')
    require(e['code']==bind(__file__) and e['protocol']==bind(PROTOCOL) and e['scope']==bind(SCOPE)
            and e['authority_clarification']==bind(CLARIFY) and e['output']==str(OUT),'Frozen evaluator/root scope')
    require(platform.python_version()=='3.10.20' and np.__version__=='1.26.4','Pinned CPU runtime')
    p=read(PROTOCOL);require(read(check(p['checks']))['PASS'] and read(PREP)['PASS'],'Invented checks/preparation passed')
    require(e['prediction_seal']['path']==str(PRED),'Exact both-real prediction seal path')
    seal=read(check(e['prediction_seal']));outputs=validate_prediction_seal(seal,p)
    # No real annotation, old baseline score or new prediction body has been read above.
    for b in p['operator_codes']+[p['original_raw_complete'],p['source_fit_complete'],p['readiness'],
            p['prediction_producer_code'],p['prediction_producer_protocol'],p['prediction_producer_preparation']]:check(b)
    fit=read(check(p['source_fit_complete']))
    require(fit['completed'] and fit['PASS'] and fit['all2_frozen_before_outputs_or_evaluation_access'],'Both source heads fixed before real input/evaluation')
    for arm in ARMS:
        require(fit['arms'][arm]['updates']==8000 and seal['arms'][arm]['checkpoint']==fit['arms'][arm]['checkpoint']
                and seal['arms'][arm]['source_final_state_sha']==fit['arms'][arm]['final_state_sha'], 'Exact fixed final8000 source head provenance; no checkpoint body read')
    check(p['current']['annotation_authority']);check(p['public']['definition'])
    require(not OUT.exists(),'Single fixed new output; no overwrite/retry')
    OUT.mkdir();write(OUT/'BEFORE_ANNOTATION_AND_SAVED_SCORE_READ.json',dict(completed=True,entry=bind(entry_path),prediction_seal=e['prediction_seal'],both_models_closed=True,annotation_or_score_bodies_read_so_far=False))
    try:
        packets={};metadata={}
        for domain in DOMAINS:
            metadata[domain]=lines(check(p['domains'][domain]['metadata']));packets[domain]={};reference=None
            for tag in TAGS:
                packet=load_packet(outputs[tag][domain],metadata[domain],domain,reference)
                packets[domain][tag]=packet;reference=packet
        current_out=OUT/'current1784';current_out.mkdir();public_out=OUT/'publicheld360';public_out.mkdir()
        current=evaluate_current(p['current'],packets['CURRENT223_1784'],metadata['CURRENT223_1784'],current_out)
        public=evaluate_public(p['public'],packets['PUBLIC_HELD45_360'],metadata['PUBLIC_HELD45_360'],public_out)
        write(current_out/'CURRENT_HUMAN_SEMANTIC_AND_SEPARATE_ASSOCIATIONS.json',current)
        write(public_out/'PUBLIC_WEAK_OWNER_AND_SEPARATE_SUPPORT_ASSOCIATIONS.json',public)
        for tag in TAGS:
            for b in outputs[tag].values():check(b)
        # Seal all actually consumed annotation/baseline bindings unchanged after evaluation.
        for k in ('prior','truth','manual','historical_original','frozen_annotation_ledger'):check(p['current'][k])
        for b in p['current']['historical_three_tables'].values():check(b)
        for k in ('inputs','consensus','v61'):check(p['public'][k])
        for b in (p['scope'],p['authority_clarification'],p['original_raw_complete'],e['prediction_seal']):check(b)
        files={str(path.relative_to(OUT)):bind(path) for path in sorted(OUT.rglob('*')) if path.is_file()}
        complete=dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),entry=bind(entry_path),prediction_seal=e['prediction_seal'],
            outputs=files,models=list(MODELS),current_all1784=True,public_all360=True,new_model_domain_rows4288=True,
            primary_current_D1147_E90_N1237=True,current_human_semantic_authority_valid=True,primary_quality_owner_error_gate=False,
            public_reference_weak_only=True,public_legacy_missing=True,all_both_predictions_closed_before_any_annotation_or_score_body=True,
            repeated_development_exposure_not_fresh_validation=True,new_forward_fit_RGB_features_weights_threshold_or_model_data_choice=0,
            old_annotations_outputs_and_inputs_unchanged=True,new_training_admission=False,selected_model=None,adopted=False,fully_solved=False,automatic_retry=False)
        write(OUT/'COMPLETE.json',complete)
        print(json.dumps(dict(completed=True,PASS=True,complete=bind(OUT/'COMPLETE.json'))))
    except BaseException:
        write(OUT/'FAILED.json',dict(completed=False,PASS=False,traceback=traceback.format_exc(),automatic_retry=False))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--entry',type=Path)
    args=parser.parse_args();prepare() if args.prepare else run(args.entry)
