"""Saved-only V220 all16 evaluation. Preparation touches source/contract JSON + invented data.

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
import os
import stat
import tempfile
import traceback

import numpy as np

H = Path(__file__).absolute().parent
P = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
V = P/'query_visibility_expected_part_weak_frozen_real_benchmark_20261025_v220'
OLD195=P/'query_visibility_grouped_direct_end_heads_frozen_real_benchmark_20261024_v195'
OLD203=P/'query_visibility_spatial_q8_heads_frozen_real_benchmark_20261025_v203'
OLD200=P/'query_visibility_native_heads_frozen_real_benchmark_20261024_v200'
INHERITED_PROTOCOL=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_dense_point_heads_frozen_real_benchmark_20261025_v213/inference_evaluation_preparation_b/PROTOCOL_BEFORE_REAL_ANNOTATION_OR_SAVED_SCORE_BODIES.json')
OLD191=P/'query_visibility_direct_end_heads_frozen_real_benchmark_20261024_v191'
OLD184 = P/'query_visibility_jitter_heads_frozen_real_benchmark_20261024_v184'
SCOPE = V/'ROOT_PROSPECTIVE_FROZEN_EXPECTED_PART_HEADS_REAL_COMPARISON_SCOPE.json'
CLARIFY = OLD184/'ROOT_REFERENCE_AUTHORITY_CLARIFICATION_BEFORE_REAL_BODIES.json'
ADDENDUM=OLD200/'ROOT_TRANSITIVE_SAVED_EVALUATOR_CODE_ONLY_ADDENDUM.json'
DECODER_SPEC=P/'query_visibility_direct_end_paired_source_training_20261024_v190/ROOT_DIRECT_LOGIT_DECODER_NUMERICAL_SPEC_BEFORE_FIT.json'
OLD_PROTOCOL = OLD184/'evaluation_preparation_b/PROTOCOL_BEFORE_REAL_ANNOTATION_OR_SAVED_SCORE_BODIES.json'
PRIOR_PRED = OLD184/'frozen_C52_C53_real_predictions_v1/PREDICTION_SEAL.json'
RAW_COMPLETE = OLD184/'original_real_raw8_inputs_v1/COMPLETE.json'
RAW10_COMPLETE=OLD184/'real_raw10_SAM96_inputs_v1/COMPLETE.json'
PRED = V/'frozen_C64_C65_real_predictions_v1/PREDICTION_SEAL.json'
PRED_CODE = H/'infer_both_frozen_expected_part_heads.py'
PRED_PROTOCOL = H/'PROTOCOL_BEFORE_ANY_REAL_ARRAY_OR_FINAL_WEIGHT_BODIES.json'
PRED_PREP = H/'PREDICTION_PREPARATION_COMPLETE.json'
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
ARMS = ('C64_STRONG_ONLY_RAW_Q8','C65_STRONG_PLUS_EXPECTED_PART_WEAK_RAW_Q8')
ARM_SPATIAL=dict(zip(ARMS,(True,True)))
ARM_WEAK={ARMS[0]:(False,0.),ARMS[1]:(True,.25)}
SOURCE_ARM_MAP={ARMS[0]:'STRONG_ONLY_CONTROL',ARMS[1]:'STRONG_PLUS_EXPECTED_PART_WEAK'}
OLD213=P/'query_visibility_dense_point_heads_frozen_real_benchmark_20261025_v213'
PRIOR_DENSE_ARMS=('C62_DENSE_POINT_RAW_Q8','C63_DENSE_POINT_WEAK_BRIDGE_RAW_Q8')
PRIOR_DENSE_TAGS=('C62','C63')
PRIOR_DENSE_PRED=OLD213/'frozen_C62_C63_real_predictions_v1/PREDICTION_SEAL.json'
PRIOR_SPATIAL_ARMS=('C60_RAW96_NATIVE_MASK_ZERO_XY_NOMINAL_CONTROL','C61_RAW96_NATIVE_MASK_Q8_XY_NOMINAL')
PRIOR_SPATIAL_TAGS=('C60','C61')
PRIOR_SPATIAL_PRED=OLD203/'frozen_C60_C61_real_predictions_v1/PREDICTION_SEAL.json'
PRIOR_GROUPED_ARMS=('C56_RAW96_DIRECT_END_GROUPED','C57_RAW96_PLUS_FROZEN_SAM_DIRECT_END_GROUPED')
PRIOR_GROUPED_TAGS=('C56','C57')
PRIOR_GROUPED_PRED=OLD195/'frozen_C56_C57_real_predictions_v1/PREDICTION_SEAL.json'
TAGS = ('C64','C65')
PRIOR_DIRECT_ARMS=('C54_RAW96_DIRECT_END','C55_RAW96_PLUS_FROZEN_SAM_DIRECT_END')
PRIOR_DIRECT_TAGS=('C54','C55')
PRIOR_NATIVE_ARMS=('C58_RAW96_DIRECT_RESIDUAL_ZERO_NATIVE_CONTROL','C59_RAW96_DIRECT_RESIDUAL_NATIVE_ONEHOT')
PRIOR_NATIVE_TAGS=('C58','C59')
PRIOR_NATIVE_PRED=OLD200/'frozen_C58_C59_real_predictions_v1/PREDICTION_SEAL.json'
DIRECT_TAGS=PRIOR_DIRECT_TAGS+PRIOR_GROUPED_TAGS+PRIOR_NATIVE_TAGS+PRIOR_SPATIAL_TAGS+PRIOR_DENSE_TAGS+TAGS
PRIOR_DIRECT_PRED=OLD191/'frozen_C54_C55_real_predictions_v1/PREDICTION_SEAL.json'
PRIOR_SIX_COMPLETE=OLD191/'saved_real_head_evaluation_v1/COMPLETE.json'
PRIOR_CONTRACT=OLD191/'evaluation_preparation_a/PROTOCOL_BEFORE_REAL_ANNOTATION_OR_SAVED_SCORE_BODIES.json'
OLD_ARMS=('C52_RAW96_JITTER_PAIRED_SOURCE','C53_RAW96_SAM_JITTER_PAIRED_SOURCE')
REGIONAL_TAGS=('C52','C53')
OLD_TWELVE_TAGS=REGIONAL_TAGS+PRIOR_DIRECT_TAGS+PRIOR_GROUPED_TAGS+PRIOR_NATIVE_TAGS+PRIOR_SPATIAL_TAGS
OLD_TAGS=OLD_TWELVE_TAGS+PRIOR_DENSE_TAGS
ALLTAGS=OLD_TAGS+TAGS
BASELINES = ('v61', 'legacy')
MODELS = BASELINES+ALLTAGS
DOMAINS = {'CURRENT223_1784': (1784, 0, 45), 'PUBLIC_HELD45_360': (360, 584, 108)}
META = ('row_id', 'original_global_index', 'native_slot', 'query_xy', 'L', 'original_query_available')
OWNER = ['TARGET_VISIBLE_SUPPORTED', 'FOREIGN_VISIBLE_SUPPORTED', 'BACKGROUND_SUPPORTED']
OWNER_ORDER = ['TARGET', 'FOREIGN', 'BACKGROUND']
SUPPORT_ORDER = ['visible_end_support', 'foreign_front_support']
UNKNOWN = 'UNKNOWN'
PAIRS = tuple((a,b) for b in ALLTAGS for a in BASELINES)+tuple((a,b) for i,a in enumerate(ALLTAGS) for b in ALLTAGS[i+1:])



def require(value, message):
    if not value:
        raise ValueError(message)


def safe(path):
    # Pure lexical refusal happens before stat/open/hash/resolve.
    p=Path(path).absolute()
    require(P in p.parents and '..' not in p.parts and all('v112' not in x.casefold() for x in p.parts),'Exact safe project path before access')
    return p

def opened(path,create=False):
    p=safe(path);fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            nxt=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nxt
        flags=(os.O_WRONLY|os.O_CREAT|os.O_EXCL) if create else os.O_RDONLY
        last=os.open(p.name,flags|os.O_NOFOLLOW,0o644,dir_fd=fd)
    finally:os.close(fd)
    require(stat.S_ISREG(os.fstat(last).st_mode),'Regular file/no symlink')
    return os.fdopen(last,'w' if create else 'rb')

def sha(path):
    digest=hashlib.sha256()
    with opened(path) as handle:
        for chunk in iter(lambda:handle.read(1<<20),b''):digest.update(chunk)
    return digest.hexdigest()

def bind(path):return dict(path=str(safe(path)),sha256=sha(path))

def check(binding):
    require(bind(binding['path'])==binding,'Frozen binding changed')
    return safe(binding['path'])

def read(path):
    with opened(path) as handle:return json.load(handle)

def lines(path):
    with opened(path) as handle:return [json.loads(line) for line in handle if line.strip()]

def write(path,value):
    with opened(path,create=True) as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write('\n')

def write_lines(path,rows):
    with opened(path,create=True) as handle:
        for row in rows:handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


def equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def extract(path, names, namespace):
    with opened(path) as handle:tree=ast.parse(handle.read().decode('utf-8'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
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
    ns.update(NEW_MODELS=ALLTAGS, PAIRS=PAIRS)
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


def direct_decode(raw,available):
    require(raw.dtype==np.float32 and raw.shape==available.shape and available.dtype==bool,'FP32 direct/original availability')
    require(np.isnan(raw[~available]).all(),'Original inputU direct rawNaN')
    pred=np.full(len(raw),-1,np.int8);good=available&np.isfinite(raw)
    pred[good]=(raw[good]>=0).astype(np.int8)
    return pred

def joint(support):
    result = np.full(len(support), 4, np.int8)
    good = (support >= 0).all(1)
    result[good] = support[good,0]+2*support[good,1]
    return result


def qarray(value):
    return np.array([np.nan, np.nan] if value is None else value, np.float64)


def packet_schema(packet, n, direct=False):
    expected=set(META)|{'raw_support_logits','raw_owner_logits','support_prediction','owner_prediction',
            'regional_support_proxy_D0_E1_Uminus1','support_order','owner_order'}
    if direct:expected|={'raw_direct_end_external_logit','direct_end_prediction_D0_E1_Uminus1','direct_end_order'}
    require(set(packet)==expected,'Exact target-free old13/new16-key saved schema')
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
    if direct:
        require(packet['direct_end_order'].tolist()==['VISIBLE_END0','EXTERNAL_BLOCKED_END1','UNKNOWNminus1'],'Separate direct class order')
        d=direct_decode(packet['raw_direct_end_external_logit'],packet['original_query_available'])
        require(packet['direct_end_prediction_D0_E1_Uminus1'].dtype==np.int8 and equal(d,packet['direct_end_prediction_D0_E1_Uminus1']),'Direct raw>=0/finite/U saved decoder')


def load_packet(binding, rows, domain, reference=None, tag=None):
    with opened(check(binding)) as source_handle:
        with np.load(source_handle, allow_pickle=False) as handle:packet = {k:handle[k].copy() for k in handle.files}
    n, start, original_u = DOMAINS[domain]
    packet_schema(packet, n, tag in DIRECT_TAGS)
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
    packet['primary_prediction']=packet['direct_end_prediction_D0_E1_Uminus1'] if tag in DIRECT_TAGS else packet['regional_support_proxy_D0_E1_Uminus1']
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
                    primary=Counter(int(packet['primary_prediction'][i]) for i in ids)
                    output.append(dict(group=group,error_bin=bucket,human_semantic_state=state,model=tag,N=len(ids),
                        support_NONE=cc[0],support_VISIBLE_ONLY=cc[1],support_FOREIGN_ONLY=cc[2],support_BOTH=cc[3],support_U=cc[4],
                        primary_D=primary[0],primary_E=primary[1],primary_U=primary[-1],proxy_D=pr[0],proxy_E=pr[1],proxy_U=pr[-1],owner_T=own[0],owner_F=own[1],owner_BG=own[2],owner_U=own[-1],
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
    lookup={tag:{core.key(r):int(packets[tag]['primary_prediction'][i]) for i,r in enumerate(truth)} for tag in ALLTAGS}
    original=base.call
    def call(row,model,task,probabilities,threshold,rules=None):
        return core.decision_state(lookup[model][core.key(row)]) if model in lookup else original(row,model,task,probabilities,threshold,rules)
    base.call=call
    adapted={tag:core.decision_rows(core,prior,packets[tag]['primary_prediction']) for tag in ALLTAGS}
    old=[dict(r,raw_physical_external_logit=r['raw_query_aux_hidden_logit']) for r in adapted[TAGS[1]]]
    tables=core.calculate_all(base,truth,prior,adapted[TAGS[0]],old,manual)
    tables['coordinate_error_bins']=extra.error_bins(core,base,truth,manual,adapted[TAGS[0]],old)
    tables['D_vs_external_E_2x3']=extra.matrices(tables['metrics'])
    for av in [tables['availability'],*tables['availability_by_group']]:
        av['C64_direct_total_U']=av.pop('new_total_U'); av['C65_direct_total_U']=av.pop('V91_total_U')
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
            d[tag+'_primary_prediction']=int(packet['primary_prediction'][i])
            if tag in DIRECT_TAGS:
                d[tag+'_raw_direct_end_external_logit']=float(packet['raw_direct_end_external_logit'][i]) if np.isfinite(packet['raw_direct_end_external_logit'][i]) else None
                d[tag+'_direct_end_prediction_D0_E1_Uminus1']=int(packet['direct_end_prediction_D0_E1_Uminus1'][i])
            d[tag+'_primary_route']='DIRECT_END_RAW_GE0' if tag in DIRECT_TAGS else 'UNCHANGED_REGIONAL_E_FIRST_PROXY'
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
    native_panels=[]
    for group in ('ALL',*base.GROUPS):
        for slot in range(8):
            rr=[r for r in truth if (group=='ALL' or r['group']==group) and int(r['native_corner'])==slot and base.yes(r,'paired_same_query_RGB_D_vs_E') and r['human_state'] in (base.D,base.E)]
            for tag in MODELS:native_panels.append(dict(group=group,native_slot=slot,model=tag,**base.calculate(rr,tag,'D_vs_E',{},.5)))
    tables['primary_native8_D_E_with_U']=native_panels
    state_inventory=dict(Counter(r['human_state']for r in truth))
    require(state_inventory.get(base.S,0)==136 and sum(base.yes(r,'paired_same_query_RGB_D_vs_ES')and r['human_state']==base.S for r in truth)==21,'Original self all136 and pairedflag21 are separate from primary D/E1237')
    require(sum(r['human_state']==base.D and core.error_bucket(r)=='LE3'for r in truth)==461,'Original <=3 reference humanD461 distinct from manualnear376')
    prior_twelve=core.readcsv(check(p['prior_twelve_metrics']))
    prior_twelve_exact=exact([r for r in tables['metrics']if r['model']in BASELINES+OLD_TWELVE_TAGS],prior_twelve)
    prior_fourteen=core.readcsv(check(p['prior_fourteen_metrics']))
    prior_fourteen_exact=exact([r for r in tables['metrics']if r['model']in BASELINES+OLD_TAGS],prior_fourteen)
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
        historical_v61_legacy_tables_exact=historical,historical_twelve_metric_rows_exact=prior_twelve_exact,historical_fourteen_metric_rows_exact=prior_fourteen_exact,state_inventory=state_inventory,self_all136=True,self_pairedflag21=True,reference_LE3_humanD461=True,manual_near_D376=True,primary_quality_owner_reference_error_filter=False,
        human_semantic_authority='Existing human D/E/S with unchanged original mapping, missing/ambiguous mapping UNKNOWN; no new mesh recertification blocker.',
        prediction_scope='C64/C65 same spatial-q8 RAW dense-point/weak-bridge source finals primary direct raw>=0E/<0D, original U/nonfinite first. Saved C54/C55/C56/C57/C58/C59 keep direct routes; C52/C53 keep regional E-first. Old support2/owner3 are separate diagnostics; neither ownerF/BG nor binary E establishes physical external/self cause.',
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
    predictions={tag:packets[tag]['owner_prediction'] for tag in ALLTAGS};predictions['v61']=v61
    paired=np.logical_and.reduce([predictions[tag]>=0 for tag in (*ALLTAGS,'v61')])
    masks=dict(FULL360=np.ones(360,bool),KNOWN_CENTER_TRUTH=truth>=0,ALL15_MODEL_PAIRED_AVAILABLE=paired,KNOWN_TRUTH_AND_ALL13_PAIRED_AVAILABLE=(truth>=0)&paired)
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
            association_only=True,no_support_or_native_external_gold=True,no_ownerT_as_crop_classifier_metric=True,primary_direct_vs_crop=k.count_table(np.where(packet['primary_prediction']>=0,packet['primary_prediction'],2),crop,3,3))
    ledger=[]
    for i,r in enumerate(joined):
        row={k:r[k] for k in ('original_global_index','query_id','frame_index','native_slot','source_original_query_available','source_center4_allvalid','source_crop_constructed','center_owner','crop_presence')}
        row['v61_binary_T_nonT_prediction']=int(v61[i])
        for tag,packet in packets.items():
            row[tag+'_owner_prediction']=int(packet['owner_prediction'][i]);row[tag+'_support_prediction']=packet['support_prediction'][i].tolist()
            row[tag+'_regional_support_proxy_D0_E1_Uminus1']=int(packet['regional_support_proxy_D0_E1_Uminus1'][i])
            row[tag+'_primary_prediction']=int(packet['primary_prediction'][i])
        ledger.append(row)
    write_lines(out/'ALL360_PUBLIC_WEAK_LEDGER.jsonl',ledger)
    return dict(completed=True,rows=360,frames=45,sequence='val/000001',cohorts=cohorts,support_crop_center_associations=associations,
        coverage=dict(original_input_U=108,source_available=252,center4_invalid=112,otherwise_available_center4_invalid=4,no_constructed_crop=24,
            known_center204=int((truth>=0).sum()),unknown_center156=int((truth<0).sum()),v61_own_available=int((v61>=0).sum()),
            v61_own_U=int((v61<0).sum()),new_owner_U={tag:int((packets[tag]['owner_prediction']<0).sum()) for tag in ALLTAGS},
            new_regional_U={tag:int((packets[tag]['regional_support_proxy_D0_E1_Uminus1']<0).sum()) for tag in ALLTAGS},new_direct_U={tag:int((packets[tag]['direct_end_prediction_D0_E1_Uminus1']<0).sum()) for tag in DIRECT_TAGS}),
        legacy_baseline=dict(status='MISSING_IN_REVIEWED_LINEAGE',new_forward=False,not_imputed=True),
        reference_authority='Previously sealed prediction-blind assistant weak center-owner/crop-presence consensus only; no support or native external/self gold.',
        repeated_single_sequence_development_exposure=True,fresh_independent_validation=False,training_admission=False)


def validate_prediction_seal(seal,p):
    require(seal.get('completed') is True and seal.get('PASS') is True,'Both new real predictions completed')
    require(seal.get('all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads') is True,'Both predictions before any labels/scores')
    require(seal['scope']==p['scope'] and seal['decoder_spec']==p['decoder_spec'] and seal['input_complete']==p['input_complete'] and seal['original_raw_complete']==p['original_raw_complete'] and seal['spatial_head']==p['spatial_head'] and seal['source_context_adapter']==p['source_context_adapter'],'Exact scope/frozen decoder/original cache/spatial factory/context arithmetic')
    require(seal['source_fit_complete']['path']==p['source_fit_complete_path'],'Exact final source run')
    require(seal['code']==p['prediction_producer_code'] and seal['protocol']==p['prediction_producer_protocol'],'Frozen predictor code/protocol')
    require(set(seal['arms'])==set(ARMS) and seal['original_rows2144']==2144 and seal['model_domain_rows4288']==4288,'Both full domain census')
    require(seal['annotations_reference_GT_oldreal_scores_or_fullRGB_read'] is False and seal['all_batch_repeat2_state_and_input_bytes_exact'] is True,'Closed exposure/repeat/state')
    require(seal['selected_model'] is None and seal['adopted'] is False,'No model choice')
    for arm in ARMS:require(seal['arms'][arm]['updates']==4000 and seal['arms'][arm]['use_sam']is False and type(seal['arms'][arm]['spatial_q8'])is bool and seal['arms'][arm]['spatial_q8']is ARM_SPATIAL[arm] and type(seal['arms'][arm]['weak_bridge'])is bool and seal['arms'][arm]['weak_bridge']is ARM_WEAK[arm][0]and seal['arms'][arm]['weak_loss_weight']==ARM_WEAK[arm][1]and set(seal['arms'][arm]['outputs'])==set(DOMAINS),'Both final4000 fixed RAW/spatial arms all domains')
    return {tag:seal['arms'][arm]['outputs'] for tag,arm in zip(TAGS,ARMS)}

def prior_outputs(seal,p):
    require(seal['completed'] and seal['PASS'] and seal['all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads'],'Old two saved predictions complete')
    require(seal['code']==p['prior_prediction_producer_code'] and seal['protocol']==p['prior_prediction_producer_protocol'] and seal['original_raw_complete']==p['original_raw_complete'],'Exact V184 baseline provenance/input')
    require(set(seal['arms'])==set(OLD_ARMS),'Both old arms retained')
    return {tag:seal['arms'][arm]['outputs'] for tag,arm in zip(REGIONAL_TAGS,OLD_ARMS)}


def prior_direct_outputs(seal,p):
    require(seal['completed'] and seal['PASS'] and seal['all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads'],'Prior direct saved predictions completed')
    require(seal['code']==p['prior_direct_prediction_code'] and seal['protocol']==p['prior_direct_prediction_protocol'] and seal['input_complete']==p['input_complete'] and seal['original_raw_complete']==p['original_raw_complete'],'Exact sealed V191 baseline provenance/input')
    require(set(seal['arms'])==set(PRIOR_DIRECT_ARMS),'Both old direct arms retained')
    return {tag:seal['arms'][arm]['outputs']for tag,arm in zip(PRIOR_DIRECT_TAGS,PRIOR_DIRECT_ARMS)}


def prior_grouped_outputs(seal,p):
    require(seal['completed'] and seal['PASS'] and seal['all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads'],'Prior grouped saved predictions complete')
    require(seal['code']==p['prior_grouped_prediction_code'] and seal['protocol']==p['prior_grouped_prediction_protocol'] and seal['input_complete']==p['input_complete'] and seal['original_raw_complete']==p['original_raw_complete'],'Exact V195 saved-only baseline provenance/input')
    require(set(seal['arms'])==set(PRIOR_GROUPED_ARMS),'Both prior grouped arms retained')
    return {tag:seal['arms'][arm]['outputs']for tag,arm in zip(PRIOR_GROUPED_TAGS,PRIOR_GROUPED_ARMS)}


def prior_native_outputs(seal,p):
    require(seal['completed']and seal['PASS']and seal['all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads'],'Prior native saved predictions complete')
    require(seal['code']==p['prior_native_prediction_code']and seal['protocol']==p['prior_native_prediction_protocol']and seal['input_complete']==p['input_complete']and seal['original_raw_complete']==p['original_raw_complete'],'Exact V200 saved native provenance/input')
    require(set(seal['arms'])==set(PRIOR_NATIVE_ARMS),'Both old native arms retained')
    for i,arm in enumerate(PRIOR_NATIVE_ARMS):
        r=seal['arms'][arm];require(r['updates']==8000 and r['use_sam']is False and type(r['native_condition'])is bool and r['native_condition']is bool(i)and set(r['outputs'])==set(DOMAINS),'Exact old V2008000 RAW/native flags')
    return {tag:seal['arms'][arm]['outputs']for tag,arm in zip(PRIOR_NATIVE_TAGS,PRIOR_NATIVE_ARMS)}


def binding_same(a,b):
 return isinstance(a,dict)and isinstance(b,dict)and all(a.get(k)==b.get(k)for k in ('path','sha256'))
def resolve_root_scope(e,p):
 b=e.get('scope');require(isinstance(b,dict)and isinstance(b.get('sha256'),str)and len(b['sha256'])==64,'Concrete root V220 scope before actual')
 path=safe(b['path']);require(path.parent==V and path.suffix=='.json'and path.name.startswith('ROOT_'),'Exact new V220 administrative scope only');checked(b) if 'checked'in globals()else check(b)
 p=dict(p);p['scope']=b;return p

def validate_source_metadata(fit,review,e,p,source_entry,barrier):
 require(fit['completed']and fit['PASS']and fit['source_only']and fit['from_scratch']and fit['DEV_not_fit']and fit['protected_access']==0 and fit['selected_arm']is None,'Frozen source-only paired fit')
 require(review['completed']and review['PASS']and binding_same(review['fit_complete'],e['fit_complete'])and review['initial_seed1810_state_exact']is True and review['all8000_saved_log_rows_verified']is True,'Independent exact-fit PASS/current initial+draw proof')
 require(binding_same(fit['code'],source_entry['code'])and binding_same(fit['code'],p['source_fit_trainer'])and binding_same(fit['entry'],e['source_fit_entry'])and binding_same(source_entry['protocol'],p['source_fit_protocol']),'Exact source code/entry/protocol')
 require(binding_same(source_entry['kernels']['base_head'],p['head'])and binding_same(source_entry['kernels']['candidate'],p['candidate'])and binding_same(source_entry['kernels']['spatial_head'],p['spatial_head']),'Same C61 RAW118359 factory')
 require(fit['parameters_each']==118359 and fit['steps_each']==4000 and set(fit['arms'])==set(SOURCE_ARM_MAP.values()),'Both fixed source arms4000')
 require(binding_same(fit['all2_freeze'],p['source_both_final_freeze'])and barrier['completed']and barrier['PASS']and barrier['DEV_or_real_output_emitted']is False and set(barrier['checkpoints'])==set(SOURCE_ARM_MAP.values()),'Both final barrier before any DEV outputs')
 for arm,src in SOURCE_ARM_MAP.items():
  r=fit['arms'][src];rr=review['results'][src]
  require(binding_same(barrier['checkpoints'][src],r['checkpoint'])and rr['PASS']and rr['parameters']==118359 and rr['state_tensors']==36 and rr['all_finite']and rr['optimizer_all_steps4000'],'Exact own final checkpoint/source audit')
 return True

def prior_dense_outputs(seal,p):
    require(seal['completed']and seal['PASS']and seal['all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads'],'V213 both saved predictions complete')
    require(seal['code']==p['prior_dense_prediction_code']and seal['protocol']==p['prior_dense_prediction_protocol']and seal['input_complete']==p['input_complete']and seal['original_raw_complete']==p['original_raw_complete'],'Exact unchanged V213 saved provenance')
    require(set(seal['arms'])==set(PRIOR_DENSE_ARMS),'Both old dense arms retained')
    for i,arm in enumerate(PRIOR_DENSE_ARMS):
        r=seal['arms'][arm];require(r['updates']==4000 and r['use_sam']is False and r['spatial_q8']is True and r['weak_bridge']is bool(i)and r['weak_loss_weight']==(0.25 if i else 0.)and set(r['outputs'])==set(DOMAINS),'Old dense roles unchanged')
    return {tag:seal['arms'][arm]['outputs']for tag,arm in zip(PRIOR_DENSE_TAGS,PRIOR_DENSE_ARMS)}

def prior_spatial_outputs(seal,p):
    require(seal['completed']and seal['PASS']and seal['all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads'],'Prior V203 both saved predictions complete')
    require(seal['code']==p['prior_spatial_prediction_code']and seal['protocol']==p['prior_spatial_prediction_protocol']and seal['input_complete']==p['input_complete']and seal['original_raw_complete']==p['original_raw_complete'],'Exact old V203 saved-only provenance')
    require(set(seal['arms'])==set(PRIOR_SPATIAL_ARMS),'Both old spatial arms retained')
    for i,arm in enumerate(PRIOR_SPATIAL_ARMS):
        r=seal['arms'][arm];require(r['updates']==4000 and r['use_sam']is False and r['spatial_q8']is bool(i)and set(r['outputs'])==set(DOMAINS),'Old spatial ablation flags unchanged')
    return {tag:seal['arms'][arm]['outputs']for tag,arm in zip(PRIOR_SPATIAL_TAGS,PRIOR_SPATIAL_ARMS)}

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
    require(len(transitions)==12*len(PAIRS) and all(x['both_correct']==x['N'] for x in transitions),'Fixed paired bothnew/baselines panels')
    passed.append('Both_fixed_new_models_and_v61_legacy_transitions_N_conservation')
    fullprior=[];fulltruth=[]
    for i in range(1784):
        group=base.GROUPS[i%2];state=(base.D,base.E,base.S,'UNKNOWN')[i%4]
        row=dict(id='Invented'+str(i//8),native_corner=str(i%8),valid=i<1739,reason='' if i<1739 else 'INPUT_U',geometry_self_hidden=False,saved_roi_pixel_valid=True,image_sha256='invented',query_x='1',query_y='1',roi_side='8',pixel_x='1',pixel_y='1',group=group,human_state=state,query_valid='True',present_at_fixed_0_5='True',legacy_stored_state='B',paired_same_query_RGB_D_vs_E='True',same_slot_pipeline_D_vs_E='True',paired_same_query_RGB_D_vs_ES='True',same_slot_pipeline_D_vs_ES='True',initial_reference_error_px='3')
        fullprior.append(dict(row));fulltruth.append(dict(row))
    p54=np.zeros(1784,np.int8);p55=np.ones(1784,np.int8);p54[1739:]=-1;p55[1739:]=-1
    nr=core.decision_rows(core,fullprior,p54);orr=[dict(x,raw_physical_external_logit=x['raw_query_aux_hidden_logit']) for x in core.decision_rows(core,fullprior,p55)]
    original=base.call
    def old_call(r,model,task,probabilities,threshold,rules=None):
        if model in OLD_TAGS:return 'VISIBLE' if OLD_TAGS.index(model)%2==0 else 'ABSENT_NONVISIBLE'
        return original(r,model,task,probabilities,threshold,rules)
    base.call=old_call
    try:
        table=core.calculate_all(base,fulltruth,fullprior,nr,orr,[],expected_manual=False)
        require({x['model'] for x in table['metrics']}==set(MODELS) and table['availability']['prior_input_U']==45,'All14 actual metric routes and unchanged originalU')
        require(all(x['TP']+x['FN_decided']+x['U_visible']+x['out_of_task_S_visible']+x['other_unresolved_visible']==x['visible_support'] for x in table['metrics']),'Full known D conservation')
    finally:base.call=original
    passed.append('Actual1784_scalar_metric_adapter_all14_models_preserves_U_and_known_denominators')

    require(public.count_table([0,1,2,3,4],[0,1,2,0,1],5,3)==[[1,0,0],[0,1,0],[0,0,1],[1,0,0],[0,1,0]],'Separate support/crop association')
    passed.append('Joint_BOTH_NONE_U_vs_crop_only_association_no_native_gold_conversion')
    direct=np.array([-np.nextafter(np.float32(0),np.float32(1)),-0.,0.,np.inf,np.nan],np.float32)
    require(direct_decode(direct,np.array([1,1,1,1,0],bool)).tolist()==[0,1,1,-1,-1],'Direct exact raw sign finite/original U')
    passed.append('Direct_zero_negative_subnormal_nonfinite_originalU_and_no_rounded_sigmoid')
    toy16=dict(toy,raw_direct_end_external_logit=direct,direct_end_prediction_D0_E1_Uminus1=direct_decode(direct,av),direct_end_order=np.array(['VISIBLE_END0','EXTERNAL_BLOCKED_END1','UNKNOWNminus1']))
    packet_schema(toy16,5,True);passed.append('Actual_16key_new_direct_and_13key_old_packet_paths_remain_distinct')
    binding=dict(path='/invented/meta.json',sha256='a'*64)
    pp=dict(scope=binding,decoder_spec=binding,spatial_head=binding,source_context_adapter=binding,input_complete=binding,original_raw_complete=binding,source_fit_complete_path=binding['path'],prediction_producer_code=binding,prediction_producer_protocol=binding)
    ss=dict(completed=True,PASS=True,scope=binding,decoder_spec=binding,spatial_head=binding,source_context_adapter=binding,input_complete=binding,original_raw_complete=binding,source_fit_complete=binding,code=binding,protocol=binding,all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads=True,original_rows2144=2144,model_domain_rows4288=4288,annotations_reference_GT_oldreal_scores_or_fullRGB_read=False,all_batch_repeat2_state_and_input_bytes_exact=True,selected_model=None,adopted=False,arms={a:dict(updates=4000,use_sam=False,spatial_q8=ARM_SPATIAL[a],weak_bridge=ARM_WEAK[a][0],weak_loss_weight=ARM_WEAK[a][1],outputs={domain:binding for domain in DOMAINS}) for a in ARMS})
    validate_prediction_seal(ss,pp);rejected=0
    for change in [dict(all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads=False),dict(arms={ARMS[0]:ss['arms'][ARMS[0]]}),dict(input_complete=dict(path='/invented/drift',sha256='b'*64)),dict(annotations_reference_GT_oldreal_scores_or_fullRGB_read=True)]:
        try:validate_prediction_seal(dict(ss,**change),pp)
        except ValueError:rejected+=1
    require(rejected==4,'Actual new both-model pre-label seal adapter fails closed')
    passed.append('Actual_new_prediction_seal_both4000_scope_input_exposure_binding_failclosed')

    # Execute the exact production guard AST with authored 136 S rows.
    # D_vs_E eligibility is false for all; only the first21 D_vs_ES are true.
    with opened(__file__)as handle:tree=ast.parse(handle.read().decode())
    guard=next(n for n in ast.walk(tree)if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Name)and n.value.func.id=='require'and len(n.value.args)==2 and isinstance(n.value.args[1],ast.Constant)and n.value.args[1].value=='Original self all136 and pairedflag21 are separate from primary D/E1237')
    guard_code=compile(ast.fix_missing_locations(ast.Module(body=[guard],type_ignores=[])),str(__file__)+'::EXACT_PRODUCTION_SELF_GUARD','exec')
    fixture=[dict(human_state=base.S,paired_same_query_RGB_D_vs_E='False',paired_same_query_RGB_D_vs_ES=str(i<21))for i in range(136)]
    exec(guard_code,dict(require=require,base=base,state_inventory={base.S:136},truth=fixture))
    passed.append('Actual_SELF_guard_AST_accepts136S_21_DvsES_true_while_all_DvsE_false')
    wrong=[dict(r,paired_same_query_RGB_D_vs_E=str(i<21),paired_same_query_RGB_D_vs_ES='False')for i,r in enumerate(fixture)]
    try:exec(guard_code,dict(require=require,base=base,state_inventory={base.S:136},truth=wrong));ok=False
    except ValueError:ok=True
    require(ok,'D_vs_E must not replace separate SELF D_vs_ES membership')
    passed.append('Actual_SELF_guard_AST_rejects21_DvsE_true_but_zero_DvsES')

    try:safe(P/'blocked_v112'/'unopened');ok=False
    except ValueError:ok=True
    require(ok,'Lexical forbidden name before path access');passed.append('Forbidden_name_lexical_precheck_no_stat_open_hash_resolve')
    with tempfile.TemporaryDirectory(prefix='invented_path_guard_',dir=H) as td:
        d=Path(td);a=d/'ordinary.txt';a.write_text('invented only');link=d/'alias.txt';link.symlink_to(a)
        with opened(a) as handle:require(handle.read()==b'invented only','Ordinary explicit authored file')
        passed.append('Actual_NOFOLLOW_reader_accepts_authored_regular_file')
        try:
            with opened(link) as handle:handle.read()
        except OSError:ok=True
        else:ok=False
        require(ok,'Symlink refused before body');passed.append('Actual_NOFOLLOW_reader_refuses_symlink_without_target_read')
    require(len(MODELS)==14 and len(BASELINES+OLD_TAGS)==12 and len(ALLTAGS)==12 and len(DIRECT_TAGS)==10,'All14 current/old12/new2; public weak13 without missinglegacy')
    passed.append('All14_current_old12_direct10_and_separate_public13_model_contract')
    wrong_native=dict(ss,arms={a:dict(v)for a,v in ss['arms'].items()});wrong_native['arms'][ARMS[1]]['spatial_q8']=False
    wrong_SAM=dict(ss,arms={a:dict(v)for a,v in ss['arms'].items()});wrong_SAM['arms'][ARMS[0]]['use_sam']=True
    for wrong in (wrong_native,wrong_SAM):
        try:validate_prediction_seal(wrong,pp);ok=False
        except ValueError:ok=True
        require(ok,'Fixed spatial condition and RAW flags must not be defaulted')
    passed.append('Actual_new_seal_validator_rejects_spatialflag_or_SAM_drift')
    prior_contract=dict(prior_grouped_prediction_code=binding,prior_grouped_prediction_protocol=binding,input_complete=binding,original_raw_complete=binding)
    grouped=dict(completed=True,PASS=True,all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads=True,code=binding,protocol=binding,input_complete=binding,original_raw_complete=binding,arms={a:dict(outputs={d:binding for d in DOMAINS})for a in PRIOR_GROUPED_ARMS})
    require(set(prior_grouped_outputs(grouped,prior_contract))==set(PRIOR_GROUPED_TAGS),'Both saved V195 arms routed exactly')
    try:prior_grouped_outputs(dict(grouped,arms={PRIOR_GROUPED_ARMS[0]:grouped['arms'][PRIOR_GROUPED_ARMS[0]]}),prior_contract);ok=False
    except ValueError:ok=True
    require(ok,'No missing old grouped arm');passed.append('Actual_prior_V195_savedonly_provenance_and_botharms_guard')
    # Execute the actual old12 metric filter/guard AST on authored records.
    metric_guard=next(n for n in ast.walk(tree)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='prior_twelve_exact'for t in n.targets))
    metric_code=compile(ast.fix_missing_locations(ast.Module(body=[metric_guard],type_ignores=[])),str(__file__)+'::EXACT_OLD12_METRIC_GUARD','exec')
    allmetrics=[dict(model=m,N='1',TP='1')for m in MODELS];oldmetrics=[dict(r)for r in allmetrics if r['model']in BASELINES+OLD_TAGS]
    exact=kernels()[3];ns=dict(tables=dict(metrics=allmetrics),prior_twelve=oldmetrics,BASELINES=BASELINES,OLD_TAGS=OLD_TAGS,exact=exact)
    exec(metric_code,ns)
    drift=[dict(r,TP='2')if r['model']=='C59'else dict(r)for r in oldmetrics]
    try:exec(metric_code,dict(ns,prior_twelve=drift));ok=False
    except ValueError:ok=True
    require(ok,'Old12 saved metric drift must fail');passed.append('Actual_old12_metric_guard_AST_accepts_equal_and_rejects_C59_drift')
    prior_native_contract=dict(prior_native_prediction_code=binding,prior_native_prediction_protocol=binding,input_complete=binding,original_raw_complete=binding)
    prior_native=dict(completed=True,PASS=True,all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads=True,code=binding,protocol=binding,input_complete=binding,original_raw_complete=binding,arms={a:dict(updates=8000,use_sam=False,native_condition=bool(i),outputs={d:binding for d in DOMAINS})for i,a in enumerate(PRIOR_NATIVE_ARMS)})
    require(set(prior_native_outputs(prior_native,prior_native_contract))==set(PRIOR_NATIVE_TAGS),'Both V200 saved native arms joined')
    bad=dict(prior_native,arms={a:dict(r)for a,r in prior_native['arms'].items()});bad['arms'][PRIOR_NATIVE_ARMS[1]]['native_condition']=False
    try:prior_native_outputs(bad,prior_native_contract);ok=False
    except ValueError:ok=True
    require(ok,'Old V200 native flag drift refused');passed.append('Actual_prior_V200_savedonly_bothnative_flags_provenance_guard')
    return dict(completed=True,PASS=True,invented_only=True,checks=passed,count=len(passed),
                actual_real_annotation_old_or_new_scores_RGB_features_weights_body_or_hash_reads=0,model_forward_fit_threshold_selection=0)


def prepare():
    raise RuntimeError('Use frozen R preparation metadata; actual root scope/entry required. No protected bodies here.')

def run(entry_path):
    e=read(entry_path)
    require(all(e.get(k) is True for k in ('approved_saved_real_evaluation','all_two_real_predictions_frozen','evaluation_only',
            'no_model_data_architecture_threshold_selection','no_forward_fit_RGB_features_weights_PnP_qL_or_V112')),'Separate root saved-only evaluation entry')
    require(e['code']==bind(__file__) and e['protocol']==bind(PROTOCOL)
            and e['output']==str(OUT),'Frozen evaluator/root scope')
    require(platform.python_version()=='3.10.20' and np.__version__=='1.26.4','Pinned CPU runtime')
    p=resolve_root_scope(e,read(PROTOCOL));require(read(check(p['checks']))['PASS'] and read(PREP)['PASS'],'Invented checks/preparation passed')
    require(e['prediction_seal']['path']==str(PRED),'Exact both-real prediction seal path')
    seal=read(check(e['prediction_seal']));outputs=validate_prediction_seal(seal,p)
    p['source_fit_protocol']=seal['source_fit_protocol']
    # No real annotation, old baseline score or new prediction body has been read above.
    for b in p['operator_codes']+[p['original_raw_complete'],p['input_complete'],p['source_fit_protocol'],p['spatial_head'],p['source_context_adapter'],p['transitive_code_addendum'],p['prediction_producer_code'],p['prediction_producer_protocol'],p['prediction_producer_preparation']]:check(b)
    require(e['source_fit_complete']==seal['source_fit_complete'] and e['independent_source_fit_review']==seal['source_fit_integrity_review'],'Independent source fit closure')
    fit=read(check(e['source_fit_complete']));review=read(check(e['independent_source_fit_review']))
    source_entry=read(check(seal['source_fit_entry']));barrier=read(check(seal['source_both_final_freeze']))
    p.update(source_fit_trainer=seal['source_fit_trainer'],source_fit_protocol=seal['source_fit_protocol'],source_both_final_freeze=seal['source_both_final_freeze'])
    validate_source_metadata(fit,review,dict(fit_complete=e['source_fit_complete'],source_fit_entry=seal['source_fit_entry']),p,source_entry,barrier)
    for arm,src in SOURCE_ARM_MAP.items():require(binding_same(seal['arms'][arm]['checkpoint'],fit['arms'][src]['checkpoint'])and seal['arms'][arm]['source_final_state_sha']==fit['arms'][src]['final'],'Exact V219 source state/no weight body')
    check(p['current']['annotation_authority']);check(p['public']['definition'])
    require(not OUT.exists(),'Single fixed new output; no overwrite/retry')
    OUT.mkdir();write(OUT/'BEFORE_ANNOTATION_AND_SAVED_SCORE_READ.json',dict(completed=True,entry=bind(entry_path),prediction_seal=e['prediction_seal'],both_models_closed=True,annotation_or_score_bodies_read_so_far=False))
    try:
        require(e['prior_prediction_seal']['path']==str(PRIOR_PRED),'Exact V184 prior seal path')
        outputs.update(prior_outputs(read(check(e['prior_prediction_seal'])),p))
        require(e['prior_direct_prediction_seal']['path']==str(PRIOR_DIRECT_PRED),'Exact V191 saved direct seal');outputs.update(prior_direct_outputs(read(check(e['prior_direct_prediction_seal'])),p))
        require(e['prior_grouped_prediction_seal']['path']==str(PRIOR_GROUPED_PRED),'Exact V195 saved grouped seal');outputs.update(prior_grouped_outputs(read(check(e['prior_grouped_prediction_seal'])),p))
        require(e['prior_native_prediction_seal']['path']==str(PRIOR_NATIVE_PRED),'Exact V200 saved native seal');outputs.update(prior_native_outputs(read(check(e['prior_native_prediction_seal'])),p))
        require(e['prior_spatial_prediction_seal']['path']==str(PRIOR_SPATIAL_PRED),'Exact V203 saved spatial seal');outputs.update(prior_spatial_outputs(read(check(e['prior_spatial_prediction_seal'])),p))
        require(e['prior_dense_prediction_seal']['path']==str(PRIOR_DENSE_PRED),'Exact V213 old dense seal');outputs.update(prior_dense_outputs(read(check(e['prior_dense_prediction_seal'])),p))
        require(e['prior_fourteen_evaluation_complete']['path']==p['prior_fourteen_evaluation_complete_path'],'Exact V213 old14 evaluation')
        prior14=read(check(e['prior_fourteen_evaluation_complete']));require(prior14['completed']and prior14['PASS']and prior14['code']==p['prior_fourteen_evaluation_code']and prior14['protocol']==p['prior_fourteen_evaluation_protocol'],'Exact prior14 evaluation closure');p['current']['prior_fourteen_metrics']=prior14['outputs']['current1784/METRICS.csv']
        require(e['prior_twelve_evaluation_complete']['path']==p['prior_twelve_evaluation_complete_path'],'Exact V203 old12 evaluation');p['prior_twelve_evaluation_complete']=e['prior_twelve_evaluation_complete']
        prior_twelve=read(check(p['prior_twelve_evaluation_complete']));require(prior_twelve['completed']and prior_twelve['PASS']and prior_twelve['code']==p['prior_twelve_evaluation_code']and prior_twelve['protocol']==p['prior_twelve_evaluation_protocol'],'Prior twelve saved evaluation closure');p['current']['prior_twelve_metrics']=prior_twelve['outputs']['current1784/METRICS.csv']
        packets={};metadata={}
        for domain in DOMAINS:
            metadata[domain]=lines(check(p['domains'][domain]['metadata']));packets[domain]={};reference=None
            for tag in ALLTAGS:
                packet=load_packet(outputs[tag][domain],metadata[domain],domain,reference,tag)
                packets[domain][tag]=packet;reference=packet
        current_out=OUT/'current1784';current_out.mkdir();public_out=OUT/'publicheld360';public_out.mkdir()
        current=evaluate_current(p['current'],packets['CURRENT223_1784'],metadata['CURRENT223_1784'],current_out)
        public=evaluate_public(p['public'],packets['PUBLIC_HELD45_360'],metadata['PUBLIC_HELD45_360'],public_out)
        write(current_out/'CURRENT_HUMAN_SEMANTIC_AND_SEPARATE_ASSOCIATIONS.json',current)
        write(public_out/'PUBLIC_WEAK_OWNER_AND_SEPARATE_SUPPORT_ASSOCIATIONS.json',public)
        for tag in ALLTAGS:
            for b in outputs[tag].values():check(b)
        # Seal all actually consumed annotation/baseline bindings unchanged after evaluation.
        for k in ('prior','truth','manual','historical_original','frozen_annotation_ledger'):check(p['current'][k])
        for b in p['current']['historical_three_tables'].values():check(b)
        for k in ('inputs','consensus','v61'):check(p['public'][k])
        for b in (p['scope'],p['authority_clarification'],p['original_raw_complete'],e['prediction_seal']):check(b)
        files={str(path.relative_to(OUT)):bind(path) for path in sorted(OUT.rglob('*')) if path.is_file()}
        complete=dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),entry=bind(entry_path),prediction_seal=e['prediction_seal'],
            outputs=files,models=list(MODELS),current_all1784=True,public_all360=True,new_model_domain_rows4288=True,old_C52_to_C61_saved_only_rows21440=True,
            primary_current_D1147_E90_N1237=True,current_human_semantic_authority_valid=True,primary_quality_owner_error_gate=False,
            public_reference_weak_only=True,public_legacy_missing=True,all_both_predictions_closed_before_any_annotation_or_score_body=True,
            repeated_development_exposure_not_fresh_validation=True,new_forward_fit_RGB_features_weights_threshold_or_model_data_choice=0,
            old_annotations_outputs_and_inputs_unchanged=True,old_twelve_primary_metrics_exact=True,old_fourteen_primary_metrics_exact=True,new_training_admission=False,selected_model=None,adopted=False,fully_solved=False,automatic_retry=False)
        write(OUT/'COMPLETE.json',complete)
        print(json.dumps(dict(completed=True,PASS=True,complete=bind(OUT/'COMPLETE.json'))))
    except BaseException:
        write(OUT/'FAILED.json',dict(completed=False,PASS=False,traceback=traceback.format_exc(),automatic_retry=False))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--entry',type=Path)
    args=parser.parse_args();prepare() if args.prepare else run(args.entry)
