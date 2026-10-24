"""Independent saved-only V178 arithmetic audit. No evaluator/helper imports."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import math
import numpy as np

V = Path(__file__).resolve().parents[1]
P = V.parent
OUT = Path(__file__).resolve().parent
E = V / 'saved_five_model_evaluation_v1'
MODELS = ('v61', 'legacy', 'C47', 'C48', 'C49')
GROUPS = ('WOOD_PRIMARY105', 'SQUARE118')
D, X, S = 'DIRECT_VISIBLE', 'EXTERNAL_OCCLUDED', 'SELF_OCCLUDED'
COHORTS = ('paired_same_query_RGB', 'same_slot_pipeline')
failures, checks = [], Counter()
reads = {}


def check(ok, label, detail=None):
    checks[label] += 1
    if not bool(ok):
        failures.append(dict(check=label, detail=detail))


def digest(path):
    path = Path(path)
    assert '_v112' not in str(path).lower()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding(path):
    return dict(path=str(Path(path).resolve()), sha256=digest(path))


def read(path):
    reads[str(Path(path).resolve())] = digest(path)
    return json.loads(Path(path).read_text())


def csvread(path):
    reads[str(Path(path).resolve())] = digest(path)
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def equal(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, float)) and not isinstance(a, bool):
        return isinstance(b, (int, float)) and (a == b or (math.isnan(a) and math.isnan(b)) or abs(a-b) <= 1e-12)
    return a == b


def numeric(a, b):
    try:
        return equal(float(a), float(b))
    except (TypeError, ValueError):
        return a in (None, '', 'None', 'nan') and b in (None, '', 'None', 'nan')


def compare(got, expected, label):
    for k, value in expected.items():
        check(k in got and equal(value, got.get(k)), label, dict(field=k, expected=value, actual=got.get(k)))


def yes(v):
    return v is True or v == 'True'


def ratio(a, b):
    return a / b if b else None


def key(r):
    return r['id'], int(r['native_corner'])


def bucket(r):
    try:
        x = float(r['initial_reference_error_px'])
    except (TypeError, ValueError):
        x = float('nan')
    return 'MISSING_REFERENCE_ERROR' if not math.isfinite(x) else 'LE3' if x <= 3 else 'GT3'


protocol = read(V/'evaluation_preparation/PROTOCOL_BEFORE_SAVED_CURRENT_SCORE_OR_LABEL_BODIES.json')
predseal = read(V/'frozen_all3_direct_owner_predictions_v1/PREDICTION_SEAL.json')
eval_seal = read(E/'EVALUATION_SEAL.json')
check(eval_seal['completed'] and eval_seal['PASS'], 'completed_evaluation_barrier')
truth = csvread(protocol['truth']['path'])
manual = csvread(protocol['manual']['path'])
prior_path = Path(protocol['prior']['path'])
reads[str(prior_path)] = digest(prior_path)
prior = [json.loads(s) for s in prior_path.read_text().splitlines() if s]
for name in ('truth', 'manual', 'prior'):
    check(reads[str(Path(protocol[name]['path']).resolve())] == protocol[name]['sha256'], 'bound_original_metadata_sha')
check(len(truth) == len(prior) == 1784 and len(set(map(key, truth))) == 1784, 'original1784_unique_keys')
check(list(map(key, truth)) == list(map(key, prior)), 'original_truth_prior_row_order')
manual_by_key = {key(r): r for r in manual}

packets, owners, binaries, raw_values = {}, {}, {}, {}
for tag, arm in zip(MODELS[2:], predseal['models']):
    bp = predseal['models'][arm]['outputs']['raw_packet']
    path = Path(bp['path'])
    reads[str(path)] = digest(path)
    check(reads[str(path)] == bp['sha256'], 'sealed_prediction_packet_sha')
    with np.load(path, allow_pickle=False) as f:
        packet = {k: f[k].copy() for k in f.files}
    packets[tag] = packet
    logits = packet['raw_owner_three_logits']
    check(logits.dtype == np.float32 and logits.shape == (1784, 3), 'raw_FP32_schema')
    check(packet['owner_prediction'].dtype == np.int8 and packet['prediction_D0_E1_Uminus1'].dtype == np.int8, 'saved_int8_prediction_schema')
    check(packet['owner_class_order'].tolist() == ['TARGET_VISIBLE_SUPPORTED', 'FOREIGN_VISIBLE_SUPPORTED', 'BACKGROUND_SUPPORTED'], 'fixed_owner_class_order')
    check(packet['global_index'].dtype == np.int32 and packet['global_index'].tolist() == list(range(1784)), 'original_global_indices')
    oo, bb = [], []
    for i, row in enumerate(truth):
        avail = bool(packet['query_available'][i])
        vals = [float(logits[i, j]) for j in range(3)]
        if not avail:
            check(all(math.isnan(x) for x in vals), 'U_raw_all_NaN')
            owner, binary = -1, -1
        else:
            check(all(math.isfinite(x) for x in vals), 'available_raw_all_finite')
            owner = 0
            for j in (1, 2):
                if vals[j] > vals[owner]:
                    owner = j
            binary = 0 if owner == 0 else 1
        oo.append(owner); bb.append(binary)
        check(int(packet['owner_prediction'][i]) == owner and int(packet['prediction_D0_E1_Uminus1'][i]) == binary, 'all3_scalar_argmax_binary_U', [tag, i])
        check((str(packet['id'][i]), int(packet['native_corner'][i])) == key(row), 'all3_original_native_key', [tag, i])
        pr = prior[i]
        check(bool(packet['query_available'][i]) == bool(packet['valid'][i]) == yes(pr['valid']), 'all3_original_input_availability', [tag, i])
        check(bool(packet['valid'][i]) == (bool(packet['prior_valid'][i]) and bool(packet['context_available'][i]) and bool(packet['local_central4_available'][i])), 'frozen_private_availability_conjunction', [tag, i])
        for j, qname in enumerate(('query_x', 'query_y')):
            check(numeric(packet['original_query_xy'][i,j], pr[qname]), 'original_float64_q', [tag, i, qname])
            check(np.float32(packet['original_query_xy'][i,j]).tobytes() == packet['query_xy'][i,j].tobytes(), 'original_q_fp32_alias', [tag, i, qname])
        check(numeric(packet['original_side_L'][i], pr['roi_side']), 'original_float64_L', [tag, i])
        check(np.float32(packet['original_side_L'][i]).tobytes() == packet['side_L'][i].tobytes(), 'original_L_fp32_alias', [tag, i])
    owners[tag], binaries[tag], raw_values[tag] = oo, bb, logits
    check(oo.count(-1) == 45 and bb.count(-1) == 45, 'each_new_model_U45')
    for name in protocol['original_metadata_whitelist']:
        a = packet[name]
        fp = dict(dtype=str(a.dtype), shape=list(a.shape), bytes_sha256=hashlib.sha256(a.tobytes()).hexdigest())
        check(fp == predseal['original_input_array_fingerprints'][name], 'all12_metadata_original_fingerprints', [tag, name])
        if tag != 'C47':
            b = packets['C47'][name]
            check(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), 'all3_metadata_byte_parity', [tag, name])


def response(i, model, task='D_vs_E', inventory=False):
    r = truth[i]
    if model in binaries:
        b = binaries[model][i]
        return 'OTHER_UNRESOLVED' if b == -1 else 'VISIBLE' if b == 0 else 'ABSENT_NONVISIBLE'
    if model == 'v61':
        if inventory and (r['query_valid'] != 'True' or r['present_at_fixed_0_5'] not in ('True','False')):
            return 'OTHER_UNRESOLVED'
        return 'VISIBLE' if r['present_at_fixed_0_5'] == 'True' else 'ABSENT_NONVISIBLE'
    state = r['legacy_stored_state']
    if state in ('B', 'I'):
        return 'VISIBLE'
    if state == 'O' or (state == 'S' and task != 'D_vs_E'):
        return 'ABSENT_NONVISIBLE'
    return 'OUT_OF_TASK_GEOMETRY_S' if state == 'S' else 'ABSTAIN' if state == 'U' else 'OTHER_UNRESOLVED'


def indices(row):
    states = (D, X) if row['task'] == 'D_vs_E' else (D, S) if row['task'] == 'D_vs_S' else (D, X, S)
    flag = row['cohort']+'_'+('D_vs_ES' if row['task'] == 'D_vs_S' else row['task'])
    return [i for i,r in enumerate(truth) if r['group'] == row['group'] and r[flag] == 'True' and r['human_state'] in states]


def calculate(ii, model, task):
    cells = [[0]*5 for _ in range(2)]
    order = ('VISIBLE','ABSENT_NONVISIBLE','ABSTAIN','OUT_OF_TASK_GEOMETRY_S','OTHER_UNRESOLVED')
    for i in ii:
        truth_row = 0 if truth[i]['human_state'] == D else 1
        cells[truth_row][order.index(response(i,model,task))] += 1
    tp,fn,up,sp,op = cells[0]; fp,tn,un,sn,on = cells[1]
    n, pos, neg = len(ii), sum(cells[0]), sum(cells[1])
    return dict(N=n, frames=len({truth[i]['id'] for i in ii}), visible_support=pos, occluded_support=neg,
        TP=tp,FN_decided=fn,FP=fp,TN=tn,U_visible=up,U_occluded=un,out_of_task_S_visible=sp,out_of_task_S_occluded=sn,
        other_unresolved_visible=op,other_unresolved_occluded=on,
        visible_precision=ratio(tp,tp+fp),visible_recall_all_GT=ratio(tp,pos),occluded_precision=ratio(tn,tn+fn),occluded_recall_all_GT=ratio(tn,neg),
        abstention_rate=ratio(up+un,n),out_of_task_S_rate=ratio(sp+sn,n),unresolved_rate=ratio(up+un+sp+sn+op+on,n),
        decision_coverage=ratio(tp+fn+fp+tn,n),correct_fraction_all_GT=ratio(tp+tn,n),selective_accuracy=ratio(tp+tn,tp+fn+fp+tn))


# Derivation above is source-first. Only now load the saved ledger and numeric tables for comparisons.
ledger = csvread(E/'ALL1784_NATIVE_LEDGER.csv')
results = read(E/'CURRENT_FROZEN_BENCHMARK_RESULTS.json')
tables = results['tables']
check(len(ledger) == 1784 and list(map(key, ledger)) == list(map(key, truth)), 'ledger1784_order')
for i,(row,r) in enumerate(zip(ledger,truth)):
    compare(row, {k:r[k] for k in ('id','native_corner','group','human_state','mapping_status','human_reference_x','human_reference_y','paired_same_query_RGB_D_vs_E','same_slot_pipeline_D_vs_E','initial_reference_error_px')}, 'ledger_original_metadata')
    check(int(row['global_index']) == i and row['original_query_valid'] == r['query_valid'] and row['legacy_original_state'] == r['legacy_stored_state'], 'ledger_original_flags')
    check(yes(row['query_available']) == bool(packets['C47']['valid'][i]), 'ledger_original_input_U')
    for field, value in (('query_x', prior[i]['query_x']), ('query_y', prior[i]['query_y']), ('roi_side', prior[i]['roi_side'])):
        check(numeric(row[field],value), 'ledger_qL_original', [i,field])
    for model in MODELS:
        rr = response(i,model,inventory=True)
        check(row[model+'_original_response'] == rr and row[model+'_D_E_U'] == ('D' if rr=='VISIBLE' else 'E' if rr=='ABSENT_NONVISIBLE' else 'U'), 'ledger_all5_responses', [model,i])
    for model in MODELS[2:]:
        check(int(row[model+'_owner_prediction']) == owners[model][i] and row[model+'_owner_T_F_BG_U'] == ('U' if owners[model][i]<0 else ('T','F','BG')[owners[model][i]]), 'ledger_owner_argmax', [model,i])
        for j,cls in enumerate(('T','F','BG')):
            check(numeric(row[model+'_raw_owner_'+cls+'_logit'],float(raw_values[model][i,j])), 'ledger_raw_FP32_values', [model,i,j])

for row in tables['metrics']:
    compare(row, calculate(indices(row),row['model'],row['task']), 'all50_metric_rows')
for row in tables['coordinate_error_bins']:
    ii = [i for i in indices(row) if bucket(truth[i]) == row['error_bin']]
    compare(row,calculate(ii,row['model'],row['task']), 'all60_error_bin_metrics')
    manual_n = sum(manual_by_key.get(key(truth[i]),{}).get('annotation_coordinate_source') == 'manual_click' for i in ii)
    compare(row,dict(manual_click_reference_N=manual_n,other_or_unspecified_reference_N=len(ii)-manual_n), 'error_bin_provenance_counts')
for row in tables['confusion']:
    ii = [i for i in indices(row) if truth[i]['human_state']==row['human_state']]
    cc = Counter(response(i,row['model'],row['task']) for i in ii)
    compare(row,dict(N=len(ii),pred_visible=cc['VISIBLE'],pred_binary_hidden=cc['ABSENT_NONVISIBLE'],legacy_U=cc['ABSTAIN'],out_of_task_geometry_S=cc['OUT_OF_TASK_GEOMETRY_S'],input_or_operator_U=cc['OTHER_UNRESOLVED']), 'all120_confusion_rows')
for row in tables['human_self_binary_response']:
    ii = [i for i,r in enumerate(truth) if r['group']==row['group'] and r[row['cohort']+'_D_vs_ES']=='True' and r['human_state']==S]
    cc = Counter(response(i,row['model'],'D_vs_S') for i in ii)
    compare(row,dict(human_S_support=len(ii),binary_hidden_response=cc['ABSENT_NONVISIBLE'],visible_response=cc['VISIBLE'],unresolved=len(ii)-cc['ABSENT_NONVISIBLE']-cc['VISIBLE'],human_S_binary_hidden_response_rate=ratio(cc['ABSENT_NONVISIBLE'],len(ii)),learned_self_prediction=False,external_vs_self_identified=False), 'all20_self_response_rows')
manual_near = [r for r in manual if r['annotation_coordinate_source']=='manual_click' and r['human_state']==D and float(r['initial_reference_error_px'])<=3 and r['query_valid']=='True']
truth_idx = {key(r):i for i,r in enumerate(truth)}
for r in manual_near:
    t = truth[truth_idx[key(r)]]
    check(all(r[k]==t[k] for k in ('image_sha256','query_x','query_y')), 'manual_near_original_metadata_join')
for row in tables['manual_near_D']:
    ii = [truth_idx[key(r)] for r in manual_near if r['group']==row['group']]
    cc = Counter(response(i,row['model'],'D_vs_ES') for i in ii)
    compare(row,dict(N=len(ii),pred_visible=cc['VISIBLE'],pred_binary_hidden=cc['ABSENT_NONVISIBLE'],unresolved=len(ii)-cc['VISIBLE']-cc['ABSENT_NONVISIBLE'],semantic_D_recall=ratio(cc['VISIBLE'],len(ii))), 'all10_manual_near_rows')

for row in tables['D_vs_external_E_2x3']:
    spec=dict(row,task='D_vs_E')
    ii=indices(spec); v=calculate(ii,row['model'],'D_vs_E')
    compare(row,dict(human_D_N=v['visible_support'],human_external_E_N=v['occluded_support'],D_pred_D=v['TP'],D_pred_hidden=v['FN_decided'],D_pred_U=v['U_visible']+v['out_of_task_S_visible']+v['other_unresolved_visible'],E_pred_D=v['FP'],E_pred_hidden=v['TN'],E_pred_U=v['U_occluded']+v['out_of_task_S_occluded']+v['other_unresolved_occluded'],D_precision=v['visible_precision'],D_recall=v['visible_recall_all_GT'],external_E_response_precision=v['occluded_precision'],external_E_response_recall=v['occluded_recall_all_GT'],unresolved_rate=v['unresolved_rate'],hidden_response_not_learned_S=True), 'all20_D_E_2x3_rows')

for row in tables['paired_recovery_and_damage']:
    ii=[i for i in indices(dict(row,task='D_vs_E')) if row['truth_class']=='ALL_DE' or truth[i]['human_state']==row['truth_class']]
    cc=Counter()
    for i in ii:
        a,b=response(i,row['baseline']),response(i,row['new_model'])
        correct='VISIBLE' if truth[i]['human_state']==D else 'ABSENT_NONVISIBLE'
        if b not in ('VISIBLE','ABSENT_NONVISIBLE'): label='new_unresolved'
        elif a not in ('VISIBLE','ABSENT_NONVISIBLE'): label='baseline_unresolved_new_correct' if b==correct else 'baseline_unresolved_new_wrong'
        elif a==correct: label='both_correct' if b==correct else 'baseline_correct_new_wrong'
        else: label='baseline_wrong_new_correct' if b==correct else 'both_wrong'
        cc[label]+=1
    expected={k:cc[k] for k in ('both_correct','baseline_correct_new_wrong','baseline_wrong_new_correct','both_wrong','baseline_unresolved_new_correct','baseline_unresolved_new_wrong','new_unresolved')}
    check(sum(expected.values())==len(ii),'transition_partition_conservation')
    compare(row,dict(N=len(ii),**expected),'all108_transition_rows')

for row in tables['owner_vs_native_visibility_and_reference_error']:
    ii=[]
    for i,r in enumerate(truth):
        state=r['human_state'] if r['human_state'] in (D,X,S) else 'UNKNOWN'
        if (row['group']=='ALL' or r['group']==row['group']) and (row['error_bin']=='ALL' or bucket(r)==row['error_bin']) and state==row['original_native_visibility']: ii.append(i)
    cc=Counter(owners[row['model']][i] for i in ii)
    compare(row,dict(N=len(ii),owner_T=cc[0],owner_F=cc[1],owner_BG=cc[2],owner_U=cc[-1],original_input_U=sum(not bool(packets['C47']['valid'][i]) for i in ii)), 'all144_owner_native_error_cells')
    check(json.loads(row['raw_original_state_inventory']) == dict(Counter(truth[i]['human_state'] for i in ii)), 'owner_native_original_state_inventory')
    check(row['N']==row['owner_T']+row['owner_F']+row['owner_BG']+row['owner_U'], 'owner_row_conservation')
for model in MODELS[2:]:
    for group in ('ALL',*GROUPS):
        rr=[r for r in tables['owner_vs_native_visibility_and_reference_error'] if r['model']==model and r['group']==group]
        n=sum(1 for r in truth if group=='ALL' or r['group']==group)
        check(sum(r['N'] for r in rr if r['error_bin']=='ALL') == n == sum(r['N'] for r in rr if r['error_bin']!='ALL'), 'owner_cross_bin_partition_conservation')

for row in tables['all1784_model_inventory']:
    ii=[i for i,r in enumerate(truth) if row['group']=='ALL' or row['group']==r['group']]
    cc=Counter(response(i,row['model'],inventory=True) for i in ii)
    uu=len(ii)-cc['VISIBLE']-cc['ABSENT_NONVISIBLE']
    compare(row,dict(all_registered_slots=len(ii),D=cc['VISIBLE'],E_binary_response=cc['ABSENT_NONVISIBLE'],U=uu,U_rate=ratio(uu,len(ii)),legacy_abstain=cc['ABSTAIN'],out_of_task_geometry_S=cc['OUT_OF_TASK_GEOMETRY_S'],other_unresolved=cc['OTHER_UNRESOLVED']), 'all15_model_inventory_rows')
    check(json.loads(row['human_state_inventory']) == dict(Counter(truth[i]['human_state'] for i in ii)), 'model_inventory_state_conservation')
compare(tables['availability'],dict(rows=1784,prior_input_U=45,extra_operator_U=0,confidence_abstention_trained=False,C47_total_U=45,C48_total_U=45),'availability_summary')
for row in tables['availability_by_group']:
    ii=[i for i,r in enumerate(truth) if r['group']==row['group']]
    uu=sum(not yes(prior[i]['valid']) for i in ii)
    compare(row,dict(all_native_slots=len(ii),prior_input_U=uu,extra_operator_U=0,C47_total_U=uu,C48_total_U=uu),'availability_by_group')

# Verify each serialized numeric table matches the JSON table cell for cell.
csv_table_counts={}
for name, rows in tables.items():
    if not isinstance(rows,list): continue
    saved=csvread(E/(name.upper()+'.csv'))
    check(len(saved)==len(rows),'CSV_JSON_row_count',name)
    for a,b in zip(saved,rows):
        check(set(a)==set(b),'CSV_JSON_fields',name)
        for field,value in b.items():
            expected='' if value is None else str(value)
            check(a.get(field)==expected,'CSV_JSON_exact_cells',[name,field])
    csv_table_counts[name]=len(rows)

old_dir=P/'query_visibility_frozen_public_threshold_current_benchmark_20261024_v169/saved_seven_condition_evaluation_v1'
oldledger=csvread(old_dir/'ALL1784_NATIVE_LEDGER.csv')
check(len(oldledger)==1784,'original_V169_ledger1784')
old_shared=('id','native_corner','group','human_state','mapping_status','query_x','query_y','roi_side','query_available','paired_same_query_RGB_D_vs_E','same_slot_pipeline_D_vs_E','initial_reference_error_px','v61_original_response','v61_D_E_U','legacy_original_response','legacy_D_E_U')
for i,(a,b) in enumerate(zip(ledger,oldledger)):
    check(all(a[k]==b[k] for k in old_shared), 'V169_original_rows_and_baseline_responses_exact',i)
oldmetrics=csvread(old_dir/'METRICS.csv')
oldmetric_index={(r['cohort'],r['group'],r['task'],r['model']):r for r in oldmetrics if r['model'] in MODELS[:2]}
for row in tables['metrics']:
    if row['model'] not in MODELS[:2]: continue
    old=oldmetric_index[(row['cohort'],row['group'],row['task'],row['model'])]
    for field in ('N','visible_support','occluded_support','TP','FN_decided','FP','TN','U_visible','U_occluded','out_of_task_S_visible','out_of_task_S_occluded','other_unresolved_visible','other_unresolved_occluded'):
        check(row[field]==int(old[field]),'V169_baseline_original_metric_counts_exact',[row['model'],field])

primary=[i for i,r in enumerate(truth) if r['paired_same_query_RGB_D_vs_E']=='True' and r['human_state'] in (D,X)]
primary_truth=Counter(truth[i]['human_state'] for i in primary)
check(len(primary)==1237 and primary_truth=={D:1147,X:90},'primary1237_D1147_E90')
check(Counter(r['group'] for r in manual_near)=={'WOOD_PRIMARY105':153,'SQUARE118':223},'manualnear376_153_223')
check(sum(r['human_state']==S for r in truth)==136,'full_self136_separate')
check(sum(r['N'] for r in tables['metrics'] if r['task']=='D_vs_E' and r['cohort']=='paired_same_query_RGB' and r['model']=='C47')==1237,'primary_materials_conserve')
compare(results,dict(original_slots=1784,WOOD_frames=105,SQUARE_frames=118,original_input_valid=1739,original_input_U=45,selected_model=None,adopted=False,fully_solved=False,new_primary_quality_gold_known=0,new_crop_selection=False),'result_scope_counts')
for output in eval_seal['outputs']:
    check(digest(output['path'])==output['sha256'],'evaluation_seal_output_hashes',Path(output['path']).name)
for path, before in reads.items():
    check(digest(path)==before,'all_audit_inputs_unchanged',path)

summary=[]
for row in tables['metrics']:
    if row['task']=='D_vs_E' and row['cohort']=='paired_same_query_RGB':
        summary.append({k:row[k] for k in ('group','model','N','visible_support','occluded_support','TP','FN_decided','FP','TN','U_visible','U_occluded','out_of_task_S_visible','out_of_task_S_occluded','other_unresolved_visible','other_unresolved_occluded','visible_precision','visible_recall_all_GT','occluded_precision','occluded_recall_all_GT')})
receipt=dict(completed=True,PASS=not failures,review_type='Independent saved-output numeric audit; not a new benchmark execution or actual model forward audit',
    code=binding(__file__),evaluation_seal=binding(E/'EVALUATION_SEAL.json'),prediction_seal=binding(V/'frozen_all3_direct_owner_predictions_v1/PREDICTION_SEAL.json'),
    ledger=binding(E/'ALL1784_NATIVE_LEDGER.csv'),results=binding(E/'CURRENT_FROZEN_BENCHMARK_RESULTS.json'),
    source_first_scalar_calculation=True,evaluator_or_metric_helper_import_or_call=False,
    coverage=dict(original_rows=1784,new_models=3,scalar_raw_predictions=5352,original_metadata_arrays_each=12,all_new_owner_U=45,primary_paired=1237,primary_D=1147,primary_external_E=90,manualnear=376,manualnear_WOOD=153,manualnear_SQUARE=223,full_original_self=136,table_rows=csv_table_counts,all_numeric_table_rows=sum(csv_table_counts.values()),V169_baseline_rows=1784),
    primary_material_metrics=summary,checks=dict(checks),number_checks=sum(checks.values()),failures=failures,
    input_bindings=[dict(path=k,sha256=v) for k,v in reads.items()],all_audit_inputs_unchanged=checks['all_audit_inputs_unchanged']==len(reads),
    limits=['Existing annotation/cohort/legacy state metadata and frozen input fingerprints are authorities for preservation checks; no physical native identity or new annotation verification.',
        'Sparse manual reference <=3px and general reference-error bins remain preexisting coordinate proxies, not physical pixel/corner gold or certified same-pixel occlusion truth.',
        'Owner F/BG binary hidden response does not identify physical external versus self occlusion; original 136 human SELF rows remain separately reported.',
        'Current223 has historical CAL/development exposure; this is a fixed evaluation diagnostic, not blind transfer evidence or a selection rule.',
        'No original feature/checkpoint/RGB bodies read, no serialized checkpoint-tensor or actual-forward re-audit; metadata source preservation is checked against existing manifest/prior and sealed 12-array fingerprints.'],
    model_feature_weight_RGB_encoder_fit_render_PnP_threshold_or_data_selection_calls=0,new_labels_or_coordinate_changes=0,
    held_public_V177_new_scores_or_other_reviewer_labels_read=False,selected_model=None,adopted=False,fully_solved=False)
with (OUT/'REVIEW.json').open('x') as f:
    json.dump(receipt,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(PASS=receipt['PASS'],number_checks=receipt['number_checks'],failures=len(failures),receipt=binding(OUT/'REVIEW.json'),coverage=receipt['coverage']),ensure_ascii=False))
if failures:
    print(json.dumps(failures[:12],ensure_ascii=False));raise SystemExit(1)
