"""Independent V213 fourteen-model saved-only scalar audit. No producer imports/model/image reads."""
from pathlib import Path
from collections import Counter
import argparse,csv,hashlib,io,json,math,os,stat,sys,time,traceback
import numpy as np
H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
OUT=V/'saved_real_head_evaluation_v1'
PROTO=H/'PROTOCOL_BEFORE_ANY_SAVED_REAL_RESULT_OR_PACKET_BODIES.json'
MODELS=('v61','legacy','C52','C53','C54','C55','C56','C57','C58','C59','C60','C61','C62','C63')
TAGS=MODELS[2:]
DIRECT=('C54','C55','C56','C57','C58','C59','C60','C61','C62','C63')
NEW_ARMS=('C62_DENSE_POINT_RAW_Q8','C63_DENSE_POINT_WEAK_BRIDGE_RAW_Q8')
GROUPS=('WOOD_PRIMARY105','SQUARE118')
D,E,S='DIRECT_VISIBLE','EXTERNAL_OCCLUDED','SELF_OCCLUDED'
OWNER=('TARGET_VISIBLE_SUPPORTED','FOREIGN_VISIBLE_SUPPORTED','BACKGROUND_SUPPORTED')
FILES=('current1784/ALL1784_CURRENT_NATIVE_LEDGER.jsonl','current1784/CURRENT_HUMAN_SEMANTIC_AND_SEPARATE_ASSOCIATIONS.json','current1784/METRICS.csv','current1784/D_VS_EXTERNAL_E_2X3.csv','publicheld360/ALL360_PUBLIC_WEAK_LEDGER.jsonl','publicheld360/PUBLIC_WEAK_OWNER_AND_SEPARATE_SUPPORT_ASSOCIATIONS.json')
META=('row_id','original_global_index','native_slot','query_xy','L','original_query_available')
PACKET_KEYS=set(META)|{'raw_support_logits','raw_owner_logits','raw_direct_end_external_logit','support_prediction','owner_prediction','direct_end_prediction_D0_E1_Uminus1','regional_support_proxy_D0_E1_Uminus1','support_order','owner_order','direct_end_order'}
def need(ok, text):
    if not bool(ok):
        raise AssertionError(text)

def safe(path):
    p = Path(path).absolute()
    need(P in p.parents and '..' not in p.parts and all('v112' not in a.casefold() for a in p.parts), 'Safe exact bound project name before access')
    return p

def opened(path):
    p = safe(path)
    fd = os.open(p.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nxt
        last = os.open(p.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
    finally:
        os.close(fd)
    need(stat.S_ISREG(os.fstat(last).st_mode), 'Regular file only')
    return os.fdopen(last, 'rb')

def raw(path):
    with opened(path) as f:
        return f.read()

def bind(path):
    return dict(path=str(safe(path)), sha256=hashlib.sha256(raw(path)).hexdigest())

def checked(binding):
    b = raw(binding['path'])
    need(hashlib.sha256(b).hexdigest() == binding['sha256'], 'Frozen byte binding')
    return b

def write(path, value):
    safe(path)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def ratio(a, b):
    return a / b if b else None

def close(a, b):
    if a is None or a == '':
        return b is None or b == ''
    if b is None or b == '':
        return False
    return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)

def yes(x):
    need(x in (True, False, 'True', 'False'), 'Frozen boolean enum')
    return x is True or x == 'True'

def bucket(row):
    try:
        value = float(row['initial_reference_error_px'])
    except (ValueError, TypeError):
        value = float('nan')
    return 'MISSING_REFERENCE_ERROR' if not math.isfinite(value) else 'LE3' if value <= 3 else 'GT3'

def binary(rows, model):
    cm = [[0, 0, 0], [0, 0, 0]]
    for row in rows:
        need(row['human_state'] in (D, E), 'Only existing semantic D/E in binary task')
        pred = row[model + '_D_E_U']
        need(pred in ('D', 'E', 'U'), 'Saved D/E/U enum')
        cm[int(row['human_state'] == E)][('D', 'E', 'U').index(pred)] += 1
    nd, ne = map(sum, cm)
    return dict(N=nd + ne, CM_D_E_by_D_E_U=cm, D_N=nd, E_N=ne,
                D_precision=ratio(cm[0][0], cm[0][0] + cm[1][0]),
                D_recall=ratio(cm[0][0], nd), E_precision=ratio(cm[1][1], cm[1][1] + cm[0][1]),
                E_recall=ratio(cm[1][1], ne), U=cm[0][2] + cm[1][2],
                U_rate=ratio(cm[0][2] + cm[1][2], nd + ne))

def compare_metric(metric, result):
    c = result['CM_D_E_by_D_E_U']
    for key, value in dict(N=result['N'], visible_support=result['D_N'], occluded_support=result['E_N'],
                           TP=c[0][0], FN_decided=c[0][1], FP=c[1][0], TN=c[1][1]).items():
        need(int(metric[key]) == value, 'Semantic scalar count ' + key)
    for key, key2 in [('visible_precision', 'D_precision'), ('visible_recall_all_GT', 'D_recall'),
                      ('occluded_precision', 'E_precision'), ('occluded_recall_all_GT', 'E_recall'),
                      ('unresolved_rate', 'U_rate')]:
        need(close(metric[key], result[key2]), 'Semantic scalar ratio ' + key)
    need(sum(int(metric[k]) for k in ('U_visible', 'out_of_task_S_visible', 'other_unresolved_visible')) == c[0][2], 'D all own-U reasons')
    need(sum(int(metric[k]) for k in ('U_occluded', 'out_of_task_S_occluded', 'other_unresolved_occluded')) == c[1][2], 'E all own-U reasons')

def table(xs, ys, nx, ny):
    out = [[0] * ny for _ in range(nx)]
    for a, b in zip(xs, ys):
        need(0 <= a < nx and 0 <= b < ny, 'Matrix category range')
        out[a][b] += 1
    return out

def hard(truth, prediction, names):
    k = len(names)
    matrix = table([x if x >= 0 else k for x in truth], [x if x >= 0 else k for x in prediction], k + 1, k + 1)
    per = {}
    for cls, name in enumerate(names):
        per[name] = dict(precision_on_known_truth=ratio(matrix[cls][cls], sum(matrix[t][cls] for t in range(k))),
                         recall_including_prediction_U=ratio(matrix[cls][cls], sum(matrix[cls])))
    return dict(full_confusion=matrix, per_class=per)

def compare_hard(a, b):
    need(a['full_confusion'] == b['full_confusion'], 'Weak confusion includes truthUNKNOWN/predictionU')
    for cls, row in b['per_class'].items():
        for key, value in row.items():
            need(close(a['per_class'][cls][key], value), 'Weak scalar precision/recall')

def validate_current(rows):
    need(len(rows) == 1784 and [r['global_index'] for r in rows] == list(range(1784)), 'All1784 original ledger order')
    need(len({(r['id'], int(r['native_corner'])) for r in rows}) == 1784, 'Unique original native identities')
    need(Counter(r['group'] for r in rows) == {GROUPS[0]: 840, GROUPS[1]: 944}, 'WOOD105/SQUARE118 all native8')
    need(sum(not yes(r['query_available']) for r in rows) == 45, 'Original inputU45')
    for r in rows:
        need(int(r['native_corner']) == r['global_index'] % 8, 'Original ordered native identity only')
        for m in MODELS:
            response = r[m + '_original_response']
            expected = 'D' if response == 'VISIBLE' else 'E' if response == 'ABSENT_NONVISIBLE' else 'U'
            need(r[m + '_D_E_U'] == expected, 'Saved own response/U collapse')
        for m in TAGS:
            av = yes(r['query_available'])
            raw_s, raw_o = r[m + '_raw_support_logits'], r[m + '_raw_owner_logits']
            if av:
                need(all(x is not None and math.isfinite(x) for x in raw_s + raw_o), 'Finite saved available diagnostic scalars')
                support = [int(x > 0) for x in raw_s]
                owner = max(range(3), key=lambda k: raw_o[k])
            else:
                need(all(x is None for x in raw_s + raw_o), 'FactualU rawnull')
                support, owner = [-1, -1], -1
            proxy = -1 if not av else 1 if support[1] == 1 else 0 if support[0] == 1 else -1
            need(r[m + '_support_prediction'] == support and r[m + '_owner_prediction'] == owner, 'Independent support/firstargmax')
            need(r[m + '_regional_support_proxy_D0_E1_Uminus1'] == proxy, 'Independent E-first broad proxy')
            primary = proxy
            if m in DIRECT:
                raw_d = r[m + '_raw_direct_end_external_logit']
                need(av or raw_d is None, 'Unavailable direct rawnull')
                primary = -1 if not av or raw_d is None or not math.isfinite(raw_d) else int(raw_d >= 0)
                need(r[m + '_direct_end_prediction_D0_E1_Uminus1'] == primary, 'Direct finite raw>=0 and originalU')
            need(r[m + '_primary_prediction'] == primary and r[m + '_D_E_U'] == {-1: 'U', 0: 'D', 1: 'E'}[primary], 'Primary route separate from diagnostic heads')

def current_audit(rows, summary, csv_metrics, csv_cm):
    validate_current(rows)
    primary = [r for r in rows if yes(r['paired_same_query_RGB_D_vs_E']) and r['human_state'] in (D, E)]
    need(Counter(r['group']for r in primary)=={GROUPS[0]:645,GROUPS[1]:592},'WOOD645/SQUARE592 primary distinct from registered frame counts')
    need(len(primary) == 1237 and Counter(r['human_state'] for r in primary) == {D: 1147, E: 90}, 'Fixed primary1237 D1147/E90')
    need(sum(yes(r['same_slot_pipeline_D_vs_E']) and r['human_state'] in (D, E) for r in rows) == 1248, 'Same-slot1248')
    result = {}
    for group in ('ALL', *GROUPS, *range(8)):
        selected = [r for r in primary if group == 'ALL' or r['group'] == group or isinstance(group, int) and int(r['native_corner']) == group]
        result[str(group)] = {m: binary(selected, m) for m in MODELS}
    metrics = summary['tables']['metrics']
    checked_count = 0
    for metric in metrics:
        if metric['task'] != 'D_vs_E':
            continue
        selected = [r for r in rows if r['group'] == metric['group'] and yes(r[metric['cohort'] + '_D_vs_E']) and r['human_state'] in (D, E)]
        compare_metric(metric, binary(selected, metric['model']))
        checked_count += 1
    need(checked_count == 56, 'Both fixed current cohorts x2groups x14models')
    need(len(csv_metrics) == len(metrics), 'All metric CSV rows preserved')
    for a, b in zip(csv_metrics, metrics):
        need(all(str(a[k]) == str(b[k]) or close(a[k], b[k]) for k in a), 'Metric CSV/JSON exact scalar representation')
    for cm in csv_cm:
        selected = [r for r in rows if r['group'] == cm['group'] and yes(r[cm['cohort'] + '_D_vs_E']) and r['human_state'] in (D, E)]
        expected = binary(selected, cm['model'])
        for key, value in zip(('D_pred_D', 'D_pred_hidden', 'D_pred_U', 'E_pred_D', 'E_pred_hidden', 'E_pred_U'), sum(expected['CM_D_E_by_D_E_U'], [])):
            need(int(cm[key]) == value, 'Independent exported2x3 CM cell')
    for metric in summary['tables']['coordinate_error_bins']:
        selected = [r for r in rows if r['group'] == metric['group'] and yes(r[metric['cohort'] + '_D_vs_E']) and r['human_state'] in (D, E) and bucket(r) == metric['error_bin']]
        compare_metric(metric, binary(selected, metric['model']))
    for row in summary['tables']['all1784_model_inventory']:
        selected = [r for r in rows if row['group'] == 'ALL' or r['group'] == row['group']]
        counts = Counter(r[row['model'] + '_D_E_U'] for r in selected)
        need(row['all_registered_slots'] == len(selected) and row['D'] == counts['D'] and row['E_binary_response'] == counts['E'] and row['U'] == counts['U'], 'Full1784 model inventory ownU')
    for assoc in summary['tables']['support_joint_owner_and_reference_associations']:
        selected = [r for r in rows if (assoc['group'] == 'ALL' or r['group'] == assoc['group']) and (assoc['error_bin'] == 'ALL' or bucket(r) == assoc['error_bin']) and (r['human_state'] if r['human_state'] in (D, E, S) else 'UNKNOWN') == assoc['human_semantic_state']]
        m = assoc['model'];cc = Counter();oc = Counter();pc = Counter()
        for r in selected:
            s = r[m + '_support_prediction'];cc[4 if min(s) < 0 else s[0] + 2 * s[1]] += 1
            oc[r[m + '_owner_prediction']] += 1;pc[r[m + '_regional_support_proxy_D0_E1_Uminus1']] += 1
        need(assoc['N'] == len(selected), 'Self/reference association N')
        for key, k in [('support_NONE', 0), ('support_VISIBLE_ONLY', 1), ('support_FOREIGN_ONLY', 2), ('support_BOTH', 3), ('support_U', 4)]:
            need(assoc[key] == cc[k], 'Separate support joint association')
        for key, k in [('owner_T', 0), ('owner_F', 1), ('owner_BG', 2), ('owner_U', -1)]:
            need(assoc[key] == oc[k], 'Separate center-owner association')
        need(assoc['proxy_D'] == pc[0] and assoc['proxy_E'] == pc[1] and assoc['proxy_U'] == pc[-1], 'Separate broad proxy association')
    need({r['N'] for r in summary['tables']['manual_near_D']} == {153, 223}, 'Existing manualnear coverage preserved, no membership reconstruction')
    # This is a different association from the fixed manually clicked376 cohort.
    registered_reference_D = [r for r in rows if r['human_state'] == D and bucket(r) == 'LE3']
    reference_association = {}
    for m in MODELS:
        counts = Counter(r[m + '_D_E_U'] for r in registered_reference_D)
        reference_association[m] = dict(N=len(registered_reference_D), D=counts['D'], E=counts['E'], U=counts['U'])
    return dict(primary=result, all_rows=1784, original_input_U=45, checked_binary_metric_rows=checked_count,
                all_registered_humanD_existing_reference_LE3_association=reference_association,
                self_and_reference_associations_checked=True, manualnear_membership='Existing fixed tables153/223 coverage only; no original annotation read')

def public_audit(rows, summary):
    need(len(rows) == 360 and [r['original_global_index'] for r in rows] == list(range(584, 944)), 'Public360 original IDs')
    need(len({r['query_id'] for r in rows}) == 360, 'Public query identities')
    truth = [OWNER.index(r['center_owner']) if r['center_owner'] in OWNER else -1 for r in rows]
    crop = [('YES', 'NO', 'UNKNOWN').index(r['crop_presence']) for r in rows]
    need(sum(t >= 0 for t in truth) == 204 and sum(t < 0 for t in truth) == 156, 'Weak204/156')
    need(sum(not yes(r['source_original_query_available']) for r in rows) == 108 and sum(not yes(r['source_center4_allvalid']) for r in rows) == 112 and sum(not yes(r['source_crop_constructed']) for r in rows) == 24, 'Public original U/center/crop separate')
    for r, t in zip(rows, truth):
        need((yes(r['source_original_query_available']) and yes(r['source_center4_allvalid'])) or t == -1, 'Forced center UNKNOWN')
        need(yes(r['source_crop_constructed']) or r['crop_presence'] == 'UNKNOWN', 'Forced crop UNKNOWN')
    preds = {m: [r[m + '_owner_prediction'] for r in rows] for m in TAGS}
    preds['v61'] = [r['v61_binary_T_nonT_prediction'] for r in rows]
    need(sum(x < 0 for x in preds['v61']) == 109 and all(sum(x < 0 for x in preds[m]) == 108 for m in TAGS), 'Own251/252 availability preserved')
    paired = [all(preds[m][i] >= 0 for m in preds) for i in range(360)]
    masks = dict(FULL360=[True] * 360, KNOWN_CENTER_TRUTH=[t >= 0 for t in truth], ALL13_MODEL_PAIRED_AVAILABLE=paired, KNOWN_TRUTH_AND_ALL13_PAIRED_AVAILABLE=[t >= 0 and paired[i] for i, t in enumerate(truth)])
    need(set(summary['cohorts']) == set(masks), 'Exactly frozen four public cohorts')
    for cohort, mask in masks.items():
        for group in ('ALL', *range(8)):
            ix = [i for i, r in enumerate(rows) if mask[i] and (group == 'ALL' or r['native_slot'] == group)]
            got = summary['cohorts'][cohort][str(group)];need(got['N'] == len(ix), 'Public group N')
            need(got['original_input_U'] == sum(not yes(rows[i]['source_original_query_available']) for i in ix)
                 and got['center4_invalid'] == sum(not yes(rows[i]['source_center4_allvalid']) for i in ix)
                 and got['no_constructed_crop'] == sum(not yes(rows[i]['source_crop_constructed']) for i in ix), 'Public cohort/native missing denominators')
            for m, pr in preds.items():
                tt = [(-1 if truth[i] < 0 else int(truth[i] != 0)) for i in ix]
                pp = [pr[i] if m == 'v61' else (-1 if pr[i] < 0 else int(pr[i] != 0)) for i in ix]
                compare_hard(got['models'][m]['binary_T_nonT'], hard(tt, pp, ['T', 'NON_T']))
                if m != 'v61':compare_hard(got['models'][m]['owner3'], hard([truth[i] for i in ix], [pr[i] for i in ix], OWNER))
    for m in TAGS:
        got = summary['support_crop_center_associations'][m]
        joint = [4 if min(r[m + '_support_prediction']) < 0 else r[m + '_support_prediction'][0] + 2 * r[m + '_support_prediction'][1] for r in rows]
        proxy = [r[m + '_regional_support_proxy_D0_E1_Uminus1'] for r in rows]
        primary = [r[m + '_primary_prediction'] for r in rows]
        need(got['support_joint_vs_crop'] == table(joint, crop, 5, 3) and got['support_joint_vs_center'] == table(joint, [t if t >= 0 else 3 for t in truth], 5, 4), 'Public separate joint support tables')
        need(got['proxy_vs_crop'] == table([x if x >= 0 else 2 for x in proxy], crop, 3, 3) and got['primary_direct_vs_crop'] == table([x if x >= 0 else 2 for x in primary], crop, 3, 3), 'Public direct/proxy versus crop association only')
        need(got['owner_vs_crop'] == table([x if x >= 0 else 3 for x in preds[m]], crop, 4, 3), 'Public owner/crop association')
    need(summary['legacy_baseline']['status'] == 'MISSING_IN_REVIEWED_LINEAGE' and not summary['legacy_baseline']['new_forward'], 'No legacy imputation')
    return dict(rows360=True, weak_known=204, weak_UNKNOWN=156, cohorts4_nativegroups9=True, v61_own_U=109, twelve_head_owner_U=108, no_native_external_gold=True)

def selftest():
    rows = [dict(human_state=t, m_D_E_U=p) for t, p in [(D, 'D'), (D, 'U'), (E, 'D'), (E, 'E')]]
    b = binary(rows, 'm');need(b['CM_D_E_by_D_E_U'] == [[1, 0, 1], [1, 1, 0]] and b['D_precision'] == .5 and b['D_recall'] == .5 and b['E_precision'] == 1, 'Independent binary toy')
    fixture = dict(N=4, visible_support=2, occluded_support=2, TP=1, FN_decided=0, FP=1, TN=1,
                   visible_precision=.5, visible_recall_all_GT=.5, occluded_precision=1., occluded_recall_all_GT=.5,
                   unresolved_rate=.25, U_visible=1, out_of_task_S_visible=0, other_unresolved_visible=0,
                   U_occluded=0, out_of_task_S_occluded=0, other_unresolved_occluded=0)
    compare_metric(fixture, b)
    try:compare_metric(dict(fixture, TP=2), b);rejected = False
    except AssertionError:rejected = True
    need(rejected, 'Scalar reported metric drift refused')
    x = hard([0, 1, -1, 0], [0, -1, 0, 1], ['T', 'NON_T'])
    need(x['full_confusion'] == [[1, 1, 0], [0, 0, 1], [1, 0, 0]] and x['per_class']['T']['precision_on_known_truth'] == 1 and x['per_class']['T']['recall_including_prediction_U'] == .5, 'TruthUNKNOWN not precision denominator, predictionU remains recall')
    need(bucket(dict(initial_reference_error_px='3')) == 'LE3' and bucket(dict(initial_reference_error_px='nan')) == 'MISSING_REFERENCE_ERROR', 'Boundary/missing association')
    need(table([0, 1, 2], [1, 1, 0], 3, 2) == [[0, 1], [0, 1], [1, 0]], 'Separate association cells')
    need(close('', None) and close('0.25', .25) and not close('0.2', .25)
         and binary([], 'm')['D_precision'] is None, 'CSV/null/zero-denominator ratios')
    try:safe(V / 'blocked_v112' / 'unopened');ok = False
    except AssertionError:ok = True
    need(ok, 'Forbidden name rejected without filesystem access')
    return dict(completed=True, PASS=True, invented_only=True, checks=7, actual_result_or_ledger_body_reads=0)

def csvrows(data):return list(csv.DictReader(io.StringIO(data.decode())))
def jsonrows(data):return[json.loads(s)for s in data.splitlines()if s.strip()]
def key(r):return str(r['id']),int(r['native_corner'])

def response(row,model,task,truth):
    t=truth[key(row)]
    # Exact V72 metric operator; its task eligibility is defined separately.
    if model=='v61':return'VISIBLE'if t['present_at_fixed_0_5']=='True'else'ABSENT_NONVISIBLE'
    if model=='legacy':
        state=t['legacy_stored_state'];return'VISIBLE'if state in('B','I')else'ABSENT_NONVISIBLE'if state=='O'or(state=='S'and task!='D_vs_E')else'OUT_OF_TASK_GEOMETRY_S'if state=='S'else'ABSTAIN'if state=='U'else'OTHER_UNRESOLVED'
    return row[model+'_original_response']

def v61_full_ledger_response(truth_row):
    # V195 full1784 inventory has its separate factual availability guard.
    if truth_row['query_valid']!='True'or truth_row.get('present_at_fixed_0_5')not in('True','False'):return'OTHER_UNRESOLVED'
    return'VISIBLE'if truth_row['present_at_fixed_0_5']=='True'else'ABSENT_NONVISIBLE'

def calculated(rows,model,task,truth):
    cc=Counter((r['human_state']==D,response(r,model,task,truth))for r in rows);pos=sum(r['human_state']==D for r in rows);neg=len(rows)-pos
    tp,fn,fp,tn=(cc[True,'VISIBLE'],cc[True,'ABSENT_NONVISIBLE'],cc[False,'VISIBLE'],cc[False,'ABSENT_NONVISIBLE']);up,un=cc[True,'ABSTAIN'],cc[False,'ABSTAIN'];sp,sn=cc[True,'OUT_OF_TASK_GEOMETRY_S'],cc[False,'OUT_OF_TASK_GEOMETRY_S'];op,on=cc[True,'OTHER_UNRESOLVED'],cc[False,'OTHER_UNRESOLVED']
    need(tp+fn+up+sp+op==pos and fp+tn+un+sn+on==neg,'All response classes conserved')
    return dict(N=len(rows),frames=len({r['id']for r in rows}),visible_support=pos,occluded_support=neg,TP=tp,FN_decided=fn,FP=fp,TN=tn,U_visible=up,U_occluded=un,out_of_task_S_visible=sp,out_of_task_S_occluded=sn,other_unresolved_visible=op,other_unresolved_occluded=on,visible_precision=ratio(tp,tp+fp),visible_recall_all_GT=ratio(tp,pos),occluded_precision=ratio(tn,tn+fn),occluded_recall_all_GT=ratio(tn,neg),abstention_rate=ratio(up+un,len(rows)),out_of_task_S_rate=ratio(sp+sn,len(rows)),unresolved_rate=ratio(up+un+sp+sn+op+on,len(rows)),decision_coverage=ratio(tp+fn+fp+tn,len(rows)),correct_fraction_all_GT=ratio(tp+tn,len(rows)),selective_accuracy=ratio(tp+tn,tp+fn+fp+tn))

def compare_fields(got,expected):
    for k,v in expected.items():need(close(got[k],v),'Independent scalar cell '+k)

def extended_tables(rows,summary,truth_rows,manual,tables_csv):
    truth={key(r):r for r in truth_rows};need(len(truth)==len(rows)==1784 and all(key(r)==key(t)for r,t in zip(rows,truth_rows)),'Original truth order/native join')
    for r,t in zip(rows,truth_rows):
        for k in('human_state','group','mapping_status','human_reference_x','human_reference_y','initial_reference_error_px'):need(str(r[k])==str(t[k]),'Original semantic/reference authority unchanged '+k)
        expected=v61_full_ledger_response(t);need(r['v61_original_response']==expected and r['v61_D_E_U']==('D'if expected=='VISIBLE'else'E'if expected=='ABSENT_NONVISIBLE'else'U'),'Separate v61 full-ledger original availability/enum guard')
    tasks={'D_vs_E':(D,E),'D_vs_S':(D,S),'D_vs_ES':(D,E,S)};tables=summary['tables']
    def select(row,task):
        flag=row['cohort']+'_'+('D_vs_ES'if task=='D_vs_S'else task)
        return[r for r in rows if r['group']==row['group']and yes(truth[key(r)][flag])and r['human_state']in tasks[task]]
    for r in tables['metrics']:compare_fields(r,calculated(select(r,r['task']),r['model'],r['task'],truth))
    for r in tables['confusion']:
        rr=[x for x in select(r,r['task'])if x['human_state']==r['human_state']];cc=Counter(response(x,r['model'],r['task'],truth)for x in rr)
        compare_fields(r,dict(N=len(rr),pred_visible=cc['VISIBLE'],pred_binary_hidden=cc['ABSENT_NONVISIBLE'],legacy_U=cc['ABSTAIN'],out_of_task_geometry_S=cc['OUT_OF_TASK_GEOMETRY_S'],input_or_operator_U=cc['OTHER_UNRESOLVED']))
    for r in tables['human_self_binary_response']:
        rr=[x for x in rows if x['group']==r['group']and yes(truth[key(x)][r['cohort']+'_D_vs_ES'])and x['human_state']==S];cc=Counter(response(x,r['model'],'D_vs_S',truth)for x in rr)
        compare_fields(r,dict(human_S_support=len(rr),binary_hidden_response=cc['ABSENT_NONVISIBLE'],visible_response=cc['VISIBLE'],unresolved=len(rr)-cc['VISIBLE']-cc['ABSENT_NONVISIBLE'],human_S_binary_hidden_response_rate=ratio(cc['ABSENT_NONVISIBLE'],len(rr))))
    selected=[m for m in manual if m['annotation_coordinate_source']=='manual_click'and m['human_state']==D and float(m['initial_reference_error_px'])<=3 and yes(m['query_valid'])];need(Counter(m['group']for m in selected)=={GROUPS[0]:153,GROUPS[1]:223},'Existing manualnear376 membership')
    bykey={key(r):r for r in rows};provenance={key(r):r for r in manual}
    for m in selected:
        t=truth[key(m)];need(all(m[k]==t[k]for k in('image_sha256','query_x','query_y')),'Existing manual source query association only')
    for r in tables['manual_near_D']:
        rr=[bykey[key(m)]for m in selected if m['group']==r['group']];cc=Counter(response(x,r['model'],'D_vs_ES',truth)for x in rr);compare_fields(r,dict(N=len(rr),pred_visible=cc['VISIBLE'],pred_binary_hidden=cc['ABSENT_NONVISIBLE'],unresolved=len(rr)-cc['VISIBLE']-cc['ABSENT_NONVISIBLE'],semantic_D_recall=ratio(cc['VISIBLE'],len(rr))))
    for r in tables['coordinate_error_bins']:
        rr=[x for x in select(r,'D_vs_E')if bucket(x)==r['error_bin']];compare_fields(r,calculated(rr,r['model'],'D_vs_E',truth));mc=sum(provenance.get(key(x),{}).get('annotation_coordinate_source')=='manual_click'for x in rr);compare_fields(r,dict(manual_click_reference_N=mc,other_or_unspecified_reference_N=len(rr)-mc))
    for r in tables['primary_native8_D_E_with_U']:
        rr=[x for x in rows if(r['group']=='ALL'or x['group']==r['group'])and int(x['native_corner'])==r['native_slot']and yes(x['paired_same_query_RGB_D_vs_E'])and x['human_state']in(D,E)];compare_fields(r,calculated(rr,r['model'],'D_vs_E',truth))
    for r in tables['paired_recovery_and_damage']:
        rr=[x for x in select(r,'D_vs_E')if r['truth_class']=='ALL_DE'or x['human_state']==r['truth_class']];cc=Counter()
        for x in rr:
            aa,bb=response(x,r['baseline'],'D_vs_E',truth),response(x,r['new_model'],'D_vs_E',truth);gt='VISIBLE'if x['human_state']==D else'ABSENT_NONVISIBLE'
            name='new_unresolved'if bb not in('VISIBLE','ABSENT_NONVISIBLE')else('baseline_unresolved_new_correct'if bb==gt else'baseline_unresolved_new_wrong')if aa not in('VISIBLE','ABSENT_NONVISIBLE')else'both_correct'if aa==gt and bb==gt else'baseline_correct_new_wrong'if aa==gt else'baseline_wrong_new_correct'if bb==gt else'both_wrong';cc[name]+=1
        compare_fields(r,dict(N=len(rr),**{k:cc[k]for k in('both_correct','baseline_correct_new_wrong','baseline_wrong_new_correct','both_wrong','baseline_unresolved_new_correct','baseline_unresolved_new_wrong','new_unresolved')}))
    for r in tables['support_joint_owner_and_reference_associations']:
        rr=[x for x in rows if(r['group']=='ALL'or x['group']==r['group'])and(r['error_bin']=='ALL'or bucket(x)==r['error_bin'])and(x['human_state']if x['human_state']in(D,E,S)else'UNKNOWN')==r['human_semantic_state']];cc=Counter(x[r['model']+'_primary_prediction']for x in rr);compare_fields(r,dict(primary_D=cc[0],primary_E=cc[1],primary_U=cc[-1],original_input_U=sum(not yes(x['query_available'])for x in rr)))
    for r in[tables['availability'],*tables['availability_by_group']]:
        rr=[x for x in rows if'group'not in r or x['group']==r['group']];priorU=sum(not yes(x['query_available'])for x in rr);u58=sum(x['C62_primary_prediction']<0 for x in rr);u59=sum(x['C63_primary_prediction']<0 for x in rr)
        expected=dict(prior_input_U=priorU,extra_operator_U=u58-priorU,C62_direct_total_U=u58,C63_direct_total_U=u59);expected['all_native_slots'if'group'in r else'rows']=len(rr);compare_fields(r,expected)
    for r in tables['all1784_model_inventory']:compare_fields(r,dict(U_rate=ratio(r['U'],r['all_registered_slots'])))
    for name,table_ in tables.items():
        if isinstance(table_,list):
            need(name in tables_csv and len(table_)==len(tables_csv[name]),'Every table exported as CSV '+name)
            for saved,actual in zip(tables_csv[name],table_):
                need(set(saved)==set(actual),'CSV table column schema')
                for k,v in actual.items():
                    try:ok=close(saved[k],v)
                    except(ValueError,TypeError):ok=str(saved[k])==str(v)
                    need(ok,'Every CSV/JSON table cell '+name+'/'+k)
    need(sum(r['human_state']==S for r in rows)==136 and sum(r['human_state']==S and yes(truth[key(r)]['paired_same_query_RGB_D_vs_ES'])for r in rows)==21,'SELF all136 versus paired21')
    need(sum(r['human_state']==D and bucket(r)=='LE3'for r in rows)==461,'Existing referenceLE3 humanD461 distinct manual376')
    return dict(all_task_metric_rows=len(tables['metrics']),all_confusion_cells=True,self136_paired21=True,manual376_membership_exact=True,reference461_separate=True,native8_and_transition_tables=True,every_table_CSV_JSON_cell=True)

def packet(binding,allow,keys=None):
    need(allow.get(binding['path'])==binding,'Finite exact packet whitelist before header/hash')
    with opened(binding['path'])as f:
        h=hashlib.sha256()
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
        need(h.hexdigest()==binding['sha256'],'Saved packet SHA');f.seek(0)
        with np.load(f,allow_pickle=False)as z:
            need(keys is None and set(z.files)==PACKET_KEYS,'Exactly new saved prediction packet16; original input arrays forbidden');return{k:z[k].copy()for k in z.files}

def eq(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def fingerprint(a):
    return dict(dtype=str(a.dtype),shape=list(a.shape),sha256=hashlib.sha256(a.tobytes()).hexdigest())

def scalar_decode(ss,oo,dd,av):
    if not av:return [-1,-1],-1,-1,-1
    need(all(math.isfinite(float(x)) for x in list(ss)+list(oo)),'Available finite diagnostics')
    support=[int(float(x)>0)for x in ss];owner=max(range(3),key=lambda j:float(oo[j]))
    direct=int(float(dd)>=0)if math.isfinite(float(dd))else-1
    return support,owner,direct,1 if support[1]else 0 if support[0]else-1

def prediction_gate(seal,p):
    need(seal['completed']is True and seal['PASS']is True and seal['all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads']is True and seal['annotations_reference_GT_oldreal_scores_or_fullRGB_read']is False and seal['all_batch_repeat2_state_and_input_bytes_exact']is True,'Both new predictions sealed before any label/body')
    need(seal['code']==p['predictor_code']and seal['protocol']==p['predictor_protocol']and set(seal['arms'])==set(NEW_ARMS)and seal['selected_model']is None and seal['adopted']is False,'Frozen unconditional two models')
    need(seal['scope']==p['root_scope']and seal['decoder_spec']==p['decoder_spec']and seal['original_rows2144']==2144 and seal['model_domain_rows4288']==4288 and seal['input_complete']==p['input_complete']and seal['original_raw_complete']==p['original_raw_complete']and seal['source_fit_complete']==p['source_fit_complete']and seal['source_fit_integrity_review']==p['source_fit_review'],'Exact source-final and original-cache provenance')
    need(seal['spatial_head']==p['spatial_head']and seal['source_context_adapter']==p['source_context_adapter'],'Exact spatial/context code metadata')
    for i,arm in enumerate(NEW_ARMS):
        a=seal['arms'][arm]
        need(a['updates']==4000 and a['use_sam']is False and a['spatial_q8']is True and a['weak_bridge']is bool(i)and a['weak_loss_weight']==(.25 if i else 0)and set(a['outputs'])==set(p['input_domains']),'Both final4000 RAW/spatial; fixed weak ablation/primary')

def source_gate(seal,p):
    fit=json.loads(checked(p['source_fit_complete']));review=json.loads(checked(p['source_fit_review']))
    need(fit['completed']is True and fit['PASS']is True and fit['all2_frozen_before_any_SOURCE_outputs']is True and fit['parameters_each']==118359 and fit['steps_each']==4000 and fit['total_updates']==8000 and fit['protected_real_COMMON_VAL_access']==0 and set(fit['arms'])==set(NEW_ARMS),'Source final both4000/118359 before any real packet')
    need(fit['protocol']==p['source_fit_protocol']and fit['spatial_head']==p['spatial_head']and fit['context_adapter']==p['source_context_adapter']and fit['all2_freeze']==p['source_both_final_freeze'],'Exact source objective and barrier')
    barrier=json.loads(checked(fit['all2_freeze']))
    need(barrier['completed']is True and barrier['source_predictions_emitted']is False and barrier['checkpoints']=={a:fit['arms'][a]['checkpoint']for a in NEW_ARMS},'Both final source checkpoints before predictions')
    need(review['completed']is True and review['PASS']is True and review['fit_complete']==p['source_fit_complete'],'Independent exact-fit PASS, no guessed typed aliases')
    for i,a in enumerate(NEW_ARMS):
        f=fit['arms'][a];z=seal['arms'][a]
        need(f['updates']==4000 and f['use_sam']is False and f['spatial_q8']is True and f['weak_bridge']is bool(i)and f['weak_loss_weight']==(.25 if i else 0)and z['checkpoint']==f['checkpoint']and z['source_final_state_sha']==f['final_state_sha'],'Prediction exact source final/weak role')
    return dict(completed=True,PASS=True,all2_source_barrier_exact=True,owned_checkpoint_or_source_array_reads=0)

def prediction_audit(seal,p,current,public):
    prediction_gate(seal,p)
    allow={b['path']:b for a in NEW_ARMS for b in seal['arms'][a]['outputs'].values()}
    need(len(allow)==4,'Exactly four new prediction packets, no input NPZ permitted')
    count=0
    for domain,rows,start,U in [('CURRENT223_1784',current,0,45),('PUBLIC_HELD45_360',public,584,108)]:
        contract=p['input_domains'][domain];metadata=jsonrows(checked(contract['metadata']));n=len(rows)
        need(len(metadata)==n,'All original input-only metadata rows')
        first=None
        for tag,arm in zip(('C62','C63'),NEW_ARMS):
            z=packet(seal['arms'][arm]['outputs'][domain],allow)
            need(all(fingerprint(z[k])==contract['original_array_fingerprints'][k] for k in META),'All META6 byte fingerprints exact to pinned original arrays without input-array read')
            if first is None:first={k:z[k]for k in META}
            else:need(all(eq(z[k],first[k])for k in META),'Both packets same original META bytes')
            need(z['original_global_index'].tolist()==list(range(start,start+n))and int((~z['original_query_available']).sum())==U,'Original domain IDs and own inputU')
            need(z['row_id'].dtype.kind=='U'and len(set(z['row_id'].tolist()))==n,'Original occurrence identity')
            need(z['raw_support_logits'].dtype==z['raw_owner_logits'].dtype==z['raw_direct_end_external_logit'].dtype==np.float32 and z['raw_support_logits'].shape==(n,2)and z['raw_owner_logits'].shape==(n,3)and z['raw_direct_end_external_logit'].shape==(n,),'Raw FP32 diagnostics shape')
            for k,shape in [('support_prediction',(n,2)),('owner_prediction',(n,)),('direct_end_prediction_D0_E1_Uminus1',(n,)),('regional_support_proxy_D0_E1_Uminus1',(n,))]:need(z[k].dtype==np.int8 and z[k].shape==shape,'Saved prediction dtype/shape')
            need(z['support_order'].tolist()==['visible_end_support','foreign_front_support']and z['owner_order'].tolist()==['TARGET','FOREIGN','BACKGROUND']and z['direct_end_order'].tolist()==['VISIBLE_END0','EXTERNAL_BLOCKED_END1','UNKNOWNminus1'],'Distinct semantic orders')
            for i,(row,m)in enumerate(zip(rows,metadata)):
                av=bool(z['original_query_available'][i]);need(int(z['native_slot'][i])==m['native_slot']and str(z['row_id'][i])==m['row_id']and av==m['original_query_available']and int(z['original_global_index'][i])==m['original_global_index'],'Metadata identity/native/U join')
                q=np.asarray([np.nan,np.nan]if m['query_xy']is None else m['query_xy'],np.float64);L=np.asarray(np.nan if m['L']is None else m['L'],np.float64)
                need(np.array_equal(z['query_xy'][i],q,equal_nan=True)and np.array_equal(z['L'][i],L,equal_nan=True),'Original F64 crop q/L unchanged')
                ss,oo,dd=z['raw_support_logits'][i],z['raw_owner_logits'][i],float(z['raw_direct_end_external_logit'][i]);sp,own,direct,proxy=scalar_decode(ss,oo,dd,av)
                if not av:need(np.isnan(ss).all()and np.isnan(oo).all()and math.isnan(dd),'FactualU rawNaN')
                need(z['support_prediction'][i].tolist()==sp and int(z['owner_prediction'][i])==own and int(z['direct_end_prediction_D0_E1_Uminus1'][i])==direct and int(z['regional_support_proxy_D0_E1_Uminus1'][i])==proxy,'Independent scalar16key packet decoding')
                need(row[tag+'_support_prediction']==sp and row[tag+'_owner_prediction']==own and row[tag+'_primary_prediction']==direct and row[tag+'_regional_support_proxy_D0_E1_Uminus1']==proxy,'New ledger equals independent packet scalars')
                if domain=='CURRENT223_1784':
                    need(row['global_index']==start+i and int(row['native_corner'])==int(z['native_slot'][i])and row[tag+'_raw_support_logits']==[float(v)if math.isfinite(float(v))else None for v in ss]and row[tag+'_raw_owner_logits']==[float(v)if math.isfinite(float(v))else None for v in oo]and row[tag+'_raw_direct_end_external_logit']==(dd if math.isfinite(dd)else None),'Every new current raw ledger value')
                    for k,value in [('query_x',float(z['query_xy'][i,0])),('query_y',float(z['query_xy'][i,1])),('roi_side',float(z['L'][i]))]:need(row[k]==(value if math.isfinite(value)else None),'Current ledger F64 q/L')
                    need(yes(row['query_available'])==av,'Current own availability')
                else:need(row['original_global_index']==start+i and row['query_id']==str(z['row_id'][i]),'Public exact original query identity')
                count+=1
    return dict(new_packet_rows=count,all4288_new_saved_rows_scalar_and_originalMETA_bytes=True,input_NPZ_header_or_pixels_reads=0)

def public_authority_audit(rows,ep):
    source=jsonrows(checked(ep['public']['inputs']));consensus=jsonrows(checked(ep['public']['consensus']))
    need(len(source)==len(consensus)==len(rows)==360,'All original public weak authority records')
    for i,(r,s,c)in enumerate(zip(rows,source,consensus)):
        need(r['original_global_index']==s['original_global_index']==c['original_global_index']==584+i and r['query_id']==s['query_id']==c['query_id']and r['native_slot']==s['native_slot']==c['native_slot'],'Public original authority join')
        for k in ('center_owner','crop_presence','source_original_query_available','source_center4_allvalid','source_crop_constructed'):need(r[k]==c[k],'Original public weak consensus unchanged')
        for field,known in [('center_owner',OWNER),('crop_presence',('YES','NO'))]:
            a,b=c['reviewer_A'][field],c['reviewer_B'][field]
            need(c[field]==(a if a==b and a in known else 'UNKNOWN'),'Original A/B known-only consensus, never impute')
        need(c['source_original_query_available']==s['original_query_available']and c['source_center4_allvalid']==s['local_central4_available']and c['source_crop_constructed']==s['feature_constructed'],'Separate source input/center/crop flags')
    return dict(rows360=True,original_weak_consensus_and_input_flags_independently_joined=True,not_native_external_gold=True)

def prepare():
    old=json.loads(raw(OLD_PROTOCOL));ep=json.loads(raw(EVAL_PROTOCOL));pp=json.loads(raw(PRED_PROTOCOL));checks=selftest()
    extra=[]
    for x,expected in [(-1e-45,0),(-0.,1),(0.,1),(1e-45,1),(float('nan'),-1),(float('inf'),-1),(-float('inf'),-1)]:
        need(scalar_decode([0.,1.],[1.,1.,0.],x,True)==([0,1],0,expected,1),'Fixed direct/owner scalar fixture')
    need(scalar_decode([np.nan,np.nan],[np.nan]*3,np.nan,False)==([-1,-1],-1,-1,-1),'Unavailable remains U');extra.append('strict signs/negativezero/nonfinite/firsttie/originalU')
    a=np.array([0.,np.nan],np.float64);need(fingerprint(a)==fingerprint(a.copy())and fingerprint(a)!=fingerprint(a.astype(np.float32)),'META exact bytes/dtype');extra.append('META byte dtype drift')
    p=dict(completed=True,prepared_only=True,code=bind(__file__),root_scope=ep['scope'],evaluator_code=ep['code'],evaluator_protocol=bind(EVAL_PROTOCOL),predictor_code=pp['code'],predictor_protocol=bind(PRED_PROTOCOL),input_complete=pp['input_complete'],original_raw_complete=pp['original_raw_complete'],input_domains=pp['domains'],source_fit_complete=None,source_fit_complete_path=pp['source_fit_complete_path'],source_fit_review=None,source_both_final_freeze=None,source_fit_protocol=None,spatial_head=pp['spatial_head'],source_context_adapter=pp['source_context_adapter'],decoder_spec=pp['decoder_spec'],prior_twelve_evaluation_complete_path=ep['prior_twelve_evaluation_complete_path'],inherited_auditor=bind(OLD_CODE),inherited_auditor_protocol=bind(OLD_PROTOCOL),runtime=old['runtime'],models=list(MODELS),new_arms=list(NEW_ARMS),output_directory=str(H/'actual_review_v1'),input_NPZ_header_pixels_models_weights_forward_reads_allowed=False,exact_new_packet_count=4,actual_result_bodies_read_during_preparation=0,independent_authority='Original human truth/manual CSV and original public input+weak consensus, independent new saved scalar decoding. Old12 full frozen ledger fields/metrics unchanged. META6 declared byte fingerprints and row JSONL; no input NPZ.',limits=['Repeatedly exposed evaluation, not fresh blind. Human semantic authority retained; public weak center-owner/crop is separate.','<=3px reference461 and manual376 are associations, not physical native coordinate gold.','No model/threshold/data selection, forward or fit.','v61 original metric operator and full-ledger availability guard are separate definitions.'])
    toy=dict(completed=True,PASS=True,code=p['predictor_code'],protocol=p['predictor_protocol'],scope=p['root_scope'],decoder_spec=p['decoder_spec'],input_complete=p['input_complete'],original_raw_complete=p['original_raw_complete'],source_fit_complete=None,source_fit_integrity_review=None,spatial_head=p['spatial_head'],source_context_adapter=p['source_context_adapter'],all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads=True,annotations_reference_GT_oldreal_scores_or_fullRGB_read=False,all_batch_repeat2_state_and_input_bytes_exact=True,original_rows2144=2144,model_domain_rows4288=4288,selected_model=None,adopted=False,arms={a:dict(updates=4000,use_sam=False,spatial_q8=True,weak_bridge=bool(i),weak_loss_weight=.25 if i else 0,outputs={d:{}for d in p['input_domains']})for i,a in enumerate(NEW_ARMS)})
    prediction_gate(toy,p)
    for k,v in [('all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads',False),('source_fit_complete',{}),('annotations_reference_GT_oldreal_scores_or_fullRGB_read',True)]:
        try:prediction_gate(dict(toy,**{k:v}),p);ok=False
        except(AssertionError,KeyError):ok=True
        need(ok,'Bad prelabel gate refused')
    extra.append('Both final4000/prelabel/provenance gates positive and negatives')
    # Exercise actual response helper: metric v61 absence versus full-ledger unavailable U.
    rr=dict(id='toy',native_corner=0);tt={key(rr):dict(present_at_fixed_0_5='',query_valid='False')}
    need(response(rr,'v61','D_vs_E',tt)=='ABSENT_NONVISIBLE'and v61_full_ledger_response(tt[key(rr)])=='OTHER_UNRESOLVED','Baseline operator definitions kept distinct');extra.append('Actual v61 metric/full-ledger distinction')
    bad=Path(str(P/'DO_NOT_ACCESS_V112/absent.json'))
    try:safe(bad);ok=False
    except AssertionError:ok=True
    need(ok,'Lexical forbid before filesystem');extra.append('V112 lexical rejection')
    checks.update(checks=checks['checks']+len(extra),additional_checks=extra);write(H/'GENERATED_INVENTED_CHECKS.json',checks);p['checks']=bind(H/'GENERATED_INVENTED_CHECKS.json');write(PROTO,p)
    template=H/'ROOT_ACTUAL_SAVED_RESULT_AUDIT_ENTRY_TEMPLATE_APPROVED_FALSE.json'
    write(template,dict(approved_actual_independent_saved_result_audit=False,all_two_real_predictions_frozen=False,saved_evaluation_completed=False,automatic_retry=False,code=bind(__file__),protocol=bind(PROTO),root_scope=p['root_scope'],evaluation_complete=None,prediction_seal=None,source_fit_complete=None,source_fit_review=None,source_fit_protocol=None,source_both_final_freeze=None,prior_twelve_evaluation_complete=None))
    ready=dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTO),checks=p['checks'],entry_template=bind(template),actual_private_annotation_score_prediction_header_input_array_weight_or_model_reads=0,code_only_reads_bound=True)
    write(H/'FINAL_READY.json',ready);print(json.dumps(ready),flush=True)

def run(entry):
    e=json.loads(raw(entry));need(e['approved_actual_independent_saved_result_audit']is True and e['all_two_real_predictions_frozen']is True and e['saved_evaluation_completed']is True and e['automatic_retry']is False,'Separate postprediction/evaluation root authorization')
    need(e['code']==bind(__file__)and e['protocol']==bind(PROTO),'Frozen own code/protocol');p=json.loads(checked(e['protocol']));need(p['code']==e['code']and e['root_scope']==p['root_scope'],'Exact frozen own protocol/root scope');checked(e['root_scope'])
    need(sys.executable==p['runtime']['executable']and np.__version__==p['runtime']['numpy'],'Pinned independent scalar runtime');need(json.loads(checked(p['checks']))['PASS']is True,'Meaningful invented PASS')
    for k in('source_fit_complete','source_fit_review','source_fit_protocol','source_both_final_freeze','prior_twelve_evaluation_complete'):need(isinstance(e.get(k),dict)and set(e[k])=={'path','sha256'},'Root actual exact nonnull '+k);p[k]=e[k]
    need(p['source_fit_complete']['path']==p['source_fit_complete_path']and p['prior_twelve_evaluation_complete']['path']==p['prior_twelve_evaluation_complete_path'],'Exact source and old12 completion paths')
    need(e['evaluation_complete']['path']==str(OUT/'COMPLETE.json'),'Exact saved-evaluation output');complete=json.loads(checked(e['evaluation_complete']))
    need(complete['completed']is True and complete['PASS']is True and complete['code']==p['evaluator_code']and complete['protocol']==p['evaluator_protocol']and complete['all_both_predictions_closed_before_any_annotation_or_score_body']is True and complete['selected_model']is None and complete['adopted']is False and complete['models']==list(MODELS)and complete['old_twelve_primary_metrics_exact']is True,'Evaluation closed all14 without selection')
    need(e['prediction_seal']==complete['prediction_seal']and e['source_fit_complete']==p['source_fit_complete']and e['source_fit_review']==p['source_fit_review'],'Exact root entry artifact closure')
    ep=json.loads(checked(p['evaluator_protocol']));evaluation_entry=json.loads(checked(complete['entry']));seal=json.loads(checked(e['prediction_seal']))
    need(evaluation_entry['prediction_seal']==e['prediction_seal'],'Same preannotation all2 predictionseal');prediction_gate(seal,p);source=source_gate(seal,p)
    need(evaluation_entry['prior_twelve_evaluation_complete']==p['prior_twelve_evaluation_complete'],'Exact old12 evaluation root binding')
    marker=json.loads(checked(complete['outputs']['BEFORE_ANNOTATION_AND_SAVED_SCORE_READ.json']));need(marker['both_models_closed']is True and marker['annotation_or_score_bodies_read_so_far']is False and marker['prediction_seal']==e['prediction_seal'],'Prospective evaluation body barrier')
    destination=safe(p['output_directory']);need(destination==H/'actual_review_v1','Fresh exact actual destination');destination.mkdir(exist_ok=False)
    try:
        allowed={n:b for n,b in complete['outputs'].items()if n.startswith(('current1784/','publicheld360/'))}
        need(all(Path(n).suffix in('.json','.jsonl','.csv')for n in allowed),'Finite saved-table text only')
        data={n:checked(b)for n,b in allowed.items()};current=jsonrows(data[FILES[0]]);public=jsonrows(data[FILES[4]]);summary=json.loads(data[FILES[1]]);psummary=json.loads(data[FILES[5]])
        truth=csvrows(checked(ep['current']['truth']));manual=csvrows(checked(ep['current']['manual']));tables={}
        for k,v in summary['tables'].items():
            if isinstance(v,list):
                filename='current1784/'+k.upper()+'.csv';need(not v or filename in data,'Every nonempty table CSV');tables[k]=csvrows(data[filename])if filename in data else[]
        # Original authority membership is validated before the new ledger is used for scalar census.
        extended=extended_tables(current,summary,truth,manual,tables);authority=public_authority_audit(public,ep)
        base=current_audit(current,summary,csvrows(data[FILES[2]]),csvrows(data[FILES[3]]));weak=public_audit(public,psummary);pred=prediction_audit(seal,p,current,public)
        old=json.loads(checked(p['prior_twelve_evaluation_complete']));need(old['completed']and old['PASS'],'Closed old12 evaluation')
        for filename,newrows in [(FILES[0],current),(FILES[4],public)]:
            prior=jsonrows(checked(old['outputs'][filename]));need(len(prior)==len(newrows),'Old12 original rows retained')
            for a,b in zip(prior,newrows):need(all(b[k]==v for k,v in a.items()),'Every old12 ledger field/weak truth unchanged')
        oldmetrics=csvrows(checked(old['outputs'][FILES[2]]));newmetrics=[r for r in summary['tables']['metrics']if r['model']in MODELS[:12]];need(len(oldmetrics)==len(newmetrics),'All old12 metric rows retained')
        for a,b in zip(oldmetrics,newmetrics):
            for k,v in b.items():
                try:ok=close(a[k],v)
                except(ValueError,TypeError):ok=str(a[k])==str(v)
                need(ok,'Old12 metric schema/count/ratio unchanged')
        for b in allowed.values():checked(b)
        for b in (ep['current']['truth'],ep['current']['manual'],ep['public']['inputs'],ep['public']['consensus'],p['prior_twelve_evaluation_complete'],old['outputs'][FILES[0]],old['outputs'][FILES[2]],old['outputs'][FILES[4]],e['prediction_seal'],e['evaluation_complete']):checked(b)
        report=dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(PROTO),actual_entry=bind(entry),evaluation_complete=e['evaluation_complete'],complete=e['evaluation_complete'],prediction_seal=e['prediction_seal'],source_fit_complete=p['source_fit_complete'],source_fit_review=p['source_fit_review'],source_final_gate=source,current=base,public=weak,public_original_authority=authority,all_existing_tables=extended,new_prediction_packets=pred,old12_saved_ledger_fields_and_metric_counts_ratios_unchanged=True,primary1237_D1147_E90=True,self_all136_paired21_separate=True,reference461_manual376_separate=True,public_weak_only=True,all14_current_and_public13_contract=True,original_human_truth_membership_and_manual_reference_independently_joined=True,actual_model_instantiation_forward_fit_or_image_read=0,original_input_NPZ_header_or_array_reads=0,checkpoint_or_foundation_weight_reads=0,automatic_retry=False,selected_model=None,adopted=False,fully_solved=False,limits=p['limits']+['META6 compares exact saved-byte fingerprints declared in frozen input contract and row JSONL, without opening original input NPZ. Prediction inference/repeat/state is receipt provenance, not a new neural replay.','No all-image or browser QA. Public weak owner/crop labels cannot certify physical native end/external/self cause.'])
        write(destination/'REVIEW.json',report);print(json.dumps(dict(completed=True,PASS=True,review=bind(destination/'REVIEW.json'))),flush=True)
    except BaseException:
        write(destination/'FAILURE.json',dict(completed=False,PASS=False,traceback=traceback.format_exc(),automatic_retry=False));raise

OLD_CODE=P/'query_visibility_spatial_q8_heads_frozen_real_benchmark_20261025_v203/independent_saved_result_review_c/audit_saved_twelve_model_results.py'
OLD_PROTOCOL=OLD_CODE.parent/'PROTOCOL_BEFORE_ANY_SAVED_REAL_RESULT_OR_PACKET_BODIES.json'
EVAL_PROTOCOL=V/'inference_evaluation_preparation_b/PROTOCOL_BEFORE_REAL_ANNOTATION_OR_SAVED_SCORE_BODIES.json'
PRED_PROTOCOL=V/'inference_evaluation_preparation_b/PROTOCOL_BEFORE_ANY_REAL_ARRAY_OR_FINAL_WEIGHT_BODIES.json'
if __name__=='__main__':
    parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);args=parser.parse_args();prepare()if args.prepare else run(args.entry)
