"""Independent V200 ten-model saved-only scalar audit. No producer imports/model/image reads."""
from pathlib import Path
from collections import Counter
import argparse,csv,hashlib,io,json,math,os,stat,sys,time,traceback
import numpy as np
H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
OUT=V/'saved_real_head_evaluation_v1'
PROTO=H/'PROTOCOL_BEFORE_ANY_SAVED_REAL_RESULT_OR_PACKET_BODIES.json'
MODELS=('v61','legacy','C52','C53','C54','C55','C56','C57','C58','C59')
TAGS=MODELS[2:]
DIRECT=('C54','C55','C56','C57','C58','C59')
NEW_ARMS=('C58_RAW96_DIRECT_RESIDUAL_ZERO_NATIVE_CONTROL','C59_RAW96_DIRECT_RESIDUAL_NATIVE_ONEHOT')
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
    need(checked_count == 40, 'Both fixed current cohorts x2groups x10models')
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
    masks = dict(FULL360=[True] * 360, KNOWN_CENTER_TRUTH=[t >= 0 for t in truth], ALL9_MODEL_PAIRED_AVAILABLE=paired, KNOWN_TRUTH_AND_ALL9_PAIRED_AVAILABLE=[t >= 0 and paired[i] for i, t in enumerate(truth)])
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
    return dict(rows360=True, weak_known=204, weak_UNKNOWN=156, cohorts4_nativegroups9=True, v61_own_U=109, eight_head_owner_U=108, no_native_external_gold=True)

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
        rr=[x for x in rows if'group'not in r or x['group']==r['group']];priorU=sum(not yes(x['query_available'])for x in rr);u58=sum(x['C58_primary_prediction']<0 for x in rr);u59=sum(x['C59_primary_prediction']<0 for x in rr)
        expected=dict(prior_input_U=priorU,extra_operator_U=u58-priorU,C58_direct_total_U=u58,C59_direct_total_U=u59);expected['all_native_slots'if'group'in r else'rows']=len(rr);compare_fields(r,expected)
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
            need(set(z.files)==PACKET_KEYS if keys is None else set(keys)<=set(z.files),'Target-free output/originalMETA schema');return{k:z[k].copy()for k in(z.files if keys is None else keys)}

def eq(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()

def prediction_gate(seal,p):
    need(seal['completed']is True and seal['PASS']is True and seal['all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads']is True and seal['annotations_reference_GT_oldreal_scores_or_fullRGB_read']is False and seal['all_batch_repeat2_state_and_input_bytes_exact']is True,'Both new predictions sealed before any label/body access')
    need(seal['code']==p['predictor_code']and seal['protocol']==p['predictor_protocol']and set(seal['arms'])==set(NEW_ARMS)and seal['selected_model']is None and seal['adopted']is False,'Frozen final two, no selection')
    need(seal['original_rows2144']==2144 and seal['model_domain_rows4288']==4288 and seal['input_complete']==p['input_complete']and seal['original_raw_complete']==p['original_raw_complete']and seal['source_fit_complete']['path']==p['source_fit_complete_path'],'Both complete original domains/source-final provenance')
    need(seal['native_head']==p['native_head'],'Exact native residual head code metadata')
    need(all(seal['arms'][arm]['updates']==8000 and seal['arms'][arm]['use_sam']is False and type(seal['arms'][arm]['native_condition'])is bool and seal['arms'][arm]['native_condition']==(arm==NEW_ARMS[1])for arm in NEW_ARMS),'Both RAW final8000/native flag false true without default')

def prediction_audit(seal,p,current,public):
    prediction_gate(seal,p)
    allow={d['inputs']['path']:d['inputs']for d in p['input_domains'].values()}
    for arm in NEW_ARMS:
        need(seal['arms'][arm]['updates']==8000,'Fixed final8000 source head')
        for b in seal['arms'][arm]['outputs'].values():allow[b['path']]=b
    count=0
    for domain,rows,start,U in [('CURRENT223_1784',current,0,45),('PUBLIC_HELD45_360',public,584,108)]:
        original=packet(p['input_domains'][domain]['inputs'],allow,META);metadata=jsonrows(checked(p['input_domains'][domain]['metadata']));need(len(metadata)==len(rows),'All input-only rows')
        n=len(rows)
        for name,dtype,shape in[('original_global_index',np.int64,(n,)),('native_slot',np.int8,(n,)),('query_xy',np.float64,(n,2)),('L',np.float64,(n,)),('original_query_available',bool,(n,))]:need(original[name].dtype==dtype and original[name].shape==shape,'Original input META6 dtype/shape')
        need(original['row_id'].dtype.kind=='U'and original['row_id'].shape==(n,)and len(set(original['row_id'].tolist()))==n,'Original occurrence IDs unique')
        need(original['original_global_index'].tolist()==list(range(start,start+len(rows)))and int((~original['original_query_available']).sum())==U,'Original ID/U census')
        for tag,arm in zip(('C58','C59'),NEW_ARMS):
            z=packet(seal['arms'][arm]['outputs'][domain],allow);need(all(eq(z[k],original[k])for k in META),'All META6 byteexact to original input cache')
            need(z['raw_support_logits'].dtype==z['raw_owner_logits'].dtype==z['raw_direct_end_external_logit'].dtype==np.float32,'Raw FP32')
            need(z['raw_support_logits'].shape==(n,2)and z['raw_owner_logits'].shape==(n,3)and z['raw_direct_end_external_logit'].shape==(n,),'Raw packet shape')
            for name,shape in[('support_prediction',(n,2)),('owner_prediction',(n,)),('direct_end_prediction_D0_E1_Uminus1',(n,)),('regional_support_proxy_D0_E1_Uminus1',(n,))]:need(z[name].dtype==np.int8 and z[name].shape==shape,'Saved prediction dtype/shape')
            need(z['support_order'].tolist()==['visible_end_support','foreign_front_support']and z['owner_order'].tolist()==['TARGET','FOREIGN','BACKGROUND']and z['direct_end_order'].tolist()==['VISIBLE_END0','EXTERNAL_BLOCKED_END1','UNKNOWNminus1'],'Task orders kept separate')
            for i,(row,m)in enumerate(zip(rows,metadata)):
                av=bool(original['original_query_available'][i]);need(z['native_slot'][i]==m['native_slot']and str(z['row_id'][i])==m['row_id']and bool(z['original_query_available'][i])==m['original_query_available'],'Original native/row/availability metadata')
                q=np.asarray([np.nan,np.nan]if m['query_xy']is None else m['query_xy'],np.float64);L=np.asarray(np.nan if m['L']is None else m['L'],np.float64);need(np.array_equal(z['query_xy'][i],q,equal_nan=True)and np.array_equal(z['L'][i],L,equal_nan=True),'Original F64 q/L')
                ss,oo,dd=z['raw_support_logits'][i],z['raw_owner_logits'][i],float(z['raw_direct_end_external_logit'][i]);sp=[-1,-1];own=-1;direct=-1;proxy=-1
                if av:
                    need(np.isfinite(ss).all()and np.isfinite(oo).all(),'Available finite diagnostics');sp=[int(float(v)>0)for v in ss];own=max(range(3),key=lambda k:float(oo[k]));direct=int(dd>=0)if math.isfinite(dd)else-1;proxy=1 if sp[1]else 0 if sp[0]else-1
                else:need(np.isnan(ss).all()and np.isnan(oo).all()and math.isnan(dd),'FactualU rawNaN')
                need(z['support_prediction'][i].tolist()==sp and int(z['owner_prediction'][i])==own and int(z['direct_end_prediction_D0_E1_Uminus1'][i])==direct and int(z['regional_support_proxy_D0_E1_Uminus1'][i])==proxy,'Independent scalar new packet decoder')
                need(row[tag+'_support_prediction']==sp and row[tag+'_owner_prediction']==own and row[tag+'_primary_prediction']==direct and row[tag+'_regional_support_proxy_D0_E1_Uminus1']==proxy,'Saved ledger independent packet join')
                if domain=='CURRENT223_1784':
                    need(row['global_index']==start+i and int(row['native_corner'])==int(z['native_slot'][i])and row[tag+'_raw_support_logits']==[float(v)if math.isfinite(float(v))else None for v in ss]and row[tag+'_raw_owner_logits']==[float(v)if math.isfinite(float(v))else None for v in oo]and row[tag+'_raw_direct_end_external_logit']==(dd if math.isfinite(dd)else None),'Every saved raw ledger scalar')
                    for name,value in[('query_x',float(z['query_xy'][i,0])),('query_y',float(z['query_xy'][i,1])),('roi_side',float(z['L'][i]))]:need(row[name]==(value if math.isfinite(value)else None),'Saved ledger original F64 q/L')
                    need(yes(row['query_available'])==av,'Saved ledger original availability')
                else:need(row['original_global_index']==start+i and row['query_id']==str(z['row_id'][i]),'Public original query identity')
                count+=1
    return dict(new_packet_rows=count,all4288_new_saved_rows_scalar_and_originalMETA_bytes=True,no_input_pixels_decoded=True)

def run(entry):
    e=json.loads(raw(entry));need(e['approved_actual_independent_saved_result_audit']is True and e['all_two_real_predictions_frozen']is True and e['saved_evaluation_completed']is True and e['automatic_retry']is False,'Separate post-prediction/evaluation root audit entry')
    need(e['code']==bind(__file__)and e['protocol']==bind(PROTO),'Frozen own code/protocol');p=json.loads(checked(e['protocol']));need(p['code']==e['code'],'Protocol/code exact')
    need(sys.executable==p['runtime']['executable']and np.__version__==p['runtime']['numpy'],'Pinned saved-only scalar runtime');checked(e['root_scope'])
    need(e['complete']['path']==str(OUT/'COMPLETE.json'),'Exact new saved evaluation');complete=json.loads(checked(e['complete']));need(complete['completed']and complete['PASS']and complete['code']==p['evaluator_code']and complete['protocol']==p['evaluator_protocol']and complete['all_both_predictions_closed_before_any_annotation_or_score_body']is True and complete['selected_model']is None and complete['models']==list(MODELS)and complete['old_eight_primary_metrics_exact']is True,'Saved evaluation closed without model selection')
    ep=json.loads(checked(p['evaluator_protocol']));evaluation_entry=json.loads(checked(complete['entry']));seal=json.loads(checked(complete['prediction_seal']));need(evaluation_entry['prediction_seal']==complete['prediction_seal'],'Same all2 predictionseal before label gate')
    prediction_gate(seal,p)
    need(ep['prior_eight_evaluation_complete']==p['prior_eight_evaluation_complete'],'Exact root-bound old8 evaluation metadata contract')
    marker=json.loads(checked(complete['outputs']['BEFORE_ANNOTATION_AND_SAVED_SCORE_READ.json']));need(marker['both_models_closed']is True and marker['annotation_or_score_bodies_read_so_far']is False and marker['prediction_seal']==complete['prediction_seal'],'Prospective saved evaluation body-read barrier')
    # No raw prediction/input array or original annotation was read above.
    destination=safe(p['output_directory']);need(destination==V/'independent_saved_result_review_c/actual_review_v1','Exact fresh future audit destination')
    destination.parent.mkdir(exist_ok=True);need(stat.S_ISDIR(destination.parent.lstat().st_mode),'Ordinary audit output parent, never symlink')
    destination.mkdir(exist_ok=False)
    try:
        data={name:checked(b)for name,b in complete['outputs'].items()if name.startswith(('current1784/','publicheld360/'))}
        current=jsonrows(data[FILES[0]]);public=jsonrows(data[FILES[4]]);summary=json.loads(data[FILES[1]]);psummary=json.loads(data[FILES[5]])
        base=current_audit(current,summary,csvrows(data[FILES[2]]),csvrows(data[FILES[3]]));weak=public_audit(public,psummary)
        truth=csvrows(checked(ep['current']['truth']));manual=csvrows(checked(ep['current']['manual']));tables={}
        for k,v in summary['tables'].items():
            if isinstance(v,list):
                filename='current1784/'+k.upper()+'.csv';need(not v or filename in data,'Nonempty table has CSV');tables[k]=csvrows(data[filename])if filename in data else[]
        extended=extended_tables(current,summary,truth,manual,tables);pred=prediction_audit(seal,p,current,public)
        old=json.loads(checked(ep['prior_eight_evaluation_complete']));need(old['completed']and old['PASS'],'Closed old8 evaluation')
        for filename,newrows in[(FILES[0],current),(FILES[4],public)]:
            prior=jsonrows(checked(old['outputs'][filename]));need(len(prior)==len(newrows),'Old rows preserved')
            for oldrow,newrow in zip(prior,newrows):need(all(newrow[k]==v for k,v in oldrow.items()),'All original old8 fields/weak truth unchanged')
        oldmetrics=csvrows(checked(old['outputs'][FILES[2]]));newmetrics=[r for r in summary['tables']['metrics']if r['model']in MODELS[:8]];need(len(oldmetrics)==len(newmetrics),'All old8 metric rows retained')
        for oldrow,newrow in zip(oldmetrics,newmetrics):
            for k,v in newrow.items():
                try:ok=close(oldrow[k],v)
                except(ValueError,TypeError):ok=str(oldrow[k])==str(v)
                need(ok,'Old8 metric count/ratio/schema unchanged')
        for b in complete['outputs'].values():checked(b)
        for b in(ep['current']['truth'],ep['current']['manual'],ep['prior_eight_evaluation_complete'],old['outputs'][FILES[0]],old['outputs'][FILES[2]],old['outputs'][FILES[4]],complete['prediction_seal'],e['complete']):checked(b)
        report=dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(PROTO),actual_entry=bind(entry),complete=e['complete'],current=base,public=weak,all_existing_tables=extended,new_prediction_packets=pred,old8_saved_ledger_fields_byte_binding_unchanged=True,primary1237_D1147_E90=True,self_all136_paired21_separate=True,reference461_manual376_separate=True,public_weak_only=True,all10_current_and_public9_contract=True,actual_model_instantiation_forward_fit_or_image_read=0,automatic_retry=False,selected_model=None,adopted=False,fully_solved=False,limits=['Evaluation-only saved numerical audit. No trained model/weights/RGB/crop/feature read or forward, no model/threshold/data/architecture selection.','Old8 sealed ledger fields and source file SHA preserved; old inference not rerun. Existing human semantic authority and unknown mapping retained.','Public360 assistant weak point-owner/crop consensus is a separate task from current native D/E/S; legacy historical realCAL/repeated exposure is not fresh validation.','Manual/reference <=3px is a frozen association proxy, not physical coordinate gold or an evaluation filter.'])
        write(destination/'REVIEW.json',report);print(json.dumps(dict(completed=True,PASS=True,review=bind(destination/'REVIEW.json'))),flush=True)
    except BaseException:
        write(destination/'FAILURE.json',dict(completed=False,PASS=False,traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--selftest',action='store_true');g.add_argument('--entry',type=Path);args=ap.parse_args();print(json.dumps(selftest()))if args.selftest else run(args.entry)

