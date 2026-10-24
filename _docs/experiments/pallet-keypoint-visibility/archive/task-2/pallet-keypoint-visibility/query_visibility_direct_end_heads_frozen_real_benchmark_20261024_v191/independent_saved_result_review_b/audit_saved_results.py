"""Root-entry-gated scalar saved-ledger audit. No producer imports or arrays."""
from pathlib import Path
from collections import Counter
import argparse
import csv
import hashlib
import io
import json
import math
import os
import stat
import sys

H = Path(__file__).absolute().parent
V = H.parent
OUT = V / 'saved_real_head_evaluation_v1'
PROTO = H / 'PROTOCOL_BEFORE_RESULT_BODIES.json'
COMPLETE_SHA = '46954e8b9daa8c25d6b548ee5d6ac6421e389da520d3fc29d5727e3842300391'
MODELS = ('v61', 'legacy', 'C52', 'C53', 'C54', 'C55')
TAGS = MODELS[2:]
GROUPS = ('WOOD_PRIMARY105', 'SQUARE118')
D, E, S = 'DIRECT_VISIBLE', 'EXTERNAL_OCCLUDED', 'SELF_OCCLUDED'
OWNER = ('TARGET_VISIBLE_SUPPORTED', 'FOREIGN_VISIBLE_SUPPORTED', 'BACKGROUND_SUPPORTED')
FILES = (
    'current1784/ALL1784_CURRENT_NATIVE_LEDGER.jsonl',
    'current1784/CURRENT_HUMAN_SEMANTIC_AND_SEPARATE_ASSOCIATIONS.json',
    'current1784/METRICS.csv',
    'current1784/D_VS_EXTERNAL_E_2X3.csv',
    'publicheld360/ALL360_PUBLIC_WEAK_LEDGER.jsonl',
    'publicheld360/PUBLIC_WEAK_OWNER_AND_SEPARATE_SUPPORT_ASSOCIATIONS.json',
)


def need(ok, text):
    if not bool(ok):
        raise AssertionError(text)


def safe(path):
    p = Path(path).absolute()
    need(V in p.parents and '..' not in p.parts and all('v112' not in a.casefold() for a in p.parts), 'Safe exact V191 name before access')
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
            if m in ('C54', 'C55'):
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
    need(checked_count == 24, 'Both fixed current cohorts x2groups x6models')
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
    masks = dict(FULL360=[True] * 360, KNOWN_CENTER_TRUTH=[t >= 0 for t in truth], ALL5_MODEL_PAIRED_AVAILABLE=paired, KNOWN_TRUTH_AND_ALL5_PAIRED_AVAILABLE=[t >= 0 and paired[i] for i, t in enumerate(truth)])
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
    return dict(rows360=True, weak_known=204, weak_UNKNOWN=156, cohorts4_nativegroups9=True, v61_own_U=109, four_new_owner_U=108, no_native_external_gold=True)


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


def run(entry):
    need(sys.executable == '/home/minjae/anaconda3/envs/pallet-yolo26/bin/python' and sys.version_info[:3] == (3, 10, 20), 'Pinned stdlib scalar runtime')
    e = json.loads(raw(entry))
    need(e['approved_actual_independent_saved_result_audit'] is True and e['all_two_real_predictions_frozen'] is True and e['saved_evaluation_completed'] is True and e['automatic_retry'] is False, 'Separate root actual saved-only authorization')
    need(e['code'] == bind(__file__) and e['protocol'] == bind(PROTO), 'Frozen own code/protocol')
    protocol = json.loads(checked(e['protocol']));need(protocol['code'] == e['code'], 'Protocol/code closure')
    checked(e['root_scope'])
    need(e['complete'] == dict(path=str(OUT / 'COMPLETE.json'), sha256=COMPLETE_SHA), 'Exact completed evaluation')
    complete = json.loads(checked(e['complete']))
    need(complete['completed'] and complete['PASS'] and complete['code'] == protocol['evaluator_code'] and complete['protocol'] == protocol['evaluator_protocol'], 'Frozen actual evaluator closure')
    need(complete['all_both_predictions_closed_before_any_annotation_or_score_body'] and complete['selected_model'] is None, 'Both closed, no selection')
    destination = safe(e['output_directory']);need(destination == H / 'actual_review_v1', 'Fresh own output path');destination.mkdir(exist_ok=False)
    try:
        data = {};bindings = []
        for relative in FILES:
            binding = complete['outputs'][relative];need(binding['path'] == str(OUT / relative), 'Only newly saved result files')
            data[relative] = checked(binding);bindings.append(binding)
        jl = lambda name: [json.loads(s) for s in data[name].splitlines() if s.strip()]
        current = current_audit(jl(FILES[0]), json.loads(data[FILES[1]]), list(csv.DictReader(io.StringIO(data[FILES[2]].decode()))), list(csv.DictReader(io.StringIO(data[FILES[3]].decode()))))
        public = public_audit(jl(FILES[4]), json.loads(data[FILES[5]]))
        for binding in bindings:checked(binding)
        checked(e['complete'])
        result = dict(completed=True, PASS=True, code=bind(__file__), protocol=bind(PROTO), actual_entry=bind(entry), complete=e['complete'], consumed_files=bindings, current=current, public=public, producer_imports=0, model_or_image_or_original_annotation_or_old_score_body_reads=0, automatic_retry=False,
                      limits=['Saved-ledger scalar verification only; no original raw annotation or prediction NPZ reread. No physical/native gold certification.', 'Self/reference association recomputed; manualnear153/223 fixed-table coverage preserved without reconstructing membership.', 'Repeated evaluation exposure remains; no training, threshold, architecture, dataset, checkpoint or model selection.'])
        write(destination / 'REVIEW.json', result)
        print(json.dumps(dict(completed=True, PASS=True, review=bind(destination / 'REVIEW.json'))))
    except BaseException as exc:
        write(destination / 'FAILURE.json', dict(completed=False, PASS=False, error_type=type(exc).__name__, error=str(exc), automatic_retry=False))
        raise


if __name__ == '__main__':
    ap = argparse.ArgumentParser();g = ap.add_mutually_exclusive_group(required=True);g.add_argument('--selftest', action='store_true');g.add_argument('--entry', type=Path);args = ap.parse_args()
    print(json.dumps(selftest())) if args.selftest else run(args.entry)
