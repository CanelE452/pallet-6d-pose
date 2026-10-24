"""V91 saved-only transfer comparison. No import-time I/O or model imports.

Only generated fixtures may run during preparation. Actual execution requires
root-approved source audit/current protocol/complete inference bindings first.
Raw zE, not rounded probability, is authoritative for the V91 route.
"""
import argparse
import csv
from collections import Counter
import hashlib
import html
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PALLET = ROOT.parent
PROTOCOL = ROOT / 'CURRENT_PROTOCOL_BEFORE_CURRENT_ARTIFACTS.json'
PROTOCOL_SHA = '396747f56b6975da47bdae0a14c0dcbdf1b19e011f887c863f7e73b1f83b756c'
BASE = PALLET / 'query_visibility_cue_occlusion_20261023_v72/contract/saved_evaluate_current_scores.py'
RAW = 'V91_RAW'
GEOMETRY_INTERNAL = RAW + '+GEOMETRY'
GEOMETRY_PUBLIC = 'V91_PLUS_OLD_GEOMETRY'
OLD = 'V76_SUPPORT'
FIXED_CURRENT_PATHS = dict(
    evaluation_manifest=PALLET/'evaluation_preparation_20261022/PREPARED_QUERY_MANIFEST.csv',
    manual_provenance=PALLET/'near_error_diagnosis_20261022/real/NEAR_DIRECT_ROWS_WITH_PROVENANCE.csv',
    prior_rows=PALLET/'query_visibility_cue_occlusion_20261023_v72/private_evaluation/CUE_ATTENTION_V3/ROWS_INPUT_ONLY.jsonl',
    baseline_v76_scores=PALLET/'query_visibility_reference_local_support_20261023_v76/private_evaluation/MIXED_SUPPORT_V1/REAL_QUERY_SCORES.csv',
    baseline_v76_inference_seal=PALLET/'query_visibility_reference_local_support_20261023_v76/private_evaluation/MIXED_SUPPORT_V1/INFERENCE_SEAL.json',
    historical_baseline_metrics=PALLET/'query_visibility_reference_local_support_20261023_v76/contract/V76_CURRENT_CLASSIFICATION_METRICS.csv')
REQUIRED = ('protocol', 'source_postfit_audit', 'source_model_freeze', 'checkpoint',
    'inference_seal', 'scores', 'preparation_seal', 'artifact_bindings',
    'baseline_v76_scores', 'baseline_v76_inference_seal', 'historical_baseline_metrics',
    'evaluation_manifest', 'manual_provenance', 'prior_rows', 'base_evaluator',
    'evaluator_code', 'generated', 'independent_review')
SUFFIXES = ('CLASSIFICATION_METRICS', 'CONFUSION_BY_HUMAN_STATE', 'ERROR_STRATA',
            'GEOMETRY_TRADEOFF', 'PER_RECORDING_METRICS', 'PAIRED_CHANGE')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def bind(path): return {'path': str(Path(path).resolve()), 'sha256': sha(path)}
def read(path): return json.loads(Path(path).read_text())


def check(item):
    if not isinstance(item, dict) or not {'path', 'sha256'} <= set(item):
        raise PermissionError('Missing immutable binding')
    if sha(item['path']) != item['sha256']:
        raise PermissionError('Changed binding: ' + item['path'])


def same(a, b):
    return Path(a['path']).resolve() == Path(b['path']).resolve() and a['sha256'] == b['sha256']


def contains_binding(tree, item):
    if isinstance(tree, dict):
        if {'path', 'sha256'} <= set(tree) and same(tree, item): return True
        return any(contains_binding(v, item) for v in tree.values())
    if isinstance(tree, list): return any(contains_binding(v, item) for v in tree)
    return False


def write_new(path, data):
    with Path(path).open('x') as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def readcsv(path):
    with Path(path).open(newline='') as f: return list(csv.DictReader(f))


def writecsv(path, rows):
    if not rows: return
    with Path(path).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def bool_field(value, allow_none=False):
    text = str(value).lower()
    if text == 'true': return True
    if text == 'false': return False
    if allow_none and text in ('', 'none', 'null'): return None
    raise ValueError('Invalid explicit boolean: ' + str(value))


def key(row): return (row['id'], str(row['native_corner']))


def authoritative_call(zE, valid, geometry=False):
    # Invalid input must win even before an old geometry flag.
    if not valid: return 'OTHER_UNRESOLVED'
    if geometry is True: return 'ABSENT_NONVISIBLE'
    if zE is None or not math.isfinite(zE): raise ValueError('Valid raw zE required')
    return 'VISIBLE' if zE < 0 else 'ABSENT_NONVISIBLE'


def source_and_inference_gate(entry_path):
    """Source authorization before any current artifact SHA or body read."""
    entry = read(entry_path)
    if entry.get('schema') != 'V91_SAVED_CURRENT_EVALUATION_ENTRY_V1':
        raise PermissionError('Wrong evaluator entry schema')
    for name in ('approved', 'source_selection_complete', 'source_independent_audit_PASS',
                 'current_inference_complete', 'current_evaluation_authorized'):
        if entry.get(name) is not True: raise PermissionError('Not approved: ' + name)
    if set(REQUIRED) - set(entry): raise PermissionError('Missing entry bindings')
    for name,path in FIXED_CURRENT_PATHS.items():
        if Path(entry[name]['path']).resolve()!=path.resolve():raise PermissionError('Unapproved historical/cohort path: '+name)
    # These are code/source artifacts, not current packets.
    for name in ('protocol', 'source_postfit_audit', 'source_model_freeze', 'checkpoint',
                 'base_evaluator', 'evaluator_code', 'generated', 'independent_review'):
        check(entry[name])
    if not same(entry['evaluator_code'], bind(__file__)) or not same(entry['base_evaluator'], bind(BASE)):
        raise PermissionError('Code mismatch')
    if Path(entry['protocol']['path']).resolve() != PROTOCOL.resolve() or entry['protocol']['sha256'] != PROTOCOL_SHA:
        raise PermissionError('Prospective current protocol changed')
    protocol = read(PROTOCOL)
    if protocol.get('source_gate_all9_pass') is not True or protocol.get('PnP_allowed') is not False:
        raise PermissionError('Source gate/protocol scope')
    if not same(protocol['source_model_freeze'], entry['source_model_freeze']):
        raise PermissionError('Source candidate changed')
    check(protocol['source_fit_completion']); check(protocol['source_results'])
    result = read(protocol['source_results']['path'])
    if result.get('all_nine_pass') is not True or result.get('current_evaluation_admission') is not False:
        raise PermissionError('All nine source gates must pass; no automatic old admission')
    audit = read(entry['source_postfit_audit']['path'])
    if audit.get('PASS') is not True or not contains_binding(audit, entry['source_model_freeze']):
        raise PermissionError('Wrong or failed independent source audit')
    freeze = read(entry['source_model_freeze']['path'])
    if freeze.get('completed') is not True or freeze.get('optimizer_updates') != 2000:
        raise PermissionError('Incomplete source fit')
    if freeze.get('current_used_for_fit_selection') is not False or freeze.get('PnP') != 0:
        raise PermissionError('Source selection scope')
    if not same(freeze['checkpoint'], entry['checkpoint']): raise PermissionError('Wrong checkpoint')
    if read(entry['generated']['path']).get('PASS') is not True:
        raise PermissionError('Generated proof failed')
    review = read(entry['independent_review']['path'])
    if review.get('PASS') is not True or not contains_binding(review, entry['evaluator_code']):
        raise PermissionError('Wrong evaluator review')
    # Source checks passed. Only now may any current artifact be hashed/read.
    for name in ('artifact_bindings', 'inference_seal', 'preparation_seal', 'scores'):
        check(entry[name])
    artifacts = read(entry['artifact_bindings']['path'])
    if artifacts.get('completed') is not True or artifacts.get('authorized_after_source_audit') is not True:
        raise PermissionError('Root artifact authorization absent')
    if artifacts.get('current_tensor_RGB_GT_score_deserialization')!=0 or artifacts.get('current_forward_fit_PnP')!=0:
        raise PermissionError('Metadata-only authorization scope changed')
    inference = read(entry['inference_seal']['path'])
    for name, expected in dict(completed=True, frames=223, slots=1784, model_unchanged=True,
        current_used_for_selection=False, current_GT_read=False, new_backbone=0, optimizer_updates=0, PnP=0).items():
        if inference.get(name) != expected: raise PermissionError('Inference invariant: ' + name)
    if not same(inference['source_model_freeze'], entry['source_model_freeze']) or not same(inference['checkpoint'], entry['checkpoint']):
        raise PermissionError('Inference source mismatch')
    if not same(inference['preparation'], entry['preparation_seal']): raise PermissionError('Preparation changed')
    if not any(same(item, entry['scores']) for item in inference['outputs']): raise PermissionError('Unsealed scores')
    if inference.get('raw_E_threshold_logit') != 0 or inference.get('D_rule') != 'zE<0':
        raise PermissionError('Authoritative source decoder changed')
    prep = read(entry['preparation_seal']['path'])
    if prep.get('completed') is not True: raise PermissionError('Incomplete current preparation')
    if not contains_binding(prep, entry['source_model_freeze']): raise PermissionError('Preparation source mismatch')
    # All downstream current packets are fixed before their bodies are opened.
    for name in ('baseline_v76_scores', 'baseline_v76_inference_seal', 'historical_baseline_metrics',
                 'evaluation_manifest', 'manual_provenance', 'prior_rows'):
        check(entry[name])
    oldseal = read(entry['baseline_v76_inference_seal']['path'])
    if oldseal.get('completed') is not True or not any(same(item, entry['baseline_v76_scores']) for item in oldseal['outputs']):
        raise PermissionError('Historical V76 outputs not sealed')
    return entry, freeze, inference


def import_base(output):
    spec = importlib.util.spec_from_file_location('v91_immutable_v72_saved_evaluator', BASE)
    base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
    base.HERE = Path(output); base.ROOT = PALLET
    return base


def score_records(records, model):
    if len(records) != 1784: raise AssertionError('Every native slot must be retained')
    data = {}; invalid = 0
    for row in records:
        k = key(row)
        if k in data: raise AssertionError('Duplicate saved key')
        valid = bool_field(row['valid'])
        old_geometry = bool_field(row['geometry_self_hidden'], allow_none=True)
        geometry = bool_field(row['geometry_self_hidden_if_valid'], allow_none=True)
        if valid and geometry != old_geometry: raise AssertionError('Saved geometry changed')
        if not valid and geometry is not None: raise AssertionError('Invalid input acquired geometry negative')
        if model == RAW:
            raw = float(row['raw_physical_external_logit'])
            if not math.isfinite(raw): raise AssertionError('All raw zE must be finite')
            prediction = authoritative_call(raw, valid)
            exported = row['authoritative_decision']
            expected = 'D' if prediction == 'VISIBLE' else 'E' if prediction == 'ABSENT_NONVISIBLE' else 'UNKNOWN'
            if exported != expected: raise AssertionError('Authoritative decision disagrees with raw zE')
            visible = bool_field(row['decision_visible'], allow_none=True)
            if visible != ((raw < 0) if valid else None): raise AssertionError('Decision indicator mismatch')
        else:
            raw = None
        probability = row.get('p_visible', '')
        if valid:
            if not probability or not math.isfinite(float(probability)) or not 0 <= float(probability) <= 1:
                raise AssertionError('Valid display probability missing')
        else:
            invalid += 1
            if probability: raise AssertionError('Invalid input must have unresolved display probability')
        data[k] = dict(row=row, valid=valid, raw_zE=raw, geometry=geometry,
            p_visible=float(probability) if probability else None)
    return data, dict(rows=len(records), valid=len(records)-invalid, invalid=invalid)


def verify_saved_join(new, old, prior, truth):
    pmap = {key(r):r for r in prior}; tmap = {key(r):r for r in truth}
    if len(pmap) != 1784 or len(tmap) != 1784 or set(new) != set(old) or set(new) != set(pmap) or set(new) != set(tmap):
        raise AssertionError('Original 1784 native key coverage changed')
    if list(new)!=list(old) or list(new)!=list(pmap):raise AssertionError('Original inference/native row order changed')
    for k, v in new.items():
        n, o, p, t = v['row'], old[k]['row'], pmap[k], tmap[k]
        for name in ('image_sha256', 'query_x', 'query_y', 'roi_side', 'pixel_x', 'pixel_y', 'saved_roi_pixel_valid', 'geometry_self_hidden'):
            if n[name] != o[name] or n[name] != str(p[name]):
                # Original JSON number formatting may differ from equivalent CSV.
                if name in ('query_x', 'query_y', 'roi_side', 'pixel_x', 'pixel_y'):
                    if n[name] in ('', 'None') and o[name] in ('', 'None') and p[name] is None: continue
                    if float(n[name]) == float(o[name]) == float(p[name]): continue
                if name in ('saved_roi_pixel_valid', 'geometry_self_hidden'):
                    if bool_field(n[name], True) == bool_field(o[name], True) == p[name]: continue
                raise AssertionError(('Original metadata parity', k, name))
        if n['image_sha256'] != t['image_sha256']: raise AssertionError('Truth image mismatch')
        for name in ('query_x', 'query_y'):
            if n[name] and t[name] and abs(float(n[name])-float(t[name])) > 1e-5:
                raise AssertionError('Registered query changed')
        if v['valid'] and not bool_field(p['valid']): raise AssertionError('Old factual U became valid')
    return dict(all1784_key_coverage=True, original_id_native_order_image_q_L_pixel_valid_geometry=True,
        old_invalid_retained=True, expected_original_U=45, new_valid=sum(v['valid'] for v in new.values()),
        new_invalid=sum(not v['valid'] for v in new.values()))


def install_authoritative_decoder(base, new):
    previous = base.call
    def call(row, model, task, probabilities, threshold, rules=None):
        if model not in (RAW, GEOMETRY_INTERNAL):
            return previous(row, model, task, probabilities, threshold, rules)
        value = new[key(row)]
        return authoritative_call(value['raw_zE'], value['valid'], value['geometry'] if model == GEOMETRY_INTERNAL else False)
    base.call = call


def run_base(base, scores, model, prefix, output, freeze_source, geometry_column):
    adjunct = Path(output) / (prefix+'_COMPATIBILITY_ADJUNCT.json')
    write_new(adjunct, dict(threshold=.5, threshold_comparison='>=', model=model,
        authoritative_raw_decoder='raw_zE<0 D;elsebinarynonvisible;invalidU' if model == RAW else 'historical p_visible>=.5',
        source_model_freeze=freeze_source, current_eval_used_for_selection=False,
        private_eval_used_for_selection=False, real_eval_used_for_selection=False, new_PnP_calls=0))
    before = sys.argv
    try:
        sys.argv = [str(BASE), '--scores', str(scores), '--freeze', str(adjunct), '--score-column', 'p_visible',
            '--model', model, '--output-prefix', prefix, '--include-geometry', '--geometry-column', geometry_column,
            '--allow-unresolved-scores']
        base.main()
    finally: sys.argv = before


def alias_row(row):
    return {k: (GEOMETRY_PUBLIC if v == GEOMETRY_INTERNAL else v) for k,v in row.items()}


def unique_rows(rows):
    seen = set(); result = []
    for row in rows:
        token = json.dumps(row, sort_keys=True)
        if token not in seen: seen.add(token); result.append(row)
    return result


def cross_v76_changes(base, truth, new, old):
    rows = []
    def old_call(r, hybrid):
        v = old[key(r)]
        if hybrid and v['geometry'] is True: return 'ABSENT_NONVISIBLE'
        if not v['valid']: return 'OTHER_UNRESOLVED'
        return 'VISIBLE' if v['p_visible'] >= .5 else 'ABSENT_NONVISIBLE'
    for cohort in ('paired_same_query_RGB', 'same_slot_pipeline'):
        for group in base.GROUPS:
            for task, states in base.TASKS.items():
                flag = cohort+'_'+('D_vs_ES' if task == 'D_vs_S' else task)
                eligible = [r for r in truth if r['group'] == group and base.yes(r, flag) and r['human_state'] in states]
                if not any(r['human_state'] != base.D for r in eligible): continue
                for old_hybrid in (False, True):
                    for new_hybrid in (False, True):
                        counts = Counter()
                        for r in eligible:
                            a = old_call(r, old_hybrid); v = new[key(r)]
                            b = authoritative_call(v['raw_zE'],v['valid'],v['geometry'] if new_hybrid else False)
                            if b not in ('VISIBLE','ABSENT_NONVISIBLE'): counts['new_unresolved'] += 1; continue
                            gt = 'VISIBLE' if r['human_state'] == base.D else 'ABSENT_NONVISIBLE'
                            if a not in ('VISIBLE','ABSENT_NONVISIBLE'):
                                counts['baseline_unresolved_new_correct' if b == gt else 'baseline_unresolved_new_wrong'] += 1
                            else:
                                counts['both_correct' if a == gt and b == gt else 'baseline_correct_new_wrong' if a == gt else 'baseline_wrong_new_correct' if b == gt else 'both_wrong'] += 1
                        names = ('both_correct','baseline_correct_new_wrong','baseline_wrong_new_correct','both_wrong',
                                 'baseline_unresolved_new_correct','baseline_unresolved_new_wrong','new_unresolved')
                        rows.append(dict(cohort=cohort,group=group,task=task,baseline=OLD+('+GEOMETRY' if old_hybrid else ''),
                            new_model=GEOMETRY_PUBLIC if new_hybrid else RAW,N=len(eligible),**{n:counts[n]for n in names}))
    return rows


def manual_table(base, provenance, truth, new, old):
    bykey = {key(r):r for r in truth}
    selected = [r for r in provenance if r['annotation_coordinate_source'] == 'manual_click'
        and r['human_state'] == base.D and float(r['initial_reference_error_px']) <= 3 and base.yes(r,'query_valid')]
    for r in selected:
        original=bykey[key(r)]
        if any(r[k] != original[k] for k in ('image_sha256','query_x','query_y')): raise AssertionError('Manual near identity changed')
    expected={'WOOD_PRIMARY105':153,'SQUARE118':223}
    rows=[]
    for group in base.GROUPS:
        rr=[r for r in selected if r['group']==group]
        if len(rr) != expected[group]: raise AssertionError(('Manual historical cohort changed',group,len(rr)))
        for model in ('v61','legacy',OLD,OLD+'+GEOMETRY',RAW,GEOMETRY_INTERNAL):
            calls=Counter()
            for r in rr:
                if model.startswith(OLD):
                    v=old[key(r)]; prediction='OTHER_UNRESOLVED' if not v['valid'] else 'ABSENT_NONVISIBLE' if model.endswith('+GEOMETRY') and v['geometry'] is True else 'VISIBLE' if v['p_visible']>=.5 else 'ABSENT_NONVISIBLE'
                else: prediction=base.call(r,model,'D_vs_ES',{},.5)
                calls[prediction]+=1
            rows.append(dict(group=group,scope='MANUAL_CLICK_DIRECT_VISIBLE_QUERYVALID_ERROR_LE3',
                model=GEOMETRY_PUBLIC if model==GEOMETRY_INTERNAL else model,N=len(rr),pred_visible=calls['VISIBLE'],
                pred_absent_nonvisible=calls['ABSENT_NONVISIBLE'],pred_legacy_U=calls['ABSTAIN'],
                pred_other_unresolved=calls['OTHER_UNRESOLVED'],pred_out_of_task_S=calls['OUT_OF_TASK_GEOMETRY_S'],
                semantic_D_recall=calls['VISIBLE']/len(rr),exact_pixel_GT='UNAVAILABLE'))
    return rows


def verify_historical_counts(combined, historical):
    fields=('N','frames','visible_support','occluded_support','TP','FN_decided','FP','TN','U_visible','U_occluded',
        'out_of_task_S_visible','out_of_task_S_occluded','other_unresolved_visible','other_unresolved_occluded')
    lookup={(r['cohort'],r['group'],r['task'],r['model']):r for r in historical if r['model'] in ('v61','legacy',OLD,OLD+'+GEOMETRY')}
    checked=0
    for row in combined:
        if row['model'] not in ('v61','legacy',OLD,OLD+'+GEOMETRY'): continue
        original=lookup[(row['cohort'],row['group'],row['task'],row['model'])]
        for name in fields:
            if int(row[name]) != int(original[name]): raise AssertionError(('Historical baseline denominator/count drift',row['model'],name))
        checked+=1
    if checked != len(lookup): raise AssertionError('Historical metric row coverage changed')
    return dict(PASS=True,baseline_metric_rows_checked=checked,counts_and_denominators_exact=True)


def compact_confusions(rows):
    result=[]
    for row in rows:
        get=lambda name:int(row[name])
        up=get('U_visible')+get('out_of_task_S_visible')+get('other_unresolved_visible')
        un=get('U_occluded')+get('out_of_task_S_occluded')+get('other_unresolved_occluded')
        result.append(dict(cohort=row['cohort'],group=row['group'],task=row['task'],model=row['model'],N=get('N'),
            GT_D_pred_VISIBLE=get('TP'),GT_D_pred_ABSENT=get('FN_decided'),GT_D_pred_UNRESOLVED=up,
            GT_NEG_pred_VISIBLE=get('FP'),GT_NEG_pred_ABSENT=get('TN'),GT_NEG_pred_UNRESOLVED=un,
            unresolved_components='LegacyABSTAIN + out_of_task_geometryS + factualOTHER_UNRESOLVED; separate fields retained in full metrics'))
    return result


def render_tables(path, tables, manual, availability):
    def table(rows):
        if not rows:return '<p>No rows</p>'
        cols=list(rows[0])
        return '<table><thead><tr>'+''.join('<th>'+html.escape(c)+'</th>'for c in cols)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(c,'')))+'</td>'for c in cols)+'</tr>'for r in rows)+'</tbody></table>'
    text='<!doctype html><html lang="ko"><meta charset="utf-8"><title>V91 고정 저장 출력 비교</title><style>body{font:14px sans-serif;margin:24px}table{border-collapse:collapse;display:block;overflow:auto;margin:20px 0}th,td{border:1px solid #ccc;padding:7px;white-space:nowrap}</style><h1>V91 고정 query centre 저장 출력 비교</h1><p>Raw zE&lt;0만 visible. Binary absence는 인과 E/S 확정이 아닙니다. Geometry는 별도 saved proxy route이며 invalid U를 먼저 보존합니다. 재사용 현재 평가이며 source 약한 물리 cue와 native8 정답의 동등성을 뜻하지 않습니다.</p>'
    text+='<h2>전체 입력 availability</h2>'+table([availability])+'<h2>고정 cohort</h2>'+table(tables['CLASSIFICATION_METRICS'])+'<h2>수동 좌표 ≤3px D</h2>'+table(manual)
    for name in ('BINARY_2X3_CONFUSION','CONFUSION_BY_HUMAN_STATE','GEOMETRY_TRADEOFF','ERROR_STRATA','PAIRED_CHANGE','V91_V76_PAIRED_CHANGE','PER_RECORDING_METRICS'):
        text+='<h2>'+html.escape(name)+'</h2>'+table(tables[name])
    with Path(path).open('x')as f:f.write(text+'</html>\n')


def actual(entry_path, output):
    entry, freeze, inference=source_and_inference_gate(entry_path)
    output=Path(output);output.mkdir(exist_ok=False)
    base=import_base(output)
    records=readcsv(entry['scores']['path']);oldrecords=readcsv(entry['baseline_v76_scores']['path'])
    new,audit=score_records(records,RAW);old,old_audit=score_records(oldrecords,OLD)
    prior=lines_input_rows(entry['prior_rows']['path']);truth=readcsv(entry['evaluation_manifest']['path'])
    joined=verify_saved_join(new,old,prior,truth)
    # Exact root protocol requests the historical 45 U to be retained, with any additional U explicit.
    old_invalid=sum(not bool_field(r['valid'])for r in prior)
    if old_invalid != 45:raise AssertionError('Original 45-input-U cohort changed')
    if audit['invalid'] < old_invalid:raise AssertionError('Invalid input promoted')
    install_authoritative_decoder(base,new)
    run_base(base,entry['scores']['path'],RAW,'V91_RAW_SAVED',output,entry['source_model_freeze'],'geometry_self_hidden_if_valid')
    run_base(base,entry['baseline_v76_scores']['path'],OLD,'V76_SAVED_REPLAY',output,{},'geometry_self_hidden_if_valid')
    tables={}
    for suffix in SUFFIXES:
        rows=[]
        for prefix in ('V91_RAW_SAVED','V76_SAVED_REPLAY'):
            file=output/(prefix+'_'+suffix+'.csv')
            if file.exists():rows.extend(readcsv(file))
        tables[suffix]=unique_rows([alias_row(r)for r in rows])
        writecsv(output/('CURRENT_'+suffix+'.csv'),tables[suffix])
    historical=verify_historical_counts(tables['CLASSIFICATION_METRICS'],readcsv(entry['historical_baseline_metrics']['path']))
    tables['BINARY_2X3_CONFUSION']=compact_confusions(tables['CLASSIFICATION_METRICS'])
    writecsv(output/'CURRENT_BINARY_2X3_CONFUSION.csv',tables['BINARY_2X3_CONFUSION'])
    tables['V91_V76_PAIRED_CHANGE']=cross_v76_changes(base,truth,new,old)
    writecsv(output/'CURRENT_V91_V76_PAIRED_CHANGE.csv',tables['V91_V76_PAIRED_CHANGE'])
    manual=manual_table(base,readcsv(entry['manual_provenance']['path']),truth,new,old)
    writecsv(output/'CURRENT_MANUAL_CLICK_NEAR_D_LE3.csv',manual)
    combined={**audit,'historical_V76_valid':old_audit['valid'],'historical_V76_invalid':old_audit['invalid'],
        'original_prior_U':old_invalid,'additional_V91_factual_U':audit['invalid']-old_invalid}
    tmap={key(r):r for r in truth};per_query=[]
    for k,value in new.items():
        r=value['row'];t=tmap[k]
        per_query.append(dict(id=k[0],native_corner=k[1],group=t['group'],human_state=t['human_state'],
            image_sha256=r['image_sha256'],query_x=r['query_x'],query_y=r['query_y'],roi_side=r['roi_side'],
            original_prior_valid=bool_field(next(p['valid']for p in prior if key(p)==k)),
            V91_valid=value['valid'],V91_raw_zE=value['raw_zE'],
            V91_raw_decision=authoritative_call(value['raw_zE'],value['valid']),
            V91_geometry_decision=authoritative_call(value['raw_zE'],value['valid'],value['geometry']),
            geometry_self_hidden_original=r['geometry_self_hidden'],geometry_self_hidden_if_valid=r['geometry_self_hidden_if_valid'],
            V76_valid=old[k]['valid'],V76_p_visible=old[k]['p_visible'],
            v61_saved_present=t['present_at_fixed_0_5'],legacy_stored_state=t['legacy_stored_state']))
    writecsv(output/'CURRENT_ALL1784_SAVED_QUERY_REPLAY.csv',per_query)
    result=dict(completed=True,model=RAW,geometry_route=GEOMETRY_PUBLIC,
        raw_authoritative_decoder='valid and zE<0 ->VISIBLE;valid and zE>=0 ->ABSENT_NONVISIBLE;invalid->OTHER_UNRESOLVED',
        probabilities_are_display_only=True,tables=tables,manual_click_near_D_LE3=manual,availability=combined,
        exact_original_input_join=joined,historical_baseline_replay=historical,
        geometry_precedence='InvalidU first; separately guarded oldgeometry True overrides to binary absence before raw, allother flags raw. No learned self claim.',
        U_scope='Legacy ABSTAIN, old geometry out-of-taskS and factual input OTHER_UNRESOLVED are separate. Total unresolved rate includes all; teacherU is not current correctU.',
        target_scope='Conditional source physical cue-region auxiliary transferred at original native query q. Binary absence is not causal E/S gold. Native8/target association unproved; no mesh support gate.',
        coordinate_scope='Manual-click DIRECT_VISIBLE validquery<=3 separately; general source reference error bins may include projected coordinates and are descriptive.',
        no_current_fit_threshold_checkpoint_reselection=True,new_forward_fit_PnP=0,
        bindings={**entry, 'evaluation_entry':bind(entry_path)})
    report=output/'CURRENT_PRIMARY_RESULTS.json';write_new(report,result)
    render_tables(output/'CURRENT_TABLES.html',tables,manual,combined)
    write_new(output/'EVALUATION_SEAL.json',dict(completed=True,PASS=True,entry=bind(entry_path),evaluator=bind(__file__),
        inference=entry['inference_seal'],source_model_freeze=entry['source_model_freeze'],report=bind(report),
        files=[bind(f)for f in sorted(output.iterdir())if f.is_file()],new_forward_fit_PnP=0))
    print(json.dumps(dict(completed=True,report=bind(report),seal=bind(output/'EVALUATION_SEAL.json'))))


def lines_input_rows(path):
    return [json.loads(x)for x in Path(path).read_text().splitlines()if x.strip()]


def generated():
    # Existing source code only, no current packet/manifest/GT/score/SHA read.
    base=import_base(HERE)
    for z,expected in ((-1e300,'VISIBLE'),(-5e-324,'VISIBLE'),(-0.,'ABSENT_NONVISIBLE'),(0.,'ABSENT_NONVISIBLE'),(5e-324,'ABSENT_NONVISIBLE'),(1e300,'ABSENT_NONVISIBLE')):
        if authoritative_call(z,True)!=expected:raise AssertionError('Raw sign decoder')
        if authoritative_call(z,False,True)!='OTHER_UNRESOLVED':raise AssertionError('Invalid priority')
    if authoritative_call(-4.,True,True)!='ABSENT_NONVISIBLE':raise AssertionError('Geometry separate override')
    # Rounded pD can be exactly .5 for nonzero logit; raw negative still visible.
    if math.exp(-math.log1p(math.exp(-5e-324))) != .5:raise AssertionError('Fixture rounding assumption')
    fake={('a','0'):dict(valid=True,raw_zE=-5e-324,geometry=False)}
    install_authoritative_decoder(base,fake)
    row=dict(id='a',native_corner='0',human_state=base.D,present_at_fixed_0_5='False',legacy_stored_state='U')
    if base.call(row,RAW,'D_vs_E',{('a','0'):.5},.5)!='VISIBLE':raise AssertionError('Probability must not decide')
    if base.call(row,'v61','D_vs_E',{},.5)!='ABSENT_NONVISIBLE':raise AssertionError('v61 decoder changed')
    if base.call(row,'legacy','D_vs_E',{},.5)!='ABSTAIN':raise AssertionError('Legacy decoder changed')
    fake[('a','0')]['valid']=False;fake[('a','0')]['geometry']=True
    if base.call(row,GEOMETRY_INTERNAL,'D_vs_S',{},.5)!='OTHER_UNRESOLVED':raise AssertionError('Geometry must not erase invalidU')
    # Exact base confusion denominators preserve invalid other-U and legacy U.
    fixture=[];new={}
    for i,(truth,z,valid)in enumerate(((base.D,-1.,True),(base.D,1.,True),(base.D,-1.,False),(base.E,-1.,True),(base.E,1.,True),(base.E,1.,False))):
        r=dict(id=str(i),native_corner='0',human_state=truth,present_at_fixed_0_5='True',legacy_stored_state='B');fixture.append(r)
        new[key(r)]=dict(valid=valid,raw_zE=z,geometry=False)
    b2=import_base(HERE);install_authoritative_decoder(b2,new)
    metric=b2.calculate(fixture,RAW,'D_vs_E',{},.5)
    if [metric[k]for k in ('TP','FN_decided','FP','TN','other_unresolved_visible','other_unresolved_occluded')]!=[1,1,1,1,1,1]:raise AssertionError('CM')
    if metric['visible_recall_all_GT']!=1/3 or metric['decision_coverage']!=2/3:raise AssertionError('Denominator')
    with tempfile.TemporaryDirectory(prefix='generated_eval_',dir=HERE)as tmp:
        gate=Path(tmp)/'entry.json';gate.write_text(json.dumps(dict(schema='V91_SAVED_CURRENT_EVALUATION_ENTRY_V1',approved=False)))
        try:source_and_inference_gate(gate)
        except PermissionError:pass
        else:raise AssertionError('Unapproved entry must stop before current metadata access')
    write_new(HERE/'GENERATED_EVALUATOR_TESTS.json',dict(completed=True,PASS=True,evaluator=bind(__file__),base_evaluator_code=bind(BASE),
        raw_sign_tiny_zero_ties=True,probability_rounded_tie_ignored=True,invalid_before_geometry=True,
        legacy_and_v61_rules_unchanged=True,base_CM_and_allGT_denominators=True,unapproved_before_current_access=True,
        actual_current_manifest_GT_scores_seal_body_SHA_reads=0,new_forward_fit_PnP=0))
    print(json.dumps(dict(PASS=True,proof=bind(HERE/'GENERATED_EVALUATOR_TESTS.json'))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['generated','run'])
    parser.add_argument('--entry',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.stage=='generated':generated()
    else:
        if args.entry is None or args.output is None:parser.error('--entry and --output required for root-approved run')
        actual(args.entry,args.output)
