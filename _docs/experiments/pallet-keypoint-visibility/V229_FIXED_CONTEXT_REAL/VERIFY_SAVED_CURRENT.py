"""Package root에서 실행: python VERIFY_SAVED_CURRENT.py

저장 ledger/표의 산술만 독립 재집계한다. GT 원문·입력·특징·모델·추론은
읽거나 실행하지 않는다. 원 human authority 및 source provenance 인증기는 아니다.
"""
import csv
import json
import math
import os
import stat
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).absolute().parent
TAG = 'V229_CONTROL_FIXED_Q'
MODELS = ('v61', 'legacy', TAG)
GROUPS = ('ALL', 'WOOD_PRIMARY105', 'SQUARE118')
D, E = 'DIRECT_VISIBLE', 'EXTERNAL_OCCLUDED'
RESPONSES = ('VISIBLE', 'ABSENT_NONVISIBLE', 'ABSTAIN',
             'OUT_OF_TASK_GEOMETRY_S', 'OTHER_UNRESOLVED')

def need(ok, message):
    if not ok:
        raise AssertionError(message)

def read_csv(name):
    p = ROOT/'results'/name
    need('v112' not in str(p).lower(), 'Forbidden path literal')
    for part in reversed((p, *p.parents)):
        need(not stat.S_ISLNK(os.lstat(part).st_mode), 'Symlink path')
    with os.fdopen(os.open(p, os.O_RDONLY | os.O_NOFOLLOW), 'r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def boolean(value):
    need(value in ('True', 'False'), 'Boolean encoding')
    return value == 'True'

def ratio(a, b):
    return a/b if b else None

def compare(row, expected):
    for field, value in expected.items():
        need(field in row, 'Missing table field '+field)
        if value is None:
            need(row[field] in ('', 'None', 'null'), 'Undefined ratio '+field)
        elif isinstance(value, int):
            need(int(row[field]) == value, 'Count mismatch '+field)
        else:
            observed = float(row[field])
            need(math.isfinite(observed) and math.isclose(observed, value, rel_tol=1e-10, abs_tol=1e-12), 'Ratio mismatch '+field)

def recount(rows, model):
    # 저장 producer의 metric 함수는 import하지 않는다.
    cc = Counter((r['human_state'] == D, r[model+'_original_response']) for r in rows)
    pos = sum(r['human_state'] == D for r in rows)
    neg, n = len(rows)-pos, len(rows)
    tp, fn, fp, tn = (cc[True,'VISIBLE'], cc[True,'ABSENT_NONVISIBLE'],
                      cc[False,'VISIBLE'], cc[False,'ABSENT_NONVISIBLE'])
    up, un = cc[True,'ABSTAIN'], cc[False,'ABSTAIN']
    sp, sn = cc[True,'OUT_OF_TASK_GEOMETRY_S'], cc[False,'OUT_OF_TASK_GEOMETRY_S']
    op, on = cc[True,'OTHER_UNRESOLVED'], cc[False,'OTHER_UNRESOLVED']
    need(tp+fn+up+sp+op == pos and fp+tn+un+sn+on == neg, 'Complete confusion partition')
    return dict(N=n, frames=len({r['id'] for r in rows}), visible_support=pos, occluded_support=neg,
        TP=tp, FN_decided=fn, FP=fp, TN=tn, U_visible=up, U_occluded=un,
        out_of_task_S_visible=sp, out_of_task_S_occluded=sn,
        other_unresolved_visible=op, other_unresolved_occluded=on,
        visible_precision=ratio(tp,tp+fp), visible_recall_all_GT=ratio(tp,pos),
        occluded_precision=ratio(tn,tn+fn), occluded_recall_all_GT=ratio(tn,neg),
        abstention_rate=ratio(up+un,n), out_of_task_S_rate=ratio(sp+sn,n),
        unresolved_rate=ratio(up+un+sp+sn+op+on,n),
        decision_coverage=ratio(tp+fn+fp+tn,n), correct_fraction_all_GT=ratio(tp+tn,n),
        selective_accuracy=ratio(tp+tn,tp+fn+fp+tn))

def main():
    ledger = read_csv('ALL1784_LEDGER.csv')
    primary_table = read_csv('PRIMARY.csv')
    cm_table = read_csv('D_VS_EXTERNAL_E_2X3.csv')
    need(len(ledger) == 1784 and len({r['row_id'] for r in ledger}) == 1784, 'All1784 unique original slots')
    for i,r in enumerate(ledger):
        k = int(r['native_corner'])
        need(int(r['global_index']) == i and int(r['frame_index']) == i//8 and k == i%8, 'Original row/frame/native order')
        need(r['row_id'] == r['id']+':native'+str(k), 'Original row_id')
        need(r['group'] in GROUPS[1:], 'Material group')
        primary = boolean(r['paired_same_query_RGB_D_vs_E']) and r['human_state'] in (D,E)
        need(boolean(r['primarypairedbool']) == primary, 'Primary flag/human-state definition')
        available = boolean(r['original_query_valid'])
        state = r['V229_D_E_U']
        need(state in ('D','E','U'), 'Saved D/E/U enum')
        if not available:
            need(state == 'U' and r['U_reason'] == 'ORIGINAL_INPUT_UNAVAILABLE', 'Original unavailable stays U with original reason')
        if state == 'U':
            need(r['logit'] == '' and bool(r['U_reason']), 'U null logit/reason; available unsupported input is allowed')
        else:
            z = float(r['logit'])
            need(available and math.isfinite(z) and state == ('E' if z >= 0 else 'D'), 'Decided available finite raw logit0 sign')
        for model in MODELS:
            response = r[model+'_original_response']
            need(response in RESPONSES, 'Stored response enum')
            column = ('V229' if model == TAG else model)+'_D_E_U'
            need(r[column] == {'VISIBLE':'D','ABSENT_NONVISIBLE':'E'}.get(response,'U'), 'Response/D_E_U consistency')
    need(sum(not boolean(r['original_query_valid']) for r in ledger) == 45, 'Original input45U')
    need(len({r['id'] for r in ledger}) == 223, 'Original223 frames')
    primary = [r for r in ledger if boolean(r['primarypairedbool'])]
    need((len(primary),sum(r['human_state']==D for r in primary),sum(r['human_state']==E for r in primary)) == (1237,1147,90), 'Primary1237 D1147 E90')
    keys = {(g,m) for g in GROUPS for m in MODELS}
    need(len(primary_table) == len(cm_table) == 9, 'Nine material/model rows')
    need({(r['group'],r['model']) for r in primary_table} == keys, 'PRIMARY key coverage')
    need({(r['group'],r['model']) for r in cm_table} == keys, '2x3 key coverage')
    summary = []
    for group,model in sorted(keys):
        rows = [r for r in primary if group == 'ALL' or r['group'] == group]
        result = recount(rows,model)
        saved = next(r for r in primary_table if (r['group'],r['model']) == (group,model))
        compare(saved,result)
        matrix = dict(N=result['N'], D_D=result['TP'], D_E=result['FN_decided'],
            D_U=sum(result[k] for k in ('U_visible','out_of_task_S_visible','other_unresolved_visible')),
            E_D=result['FP'], E_E=result['TN'],
            E_U=sum(result[k] for k in ('U_occluded','out_of_task_S_occluded','other_unresolved_occluded')))
        compare(next(r for r in cm_table if (r['group'],r['model']) == (group,model)),matrix)
        summary.append(dict(group=group,model=model,CM=[[matrix['D_D'],matrix['D_E'],matrix['D_U']],
            [matrix['E_D'],matrix['E_E'],matrix['E_U']]], **result))
    print(json.dumps(dict(completed=True,PASS=True,all_slots=1784,original_input_U=45,
        new_total_U=sum(r['V229_D_E_U']=='U' for r in ledger),
        new_available_unsupported_U=sum(boolean(r['original_query_valid']) and r['V229_D_E_U']=='U' for r in ledger),
        primary_N=1237,primary_D=1147,primary_E=90,tables=summary,
        scope='Saved ledger/table arithmetic only; no GT authority, model, feature or inference replay',
        model_forward=0,fit=0,threshold_changes=0), ensure_ascii=False, allow_nan=False))

if __name__ == '__main__':
    main()
