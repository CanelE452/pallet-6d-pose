"""Independent arithmetic replay of frozen current rows; no model, fitting or RGB."""
from pathlib import Path
import csv
import hashlib
import json
import math

R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
A = R / 'V263_CURRENT_EVALUATION_A/actual_evaluation_v1'

def safe(p):
    if '..' in p.parts or any(t in str(p).lower() for t in ['v112','cache','.pyc']):
        raise ValueError(p)
    for a in reversed([p, *p.parents]):
        if a.is_symlink():
            raise ValueError(a)
    return p

def csv_rows(name):
    with safe(A / name).open() as f:
        return list(csv.DictReader(f))

def yes(v):
    return str(v).lower() in ('true','1')

def pred(v):
    return {'0':'D','1':'E','-1':'U'}.get(v, v)

def main():
    rows = csv_rows('ALL1784_CLASSIFICATION_LOCATION_JOIN.csv')
    reported = csv_rows('CLASSIFICATION.csv')
    assert len(rows) == 1784 and len({r['row_id'] for r in rows}) == 1784
    checks = []
    for group in ['ALL','WOOD_PRIMARY105','SQUARE118']:
        primary = [r for r in rows if yes(r['primarypairedbool']) and (group == 'ALL' or r['group'] == group)]
        for model, column in [('v61','v61_D_E_U'),('CAL_LEGACY','legacy_D_E_U'),
                              ('VIS_ONLY_SELECTED','VIS_ONLY_SELECTED__semantic_VIS_prediction'),
                              ('JOINT_SELECTED','JOINT_SELECTED__semantic_VIS_prediction')]:
            cm = [[0,0,0],[0,0,0]]
            for r in primary:
                truth = {'DIRECT_VISIBLE':0, 'EXTERNAL_OCCLUDED':1}[r['human_state']]
                cm[truth][['D','E','U'].index(pred(r[column]))] += 1
            ref = next(r for r in reported if r['group'] == group and r['cohort'] == 'PRIMARY' and r['model'] == model)
            expected = [[int(ref[k]) for k in ['D_D','D_E','D_U']], [int(ref[k]) for k in ['E_D','E_E','E_U']]]
            assert cm == expected, (group, model, cm, expected)
            checks.append({'group':group,'model':model,'N':len(primary),'CM':cm})
    locations = []
    for group in ['ALL','WOOD_PRIMARY105','SQUARE118']:
        eligible = []
        for r in rows:
            if group != 'ALL' and r['group'] != group:
                continue
            if r['human_state'] != 'DIRECT_VISIBLE' or r['annotation_coordinate_source'] != 'manual_click' or not yes(r['original_query_valid']):
                continue
            values = [float(r[k]) for k in ['query_x','query_y','human_reference_x','human_reference_y']]
            if not all(math.isfinite(v) for v in values):
                continue
            eligible.append((r,values))
        for model in ['LOC_ONLY_INITIAL_GEOMETRY','JOINT_SELECTED']:
            original_correct = final_correct = recovered = damaged = accepted = accepted_correct = 0
            for r,(qx,qy,rx,ry) in eligible:
                p = json.loads(r[model+'__final_xy'])
                initial = math.hypot(qx-rx,qy-ry)
                final = math.hypot(p[0]-rx,p[1]-ry)
                old_ok,new_ok = initial <= 3, final <= 3
                original_correct += old_ok; final_correct += new_ok
                recovered += not old_ok and new_ok; damaged += old_ok and not new_ok
                if yes(r[model+'__accepted']):
                    accepted += 1; accepted_correct += new_ok
                else:
                    assert p == [qx,qy], 'Rejected location changed original q'
            ref = next(r for r in csv_rows('LOCATION.csv') if r['group'] == group and r['cohort'] == 'MANUAL_CLICK_D_ALL' and r['model'] == model)
            assert len(eligible) == int(ref['scorable_N'])
            assert (recovered,damaged,accepted) == (int(ref['recovery_at3px']),int(ref['damage_at3px']),int(ref['accepted_N']))
            assert original_correct + recovered - damaged == final_correct
            locations.append({'group':group,'model':model,'N':len(eligible),
                              'original_LE3':original_correct,'final_LE3':final_correct,
                              'recovery':recovered,'damage':damaged,'accepted':accepted,
                              'accepted_correct_LE3':accepted_correct})
    result = {'PASS':True,'classification_checks':checks,'location_checks':locations,
              'fit_or_model_or_RGB_access':0, 'independent_from_metric_implementation':True,
              'input_join_sha256':hashlib.sha256(safe(A/'ALL1784_CLASSIFICATION_LOCATION_JOIN.csv').read_bytes()).hexdigest()}
    out = R/'V263_ROOT_SPATIAL_LOCALIZATION/ROOT_SAVED_METRIC_REPLAY.json'
    safe(out).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'PASS':True,'CM_checks':len(checks),'location_checks':len(locations),'output':str(out)}))

if __name__ == '__main__':
    main()
