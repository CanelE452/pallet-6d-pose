"""Independent accounting from saved SCOREs/V9/V10 only; no model/GT imports."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import statistics

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'scripts/research').is_dir())
B = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent/'v2'
CAMPAIGN = ROOT/'_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/remainder_after_test49_v1/repair_campaign_20261015_v1'
V9 = CAMPAIGN/'PROGRESS_V9_V26.json'
V10 = CAMPAIGN/'PROGRESS_V10_V29.json'
CURRENT = B/'geometry_guided_edge_eval_v4_29'
BASELINE = B/'bounded_geometric_priors_eval_v4_22'
V24 = B/'source_preserving_tangent_mask_eval_v4_24/REMAINDER_SCORE.json'
V26 = B/'prepartial_RGB_preservation_eval_v4_26/REMAINDER_SCORE.json'
E = 'SAME_FROZEN_E'
NEW = ('GEOMETRY_EDGE_ALL_BI_NATIVE', 'GEOMETRY_EDGE_ALL_BI_CONTAINMENT',
    'GEOMETRY_EDGE_SOURCE_NATIVE', 'GEOMETRY_EDGE_SOURCE_CONTAINMENT')
COUNTS = {'CAL':46, 'TEST':49, 'REMAINDER':143, 'PRIMARY192':192}
CHECKS = 0
FAILURES = []
MAXDIFF = 0.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    if path.exists():
        raise FileExistsError('Preserve previous audit artifacts: '+str(path))
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def check(path, condition, actual=None, expected=None):
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append({'path':path, 'actual':actual, 'expected':expected})


def compare(actual, expected, path):
    global MAXDIFF
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        check(path, actual == expected and type(actual) is type(expected), actual, expected)
    elif isinstance(expected, (int, float)):
        delta = abs(float(actual)-float(expected))
        MAXDIFF = max(MAXDIFF, delta)
        check(path, math.isfinite(delta) and math.isclose(actual, expected,
            rel_tol=2e-12, abs_tol=1e-10), actual, expected)
    elif isinstance(expected, dict):
        check(path+'/keys', set(actual) == set(expected), sorted(actual), sorted(expected))
        for key in expected.keys() & actual.keys():
            compare(actual[key], expected[key], path+'/'+key)
    elif isinstance(expected, list):
        check(path+'/length', len(actual) == len(expected), len(actual), len(expected))
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, path+'/'+str(i))
    else:
        raise TypeError(type(expected))


def stats(values):
    values = [float(x) for x in values if x is not None]
    ordered = sorted(values)
    if not values:
        return {'n':0, 'mean':None, 'median':None, 'P90':None, 'std':None, 'max':None}
    n = len(values)
    mean = math.fsum(values)/n
    x = (n-1)*.9
    i = math.floor(x)
    p90 = ordered[i] + (x-i)*(ordered[min(i+1,n-1)]-ordered[i])
    return {'n':n, 'mean':mean, 'median':statistics.median(values), 'P90':p90,
        'std':math.sqrt(math.fsum((v-mean)**2 for v in values)/(n-1)) if n>1 else 0., 'max':max(values)}


def summary(rows, arm):
    t = [r['metrics'][arm]['translation_cm'] for r in rows]
    r = [x['metrics'][arm]['rotation_deg'] for x in rows]
    return {'count':len(rows), 'translation_cm':stats(t), 'rotation_deg':stats(r),
        'translation_le5':sum(x <= 5. for x in t),
        'successes_5cm_5deg':sum(x <= 5. and y is not None and y <= 5. for x,y in zip(t,r)),
        'rotation_unresolved':sum(x is None for x in r),
        'fallback_count':sum(not x['accepted'][arm] for x in rows),
        'max_translation_id':rows[max(range(len(t)),key=t.__getitem__)]['id']}


def paired(rows, before, after):
    tb = [r['metrics'][before]['translation_cm'] for r in rows]
    ta = [r['metrics'][after]['translation_cm'] for r in rows]
    delta = [a-b for a,b in zip(ta,tb)]
    sb = [r['metrics'][before]['success_5cm_5deg'] for r in rows]
    sa = [r['metrics'][after]['success_5cm_5deg'] for r in rows]
    ids = [r['id'] for r in rows]
    return {'count':len(rows), 'mean_translation_delta_cm':math.fsum(delta)/len(delta),
        'translation_decreased':sum(d < -1e-8 for d in delta),
        'translation_increased':sum(d > 1e-8 for d in delta),
        'numerically_equal':sum(abs(d) <= 1e-8 for d in delta),
        'reporting_equality_tolerance_cm':1e-8,
        'translation_le5_gains':[k for k,a,b in zip(ids,ta,tb) if a<=5. and b>5.],
        'translation_le5_losses':[k for k,a,b in zip(ids,ta,tb) if a>5. and b<=5.],
        'joint_gains':[k for k,a,b in zip(ids,sa,sb) if a and not b],
        'joint_losses':[k for k,a,b in zip(ids,sa,sb) if not a and b],
        'fallback_count_before':sum(not r['accepted'][before] for r in rows),
        'fallback_count_after':sum(not r['accepted'][after] for r in rows)}


def raw_method_objects(path):
    """Retain the actual UTF-8 JSON object spans, not just reserialization."""
    text = path.read_text()
    start = text.index('"methods":') + len('"methods":')
    pos = text.index('[',start)+1
    decoder = json.JSONDecoder()
    result = []
    while True:
        while text[pos].isspace() or text[pos] == ',':
            pos += 1
        if text[pos] == ']':
            return result
        begin = pos
        _, pos = decoder.raw_decode(text,pos)
        result.append(text[begin:pos].encode('utf-8'))


def main():
    OUT.mkdir(exist_ok=False)
    inputs = [V9,V10,V24,V26]
    inputs += [folder/(part+'_SCORE.json') for folder in (CURRENT,BASELINE)
        for part in ('CAL','TEST','REMAINDER')]
    bindings = {str(p.relative_to(ROOT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in inputs}
    if sha(V10) != 'a60071ff4b95f9d172e222902b679ea13f9bf7bd9ce66b83443a635e2a50927f':
        raise ValueError('Exact root-declared immutable V10 required')
    write_new(OUT/'INPUT_BINDINGS_V2.json', {'schema':'READ_ONLY_SAVED_ACCOUNTING_INPUTS_V10',
        'UTC':datetime.now(timezone.utc).isoformat(), 'files':bindings,
        'allowed_data':'Only saved SCORE JSONs plus previousV9/newV10. No reference files opened.'})
    # Independent sample-standard deviation (N-1) and linear quantile arithmetic.
    toy = stats([1.,2.,3.,4.])
    check('selfcheck/statistics', toy['mean']==2.5 and toy['median']==2.5
        and abs(toy['P90']-3.7)<1e-14 and abs(toy['std']-math.sqrt(5./3.))<1e-14)
    old, new = read(V9), read(V10)
    check('prior29/semantic',new['methods'][:29]==old['methods'])
    oldraw, newraw = raw_method_objects(V9), raw_method_objects(V10)
    check('prior29/raw_object_bytes',oldraw==newraw[:29])
    check('prior29/population_rows',sum(len(m['populations'])for m in old['methods'])==116)
    check('methods/order_and_cardinality',len(new['methods'])==33
        and len({m['method']for m in new['methods']})==33
        and tuple(m['method']for m in new['methods'][29:])==NEW)
    for key,value in {'method_count':33,'previous_method_count':29,'new_method_count':4,
            'population_table_rows':132,'paired_control_count':8,'separate_CAL':46,
            'goal_status':'active','goal_completion_declared':False,'default_changed':False,
            'default_method':E,'original_V9_methods_retained_exactly':True,
            'exact_baseline_consistency_verified':True,'all_methods_uniform_full_population':True,
            'no_primary_native_compatibility_or_fallback_filter':True,
            'no_new_reference_numeric_binding_changes':True,'no_per_frame_best_method_mixture':True,
            'no_new_images_labels_inference_or_optimization_in_accounting':True,
            'S_image_2D_or_regions':0,'U_regions_only':True,'new_precise_2D_corners':0,
            'actual_other_instance_occluder_masks_in_current_trial':0,
            'old_global_symmetric_mask_or_tie_used_in_new_methods':False}.items():
        compare(new[key],value,'metadata/'+key)
    compare(new['primary'],old['primary'],'primary/unchanged')
    compare(new['primary'],{'TEST49':49,'REMAINDER143':143,'total':192,'severe':0},'primary/exact')
    check('protected/preserved',new['protected_user_sha256']==old['protected_user_sha256'])
    check('default/unique', [m['method']for m in new['methods']if m['current_default']]==[E])
    evidence={x['path']:x for x in new['evidence']}
    for path in (V9,*[CURRENT/(p+'_SCORE.json')for p in ('CAL','TEST','REMAINDER')]):
        b=evidence[str(path.relative_to(ROOT))]
        check('evidence/'+path.name,b['sha256']==sha(path)and b['bytes']==path.stat().st_size)
    scores={p:read(CURRENT/(p+'_SCORE.json'))for p in ('CAL','TEST','REMAINDER')}
    older={p:read(BASELINE/(p+'_SCORE.json'))for p in scores}
    populations={p:d['rows']for p,d in scores.items()}
    populations['PRIMARY192']=populations['TEST']+populations['REMAINDER']
    reference_bindings={}
    all_ids=[]
    for part,data in scores.items():
        rows=data['rows']; ids=[r['id']for r in rows]
        check(part+'/rows_and_unique',len(rows)==data['count']==COUNTS[part]and len(set(ids))==len(ids))
        check(part+'/original_order',ids==[r['id']for r in older[part]['rows']])
        compare(data['reference_binding'],older[part]['reference_binding'],part+'/original_reference_binding')
        reference_bindings[part]=data['reference_binding']
        check(part+'/no_filter',data['all_rows_kept']is True and data['new_parameter_or_source_selection_by_metrics']is False)
        all_ids.extend(ids)
        for i,row in enumerate(rows):
            check(part+'/'+str(i)+'/all5arms',set(row['metrics'])==set((E,*NEW))
                and set(row['accepted'])==set((E,*NEW))and set(row['fixed_family'])==set((E,*NEW)))
            baseline=older[part]['rows'][i]
            for field in ('metrics','accepted','fixed_family'):
                check(part+'/'+str(i)+'/exact_e_'+field,row[field][E]==baseline[field][E])
            check(part+'/'+str(i)+'/S_U_zero',row['tangent_control_audit']['S_U_point_costs']==0)
            for arm,m in row['metrics'].items():
                t=m['translation_cm']; r=m['rotation_deg']
                check(part+'/'+str(i)+'/'+arm+'/finiteT',isinstance(t,(int,float))and math.isfinite(t)and t>=0)
                check(part+'/'+str(i)+'/'+arm+'/finiteR',r is None or (isinstance(r,(int,float))and math.isfinite(r)and r>=0))
                check(part+'/'+str(i)+'/'+arm+'/accepted_bool',type(row['accepted'][arm])is bool)
                check(part+'/'+str(i)+'/'+arm+'/joint_rule',m['success_5cm_5deg']==(t<=5. and r is not None and r<=5.))
    check('all238/no_overlap',len(all_ids)==len(set(all_ids))==238)
    recomputed={}
    for arm in (E,*NEW):
        method=next(m for m in new['methods']if m['method']==arm)
        recomputed[arm]={part:summary(rows,arm)for part,rows in populations.items()}
        for part,saved in recomputed[arm].items():
            compare(method['populations'][part],saved,arm+'/'+part+'/campaign')
            if part!='PRIMARY192':
                compare(scores[part]['summary'][arm],{k:v for k,v in saved.items()if k!='count'},arm+'/'+part+'/saved_summary')
        check(arm+'/TEST49_flag',method['original_TEST49_all_translation_le5']==(recomputed[arm]['TEST']['translation_le5']==49))
        check(arm+'/PRIMARY_target_flag',method['numerical_primary192_target_met']==(recomputed[arm]['PRIMARY192']['translation_le5']==192))
        if arm in NEW:
            check(arm+'/not_default',method['current_default']is False)
            compare(new['original49_preserved_by_new_method'][arm],recomputed[arm]['TEST']['translation_le5']==49,arm+'/preservation')
    expected_pairs=[(E,a)for a in NEW]+[(NEW[0],NEW[1]),(NEW[2],NEW[3]),(NEW[0],NEW[2]),(NEW[1],NEW[3])]
    check('pairs/exact8',[(p['before'],p['after'])for p in new['paired_controls']]==expected_pairs)
    for i,pair in enumerate(new['paired_controls']):
        for part,rows in populations.items():
            compare(pair['populations'][part],paired(rows,pair['before'],pair['after']),f'pairs/{i}/{part}')
    check('no_adoption/target',new['numerical_target_met_any_new_method']==any(
        recomputed[a]['PRIMARY192']['translation_le5']==192 for a in NEW))
    # Optional source-origin question: old176 is a different V24 arm, not E.
    ident='wood_night_01:031426'; origin={}
    for version,path in [('V24',V24),('V26',V26),('V29',CURRENT/'REMAINDER_SCORE.json')]:
        data=read(path);row=next(r for r in data['rows']if r['id']==ident)
        origin[version]={'score_path':str(path.relative_to(ROOT)),'score_sha256':sha(path),
            'reference_binding':data['reference_binding'],'saved_row':row}
    check('031426/reference_same',origin['V24']['reference_binding']==origin['V26']['reference_binding']==origin['V29']['reference_binding'])
    for field in ('metrics','accepted','fixed_family'):
        check('031426/E_'+field,origin['V24']['saved_row'][field][E]==origin['V26']['saved_row'][field][E]==origin['V29']['saved_row'][field][E])
    for arm in ('SOURCE_PRESERVING_TCAL_MASK_FIT','SOURCE_PRESERVING_TCAL_MASK_FIT_E_FAMILY'):
        check('031426/176_origin/'+arm,origin['V24']['saved_row']['metrics'][arm]['translation_cm']==176.53513064809076)
    write_new(OUT/'WOOD031426_PREVIOUS176_PROVENANCE_V2.json',{'id':ident,'sources':origin,
        'conclusion':'176.53513064809076cm/82.30686114698874deg is V24 SOURCE_PRESERVING_TCAL_MASK_FIT or its E_FAMILY control; SAME_FROZEN_E is208.2242047986846cm/83.75120342488951deg in V24,V26,V29 with identical reference binding.',
        'new_reference_or_image_or_saved_pose_or_inference_read':False})
    for path in inputs:
        check('posthash/'+path.name,sha(path)==bindings[str(path.relative_to(ROOT))]['sha256'])
    receipt={'schema':'INDEPENDENT_SAVED_SCORE_CAMPAIGN_ACCOUNTING_V10_V29_REVIEW','UTC':datetime.now(timezone.utc).isoformat(),
        'passed':not FAILURES,'checks':CHECKS,'failure_count':len(FAILURES),'failures':FAILURES,
        'maximum_numeric_difference':MAXDIFF,'source_sha256':sha(Path(__file__)),
        'correction_to_preserved_V1':{'receipt_path':str((OUT.parent/'ACCOUNTING_REVIEW_V1.json').relative_to(ROOT)),'receipt_sha256':sha(OUT.parent/'ACCOUNTING_REVIEW_V1.json'),'reason':'V1 independent audit assumed population std N. Saved SCORE and campaign use sample std N-1; all V1 failures were std cells. No source SCORE or campaign changes.'},
        'input_bindings':bindings,'prior29_methods116rows_raw_object_bytes_and_semantics_preserved':oldraw==newraw[:29]and old['methods']==new['methods'][:29],
        'method_count':33,'population_rows':132,'paired_controls':8,'reference_bindings_unchanged':reference_bindings,
        'independently_recomputed':recomputed,
        'V29_status':'Completed saved 238x5 scoring is the only new-method result audited.',
        'V30_status':'Separate live retry was reported3/192 at assignment; not read or represented as completed by this audit.',
        'no_default_adoption':True,'goal_status':new['goal_status'],
        'scope':['Only10 saved SCORE/V9/V10 JSON files read; root builder, models, full observations/predictions, images and reference annotations were not opened.',
            'Statistics independently use Python fsum/median, explicit linear90th quantile and sample standard deviation (N-1); no builder/scorer numerical helpers imported.',
            'All238 row IDs/order, baseline full metric dictionaries/accepted/family, four new methods on four populations, every8paired gain/loss list/fallback and original reference bindings checked.',
            'Previous29 methods are preserved in actual UTF8 JSON object spans and semantics; source input bytes rechecked after audit.',
            'This verifies saved-score arithmetic and preservation, not raw pose/GT correctness, statistical independence or method adoption.']}
    write_new(OUT/'ACCOUNTING_REVIEW_V2.json',receipt)
    artifacts={p.name:{'sha256':sha(p),'bytes':p.stat().st_size}for p in OUT.iterdir()if p.is_file()and p.name!='ACCOUNTING_REVIEW_SEAL_V2.json'}
    write_new(OUT/'ACCOUNTING_REVIEW_SEAL_V2.json',{'receipt_sha256':sha(OUT/'ACCOUNTING_REVIEW_V2.json'),
        'artifacts':artifacts,'input_bindings':bindings,'passed':not FAILURES})
    print(json.dumps({'passed':not FAILURES,'checks':CHECKS,'failures':FAILURES,
        'max_numeric_difference':MAXDIFF,'receipt_sha256':sha(OUT/'ACCOUNTING_REVIEW_V2.json')},ensure_ascii=False))
    for arm in NEW:
        print(arm,{p:{k:recomputed[arm][p][k]for k in ('translation_le5','successes_5cm_5deg','fallback_count','max_translation_id')}for p in ('TEST','REMAINDER','PRIMARY192')})
    if FAILURES:
        raise SystemExit(1)


if __name__=='__main__':
    main()
