"""Independent V11 saved-score accounting only; no experiment imports."""
from pathlib import Path
from datetime import datetime, timezone
from html.parser import HTMLParser
import hashlib, json, math, statistics
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'scripts/research').is_dir())
B=Path(__file__).resolve().parent.parent
OUT=Path(__file__).resolve().parent/'v1'
CAMPAIGN=ROOT/'_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/remainder_after_test49_v1/repair_campaign_20261015_v1'
PREVIOUS=CAMPAIGN/'PROGRESS_V10_V29.json'
CURRENT=CAMPAIGN/'PROGRESS_V11_V30.json'
HTML=CAMPAIGN/'RESULTS_V11_V30.html'
CAMPAIGN_SEAL=CAMPAIGN/'PROGRESS_V11_V30_SEAL.json'
OLD_RUN=B/'geometry_guided_edge_eval_v4_29'
NEW_RUN=B/'geometry_guided_edge_eval_v4_30'
E='SAME_FROZEN_E'
OLD_ARMS=('GEOMETRY_EDGE_ALL_BI_NATIVE','GEOMETRY_EDGE_ALL_BI_CONTAINMENT','GEOMETRY_EDGE_SOURCE_NATIVE','GEOMETRY_EDGE_SOURCE_CONTAINMENT')
NEW_ARMS=('ASSOCIATED_EDGE_ALL_BI_NATIVE','ASSOCIATED_EDGE_ALL_BI_CONTAINMENT','ASSOCIATED_EDGE_SOURCE_NATIVE','ASSOCIATED_EDGE_SOURCE_CONTAINMENT')
COUNTS={'CAL':46,'TEST':49,'REMAINDER':143,'PRIMARY192':192}
CHECKS=0
FAILURES=[]
MAXDIFF=0.
EXPECTED_CURRENT_SHA='38cd50462e3d3f9e3a4f0d982d9775b33668e22ab1b92e7b17659507a47216a3'
EXPECTED_PREVIOUS_SHA='a60071ff4b95f9d172e222902b679ea13f9bf7bd9ce66b83443a635e2a50927f'

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


def paired(before_rows,after_rows,before,after):
    check('paired/ordered_ids',[r['id']for r in before_rows]==[r['id']for r in after_rows])
    tb=[r['metrics'][before]['translation_cm']for r in before_rows]
    ta=[r['metrics'][after]['translation_cm']for r in after_rows]
    delta=[a-b for a,b in zip(ta,tb)]
    sb=[r['metrics'][before]['success_5cm_5deg']for r in before_rows]
    sa=[r['metrics'][after]['success_5cm_5deg']for r in after_rows]
    ids=[r['id']for r in before_rows]
    return dict(count=len(ids),mean_translation_delta_cm=math.fsum(delta)/len(delta),
        translation_decreased=sum(d < -1e-8 for d in delta),translation_increased=sum(d > 1e-8 for d in delta),
        numerically_equal=sum(abs(d)<=1e-8 for d in delta),reporting_equality_tolerance_cm=1e-8,
        translation_le5_gains=[k for k,a,b in zip(ids,ta,tb)if a<=5. and b>5.],
        translation_le5_losses=[k for k,a,b in zip(ids,ta,tb)if a>5. and b<=5.],
        joint_gains=[k for k,a,b in zip(ids,sa,sb)if a and not b],
        joint_losses=[k for k,a,b in zip(ids,sa,sb)if not a and b],
        fallback_count_before=sum(not r['accepted'][before]for r in before_rows),
        fallback_count_after=sum(not r['accepted'][after]for r in after_rows))


class Table(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True);self.rows=[];self.cells=[];self.cell=None
    def handle_starttag(self,tag,attrs):
        if tag=='tr':self.cells=[]
        if tag=='td':self.cell=[]
    def handle_data(self,data):
        if self.cell is not None:self.cell.append(data)
    def handle_endtag(self,tag):
        if tag=='td':self.cells.append(''.join(self.cell));self.cell=None
        if tag=='tr' and self.cells:self.rows.append(self.cells)


def populations(report):
    result={part:report[part]['rows']for part in ('CAL','TEST','REMAINDER')}
    result['PRIMARY192']=result['TEST']+result['REMAINDER']
    return result


def main():
    OUT.mkdir(exist_ok=False)
    files=[PREVIOUS,CURRENT,HTML,CAMPAIGN_SEAL]
    files.extend(folder/(part+'_SCORE.json')for folder in(OLD_RUN,NEW_RUN)for part in('CAL','TEST','REMAINDER'))
    files.extend(folder/'SCORE_SEAL.json'for folder in(OLD_RUN,NEW_RUN))
    files.extend(NEW_RUN/'independent_root_audit_v1'/name for name in('AUDIT.json','AUDIT_SEAL.json'))
    bindings={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size)for p in files}
    assert sha(CURRENT)==EXPECTED_CURRENT_SHA and sha(PREVIOUS)==EXPECTED_PREVIOUS_SHA
    write_new(OUT/'INPUT_BINDINGS_V1.json',dict(schema='READ_ONLY_SAVED_V11_ACCOUNTING_INPUT_BINDINGS',files=bindings,
        source_sha256=sha(Path(__file__)),no_model_or_label_or_full_OBS_or_prediction_IO=True))
    old,new=read(PREVIOUS),read(CURRENT)
    check('selfcheck/sample_std_N_minus_1',abs(stats([1.,2.,3.,4.])['std']-math.sqrt(5./3.))<1e-14)
    check('selfcheck/linear_P90',abs(stats([1.,2.,3.,4.])['P90']-3.7)<1e-14)
    oldraw,newraw=raw_method_objects(PREVIOUS),raw_method_objects(CURRENT)
    check('old33/semantic',new['methods'][:33]==old['methods'])
    check('old33/actual_UTF8_object_bytes',newraw[:33]==oldraw)
    check('old33/population_rows',sum(len(m['populations'])for m in old['methods'])==132)
    check('all37/order_unique',len(new['methods'])==37 and len({m['method']for m in new['methods']})==37
        and tuple(m['method']for m in new['methods'][33:])==NEW_ARMS)
    for key,value in dict(method_count=37,previous_method_count=33,new_method_count=4,population_table_rows=148,
            paired_control_count=8,separate_CAL=46,goal_status='active',goal_completion_declared=False,
            default_changed=False,default_method=E,original_V10_methods_retained_exactly=True,
            exact_baseline_consistency_verified=True,all_methods_uniform_full_population=True,
            no_primary_native_compatibility_or_fallback_filter=True,no_new_reference_numeric_binding_changes=True,
            no_per_frame_best_method_mixture=True,no_new_images_labels_inference_or_optimization_in_accounting=True,
            S_image_2D_or_regions=0,U_regions_only=True,new_precise_2D_corners=0,
            actual_other_instance_occluder_masks_in_current_trial=0,targetmask_absence_is_unknown=True,
            old_global_symmetric_mask_or_tie_used_in_new_methods=False,
            V30_association_gate_and_population_CAL_noise_are_bundled_changes=True,
            no_e_numeric_pose_prior_or_fallback=True).items():
        compare(new[key],value,'metadata/'+key)
    compare(new['primary'],old['primary'],'primary/unchanged')
    compare(new['primary'],dict(TEST49=49,REMAINDER143=143,total=192,severe=0),'primary/exact')
    check('protected/preserved',new['protected_user_sha256']==old['protected_user_sha256'])
    check('default/unique',[m['method']for m in new['methods']if m['current_default']]==[E])
    seal=read(CAMPAIGN_SEAL)
    for key,path in(('json',CURRENT),('html',HTML)):
        check('campaign_seal/'+key,seal[key]['sha256']==sha(path)and seal[key]['bytes']==path.stat().st_size)
    compare({k:seal[k]for k in('method_count','population_table_rows','paired_control_count')},
        dict(method_count=37,population_table_rows=148,paired_control_count=8),'seal/counters')
    evidence={item['path']:item for item in new['evidence']}
    for path in(PREVIOUS,*[NEW_RUN/(part+'_SCORE.json')for part in('CAL','TEST','REMAINDER')]):
        item=evidence[str(path.relative_to(ROOT))]
        check('evidence/'+str(path.relative_to(ROOT)),item['sha256']==sha(path)and item['bytes']==path.stat().st_size)
    oldscores={part:read(OLD_RUN/(part+'_SCORE.json'))for part in('CAL','TEST','REMAINDER')}
    scores={part:read(NEW_RUN/(part+'_SCORE.json'))for part in('CAL','TEST','REMAINDER')}
    scoreseals={name:read(folder/'SCORE_SEAL.json')for name,folder in(('V29',OLD_RUN),('V30',NEW_RUN))}
    allids=[];reference_bindings={}
    for part,data in scores.items():
        rows=data['rows'];ids=[row['id']for row in rows];oldrows=oldscores[part]['rows']
        check(part+'/all_rows_unique',len(rows)==data['count']==COUNTS[part]and len(set(ids))==COUNTS[part])
        check(part+'/original_order',ids==[r['id']for r in oldrows])
        check(part+'/no_filter',data['all_rows_kept']is True and data['new_parameter_or_source_selection_by_metrics']is False)
        compare(data['reference_binding'],oldscores[part]['reference_binding'],part+'/reference_binding')
        reference_bindings[part]=data['reference_binding'];allids.extend(ids)
        for run,folder in(('V29',OLD_RUN),('V30',NEW_RUN)):
            check(run+'/'+part+'/score_seal',scoreseals[run]['score_sha256'][part]==sha(folder/(part+'_SCORE.json')))
        for i,(row,prev)in enumerate(zip(rows,oldrows)):
            check(part+'/'+str(i)+'/all5arms',all(set(row[key])==set((E,*NEW_ARMS))for key in('metrics','accepted','fixed_family')))
            for key in('metrics','accepted','fixed_family'):
                check(part+'/'+str(i)+'/exact_e/'+key,row[key][E]==prev[key][E])
            check(part+'/'+str(i)+'/S_U_zero',row['tangent_control_audit']['S_U_point_costs']==0)
            for arm,value in row['metrics'].items():
                t,r=value['translation_cm'],value['rotation_deg']
                check(part+'/'+str(i)+'/'+arm+'/finiteT',isinstance(t,(float,int))and math.isfinite(t)and t>=0)
                check(part+'/'+str(i)+'/'+arm+'/finiteR',r is None or(isinstance(r,(float,int))and math.isfinite(r)and r>=0))
                check(part+'/'+str(i)+'/'+arm+'/joint',value['success_5cm_5deg']==(t<=5. and r is not None and r<=5.))
                check(part+'/'+str(i)+'/'+arm+'/accepted_bool',type(row['accepted'][arm])is bool)
    check('all238/no_duplicate_or_dropout',len(allids)==len(set(allids))==238)
    allpops=populations(scores);oldpops=populations(oldscores)
    recomputed={}
    for arm in(E,*NEW_ARMS):
        method=next(m for m in new['methods']if m['method']==arm)
        recomputed[arm]={part:summary(rows,arm)for part,rows in allpops.items()}
        for part,result in recomputed[arm].items():
            compare(method['populations'][part],result,arm+'/'+part+'/campaign')
            if part!='PRIMARY192':compare(scores[part]['summary'][arm],{k:v for k,v in result.items()if k!='count'},arm+'/'+part+'/score_summary')
        check(arm+'/49preserveflag',method['original_TEST49_all_translation_le5']==(recomputed[arm]['TEST']['translation_le5']==49))
        check(arm+'/192targetflag',method['numerical_primary192_target_met']==(recomputed[arm]['PRIMARY192']['translation_le5']==192))
        if arm in NEW_ARMS:
            check(arm+'/not_default',method['current_default']is False)
            check(arm+'/preservation',new['original49_preserved_by_new_method'][arm]==method['original_TEST49_all_translation_le5'])
            index=NEW_ARMS.index(arm);contract=method['intervention_contract']
            compare(contract['source_policy'],'ALL_BI_REPLACE'if index<2 else'RGB_NONFALLBACK_PRESERVE',arm+'/policy')
            compare(contract['containment_weight'],float(index%2),arm+'/Cweight')
            compare(contract['full_pose_TP_topnormal_prior_weight'],0.,arm+'/prior0')
            compare(contract['noise_observation_family'],'OBSERVED_BI_ASSOCIATED_MULTIGUIDE_ACTUAL_NATIVE_EDGES_V30',arm+'/family')
            compare(contract['original_named_BI_to_actual_infinite_line_normal_reach_px'],43.,arm+'/normalgate')
            compare(contract['old_V29_noise_relabelled'],False,arm+'/oldnoise')
            compare(contract['old_symmetric_mask_cost_or_numeric_tie_break_used'],False,arm+'/no_oldmasktie')
    definitions=[('V29','V30',a,b)for a,b in zip(OLD_ARMS,NEW_ARMS)]
    definitions += [('V30','V30',NEW_ARMS[a],NEW_ARMS[b])for a,b in((0,1),(2,3),(0,2),(1,3))]
    check('pairs/exact8',[(p['before_run'],p['after_run'],p['before'],p['after'])for p in new['paired_controls']]==definitions)
    runpops={'V29':oldpops,'V30':allpops}
    for i,pair in enumerate(new['paired_controls']):
        for part in COUNTS:
            result=paired(runpops[pair['before_run']][part],runpops[pair['after_run']][part],pair['before'],pair['after'])
            compare(pair['populations'][part],result,'paired/'+str(i)+'/'+part)
    check('target/anynew',new['numerical_target_met_any_new_method']==any(recomputed[a]['PRIMARY192']['translation_le5']==192 for a in NEW_ARMS))
    parser=Table();parser.feed(HTML.read_text())
    check('HTML/148rows',len(parser.rows)==148)
    for row,(part,method)in zip(parser.rows,[(part,m)for part in('PRIMARY192','TEST','REMAINDER','CAL')for m in new['methods']]):
        q=method['populations'][part]
        values=[q['count'],q['translation_le5'],q['successes_5cm_5deg'],q['translation_cm']['mean'],q['translation_cm']['max'],
            q['max_translation_id'],q['rotation_deg']['mean'],q['rotation_deg']['max'],q['rotation_unresolved'],q['fallback_count'],
            '예'if method['original_TEST49_all_translation_le5']else'아니오']
        expected=[method['label_KO']]+['미확정'if v is None else f'{v:.3f}'if isinstance(v,float)else str(v)for v in values]
        compare(row,expected,'HTML/'+part+'/'+method['method'])
    audit=read(NEW_RUN/'independent_root_audit_v1/AUDIT.json')
    auditseal=read(NEW_RUN/'independent_root_audit_v1/AUDIT_SEAL.json')
    check('rootaudit/no_failure',audit['failure_count']==0)
    check('rootaudit/seal',auditseal['receipt_sha256']==sha(NEW_RUN/'independent_root_audit_v1/AUDIT.json'))
    for path in files:check('posthash/'+str(path.relative_to(ROOT)),sha(path)==bindings[str(path.relative_to(ROOT))]['sha256'])
    receipt=dict(schema='INDEPENDENT_SAVED_SCORE_CAMPAIGN_ACCOUNTING_V11_V30_REVIEW_V1',UTC=datetime.now(timezone.utc).isoformat(),
        passed=not FAILURES,checks=CHECKS,failure_count=len(FAILURES),failures=FAILURES,maximum_numeric_difference=MAXDIFF,
        source_sha256=sha(Path(__file__)),input_bindings=bindings,
        previous33_methods132rows_semantically_and_raw_UTF8_object_bytes_preserved=old['methods']==new['methods'][:33]and oldraw==newraw[:33],
        method_count=37,population_rows=148,paired_controls=8,HTML_table_rows=len(parser.rows),
        reference_bindings_unchanged=reference_bindings,independently_recomputed=recomputed,
        no_default_adoption=True,goal_status=new['goal_status'],all192_target_unmet=not new['numerical_target_met_any_new_method'],
        scope=['Only small saved SCORE, previous/new campaign JSON, HTML and metadata seals/audit are decoded; no experiment numerical helper is imported.',
            'Independent math.fsum/median/linear90th percentile/sample standard deviationN-1; all threshold/joint/fallback/max-ID and eight paired lists recomputed.',
            'Every238 ID/order and full e metrics/accepted/fixed-family alias preserved; previous33 actual JSON object spans preserved byte-for-byte.',
            'No raw model/observation/prediction journal/image/reference annotation/optimizer/posefit read or execution; old sources/results are unchanged.',
            'This verifies saved-score arithmetic and preservation, not pose truth, finite-edge identity, independence, uniqueness or method adoption.'])
    write_new(OUT/'ACCOUNTING_REVIEW_V1.json',receipt)
    artifacts={p.name:dict(sha256=sha(p),bytes=p.stat().st_size)for p in OUT.iterdir()if p.is_file()}
    write_new(OUT/'ACCOUNTING_REVIEW_SEAL_V1.json',dict(receipt_sha256=sha(OUT/'ACCOUNTING_REVIEW_V1.json'),artifacts=artifacts,input_bindings=bindings,passed=not FAILURES))
    print(json.dumps(dict(passed=not FAILURES,checks=CHECKS,failure_count=len(FAILURES),failures=FAILURES,max_numeric_difference=MAXDIFF,
        receipt_sha256=sha(OUT/'ACCOUNTING_REVIEW_V1.json')),ensure_ascii=False))
    for arm in NEW_ARMS:print(arm,{part:{key:recomputed[arm][part][key]for key in('translation_le5','successes_5cm_5deg','fallback_count','max_translation_id')}for part in('TEST','REMAINDER','PRIMARY192')})
    if FAILURES:raise SystemExit(1)


if __name__=='__main__':
    main()
