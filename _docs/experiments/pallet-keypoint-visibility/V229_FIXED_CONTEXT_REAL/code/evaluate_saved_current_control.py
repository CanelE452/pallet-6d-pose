"""V229 저장 판정만 평가한다. 실제 root entry와 추론 동결 전 GT/점수 접근은 없다.

기존 human visibility/primary/좌표 진단 계약을 그대로 사용한다. 모델 로드,
forward, 문턱 선택, GT 좌표/instance/native 순서 보정은 수행하지 않는다.
"""
import argparse, ast, csv, hashlib, io, json, math, os, stat
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

H = Path(__file__).absolute().parent
TAG = 'V229_CONTROL_FIXED_Q'
MODELS = ('v61', 'legacy', TAG)
D, E, S = 'DIRECT_VISIBLE', 'EXTERNAL_OCCLUDED', 'SELF_OCCLUDED'
GROUPS = ('WOOD_PRIMARY105', 'SQUARE118')
TASKS = {'D_vs_E': (D, E), 'D_vs_S': (D, S), 'D_vs_ES': (D, E, S)}

def require(ok, message):
    if not ok: raise ValueError(message)

def safe_path(value):
    text = str(value)
    require('v112' not in text.lower(), 'Forbidden V112 literal before filesystem')
    p = Path(text)
    require(p.is_absolute() and '..' not in p.parts, 'Absolute closed path required')
    for x in reversed((p, *p.parents)):
        require(not stat.S_ISLNK(os.lstat(x).st_mode), 'Symlink forbidden')
    return p

def raw(binding):
    p = safe_path(binding['path'])
    require(p.suffix.lower() in ('.json', '.jsonl', '.csv', '.py'), 'Text-only evaluator')
    with os.fdopen(os.open(p, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as f: data = f.read()
    require(hashlib.sha256(data).hexdigest() == binding['sha256'], 'Bound SHA mismatch: '+str(p))
    if 'size_bytes' in binding: require(len(data) == binding['size_bytes'], 'Bound size mismatch')
    return data

def document(binding): return json.loads(raw(binding))
def csv_rows(binding): return list(csv.DictReader(io.StringIO(raw(binding).decode('utf-8-sig'))))
def json_lines(binding): return [json.loads(x) for x in raw(binding).splitlines() if x.strip()]
def bound(path):
    data = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))

def save_json(path, value):
    with path.open('x', encoding='utf-8') as f: json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')

def save_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(rows)

def load_functions(binding, names, namespace):
    # 운영 모듈 import 없이 hash 고정된 순수 함수만 사용한다.
    tree = ast.parse(raw(binding).decode('utf-8'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes} == set(names), 'Frozen metric function closure')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), binding['path'], 'exec'), namespace)

def main(entry_path):
    entry = document(dict(path=str(entry_path), sha256=hashlib.sha256(safe_path(entry_path).read_bytes()).hexdigest()))
    require(entry.get('approved_saved_current_evaluation') is True, 'Separate true root entry required')
    require(entry.get('selected_arm') == 'CONTROL_FIXED_Q' and entry.get('threshold_logit') == 0, 'Fixed SOURCE selection/logit0')
    require(entry.get('selection_fixed_before_current_annotations_and_scores') is True, 'Selection chronology required')
    require(safe_path(entry['code']['path']) == Path(__file__).absolute(), 'Exact evaluator code binding'); raw(entry['code'])
    require(safe_path(entry['route']['path']) == H/'EVALUATION_ROUTE.json', 'Exact prepared route')
    route = document(entry['route']); raw(route['inherited_protocol'])
    selection = document(entry['selection_freeze'])
    require(bool(entry['selection_expected_fields']), 'Explicit fixed selection fields required')
    require(all(selection.get(k) == v for k,v in entry['selection_expected_fields'].items()), 'Selection freeze fields')
    pred_complete = document(entry['prediction_complete']); freeze = document(entry['prediction_freeze'])
    for obj in (pred_complete, freeze): require(obj.get('completed') is True and obj.get('PASS') is True, 'Prediction completion/freeze barrier')
    expected = entry['prediction_expected_fields']
    require(expected.get('selected_arm') == 'CONTROL_FIXED_Q' and expected.get('annotations_or_baseline_scores_read') == 0, 'Fixed producer access contract')
    require(all(pred_complete.get(k) == v for k,v in expected.items()), 'Prediction COMPLETE expected fields')
    for key in ('rows', 'prediction_freeze'):
        require(pred_complete[key] == entry['prediction_'+key] if key == 'rows' else pred_complete[key] == entry[key], 'Prediction exact closure '+key)
    predictions = json_lines(entry['prediction_rows'])
    require(len(predictions) == 1784, 'All original 1784 predictions retained')
    # 이 지점까지 GT/baseline 본문은 미열람. 이후 root가 명시한 경로만 읽는다.
    require(entry['current_bindings'] == route['current_bindings'], 'No replacement GT/baseline paths')
    ns = dict(Counter=Counter, math=math, require=require, D=D, E=E, S=S, GROUPS=GROUPS,
              c=SimpleNamespace(require=require), PAIRS=(('v61', TAG), ('legacy', TAG)))
    load_functions(route['code_bindings']['metric'], ('yes','ratio','call','calculate'), ns)
    load_functions(route['code_bindings']['core'], ('key','boolean','numeric_same','historical_replay'), ns)
    load_functions(route['code_bindings']['strata'], ('assert_row_order','error_bucket','paired_transitions'), ns)
    key, yes, same = ns['key'], ns['yes'], ns['numeric_same']
    bodies = entry['current_bindings']
    truth, prior = csv_rows(bodies['truth']), json_lines(bodies['prior'])
    manual, quality = csv_rows(bodies['manual']), csv_rows(bodies['frozen_annotation_ledger'])
    ns['assert_row_order'](SimpleNamespace(key=key), truth, prior)
    require(len(quality) == 1784 and [key(r) for r in quality] == [key(r) for r in truth], 'Frozen quality association only')
    lookup = {}
    for i,(t,r,p) in enumerate(zip(truth,prior,predictions)):
        require(p['domain'] == 'CURRENT223_1784' and p['id'] == t['id'], 'Original current domain/id')
        require(p.get('original_global_index',p.get('global_index')) == i and p['native_slot'] == int(t['native_corner']), 'Original row/native order')
        if 'global_index' in p: require(p['global_index'] == i, 'Global index alias')
        require(p['row_id'] == t['id']+':native'+str(int(t['native_corner'])), 'Original row_id')
        require(p['frame_index'] == i//8 and p['native_slot'] == i%8, 'Original frame/native slots')
        if 'group_id' in r: require(p['group_id'] == r['group_id'], 'Original input group')
        require(same(r['query_x'],p['query_xy'][0]) and same(r['query_y'],p['query_xy'][1]) and same(r['roi_side'],p['L']), 'Unchanged original q/L')
        available = ns['boolean'](r['valid']); require(p['original_query_available'] is available, 'Original input availability')
        state,z = p['prediction'],p['logit']; require(state in ('D','E','U'), 'Saved D/E/U only')
        if state == 'U': require(z is None and bool(p.get('U_reason')), 'U reason/null logit retained')
        else:
            require(available and isinstance(z,(int,float)) and math.isfinite(z), 'Decided finite available input')
            require(state == ('E' if z >= 0 else 'D'), 'Fixed raw logit0 sign')
        if not available: require(state == 'U', 'Original U must remain U')
        lookup[key(t)] = state
    original_call = ns['call']
    def call(row, model, task, probabilities=None, threshold=.5, rules=None):
        if model == TAG: return {'D':'VISIBLE','E':'ABSENT_NONVISIBLE','U':'OTHER_UNRESOLVED'}[lookup[key(row)]]
        return original_call(row,model,task,probabilities or {},threshold,rules)
    ns['call'] = call
    calculate = lambda rr,m,task='D_vs_E': ns['calculate'](rr,m,task,{},.5)
    selected = lambda rr,cohort,task: [r for r in rr if yes(r,cohort+('_D_vs_ES' if task=='D_vs_S' else '_'+task)) and r['human_state'] in TASKS[task]]
    primary = selected(truth,'paired_same_query_RGB','D_vs_E')
    require((len(primary),sum(r['human_state']==D for r in primary),sum(r['human_state']==E for r in primary)) == (1237,1147,90), 'Frozen primary1237 D1147 E90')
    require(len(selected(truth,'same_slot_pipeline','D_vs_E')) == 1248, 'Distinct same-slot1248')
    require(sum(r['human_state']==S for r in truth)==136 and sum(yes(r,'paired_same_query_RGB_D_vs_ES') and r['human_state']==S for r in truth)==21, 'All SELF136/paired21')
    require(sum(not ns['boolean'](r['valid']) for r in prior)==45, 'Original45 unavailable')
    metrics, coord, self_rows, native = [], [], [], []
    for cohort in ('paired_same_query_RGB','same_slot_pipeline'):
        for group in GROUPS:
            rr = [r for r in truth if r['group']==group]
            for task in TASKS:
                qq = selected(rr,cohort,task)
                if not any(r['human_state']!=D for r in qq): continue
                metrics.extend(dict(cohort=cohort,group=group,task=task,model=m,**calculate(qq,m,task)) for m in MODELS)
            for bucket in ('LE3','GT3','MISSING_REFERENCE_ERROR'):
                qq = [r for r in selected(rr,cohort,'D_vs_E') if ns['error_bucket'](r)==bucket]
                coord.extend(dict(cohort=cohort,group=group,error_bin=bucket,model=m,**calculate(qq,m)) for m in MODELS)
            qq = [r for r in rr if yes(r,cohort+'_D_vs_ES') and r['human_state']==S]
            for m in MODELS:
                cc = Counter(call(r,m,'D_vs_ES') for r in qq)
                self_rows.append(dict(cohort=cohort,group=group,model=m,N=len(qq),**dict(cc),scope='Binary response on human SELF, not learned SELF/external cause'))
    replay = ns['historical_replay'](metrics,csv_rows(bodies['historical_original']),{'v61','legacy'})
    for group in ('ALL',*GROUPS):
        for k in range(8):
            rr = [r for r in primary if int(r['native_corner'])==k and (group=='ALL' or r['group']==group)]
            native.extend(dict(group=group,native_corner=k,model=m,**calculate(rr,m)) for m in MODELS)
    manual_near = [r for r in manual if r['annotation_coordinate_source']=='manual_click' and r['human_state']==D and float(r['initial_reference_error_px'])<=3 and r['query_valid']=='True']
    require(len(manual_near)==376 and sum(r['human_state']==D and ns['error_bucket'](r)=='LE3' for r in truth)==461, 'Manual376 distinct from LE3 humanD461')
    manual_table = [dict(group=g,model=m,**calculate([r for r in manual_near if g=='ALL' or r['group']==g],m)) for g in ('ALL',*GROUPS) for m in MODELS]
    ledger=[]
    for i,(r,q,p) in enumerate(zip(truth,quality,predictions)):
        finite_value=lambda x: float(x) if isinstance(x,(int,float)) and math.isfinite(x) else None
        row = dict(global_index=i,id=r['id'],row_id=p['row_id'],frame_index=p['frame_index'],native_corner=int(r['native_corner']),group=r['group'],material=r.get('material','WOOD' if r['group']==GROUPS[0] else 'SQUARE'),human_state=r['human_state'],primarypairedbool=bool(yes(r,'paired_same_query_RGB_D_vs_E') and r['human_state'] in (D,E)),paired_same_query_RGB_D_vs_E=r['paired_same_query_RGB_D_vs_E'],same_slot_pipeline_D_vs_E=r['same_slot_pipeline_D_vs_E'],query_x=finite_value(p['query_xy'][0]),query_y=finite_value(p['query_xy'][1]),roi_side=finite_value(p['L']),original_query_valid=p['original_query_available'],human_reference_x=r.get('human_reference_x',''),human_reference_y=r.get('human_reference_y',''),initial_reference_error_px=r['initial_reference_error_px'],annotation_coordinate_source=q.get('annotation_coordinate_source','UNKNOWN_UNCERTIFIED'),mapping_status=r['mapping_status'],logit=p['logit'],U_reason=p['U_reason'])
        for m in MODELS:
            response = 'OTHER_UNRESOLVED' if m=='v61' and (r['query_valid']!='True' or r.get('present_at_fixed_0_5') not in ('True','False')) else call(r,m,'D_vs_E')
            row[m+'_original_response']=response; row[('V229' if m==TAG else m)+'_D_E_U']={'VISIBLE':'D','ABSENT_NONVISIBLE':'E'}.get(response,'U')
        ledger.append(row)
    historical_ledger=csv_rows(bodies['historical_native_ledger'])
    require(len(historical_ledger)==1784, 'Historical baseline ledger length')
    for a,b in zip(ledger,historical_ledger):
        require(str(a['id'])==b['id'] and str(a['native_corner'])==b['native_corner'], 'Baseline ledger order')
        for m in ('v61','legacy'):
            for suffix in ('_original_response','_D_E_U'): require(a[m+suffix]==b[m+suffix], 'Unchanged baseline ledger')
    primary_table = [dict(group=g,model=m,**calculate([r for r in primary if g=='ALL' or r['group']==g],m)) for g in ('ALL',*GROUPS) for m in MODELS]
    transitions=ns['paired_transitions'](SimpleNamespace(GROUPS=GROUPS,D=D,E=E,yes=yes),truth,lambda r,m:call(r,m,'D_vs_E'))
    inventory=[]
    for g in ('ALL',*GROUPS):
        rr=[r for r in ledger if g=='ALL' or r['group']==g]
        for m in MODELS:
            cc=Counter(r[('V229' if m==TAG else m)+'_D_E_U'] for r in rr)
            inventory.append(dict(group=g,model=m,N=len(rr),D=cc['D'],E=cc['E'],U=cc['U']))
    matrices=[dict(group=r['group'],model=r['model'],N=r['N'],D_D=r['TP'],D_E=r['FN_decided'],D_U=sum(r[k] for k in ('U_visible','out_of_task_S_visible','other_unresolved_visible')),E_D=r['FP'],E_E=r['TN'],E_U=sum(r[k] for k in ('U_occluded','out_of_task_S_occluded','other_unresolved_occluded'))) for r in primary_table]
    output=Path(entry['output']); require('v112' not in str(output).lower() and output.is_absolute(),'Output path')
    safe_path(output.parent); output.mkdir(exist_ok=False)
    tables={'METRICS':metrics,'PRIMARY':primary_table,'D_VS_EXTERNAL_E_2X3':matrices,'COORDINATE_ERROR_BINS':coord,'HUMAN_SELF_BINARY_RESPONSE':self_rows,'MANUAL_NEAR_D':manual_table,'PRIMARY_NATIVE8':native,'PAIRED_RECOVERY_AND_DAMAGE':transitions,'ALL1784_MODEL_INVENTORY':inventory,'ALL1784_LEDGER':ledger}
    for name,rows in tables.items(): save_csv(output/(name+'.csv'),rows)
    save_json(output/'SUMMARY.json',dict(completed=True,PASS=True,phase='V229',models=MODELS,primary_N=1237,primary_D=1147,primary_E=90,same_slot_N=1248,all_slots=1784,original_input_U=45,self_all=136,self_paired=21,manual_near_D=376,LE3_reference_human_D=461,coordinate_bins=['LE3','GT3','MISSING_REFERENCE_ERROR'],primary_is_coordinate_filter=False,primary_uses_additional_availability_filter=False,historical_baseline_metric_rows_exact=replay,tables=tables,threshold_logit=0,selected_model='CONTROL_FIXED_Q',selection_changed_by_evaluation=False,new_training_or_forward_or_PnP=0,reference_scope='Frozen reference association, not corrected coordinates or new point gold',repeated_development_exposure=True,fresh_blind_validation=False,legacy_CAL_overlap_preserved=True))
    outputs={p.name:bound(p) for p in output.iterdir()}
    complete=dict(completed=True,PASS=True,code=entry['code'],route=entry['route'],selection_freeze=entry['selection_freeze'],prediction_complete=entry['prediction_complete'],prediction_freeze=entry['prediction_freeze'],prediction_rows=entry['prediction_rows'],current_bindings=bodies,outputs=outputs,model_forward=0,fit=0,PnP=0,threshold_changes=0,primary_and_mapping_unchanged=True)
    save_json(output/'COMPLETE.json',complete); print(json.dumps(dict(completed=True,PASS=True,complete=bound(output/'COMPLETE.json'))))

if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--entry',required=True)
    main(parser.parse_args().entry)
