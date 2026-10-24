"""Independent scalar verification of closed V184 saved ledgers and aggregates.

No evaluator/producer import and no NumPy/model/array/pixel loader. --prepare
opens only prospective rules and invented rows; actual bodies need root entry.
"""
from pathlib import Path
from collections import Counter
import argparse
import csv
import hashlib
import json
import math
import platform
import traceback

H=Path(__file__).resolve().parent
V=H.parent
BASE=V/'saved_real_head_evaluation_v1'
COMPLETE=BASE/'COMPLETE.json'
EXPECTED_COMPLETE='bd6454bfd495cd218e11725090f148a61f6a566f441f83df41963e30a76d241a'
RULES=H/'RULES_V2_BEFORE_ORIGINAL_COHORT_BODY.json'
REPAIR=V/'ROOT_MINIMAL_INDEPENDENT_REAL_AUDITOR_COHORT_SCHEMA_REPAIR.json'
PROTOCOL=H/'PROTOCOL_V2_BEFORE_ORIGINAL_COHORT_BODY.json'
CHECKS=H/'GENERATED_INVENTED_CHECKS_V2.json'
PREP=H/'PREPARATION_COMPLETE_V2.json'
OUTPUT=H/'REVIEW_V2_COHORT_JOIN.json'
GROUPS=('WOOD_PRIMARY105','SQUARE118')
MODELS=('v61','legacy','C52','C53')
NEW=('C52','C53')
D,E,S='DIRECT_VISIBLE','EXTERNAL_OCCLUDED','SELF_OCCLUDED'
STATES={'D_vs_E':(D,E),'D_vs_S':(D,S),'D_vs_ES':(D,E,S)}
OWNER=['TARGET_VISIBLE_SUPPORTED','FOREIGN_VISIBLE_SUPPORTED','BACKGROUND_SUPPORTED']
U='UNKNOWN'


def require(value,message):
    if not value:raise AssertionError(message)


def sha(path):
    require('_v112' not in str(path).lower(),'V112 prohibited')
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def bind(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def check(b):require(bind(b['path'])==b,'Immutable saved file binding');return Path(b['path'])
def read(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x]
def csvrows(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def same(a,b,where='value'):
    if isinstance(a,dict):
        require(isinstance(b,dict) and set(a)<=set(b),where+' dictionary coverage')
        for k in a:same(a[k],b[k],where+'.'+str(k))
    elif isinstance(a,(list,tuple)):
        require(isinstance(b,(list,tuple)) and len(a)==len(b),where+' list shape')
        for i,(x,y) in enumerate(zip(a,b)):same(x,y,where+'['+str(i)+']')
    elif isinstance(a,float):require(isinstance(b,(float,int)) and math.isclose(a,float(b),rel_tol=0,abs_tol=1e-15),where+' float mismatch')
    else:require(a==b,where+' mismatch: '+repr(a)+' != '+repr(b))


def yes(value):
    require(value in (True,False,'True','False'),'Strict saved boolean')
    return value is True or value=='True'


def finite(value):
    try:return math.isfinite(float(value))
    except (TypeError,ValueError):return False


def ratio(a,b):return a/b if b else None
def key(row):return str(row['id']),int(row['native_corner'])
def bucket(row):
    value=row.get('initial_reference_error_px')
    return 'MISSING_REFERENCE_ERROR' if not finite(value) else 'LE3' if float(value)<=3 else 'GT3'


def response(row,model,task='D_vs_E',ledger=False):
    if model=='legacy':
        state=row['legacy_original_state']
        return ('VISIBLE' if state in ('B','I') else 'ABSENT_NONVISIBLE' if state=='O'
                else ('OUT_OF_TASK_GEOMETRY_S' if task=='D_vs_E' else 'ABSENT_NONVISIBLE') if state=='S'
                else 'ABSTAIN' if state=='U' else 'OTHER_UNRESOLVED')
    if model=='v61':
        probability=row.get('v61_saved_p_present')
        if ledger and (not yes(row['original_query_valid']) or not finite(probability)):
            return 'OTHER_UNRESOLVED'
        # Original historical operator: saved fixed0.5 probability. No fitted threshold.
        return 'VISIBLE' if finite(probability) and float(probability)>=.5 else 'ABSENT_NONVISIBLE'
    pred=int(row[model+'_regional_support_proxy_D0_E1_Uminus1'])
    require(pred in (-1,0,1),'Regional saved decision enum')
    return 'OTHER_UNRESOLVED' if pred==-1 else 'VISIBLE' if pred==0 else 'ABSENT_NONVISIBLE'


def metric(rows,model,task):
    counts={name:0 for name in ('TP','FN_decided','FP','TN','U_visible','U_occluded','out_of_task_S_visible','out_of_task_S_occluded','other_unresolved_visible','other_unresolved_occluded')}
    pos=neg=0
    for row in rows:
        positive=row['human_state']==D
        pos+=int(positive);neg+=int(not positive)
        call=response(row,model,task)
        field=({'VISIBLE':'TP','ABSENT_NONVISIBLE':'FN_decided','ABSTAIN':'U_visible','OUT_OF_TASK_GEOMETRY_S':'out_of_task_S_visible','OTHER_UNRESOLVED':'other_unresolved_visible'} if positive else
               {'VISIBLE':'FP','ABSENT_NONVISIBLE':'TN','ABSTAIN':'U_occluded','OUT_OF_TASK_GEOMETRY_S':'out_of_task_S_occluded','OTHER_UNRESOLVED':'other_unresolved_occluded'})[call]
        counts[field]+=1
    tp,fn,fp,tn=(counts[k] for k in ('TP','FN_decided','FP','TN'))
    abstain=counts['U_visible']+counts['U_occluded'];geometry=counts['out_of_task_S_visible']+counts['out_of_task_S_occluded']
    unresolved=abstain+geometry+counts['other_unresolved_visible']+counts['other_unresolved_occluded']
    require(tp+fn+counts['U_visible']+counts['out_of_task_S_visible']+counts['other_unresolved_visible']==pos,'D partition')
    require(fp+tn+counts['U_occluded']+counts['out_of_task_S_occluded']+counts['other_unresolved_occluded']==neg,'E/S partition')
    return dict(N=len(rows),frames=len({r['id'] for r in rows}),visible_support=pos,occluded_support=neg,**counts,
        visible_precision=ratio(tp,tp+fp),visible_recall_all_GT=ratio(tp,pos),occluded_precision=ratio(tn,tn+fn),occluded_recall_all_GT=ratio(tn,neg),
        abstention_rate=ratio(abstain,len(rows)),out_of_task_S_rate=ratio(geometry,len(rows)),unresolved_rate=ratio(unresolved,len(rows)),
        decision_coverage=ratio(tp+fn+fp+tn,len(rows)),correct_fraction_all_GT=ratio(tp+tn,len(rows)),selective_accuracy=ratio(tp+tn,tp+fn+fp+tn))


def select(rows,cohort,group,task,error=None):
    flag=cohort+'_'+('D_vs_ES' if task=='D_vs_S' else task)
    return [r for r in rows if r['group']==group and yes(r[flag]) and r['human_state'] in STATES[task]
            and (error is None or bucket(r)==error)]


def new_decode(row,tag):
    raw_s=row[tag+'_raw_support_logits'];raw_o=row[tag+'_raw_owner_logits'];available=yes(row['query_available'])
    require(len(raw_s)==2 and len(raw_o)==3,'Rawsupport2/owner3 ledger shape')
    if not available:
        require(all(x is None for x in raw_s+raw_o),'InputU raw NaN serializednull')
        return [-1,-1],-1,-1
    require(all(finite(x) for x in raw_s+raw_o),'Available raw ledger finite')
    support=[int(float(x)>0) for x in raw_s]
    owner=max(range(3),key=lambda i:float(raw_o[i]))
    proxy=1 if support[1] else 0 if support[0] else -1
    return support,owner,proxy


def support_joint(support):
    require(len(support)==2 and (all(x in (0,1) for x in support) or support==[-1,-1]),'Support enum')
    return 4 if support==[-1,-1] else support[0]+2*support[1]


def verify_current(rows,result,manual):
    require(len(rows)==1784 and [r['global_index'] for r in rows]==list(range(1784)) and len({key(r) for r in rows})==1784,'Original1784 ordered unique rows')
    require(len({r['id'] for r in rows})==223 and sum(not yes(r['query_available']) for r in rows)==45,'223frames/45inputU')
    for group,frames in zip(GROUPS,(105,118)):
        rr=[r for r in rows if r['group']==group];require(len(rr)==frames*8 and len({r['id'] for r in rr})==frames,'Whole group/native slots')
    for row in rows:
        require(int(row['native_corner']) in range(8),'Native0..7')
        for tag in NEW:
            support,owner,proxy=new_decode(row,tag)
            same(support,row[tag+'_support_prediction'],tag+' support decoder');same(owner,row[tag+'_owner_prediction'],tag+' argmax decoder')
            same(proxy,row[tag+'_regional_support_proxy_D0_E1_Uminus1'],tag+' Efirst proxy')
        for model in MODELS:
            call=response(row,model,ledger=True)
            same(call,row[model+'_original_response'],model+' ledger response')
            same('D' if call=='VISIBLE' else 'E' if call=='ABSENT_NONVISIBLE' else 'U',row[model+'_D_E_U'],model+' ledger D/E/U')
    primary=[r for r in rows if yes(r['paired_same_query_RGB_D_vs_E']) and r['human_state'] in (D,E)]
    require(len(primary)==1237 and sum(r['human_state']==D for r in primary)==1147 and sum(r['human_state']==E for r in primary)==90,'Fixed primary D1147/E90')
    require(sum(yes(r['same_slot_pipeline_D_vs_E']) and r['human_state'] in (D,E) for r in rows)==1248,'Same-slot1248')
    tables=result['tables'];checked=Counter()
    for saved in tables['metrics']:
        rr=select(rows,saved['cohort'],saved['group'],saved['task'])
        same(metric(rr,saved['model'],saved['task']),saved,'Current metric');checked['metric_rows']+=1
    for saved in tables['confusion']:
        rr=[r for r in select(rows,saved['cohort'],saved['group'],saved['task']) if r['human_state']==saved['human_state']]
        counts=Counter(response(r,saved['model'],saved['task']) for r in rr)
        same(dict(N=len(rr),pred_visible=counts['VISIBLE'],pred_binary_hidden=counts['ABSENT_NONVISIBLE'],legacy_U=counts['ABSTAIN'],
            out_of_task_geometry_S=counts['OUT_OF_TASK_GEOMETRY_S'],input_or_operator_U=counts['OTHER_UNRESOLVED']),saved,'Current confusion');checked['confusion_rows']+=1
    for saved in tables['human_self_binary_response']:
        rr=[r for r in rows if r['group']==saved['group'] and yes(r[saved['cohort']+'_D_vs_ES']) and r['human_state']==S]
        counts=Counter(response(r,saved['model'],'D_vs_S') for r in rr)
        same(dict(human_S_support=len(rr),binary_hidden_response=counts['ABSENT_NONVISIBLE'],visible_response=counts['VISIBLE'],
            unresolved=len(rr)-counts['ABSENT_NONVISIBLE']-counts['VISIBLE'],human_S_binary_hidden_response_rate=ratio(counts['ABSENT_NONVISIBLE'],len(rr))),saved,'Self separate');checked['self_rows']+=1
    bykey={key(r):r for r in rows};provenance={key(r):r for r in manual}
    near=[]
    for r in manual:
        if r['annotation_coordinate_source']=='manual_click' and r['human_state']==D and finite(r['initial_reference_error_px']) and float(r['initial_reference_error_px'])<=3 and yes(r['query_valid']):
            require(key(r) in bykey,'Original manualnear ID join')
            row=bykey[key(r)]
            require(row['human_state']==D and row['group']==r['group'],'Original manualnear semantic/group')
            for old,new in (('query_x','query_x'),('query_y','query_y')):
                require(float(row[new])==float(r[old]),'Manualnear unchanged q')
            near.append(row)
    require(len(near)==376 and sum(r['group']==GROUPS[0] for r in near)==153 and sum(r['group']==GROUPS[1] for r in near)==223,'Original exact manualnear membership376')
    for saved in tables['manual_near_D']:
        rr=[r for r in near if r['group']==saved['group']];counts=Counter(response(r,saved['model'],'D_vs_ES') for r in rr)
        same(dict(N=len(rr),pred_visible=counts['VISIBLE'],pred_binary_hidden=counts['ABSENT_NONVISIBLE'],
            unresolved=len(rr)-counts['VISIBLE']-counts['ABSENT_NONVISIBLE'],semantic_D_recall=ratio(counts['VISIBLE'],len(rr))),saved,'Original manualnear');checked['manualnear_rows']+=1
    for saved in tables['coordinate_error_bins']:
        rr=select(rows,saved['cohort'],saved['group'],'D_vs_E',saved['error_bin'])
        expected=metric(rr,saved['model'],'D_vs_E');nmanual=sum(provenance.get(key(r),{}).get('annotation_coordinate_source')=='manual_click' for r in rr)
        expected.update(manual_click_reference_N=nmanual,other_or_unspecified_reference_N=len(rr)-nmanual)
        same(expected,saved,'Frozen reference bin');checked['reference_bin_rows']+=1
    for saved in tables['D_vs_external_E_2x3']:
        rr=select(rows,saved['cohort'],saved['group'],'D_vs_E');m=metric(rr,saved['model'],'D_vs_E')
        same(dict(human_D_N=m['visible_support'],human_external_E_N=m['occluded_support'],D_pred_D=m['TP'],D_pred_hidden=m['FN_decided'],
            D_pred_U=m['U_visible']+m['out_of_task_S_visible']+m['other_unresolved_visible'],E_pred_D=m['FP'],E_pred_hidden=m['TN'],
            E_pred_U=m['U_occluded']+m['out_of_task_S_occluded']+m['other_unresolved_occluded'],D_precision=m['visible_precision'],D_recall=m['visible_recall_all_GT'],
            external_E_response_precision=m['occluded_precision'],external_E_response_recall=m['occluded_recall_all_GT'],unresolved_rate=m['unresolved_rate']),saved,'D/E2x3');checked['two_by_three_rows']+=1
    for saved in tables['all1784_model_inventory']:
        rr=rows if saved['group']=='ALL' else [r for r in rows if r['group']==saved['group']]
        counts=Counter(response(r,saved['model'],ledger=True) for r in rr);n=len(rr)
        same(dict(all_registered_slots=n,D=counts['VISIBLE'],E_binary_response=counts['ABSENT_NONVISIBLE'],U=n-counts['VISIBLE']-counts['ABSENT_NONVISIBLE'],
            U_rate=ratio(n-counts['VISIBLE']-counts['ABSENT_NONVISIBLE'],n),legacy_abstain=counts['ABSTAIN'],out_of_task_geometry_S=counts['OUT_OF_TASK_GEOMETRY_S'],other_unresolved=counts['OTHER_UNRESOLVED']),saved,'All1784 ownavailability inventory')
        same(dict(Counter(r['human_state'] for r in rr)),json.loads(saved['human_state_inventory']),'Human states inventory');checked['inventory_rows']+=1
    for saved in tables['support_joint_owner_and_reference_associations']:
        rr=[r for r in rows if (saved['group']=='ALL' or r['group']==saved['group'])
            and (saved['error_bin']=='ALL' or bucket(r)==saved['error_bin'])
            and (r['human_state'] if r['human_state'] in (D,E,S) else U)==saved['human_semantic_state']]
        tag=saved['model'];joint=Counter(support_joint(r[tag+'_support_prediction']) for r in rr)
        owner=Counter(r[tag+'_owner_prediction'] for r in rr);proxy=Counter(r[tag+'_regional_support_proxy_D0_E1_Uminus1'] for r in rr)
        same(dict(N=len(rr),support_NONE=joint[0],support_VISIBLE_ONLY=joint[1],support_FOREIGN_ONLY=joint[2],support_BOTH=joint[3],support_U=joint[4],
            proxy_D=proxy[0],proxy_E=proxy[1],proxy_U=proxy[-1],owner_T=owner[0],owner_F=owner[1],owner_BG=owner[2],owner_U=owner[-1],
            original_input_U=sum(not yes(r['query_available']) for r in rr)),saved,'Support/owner/reference association');checked['association_rows']+=1
    for saved in tables['paired_recovery_and_damage']:
        rr=[r for r in select(rows,saved['cohort'],saved['group'],'D_vs_E') if saved['truth_class']=='ALL_DE' or r['human_state']==saved['truth_class']]
        counts=Counter()
        for row in rr:
            a=response(row,saved['baseline']);b=response(row,saved['new_model']);truth='VISIBLE' if row['human_state']==D else 'ABSENT_NONVISIBLE'
            name=('new_unresolved' if b not in ('VISIBLE','ABSENT_NONVISIBLE') else
                  ('baseline_unresolved_new_correct' if b==truth else 'baseline_unresolved_new_wrong') if a not in ('VISIBLE','ABSENT_NONVISIBLE') else
                  'both_correct' if a==truth and b==truth else 'baseline_correct_new_wrong' if a==truth else 'baseline_wrong_new_correct' if b==truth else 'both_wrong')
            counts[name]+=1
        fields=('both_correct','baseline_correct_new_wrong','baseline_wrong_new_correct','both_wrong','baseline_unresolved_new_correct','baseline_unresolved_new_wrong','new_unresolved')
        same(dict(N=len(rr),**{k:counts[k] for k in fields}),saved,'Paired changes');checked['paired_rows']+=1
    require(result['primary_quality_owner_reference_error_filter'] is False and result['human_semantic_authority'].startswith('Existing human D/E/S'),'Existing authority/primary gates')
    return dict(checked),dict(rows1784=True,primary_D1147_E90_N1237=True,manual_original_membership376=True,U45_preserved=True,
        historical_original_metric_rows_exact_attestation=result['historical_original_metric_rows_exact'],
        historical_v61_legacy_tables_exact_attestation=result['historical_v61_legacy_tables_exact'],
        baseline_original_tables_not_reopened_in_this_audit=True)


def matrix(truth,prediction,names):
    k=len(names);cm=[[0]*(k+1) for _ in range(k+1)]
    require(len(truth)==len(prediction),'Metric row census')
    for t,p in zip(truth,prediction):
        require(t in range(-1,k) and p in range(-1,k),'Unknown-aware class enum')
        cm[k if t<0 else t][k if p<0 else p]+=1
    details={}
    for j,name in enumerate(names):
        tp=cm[j][j];predicted=sum(cm[i][j] for i in range(k));support=sum(cm[j])
        details[name]=dict(TP=tp,predicted_on_known_truth=predicted,known_truth_support=support,
            precision_on_known_truth=ratio(tp,predicted),recall_including_prediction_U=ratio(tp,support))
    return dict(N=len(truth),truth_class_order=names+[U],prediction_class_order=names+['U'],full_confusion=cm,
        known_truth_confusion=cm[:k],known_truth=sum(sum(row) for row in cm[:k]),unknown_truth=sum(cm[k]),
        prediction_U_all=sum(row[k] for row in cm),prediction_U_on_known_truth=sum(row[k] for row in cm[:k]),
        predictions_on_unknown_truth=cm[k],per_class=details,precision_excludes_unknown_truth=True,weak_reference_only=True)


def binary(owner):return -1 if owner<0 else 0 if owner==0 else 1
def crosstab(a,b,na,nb):
    table=[[0]*nb for _ in range(na)]
    for x,y in zip(a,b):require(x in range(na) and y in range(nb),'Association enum');table[x][y]+=1
    return table


def numeric_same(a,b):
    missing=lambda x:x is None or str(x).lower() in ('','none','nan')
    return missing(a) and missing(b) or not missing(a) and not missing(b) and float(a)==float(b)


def join_original_cohorts(rows,truth,n):
    """Only restore two explicitly saved original cohort flags; never infer them."""
    require(len(rows)==len(truth)==n,'Original fixed cohort allrow census')
    require(len({key(r) for r in rows})==len({key(r) for r in truth})==n,'Original fixed cohort unique native keys')
    require([key(r) for r in rows]==[key(r) for r in truth],'Original fixed cohort complete key order')
    joined=[]
    fields=('paired_same_query_RGB_D_vs_ES','same_slot_pipeline_D_vs_ES')
    for row,t in zip(rows,truth):
        require(row['group']==t['group'] and row['human_state']==t['human_state'],'Original fixed group/human state')
        require(numeric_same(row['query_x'],t['query_x']) and numeric_same(row['query_y'],t['query_y'])
                and numeric_same(row['roi_side'],t['v61_roi_side_raw_px']),'Original unchanged q/L')
        for name in ('paired_same_query_RGB_D_vs_E','same_slot_pipeline_D_vs_E'):
            require(yes(row[name])==yes(t[name]),'Original existing D/E cohort unchanged')
        copy=dict(row)
        for name in fields:
            yes(t[name])
            if name in row:require(yes(row[name])==yes(t[name]),'No cohort overwrite')
            copy[name]=t[name]
        require({k:v for k,v in copy.items() if k not in fields}=={k:v for k,v in row.items() if k not in fields},'Only two original flags joined')
        joined.append(copy)
    return joined


def verify_public(rows,result):
    require(len(rows)==360 and [r['original_global_index'] for r in rows]==list(range(584,944)),'All held360 original IDs')
    require(len({r['frame_index'] for r in rows})==45 and {r['frame_index'] for r in rows}==set(range(73,118)),'Original45 frames')
    truth=[];crop=[];prediction={t:[] for t in (*NEW,'v61')}
    for row in rows:
        require(row['center_owner'] in OWNER+[U] and row['crop_presence'] in ['YES','NO',U],'Frozen weak enums')
        if not yes(row['source_original_query_available']) or not yes(row['source_center4_allvalid']):require(row['center_owner']==U,'Original forced centerU')
        if not yes(row['source_crop_constructed']):require(row['crop_presence']==U,'Missing cropU')
        truth.append(OWNER.index(row['center_owner']) if row['center_owner'] in OWNER else -1);crop.append(['YES','NO',U].index(row['crop_presence']))
        for tag in NEW:
            prediction[tag].append(int(row[tag+'_owner_prediction']))
            support=row[tag+'_support_prediction'];joint=support_joint(support)
            expected=1 if support[1]==1 else 0 if support[0]==1 else -1
            require(row[tag+'_regional_support_proxy_D0_E1_Uminus1']==expected,'Saved public E-first fromsupport')
            if not yes(row['source_original_query_available']):require(joint==4 and prediction[tag][-1]==-1,'SourceU preserved')
        prediction['v61'].append(int(row['v61_binary_T_nonT_prediction']))
    require(sum(t>=0 for t in truth)==204 and sum(t<0 for t in truth)==156,'Original known204/unknown156')
    checked=Counter()
    for cohort,strata in result['cohorts'].items():
        for group,saved in strata.items():
            ids=[]
            for i,row in enumerate(rows):
                paired=all(prediction[tag][i]>=0 for tag in (*NEW,'v61'))
                included=(cohort=='FULL360' or cohort=='KNOWN_CENTER_TRUTH' and truth[i]>=0
                    or cohort=='ALL3_MODEL_PAIRED_AVAILABLE' and paired or cohort=='KNOWN_TRUTH_AND_ALL3_PAIRED_AVAILABLE' and paired and truth[i]>=0)
                if included and (group=='ALL' or row['native_slot']==int(group)):ids.append(i)
            same(dict(N=len(ids),original_input_U=sum(not yes(rows[i]['source_original_query_available']) for i in ids),
                center4_invalid=sum(not yes(rows[i]['source_center4_allvalid']) for i in ids),no_constructed_crop=sum(not yes(rows[i]['source_crop_constructed']) for i in ids)),saved,'Public stratum inventory')
            for tag in (*NEW,'v61'):
                pred=[prediction[tag][i] for i in ids];gt=[truth[i] for i in ids]
                same(matrix([binary(x) for x in gt],pred if tag=='v61' else [binary(x) for x in pred],['T','NON_T']),saved['models'][tag]['binary_T_nonT'],'Weak binary CM/P/R/U');checked['binary_metrics']+=1
                if tag!='v61':same(matrix(gt,pred,OWNER),saved['models'][tag]['owner3'],'Weak owner3 CM/P/R/U');checked['owner3_metrics']+=1
    center=[x if x>=0 else 3 for x in truth]
    for tag,saved in result['support_crop_center_associations'].items():
        joint=[support_joint(r[tag+'_support_prediction']) for r in rows];proxy=[r[tag+'_regional_support_proxy_D0_E1_Uminus1'] for r in rows]
        owner=prediction[tag]
        same(dict(support_joint_vs_crop=crosstab(joint,crop,5,3),support_joint_vs_center=crosstab(joint,center,5,4),
            proxy_vs_crop=crosstab([x if x>=0 else 2 for x in proxy],crop,3,3),owner_vs_crop=crosstab([x if x>=0 else 3 for x in owner],crop,4,3)),saved,'Weak crop/center associations')
        checked['association_models']+=1
    expected=dict(original_input_U=sum(not yes(r['source_original_query_available']) for r in rows),source_available=sum(yes(r['source_original_query_available']) for r in rows),
        center4_invalid=sum(not yes(r['source_center4_allvalid']) for r in rows),otherwise_available_center4_invalid=sum(yes(r['source_original_query_available']) and not yes(r['source_center4_allvalid']) for r in rows),
        no_constructed_crop=sum(not yes(r['source_crop_constructed']) for r in rows),known_center204=sum(t>=0 for t in truth),unknown_center156=sum(t<0 for t in truth),
        v61_own_available=sum(x>=0 for x in prediction['v61']),v61_own_U=sum(x<0 for x in prediction['v61']),
        new_owner_U={tag:sum(x<0 for x in prediction[tag]) for tag in NEW},
        new_regional_U={tag:sum(row[tag+'_regional_support_proxy_D0_E1_Uminus1']<0 for row in rows) for tag in NEW})
    same(expected,result['coverage'],'Public independent availability/UNKNOWN denominators')
    require(expected['original_input_U']==108 and expected['source_available']==252 and expected['v61_own_available']==251 and expected['v61_own_U']==109,'Original source/v61 differing availability')
    require(result['legacy_baseline']['status']=='MISSING_IN_REVIEWED_LINEAGE' and result['training_admission'] is False,'Missing legacy not imputed/no training')
    return dict(checked),expected


def tests():
    passed=[]
    owner_gt=[0,1,2,-1,0,1];owner_pred=[0,2,-1,1,-1,1]
    m=matrix(owner_gt,owner_pred,['T','F','BG'])
    require(m['full_confusion']==[[1,0,0,1],[0,1,1,0],[0,0,0,1],[0,1,0,0]] and m['per_class']['F']['precision_on_known_truth']==1 and m['per_class']['T']['recall_including_prediction_U']==.5,'Unknown vsabstention precision/recall')
    require(matrix([binary(x) for x in owner_gt],[binary(x) for x in owner_pred],['T','NON_T'])['full_confusion']==[[1,0,1],[0,2,1],[0,1,0]],'Samecenter binarycollapse')
    passed.append('Independent_scalar_ownerCM_P_R_unknowntruth_predictionU_and_binarycollapse')
    row=dict(query_available=True,C52_raw_support_logits=[0.,1.],C52_raw_owner_logits=[1.,1.,0.])
    require(new_decode(row,'C52')==([0,1],0,1),'Strictsupport/firsttie')
    row['C52_raw_support_logits']=[1.,1.];require(new_decode(row,'C52')==([1,1],0,1),'BOTH Efirst')
    row['C52_raw_support_logits']=[0.,0.];require(new_decode(row,'C52')==([0,0],0,-1),'NONEU')
    row.update(query_available=False,C52_raw_support_logits=[None,None],C52_raw_owner_logits=[None,None,None]);require(new_decode(row,'C52')==([-1,-1],-1,-1),'InputU')
    passed.append('Independent_FP32_saved_scalar_strict0_BOTH_NONE_Efirst_ownerfirsttie_inputU')
    rows=[dict(id='a',human_state=D,legacy_original_state='U'),dict(id='b',human_state=E,legacy_original_state='S')]
    m=metric(rows,'legacy','D_vs_E');require(m['N']==2 and m['U_visible']==1 and m['out_of_task_S_occluded']==1 and m['unresolved_rate']==1,'Legacy ownS/U')
    require(metric(rows,'legacy','D_vs_ES')['TN']==1,'LegacyS taskroute')
    passed.append('Separate_legacy_ABSTAIN_geometryS_routing_and_full_known_denominators')
    rows=[dict(id='i',human_state=D,C52_regional_support_proxy_D0_E1_Uminus1=0),dict(id='j',human_state=D,C52_regional_support_proxy_D0_E1_Uminus1=-1),
          dict(id='k',human_state=E,C52_regional_support_proxy_D0_E1_Uminus1=0),dict(id='l',human_state=E,C52_regional_support_proxy_D0_E1_Uminus1=1)]
    m=metric(rows,'C52','D_vs_E');require(m['visible_precision']==.5 and m['visible_recall_all_GT']==.5 and m['occluded_recall_all_GT']==.5 and m['decision_coverage']==.75,'Scalar metric counts')
    passed.append('Independent_binary_CM_precision_recall_and_U_coverage')
    require(response(dict(v61_saved_p_present='.5',original_query_valid='True'),'v61',ledger=True)=='VISIBLE'
        and response(dict(v61_saved_p_present='',original_query_valid='False'),'v61',ledger=True)=='OTHER_UNRESOLVED','v61ownU')
    passed.append('Historical_v61_fixed0_5_zero_boundary_and_ownavailability_separate')
    require(bucket(dict(initial_reference_error_px='3'))=='LE3' and bucket(dict(initial_reference_error_px='3.01'))=='GT3'
        and bucket({})=='MISSING_REFERENCE_ERROR','Frozen referencebins')
    passed.append('Frozen_3px_and_missing_reference_no_primaryfilter')
    require(crosstab([0,1,2,3,4],[0,1,2,0,1],5,3)==[[1,0,0],[0,1,0],[0,0,1],[1,0,0],[0,1,0]],'Joint no merge')
    passed.append('Joint4states_plus_U_separate_from_weakcroppresence')
    require(matrix([],[],['T','NON_T'])['per_class']['T']['precision_on_known_truth'] is None,'Zero support no fake success')
    passed.append('Empty_native_stratum_zero_support_kept_None_metrics')
    malformed=0
    for fn in [lambda:support_joint([1,-1]),lambda:yes('unknown'),lambda:matrix([3],[0],['T','F','BG'])]:
        try:fn()
        except AssertionError:malformed+=1
    require(malformed==3,'Malformed states rejected')
    passed.append('Malformed_saved_enum_and_partial_supportU_rejected')
    row=dict(id='i',native_corner=7,group=GROUPS[0],human_state=S,query_x=1.,query_y=2.,roi_side=3.,
        paired_same_query_RGB_D_vs_E='False',same_slot_pipeline_D_vs_E='False')
    t=dict(row,v61_roi_side_raw_px=3.,paired_same_query_RGB_D_vs_ES='True',same_slot_pipeline_D_vs_ES='True')
    joined=join_original_cohorts([row],[t],1)
    require(select(joined,'paired_same_query_RGB',GROUPS[0],'D_vs_S')==joined and len(row)==9,'Actual missingflag restore helper, original row unmodified')
    rejected=0
    for delta in [dict(native_corner=6),dict(group=GROUPS[1]),dict(human_state=D),dict(query_x=1.01),dict(v61_roi_side_raw_px=4.),dict(paired_same_query_RGB_D_vs_E='True')]:
        try:join_original_cohorts([row],[dict(t,**delta)],1)
        except AssertionError:rejected+=1
    require(rejected==6,'Original cohort exact metadata joins reject drift')
    passed.append('Actual_V2_missing_DvsES_flags_exact_identity_qL_existingDE_regression_no_membership_inference')
    return dict(completed=True,PASS=True,invented_only=True,checks=passed,count=len(passed),actual_metric_ledger_label_score_body_or_hash_reads=0)


def prepare():
    require(platform.python_version()=='3.10.20','Pinned Python')
    rules=read(RULES)
    require(bind(REPAIR)==rules['repair_scope'] and read(REPAIR)['approved_minimal_independent_auditor_schema_repair'] is True,'Frozen root repair approval')
    require(rules['actual_metric_or_ledger_bodies_read_before_rule'] is False and rules['evaluation_complete_expected']==dict(path=str(COMPLETE),sha256=EXPECTED_COMPLETE),'Prospective rule fixed before actual bodies')
    write(CHECKS,tests())
    p=dict(completed=True,prepared_only=True,code=bind(__file__),rules=bind(RULES),checks=bind(CHECKS),repair_scope=bind(REPAIR),fixed_cohort_membership=rules['fixed_cohort_membership'],
        evaluation_complete=dict(path=str(COMPLETE),sha256=EXPECTED_COMPLETE),original_manual_membership=rules['original_manual_membership'],
        inputs=dict(current_ledger='current1784/ALL1784_CURRENT_NATIVE_LEDGER.jsonl',current_report='current1784/CURRENT_HUMAN_SEMANTIC_AND_SEPARATE_ASSOCIATIONS.json',
            public_ledger='publicheld360/ALL360_PUBLIC_WEAK_LEDGER.jsonl',public_report='publicheld360/PUBLIC_WEAK_OWNER_AND_SEPARATE_SUPPORT_ASSOCIATIONS.json'),
        independent_kernel='Python scalar loops only, no evaluator/producer/helper import; no NumPy/NPZ/rawpixel/checkpoint loader.',
        output=str(OUTPUT),automatic_retry=False,selected_model=None,new_training_admission=False,actual_saved_metric_bodies_read_during_prepare=0)
    write(PROTOCOL,p)
    template=H/'ROOT_ACTUAL_INDEPENDENT_SAVED_REVIEW_V2_ENTRY_TEMPLATE_APPROVED_FALSE.json'
    write(template,dict(approved_independent_saved_real_metric_review_V2=False,evaluation_only=True,no_model_forward_fit_threshold_data_or_arm_choice=True,
        code=bind(__file__),protocol=bind(PROTOCOL),repair_scope=bind(REPAIR),evaluation_complete=p['evaluation_complete'],output=str(OUTPUT)))
    write(PREP,dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),checks=bind(CHECKS),entry_template=bind(template),actual_metric_ledger_label_score_body_or_hash_reads=0))
    print(json.dumps(dict(FINAL_READY=True,code=bind(__file__),protocol=bind(PROTOCOL),checks=bind(CHECKS),preparation=bind(PREP))))


def run(entry_path):
    e=read(entry_path);require(e.get('approved_independent_saved_real_metric_review_V2') is True and e.get('evaluation_only') is True
        and e.get('no_model_forward_fit_threshold_data_or_arm_choice') is True,'Separate approved saved-audit entry')
    require(e['code']==bind(__file__) and e['protocol']==bind(PROTOCOL) and e['evaluation_complete']==dict(path=str(COMPLETE),sha256=EXPECTED_COMPLETE)
        and e['output']==str(OUTPUT),'Frozen audit code/protocol/completion/output')
    p=read(PROTOCOL);check(p['rules']);require(e['repair_scope']==p['repair_scope']==bind(REPAIR),'Exact minimal repair scope');require(read(check(p['checks']))['PASS'] and not OUTPUT.exists(),'Inventedchecks/single actualaudit')
    complete=read(check(p['evaluation_complete']));require(complete['completed'] and complete['PASS']
        and complete['all_both_predictions_closed_before_any_annotation_or_score_body'] is True and complete['selected_model'] is None,'Bothclosed actualsaved evaluation')
    consumed=[]
    try:
        data={}
        for name,relative in p['inputs'].items():
            b=complete['outputs'][relative];require(b['path']==str(BASE/relative),'Exact sealed outputpath');path=check(b);consumed.append(b)
            data[name]=lines(path) if path.suffix=='.jsonl' else read(path)
        manual=csvrows(check(p['original_manual_membership']));consumed.append(p['original_manual_membership'])
        original_cohorts=csvrows(check(p['fixed_cohort_membership']));consumed.append(p['fixed_cohort_membership'])
        joined_current=join_original_cohorts(data['current_ledger'],original_cohorts,1784)
        ccounts,cdenom=verify_current(joined_current,data['current_report'],manual)
        pcounts,pdenom=verify_public(data['public_ledger'],data['public_report'])
        for b in consumed+[p['evaluation_complete'],p['code'],p['rules'],p['checks'],p['repair_scope']]:check(b)
        review=dict(completed=True,PASS=True,audit_kind='Independent saved-ledger Python scalar recomputation; no producer/evaluator/helper import',
            code=bind(__file__),protocol=bind(PROTOCOL),rules=p['rules'],repair_scope=p['repair_scope'],original_two_cohort_flags_only_joined_after_exact1784_identity=True,entry=bind(entry_path),evaluation_complete=p['evaluation_complete'],
            current_checked=ccounts,current_denominators_and_baseline_attestation=cdenom,public_checked=pcounts,public_denominators=pdenom,
            consumed_bindings=consumed,all_consumed_saved_files_unchanged=True,models=list(MODELS),
            numerical_agreement_is_not_weak_label_physical_gold=True,current_human_semantic_authority_preserved=True,
            limitations=['No originalpixel/model/NPZ/checkpoint access; current raw support/owner is independently decoded only from saved ledger scalars.',
                'Public ledger contains saved decisions, not raw logits; CM/P/R/U and associations are recomputed, raw model/argmax proof remains the separate frozen prediction audit.',
                'Historical original baseline tables were not reopened here; their completed-evaluator exact replay attestation is preserved and all ledger-derived baseline metrics/ownavailability recounted.',
                'Development-exposed current223/publicheld45; no fresh independent generalization or automatic adoption claim.'],
            new_model_forward_fit_threshold_or_data_arm_selection=0,new_training_admission=False,selected_model=None,adopted=False,fully_solved=False,automatic_retry=False)
        write(OUTPUT,review);print(json.dumps(dict(completed=True,PASS=True,review=bind(OUTPUT))))
    except BaseException:
        write(H/'FAILED_ACTUAL_SAVED_REVIEW_V2.json',dict(completed=False,PASS=False,traceback=traceback.format_exc(),automatic_retry=False));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--entry',type=Path)
    args=parser.parse_args();prepare() if args.prepare else run(args.entry)
