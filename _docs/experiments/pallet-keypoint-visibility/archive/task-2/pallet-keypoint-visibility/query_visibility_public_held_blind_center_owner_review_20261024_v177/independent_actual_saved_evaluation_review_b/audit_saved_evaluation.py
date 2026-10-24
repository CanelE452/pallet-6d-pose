"""Independent stdlib scalar-loop audit of frozen V177 saved CSV/JSON only.
Does not import evaluator, NumPy, Torch, model, or feature helpers.
"""
import csv,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
D=Path(__file__).resolve().parent; V=D.parent; S=V/'saved_public_held_owner_evaluation_v1'
OWN=['TARGET_VISIBLE_SUPPORTED','FOREIGN_VISIBLE_SUPPORTED','BACKGROUND_SUPPORTED','UNKNOWN']
CROP=['YES','NO','UNKNOWN'];MODELS=['C47','C48','C49','v61']

def bind(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def load(p):return json.loads(p.read_text())
def ratio(n,d):return n/d if d else None

def metrics(rows,model,three):
    names=OWN[:3] if three else ['T','NON_T'];K=len(names)
    cm=[[0 for _ in range(K+1)] for _ in range(K+1)]
    for r in rows:
        truth=OWN.index(r['center_owner']);raw=r[model+'_prediction']
        if three:t=truth;p=K if raw==-1 else raw
        else:t=2 if truth==3 else (0 if truth==0 else 1);p=2 if raw==-1 else (0 if raw==0 else 1)
        cm[t][p]+=1
    known=0;unknown=0;up=0;up_known=0;classes={}
    for t in range(K+1):
        for p in range(K+1):
            n=cm[t][p]
            if t==K:unknown+=n
            else:known+=n
            if p==K:
                up+=n
                if t<K:up_known+=n
    for c,name in enumerate(names):
        support=0;predicted=0
        for p in range(K+1):support+=cm[c][p]
        for t in range(K):predicted+=cm[t][c]
        classes[name]={'TP':cm[c][c],'predicted_on_known_truth':predicted,'known_truth_support':support,'precision_on_known_truth':ratio(cm[c][c],predicted),'recall_including_prediction_U':ratio(cm[c][c],support)}
    return {'N':len(rows),'truth_class_order':names+['UNKNOWN'],'prediction_class_order':names+['U'],'full_confusion':cm,'known_truth_confusion':cm[:K],'known_truth':known,'unknown_truth':unknown,'prediction_U_all':up,'prediction_U_on_known_truth':up_known,'predictions_on_unknown_truth':cm[K],'per_class':classes,'precision_excludes_unknown_truth':True,'weak_reference_only':True}

leaves=0

def exact(a,b,path=''):
    global leaves
    assert type(a) is type(b),(path,type(a).__name__,type(b).__name__)
    if isinstance(a,dict):
        assert set(a)==set(b),(path,'keys',set(a)^set(b))
        for k in a:exact(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,list):
        assert len(a)==len(b),(path,'length')
        for i,(x,y) in enumerate(zip(a,b)):exact(x,y,path+'/'+str(i))
    else:assert a==b,(path,a,b);leaves+=1

def main():
    complete_path=S/'COMPLETE.json';complete=load(complete_path)
    assert bind(complete_path)['sha256']=='ad7e8d329a9dcffd136a481e978285840fd356b4134fb12d56c02aff5ba0dc24'
    assert complete['completed'] is True and complete['PASS'] is True
    paths={k:Path(v['path']) for k,v in complete['outputs'].items()}
    before={k:bind(p) for k,p in paths.items()}
    for k,b in before.items():assert b==complete['outputs'][k]
    consensus=[json.loads(x) for x in paths['consensus'].read_text().splitlines()]
    with paths['ledger'].open(newline='') as f:rows=list(csv.DictReader(f))
    report=load(paths['report'])
    assert len(rows)==len(consensus)==360
    assert [int(r['original_global_index']) for r in rows]==list(range(584,944))
    assert [r['original_global_index'] for r in consensus]==list(range(584,944))
    agree={name:[[0 for _ in order] for _ in order] for name,order in [('center_owner',OWN),('crop_presence',CROP)]}
    center_cross=[[0 for _ in CROP] for _ in OWN]
    model_cross={m:[[0 for _ in CROP] for _ in range(3 if m=='v61' else 4)] for m in MODELS}
    counts={'original_input_U':0,'center4_invalid':0,'input_available_center4_invalid':0,'no_constructed_crop':0,'consensus_center_known':0,'consensus_center_unknown':0,'consensus_center_unknown_beyond_forced112':0,'consensus_crop_known':0,'consensus_crop_unknown':0,'model_prediction_U':{m:0 for m in MODELS},'all4_model_paired_available':0}
    frames=set();native=[0]*8;forced=0
    for r,c in zip(rows,consensus):
        for key in ['original_global_index','frame_index','native_slot']:r[key]=int(r[key]);assert r[key]==c[key]
        for key in ['source_original_query_available','source_center4_allvalid','source_crop_constructed']:
            assert r[key] in ['True','False'];r[key]=(r[key]=='True');assert r[key] is c[key]
        assert r['query_id']==c['query_id']
        for m in MODELS:r[m+'_prediction']=int(r[m+'_prediction']);assert r[m+'_prediction'] in ([-1,0,1] if m=='v61' else [-1,0,1,2])
        a=c['reviewer_A'];b=c['reviewer_B'];fr=not c['source_original_query_available'] or not c['source_center4_allvalid']
        for rev in [a,b]:
            for key in ['original_global_index','frame_index','native_slot']:assert rev[key]==c[key]
            assert rev['center_owner'] in OWN and rev['crop_presence'] in CROP
            if fr:assert rev['center_owner']=='UNKNOWN'
            if not c['source_crop_constructed']:assert rev['crop_presence']=='UNKNOWN'
        center=a['center_owner'] if a['center_owner']==b['center_owner'] and not fr else 'UNKNOWN'
        crop=a['crop_presence'] if a['crop_presence']==b['crop_presence'] and c['source_crop_constructed'] else 'UNKNOWN'
        assert r['center_owner']==c['center_owner']==center
        assert r['crop_presence']==c['crop_presence']==crop
        assert c['training_admission'] is False and c['human_pixel_physical_native_gold'] is False
        for key,order in [('center_owner',OWN),('crop_presence',CROP)]:agree[key][order.index(a[key])][order.index(b[key])]+=1
        ci=OWN.index(center);cp=CROP.index(crop);center_cross[ci][cp]+=1
        for m in MODELS:
            pred=r[m+'_prediction'];p=(2 if m=='v61' else 3) if pred==-1 else pred;model_cross[m][p][cp]+=1
            if pred==-1:counts['model_prediction_U'][m]+=1
            if not r['source_original_query_available']:assert pred==-1
        if not r['source_original_query_available']:counts['original_input_U']+=1
        if not r['source_center4_allvalid']:counts['center4_invalid']+=1
        if r['source_original_query_available'] and not r['source_center4_allvalid']:counts['input_available_center4_invalid']+=1
        if not r['source_crop_constructed']:counts['no_constructed_crop']+=1
        if center=='UNKNOWN':
            counts['consensus_center_unknown']+=1
            if not fr:counts['consensus_center_unknown_beyond_forced112']+=1
        else:counts['consensus_center_known']+=1
        if crop=='UNKNOWN':counts['consensus_crop_unknown']+=1
        else:counts['consensus_crop_known']+=1
        if all(r[m+'_prediction']!=-1 for m in MODELS):counts['all4_model_paired_available']+=1
        forced+=int(fr);frames.add(r['frame_index']);native[r['native_slot']]+=1
    assert frames==set(range(73,118)) and native==[45]*8 and forced==112
    assert counts['consensus_center_known']==204 and counts['consensus_center_unknown']==156
    assert 360-counts['original_input_U']==252 and counts['all4_model_paired_available']==251
    exact(counts,report['coverage'],'coverage');exact(counts,complete['coverage'],'complete/coverage')
    agreement={}
    for key,order in [('center_owner',OWN),('crop_presence',CROP)]:
        table=agree[key];diag=0;known_diag=0
        for i in range(len(order)):
            diag+=table[i][i]
            if i<len(order)-1:known_diag+=table[i][i]
        agreement[key]={'class_order':order,'table':table,'exact_all':diag,'exact_known':known_diag,'disagreement':360-diag}
    exact(agreement,report['reviewer_agreement'],'reviewer_agreement')
    crop_associations={'consensus_center_vs_crop':{'center_order':OWN,'crop_order':CROP,'table':center_cross}}
    for m in MODELS:crop_associations[m]={'prediction_order':(['T','NON_T','U'] if m=='v61' else OWN[:3]+['U']),'crop_order':CROP,'table':model_cross[m],'association_only_not_crop_classifier':True}
    exact(crop_associations,report['crop_associations'],'crop_associations')
    rebuilt={};nmetric=0
    for cohort in ['FULL360','KNOWN_CENTER_TRUTH','ALL4_MODEL_PAIRED_AVAILABLE','KNOWN_TRUTH_AND_ALL4_PAIRED_AVAILABLE']:
        chosen=[]
        for r in rows:
            known=r['center_owner']!='UNKNOWN';paired=all(r[m+'_prediction']!=-1 for m in MODELS)
            if cohort=='FULL360' or (cohort=='KNOWN_CENTER_TRUTH' and known) or (cohort=='ALL4_MODEL_PAIRED_AVAILABLE' and paired) or (cohort=='KNOWN_TRUTH_AND_ALL4_PAIRED_AVAILABLE' and known and paired):chosen.append(r)
        groups={}
        for slot in ['ALL']+[str(n) for n in range(8)]:
            rr=[r for r in chosen if slot=='ALL' or r['native_slot']==int(slot)]
            g={'N':len(rr),'original_input_U':0,'center4_invalid':0,'no_constructed_crop':0,'models':{}}
            for r in rr:
                g['original_input_U']+=int(not r['source_original_query_available']);g['center4_invalid']+=int(not r['source_center4_allvalid']);g['no_constructed_crop']+=int(not r['source_crop_constructed'])
            for m in MODELS:
                g['models'][m]={'binary_T_nonT':metrics(rr,m,False)};nmetric+=1
                if m!='v61':g['models'][m]['owner3']=metrics(rr,m,True);nmetric+=1
            groups[slot]=g
        rebuilt[cohort]=groups
    exact(rebuilt,report['cohorts'],'cohorts');assert nmetric==252
    for k,p in paths.items():assert bind(p)==before[k]
    assert bind(complete_path)['sha256']=='ad7e8d329a9dcffd136a481e978285840fd356b4134fb12d56c02aff5ba0dc24'
    summary={m:{k:rebuilt['FULL360']['ALL']['models'][m][k] for k in rebuilt['FULL360']['ALL']['models'][m]} for m in MODELS}
    result={'schema':'V177_B_INDEPENDENT_SAVED_NUMERIC_REVIEW_v1','UTC':datetime.now(timezone.utc).isoformat(),'completed':True,'PASS':True,'method':'Independent Python stdlib scalar loops over saved CSV and JSONL; rebuilt all matrices and integer supports then direct division. No evaluator/helper import or model/feature/NPZ read. Exact structural equality including null denominators.','code':bind(Path(__file__)),'evaluation_complete':bind(complete_path),'saved_inputs':before,'rows':360,'frames':45,'cohorts':4,'native_groups_per_cohort':9,'metric_blocks_recomputed':252,'exact_compared_scalar_leaves':leaves,'coverage':counts,'model_available':{m:360-counts['model_prediction_U'][m] for m in MODELS},'checks':{'CSV_JSONL_ID_native_flags_labels_exact':True,'reviewer_A_B_consensus_and_forced_U_rebuilt':True,'all4_binary_and_3owner_confusions_PR_U_all36strata_exact':True,'all_crop_cross_association_tables_exact':True,'UNKNOWN_truth_excluded_from_precision_and_kept_in_full_CM':True,'all360_native8_and45frames_preserved':True,'all_saved_input_hashes_unchanged':True},'FULL360_ALL_numeric_excerpt':summary,'reviewer_agreement':agreement,'scope_limitations':['Saved-data numerical consistency audit, not evidence of physical owner/native/coordinate gold.','This reviewer created B weak observations and is independent of evaluator implementation, not an independent human truth annotator.','One previously exposed public video, 45 correlated frames/360 slots; not fresh independent full-model validation.','No source pixel/GT/native/feature/weight body was read in this audit; saved prediction derivation is inherited from bound evaluator COMPLETE, not re-forwarded or re-argmaxed here.'],'model_forward_fit_threshold_or_selection_calls':0,'observation_or_original_label_changes':0,'private_current_V112_access':False,'training_admission':False,'selected_model':None,'adopted':False,'fully_solved':False}
    out=D/'REVIEW.json'
    with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'PASS':True,'review':bind(out),'code':bind(Path(__file__)),'metric_blocks':nmetric,'scalar_leaves':leaves,'coverage':counts},ensure_ascii=False))

if __name__=='__main__':main()
