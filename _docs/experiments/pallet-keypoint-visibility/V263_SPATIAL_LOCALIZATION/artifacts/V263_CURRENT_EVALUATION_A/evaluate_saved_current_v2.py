"""Post-freeze, saved-output current evaluation. No model, encoder, fit or RGB I/O."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import sys
import numpy as np
sys.dont_write_bytecode = True

D='DIRECT_VISIBLE'; E='EXTERNAL_OCCLUDED'; S='SELF_OCCLUDED'
GROUPS=('WOOD_PRIMARY105','SQUARE118')
ARMS={'ORIGINAL':(False,False),'LOC_ONLY_INITIAL_GEOMETRY':(False,True),'VIS_ONLY_SELECTED':(True,False),'JOINT_SELECTED':(True,True)}
CELLS=('TRUE_D_TO_D','MANUAL_D_LE3_TO_E','TRUE_E_TO_E','TRUE_E_TO_D','MANUAL_D_LOC_RECOVERY','MANUAL_D_LOC_DAMAGE')

def need(v,m):
    if not bool(v): raise ValueError(m)

def safe(value,fresh=False):
    p=Path(value);need(p.is_absolute() and '..' not in p.parts,'Absolute nontraversal path')
    need(not any(s in str(p).casefold() for s in ('v112','cache','.pyc','source_holdout','private_evaluation_held')),'Forbidden before filesystem')
    for a in [*reversed(p.parents),p]:
        try: mode=os.lstat(a).st_mode
        except FileNotFoundError: continue
        need(not stat.S_ISLNK(mode),'No artifact/ancestor symlink')
    if fresh: need(not p.exists(),'Fresh immutable output; no overwrite/retry')
    return p

def identity(b):
    need(set(b)=={'path','sha256','size_bytes'},'Exact3 binding schema')
    safe(b['path']);need(len(b['sha256'])==64 and type(b['size_bytes'])is int,'SHA/size')
    return b['path'],b['sha256'],b['size_bytes']

def read(b,white):
    need(identity(b)in{identity(x)for x in white},'Explicit whitelist before body read')
    raw=safe(b['path']).read_bytes();need(len(raw)==b['size_bytes']and hashlib.sha256(raw).hexdigest()==b['sha256'],'Exact SHA/size')
    return raw

def binding(p):
    p=safe(p);raw=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),size_bytes=len(raw))

def put(p,x):
    with safe(p,True).open('x',encoding='utf8')as f:json.dump(x,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    return binding(p)

def csvwrite(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with safe(p,True).open('x',encoding='utf8',newline='')as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    return binding(p)

def number(x):
    try: v=float(x)
    except(TypeError,ValueError): return None
    return v if math.isfinite(v)else None

def point(x):
    if not isinstance(x,(list,tuple))or len(x)!=2:return None
    y=[number(v)for v in x];return y if all(v is not None for v in y)else None

def ratio(a,b):return a/b if b else None
def distance(a,b):return math.hypot(a[0]-b[0],a[1]-b[1])
def quantiles(values):
    return dict(N=len(values),median=float(np.quantile(values,.5)),p90=float(np.quantile(values,.9)),max=max(values))if values else dict(N=0,median=None,p90=None,max=None)

def cm(rows,pred):
    counts=Counter(('D'if r['human_state']==D else'E',pred[int(r['global_index'])])for r in rows)
    nd=sum(r['human_state']==D for r in rows);ne=len(rows)-nd
    dd,de,du,ed,ee,eu=[counts[x]for x in [('D','D'),('D','E'),('D','U'),('E','D'),('E','E'),('E','U')]]
    return dict(N=len(rows),truth_D=nd,truth_E=ne,D_D=dd,D_E=de,D_U=du,E_D=ed,E_E=ee,E_U=eu,
        D_precision=ratio(dd,dd+ed),D_recall=ratio(dd,nd),E_precision=ratio(ee,ee+de),E_recall=ratio(ee,ne),
        U=du+eu,U_rate=ratio(du+eu,len(rows)),accuracy_including_U_as_unresolved=ratio(dd+ee,len(rows)),
        balanced_recall=(dd/nd+ee/ne)/2 if nd and ne else None)

def manual(r):return r['human_state']==D and r['annotation_coordinate_source']=='manual_click'
def original_available(r):return r['original_query_valid']=='True'
def reference(r):return point([r['human_reference_x'],r['human_reference_y']])
def query(r):return point([r['query_x'],r['query_y']])

def location_rows(rows,predictions,model):
    scored=[];unscorable=Counter()
    for r in rows:
        i=int(r['global_index']);q=query(r);ref=reference(r)
        reason='ORIGINAL_INPUT_U'if not original_available(r)else'NONFINITE_OR_ABSENT_Q'if q is None else'NONFINITE_OR_ABSENT_REFERENCE'if ref is None else None
        if reason:unscorable[reason]+=1;continue
        p=predictions[model][i]if model!='ORIGINAL_QUERY'else dict(accepted=False,final_xy=q,LOC_proposed_xy=None)
        final=point(p['final_xy']);proposal=point(p['LOC_proposed_xy'])
        need(final is not None,'Finite final point for scorable original query')
        initial=distance(q,ref);after=distance(final,ref);old=number(r['initial_reference_error_px'])
        scored.append(dict(global_index=i,row_id=r['row_id'],model=model,initial_error_px=initial,final_error_px=after,
            proposed_error_px=distance(proposal,ref)if proposal is not None else None,accepted=p['accepted'],initial_LE3=initial<=3,final_LE3=after<=3,
            recovered=initial>3 and after<=3,damaged=initial<=3 and after>3,stored_initial_error=old,
            stored_recomputed_absdiff=abs(old-initial)if old is not None else None))
    return scored,unscorable

def location_metrics(rows,predictions,model):
    scored,unscorable=location_rows(rows,predictions,model);accepted=[r for r in scored if r['accepted']]
    return dict(N=len(rows),scorable_N=len(scored),unscorable_N=len(rows)-len(scored),unscorable_reasons=dict(unscorable),
        accepted_N=len(accepted),accepted_rate_of_scorable=ratio(len(accepted),len(scored)),
        accepted_accuracy_LE3=ratio(sum(r['final_LE3']for r in accepted),len(accepted)),
        all_scorable_accuracy_LE3=ratio(sum(r['final_LE3']for r in scored),len(scored)),
        initial_accuracy_LE3=ratio(sum(r['initial_LE3']for r in scored),len(scored)),
        recovery_at3px=sum(r['recovered']for r in scored),damage_at3px=sum(r['damaged']for r in scored),
        final_error_px=quantiles([r['final_error_px']for r in scored]),accepted_final_error_px=quantiles([r['final_error_px']for r in accepted]),
        original_error_px=quantiles([r['initial_error_px']for r in scored]),
        proposed_error_px=quantiles([r['proposed_error_px']for r in scored if r['proposed_error_px']is not None]),
        stored_initial_error_recompute_maxabs=max((r['stored_recomputed_absdiff']for r in scored if r['stored_recomputed_absdiff']is not None),default=None))

def normalize(rows,arm,fields):
    need(len(rows)==1784,'All1784 output rows retained');result=[]
    for i,r in enumerate(rows):
        def get(k):return r[fields[k]]
        q=point(get('original_q0'));f=point(get('final_xy'));proposed=point(get('LOC_proposed_xy'))
        index=int(get('query_index'));native=int(get('native_slot'));available=get('available');accepted=get('accepted')
        need(index==i and native==i%8 and type(available)is bool and type(accepted)is bool,'Fixed index/native/boolean schema')
        pred=get('semantic_VIS_prediction');logit=number(get('visibility_E_logit'))
        if ARMS[arm][0]:need(pred in('D','E','U')and(pred=='U'or logit is not None),'Learned semantic VIS namespace/finite logit')
        else:need(pred in('N_A','N/A',None)and logit is None,'LOC-only classification N/A, not a visibility classifier')
        if not available:
            need(not accepted,'Original input U cannot accept a correction')
            if ARMS[arm][0]:need(pred=='U','Original unavailable query remains semantic U')
        if accepted:need(ARMS[arm][1]and proposed is not None and f==proposed,'Accepted trained proposal becomes final point')
        else:need(f==q,'Rejected/untrained location retains original q')
        result.append(dict(query_index=i,native_slot=native,frame_id=get('frame_id'),original_q0=q,L=number(get('L')),available=available,
            semantic_VIS_prediction=pred,visibility_E_logit=logit,LOC_proposed_xy=proposed,accepted=accepted,final_xy=f,
            location_state=get('location_state'),crop_L=number(get('crop_L'))))
    return result

def evaluate(ledger,predictions,old_predictions,paired_flags):
    need(len(ledger)==len(paired_flags)==1784,'Original full ledger/flags1784')
    for i,r in enumerate(ledger):
        need(int(r['global_index'])==i and int(r['frame_index'])==i//8 and int(r['native_corner'])==i%8 and r['row_id']==r['id']+':native'+str(i%8),'Original exact row/native identity')
        for arm,rows in predictions.items():
            p=rows[i];need(p['frame_id']==r['id']and p['native_slot']==i%8 and p['available']==original_available(r),'No frame/native/U remapping')
            need(p['original_q0']==query(r)and p['L']==number(r['roi_side']),'Original q/L unchanged')
    primary=[r for r in ledger if r['paired_same_query_RGB_D_vs_E']=='True'and r['human_state']in(D,E)]
    near=[r for r in ledger if manual(r)and original_available(r)and number(r['initial_reference_error_px'])is not None and number(r['initial_reference_error_px'])<=3]
    need((len(primary),sum(r['human_state']==D for r in primary),sum(r['human_state']==E for r in primary))==(1237,1147,90),'Frozen PRIMARY1237 D1147/E90; no coordinate/availability filter')
    need(len(near)==376,'Frozen manual-click D valid-input error<=3 subset376')
    need(sum(r['human_state']==S for r in ledger)==136 and sum(not original_available(r)for r in ledger)==45,'Original SELF136 and inputU45')
    classes={'v61':[r['v61_D_E_U']for r in ledger],'CAL_LEGACY':[r['legacy_D_E_U']for r in ledger]}
    for name,rows in old_predictions.items():
        need(len(rows)==1784,'Frozen V261 predictions1784')
        for i,(p,r)in enumerate(zip(rows,ledger)):
            need(p['original_global_index']==i and p['row_id']==r['row_id']and p['id']==r['id']and p['prediction']in('D','E','U'),'V261 exact saved row join')
        classes[name]=[p['prediction']for p in rows]
    classes.update({a:[r['semantic_VIS_prediction']for r in rows]for a,rows in predictions.items()if ARMS[a][0]})
    cohorts={'PRIMARY':primary,'ALL_REGISTERED_D_E':[r for r in ledger if r['human_state']in(D,E)],'MANUAL_CLICK_D_ALL':[r for r in ledger if manual(r)],'MANUAL_CLICK_D_LE3_VALID_INPUT':near}
    classification=[];location=[];self_tables=[];recovery=[];point_details=[]
    for group in('ALL',*GROUPS):
        for cohort,rr in cohorts.items():
            take=[r for r in rr if group=='ALL'or r['group']==group]
            for name,pp in classes.items():classification.append(dict(group=group,cohort=cohort,model=name,**cm(take,pp)))
            for a in ARMS:
                if not ARMS[a][0]:classification.append(dict(group=group,cohort=cohort,model=a,N=len(take),status='N_A_UNTRAINED_SEMANTIC_VIS'))
        for cohort in('MANUAL_CLICK_D_ALL','MANUAL_CLICK_D_LE3_VALID_INPUT'):
            take=[r for r in cohorts[cohort]if group=='ALL'or r['group']==group]
            for a in('ORIGINAL_QUERY',*[a for a in ARMS if ARMS[a][1]]):
                location.append(dict(group=group,cohort=cohort,model=a,**location_metrics(take,predictions,a)))
                if group=='ALL'and cohort=='MANUAL_CLICK_D_ALL':point_details.extend(location_rows(take,predictions,a)[0])
        for cohort in('ALL_REGISTERED','PAIRED_SAME_QUERY'):
            rr=[r for i,r in enumerate(ledger)if r['human_state']==S and(group=='ALL'or r['group']==group)and(cohort=='ALL_REGISTERED'or paired_flags[i])]
            for a,pp in classes.items():self_tables.append(dict(group=group,cohort=cohort,model=a,N=len(rr),responses=dict(Counter(pp[int(r['global_index'])]for r in rr)),interpretation='Binary response on humanSELF; no learnedSELF correctness'))
        pp=[r for r in primary if group=='ALL'or r['group']==group]
        for new in('VIS_ONLY_SELECTED','JOINT_SELECTED'):
            for base in('v61','CAL_LEGACY',*old_predictions):
                recover=damage=0
                for r in pp:
                    i=int(r['global_index']);truth='D'if r['human_state']==D else'E'
                    recover+=classes[base][i]!=truth and classes[new][i]==truth;damage+=classes[base][i]==truth and classes[new][i]!=truth
                recovery.append(dict(group=group,baseline=base,new=new,N=len(pp),recovered=recover,damaged=damage,net=recover-damage))
    joint=predictions['JOINT_SELECTED'];selection=[]
    for group in GROUPS:
        for cell in CELLS:
            eligible=[]
            for r in ledger:
                if r['group']!=group:continue
                i=int(r['global_index']);p=joint[i];truth=r['human_state'];pred=p['semantic_VIS_prediction'];old=number(r['initial_reference_error_px'])
                q=query(r);ref=reference(r);final=point(p['final_xy']);scorable=manual(r)and original_available(r)and q is not None and ref is not None and final is not None
                a=distance(q,ref)if scorable else None;b=distance(final,ref)if scorable else None
                same_query_primary=r['paired_same_query_RGB_D_vs_E']=='True'and truth in(D,E)
                yes={'TRUE_D_TO_D':same_query_primary and truth==D and pred=='D','MANUAL_D_LE3_TO_E':manual(r)and original_available(r)and old is not None and old<=3 and pred=='E',
                    'TRUE_E_TO_E':same_query_primary and truth==E and pred=='E','TRUE_E_TO_D':same_query_primary and truth==E and pred=='D','MANUAL_D_LOC_RECOVERY':scorable and a>3 and b<=3,'MANUAL_D_LOC_DAMAGE':scorable and a<=3 and b>3}[cell]
                if yes:eligible.append(i)
            selection.append(dict(group=group,cell=cell,census_N=len(eligible),empty=not eligible,selected_global_index=min(eligible)if eligible else None))
    return dict(classification=classification,location=location,human_SELF=self_tables,classification_recovery=recovery,point_details=point_details,selection=selection,
        counts=dict(all1784=1784,primary1237=1237,primaryD=1147,primaryE=90,manual_D_LE3_valid_input376=376,manual_D_all=len(cohorts['MANUAL_CLICK_D_ALL']),inputU45=45,humanSELF136=136),
        interpretation=['Repeated exposed current evaluation; no model/threshold/crop/acceptance selection','LOC-only classification N/A; VIS-only location untrained, original query baseline separate','Existing manual-click coordinates are evaluation reference, not new physical native gold','HumanSELF separate; no training or learnedSELF inference','Result-based12-case diagnostic: full cell census and min original index, empty preserved; not training/configuration selection'])

def contains_binding(tree,b):
    if isinstance(tree,dict):
        if all(tree.get(k)==v for k,v in b.items()):return True
        return any(contains_binding(x,b)for x in tree.values())
    if isinstance(tree,list):return any(contains_binding(x,b)for x in tree)
    return False

def run(entry_path):
    ep=safe(entry_path);e=json.loads(ep.read_bytes());need(e.get('approved_actual_current_evaluation')is True,'Root true entry required')
    scope=json.loads(read(e['root_scope'],[e['root_scope']]))
    need(scope.get('approved_actual_current_evaluation')is True and scope['entry_contract']=={k:v for k,v in e.items()if k!='root_scope'},'Exact Root read scope')
    need(e['code']==binding(Path(__file__).absolute())and e['fit_allowed']is False and e['automatic_retry']is False,'Selfpin/no fit/retry')
    out=safe(e['output_directory'],True);out.mkdir()
    try:
        early=e['early_metadata_whitelist'];freeze=json.loads(read(e['ALL4_FINAL_FREEZE'],early));complete=json.loads(read(e['current_output_complete'],early))
        pf=json.loads(read(e['current_prediction_freeze'],early))
        need(freeze.get('completed')is True and complete.get('completed')is True and pf.get('completed')is True,'All4 final/config/current predictions frozen before late GT')
        need(contains_binding(complete,e['ALL4_FINAL_FREEZE'])and contains_binding(complete,e['current_prediction_freeze']),'Current output closure binds both freezes')
        need(set(e['predictions'])==set(ARMS),'Exact4 arms')
        predictions={}
        for a,b in e['predictions'].items():
            need(contains_binding(complete,b)and contains_binding(pf,b),'Each full prediction bound before late evaluation')
            raw=[json.loads(x)for x in read(b,e['prediction_whitelist']).splitlines()if x]
            predictions[a]=normalize(raw,a,e['prediction_fields'])
        seal=put(out/'ALL4_CONSUMER_PREDICTIONS_CHECKED_BEFORE_LATE_GT.json',dict(completed=True,PASS=True,ALL4_FINAL_FREEZE=e['ALL4_FINAL_FREEZE'],prediction_freeze=e['current_prediction_freeze'],predictions=e['predictions'],late_GT_or_baseline_body_reads=0))
        late=e['late_evaluation_whitelist']
        ledger=list(csv.DictReader(io.StringIO(read(e['evaluation_ledger'],late).decode('utf-8-sig'))))
        flags=list(csv.DictReader(io.StringIO(read(e['cohort_flags'],late).decode('utf-8-sig'))))
        need(len(flags)==1784 and all(a['id']==b['id']and a['native_corner']==b['native_corner']and a['human_state']==b['human_state']for a,b in zip(ledger,flags)),'Original ordered flags/ledger identity; no GT matching')
        paired=[r['paired_same_query_RGB_D_vs_ES']=='True'for r in flags]
        old={a:[json.loads(x)for x in read(b,late).splitlines()if x]for a,b in e['V261_saved_predictions'].items()}
        result=evaluate(ledger,predictions,old,paired)
        tables={}
        for name in('classification','location','human_SELF','classification_recovery','point_details'):
            tables[name]=csvwrite(out/(name.upper()+'.csv'),[{k:json.dumps(v,ensure_ascii=False,sort_keys=True)if isinstance(v,(dict,list))else v for k,v in r.items()}for r in result[name]])
        selected=put(out/'CASE_IDS_BEFORE_PIXEL_READ.json',dict(completed=True,policy=e['fixed12_policy'],cells=result['selection'],selected_global_indices=sorted({r['selected_global_index']for r in result['selection']if not r['empty']}),pixel_reads_before_selection=0,no_empty_cell_replacement=True))
        tables['selection']=selected
        joined=[]
        for i,r in enumerate(ledger):
            z=dict(r)
            for a in ARMS:
                for k,v in predictions[a][i].items():z[a+'__'+k]=json.dumps(v)if isinstance(v,(dict,list))else v
            joined.append(z)
        tables['joined_rows']=csvwrite(out/'ALL1784_CLASSIFICATION_LOCATION_JOIN.csv',joined)
        summary=put(out/'SUMMARY.json',dict(completed=True,PASS=True,counts=result['counts'],interpretation=result['interpretation'],classification=result['classification'],location=result['location'],human_SELF=result['human_SELF'],classification_recovery=result['classification_recovery'],fixed12_census=result['selection'],threshold_changed=False,model_selected=False,location_distance_threshold_px=3,quantile_method='linear',consumer_pre_GT_seal=seal))
        cb=put(out/'COMPLETE.json',dict(completed=True,PASS=True,code=e['code'],entry=binding(ep),ALL4_FINAL_FREEZE=e['ALL4_FINAL_FREEZE'],current_output_complete=e['current_output_complete'],prediction_freeze=e['current_prediction_freeze'],predictions=e['predictions'],consumer_pre_GT_seal=seal,summary=summary,outputs=tables,fit_or_model_or_encoder_or_RGB_reads=0,model_threshold_query_native_or_reference_changes=0,fresh_blind_evaluation=False))
        print(json.dumps(cb),flush=True)
    except Exception as ex:
        put(out/'FAILURE.json',dict(completed=False,PASS=False,error_type=type(ex).__name__,error=str(ex),partial_preserved=True,automatic_retry=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entry',required=True);run(p.parse_args().entry)
