"""Saved source weak-end scalar evaluation. NumPy only; no models or forwards.

Actual I/O requires separate root entry after BOTH final4000 states freeze.
Immutable component TRAIN/HOLDOUT roles remain distinct from nominal fit masks.
"""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,csv,hashlib,json,os,stat,traceback
import numpy as np
H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
PROTOCOL=H/'PROTOCOL_BEFORE_SAVED_SOURCE_OUTPUT_OR_TARGET_BODIES.json'
OUT=V/'saved_source_direct_end_evaluation_v2_PARTIAL_UNKNOWN_PRESERVED'
FIT=V/'runs/C60_C61_SPATIAL_Q8_NOMINAL_SOURCE_FIXED4000/COMPLETE.json'
ARMS=('C60_RAW96_NATIVE_MASK_ZERO_XY_NOMINAL_CONTROL','C61_RAW96_NATIVE_MASK_Q8_XY_NOMINAL')
META=('row_id','original_global_index','native_slot','query_xy','L','original_query_available')
ROWS=META+('raw_support_logits','raw_owner_logits','raw_direct_end_external_logit','support_prediction','owner_prediction','direct_end_prediction_D0_E1_Uminus1','regional_support_proxy_D0_E1_Uminus1')
ORDERS=('support_order','owner_order','direct_end_order')

def need(v,m):
    if not bool(v):raise AssertionError(m)

def safe(path):
    p=Path(path);need(p.is_absolute()and'..'not in p.parts and all('v112'not in q.casefold()for q in p.parts),'Explicit lexical safe path before access')
    need(p.is_relative_to(P),'Only exact source project paths')
    need(not any(q in str(p).casefold()for q in ('current223','_v177/','_v178/','_v184/','_v191/','_v195/','_v200/')),'No protected real/eval phase')
    return p

def opened(path):
    p=safe(path);fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for q in p.parts[1:-1]:
            nxt=os.open(q,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nxt
        leaf=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    need(stat.S_ISREG(os.fstat(leaf).st_mode),'Only ordinary files')
    return os.fdopen(leaf,'rb')

def sha(path):
    h=hashlib.sha256()
    with opened(path)as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def bind(p):return dict(path=str(safe(p)),sha256=sha(p))
def checked(b):need(sha(b['path'])==b['sha256'],'Exact bound bytes');return b['path']
def read(p):
    with opened(p)as f:return json.load(f)
def read_bound(b):return read(checked(b))
def lines(b):
    with opened(checked(b))as f:return [json.loads(x)for x in f if x.strip()]
def arrays(b,keys):
    with opened(b['path'])as f:
        h=hashlib.sha256()
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
        need(h.hexdigest()==b['sha256'],'Saved NPZ exact SHA');f.seek(0)
        with np.load(f,allow_pickle=False)as z:
            need(set(z.files)==set(keys),'Exact closed saved keys')
            return {k:z[k]for k in keys}
def write(p,x):
    p=safe(p)
    with p.open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def equal(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and (np.array_equal(a,b,equal_nan=True)if a.dtype.kind in'fc'else np.array_equal(a,b))
def numeric(value,q=False):return np.asarray([np.nan,np.nan]if q and value is None else np.nan if value is None else value,np.float64)

def assert_meta(record,packet,i):
    for k in META:
        if k in ('query_xy','L'):need(np.array_equal(numeric(record[k],k=='query_xy'),np.asarray(packet[k][i]),equal_nan=True),'Original F64 q/L exact')
        else:need(record[k]==packet[k][i],'Original META6 exact '+k)

def join_roles(metadata,split_rows,targets):
    bykey={tuple(r['control_join_key']):r for r in split_rows}
    need(len(bykey)==len(split_rows)==len(metadata)==len(targets),'Exact all source control keys')
    aligned=[];used=set()
    for i,(m,target)in enumerate(zip(metadata,targets)):
        key=tuple(m['control_join_key']);need(key==(m['domain'],m['attempt_id'],m['variant'],m['native_slot'])and key in bykey and key not in used,'Exact canonical control key, no index alias');used.add(key);s=bykey[key]
        for k in META:
            if k in ('query_xy','L'):need(np.array_equal(numeric(m[k],k=='query_xy'),numeric(s[k],k=='query_xy'),equal_nan=True),'Split original q/L')
            else:need(m[k]==s[k],'Split original identity '+k)
        need(s['role']in('TRAIN','HOLDOUT')and m['cache_index']==i,'Immutable component role and current cache order')
        need(m['source_packet']==s['source_packet']and m['authority']==s['authority']and int(target)==s['stored_weak_direct_target'],'Original target/authority/source packet')
        need(target<0 or m['root_admitted']is True,'Only originally admitted weak target')
        aligned.append(dict(s,source_split_cache_index=s['cache_index'],cache_index=i))
    need(used==set(bykey),'No extra or missing source rows');return aligned

def metric(target,pred,available,indices):
    cm=[[0,0,0],[0,0,0]];unknown=Counter();input_u=0;known=0;unknown_n=0
    for i in indices:
        y=int(target[i]);p=int(pred[i]);av=bool(available[i]);need(y in(-1,0,1)and p in(-1,0,1),'D/E/U only')
        need(av or(y==p==-1),'InputU remains UNKNOWN')
        if not av:input_u+=1
        if y<0:unknown_n+=1;unknown[str(p)]+=1
        else:known+=1;cm[y][2 if p<0 else p]+=1
    divide=lambda a,b:a/b if b else None
    D=sum(cm[0]);E=sum(cm[1]);U=cm[0][2]+cm[1][2]
    return dict(rows=len(indices),known=known,D=D,E=E,target_UNKNOWN=unknown_n,original_inputU=input_u,available_target_UNKNOWN=unknown_n-input_u,CM_D_E_by_D_E_U=cm,D_precision=divide(cm[0][0],cm[0][0]+cm[1][0]),E_precision=divide(cm[1][1],cm[0][1]+cm[1][1]),D_recall=divide(cm[0][0],D),E_recall=divide(cm[1][1],E),known_prediction_U=U,known_prediction_U_rate=divide(U,known),UNKNOWN_target_prediction_counts={k:unknown[k]for k in('-1','0','1')})

def paired(target,predictions,indices):
    counts=Counter()
    for i in indices:
        if target[i]<0:continue
        a=int(predictions[ARMS[0]][i])==int(target[i]);b=int(predictions[ARMS[1]][i])==int(target[i]);counts[(a,b)]+=1
    return dict(known=sum(counts.values()),both_correct=counts[True,True],both_wrong=counts[False,False],recovered_by_XY=counts[False,True],damaged_by_XY=counts[True,False],net_correct_change=counts[False,True]-counts[True,False])

def triplets(metadata,roles,targets,predictions):
    groups=defaultdict(dict)
    for i,m in enumerate(metadata):
        key=(m['domain'],m['attempt_id'],int(m['native_slot']));need(m['variant']in('BEFORE','COVER','SHAM')and m['variant']not in groups[key],'Original native-attempt variant membership unique');groups[key][m['variant']]=i
    all_rows=[]
    for key,rr in sorted(groups.items()):
        present=[v for v in ('BEFORE','COVER','SHAM')if v in rr];missing=[v for v in ('BEFORE','COVER','SHAM')if v not in rr];ii=[rr[v]for v in present]
        need(len({roles[i]['role']for i in ii})==1 and len({roles[i]['component_id']for i in ii})==1,'One original component role for all present variants')
        yy=[int(targets[i])for i in ii];known=any(y>=0 for y in yy)
        if known:need(not missing and yy==[0,1,0],'Any known group requires whole existing D/E/D weak triplet')
        else:need(all(y==-1 for y in yy),'All UNKNOWN present targets retained')
        record=dict(domain=key[0],attempt_id=key[1],native_slot=key[2],role=roles[ii[0]]['role'],indices=ii,present_variants=present,missing_variants=missing,complete=not missing,known=known,targets=yy,predictions={a:[int(predictions[a][i])for i in ii]for a in ARMS})
        record['all3_correct']={a:known and record['predictions'][a]==[0,1,0]for a in ARMS};all_rows.append(record)
    summaries={}
    for cohort in ('FULL','TRAIN','HOLDOUT'):
        take=[r for r in all_rows if cohort=='FULL'or r['role']==cohort];known=[r for r in take if r['known']];paired_counts=Counter((r['all3_correct'][ARMS[0]],r['all3_correct'][ARMS[1]])for r in known)
        summaries[cohort]=dict(physical_all_native_attempt_groups=len(take),physical_present_variant_rows=sum(len(r['indices'])for r in take),complete_native_triplets=sum(r['complete']for r in take),partial_UNKNOWN_native_attempt_groups=sum(not r['complete']for r in take),known_D_E_D_triplets=len(known),UNKNOWN_groups=len(take)-len(known),missing_variant_counts={v:sum(v in r['missing_variants']for r in take)for v in ('BEFORE','COVER','SHAM')},models={a:dict(all3_correct=sum(r['all3_correct'][a]for r in known),all3_accuracy=sum(r['all3_correct'][a]for r in known)/len(known)if known else None)for a in ARMS},paired=dict(both_correct=paired_counts[True,True],both_wrong=paired_counts[False,False],recovered_by_XY=paired_counts[False,True],damaged_by_XY=paired_counts[True,False],net_correct_change=paired_counts[False,True]-paired_counts[True,False]))
    return summaries,all_rows


def fresh(path):
    p=safe(path);need(p==OUT,'One fixed fresh saved evaluation directory')
    current=Path(p.anchor)
    for q in p.parts[1:-1]:
        current/=q;need(stat.S_ISDIR(current.lstat().st_mode),'No output symlinks')
    p.mkdir(exist_ok=False);return p

def evaluate(entry_path):
    e=read(entry_path);need(e['approved_saved_source_direct_evaluation']is True and e['source_only']is True and e['automatic_retry']is False,'Separate approved post-barrier saved-only entry')
    p=read_bound(e['protocol']);need(e['code']==bind(__file__)==p['code']and e['protocol']==bind(PROTOCOL),'Frozen evaluator/protocol')
    tests=read_bound(p['checks']);need(tests['completed']is True and tests['PASS']is True and tests['code']==e['code'],'Completed meaningful invented checks')
    read_bound(e['root_scope']);need(e['fit_complete']['path']==str(FIT),'Exact source fit COMPLETE')
    fit=read_bound(e['fit_complete']);need(fit['completed']is True and fit['PASS']is True and fit['source_only']is True and fit['from_scratch']is True and fit['parameters_each']==118359 and fit['steps_each']==4000 and fit['total_updates']==8000 and fit['all2_frozen_before_any_TRAIN_or_HOLDOUT_outputs']is True,'BOTH final4000 closed before source output/target bodies')
    need(fit['protocol']==p['fit_protocol']and fit['code']==p['fit_code']and set(fit['arms'])==set(ARMS),'Exact predeclared final source arms')
    barrier=read_bound(fit['all2_freeze']);need(barrier['completed']is True and barrier['checkpoints']=={a:fit['arms'][a]['checkpoint']for a in ARMS}and barrier['source_TRAIN_or_HOLDOUT_prediction_emitted']is False and barrier['checkpoint_or_arm_selection']is False,'Exact all2 freeze and no selection')
    need(fit['training_caches']['direct']==p['direct_cache_complete']and fit['source_split']==p['split_complete'],'Original source direct cache and component split')
    for j,a in enumerate(ARMS):need(fit['arms'][a]['updates']==4000 and fit['arms'][a]['use_sam']is False and fit['arms'][a]['spatial_q8']is bool(j)and fit['arms'][a]['initial_state_sha']==fit['same_initial_state_sha']and fit['arms'][a]['old_draw_sha']==fit['same_filtered_old192_draw_sha']and fit['arms'][a]['direct_draw_sha']==fit['same_direct32_draw_sha'],'Matched fresh initialization/draws and RAW spatial flags')
    # First actual output/target opens occur only after every all2 guard above.
    done=read_bound(p['direct_cache_complete']);split=read_bound(p['split_complete']);need(done['completed']is True and done['PASS']is True and done['domain']=='DIRECT_CONTROL'and done['rows']==3144 and split['completed']is True and split['PASS']is True,'Original all-row completed source inputs')
    target_packet=arrays(done['targets'],META+('direct_end_external_target','authority'));target=target_packet['direct_end_external_target'];av=target_packet['original_query_available'];need(target.dtype==np.int8 and target.shape==(3144,)and np.isin(target,[-1,0,1]).all()and av.dtype==bool and av.shape==(3144,),'Original weak target/availability schema')
    need([int((target==v).sum())for v in(0,1)]==[248,124]and int((~av).sum())==112 and (target[~av]==-1).all(),'Full372 weak labels and originalU112 unchanged')
    metadata=lines(done['metadata']);need(len(metadata)==3144,'All source physical rows')
    for i,m in enumerate(metadata):assert_meta(m,target_packet,i);need(m['authority']==str(target_packet['authority'][i]),'Distinct original weak authority')
    roles=join_roles(metadata,lines(split['outputs']['DIRECT_CONTROL_SPLIT_ROWS.jsonl']),target);predictions={};fulls={}
    for arm in ARMS:
        full=arrays(fit['arms'][arm]['outputs']['DIRECT_CONTROL'],ROWS+ORDERS);fulls[arm]=full
        for k in META:need(equal(full[k],target_packet[k]),'Full saved prediction exact original META6 '+k)
        raw=full['raw_direct_end_external_logit'];pred=full['direct_end_prediction_D0_E1_Uminus1'];need(raw.dtype==np.float32 and raw.shape==(3144,)and pred.dtype==np.int8 and pred.shape==(3144,)and np.isfinite(raw[av]).all()and np.isnan(raw[~av]).all(),'Saved direct scalar schema/U')
        expected=np.full(3144,-1,np.int8);expected[av]=(raw[av]>=0).astype(np.int8);need(equal(pred,expected),'Frozen raw>=0 E else D; inputU retained');predictions[arm]=pred
        for cohort in ('TRAIN','HOLDOUT'):
            ii=np.asarray([i for i,r in enumerate(roles)if r['role']==cohort],np.int64);subset=arrays(fit['arms'][arm]['source_cohorts']['DIRECT_CONTROL'][cohort]['outputs'],ROWS+ORDERS)
            for k in ROWS:need(equal(subset[k],full[k][ii]),'Original component cohort exact subset '+k)
            for k in ORDERS:need(equal(subset[k],full[k]),'Original output order unchanged')
    summary={};csvrows=[]
    for cohort in ('FULL','TRAIN','HOLDOUT'):
        ii=[i for i,r in enumerate(roles)if cohort=='FULL'or r['role']==cohort]
        def cell(jj):return dict(models={a:metric(target,predictions[a],av,jj)for a in ARMS},paired_correctness=paired(target,predictions, jj))
        summary[cohort]=dict(all=cell(ii),by_domain={d:cell([i for i in ii if metadata[i]['domain']==d])for d in sorted({m['domain']for m in metadata})},by_native={str(k):cell([i for i in ii if metadata[i]['native_slot']==k])for k in range(8)},by_authority={a:cell([i for i in ii if str(target_packet['authority'][i])==a])for a in sorted(set(target_packet['authority'].tolist()))})
        for stratum,record in [('ALL',summary[cohort]['all'])]+[('DOMAIN:'+d,r)for d,r in summary[cohort]['by_domain'].items()]+[('NATIVE:'+k,r)for k,r in summary[cohort]['by_native'].items()]:
            for arm,m in record['models'].items():csvrows.append(dict(cohort=cohort,stratum=stratum,arm=arm,**{k:v for k,v in m.items()if k not in('CM_D_E_by_D_E_U','UNKNOWN_target_prediction_counts')},CM_D_E_by_D_E_U=json.dumps(m['CM_D_E_by_D_E_U'])))
    for cohort,counts in [('FULL',[248,124]),('TRAIN',[218,109]),('HOLDOUT',[30,15])]:need([summary[cohort]['all']['models'][ARMS[0]][k]for k in('D','E')]==counts,'Fixed known weak source denominator')
    need(all('SYNTH'in metadata[i]['domain']for i,r in enumerate(roles)if r['role']=='HOLDOUT'and target[i]>=0),'HOLD45 known synthetic only; real known0')
    triplet_summary,triplet_rows=triplets(metadata,roles,target,predictions)
    need([triplet_summary[c]['known_D_E_D_triplets']for c in('FULL','TRAIN','HOLDOUT')]==[124,109,15],'Fixed known weak triplet denominators')
    need(sum(len(r['indices'])for r in triplet_rows)==len(metadata),'Every physical source row retained exactly once, no fabricated variants')
    out=fresh(e['output_directory'])
    try:
        readout=out/'SOURCE_DIRECT_WEAK_METRICS.json';write(readout,dict(completed=True,PASS=True,source_only=True,cohorts=summary,triplets=triplet_summary,CM_orientation='rows weakD0,E1; columns predictedD0,E1,Uminus1',source_fit_complete=e['fit_complete'],weak_finite_end_not_nativephysicalgold=True,source_HOLDOUT_previously_exposed_not_freshblind=True,HOLDOUT_known_real_end_labels=0,original_component_roles_not_nominal_fit_complement=True,unknown_target_rows_never_scored_as_negative=True,U0_not_uncertainty_solved=True,no_threshold_arm_checkpoint_or_data_selection=True,adopted=False,fully_solved=False))
        table=out/'SOURCE_DIRECT_WEAK_CONFUSIONS_AND_STRATA.csv'
        with table.open('x',newline='')as f:
            w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
        ledger=out/'ALL3144_SOURCE_DIRECT_ROWS.jsonl'
        with ledger.open('x')as f:
            for i,m in enumerate(metadata):
                r={k:m[k]for k in META};r.update(cache_index=i,control_join_key=m['control_join_key'],domain=m['domain'],attempt_id=m['attempt_id'],variant=m['variant'],component_id=roles[i]['component_id'],role=roles[i]['role'],direct_weak_target=int(target[i]),authority=str(target_packet['authority'][i]),models={a:dict(raw_direct_end_external_logit=float(fulls[a]['raw_direct_end_external_logit'][i])if av[i]else None,prediction=int(predictions[a][i]),correct=bool(predictions[a][i]==target[i])if target[i]>=0 else None)for a in ARMS});f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
        triplet_path=out/'ALL_PHYSICAL_NATIVE_ATTEMPT_VARIANT_STATUSES.jsonl'
        with triplet_path.open('x')as f:
            for r in triplet_rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
        for b in [e['fit_complete'],fit['all2_freeze'],p['direct_cache_complete'],p['split_complete'],done['targets'],done['metadata'],split['outputs']['DIRECT_CONTROL_SPLIT_ROWS.jsonl'],e['protocol'],e['code']]:checked(b)
        for a in ARMS:
            checked(fit['arms'][a]['outputs']['DIRECT_CONTROL'])
            for cohort in('TRAIN','HOLDOUT'):checked(fit['arms'][a]['source_cohorts']['DIRECT_CONTROL'][cohort]['outputs'])
        write(out/'COMPLETE.json',dict(completed=True,PASS=True,source_only=True,code=e['code'],protocol=e['protocol'],entry=bind(entry_path),source_fit_complete=e['fit_complete'],all2_freeze=fit['all2_freeze'],direct_cache=p['direct_cache_complete'],split_complete=p['split_complete'],outputs=dict(metrics=bind(readout),table=bind(table),ledger=bind(ledger),triplets=bind(triplet_path)),original_rows=3144,known_FULL372_TRAIN327_HOLDOUT45=True,target_UNKNOWN=2772,original_inputU=112,known_triplets_FULL124_TRAIN109_HOLDOUT15=True,original_component_roles_preserved=True,no_models_or_forward_or_weights_or_images_or_new_sampling=True,model_or_threshold_or_data_selection=False,adopted=False,fully_solved=False))
        print(json.dumps(dict(completed=True,PASS=True,complete=bind(out/'COMPLETE.json')),ensure_ascii=False),flush=True)
    except BaseException:
        write(out/'FAILED_NO_AUTOMATIC_RETRY.json',dict(completed=False,PASS=False,automatic_retry=False,traceback=traceback.format_exc()));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--entry',type=Path,required=True);a=ap.parse_args();evaluate(a.entry)
