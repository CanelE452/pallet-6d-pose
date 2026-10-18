"""Postseal independent saved-pose, aggregate, transition and CSV verifier.

Production metric/aggregate/paired modules are deliberately not imported.
Nothing opens a real reference until the explicit GO and all byte seals pass.
"""
from __future__ import annotations
import argparse,csv,gzip,hashlib,itertools,json,math,resource,time,traceback
from collections import Counter
from pathlib import Path
import numpy as np
from .independent_checks import independent_actions,equivalent

SEEDS=(20261018,20261019,20261020)
GROUPS=('R_CONT_F0','R_ORDER_F0','R_ORDER_F1_ZERO','R_ORDER_F1')
VISUAL=('V_ORDER_NULL','V_ORDER_RGB')
BASEGROUPS=('D1_H1_MLP','D1_H1_PAIR_MLP','D1_H1_PAIR_ROLE_AFFINE')
BASELINES=('N0','E0','E0_REGENERATED','POSEFIX_SUBPIX_PF_ANCHOR','POSEFIX_SUBPIX_YOLO_ANCHOR')+tuple(g+'_s'+str(s)for g in BASEGROUPS for s in SEEDS)
COMPARATORS=BASELINES[:5]+tuple(g+'_s20261018'for g in BASEGROUPS)
POPS={'PRIMARY192':('TEST','REMAINDER'),'TEST49':('TEST',),'REMAINDER143':('REMAINDER',),'CAL46':('CAL',)}
NS={'PRIMARY192':192,'TEST49':49,'REMAINDER143':143,'CAL46':46}
ROLES=('N0','E0','OTHER','NO_POSE');TRANSITIONS=('BOTH_SUCCESS','RECOVERY','DAMAGE','BOTH_FAILURE')


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb')as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def rows(path):
    with gzip.open(path,'rt')as f:
        for line in f:
            if line.strip():yield json.loads(line)
def write(path,value):
    with Path(path).open('x')as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def csvrows(path):
    opener=gzip.open if str(path).endswith('.gz')else open
    with opener(path,'rt',newline='')as f:return list(csv.DictReader(f))
def registered(visual):return BASELINES+tuple(g+'_s'+str(s)for g in GROUPS+(VISUAL if visual else())for s in SEEDS)
def contrasts(visual):
    pairs=[(b,g+'_s'+str(s))for g in GROUPS+(VISUAL if visual else())for s in SEEDS for b in COMPARATORS]
    direct=[('R_ORDER_F0','R_CONT_F0'),('R_ORDER_F0','R_ORDER_F1_ZERO'),('R_ORDER_F1_ZERO','R_ORDER_F1')]
    if visual:direct += [('R_ORDER_F1','V_ORDER_NULL'),('V_ORDER_NULL','V_ORDER_RGB')]
    for seed in SEEDS:pairs.extend((a+'_s'+str(seed),b+'_s'+str(seed))for a,b in direct)
    return list(dict.fromkeys(pairs))


def metric(pose,reference,stable=False):
    empty=dict(pose_available=False,translation_cm=None,rotation_deg=None,success=False,metric_error=None)
    try:
        if type(pose.get('available'))is not bool:raise ValueError('Explicit availability')
        if not pose['available']:return empty
        R=np.asarray(pose['R_physical'],float);t=np.asarray(pose['centroid'],float);G=np.asarray(reference['R'],float)
        if R.shape!=(3,3)or t.shape!=(3,)or not np.isfinite(R).all()or not np.isfinite(t).all():raise ValueError('Malformed pose')
        if np.max(np.abs(R.T@R-np.eye(3)))>1e-7 or not math.isclose(float(np.linalg.det(R)),1.,abs_tol=1e-7):raise ValueError('Improper rotation')
        if reference['order']!=2 or G.shape!=(3,3):raise ValueError('Original C2 reference required')
        d=t-np.asarray(reference['t'],float)
        distance=(math.sqrt(math.fsum(float(x)*float(x)for x in d))if stable else float(np.linalg.norm(d)))*100.
        angles=[]
        for phase in(0,1):
            if stable:S=np.eye(3)if phase==0 else np.diag([-1.,1.,-1.])
            else:
                angle=math.pi*phase;c,s=math.cos(angle),math.sin(angle)
                S=np.array([[c,0.,s],[0.,1.,0.],[-s,0.,c]])
            delta=(G@S).T@R;cosine=max(-1.,min(1.,float((np.trace(delta)-1.)/2.)))
            if stable:
                skew=[float(delta[2,1]-delta[1,2]),float(delta[0,2]-delta[2,0]),float(delta[1,0]-delta[0,1])]
                angle=math.atan2(.5*math.sqrt(math.fsum(x*x for x in skew)),cosine)
            else:angle=math.acos(cosine)
            angles.append(math.degrees(angle))
        rotation=min(angles)
        return dict(pose_available=True,translation_cm=distance,rotation_deg=rotation,success=distance<=5. and rotation<=5.,metric_error=None)
    except(ValueError,KeyError,TypeError,ArithmeticError)as exc:return dict(empty,metric_error=type(exc).__name__)


def quantile(values,p):
    a=sorted(values)
    if not a:return None
    h=(len(a)-1)*p;i=math.floor(h);weight=h-i
    return a[i]if i==len(a)-1 else a[i]+(a[i+1]-a[i])*weight
def distribution(values,n):
    a=[float(v)for v in values if v is not None];mean=math.fsum(a)/len(a)if a else None
    variance=math.fsum((v-mean)**2 for v in a)/(len(a)-1)if len(a)>1 else None
    return dict(n=len(a),full_denominator=n,missing=n-len(a),mean=mean,sample_std=math.sqrt(variance)if variance is not None else None,
        sample_variance=variance,median=quantile(a,.5),P90=quantile(a,.9),P95=quantile(a,.95),max=max(a)if a else None,ddof=1,available_only=True,tail_trimming=False)
def status(entry):
    original=entry.get('prediction_status');out=dict(original)if isinstance(original,dict)else{}
    for k in('accepted','fallback','technical_failure','solver_rejected','timeout','status','reason'):out.setdefault(k,entry.get(k))
    if isinstance(original,str):out['status']=original
    return out
def source(pose,feature):
    if not pose.get('available'):return'NO_POSE'
    scale=feature.get('object_scale_m',1.)
    return'N0'if equivalent(pose,feature.get('fallback'),scale)else'E0'if equivalent(pose,feature.get('e0_regenerated'),scale)else'OTHER'
def transition(before,after):return'BOTH_SUCCESS'if before and after else'RECOVERY'if after else'DAMAGE'if before else'BOTH_FAILURE'


def aggregate(records,arm):
    metrics=[r['metrics'][arm]for r in records];n=len(records);successes=sum(m['success']for m in metrics)
    counts={}
    for key in('accepted','fallback','technical_failure','solver_rejected','timeout'):
        v=[r['status'][arm].get(key)for r in records];counts[key]=dict(true_n=sum(x is True for x in v),false_n=sum(x is False for x in v),unknown_n=sum(x is None for x in v))
    counts['raw_status']=dict(Counter(str(r['status'][arm].get('status'))for r in records))
    return dict(denominator=n,successes=successes,success_rate=successes/n if n else None,no_pose=sum(not m['pose_available']for m in metrics),
        translation_cm=distribution([m['translation_cm']for m in metrics],n),rotation_deg=distribution([m['rotation_deg']for m in metrics],n),
        T_gt50cm=sum(m['translation_cm']is not None and m['translation_cm']>50 for m in metrics),T_gt100cm=sum(m['translation_cm']is not None and m['translation_cm']>100 for m in metrics),
        R_gt45deg=sum(m['rotation_deg']is not None and m['rotation_deg']>45 for m in metrics),status_counts=counts,failures=[r['id']for r in records if not r['metrics'][arm]['success']])


class Bootstrap:
    def __init__(self):self.schedules={};self.proofs={}
    def interval(self,values,records,unit):
        labels=sorted(set(r[unit]for r in records));k=len(labels)
        if k<2:return dict(CI95=None,reason='fewer than two original clusters',cluster_count=k)
        key=(unit,tuple(labels))
        if key not in self.schedules:
            rng=np.random.default_rng(20260917);weights=rng.multinomial(k,np.repeat(1./k,k),10000);self.schedules[key]=weights
            self.proofs[unit+'/'+'|'.join(labels)]=dict(unit=unit,labels=labels,seed=20260917,draws=10000,shape=list(weights.shape),draw_array_sha256=hashlib.sha256(weights.astype('<i8').tobytes()).hexdigest())
        weights=self.schedules[key]
        sums=np.asarray([sum(float(v)for r,v in zip(records,values)if r[unit]==label and v is not None)for label in labels]);counts=np.asarray([sum(v is not None for r,v in zip(records,values)if r[unit]==label)for label in labels])
        ns=np.matmul(weights,counts);numer=np.matmul(weights,sums);draws=numer[ns>0]/ns[ns>0]
        return dict(CI95=[quantile(draws,.025),quantile(draws,.975)]if len(draws)else None,cluster_count=k,clusters=labels,cluster_sizes=dict(Counter(r[unit]for r in records)),valid_resamples=len(draws),empty_resamples=int(np.sum(ns==0)),seed=20260917,resamples=10000,unit=unit,method='whole-cluster percentile bootstrap, paired frame-weighted ratio; linear quantile')


def continuous(records,before,after):
    out=dict(denominator=len(records))
    for axis in('translation_cm','rotation_deg'):
        vals=[r['metrics'][after][axis]-r['metrics'][before][axis]for r in records if r['metrics'][after][axis]is not None and r['metrics'][before][axis]is not None]
        out[axis]=dict(common_available_n=len(vals),sum=float(sum(vals)),mean=math.fsum(vals)/len(vals)if vals else None,improved_n=sum(v<0 for v in vals),worsened_n=sum(v>0 for v in vals),equal_n=sum(v==0 for v in vals),minimum=min(vals)if vals else None,maximum=max(vals)if vals else None)
    return out
def paired(records,before,after,bootstrap):
    groups={g:[]for g in TRANSITIONS};delta=[[],[],[]]
    for r in records:
        a,b=r['metrics'][before],r['metrics'][after];groups[transition(a['success'],b['success'])].append(r['id']);delta[0].append(int(b['success'])-int(a['success']))
        for store,axis in zip(delta[1:],('translation_cm','rotation_deg')):store.append(b[axis]-a[axis]if a[axis]is not None and b[axis]is not None else None)
    changes={}
    for key,vals in zip(('success_rate','translation_cm','rotation_deg'),delta):
        valid=[v for v in vals if v is not None]
        changes[key]=dict(mean=math.fsum(valid)/len(valid)if valid else None,common_available_n=len(valid),improved=sum(v>0 if key=='success_rate'else v<0 for v in valid),worsened=sum(v<0 if key=='success_rate'else v>0 for v in valid),recording_cluster_CI=bootstrap.interval(vals,records,'cluster_id'),named_session_cluster_sensitivity_CI=bootstrap.interval(vals,records,'session'))
    both=[r for r in records if r['id']in groups['BOTH_FAILURE']]
    return dict(before=before,after=after,denominator=len(records),counts={k:len(v)for k,v in groups.items()},ids=groups,success_difference_count=sum(delta[0]),success_difference_percentage_points=100.*sum(delta[0])/len(records),deltas=changes,both_failure_tail={a:aggregate(both,a)for a in(before,after)}if both else{},both_failure_continuous=continuous(both,before,after))


def run(run_dir,reference_dir,features,evaluation_dir,output,*,GO=False):
    if not GO:raise PermissionError('Postseal --scoring-GO required before any real reference decode')
    run_dir=Path(run_dir).resolve();reference_dir=Path(reference_dir).resolve();features=Path(features).resolve();evaluation_dir=Path(evaluation_dir).resolve();output=Path(output).resolve()
    started=time.monotonic();cpu=time.process_time();checks=Counter();failures=[];notes=[];bindings={};bootstrap=Bootstrap()
    def require(value,message):
        if not value:raise ValueError(message)
    def check(name,value,detail=None):
        checks[name]+=1
        if not bool(value):failures.append(dict(check=name,detail=detail))
    def compare(expected,actual,path=''):
        if isinstance(expected,dict):
            check('mapping_keys',isinstance(actual,dict)and set(expected)==set(actual),path)
            if isinstance(actual,dict):
                for key in set(expected)&set(actual):compare(expected[key],actual[key],path+'/'+str(key))
        elif isinstance(expected,(list,tuple)):
            check('list_shape',isinstance(actual,(list,tuple))and len(expected)==len(actual),path)
            if isinstance(actual,(list,tuple)):
                for i,(a,b)in enumerate(zip(expected,actual)):compare(a,b,path+'/'+str(i))
        elif isinstance(expected,(bool,int,str))or expected is None:check('exact_value',expected==actual,path)
        else:
            try:
                rel=1e-9 if'variance'in path else 1e-10;error=abs(float(expected)-float(actual));bound=1e-7+rel*abs(float(expected))
                check('aggregate_numeric',math.isfinite(float(actual))and error<=bound,dict(path=path,expected=float(expected),actual=float(actual),absolute_difference=error,bound=bound))
            except(TypeError,ValueError):check('aggregate_numeric',False,path)
    def csvcheck(name,keyfields,expected):
        actual=csvrows(evaluation_dir/name);ak=[tuple(r[k]for k in keyfields)for r in actual];ek=[tuple(str(r[k])for k in keyfields)for r in expected]
        check('CSV_unique_keys/'+name,len(ak)==len(set(ak)));check('CSV_full_rows/'+name,len(actual)==len(expected)and set(ak)==set(ek))
        lookup={key:r for key,r in zip(ak,actual)}
        for key,row in zip(ek,expected):
            if key not in lookup:continue
            got=lookup[key];check('CSV_columns/'+name,set(row)==set(got))
            for column,want in row.items():
                value=got.get(column)
                if want is None:check('CSV_null',value=='',dict(file=name,key=key,column=column));continue
                if type(want)is bool:check('CSV_boolean',value==str(want),dict(file=name,key=key,column=column));continue
                if isinstance(want,(int,float)):
                    try:compare(want,float(value)if isinstance(want,float)else int(value),'CSV/'+name+'/'+column)
                    except(ValueError,TypeError):check('CSV_numeric_parse',False,dict(file=name,key=key,column=column))
                else:check('CSV_string',value==str(want),dict(file=name,key=key,column=column))
    try:
        manifest=read(run_dir/'PREDICTION_MANIFEST.json');plan=run_dir/'EXPERIMENT_PLAN.json';pp=run_dir/'PREDICTIONS.jsonl.gz'
        require(manifest['status']=='ALL_REGISTERED_OUTPUTS_SEALED_BEFORE_REAL_SCORING'and manifest['real_reference_inputs']is False,'All-output prediction manifest missing')
        require(manifest['experiment_plan_sha256']==sha(plan)and manifest['artifacts'][pp.name]==sha(pp),'Plan/prediction bytes changed')
        require(manifest['real_features_sha256']==sha(features),'Original common real bank changed')
        methods=registered(manifest['visual_executed']);require(list(methods)==manifest['arms'],'Registered full arm inventory changed')
        for b in manifest.get('model_and_input_bindings',[]):require(sha(b['path'])==b['sha256'],'Frozen input/model bytes changed')
        ledger=list(rows(pp));require(len(ledger)==238 and len({r['id']for r in ledger})==238 and[r['id']for r in ledger]==manifest['ordered_ids'],'Full ordered238 predictions required')
        require(all(set(r['predictions'])==set(methods)for r in ledger),'A failing arm/row disappeared')
        es=read(evaluation_dir/'EVALUATION_SEAL.json');require(es['status']=='COMPLETE'and es['prediction_manifest_sha256']==sha(run_dir/'PREDICTION_MANIFEST.json'),'New evaluation is not bound to predictions')
        for name,digest in es['artifacts'].items():require(sha(evaluation_dir/name)==digest,'Sealed evaluation artifact changed: '+name)
        rp=reference_dir/'FIXED_REFERENCES.json';rs=reference_dir/'SCORE_SEAL.json';require(sha(rp)==read(rs)['artifacts'][rp.name]==es['reference_sha256'],'Original reference binding changed')
        for label,path in dict(predictions=pp,prediction_manifest=run_dir/'PREDICTION_MANIFEST.json',plan=plan,real_features=features,evaluation_seal=evaluation_dir/'EVALUATION_SEAL.json',references=rp,reference_seal=rs).items():bindings[label]=dict(path=str(path),sha256=sha(path))
        output.mkdir(parents=True,exist_ok=False)
        # First real-reference JSON decode is strictly below complete seal checks.
        references=read(rp)['references'];feature_by={f['id']:f for f in rows(features)};actual={r['id']:r for r in rows(evaluation_dir/'SCORED_ROWS.jsonl.gz')};require(set(actual)=={r['id']for r in ledger}==set(feature_by),'Reference-free ID join changed')
        scored=[];coverage=[];expression_differences=[];rowcsv=[]
        for row in ledger:
            fid=row['id'];f=feature_by[fid];ref=references[row['partition']][fid];value={k:row[k]for k in('id','partition','session','cluster_id')}
            check('identity_join',all(row[k]==f[k]for k in('partition','session','cluster_id')),fid)
            action_sets=[independent_actions(f),independent_actions(f,True)];bank_metrics=[[metric(c['actual_pose'],ref)for c in bank]for bank in action_sets];has=[any(v['success']for v in values)for values in bank_metrics]
            value.update(poses={},metrics={},status={},action_sources={},full_bank_has_success=has[0],pair_bank_has_success=has[1],full_action_count=len(action_sets[0]),pair_action_count=len(action_sets[1]))
            for arm in methods:
                pred=row['predictions'][arm];pose=pred['actual_pose'];m=metric(pose,ref);stable=metric(pose,ref,True);prior=actual[fid]['metrics'][arm]
                check('raw_pose_availability',m['pose_available']==prior['pose_available'],dict(id=fid,arm=arm));check('raw_joint_success',m['success']==prior['success'],dict(id=fid,arm=arm))
                for axis,bound in(('translation_cm',1e-8),('rotation_deg',1e-6)):
                    a,b=m[axis],prior[axis]
                    check('saved_pose_original_metric/'+axis,(a is None and b is None)or(a is not None and b is not None and abs(a-b)<=bound),dict(id=fid,arm=arm,independent=a,production=b,bound=bound))
                    if a!=stable[axis]:expression_differences.append(dict(id=fid,arm=arm,axis=axis,original=a,stable=stable[axis],original_success=m['success'],stable_success=stable['success']))
                check('metric_failure_visibility',bool(m['metric_error'])==bool(prior['metric_error']))
                if m['metric_error']is not None:m['metric_error']=prior['metric_error'] # exact repr is explanatory metadata, not an independent numeric result
                value['poses'][arm]=pose;value['metrics'][arm]=m;value['status'][arm]=status(pred);value['action_sources'][arm]=source(pose,f)
                if arm not in BASELINES:
                    check('unchanged_actual_action_membership',not pose.get('available')or any(pose==c['actual_pose']for c in action_sets[0]),dict(id=fid,arm=arm))
                    check('all_failure_bank_cannot_new_succeed',has[0]or not m['success'],dict(id=fid,arm=arm))
                rowcsv.append(dict(id=fid,partition=row['partition'],session=row['session'],cluster_id=row['cluster_id'],arm=arm,**m,source=value['action_sources'][arm],full_bank_has_success=has[0],pair_bank_has_success=has[1],fallback=value['status'][arm].get('fallback'),status=value['status'][arm].get('status')))
            compare(value,actual[fid],'SCORED/'+fid);scored.append(value)
            coverage.append(dict(id=fid,partition=row['partition'],full_action_count=len(action_sets[0]),pair_action_count=len(action_sets[1]),full_has_success=has[0],pair_has_success=has[1],full_success_action_ids=[c['candidate_id']for c,m in zip(action_sets[0],bank_metrics[0])if m['success']],pair_success_action_ids=[c['candidate_id']for c,m in zip(action_sets[1],bank_metrics[1])if m['success']]))
        compare(coverage,list(rows(evaluation_dir/'CANDIDATE_COVERAGE.jsonl.gz')),'CANDIDATE_COVERAGE')
        primary=[r for r in scored if r['partition']in('TEST','REMAINDER')];check('fixed_full_bank151_41',sum(r['full_bank_has_success']for r in primary)==151 and sum(not r['full_bank_has_success']for r in primary)==41);check('fixed_pair_bank133_59',sum(r['pair_bank_has_success']for r in primary)==133 and sum(not r['pair_bank_has_success']for r in primary)==59)
        evaluation=read(evaluation_dir/'EVALUATION.json');tables=[];transitioncsv=[];sourcecsv=[];sourcechange=[];risk=[];riskdetail=[];threeway=[];selection=[]
        for population,parts in POPS.items():
            subset=[r for r in scored if r['partition']in parts];check('full_denominator/'+population,len(subset)==NS[population])
            for arm in methods:
                summary=aggregate(subset,arm)
                for axis in('translation_cm','rotation_deg'):summary[axis]['variance_ddof1']=summary[axis]['sample_variance']
                compare(summary,evaluation['populations'][population][arm],population+'/'+arm)
                table={k:summary[k]for k in('denominator','successes','success_rate','no_pose','T_gt50cm','T_gt100cm','R_gt45deg')};table.update(population=population,arm=arm)
                for axis in('translation_cm','rotation_deg'):table.update({axis+'/'+k:v for k,v in summary[axis].items()})
                tables.append(table)
                for role in ROLES:sourcecsv.append(dict(population=population,arm=arm,source=role,n=sum(r['action_sources'][arm]==role for r in subset),denominator=len(subset)))
                for before in COMPARATORS:
                    for group in TRANSITIONS:
                        chosen=[r for r in subset if transition(r['metrics'][before]['success'],r['metrics'][arm]['success'])==group]
                        for role in ROLES:sourcechange.append(dict(population=population,arm=arm,before=before,transition=group,source=role,n=sum(r['action_sources'][arm]==role for r in chosen),transition_denominator=len(chosen)))
                for kind,flag in(('FULL_FIXED_ACTION_BANK','full_bank_has_success'),('PAIR_FIXED_ACTION_BANK','pair_bank_has_success')):
                    chosen=[r for r in subset if not r[flag]]
                    for before in COMPARATORS:
                        stats=continuous(chosen,before,arm);entry=dict(population=population,fixed_set=kind,before=before,arm=arm,denominator=len(chosen))
                        for axis in('translation_cm','rotation_deg'):entry.update({axis+'/'+k:v for k,v in stats[axis].items()})
                        risk.append(entry)
                        for r in chosen:
                            mb,ma=r['metrics'][before],r['metrics'][arm]
                            riskdetail.append(dict(population=population,fixed_set=kind,before=before,arm=arm,id=r['id'],source=r['action_sources'][arm],before_available=mb['pose_available'],after_available=ma['pose_available'],before_translation_cm=mb['translation_cm'],after_translation_cm=ma['translation_cm'],delta_translation_cm=ma['translation_cm']-mb['translation_cm']if ma['translation_cm']is not None and mb['translation_cm']is not None else None,before_rotation_deg=mb['rotation_deg'],after_rotation_deg=ma['rotation_deg'],delta_rotation_deg=ma['rotation_deg']-mb['rotation_deg']if ma['rotation_deg']is not None and mb['rotation_deg']is not None else None,T_gt50cm=ma['translation_cm']is not None and ma['translation_cm']>50,T_gt100cm=ma['translation_cm']is not None and ma['translation_cm']>100,R_gt45deg=ma['rotation_deg']is not None and ma['rotation_deg']>45))
                for a,b,c in itertools.product((False,True),repeat=3):
                    ids=[r['id']for r in subset if(r['metrics']['N0']['success'],r['metrics']['E0']['success'],r['metrics'][arm]['success'])==(a,b,c)]
                    threeway.append(dict(population=population,arm=arm,N0_success=a,E0_success=b,new_success=c,n=len(ids),denominator=len(subset),ids='|'.join(ids)))
                flag='pair_bank_has_success'if'_PAIR_'in arm else'full_bank_has_success'
                selection.append(dict(population=population,arm=arm,bank='PAIR'if'_PAIR_'in arm else'FULL',denominator=len(subset),successful_action_exists_n=sum(r[flag]for r in subset),selected_success_n=sum(r['metrics'][arm]['success']for r in subset),selection_loss_n=sum(r[flag]and not r['metrics'][arm]['success']for r in subset),all_actions_failure_n=sum(not r[flag]for r in subset)))
            for before,after in contrasts(manifest['visual_executed']):
                p=paired(subset,before,after,bootstrap);compare(p,evaluation['paired'][population][before+'__TO__'+after],population+'/'+before+'__TO__'+after)
                for r in subset:
                    mb,ma=r['metrics'][before],r['metrics'][after]
                    transitioncsv.append(dict(population=population,before=before,after=after,transition=transition(mb['success'],ma['success']),id=r['id'],before_translation_cm=mb['translation_cm'],after_translation_cm=ma['translation_cm'],before_rotation_deg=mb['rotation_deg'],after_rotation_deg=ma['rotation_deg']))
        csvcheck('ROW_METRICS.csv.gz',('id','arm'),rowcsv);csvcheck('COMPARISON_ALL_ARMS.csv',('population','arm'),tables)
        csvcheck('PAIRED_TRANSITIONS.csv.gz',('population','before','after','id'),transitioncsv);csvcheck('ACTION_SOURCE_COUNTS.csv',('population','arm','source'),sourcecsv)
        csvcheck('GAIN_DAMAGE_ACTUAL_SOURCE.csv',('population','arm','before','transition','source'),sourcechange);csvcheck('ALL_FAILURE_RISK.csv',('population','fixed_set','before','arm'),risk)
        csvcheck('ALL_FAILURE_RISK_ROWS.csv.gz',('population','fixed_set','before','arm','id'),riskdetail);csvcheck('THREE_WAY_SUCCESS.csv',('population','arm','N0_success','E0_success','new_success'),threeway)
        csvcheck('SELECTION_DIAGNOSIS.csv',('population','arm'),selection)
        primary_clusters=len(set(r['cluster_id']for r in primary));primary_sessions=len(set(r['session']for r in primary));check('original_primary6_recordings8_sessions',primary_clusters==6 and primary_sessions==8)
        for name,b in bindings.items():check('input_bytes_preserved',sha(b['path'])==b['sha256'],name)
        with gzip.open(output/'NUMERICAL_EXPRESSION_DIFFERENCES.jsonl.gz','xt')as f:
            for item in expression_differences:f.write(json.dumps(item,sort_keys=True,allow_nan=False)+'\n')
        write(output/'BOOTSTRAP_DRAW_PROOFS.json',dict(seed=20260917,draws=10000,independent_schedules=bootstrap.proofs,PRIMARY_recording_clusters=primary_clusters,PRIMARY_named_sessions=primary_sessions))
        result=dict(schema='RISK_INDEPENDENT_EVALUATION_CHECK_V1',status='PASS'if not failures else'FAIL',checks=sum(checks.values()),checks_by_kind=dict(checks),failure_count=len(failures),failures=failures,
            input_bindings=bindings,rows=238,arms=len(methods),CSV_rows_checked=dict(zip(('ROW_METRICS','COMPARISON','PAIRED_TRANSITIONS','ACTION_SOURCE_COUNTS','GAIN_DAMAGE_ACTUAL_SOURCE','ALL_FAILURE_RISK','ALL_FAILURE_RISK_ROWS','THREE_WAY_SUCCESS','SELECTION_DIAGNOSIS'),map(len,(rowcsv,tables,transitioncsv,sourcecsv,sourcechange,risk,riskdetail,threeway,selection)))),
            fixed_primary_bank=dict(full_success_possible=151,full_all_failure=41,pair_success_possible=133,pair_all_failure=59),numerical_expression_differences=len(expression_differences),
            strict_delta_direction_source='Independent original norm/acos/sin(pi) expression; stable fsum/atan2 differences are recorded separately and never change strict improved/worsened/tie counts.',
            tolerances=dict(original_translation_cm_absolute=1e-8,original_rotation_deg_absolute=1e-6,aggregate_absolute=1e-7,aggregate_relative=1e-10,variance_relative=1e-9),
            bootstrap_draw_proofs_sha256=sha(output/'BOOTSTRAP_DRAW_PROOFS.json'),numerical_expression_differences_sha256=sha(output/'NUMERICAL_EXPRESSION_DIFFERENCES.jsonl.gz'),source_sha256=sha(__file__),
            production_metric_aggregate_paired_imports=0,new_model_forwards=0,new_fits=0,new_PnP_optimizer_solves=0,real_reference_decode_after_complete_prediction_and_evaluation_seals=True,
            cost=dict(wall_seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,threads=1))
        write(output/'VERIFICATION.json',result);return result
    except BaseException as exc:
        if output.exists()and not(output/'FAILED_ATTEMPT.json').exists():write(output/'FAILED_ATTEMPT.json',dict(error=repr(exc),traceback=traceback.format_exc(),checks=dict(checks),failures=failures,wall_seconds=time.monotonic()-started))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--reference-dir',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--evaluation-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--scoring-GO',action='store_true');a=p.parse_args();r=run(a.run_dir,a.reference_dir,a.features,a.evaluation_dir,a.output,GO=a.scoring_GO);print(json.dumps({k:r[k]for k in('status','checks','failure_count','rows','arms','CSV_rows_checked','cost')}));raise SystemExit(0 if r['status']=='PASS'else 1)
