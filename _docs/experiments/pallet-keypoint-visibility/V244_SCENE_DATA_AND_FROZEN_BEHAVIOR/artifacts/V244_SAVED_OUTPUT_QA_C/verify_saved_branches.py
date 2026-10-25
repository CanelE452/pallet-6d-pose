"""stdlib only: declared saved CSV arithmetic. No checkpoint/features or original pointers.

Portable: python -B verify_saved_branches.py --saved-dir <copied actual_diagnosis_v1>
Optional --output-directory writes a fresh review; without it one stdout JSON is returned.
"""
import argparse,csv,hashlib,io,json,math,os,stat,struct
from collections import Counter
from pathlib import Path

COMPLETE_SHA='3df1dbbe5aa6a199b7e4e35c77b1b88c28b3eb92a99b2f635925078c41222293'
NAMES={'rows_CSV':'ALL7617_OWN_BRANCH_ROWS.csv','summary':'SUMMARY.json','strata_CSV':'FAMILY_ROLE_TARGET_BRANCH_STATS.csv','replay_QA':'TOTAL_REPLAY_QA.json','input_QA':'CANONICAL_SOURCE_INPUT_QA.json'}
BRANCHES={'own_base':'own_base_prediction','own_residual_sign_only':'own_residual_sign_prediction','own_total':'own_total_prediction'}
STATS=('selected_own_base_logit','selected_own_residual_logit','selected_own_total_logit','target_aligned_residual','base_target_margin','total_target_margin')

def need(ok,msg):
    if not ok:raise ValueError(msg)

def safe(p):
    p=Path(p).absolute();need('..' not in p.parts and not any(t in str(p).casefold() for t in ('v112','__pycache__','.pyc','current223','publicheld','private_evaluation','source_holdout','oldval','v177','v178','v182')),'lexical before FS')
    for q in reversed(p.parents):need(stat.S_ISDIR(os.lstat(q).st_mode),'symlink/non-directory ancestor')
    try:need(stat.S_ISDIR(os.lstat(p).st_mode) or stat.S_ISREG(os.lstat(p).st_mode),'symlink/nonregular leaf')
    except FileNotFoundError:pass
    return p

def read(p,b=None):
    with os.fdopen(os.open(safe(p),os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:raw=f.read()
    sha=hashlib.sha256(raw).hexdigest()
    if b:need(sha==b['sha256'] and len(raw)==b['size_bytes'],'declared saved bytes changed')
    return raw,{'path':str(p),'sha256':sha,'size_bytes':len(raw)}

def same(a,b):
    if isinstance(a,dict):need(isinstance(b,dict) and set(a)==set(b),'JSON keys');return all(same(a[k],b[k]) for k in a)
    if isinstance(a,list):need(isinstance(b,list) and len(a)==len(b),'JSON list');return all(same(x,y) for x,y in zip(a,b))
    if a is None:return b is None
    if isinstance(a,float):return isinstance(b,(int,float)) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10)
    return a==b

def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def pred(z):return 'E' if z>=0 else 'D'
def boolean(s):need(s in ('True','False'),'CSV boolean encoding');return s=='True'

def dist(v):
    if not v:return dict(N=0,mean=None,std=None,min=None,p25=None,median=None,p75=None,max=None,negative=0,zero=0,positive=0)
    v=sorted(v);n=len(v);mean=math.fsum(v)/n
    def q(p):
        a=(n-1)*p;i=math.floor(a);j=math.ceil(a);return v[i]+(v[j]-v[i])*(a-i)
    return dict(N=n,mean=mean,std=math.sqrt(math.fsum((x-mean)**2 for x in v)/n),min=v[0],p25=q(.25),median=q(.5),p75=q(.75),max=v[-1],negative=sum(x<0 for x in v),zero=sum(x==0 for x in v),positive=sum(x>0 for x in v))

def recount(rows):
    metrics={}
    for branch,field in BRANCHES.items():
        metrics[branch]={}
        for f in (0,1):
            for role in ('FIT','DEV'):
                rr=[r for r in rows if r['family']==f and r['role']==role];c=Counter((r['target'],r[field]) for r in rr)
                cm=[[c[y,p] for p in ('D','E','U')] for y in (0,1)]
                item=dict(rows=len(rr),known_DE=len(rr),CM_D_E_by_D_E_U=cm,target_UNKNOWN=0,original_input_U=sum(not r['available'] for r in rr))
                for label,k in [('D',0),('E',1)]:
                    truth=sum(cm[k]);prediction=cm[0][k]+cm[1][k];tp=cm[k][k]
                    item[label]=dict(truth=truth,predicted=prediction,precision=tp/prediction if prediction else None,recall=tp/truth if truth else None)
                metrics[branch][f'family{f}_{role}']=item
    strata=[]
    for f in (0,1):
        for role in ('FIT','DEV'):
            for y in (0,1):
                rr=[r for r in rows if r['family']==f and r['role']==role and r['target']==y]
                s=dict(family=f,role=role,target=y,rows=len(rr),own_base_correct=sum(r['base_correct'] for r in rr),own_total_correct=sum(r['total_correct'] for r in rr))
                s.update({k:dist([r[k] for r in rr]) for k in STATS})
                s['crossings']={k:sum(r['threshold_crossing']==k for r in rr) for k in ('D_TO_E','E_TO_D','NONE')}
                s['effects']={k:sum(r['target_effect']==k for r in rr) for k in ('RECOVERED','DAMAGED','UNCHANGED_CORRECT','UNCHANGED_WRONG')}
                strata.append(s)
    return metrics,strata

def verify(saved):
    saved=safe(saved);raw,cb=read(saved/'COMPLETE.json');need(cb['sha256']==COMPLETE_SHA,'exact frozen own-branch COMPLETE')
    complete=json.loads(raw);need(complete['completed'] is True and complete['PASS'] is True,'frozen complete PASS')
    need(complete['new_optimizer_updates']==0 and complete['new_current_inference']==0 and complete['independent_CONTROL_checkpoint_used'] is False,'no fit/current/independent CONTROL')
    data={};inputs={'COMPLETE':cb}
    for key,name in NAMES.items():
        b=complete[key];need(Path(b['path']).name==name,'fixed admitted output basename; original pointers inert')
        raw,inputs[key]=read(saved/name,b)
        data[key]=list(csv.DictReader(io.StringIO(raw.decode(),newline=''))) if key.endswith('CSV') else json.loads(raw)
    rows=data['rows_CSV'];need(len(rows)==7617 and len({r['row_id'] for r in rows})==7617,'all7617 unique saved rows')
    family_roles=Counter();counts=Counter();max_selected_delta=0.
    for i,r in enumerate(rows):
        r['sample_index']=int(r['sample_index']);r['family']=int(r['family']);r['target']=int(r['target']);r['available']=boolean(r['available'])
        need(r['sample_index']==i and r['family'] in (0,1) and r['target'] in (0,1) and r['role'] in ('FIT','DEV') and r['available'],'original known row order/role/family/availability')
        counts[r['role']]+=1;family_roles[r['family'],r['role'],r['target']]+=1
        numeric=[k for k in r if 'logit' in k or 'margin' in k or k in ('total_replay_delta','target_aligned_residual')]
        for k in numeric:r[k]=float(r[k]);need(math.isfinite(r[k]),'finite saved scalar')
        for f in (0,1):need(r[f'own_total_logit{f}']==f32(r[f'own_base_logit{f}']+r[f'own_residual_logit{f}']),'FP32 both-family additive reconstruction')
        f=r['family'];y=r['target'];s=2*y-1;b=r['own_base_logit'+str(f)];d=r['own_residual_logit'+str(f)];t=r['own_total_logit'+str(f)]
        need(r['selected_own_base_logit']==b and r['selected_own_residual_logit']==d and r['selected_own_total_logit']==t,'same family scalar for all three branches')
        need(r['own_base_prediction']==pred(b) and r['own_residual_sign_prediction']==pred(d) and r['own_total_prediction']==pred(t)==r['saved_total_prediction'],'binary threshold0 and saved sign exact')
        need(r['target_aligned_residual']==s*d and r['base_target_margin']==s*b and r['total_target_margin']==s*t,'signed target direction/margins')
        need(r['total_replay_delta']==t-r['saved_total_logit'] and abs(t-r['saved_total_logit'])<=1e-5+1e-5*abs(r['saved_total_logit']),'selected saved TOTAL inherited atol=rtol1e-5')
        max_selected_delta=max(max_selected_delta,abs(r['total_replay_delta']))
        base_correct=pred(b)==('E' if y else 'D');total_correct=pred(t)==('E' if y else 'D')
        need(boolean(r['base_correct'])==base_correct and boolean(r['total_correct'])==total_correct,'correctness flags')
        r['base_correct'],r['total_correct']=base_correct,total_correct
        effect={(False,True):'RECOVERED',(True,False):'DAMAGED',(True,True):'UNCHANGED_CORRECT',(False,False):'UNCHANGED_WRONG'}[base_correct,total_correct]
        crossing={('D','E'):'D_TO_E',('E','D'):'E_TO_D'}.get((pred(b),pred(t)),'NONE')
        need(r['target_effect']==effect and r['threshold_crossing']==crossing,'target effect/crossing')
    need(counts=={'FIT':5876,'DEV':1741} and sum(v for (f,_,_),v in family_roles.items() if f==0)==6096 and sum(v for (f,_,_),v in family_roles.items() if f==1)==1521,'fixed separated family/role denominators')
    metrics,strata=recount(rows);summary=data['summary']
    need(summary['completed'] is True and summary['PASS'] is True and summary['rows']==7617 and summary['FIT']==5876 and summary['DEV']==1741 and summary['threshold']==0 and summary['two_binary_outputs_not_argmax'] is True and summary['new_training_updates']==0 and summary['new_current_inference']==0 and summary['current_admission'] is False,'summary fixed scope')
    need(same(metrics,summary['branch_metrics']) and same(strata,summary['family_role_target_stats']),'all branch CM/P/R and distributions/crossing/effects recomputed')
    flat=[]
    for s in strata:
        z={k:s[k] for k in ('family','role','target','rows','own_base_correct','own_total_correct')}
        for k in STATS:z.update({k+'_'+a:b for a,b in s[k].items()})
        z.update({'crossing_'+k:v for k,v in s['crossings'].items()});z.update({'effect_'+k:v for k,v in s['effects'].items()});flat.append(z)
    need(len(data['strata_CSV'])==8,'eight role/family/target strata')
    for actual,expected in zip(data['strata_CSV'],flat):
        need(set(actual)==set(expected),'flat strata columns')
        for k,v in expected.items():
            got=actual[k] if isinstance(v,str) else None if actual[k]=='' else float(actual[k])
            need(same(v,got),'strata CSV '+k)
    q=data['replay_QA'];iq=data['input_QA']
    need(q['completed'] is True and q['PASS'] is True and q['rows']==7617 and q['logits_checked']==15234 and q['predictions_exact']==7617 and q['atol']==q['rtol']==1e-5 and q['additive_FP32_sum_byte_exact'] is True and q['own_checkpoint_only'] is True and q['independent_CONTROL_used'] is False and q['state_sha256_before']==q['state_sha256_after'],'upstream both-logit frozen-head replay scope')
    need(max_selected_delta<=q['max_abs']+1e-12 and iq['completed'] is True and iq['PASS'] is True and iq['checked']==7617 and iq['source_group_FIT_DEV_overlap']==0 and iq['tolerance_used']==0 and iq['features_overwritten']==0,'upstream input QA binding and selected replay bound')
    yy=next(s for s in strata if (s['family'],s['role'],s['target'])==(1,'DEV',1))
    observation={'rows':yy['rows'],'own_base_correct':yy['own_base_correct'],'own_total_correct':yy['own_total_correct'],'residual_wrong_direction':yy['target_aligned_residual']['negative'],'residual_right_direction':yy['target_aligned_residual']['positive'],'residual_zero':yy['target_aligned_residual']['zero'],'median_own_base':yy['selected_own_base_logit']['median'],'median_own_residual':yy['selected_own_residual_logit']['median'],'median_own_total':yy['selected_own_total_logit']['median']}
    return dict(completed=True,PASS=True,inputs=inputs,source_diagnostic_complete=cb,rows=7617,counts={'FIT':5876,'DEV':1741,'SYN':6096,'YCB':1521},branch_metrics=metrics,family_role_target_stats=strata,YCB_DEV_E_observation=observation,selected_TOTAL_max_abs_delta=max_selected_delta,all_both_family_FP32_sums_exact=True,all7617_selected_predictions_and_saved_signs_exact=True,numeric_statistics_comparison_tolerance=1e-10,current_admission=False,new_training_or_model_selection=False,raw_CP_features_samples_geometry_RGB_GT_or_current_held_reads=0,limits=['Saved CSV arithmetic only; original samples/feature/teacher/checkpoint provenance is cited through bound upstream QA, not independently reopened.','Both15234 CPU replay is an upstream receipt; this verifier independently checks both ownbranch sums and7617 selected saved deltas/signs.','Ownbase belongs to TREAT; no separate CONTROL comparison. SOURCE DEV is exposed, not blind; residual behavior does not identify a causal image/occluder mechanism.'])

def write(path,value):
    with os.fdopen(os.open(safe(path),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'w') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    return read(path)[1]

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--saved-dir',required=True);p.add_argument('--output-directory');a=p.parse_args()
    result=verify(a.saved_dir)
    if a.output_directory:
        out=safe(a.output_directory);need(not out.exists(),'fresh output only');out.mkdir()
        result['verifier_code']=read(Path(__file__))[1];receipt=write(out/'REVIEW.json',result)
        print(json.dumps({'REVIEW':receipt,'YCB_DEV_E_observation':result['YCB_DEV_E_observation']},ensure_ascii=False))
    else:print(json.dumps(result,ensure_ascii=False))
