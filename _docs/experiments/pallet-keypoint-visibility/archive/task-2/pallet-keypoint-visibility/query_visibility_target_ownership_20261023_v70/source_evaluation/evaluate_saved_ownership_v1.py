"""Exact-query saved-score comparison only; no model/RGB/feature/current reads.

--mock reads generated arrays only. Actual GT and saved score arrays are gated
by a root-approved evaluation ENTRYLOCK and the completed V70 source freeze.
"""
from pathlib import Path
from collections import Counter
import argparse, copy, csv, hashlib, json, math
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT.parent
V66=BASE/'query_visibility_controlled_real_20261022_v66'
V68=BASE/'query_visibility_sam21_frozen_20261023_v68'
V63=BASE/'query_visibility_public_real_20261022_v63'
PRIMARY='V70_OWNERSHIP_LOO_PAIR_V1'
SOURCES=('SYNTH','ORIGINAL_YCB','CONTROLLED_YCB','PALLOX')
COUNTS=dict(SYNTH=1742,ORIGINAL_YCB=3200,CONTROLLED_YCB=3200,PALLOX=448)
OLD_MODELS=('V65_BASELINE','V66_CONTROLLED_PLUS_PALLOX','V67_VISIBLE_BCE1')
GT_KEYS=('sample_id','source','group_id','image_sha256','global_id','global_source',
 'global_image_sha256','target_index','dense_allowed','dense_valid','dense_exact_state','dense_radius2_state')
OLD_KEYS=('sample_id','source','group_id','image_sha256','center_native_valid','center_allowed',
 'GT_state','visible_probability','raw_SAM_visible_probability')
V61_KEYS=('sample_id','source','group_id','image_sha256','global_id','native_valid','logit','pV')
SAM_KEYS=('sample_id','source','group_id','global_id','native_valid','exact_logit','exact_visible')
NEW_KEYS=('sample_id','source','group_id','image_sha256','global_id','native_valid','allowed',
 'exact_GT','known_VE','raw_logit_visible_given_VE','raw_probability_visible_given_VE')
FORBIDDEN={'x','rgb_native','sam','context','box_grid','global_rgb','global_sam','global_prompt','prototype'}
POLICY_SHA='71935f677efabf7b20a6b246f2e5241a0e564eed0287d76663fda8753bfbb5f0'
LABEL_SEAL_SHA='45ffa631781e1efef4f5137393b3340d4fd91a1ca137486c8638dbc59fe83b01'
DENY_SHA='58ef8b2ca79b9cb511fe7b5813d22b31f6ff82ef5049a2be340ec293bdbef987'
V61_SEAL_SHA='5192a2214cd97660c375785aee554d48c4675bd7407570b38af9d2f28b7ddb1b'

def require(x,m):
    if not x: raise ValueError(m)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def rows(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def bound(b,path=None,digest=None):
    p=Path(b['path']).resolve()
    require(path is None or p==Path(path).resolve(),'Unexpected source path '+str(p))
    require(digest is None or b['sha256']==digest,'Unexpected immutable digest '+str(p))
    require(sha(p)==b['sha256'],'Changed source metadata/output '+str(p))
    return p
def same(a,b):return Path(a['path']).resolve()==Path(b['path']).resolve() and a['sha256']==b['sha256']
def boolean(v):
    if v in (True,'True','true',1,'1'):return True
    if v in (False,'False','false',0,'0'):return False
    raise ValueError('Explicit boolean required')
def arrays(p,keys):
    require(not(set(keys)&FORBIDDEN),'Feature arrays are forbidden')
    with np.load(p,allow_pickle=False) as z:
        require(set(keys)<=set(z.files),'Missing saved identity/GT/score keys')
        return {k:z[k] for k in keys}
def frac(a,b):return int(a)/int(b) if b else None
def sigmoid(z):
    z=np.asarray(z,np.float64);p=np.empty(z.shape,np.float64);m=z>=0
    p[m]=1/(1+np.exp(-z[m]));e=np.exp(z[~m]);p[~m]=e/(1+e);return p

def states(allowed,valid):
    a=np.asarray(allowed,np.int64);v=np.asarray(valid)
    require(a.shape==v.shape and a.ndim==1 and v.dtype==bool,'Aligned compatible/native-valid arrays')
    require(np.isin(a,np.arange(1,64)).all() and (a[~v]==63).all(),'Nonempty1..63; invalid remains all63')
    answer=np.full((len(a),2),-1,np.int8)
    for k,bits in enumerate(((1,6,56),(11,20,32))):
        options=np.stack([(a&b)!=0 for b in bits],1);one=options.sum(1)==1
        answer[v&one,k]=options[v&one].argmax(1)
    return answer

def decide(score,valid,strict=False,threshold=.5):
    s=np.asarray(score);v=np.asarray(valid)
    require(s.shape==v.shape and s.ndim==1,'Exact-only score shape')
    require(np.isfinite(s[v]).all(),'Finite scores for native-valid queries')
    out=np.full(len(s),-1,np.int8)
    out[v]=((s[v]>threshold) if strict else (s[v]>=threshold)).astype(np.int8)
    return out

def binary(truth,pred):
    require(np.isin(truth,[0,1]).all() and np.isin(pred,[-1,0,1]).all(),'Binary known truth/prediction')
    tp=int(((truth==1)&(pred==1)).sum());fp=int(((truth==0)&(pred==1)).sum())
    pos=int((truth==1).sum());neg=len(truth)-pos
    tn=int(((truth==0)&(pred==0)).sum())
    return dict(N=len(truth),positive=pos,negative=neg,TP=tp,FP=fp,FN=pos-tp,TN=tn,
      U=int((pred<0).sum()),precision=frac(tp,tp+fp),recall=frac(tp,pos),nonvisible_recall=frac(tn,neg))

def nll(model,y,mask):
    if not mask.any():return dict(N=0,mean=None,nonfinite=0)
    if model.get('logit') is not None:
        z=model['logit'][mask];loss=np.logaddexp(0,z)-y[mask]*z
    else:
        p=model['p'][mask]
        with np.errstate(divide='ignore',invalid='ignore'):
            loss=np.where(y[mask]==1,-np.log(p),-np.log1p(-p))
    finite=np.isfinite(loss)
    return dict(N=int(mask.sum()),mean=float(loss.mean()) if finite.all() else None,
      nonfinite=int((~finite).sum()),finite_loss_sum=float(loss[finite].sum()),
      probability_clipping_applied=False)

def cohort_summary(gt,model,mask):
    st=gt['states'][:,0];valid=gt['valid']&mask;pred=model['prediction']
    known=valid&(st>=0);unknown=valid&(st<0);ve=valid&np.isin(st,[0,1])
    table=np.zeros((3,3),np.int64);column=np.where(pred==1,0,np.where(pred==0,1,2))
    for g,c in zip(st[known],column[known]):table[int(g),int(c)]+=1
    def absence(state):
        m=valid&(st==state)
        return dict(N=int(m.sum()),binary_nonvisible=int((m&(pred==0)).sum()),
          visible=int((m&(pred==1)).sum()),U=int((m&(pred<0)).sum()),
          recall=frac((m&(pred==0)).sum(),m.sum()),
          scope='Truth stratum only; binary absence is not inferred causal E/O')
    marginal_neg=valid&((gt['allowed']&1)==0);marginal_pos=valid&(gt['allowed']==1)
    mk=marginal_neg|marginal_pos
    return dict(total=int(mask.sum()),native_valid=int(valid.sum()),native_invalid=int((mask&~gt['valid']).sum()),
      known_class_GT=int(known.sum()),UNKNOWN_class_GT=int(unknown.sum()),
      class_known_confusion=dict(GT_rows=['V','E','O'],prediction_columns=['VISIBLE','ABSENT','U'],counts=table.tolist()),
      visible_binary_ALL_known_VEO=binary((st[known]==0).astype(np.int8),pred[known]),
      conditional_known_VE=dict(scope='KnownV/E only; O and UNKNOWN excluded from this conditional denominator',
        **binary((st[ve]==0).astype(np.int8),pred[ve]),NLL=nll(model,(st==0).astype(float),ve)),
      known_E_to_binary_nonvisible=absence(1),known_O_to_binary_nonvisible=absence(2),
      UNKNOWN_predictions={name:int((unknown&(column==j)).sum())for j,name in enumerate(['VISIBLE','ABSENT','U'])},
      invalid_predictions={name:int((mask&~gt['valid']&(column==j)).sum())for j,name in enumerate(['VISIBLE','ABSENT','U'])},
      partial_binary_known=dict(scope='Different compatible-set binary-known cohort; not quality/selector denominator',
        **binary(marginal_pos[mk].astype(np.int8),pred[mk])))

def summarize(gt,models):
    result={}
    for name,m in models.items():
        by={s:cohort_summary(gt,m,gt['source']==s) for s in SOURCES}
        ns=[by[s]['conditional_known_VE']['NLL']['mean']for s in SOURCES]
        result[name]=dict(decision_rule=m['rule'],support='exact_only',radius2_output_manufactured=False,
          ties_at_frozen_threshold=m['ties'],per_source=by,
          macro_source_knownVE_NLL=float(np.mean(ns)) if all(x is not None for x in ns) else None,
          macro_NLL_scope='Equal mean of four unweighted knownVE source means; comparison only, no reselection')
    return result

def diagnostic_checks(result,strict):
    checks=[]
    def add(category,s,metric,value,minimum):
        checks.append(dict(category=category,source=s,metric=metric,value=value,minimum=minimum,
          passed=value is not None and value>=minimum))
    for s in SOURCES[1:]:
        b=result['per_source'][s]['visible_binary_ALL_known_VEO']
        for metric in ['precision','recall']:add('public_ALL_known_VEO_visible',s,metric,b[metric],.8)
    for s in ['CONTROLLED_YCB','PALLOX']:
        add('known_E_binary_nonvisible',s,'recall',result['per_source'][s]['known_E_to_binary_nonvisible']['recall'],.6)
    for s,N,minimum in [('ORIGINAL_YCB',714,679),('CONTROLLED_YCB',296,278)]:
        require(strict[s]['N']==N,'Frozen strict25V denominator changed')
        add('strict25_pixel_core',s,'TP',strict[s]['TP'],minimum)
    require(len(checks)==10,'All10 fixed diagnostics required')
    return checks

def eligibility(checks,integrity=True):
    broad=[c for c in checks if c['category']!='strict25_pixel_core']
    require(len(broad)==8,'Exactly8 prefrozen broader checks')
    return dict(legacy10_diagnostic_status='PASS' if all(c['passed']for c in checks) else 'HOLD',
      broader8_pass=all(c['passed']for c in broad),integrity_PASS=bool(integrity),
      exploratory_human_comparison_eligible=bool(integrity and all(c['passed']for c in broad)),
      current_inference_allowed=False,automatic_current_gate_issued=False,
      root_separate_once_frozen_authorization_required=True,
      strict25_alone_does_not_block_first_exploratory_comparison=True,solved=False,deployment_approved=False)

def preflight(entry_path,out):
    require(entry_path is not None,'Root source evaluation ENTRYLOCK required')
    e=read(entry_path)
    require(e['schema']=='V70_SAVED_OWNERSHIP_EVALUATION_ENTRYLOCK_V1','Evaluation schema')
    for k,x in dict(approved=True,source_only=True,current223_allowed=False,model_inference_allowed=False,model_training_allowed=False).items():
        require(e.get(k) is x,'Scope '+k)
    bound(e['evaluator'],__file__)
    require(Path(out).resolve()==Path(e['output_directory']).resolve() and not Path(out).exists(),'Preserve locked output')
    require(Path(out).resolve().is_relative_to(ROOT),'V70 output only')
    require(e['expected']==dict(queries=8590,targets=1091,native_valid=8584,native_invalid=6),'Fixed source cohort')
    require(e['thresholds']==dict(head_probability_ge=.5,MobileSAM_probability_strict_gt=.5,SAM21_logit_strict_gt=0.),'Fixed thresholds')
    inputs=e['inputs'];policy_path=bound(inputs['policy'],ROOT/'contract/CURRENT_EXPLORATORY_ELIGIBILITY_POLICY_BEFORE_FIT.json',POLICY_SHA)
    policy=read(policy_path);require(policy['frozen_before_V70_fit_or_model_outputs'] is True,'Prefit eligibility policy')
    label_path=bound(inputs['label_cohort_seal'],ROOT/'contract/LABEL_COHORT_SEAL.json',LABEL_SEAL_SHA)
    label=read(label_path);require(label['completed'] is True,'Source label seal')
    lb={Path(b['path']).name:b for b in label['artifacts']}
    pair_path=bound(lb['VAL_PAIR_INDEX.csv'],ROOT/'contract/VAL_PAIR_INDEX.csv')
    protocol_path=bound(lb['LABEL_COHORT_PROTOCOL.json'],ROOT/'contract/LABEL_COHORT_PROTOCOL.json')
    prior_path=bound(inputs['GT_prior_seal'],V66/'joined_source_cache/SEAL.json')
    prior=read(prior_path);registered=next(x for x in prior['outputs']if x['candidate']=='CONTROLLED_PLUS_PALLOX'and x['split']=='VAL')
    gt_path=V66/'joined_source_cache/CONTROLLED_PLUS_PALLOX_VAL.npz'
    require(Path(inputs['GT_cache']['path']).resolve()==gt_path.resolve() and inputs['GT_cache']['sha256']==registered['sha256'] and gt_path.stat().st_size==registered['bytes'],'Prior immutable GT cache binding')
    deny_path=bound(inputs['denylist'],V63/'contract/EVALUATION_SHA256_DENYLIST.json',DENY_SHA)
    deny=set(read(deny_path)['image_sha256']);require(len(deny)==223,'Complete exclusion manifest')
    selection_path=bound(inputs['source_selection_seal'],ROOT/'SOURCE_SELECTION_SEAL.json');sel=read(selection_path)
    for k,x in dict(completed=True,source_only=True,current_eval_used_for_selection=False,current_inference_allowed=False).items():require(sel.get(k)is x,'Source selection '+k)
    require(sel['primary']=='OWNERSHIP_LOO_PAIR_V1' and sel['threshold']==.5,'Fixed primary/threshold')
    freeze_path=bound(inputs['model_freeze']);require(same(sel['model_freeze'],inputs['model_freeze']),'Selection/freeze drift')
    freeze=read(freeze_path)
    for k,x in dict(completed=True,source_only=True,current223_used=False,eval_data_used=False,PnP_used=False,current_inference_allowed=False,source_only_selection=True,checkpoint_selection_uses_quality_gates=False).items():require(freeze.get(k)is x,'Model freeze '+k)
    require(freeze['optimizer_updates']==2000 and freeze['threshold']==.5 and freeze['parameters']==40137,'One fixed fit budget/head')
    require(freeze['checkpoint_sha256']==sel['checkpoint_sha256'],'Selected checkpoint differs')
    require(same(freeze['inputs']['val'],inputs['GT_cache']) and same(freeze['inputs']['source_label_contract'],inputs['label_cohort_seal']),'Fit source/label cohort changed')
    for b in freeze['code'].values():bound(b)  # Source code only; never import a model.
    fit_protocol=bound(freeze['inputs']['protocol']);plan=read(fit_protocol)
    require(plan['training']==freeze['training_plan'] and plan['source_only'] is True and plan['current_inference_allowed'] is False,'Frozen training plan')
    fit_entry_path=bound(freeze['entry_lock']);fit_entry=read(fit_entry_path)
    require(fit_entry['approved'] is True and fit_entry['source_only'] is True and fit_entry['current223_allowed'] is False,'Fit authorization scope')
    for b in freeze['outputs']:bound(b)
    outputs={Path(b['path']).name:Path(b['path']).resolve()for b in freeze['outputs']}
    primary_path=outputs['VALIDATION_SOURCE_OUTPUTS.npz'];history_path=outputs['HISTORY.json']
    require(primary_path.parent==freeze_path.parent,'Selected saved outputs must accompany freeze')
    history=read(history_path);require([h['update']for h in history]==list(range(100,2001,100)),'Fixed20 validation points')
    best=float('inf');bestrow=None
    for h in history:
        value=h['VAL_macro_knownVE_NLL'];require(math.isfinite(value),'Finite selection criterion')
        improved=value<best;require(h['saved_best']is improved,'Strict minimum history changed')
        if improved:best,bestrow=value,h
    require(bestrow['update']==freeze['best_update'] and best==freeze['best_source_macro_knownVE_NLL'],'Earliest strict source minimum')
    # Earlier completed exact source comparison binds all historical saved heads.
    prior_report_path=bound(inputs['prior_comparison_report'],V68/'source_evaluation/SAM21_FIXED_COMPARISON/SOURCE_RESULTS.json')
    old=read(prior_report_path)
    require(old['completed']is True and old['source_only']is True and old['current_RGB_GT_scores_inputs_read']is False and old['queries']==8590,'Prior source comparison scope')
    require(old['bindings']['GT_cache_prior_binding']['sha256']==registered['sha256'],'Prior comparison used another source GT')
    oldfiles={}
    for name in OLD_MODELS:
        m=old['bindings']['baselines'][name];bound(m['report']);oldfiles[name]=bound(m['output'])
    sam_path=bound(old['bindings']['SAM_saved_npz']);sam_csv=bound(old['bindings']['SAM_saved_csv'])
    v61_seal_path=bound(inputs['v61_seal'],ROOT/'source_evaluation/V61_FROZEN/SEAL.json',V61_SEAL_SHA)
    v61=read(v61_seal_path)
    require(v61['completed']is True and v61['source_only']is True and v61['model_unchanged']is True,'V61 frozen source baseline')
    require(v61['model_state_sha256']==v61['final_model_state_sha256'] and v61['queries']==8590 and v61['native_valid']==8584 and v61['comparison']=='>=' and v61['threshold']==.5,'V61 source counts/decision rule')
    require(same(v61['bindings']['cache'],inputs['GT_cache']),'V61 source canvas differs')
    v61_path=bound(v61['outputs'][0])
    strict_audit_path=bound(inputs['strict_reference_audit'],V66/'source_audit/SOURCE_BOUNDARY_FAILURE_DIAGNOSIS.json')
    strict_audit=read(strict_audit_path);require(strict_audit['completed']is True and strict_audit['current_RGB_GT_scores_inputs_read']==0,'Source-only boundary audit')
    strict_path=bound(next(x for x in strict_audit['output_bindings']if Path(x['path']).name=='SOURCE_BOUNDARY_FAILURE_DIAGNOSIS_QUERY_ROWS.csv'))
    return dict(entry=e,GT=gt_path,deny=deny,primary=primary_path,freeze=freeze,oldfiles=oldfiles,
      sam_path=sam_path,sam_csv=sam_csv,v61_path=v61_path,strict_path=strict_path,pair_path=pair_path,
      bindings=dict(entry=ref(entry_path),evaluator=ref(__file__),policy=ref(policy_path),label_seal=ref(label_path),
        label_protocol=ref(protocol_path),GT_prior_seal=ref(prior_path),GT_prior_binding=registered,
        source_selection_seal=ref(selection_path),model_freeze=ref(freeze_path),fit_protocol=ref(fit_protocol),
        fit_entry=ref(fit_entry_path),history=ref(history_path),primary=ref(primary_path),
        old_comparison=ref(prior_report_path),oldfiles={k:ref(v)for k,v in oldfiles.items()},
        v61_seal=ref(v61_seal_path),v61=ref(v61_path),SAM21=ref(sam_path),SAM21_CSV=ref(sam_csv),
        strict_reference=ref(strict_path),VAL_pair_index=ref(pair_path),denylist=ref(deny_path)))

def load_gt(plan):
    d=arrays(plan['GT'],GT_KEYS);n=len(d['sample_id']);ti=d['target_index']
    require(n==8590 and len(d['global_id'])==1091 and len(set(d['sample_id']))==n,'Canonical GT identity coverage')
    require(set(ti)==set(range(1091)) and dict(Counter(d['source']))==COUNTS,'Target/source counts')
    require(np.array_equal(d['global_source'][ti],d['source'])and np.array_equal(d['global_image_sha256'][ti],d['image_sha256']),'GT target table')
    require(not(set(map(str,d['image_sha256']))&plan['deny']),'Evaluation RGB SHA in source')
    a=d['dense_allowed'][:,4,4];v=d['dense_valid'][:,4,4];st=states(a,v)
    require(v.sum()==8584 and np.array_equal(st[:,0],d['dense_exact_state'][:,4,4])and np.array_equal(st[:,1],d['dense_radius2_state'][:,4,4]),'Compatible bits/exact and near GT')
    return dict(sample_id=d['sample_id'],source=d['source'],group_id=d['group_id'],image_sha256=d['image_sha256'],global_id=d['global_id'][ti],allowed=a,valid=v,states=st)

def check_identity(d,gt,keys):
    for k in keys:require(np.array_equal(d[k],gt[k]),'Canonical score/GT identity '+k)

def make_model(p,valid,rule,strict=False,z=None,threshold=.5):
    p=np.asarray(p,np.float64)
    require(p.shape==valid.shape and np.isfinite(p[valid]).all()and ((p[valid]>=0)&(p[valid]<=1)).all(),'Saved exact probabilities')
    return dict(p=p,logit=None if z is None else np.asarray(z,np.float64),
      prediction=decide(p,valid,strict,threshold),rule=rule,ties=int((p[valid]==threshold).sum()))

def load_models(plan,gt):
    d=arrays(plan['primary'],NEW_KEYS);check_identity(d,gt,('sample_id','source','group_id','image_sha256','global_id'))
    require(np.array_equal(d['native_valid'],gt['valid'])and np.array_equal(d['allowed'],gt['allowed'])and np.array_equal(d['exact_GT'],gt['states'][:,0]),'V70 score canvas/GT drift')
    ve=gt['valid']&np.isin(gt['states'][:,0],[0,1]);require(np.array_equal(d['known_VE'],ve),'V70 conditional membership')
    z=d['raw_logit_visible_given_VE'];p=d['raw_probability_visible_given_VE']
    require(z.shape==p.shape==(8590,)and np.isfinite(z).all()and np.isfinite(p).all(),'Raw network exact fields finite; invalid remains masked U')
    require(np.max(np.abs(sigmoid(z)-p))<1e-12,'V70 saved logit/probability parity')
    models={PRIMARY:make_model(p,gt['valid'],'pVisible_given_VE>=.5; conditional negative is binary ABSENT, not causal E',z=z)}
    raw=None
    for name,path in plan['oldfiles'].items():
        old=arrays(path,OLD_KEYS);check_identity(old,gt,('sample_id','source','group_id','image_sha256'))
        require(np.array_equal(old['center_native_valid'],gt['valid'])and np.array_equal(old['center_allowed'],gt['allowed'])and np.array_equal(old['GT_state'],gt['states']),'Historical exact/near GT identity')
        require(old['visible_probability'].shape==(8590,2),'Historical score shape')
        models[name]=make_model(old['visible_probability'][:,0],gt['valid'],'saved exact pVisible>=.5')
        if raw is None:raw=old['raw_SAM_visible_probability']
        else:require(np.array_equal(raw,old['raw_SAM_visible_probability']),'Frozen MobileSAM score drift')
    models['MOBILE_SAM_RAW']=make_model(raw[:,0],gt['valid'],'saved exact MobileSAM p>.5 strict, logit0 tie negative',strict=True)
    d=arrays(plan['v61_path'],V61_KEYS);check_identity(d,gt,('sample_id','source','group_id','image_sha256','global_id'))
    require(np.array_equal(d['native_valid'],gt['valid']),'V61 native validity')
    require(np.max(np.abs(sigmoid(d['logit'])-d['pV']))<1e-7,'V61 saved logit/probability parity')
    models['V61_FROZEN']=make_model(d['pV'],gt['valid'],'saved original V61 exact pV>=.5; no R2 output',z=d['logit'])
    d=arrays(plan['sam_path'],SAM_KEYS);check_identity(d,gt,('sample_id','source','group_id','global_id'))
    require(np.array_equal(d['native_valid'],gt['valid']),'SAM21 native validity')
    z=d['exact_logit'];require(np.isnan(z[~gt['valid']]).all(),'Invalid SAM21 remains NaN/U')
    pred=decide(z,gt['valid'],True,0.);require(np.array_equal(pred,d['exact_visible']),'SAM21 strict native logit0')
    sam=make_model(sigmoid(z),gt['valid'],'saved selected-native-f32 SAM21 logit>0 strict',z=z)
    sam.update(prediction=pred,ties=int((z[gt['valid']]==0).sum()));models['SAM21_BASE_PLUS_FROZEN']=sam
    rr=rows(plan['sam_csv']);require(len(rr)==8590,'SAM21 saved canonical CSV')
    for i,r in enumerate(rr):
        require(int(r['joined_query_index'])==i,'SAM21 canonical row index')
        for key in ['sample_id','source','group_id','global_id','image_sha256']:require(r[key]==str(gt[key][i]),'SAM21 CSV/GT '+key)
        require(boolean(r['native_valid'])==bool(gt['valid'][i])and int(r['exact_visible'])==int(pred[i]),'SAM21 CSV native/decision parity')
    return models

def boundary_tables(plan,gt,models):
    rr=rows(plan['strict_path']);lookup={str(s):i for i,s in enumerate(gt['sample_id'])}
    strict={name:{}for name in models};boundary={name:{}for name in models};flags={}
    for source,N,anchor in [('ORIGINAL_YCB',714,679),('CONTROLLED_YCB',296,278)]:
        refs=[r for r in rr if r['model']=='V65_BASELINE'and r['source']==source]
        require(len(refs)==3200 and {r['sample_id']for r in refs}==set(gt['sample_id'][gt['source']==source]),'Same frozen boundary IDs')
        selected=[];cohorts={}
        for r in refs:
            i=lookup[r['sample_id']]
            require(int(r['allowed'])==gt['allowed'][i]and boolean(r['native_valid'])==bool(gt['valid'][i])and int(r['exact_GT'])==gt['states'][i,0],'Boundary partial GT drift')
            flags[i]=dict(strict25_V=boolean(r['strict25_V']),strict25_E=boolean(r['strict25_E']),visible_distance_bin=r['visible_distance_bin'])
            if boolean(r['strict25_V']):require(gt['states'][i,0]==0,'Strict V must knownV');selected.append(i)
            if gt['valid'][i]and gt['states'][i,0]==0:cohorts.setdefault('knownV_distance_'+r['visible_distance_bin'],[]).append(i)
            if gt['valid'][i]and gt['states'][i,0]==1:cohorts.setdefault('knownE_strict25'if boolean(r['strict25_E'])else'knownE_boundary',[]).append(i)
        require(len(selected)==N and sum(float(r['exact_pV'])>=.5 for r in refs if boolean(r['strict25_V']))==anchor,'Frozen V65 strict25 denominator/TP')
        idsha=hashlib.sha256(('\n'.join(sorted(str(gt['sample_id'][i])for i in selected))+'\n').encode()).hexdigest()
        for name,m in models.items():
            p=m['prediction'];tp=int((p[selected]==1).sum())
            strict[name][source]=dict(N=N,TP=tp,FN=N-tp,U=int((p[selected]<0).sum()),recall=frac(tp,N),sorted_sample_ids_sha256=idsha)
            boundary[name][source]={}
            for cname,indices in cohorts.items():
                ix=np.asarray(indices);truth=gt['states'][ix,0];visible=int((p[ix]==1).sum());absent=int((p[ix]==0).sum())
                boundary[name][source][cname]=dict(N=len(ix),visible=visible,binary_nonvisible=absent,U=int((p[ix]<0).sum()),
                  correct_visible_recall=frac(visible,len(ix))if(truth==0).all()else None,
                  known_E_to_binary_nonvisible_recall=frac(absent,len(ix))if(truth==1).all()else None)
    return strict,boundary,flags

def pair_tables(plan,gt,models):
    rr=rows(plan['pair_path']);require(len(rr)==3200,'VAL pair index size')
    groups={'V_TO_V':[],'V_TO_E':[]};used=set()
    for r in rr:
        require(r['split']=='VAL','TRAIN pair cannot enter VAL comparison')
        i,j=int(r['original_index']),int(r['controlled_index'])
        require((i,j)not in used,'Duplicate VAL pair');used.add((i,j))
        require(str(gt['sample_id'][i])==r['original_sample_id']and str(gt['sample_id'][j])==r['controlled_sample_id'],'Pair canonical identity')
        require(gt['source'][i]=='ORIGINAL_YCB'and gt['source'][j]=='CONTROLLED_YCB','Pair domains')
        require(gt['states'][i,0]==int(r['original_exact_state'])and gt['states'][j,0]==int(r['controlled_exact_state']),'Pair exact truth')
        if r['pair_type']in groups:groups[r['pair_type']].append((i,j))
    require(len(groups['V_TO_V'])==1126 and len(groups['V_TO_E'])==445,'Frozen paired VAL cohort')
    result={}
    for name,m in models.items():
        result[name]={}
        for kind,pairs in groups.items():
            ix=np.asarray(pairs);a=m['prediction'][ix[:,0]];b=m['prediction'][ix[:,1]]
            matrix=np.zeros((3,3),np.int64);ca=np.where(a==1,0,np.where(a==0,1,2));cb=np.where(b==1,0,np.where(b==0,1,2))
            np.add.at(matrix,(ca,cb),1)
            result[name][kind]=dict(N=len(ix),original_visible=int((a==1).sum()),controlled_visible=int((b==1).sum()),
              controlled_binary_nonvisible=int((b==0).sum()),joint_correct=int(((a==1)&(b==(1 if kind=='V_TO_V'else 0))).sum()),
              original_visible_to_controlled_nonvisible=int(((a==1)&(b==0)).sum()),
              original_nonvisible_to_controlled_visible=int(((a==0)&(b==1)).sum()),
              controlled_visible_recall=frac((b==1).sum(),len(ix))if kind=='V_TO_V'else None,
              known_E_to_binary_nonvisible_recall=frac((b==0).sum(),len(ix))if kind=='V_TO_E'else None,
              prediction_transition_matrix=dict(rows='original VISIBLE/ABSENT/U',columns='controlled VISIBLE/ABSENT/U',counts=matrix.tolist()))
    return result

def write_results(out,report,gt,models,flags):
    out=Path(out);require(not out.exists(),'Preserve saved comparison');out.mkdir(parents=True)
    flat=[]
    for name,m in report['models'].items():
        for source,b in m['per_source'].items():
            for cohort,key in [('ALL_known_VEO','visible_binary_ALL_known_VEO'),('conditional_known_VE','conditional_known_VE')]:
                z=b[key]
                flat.append(dict(model=name,source=source,support='exact',cohort=cohort,
                  N=z['N'],TP=z['TP'],FP=z['FP'],FN=z['FN'],TN=z['TN'],U=z['U'],precision=z['precision'],recall=z['recall'],
                  known_E_N=b['known_E_to_binary_nonvisible']['N'],known_E_to_binary_nonvisible_recall=b['known_E_to_binary_nonvisible']['recall'],
                  known_O_N=b['known_O_to_binary_nonvisible']['N'],known_O_to_binary_nonvisible_recall=b['known_O_to_binary_nonvisible']['recall'],
                  UNKNOWN_class_GT=b['UNKNOWN_class_GT'],conditional_NLL=b['conditional_known_VE']['NLL']['mean']))
    with(out/'SOURCE_RESULTS.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    qrows=[]
    for name,m in models.items():
        for i,sample in enumerate(gt['sample_id']):
            valid=bool(gt['valid'][i]);z=m.get('logit');flag=flags.get(i,{})
            qrows.append(dict(model=name,index=i,sample_id=str(sample),source=str(gt['source'][i]),global_id=str(gt['global_id'][i]),
              image_sha256=str(gt['image_sha256'][i]),native_valid=valid,allowed=int(gt['allowed'][i]),exact_GT=int(gt['states'][i,0]),
              conditional_known_VE=bool(valid and gt['states'][i,0]in(0,1)),p_visible=float(m['p'][i])if valid else None,
              logit=float(z[i])if valid and z is not None else None,prediction=int(m['prediction'][i]),status='VALID'if valid else'U',
              strict25_V=flag.get('strict25_V'),strict25_E=flag.get('strict25_E'),visible_distance_bin=flag.get('visible_distance_bin')))
    with(out/'PER_QUERY.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(qrows[0]));w.writeheader();w.writerows(qrows)
    snapshot=out/'evaluate_saved_ownership.py';snapshot.write_bytes(Path(__file__).read_bytes())
    report.update(completed=True,outputs=[ref(out/n)for n in ['SOURCE_RESULTS.csv','PER_QUERY.csv','evaluate_saved_ownership.py']])
    (out/'SOURCE_RESULTS.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(report=ref(out/'SOURCE_RESULTS.json'),status=report['eligibility_policy_result'])))

def evaluate(entry,out):
    plan=preflight(entry,out);gt=load_gt(plan);models=load_models(plan,gt);summary=summarize(gt,models)
    strict,boundary,flags=boundary_tables(plan,gt,models);pairs=pair_tables(plan,gt,models)
    macro=summary[PRIMARY]['macro_source_knownVE_NLL'];frozen=plan['freeze']['best_source_macro_knownVE_NLL']
    require(macro is not None and abs(macro-frozen)<=1e-6,'Saved selected-output versus source selector NLL drift')
    checks=diagnostic_checks(summary[PRIMARY],strict[PRIMARY]);status=eligibility(checks)
    require(sha(__file__)==plan['entry']['evaluator']['sha256'],'Evaluator changed during scoring')
    report=dict(schema='V70_FROZEN_SOURCE_EXACT_OWNERSHIP_COMPONENT_COMPARISON_V1',primary=PRIMARY,
      models=summary,strict25_visible=strict,boundary_and_strict25E=boundary,paired_original_controlled=pairs,
      source_diagnostic_checks=checks,eligibility_policy_result=status,bindings=plan['bindings'],
      selector_replay=dict(frozen_GPU_macro_knownVE_NLL=frozen,saved_output_recomputed_macro_NLL=macro,difference=macro-frozen,checkpoint_reselected=False),
      queries=8590,targets=1091,native_valid=8584,native_invalid=6,source_only=True,
      primary_visible_precision_scope='ALL native-valid exact-class-known V/E/O; UNKNOWN separate. Conditional knownVE uses a different denominator.',
      negative_output_scope='Binary nonvisibility only. V70 is trained conditional on knownV/E; O and UNKNOWN are unsupported for causal E prediction. No causal E/O precision is produced.',
      reference_semantics='Exact-query source ownership component; neither crop-support nor a hidden semantic corner/reference identity is established.',
      radius2_output_manufactured=False,solved=False,current_inference_allowed=False,automatic_current_gate_issued=False,
      source_RGB_reads=0,feature_array_reads=0,model_weight_reads=0,current_data_reads=0,new_model_forward_calls=0,optimizer_updates=0,PnP_calls=0,
      checkpoint_or_threshold_selected=False,GT_array_whitelist=list(GT_KEYS))
    write_results(out,report,gt,models,flags)

def mock(path):
    path=Path(path);require(not path.exists(),'Preserve generated evidence')
    a=np.tile(np.arange(1,64,dtype=np.uint8),4);v=np.ones(len(a),bool);v[-1]=False
    st=states(a,v);source=np.repeat(np.asarray(SOURCES),63)
    gt=dict(allowed=a,valid=v,states=st,source=source)
    p=np.full(len(a),.5);model=make_model(p,v,'inclusive')
    strict_model=make_model(p,v,'strict',True)
    require((model['prediction'][v]==1).all()and(strict_model['prediction'][v]==0).all()and(model['prediction'][~v]==-1).all(),'Fixed ties/invalid U')
    result=summarize(gt,{'MOCK':model})['MOCK']
    for s in SOURCES:
        m=result['per_source'][s]
        require(m['known_class_GT']==11 and m['conditional_known_VE']['N']==4,'ALL-known11 versus conditional4 denominator')
        require(m['conditional_known_VE']['positive']==1 and m['conditional_known_VE']['negative']==3,'KnownVE masks1/2/4/6')
        require(m['partial_binary_known']['N']==32,'Different marginal-known denominator')
        require(sum(m['UNKNOWN_predictions'].values())==m['UNKNOWN_class_GT'],'UNKNOWN coverage')
        require(m['visible_binary_ALL_known_VEO']['precision']==1/11 and m['conditional_known_VE']['precision']==1/4,'Precision cohorts never collapse')
    # Broader8 may pass while strict25 fails: keep diagnostic HOLD, eligibility true.
    dummy=dict(per_source={s:dict(visible_binary_ALL_known_VEO=dict(precision=.8,recall=.8),known_E_to_binary_nonvisible=dict(recall=.6))for s in SOURCES})
    strict=dict(ORIGINAL_YCB=dict(N=714,TP=679),CONTROLLED_YCB=dict(N=296,TP=278))
    good=diagnostic_checks(dummy,strict);require(all(c['passed']for c in good),'Equality passes')
    badstrict=copy.deepcopy(strict);badstrict['CONTROLLED_YCB']['TP']=277
    policy=eligibility(diagnostic_checks(dummy,badstrict));require(policy['legacy10_diagnostic_status']=='HOLD'and policy['exploratory_human_comparison_eligible']and not policy['current_inference_allowed'],'Prefit10 versus8 eligibility separation')
    for c in good:
        d,s=copy.deepcopy(dummy),copy.deepcopy(strict)
        if c['category']=='strict25_pixel_core':s[c['source']]['TP']-=1
        elif c['category']=='known_E_binary_nonvisible':d['per_source'][c['source']]['known_E_to_binary_nonvisible']['recall']=math.nextafter(.6,-math.inf)
        else:d['per_source'][c['source']]['visible_binary_ALL_known_VEO'][c['metric']]=math.nextafter(.8,-math.inf)
        rr=diagnostic_checks(d,s);require(sum(not x['passed']for x in rr)==1,'Each fixed gate independently fails')
        require(eligibility(rr)['exploratory_human_comparison_eligible']==(c['category']=='strict25_pixel_core'),'Any broader8 fail blocks eligibility')
    # Equal-source knownVE NLL differs from all-row pooled NLL.
    gt2=dict(allowed=np.ones(5,np.uint8),valid=np.ones(5,bool),states=np.zeros((5,2),np.int8),source=np.array([SOURCES[0]]+list(SOURCES[1:])+[SOURCES[3]]))
    mm=make_model(np.array([.8,.7,.6,.9,.1]),gt2['valid'],'inclusive')
    rr=summarize(gt2,{'X':mm})['X'];expected=np.mean([-math.log(.8),-math.log(.7),-math.log(.6),(-math.log(.9)-math.log(.1))/2])
    require(abs(rr['macro_source_knownVE_NLL']-expected)<1e-14,'Equal-source conditional NLL')
    try:preflight(None,ROOT/'UNUSED_GENERATED_RESULT')
    except ValueError:pass
    else:raise AssertionError('Missing entry must fail before arrays')
    report=dict(schema='V70_GENERATED_SAVED_EXACT_EVALUATOR_MOCK_V1',completed=True,status='PASS',evaluator=ref(__file__),
      all63_compatible_masks=True,ALL_knownVEO_vs_conditionalVE_vs_marginal_known_separate=True,
      macro_source_conditional_NLL=True,strict_MobileSAM_SAM21_vs_inclusive_head_ties=True,
      invalid_U_UNKNOWN_preserved=True,no_causal_Eprecision_or_fakeR2=True,
      prefrozen_10_diagnostics_vs8_eligibility_and_all_individual_failures=True,
      current_authorization_automatically_issued=False,actual_GT_scores_RGB_features_weights_current_reads=0,
      model_forward_calls=0,optimizer_updates=0,PnP_calls=0)
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(ref(path)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--entry-lock',type=Path);p.add_argument('--out',type=Path);p.add_argument('--mock',type=Path)
    a=p.parse_args()
    if a.mock:require(a.entry_lock is None and a.out is None,'Generated mock cannot mix actual inputs');mock(a.mock)
    elif a.entry_lock and a.out:evaluate(a.entry_lock,a.out)
    else:p.error('--mock OR --entry-lock and --out required')
