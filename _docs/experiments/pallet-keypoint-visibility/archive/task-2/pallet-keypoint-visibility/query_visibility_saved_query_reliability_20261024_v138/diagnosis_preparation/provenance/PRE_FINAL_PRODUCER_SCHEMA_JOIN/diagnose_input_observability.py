"""Prospective V138 saved input ranks; explicit root entry required.
TRAIN-input bins are completed and sealed before any GT/C29-score body read.
No model, fit, RGB, feature encoder, private data, PnP or coordinate correction.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse,hashlib,importlib.util,json,collections
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
POLICY=V/'ROOT_SAVED_INPUT_RELIABILITY_POLICY_BEFORE_FEATURES.json';POLICY_SHA='cd94b9d260d0d2111dc9ad0ced34dd437577a0c48c799f8654db5ba6973cccfd'
ADDENDUM=V/'ROOT_PROSPECTIVE_EXTERNAL_SIGNAL_TARGET_ADDENDUM_BEFORE_GT.json';ADDENDUM_SHA='b80d82fec62caa730920d2de6b8248e2b1b1a1c42f670ad6b454d9f5deb13ffe'
PROTOCOL=H/'PROTOCOL_BEFORE_GT_JOIN.json';INPUT_SEAL=V/'input_only_reliability_v1/INPUT_SEAL.json';OUT=V/'saved_observability_diagnosis_v1'
MATH=H/'observability_math.py';TARGET=H/'target_contract.py';DESIGN=H/'PROSPECTIVE_DIAGNOSTIC_DESIGN.json'
RISK_NAMES=('risk_low_detector_confidence','risk_low_native_keypoint_confidence','risk_local_invalid_fraction','risk_central32_invalid_fraction','risk_context_invalid_fraction','risk_input_unavailable','risk_native_bounds_invalid','risk_query_outside_pred_bbox_normalized','risk_global_edge_concurrence_rms','risk_native_heldout_edge_rms')
CORPORA={'OLD_NATIVE':dict(N=6144,prefix='V129',trainbases=128,GT_seal=P/'query_visibility_actual_initializer_source_20261024_v132/saved_initializer_diagnosis_v1/DIAGNOSIS_SEAL.json',score=P/'query_visibility_continuous_source_coverage_control_20261024_v137/runs/C28_C29_CONTINUOUS_COVERAGE_FIXED8000/C29_CONTINUOUS_NATIVE_DATA_ADDITION/FINAL_NATIVE6144_SCORES_INPUT_AND_OUTPUT.npz'),
 'CONTINUOUS_NATIVE':dict(N=20480,prefix='V136',trainbases=512,GT_seal=P/'query_visibility_continuous_native_source_20261024_v136/saved_initializer_diagnosis_v1/DIAGNOSIS_SEAL.json',score=P/'query_visibility_continuous_source_coverage_control_20261024_v137/runs/C28_C29_CONTINUOUS_COVERAGE_FIXED8000/C29_CONTINUOUS_NATIVE_DATA_ADDITION/FINAL_NEW_NATIVE20480_SCORES_INPUT_AND_OUTPUT.npz')}
CONTROL=P/'query_visibility_continuous_source_coverage_control_20261024_v137/runs/C28_C29_CONTINUOUS_COVERAGE_FIXED8000/CONTROL_COMPLETE.json'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def same(a,b):return a['path']==b['path']and a['sha256']==b['sha256']
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def lines(p):return[json.loads(s)for s in Path(p).read_text().splitlines()if s]
def packet(p):
 with np.load(p,allow_pickle=False)as f:return{k:f[k].copy()for k in f.files}
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
maths=module(MATH,'prospective_observability_math');targets=module(TARGET,'prospective_observability_targets')

def load_inputs(seal):
 assert seal['completed']and seal['PASS']
 for flag in('GT_read','C29score_read','v61score_read','RGB_body_read','visual_feature_body_read'):assert seal[flag]is False,flag
 x=packet(check(seal['table']));schema=read(check(seal['schema']))
 assert x['risks'].shape==(26624,10)and x['risk_valid'].shape==(26624,10)and x['risk_valid'].dtype==bool
 assert x['risks'].dtype==np.float64 and x['descriptors'].shape==x['descriptor_valid'].shape==(26624,58)and x['descriptors'].dtype==np.float64 and x['descriptor_valid'].dtype==bool
 assert x['risk_names'].tolist()==list(RISK_NAMES)and seal['risk_names']==list(RISK_NAMES)
 assert len(x['descriptor_names'])==len(set(x['descriptor_names'].tolist()))==58
 assert np.array_equal(x['risk_valid'],np.isfinite(x['risks']))and np.array_equal(x['descriptor_valid'],np.isfinite(x['descriptors']))
 assert x['G42'].shape==(26624,42)and x['G42'].dtype==np.float32 and x['G42_constructed'].shape==(26624,)and x['G42_constructed'].dtype==bool
 assert not x['descriptor_valid'][~x['G42_constructed'],:42].any(),'Missing geometry placeholders cannot be ranked.'
 for j in range(42):
  valid=x['descriptor_valid'][:,j];assert np.array_equal(x['descriptors'][valid,j],x['G42'][valid,j].astype(np.float64))
 assert x['q8_image_normalized'].shape==x['q8_image_normalized_valid'].shape==(26624,16)
 for j in range(16):
  valid=x['descriptor_valid'][:,42+j];assert np.array_equal(valid,x['q8_image_normalized_valid'][:,j])and np.array_equal(x['descriptors'][valid,42+j],x['q8_image_normalized'][valid,j])
 assert np.array_equal(x['global_index'],np.arange(26624))and x['query_xy'].shape==(26624,2)and x['L'].shape==(26624,)
 cursor=0
 for corpus,c in CORPORA.items():
  N=c['N'];ids=np.flatnonzero(x['corpus']==corpus);assert np.array_equal(ids,np.arange(cursor,cursor+N));cursor+=N
  for key,value in dict(source_occurrence_index=np.arange(N),scene_index=np.arange(N)//8,base_index=np.arange(N)//32,native_slot=np.arange(N)%8).items():assert np.array_equal(x[key][ids],value),key
  assert np.array_equal(x['occurrence_id'][ids],np.array([f"{c['prefix']}_SCENE_{i//8:04d}:native{i%8}"for i in range(N)]))
  assert np.array_equal(x['split'][ids],np.where(np.arange(N)//32<c['trainbases'],'TRAIN','VALIDATION'))
 return x,schema

def seal_input_bins(x,inputseal):
 values=np.c_[x['risks'],x['descriptors']];names=list(RISK_NAMES)+x['descriptor_names'].tolist();bins=np.full(values.shape,-1,np.int8);proof={}
 for corpus in CORPORA:
  ids=np.flatnonzero(x['corpus']==corpus);train=x['split'][ids]=='TRAIN';proof[corpus]={}
  for j,name in enumerate(names):
   bb,pp=maths.train_input_quintiles(values[ids,j],train);bins[ids,j]=bb;proof[corpus][name]=pp
 np.savez_compressed(OUT/'ALL26624_TRAIN_INPUT_QUINTILE_BINS.npz',bins=bins,score_defined=np.isfinite(values),signal_names=np.array(names),global_index=x['global_index'],corpus=x['corpus'],source_occurrence_index=x['source_occurrence_index'])
 write(OUT/'TRAIN_INPUT_BIN_PROVENANCE.json',dict(completed=True,PASS=True,all26624_preserved=True,proof=proof,numeric_boundaries_exported=False,all_TRAIN_input_distribution_not_teacher_or_VAL=True,GT_or_C29_score_body_reads=0))
 write(OUT/'BINNING_SEAL_BEFORE_GT.json',dict(completed=True,PASS=True,input_seal=inputseal,bins=bind(OUT/'ALL26624_TRAIN_INPUT_QUINTILE_BINS.npz'),provenance=bind(OUT/'TRAIN_INPUT_BIN_PROVENANCE.json'),GT_read=False,C29score_read=False,all26624_preserved=True))
 return values,names,bins

def join_GT_and_scores(e,x):
 out={}
 for corpus,c in CORPORA.items():
  assert same(e['GT_seals'][corpus],bind(c['GT_seal']))and same(e['C29_scores'][corpus],bind(c['score']))
  ds=read(c['GT_seal']);assert ds['completed']and ds['PASS']and ds['RGB_read']is False
  ledger=next(b for b in ds['outputs']if Path(b['path']).name==f"ALL{c['N']}_SLOTS_TWO_CONVENTIONS_GT_ONLY.jsonl");rr=lines(check(ledger));assert len(rr)==2*c['N'];gt=rr[::2]
  score=packet(c['score']);ids=np.flatnonzero(x['corpus']==corpus);pred=score['prediction'];av=score['query_available'];raw=score['raw_head_logit_before_U']
  assert len(pred)==c['N']and np.array_equal(pred,np.where(av,(raw>=0).astype(np.int8),-1))and np.isfinite(raw[score['head_forwarded']]).all()and np.isnan(raw[~score['head_forwarded']]).all()
  assert np.array_equal(x['query_xy'][ids],score['query_xy'],equal_nan=True)and np.array_equal(x['L'][ids],score['L'],equal_nan=True)and np.array_equal(x['occurrence_id'][ids],score['occurrence_id'])
  state=[];error=[]
  for i,r in enumerate(gt):
   assert r['convention']=='CF_PRIMARY_DECLARED'and r['scene_index']==i//8 and r['base_index']==i//32 and r['native_slot']==i%8 and r['query_id']==x['occurrence_id'][ids[i]]and r['split']==x['split'][ids[i]]
   q=[np.nan,np.nan]if r['query_xy']is None else r['query_xy'];side=np.nan if r['roi_side']is None else r['roi_side']
   assert np.array_equal(x['query_xy'][ids[i]],q,equal_nan=True)and(np.isnan(side)and np.isnan(x['L'][ids[i]])or side==x['L'][ids[i]])
   state.append(r['teacher_state']);error.append(np.nan if r['error_px']is None else r['error_px'])
  state=np.array(state);error=np.array(error);assert np.array_equal(state,score['raw_teacher_state_CF'])
  # This validity risk partly defines prediction-U; report that overlap, not a discovery.
  assert np.array_equal(x['risks'][ids,5],(~av).astype(float))
  out[corpus]=dict(ids=ids,teacher=state,error=error,available=av,prediction=pred,targets=targets.targets(state,error,pred,av),GT_ledger=ledger,score=e['C29_scores'][corpus])
 return out

def teacher_masks(state):return{'ALL':np.ones(len(state),bool),'DE':np.isin(state,['D','E']),**{s:state==s for s in'DESU'}}

def diagnose(values,names,bins,x,joined):
 path=OUT/'ALL_FIXED_SIGN_AUC_AND_TRAIN_BIN_COUNTS.jsonl';summary={};accounting={};record_count=0
 with path.open('x')as f:
  for corpus,data in joined.items():
   ids=data['ids'];score=values[ids];bb=bins[ids];state=data['teacher'];available=data['available'];error=data['error'];folds=x['split'][ids]
   accounting[corpus]={};summary[corpus]={}
   for fold in('ALL','TRAIN','VALIDATION'):
    foldmask=np.ones(len(ids),bool)if fold=='ALL'else folds==fold
    accounting[corpus][fold]=dict(original_rows=int(foldmask.sum()),teacher_states=dict(collections.Counter(state[foldmask].tolist())),C29_input_U_all_states=int((foldmask&~available).sum()),known_DE=int((foldmask&np.isin(state,['D','E'])).sum()),known_DE_input_U=int((foldmask&np.isin(state,['D','E'])&~available).sum()),finite_GT_qerror=int((foldmask&np.isfinite(error)).sum()),NO_QUERY_qerror_unknown=int((foldmask&~np.isfinite(error)).sum()))
    summary[corpus][fold]={}
    for teacher,tmask in teacher_masks(state).items():
     base=foldmask&tmask
     for scope,mask in(('ALL_ORIGINAL',base),('C29_INPUT_AVAILABLE_ONLY_SUPPLEMENT',base&available)):
      for target_name in('GT_ERROR_GT3','C29_DE_MISTAKE_WHEN_INPUT_AVAILABLE','C29_KNOWN_FAILURE_INCLUDING_INPUT_U'):
       target=data['targets'][target_name]
       for j,name in enumerate(names):
        auc=maths.fixed_auc(score[:,j],target,mask);row=dict(corpus=corpus,fold=fold,teacher=teacher,availability_scope=scope,target=target_name,signal=name,kind='PRIMARY_FIXED_HIGHER_RISK'if j<10 else'RAW_VALUE_SIGNED_DESCRIPTIVE_NO_RISK_POLARITY',auc=auc,TRAIN_input_quantile_bins=maths.target_bin_counts(bb[:,j],target,mask),
            fixed_sign_not_selected=True,definitional_inputU_overlap=name in('risk_input_unavailable','risk_native_bounds_invalid'))
        f.write(json.dumps(row,allow_nan=False)+'\n');record_count+=1
        if teacher in('ALL','D','E')and scope=='ALL_ORIGINAL'and j<10:summary[corpus][fold].setdefault(teacher,{}).setdefault(target_name,{})[name]=auc
    # Prospective addendum: direct external E1/D0, same10 primary risks only.
    direct=data['targets']['EXTERNAL_TEACHER_E1_D0']
    for errgroup,em in(('ALL_GT_ERROR',np.ones(len(ids),bool)),('GT_ERROR_LE3',np.isfinite(error)&(error<=3)),('GT_ERROR_GT3',np.isfinite(error)&(error>3)),('NO_QUERY',~np.isfinite(error))):
     for scope,mask in(('ALL_ORIGINAL',foldmask&em),('C29_INPUT_AVAILABLE_ONLY_SUPPLEMENT',foldmask&em&available)):
      for j,name in enumerate(RISK_NAMES):
       auc=maths.fixed_auc(score[:,j],direct,mask);row=dict(corpus=corpus,fold=fold,teacher='ALL_WITH_SU_OUTSIDE_TARGET',availability_scope=scope,target='EXTERNAL_TEACHER_E1_D0',GT_error_diagnostic_stratum=errgroup,signal=name,kind='PRIMARY_FIXED_HIGHER_RISK',auc=auc,TRAIN_input_quantile_bins=maths.target_bin_counts(bb[:,j],direct,mask),fixed_sign_not_selected=True,definitional_inputU_overlap=name in('risk_input_unavailable','risk_native_bounds_invalid'))
       f.write(json.dumps(row,allow_nan=False)+'\n');record_count+=1
       if scope=='ALL_ORIGINAL':summary[corpus][fold].setdefault('EXTERNAL_E1_D0_BY_GTERROR',{}).setdefault(errgroup,{})[name]=auc
 return summary,accounting,path,record_count

def run(entry):
 e=read(entry);assert e['approved_saved_observability_diagnosis']is True and all(e[k]is False for k in('model_forward','fit','current223','V112','PnP','q_L_correction','automatic_retry'))
 for key,path in(('code',Path(__file__)),('protocol',PROTOCOL),('policy',POLICY),('addendum',ADDENDUM),('input_seal',INPUT_SEAL)):assert same(e[key],bind(path))
 assert sha(POLICY)==POLICY_SHA and sha(ADDENDUM)==ADDENDUM_SHA and not OUT.exists()
 p=read(PROTOCOL)
 for key,path in(('math',MATH),('targets',TARGET),('design',DESIGN)):assert same(p[key],bind(path))
 for key in('generated_math','generated_targets'):assert read(check(p[key]))['PASS']
 assert np.__version__==p['runtime']['NumPy']
 seal=read(INPUT_SEAL);x,schema=load_inputs(seal);OUT.mkdir();values,names,bins=seal_input_bins(x,e['input_seal'])
 # No GT/score body was read above this line. Bin seal is durable before joins.
 assert same(e['C29_control'],bind(CONTROL));control=read(CONTROL);assert control['completed']and control['PASS']
 joined=join_GT_and_scores(e,x);summary,accounting,table,nrows=diagnose(values,names,bins,x,joined)
 write(OUT/'SOURCE_RESULTS.json',dict(completed=True,PASS=True,source_only=True,goal_status='active',fully_solved=False,code=bind(__file__),entry=bind(entry),policy=e['policy'],addendum=e['addendum'],input_seal=e['input_seal'],binning_seal=bind(OUT/'BINNING_SEAL_BEFORE_GT.json'),full_table=bind(table),records=nrows,all26624_preserved=True,accounting=accounting,summary=summary,
  fixed_primary_signal_order=list(RISK_NAMES),no_sign_or_feature_or_threshold_or_model_selection=True,no_numerical_cut_values_exported=True,
  limitations=['Input-unavailable/bounds flags partly define C29predictionU, so allknownfailure association is partly definitional; availability-only supplement remains explicit.','RawG42/role coordinate AUC reports larger raw value; no universal risk sign and no best-sign flip. Missing/degenerate placeholders masked.','Lowconfidence can reflect external occlusion or displacement; teacherD/E-conditioned position/failure tables and prospective E-vs-D target distinguish associations.','GTerror strata are diagnosis only, never a deployed filter/U rule or coordinate correction.','Same-base rows/variants are correlated. Univariate AUC neither proves causality/application transfer nor excludes multivariate/image observability.'],model_fit_RGB_feature_body_private_V112_PnP_qcorrection_newselection=0))
 write(OUT/'DIAGNOSIS_SEAL.json',dict(completed=True,PASS=True,input_seal=e['input_seal'],binning_seal=bind(OUT/'BINNING_SEAL_BEFORE_GT.json'),results=bind(OUT/'SOURCE_RESULTS.json'),full_table=bind(table),all26624_preserved=True,code=bind(__file__),no_deployment_cut_or_abstention_export=True))
 print(json.dumps(dict(completed=True,PASS=True,seal=bind(OUT/'DIAGNOSIS_SEAL.json'),records=nrows,accounting=accounting)),flush=True)

def prepare():
 assert sha(POLICY)==POLICY_SHA and sha(ADDENDUM)==ADDENDUM_SHA
 for filename in('GENERATED_MATH_CHECKS.json','GENERATED_TARGET_CONTRACT_CHECKS.json'):assert read(H/filename)['PASS']
 p=dict(completed=True,prepared_only=True,code=bind(__file__),policy=bind(POLICY),addendum=bind(ADDENDUM),math=bind(MATH),targets=bind(TARGET),design=bind(DESIGN),generated_math=bind(H/'GENERATED_MATH_CHECKS.json'),generated_targets=bind(H/'GENERATED_TARGET_CONTRACT_CHECKS.json'),
  expected_input_seal=str(INPUT_SEAL),expected_rows26624=True,primary_risk_names=list(RISK_NAMES),descriptors58_raw_signed_no_risk_sign_choice=True,TRAIN_input_bins_before_any_GT_score_body_read=True,no_numerical_cut_export=True,root_entry_required=True,runtime=dict(executable='/home/minjae/anaconda3/envs/pallet-yolo26/bin/python',NumPy='1.26.4'),output=str(OUT))
 write(PROTOCOL,p);write(H/'ROOT_DIAGNOSIS_ENTRY_TEMPLATE.json',dict(approved_saved_observability_diagnosis=False,model_forward=False,fit=False,current223=False,V112=False,PnP=False,q_L_correction=False,automatic_retry=False,
  code=p['code'],protocol=bind(PROTOCOL),policy=p['policy'],addendum=p['addendum'],input_seal=None,C29_control=None,GT_seals={c:None for c in CORPORA},C29_scores={c:None for c in CORPORA},fill_after_complete='Bind completed NEW V138 INPUTseal andfixed sourceGT/C29 scorebindings. No actual bodyreads beforeapproved root entry.'))
 print(json.dumps(dict(prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),template=bind(H/'ROOT_DIAGNOSIS_ENTRY_TEMPLATE.json'))),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);arg=ap.parse_args()
 if arg.prepare and arg.entry is None:prepare()
 elif arg.entry and not arg.prepare:run(arg.entry)
 else:raise SystemExit('Use --prepare or root-approved --entry after INPUTseal.')
