"""Root 허가 뒤 metadata만으로 entry 조립. 배열/가중치/CSV 본문은 절대 열지 않음."""
import argparse,copy,hashlib,json,os,stat
from pathlib import Path

H=Path(__file__).absolute().parent;R=H.parent
CHECKER=R/'V242_CF_FEATURE_PACKET_QA_C'
PINS={
 'template':(str(CHECKER/'ROOT_ENTRY_FALSE_TEMPLATE.json'),'fa4be6f2ff8769418a12e722d743797d68349787866e74013b214b6b33dcc8c0'),
 'code':(str(CHECKER/'verify_features.py'),'7541bdb58d6582a301df5857e7053240759021bc4a28c6c7a12f5e1b37288a17'),
 'io_code':(str(CHECKER/'packet_io.py'),'6820bd7080dac8bbd5cc25cdb7ea0191672752f21eb4183265a8d56cf500fbcc'),
 'protocol':(str(CHECKER/'PROTOCOL_BEFORE_CF_FEATURE_PACKET_READS.json'),'507b32101a1486860adb0bb0f7c59621c4cedee0767f73fcd1cc24682d36aaad'),
 'root_protocol':(str(R/'V245_ROOT_SCENE_CF_ACTUAL/PROTOCOL_BEFORE_CF_FEATURES_AND_FIXED_SOURCE_FIT.json'),'5c5eab43fe3fee635c42647acd05385e8ea9a32156fb7ce04010b9b54315e7c9'),
 'data_complete':(str(R/'V242_ROOT_SCENE_CF/scene_pairs_v1/COMPLETE.json'),'12f973806eccdbeae25afea8f6e23fb43da738f2d412d7dbdbb82ea8d087d8bc'),
 'data_QA':(str(R/'V242_SCENE_DATA_QA_C/actual_review_v1/REVIEW.json'),'33f996e71759f0d49a1587e1ff3b7c2927b6e7e61e0b55abecf53096df334bd4')}
FEATURES={r+'_'+k+'.npy' for r in ('COVER','ADJACENT_SHAM') for k in ('BASE1186','NODE_CLS','CONTEXT_CLS')}|{'RELATIVE_XY8.npy','PARENT_INDICES.npy','CONSTRUCTED.npy'}
DATA={'PARENT_INDICES.npy','CONSTRUCTED.npy','PROOF_PASS.npy','GROUP_NODE_VALID96.npy','GROUP_CONTEXT_VALID224.npy','GROUP_GEOMETRY.npz'}
BASELINE='BASELINE_V239_CANONICAL_CONTEXT';TREAT='TREAT_SCENE_CF_MATCHED_DATA'

def need(ok,msg):
    if not ok:raise ValueError(msg)

def lexical(value):
    p=Path(value);s=str(p).casefold();need(p.is_absolute() and '..' not in p.parts and not any(t in s for t in ('v112','__pycache__','.pyc','current223','publicheld','oldval','source_holdout','private_evaluation','v177','v178','v182')),'lexical before FS');return p

def safe(value):
    p=lexical(value)
    for q in reversed(p.parents):need(stat.S_ISDIR(os.lstat(q).st_mode),'parent symlink/non-directory')
    try:need(stat.S_ISREG(os.lstat(p).st_mode) or stat.S_ISDIR(os.lstat(p).st_mode),'leaf symlink/nonregular')
    except FileNotFoundError:pass
    return p

def identity(b):
    need(isinstance(b,dict) and {'path','sha256'}<=set(b),'binding fields');p=lexical(b['path']);s=b['sha256']
    need(isinstance(s,str) and len(s)==64 and set(s)<=set('0123456789abcdef'),'SHA format');return str(p),s

def bound(b):
    identity(b);return {k:b[k] for k in ('path','sha256','size_bytes') if k in b}

def raw(p):
    with os.fdopen(os.open(safe(p),os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:return f.read()

def binding(p):
    b=raw(p);return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),size_bytes=len(b))

def checked(b,white,pin=None):
    need(identity(b) in {identity(x) for x in white} and (pin is None or identity(b)==pin),'closed metadata/CODE whitelist before FS')
    p=lexical(b['path']);need(p.suffix in ('.json','.jsonl','.py','.md'),'metadata/CODE only; no CSV/array/model read')
    data=raw(p);need(hashlib.sha256(data).hexdigest()==b['sha256'] and ('size_bytes' not in b or len(data)==b['size_bytes']),'bound metadata bytes changed')
    return json.loads(data) if p.suffix=='.json' else None

def passed(o):need(o.get('completed') is True and o.get('PASS') is True,'terminal metadata PASS required')
def same(a,b):need(identity(a)==identity(b),'exact binding identity')

def unique(bs):
    out={}
    for b in bs:out[identity(b)]=bound(b)
    return list(out.values())

def feature_entry(cfg,white):
    c=checked(cfg['feature_complete'],white);passed(c)
    need(c['schema']=='V242_CANONICAL_CF_FEATURES_V1' and c['code']['sha256']=='c025749399f86fe6c0f09fed6ef1c2510bb44a45fc61a196cecb0446864f4335','frozen CF feature producer V2')
    template=checked(cfg['template'],white,PINS['template']);e=copy.deepcopy(template)
    for k in ('code','io_code','protocol'):same(e[k],cfg[k]);checked(cfg[k],white,PINS[k])
    dc=checked(cfg['data_complete'],white,PINS['data_complete']);q=checked(cfg['data_QA'],white,PINS['data_QA']);passed(dc);passed(q)
    same(c['data_complete'],cfg['data_complete']);same(c['data_QA'],cfg['data_QA']);same(q['dataset_complete'],cfg['data_complete'])
    scope=checked(cfg['root_QA_scope'],white);need(scope.get('approved_independent_SOURCE_CF_feature_packet_QA') is True,'separate explicit Root numeric QA scope')
    checked(cfg['feature_root_protocol'],white,PINS['root_protocol']);same(c['root_protocol'],cfg['feature_root_protocol'])
    need(c['rows']==5406 and dc['counts']['attempted']==5406 and c['base_state_sha256_before']==c['base_state_sha256_after']=='bb4c7a80aa873bf3d34c3c1d44a28cee331503d7ccf52aba85f2d23edb65c3c3','all rows/frozen official state metadata')
    need(set(c['features'])==FEATURES and c['features']==c['outputs'],'exact nine future feature bindings')
    need(c['samples_npz']['sha256']=='837bbe7ff406e27c6ed2981493a83c2adab830cd5b2de498e4a48de456456e99','original samples binding only')
    e.update(approved_independent_SOURCE_CF_feature_packet_QA=True,root_scope=bound(cfg['root_QA_scope']),root_protocol=bound(cfg['feature_root_protocol']),feature_complete=bound(cfg['feature_complete']),data_complete=bound(cfg['data_complete']),data_QA=bound(cfg['data_QA']),sharing_QA=bound(c['canonical_sharing_QA_binding']),features={k:bound(c['features'][k]) for k in sorted(FEATURES)},data_arrays={k:bound(dc['outputs'][k]) for k in sorted(DATA)},pair_metadata=bound(dc['outputs']['PAIR_METADATA.jsonl']),samples_npz=bound(c['samples_npz']))
    # New CF sharing receipt is not old original-packet or original-input QA.
    need(Path(e['sharing_QA']['path']).name=='CANONICAL_CF_SHARING_QA.json','new CF sharing receipt only')
    e['metadata_whitelist']=unique([e[k] for k in ('root_scope','root_protocol','feature_complete','data_complete','data_QA','sharing_QA','pair_metadata')])
    e['payload_whitelist']=unique([*e['features'].values(),*e['data_arrays'].values(),e['samples_npz']])
    e['code_whitelist']=unique([e[k] for k in ('code','io_code','protocol')]);need(len(e['payload_whitelist'])==16,'exact16 future nonpixel payload leaves')
    return e

def report_config(cfg,white):
    c=checked(cfg['fit_complete'],white);passed(c)
    checked(cfg['root_protocol'],white,PINS['root_protocol']);same(c['root_protocol'],cfg['root_protocol'])
    need(c['schema']=='V243_SCENE_CF_DATA_ONLY_HEAD_TRIAL_V1' and c['code']['sha256']=='e18e6e601742dd2ad52cfbd417b5209852375d8a3d163687c7c2f0eb7b2d110d','frozen trainer V2')
    need(c['new_optimizer_updates']=={BASELINE:0,TREAT:1000} and c['head_parameters']=={BASELINE:86054,TREAT:86054} and c['new_DEV_head_output_before_finalfreeze']==0 and c['current_admission'] is False,'baseline reused0/newTREAT1000/final barrier')
    for k in ('CF_data_complete','CF_data_QA'):same(c[k],cfg[k])
    cfq=checked(c['CF_feature_QA'],white);passed(cfq);same(cfq['feature_complete'],c['CF_feature_complete']);same(cfq['dataset_complete'],c['CF_data_complete'])
    same(cfq['data_QA'],c['CF_data_QA']);need(identity(c['CF_data_complete'])==PINS['data_complete'] and identity(c['CF_data_QA'])==PINS['data_QA'],'original dataset and independent data QA unchanged')
    freeze=checked(c['freeze'],white);passed(freeze)
    need(freeze['new_optimizer_updates']=={BASELINE:0,TREAT:1000} and freeze['all_final_updates']=={BASELINE:1000,TREAT:1000} and freeze['new_DEV_head_output_before_finalfreeze']==0,'bothfinal1000 before new DEV; baseline inherited not new1000')
    for arm in (BASELINE,TREAT):same(freeze['checkpoints'][arm],c['checkpoints'][arm]);same(freeze['logs'][arm],c['logs'][arm])
    for k in ('global_feature_QA','baseline_reuse','draw_ledger'):same(freeze[k],c[k])
    checked(cfg['prospective_plan_binding'],white)
    need(isinstance(cfg['expected_source_plan_sha256'],str) and len(cfg['expected_source_plan_sha256'])==64 and c['source_plan_sha256']==freeze['source_plan_sha256']==cfg['expected_source_plan_sha256'],'pre-feature prospective source-plan versus actual metadata')
    need(c['draws_sha256']==freeze['draws_sha256']=='20a1fc74ea15a04a16fe29f8e5b9a114b4629677710ed43c1628b2a1ac00b734','original PCG17 1000draw SHA unchanged')
    g=checked(c['global_feature_QA'],white);passed(g);need(g['before_optimizer'] is True and g['new_DEV_head_output']==0,'all original/CF feature QA before optimizer')
    reuse=checked(c['baseline_reuse'],white);passed(reuse);need(reuse['new_optimizer_updates']==0 and reuse['numerical_checkpoint_loads']==0 and reuse['baseline_scores_interpreted_before_new_finalfreeze']==0,'baseline byte-copy and no prefreeze score interpretation')
    return dict(prepared=True,approved_saved_SOURCE_result_reads=False,requires_separate_Root_saved_output_scope=True,fit_complete=bound(cfg['fit_complete']),freeze=bound(c['freeze']),global_feature_QA=bound(c['global_feature_QA']),baseline_reuse=bound(c['baseline_reuse']),feature_QA=bound(c['CF_feature_QA']),natural_CSVs={a:bound(c['outputs'][a]) for a in (BASELINE,TREAT)},CF_FIT_CSV=bound(c['CF_FIT_output']),producer_summary=bound(c['summary']),all_original_rows=15040,known_natural_rows_each=7617,masked_targetminus1_retained=7423,CF_diagnostic_rows=10812,CF_is_not_primary_natural_denominator=True,new_optimizer_updates={BASELINE:0,TREAT:1000},threshold_logit=0,current_admission=False,selected_arm=None,model_adoption=False)

def run(config_path):
    cfg=json.loads(raw(config_path));need(cfg.get('approved_metadata_only_entry_assembly') is True and cfg.get('automatic_retry') is False,'Root assembly authorization; default false')
    white=cfg['metadata_and_CODE_whitelist'];mode=cfg['mode']
    result=feature_entry(cfg,white) if mode=='feature_QA' else report_config(cfg,white) if mode=='fit_report_readiness' else None
    need(result is not None,'explicit mode');out=safe(cfg['output_directory']);need(out.parent==H and not out.exists(),'fresh own assembly directory');out.mkdir()
    path=out/('ROOT_ASSEMBLED_CF_FEATURE_QA_ENTRY.json' if mode=='feature_QA' else 'SAVED_REPORT_CONFIG_FALSE.json')
    with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'w') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'completed':True,'PASS':True,'metadata_assembly_only':True,'output':binding(path),'array_model_feature_or_CSV_body_reads':0}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);run(p.parse_args().config)
