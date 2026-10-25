"""Final V261 current inference, then frozen saved-ledger comparison only.

No payload I/O on import. Root must bind BOTH final heads before current input
metadata/RGB access. Both new prediction files freeze before any GT/v61 ledger.
"""
import argparse, ast, csv, hashlib, io, json, math, os, platform, stat, sys, traceback, types
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

sys.dont_write_bytecode=True
HERE=Path(__file__).absolute().parent;R=HERE.parent
POLICY={'path':str(R/'V261_ROOT_FINAL_PUBLIC_REAL_COMPARISON/PROSPECTIVE_FINAL_ONE_COMPARISON_POLICY.json'),'sha256':'5bfeaf8d690b39c6e9c287d1b25340a1ee7b538d90d2d9447be9fa61e94da54c','size_bytes':4732}
DEPENDENCY={'path':str(R/'V260_SINGLE_QUERY_FEATURE_TRANSPORT_B/DEPENDENCY_CLOSURE.json'),'sha256':'900ea2373160158c25ef324f5171e3a49ee0bbac52776d32f93d1eae99ab5b17','size_bytes':8660}
ARMS=('CONTROL_ORIGINAL_SOURCE_FIT','TREAT_ORIGINAL_PLUS_PUBLIC_SURFACE_AUX')
INPUT_SHA='4a9d8e6c781d2116a22ad824d7e7e5f7bbb45e6458497e8cde3dea452d846dbd'
INPUT_QA_SHA='5552585cec218e094a38006c6054e312380441897315e66f23f18d169627172f'

def need(ok,message):
    if not bool(ok):raise ValueError(message)

def lexical(value):
    s=str(value);need(not any(x in s.casefold()for x in('v112','cache','.pyc','source_holdout','private_evaluation_held')),'Forbidden literal before filesystem')
    p=Path(s);need(p.is_absolute()and'..'not in p.parts,'Closed absolute path');return p

def regular(value,exists=True):
    p=lexical(value)
    for a in [*reversed(p.parents),p]:
        try:mode=os.lstat(a).st_mode
        except FileNotFoundError:need(not exists,'Missing exact artifact');continue
        need(not stat.S_ISLNK(mode),'No artifact or ancestor symlink; encodedalias is provenance only')
    return p

def key(b):
    need(set(b)=={'path','sha256','size_bytes'},'Closed binding schema');lexical(b['path'])
    need(isinstance(b['sha256'],str)and len(b['sha256'])==64 and type(b['size_bytes'])is int and b['size_bytes']>=0,'SHA/size binding')
    return b['path'],b['sha256'],b['size_bytes']

def checked(b,white):
    need(key(b)in{key(x)for x in white},'Exact Root whitelist before read')
    p=regular(b['path']);h=hashlib.sha256();size=0
    with p.open('rb')as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block);size+=len(block)
    need(h.hexdigest()==b['sha256']and size==b['size_bytes'],'Exact artifact SHA/size');return p

def document(b,white):return json.loads(checked(b,white).read_bytes())

def binding(p):
    p=regular(p);raw=p.read_bytes();return{'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)}

def write_json(p,value):
    p=regular(p,False);need(not p.exists(),'Fresh immutable file')
    with p.open('x',encoding='utf-8')as f:json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    return binding(p)

def write_lines(p,rows):
    with regular(p,False).open('x',encoding='utf-8')as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n')
    return binding(p)

def write_csv(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with regular(p,False).open('x',newline='',encoding='utf-8')as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    return binding(p)

def module(b,white,name):
    p=checked(b,white);m=types.ModuleType(name);m.__file__=str(p);sys.modules[name]=m
    exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__);return m

def pure(b,white,names,extra=None):
    p=checked(b,white);tree=ast.parse(p.read_bytes());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names]
    need({n.name for n in nodes}==set(names),'Exact pure CODE definitions')
    ns={'np':np,'torch':torch,'F':F,'hashlib':hashlib,'json':json,'math':math,'Counter':Counter,'require':need};ns.update(extra or{})
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
    return types.SimpleNamespace(**{n:ns[n]for n in names},namespace=ns)

def run(entry_path):
    entry_p=regular(entry_path);entry=json.loads(entry_p.read_bytes())
    need(entry.get('approved_actual_final_current_comparison')is True,'Separate Root true entry required before current payloads')
    scope=document(entry['root_scope'],[entry['root_scope']])
    need(scope.get('approved_actual_final_current_comparison')is True and scope.get('entry_contract')=={k:v for k,v in entry.items()if k!='root_scope'},'Exact Root approved scope')
    need(entry['policy']==POLICY and entry['dependency_closure']==DEPENDENCY,'Frozen final-one policy and official implementation')
    need(entry.get('fit_allowed')is False and entry.get('PnP_allowed')is False and entry.get('automatic_retry')is False,'No fit/PnP/automatic retry')
    own=binding(Path(__file__).absolute());need(entry['code']==own,'Exact executing code')
    cw=entry['code_whitelist'];tw=entry['training_metadata_whitelist'];iw=entry['input_metadata_whitelist'];pw=entry['payload_whitelist']
    need({key(b)for b in tw}=={key(b)for b in(POLICY,DEPENDENCY,entry['training_complete'],entry['training_freeze'])}and len(tw)==4,'Closed training policy/official implementation/COMPLETE/freeze metadata')
    need({key(b)for b in iw}=={key(entry['input_manifest']),key(entry['input_QA'])}and len(iw)==2,'Only original current-only input manifest and QA')
    checked(own,cw);policy=document(POLICY,tw);dep=document(DEPENDENCY,tw)
    # No current input, baseline or human-ledger read above this training freeze.
    training=document(entry['training_complete'],tw);freeze=document(entry['training_freeze'],tw)
    need(training.get('completed')is True and training.get('PASS')is True and training['policy']==POLICY and training['freeze']==entry['training_freeze'],'Final paired training COMPLETE')
    need(training['new_updates']=={a:1000 for a in ARMS}and training['head_parameters_each']==76098,'Both final1000 old architecture')
    need(freeze.get('completed')is True and freeze.get('PASS')is True and freeze['updates']=={a:1000 for a in ARMS}and freeze['DEV_or_current_or_v61_head_forwards_before_barrier']==0,'Both final heads/logs/draws frozen before current')
    need(training['checkpoints']==entry['checkpoints'],'Exact two finalCP bindings')
    need(entry['input_manifest']['sha256']==INPUT_SHA and entry['input_QA']['sha256']==INPUT_QA_SHA,'Original current-only223/1784 metadata authority')
    cb=entry['additional_code'];cfg=dep['model_loader_config']
    expected_codes=[own,entry['core_code'],*dep['code'].values(),*cfg['code_whitelist'],*cb.values()]
    need({key(b)for b in cw}=={key(b)for b in expected_codes}and len(cw)==len({key(b)for b in expected_codes}),'Closed CODE whitelist')
    for b in cw:checked(b,cw)
    runtime=dep['runtime'];need(platform.python_version()==runtime['python']and np.__version__==runtime['numpy']and torch.__version__==runtime['torch'],'Pinned MLPY runtime')
    core=module(entry['core_code'],cw,'v261_current_core')
    crop=module(dep['code']['crop_module'],cw,'crop_modules')
    loader=module(dep['code']['model_loader'],cw,'v261_current_official_loader');loader.configure_runtime()
    canonical=pure(dep['code']['canonical'],cw,['state_sha','node_pixel_batch'])
    rawmath=pure(dep['code']['spatial_crop'],cw,['crop'])
    prep=pure(dep['code']['preprocess'],cw,['preprocess_views'],{'MEAN':(.485,.456,.406),'STD':(.229,.224,.225)})
    cond=pure(cb['condition32'],cw,['condition32'])
    legacy_input=pure(cb['V229_input_contract'],cw,['validate_current'],{'identity':lambda b:(str(Path(os.path.abspath(b['path']))),b['sha256']),'ALIAS_CODE':Path(cb['alias_resolver']['path']),'ALIAS_SHA':cb['alias_resolver']['sha256']})
    heads={};head_before={}
    for arm in ARMS:
        cp=torch.load(checked(entry['checkpoints'][arm],pw),map_location='cpu',weights_only=True)
        need(cp['arm']==arm and cp['updates']==1000 and cp['head_parameters']==76098 and cp['use_lora']is False,'Only final76098 checkpoint')
        need(cp['initial_head_sha256']==training['initial_state_sha256']==freeze['initial_state_sha256'],'Original identical seed17 init')
        head=crop.CropFamilyHead();head.load_state_dict(cp['head_state_dict'],strict=True);head.float().cpu().eval().requires_grad_(False)
        need(sum(p.numel()for p in head.parameters())==76098 and len(head.state_dict())==4 and all(torch.isfinite(v).all()for v in head.state_dict().values()),'Strict finite existing head')
        head_before[arm]=canonical.state_sha(head.state_dict());need(head_before[arm]==cp['final_head_sha256'],'Final saved state SHA')
        need(head_before[arm]==freeze['checkpoints_and_logs'][arm]['final_state_sha256'],'Fit freeze state authority')
        heads[arm]=head
    # Only now admit current input-only metadata and exact canonical image bindings.
    input_data=document(entry['input_manifest'],iw);qa=document(entry['input_QA'],iw)
    need(qa.get('completed')is True and qa.get('PASS')is True,'Original input metadata QA')
    frames=legacy_input.validate_current(input_data,np)
    transport=entry['encoded_alias_transport']
    need(transport['mode']=='ROOT_EXPLICIT_V229_ALIAS_TO_REGULAR_CANONICAL_SAME_ENCODED_BYTES'and transport['resolver_code']==cb['alias_resolver'],'Narrow declared alias provenance only')
    mappings=transport['canonical_mappings'];need(len(mappings)==223,'Exactly223 closed image transports')
    for frame,m in zip(frames,mappings):
        need(m['frame_index']==frame['frame_index']and m['alias_binding']==frame['source_fullRGB']['binding'],'Original encodedalias metadata identity')
        need(m['canonical_binding']['sha256']==m['alias_binding']['sha256'],'Same encoded bytes, not re-encoding')
        lexical(m['alias_binding']['path']);key(m['canonical_binding'])
    expected_payloads=[cfg['official_state'],*entry['checkpoints'].values(),*[m['canonical_binding']for m in mappings]]
    need({key(b)for b in pw}=={key(b)for b in expected_payloads}and len(pw)==len({key(b)for b in expected_payloads}),'Only two finalCP/officialstate/223 regular encoded images')
    # Alias paths are never lstat/readlink/opened by this driver. Root resolves them
    # under the pinned old resolver policy; actual target ancestry is checked here.
    checked(cfg['official_state'],pw)
    encoder=loader.load_backbone(cfg,runtime['device'],treatment=False).float().eval().requires_grad_(False)
    encoder_before=canonical.state_sha(encoder.state_dict());need(encoder_before==dep['expected_encoder_state_sha256'],'Exact frozen official encoder')
    op=types.SimpleNamespace(crop=rawmath.crop,preprocess_views=prep.preprocess_views,node_pixel_batch=canonical.node_pixel_batch,crop_module=crop,condition32=cond.condition32)
    out=regular(entry['output_directory'],False);need(not out.exists(),'Fresh output/no resume');out.mkdir();stage='current_inference'
    try:
        from PIL import Image,__version__ as PIL_VERSION
        predictions={a:[]for a in ARMS};facts=[];repeated=False;all_features=[];all_supported=[]
        for frame,m in zip(frames,mappings):
            p=checked(m['canonical_binding'],pw)
            with Image.open(p)as im:rgb=np.asarray(im.convert('RGB'),dtype=np.uint8).copy()
            need(hashlib.sha256(rgb.tobytes()).hexdigest()==frame['decoded_RGB_sha256'],'Original decoded RGB exact')
            feature,supported,reasons,fact=core.canonical_frame_features(frame,rgb,encoder,op,runtime['device'],not repeated)
            all_features.append(feature);all_supported.append(supported)
            if fact['first_batch_repeat_byteexact']:repeated=True
            rr=core.prediction_rows(frame,feature,supported,reasons,heads)
            for arm in ARMS:predictions[arm].extend(rr[arm])
            fact.update({'id':frame['id'],'group_id':frame['group_id'],'alias_binding_provenance_only':m['alias_binding'],'canonical_encoded_binding':m['canonical_binding'],'decoded_RGB_sha256':frame['decoded_RGB_sha256'],'actual_decoder':{'library':'Pillow','version':PIL_VERSION,'operation':"Image.open(...).convert('RGB')"},'q8_L8_available8_sha256':hashlib.sha256(json.dumps({'q8':frame['query_xy'],'L8':frame['L_by_native'],'available8':frame['original_query_available']},sort_keys=True,allow_nan=False).encode()).hexdigest()});facts.append(fact)
            if(frame['frame_index']+1)%25==0:print(json.dumps({'current_frames':frame['frame_index']+1,'total':223}),flush=True)
        need(repeated,'First supported canonical batch repeat required')
        for arm in ARMS:
            need(len(predictions[arm])==1784 and[r['original_global_index']for r in predictions[arm]]==list(range(1784)),'Both complete ordered1784')
            need(sum(not r['original_query_available']for r in predictions[arm])==45 and all(r['prediction']=='U'for r in predictions[arm]if not r['original_query_available']),'Original45 inputU preserved')
            need(canonical.state_sha(heads[arm].state_dict())==head_before[arm],'Frozen head unchanged')
        need(canonical.state_sha(encoder.state_dict())==encoder_before and all(not p.requires_grad for m in(encoder,*heads.values())for p in m.parameters())and all(not m.training for model in(encoder,*heads.values())for m in model.modules()),'All model states frozen/eval')
        pb={a:write_lines(out/('ALL1784_'+a+'_PREDICTIONS.jsonl'),predictions[a])for a in ARMS}
        matrix=np.concatenate(all_features);support=np.concatenate(all_supported)
        need(matrix.shape==(1784,1186)and matrix.dtype==np.float32 and np.isfinite(matrix).all()and not matrix[~support].any(),'All1784 generated features, explicit unsupported zeros')
        np.save(out/'ALL1784_CANONICAL_FEATURES1186.npy',matrix,allow_pickle=False)
        rr=predictions[ARMS[0]]
        np.savez(out/'CURRENT_INPUT_ONLY_FEATURE_METADATA.npz',supported=support,original_available=np.asarray([r['original_query_available']for r in rr],bool),native_slot=np.asarray([r['native_slot']for r in rr],np.int8),frame_index=np.asarray([r['frame_index']for r in rr],np.int64),frame_id=np.asarray([r['id']for r in rr]),row_id=np.asarray([r['row_id']for r in rr]),query_xy=np.asarray([r['query_xy']for r in rr],np.float64),L=np.asarray([r['L']for r in rr],np.float64))
        features_binding=binding(out/'ALL1784_CANONICAL_FEATURES1186.npy');feature_metadata=binding(out/'CURRENT_INPUT_ONLY_FEATURE_METADATA.npz')
        fb=write_json(out/'INPUT_FRAME_FACTS.json',{'completed':True,'frames':facts,'feature_layout':'fullCLS384,fixedQpatch384,centreCLS384,fractions2,unchangednativecondition32','generated_feature_cache_saved':True,'no_GT_labels_or_baselines_in_features':True,'global_bbox_context_encoder_used':False,'alias_transport':'Explicit Root map only; no alias filesystem follow by driver'})
        pf=write_json(out/'BOTH_CURRENT_PREDICTIONS_FROZEN_BEFORE_GT_OR_V61_READOUT.json',{'completed':True,'PASS':True,'predictions':pb,'features':features_binding,'input_only_feature_metadata':feature_metadata,'input_facts':fb,'training_complete':entry['training_complete'],'training_freeze':entry['training_freeze'],'head_checkpoints':entry['checkpoints'],'head_states_before_and_after':head_before,'official_encoder_state_before_and_after':encoder_before,'rows_each':1784,'frames':223,'original_input_U':45,'predicted_U':{a:sum(r['prediction']=='U'for r in predictions[a])for a in ARMS},'output_MAIN0_logit_threshold':0,'GT_cohort_or_v61_baseline_body_reads_before_this_freeze':0,'new_training_updates':0,'PnP_changes':0})
        evaluation={};stage='postfreeze_saved_ledger_evaluation'
        if entry['evaluate_after_prediction_freeze']:
            late=entry['late_evaluation_whitelist'];need({key(b)for b in late}=={key(entry['evaluation_ledger']),key(entry['cohort_flags'])}and len(late)==2,'Only frozen ledger and original ordered cohortflags')
            ledger=list(csv.DictReader(io.StringIO(checked(entry['evaluation_ledger'],late).read_text(encoding='utf-8-sig'))))
            cohort=list(csv.DictReader(io.StringIO(checked(entry['cohort_flags'],late).read_text(encoding='utf-8-sig'))))
            need(len(cohort)==1784 and all(a['id']==b['id']and int(a['native_corner'])==int(b['native_corner'])and a['human_state']==b['human_state']for a,b in zip(ledger,cohort)),'Original cohortflags identical1784 order/semantic labels; no matching')
            flags=[r['paired_same_query_RGB_D_vs_ES']=='True'for r in cohort]
            metric=pure(cb['metric'],cw,['ratio','calculate'],{'D':core.D});error=pure(cb['strata'],cw,['error_bucket'])
            metric.error_bucket=error.error_bucket
            result=core.evaluate_saved_predictions(ledger,predictions,metric,flags)
            for name,rows in result['tables'].items():evaluation[name]=write_csv(out/(name+'.csv'),rows)
            evaluation['ALL1784_FINAL_COMPARISON_LEDGER']=write_csv(out/'ALL1784_FINAL_COMPARISON_LEDGER.csv',result.pop('ledger'))
            result.pop('tables');evaluation['SUMMARY']=write_json(out/'SUMMARY.json',dict(result,prediction_freeze=pf,reference_scope='Frozen old human association; not corrected q or new point gold',no_further_fit_model_or_threshold_selection=True))
        complete=write_json(out/'COMPLETE.json',{'completed':True,'PASS':True,'code':own,'policy':POLICY,'dependency_closure':DEPENDENCY,'entry':binding(entry_p),'training_complete':entry['training_complete'],'training_freeze':entry['training_freeze'],'checkpoints':entry['checkpoints'],'input_manifest':entry['input_manifest'],'input_QA':entry['input_QA'],'predictions':pb,'features':features_binding,'input_only_feature_metadata':feature_metadata,'input_facts':fb,'prediction_freeze':pf,'saved_evaluation_outputs':evaluation,'evaluation_ledger':entry['evaluation_ledger']if entry['evaluate_after_prediction_freeze']else None,'cohort_flags':entry['cohort_flags']if entry['evaluate_after_prediction_freeze']else None,'new_optimizer_updates':0,'baseline_model_forwards':0,'threshold_changes':0,'query_or_native_reordering':0,'PnP_changes':0,'automatic_model_adoption':False,'subsequent_recipe_or_threshold_retry_permitted':False})
        print(json.dumps({'completed':True,'PASS':True,'COMPLETE':complete}),flush=True);return complete
    except BaseException as exc:
        write_json(out/'FAILED_NO_AUTOMATIC_RETRY.json',{'completed':False,'stage':stage,'error':str(exc),'traceback':traceback.format_exc(),'automatic_retry':False,'new_optimizer_updates':0});raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--entry',required=True);run(ap.parse_args().entry)
