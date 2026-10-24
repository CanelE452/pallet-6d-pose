"""V184 frozen official SAM1 real-input evaluation cache. No classifier or crop forward.
--prepare: final C input metadata/code and invented arrays only, no real RGB/weights.
--entry: separately root-bound, both C52/C53 frozen and independently audited first.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,ast,contextlib,hashlib,json,os,stat,subprocess,sys,tempfile,time,traceback
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
OLD=P/'query_visibility_frozen_sam_source_baseline_20261024_v180'
V183=P/'query_visibility_query_jitter_paired_source_training_20261024_v183'
SCOPE=V/'ROOT_SCOPE_BEFORE_NEW_REAL_OUTPUTS_OR_V183_SOURCE_RESULTS.json';SCOPE_SHA='05c628ad960a5715fedd96fb4e3cde7b2a67ee3ce188e49eb644867f4e3e0b25'
CLARIFY=V/'ROOT_REFERENCE_AUTHORITY_CLARIFICATION_BEFORE_REAL_BODIES.json';CLARIFY_SHA='3e60202c1a000b2c5837f1fe826f26d05bc1201edc019e8e3ecf37f709f33fe6'
INPUT_CODE=V/'input_preparation_c/prepare_original_real_raw8.py';INPUT_CODE_SHA='e50a39464728c97028c689316de349507538c2d74b9bd2ffade69dc372cb334a'
INPUT_PROTOCOL=V/'input_preparation_c/PROTOCOL_BEFORE_REAL_INPUT_METADATA_OR_RAW8_BODIES.json'
INPUT_PREP=V/'input_preparation_c/PREPARATION_COMPLETE.json';INPUT_COMPLETE=V/'original_real_raw8_inputs_v1/COMPLETE.json'
FIT=V183/'runs/C52_C53_JITTER_PAIRED_SOURCE_FIXED8000/COMPLETE.json';FIT_SHA='7e6c231a7daabb69c0b545e06bb04edd0a3da8f2eb5b3ed313e9faa85435ec8c'
FIT_AUDIT=V183/'independent_actual_fit_review_a/REVIEW.json';FIT_AUDIT_SHA='421543bc3bac6353e1251707a13b148fbcce7bff8cfc5708ef99db9e3caabe14'
LEGACY=OLD/'runner_preparation/run_frozen_sam_source.py';LEGACY_SHA='b5d7f5591f83b231887270759de5059a27380ed58419acc3eef01ebdde0f8fa6'
OLD_PROTOCOL=OLD/'runner_preparation/PROTOCOL_BEFORE_WEIGHTS_OR_SOURCE_RGB_BODIES.json'
PROTOCOL=H/'PROTOCOL_BEFORE_REAL_RGB_OR_SAM_WEIGHT_BODIES.json';TESTS=H/'GENERATED_INVENTED_CHECKS.json';OUT=V/'frozen_real268_sam_predictions_v1'
NAMES={'predict_warn_only_call','actual_cuda_cumsum_guard','state_sha','import_runtime','git'}
DOMAINS={'CURRENT223_1784':dict(groups=223,rows=1784,U=45),'PUBLIC_HELD45_360':dict(groups=45,rows=360,U=108)}
ARMS={'C52_RAW96_JITTER_PAIRED_SOURCE','C53_RAW96_SAM_JITTER_PAIRED_SOURCE'}
SAM_KEYS={'image_logits','lowres_logits','predicted_quality','mask_logits_gt0','box_xyxy','original_query_xy','original_L','native_slot','original_global_index','original_occurrence_id','original_query_available','prediction_available'}
def require(ok,msg):
    if not bool(ok):raise AssertionError(msg)
def safe_path(path):
    """Reject forbidden symlink expansions before touching their targets."""
    original=str(path);require('_v112' not in original.lower(),'Forbidden path before filesystem access')
    pending=os.path.abspath(original).split('/')[1:];current='/';hops=0
    while pending:
        candidate=os.path.normpath(os.path.join(current,pending.pop(0)));require('_v112' not in candidate.lower(),'Forbidden component before access');status=os.lstat(candidate)
        if stat.S_ISLNK(status.st_mode):
            target=os.readlink(candidate);require('_v112' not in target.lower(),'Forbidden symlink target before access');expanded=os.path.normpath(target if os.path.isabs(target) else os.path.join(os.path.dirname(candidate),target));require('_v112' not in expanded.lower(),'Forbidden expanded target');hops+=1;require(hops<=40,'Symlink loop/depth');pending=expanded.split('/')[1:]+pending;current='/'
        else:current=candidate
    return Path(current)
def sha(p):
    h=hashlib.sha256()
    with safe_path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def bind(p):return dict(path=str(Path(p).absolute()),sha256=sha(p))
def check(b):require(sha(b['path'])==b['sha256'],'SHA '+b['path']);return Path(b['path'])
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def read(p):return json.loads(safe_path(p).read_text())
def rows(p):return [json.loads(s) for s in safe_path(p).read_text().splitlines() if s]
def write(p,x,exclusive=True):
    with Path(p).open('x' if exclusive else 'w') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def writerows(p,rr):
    with Path(p).open('x') as f:
        for r in rr:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def same(a,b):
    a=np.asarray(a);b=np.asarray(b);return a.shape==b.shape and (np.array_equal(a,b,equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a,b))
def qarray(q):return np.asarray([np.nan,np.nan] if q is None else q,np.float64)
def helper_functions():
    require(sha(LEGACY)==LEGACY_SHA and sha(INPUT_CODE)==INPUT_CODE_SHA,'Pinned official runtime and final input source')
    old=read(OLD_PROTOCOL);pin=read(check(old['source_pin']));tree=ast.parse(LEGACY.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in NAMES];require({n.name for n in nodes}==NAMES,'Exact5 runtime helper ASTs')
    ns=dict(np=np,Path=Path,sys=sys,hashlib=hashlib,subprocess=subprocess,arrsha=ash,VENDOR=OLD/'vendor/segment-anything',COMMIT=pin['exact_commit'],RUNTIME=old['runtime'])
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(LEGACY)+'::EXACT_RUNTIME_HELPERS','exec'),ns)
    source=read(INPUT_PROTOCOL);alias=source['sources']['current']['parent_alias_renderer'];tree=ast.parse(check(alias).read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='safe_original_image_path')
    env=dict(Path=Path,os=os,stat=stat,sha=sha);exec(compile(ast.Module(body=[node],type_ignores=[]),alias['path']+'::SAFE_ALIAS_ONLY','exec'),env);ns['safe_original_image_path']=env['safe_original_image_path']
    tree=ast.parse(INPUT_CODE.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='box_prompt');env=dict(np=np,require=require);exec(compile(ast.Module(body=[node],type_ignores=[]),str(INPUT_CODE)+'::EXACT_INPUT_PROMPT_ONLY','exec'),env);ns['box_prompt']=env['box_prompt']
    return ns,old,pin,alias
def numeric_meta(rr):
    return dict(original_query_xy=np.asarray([qarray(r['query_xy']) for r in rr],np.float64),original_L=np.asarray([np.nan if r['L'] is None else r['L'] for r in rr],np.float64),native_slot=np.asarray([r['native_slot'] for r in rr],np.int8),original_global_index=np.asarray([r['original_global_index'] for r in rr],np.int64),original_occurrence_id=np.asarray([r['row_id'] for r in rr]),original_query_available=np.asarray([r['original_query_available'] for r in rr],bool))
def validate_groups(groups,helper):
    require(len(groups)==268 and len({g['group_id'] for g in groups})==268,'All268 originalgroups')
    allrows=[]
    for g in groups:
        require(g['domain'] in DOMAINS and len(g['original_input_metadata8'])==8 and g['native_slot']==list(range(8)),'Domain/native8 group')
        rr=g['original_input_metadata8'];m=numeric_meta(rr);require(m['native_slot'].tolist()==list(range(8)),'Original native order')
        require(all(r['domain']==g['domain'] and r['group_id']==g['group_id'] and r['frame_index']==g['frame_index'] and r['id']==g['id'] for r in rr),'Group identity')
        require(same(m['original_query_xy'],np.asarray(g['original_query_xy8'],np.float64)) and same(m['original_query_xy'],np.asarray(g['actual_crop_grid_query_xy8'],np.float64)) and same(m['original_L'],np.asarray(g['original_L8'],np.float64)) and same(m['original_L'],np.asarray(g['actual_crop_grid_L8'],np.float64)),'OriginalF64 actualcropgrid identities, notFP32 aliases')
        require(m['original_query_available'].tolist()==g['original_query_available8'],'Factual originalU')
        box,why=helper['box_prompt'](m['original_query_xy'],m['original_query_available']);require((box is not None)==g['prompt_available'] and why==g['prompt_reason'] and (box.tolist() if box is not None else None)==g['box_xyxy'],'Exact finiteavailable q8 minmax/no clipping')
        require(g['prompt_GT_or_clip_or_expansion'] is False and isinstance(g['decoded_RGB_sha256'],str) and len(g['decoded_RGB_sha256'])==64,'No GT prompt and inherited decodedRGB SHA')
        require(len(g['raw_hw'])==2 and all(isinstance(n,int) and n>0 for n in g['raw_hw']),'Arbitrary originalHW')
        if g['domain']=='CURRENT223_1784':require(g['source_fullRGB']['kind']=='ORIGINAL_ENCODED_RGB_SAFE_ALIAS','Current original encoded source only')
        else:require(g['source_fullRGB']['kind']=='CANONICAL_RGB_NPY','Public original canonical source only')
        allrows.extend(rr)
    for domain,n in DOMAINS.items():
        gg=[g for g in groups if g['domain']==domain];rr=[r for g in gg for r in g['original_input_metadata8']];require(len(gg)==n['groups'] and len(rr)==n['rows'] and sum(not r['original_query_available'] for r in rr)==n['U'],'All domain counts/factualU')
        expected=list(range(1784)) if domain=='CURRENT223_1784' else list(range(584,944));require([r['original_global_index'] for r in rr]==expected,'Original domain-global IDs, no concat aliases')
        require(len({r['row_id'] for r in rr})==n['rows'],'Unique domain occurrence spelling')
    return allrows
def load_image(g,helper,alias_binding):
    source=g['source_fullRGB'];b=source['binding']
    if source['kind']=='ORIGINAL_ENCODED_RGB_SAFE_ALIAS':
        require(g['domain']=='CURRENT223_1784' and source['safe_alias_loader_source']==alias_binding and source['decoder']==dict(Pillow='12.2.0',numpy='1.26.4'),'Pinned current decoder/safealias source')
        path=helper['safe_original_image_path'](b)
        from PIL import Image
        with Image.open(path) as im:rgb=np.asarray(im.convert('RGB'),dtype=np.uint8).copy()
    else:
        require(g['domain']=='PUBLIC_HELD45_360' and source['kind']=='CANONICAL_RGB_NPY' and source['expected_dtype']=='uint8' and source['expected_shape']==g['raw_hw']+[3],'Pinned canonical publicRGB schema')
        path=check(b);rgb=np.load(safe_path(path),allow_pickle=False)
    require(rgb.dtype==np.uint8 and list(rgb.shape)==g['raw_hw']+[3],'Exact original RGBHWC shape/dtype')
    rgb=np.ascontiguousarray(rgb);require(ash(rgb)==g['decoded_RGB_sha256'],'Exact decodedRGB SHA');return rgb
def infer_image(predictor,rgb,box,torch,helper,counters):
    h,w=rgb.shape[:2]
    with torch.inference_mode():
        require(torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled(),'Strict before originalRGB encoder')
        counters['set_image_calls']+=1;predictor.set_image(rgb,image_format='RGB');counters['predict_calls']+=1;audit=dict(predict_call=counters['predict_calls']);counters['predict_warn_only_scopes'].append(audit)
        logits,quality,low=helper['predict_warn_only_call'](torch,lambda:predictor.predict(point_coords=None,point_labels=None,box=box.copy(),mask_input=None,multimask_output=False,return_logits=True),audit)
    require(logits.shape==(1,h,w) and quality.shape==(1,) and low.shape==(1,256,256),'SAM arbitraryHW original-size outputs')
    require(all(a.dtype==np.float32 and np.isfinite(a).all() for a in (logits,quality,low)),'FP32 finite rawSAMoutputs')
    return dict(image_logits=logits[0].copy(),lowres_logits=low[0].copy(),predicted_quality=quality.copy(),mask_logits_gt0=logits[0]>0)
def save_packet(path,a):
    require(set(a)==SAM_KEYS,'Exact12 pergroupSAMkeys')
    with path.open('xb') as f:np.savez_compressed(f,**a)
    with np.load(path,allow_pickle=False) as z:require(set(z.files)==SAM_KEYS and all(z[k].dtype==v.dtype and z[k].shape==v.shape and z[k].tobytes()==v.tobytes() for k,v in a.items()),'Saved12keyfullbyteparity')
    return bind(path)
def status(original,prompt,runtime):
    return (False,'ORIGINAL_INPUT_U') if not original else (False,'PROMPT_UNAVAILABLE') if not prompt else (False,'RUNTIME_UNAVAILABLE') if not runtime else (True,None)
def invented(helper):
    tests={}
    class Torch:
        strict=True;warn=False
        def are_deterministic_algorithms_enabled(self):return self.strict
        def is_deterministic_algorithms_warn_only_enabled(self):return self.warn
        def use_deterministic_algorithms(self,on,warn_only=False):self.strict=on;self.warn=warn_only
        def inference_mode(self):return contextlib.nullcontext()
    class Predictor:
        def set_image(self,rgb,image_format):require(image_format=='RGB','RGB API');self.hw=rgb.shape[:2]
        def predict(self,**kw):
            require(kw['point_coords'] is kw['point_labels'] is kw['mask_input'] is None and kw['multimask_output'] is False and kw['return_logits'] is True,'Box-only exactAPI');return np.zeros((1,)+self.hw,np.float32),np.array([.5],np.float32),np.zeros((1,256,256),np.float32)
    t=Torch();p=Predictor();ct=dict(set_image_calls=0,predict_calls=0,predict_warn_only_scopes=[]);box=np.array([-2.,0.,1300.,740.],np.float64);a=infer_image(p,np.zeros((23,41,3),np.uint8),box,t,helper,ct)
    tests['actual_infer_adapter_arbitraryHW_API_strictfinally']=a['image_logits'].shape==(23,41) and t.strict and not t.warn and ct['set_image_calls']==ct['predict_calls']==1
    tests['zero_logit_false_unclipped_originalbox_unchanged']=not a['mask_logits_gt0'].any() and box.tolist()==[-2.,0.,1300.,740.]
    class Bad(Predictor):
        def predict(self,**kw):raise RuntimeError('invented only')
    try:infer_image(Bad(),np.zeros((3,5,3),np.uint8),box,t,helper,ct)
    except RuntimeError:tests['exception_propagates_and_restores_strict']=t.strict and not t.warn
    else:tests['exception_propagates_and_restores_strict']=False
    q=np.array([[-2.,0.],[1300.,740.],[np.nan,np.nan],[-1,-1],[9999,9999],[1,2],[3,4],[4,5]],np.float64);av=np.array([1,1,0,0,0,0,0,0],bool);b,why=helper['box_prompt'](q,av)
    tests['availablefinite_only_q8_notGT_or_unavailable_extremes']=b.tolist()==[-2.,0.,1300.,740.]
    tests['allmissing_or_degenerate_prompt_not_factual_inputU']=helper['box_prompt'](q,np.zeros(8,bool))[0] is None and helper['box_prompt'](np.ones((8,2)),np.ones(8,bool))[0] is None and av.sum()==2
    rr=[dict(row_id='PUBLIC:'+str(k),query_xy=None if k==2 else [float(k),float(k+1)],L=None if k==2 else 96.,native_slot=k,original_global_index=584+k,original_query_available=k!=2) for k in range(8)];m=numeric_meta(rr)
    tests['metadata_nan_native_ID_rowspelling_U_preserved']=np.isnan(m['original_query_xy'][2]).all() and np.isnan(m['original_L'][2]) and m['native_slot'].tolist()==list(range(8)) and m['original_global_index'].tolist()==list(range(584,592)) and m['original_occurrence_id'][3]=='PUBLIC:3' and not m['original_query_available'][2]
    tests['all8_availability_truth_combinations_Ufirst']=sum(status(a,b,c)[0] for a in [False,True] for b in [False,True] for c in [False,True])==1 and status(False,True,True)==(False,'ORIGINAL_INPUT_U')
    packet=dict(a);packet.update(m);packet.update(box_xyxy=box,prediction_available=m['original_query_available'].copy());save_packet(H/'INVENTED_12KEY_PACKET_ONLY.npz',packet);tests['actual12keyserializer_fullbyteparity']=True
    with tempfile.TemporaryDirectory(prefix='v184_sam_alias_invented_') as td:
        target=Path(td)/'source';target.write_bytes(b'invented only');alias=Path(td)/'alias';alias.symlink_to(target);bb=dict(path=str(alias),sha256=sha(target));ok=helper['safe_original_image_path'](bb)==target
        bad=Path(td)/'bad';bad.symlink_to(Path(td)/'forbidden_v112'/'never-created')
        try:helper['safe_original_image_path'](dict(path=str(bad),sha256='0'*64))
        except AssertionError:rejected=True
        else:rejected=False
        tests['exact_alias_helper_sourcehash_and_forbidden_expansion_reject']=ok and rejected
    grid=np.ones((64,64),np.float32);tests['CPUones64_cumsum_integer_expectation_for_runtime_guard']=np.array_equal(grid.cumsum(0),np.repeat(np.arange(1,65,dtype=np.float32)[:,None],64,axis=1)) and np.array_equal(grid.cumsum(1),np.repeat(np.arange(1,65,dtype=np.float32)[None,:],64,axis=0))
    require(all(tests.values()),'Invented preparation tests');return {k:bool(v) for k,v in tests.items()}
def frozen_head_metadata(fit_binding,audit_binding):
    fit=read(check(fit_binding));audit=read(check(audit_binding));require(fit['completed'] and fit['PASS'] and fit['total_updates']==16000 and fit['updates_each']==8000 and fit['all2_frozen_before_outputs_or_evaluation_access'] and set(fit['arms'])==ARMS,'Both prescribed final8000heads frozen')
    require(audit['completed'] and audit['PASS'] and audit['actual_complete']==fit_binding,'Independent actualfit audit exactbinding')
    freeze=read(check(fit['all2_freeze']));require(freeze['completed'] and not freeze['saved_outputs_emitted'] and not freeze['validation_cache_or_targets_read'] and not freeze['common_diagnostic_inputs_or_targets_read'],'All2 freeze metadata')
    require(all(fit['arms'][arm]['checkpoint']==freeze['checkpoints'][arm] and fit['arms'][arm]['updates']==8000 for arm in ARMS),'Fixed finalhead receipt bindings only')
    return fit['all2_freeze']
def prepare():
    require(sha(SCOPE)==SCOPE_SHA and sha(CLARIFY)==CLARIFY_SHA and sha(FIT)==FIT_SHA and sha(FIT_AUDIT)==FIT_AUDIT_SHA,'Frozen scope/authority/finalfit')
    helper,old,pin,alias=helper_functions();c=read(INPUT_COMPLETE);require(c['completed'] and c['PASS'] and c['groups268']==268 and c['rows2144']==2144 and c['SAM_arrays_present'] is False,'Final completed originalinput adapter')
    require(c['code']==bind(INPUT_CODE) and c['protocol']==bind(INPUT_PROTOCOL) and c['fit_complete']==bind(FIT),'FinalCinput provenance')
    gg=rows(check(c['groups']));rr=validate_groups(gg,helper);tests=invented(helper);write(TESTS,dict(completed=True,PASS=True,code=bind(__file__),checks=tests,count=len(tests),actual_real_RGB_raw8_NPZ_weight_model_forward_or_annotation_reads=0))
    freeze=frozen_head_metadata(bind(FIT),bind(FIT_AUDIT));p=dict(completed=True,prepared_only=True,code=bind(__file__),scope=bind(SCOPE),authority_clarification=bind(CLARIFY),input_code=bind(INPUT_CODE),input_protocol=bind(INPUT_PROTOCOL),input_preparation=bind(INPUT_PREP),input_complete=bind(INPUT_COMPLETE),groups_manifest=c['groups'],frozen_fit=bind(FIT),independent_fit_audit=bind(FIT_AUDIT),all2_freeze=freeze,legacy_exact_helpers=bind(LEGACY),legacy_protocol=bind(OLD_PROTOCOL),source_pin=old['source_pin'],checkpoint=old['checkpoint'],runtime=old['runtime'],safe_alias_source=alias,tests=bind(TESTS),output=str(OUT),counts=dict(groups=268,rows=2144,domains=DOMAINS,prompt_unavailable_groups=sum(not g['prompt_available'] for g in gg)),
        model='Exact officialSAM1ViTB/checkpoint/runtime fromV180; evalFP32/gradFalse/noTF32/noqualitychoice. Bothnewheads alreadyfinal8000/audited, their weights notread orforwarded here.',
        prompt='Exact finalCbox_prompt AST: originalF64 finiteavailableinitialq8 minmax only, sentinel excluded. No GT/reference/nativecorrection/clipping/expansion, noFP32 alias substitution.',
        RGB='All268 original fullRGB verified evenpromptmissing. CURRENT exactV156 safealias+Pillow12.2 convertRGB/sourceencodedSHA+decodedSHA+HW. PUBLIC canonicalNPY uint8HWC/sourcefileSHA+decodedSHA+HW. No crop sampling, oldraw8/P3/features or annotation arrays opened.',
        runtime_policy='Pinned exact import_runtime/state_sha/predict_warn_only_call/actual_cuda_cumsum_guard AST. set_image strict; predict-only warn_only with finally strictrestore. Beforeimages CPUvsCUDAones64 cumsum byteguard. Oneforward/promptavailablegroup plusfixedfirstpromptavailable repeat, no oldSAM reuse or retries; model/checkpoint/source/input hashes unchanged.',
        output_schema='COMPLETE.groups orderedlist268 carrying exactoriginaldomain/groupmetadata+sourceRGBproof+outputs binding|null,prompt/runtime/reasons. Exact12groupNPZ raw4+boxF64/qF64/LF64/nativei8/originalglobali64/original_occurrence_id=Crow_id unchanged/originalav/predav. Rowstatus2144 separate. originalIDs overlap acrossdomains preserved; no concat ornewnative IDs. Firstrepeat packet also12keys.',
        availability='FactualU CURRENT45/PUBLIC108 unchanged. SAM predictionavailability originalav AND prompt AND runtime only; nofakezero masks fornullpackets. Later SAM96 pixel-domain availability separate, never a new classifier inputU.',
        preparation_scope='FinalCmetadata268groups/2144inputslots plusinventedarrays/text only. No actualrealRGB/raw8NPZ/weights/annotation/output bodies, no newmodel/classifierforward/fit.',training_admission=False,selected_model=None,fully_solved=False)
    write(PROTOCOL,p);template=H/'ROOT_ACTUAL_SAM_ENTRY_TEMPLATE_APPROVED_FALSE.json';write(template,dict(approved_frozen_real_sam=False,evaluation_only=True,both_heads_frozen=True,code=p['code'],protocol=bind(PROTOCOL),scope=p['scope'],authority_clarification=p['authority_clarification'],input_complete=p['input_complete'],frozen_fit=p['frozen_fit'],independent_fit_audit=p['independent_fit_audit'],checkpoint=dict(path=p['checkpoint']['path'],sha256=p['checkpoint']['sha256']),automatic_retry=False))
    write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=p['code'],protocol=bind(PROTOCOL),checks=p['tests'],groups=p['groups_manifest'],input_complete=p['input_complete'],entry_template=bind(template),actual_real_RGB_raw8_NPZ_weight_annotation_or_forward_reads=0))
    print(json.dumps(dict(PASS=True,prepared_only=True,code=p['code'],protocol=bind(PROTOCOL),preparation=bind(H/'PREPARATION_COMPLETE.json'))),flush=True)
def run(entry):
    e=read(entry);require(e.get('approved_frozen_real_sam') is True and e.get('evaluation_only') is True and e.get('both_heads_frozen') is True and e.get('automatic_retry') is False,'Separate root realSAM evaluation entry')
    for key,path in [('code',Path(__file__)),('protocol',PROTOCOL),('scope',SCOPE),('authority_clarification',CLARIFY)]:require(e[key]==bind(path),'Entry '+key)
    p=read(PROTOCOL)
    for key in ['input_complete','frozen_fit','independent_fit_audit']:require(e[key]==p[key],'Entry dependency '+key)
    for key in ['input_code','input_protocol','input_preparation','tests','groups_manifest','legacy_exact_helpers','legacy_protocol','source_pin','safe_alias_source']:check(p[key])
    require(sha(SCOPE)==SCOPE_SHA and sha(CLARIFY)==CLARIFY_SHA and sha(INPUT_CODE)==INPUT_CODE_SHA,'Frozen calculation/source scope');frozen_head_metadata(e['frozen_fit'],e['independent_fit_audit'])
    c=read(check(e['input_complete']));require(c['groups']==p['groups_manifest'] and c['fit_complete']==e['frozen_fit'] and c['PASS'],'Exact input completion');helper,old,pin,alias=helper_functions();groups=rows(check(p['groups_manifest']));validate_groups(groups,helper)
    require(helper['git']('rev-parse','HEAD')==pin['exact_commit'] and helper['git']('status','--porcelain','--untracked-files=no')=='','Official frozen clean source');[check(b) for b in pin['source_files']]
    checkpoint=dict(path=p['checkpoint']['path'],sha256=p['checkpoint']['sha256']);require(e['checkpoint']==checkpoint,'Pinned officialSAM checkpoint')
    require(not OUT.exists(),'Fresh output/nooverwrite orretry');OUT.mkdir();write(OUT/'STARTED.json',dict(entry=bind(entry),protocol=bind(PROTOCOL),groups268=268,rows2144=2144,PID=os.getpid(),startedUTC=datetime.now(timezone.utc).isoformat()))
    results=[dict(g,runtime_available=False,runtime_reason='NOT_ATTEMPTED',outputs=None,RGB_verified=False,observed_decoded_RGB_sha256=None) for g in groups];first=next((j for j,g in enumerate(groups) if g['prompt_available']),None)
    counters=dict(set_image_calls=0,predict_calls=0,predict_warn_only_scopes=[]);state_before=state_after=None;model=torch=None;repeat=dict(attempted=False,PASS=False,reason='NOT_ATTEMPTED');guard=None;fatal=None;start=time.monotonic()
    try:
        check(checkpoint);require(safe_path(checkpoint['path']).stat().st_size==p['checkpoint']['expected_bytes'],'Checkpointbytes')
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch,Predictor,registry,actual=helper['import_runtime']();require(actual==p['runtime'],'Pinned runtime');torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True,warn_only=False);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        require(torch.cuda.is_available(),'Pinned CUDA');guard=helper['actual_cuda_cumsum_guard'](torch);write(OUT/'CUDA_CUMSUM_PARITY_GUARD.json',guard);require(guard['PASS'],'Runtimecumsumguard')
        state=torch.load(checkpoint['path'],map_location='cpu',weights_only=True);require(isinstance(state,dict) and all(isinstance(x,torch.Tensor) for x in state.values()),'Official weight state only');model=registry['vit_b'](checkpoint=None);model.load_state_dict(state,strict=True);del state;model.requires_grad_(False).eval().float().to('cuda');require(not model.training and model.mask_threshold==0. and all(not t.requires_grad and t.dtype==torch.float32 for t in model.parameters()),'FrozenFP32model')
        state_before=helper['state_sha'](model);predictor=Predictor(model)
        for j,g in enumerate(results):
            try:
                rgb=load_image(g,helper,alias);g['RGB_verified']=True;g['observed_decoded_RGB_sha256']=ash(rgb)
                if not g['prompt_available']:g['runtime_reason']='PROMPT_UNAVAILABLE'
                else:
                    data=infer_image(predictor,rgb,np.asarray(g['box_xyxy'],np.float64),torch,helper,counters);data.update(numeric_meta(g['original_input_metadata8']));data.update(box_xyxy=np.asarray(g['box_xyxy'],np.float64),prediction_available=data['original_query_available'].copy());g['outputs']=save_packet(OUT/('GROUP_%04d_OUTPUTS.npz'%j),data);g['array_sha256']={k:ash(v) for k,v in data.items()};g['runtime_available']=True;g['runtime_reason']=None
            except Exception as exc:g['runtime_reason']=type(exc).__name__+': '+str(exc);g['failure_traceback']=traceback.format_exc()
            if (j+1)%8==0 or j+1==len(results):
                progress=dict(groups_done=j+1,groups268=268,predict_calls=counters['predict_calls'],runtime_failed=sum(s['prompt_available'] and not s['runtime_available'] for s in results[:j+1]),seconds=time.monotonic()-start);write(OUT/'PROGRESS.json',progress,False);print(json.dumps(progress),flush=True)
        if first is not None and results[first]['runtime_available']:
            g=results[first];rgb=load_image(g,helper,alias);data=infer_image(predictor,rgb,np.asarray(g['box_xyxy'],np.float64),torch,helper,counters);data.update(numeric_meta(g['original_input_metadata8']));data.update(box_xyxy=np.asarray(g['box_xyxy'],np.float64),prediction_available=data['original_query_available'].copy());comparison={k:ash(v)==g['array_sha256'][k] for k,v in data.items()};repeat=dict(attempted=True,group_id=g['group_id'],fixed_first_prompt_available_ordinal=first,PASS=all(comparison.values()),array_byte_equal=comparison,outputs=save_packet(OUT/'REPEAT_FIXED_FIRST_PROMPT_GROUP.npz',data))
        else:repeat=dict(attempted=False,PASS=False,reason='FIXED_FIRST_PROMPT_GROUP_UNAVAILABLE_NO_RETRY',ordinal=first)
    except BaseException as exc:fatal=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    try:state_after=helper['state_sha'](model) if model is not None and state_before is not None else None
    except Exception as exc:fatal=fatal or dict(type=type(exc).__name__,message='Finalstatehash '+str(exc))
    finalrows=[]
    for g in results:
        for r in g['original_input_metadata8']:
            available,reason=status(r['original_query_available'],g['prompt_available'],g['runtime_available']);finalrows.append(dict(r,prompt_available=g['prompt_available'],runtime_available=g['runtime_available'],prediction_available=available,prediction_U_reason=reason,SAM_group_output=g['outputs']))
    rp=OUT/'ALL2144_ORIGINAL_INPUTS_AND_SAM_STATUS.jsonl';writerows(rp,finalrows);unchanged=True
    try:
        for g in groups:
            if g['source_fullRGB']['kind']=='ORIGINAL_ENCODED_RGB_SAFE_ALIAS':helper['safe_original_image_path'](g['source_fullRGB']['binding'])
            else:check(g['source_fullRGB']['binding'])
        for b in [p['groups_manifest'],p['input_complete'],p['source_pin'],p['safe_alias_source'],checkpoint]:check(b)
        [check(b) for b in pin['source_files']]
    except Exception as exc:unchanged=False;fatal=fatal or dict(type=type(exc).__name__,message='Finalimmutablebinding '+str(exc))
    passed=fatal is None and all(g['RGB_verified'] and (g['runtime_available'] or not g['prompt_available']) for g in results) and repeat['PASS'] and state_before is not None and state_before==state_after and unchanged
    complete=dict(completed=True,PASS=passed,evaluation_only=True,entry=bind(entry),code=bind(__file__),protocol=bind(PROTOCOL),scope=p['scope'],authority_clarification=p['authority_clarification'],input_complete=p['input_complete'],groups_manifest=p['groups_manifest'],frozen_fit=p['frozen_fit'],independent_fit_audit=p['independent_fit_audit'],groups=results,group_count=268,rows=2144,all_rows=bind(rp),checkpoint=checkpoint,source_pin=p['source_pin'],source_commit=pin['exact_commit'],cuda_cumsum_guard=None if guard is None else bind(OUT/'CUDA_CUMSUM_PARITY_GUARD.json'),first_prompt_group_repeat=repeat,model_state_before_sha256=state_before,model_state_after_sha256=state_after,model_state_byte_unchanged=state_before is not None and state_before==state_after,input_checkpoint_source_hashes_unchanged=unchanged,actual_forward_call_counters=counters,original_inputU_by_domain={k:n['U'] for k,n in DOMAINS.items()},SAM_predictionU_by_domain={k:sum(not r['prediction_available'] for r in finalrows if r['domain']==k) for k in DOMAINS},all_original_occurrences_retained=True,original_ID_qL_native_availability_unchanged=True,GT_annotation_reference_score_classifier_forward_or_fit=0,new_crop_sampling=0,training_admission=False,no_automatic_retry=True,fatal=fatal,selected_model=None,fully_solved=False,seconds=time.monotonic()-start)
    write(OUT/'COMPLETE.json',complete)
    if not passed:raise SystemExit('Frozen realSAM execution failed; all statuses preserved, no automatic retry.')
    print(json.dumps(dict(PASS=True,complete=bind(OUT/'COMPLETE.json'))),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);a=ap.parse_args();prepare() if a.prepare else run(a.entry)
