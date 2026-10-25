"""V229 selected V227 CONTROL_FIXED_Q. Exact true entry before any payload.

원 q8/L/native/availability 보존, 고정 logit0, 새 학습·GT 입력·좌표 보정 없음.
"""
import argparse
import ast
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import sys
import time
import traceback
import types

sys.dont_write_bytecode=True
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
R=P/'publication_root_preparation'
PROTOCOL=R/'V229_SELECTED_FIXED_CONTEXT_REAL/CANDIDATE_AND_EVALUATION_PROTOCOL_BEFORE_CURRENT_INPUTS.json'
PROTOCOL_SHA='be3a9d24cee1b18458d3214bbd20833a37786af55d0c9bd2f16ed379129f6c41'
CLARIFICATION=R/'V229_SELECTED_FIXED_CONTEXT_REAL/CANDIDATE_AUTHORITY_CLARIFICATIONS.json'
CLARIFICATION_SHA='d4ff8ca2f86b62c23bfa804443932ab6126d7d6dbb115dd98296062c35de9619'
EXTRACTOR=R/'V227_FROZEN_FEATURE_CODE_C/extract_context_features.py'
EXTRACTOR_SHA='0279eca179150242b0542d371201cdeb6c32e06fa38df07d4333ae532ae0ed9f'
SOURCE_PROTOCOL_SHA='8f92b3da3d6e66df4e1bac2c7121ef975d2eb5f6b2c19ebf9215977881399462'
ROUTE=R/'V229_REAL_INPUT_ROUTE_C/INPUT_ROUTE.json'
ROUTE_SHA='e0295176597d3e7271e98429b3364587fd16b72b46671d34596741f25c480fda'
INPUT_MANIFEST=R/'V229_REAL_QA_B/current223_input_metadata_v1/INPUT_MANIFEST_CURRENT223_ONLY.json'
INPUT_MANIFEST_SHA='4a9d8e6c781d2116a22ad824d7e7e5f7bbb45e6458497e8cde3dea452d846dbd'
INPUT_QA=R/'V229_REAL_QA_B/current223_input_metadata_v1/INPUT_METADATA_QA.json'
INPUT_QA_SHA='5552585cec218e094a38006c6054e312380441897315e66f23f18d169627172f'
ALIAS_CODE=P/'query_visibility_separated_frozen_current_benchmark_20261024_v156/crop_review_preparation/render_fixed_saved_crops_alias_repair.py'
ALIAS_SHA='1489d086d1c72d9ff92c6b5677fbd6eeb8d56ff663f831134f8e58adfb0ff80c'
OUTPUT_NAMES={'rows':'ALL1784_FIXED_CONTEXT_PREDICTIONS.jsonl','input_facts':'INPUT_ONLY_FRAME_FACTS.json','freeze':'PREDICTIONS_FROZEN_BEFORE_ANNOTATIONS_OR_BASELINE_SCORES.json','complete':'COMPLETE.json'}
SOURCES={
 'local_crop':(P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py','6a6c2543e28715e3d47e15c543c25f75ff7d3efa3b881e523bd3451817fc004b',['crop']),
 'local_preprocess':(P/'query_visibility_frozen_rgb_head_20261023_v105/frozen_rgb_feature_preparation/prepare_frozen_rgb_features.py','cc1e2c331d41aaea5b3d42d144c9abbc40231dd3784d24f52dcc58f41d607578',['preprocess_views']),
 'context_math':(P/'query_visibility_target_context_rgb_20261023_v107/target_context_math.py','aa7cd8e2568c8b7d30827e8681b4b87899944bd23cfa1b27948ceda5b55d07f3',['query_extent','valid_box','target_context_grid','sample_target_context']),
 'context_preprocess':(P/'query_visibility_target_relative_rgb_head_20261023_v108/context_feature_preparation/prepare_target_context_features.py','8deae91931c12bcc1692c98ce74fa5e703cfb0d57da980b126d5ca45ca74873e',['preprocess_target']),
 'query_sample':(P/'query_visibility_spatial_target_query_features_20261023_v111/feature_preparation/prepare_spatial_query_features.py','cbf2f1cad553673b727c39fbe5c9c4263dfbdac35dba0aeea086f956e74ff19e',['sample_Q_raw','state_sha'])}
WEIGHT=P/'query_visibility_spatial_target_query_features_20261023_v111/assets/dinov2_vits14_reg4_pretrain.pth'
WEIGHT_SHA='f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb'
VENDOR=Path('/home/minjae/Documents/github/pallet-pose/outputs/pallet_type_selftrain_v1/selftrain_recovery_v1/dino_localization_assets/dinov2')
VENDOR_REVISION='7764ea0f912e53c92e82eb78a2a1631e92725fc8'
VENDOR_PINS={
 'LICENSE':'600cc67cc4cb2f5ea317dcfc687ad1c74dc4bec8782bbe9db0afd83513b935b7',
 'MODEL_CARD.md':'70ca59606bee0a5fbb1baec80e7e29a93cd7cfbe26ca1910c52a852c4aab09d0',
 'hubconf.py':'c1f5090e78ff940b72c076d2bf9c0310d1707c946b3d10e2d6f2b0bdf56a6f64',
 'dinov2/hub/backbones.py':'871fca671b12a9ff02e810654baf509e97ccf461bf8196ce5ddeefff2fd87d3e',
 'dinov2/hub/utils.py':'579613e3d7b82c2a387eddc215ad9f076b8740f9618229628395fe5755dbd131',
 'dinov2/models/vision_transformer.py':'7799a260f2d7d0fe197331d08502fb8c542f9b7424723650f6a39b64fa2639ea',
 'dinov2/layers/attention.py':'79c7be7a452b3aad96698ec38d5d5150b9f4d8ac084fa93324510dc9f624775d',
 'dinov2/layers/block.py':'60c0ac7dfa4474be313fabfa5a23d82faf6f0cecd4e720a88be35de9788cb636'}


def require(condition,message):
    if not condition:raise AssertionError(message)


def safe(path):
    p=Path(os.path.abspath(os.fspath(path)))
    require('v112' not in str(p).casefold(),'V112 lexical reject before FS')
    for q in reversed((p,*p.parents)):require(not q.is_symlink(),f'Symlink: {q}')
    return p


def read(path):
    fd=os.open(safe(path),os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as f:return f.read()


def digest(b):return hashlib.sha256(b).hexdigest()


def binding(path):
    b=read(path);return {'path':str(safe(path)),'sha256':digest(b),'size_bytes':len(b)}


def checked(b):
    raw=read(b['path']);require(digest(raw)==b['sha256'],f'SHA: {b["path"]}')
    if 'size_bytes' in b:require(len(raw)==b['size_bytes'],'Bound size')
    return raw


def identity(b):return str(Path(os.path.abspath(b['path']))),b['sha256']


def write(path,obj):
    with safe(path).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def pure_operators(np,torch,F):
    raw=read(EXTRACTOR);require(digest(raw)==EXTRACTOR_SHA,'Pinned source extractor')
    tree=ast.parse(raw);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='condition32')
    ns={'np':np,'torch':torch,'F':F,'MEAN':(.485,.456,.406),'STD':(.229,.224,.225),'hashlib':hashlib,'require':require}
    for path,sha,names in SOURCES.values():
        b=read(path);require(digest(b)==sha,'Pinned pure source')
        functions=[n for n in ast.parse(b).body if isinstance(n,ast.FunctionDef) and n.name in names]
        require({n.name for n in functions}==set(names),'Exact pure functions')
        exec(compile(ast.Module(body=functions,type_ignores=[]),str(path)+'::pure_only','exec'),ns)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(EXTRACTOR)+'::condition_only','exec'),ns)
    return ns


def runtime_metadata():
    return {'executable':'/home/minjae/anaconda3/envs/pallet-yolo26/bin/python','torch':'2.1.1+cu118','numpy':'1.26.4','CUDA':'11.8','encoder_device':'cuda','head_device':'cpu'}


def vendor_metadata():
    return {'path':str(VENDOR),'revision':VENDOR_REVISION,'files':[{'path':str(VENDOR/n),'sha256':h} for n,h in VENDOR_PINS.items()]}


def entry_template():
    p=json.loads(checked({'path':str(PROTOCOL),'sha256':PROTOCOL_SHA}))
    return {'approved_current_input_only_inference':False,'approved_source_preflight_only':False,'annotations_or_baseline_scores_allowed':False,'fit_allowed':False,'PnP_allowed':False,'automatic_retry':False,
        'code':binding(__file__),'protocol':{'path':str(PROTOCOL),'sha256':PROTOCOL_SHA},'clarification':{'path':str(CLARIFICATION),'sha256':CLARIFICATION_SHA},
        'source_extractor':{'path':str(EXTRACTOR),'sha256':EXTRACTOR_SHA},'selected_head':p['head'],'selected_checkpoint':p['checkpoint'],'source_training_complete':p['source_training_complete'],
        'runtime':runtime_metadata(),'vendor':vendor_metadata(),'encoder_weights':{'path':str(WEIGHT),'sha256':WEIGHT_SHA},
        'input_route':{'path':str(ROUTE),'sha256':ROUTE_SHA},'input_manifest':{'path':str(INPUT_MANIFEST),'sha256':INPUT_MANIFEST_SHA},'input_route_QA':{'path':str(INPUT_QA),'sha256':INPUT_QA_SHA},'input_loader_code':{'path':str(ALIAS_CODE),'sha256':ALIAS_SHA},
        'source_preflight':{'feature_complete':{'path':str(R/'V227_ROOT_CONTEXT_PROBE/context_features_v1/COMPLETE.json'),'sha256':'1e7fe53291e77588230949a14f62b5b78f407edf248347607d21a22c71f7ca11'},'frame_index':0,'feature_exact':True,'logit_atol':1e-5},
        'output_directory':None,'output_bindings':{k:None for k in OUTPUT_NAMES}}


def infer_frame(frame,rgb,encoder,head,op,np,torch,F,repeat_required,capture=None):
    """frame is input-only q8/availability + original all8 slots. RGB uint8 HWC."""
    require(rgb.dtype==np.uint8 and list(rgb.shape)==frame['raw_hw']+[3],'Original RGB uint8 HWC/shape')
    require(rgb.flags.c_contiguous,'Original contiguous RGB')
    norm=lambda x:F.normalize(x.float(),p=2,dim=-1,eps=1e-12)
    rows=[];pending=[];local=[];valid=[]
    for q in frame['queries']:
        row={'domain':q.get('domain','SOURCE_PREFLIGHT'),'id':q.get('id',frame.get('id')),'group_id':q.get('group_id',frame.get('group_id')),'global_index':q['global_index'],'original_global_index':q['global_index'],'row_id':q['row_id'],'frame_index':frame['frame_index'],'native_slot':q['native_slot'],'query_xy':q['query_xy'],'L':q['L'],'original_query_available':q['original_query_available'],'logit':None,'prediction':'U','U_reason':None}
        rows.append(row)
        if not q['original_query_available']:row['U_reason']='ORIGINAL_INPUT_UNAVAILABLE';continue
        xy=np.asarray(q['query_xy'],np.float64);L=q['L']
        if xy.shape!=(2,) or not np.isfinite(xy).all() or L is None or not np.isfinite(L) or L<=0:
            row['U_reason']='NONFINITE_OR_NONPOSITIVE_ORIGINAL_Q_L';continue
        a,v=op['crop'](rgb,xy,L,96)
        if not v.any():row['U_reason']='ALL_INVALID_ORIGINAL_LOCAL96';continue
        pending.append((q,row));local.append(a);valid.append(v)
    box=op['query_extent'](frame['query_xy'])
    facts={'frame_index':frame['frame_index'],'context_box_xyxy':box,'original_query_xy':frame['query_xy'],'original_query_available':frame['original_query_available'],'local_supported':len(pending),'encoder_called':False}
    if not pending:return rows,facts,False
    if box is None:
        for _,row in pending:row['U_reason']='INVALID_ORIGINAL_Q8_CONTEXT_BOX'
        return rows,facts,False
    context,context_valid=op['sample_target_context'](rgb,box)
    facts['context_valid_fraction']=float(context_valid.mean())
    if not context_valid.any():
        for _,row in pending:row['U_reason']='ALL_INVALID_ORIGINAL_Q8_CONTEXT'
        return rows,facts,False
    tv=op['preprocess_target'](torch.as_tensor(context.transpose(2,0,1)[None],device='cuda'),torch.as_tensor(context_valid[None,None],device='cuda'))
    lv,_=op['preprocess_views'](torch.as_tensor(np.stack(local),device='cuda'),torch.as_tensor(np.stack(valid),device='cuda'))
    views=torch.cat((tv,lv[:len(pending)]),0)
    with torch.inference_mode():
        z=encoder.forward_features(views)
        require(z['x_norm_clstoken'].shape==(len(pending)+1,384) and z['x_norm_patchtokens'].shape==(len(pending)+1,256,384),'Frozen reg4 descriptor shape')
        if repeat_required:
            zz=encoder.forward_features(views)
            require(all(torch.equal(z[k],zz[k]) for k in ('x_norm_clstoken','x_norm_patchtokens')),'First supported encoder batch repeat exact')
        cls=norm(z['x_norm_clstoken'][:1]);tokens=norm(z['x_norm_patchtokens'][:1]);local_cls=norm(z['x_norm_clstoken'][1:])
        fixed,inside=op['sample_Q_raw'](tokens.expand(len(pending),-1,-1),[q['query_xy'] for q,_ in pending],[box]*len(pending),[True]*len(pending))
        fixed=norm(fixed)
        supported=[j for j,x in enumerate(inside.cpu().numpy()) if bool(x)]
        for j,(q,row) in enumerate(pending):
            if j not in supported:row['U_reason']='FIXED_Q_READ_UNSUPPORTED_BY_ORIGINAL_CONTEXT'
        if supported:
            take=torch.tensor(supported,device='cuda',dtype=torch.int64)
            cond=np.stack([op['condition32'](frame,pending[j][0],np) for j in supported])
            b=np.asarray(box,np.float64)
            active=np.stack([(np.asarray(pending[j][0]['query_xy'],np.float64)-b[:2])/(b[2:]-b[:2]) for j in supported]).astype(np.float32)
            features=(local_cls[take].cpu(),fixed[take].cpu(),cls.expand(len(supported),-1).cpu(),tokens.expand(len(supported),-1,-1).cpu(),torch.from_numpy(cond),torch.from_numpy(active))
            require(all(torch.isfinite(t).all() for t in features),'Finite original input descriptors')
            logits=head(*features)
            if repeat_required:require(torch.equal(logits,head(*features)),'First supported frozen head batch repeat exact')
            require(torch.isfinite(logits).all(),'Finite frozen logits')
            for j,value in zip(supported,logits.numpy()):
                row=pending[j][1];row['logit']=float(value);row['prediction']='E' if value>=0 else 'D';row['U_reason']=None
            if capture is not None:
                capture.update({k:t.numpy() for k,t in zip(('localCLS','fixedQ','contextCLS','tokens','condition32','normalized_active_q'),features)})
                capture['row_id']=[pending[j][0]['row_id'] for j in supported]
                capture['logit']=logits.numpy()
    facts.update({'encoder_called':True,'local96_RGB_SHA':[digest(a.tobytes()) for a in local],'local96_valid_SHA':[digest(v.tobytes()) for v in valid]})
    return rows,facts,bool(supported)


def validate_current(data,np):
    frames,rows=data['frames223'],data['rows1784']
    require(len(frames)==223 and len(rows)==1784,'Exact current-only223/1784')
    same=lambda a,b:np.array_equal(np.asarray(a,np.float64),np.asarray(b,np.float64),equal_nan=True)
    result=[]
    for fi,g in enumerate(frames):
        require(g['domain']=='CURRENT223_1784' and g['frame_index']==fi and g['group_id']==f'CURRENT_FRAME_{fi:04d}','Current group identity')
        require(g['native_slot']==list(range(8)),'Original all8 natives')
        q8=np.asarray(g['original_query_xy8'],np.float64);L8=np.asarray(g['original_L8'],np.float64);av=g['original_query_available8']
        require(q8.shape==(8,2) and L8.shape==(8,) and len(av)==8 and all(type(a) is bool for a in av),'Original q8/L8/availability schema')
        rr=rows[fi*8:fi*8+8];group_rows=g['original_input_metadata8'];require(len(group_rows)==8,'Group metadata8')
        for k,(q,t) in enumerate(zip(rr,group_rows)):
            require(q==t,'Current group/row exact metadata')
            require(q['domain']=='CURRENT223_1784' and q['frame_index']==fi and q['domain_index']==fi*8+k and q['original_global_index']==fi*8+k and q['native_slot']==k,'Ordered current META')
            require(q['id']==g['id'] and q['group_id']==g['group_id'] and q['row_id']==g['id']+':native'+str(k),'Original ID/native identity')
            require(same(q['query_xy'],q8[k]) and same(q['actual_crop_grid_query_xy'],q8[k]) and same(q['L'],L8[k]) and same(q['actual_crop_grid_L'],L8[k]),'Exact F64 q/L and crop grid')
            require(q['original_query_available']==av[k] and type(q['original_query_available']) is bool,'Original availability')
        require(g['source_fullRGB']['kind']=='ORIGINAL_ENCODED_RGB_SAFE_ALIAS','Original encoded RGB alias authority')
        require(identity(g['source_fullRGB']['safe_alias_loader_source'])==(str(ALIAS_CODE),ALIAS_SHA),'Pinned original alias loader')
        require(isinstance(g['decoded_RGB_sha256'],str) and len(g['decoded_RGB_sha256'])==64,'Decoded original RGB SHA required')
        frame={'frame_index':fi,'id':g['id'],'group_id':g['group_id'],'raw_hw':g['raw_hw'],'query_xy':g['original_query_xy8'],'L_by_native':g['original_L8'],'original_query_available':av,
            'queries':[{**q,'global_index':q['original_global_index']} for q in rr],'source_fullRGB':g['source_fullRGB'],'decoded_RGB_sha256':g['decoded_RGB_sha256']}
        result.append(frame)
    require(sum(not q['original_query_available'] for q in rows)==45,'Original inputU45')
    require(len({r['row_id'] for r in rows})==1784,'Unique original rowIDs')
    return result


def alias_loader():
    raw=read(ALIAS_CODE);require(digest(raw)==ALIAS_SHA,'Pinned alias code')
    node=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='safe_original_image_path')
    def lexical(p):require('v112' not in os.fspath(p).casefold(),'Alias/target V112 lexical reject before FS')
    def lstat(p):lexical(p);return os.lstat(p)
    def readlink(p):
        lexical(p);target=os.readlink(p);lexical(target);return target
    ns={'os':types.SimpleNamespace(path=os.path,lstat=lstat,readlink=readlink),'stat':stat,'Path':Path,'sha':lambda p:digest(read(p))}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(ALIAS_CODE)+'::safe_alias_only','exec'),ns)
    return ns['safe_original_image_path']


def initialize(e,np,torch,F):
    require(sys.executable==e['runtime']['executable'] and str(torch.__version__)==e['runtime']['torch'] and np.__version__==e['runtime']['numpy'] and torch.version.cuda==e['runtime']['CUDA'],'Pinned V227 runtime')
    require(torch.cuda.is_available(),'Pinned CUDA encoder runtime')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    op=pure_operators(np,torch,F)
    for b in e['vendor']['files']:checked(b)
    weights_raw=checked(e['encoder_weights'])
    require(not any(k=='dinov2' or k.startswith('dinov2.') for k in sys.modules),'No alternate preloaded DINO vendor')
    sys.path.insert(0,str(VENDOR))
    from dinov2.hub.backbones import dinov2_vits14_reg
    encoder=dinov2_vits14_reg(pretrained=False)
    weights=torch.load(io.BytesIO(weights_raw),map_location='cpu',weights_only=True)
    require(tuple(weights['register_tokens'].shape)==(1,4,384),'Pinned reg4 state')
    encoder.load_state_dict(weights,strict=True);del weights,weights_raw
    encoder=encoder.cuda().float().eval().requires_grad_(False)
    require(sum(p.numel() for p in encoder.parameters())==22058112,'Pinned encoder parameters')
    head_raw=checked(e['selected_head']);require(digest(head_raw)=='042ffd9c7bced899e0de2546109cdaafff54db27b47d7cd24113eb056570c5dc','Selected CONTROL head code')
    ns={};exec(compile(head_raw,e['selected_head']['path'],'exec'),ns)
    head=ns['ContextReadProbe'](False)
    c=torch.load(io.BytesIO(checked(e['selected_checkpoint'])),map_location='cpu',weights_only=True)
    require(c['arm']=='CONTROL_FIXED_Q' and c['updates']==300 and c['seed']==17 and c['use_attention'] is False and c['parameters']==4561,'Selected frozen source state')
    require(c['protocol']['sha256']==SOURCE_PROTOCOL_SHA,'Original source protocol closure')
    head.load_state_dict(c['state_dict'],strict=True);head=head.cpu().float().eval().requires_grad_(False)
    require(sum(p.numel() for p in head.parameters())==4561 and len(head.state_dict())==6 and all(torch.isfinite(x).all() for x in head.state_dict().values()),'Strict finite selected head')
    training=json.loads(checked(e['source_training_complete']))
    require(training['completed'] and training['PASS'] and training['steps_each']==300 and training['parameters_each']==4561 and training['checkpoints']['CONTROL_FIXED_Q']['sha256']==e['selected_checkpoint']['sha256'],'Published source final closure')
    return encoder,head,op


def source_preflight(s,encoder,head,op,np,torch,F):
    require(s['frame_index']==0 and s['feature_exact'] is True and s['logit_atol']==1e-5,'Predeclared first SOURCE frame replay')
    require(identity(s['feature_complete'])==(str(R/'V227_ROOT_CONTEXT_PROBE/context_features_v1/COMPLETE.json'),'1e7fe53291e77588230949a14f62b5b78f407edf248347607d21a22c71f7ca11'),'Exact SOURCE feature COMPLETE')
    complete=json.loads(checked(s['feature_complete']));require(complete['completed'] and complete['PASS'],'Existing source features')
    frames=[json.loads(line) for line in checked(complete['input_manifest']).splitlines() if line]
    with np.load(io.BytesIO(checked(complete['features'])),allow_pickle=False) as z:
        ix=np.flatnonzero(z['frame_index']==0);require(len(ix)>0,'Fixed SOURCE frame has original eligible batch')
        expected={k:z[k][ix] for k in ('localCLS','fixedQ','condition32','normalized_active_q')}
        ci=int(z['context_index'][ix[0]]);expected['contextCLS']=np.repeat(z['contextCLS'][ci:ci+1],len(ix),axis=0);expected['tokens']=np.repeat(z['tokens'][ci:ci+1],len(ix),axis=0)
        ids=z['row_id'][ix].tolist()
    f=next(f for f in frames if f['original_frame_index']==0)
    qmap={q['row_id']:q for q in f['queries']}
    queries=[{k:qmap[rid][k] for k in ('row_id','native_slot','query_xy','L','original_query_available')} for rid in ids]
    for q in queries:q['global_index']=qmap[q['row_id']]['original_global_index']
    frame={'frame_index':0,'id':f['source_id'],'raw_hw':f['raw_hw'],'query_xy':f['query_xy'],'original_query_available':f['original_query_available'],'queries':queries}
    rgb=np.load(io.BytesIO(checked(f['canonical_RGB_NPY'])),allow_pickle=False)
    require(digest(np.ascontiguousarray(rgb).tobytes())==f['image']['decoded_RGB_sha256'],'Original SOURCE RGB decoded SHA')
    capture={};rows,facts,supported=infer_frame(frame,rgb,encoder,head,op,np,torch,F,True,capture)
    require(supported and capture['row_id']==ids and all(r['prediction']!='U' for r in rows),'Same SOURCE eligible batch')
    require(all(np.array_equal(capture[k],expected[k]) for k in expected),'Original SOURCE descriptor/operator exact replay')
    with torch.inference_mode():old_logits=head(*(torch.from_numpy(np.ascontiguousarray(expected[k])) for k in ('localCLS','fixedQ','contextCLS','tokens','condition32','normalized_active_q'))).numpy()
    err=float(np.max(np.abs(capture['logit']-old_logits)))
    require(err<=1e-5 and np.array_equal(capture['logit']>=0,old_logits>=0),'Fixed SOURCE head logit parity')
    return {'completed':True,'PASS':True,'source_frame_index':0,'original_eligible_batch_rows':len(ids),'row_id':ids,'same_source_batch_configuration':True,'descriptors_exact':True,'head_maxabs_logit':err,'head_signs_exact':True,'first_batch_repeat_exact':True,'canonical_RGB':f['canonical_RGB_NPY'],'facts':facts,'new_training_or_optimizer_updates':0}


def run(entry_path):
    raw=read(entry_path);e=json.loads(raw)
    current=e['approved_current_input_only_inference'] is True
    require(current != (e['approved_source_preflight_only'] is True),'One explicit execution scope')
    require(all(e[k] is False for k in ('annotations_or_baseline_scores_allowed','fit_allowed','PnP_allowed','automatic_retry')),'Input-only frozen inference gate')
    require(identity(e['code'])==identity(binding(__file__)),'Exact runner code');checked(e['code'])
    require(identity(e['protocol'])==(str(PROTOCOL),PROTOCOL_SHA) and identity(e['clarification'])==(str(CLARIFICATION),CLARIFICATION_SHA),'Pinned selected protocol/clarification')
    protocol=json.loads(checked(e['protocol']));checked(e['clarification'])
    require(protocol['selected_arm']=='CONTROL_FIXED_Q' and protocol['no_more_training_this_phase'] is True,'Fixed selected arm')
    for key,source in [('selected_head','head'),('selected_checkpoint','checkpoint'),('source_training_complete','source_training_complete')]:require(identity(e[key])==identity(protocol[source]),'Selected source pin '+key)
    require(identity(e['source_extractor'])==(str(EXTRACTOR),EXTRACTOR_SHA),'Original extractor pin');checked(e['source_extractor'])
    require(e['runtime']==runtime_metadata() and e['vendor']==vendor_metadata() and identity(e['encoder_weights'])==(str(WEIGHT),WEIGHT_SHA),'Exact V227 encoder/runtime')
    if current:
        for key,path,sha in [('input_route',ROUTE,ROUTE_SHA),('input_manifest',INPUT_MANIFEST,INPUT_MANIFEST_SHA),('input_route_QA',INPUT_QA,INPUT_QA_SHA),('input_loader_code',ALIAS_CODE,ALIAS_SHA)]:require(identity(e[key])==(str(path),sha),'Current-only input pin '+key)
        checked(e['input_route']);qa=json.loads(checked(e['input_route_QA']));require(qa['completed'] and qa['PASS'],'Independent current-only metadata QA')
    out=safe(e['output_directory']);require(out.parent==PROTOCOL.parent and not out.exists(),'Fresh direct-child output')
    require(e['output_bindings']=={k:str(out/n) for k,n in OUTPUT_NAMES.items()},'Fixed output closure')
    os.environ['XFORMERS_DISABLED']='1';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import numpy as np
    import torch
    from torch.nn import functional as F
    out.mkdir();write(out/'STARTED.json',{'entry':{'path':str(safe(entry_path)),'sha256':digest(raw)},'code':e['code'],'protocol':e['protocol'],'annotations_or_scores_read':0,'automatic_retry':False})
    try:
        started=time.monotonic();encoder,head,op=initialize(e,np,torch,F)
        before={'encoder':op['state_sha'](encoder),'head':op['state_sha'](head)}
        preflight=source_preflight(e['source_preflight'],encoder,head,op,np,torch,F)
        write(out/'SOURCE_FRAME0_PREPROCESS_AND_LOGIT_REPLAY.json',preflight)
        if not current:
            after={'encoder':op['state_sha'](encoder),'head':op['state_sha'](head)};require(before==after,'Frozen states unchanged')
            write(out/'SOURCE_PREFLIGHT_COMPLETE.json',{'completed':True,'PASS':True,'entry':{'path':str(safe(entry_path)),'sha256':digest(raw)},'code':e['code'],'replay':binding(out/'SOURCE_FRAME0_PREPROCESS_AND_LOGIT_REPLAY.json'),'states_before':before,'states_after':after,'current_inputs_read':0,'current_forward':0})
            print(json.dumps({'completed':True,'PASS':True,'source_preflight':binding(out/'SOURCE_PREFLIGHT_COMPLETE.json')}),flush=True);return
        frames=validate_current(json.loads(checked(e['input_manifest'])),np)
        checked(e['input_loader_code']);load_alias=alias_loader()
        from PIL import Image,__version__ as PIL_VERSION
        all_rows=[];facts=[];repeated=False
        for frame in frames:
            source=frame['source_fullRGB'];image=source['binding']
            require('v112' not in image['path'].casefold(),'Original alias before FS')
            actual_path=load_alias(image)
            rgb=np.asarray(Image.open(actual_path).convert('RGB'),dtype=np.uint8).copy()
            require(digest(rgb.tobytes())==frame['decoded_RGB_sha256'],'Original decoded RGB identity despite decoder-version differences')
            rows,info,supported=infer_frame(frame,rgb,encoder,head,op,np,torch,F,not repeated)
            if supported:repeated=True
            all_rows.extend(rows)
            info.update({'id':frame['id'],'group_id':frame['group_id'],'original_image':image,'safe_canonical_path':str(actual_path),'original_decoder_metadata':source['decoder'],'actual_decoder':{'library':'Pillow','version':PIL_VERSION,'operation':"Image.open(...).convert('RGB')"},'decoded_RGB_sha256':frame['decoded_RGB_sha256'],'decoded_RGB_exact':True,'q8_L_U_binding_sha256':digest(json.dumps({'q8':frame['query_xy'],'L8':frame['L_by_native'],'available8':frame['original_query_available']},sort_keys=True,allow_nan=False).encode())})
            facts.append(info)
            if (frame['frame_index']+1)%25==0:print(json.dumps({'frames':frame['frame_index']+1,'total':223}),flush=True)
        require(len(all_rows)==1784 and [r['original_global_index'] for r in all_rows]==list(range(1784)),'All original output rows retained')
        require(sum(not r['original_query_available'] for r in all_rows)==45 and all(r['prediction']=='U' and r['U_reason']=='ORIGINAL_INPUT_UNAVAILABLE' for r in all_rows if not r['original_query_available']),'InputU45 preserved')
        require(all((r['logit'] is None and r['prediction']=='U' and r['U_reason']) or (r['logit'] is not None and np.isfinite(r['logit']) and r['prediction'] in ('D','E') and r['U_reason'] is None) for r in all_rows),'Finite D/E or explicit U')
        require(repeated or not any(f['encoder_called'] for f in facts),'First supported current batch repeat')
        after={'encoder':op['state_sha'](encoder),'head':op['state_sha'](head)}
        require(before==after and all(not p.requires_grad for m in (encoder,head) for p in m.parameters()) and all(not m.training for model in (encoder,head) for m in model.modules()),'Frozen states and eval preserved')
        with safe(out/OUTPUT_NAMES['rows']).open('x',encoding='utf-8') as f:
            for row in all_rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        write(out/OUTPUT_NAMES['input_facts'],{'completed':True,'PASS':True,'frames':facts,'first_supported_current_batch_repeat_exact':repeated,'Pillow_actual_version':PIL_VERSION,'all_original_encoded_and_decoded_RGB_SHA_exact':True,'no_current_RAW_NPZ_or_GT_or_score_read':True})
        rows_binding=binding(out/OUTPUT_NAMES['rows']);facts_binding=binding(out/OUTPUT_NAMES['input_facts'])
        freeze={'completed':True,'PASS':True,'predictions_frozen':True,'selected_arm':'CONTROL_FIXED_Q','threshold_logit':0,'rows1784':rows_binding,'input_facts':facts_binding,'selected_checkpoint':e['selected_checkpoint'],'selected_head':e['selected_head'],'entry':{'path':str(safe(entry_path)),'sha256':digest(raw)},'states_before':before,'states_after':after,'annotations_or_baseline_scores_read':0,'new_training_updates':0}
        write(out/OUTPUT_NAMES['freeze'],freeze)
        write(out/OUTPUT_NAMES['complete'],{'completed':True,'PASS':True,'selected_arm':'CONTROL_FIXED_Q','threshold_logit':0,'code':e['code'],'protocol':e['protocol'],'clarification':e['clarification'],'entry':freeze['entry'],'selected_checkpoint':e['selected_checkpoint'],'selected_head':e['selected_head'],'encoder_weights':e['encoder_weights'],'input_manifest':e['input_manifest'],'input_route_QA':e['input_route_QA'],'rows':rows_binding,'input_facts':facts_binding,'prediction_freeze':binding(out/OUTPUT_NAMES['freeze']),'source_preflight':binding(out/'SOURCE_FRAME0_PREPROCESS_AND_LOGIT_REPLAY.json'),'frames':223,'rows1784':1784,'original_input_U':45,'predicted_U':sum(r['prediction']=='U' for r in all_rows),'states_before':before,'states_after':after,'first_supported_current_batch_repeat_exact':repeated,'annotations_or_baseline_scores_read':0,'new_training_or_PnP_or_GT_input':0,'feature_cache_stored':False,'seconds':time.monotonic()-started})
        print(json.dumps({'completed':True,'PASS':True,'COMPLETE':binding(out/OUTPUT_NAMES['complete'])}),flush=True)
    except BaseException:
        write(out/'FAILED_NO_AUTOMATIC_RETRY.json',{'completed':False,'PASS':False,'traceback':traceback.format_exc(),'automatic_retry':False});raise


def self_test():
    import numpy as np
    import torch
    from torch.nn import functional as F
    op=pure_operators(np,torch,F)
    path=P/'query_visibility_cue_occlusion_20261023_v72/private_evaluation/current_input_adapter.py'
    raw=read(path);require(digest(raw)=='69efc742474527d9f60f70d4ef2549a891d4e0179a8493735db140e2aa7b97ab','Pinned original crop code')
    node=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='crop96')
    ns={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path)+'::crop_only','exec'),ns)
    rgb=np.arange(17*19*3,dtype=np.uint16).reshape(17,19,3).astype(np.uint8)
    for q,L in [([5.25,6.75],12.),([-.45,.2],16.),([18.7,16.3],20.)]:
        a,v=op['crop'](rgb,q,L,96);b,w=ns['crop96'](rgb,q,L)
        require(np.array_equal(a,b) and np.array_equal(v,w),'V90 and original V72 crop parity')
    q8=np.arange(16,dtype=float).reshape(8,2).tolist();av=[True]*8;av[2]=False
    frame={'frame_index':0,'id':'SYNTHETIC','raw_hw':[17,19],'query_xy':q8,'original_query_available':av,'queries':[]}
    q={'native_slot':3,'query_xy':q8[3],'L':2.}
    c=op['condition32'](frame,q,np)
    require(c.shape==(32,) and c.dtype==np.float32 and c[3]==1 and c[:8].sum()==1 and c[8+2]==c[8+3]==0 and np.array_equal(c[16:].reshape(8,2)[[2,3]],np.zeros((2,2))),'Original condition32/self/U identity')
    unavailable={**frame,'queries':[{'global_index':k,'row_id':f'SYNTHETIC:native{k}','native_slot':k,'query_xy':q8[k],'L':16.,'original_query_available':False} for k in range(8)]}
    rows,_,supported=infer_frame(unavailable,rgb,None,None,op,np,torch,F,False)
    require(not supported and all(r['prediction']=='U' and r['U_reason']=='ORIGINAL_INPUT_UNAVAILABLE' for r in rows),'Original U never forwarded')
    invalid={**frame,'query_xy':[[5.,5.]]*8,'queries':[{'global_index':0,'row_id':'SYNTHETIC:native0','native_slot':0,'query_xy':[5.,5.],'L':16.,'original_query_available':True}]}
    rows,_,supported=infer_frame(invalid,rgb,None,None,op,np,torch,F,False)
    require(not supported and rows[0]['U_reason']=='INVALID_ORIGINAL_Q8_CONTEXT_BOX','Invalid context U without forced E')
    outside={**frame,'queries':[{'global_index':0,'row_id':'SYNTHETIC:native0','native_slot':0,'query_xy':[1e6,1e6],'L':16.,'original_query_available':True}]}
    rows,_,supported=infer_frame(outside,rgb,None,None,op,np,torch,F,False)
    require(not supported and rows[0]['U_reason']=='ALL_INVALID_ORIGINAL_LOCAL96','All-invalid local U without forced E')
    try:safe('/V112_FORBIDDEN_SYNTHETIC_PATH')
    except AssertionError:pass
    else:raise AssertionError('V112 lexical guard')
    print(json.dumps({'completed':True,'PASS':True,'synthetic_arrays_only':True,'checks':['source_V90_vs_current_V72_crop_exact','original_condition32_self_U','inputU_no_forward','invalid_context_U','allinvalid_local_U','V112_before_FS'],'actual_input_or_weight_or_RGB_or_GT_reads':0,'encoder_or_head_instantiation':0,'optimizer_steps':0},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--entry');g.add_argument('--entry-template',action='store_true');g.add_argument('--self-test',action='store_true');a=parser.parse_args()
    if a.self_test:self_test()
    elif a.entry_template:print(json.dumps(entry_template(),ensure_ascii=False,indent=2))
    else:run(a.entry)
