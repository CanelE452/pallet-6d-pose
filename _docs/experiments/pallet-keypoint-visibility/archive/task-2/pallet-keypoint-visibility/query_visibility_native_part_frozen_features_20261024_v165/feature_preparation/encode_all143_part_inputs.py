"""Exact C111/MR frozen1928 source counterfactual features. Prepare is pixel-free.
Actual source payload/weight/cache bodies require separate approved root entry.
"""
import os,sys
os.environ['XFORMERS_DISABLED']='1';os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');sys.dont_write_bytecode=True
from pathlib import Path
import argparse,ast,hashlib,json,time,traceback
import numpy as np

H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
SCOPE=V/'ROOT_SCOPE_BEFORE_FROZEN_PART_FEATURE_PREPARATION.json'
SCOPE_SHA='9f09d2ecffada8211b11b275ae14d2d98121b440d0dfdc71768e86c30c4e4147'
OLD=P/'query_visibility_mr6d_full_public_source_20261024_v153'
OLD_COMPLETE=OLD/'full230_native_DINO_v1/FEATURE_COMPLETE.json'
PROTOCOL=H/'PROTOCOL_BEFORE_ANY_ENCODER_OR_INPUT_BODIES.json'
INPUTS=H/'PAYLOAD143_AND_STATUS198_INPUT_BINDINGS_BEFORE_BODIES.json'
GENERATED=H/'GENERATED_INVENTED_FEATURE_CHECKS.json'
OUT=V/'frozen_part_features_v1'
N=1144;PAYLOADS=143;VARIANTS=('BEFORE','COVER','ADJACENT_SHAM')
PIXEL_KEYS=('rgb96','valid96','central32','target224','target_valid','fractions','query_xy','L','native_slot','query_id','crop_constructed','native_continuous_valid','nearest_valid','central4_available','context_available')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def pure(path,names,ns):
    tree=ast.parse(Path(path).read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path)+'::exact_pure_AST','exec'),ns);return ns
def operators(p):
    # Same V147/V153 pure-AST mechanism, only required C111/C8/C9 functions.
    # No current adapter, native-grid initializer, model module or source data.
    import torch
    from torch.nn import functional as F
    ns=dict(np=np,torch=torch,F=F,hashlib=hashlib)
    pure(check(p['C111_producer']),('sample_Q_raw','assemble','state_sha'),ns)
    c8=pure(check(p['C8_preprocess']),('preprocess_views',),dict(torch=torch,F=F,MEAN=(.485,.456,.406),STD=(.229,.224,.225)))
    c9=pure(check(p['C9_target_preprocess']),('preprocess_target',),dict(torch=torch,F=F,MEAN=(.485,.456,.406),STD=(.229,.224,.225)))
    ns['preprocess_views']=c8['preprocess_views'];ns['preprocess_target']=c9['preprocess_target'];return ns
def views17(a,op,device):
    import torch
    # Match the original V153 contiguous CHW local and original target transpose.
    local=np.ascontiguousarray(a['rgb96'].transpose(0,3,1,2));valid=np.ascontiguousarray(a['valid96'][:,None])
    lv,_=op['preprocess_views'](torch.as_tensor(local,device=device),torch.as_tensor(valid,device=device))
    target=a['target224'];vt=a['target_valid']
    tv=op['preprocess_target'](torch.as_tensor(target.transpose(2,0,1)[None],device=device),torch.as_tensor(vt[None,None],device=device))
    z=torch.cat((tv,lv),0)
    assert z.shape==(17,3,224,224) and z.dtype==torch.float32 and torch.isfinite(z).all()
    return z
def assembly_signature(a,box):
    # Final-feature reuse requires every non-token assembly operand, not views alone.
    values=[('query_xy',a['query_xy']),('context_box',np.asarray(box,np.float64)),('fractions',a['fractions']),('central4_available',a['central4_available']),('crop_constructed',a['crop_constructed'])]
    out=b''
    for name,z in values:
        z=np.ascontiguousarray(z);out+=name.encode()+str(z.dtype).encode()+str(z.shape).encode()+z.tobytes()
    return out
def reusable(cached,viewbytes,assemblybytes,provenance):
    return cached['view_bytes']==viewbytes and cached['assembly_bytes']==assemblybytes and cached['provenance']==provenance
def assemble_tokens(z,a,box,op):
    import torch
    cl,pt=z['x_norm_clstoken'],z['x_norm_patchtokens'];q=a['query_xy']
    x,inside=op['assemble'](cl[:1].repeat_interleave(8,0),pt[:1].repeat_interleave(8,0),cl[1:9],pt[1:9],cl[9:],q,[box]*8,a['fractions'],a['central4_available'])
    x=torch.where(torch.as_tensor(a['crop_constructed'],device=x.device)[:,None],x,torch.zeros_like(x))
    return x.cpu().numpy(),inside
def input_validate(a,g,frame):
    rr=g['rows'];assert len(rr)==8
    assert a['rgb96'].shape==(8,96,96,3) and a['rgb96'].dtype==np.uint8
    assert a['valid96'].shape==(8,96,96) and a['valid96'].dtype==bool
    assert a['target224'].shape==(224,224,3) and a['target224'].dtype==np.float32 and a['target_valid'].shape==(224,224) and a['target_valid'].dtype==bool
    assert exact(a['central32'],a['rgb96'][:,32:64,32:64]) and a['fractions'].dtype==np.float32
    assert np.isfinite(a['target224']).all() and (a['target224']>=0).all() and (a['target224']<=255).all()
    for k,r in enumerate(rr):
        assert int(a['native_slot'][k])==r['native_slot']==k and str(a['query_id'][k])==r['original_query_id']
        assert a['query_xy'][k].tolist()==r['query_xy']==frame['query_xy'][k] and float(a['L'][k])==r['L']==frame['L'][k]
        for key,mkey in (('crop_constructed','feature_constructed'),('native_continuous_valid','query_available'),('nearest_valid','nearest_valid'),('central4_available','central4_available'),('context_available','context_available')):assert bool(a[key][k])==r[mkey]
    fractions=np.array([[np.float32(a['target_valid'].mean()),np.float32(a['valid96'][k].mean()),np.float32(a['valid96'][k,32:64,32:64].mean())] for k in range(8)],np.float32)
    assert exact(fractions,a['fractions'])
    assert np.array_equal(a['central4_available'],a['valid96'][:,47:49,47:49].all((1,2)))
    assert np.array_equal(a['crop_constructed'],frame['feature_constructed']) and np.array_equal(a['native_continuous_valid'],frame['native_continuous_valid'])
    return True
def prepare_inputs(scope):
    complete=read(check(scope['source_pixel_complete']));audit=read(check(scope['source_saved_array_audit']))
    source=lines(check(scope['source_manifest']));original=read(check(scope['source_inputs']))
    assert complete['completed'] and complete['all66_attempts_retained'] and complete['proofs']==scope['source_manifest']
    assert audit['completed'] and audit['PASS'] and audit['source_generation_complete']==scope['source_pixel_complete']
    assert audit['all66_manifest']==scope['source_manifest'] and audit['saved_NPZ_files']==194
    assert len(source)==len(original['attempts'])==66
    fm={f['frame_index']:f for f in original['frames']};groups=[];statuses=[];rr=[]
    for pair_index,(r,assignment) in enumerate(zip(source,original['attempts'])):
        assert r['pair_id']==assignment['pair_id'] and r['region']==assignment['region']
        fr=fm[r['frame_index']];assert fr['split']=='SOURCE_DEVELOPMENT'
        for variant in VARIANTS:
            vb=r['variants'][variant];present='path' in vb
            start=len(rr) if present else None
            status=dict(status_index=len(statuses),pair_index=pair_index,pair_id=r['pair_id'],case_id=r['case_id'],
                variant=variant,original_frame_index=r['frame_index'],source_id=fr['source_id'],split=fr['split'],
                pair_pixel_proof_PASS=bool(r['PASS']),generation_reason=r['reason'],payload=vb if present else None,
                constructed_payload=present,feature_index=start,feature_indices=list(range(start,start+8)) if present else None,
                missing_reason=None if present else vb['reason'],training_admitted=False)
            statuses.append(status)
            if not present: continue
            for k in range(8):
                fact=fr['facts'][k];originalq=fr['original_query_input_metadata'][k]
                assert fact['native_slot']==fr['native_slot'][k]==k and fact['query_id']==fr['query_id'][k]
                assert fact['query_xy']==fr['query_xy'][k] and fact['L']==fr['L'][k]
                rr.append(dict(occurrence_index=len(rr),input_occurrence_id=f"V165::{r['pair_id']}::{variant}::native{k}",
                    pair_id=r['pair_id'],case_id=r['case_id'],variant=variant,variant_payload=vb,
                    original_query_id=fr['query_id'][k],original_global_index=fr['global_index'][k],
                    original_frame_index=r['frame_index'],source_id=fr['source_id'],split=fr['split'],native_slot=k,
                    query_xy=fr['query_xy'][k],L=fr['L'][k],original_context_box_xyxy=fr['context_box_xyxy'],
                    query_available=bool(fr['native_continuous_valid'][k]),nearest_valid=bool(fr['nearest_valid'][k]),
                    central4_available=bool(fr['local_central4_available'][k]),context_available=bool(fr['context_available'][k]),
                    feature_constructed=bool(fr['feature_constructed'][k]),
                    original_input_flags={z:bool(originalq[z]) for z in ('no_detection','finite_non_sentinel','query_RGB96_centre_domain_valid','native_grid_any_valid')},
                    native_visibility_target=-1,point_quality_target=-1,owner_target=-1,
                    pair_pixel_proof_PASS=bool(r['PASS']),generation_reason=r['reason'],training_admitted=False))
            groups.append(dict(payload_index=len(groups),occurrence_start=start,status_index=status['status_index'],variant=variant,
                pair_id=r['pair_id'],original_frame_index=r['frame_index'],payload=vb,context_box_xyxy=fr['context_box_xyxy'],rows=rr[start:start+8]))
    ids={g['original_frame_index'] for g in groups};frames=[fr for fr in original['frames'] if fr['frame_index'] in ids]
    assert len(statuses)==198 and sum(not s['constructed_payload'] for s in statuses)==55
    assert len(groups)==143 and len(rr)==1144 and len({r['input_occurrence_id'] for r in rr})==1144
    assert len(frames)==17 and len({g['original_frame_index'] for g in groups if g['variant']=='BEFORE'})==17
    assert sum(g['variant']=='BEFORE' for g in groups)==51 and sum(g['variant']!='BEFORE' for g in groups)==92
    assert sum(not r['query_available'] for r in rr)==178 and all(r['feature_constructed'] for r in rr)
    assert sum(g['variant']=='BEFORE' and not g['rows'][0]['pair_pixel_proof_PASS'] for g in groups)==5
    return dict(completed=True,payload_count=143,occurrences=N,groups=groups,frames=frames,statuses198=statuses,inputU=178,
        missing_variants=55,failed_BEFORE_payloads_retained=5,source_manifest=scope['source_manifest'],
        source_inputs=scope['source_inputs'],source_NPZ_weight_or_feature_bodies_read=False)
def generated(p):
    import torch
    torch.set_num_threads(1);op=operators(p)
    a=dict(rgb96=np.arange(8*96*96*3,dtype=np.uint8).reshape(8,96,96,3),valid96=np.ones((8,96,96),bool),target224=np.full((224,224,3),110,np.float32),target_valid=np.ones((224,224),bool),query_xy=np.array([[10.,10.],[20.,10.],[20.,20.],[10.,20.],[12.,12.],[18.,12.],[18.,18.],[12.,18.]],np.float64),fractions=np.ones((8,3),np.float32),central4_available=np.ones(8,bool),crop_constructed=np.ones(8,bool))
    a['central32']=a['rgb96'][:,32:64,32:64].copy();box=[10.,10.,20.,20.]
    v=views17(a,op,'cpu');checks={'view_shape17':tuple(v.shape)==(17,3,224,224),'view_dtype_FP32':v.dtype==torch.float32}
    local=torch.from_numpy(np.ascontiguousarray(a['rgb96'].transpose(0,3,1,2)));vm=torch.from_numpy(a['valid96'][:,None]);lv,_=op['preprocess_views'](local,vm)
    tv=op['preprocess_target'](torch.from_numpy(a['target224'].transpose(2,0,1)[None]),torch.from_numpy(a['target_valid'][None,None]))
    checks['target1_full8_central8_exact']=exact(v.numpy(),torch.cat((tv,lv),0).numpy())
    checks['contiguous_CHW_matches_original_V153']=local.is_contiguous()
    b={k:x.copy() if isinstance(x,np.ndarray) else x for k,x in a.items()};b['valid96'][:,0:5]=False;b['rgb96'][:,0:5]=0
    w=views17(b,op,'cpu');b['rgb96'][:,0:5]=255;checks['invalid_local_pixels_meanfill_invariance']=exact(w.numpy(),views17(b,op,'cpu').numpy())
    b={k:x.copy() if isinstance(x,np.ndarray) else x for k,x in a.items()};b['target_valid'][:5]=False;b['target224'][:5]=0
    w=views17(b,op,'cpu');b['target224'][:5]=255;checks['invalid_context_meanfill_invariance']=exact(w.numpy(),views17(b,op,'cpu').numpy())
    cl=torch.arange(17*384,dtype=torch.float32).reshape(17,384)+1;pt=torch.arange(17*256*384,dtype=torch.float32).reshape(17,256,384)/1e4+1
    x,inside=assemble_tokens(dict(x_norm_clstoken=cl,x_norm_patchtokens=pt),a,box,op)
    checks['feature_shape1928']=x.shape==(8,1928);checks['feature_finite_FP32']=x.dtype==np.float32 and np.isfinite(x).all()
    checks['Qinside_final_scalar_exact']=np.array_equal(x[:,1927],inside.astype(np.float32))
    checks['five_descriptors_unit_norm']=all(np.allclose(np.linalg.norm(x[:,i:i+384],axis=1),1,rtol=1e-6,atol=1e-6) for i in range(0,1920,384))
    b={k:z.copy() if isinstance(z,np.ndarray) else z for k,z in a.items()};b['query_xy'][0]=[-99.,-99.];b['central4_available'][1]=False;b['crop_constructed'][2]=False
    y,yi=assemble_tokens(dict(x_norm_clstoken=cl,x_norm_patchtokens=pt),b,box,op)
    checks['outside_Q_zero_without_row_drop']=not yi[0] and not y[0,384:768].any() and y.shape==(8,1928)
    checks['invalid_center4_P_zero']=not y[1,1536:1920].any();checks['unconstructed_all1928_zero']=not y[2].any()
    vb=v.numpy().tobytes();ab=assembly_signature(a,box);cache=dict(view_bytes=vb,assembly_bytes=ab,provenance='PINNED')
    checks['exact_reuse_accepted']=reusable(cache,vb,ab,'PINNED')
    checks['different_viewbytes_rejected']=not reusable(cache,vb+b'x',ab,'PINNED')
    checks['different_assembly_rejected']=not reusable(cache,vb,assembly_signature(b,box),'PINNED')
    checks['different_encoder_provenance_rejected']=not reusable(cache,vb,ab,'OTHER')
    for key in ('query_xy','fractions','central4_available','crop_constructed'):
        z={k:t.copy() if isinstance(t,np.ndarray) else t for k,t in a.items()};z[key].flat[0]=not z[key].flat[0] if z[key].dtype==bool else z[key].flat[0]+.01
        checks['reuse_requires_'+key]=not reusable(cache,vb,assembly_signature(z,box),'PINNED')
    checks['reuse_requires_contextbox']=not reusable(cache,vb,assembly_signature(a,[9.,10.,20.,20.]),'PINNED')
    # The forward preprocessor receives only explicit RGB/valid arrays; extras inert.
    b={**a,'pasted_alpha':np.ones((3,3),bool),'owner_target':np.array([1]),'native_slot':np.arange(8)[::-1]}
    checks['alpha_owner_native_metadata_not_preprocessor_input']=exact(v.numpy(),views17(b,op,'cpu').numpy())
    assert all(bool(z) for z in checks.values()),[k for k,z in checks.items() if not z]
    return dict(completed=True,PASS=True,invented_only=True,checks={k:bool(v) for k,v in checks.items()},number_checks=len(checks),source_input_pixel_feature_checkpoint_bodies_read=0,encoder_model_fit_calls=0,runtime=dict(executable=sys.executable,torch=str(torch.__version__),numpy=np.__version__))
def prepare():
    assert sha(SCOPE)==SCOPE_SHA;scope=read(SCOPE);assert not scope['actual_encoder_allowed_by_this_file'] and not scope['fit_allowed']
    previous=read(check(scope['previous_feature_protocol']));check(scope['previous_frozen_feature_producer'])
    parent=read(check(previous['parent_protocol']));inputs=prepare_inputs(scope);write(INPUTS,inputs)
    p=dict(completed=True,prepared_only=True,code=bind(__file__),scope=bind(SCOPE),
        previous_feature_protocol=scope['previous_feature_protocol'],previous_frozen_feature_producer=scope['previous_frozen_feature_producer'],
        source_pixel_complete=scope['source_pixel_complete'],source_saved_array_audit=scope['source_saved_array_audit'],
        source_manifest=scope['source_manifest'],source_inputs=scope['source_inputs'],source_generator_protocol=scope['source_generator_protocol'],
        parent_protocol=previous['parent_protocol'],original_feature_complete=previous['original_feature_complete'],
        original_features=previous['original_features'],inputs=bind(INPUTS),output=str(OUT),
        feature_layout=parent['feature_layout'],expected_encoder_state=parent['expected_encoder_state'],runtime=parent['runtime'],
        **{k:parent[k] for k in ('C111_producer','C8_preprocess','C9_target_preprocess','helper_code','weight','official_repo','official_revision','official_source_bindings')},
        occurrences=N,payloads=143,attempts=66,potential_variant_statuses=198,missing_variants=55,
        unique_BEFORE_frames=17,CF_payloads=92,actual_encoder_calls=109,BEFORE_duplicate_reuses=34,
        failed_BEFORE_payloads_retained=5,views_per_encoder_call=17,view_order='target1/full8/central8',
        forward_inputs=previous['forward_inputs'],availability='Original MR continuous-only query_available;178 inputU preserved. Construct/nearest/context/central4/Qinside separate; no filtering or U resurrection.',
        reuse=previous['reuse'],before_parity='Actual encode all17 unique BEFORE x8 byte-equal original V153 TRAIN features, all8RGB96/valid, target224 hashes and original17view preprocessing replay.',
        missing_status='All198 statuses retained;55 absent payloads carry null feature_index/feature_indices and reason. No fake zero features for missing payloads.',
        original_shared_container_access='Actual extraction hashes/opens shared V153 NPZ and decompresses each specified1840-row input-only feature/metadata member; immediately selects only17 explicitly bound TRAIN frames x8 rows. No held-feature statistics/labels or selection.',
        weight_feature_or_source_payload_bodies_read_during_preparation=0,encoder_model_fit_calls_during_preparation=0,
        automatic_retry=False,training_admitted=False)
    g=generated(p)
    g['checks'].update(metadata_all198_statuses_retained=len(inputs['statuses198'])==198,
        metadata_missing55_null_no_fake_features=sum(s['feature_index'] is None and s['feature_indices'] is None and s['payload'] is None and bool(s['missing_reason']) for s in inputs['statuses198'])==55,
        metadata_all143_payloads_including_failed5_BEFORE=len(inputs['groups'])==143 and inputs['failed_BEFORE_payloads_retained']==5,
        metadata_occurrences_dense_unique1144=[r['occurrence_index'] for x in inputs['groups'] for r in x['rows']]==list(range(N)),
        metadata_original_continuous_inputU178=inputs['inputU']==178)
    assert all(g['checks'].values());g['number_checks']=len(g['checks']);write(GENERATED,g)
    p['generated']=bind(GENERATED);write(PROTOCOL,p)
    template=H/'ROOT_FEATURE_ENTRY_TEMPLATE_APPROVED_FALSE.json'
    write(template,dict(approved_actual_frozen_feature_extraction=False,code=p['code'],scope=p['scope'],protocol=bind(PROTOCOL),inputs=p['inputs'],generated=p['generated'],source_only=True,fit_allowed=False,classifier_allowed=False,initializer_allowed=False,current_allowed=False,V112_allowed=False,PnP_allowed=False,GT_pose_owner_label_reads_allowed=False,automatic_retry=False,output=str(OUT)))
    write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=p['code'],protocol=bind(PROTOCOL),inputs=bind(INPUTS),generated=bind(GENERATED),entry_template=bind(template),
        input_occurrences=N,variant_payloads=143,potential_statuses=198,missing_variants=55,planned_encoder_calls=109,planned_BEFORE_reuses=34,
        source_payload_feature_weight_bodies_read=0,actual_encoder_or_fit_calls=0,training_admitted=False))
    print(json.dumps(dict(code=bind(__file__),protocol=bind(PROTOCOL),preparation=bind(H/'PREPARATION_COMPLETE.json'))),flush=True)
def run(entry):
    e=read(entry);assert e['approved_actual_frozen_feature_extraction'] is True and e['source_only'] is True
    for k in ('fit_allowed','classifier_allowed','initializer_allowed','current_allowed','V112_allowed','PnP_allowed','GT_pose_owner_label_reads_allowed','automatic_retry'):assert e[k] is False
    p=read(check(e['protocol']));assert e['protocol']==bind(PROTOCOL) and p['code']==bind(__file__) and sha(SCOPE)==SCOPE_SHA
    for k in ('code','scope','inputs','generated'):assert e[k]==p[k];check(p[k])
    for k in ('previous_feature_protocol','previous_frozen_feature_producer','source_pixel_complete','source_saved_array_audit','source_manifest','source_inputs','source_generator_protocol','parent_protocol','original_feature_complete','C111_producer','C8_preprocess','C9_target_preprocess','helper_code'):check(p[k])
    for b in p['official_source_bindings']:check(b)
    assert read(check(p['generated']))['PASS'] and not OUT.exists() and e['output']==p['output']==str(OUT)
    inputs=read(check(p['inputs']));groups=inputs['groups'];frames={f['frame_index']:f for f in inputs['frames']}
    assert len(groups)==143 and len(frames)==17 and len(inputs['statuses198'])==198
    import torch
    assert str(torch.__version__)==p['runtime']['torch']=='2.1.1+cu118' and np.__version__=='1.26.4' and Path(sys.executable).resolve()==Path(p['runtime']['executable']).resolve() and torch.cuda.is_available()
    assert sys.version==p['runtime']['python'] and torch.version.cuda==p['runtime']['CUDA'] and torch.cuda.get_device_name(0)==p['runtime']['GPU']
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    OUT.mkdir();write(OUT/'STARTED.json',dict(completed=False,pid=os.getpid(),entry=bind(entry),protocol=bind(PROTOCOL),automatic_retry=False));started=time.monotonic()
    try:
        op=operators(p);sys.path.insert(0,p['official_repo']);from dinov2.hub.backbones import dinov2_vits14_reg
        encoder=dinov2_vits14_reg(pretrained=False);weights=torch.load(check(p['weight']),map_location='cpu',weights_only=True);assert tuple(weights['register_tokens'].shape)==(1,4,384)
        encoder.load_state_dict(weights,strict=True);del weights;encoder=encoder.cuda().float().eval().requires_grad_(False)
        assert sum(t.numel() for t in encoder.parameters())==22058112
        state_before=op['state_sha'](encoder);assert state_before==p['expected_encoder_state']
        # Shared1840-row container is actually decompressed; only named TRAIN rows retained/compared.
        allowed_ids=np.array(sorted(j for f in frames.values() for j in f['global_index']),np.int64)
        assert allowed_ids.shape==(136,) and len(set(allowed_ids.tolist()))==136
        old={};container_members=('features','global_index','frame_index','native_slot','query_id','query_xy','L','query_available','feature_constructed')
        with np.load(check(p['original_features']),allow_pickle=False) as z:
            for key in container_members:
                full=z[key]
                if key=='features':assert full.shape==(1840,1928) and full.dtype==np.float32
                assert len(full)==1840
                old[key]=full[allowed_ids].copy();del full
        assert old['global_index'].tolist()==allowed_ids.tolist()
        old_lookup={int(g):j for j,g in enumerate(allowed_ids)}
        shared_access=dict(shared_container=p['original_features'],container_rows=1840,decompressed_members=list(container_members),
            compared_TRAIN_frames=sorted(frames),retained_TRAIN_global_indices=allowed_ids.tolist(),retained_rows=136,
            full_container_decompression_disclosed=True,held_feature_statistics_or_labels_inspected=False)
        provenance=hashlib.sha256(json.dumps({k:p[k] for k in ('C111_producer','C8_preprocess','C9_target_preprocess','runtime','expected_encoder_state')},sort_keys=True).encode()).hexdigest()
        cache={};features=[];allinside=[];allfractions=[];facts=[];reuses=[];beforeparity=[];lineage=[];calls=0;checked=[]
        for g in groups:
            frame=frames[g['original_frame_index']]
            with np.load(check(g['payload']),allow_pickle=False) as z:a={k:z[k].copy() for k in PIXEL_KEYS}
            input_validate(a,g,frame);checked.append(g['payload']);box=g['context_box_xyxy'];views=views17(a,op,'cuda');viewarray=views.cpu().numpy();viewbytes=viewarray.tobytes();assemblybytes=assembly_signature(a,box)
            first_before=g['variant']=='BEFORE' and g['original_frame_index'] not in cache
            if g['variant']=='BEFORE' and not first_before:
                c=cache[g['original_frame_index']];assert reusable(c,viewbytes,assemblybytes,provenance),'BEFORE duplicate input/assembly/provenance mismatch; fail closed'
                x=c['features'].copy();inside=c['inside'].copy();reuses.append(dict(payload_index=g['payload_index'],pair_id=g['pair_id'],variant='BEFORE',original_frame_index=g['original_frame_index'],reused_payload_index=c['payload_index'],exact17view_bytes=True,exact_assembly_operand_bytes=True,provenance_sha256=provenance,view17_FP32_sha256=ash(viewarray),assembly_operands_sha256=hashlib.sha256(assemblybytes).hexdigest(),features_sha256=ash(x)))
            else:
                with torch.inference_mode():z=encoder.forward_features(views);calls+=1;x,inside=assemble_tokens(z,a,box,op)
            if first_before:
                ids=frame['global_index'];ii=np.asarray([old_lookup[j] for j in ids],np.int64)
                assert old['global_index'][ii].tolist()==ids and old['frame_index'][ii].tolist()==[g['original_frame_index']]*8 and old['native_slot'][ii].tolist()==list(range(8))
                assert old['query_id'][ii].tolist()==frame['query_id'] and exact(old['query_xy'][ii],a['query_xy']) and exact(old['L'][ii],a['L']) and np.array_equal(old['query_available'][ii],a['native_continuous_valid']) and np.array_equal(old['feature_constructed'][ii],a['crop_constructed'])
                assert exact(x,old['features'][ii]),'BEFORE actual feature1928 parity failed; preserve failure/no retry'
                with np.load(check(frame['existing_factual96_packet']),allow_pickle=False) as z:original_rgb=z['rgb96'].copy();original_valid=z['valid96'].copy()
                assert exact(original_rgb,a['rgb96']) and exact(original_valid,a['valid96'])
                assert all(ash(a['target224'])==f['target224_F32_HWC_sha256'] and ash(a['target_valid'])==f['target_valid_sha256'] for f in frame['facts'])
                assert all(ash(a['rgb96'][j])==f['RGB96_HWC_sha256'] and ash(a['valid96'][j])==f['valid96_sha256'] for j,f in enumerate(frame['facts']))
                original_a={**a,'rgb96':original_rgb,'valid96':original_valid};original_views=views17(original_a,op,'cuda').cpu().numpy();assert exact(viewarray,original_views)
                cache[g['original_frame_index']]=dict(view_bytes=viewbytes,assembly_bytes=assemblybytes,provenance=provenance,features=x.copy(),inside=inside.copy(),payload_index=g['payload_index'])
                beforeparity.append(dict(original_frame_index=g['original_frame_index'],global_indices=ids,all8_features1928_byte_equal=True,all8_RGB96_valid96_byte_equal=True,target224_original_fact_hash_equal=True,preprocess17view_original_replay_byte_equal=True,view17_FP32_sha256=ash(viewarray),features_sha256=ash(x),original_saved_features_sha256=ash(old['features'][ii])))
            assert x.shape==(8,1928) and x.dtype==np.float32 and np.isfinite(x).all() and not x[~a['crop_constructed']].any()
            assert np.array_equal(x[:,1927],inside.astype(np.float32)) and not x[~inside,384:768].any()
            features.append(x);allinside.extend(inside);allfractions.extend(a['fractions'])
            facts.append(dict(payload_index=g['payload_index'],pair_id=g['pair_id'],variant=g['variant'],original_frame_index=g['original_frame_index'],payload=g['payload'],input_array_sha256={k:ash(a[k]) for k in PIXEL_KEYS},preprocess17view_FP32_sha256=ash(viewarray),assembly_operands_sha256=hashlib.sha256(assemblybytes).hexdigest(),features_all8_sha256=ash(x),actual_encoder_call=not(g['variant']=='BEFORE' and not first_before),encoder_provenance_sha256=provenance))
            for k,r in enumerate(g['rows']):lineage.append({**r,'feature_index':g['occurrence_start']+k,'payload_index':g['payload_index'],'unit_Q_inside':bool(inside[k]),'feature1928_sha256':ash(x[k])})
            if (g['payload_index']+1)%10==0:
                progress=dict(pid=os.getpid(),payloads=g['payload_index']+1,rows=len(lineage),encoder_calls=calls,BEFORE_reuses=len(reuses),elapsed_seconds=time.monotonic()-started)
                (OUT/'PROGRESS.json').write_text(json.dumps(progress,indent=2)+'\n');print(json.dumps(progress),flush=True)
        assert calls==109 and len(cache)==len(beforeparity)==17 and len(reuses)==34 and len(features)==143 and len(lineage)==N
        x=np.concatenate(features);rr=[r for g in groups for r in g['rows']];available=np.array([r['query_available'] for r in rr],bool);constructed=np.array([r['feature_constructed'] for r in rr],bool)
        assert (~available).sum()==178 and constructed.all() and x.shape==(1144,1928)
        state_after=op['state_sha'](encoder);assert state_before==state_after and all(not t.requires_grad for t in encoder.parameters()) and all(not m.training for m in encoder.modules())
        for b in checked+[p['weight'],p['original_features']]:check(b)
        output=OUT/'PART1144_FEATURES_INPUT_ONLY.npz'
        arrays=dict(features=x,occurrence_index=np.arange(N,dtype=np.int32),input_occurrence_id=np.array([r['input_occurrence_id'] for r in rr]),pair_id=np.array([r['pair_id'] for r in rr]),variant=np.array([r['variant'] for r in rr]),original_query_id=np.array([r['original_query_id'] for r in rr]),original_global_index=np.array([r['original_global_index'] for r in rr],np.int32),original_frame_index=np.array([r['original_frame_index'] for r in rr],np.int32),source_id=np.array([r['source_id'] for r in rr]),split=np.array([r['split'] for r in rr]),native_slot=np.array([r['native_slot'] for r in rr],np.int8),query_xy=np.array([r['query_xy'] for r in rr],np.float64),L=np.array([r['L'] for r in rr],np.float64),query_available=available,native_continuous_valid=available,nearest_valid=np.array([r['nearest_valid'] for r in rr],bool),feature_constructed=constructed,local_central4_available=np.array([r['central4_available'] for r in rr],bool),context_available=np.array([r['context_available'] for r in rr],bool),support_fractions=np.asarray(allfractions,np.float32),context_box_xyxy=np.array([r['original_context_box_xyxy'] for r in rr],np.float64),unit_Q_inside=np.asarray(allinside,bool),occurrence_unit=np.arange(N,dtype=np.int32))
        arrays.update(native_visibility_target=np.full(N,-1,np.int8),point_quality_target=np.full(N,-1,np.int8),owner_target=np.full(N,-1,np.int8))
        for name in ('no_detection','finite_non_sentinel','query_RGB96_centre_domain_valid','native_grid_any_valid'):
            arrays['original_'+name]=np.array([r['original_input_flags'][name] for r in rr],bool)
        np.savez_compressed(output,**arrays)
        def jsonl(name,rows):
            path=OUT/name
            with path.open('x') as f:
                for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            return bind(path)
        facts_binding=jsonl('PAYLOAD143_INPUT_PREPROCESS_FEATURE_FACTS.jsonl',facts);lineage_binding=jsonl('OCCURRENCE_FEATURE_LINEAGE1144.jsonl',lineage)
        status_binding=jsonl('ALL198_VARIANT_STATUS_AND_FEATURE_INDEX.jsonl',inputs['statuses198'])
        parity=OUT/'BEFORE17_ORIGINAL_FEATURE_PARITY.json';write(parity,dict(completed=True,PASS=True,frames=17,rows=136,original_features=p['original_features'],all17_actual_encoder_calls=True,shared_container_access=shared_access,records=beforeparity))
        reusefile=OUT/'BEFORE34_EXACT_REUSE_RECORDS.json';write(reusefile,dict(completed=True,PASS=True,reuses=34,records=reuses))
        write(OUT/'FEATURE_COMPLETE.json',dict(completed=True,PASS=True,code=p['code'],scope=p['scope'],protocol=bind(PROTOCOL),entry=bind(entry),inputs=p['inputs'],features=bind(output),payload_facts=facts_binding,occurrence_lineage=lineage_binding,all198_variant_statuses=status_binding,BEFORE_original_parity=bind(parity),BEFORE_reuse=bind(reusefile),rows=N,payloads=143,pairs=66,potential_variant_statuses=198,missing_variants=55,feature_shape=[1144,1928],inputU=178,feature_constructed=1144,failed_BEFORE_payloads_retained=5,actual_encoder_calls=calls,views_per_call=17,BEFORE_unique_calls=17,CF_calls=92,BEFORE_reuses=34,shared_original_container_access=shared_access,frozen_encoder_state_before=state_before,frozen_encoder_state_after=state_after,encoder_state_unchanged=True,only_RGB17views_encoder_input=True,alpha_owner_native_labels_GT_pose_not_encoder_inputs=True,original_qL_native_split_availability_unchanged=True,all_occurrences_preserved=True,UNKNOWN_targets_never_used=True,classifier_fit_initializer_current_V112_PnP_calls=0,training_admitted=False,automatic_retry=False,seconds=time.monotonic()-started))
        print(json.dumps(dict(completed=True,feature_complete=bind(OUT/'FEATURE_COMPLETE.json'))),flush=True)
    except BaseException:
        write(OUT/'FAILED_EXECUTION.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
    if a.prepare and not a.entry:prepare()
    elif a.entry and not a.prepare:run(a.entry)
    else:raise SystemExit('Prepare only first; root --entry required for actual bodies/encoder')
