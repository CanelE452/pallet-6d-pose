"""Exact C111/MR frozen1928 source counterfactual features. Prepare is pixel-free.
Actual source payload/weight/cache bodies require separate approved root entry.
"""
import os,sys
os.environ['XFORMERS_DISABLED']='1';os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');sys.dont_write_bytecode=True
from pathlib import Path
import argparse,ast,hashlib,json,time,traceback
import numpy as np

H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
SCOPE=V/'ROOT_SCOPE_BEFORE_FEATURE_PREPARATION.json'
SCOPE_SHA='77a0e727b715f93a51fdf6828956101acf911696bdd80f63d7b608aad64260f7'
OLD=P/'query_visibility_mr6d_full_public_source_20261024_v153'
OLD_COMPLETE=OLD/'full230_native_DINO_v1/FEATURE_COMPLETE.json'
PROTOCOL=H/'PROTOCOL_BEFORE_ANY_ENCODER_OR_INPUT_BODIES.json'
INPUTS=H/'PAYLOAD477_INPUT_BINDINGS_BEFORE_BODIES.json'
GENERATED=H/'GENERATED_INVENTED_FEATURE_CHECKS.json'
OUT=V/'frozen_cf_features_v1'
N=3816;PAYLOADS=477;VARIANTS=('BEFORE','COVER','ADJACENT_SHAM')
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
    meta=read(check(scope['metadata_complete']));rr=lines(check(meta['manifest']));frames=read(check(meta['original_source_frame_metadata']))['frames']
    assert len(rr)==N and len(frames)==46 and meta['variant_payloads']==477 and meta['inputU']==360 and meta['all_native_visibility_and_quality_UNKNOWN']
    groups=[]
    for i in range(0,N,8):
        rows=rr[i:i+8];a=rows[0];assert [r['native_slot'] for r in rows]==list(range(8))
        assert all(r['pair_id']==a['pair_id'] and r['variant']==a['variant'] and r['variant_payload']==a['variant_payload'] and r['original_frame_index']==a['original_frame_index'] and r['split']=='SOURCE_DEVELOPMENT' and r['native_visibility_target']==r['point_quality_target']==-1 and not r['training_admitted'] for r in rows)
        assert a['variant']==VARIANTS[(i//8)%3]
        groups.append(dict(payload_index=i//8,occurrence_start=i,variant=a['variant'],pair_id=a['pair_id'],original_frame_index=a['original_frame_index'],payload=a['variant_payload'],context_box_xyxy=a['original_context_box_xyxy'],rows=rows))
    assert len(groups)==477 and len({g['pair_id'] for g in groups})==159
    assert sum(not r['query_available'] for r in rr)==360 and all(r['feature_constructed'] for r in rr)
    assert len({g['original_frame_index'] for g in groups if g['variant']=='BEFORE'})==46
    assert len({g['pair_id'] for g in groups if not g['rows'][0]['pair_pixel_proof_PASS']})==2
    return dict(completed=True,payload_count=477,occurrences=N,groups=groups,frames=frames,manifest=meta['manifest'],metadata_complete=scope['metadata_complete'],original_source_frame_metadata=meta['original_source_frame_metadata'],inputU=360,failed_pairs_retained=2,source_NPZ_weight_or_feature_bodies_read=False)
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
    parent=read(check(scope['original_feature_protocol']));oldcomplete=read(OLD_COMPLETE);inputs=prepare_inputs(scope);write(INPUTS,inputs)
    p=dict(completed=True,prepared_only=True,code=bind(__file__),scope=bind(SCOPE),requirements=scope['requirements'],source_phase=scope['source_phase'],metadata_complete=scope['metadata_complete'],parent_protocol=scope['original_feature_protocol'],original_feature_complete=bind(OLD_COMPLETE),original_features=oldcomplete['features'],inputs=bind(INPUTS),output=str(OUT),feature_layout=parent['feature_layout'],expected_encoder_state=parent['expected_encoder_state'],runtime=parent['runtime'],**{k:parent[k] for k in ('C111_producer','C8_preprocess','C9_target_preprocess','helper_code','weight','official_repo','official_revision','official_source_bindings')},occurrences=N,payloads=477,unique_BEFORE_frames=46,CF_payloads=318,actual_encoder_calls=364,BEFORE_duplicate_reuses=113,views_per_encoder_call=17,view_order='target1/full8/central8',forward_inputs='RGB-only ImageNet-normalized FP3217views; geometric validity meanfill only; no alpha/owner/native_slot/labels/GT/pose forwarded',availability='Original MR continuous-only query_available;360 inputU preserved. Other mask components separate; no row filtering/no U resurrection',reuse='BEFORE per original frame only, exact17view tensor bytes + q8/box/fractions/central4/constructed assembly bytes + pinned operator/encoder/runtime provenance; record every duplicate',before_parity='Actual encode all46uniqueBEFORE x8 byte-equal original V153 features, all8RGB96/valid and original target224 hashes + original17view preprocess replay',weight_feature_or_source_payload_bodies_read_during_preparation=0,encoder_model_fit_calls_during_preparation=0,automatic_retry=False,training_admitted=False)
    g=generated(p);write(GENERATED,g);p['generated']=bind(GENERATED);write(PROTOCOL,p)
    template=H/'ROOT_FEATURE_ENTRY_TEMPLATE_APPROVED_FALSE.json';write(template,dict(approved_actual_frozen_feature_extraction=False,code=p['code'],scope=p['scope'],protocol=bind(PROTOCOL),inputs=p['inputs'],generated=p['generated'],source_only=True,fit_allowed=False,classifier_allowed=False,initializer_allowed=False,current_allowed=False,V112_allowed=False,PnP_allowed=False,GT_pose_owner_label_reads_allowed=False,automatic_retry=False,output=str(OUT)))
    write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),inputs=bind(INPUTS),generated=bind(GENERATED),entry_template=bind(template),input_occurrences=3816,variant_payloads=477,planned_encoder_calls=364,planned_BEFORE_reuses=113,source_payload_feature_weight_bodies_read=0,actual_encoder_or_fit_calls=0,training_admitted=False))
    print(json.dumps(dict(code=bind(__file__),protocol=bind(PROTOCOL),preparation=bind(H/'PREPARATION_COMPLETE.json'))),flush=True)
def run(entry):
    e=read(entry);assert e['approved_actual_frozen_feature_extraction'] is True and e['source_only'] is True
    for k in ('fit_allowed','classifier_allowed','initializer_allowed','current_allowed','V112_allowed','PnP_allowed','GT_pose_owner_label_reads_allowed','automatic_retry'):assert e[k] is False
    p=read(check(e['protocol']));assert e['protocol']==bind(PROTOCOL) and p['code']==bind(__file__) and sha(SCOPE)==SCOPE_SHA
    for k in ('code','scope','inputs','generated'):assert e[k]==p[k];check(p[k])
    for k in ('requirements','source_phase','metadata_complete','parent_protocol','original_feature_complete','C111_producer','C8_preprocess','C9_target_preprocess','helper_code'):check(p[k])
    for b in p['official_source_bindings']:check(b)
    assert read(check(p['generated']))['PASS'] and not OUT.exists() and e['output']==p['output']==str(OUT)
    inputs=read(check(p['inputs']));groups=inputs['groups'];frames={f['frame_index']:f for f in inputs['frames']}
    assert len(groups)==477 and len(frames)==46
    import torch
    assert str(torch.__version__)==p['runtime']['torch']=='2.1.1+cu118' and np.__version__=='1.26.4' and Path(sys.executable).resolve()==Path(p['runtime']['executable']).resolve() and torch.cuda.is_available()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    OUT.mkdir();write(OUT/'STARTED.json',dict(completed=False,pid=os.getpid(),entry=bind(entry),protocol=bind(PROTOCOL),automatic_retry=False));started=time.monotonic()
    try:
        op=operators(p);sys.path.insert(0,p['official_repo']);from dinov2.hub.backbones import dinov2_vits14_reg
        encoder=dinov2_vits14_reg(pretrained=False);weights=torch.load(check(p['weight']),map_location='cpu',weights_only=True);assert tuple(weights['register_tokens'].shape)==(1,4,384)
        encoder.load_state_dict(weights,strict=True);del weights;encoder=encoder.cuda().float().eval().requires_grad_(False)
        assert sum(t.numel() for t in encoder.parameters())==22058112
        state_before=op['state_sha'](encoder);assert state_before==p['expected_encoder_state']
        # Original saved features are opened only after root actual entry.
        with np.load(check(p['original_features']),allow_pickle=False) as z:
            old={k:z[k].copy() for k in ('features','global_index','frame_index','native_slot','query_id','query_xy','L','query_available','feature_constructed')}
        assert old['features'].shape==(1840,1928) and old['features'].dtype==np.float32
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
                ids=frame['global_index'];ii=np.asarray(ids,np.int64)
                assert old['global_index'][ii].tolist()==ids and old['frame_index'][ii].tolist()==[g['original_frame_index']]*8 and old['native_slot'][ii].tolist()==list(range(8))
                assert old['query_id'][ii].tolist()==frame['query_id'] and exact(old['query_xy'][ii],a['query_xy']) and exact(old['L'][ii],a['L']) and np.array_equal(old['query_available'][ii],a['native_continuous_valid'])
                assert exact(x,old['features'][ii]),'BEFORE actual feature1928 parity failed; preserve failure/no retry'
                with np.load(check(frame['existing_factual96_packet']),allow_pickle=False) as z:original_rgb=z['rgb96'].copy();original_valid=z['valid96'].copy()
                assert exact(original_rgb,a['rgb96']) and exact(original_valid,a['valid96'])
                assert all(ash(a['target224'])==f['target224_F32_HWC_sha256'] and ash(a['target_valid'])==f['target_valid_sha256'] for f in frame['facts'])
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
        assert calls==364 and len(cache)==len(beforeparity)==46 and len(reuses)==113 and len(features)==477 and len(lineage)==N
        x=np.concatenate(features);rr=[r for g in groups for r in g['rows']];available=np.array([r['query_available'] for r in rr],bool);constructed=np.array([r['feature_constructed'] for r in rr],bool)
        assert (~available).sum()==360 and constructed.all() and x.shape==(3816,1928)
        state_after=op['state_sha'](encoder);assert state_before==state_after and all(not t.requires_grad for t in encoder.parameters()) and all(not m.training for m in encoder.modules())
        for b in checked+[p['weight'],p['original_features']]:check(b)
        output=OUT/'CF3816_FEATURES_INPUT_ONLY.npz'
        arrays=dict(features=x,occurrence_index=np.arange(N,dtype=np.int32),input_occurrence_id=np.array([r['input_occurrence_id'] for r in rr]),pair_id=np.array([r['pair_id'] for r in rr]),variant=np.array([r['variant'] for r in rr]),original_query_id=np.array([r['original_query_id'] for r in rr]),original_global_index=np.array([r['original_global_index'] for r in rr],np.int32),original_frame_index=np.array([r['original_frame_index'] for r in rr],np.int32),source_id=np.array([r['source_id'] for r in rr]),split=np.array([r['split'] for r in rr]),native_slot=np.array([r['native_slot'] for r in rr],np.int8),query_xy=np.array([r['query_xy'] for r in rr],np.float64),L=np.array([r['L'] for r in rr],np.float64),query_available=available,native_continuous_valid=available,nearest_valid=np.array([r['nearest_valid'] for r in rr],bool),feature_constructed=constructed,local_central4_available=np.array([r['central4_available'] for r in rr],bool),context_available=np.array([r['context_available'] for r in rr],bool),support_fractions=np.asarray(allfractions,np.float32),context_box_xyxy=np.array([r['original_context_box_xyxy'] for r in rr],np.float64),unit_Q_inside=np.asarray(allinside,bool),occurrence_unit=np.arange(N,dtype=np.int32))
        np.savez_compressed(output,**arrays)
        def jsonl(name,rows):
            path=OUT/name
            with path.open('x') as f:
                for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            return bind(path)
        facts_binding=jsonl('PAYLOAD477_INPUT_PREPROCESS_FEATURE_FACTS.jsonl',facts);lineage_binding=jsonl('OCCURRENCE_FEATURE_LINEAGE3816.jsonl',lineage)
        parity=OUT/'BEFORE46_ORIGINAL_FEATURE_PARITY.json';write(parity,dict(completed=True,PASS=True,frames=46,rows=368,original_features=p['original_features'],all46_actual_encoder_calls=True,records=beforeparity))
        reusefile=OUT/'BEFORE113_EXACT_REUSE_RECORDS.json';write(reusefile,dict(completed=True,PASS=True,reuses=113,records=reuses))
        write(OUT/'FEATURE_COMPLETE.json',dict(completed=True,PASS=True,code=p['code'],scope=p['scope'],protocol=bind(PROTOCOL),entry=bind(entry),inputs=p['inputs'],features=bind(output),payload_facts=facts_binding,occurrence_lineage=lineage_binding,BEFORE_original_parity=bind(parity),BEFORE_reuse=bind(reusefile),rows=N,payloads=477,pairs=159,feature_shape=[3816,1928],inputU=360,feature_constructed=3816,failed_pairs_retained=2,actual_encoder_calls=calls,views_per_call=17,BEFORE_unique_calls=46,CF_calls=318,BEFORE_reuses=113,frozen_encoder_state_before=state_before,frozen_encoder_state_after=state_after,encoder_state_unchanged=True,only_RGB17views_encoder_input=True,alpha_owner_native_labels_GT_pose_not_encoder_inputs=True,original_qL_native_split_availability_unchanged=True,all_occurrences_preserved=True,UNKNOWN_targets_never_used=True,classifier_fit_initializer_current_V112_PnP_calls=0,training_admitted=False,automatic_retry=False,seconds=time.monotonic()-started))
        print(json.dumps(dict(completed=True,feature_complete=bind(OUT/'FEATURE_COMPLETE.json'))),flush=True)
    except BaseException:
        write(OUT/'FAILED_EXECUTION.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
    if a.prepare and not a.entry:prepare()
    elif a.entry and not a.prepare:run(a.entry)
    else:raise SystemExit('Prepare only first; root --entry required for actual bodies/encoder')
