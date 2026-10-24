"""V147 input-only frozen DINO; preparation reads metadata/code only.

Runtime reuses exact C111 assembly, C8/C9 preprocessing and initial q8 minmax.
No classifier, initializer, semantic annotation, source pose or depth loader.
"""
import os
import sys
os.environ['XFORMERS_DISABLED'] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
from pathlib import Path
import argparse, ast, hashlib, json, time, traceback
import numpy as np

H = Path(__file__).resolve().parent
V, P = H.parent, H.parent.parent
OLD = P / 'query_visibility_public_native_source_expansion_20261024_v146'
C111 = P / 'query_visibility_spatial_target_query_features_20261023_v111'
PARENT = C111 / 'feature_preparation/FEATURE_PROTOCOL_BEFORE_SOURCE_FORWARD.json'
CONTEXT = C111 / 'current_evaluation_preparation/current_adapter.py'
POLICY = V / 'ROOT_PUBLIC_NATIVE_FEATURE_POLICY_BEFORE_ENCODER_FORWARD.json'
POLICY_SHA = 'ed363cbb4ea9cdee33ceee60c9c66eb75a38c0fe57652405dded9a660ffc4382'
PROTOCOL = H / 'PROTOCOL_BEFORE_DINO_FORWARD.json'
INPUTS = H / 'PUBLIC64_FRAMES512_INPUT_ONLY.json'
GENERATED = H / 'GENERATED_OPERATOR_CHECKS.json'
OUT = V / 'public_native_DINO512_v1'
BOX_MODE = 'FROZEN_INITIAL_Q8_UNCLIPPED_UNEXPANDED_MINMAX'
N, FRAMES = 512, 64

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()

def bind(path): return dict(path=str(Path(path).resolve()), sha256=sha(path))
def check(b):
    assert sha(b['path']) == b['sha256'], b['path']
    return Path(b['path'])
def read(path): return json.loads(Path(path).read_text())
def lines(path): return [json.loads(line) for line in Path(path).read_text().splitlines() if line]
def write(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
def ash(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def byte_equal(a,b): return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
def require(condition, message):
    if not condition: raise ValueError(message)

def pure(path, names, ns):
    tree = ast.parse(Path(path).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path)+'::exact_pure_AST', 'exec'), ns)
    return ns

def operators(p):
    import torch
    from torch.nn import functional as F
    ns = dict(np=np, torch=torch, F=F, hashlib=hashlib, require=require)
    pure(p['C111_producer']['path'], ('sample_Q_raw','assemble','state_sha'), ns)
    pure(p['initial_context_code']['path'], ('initial_context_box',), ns)
    pure(p['local_crop_code']['path'], ('crop',), ns)
    pure(p['target_crop_code']['path'], ('valid_box','target_context_grid','sample_target_context'), ns)
    pure(p['native_grid_code']['path'], ('native_grid',), ns)
    c8 = pure(p['C8_preprocess']['path'], ('preprocess_views',), dict(torch=torch,F=F,MEAN=(.485,.456,.406),STD=(.229,.224,.225)))
    c9 = pure(p['C9_target_preprocess']['path'], ('preprocess_target',), dict(torch=torch,F=F,MEAN=(.485,.456,.406),STD=(.229,.224,.225)))
    ns['preprocess_views'], ns['preprocess_target'] = c8['preprocess_views'], c9['preprocess_target']
    return ns

def whitelist(source, op):
    """Read original R0 metadata; never open contact facts containing refs/states."""
    manifest = read(check(source['RGB_manifest']))
    init = read(check(source['initializer_prediction_seal']))
    contact = read(check(source['contact_complete']))
    assert init['completed'] and init['PASS'] and init['GT_read'] is False
    assert contact['completed'] and contact['PASS'] and contact['frames'] == FRAMES
    assert contact['initializer_prediction_seal'] == source['initializer_prediction_seal']
    assert contact['RGB_transport_seal'] == source['RGB_transport']
    names = {Path(b['path']).name:b for b in init['outputs']}
    pb = names['FROZEN_PROPOSALS_ALL64_INPUT_ONLY.jsonl']
    qb = names['ORIGINAL_NATIVE_QUERY512_INPUT_ONLY.jsonl']
    nb = names['ORIGINAL_NATIVE_QUERY512_AND_VALID9X9_INPUT_ONLY.npz']
    proposals, queries = lines(check(pb)), lines(check(qb))
    assert len(proposals) == FRAMES and len(queries) == N and len(manifest['rows']) == FRAMES
    packets = {r['frame_index']:r['crop_packet'] for r in contact['sheets']}
    frames = []
    for i,(m,s) in enumerate(zip(manifest['rows'],proposals)):
        assert m['frame_index'] == s['frame_index'] == i
        for k in ('id','source_id','source','split','raw_hw','canonical_RGB_NPY','decoded_RGB_sha256'):
            assert m[k] == s[k], (i,k)
        selected = None if not s['candidates'] else int(np.argmax([c['score'] for c in s['candidates']]))
        assert selected == s['selected_index'] and selected is not None
        q9 = np.asarray(s['candidates'][selected]['keypoints_xy'], np.float64)
        assert q9.shape == (9,2) and np.isfinite(q9).all()
        q8 = q9[:8]
        box = op['initial_context_box'](q8)
        assert op['valid_box'](box)
        side = float(op['native_grid'](q9, s['raw_hw'])[2])
        qq = queries[i*8:i*8+8]
        for k,r in enumerate(qq):
            assert r['frame_index'] == i and r['native_slot'] == k
            assert r['frame_id'] == s['id'] and r['source_id'] == s['source_id'] and r['split'] == s['split']
            assert r['query_id'] == s['id']+':native'+str(k)
            assert r['query_xy'] == q8[k].tolist() and r['L'] == side and not r['no_detection']
            x,y = r['query_xy']; hh,ww = s['raw_hw']
            assert r['native_continuous_valid'] == bool(0 <= x < ww and 0 <= y < hh)
        # Whitelist excludes original image path, selectedbbox, pose/ref/labels.
        frames.append(dict(frame_index=i,id=s['id'],source_id=s['source_id'],split=s['split'],
            raw_hw=s['raw_hw'],canonical_RGB_NPY=s['canonical_RGB_NPY'],decoded_RGB_sha256=s['decoded_RGB_sha256'],
            selected_index=selected,initial_points=q8.tolist(),L=side,context_box_xyxy=box.tolist(),
            crop_packet=packets[i],query_ids=[r['query_id'] for r in qq],
            native_continuous_valid=[r['native_continuous_valid'] for r in qq],
            nearest_valid=[r['query_nearest_pixel_valid'] for r in qq]))
    assert sum(sum(r['native_continuous_valid']) for r in frames) == 463
    assert sum(r['split']=='TRAIN_LAB' for r in frames) == sum(r['split']=='VAL_HALL' for r in frames) == 32
    return frames, dict(proposals=pb,queries=qb,query_npz=nb)

def generated(p, op):
    import torch
    torch.set_num_threads(1)
    # Invented RGB only: exact HWC/CHW and masks; no saved source crop reads.
    rgb = np.arange(40*50*3,dtype=np.uint8).reshape(40,50,3)
    crop,valid = op['crop'](rgb,[2.,3.],16.,96)
    assert crop.shape==(3,96,96) and valid.shape==(1,96,96)
    hwc = crop.transpose(1,2,0).copy()
    assert byte_equal(hwc.transpose(2,0,1).copy(),crop)
    a=np.stack([crop]*8);v=np.stack([valid]*8)
    changed=a.copy();changed[~np.broadcast_to(v,a.shape)]=251
    local,fractions=op['preprocess_views'](torch.from_numpy(a),torch.from_numpy(v))
    other,_=op['preprocess_views'](torch.from_numpy(changed),torch.from_numpy(v))
    assert torch.equal(local,other) and local.shape==(16,3,224,224)
    q8=np.array([[10.,12.],[30.,12.],[30.,28.],[10.,28.],[12.,14.],[28.,14.],[28.,26.],[12.,26.]])
    box=op['initial_context_box'](q8);assert box.tolist()==[10.,12.,30.,28.]
    outside=q8.copy();outside[0]=[-10.,-20.]
    assert op['initial_context_box'](outside).tolist()==[-10.,-20.,30.,28.]
    sentinel=np.full((8,2),-1.);assert op['initial_context_box'](sentinel) is None
    assert op['initial_context_box'](np.ones((8,2))) is None
    try: op['initial_context_box'](np.r_[q8,[[999.,999.]]])
    except ValueError: pass
    else: raise AssertionError('Centroid admitted to native8 context')
    t,vt=op['sample_target_context'](rgb,box)
    target=op['preprocess_target'](torch.from_numpy(t.transpose(2,0,1)[None]),torch.from_numpy(vt[None,None]))
    views=torch.cat((target,local),0);assert views.shape==(17,3,224,224)
    assert torch.equal(views[0:1],target) and torch.equal(views[1:9],local[:8]) and torch.equal(views[9:],local[8:])
    # Mock raw DINO tokens, not an encoder: inside/outside gates retain all8.
    cls=torch.arange(17*384,dtype=torch.float32).reshape(17,384)+1
    patch=torch.arange(17*256*384,dtype=torch.float32).reshape(17,256,384)/10000+1
    qs=q8.copy();qs[0]=[-100.,-100.]
    fra=np.repeat([[float(vt.mean()),float(valid.mean()),float(valid[:,32:64,32:64].mean())]],8,0)
    z,inside=op['assemble'](cls[:1].repeat_interleave(8,0),patch[:1].repeat_interleave(8,0),cls[1:9],patch[1:9],cls[9:],qs,[box]*8,fra,[False]+[True]*7)
    assert z.shape==(8,1928) and torch.isfinite(z).all() and not inside[0] and inside[1:].all()
    assert not z[0,384:768].any() and not z[0,1536:1920].any() and z[0,1927]==0
    original_availability=np.array([True]*7+[False]);assert original_availability[0] and not inside[0]
    # Qoutside cannot alter this immutable input flag.
    assert original_availability.sum()==7 and len(z)==8
    grid=op['native_grid'];nine=np.r_[q8,[[0.,0.]]];aL=grid(nine,[40,50])[2];nine[8]=[1e6,-1e6]
    assert grid(nine,[40,50])[2]==aL
    write(GENERATED,dict(completed=True,PASS=True,code=bind(__file__),
        exact_C111_C8_C9_AST_reuse=True,HWC_to_CHW_roundtrip=True,invalid_RGB_meanfill_invariance=True,
        view_order_1target_8full_8central=True,q8_unclipped_minmax_and_centroid_rejection=True,
        mock_tokens_1928_inside_Pzero_and_all8_retention=True,Qinside_does_not_change_availability=True,
        source_RGB_crop_features_weights_GT_models_read_or_forward=0))

def prepare():
    assert sha(POLICY) == POLICY_SHA
    policy=read(POLICY); source=policy['source_input_only']
    assert policy['context_box_mode'] == BOX_MODE, 'Require prospectively repaired q8 context policy'
    parent=read(PARENT);initprotocol=read(OLD/'initializer_preparation/PROTOCOL_BEFORE_SOURCE_FORWARD.json')
    p=dict(completed=True,prepared_only=True,code=bind(__file__),policy=bind(POLICY),parent_protocol=bind(PARENT),
        C111_producer=parent['code'],initial_context_code=bind(CONTEXT),native_grid_code=initprotocol['native_grid'],
        **{k:parent[k] for k in ('C8_preprocess','C9_target_preprocess','local_crop_code','target_crop_code','asset_runtime','asset_plan','weight','official_repo','official_revision','official_source_bindings','runtime')},
        source_input_only=source,context_box_mode=BOX_MODE,rows=N,frames=FRAMES,output=str(OUT),
        feature_layout=parent['schema']['layout'],expected_encoder_state=read(parent['asset_runtime']['path'])['state_before_SHA256'],
        calls=dict(primary=64,repeat_first=1,total=65,views_per_call=17),classifier_or_R0_or_P3_calls=0,
        labels_pose_depth_reference_body_reads=0,preparation_actual_RGB_crop_features_weights_body_reads=0)
    for k in ('C111_producer','initial_context_code','native_grid_code','C8_preprocess','C9_target_preprocess','local_crop_code','target_crop_code','asset_runtime','asset_plan'):check(p[k])
    op=operators(p);frames,raw_bindings=whitelist(source,op)
    write(INPUTS,dict(completed=True,RGB_GT_free=True,frames=frames,original_input_bindings=raw_bindings,
        original_query_availability463=True,context_box_mode=BOX_MODE,no_semantic_or_reference_fields=True))
    p['inputs']=bind(INPUTS);p['original_input_bindings']=raw_bindings
    generated(p,op);p['generated']=bind(GENERATED)
    write(PROTOCOL,p)
    template=dict(approved_actual_DINO_feature_extraction=False,code=p['code'],policy=p['policy'],protocol=bind(PROTOCOL),
        generated=p['generated'],inputs=p['inputs'],source_only=True,fit_allowed=False,classifier_allowed=False,
        initializer_allowed=False,current_allowed=False,PnP_allowed=False,semantic_pose_depth_reads_allowed=False,
        automatic_retry=False,output=str(OUT))
    write(H/'ROOT_FEATURE_ENTRY_TEMPLATE.json',template)
    ready=dict(completed=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),generated=bind(GENERATED),
        inputs=bind(INPUTS),template=bind(H/'ROOT_FEATURE_ENTRY_TEMPLATE.json'),actual_models_or_payloads_read=0,
        command=str(Path(sys.executable).resolve())+' '+str(Path(__file__).resolve())+' --entry ROOT_ACTUAL_FEATURE_ENTRY.json')
    write(H/'PREPARATION_READY.json',ready);print(json.dumps(ready),flush=True)

def run(entry):
    e=read(entry)
    assert e['approved_actual_DINO_feature_extraction'] is True and e['source_only'] is True
    for k in ('fit_allowed','classifier_allowed','initializer_allowed','current_allowed','PnP_allowed','semantic_pose_depth_reads_allowed','automatic_retry'):assert e[k] is False
    p=read(check(e['protocol']));assert e['protocol']==bind(PROTOCOL)
    for k in ('code','policy','generated','inputs'):
        assert e[k]==p[k];check(p[k])
    assert e['code']==bind(__file__) and e['policy']==bind(POLICY) and e['output']==str(OUT)==p['output'] and not OUT.exists()
    assert sha(POLICY)==POLICY_SHA and p['context_box_mode']==BOX_MODE
    assert read(check(p['generated']))['PASS']
    for k in ('parent_protocol','C111_producer','initial_context_code','native_grid_code','C8_preprocess','C9_target_preprocess','local_crop_code','target_crop_code','asset_runtime','asset_plan','weight'):check(p[k])
    for b in p['official_source_bindings']:check(b)
    for b in p['source_input_only'].values():check(b)
    import torch
    assert str(torch.__version__)==p['runtime']['torch']=='2.1.1+cu118'
    assert np.__version__=='1.26.4' and Path(sys.executable).resolve()==Path(p['runtime']['executable']).resolve() and torch.cuda.is_available()
    op=operators(p);frames,original=whitelist(p['source_input_only'],op)
    packet=read(check(p['inputs']));assert packet['frames']==frames and original==p['original_input_bindings']
    # Independent raw-array join uses only original saved R0 input arrays.
    with np.load(check(original['query_npz']),allow_pickle=False) as f:
        for i,r in enumerate(frames):
            sl=slice(i*8,i*8+8)
            assert byte_equal(f['query_xy'][sl],np.asarray(r['initial_points'],np.float64))
            assert np.all(f['L'][sl]==r['L']) and np.array_equal(f['continuous_valid'][sl],r['native_continuous_valid'])
            assert f['query_id'][sl].tolist()==r['query_ids']
    OUT.mkdir()
    write(OUT/'INPUT_SEAL_BEFORE_ENCODER.json',dict(completed=True,PASS=True,entry=bind(entry),inputs=p['inputs'],
        source_input_only=p['source_input_only'],rows=N,frames=64,available463_OOB49_unchanged=True,
        context_box_mode=BOX_MODE,labels_pose_depth_reference_body_reads=0))
    write(OUT/'STARTED.json',dict(completed=False,entry=bind(entry),inputs=p['inputs'],automatic_retry=False))
    started=time.monotonic()
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
        sys.path.insert(0,p['official_repo']);from dinov2.hub.backbones import dinov2_vits14_reg
        encoder=dinov2_vits14_reg(pretrained=False)
        weights=torch.load(p['weight']['path'],map_location='cpu',weights_only=True)
        assert tuple(weights['register_tokens'].shape)==(1,4,384)
        encoder.load_state_dict(weights,strict=True);del weights
        encoder=encoder.cuda().float().eval().requires_grad_(False)
        assert sum(t.numel() for t in encoder.parameters())==22058112
        before=op['state_sha'](encoder);assert before==p['expected_encoder_state']
        features=[];insidebits=[];facts=[];first=None;calls=0;checked=[]
        central4bits=[];contextbits=[];fractions_all=[]
        for i,r in enumerate(frames):
            rgb=np.load(check(r['canonical_RGB_NPY']),allow_pickle=False)
            assert rgb.dtype==np.uint8 and list(rgb.shape)==r['raw_hw']+[3] and ash(rgb)==r['decoded_RGB_sha256']
            with np.load(check(r['crop_packet']),allow_pickle=False) as f:
                assert set(f.files)=={'rgb96','valid96','global_index'}
                hwc=f['rgb96'].copy();valid=f['valid96'].copy();ix=f['global_index'].copy()
            assert hwc.dtype==np.uint8 and hwc.shape==(8,96,96,3) and valid.dtype==bool and valid.shape==(8,96,96)
            assert np.array_equal(ix,np.arange(i*8,i*8+8))
            local=np.ascontiguousarray(hwc.transpose(0,3,1,2));vmask=valid[:,None]
            # Saved original crops are reused exactly; recomputation is parity-only.
            for k,q in enumerate(r['initial_points']):
                a,v=op['crop'](rgb,q,r['L'],96)
                assert byte_equal(a,local[k]) and byte_equal(v,vmask[k])
            target,vt=op['sample_target_context'](rgb,r['context_box_xyxy'])
            fractions=np.array([[np.float32(vt.mean()),np.float32(vmask[k].mean()),np.float32(vmask[k,:,32:64,32:64].mean())] for k in range(8)],np.float32)
            central4=[bool(vmask[k,:,47:49,47:49].all()) for k in range(8)]
            lv,_=op['preprocess_views'](torch.as_tensor(local,device='cuda'),torch.as_tensor(vmask,device='cuda'))
            tv=op['preprocess_target'](torch.as_tensor(target.transpose(2,0,1)[None],device='cuda'),torch.as_tensor(vt[None,None],device='cuda'))
            views=torch.cat((tv,lv),0);assert views.shape==(17,3,224,224) and views.dtype==torch.float32
            with torch.inference_mode():
                z=encoder.forward_features(views);calls+=1
                if i==0:
                    first_views=views.clone()
                    first={key:z[key].cpu().numpy().copy() for key in ('x_norm_clstoken','x_norm_patchtokens')}
                cl,pt=z['x_norm_clstoken'],z['x_norm_patchtokens']
                x,inside=op['assemble'](cl[:1].repeat_interleave(8,0),pt[:1].repeat_interleave(8,0),cl[1:9],pt[1:9],cl[9:],r['initial_points'],[r['context_box_xyxy']]*8,fractions,central4)
            features.append(x.cpu().numpy());insidebits.extend(inside.tolist())
            central4bits.extend(central4);contextbits.extend([bool(vt.any())]*8);fractions_all.extend(fractions.tolist())
            for k in range(8):
                facts.append(dict(global_index=i*8+k,frame_index=i,frame_id=r['id'],source_id=r['source_id'],split=r['split'],native_slot=k,
                    query_id=r['query_ids'][k],query_xy=r['initial_points'][k],L=r['L'],selected_index=r['selected_index'],
                    context_box_xyxy=r['context_box_xyxy'],native_continuous_valid=r['native_continuous_valid'][k],
                    Q_inside=bool(inside[k]),local_central4_available=central4[k],context_available=bool(vt.any()),
                    nearest_valid=r['nearest_valid'][k],RGB96_HWC_sha256=ash(hwc[k]),RGB96_CHW_sha256=ash(local[k]),valid96_sha256=ash(valid[k]),
                    target224_F32_HWC_sha256=ash(target),target_valid_sha256=ash(vt),fractions=fractions[k].tolist()))
            checked.extend([r['canonical_RGB_NPY'],r['crop_packet']])
            if (i+1)%16==0:print(json.dumps(dict(frames=i+1,rows=(i+1)*8,encoder_calls=calls,seconds=time.monotonic()-started)),flush=True)
        # Repeat first after all64 original frames, using identical 17view shape.
        with torch.inference_mode():again=encoder.forward_features(first_views);calls+=1
        for key in ('x_norm_clstoken','x_norm_patchtokens'):assert byte_equal(first[key],again[key].cpu().numpy())
        x=np.concatenate(features);inside=np.array(insidebits,bool)
        available=np.array([v for r in frames for v in r['native_continuous_valid']],bool)
        assert x.shape==(N,1928) and x.dtype==np.float32 and np.isfinite(x).all() and available.sum()==463 and calls==65 and first is not None
        assert np.array_equal(x[:,1927],inside.astype(np.float32)) and not x[~inside,384:768].any()
        after=op['state_sha'](encoder);assert before==after and all(not t.requires_grad for t in encoder.parameters()) and all(not m.training for m in encoder.modules())
        for b in checked+[p['weight']]:check(b)
        del encoder
        output=OUT/'PUBLIC_NATIVE512_FEATURES_INPUT_ONLY.npz'
        np.savez_compressed(output,features=x,global_index=np.arange(N,dtype=np.int32),frame_index=np.repeat(np.arange(64,dtype=np.int32),8),
            native_slot=np.tile(np.arange(8,dtype=np.int8),64),query_id=np.array([q for r in frames for q in r['query_ids']]),
            frame_id=np.repeat([r['id'] for r in frames],8),source_id=np.repeat([r['source_id'] for r in frames],8),split=np.repeat([r['split'] for r in frames],8),
            query_xy=np.array([q for r in frames for q in r['initial_points']],np.float64),L=np.repeat([r['L'] for r in frames],8).astype(np.float64),
            selected_index=np.repeat([r['selected_index'] for r in frames],8).astype(np.int32),query_available=available,native_continuous_valid=available,
            nearest_valid=np.array([v for r in frames for v in r['nearest_valid']],bool),
            local_central4_available=np.array(central4bits,bool),context_available=np.array(contextbits,bool),
            support_fractions=np.array(fractions_all,np.float32),unit_Q_inside=inside,occurrence_unit=np.arange(N,dtype=np.int32))
        factfile=OUT/'FEATURE_INPUT_FACTS512_INPUT_ONLY.jsonl'
        with factfile.open('x') as f:
            for row in facts:f.write(json.dumps(row,allow_nan=False)+'\n')
        write(OUT/'FEATURE_COMPLETE.json',dict(completed=True,PASS=True,code=p['code'],protocol=bind(PROTOCOL),policy=p['policy'],entry=bind(entry),
            inputs=p['inputs'],input_seal=bind(OUT/'INPUT_SEAL_BEFORE_ENCODER.json'),source_input_only=p['source_input_only'],features=bind(output),inputfacts=bind(factfile),
            rows=N,frames=64,features_shape=[N,1928],available=463,finite_OOB=49,all512_preserved=True,Q_outside_never_adds_U=True,
            context_box_mode=BOX_MODE,primary_encoder_calls=64,repeat_first_calls=1,total_encoder_calls=calls,views_per_call=17,
            first_repeat_raw_tokens_bitexact=True,frozen_encoder_state_before=before,frozen_encoder_state_after=after,
            encoder_state_unchanged=True,weight_and_inputs_unchanged=True,saved_RGB96_exact_reuse=True,register_features_used=False,
            labels_pose_depth_reference_body_reads=0,classifier_R0_P3_fit_private_PnP_calls=0,seconds=time.monotonic()-started))
        print(json.dumps(dict(completed=True,PASS=True,feature_complete=bind(OUT/'FEATURE_COMPLETE.json'))),flush=True)
    except BaseException:
        write(OUT/'FAILED_EXECUTION.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--entry',type=Path);args=parser.parse_args()
    if args.prepare and args.entry is None:prepare()
    elif args.entry and not args.prepare:run(args.entry)
    else:raise SystemExit('Use metadata/invented --prepare or root --entry')
