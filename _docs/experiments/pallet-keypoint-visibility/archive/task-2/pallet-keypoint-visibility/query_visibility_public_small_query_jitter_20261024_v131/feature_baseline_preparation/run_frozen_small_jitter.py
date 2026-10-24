"""Source-only chunked RGB -> exact frozen DINO1928 -> frozen C16/C23; small-jitter diagnosis only.

No GT reader and no model/payload I/O on import. Root binds completed corpora.
"""
import os, sys
os.environ['XFORMERS_DISABLED'] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
from pathlib import Path
import argparse, ast, collections, hashlib, importlib.util, json, time, traceback
import numpy as np

H = Path(__file__).resolve().parent
V = H.parent
P = V.parent
PARENT_V = P / 'query_visibility_registered_native_diversity_20261023_v129'
PARENT = PARENT_V / 'feature_preparation/FROZEN_FEATURE_PROTOCOL_BEFORE_FORWARD.json'
PARENT_CODE = PARENT_V / 'feature_preparation/run_frozen_C16_corpora.py'
PROTOCOL = H / 'FROZEN_FEATURE_PROTOCOL_BEFORE_FORWARD.json'
OUT = V / 'frozen_C16_C23_small_jitter_v1'
POLICY=V/'ROOT_SMALL_QUERY_DIAGNOSIS_POLICY_BEFORE_INPUTS.json'
GENERATED=H/'GENERATED_SMALL_JITTER_CHECKS.json'
BATCH = 8

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for z in iter(lambda: f.read(1 << 20), b''): h.update(z)
    return h.hexdigest()

def bind(p): return {'path': str(Path(p).resolve()), 'sha256': sha(p)}
def check(b):
    assert sha(b['path']) == b['sha256'], b['path']
    return Path(b['path'])
def read(p): return json.loads(Path(p).read_text())
def write(p, obj):
    with Path(p).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
def ash(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def module(p, name):
    s = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m
def pure(p, names):
    nodes = [n for n in ast.parse(Path(p).read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    ns = {'np': np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(p) + '::pure', 'exec'), ns)
    return ns

def validate_inputs(rows):
    assert rows and [r['occurrence_index'] for r in rows] == list(range(len(rows)))
    assert len({r['id'] for r in rows}) == len(rows)
    forbidden = {'reference_xy', 'raw_state', 'teacher_label', 'owner', 'part_id', 'witnesses', 'error_px'}
    for r in rows:
        assert not forbidden.intersection(r)
        q8 = np.asarray(r['initial_points'], np.float64)
        q = np.asarray(r['query_xy'], np.float64)
        j = r['carrier_slot']
        assert q8.shape == (8, 2) and q.shape == (2,) and 0 <= j < 8
        assert np.isfinite(q8).all() and np.array_equal(q8[j], q)
        good = np.isfinite(q8).all(1) & ~(q8 == [-1, -1]).all(1)
        points = q8[good]
        assert len(points) >= 2
        box = np.r_[points.min(0), points.max(0)]
        assert np.all(box[2:] > box[:2]) and box.tolist() == r['context_box_xyxy']
        assert float(np.clip(.25 * np.ptp(points, axis=0).max(), 16, max(r['raw_hw']))) == r['L']
        assert r['full_RGB_format'] == 'NPZ_RGB_UINT8_HWC' and r['full_RGB_key'] == 'RGB_uint8'
        assert r['rgb96_row'] >= 0 and set(r['rgb96_chunk']) >= {'path', 'sha256'}

def mapping(names,indices,requested,tag='PUBLIC_AUX'):
    chosen=np.flatnonzero(names==tag);lookup={int(indices[i]):int(i)for i in chosen}
    assert len(lookup)==len(chosen) and set(lookup)==set(range(918))
    return np.array([lookup[int(j)]for j in requested],np.int64)

def reference_scores(path,parent_local,modelname,centre_rows):
    # Lazy NPZ whitelist: do not inspect labels/raw_state/GT members in C23 scores.
    with np.load(path,allow_pickle=False)as f:
        names=np.array(f['corpus'],copy=True);local=np.array(f['source_occurrence_index'],copy=True);indices=mapping(names,local,parent_local)
        raw=np.array(f['raw_head_logit_before_U'][indices],copy=True);available=np.array(f['query_available'][indices],copy=True)
        if modelname=='C23':
            assert np.array_equal(f['query_xy_raw'][indices],np.array([r['query_xy']for r in centre_rows]))and np.array_equal(f['L_raw'][indices],np.array([r['L']for r in centre_rows]))
    return indices,raw,available

def authorize(entry):
    e=read(entry);assert e['approved_small_jitter_features_and_two_heads']is True
    assert all(e[k]is False for k in ('fit','current223','V112','PnP','initializer','automatic_retry'))
    for k in ('code','protocol','generated','source_policy','input_complete','input_rows'):check(e[k])
    p=read(PROTOCOL);assert e['code']==p['code']==bind(__file__) and e['protocol']==bind(PROTOCOL) and e['source_policy']==p['source_policy']==bind(POLICY) and e['generated']==p['generated']
    assert read(e['generated']['path'])['PASS'] and read(e['generated']['path'])['code']==e['code']
    for k in ('parent_protocol','parent_code','C11_producer','C11_protocol','C8_preprocess','C9_target_preprocess','local_crop_code','target_crop_code','asset_runtime','asset_plan','weight','source_model_freeze','checkpoint','head_class_code','availability_code','old_public_inputs','old_C16_prediction_seal','old_feature_receipt','old_features','old_C16_scores','C23_fit_complete','C23_freeze','C23_checkpoint','old_C23_scores'):check(p[k])
    for b in p['official_source_bindings']:check(b)
    policy=read(POLICY);assert policy['parent_source']['inputs']==p['old_public_inputs'] and policy['parent_models']['C16']==p['old_C16_prediction_seal'] and policy['parent_models']['C23']==p['C23_freeze']
    c16=read(p['source_model_freeze']['path']);assert c16['completed'] and c16['selected_update']==4500 and c16['checkpoint']==p['checkpoint'] and c16['threshold_logit']==0 and not c16['current_used_for_fit_or_selection']
    s16=read(p['old_C16_prediction_seal']['path']);fr=read(p['old_feature_receipt']['path']);fit=read(p['C23_fit_complete']['path']);c23=read(p['C23_freeze']['path'])
    assert s16['completed']and s16['PASS']and s16['checkpoint']==p['checkpoint']and s16['features']==p['old_features']and s16['feature_receipt']==p['old_feature_receipt']and s16['predictions']==p['old_C16_scores']and not s16['GT_read']
    assert fr['completed']and fr['PASS']and fr['output']==p['old_features']and fr['features_shape']==[56214,1928]and fr['first_batch_repeat_exact']and fr['encoder_state_unchanged']and not fr['GT_read']
    assert fit['completed']and fit['PASS']and fit['optimizer_updates']==8000 and fit['model_freeze']==p['C23_freeze']and fit['scores']==p['old_C23_scores']and fit['current_evaluation_authorized']is False
    assert c23['completed']and c23['selected_update']==8000 and c23['checkpoint']==p['C23_checkpoint']and c23['head_class_code']==p['head_class_code']and c23['threshold_logit']==0 and not c23['current_used_for_fit_or_selection']
    done=read(e['input_complete']['path']);packet=read(e['input_rows']['path']);old=read(p['old_public_inputs']['path'])['rows']
    assert done['completed']and done['PASS']and done['input_manifest']==e['input_rows']and packet['completed']and len(packet['rows'])==len(old)==918
    rows=packet['rows'];validate_inputs(rows);offsets=policy['query_design']['offsets_native_px']
    for i,r in enumerate(rows):
        j=(i//17)%9;centre=(i//153)*153+4*17+i%17;parent=old[centre]
        assert r['condition_index']==j and r['parent_occurrence_index']==i and r['parent_center_occurrence_index']==centre
        assert r['full_RGB']==parent['full_RGB']and r['decoded_RGB_sha256']==parent['decoded_RGB_sha256']and r['raw_hw']==parent['raw_hw']and r['split']==parent['split']
        q8=np.array(parent['initial_points']);q8[0]+=offsets[j];assert np.array_equal(q8,r['initial_points'])and r['carrier_slot']==0
        if j==4:
            for k in ('query_xy','initial_points','L','context_box_xyxy','rgb96_sha256','valid96_sha256','query_available','factual_query_available','context_available'):assert r[k]==parent[k],k
    assert not OUT.exists(),'Preserve failed/completed attempt; no automatic retry.'
    return e,p,[('PUBLIC_SMALL_JITTER',rows)]

def run(entry):
    e, p, corpora = authorize(entry)
    import torch
    assert torch.__version__ == p['runtime']['torch'] == '2.1.1+cu118'
    assert Path(sys.executable).resolve() == Path(p['runtime']['executable']).resolve() and torch.cuda.is_available()
    rows = []; origin = []
    for name, rr in corpora:
        rows.extend(rr)
        origin.extend((name, i) for i in range(len(rr)))
    N = len(rows)
    OUT.mkdir(); (OUT / 'feature_chunks').mkdir()
    write(OUT / 'STARTED.json', {'entry': bind(entry), 'rows': N, 'GT_read': False, 'automatic_retry': False})
    started = time.monotonic()
    try:
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
        producer = module(p['C11_producer']['path'], 'v129_exact_C11')
        c8 = module(p['C8_preprocess']['path'], 'v129_exact_C8')
        c9 = module(p['C9_target_preprocess']['path'], 'v129_exact_C9')
        tm = module(p['target_crop_code']['path'], 'v129_exact_context')
        crop = producer.pure_crop()
        # Pure geometry helper only; parent I/O/prepare/run functions are never invoked.
        availability = pure(Path(p['availability_code']['path']), ('factual_availability',))['factual_availability']
        image_cache = collections.OrderedDict(); crop_cache = collections.OrderedDict()
        checked = set()
        def load_checked(binding):
            key = (binding['path'], binding['sha256'])
            if key not in checked: check(binding); checked.add(key)
            return key
        def native(r):
            key = load_checked(r['full_RGB'])
            if key not in image_cache:
                with np.load(key[0], allow_pickle=False) as f: image_cache[key] = np.array(f['RGB_uint8'], copy=True)
                image = image_cache[key]
                assert image.dtype == np.uint8 and image.shape == tuple(r['raw_hw']) + (3,) and ash(image) == r['decoded_RGB_sha256']
                if len(image_cache) > 4: image_cache.popitem(last=False)
            image_cache.move_to_end(key)
            return image_cache[key]
        def stored(r):
            key = load_checked(r['rgb96_chunk'])
            if key not in crop_cache:
                with np.load(key[0], allow_pickle=False) as f: crop_cache[key] = (f['rgb96'], f['valid96'], f['occurrence_index'])
                if len(crop_cache) > 2: crop_cache.popitem(last=False)
            crop_cache.move_to_end(key)
            rgb, valid, indices = crop_cache[key]; j = r['rgb96_row']
            assert indices[j] == r['occurrence_index']
            assert ash(rgb[j]) == r['rgb96_sha256'] and ash(valid[j]) == r['valid96_sha256']
            return rgb[j].transpose(2, 0, 1).copy(), valid[j][None].copy()
        sys.path.insert(0, p['official_repo'])
        from dinov2.hub.backbones import dinov2_vits14_reg
        encoder = dinov2_vits14_reg(pretrained=False)
        state = torch.load(p['weight']['path'], map_location='cpu', weights_only=True)
        encoder.load_state_dict(state, strict=True); del state
        encoder = encoder.cuda().float().eval().requires_grad_(False)
        encstate = producer.state_sha(encoder)
        assert encstate == read(p['asset_runtime']['path'])['state_before_SHA256']
        features = np.empty((N, 1928), np.float32)
        factual = np.zeros(N, bool); context = np.zeros(N, bool); qinside = np.zeros(N, bool)
        calls = 0; repeated = False; chunks = []; checkpoint_start = 0
        for start in range(0, N, BATCH):
            batch = rows[start:start + BATCH]; n = len(batch)
            lr = []; lv = []; tr = []; tv = []; fractions = []; central4 = []
            for offset, r in enumerate(batch):
                image = native(r); a, valid = crop(image, np.array(r['query_xy'], np.float64), r['L'], 96)
                saved, savedvalid = stored(r)
                assert np.array_equal(a, saved) and np.array_equal(valid, savedvalid), ('native crop replay', start + offset)
                target, vt = tm.sample_target_context(image, r['context_box_xyxy'])
                fa, ca, aa, c4 = availability(r['query_xy'], r['raw_hw'], valid, vt)
                assert fa == r['factual_query_available'] and ca == r['context_available'] and aa == r['query_available']
                factual[start + offset] = fa; context[start + offset] = ca
                lr.append(a); lv.append(valid); tr.append(target.transpose(2, 0, 1)); tv.append(vt[None])
                fractions.append([np.float32(vt.mean()), np.float32(valid.mean()), np.float32(valid[:, 32:64, 32:64].mean())]); central4.append(c4)
            local, _ = c8.preprocess_views(torch.as_tensor(np.stack(lr), device='cuda'), torch.as_tensor(np.stack(lv), device='cuda'))
            target = c9.preprocess_target(torch.as_tensor(np.stack(tr), device='cuda'), torch.as_tensor(np.stack(tv), device='cuda'))
            views = torch.cat((target, local), 0)
            with torch.inference_mode():
                z = encoder.forward_features(views); calls += 1
                if not repeated:
                    again = encoder.forward_features(views); calls += 1
                    assert all(torch.equal(z[k], again[k]) for k in ('x_norm_clstoken', 'x_norm_patchtokens'))
                    repeated = True
                cl = z['x_norm_clstoken']; pt = z['x_norm_patchtokens']
                x, inside = producer.assemble(cl[:n], pt[:n], cl[n:2*n], pt[n:2*n], cl[2*n:], [r['query_xy'] for r in batch], [r['context_box_xyxy'] for r in batch], fractions, central4)
            end = start + n
            features[start:end] = x.cpu().numpy(); qinside[start:end] = np.asarray(inside, bool)
            if end - checkpoint_start >= 2048 or end == N:
                f = OUT / 'feature_chunks' / f'UNITS_{checkpoint_start:06d}_{end:06d}.npz'
                np.savez_compressed(f, features=features[checkpoint_start:end], global_index=np.arange(checkpoint_start, end, dtype=np.int32))
                chunks.append(bind(f)); checkpoint_start = end
                print(json.dumps({'feature_rows_complete': end, 'rows': N, 'encoder_calls': calls, 'elapsed_seconds': round(time.monotonic()-started, 2)}), flush=True)
        assert np.isfinite(features).all() and np.array_equal(features[:, 1927], qinside.astype(np.float32))
        assert not features[~qinside, 384:768].any() and producer.state_sha(encoder) == encstate
        assert all(not t.requires_grad for t in encoder.parameters()) and all(not t.training for t in encoder.modules())
        del encoder
        available = factual & context
        ff = OUT / 'SOURCE_FEATURES_INPUT_ONLY.npz'
        np.savez_compressed(ff, features=features, query_available=available, factual_query_available=factual, context_available=context, Q_inside=qinside, global_index=np.arange(N, dtype=np.int32), corpus=np.array([x[0] for x in origin]), source_occurrence_index=np.array([x[1] for x in origin], np.int32))
        write(OUT / 'FEATURE_COMPLETE.json', {'completed': True, 'PASS': True, 'entry': bind(entry), 'code': e['code'], 'protocol': e['protocol'], 'input_manifest': e['input_rows'], 'input_complete': e['input_complete'], 'output': bind(ff), 'feature_chunks': chunks, 'rows': N, 'features_shape': list(features.shape), 'encoder_calls': calls, 'first_batch_repeat_exact': True, 'encoder_state_unchanged': True, 'GT_read': False, 'all_saved_crop_replay_exact': True})
        # Old reference maps are input identity only. Teacher/label members are never read.
        with np.load(p['old_features']['path'],allow_pickle=False)as f:
            old_names=np.array(f['corpus'],copy=True);old_local=np.array(f['source_occurrence_index'],copy=True)
        centre=np.array([i for i,r in enumerate(rows)if r['condition_index']==4],np.int64);parent_local=np.array([rows[i]['parent_center_occurrence_index']for i in centre],np.int64);old_global=mapping(old_names,old_local,parent_local)
        wanted={int(g):j for j,g in enumerate(old_global)};reference=np.empty((len(centre),1928),np.float32);seen=set()
        for binding in read(p['old_feature_receipt']['path'])['feature_chunks']:
            check(binding)
            with np.load(binding['path'],allow_pickle=False)as f:
                indices=np.array(f['global_index'],copy=True);take=[k for k,g in enumerate(indices)if int(g)in wanted]
                if take:
                    x=f['features']
                    for k in take:
                        g=int(indices[k]);assert g not in seen;reference[wanted[g]]=x[k];seen.add(g)
        assert seen==set(wanted)and np.array_equal(features[centre],reference),'Centre1928 feature parity is exact; no tolerance relaxation.'
        centreproof=dict(completed=True,PASS=True,centre_rows=len(centre),new_indices=centre.tolist(),parent_local_indices=parent_local.tolist(),parent_feature_global_indices=old_global.tolist(),centre_features_bitexact=True,parent_feature_receipt=p['old_feature_receipt'])
        assert len(centre)==102
        hm=module(p['head_class_code']['path'],'v131_frozen_same_C11_head');prediction_bindings={}
        for modelname,ck,update,scorekey in (('C16',p['checkpoint'],4500,'old_C16_scores'),('C23',p['C23_checkpoint'],8000,'old_C23_scores')):
            payload=torch.load(ck['path'],map_location='cpu',weights_only=True);assert payload['update']==update and payload['head_class_code']==p['head_class_code']
            head=hm.SpatialTargetQueryHead().cuda().float().eval().requires_grad_(False);head.load_state_dict(payload['state_dict'],strict=True);del payload
            hs=producer.state_sha(head);logits=[]
            with torch.inference_mode():
                for lo in range(0,N,512):
                    x=torch.as_tensor(features[lo:lo+512],device='cuda');z=head(x)
                    if lo==0:assert torch.equal(z,head(x))
                    logits.append(z.cpu().numpy())
            raw=np.concatenate(logits).astype(np.float32);assert raw.shape==(N,)and np.isfinite(raw).all()and producer.state_sha(head)==hs
            if modelname=='C23':assert hs==read(p['C23_freeze']['path'])['model_state_sha256']
            reference_indices,previous,previous_available=reference_scores(p[scorekey]['path'],parent_local,modelname,[rows[i]for i in centre])
            assert np.array_equal(raw[centre],previous)and np.array_equal(available[centre],previous_available),'Exact centre raw score/availability mismatch; do not widen tolerance.'
            centreproof[modelname]=dict(raw_scores_bitexact=True,availability_bitexact=True,previous_scores=p[scorekey],previous_global_indices=reference_indices.tolist(),head_state_sha256=hs)
            pred=np.where(available,(raw>=0).astype(np.int8),-1).astype(np.int8);pf=OUT/(modelname+'_PREDICTIONS_INPUT_ONLY.npz')
            np.savez_compressed(pf,query_logit=np.where(available,raw,0),raw_head_logit_before_U=raw,prediction=pred,query_available=available,global_index=np.arange(N,dtype=np.int32),corpus=np.full(N,'PUBLIC_SMALL_JITTER'),source_occurrence_index=np.arange(N,dtype=np.int32),occurrence_id=np.array([r['id']for r in rows]),query_xy=np.array([r['query_xy']for r in rows]),L=np.array([r['L']for r in rows]))
            prediction_bindings[modelname]=dict(predictions=bind(pf),checkpoint=ck,update=update,head_state_sha256=hs,head_state_unchanged=True,repeat_exact=True);del head
        cp=OUT/'CENTRE_PARITY.json';write(cp,centreproof)
        seal=OUT/'PREDICTION_SEAL.json';write(seal,dict(completed=True,PASS=True,entry=bind(entry),code=e['code'],protocol=e['protocol'],input_complete=e['input_complete'],input_manifest=e['input_rows'],features=bind(ff),feature_receipt=bind(OUT/'FEATURE_COMPLETE.json'),models=prediction_bindings,centre_parity=bind(cp),rows=N,available=int(available.sum()),threshold_logit=0,decoder='D raw<0,E raw>=0; factual/context unavailable U first',GT_read=False,fit_current_V112_PnP=0,source_gates_unchanged=True,C23_application_admitted=False,seconds=time.monotonic()-started))
        print(json.dumps(dict(completed=True,seal=bind(seal),centre_rows=len(centre),seconds=time.monotonic()-started)),flush=True)
    except BaseException:
        write(OUT / 'FAILED_NO_RESTART.json', {'traceback': traceback.format_exc(), 'automatic_retry': False})
        raise

def prepare():
    parent=read(PARENT);policy=read(POLICY);keys=('C11_producer','C11_protocol','C8_preprocess','C9_target_preprocess','local_crop_code','target_crop_code','asset_runtime','asset_plan','weight','source_model_freeze','checkpoint','head_class_code','official_repo','official_revision','official_source_bindings','runtime','availability_code');p={k:parent[k]for k in keys}
    proof=read(GENERATED);assert proof['PASS']and proof['code']==bind(__file__)
    oldseal=read(policy['parent_models']['C16']['path']);c23freeze=read(policy['parent_models']['C23']['path']);fitpath=Path(policy['parent_models']['C23']['path']).parent/'FIT_COMPLETE.json';fit=read(fitpath)
    p.update(completed=True,code=bind(__file__),generated=bind(GENERATED),parent_protocol=bind(PARENT),parent_code=bind(PARENT_CODE),source_policy=bind(POLICY),old_public_inputs=policy['parent_source']['inputs'],old_C16_prediction_seal=policy['parent_models']['C16'],old_feature_receipt=oldseal['feature_receipt'],old_features=oldseal['features'],old_C16_scores=oldseal['predictions'],C23_fit_complete=bind(fitpath),C23_freeze=policy['parent_models']['C23'],C23_checkpoint=c23freeze['checkpoint'],old_C23_scores=fit['scores'],output=str(OUT),feature_batch=8,head_batch=512,centre_parity='Exact102features/rawscores/inputflags; no tolerance relaxation; indices joined by corpus+source_occurrence_index.',GT_read=False,C23_application_admitted=False,source_gates_unchanged=True,no_fit_current_V112_PnP=True)
    write(PROTOCOL,p);t=dict(approved_small_jitter_features_and_two_heads=False,fit=False,current223=False,V112=False,PnP=False,initializer=False,automatic_retry=False,code=p['code'],protocol=bind(PROTOCOL),generated=bind(GENERATED),source_policy=bind(POLICY),input_complete=None,input_rows=None);tp=H/'ROOT_ENTRY_TEMPLATE.json';write(tp,t)
    print(json.dumps(dict(code=p['code'],protocol=bind(PROTOCOL),generated=bind(GENERATED),template=bind(tp),command=p['runtime']['executable']+' '+str(Path(__file__).resolve())+' --entry ROOT_ACTUAL_SMALL_JITTER_FEATURE_ENTRY.json',actual_forward=0)),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--entry',type=Path);o=a.parse_args()
    if o.prepare:prepare()
    elif o.entry:run(o.entry)
    else:raise SystemExit('Use --prepare or root --entry.')
