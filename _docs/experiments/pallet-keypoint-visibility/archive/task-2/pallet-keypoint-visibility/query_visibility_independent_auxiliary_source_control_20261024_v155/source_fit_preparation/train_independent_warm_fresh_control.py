"""Two fixed source-only fits. Preparation never loads features or weights.

Actual fit requires a root entry and a separately sealed source supervision join.
All original queries are retained; visibility never uses a GT-error filter.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F

H = Path(__file__).resolve().parent
V, P = H.parent, H.parent.parent
DESIGN = V / 'MATCHED_WARM_FRESH_DESIGN_BEFORE_NEW_SOURCE_FIT.json'
DESIGN_SHA = 'ced6d7edb67e5e5c009d840802a759718801a2ae27a5ee3a25f7807964fcdcd7'
HELPER = P / 'query_visibility_public_native_supervision_control_20261024_v148/source_fit_preparation/train_public_native_control.py'
HELPER_SHA = '8613a6a9fe41c0bcb842e6ebb20b0e424f974e3a28df7c3e3343e95eb2899936'
PARENT_PROTOCOL = HELPER.parent / 'PROTOCOL_BEFORE_SOURCE_FIT.json'
PARENT_PROTOCOL_SHA = '17167acf75e9200d987e63842ebcf4142f6f5a5f704996509c5a6d08db65bb16'
S = P / 'query_visibility_mr6d_full_public_source_20261024_v153'
HEAD = V / 'independent_head_preparation/independent_query_head.py'
HEAD_SHA = '3c54dc75d6650aabce8b76a277998fded6f77f22293ba408bdddd998104bb39a'
QUALITY = S / 'head_preparation/saved_synthetic_point_quality_v1/RESULTS.json'
MR_FEATURE_SEAL = S / 'full230_native_DINO_v1/FEATURE_COMPLETE.json'
PALLOX_FEATURE_SEAL = P / 'query_visibility_public_native_features_20261024_v147/public_native_DINO512_v1/FEATURE_COMPLETE.json'
PROTOCOL = H / 'PROTOCOL_BEFORE_SOURCE_FEATURE_OR_CHECKPOINT_BODIES.json'
RUN = V / 'runs/C37_C38_INDEPENDENT_AUX_WARM_FRESH_FIXED8000'
ARMS = ('C37_WARM_VISIBILITY_INDEPENDENT_AUX_FIXED8000', 'C38_FRESH_VISIBILITY_INDEPENDENT_AUX_FIXED8000')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def bind(p):
    return dict(path=str(Path(p).resolve()), sha256=sha(p))

def read(p):
    return json.loads(Path(p).read_text())

def lines(p):
    return [json.loads(s) for s in Path(p).read_text().splitlines() if s]

def check(b):
    assert '_v112' not in b['path'].lower()
    assert sha(b['path']) == b['sha256'], b['path']
    return Path(b['path'])

def write(p, obj):
    with Path(p).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def load_module(name, path, digest):
    assert sha(path) == digest
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def pools(y, available, training):
    return [np.flatnonzero(available & training & (y == k)) for k in (0, 1)]

def draw_balanced(rng, pair, each):
    assert len(pair) == 2 and all(len(p) for p in pair)
    return np.r_[rng.choice(pair[0], each, replace=True), rng.choice(pair[1], each, replace=True)].astype(np.int64)

def labels_and_masks(rows):
    assert len(rows) == 1840
    for i, r in enumerate(rows):
        assert r['global_index'] == i and r['frame_index'] == i//8 and r['native_slot'] == i%8
        assert r['entity_target'] in (-1, 0, 1) and r['role_target'] in (-1, 0, 1)
        assert r['split'] in ('SOURCE_DEVELOPMENT', 'SOURCE_VALIDATION')
        if r['role_target'] >= 0:
            assert r['entity_target'] == 1 and r['role_correspondence_supported']
            assert r['role_state'] == ('D', 'E')[r['role_target']]
        assert r['point_quality_label'] == -1
        assert not r['human_gold'] and r['root_adjudicated_weak']
    return dict(entity=np.array([r['entity_target'] for r in rows], np.int8),
        role=np.array([r['role_target'] for r in rows], np.int8),
        state=np.array([r['role_state'] for r in rows]),
        occluder=np.array([r['occluder_or_uncertainty'] for r in rows]),
        training=np.array([r['split'] == 'SOURCE_DEVELOPMENT' for r in rows]),
        available=np.array([r['original_input_valid'] for r in rows], bool))

def generated_adapter_checks():
    # Meaningful split/availability and class-balancing checks with invented IDs.
    y = np.array([0,1,0,1,0,1,-1,1], np.int8)
    av = np.array([1,1,1,1,0,1,1,1], bool)
    tr = np.array([1,1,0,0,1,0,1,0], bool)
    pair = pools(y,av,tr)
    assert pair[0].tolist()==[0] and pair[1].tolist()==[1]
    a, b = [draw_balanced(np.random.default_rng(1550),pair,16) for _ in range(2)]
    assert np.array_equal(a,b) and np.bincount(y[a],minlength=2).tolist()==[16,16]
    z=torch.tensor([.2,-.4,.8],requires_grad=True)
    l=.5*F.binary_cross_entropy_with_logits(z[:2],torch.zeros(2))+.5*F.binary_cross_entropy_with_logits(z[2:],torch.ones(1))
    l.backward()
    expected=(torch.sigmoid(z.detach())-torch.tensor([0.,0.,1.]))*torch.tensor([.25,.25,.5])
    assert torch.allclose(z.grad,expected,atol=1e-8,rtol=1e-6)
    return dict(PASS=True, invented_CPU_arrays_only=True,
        held_validation_unknown_and_unavailable_never_sampled=True,
        balanced_draws_replay_exact_and_class_balanced=True,
        balanced_BCE_gradient_independently_checked=True,
        source_features_weights_or_model_body_calls=0)

def prepare(label_seal):
    assert sha(DESIGN)==DESIGN_SHA and sha(HELPER)==HELPER_SHA and sha(HEAD)==HEAD_SHA
    assert sha(PARENT_PROTOCOL)==PARENT_PROTOCOL_SHA
    generated=read(V/'independent_head_preparation/GENERATED_INDEPENDENCE_CHECKS.json')
    assert generated['PASS'] and generated['code']==bind(HEAD)
    design=read(DESIGN)
    assert design['source_labels']==bind(label_seal)
    assert design['root_policy']==bind(V/'ROOT_POLICY_BEFORE_FROZEN_PROBE_AND_INDEPENDENT_AUXILIARY_CONTROL.json')
    seal=read(label_seal)
    assert seal['completed'] and seal['PASS'] and seal['source_only'] and seal['source_fit_admitted']
    assert not seal['visibility_scores_used_for_labels']
    assert seal['frames']==230 and seal['rows']==1840 and seal['all_original_queries_retained']
    assert not seal['human_gold'] and not seal['exact_real_point_gold']
    rows=lines(check(seal['labels']))
    meta=labels_and_masks(rows)
    assert meta['training'].sum()==1480 and (~meta['training']).sum()==360
    assert all(len(p) for p in pools(meta['entity'],meta['available'],meta['training']))
    assert all(len(p) for p in pools(meta['role'],meta['available'],meta['training']))
    old=read(PARENT_PROTOCOL);mf=read(MR_FEATURE_SEAL);pf=read(PALLOX_FEATURE_SEAL);quality=read(QUALITY)
    assert mf['completed'] and mf['rows']==1840 and pf['completed']
    assert quality['total_rows']==26624 and quality['all_original_queries_L_order_roles_and_visibility_masks_preserved']
    p=dict(completed=True, prepared_only=True, code=bind(__file__), design=bind(DESIGN),root_policy=design['root_policy'],
        helper=bind(HELPER),head=bind(HEAD),parent_protocol=bind(PARENT_PROTOCOL),
        anchor_checkpoint=old['anchor_checkpoint'],anchor_freeze=old['anchor_freeze'],
        source_supervision_seal=bind(label_seal),source_labels=seal['labels'],
        MR_feature_seal=bind(MR_FEATURE_SEAL),MR_features=mf['features'],MR_inputfacts=mf['inputfacts'],
        Pallox_feature_seal=bind(PALLOX_FEATURE_SEAL),Pallox_features=pf['features'],Pallox_inputfacts=pf['inputfacts'],
        quality=bind(QUALITY),quality_labels=[quality['source_families'][k]['labels'] for k in ('OLD6144','NEW20480')],
        metadata_generated_checks=generated_adapter_checks(),
        original_head_generated_checks=bind(V/'independent_head_preparation/GENERATED_INDEPENDENCE_CHECKS.json'),
        arms=list(ARMS),updates_each=8000,seed_each=1550,original_native35=old['fixed_public_pools_and_native35'],
        runtime=old['runtime'],source_labels_only_read_during_preparation=True,
        feature_checkpoint_or_saved_score_bodies_read_during_preparation=False)
    write(PROTOCOL,p)
    print(json.dumps(dict(completed=True,prepared_only=True,protocol=bind(PROTOCOL))),flush=True)

def binary_metric(y,pred,mask):
    known=mask & np.isin(y,[0,1]);cm=[]
    for t in (0,1):
        cm.append([int((known & (y==t) & (pred==pr)).sum()) for pr in (0,1,-1)])
    def ratio(n,d):return n/d if d else None
    return dict(CM_rows0_1_columns0_1_U=cm,N=int(known.sum()),
        negative_recall=ratio(cm[0][0],sum(cm[0])),positive_recall=ratio(cm[1][1],sum(cm[1])),
        negative_precision=ratio(cm[0][0],cm[0][0]+cm[1][0]),
        positive_precision=ratio(cm[1][1],cm[0][1]+cm[1][1]),
        U_rate=ratio(cm[0][2]+cm[1][2],int(known.sum())),
        all_mask_rows=int(mask.sum()),unknown_truth=int((mask & ~np.isin(y,[0,1])).sum()))

def mr_report(raw,pack,meta):
    av=pack['query_available'];pred=np.where(av,(raw[:,0]>=0).astype(np.int8),-1)
    entity=np.where(av,(raw[:,1]>=0).astype(np.int8),-1)
    gated=np.where(av & (raw[:,1]>=0),pred,-1)
    groups={}
    for name,mask in [('ALL',np.ones(1840,bool)),('SOURCE_DEVELOPMENT',meta['training']),('SOURCE_VALIDATION',~meta['training'])]:
        groups[name]=dict(visibility=binary_metric(meta['role'],pred,mask),
            visibility_inputvalid=binary_metric(meta['role'],pred,mask&av),
            entity=binary_metric(meta['entity'],entity,mask),entity_inputvalid=binary_metric(meta['entity'],entity,mask&av),
            secondary_entity_gated_visibility=binary_metric(meta['role'],gated,mask),
            additional_entity_rejection_U=int((mask&av&(raw[:,1]<0)).sum()),
            semantic_states={str(s):int((mask&(meta['state']==s)).sum()) for s in np.unique(meta['state'])})
        for k in range(8):
            m=mask & (pack['native_slot']==k)
            groups[name+f'::native{k}']=dict(visibility=binary_metric(meta['role'],pred,m),entity=binary_metric(meta['entity'],entity,m))
        for occluder in np.unique(meta['occluder']):
            m=mask & (meta['occluder']==occluder)
            groups[name+'::occluder::'+str(occluder)]=dict(visibility=binary_metric(meta['role'],pred,m),entity=binary_metric(meta['entity'],entity,m))
    return groups

def fit(entry):
    e=read(entry);assert e['approved_actual_source_fit'] and e['source_only']
    assert not any(e[k] for k in ('private_allowed','new_encoder_R0_allowed','PnP_allowed','q_L_change_allowed','automatic_retry'))
    assert e['code']==bind(__file__) and e['protocol']==bind(PROTOCOL) and not RUN.exists()
    p=read(PROTOCOL);assert p['design']==bind(DESIGN) and sha(DESIGN)==DESIGN_SHA
    h=load_module('v155_saved_v148_helpers',HELPER,HELPER_SHA)
    head=load_module('v155_separated_head',HEAD,HEAD_SHA)
    previous=read(check(p['parent_protocol']))
    frozen=[p[k] for k in ('code','design','root_policy','helper','head','parent_protocol','anchor_checkpoint','anchor_freeze',
        'source_supervision_seal','source_labels','MR_feature_seal','MR_features','MR_inputfacts','Pallox_feature_seal',
        'Pallox_features','Pallox_inputfacts','quality','original_head_generated_checks','original_native35')]+p['quality_labels']
    frozen += [previous[k] for k in ('C32_draws','public_labels','public_feature_protocol','fixed_public_pools_and_native35')]
    for native in previous['native']:
        frozen += [native[k] for k in ('feature_seal','features','GT','baseline')]
    for b in frozen:check(b)
    assert Path(sys.executable).resolve()==Path(p['runtime']['executable']).resolve()
    assert torch.__version__==p['runtime']['torch'] and np.__version__==p['runtime']['numpy'] and torch.cuda.is_available()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    oldentry=dict(public_feature_seal=p['Pallox_feature_seal'],public_features=p['Pallox_features'],public_feature_inputfacts=p['Pallox_inputfacts'])
    packs,metas,baselines,public,pub,oldprefix=h.load_inputs(previous,oldentry)
    prefix=[]
    for step,r in enumerate(lines(check(previous['C32_draws'])),1):
        assert r['update']==step and r['extra_source']=='V136_NATIVE'
        ix=np.asarray(r['native_occurrences'],np.int64);ex=np.asarray(r['extra_native_occurrences'],np.int64)
        assert ix.shape==ex.shape==(64,)
        for indices,meta,a in ((ix,metas[0],packs[0]),(ex,metas[1],packs[1])):
            assert (meta['split'][indices]=='TRAIN').all() and a['query_available'][indices].all()
            assert np.array_equal(meta['target'][indices],np.r_[np.ones(16),np.zeros(32),np.ones(16)])
        prefix.append((ix,ex,np.asarray(r['native_pair_pool_positions'],np.int64),np.asarray(r['extra_pair_pool_positions'],np.int64)))
    assert len(prefix)==8000 and len(oldprefix)==2000
    assert all(np.array_equal(a,b) for x,y in zip(prefix[:2000],oldprefix) for a,b in zip(x,y))
    mr=h.packet(p['MR_features']);rr=lines(check(p['source_labels']));rm=labels_and_masks(rr)
    facts=lines(check(p['MR_inputfacts']))
    assert mr['features'].shape==(1840,1928) and mr['features'].dtype==np.float32 and np.isfinite(mr['features']).all()
    for i,(r,f) in enumerate(zip(rr,facts)):
        assert i==f['global_index']==int(mr['global_index'][i])
        assert r['query_id']==f['query_id']==str(mr['query_id'][i])
        assert r['id']==f['frame_id']==str(mr['frame_id'][i]) and r['source_id']==f['source_id']==str(mr['source_id'][i])
        assert r['split']==f['split']==str(mr['split'][i])
        q=np.asarray(r['query_xy'] if r['query_xy'] is not None else [np.nan,np.nan],np.float64)
        assert np.array_equal(mr['query_xy'][i],q,equal_nan=True)
        side=np.nan if r['L'] is None else r['L'];assert (np.isnan(side) and np.isnan(mr['L'][i])) or side==mr['L'][i]
        assert bool(mr['query_available'][i])==r['original_input_valid']==f['native_continuous_valid']
        assert bool(mr['nearest_valid'][i])==r['nearest_valid']==f['nearest_valid']
    assert np.array_equal(mr['query_available'],rm['available'])
    entity_frame_support={}
    for fold,mask in [('TRAIN',rm['training']),('VALIDATION',~rm['training'])]:
        counts={'positive_frames':0,'negative_frames':0,'known_rows':0}
        for frame in np.unique(mr['frame_index'][mask]):
            group=mask & (mr['frame_index']==frame)
            known=np.unique(rm['entity'][group & (rm['entity']>=0)])
            assert len(known)<=1, 'Conflicting entity labels within frame; do not relabel/filter.'
            constructed=np.flatnonzero(group & mr.get('feature_constructed',np.ones(1840,bool)))
            if len(constructed):
                assert np.array_equal(mr['features'][constructed,:384],np.repeat(mr['features'][constructed[:1],:384],len(constructed),axis=0))
            avknown=group & rm['available'] & (rm['entity']>=0)
            counts['known_rows']+=int(avknown.sum())
            if avknown.any():counts['positive_frames' if int(known[0]) else 'negative_frames']+=1
        entity_frame_support[fold]=counts
    qualities=[];quality_pools=[]
    for i,(pack,meta,b) in enumerate(zip(packs,metas,p['quality_labels'])):
        qr=lines(check(b));assert len(qr)==len(pack['features'])
        for j,(r,m) in enumerate(zip(qr,meta['rows'])):
            assert r['global_index']==j and r['query_id']==m['query_id'] and r['split']==m['split']
        target=np.array([-1 if r['point_quality_label'] is None else r['point_quality_label'] for r in qr],np.int8)
        qualities.append(target);quality_pools.append(pools(target,pack['query_available'],meta['split']=='TRAIN'))
        assert all(len(pool) for pool in quality_pools[-1])
    ep=pools(rm['entity'],rm['available'],rm['training']);assert all(len(pool) for pool in ep)
    vi=np.flatnonzero(rm['available']&rm['training']&(rm['role']>=0))
    assert (rm['role'][vi]==0).any() and (rm['role'][vi]==1).any()
    pi=np.sort(np.r_[pub['pools'][0],pub['pools'][1]])
    allpacks=packs+[public,mr]
    for a in allpacks:
        assert not (a['query_available'] & ~a.get('feature_constructed',np.ones(len(a['features']),bool))).any()
    xs=[torch.as_tensor(a['features'],device='cuda') for a in allpacks]
    z42=torch.zeros((max(512,len(vi)),42),device='cuda')
    yn=[torch.as_tensor(m['target'],dtype=torch.float32,device='cuda') for m in metas]
    an=[torch.as_tensor(a['query_available'],device='cuda') for a in packs]
    yp=torch.as_tensor(pub['target'][pi],dtype=torch.float32,device='cuda')
    ym=torch.as_tensor(rm['role'][vi],dtype=torch.float32,device='cuda')
    ye=torch.as_tensor(rm['entity'],dtype=torch.float32,device='cuda')
    yq=[torch.as_tensor(y,dtype=torch.float32,device='cuda') for y in qualities]
    def balance(z,y):
        assert (y==0).any() and (y==1).any()
        return .5*F.binary_cross_entropy_with_logits(z[y==0],y[y==0])+.5*F.binary_cross_entropy_with_logits(z[y==1],y[y==1])
    def predict(model):
        model.eval();outputs=[]
        with torch.inference_mode():
            for a,x in zip(allpacks,xs):
                raw=np.full((len(x),3),np.nan,np.float32)
                ix=np.flatnonzero(a.get('feature_constructed',np.ones(len(x),bool)))
                for lo in range(0,len(ix),512):
                    ii=ix[lo:lo+512];o=model(x[ii],z42[:len(ii)])
                    raw[ii]=torch.stack([o[k] for k in ('external_occlusion','target_entity','coarse_point_quality')],1).cpu().numpy()
                outputs.append(raw)
        return outputs
    def save_outputs(directory,prefix_name,raws):
        for name,raw,a in zip(('OLD6144','NEW20480','PALLOX512','MR1840'),raws,allpacks):
            av=a['query_available'];pred=np.where(av,(raw[:,0]>=0).astype(np.int8),-1)
            np.savez_compressed(directory/f'{prefix_name}_{name}_OUTPUTS.npz',raw_three_logits=raw,
                primary_visibility_prediction=pred,secondary_entity_gated_visibility=np.where(av&(raw[:,1]>=0),pred,-1),
                **{k:a[k] for k in a if k!='features'})
    RUN.mkdir(parents=True);started=time.monotonic();pending={};common_aux_initial=None;common_aux_final=None;common_aux_outputs=None;common_draw=None
    def auxiliary_state_sha(model):
        digest=hashlib.sha256();seen=0
        for name,t in sorted(model.state_dict().items()):
            if name.startswith(('entity_head.','quality_head.')):
                a=t.detach().cpu().contiguous().numpy();digest.update(name.encode()+str(a.dtype).encode()+str(a.shape).encode()+a.tobytes());seen+=a.size
        assert seen==246530
        return digest.hexdigest()
    write(RUN/'STARTED.json',dict(completed=False,entry=bind(entry),protocol=bind(PROTOCOL),actual_updates=0,automatic_retry=False))
    try:
        for arm_index,arm in enumerate(ARMS):
            directory=RUN/arm;directory.mkdir();torch.manual_seed(1550);torch.cuda.manual_seed_all(1550)
            model=head.IndependentAuxiliaryQueryHead().cuda().float()
            factory_fresh_hash=h.state_sha(model)
            if arm_index==0:
                payload=torch.load(p['anchor_checkpoint']['path'],map_location='cpu',weights_only=True)
                assert payload['update']==8000;model.load_visibility_state(payload['state_dict']);del payload
            else:assert h.state_sha(model)==factory_fresh_hash
            aux_initial=auxiliary_state_sha(model)
            if common_aux_initial is None:common_aux_initial=aux_initial
            else:assert common_aux_initial==aux_initial
            assert sum(t.numel() for t in model.parameters())==471043 and not model.geometry.weight.any()
            initial_hash=h.state_sha(model);initial=predict(model);repeat=predict(model)
            assert all(h.eq(a,b) for a,b in zip(initial,repeat))
            if arm_index==0:
                for a,b in zip(initial[:2],baselines):assert h.eq(a[:,0].copy(),b['raw_head_logit_before_U'])
            for raw,a in zip(initial,allpacks):
                ix=a.get('feature_constructed',np.ones(len(raw),bool));assert not raw[ix,1:].any()
            save_outputs(directory,'INITIAL',initial)
            write(directory/'INITIAL_PARITY.json',dict(completed=True,PASS=True,C32_visibility_old_new_raw_bitexact=(arm_index==0),fresh_visibility_factory_state_unchanged_before_fit=(arm_index==1),auxiliary_state_sha=aux_initial,
                all4_datasets_repeated_three_outputs_bitexact=True,auxiliary_logits_initial_zero=True,initial_state=initial_hash))
            vp=list(model.visibility_parameters());ap=list(model.auxiliary_parameters())
            assert not (set(map(id,vp)) & set(map(id,ap)))
            assert set(map(id,vp+ap))==set(map(id,model.parameters()))
            optimizer_vis=torch.optim.AdamW(vp,lr=1e-4,weight_decay=.001)
            optimizer_aux=torch.optim.AdamW(ap,lr=1e-4,weight_decay=.001)
            rng=np.random.default_rng(1550);dh=hashlib.sha256();armstart=time.monotonic()
            with (directory/'ALL8000_DRAWS_AND_LOSSES.jsonl').open('x') as log:
                for step,(oi,ni,opos,npos) in enumerate(prefix,1):
                    ei=draw_balanced(rng,ep,32);qi=[draw_balanced(rng,pair,16) for pair in quality_pools]
                    for ix in (oi,ni,opos,npos,pi,vi,ei,*qi):dh.update(ix.astype('<i8').tobytes())
                    model.train();optimizer_vis.zero_grad(set_to_none=True);optimizer_aux.zero_grad(set_to_none=True)
                    o=[model(xs[i][ix],z42[:len(ix)]) for i,ix in enumerate((oi,ni))]
                    po=model(xs[2][pi],z42[:len(pi)]);mo=model(xs[3][vi],z42[:len(vi)])
                    eo=model(xs[3][ei],z42[:len(ei)])['target_entity'];eo.retain_grad()
                    qo=[model(xs[i][ix],z42[:len(ix)])['coarse_point_quality'] for i,ix in enumerate(qi)]
                    for z in qo:z.retain_grad()
                    nl=[h.q.native_bce(o[i]['external_occlusion'],yn[i][ix],an[i][ix]) for i,ix in enumerate((oi,ni))]
                    pl=balance(po['external_occlusion'],yp);ml=balance(mo['external_occlusion'],ym)
                    el=balance(eo,ye[ei]);ql=sum(balance(z,yq[i][ix]) for i,(z,ix) in enumerate(zip(qo,qi)))/2
                    loss=sum(nl)+pl+ml+el+ql
                    assert torch.isfinite(loss);loss.backward()
                    grad_vis=torch.nn.utils.clip_grad_norm_(vp,5);grad_aux=torch.nn.utils.clip_grad_norm_(ap,5)
                    assert torch.isfinite(grad_vis) and torch.isfinite(grad_aux)
                    optimizer_vis.step();optimizer_aux.step();assert not model.geometry.weight.any()
                    log.write(json.dumps(dict(update=step,old_indices=oi.tolist(),new_indices=ni.tolist(),entity_indices=ei.tolist(),
                        quality_old_indices=qi[0].tolist(),quality_new_indices=qi[1].tolist(),MR_visibility_indices=vi.tolist(),
                        old_native_BCE=float(nl[0].detach()),new_native_BCE=float(nl[1].detach()),Pallox_BCE=float(pl.detach()),
                        MR_BCE=float(ml.detach()),entity_BCE=float(el.detach()),quality_BCE=float(ql.detach()),
                        auxiliary_weight=1,total_loss=float(loss.detach()),visibility_grad_norm=float(grad_vis),auxiliary_grad_norm=float(grad_aux)),allow_nan=False)+'\n')
                    if step%250==0:print(json.dumps(dict(arm=arm,update=step,seconds=time.monotonic()-armstart)),flush=True)
            if common_draw is None:common_draw=dh.hexdigest()
            else:assert common_draw==dh.hexdigest()
            final_hash=h.state_sha(model);assert final_hash!=initial_hash
            final=predict(model);again=predict(model)
            assert all(h.eq(a,b) for a,b in zip(final,again)) and h.state_sha(model)==final_hash
            aux_final=auxiliary_state_sha(model)
            if common_aux_final is None:
                common_aux_final=aux_final;common_aux_outputs=[a[:,1:].copy() for a in final]
            else:
                assert common_aux_final==aux_final
                assert all(h.eq(a[:,1:].copy(),b) for a,b in zip(final,common_aux_outputs))
            save_outputs(directory,'FINAL',final)
            checkpoint=directory/'FINAL_CHECKPOINT.pt'
            torch.save(dict(state_dict=h.q.clone_cpu(model.state_dict()),visibility_optimizer_state=h.q.clone_cpu(optimizer_vis.state_dict()),auxiliary_optimizer_state=h.q.clone_cpu(optimizer_aux.state_dict()),
                update=8000,parent_C32_update=8000,seed=1550,arm=arm,auxiliary_weight=1,visibility_initialization=("C32" if arm_index==0 else "FRESH"),architecture=bind(HEAD),
                policy=bind(DESIGN),protocol=bind(PROTOCOL),selected_model=None),checkpoint)
            pending[arm]=dict(raw=final,checkpoint=bind(checkpoint),draw_sha=dh.hexdigest(),final_state=final_hash,auxiliary_initial_state=aux_initial,auxiliary_final_state=aux_final,seconds=time.monotonic()-armstart)
            del model,optimizer_vis,optimizer_aux;torch.cuda.empty_cache()
        # All held validation reports are emitted only after both fixed fits finish.
        limits=read(check(p['original_native35']))['native35_limits'];summaries={}
        for arm,item in pending.items():
            directory=RUN/arm;raw=item.pop('raw')
            nr=h.native_report([a[:,0].copy() for a in raw[:2]],packs,metas,limits)
            pr=h.public_report(raw[2][:,0].copy(),pub);mrrep=mr_report(raw[3],mr,rm)
            qualityrep={}
            for name,a,m,z,y in zip(('OLD6144','NEW20480'),packs,metas,raw,qualities):
                pred=np.where(a['query_available'],(z[:,2]>=0).astype(np.int8),-1)
                qualityrep[name]={}
                for fold in ('TRAIN','VALIDATION'):
                    mask=m['split']==fold
                    qualityrep[name][fold]=binary_metric(y,pred,mask)
                    for factor in ('material','aspect','foreign_family','geometry_family'):
                        for value in np.unique(m[factor]):
                            qualityrep[name][fold+'::'+factor+'::'+str(value)]=binary_metric(y,pred,mask&(m[factor]==value))
                    slots=np.asarray([r['native_slot'] for r in m['rows']])
                    for slot in range(8):
                        qualityrep[name][fold+f'::native{slot}']=binary_metric(y,pred,mask&(slots==slot))
            held=mrrep['SOURCE_VALIDATION'];vis=held['visibility_inputvalid'];ent=held['entity_inputvalid']
            vals=[vis['negative_recall'],vis['positive_recall'],ent['negative_recall'],ent['positive_recall']]
            sanity=all(v is not None and v>=limit for v,limit in zip(vals,[.90,.80,.80,.80]))
            report=dict(completed=True,arm=arm,update=8000,auxiliary_weight=1,visibility_initialization=("C32" if ARMS.index(arm)==0 else "FRESH"),native=nr,Pallox=pr,MR=mrrep,
                synthetic_point_quality=qualityrep,MR_weak_sanity_PASS=sanity,
                prospective_source_checks_all_pass=bool(nr['native35_all_pass'] and pr['sanity']['PASS'] and sanity),
                selected_model=None,fully_solved=False,adopted=False,current_allowed=False,
                primary_visibility_never_filtered_by_entity_quality_or_GTerror=True,source_labels_weak_not_human_gold=True,
                secondary_entity_gate_logit0_only=True,point_quality_not_a_visibility_gate=True,entity_frame_support=entity_frame_support,**item)
            write(directory/'SOURCE_RESULTS.json',report)
            write(directory/'MODEL_FREEZE.json',dict(completed=True,checkpoint=item['checkpoint'],parameters=471043,update=8000,
                source_checks_all_pass=report['prospective_source_checks_all_pass'],diagnostic_only=True,current_allowed=False,
                selected_model=None,threshold_logit=0,all_geometry_input_zero=True))
            write(directory/'FIT_COMPLETE.json',dict(completed=True,optimizer_updates=8000,source_results=bind(directory/'SOURCE_RESULTS.json'),
                model_freeze=bind(directory/'MODEL_FREEZE.json'),draw_sha=item['draw_sha']))
            summaries[arm]=dict(fit=bind(directory/'FIT_COMPLETE.json'),native35_passed=nr['native35_passed'],MR_weak_sanity_PASS=sanity)
        for b in frozen:check(b)
        write(RUN/'CONTROL_COMPLETE.json',dict(completed=True,PASS=True,arms=summaries,updates_each=8000,
            source_only=True,all_original_native26624_public512_MR1840_retained=True,
            equal_auxiliary_initial_and_final_states_and_outputs_and_common_draws=True,visibility_initialization_is_only_between_arm_intervention=True,draw_sha=common_draw,entity_frame_support=entity_frame_support,
            selected_model=None,fully_solved=False,new_private_or_encoder_R0_PnP_xy_calls=0,
            original_bound_inputs_unchanged=True,code=bind(__file__),protocol=bind(PROTOCOL),entry=bind(entry),seconds=time.monotonic()-started))
        print(json.dumps(dict(completed=True,control=bind(RUN/'CONTROL_COMPLETE.json'))),flush=True)
    except BaseException:
        write(RUN/'FAILED_EXECUTION.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',type=Path);parser.add_argument('--entry',type=Path)
    args=parser.parse_args()
    if args.prepare and not args.entry:prepare(args.prepare)
    elif args.entry and not args.prepare:fit(args.entry)
    else:raise SystemExit('Use --prepare source_supervision_seal or root --entry')
