"""Fixed RBF basis from eligible source TRAIN inputs, never target quality.

Import and selfcheck are pure. Actual build requires a sealed BASIS_PROTOCOL.
The existing NPZ container includes VAL input rows; only eligible TRAIN rows
are selected. No label container, validation quality, or real data is opened.
"""
from . import common as C
from scripts.research.pallet_pose_anchor_risk_20261001_v1 import convex_train as RISK
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

CONTEXT_DIM = 189
RBF_DIM = 64
BASIS_SCHEMA = 'pallet_pose_anchor_rbf_basis_v1'
RUNTIME_SCHEMA = 'pallet_pose_anchor_rbf_runtime_v1'
CENTER_RULE = 'sha256_identity_first64_byte_distinct_float64_context'
BANDWIDTH_RULE = 'median_positive_pairwise_squared_center_distance'
EXPERTS = ('R0', 'DIVERSE251_s1', 'DIVERSE251_s2', 'DIVERSE251_s3')
HYP = ('long-face-front', 'short-face-front')
READS = None


def array_sha(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(str(value.dtype).encode())
    h.update(json.dumps(list(value.shape)).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def identity_json(fid, expert, hypothesis):
    assert isinstance(fid, str) and expert in EXPERTS and hypothesis in HYP
    return json.dumps([fid, expert, hypothesis], ensure_ascii=False, separators=(',', ':'))


def select_centers(identities, contexts, count=RBF_DIM):
    contexts = np.asarray(contexts)
    assert contexts.dtype == np.float64 and contexts.shape == (len(identities), CONTEXT_DIM)
    assert np.isfinite(contexts).all() and count == RBF_DIM
    canonical = [identity_json(*identity) for identity in identities]
    assert len(set(canonical)) == len(canonical), 'POOL_IDENTITY_MUST_BE_UNIQUE'
    hashes = [hashlib.sha256(s.encode('utf-8')).hexdigest() for s in canonical]
    order = sorted(range(len(canonical)), key=lambda i: (hashes[i], canonical[i]))
    seen, chosen = set(), []
    for i in order:
        key = np.ascontiguousarray(contexts[i], dtype=np.float64).tobytes()
        if key in seen:
            continue
        seen.add(key)
        chosen.append(i)
        if len(chosen) == count:
            break
    assert len(chosen) == count, 'LESS_THAN_64_BYTE_DISTINCT_CONTEXTS_STOP_NO_FALLBACK'
    centers = np.ascontiguousarray(contexts[chosen], dtype=np.float64)
    selected = [dict(identity=list(identities[i]), canonical_json=canonical[i], identity_sha256=hashes[i],
                     context_sha=array_sha(contexts[i]), pool_index=int(i)) for i in chosen]
    return centers, selected


def bandwidth_squared(centers):
    centers = np.asarray(centers, np.float64)
    assert centers.shape == (RBF_DIM, CONTEXT_DIM) and np.isfinite(centers).all()
    distance = np.asarray([np.sum((centers[i]-centers[j])**2, dtype=np.float64)
                           for i in range(RBF_DIM) for j in range(i+1, RBF_DIM)], np.float64)
    assert np.isfinite(distance).all() and (distance >= 0).all()
    positive = distance[distance > 0]
    assert len(positive), 'NO_POSITIVE_FINITE_BANDWIDTH_STOP_NO_FALLBACK'
    width = float(np.median(positive))
    assert np.isfinite(width) and width > 0
    return width


def validate_basis(basis, mean, std):
    assert basis['schema'] == RUNTIME_SCHEMA
    assert basis['context_dim'] == CONTEXT_DIM and basis['rbf_dim'] == RBF_DIM
    centers = np.asarray(basis['centers'], np.float64)
    assert centers.shape == (RBF_DIM, CONTEXT_DIM) and np.isfinite(centers).all()
    assert len({np.ascontiguousarray(row).tobytes() for row in centers}) == RBF_DIM
    width = float(basis['bandwidth_squared'])
    assert np.isfinite(width) and width > 0 and width == bandwidth_squared(centers)
    mean, std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
    assert mean.shape == std.shape == (94,) and np.isfinite(mean).all() and np.isfinite(std).all()
    assert (std >= np.float32(1e-6)).all()
    assert basis['normalization_sha'] == array_sha(np.stack([mean, std]))
    return centers, width


def map_context(context, valid, basis):
    """Append deterministic RBF64; no second normalization or label input."""
    context, valid = np.asarray(context), np.asarray(valid)
    assert context.dtype == np.float64 and valid.dtype == bool
    assert context.shape == (*valid.shape, CONTEXT_DIM) and np.isfinite(context).all()
    assert not context[~valid].any()
    assert basis['schema'] == RUNTIME_SCHEMA and basis['context_dim'] == CONTEXT_DIM and basis['rbf_dim'] == RBF_DIM
    centers = np.asarray(basis['centers'], np.float64)
    width = float(basis['bandwidth_squared'])
    assert centers.shape == (RBF_DIM, CONTEXT_DIM) and np.isfinite(centers).all()
    assert np.isfinite(width) and width > 0
    out = np.zeros((*valid.shape, CONTEXT_DIM+RBF_DIM), np.float64)
    out[:, :, :CONTEXT_DIM] = context
    active = context[valid]
    mapped = np.empty((len(active), RBF_DIM), np.float64)
    # Direct squared differences avoid cancellation in ||x||²+||c||²-2x.c.
    # Bounded chunks limit scratch memory without changing reduction order.
    for start in range(0, len(active), 128):
        d2 = np.sum((active[start:start+128, None, :]-centers[None, :, :])**2, axis=2, dtype=np.float64)
        assert np.isfinite(d2).all() and (d2 >= 0).all()
        mapped[start:start+128] = np.exp(-d2 / (2.*width))
    out[valid, CONTEXT_DIM:] = mapped
    assert np.isfinite(out).all() and not out[~valid].any()
    assert (out[valid, CONTEXT_DIM:] >= 0).all() and (out[valid, CONTEXT_DIM:] <= 1).all()
    return out


def install_guard():
    global READS
    if READS is not None:
        return
    READS = set()
    def hook(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        name, mode, flags = str(p), args[1], args[2]
        writing = ((isinstance(mode, str) and any(c in mode for c in 'wax+')) or
                   (isinstance(flags, int) and bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))))
        if writing:
            assert p == C.DOC/'RBF_BASIS.json', ('BASIS_WRITE_DENIED', name)
            return
        assert not any(token in name for token in ('/data/evaluation/', '/real_gt_v2/', '/annotations/',
            'SOURCE_TRAIN_LABELS', 'SOURCE_VAL_GATE', 'SOURCE_VAL_METRICS', 'REAL_FEATURE', 'REAL_CHOICES',
            'GEOMETRY_SIDETABLE', 'SYNTH_RECORDS', 'SYNTH_LABELS', 'POSE_METRICS', 'CANDIDATE_BOUNDS',
            'GEOMETRY_RESOLVED_POSE_GT', 'SOURCE_MANIFEST')), ('BASIS_QUALITY_OR_RAW_REFERENCE_DENIED', name)
        assert p.suffix.lower() not in ('.png','.jpg','.jpeg','.pt','.pth','.onnx'), ('BASIS_IMAGE_CHECKPOINT_DENIED',name)
        if p.suffix == '.npz':
            assert p == C.PARENT_RAW/'SOURCE_FEATURES.npz', ('BASIS_ARRAY_DENIED', name)
        if p.is_relative_to(C.ROOT): READS.add(str(p.relative_to(C.ROOT)))
    sys.addaudithook(hook)


def load_input_only(protocol):
    """Read only fixed eligibility and input features; no target containers."""
    install_guard()
    required = {'source_contract','source_features','source_feature_lock','parent_train_protocol'}
    assert set(protocol['inputs']) == required
    for binding in protocol['inputs'].values(): C.verify(binding)
    assert protocol['inputs']['parent_train_protocol'] == C.bind(C.RISK_DOC/'TRAIN_PROTOCOL.json')
    parent = C.read(C.ROOT/protocol['inputs']['parent_train_protocol']['path'])
    for new, old in [('source_contract','source_contract'),('source_features','features'),('source_feature_lock','feature_lock')]:
        assert protocol['inputs'][new] == parent['inputs'][old]
    contract = C.read(C.ROOT/protocol['inputs']['source_contract']['path'])
    assert contract['complete'] and contract['status'] == 'PASS'
    eligible = set(contract['fit_eligibility']['eligible_ids']['TRAIN'])
    assert len(eligible)==2598 and eligible.isdisjoint(contract['fit_eligibility']['eligible_ids']['VAL'])
    lock = C.read(C.ROOT/protocol['inputs']['source_feature_lock']['path'])
    assert lock['features'] == protocol['inputs']['source_features']
    assert not lock['source_targets_read'] and not lock['real_targets_read']
    with np.load(C.ROOT/protocol['inputs']['source_features']['path'],allow_pickle=False) as z:
        assert tuple(z['hypothesis_names'].tolist()) == HYP, 'SOURCE_FEATURE_HYPOTHESIS_SLOT_CONTRACT'
        all_ids, split = z['ids'], z['split']
        idx = np.asarray([i for i,fid in enumerate(all_ids) if fid in eligible],np.int64)
        ids = all_ids[idx]
        assert len(set(ids.tolist()))==len(ids)==2598 and set(ids.tolist())==eligible
        assert (split[idx]=='TRAIN').all()
        anchor = z['R0_GEO_index'][idx]
        features = {m:z[m+'_geo'][idx] for m in EXPERTS}
        valid = {m:z[m+'_valid'][idx] for m in EXPERTS}
    assert anchor.dtype==np.int64 and anchor.shape==(2598,)
    for m in EXPERTS:
        assert features[m].dtype==np.float32 and features[m].shape==(2598,2,94)
        assert valid[m].dtype==bool and valid[m].shape==(2598,2)
        assert np.isfinite(features[m][valid[m]]).all()
    vals = features['R0'][valid['R0']]
    mean, std = vals.mean(0), np.maximum(vals.std(0),np.float32(1e-6))
    assert mean.dtype==std.dtype==np.float32
    identities,vectors=[],[]
    contexts={}
    contexts['R0']=RISK.context_inputs(features['R0'],valid['R0'],anchor,mean,std)
    for expert in EXPERTS[1:]:
        paired=RISK.context_inputs(np.concatenate([features['R0'],features[expert]],axis=1),
            np.concatenate([valid['R0'],valid[expert]],axis=1),anchor,mean,std)
        np.testing.assert_array_equal(paired[:,:2],contexts['R0'])
        contexts[expert]=paired[:,2:]
    for expert in EXPERTS:
        for i,fid in enumerate(ids.tolist()):
            for j,hyp in enumerate(HYP):
                if valid[expert][i,j]:
                    identities.append([fid,expert,hyp]);vectors.append(contexts[expert][i,j])
    vectors=np.ascontiguousarray(vectors,dtype=np.float64)
    return dict(ids=ids,source_index=idx,anchor_index=anchor,mean=mean,std=std,
                normalization_sha=array_sha(np.stack([mean,std])),identities=identities,vectors=vectors,
                input_hashes={m:dict(raw_sha=array_sha(features[m]),valid_sha=array_sha(valid[m]),
                                    context189_sha=array_sha(contexts[m])) for m in EXPERTS})


def build():
    install_guard()
    protocol=C.protocol('BASIS_PROTOCOL')
    for key,val in dict(train_rows=2598,context_dim=CONTEXT_DIM,rbf_dim=RBF_DIM,center_rule=CENTER_RULE,
                        bandwidth_rule=BANDWIDTH_RULE,normalization='old_float32_then_float64',extra_normalization=False).items():
        assert protocol[key]==val,(key,protocol.get(key))
    if (C.DOC/'RBF_BASIS.json').exists():
        previous=C.read(C.DOC/'RBF_BASIS.json')
        assert previous['complete'] and previous['PASS'] and previous['protocol']==C.bind(C.DOC/'BASIS_PROTOCOL.json')
        validate_basis(previous['basis'],previous['mean'],previous['std'])
        return previous
    data=load_input_only(protocol)
    centers,selected=select_centers(data['identities'],data['vectors'])
    width=bandwidth_squared(centers)
    runtime=dict(schema=RUNTIME_SCHEMA,context_dim=CONTEXT_DIM,rbf_dim=RBF_DIM,centers=centers.tolist(),
                 bandwidth_squared=width,normalization_sha=data['normalization_sha'])
    validate_basis(runtime,data['mean'],data['std'])
    output=dict(schema=BASIS_SCHEMA,complete=True,PASS=True,created_at=C.now(),protocol=C.bind(C.DOC/'BASIS_PROTOCOL.json'),
        basis=runtime,center_rule=CENTER_RULE,bandwidth_rule=BANDWIDTH_RULE,extra_normalization=False,
        mean=data['mean'].tolist(),std=data['std'].tolist(),normalization_sha=data['normalization_sha'],
        center_records=selected,center_identities=[r['identity'] for r in selected],
        center_identity_sha256=[r['identity_sha256'] for r in selected],centers_sha=array_sha(centers),
        pool_identities_sha=hashlib.sha256(json.dumps(data['identities'],ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),
        pool_context_sha=array_sha(data['vectors']),pool_candidates=len(data['identities']),
        train_rows=2598,source_ids_sha=array_sha(data['ids']),source_index_sha=array_sha(data['source_index']),
        anchor_index_sha=array_sha(data['anchor_index']),input_hashes=data['input_hashes'],
        input_bindings=protocol['inputs'],source_TRAIN_only=True,labels_read=False,VAL_quality_read=False,real_targets_read=False,
        feature_container_disclosure='SOURCE_FEATURES.npz contains TRAIN and VAL inputs; only fixed eligible TRAIN2598 rows are used.',
        normalization_source='All valid R0 candidates in fixed eligible TRAIN rows; float32 mean/std with floor float32(1e-6).',
        objective_evaluations=0,new_fits=0,new_routes=0,read_paths=sorted(READS))
    C.save(C.DOC/'RBF_BASIS.json',output)
    return output


def selfcheck():
    rng=np.random.default_rng(2026100164)
    vectors=rng.normal(size=(70,189)).astype(np.float64)
    vectors[1]=vectors[0]
    identities=[[f'toy-{i:03d}','R0',HYP[i%2]] for i in range(70)]
    centers,records=select_centers(identities,vectors)
    perm=rng.permutation(len(vectors))
    centers2,records2=select_centers([identities[i] for i in perm],vectors[perm])
    np.testing.assert_array_equal(centers,centers2)
    assert [r['identity'] for r in records]==[r['identity'] for r in records2]
    mean,std=np.zeros(94,np.float32),np.ones(94,np.float32)
    basis=dict(schema=RUNTIME_SCHEMA,context_dim=189,rbf_dim=64,centers=centers.tolist(),
               bandwidth_squared=bandwidth_squared(centers),normalization_sha=array_sha(np.stack([mean,std])))
    validate_basis(basis,mean,std)
    ctx=vectors[:8].reshape(2,4,189);valid=np.array([[1,1,1,1],[1,0,0,0]],bool);ctx[~valid]=0
    x=map_context(ctx,valid,basis)
    np.testing.assert_array_equal(x[:,:,:189],ctx)
    expected=np.zeros((2,4,64),np.float64)
    for i,j in zip(*np.nonzero(valid)):
        for k,center in enumerate(centers): expected[i,j,k]=np.exp(-np.sum((ctx[i,j]-center)**2)/(2*basis['bandwidth_squared']))
    np.testing.assert_array_equal(x[:,:,189:],expected)
    assert not x[~valid].any()
    try:select_centers(identities[:63],vectors[:63])
    except AssertionError:pass
    else:raise AssertionError('Too few centers accepted')
    print(json.dumps(dict(PASS=True,pure_toy=True,centers64=True,permutation_invariant=True,scalar_RBF_bit_exact=True,actual_inputs_read=0)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['selfcheck','build']);args=parser.parse_args()
    if args.action=='selfcheck':selfcheck()
    else:print(json.dumps(dict(PASS=build()['PASS'],basis=str(C.DOC/'RBF_BASIS.json'))))
