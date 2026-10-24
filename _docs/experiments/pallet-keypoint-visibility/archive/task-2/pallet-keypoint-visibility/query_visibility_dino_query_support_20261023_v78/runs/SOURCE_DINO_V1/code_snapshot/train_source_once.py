"""V78 one source-only pooled-DINO fit, gated after feature QA and root approval.

Importing this file does not read any data or instantiate a model.  V77's exact
sampler, natural-VAL selector, report and score serializer are reused by bound
code import.  Only compact DINO inputs and GT/geometry metadata are loaded;
the historical RGB/P3 bank and any backbone are never loaded here.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, random, shutil, sys, time
import numpy as np
import torch
from torch.nn import functional as F

R = Path(__file__).resolve().parent
P = R.parent
V77 = P/'query_visibility_controlled_pairs_20261023_v77'
DECISIONS = R/'EXPERIMENT_DECISIONS_BEFORE_FEATURES.json'
DECISION_SHA = 'a008fc7d35e9aa1d0d4b7d631a796f238aaba7f0aa052fe7da087080b70115e9'
MODEL = R/'model/dino_pooled_query_support_head.py'
RUN = R/'runs/SOURCE_DINO_V1'
ENTRY = R/'TRAINER_ENTRYLOCK.json'
INPUTS = R/'COMPLETED_DINO_INPUT_BINDINGS.json'
SOURCES = ('SYNTHETIC', 'YCB_PUBLIC_REAL')
FORWARD_FIELDS = ('local_features', 'target_features', 'context42', 'validity_fractions')
FEATURE_KEYS = FORWARD_FIELDS + ('query8_xy', 'L', 'query_center_valid',
    'ROI_anyvalid', 'localany', 'targetany', 'id', 'source', 'split',
    'domain', 'original_row_index', 'frame_index')

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1 << 20), b''):
            h.update(x)
    return h.hexdigest()

def b(p):
    return dict(path=str(Path(p).resolve()), sha256=sha(p))

def check(x):
    assert sha(x['path']) == x['sha256'], x['path']

def load(p):
    return json.loads(Path(p).read_text())

def write(p, value):
    p = Path(p)
    assert not p.exists(), str(p)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def imp(binding, name):
    check(binding)
    spec = importlib.util.spec_from_file_location(name, binding['path'])
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def same(a, z):
    return np.array_equal(a, z, equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a, z)

def decisions(check_all=False):
    assert sha(DECISIONS) == DECISION_SHA
    p = load(DECISIONS)
    assert p['head']['parameter_count'] == 48929
    assert p['optimization']['updates'] == 2000
    assert p['optimization']['seed'] == 20261023
    assert p['optimization']['threshold'] == .5
    assert p['admission']['current_stage'].startswith('NO current223')
    if check_all:
        for x in p['bindings'].values():
            check(x)
    return p

def old_helpers(p):
    paired = imp(p['bindings']['V77_source_trainer'], 'bound_v77_exact_helpers')
    source = load(p['bindings']['V77_source_inputs']['path'])
    base = imp(source['bindings']['original_loader_code'], 'bound_v76_exact_admission')
    return paired, base

def metadata_from_compact(d, original, after, paired, base):
    """No features/GT enter model forward; admission is exactly the V77 policy."""
    assert len(original['id']) == 5203 and len(after['id']) == 4799
    Nframes, A = 10002, 4799
    assert np.array_equal(d['frame_index'], np.arange(Nframes))
    orig = after['original_row_index'].astype(np.int64)
    anchor = after['anchor_k'].astype(np.int64)
    assert np.array_equal(after['after_row_index'], np.arange(A))
    assert len(np.unique(orig)) == A and ((orig >= 0) & (orig < 5203)).all()
    assert ((anchor >= 0) & (anchor < 8)).all()
    gt = {k:np.concatenate((original[k], after[k])) for k in original}
    assert len(np.unique(gt['id'])) == Nframes
    for k in ('id', 'source', 'split'):
        assert np.array_equal(d[k], gt[k]), ('compact_identity', k)
    assert np.array_equal(d['domain'], np.r_[np.zeros(5203, np.int8), np.ones(A, np.int8)])
    assert np.array_equal(d['original_row_index'], np.r_[np.arange(5203), orig])
    for k in ('query8_xy', 'L', 'context42', 'query_center_valid',
              'ROI_anyvalid', 'localany', 'targetany'):
        assert same(d[k][5203:], d[k][orig]), ('unchanged_pair_geometry', k)
    for k in ('source', 'split', 'native_GTref8', 'convex_self_proxy8_separate'):
        assert same(after[k], original[k][orig]), ('unchanged_GT_reference', k)
    assert all(after['id'][i] == original['id'][j]+'::V77_OPAQUE_AFTER' for i,j in enumerate(orig))
    primary, square, offset, available = base.admission(gt['label8'],
        d['query_center_valid'], d['ROI_anyvalid'], d['localany'], d['targetany'],
        gt['native_GTref8'], d['query8_xy'], d['L'])
    assert np.array_equal(after['original_primary8'], primary[:5203][orig])
    assert np.array_equal(after['query_valid8'], d['query_center_valid'][5203:])
    assert np.array_equal(after['reference_in_local_square8'], square[5203:])
    assert np.array_equal(orig, np.flatnonzero((primary[:5203] & (original['label8'] == 1)).any(-1)))
    meta = dict(y=gt['label8'].reshape(-1).astype(np.int64),
        primary=primary.reshape(-1), square=square.reshape(-1), offset=offset.reshape(-1, 2),
        qvalid=d['query_center_valid'].reshape(-1), input_available=available.reshape(-1),
        source=np.repeat([SOURCES.index(s) for s in gt['source']], 8),
        train=np.repeat(gt['split'] == 'train', 8), val=np.repeat(gt['split'] == 'val', 8),
        near_boundary=gt['near_boundary8'].reshape(-1), rule=gt['rule8'].reshape(-1),
        self_visible_proxy=gt['convex_self_proxy8_separate'].reshape(-1),
        domain=np.repeat(d['domain'], 8), all_frame_id=gt['id'],
        all_frame_source=gt['source'], all_frame_split=gt['split'],
        after_original_row=orig, anchor_k=anchor, after_source=after['source'],
        after_split=after['split'], N=Nframes*8, A=A)
    before_query = orig*8+anchor
    after_query = (5203+np.arange(A))*8+anchor
    assert np.array_equal(after['anchor_training_target'], np.where(after['pair_anchor_success'], 0, -1))
    good = paired.pair_eligibility(meta['primary'][before_query], meta['y'][before_query],
        meta['primary'][after_query], meta['y'][after_query],
        after['pair_anchor_success'], after['anchor_training_target'])
    is_anchor = np.zeros(meta['N'], bool)
    is_anchor[after_query] = True
    meta.update(pair_before_query=before_query, pair_after_query=after_query, pair_good=good,
        natural_pools={(s,c):np.flatnonzero((meta['domain'] == 0) & meta['primary'] & meta['train'] & (meta['source'] == s) & (meta['y'] == c)) for s in range(2) for c in range(2)},
        nonanchor_pools={(s,c):np.flatnonzero((meta['domain'] == 1) & ~is_anchor & meta['primary'] & meta['train'] & (meta['source'] == s) & (meta['y'] == c)) for s in range(2) for c in range(2)},
        pair_train_pools={s:np.flatnonzero(good & (after['split'] == 'train') & (after['source'] == SOURCES[s])) for s in range(2)},
        valpools={(s,c):np.flatnonzero((meta['domain'] == 0) & meta['primary'] & meta['val'] & (meta['source'] == s) & (meta['y'] == c)) for s in range(2) for c in range(2)})
    assert all(len(v) > 0 for name in ('natural_pools', 'nonanchor_pools', 'pair_train_pools', 'valpools') for v in meta[name].values()), 'No silent empty-pool fallback'
    assert meta['N'] == 80016
    return meta

def preflight_bindings(inputs):
    for x in inputs['bindings'].values():
        check(x)
    man = load(inputs['bindings']['feature_manifest']['path'])
    qa = load(inputs['bindings']['feature_QA']['path'])
    seal = load(inputs['bindings']['feature_seal']['path'])
    assert man['completed'] is True and man['frame_count'] == 10002
    assert qa['PASS'] is True and seal['completed'] is True and seal['PASS'] is True
    assert seal['manifest'] == inputs['bindings']['feature_manifest']
    assert seal['QA'] == inputs['bindings']['feature_QA']
    return man

def load_source(p, inputs, device):
    """Actual source data may only be opened from the root-approved run stage."""
    man = preflight_bindings(inputs)
    paired, base = old_helpers(p)
    parts = {k:[] for k in FEATURE_KEYS}
    for chunk in man['chunks']:
        check(chunk['file'])
        with np.load(chunk['file']['path'], allow_pickle=False) as z:
            for k in FEATURE_KEYS:
                parts[k].append(z[k])
    d = {k:np.ascontiguousarray(np.concatenate(v)) for k,v in parts.items()}
    del parts
    assert d['local_features'].shape == (10002, 8, 2, 384) and d['local_features'].dtype == np.float16
    assert d['target_features'].shape == (10002, 384) and d['target_features'].dtype == np.float16
    assert d['validity_fractions'].shape == (10002, 8, 3) and d['validity_fractions'].dtype == np.float16
    assert d['context42'].shape == (10002, 8, 42) and d['context42'].dtype == np.float32
    for k in FORWARD_FIELDS:
        assert np.isfinite(d[k]).all(), k
    assert ((d['validity_fractions'] >= 0) & (d['validity_fractions'] <= 1)).all()
    assert np.array_equal(d['validity_fractions'][..., 1] > 0, d['localany'])
    assert np.array_equal(d['validity_fractions'][..., 2] > 0, np.broadcast_to(d['targetany'][:, None], (10002, 8)))
    src = load(p['bindings']['V77_source_inputs']['path'])
    olddec = load(p['bindings']['V77_decisions']['path'])
    gtkeys = ('id', 'source', 'split', 'label8', 'native_GTref8',
              'convex_self_proxy8_separate', 'near_boundary8', 'rule8')
    check(olddec['bindings']['teacher_GT'])
    check(src['bindings']['after_teacher_GT'])
    with np.load(olddec['bindings']['teacher_GT']['path'], allow_pickle=False) as z:
        original = {k:z[k] for k in gtkeys}
    with np.load(src['bindings']['after_teacher_GT']['path'], allow_pickle=False) as z:
        after = {k:z[k] for k in gtkeys + ('original_row_index', 'after_row_index', 'anchor_k',
            'pair_anchor_success', 'anchor_training_target', 'original_primary8',
            'query_valid8', 'reference_in_local_square8')}
    meta = metadata_from_compact(d, original, after, paired, base)
    # Reconcile every domain/source/split class and all training pool counts
    # with the already sealed independent V77 audit; no old feature bank read.
    oldaudit = load(inputs['bindings']['V77_independent_metadata_facts']['path'])
    assert oldaudit['PASS'] is True
    for row in oldaudit['counts']:
        domain = 0 if row['domain'] == 'NATURAL_ORIGINAL' else 1
        s = SOURCES.index(row['source'])
        m = (meta['domain'] == domain) & (meta['source'] == s) & meta[row['split']]
        expected = dict(all_slots=int(m.sum()), U=int(np.sum(m & (meta['y'] == -1))),
            C=int(np.sum(m & (meta['y'] == 0))), V=int(np.sum(m & (meta['y'] == 1))),
            primary_C=int(np.sum(m & meta['primary'] & (meta['y'] == 0))),
            primary_V=int(np.sum(m & meta['primary'] & (meta['y'] == 1))))
        for k,v in expected.items():
            assert row[k] == v, ('Changed_V77_cohort', row['source'], row['split'], k)
    counts = pool_counts(meta)
    expected_pools = oldaudit['train_pools']
    for s, name in enumerate(SOURCES):
        for c in range(2):
            assert counts['natural'][f'{name}|{c}'] == expected_pools[f'natural|{name}|{c}']
            assert counts['after_nonanchor'][f'{name}|{c}'] == expected_pools[f'after_nonanchor|{name}|{c}']
        assert counts['pairs_train'][name] == expected_pools[f'matched_pairs|{name}']
    gpu = {k:torch.from_numpy(d[k] if k == 'target_features' else d[k].reshape(-1, *d[k].shape[2:])).to(device) for k in FORWARD_FIELDS}
    del d
    return paired, base, gpu, torch.as_tensor(meta['y'], device=device), meta

def pool_counts(meta):
    return dict(natural={f'{SOURCES[s]}|{c}':len(meta['natural_pools'][s,c]) for s in range(2) for c in range(2)},
        after_nonanchor={f'{SOURCES[s]}|{c}':len(meta['nonanchor_pools'][s,c]) for s in range(2) for c in range(2)},
        pairs_train={SOURCES[s]:len(meta['pair_train_pools'][s]) for s in range(2)})

def forward(model, gpu, ids):
    ix = torch.as_tensor(ids, dtype=torch.long, device=gpu['local_features'].device)
    return model(**{k:v[ix//8 if k == 'target_features' else ix] for k,v in gpu.items()})

def predict(model, gpu, ids, labels, meta):
    model.eval()
    zs, ns = [], []
    with torch.inference_mode():
        for start in range(0, len(ids), 128):
            chosen = ids[start:start+128]
            out = forward(model, gpu, chosen)['support_logit']
            assert torch.isfinite(out).all()
            zs.append(out.cpu().numpy())
            ns.append(F.binary_cross_entropy_with_logits(out,
                labels[torch.as_tensor(chosen, device=labels.device)].clamp(0, 1).float(),
                reduction='none').cpu().numpy())
    return dict(logits=np.concatenate(zs), NLL_F32=np.concatenate(ns),
                input_available=meta['input_available'][ids])

def generated(output):
    """Only own protocol/code and artificial tensors; no source payloads opened."""
    p = decisions()
    paired = imp(p['bindings']['V77_source_trainer'], 'generated_exact_v77_sampler')
    basebinding = dict(path=str(P/'query_visibility_reference_local_support_20261023_v76/mixed_fit/train_mixed_once.py'), sha256='dab2c7accabad96d52f02ba80a7448a5835490a08498642bbbfe22860775e908')
    base = imp(basebinding, 'generated_v76_admission')
    M = imp(b(MODEL), 'generated_v78_head')
    torch.set_num_threads(1)
    torch.manual_seed(20261023)
    model = M.DinoPooledQuerySupportHead()
    assert sum(v.numel() for v in model.parameters()) == 48929
    source = np.repeat([0, 1], 48)
    y = np.tile(np.repeat([0, 1], 24), 2)
    meta = dict(natural_pools={(s,c):np.flatnonzero((source == s) & (y == c))[:8] for s in range(2) for c in range(2)},
        nonanchor_pools={(s,c):np.flatnonzero((source == s) & (y == c))[8:16] for s in range(2) for c in range(2)},
        pair_train_pools={0:np.arange(4), 1:np.arange(4, 8)},
        pair_before_query=np.r_[np.arange(40, 44), np.arange(88, 92)],
        pair_after_query=np.r_[np.arange(16, 20), np.arange(64, 68)])
    rng = np.random.default_rng(20261023)
    sampler_hash = hashlib.sha256()
    for _ in range(2000):
        ids = paired.sample(meta, rng)
        sampler_hash.update(ids.tobytes())
        assert len(ids) == 96
        assert all(np.sum((source[ids] == s) & (y[ids] == c)) == 24 for s in range(2) for c in range(2))
        assert all(y[ids[k]] == 1 and y[ids[k+1]] == 0 and source[ids[k]] == source[ids[k+1]] for k in range(64, 96, 2))
    assert paired.pair_eligibility(np.ones(3, bool), np.ones(3, int), np.ones(3, bool), np.array([0,0,-1]), np.array([1,0,1], bool), np.array([0,-1,0])).tolist() == [True,False,False]
    yy = np.array([[1,0,-1,1,0,1,1,0]])
    q = np.zeros((1,8,2)); r = q.copy(); r[0,3] = [.5,0]; r[0,4] = [-.5,0]; r[0,5] = [0,np.nan]
    qv = np.ones((1,8),bool); qv[0,6] = False
    roi = np.ones((1,8),bool); roi[0,7] = False
    pm, sq, _, _ = base.admission(yy, qv, roi, np.ones_like(roi), np.ones(1,bool), r, q, np.ones(1))
    assert pm.tolist() == [[True,True,False,False,True,False,False,False]]
    assert sq[0,4] and not sq[0,3]
    z = torch.randn(96, requires_grad=True); target = torch.as_tensor(y).float()
    loss = F.binary_cross_entropy_with_logits(z, target)
    parts = torch.stack([F.binary_cross_entropy_with_logits(z[i:i+32],target[i:i+32]) for i in range(0,96,32)])
    assert torch.allclose(loss, parts.mean()); loss.backward(); assert torch.isfinite(z.grad).all()
    # Selector: four original VAL strata only; after rows cannot influence it.
    fake = dict(N=12, valpools={(s,c):np.array([s*4+c*2,s*4+c*2+1]) for s in range(2) for c in range(2)})
    ids = np.arange(12); nll = np.arange(12, dtype=np.float32)
    a,_ = paired.natural_val_metric(dict(NLL_F32=nll), ids, fake)
    nll[8:] = 1e6
    aa,_ = paired.natural_val_metric(dict(NLL_F32=nll), ids, fake)
    assert a == aa == 3.5
    # The trainer gathers targets by frame, including repeated query IDs.
    gpu = dict(local_features=torch.randn(24,2,384).half(),target_features=torch.randn(3,384).half(),context42=torch.randn(24,42),validity_fractions=torch.rand(24,3).half())
    out = forward(model, gpu, np.array([8,0,23,8]))['support_logit']
    assert out.shape == (4,) and not out.any()
    F.binary_cross_entropy_with_logits(out, torch.tensor([0.,1.,1.,0.])).backward()
    assert torch.isfinite(model.fusion[-1].weight.grad).all()
    write(output, dict(PASS=True, code=b(Path(__file__)), decisions=b(DECISIONS), model=b(MODEL),
        old_sampler=p['bindings']['V77_source_trainer'], old_admission=basebinding,
        parameter_count=48929, generated_sampler_batches=2000, sampler_draws_sha256=sampler_hash.hexdigest(),
        source_class24=True, equal32components=True, paired_VC_adjacent=True,
        failed_pair_rejected=True, halfopen_admission_unchanged=True,
        natural_VAL_only_fourstrata=True, frame_target_gather=True,
        no_source_payloads_or_weights_read=True, optimizer_updates=0, current_reads=0, PnP=0))
    print(json.dumps(dict(PASS=True, generated=b(output))), flush=True)

def freeze(args):
    p = decisions(check_all=True)
    proof = load(R/'GENERATED_TRAINER_TESTS.json')
    hp = load(R/'model/GENERATED_HEAD_TESTS.json')
    assert proof['PASS'] and proof['code'] == b(Path(__file__)) and proof['model'] == b(MODEL)
    assert hp['PASS'] and hp['model'] == b(MODEL)
    auth = load(args.authorization)
    assert auth['approved'] is True and auth['allow_one_source_fit'] is True
    assert auth['decisions'] == b(DECISIONS) and auth['trainer'] == b(Path(__file__))
    assert auth['model'] == b(MODEL) and auth['current_reads_allowed'] is False
    bindings = dict(feature_manifest=b(args.feature_manifest), feature_QA=b(args.feature_qa),
        feature_seal=b(args.feature_seal), V77_source_inputs=p['bindings']['V77_source_inputs'],
        V77_independent_metadata_facts=b(V77/'independent_saved_audit/INDEPENDENT_PAIR_DATA_AND_INPUT_FACTS_AUDIT.json'))
    for k in ('feature_manifest','feature_QA','feature_seal'):
        assert auth[k] == bindings[k]
    inputs = dict(completed=True, schema='V78_COMPACT_DINO_INPUT_BINDINGS', bindings=bindings,
        frame_count=10002, query_slots=80016, no_GT_forward=True, current_reads=0, PnP=0)
    preflight_bindings(inputs)
    # Freeze one immutable entry only after completed feature QA and explicit root authorization.
    assert not INPUTS.exists() and not ENTRY.exists() and not RUN.exists()
    write(INPUTS, inputs)
    write(ENTRY, dict(schema='V78_ONE_SOURCE_DINO_FIT_ENTRY', approved=True,
        decisions=b(DECISIONS), trainer=b(Path(__file__)), model_code=b(MODEL),
        generated=b(R/'GENERATED_TRAINER_TESTS.json'), head_generated=b(R/'model/GENERATED_HEAD_TESTS.json'),
        source_inputs=b(INPUTS), authorization=b(args.authorization), run_directory=str(RUN),
        runtime=dict(python=str(Path('/home/minjae/anaconda3/bin/python').resolve()), torch='2.5.1+cu121', device='cuda:0', threads=1),
        restart_after_first_update=False, forward_fields=list(FORWARD_FIELDS), current_read_or_PnP=False))
    print(json.dumps(dict(ENTRY_FROZEN=b(ENTRY))), flush=True)

def run():
    p = decisions(check_all=True)
    entry = load(ENTRY)
    assert entry['approved'] is True and entry['trainer'] == b(Path(__file__))
    assert entry['model_code'] == b(MODEL) and entry['decisions'] == b(DECISIONS)
    assert entry['source_inputs'] == b(INPUTS) and entry['run_directory'] == str(RUN)
    for k in ('generated','head_generated','authorization'):
        check(entry[k])
    auth = load(entry['authorization']['path'])
    assert auth['approved'] is True and auth['allow_one_source_fit'] is True and auth['current_reads_allowed'] is False
    inputs = load(INPUTS); preflight_bindings(inputs)
    assert not RUN.exists(), 'Preserve existing run, including failed pre-update attempts; no automatic restart'
    assert Path(sys.executable).resolve() == Path(entry['runtime']['python']).resolve()
    assert str(torch.__version__) == entry['runtime']['torch'] and torch.cuda.is_available()
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Set deterministic launch environment before run'
    RUN.mkdir(parents=True); (RUN/'code_snapshot').mkdir()
    for binding in (entry['trainer'],entry['model_code'],entry['decisions'],entry['source_inputs'],entry['authorization'],p['bindings']['V77_source_trainer']):
        shutil.copy2(binding['path'],RUN/'code_snapshot'/Path(binding['path']).name)
    write(RUN/'RUN_STARTED.json',dict(entry=b(ENTRY),optimizer_updates=0,current_reads=0,PnP=0))
    torch.set_num_threads(1); torch.manual_seed(20261023); torch.cuda.manual_seed_all(20261023)
    np.random.seed(20261023); random.seed(20261023)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.cuda.reset_peak_memory_stats(); start=time.monotonic()
    paired,base,gpu,labels,meta=load_source(p,inputs,'cuda:0')
    M=imp(entry['model_code'],'bound_random_v78_model'); model=M.DinoPooledQuerySupportHead().cuda()
    assert sum(v.numel() for v in model.parameters()) == 48929
    assert not model.fusion[-1].weight.any() and not model.fusion[-1].bias.any()
    ids=paired.sample(meta,np.random.default_rng(20261023)); ix=torch.as_tensor(ids,device='cuda')
    out=forward(model,gpu,ids)['support_logit']; loss=F.binary_cross_entropy_with_logits(out,labels[ix].float())
    assert torch.isfinite(loss); loss.backward(); model.zero_grad(set_to_none=True)
    assert meta['primary'][ids].all() and meta['train'][ids].all()
    assert all(np.sum((meta['source'][ids]==s)&(meta['y'][ids]==c))==24 for s in range(2) for c in range(2))
    pools=pool_counts(meta)
    write(RUN/'SOURCE_GPU_PREFLIGHT.json',dict(PASS=True,loss=float(loss.detach()),pools=pools,
        frames=10002,slots=80016,optimizer_updates=0,GTforward=False,backbone_or_R0_forward=0,
        current_reads=0,PnP=0,peak_GPU_memory_bytes=torch.cuda.max_memory_allocated()))
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},RUN/'UNTRAINED_UPDATE0.pt')
    allids=np.arange(meta['N']); validids=np.sort(np.concatenate(list(meta['valpools'].values())))
    zero=predict(model,gpu,allids,labels,meta)
    assert not zero['logits'].any()
    paired.save_scores(RUN/'BASELINE0_SOURCE_SCORES.npz',zero,meta)
    write(RUN/'BASELINE0_RESULTS.json',dict(selection_eligible=False,results=paired.report(base,zero,meta)))
    del zero
    opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.001,betas=(.9,.999))
    rng=np.random.default_rng(20261023); best=float('inf'); beststep=None; history=[]
    for step in range(1,2001):
        ids=paired.sample(meta,rng); ix=torch.as_tensor(ids,device='cuda')
        assert meta['primary'][ids].all() and meta['train'][ids].all()
        model.train(); opt.zero_grad(set_to_none=True)
        z=forward(model,gpu,ids)['support_logit']
        loss=F.binary_cross_entropy_with_logits(z,labels[ix].float())
        assert torch.isfinite(loss); loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.); assert torch.isfinite(grad)
        opt.step()
        if step==1:
            write(RUN/'FIRST_UPDATE.json',dict(update=1,loss=float(loss.detach()),restart_allowed=False,current_reads=0,PnP=0,entry=b(ENTRY)))
            print('FIRST_UPDATE_COMMITTED',flush=True)
        if step%100==0:
            val=predict(model,gpu,validids,labels,meta)
            metric,by=paired.natural_val_metric(val,validids,meta); saved=metric<best
            if saved:
                best=metric; beststep=step
                torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},RUN/'BEST_SUPPORT_HEAD.pt')
            rec=dict(update=step,train_loss=float(loss.detach()),naturalVAL4strata_macroBCE=metric,
                stratumBCE=by,saved=saved,best_update=beststep,seconds=time.monotonic()-start)
            history.append(rec); (RUN/'HISTORY.json').write_text(json.dumps(history,indent=2,allow_nan=False)+'\n')
            print(json.dumps(rec),flush=True)
    model.load_state_dict(torch.load(RUN/'BEST_SUPPORT_HEAD.pt',map_location='cuda',weights_only=True))
    model.eval().requires_grad_(False)
    write(RUN/'MODEL_FREEZE.json',dict(completed=True,optimizer_updates=2000,best_update=beststep,
        best_naturalVAL4strata_macroBCE=best,checkpoint=b(RUN/'BEST_SUPPORT_HEAD.pt'),
        model_code=entry['model_code'],decisions=b(DECISIONS),trainer=b(Path(__file__)),entry=b(ENTRY),
        source_inputs=b(INPUTS),parameter_count=48929,threshold=.5,random_initialization=True,
        current_used_for_fit_or_checkpoint_or_threshold_selection=False,PnP=0,fully_solved=False))
    selected=predict(model,gpu,allids,labels,meta)
    paired.save_scores(RUN/'SELECTED_SOURCE_SCORES.npz',selected,meta)
    write(RUN/'SOURCE_RESULTS.json',dict(completed=True,model_freeze=b(RUN/'MODEL_FREEZE.json'),
        best_update=beststep,selection_naturalVALmacroBCE=best,results=paired.report(base,selected,meta),
        pools=pools,scope='Weak reference-local support; public GT-derived query emulation. Natural VAL only selector, no human semantic gold or current evaluation.',
        GTforward=False,current_reads=0,current_PnP=0))
    write(RUN/'FIT_COMPLETE.json',dict(completed=True,optimizer_updates=2000,best_update=beststep,
        model_freeze=b(RUN/'MODEL_FREEZE.json'),history=b(RUN/'HISTORY.json'),
        scores=b(RUN/'SELECTED_SOURCE_SCORES.npz'),source_results=b(RUN/'SOURCE_RESULTS.json'),
        seconds=time.monotonic()-start,peak_GPU_memory_bytes=torch.cuda.max_memory_allocated(),current_PnP=0))
    print(json.dumps(dict(COMPLETE=b(RUN/'FIT_COMPLETE.json'),best_update=beststep,best_BCE=best)),flush=True)

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=('generated','freeze','run'))
    ap.add_argument('--output',type=Path,default=R/'GENERATED_TRAINER_TESTS.json')
    ap.add_argument('--feature-manifest',type=Path)
    ap.add_argument('--feature-qa',type=Path)
    ap.add_argument('--feature-seal',type=Path)
    ap.add_argument('--authorization',type=Path)
    args=ap.parse_args()
    if args.stage=='generated':
        generated(args.output)
    elif args.stage=='freeze':
        assert all(getattr(args,k) is not None for k in ('feature_manifest','feature_qa','feature_seal','authorization'))
        freeze(args)
    else:
        run()
