"""Frozen source-only C23 on the exact saved V132 C16 features; no GT reader."""
from pathlib import Path
import argparse, ast, hashlib, importlib.util, json, os, sys, time, traceback
import numpy as np

sys.dont_write_bytecode = True
H = Path(__file__).resolve().parent
V = H.parent
P = V.parent
POLICY = H / 'POLICY_BEFORE_SAVED_FEATURE_HEAD.json'
PROTOCOL = H / 'PROTOCOL_BEFORE_FORWARD.json'
C16_SEAL = V / 'frozen_C16_actual_initializer_v1/PREDICTION_SEAL.json'
C23_FREEZE = P / 'query_visibility_source_coverage_duration_20261024_v130/runs/C22_C23_FIXED8000_SOURCE_CONTROL/C23_PUBLIC_PLUS_NATIVE_FIXED8000/MODEL_FREEZE.json'
OUT = V / 'frozen_C23_actual_initializer_v1'
N = 6144
COMMON = ('global_index', 'corpus', 'source_occurrence_index', 'occurrence_id', 'scene_index',
          'base_index', 'native_slot', 'query_xy', 'L', 'native_continuous_valid',
          'native_nearest_valid', 'no_detection')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def check(binding):
    assert sha(binding['path']) == binding['sha256'], binding['path']


def same(a, b):
    return a['path'] == b['path'] and a['sha256'] == b['sha256']


def equal(a, b):
    return np.array_equal(a, b, equal_nan=True) if a.dtype.kind == 'f' else np.array_equal(a, b)


def decode(raw, available):
    assert raw.shape == available.shape and np.isfinite(raw[available]).all()
    prediction = np.full(len(raw), -1, np.int8)
    prediction[available] = (raw[available] >= 0).astype(np.int8)
    return prediction


def raw_parity(actual, previous, forwarded):
    assert np.array_equal(actual[forwarded], previous[forwarded])
    assert np.isnan(actual[~forwarded]).all() and np.isnan(previous[~forwarded]).all()


def state_sha(model):
    # Exact C11 state SHA convention, also used by C16 and V130 C23 freezes.
    h = hashlib.sha256()
    for k, t in sorted(model.state_dict().items()):
        a = t.detach().cpu().contiguous()
        h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(str(tuple(a.shape)).encode())
        h.update(a.numpy().tobytes())
    return h.hexdigest()


def prepare():
    assert sha(C16_SEAL) == '71f2080c9c287d5b146d7a35e48a9db629e0f45e689715942bbda0bbb37c2135'
    assert sha(C23_FREEZE) == 'bcd081160a6daf90b5dab25cd13f700e08d543fb9043ddff4d7223dc6f639961'
    c16 = read(C16_SEAL); c23 = read(C23_FREEZE)
    assert c16['completed'] and c16['PASS'] and c16['GT_read'] is False
    check(c16['protocol']); parent = read(c16['protocol']['path'])
    assert c16['rows'] == N and c16['head_forwarded'] == 6088 and c16['available'] == 6056
    assert c23['completed'] and c23['selected_update'] == 8000 and c23['threshold_logit'] == 0
    assert not c23['current_used_for_fit_or_selection'] and c23['fixed_final_not_VAL_selection']
    assert same(c23['head_class_code'], parent['c16']['head_class_code'])
    policy = dict(completed=True, source_only_frozen_C23_diagnosis=True,
                  purpose='Compare existing frozen C23 using identical actual-query C16 saved features after C16 source diagnosis; no model admission or arm choice.',
                  original_source_policy=parent['source_policy'], rows=N, constructible=6088, available=6056,
                  batch512=True, threshold_logit=0, C16_exact_raw_parity_required=True,
                  new_fit_checkpoint_threshold_choice_current223_V112_PnP_q_correction=False,
                  C23_application_admitted=False, GT_read_or_hash=False, automatic_retry=False)
    write(POLICY, policy)
    raw = np.array([-1., -0., 0., 1., np.nan], np.float32)
    av = np.array([True, True, True, True, False]); forwarded = np.array([True, True, True, True, False])
    assert decode(raw, av).tolist() == [0, 1, 1, 1, -1]
    raw_parity(raw, raw.copy(), forwarded)
    broken = raw.copy(); broken[0] = np.nextafter(broken[0], np.float32(0.))
    try:
        raw_parity(broken, raw, forwarded)
    except AssertionError:
        pass
    else:
        raise AssertionError('Raw-score parity must reject a one-ULP change')
    generated = H / 'GENERATED_DECODER_AND_EXACT_PARITY.json'
    write(generated, dict(completed=True, PASS=True, code=bind(__file__),
                         signedzero_raw0_D_E_U_preserved=True, missing_raw_NaN_and_one_ULP_mismatch_rejected=True,
                         actual_saved_array_weight_model_GT_reads=0))
    p = dict(completed=True, code=bind(__file__), policy=bind(POLICY), generated=bind(generated),
             C16_seal=bind(C16_SEAL), C16_protocol=c16['protocol'], C16_features=c16['features'], C16_predictions=c16['predictions'],
             C16_freeze=parent['c16']['source_model_freeze'], C16_checkpoint=parent['c16']['checkpoint'],
             C23_freeze=bind(C23_FREEZE), C23_checkpoint=c23['checkpoint'], head_class_code=c23['head_class_code'],
             runtime=parent['c16']['runtime'], output=str(OUT), rows=N, head_forwarded=6088, available=6056,
             batch=512, threshold_logit=0, common_array_whitelist=list(COMMON),
             GT_read=False, C23_application_admitted=False, no_encoder_RGB_query_change=True)
    write(PROTOCOL, p)
    entry = dict(approved_frozen_C23_saved_feature_diagnosis=False, fit=False, current223=False, V112=False,
                 PnP=False, q_correction=False, model_admission=False, automatic_retry=False,
                 code=p['code'], policy=p['policy'], protocol=bind(PROTOCOL))
    write(H / 'ROOT_ENTRY_TEMPLATE.json', entry)
    print(json.dumps(dict(code=p['code'], protocol=bind(PROTOCOL), entry_template=bind(H/'ROOT_ENTRY_TEMPLATE.json'),
                          command=p['runtime']['executable']+' '+str(Path(__file__).resolve())+' --entry ROOT_ACTUAL_C23_SAVED_FEATURE_ENTRY.json',
                          actual_model_calls=0)), flush=True)


def run(entrypath):
    e = read(entrypath)
    assert e['approved_frozen_C23_saved_feature_diagnosis'] is True
    assert all(e[k] is False for k in ('fit', 'current223', 'V112', 'PnP', 'q_correction', 'model_admission', 'automatic_retry'))
    assert same(e['code'], bind(__file__)) and same(e['protocol'], bind(PROTOCOL)) and same(e['policy'], bind(POLICY))
    p = read(PROTOCOL)
    assert same(p['code'], e['code']) and same(p['policy'], e['policy']) and p['output'] == str(OUT)
    for key in ('generated', 'C16_seal', 'C16_protocol', 'C16_features', 'C16_predictions',
                'C16_freeze', 'C16_checkpoint', 'C23_freeze', 'C23_checkpoint', 'head_class_code'):
        check(p[key])
    assert read(p['generated']['path'])['PASS']
    c16seal = read(p['C16_seal']['path']); c23freeze = read(p['C23_freeze']['path'])
    assert c16seal['completed'] and c16seal['PASS'] and not c16seal['GT_read']
    assert same(c16seal['features'], p['C16_features']) and same(c16seal['predictions'], p['C16_predictions'])
    assert c16seal['all768_scenes6144_slots_preserved'] and c16seal['encoder_state_unchanged'] and c16seal['head_state_unchanged']
    assert c23freeze['selected_update'] == 8000 and c23freeze['threshold_logit'] == 0
    assert same(c23freeze['checkpoint'], p['C23_checkpoint']) and not c23freeze['current_used_for_fit_or_selection']
    with np.load(p['C16_features']['path'], allow_pickle=False) as f:
        features = np.array(f['features'], copy=True)
        can = np.array(f['feature_constructed'], copy=True); available = np.array(f['query_available'], copy=True)
        factual = np.array(f['factual_query_available'], copy=True); context = np.array(f['context_available'], copy=True)
        qinside = np.array(f['Q_inside'], copy=True); common = {k: np.array(f[k], copy=True) for k in COMMON}
    with np.load(p['C16_predictions']['path'], allow_pickle=False) as f:
        before = np.array(f['raw_head_logit_before_U'], copy=True); previous_prediction = np.array(f['prediction'], copy=True)
        assert np.array_equal(can, f['head_forwarded']) and np.array_equal(available, f['query_available'])
        for k in COMMON:
            assert equal(common[k], f[k]), k
    assert features.shape == (N, 1928) and features.dtype == np.float32 and np.isfinite(features).all()
    assert can.dtype == available.dtype == bool and can.shape == available.shape == (N,)
    assert int(can.sum()) == 6088 and int(available.sum()) == 6056
    assert not features[~can].any() and np.array_equal(available, can & factual & context)
    assert np.array_equal(~can, common['no_detection'])
    assert np.array_equal(can & ~available, can & ~(common['native_continuous_valid'] & common['native_nearest_valid']))
    assert np.array_equal(common['global_index'], np.arange(N)) and len(set(common['occurrence_id'])) == N
    assert np.array_equal(features[:, 1927], qinside.astype(np.float32))
    assert np.array_equal(decode(before, available), previous_prediction)
    assert not OUT.exists(), 'Preserve existing output/failure; no automatic retry.'
    OUT.mkdir()
    write(OUT/'STARTED.json', dict(entry=bind(entrypath), code=p['code'], protocol=bind(PROTOCOL),
          features=p['C16_features'], rows=N, head_forwarded=int(can.sum()), available=int(available.sum()), GT_read=False))
    started = time.monotonic()
    try:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        import torch
        assert torch.__version__ == '2.1.1+cu118' and np.__version__ == '1.26.4'
        assert Path(sys.executable).resolve() == Path(p['runtime']['executable']).resolve()
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        node = next(n for n in ast.parse(Path(p['head_class_code']['path']).read_text()).body
                    if isinstance(n, ast.ClassDef) and n.name == 'SpatialTargetQueryHead')
        namespace = {'torch': torch}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), p['head_class_code']['path'], 'exec'), namespace)
        ix = np.flatnonzero(can)
        parity = None
        c23raw = None
        c23state = None
        for name, update, ck in (('C16', 4500, p['C16_checkpoint']), ('C23', 8000, p['C23_checkpoint'])):
            payload = torch.load(ck['path'], map_location='cpu', weights_only=True)
            assert payload['update'] == update and same(payload['head_class_code'], p['head_class_code'])
            head = namespace['SpatialTargetQueryHead']().cuda().float().eval().requires_grad_(False)
            head.load_state_dict(payload['state_dict'], strict=True); del payload
            assert sum(t.numel() for t in head.parameters()) == 221825
            headstate = state_sha(head)
            assert headstate == (c16seal['head_state_sha256'] if name == 'C16' else c23freeze['model_state_sha256'])
            raw = np.full(N, np.nan, np.float32)
            for lo in range(0, len(ix), 512):
                ii = ix[lo:lo+512]
                with torch.inference_mode():
                    x = torch.as_tensor(features[ii], device='cuda'); z = head(x)
                    if lo == 0:
                        assert torch.equal(z, head(x))
                    raw[ii] = z.cpu().numpy()
            assert np.isfinite(raw[can]).all() and np.isnan(raw[~can]).all()
            assert state_sha(head) == headstate and all(not t.requires_grad for t in head.parameters())
            assert all(not m.training for m in head.modules())
            check(ck)
            if name == 'C16':
                raw_parity(raw, before, can)
                assert np.array_equal(decode(raw, available), previous_prediction)
                parity = dict(completed=True, PASS=True, original_predictions=p['C16_predictions'],
                              all6088_raw_scores_bitexact=True, missing56_raw_NaN=True,
                              all6144_predictions_and_input_flags_exact=True, same_constructible_batch512=True)
                write(OUT/'C16_EXACT_REPLAY.json', parity)
            else:
                c23raw, c23state = raw, headstate
            del head
        prediction = OUT/'C23_PREDICTIONS_INPUT_ONLY.npz'
        np.savez_compressed(prediction, raw_head_logit_before_U=c23raw,
                            query_logit=np.where(available, c23raw, 0).astype(np.float32),
                            prediction=decode(c23raw, available), query_available=available, head_forwarded=can,
                            factual_query_available=factual, context_available=context, Q_inside=qinside, **common)
        for key in ('C16_features', 'C16_predictions', 'C16_checkpoint', 'C23_checkpoint'):
            check(p[key])
        seal = dict(completed=True, PASS=True, code=p['code'], protocol=bind(PROTOCOL), policy=p['policy'],
                    entry=bind(entrypath), initializer_seal=c16seal['initializer_seal'],
                    C16_seal=p['C16_seal'], features=p['C16_features'], predictions=bind(prediction),
                    C23_freeze=p['C23_freeze'], checkpoint=p['C23_checkpoint'],
                    C16_exact_replay=bind(OUT/'C16_EXACT_REPLAY.json'), selected_update=8000, threshold_logit=0,
                    head_state_sha256=c23state, head_state_unchanged=True, head_first_repeat_exact=True,
                    all768_scenes6144_slots_preserved=True, rows=N, head_forwarded=6088, available=6056, input_U=88,
                    no_input_raw_NaN=56, finite_OOB_raw_retained_U=32, GT_read=False,
                    no_RGB_encoder_initializer_or_query_change=True, no_new_fit_or_model_threshold_choice=True,
                    C23_application_admitted=False, fit_current_V112_PnP_q_correction=0, seconds=time.monotonic()-started)
        write(OUT/'PREDICTION_SEAL.json', seal)
        print(json.dumps(dict(completed=True, PASS=True, seal=bind(OUT/'PREDICTION_SEAL.json'), seconds=time.monotonic()-started)), flush=True)
    except BaseException:
        write(OUT/'FAILED_NO_AUTOMATIC_RETRY.json', dict(traceback=traceback.format_exc(), automatic_retry=False))
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true'); parser.add_argument('--entry', type=Path)
    args=parser.parse_args()
    if args.prepare:
        assert args.entry is None
        prepare()
    elif args.entry:
        run(args.entry)
    else:
        raise SystemExit('Use --prepare or root --entry')
