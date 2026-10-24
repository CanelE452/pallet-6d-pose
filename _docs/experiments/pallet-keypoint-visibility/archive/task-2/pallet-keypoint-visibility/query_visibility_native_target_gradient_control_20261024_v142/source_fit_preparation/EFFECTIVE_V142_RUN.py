def run(entry_path):
    e = read(entry_path); assert e['approved_source_fit'] is True
    assert all(e[k] is False for k in ('current223', 'V112', 'PnP', 'q_correction', 'automatic_retry'))
    assert sha(POLICY) == POLICY_SHA and same(e['code'], bind(__file__)) and same(e['protocol'], bind(PROTOCOL))
    p = read(PROTOCOL)
    validate_extra_bindings(p, e)
    for key in ('code', 'policy', 'generated'):
        assert same(e[key], p[key])
    for key in ('policy', 'generated', 'parent_trainer', 'source_binding_policy', 'source_helper', 'historical_C24_fit', 'historical_C24_trace',
                'historical_C26_fit', 'historical_C26_trace', 'pair_coverage', 'C16_freeze', 'C16_checkpoint', 'new_data_policy', 'new_render',
                'new_initializer_seal', 'new_feature_seal', 'new_diagnosis', 'new_GT_ledger', 'pair_input_motion_diagnosis'):
        check(p[key])
    assert read(GENERATED)['PASS'] and same(read(GENERATED)['code'], e['code']) and not RUN.exists()
    source = read(p['source_binding_policy']['path']); data = q.load_source_data_zero_geometry(source)
    oldmeta = data['nmeta'][CONVENTIONS[0]]; oldpairs = q.make_pairpool(oldmeta, data['npack']['query_available']); q.assert_pair_coverage(oldpairs)
    new = load_new(p); newmeta = new['meta'][CONVENTIONS[0]]
    baseline_old = q.old_reports(data, data['baseline']['query_logit']); baseline_native = q.native_report(data['nbaseline']['raw_head_logit_before_U'], data['npack']['query_available'], oldmeta)
    baseline_new = new_report(new['baseline']['raw_head_logit_before_U'], new['packet']['query_available'], newmeta)
    assert m.NNATIVE == NATIVE and len(oldpairs) == 125
    assert torch.__version__ == p['runtime']['torch'] and np.__version__ == p['runtime']['numpy'] and torch.cuda.is_available()
    assert Path(sys.executable).resolve() == Path(p['runtime']['executable']).resolve()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    xo = torch.as_tensor(data['op']['features'], device='cuda'); xf = torch.as_tensor(data['npk']['features'], device='cuda'); xt = torch.cat((xo, xf))
    units = torch.as_tensor(data['units'], device='cuda', dtype=torch.long)
    oldtarget = torch.as_tensor(data['oldmeta']['external_target'], device='cuda', dtype=torch.float32); oldav = torch.as_tensor(data['available'], device='cuda')
    xn = torch.as_tensor(data['npack']['features'], device='cuda'); xx = torch.as_tensor(new['packet']['features'], device='cuda')
    nt = torch.as_tensor(oldmeta['target'], device='cuda', dtype=torch.float32); na = torch.as_tensor(data['npack']['query_available'], device='cuda')
    et = torch.as_tensor(newmeta['target'], device='cuda', dtype=torch.float32); ea = torch.as_tensor(new['packet']['query_available'], device='cuda')
    zeros = torch.zeros((max(len(xo), len(xf), 512), 42), device='cuda')
    ni_all = np.flatnonzero(data['npack']['feature_constructed']); ei_all = np.flatnonzero(new['packet']['feature_constructed'])
    RUN.mkdir(parents=True); write(RUN / 'STARTED.json', dict(entry=bind(entry_path), protocol=bind(PROTOCOL), actual_updates=0,
        retained_old51004_native6144_new20480=True, old_pair_count125=len(oldpairs), new_pair_count=len(new['pairs']), actual_geometry_input_used=False))
    write(RUN / 'BASELINE_REPORT_BEFORE_FIT.json', dict(old=baseline_old, native_CF=baseline_native, new_native_CF=baseline_new))
    write(RUN / 'NEW_NATIVE_TRAIN_POOL_BEFORE_FIT.json', dict(completed=True, pairs=new['pairs'].tolist(), uniform_D=new['pools'][0].tolist(),
        uniform_E=new['pools'][1].tolist(), only_TRAIN_CF_state_canonical_availability=True, historical_area_qerror_modelscore_VAL_not_filters=True))
    started = time.monotonic(); arms = {}; shared_reference = None; extra_rng_reference = None
    try:
        for arm in ARMS:
            directory = RUN / arm; directory.mkdir(); torch.manual_seed(1330); torch.cuda.manual_seed_all(1330)
            model = q.GeometryQueryHead().cuda().float(); payload = torch.load(p['C16_checkpoint']['path'], map_location='cpu', weights_only=True)
            assert payload['update'] == 4500
            missing, unexpected = model.load_state_dict(payload['state_dict'], strict=False)
            assert missing == ['geometry.weight'] and not unexpected and not model.geometry.weight.any()
            assert all(torch.equal(v.cpu(), payload['state_dict'][k]) for k, v in model.state_dict().items() if k != 'geometry.weight'); del payload
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.001)
            gens = [torch.Generator().manual_seed(s) for s in (208, 214, 215, 216, 217)]; flatgen = torch.Generator().manual_seed(218)
            pg = torch.Generator().manual_seed(1334); ug = torch.Generator().manual_seed(1332)
            epg = torch.Generator().manual_seed(1374); eug = torch.Generator().manual_seed(1372)
            extra_pool, extra_uniform = new['pairs'], new['pools']
            ex, extarget, exav = xx, et, ea
            shared_hash = hashlib.sha256(); extra_hash = hashlib.sha256()
            exposure_old = np.zeros(NOLD, np.int64); exposure_native = np.zeros(NATIVE, np.int64); exposure_new = np.zeros(NEWN, np.int64)

            def checkpoint(step):
                return dict(state_dict=q.clone_cpu(model.state_dict()), optimizer_state=q.clone_cpu(optimizer.state_dict()), update=step,
                    C16_parent_update=4500, arm=arm, threshold_logit=0, policy=bind(POLICY), trainer=bind(__file__), RNG=dict(
                    old=[g.get_state() for g in gens], flat=flatgen.get_state(), native_pair=pg.get_state(), native_uniform=ug.get_state(),
                    extra_pair=epg.get_state(), extra_uniform=eug.get_state(), torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state_all()))

            @torch.no_grad()
            def predict():
                model.eval(); rawunits = torch.cat((model(xo, zeros[:len(xo)]), model(xf, zeros[:len(xf)])))
                oldz = np.where(data['available'], rawunits[units].cpu().numpy(), 0).astype(np.float32)
                outputs = []
                for tensor, constructed, size in ((xn, ni_all, NATIVE), (xx, ei_all, NEWN)):
                    raw = np.full(size, np.nan, np.float32)
                    for lo in range(0, len(constructed), 512):
                        indices = constructed[lo:lo + 512]; ix = torch.as_tensor(indices, device='cuda', dtype=torch.long)
                        raw[indices] = model(tensor[ix], zeros[:len(ix)]).cpu().numpy()
                    outputs.append(raw)
                return oldz, outputs[0], outputs[1]

            @torch.no_grad()
            def evaluate(step):
                oldz, raw, eraw = predict()
                _, parts = data['h'].selector14(data['c12'], oldz, data['available'], data['meta'], data['new'], data['d'])
                nr = q.native_report(raw, data['npack']['query_available'], oldmeta)
                en = new_report(eraw, new['packet']['query_available'], newmeta)
                with (directory / 'HISTORY.jsonl').open('a') as f:
                    f.write(json.dumps(dict(update=step, old14_BCE_parts=parts, old14=q.recall14(oldz, data['available'], data['oldmeta']),
                        native_TRAIN=nr['TRAIN'], native_VAL=nr['VALIDATION'], new_native_TRAIN=en['TRAIN'], new_native_VAL=en['VALIDATION']), allow_nan=False) + '\n')
                return oldz, raw, eraw

            old0, native0, new0 = evaluate(0)
            parity = dict(completed=True, all51004_old_scores_bitexact=equal_array(old0, data['baseline']['query_logit']),
                all6144_native_raw_scores_bitexact=equal_array(native0, data['nbaseline']['raw_head_logit_before_U']),
                all20480_new_raw_scores_bitexact=equal_array(new0, new['baseline']['raw_head_logit_before_U']),
                original_constructed_compact512_batches_and_last_lengths_exact=True, available_only_compaction=False, geometry_weights_zero=True, actual_updates=0)
            parity['PASS'] = bool(parity['all51004_old_scores_bitexact'] and parity['all6144_native_raw_scores_bitexact'] and parity['all20480_new_raw_scores_bitexact'])
            # Convert NumPy booleans before saving any replay receipt.
            parity = {k: bool(v) if isinstance(v, np.bool_) else v for k, v in parity.items()}
            write(directory / 'UPDATE0_C16_EXACT_PARITY.json', parity); assert parity['PASS'], 'Initialization parity failure; preserve before any fit.'
            torch.save(checkpoint(0), directory / 'INITIAL_CHECKPOINT.pt')
            with (directory / 'TRAIN_DRAWS_AND_LOSSES.jsonl').open('x') as trace, Path(p['historical_C26_trace']['path']).open() as previous, Path(p['historical_C24_trace']['path']).open() as c24, Path(p['historical_C29_trace']['path']).open() as c29:
                for step in range(1, 8001):
                    ix, pairs, audit, *_ = data['h'].draw192(data['c12'], data['c10'], data['pools'], data['mp'], data['wp'], data['fp'], gens, flatgen)
                    ni, npi = q.draw_native64(oldpairs, data['native_pools'], pg, ug)
                    ei, epi = q.draw_native64(extra_pool, extra_uniform, epg, eug)
                    prior = json.loads(next(previous)); original = json.loads(next(c24))
                    historical = json.loads(next(c29)); assert_draw_replay(historical, step, ix, pairs, ni, npi, ei, epi)
                    assert prior['update'] == original['update'] == step
                    for history in (prior, original):
                        assert np.array_equal(ix, history['old_occurrences']) and np.array_equal(pairs, history['old_matched_cover_sham_positions'])
                    assert np.array_equal(ni, prior['native_occurrences']) and np.array_equal(npi, prior['native_pair_pool_positions'])
                    assert (data['oldmeta']['split'][ix] == 'TRAIN').all() and data['available'][ix].all()
                    assert (oldmeta['split'][ni] == 'TRAIN').all() and data['npack']['query_available'][ni].all()
                    assert np.array_equal(oldmeta['target'][ni], np.r_[np.ones(16, np.int8), np.zeros(32, np.int8), np.ones(16, np.int8)])
                    extra_meta = newmeta
                    assert (extra_meta['split'][ei] == 'TRAIN').all()
                    extra_available = new['packet']['query_available']
                    assert extra_available[ei].all()
                    assert np.array_equal(extra_meta['target'][ei], np.r_[np.ones(16, np.int8), np.zeros(32, np.int8), np.ones(16, np.int8)])
                    for value in (ix, pairs, ni): shared_hash.update(value.astype('<i8').tobytes())
                    extra_hash.update(ei.astype('<i8').tobytes()); extra_hash.update(epi.astype('<i8').tobytes())
                    np.add.at(exposure_old, ix, 1); np.add.at(exposure_native, ni, 1)
                    np.add.at(exposure_new, ei, 1)
                    oi = torch.as_tensor(ix, device='cuda', dtype=torch.long); nn = torch.as_tensor(ni, device='cuda', dtype=torch.long)
                    ee = torch.as_tensor(ei, device='cuda', dtype=torch.long); pp = torch.as_tensor(pairs, device='cuda', dtype=torch.long)
                    model.train(); optimizer.zero_grad(set_to_none=True)
                    zo = model(xt[units[oi]], zeros[:192]); zn = model(xn[nn], zeros[:64]); ze = model(ex[ee], zeros[:64])
                    oldloss, bce, hinge, flatbce = data['h'].loss192(zo, oldtarget[oi], oldav[oi], pp)
                    nl = q.native_bce(zn, nt[nn], na[nn]); el = q.native_bce(ze, extarget[ee], exav[ee]); loss = oldloss * 0. + nl + el
                    zo.retain_grad()
                    assert torch.isfinite(loss); loss.backward()
                    assert zo.grad is not None and torch.count_nonzero(zo.grad).item() == 0
                    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5); assert torch.isfinite(grad); optimizer.step()
                    assert not model.geometry.weight.any()
                    trace.write(json.dumps(dict(update=step, old_occurrences=ix.tolist(), old_matched_cover_sham_positions=pairs.tolist(),
                        native_occurrences=ni.tolist(), native_pair_pool_positions=npi.tolist(), extra_native_occurrences=ei.tolist(),
                        extra_pair_pool_positions=epi.tolist(), extra_source='V136_NATIVE', AUX_old192_loss_coefficient=0, AUX_old192_output_gradient_nonzero=0, AUX_old192_full_loss_value=float(oldloss.detach()),
                        loss=float(loss.detach()), old_BCE=float(bce.detach()), old_hinge=float(hinge.detach()), flat_BCE=float(flatbce.detach()),
                        native_BCE=float(nl.detach()), extra_native_BCE=float(el.detach()), native_hinge_weight=0,
                        gradient_before_clip=float(grad)), allow_nan=False) + '\n')
                    if step % 100 == 0: evaluate(step)
                    if step % 1000 == 0: print(json.dumps(dict(arm=arm, updates=step, seconds=time.monotonic() - started)), flush=True)
                assert next(previous, None) is None and next(c24, None) is None and next(c29, None) is None
            assert shared_hash.hexdigest() == p['historical_C29_draw_hash'] and extra_hash.hexdigest() == p['historical_C29_extra_draw_hash']
            shared = shared_hash.hexdigest(); rng_hash = hashlib.sha256(epg.get_state().numpy().tobytes() + eug.get_state().numpy().tobytes()).hexdigest()
            if shared_reference is None: shared_reference, extra_rng_reference = shared, rng_hash
            else: assert shared == shared_reference and rng_hash == extra_rng_reference
            torch.save(checkpoint(8000), directory / 'FINAL_RESUME_CHECKPOINT.pt'); oldz, raw, eraw = predict()
            report = q.old_reports(data, oldz); nr = q.native_report(raw, data['npack']['query_available'], oldmeta)
            en = new_report(eraw, new['packet']['query_available'], newmeta)
            gates = q.make_gates(report, nr, baseline_old['old14'], baseline_native); gates.update(new_gates(en, baseline_new)); assert len(gates) == 64
            rank = [v['value'] for k, v in gates.items() if not k.startswith('HISTORICAL15::')]; assert len(rank) == 49
            worst = None if any(x is None for x in rank) else min(rank)
            eligible = all(g['PASS'] for g in gates.values())
            split_report = qualification_split(gates)
            write(directory / 'QUALIFICATION_SPLIT.json', split_report)
            write(directory / 'FIXED8_CASES_AND_TWO_STRIPS.json', fixed_case_scores(eraw, new['packet'], newmeta, p))
            np.savez_compressed(directory / 'FINAL_OLD51004_SOURCE_SCORES.npz', query_logit=oldz,
                prediction=np.where(data['available'], (oldz >= 0).astype(np.int8), -1), query_available=data['available'],
                global_occurrence_index=np.arange(NOLD), input_unit_index=data['units'], external_target=data['oldmeta']['external_target'],
                raw_teacher_state=data['oldmeta']['typed_state'], split=data['oldmeta']['split'], domain=data['oldmeta']['domain'], source=data['oldmeta']['source'], variant=data['oldmeta']['variant'])
            saved_native(directory / 'FINAL_NATIVE6144_SCORES_INPUT_AND_OUTPUT.npz', raw, data['npack'], oldmeta, NATIVE)
            saved_native(directory / 'FINAL_NEW_NATIVE20480_SCORES_INPUT_AND_OUTPUT.npz', eraw, new['packet'], newmeta, NEWN)
            np.savez_compressed(directory / 'SAMPLING_EXPOSURE.npz', old51004=exposure_old, native6144=exposure_native, new_native20480=exposure_new)
            write(directory / 'SOURCE_RESULTS.json', dict(completed=True, arm=arm, selected_update=8000, VAL_checkpoint_choice=False, threshold_logit=0,
                old=report, native_CF=nr, native_V4_sensitivity_only=q.native_report(raw, data['npack']['query_available'], data['nmeta'][CONVENTIONS[1]]),
                new_native_CF=en, new_native_obsolete_area_V4_sensitivity_only=new_report(eraw, new['packet']['query_available'], new['meta'][CONVENTIONS[1]]),
                gates=gates, qualification_split=split_report, AUX_old192_loss_coefficient=0, selected_model=None, diagnostic_only=True, interpretation_limits=p['interpretation_limits'], source_eligible=eligible, worst_required49_class_recall=worst, shared_old192_native64_sha256=shared,
                extra_native_draw_sha256=extra_hash.hexdigest(), extra_rng_consumption_sha256=rng_hash,
                old_pair_count=125, new_pair_count=len(new['pairs']), historical_old192_and_native64_all8000_exact=True,
                all_old51004_native6144_new20480_preserved=True, teacher_SU_no_external_gradient=True, actual_geometry_input_used=False,
                semantic_target='Declared CF registered native reference visibility; constructed anchors coincide with support mesh, separate from actual query coordinate error; no error filtering.', current_V112_PnP=0))
            write(directory / 'MODEL_FREEZE.json', dict(completed=True, arm=arm, update=8000, selected_update=8000,
                checkpoint=bind(directory / 'FINAL_RESUME_CHECKPOINT.pt'), threshold_logit=0, parameters=224513, source_eligible=eligible,
                policy=bind(POLICY), trainer=bind(__file__), C16_freeze=p['C16_freeze'], all_geometry_input_zero=True,
                encoder_trainable_parameters=0, selected_model=None, diagnostic_only=True, AUX_old192_loss_coefficient=0, current_allowed=False, V112_allowed=False, PnP=0))
            write(directory / 'FIT_COMPLETE.json', dict(completed=True, optimizer_updates=8000, arm=arm,
                results=bind(directory / 'SOURCE_RESULTS.json'), freeze=bind(directory / 'MODEL_FREEZE.json'),
                initialization_parity=bind(directory / 'UPDATE0_C16_EXACT_PARITY.json'), shared_draw_sha256=shared,
                source_eligible=eligible, worst_required49_class_recall=worst, qualification_split=bind(directory / 'QUALIFICATION_SPLIT.json'), fixed_cases=bind(directory / 'FIXED8_CASES_AND_TWO_STRIPS.json'), selected_model=None, diagnostic_only=True))
            arms[arm] = dict(source_eligible=eligible, worst_required49_class_recall=worst,
                fit=bind(directory / 'FIT_COMPLETE.json'), freeze=bind(directory / 'MODEL_FREEZE.json'))
            del model, optimizer; torch.cuda.empty_cache()
        eligible = [a for a in ARMS if arms[a]['source_eligible']]
        selected = None  # Root policy: no model admission/selection in this diagnostic phase.
        write(RUN / 'CONTROL_COMPLETE.json', dict(completed=True, PASS=True, arms=arms, selected_arm_source_only=selected,
            same_all8000_old192_native64_draws=True, shared_draw_sha256=shared_reference, extra_same_algorithm_rng_consumption=True,
            extra_rng_consumption_sha256=extra_rng_reference, extra_occurrence_ids_all8000_exact_C29=True, final_update8000_single_arm=True, new_baseline_fits=0, AUX_old192_loss_coefficient=0, model_selection_prohibited=True, selected_model=None,
            old42_and_new22_required_gates=True, original27_plus_new22_rank=True, application_solved=False,
            private_current_V112_PnP_q_correction_calls=0, seconds=time.monotonic() - started, code=bind(__file__), policy=bind(POLICY), protocol=bind(PROTOCOL)))
        print(json.dumps(read(RUN / 'CONTROL_COMPLETE.json')), flush=True)
    except BaseException:
        write(RUN / 'FAILED_EXECUTION.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
