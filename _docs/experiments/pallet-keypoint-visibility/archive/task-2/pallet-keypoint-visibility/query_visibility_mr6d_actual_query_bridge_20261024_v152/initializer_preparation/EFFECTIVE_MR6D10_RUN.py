def run(entry):
    e = read(entry)
    assert e['approved_actual_public_initializer'] is True and all((e[k] is False for k in FORBIDDEN))
    assert sha(POLICY) == POLICY_SHA and same(e['code'], bind(__file__)) and same(e['protocol'], bind(PROTOCOL))
    p = read(PROTOCOL)
    for k in ('code', 'policy', 'generated', 'inputs'):
        assert same(e[k], p[k])
        check(p[k])
    for k in ('reference_code', 'reference_protocol', 'backbone_code', 'checkpoint', 'native_grid'):
        check(p[k])
    assert read(GENERATED)['PASS'] and (not OUT.exists()) and (p['frames'] == 10) and (p['native_slots'] == 80) and (p['output'] == str(OUT))
    rows = lines(INPUTS)
    assert len(rows) == 10 and all((set(r) == INPUT_KEYS and r['frame_index'] == i for (i, r) in enumerate(rows)))
    validate_transport(e, p, rows)
    OUT.mkdir()
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['YOLO_CONFIG_DIR'] = str(OUT / 'runtime/Ultralytics')
    os.environ['MPLCONFIGDIR'] = str(OUT / 'runtime/matplotlib')
    import cv2, torch
    assert str(torch.__version__) == p['runtime']['torch'] and cv2.__version__ == p['runtime']['cv2'] and (Path(sys.executable).resolve() == Path(p['runtime']['python']).resolve())
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    spec = importlib.util.spec_from_file_location('v152_exact_R0_factory', p['backbone_code']['path'])
    factory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(factory)
    model = factory.FrozenYoloFeatures(p['checkpoint']['path'], device=p['runtime']['device'])

    def image(row):
        assert sha(row['image']) == row['image_sha256']
        rgb = np.load(check(row['canonical_RGB_NPY']), allow_pickle=False)
        return array_to_bgr(rgb, row)
    hooks = [0]
    handle = model.yolo.model.register_forward_pre_hook(lambda *_: hooks.__setitem__(0, hooks[0] + 1))
    loaded = state(model.yolo.model)
    model.predict(np.zeros(tuple(rows[0]['raw_hw']) + (3,), np.uint8), already_padded=False, cpu_features=True)
    warm = state(model.yolo.model)
    warmhooks = hooks[0]
    write(OUT / 'STARTED.json', dict(completed=True, entry=bind(entry), protocol=bind(PROTOCOL), inputs=p['inputs'], loaded_state_SHA=loaded, warmed_state_SHA=warm, generated_blank_warmup_hooks=warmhooks, source_forward_calls=0, GT_read=False))
    proposals = []
    queries = []
    qpack = {k: [] for k in query_arrays(dict(candidates=[], selected_index=None), [1, 1])}
    offsets = [0]
    candidate_arrays = {k: [] for k in ('box_xyxy', 'keypoints_xy', 'keypoints_conf', 'score')}
    first = None
    firstp3 = None
    start = time.monotonic()
    try:
        for (i, row) in enumerate(rows):
            z = model.predict(image(row), already_padded=False, cpu_features=True)
            s = serial(z)
            (gain, offset) = factory.canvas_affine(z['canvas_shape'], z['input_shape'])
            qa = query_arrays(z, row['raw_hw'])
            proposals.append(dict(**row, **s, affine_gain=gain, affine_offset=offset.tolist(), actual_original_initializer=True, source_emulation=False))
            for (key, value) in qa.items():
                qpack[key].append(value)
            for c in z['candidates']:
                for key in candidate_arrays:
                    candidate_arrays[key].append(np.asarray(c[key], np.float64))
            offsets.append(offsets[-1] + len(z['candidates']))
            for k in range(8):
                queries.append(clean(dict(query_id=row['id'] + ':native' + str(k), frame_index=i, frame_id=row['id'], source=row['source'], source_id=row['source_id'], split=row['split'], native_slot=k, query_xy=qa['query_xy'][k], L=qa['L'][k], no_detection=z['selected_index'] is None, finite_non_sentinel=qa['finite_non_sentinel'][k], native_continuous_valid=qa['continuous_valid'][k], query_nearest_pixel_valid=qa['nearest_valid'][k], query_RGB96_centre_domain_valid=qa['RGBcentre_valid'][k], native_grid_any_valid=qa['native_grid_valid'][k].any(), association_status='UNASSIGNED_BY_INPUT_PRODUCER', actual_original_initializer=True, source_emulation=False)))
            if i == 0:
                first = s
                firstp3 = z['p3'].numpy().copy()
        repeat = model.predict(image(rows[0]), already_padded=False, cpu_features=True)
        assert serial(repeat) == first and repeat['p3'].numpy().tobytes() == firstp3.tobytes()
        assert state(model.yolo.model) == warm and sha(p['checkpoint']['path']) == p['checkpoint']['sha256'] and all((not m.training for m in model.yolo.model.modules())) and all((not p.requires_grad for p in model.yolo.model.parameters()))
        for (name, records) in (('FROZEN_PROPOSALS_ALL10_INPUT_ONLY.jsonl', proposals), ('ORIGINAL_NATIVE_QUERY80_INPUT_ONLY.jsonl', queries)):
            with (OUT / name).open('x') as f:
                for row in records:
                    f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        arrays = {key: np.concatenate(value, axis=0) for (key, value) in qpack.items()}
        arrays.update(global_index=np.arange(80, dtype=np.int32), frame_index=np.repeat(np.arange(10, dtype=np.int32), 8), native_slot=np.tile(np.arange(8, dtype=np.int8), 10), query_id=np.array([r['query_id'] for r in queries]), frame_id=np.repeat([r['id'] for r in rows], 8), split=np.repeat([r['split'] for r in rows], 8), no_detection=np.repeat([z['selected_index'] is None for z in proposals], 8))
        np.savez_compressed(OUT / 'ORIGINAL_NATIVE_QUERY80_AND_VALID9X9_INPUT_ONLY.npz', **arrays)
        sizes = {'box_xyxy': (4,), 'keypoints_xy': (9, 2), 'keypoints_conf': (9,), 'score': ()}
        allc = {key: np.stack(value) if value else np.empty((0,) + sizes[key], np.float64) for (key, value) in candidate_arrays.items()}
        np.savez_compressed(OUT / 'ALL_CANDIDATES_RAW9_AND_BOXES_INPUT_ONLY.npz', frame_offsets=np.array(offsets, np.int32), selected_index=np.array([-1 if z['selected_index'] is None else z['selected_index'] for z in proposals], np.int32), **allc)
        handle.remove()
        model.close()
        result = dict(completed=True, PASS=True, frames=10, native_slots=80, no_detection_frames=sum((r['selected_index'] is None for r in proposals)), candidate_count=offsets[-1], finite_non_sentinel_slots=int(arrays['finite_non_sentinel'].sum()), continuous_valid_slots=int(arrays['continuous_valid'].sum()), nearest_valid_slots=int(arrays['nearest_valid'].sum()), finite_OOB_slots=int((arrays['finite_non_sentinel'] & ~arrays['continuous_valid']).sum()), all10_and80_retained=True, all_candidates_raw9_and_firstargmax_unchanged=True, centroid_excluded_from_native_q_and_L=True, warmup_state_SHA=warm, after_state_unchanged=True, checkpoint_unchanged=True, repeat_first_serial_and_P3_exact=True, no_saved_fullP3_duplicate=True, explicit_predict_calls=12, total_forward_hooks=hooks[0], generated_warmup_hooks=warmhooks, input_seal=p['inputs'], GT_pose_K_reference_semantic_mask_or_historic_score_reads=0, visibility_classifier_fit_private_V112_PnP_q_L_correction_calls=0, association_not_inferred=True, seconds=time.monotonic() - start)
        write(OUT / 'RESULT.json', result)
        outputs = [bind(OUT / name) for name in ('FROZEN_PROPOSALS_ALL10_INPUT_ONLY.jsonl', 'ORIGINAL_NATIVE_QUERY80_INPUT_ONLY.jsonl', 'ORIGINAL_NATIVE_QUERY80_AND_VALID9X9_INPUT_ONLY.npz', 'ALL_CANDIDATES_RAW9_AND_BOXES_INPUT_ONLY.npz', 'RESULT.json')]
        write(OUT / 'PREDICTION_SEAL.json', dict(completed=True, PASS=True, code=bind(__file__), policy=e['policy'], protocol=bind(PROTOCOL), entry=bind(entry), inputs=p['inputs'], canonical_transport_seal=e['canonical_transport_seal'], outputs=outputs, frames=10, rows=80, GT_read=False, historical_predictions_read=False, all10_frames80_native_slots_preserved=True, actual_original_initializer=True, source_emulation=False))
        print(json.dumps(dict(completed=True, PASS=True, seal=bind(OUT / 'PREDICTION_SEAL.json'), result=bind(OUT / 'RESULT.json'))), flush=True)
    except BaseException:
        write(OUT / 'FAILED_EXECUTION.json', dict(completed=False, traceback=traceback.format_exc(), automatic_retry=False))
        raise
