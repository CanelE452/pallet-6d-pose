"""Independent extractor review: code and generated arrays/metadata only."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import ast
import copy
import csv
import hashlib
import importlib.util
import json
import math
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'extract_source_masks.py'
OUT = ROOT / 'source_audit/INDEPENDENT_EXTRACTOR_REVIEW.json'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))

def require(condition, message):
    if not condition:
        raise AssertionError(message)

def main():
    require(not OUT.exists(), 'Preserve previous independent review')
    original_hash = sha(CODE)
    require(original_hash == '185ccea1e43ff8cf43e57694c8e382d2dd8911538286a1a8f44218fdefdb81aa', 'Review final frozen candidate only')
    spec = importlib.util.spec_from_file_location('independent_v68_extractor', CODE)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    built_in_path = ROOT / 'source_audit/INDEPENDENT_EXTRACTOR_GENERATED_MOCK.json'
    m.mock(built_in_path)
    rng = np.random.default_rng(6801)
    point_cases = 0
    for h, w in ((1, 1), (3, 5), (480, 640), (480, 720), (540, 960), (560, 560)):
        logits = rng.normal(size=(h, w)).astype(np.float32)
        queries = [(0., 0.), (.5, .5), (w - 1., h - 1.), (w - .51, h - .51),
                   (w - .49, h - .49), (-.1, .1), (float(w), 0.)]
        queries += [tuple(q) for q in rng.uniform((-1, -1), (w + 1, h + 1), size=(15, 2))]
        for q in queries:
            px, py = int(math.floor(q[0] + .5)), int(math.floor(q[1] + .5))
            valid = 0 <= q[0] < w and 0 <= q[1] < h and 0 <= px < w and 0 <= py < h
            result = m.sample_query(logits, q, valid, [px, py])
            require(result['native_valid'] == valid, 'Independent geometry')
            if valid:
                values = [float(logits[y, x]) for y in range(h) for x in range(w)
                          if abs(x - px) <= 2 and abs(y - py) <= 2]
                require(result['exact_logit'] == float(logits[py, px]), 'Exact native lookup')
                require(result['radius2_logit'] == max(values), 'Independent clipped neighborhood')
                require(result['radius2_window_pixels'] == len(values), 'Window pixel count')
                require(result['exact_visible'] == int(logits[py, px] > 0), 'Strict sign')
            else:
                require(result['status'] == 'U' and result['exact_visible'] == -1, 'Invalid cannot become negative')
                require(math.isnan(result['exact_logit']) and math.isnan(result['radius2_logit']), 'Invalid NaN')
            point_cases += 1
    zeros = np.zeros((5, 5), np.float32)
    for value, expected in ((0., 0), (-0., 0), (np.nextafter(np.float32(0), np.float32(1)), 1),
                            (np.nextafter(np.float32(0), np.float32(-1)), 0)):
        zeros[:] = value
        require(m.sample_query(zeros, [2., 2.], True, [2, 2])['exact_visible'] == expected, 'Zero/subnormal sign')
    mask = np.stack([np.full((3, 5), value, np.float32) for value in (-1, 0, 1)])
    for quality, expected in (([.8, .8, .8], 0), ([.1, .9, .9], 1), ([.1, .2, .9], 2)):
        selected, _, index = m.select_mask(mask, np.asarray(quality, np.float32), (3, 5))
        require(index == expected and np.array_equal(selected, mask[expected]), 'Quality max/tie-first')

    # Full generated 1091-target/8590-query identities, with no real image paths.
    counts = {'SYNTH': (235, [4] + [5] * 4 + [6] * 49 + [7] * 24 + [8] * 157),
              'ORIGINAL_YCB': (400, [8] * 400), 'CONTROLLED_YCB': (400, [8] * 400),
              'PALLOX': (56, [8] * 56)}
    targets, mapping = [], []
    for source, (num_targets, query_counts) in counts.items():
        require(len(query_counts) == num_targets, 'Generated cohort recipe')
        for local, nq in enumerate(query_counts):
            ti = len(targets); gid = 'MOCK:' + source + ':' + str(local)
            hw = [540, 960] if source == 'SYNTH' and local % 4 == 0 else [480, 720] if source == 'SYNTH' and local % 4 == 1 else [560, 560] if source == 'SYNTH' and local % 4 == 2 else [480, 640]
            image = '/GENERATED_NOT_A_REAL_RGB/' + str(ti) + '.png'; digest = f'{ti:064x}'
            target = dict(id=gid, global_id=gid, joined_target_index=ti, source=source,
                source_metadata_label=source, group_id=gid, split='VAL', image=image,
                image_sha256=digest, raw_hw=hw, box=[0., 0., float(hw[1] - 1), float(hw[0] - 1)], side=20., queries=[])
            for k in range(nq):
                qi = len(mapping); q = [20. + k, 20. + k]
                if source == 'PALLOX' and local == 0 and k < 6:
                    q = [hw[1] - .1, 20.]
                valid, pixel, _ = m.native_pixel(q, hw)
                sid = gid + ':' + str(k)
                target['queries'].append(dict(sample_id=sid, joined_query_index=qi, native_corner=k,
                    query=q, pixel=list(pixel), native_input_valid=valid))
                mapping.append(dict(joined_query_index=qi, joined_target_index=ti, sample_id=sid, global_id=gid,
                    source=source, group_id=gid, native_corner=k, query_x=q[0], query_y=q[1], pixel_x=pixel[0],
                    pixel_y=pixel[1], native_input_valid=valid, side=20., image=image, image_sha256=digest))
            targets.append(target)
    require((len(targets), len(mapping)) == (1091, 8590), 'Full generated IDs')
    protocol_counts = {source: dict(queries=sum(qc), targets=nt) for source, (nt, qc) in counts.items()}
    with tempfile.TemporaryDirectory(prefix='generated_extractor_review_', dir=ROOT / 'source_audit') as td:
        td = Path(td); manifest = td / 'GENERATED_INPUTS.jsonl'; qmap = td / 'GENERATED_MAP.csv'
        def save_generated(items, rows):
            manifest.write_text(''.join(json.dumps(r) + '\n' for r in items))
            with qmap.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=m.MAP_KEYS); writer.writeheader(); writer.writerows(rows)
            return dict(inputs=dict(manifest=ref(manifest), query_id_map=ref(qmap)))
        identity_lock = save_generated(targets, mapping)
        got, gotmap = m.load_source_identity(identity_lock, dict(fixed_inputs=dict(source_counts=protocol_counts)), set())
        require(len(got) == 1091 and len(gotmap) == 8590, 'Generated identity loader')
        identity_rejections = []
        for name, mutate in (
            ('hidden_target_GT', lambda ts: ts[0].update(exact_GT=0)),
            ('hidden_query_GT', lambda ts: ts[0]['queries'][0].update(label=0)),
            ('changed_query_coordinate', lambda ts: ts[0]['queries'][0]['query'].__setitem__(0, 22.)),
            ('invalid_as_negative', lambda ts: ts[-56]['queries'][0].update(native_input_valid=True)),
        ):
            changed = copy.deepcopy(targets); mutate(changed); fake = save_generated(changed, mapping)
            try:
                m.load_source_identity(fake, dict(fixed_inputs=dict(source_counts=protocol_counts)), set())
            except ValueError:
                identity_rejections.append(name)
            else:
                raise AssertionError(name)
        save_generated(targets, mapping)
        try:
            m.load_source_identity(identity_lock, dict(fixed_inputs=dict(source_counts=protocol_counts)), {targets[0]['image_sha256']})
        except ValueError:
            identity_rejections.append('denylisted_identity')
        else:
            raise AssertionError('Denylisted RGB identity')

        # ENTRYLOCK guard fixtures are virtual metadata only. In particular,
        # no runtime wheel, actual checkpoint, real manifest or image is hashed.
        template = json.loads((ROOT / 'ENTRYLOCK_TEMPLATE.json').read_text())
        template['approved'] = True
        template['output_directory'] = str(td / 'UNCREATED_OUTPUT')
        template['inputs']['query_id_map']['sha256'] = 'MOCK_MAP_SHA'
        fake_protocol = dict(model=dict(code_revision='2b90b9f5ceec907a1c18123530e92e794ad901a4', torch_compile=False, training=False),
                             provenance=dict(no_current_access=True, no_current_inference_authorization=True))
        fake_runtime = dict(READY=True, repo_revision=fake_protocol['model']['code_revision'], bindings=[], wheels=[],
                            pinned_versions={'torch': '2.5.1+cu121'}, checkpoint=dict(sha256=m.CHECKPOINT_SHA))
        fake_seal = dict(completed=True, outputs=[copy.deepcopy(template['inputs'][k]) for k in ('manifest', 'query_id_map')])
        original_registry = {str(Path(b['path']).resolve()): b['sha256'] for b in template['inputs'].values()}
        original_registry[str(CODE.resolve())] = original_hash
        entry = td / 'VIRTUAL_ENTRYLOCK.json'
        def run_guard(lock, protocol, runtime, seal, *, package='2.5.1+cu121', denied_count=223, checkpoint_bytes=323606802):
            reads = []
            documents = {str(entry.resolve()): lock,
                str((ROOT / 'PROTOCOL.json').resolve()): protocol,
                str((m.RUNTIME / 'RUNTIME_SEAL.json').resolve()): runtime,
                str(Path(lock['inputs']['denylist']['path']).resolve()): {'image_sha256': ['MOCK_DENY_' + str(i) for i in range(denied_count)]},
                str((ROOT / 'source_inputs/SEAL.json').resolve()): seal}
            def fake_json(path):
                key = str(Path(path).resolve()); reads.append(key)
                require(key in documents, 'Attempt to read a source/input file in guard')
                return copy.deepcopy(documents[key])
            def fake_bound(binding, expected_path=None, expected_sha=None):
                path = Path(binding['path']).resolve()
                if expected_path is not None:
                    m.require(path == Path(expected_path).resolve(), 'Wrong generated path')
                if expected_sha is not None:
                    m.require(binding['sha256'] == expected_sha, 'Wrong generated expected SHA')
                m.require(original_registry.get(str(path)) == binding['sha256'], 'Changed generated bytes')
                if path == m.RUNTIME / 'checkpoints/sam2.1_hiera_base_plus.pt':
                    return SimpleNamespace(stat=lambda: SimpleNamespace(st_size=checkpoint_bytes))
                return path
            with patch.object(m, 'read_json', fake_json), patch.object(m, 'bound', fake_bound), \
                 patch.object(m.sys, 'prefix', str(m.RUNTIME / 'venv')), \
                 patch.object(m.sys, 'executable', str(m.RUNTIME / 'venv/bin/python')), \
                 patch.object(m.importlib.metadata, 'version', lambda name: package):
                m.validate_entry_lock(entry, td / 'UNCREATED_OUTPUT')
            require(not any(x.endswith(('.jsonl', '.csv', '.pt', '.npz', '.png')) for x in reads), 'Guard touched source/weights')
            return reads
        run_guard(template, fake_protocol, fake_runtime, fake_seal)
        mutations = []
        for field, value in (('approved', False), ('source_only', False), ('current223_allowed', True), ('model_training_allowed', True), ('seed', 0)):
            mutations.append((field, lambda state, field=field, value=value: state[0].update({field: value})))
        for role in ('source_protocol', 'runtime_seal', 'input_seal', 'manifest', 'checkpoint', 'denylist', 'query_id_map'):
            mutations.append(('SHA_' + role, lambda state, role=role: state[0]['inputs'][role].update(sha256='CORRUPT')))
        mutations += [
            ('code_SHA', lambda s: s[0]['extractor'].update(sha256='CORRUPT')),
            ('wrong_target_count', lambda s: s[0]['expected'].update(targets=1090)),
            ('wrong_query_count', lambda s: s[0]['expected'].update(queries=8589)),
            ('wrong_runtime_recipe', lambda s: s[0]['runtime'].update(autocast='float32')),
            ('wrong_output', lambda s: s[0].update(output_directory=str(td / 'DIFFERENT'))),
            ('extra_GT_role', lambda s: s[0]['inputs'].update(GT={'path': '/FAKE_GT', 'sha256': 'x'})),
            ('runtime_not_READY', lambda s: s[2].update(READY=False)),
            ('wrong_repo_revision', lambda s: s[2].update(repo_revision='CORRUPT')),
            ('protocol_compile', lambda s: s[1]['model'].update(torch_compile=True)),
            ('protocol_current_access', lambda s: s[1]['provenance'].update(no_current_access=False)),
            ('incomplete_input_seal', lambda s: s[3].update(completed=False)),
            ('unregistered_identity', lambda s: s[3].update(outputs=[])),
            ('runtime_path_escape', lambda s: s[2].update(bindings=[dict(path='/FAKE_OUTSIDE_RUNTIME', sha256='x')])),
        ]
        guard_rejections = []
        for name, mutate in mutations:
            state = copy.deepcopy([template, fake_protocol, fake_runtime, fake_seal]); mutate(state)
            try:
                run_guard(*state)
            except ValueError:
                guard_rejections.append(name)
            else:
                raise AssertionError('Guard accepted ' + name)
        for name, kwargs in (('wrong_package_version', dict(package='WRONG')), ('incomplete_denylist', dict(denied_count=222)), ('wrong_checkpoint_size', dict(checkpoint_bytes=1))):
            try:
                run_guard(template, fake_protocol, fake_runtime, fake_seal, **kwargs)
            except ValueError:
                guard_rejections.append(name)
            else:
                raise AssertionError(name)
    tree = ast.parse(CODE.read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    ext = functions['extract'].body
    require(ext[0].value.func.id == 'validate_entry_lock' and ext[1].value.func.id == 'load_source_identity', 'Guard ordering changed')
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    require(not any('torch' in ast.dump(n) or 'sam2' in ast.dump(n) or 'PIL' in ast.dump(n) for n in imports), 'Top-level model/image import')
    calls = [n for n in ast.walk(functions['extract']) if isinstance(n, ast.Call)]
    pred = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == 'predict']
    require(len(pred) == 1, 'Single prediction recipe')
    keyword = {x.arg: ast.dump(x.value) for x in pred[0].keywords}
    require(keyword['point_coords'] == 'Constant(value=None)' and keyword['point_labels'] == 'Constant(value=None)', 'Query point prompt forbidden')
    require(keyword['multimask_output'] == 'Constant(value=True)' and keyword['return_logits'] == 'Constant(value=True)', 'Mask recipe changed')
    require(sha(CODE) == original_hash, 'Reviewed code changed during test')
    report = dict(schema='V68_INDEPENDENT_EXTRACTOR_CODE_GENERATED_MOCK_REVIEW_V1', PASS=True, blockers=[],
        code=ref(CODE), protocol=ref(ROOT / 'PROTOCOL.json'), template=ref(ROOT / 'ENTRYLOCK_TEMPLATE.json'),
        built_in_mock=ref(built_in_path), review_code=ref(__file__),
        independent_native_scalar_cases=point_cases, generic_native_HW_cases=6,
        strict_zero_signed_zero_and_float32_subnormal_PASS=True, max_quality_exact_tie_first_PASS=True,
        full_generated_identity_cohort=dict(targets=1091, queries=8590, native_valid=8584, native_invalid=6),
        generated_identity_rejections=identity_rejections, generated_ENTRYLOCK_rejections=guard_rejections,
        guard_order_before_model_image_and_source_reads_PASS=True, explicit_bbox_only_predict_AST_PASS=True,
        whitelisted_identity_fields_only_PASS=True, canonical_query_index_order_PASS=True,
        output_interpretation='Strict exact-logit>0 binary visibility. Zero is valid nonvisible; native-invalid is NaN/-1/U. Radius2 is separately named support. No E/O/self mechanism prediction.',
        caveats=['Runtime/model/hash checks were code-reviewed and tested with virtual metadata. Actual runtime weights were not read by this independent review.',
                 'No actual source forward was performed. This review does not authorize current inference or substitute for a root-approved immutable ENTRYLOCK.'],
        actual_source_RGB_reads=0, actual_source_manifest_or_map_reads=0, actual_weights_reads=0,
        actual_current_reads=0, actual_feature_or_GT_reads=0, model_forward_calls=0, optimizer_updates=0, GPU_calls=0)
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(PASS=True, report=ref(OUT), native_cases=point_cases, guard_rejections=len(guard_rejections))))

if __name__ == '__main__':
    main()
