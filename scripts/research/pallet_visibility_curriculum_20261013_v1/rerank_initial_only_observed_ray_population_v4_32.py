"""Uniform cached V32 initial-only observation operator, no new pixels/poses.

The source-reviewed F32 uniform row adapter replaces the old F31 private rank
alias. Original/measured queries only enter the graph; old pose guides remain
archive metadata. Frozen V31 storage/pool/canonical/concat code is retained.
A separate central32 declaration and explicit GO precede all actual stages.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
from itertools import zip_longest
import json
import multiprocessing
from pathlib import Path
import numpy as np

from . import rerank_observed_bi_edge_population_v4_30 as PARENT
from . import initial_only_observed_ray_association_v4_32 as F
from . import initial_only_observed_ray_observation_v4_32 as OBS32
from . import bind_operational_geometry_guided_eval_v4_29d as OP

C, OLD, D = PARENT.C, PARENT.OLD, PARENT.D
B = D.B
OUTPUT = B / 'initial_only_observed_ray_observations_v4_32'
PLAN = B / 'initial_only_observed_ray_population_plan_v4_32/PREDECLARATION.json'
DECLARATION = B / 'initial_only_observed_ray_method_plan_v4_32/PREDECLARATION.json'
IO_PROOF = B / 'pure_cached_json_io_benchmark_v4_30_checks'
IO_RECEIPT_SHA256 = '1e0f0c4be64361149c91d65299f93f981e45f6fde3760b4cc328eb0617e1c9af'
PARENT_SOURCE_SHA256 = 'd9ec2f97801d587057ba132f611c0bddaef9660a121b92177738dd5e163de421'
ADMISSION_SOURCE_SHA256 = 'd4ab01e8ffb0d3ee00e18730fdd7976ca9debf84e7c432c216df6cc648931674'
ROW_ADAPTER_SHA256 = 'c7cff51b0ed92ec9cf69b37d73252f6d6ec45f4837e1c688e5cf0233661bca72'
STORAGE_PARENT = Path(__file__).with_name('rerank_observed_anchor_native_ray_population_v4_31.py')
STORAGE_PARENT_SHA256 = '486bcbfcd286c1215b7c1fe6a79c56a54bfa232b1e4bef1422fadf0bb909fd16'
SOURCE_CLOSURE = B / 'frozen_e_zero_edge_native_source_audit_v4_31/COARSE_Q_GUIDE_SOURCE_CLOSURE_V1.json'
ANCESTRY = B / 'initial_only_raw_query_source_audit_v4_32/AUDIT_V1.json'
ANCESTRY_SHA256 = 'e0db2eab479fe2527aa394465f7650c67999de385f4be8cae5c2b26e6b908620'
RAW_OBSERVATION_GENERATION = F.P.OBSERVATION_GENERATION
SELECTION_PROFILE = F.SELECTION_PROFILE
PARTITIONS = {'CAL': 46, 'REST': 192}
STORAGE_RECIPE = dict(row_JSON='json.dumps(C.finite(row),ensure_ascii=False,separators=(comma,colon),allow_nan=False)+newline',
    row_write_calls=1, gzip_level=1, encoding='utf-8', final='binary concatenate sealed CAL gzip then REST gzip',
    final_reJSONencode_or_recompression=False, full_C_rows_ID_and_canonical_digest_roundtrip=True)


def _binding(path):
    path = Path(path)
    return dict(path=str(path.resolve().relative_to(C.ROOT)), sha256=C.sha(path))


def _union(*maps):
    result = {}
    for values in maps:
        PARENT._union(result, values)
    return result


def _verify(*maps):
    for mapping in maps:
        D.R._verify(mapping)


def _pins():
    if (C.sha(PARENT.__file__) != PARENT_SOURCE_SHA256
            or C.sha(F.__file__) != ADMISSION_SOURCE_SHA256
            or C.sha(OBS32.__file__) != ROW_ADAPTER_SHA256
            or C.sha(STORAGE_PARENT) != STORAGE_PARENT_SHA256):
        raise ValueError('Exact F32 row/source gates and unchanged V31 storage ancestor required')
    if (C.sha(SOURCE_CLOSURE) != F.INITIAL_SOURCE_CLOSURE_SHA256
            or C.sha(ANCESTRY) != ANCESTRY_SHA256):
        raise ValueError('Pinned independent initial geometry/q and raw-query ancestry required')


def _central(expected_sha256):
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(c not in '0123456789abcdef' for c in expected_sha256)
            or C.sha(DECLARATION) != expected_sha256):
        raise ValueError('Explicit exact central32 SHA required; no default/substitution')
    central = C.read(DECLARATION)
    if (central['selection_profile'] != SELECTION_PROFILE
            or central['raw_observation_generation'] != RAW_OBSERVATION_GENERATION
            or central['partitions'] != dict(CAL=46, TEST=49, REMAINDER=143, PRIMARY=192, severe=0)):
        raise ValueError('Uniform all238 central32 population required')
    for path, expected in ((SOURCE_CLOSURE, F.INITIAL_SOURCE_CLOSURE_SHA256),
                           (ANCESTRY, ANCESTRY_SHA256)):
        if central['input_sha256'].get(str(path.relative_to(C.ROOT))) != expected:
            raise ValueError('Central32 must bind independent initial and raw-query source receipts')
    _verify(*(central[k] for k in ('source_sha256', 'input_sha256', 'protected_user_sha256')))
    return central


def save_rows_large(path, values):
    """Same decompressed JSON values; a single large text write per row."""
    with gzip.open(Path(path), 'xt', encoding='utf-8', compresslevel=1) as stream:
        for row in values:
            stream.write(json.dumps(C.finite(row), ensure_ascii=False,
                separators=(',', ':'), allow_nan=False) + '\n')


def concatenate_gzip_members(destination, members):
    """Copy exact sealed gzip bytes in order; no JSON or gzip encoder used."""
    destination = Path(destination)
    members = list(members)
    if len(members) < 2:
        raise ValueError('At least two explicitly ordered gzip members required')
    for member in members:
        if C.sha(member['path']) != member['sha256']:
            raise ValueError('Sealed gzip member changed before binary copy')
    copies = []
    with destination.open('xb') as output:
        for member in members:
            path = Path(member['path']);digest = hashlib.sha256();size = 0
            with path.open('rb') as source:
                for chunk in iter(lambda: source.read(8*1024*1024), b''):
                    output.write(chunk);digest.update(chunk);size += len(chunk)
            if digest.hexdigest() != member['sha256'] or C.sha(path) != member['sha256']:
                raise ValueError('Source gzip member changed during/after binary copy')
            copies.append(dict(path=str(path), sha256=digest.hexdigest(), bytes=size))
    return copies


def _validate_selection_counts(row):
    observation = row['observation'];refs = observation['selected_chain_refs']
    names = [ref['line_name'] for ref in refs]
    if (len(names) != len(set(names)) or set(names) != set(observation['joint_graph']['selected_line_names'])
            or observation['selected_physical_edges'] != len(refs)
            or row['summary']['selected_physical_edges'] != len(refs)
            or row['summary']['zero_selected_edges'] != (len(refs) == 0)
            or len({tuple(sorted(ref['native_edge'])) for ref in refs}) != len(refs)):
        raise ValueError('Graph, refs, physical-edge count and authoritative summary must agree')


def _correct_selection_metadata(new_row, parent_observation):
    observation = new_row['observation']
    observation['parent_legacy_selected_physical_edges'] = deepcopy(parent_observation['selected_physical_edges'])
    observation['selected_physical_edges'] = len(observation['selected_chain_refs'])
    observation['cached_admission_generation'] = F.OBSERVATION_GENERATION
    observation['old_V30_noise_validated_for_this_selection_population'] = False
    if D.R._canonical_sha(observation.get('partial_line_ledger')) != D.R._canonical_sha(parent_observation.get('partial_line_ledger')):
        raise ValueError('All raw partial alternatives must remain unchanged')
    _validate_selection_counts(new_row)
    return new_row


def rerank_cached_row(row, configuration, mask):
    """Uniform pure F32 adapter; no legacy graph or pose-guide reconstruction."""
    _pins()
    new, summary = OBS32.rerank_cached_row(row, configuration, mask)
    if new['observation']['external_pose_guides_consumed'] != 0:
        raise ValueError('No external pose guide influence allowed')
    return _correct_selection_metadata(new, row['observation']), summary


def validate_journal(path, entries):
    """Streaming ordered ID and every canonical row digest, including zeros."""
    count = 0;seen = set()
    for row, entry in zip_longest(C.rows(path), entries):
        if row is None or entry is None:
            raise ValueError('No missing or extra JSONL rows allowed')
        if (row['id'] != entry['id'] or row['id'] in seen
                or row['partition'] != entry['partition']
                or D.R._canonical_sha(row) != entry['canonical_row_sha256']):
            raise ValueError('Original order, unique IDs and every canonical row digest required')
        _validate_selection_counts(row)
        seen.add(row['id']);count += 1
    return dict(rows=count, canonical_row_digests_verified=count, missing_duplicate_or_extra_rows=0)


def _sources():
    sources = {}
    for module in (PARENT, F, F.PARENT, OBS32, F.F, F.P, F.G, F.P.ORDER, F.P.VIS, F.G.NMS,
            F.B, F.M, D, D.R, OLD, OLD.V, OP, C):
        path = Path(module.__file__).resolve()
        sources[str(path.relative_to(C.ROOT))] = C.sha(path)
    for path in (Path(__file__), Path(__file__).with_name('test_rerank_initial_only_observed_ray_population_v4_32.py'), STORAGE_PARENT):
        sources[str(path.resolve().relative_to(C.ROOT))] = C.sha(path)
    return sources


def declare(*, root_metadata_GO=False, expected_central_plan_sha256=None):
    if root_metadata_GO is not True:
        raise PermissionError('Root metadata GO before declaration IO')
    if PLAN.exists():
        raise FileExistsError('Preserve every operator declaration')
    _pins();central = _central(expected_central_plan_sha256)
    ancestry = C.read(ANCESTRY)
    if (ancestry.get('verdict') != 'NO_CROSS_QUERY_NMS_OR_PREGRAPH_CACHED_GUIDE_QUOTA_FOUND'
            or ancestry.get('actual_rows_or_images_or_arrays_read') != 0):
        raise ValueError('Independent original/measured query budget source audit required')
    _verify(ancestry['source_sha256'])
    if C.sha(IO_PROOF / 'OWNER_RECEIPT_V1.json') != IO_RECEIPT_SHA256:
        raise ValueError('Exact existing pure11MB storage proof required')
    sources = _union(central['source_sha256'], _sources())
    proof = C.read(IO_PROOF / 'OWNER_RECEIPT_V1.json')
    proof_paths = [IO_PROOF / name for name in ('OWNER_RECEIPT_V1.json', 'RESULT.json',
        'benchmark_cached_json_io_v1.py', 'PLAN.json', 'PLAN_SEAL.json', 'RESULT_SEAL.json')]
    for path in proof_paths[1:]:
        if C.sha(path) != proof['source_and_result_copies'][path.name]['sha256']:
            raise ValueError('Existing IO proof source/plan/result copies must remain exact')
    inputs = _union(central['input_sha256'], {str(p.relative_to(C.ROOT)): C.sha(p)
        for p in (DECLARATION, SOURCE_CLOSURE, ANCESTRY, *proof_paths)})
    _verify(sources, inputs, central['protected_user_sha256'])
    PLAN.parent.mkdir(parents=True)
    C.write(PLAN, dict(schema='V32_INITIAL_ONLY_CACHED_ALL238_OPERATOR_PREDECLARATION',
        UTC=datetime.now(timezone.utc).isoformat(), central_plan_sha256=expected_central_plan_sha256,
        source_sha256=sources, input_sha256=inputs, protected_user_sha256=central['protected_user_sha256'],
        storage=STORAGE_RECIPE, selection_profile=SELECTION_PROFILE,
        raw_observation_generation=RAW_OBSERVATION_GENERATION, rows=238, CAL=46, REST=192,
        maximum_workers=3, maximum_pending=3, parent_retains_only_small_entries=True,
        uniform_F32_row_adapter_no_external_pose_guides=True,
        initial_source_closure=_binding(SOURCE_CLOSURE), raw_query_source_audit=_binding(ANCESTRY),
        old_pose_guides_preserved_as_archive_metadata_only=True,
        cached_pose_bank_or_baseline_numeric_inputs_to_current_graph=False, selected_physical_edges_matches_new_refs=True,
        parent_legacy_count_preserved=True, all_raw_partial_alternatives_preserved=True,
        original_full29_operational_ancestry_and_exact_memo_proof_required=True,
        original_S_U_q_covariance_geometry_ray_mask_guide_and_indices_unchanged=True,
        CAL_before_labels_then53_immutable_before_REST=True,
        all_frames_and_zero_selected_rows_retained=True, new_RGB_NMS_NN_labels_or_posefit=0,
        cached_support_rounded_diagnostic_TLS_refits_allowed=True,
        old_noise_transferred=False, actual_CAL_REST_or_fit_in_declare=0))
    for relative, digest in sources.items():
        target = PLAN.parent / 'code_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True);target.write_bytes((C.ROOT / relative).read_bytes())
        if C.sha(target) != digest:
            raise ValueError('Exact source snapshot required')
    C.write(PLAN.parent / 'PLAN_SEAL.json', dict(plan_sha256=C.sha(PLAN), source_sha256=sources))


def prepare(*, root_metadata_GO=False, expected_central_plan_sha256=None, output=OUTPUT):
    if root_metadata_GO is not True:
        raise PermissionError('Root metadata GO before any input IO')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve every preparation')
    _pins();central = _central(expected_central_plan_sha256)
    plan, pseal = C.read(PLAN), C.read(PLAN.parent / 'PLAN_SEAL.json')
    if (C.sha(PLAN) != pseal['plan_sha256'] or plan['central_plan_sha256'] != expected_central_plan_sha256):
        raise ValueError('Exact operator/central declaration required')
    closure = OP.verify_operational_closure()
    parent = C.read(OLD.OUTPUT / 'PROTOCOL.json');fullseal = C.read(OLD.OUTPUT / 'OBSERVATION_SEAL.json')
    ids = parent['ids'];entries = {}
    paths = [PLAN, PLAN.parent / 'PLAN_SEAL.json', DECLARATION,
        OLD.OUTPUT / 'PROTOCOL.json', OLD.OUTPUT / 'PREPARATION_SEAL.json', OLD.OUTPUT / 'OBSERVATION_SEAL.json']
    for part, count in PARTITIONS.items():
        indexpath = OLD.OUTPUT / (part + '_ROW_INDEX.json')
        sealpath = OLD.OUTPUT / (part + '_OBSERVATION_SEAL.json')
        index, seal = C.read(indexpath)['entries'], C.read(sealpath)
        selected_ids = ids[:46] if part == 'CAL' else ids[46:]
        if (len(index) != count or [e['id'] for e in index] != selected_ids
                or seal['rows'] != count or seal['ordered_ids'] != selected_ids
                or seal['row_index_sha256'] != C.sha(indexpath)
                or fullseal['partition_seals'][part] != C.sha(sealpath)):
            raise ValueError('Exact original CAL46 and REST192 ordered indices/seals required')
        entries[part] = index;paths.extend((indexpath, sealpath))
    if (len(ids) != 238 or len(set(ids)) != 238 or ids != closure['ordered_ids']
            or fullseal['rows'] != 238 or fullseal['ordered_ids'] != ids
            or Counter(e['partition'] for e in entries['CAL'] + entries['REST']) != Counter(CAL=46, TEST=49, REMAINDER=143)):
        raise ValueError('No population filtering or severe rows allowed')
    sources = _union(closure['source_sha256'], plan['source_sha256'], central['source_sha256'])
    inputs = _union(closure['input_sha256'], plan['input_sha256'],
        {str(p.relative_to(C.ROOT)): C.sha(p) for p in paths})
    protected = _union(closure['protected_user_sha256'], central['protected_user_sha256'])
    for entry in entries['CAL'] + entries['REST']:
        path = OLD._row_path(OLD.OUTPUT, entry['id'])
        if C.sha(path) != entry['sha256']:
            raise ValueError('Every exact original raw row required')
        PARENT._union(inputs, {str(path.relative_to(C.ROOT)): entry['sha256']})
    _verify(sources, inputs, protected, closure['original_immutable_file_sha256'])
    output.mkdir(parents=True);(output / 'row_spool').mkdir()
    protocol = dict(schema='V32_INITIAL_ONLY_CACHED_CAL46_THEN_REST192_PROTOCOL', ids=ids, CAL_ids=ids[:46], REST_ids=ids[46:],
        source_sha256=sources, input_sha256=inputs, protected_user_sha256=protected,
        original_parent_immutable_file_sha256=closure['original_immutable_file_sha256'],
        operational_closure=closure, central_plan_sha256=expected_central_plan_sha256,
        selection_profile=SELECTION_PROFILE, raw_observation_generation=RAW_OBSERVATION_GENERATION,
        configuration=parent['configuration'], parent_entries=entries, storage=STORAGE_RECIPE,
        maximum_workers=3, maximum_pending=3, same_numeric_worker_for_CAL_and_REST=True,
        selected_count_is_new_refs_with_parent_legacy_preserved=True,
        raw_partial_alternatives_preserved=True, new_RGB_NMS_NN_labels_posefit=0)
    C.write(output / 'PROTOCOL.json', protocol)
    C.write(output / 'PREPARATION_SEAL.json', dict(rows=46, ordered_CAL_ids=ids[:46],
        protocol_sha256=C.sha(output / 'PROTOCOL.json'), before_any_cached_row_mask_decode_or_rank=True))
    for relative, digest in sources.items():
        target = output / 'code_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True);target.write_bytes((C.ROOT / relative).read_bytes())
        if C.sha(target) != digest:
            raise ValueError('Exact operator source snapshot required')
    return _binding(output / 'PREPARATION_SEAL.json')


def _load(output, expected_sha256):
    _pins();_central(expected_sha256)
    protocol, prep = C.read(output / 'PROTOCOL.json'), C.read(output / 'PREPARATION_SEAL.json')
    if (protocol['central_plan_sha256'] != expected_sha256
            or C.sha(output / 'PROTOCOL.json') != prep['protocol_sha256']
            or protocol['selection_profile'] != SELECTION_PROFILE):
        raise ValueError('Same caller central SHA and frozen preparation required on every stage')
    _verify(*(protocol[k] for k in ('source_sha256', 'input_sha256', 'protected_user_sha256',
        'original_parent_immutable_file_sha256')))
    return protocol


def _worker(case, configuration, output):
    row = OLD._read_single(OLD._row_path(OLD.OUTPUT, case['id']), case)
    path = C.ROOT / row['mask_binding']['path']
    if C.sha(path) != row['mask_binding']['sha256']:
        raise ValueError('Exact original cached mask bytes required')
    with np.load(path, allow_pickle=False) as archive:
        mask = archive['pallet_mask_clean'].copy()
    new, summary = rerank_cached_row(row, configuration, mask)
    new['parent_observation_row'] = dict(case)
    destination = OLD._row_path(output, case['id']);save_rows_large(destination, [new])
    entry = OLD._index_row(new, destination)
    del row, new, mask
    roundtrip = OLD._read_single(destination, entry)
    _validate_selection_counts(roundtrip)
    if roundtrip['selection_profile'] != SELECTION_PROFILE:
        raise ValueError('Exact new selection profile roundtrip required')
    return entry, summary


def _cal_immutable(output):
    paths = [output / name for name in ('PROTOCOL.json', 'PREPARATION_SEAL.json', 'CAL_EXECUTION_PROTOCOL.json',
        'CAL_ROW_INDEX.json', 'CAL_OBSERVATIONS.jsonl.gz', 'CAL_SUMMARY.json', 'CAL_OBSERVATION_SEAL.json')]
    for entry in C.read(output / 'CAL_ROW_INDEX.json')['entries']:
        path = OLD._row_path(output, entry['id'])
        if C.sha(path) != entry['sha256']:
            raise ValueError('Original sealed CAL spool bytes changed before REST snapshot')
        paths.append(path)
    if len(paths) != 53:
        raise ValueError('Exactly seven CAL metadata/payload files plus46 spools required')
    return {str(path.relative_to(C.ROOT)): C.sha(path) for path in paths}


def _prepare_rest(output, protocol):
    paths = [output / n for n in ('REST_PROTOCOL.json', 'REST_PREPARATION_SEAL.json')]
    if any(p.exists() for p in paths):
        raise FileExistsError('Preserve every REST metadata attempt')
    seal = C.read(output / 'CAL_OBSERVATION_SEAL.json')
    if (seal['rows'] != 46 or seal['ordered_ids'] != protocol['CAL_ids']
            or seal['protocol_sha256'] != C.sha(output / 'PROTOCOL.json')
            or seal['preparation_seal_sha256'] != C.sha(output / 'PREPARATION_SEAL.json')
            or seal['execution_protocol_sha256'] != C.sha(output / 'CAL_EXECUTION_PROTOCOL.json')
            or seal['summary_sha256'] != C.sha(output / 'CAL_SUMMARY.json')
            or seal['observations_sha256'] != C.sha(output / 'CAL_OBSERVATIONS.jsonl.gz')
            or seal['row_index_sha256'] != C.sha(output / 'CAL_ROW_INDEX.json')):
        raise ValueError('Complete sealed CAL46 required before REST')
    immutable = _cal_immutable(output)
    rest = {**protocol, 'schema': 'V32_INITIAL_ONLY_REST192_APPEND_AFTER_IMMUTABLE_CAL46',
        'original_CAL_file_sha256': immutable,
        'input_sha256': _union(protocol['input_sha256'], immutable)}
    C.write(paths[0], rest)
    C.write(paths[1], dict(rows=192, ordered_ids=protocol['REST_ids'],
        protocol_sha256=C.sha(paths[0]), CAL_observation_seal=_binding(output / 'CAL_OBSERVATION_SEAL.json'),
        before_any_REST_cached_row_mask_decode_or_rank=True))
    return rest


def _load_rest(output, expected_sha256):
    protocol = _load(output, expected_sha256)
    rest, prep = C.read(output / 'REST_PROTOCOL.json'), C.read(output / 'REST_PREPARATION_SEAL.json')
    if (rest['central_plan_sha256'] != expected_sha256 or rest['ids'] != protocol['ids']
            or C.sha(output / 'REST_PROTOCOL.json') != prep['protocol_sha256']):
        raise ValueError('Frozen REST append closure required')
    _verify(*(rest[k] for k in ('source_sha256', 'input_sha256', 'protected_user_sha256', 'original_CAL_file_sha256')))
    return rest


def _bounded_pool(cases, configuration, output, workers, part, worker=_worker):
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'))
    tasks, pending, entries, summaries = iter(cases), {}, {}, {}
    active_id = None
    def fill():
        while len(pending) < workers:
            case = next(tasks, None)
            if case is None:
                return
            if OLD._row_path(output, case['id']).exists():
                raise FileExistsError('Never overwrite any row spool')
            pending[pool.submit(worker, case, configuration, output)] = case['id']
    try:
        fill()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                ident = pending.pop(future);active_id = ident;entry, summary = future.result()
                if entry['id'] != ident or ident in entries:
                    raise ValueError('Exact unique pending ID required')
                entries[ident] = entry;summaries[ident] = summary
                C.write(output / (part + '_LIVE_PROGRESS.json'), dict(completed=len(entries), total=len(cases), id=ident))
                print(dict(partition=part, completed=len(entries), total=len(cases), id=ident), flush=True)
                active_id = None
            fill()
    except BaseException as error:
        pending_ids = list(pending.values())
        for future in pending:
            future.cancel()
        for process in list((getattr(pool, '_processes', {}) or {}).values()):
            if process.is_alive():
                process.terminate()
        pool.shutdown(wait=False, cancel_futures=True)
        C.write(output / (part + '_FAILURE_RECEIPT.json'), dict(exception=type(error).__name__, reason=str(error),
            completed_ids=list(entries), pending_ids=pending_ids, failed_id=active_id))
        raise
    else:
        pool.shutdown(wait=True)
    return entries, summaries


def _emit_partition(output, protocol, part, entries, summaries):
    ids = protocol[part + '_ids']
    if len(ids) != PARTITIONS[part] or set(entries) != set(ids) or set(summaries) != set(ids):
        raise ValueError('All partition rows, including zero selections, required')
    ordered = [entries[i] for i in ids]
    C.write(output / (part + '_ROW_INDEX.json'), dict(entries=ordered))
    journal = output / (part + '_OBSERVATIONS.jsonl.gz')
    save_rows_large(journal, (OLD._read_single(OLD._row_path(output, ident), entries[ident]) for ident in ids))
    validation = validate_journal(journal, ordered)
    C.write(output / (part + '_SUMMARY.json'), dict(rows=len(ids), records=[summaries[i] for i in ids],
        zero_selected=sum(summaries[i]['zero_selected'] for i in ids), all_rows_kept=True, selection_profile=SELECTION_PROFILE))
    _verify(*(protocol[k] for k in ('source_sha256', 'input_sha256', 'protected_user_sha256')))
    if part == 'REST':
        _verify(protocol['original_CAL_file_sha256'])
    proto = 'PROTOCOL.json' if part == 'CAL' else 'REST_PROTOCOL.json'
    prep = 'PREPARATION_SEAL.json' if part == 'CAL' else 'REST_PREPARATION_SEAL.json'
    C.write(output / (part + '_OBSERVATION_SEAL.json'), dict(rows=len(ids), ordered_ids=ids,
        observations_sha256=C.sha(journal), row_index_sha256=C.sha(output / (part + '_ROW_INDEX.json')),
        summary_sha256=C.sha(output / (part + '_SUMMARY.json')), protocol_sha256=C.sha(output / proto),
        preparation_seal_sha256=C.sha(output / prep), execution_protocol_sha256=C.sha(output / (part + '_EXECUTION_PROTOCOL.json')),
        selection_profile=SELECTION_PROFILE, raw_observation_generation=RAW_OBSERVATION_GENERATION,
        source_sha256=protocol['source_sha256'], canonical_validation=validation))


def rerank(part, *, root_actual_rerank_GO=False, expected_central_plan_sha256=None, workers=3, output=OUTPUT):
    if root_actual_rerank_GO is not True:
        raise PermissionError('Separate root partition cached-rerank GO before IO')
    if part not in PARTITIONS or isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 3:
        raise ValueError('CAL/REST only with bounded1..3 workers')
    output = Path(output);protocol = _load(output, expected_central_plan_sha256)
    if part == 'REST':
        _prepare_rest(output, protocol);protocol = _load_rest(output, expected_central_plan_sha256)
    execution = output / (part + '_EXECUTION_PROTOCOL.json')
    if execution.exists():
        raise FileExistsError('Preserve every actual partition attempt')
    C.write(execution, dict(workers=workers, maximum_pending=workers, UTC=datetime.now(timezone.utc).isoformat(),
        central_plan_sha256=expected_central_plan_sha256, source_sha256=C.sha(__file__),
        proper_module_main_required=True, raw_cached_rank_only=True))
    entries, summaries = _bounded_pool(protocol['parent_entries'][part], protocol['configuration'], output, workers, part)
    _emit_partition(output, protocol, part, entries, summaries)
    return _binding(output / (part + '_OBSERVATION_SEAL.json'))


def finalize(*, root_finalize_GO=False, expected_central_plan_sha256=None, output=OUTPUT):
    if root_finalize_GO is not True:
        raise PermissionError('Root binary all238 finalize GO before IO')
    output = Path(output);protocol = _load_rest(output, expected_central_plan_sha256)
    if any((output / n).exists() for n in ('FINAL_PROTOCOL.json', 'OBSERVATIONS.jsonl.gz', 'OBSERVATION_SEAL.json')):
        raise FileExistsError('Preserve every finalization attempt')
    seals, entries, members = {}, [], []
    for part, count in PARTITIONS.items():
        sealpath = output / (part + '_OBSERVATION_SEAL.json');seal = C.read(sealpath)
        indexpath = output / (part + '_ROW_INDEX.json');index = C.read(indexpath)['entries']
        ids = protocol[part + '_ids'];journal = output / (part + '_OBSERVATIONS.jsonl.gz')
        if (seal['rows'] != count or seal['ordered_ids'] != ids or [e['id'] for e in index] != ids
                or seal['row_index_sha256'] != C.sha(indexpath) or seal['observations_sha256'] != C.sha(journal)
                or seal['selection_profile'] != SELECTION_PROFILE
                or seal['raw_observation_generation'] != RAW_OBSERVATION_GENERATION
                or seal['protocol_sha256'] != C.sha(output / ('PROTOCOL.json' if part == 'CAL' else 'REST_PROTOCOL.json'))
                or seal['preparation_seal_sha256'] != C.sha(output / ('PREPARATION_SEAL.json' if part == 'CAL' else 'REST_PREPARATION_SEAL.json'))
                or seal['execution_protocol_sha256'] != C.sha(output / (part + '_EXECUTION_PROTOCOL.json'))):
            raise ValueError('Both complete sealed partitions required')
        entries.extend(index);members.append(dict(path=journal, sha256=seal['observations_sha256']))
        seals[part] = C.sha(sealpath)
    if [e['id'] for e in entries] != protocol['ids'] or len(entries) != 238:
        raise ValueError('Exact original ordered238 required')
    paths = [output / name for name in ('PROTOCOL.json', 'PREPARATION_SEAL.json', 'REST_PROTOCOL.json',
        'REST_PREPARATION_SEAL.json', 'CAL_EXECUTION_PROTOCOL.json', 'REST_EXECUTION_PROTOCOL.json',
        'CAL_OBSERVATION_SEAL.json', 'REST_OBSERVATION_SEAL.json', 'CAL_ROW_INDEX.json', 'REST_ROW_INDEX.json')]
    inputs = _union(protocol['input_sha256'], {str(p.relative_to(C.ROOT)): C.sha(p) for p in paths})
    final = {**protocol, 'schema': 'V32_INITIAL_ONLY_ALL238_MULTI_MEMBER_FINAL_PROTOCOL', 'input_sha256': inputs,
        'partition_seals': seals, 'final_gzip_members': ['CAL_OBSERVATIONS.jsonl.gz', 'REST_OBSERVATIONS.jsonl.gz'],
        'original_CAL_protocol_unchanged': _binding(output / 'PROTOCOL.json'),
        'selected_physical_edges_is_new_count_parent_legacy_preserved': True}
    _verify(final['source_sha256'], inputs, final['protected_user_sha256'], final['original_CAL_file_sha256'])
    C.write(output / 'FINAL_PROTOCOL.json', final)
    copies = concatenate_gzip_members(output / 'OBSERVATIONS.jsonl.gz', members)
    validation = validate_journal(output / 'OBSERVATIONS.jsonl.gz', entries)
    _verify(final['source_sha256'], inputs, final['protected_user_sha256'], final['original_CAL_file_sha256'])
    C.write(output / 'OBSERVATION_SEAL.json', dict(rows=238, ordered_ids=protocol['ids'],
        observations_sha256=C.sha(output / 'OBSERVATIONS.jsonl.gz'), protocol_sha256=C.sha(output / 'FINAL_PROTOCOL.json'),
        partition_seals=seals, gzip_members=copies, canonical_validation=validation,
        selection_profile=SELECTION_PROFILE, raw_observation_generation=RAW_OBSERVATION_GENERATION,
        source_sha256=final['source_sha256'], all_rows_and_zero_support_cases_retained=True,
        final_JSON_reencoding_or_compression_calls=0))
    C.write(output / 'FINAL_COMPLETION_RECEIPT_V32.json', dict(rows=238, CAL=46, TEST=49, REMAINDER=143,
        final_protocol=_binding(output / 'FINAL_PROTOCOL.json'), observation_seal=_binding(output / 'OBSERVATION_SEAL.json'),
        source_sha256=final['source_sha256'], input_sha256=inputs,
        protected_user_sha256=final['protected_user_sha256'], original_CAL_file_sha256=final['original_CAL_file_sha256'],
        original_parent_immutable_file_sha256=final['original_parent_immutable_file_sha256'],
        canonical_validation=validation, gzip_source_copy_checks=copies, new_RGB_NMS_NN_labels_posefit=0))
    return _binding(output / 'OBSERVATION_SEAL.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('declare', 'prepare', 'CAL', 'REST', 'finalize'))
    parser.add_argument('--central-plan-sha256', required=True)
    parser.add_argument('--root-metadata-GO', action='store_true')
    parser.add_argument('--root-CAL-rerank-GO', action='store_true')
    parser.add_argument('--root-REST-rerank-GO', action='store_true')
    parser.add_argument('--root-finalize-GO', action='store_true')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args();options = dict(expected_central_plan_sha256=args.central_plan_sha256)
    if args.stage == 'declare':
        declare(root_metadata_GO=args.root_metadata_GO, **options)
    elif args.stage == 'prepare':
        print(prepare(root_metadata_GO=args.root_metadata_GO, **options))
    elif args.stage == 'finalize':
        print(finalize(root_finalize_GO=args.root_finalize_GO, **options))
    else:
        go = args.root_CAL_rerank_GO if args.stage == 'CAL' else args.root_REST_rerank_GO
        print(rerank(args.stage, root_actual_rerank_GO=go, workers=args.workers, **options))
