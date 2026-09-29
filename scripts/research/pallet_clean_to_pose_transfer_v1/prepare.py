"""Freeze the clean78 matched dataset without opening evaluation coordinates.

Only new-namespace text, directories and symlinks are written. Existing image
and label bytes are reused; YOLO cache paths remain inside this namespace.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from . import common as C

PARENT = C.ROOT / '_docs/experiments/pallet_type_selftrain_v1/selftrain_recovery_v1/pose_only/PROTOCOL.json'
ARMS = {f'CLEAN_{target}_{condition}':dict(target=target, condition=condition)
        for target in ('RAW', 'REF') for condition in ('CLEAR', 'OCC')}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def label_path(image):
    image = Path(image)
    parts = list(image.parts)
    assert parts.count('images') == 1, image
    parts[parts.index('images')] = 'labels'
    return Path(*parts).with_suffix('.txt')


def parse_label(path):
    rows = np.array([line.split() for line in Path(path).read_text().splitlines()], dtype=np.float64)
    assert rows.shape == (1, 32), (path, rows.shape)
    assert np.isfinite(rows).all()
    points = rows[0, 5:].reshape(9, 3)
    assert np.isin(points[:, 2], (0, 1, 2)).all()
    return rows[0, :5].tolist(), points.tolist()


def sample_real_names(names, count=512):
    names = sorted(names)
    assert names and len(set(names)) == len(names)
    return np.random.default_rng(9021).choice(names, count, replace=True).tolist()


def local_link(source, destination):
    """Never overwrite a destination or write through an old symlink."""
    source, destination = Path(source), Path(destination)
    assert source.is_file(), source
    assert destination.is_absolute() and destination.parent.resolve().is_relative_to(C.RAW)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        assert destination.is_symlink() and destination.resolve() == source.resolve(), destination
    else:
        destination.symlink_to(source.resolve())
    return destination


def pair_summary(labels, image_names):
    summaries = {}
    for target in ('RAW', 'REF'):
        rows = [parse_label(labels[target][name]) for name in image_names]
        summaries[target] = dict(
            boxes=digest([r[0] for r in rows]),
            support=digest([[p[2] for p in r[1]] for r in rows]),
            coordinates=digest([[p[:2] for p in r[1]] for r in rows]),
            supervised=sum(p[2] == 2 for r in rows for p in r[1]),
            ignored=sum(p[2] == 1 for r in rows for p in r[1]),
            invisible=sum(p[2] == 0 for r in rows for p in r[1]))
    assert summaries['RAW']['boxes'] == summaries['REF']['boxes'], 'RAW/REF boxes differ'
    assert summaries['RAW']['support'] == summaries['REF']['support'], 'RAW/REF export support differs'
    assert summaries['RAW']['coordinates'] != summaries['REF']['coordinates'], 'No coordinate intervention'
    return summaries


def prepare():
    destination = C.DOC / 'PRIMARY_PROTOCOL.json'
    if destination.exists():
        protocol = C.read(destination)
        for binding in protocol['inputs'] + protocol['sources'] + [protocol['initialization']]:
            C.verify(binding)
        print('PRIMARY_PROTOCOL_ALREADY_LOCKED', protocol['train_unique_images'], flush=True)
        return protocol

    lock_path = C.DOC / 'CLEAN_LOCK.json'
    private_path = C.RAW / 'CLEAN_LOCKED_PRIVATE.json'
    lock, private, parent = C.read(lock_path), C.read(private_path), C.read(PARENT)
    assert lock['locked_before_fit_and_new_scoring']
    for binding in lock['inputs']:
        C.verify(binding)
    assert lock['clean_locked_count'] == len(private['rows']) == 78
    assert lock['policy']['no_prediction_based_selection']
    assert lock['policy']['no_new_training_RGB'] and lock['manual_coordinates_added'] == 0
    rows = sorted(private['rows'], key=lambda row:row['train_id'])
    for row in rows:
        assert row['used'] and row['eligible_for_primary'] and row['EXTERNAL_CLEAN'] == 'YES'
        assert row['marker_board'] == 'NOT_OBSERVED'
        C.verify(row['image'])
    assert len({r['train_id'] for r in rows}) == len({r['image']['sha256'] for r in rows}) == 78

    original = {}
    for target in ('RAW', 'REF'):
        binding = parent['datasets'][target]['train_list']; C.verify(binding)
        slots = [Path(p) for p in (C.ROOT / binding['path']).read_text().splitlines()]
        assert len(slots) == 1024
        source = [p for p in slots if p.name.startswith('syn__')]
        real = {p.name:p for p in slots if not p.name.startswith('syn__')}
        assert len(source) == len(set(source)) == 512 and len(real) == 217
        original[target] = dict(slots=slots, source=source, real=real)
    assert [p.name for p in original['RAW']['slots']] == [p.name for p in original['REF']['slots']]
    assert [p.resolve() for p in original['RAW']['source']] == [p.resolve() for p in original['REF']['source']]
    names = [r['train_id']+'.png' for r in rows]
    assert all(name in original['RAW']['real'] and name in original['REF']['real'] for name in names)
    labels = {target:{name:label_path(original[target]['real'][name]) for name in names} for target in ('RAW','REF')}
    export_summary = pair_summary(labels, names)
    real_order = sample_real_names(names)
    slot_summary = pair_summary(labels, real_order)
    sampled_counts = Counter(real_order)
    used_rows = [r for r in rows if r['train_id']+'.png' in sampled_counts]
    # Sampling remains the predeclared replacement rule even if it omits a row.
    # Do not repair coverage after looking at targets or evaluation errors.

    from scripts.research.pallet_oracle_mechanism_followup_v1 import cycle_affine_eval as E
    eval_rows, unused_predictions, unused_poses = E.metadata('PLASTIC')
    del unused_predictions, unused_poses
    assert len(eval_rows) == 128
    safe_eval = [{key:r[key] for key in ('id','image','recording','severity')} for r in eval_rows]
    train_shas = {r['image']['sha256'] for r in rows}
    assert not train_shas & {r['image']['sha256'] for r in safe_eval}, 'Train/eval SHA overlap'
    assert not {r['recording'] for r in rows} & {r['recording'] for r in safe_eval}, 'Train/eval recording overlap'
    assert not {r['train_id'] for r in rows} & {r['id'] for r in safe_eval}, 'Train/eval ID overlap'
    severity = Counter(r['severity'] for r in safe_eval)
    assert severity == dict(CLEAN=29, MODERATE_OCCLUSION=21, SEVERE_OCCLUSION=78), severity

    inputs = []
    for target in ('RAW','REF'):
        for name in names:
            inputs.extend([C.bind(original[target]['real'][name]), C.bind(labels[target][name])])
        for source in original[target]['source']:
            inputs.extend([C.bind(source), C.bind(label_path(source))])

    # Copy the *list contract* but link images/labels into a new validation tree.
    # The framework's final source-only bookkeeping validator can then create
    # its label cache here without changing any historical dataset directory.
    old_val = C.ROOT / parent['datasets']['RAW']['data']['path']
    old_val = old_val.parent / 'val.txt'
    original_val = [Path(p) for p in old_val.read_text().splitlines()]
    old_ref_val = C.ROOT / parent['datasets']['REF']['data']['path']
    assert old_val.read_text() == (old_ref_val.parent/'val.txt').read_text()
    assert original_val and all('PLASTIC__' not in p.name for p in original_val)
    validation = []
    for index, image in enumerate(original_val):
        name = f'{index:04d}_{image.name}'
        target_image = C.RAW/'dataset/validation/images'/name
        target_label = C.RAW/'dataset/validation/labels'/Path(name).with_suffix('.txt')
        local_link(image, target_image); local_link(label_path(image), target_label)
        inputs.extend([C.bind(image), C.bind(label_path(image))]); validation.append(str(target_image))

    datasets = {}
    for target in ('RAW','REF'):
        folder = C.RAW/'dataset'/target
        for image in original[target]['source']:
            local_link(image, folder/'images'/image.name)
            local_link(label_path(image), folder/'labels'/image.with_suffix('.txt').name)
        for name in names:
            local_link(original[target]['real'][name], folder/'images'/name)
            local_link(labels[target][name], folder/'labels'/Path(name).with_suffix('.txt'))
        paths = [folder/'images'/p.name for p in original[target]['source']]
        paths += [folder/'images'/name for name in real_order]
        C.save(folder/'train.txt', '\n'.join(str(p) for p in paths)+'\n', True)
        C.save(folder/'val.txt', '\n'.join(validation)+'\n', True)
        yaml = (f'path: {folder}\ntrain: {folder/"train.txt"}\nval: {folder/"val.txt"}\n'
                'nc: 1\nnames: [pallet]\nkpt_shape: [9, 3]\nflip_idx: [1, 0, 3, 2, 5, 4, 7, 6, 8]\n')
        C.save(folder/'data.yaml', yaml, True)
        datasets[target] = dict(data=C.bind(folder/'data.yaml'), train_list=C.bind(folder/'train.txt'),
            val_list=C.bind(folder/'val.txt'), selected_real=78, sampled_real_unique=len(sampled_counts))
        inputs.extend(datasets[target][key] for key in ('data','train_list','val_list'))

    manifest = dict(clean_lock=C.bind(lock_path), rows=rows, sampled_occurrences_per_epoch=dict(sampled_counts),
        image_order=names, real_slot_order=real_order,
        source_slot_order=[str(p) for p in original['RAW']['source']], evaluation_metadata=safe_eval,
        no_evaluation_reference_coordinates_read=True,
        metadata_helper_note='E.metadata also loads frozen model predictions/poses, discarded without reading values or using them for selection.')
    manifest_path = C.RAW/'PRIMARY_INPUT_BINDINGS_PRIVATE.json'
    C.save(manifest_path, manifest, True)
    preflight = dict(passed=True, scope='Dataset/export preflight only; transformed tensor/gradient/seed parity is a separate pre-fit gate.',
        clean_locked=78, sampled_real_unique=len(sampled_counts), real_slots_per_epoch=512, source_slots_per_epoch=512,
        RAW_REF_name_order_sha256=digest([p.name for p in original['RAW']['source']]+real_order),
        source_resolved_order_sha256=digest([str(p.resolve()) for p in original['RAW']['source']]),
        export_unique=export_summary, real_slots=slot_summary, same_boxes=True, same_export_support=True,
        only_target_coordinates_intentionally_differ=True, old_label_bytes_reused=True,
        source_membership_order_exact=True, new_validation_cache_namespace=True,
        train_eval_exact_ID_overlap=0, train_eval_SHA_overlap=0, train_eval_recording_overlap=0,
        train_recordings=dict(Counter(r['recording'] for r in used_rows)), evaluation_recordings=dict(Counter(r['recording'] for r in safe_eval)),
        no_evaluation_reference_coordinates_read=True, new_manual_coordinates=0, new_training_RGB=0,
        preflight_correction='Initial pre-write assertion expected abbreviated severity names; metadata uses MODERATE_OCCLUSION/SEVERE_OCCLUSION. Assertion corrected, no population/reference/target change, no fits started.',
        private_manifest=C.bind(manifest_path), new_fits=0, optimizer_updates=0)
    C.save(C.DOC/'DATASET_PAIR_PREFLIGHT.json', preflight, True)
    teacher_method = C.ROOT/'_docs/experiments/pallet_material_selftrain_closure_v1/METHOD_LOCK.json'
    teacher = C.read(teacher_method)
    C.verify(parent['initialization']); C.verify(teacher['plastic_teacher_checkpoint'])
    inputs += [C.bind(private_path), C.bind(manifest_path), C.bind(old_val)]
    inputs = list({b['path']:b for b in inputs}.values())
    args = dict(parent['args'], lr0=1e-5)
    protocol = dict(created_utc=C.now(), locked_before_fit=True, arms=ARMS, args=args,
        initialization=parent['initialization'], datasets=datasets, seeds=[42],
        materials=['PLASTIC'], train_unique_images=len(sampled_counts), clean_locked_count=78,
        epochs=5, updates_per_fit=320, primary_fits=4, real_slots_per_epoch=512, source_slots_per_epoch=512,
        real_exposures_per_fit=2560, source_exposures_per_fit=2560, checkpoint_selection='last only',
        sampler=dict(seed=9021, policy='Sort clean78 original image aliases, replacement512; same slots every epoch, loader shuffles.'),
        trainable='Existing pose branches+flow only; backbone/detector/all buffers exact frozen.',
        teacher=teacher['plastic_teacher_checkpoint'], teacher_manual_budget=teacher['teacher_manual_budget'], teacher_updates=0,
        target_contract='Reuse original RAW/REF export label bytes and image padding. Coordinates differ; no manual substitution or hidden PnP addition.',
        support_contract='For real data, same base affine applied to RAW and REF; common post-affine v2 intersection supervises both, others demoted to true-ignore v1. Original ignored v1 remains ignored; source unchanged. Identical across CLEAR/OCC.',
        masking=dict(schedule=.5, area_fractions=[.1,.2,.3], aspects=[.5,1.,2.], max_positions=32,
            min_covered=1, min_remaining=2, canonical='REF', domain='full transformed 640 canvas',
            fill='8x8 random RGB bilinear-resized; existing v2 random_plan principles',
            S2_feasibility=False, center_coverage=False, source_occlusion=False,
            target_v_unchanged_by_occlusion=True, clear_occ_base_RGB_exact=True),
        deviation='Clean subset and common post-affine support are explicit new protocol changes; old S1 native-canvas/S2-paired placement is not reproduced bit-exact.',
        loader_seed_contract='Original generator constant +(training_seed-42), set before InfiniteDataLoader iterator construction; RAW/REF/CLEAR/OCC share seed;42 and43 streams must differ in CPU preflight.',
        evaluation=dict(material='PLASTIC', full=128, clean=29, moderate=21, severe=78, primary=99,
            primary_name='MODERATE_PLUS_SEVERE', primary_metrics=['translation_cm_median','translation_cm_P90','rotation_deg_median','rotation_deg_P90'],
            final_geometry='Same production D9, corner0..7 solve; no model-specific selector at primary stage.',
            independent_test=False, prediction_lock_before_reference_scoring=True),
        parent_protocol=C.bind(PARENT), clean_lock=C.bind(lock_path), dataset_preflight=C.bind(C.DOC/'DATASET_PAIR_PREFLIGHT.json'),
        sources=[C.bind(PARENT), C.bind(lock_path), C.bind(teacher_method), C.bind(Path(__file__))], inputs=inputs,
        implementation_lock='CODE_LOCK.json must be frozen and checked separately before first fit.',
        new_manual_coordinates=0, new_training_RGB=0, no_evaluation_reference_coordinates_read=True)
    C.save(destination, protocol, True)
    print('PRIMARY_PROTOCOL_LOCKED',dict(unique=len(sampled_counts),accepted_clean=78,
        real_slots=512, source_slots=512, validation=len(validation), inputs=len(inputs)),flush=True)
    return protocol


if __name__ == '__main__':
    prepare()
