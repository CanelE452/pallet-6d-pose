"""Materialize original YCB TRAIN metadata under a prospectively frozen split."""
import collections
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).absolute().parent
PROJECT = ROOT.parent.parent


def safe(path):
    path = Path(path)
    if 'v112' in str(path).lower() or '__pycache__' in path.parts:
        raise ValueError('Forbidden path')
    return path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    safe(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')


def main():
    scope_path = ROOT / 'DATA_MANIFEST_SCOPE_BEFORE_SOURCE_ROWS.json'
    scope_data = safe(scope_path).read_bytes()
    scope = json.loads(scope_data)
    data = []
    for binding in scope['read_exact']:
        raw = safe(binding['path']).read_bytes()
        if digest(raw) != binding['sha256']:
            raise ValueError('Bound source changed')
        data.append(raw)
    inputs = [json.loads(line) for line in data[0].splitlines()]
    assert len(inputs) == 1000
    rows = [json.loads(line) for line in data[1].splitlines()]
    assert len(rows) == 8000
    fit = set(scope['prospective_group_split']['fit_scene_ids'])
    dev = set(scope['prospective_group_split']['dev_scene_ids'])
    assert not fit & dev
    rgb_root = PROJECT / 'query_visibility_public_real_20261022_v63/ycb/data/train_real'
    images = {}
    frame_to_image = {}
    frame_roles = {}
    input_rows = []
    target_rows = []
    counts = collections.Counter()
    for i, inp in enumerate(inputs):
        assert inp['input_id'] == f'Y109_SOURCE_{i:06d}'
        assert inp['instance_row_index'] == i
        assert inp['split'] == 'TRAIN' and inp['native_mapping'] == 'UNKNOWN'
        image = inp['image']
        image_path = safe(image['path'])
        assert image_path.is_absolute() and '..' not in image_path.parts
        image_path.relative_to(rgb_root)  # Lexical only; no image filesystem access.
        if str(image_path) in images:
            assert images[str(image_path)] == image['sha256']
        images[str(image_path)] = image['sha256']
        group = rows[i * 8:(i + 1) * 8]
        scenes = {f"{int(row['scene_id']):06d}" for row in group}
        assert len(scenes) == 1
        scene = scenes.pop()
        assert scene in fit | dev
        role = 'INTERNAL_SOURCE_FIT' if scene in fit else 'INTERNAL_SOURCE_DEV'
        frame = (scene, str(group[0]['source_frame_id']))
        if frame in frame_to_image:
            assert frame_to_image[frame] == (str(image_path), image['sha256'])
        frame_to_image[frame] = (str(image_path), image['sha256'])
        if frame in frame_roles:
            assert frame_roles[frame] == role
        frame_roles[frame] = role
        input_rows.append(dict(inp, internal_source_role=role))
        for k, row in enumerate(group):
            assert row['slot_index'] == i * 8 + k
            assert row['input_id'] == inp['input_id']
            assert row['instance_row_index'] == i and row['logical_aux_slot'] == k
            assert row['q'] == inp['initial_points'][k]
            assert row['L_input'] == inp['roi_side']
            assert (f"{int(row['scene_id']):06d}", str(row['source_frame_id'])) == frame
            assert row['native_mapping'] == 'UNKNOWN'
            state = row['original_teacher_immutable']['raw_surface_teacher_state']
            assert state in ('D', 'E', 'S', 'U')
            target = {'D': 0, 'E': 1, 'S': -1, 'U': -1}[state]
            target_rows.append({
                'slot_index': row['slot_index'], 'input_id': inp['input_id'],
                'instance_row_index': i, 'logical_aux_slot': k,
                'original_split': 'TRAIN', 'internal_source_role': role,
                'scene_id': scene, 'source_frame_id': row['source_frame_id'],
                'source_instance_id': row['source_instance_id'],
                'object_id': row['object_id'], 'frame_index': row['frame_index'],
                'original_associated_r_state': state, 'external_target': target,
                'native_mapping': 'UNKNOWN',
                'authority': 'Original associated-r surface teacher; q-owner unused',
            })
            counts[(role, state)] += 1
    assert len(images) == len(frame_to_image) == 700
    assert sum(counts.values()) == 8000
    assert {s for s, _ in frame_to_image} == fit | dev
    out = safe(ROOT / 'ycb_manifest_v1')
    out.mkdir(exist_ok=False)
    for name, records in [('INPUTS1000.jsonl', input_rows),
                          ('TARGETS8000_TEACHER_ONLY.jsonl', target_rows)]:
        with safe(out / name).open('x') as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    write_json(out / 'RGB700_WHITELIST.json', {
        'bindings': [{'path': path, 'sha256': value}
                     for path, value in sorted(images.items())],
        'payload_stat_hash_read_decode_this_execution': 0,
    })
    summary = {
        'completed': True, 'PASS': True, 'scope_sha256': digest(scope_data),
        'source_files_read': 2, 'instances': 1000, 'slots': 8000, 'RGB_bindings': 700,
        'original_split': 'TRAIN', 'native_mapping': 'UNKNOWN',
        'split_selected_before_source_read': True,
        'split_changed_based_on_labels': False,
        'independent_component_claim': False, 'fresh_blind_claim': False,
        'counts': {role: {state: counts[(role, state)] for state in ('D', 'E', 'S', 'U')}
                   for role in ('INTERNAL_SOURCE_FIT', 'INTERNAL_SOURCE_DEV')},
        'frames_by_role': dict(collections.Counter(frame_roles.values())),
        'GT_reference_and_q_observation_not_forward_channels': True,
        'RGB_weight_model_forward_fit_PnP': 0,
        'outputs': [],
    }
    for name in ['INPUTS1000.jsonl', 'TARGETS8000_TEACHER_ONLY.jsonl',
                 'RGB700_WHITELIST.json']:
        raw = safe(out / name).read_bytes()
        summary['outputs'].append({'relative_path': name, 'sha256': digest(raw),
                                   'size_bytes': len(raw)})
    write_json(out / 'COMPLETE.json', summary)
    print(json.dumps({key: summary[key] for key in
                      ['completed', 'PASS', 'instances', 'slots', 'RGB_bindings',
                       'frames_by_role', 'RGB_weight_model_forward_fit_PnP']}))


if __name__ == '__main__':
    main()
