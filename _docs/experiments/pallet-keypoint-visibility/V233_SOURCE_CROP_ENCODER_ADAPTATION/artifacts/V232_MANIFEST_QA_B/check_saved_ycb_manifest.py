"""Independent saved JSON/JSONL check only. RGB/GT/NPZ/model paths stay inert."""
import ast
import collections
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat

H = Path(__file__).parent
R = H.parent
P = R.parent
ROOT = R / 'V232_ROOT_SOURCE_ADAPTATION'
OUT = ROOT / 'ycb_manifest_v1'
ORIGINAL_INPUT = P / 'query_visibility_ycb_real_surface_source_20261023_v109/query_preparation/INPUTS1000_QUERY_EMULATION_ONLY.jsonl'
ORIGINAL_OBSERVATION = R / 'V228_ROOT_QUERY_OBSERVATION/query_observations_v1/QUERY_OBSERVATIONS8000.jsonl'
RGB_ROOT = PurePosixPath(str(P / 'query_visibility_public_real_20261022_v63/ycb/data/train_real'))
ADMIN_NAMES = ['DATA_MANIFEST_SCOPE_BEFORE_SOURCE_ROWS.json', 'DATA_MANIFEST_SCOPE_V2_SHA_CORRECTION.json',
               'materialize_ycb_manifest.py', 'materialize_ycb_manifest_v2.py',
               'ACTUAL_YCB_MANIFEST_ENTRY.json', 'ACTUAL_YCB_MANIFEST_ENTRY_V2.json',
               'MANIFEST_V1_TERMINAL_FAILURE.json']
OUTPUT_NAMES = ['INPUTS1000.jsonl', 'TARGETS8000_TEACHER_ONLY.jsonl', 'RGB700_WHITELIST.json', 'COMPLETE.json']
ALLOWED = {str(ROOT / n) for n in ADMIN_NAMES} | {str(OUT / n) for n in OUTPUT_NAMES} | {str(ORIGINAL_INPUT), str(ORIGINAL_OBSERVATION)}
bindings = {}

def read(path):
    path = Path(path)
    assert str(path) in ALLOWED
    assert 'v112' not in str(path).casefold()
    cursor = Path('/')
    for part in path.parts[1:]:
        cursor /= part
        assert not stat.S_ISLNK(os.lstat(cursor).st_mode)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        raw = stream.read()
    bindings[str(path)] = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    return raw

admin_raw = {n: read(ROOT / n) for n in ADMIN_NAMES}
scope1 = json.loads(admin_raw[ADMIN_NAMES[0]])
scope = json.loads(admin_raw[ADMIN_NAMES[1]])
entry = json.loads(admin_raw['ACTUAL_YCB_MANIFEST_ENTRY_V2.json'])
assert entry['scope_sha256'] == bindings[str(ROOT / ADMIN_NAMES[1])]['sha256']
assert entry['code_sha256'] == bindings[str(ROOT / 'materialize_ycb_manifest_v2.py')]['sha256']
assert entry['model_RGB_weight_fit_PnP_authorized'] is False
code1 = admin_raw['materialize_ycb_manifest.py'].decode()
code2 = admin_raw['materialize_ycb_manifest_v2.py'].decode()
assert code1.replace('DATA_MANIFEST_SCOPE_BEFORE_SOURCE_ROWS.json', 'DATA_MANIFEST_SCOPE_V2_SHA_CORRECTION.json') == code2
same1 = dict(scope1)
same2 = {k: v for k, v in scope.items() if k != 'correction'}
same1['read_exact'] = scope['read_exact']
assert same1 == same2
assert scope['read_exact'][0]['path'] == str(ORIGINAL_INPUT)
assert scope['read_exact'][1]['path'] == str(ORIGINAL_OBSERVATION)
assert scope['read_exact'][1]['sha256'] == 'c20440d6add5969a5d2655aeba5a0abe8009737ade2171049fbf217491c4496f'
for item in scope['read_exact']:
    assert re.fullmatch('[0-9a-f]{64}', item['sha256'])

raw_input = read(ORIGINAL_INPUT)
raw_obs = read(ORIGINAL_OBSERVATION)
assert bindings[str(ORIGINAL_INPUT)]['sha256'] == scope['read_exact'][0]['sha256']
assert bindings[str(ORIGINAL_OBSERVATION)]['sha256'] == scope['read_exact'][1]['sha256']
original_inputs = [json.loads(line) for line in raw_input.splitlines()]
observations = [json.loads(line) for line in raw_obs.splitlines()]
assert len(original_inputs) == 1000 and len(observations) == 8000

complete = json.loads(read(OUT / 'COMPLETE.json'))
assert complete['completed'] is True and complete['PASS'] is True
assert complete['scope_sha256'] == entry['scope_sha256']
assert {b['relative_path'] for b in complete['outputs']} == set(OUTPUT_NAMES[:3])
raw_output = {}
for item in complete['outputs']:
    path = OUT / item['relative_path']
    raw_output[item['relative_path']] = read(path)
    actual = bindings[str(path)]
    assert actual['sha256'] == item['sha256'] and actual['size_bytes'] == item['size_bytes']
inputs = [json.loads(line) for line in raw_output['INPUTS1000.jsonl'].splitlines()]
targets = [json.loads(line) for line in raw_output['TARGETS8000_TEACHER_ONLY.jsonl'].splitlines()]
white = json.loads(raw_output['RGB700_WHITELIST.json'])
assert len(inputs) == 1000 and len(targets) == 8000

fit = set(scope['prospective_group_split']['fit_scene_ids'])
dev = set(scope['prospective_group_split']['dev_scene_ids'])
assert fit == {'000000','000001','000002','000003','000004'}
assert dev == {'000006','000007'} and not fit & dev
scene_roles = {s: 'INTERNAL_SOURCE_FIT' for s in fit}
scene_roles.update({s: 'INTERNAL_SOURCE_DEV' for s in dev})
images = {}
frame_images = {}
frame_roles = {}
frame_instances = collections.defaultdict(set)
counts = collections.Counter()
instance_counts = collections.Counter()
scene_counts = collections.Counter()
input_fields = {'input_id','instance_row_index','image','raw_hw','initial_points','roi_side','query_box_xyxy','split','source','source_emulation','native_mapping'}
target_fields = {'slot_index','input_id','instance_row_index','logical_aux_slot','original_split','internal_source_role','scene_id','source_frame_id','source_instance_id','object_id','frame_index','original_associated_r_state','external_target','native_mapping','authority'}
for i, (original, generated) in enumerate(zip(original_inputs, inputs)):
    assert set(original) == input_fields
    assert set(generated) == input_fields | {'internal_source_role'}
    assert {k: v for k, v in generated.items() if k != 'internal_source_role'} == original
    assert original['input_id'] == f'Y109_SOURCE_{i:06d}' and original['instance_row_index'] == i
    assert original['split'] == 'TRAIN' and original['native_mapping'] == 'UNKNOWN'
    assert original['source_emulation'] is True
    assert len(original['initial_points']) == 8
    image = original['image']
    assert set(image) == {'path','sha256'} and re.fullmatch('[0-9a-f]{64}', image['sha256'])
    path = PurePosixPath(image['path'])  # Lexical only: no RGB FS access.
    assert path.is_absolute() and '..' not in path.parts and 'v112' not in str(path).casefold()
    relative = path.relative_to(RGB_ROOT)
    assert len(relative.parts) == 3 and relative.parts[1] == 'rgb'
    assert re.fullmatch('[0-9]{6}', relative.parts[0]) and re.fullmatch('[0-9]{6}\\.png', relative.parts[2])
    scene = relative.parts[0]
    role = scene_roles[scene]
    assert generated['internal_source_role'] == role
    instance_counts[role] += 1
    scene_counts[(scene, 'instances')] += 1
    frame = (scene, int(path.stem))
    if str(path) in images:
        assert images[str(path)] == image['sha256']
    images[str(path)] = image['sha256']
    if frame in frame_images:
        assert frame_images[frame] == image
        assert frame_roles[frame] == role
    frame_images[frame] = image
    frame_roles[frame] = role
    for k in range(8):
        index = 8*i+k
        old = observations[index]
        target = targets[index]
        assert set(target) == target_fields
        assert old['slot_index'] == target['slot_index'] == index
        assert old['input_id'] == target['input_id'] == original['input_id']
        assert old['instance_row_index'] == target['instance_row_index'] == i
        assert old['logical_aux_slot'] == target['logical_aux_slot'] == k
        assert old['q'] == original['initial_points'][k] and old['L_input'] == original['roi_side']
        assert f"{int(old['scene_id']):06d}" == target['scene_id'] == scene
        assert old['source_frame_id'] == target['source_frame_id'] == frame[1]
        for key in ['source_instance_id','object_id','frame_index']:
            assert old[key] == target[key]
        assert old['native_mapping'] == target['native_mapping'] == 'UNKNOWN'
        assert target['original_split'] == 'TRAIN' and target['internal_source_role'] == role
        state = old['original_teacher_immutable']['raw_surface_teacher_state']
        assert state in {'D','E','S','U'} and target['original_associated_r_state'] == state
        assert target['external_target'] == (int(state == 'E') if state in {'D','E'} else -1)
        assert target['authority'] == 'Original associated-r surface teacher; q-owner unused'
        counts[(role,state)] += 1
        frame_instances[frame].add(i)
        scene_counts[(scene, state)] += 1
assert len(images) == len(frame_images) == 700
assert {b['path']: b['sha256'] for b in white['bindings']} == images
assert len(white['bindings']) == len(images) and white['bindings'] == [{'path': p, 'sha256': images[p]} for p in sorted(images)]
assert white['payload_stat_hash_read_decode_this_execution'] == 0
derived_counts = {role: {state: counts[(role,state)] for state in 'DESU'} for role in ['INTERNAL_SOURCE_FIT','INTERNAL_SOURCE_DEV']}
assert derived_counts == complete['counts']
assert dict(collections.Counter(frame_roles.values())) == complete['frames_by_role']
assert not {frame for frame, role in frame_roles.items() if role=='INTERNAL_SOURCE_FIT'} & {frame for frame, role in frame_roles.items() if role=='INTERNAL_SOURCE_DEV'}
assert complete['instances']==1000 and complete['slots']==8000 and complete['RGB_bindings']==700
assert complete['independent_component_claim'] is False and complete['fresh_blind_claim'] is False
assert complete['split_changed_based_on_labels'] is False
# Producer text uses immutable associated-r state only, not q observation owner.
assert "row['original_teacher_immutable']['raw_surface_teacher_state']" in code2
assert 'q_observation_not_semantic_target' not in code2 and 'q_centre_observation' not in code2
assert code2.index("scope = json.loads(scope_data)") < code2.index("raw = safe(binding['path']).read_bytes()")
assert code2.index("inputs = [json.loads(line)") > code2.index("if digest(raw) != binding['sha256']")

review = dict(completed=True,PASS=True,scope='Independent saved SOURCE JSON/JSONL values and CODE/scope review; no pixel/GT raw/array/model execution or recipe selection.',
    bindings=list(bindings.values()),checks={'exact_original1000_fields_and_q8_L_box_order':True,'all8000_original_associated_r_states_preserved_not_q_owner':True,
    'scene_split_before_source_value_use_code_and_entry_evidence':True,'all_frame_instances_slots_same_role':True,'RGB700_exact_dedup_lexical_prefix_and_SHA_format':True,
    'all_saved_output_bytes_match_COMPLETE':True,'V2_scope_digest_only_repair_and_producer_single_scope_filename_change':True},
    counts=derived_counts,instances_by_role=dict(instance_counts),frames_by_role=dict(collections.Counter(frame_roles.values())),
    scene_counts={scene:{key:scene_counts[(scene,key)] for key in ['instances','D','E','S','U']} for scene in sorted(scene_roles)},
    original_total_states={s:sum(counts[(r,s)] for r in ['INTERNAL_SOURCE_FIT','INTERNAL_SOURCE_DEV']) for s in 'DESU'},
    chronology={'V1_failure_preserved':True,'V1_was_SHA_transcription_failure_not_data_selection_change':True,'original_BAD_READINESS_untouched':True,'actual_root_execution_received':'902b01 exit0; not independently rerun'},
    authority={'source_original_role':'TRAIN','internal_role_only_added':True,'native_mapping':'UNKNOWN','input_query_origin':'GT-derived associated surface r + <=4px SOURCE jitter emulation; not initializer performance',
               'target':'associated-r surface D/E/S/U, D0/E1/SU-1, unchanged','q_ownership_is_not_target':True,'SOURCE_internal_DEV_preexposed':True,'training_or_model_selection_admission':False},
    limits=['RGB path/SHA bindings validated as text only; no RGB file stat/hash/decode or actual image symlink check.',
            'No original raw GT/reference/teacher NPZ read; original immutable teacher fields were read only from specifically authorized saved observation JSONL.',
            'All8 slots and all instances of every existing frame share role; future augmentation descendants must inherit role, but no future descendants exist to test here.',
            'Scene separation is not physical component/object independence proof; all1000 original TRAIN instances were historically exposed.',
            'Pixel-derived availability is not newly computed/certified by metadata-only manifest.'],
    access={'original_SOURCE_JSONL_files':2,'original_SOURCE_instances':1000,'saved_SOURCE_observation_rows':8000,'RGB_weights_original_GT_NPZ_pixel_protected_FS':0,'V112_or_forbidden_cache_FS':0,'new_fit_forward_PnP':0,'broad_scan_Git_network':0})
raw=(json.dumps(review,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
fd=os.open(H/'REVIEW.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o644)
with os.fdopen(fd,'wb') as stream:stream.write(raw)
print(json.dumps({'completed':True,'PASS':True,'review_path':str(H/'REVIEW.json'),'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw),'counts':derived_counts,'instances_by_role':dict(instance_counts)}))
