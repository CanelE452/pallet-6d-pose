"""Prepare C's unobserved SOURCE TRAIN rows from exactly three JSON documents."""
from pathlib import Path
import csv, hashlib, json, os, stat

H = Path(__file__).absolute().parent
V = H.parent
BINDINGS = {
    'scope': {'path': str(V/'ROOT_PROSPECTIVE_PUBLIC_TRAIN_NATIVE_WEAK_OBSERVATION_SCOPE.json'), 'sha256': '78b8eb2cd37e0aa4cd7984269c0704940f153ac43830a6ae39daf6798a21fc53'},
    'preview_preparation': {'path': str(V/'preview_preparation_a/FINAL_READY_V2_LAYOUT.json'), 'sha256': 'e6356e497071dfa7eacf3ce66ee33310ae1f734a5e065757c3cdbb1d2e98d6fa'},
    'selection': {'path': str(V/'preview_preparation_a/SELECTED6_BEFORE_SOURCE_PIXELS_V2_LAYOUT.json'), 'sha256': '244d05cb827d77ad1bcfa5a4c5b80bcab30d95aedd04213b97fcc7d56dbbd625'},
}
ENUMS = {
    'part_visibility_D_E_S_UNKNOWN': ['D', 'E', 'S', 'UNKNOWN', 'UNREVIEWED'],
    'query_relation_on_near_off_UNKNOWN': ['ON', 'NEAR', 'OFF', 'UNKNOWN', 'UNREVIEWED'],
    'target_instance_identified': ['YES', 'NO', 'UNKNOWN', 'UNREVIEWED'],
    'intended_part_identified': ['YES', 'NO', 'UNKNOWN', 'UNREVIEWED'],
    'location_error_possible': ['YES', 'NO', 'UNKNOWN', 'UNREVIEWED'],
}

def rb(p):
    p = Path(p)
    assert p.is_absolute() and '..' not in p.parts and V in p.parents
    assert all('v112' not in x.casefold() for x in p.parts)
    assert p.suffix in {'.json', '.py', '.md', '.csv'}
    fd = os.open(p.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for x in p.parts[1:-1]:
            n = os.open(x, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = n
        n = os.open(p.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
    finally:
        os.close(fd)
    with os.fdopen(n, 'rb') as f:
        assert stat.S_ISREG(os.fstat(f.fileno()).st_mode)
        return f.read()

def binding(p):
    return {'path': str(p), 'sha256': hashlib.sha256(rb(p)).hexdigest()}

def put(p, x):
    with p.open('x') as f:
        json.dump(x, f, ensure_ascii=False, indent=2); f.write('\n')

def validate(row):
    for k, values in ENUMS.items():
        assert row[k] in values
    if not row['original_query_available']:
        assert not row['reviewed']
        assert all(row[k] == 'UNKNOWN' for k in ENUMS)
    if row['part_visibility_D_E_S_UNKNOWN'] in {'D', 'E'}:
        assert row['reviewed'] and row['original_query_available']
        assert row['target_instance_identified'] == row['intended_part_identified'] == 'YES'
        assert row['short_visual_evidence'].strip()
    if row['part_visibility_D_E_S_UNKNOWN'] == 'E':
        assert row['external_occluder_evidence'].strip()
    if row['reviewed']:
        assert all(row[k] != 'UNREVIEWED' for k in ENUMS)

def main():
    docs = {}
    for key, b in BINDINGS.items():
        z = rb(b['path']); assert hashlib.sha256(z).hexdigest() == b['sha256']
        docs[key] = json.loads(z)
    assert docs['scope']['actual_observations_require_completed_preview_identity_QA'] is True
    assert docs['preview_preparation']['selection'] == BINDINGS['selection']
    frames = docs['selection']['frames']; assert len(frames) == 6
    rows = []
    for fi, frame in enumerate(frames):
        assert frame['role'] == 'TRAIN' and len(frame['native8']) == 8
        assert [r['native_slot'] for r in frame['native8']] == list(range(8))
        for original in frame['native8']:
            assert original['role'] == 'TRAIN'
            available = original['original_query_available']
            assert type(available) is bool
            row = {k: original[k] for k in ['row_id', 'original_global_index', 'native_slot', 'query_xy', 'L', 'original_query_available']}
            row.update(frame_ordinal=fi, frame_key=frame['frame_key'], frame_id=frame['frame_id'], component_id=frame['component_id'], role='TRAIN', observer='C', reviewed=False,
                       authority='INDEPENDENT_ASSISTANT_SOURCE_TRAIN_WEAK_OBSERVATION',
                       external_occluder_evidence='', short_visual_evidence='', viewed_generated_images=[],
                       unavailable_reason='ORIGINAL_INPUT_U_NO_VISUAL_LABEL' if not available else '',
                       training_admission=False, physical_native_gold=False)
            row.update({k: 'UNREVIEWED' if available else 'UNKNOWN' for k in ENUMS})
            validate(row); rows.append(row)
    assert len(rows) == 48 and len({r['row_id'] for r in rows}) == 48
    assert sum(r['original_query_available'] for r in rows) == 40
    assert all(not r['reviewed'] for r in rows)
    # Invented state checks only: U cannot receive D; D needs identities; E needs foreground evidence.
    checks = []
    base = next(r for r in rows if r['original_query_available'])
    tests = [
        dict(base, original_query_available=False, part_visibility_D_E_S_UNKNOWN='D'),
        dict(base, reviewed=True, part_visibility_D_E_S_UNKNOWN='D', short_visual_evidence='invented'),
        dict(base, reviewed=True, part_visibility_D_E_S_UNKNOWN='E', target_instance_identified='YES', intended_part_identified='YES', query_relation_on_near_off_UNKNOWN='ON', location_error_possible='NO', short_visual_evidence='invented'),
    ]
    for item in tests:
        try: validate(item)
        except AssertionError: checks.append(True)
        else: raise AssertionError('Invented invalid observation was accepted')
    put(H/'EMPTY48_OBSERVATION_TEMPLATE.json', {'completed': False, 'prepared_only': True, 'source_bindings': BINDINGS, 'rows': rows})
    flat = [{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in r.items()} for r in rows]
    with (H/'EMPTY48_OBSERVATION_TEMPLATE.csv').open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
    put(H/'PROTOCOL_BEFORE_C_IMAGE_OBSERVATION.json', {
        'completed': True, 'prepared_only': True, 'observer': 'C', 'code': binding(__file__), 'bindings': BINDINGS,
        'rows': 48, 'original_U_forced_UNKNOWN': 8, 'available_UNREVIEWED': 40, 'enums': ENUMS,
        'definitions': docs['scope']['observation_contract'],
        'identity_context': 'Original q8/full-frame and original crop can support assistant weak identification; no mesh/gold certification prerequisite. Missing selected detection box has no GT/fallback replacement.',
        'query_relation': 'ON/NEAR/OFF is a visual relationship, not coordinate gold or a new distance threshold. It is recorded independently from visibility.',
        'D_E_gate': 'Both require identified target and intended part plus visual evidence; E additionally requires identifiable external foreground. Other material, absence, background, OFF or own structure alone does not imply E.',
        'S_and_location': 'S is separate own-structure occlusion; possible location error and wrong part/instance remain separate, never automatic E.',
        'independence': 'No observer A values/files viewed. Value-free enum agreement only; no cross-observer comparison before both observations sealed.',
        'actual_observation_gate': 'Wait for root notice of completed preview/identity QA AND separate actual C observation authorization. Until then zero images, headers, arrays, models or fit/eval.',
        'consensus_and_training': 'This template performs no consensus or target generation. Root later handles exact D/E on/near agreement, U/disagreement/S/location exclusions and any separate admission.',
        'actual_image_views': 0, 'source_array_header_or_body_reads': 0, 'A_observation_reads': 0,
        'training_or_evaluation': False, 'goal_solved': False,
    })
    put(H/'INVENTED_SCHEMA_CHECKS.json', {'completed': True, 'PASS': True, 'invented_only': True, 'rejected_U_D_missing_identity_and_E_missing_foreground': checks, 'actual_48_initial_states_checked': True, 'image_views': 0})
    put(H/'PREPARATION_COMPLETE.json', {'completed': True, 'PASS': True, 'prepared_only': True, 'code': binding(__file__), 'protocol': binding(H/'PROTOCOL_BEFORE_C_IMAGE_OBSERVATION.json'), 'template': binding(H/'EMPTY48_OBSERVATION_TEMPLATE.json'), 'CSV': binding(H/'EMPTY48_OBSERVATION_TEMPLATE.csv'), 'checks': binding(H/'INVENTED_SCHEMA_CHECKS.json'), 'rows': 48, 'available_UNREVIEWED': 40, 'original_U_UNKNOWN': 8, 'actual_image_views': 0, 'A_observation_reads': 0})
    print(json.dumps(binding(H/'PREPARATION_COMPLETE.json')))

if __name__ == '__main__':
    main()
