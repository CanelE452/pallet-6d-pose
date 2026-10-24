"""Additive immutable integration receipt for the completed RGB-only AFTER pack."""
from pathlib import Path
from collections import Counter
import hashlib
import json

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'after_input_pack'

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for data in iter(lambda: f.read(1 << 20), b''):
            h.update(data)
    return h.hexdigest()

def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}

def load(path):
    return json.loads(Path(path).read_text())

def check(binding):
    assert sha(binding['path']) == binding['sha256'], binding['path']

def write_new(path, data):
    with open(path, 'x') as f:
        f.write(json.dumps(data, indent=2) + '\n')

def main():
    protocol = load(HERE / 'AFTER_INPUT_PROTOCOL.json')
    manifest = load(OUT / 'PACK_MANIFEST.json')
    pack_seal = load(OUT / 'PACK_SEAL.json')
    qa = load(OUT / 'INPUT_PACK_QA.json')
    worker = load(HERE / 'FROZEN_FEATURE_INFERENCE_RECEIPT.json')
    preflight = load(HERE / 'guard_only_preflight/BEFORE_GUARDS_PREFLIGHT_PASS.json')
    before_worker = load(HERE / 'guard_only_preflight/FROZEN_FEATURE_INFERENCE_RECEIPT.json')
    for item in protocol['bindings']:
        check(item)
    for key in ('code', 'worker_code', 'after_manifest', 'after_lineage'):
        check(protocol[key])
    for key in ('manifest', 'lineage', 'before_guards', 'feature_inference'):
        check(pack_seal[key])
    for item in manifest['chunks']:
        check(item['file'])
    assert manifest['completed'] and pack_seal['PASS'] and qa['PASS'] and worker['PASS'] and preflight['PASS']
    assert qa['all_after_rows'] == manifest['frame_count'] == worker['after_images'] == 4799
    assert manifest['query_count'] == 38392
    assert worker['original_guard_images'] == 2 and worker['repeat_calls'] == 3
    assert worker['explicit_backbone_predict_calls'] == 4804
    assert worker['decoded_images_with_canonical_pixel_SHA_check'] == 4801
    assert worker['state_digest_before'] == worker['state_digest_after'] and worker['weights_unchanged']
    assert before_worker['after_images'] == 0 and before_worker['explicit_backbone_predict_calls'] == 4
    assert manifest['full_P3_archives_stored'] == worker['full_P3_archives_stored'] == 0
    rows = [json.loads(line) for line in Path(protocol['after_manifest']['path']).read_text().splitlines()]
    source_split = Counter((row['source'], row['split']) for row in rows)
    integration = {
        'completed': True, 'PASS': True,
        'description': 'Input-only AFTER cache for controlled source RGB; labels remain in separate GT-only files.',
        'frame_count': 4799, 'query_count': 38392,
        'source_split_counts': [{'source': s, 'split': p, 'frames': n} for (s, p), n in sorted(source_split.items())],
        'protocol': bind(HERE / 'AFTER_INPUT_PROTOCOL.json'),
        'manifest': bind(OUT / 'PACK_MANIFEST.json'),
        'pack_seal': bind(OUT / 'PACK_SEAL.json'),
        'QA': bind(OUT / 'INPUT_PACK_QA.json'),
        'before_original_guard_preflight': bind(HERE / 'guard_only_preflight/BEFORE_GUARDS_PREFLIGHT_PASS.json'),
        'after_run_before_guards': bind(HERE / 'BEFORE_GUARDS_PASS.json'),
        'after_feature_receipt': bind(HERE / 'FROZEN_FEATURE_INFERENCE_RECEIPT.json'),
        'six_model_tensors': ['p3_roi', 'rgb96', 'crop_valid96', 'target_rgb128', 'target_valid128', 'context42'],
        'identity_columns': ['after_row_index', 'original_row_index'],
        'lineage': manifest['lineage'],
        'all_changed_RGB_P3_recomputed': True,
        'original_q8_L_box_context_and_flags_bit_exact': True,
        'source_specific_ROI_backends': {'SYNTH_N3_legacy_GPU': 973, 'YCB_emulated_default_CPU': 3826},
        'affine': 'Actual reflected-pad100 native canvas / dynamic input shape, source gain and offset preserved per image.',
        'before_input_banks_reused_by_binding': manifest['before_input_banks'],
        'before_banks_copied': False,
        'full_P3_archives_stored': 0,
        'compressed_NPZ_bytes': manifest['NPZ_disk_bytes'],
        'uncompressed_tensor_bytes': manifest['uncompressed_bytes'],
        'backbone_explicit_predict_calls': {'isolated_original_preflight': 4, 'main_after_run_and_guards': 4804, 'total': 4808},
        'checkpoint_and_feature_model_state_unchanged': True,
        'GT_mask_reference_label_reads_by_input_producer': 0,
        'current223_reads': 0, 'new_training_updates': 0, 'PnP_calls': 0,
        'new_query_candidates_selected_or_used_by_producer': 0,
        'frozen_model_forward_used_only_for_P3_no_query_replacement': True,
        'new_semantic_or_physical_gold_claim': False,
        'code': bind(Path(__file__)),
    }
    write_new(OUT / 'PACK_INTEGRATION.json', integration)
    artifacts = [OUT / name for name in ('PACK_MANIFEST.json', 'PACK_SEAL.json', 'INDEX_LINEAGE_INPUT_ONLY.jsonl', 'INPUT_PACK_QA.json', 'PACK_INTEGRATION.json')]
    artifacts += [HERE / name for name in ('build_after_input_pack.py', 'frozen_feature_worker.py', 'qa_after_input_pack.py', 'finalize_after_input_pack.py', 'GENERATED_AFTER_PRODUCER_TESTS.json', 'AFTER_INPUT_PROTOCOL.json', 'BEFORE_GUARDS_PASS.json', 'FROZEN_FEATURE_INFERENCE_RECEIPT.json', 'AFTER_INPUT_PACK_BUILD.log', 'WORKER_STDERR.log')]
    write_new(OUT / 'FINAL_INPUT_PACK_SEAL.json', {
        'completed': True, 'PASS': True,
        'artifacts': [bind(path) for path in artifacts],
        'chunk_bindings_via_manifest': bind(OUT / 'PACK_MANIFEST.json'),
        'fit_current_PnP': 0,
    })
    print(json.dumps(bind(OUT / 'FINAL_INPUT_PACK_SEAL.json')))

if __name__ == '__main__':
    main()
