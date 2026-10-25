"""Join approved original TRAIN metadata only; never open referenced RGB/arrays."""
import hashlib
import json
import re
from pathlib import Path

P = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
R = P / 'publication_root_preparation'
OUT = R / 'V227_ROOT_CONTEXT_PROBE'
SOURCE = P / 'query_visibility_mr6d_full_public_source_20261024_v153/full_source_v1/RGB230_INPUT_ONLY_MANIFEST.json'


def read(path):
    assert 'v112' not in str(path).lower()
    return path.read_bytes()


def binding(path):
    b = read(path)
    return dict(path=str(path), sha256=hashlib.sha256(b).hexdigest(), size_bytes=len(b))


def row_spans(b):
    start = re.search(rb'"rows"\s*:\s*\[', b)
    assert start
    depth = 0
    quoted = escaped = False
    left = None
    for i in range(start.end(), len(b)):
        c = b[i]
        if quoted:
            if escaped:
                escaped = False
            elif c == 92:
                escaped = True
            elif c == 34:
                quoted = False
            continue
        if c == 34:
            quoted = True
        elif c == 123:
            if depth == 0:
                left = i
            depth += 1
        elif c == 125:
            depth -= 1
            assert depth >= 0
            if depth == 0:
                yield b[left:i + 1]
        elif c == 93 and depth == 0:
            return
    raise ValueError('rows array not terminated')


def main():
    assert (OUT / 'SCOPE_BEFORE_RGB_INPUT_BINDING.json').is_file()
    whitepath = R / 'V224_ROOT_READINESS/ORIGIN_QUERY_TRAIN_WHITELIST.json'
    candidatepath = R / 'V225_GROUP_REVIEW_A/CANDIDATE_SOURCE_FIT_DEV_MANIFEST.json'
    white = json.loads(read(whitepath))
    candidate = json.loads(read(candidatepath))
    frames = {r['original_frame_index']: r for r in white['frames']}
    assert len(frames) == 151 and len(white['row_id_whitelist']) == 1208
    rows = candidate['rows']
    assert len(rows) == 1208
    assert {r['row_id'] for r in rows} == set(white['row_id_whitelist'])
    byframe = {}
    for r in rows:
        byframe.setdefault(r['original_frame_index'], []).append(r)
    # Binary traversal plus the frame_index header is disclosed; only approved
    # row bodies are decoded. No referenced payload path is followed.
    raw = read(SOURCE)
    approved = {}
    traversed = excluded = 0
    for body in row_spans(raw):
        traversed += 1
        header = re.search(rb'^\{\s*"frame_index"\s*:\s*(\d+)', body)
        assert header
        index = int(header.group(1))
        if index not in frames:
            excluded += 1
            continue
        row = json.loads(body)
        assert index not in approved and row['frame_index'] == index
        assert row['source_id'] == frames[index]['source_id']
        assert row['raw_hw'] == [720, 1280]
        assert all('v112' not in x.lower() for x in [row['image'], row['canonical_RGB_NPY']['path']])
        approved[index] = row
    assert traversed == 230 and excluded == 79 and set(approved) == set(frames)
    output = []
    for i in sorted(frames):
        rr = sorted(byframe[i], key=lambda r: r['native_slot'])
        assert [r['native_slot'] for r in rr] == list(range(8))
        assert [r['row_id'] for r in rr] == frames[i]['row_ids_by_native']
        assert all(r['original_component_role'] == 'TRAIN' and r['component_id'] == frames[i]['component_id'] for r in rr)
        assert len({r['proposed_role'] for r in rr}) == 1
        img = approved[i]
        output.append(dict(
            original_frame_index=i, source_id=frames[i]['source_id'], component_id=frames[i]['component_id'],
            original_component_role='TRAIN', candidate_role=rr[0]['proposed_role'],
            image=dict(path=img['image'], sha256=img['image_sha256'], decoded_RGB_sha256=img['decoded_RGB_sha256']),
            canonical_RGB_NPY=img['canonical_RGB_NPY'], raw_hw=img['raw_hw'], decoder=img['decoder'],
            image_binding_authority='Previously recorded input-only bindings; current payload bytes not opened or verified.',
            native_slots=list(range(8)), query_xy=[r['query_xy'] for r in rr],
            original_query_available=[r['original_query_available'] for r in rr],
            L_by_native=[r['L'] for r in rr], queries=rr,
            GT_pose_ROI_mask_or_reference_input=False, training_admission=False,
        ))
    target = OUT / 'ORIGINAL_TRAIN151_CONTEXT_INPUT_BINDINGS.jsonl'
    with target.open('x') as f:
        for row in output:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    summary = dict(
        completed=True, PASS=True, rows151=len(output), query_rows1208=len(rows),
        original_component_role='TRAIN', candidate_split_admitted=False, training_admission=False,
        source_manifest=binding(SOURCE), approved_whitelist=binding(whitepath), candidate=binding(candidatepath),
        raw_metadata_rows_and_identity_headers_traversed=traversed,
        approved_bodies_JSON_decoded=len(approved), excluded_bodies_not_JSON_decoded=excluded,
        excluded_raw_bytes_and_identity_headers_traversed=True, output=binding(target),
        original_q_L_native_U_and_labels_unchanged=True,
        referenced_RGB_NPY_array_GT_or_model_payloads_opened=0,
        actual_image_bytes_or_current_hash_verified=False, new_targets=0,
        new_training_forward_PnP=0, V112_filesystem_access=0,
        reuse_warning='Original-query role D/E plus frozen q8-context already exists in V155 code. This input join is not admission to repeat that probe.',
    )
    (OUT / 'CONTEXT_BINDING_SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k:summary[k] for k in ['completed','rows151','query_rows1208','raw_metadata_rows_and_identity_headers_traversed','approved_bodies_JSON_decoded','excluded_bodies_not_JSON_decoded','referenced_RGB_NPY_array_GT_or_model_payloads_opened']}))


if __name__ == '__main__':
    main()
