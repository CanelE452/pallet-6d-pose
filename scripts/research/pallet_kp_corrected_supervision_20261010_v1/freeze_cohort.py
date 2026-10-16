"""Freeze existing human easy/medium labels before corrected real inference.

Reads existing metadata only. No model, geometry solver, rendering or labels
are generated. Output files are write-once; a second run checks exact bytes.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

NAME = 'pallet_kp_corrected_supervision_20261010_v1'
AUTHORITY = '_docs/experiments/pallet_static_registry_review_20261003_v1/native_severity_check_20261004_v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding(path, root):
    return dict(path=str(Path(path).relative_to(root)), sha256=digest(path), bytes=Path(path).stat().st_size)


def freeze(path, value):
    content = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
    if path.exists():
        assert path.read_bytes() == content, 'Frozen output already differs: ' + str(path)
    else:
        path.write_bytes(content)


def run(root, source):
    doc = root / '_docs/experiments' / NAME
    doc.mkdir(parents=True, exist_ok=True)
    inp = root / '_docs/experiments/pallet_observation_refiner_20261009_v1/INPUTS.json'
    authority_path = source / AUTHORITY / 'EFFECTIVE_FRAME_LABELS.json'
    snapshot_path = source / AUTHORITY / 'STATIC_SEVERITY_INPUTS_SNAPSHOT.json'
    report_path = source / AUTHORITY / 'REPORT_KO.md'
    runtime_path = source / 'data/pallet/results/pallet_n3_completion_v3/runtime/dope_seed1.json'
    inputs = json.loads(inp.read_text())['frames']
    authority = json.loads(authority_path.read_text())
    snapshot = json.loads(snapshot_path.read_text())
    labels = [r for r in authority['rows'] if r['population'] == 'DEV319']
    by_id = {r['frame_id']: r for r in labels}
    frames = {r['id']: r for r in inputs}
    assert len(labels) == len(by_id) == len(frames) == 319 and set(by_id) == set(frames)
    assert Counter(r['severity'] for r in labels) == dict(clean=153, moderate=92, severe=74)
    for fid, r in by_id.items():
        f = frames[fid]
        assert r['session'] == f['session'] and r['image']['path'] == f['image']
        assert r['image']['sha256'] == f['image_sha256']
    included = sorted(fid for fid, r in by_id.items() if r['severity'] in ('clean', 'moderate'))
    excluded = sorted(set(frames) - set(included))
    assert len(included) == 245 and len(excluded) == 74 and not set(included).intersection(excluded)
    source_rows = [dict(id=fid, source_row=by_id[fid],
                        recorded_input=snapshot['records'].get(by_id[fid]['case_id']))
                   for fid in sorted(frames)]
    evidence = dict(schema='existing_native_human_severity_source_evidence_v1',
                    source_bindings=[binding(p, source) for p in (authority_path, snapshot_path, report_path)],
                    authority_metadata={k:v for k,v in authority.items() if k != 'rows'},
                    snapshot_metadata={k:v for k,v in snapshot.items() if k not in ('records','history')},
                    explicit_records=316, retained_approved_records=3,
                    original_approved_counts=dict(Counter(r['original_approved']['status'] for r in labels)),
                    label_changes_from_original_approved=sum(r['severity'] != r['original_approved']['status'] for r in labels),
                    rows=source_rows)
    evidence_path = doc / 'COHORT_SOURCE_LABELS.json'
    freeze(evidence_path, evidence)
    def item(fid):
        r = by_id[fid]
        return dict(id=fid, session=r['session'], label=r['severity'], image=r['image'],
                    source_case_id=r['case_id'], source=r['source'])
    cohort = dict(schema='corrected_supervision_easy_medium_cohort_v1',
                  freeze_stage='before_corrected_real_observations_or_accuracy_read',
                  user_scope='existing easy and medium labels only; exclude difficult/heavily occluded category',
                  included_labels=['clean','moderate'], excluded_labels=['severe'],
                  label_display=dict(clean='easy / 가림 없음', moderate='medium / 중간', severe='hard / 어려움·심한 가림'),
                  count=len(included), ids=included, frames=[item(fid) for fid in included],
                  excluded_count=len(excluded), excluded_ids=excluded,
                  counts=dict(clean=153, moderate=92, severe_excluded=74, original=319),
                  sessions=sorted({frames[fid]['session'] for fid in included}),
                  session_counts=dict(sorted(Counter(frames[fid]['session'] for fid in included).items())),
                  bindings=[binding(inp, root), binding(evidence_path, root)],
                  disjoint=True, complete_original_population_partition=True,
                  selection_uses_prediction_or_pose_error=False,
                  human_identity_confirmed=False, reference_or_corner_truth_newly_certified=False,
                  limitations=[
                      'Medium labels can include partial blocking. This cohort is not a zero-occlusion cohort.',
                      'The stored UI class criterion was not confirmed as external occlusion versus overall difficulty.',
                      'The existing 316 explicit input records and 3 retained approved labels are reused without reannotation.',
                      'The old full-319 experiment remains unchanged; its results are not the primary denominator here.'])
    cohort_path = doc / 'COHORT.json'
    freeze(cohort_path, cohort)
    sessions = cohort['sessions']
    pools = {s:sorted(fid for fid in included if frames[fid]['session'] == s) for s in sessions}
    panel_ids = [pools[s][round_index] for round_index in range(2) for s in sessions if len(pools[s]) > round_index]
    panel_ids += [fid for fid in included if fid not in panel_ids][:26-len(panel_ids)]
    assert len(panel_ids) == len(set(panel_ids)) == 26 and set(panel_ids).issubset(included)
    old_panel = json.loads(runtime_path.read_text())['selected']
    session_metadata = {}
    for r in old_panel:
        previous = session_metadata.setdefault(r['session_id'], r)
        assert previous['object_type'] == r['object_type'] and previous['dimensions_wdh_m'] == r['dimensions_wdh_m']
    panel_frames = []
    for fid in panel_ids:
        f, label = frames[fid], by_id[fid]
        dims = [f['xyz'][0], f['xyz'][2], f['xyz'][1]]
        assert dims == session_metadata[f['session']]['dimensions_wdh_m']
        assert digest(source / f['image']) == f['image_sha256']
        panel_frames.append(dict(frame_id=fid, session_id=f['session'], image_key=f['image'],
                                 image=label['image'], original_hw=f['raw_hw'], camera_intrinsics=f['K'],
                                 dimensions_wdh_m=dims, object_type=session_metadata[f['session']]['object_type'],
                                 label=label['severity'], input_selected_index=f['selected_index']))
    freeze(doc / 'RUNTIME_PANEL.json', dict(schema='eligible_actual_runtime_panel_v1', cohort=binding(cohort_path, root),
        selection='sorted session round-robin first two eligible frames per session; fill sorted unused to 26',
        frames=panel_frames, count=26, sessions=len(sessions), session_counts=dict(Counter(r['session_id'] for r in panel_frames)),
        binding_input_registry_metadata=binding(runtime_path, source), selection_uses_accuracy=False))
    cases = []
    for label in ('clean','moderate'):
        cpools = {s:[fid for fid in pools[s] if by_id[fid]['severity'] == label] for s in sessions}
        chosen = [cpools[s][0] for s in sessions if cpools[s]][:3]
        chosen += [fid for fid in included if by_id[fid]['severity'] == label and fid not in chosen][:3-len(chosen)]
        assert len(chosen) == 3
        cases.extend(item(fid) for fid in chosen)
    freeze(doc / 'VISUAL_CASE_PROTOCOL.json', dict(schema='eligible_real_visual_case_selection_v1',
        cohort=binding(cohort_path, root), count_cases=6, count_panels=12,
        selection='for each label clean/moderate: first sorted eligible frame in first three sorted sessions with that label; fill sorted unused if needed',
        cases=cases, panel_methods=['N3_SUBPIX','CORRECTED_IMAGE_ROLE'],
        selection_uses_corrected_accuracy=False, posthoc_illustration_not_performance_evidence=True,
        raw_rgb_publication=False, new_annotation=False))
    print(json.dumps(dict(cohort_count=245, session_counts=cohort['session_counts'], panel_ids=panel_ids,
                          case_ids=[r['id'] for r in cases], cohort_sha256=digest(cohort_path))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.source_root.resolve())
