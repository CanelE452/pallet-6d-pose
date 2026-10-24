"""Exact input collisions and paired RGB changes on saved TRAIN diagnostic only."""
from pathlib import Path
import hashlib, json
import numpy as np

HERE = Path(__file__).resolve().parent


def binding(path):
    path = Path(path).resolve()
    assert path.is_relative_to(HERE)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def load(b):
    assert binding(b['path']) == b
    with np.load(b['path'], allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def run():
    cp = HERE / 'common_query_diagnostic_v1/COMPLETE.json'
    c = json.loads(cp.read_text())
    assert c['PASS'] and c['diagnostic_only'] and not c['fit_admission']
    x = load(c['inputs'])
    y = load(c['targets'])['support_target']
    cube = np.arange(4096).reshape(128, 4, 8)
    before, cover = cube[:, 0, :].reshape(-1), cube[:, 1, :].reshape(-1)
    av = x['original_query_available']
    assert int((~av).sum()) == 96
    coll = {}
    for i in np.flatnonzero(av):
        # Absolute q/L/native IDs are metadata, not channels of the frozen head.
        key = hashlib.sha256(x['rgb96'][i].tobytes() + x['valid96'][i].tobytes()).hexdigest()
        coll.setdefault(key, []).append(int(i))
    conflicts = []
    for ids in coll.values():
        for j in range(2):
            if len({int(y[i, j]) for i in ids if y[i, j] >= 0}) > 1:
                conflicts.append(dict(rows=ids, support=j))
    diff = np.abs(x['rgb96'][before].astype(np.int16) - x['rgb96'][cover].astype(np.int16))
    changed = np.any(diff, axis=3)
    counts = changed.sum((1, 2))
    truth_flip = (y[before] >= 0) & (y[cover] >= 0) & (y[before] != y[cover])
    flips = {}
    for j, name in enumerate(('visible_end', 'foreign_front')):
        take = truth_flip[:, j]
        values = counts[take]
        mae = diff[take].mean((1, 2, 3))
        flips[name] = dict(known_truth_flip_pairs=int(take.sum()),
            exact_same_RGB_with_known_truth_flip=int((take & (counts == 0)).sum()),
            changed_RGB_pixel_count=dict(min=int(values.min()), median=float(np.median(values)), max=int(values.max())) if len(values) else None,
            whole_crop_channel_absolute_difference_mean=dict(min=float(mae.min()), median=float(np.median(mae)), max=float(mae.max())) if len(mae) else None)
    out = HERE / 'saved_raw_cue_audit_v1'
    assert not out.exists()
    out.mkdir()
    ledger = out / 'ALL1024_BEFORE_COVER_RGB_CHANGE_ROWS.jsonl'
    with ledger.open('x') as f:
        for k, (a, b) in enumerate(zip(before, cover)):
            row = dict(base_index=k // 8, native_slot=k % 8, before_diagnostic_index=int(a), cover_diagnostic_index=int(b),
                original_anchor_available=bool(av[a]), changed_RGB_pixels=int(counts[k]),
                before_support=y[a].tolist(), cover_support=y[b].tolist(),
                channel_MAE=float(diff[k].mean()), channel_max_difference=int(diff[k].max()))
            f.write(json.dumps(row) + '\n')
    result = dict(completed=True, PASS=True, code=binding(__file__), raw_complete=binding(cp), inputs=c['inputs'], targets=c['targets'],
        diagnostic_rows=4096, available_rows=int(av.sum()), distinct_available_RAW_inputs=len(coll),
        exact_RAW_input_conflicting_known_target_groups=len(conflicts), conflicts=conflicts,
        BEFORE_COVER_pairs=1024, available_pairs=int(av[before].sum()), identical_RGB_pairs=int((counts == 0).sum()),
        support_truth_flips=flips, rows=binding(ledger),
        limits='No exact byte-identical input/contradictory known target is found if conflict count0. This does not certify recognizability, contrast at the particular end, robustness, native identity, heldout accuracy or real transfer.',
        source_TRAIN_only=True, no_model_forward_fit_or_selection=True, selected_model=None, fully_solved=False)
    for b in (c['inputs'], c['targets']):
        assert binding(b['path']) == b
    (out / 'COMPLETE.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    run()
