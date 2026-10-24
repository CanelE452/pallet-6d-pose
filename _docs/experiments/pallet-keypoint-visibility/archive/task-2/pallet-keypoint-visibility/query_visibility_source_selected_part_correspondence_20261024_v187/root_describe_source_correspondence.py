"""Describe sealed source-only metadata; no input asset or model imports."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import os
import stat

V = Path(__file__).absolute().parent
P = V.parent
SCOPE = V / 'ROOT_DESCRIPTIVE_SOURCE_AGGREGATION_SCOPE_BEFORE_QUERY_DIAGNOSTIC_BODIES.json'
OUT = V / 'descriptive_source_tables_v1'


def safe_open(path):
    p = Path(path)
    assert p.is_absolute() and '..' not in p.parts and P in p.parents
    assert all('v112' not in s.casefold() for s in p.parts)
    fd = os.open(P, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in p.relative_to(P).parts[:-1]:
            new = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = new
        final = os.open(p.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
    finally:
        os.close(fd)
    assert stat.S_ISREG(os.fstat(final).st_mode)
    return os.fdopen(final, 'rb')


def bind(path):
    with safe_open(path) as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest}


def read(path):
    with safe_open(path) as f:
        return json.load(f)


def checked(binding):
    assert bind(binding['path']) == binding
    return Path(binding['path'])


def lines(binding):
    with safe_open(checked(binding)) as f:
        return [json.loads(line) for line in f if line.strip()]


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def quantiles(values):
    x = sorted(values)
    def at(q):
        index = (len(x) - 1) * q
        lo = int(index)
        hi = min(lo + 1, len(x) - 1)
        return x[lo] + (index - lo) * (x[hi] - x[lo])
    return dict(n=len(x), **{k: at(q) if x else None for k, q in
                            [('min', 0), ('p25', .25), ('median', .5), ('p75', .75), ('max', 1)]})


def query_summary(rows):
    available = [r for r in rows if r['original_query_available']]
    weak = [r for r in available if r['nominal_correspondence']['status'] == 'WEAK_INSTANCE_NOMINAL_REFERENCE']
    valid = [r for r in weak if r['nominal_correspondence']['nominal_discrepancy_and_crop_diagnostic']['valid_nominal_discrepancy']]
    diagnostics = [r['nominal_correspondence']['nominal_discrepancy_and_crop_diagnostic'] for r in valid]
    return dict(total=len(rows), input_available=len(available), inputU=len(rows)-len(available),
                all_row_lookup_status=dict(Counter(r['nominal_correspondence']['status'] for r in rows)),
                available_lookup_status=dict(Counter(r['nominal_correspondence']['status'] for r in available)),
                eligible_available_weak_nominal=len(valid),
                eligible_inside_original_crop_extent=sum(d['inside_outer_crop_extent'] for d in diagnostics),
                eligible_outside_original_crop_extent=sum(not d['inside_outer_crop_extent'] for d in diagnostics),
                eligible_inside_sample_center_rectangle=sum(d['inside_sample_center_rectangle'] for d in diagnostics),
                eligible_reference_in_original_sampler_domain=sum(d['reference_in_original_sampler_valid_domain'] for d in diagnostics),
                nominal_distance_px=quantiles([d['nominal_distance_px'] for d in diagnostics]),
                nominal_distance_over_L=quantiles([d['nominal_distance_over_L'] for d in diagnostics]))


def main():
    scope = read(SCOPE)
    c = read(checked(scope['C_complete']))
    a = read(checked(scope['nominal_complete']))
    assert c['PASS'] and a['PASS'] and read(checked(scope['instance_independent_review']))['PASS']
    frames = lines(c['outputs']['ALL185_FRAME_INSTANCE_ASSOCIATIONS.jsonl'])
    queries = lines(a['outputs']['queries'])
    instances = lines(a['outputs']['instances'])
    refs = lines(a['outputs']['references'])
    assert len(frames) == 185 and len(queries) == 1480 and len(instances) == 258 and len(refs) == 2064
    assert sum(not r['original_query_available'] for r in queries) == 194
    assert Counter(r['native_slot'] for r in queries) == Counter({i: 185 for i in range(8)})
    assert all(r['semantic_state'] == 'UNKNOWN' and r['training_admission'] is False
               and r['native_visibility_target'] == r['point_quality_target'] == -1 for r in refs)
    scenes = sorted({f['scene'] for f in frames})
    per_scene = []
    for scene in scenes:
        selected = [f for f in frames if f['scene'] == scene]
        objects = Counter(str(f['diagnostic_association']['selected_obj_id']) for f in selected)
        per_scene.append(dict(scene=scene, frames=len(selected), selected_obj1=objects['1'],
                              selected_other=sum(n for k, n in objects.items() if k not in ('1', 'None')),
                              selected_UNKNOWN=objects['None'], selected_obj_ids=dict(objects)))
    native = {str(k): query_summary([r for r in queries if r['native_slot'] == k]) for k in range(8)}
    result = dict(completed=True, PASS=True, scope=bind(SCOPE), code=bind(Path(__file__).absolute()),
                  source_receipts={'C': scope['C_complete'], 'nominal': scope['nominal_complete'],
                                   'independent_instance': scope['instance_independent_review']},
                  frames=185, all_GT_observations=c['counts']['all_GT_instance_observations'],
                  obj1_instances=len(instances), all_obj1_nominal_references=len(refs),
                  frame_status=c['counts']['frame_association_states'], selected_obj_ids=c['counts']['selected_obj_ids'],
                  per_scene=per_scene, all_queries=query_summary(queries), per_native=native,
                  fit_admission=False, new_labels=False, evaluation_or_source_pixels_or_models_read=0,
                  interpretation='Weak bbox association and nominal discrepancy only. This is not object identity gold, physical coordinate error, or native D/E/S truth. No fitted threshold or data/model selection.')
    OUT.mkdir(exist_ok=False)
    write(OUT / 'SUMMARY.json', result)
    with (OUT / 'PER_SCENE_INSTANCE_MATCH.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, ['scene', 'frames', 'selected_obj1', 'selected_other', 'selected_UNKNOWN'])
        writer.writeheader()
        writer.writerows({k: r[k] for k in writer.fieldnames} for r in per_scene)
    fields = ['native_slot', 'total', 'input_available', 'inputU', 'eligible_available_weak_nominal',
              'eligible_inside_original_crop_extent', 'eligible_outside_original_crop_extent']
    with (OUT / 'PER_NATIVE_ORIGINAL_CROP_COVERAGE.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for k, row in native.items():
            writer.writerow(dict(native_slot=int(k), **{field: row[field] for field in fields[1:]}))
    receipt = dict(completed=True, PASS=True, summary=bind(OUT / 'SUMMARY.json'),
                   outputs=[bind(OUT / name) for name in ['SUMMARY.json', 'PER_SCENE_INSTANCE_MATCH.csv',
                                                         'PER_NATIVE_ORIGINAL_CROP_COVERAGE.csv']],
                   scope=bind(SCOPE), code=bind(Path(__file__).absolute()), automatic_retry=False)
    write(OUT / 'COMPLETE.json', receipt)
    print(json.dumps(dict(complete=bind(OUT / 'COMPLETE.json'), selected_obj_ids=result['selected_obj_ids'],
                          per_scene=per_scene, all_queries=result['all_queries'], per_native=native)))


if __name__ == '__main__':
    main()
