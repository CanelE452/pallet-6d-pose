"""Reference-free exhaustive preservation check for original-packet unions."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import time
from . import s1_union as U


def audit(run, output):
    run, output = Path(run), Path(output)
    started = time.monotonic()
    result = dict(schema='EXHAUSTIVE_ORIGINAL_BANK_UNION_AUDIT_V1',
                  numeric_reference_reads=0, new_solves=0, new_fits=0, domains={})
    for domain, expected in (('real', 238), ('synthetic', 1003)):
        base_path = run/f's1_{domain}_union'/'HEAD_FEATURES.jsonl.gz'
        base = list(U.rows(base_path))
        if len(base) != expected or len({f['id'] for f in base}) != expected:
            raise ValueError('Incomplete or duplicate base IDs')
        report = dict(base_sha256=U.sha(base_path), arms={})
        for arm in ('S3_COMMON', 'S3_BI', 'S4_PL', 'S4_CONTOUR', 'S4_IMAGE'):
            path = run/f'{arm.lower()}_{domain}_union'/'HEAD_FEATURES.jsonl.gz'
            frames = list(U.rows(path))
            if [f['id'] for f in frames] != [f['id'] for f in base]:
                raise ValueError('Union full ordered IDs differ')
            count = Counter(); changes = []
            for old, new in zip(base, frames):
                if len(new['candidates']) < len(old['candidates']):
                    raise ValueError('Original pose dropped')
                for a, b in zip(old['candidates'], new['candidates']):
                    if a['candidate_id'] != b['candidate_id'] or a['actual_pose'] != b['actual_pose']:
                        raise ValueError('Original actual pose or ordering changed')
                    if a['features'][:26] != b['features'][:26] or a['features'][31:] != b['features'][31:]:
                        raise ValueError('Original individual scalar or validity changed')
                    if not set(a.get('source_aliases', [])) <= set(b.get('source_aliases', [])):
                        raise ValueError('Original source alias lost')
                    count['old_candidates_pose_and_individual_features_exact'] += 1
                    if a['eligible'] != b['eligible']:
                        changes.append(dict(id=old['id'], candidate_id=a['candidate_id'],
                                            old=a['eligible'], new=b['eligible'],
                                            complete_action_aliases=b.get('complete_action_aliases', [])))
                count['frames'] += 1
            report['arms'][arm] = dict(counts=dict(count), old_candidate_eligibility_changes=len(changes),
                changes=changes, feature_sha256=U.sha(path),
                feature_seal_sha256=U.sha(path.parent/'FEATURE_SEAL.json'))
        result['domains'][domain] = report
    result['PASS'] = all(a['old_candidate_eligibility_changes'] == 0
                         for d in result['domains'].values() for a in d['arms'].values())
    result['wall_seconds'] = time.monotonic()-started
    U.write(output, result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=audit(a.run,a.output)
    print(U.canonical(dict(PASS=r['PASS'],wall_seconds=r['wall_seconds'])),flush=True)

if __name__=='__main__':main()
