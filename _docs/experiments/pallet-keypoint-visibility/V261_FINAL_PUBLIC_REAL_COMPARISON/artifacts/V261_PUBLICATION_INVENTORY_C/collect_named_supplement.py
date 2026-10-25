"""Small explicit CODE/policy supplement to frozen base proposal; metadata only."""
import hashlib
import json
import os
from pathlib import Path
import stat

R = Path(__file__).absolute().parent.parent
H = Path(__file__).absolute().parent
DIRS = ('V261_PUBLIC_AUX_FEATURE_BATCH_A', 'V261_PUBLIC_FEATURE_ENTRY_PREP_A')


def safe(p):
    p = Path(p)
    assert p.is_absolute() and '..' not in p.parts
    assert not any(t in str(p).casefold() for t in ('v112', 'cache', '.pyc'))
    for a in [*reversed(p.parents), p]:
        assert not stat.S_ISLNK(os.lstat(a).st_mode)
    return p


def raw(p):
    with os.fdopen(os.open(safe(p), os.O_RDONLY | os.O_NOFOLLOW), 'rb') as f:
        return f.read()


def bind(p):
    data = raw(p)
    return {'path': str(p), 'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)}


def run():
    base_path = H/'ALLOWLIST_PROPOSAL.json'
    base_bytes = raw(base_path)
    assert hashlib.sha256(base_bytes).hexdigest() == '39acebfc383a277dae5cc9d55620da7637ca07c2a03aba8a8bb225be5aa2ae05'
    base = json.loads(base_bytes)
    seen = {r['source']['path'] for r in base['files']}
    rows = []
    for directory in DIRS:
        for name in sorted(os.listdir(safe(R/directory))):
            if any(t in name.casefold() for t in ('v112', 'cache', '.pyc')):
                continue
            p = R/directory/name
            mode = os.lstat(p).st_mode
            assert not stat.S_ISLNK(mode)
            if not stat.S_ISREG(mode) or str(p) in seen:
                continue
            assert p.suffix.casefold() in ('.py', '.json', '.md', '.txt', '.diff')
            rows.append({'relative_path': 'artifacts/'+p.relative_to(R).as_posix(),
                         'group': 'attempts_code_protocols', 'source': bind(p),
                         'inventory_evidence': 'literal-bound A CODE parent, named shallow CODE/metadata bytes'})
            seen.add(str(p))
    # Named public teacher policy is saved metadata; no policy/body or old arrays followed.
    complete_path = R/'V261_ROOT_PUBLIC_BATCH_TEACHER/group_0p55m_hall_v1/COMPLETE.json'
    complete = json.loads(raw(complete_path))
    policy = {k: complete['policy'][k] for k in ('path', 'sha256', 'size_bytes')}
    if policy['path'] not in seen:
        p = Path(policy['path'])
        assert p.is_relative_to(R) and p.suffix == '.json'
        rows.append({'relative_path': 'artifacts/'+p.relative_to(R).as_posix(),
                     'group': 'public_teacher_policy', 'source': policy,
                     'inventory_evidence': 'exact saved public compiler policy binding only; no source stat/read'})
    result = {'completed': True, 'proposal_only': True, 'base_proposal': bind(base_path),
              'additional_files': rows, 'additional_count': len(rows),
              'no_payload_array_model_image_or_Git_network': True,
              'runtime_limit': 'Only exact CODE closure was copied. Full external DINO vendor tree is not inventoried; official encoder file and pinned revision remain explicit dependencies. No automatic download/install.',
              'existing_source_packet_limit': 'Old SOURCE7617 matrices/checkpoints were not recopied. Existing published dependency bindings stay inert; Root CPU replay closes its exact public inputs.',
              'future_Root_additions': ['readout/plots/status/attribution/CPU replay/actual QA',
                                       'this supplement and inventory FINAL_READY files']}
    p = H/'NAMED_CODE_POLICY_SUPPLEMENT.json'
    with p.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps({'completed': True, 'supplement': bind(p), 'additional_files': len(rows)}, sort_keys=True))


if __name__ == '__main__':
    run()
