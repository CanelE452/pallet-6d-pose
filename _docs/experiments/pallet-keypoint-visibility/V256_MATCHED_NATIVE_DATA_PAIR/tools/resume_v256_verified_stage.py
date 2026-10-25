"""Verify the staged V256 archive, preserving one frozen historical EOF blank."""
from pathlib import Path
import hashlib
import json
import stat
import subprocess

R = Path(__file__).absolute().parent
H = R / 'V256_ROOT_NATIVE_DATA_FIT'
PKG = R / 'V256_PUBLICATION_PACKAGE_V2'
REPO = Path('/home/minjae/Documents/pallet-visibility-publication-20261024')
DEST = Path('_docs/experiments/pallet-keypoint-visibility/V256_MATCHED_NATIVE_DATA_PAIR')
PARENT = 'a82d76f37545ab7b6bba84b6eea0e8a8414e98c0'
FACTORY = str(DEST / 'frozen_code/extraction/13_backbone_factory.py')


def safe(p):
    p = Path(p)
    assert p.is_absolute() and '..' not in p.parts
    assert not any(t in str(p).casefold() for t in ('v112', 'cache', '.pyc'))
    for a in reversed([p, *p.parents]):
        try:
            assert not stat.S_ISLNK(a.lstat().st_mode), str(a)
        except FileNotFoundError:
            pass
    return p


def read(p):
    return safe(p).read_bytes()


def sha(b):
    return hashlib.sha256(b).hexdigest()


def git(*args, inp=None, check=True):
    return subprocess.run(['git', *args], cwd=safe(REPO), input=inp,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          check=check)


def main():
    assert git('rev-parse', 'HEAD').stdout.decode().strip() == PARENT
    mb = read(PKG / 'MANIFEST.json')
    assert sha(mb) == '19f3343ad5d9c360c7ccc726fd49456e80df2ba03542022d44b7232ee99bdb3f'
    m = json.loads(mb)
    assert m['completed'] and m['PASS']
    expected = {}
    for row in m['files']:
        rel = Path(row['relative_path'])
        assert not rel.is_absolute() and '..' not in rel.parts
        b = read(PKG / rel)
        assert sha(b) == row['sha256'] and len(b) == row['size_bytes']
        expected[str(DEST / rel)] = b
    expected[str(DEST / 'MANIFEST.json')] = mb
    for name in ('ACTUAL_PACKAGED_FULL_RESTORE_QA.json', 'ACTUAL_MAIN_PUBLICATION_ENTRY.json'):
        expected[str(DEST / 'PUBLICATION_RECEIPTS' / name)] = read(H / name)
    refs = {
        'readme.md': ('V256：새 합성 가림 데이터의 동일 모델 비교', '_docs/experiments/pallet-keypoint-visibility/V256_MATCHED_NATIVE_DATA_PAIR/README_KO.md'),
        '_docs/experiments/pallet-keypoint-visibility/README.md': ('V256：기존 데이터와 새 합성 가림 대조의 동일 모델 비교', 'V256_MATCHED_NATIVE_DATA_PAIR/README_KO.md'),
        '_docs/experiments/pallet-keypoint-visibility/EXPERIMENT_INDEX_KO.md': ('V256：가림 대조 데이터 추가 효과와 공개 실사 미해결 기록', 'V256_MATCHED_NATIVE_DATA_PAIR/README_KO.md'),
    }
    preserved = {}
    for rel, (title, url) in refs.items():
        safe(REPO / rel)
        old = git('show', 'HEAD:' + rel).stdout
        prefix = ('> [' + title + '](' + url + ') — 동일 모델·초기값·1,000회 학습으로 새 합성 데이터 추가 효과를 비교했습니다. 합성 가림 recall은 증가했지만 precision·FIT 대조 패턴은 악화했고 공개 YCB 가림은 두 모델 모두 0/57입니다. 모델 채택은 하지 않았습니다.\n\n').encode()
        expected[rel] = prefix + old
        preserved[rel] = {'old_sha256': sha(old), 'old_size_bytes': len(old),
                          'prefix_size_bytes': len(prefix)}
    staged = git('diff', '--cached', '--name-only', '-z').stdout.decode().strip('\0').split('\0')
    assert sorted(staged) == sorted(expected)
    for rel, b in expected.items():
        assert read(REPO / rel) == b and git('show', ':' + rel).stdout == b
    assert sha(expected[FACTORY]) == 'c293292ded3805ee1b50f4c6d1c13689c18f35c5d71cb411643a26a6770c7b7b'
    config = ('-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol')
    diag = git(*config, 'diff', '--cached', '--check', check=False)
    expected_diag = (FACTORY + ':14: new blank line at EOF.\n').encode()
    assert diag.returncode == 2 and diag.stdout == expected_diag and not diag.stderr
    git(*config, 'diff', '--cached', '--check', '--', '.', ':(exclude)' + FACTORY)
    failure = json.loads(read(H / 'ROOT_STAGE_WHITESPACE_FAILURE.json'))
    code = read(Path(__file__))
    receipt = {
        'completed': True, 'PASS': True, 'parent': PARENT,
        'full_restore_QA': {'sha256': sha(read(H / 'ACTUAL_PACKAGED_FULL_RESTORE_QA.json'))},
        'all_existing_staged_and_working_bytes_exact': True,
        'files': [{'relative_path': p, 'sha256': sha(b), 'size_bytes': len(b)}
                  for p, b in sorted(expected.items())],
        'registry_preserved_suffixes': preserved,
        'preserved_stage_failure': failure,
        'narrow_archive_whitespace_exception': {
            'relative_path': FACTORY, 'sha256': sha(expected[FACTORY]),
            'exact_diagnostic': diag.stdout.decode(),
            'reason': 'Frozen historical source is preserved byte for byte; no source normalization or scientific change.',
            'all_other_staged_paths_diff_check_pass': True,
        },
        'resume_code': {'path': str(Path(__file__)), 'sha256': sha(code), 'size_bytes': len(code)},
        'commit_or_push_performed': False, 'force_push': False,
    }
    rb = (json.dumps(receipt, ensure_ascii=False, indent=2) + '\n').encode()
    extras = {
        str(DEST / 'tools/resume_v256_verified_stage.py'): code,
        str(DEST / 'PUBLICATION_RECEIPTS/ROOT_STAGE_ARCHIVE_EXCEPTION_QA.json'): rb,
    }
    for rel, b in extras.items():
        p = safe(REPO / rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('xb') as f:
            f.write(b)
    git('add', '-f', '--pathspec-from-file=-', '--pathspec-file-nul',
        inp=b'\0'.join(p.encode() for p in sorted(extras)) + b'\0')
    staged = git('diff', '--cached', '--name-only', '-z').stdout.decode().strip('\0').split('\0')
    assert sorted(staged) == sorted([*expected, *extras])
    for rel, b in extras.items():
        assert git('show', ':' + rel).stdout == b
    git(*config, 'diff', '--cached', '--check', '--', '.', ':(exclude)' + FACTORY)
    out = safe(R / 'V256_STAGED_BLOB_QA.json')
    with out.open('xb') as f:
        f.write(rb)
    print(json.dumps({'completed': True, 'PASS': True, 'changed_files': len(staged),
                      'package_leaves': len(m['files']), 'preserved_registry_suffixes': 3,
                      'exact_frozen_archive_EOF_exception': 1}, ensure_ascii=False))


if __name__ == '__main__':
    main()
