"""Explicit publication copy/stage/blob checks; no broad staging or model work."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess

R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
REPO = Path('/home/minjae/Documents/pallet-visibility-publication-20261024')
BASE = '_docs/experiments/pallet-keypoint-visibility'
NAME = 'V244_SCENE_DATA_AND_FROZEN_BEHAVIOR'
PARENT = '8c2ae5c47a831339f626e5d502ef9092a9a6e474'


def safe(rel):
    assert isinstance(rel, str)
    assert not any(s in rel.casefold() for s in ('v112', '__pycache__', '.pyc'))
    p = PurePosixPath(rel)
    assert not p.is_absolute() and '..' not in p.parts and '\\' not in rel
    return rel


def sha(b):
    return hashlib.sha256(b).hexdigest()


def guarded(p):
    p = Path(p)
    assert p.is_absolute() and '..' not in p.parts
    assert not any(s in str(p).casefold() for s in ('v112', '__pycache__', '.pyc'))
    for ancestor in reversed((p, *p.parents)):
        assert not ancestor.is_symlink()
    return p


def git(*args):
    return subprocess.check_output(['git', *args], cwd=guarded(REPO))


def jsave(p, obj):
    with guarded(p).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


def prepare():
    assert git('rev-parse', 'HEAD').decode().strip() == PARENT
    assert not git('status', '--porcelain=v1', '-uno').strip()
    assert not git('diff', '--cached', '--name-only', '-z')
    package = R / 'V244_PUBLICATION_PACKAGE'
    mb = guarded(package / 'MANIFEST.json').read_bytes()
    manifest = json.loads(mb)
    assert manifest['scientific_status']['new_training_updates'] == 0
    assert manifest['scientific_status']['V242_data_QA_PASS'] is True
    assert manifest['scientific_status']['V244_both_frozen_diagnostics_complete'] is True
    qa = R / 'V244_PUBLICATION_VERIFY_C/REVIEW.json'
    qb = guarded(qa).read_bytes()
    q = json.loads(qb)
    assert q['completed'] and q['PASS']
    assert q['manifest_sha256'] == sha(mb)
    replay = R / 'V244_PORTABLE_SAVED_OUTPUT_REPLAY.json'
    rb = guarded(replay).read_bytes()
    assert json.loads(rb)['PASS'] is True
    dest = REPO / BASE / NAME
    assert not guarded(dest).exists()
    copied = []

    def put(rel, data):
        p = guarded(REPO / safe(rel))
        assert not p.is_symlink() and not p.exists()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        copied.append({'path': rel, 'sha256': sha(data), 'bytes': len(data)})

    for row in manifest['files']:
        rel = safe(row['path'])
        p = guarded(package / rel)
        assert not p.is_symlink()
        b = p.read_bytes()
        assert (len(b), sha(b)) == (row['bytes'], row['sha256'])
        put(f'{BASE}/{NAME}/{rel}', b)
    put(f'{BASE}/{NAME}/MANIFEST.json', mb)
    put(f'{BASE}/V244_PUBLICATION_REVIEW.json', qb)
    put(f'{BASE}/V244_PORTABLE_SAVED_OUTPUT_REPLAY.json', rb)
    prefixes = {'readme.md': '## 팔레트 가림: V242 실제 데이터 검산·V244 동결 모델 진단\n\n[최신 결과·시각화·코드·검증 명령](_docs/experiments/pallet-keypoint-visibility/V244_SCENE_DATA_AND_FROZEN_BEHAVIOR/README_KO.md). 같은 원 장면에서 가림 대조 입력5,406개를 시도했고5,367개가 교사 조건을 통과했습니다. 기존 모델 내부 점수와 옛 학습 쌍 동작도 추가 진단했습니다. 새 학습·현재 평가셋 새 사용·PnP는0이며 성능 개선 완료를 뜻하지 않습니다. 이전 기록은 아래에 보존합니다.\n\n', '_docs/experiments/pallet-keypoint-visibility/README.md': '## V242/V244: 실제 장면 합성 검산과 동결 모델 행동 진단\n\n[전체 설명·표·실제 crop·코드](V244_SCENE_DATA_AND_FROZEN_BEHAVIOR/README_KO.md). 실제 생성5,367쌍/미생성39개 보존, 원 BEFORE·q8·문맥·교사 조건의 독립 전량 검산 완료. TREAT 자체 base/residual/total과 V235 학습 COVER/SHAM 기능을 분리 진단했습니다. V243는 코드 준비 상태이며 새 학습 성능 결과가 아닙니다. [공개 검산](V244_PUBLICATION_REVIEW.json).\n\n', '_docs/experiments/pallet-keypoint-visibility/EXPERIMENT_INDEX_KO.md': '## V242–V244 · 실제 데이터 검산 / 새 학습 전 동결 행동 진단\n\n[상태·진단 표·고정4사례·실패 원본·코드·재검산](V244_SCENE_DATA_AND_FROZEN_BEHAVIOR/README_KO.md). V2425,406부모 전부 시도,5,367proof PASS/39미생성 보존. V244는 V240 TREAT 자체 분기와 V235 옛 CF 쌍의 동결 forward 결과입니다. 새 optimizer0, current/held 새 payload0, PnP0, 모델 미채택. V241 준비 자료와 이전 결과는 아래에 유지합니다.\n\n'}
    entries = []
    for rel, prefix in prefixes.items():
        p = guarded(REPO / safe(rel))
        assert not p.is_symlink()
        old = p.read_bytes()
        assert old == git('show', PARENT + ':' + rel)
        pre = prefix.encode()
        p.write_bytes(pre + old)
        assert p.read_bytes()[len(pre):] == old
        entries.append({'path': rel, 'sha256': sha(pre + old), 'bytes': len(pre + old),
                        'old_sha256': sha(old), 'old_bytes_preserved_as_suffix': True})
    jsave(R / 'V244_MAIN_PREPARATION.json', {'completed': True, 'PASS': True,
        'parent_commit': PARENT, 'manifest_sha256': sha(mb), 'copied': copied,
        'entry_points': entries, 'two_ignored_cache_paths_not_touched': True,
        'new_model_or_payload_execution': 0})
    print(json.dumps({'completed': True, 'PASS': True, 'new_files': len(copied), 'entry_points': len(entries)}))


def check(stage=False, commit=False):
    rec = json.loads(guarded(R / 'V244_MAIN_PREPARATION.json').read_bytes())
    rows = rec['copied'] + rec['entry_points']
    names = [safe(row['path']) for row in rows]
    assert len(names) == len(set(names))
    if commit:
        assert git('rev-parse', 'HEAD^').decode().strip() == PARENT
        changed = set(git('diff-tree', '--no-commit-id', '--name-only', '-r', '-z', 'HEAD').decode().strip('\0').split('\0'))
    else:
        assert git('rev-parse', 'HEAD').decode().strip() == PARENT
        for row in rows:
            p = guarded(REPO / row['path'])
            assert not p.is_symlink()
            b = p.read_bytes()
            assert (len(b), sha(b)) == (row['bytes'], row['sha256'])
        if stage:
            assert not git('diff', '--cached', '--name-only', '-z')
            allow = guarded(R / 'V244_STAGE_ALLOWLIST.nul')
            with allow.open('xb') as f:
                f.write(b''.join(name.encode() + b'\0' for name in names))
            subprocess.run(['git', 'add', '-f', '--pathspec-from-file=' + str(allow), '--pathspec-file-nul'], cwd=guarded(REPO), check=True)
        changed = set(git('diff', '--cached', '--name-only', '-z').decode().strip('\0').split('\0'))
    assert changed == set(names), (changed - set(names), set(names) - changed)
    for row in rows:
        revision = ('HEAD:' if commit else ':') + row['path']
        b = git('show', revision)
        assert (len(b), sha(b)) == (row['bytes'], row['sha256'])
    mode = 'COMMIT' if commit else 'STAGED'
    jsave(R / f'V244_{mode}_QA.json', {'completed': True, 'PASS': True,
        'files_verified': len(rows), 'all_blob_SHA_and_bytes_exact': True,
        'manifest_sha256': rec['manifest_sha256'], 'explicit_allowlist_only': True,
        'revision': git('rev-parse', 'HEAD').decode().strip(), 'force_push': False})
    print(json.dumps({'completed': True, 'PASS': True, 'mode': mode, 'files_verified': len(rows)}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('prepare', 'stage', 'check-commit'))
    args = ap.parse_args()
    if args.mode == 'prepare':
        prepare()
    else:
        check(stage=args.mode == 'stage', commit=args.mode == 'check-commit')
