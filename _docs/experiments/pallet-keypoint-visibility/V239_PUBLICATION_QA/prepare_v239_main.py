"""Copy a byte-verified SOURCE checkpoint and prefix three repository entry points."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import subprocess

R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
REPO = Path('/home/minjae/Documents/pallet-visibility-publication-20261024')
BASE = '_docs/experiments/pallet-keypoint-visibility'
NAME = 'V239_SOURCE_CONTEXT_INPUTS_AND_PREPARATION'
PARENT = 'b97c7cd7f37b3dead7fa561374a1cb685c3b6f34'


def safe(rel):
    assert isinstance(rel, str)
    assert not any(x in rel.lower() for x in ('v112', '__pycache__', '.pyc'))
    p = PurePosixPath(rel)
    assert not p.is_absolute() and '..' not in p.parts and '\\' not in rel
    return rel


def sha(b):
    return hashlib.sha256(b).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest-sha256', required=True)
    a = ap.parse_args()
    assert git('rev-parse', 'HEAD').decode().strip() == PARENT
    assert not git('status', '--porcelain=v1', '-uno').strip()
    assert not git('diff', '--cached', '--name-only', '-z')
    package = R / 'V239_PUBLICATION_PACKAGE'
    manifest_bytes = (package / 'MANIFEST.json').read_bytes()
    assert sha(manifest_bytes) == a.manifest_sha256
    manifest = json.loads(manifest_bytes)
    dest = REPO / BASE / NAME
    assert not dest.exists()
    copied = []
    for row in manifest['files']:
        rel = safe(row['path'])
        p = package / rel
        assert not p.is_symlink()
        b = p.read_bytes()
        assert (len(b), sha(b)) == (row['bytes'], row['sha256'])
        dst = dest / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b)
        copied.append({'path': f'{BASE}/{NAME}/{rel}', 'bytes': len(b), 'sha256': sha(b)})
    (dest / 'MANIFEST.json').write_bytes(manifest_bytes)
    copied.append({'path': f'{BASE}/{NAME}/MANIFEST.json', 'bytes': len(manifest_bytes), 'sha256': sha(manifest_bytes)})
    prefixes = {
        'readme.md': f'''## 팔레트 가림 실험 최신 공개: V237–V239

[상태표·실제 crop·진단 CSV·코드·검증 기록]({BASE}/{NAME}/README_KO.md). 초기 query 입력 7,617개의 RGB/valid 일치를 확인했습니다. V238은 학습 전 특징 검증에서 중단됐고, V239는 공통 특징 생성과 두 모델 비교 코드를 준비한 단계입니다. 새 V238/V239 학습 성능은 없습니다. 현재223 새 사용·PnP는 0입니다.

''',
        f'{BASE}/README.md': f'''## 최신 공개 V237–V239: 입력 복원 완료·학습 전 중단 진단·다음 비교 준비

[전체 보고서와 상태표]({NAME}/README_KO.md) · [원 문맥과 8개 실제 crop]({NAME}/artifacts/V237_ROOT_SOURCE_CONTEXT_INPUTS/fixed4_original_input_gallery/FIXED4_SOURCE_INPUT_CONTEXT_AND_Q8.png) · [FIT 5,876개 행별 특징 차이]({NAME}/artifacts/V238_ALL_FIT_DESCRIPTOR_DIAG/ALL5876_DESCRIPTOR_DIFFERENCES.csv).

V237은 원 crop 7,617개의 픽셀·valid SHA 일치와 독립 입력 QA를 완료했습니다. V238은 1e-6 특징 검증 초과로 optimizer 전에 중단됐습니다(66/5,876개, 최대 약 1.55158e-6). V239는 한 번 생성한 공통 특징으로 두 모델을 새로 비교하도록 준비했으며 실제 생성·학습·새 DEV 결과는 아직 없습니다. 완료·실패·준비 상태와 한계를 구분해 보존합니다. [공개 검증 기록](V239_PUBLICATION_REVIEW.json).

''',
        f'{BASE}/EXPERIMENT_INDEX_KO.md': f'''## V237–V239 · 원 입력 복원 완료 / 특징 검증 실패 진단 / 공통 특징 비교 준비

[실행 코드·진단 CSV·실제 crop·상태표·검산 안내]({NAME}/README_KO.md). V237: SOURCE 15,040행, 알려진 D/E 7,617개, 그룹 1,767개를 유지하고 원 RGB/valid 일치를 확인했습니다. V238: 실제 시도는 특징 일치 검사에서 중단, 새 optimizer 0회·새 DEV 출력 0개입니다. 기존 CONTROL 복사본은 이전 모델 기록입니다. V239: 공통 특징과 CONTROL/TREAT 각각 1,000회 학습 코드를 준비했으며 실제 실행 전입니다. 허용 오차 완화·현재223 새 사용·PnP는 없습니다. 이전 기록을 아래에 유지합니다.

''',
    }
    entries = []
    for rel, prefix in prefixes.items():
        p = REPO / safe(rel)
        assert not p.is_symlink()
        old = p.read_bytes()
        assert old == git('show', PARENT + ':' + rel)
        pre = prefix.encode()
        p.write_bytes(pre + old)
        assert p.read_bytes()[len(pre):] == old
        entries.append({'path': rel, 'old_sha256': sha(old), 'old_bytes': len(old),
                        'prefix_bytes': len(pre), 'sha256': sha(pre + old), 'bytes': len(pre + old),
                        'old_bytes_preserved_as_suffix': True})
    receipt = {'completed': True, 'PASS': True, 'parent_commit': PARENT,
               'manifest_sha256': a.manifest_sha256, 'package': str(package),
               'destination': str(dest), 'copied': copied, 'entry_points': entries,
               'new_current_or_held_payload': 0, 'cache_paths_touched': False,
               'absolute_metadata_pointer_targets_followed': False}
    out = R / 'V239_MAIN_PREPARATION.json'
    with out.open('x') as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'completed': True, 'PASS': True, 'copied': len(copied), 'entrypoints': len(entries),
                      'receipt_sha256': sha(out.read_bytes())}))


if __name__ == '__main__':
    main()
