"""Copy a byte-verified SOURCE checkpoint and prefix three repository entry points."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import subprocess

R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
REPO = Path('/home/minjae/Documents/pallet-visibility-publication-20261024')
BASE = '_docs/experiments/pallet-keypoint-visibility'
NAME = 'V240_CANONICAL_CONTEXT_ACTUAL_RESULTS'
PARENT = 'b36d6d7e0a7633cfbef416285207503688b16d0a'


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
    package = R / 'V240_PUBLICATION_PACKAGE'
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
    prefixes = {'readme.md': '## 팔레트 가림 실험 최신 실제 결과: V239–V240\n\n[모델·특징·로그·지표·실제 crop·CPU 재검산](_docs/experiments/pallet-keypoint-visibility/V240_CANONICAL_CONTEXT_ACTUAL_RESULTS/README_KO.md). 공통 특징으로 local/context 두 모델을 각각 1,000회 학습했습니다. 공개 YCB-V SOURCE 내부 DEV 가림은 0/57→1/57이며 사전 기준 미달로 미채택입니다. 현재223 새 평가·PnP는 없습니다.\n\n', '_docs/experiments/pallet-keypoint-visibility/README.md': '## 최신 V239–V240: 공통 특징·두 모델 실제 학습 완료 / SOURCE 기준 미달\n\n[전체 보고서·재현 명령](V240_CANONICAL_CONTEXT_ACTUAL_RESULTS/README_KO.md) · [CM/precision/recall/보류율](V240_CANONICAL_CONTEXT_ACTUAL_RESULTS/artifacts/V240_SOURCE_REPORT_C/results/PORTABLE_REPORT_KO.md) · [성공·실패 crop](V240_CANONICAL_CONTEXT_ACTUAL_RESULTS/artifacts/V239_SOURCE_GALLERY_A/gallery_v1/FIXED12_SOURCE_DEV.png) · [문맥과 8개 query](V240_CANONICAL_CONTEXT_ACTUAL_RESULTS/artifacts/V239_SOURCE_GALLERY_A/gallery_v1/PORTABLE_GALLERY_KO.html).\n\n7,617개 활성 특징의 byte 일치와 원 좌표·분할 정합, 두 모델 각각 새 1,000회 학습과 동결, 독립 완료 검산·지표 재계산·CPU 예측 전량 일치를 확인했습니다. 공개 실사 SOURCE DEV E는 local0/57→context1/57, 가족 평균 균형 recall gain+0.345657pp로 사전 기준 FAIL입니다. 모델 채택·현재223 새 평가·문턱 조정은 없습니다. 6개 frozen 특징 배열과 두 작은 모델도 제공해 원 이미지·DINO 없이 CPU 검산할 수 있습니다. [공개 검증 기록](V240_PUBLICATION_REVIEW.json).\n\n', '_docs/experiments/pallet-keypoint-visibility/EXPERIMENT_INDEX_KO.md': '## V239–V240 · canonical 특징·local/context 새 학습·검산 완료 / 미채택\n\n[코드·6개 frozen 특징·두 모델·로그·행별 결과·표·crop·CPU replay](V240_CANONICAL_CONTEXT_ACTUAL_RESULTS/README_KO.md). 알려진 D/E 7,617개에서 입력 byte 정합을 확인하고 두 모델을 각각 동일 원 seed17/draw로 1,000회 학습했습니다. 합성 DEV context D/E recall98.76/91.07%, YCB DEV97.65/1.75%(E1/57), source gate FAIL(+0.345657pp). 공개 q-r≤3px E30도 모두 미검출입니다. 현재223/held 새 사용0·새 후보 admission=false·PnP0이며 원 q/L/native/U/FIT·DEV는 유지했습니다. 이전 준비·실패 기록을 아래에 보존합니다.\n\n'}
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
    out = R / 'V240_MAIN_PREPARATION.json'
    with out.open('x') as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'completed': True, 'PASS': True, 'copied': len(copied), 'entrypoints': len(entries),
                      'receipt_sha256': sha(out.read_bytes())}))


if __name__ == '__main__':
    main()
