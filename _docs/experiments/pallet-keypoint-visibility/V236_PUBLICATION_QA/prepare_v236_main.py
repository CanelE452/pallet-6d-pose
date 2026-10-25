"""Copy only the sealed V236 package and prefix three existing entry points."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import subprocess

ROOT = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
PACKAGE = ROOT / 'V236_PUBLICATION_PACKAGE'
REPO = Path('/home/minjae/Documents/pallet-visibility-publication-20261024')
BASE = '_docs/experiments/pallet-keypoint-visibility'
NAME = 'V236_SOURCE_DIAGNOSIS_AND_PAIRED_TRAINING'
DEST = REPO / BASE / NAME
EXPECTED_PARENT = '66e5b02833151e1e3c50d5914ab02a32bdfaf6dc'
EXPECTED_MANIFEST = '1209aa19c7ec860289f6134e1f42629dce871d2560c2f1045151f6ccb43f58c2'


def safe(value):
    s = str(value)
    assert 'v112' not in s.lower(), 'Lexically excluded version'
    assert '__pycache__' not in s and not s.endswith('.pyc'), 'Cache excluded'
    return s


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO)


assert git('rev-parse', 'HEAD').decode().strip() == EXPECTED_PARENT
assert not git('status', '--porcelain=v1', '-uno').strip()
assert not git('diff', '--cached', '--name-only', '-z')
manifest_bytes = (PACKAGE / safe('MANIFEST.json')).read_bytes()
assert sha(manifest_bytes) == EXPECTED_MANIFEST
manifest = json.loads(manifest_bytes)
assert len(manifest['files']) == 222
assert not DEST.exists()
copied = []
for row in manifest['files']:
    rel = PurePosixPath(safe(row['path']))
    assert not rel.is_absolute() and '..' not in rel.parts
    source = PACKAGE / rel
    assert not source.is_symlink()
    data = source.read_bytes()
    assert len(data) == row['bytes'] and sha(data) == row['sha256']
    target = DEST / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    copied.append({'path': f'{BASE}/{NAME}/{rel}', 'bytes': len(data), 'sha256': sha(data)})
(DEST / 'MANIFEST.json').write_bytes(manifest_bytes)
copied.append({'path': f'{BASE}/{NAME}/MANIFEST.json', 'bytes': len(manifest_bytes), 'sha256': sha(manifest_bytes)})

prefixes = {
    'readme.md': f'''## 팔레트 가림 실험 최신 공개: V234–V236

[진단·실제 학습·표·이미지·재검산 안내]({BASE}/{NAME}/README_KO.md). 같은 점을 덮는 COVER와 옆에 두는 SHAM 5,367쌍으로 고정 head를 학습했으나 공개 실사 내부 DEV 가림은 0/57로 미채택입니다. 코드·작은 모델·훈련로그·행별 출력·실제 성공/실패 crop·전체 장면을 포함합니다. 현재223 평가를 이번 학습·선택·추론에 사용하지 않았습니다.

''',
    f'{BASE}/README.md': f'''## 최신 공개 V234–V236: 실패 원인 진단과 같은 점 가림 대조

[전체 보고서·실행 조건·재검산]({NAME}/README_KO.md) · [CM/precision/recall/보류율]({NAME}/artifacts/V235_SOURCE_REPORT_C/results/PORTABLE_REPORT_KO.md) · [성공·실패 crop]({NAME}/artifacts/V235_SOURCE_GALLERY_A/gallery_v1/FIXED12_SOURCE_DEV.png) · [crop과 공개 원 장면]({NAME}/artifacts/V236_CONTEXT_FRAMES_B/three_source_context_v1/THREE_SOURCE_CONTEXT_BOARD.png).

실제 SOURCE 대조쌍5,367개와 고정 head1000회 학습·독립 QA를 완료했습니다. 공개 실사 내부 DEV 외부 가림은 0/57이며 SOURCE 기준 미달로 모델을 채택하지 않았습니다. 실패57개 중56개는 crop 밖 다른 원 query에 대상 픽셀이 있음을 확인했고, 다음 입력 검증 경로를 기록했습니다. 새 context 모델은 아직 학습하지 않았습니다. [배포 검증 기록](V236_PUBLICATION_REVIEW.json). 이전 기록은 아래에 유지합니다.

''',
    f'{BASE}/EXPERIMENT_INDEX_KO.md': f'''## V234–V236 · SOURCE 실패 진단·같은 점 COVER/SHAM 학습 완료 / 미채택

[코드·모델·로그·원행 출력·표·시각화·검산]({NAME}/README_KO.md). V234는 라벨/좌표/물체×표면점 분포와 기존 코드를 감사했습니다. V235는 공개 FIT donor64개로 성공5,367쌍을 만들고 같은 고정 encoder/head를1000회 학습했습니다. SYN DEV D/E recall98.68/92.86%, YCB DEV96.76/0%, 가족 평균 균형 recall gain+0.113491pp로 사전3pp 기준 미달입니다. V236은 공개 원 장면3개와 다른 q8의 대상 픽셀을 대조했습니다. 현재223 새 사용0, 새 후보 admission=false, PnP/좌표 회귀 없음입니다.

''',
}
changes = []
for rel, prefix in prefixes.items():
    target = REPO / safe(rel)
    assert not target.is_symlink()
    old = target.read_bytes()
    new_prefix = prefix.encode('utf-8')
    target.write_bytes(new_prefix + old)
    assert target.read_bytes()[len(new_prefix):] == old
    changes.append({'path': rel, 'old_sha256': sha(old), 'old_bytes': len(old),
                    'prefix_bytes': len(new_prefix), 'sha256': sha(new_prefix + old),
                    'bytes': len(new_prefix + old), 'old_bytes_preserved_as_suffix': True})

receipt = {'completed': True, 'parent_commit': EXPECTED_PARENT, 'package': str(PACKAGE),
           'destination': str(DEST), 'manifest_sha256': EXPECTED_MANIFEST,
           'copied': copied, 'entry_points': changes,
           'package_file_count': len(copied), 'current_or_held_new_payload': 0,
           'excluded_cache_paths_touched': False, 'absolute_metadata_pointers_followed': False}
out = ROOT / 'V236_MAIN_PREPARATION.json'
out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'completed': True, 'copied': len(copied), 'entry_points': len(changes),
                  'receipt': str(out), 'receipt_sha256': sha(out.read_bytes())}))
