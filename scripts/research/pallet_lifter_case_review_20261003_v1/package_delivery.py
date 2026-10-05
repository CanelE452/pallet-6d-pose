"""Package the isolated review preparation and verified source captures; no execution."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
import zipfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NAME = HERE.name
OUT = ROOT / 'data/pallet/results' / NAME
DOC = ROOT / '_docs/experiments' / NAME
PREFIX = 'pallet_lifter_case_review_20261003'


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def main():
    downloads = Path.home() / 'Downloads'
    assert downloads.is_dir()
    destination = downloads / (PREFIX + '.zip')
    if destination.exists():
        destination = downloads / (PREFIX + '_' + datetime.now().strftime('%H%M%S_%f') + '.zip')

    # Preserve all existing evidence byte for byte. A mismatch stops packaging.
    evidence = json.loads((DOC / 'MANIFEST.json').read_text(encoding='utf-8'))
    for item in evidence['artifacts']:
        data = (ROOT / item['path']).read_bytes()
        assert len(data) == item['bytes'] and sha_bytes(data) == item['sha256'], item['path']

    files = {}
    skipped = []
    for directory in (HERE, OUT, DOC):
        for path in sorted(directory.rglob('*')):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT)
            if '.git' in relative.parts or '__pycache__' in relative.parts or path.suffix == '.pyc':
                continue
            if path.name in ('server.pid', 'server_stdout.log', 'server_stderr.log', 'GPU_0.lock') or path.suffix == '.pending':
                skipped.append(relative.as_posix())
                continue
            if path.is_symlink():
                raise RuntimeError('Symlinks must be reviewed before packaging: ' + str(relative))
            files[relative.as_posix()] = path

    capture_map = json.loads((OUT / 'LIFTER_INPUT_AND_TIME_MAP.json').read_text(encoding='utf-8'))
    capture_count = 0
    for session in capture_map['sessions']:
        assert session['status'] == 'VERIFIED_COMPLETE'
        for item in session['files']:
            path = ROOT / 'extracted/depth_cam/rec' / item['filename']
            data = path.read_bytes()
            assert len(data) == item['bytes'] and sha_bytes(data) == item['sha256'], str(path)
            files[path.relative_to(ROOT).as_posix()] = path
            capture_count += 1
    assert capture_count == 20

    readme = '''CLI A 리프터 사례 검수 준비 묶음 (2026-10-03)

ZIP 전체를 한 폴더에 압축 해제하고, 이 안내 파일이 있는 폴더를
PowerShell 작업 폴더로 사용하세요. scripts/data/_docs/extracted 경로를 유지하세요.

1. 검수 화면 시작 (Python 3.12 표준 라이브러리만 사용)
py -3.12 -X utf8 scripts/research/pallet_lifter_case_review_20261003_v1/start_review.py
브라우저 주소: http://127.0.0.1:8765/
이미 같은 주소에서 기존 검수 서버를 사용 중이면, 그 창의 저장은 기존
작업 폴더에 반영됩니다. 다른 복사본을 열 때에는 --port 8766 등 빈 포트를
시작 명령 끝에 붙이고 해당 포트 주소를 열어 주세요.

2. 저장 및 재개
화면의 '진행 중 저장' 또는 '사람 검수 완료 저장'으로 저장하세요.
저장 파일: data/pallet/results/pallet_lifter_case_review_20261003_v1/review/annotations_in_progress.json
제출한 주석 JSON을 화면에서 내보내고 아래 명령으로 검증/평가를 재개하세요.
py -3.12 -X utf8 scripts/research/pallet_lifter_case_review_20261003_v1/resume.py "내보낸_JSON의_경로"
같은 복사본에 저장된 기록을 사용하면 JSON 인자를 생략합니다.
재개는 numpy/opencv와, 실제 모델 실행 시 torch/ultralytics/scipy 등 기존 평가
환경이 필요합니다. 원래 평가 소스와 requirements.txt는 evaluation_checkout에 포함됩니다.

3. 현재 상태
네 원본 세션 8,910개 저장 프레임과 시간 대응을 검증했습니다.
수동 검수 표본 120장, 반복 검수 24장을 준비했습니다.
기존 846개 결과의 완료 기록만 확보했으며 원예측/원계획은 미확보입니다.
고정 Base/N3 두 가중치가 없어 신규 추론 0, 재사용 0입니다.
모델 추론 BLOCKED_CONTRACT, 수동 가시 코너 정확도 WAITING_HUMAN,
독립 물리 정확도 BLOCKED_REFERENCE 상태이며 완료 정확도 수치는 없습니다.
가중치 제공 위치/정확한 해시는 scripts/research/pallet_lifter_case_review_20261003_v1/README_KO.md를 확인하세요.

4. 포함 자료
검수 도구, 120장 원본 PNG, 결과/해시/고정 계획, 한국어 최종 보고서,
읽기 전용 원고 복사본 및 사례 교체 패치, 기준 평가 코드,
해시 검증된 네 세션의 원본 영상/meta/timing/state/control 기록 20파일.
Git 내부, Python 캐시, 현재 서버 PID/로그/잠금은 제외했습니다.
175419 변경 영상은 주 평가에서 제외되어 원본 영상을 이 ZIP에 넣지 않았습니다.
이 묶음은 기존 고정 계획의 검수·재개용입니다. 제외 영상과 원래 Git
환경까지 요구하는 audit.py의 재-audit 명령은 이 묶음의 실행 범위에 포함되지 않습니다.
새 학습, 실제 리프터 제어, 정적 자료 수정, 공용 원고 적용을 수행하지 않았습니다.

보고서: _docs/experiments/pallet_lifter_case_review_20261003_v1/FINAL_REPORT_KO.md
검수 안내: scripts/research/pallet_lifter_case_review_20261003_v1/review/README.md
ZIP 파일별 SHA-256: PACKAGE_MANIFEST.json
이 파일은 전달 시점의 복사본입니다. 추후 저장한 주석은 이 ZIP에 자동 반영되지 않습니다.
'''.encode('utf-8')

    records = []
    # Create exclusively: never replace another deliverable.
    try:
        with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
            readme_name = PREFIX + '/README_다운로드_KO.txt'
            archive.writestr(readme_name, readme)
            records.append({'path': 'README_다운로드_KO.txt', 'bytes': len(readme), 'sha256': sha_bytes(readme)})
            for relative, path in sorted(files.items()):
                # Atomic source saves are captured as one immutable byte snapshot.
                data = path.read_bytes()
                archive.writestr(PREFIX + '/' + relative, data)
                records.append({'path': relative, 'bytes': len(data), 'sha256': sha_bytes(data)})
            manifest = {'schema_version': 'lifter_case_delivery_package_v1',
                        'created_at_local': datetime.now().astimezone().isoformat(),
                        'source_kind': 'machine_snapshot', 'preserves_project_relative_paths': True,
                        'verified_original_capture_files': capture_count,
                        'excluded_runtime_files': sorted(skipped),
                        'fixed_model_weights_included': False,
                        'prediction_or_accuracy_completion_claim': False,
                        'files': records}
            archive.writestr(PREFIX + '/PACKAGE_MANIFEST.json',
                             json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8'))
        with zipfile.ZipFile(destination) as archive:
            assert archive.testzip() is None, 'ZIP CRC failed'
            assert len(archive.namelist()) == len(records) + 1
            for item in records:
                data = archive.read(PREFIX + '/' + item['path'])
                assert len(data) == item['bytes'] and sha_bytes(data) == item['sha256'], item['path']
            # Real annotation context initialization validates the actual image hashes
            # after extraction, with no creation of human records or server process.
            import tempfile
            import sys
            with tempfile.TemporaryDirectory(prefix='lifter_zip_verify_') as temp:
                for item in records:
                    if item['path'].startswith('data/pallet/results/'):
                        archive.extract(PREFIX + '/' + item['path'], temp)
                copied_out = Path(temp) / PREFIX / 'data/pallet/results' / NAME
                sys.path.insert(0, str(HERE / 'review'))
                from serve import Context
                context = Context(copied_out/'review/MANIFEST.json', copied_out/'LIFTER_EVALUATION_PLAN.json',
                                  copied_out/'review/CORNER_CONTRACT.json', copied_out/'review/annotations_in_progress.json')
                catalog = context.catalog()
                assert len(catalog['frames']) == 120
                counts = catalog['counts']
    except Exception:
        # Preserve a failed archive for diagnosis and never present it as verified.
        raise

    digest = hashlib.sha256()
    with destination.open('rb') as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b''):
            digest.update(block)
    checksum = destination.with_suffix('.zip.sha256.txt')
    checksum.write_text(digest.hexdigest() + '  ' + destination.name + '\n', encoding='utf-8')
    print(json.dumps({'zip_path': str(destination), 'bytes': destination.stat().st_size,
                      'sha256': digest.hexdigest(), 'archive_entries': len(records) + 1,
                      'capture_files': capture_count, 'review_context_counts': counts,
                      'verification': 'CRC_AND_ALL_FILE_SHA256_AND_EXTRACTED_REVIEW_CONTEXT_PASS'},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
