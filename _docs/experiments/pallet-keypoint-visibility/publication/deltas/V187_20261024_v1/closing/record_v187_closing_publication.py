"""Record an already verified public payload; seal a fresh closing-only plan."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import stat

P = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
ADMIN = P / 'publication_root_preparation'
REMOTE = Path('/home/minjae/Documents/publication_remote_verification_b_v187_20261024')
REPO = Path('/home/minjae/Documents/pallet-visibility-publication-20261024')
PREFIX = '_docs/experiments/pallet-keypoint-visibility/'
DELTA = 'publication/deltas/V187_20261024_v1/'
PAYLOAD = '30478a5ce93243fe297aeecb6337bba2b4276098'


def read_bytes(path):
    assert path.is_absolute() and '..' not in path.parts
    assert all('v112' not in part.casefold() for part in path.parts)
    fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:-1]:
            new = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = new
        final = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
    finally:
        os.close(fd)
    assert stat.S_ISREG(os.fstat(final).st_mode)
    with os.fdopen(final, 'rb') as f:
        return f.read()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def bind(path):
    return {'path': str(path), 'sha256': sha(read_bytes(path))}


def new_file(path, data):
    assert '..' not in path.parts and all('v112' not in s.casefold() for s in path.parts)
    current = Path(path.anchor)
    for part in path.parts[1:-1]:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            current.mkdir()
            mode = current.lstat().st_mode
        assert stat.S_ISDIR(mode)
    with path.open('xb') as f:
        f.write(data)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()


def main():
    state = json.loads(read_bytes(ADMIN / 'V187_DELTA_PUSH_v1/STATE.json'))
    review_path = REMOTE / 'actual_remote_metadata_review_v1/REVIEW.json'
    review = json.loads(read_bytes(review_path))
    build = json.loads(read_bytes(ADMIN / 'V187_DELTA_BUILD_v1/BUILD_COMPLETE.json'))
    assert state['completed_all'] and state['pending'] is None and state['expected_remote'] == PAYLOAD
    assert review['completed'] and review['PASS'] and review['public'] and review['default_branch'] == 'main'
    assert review['expected_payload_commit'] == review['remote_main_before'] == review['remote_main_after'] == PAYLOAD
    assert review['selected_files_verified'] == 5
    assert review['remote_content_raw_blob_contents_base64_or_pixel_reads'] == 0
    assert sha(read_bytes(Path(build['base_manifest']['path']))) == build['base_manifest']['sha256']
    assert sha(read_bytes(Path(build['delta_manifest']['path']))) == build['delta_manifest']['sha256']
    explicit = [
        (ADMIN / 'ROOT_V187_DELTA_PUSH_PLAN.json', 'ROOT_VERIFIED_PAYLOAD_PUSH_PLAN.json'),
        (ADMIN / 'V187_DELTA_PUSH_v1/STATE.json', 'PAYLOAD_PUSH_STATE.json'),
        (ADMIN / 'ROOT_V187_SOURCE_FREEZE_BEFORE_INVENTORY.json', 'ROOT_SOURCE_FREEZE.json'),
        (ADMIN / 'ROOT_V187_ACTUAL_DELTA_INVENTORY_ENTRY.json', 'ROOT_ACTUAL_INVENTORY_ENTRY.json'),
        (ADMIN / 'V187_DELTA_INVENTORY_v1/INVENTORY_COMPLETE.json', 'INVENTORY_COMPLETE.json'),
        (ADMIN / 'V187_DELTA_INVENTORY_v1/V187_DELTA_CANDIDATE_MANIFEST.jsonl', 'CANDIDATE_MANIFEST.jsonl'),
        (ADMIN / 'V187_DELTA_INVENTORY_v1/DIRECTORY_PRUNE_RECORDS.json', 'DIRECTORY_PRUNE_RECORDS.json'),
        (ADMIN / 'ROOT_V187_ACTUAL_INDEPENDENT_DELTA_AUDIT_ENTRY_V2_DIRECTORY_LINKS.json', 'ROOT_ACTUAL_BYTE_AUDIT_ENTRY.json'),
        (ADMIN / 'ROOT_V187_ACTUAL_VERIFIED_DELTA_PUSH_PLAN_ENTRY.json', 'ROOT_ACTUAL_PUSH_PLAN_ENTRY.json'),
        (review_path, 'REMOTE_PUBLICATION_REVIEW_B.json'),
        (REMOTE / 'verify_remote_publication_metadata.py', 'verify_remote_publication_metadata.py'),
        (REMOTE / 'PROTOCOL_BEFORE_REMOTE_API_OR_PUBLISHED_FILE_READS.json', 'REMOTE_METADATA_PROTOCOL.json'),
        (REMOTE / 'ROOT_SCOPE_BEFORE_ACTUAL_REMOTE_API_OR_PUBLISHED_FILE_BYTES.json', 'ROOT_REMOTE_SCOPE.json'),
        (REMOTE / 'ROOT_ACTUAL_REMOTE_METADATA_ENTRY.json', 'ROOT_ACTUAL_REMOTE_ENTRY.json'),
        (ADMIN / 'V187_CLOSING_STATIC_REVIEW_A.json', 'CLOSING_STATIC_REVIEW_A.json'),
        (Path(__file__).absolute(), 'record_v187_closing_publication.py'),
    ]
    files = []
    copies = []
    for source, name in explicit:
        data = read_bytes(source)
        destination = PREFIX + DELTA + 'closing/' + name
        new_file(REPO / destination, data)
        files.append({'path': destination, 'size_bytes': len(data), 'sha256': sha(data)})
        copies.append({'source': str(source), 'published_relative_to_repository': destination,
                       'sha256': sha(data), 'size_bytes': len(data)})
    receipt = {
        'completed': True, 'PASS': True, 'recorded_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'repository': 'https://github.com/CanelE452/pallet-6d-pose', 'branch': 'main', 'public': True,
        'verified_payload_commit': PAYLOAD, 'payload_physical_files': 305,
        'new_original_source_files': 287, 'new_original_source_bytes': 52995055,
        'independent_byte_review': bind(ADMIN / 'V187_DELTA_INDEPENDENT_REVIEW_A_V2_DIRECTORY_LINKS/REVIEW.json'),
        'remote_metadata_review': bind(review_path), 'immutable_base_manifest': build['base_manifest'],
        'immutable_delta_manifest': build['delta_manifest'], 'manifest_chain': build['new_chain'],
        'portable_index': PREFIX + 'EXPERIMENT_INDEX_THROUGH_V187_KO.md',
        'portable_report': PREFIX + 'archive/task-2/pallet-keypoint-visibility/query_visibility_source_selected_part_correspondence_20261024_v187/PORTABLE_REPORT_KO.md',
        'verification': {'all_new_original_byte_roundtrip': True, 'all_physical_payload_SHA_and_Git_blob_identity': True,
                         'portable_pages': 3, 'portable_local_links': 320, 'credential_format_detections': 0,
                         'public_remote_selected_blob_metadata_matches': 5, 'old_manifest_covered_paths_overwritten': False},
        'copies': copies, 'ongoing_main_push_authorization': True,
        'policy': 'Publish each completed attempt with source, definitions, results, failures, visuals, tables and independent verification; preserve original history and declared exclusions.',
        'model_adoption_or_full_goal_success': False, 'scope': 'Source correspondence diagnostic publication only.',
        'receipt_cycle_rule': 'This receipt binds the verified payload commit. This receipt and closing copies are subsequently bound by a separate exact Git push plan; no self-referential own-commit hash.',
    }
    receipt_data = encoded(receipt)
    path = PREFIX + DELTA + 'MAIN_PUBLICATION_RECEIPT.json'
    new_file(REPO / path, receipt_data)
    files.append({'path': path, 'size_bytes': len(receipt_data), 'sha256': sha(receipt_data)})
    note = '''# V187 main 게시 확인

V187의 원본 연구 기록 287개(52,995,055바이트), 코드·실패 기록·표·원본/표시 crop과 검증 기록을 main에 추가했다. 기존 V186까지의 archive와 manifest는 유지했다. 이번 기록은 공개 TRAIN의 선택 물체·명목 부위 대응 진단이며, 새 모델 성능 개선이나 전체 가림 목표 해결을 뜻하지 않는다.

- [최신 전체 실험 인덱스](../../../EXPERIMENT_INDEX_THROUGH_V187_KO.md)
- [V187 보고서·표·그림](../../../archive/task-2/pallet-keypoint-visibility/query_visibility_source_selected_part_correspondence_20261024_v187/PORTABLE_REPORT_KO.md)
- [기존 전체 검토 이미지 목록](../../../REVIEW_IMAGE_CATALOG_KO.md)
- [게시 영수증 원문](MAIN_PUBLICATION_RECEIPT.json)
- [독립 원본 복원·해시·내부 링크 검산](verification/REVIEW_00.json)
- [독립 공개 저장소·원격 commit·5파일 blob 검산](closing/REMOTE_PUBLICATION_REVIEW_B.json)
- [정확한 payload push 파일 목록](closing/ROOT_VERIFIED_PAYLOAD_PUSH_PLAN.json)

원본 raw/gzip 복원·SHA, 새 물리 파일의 Git blob 동일성, 내부 링크 320개, 공개 main의 지정 5파일 메타데이터가 일치했다. 원격에서 콘텐츠나 픽셀을 다시 읽지는 않았다. 대용량 원본 입력 은행·환경·공통 기반 모델·금지 경로는 기존 제외 정책을 유지한다. 앞으로도 완료한 시도별 방법·결과·실패·시각화·검증을 main에 추가한다.

검증된 payload commit: [30478a5](https://github.com/CanelE452/pallet-6d-pose/commit/30478a5ce93243fe297aeecb6337bba2b4276098). 이 문서와 closing 증빙은 별도 후속 commit에 들어가며, 영수증은 자기 commit을 순환 참조하지 않는다.
'''
    data = note.encode()
    path = PREFIX + DELTA + 'PUBLICATION_RECEIPT_KO.md'
    new_file(REPO / path, data)
    files.append({'path': path, 'size_bytes': len(data), 'sha256': sha(data)})
    assert len({f['path'] for f in files}) == len(files)
    assert sum(f['size_bytes'] for f in files) < 480 * 1024**2
    assert all(f['size_bytes'] < 100 * 1024**2 for f in files)
    plan = {'authorized_main_push': True, 'repository': str(REPO),
            'remote': 'https://github.com/CanelE452/pallet-6d-pose.git', 'branch': 'main',
            'initial_remote_main': PAYLOAD, 'batch_max_bytes': 480 * 1024**2,
            'batches': [{'files': files, 'message': 'Record V187 verified public main receipt and independent remote proof'}],
            'scope': 'Fresh V187 closing verification records only, no payload or old manifest mutation.'}
    new_file(ADMIN / 'ROOT_V187_CLOSING_PUSH_PLAN.json', encoded(plan))
    new_file(P / 'MAIN_PUBLICATION_RECEIPT_V187_20261024.json', receipt_data)
    print(json.dumps({'completed': True, 'PASS': True, 'files': len(files),
                      'bytes': sum(f['size_bytes'] for f in files),
                      'closing_plan': bind(ADMIN / 'ROOT_V187_CLOSING_PUSH_PLAN.json')}))


if __name__ == '__main__':
    main()
