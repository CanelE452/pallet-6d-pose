"""Read-only, named-source inventory. No RGB decoding, learning or model inference."""
from pathlib import Path
import collections
import csv
import datetime
import hashlib
import html
import io
import json
import zipfile

ROOT = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
OUT = ROOT / 'publication_root_preparation/V263_ROOT_SPATIAL_LOCALIZATION'
REPO = Path('/home/minjae/Documents/github/pallet-pose')
DOWNLOADS = Path('/home/minjae/Downloads')

def safe(path):
    p = Path(path)
    if '..' in p.parts or any(t in str(p).lower() for t in ('v112', 'cache', '.pyc')):
        raise ValueError(str(p))
    for a in reversed([p, *p.parents]):
        if a.is_symlink():
            raise ValueError('symlink: ' + str(a))
    return p

def binding(path):
    p = safe(path)
    b = p.read_bytes()
    return {'path': str(p), 'sha256': hashlib.sha256(b).hexdigest(), 'size_bytes': len(b)}

def read_json(path):
    return json.loads(safe(path).read_bytes())

def counts(values):
    return dict(sorted(collections.Counter(values).items()))

def zip_image_hashes(z):
    result = []
    for info in z.infolist():
        if not info.is_dir() and Path(info.filename).suffix.lower() in ('.png', '.jpg', '.jpeg'):
            # Hash encoded file bytes only; never decode or display candidate pixels.
            with z.open(info) as f:
                h = hashlib.sha256()
                for b in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(b)
            result.append({'member': info.filename, 'sha256': h.hexdigest()})
    return result

def main():
    safe(OUT)
    manifest_path = ROOT / 'publication_root_preparation/V229_REAL_QA_B/current223_input_metadata_v1/INPUT_MANIFEST_CURRENT223_ONLY.json'
    current = read_json(manifest_path)
    current_sha = {f['source_fullRGB']['binding']['sha256'] for f in current['frames223']}
    audit_path = REPO / '_docs/experiments/pallet_combined_closeout_20261003_v1/closeout_20261006_v1/visibility_square/STATIC_VISIBILITY_MERGE_AUDIT.json'
    audit = read_json(audit_path)
    frame_csv = REPO / 'data/evaluation/pallet_eval_v1/manifests/ALL_AVAILABLE_POSITIVE.csv'
    with safe(frame_csv).open(newline='') as f:
        registered = list(csv.DictReader(f))
    registered_by_sha = {r['image_sha256']: r for r in registered}
    grouped = collections.defaultdict(list)
    for row in audit['rows']:
        grouped[(row['case_id'], row['image_sha256'])].append(row)
    extra = []
    for (case, sha), rows in grouped.items():
        if sha not in current_sha:
            reg = registered_by_sha.get(sha, {})
            extra.append({'case_id': case, 'image_sha256': sha,
                          'population': rows[0]['population'],
                          'object_type': reg.get('object_type', 'square_GREEN0918' if rows[0]['population'] == 'GREEN0918' else 'UNKNOWN'),
                          'registered_role': reg.get('population_role', 'UNKNOWN'),
                          'session': reg.get('session_id', 'GREEN0918'),
                          'categories': counts(r['category'] for r in rows)})
    green_path = DOWNLOADS / 'green_pallet_real_raw_json_851_20260917.zip'
    with zipfile.ZipFile(safe(green_path)) as z:
        manifest_name = next(i.filename for i in z.infolist() if i.filename.endswith('/manifest.json'))
        green = json.loads(z.read(manifest_name))
        green_readme = z.read(next(i.filename for i in z.infolist() if i.filename.endswith('/README.txt'))).decode('utf8')
    green_sha = {r['image_sha256'] for r in green}
    archive_rows = []
    archive_hashes = {}
    for name in ['kimjihoon_9-4_sampled_manual_gt.zip', 'kimjihoon_manual_gt.zip',
                 'forklift_20260903_every_10_frames_manual_gt.zip']:
        path = safe(DOWNLOADS / name)
        annotation_counts = collections.Counter()
        annotation_splits = collections.Counter()
        with zipfile.ZipFile(path) as z:
            images = zip_image_hashes(z)
            annotation_n = 0
            for info in z.infolist():
                if info.filename.endswith('.json') and not info.is_dir():
                    d = json.loads(z.read(info))
                    if not isinstance(d, dict) or 'objects' not in d:
                        continue
                    annotation_n += 1
                    for obj in d.get('objects', []):
                        annotation_splits[obj.get('split', 'UNKNOWN')] += 1
                        for kp in obj.get('keypoint_annotations', [])[:8]:
                            annotation_counts[str(kp.get('source', 'UNKNOWN')) + '/' + str(kp.get('reason', 'UNKNOWN'))] += 1
        hs = {r['sha256'] for r in images}
        archive_hashes[name] = images
        archive_rows.append({'path': str(path), 'images': len(images), 'unique_encoded_sha': len(hs),
                             'annotation_json': annotation_n, 'overlap_current223': len(hs & current_sha),
                             'overlap_green851': len(hs & green_sha),
                             'object_split_counts': dict(annotation_splits), 'native8_source_reason_counts': dict(annotation_counts),
                             'independent_visibility_evaluation_ready': False,
                             'reason': 'Raw keypoint visible/unknown fields do not distinguish external from self occlusion. Encoded-image overlap and prior training must be preserved.'})
    raw_path = safe(DOWNLOADS / '김민재_9월6일.zip')
    with zipfile.ZipFile(raw_path) as z:
        raw_readme = z.read('README.txt').decode('utf8')
        source_summary = json.loads(z.read('SOURCE_SUMMARY.json'))
        images = zip_image_hashes(z)
        raw_hash = {r['sha256'] for r in images}
        json_names = [i.filename for i in z.infolist() if i.filename.endswith('.json')]
    archive_hashes[raw_path.name] = images
    result = {
        'schema': 'V263_ADDITIONAL_REAL_DATA_NAMED_SOURCE_INVENTORY',
        'created_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'request': '사용자: 다른게 있는지 일단 확인해봐 (223장 외 로컬 자료 존재 및 사용 가능성 확인)',
        'scope': 'Named metadata paths and archive central directories/JSON/README; candidate image encoded SHA only. Not an exhaustive system search.',
        'rgb_decodes': 0, 'model_forwards': 0, 'new_training_or_selection': 0,
        'source_bindings': [binding(manifest_path), binding(audit_path), binding(frame_csv)],
        'current223_unique_encoded_sha': len(current_sha),
        'existing_human_visibility': {
            'all_frames': len(grouped), 'outside_current223_frames': len(extra),
            'outside_by_type': counts(r['object_type'] for r in extra),
            'outside_by_session': counts(r['session'] for r in extra),
            'outside_rows': extra,
            'status': 'Previously registered DEV319/GREEN0918 human sidecar; extra rows are historical development material, not newly collected independent validation.'},
        'green851': {'path': str(green_path), 'images': len(green), 'unique_encoded_sha': len(green_sha),
                     'sessions': counts(r['session'] for r in green), 'overlap_current223': len(green_sha & current_sha),
                     'prior_training_explicit_in_README': True, 'README': green_readme,
                     'status': 'Explicitly 851 original pairs used in augmentation training. Not an unseen held-out set.'},
        'manual_annotation_archives': archive_rows,
        'raw_20260906': {'path': str(raw_path), 'images': len(images), 'unique_encoded_sha': len(raw_hash),
                         'source_videos': len(source_summary), 'overlap_current223': len(raw_hash & current_sha),
                         'overlap_green851': len(raw_hash & green_sha), 'JSON_members': json_names,
                         'README': raw_readme, 'source_summary': source_summary,
                         'visibility_annotation_present': False, 'prior_experiment_exposure': 'UNKNOWN',
                         'status': 'Additional real image candidate pool; no native D/E visibility gold in this archive, and no proof of untouched capture history.'},
        'conclusion': '223장 외 실사는 존재. 바로 독립 가림 검증셋으로 쓸 수 있다고 인증된 추가 자료는 아직 없음. 합성 비교와 기존223 개발평가를 유지하며 추가 자료는 별도 조사 목록으로 보존.',
        'limitations': ['Exact encoded SHA equality does not catch re-encoding or adjacent video frames.',
                        'No claim that all filesystem locations or historical experiments were searched.',
                        'No data was admitted to training/evaluation or relabeled by this inventory.']
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    (OUT / 'ADDITIONAL_REAL_DATA_INVENTORY.json').write_text(payload)
    (OUT / 'ADDITIONAL_REAL_ENCODED_SHA_LEDGER.json').write_text(json.dumps(archive_hashes, ensure_ascii=False, indent=2) + '\n')
    cells = [
        ('기존 가시성 주석의 추가 프레임', str(len(extra)), '목재20·기존 플라스틱194·정사각형1', '과거 개발 자료. 독립 검증 아님'),
        ('green851', str(len(green)), '28세션·수동 키포인트', 'README에 기존 증강 학습 사용 명시'),
    ]
    for r in archive_rows:
        cells.append((Path(r['path']).name, str(r['images']), '851묶음과 동일 이미지 '+str(r['overlap_green851'])+'장', '외부/자기 가림 정답 분리 안 됨'))
    cells.append(('2026-09-06 원본 추출 ZIP', str(len(images)), str(len(source_summary))+'개 영상에서 추출', '가시성 정답 없음·과거 노출 미확인'))
    table = ''.join('<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in row)+'</tr>' for row in cells)
    page = '<!doctype html><html lang="ko"><meta charset="utf-8"><title>223장 외 실사 자료 확인</title><style>body{font:16px sans-serif;max-width:1100px;margin:40px auto;line-height:1.6}table{border-collapse:collapse;width:100%}th,td{border:1px solid #bbb;padding:12px;text-align:left}th{background:#eee}</style><h1>223장 외 실사 자료 확인</h1><p>'+html.escape(result['conclusion'])+'</p><table><tr><th>자료</th><th>프레임</th><th>확인 내용</th><th>현재 상태</th></tr>'+table+'</table><p>장수는 서로 중복될 수 있어 합산하면 안 됩니다. RGB 디코딩·모델 추론·학습·자료 편입은 하지 않았습니다. 상세 경로·SHA·한계는 <a href="ADDITIONAL_REAL_DATA_INVENTORY.json">JSON</a>에 있습니다.</p></html>'
    (OUT / 'ADDITIONAL_REAL_DATA_READOUT_KO.html').write_text(page)
    print(json.dumps({'output': str(OUT / 'ADDITIONAL_REAL_DATA_INVENTORY.json'),
                      'outside_visibility': len(extra), 'outside_by_type': result['existing_human_visibility']['outside_by_type'],
                      'manual_archive_overlap': archive_rows,
                      'raw1505': {k: result['raw_20260906'][k] for k in ['images','overlap_current223','overlap_green851','source_videos']},
                      'green851_overlap_current': result['green851']['overlap_current223']}, ensure_ascii=False))

if __name__ == '__main__':
    main()
