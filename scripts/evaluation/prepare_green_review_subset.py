"""Build a prediction-blind REVIEW proposal, never edit splits or freeze evaluation."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / 'outputs/green_paper_review_20260918'
OUT = REVIEW / 'REVIEW_150_PROPOSAL.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evenly_spaced(rows, count):
    """One midpoint per equal-count temporal bin; deterministic, without replacement."""
    rows = sorted(rows, key=lambda r: (int(Path(r['image']).stem), r['id']))
    if count < 0 or count > len(rows):
        raise ValueError('Insufficient eligible frames; do not silently refill from other conditions')
    return [rows[((2 * k + 1) * len(rows)) // (2 * count)] for k in range(count)]


def stratified_sample(rows, count):
    """Proportional folder quotas (not an assertion that folders are independent)."""
    groups = defaultdict(list)
    for row in rows:
        groups[row['session']].append(row)
    if count > len(rows) or count < 0:
        raise ValueError('Insufficient eligible frames')
    if not count:
        return []
    quotas = {name: count * len(items) // len(rows) for name, items in groups.items()}
    remainders = sorted(groups, key=lambda name: (-(count * len(groups[name]) % len(rows)), name))
    for name in remainders[:count - sum(quotas.values())]:
        quotas[name] += 1
    return [r for name in sorted(groups) for r in evenly_spaced(groups[name], quotas[name])]


def select(rows):
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate IDs in inventory')
    eligible = [r for r in rows if r['split'] == 'eval']
    trunc = [r for r in eligible if 'truncation' in r['groups']]
    handheld = [r for r in eligible if 'handheld_multiview' in r['groups'] and 'truncation' not in r['groups']]
    evening = [r for r in eligible if 'evening_capture_candidate' in r['groups'] and 'truncation' not in r['groups']]
    dusk = [r for r in evening if 'darker_dusk_candidate' in r['groups']]
    bright = [r for r in evening if 'darker_dusk_candidate' not in r['groups']]
    if len(trunc) != 47 or len(dusk) != 7:
        raise ValueError('47 truncation / 7 dusk proposal no longer matches saved labels; review the changed scope')
    selected = []
    for bucket, items in (
        ('truncation', trunc),
        ('nontruncated_handheld', stratified_sample(handheld, 73)),
        ('evening', dusk + stratified_sample(bright, 23)),
    ):
        selected.extend(dict(r, selection_bucket=bucket) for r in items)
    if len(selected) != 150 or len({r['id'] for r in selected}) != 150:
        raise ValueError('Expected 150 non-overlapping IDs')
    if len({r['image_sha256'] for r in selected}) != 150:
        raise ValueError('Duplicate image content; requires review, no silent replacement')
    return sorted(selected, key=lambda r: (r['selection_bucket'], r['session'], r['id']))


def build():
    source = REVIEW / 'CONDITION_CANDIDATES.json'
    inventory = json.loads(source.read_text())
    # Guard against concurrent saves; no label writes anywhere in this program.
    before = {p: sha(p) for p in (REVIEW / 'full_session_annotations').glob('*/*.json')}
    rows = []
    for old in inventory['records']:
        row = dict(old)
        ann = ROOT / row['annotation']
        obj = json.loads(ann.read_text())['objects'][0]
        row['split'] = obj['split']
        row['groups'] = [g for g in row['groups'] if g != 'truncation']
        if obj.get('truncation', {}).get('is_truncated') is True:
            row['groups'].append('truncation')
        row['annotation_sha256'] = before[ann]
        if sha(ROOT / row['image']) != row['image_sha256']:
            raise ValueError('Source image changed: ' + row['id'])
        rows.append(row)
    selected = select(rows)
    sizes = Counter()
    for row in selected:
        doc = json.loads((ROOT / row['annotation']).read_text())
        camera = doc['camera_data']
        sizes[f"{camera['width']}x{camera['height']}"] += 1
    if any(sha(p) != digest for p, digest in before.items()):
        raise ValueError('Annotation changed during proposal generation; rerun after saving')
    result = dict(
        status='PROPOSED_150_REQUIRES_HUMAN_REVIEW_NOT_FINAL_EVAL',
        union_frames=len(selected), records=selected,
        selection_method='All 47 truncation; proportional-session midpoint temporal sampling of 73 nontruncated handheld and 23 brighter evening; all 7 darker dusk',
        prediction_accessed=False, source_labels_modified=False, final_membership_frozen=False,
        selection_not_power_calculation=True,
        source_inventory=dict(path=str(source.relative_to(ROOT)), sha256=sha(source)),
        bucket_counts=dict(Counter(r['selection_bucket'] for r in selected)),
        session_counts=dict(Counter(r['session'] for r in selected)),
        overlapping_condition_counts=dict(Counter(g for r in selected for g in r['groups'])),
        image_dimensions=sizes,
        caveats=[
            'Temporal spacing reduces adjacency but is not verified visual deduplication or guaranteed viewpoint diversity.',
            'Shared capture folders are not independent sessions; handheld folders form one continuous recording.',
            'All truncation frames are retained by request, including potentially near-identical frames.',
            'Prior training/development overlap must be audited before any independent-test claim.',
            'Current EVAL flags are not evidence of completed human review.',
            'Nonselected images are not changed to TRAIN and are not authorized for new training.',
        ],
    )
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if OUT.exists() and OUT.read_text() != text:
        raise ValueError('Existing proposal differs: preserve it and use a new proposal version')
    if not OUT.exists():
        OUT.write_text(text)
    lines = ['# 150장 검토 제안 — 최종 평가셋 아님', '',
             '모델 예측·오차를 사용하지 않았으며 라벨과 split은 변경하지 않았다.',
             '잘림 47 + 잘리지 않은 다각도 73 + 저녁 30(더 어두운 후보 7 포함).',
             '나머지는 세션별 비례 배분 후 프레임 순서의 균등 구간에서 한 장씩 선택했다.',
             '시점 다양성·유사 프레임·라벨 정확도는 사람이 검수해야 한다. 150은 검수 예산이지 통계적 충분성 보장이 아니다.', '',
             '검토 창: `python outputs/green_paper_review_20260918/open_condition_review.py --manifest REVIEW_150_PROPOSAL.json --start-session forklift_v4_20260903_192254`', '',
             'Tab/SESSION으로 세션 이동, n/p로 사진 이동. 제외 시 v → TRAIN → s 저장.',
             '제외한 사진도 검토 창에 남는다. 새 사진 대체는 모델 결과를 보기 전에 결정한다.',
             '승인 전 freeze/infer/score 금지. 기존 freeze는 전체 EVAL을 읽으므로 이 목록 전용 실행 경로를 준비해야 한다.', '',
             '| 세션 | 후보 수 |', '|---|---:|']
    lines += [f'| {name} | {n} |' for name, n in sorted(result['session_counts'].items())]
    lines += ['', '| ID | 선택 구분 | 이미지 |', '|---|---|---|']
    lines += [f"| {r['id']} | {r['selection_bucket']} | [보기]({ROOT / r['image']}) |" for r in selected]
    (REVIEW / 'REVIEW_150_PROPOSAL.md').write_text('\n'.join(lines) + '\n')
    cards = []
    labels = {'truncation': '잘림', 'nontruncated_handheld': '잘리지 않은 다각도', 'evening': '저녁·해질녘'}
    for bucket, label in labels.items():
        cards.append(f'<h2>{label}</h2><div class="grid">')
        for row in selected:
            if row['selection_bucket'] != bucket:
                continue
            uri = html.escape((ROOT / row['image']).as_uri(), quote=True)
            cards.append(f'<figure><a href="{uri}"><img loading="lazy" src="{uri}"></a><figcaption>{html.escape(row["id"])}</figcaption></figure>')
        cards.append('</div>')
    (REVIEW / 'REVIEW_150_GALLERY.html').write_text(
        '<!doctype html><html lang="ko"><meta charset="utf-8"><title>150장 평가 후보 검토</title>'
        '<style>body{font-family:sans-serif;margin:24px;background:#eee}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:12px}figure{margin:0;background:white;padding:8px}img{width:100%}figcaption{overflow-wrap:anywhere}</style>'
        '<h1>150장 후보 — 최종 평가셋 아님</h1><p>원본 사진 모아보기. 이 페이지는 읽기 전용이며 split 변경은 어노테이션 창에서 저장하세요.</p>'
        + ''.join(cards) + '</html>')
    print(json.dumps({k: result[k] for k in ('status', 'union_frames', 'bucket_counts', 'session_counts', 'image_dimensions')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    build()
