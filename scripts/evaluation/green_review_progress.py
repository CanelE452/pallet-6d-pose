"""Live, separate review counts; never promotes data to a paper evaluation set."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import time

from scripts.evaluation.eval_workspace import atomic_write_text

BEGIN = '<!-- GREEN_REVIEW_PROGRESS_BEGIN -->'
END = '<!-- GREEN_REVIEW_PROGRESS_END -->'


def append_review_section(report: str, section: str) -> str:
    if BEGIN in report and END in report:
        before, rest = report.split(BEGIN, 1)
        _, after = rest.split(END, 1)
        return before.rstrip() + '\n\n' + section.rstrip() + after
    return report.rstrip() + '\n\n' + section.rstrip() + '\n'


def collect(review: Path, repo: Path) -> dict:
    manifest = json.loads((review / 'full_sessions_manifest.json').read_text())
    rows = []
    errors = []
    for session in manifest['records']:
        base = review / 'full_session_annotations' / (session['session'] + '_manual_gt')
        original = {r['name'] for r in session['source_annotations']}
        counts = Counter()
        for path in sorted(base.glob('*.json')):
            try:
                doc = json.loads(path.read_text())
                obj = doc['objects'][0]
                if obj.get('object_type', doc.get('object_type')) != 'plastic_standard_110x110x15':
                    raise ValueError('unexpected object type')
                if len(obj.get('keypoint_annotations', [])) != 9:
                    raise ValueError('missing 9-keypoint annotation')
                rgb = repo / session['source'] / 'rgb'
                if not any((rgb / (path.stem + suffix)).is_file() for suffix in ('.png', '.jpg', '.jpeg')):
                    raise ValueError('missing raw image')
                split = obj.get('split', 'unknown')
                if split not in {'train', 'eval'}:
                    raise ValueError('unknown split')
            except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
                errors.append(f'{session["session"]}/{path.name}: {exc}')
                continue
            counts['annotated'] += 1
            counts[split] += 1
            counts['new'] += path.name not in original
        rows.append(dict(session=session['session'], frames=session['frame_count'],
                         original=len(original), **{k: counts[k] for k in ('annotated', 'train', 'eval', 'new')}))
    totals = {k: sum(r[k] for r in rows) for k in ('frames', 'original', 'annotated', 'train', 'eval', 'new')}
    return dict(rows=rows, totals=totals, errors=errors, paper_admitted=0)


def render(data: dict) -> str:
    t = data['totals']
    lines = [BEGIN, '## 초록 정사각형 팔레트 — 추가 검토 진행률 (자동)', '',
             '**아래는 별도 검토 폴더 집계이며 위 논문 평가셋 합계에는 포함하지 않습니다.**', '',
             '| 항목 | 수 |', '|---|---:|',
             f'| 전체 촬영 프레임 | {t["frames"]} |',
             f'| 최초 복사한 기존 라벨 | {t["original"]} |',
             f'| 현재 유효 라벨 | {t["annotated"]} |',
             f'| 새로 라벨링한 프레임 | {t["new"]} |',
             f'| EVAL로 지정·저장한 검토 후보 | **{t["eval"]}** |',
             f'| TRAIN으로 남아 있는 라벨 | {t["train"]} |',
             f'| 미어노테이션 프레임 | {t["frames"] - t["annotated"]} |',
             f'| 읽기/구조 오류 (집계 제외) | {len(data["errors"])} |', '',
             '원본 851장은 모두 TRAIN이었다. EVAL 수는 검토 복사본의 저장된 split을 센다.',
             '새 프레임은 기본 EVAL이며 기존 TRAIN은 v로 바꾼 뒤 s로 저장한다.',
             '단순 열람·미저장 변경은 집계하지 않는다. EVAL 지정은 라벨 품질 검증이나',
             '독립 테스트 적격성 확인을 뜻하지 않으며, 정식 평가 편입은 별도 절차다.',
             '100장은 임의의 확정 목표로 추가하지 않았다. 촬영 세션별 프레임 수이며 SHA 중복 제거 전 수다.', '',
             '| 촬영 세션 | 전체 | 라벨 | 신규 | EVAL |', '|---|---:|---:|---:|---:|']
    lines += [f'| {r["session"]} | {r["frames"]} | {r["annotated"]} | {r["new"]} | {r["eval"]} |' for r in data['rows']]
    if data['errors']:
        lines += ['', '집계 오류:'] + [f'- {e}' for e in data['errors']]
    return '\n'.join(lines + ['', END, ''])


def update(review: Path, workspace: Path, repo: Path) -> dict:
    data = collect(review, repo)
    section = render(data)
    reports = workspace / 'reports'
    targets = {
        reports / 'GREEN_REVIEW_PROGRESS.md': section,
        review / 'ANNOTATION_PROGRESS.md': section,
        review / 'progress.json': json.dumps(data, ensure_ascii=False, indent=2) + '\n',
    }
    main = reports / 'ANNOTATION_PROGRESS.md'
    if main.exists():
        targets[main] = append_review_section(main.read_text(), section)
    for path, content in targets.items():
        if not path.exists() or path.read_text() != content:
            atomic_write_text(path, content)
    return data


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-root', type=Path, default=repo / 'outputs/green_paper_review_20260918')
    parser.add_argument('--eval-root', type=Path, default=repo / 'data/evaluation/pallet_eval_v1')
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    # One watcher per review folder, including when the GUI is reopened.
    with (args.review_root / '.progress.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Progress watcher already running.', flush=True)
            return
        previous = None
        while True:
            try:
                data = update(args.review_root, args.eval_root, repo)
                summary = (data['totals'], data['errors'])
                if summary != previous:
                    print(json.dumps(summary, ensure_ascii=False), flush=True)
                    previous = summary
            except (OSError, ValueError, KeyError) as exc:
                print(f'[WARN] progress refresh: {exc}', flush=True)
                if not args.watch:
                    raise
            if not args.watch:
                return
            time.sleep(3)


if __name__ == '__main__':
    main()
