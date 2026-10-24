"""Publish a completed, independently checked V221 review to authorized main."""
import hashlib
import json
import os
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

P = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
R = P / 'publication_root_preparation'
PHASE = 'query_visibility_review_reconciliation_source_history_20261025_v221'
S = P / PHASE
REPO = Path('/home/minjae/Documents/pallet-visibility-publication-20261024')
DOCREL = Path('_docs/experiments/pallet-keypoint-visibility')
ARC = DOCREL / 'archive/task-2/pallet-keypoint-visibility' / PHASE
BASE = '59cf1e512fc8084c66f1eedc5195b4434ab9e28d'


def checked(path):
    path = Path(path)
    if 'v112' in str(path).lower():
        raise ValueError('forbidden path rejected before access')
    for item in (*reversed(path.parents), path):
        if item.is_symlink():
            raise ValueError('symlink rejected')
    return path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.run(['git', *args], cwd=REPO, check=True, text=True,
                          capture_output=True).stdout.strip()


def record(path):
    data = checked(path).read_bytes()
    return {'path': str(path.relative_to(S)), 'sha256': digest(data), 'bytes': len(data)}


def own_files():
    files = []
    for root, dirs, names in os.walk(checked(S), followlinks=False):
        for name in dirs:
            checked(Path(root) / name)
        for name in names:
            f = checked(Path(root) / name)
            if f.suffix not in {'.json', '.md', '.csv', '.py', '.png'}:
                raise ValueError('unexpected new-phase file type')
            files.append(f)
    return sorted(files)


def dump(path, value):
    checked(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    assert git('rev-parse', 'HEAD') == BASE
    assert not git('status', '--short', '--untracked-files=no')
    assert not git('diff', '--cached', '--name-only')
    git('fetch', 'origin', 'main')
    assert git('rev-parse', 'FETCH_HEAD') == BASE, 'remote changed; stop before mutation'
    qa_dir = checked(R / 'V221_ROOT_RECONCILIATION_QA_B')
    qa = json.loads(checked(qa_dir / 'QA.json').read_text())
    # Require a positive independent result, with no string guessing of numeric metrics.
    assert qa.get('completed') is True
    assert qa.get('PASS') is True
    assert qa['inputs']['report']['sha256'] == digest(checked(S / 'PORTABLE_REPORT_KO.md').read_bytes())
    assert digest(checked(S / 'PORTABLE_REPORT_KO.md').read_bytes()) == 'f9d461ddbdc32ea1247da88d186c5868e186dbfae4b85a59d7daaa0399624c79'
    assert not checked(S / 'ROOT_PHASE_FREEZE.json').exists()
    checked(S / 'independent_root_qa').mkdir(exist_ok=False)
    for name in ['QA.json', 'KO.md']:
        shutil.copyfile(checked(qa_dir / name), checked(S / 'independent_root_qa' / name))
    shutil.copyfile(checked(__file__), checked(S / 'review_preparation/publish_review.py'))
    original_bindings = json.loads(checked(S / 'ROOT_REVIEW_BINDINGS.json').read_text())
    for item in original_bindings['archived_originals']:
        original = checked(item['original_path']).read_bytes()
        archived = checked(S / item['archive_path']).read_bytes()
        assert digest(original) == item['sha256'] == digest(archived)
    before = [record(f) for f in own_files()]
    dump(S / 'ROOT_PHASE_FREEZE.json', {
        'completed': True, 'status': 'PASS', 'phase': 'V221',
        'recorded_UTC': datetime.now(timezone.utc).isoformat(),
        'kind': 'REVIEW_AND_SOURCE_METADATA_AUDIT_NOT_NEW_MODEL_RESULT',
        'files_except_this_freeze': before, 'independent_qa_sha256': digest(checked(qa_dir / 'QA.json').read_bytes()),
        'new_training_updates': 0, 'new_model_forwards': 0, 'new_real_benchmark': False,
        'raw_evaluation_reads': 0, 'new_PnP': 0, 'task_solved': False, 'selected_model': None,
    })
    source = [record(f) for f in own_files()]
    assert not checked(REPO / ARC).exists()
    for item in source:
        origin = checked(S / item['path'])
        target = checked(REPO / ARC / item['path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, target)
        assert digest(target.read_bytes()) == item['sha256']
    headline = ('# V221 첨부 V193 검토 대조·SOURCE 이력 감사\n\n'
                '새 모델 결과가 아닌 진단입니다. 최신 실제 학습 V219, 실사 비교 V220입니다. '
                '문서의 C55→최신 C65는 D 정답 +70, 외부 E 정답 −23으로 함께 개선되지 않았습니다. '
                'DINO·same-query COVER/SHAM·paired hinge는 이미 시도했고, 원 query/의도 부위/native 대응이 남은 과제입니다. '
                '공개 실사 학습은 현재 허용되며 직접 촬영한 평가223장은 계속 fit·문턱·모델 선택에서 제외합니다. '
                '이번 학습·forward·새 실사 평가·PnP 0, 모델 채택 없음, 목표 active입니다.\n\n')
    prefix = f'archive/task-2/pallet-keypoint-visibility/{PHASE}/'
    index = headline + f'[전체 검토]({prefix}PORTABLE_REPORT_KO.md) · [비교 그림]({prefix}VISIBLE_EXTERNAL_TRADEOFF.png) · [6모델 CM·지표 CSV]({prefix}ALREADY_PUBLISHED_COUNTS_AND_METRICS.csv)\n\n'
    index += f'[V95/V163 감사]({prefix}source_history_a/PORTABLE_KO.md) · [DINO·paired loss 감사]({prefix}source_history_b/KO.md) · [SOURCE 행별 ROI·query·split 표]({prefix}source_readiness_c/KO.md)\n\n'
    index += '[이전 V220 전체 인덱스·실제 crop·모델 결과](EXPERIMENT_INDEX_THROUGH_V220_KO.md)\n'
    index_path = DOCREL / 'EXPERIMENT_INDEX_THROUGH_V221_KO.md'
    checked(REPO / index_path).write_text(index)
    entrypoints = [Path('readme.md'), DOCREL / 'README.md', DOCREL / 'EXPERIMENT_INDEX_KO.md']
    for rel in entrypoints:
        f = checked(REPO / rel)
        old = f.read_text()
        link = str(index_path) if rel == Path('readme.md') else 'EXPERIMENT_INDEX_THROUGH_V221_KO.md'
        note = ('[V221 검토 대조·중복 시도와 남은 과제](' + link + ')\n\n'
                '최신 모델은 보이는 점 보존이 늘었지만 외부 가림 검출은 함께 좋아지지 않았습니다. '
                '첨부 V193 조언과 현재 계약·데이터·실험 이력을 대조한 결과이며 이번 새 학습·실사 추론은 없습니다.\n\n---\n\n')
        f.write_text(note + old)
    manifest_path = DOCREL / 'MANIFEST_CHAIN_AFTER_V221.json'
    dump(REPO / manifest_path, {
        'completed': True, 'phase': 'V221', 'previous_main': BASE, 'archive': str(ARC),
        'files': source,
        'source_bytes': sum(item['bytes'] for item in source),
        'new_models': 0, 'new_training_updates': 0, 'new_real_benchmark': False,
        'excluded_without_payload_access': ['raw datasets', 'model/feature caches', 'old protected arrays/labels/images'],
        'user_attachment_copied': False,
        'portable_report': str(ARC / 'PORTABLE_REPORT_KO.md'),
        'verification': 'Every copied file matches frozen source SHA. Original audit text is preserved; portable report and A derivative use repository-relative links.'
    })
    # Check navigation, without reading target payloads or following forbidden paths.
    link_files = [checked(REPO / index_path), checked(REPO / ARC / 'PORTABLE_REPORT_KO.md'),
                  checked(REPO / ARC / 'source_history_a/PORTABLE_KO.md'),
                  checked(REPO / ARC / 'source_history_b/KO.md'), checked(REPO / ARC / 'source_readiness_c/KO.md')]
    checked_links = 0
    for f in link_files:
        for href in re.findall(r'\]\(([^)]+)\)', f.read_text()):
            if '://' in href or href.startswith('#'):
                continue
            href = urllib.parse.unquote(href.split('#', 1)[0])
            assert not href.startswith('/'), f'unportable link: {href}'
            candidate = Path(os.path.abspath(f.parent / href))
            assert candidate.is_relative_to(REPO), 'relative link escaped publication checkout'
            target = checked(candidate)
            assert target.is_file(), f'missing navigation target: {href}'
            checked_links += 1
    stage = [str(ARC), str(index_path), str(manifest_path), *map(str, entrypoints)]
    git('add', '--', *stage)
    staged = git('diff', '--cached', '--name-only').splitlines()
    expected = {str(ARC / item['path']) for item in source}
    expected.update(map(str, [index_path, manifest_path, *entrypoints]))
    assert set(staged) == expected
    git('diff', '--cached', '--check')
    for item in source:
        blob = subprocess.run(['git', 'show', f":{ARC / item['path']}"], cwd=REPO,
                              check=True, capture_output=True).stdout
        assert digest(blob) == item['sha256']
    commit_message = 'docs: reconcile V193 review with V220 and audit prior source experiments'
    git('commit', '-m', commit_message)
    commit = git('rev-parse', 'HEAD')
    git('push', 'origin', 'HEAD:main')
    remote = git('ls-remote', 'origin', 'refs/heads/main').split()[0]
    assert remote == commit
    assert not git('status', '--short', '--untracked-files=no')
    for item in source:
        assert record(S / item['path']) == item
        assert digest(checked(REPO / ARC / item['path']).read_bytes()) == item['sha256']
    web = []
    for rel in [index_path, ARC / 'PORTABLE_REPORT_KO.md', ARC / 'VISIBLE_EXTERNAL_TRADEOFF.png',
                ARC / 'source_readiness_c/NOMINAL_D19_SOURCE_ROLE_AND_QUERY_PROVENANCE.csv']:
        url = f'https://raw.githubusercontent.com/CanelE452/pallet-6d-pose/{commit}/{rel}'
        req = urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'Codex-publication-verification'})
        try:
            with urllib.request.urlopen(req, timeout=25) as response:
                web.append({'url': url, 'http_status': response.status})
        except Exception as exc:
            web.append({'url': url, 'verification_error': type(exc).__name__})
    receipt = {
        'completed': True, 'phase': 'V221', 'main_pushed_verified': True, 'commit': commit,
        'previous_main': BASE, 'files_committed': len(staged), 'science_files': len(source),
        'science_bytes': sum(item['bytes'] for item in source), 'navigation_links_checked': checked_links,
        'source_and_archive_bytes_unchanged_after_push': True, 'http_HEAD_checks': web,
        'new_training_updates': 0, 'new_real_benchmark': False, 'new_models': 0, 'task_solved': False,
        'index_url': f'https://github.com/CanelE452/pallet-6d-pose/blob/{commit}/{index_path}',
        'report_url': f'https://github.com/CanelE452/pallet-6d-pose/blob/{commit}/{ARC}/PORTABLE_REPORT_KO.md',
    }
    dump(R / 'ROOT_V221_MAIN_PUBLICATION_RECEIPT.json', receipt)
    handoff_path = checked(P / 'ACTIVE_HANDOFF_KO.json')
    handoff = json.loads(handoff_path.read_text())
    prior = {'sha256': digest(handoff_path.read_bytes()), 'canonical_current_fields': handoff.get('canonical_current_fields')}
    handoff['latest_review'] = {'phase': 'V221', 'report': str(S / 'PORTABLE_REPORT_KO.md'),
                                'freeze': str(S / 'ROOT_PHASE_FREEZE.json'), 'new_model_result': False}
    publication_receipt = {'path': str(R / 'ROOT_V221_MAIN_PUBLICATION_RECEIPT.json'),
                           'sha256': digest(checked(R / 'ROOT_V221_MAIN_PUBLICATION_RECEIPT.json').read_bytes())}
    handoff['publication_current'] = {'status': 'MAIN_PUSHED_VERIFIED', 'commit': commit,
                                      'phases': ['V221'], 'index_url': receipt['index_url'],
                                      'receipt': publication_receipt, 'post_push_QA': publication_receipt}
    handoff['active_work'] = {
        'status': 'V221_REVIEW_RECONCILIATION_AND_SOURCE_HISTORY_AUDIT_COMPLETE_MAIN_PUBLISHED',
        'completed_phases': ['V219', 'V220', 'V221'], 'latest_training_phase': 'V219',
        'latest_real_benchmark_phase': 'V220', 'new_training_after_real_comparison_started': False,
        'fully_solved': False, 'selected_model': None,
        'next_action': 'Continue SOURCE evidence alignment: original initializer query/native/expected finite part, preserving weak-region versus native-point authority; avoid repeating DINO or COVER/SHAM/hinge as new ideas. A future matched fit must change only aligned source supervision, retain all DEV descendants outside every loss/normalizer, and freeze all settings before protected measurement. No PnP or current223-driven data/model/threshold selection.'}
    fields = handoff.setdefault('canonical_current_fields', [])
    if 'latest_review' not in fields:
        fields.append('latest_review')
    handoff['updated_at_UTC'] = datetime.now(timezone.utc).isoformat()
    dump(handoff_path, handoff)
    dump(R / 'ROOT_V221_FINAL_TURN_RECEIPT.json', {
        'completed': True, 'phase': 'V221', 'kind': 'REVIEW_AND_AUDIT_COMPLETE_NO_NEW_MODEL_RESULT',
        'publication_receipt': publication_receipt, 'commit': commit,
        'handoff_previous': prior, 'handoff_after_sha256': digest(handoff_path.read_bytes()),
        'source_and_published_archive_unchanged': True,
        'new_training_updates': 0, 'new_model_forwards': 0, 'new_real_benchmark': False,
        'new_PnP': 0, 'selected_model': None, 'task_solved': False, 'goal_status': 'active'})
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
