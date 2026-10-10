"""Read-only verification of immutable GitHub artifacts and both branch refs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPOSITORY = 'CanelE452/pallet-6d-pose'
RESEARCH_BRANCH = 'research/feature-gradient-joint-refine-20261010'
DOC = '_docs/experiments/pallet_feature_gradient_joint_20261010'
SOURCE = 'scripts/research/pallet_feature_gradient_joint_20261010'


def git(*args):
    return subprocess.check_output(['git', *args])


def get(url):
    request = Request(url, headers={'User-Agent': 'pallet-joint-artifact-verifier/1.0'})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=45) as response:
                assert response.status == 200, (url, response.status)
                return response.read()
        except (HTTPError, URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(1 + attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit', required=True, help='Immutable full commit SHA')
    parser.add_argument('--output', type=Path, required=True, help='Fresh receipt path')
    args = parser.parse_args()
    commit = git('rev-parse', args.commit + '^{commit}').decode().strip()
    assert commit == args.commit and len(commit) == 40
    assert not args.output.exists(), 'Receipt already exists; use a fresh output path'
    remote = 'https://github.com/' + REPOSITORY + '.git'
    raw_refs = git('ls-remote', remote, 'refs/heads/main', 'refs/heads/' + RESEARCH_BRANCH).decode()
    refs = {ref: sha for sha, ref in (line.split() for line in raw_refs.splitlines())}
    assert refs.get('refs/heads/main') == commit, refs
    assert refs.get('refs/heads/' + RESEARCH_BRANCH) == commit, refs
    files = git('ls-tree', '-r', '--name-only', commit, '--', DOC, SOURCE).decode().splitlines()
    assert len([p for p in files if p.endswith('.png')]) == 6
    for filename in ('RESULT_KO.md', 'METHOD_KO.md', 'REPRODUCE.md', 'README.md',
                     'VERIFICATION.json', 'SHA256_MANIFEST.json'):
        assert DOC + '/' + filename in files

    def verify_file(path):
        expected = git('show', commit + ':' + path)
        url = 'https://raw.githubusercontent.com/' + REPOSITORY + '/' + commit + '/' + path
        observed = get(url)
        expected_sha = hashlib.sha256(expected).hexdigest()
        assert hashlib.sha256(observed).hexdigest() == expected_sha, path
        return dict(path=path, bytes=len(observed), sha256=expected_sha, raw_url=url,
                    status='HTTP_200_SHA256_MATCH')

    with ThreadPoolExecutor(max_workers=4) as pool:
        artifacts = list(pool.map(verify_file, files))
    rendered = []
    for path in files:
        if path.endswith('.png') or path.endswith('.md'):
            url = 'https://github.com/' + REPOSITORY + '/blob/' + commit + '/' + path
            get(url)
            rendered.append(dict(path=path, url=url, status='HTTP_200'))
    receipt = dict(status='PASS', repository=REPOSITORY, artifact_commit_sha=commit,
                   observed_remote_refs=refs, all_artifacts=artifacts,
                   rendered_report_and_figure_links=rendered,
                   verified_at=datetime.now(timezone.utc).isoformat(), force_push=False,
                   receipt_scope='Immutable supplied commit and remote refs observed at this time; receipt may be added in a later normal commit.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps(dict(status='PASS', commit=commit, artifacts=len(artifacts),
                          rendered_links=len(rendered), refs=refs)), flush=True)


if __name__ == '__main__':
    main()
