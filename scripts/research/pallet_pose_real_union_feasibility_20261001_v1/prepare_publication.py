"""Publish only this diagnostic's reviewed artifacts to the isolated checkout."""
from . import common as C
from pathlib import Path
import ast
import re
import shutil
import subprocess
from urllib.parse import unquote

CHECKOUT = Path('/tmp/pallet-pose-github-review-20260930')


def main():
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=CHECKOUT, text=True).strip()
    base = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CHECKOUT, text=True).strip()
    assert base == subprocess.check_output(['git', 'rev-parse', 'origin/main'], cwd=CHECKOUT, text=True).strip()
    results = C.read(C.DOC / 'RESULTS.json')
    assert results['complete'] and not results['method_success'] and not results['goal_complete']
    for name in ('VERIFICATION.json', 'PUBLIC_REVIEW.json'):
        review = C.read(C.DOC / name)
        assert review['complete'] and review['PASS'], name
    report = C.read(C.DOC / 'REPORT_DATA.json')
    assert report['complete'] and report['CSV_rows'] == 1038
    files = [C.ROOT / '.gitignore', C.ROOT / 'readme.md']
    for folder in (C.DOC, C.HERE):
        files += [p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                  and p.name != 'PUBLICATION_MANIFEST.json']
    files = sorted(set(files))
    assert all(p.suffix in ('.md', '.json', '.csv', '.png', '.jpg', '.py') or p.name == '.gitignore' for p in files)
    for p in files:
        if p.suffix == '.py':
            ast.parse(p.read_text(), filename=str(p))
    manifest = dict(complete=True, created_at=C.now(), repository='CanelE452/pallet-6d-pose',
        branch='main', base_commit=base, namespace=C.NAME, files=[C.bind(p) for p in files],
        public_files_including_this_manifest=len(files) + 1,
        images=sum(p.suffix in ('.png', '.jpg') for p in files), actual_RGB_frames=6,
        diagnostic_gate_results={d: v['stability']['PASS'] for d, v in results['diagnostics'].items()},
        diagnostic_only=True, new_fits=0, new_image_forwards=0, new_PnP=0,
        new_reference_metric_calls=0, new_learned_real_routing=0,
        method_success=False, stable_joint_improvement_achieved=False, goal_complete=False,
        source_workspace_preserved='Only listed artifacts copied; original dirty source .git and unrelated user modifications remain untouched.')
    manifest_path = C.DOC / 'PUBLICATION_MANIFEST.json'
    C.save(manifest_path, manifest)
    files.append(manifest_path)
    for p in files:
        target = CHECKOUT / p.relative_to(C.ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
        assert p.read_bytes() == target.read_bytes()
    links = 0
    for p in files:
        if p.suffix != '.md' or p.name == 'readme.md':
            continue
        target = CHECKOUT / p.relative_to(C.ROOT)
        for url in re.findall(r'\]\(([^)]+)\)', target.read_text()):
            if url.startswith(('https://', 'http://', '#')):
                continue
            url = unquote(url.split('#')[0])
            if not url:
                continue
            resolved = (target.parent / url).resolve()
            assert resolved.is_relative_to(CHECKOUT) and resolved.exists(), (str(p), url)
            links += 1
    print('PUBLICATION_PREPARED', dict(files=len(files), images=manifest['images'], links=links, base=base))


if __name__ == '__main__':
    main()
