"""Copy only this phase's reviewable artifacts to an isolated publication checkout."""
from . import common as C
import ast
import json
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import unquote

CHECKOUT = Path('/tmp/pallet-pose-github-review-20260930')
NAMES = ('pallet_pose_selector_objective_audit_20261001_v1', C.NAME)


def main():
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=CHECKOUT, text=True).strip()
    base = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CHECKOUT, text=True).strip()
    remote = subprocess.check_output(['git', 'rev-parse', 'origin/main'], cwd=CHECKOUT, text=True).strip()
    assert base == remote
    gate = C.read(C.DOC / 'SOURCE_VAL_GATE.json')
    assert gate['complete'] and not gate['PASS'] and gate['checks_passed'] == 41
    for name in ('PUBLIC_REVIEW.json', 'SOURCE_VAL_VERIFICATION.json', 'TRAIN_CONVERGENCE.json'):
        assert (C.DOC / name).exists(), name
    files = [C.ROOT / '.gitignore', C.ROOT / 'readme.md']
    for namespace in NAMES:
        for folder in (C.ROOT / '_docs/experiments' / namespace, C.ROOT / 'scripts/research' / namespace):
            files.extend(p for p in folder.rglob('*') if p.is_file()
                         and '__pycache__' not in p.parts and p.name != 'PUBLICATION_MANIFEST.json')
    files = sorted(set(files))
    assert all(p.suffix in ('.md', '.json', '.csv', '.png', '.py') or p.name == '.gitignore' for p in files)
    for p in files:
        if p.suffix == '.py':
            ast.parse(p.read_text(), filename=str(p))
    manifest_path = C.DOC / 'PUBLICATION_MANIFEST.json'
    manifest = dict(complete=True, created_at=C.now(), base_commit=base,
        repository='CanelE452/pallet-6d-pose', branch='main', namespaces=list(NAMES),
        files=[C.bind(p) for p in files], image_count=sum(p.suffix == '.png' for p in files),
        performance_verdict='Four new scorer fits converged; source VAL failed4/45; no new real routing. '
            'Stable real joint T/R improvement remains unachieved.',
        new_fits=4, source_VAL_rows=1024, real_new_routing=False,
        stable_joint_improvement_achieved=False,
        source_workspace_preserved='Original source .git and unrelated user changes are not committed from the dirty checkout.')
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
    print(json.dumps(dict(files=len(files), images=manifest['image_count'],
                          local_markdown_links_checked=links, base_commit=base), indent=2))


if __name__ == '__main__':
    main()
