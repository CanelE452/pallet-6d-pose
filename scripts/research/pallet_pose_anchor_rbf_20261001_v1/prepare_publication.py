"""Publish only reviewed fixed-RBF artifacts through an isolated checkout."""
from pathlib import Path
import ast
import re
import shutil
import subprocess
from urllib.parse import unquote
from . import common as C

CHECKOUT = Path('/tmp/pallet-pose-github-review-20260930')


def main():
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=CHECKOUT, text=True).strip()
    base = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CHECKOUT, text=True).strip()
    assert base == subprocess.check_output(['git', 'rev-parse', 'origin/main'], cwd=CHECKOUT, text=True).strip()
    assert base == '1d700958e53a1facd39253cdc74e11cd24266309'
    for name in ('PREFIT_REVIEW.json', 'SOURCE_VAL_VERIFICATION.json',
                 'TRAIN_CONVERGENCE.json', 'REAL_VERIFICATION.json', 'PUBLIC_REVIEW.json'):
        review = C.read(C.DOC / name)
        assert review['complete'] and review['PASS'], name
    # Publication review binds every previously finalized artifact. No source
    # score, training, reference geometry, or model execution occurs here.
    review = C.read(C.DOC / 'PUBLIC_REVIEW.json')
    for binding in review['reviewed_artifacts']:
        C.verify(binding)
    source = C.read(C.DOC / 'SOURCE_VAL_GATE.json')
    real = C.read(C.DOC / 'REAL_RESULTS.json')
    train = C.read(C.DOC / 'TRAINING_COMPLETE.json')
    assert source['complete'] and source['PASS']
    assert source['checks_passed'] == source['checks_total'] == 45
    assert real['complete'] and real['full_frame_rows'] == 2249
    assert real['baseline_metric_parity_checks'] == 1557
    real_passed = sum(g['PASS'] for g in real['stability']['gates'].values())
    assert real['stability']['PASS'] == (real_passed == 5)
    assert real['stability']['goal_complete'] == real['stability']['PASS']
    assert train['fit_count'] == 4 and train['all_certified']
    assert train['total_objective_calls'] == 1573
    files = [C.ROOT / '.gitignore', C.ROOT / 'readme.md']
    for folder in (C.DOC, C.HERE):
        files += [p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                  and p.name != 'PUBLICATION_MANIFEST.json']
    files = sorted(set(files))
    assert all(p.suffix in ('.md', '.json', '.csv', '.png', '.jpg', '.py') or p.name == '.gitignore' for p in files)
    reviewed = {b['path'] for b in review['reviewed_artifacts']}
    exempt = {str((C.DOC / name).relative_to(C.ROOT)) for name in ('PUBLIC_REVIEW.json', 'PUBLIC_REVIEW_KO.md')}
    exempt.add(str(Path(__file__).resolve().relative_to(C.ROOT)))
    assert {str(p.relative_to(C.ROOT)) for p in files} <= reviewed | exempt
    for p in files:
        if p.suffix == '.py':
            ast.parse(p.read_text(), filename=str(p))
    manifest = dict(complete=True, created_at=C.now(), repository='CanelE452/pallet-6d-pose',
        branch='main', base_commit=base, namespace=C.NAME, files=[C.bind(p) for p in files],
        public_files_including_this_manifest=len(files) + 1,
        images=sum(p.suffix in ('.png', '.jpg') for p in files), actual_RGB_frames=6,
        source_VAL_gate_PASS=True, source_VAL_checks_passed=45, source_VAL_checks_total=45,
        real_stability_gate_PASS=real['stability']['PASS'], real_stability_checks_passed=real_passed, real_stability_checks_total=5,
        original_goal_and_matched_intervention_required=True,
        new_fits=4, objective_calls=1573, optimizer_iterations=1433,
        new_image_forwards=0, new_PnP=0, real_scored_pose_rows=2249,
        new_real_reference_metric_calls=2249, real_baseline_metric_parity_checks=1557,
        new_learned_real_routing=692, method_success=real['stability']['PASS'],
        stable_joint_improvement_achieved=real['stability']['PASS'], goal_complete=real['stability']['goal_complete'],
        feature_dim=253, frozen_RBF_centers=64, basis=C.bind(C.DOC / 'RBF_BASIS.json'),
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
