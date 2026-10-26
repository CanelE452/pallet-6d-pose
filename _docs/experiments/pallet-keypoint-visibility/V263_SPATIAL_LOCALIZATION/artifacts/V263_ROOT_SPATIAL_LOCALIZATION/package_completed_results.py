"""Package named V263 outputs, preserving bytes and explicitly excluding huge tokens.

Run only after all workers have sealed COMPLETE outputs. No experiments or Git actions.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess

R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
REPO = Path('/home/minjae/Documents/pallet-visibility-publication-20261024')
DEST = REPO / '_docs/experiments/pallet-keypoint-visibility/V263_SPATIAL_LOCALIZATION'
NAMES = ['V263_ROOT_SPATIAL_LOCALIZATION', 'V263_LOCALIZATION_DATA_A',
         'V263_ORACLE_DIAGNOSTIC_B', 'V263_SPATIAL_HEAD_C',
         'V263_CURRENT_EVALUATION_A', 'V263_CURRENT_UTILITY_B']

def safe(p):
    p = Path(p)
    if '..' in p.parts or any(t in str(p).lower() for t in ['v112', 'cache', '.pyc']):
        raise ValueError(p)
    for a in reversed([p, *p.parents]):
        if a.is_symlink():
            raise ValueError('symlink: ' + str(a))
    return p

def digest(p):
    h = hashlib.sha256()
    with safe(p).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def main():
    safe(DEST).mkdir(exist_ok=True)
    rows, omitted, staged = [], [], []
    for name in NAMES:
        source = safe(R / name)
        if not source.is_dir():
            raise FileNotFoundError(source)
        # These exact six named output trees are the entire traversal scope.
        for current, dirs, files in os.walk(source, followlinks=False):
            dirs[:] = sorted(d for d in dirs if not any(t in d.lower() for t in ['v112','cache','.pyc']))
            for d in dirs:
                safe(Path(current) / d)
            for filename in sorted(files):
                if any(t in filename.lower() for t in ['v112','cache','.pyc']):
                    continue
                p = safe(Path(current) / filename)
                if filename.endswith('.nul') or filename in ['RESULTS_PACKAGE_MANIFEST.json', 'PREVIOUS_STATE_V262.json']:
                    continue
                size = p.stat().st_size
                h = digest(p)
                relative = Path('artifacts') / name / p.relative_to(source)
                row = {'original_absolute_path': str(p), 'package_path': str(relative),
                       'sha256': h, 'size_bytes': size}
                if size > 95 * 1024 * 1024:
                    row['reason'] = 'Large frozen spatial token tensor; replay with preserved crop bytes, official encoder binding and extraction CODE.'
                    omitted.append(row)
                    continue
                target = safe(DEST / relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and digest(target) != h:
                    committed = subprocess.run(['git','cat-file','-e','HEAD:'+str(target.relative_to(REPO))],cwd=REPO,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode == 0
                    if committed:
                        raise RuntimeError('Refuse changing already published artifact: ' + str(target))
                shutil.copyfile(p, target)
                if digest(target) != h:
                    raise RuntimeError('copy mismatch')
                rows.append(row)
                staged.append(str(target.relative_to(REPO)))
    # The original SIMPLE crop packet is small, synthetic only, and directly used.
    source = safe(R.parent / 'query_visibility_expected_part_weak_paired_source_training_20261025_v219/source_fit_rows_v1/SYNTH_AUG65536_INPUTS.npz')
    expected = 'eb8b24014863012779114407630a66519010184ea8d3e864477de3550d38ef34'
    if digest(source) != expected:
        raise RuntimeError('original synthetic input changed')
    relative = Path('source_inputs/SYNTH_AUG65536_INPUTS.npz')
    target = safe(DEST / relative)
    target.parent.mkdir(exist_ok=True)
    shutil.copyfile(source, target)
    rows.append({'original_absolute_path': str(source), 'package_path': str(relative),
                 'sha256': expected, 'size_bytes': source.stat().st_size})
    staged.append(str(target.relative_to(REPO)))
    report = {'completed': True, 'PASS': True, 'copied_files': len(rows), 'files': rows,
              'excluded_large_recomputable_files': omitted,
              'raw_current_or_additional_real_archives_copied': False,
              'path_contract': 'Original absolute paths are provenance, not portable commands. package_path is relative to this manifest.'}
    manifest = DEST / 'RESULTS_PACKAGE_MANIFEST.json'
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    staged.append(str(manifest.relative_to(REPO)))
    (R / 'V263_ROOT_SPATIAL_LOCALIZATION/RESULTS_STAGE_ALLOWLIST.nul').write_bytes(b'\0'.join(p.encode() for p in staged) + b'\0')
    print(json.dumps({'copied_files': len(rows), 'excluded': len(omitted),
                      'manifest': str(manifest), 'stage_paths': len(staged)}))

if __name__ == '__main__':
    main()
