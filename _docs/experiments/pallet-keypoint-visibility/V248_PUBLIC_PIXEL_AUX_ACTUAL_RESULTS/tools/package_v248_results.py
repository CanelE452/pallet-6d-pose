"""Bounded V248 publication copy; no model, git mutation, or source reprocessing."""
from pathlib import Path
import argparse
import hashlib
import json
import os

R = Path(__file__).absolute().parent
H = R / 'V248_ROOT_PUBLIC_PIXEL_AUX'
PACKAGE = R / 'V248_PUBLICATION_PACKAGE'
DIRS = {
    'V248_ROOT_PUBLIC_PIXEL_AUX': ('aux_sample_records_v1', 'aux_canonical_pixel_features_v1', 'matched_source_public_pixel_aux_fit_v1'),
    'V248_PUBLIC_PIXEL_AUX_BUILDER_B': (),
    'V248_PUBLIC_PIXEL_AUX_TRAINER_A': (),
    'V248_PALLOX_AUX_SAMPLER_READINESS_B': (),
    'V248_EXISTING_TRAINER_READINESS_A': (),
    'V248_PUBLIC_AUX_POLICY_QA_C': (),
    'V248_PUBLIC_AUX_TRAINER_CODE_QA_C': (),
    'V248_PREVIOUS_PROGRESS_QA_C': (),
    'V248_REFERENCE_DOCUMENT_QA_C': (),
    'V248_SAVED_SOURCE_QA_C': ('actual_review_v1', 'portable_replay_v1'),
    'V248_PUBLISHED_DEPENDENCIES_A': (),
}


def safe(p):
    p = Path(p)
    if not p.is_absolute() or '..' in p.parts:
        raise ValueError('Absolute normalized artifact paths only')
    if any(s in str(p).casefold() for s in ('v112', '__pycache__', '.pyc')):
        raise ValueError('Forbidden explicit artifact path before access')
    if any(a.is_symlink() for a in [p, *p.parents]):
        raise ValueError('Symlink artifact rejected')
    return p


def read(p):
    return safe(p).read_bytes()


def sha(b):
    return hashlib.sha256(b).hexdigest()


def verify(package):
    package = safe(package)
    m = json.loads(read(package / 'MANIFEST.json'))
    for row in m['files']:
        rel = Path(row['relative_path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Relative package leaf only')
        b = read(package / rel)
        if len(b) != row['size_bytes'] or sha(b) != row['sha256']:
            raise ValueError('Published leaf mismatch: '+str(rel))
    # Only manifest-listed relative leaves are followed. No original absolute pointer fallback.
    print(json.dumps({'completed': True, 'PASS': True, 'leaves': len(m['files']),
                      'bytes': sum(f['size_bytes'] for f in m['files']),
                      'original_pointer_fallback': False, 'model_execution': 0}))


def build(report_qa):
    qa = json.loads(read(report_qa))
    if qa.get('PASS') is not True:
        raise ValueError('Final report QA required')
    if safe(PACKAGE).exists():
        raise ValueError('Fresh package only')
    PACKAGE.mkdir()
    files = []
    seen = set()

    def take(src, relative):
        if relative in seen:
            raise ValueError('Duplicate destination')
        seen.add(relative)
        b = read(src)
        if len(b) >= 95_000_000:
            raise ValueError('Unexpected large publication leaf')
        dst = safe(PACKAGE / relative)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open('xb') as f:
            f.write(b)
        files.append({'relative_path': relative, 'original_path': str(src),
                      'sha256': sha(b), 'size_bytes': len(b)})

    def direct(folder, prefix, allowed_children=()):
        # Names-only inventory of one explicit newly-created V248 directory.
        for name in sorted(os.listdir(safe(folder))):
            src = safe(folder / name)
            if src.is_dir():
                if name not in allowed_children:
                    raise ValueError('Undeclared child directory '+name)
                direct(src, prefix+'/'+name)
            elif src.is_file():
                if src.suffix not in ('.py', '.json', '.jsonl', '.md', '.csv', '.npy', '.npz', '.pt', '.png', '.html'):
                    raise ValueError('Unexpected artifact suffix')
                take(src, prefix+'/'+name)
            else:
                raise ValueError('Nonregular artifact')

    for dirname, children in DIRS.items():
        direct(R / dirname, 'artifacts/'+dirname, children)
    direct(Path(report_qa).parent, 'artifacts/'+Path(report_qa).parent.name)
    direct(R / 'V248_FIXED_SOURCE_AUX_VISUALS_B', 'visuals')
    for src, dest in [
        (H/'PUBLIC_README_KO.md', 'README_KO.md'),
        (H/'ROOT_READOUT_KO.html', 'ROOT_READOUT_KO.html'),
        (H/'DEV_COMPARISON.csv', 'DEV_COMPARISON.csv'),
        (H/'SOURCE_CONFUSION_OVERVIEW.png', 'SOURCE_CONFUSION_OVERVIEW.png'),
        (R/'V248_FIXED_SOURCE_AUX_VISUALS_B/GALLERY.html', 'visuals/PORTABLE_GALLERY_KO.html'),
        (R/'build_v248_public_readout.py', 'tools/build_v248_public_readout.py'),
        (Path(__file__).absolute(), 'tools/package_v248_results.py'),
    ]:
        take(src, dest)
    m = {'schema': 'V248_EXPLICIT_NEW_SOURCE_PUBLICATION_LEAVES_V1',
         'completed': True, 'PASS': True, 'files': files,
         'new_source_fit_updates': 2000, 'SOURCE_gate_PASS': False,
         'model_adopted': False, 'new_current_inference': 0, 'PnP': 0,
         'metadata_absolute_paths': 'Provenance strings only. Portable verification follows relative leaves, never original pointers.',
         'weights': 'New small SOURCE head checkpoints included; no official DINO or SAM checkpoint copied.',
         'reproduction': 'CSV+summary arithmetic is portable. Original fixed path-pinned training entries require explicit path remapping to reproduce; existing published dependencies are mapped separately.'}
    with safe(PACKAGE/'MANIFEST.json').open('x') as f:
        json.dump(m, f, ensure_ascii=False, indent=2); f.write('\n')
    verify(PACKAGE)


if __name__=='__main__':
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--build', type=Path, metavar='FINAL_REPORT_QA_JSON')
    g.add_argument('--verify', type=Path, metavar='PACKAGE_DIRECTORY')
    args = p.parse_args()
    if args.build:
        build(args.build)
    else:
        verify(args.verify)
