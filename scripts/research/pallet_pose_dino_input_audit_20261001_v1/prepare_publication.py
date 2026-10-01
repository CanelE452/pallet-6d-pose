"""Copy only the reviewed input audit, not private RGB/token arrays, to checkout."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[3]
NAME='pallet_pose_dino_input_audit_20261001_v1'
DOC=ROOT/'_docs/experiments'/NAME
HERE=Path(__file__).resolve().parent
CHECKOUT=Path('/tmp/pallet-pose-github-review-20260930')


def binding(p):
    data=p.read_bytes()
    return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))


def main():
    review=json.loads((DOC/'PUBLIC_REVIEW.json').read_text())
    assert review['PASS'] and not review['stable_joint_improvement_achieved']
    for b in review['reviewed_artifacts']:
        assert binding(ROOT/b['path'])==b,b['path']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=CHECKOUT,text=True).strip()
    base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CHECKOUT,text=True).strip()
    assert base=='232e2bb3407610bdc39f7db59472706cb11adf69',base
    paths=[ROOT/'readme.md',ROOT/'.gitignore',*sorted(HERE.glob('*.py')),
           *sorted(DOC.glob('*.md')),*sorted(DOC.glob('*.json')),*sorted((DOC/'figures').glob('*.png')),
           *sorted((DOC/'figures').glob('*.jpg'))]
    manifest_path=DOC/'PUBLICATION_MANIFEST.json'
    assert manifest_path not in paths and len(paths)==len(set(paths))
    reviewed={ROOT/b['path'] for b in review['reviewed_artifacts']}
    assert set(paths)==reviewed|{DOC/'PUBLIC_REVIEW.json',DOC/'PUBLIC_REVIEW_KO.md'}
    assert all(p.stat().st_size<20_000_000 for p in paths)
    manifest=dict(schema='pallet_pose_dino_input_audit_publication_v1',base_commit=base,
        files=[binding(p) for p in paths],public_files_including_manifest=len(paths)+1,
        images=sum(p.suffix in ('.png','.jpg') for p in paths),
        status='TRAIN_INPUTS_VERIFIED_NO_NEW_FIT_OR_T_R_EVALUATION',
        stable_joint_improvement_achieved=False,new_fits=0,new_source_VAL_evaluations=0,new_real_evaluations=0,
        source_TRAIN_rows=2598,image_forwards=2597,valid_candidates_verified=20776,
        latest_method_source_gate='43/45 FAIL; prior asymmetric experiment unchanged')
    with manifest_path.open('x') as f:
        json.dump(manifest,f,ensure_ascii=False,indent=2);f.write('\n')
    paths.append(manifest_path)
    for p in paths:
        q=CHECKOUT/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,q)
        assert q.read_bytes()==p.read_bytes()
    print('PUBLICATION_PREPARED',len(paths),'files',manifest['images'],'images',base)


if __name__=='__main__':main()
