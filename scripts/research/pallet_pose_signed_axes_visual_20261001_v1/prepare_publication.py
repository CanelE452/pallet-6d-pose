"""Copy the reviewed native-input and656 experiment bundle to isolated checkout."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[3]
NAME='pallet_pose_signed_axes_visual_20261001_v1'
NATIVE='pallet_pose_dino_native_inputs_20261001_v1'
DOC=ROOT/'_docs/experiments'/NAME
CHECKOUT=Path('/tmp/pallet-pose-github-review-20260930')
BASE='3bccfa40bccd0e2aa8f37429ed7986a96963c154'

def binding(p):
    payload=p.read_bytes()
    return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload))

def inventory():
    paths=[ROOT/'readme.md',ROOT/'.gitignore']
    for name in (NATIVE,NAME):
        d=ROOT/'_docs/experiments'/name;h=ROOT/'scripts/research'/name
        paths += sorted(h.glob('*.py'))+sorted(d.glob('*.md'))+sorted(d.glob('*.json'))+sorted(d.glob('*.csv'))
        paths += sorted((d/'figures').glob('*.png'))+sorted((d/'figures').glob('*.jpg'))+sorted((d/'model_parameters').glob('*.json'))
    return paths

def main():
    r=json.loads((DOC/'PUBLIC_REVIEW.json').read_text());assert r['complete'] and r['PASS']
    for b in r['reviewed_artifacts']:assert binding(ROOT/b['path'])==b,b['path']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=CHECKOUT,text=True).strip()
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=CHECKOUT,text=True).strip()==BASE
    paths=inventory();manifest_path=DOC/'PUBLICATION_MANIFEST.json'
    assert len(paths)==len(set(paths)) and manifest_path not in paths
    reviewed={ROOT/b['path'] for b in r['reviewed_artifacts']}
    assert set(paths)==reviewed|{DOC/'PUBLIC_REVIEW.json',DOC/'PUBLIC_REVIEW_KO.md'}
    assert all(p.stat().st_size<20_000_000 for p in paths)
    data=json.loads((DOC/'REPORT_DATA.json').read_text())
    manifest=dict(schema='pallet_pose_signed_axes_visual_publication_v1',base_commit=BASE,
        files=[binding(p) for p in paths],public_files_including_manifest=len(paths)+1,
        images=sum(p.suffix in ('.jpg','.png') for p in paths),
        source_VAL_checks_passed=data['source_checks_passed'],source_VAL_checks_total=45,
        new_fits=4,current_real_evaluated=data['current_real_evaluated'],
        stable_joint_improvement_achieved=data['stable_joint_improvement_achieved'])
    with manifest_path.open('x') as f:json.dump(manifest,f,ensure_ascii=False,indent=2);f.write('\n')
    paths.append(manifest_path)
    for p in paths:
        q=CHECKOUT/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
        assert q.read_bytes()==p.read_bytes()
    print('VISUAL656_PUBLICATION_PREPARED',len(paths),'files',manifest['images'],'images',flush=True)

if __name__=='__main__':main()
