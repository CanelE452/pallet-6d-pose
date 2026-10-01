"""Publish the rejected optimization experiment without inventing evaluations."""
from pathlib import Path
import ast
import re
import shutil
import subprocess
from urllib.parse import unquote
from . import common as C

CHECKOUT=Path('/tmp/pallet-pose-github-review-20260930')
BASE='e959f803cd1542031ee161cc32be6b44bf85ec9b'


def main():
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=CHECKOUT,text=True).strip()
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CHECKOUT,text=True).strip()
    assert head==BASE==subprocess.check_output(['git','rev-parse','origin/main'],cwd=CHECKOUT,text=True).strip()
    for name in ('PREFIT_REVIEW.json','REJECTED_FIT_VERIFICATION.json','EVALUATION_NOT_RUN.json','PUBLIC_REVIEW.json'):
        value=C.read(C.DOC/name);assert value['complete'] and value['PASS'],name
    rejected=C.read(C.DOC/'REJECTED_R0_ONLY.json')
    cert=rejected['certificate']
    assert rejected['complete'] and not rejected['accepted'] and not cert['PASS'] and not cert['optimizer_success']
    assert cert['iterations']==1000 and cert['objective_calls']==1108
    assert not (C.DOC/'TRAINING_COMPLETE.json').exists()
    review=C.read(C.DOC/'PUBLIC_REVIEW.json')
    for item in review['reviewed_artifacts']:C.verify(item)
    files=[C.ROOT/'.gitignore',C.ROOT/'readme.md']
    for folder in (C.DOC,C.HERE):
        files += [p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='PUBLICATION_MANIFEST.json']
    files=sorted(set(files))
    reviewed={b['path'] for b in review['reviewed_artifacts']}
    exempt={str((C.DOC/name).relative_to(C.ROOT)) for name in ('PUBLIC_REVIEW.json','PUBLIC_REVIEW_KO.md')}
    exempt.add(str(Path(__file__).resolve().relative_to(C.ROOT)))
    assert {str(p.relative_to(C.ROOT)) for p in files}<=reviewed|exempt
    for p in files:
        assert p.suffix in ('.md','.json','.csv','.png','.jpg','.py') or p.name=='.gitignore'
        if p.suffix=='.py':ast.parse(p.read_text())
    manifest=dict(complete=True,created_at=C.now(),repository='CanelE452/pallet-6d-pose',branch='main',base_commit=head,
        namespace=C.NAME,files=[C.bind(p) for p in files],public_files_including_this_manifest=len(files)+1,
        status='TRAIN_CONVERGENCE_FAILED',new_fit_attempts=1,rejected_fit_attempts=1,certified_models=0,
        skipped_fit_models=['UNION_s1','UNION_s2','UNION_s3'],objective_calls=1108,optimizer_iterations=1000,
        source_VAL_evaluated=False,learned_real_evaluated=False,current_method_T_R_effect_measured=False,
        new_source_VAL_routes=0,new_learned_real_routes=0,new_reference_pose_metric_calls=0,new_image_forwards=0,new_PnP=0,
        image_count=sum(p.suffix in ('.png','.jpg') for p in files),actual_RGB_frames=6,
        figure_scope='Original real RGB with dimensions and historical R0 only; no current learned real predictions.',
        method_success=False,stable_joint_improvement_achieved=False,goal_complete=False,
        source_workspace_preserved='Explicit artifact copy through isolated checkout; original dirty git index and unrelated edits untouched.')
    path=C.DOC/'PUBLICATION_MANIFEST.json';C.save(path,manifest);files.append(path)
    for p in files:
        q=CHECKOUT/p.relative_to(C.ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
        assert q.read_bytes()==p.read_bytes()
    links=0
    for p in files:
        if p.suffix!='.md' or p.name=='readme.md':continue
        q=CHECKOUT/p.relative_to(C.ROOT)
        for link in re.findall(r'\]\(([^)]+)\)',q.read_text()):
            if link.startswith(('https://','http://','#')):continue
            rel=unquote(link.split('#')[0])
            if not rel:continue
            target=(q.parent/rel).resolve()
            assert target.is_relative_to(CHECKOUT) and target.exists(),(str(p),link)
            links+=1
    print('PUBLICATION_PREPARED',dict(files=len(files),images=manifest['image_count'],links=links,base=head))


if __name__=='__main__':main()
