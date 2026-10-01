"""Publish the immutable input-audit receipt using its actual artifacts schema.

The earlier invocation stopped while the final receipt was still pending,
before creating a manifest or copying files. Static review then identified its
expectation of the previous experiment's reviewed_artifacts key. This adapter
keeps all 38 files bound by PUBLIC_REVIEW unchanged. Root README and .gitignore
are publication entry points reviewed separately; they are not misreported as
part of the 38-file mathematical/public-report receipt.
"""
from pathlib import Path
import ast
import re
import shutil
import subprocess
from urllib.parse import unquote
from . import common as C

CHECKOUT=Path('/tmp/pallet-pose-github-review-20260930')
BASE='f78faaa6456849bb5a111cf80ad85e32d8ce0bcf'


def main():
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=CHECKOUT,text=True).strip()
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CHECKOUT,text=True).strip()
    assert head==BASE==subprocess.check_output(['git','rev-parse','origin/main'],cwd=CHECKOUT,text=True).strip()
    for name in ('FEATURE_AUDIT.json','REPRESENTATION_AUDIT.json','VERIFICATION.json','PUBLIC_REVIEW.json'):
        value=C.read(C.DOC/name);assert value['complete'] and value['PASS'],name
    result=C.read(C.DOC/'REPRESENTATION_AUDIT.json')
    assert result['source_TRAIN_only'] and result['diagnostic_only']
    assert result['new_fits']==result['new_policy_argmin']==result['new_PnP_calls']==0
    assert not result['method_success'] and not result['goal_complete']
    assert not result['VAL_quality_read'] and not result['real_targets_read']
    C.verify(result['arrays'])
    review=C.read(C.DOC/'PUBLIC_REVIEW.json')
    for item in review['artifacts']:C.verify(item)
    files=[C.ROOT/'.gitignore',C.ROOT/'readme.md']
    for folder in (C.DOC,C.HERE):
        files += [p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='PUBLICATION_MANIFEST.json']
    files=sorted(set(files))
    reviewed={b['path'] for b in review['artifacts']}
    exempt={str((C.DOC/name).relative_to(C.ROOT)) for name in ('PUBLIC_REVIEW.json','PUBLIC_REVIEW_KO.md')}
    exempt.add(str(Path(__file__).resolve().relative_to(C.ROOT)))
    exempt.update(('.gitignore','readme.md'))
    assert {str(p.relative_to(C.ROOT)) for p in files}<=reviewed|exempt
    for p in files:
        assert p.suffix in ('.md','.json','.csv','.png','.jpg','.py') or p.name=='.gitignore'
        if p.suffix=='.py':ast.parse(p.read_text())
    manifest=dict(complete=True,created_at=C.now(),repository='CanelE452/pallet-6d-pose',branch='main',base_commit=head,
        namespace=C.NAME,public_review=C.bind(C.DOC/'PUBLIC_REVIEW.json'),reviewed_file_count=len(review['artifacts']),
        publication_schema_adapter=C.bind(__file__),separately_reviewed_entry_points=[C.bind(C.ROOT/'.gitignore'),C.bind(C.ROOT/'readme.md')],
        namespace_note='Original 38 reviewed artifacts unchanged; this adapter consumes artifacts rather than the previous experiment reviewed_artifacts schema.',files=[C.bind(p) for p in files],public_files_including_this_manifest=len(files)+1,
        status='TRAIN_INPUT_AUDIT_ONLY',input_novelty_status=result['novelty']['status'],
        new_fits=0,optimizer_steps=0,new_policy_argmin=0,source_VAL_evaluated=False,
        learned_real_evaluated=False,current_method_real_T_R_effect_measured=False,
        new_image_forwards=0,new_PnP=0,image_count=sum(p.suffix in ('.png','.jpg') for p in files),actual_RGB_frames=6,
        figure_scope='TRAIN input diagnostics and six fixed TRAIN source RGB frames with dimensions and enlarged residual arrows; no new pose performance measurements.',
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
