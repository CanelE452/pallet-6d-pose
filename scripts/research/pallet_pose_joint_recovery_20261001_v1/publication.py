"""Validate this completed audit and stage exactly its public files.

No commit, push, model fitting, scoring, or original-worktree git mutation.
"""
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
from PIL import Image

from .selector_attribution import ROOT, DOC, RAW, NAME, PREV, read, save, bind

CHECKOUT = Path('/tmp/pallet-pose-github-review-20260930')
BASE = '51907cf35173ade213eec154abb2e9bcd307adf3'
HERE = Path(__file__).resolve().parent


def bindings(value):
    if isinstance(value,dict):
        if 'path' in value and 'sha256' in value:
            yield value
        else:
            for child in value.values():
                yield from bindings(child)
    elif isinstance(value,list):
        for child in value:
            yield from bindings(child)


def main():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=CHECKOUT,text=True).strip() == BASE
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=CHECKOUT,text=True).strip()
    required = ['REPORT_KO.md','ASSOCIATION_AUDIT_KO.md','CONTINUOUS_POSE_AUDIT_KO.md',
        'SELECTOR_FEASIBILITY_KO.md','FINAL_REVIEW_KO.md','BOUNDS_REVIEW_KO.md','ASSOCIATION_ALL8.jpg']
    assert all((DOC/f).exists() for f in required)
    assert not (DOC/'PUBLICATION_MANIFEST.json').exists()
    inputs=[]
    for path in DOC.glob('*.json'):
        for b in bindings(read(path)):
            p=Path(b['path'])
            if not p.is_absolute(): p=ROOT/p
            assert p.is_file(),str(p)
            assert hashlib.sha256(p.read_bytes()).hexdigest() == b['sha256'],str(p)
            inputs.append(dict(path=str(p),sha256=b['sha256']))
    bound=read(DOC/'CANDIDATE_BOUNDS.json')
    assert bind(PREV/'RESULTS.json') == bound['original_primary_result']
    assert bound['old_oracle_numeric_parity_checks'] == 48
    assert bound['bounds']['DIVERSE251_mean3seeds']['natural_T_tail_guard_impossible']
    assert not bound['bounds']['R0']['full_goal_excluded_with_this_fixed_pool']
    frames=list(csv.DictReader((DOC/'CANDIDATE_BOUND_ROWS.csv').open()))
    margin=list(csv.DictReader((DOC/'SELECTOR_MARGIN_ROWS.csv').open()))
    assert len(frames)==3114 and len(margin)==1557
    assert len({(r['model'],r['id'],r['oracle']) for r in frames})==3114
    assert len({(r['model'],r['id']) for r in margin})==1557
    groups=read(ROOT/'data/pallet/results/pallet_pose_stable_improvement_20261001_v1/EVAL_GROUPS.json')
    numeric_checks=0
    for model,pops in bound['summaries'].items():
        for pop,oracles in pops.items():
            for oracle,summary in oracles.items():
                chosen=[r for r in frames if r['model']==model and r['id'] in groups[pop] and r['oracle']==oracle]
                assert len(chosen)==len(groups[pop])
                for metric,column in [('translation_cm','T_cm'),('rotation_deg','R_deg')]:
                    values=np.array([float(r[column]) for r in chosen])
                    assert np.isfinite(values).all() and (values>=0).all()
                    for stat,q in [('median',.5),('P90',.9)]:
                        assert abs(float(np.quantile(values,q))-summary['conditional'][metric][stat])<1e-7
                        numeric_checks+=1
    pictures=[]
    for path in sorted(DOC.rglob('*')):
        if path.suffix.lower() in ('.png','.jpg'):
            with Image.open(path) as im:
                im.load();pictures.append(dict(file=bind(path),size=list(im.size)))
    assert len(pictures)==3
    codes=[]
    for path in HERE.glob('*.py'):
        ast.parse(path.read_text());codes.append(bind(path))
    # GitHub must have every relative Markdown target. Local private prior files
    # cannot make a broken public link appear valid.
    links=[]
    for path in DOC.glob('*.md'):
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',path.read_text()):
            if target.startswith(('http://','https://','mailto:','#')):continue
            target=target.split('#',1)[0]
            resolved=(path.parent/target).resolve()
            if resolved == DOC/'PUBLICATION_MANIFEST.json':continue
            if resolved.is_relative_to(DOC) or resolved.is_relative_to(HERE):
                assert resolved.exists(),(path.name,target)
            else:
                relative=resolved.relative_to(ROOT)
                assert (CHECKOUT/relative).exists(),('UNPUBLISHED_LINK',path.name,target)
            links.append(dict(document=path.name,target=target))
    validation=dict(status='PASS',primary_verdict_unchanged=True,goal_achieved=False,
        type='Completed diagnostics and publication checks; performance goal remains unmet',
        candidate_rows=len(frames),margin_rows=len(margin),csv_numeric_checks=numeric_checks,
        verified_input_binding_count=len(inputs),inputs=inputs,markdown_links=links,
        pictures=pictures,python_AST_checked=codes,
        visual_review='Root inspected both final charts and all-eight association contact sheet; data auditor also inspected all original images.',
        publication_base=BASE,new_fits=0,image_forwards=0)
    save(DOC/'VALIDATION.json',validation)
    paths=sorted([ROOT/'.gitignore',ROOT/'readme.md',
        *[p for p in DOC.rglob('*') if p.is_file()],*[p for p in HERE.rglob('*.py') if p.is_file()]])
    manifest=dict(base_commit=BASE,goal_achieved=False,performance_verdict='STABLE_JOINT_IMPROVEMENT_NOT_ESTABLISHED',
        public_files_including_this_manifest=len(paths)+1,images=len(pictures),files=[bind(p) for p in paths],
        excluded='Model weights, raw feature NPZ and original input datasets stay local; bindings allow audit in the originating workspace.',
        verified_before_copy=bind(DOC/'VALIDATION.json'))
    save(DOC/'PUBLICATION_MANIFEST.json',manifest)
    paths.append(DOC/'PUBLICATION_MANIFEST.json')
    for path in paths:
        target=CHECKOUT/path.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        assert target.read_bytes()==path.read_bytes()
    save(RAW/'PUBLICATION_COPY.json',dict(base=BASE,files=[bind(p) for p in paths],checkout=str(CHECKOUT)))
    print(json.dumps(dict(status='READY_FOR_COMMIT',public_files=len(paths),images=len(pictures),
        numeric_checks=numeric_checks,links=len(links),input_bindings=len(inputs)),ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
