"""Publish the certified asymmetric-loss results with actual evaluation status."""
from pathlib import Path
import ast
import re
import shutil
import subprocess
from urllib.parse import unquote
from . import common as C

CHECKOUT=Path('/tmp/pallet-pose-github-review-20260930')
BASE='935e45c69a5db666da2ca1404d7fc9fdfe8c3ad0'


def main():
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=CHECKOUT,text=True).strip()
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CHECKOUT,text=True).strip()
    assert head==BASE==subprocess.check_output(['git','rev-parse','origin/main'],cwd=CHECKOUT,text=True).strip()
    for name in ('PREFIT_REVIEW.json','TRAIN_CONVERGENCE.json','SOURCE_VAL_VERIFICATION.json','PUBLIC_REVIEW.json'):
        value=C.read(C.DOC/name);assert value['complete'] and value['PASS'],name
    training=C.read(C.DOC/'TRAINING_COMPLETE.json')
    assert training['complete'] and training['all_certified'] and training['fit_count']==4
    assert training['total_objective_calls']==sum(C.read(C.DOC/f'FIT_{m}.json')['objective_calls'] for m in C.MODEL_NAMES)
    assert training['total_iterations']==sum(C.read(C.DOC/f'FIT_{m}.json')['iterations'] for m in C.MODEL_NAMES)
    gate=C.read(C.DOC/'SOURCE_VAL_GATE.json')
    assert gate['complete'] and gate['checks_total']==45
    real=None
    if gate['PASS']:
        assert gate['checks_passed']==45 and not (C.DOC/'REAL_EVALUATION_NOT_RUN.json').exists()
        real=C.read(C.DOC/'REAL_RESULTS.json');verified=C.read(C.DOC/'REAL_VERIFICATION.json')
        assert real['complete'] and verified['complete'] and verified['PASS']
        assert real['full_frame_rows']==2249 and len(real['models'])==13
        assert real['real_stability_contract']=='matched_intervention_AND_original_SINGLE251_stability'
    else:
        not_run=C.read(C.DOC/'REAL_EVALUATION_NOT_RUN.json')
        assert not_run['complete'] and not not_run['method_success'] and not not_run['goal_complete']
        assert len(not_run['absence_checks'])==12 and all(not_run['absence_checks'].values())
        for path in not_run['absence_checks']:assert not (C.ROOT/path).exists(),path
    stable=bool(real is not None and real['stability']['PASS'])
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
        status='REAL_STABILITY_PASS' if stable else 'REAL_STABILITY_FAIL' if real is not None else 'SOURCE_GATE_FAILED_REAL_NOT_RUN',new_fit_attempts=4,rejected_fit_attempts=0,certified_models=4,
        objective_calls=training['total_objective_calls'],optimizer_iterations=training['total_iterations'],source_VAL_evaluated=True,source_VAL_gate_PASS=gate['PASS'],
        source_VAL_checks_passed=gate['checks_passed'],source_VAL_checks_total=45,
        learned_real_evaluated=real is not None,current_method_real_T_R_effect_measured=real is not None,
        new_source_VAL_routes=4096,new_learned_real_routes=692 if real is not None else 0,real_frame_metric_rows=2249 if real is not None else 0,new_image_forwards=0,new_PnP=0,
        input_feature_dim=271, added_direction_features=0, underprediction_cost=2., overprediction_cost=1., changed_factor='Add fixed unit underprediction Huber term; all271 inputs/solver/targets/runtime/gates unchanged',
        image_count=sum(p.suffix in ('.png','.jpg') for p in files),actual_RGB_frames=6,
        figure_scope='Training/source/actual real metrics and same6realRGB dimensions with4models' if real is not None else 'Training/source metrics and6historical realR0 RGB examples with dimensions; no current learned real prediction',
        method_success=stable,stable_joint_improvement_achieved=stable,goal_complete=False,
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
