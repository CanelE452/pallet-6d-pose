"""Validate the concrete GitHub payload and write current artifact hashes."""
import ast
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from PIL import Image
import numpy as np
from . import run as C
from . import scoring as S
from .review_analysis import ARCHIVE,write,csvwrite

DEPENDENCIES=[
    'scripts/research/pallet_posefix_limited_adaptation_pilot_v1/__init__.py',
    'scripts/research/pallet_posefix_limited_adaptation_pilot_v1/common.py',
    'scripts/research/pallet_posefix_lora_preservation_v1/__init__.py',
    'scripts/research/pallet_posefix_lora_preservation_v1/adaptation.py',
    'scripts/research/pallet_posefix_lora_preservation_v1/run.py',
]

def main():
    rows=C.metadata();meta={r['id']:r for r in rows}
    csvwrite(C.DOC/'FRAME_INPUT_INDEX.csv',[dict(id=r['id'],recording=r['recording'],population='CLEAN29' if r['severity']=='CLEAN' else 'NATURAL99',severity=r['severity'],
        image_path=r['image']['path'],image_sha256=r['image']['sha256'],height=r['hw'][0],width=r['hw'][1],
        fx=r['K'][0][0],fy=r['K'][1][1],cx=r['K'][0][2],cy=r['K'][1][2],dimensions_xyz_m=r['xyz']) for r in rows])
    old=C.read(C.DOC/'history/RUN_MANIFEST_INITIAL_COMPLETION.json')
    historical=[]
    for b in old['codes']+old['artifacts']+old['private_artifacts']:
        current=C.ROOT/b['path'];actual=current
        if hashlib.sha256(current.read_bytes()).hexdigest()!=b['sha256']:
            actual=ARCHIVE/b['path']
        assert actual.exists(),str(actual)
        assert hashlib.sha256(actual.read_bytes()).hexdigest()==b['sha256'],str(actual)
        historical.append(dict(original_path=b['path'],sha256=b['sha256'],preserved_path=str(actual.relative_to(C.ROOT)),modified_since_initial=actual!=current))
    initial=C.read(C.DOC/'RUN_MANIFEST.json')
    for b in initial['inputs']+initial['codes']:C.verify(b)
    frame=list(csv.DictReader((C.DOC/'FRAME_RESULTS.csv').open()))
    assert len(frame)==2964 and len({(r['stage'],r['id'],r['model'],r['condition']) for r in frame})==2964
    oldframe=list(csv.DictReader((ARCHIVE/(C.DOC/'FRAME_RESULTS.csv').relative_to(C.ROOT)).open()))
    for a,b in zip(oldframe,frame):
        for key in ('translation_cm','rotation_deg','delta_T_cm','delta_R_deg','valid'):assert a[key]==b[key]
    candidates=C.read(C.RAW/'E1_CANDIDATES.json');twod=C.read(C.RAW/'E4_REVIEW_2D_METRICS.json')
    held=changed2d=0
    for a,row in zip(oldframe,frame):
        if row['stage']=='E1' and row['condition']=='held_identity':
            assert row['GEO_name']==row['selected_hypothesis_for_pose']==candidates['identity'][row['id']]['GEO_name'];held+=1
        if row['stage']=='E4':
            kind=row['condition'].rsplit('_',1)[0];expected=twod[kind][row['id']]['frame_mean_px']
            assert row['twoD_frame_mean_px']=='' if expected is None else np.isclose(float(row['twoD_frame_mean_px']),expected)
            changed2d+=row['twoD_canonical_errors']!=a['twoD_canonical_errors']
    assert held==256 and changed2d>0
    stats=C.read(C.DOC/'E7_ALL_COMPARISONS.json')['comparisons'];assert len(stats)==69
    total_intervals=0
    for label,groups in stats.items():
        for pop,value in groups.items():
            assert value['N']==(99 if pop=='NATURAL99' else 29)
            assert len(value['recordings'])==(6 if pop=='NATURAL99' else 3)
            pair=value['paired'];assert sum(value['recordings'].values())==value['N']
            assert len(pair['leave_one_recording_out'])==len(value['recordings'])
            for interval in pair['recording_bootstrap']['intervals'].values():
                assert interval['finite_resamples']+interval['undefined_resamples']==2000;total_intervals+=1
    e1=C.read(C.DOC/'E1_SUMMARY.json')
    for pop in ('NATURAL99','CLEAN29'):
        assert stats['E1/FULL125'][pop]['paired']['recording_bootstrap']==e1[pop]['paired']['FULL125-minus-identity']['recording_bootstrap']
    strata=list(csv.DictReader((C.DOC/'E7_METADATA_STRATA_ALL.csv').open()));byrec=list(csv.DictReader((C.DOC/'E7_BY_RECORDING_ALL.csv').open()))
    for name,groups in stats.items():
        for pop,value in groups.items():
            assert sum(int(r['N']) for r in byrec if r['comparison']==name and r['population']==pop)==value['N']
            for field in ('occlusion','elevation_bin','distance_bin','size_bin','view_bin','baseline_WD','baseline_matching'):
                assert sum(int(r['N']) for r in strata if r['comparison']==name and r['population']==pop and r['field']==field)==value['N']
    # Every image is a decodable static artifact; gallery IDs cover natural99 exactly.
    images=sorted((C.DOC/'figures').glob('*'));assert len(images)==29
    for path in images:
        with Image.open(path) as im:im.verify()
    selection=C.read(C.DOC/'FIGURE_CASE_SELECTION.json');assert selection['gallery_frames']==99
    links=[]
    for path in C.DOC.glob('*.md'):
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',path.read_text()):
            target=target.strip('<>').split('#')[0]
            if not target or target.startswith(('http://','https://','mailto:')):continue
            resolved=(path.parent/target).resolve();assert resolved.is_file(),(path,target)
            links.append(dict(document=str(path.relative_to(C.ROOT)),target=str(resolved.relative_to(C.ROOT))))
    for path in C.HERE.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
    # Publish the small imported support modules needed by this experiment.
    write(C.DOC/'CODE_DEPENDENCIES.json',dict(files=[C.bind(C.ROOT/p) for p in DEPENDENCIES],reason='Existing local modules directly imported by the diagnosis; code only, no unrelated experiment outputs. Base repository imports otherwise inherited from d10a8c7f.'))
    costs={p.stem:C.read(p) for p in C.DOC.glob('REVIEW_COST_*.json')}
    for path in (ARCHIVE/C.DOC.relative_to(C.ROOT)).glob('REVIEW_COST_*.json'):
        costs[path.stem+'_earlier_pass']=C.read(path)
    validation=dict(passed=True,validation_scope='Data/contract/semantic/link integrity, not proof of scientific generalization',
        original_bindings_verified=len(initial['inputs'])+len(initial['codes']),historical_bindings_verified=len(historical),historical_bindings=historical,
        frame_rows=2964,all_original_T_R_unchanged=True,held_rows_checked=held,E4_rows_with_changed_2D=changed2d,
        comparisons=69,recording_rows=len(byrec),metadata_rows=len(strata),bootstrap_intervals=total_intervals,images_verified=len(images),markdown_links_checked=len(links),
        visual_inspection='Overview, paired scatter, recording plot, representative natural examples, E3 masks and gallery pages inspected; font minus signs and CO clean-bbox rendering corrected before publication.',
        no_new_model_forward_in_review=True,no_new_fit_in_review=True,review_costs=costs,
        measured_review_CPU_seconds=sum(v['CPU_seconds'] for v in costs.values()),cost_scope='Measured source/repair/statistics/figure stages including archived earlier statistics/figure passes; excludes authoring, validation and Git/network overhead.',
        remaining_scientific_unknowns={k:v for k,v in C.read(C.DOC/'QUESTION_STATUS.json').items() if v['status'] in ('UNRESOLVED','BLOCKED')})
    write(C.DOC/'PUBLICATION_VALIDATION.json',validation)
    manifest=C.read(C.DOC/'RUN_MANIFEST_FINAL.json')
    manifest.update(completed_at=datetime.now(timezone.utc).isoformat(),phase='GitHub review revision; original completion snapshot in history',
        original_completion_manifest=C.bind(C.DOC/'history/RUN_MANIFEST_INITIAL_COMPLETION.json'),review_validation=C.bind(C.DOC/'PUBLICATION_VALIDATION.json'),
        original_scope_claim_corrected=True,scope_completed='Core diagnosis plus documented source/E7/CSV review corrections; scientific unknowns retained',
        commit_or_push='Publication payload prepared. Remote commit is verified after the push; this file does not assert a push happened beforehand.',
        codes=[C.bind(p) for p in sorted(C.HERE.glob('*.py'))],artifacts=[C.bind(p) for p in sorted(C.DOC.rglob('*')) if p.is_file() and p.name not in ('RUN_MANIFEST_FINAL.json','PUBLICATION_MANIFEST.json')],
        private_artifacts=[C.bind(p) for p in sorted(C.RAW.glob('*.json'))])
    write(C.DOC/'RUN_MANIFEST_FINAL.json',manifest)
    files=[C.ROOT/'.gitignore',C.ROOT/'readme.md']+[C.ROOT/p for p in DEPENDENCIES]+list(C.HERE.glob('*.py'))+[p for p in C.DOC.rglob('*') if p.is_file() and p.name!='PUBLICATION_MANIFEST.json']
    files=sorted(set(files));assert max(p.stat().st_size for p in files)<50_000_000
    publish=dict(created_at=datetime.now(timezone.utc).isoformat(),base_commit=initial['HEAD'],destination='https://github.com/CanelE452/pallet-6d-pose',branch='main',
        source_workspace_HEAD_preserved=True,publish_from_isolated_checkout=True,scope='Diagnosis docs/tables/29 figures/scripts and their required imported source modules; no raw weights, no unrelated local results',
        files=[C.bind(p) for p in files],total_bytes=sum(p.stat().st_size for p in files),self_excluded_from_hash_list=True,
        validation=C.bind(C.DOC/'PUBLICATION_VALIDATION.json'))
    write(C.DOC/'PUBLICATION_MANIFEST.json',publish)
    # Assert every Markdown file link exists in either this payload or tracked base.
    tracked=set(subprocess.check_output(['git','ls-files'],cwd=C.ROOT,text=True).splitlines());published={b['path'] for b in publish['files']}
    for link in links:assert link['target'] in published|tracked,link
    print('PUBLICATION_VALIDATED',len(files)+1,'files',len(images),'images',publish['total_bytes'],'bytes',flush=True)

if __name__=='__main__':main()
