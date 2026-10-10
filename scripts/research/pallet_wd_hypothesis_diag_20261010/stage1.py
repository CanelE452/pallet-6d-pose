"""Stage0 actual parity and Stage1 GT-parity diagnostic, no verdict/rule trial."""
from collections import Counter
import copy
import time
import cv2
import numpy as np
from . import common as C
from . import inputs as I
from . import solver
from .rules import select_rule

def _scored(locked,truth,diagnostics,seal_sha):
    pose=C.pose_api();result=copy.deepcopy(locked);fid=locked['id'];choices=locked['selection']
    oracle_hyp=diagnostics['hypOracle']
    oracle=choices['candidates'][oracle_hyp]['actual_pose']
    S0=pose.metric((fid,choices['actual_pose'],truth));ORACLE=pose.metric((fid,oracle,truth))
    result.pop('selection')
    result.update(pose={'S0':S0,'ORACLE':ORACLE},S0=S0,ORACLE=ORACLE,
        actual_pose={'S0':choices['actual_pose'],'ORACLE':oracle},hypS0=choices['hyp'],
        candidates=choices['candidates'],formal_score_gap_px=choices['formal_score_gap_px'],
        formal_gap_semantics=choices['formal_gap_semantics'],simplechoice=choices['simplechoice'],
        S0_selector_status=choices['selector_status'],selection_status=choices['status'],
        source_flag=locked['source_flag'],pose_symmetry_order=truth['order'],
        reference_seal_sha256=seal_sha,oracle_diagnostic_only=True,
        inference_reference_inputs=False,oracle_definition='GT parity supplied; same predicted coordinates and8-corner final solver; no min-error candidate search',
        **diagnostics)
    return result

def _verify_fixed_inputs():
    for b in C.read(C.DOC/'INPUT_AUDIT.json')['bindings']:
        root=C.ROOT if b['owner']=='published_worktree' else C.SOURCE
        assert C.sha(root/b['path'])==b['sha256'], ('Fixed input changed',b['path'])

def _select_population(population,loader,started):
    inputs=loader();locked=[]
    for i,r in enumerate(inputs):
        row=copy.deepcopy(r)
        row['selection']=select_rule('S0',r['qFinal'],r['fixed_metadata']['K'],r['fixed_metadata']['dimensions_pnp_WH_D_m'],r['source_flag'],r['prediction_support'])
        locked.append(row)
        if (i+1)%250==0 or i+1==len(inputs):
            print('STAGE1_SELECTION',population,i+1,len(inputs),'seconds',round(time.monotonic()-started,1),flush=True)
    path=C.DOC/f'STAGE1_SELECTIONS_{population}.jsonl.gz';C.write_rows(path,locked)
    C.verify_core()
    seal=dict(status='SEALED_S0_AND_BOTH_CANDIDATES_BEFORE_THIS_POPULATION_REFERENCE_VALUE_SCORING',
        population=population,rows=len(locked),selection_path=path.name,selection_sha256=C.sha(path),
        source_lock_path=C.source_lock_path().name,source_lock_sha256=C.sha(C.source_lock_path()),input_audit_sha256=C.sha(C.DOC/'INPUT_AUDIT.json'),
        schema_audit_before_seal='Only reference field presence/cardinality/review statuses; no pose reference values were consumed by selectors.',
        selected_inputs='Whitelisted frozen qFinal/K/fixed registry dimensions/support/source flag only',
        method_reference_access=False,actual_original_F_calls=len(locked),stage2_rules_executed=0,
        model_forwards=0,training_updates=0,elapsed_seconds=time.monotonic()-started)
    seal_path=C.DOC/f'STAGE1_SELECTION_SEAL_{population}.json';C.write(seal_path,seal)
    return path,seal_path,len(locked)

def run():
    assert C.read(C.DOC/'INPUT_AUDIT.json')['status']=='PASS'
    C.verify_core()
    _verify_fixed_inputs()
    for name in ('STAGE1_SELECTIONS_REAL.jsonl.gz','STAGE1_SELECTIONS_SYNTH.jsonl.gz','STAGE1_SELECTIONS_AUX.jsonl.gz','STAGE1_SELECTION_SEAL.json',
                 'STAGE1_SELECTION_SEAL_REAL.json','STAGE1_SELECTION_SEAL_SYNTH.json','STAGE1_SELECTION_SEAL_AUX.json',
                 'STAGE0_PARITY.json','STAGE1_ROWS_REAL.jsonl.gz','STAGE1_ROWS_SYNTH.jsonl.gz','STAGE1_ROWS_AUX.jsonl.gz'):
        assert not (C.DOC/name).exists(), ('Preserve existing output',name)
    cv2.setNumThreads(1);pose=C.pose_api();solver.configure(pose)
    started=time.monotonic();counts=Counter();paths={};seals={}
    paths['REAL'],seals['REAL'],counts['REAL']=_select_population('REAL',I.real_inputs,started)
    seal_sha=C.sha(seals['REAL'])
    # Stage0 must actually PASS before any synthetic/auxiliary solver call.
    # REAL choices have already been sealed; now read REAL references/errors.
    old={(r['id'],r['method'],r['seed']):r for r in C.rows(C.REAL) if r['method'] in C.METHODS}
    truth,diagnostics=I.truth('REAL')
    real=[];parity=[]
    for locked in C.rows(paths['REAL']):
        row=_scored(locked,truth[locked['id']],diagnostics[locked['id']],seal_sha)
        previous=old[(row['id'],row['method'],row['seed'])]
        a=row['pose']['S0'];b=previous['pose']
        available=a['available']==b['available'];hyp=row['hypS0']==previous['final_hypothesis']
        deltaT=None if not a['available'] else abs(a['translation_cm']-b['translation_cm'])
        deltaR=None if not a['available'] else abs(a['rotation_deg']-b['rotation_deg'])
        ok=available and hyp and (not a['available'] or (deltaT<=.01 and deltaR<=.01))
        parity.append(dict(id=row['id'],method=row['method'],seed=row['seed'],hypothesis_exact=hyp,
                           availability_exact=available,delta_translation_cm=deltaT,delta_rotation_deg=deltaR,PASS=ok))
        real.append(row)
    status='PASS' if all(r['PASS'] for r in parity) else 'FAIL_STOP'
    receipt=dict(status=status,rows=len(parity),actual_original_F_calls=counts['REAL'],
        hypothesis_exact=sum(r['hypothesis_exact'] for r in parity),availability_exact=sum(r['availability_exact'] for r in parity),
        translation_cm_tolerance=.01,rotation_deg_tolerance=.01,
        max_translation_cm_delta=max((r['delta_translation_cm'] or 0 for r in parity)),
        max_rotation_deg_delta=max((r['delta_rotation_deg'] or 0 for r in parity)),
        simple_same_as_official=sum(r['simplechoice']==r['hypS0'] for r in real),simple_comparison_rows=len(real),
        source_lock_path=C.source_lock_path().name,source_lock_sha256=C.sha(C.source_lock_path()),selection_seal_sha256=seal_sha,
        failures=[r for r in parity if not r['PASS']],per_row=parity)
    C.write(C.DOC/'STAGE0_PARITY.json',receipt)
    assert status=='PASS', 'Parity failed: stop; do not publish diagnostic scores or try alternatives'
    C.write_rows(C.DOC/'STAGE1_ROWS_REAL.jsonl.gz',real)
    for population in ('SYNTH','AUX'):
        loader=I.synth_inputs if population=='SYNTH' else I.aux_inputs
        paths[population],seals[population],counts[population]=_select_population(population,loader,started)
        seal_sha=C.sha(seals[population])
        truth,diagnostics=I.truth(population)
        C.write_rows(C.DOC/f'STAGE1_ROWS_{population}.jsonl.gz',
            (_scored(r,truth[r['id']],diagnostics[r['id']],seal_sha) for r in C.rows(paths[population])))
        print('STAGE1_SCORED',population,counts[population],flush=True)
    C.verify_core()
    _verify_fixed_inputs()
    C.write(C.DOC/'STAGE1_SELECTION_SEAL.json',dict(status='ALL_POPULATIONS_SEALED_BEFORE_THEIR_REFERENCE_SCORING',
        populations={p:dict(rows=counts[p],selection_path=paths[p].name,selection_sha256=C.sha(paths[p]),
                           seal_path=seals[p].name,seal_sha256=C.sha(seals[p])) for p in counts},
        execution_order='REAL selection/seal -> REAL parity PASS -> SYNTH selection/seal/reference -> AUX selection/seal/reference',
        source_lock_path=C.source_lock_path().name,source_lock_sha256=C.sha(C.source_lock_path()),stage0_parity_sha256=C.sha(C.DOC/'STAGE0_PARITY.json')))
    seal_sha=C.sha(C.DOC/'STAGE1_SELECTION_SEAL.json')
    C.write(C.DOC/'STAGE1_EXECUTION.json',dict(status='COMPLETE_STAGE1_DIAGNOSTIC_ONLY_NO_VERDICT',
        actual_original_F_calls=dict(counts),actual_original_F_calls_total=sum(counts.values()),
        source_lock_path=C.source_lock_path().name,source_lock_sha256=C.sha(C.source_lock_path()),selection_seal_sha256=seal_sha,
        reference_only_SYNTH_margin_solver_calls=1985*2,
        outputs={p:C.binding(C.DOC/f'STAGE1_ROWS_{p}.jsonl.gz',C.ROOT) for p in counts},
        stage2_rules_executed=0,network_inference=0,training_updates=0,
        elapsed_seconds=time.monotonic()-started,all_fixed_inputs_preserved=True))
    print('STAGE1 COMPLETE',dict(counts),round(time.monotonic()-started,1),flush=True)

if __name__=='__main__':run()
