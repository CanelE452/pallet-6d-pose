"""Publication-gated fixed S1/S2 execution; new choices sealed before scoring."""
import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import time
import cv2
from . import common as C
from . import inputs as I
from . import solver
from . import rules
from . import statistics as M
from . import verdict as V
from . import stage1 as S
from .stage1_resume import minimal_truth

@contextmanager
def solver_counter():
    counts=Counter();original={name:getattr(cv2,name) for name in ('solvePnP','solvePnPRefineLM')}
    for name,fn in original.items():
        def call(*args,_name=name,_fn=fn,**kwargs):
            counts[_name]+=1
            return _fn(*args,**kwargs)
        setattr(cv2,name,call)
    try:yield counts
    finally:
        for name,fn in original.items():setattr(cv2,name,fn)

def gate(rule,population):
    assert C.read(C.DOC/'STAGE1_PUBLICATION.json')['status']=='PASS', 'Publish/review Stage1 first'
    method=C.read(C.DOC/'METHOD_LOCK.json');source=C.read(C.DOC/'SOURCE_LOCK_STAGE2.json')
    assert method['status'].startswith('LOCKED') and source['status'].startswith('LOCKED')
    assert source['method_lock_sha256']==C.sha(C.DOC/'METHOD_LOCK.json'), 'Method/source lock binding mismatch'
    assert 'new_core' in source
    for key in ('new_core','original_core'):
        for b in source.get(key,[]):
            owner=C.SOURCE if b.get('owner')=='historical_source' or key=='original_core' else C.ROOT
            assert C.sha(owner/b['path'])==b['sha256'], ('Stage2 source changed',b['path'])
    C.verify_core();S._verify_fixed_inputs()
    assert C.read(C.DOC/'STAGE0_PARITY.json')['status']=='PASS'
    assert rule in ('S1','S2','S3') and population in ('REAL','SYNTH')
    if population=='REAL' and rule in ('S1','S2'):
        synthetic=C.read(C.DOC/f'RESULTS_{rule}_SYNTH.json')
        assert V.synth_gate(synthetic['primary'])['run_real'], 'SYNTH WORSENED: REAL execution forbidden'
    return method,source

def stage1_baseline(population):
    return [{**r,'pose':r['pose']['S0'],'hyp':r['hypS0']} for r in C.rows(C.DOC/f'STAGE1_ROWS_{population}.jsonl.gz')]

def save_comparison(result,rule,population):
    result=dict(result)
    if population=='SYNTH' and rule in ('S1','S2'):result['synthetic_gate']=V.synth_gate(result['primary'])
    C.write(C.DOC/f'RESULTS_{rule}_{population}.json',result)
    for field in ('metrics','paired','failures'):
        C.write(C.DOC/f'{field.upper()}_{rule}_{population}.json',dict(population=population,rule=rule,**{field:result[field]}))
    return result

def run(rule,population):
    assert rule in ('S1','S2')
    gate(rule,population)
    names=[f'STAGE2_SELECTIONS_{rule}_{population}.jsonl.gz',f'STAGE2_SELECTION_SEAL_{rule}_{population}.json',
           f'STAGE2_ROWS_{rule}_{population}.jsonl.gz',f'RESULTS_{rule}_{population}.json']
    assert not any((C.DOC/name).exists() for name in names), 'Preserve prior rule execution'
    started=time.monotonic();cv2.setNumThreads(1);pose=C.pose_api();solver.configure(pose)
    records=I.real_inputs() if population=='REAL' else I.synth_inputs()
    expected=3828 if population=='REAL' else 23820
    assert len(records)==expected
    choices=[]
    with solver_counter() as counts:
        for i,r in enumerate(records):
            before=C.digest(r['qFinal']);row=copy.deepcopy(r)
            row['selection']=rules.select_rule(rule,r['qFinal'],r['fixed_metadata']['K'],
                r['fixed_metadata']['dimensions_pnp_WH_D_m'],r['source_flag'],r['prediction_support'])
            assert before==C.digest(row['qFinal'])
            choices.append(row)
            if (i+1)%1000==0 or i+1==len(records):print('STAGE2_SELECT',rule,population,i+1,len(records),round(time.monotonic()-started,1),flush=True)
    choice_path=C.DOC/names[0];C.write_rows(choice_path,choices)
    seal_path=C.DOC/names[1]
    C.write(seal_path,dict(status='SEALED_RULE_CHOICES_BEFORE_REFERENCE_SCORING',rule=rule,population=population,
        rows=len(choices),choice_path=choice_path.name,choice_sha256=C.sha(choice_path),
        method_lock_sha256=C.sha(C.DOC/'METHOD_LOCK.json'),source_lock_sha256=C.sha(C.DOC/'SOURCE_LOCK_STAGE2.json'),
        solver_calls=dict(counts),references_consumed_by_selector=False,human_visibility_inputs=False,
        coordinate_changes=0,SubPix_calls=0,model_forwards=0,training_updates=0))
    # All choices are already public numeric evidence before any new metric call.
    truth,_=minimal_truth(population,with_margin=False)
    changed=[];seal_sha=C.sha(seal_path)
    for locked in C.rows(choice_path):
        r=copy.deepcopy(locked);selection=r.pop('selection');fid=r['id']
        r.update(pose=pose.metric((fid,selection['actual_pose'],truth[fid])),actual_pose=selection['actual_pose'],
            hyp=selection['hyp'],fallback=selection['fallback'],subset_indices=selection['subset_indices'],
            selection_status=selection['status'],candidates=selection['candidates'],
            candidate_scores=selection['candidate_scores'],reference_seal_sha256=seal_sha,
            inference_reference_inputs=False)
        if 'visibility' in selection:r['visibility']=selection['visibility']
        changed.append(r)
    path=C.DOC/names[2];C.write_rows(path,changed)
    changed=list(C.rows(path));baseline=stage1_baseline(population)
    result=M.compare(baseline,changed,'REAL_DEV' if population=='REAL' else 'SYNTH_HELDOUT',rule)
    result=save_comparison(result,rule,population)
    gate(rule,population)
    C.write(C.DOC/f'EXECUTION_{rule}_{population}.json',dict(status='COMPLETE',rule=rule,population=population,
        rows=len(changed),solver_calls=dict(counts),choice_sha256=C.sha(choice_path),selection_seal_sha256=seal_sha,
        scored_rows_sha256=C.sha(path),elapsed_seconds=time.monotonic()-started,
        original_coordinates_identical=True,SubPix_calls=0,network_forwards=0,training_updates=0,
        reference_margin_solver_calls=0,source_lock_sha256=C.sha(C.DOC/'SOURCE_LOCK_STAGE2.json')))
    print('STAGE2 COMPLETE',rule,population,result.get('synthetic_gate'),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rule',required=True,choices=('S1','S2'));p.add_argument('--population',required=True,choices=('REAL','SYNTH'))
    a=p.parse_args();run(a.rule,a.population)
