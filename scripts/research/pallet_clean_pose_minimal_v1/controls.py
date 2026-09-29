"""Missing same-GEO controls using only frozen historical predictions/candidates.

Run ``freeze`` and ``score`` in separate processes. The former has the reused
reference-read guard; the latter indexes cached candidate metrics by names
already sealed. Original files, checkpoints, fit ledger, and results stay intact.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from scripts.research.pallet_clean_to_pose_transfer_v1 import common as OLD
from scripts.research.pallet_clean_to_pose_transfer_v1 import selector_compat as ENGINE
from . import common as C

MISSING={'R0':'R0','OLD_REF':'OLD_REF','RAW_CLEAR_S42':'CLEAN_RAW_CLEAR','REF_CLEAR_S42':'CLEAN_REF_CLEAR'}
LOCK=C.DOC/'SAME_GEO_CONTROLS_LOCK.json'
RESULT=C.DOC/'CONTROL_RESULTS.json'


def configure(alias):
    assert alias in MISSING
    ENGINE.C=C;ENGINE.SEED=42;ENGINE.ARM=MISSING[alias]
    ENGINE.PRIVATE=C.RAW/'controls'/alias
    ENGINE.LOCK=C.DOC/f'CONTROL_{alias}_LOCK.json'
    ENGINE.RESULT=C.DOC/f'CONTROL_{alias}_RESULTS.json'
    return dict(arm=ENGINE.ARM,private=ENGINE.PRIVATE,lock=ENGINE.LOCK,result=ENGINE.RESULT)


def old_pair_lock(seed):
    path=OLD.DOC/f'SELECTOR_PAIR_LOCK_S{seed}.json'
    lock=OLD.read(path)
    assert lock['seed']==seed and lock['same_frozen_old_GEO']
    for binding in lock['sources']:OLD.verify(binding)
    for child in lock['arms'].values():
        OLD.verify(child['lock'])
        value=OLD.read(OLD.ROOT/child['lock']['path'])
        assert value['old_scorer']==lock['scorer']
        for binding in value['files']+value['sources']:OLD.verify(binding)
    return path,lock


def freeze():
    if LOCK.exists():
        verify_lock();print('SAME_GEO_CONTROLS_ALREADY_LOCKED',flush=True);return
    start=time.monotonic();children={};scorer=None
    for alias in MISSING:
        configured=configure(alias);ENGINE.freeze();child=ENGINE.verify_lock()
        assert child['arm']==configured['arm'] and child['seed']==42
        if scorer is not None:assert scorer==child['old_scorer']
        scorer=child['old_scorer']
        children[alias]=dict(arm=child['arm'],lock=C.bind(configured['lock']),counts=child['counts'])
    reused={}
    for seed in (42,43):
        path,pair=old_pair_lock(seed);assert pair['scorer']==scorer
        reused[str(seed)]=C.bind(path)
    sources=[value['lock'] for value in children.values()]+list(reused.values())
    sources += [C.bind(Path(__file__)),C.bind(Path(ENGINE.__file__)),C.bind(Path(C.__file__)),C.bind(C.DOC/'START.json')]
    C.save(LOCK,dict(created_at=C.now(),children=children,reused_OCC_pairs=reused,
        same_frozen_scorer=scorer,decision_guard_active=True,
        new_control_selection_locked_before_scoring=True,new_controls_gt_or_metric_reads=False,
        current_primary='same128 frames / natural99, unchanged candidate generation and final candidate poses',
        retrospective_status='These missing controls are added after previously viewed DEV results; not retroactively preregistered.',
        new_student_fits=0,new_selector_fits=0,optimizer_updates=0,GPU_seconds=0,
        sources=sources,seconds=time.monotonic()-start),True)
    print('SAME_GEO_CONTROLS_LOCKED',{k:v['counts'] for k,v in children.items()},flush=True)


def verify_lock():
    result=C.read(LOCK)
    assert result['new_control_selection_locked_before_scoring'] and not result['new_controls_gt_or_metric_reads']
    for binding in result['sources']:C.verify(binding)
    for alias in MISSING:
        configure(alias);child=ENGINE.verify_lock()
        assert child['old_scorer']==result['same_frozen_scorer']
    return result


def contrast_pairs():
    methods=(*MISSING,'RAW_OCC_S42','REF_OCC_S42','RAW_OCC_S43','REF_OCC_S43')
    pairs=[(method+'_D9',method+'_GEO') for method in methods]
    pairs += [('R0_GEO',method+'_GEO') for method in methods if method!='R0']
    pairs += [('RAW_CLEAR_S42_GEO','REF_CLEAR_S42_GEO'),
        ('RAW_OCC_S42_GEO','REF_OCC_S42_GEO'),('RAW_OCC_S43_GEO','REF_OCC_S43_GEO'),
        ('RAW_CLEAR_S42_GEO','RAW_OCC_S42_GEO'),('REF_CLEAR_S42_GEO','REF_OCC_S42_GEO'),
        ('OLD_REF_GEO','REF_CLEAR_S42_GEO'),('OLD_REF_GEO','REF_OCC_S42_GEO'),('OLD_REF_GEO','REF_OCC_S43_GEO'),
        ('RAW_OCC_S42_GEO','RAW_OCC_S43_GEO'),('REF_OCC_S42_GEO','REF_OCC_S43_GEO'),
        ('RAW_CLEAR_S42_D9','REF_CLEAR_S42_D9'),('RAW_OCC_S42_D9','REF_OCC_S42_D9'),('RAW_OCC_S43_D9','REF_OCC_S43_D9')]
    assert len(pairs)==len(set(pairs))
    return pairs


def candidate_metric_by_name(options,selected_name,fallback):
    """No score-dependent sorting; the name must already be frozen externally."""
    found=[row['metric'] for row in options if row['name']==selected_name]
    assert len(found)<=1
    return found[0] if found else fallback


def score():
    lock=verify_lock()
    if RESULT.exists():
        previous=C.read(RESULT)
        for binding in previous['sources']+previous['private_artifacts']:C.verify(binding)
        print('SAME_GEO_CONTROLS_ALREADY_SCORED',flush=True);return
    from scripts.research.pallet_clean_to_pose_transfer_v1 import eval_student as E
    from scripts.research.pallet_pose_objective_followup_v2 import metric_baseline as M
    start=time.monotonic();metrics={};poses={};association={};sources=[C.bind(LOCK)];children={}
    p=E.paths(42);rows=C.read(p['metadata']);groups=E.group_ids(rows)
    primary_poses=C.read(p['poses'])
    for alias in MISSING:
        configured=configure(alias);ENGINE.score()
        child=C.read(configured['result'])
        for binding in child['sources']+child['private_artifacts']:C.verify(binding)
        data=C.read(configured['private']/'SELECTED_METRICS_PRIVATE.json')
        metrics[alias+'_D9']=data['D9'];metrics[alias+'_GEO']=data['OLD_GEO']
        poses[alias+'_D9']=primary_poses[configured['arm']]
        poses[alias+'_GEO']=C.read(configured['private']/'POSES.json')
        for selector in ('D9','GEO'):
            association[alias+'_'+selector]=dict(predictions=C.bind(p['predictions']),prediction_arm=configured['arm'],
                candidates=C.bind(p['candidates']),candidate_arm=configured['arm'],
                decisions=C.bind(configured['private']/'DECISIONS.json') if selector=='GEO' else None,
                metadata=C.bind(p['metadata']))
        children[alias]=C.bind(configured['result'])
        sources.extend([children[alias],*child['private_artifacts']])
    for seed in (42,43):
        path=OLD.DOC/f'SELECTOR_PAIR_RESULTS_S{seed}.json';pair=C.read(path)
        assert pair['scorer']==lock['same_frozen_scorer']
        for binding in pair['sources']+pair['private_artifacts']:C.verify(binding)
        private=OLD.RAW/f'selector_pair_S{seed}/MATCHED_SELECTED_METRICS_PRIVATE.json'
        data=C.read(private)
        assert data['R0']==metrics['R0_D9'] and data['OLD_REF']==metrics['OLD_REF_D9']
        for target in ('RAW','REF'):
            childlock=C.read(C.ROOT/pair['children'][target]['lock']['path'])
            candidates=next(b for b in childlock['sources'] if b['path'].endswith('/CANDIDATES.json'))
            predictions=next(b for b in childlock['sources'] if b['path'].endswith('/PREDICTIONS.json'))
            metadata=next(b for b in childlock['sources'] if b['path'].endswith('/METADATA.json'))
            selectedposes=next(b for b in childlock['files'] if b['path'].endswith('/POSES.json'))
            decisions=next(b for b in childlock['files'] if b['path'].endswith('/DECISIONS.json'))
            assert C.read(C.ROOT/metadata['path'])==rows, 'Control metadata differs across seeds'
            arm='CLEAN_'+target+'_OCC';candidate_rows=C.read(C.ROOT/candidates['path'])[arm]
            for selector in ('D9','GEO'):
                metrics[f'{target}_OCC_S{seed}_{selector}']=data[f'{target}_{selector}']
                poses[f'{target}_OCC_S{seed}_{selector}']=({fid:record['current'] for fid,record in candidate_rows.items()}
                    if selector=='D9' else C.read(C.ROOT/selectedposes['path']))
                association[f'{target}_OCC_S{seed}_{selector}']=dict(predictions=predictions,prediction_arm=arm,
                    candidates=candidates,candidate_arm=arm,decisions=decisions if selector=='GEO' else None,metadata=metadata)
        sources.extend([C.bind(path),C.bind(private)])
    assert len(metrics)==16
    assert all(set(values)==set(groups['FULL128']) for values in metrics.values())
    summaries={group:{arm:M.summarize(values[fid] for fid in ids) for arm,values in metrics.items()} for group,ids in groups.items()}
    pairs=contrast_pairs()
    paired={group:{after+'-minus-'+before:M.paired(metrics[before],metrics[after],ids) for before,after in pairs} for group,ids in groups.items()}
    classification={after+'-minus-'+before:M.classify_candidate(summaries['NATURAL99'][after],summaries['NATURAL99'][before]) for before,after in pairs}
    natural=summaries['NATURAL99']
    pareto=M.pareto_front([dict(card_id=arm,translation_cm=value['full_population']['translation_cm']['median'],
        rotation_deg=value['full_population']['rotation_deg']['median']) for arm,value in natural.items() if arm.endswith('_GEO')])
    loto={rec:{after+'-minus-'+before:M.paired(metrics[before],metrics[after],
        [r['id'] for r in rows if r['severity']!='CLEAN' and r['recording']!=rec]) for before,after in pairs}
        for rec in sorted({r['recording'] for r in rows})}
    private=C.RAW/'CONTROL_FRAME_METRICS_PRIVATE.json';C.save(private,M.clean(metrics),True)
    poses_path=C.RAW/'CONTROL_POSES_PRIVATE.json';C.save(poses_path,poses,True)
    association_path=C.RAW/'CONTROL_CASE_BINDINGS_PRIVATE.json';C.save(association_path,association,True)
    sources += [C.bind(p['metadata']),C.bind(Path(__file__)),C.bind(Path(M.__file__)),
        C.bind(OLD.DOC/'SELECTOR_SUPERVISION_PROVENANCE.json')]
    result=dict(created_at=C.now(),groups=summaries,paired=paired,
        classifications_NATURAL99=classification,pareto_GEO_by_T_R_median_only=pareto,
        leave_one_recording_out_NATURAL99=loto,children=children,
        available_models=list(metrics),missing_models=['RAW_CLEAR_S43','REF_CLEAR_S43'],
        current_teacher_budget='9images/38manualcorners',historical_GEO_ancestry='Additional disjoint10plasticimages/48corners; complete correctedstudent+GEO union19images/86corners.',
        R0_GEO_ancestry='R0 plus historical GEO: indirect10plasticimages/48corners, not zero-human-supervision.',
        reference='Same repeated development128 / natural99; geometry-derived reference, not independent physical6D GT.',
        contrast_limits='CLEAR/OCC matched only seed42. OCC repeats42/43. OLD_REF217 versus clean78 changes membership/exposure composition, not clean-status-only causality.',
        decision_scope='Descriptive controls/Pareto only; practical stop/configuration choice remains a separate evidence decision.',
        no_missing_seed_fabrication=True,old_results_preserved=True,new_student_fits=0,new_selector_fits=0,
        optimizer_updates=0,GPU_seconds=0,sources=sources,private_artifacts=[C.bind(private),C.bind(poses_path),C.bind(association_path)],seconds=time.monotonic()-start)
    C.save(RESULT,M.clean(result),True)
    print('SAME_GEO_CONTROLS_RESULTS',{
        arm:{'T':value['full_population']['translation_cm']['median'],'R':value['full_population']['rotation_deg']['median'],
             'TP90':value['full_population']['translation_cm']['P90'],'RP90':value['full_population']['rotation_deg']['P90']}
        for arm,value in natural.items() if arm.endswith('_GEO')},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('freeze','score'))
    args=parser.parse_args();{'freeze':freeze,'score':score}[args.phase]()
