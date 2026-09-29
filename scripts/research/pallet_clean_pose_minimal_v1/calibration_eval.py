"""Apply the single source-calibrated scorer to every frozen control.

Freeze: no real reference/metric reads. Score: cached candidate metrics are
indexed by presealed names. Neither candidate generation nor final poses change.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np

from . import common as C
from . import calibration as K
from . import controls as B

MODELS = (*B.MISSING, 'RAW_OCC_S42','REF_OCC_S42','RAW_OCC_S43','REF_OCC_S43')
PRIVATE = C.RAW/'calibration_eval'
LOCK = C.DOC/'CURRENT_GEO_PREDICTIONS_LOCK.json'
RESULT = C.DOC/'CURRENT_GEO_RESULTS.json'


def contrasts():
    pairs=[]
    for model in MODELS:
        pairs += [(model+'_D9',model+'_NEWGEO'),(model+'_GEO',model+'_NEWGEO')]
    pairs += [('R0_NEWGEO',model+'_NEWGEO') for model in MODELS if model!='R0']
    pairs += [('OLD_REF_NEWGEO',model+'_NEWGEO') for model in MODELS if model not in ('R0','OLD_REF')]
    pairs += [('OLD_REF_GEO',model+'_NEWGEO') for model in MODELS]
    pairs += [(before.replace('_GEO','_NEWGEO'),after.replace('_GEO','_NEWGEO'))
        for before,after in B.contrast_pairs() if before.endswith('_GEO') and after.endswith('_GEO')]
    return list(dict.fromkeys(pairs))


def verify():
    lock=C.read(LOCK)
    assert lock['reference_reads_before_decisions'] is False
    assert lock['same_common_scorer_all_models'] and lock['candidate_poses_unchanged']
    for binding in lock['sources']+lock['files']:C.verify(binding)
    return lock


def freeze():
    if LOCK.exists():verify();print('CURRENT_GEO_ALREADY_LOCKED',flush=True);return
    from scripts.research.pallet_clean_to_pose_transfer_v1.selector_compat import guard_reference_reads
    reads=guard_reference_reads()
    import torch
    from scripts.research.pallet_selector_recovery_v1 import models as M,features as F,common as U
    from scripts.research.pallet_clean_to_pose_transfer_v1 import eval_student as E
    torch.set_num_threads(2); started=time.monotonic()
    fit=C.read(K.FIT); assert fit['selector_fits']==1 and fit['no_real_reference_read']
    C.verify(fit['checkpoint'])
    for binding in fit['sources']:C.verify(binding)
    checkpoint=torch.load(C.ROOT/fit['checkpoint']['path'],map_location='cpu',weights_only=False)
    assert checkpoint['d']==94 and checkpoint['variant']=='GEO_LINEAR'
    association_path=C.RAW/'CONTROL_CASE_BINDINGS_PRIVATE.json'
    association=C.read(association_path); metadata=None; poses={}; decisions={}; features={}; counts={}; shifts={}
    sources=[C.bind(K.FIT),fit['checkpoint'],C.bind(association_path),C.bind(Path(__file__)),
        C.bind(Path(F.__file__)),C.bind(Path(M.__file__)),C.bind(C.DOC/'SELECTOR_CALIBRATION_PARENT_BINDING.json')]
    cache={}
    def load(binding):
        C.verify(binding)
        if binding['path'] not in cache:cache[binding['path']]=C.read(C.ROOT/binding['path'])
        return cache[binding['path']]
    for model in MODELS:
        binding=association[model+'_GEO']; rows=load(binding['metadata']); E.validate_membership(rows)
        if metadata is None:metadata=rows
        assert rows==metadata
        predictions=load(binding['predictions'])[binding['prediction_arm']]
        candidates=load(binding['candidates'])[binding['candidate_arm']]
        sources += [binding['metadata'],binding['predictions'],binding['candidates']]
        arm=model+'_NEWGEO'; poses[arm]={}; decisions[arm]={}; features[arm]={}; count=Counter(); finite=[]
        for row in rows:
            fid=row['id']; record=candidates[fid]; feature=F.extract(predictions[fid],row['K'],row['xyz'],row['hw'])
            assert feature['selection']==record['selected_name']
            assert [h['name'] for h in feature['hypotheses']]==sorted(h['name'] for h in record['hypotheses'])
            selected=feature['selection']; score=None
            if feature['valid']:
                x=np.asarray(feature['features'],np.float32); score=M.scores(checkpoint,x[None])[0]
                index=int(M.selection(score[None],U.HYP)[0])
                assert index==1-int(M.selection(score[None,::-1],U.HYP[::-1])[0])
                selected=U.HYP[index]; finite.append(x)
            found=next((hyp for hyp in record['hypotheses'] if hyp['name']==selected),None)
            pose=found['pose'] if found is not None else record['current']
            assert not pose['available'] or pose['selected_hypothesis']==selected
            poses[arm][fid]=pose; features[arm][fid]=feature
            decisions[arm][fid]=dict(selected=selected,D9_selected=record['selected_name'],changed=selected!=record['selected_name'],
                valid_pair=bool(feature['valid']),scores=score.tolist() if score is not None else None)
            count['frames']+=1;count['valid_pair']+=bool(feature['valid']);count['changed']+=selected!=record['selected_name']
        counts[arm]=dict(count)
        standardized=(np.asarray(finite).reshape(-1,94)-checkpoint['mean'])/checkpoint['std']
        shifts[arm]=dict(valid_candidate_rows=len(standardized),fraction_abs_z_gt3=float((np.abs(standardized)>3).mean()),
            fraction_abs_z_gt5=float((np.abs(standardized)>5).mean()),used_for_selection_policy=False)
    files=[]
    for name,value in [('POSES',poses),('DECISIONS',decisions),('FEATURES',features),('METADATA',metadata)]:
        path=PRIVATE/(name+'.json');C.save(path,value,True);files.append(C.bind(path))
    sources=list({binding['path']:binding for binding in sources}.values())
    C.save(LOCK,dict(created_at=C.now(),files=files,sources=sources,checkpoint=fit['checkpoint'],
        models=list(MODELS),counts=counts,feature_shift_against_current_TRAIN_normalization=shifts,
        selection='One shared Linear94, lower score with hypothesis-name tie; invalid pair retains D9.',
        same_common_scorer_all_models=True,candidate_poses_unchanged=True,reference_reads_before_decisions=False,
        read_guard_active=True,read_paths=sorted(set(reads)),candidate_order_swap_invariant=True,
        new_selector_fits=0,new_student_fits=0,optimizer_steps=0,seconds=time.monotonic()-started),True)
    print('CURRENT_GEO_LOCKED',counts,flush=True)


def score():
    lock=verify()
    if RESULT.exists():
        result=C.read(RESULT)
        for binding in result['sources']+result['private_artifacts']:C.verify(binding)
        print('CURRENT_GEO_ALREADY_SCORED',flush=True);return
    from scripts.research.pallet_clean_to_pose_transfer_v1 import eval_student as E
    from scripts.research.pallet_pose_objective_followup_v2 import metric_baseline as M
    started=time.monotonic(); old=C.read(B.RESULT)
    for binding in old['sources']+old['private_artifacts']:C.verify(binding)
    metrics=C.read(C.RAW/'CONTROL_FRAME_METRICS_PRIVATE.json')
    poses=C.read(C.RAW/'CONTROL_POSES_PRIVATE.json')
    association=C.read(C.RAW/'CONTROL_CASE_BINDINGS_PRIVATE.json')
    metadata=C.read(PRIVATE/'METADATA.json'); decisions=C.read(PRIVATE/'DECISIONS.json')
    poses.update(C.read(PRIVATE/'POSES.json')); groups=E.group_ids(metadata)
    sources=[C.bind(LOCK),C.bind(B.RESULT),*old['private_artifacts'],C.bind(Path(__file__)),C.bind(Path(M.__file__))]
    cached={}
    for model in MODELS:
        arm=model+'_NEWGEO'; binding=association[model+'_GEO']; candidate_path=C.ROOT/binding['candidates']['path']
        metric_path=candidate_path.parent/'ORACLE_METRICS_SELECTIONS_PRIVATE.json'
        # Existing scores are reopened only AFTER every new selector name is frozen.
        if metric_path not in cached:
            if (candidate_path.parent/'CANDIDATE_METRICS_LOCK.json').exists():
                seal_path=candidate_path.parent/'CANDIDATE_METRICS_LOCK.json'
                seal=C.read(seal_path); C.verify(seal['file'])
                assert seal['file']['path']==str(metric_path.relative_to(C.ROOT))
            else:
                seal_path=C.OLD.DOC/'CANDIDATE_ORACLE_S42.json'; seal=C.read(seal_path)
                expected=next(b for b in seal['private_artifacts'] if b['path']==str(metric_path.relative_to(C.ROOT)))
                C.verify(expected)
            sources.append(C.bind(seal_path))
            cached[metric_path]=C.read(metric_path)['candidate_metrics']; sources.append(C.bind(metric_path))
        options=cached[metric_path][binding['candidate_arm']]; metrics[arm]={}
        for row in metadata:
            fid=row['id']; selected=decisions[arm][fid]['selected']
            metrics[arm][fid]=B.candidate_metric_by_name(options[fid],selected,metrics[model+'_D9'][fid])
            if not decisions[arm][fid]['changed']:assert metrics[arm][fid]==metrics[model+'_D9'][fid]
        association[arm]=dict(binding,decisions=C.bind(PRIVATE/'DECISIONS.json'),decisions_arm=arm,
            poses=C.bind(PRIVATE/'POSES.json'),poses_arm=arm)
    assert len(metrics)==24 and all(set(arm)==set(groups['FULL128']) for arm in metrics.values())
    summaries={group:{arm:M.summarize(data[fid] for fid in ids) for arm,data in metrics.items()} for group,ids in groups.items()}
    pairs=contrasts()
    paired={group:{after+'-minus-'+before:M.paired(metrics[before],metrics[after],ids) for before,after in pairs} for group,ids in groups.items()}
    classifications={after+'-minus-'+before:M.classify_candidate(summaries['NATURAL99'][after],summaries['NATURAL99'][before]) for before,after in pairs}
    loto={rec:{after+'-minus-'+before:M.paired(metrics[before],metrics[after],
        [row['id'] for row in metadata if row['severity']!='CLEAN' and row['recording']!=rec]) for before,after in pairs}
        for rec in sorted({row['recording'] for row in metadata})}
    pareto=M.pareto_front([dict(card_id=arm,translation_cm=value['full_population']['translation_cm']['median'],
        rotation_deg=value['full_population']['rotation_deg']['median']) for arm,value in summaries['NATURAL99'].items()])
    private=[]
    for name,value in [('CURRENT_GEO_FRAME_METRICS_PRIVATE',M.clean(metrics)),('CURRENT_GEO_POSES_PRIVATE',poses),('CURRENT_GEO_CASE_BINDINGS_PRIVATE',association)]:
        path=C.RAW/(name+'.json');C.save(path,value,True);private.append(C.bind(path))
    result=dict(created_at=C.now(),groups=summaries,paired=paired,classifications_NATURAL99=classifications,
        leave_one_recording_out_NATURAL99=loto,pareto_all_by_T_R_median_only=pareto,available_models=list(metrics),
        counts=lock['counts'],same_2D_outputs=True,same_candidate_poses=True,scorer_fits_total_this_extension=1,
        new_fits_this_evaluation=0,checkpoint=lock['checkpoint'],sources=sources,private_artifacts=private,
        evidence='Repeated real DEV; one synthetic-only calibration with frozen recipe. No independent physical6D test.',
        scope='New scorer training ancestry9/38; complete research already involved19/86. R0+newGEO still indirectly depends on the two calibration students.',
        no_model_specific_scorer_or_posthoc_routing=True,seconds=time.monotonic()-started)
    C.save(RESULT,M.clean(result),True)
    print('CURRENT_GEO_RESULTS',{arm:{'T':v['full_population']['translation_cm']['median'],'R':v['full_population']['rotation_deg']['median']}
        for arm,v in summaries['NATURAL99'].items() if arm.endswith('_NEWGEO')},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('freeze','score'))
    args=parser.parse_args();globals()[args.phase]()
