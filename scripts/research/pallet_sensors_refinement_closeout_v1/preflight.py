"""Freeze methods/populations before new performance calculations or D training."""
import numpy as np
from env import *
def run():
    if (DOC/'METHOD_FREEZE.json').exists():verify();print('PREFLIGHT_RESUMED');return
    assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()=='main'
    start=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    subprocess.check_call(['git','merge-base','--is-ancestor','bb952ad4d81be39a3c1a88d97f16b741b521e632','HEAD'])
    assert sha(R0)==C.R0_SHA
    ps=read(B/'P_SELECTION.json');ls=read(B/'LINE_SOURCE_BINDING.json');pe=old('paper_evaluation')
    files=[R0,*PCODE.glob('*.py'),*C.LINE_CODE.glob('*.py'),*B.glob('*.json'),*B.glob('*.md'),LINE/'TRAIN_PROTOCOL.json',LINE/'SELECTION.json',LINE/'SOURCE_MANIFEST.json',LINE/'SOURCE_DATA_AUDIT.json',LINE/'cache/CACHE_MANIFEST.json',LINE/'cache/CACHE_COMPLETE.json',LINE/'baseline/FULL_CANDIDATES.json',LINE/'VERDICT.json']
    for s in (1,2,3):
        p=BRAW/f'runs/seed{s}/last.pt';assert sha(p)==ps['checkpoints'][str(s)]
        l=ROOT/ls['seeds'][str(s)]['checkpoint'];assert sha(l)==ls['seeds'][str(s)]['checkpoint_sha256']
        files += [p,l,BRAW/f'order_seed{s}.npy']
        for directory in [BRAW/f'evaluation/P{s}',LINE/f'evaluation/image_line_only_seed{s}']:
            files += [directory/n for n in ('PREDICTIONS.json','PAPER_2D.json','PAPER_2D_per_frame.csv','POSE_PER_FRAME_BY_ARM.json')]
    files += [Path(p) for p in pe.fixed_inputs()]
    files += list((ROOT/'_docs/paper/final').rglob('*.md'))+list((ROOT/'_docs/paper/final').rglob('*.json'))
    data=dataset()
    arrays=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in (LINE/'cache').glob('*.npy')]
    write(DOC/'SOURCE_BINDING.json',dict(start_sha=start,files=[bound(p) for p in sorted(set(files))],cache_arrays=arrays,
        cache_integrity='Existing completion/shard manifest verified; mmap read-only, size+mtime tracked; multi-GB features not rehashed every stage',
        original_dirty=subprocess.check_output(['git','status','--short'],text=True)))
    config=read(B/'PARAMETER_BUDGET_LOCK.json')
    write(DOC/'METHOD_FREEZE.json',dict(R0=bound(R0),P_checkpoints=ps['checkpoints'],P_config=config,
        temperatures=ps['temperatures'],selected_rule=ps['selected_rule'],selection_sha256=sha(B/'P_SELECTION.json'),
        code={p.name:sha(p) for p in [PCODE/'generic_point_refiner.py',PCODE/'point_inference.py',C.LINE_CODE/'features.py']},
        preprocessing=read(LINE/'TRAIN_PROTOCOL.json')['architecture'],
        restoration='decode applies lambda once; restore_refinement uses lambda only as zero guard and adds delta/gain. Cap=image diagonal*fraction*gain input pixels.',
        indices='original 9 indices, corners0..7 refined, center8 fixed; no added symmetry minimization',
        dimensions='Not neural inputs; registered physical dimensions supplied to downstream canonical PnP',P_retraining=0,L_retraining=0,R0_retraining=0))
    pop=pe.population();metadata=read(pe.POS);base=read(LINE/'baseline/FULL_CANDIDATES.json')
    rows=[]
    for item in pop.positive.items:
        target=pe.E._legacy_forbidden_target(item);cand=base['frames'][pe.canonical_key(item.image)]
        top=max(cand,key=lambda c:c['score']) if cand else None
        matched=top is not None and pe.E._box_iou(np.asarray(top['box_xyxy']),target.box_xyxy)>=.5 and top['keypoints_xy'] is not None
        rows.append(dict(frame_id=item.frame_id,image=str(item.image),label=str(item.label),session_id=next(r['session_id'] for r in metadata['items'] if r['frame_id']==item.frame_id),supervised=int(target.keypoint_supervision_mask.sum()),R0_matched=bool(matched)))
    write(DOC/'POPULATION_AND_METRIC_LOCK.json',dict(primary='mean seed pooled supervised9kp median [original px]; detected precision, no frame-median averaging',
        positive=len(rows),negative=len(pop.negative.items),sessions=sorted({r['session_id'] for r in rows}),
        matched=sum(r['R0_matched'] for r in rows),matched_supervised=sum(r['supervised'] for r in rows if r['R0_matched']),total_supervised=sum(r['supervised'] for r in rows),frames=rows,
        negative_session_metadata=sum(bool(v.get('session_id')) for v in base['frame_metadata'].values() if v['kind']=='negative'),
        synthetic_partitions={p:int((data.partitions==p).sum()) for p in np.unique(data.partitions)},synthetic_matched_train=len(data.train_rows),
        synthetic_source_audit=read(LINE/'SOURCE_DATA_AUDIT.json'),bootstrap=dict(draws=10000,seed=20260914,primary='session cluster',secondary='frame',same_multiplicity_all_models=True),
        safety='ALL_GT_PCK5/10/20 with fixed total supervised denominator, no detections or missing points count zero',role='reused development; no independent confirmation'))
    write(DOC/'COMPARISON_SCOPE.json',dict(primary='mean_s median(P_s)-median(single R0)',mechanism='mean_s[median(P_s)-median(D_s)]',
        L='historical structural control; historical line verdict unchanged',D='direct regression ablation, not PoseFix or CRT-6D reproduction',
        secondary=['P90','frame mean','gross20','8 corners','canonical pose','runtime'],
        evidence_rule='P_DEV_SUPPORTED if primary session CI high<0 without observed safety/pose harm; P_DEV_HARM_OR_TRADEOFF if observed primary/safety/pose tradeoff; otherwise P_DEV_UNRESOLVED. All effects reported, no composite publication gate.',
        exploratory='All contrasts beyond single P-R0 primary, no familywise guarantee',confirmation='requires frozen comparators and independently collected sessions'))
    write(DOC/'RUNTIME_PROTOCOL.json',dict(keys=read(B/'REAL_EVALUATION_LOCK.json')['timing']['keys'],models=['R0']+[f'{a}{s}' for a in ('P','L','D') for s in (1,2,3)],
        warmup=20,repeats=5,order='cyclic model rotation by repeat and reverse on odd blocks',batch=1,dtype='YOLO FP32, features FP16 round-trip, head FP32',cpu_threads=4,
        scopes=['decoded BGR to original2D','same plus canonical MAIN prediction-only PnP'],exclude=['disk decoding','model loading'],device='actual RTX3080; desktop only',
        sample_selection='exact historical26 list; every sample retained',concurrent_scoring_training=False))
    write(DOC/'CURRENT_RESULT.json',dict(EXECUTION='PARTIAL',stage='METHOD_AND_POPULATION_FROZEN',new_main_updates=0))
    print('LOCKS_READY',len(rows),len(data.train_rows),flush=True)
if __name__=='__main__':run()
