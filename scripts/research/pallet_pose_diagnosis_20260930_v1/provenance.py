"""Recording lineage, reference provenance, and existing metadata strata."""
from pathlib import Path
from collections import Counter
import csv
import json
import time
import numpy as np
from . import run as C
from scripts.research.pallet_visible_transfer_closure_v1 import common as V

def main():
    start,cpu=time.monotonic(),time.process_time();rows=C.metadata();evalsha={r['image']['sha256']:r['id'] for r in rows}
    split=C.read(C.ROOT/'_docs/experiments/pallet_existing_data_transfer_v1/SPLIT_LOCK.json')
    gdpath=C.ROOT/'data/pallet/results/site_environment_audit_v1/SOURCE_RECORDING_GROUPS.json';gd=C.read(gdpath)
    aliases={s['session_key']:g['recording_id'] for g in gd['groups'] if not g['is_collection'] for s in g['sessions']}
    parent={x:x for x in set(aliases.values())}
    def root(x):
        while parent[x]!=x:x=parent[x]
        return x
    for m in split['recording_merges']:parent[root(m['b'])]=root(m['a'])
    def overlap(records):
        shas={r['image']['sha256'] for r in records};recs={root(r['recording']) for r in records}
        return dict(training_frames=len(records),training_recordings=sorted(recs),
            image_overlap=[r['id'] for r in rows if r['image']['sha256'] in shas],
            recording_overlap=[r['id'] for r in rows if r['recording'] in recs],
            clean29_image_overlap=[r['id'] for r in rows if r['severity']=='CLEAN' and r['image']['sha256'] in shas],
            clean29_recording_overlap=[r['id'] for r in rows if r['severity']=='CLEAN' and r['recording'] in recs])
    old= C.read(V.RAW/'TRAIN_TARGETS_PRIVATE.json');oldck=V.checkpoint('REF_LR5');C.verify(oldck)
    cleanpath=C.ROOT/'data/pallet/results/pallet_clean_to_pose_transfer_v1/CLEAN_LOCKED_PRIVATE.json';clean=C.read(cleanpath)['rows'];assert len(clean)==78
    for r in old:C.verify(r['image'])
    train=C.read(C.ROOT/'_docs/experiments/pallet_posefix_crop_completion_v2/TRAIN_INPUT_AUDIT.json')['real']
    full=[dict(r,recording=root(aliases[r['session']])) for r in train]
    for r in full:C.verify(r['image']);C.verify(r['old'])
    # Resolve actual REALFT prepared image membership back to original files.
    ft=C.ROOT/'challenge/yolo_pose_one_model/datasets/ft_a';prep=C.read(ft/'_prepare_real_train.json')
    dirs={Path(p).name.removesuffix('_manual_gt'):p for p in prep['positive_dirs']}
    unique=sorted({tuple(p.stem.split('__')[1:3]) for p in (ft/'images/train').iterdir() if p.name.startswith('real__')})
    realft=[]
    for session,frame in unique:
        path=C.ROOT/dirs[session]/(frame+'.png');assert path.exists(),path
        realft.append(dict(id=session+':'+frame,image=C.bind(path),recording=root(aliases[dirs[session]]),source_directory=dirs[session]))
    assert len(realft)==157
    lineage=dict(R0=dict(checkpoint=split['R0_provenance']['baseline'],provenance=split['R0_provenance'],
        scope='Pallet fitting synthetic-only. Upstream COCO-pose supervision exists; selection uses historical reused DEV.'),
        PRIOR1=dict(checkpoint=C.read(C.L.DOC/'EXPERIMENT_PROTOCOL.json')['base'],scope='synthetic PoseFix prior seed1 last6000; R0 synthetic source inference inputs; prior selection reused development'),
        FULL125=dict(checkpoint=C.read(C.L.DOC/'FIT_FULL.json')['checkpoint'],parent='PRIOR1',fit='FULL last300, BN statistics and affine frozen; actual253 one-recording OCC inputs / frozen clean Replay+selfocclusionPnP targets / source replay',**overlap(full)),
        OLD_REF217=dict(checkpoint=oldck,parent='R0',fit='pose-only REF_LR5 five epochs, same frozen detector; separate student',**overlap(old)),
        ST_clean78=dict(**overlap(clean),note='78 subset of USED217, not the253 FULL125 training set, not clean29'),
        REALFT_A=dict(checkpoint=C.read(C.RAW/'COST_E6_infer.json')['checkpoint'],**overlap(realft),
            initialization='stage_a_synth_640_b32_seed42 best.pt, different from R0 G38',training=C.read(ft/'_build_ft.json'),validation='1000 synthetic images; actual directory has no real validation images',
            selection_caveat='Report selection_policy mentions epoch60 last, while resolved file is best.pt and args epochs40. Exact hashed file used; historical selection claim unresolved.'))
    C.save(C.RAW/'LINEAGE_MEMBERSHIP.json',dict(FULL125=full,OLD_REF217=old,ST_clean78=clean,REALFT_A=realft))
    C.save(C.DOC/'MODEL_LINEAGE.json',dict(models=lineage,sources=[C.bind(gdpath),C.bind(cleanpath),C.bind(V.RAW/'TRAIN_TARGETS_PRIVATE.json'),
        C.bind(ft/'_build_ft.json'),C.bind(ft/'_prepare_real_train.json'),C.bind(ft/'data.yaml'),C.bind(C.ROOT/'challenge/yolo_pose_one_model/runs_ft/ft_a_real157_neg259_synth12k/args.yaml')],
        historical_DEV_selection_exposure='All evaluated populations repeatedly reused; zero fitting overlap does not create an independent test.'))
    metrics=C.read(C.RAW/'E1_POSE_METRICS.json');metrics['REALFT_A']=C.read(C.RAW/'E6_POSE_METRICS.json');two=C.read(C.RAW/'E1_2D_METRICS.json')
    conditions=list(csv.DictReader((C.ROOT/'data/evaluation/pallet_eval_v1/manifests/frames.csv').open()))
    cond={}
    for v in conditions:
        key='data/evaluation/pallet_eval_v1/'+v['image_path']
        if key in cond:
            assert all(v[k]==cond[key][k] for k in ('occlusion','distance_bin','size_bin','elevation_bin','view_bin'))
        cond[key]=v
    provenance={};sources=Counter();statuses=Counter()
    annotations={r['id']:r['annotation'] for r in split['heldout']}
    for r in rows:
        i=r['id'];a=C.read(C.ROOT/annotations[i]['path']);obj=a['objects'][0]
        entries=obj['keypoint_annotations'];sources.update(e.get('source','MISSING') for e in entries[:8])
        statuses.update([str(a.get('real_gt_v2_migration',{}).get('status','MISSING'))])
        provenance[i]=dict(annotation=annotations[i],point_sources=[e.get('source','MISSING') for e in entries[:8]],
            object_gt_source=obj.get('gt_source'),migration=a.get('real_gt_v2_migration'),camera=a['camera_data'],
            condition_metadata=cond.get(r['image']['path']),recording=r['recording'],severity=r['severity'])
    summary={}
    for pop,ids in [('NATURAL99',[r['id'] for r in rows if r['severity']!='CLEAN']),('CLEAN29',[r['id'] for r in rows if r['severity']=='CLEAN'])]:
        rr=[r for r in rows if r['id'] in ids];groups={}
        for field in ('occlusion','elevation_bin','distance_bin','size_bin','view_bin'):
            for value in sorted({(cond.get(r['image']['path']) or {}).get(field,'UNKNOWN') for r in rr}):
                groups[field+':'+value]=[r['id'] for r in rr if (cond.get(r['image']['path']) or {}).get(field,'UNKNOWN')==value]
        for matched in (True,False):groups['matched:'+str(matched)]=[r['id'] for r in rr if two['identity'][r['id']]['matched']==matched]
        cand=C.read(C.RAW/'E1_CANDIDATES.json')['identity']
        for wd in sorted({cand[i]['GEO_name'] for i in ids}):groups['baseline_WD:'+str(wd)]=[i for i in ids if cand[i]['GEO_name']==wd]
        ft_overlap=set(lineage['REALFT_A']['recording_overlap'])
        for label,predicate in [('overlap',lambda i:i in ft_overlap),('disjoint',lambda i:i not in ft_overlap)]:groups['REALFT_recording:'+label]=[i for i in ids if predicate(i)]
        summary[pop]={}
        for label,sub in groups.items():
            rrsub=[r for r in rr if r['id'] in sub]
            summary[pop][label]=dict(N=len(sub),recordings=dict(Counter(r['recording'] for r in rrsub)),
                models={a:C.summarize(v[i] for i in sub) for a,v in metrics.items() if 'held' not in a},
                paired={a:C.paired(metrics['identity'],metrics[a],rrsub,False) for a in ('FULL125','REALFT_A')} if sub else {})
    C.save(C.RAW/'E7_REFERENCE_PROVENANCE.json',provenance)
    C.save(C.DOC/'E7_SUBGROUP_SUMMARY.json',summary)
    sourcepath=C.ROOT/'_docs/experiments/pallet_clean19_pose_sensitive_diag_v1/SOURCE_GEOMETRY_BINDING.json';source=C.read(sourcepath)
    binding_status=[]
    for r in source['records']:
        for k in ('image','label','renderer'):
            b=r[k]
            try:C.verify(b);status='VERIFIED'
            except FileNotFoundError:status='MISSING'
            except AssertionError:status='HASH_MISMATCH'
            binding_status.append(dict(id=r['id'],kind=k,binding=b,status=status))
    C.save(C.RAW/'E1_SOURCE_BINDING_AUDIT.json',binding_status)
    C.save(C.DOC/'E7_REFERENCE_LIMITS.json',dict(point_source_counts_full128=dict(sources),annotation_migration_counts=dict(statuses),
        empirical_noise_floor=dict(status='BLOCKED',missing='Independent repeated original clicks / annotator-repeat linkage / empirical covariance',
            search='review JSON/MD and prior reference-gap reports: no reclick/reannotation/inter-/intra-annotator records found',
            projected_points_not_independent_clicks=True,MDE_not_estimated=True),
        source256=dict(input_binding=C.bind(sourcepath),verified=dict(Counter(x['status'] for x in binding_status)),
            pose_recompute_status='SKIPPED',reason='Available SOURCE_BASE/FULL and SOURCE_ROWS save scalar errors, not operational xy. E1/E3 directly resolve current transform/output linkage; no remaining source-specific decision warrants1024 new forwards. No old source R copied to C2.',
            caveat='Renderer linkage is not independent real reference validation; source stress damages coordinates, not RGB occlusion'),
        reference='Geometry-resolved K+mixed-provenance2D+known dimensions. Axis resolution source manual_gt_keypoints_plus_known_geometry, not independent signed physical-axis adjudication.',
        bootstrap='Same paired whole-recording draws,2000/seed20260929, six natural and three clean recordings; undefined draws counted; exploratory reused DEV only.',
        distortion='Existing solver passes distCoeffs=None to solvePnP and RefineLM; no new distortion correction introduced.'))
    C.ledger('E0_E7_provenance',start,cpu)
    print('LINEAGE_COMPLETE', {a:len(v.get('recording_overlap',[])) for a,v in lineage.items()},flush=True)

if __name__=='__main__':main()
