"""Metadata-only fixed training pool; prohibit target evaluation recording overlap."""
from collections import Counter
from pathlib import Path
import numpy as np
from . import common as C


def prepare():
    dev=C.ROOT/'_docs/experiments/pallet_dim_conditioned_p_v1/DEV_CACHE_COMPLETE.json'
    green=C.ROOT/'_docs/paper/final_dimension_v1/green150_saved_labels_v1/DATASET_SNAPSHOT.json'
    groups=C.ROOT/'data/pallet/results/site_environment_audit_v1/SOURCE_RECORDING_GROUPS.json'
    cache=C.ROOT/'data/pallet/results/paper_selftrain_v1/teacher_cache/R0_TEACHER_CACHE.json'
    base=C.ROOT/'data/pallet/results/pallet_line_pose_v1/baseline/FULL_CANDIDATES.json'
    d=C.read(dev);g=C.read(green); gg=C.read(groups)
    evalpaths=[r['key'] for r in d['records']]+[r['image']['path'] for r in g['records']]
    evalhash={r['image_sha256'] for r in C.read(base)['frame_metadata'].values()}|{r['image']['sha256'] for r in g['records']}
    keygroup={s['session_key']:r['recording_id'] for r in gg['groups'] if not r['is_collection'] for s in r['sessions']}
    prohibited={v for k,v in keygroup.items() if any(Path(p).is_relative_to(k) for p in evalpaths)}
    # Unknown-inventory recent sessions also remain protected through explicit image roots.
    evalroots={str(Path(p).parent.parent) for p in evalpaths}
    aliases={k for k,v in keygroup.items() if v in prohibited}
    def excluded(session):
        return session in evalroots or keygroup.get(session) in prohibited or any(
            (r['session_a']==session and r['session_b'] in aliases) or
            (r['session_b']==session and r['session_a'] in aliases) for r in gg['partial_overlap_pairs'])
    records=[];excluded_rows=[];sources=[C.bound(p) for p in [dev,green,groups,cache,base,Path(__file__),C.HERE/'common.py']]
    for row in C.read(cache)['entries']:
        session=str(Path(row['image_path']).parent.parent)
        if excluded(session) or row['image_sha256'] in evalhash:
            excluded_rows.append(dict(image=row['image_path'],reason='evaluation identity or recording'));continue
        b=C.bound(C.ROOT/row['image_path']);assert b['sha256']==row['image_sha256']
        records.append(dict(id='PLASTIC__'+row['image_sha256'],kind='PLASTIC',object_type=C.TYPES['PLASTIC'],image=b,
            session=session,K=row['camera_matrix'],type_source='existing build_pseudo_manifests.POOL_OBJECT_TYPE',recording_id=keygroup[session]))
    root=C.ROOT/'challenge/data/01_real/_live_captures/forklift_v4_20260901/sessions'
    # Use four existing sessions with explicit square-pallet annotation metadata.
    for session in ['forklift_v4_173507','forklift_v4_174126','forklift_v4_174342','forklift_v4_174925']:
        path=root/session;rel=str(path.relative_to(C.ROOT));assert not excluded(rel)
        annotations=sorted((C.ROOT/'challenge/data/01_real/live_capture_gt'/f'{session}_manual_gt').glob('*.json'))
        assert annotations
        # Never train on manually annotated frames here; their coordinates are not targets.
        annotated={p.stem for p in annotations}
        for p in annotations:
            doc=C.read(p);assert doc['objects'][0]['object_type']==C.TYPES['GREEN']
        sources += [C.bound(annotations[0]),C.bound(path/'cam_K.txt')]
        candidates=[p for p in sorted((path/'rgb').glob('*.png')) if p.stem not in annotated]
        count=min(250,len(candidates))
        selected=[candidates[((2*k+1)*len(candidates))//(2*count)] for k in range(count)]
        for p in selected:
            b=C.bound(p);assert b['sha256'] not in evalhash
            records.append(dict(id='GREEN__'+b['sha256'],kind='GREEN',object_type=C.TYPES['GREEN'],image=b,
                session=rel,K=np.loadtxt(path/'cam_K.txt').reshape(3,3).tolist(),type_source=str(annotations[0].relative_to(C.ROOT)),recording_id=keygroup[rel]))
    assert len({r['image']['sha256'] for r in records})==len(records)
    result=dict(records=records,counts=dict(Counter(r['kind'] for r in records)),sources=sources,
        evaluation_counts=dict(Counter(r['object_type'] for r in d['records']),**{C.TYPES['GREEN']:len(g['records'])}),
        evaluation_recording_ids=sorted(prohibited),excluded=excluded_rows,no_eval_content_overlap=True,no_eval_recording_overlap=True,
        green_sampling='250 midpoint-spaced unannotated frames per Sep1 session, before model predictions; Sep2/3/4 evaluation sessions untouched',
        historical_exposure='Sep1 square captures have previous research exposure; no independent-test claim. Current students evaluated on fixed DEV319+GREEN150 only.',
        wood=dict(status='BLOCKED_NO_CALIBRATED_DISJOINT_POOL',
            evaluation_overlaps=['pallet_20260618_183705','pallet_20260618_184309','real_unlabeled_day_20260830','real_unlabeled_night_20260830'],
            separate_candidate='data/pallet/raw_data/wood/selected/20260618_132917',candidate_images=225,
            reason='Separate portrait video has no K/hfov; historical July9 annotation instructions explicitly require --K/--hfov. Do not invent calibration or weaken requested filters.',
            evidence=C.bound(C.ROOT/'_docs/history/2026-07-09.md')))
    C.freeze(C.DOC/'POOL.json',result)
    print(result['counts'],result['wood'],flush=True)


if __name__=='__main__':prepare()
