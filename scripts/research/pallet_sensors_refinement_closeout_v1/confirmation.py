"""History-based leakage gate and frozen-model evaluation entry for future data."""
import argparse,csv
import numpy as np
from env import *
from paired_stats_helper import paired_pooled_median
def history():
    records={};source_files=[]
    roots=[ROOT/'_docs/experiments',ROOT/'challenge/real_gt_v2/manifests']
    # Existing experiment manifests contain observation/selection history, not only optimizer inputs.
    def walk(x,path):
        if isinstance(x,dict):
            img=x.get('image',x.get('image_path',x.get('image_key')));session=x.get('session_id',x.get('session'));h=x.get('image_sha256',x.get('sha256') if img else None)
            if isinstance(img,str):
                key=(img,session if isinstance(session,str) else '')
                records.setdefault(key,dict(image=img,session_id=key[1],image_sha256=h if isinstance(h,str) else None,history_files=[]))['history_files'].append(str(path.relative_to(ROOT)))
                if isinstance(h,str) and records[key]['image_sha256'] is None:records[key]['image_sha256']=h
            for value in x.values():walk(value,path)
        elif isinstance(x,list):
            for value in x:walk(value,path)
    for root in roots:
        for p in sorted(root.rglob('*.json')):
            if DOC in p.parents or p.stat().st_size>30_000_000:continue
            try:v=read(p)
            except (ValueError,OSError):continue
            walk(v,p)
            if any(k in p.name for k in ('SPLIT','MANIFEST','SOURCE','SELECTION','QA','AUDIT','LOCK')):source_files.append(bound(p))
    # Explicit complete main populations and synthetic source; avoid assuming only training use contaminates a confirmation set.
    for p in [LINE/'baseline/FULL_CANDIDATES.json',LINE/'SOURCE_MANIFEST.json']:
        v=read(p);walk(v,p);source_files.append(bound(p))
        if 'frame_metadata' in v:
            for image,m in v['frame_metadata'].items():walk(dict(image=image,**m),p)
    values=list(records.values());known_sessions=sorted({r['session_id'] for r in values if r['session_id']})
    write(RAW/'confirmation/HISTORY_IMAGES.json',dict(records=values,known_sessions=known_sessions,sources=source_files,scope='available recorded usage; missing capture links require human declaration, not proof of independence'))
    finals={p.name:read(p) for p in (ROOT/'challenge/real_gt_v2/manifests').glob('FINAL*.json')}
    write(DOC/'CONFIRMATION_READINESS.json',dict(status='NEW_CONFIRMATION_DATA_REQUIRED',available_new_confirmed_frames=0,existing_final_manifests=finals,
        history_records=len(values),known_sessions=known_sessions,history_manifest=bound(RAW/'confirmation/HISTORY_IMAGES.json'),
        incomplete_lineage_policy='Reject confirmation until prior capture/source-video relations and all model-selection exposure are declared and reviewed',all_comparators_frozen=False,prior_baseline_pending=True))
    write(DOC/'CONFIRMATION_SCHEMA.json',dict(schema='pallet_independent_confirmation_v1',required_panel_fields=['comparators_frozen','prior_baseline_completed','capture_history_reviewed','frames'],
        required_frame_fields=['frame_id','session_id','capture_group_id','source_video_id','image','image_sha256','new_independent_capture','annotators_blinded','box_xyxy','keypoints_xy','supervision','intrinsics','dimensions_m'],
        predictions_schema='methods R0,P1,P2,P3,D1,D2,D3,PRIOR1,PRIOR2,PRIOR3 -> frame_id -> candidate list with score,box_xyxy,keypoints_xy',
        missing_prediction_policy='Frame entry must exist, empty list means no detection and zero PCK hits. Nonfinite predicted point receives no hit; conditional precision support must be audited.',
        provenance_policy='Registered weights/code/rules frozen before unblinding; no tuning or further fitting after opening',primary='pooled supervised9 median, mean seed deltas versus single R0; session95 CI',safety='all-GT PCK and coverage, no invented deployment tolerance'))
    with (DOC/'CONFIRMATION_CAPTURE_TEMPLATE.csv').open('w') as f:
        csv.writer(f).writerow(['frame_id','session_id','capture_group_id','source_video_id','timestamp','image','image_sha256','indoor_outdoor','day_night','material','measured_size_m','distance_m','intrinsics_file','annotator1','annotator2','blinding_confirmed','prior_usage','notes'])
    print('CONFIRMATION_HISTORY_READY',len(values),flush=True)
def gate(panel,hist):
    for key in ('comparators_frozen','prior_baseline_completed','capture_history_reviewed'):assert panel.get(key) is True,key
    frames=panel['frames'];assert frames and len({f['frame_id'] for f in frames})==len(frames)
    assert len({f['image_sha256'] for f in frames})==len(frames),'Duplicate confirmation image bytes'
    hashes={r['image_sha256'] for r in hist['records'] if r.get('image_sha256')};paths={str((ROOT/r['image']).resolve()) for r in hist['records']};sessions=set(hist['known_sessions'])
    for f in frames:
        assert f['new_independent_capture'] is True and f['annotators_blinded'] is True
        assert f['capture_group_id'] and f['source_video_id'] and f['session_id'] not in sessions
        assert str(Path(f['image']).resolve()) not in paths and f['image_sha256'] not in hashes
        assert sha(f['image'])==f['image_sha256']
        assert np.asarray(f['keypoints_xy']).shape==(9,2) and np.asarray(f['supervision']).shape==(9,)
def evaluate(panel_path,prediction_path):
    panel=read(panel_path);hist=read(RAW/'confirmation/HISTORY_IMAGES.json');gate(panel,hist);pred=read(prediction_path)
    freeze_path=DOC/'CONFIRMATION_MODEL_FREEZE.json'
    assert freeze_path.exists(),'Register completed prior/P/D/R0 model panel before unblinding'
    model_lock=read(freeze_path);assert model_lock['all_comparators_complete'] is True
    assert panel['model_freeze_sha256']==pred['model_freeze_sha256']==sha(freeze_path)
    for name,entry in model_lock['methods'].items():
        assert pred['checkpoint_sha256'][name]==entry['checkpoint_sha256']
        assert sha(ROOT/entry['checkpoint'])==entry['checkpoint_sha256']
    labels=['R0']+[f'{a}{s}' for a in ('P','D','PRIOR') for s in (1,2,3)];stores={};metrics={};E=old('paper_evaluation').E
    for name in labels:
        assert set(pred['methods'][name])=={f['frame_id'] for f in panel['frames']}
        rows=[];hits={t:0 for t in (5,10,20)};den=0;matched=0
        for f in panel['frames']:
            mask=np.asarray(f['supervision'],bool);gt=np.asarray(f['keypoints_xy'],float);assert np.isfinite(gt[mask]).all();den+=int(mask.sum())
            candidates=pred['methods'][name][f['frame_id']];top=max(candidates,key=lambda c:c['score']) if candidates else None
            e=np.empty(0)
            if top is not None and E._box_iou(np.asarray(top['box_xyxy']),np.asarray(f['box_xyxy']))>=.5 and top['keypoints_xy'] is not None:
                q=np.asarray(top['keypoints_xy'],float);assert q.shape==(9,2)
                e=np.linalg.norm(q[mask]-gt[mask],axis=-1);assert np.isfinite(e).all(),'Nonfinite supervised prediction; preserve failure, review precision estimand'
                matched+=1
                for threshold in hits:hits[threshold]+=int((e<=threshold).sum())
            rows.append(e)
        stores[name]=rows;metrics[name]=dict(ALL_GT_PCK={str(t):hits[t]/den for t in hits},matched_frames=matched,total_frames=len(rows),supervised_denominator=den,median_px=float(np.median(np.concatenate(rows))))
    paired=paired_pooled_median(stores['R0'],[stores[f'P{s}'] for s in (1,2,3)],[f['session_id'] for f in panel['frames']])
    write(RAW/'confirmation/EVALUATION.json',dict(panel=bound(panel_path),predictions=bound(prediction_path),metrics=metrics,paired=paired,
        pose='Not automatically evaluated: bind independent measured6D or geometry-reconstructed reference and coordinate convention before any pose claim'))
def test_gate():
    try:gate(dict(comparators_frozen=False),{});raise AssertionError('gate should reject')
    except AssertionError as e:assert str(e)=='comparators_frozen'
    historical=read(RAW/'confirmation/HISTORY_IMAGES.json')
    example=next(r for r in historical['records'] if r.get('session_id'))
    frame=dict(frame_id='synthetic_test',session_id=example['session_id'],capture_group_id='declared-new',source_video_id='declared-new',image=example['image'],image_sha256=example['image_sha256'],new_independent_capture=True,annotators_blinded=True)
    panel=dict(comparators_frozen=True,prior_baseline_completed=True,capture_history_reviewed=True,frames=[frame])
    rejected=False
    try:gate(panel,historical)
    except AssertionError:rejected=True
    assert rejected,'Known historical session must be rejected despite new-capture boolean'
    write(DOC/'CONFIRMATION_GATE_TESTS.json',dict(PASS=True,unfrozen_panel_rejected=True,recorded_prior_session_rejected=True,actual_confirmation_evaluated=False,scope='unfrozen and contaminated inputs rejected; future-data end-to-end requires actual independent inputs'))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--panel',type=Path);parser.add_argument('--predictions',type=Path);a=parser.parse_args()
    if a.panel:evaluate(a.panel,a.predictions)
    else:history();test_gate()
