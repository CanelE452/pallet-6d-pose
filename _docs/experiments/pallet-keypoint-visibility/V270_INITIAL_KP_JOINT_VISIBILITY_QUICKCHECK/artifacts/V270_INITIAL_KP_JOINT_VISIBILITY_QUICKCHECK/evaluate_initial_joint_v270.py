"""Late human-state evaluation of sealed whole-frame semantic KP outputs.

No image, detector, learned model, optimizer, pose solver, or permutation matching
is used here. --replay DIR recomputes all reported scientific numbers using only
compact saved input data. The first --evaluate run writes a consumer seal before
opening the current human-state ledger or any historical prediction files.
"""
from pathlib import Path
import argparse, csv, gzip, hashlib, json, math, shutil, time
import numpy as np

HERE = Path(__file__).absolute().parent
R = HERE.parent
D, E, S = 'DIRECT_VISIBLE', 'EXTERNAL_OCCLUDED', 'SELF_OCCLUDED'
SEMANTIC_ARMS = ('VIS_ONLY', 'JOINT')
ALL_ARMS = ('XY_ONLY', 'VIS_ONLY', 'JOINT')
IDENTITY_SHA = '33a07714f4f08938783e678b5af479bc1864c99933a48bc6e7da21585cdbe7cc'
LEDGER_SHA = 'df7b619649ede26af4e21644c22b10e84b334c6bc8708f4e5a668ec373db5175'

def safe(p):
    p = Path(p)
    assert p.is_absolute(), str(p)
    assert not any(x in str(p).lower() for x in ('v112', 'cache', '.pyc', '..')), str(p)
    for a in [p, *p.parents]:
        assert not a.is_symlink(), str(a)
    return p

def binding(p):
    p = safe(p)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return dict(path=str(p), sha256=h.hexdigest(), size_bytes=p.stat().st_size)

def verify(b):
    actual = binding(b['path'])
    assert actual['sha256'] == b['sha256'], (actual, b)
    for key in ('size_bytes', 'size', 'bytes'):
        if key in b:
            assert actual['size_bytes'] == b[key], (actual, b)
    return safe(b['path'])

def read_json(p):
    return json.loads(safe(p).read_bytes())

def dump(p, value):
    safe(p).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                 sort_keys=True, allow_nan=False) + '\n')

def ratio(a, b):
    return a / b if b else None

def cm(truth, pred):
    assert len(truth) == len(pred)
    c = {t + '_' + p: sum(a == t and b == p for a, b in zip(truth, pred))
         for t in ('D', 'E') for p in ('D', 'E', 'U')}
    dd, de, du, ed, ee, eu = [c[k] for k in ('D_D', 'D_E', 'D_U', 'E_D', 'E_E', 'E_U')]
    n = len(truth)
    dr, er = ratio(dd, dd + de + du), ratio(ee, ed + ee + eu)
    return dict(n=n, actual_D=dd+de+du, actual_E=ed+ee+eu,
                confusion_matrix=c,
                D_precision=ratio(dd, dd + ed), D_recall=dr,
                D_F1=ratio(2*dd, 2*dd+de+ed+du),
                E_precision=ratio(ee, ee + de), E_recall=er,
                E_F1=ratio(2*ee, 2*ee+de+ed+eu),
                U_rate=ratio(du + eu, n), coverage=ratio(n-du-eu, n),
                accuracy_all=ratio(dd+ee, n),
                accepted_accuracy=ratio(dd+ee, n-du-eu),
                balanced_recall=(dr+er)/2 if dr is not None and er is not None else None)

def coord_summary(errors, total):
    a = np.array(errors, dtype=np.float64)
    return dict(n_eligible=total, n_available=len(a), n_unavailable=total-len(a),
                mean_px=float(a.mean()) if len(a) else None,
                median_px=float(np.median(a)) if len(a) else None,
                p90_px=float(np.percentile(a, 90)) if len(a) else None,
                p95_px=float(np.percentile(a, 95)) if len(a) else None,
                within3px=ratio(int((a <= 3).sum()), total),
                within5px=ratio(int((a <= 5).sum()), total),
                within10px=ratio(int((a <= 10).sum()), total))

def transitions(truth, before, after):
    counts = {t+'_'+p+'_'+q: sum(a == t and b == p and c == q
                               for a, b, c in zip(truth, before, after))
              for t in ('D','E') for p in ('D','E','U') for q in ('D','E','U')}
    summary = dict(BOTH_CORRECT=0, JOINT_ONLY_CORRECT=0,
                   VIS_ONLY_CORRECT=0, BOTH_WRONG=0)
    for t, a, b in zip(truth, before, after):
        name = 'BOTH_CORRECT' if a == t and b == t else (
            'JOINT_ONLY_CORRECT' if b == t else 'VIS_ONLY_CORRECT' if a == t else 'BOTH_WRONG')
        summary[name] += 1
    summary['net_correct_JOINT_minus_VIS_ONLY'] = summary['JOINT_ONLY_CORRECT'] - summary['VIS_ONLY_CORRECT']
    return dict(transitions=counts, outcomes=summary)

def coordinate_arrays(identity, predictions):
    result = {k: np.array([f[source] for f in identity], dtype=np.float64)
              for k, source in (('OLD_R0_SOURCE', 'original_R0_q0'),
                                ('OLD_N3_SOURCE', 'original_N3_qN'),
                                ('OLD_N3_SUBPIX_INPUT', 'stored_final_native_input'))}
    for arm in ('R0_MATCHED_SQUARE640', *ALL_ARMS):
        source = 'R0_xy_original' if arm == 'R0_MATCHED_SQUARE640' else arm+'_xy_original'
        result[arm] = np.asarray(predictions[source], dtype=np.float64)
    return result

def build_labels(predictions, historical):
    available = np.asarray(predictions['frame_available'], dtype=bool)
    labels = {name: list(seq) for name, seq in historical.items()}
    for arm in SEMANTIC_ARMS:
        p = np.asarray(predictions[arm+'_E_probability'])[:, :8].reshape(-1)
        labels[arm] = np.where(np.repeat(available, 8), np.where(p >= .5, 'E', 'D'), 'U').tolist()
    return labels

def compute(rows, predictions, identity, historical):
    """Pure arithmetic; no reads. Native slots are fixed before this function."""
    n = len(rows)
    assert n == 1784 and len(identity) == 223
    truth = [x['truth'] for x in rows]
    labels = build_labels(predictions, historical)
    available = np.asarray(predictions['frame_available'], dtype=bool)
    cohort = {
        'PRIMARY_PAIRED1237': [i for i,r in enumerate(rows) if r['primary'] and r['truth'] in ('D','E')],
        'ALL_REGISTERED_D_E': [i for i,r in enumerate(rows) if r['truth'] in ('D','E')],
        'MANUAL_D_ALL': [i for i,r in enumerate(rows) if r['manual_D']],
        'MANUAL_D_INITIAL_WITHIN3': [i for i,r in enumerate(rows) if r['manual_D'] and r['old_near3']],
    }
    primary = cohort['PRIMARY_PAIRED1237']
    assert len(primary) == 1237
    assert sum(truth[i] == 'D' for i in primary) == 1147
    assert sum(truth[i] == 'E' for i in primary) == 90
    metrics, changes = {}, {}
    for name, indices in cohort.items():
        metrics[name], changes[name] = {}, {}
        for group in ('ALL', 'WOOD_PRIMARY105', 'SQUARE118'):
            ids = [i for i in indices if group == 'ALL' or rows[i]['group'] == group]
            t = [truth[i] for i in ids]
            metrics[name][group] = {arm: cm(t, [p[i] for i in ids]) for arm,p in labels.items()}
            changes[name][group] = transitions(t, [labels['VIS_ONLY'][i] for i in ids],
                                               [labels['JOINT'][i] for i in ids])
    self_counts = {}
    for group in ('ALL','WOOD_PRIMARY105','SQUARE118'):
        ids = [i for i,r in enumerate(rows) if r['truth'] == 'S' and (group == 'ALL' or r['group'] == group)]
        self_counts[group] = dict(n=len(ids), predictions={
            arm: {k: sum(p[i] == k for i in ids) for k in ('D','E','U')}
            for arm,p in labels.items()})
    coords = coordinate_arrays(identity, predictions)
    coordinates = {}
    for cohort_name in ('MANUAL_D_ALL','MANUAL_D_INITIAL_WITHIN3'):
        coordinates[cohort_name] = {}
        for group in ('ALL','WOOD_PRIMARY105','SQUARE118'):
            ids = [i for i in cohort[cohort_name] if group == 'ALL' or rows[i]['group'] == group]
            summary = {}
            for arm, xy in coords.items():
                errs = [float(np.linalg.norm(xy[i//8, i%8] - np.asarray(rows[i]['manual_xy'])))
                        for i in ids if not arm in (*ALL_ARMS,'R0_MATCHED_SQUARE640') or available[i//8]]
                summary[arm] = coord_summary(errs,len(ids))
            paired = [i for i in ids if available[i//8]]
            before = np.array([np.linalg.norm(coords['VIS_ONLY'][i//8,i%8] - rows[i]['manual_xy']) for i in paired])
            after = np.array([np.linalg.norm(coords['JOINT'][i//8,i%8] - rows[i]['manual_xy']) for i in paired])
            delta = after-before
            summary['JOINT_vs_VIS_ONLY_paired'] = dict(n=len(paired),
                mean_error_delta_px=float(delta.mean()) if len(delta) else None,
                improved_gt0_01px=int((delta < -.01).sum()), harmed_gt0_01px=int((delta > .01).sum()),
                unchanged_within0_01px=int((np.abs(delta) <= .01).sum()))
            coordinates[cohort_name][group] = summary
    return dict(PASS=True, frames=223, native_corner_rows=1784,
        semantic_threshold=.5, candidate_threshold=.001,
        new_available_frames=int(available.sum()),
        new_U_policy='frame_available only: frozen top detector confidence >= .001; old crop validity/old45 U/xy OOB/kobj do not gate semantic outputs',
        principal_comparison='VIS_ONLY vs JOINT on identical frame/native slots',
        XY_ONLY_visibility='N/A: semantic readout untrained; no objectness-to-visibility conversion',
        coordinate_scope='Secondary diagnostic: manually clicked DIRECT_VISIBLE references only; all eligible points independent of predicted E; no hidden-point coordinate truth claim',
        OLD_R0_SOURCE_coordinate_limit='Source q0 used original real preprocessing; new outputs use prospectively fixed current full-frame preprocessing. Old q0 is a contextual coordinate reference. R0_MATCHED_SQUARE640 is the unmodified R0 from the exact same frozen candidate/current forward and square640 preprocessing as the three new arms.',
        historical_reference_limit='Saved historical models may use old query availability and different input/target definitions; PRIMARY fixes original human native identities and subjects but does not equate tasks.',
        no_new_GT_matching=True, centroid_excluded=True, human_state_census={s:sum(r.get('human_state', r['truth'])==s for r in rows) for s in sorted(set(r.get('human_state',r['truth'])for r in rows))},
        metrics=metrics, VIS_ONLY_to_JOINT=changes, SELF_separate=self_counts,
        manual_D_coordinate_diagnostic=coordinates)

def validate_predictions(predictions):
    assert np.array_equal(predictions['frame_index'], np.arange(223))
    a = np.asarray(predictions['frame_available'])
    assert a.shape == (223,) and a.dtype == np.bool_
    candidate = predictions['candidate_score']
    assert candidate.shape == (223,) and np.isfinite(candidate).all()
    assert np.array_equal(a, candidate >= .001)
    assert predictions['anchor_index'].shape == (223,)
    assert predictions['raw_to_input_affine'].shape == (223,2,3)
    for arm in ('R0',*ALL_ARMS):
        xy=predictions[arm+'_xy_original']; assert xy.shape == (223,9,2) and np.isfinite(xy).all()
        z=predictions[arm+'_canvas_logit']; assert z.shape == (223,9) and np.isfinite(z).all()
    for arm in SEMANTIC_ARMS:
        p,z=predictions[arm+'_E_probability'],predictions[arm+'_E_logit']
        assert p.shape == z.shape == (223,9) and np.isfinite(p).all() and np.isfinite(z).all()
        assert np.logical_and(p>=0,p<=1).all()
        # Stable sigmoid, with only normal float32 export tolerance.
        recon = np.exp(-np.logaddexp(0.,-z.astype(np.float64)))
        assert np.max(np.abs(recon-p)) <= 1e-6
    assert 'XY_ONLY_E_probability' not in predictions
    k=predictions['R0_kobj_probability'];assert k.shape == (223,9) and np.isfinite(k).all()

def read_ledger(identity):
    source = safe(R/'V263_CURRENT_EVALUATION_A/actual_evaluation_v1/ALL1784_CLASSIFICATION_LOCATION_JOIN.csv')
    assert binding(source)['sha256'] == LEDGER_SHA
    with source.open(newline='') as f:
        raw = list(csv.DictReader(f))
    assert len(raw) == 1784
    rows=[]
    for i,r in enumerate(raw):
        fi,ni=i//8,i%8
        assert int(r['global_index']) == i and int(r['frame_index']) == fi
        assert int(r['native_corner']) == ni and r['id'] == identity[fi]['id']
        assert r['row_id'] == r['id']+':native'+str(ni)
        assert r['group'] == identity[fi]['group']
        state = r['human_state']; assert state in (D,E,S,'UNANNOTATED','OUT_OF_FRAME','UNKNOWN_MAPPING'), state
        manual = state == D and r['annotation_coordinate_source'] == 'manual_click'
        ref = [float(r['human_reference_x']),float(r['human_reference_y'])] if manual else None
        if manual: assert np.isfinite(ref).all()
        old_query = [float(r['query_x']), float(r['query_y'])]
        assert np.array_equal(np.asarray(old_query), np.asarray(identity[fi]['stored_final_native_input'][ni]))
        err=float(r['initial_reference_error_px']) if manual else None
        rows.append(dict(global_index=i,frame_index=fi,native_slot=ni,frame_id=r['id'],row_id=r['row_id'],
            group=r['group'],truth={D:'D',E:'E',S:'S','UNANNOTATED':'UNKNOWN','OUT_OF_FRAME':'UNKNOWN','UNKNOWN_MAPPING':'UNKNOWN'}[state],human_state=state,
            primary=r['paired_same_query_RGB_D_vs_E']=='True',manual_D=manual,manual_xy=ref,
            old_near3=manual and r['original_query_valid']=='True' and err<=3,
            old_query_xy=old_query, old_initial_reference_error_px=err,
            legacy_prediction=r['legacy_D_E_U']))
    return rows, binding(source)

def historical_predictions(rows):
    result={'CAL_LEGACY':[r['legacy_prediction'] for r in rows]}; bindings={}
    for dirname,arms in (
        ('V264_HEATMAP_LOCAL_VISIBILITY',('HEATMAP_LOCAL',)),
        ('V267_FULL_IMAGE_CLIP_VISIBLE_MASK',('FULL_CLIP','FULL_CLIP_SURGERY_ARCH')),
        ('V269_FULL_IMAGE_DECLIP_MATCHED_BACKBONE',('EVA_STANDARD','EVA_CSA','EVA_DECLIP'))):
        p=safe(R/dirname/'CURRENT_ALL1784_PREDICTIONS.jsonl')
        values=[json.loads(line) for line in p.read_text().splitlines() if line]
        assert len(values)==1784
        for i,r in enumerate(values):
            assert int(r['global_index']) == i
            if 'frame_index' in r: assert int(r['frame_index']) == i//8
            if 'row_id' in r: assert r['row_id']==rows[i]['row_id']
        for arm in arms:
            result[arm]=[r[arm]['pred'] for r in values]
            assert all(x in ('D','E','U') for x in result[arm])
        bindings[dirname]=binding(p)
    return result,bindings

def save_tables(out, rows, predictions, identity, historical, results):
    labels=build_labels(predictions,historical); coords=coordinate_arrays(identity,predictions)
    table=[]
    for i,r in enumerate(rows):
        t=dict(r);t.pop('manual_xy');t.pop('old_query_xy')
        t['old_query_x'],t['old_query_y']=r['old_query_xy']
        t['frame_available']=bool(predictions['frame_available'][i//8])
        t['anchor_index']=int(predictions['anchor_index'][i//8]);t['candidate_score']=float(predictions['candidate_score'][i//8])
        t['manual_reference_x'],t['manual_reference_y']=r['manual_xy'] if r['manual_xy'] is not None else (None,None)
        for name, values in labels.items():t[name+'_D_E_U']=values[i]
        for arm in SEMANTIC_ARMS:
            t[arm+'_p_E']=float(predictions[arm+'_E_probability'][i//8,i%8])
            t[arm+'_E_logit']=float(predictions[arm+'_E_logit'][i//8,i%8])
        t['R0_kobj_auxiliary']=float(predictions['R0_kobj_probability'][i//8,i%8])
        for arm,xy in coords.items():
            t[arm+'_x'],t[arm+'_y']=xy[i//8,i%8].tolist()
            if r['manual_D']:
                t[arm+'_manual_D_error_px']=float(np.linalg.norm(xy[i//8,i%8]-r['manual_xy']))
        table.append(t)
    def write(name, values):
        keys=list(dict.fromkeys(k for r in values for k in r))
        with safe(out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(values)
    write('ALL1784_CLASSIFICATION_COORDINATE_JOIN.csv',table)
    write('PRIMARY_EXTERNAL90.csv',[x for x in table if x['primary'] and x['truth']=='E'])
    write('MANUAL_D_COORDINATES.csv',[x for x in table if x['manual_D']])
    flat=[]
    for cohort,groups in results['metrics'].items():
        for group,arms in groups.items():
            for arm,m in arms.items():
                r=dict(cohort=cohort,group=group,arm=arm)
                r.update({k:v for k,v in m.items() if k!='confusion_matrix'});r.update(m['confusion_matrix']);flat.append(r)
    write('METRICS.csv',flat)
    change=[]
    for x in table:
        if x['truth'] in ('D','E') and x['VIS_ONLY_D_E_U'] != x['JOINT_D_E_U']:
            change.append(x)
    write('VIS_ONLY_TO_JOINT_CHANGED_ROWS.csv',change)
    # Select deterministic panels before opening any RGB (this script never does).
    primary=[r for r in rows if r['primary'] and r['truth'] in ('D','E')]
    choices=[]
    for outcome in ('BOTH_CORRECT','JOINT_ONLY_CORRECT','VIS_ONLY_CORRECT','BOTH_WRONG'):
        for r in primary:
            i=r['global_index'];a,b=labels['VIS_ONLY'][i],labels['JOINT'][i];t=r['truth']
            o='BOTH_CORRECT' if a==t and b==t else 'JOINT_ONLY_CORRECT' if b==t else 'VIS_ONLY_CORRECT' if a==t else 'BOTH_WRONG'
            if o==outcome:
                choices.append(dict(global_index=i,reason='first_primary_'+outcome));break
    for group in ('WOOD_PRIMARY105','SQUARE118'):
        for truth in ('D','E'):
            for correct in (True,False):
                for r in primary:
                    i=r['global_index']
                    if r['group']==group and r['truth']==truth and (labels['JOINT'][i]==truth)==correct:
                        choices.append(dict(global_index=i,reason=f'first_{group}_{truth}_JOINT_correct_{correct}'));break
    for i in (200,370,908,1515,996,852,665,1354,300,800,1274):
        choices.append(dict(global_index=i,reason='fixed_historical_case'))
    unique=[]
    for x in choices:
        if x['global_index'] not in [y['global_index'] for y in unique]:unique.append(x)
    dump(out/'VISUAL_CASE_SELECTION_BEFORE_RGB.json',dict(PASS=True,selection='Deterministic indices after immutable prediction seal and late GT; no threshold/checkpoint/recipe selection',cases=unique))
    return table

def evaluate(out):
    assert not safe(out).exists(), 'Use a fresh evaluation output; never overwrite a result'
    source=safe(HERE/'CURRENT_PREDICTIONS_FROZEN.json');f=read_json(source)
    assert f['PASS'] and f['completed'] and f['human_GT_reads']==0
    predpath=verify(f['predictions_npz']); configpath=verify(f['configuration'])
    identitypath=verify(f['input_channel_native_identity'])
    assert binding(identitypath)['sha256']==IDENTITY_SHA
    identitydoc=read_json(identitypath);identity=identitydoc['records']
    assert identitydoc['PASS'] and identitydoc['human_E_D_label_values_read']==False
    assert identitydoc['raw_model_channel_to_current_native_slot9']==list(range(9))
    assert [r['frame_index'] for r in identity]==list(range(223))
    config=read_json(configpath);assert config['PASS']
    with np.load(safe(predpath),allow_pickle=False) as z:predictions={k:z[k] for k in z.files}
    validate_predictions(predictions)
    safe(out).mkdir()
    dump(out/'CONSUMER_SEAL_BEFORE_HUMAN_GT.json',dict(PASS=True,completed=True,
        completed_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),code=binding(Path(__file__).absolute()),
        prediction_freeze=binding(source),predictions_npz=binding(predpath),configuration=binding(configpath),
        identity=binding(identitypath),current_human_GT_values_read=0,historical_prediction_files_read=0,
        current_RGB_reads=0,new_model_forwards=0,new_fits=0,new_PnP=0,new_permutation_matching=0,
        new_availability='frame/candidate only; no crop/query/OOB/canvas confidence gates',
        semantic_sigmoid_parity_checked=True))
    rows,ledgerbinding=read_ledger(identity)
    historical,historicalbindings=historical_predictions(rows)
    results=compute(rows,predictions,identity,historical)
    dump(out/'RESULTS.json',results)
    save_tables(out,rows,predictions,identity,historical,results)
    replay=safe(out/'portable_replay_v1');replay.mkdir()
    compact=dict(rows=rows,identity=[{k:r[k] for k in ('frame_index','id','group','raw_hw',
                'original_R0_q0','original_N3_qN','stored_final_native_input')} for r in identity],historical=historical)
    payload=json.dumps(compact,separators=(',',':'),sort_keys=True,allow_nan=False).encode()
    safe(replay/'REPLAY_INPUTS.json.gz').write_bytes(gzip.compress(payload,mtime=0))
    shutil.copyfile(safe(predpath),safe(replay/'CURRENT_RAW223_PREDICTIONS.npz'))
    dump(replay/'REFERENCE_RESULTS.json',results)
    dump(replay/'MANIFEST.json',dict(PASS=True,inputs={n:{k:v for k,v in binding(replay/n).items() if k!='path'}
        for n in ('REPLAY_INPUTS.json.gz','CURRENT_RAW223_PREDICTIONS.npz','REFERENCE_RESULTS.json')},
        provenance=dict(ledger=ledgerbinding,historical_predictions=historicalbindings,
                        predictions=binding(predpath),prediction_freeze=binding(source),identity=binding(identitypath)),
        replay='python -B evaluate_initial_joint_v270.py --replay /absolute/path/to/portable_replay_v1 --output /absolute/new/replay/output',
        no_original_RGB_or_model_or_source_ledger_required=True))
    dump(out/'COMPLETE.json',dict(PASS=True,completed=True,code=binding(Path(__file__).absolute()),
        consumer_seal=binding(out/'CONSUMER_SEAL_BEFORE_HUMAN_GT.json'),results=binding(out/'RESULTS.json'),
        late_human_GT=ledgerbinding,historical_predictions=historicalbindings,
        outputs={p.name:binding(p) for p in (out/'ALL1784_CLASSIFICATION_COORDINATE_JOIN.csv',out/'PRIMARY_EXTERNAL90.csv',
            out/'MANUAL_D_COORDINATES.csv',out/'METRICS.csv',out/'VIS_ONLY_TO_JOINT_CHANGED_ROWS.csv',
            out/'VISUAL_CASE_SELECTION_BEFORE_RGB.json',replay/'MANIFEST.json')},
        new_fits=0,new_model_forwards=0,new_PnP=0,current_RGB_reads=0))
    print(json.dumps({a:results['metrics']['PRIMARY_PAIRED1237']['ALL'][a] for a in SEMANTIC_ARMS},indent=2))

def replay_run(source,out):
    manifest=read_json(source/'MANIFEST.json')
    for name,b in manifest['inputs'].items():
        actual=binding(source/name)
        assert actual['sha256']==b['sha256'] and actual['size_bytes']==b['size_bytes']
    payload=json.loads(gzip.decompress(safe(source/'REPLAY_INPUTS.json.gz').read_bytes()))
    with np.load(safe(source/'CURRENT_RAW223_PREDICTIONS.npz'),allow_pickle=False) as z:pred={k:z[k] for k in z.files}
    validate_predictions(pred)
    result=compute(payload['rows'],pred,payload['identity'],payload['historical'])
    reference=read_json(source/'REFERENCE_RESULTS.json')
    assert result==reference,'Saved-data arithmetic replay differs'
    assert not safe(out).exists();safe(out).mkdir()
    dump(out/'RESULTS.json',result)
    dump(out/'COMPLETE.json',dict(PASS=True,exact_reference_match=True,original_RGB_reads=0,
        original_annotation_reads=0,model_loads=0,new_fits=0,source_manifest=binding(source/'MANIFEST.json'),
        replay_code=binding(Path(__file__).absolute()),results=binding(out/'RESULTS.json')))
    print('PASS: compact CPU replay exactly matches all saved scientific numbers')

def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluate',action='store_true');p.add_argument('--replay',type=Path)
    p.add_argument('--output',type=Path);a=p.parse_args()
    assert bool(a.evaluate)!=bool(a.replay)
    if a.evaluate:evaluate(safe(a.output or HERE/'evaluation_v1'))
    else:
        assert a.output is not None
        replay_run(safe(a.replay),safe(a.output))

if __name__=='__main__':main()
