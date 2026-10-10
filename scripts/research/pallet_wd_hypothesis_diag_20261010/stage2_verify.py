"""Independent saved Stage2 numerical/rule audit; no F/PnP/model imports."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from . import verify as H

FACES = {'front': (0,1,2,3), 'back': (4,7,6,5), 'top': (0,4,5,1), 'bottom': (3,2,6,7), 'left': (0,3,7,4), 'right': (1,5,6,2)}
NAMES = ('long-face-front', 'short-face-front')
BOTTOM = (2,3,6,7)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',',':'), allow_nan=False).encode()).hexdigest()

def indicator(pose):
    available = bool(pose.get('available'))
    return dict(available=available,
                confusion_rate=float(available and pose['rotation_deg'] > 45 and abs(pose['yaw_deg']) >= 60),
                success_rate=float(available and pose['translation_cm'] < 5 and pose['rotation_deg'] < 5))

def vector(records):
    return H.vectors([{'pose': {'S0': row['pose']}} for row in records], 'S0')

def visibility(q, support):
    usable = [bool(support[i] and np.isfinite(q[i]).all() and not np.array_equal(q[i], [-1,-1])) for i in range(8)]
    facing, areas = {}, {}
    for name, indices in FACES.items():
        if not all(usable[i] for i in indices):
            facing[name] = areas[name] = None
        else:
            area = math.fsum(float(q[a,0]*q[b,1]-q[b,0]*q[a,1]) for a,b in zip(indices,indices[1:]+indices[:1])) / 2
            areas[name] = area;facing[name] = area > 0
    visible = [any(facing[name] is None or facing[name] for name, indices in FACES.items() if index in indices) for index in range(8)]
    retained = [index for index in range(8) if usable[index] and visible[index]]
    return usable, visible, retained, areas, facing

def candidate_score(check, candidate, row):
    actual = candidate['actual_pose']
    if not actual.get('available'):
        assert not candidate['success']
        return
    q = np.array(row['qFinal'], float);K = np.array(row['fixed_metadata']['K'], float)
    X = H.cuboid(actual['cf_extents']);Rcf = np.array(actual['R_cf']);t = np.array(actual['centroid'])
    camera = X @ Rcf.T + t
    uvz = np.vstack([camera, t]) @ K.T
    projected = uvz[:,:2] / uvz[:,2:]
    for field, indices in [('reprojection_rmse_9_px', list(range(9))), ('subset_center_rmse_px', candidate['subset_indices']+[8])]:
        expected = math.sqrt(math.fsum(float(value)**2 for value in (projected[indices]-q[indices]).ravel()) / len(indices)) if np.isfinite(q[indices]).all() else None
        check.close(candidate[field], expected, field)
    errors = [math.hypot(*(a-b)) for a,b in zip(projected[candidate['subset_indices']],q[candidate['subset_indices']])]
    expected = math.fsum(errors) / len(errors)
    check.close(candidate['reprojection_mean_8_px'], expected, 'fit subset residual')
    check.close(actual['reprojection_px'], expected, 'actual residual')
    xyz = row['fixed_metadata']['dimensions_pnp_WH_D_m']
    Q = np.eye(3) if abs(actual['cf_extents'][0]-xyz[0]) < 1e-6 else H.rotations(4)[1]
    physical = Rcf @ Q
    if row['source_flag']:
        physical = physical @ np.diag([1.,-1.,-1.])
    check.close(actual['R_physical'], physical.tolist(), 'physical basis')

def verify_choice(check, rule, choice, stage1):
    q = np.array(choice['qFinal'], float);selection = choice['selection']
    assert selection['reference_inputs'] is False and selection['rule'] == rule
    if rule == 'S1':
        usable, visible, retained, areas, facing = visibility(q, choice['prediction_support'])
        published = selection['visibility']
        assert published['visible_mask'] == visible and published['usable_mask'] == usable
        assert published['facing'] == facing and published['visible_supported_count'] == len(retained)
        for name, area in areas.items():
            check.close(published['areas'][name], area, 'visibility area')
        if len(retained) < 6:
            assert selection['fallback'] and selection['fallback_reason'] == 'VISIBLE_SUPPORTED_CORNERS_LT6'
            assert selection['actual_pose'] == stage1['actual_pose']['S0'] and selection['hyp'] == stage1['hypS0']
            assert isinstance(selection['candidates'], dict)
            for candidate in selection['candidates'].values():
                candidate_score(check, candidate, choice)
            return
        subsets = [retained]
    else:
        subsets = [list(range(8))] + [[index for index in range(8) if index != removed] for removed in range(8)]
    assert not selection['fallback']
    candidates = selection['candidates']
    assert len(candidates) == 2*len(subsets)
    valid = []
    for index, candidate in enumerate(candidates):
        assert candidate['hypothesis'] == NAMES[index%2] and candidate['subset_number'] == index//2
        assert candidate['subset_indices'] == subsets[index//2]
        candidate_score(check, candidate, choice)
        check.close(selection['candidate_scores'][index], candidate['subset_center_rmse_px'], 'candidate score')
        if candidate['success'] and candidate['subset_center_rmse_px'] is not None:
            valid.append((candidate['subset_center_rmse_px'],index,candidate))
    selected = min(valid, key=lambda item:(item[0],item[1]))[2] if valid else None
    assert selection['actual_pose'] == (selected['actual_pose'] if selected else {'available':False})
    assert selection['hyp'] == (selected['hypothesis'] if selected else None)
    assert selection['subset_indices'] == (selected['subset_indices'] if selected else [])

def compare(check, baseline, changed, result, population, rule):
    assert result['population'] == population and result['rule'] == rule
    groups = defaultdict(list)
    for row in baseline:
        groups[row['method']].append(row)
    changed_index = {(row['seed'],row['method'],row['id']):row for row in changed}
    assert len(changed_index) == len(changed)
    assert set(groups) == set(result['metrics']) == set(result['paired']) == set(result['failures'])
    for method, records in sorted(groups.items()):
        ids = sorted({row['id'] for row in records});seeds = sorted({row['seed'] for row in records})
        assert seeds == [1,2,3]
        before_index = {(row['seed'],row['id']):row for row in records}
        before = [[before_index[(seed,fid)] for fid in ids] for seed in seeds]
        after = [[changed_index[(seed,method,fid)] for fid in ids] for seed in seeds]
        frame_level = population == 'SYNTH_HELDOUT' or result['descriptive_only']
        draw = H.IndependentDraws(ids if frame_level else [row['session'] for row in before[0]])
        secondary = H.IndependentDraws([row['session'] for row in before[0]]) if population == 'SYNTH_HELDOUT' else None
        packet = result['metrics'][method];pair = result['paired'][method]
        assert packet['frames'] == len(ids) and packet['bootstrap'] == pair['bootstrap']
        metadata = packet['bootstrap']
        assert metadata['units'] == draw.n and metadata['master_frames'] == metadata['frames'] == len(ids)
        assert metadata['seed'] == 20260917 and metadata['resamples'] == 10000 and metadata['draws_sha256_uint16_le'] == draw.digest
        assert metadata['level'] == ('frame' if frame_level else 'cluster')
        vectors_before = [vector(rows) for rows in before];vectors_after = [vector(rows) for rows in after]
        averages_before = H.average(vectors_before);averages_after = H.average(vectors_after)
        for position, seed in enumerate(seeds):
            H.verify_summary(check,packet['per_seed'][str(seed)]['S0'],vectors_before[position],draw,range(len(ids)))
            H.verify_summary(check,packet['per_seed'][str(seed)][rule],vectors_after[position],draw,range(len(ids)))
        H.verify_summary(check,packet['seed_mean']['S0'],averages_before,draw,range(len(ids)))
        H.verify_summary(check,packet['seed_mean'][rule],averages_after,draw,range(len(ids)))
        contrasts = []
        for seed, vb, va in zip(seeds,vectors_before,vectors_after):
            contrasts.append({key:[a-b for a,b in zip(va[key],vb[key])] for key in (*H.NUMERIC,*H.RATES)})
        means = {key:[a-b for a,b in zip(averages_after[key],averages_before[key])] for key in (*H.NUMERIC,*H.RATES)}
        for key in (*H.NUMERIC,*H.RATES):
            target = pair['seed_mean'][key];expected = H.describe(means[key])
            check.close(target['delta'],expected['mean'],'paired delta')
            check.close(target['CI95'],draw.interval(means[key]),'paired CI95')
            assert target['paired_frames'] == expected['n'] and target['binary_before_seed_mean'] == key.endswith('_rate')
            if secondary:
                check.close(target['scenario_cluster_secondary_CI95'],secondary.interval(means[key]),'secondary CI95')
            deltas = []
            for seed, contrast in zip(seeds,contrasts):
                value = H.describe(contrast[key]);observed = pair['per_seed'][str(seed)][key]
                check.close(observed['delta'],value['mean'],'seed paired delta')
                check.close(observed['CI95'],draw.interval(contrast[key]),'seed paired CI')
                assert observed['paired_frames'] == value['n'] and observed['binary_before_seed_mean'] == key.endswith('_rate')
                if secondary:
                    check.close(observed['scenario_cluster_secondary_CI95'],secondary.interval(contrast[key]),'seed secondary CI')
                deltas.append(value['mean'])
            check.close(target['per_seed_delta'],deltas,'seed deltas')
            assert target['improved_seeds'] == sum(value is not None and (value>0 if key in ('success_rate','IoU3D') else value<0) for value in deltas)
        for seed, old, new in zip(seeds,before,after):
            target = result['failures'][method][str(seed)]
            fields = {name:[] for name in ('success_to_failure_ids','failure_to_success_ids','confusion_recovery_ids','confusion_damage_ids','hypothesis_change_ids','fallback_ids','unavailable_ids')}
            transitions=[]
            for a,b in zip(old,new):
                assert a['id'] == b['id'] and a['qFinal'] == b['qFinal']
                ia,ib = indicator(a['pose']),indicator(b['pose']);fid=a['id']
                for name,hit in [('success_to_failure_ids',ia['success_rate'] and not ib['success_rate']),('failure_to_success_ids',not ia['success_rate'] and ib['success_rate']),('confusion_recovery_ids',ia['confusion_rate'] and not ib['confusion_rate']),('confusion_damage_ids',not ia['confusion_rate'] and ib['confusion_rate']),('hypothesis_change_ids',a['hyp']!=b['hyp']),('fallback_ids',b.get('fallback')),('unavailable_ids',not ib['available'])]:
                    if hit:fields[name].append(fid)
                if a['hyp']!=b['hyp']:transitions.append(dict(id=fid,before=a['hyp'],after=b['hyp']))
            assert target['transitions'] == transitions
            for name,ids_expected in fields.items():
                assert target[name] == ids_expected and target[name.removesuffix('_ids')+'_count'] == len(ids_expected)
        print('INDEPENDENT_STAGE2_STATISTICS',rule,population,method,'PASS',flush=True)
    assert result['primary'] == result['paired']['N3_THEN_SUBPIX']['seed_mean']

def plane(normals, bottoms):
    median = np.array([H.quantile([normal[axis] for normal in normals],.5) for axis in range(3)])
    length = math.sqrt(math.fsum(float(value)**2 for value in median))
    if not math.isfinite(length) or length <= 1e-12:return None
    normal = median / length
    offset = H.quantile([math.fsum(float(a)*float(b) for a,b in zip(point,normal)) for point in bottoms],.5)
    return normal,offset

def truth_rows(stage1, geometry):
    result={};references={}
    for row in stage1:
        fid=row['id']
        if row['source_flag']:
            if geometry is None:continue
            G,t,xyz,oracle=geometry[fid]
            down=-G[:,1]
        else:
            Gcf=np.array(row['reference_R_cf']);cf=np.array(row['reference_cf_extents']);xyz=np.array(row['fixed_metadata']['dimensions_pnp_WH_D_m'])
            t=np.array(row['reference_bottom_center'])-Gcf@np.array([0,cf[1]/2,0])
            Q=np.eye(3) if abs(cf[0]-xyz[0])<1e-6 else H.rotations(4)[1]
            G=Gcf@Q;down=G[:,1];oracle=row['hypOracle']
        result[fid]=(G,t,xyz,oracle)
        references[fid]=dict(normal=down,bottom=t+down*xyz[1]/2)
    return result,references

def verify_geometry(check,row,truth):
    if truth is None:return 0
    G,t,xyz,oracle=truth
    proxy=dict(row,pose={'S0':row['pose'],'ORACLE':row['pose']},actual_pose={'S0':row['actual_pose'],'ORACLE':row['actual_pose']},
               pose_symmetry_order=row.get('pose_symmetry_order',row['fixed_metadata'].get('canonical_symmetry_order',2)),
               hypOracle=oracle,oracle_diagnostic_only=True,inference_reference_inputs=False)
    H.verify_geometry(check,proxy,truth)
    return int(row['pose']['available'])

def stage1_failure_lists(doc, populations):
    receipt=H.read(doc/'STAGE1_FAILURES.json');assert receipt['additional_F_calls']==0 and receipt['missing_in_full_denominator']
    checked=0
    for population,rows in populations.items():
        groups=defaultdict(list)
        for row in rows:groups[f"{row['backbone']}::{row['method']}::seed{row['seed']}"].append(row)
        assert set(groups)==set(receipt['populations'][population])
        for key,records in groups.items():
            target=receipt['populations'][population][key];assert target['frames']==len(records)
            fields=defaultdict(list)
            for row in sorted(records,key=lambda r:r['id']):
                a,b=indicator(row['pose']['S0']),indicator(row['pose']['ORACLE'])
                for arm,value in [('S0',a),('ORACLE',b)]:
                    for label,hit in [('confusion',value['confusion_rate']),('unsuccessful',not value['success_rate']),('unavailable',not value['available'])]:
                        if hit:fields[f'{arm}_{label}_ids'].append(row['id'])
                for label,hit in [('confusion_recovery',a['confusion_rate'] and not b['confusion_rate']),('confusion_damage',not a['confusion_rate'] and b['confusion_rate']),('success_damage',a['success_rate'] and not b['success_rate']),('success_recovery',not a['success_rate'] and b['success_rate']),('hypothesis_change',row['hypS0']!=row['hypOracle'])]:
                    if hit:fields[label+'_ids'].append(row['id'])
            for field,count in target['counts'].items():
                assert target[field+'_ids']==fields[field+'_ids'] and count==len(fields[field+'_ids']);checked+=1
    return checked

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--doc',type=Path,default=H.DOC);parser.add_argument('--source-root',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args();doc=args.doc;output=args.output or doc/'STAGE2_VERIFICATION.json';assert not output.exists()
    execution=H.read(doc/'EXECUTION_STAGE2.json');assert execution['status']=='COMPLETE' and execution['phase']=='STAGE2'
    source=H.read(doc/'SOURCE_LOCK_STAGE2.json');method=H.read(doc/'METHOD_LOCK.json')
    assert source['method_lock_sha256']==execution['method_lock_sha256']==H.sha(doc/'METHOD_LOCK.json')
    assert execution['source_lock_sha256']==H.sha(doc/'SOURCE_LOCK_STAGE2.json')
    assert execution['stage1_publication_sha256']==H.sha(doc/'STAGE1_PUBLICATION.json')
    assert H.read(doc/'STAGE1_PUBLICATION.json')['status']=='PASS'
    assert H.sha(H.__file__)==H.read(doc/'STAGE1_VERIFICATION.json')['source_sha256']
    assert method['status'].startswith('LOCKED') and source['status'].startswith('LOCKED')
    source_checks=0
    for key in ('new_core','original_core'):
        for binding in source[key]:
            root=(args.source_root or H.ROOT) if binding.get('owner')=='historical_source' or key=='original_core' else H.ROOT
            assert H.sha(root/binding['path'])==binding['sha256'],binding['path'];source_checks+=1
    for key in ('network_forwards','training_updates','additional_SubPix_calls','coordinates_changed','depth_model_inference'):
        assert execution[key]==0
    assert execution['S4_executed'] is False
    populations={p:H.rows(doc/f'STAGE1_ROWS_{p}.jsonl.gz') for p in ('REAL','SYNTH','AUX')}
    stage1_indices={p:{(r['seed'],r['method'],r['id']):r for r in rows} for p,rows in populations.items()}
    baselines={p:[dict(row,pose=row['pose']['S0'],hyp=row['hypS0']) for row in rows] for p,rows in populations.items()}
    geometry=None
    if args.source_root:
        path=args.source_root/'challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz'
        audit=H.read(doc/'INPUT_AUDIT.json');binding=next(b for b in audit['bindings'] if b['path'].endswith('GEOMETRY_SIDETABLE.npz'));assert H.sha(path)==binding['sha256']
        with np.load(path,allow_pickle=False) as archive:data={k:archive[k] for k in ('stems','R','t','dims','Xcf')}
        indices={str(stem):i for i,stem in enumerate(data['stems'])};geometry={}
        for fid in {r['id'] for r in populations['SYNTH']}:
            i=indices[fid];X=data['Xcf'][i];oracle=NAMES[0] if np.linalg.norm(X[1]-X[0])>np.linalg.norm(X[4]-X[0]) else NAMES[1]
            geometry[fid]=(data['R'][i],data['t'][i],data['dims'][i],oracle)
    truth={};references={}
    for p in ('REAL','SYNTH'):truth[p],references[p]=truth_rows(populations[p],geometry)
    check=H.Check();verified={};allresults={};geometry_count=0;selection_count=0;peer_checks=0;candidate_checks=0
    verdicts=H.read(doc/'VERDICT_STAGE2.json');assert verdicts==H.read(doc/'VERDICT.json') and verdicts['no_combined_method_verdict']
    for rule in ('S1','S2','S3'):
        allresults[rule]={}
        for population in ('SYNTH','REAL'):
            path=doc/f'RESULTS_{rule}_{population}.json'
            if not path.exists():
                assert rule in ('S1','S2') and population=='REAL'
                skip=H.read(doc/f'SKIPPED_{rule}_REAL.json');synthetic=allresults[rule]['SYNTH'];primary=synthetic['primary']
                worsened=primary['confusion_rate']['CI95'][0]>0 or primary['success_rate']['CI95'][1]<0
                assert worsened and skip['status']=='SKIPPED_SYNTH_WORSENED' and skip['actual_rule_selector_calls']==0
                assert skip['skipped_frame_ids']==sorted({r['id'] for r in populations['REAL']}) and skip['expected_rows']==3828 and skip['expected_frames']==319
                assert not (doc/f'STAGE2_ROWS_{rule}_REAL.jsonl.gz').exists() and not (doc/f'STAGE2_SELECTIONS_{rule}_REAL.jsonl.gz').exists()
                allresults[rule]['REAL']=skip;verified[rule+'_REAL']='SKIPPED_SYNTH_WORSENED';continue
            result=H.read(path);allresults[rule][population]=result
            selected=H.rows(doc/f'STAGE2_SELECTIONS_{rule}_{population}.jsonl.gz');changed=H.rows(doc/f'STAGE2_ROWS_{rule}_{population}.jsonl.gz')
            seal_path=doc/f'STAGE2_SELECTION_SEAL_{rule}_{population}.json';seal=H.read(seal_path);receipt=H.read(doc/f'EXECUTION_{rule}_{population}.json')
            assert len(selected)==len(changed)==seal['rows']==receipt['rows']
            assert seal['choice_sha256']==H.sha(doc/f'STAGE2_SELECTIONS_{rule}_{population}.jsonl.gz')
            assert seal['source_lock_sha256']==H.sha(doc/'SOURCE_LOCK_STAGE2.json')
            if rule!='S3':
                assert seal['method_lock_sha256']==H.sha(doc/'METHOD_LOCK.json')
                assert seal['references_consumed_by_selector'] is False and seal['human_visibility_inputs'] is False
                assert seal['coordinate_changes']==seal['SubPix_calls']==seal['model_forwards']==seal['training_updates']==0
                assert receipt['choice_sha256']==seal['choice_sha256'] and receipt['selection_seal_sha256']==H.sha(seal_path)
                assert receipt['scored_rows_sha256']==H.sha(doc/f'STAGE2_ROWS_{rule}_{population}.jsonl.gz')
            else:
                assert seal['new_PnP_fits']==receipt['new_PnP_fits']==0 and receipt['reused_stage1_candidate_fits']
            for key in ('SubPix_calls','network_forwards','training_updates'):assert receipt[key]==0
            selected_index={(r['seed'],r['method'],r['id']):r for r in selected};assert len(selected_index)==len(selected)
            baseline=[r for r in baselines[population] if (r['seed'],r['method'],r['id']) in selected_index]
            s3eligibility=None;eligible_sessions=[]
            if rule=='S3':
                eligibility_path=doc/f'S3_ELIGIBILITY_{population}.json';s3eligibility=H.read(eligibility_path)
                assert seal['eligibility_sha256']==receipt['eligibility_sha256']==H.sha(eligibility_path)
                if population=='REAL':
                    groups=defaultdict(list)
                    for r in populations['REAL']:groups[r['session']].append(r['id'])
                    for session in groups:groups[session]=sorted(set(groups[session]))
                    assert set(groups)==set(s3eligibility['sessions'])
                    for session,ids in sorted(groups.items()):
                        normals=[references['REAL'][fid]['normal'] for fid in ids];bottoms=[references['REAL'][fid]['bottom'] for fid in ids]
                        pooled=plane(normals,bottoms);offsets=[math.fsum(float(a)*float(b) for a,b in zip(n,p)) for n,p in zip(normals,bottoms)]
                        sd=H.describe(offsets)['std'];angles=[math.degrees(math.acos(min(1.,max(-1.,math.fsum(float(a)*float(b) for a,b in zip(n,pooled[0])))))) for n in normals] if pooled is not None else []
                        rms=math.sqrt(math.fsum(a*a for a in angles)/len(angles)) if angles else None
                        eligible=len(ids)>1 and sd is not None and rms is not None and sd<=.05 and rms<=2
                        packet=s3eligibility['sessions'][session]
                        assert packet['frames']==len(ids) and packet['frame_ids']==ids and packet['eligible']==eligible
                        check.close(packet['offset_SD_m'],sd,'S3 session offset SD');check.close(packet['normal_angle_RMS_deg'],rms,'S3 session normal RMS',absolute=2e-5)
                        check.close(packet['componentwise_median_normal'],None if pooled is None else pooled[0].tolist(),'S3 normal')
                        if eligible:eligible_sessions.append(session)
                    eligible_ids=sorted(fid for session in eligible_sessions for fid in groups[session])
                    assert s3eligibility['eligible_sessions']==eligible_sessions and s3eligibility['eligible_ids']==eligible_ids
                    assert s3eligibility['own_GT_used_in_LOO_plane'] is False
                    assert set(r['id'] for r in selected)==set(eligible_ids)
                else:
                    assert s3eligibility['status']=='ORACLE_DIAGNOSTIC_ONLY' and s3eligibility['frames']==1985
            for row in changed:
                key=row['seed'],row['method'],row['id'];original=selected_index[key];previous=stage1_indices[population][key]
                assert row['qFinal']==original['qFinal']==previous['qFinal'] and row['fixed_metadata']==original['fixed_metadata']==previous['fixed_metadata']
                assert row['reference_seal_sha256']==H.sha(seal_path)
                choice=original['selection'];assert row['actual_pose']==choice['actual_pose'] and row['hyp']==choice['hyp'] and row['fallback']==choice['fallback']
                if rule!='S3':
                    verify_choice(check,rule,original,previous);selection_count+=1
                    candidate_checks+=len(choice['candidates'])
                    assert row['candidates']==choice['candidates'] and row['candidate_scores']==choice['candidate_scores']
                elif references[population]:
                    fid=row['id']
                    if population=='REAL':
                        peers=[other for other in s3eligibility['sessions'][row['session']]['frame_ids'] if other!=fid]
                        assert fid not in peers and original['own_reference_excluded'] and not original['oracle_diagnostic_only']
                        assert original['reference_peer_ids']==peers and original['reference_peer_count']==len(peers) and original['reference_peer_ids_sha256']==digest(peers)
                        pooled=plane([references[population][other]['normal'] for other in peers],[references[population][other]['bottom'] for other in peers]);peer_checks+=1
                    else:
                        assert original['reference_peer_ids']==[] and original['reference_peer_count']==0 and original['oracle_diagnostic_only']
                        ref=references[population][fid];pooled=(ref['normal'],math.fsum(float(a)*float(b) for a,b in zip(ref['normal'],ref['bottom'])))
                    check.close(original['plane_normal'],None if pooled is None else pooled[0].tolist(),'S3 LOO normal')
                    check.close(original['plane_offset_m'],None if pooled is None else pooled[1],'S3 LOO offset')
                    scores={};proposals=[]
                    for order,name in enumerate(NAMES):
                        actual=previous['candidates'][name]['actual_pose']
                        if pooled is None or not actual.get('available'):scores[name]=None;continue
                        corners=H.cuboid(actual['cf_extents'])@np.array(actual['R_cf']).T+np.array(actual['centroid'])
                        distances=[abs(math.fsum(float(a)*float(b) for a,b in zip(point,pooled[0]))-pooled[1]) for point in corners[list(BOTTOM)]]
                        scores[name]=math.fsum(distances)/4;proposals.append((scores[name],order,name))
                    if pooled is None:
                        assert choice['status']=='INVALID_REFERENCE_PLANE' and not choice['actual_pose']['available']
                    else:
                        for name,value in scores.items():check.close(choice['plane_scores_m'][name],value,'S3 plane score')
                        winner=min(proposals) if proposals else None
                        assert choice['hyp']==(winner[2] if winner else None)
                        assert choice['actual_pose']==(previous['candidates'][winner[2]]['actual_pose'] if winner else {'available':False})
                if row['id'] in truth[population]:
                    geometry_count+=verify_geometry(check,dict(row,pose_symmetry_order=previous['pose_symmetry_order']),truth[population][row['id']])
            if changed:
                compare(check,baseline,changed,result,'SYNTH_HELDOUT' if population=='SYNTH' else 'REAL_DEV',rule)
            else:
                assert rule=='S3' and result['primary'] is None and result['not_estimable_metadata'] and not result['metrics']
            if rule=='S3':
                assert result['verdict']==('FEASIBILITY_ONLY' if population=='REAL' else 'ORACLE_DIAGNOSTIC_ONLY')
            for field in ('metrics','paired','failures'):
                assert H.read(doc/f'{field.upper()}_{rule}_{population}.json')[field]==result[field]
            verified[rule+'_'+population]='PASS_'+str(len(changed))
        if rule!='S3':
            synth=allresults[rule]['SYNTH']['primary'];real=allresults[rule]['REAL'].get('primary') if 'metrics' in allresults[rule]['REAL'] else None
            worse=lambda p:p['confusion_rate']['CI95'][0]>0 or p['success_rate']['CI95'][1]<0
            gate_worse=worse(synth)
            assert allresults[rule]['SYNTH']['synthetic_gate']==dict(verdict='WORSENED' if gate_worse else 'UNRESOLVED',run_real=not gate_worse,basis='synthetic frame-bootstrap primary')
            expected='WORSENED' if gate_worse or (real is not None and worse(real)) else 'UNRESOLVED'
            if expected!='WORSENED' and real is not None and real['confusion_rate']['CI95'][1]<0 and synth['confusion_rate']['delta']<0 and synth['confusion_rate']['improved_seeds']>=2 and real['success_rate']['CI95'][1]>=0 and synth['success_rate']['CI95'][1]>=0:expected='SUPPORTED'
            assert verdicts['rules'][rule]['verdict']==expected
        else:
            assert verdicts['rules']['S3']['verdict']=='FEASIBILITY_ONLY' and verdicts['rules']['S3']['synthetic_oracle_has_method_verdict'] is False
    for field in ('metrics','paired','failures'):
        wrapper=H.read(doc/f'{field.upper()}.json');assert wrapper['combined_method_verdict'] is False
        for rule,population_results in allresults.items():
            for population,result in population_results.items():assert wrapper['rules'][rule][population]==result.get(field,result)
    failure_count=stage1_failure_lists(doc,populations)
    result=dict(status='PASS',phase='STAGE2',independent_numeric_comparisons=check.comparisons,
                max_absolute_difference=check.max_absolute,max_scaled_relative_difference=check.max_relative,
                rules=verified,source_bindings_checked=source_checks,independent_candidate_score_checks=candidate_checks,
                independently_verified_choices=selection_count,independent_pose_T_R_yaw_ADD_count=geometry_count,
                S3_leave_own_frame_out_checks=peer_checks,SYNTH_geometry_checked=geometry is not None,
                Stage1_failure_list_count_checks=failure_count,frozen_coordinates_exact=True,
                primary_and_secondary_bootstrap_verified=True,all_transition_IDs_verified=True,
                rule_specific_verdicts_verified=True,combined_method_verdict=False,
                additional_F_calls=0,additional_PnP_calls=0,additional_model_forwards=0,training_updates=0,
                IoU3D_statistics_verified=True,IoU3D_geometry_independently_recomputed=False,
                source_sha256=H.sha(__file__),independent_stage1_helper_sha256=H.sha(H.__file__))
    with output.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print('INDEPENDENT_STAGE2_PASS',check.comparisons,check.max_absolute,flush=True)

if __name__=='__main__':main()
