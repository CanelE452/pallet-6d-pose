"""Independent saved Metric3D depth/accuracy/S4 audit; no model or PnP imports.

The official network is never constructed here. Private native depth maps,
frozen predicted coordinates and reference geometry are read to independently
check the completed experiment, including a failed accuracy gate.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import tarfile
import cv2
import numpy as np
from . import verify as H
from . import stage2_verify as J

FINAL4 = {'eval_pallet07', 'eval_pallet09', 'eval_night08', 'eval_night09'}
PRIMARY = 'N3_THEN_SUBPIX'
MODEL_SOURCE_SHA = '6859f6024544916afe44adcbbcf7e5837032db732e406a5129df44240c2dc223'
MODEL_WEIGHT_SHA = 'b34b2a2be9148054991cef7e417930e1320602ba7bc503b0ee4e7888543728f6'
SOURCE_COMMIT = 'eb5b6fac0dc155e4e52f576e304fbf11655ff339'
HF_REVISION = '80d2d1410afb4b23cd9d18c6be9144483d4b70b6'


def key(row):
    return row['id'], row['method'], row['seed']


def independent_roi(depth, predicted):
    coordinates = np.asarray(predicted, dtype=float)
    if depth.ndim != 2 or coordinates.shape != (9, 2) or not np.isfinite(coordinates[:4]).all() or np.any(np.all(coordinates[:4] == -1, axis=1)):
        return dict(available=False, reason='invalid_depth_or_predicted_quad', pixels=0, median_m=None)
    centroid = np.array([math.fsum(float(value) for value in coordinates[:4, axis])/4 for axis in range(2)])
    quad = np.array([[centroid[axis]+.85*(float(point[axis])-centroid[axis]) for axis in range(2)] for point in coordinates[:4]])
    mask = np.zeros(depth.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(quad).astype(np.int32)], 1)
    values = depth[np.logical_and(mask.astype(bool), np.logical_and(np.isfinite(depth), depth > 0))]
    if len(values):
        # The saved maps and producer's median have float32 arithmetic. Use an
        # independent order-statistic implementation with the same precision.
        ordered = np.sort(values)
        center = len(ordered)//2
        median = float(ordered[center]) if len(ordered)%2 else float(np.float32(ordered[center-1]+ordered[center])*np.float32(.5))
    else:
        median = None
    return dict(available=bool(len(values)), reason=None if len(values) else 'no_positive_finite_depth_in_quad',
                pixels=len(values), median_m=median, shrink_factor=.85, quad_native_px=quad.tolist())


def compare_measurement(check, published, expected):
    assert published['available'] == expected['available'] and published['reason'] == expected['reason'] and published['pixels'] == expected['pixels']
    check.close(published['median_m'], expected['median_m'], 'front ROI median', absolute=1e-9)
    if 'quad_native_px' in expected:
        check.close(published['quad_native_px'], expected['quad_native_px'], 'shrunk ROI vertices', absolute=1e-9)
        assert published['shrink_factor'] == .85
    else:
        assert 'quad_native_px' not in published


def describe_check(check, observed, values):
    expected = H.describe(values)
    for name, value in expected.items():
        check.close(observed[name], value, 'depth '+name, absolute=1e-10)
    # Accuracy statistics were prespecified as descriptive moments/quantiles;
    # producer does not claim a bootstrap CI for these depth distributions.
    assert observed['CI95'] is None
    if expected['n']:
        assert observed['variance_ddof'] == 1 and observed['CI95_target'] == 'mean'


def official_artifacts(modelroot):
    assert H.sha(modelroot/'official-source.tar.gz') == MODEL_SOURCE_SHA
    weight = modelroot/'metric_depth_vit_small_800k.pth'
    assert weight.stat().st_size == 150120967 and H.sha(weight) == MODEL_WEIGHT_SHA
    receipt = H.read(modelroot/'FETCH_RECEIPT.json')
    assert receipt['source_commit'] == SOURCE_COMMIT and receipt['checkpoint_repo_revision'] == HF_REVISION
    files = 0
    with tarfile.open(modelroot/'official-source.tar.gz', 'r:gz') as archive:
        for member in archive.getmembers():
            relative = Path(*Path(member.name).parts[1:])
            if not member.isfile() or not relative.parts:
                continue
            if relative.suffix == '.py' and (relative.parts[0] == 'mono' or relative.name == 'hubconf.py'):
                assert H.sha(modelroot/'Metric3D'/relative) == hashlib.sha256(archive.extractfile(member).read()).hexdigest()
                files += 1
    assert files == 49
    return files


def runtime_amendment(doc,private):
    path=doc/'STAGE3_RUNTIME_AMENDMENT.json'
    if not path.exists():return None
    amendment=H.read(path)
    assert amendment['status']=='LOCKED_BEFORE_FIRST_DEPTH_FORWARD'
    assert amendment['sole_missing_key']=='depth_model.encoder.mask_token' and amendment['unexpected_keys']==[]
    fresh=amendment.get('kind')=='FRESH_REPRODUCTION_OFFICIAL_LOADER'
    assert amendment['actual_prior_model_constructions']==(0 if fresh else 2) and amendment['actual_prior_model_forwards']==0
    assert amendment['checkpoint_or_model_change'] is False and amendment['configuration_or_threshold_change'] is False and amendment['GT_accuracy_read_before_amendment'] is False
    binding=amendment['runtime_binding'];assert H.sha(H.ROOT/binding['path'])==binding['sha256']
    failed=private/'depth_accuracy_attempt_01'
    if fresh:
        assert not failed.exists() and 'diagnostic' not in amendment and 'previous_source_lock_path' not in amendment
        entry=amendment['fresh_entry_binding'];assert H.sha(H.ROOT/entry['path'])==entry['sha256']
    else:
        assert amendment.get('kind','HISTORICAL_GUARD_RECOVERY')=='HISTORICAL_GUARD_RECOVERY'
        assert amendment['previous_source_lock_path']=='SOURCE_LOCK_STAGE3.json' and amendment['previous_source_lock_sha256']==H.sha(doc/'SOURCE_LOCK_STAGE3.json')
        assert sorted(p.name for p in failed.iterdir())==['DEPTH_INFERENCE_LOCK.json']
        assert H.sha(failed/'DEPTH_INFERENCE_LOCK.json')==amendment['preserved_failed_private_lock_sha256']
        failed_lock=H.read(failed/'DEPTH_INFERENCE_LOCK.json');assert failed_lock['model_forwards']==failed_lock['models_constructed']==0
        diagnostic=H.read(private/'metric3d/CHECKPOINT_KEY_DIAGNOSTIC.json')
        assert diagnostic==amendment['diagnostic'] and diagnostic['missing_keys']==diagnostic['missing_parameters']==['depth_model.encoder.mask_token']
        assert diagnostic['unexpected_keys']==diagnostic['missing_buffers']==[] and diagnostic['unexpected_shapes']=={} and diagnostic['model_forwards']==0
    hub=(private/'metric3d/Metric3D/hubconf.py').read_text()
    small=hub[hub.index('def metric3d_vit_small('):hub.index('def metric3d_vit_large(')]
    assert 'strict=False' in small
    pipeline=(private/'metric3d/Metric3D/mono/model/model_pipelines/dense_pipeline.py').read_text()
    assert 'features = self.encoder(input)' in pipeline
    dino=(private/'metric3d/Metric3D/mono/model/backbones/ViT_DINO.py').read_text().splitlines()
    assert 'self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))' in dino[996]
    assert 'if masks is not None:' in dino[1121] and 'self.mask_token' in dino[1122]
    recovery=H.read(doc/'STAGE3_RECOVERY_EXECUTION.json')
    assert recovery['status']=='COMPLETE' and recovery['failed_guard_model_constructions']==recovery['diagnostic_model_constructions']==(0 if fresh else 1)
    assert recovery['prior_attempt_forwards']==recovery['training_updates']==0
    if fresh:assert recovery['kind']=='FRESH_REPRODUCTION_OFFICIAL_LOADER'
    else:assert recovery['existing_source_lock_preserved']
    assert recovery['runtime_source_sha256']==binding['sha256'] and recovery['runtime_amendment_path']==path.name
    return amendment


def rgb_bindings(source, root):
    real = H.read(root/'_docs/experiments/pallet_n3_subpix_final_20261010/INPUT_AUDIT.json')
    result = {('REAL', row['id']): (row['image']['path'], row['image']['sha256'], [0]*4) for row in real['inputs']}
    synth = H.read(source/'data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json')
    result.update({('SYNTH', row['id']): (row['image'], row['image_sha256'], [int(row['reflect_pad_px'])]*4)
                   for row in synth['records'] if row['partition'] == 'heldout'})
    return result


def verify_cache(check, cache_dir, expected_ids, selections, bindings, source, amendment=None, doc=None):
    manifest = H.read(cache_dir/'DEPTH_CACHE_MANIFEST.json')
    lock = H.read(cache_dir/'DEPTH_INFERENCE_LOCK.json')
    assert manifest['status'] == 'DEPTH_CACHE_COMPLETE' and manifest['private_only'] and not manifest['GT_used_for_inference']
    assert manifest['frames'] == manifest['model_forwards'] == len(expected_ids) and manifest['training_updates'] == 0
    assert manifest['fresh_model_selection'] is False
    assert lock['status'] == 'PINNED_BEFORE_MODEL_CREATION_AND_FORWARD'
    assert lock['models_constructed'] == lock['model_forwards'] == 0 and not lock['GT_used_for_inference']
    assert lock['frames'] == len(expected_ids) and H.sha(H.ROOT/'scripts/research/pallet_wd_hypothesis_diag_20261010/depth.py') == lock['adapter_source_sha256']
    for packet in (lock['model'], manifest['checkpoint']):
        assert packet['model']=='Metric3D-v2 ViT-Small RAFT-4' and packet['input_hw']==[616,1064]
        assert packet['source_commit'] == SOURCE_COMMIT and packet['hf_revision'] == HF_REVISION
        assert packet['checkpoint_sha256'] == MODEL_WEIGHT_SHA and packet['source_archive_sha256'] == MODEL_SOURCE_SHA
        assert packet['verified_extracted_source_files'] == 49 and packet['checkpoint_bytes'] == 150120967
        assert packet['source_hubconf_sha256']==H.sha(cache_dir.parent/'metric3d/Metric3D/hubconf.py')
    assert manifest['checkpoint']['models_constructed'] == 1 and manifest['checkpoint']['checkpoint_unexpected_keys'] == []
    if amendment:
        assert manifest['checkpoint']['checkpoint_missing_keys']==['depth_model.encoder.mask_token']
        assert manifest['checkpoint']['official_loader_strict'] is False and manifest['checkpoint']['missing_mask_token_unused_for_unmasked_RGB']
        assert manifest['checkpoint']['runtime_adapter_source_sha256']==amendment['runtime_binding']['sha256']
        assert manifest['checkpoint']['runtime_amendment_sha256']==H.sha((doc or H.DOC)/'STAGE3_RUNTIME_AMENDMENT.json')
    else:assert manifest['checkpoint']['checkpoint_missing_keys']==[]
    locked = {(row['population'], row['id']): row for row in lock['inputs']}
    cached = {(row['population'], row['id']): row for row in manifest['rows']}
    assert len(locked) == len(cached) == len(expected_ids) and set(locked) == set(cached) == set(expected_ids)
    input_sequence=[(row['population'],row['id']) for row in lock['inputs']]
    assert input_sequence==[(row['population'],row['id']) for row in manifest['rows']]
    measurements = {}; rgb_checks = 0; cache_checks = 0
    for number, entry in enumerate(manifest['rows'], 1):
        identity = entry['population'], entry['id']; population, fid = identity
        original = selections[population][fid][0]
        pin = locked[identity]
        assert set(pin) == {'id','population','rgb_sha256','K','raw_hw','crop_lrtb'}
        assert entry['K'] == pin['K'] == original['fixed_metadata']['K']
        assert entry['native_hw'] == pin['raw_hw'] == original['raw_hw']
        path, expected_sha, crop = bindings[identity]; path = Path(path)
        if not path.is_absolute(): path = source/path
        assert H.sha(path) == expected_sha == pin['rgb_sha256'] == entry['rgb_sha256']
        assert entry['crop_lrtb'] == pin['crop_lrtb'] == crop == ([100]*4 if population == 'SYNTH' else [0]*4)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert image is not None
        left, top, right, bottom = crop; height, width = image.shape[:2]
        native_h, native_w = height-top-bottom, width-left-right
        assert [native_h, native_w] == original['raw_hw']
        scale = min(616/native_h, 1064/native_w)
        resized = [int(native_h*scale), int(native_w*scale)]
        pad_h, pad_w = 616-resized[0], 1064-resized[1]
        pads = [pad_h//2,pad_h-pad_h//2,pad_w//2,pad_w-pad_w//2]
        K = np.array(original['fixed_metadata']['K'], dtype=float)
        assert K.shape == (3,3) and np.isfinite(K).all() and K[0,0]>0 and K[1,1]>0
        assert np.array_equal(K[2],[0.,0.,1.]) and K[0,1] == K[1,0] == 0
        check.close(entry['resize_scale'],scale,'native resize scale')
        assert entry['resized_hw'] == resized and entry['pads_tblr'] == pads
        check.close(entry['scaled_intrinsic'],[K[0,0]*scale,K[1,1]*scale,K[0,2]*scale,K[1,2]*scale],'scaled K')
        check.close(entry['canonical_to_real_scale'],K[0,0]*scale/1000,'metric depth scale')
        assert entry['model_forward'] == number and entry['depth_units'] == 'm' and entry['official_clip_m'] == [0.,300.]
        assert math.isfinite(entry['inference_seconds']) and entry['inference_seconds'] >= 0
        cache_name = hashlib.sha256((population+':'+fid).encode()).hexdigest()+'.npz'
        assert entry['cache'] == cache_name and H.sha(cache_dir/cache_name) == entry['cache_sha256']
        with np.load(cache_dir/cache_name,allow_pickle=False) as archive:
            assert archive.files == ['depth_m']; depth = archive['depth_m']
        assert depth.dtype == np.float32 and list(depth.shape) == original['raw_hw']
        finite = depth[np.isfinite(depth)]
        assert not len(finite) or (finite.min() >= 0 and finite.max() <= 300)
        for row in selections[population][fid]:
            measurements[(population,*key(row))] = independent_roi(depth,row['qFinal'])
        rgb_checks += 1; cache_checks += 1
        if number%100 == 0: print('INDEPENDENT_DEPTH_MAPS',number,len(expected_ids),flush=True)
    return manifest, measurements, rgb_checks, cache_checks


def reference_depths(source, ids):
    real = H.read(source/'data/pallet/results/paper_pose_metric_closure_v1/GEOMETRY_RESOLVED_POSE_GT.json')['frames']
    fronts = {}; geometry = {}
    for fid in ids['REAL']:
        row = real[fid.replace(':','__',1)];d = row['physical_dimensions_m']
        xyz = [d['across'],d['height'],d['along']]
        points = H.cuboid(xyz);R = np.array(row['R_gt_representative']);t = np.array(row['t_gt'])
        fronts['REAL',fid] = math.fsum(float(value) for value in (points[:4]@R.T+t)[:,2])/4
    path = source/'challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz'
    with np.load(path,allow_pickle=False) as archive:
        data = {name:archive[name] for name in ('stems','Xcf','R','t','dims')}
    lookup = {str(stem):index for index,stem in enumerate(data['stems'])}
    for fid in ids['SYNTH']:
        index = lookup[fid];X = data['Xcf'][index];R = data['R'][index];t = data['t'][index]
        fronts['SYNTH',fid] = math.fsum(float(value) for value in (X[:4]@R.T+t)[:,2])/4
        parity = J.NAMES[0] if np.linalg.norm(X[1]-X[0]) > np.linalg.norm(X[4]-X[0]) else J.NAMES[1]
        geometry[fid] = (R,t,data['dims'][index],parity)
    assert all(math.isfinite(value) and value>0 for value in fronts.values())
    return fronts,geometry


def verify_accuracy(check,population,doc,expected,fronts,frames):
    measurements = H.rows(doc/f'DEPTH_MEASUREMENTS_{population}.jsonl.gz')
    seal = H.read(doc/f'DEPTH_MEASUREMENT_SEAL_{population}.json')
    assert seal['status'] == 'SEALED_BEFORE_REFERENCE_DEPTH_ACCURACY' and seal['GT_input'] is False
    assert seal['rows'] == len(measurements) == frames*12 and seal['measurement_path'] == f'DEPTH_MEASUREMENTS_{population}.jsonl.gz'
    assert seal['measurement_sha256'] == H.sha(doc/seal['measurement_path'])
    index = {key(row):row for row in measurements}
    assert len(index) == len(measurements) and set(index) == {identifier[1:] for identifier in expected if identifier[0] == population}
    for identifier,row in index.items(): compare_measurement(check,row,expected[(population,*identifier)])
    scored = H.rows(doc/f'DEPTH_ACCURACY_ROWS_{population}.jsonl.gz')
    assert len(scored) == len(measurements)
    grouped = defaultdict(list)
    for row in scored:
        original = index[key(row)]
        for field in original: assert row[field] == original[field]
        z = fronts[population,row['id']]
        signed = (expected[(population,*key(row))]['median_m']-z)/z if row['available'] else None
        check.close(row['reference_front_mean_z_m'],z,'reference front depth',absolute=1e-10)
        check.close(row['relative_error'],signed,'relative depth error',absolute=1e-10)
        check.close(row['absolute_relative_error'],None if signed is None else abs(signed),'absolute depth error',absolute=1e-10)
        grouped[row['method']].append(dict(row,relative_error=signed,absolute_relative_error=None if signed is None else abs(signed)))
    published = H.read(doc/f'DEPTH_ACCURACY_{population}.json')
    assert published['population'] == population and published['threshold_model_selection_final_test_frames'] == 0 and set(published['methods']) == set(H.METHODS)
    derived = {}
    for method,records in sorted(grouped.items()):
        packet = published['methods'][method];by_id=defaultdict(list)
        assert packet['frames'] == frames and set(packet['per_seed']) == {'1','2','3'}
        for seed in (1,2,3):
            selected = sorted([row for row in records if row['seed'] == seed],key=lambda row:row['id'])
            valid = [row for row in selected if row['available']]
            observed = packet['per_seed'][str(seed)]
            assert observed['frames'] == len(selected) == frames and observed['available'] == len(valid)
            assert observed['unavailable_ids'] == [row['id'] for row in selected if not row['available']]
            describe_check(check,observed['absolute_relative_error'],[row['absolute_relative_error'] for row in valid])
            describe_check(check,observed['signed_relative_error'],[row['relative_error'] for row in valid])
            check.close(observed['signed_bias'],math.fsum(row['relative_error'] for row in valid)/len(valid) if valid else None,'seed signed bias',absolute=1e-10)
        for row in records: by_id[row['id']].append(row)
        means=[];biases=[];missing=[]
        for fid,values in sorted(by_id.items()):
            assert len(values) == 3 and {row['seed'] for row in values} == {1,2,3}
            if not all(row['available'] for row in values): missing.append(fid);continue
            means.append(math.fsum(abs(row['relative_error']) for row in values)/3)
            biases.append(math.fsum(row['relative_error'] for row in values)/3)
        assert packet['complete_three_seed_frames'] == len(means) and packet['unavailable_primary_ids'] == missing
        assert packet['aggregation'] == 'absolute errors first, then three-seed mean per frame, then frame median'
        describe_check(check,packet['seed_mean_absolute_relative_error'],means)
        describe_check(check,packet['signed_relative_error_seed_mean'],biases)
        check.close(packet['signed_bias'],math.fsum(biases)/len(biases) if biases else None,'seed mean signed bias',absolute=1e-10)
        derived[method]=dict(frames=frames,complete=len(means),missing=missing,median=H.quantile(means,.5) if means else None)
        print('INDEPENDENT_DEPTH_ACCURACY',population,method,'PASS',flush=True)
    return derived,measurements


def s4_choice(check,published,previous,measurement,margin):
    chosen=previous['selection']['hyp'];gap=previous['selection'].get('formal_score_gap_px');trigger=gap is not None and gap<margin
    depths={};available=bool(measurement and measurement['available'])
    if trigger and available:
        options=[]
        for position,(name,candidate) in enumerate(previous['selection']['candidates'].items()):
            actual=candidate['actual_pose']
            if not actual.get('available'):continue
            camera=H.cuboid(actual['cf_extents'])@np.array(actual['R_cf']).T+np.array(actual['centroid'])
            z=math.fsum(float(value) for value in camera[:4,2])/4
            depths[name]=z;options.append((abs(z-measurement['median_m']),position,name))
        if options:chosen=min(options)[2]
    assert published['hyp'] == chosen
    candidates=previous['selection']['candidates']
    assert published['actual_pose'] == (candidates[chosen]['actual_pose'] if chosen in candidates else previous['selection']['actual_pose'])
    for name in previous:
        if name != 'selection': assert published[name] == previous[name]
    assert published['rule'] == 'S4' and published['fallback'] == bool(trigger and not available)
    assert published['inference_reference_inputs'] is False and published['original_F_calls'] == published['new_PnP_calls'] == 0
    tie=published['depth_tie'];assert tie['m_px'] == margin and tie['trigger'] == trigger and tie['measured_depth_available'] == available
    check.close(tie['measured_front_depth_m'],None if measurement is None else measurement['median_m'],'S4 measured depth')
    assert set(tie['candidate_front_mean_z_m']) == set(depths)
    for name,z in depths.items():check.close(tie['candidate_front_mean_z_m'][name],z,'S4 candidate front depth')


def verify_s4(check,doc,selections,measured,geometry,original_rows,execution):
    truths={}
    for p in ('REAL','SYNTH'): truths[p],_=J.truth_rows(original_rows[p],geometry)
    indices={p:{key(row):row for records in selections[p].values() for row in records} for p in selections}
    baselines={p:[dict(row,pose=row['pose']['S0'],hyp=row['hypS0']) for row in original_rows[p]] for p in original_rows}
    primary_grids={};choice_count=0;pose_count=0
    lock=H.read(doc/'S4_MARGIN_LOCK.json')
    assert lock['status'] == 'FROZEN_ON_SYNTH_BEFORE_ANY_REAL_S4_SELECTION' and not lock['real_reference_for_margin_selection'] and not lock['final_test4_for_margin_selection']
    for label,population,margin in [(f'm{m}','SYNTH',m) for m in (1,2,3)]+[('S4','SYNTH',lock['margin_px']),('S4','REAL',lock['margin_px'])]:
        selected=H.rows(doc/f'STAGE3_SELECTIONS_{label}_{population}.jsonl.gz')
        seal=H.read(doc/f'STAGE3_SELECTION_SEAL_{label}_{population}.json')
        assert seal['status'] == 'SEALED_BEFORE_REFERENCE_POSE_SCORING' and seal['rows'] == len(selected)
        assert seal['selection_path'] == f'STAGE3_SELECTIONS_{label}_{population}.jsonl.gz' and seal['selection_sha256'] == H.sha(doc/seal['selection_path'])
        assert seal['margin_px'] == margin and seal['coordinates_changed'] is False and seal['GT_for_selection'] is False and seal['new_F_calls'] == seal['new_PnP_calls'] == 0
        chosen_index={key(row):row for row in selected};assert len(chosen_index) == len(selected)
        for row in selected:s4_choice(check,row,indices[population][key(row)],measured.get((population,*key(row))),margin);choice_count+=1
        if label != 'S4':
            assert len(selected) == 1985*3 and {row['method'] for row in selected} == {PRIMARY}
        else: assert len(selected) == (1985 if population=='SYNTH' else 319)*12
        scored=H.rows(doc/f'STAGE3_ROWS_{label}_{population}.jsonl.gz');assert len(scored) == len(selected)
        for row in scored:
            original=chosen_index[key(row)]
            for field in original:assert row[field] == original[field]
            proxy=dict(row,pose_symmetry_order=indices[population][key(row)]['pose_symmetry_order'])
            pose_count+=J.verify_geometry(check,proxy,truths[population][row['id']])
        selected_keys=set(chosen_index);baseline=[row for row in baselines[population] if key(row) in selected_keys]
        if label=='S4':
            result=H.read(doc/f'RESULTS_S4_{population}.json')
            J.compare(check,baseline,scored,result,'REAL_DEV' if population=='REAL' else 'SYNTH_HELDOUT','S4')
        else:
            # Margins were ranked on primary SYNTH binary confusion only.
            rate=math.fsum(J.indicator(row['pose'])['confusion_rate'] for row in scored)/len(scored)
            primary_grids[str(margin)]=rate
            check.close(lock['grid'][str(margin)]['confusion_rate'],rate,'S4 margin confusion')
            frame_ids=sorted({row['id'] for row in scored});draw=H.IndependentDraws(frame_ids)
            old_index={(row['seed'],row['id']):row for row in baseline};new_index={(row['seed'],row['id']):row for row in scored}
            vb=[J.vector([old_index[(seed,fid)] for fid in frame_ids]) for seed in (1,2,3)]
            va=[J.vector([new_index[(seed,fid)] for fid in frame_ids]) for seed in (1,2,3)]
            old_mean,new_mean=H.average(vb),H.average(va)
            for metric in (*H.NUMERIC,*H.RATES):
                delta=[a-b for a,b in zip(new_mean[metric],old_mean[metric])]
                target=lock['grid'][str(margin)]['primary'][metric]
                check.close(target['delta'],H.describe(delta)['mean'],'margin primary delta')
                check.close(target['CI95'],draw.interval(delta),'margin primary CI')
    winner=min((rate,int(m)) for m,rate in primary_grids.items())[1]
    assert lock['margin_px'] == winner == execution['margin_px']
    synth=H.read(doc/'RESULTS_S4_SYNTH.json')['primary'];real=H.read(doc/'RESULTS_S4_REAL.json')['primary']
    worse=lambda p:p['confusion_rate']['CI95'][0]>0 or p['success_rate']['CI95'][1]<0
    verdict='WORSENED' if worse(synth) or worse(real) else 'UNRESOLVED'
    if verdict!='WORSENED' and real['confusion_rate']['CI95'][1]<0 and synth['confusion_rate']['delta']<0 and synth['confusion_rate']['improved_seeds']>=2 and real['success_rate']['CI95'][1]>=0 and synth['success_rate']['CI95'][1]>=0:verdict='SUPPORTED'
    assert H.read(doc/'VERDICT_S4.json')['verdict'] == verdict
    return choice_count,pose_count,verdict


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--doc',type=Path,default=H.DOC)
    parser.add_argument('--source-root',type=Path,default=Path(os.environ.get('PALLET_SOURCE_ROOT',str(H.ROOT))))
    parser.add_argument('--private-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();doc=args.doc;source=args.source_root;private=args.private_dir
    output=args.output or doc/'DEPTH_VERIFICATION.json';assert not output.exists(),'Preserve completed independent evidence'
    gate=H.read(doc/'DEPTH_GATE.json');execution=H.read(doc/'STAGE3_EXECUTION.json');check=H.Check()
    assert H.read(doc/'STAGE2_REPORT_RECEIPT.json')['status'] == 'COMPLETE' and H.read(doc/'STAGE2_VERIFICATION.json')['status'] == 'PASS'
    source_lock=H.read(doc/'SOURCE_LOCK_STAGE3.json');assert source_lock['status'] == 'LOCKED_BEFORE_DEPTH_MODEL_CONSTRUCTION'
    for binding in source_lock['bindings']:assert H.sha(H.ROOT/binding['path']) == binding['sha256']
    core_path=doc/'STAGE1_FINAL_SOURCE_LOCK.json'
    core_lock=H.read(core_path if core_path.exists() else doc/'STAGE1_SOURCE_LOCK.json')
    for binding in core_lock['new_core']+core_lock['original_core']:
        root=source if binding['owner']=='historical_source' else H.ROOT
        assert H.sha(root/binding['path']) == binding['sha256']
    official_count=official_artifacts(private/'metric3d')
    amendment=runtime_amendment(doc,private)
    input_audit=H.read(doc/'INPUT_AUDIT.json')
    for binding in input_audit['bindings']:
        root=H.ROOT if binding['owner']=='published_worktree' else source
        assert H.sha(root/binding['path'])==binding['sha256']
    selections={};original_rows={};selection_bindings=0
    for population in ('REAL','SYNTH'):
        seal=H.read(doc/f'STAGE1_SELECTION_SEAL_{population}.json');assert H.sha(doc/seal['selection_path']) == seal['selection_sha256']
        values=H.rows(doc/seal['selection_path']);groups=defaultdict(list)
        for row in values:groups[row['id']].append(row)
        assert len(values) == (3828 if population=='REAL' else 23820)
        assert all(len(records)==12 and {row['method'] for row in records}==set(H.METHODS) and {row['seed'] for row in records}=={1,2,3} for records in groups.values())
        selections[population]=groups;original_rows[population]=H.rows(doc/f'STAGE1_ROWS_{population}.jsonl.gz');selection_bindings+=1
    expected_ids={('SYNTH',fid) for fid in selections['SYNTH']} | {('REAL',fid) for fid,records in selections['REAL'].items() if records[0]['session'] not in FINAL4}
    assert Counter(p for p,_ in expected_ids) == {'SYNTH':1985,'REAL':231}
    bindings=rgb_bindings(source,H.ROOT)
    manifest,expected,rgb_checks,cache_checks=verify_cache(check,private/'depth_accuracy_cache',expected_ids,selections,bindings,source,amendment,doc)
    receipt=H.read(doc/'DEPTH_INFERENCE_RECEIPT.json')
    assert receipt['frames'] == receipt['model_forwards'] == 2216 and receipt['private_cache'] and not receipt['GT_used_for_inference'] and receipt['training_updates'] == 0
    assert receipt['inference_lock_sha256'] == H.sha(private/'depth_accuracy_cache/DEPTH_INFERENCE_LOCK.json') and receipt['checkpoint'] == manifest['checkpoint']
    ids={p:{fid for population,fid in expected_ids if population==p} for p in ('REAL','SYNTH')}
    fronts,geometry=reference_depths(source,ids)
    derived={};measured={}
    for population,frames in [('SYNTH',1985),('REAL',231)]:
        derived[population],rows=verify_accuracy(check,population,doc,expected,fronts,frames)
        measured.update({(population,*key(row)):row for row in rows})
    primary=derived['REAL'][PRIMARY];passed=primary['complete']==231 and primary['median'] is not None and primary['median']<=.05
    assert gate['status'] == ('PASS' if passed else 'FAIL_STOP_S4') and gate['frames'] == 231 and gate['primary_method'] == PRIMARY
    check.close(gate['absolute_relative_error_median'],primary['median'],'fixed gate abs mean then median',absolute=1e-10)
    assert gate['threshold'] == .05 and gate['all_primary_seed_estimates_available'] == (primary['complete']==231) and gate['missing_ids'] == primary['missing']
    assert gate['S4_executed'] is False and gate['margin_tuning_executed'] is False and gate['final_test4_in_accuracy_or_m_selection'] is False and gate['model_forwards'] == 2216
    assert gate['aggregation'] == 'mean three seed absolute errors per ID, then median over231 IDs'
    assert execution['new_F_calls'] == execution['new_PnP_calls'] == execution['training_updates'] == execution['final_test4_tuning_frames'] == 0
    choice_count=pose_count=0;s4verdict=None
    if not passed:
        assert execution['status'] == 'COMPLETE_ACCURACY_GATE_FAIL_STOP_S4' and execution['accuracy_gate'] == 'FAIL_STOP_S4'
        assert execution['actual_depth_forwards'] == 2216 and execution['actual_model_constructions'] == 1
        assert execution['S4_REAL_evaluations'] == execution['margin_tuning_evaluations'] == 0 and execution['source_hashes_unchanged']
        assert not (private/'depth_final88_cache').exists()
        assert not (doc/'S4_MARGIN_LOCK.json').exists() and not (doc/'VERDICT_S4.json').exists()
        assert not list(doc.glob('STAGE3_SELECTIONS_*.jsonl.gz')) and not list(doc.glob('STAGE3_ROWS_*.jsonl.gz')) and not list(doc.glob('RESULTS_S4_*.json'))
    else:
        assert execution['status'] == 'COMPLETE' and execution['accuracy_gate'] == 'PASS' and execution['actual_depth_forwards'] == 2304 and execution['actual_model_constructions'] == 2 and execution['S4_REAL_evaluations'] == 1
        final_ids={('REAL',fid) for fid,records in selections['REAL'].items() if records[0]['session'] in FINAL4};assert len(final_ids)==88
        _,final_expected,nrgb,ncache=verify_cache(check,private/'depth_final88_cache',final_ids,selections,bindings,source,amendment,doc)
        final_rows=H.rows(doc/'DEPTH_MEASUREMENTS_FINAL88.jsonl.gz');assert len(final_rows)==88*12
        for row in final_rows:compare_measurement(check,row,final_expected[('REAL',*key(row))]);measured[('REAL',*key(row))]=row
        rgb_checks+=nrgb;cache_checks+=ncache
        choice_count,pose_count,s4verdict=verify_s4(check,doc,selections,measured,geometry,original_rows,execution)
    for binding in source_lock['bindings']:assert H.sha(H.ROOT/binding['path']) == binding['sha256']
    if amendment:
        recovery=H.read(doc/'STAGE3_RECOVERY_EXECUTION.json')
        assert recovery['actual_model_constructions_total']==execution['actual_model_constructions']+amendment['actual_prior_model_constructions'] and recovery['actual_depth_forwards_total']==execution['actual_depth_forwards']
    result=dict(status='PASS',phase='STAGE3_DEPTH_ACCURACY',depth_gate=gate['status'],
                independent_numeric_comparisons=check.comparisons,max_absolute_difference=check.max_absolute,
                max_scaled_relative_difference=check.max_relative,independent_depth_maps_checked=cache_checks,
                independent_RGB_sha_native_hw_K_crop_checks=rgb_checks,official_source_files_checked=official_count,
                official_checkpoint_sha256_checked=True,official_source_archive_sha256_checked=True,
                original_input_audit_SHA_bindings_checked=len(input_audit['bindings']),
                official_strict_false_runtime_amendment_verified=amendment is not None,
                prior_failed_attempt_and_diagnostic_forwards=0 if amendment else None,
                actual_model_constructions_total=execution['actual_model_constructions']+(amendment['actual_prior_model_constructions'] if amendment else 0),
                frozen_Stage1_selection_sha_bindings_checked=selection_bindings,
                independent_front_ROI_measurements_checked=len(measured),
                independently_verified_reference_front_depths=len(fronts),
                all_four_methods_all_three_seeds_verified=True,
                ABS_before_three_seed_mean_then_231_frame_median=True,
                reference_depth_positivity_verified=True,accuracy_statistics_ddof=1,
                depth_accuracy_CI95='NONE; descriptive moments/quantiles only',
                missing_primary_depth_ids=primary['missing'],depth_accuracy_gate_median=primary['median'],
                final_test4_threshold_model_selection_frames=0,final_test4_excluded_from_accuracy=True,
                S4_executed=passed,S4_independently_verified_choices=choice_count,S4_independent_T_R_yaw_ADD_count=pose_count,
                S4_verdict=s4verdict,no_S4_or_margin_selection_after_failed_gate_verified=not passed,
                additional_F_calls=0,additional_PnP_calls=0,additional_model_forwards=0,training_updates=0,
                IoU3D_geometry_independently_recomputed=False,
                source_sha256=H.sha(__file__),independent_stage1_helper_sha256=H.sha(H.__file__),
                independent_stage2_helper_sha256=H.sha(J.__file__))
    with output.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print('INDEPENDENT_DEPTH_PASS',check.comparisons,check.max_absolute,gate['status'],primary['median'],flush=True)


if __name__ == '__main__':main()
