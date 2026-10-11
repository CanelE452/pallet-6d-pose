"""Audit schemas/hash bindings and fix the rounded-display parity conflict."""
from collections import Counter
from pathlib import Path
import numpy as np
from . import common as C

EXPECTED = {
    'BASE': [12.188, 14.233, 41.2, 14.7],
    'N3_DIM_SYM': [10.833, 10.877, 46.9, 10.6],
    'SUBPIX': [10.897, 13.533, 55.1, 13.5],
    'N3_THEN_SUBPIX': [10.127, 11.391, 59.3, 11.2],
    'JOINT_FIXED_ISOTROPIC': [10.296, 11.120, 51.7, 10.9],
    'FG_JOINT_POSTERIOR': [10.784, 11.326, 49.8, 11.2]}

def run():
    C.method_lock()
    assert not (C.DOC / 'INPUT_AUDIT.json').exists()
    observed = list(C.rows(C.JOINT))
    assert len(observed) == 6 * 3 * 319
    required = {'pose', 'corner', 'qFinal', 'q0', 'qN', 'fixed_metadata', 'final_hypothesis', 'grade', 'session', 'seed', 'prediction_support', 'raw_hw'}
    assert all(required <= r.keys() for r in observed)
    assert all({'K', 'dimensions_pnp_WH_D_m', 'canonical_symmetry_order'} <= r['fixed_metadata'].keys() for r in observed)
    base = {r['id']: r for r in observed if r['method'] == 'BASE' and r['seed'] == 1}
    assert len(base) == 319 and Counter(r['grade'] for r in base.values()) == {'clean':153, 'moderate':92, 'severe':74}
    ids = sorted(i for i, r in base.items() if r['grade'] in ('clean', 'moderate'))
    assert len(ids) == 245 and len({base[i]['session'] for i in ids}) == 13
    ids_packet = dict(status='SEALED_BEFORE_ANY_CURRENT_F_CALL', ids=ids, frames=245,
        grades={'clean':153,'moderate':92}, sessions=sorted({base[i]['session'] for i in ids}),
        ids_sha256=C.digest(ids), selection='annotation clean/moderate chosen after DEV inspection; not independent test',
        method_lock_sha256=C.sha(C.DOC/'METHOD_LOCK.json'))
    C.write('RECT_245_IDS.json', ids_packet)
    comparisons=[]
    for method, expected in EXPECTED.items():
        selected=[r for r in observed if r['method']==method and r['id'] in set(ids)]
        assert len(selected)==245*3 and all(r['pose']['available'] for r in selected)
        metrics=[]
        for i in ids:
            rr=[r['pose'] for r in selected if r['id']==i]
            metrics.append([np.mean([r['translation_cm'] for r in rr]), np.mean([r['rotation_deg'] for r in rr]),
                np.mean([100.*(r['translation_cm']<5 and r['rotation_deg']<5) for r in rr]),
                np.mean([100.*(r['rotation_deg']>45 and abs(r['yaw_deg'])>=60) for r in rr])])
        actual=np.mean(metrics, axis=0)
        for field, a, e, decimals in zip(('translation_cm','rotation_deg','success_pct','confusion_pct'),actual,expected,(3,3,1,1)):
            displayed=float(f'{a:.{decimals}f}')
            comparisons.append(dict(method=method,metric=field,actual_full_precision=float(a),attachment_display=e,
                decimals=decimals,rounded_actual=displayed,rounded_delta=abs(displayed-e),original_delta=abs(float(a)-e),
                original_tolerance_pass=abs(float(a)-e)<=1e-6,PASS=abs(displayed-e)<=1e-6))
    variance=[]
    for method, expected in [('BASE',[538.6,851.4]),('N3_THEN_SUBPIX',[578.1,659.3])]:
        for field, e in zip(('translation_cm','rotation_deg'),expected):
            values=[np.mean([r['pose'][field] for r in observed if r['id']==i and r['method']==method]) for i in ids]
            actual=float(np.var(values,ddof=1)); displayed=float(f'{actual:.1f}')
            variance.append(dict(method=method,metric=field,actual_variance_ddof1=actual,attachment_display=e,
                rounded_actual=displayed,PASS=abs(displayed-e)<=1e-6))
    parity_status='PASS' if all(r['PASS'] for r in comparisons+variance) else 'FAIL_STOP'
    C.write('TABLE_DISPLAY_PARITY.json',dict(status=parity_status,user_authorized_resolution='Compare printed reference cells at their declared decimal precision; original strict actual-F tolerances remain unchanged.',
        source_sha256=C.sha(C.JOINT),absolute_tolerance_after_display_rounding=1e-6,comparisons=comparisons,variance=variance,
        original_raw_tolerance_pass=sum(r['original_tolerance_pass'] for r in comparisons), original_raw_tolerance_total=len(comparisons)))
    assert parity_status=='PASS', 'Rounded-display parity failed; stop'
    bindings=[C.binding(p) for p in (C.JOINT,C.SQUARE,C.REUSE,C.OLD_RAW,
        C.ROOT/'_docs/experiments/pallet_n3_completion_v3/TABLES.json',
        C.ROOT/'_docs/experiments/pallet_square6d_manualpnp_20261010/INPUT_AUDIT.json',
        C.ROOT/'_docs/experiments/pallet_symmetry_three_line_v1/OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json',
        C.SOURCE/'data/pallet/results/pallet_posefix_replay_diagnosis_v1/TARGETS.json',
        C.SOURCE/'data/pallet/results/paper_pose_metric_closure_v1/GEOMETRY_RESOLVED_POSE_GT.json')]
    reuse=C.read(C.REUSE); seed_counts={}; ablation={}
    for backbone in ('dope','resnet18'):
        path=C.SOURCE/f'data/pallet/results/pallet_n3_completion_v3/predictions/{backbone}_DEV319.json'
        packet=C.read(path); assert packet['complete'] and packet['dataset']=='DEV319'
        assert {r['id'] for r in packet['frames']}==set(base) and len(packet['frames'])==319
        present=[s for s in C.SEEDS if all(f'N3_seed{s}' in r['predictions'] for r in packet['frames'])]
        assert present==list(C.SEEDS)
        for r in packet['frames']:
            assert r['raw_shape_hw']==base[r['id']]['raw_hw']
            for key in ('base',*(f'N3_seed{s}' for s in present)):
                p=r['predictions'][key];assert len(p['valid'])==9 and (p['points'] is None or np.asarray(p['points']).shape==(9,2))
        seed_counts[backbone]=dict(seeds=present,frames_per_seed=319)
        bindings.append(C.binding(path))
    for family, arm in [('N0','N0_BASE_REPLAY'),('N1','N1_SYM_ONLY'),('N2','N2_DIM_ONLY'),('P','OLD_P')]:
        ablation[family]=[]
        for s in C.SEEDS:
            b=reuse['source_bindings']['core']['dcp_predictions'][f'{arm}_seed{s}']['prediction']
            p=C.SOURCE/b['path'];assert C.sha(p)==b['sha256']
            packet=C.read(p);assert packet['complete'] and packet['GT_input'] is False
            assert {r['id'] for r in packet['records']}==set(base) and len(packet['records'])==319
            bindings.append(C.binding(p));ablation[family].append(s)
    for s in C.SEEDS:
        for f in ('D','L','PoseFix'):
            b=reuse['source_bindings']['comparators'][f'H_{f}_seed{s}'];p=C.SOURCE/b['path']
            assert C.sha(p)==b['sha256'];bindings.append(C.binding(p))
            if f=='PoseFix':
                with np.load(p,allow_pickle=False) as z:
                    assert z['points'].shape==(319,2,4,9,2) and set(z['ids'].astype(str))==set(base)
                    assert str(z['checkpoint_sha256'].item())==b['checkpoint_sha256']
                    assert b['selected_tensor']=='points[:,0,1]'
        bindings.append(C.binding(C.SOURCE/f'data/pallet/results/pallet_posefix_replay_diagnosis_v1/predictions/seed{s}_REAL_DEV.npz'))
    rgb=[]
    manifest=C.original_manifest();assert len(manifest)==319 and {r['id'] for r in manifest}==set(base)
    for r in manifest:
        for name in ('image','annotation'):
            p=C.SOURCE/r[name]['path'];assert C.sha(p)==r[name]['sha256'];bindings.append(C.binding(p))
        assert C.digest(r['K'])==C.digest(base[r['id']]['fixed_metadata']['K'])
        rgb.append(dict(id=r['id'],image=r['image'],annotation=r['annotation'],raw_hw=r['raw_hw']))
    square_audit=C.read(C.ROOT/'_docs/experiments/pallet_square6d_manualpnp_20261010/INPUT_AUDIT.json')
    for r in square_audit['inputs']:
        for name in ('image','annotation'):
            p=C.SOURCE/r[name]['path'];assert C.sha(p)==r[name]['sha256'];bindings.append(C.binding(p))
    oldsources=[C.SOURCE/'scripts/research/pallet_dim_conditioned_p_v1/pose.py',
        C.SOURCE/'scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py',
        C.SOURCE/'challenge/evaluation_v2/pnp_selector.py',
        C.SOURCE/'scripts/research/pallet_training_free_compare_20261007_v1/methods.py']
    bindings.extend(C.binding(p) for p in oldsources)
    # De-duplicate only exact same path-owner pairs, retaining immutable hashes.
    bindings=list({(b['owner'],b['path']):b for b in bindings}.values())
    C.write('INPUT_AUDIT.json',dict(status='PASS',new_output_namespace_preexisting=False,bindings=bindings,
        rectangle_frames=319,paper_rectangle_frames=245,square_frames=118,seeds=C.SEEDS,native_backbone_seed_counts=seed_counts,
        ablation_seed_counts=ablation,all319_RGB_available=True,images=rgb,
        PoseFix_tensor='points[:,0,1]: existing first RAW chain/pass1; uncapped',
        manual_reference_inputs_in_correction=False,manual_reference_inputs_in_F=False,
        final_test4_threshold_or_model_selection=False,actual_F_calls=0,model_forwards=0,new_training_updates=0,
        id_seal_sha256=C.sha(C.DOC/'RECT_245_IDS.json'),table_parity_sha256=C.sha(C.DOC/'TABLE_DISPLAY_PARITY.json')))
    print('INPUT_SCHEMA_SHA_AND_ROUNDED_TABLE_PARITY_PASS',len(bindings),flush=True)

if __name__=='__main__':run()
