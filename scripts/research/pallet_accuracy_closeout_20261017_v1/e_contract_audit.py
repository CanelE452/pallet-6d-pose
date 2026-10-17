"""Read-only E0 observation-contract audit; no pose fitting or new ranking.

The CAL residuals below use the archived derived pose and human metadata only
for descriptive diagnostics. They are never exported as selector inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np

from scripts.research.pallet_visibility_curriculum_20261013_v1 import statewise_uncertainty_calibration_v4 as CAL


BASE = Path('_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def rows(path):
    with gzip.open(path, 'rt') as f:
        for line in f:
            yield json.loads(line)


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def csv_write(path, values):
    if not values:
        return
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(values[0]))
        w.writeheader()
        w.writerows(values)


def dist(values):
    a = np.asarray([x for x in values if x is not None], float)
    if not len(a):
        return dict(n=0, mean=None, sample_sd=None, median=None, p90=None, p95=None, maximum=None)
    return dict(n=len(a), mean=float(a.mean()), sample_sd=float(a.std(ddof=1)) if len(a) > 1 else None,
                median=float(np.median(a)), p90=float(np.quantile(a, .9)), p95=float(np.quantile(a, .95)), maximum=float(a.max()))


def stat_rows(values, group_keys, fields, cohort):
    grouped = defaultdict(list)
    for r in values:
        grouped[tuple(r[k] for k in group_keys)].append(r)
    result = []
    for key, group in sorted(grouped.items()):
        for field in fields:
            result.append(dict(cohort=cohort, **dict(zip(group_keys, key)), field=field,
                               **dist([r[field] for r in group])))
    return result


def plain_group(m):
    return m['state'] + (':fallback' if m['source'] == 'VISIBLE_BACKBONE_FALLBACK' else ':corrected')


def witness(repo, trace_path, output, n3_journal=None):
    """Explanatory archived-observation plot, never an owner correction."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    ident, k = 'wood_184309:000372', 3
    repo, output = Path(repo), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model_dir = repo / BASE / 'calibration/final_adapter_CAL46'
    protocol = json.loads((model_dir/'PROTOCOL.json').read_text())
    cal_row = next(r for r in rows(protocol['input_binding']['CAL_observations']) if r['id']==ident)
    labels = json.loads(Path(protocol['input_binding']['CAL_annotations']).read_text())
    label = labels['rows'][ident]
    point_rows, cases = CAL._cal_dataset([cal_row], {'rows':{ident:label}})
    original = next(p for p in point_rows if p['native_corner']==k)
    trace = next(r for r in rows(trace_path) if r['id']==ident)
    oracle = json.loads((repo/'_docs/experiments/pallet_visibility_curriculum_20261013_v1/ORACLE_CAL_INPUTS.json').read_text())
    oracle_frame = next(f for f in oracle['frames'] if f['id']==ident)
    geometry = cal_row['runtime']['initial_geometry']
    assert oracle_frame['scoring_metadata'] == label['metadata']
    assert oracle_frame['target'] == label['target']
    assert oracle_frame['points']['N3_SUBPIX'] == cal_row['runtime']['original_raw_points']
    assert oracle_frame['initial_poses']['N3_SUBPIX']['R_cf'] == geometry['R_cf']
    assert oracle_frame['initial_poses']['N3_SUBPIX']['centroid'] == geometry['centroid']
    assert oracle_frame['initial_poses']['N3_SUBPIX']['R_physical'] == geometry['R_physical']
    alignment = CAL._reference_alignment(cal_row,label)
    xyz,Q = np.asarray(geometry['native_xyz']),np.asarray(geometry['physical_registration_Q'])
    truth=label['target']['truth']
    Qref=np.asarray(truth['body_R']).T @ np.asarray(truth['R'])
    mapped=CAL._cuboid(truth['body_xyz'])[label['metadata']['native_to_reference_corner_ids']] @ Qref
    geometry_phases=[]
    for phase in range(truth['order']):
        angle=phase*2*np.pi/truth['order']
        S=np.asarray([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
        ref_pose=np.asarray(truth['R']) @ S @ Q.T
        projection=CAL._project(xyz,ref_pose,truth['t'],cal_row['K'])
        geometry_phases.append(dict(phase=phase,per_index_3D_max_gap_m=float(np.linalg.norm(xyz@Q@S.T-mapped,axis=1).max()),
            reference_projection_k3_xy=projection[k].tolist(),
            reference_projection_k3_vs_original_N3_xy_px=float(np.linalg.norm(projection[k]-original['xy']))))
    n3_binding=None
    if n3_journal is not None:
        saved=next(r for r in rows(n3_journal) if r['id']==ident and r['method']=='N3_SUBPIX')
        assert saved['native_points']==oracle_frame['points']['N3_SUBPIX']
        assert saved['corner']['branch']==label['metadata']['frozen_baseline_branch']
        assert saved['actual_pose']['R_cf']==geometry['R_cf']
        assert saved['actual_pose']['R_physical']==geometry['R_physical']
        assert saved['actual_pose']['centroid']==geometry['centroid']
        assert saved['final_hypothesis']=='short-face-front'
        assert label['metadata']['native_to_reference_corner_ids']==label['target']['permutations'][saved['corner']['branch']][:8]
        n3_binding=dict(path=str(n3_journal),sha256=sha(n3_journal),method=saved['method'],
            frozen_corner_branch=saved['corner']['branch'],final_hypothesis=saved['final_hypothesis'],
            points_exact=True,R_cf_exact=True,R_physical_exact=True,centroid_exact=True,
            mapping_exact_target_permutation_of_original_N3_branch=True,
            RAW_vs_N3_branch_mixing_found=False)
    m = next(m for m in trace['measurements'] if m['native_corner']==k)
    old_xy, new_xy = np.asarray(original['xy']), np.asarray(m['xy'])
    mapped_id = label['metadata']['native_to_reference_corner_ids'][k]
    annotation = np.asarray(label['target']['gt'])
    mapped_xy = annotation[mapped_id]
    reference_xy = old_xy - np.asarray(original['reference_error_xy'])
    ref_errors = np.linalg.norm(annotation[:8]-new_xy, axis=1)
    nearest_id = int(ref_errors.argmin())
    model = json.loads((model_dir/'MODEL.json').read_text())
    applied = CAL.apply_calibrated_measurements([m], model, nonself_scale_px=original['nonself_scale_px'])[0]
    image_path = repo / cal_row['image']
    assert sha(image_path)==cal_row['image_sha256']
    arr=np.asarray(Image.open(image_path).convert('RGB'))
    fig,axes=plt.subplots(1,2,figsize=(18,5.7))
    for ax in axes:
        ax.imshow(arr)
        ax.set_xlim(0,arr.shape[1]);ax.set_ylim(arr.shape[0],0);ax.axis('off')
    axes[0].set_title('Original RGB, unchanged')
    ax=axes[1]
    for i,q in enumerate(annotation[:8]):
        ax.scatter(*q,s=50,facecolors='none',edgecolors='white',linewidths=1)
        ax.text(q[0]+9,q[1]-7,f'ref{i}',color='white',fontsize=9,bbox=dict(facecolor='black',alpha=.6,pad=1))
    ax.scatter(*old_xy,s=100,marker='x',c='#1f77ff',linewidths=2,label='Archived N3+SubPix k3')
    ax.scatter(*new_xy,s=110,marker='+',c='#ffad00',linewidths=2,label='Archived RGB recovery k3')
    ax.plot([new_xy[0],mapped_xy[0]],[new_xy[1],mapped_xy[1]],c='#f43c38',linestyle='--',linewidth=1.5)
    ax.scatter(*mapped_xy,s=220,facecolors='none',edgecolors='#f43c38',linewidths=2,label=f'Fixed mapped reference {mapped_id}')
    ax.scatter(*annotation[nearest_id],s=140,facecolors='none',edgecolors='#46ff70',linewidths=1.8,label=f'Nearby reference {nearest_id}, diagnostic only')
    ax.legend(loc='upper right',fontsize=8,framealpha=.9)
    ax.set_title('Archived channel-to-reference mapping; physical owner remains unverified')
    fig.suptitle(f'{ident}: fixed k3 -> ref{mapped_id} residual {np.linalg.norm(new_xy-mapped_xy):.3f}px; nearby ref{nearest_id} {ref_errors[nearest_id]:.3f}px',fontsize=13)
    fig.text(.5,.015,'Existing derived reference and annotations are not independent physical ground truth. Nearest ref is shown for diagnosis; no mapping is changed.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.045,1,.94]);fig.savefig(output/'CAL_OWNER_PHASE_WITNESS.png',dpi=140);plt.close(fig)
    result=dict(id=ident,native_corner=k,image=cal_row['image'],image_sha256=sha(image_path),image_hw=list(arr.shape[:2]),
        original_xy=old_xy.tolist(),actual_recovered_xy=new_xy.tolist(),actual_measurement_source=m['source'],
        fixed_native_to_reference_corner_ids=label['metadata']['native_to_reference_corner_ids'],fixed_mapped_reference_id=mapped_id,
        fixed_mapped_annotation_xy=mapped_xy.tolist(),derived_reference_projection_xy=reference_xy.tolist(),
        fixed_mapping_annotation_residual_px=float(np.linalg.norm(new_xy-mapped_xy)),
        fixed_mapping_derived_reference_residual_px=float(np.linalg.norm(new_xy-reference_xy)),
        nearby_reference_id_diagnostic_only=nearest_id,nearby_annotation_xy=annotation[nearest_id].tolist(),
        nearby_annotation_distance_px_diagnostic_only=float(ref_errors[nearest_id]),
        actual_fit_covariance_px2=m['fit_covariance_px2'], hypothetical_current_adapter_covariance_px2=applied['fit_covariance_px2'],
        actual_max_std_px=float(np.sqrt(np.linalg.eigvalsh(m['fit_covariance_px2']).max())),
        hypothetical_max_std_px=float(np.sqrt(np.linalg.eigvalsh(applied['fit_covariance_px2']).max())),
        actual_reliability=m['fit_reliability_weight'],hypothetical_reliability=applied['fit_reliability_weight'],
        original_geometry_alignment_phase=cases[0]['phase'],original_geometry_alignment_gap_m=cases[0]['geometry_gap_m'],
        independent_per_index_geometry_phase_checks=geometry_phases,
        original_N3_archive_binding=n3_binding,
        oracle_to_isolated_metadata_and_target_exact=True,
        initial_N3_pose_and_points_to_fresh_CAL_input_exact=True,
        original_initial_projection_k3_xy=CAL._project(xyz,geometry['R_cf'],geometry['centroid'],cal_row['K'])[k].tolist(),
        original_initial_cf_extents=geometry['cf_extents'],original_physical_registration_Q=Q.tolist(),
        metadata=label['metadata'],human_state_for_original_channel=original['human_state'],
        independent_physical_GT=False,raw_image_actually_viewed_by_root_subagent=True,
        annotation_or_reference_or_owner_mapping_modified=False,
        interpretation='Large archived mapped residual does not establish a large physical localization error. The RGB recovery is near a different annotated corner, so owner/phase uncertainty prevents treating the new recovery source as authenticated precise physical evidence.')
    write(output/'CAL_OWNER_PHASE_WITNESS.json',result)
    return result


def audit(repo, typed_path, trace_path, output):
    wall, cpu = time.monotonic(), time.process_time()
    repo, output = Path(repo), Path(output)
    if output.exists():
        raise FileExistsError('Preserve previous audit: ' + str(output))
    output.mkdir(parents=True)
    packets = list(rows(typed_path))
    assert len(packets) == 238 and len({p['id'] for p in packets}) == 238
    assert Counter(p['partition'] for p in packets) == {'CAL': 46, 'TEST': 49, 'REMAINDER': 143}
    by_id = {p['id']: p for p in packets}
    model_path = repo / BASE / 'calibration/final_adapter_CAL46/MODEL.json'
    protocol_path = model_path.parent / 'PROTOCOL.json'
    protocol, model = json.loads(protocol_path.read_text()), json.loads(model_path.read_text())
    binding = protocol['input_binding']
    cal_path = Path(binding['CAL_observations'])
    labels_path = Path(binding['CAL_annotations'])
    assert sha(cal_path) == binding['CAL_observations_sha256']
    assert sha(labels_path) == binding['CAL_annotations_sha256']
    cal_rows = list(rows(cal_path))
    labels = json.loads(labels_path.read_text())
    cal_points, cal_cases = CAL._cal_dataset(cal_rows, labels)
    assert len(cal_cases) == 46
    cal_case_map = {r['id']: r for r in cal_cases}
    cal_point_map = {(r['id'], r['native_corner']): r for r in cal_points}
    assert set(model['fit_ids']) == {p['id'] for p in packets if p['partition'] == 'CAL'}
    archive_path = repo / packets[0]['provenance']['archive_path']
    assert sha(archive_path) == packets[0]['provenance']['archive_sha256']
    archive_original = {}
    for r in rows(archive_path):
        assert r['id'] in by_id and r['id'] not in archive_original
        s = r['finish_input']['_original_source']
        archive_original[r['id']] = dict(visible=s['calibrated_visible_measurements'],
            scale=s['ot_domains']['nonself_prediction_scale_px'], geometry=s['initial_geometry'],
            original_correction=s['visible_correction'])
    assert set(archive_original) == set(by_id)
    checks = Counter()
    max_errors = defaultdict(float)
    observations, frames, partials = [], [], []
    for p in packets:
        s, ident = p['solver_input'], p['id']
        orig = archive_original[ident]
        xyz, Q = np.asarray(s['xyz']), np.asarray(p['registered_family']['physical_registration_Q'])
        assert np.allclose(Q.T @ Q, np.eye(3), atol=1e-9, rtol=0) and np.linalg.det(Q) > 0
        assert np.array_equal(xyz, np.asarray(p['registered_family']['native_xyz']))
        assert np.array_equal(Q, np.asarray(s['coarse_pose']['physical_registration_Q']))
        old_xyz = np.asarray(orig['geometry']['native_xyz'])
        old_Q = np.asarray(orig['geometry']['physical_registration_Q']) if 'physical_registration_Q' in orig['geometry'] else np.asarray(orig['geometry']['R_cf']).T @ np.asarray(orig['geometry']['R_physical'])
        d = np.linalg.norm((xyz @ Q)[:, None] - (old_xyz @ old_Q)[None, :], axis=2)
        perm = d.argmin(1)
        assert len(set(map(int, perm))) == 8 and d[np.arange(8), perm].max() < 1e-8
        max_errors['registered_unordered_physical_vertex_gap_m'] = max(max_errors['registered_unordered_physical_vertex_gap_m'], float(d[np.arange(8), perm].max()))
        own_old = {m['native_corner']: m for m in orig['visible']}
        active, partial = set(p['activation']['complete_ids']), set(p['activation']['all_partial_ids'])
        assert not active & partial
        assert active | partial == {m['native_corner'] for m in s['visible']}
        assert partial == {b['native_corner'] for b in s['boundary_observations']}
        assert all(s['states'][k] in ('B', 'I') for k in active | partial)
        current_expected = CAL.apply_calibrated_measurements(s['visible'], model, states=s['states'], nonself_scale_px=orig['scale'])
        exp_map = {m['native_corner']: m for m in current_expected}
        orig_expected = CAL.apply_calibrated_measurements(orig['visible'], model, nonself_scale_px=orig['scale'])
        orig_exp_map = {m['native_corner']: m for m in orig_expected}
        frame_changed = 0
        for m in s['visible']:
            k = m['native_corner']
            old, expected = own_old[k], exp_map[k]
            cov, fit, exp_cov = map(np.asarray, (m['covariance_px2'], m['fit_covariance_px2'], expected['fit_covariance_px2']))
            L = np.linalg.inv(np.linalg.cholesky(fit))
            inv_err = float(np.max(np.abs(L.T @ L - np.linalg.inv(fit))))
            assert inv_err < 1e-12 and np.linalg.eigvalsh(cov).min() > 0 and np.linalg.eigvalsh(fit).min() > 0
            assert m.get('measurement_is_pose_projection') is False
            assert s['states'][k] == m['state'] and orig['geometry']['initial_self_hidden'][k] is False
            old_adapter_gap = float(np.max(np.abs(np.asarray(old['fit_covariance_px2']) - orig_exp_map[k]['fit_covariance_px2'])))
            assert old_adapter_gap < 1e-9
            copied_gap = float(np.max(np.abs(fit - np.asarray(old['fit_covariance_px2']))))
            assert copied_gap == 0 and m['fit_reliability_weight'] == old['fit_reliability_weight']
            gap = float(np.max(np.abs(fit - exp_cov)))
            changed = gap > 1e-9 or m['fit_reliability_weight'] != expected['fit_reliability_weight']
            frame_changed += changed
            fit_floor = fit - cov
            cal_p = cal_point_map.get((ident, k))
            ref_error, ann_error = None, None
            hstate, compatible = '', None
            if cal_p is not None:
                hstate, compatible = cal_p['human_state'], cal_p['compatible']
                if cal_p['reference_error_xy'] is not None:
                    # Same original semantic channel, before optional W/D hypothesis.
                    ref = np.asarray(cal_p['xy']) - np.asarray(cal_p['reference_error_xy'])
                    ref_error = float(np.linalg.norm(np.asarray(m['xy']) - ref))
                if cal_p['annotation_error_xy'] is not None:
                    ann = np.asarray(cal_p['xy']) - np.asarray(cal_p['annotation_error_xy'])
                    ann_error = float(np.linalg.norm(np.asarray(m['xy']) - ann))
            row = dict(id=ident, partition=p['partition'], session=p['session'], native_corner=k,
                state=m['state'], role='FULL' if k in active else 'PARTIAL_ARCHIVE',
                family=p['registered_family']['name'], physical_owner_original_native_id=int(perm[k]),
                same_channel_owner_changed=int(perm[k]) != k, owner_semantic_status=p['source_group_semantic_owner_status'],
                original_source=old['source'], actual_source=m['source'], original_cal_group=plain_group(old), current_cal_group=plain_group(m),
                source_changed=old['source'] != m['source'], xy_changed=old['xy'] != m['xy'], conditional_covariance_changed=old['covariance_px2'] != m['covariance_px2'],
                fit_override_exactly_copied=True, actual_vs_current_adapter_changed=changed,
                observed_std_max_px=float(np.sqrt(np.linalg.eigvalsh(cov).max())), fit_std_max_px=float(np.sqrt(np.linalg.eigvalsh(fit).max())),
                reapplied_fit_std_max_px=float(np.sqrt(np.linalg.eigvalsh(exp_cov).max())),
                fit_minus_current_observed_min_eigenvalue_px2=float(np.linalg.eigvalsh(fit_floor).min()),
                current_adapter_covariance_max_abs_gap_px2=gap, actual_reliability=m['fit_reliability_weight'],
                current_adapter_reliability=expected['fit_reliability_weight'], whitening_inverse_max_abs_error=inv_err,
                cal_human_state=hstate, cal_geometry_compatible=compatible,
                cal_original_semantic_channel_reference_residual_px=ref_error,
                cal_direct_annotation_residual_px=ann_error)
            observations.append(row)
            checks['measurements_checked'] += 1
            max_errors['whitening_identity_max_abs'] = max(max_errors['whitening_identity_max_abs'], inv_err)
            max_errors['original_adapter_covariance_max_abs'] = max(max_errors['original_adapter_covariance_max_abs'], old_adapter_gap)
        by_m = {m['native_corner']:m for m in s['visible']}
        for b in s['boundary_observations']:
            k = b['native_corner']
            original = by_m[k]
            a = b.get('association_provenance', {})
            if b.get('explicit_complete_BI_likelihood_revocation'):
                for field, expected in [('original_observed_xy', original['xy']), ('original_observed_covariance_px2', original['covariance_px2']), ('original_effective_fit_covariance_px2', original['fit_covariance_px2'])]:
                    assert np.array_equal(np.asarray(a[field]), np.asarray(expected))
                assert a['original_measurement_source'] == original['source']
            assert b['fit_reliability_weight'] == original['fit_reliability_weight']
            for line in b['lines']:
                n = np.asarray(line['normal_unit_xy'])
                assert abs(np.linalg.norm(n) - 1) < 1e-7 and line['normal_variance_px2'] > 0
                interval = line['support_tangent_interval_px']
                assert len(interval) == 2 and interval[1] >= interval[0]
            partials.append(dict(id=ident, partition=p['partition'], session=p['session'], native_corner=k,
                state=b['state'], typed=bool(b.get('explicit_complete_BI_likelihood_revocation')), line_count=len(b['lines']),
                source=original['source'], normal_variance_min_px2=min(l['normal_variance_px2'] for l in b['lines']),
                normal_variance_max_px2=max(l['normal_variance_px2'] for l in b['lines']),
                reliability=b['fit_reliability_weight'], complete_cost_active=False,
                grouped_radial_loss_not_independent_line_sum=True,
                actual_rgb_line_ids=json.dumps(a.get('actual_RGB_line_ids', [])),
                revocation_reason=a.get('revocation_reason', 'LEGACY_FALLBACK_B')))
        frames.append(dict(id=ident, partition=p['partition'], session=p['session'], family=p['registered_family']['name'],
            S_count=s['states'].count('S'), B_count=s['states'].count('B'), I_count=s['states'].count('I'),
            O_count=s['states'].count('O'), T_count=s['states'].count('T'), U_count=s['states'].count('U'),
            full_count=len(active), partial_count=len(partial), typed_partial_count=len(p['activation']['typed_partial_ids']),
            outline_supported=bool(s['outline_observation'].get('supported')), reference_is_independent_physical_GT=False,
            current_adapter_mismatch_points=frame_changed, same_RGB_point_partial_outline_independence_certified=False))
        checks['frames_checked'] += 1
    cal_original_table = []
    for p in cal_points:
        m = dict(native_corner=p['native_corner'], state=p['state'], xy=p['xy'], covariance_px2=p['covariance_px2'], source=p['source'])
        applied = CAL.apply_calibrated_measurements([m], model, nonself_scale_px=p['nonself_scale_px'])[0]
        cal_original_table.append(dict(id=p['id'], session=p['session'], native_corner=p['native_corner'], state=p['state'], group=p['group'], source=p['source'],
            compatible=p['compatible'], human_state=p['human_state'], human_DIRECT=p['human_DIRECT'],
            observed_std_max_px=float(np.sqrt(np.linalg.eigvalsh(p['covariance_px2']).max())),
            effective_std_max_px=float(np.sqrt(np.linalg.eigvalsh(applied['fit_covariance_px2']).max())),
            reliability=applied['fit_reliability_weight'],
            reference_residual_px=float(np.linalg.norm(p['reference_error_xy'])) if p['reference_error_xy'] is not None else None,
            direct_annotation_residual_px=float(np.linalg.norm(p['annotation_error_xy'])) if p['annotation_error_xy'] is not None else None))
    source_counts = []
    for cohort, selected in [('CAL46', [r for r in observations if r['partition']=='CAL']), ('PRIMARY192', [r for r in observations if r['partition']!='CAL'])]:
        cnt = Counter((r['session'], r['state'], r['role'], r['original_source'], r['actual_source']) for r in selected)
        source_counts.extend(dict(cohort=cohort, session=k[0], state=k[1], role=k[2], original_source=k[3], actual_source=k[4], point_count=v) for k,v in sorted(cnt.items()))
    summaries = []
    for cohort, selected in [('CAL46', [r for r in observations if r['partition']=='CAL']), ('PRIMARY192', [r for r in observations if r['partition']!='CAL'])]:
        summaries += stat_rows(selected, ['session','state','role','actual_source'],
            ['observed_std_max_px','fit_std_max_px','reapplied_fit_std_max_px','actual_reliability','current_adapter_reliability','current_adapter_covariance_max_abs_gap_px2'], cohort)
    cal_distributions = stat_rows(cal_original_table, ['session','group','source','human_state'],
        ['observed_std_max_px','effective_std_max_px','reliability','reference_residual_px','direct_annotation_residual_px'], 'CAL46_ORIGINAL_FIT_INPUT')
    csv_write(output/'E_POINT_CONTRACT_ROWS.csv', observations)
    csv_write(output/'E_FRAME_CONTRACT_ROWS.csv', frames)
    csv_write(output/'E_PARTIAL_CONTRACT_ROWS.csv', partials)
    csv_write(output/'CAL_PROVENANCE_POINTS.csv', cal_original_table)
    csv_write(output/'CAL_PROVENANCE_SOURCE_COUNTS.csv', source_counts)
    csv_write(output/'CAL_PROVENANCE_DISTRIBUTIONS.csv', cal_distributions)
    csv_write(output/'E_SCALE_DISTRIBUTIONS.csv', summaries)
    csv_write(output/'CAL_CASE_COMPATIBILITY.csv', [{k:r[k] for k in ('id','session','compatible','geometry_gap_m','phase','nonself_scale_px','independent_physical_pose_truth','fit_eligible','exclusion_reason')} for r in cal_cases])
    summary = dict(schema='E0_INFORMATION_CONTRACT_AUDIT_V1', no_pose_fit=True, no_new_candidate_scores_or_selection=True,
        no_calibration_fit=True, original_model_sha256=sha(model_path), original_model_schema=model['schema'], original_model_groups=model['groups'],
        original_fit_session_counts=dict(Counter(r['session'] for r in cal_cases)), original_fit_compatible_frames=sum(r['compatible'] for r in cal_cases),
        original_fit_point_count=len(cal_points), original_fit_compatible_points=sum(r['compatible'] for r in cal_points),
        original_fit_state_counts=dict(Counter(r['state'] for r in cal_points)), original_fit_source_counts=dict(Counter(r['source'] for r in cal_points)),
        original_fit_group_counts=dict(Counter(r['group'] for r in cal_points)), original_fit_human_counts=dict(Counter(r['human_state'] for r in cal_points)),
        original_fit_input_binding={k:binding[k] for k in ('CAL_observations_sha256','CAL_annotations_sha256')},
        checks=dict(checks), maximum_numeric_errors=dict(max_errors),
        by_cohort={cohort:dict(frames=sum(r['partition']==part for r in frames) if part=='CAL' else sum(r['partition']!='CAL' for r in frames),
            points=len(selected), full_points=sum(r['role']=='FULL' for r in selected), partial_archived_points=sum(r['role']=='PARTIAL_ARCHIVE' for r in selected),
            source_changed_points=sum(r['source_changed'] for r in selected), covariance_changed_points=sum(r['conditional_covariance_changed'] for r in selected),
            adapter_mismatch_points=sum(r['actual_vs_current_adapter_changed'] for r in selected), adapter_mismatch_full_points=sum(r['actual_vs_current_adapter_changed'] and r['role']=='FULL' for r in selected),
            mismatch_frame_count=len({r['id'] for r in selected if r['actual_vs_current_adapter_changed']}),
            fit_minus_current_observed_negative_eigenvalue_points=sum(r['fit_minus_current_observed_min_eigenvalue_px2'] < -1e-9 for r in selected),
            same_channel_owner_changed_points=sum(r['same_channel_owner_changed'] for r in selected))
            for cohort,part,selected in [('CAL46','CAL',[r for r in observations if r['partition']=='CAL']),('PRIMARY192','PRIMARY',[r for r in observations if r['partition']!='CAL'])]},
        candidate_semantic_owner_status='UNVERIFIED: geometric bijection is not image-semantic authentication',
        source_scale_status='OBSERVATION_MODEL_MISMATCH: intentional frozen-old-noise design, not proven accidental implementation error',
        limitation='CAL references are existing geometry-derived poses and reused human labels; not independent physical metrology. Same RGB evidence can be correlated.',
        wall_seconds=time.monotonic()-wall, cpu_seconds=time.process_time()-cpu, peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(output/'CAL_PROVENANCE_SUMMARY.json', summary)
    used = [typed_path, trace_path, model_path, protocol_path, cal_path, labels_path, archive_path]
    write(output/'AUDIT_INPUT_SHA256.json', {str(Path(p)):sha(p) for p in used})
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--typed',type=Path,required=True)
    p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--witness-only',action='store_true')
    p.add_argument('--n3-journal',type=Path)
    a=p.parse_args()
    print(json.dumps(witness(a.repo,a.trace,a.output,a.n3_journal) if a.witness_only else audit(a.repo,a.typed,a.trace,a.output),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
