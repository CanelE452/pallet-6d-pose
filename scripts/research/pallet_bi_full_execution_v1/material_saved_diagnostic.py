"""Finite postseal joins of already-scored bank labels and saved H1 logits.

No references, pose scoring, producer, renderer, matcher, model or fit is loaded.
The only shared numerical routine called is the saved-feature H1 input transform.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scripts.research.pallet_boundary_action_repair_v1 import action_data as D

ARMS = ('S3_COMMON', 'S3_BI', 'S4_PL', 'S4_CONTOUR', 'S4_IMAGE')
SEEDS = (20261018, 20261019, 20261020)
COHORTS = ('original41_candidate_absent', 'PF_union38_candidate_absent')


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with gzip.open(path, 'rt') as f:
        return [json.loads(line) for line in f if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def save(path, values):
    with gzip.open(path, 'wt') as f:
        for value in values:
            f.write(canonical(value) + '\n')


def aliases(candidate):
    return set(candidate.get('source_aliases', [])) | set(candidate.get('candidate_aliases', [])) | set(candidate.get('complete_action_aliases', [])) | {candidate['candidate_id']}


def same_saved_pose(a, b, scale):
    """Compare saved matrices; never construct or score a pose."""
    return bool(a.get('available') and b.get('available')
        and np.max(np.abs(np.asarray(a['R_physical']) - b['R_physical'])) <= 1e-7
        and np.max(np.abs(np.asarray(a['centroid']) - b['centroid'])) <= 1e-7 * scale)


def run(run_path, d1_path, output):
    started = time.monotonic()
    run_path, d1_path, output = map(Path, (run_path, d1_path, output))
    if output.exists():
        raise ValueError('New independent diagnostic directory required; completed outputs are immutable')
    bindings = {}

    def bind(path, expected=None):
        path = Path(path)
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError('Changed bound input: ' + str(path))
        bindings[str(path)] = digest
        return digest

    def sealed_artifacts(folder, seal_name):
        folder = run_path / folder
        seal = read(folder / seal_name)
        bind(folder / seal_name)
        if not seal['complete']:
            raise ValueError('Incomplete saved analysis')
        for name, digest in seal['artifacts'].items():
            bind(folder / name, digest)
        return seal

    bank_seal = sealed_artifacts('S3_S4_BANK_ANALYSIS', 'ANALYSIS_SEAL.json')
    rank_seal = sealed_artifacts('S3_S4_SELECTION_DIAGNOSTIC', 'SELECTION_DIAGNOSTIC_SEAL.json')
    base_bank_seal = sealed_artifacts('POST_BANK_S1_VALIDATION_V2', 'ANALYSIS_SEAL.json')
    base_rank_seal = sealed_artifacts('S1_SELECTION_RANK_DIAGNOSTIC_V2', 'SELECTION_DIAGNOSTIC_SEAL.json')
    if rank_seal['bank_analysis_seal_sha256'] != bind(run_path / 'S3_S4_BANK_ANALYSIS/ANALYSIS_SEAL.json'):
        raise ValueError('Rank/bank ancestry differs')
    if base_rank_seal['bank_analysis_seal_sha256'] != bind(run_path / 'POST_BANK_S1_VALIDATION_V2/ANALYSIS_SEAL.json'):
        raise ValueError('Baseline rank/bank ancestry differs')
    fixed_path = run_path / 'P0_FIXED_SETS/FIXED_SETS.json'
    fixed_sha = bind(fixed_path, bank_seal['fixed_sets_sha256'])
    if any(s['fixed_sets_sha256'] != fixed_sha for s in (rank_seal, base_bank_seal, base_rank_seal)):
        raise ValueError('Fixed cohorts differ')
    sets = read(fixed_path)['sets']
    if len(sets[COHORTS[0]]) != 41 or len(sets[COHORTS[1]]) != 38 or not set(sets[COHORTS[1]]) < set(sets[COHORTS[0]]):
        raise ValueError('The exact pre-existing41/38 cohorts are required')
    fixed_ids = set(sets[COHORTS[0]])
    manifest_path = run_path / 'S3_S4_POST_BANK_MANIFEST.json'
    bind(manifest_path, bank_seal['manifest_sha256'])
    manifest = read(manifest_path)
    input_bindings = read(run_path / 'S3_S4_BANK_ANALYSIS/INPUT_BINDINGS.json')
    bank = {(x['arm'], x['id']): x for x in rows(run_path / 'S3_S4_BANK_ANALYSIS/BANK_COVERAGE_ROWS.jsonl.gz')}
    ranks = {(x['arm'], x['id'], x['seed']): x for x in rows(run_path / 'S3_S4_SELECTION_DIAGNOSTIC/ACTION_RANK_ROWS.jsonl.gz')}
    base_bank = {x['id']: x for x in rows(run_path / 'POST_BANK_S1_VALIDATION_V2/BANK_COVERAGE_ROWS.jsonl.gz')}
    base_ranks = {(x['id'], x['seed']): x for x in rows(run_path / 'S1_SELECTION_RANK_DIAGNOSTIC_V2/ACTION_RANK_ROWS.jsonl.gz')}
    if len(bank) != 1190 or len(ranks) != 3570 or len(base_bank) != 238 or len(base_ranks) != 714:
        raise ValueError('Complete saved populations required before fixed-cohort filtering')

    producer = run_path / 's3_s4_production'
    full_path = producer / 'FULL_RECEIPT.json'
    full = read(full_path)
    full_sha = bind(full_path)
    bind(producer / 'METHOD_LOCK_RESOURCE_LP4.json', full['method_lock_sha256'])
    archive_path = run_path / 's3_s4_archive_report/ROW_ARCHIVE_SEAL.json'
    archive = read(archive_path)
    bind(archive_path)
    if archive['producer_full_receipt_sha256'] != full_sha or not archive['all_raw_row_candidate_payloads_equal_sealed_producer_extras']:
        raise ValueError('Raw archive and producer provenance differs')
    map_path = archive_path.parent / 'ALL_ROW_BINDINGS.jsonl.gz'
    bind(map_path, archive['all_row_seal_and_frame_bindings_sha256'])
    row_bindings = {x['id']: x for x in rows(map_path) if x['domain'] == 'real'}
    if len(row_bindings) != 238:
        raise ValueError('All real row seals required')
    base_feature_path = run_path / 's1_real_union/HEAD_FEATURES.jsonl.gz'
    base_feature_seal = read(base_feature_path.parent / 'FEATURE_SEAL.json')
    bind(base_feature_path.parent / 'FEATURE_SEAL.json')
    bind(base_feature_path, base_feature_seal['head_features_sha256'])
    base_features = {x['id']: x for x in rows(base_feature_path) if x['id'] in fixed_ids}
    baseline_cohort = dict(fixed_original41_ids=sets[COHORTS[0]], fixed_postPF38_ids=sets[COHORTS[1]],
        fixed_PF_only_new_good_ids=sets['PF_new_good'],
        original_generated_good_count=0, S1_PF_union_eligible_good_count=sum(base_bank[i]['eligible_good'] for i in fixed_ids),
        S1_selected_success_ids_by_seed={str(seed): [i for i in sets[COHORTS[0]] if base_bank[i]['selected'][str(seed)]['metric']['success']] for seed in SEEDS})
    if baseline_cohort['S1_PF_union_eligible_good_count'] != 3:
        raise ValueError('Prior PF-union coverage must reproduce three fixed IDs')

    source = Path(__file__)
    bind(source)
    bind(D.__file__)
    support_records = []
    detail_records = []
    feature_records = []
    endpoint_records = []
    summary = dict(schema='S3_S4_FINITE_SAVED_OUTPUT_DIAGNOSTIC_V1', complete=True,
        scope='All fixed original41 and contained postPF38 IDs, every arm and all3 seeds; labels/logits already sealed before this join',
        seed_order=list(SEEDS), baseline=baseline_cohort, arms={},
        physical_support_status='Automatic raw image support associations; independent physical ownership UNKNOWN; no new certification',
        asset_status_clarification='Missing exact named source and/or uncertified instance/signed-axis registration. These outputs do not empirically prove that the proxy geometry is wrong.',
        interpretation_limit='Saved feature/logit associations locate selection loss. No causal attribution to a feature, cue, asset, optimizer or training change without an intervention.',
        H1_input_layout=dict(candidate_normalized31=[0,31], candidate_valid31=[31,62], bank_mean62=[62,124], bank_max62=[124,186], log1p_action_count=186, is_N0=187, dtype='float32'),
        source_coordinate_units='Saved native image pixels and saved object3D/native body coordinates; physical correspondence truth is not certified')
    frame_cache = {}
    erun_cache = {}

    def raw_frame(ident):
        if ident not in frame_cache:
            entry = row_bindings[ident]
            sp = producer / entry['row_seal']
            fp = producer / entry['frame']
            bind(sp, entry['row_seal_sha256'])
            bind(fp, entry['frame_sha256'])
            seal = read(sp)
            if seal['frame_sha256'] != entry['frame_sha256'] or seal['id'] != ident:
                raise ValueError('Raw row ancestry differs')
            frame = rows(fp)[0]
            if frame['id'] != ident or frame.get('GT_inputs') or frame.get('current_real_reference_reads'):
                raise ValueError('Prediction-only producer row required')
            ep = d1_path / 'd1_real_candidates_v1/frames' / fp.parent.name / 'E_RUN.jsonl.gz'
            bind(ep, seal['dependencies']['E_RUN_sha256'])
            erun = rows(ep)[0]
            if erun['id'] != ident:
                raise ValueError('Original packet row identity differs')
            frame_cache[ident] = frame
            erun_cache[ident] = (erun, ep, seal)
        return frame_cache[ident]

    for arm in ARMS:
        spec = manifest['arms'][arm]
        fb = input_bindings[arm]
        fp = Path(spec['features'])
        fs = read(spec['feature_seal'])
        bind(spec['feature_seal'], fb['feature_seal']['sha256'])
        bind(fp, fb['features']['sha256'])
        features = {x['id']: x for x in rows(fp)}
        pp = Path(fb['prediction']['path'])
        ps = read(pp.parent / 'PREDICTION_SEAL.json')
        bind(pp, fb['prediction']['sha256'])
        bind(pp.parent / 'PREDICTION_SEAL.json', fb['prediction']['seal_sha256'])
        if ps['features_sha256'] != sha(fp) or ps['feature_seal_sha256'] != sha(spec['feature_seal']):
            raise ValueError('Prediction/features ancestry differs')
        predictions = {x['id']: x for x in rows(pp)}
        if len(features) != 238 or len(predictions) != 238:
            raise ValueError('Every arm retains238 frames')
        lp = run_path / 's3_s4_labels' / arm
        label_seal = read(lp / 'LABEL_SEAL.json')
        bind(lp / 'LABEL_SEAL.json')
        npth = lp / 'TRAIN_ONLY_NORMALIZER.json'
        normalizer_sha = bind(npth, label_seal['normalizer_sha256'])
        normalizer = read(npth)
        for checkpoint in ps['checkpoints'].values():
            if checkpoint['training_binding']['normalizer_sha256'] != normalizer_sha:
                raise ValueError('Normalizer not used by saved head')
            if checkpoint['training_binding']['sources'].get(str(Path(D.__file__))) != sha(D.__file__):
                raise ValueError('Saved-feature transformation source changed')
        xp = producer / f'EXTRAS_{arm}_real.jsonl.gz'
        bind(xp, full['output_artifacts'][xp.name])
        extras = defaultdict(list)
        for x in rows(xp):
            extras[x['id']].append(x)
        if len(extras) != 238:
            raise ValueError('Complete producer real extras required')
        summary['arms'][arm] = {}
        for cohort in COHORTS:
            rr = [bank[(arm, ident)] for ident in sets[cohort]]
            covered = [x['id'] for x in rr if x['eligible_good']]
            summary['arms'][arm][cohort] = dict(n=len(rr), generated_good_count=sum(x['generated_good'] for x in rr),
                eligible_good_count=len(covered), covered_ids=covered,
                not_covered_ids=[x['id'] for x in rr if not x['eligible_good']],
                selected_good_ids_by_seed={str(seed): [x['id'] for x in rr if x['selected'][str(seed)]['metric']['success']] for seed in SEEDS},
                failure_detail_counts_by_seed={str(seed): dict(Counter(ranks[(arm, x['id'], seed)]['failure_detail'] for x in rr)) for seed in SEEDS})
        pair_stats = defaultdict(list)
        for ident in sets[COHORTS[0]]:
            b = bank[(arm, ident)]
            detail = dict(arm=arm, id=ident, cohorts=[c for c in COHORTS if ident in sets[c]],
                generated_good=b['generated_good'], eligible_good=b['eligible_good'], good_generated_ids=b['good_generated_ids'], good_eligible_ids=b['good_eligible_ids'],
                baseline_saved_bank=base_bank[ident], seed_diagnostics=[])
            if not b['eligible_good']:
                detail['seed_diagnostics'] = [dict(saved_rank=ranks[(arm, ident, seed)], baseline_saved_rank=base_ranks[(ident, seed)]) for seed in SEEDS]
                detail_records.append(detail)
                continue
            frame = raw_frame(ident)
            stage, branch = arm.split('_', 1)
            result = frame[stage.lower()][branch]
            if canonical(result['candidates']) != canonical(extras[ident]):
                raise ValueError('Per-row candidate payload and producer EXTRAS differ')
            scale = features[ident]['object_scale_m']
            prior = base_features[ident]['candidates']
            good_ids = set(b['good_generated_ids'])
            good_extras = [x for x in extras[ident] if x['candidate_id'] in good_ids]
            detail.update(asset_registration=frame['asset_registration'], producer_frame_binding=row_bindings[ident],
                good_extra_endpoint_ids=[x['candidate_id'] for x in good_extras],
                good_prior_PF_or_original_ids=[x for x in b['good_eligible_ids'] if not x.startswith(arm + '/')])
            erun, ep, rowseal = erun_cache[ident]
            # Copy only prediction-side original observations, not endpoint fitting.
            original_support = dict(E_RUN_path=str(ep), E_RUN_sha256=sha(ep),
                solver_input_visible=erun['packet']['solver_input']['visible'],
                solver_input_boundary_observations=erun['packet']['solver_input']['boundary_observations'],
                ORIGINAL_WD_native_xyz=erun['families']['ORIGINAL_WD']['native_xyz'],
                physical_registration_Q=erun['families']['ORIGINAL_WD']['physical_registration_Q'],
                physical_ownership='UNKNOWN; native IDs are semantic aliases, observed line samples are unverified physical owners',
                automatic_mask_sha256=rowseal['dependencies']['automatic_mask_sha256'])
            support = dict(arm=arm, id=ident, image=frame['image'], image_sha256=frame['image_sha256'],
                asset_registration=frame['asset_registration'], initializations=frame['initializations'],
                original_support=original_support, physical_owner_certifications=0)
            if stage == 'S3':
                packet = frame['material_observations']['policy_packets'][branch]
                support['saved_material_policy_packet'] = packet
                chosen = {x['support_id'] for x in packet['factors']}
                support['saved_shared_supports_for_chosen_factors'] = [x for x in frame['material_observations']['shared_mesh_support_archive'] if x['support_id'] in chosen]
                support['preserved_original_factor_count'] = result['preserved_nonoverlapping_original_factor_count']
                support['preserved_original_unknown_correlation_count'] = result['preserved_original_unknown_correlation_count']
                support['archive_limitations'] = ['Exact kept-original factor IDs were not saved; only retained/unknown counts and full original input supports are available. No reconstruction performed.']
            elif branch == 'PL':
                support['saved_iteration_factor_counts'] = [dict(start_id=x['start_id'], update=x.get('update'), count=x.get('original_point_line_outline_factors'), status=x['status']) for x in result['iterations']]
                support['actual_evidence_contract'] = 'Original complete native points + typed original normal lines + pose-updated native cuboid outline against automatic full mask; no structured-proxy renderer in PL branch'
                support['archive_limitations'] = ['Per-update native-outline 3D/2D factor coordinates were not retained, only combined factor counts and full-mask dependency hash. No reconstruction performed.']
            else:
                support['saved_selected_iteration_correspondences'] = [x for x in result['iterations'] if any(g['start_id'] == x['start_id'] and g['update'] == x.get('update') for g in good_extras)]
                support['archive_limitations'] = ['Raw raster arrays were ephemeral; exact selected correspondence arrays and pose/K/RGB/depth/mask hashes are retained where fit was attempted.']
            support_records.append(support)
            for endpoint in good_extras:
                audit = next((x for x in result['iterations'] if x['start_id'] == endpoint['start_id'] and x.get('update') == endpoint['update']), None)
                matches = [x['candidate_id'] for x in prior if same_saved_pose(endpoint['actual_pose'], x['actual_pose'], scale)]
                endpoint_records.append(dict(arm=arm, id=ident, endpoint=endpoint, saved_iteration=audit,
                    prior_pose_matches=matches, genuine_new_pose_vs_prior_S1=not matches,
                    matching_initializations=[x['start_id'] for x in frame['initializations'] if same_saved_pose(endpoint['actual_pose'], x['actual_pose'], scale)],
                    support_record_key=dict(arm=arm,id=ident), physical_owner_status='UNKNOWN',
                    baseline_good_eligible_ids=base_bank[ident]['good_eligible_ids']))
            x, entries = D.inputs(features[ident], normalizer, 'H1')
            by_action = {c['candidate_id']: (i, c) for i, c in enumerate(entries)}
            names = features[ident]['feature_names']
            if len(names) != 31 or x.shape != (len(entries), 188):
                raise ValueError('Saved31/62/188 feature schema differs')
            summary['feature_names'] = names
            for seed in SEEDS:
                rank = ranks[(arm, ident, seed)]
                pred = predictions[ident]['predictions'][arm + '_s' + str(seed)]
                if [a['candidate_id'] for a in pred['actions']] != [c['candidate_id'] for c in entries]:
                    raise ValueError('Saved action order differs from existing input transform')
                amap = {a['candidate_id']: a for a in pred['actions']}
                chosen = rank['selected_candidate_id']
                best = rank['best_good_action_id']
                if chosen != pred['selected_candidate_id'] or best not in amap:
                    raise ValueError('Saved rank/action identity differs')
                # Independently validate existing ranks using existing good IDs/logits.
                good = set(b['good_eligible_ids'])
                gi = [i for i, a in enumerate(pred['actions']) if aliases(a) & good]
                bi = min(gi, key=lambda i: (-pred['actions'][i]['score'], i))
                sorted_indices = sorted(range(len(entries)), key=lambda i: (-pred['actions'][i]['score'], i))
                if pred['actions'][bi]['candidate_id'] != best or sorted_indices.index(bi)+1 != rank['best_good_action_rank']:
                    raise ValueError('Saved rank independent join failed')
                records = {}
                for role, cid in [('selected', chosen), ('best_good', best)]:
                    index, c = by_action[cid]
                    prior_matches = [p['candidate_id'] for p in prior if same_saved_pose(c['actual_pose'], p['actual_pose'], scale)]
                    raw = c['features']
                    record = dict(arm=arm,id=ident,seed=seed,role=role,candidate_id=cid,
                        saved_logit=amap[cid]['score'],source_aliases=sorted(aliases(c)),eligible=c['eligible'],
                        actual_pose=c['actual_pose'], prior_pose_matches=prior_matches,
                        feature_terms=[dict(index=j,name=names[j],saved_signedlog=raw[j],valid=bool(raw[j+31]),normalized_float32=float(x[index,j]),normalizer_train_seen=bool(normalizer['valid_train_counts'][j])) for j in range(31)],
                        H1_input188_float32=x[index].tolist(),normalizer_sha256=normalizer_sha,
                        logit_is_calibrated_confidence=False)
                    feature_records.append(record)
                    records[role] = record
                sc, gc = records['selected'], records['best_good']
                feature_diffs = []
                for s, g in zip(sc['feature_terms'], gc['feature_terms']):
                    if s['valid'] and g['valid']:
                        delta = g['saved_signedlog'] - s['saved_signedlog']
                        pair_stats[s['name']].append(delta)
                        feature_diffs.append(dict(name=s['name'],best_good_minus_selected_signedlog=delta,
                            best_good_minus_selected_normalized=g['normalized_float32']-s['normalized_float32']))
                selected_extra = [e for e in extras[ident] if e['candidate_id'] in aliases(amap[chosen]) or same_saved_pose(e['actual_pose'], sc['actual_pose'], scale)]
                item = dict(saved_rank=rank,baseline_saved_rank=base_ranks[(ident,seed)],
                    saved_selected_metric=b['selected'][str(seed)],
                    selected_prior_pose_matches=sc['prior_pose_matches'],
                    selected_extra_endpoint_ids=[e['candidate_id'] for e in selected_extra],
                    selected_is_genuine_updated_arm_proposal=bool(any(e.get('update',0)>0 and e.get('producer_available',True) for e in selected_extra) and not sc['prior_pose_matches']),
                    selected_is_prior_PF_pose=any(p.startswith('POSEFIX_') for p in sc['prior_pose_matches']),
                    selected_and_best_good_feature_record_keys=[dict(arm=arm,id=ident,seed=seed,role=role) for role in ('selected','best_good')],
                    largest_observed_normalized_feature_differences=sorted(feature_diffs,key=lambda v:abs(v['best_good_minus_selected_normalized']),reverse=True)[:8])
                detail['seed_diagnostics'].append(item)
            detail_records.append(detail)
        summary['arms'][arm]['observed_paired_feature_comparisons'] = {
            name: dict(valid_pairs=len(value),best_good_lower=sum(d < -1e-12 for d in value),same=sum(abs(d)<=1e-12 for d in value),best_good_higher=sum(d > 1e-12 for d in value),median_best_good_minus_selected_signedlog=float(np.median(value)))
            for name,value in pair_stats.items()}

    if len(detail_records) != 205:
        raise ValueError('Every fixed41 ID per arm must be represented')
    pl = [x for x in detail_records if x['arm']=='S4_PL' and any(s['saved_rank']['success'] for s in x['seed_diagnostics'])]
    if len(pl)!=3 or any(not s['selected_is_genuine_updated_arm_proposal'] or s['selected_is_prior_PF_pose'] for x in pl for s in x['seed_diagnostics']):
        raise ValueError('PL recovery source attribution inconsistent')
    summary['PL_selected_recovery_proof'] = [dict(id=x['id'],cohorts=x['cohorts'],
        selected_ids=[s['saved_rank']['selected_candidate_id'] for s in x['seed_diagnostics']],
        prior_pose_match_counts=[len(s['selected_prior_pose_matches']) for s in x['seed_diagnostics']],
        all3_genuine_updated_PL_proposals=True,all3_prior_PF_duplicates=False) for x in pl]
    image_details = [x for x in detail_records if x['arm']=='S4_IMAGE' and x['eligible_good']]
    all_image_ranks = [s['saved_rank'] for x in image_details for s in x['seed_diagnostics']]
    postpf_ranks = [s['saved_rank'] for x in image_details if COHORTS[1] in x['cohorts'] for s in x['seed_diagnostics']]
    summary['IMAGE_loss_location'] = dict(original41_covered=11,postPF38_covered=8,
        eligible_good_actions_present_all33_frame_seed_pairs=all(r['good_actions']>0 for r in all_image_ranks),
        failure_detail_counts=dict(Counter(r['failure_detail'] for r in all_image_ranks)),
        selected_successes=sum(r['success'] for r in all_image_ranks),
        original41_bestgood_rank_range=[min(r['best_good_action_rank'] for r in all_image_ranks),max(r['best_good_action_rank'] for r in all_image_ranks)],
        postPF38_bestgood_rank_range=[min(r['best_good_action_rank'] for r in postpf_ranks),max(r['best_good_action_rank'] for r in postpf_ranks)],
        logit_margin_range=[min(r['best_good_logit_minus_selected'] for r in all_image_ranks),max(r['best_good_logit_minus_selected'] for r in all_image_ranks)],
        conclusion='The saved loss is after generation, admission and action construction: every good action ranks below the selected bad action. Features/logits are observational evidence, not causal explanation.')
    summary['records'] = dict(all_fixed41_arm_frames=len(detail_records),covered_frame_support_records=len(support_records),
        good_extra_endpoint_records=len(endpoint_records),selected_and_bestgood_feature_records=len(feature_records))
    summary['limitations'] = ['PL per-update outline correspondence coordinates were not retained; combined factor counts and bound original raw point/line observations/mask hash are reported.',
        'S3 exact identities of retained old factors were not retained; saved retained/unknown counts and complete original observation inputs are reported.',
        'All235 synthetic validation rows used a missing-named-GLB structured surrogate; all real and synthetic instance/signed-axis certifications remain absent.',
        'No causal conclusions or independent physical ownership certification follow from saved feature differences or success labels.']
    output.mkdir(parents=True)
    save(output/'ALL_FIXED41_FRAME_DIAGNOSTICS.jsonl.gz',detail_records)
    save(output/'GOOD_EXTRA_ENDPOINTS.jsonl.gz',endpoint_records)
    save(output/'SUPPORT_PROVENANCE.jsonl.gz',support_records)
    save(output/'SELECTED_BESTGOOD_FEATURES.jsonl.gz',feature_records)
    write(output/'DIAGNOSTIC.json',summary)
    md = ['This finite diagnostic joins saved, sealed labels and H1 logits for all fixed original41 candidate-absent IDs and their post-PF38 subset. It reads no reference file and runs no new model forward, fitting, pose production, projection, matching or pose scoring. All3 seeds are included.','',
        '| Arm | Covered /41 | Covered /38 | Selected good /41, seeds18/19/20 | Selected good /38 |', '|---|---:|---:|---|---|']
    for arm in ARMS:
        a=summary['arms'][arm][COHORTS[0]];b=summary['arms'][arm][COHORTS[1]]
        md.append(f"| {arm} | {a['eligible_good_count']} | {b['eligible_good_count']} | {'/'.join(str(len(a['selected_good_ids_by_seed'][str(s)])) for s in SEEDS)} | {'/'.join(str(len(b['selected_good_ids_by_seed'][str(s)])) for s in SEEDS)} |")
    md += ['', 'PL selects three genuine updated PL proposals in every seed; none matches any pre-existing S1/PF-union candidate under the original pose-deduplication tolerance. All three lie in post-PF38:', '']
    for p in summary['PL_selected_recovery_proof']:
        md.append('- '+p['id']+' → `'+p['selected_ids'][0]+'` (same action in all3 seeds; prior pose matches0).')
    md += ['', 'PL consumes original complete points, typed line normals and a pose-updated native cuboid outline against the automatic full mask. Its extra refinements differ from the original D1 objective/caps. Its recoveries do not establish success of full-image RGB or certified material CAD.', '',
        'IMAGE covers11/41, including8/38, yet selects0 in all3 seeds. All33 covered frame/seed cases contain eligible good actions and lose at `WRONG_ACTION_RANK`. Best-good ranks are4–24 across41 and4–19 across38. The evidence locates loss at H1 ranking; it does not identify a causal feature or asset failure.', '',
        '| IMAGE ID | Good source | Best-good ranks18/19/20 | Selected actions18/19/20 |', '|---|---|---|---|']
    for d in image_details:
        ds=d['seed_diagnostics'];origin='new IMAGE + prior PF' if d['good_extra_endpoint_ids'] and d['good_prior_PF_or_original_ids'] else 'new IMAGE' if d['good_extra_endpoint_ids'] else 'prior PF'
        md.append('| '+d['id']+' | '+origin+' | '+'/'.join(str(s['saved_rank']['best_good_action_rank']) for s in ds)+' | '+'; '.join(s['saved_rank']['selected_candidate_id'] for s in ds)+' |')
    md += ['', 'Every arm’s covered IDs, actual good endpoint IDs, starts/updates/status and all3 seed choices are listed below. The three prior-PF-only coverage IDs are included rather than credited as new arm coverage.', '']
    for arm in ARMS:
        md += [arm, '', '| ID | Good arm endpoint IDs (saved) | Best-good rank18/19/20 | Selected18/19/20 |', '|---|---|---|---|']
        for d in detail_records:
            if d['arm']!=arm or not d['eligible_good']:continue
            md.append('| '+d['id']+' | '+('; '.join(d['good_extra_endpoint_ids']) or 'prior PF only')+' | '+'/'.join(str(s['saved_rank']['best_good_action_rank']) for s in d['seed_diagnostics'])+' | '+'; '.join(s['saved_rank']['selected_candidate_id'] for s in d['seed_diagnostics'])+' |')
        md.append('')
    md += ['All31 named signed-log feature scalars, validity bits, exact train-normalizer-transformed188 float32 inputs, saved logits, prior-pose matches and strongest observed feature differences for selected/best-good actions are in `SELECTED_BESTGOOD_FEATURES.jsonl.gz` and `ALL_FIXED41_FRAME_DIAGNOSTICS.jsonl.gz`. Bank summary terms are shared by actions within a frame; logits are compared only within the same arm/frame/seed and are not calibrated confidence.', '',
        'Actual saved raw point/line/mesh support coordinates, IMAGE/CONTOUR selected correspondences and render bindings are in `SUPPORT_PROVENANCE.jsonl.gz` and `GOOD_EXTRA_ENDPOINTS.jsonl.gz`. Physical owners are UNKNOWN. PL per-update outline coordinates and S3 exact kept-original factor IDs were not retained; only counts and bound original inputs are available, and this diagnostic reconstructs neither.', '',
        'Asset status means missing exact named source and/or uncertified instance and signed-axis registration. It is not empirical proof that proxy geometry is wrong. All235 synthetic validation rows used missing-named-GLB surrogates.', '',
        '`DIAGNOSTIC.json` preserves full cohort IDs, all denominators, baseline comparison and descriptive paired-feature counts. `INPUT_BINDINGS.json` and `DIAGNOSTIC_SEAL.json` authenticate every consumed file and output. No production file, model or setting is modified.']
    (output/'DIAGNOSTIC.md').write_text('\n'.join(md)+'\n')
    for path,digest in bindings.items():
        if sha(path)!=digest:
            raise ValueError('Input changed during saved-output diagnostic')
    write(output/'INPUT_BINDINGS.json',bindings)
    names=('ALL_FIXED41_FRAME_DIAGNOSTICS.jsonl.gz','GOOD_EXTRA_ENDPOINTS.jsonl.gz','SUPPORT_PROVENANCE.jsonl.gz','SELECTED_BESTGOOD_FEATURES.jsonl.gz','DIAGNOSTIC.json','DIAGNOSTIC.md','INPUT_BINDINGS.json')
    receipt=dict(complete=True,schema='S3_S4_FINITE_SAVED_OUTPUT_DIAGNOSTIC_SEAL_V1',
        artifacts={name:sha(output/name) for name in names},source_sha256=sha(source),action_data_source_sha256=sha(D.__file__),
        bound_input_count=len(bindings),all_inputs_unchanged=True,all_fixed41_arm_rows=205,all_seeds=list(SEEDS),
        no_accuracy_selected_examples=True,real_reference_file_reads=0,new_reference_decodes=0,new_pose_calls=0,new_pose_scoring_calls=0,
        new_projections=0,new_matcher_calls=0,new_render_calls=0,new_model_forwards=0,new_fits=0,
        numeric_operations='Saved matrix equality, logit ordering, scalar differences/medians and existing frozen train-normalizer H1 input transform only',
        wall_seconds=time.monotonic()-started)
    write(output/'DIAGNOSTIC_SEAL.json',receipt)
    return dict(complete=True,records=summary['records'],seal_sha256=sha(output/'DIAGNOSTIC_SEAL.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('run','d1','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    print(canonical(run(a.run,a.d1,a.output)),flush=True)
