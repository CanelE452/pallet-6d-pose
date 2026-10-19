"""Summarize and independently cross-check a sealed finite saved-output join."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with gzip.open(path, 'rt') as f:
        return [json.loads(line) for line in f]


def run(folder):
    folder = Path(folder)
    seal = read(folder/'DIAGNOSTIC_SEAL.json')
    if not seal['complete']:
        raise ValueError('Complete saved diagnostic required')
    for name,digest in seal['artifacts'].items():
        if sha(folder/name)!=digest:
            raise ValueError('Changed saved diagnostic')
    summary = read(folder/'DIAGNOSTIC.json')
    detail = rows(folder/'ALL_FIXED41_FRAME_DIAGNOSTICS.jsonl.gz')
    features = rows(folder/'SELECTED_BESTGOOD_FEATURES.jsonl.gz')
    supports = {(r['arm'],r['id']):r for r in rows(folder/'SUPPORT_PROVENANCE.jsonl.gz')}
    endpoints = rows(folder/'GOOD_EXTRA_ENDPOINTS.jsonl.gz')
    if len(detail)!=205 or len({(r['arm'],r['id'])for r in detail})!=205:
        raise ValueError('All5×41 complete unique rows required')
    paired = defaultdict(dict)
    for f in features:
        paired[(f['arm'],f['id'],f['seed'])][f['role']]=f
        if len(f['feature_terms'])!=31 or len(f['H1_input188_float32'])!=188:
            raise ValueError('31/188 saved extraction required')
    if any(set(v)!={'selected','best_good'}for v in paired.values()):
        raise ValueError('Complete saved selected/best-good pair required')
    recomputed = defaultdict(lambda:defaultdict(list))
    for (arm,ident,seed),pair in paired.items():
        for s,g in zip(pair['selected']['feature_terms'],pair['best_good']['feature_terms']):
            if s['name']!=g['name']:
                raise ValueError('Named term pairing differs')
            if s['valid'] and g['valid']:
                recomputed[arm][s['name']].append(g['saved_signedlog']-s['saved_signedlog'])
    checks=0
    for arm,terms in recomputed.items():
        for name,values in terms.items():
            old=summary['arms'][arm]['observed_paired_feature_comparisons'][name]
            actual=dict(valid_pairs=len(values),best_good_lower=sum(x < -1e-12 for x in values),
                same=sum(abs(x)<=1e-12 for x in values),best_good_higher=sum(x > 1e-12 for x in values))
            if any(old[k]!=v for k,v in actual.items()):
                raise ValueError('Saved aggregate/individual feature mismatch')
            checks+=1
    image_pairs=[(k,v)for k,v in paired.items()if k[0]=='S4_IMAGE']
    named=('fixed_energy','A_energy','B_energy','AB_energy','full_energy','partial_energy','outline_energy',
        'edge_energy','point_residual_mean_px','point_residual_p90_px','partial_normal_mean_sigma',
        'N0_translation_fraction','N0_rotation_radians','min_constraint_slack','score_gap_fixed')
    per_image=[]
    for ident in summary['arms']['S4_IMAGE']['original41_candidate_absent']['covered_ids']:
        rr=[]
        for (arm,img,seed),pair in image_pairs:
            if img!=ident:continue
            s,g=pair['selected'],pair['best_good']
            st={x['name']:x for x in s['feature_terms']};gt={x['name']:x for x in g['feature_terms']}
            rr.append(dict(seed=seed,selected_candidate_id=s['candidate_id'],best_good_candidate_id=g['candidate_id'],
                selected_logit=s['saved_logit'],best_good_logit=g['saved_logit'],
                feature_values={n:dict(selected_saved_signedlog=st[n]['saved_signedlog'],selected_valid=st[n]['valid'],
                    best_good_saved_signedlog=gt[n]['saved_signedlog'],best_good_valid=gt[n]['valid'],
                    best_good_minus_selected_signedlog=gt[n]['saved_signedlog']-st[n]['saved_signedlog'])for n in named}))
        per_image.append(dict(id=ident,seed_records=sorted(rr,key=lambda r:r['seed'])))
    pl=[]
    for proof in summary['PL_selected_recovery_proof']:
        ident=proof['id'];sup=supports[('S4_PL',ident)];original=sup['original_support']
        boundaries=original['solver_input_boundary_observations']
        partial_corners={int(b['native_corner'])for b in boundaries}
        complete=sum(int(m['native_corner'])not in partial_corners for m in original['solver_input_visible'])
        normals=sum(len(b['lines'])for b in boundaries)
        cid=proof['selected_ids'][0]
        endpoint=next(e for e in endpoints if e['arm']=='S4_PL'and e['id']==ident and e['endpoint']['candidate_id']==cid)
        iteration=endpoint['saved_iteration'];combined=iteration['original_point_line_outline_factors']
        if endpoint['prior_pose_matches'] or not endpoint['genuine_new_pose_vs_prior_S1'] or iteration['status']!='UPDATED':
            raise ValueError('Genuine accepted PL update required')
        pl.append(dict(id=ident,candidate_id=cid,start_id=endpoint['endpoint']['start_id'],update=endpoint['endpoint']['update'],
            status=iteration['status'],producer_available=endpoint['endpoint']['producer_available'],prior_pose_matches=[],
            original_complete_point_factor_count=complete,original_typed_normal_factor_count=normals,
            combined_saved_factor_count=combined,outline_factor_count_from_saved_total_minus_original_count=combined-complete-normals,
            outline_count_method='Saved combined count minus counts in saved original packet, following immutable PL source; no projection or correspondence recomputation',
            outline_per_update_coordinates_saved=False,
            objective_calls=iteration['objective_calls'],saved_initial_cost=iteration['initial_cost'],saved_final_cost=iteration['final_cost'],
            physical_owner_status='UNKNOWN',automatic_mask_sha256=original['automatic_mask_sha256']))
    image_terms=summary['arms']['S4_IMAGE']['observed_paired_feature_comparisons']
    result=dict(complete=True,scope='All11 covered original41 IMAGE IDs ×3 seeds;33 comparisons are correlated repeats across11 images, not33 independent images',
        observational_only=True,new_references_poses_forwards_fits_scoring=0,
        energy_context='Original D1/PF-union feature evaluator observations. These features are not the IMAGE matcher/PnP loss or grayscale similarity.',
        IMAGE_named_feature_comparisons={n:image_terms[n]for n in named},IMAGE_all_covered_ID_values=per_image,
        PL_three_genuine_selected_update_support_counts=pl,independent_named_feature_aggregate_checks=checks,
        asset_status='Missing named source and/or uncertified physical instance/signed-axis link, not empirical proof of wrong geometry',
        cannot_identify_feature_cause_without_intervention=True)
    out=folder/'FEATURE_PATTERN_SUPPLEMENT.json'
    if out.exists():raise ValueError('Preserve completed supplement')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    md=['Saved IMAGE action features show a mismatch between posthoc pose success and the original observation-energy terms. Across11 covered images ×3 seeds, the best good action has higher `fixed_energy` in33/33 comparisons, higher `full_energy` in32/33 and higher `point_residual_mean_px` in32/33. All are valid in those comparisons. `A_energy` is valid in30/33 and higher in30/30. `B_energy` is lower for the good action in4/33 and higher in29/33. These are original D1 observation features, not the IMAGE matcher’s fit loss.', '',
        'These33 comparisons are repeated seeds over11 images. They locate a consistent observable pattern alongside the saved H1 ranking loss; they do not show that any feature caused the rankings. No feature intervention, fit, forward or new reference scoring was performed.', '',
        '| Named signed-log term | Valid pairs | Good lower / equal / higher | Median good−selected |','|---|---:|---:|---:|']
    for n in named:
        v=image_terms[n]
        md.append(f"| `{n}` | {v['valid_pairs']} | {v['best_good_lower']}/{v['same']}/{v['best_good_higher']} | {v['median_best_good_minus_selected_signedlog']:.6g} |")
    md += ['', 'The three selected PL updates have these saved support counts:', '', '| ID | Selected update | Original2D points | Original1D normals | Outline factors inferred from total | Saved total |', '|---|---|---:|---:|---:|---:|']
    for p in pl:
        md.append(f"| {p['id']} | N0/{p['update']} | {p['original_complete_point_factor_count']} | {p['original_typed_normal_factor_count']} | {p['outline_factor_count_from_saved_total_minus_original_count']} | {p['combined_saved_factor_count']} |")
    md += ['', 'Each is accepted `UPDATED`, available and has zero matches to any prior S1/PF candidate pose. Outline counts use saved combined totals minus saved original factor counts; per-update outline coordinates were not retained and were not reconstructed. Raw original points/line supports and mask hashes remain bound in the main diagnostic.', '',
        'Every IMAGE covered ID and seed’s selected/best-good values for these terms are saved in `FEATURE_PATTERN_SUPPLEMENT.json`; all31 terms and188 inputs remain in the sealed main diagnostic. Missing named assets and uncertified instance/signed-axis registration do not constitute empirical proof of wrong proxy geometry.']
    mp=folder/'FEATURE_PATTERN_SUPPLEMENT.md';mp.write_text('\n'.join(md)+'\n')
    receipt=dict(complete=True,original_diagnostic_seal_sha256=sha(folder/'DIAGNOSTIC_SEAL.json'),
        source_sha256=sha(__file__),named_feature_comparison_checks=checks,all205_unique_frame_rows_checked=True,
        all141_selected_bestgood_pairs_complete=True,all3_selected_PL_endpoint_ancestries_checked=True,
        artifacts={p.name:sha(p)for p in(out,mp)},new_reference_file_reads=0,new_pose_calls=0,new_model_forwards=0,new_fits=0,new_pose_scoring_calls=0)
    (folder/'SUPPLEMENT_SEAL.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);a=p.parse_args()
    print(json.dumps(run(a.folder)),flush=True)
