"""V192 source TRAIN lineage census; text metadata only, no metrics or arrays."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json

HOME = Path(__file__).parent


def safe(p):
    p = Path(p)
    assert not any('v112' in s.lower() for s in p.parts)
    for q in [p, *p.parents]:
        assert not q.is_symlink(), 'No symlink traversal'
    return p


def read(p):
    return json.loads(safe(p).read_text())


def sha(p):
    return hashlib.sha256(safe(p).read_bytes()).hexdigest()


def rows(binding):
    p = safe(binding['path'])
    data = p.read_bytes()
    assert hashlib.sha256(data).hexdigest() == binding['sha256']
    return [json.loads(line) for line in data.splitlines() if line]


def count(values):
    return dict(sorted(Counter(str(v) for v in values).items()))


def key(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def summary(rr):
    hashes = [r.get('actual_rgb96_sha256') or r.get('actual_raw_RGB96_sha256') or
              r.get('raw_RGB96_sha256') for r in rr]
    good_hashes = [h for h in hashes if h is not None]
    available_hashes = [h for r, h in zip(rr, hashes)
                        if h is not None and r['original_query_available']]
    return dict(rows=len(rr), original_inputU=sum(not r['original_query_available'] for r in rr),
                native_rows=count(r['native_slot'] for r in rr),
                unique_occurrence_ids=len({r['row_id'] for r in rr}),
                unique_original_indices=len({r['original_global_index'] for r in rr}),
                distinct_q_L_native=len({key([r['query_xy'], r['L'], r['native_slot']]) for r in rr}),
                q_or_L_missing=sum(r['query_xy'] is None or r['L'] is None for r in rr),
                inherited_RGB96_hash_present=len(good_hashes),
                distinct_inherited_RGB96_hashes=len(set(good_hashes)),
                available_distinct_inherited_RGB96_hashes=len(set(available_hashes)),
                inherited_hash_duplicate_excess=len(good_hashes)-len(set(good_hashes)),
                inherited_hash_groups_are_exact_bytes_not_independent_scene_certification=True)


def unique(rr, field):
    return len({key(r[field]) for r in rr})


def public_source(rr):
    return dict(frames=unique([r['original'] for r in rr], 'frame_index'),
                source_images=unique([r['original'] for r in rr], 'source_id'),
                sequences=count(r['original']['source_id'].split(':')[0] for r in rr),
                frame_indices=sorted({r['original']['frame_index'] for r in rr}),
                original_query_ids=unique(rr, 'row_id'))


def main():
    rule = read(HOME/'PROSPECTIVE_METADATA_AUDIT_RULES.json')
    note = read(HOME/'PROSPECTIVE_SCHEMA_ADAPTER_NOTE.json')
    data = {k: rows(b) for k, b in rule['inputs'].items()}
    protocol_binding = note['additional_metadata_inputs']['raw_cache_protocol']
    assert sha(protocol_binding['path']) == protocol_binding['sha256']
    protocol = read(protocol_binding['path'])
    aug, scenes, public, cf, direct, census, real, syn = [data[k] for k in
        ('aug_rows','synth_scene_metadata','public_rows','cf_rows','direct_rows',
         'direct_census','real_attempts','synthetic_attempts')]
    assert [len(x) for x in (aug,scenes,public,cf,direct,census,real,syn)] == [65536,512,1480,3816,2792,3270,66,1024]
    assert all(r['split']=='TRAIN' for r in aug+scenes)
    assert all(r['split']=='SOURCE_DEVELOPMENT' for r in public+cf)
    pk=set(protocol['public_known_original_indices'])
    ck=set(protocol['CF_known_occurrence_indices'])
    public_known=[r for r in public if r['original_global_index'] in pk]
    cf_known=[r for r in cf if r['domain_index'] in ck]
    assert len(public_known)==253 and len(cf_known)==314
    factors={r['base_index']:r['factors'] for r in scenes}
    scene_by_index={r['scene_index']:r for r in scenes}
    census_by_key={(r['domain'],r['attempt_id'],r['variant']):r for r in census}
    assert len(census_by_key)==3270
    known=[]
    for r in direct:
        c=census_by_key[(r['domain'],r['attempt_id'],r['variant'])]
        assert r['root_admitted']==c['direct_training_admitted']
        if r['root_admitted'] and r['native_slot']==c['selected_native_slot']:
            known.append(dict(r, direct_target=1 if r['variant']=='COVER' else 0))
    assert len(known)==306
    known_syn=[r for r in known if r['domain']=='SYNTH_FINITE_END']
    known_real=[r for r in known if r['domain']=='REAL_SEMANTIC_END_WEAK']
    real_by_id={r['pair_id']:r for r in real}
    syn_by_id={r['attempt_id']:r for r in syn}
    sk={r['attempt_id'] for r in known_syn};rk={r['attempt_id'] for r in known_real}
    accepted_syn=[syn_by_id[k] for k in sorted(sk)]
    accepted_real=[real_by_id[k] for k in sorted(rk)]
    public_ids={r['row_id'] for r in public};pk_ids={r['row_id'] for r in public_known}
    cf_original_ids={r['original']['original_query_id'] for r in cf}
    cf_known_ids={r['original']['original_query_id'] for r in cf_known}
    direct_real_ids={r['original_occurrence_id'] for r in known_real}
    direct_syn_ids={r['original_occurrence_id'] for r in known_syn}
    pub_hash={r['actual_raw_RGB96_sha256'] for r in public}
    aug_hash={r['actual_rgb96_sha256'] for r in aug}
    output=dict(completed=True,PASS=True,source_metadata_only=True,
        rules=dict(path=str(HOME/'PROSPECTIVE_METADATA_AUDIT_RULES.json'),sha256=sha(HOME/'PROSPECTIVE_METADATA_AUDIT_RULES.json')),
        adapter_note=dict(path=str(HOME/'PROSPECTIVE_SCHEMA_ADAPTER_NOTE.json'),sha256=sha(HOME/'PROSPECTIVE_SCHEMA_ADAPTER_NOTE.json')),
        code=dict(path=str(Path(__file__)),sha256=sha(__file__)), input_bindings=rule['inputs'],
        all_domains={k:summary(rr) for k,rr in [('SYNTH_AUG65536',aug),('PUBLIC_TRAIN1480',public),('CF3816',cf),('DIRECT_CONTROL',direct)]})
    output['synthetic_augmentation']=dict(original_bases=len(factors),original_scenes=len(scenes),
        original_decoded_RGB_hashes=unique(scenes,'decoded_RGB_SHA'),
        families=count(r['family'] for r in aug),
        family_source_image_counts={f:len({r['scene_index'] for r in aug if r['family']==f}) for f in sorted({r['family'] for r in aug})},
        original_source_query_ids=unique(aug,'source_image_original_query_id'),
        anchor_query_ids=unique(aug,'anchor_original_query_id'),
        family_U={f:sum(not r['original_query_available'] for r in aug if r['family']==f) for f in sorted({r['family'] for r in aug})},
        factor_base_counts={f:count(v.get(f,'MISSING') for v in factors.values()) for f in sorted({k for v in factors.values() for k in v})})
    output['public']=dict(all=public_source(public),owner_known=public_source(public_known),owner_known_native=count(r['native_slot'] for r in public_known),
        support_direct_regional_truth='NOT_DEFINED_IN_THIS_POOL; owner253 only')
    output['CF']=dict(original_frames=len({r['original']['original_frame_index'] for r in cf}),
        original_query_ids=len(cf_original_ids),pairs=len({r['original']['pair_id'] for r in cf}),
        variants=count(r['original']['variant'] for r in cf),
        owner_known_rows=len(cf_known),owner_known_native=count(r['native_slot'] for r in cf_known),
        owner_known_original_query_ids=len(cf_known_ids),owner_known_frames=len({r['original']['original_frame_index'] for r in cf_known}),
        original_query_ids_also_public=len(cf_original_ids&public_ids),
        known_original_query_ids_also_public_ownerknown=len(cf_known_ids&pk_ids),
        endpoint_external_truth='NOT_DEFINED; known owner F/T must not become native-end E/D')
    output['direct']=dict(full_census=len(census),physical_groups=len(direct)//8,
        missing_variant_null=sum(c['cache_row_indices'] is None for c in census),
        known_rows=len(known),unknown_rows=len(direct)-len(known),known_target=count(r['direct_target'] for r in known),
        known_native=count(r['native_slot'] for r in known),
        authority_known=count(r['authority'] for r in known),
        synthetic_accepted_attempts=len(accepted_syn),synthetic_accepted_source_bases=len({r['base_index'] for r in accepted_syn}),
        synthetic_native=count(r['native_slot'] for r in accepted_syn),
        synthetic_accepted_factor_attempts={f:count(r['factors'].get(f,'MISSING') for r in accepted_syn) for f in sorted({k for r in accepted_syn for k in r['factors']})},
        real_accepted_attempts=len(accepted_real),real_accepted_cases=len({r['case_id'] for r in accepted_real}),
        real_accepted_case_ids=sorted({r['case_id'] for r in accepted_real}),
        real_accepted_query_ids=sorted(direct_real_ids),real_accepted_donors=sorted({r['donor_id'] for r in accepted_real}),
        real_accepted_source_frames=len({r['original_input']['original_flags']['frame_index'] for r in accepted_real}),
        real_known_native=count(r['native_slot'] for r in known_real),
        real_known_origins_also_public=len(direct_real_ids&public_ids),
        real_known_origins_also_public_ownerknown=len(direct_real_ids&pk_ids),
        real_known_origins_also_CFanchors=len(direct_real_ids&cf_known_ids),
        synthetic_known_origins_also_original_aug=len(direct_syn_ids & {r['source_image_original_query_id'] for r in aug}),
        direct_before_rows_exact_hash_also_aug=sum(r['variant']=='BEFORE' and r['raw_RGB96_sha256'] in aug_hash for r in direct if r['domain']=='SYNTH_FINITE_END'),
        direct_before_rows_exact_hash_also_public=sum(r['variant']=='BEFORE' and r['raw_RGB96_sha256'] in pub_hash for r in direct if r['domain']=='REAL_SEMANTIC_END_WEAK'))
    saved=[]
    for r in accepted_syn:
        e=r['closed_proof']['actual_proof_metadata']['evidence_bindings']
        saved.append(dict(attempt_id=r['attempt_id'],native_slot=r['native_slot'],base_index=r['base_index'],**{k:e[k] for k in ['end_related96_samples','end_related96_COVER_changed_RGB_samples','end_related96_SHAM_changed_RGB_samples']}))
    output['saved_synthetic_end_signal']=dict(accepted_attempts=len(saved),
        end_related_sample_counts=count(r['end_related96_samples'] for r in saved),
        cover_changed_sample_counts=count(r['end_related96_COVER_changed_RGB_samples'] for r in saved),
        sham_changed_sample_counts=count(r['end_related96_SHAM_changed_RGB_samples'] for r in saved),
        scope='Existing producer evidence only; no new pixel replay, RGB magnitude or visual readability measurement.')
    output['independent_validation_status']=dict(for_existing_C54_C55='NO_SOURCE_HOLDOUT: all306 admitted known direct rows were in both training pools; a posthoc subset is TRAIN apparent.',
        prospective_group_units=['synthetic base (all variants/native/jitters/derived controls together)','public original source frame/sequence and selected-instance lineage if separately known','CF original anchor frame plus donor scene/object lineage connected across all replicas/triplets','real end original frame/case and donor lineage; join to original public/CF pool'],
        needed_before_any_future_fit='Freeze disjoint lineage groups and report train/held class/native8/factor coverage; no current/held/VAL score used. If no group-disjoint support exists, report that rather than promote row holdout.',
        object_and_texture_identity_limit='Procedural asset_seed and inherited source hashes are covariates, not independent physical-object/texture certification; real selected-instance/part and donor object lineage need their explicit upstream metadata.')
    output['new_model_fit_threshold_or_evaluation_reads']=0
    output['new_array_image_header_or_asset_body_reads']=0
    output['training_admission_or_original_label_changes']=0
    output['fully_solved']=False
    with (HOME/'SOURCE_TRAIN_LINEAGE_COVERAGE.json').open('x') as f:json.dump(output,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    with (HOME/'EXISTING_SYNTHETIC_END_SIGNAL_COUNTS.jsonl').open('x') as f:
        for r in saved:f.write(json.dumps(r,sort_keys=True)+'\n')
    print(json.dumps({k:output[k] for k in ['all_domains','public','CF','direct','saved_synthetic_end_signal']},ensure_ascii=False,indent=2))
    print('REPORT_SHA256',sha(HOME/'SOURCE_TRAIN_LINEAGE_COVERAGE.json'))


if __name__=='__main__':
    main()
