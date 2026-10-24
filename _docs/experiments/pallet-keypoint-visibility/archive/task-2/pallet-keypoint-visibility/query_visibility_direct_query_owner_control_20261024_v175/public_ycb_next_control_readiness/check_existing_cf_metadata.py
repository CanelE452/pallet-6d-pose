"""Read-only V161/V162 text lineage join. Never open/hash an asset NPZ or pixel."""
from pathlib import Path
import collections,hashlib,json,math
H=Path(__file__).resolve().parent;P=H.parents[1]
V161=P/'query_visibility_public_train_owner_counterfactual_20261024_v161'
V162=P/'query_visibility_counterfactual_frozen_features_20261024_v162'
def sha_text(p):
    p=Path(p);assert p.suffix in ('.json','.jsonl','.py');return hashlib.sha256(p.read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha_text(p))
def check(b):
    p=Path(b['path']).resolve();assert p.is_relative_to(V161) or p.is_relative_to(V162)
    assert p.suffix in ('.json','.jsonl','.py') and sha_text(p)==b['sha256'];return p
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def footprint(q,L):
    pixels=set()
    for j in (47,48):
        y=q[1]+((j+.5)/96-.5)*L;y0=math.floor(y);dy=y-y0
        for i in (47,48):
            x=q[0]+((i+.5)/96-.5)*L;x0=math.floor(x);dx=x-x0
            for ox,oy,w in ((0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)):
                if w>0:pixels.add((x0+ox,y0+oy))
    return sorted(pixels,key=lambda a:(a[1],a[0]))
def main():
    designpath=V161/'ROOT_SOURCE_GENERATION_DESIGN_BEFORE_PIXELS.json';design=read(designpath)
    pixels_path=V161/'source_pixels_v1/COMPLETE.json';pixels=read(pixels_path)
    feature_path=V162/'frozen_cf_features_v1/FEATURE_COMPLETE.json';feature=read(feature_path);assert feature['PASS']
    fitpath=V162/'controlled_fit_preparation/PROTOCOL_BEFORE_SOURCE_FIT_FEATURE_BODIES.json';fit=read(fitpath)
    admissionpath=V162/'controlled_fit_preparation/FINAL_OWNER_AUXILIARY_ADMISSION_COMPLETE.json';admission=read(admissionpath);assert admission['PASS']
    meta=read(check(feature['inputs']));manifest=lines(check(meta['manifest']));lineage=lines(check(feature['occurrence_lineage']))
    owners=lines(check(fit['CF_owner_sidecar']));proofrows=lines(check(pixels['proofs']));proofs={r['pair_id']:r for r in proofrows};assignments=lines(check(pixels['assignments']))
    builder=read(check(pixels['protocol']));check(builder['code'])
    mapping=read(V162/'controlled_fit_preparation/PROSPECTIVE_OWNER_MAPPING_COMPLETE.json');check(mapping['code']);check(fit['code'])
    feature_protocol=read(check(feature['protocol']));check(feature_protocol['code'])
    assert len(manifest)==len(owners)==len(lineage)==3816 and len(proofs)==len(assignments)==159
    assert len({r['input_occurrence_id'] for r in manifest})==3816 and sum(r['PASS'] for r in proofrows)==157
    fields=('input_occurrence_id','pair_id','variant','original_query_id','original_global_index','original_frame_index','source_id','split','native_slot','query_xy','L','query_available','feature_constructed','central4_available','pair_pixel_proof_PASS','selected_anchor_native_slot')
    mismatch=[];known=[];checks=collections.Counter()
    for i,(m,o,f) in enumerate(zip(manifest,owners,lineage)):
        assert o['cf_global_index']==f['feature_index']==i and f['payload_index']==i//8
        for k in fields:assert m[k]==o[k]==f[k],(i,k)
        assert all(f[k]==v for k,v in m.items())
        p=proofs[m['pair_id']];assert m['pair_pixel_proof_PASS']==p['PASS'] and m['split']=='SOURCE_DEVELOPMENT'
        eligible=p['PASS'] and m['variant'] in ('COVER','ADJACENT_SHAM') and m['native_slot']==m['selected_anchor_native_slot']
        target=(1 if m['variant']=='COVER' else 0) if eligible else -1
        assert o['owner_aux_target']==target and o['owner_auxiliary_admitted_under_root_design']==eligible
        assert o['native_visibility_target']==o['point_quality_target']==m['native_visibility_target']==m['point_quality_target']==-1
        assert m['variant_payload']==p['variants'][m['variant']]
        if eligible:
            assert m['query_available'] and m['feature_constructed'] and m['central4_available']
            assert m['original_query_id']==p['anchor_query_id'] and m['query_xy']==p['original_q'] and m['L']==p['original_L'] and m['native_slot']==p['native_slot']
            known.append(o);checks[m['variant']]+=1
    assert checks=={'COVER':157,'ADJACENT_SHAM':157}
    assert collections.Counter(r['owner_aux_target'] for r in owners)=={-1:3502,0:157,1:157}
    assert sum(not r['query_available'] for r in manifest)==360 and feature['inputU']==360 and feature['feature_shape']==[3816,1928]
    required_proof=('center4_footprint_inside','all_contributing_raw_cover_alpha1','all_contributing_raw_sham_alpha0','frozen_sampled_cover_alpha_center4_255','frozen_sampled_sham_alpha_center4_0','sham_center4_RGB_bit_equal_before','sham_some_selected_local_donor_support','cover_sham_equal_pasted_alpha_mass','positive_pasted_mass')
    for p in proofrows:
        assert footprint(p['original_q'],p['original_L'])==[tuple(x) for x in p['proof_detail']['raw_footprint_xy']]
        assert len(p['proof_detail']['raw_footprint_xy'])==p['proof_detail']['raw_footprint_count']
        if p['PASS']:assert all(p['proof'][k] for k in required_proof)
        assert p['proof_detail']['cover_pasted_alpha_mass']==p['proof_detail']['sham_pasted_alpha_mass']>0
        assert not p['replacement_or_retry'] and p['new_native_visibility_target']==p['new_point_quality_target']==[-1]*8
    assert {p['pair_id'] for p in proofrows if not p['PASS']}=={'V161_A01_R2','V161_A38_R2'}
    assert all(not p['proof']['sham_some_selected_local_donor_support'] for p in proofrows if not p['PASS'])
    anchor_ids={a['anchor_query_id'] for a in assignments};frames={a['frame_index'] for a in assignments};donors=collections.Counter(a['donor_id'] for a in assignments)
    assert len(anchor_ids)==53 and len(frames)==46 and len(donors)==59 and set(donors.values())=={2,3}
    natives={a['native_slot'] for a in assignments};assert natives=={0,1,4,5,6,7}
    assert collections.Counter(a['anchor_query_id'] for a in assignments).values() and all(z==3 for z in collections.Counter(a['anchor_query_id'] for a in assignments).values())
    bindings=[bind(designpath),bind(pixels_path),pixels['proofs'],pixels['assignments'],pixels['protocol'],builder['code'],bind(feature_path),feature['protocol'],feature_protocol['code'],feature['inputs'],meta['manifest'],feature['occurrence_lineage'],feature['payload_facts'],bind(fitpath),fit['code'],fit['CF_owner_sidecar'],bind(admissionpath),bind(V162/'controlled_fit_preparation/PROSPECTIVE_OWNER_MAPPING_COMPLETE.json'),mapping['code']]
    for b in bindings:check(b)
    result=dict(completed=True,PASS=True,review_kind='Existing public-source controlled MR+YCB CF metadata lineage, not new training admission or pixel/feature validation',code=bind(__file__),bindings=bindings,
        metadata_join=dict(rows=3816,unique_occurrence_IDs=3816,pairs=159,variants=477,known_owner314=dict(COVER_FOREIGN=157,ADJACENT_SHAM_TARGET=157,BG=0),UNKNOWN=3502,inputU=360,all_CF_native_visibility_quality_UNKNOWN=True,identity_fields_exact=list(fields),feature_index_equals_cf_global_index=True,manifest_feature_lineage_exact=True,failed_pairs_retained=['V161_A01_R2','V161_A38_R2']),
        footprint_metadata_check=dict(all159_recorded_raw_support_sets_recomputed_from_original_qL=True,central96_sample_indices='47:49 ×47:49',source_support='All positive float64 bilinear source-pixel coefficients of the four savedRGB96 center samples; not entire crop or physical native region',PASS157_all_required_proof_flags_true=True,same_transformed_donor_equal_positive_paste_mass=True,actual_alpha_RGB_validity_or_array_HASH_reverification_performed=False),
        authority=dict(COVER_FOREIGN='Existing source-proof metadata certifies hard overwrite by transformed YCB exclusive erode1 alpha on every contributing raw pixel of selected saved96 central4; sampled alpha center4=255. Controlled pasted donor visibility, not natural optical opacity or native E.',
            SHAM_TARGET='Existing original MR TARGET53 assistant-weak owner lineage plus zero transformed alpha on same central4 support and exact BEFORE/SHAM center4RGB equality. SHAM adds no independent dense instance/physical native gold; source weak TARGET authority is retained, not upgraded.',
            unknown_preservation='BEFORE duplicate rows, failed2 pairs and all nonanchor slots remainUNKNOWN. Existing V162 root-design-only owner admission is preserved historically, not reauthorized or reinterpreted for C49.',
            target_visibility_quality='No new native visibility/quality/coordinate labels; all−1. Point ownership, crop-presence and native region visibility remain separate.'),
        corpus_limits=dict(anchor_queries=53,anchor_frames=46,donors=59,donor_reuse_counts=dict(collections.Counter(donors.values())),present_anchor_native_slots=sorted(natives),missing_anchor_native_slots=[2,3],all8_feature_slots_preserved=True,independent_real_photos=False,source_split='SOURCE_DEVELOPMENT only; no new independent real validation',BEFORE_feature_parity='Existing FEATURE_COMPLETE records original46frames×8 bitexact,364actual encoder calls and113 guarded BEFORE reuses; not reexecuted here.'),
        existing_feature_packet=dict(binding_inherited_not_hashed_or_opened=feature['features'],shape=feature['feature_shape'],rows=3816,payloads=477,feature_constructed=3816,source_input_view='same17 RGB-only views, no alpha/owner/native/GT input',metadata_complete=bind(feature_path),independent_feature_audit=fit['CF_feature_audit']),
        C49_or_other_separate_owner_control_required_contract=dict(input_join_keys=['cf_global_index == feature_index','input_occurrence_id','pair_id','variant','original_query_id','original_global_index','original_frame_index','source_id','split','native_slot','query_xy','L','query_available','feature_constructed','central4_available','variant_payload binding'],
            feature_schema='features float32[3816,1928] + inherited original occurrence/native/qL/split/availability metadata; no new feature/weight body inspected here.',
            label_schema='owner_aux_target int −1/0/1/2 with currentknown0/1 only; classorder TARGET/FOREIGN/BG; native_visibility_target/point_quality_target remain−1; weak authority/proofPASS retained.',
            before_new_training=['Freeze a separate source-only control design, exact completed feature/admission/manifest SHA bindings and runtime packet metadata identity checks.','Declare owner loss meaning and original native/quality/entity separation; no automatic ownerF→nativeE or center4→wholecrop presence conversion.','Preserve source53/46 correlated groups, BEFORE duplication exclusion, two failures, all3816 inputU360 and UNKNOWN3502. Any evaluation split/criterion is a separate frozen design, not chosen here.','Standalone original YCB queries still need their own q/context/L, selected-instance/target-vs-foreign role, split/availability and exact frozen-feature join; no such standalone packet certified by this lineage.']),
        verified_vs_unconfirmed=dict(verified='Exact text-metadata row joins, order, source split/availability,314 rule,159 footprint coordinate enumeration,manifest/code/receipt hashes and source lineage.',unconfirmed=['No sourceRGB/alpha/mask/crop/feature bytes or NPZ headers reloaded; prior actual proof/audit authority inherited.','No new exact-q owner6 or dense/full96 ownership labels were generated from the old central4 proof.','No standalone YCB native/corner/coordinate visibility authority or independent real-data transfer success is established.','No existing admission is extended to a new C49 control; root must freeze its separate protocol.']),
        new_assets_downloaded=0,new_pixels_feature_weight_bodies_headers_or_hashes=0,model_encoder_fit_calls=0,current223_private_V112_or_project_search=0,new_labels_qL_thresholds_or_admission=False,selected_model=None,fully_solved=False)
    with (H/'REVIEW.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(PASS=True,review=bind(H/'REVIEW.json'),known314=checks)),flush=True)
if __name__=='__main__':main()
