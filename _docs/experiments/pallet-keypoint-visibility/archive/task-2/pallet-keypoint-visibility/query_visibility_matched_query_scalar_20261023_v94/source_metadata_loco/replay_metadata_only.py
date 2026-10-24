"""Immutable public-query/weak-cue metadata audit. No image, NPZ, model or current reader."""
from pathlib import Path
import collections, datetime, hashlib, json, math

ROOT=Path(__file__).resolve().parent
V94=ROOT.parent
P=V94.parent
V92=P/'query_visibility_flat_pallet_cue_bridge_20261023_v92'

def bind(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(s) for s in p.read_text().splitlines()]
def save(p,x):
    assert not p.exists();p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def valid(q,h,w):
    if q is None:return False,False
    finite=all(math.isfinite(a) for a in q) and q!=[-1,-1]
    continuous=finite and 0<=q[0]<w and 0<=q[1]<h
    nearest=finite and 0<=math.floor(q[0]+.5)<w and 0<=math.floor(q[1]+.5)<h
    return continuous,nearest

def main():
    inventory=read(V92/'EXACT_SOURCE_PATHS_AND_NATIVE_QUERY_FEASIBILITY.json')
    names={
      'scope':V94/'SCOPE_BEFORE_MATCHED_QUERY_CONTRACT.json',
      'prior_inventory':V92/'EXACT_SOURCE_PATHS_AND_NATIVE_QUERY_FEASIBILITY.json',
      'selected_inputs':V92/'rgb96_preparation/FIXED20_SELECTED_QUERY_INPUTS_ONLY.jsonl',
      'slot_inputs':V92/'rgb96_review/ROWS_INPUT_ONLY.jsonl',
      'input_seal':V92/'rgb96_review/RGB96_REVIEW_SEAL.json',
      'input_producer_protocol':V92/'RGB96_PRODUCER_PROTOCOL_BEFORE_PIXELS.json',
      'input_producer_code':V92/'rgb96_preparation/prepare_fixed20_rgb96.py',
      'weak_rows':V92/'combined_visual_review/ALL160_WEAK_VISUAL_OBSERVATIONS.jsonl',
      'weak_summary':V92/'combined_visual_review/COMBINED20_SOURCE_QUERY_VISUAL_REVIEW.json',
      'weak_seal':V92/'combined_visual_review/SEAL.json',
      'fixed6_selection':V92/'independent_qwen3_saved_output_audit/FIXED6_QUALITATIVE_SELECTION_BEFORE_OUTPUTS.json',
      'root_focused_visual_addendum':V92/'qwen3_saved_visual_diagnosis/ROOT_ALL6_ACTUAL_VIEW_OBSERVATIONS_KO.json',
      'V92_phase_seal':V92/'ROOT_PHASE_RESULT_SEAL.json'}
    for k in ['saved_query_proposals','saved_query_producer','saved_query_protocol','prior_physical_cue_metadata','source_group_freeze','official_split_evidence']:
        d=inventory['exact_source_paths'][k];p=Path(d['path'])
        assert bind(p)['sha256']==d['sha256'];names[k]=p
    frames=lines(names['selected_inputs']);rows=lines(names['slot_inputs']);observations=lines(names['weak_rows'])
    proposals=lines(names['saved_query_proposals']);cues=lines(names['prior_physical_cue_metadata'])
    assert len(frames)==len(proposals)==20 and len(rows)==len(observations)==160 and len(cues)==22
    candidate_count=0;frame_summaries=[]
    for i,(f,p) in enumerate(zip(frames,proposals)):
        assert f['frame_index']==i and f['input_id']==p['input_id'] and f['image_sha256']==p['image_sha256']
        expected_split='TRAIN' if f['source_subset'] in [2,3,5] else 'VALIDATION'
        assert f['source_subset'] in [1,2,3,4,5] and f['split']==expected_split
        candidates=p['candidates'];candidate_count+=len(candidates)
        if candidates:
            chosen=max(range(len(candidates)),key=lambda k:candidates[k]['score'])
            assert chosen==p['selected_index']==f['selected_index']
            c=candidates[chosen];assert c['keypoints_xy'][:8]==f['query8_xy'] and c['box_xyxy']==f['selected_predictor_box_xyxy']
            good=[q for q in f['query8_xy'] if all(math.isfinite(a) for a in q) and q!=[-1,-1]]
            span=max(max(q[d] for q in good)-min(q[d] for q in good) for d in [0,1]) if len(good)>=2 else 0
            expected_L=min(max(.25*span,16),max(f['raw_hw']))
            assert f['L']==expected_L
        else:assert p['selected_index'] is f['selected_index'] is f['query8_xy'] is f['L'] is None
        frame_summaries.append(dict(frame_index=i,input_id=f['input_id'],source_subset=f['source_subset'],split=f['split'],image_sha256=f['image_sha256'],candidate_count=len(candidates),selected_index=f['selected_index'],L=f['L'],query8=f['query8_xy'],old_initializer_group_state=f['old_initializer_group_state']))
    assert candidate_count==120
    diagnostics=[]
    for i,(row,obs) in enumerate(zip(rows,observations)):
        fi,k=divmod(i,8);f=frames[fi]
        assert row['frame_index']==obs['frame_index']==fi and row['native_slot']==obs['native_slot']==k and obs['row_index']==i
        for key in ['input_id','image_sha256','query_xy','L','query_exists','query_center_valid','continuous_valid','nearest_valid','crop_valid_fraction','input_status']:assert row[key]==obs[key]
        assert row['split']==obs['source_split']==f['split']
        q=None if f['query8_xy'] is None else f['query8_xy'][k]
        assert row['query_xy']==q and row['L']==f['L'] and row['query_exists']==(q is not None)
        c,n=valid(q,*f['raw_hw']);assert row['continuous_valid']==c and row['nearest_valid']==n and row['query_center_valid']==(c and n)
        assert obs['expected_native_part_identity']==obs['semantic_D_E_S']=='UNKNOWN'
        assert obs['usable_training_gold'] is False and obs['new_training_label'] is False
        diagnostics.append(dict(row_index=i,frame_index=fi,input_id=f['input_id'],native_slot=k,split=f['split'],image_sha256=f['image_sha256'],query_xy=q,L=f['L'],input_status=row['input_status'],query_center_valid=row['query_center_valid'],weak_local_observation_original=obs['query_local_evidence'],weak_target_association_original=obs['target_instance_visual_association'],verified_specific_cue_query_binding=False,known_source_scalar_label=False,metadata_status='EXISTING_UNLABELLED_METADATA_ONLY',known_focused_case2_caution=(i==57)))
    split_counts={}
    for split in ['TRAIN','VALIDATION']:
        selected=[x for x in rows if x['split']==split]
        o=[x for x in observations if x['source_split']==split]
        split_counts[split]=dict(frames=sum(f['split']==split for f in frames),slots=len(selected),valid=sum(x['query_center_valid'] for x in selected),no_query=sum(not x['query_exists'] for x in selected),invalid_query=sum(x['query_exists'] and not x['query_center_valid'] for x in selected),crop_anyvalid=sum(x['crop_anyvalid'] for x in selected),weak_local_categories=dict(collections.Counter(x['query_local_evidence'] for x in o)))
    cue_geometry=[]
    for cue in cues:
        assert cue['native8_mapping']=='UNKNOWN' and not cue['human_gold'] and not cue['admitted_training_teacher'] and cue['training_rows_created']==0
        assert cue['initial_query_xy'] is None and cue['initializer_detection_or_raw_index_used'] is None
        for f in frames:
            if cue['image_sha256']!=f['image_sha256']:continue
            for row in rows[f['frame_index']*8:(f['frame_index']+1)*8]:
                if not row['query_exists']:state='NO_QUERY'
                else:
                    q,L=row['query_xy'],row['L'];box=[q[0]-L/2,q[1]-L/2,q[0]+L/2,q[1]+L/2];u=cue['uncertainty_bbox_half_open_xyxy']
                    full=u[0]>=box[0] and u[1]>=box[1] and u[2]<=box[2] and u[3]<=box[3]
                    overlaps=min(box[2],u[2])>max(box[0],u[0]) and min(box[3],u[3])>max(box[1],u[1])
                    state='FULL' if full else ('PARTIAL' if overlaps else 'MISS')
                cue_geometry.append(dict(cue_refID=cue['cue_refID'],frame_index=f['frame_index'],native_slot=row['native_slot'],split=f['split'],geometry_only_availability=state,target_correspondence_to_prediction='UNVERIFIED',not_a_new_label=True))
    summary=read(names['weak_summary'])
    assert dict(collections.Counter(x['query_local_evidence'] for x in observations))==summary['counts']['query_local_evidence']
    assert len({x['cue_refID'] for x in cue_geometry})==10 and len({x['frame_index'] for x in cue_geometry})==5
    assert dict(collections.Counter(x['geometry_only_availability'] for x in cue_geometry))==dict(NO_QUERY=16,MISS=64)
    assert observations[57]['query_local_evidence']=='VISIBLE_DISTINCT_CORNER_EDGE_FOOT'
    fixed=read(names['fixed6_selection'])['fixed6'];assert fixed[1]['slot_index']==57
    addendum=read(names['root_focused_visual_addendum']);assert addendum['primary_predictions_or_weak_observations_modified'] is False
    rowpath=ROOT/'ALL160_METADATA_READINESS.jsonl';assert not rowpath.exists();rowpath.write_text(''.join(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n' for x in diagnostics))
    save(ROOT/'PRIOR_CUE_VS_ORIGINAL_QUERY_GEOMETRY_ONLY.json',dict(completed=True,not_new_annotation_or_training_labels=True,counts=dict(collections.Counter(x['geometry_only_availability'] for x in cue_geometry)),rows=cue_geometry))
    report=dict(completed=True,PASS=True,metadata_audit_PASS=True,training_ready=False,status='METADATA_READY_FOR_PROSPECTIVE_SOURCE_ANNOTATION',code=bind(Path(__file__)),bindings={k:bind(p) for k,p in names.items()},
      available_saved_inputs=dict(frames=20,all_candidates=120,detected_frames=18,no_detection_frames=2,native_slots=160,predicted_slots=144,valid=122,OOB_or_invalid=22,no_query=16,split_counts=split_counts,L_min=min(f['L'] for f in frames if f['L'] is not None),L_max=max(f['L'] for f in frames if f['L'] is not None),all20_first_argmax_q8_L_order_exact=True,unmodified_q_native_order='Original selected prediction rows0..7, centroid row8 excluded; original padding already removed, no GT target selection/clip/recenter/permutation.'),
      coarse_observations=dict(original_categories=summary['counts']['query_local_evidence'],frame_target_association=summary['counts']['target_instance_visual_association'],specific_reference_r_unc_support_in_all160_observation_schema=False,verified_query_expectedcue_binding=0,native_or_human_semantic_gold=0,independent_human_consensus=False),
      previous_specific_cue_proposals=dict(total_proposals=22,images=12,TRAIN=11,VALIDATION=11,overlap_fixed20_cues=10,overlap_fixed20_frames=5,on_nodetection_cues=2,all_native_mapping_UNKNOWN=True,admitted_teacher=0,actual_selected_query_associations=0,geometric_cue_slot_pairs=80,geometry_counts=dict(NO_QUERY=16,MISS=64,FULL=0,PARTIAL=0),meaning='Existing r/unc belong to separately selected official representative-bbox targets; same RGB does not prove same selected prediction target. Every available original q crop misses these uncertainty boxes. This is a geometric diagnostic, not relabelling or a reason to move q/enlarge L.'),
      known_issue=dict(slot_index=57,frame_index=7,native_slot=1,input_id=frames[7]['input_id'],original_coarse_category=observations[57]['query_local_evidence'],root_focused_case2='Cardboard dominates the focused crop and the prior coarse distinctcue observation became unreliable. Original record/count34 preserved; separate caution only, no silent new label.',source=bind(names['root_focused_visual_addendum'])),
      split_scope=dict(TRAIN_subsets=[2,3,5],VALIDATION_subsets=[1,4],fixed20_counts='12TRAIN/8VAL',capture_family='Author environment grouping, not independent-site certification. Original UNASSIGNED_GROUP_HOLD lineage remains unchanged; later official/group metadata is separate prospective partition evidence.',source_development_exposure='These fixed20, weak observations and model probes have already been examined during development; no fresh source generalization claim.'),
      availability_limits=dict(RGB96_and_validity_pack='Existing sealed V92 input pack available, not reopened here.',P3='No compatible V94 P3 input pack is bound by these V92 RGB96 artifacts; hooks/query metadata are not a certified feature tensor cache. Locate/bind or separately authorize feature production before a P3 model fit.',official_LOCO_bbox='Object bbox annotations are not pixel masks, physicalcue visibility or native8 gold.',full5593_actual_queries='Only canonical fixed20 actual saved outputs were located by the bound prior inventory; no claim of full5593 or extension40 query coverage.'),
      minimum_next_source_only_work=[
       'Freeze a source-only inspection/annotation policy keyed by these original160slots and official splits. Inspect unmarked actual RGB96 at originalq/L plus full scene for target association; keep no-query16 and invalid22 as explicit U. Prioritize neither model scores nor an assumed positive label from the coarse34.',
       'For each valid slot, record whether exactly one intended structural cue can be associated with the selected target and original crop. Seal a named cue_refID, target/part-line description, r/uncertainty and essential support extent as GT-only metadata. Multiple plausible cue/target choices, material-only, wrong target or missing expected reference stay typed U.',
       'Auxiliary physical-cue training does not require native8 mapping for every example. A separate conditional-AUX label tier with unknown native correspondence is legitimate when target/cue identity is identifiable from the permitted inputs. It must not replace original nativesemantic evaluation.',
       'Before any controlled generation, freeze sameq/L before/fullopaque-cover/cue-preserving-sham construction and all failures. Both uncertainty and essential structural support must be covered/preserved as appropriate; donor/family split and repeated-cue grouping stay fixed. No naturalE from a foreign/background crop, no maskpixel or woodmaterial as expectedcue gold.',
       'Audit identical permitted input with incompatible hidden-reference labels. Annotation cannot silently choose one unseen reference among indistinguishable alternatives. Leave such casesU or declare a separate inference-available identity input contract; do not add GTcueID/r/mask at inference.',
       'Train/sourceVAL/deployment read the same original-q-centred single scalar. r/uncertainty may establish GT eligibility and diagnostics only; no r-sampled map/mosaic/besttile readout, q re-centering or L expansion. One sealed before/cover/sham can test operator mechanics only; this metadata audit sets no training budget/threshold/selector/gates.'
      ],
      output_metadata=dict(all160=bind(rowpath),prior_cue_geometry=bind(ROOT/'PRIOR_CUE_VS_ORIGINAL_QUERY_GEOMETRY_ONLY.json')),
      limits=['All findings are metadata replay and previously sealed AI observations. No new source image inspection, model-derived label or training annotation was created.','Geometric FULL would still not prove intended-target or cue ownership; here existing proposals have no FULL candidates anyway.','There are no verified same-reference natural external/self labels in this LOCO20 evidence, and no auxiliary/native/human gold admission. Absence of a matched cue is an annotation/readiness gap, not proof that the broader task is unlearnable.'],
      operations=dict(raw_RGB_NPY_NPZ_features_weights_models_current_reads=0,new_forward_fit_PnP=0,new_training_labels=0,V92_V93_mutations=0),created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(ROOT/'REVIEW.json',report)
    save(ROOT/'SEAL.json',dict(completed=True,PASS=True,review=bind(ROOT/'REVIEW.json'),metadata_rows=report['output_metadata']))
    print(json.dumps(dict(PASS=True,review=bind(ROOT/'REVIEW.json'),training_ready=False,split_counts=split_counts,geometry_counts=report['previous_specific_cue_proposals']['geometry_counts']),ensure_ascii=False))

if __name__=='__main__':main()
