"""Serialize reviewer B's actual image observations; no inference or recropping."""
from pathlib import Path
import collections
import datetime
import hashlib
import json

HERE = Path(__file__).resolve().parent
P = HERE.parent.parent
V92 = P / 'query_visibility_flat_pallet_cue_bridge_20261023_v92'

def binding(p):
    p = Path(p)
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}

def write(p, d):
    assert not p.exists(), p
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def obs(state, kind, note, bbox=None, unc=None, multi=False, appearance=None):
    d = dict(state=state, kind=kind, note=note, multi=multi)
    if bbox is not None:
        d.update(bbox=bbox, unc=unc)
    if appearance is not None:
        d['appearance'] = appearance
    return d

def main():
    final_working = {
      'reviewed_frames': list(range(14,20)),
      'receipts': 'All six original context figures and six exact stored96 sheets actually viewed with view_image. No source recropping.',
      'frames': {
       '14': {
        'scene': 'Small distant selected box lies between yellow rack protector and a blurred brown/red base; a unique pallet is not identifiable.',
        'ownership': 'AMBIGUOUS',
        'slots': {
         **{str(k): obs('NO','NONE_YELLOW_RACK_GUARD','Yellow steel protector dominates; no demonstrable pallet structural cue.') for k in [0,3,4,7]},
         **{str(k): obs('UNKNOWN','BLURRED_DISTANT_BASE','Blurred brown/red base and floor might contain pallet structure, but object ownership and exact cue cannot be established.') for k in [1,2,5,6]},
        }},
       '15': {
        'scene': 'Selected box spans a wooden pallet, steel rack column/black foot and another loaded pallet. Repeated supports cannot be assigned uniquely.',
        'ownership': 'AMBIGUOUS',
        'slots': {
         '0': obs('UNKNOWN','CANDIDATE_WOOD_FOOT_BY_RACK','A wood foot is visible beside the yellow/steel upright; selected-pallet identity remains ambiguous.',[14,39,45,84],6,True,'YES'),
         '1': obs('UNKNOWN','RACK_FOOT_AND_BACKGROUND_PALLET','Black rack base overlaps a wooden pallet behind; the intended support is ambiguous.',multi=True),
         '2': obs('NO','NONE_RACK_FOOT_FLOOR','Black steel rack foot and floor dominate; no distinct wooden pallet cue is localized.'),
         '3': obs('UNKNOWN','CANDIDATE_WOOD_FOOT_FLOOR','Wood foot/floor turn is visible at upper left beside the yellow protector, but ownership is not unique.',[5,10,43,49],6,True,'YES'),
         '4': obs('UNKNOWN','REPEATED_WOOD_SUPPORTS','Several wood blocks/opening rims appear across the selected box; no unique selected instance or cue.',multi=True),
         '5': obs('UNKNOWN','WOOD_BLOCK_BEHIND_STEEL_CROSSBAR','A wooden support below green cargo is partly behind black steel; its ownership by the selected target is unproved.',multi=True,appearance='YES'),
         '6': obs('UNKNOWN','CANDIDATE_WOOD_BLOCK_OPENING','Stamped wood block/opening edge is visible under a steel crossbar; which pallet is selected remains uncertain.',[27,48,68,83],6,True,'YES'),
         '7': obs('UNKNOWN','WOOD_FOOT_FLOOR_WITH_OTHER_SUPPORTS','Wood/floor geometry is present, but several adjacent pallet supports compete.',multi=True),
        }},
       '16': {
        'scene': 'Selected box is beneath a red-sided wheeled cart next to another white cart, not an identifiable pallet.',
        'ownership': 'NOT_PALLET',
        'slots': {str(k): obs('NO','NONE_WHEELED_CART_OR_FLOOR','Cart body/frame, wheel/base shadows, adjacent white cart or floor; no demonstrable pallet structure.') for k in range(8)}},
       '17': {
        'scene': 'Box spans several gray plastic pallet-like bases below red bins along the rack. Concrete openings and supports are visible, but selected single-instance ownership is ambiguous.',
        'ownership': 'AMBIGUOUS',
        'slots': {
         '0': obs('UNKNOWN','TRUNCATED_PLASTIC_BASE_JUNCTION','Base rim/support junction lies at the lower crop boundary; repeated adjacent bases make target association uncertain.',multi=True,appearance='YES'),
         '1': obs('UNKNOWN','PLASTIC_SUPPORT_OPENING_JUNCTION','Gray support joins a rim and two dark openings, but the selected pallet among adjacent bases is uncertain.',[35,41,79,89],5,True,'YES'),
         '2': obs('UNKNOWN','PLASTIC_FOOT_FLOOR_TURN','Gray foot and opening meet floor in upper crop; selected-instance ownership remains ambiguous.',[26,1,70,39],6,True,'YES'),
         '3': obs('UNKNOWN','PLASTIC_SUPPORT_RIM_JUNCTION','Large gray support/rim junction is visible, with multiple plausible bases in the box.',[13,25,64,87],5,True,'YES'),
         '4': obs('UNKNOWN','PLASTIC_SUPPORT_AND_OPENINGS','Support separates two openings below red cargo; multiple base instances prevent unique binding.',[33,48,74,94],5,True,'YES'),
         '5': obs('UNKNOWN','REPEATED_BASES_BEHIND_YELLOW_GUARD','Several gray openings and feet continue behind a yellow guard; no unique selected cue.',multi=True),
         '6': obs('UNKNOWN','TRUNCATED_BASES_GUARD_FLOOR','Gray base is confined to the upper edge alongside yellow guard and floor; ownership uncertain.',multi=True),
         '7': obs('UNKNOWN','PLASTIC_SUPPORT_OPENING_JUNCTION','Gray support/opening junction is clear but one selected pallet cannot be isolated from the row.',[27,22,74,70],5,True,'YES'),
        }},
       '18': {
        'scene': 'Large gray plastic pallet stack is left of the selected box. The box instead spans a small gray base behind a yellow guard and red bins; ownership is ambiguous.',
        'ownership': 'AMBIGUOUS',
        'slots': {
         '0': obs('UNKNOWN','CANDIDATE_PLASTIC_DECK_FOOT_JUNCTION','Gray molded deck/foot junction appears behind yellow guard, but intended pallet instance is not established.',[25,28,61,76],6,True,'YES'),
         '1': obs('NO','NONE_RED_BIN','Red plastic cargo bin side/handles; no pallet structure.'),
         '2': obs('NO','NONE_RED_BIN','Red plastic cargo bin side/label; no pallet structure.'),
         '3': obs('UNKNOWN','CANDIDATE_PLASTIC_FOOT_FLOOR','Gray foot/floor silhouette is visible behind guard, with ambiguous selected-base ownership.',[9,24,52,58],6,False,'YES'),
         '4': obs('UNKNOWN','PARTIAL_PLASTIC_DECK_OR_CONTAINER','Gray top/rib boundary could be pallet deck but identity is uncertain; not sufficient for a definitive support box.'),
         '5': obs('NO','NONE_YELLOW_GUARD_RED_BIN','Yellow rack guard and red bin dominate.'),
         '6': obs('NO','NONE_YELLOW_GUARD_RED_BIN','Yellow rack guard and red bin dominate.'),
         '7': obs('UNKNOWN','CANDIDATE_PLASTIC_SUPPORT_OPENING','Molded vertical support/opening and floor appear, but selected target is not unique.',[34,28,72,81],6,True,'YES'),
        }},
       '19': {
        'scene': 'Selected box spans multiple receding loaded wooden pallets along the left rack; a unique intended pallet is not supported.',
        'ownership': 'AMBIGUOUS',
        'slots': {
         '0': obs('NO','NONE_CARD_BOARD','Cardboard cargo face with black out-of-frame padding, not a localized pallet structure.'),
         '1': obs('UNKNOWN','WOOD_SUPPORT_FORK_OPENING','Wood blocks and fork-opening rim are clear, but the selected pallet within the row is ambiguous.',[26,43,74,88],6,True,'YES'),
         '2': obs('UNKNOWN','TRUNCATED_SUPPORT_FLOOR_EDGE','Pallet support/opening lies at the upper crop edge; floor dominates and target ownership is ambiguous.',multi=True),
         '3': obs('NO','NONE_CARD_BOARD_PARTIAL_EDGE','Cardboard cargo and a truncated straight bottom strip, insufficient distinct pallet structure.'),
         '4': obs('UNKNOWN','CARGO_EDGE_PARTIAL_WOOD_BASE','Wrapped cargo edge and partial wood base at bottom; no uniquely localized selected cue.',multi=True),
         '5': obs('UNKNOWN','REPEATED_BACKGROUND_PALLET_FEET','Several small pallet feet and yellow posts recede along the rack; ownership/cue identity ambiguous.',multi=True),
         '6': obs('UNKNOWN','REPEATED_BACKGROUND_PALLET_FEET','Same receding row of pallet feet and floor; no unique selected instance.',multi=True),
         '7': obs('UNKNOWN','WOOD_SUPPORT_OPENING_JUNCTION','Wood support and deck/opening junction are visible below wrapped cargo, with multiple candidate pallets in context.',[27,42,73,89],6,True,'YES'),
        }},
      }}
    write(HERE/'OBSERVATION_WORKING_14_19.json',final_working)
    frames={}
    working=sorted(HERE.glob('OBSERVATION_WORKING_*.json'))
    for p in working:
        d=json.loads(p.read_text())
        for k,v in d['frames'].items():
            assert int(k) not in frames
            frames[int(k)]=v
    assert sorted(frames)==list(range(20))
    source_rows=V92/'rgb96_review/ROWS_INPUT_ONLY.jsonl'
    rows=[json.loads(x) for x in source_rows.read_text().splitlines()]
    assert len(rows)==160
    manifest=json.loads((HERE/'DERIVATIVE_SHEETS_MANIFEST.json').read_text())
    sheets={x['frame_index']:x for x in manifest['frames']}
    output=[]
    frame_receipts=[]
    for f in range(20):
        context=V92/f'rgb96_review/figures/frame_{f:02d}.png'
        frame_receipts.append({
            'frame_index':f,'context_actual_viewed':True,'context':binding(context),
            'exact96_sheet_actual_viewed':True,'exact96_sheet':binding(sheets[f]['path']),
            'all_8_saved_crops_reviewed':True,'scene_observation':frames[f]['scene'],
            'target_ownership_confidence':frames[f].get('ownership','UNKNOWN_NO_VALID_QUERY'),
            'view_method':'tools.view_image actual image result; full context and exact96 nearest enlargement',
        })
    for i,r in enumerate(rows):
        f,k=r['frame_index'],r['native_slot']
        assert i==f*8+k
        frame=frames[f]
        invalid=not r['query_center_valid']
        if invalid:
            kind='NO_QUERY' if not r['query_exists'] else 'OOB_OR_INVALID_QUERY'
            o=obs('UNKNOWN',kind,'Original factual U retained; no replacement query, support box, or negative training label.')
        else:
            o=frame.get('slots',{}).get(str(k),frame.get('valid_default'))
            assert o is not None,(f,k)
        bbox=o.get('bbox')
        if bbox is not None:
            assert len(bbox)==4 and 0<=bbox[0]<bbox[2]<=96 and 0<=bbox[1]<bbox[3]<=96
            assert o['state']!='NO' and 0<o['unc']<=16
        if o['state']=='YES':
            assert frame.get('ownership')=='HIGH' and not invalid and bbox is not None
        centre=None if bbox is None else [(bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2]
        rec={
            'row_index':i,'frame_index':f,'input_id':r['input_id'],'native_slot':k,'split':r['split'],
            'image_sha256':r['image_sha256'],'original_q':r['query_xy'],'original_L':r['L'],
            'source_validity':{x:r[x] for x in ['query_exists','query_center_valid','continuous_valid','nearest_valid','crop_anyvalid','crop_valid_fraction','input_status']},
            'target_ownership_confidence':frame.get('ownership','UNKNOWN_NO_VALID_QUERY') if not invalid else 'UNKNOWN_NO_VALID_QUERY',
            'visible_aux_structuralcue':o['state'],
            'appearance_structuralcue_observed':o.get('appearance','YES' if o['state']=='YES' else ('NO' if o['state']=='NO' else 'UNKNOWN')),
            'cue_kind':o['kind'],'crop_support_bbox_xyxy96_or_null':bbox,
            'support_uncertainty_px':o.get('unc'),
            'bbox_interpretation':'Approximate visible structural support extent, half-open crop coordinates; uncertainty is weak AI spatial imprecision, not a calibrated pixel certificate.' if bbox else None,
            'bbox_scope':'SELECTED_PALLET_AUX_SUPPORT' if o['state']=='YES' else ('OBSERVED_STRUCTURE_CANDIDATE_OWNERSHIP_UNKNOWN' if bbox else None),
            'crop_bbox_center_xy96':centre,
            'bbox_center_minus_original_q_crop_px':None if centre is None else [v-47.5 for v in centre],
            'multiplecue_flag':o.get('multi',False),'notes':o['note'],
            'native_expected_part_state':'UNKNOWN','native8_semantic_mapping':'UNKNOWN',
            'expected_cue_binding':'UNKNOWN','native_human_gold':False,'training_admitted':False,
            'absence_is_semantic_E_or_S':False,'original_query_changed':False,
            'original_context_figure':str(V92/f'rgb96_review/figures/frame_{f:02d}.png'),
            'exact96_review_sheet':str(HERE/f'exact96_review_sheets/frame_{f:02d}_EXACT96.png'),
            'exact_source_RGB96_channel_bytes_SHA256':sheets[f]['exact_source_RGB96_channel_bytes_SHA256'][k],
        }
        output.append(rec)
    outpath=HERE/'ALL160_AUXILIARY_CUE_OBSERVATIONS.jsonl'
    assert not outpath.exists()
    outpath.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in output))
    counts=lambda rr:dict(collections.Counter(r['visible_aux_structuralcue'] for r in rr))
    actualvalid=[r for r in output if r['source_validity']['query_center_valid']]
    yes=[r for r in output if r['visible_aux_structuralcue']=='YES']
    assert len(actualvalid)==122
    assert sum(not r['source_validity']['query_exists'] for r in output)==16
    assert sum(r['source_validity']['query_exists'] and not r['source_validity']['query_center_valid'] for r in output)==22
    assert all(r['native_expected_part_state']=='UNKNOWN' and not r['training_admitted'] for r in output)
    # Byte-to-byte original coordinate metadata equality, including missing values represented as null.
    assert all(a['original_q']==r['query_xy'] and a['original_L']==r['L'] and a['input_id']==r['input_id'] and a['native_slot']==r['native_slot'] for a,r in zip(output,rows))
    write(HERE/'ACTUAL_VIEW_RECEIPTS.json',{
        'completed':True,'actual_context_figures_viewed':20,'actual_exact96_sheets_viewed':20,
        'actual_image_view_count':40,'valid_RGB96_crops_inspected':122,'factual_U_placeholders_inspected':38,
        'all_frames':frame_receipts,'derivatives':binding(HERE/'DERIVATIVE_SHEETS_MANIFEST.json'),
        'no_source_recrop_or_recenter':True,'new_model_fit_current_PnP':0,
    })
    write(HERE/'REVIEW.json',{
        'completed':True,'PASS':True,'reviewer':'B independent AI visual observation',
        'PASS_scope':'Full requested visual coverage, serialization and preservation checks only. Not gold annotation, native correspondence, training admission, or task success.',
        'policy':binding(HERE/'POLICY_BEFORE_ACTUAL_VIEWS.json'),
        'root_scope':binding(HERE.parent/'ROOT_PUBLIC_CUE_REVIEW_SCOPE_BEFORE_ACTUAL_CROPS.json'),
        'source_rows':binding(source_rows),'original_RGB96_seal':binding(V92/'rgb96_review/RGB96_REVIEW_SEAL.json'),
        'observations':binding(outpath),'view_receipts':binding(HERE/'ACTUAL_VIEW_RECEIPTS.json'),
        'counts':{
            'original_frames':20,'all_original_rows':160,'valid':122,'no_query_U':16,'OOB_or_invalid_U':22,
            'all160_auxiliary_appearance_states':counts(output),'valid122_auxiliary_appearance_states':counts(actualvalid),
            'by_split':{s:{'all_rows':sum(r['split']==s for r in output),'all_states':counts([r for r in output if r['split']==s]),'valid_states':counts([r for r in actualvalid if r['split']==s])} for s in sorted({r['split'] for r in output})},
            'positive_frames':sorted({r['frame_index'] for r in yes}),
            'positive_rows':len(yes),'positive_with_bbox':sum(r['crop_support_bbox_xyxy96_or_null'] is not None for r in yes),
            'UNKNOWN_candidate_bbox_rows':sum(r['visible_aux_structuralcue']=='UNKNOWN' and r['crop_support_bbox_xyxy96_or_null'] is not None for r in output),
            'native_expected_state_UNKNOWN':160,'human_gold':0,'training_admitted':0,
        },
        'critical_interpretation':[
            'YES means a visible 2D pallet structure with supported selected-instance ownership and weak bounding extent in the original crop. It does not identify the expected native part or prove semantic D.',
            'UNKNOWN can still contain clearly visible pallet-like geometry: repeated instances, rack overlap, ambiguous base identity or blur prevent assigning that geometry to the intended target. Candidate bboxes are not positives.',
            'NO is a local appearance observation of cardboard, bins, rack, wheeled cart, floor or insufficient distinct pallet structure. It is never an E/S training negative.',
            'Original V92 coarse34 are proposals only. B does not inherit them as positives or rewrite any parent record. In frame7/native1 (old focused case2), cardboard appearance remains explicit and no pallet auxiliary cue is asserted.',
            'Uncertainty denotes weak AI localization imprecision around a support rectangle; multiple candidate cues may coexist. No coordinate is snapped or recentered, and query-centre membership is not required for visible support elsewhere in the crop.',
            'All 38 no-query/OOB rows remain in the denominator as factual UNKNOWN, with null cue boxes. No cue is invented from the full-scene view for an unavailable query crop.',
        ],
        'independence_and_limits':[
            'Reviewer C new labels were not read. Previous V92 observations, combined metadata and parent case2 caution were already exposed; this is not blinded or independent human gold.',
            '20 source contexts were saved rendering aids; exact crops were independently viewed as 4x nearest displays of unchanged stored96. No new initializer call, raw-source crop, geometry, model, weight, private/current or fit operation.',
            'Selected box and material are not visibility evidence by themselves. Every native expectedpart/native8 mapping remains UNKNOWN. Auxiliary evidence cannot resolve hidden reference identity or guarantee learnability from the candidate three tensors.',
            'Same source scenes and camera families are reused development data. Split labels are retained without a claim of independently captured validation generalization.',
        ],
        'next_admissible_use':'Root may compare independently sealed annotations and inspect disagreements. No training is admitted here; same original-q auxiliary positives would require a separate evidence/admission and controlled-cover/sham protocol.',
        'invariants':{'q_L_ID_order_exact':True,'all160_preserved':True,'new_crop_recenter_query':0,'other_reviewer_new_labels_read':0,'new_model_forward_fit_private_current_PnP':0},
        'created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    })
    bindings=[binding(x) for x in working+[HERE/'OBSERVATION_WORKING_14_19.json'] if x.exists()]
    # Glob already contains the final working file; retain each path once in the seal.
    bindings=list({x['path']:x for x in bindings}.values())
    write(HERE/'REVIEW_SEAL.json',{
        'completed':True,'PASS':True,'scope':'Actual saved public source visual annotation B only; no training/native/human gold admission.',
        'review':binding(HERE/'REVIEW.json'),'observations':binding(outpath),
        'receipts':binding(HERE/'ACTUAL_VIEW_RECEIPTS.json'),
        'policy':binding(HERE/'POLICY_BEFORE_ACTUAL_VIEWS.json'),'serializer':binding(Path(__file__)),
        'working_observations':bindings,'derivatives':binding(HERE/'DERIVATIVE_SHEETS_MANIFEST.json'),
    })
    print(json.dumps({'review':binding(HERE/'REVIEW.json'),'seal':binding(HERE/'REVIEW_SEAL.json'),'counts':json.loads((HERE/'REVIEW.json').read_text())['counts']},ensure_ascii=False))

if __name__=='__main__': main()
