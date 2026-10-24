"""Freeze RGB-only operational reference proposals for extension indices 20..39."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
R=Path(__file__).resolve().parent
E=R.parent

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p): return {'path':str(Path(p).resolve()),'sha256':sha(p)}

# Chosen from actual native RGB crops and coordinate-grid views only.
# These are observed physical edge turns/endpoints, not inferred cuboid corners.
PROPOSALS={
23:[([268,868],[261,861,276,876],'LOWER_NEAR_WOOD_SUPPORT_EDGE_TURN',
     'The visible near/front wooden block has a vertical edge that meets its lower outer contour above the floor. The point belongs to the wood base beneath the red containers, not to a container or the yellow rack.',
     'The wood/floor contour is soft and the block is worn; the uncertainty spans the visible edge transition.'),
    ([547,844],[538,836,556,853],'LOWER_IMAGE_RIGHT_WOOD_SUPPORT_END',
     'The exposed image-right wooden support ends against the floor beneath the same loaded low pallet. Its lower outer turn is visible beyond the right end of the front opening.',
     'Perspective compresses the far support and the edge is blurred; the range is wider than a single material pixel.')],
28:[([1503,679],[1496,672,1512,687],'LOWER_IMAGE_LEFT_WOOD_SUPPORT_END',
     'The exposed image-left wood support below the white cargo meets the floor at a visible lower edge turn. The nearby dark rectangular opening and connected deck establish association to the selected low pallet.',
     'Shadow and rounded worn wood soften the support/floor corner. The yellow guard is separate and does not define this point.'),
    ([1668,750],[1659,741,1678,759],'LOWER_NEAR_WOOD_SUPPORT_EDGE_TURN',
     'The near/front wood block has a visible lower-left turn where its vertical face meets the sloping lower contour. It is left of the yellow guard and belongs to the same continuous pallet deck.',
     'The block/floor transition is shaded and slightly blurred. The obscured far-right pallet end is not reconstructed.')],
30:[([1064,878],[1053,868,1076,889],'LOWER_IMAGE_RIGHT_WOOD_SUPPORT_END',
     'An exposed wooden support beneath the brown carton load terminates against the floor at the image-right side of the selected base. The dark opening and connected deck distinguish it from a carton corner.',
     'Motion blur makes the physical endpoint broad. The near support touches the image bottom and is deliberately not given a reference.')],
31:[([510,771],[503,763,518,780],'LOWER_NEAR_WOOD_SUPPORT_EDGE_TURN',
     'The near wooden support below the black carton load has a visible right vertical edge ending at the floor contour. This is a wood-base edge, not a black-carton edge.',
     'The bright wood/floor boundary is soft and worn. The reference range remains inside the source image.'),
    ([577,728],[570,720,585,737],'LOWER_IMAGE_RIGHT_WOOD_SUPPORT_EDGE_TURN',
     'The farther exposed wood support has a lower-left contour turn beneath the connected deck. The visible opening separates it from the nearer wood block on the same pallet.',
     'Perspective and slight blur make the endpoint uncertain. No ideal cuboid vertex or unseen end is inferred.')],
37:[([738,548],[730,540,747,557],'LOWER_IMAGE_LEFT_PLASTIC_SUPPORT_EDGE_TURN',
     'The image-left gray plastic base beneath the red containers has an exposed vertical edge that turns into the lower sloping contour. The gray support and its opening are distinct from the red cargo.',
     'The molded plastic is rounded and low contrast against the floor; only this localized edge turn is proposed, not an arbitrary point along the long bottom edge.')]
}
U_REASONS={
20:'Elevated wood base behind red rack beam is blurred and partly image-cut; no stable pallet-only endpoint can be localized.',
21:'Small elevated wood base is crossed by rack beam and adjacent steel post. Blur prevents a stable exposed endpoint associated only with the selected pallet.',
22:'Elevated wood base is crossed by rack structure and labels; visible strips do not establish a stable physical endpoint.',
24:'Floor wood parts are visible, but the image-left end is cut and the yellow guard obscures the other end. Remaining support edges are blurred and do not provide a stable endpoint under this rule.',
25:'Floor wood base is strongly motion-blurred. Approximate wood patches are visible but individual endpoint localization is not stable; no arbitrary nearby foreground point is substituted.',
26:'Elevated wood base is crossed by steel rack and yellow beam, with blur and overlapping cargo; individual endpoint association is ambiguous.',
27:'Loaded wood base is motion-blurred and its near lower end meets the image boundary. Visible edge segments do not yield a sufficiently localized stable endpoint.',
29:'Elevated base is truncated at the image top and crossed by rack/cargo edges; no stable front-base endpoint is established.',
32:'Small blue base near the image-right edge is blurred and partly cut. Layer/part association and endpoint localization are ambiguous.',
33:'Small elevated gray base is blurred and crossed by diagonal steel; no stable base-only physical endpoint is localized.',
34:'The representative pallet is largely occluded by the vehicle and nearby structure. Visible parts do not establish an associated reference; this is U, not a natural E annotation.',
35:'Elevated base appears as an image-top sliver. Truncation and rack overlap prevent a stable exposed endpoint.',
36:'The tiny rear wood base is blurred at native resolution; a physical endpoint cannot be localized reliably.',
38:'Gray elevated base has visible parts but yellow beam/steel overlap and soft boundaries make physical endpoint association ambiguous. No lower hidden corner is reconstructed.',
39:'Loaded floor base is partly covered by the person and cut at the image bottom. The remaining far edge does not establish a stable associated endpoint; no natural E is generated.'}

def main():
    manifest=E/'EXTENSION_40_RGB_REVIEW_MANIFEST.jsonl'
    assert sha(manifest)=='6e79f3511157c1da2a4136367b4ca81cd9bdb53ff72389a3a44deb3394f08bd1'
    rows=[json.loads(x) for x in manifest.read_text().splitlines()]
    frames=[];refs=[]
    for f in rows[20:40]:
        n=f['extension_index'];assert 20<=n<=39
        assert sha(f['image'])==f['image_sha256']
        points=[]
        for k,(xy,bb,role,evidence,unc) in enumerate(PROPOSALS.get(n,[])):
            h,w=f['raw_hw'];assert 0<=bb[0]<=xy[0]<bb[2]<=w and 0<=bb[1]<=xy[1]<bb[3]<=h
            ref={'cue_refID':f'LOCO_V74_EXT{n:02d}:RGB_REF{k}', 'extension_index':n,
                 'source_subset':f['source_subset'],'source_SHA_rank':f['source_SHA_rank'],
                 'source_split':f['root_authorized_source_split'],
                 'image':f['image'],'image_sha256':f['image_sha256'],'raw_hw':f['raw_hw'],'image_id':f['image_id'],
                 'representative_annotation_id':f['representative_annotation']['id'],
                 'original_representative_bbox_xywh':f['representative_annotation']['bbox'],
                 'reference_xy':xy,'uncertainty_bbox_half_open_xyxy':bb,'reference_role_2D':role,
                 'state_proposal':'VISIBLE_REFERENCE_CONDITIONAL_AI','actual_RGB_evidence':evidence,
                 'localization_uncertainty_reason':unc,'target_instance_association':'AI visual association to the same frozen official representative entity; source bbox identity is preserved. Pending root visual review and any separately frozen ownership sanity.',
                 'native8_mapping':'UNKNOWN','physical_cuboid_vertex_correspondence':'UNVERIFIED; observed base/support/deck edge 2D cue only',
                 'source_evidence_tier':'AI_OBSERVATIONAL_NONHUMAN_CONDITIONAL_TEACHER_PROPOSAL',
                 'human_verified':False,'official_corner_annotation':False,'admitted_training_teacher':False,
                 'natural_E_or_S_generated':False,'crossframe_same_physical_reference':'UNKNOWN',
                 'initial_query_xy':None,'initializer_detection_or_raw_index_used':None,'source_mask_or_model_used_for_geometry':False}
            points.append(ref);refs.append(ref)
        assert len(points)<=2
        frames.append({**f,'reference_proposals':points,'frame_reference_state':'CONDITIONAL_VISIBLE_REFERENCE_CANDIDATES' if points else 'U',
                       'U_reason':None if points else U_REASONS[n],
                       'single_pallet_vs_stack_observation':'single exposed base portion conditionally associated to representative entity' if points else 'not established for a stable reference',
                       'other_visible_instances_not_substituted':True})
    counts={'frames':20,'official_representative_entities':20,'frames_with_conditional_visible_references':5,'frames_U_null':15,
            'conditional_visible_reference_proposals':8,'wood_appearance_reference_proposals':7,'gray_plastic_appearance_reference_proposals':1,
            'native8_mapped':0,'human_verified':0,'natural_E':0,'natural_S':0,'training_teachers_admitted':0}
    assert len(refs)==8 and len(frames)==20
    result={'schema':'V74_EXTENSION40_AGENT_B_RGB_REFERENCE_LEDGER_V1','frozen_UTC':datetime.now(timezone.utc).isoformat(),
            'status':'CONDITIONAL_AI_SOURCE_REFERENCE_PROPOSALS_PENDING_ROOT_REVIEW_NOT_GOLD_OR_TRAIN_ADMISSION',
            'assigned_extension_indices':list(range(20,40)),'protocol':bind(R/'PROTOCOL.json'),'parent_manifest':bind(manifest),
            'parent_selection_lock':bind(E/'EXTENSION_40_SELECTION_LOCK.json'),'frames':frames,'references':refs,'counts':counts,
            'shared_operational_definition':'An observed 2D turn or end of an exposed pallet base, deck, or support edge, tied to one frozen representative entity with a visually localized range. Reference identity/state remain fixed under any future query jitter. Exact query-pixel occupancy is a separate audit. Rounded or part edges are not canonical cuboid corners.',
            'split_statement':'Root-frozen source grouping is TRAIN subsets 2/3/5 and VAL subsets 1/4. This annotation pass has previewed its assigned source images in both groups; it does not establish new independent captures.',
            'geometry_freeze_evidence':{'model_outputs_or_scores_read':0,'ownership_masks_read':0,'new_model_forward_fit_current_PnP':0,
                                      'all_assigned_raw_fullframe_and_nativecrop_reviewed':True,'preliminary_RGB_only_localization_previews_preserved':True,
                                      'refinement_note':'Rough localization plans were refined or rejected with actual RGB and native coordinate grids before this freeze. No model, SAM, ownership mask or score was used; no bbox endpoint was copied as a reference.'},
            'limits':['These are nonhuman AI proposals, not official semantic D/E/S or native8 annotations.',
                      'U denotes no stable associated operational reference under this rule; it does not mean pallet absence or confirmed occlusion.',
                      'Visible-looking wood/plastic supports are conditional visual evidence. No mask positive automatically certifies semantic D.',
                      'No initializer detection is selected, shifted or recentered using these proposals. All original source geometry and frames are preserved.',
                      'Future controlled cover may use a separately frozen reference state; no natural E or S is created here.'],
            'audit_code':bind(Path(__file__))}
    out=R/'RGB_REFERENCE_PROPOSAL_LEDGER.json';assert not out.exists();out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    jl=R/'RGB_REFERENCE_PROPOSALS.jsonl';assert not jl.exists();jl.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in refs))
    lock=R/'RGB_REFERENCE_GEOMETRY_LOCK.json';assert not lock.exists()
    lock.write_text(json.dumps({'schema':'V74_EXTENSION40_AGENT_B_RGB_ONLY_GEOMETRY_LOCK','frozen_before_any_ownership_sanity':True,
                               'model_output_read':0,'ownership_mask_read':0,'bindings':[bind(out),bind(jl),bind(R/'PROTOCOL.json'),bind(manifest),bind(Path(__file__))]},indent=2)+'\n')
    print(json.dumps({'ledger':bind(out),'geometry_lock':bind(lock),'counts':counts},ensure_ascii=False))
if __name__=='__main__':main()
