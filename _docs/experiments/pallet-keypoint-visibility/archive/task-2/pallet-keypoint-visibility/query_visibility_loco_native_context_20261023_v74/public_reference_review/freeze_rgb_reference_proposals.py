"""AI-observational source-reference proposals; no predictedq/mask/model reads."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))

# Native coordinates chosen by actual rawRGB+nativecoordinategrid visual review.
# The preliminary marker previews are preserved; this is the final RGB-only freeze.
PROPOSALS={
0:[([1656,892],[1648,884,1665,901],'UPPER_FRONT_SUPPORT_EDGE_TURN','Exposed upper-right turn of the near/front lower wooden support face on the lowest discernible pallet in the frozen bbox. The horizontal wood edge turns into a vertical wood support edge, distinct from racksteel.','Blurred/wornwood edge transition; a range is retained rather than exact materialpixel truth.'),
   ([1838,901],[1830,893,1847,911],'LOWER_IMAGE_RIGHT_SUPPORT_END','Lower outer wood-support/floor endpoint at the image-right end of the same lowest discernible woodpallet; upperpallet layers are not the reference.','Wood/floor edge is soft and perspective compresses the far support. No canonicalcuboidassignment.')],
1:[([239,417],[235,412,245,423],'LOWER_IMAGE_RIGHT_BASE_END','Exposed rightmost wooden supportfoot meets the floor, under the loaded carton. The adjacent perforatedsteel upright is background, not the referenced woodface.','Lowresolution and antialiasing around the wood/floor endpoint.'),
   ([242,389],[237,384,248,396],'UPPER_IMAGE_RIGHT_DECK_END','Right end of the visible thin wooddeck/support junction below the carton and before the steelrack upright.','Wood/rack adjacency and soft deck thickness; left end is image-truncated, not completed.')],
8:[([939,706],[932,699,947,714],'LOWER_IMAGE_LEFT_FRONT_SUPPORT_END','Lower exposed near/front image-left woodsupport endpoint of the floorpallet carrying the grayboxes. The corner belongs to the woodbase, not a boxcorner.','Woodfloor contrast is soft; endpoint lies near imagebottom, with all uncertainty still inside actualRGB.'),
   ([1197,645],[1188,636,1206,654],'LOWER_IMAGE_RIGHT_SUPPORT_END','Exposed far-right pallet supportfoot/floor turn under the same loadedwoodbase; bboxcontext establishes the same pallet.','Perspective and darkunderside make the far endpoint less sharp. Conditionalproposal only.')],
16:[([1665,795],[1659,789,1673,802],'LOWER_IMAGE_LEFT_PLASTIC_SUPPORT_END','Exposed lower image-left grayplastic support/floor turn of the loaded lowpallet selected by the frozen bbox. Redcrate edges are cargo, not references.','Softgray/floor boundary and curvedplastic corner; this is an operational2Dcue, not an idealcuboidvertex.'),
    ([1655,721],[1649,715,1663,729],'UPPER_IMAGE_LEFT_PLASTIC_DECK_END','Upper image-left grayplastic rim/support turn directly under the redcargo.','Roundedplastic geometry and adjacentcargo create a localizationrange. Rightpallet extent is image-cut; no hiddencornercompletion.')],
17:[([223,494],[217,488,230,502],'LOWER_IMAGE_RIGHT_PLASTIC_SUPPORT_END','Right end of exposed grayplastic foot onfloor beneath the redcontainers within the selected singlebase segment.','Lowresolution/roundedfoot; do not use precise maskownership assemanticvisibility.'),
    ([233,445],[227,439,240,453],'UPPER_IMAGE_RIGHT_PLASTIC_DECK_END','Upper right plasticrim/verticalsupport turn beneath redcargo, not the redcargo corner or nearbydoorframe.','Darkslot and cargo adjacency leave uncertainty. No semanticnative8mapping.')]
}
U_REASONS={
2:'Largestbbox is an elevated loadedwoodbase partly crossed by rackupright and blurred. A stable individualbase endpoint cannot be localized under the fixed rule.',
3:'Elevated smallwoodbase under redbins is image-top truncated and crossed by rackupright/beam; individual endpoint association ambiguous.',
4:'Largestbbox is a dark underside/elevated partly image-cut pallet. Visible slats do not establish a stable near/front base endpoint.',
5:'Largest elevatedbase is strongly motionblurred, near imageedge and behind steel/IBC edges; no reliable physicalcorner localization.',
6:'Small elevated pallet under bluecrates is crossed by racksteel. Blur/darkness and neighboringbase parts prevent unambiguous endpoint association.',
7:'Floor woodbase under barrels is visible, but native endpoint localization is blurred/overlapping in the frozen180x67bbox. Preliminary markers did not establish stable endpoints; retainU instead of relocating to a visuallyconvenient differentpart.',
9:'Large elevatedwoodbase shows underside slats but near/front cuboid end is obscured by diagonal rackbeam. Multiple edgejunctions cannot be assigned uniquely.',
10:'Elevated wood strips are crossed by redverticalrack/beam and labels; largestbbox does not yield a stable pallet-only physicalendpoint.',
11:'Largestimage-top/rightrack base is strongly blurred and partly cut. Woodedge and cargo/rack boundaries are not separable at nativeprecision.',
12:'Largest elevatedbase is image-top cut and crossed by steel diagonalstructure, with motionblur; no stable exposedendpoint.',
13:'Floorbase under whitecargo is visible but blurred, dark and partly blocked by steel/yellowguard. Foot/floor endpoint localization not stable enough for thisproposal.',
14:'Largest elevated darkbase behind steelpost7/cargo is heavily blurred; visible pallet-only endpoint association ambiguous.',
15:'Largest elevated underface near image-top behind steel columns/beam has no uniquely localized frontbase endpoint.',
18:'Grayplasticpallets form a visible stack; frozenlargestbbox spans more than one apparent layer. A clearphysicaledge exists but mapping it to the selected annotation as one pallet is not established; no arbitrarybottom-layer reassignment.',
19:'Wrappedloadedbase is partly covered by packaging. Pallet/cargo/film corner association is ambiguous; no referencecoordinate is invented and no naturalElabel is generated.'}

def main():
    lock=R/'REPRESENTATIVE_INSTANCE_LOCK.json';assert sha(lock)=='1e77ed9eba4b05d0b03f2063912a2f06b8db0d7ce0e30558ed850fec1032c27f'
    records=json.loads(lock.read_text())['records'];frames=[];refs=[]
    for f in records:
        n=f['selection_index'];points=[]
        for j,(xy,box,role,evidence,uncertain) in enumerate(PROPOSALS.get(n,[])):
            h,w=f['raw_hw'];assert 0<=xy[0]<w and 0<=xy[1]<h
            assert 0<=box[0]<=xy[0]<box[2]<=w and 0<=box[1]<=xy[1]<box[3]<=h
            ref=dict(cue_refID=f"{f['review_id']}:RGB_REF{j}",image=f['image'],image_sha256=f['image_sha256'],image_id=f['image_id'],representative_annotation_id=f['representative_annotation']['id'],
              reference_xy=xy,uncertainty_bbox_half_open_xyxy=box,reference_role_2D=role,
              state_proposal='VISIBLE_REFERENCE_CONDITIONAL_AI',actual_RGB_evidence=evidence,localization_uncertainty_reason=uncertain,
              native8_mapping='UNKNOWN',physical_cuboid_vertex_correspondence='UNVERIFIED; observedbase/support/deckedge2Dcueonly',
              target_instance_association='AIvisual association to the same frozenofficialrepresentativebbox; sourceannotationidentityheldfixed. Pendingrootreview/ownershipsanity.',
              source_evidence_tier='AI_OBSERVATIONAL_NONHUMAN_CONDITIONAL_TEACHER_PROPOSAL',human_verified=False,official_corner_annotation=False,
              admitted_training_teacher=False,natural_E_or_S_generated=False,crossframe_same_physical_reference='UNKNOWN',
              initial_query_xy=None,initializer_detection_or_raw_index_used=None,source_mask_or_model_used_for_geometry=False)
            points.append(ref);refs.append(ref)
        assert bool(points)==(n in PROPOSALS)
        frames.append(dict(**f,reference_proposals=points,frame_reference_state='CONDITIONAL_VISIBLE_REFERENCE_CANDIDATES' if points else 'U',
          U_reason=None if points else U_REASONS[n],single_pallet_vs_stack_observation=('lowestdiscerniblewoodpalletwithinlargerstack;conditionalunitassociation' if n==0 else 'stack_layer_to_annotation_ambiguous' if n==18 else 'singleexposedbaseportionconditional' if points else 'notestablished_for_reference'),
          other_visible_instances_not_substituted=True))
    result=dict(schema='V74_FIXED20_RGB_OPERATIONAL_REFERENCE_PROPOSAL_LEDGER_V1',frozen_UTC=datetime.now(timezone.utc).isoformat(),
      status='CONDITIONAL_AI_SOURCE_REFERENCE_PROPOSALS_PENDING_ROOT_REVIEW_NOT_GOLD_OR_TRAIN_ADMISSION',
      representative_lock=bind(lock),protocol=bind(R/'REFERENCE_REVIEW_PROTOCOL.json'),frames=frames,references=refs,
      counts=dict(frames=20,official_representative_entities=20,frames_with_conditional_visible_references=5,frames_U_null=15,conditional_visible_reference_proposals=10,native8_mapped=0,human_verified=0,natural_E=0,natural_S=0,training_teachers_admitted=0),
      shared_operational_definition='An observed2Dturn/end of exposedpalletbase/deck/supportedges, tiedtoonefrozenrepresentativeentity, with visuallylocalizedrange. Referenceidentity/state remainfixedunderfuturequeryjitter; exactquerypixeloccupancy is a distinctaudit. Rounded/internalpart edges are not relabeled ascanonicalcuboidcorner.',
      original_reference_and_model_input_preservation='No predictedquery/box existsinthisledger. Originalofficialbbox and20sourceframes unchanged. Refxy is never usedtoselectanotherdetection or alterinitializer.',
      geometry_freeze_evidence=dict(model_outputs_or_scores_read=0,ownership_masks_read=0,newmodel_forward_fit_current_PnP=0,
        raw_fixed20_fullframe_and_nativecrop_visual_review=True,preliminary_RGB_only_previews_preserved=['VISUAL_LOCALIZATION_PREVIEW_NOT_FINAL.png','VISUAL_LOCALIZATION_PREVIEW_V2_NOT_FINAL.png'],
        refinement_note='Initialmanualroughlocations were refined/dropped using onlyactualRGB/coordinategrids before thisfinalgeometryfreeze. No model/mask/GTpixelbasedrelocation. Frame7droppedtoU; no replacemententity.'),
      group_split='OriginalLOCOsubset/session grouping remains pending the separateV74sourcegroupaudit. ThisledgerdoesnotcreateindependentTRAIN/VAL.',
      limits=['Officialmanualbbox is targetidentity evidence, not semanticD/E/S or native8cornergold.','Conditionalvisibleproposals require rootvisualreview and separateownershipsanity; M4positive is not an automaticsemanticDcertificate.','Uframes include clearotherpallets or visibleparts; Umeans no stableassociatedreference underthisfrozenrepresentative rule, not absence/occlusion.','SOURCE20 selection exposure and thisAIannotationmust be declared in futurefit/evalprotocol; no currentdata is used.'],
      audit_code=bind(Path(__file__)))
    out=R/'RGB_REFERENCE_PROPOSAL_LEDGER.json';assert not out.exists();out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    rows=R/'RGB_REFERENCE_PROPOSALS.jsonl';assert not rows.exists();rows.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in refs))
    seal=R/'RGB_REFERENCE_GEOMETRY_LOCK.json';assert not seal.exists();seal.write_text(json.dumps(dict(schema='V74_RGB_ONLY_OPERATIONAL_REFERENCE_GEOMETRY_LOCK',frozen_before_ownership_sanity=True,model_output_read=0,bindings=[bind(out),bind(rows),bind(lock),bind(R/'REFERENCE_REVIEW_PROTOCOL.json'),bind(Path(__file__))]),indent=2)+'\n')
    print(json.dumps(dict(ledger=bind(out),geometry_lock=bind(seal),counts=result['counts']),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
