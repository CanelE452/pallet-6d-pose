"""Freeze human-unverified AI cue ranges before computing crop availability."""
from pathlib import Path
import hashlib,json
OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
LOCAL=[
 {'status':'LOCALIZED_AI_D_CUE_PROPOSAL','xy':[296,291],'bbox':[292.5,287.5,300.5,293.5],
  'cue':'Front-right lower dark plastic body bevel/ground-boundary turning region, not the nearby small rectangular fork hole.',
  'association':'Same front-right lower extremity named by prior native2, represented now by the visible lower bevel/outline turn. Old renderer r remains separate.',
  'uncertainty_reason':'Rounded body corner and2–3px blur spread the boundary; the rectangular region includes the lower contour transition rather than asserting a mathematical vertex.',
  'confidence':'medium'},
 {'status':'U_UNRESOLVED_LOCALIZATION_OR_ASSOCIATION','xy':None,'bbox':None,
  'cue':'Dark wood deck/end-face/slat support junction near native1.',
  'association':'There are multiple dark slat/end-face boundaries. Existing D_candidate was a coarse near-corner observation; at this zoom one unique named corner cannot be localized without choosing one of several edges.',
  'uncertainty_reason':'Low light and blur merge several wood edges. No arbitrary nearest visible slat point is substituted.',
  'confidence':'low'},
 {'status':'LOCALIZED_AI_D_CUE_PROPOSAL','xy':[332,287],'bbox':[326.5,282.5,337.5,291.5],
  'cue':'Near/front-right lower rounded dark plastic body outline against gray ground, distinct from shelf vertical edge at right.',
  'association':'The lower body corner region corresponds conditionally to native2 front-right lower extent. A whole visible curve region is the new operational cue; it is not claimed to be a single mesh vertex.',
  'uncertainty_reason':'Broad rounded corner and shadow soften the contour;11x9native cells retain this complete local turning region.',
  'confidence':'medium'},
 {'status':'LOCALIZED_AI_D_CUE_PROPOSAL','xy':[331,270],'bbox':[324.5,265.5,337.5,275.5],
  'cue':'Near/front-right upper plastic deck/body corner transition, including the chamfered light deck rim and adjacent dark body turn.',
  'association':'Conditionally associated with native1 upper front-right extent. The light triangle alone is not treated as a new independent pallet landmark; the rim/body turning region is the cue.',
  'uncertainty_reason':'Chamfer has no unique sharp vertex. The13x10native-cell region covers plausible visual corner positions; xy is only an integer representative inside this region.',
  'confidence':'medium'},
 {'status':'LOCALIZED_AI_D_CUE_PROPOSAL','xy':[281,378],'bbox':[277.5,374.5,285.5,382.5],
  'cue':'Left/near upper plastic deck/body corner where the diagonal top outline turns into the vertical near body edge.',
  'association':'Same upper left/front extent as native0, but the directly observed rounded/meshed corner is several pixels from old cuboid r. It gets a NEW cue_refID; old r is not corrected.',
  'uncertainty_reason':'Raster blur and chamfer create an8x8native-cell corner region. Ground/shadow farther below is excluded from the proposed upper-corner cue.',
  'confidence':'medium'},
 {'status':'LOCALIZED_AI_D_CUE_PROPOSAL','xy':[210,214],'bbox':[206.5,210.5,213.5,218.5],
  'cue':'Left/far upper green pallet outer corner where the top border bends into the side border, not a top-grid hole.',
  'association':'Conditional correspondence to native4 back upper-left extent. Visible outer pallet border gives a distinct2D cue near renderer r.',
  'uncertainty_reason':'Rounded low-resolution border and texture background create a7x8native-cell range; no common radius was applied.',
  'confidence':'medium-high'},
 {'status':'U_UNRESOLVED_LOCALIZATION_OR_ASSOCIATION','xy':None,'bbox':None,
  'cue':'Projected lower left pallet body corner near dark floor shadow.',
  'association':'Body bottom and cast shadow merge near native3. Upper corner is visible, but choosing it would change the named lower reference. No coordinate is proposed.',
  'uncertainty_reason':'Cannot unambiguously separate the physical lower edge from shadow or virtual cuboid lower extent in existing RGB crop.',
  'confidence':'low'},
 {'status':'LOCALIZED_AI_D_CUE_PROPOSAL','xy':[313,345],'bbox':[309.5,341.5,317.5,349.5],
  'cue':'Near/front green support/body lower tip where two visible side faces terminate at the ground.',
  'association':'Conditionally associated with native3 front lower-left extent. The visible support end, rather than an adjacent grid hole or shadow pixel, is the new2D cue.',
  'uncertainty_reason':'Rounded lower tip and dark texture boundary span8x8native cells. Integer xy represents the region, not subpixel certainty.',
  'confidence':'medium'},
]
def main():
    dest=OUT/'VISUAL_LOCALIZATION_OBSERVATIONS_LOCK.json';assert not dest.exists()
    lock=json.loads((OUT/'CUE_PROPOSAL_SELECTION_LOCK.json').read_text());grid=json.loads((OUT/'LOCALIZATION_GRID_OUTPUTS.json').read_text())
    rows=[]
    for c,o in zip(lock['candidates'],LOCAL):
        row={'candidate_point_ID':c['candidate_point_ID'],'NEW_cue_refID':'AI_CUE_V69:'+c['candidate_point_ID']+':visible-body-corner-v1',
             'proposal_status':o['status'],'associated_cue_xy':o['xy'],'uncertainty_bbox_half_open_xyxy':o['bbox'],
             'cue_observation':o['cue'],'native_corner_association_evidence':o['association'],'visual_uncertainty_explanation':o['uncertainty_reason'],
             'AI_localization_confidence_only':o['confidence'],'human_verified':False,'semantic_gold':False,'training_admitted':False}
        if o['xy'] is not None:
            x,y=o['xy'];b=o['bbox'];assert b[0]<=x<b[2] and b[1]<=y<b[3]
        rows.append(row)
    result={'schema':'AI_VISUAL_CUE_LOCALIZATION_LOCK_V1','status':'FROZEN_AFTER_VISUAL_LOCALIZATION_BEFORE_CROP_AVAILABILITY_COMPUTATION',
        'selection_lock_sha256':sha(OUT/'CUE_PROPOSAL_SELECTION_LOCK.json'),'grid_manifest_sha256':sha(OUT/'LOCALIZATION_GRID_OUTPUTS.json'),
        'localization_basis':'Actually viewed all8 coordinate-grid/native96 rasters and raw duplicates reconstructed from prior review sheets. No new originalRGB decode, GT masks or model/performance used.',
        'coordinate_semantics':'Integer xy is a representative native pixel center of an explicitly bounded visible corner/edge turning region, not an exact recovered3D or mesh vertex. Half-integer bbox edges specify full native-cell coverage. Ranges differ by actual visual blur/bevel ambiguity; not tuned from crop availability.',
        'old_reference_unchanged':'NEW cue ID/xy is distinct from old renderer r/native8 GT. Original r,q,L,box and native-index lineage remain byte/value-identical in separate metadata.',
        'admission':'AI observational conditional proposals only, root review pending. No human annotation, no natural E/S admission, no fit/composites authorized.',
        'counts':{'selected':8,'localized_AI_D_proposals':6,'U_no_point_invented':2},'observations':rows,
        'script_sha256':sha(Path(__file__))}
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'path':str(dest),'sha256':sha(dest),'counts':result['counts']}))
if __name__=='__main__':main()
