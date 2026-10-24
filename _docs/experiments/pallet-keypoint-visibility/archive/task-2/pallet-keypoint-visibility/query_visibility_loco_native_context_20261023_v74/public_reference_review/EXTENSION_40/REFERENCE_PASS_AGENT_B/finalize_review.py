from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
R=Path(__file__).resolve().parent
E=R.parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def save(name,x):
 p=R/name;assert not p.exists();p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');return p

ledger=R/'RGB_REFERENCE_PROPOSAL_LEDGER.json';assert sha(ledger)=='3ec1c83052dd57fd740855925fa9e492ff2629ce7df0925e6cd3fdefc4dd3664'
a=json.loads(ledger.read_text());original=[json.loads(x) for x in (E/'EXTENSION_40_RGB_REVIEW_MANIFEST.jsonl').read_text().splitlines()]
checks=[]
for f in a['frames']:
 n=f['extension_index'];o=original[n]
 checks.append({'extension_index':n,'all_parent_fields_byte_value_preserved':all(f[k]==v for k,v in o.items()),'image_sha_match':sha(f['image'])==f['image_sha256'],
                'max_two_refs':len(f['reference_proposals'])<=2,'all_refs_valid_range':all(0<=r['uncertainty_bbox_half_open_xyxy'][0]<=r['reference_xy'][0]<r['uncertainty_bbox_half_open_xyxy'][2]<=f['raw_hw'][1] and 0<=r['uncertainty_bbox_half_open_xyxy'][1]<=r['reference_xy'][1]<r['uncertainty_bbox_half_open_xyxy'][3]<=f['raw_hw'][0] for r in f['reference_proposals'])})
assert all(all(v for k,v in q.items() if k!='extension_index') for q in checks)
post=save('POSTCHECK.json',{'schema':'V74_AGENT_B_SOURCE_GEOMETRY_POSTCHECK','PASS':True,'ledger':bind(ledger),'checks':checks,
                           'original_fixed20_geometry_unchanged':sha(E.parent/'RGB_REFERENCE_GEOMETRY_LOCK.json')=='632bfb81bc3107d231045e9917707b730e0b3fdc7469f1564becbb78fc2b2006',
                           'original_extension_manifest_unchanged':sha(E/'EXTENSION_40_RGB_REVIEW_MANIFEST.jsonl')=='6e79f3511157c1da2a4136367b4ca81cd9bdb53ff72389a3a44deb3394f08bd1',
                           'model_mask_current_forward_fit_PnP':0})
# Additive visual review only. It does not change proposal coordinates or training labels.
dis=[]
for r in a['references']:
 n=r['extension_index'];hold=n in [30,37]
 dis.append({'cue_refID':r['cue_refID'],'reference_xy_unchanged':r['reference_xy'],
             'disposition':'HOLD_NOT_FOR_POSITION_TEACHER' if hold else 'CONDITIONAL_AUXILIARY_CANDIDATE_PENDING_ROOT_PROTOCOL',
             'reason':('Motion blur is too large for reliable position supervision despite a broad visible-looking support endpoint.' if n==30 else 'Low-contrast rounded plastic contour does not establish a uniquely localizable molded turn in final raw/marked inspection.' if n==37 else 'Frozen range visually covers an exposed wooden support edge turn/end on the selected low base; remains a nonhuman conditional proposal.'),
             'human_or_native8_gold':False,'admitted_training':False})
qa=save('POST_FREEZE_RGB_VISUAL_REVIEW_ADDENDUM.json',{'schema':'V74_AGENT_B_POST_FREEZE_RGB_VISUAL_REVIEW','created_UTC':datetime.now(timezone.utc).isoformat(),
         'frozen_ledger':bind(ledger),'frozen_geometry_lock':bind(R/'RGB_REFERENCE_GEOMETRY_LOCK.json'),
         'original_proposal_counts':a['counts'],'conditional_auxiliary_review_candidates':6,'candidate_frames':[23,28,31],
         'HOLD_proposals':2,'HOLD_only_frames':[30,37],'original_U_frames':15,'all_frames_retained':20,
         'dispositions':dis,'root_visual_review_agreement':'Root separately viewed the actual source raw/marked figure and requested HOLD for EXT30 and EXT37. No coordinates were moved.',
         'original_geometry_or_labels_modified':False,'teacher_admission':False,'new_E_S_or_native8_created':0,
         'limits':'This is a source annotation proposal quality disposition, not a model score or semantic/pixel accuracy evaluation. No mask/model/current data was read.'})
rootledger=E/'REFERENCE_PASS_ROOT_A/RGB_REFERENCE_PROPOSAL_LEDGER.json'
rootreview=save('INDEPENDENT_ROOT_A_RGB_REVIEW.json',{'schema':'V74_ROOT_A_INDEPENDENT_RGB_VISUAL_REVIEW_BY_AGENT_B','reviewed_root_ledger':bind(rootledger),
       'images':[bind(R/'INDEPENDENT_ROOT_A_NATIVE96_REVIEW.png'),bind(R/'INDEPENDENT_ROOT_A_FIXED_COORDINATE_GRID.png'),bind(R/'INDEPENDENT_ROOT_A_TWO_FINAL_COORDINATE_GRID.png')],
       'review_scope':'All ten frozen A proposals viewed as native96 raw/marked crops; selected fixed coordinate grids for ambiguous ranges. No coordinates moved and no replacement proposed.',
       'clear_blocker':[{'extension_index':17,'ref_suffix':'ROOT_RGB_REF1','xy':[1258,570],'reason':'Fixed uncertainty range lies on floor, with no corresponding exposed wood endpoint visible in that range. Recommend HOLD/U.'}],
       'conservative_root_HOLD_agreement':[{'extension_index':5,'ref_suffix':'ROOT_RGB_REF1','xy':[306,417],'reason':'Wood/floor transition is at or outside the very edge of the uncertainty range; not sufficiently clear for reliable position teacher.'},{'extension_index':17,'ref_suffix':'ROOT_RGB_REF0','xy':[1145,600],'reason':'Very low-contrast wood lower contour; conservative HOLD is appropriate.'},{'extension_index':18,'ref_suffix':'ROOT_RGB_REF0','xy':[302,620],'reason':'Range is dominated by floor near the wood contour; conservative HOLD is appropriate.'}],
       'two_explicit_final_checks':[{'extension_index':11,'ref_suffix':'ROOT_RGB_REF0','xy':[921,657],'opinion':'Conditional auxiliary candidate: the fixed range includes the near support lower contour around y653–657. Exact center-pixel occupancy is not certified.'},{'extension_index':18,'ref_suffix':'ROOT_RGB_REF1','xy':[300,592],'opinion':'Conditional auxiliary candidate: the fixed range includes the bright wood deck/support edge turn directly below gray cargo. It is an observed part cue, not a canonical cuboid vertex.'}],
       'remaining_A_proposals':'No clear gross association/placement blocker seen in EXT4 both, EXT5 REF0 and EXT11 REF1; all remain nonhuman conditional proposals subject to root protocol.',
       'native8_gold_added':0,'human_gold_added':0,'natural_E_S_added':0,'model_mask_current_forward_fit_PnP':0,'original_A_files_modified':False})
result=save('FINAL_REVIEW_SUMMARY.json',{'schema':'V74_AGENT_B_REFERENCE_PASS_FINAL_SUMMARY','status':'COMPLETED_RGB_ONLY_WITH_ADDITIVE_HOLD_DISPOSITION',
      'assigned_indices':list(range(20,40)),'proposal_ledger':bind(ledger),'geometry_lock':bind(R/'RGB_REFERENCE_GEOMETRY_LOCK.json'),
      'actual_gallery_result':bind(R/'ACTUAL_REFERENCE_REVIEW_RESULT.json'),'postcheck':bind(post),'post_freeze_visual_disposition':bind(qa),'independent_root_A_review':bind(rootreview),
      'counts':{'reviewed_frames':20,'original_RGB_proposals':8,'original_proposal_frames':5,'original_U_frames':15,'candidate_auxiliary_refs_after_visual_HOLD':6,'candidate_auxiliary_frames':3,'HOLD_refs':2,'training_admitted':0,'native8_or_human_gold':0,'natural_E_or_S':0},
      'all_original_frames_and_geometry_preserved':True,'model_mask_current_forward_fit_PnP':0})
files=[R/'PROTOCOL.json',R/'freeze_references.py',ledger,R/'RGB_REFERENCE_PROPOSALS.jsonl',R/'RGB_REFERENCE_GEOMETRY_LOCK.json',R/'render_gallery.py',R/'ACTUAL_REFERENCE_REVIEW_RESULT.json',post,qa,rootreview,result,Path(__file__)]
files+=sorted(R.glob('FROZEN_REFERENCE_PAGE_*.png'))+[R/'EIGHT_FROZEN_REFERENCE_NATIVE_RAW_MARKED.png']+sorted((R/'FRAME_CARDS').glob('*.png'))
seal=save('FINAL_REVIEW_SEAL.json',{'schema':'V74_AGENT_B_REFERENCE_REVIEW_SEAL','created_UTC':datetime.now(timezone.utc).isoformat(),'bindings':[bind(p) for p in files],
              'no_model_mask_current_forward_fit_PnP':True,'all_frozen_proposal_geometry_preserved':True,'status':'CONDITIONAL_SOURCE_PROPOSALS_ONLY_NO_TRAINING_ADMISSION'})
print(json.dumps({'summary':bind(result),'seal':bind(seal),'disposition':bind(qa),'root_A_review':bind(rootreview)}))
