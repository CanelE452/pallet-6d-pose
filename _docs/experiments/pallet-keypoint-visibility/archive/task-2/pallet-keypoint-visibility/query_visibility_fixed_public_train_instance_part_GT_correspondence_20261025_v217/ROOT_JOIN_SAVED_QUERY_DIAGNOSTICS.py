"""Join only the two completed V217 saved ledgers; no payload or model access."""
import csv, hashlib, io, json, math, statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
V=Path(__file__).parent

def read(rel,sha):
    assert 'v112' not in rel.lower()
    p=V/rel
    assert not p.is_symlink()
    b=p.read_bytes(); assert hashlib.sha256(b).hexdigest()==sha
    return json.loads(b)

a=read('fixed12_mask_depth_query_diagnostics_v2/ALL_DIAGNOSTIC_ROWS.json','4708c8958e43436ff236807e75e6e7a407016bb245f0959fb97b0ce1f18585a3')
b=read('nominal_candidate_query_projection_v1/ALL120_CANDIDATE_NATIVE_QUERY_DIAGNOSTICS.json','e1c45b308b2a8263812043ce4bb78e4fff616506b3366bdbc174b7fa6d021e2e')
pa={(r['frame_position'],r['candidate_gt_list_index'],r['native_slot']):r for r in a['candidate_native_rows120']}
pb={(r['original_input']['frame_position'],r['diagnostic_candidate_gt_list_index'],r['native_slot']):r for r in b['rows120']}
assert len(pa)==len(pb)==120 and pa.keys()==pb.keys()
candidates={(r['frame_position'],r['candidate_gt_list_index']):r for r in a['candidate_rows15']}
rows=[]; unique={}
for key in sorted(pa):
    x,y=pa[key],pb[key]; o=y['original_input']; qd=y['query_diagnostic']; c=candidates[key[:2]]
    assert x['query_xy']==o['query_xy'] and x['L']==o['L'] and x['original_query_available']==o['original_query_available']
    ref=y['nominal_reference_xy']; adr=qd['RAW96_reference_address']; delta=qd['delta_reference_minus_q_px']
    avail=x['original_query_available']; assert bool(avail)!=bool(x['original_inputU'])
    r={
      'frame_position':key[0],'frame_key':x['frame_key'],'GT_candidate':key[1],'native':key[2],
      'source_role':x['internal_source_role'],'row_id':x['row_id'],'original_inputU':x['original_inputU'],
      'q_x':x['query_xy'][0],'q_y':x['query_xy'][1],'L':x['L'],
      'nominal_ref_x':ref[0] if ref else None,'nominal_ref_y':ref[1] if ref else None,
      'nominal_ref_in_image':bool(0<=ref[0]<c['width'] and 0<=ref[1]<c['height']) if ref else None,
      'nominal_delta_px':math.hypot(*delta) if delta else None,
      'algebraic_RAW96_ref_x':adr[0] if adr else None,'algebraic_RAW96_ref_y':adr[1] if adr else None,
      'nominal_ref_inside_original96_address_range':bool(0<=adr[0]<=95 and 0<=adr[1]<=95) if adr else None,
      'fullmask_at_raw_nearest_q':x['full_mask_at_nearest'],
      'visibleproxy_at_raw_nearest_q':x['visible_mask_at_nearest'],
      'positive_depth_at_raw_nearest_q':x['positive_finite_depth_at_nearest'],
      'historical_V216_local_weak_part':x['saved_V216_root_weak_part_state'],
      'target_correspondence_caution':('STANDING_PALLET_LOCAL_BOARD_WITNESS_NOMINAL_NATIVE_REFERENCE_OUTSIDE_ORIGINAL_CROP' if key[0]==6 and key[2] in (4,5,6) else 'INSTANCE_AND_INTENDED_NATIVE_NOT_CERTIFIED'),
      'selected_detector_GT_instance':'UNKNOWN','native_D_E_SELF':'UNKNOWN','training_admission':False,
      'nominal_reference_is_physical_gold':False,'RAW96_mask_sampler_replayed':False,
    }
    rows.append(r)
    uk=(key[0],key[2]); u=(x['row_id'],x['internal_source_role'],x['original_inputU'],x['saved_V216_root_weak_part_state'])
    assert unique.setdefault(uk,u)==u
assert len(unique)==96 and sum(t[2] for t in unique.values())==11
assert sum(r['original_inputU'] for r in rows)==17
vals=[r['nominal_delta_px'] for r in rows if r['nominal_delta_px'] is not None]
summary={
 'frames':12,'GT_candidates':15,'original_unique_queries':96,'original_U':11,
 'original_roles':dict(Counter(t[1] for t in unique.values())),
 'candidate_expanded_rows':120,'candidate_expanded_U':17,'candidate_expanded_roles':dict(Counter(r['source_role'] for r in rows)),
 'historical_unique_weak_part_D':sum(t[3]=='D' for t in unique.values()),
 'historical_candidate_expanded_weak_part_D':sum(r['historical_V216_local_weak_part']=='D' for r in rows),
 'standing_local_board_witnesses_with_nominal_native_ref_outside_original_crop':3,
 'nominal_reference_in_image':sum(r['nominal_ref_in_image'] is True for r in rows),
 'nominal_reference_outside_image':sum(r['nominal_ref_in_image'] is False for r in rows),
 'available_candidate_query_deltas':len(vals),'nominal_delta_at_most3px':sum(v<=3 for v in vals),
 'nominal_delta_over3px':sum(v>3 for v in vals),'nominal_delta_median_px':statistics.median(vals),
 'new_native_D_E_SELF_gold':0,'training_admitted':0,'new_fit_model_forwards_PnP':0,
 'interpretation':'All GT candidates retained. Candidate duplicates are not independent queries. Nominal cuboid convention is uncertain and differs from deployed roles in some standing cases; displacement is not certified physical coordinate error. Raw nearest-q mask samples are not RAW96 support replay. Local weak part observations are not native visibility gold.'}
assert summary['nominal_reference_in_image']==70 and summary['nominal_reference_outside_image']==50
assert len(vals)==103 and summary['nominal_delta_at_most3px']==2
out={'completed':True,'PASS':True,'inputs':{'pixel_rows_sha256':'4708c8958e43436ff236807e75e6e7a407016bb245f0959fb97b0ce1f18585a3','nominal_rows_sha256':'e1c45b308b2a8263812043ce4bb78e4fff616506b3366bdbc174b7fa6d021e2e'},'summary':summary,'rows120':rows,'UTC':datetime.now(timezone.utc).isoformat()}
p=V/'ROOT_QUERY_INSTANCE_PART_SUPPORT_SYNTHESIS.json'; assert not p.exists(); p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
p=V/'ROOT_QUERY_INSTANCE_PART_SUPPORT_SYNTHESIS.csv'; assert not p.exists()
with p.open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(summary,ensure_ascii=False))
