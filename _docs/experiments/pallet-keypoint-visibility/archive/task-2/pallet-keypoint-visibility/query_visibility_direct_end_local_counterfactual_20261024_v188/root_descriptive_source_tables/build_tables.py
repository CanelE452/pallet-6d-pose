"""Descriptive SOURCE-only ledgers. No pixel/model/evaluation loaders."""
from pathlib import Path
from collections import Counter
import json, hashlib, math, csv

H = Path(__file__).absolute().parent
V = H.parent

def read(p):
    assert all('v112' not in s.lower() for s in p.parts)
    return json.loads(p.read_text())

def lines(p):
    assert all('v112' not in s.lower() for s in p.parts)
    return [json.loads(s) for s in p.read_text().splitlines() if s]

def binding(p):
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def write(p, a):
    with p.open('x') as f:
        json.dump(a, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def table(p, rows):
    with p.open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def scalar_support(q, length, hw):
    """Independently enumerate positive original bilinear taps, no producer import."""
    result = set()
    height, width = hw
    for j in range(96):
        y = q[1] + ((j+.5)/96-.5)*length
        for i in range(96):
            x = q[0] + ((i+.5)/96-.5)*length
            if not (-.5 <= x < width-.5 and -.5 <= y < height-.5):
                continue
            ix, iy = math.floor(x), math.floor(y)
            fx, fy = x-ix, y-iy
            for px, py, weight in ((ix,iy,(1-fx)*(1-fy)),(ix+1,iy,fx*(1-fy)),
                                    (ix,iy+1,(1-fx)*fy),(ix+1,iy+1,fx*fy)):
                if weight > 0 and 0 <= px < width and 0 <= py < height:
                    result.add((px,py))
    return result

def main():
    syn = lines(V/'synthetic_direct_end_controls_v1/ALL1024_END_CONTROL_ATTEMPTS.jsonl')
    var = lines(V/'synthetic_direct_end_controls_v1/ALL3072_VARIANT_STATUS.jsonl')
    real = lines(V/'real_end_local_pairs_v1/ALL66_REAL_END_PAIR_ATTEMPTS.jsonl')
    obs = lines(V/'actual_fixed22_end_observation_a/ALL22_ENDPOINT_OBSERVATIONS.jsonl')
    assert len(syn)==1024 and len(var)==3072 and len(real)==66 and len(obs)==22
    cells = []
    for cell in sorted({tuple(r['factors'][k] for k in ('material','aspect','geometry_family')) for r in syn}):
        rr=[r for r in syn if tuple(r['factors'][k] for k in ('material','aspect','geometry_family'))==cell]
        cells.append(dict(zip(('material','aspect','geometry_family'),cell), attempts=len(rr),
                          supported=sum(r['direct_end_states']==[0,1,0] for r in rr),
                          UNKNOWN=sum(r['direct_end_states']!=[0,1,0] for r in rr),
                          input_U=sum(not r['original_input_flags']['query_available'] for r in rr)))
    native=[]
    for k in range(8):
        ss=[r for r in syn if r['native_slot']==k];rr=[r for r in real if r['original_input']['native_slot']==k]
        native.append(dict(native_slot=k,synthetic_attempts=len(ss),synthetic_supported=sum(r['direct_end_states']==[0,1,0] for r in ss),
                           synthetic_UNKNOWN=sum(r['direct_end_states']!=[0,1,0] for r in ss),
                           real_attempts=len(rr),real_supported=sum(r['PASS'] for r in rr),real_UNKNOWN_or_failure=sum(not r['PASS'] for r in rr)))
    locality=[]
    for row in obs:
        endpoint=row['endpoint_observation']; boxes=endpoint['candidate_enclosures_xyxy']; rr=[r for r in real if r['case_id']==row['case_id']]
        rec=dict(case_id=row['case_id'],native_slot=row['native_slot'],L=row['original_L'],
                 observation=row['endpoint_observation_status'],candidate_pixels=0,raw_support_missing_pixels=0,
                 missing_outside_support_bounding_box=0,missing_inside_support_bounding_box_holes=0,
                 local_class='DIRECT_END_OBSERVATION_UNKNOWN',attempts=len(rr),supported=sum(r['PASS']for r in rr),
                 status_counts=json.dumps(dict(Counter(r['status']for r in rr)),ensure_ascii=False),
                 reasons=json.dumps([r.get('reason',[])for r in rr],ensure_ascii=False))
        if boxes:
            original=rr[0]['original_input'];support=scalar_support(original['query_xy'],original['L'],original['raw_hw'])
            pixels={(x,y)for b in boxes for y in range(math.floor(b[1]),math.ceil(b[3])+1)for x in range(math.floor(b[0]),math.ceil(b[2])+1)}
            missing=pixels-support
            xs=[p[0]for p in support];ys=[p[1]for p in support]
            xmin,xmax,ymin,ymax=min(xs),max(xs),min(ys),max(ys)
            internal={(x,y)for x,y in missing if xmin<=x<=xmax and ymin<=y<=ymax}
            rec.update(candidate_pixels=len(pixels),raw_support_missing_pixels=len(missing),
                       missing_outside_support_bounding_box=len(missing-internal),missing_inside_support_bounding_box_holes=len(internal),
                       local_class='ALL_CANDIDATE_RAW_PIXELS_IN_SUPPORT'if not missing else 'INTERNAL_DOWNSAMPLING_SUPPORT_HOLES'if missing==internal else 'CANDIDATE_CROSSES_SUPPORT_EXTENT')
        elif row['no_local_opportunity_due_to_outside_original_crop']:
            rec['local_class']='OBSERVED_INTENDED_END_OUTSIDE_ORIGINAL_CROP_UNKNOWN_NOT_E'
        locality.append(rec)
    for name,rows in [('SYNTHETIC_FACTOR_CELLS.csv',cells),('FULL_NATIVE8_DENOMINATORS.csv',native),('ALL22_REAL_LOCALITY_DIAGNOSTIC.csv',locality)]:table(H/name,rows)
    payloads=[b for r in real for b in r['variants'].values() if 'path'in b]
    summary=dict(completed=True,PASS=True,descriptive_source_only=True,synthetic=dict(attempts=1024,variants=3072,
        supported_triplets=sum(r['direct_end_states']==[0,1,0]for r in syn),unknown_triplets=sum(r['direct_end_states']!=[0,1,0]for r in syn),
        status=dict(Counter(r['status']for r in syn)),reason_occurrences=dict(Counter(s for r in syn for s in r['reasons'])),
        known_selected_variant_labels=dict(Counter(str(r['direct_end_external_target'])for r in var)),
        all8_attempted_packet_slots=24576,original_before_U=sum(not r['original_input_flags']['query_available']for r in syn),
        all_original1024_before_crop_parity=read(V/'synthetic_direct_end_controls_v1/COMPLETE.json')['counts']['before_RGB96_valid96_exactparity']),
        real=dict(cases=22,attempts=66,variant_statuses=198,constructed=sum(r['constructed']for r in real),supported=sum(r['PASS']for r in real),
            failed_or_UNKNOWN=sum(not r['PASS']for r in real),status=dict(Counter(r['status']for r in real)),actual_payloads=len(payloads),
            missing_payloads=198-len(payloads),actual_payload_native_slots=8*len(payloads),all_attempted_variant_native_slots=1584,
            reason_occurrences=dict(Counter(str(r.get('reason',''))for r in real if not r['PASS'])),
            case_locality_classes=dict(Counter(r['local_class']for r in locality))),
        factor_cells=cells,native8=native,all22_locality=locality,
        interpretation='Coverage statuses describe a conservative source control protocol, not model accuracy or physical coordinate error. Requiring every raw enclosure pixel to participate in the original four-tap RGB96 is stricter than asking whether the sampled crop contains recognizable endpoint evidence; downsampling holes are reported separately from crop-extent errors. No source envelope is shrunk after results.',
        labels='Direct finite-end weak control is separate from old nativeD/E/S, center owner, broad support and coordinate quality. All new native/quality/owner targets remain UNKNOWN; training admission false.',
        all8_full_goal='Synthetic full material/aspect/geometry/native8 denominator retained. Real fixed22 is WOOD only, requestednative7 absent; supported end controls alone do not qualify full task.',
        model_forward=0,fit=0,evaluation_data_reads=0,PnP=0,selected_model=None,fully_solved=False)
    write(H/'SUMMARY.json',summary)
    write(H/'COMPLETE.json',dict(completed=True,PASS=True,code=binding(Path(__file__).absolute()),summary=binding(H/'SUMMARY.json'),
                               tables=[binding(H/n)for n in ('SYNTHETIC_FACTOR_CELLS.csv','FULL_NATIVE8_DENOMINATORS.csv','ALL22_REAL_LOCALITY_DIAGNOSTIC.csv')]))
    print(json.dumps(dict(complete=binding(H/'COMPLETE.json'),synthetic=summary['synthetic'],real=summary['real']),ensure_ascii=False))

if __name__=='__main__':main()
