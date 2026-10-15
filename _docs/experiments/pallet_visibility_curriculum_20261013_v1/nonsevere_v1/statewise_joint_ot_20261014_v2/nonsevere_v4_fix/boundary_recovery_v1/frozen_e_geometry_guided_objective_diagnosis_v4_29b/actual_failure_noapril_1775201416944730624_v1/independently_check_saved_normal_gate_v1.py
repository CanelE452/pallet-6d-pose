"""Independent scalar arithmetic only, from two immutable saved JSON files."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import math

OUT=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    process=OUT/'PROCESS_DATA.json';claim=OUT/'V30_NORMAL_GATE_LIMITATION_V1.json'
    d=json.loads(process.read_text());expected=json.loads(claim.read_text())
    count=0;maximum=0.
    def check(value):
        nonlocal count
        count+=1
        if not value:raise AssertionError('Check '+str(count))
    def near(value,saved):
        nonlocal maximum
        difference=abs(float(value)-float(saved));maximum=max(maximum,difference)
        check(math.isfinite(difference)and difference<=1e-10)
    check(sha(process)==expected['input']['sha256']=='cd95741fb1b16bcad719053c7aa1e3c2bb45fa605573352d49acb72defbdf6c7')
    check(d['id']==expected['id']=='eval_noapril:1775201416944730624')
    check(d['edge']==expected['edge']==[0,3])
    nx,ny=map(float,d['TLS_normal']);offset=float(d['TLS_offset']);norm=math.hypot(nx,ny)
    near(norm,1.)
    axis=(-ny,nx)
    dot=lambda a,b:math.fsum([float(a[0])*float(b[0]),float(a[1])*float(b[1])])
    distance={};positions={}
    for k in expected['edge']:
        records=[r for r in d['gated_archived_BI']if r['native_corner']==k]
        check(len(records)==1 and records[0]['state']in('B','I'))
        xy=records[0]['xy']
        check(len(xy)==2 and all(math.isfinite(float(v))for v in xy))
        distance[str(k)]=abs(dot((nx,ny),xy)+offset)/norm
        positions[str(k)]=dot(axis,xy)
        near(distance[str(k)],expected['infinite_TLS_normal_distance_to_original_BI_q_px'][str(k)])
        near(positions[str(k)],expected['original_BI_q_tangent_positions_px'][str(k)])
        # A simultaneous sign reversal defines the same unoriented TLS line.
        near(abs(dot((-nx,-ny),xy)-offset)/norm,distance[str(k)])
    points=d['actual_inlier_xy']
    check(len(points)==expected['actual_inlier_count']==16)
    coordinates=[dot(axis,p)for p in points]
    actual=[min(coordinates),max(coordinates)];qspan=[min(positions.values()),max(positions.values())]
    for computed,saved in zip(actual,expected['actual_support_tangent_interval_px']):near(computed,saved)
    for computed,saved in zip(actual,d['actual_support_tangent_interval_px']):near(computed,saved)
    for computed,saved in zip(qspan,expected['original_BI_segment_tangent_interval_px']):near(computed,saved)
    gap=max(actual[0]-qspan[1],qspan[0]-actual[1],0.)
    near(gap,expected['disjoint_finite_support_gap_px'])
    negative_actual=[-actual[1],-actual[0]];negative_qspan=[-qspan[1],-qspan[0]]
    near(max(negative_actual[0]-negative_qspan[1],negative_qspan[0]-negative_actual[1],0.),gap)
    radius=expected['existing_V30_normal_gate_radius_px']
    check(radius==43.)
    passed=all(value<=radius for value in distance.values())
    check(passed is expected['same_V29_candidate_passes_V30_normal_gate'] is True)
    check(gap>0. and qspan[0]>actual[1])
    check(expected['not_a_claim_this_candidate_is_selected_by_full_V30_graph']is True)
    check(expected['new_image_measurements_posefit_reference_decode_or_runtime_change']==0)
    # Recheck the exact two source files after arithmetic; no opaque image or
    # additional frame is opened, and no runtime/noise/selection module imported.
    before={process.name:sha(process),claim.name:sha(claim)}
    for name,digest in before.items():check(sha(OUT/name)==digest)
    receipt=dict(schema='INDEPENDENT_SAVED_SCALAR_V30_NORMAL_GATE_LIMITATION_CHECK_V1',
        UTC=datetime.now(timezone.utc).isoformat(),passed=True,checks=count,failure_count=0,
        maximum_numeric_difference=maximum,source_sha256=sha(Path(__file__)),
        input_sha256=before,id=d['id'],edge=d['edge'],normal_unit_length=norm,
        recomputed=dict(normal_distance_px=distance,archived_BI_tangent_positions_px=positions,
            actual_support_interval_px=actual,archived_BI_segment_interval_px=qspan,disjoint_gap_px=gap,
            existing_normal_gate_radius_px=radius,same_cached_candidate_passes_normal_gate=passed),
        interpretation='Both endpoint q values are near the same infinite TLS line while their along-line segment is disjoint from the actual support. Normal-distance gating alone cannot distinguish this finite association. This does not assert selection by the full V30 graph.',
        scope='Pure Python scalar fsum/hypot/dot/minmax and line-sign-invariance checks on saved PROCESS_DATA and root limitation JSON only; no NumPy or runtime model/selector/observation imports.',
        new_row_score_RGB_GT_NN_TLS_posefit_or_runtime_changes=0)
    p=OUT/'INDEPENDENT_NORMAL_GATE_LIMITATION_CHECK_V1.json'
    with p.open('x')as stream:json.dump(receipt,stream,ensure_ascii=False,indent=2);stream.write('\n')
    q=OUT/'INDEPENDENT_NORMAL_GATE_LIMITATION_CHECK_SEAL_V1.json'
    with q.open('x')as stream:json.dump(dict(receipt_sha256=sha(p),input_sha256=before,passed=True),stream,indent=2);stream.write('\n')
    print(json.dumps(dict(checks=count,passed=True,max_numeric_difference=maximum,receipt_sha256=sha(p),seal_sha256=sha(q),recomputed=receipt['recomputed'])))


if __name__=='__main__':main()
