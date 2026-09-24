import numpy as np
from . import common as C
from scripts.annotate.convert_to_camera_facing_v4 import get_origin_3d,compute_perm_v4

def main():
    obj=C.object_data();q=C.points();direct=C.direct_mask();candidates=C.read(C.DOC/'C4_CANDIDATES.json')['candidates']
    # Native-frame support is read from an actual frozen training tensor, NOT teacher/model coordinates.
    path=C.ROOT/'data/pallet/results/pallet_existing_data_transfer_v1/cache/M/0002.npz'
    with np.load(path) as z:support=(z['keypoints'][0,:,2]==2).tolist()
    provenance=[]
    for i,p in enumerate(obj['keypoint_annotations']):
        provenance.append(dict(corner=i,source=p['source'],visibility=p['visibility'],reason=p['reason'],
            in_frame=p['in_frame'],extrapolated=obj['extrapolated_mask'][i],direct_manual=bool(direct[i]),T2_support=support[i]))
    meta={k:obj.get(k) for k in ('keypoint_frame','migration_status','pose_status','canonical_pose')}
    meta['axis_assignment_confirmed']=obj['camera_facing_pnp']['axis_assignment_confirmed']
    C.put(C.DOC/'ANNOTATION_PROVENANCE.json',dict(points=provenance,identity=meta,
        direct_manual_count=int(direct[:8].sum()),T2_support_count=sum(support[:8]),support_binding=C.bind(path),exact_xy_private=True))
    C.put(C.RAW/'ANNOTATION_COORDINATES_PRIVATE.json',dict(xy=q.tolist(),provenance=provenance))
    original=get_origin_3d(obj)
    # No independent renderer origin3D in this real annotation. Analytic geometry is only supplemental.
    d=obj['physical_dimensions_m'];base=C.xyz(d['x'],d['y'],d['z'])
    origin=np.c_[base[:8,0],base[:8,2],-base[:8,1]]
    historical=compute_perm_v4(original if original is not None else origin,q)
    area_c4=next((c['c4'] for c in candidates if c['perm_new_to_stored']==historical),None)
    rows=[]
    for c in candidates:
        p=c['perm_new_to_stored'];v=q[p];m=direct[p]
        def ordering(pairs,axis):
            pairs=[(a,b) for a,b in pairs if m[a] and m[b]]
            return dict(available=len(pairs),violations=sum(v[a,axis]>=v[b,axis] for a,b in pairs),
                        margins_px=[float(v[b,axis]-v[a,axis]) for a,b in pairs])
        lr=ordering(C.LR,0);tb=ordering(C.TB,1)
        front=C.area(v[:4]);rear=C.area(v[4:8]);side1=C.area(v[[0,4,7,3]]);side2=C.area(v[[1,5,6,2]])
        rows.append(dict(c4=c['c4'],LR=lr,TB=tb,independent_direct_manual_count=int(m[:8].sum()),
            direct_front_vertices=int(m[:4].sum()),direct_front_finite_edges=sum(bool(m[a] and m[b]) for a,b in [(0,1),(1,2),(2,3),(3,0)]),
            front_rear_depth='UNKNOWN_FROM_2D',front_area_px2=front,rear_area_px2=rear,
            area_provenance='MIXED_MANUAL_AND_PNP_PROJECTED_NOT_INDEPENDENT',
            axis_area_diff_px2=abs(front-rear),other_axis_area_diff_px2=abs(side1-side2),
            boundary_outside_count=int(((v[:8,0]<0)|(v[:8,0]>=640)|(v[:8,1]<0)|(v[:8,1]>=480)).sum()),
            historical_area_compatible=c['c4']==area_c4,
            direct_order_compatible=bool(lr['violations']==0 and tb['violations']==0)))
    # Convert numpy bool/int returned by comparisons to strict JSON scalars.
    for row in rows:
        for k in ('LR','TB'):row[k]['violations']=int(row[k]['violations'])
    axis_diffs=sorted({round(r['axis_area_diff_px2'],9) for r in rows},reverse=True)
    result=dict(frame=C.FRAME,candidates=rows,geometry_model_payloads_loaded=False,
        prior_model_exposure_disclosed=True,current_stored_c4='YAW_0',
        historical_area=dict(status='AREA_RULE_INDEPENDENT_UNAVAILABLE' if original is None else 'ORIGIN_AVAILABLE_MIXED_2D',
            independent=False,reason='No original cuboid/keypoints_3d_world. Analytic cuboid + stored mixed manual/PnP coordinates used only as supplement.',
            supplemental_c4=area_c4,supplemental_perm=historical,
            AREA_DIFF=axis_diffs[0],TIE_MARGIN=axis_diffs[0]-axis_diffs[1],
            projection_recomputed=False,exact_converter_called=True),
        requires_human_review=True,reason='Multiple direct-order-compatible candidates; missing front vertices and mixed-provenance areas cannot uniquely establish near/front semantic role.')
    C.put(C.DOC/'GEOMETRY_ONLY_CANDIDATES.json',result)
    C.put(C.RAW/'GEOMETRY_LOCK.json',dict(geometry=C.bind(C.DOC/'GEOMETRY_ONLY_CANDIDATES.json'),
        annotation=C.bind(C.ANN),rgb=C.bind(C.RGB),model_payload_open_count_in_geometry=0))
    C.figure_table(C.FIG/'04_geometry_candidate_table.png','Direct evidence vs mixed-provenance area (NOT a semantic selector)',
        ['C4','Direct LR violations / n','Direct TB violations / n','Front direct vertices','Front/rear area px2'],
        [[r['c4'],f'{r["LR"]["violations"]}/{r["LR"]["available"]}',f'{r["TB"]["violations"]}/{r["TB"]["available"]}',r['direct_front_vertices'],f'{r["front_area_px2"]:.1f} / {r["rear_area_px2"]:.1f}'] for r in rows])
    print([(r['c4'],r['direct_order_compatible']) for r in rows]);print(result['historical_area'])

if __name__=='__main__':main()
