"""Source-test candidate coverage and mask support, without any ray/pose fit.

The cuboid edge facing test is only a proxy. A mask-supported, proxy-visible
physical correspondence labeled NONE is a diagnostic suspect, not proof that
its exact mesh depth label is wrong; this pass does not silently relabel it.
"""
from pathlib import Path
from collections import Counter
import time
import numpy as np
import cv2
from .source_ceiling import C,S,DOC,EDGES,query_geometry,stats


def run():
    begin=time.monotonic();families=S.selected_families();ceil=list(C.iter_rows(DOC/'SOURCE_CEILING_ROWS.jsonl.gz'));records=[]
    for source in ceil[896:1024]:
        family=families[source['index']];paths=S.locate(family);ann=C.read(paths['label']);g=S.geometry(family,ann)
        uv=np.asarray(ann['objects'][0]['projected_cuboid']);points=np.asarray(source['frozen_selected_points']);center,normal,queryvalid=query_geometry(points)
        visible=cv2.imread(str(paths['visible']),0);assert visible is not None and list(visible.shape)==g['hw'];h,w=g['hw'];camera=-g['R'].T@g['t']
        targets=source['targets'];valid=np.asarray(targets['valid']);lo=np.asarray(targets['lo']);roles=np.asarray(source['predicted_role_query_ids']);qq=[]
        for query in range(84):
            edge=query//7;a,b=EDGES[edge];tangent=uv[b]-uv[a];matrix=np.column_stack([normal[query],-tangent]);det=float(np.linalg.det(matrix))
            kind='IGNORE' if not valid[query] else 'POSITIVE' if lo[query]<65 else 'NONE';record=dict(query=query,edge=edge,target=kind,predicted_role=int(roles[query]),source_edge_physical=edge in source['physical_edge_ids'])
            if abs(det)<1e-8:record['intersection']='DEGENERATE';qq.append(record);continue
            offset,fraction=np.linalg.solve(matrix,uv[a]-center[query]);position=center[query]+offset*normal[query]
            in_search=abs(offset)<=32;in_segment=0<=fraction<=1;in_image=0<=position[0]<w and 0<=position[1]<h
            ix,iy=int(round(position[0]-.5)),int(round(position[1]-.5));l=max(0,ix-1);r=min(w,ix+2);t=max(0,iy-1);bmask=min(h,iy+2)
            support=bool(in_image and (visible[t:bmask,l:r]>127).any())
            fixed=np.isclose(g['X'][a],g['X'][b]);plane=g['X'][a]
            facing=np.sign(plane[fixed])*(camera[fixed]-plane[fixed]);proxy_visible=bool((facing>0).any())
            record.update(intersection='FINITE',offset_px=float(offset),fraction=float(fraction),position=position.tolist(),in_search=in_search,in_segment=in_segment,in_image=in_image,
                          supplied_visible_mask_support_3x3=support,cuboid_edge_face_proxy_visible=proxy_visible,
                          no_match_with_geometric_coverage_mask_support_and_proxy_visibility=bool(kind=='NONE' and in_search and in_segment and in_image and support and proxy_visible))
            qq.append(record)
        gt_inside=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h);pred_inside=(points[:8,0]>=0)&(points[:8,0]<w)&(points[:8,1]>=0)&(points[:8,1]<h)
        error=np.linalg.norm(points[:8]-uv,axis=1)
        records.append(dict(id=source['id'],index=source['index'],partition='source_test',raw_hw=g['hw'],source_ceiling_sha256=C.digest(source),
            source_annotation=C.binding(paths['label']),visible_mask=C.binding(paths['visible']),queries=qq,
            initial_corner_errors_all_px=error.tolist(),initial_corner_errors_GT_in_frame_px=error[gt_inside].tolist(),
            initial_corner_errors_both_in_frame_px=error[gt_inside&pred_inside].tolist(),GT_out_of_frame_corner_ids=np.flatnonzero(~gt_inside).tolist(),
            source_initial_inferred_pose_not_recomputed=True,raycast=None))
    q=[q for r in records for q in r['queries']];none=[r for r in q if r['target']=='NONE'];positive=[r for r in q if r['target']=='POSITIVE'];covered=lambda r:r.get('in_search') and r.get('in_segment') and r.get('in_image')
    suspects=[r for r in q if r.get('no_match_with_geometric_coverage_mask_support_and_proxy_visibility')]
    summary=dict(families=128,queries=len(q),targets=dict(Counter(r['target'] for r in q)),
        positive_abs_offsets_px=stats([abs(r['offset_px']) for r in positive]),
        none_out_of_search=int(sum(not r.get('in_search',False) for r in none)),none_out_of_segment=int(sum(not r.get('in_segment',False) for r in none)),
        none_with_geometric_search_segment_image_coverage=sum(bool(covered(r)) for r in none),
        none_covered_mask_supported=sum(bool(covered(r) and r['supplied_visible_mask_support_3x3']) for r in none),
        none_covered_mask_supported_proxy_visible=len(suspects),suspect_role_counts=dict(Counter(r['predicted_role'] for r in suspects)),suspect_edge_counts=dict(Counter(r['edge'] for r in suspects)),
        source_corner_error_all_px=stats([v for r in records for v in r['initial_corner_errors_all_px']]),
        source_corner_error_GT_in_frame_px=stats([v for r in records for v in r['initial_corner_errors_GT_in_frame_px']]),
        source_corner_error_both_in_frame_px=stats([v for r in records for v in r['initial_corner_errors_both_in_frame_px']]),
        out_of_frame_GT_corners=sum(len(r['GT_out_of_frame_corner_ids']) for r in records),
        suspect_examples=[dict(id=r['id'],queries=[q for q in r['queries'] if q.get('no_match_with_geometric_coverage_mask_support_and_proxy_visibility')][:3]) for r in records if any(q.get('no_match_with_geometric_coverage_mask_support_and_proxy_visibility') for q in r['queries'])][:5])
    C.save_rows(DOC/'SOURCE_NO_RAY_GEOMETRY_ROWS.jsonl.gz',records)
    result=dict(schema='source_existing_candidate_geometry_without_rays_v1',summary=summary,
        interpretation='Mask support and cuboid face visibility are necessary diagnostics, not a replacement for actual mesh depth. Suspect NONE labels stay unchanged.',
        original_target_generation_comment_does_not_match_operation='training.targets comments a tiny inset but casts unchanged exact physical-edge positions',
        new_detector_forwards=0,new_head_forwards=0,new_optimizer_updates=0,new_PnP_calls=0,new_auxiliary_rays=0,new_RGB=0,
        rows=C.binding(DOC/'SOURCE_NO_RAY_GEOMETRY_ROWS.jsonl.gz'),source_ceiling=C.binding(DOC/'SOURCE_CEILING_ROWS.jsonl.gz'),code=C.binding(Path(__file__)),wall_seconds=time.monotonic()-begin)
    C.write(DOC/'SOURCE_NO_RAY_GEOMETRY_AUDIT.json',result)
    import json
    print('SOURCE_NO_RAY_GEOMETRY_COMPLETE',json.dumps(C.finite(summary)),flush=True)


if __name__=='__main__':run()
