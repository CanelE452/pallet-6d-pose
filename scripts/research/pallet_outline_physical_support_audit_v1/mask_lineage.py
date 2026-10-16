"""Original saved-mask transforms and exact observed-sample parent mapping.

These are audit reconstructions of the old data transform, not replacement
observations. No RGB line detector, neural model, objective or pose solver runs.
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import time

import cv2
import numpy as np

from .bindings import load_cases,load_mask_arrays,read,sha,save,array_sha

LEGACY='scripts.research.pallet_visibility_curriculum_20261013_v1.'


def analyze_case(case,source_root):
    started,cpu=time.monotonic(),time.process_time()
    arrays=load_mask_arrays(case);stored=case['outline']
    typed=any(r.get('explicit_complete_BI_likelihood_revocation') is True for r in case['packet']['solver_input']['boundary_observations'])
    # D's caller privately rebinds G10._run's G module to v4.13 only when
    # typed factors exist; the no-typed branch reuses G10 exactly. Reconstruct
    # with that actual original observer, although the two transforms agree.
    observer_name='statewise_partial_BI_outline_solver_v4_13' if typed else 'statewise_global_outline_solver_v4_10'
    G=importlib.import_module(LEGACY+observer_name)
    M=importlib.import_module(LEGACY+'capture_boundary_recovery_masks_v4_6')
    clean=arrays['pallet_mask_clean'];raw=arrays['pallet_mask_raw']
    reconstructed=G.observe_global_outline(clean,stored['image_hw'],
        image_binding=stored['image_binding'],configuration=stored['configuration'])
    checks={}
    for field in ('polygon_xy','observed_boundary_segments_xy','observed_boundary_samples_xy'):
        checks[field+'_exact']=np.array_equal(reconstructed[field],stored[field])
    for field in ('image_hw','configuration','mask_array_sha256','mask_area_px','foreground_hull_area_px','supported'):
        checks[field+'_exact']=reconstructed[field]==stored[field]
    checks['compatibility_alias_is_clean_exact']=np.array_equal(arrays['pallet_mask'],clean)
    fresh=case['original_observation_frame'];full=None if fresh is None else fresh.get('full_mask_provenance')
    # Caller source uses defaults (.5 threshold, .01 component ratio). The
    # original provenance independently records the effective area ratio.
    cleanup_config=dict(M.DEFAULT_CONFIGURATION)
    if full is not None:
        cleanup_config['minimum_component_area_ratio']=full['component_cleanup']['component_area_ratio']
    reconstructed_clean,cleanup_audit=M.clean_boundary_components(raw,config=cleanup_config)
    checks['raw_to_clean_original_transform_exact']=np.array_equal(reconstructed_clean,clean)
    if full is not None:
        index=full['selected_mask_index'];quality=arrays['sam_candidate_quality']
        checks['selected_SAM_index_original_max_first_tie']=index==int(np.argmax(quality))
        checks['raw_mask_is_original_selected_candidate_exact']=np.array_equal(arrays['sam_candidate_masks'][index].astype(np.float32),raw)
        checks['uncalibrated_quality_archive_exact']=np.array_equal(np.asarray(full['uncalibrated_mask_quality'],quality.dtype),quality)
        checks['cleanup_audit_original_exact']=cleanup_audit==full['component_cleanup']
    if not all(checks.values()):raise ValueError('Original mask/outline transform parity failed: '+str(checks))
    polygon=np.asarray(stored['polygon_xy'],float);h,w=stored['image_hw'];margin=stored['configuration']['frame_margin_px']
    bounds=((margin,w-1-margin),(margin,h-1-margin));spacing=stored['configuration']['observed_sample_spacing_px']
    used=[];excluded=[];sample_parent=[];recreated=[]
    for edge_index,(a,b) in enumerate(zip(polygon,np.roll(polygon,-1,axis=0))):
        clipped=G._clip(a,b,bounds);parts=[]
        if clipped is None:
            excluded.append(dict(original_hull_edge_index=edge_index,xy=[a.tolist(),b.tolist()],reason='ORIGINAL_FRAME_MARGIN_CLIP'))
            continue
        first,last=clipped
        if not np.array_equal(a,first):parts.append([a.tolist(),first.tolist()])
        if not np.array_equal(last,b):parts.append([last.tolist(),b.tolist()])
        for part in parts:excluded.append(dict(original_hull_edge_index=edge_index,xy=part,reason='ORIGINAL_FRAME_MARGIN_CLIP'))
        length=float(np.linalg.norm(last-first));count=max(1,int(np.ceil(length/spacing)))
        positions=first[None,:]+((np.arange(count)+.5)/count)[:,None]*(last-first)
        offset=len(sample_parent);segment_id=len(used);indices=list(range(offset,offset+count))
        used.append(dict(segment_id=segment_id,original_hull_edge_index=edge_index,
            xy=[first.tolist(),last.tolist()],full_parent_xy=[a.tolist(),b.tolist()],excluded_parts_xy=parts,
            sample_indices=indices,observed_sample_count=count,length_px=length,
            segment_kind='ORIGINAL_CONVEX_HULL_FINITE_SEGMENT',
            constructed_geometry_does_not_establish_physical_error=True))
        sample_parent.extend([segment_id]*count);recreated.extend(positions.tolist())
    if not np.array_equal(recreated,stored['observed_boundary_samples_xy']):raise ValueError('Parent sample ordering differs')
    if not np.array_equal([s['xy'] for s in used],stored['observed_boundary_segments_xy']):raise ValueError('Stored finite segment order differs')
    def contours(mask):
        # Display-only contours of the original thresholded mask. All holes and
        # disconnected pieces are retained; no morphological operation runs.
        values=cv2.findContours((np.asarray(mask)>=.5).astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[0]
        return [c.reshape(-1,2).tolist() for c in values]
    result=dict(id=case['id'],ordinal=case['ordinal'],polygon_xy=stored['polygon_xy'],used_segments=used,
        excluded_frame_parts=excluded,sample_parent_segment_ids=sample_parent,
        observed_boundary_samples_xy=stored['observed_boundary_samples_xy'],
        raw_mask_contours_xy=contours(raw),clean_mask_contours_xy=contours(clean),
        checks={**checks,'parent_sample_order_exact':True,'finite_segment_order_exact':True},
        mask_lineage=dict(actual_outline_observer=observer_name+'.observe_global_outline',actual_outline_array='pallet_mask_clean',compatibility_alias='pallet_mask',
            SAM_selected_binary_array='pallet_mask_raw',soft_state_array='state_pallet_mask',tight_binary_array='pallet_mask_tight',
            SAM_candidates_array='sam_candidate_masks',SAM_quality_array='sam_candidate_quality',
            original_mask_threshold=cleanup_config['mask_threshold'],original_component_area_ratio=cleanup_config['minimum_component_area_ratio'],
            cleanup_reconstruction=cleanup_audit,full_capture_provenance=full,
            tight_capture_provenance=None if fresh is None else fresh.get('tight_mask_provenance'),
            SAM_decoder_logits='NOT_ARCHIVED_IN_THIS_NPZ; saved SAM masks are binary, not decoder soft probabilities',
            soft_state_mask_is_not_the_outline_mask=True,contours_display_only=True,mask_semantic_owner_verified=False),
        configuration=stored['configuration'],image_hw=stored['image_hw'],image_sha256=case['image_sha256'],
        mask_file_sha256=case['mask_file_sha256'],mask_array_float64_sha256=array_sha(np.asarray(clean,float)),
        source_sha256={str(Path(G.__file__).relative_to(source_root)):sha(G.__file__),str(Path(M.__file__).relative_to(source_root)):sha(M.__file__),
                       str(Path(__file__).relative_to(source_root)):sha(__file__)},
        new_mask_generations=0,new_RGB_line_detections=0,new_cost_evaluations=0,new_pose_calls=0,
        elapsed_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu)
    return result


def run(source_root,score_private,state_private,output):
    from .audit_guard import SupportAuditOnly
    started,cpu=time.monotonic(),time.process_time();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with SupportAuditOnly() as guard:
        cases=load_cases(source_root,score_private,state_private)
        results=[analyze_case(case,source_root) for case in cases]
    for result in results:save(output/(result['id'].replace(':','__')+'.json'),result)
    summary=dict(schema='ORIGINAL_SIX_MASK_LINEAGE_PARITY_V1',cases=len(results),all_mask_outline_checks_pass=all(all(r['checks'].values()) for r in results),
        used_parent_segments=sum(len(r['used_segments']) for r in results),observed_samples=sum(len(r['sample_parent_segment_ids']) for r in results),
        frames=[dict(id=r['id'],ordinal=r['ordinal'],used_parent_segments=len(r['used_segments']),excluded_frame_parts=len(r['excluded_frame_parts']),
                     observed_samples=len(r['sample_parent_segment_ids']),checks=r['checks'],path=r['id'].replace(':','__')+'.json') for r in results],
        runtime_guard=guard.receipt(),new_mask_generations=0,new_cost_evaluations=0,new_pose_calls=0,
        elapsed_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu)
    save(output/'MASK_LINEAGE_SUMMARY.json',summary);return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('source-root','score-private','state-private','output'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args();value=run(args.source_root,args.score_private,args.state_private,args.output)
    print({k:value[k] for k in ('cases','all_mask_outline_checks_pass','used_parent_segments','observed_samples','elapsed_seconds','process_cpu_seconds')})
